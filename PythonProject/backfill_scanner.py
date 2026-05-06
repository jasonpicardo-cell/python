import os
import time as t
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Import your working authentication
from utils import get_fyers_session, fetch_live_fyers_symbols
#------------
# Use this to fetch 5 yr data
#    target_start = datetime(2007, 5, 28)
#    target_end = datetime(2016, 5, 27)
#--------------
# ==========================================
# 1. SETUP PATHS & LIMITS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
BLACKLIST_FILE = os.path.join(BASE_DIR, "blacklist.txt")
os.makedirs(CACHE_DIR, exist_ok=True)

MAX_CONCURRENT_STOCKS = 2
bouncer = threading.Semaphore(MAX_CONCURRENT_STOCKS)
blacklist_lock = threading.Lock()
fyers = None


# ==========================================
# 2. THE HISTORICAL BACKFILLER
# ==========================================
def fetch_historical_backfill(fyers_symbol):
    file_path = os.path.join(CACHE_DIR, f"{fyers_symbol.replace(':', '_')}.csv")

    # 🛑 Define our exact target window
    target_start = datetime(2007, 5, 28)
    target_end = datetime(2016, 5, 27)

    df_existing = None

    # Check what data we already have
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path, index_col='Datetime', parse_dates=True).sort_index()

            # If the oldest date in the file is already near 2016, skip it!
            if not df_existing.empty and df_existing.index[0] <= target_start + timedelta(days=10):
                return "ALREADY_BACKFILLED"

            # If our existing data stretches into our target window (e.g., starts in 2020),
            # adjust the target_end so we don't fetch overlapping data unnecessarily.
            if not df_existing.empty and df_existing.index[0] < target_end:
                target_end = df_existing.index[0] - timedelta(days=1)

        except Exception as e:
            print(f"⚠️ Read error on {fyers_symbol}: {e}")

    # Now fetch the historical chunks moving backwards
    all_candles = []
    current_end = target_end

    # A 5-year gap takes about 6 loops of 360 days
    loops = 0
    while current_end >= target_start and loops < 6:
        loops += 1
        current_start = max(target_start, current_end - timedelta(days=360))

        data = {
            "symbol": fyers_symbol, "resolution": "1D", "date_format": "1",
            "range_from": current_start.strftime('%Y-%m-%d'),
            "range_to": current_end.strftime('%Y-%m-%d'), "cont_flag": "1"
        }

        t.sleep(0.35)  # Cruise Control for Fyers limit

        try:
            response = fyers.history(data=data)

            if response.get('s') == 'error':
                msg = str(response.get('message', '')).lower()
                if "limit" in msg or "429" in msg or "too many" in msg:
                    return "RATE_LIMIT"
                if loops == 1:
                    return f"FYERS_REJECTED: {msg}"
                break  # If it fails deep in history, just break and save what we got

            if response.get('s') == 'ok' and response.get('candles'):
                all_candles.extend(response['candles'])
            else:
                # No more data exists further back (Likely hit the stock's IPO date)
                break

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                return "RATE_LIMIT"
            return f"NETWORK_ERROR: {str(e)}"

        # Move the window backwards for the next chunk
        current_end = current_start - timedelta(days=1)

    # If we found no historical data (e.g., IPO was in late 2021)
    if not all_candles:
        return "NO_OLDER_DATA"

    try:
        # Format the newly found historical data
        df_history = pd.DataFrame(all_candles, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df_history['Datetime'] = pd.to_datetime(df_history['Datetime'], unit='s')
        df_history = df_history.set_index('Datetime')

        # Slap the old data on top of the existing data
        if df_existing is not None and not df_existing.empty:
            df_combined = pd.concat([df_history, df_existing])
        else:
            df_combined = df_history

        # Clean duplicates and sort flawlessly
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
        df_combined.to_csv(file_path)
        return "BACKFILL_SUCCESS"
    except Exception:
        return "ERROR"


# ==========================================
# 3. THE THREAD WORKER
# ==========================================
def threaded_worker(stock_item):
    try:
        stock_name, symbol = stock_item
        retries = 0

        with bouncer:
            while retries < 3:
                status = fetch_historical_backfill(symbol)
                status = str(status).strip()

                if status == "BACKFILL_SUCCESS":
                    print(f"✅ {stock_name} (Successfully backfilled 2016-2021 data).", flush=True)
                    break
                elif status == "ALREADY_BACKFILLED":
                    print(f"👍 {stock_name} (Already contains 2016 data).", flush=True)
                    break
                elif status == "NO_OLDER_DATA":
                    print(f"⏩ {stock_name} - No older data (Likely IPO'd after 2021).", flush=True)
                    break
                elif status == "RATE_LIMIT":
                    retries += 1
                    print(f"⏳ Rate-Limit hit on {stock_name} (Attempt {retries}/3). Flushing bucket for 60s...",
                          flush=True)
                    t.sleep(60)
                elif status.startswith("FYERS_REJECTED:") or status.startswith("NETWORK_ERROR:"):
                    print(f"🛑 {stock_name} failed -> {status}", flush=True)
                    break
                else:
                    print(f"❌ Error on {stock_name}. Skipping. (Status: {status})", flush=True)
                    break

            if retries >= 3:
                print(f"🛑 {stock_name} abandoned after 3 rate-limit strikes.", flush=True)

    except Exception as e:
        print(f"🔥 FATAL THREAD CRASH on {stock_item[0]}: {e}", flush=True)


# ==========================================
# 4. THE TURBO SYNC MANAGER
# ==========================================
def run_historical_sync():
    all_stocks = fetch_live_fyers_symbols()
    to_process = []

    # Apply the Blacklist
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r") as f:
            blacklist = set(line.strip().upper() for line in f if line.strip())

    for name, sym in all_stocks.items():
        name_clean = str(name).strip().upper()
        sym_clean = str(sym).strip().upper()

        if sym_clean in blacklist or name_clean in blacklist:
            continue

        to_process.append((name, sym))

    print(f"📊 Total BSE stocks loaded: {len(all_stocks)}")
    print(f"🏃 Remaining stocks to backfill: {len(to_process)}")

    if not to_process:
        return

    print("🚀 Firing up the 2016-2021 Time Machine...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STOCKS) as executor:
        executor.map(threaded_worker, to_process)

    print("\n✅ Deep History Backfill Complete!")


if __name__ == "__main__":
    print("🔐 Authenticating with Fyers...")
    fyers = get_fyers_session()

    if fyers:
        run_historical_sync()