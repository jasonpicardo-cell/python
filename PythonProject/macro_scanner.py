import os
import time as t
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Import your working authentication
from utils import get_fyers_session, fetch_live_fyers_symbols

# 🛑 IMPORTANT NOTE 🛑
# RUN only after 4 pm or before 7 am
# ==========================================
# 1. SETUP PATHS & LIMITS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "bse_data_cache")
BLACKLIST_FILE = os.path.join(BASE_DIR, "blacklist.txt") # 🛑 ADD THIS
os.makedirs(CACHE_DIR, exist_ok=True)

# 🛑 Reduced to 2 to prevent hitting the 200/min Fyers limit too fast
MAX_CONCURRENT_STOCKS = 2
bouncer = threading.Semaphore(MAX_CONCURRENT_STOCKS)
blacklist_lock = threading.Lock() # 🛑 ADD THIS (Prevents threads from writing to the file at the exact same time)
fyers = None


# ==========================================
# 2. THE GREEDY FETCHER
# ==========================================
# ==========================================
# 2. THE GREEDY FETCHER
# ==========================================
# ==========================================
# 2. THE GREEDY FETCHER
# ==========================================
def fetch_data_macro(fyers_symbol):
    file_path = os.path.join(CACHE_DIR, f"{fyers_symbol.replace(':', '_')}.csv")
    today = datetime.now()

    # ==========================================
    # SCENARIO A: FILE EXISTS (DELTA SYNC)
    # ==========================================
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            # 1. Read existing data
            df_existing = pd.read_csv(file_path, index_col='Datetime', parse_dates=True)

            if not df_existing.empty:
                # 2. Find the last recorded date and add 1 day
                last_date = df_existing.index[-1]
                start_date = last_date + timedelta(days=1)

                # 3. Check if already up-to-date
                if start_date.date() > today.date():
                    return "ALREADY_SYNCED"

                # 4. Fetch only the missing days
                data = {
                    "symbol": fyers_symbol, "resolution": "1D", "date_format": "1",
                    "range_from": start_date.strftime('%Y-%m-%d'),
                    "range_to": today.strftime('%Y-%m-%d'), "cont_flag": "1"
                }

                # 🛑 THE CRUISE CONTROL: Naturally pace the script to ~180 calls/minute
                t.sleep(0.35)
                response = fyers.history(data=data)

                # 🛑 Catch API-level errors during append
                if response.get('s') == 'error':
                    msg = str(response.get('message', '')).lower()
                    if "limit" in msg or "429" in msg or "too many" in msg:
                        return "RATE_LIMIT"
                    return f"FYERS_REJECTED: {msg}"

                if response.get('s') == 'ok' and response.get('candles'):
                    # 5. Format new data
                    new_candles = response['candles']
                    df_new = pd.DataFrame(new_candles, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df_new['Datetime'] = pd.to_datetime(df_new['Datetime'], unit='s')
                    df_new = df_new.set_index('Datetime')

                    # 6. Append and clean duplicates
                    df_combined = pd.concat([df_existing, df_new])
                    df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()

                    # 7. Save back to CSV
                    df_combined.to_csv(file_path)
                    t.sleep(0.3)  # Respect API limits even on quick appends
                    return "APPEND_SUCCESS"
                else:
                    return "NO_NEW_CANDLES"

        except Exception as e:
            # Catch network errors during append, OR fall through if CSV is corrupted
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                return "RATE_LIMIT"
            print(f"⚠️ Read/Append error on {fyers_symbol}: {e}. Rebuilding full history...")
            # If we hit this, the file is corrupted. We fall through to SCENARIO B to rebuild it!

    # ==========================================
    # SCENARIO B: NO FILE EXISTS (FULL 5-YEAR SYNC)
    # ==========================================
    all_candles = []

    for i in range(5):
        chunk_end = today - timedelta(days=i * 360)
        chunk_start = chunk_end - timedelta(days=360)

        data = {
            "symbol": fyers_symbol, "resolution": "1D", "date_format": "1",
            "range_from": chunk_start.strftime('%Y-%m-%d'),
            "range_to": chunk_end.strftime('%Y-%m-%d'), "cont_flag": "1"
        }

        try:
            response = fyers.history(data=data)

            # 🛑 1. Catch API-level errors and read the exact message
            if response.get('s') == 'error':
                msg = str(response.get('message', '')).lower()
                if "limit" in msg or "429" in msg or "too many" in msg:
                    return "RATE_LIMIT"

                # If it fails on the very first chunk, tell us WHY
                if i == 0:
                    return f"FYERS_REJECTED: {msg}"
                break

            if response.get('s') == 'ok' and response.get('candles'):
                all_candles.extend(response['candles'])
                t.sleep(0.3)
            else:
                break

        except Exception as e:
            # 🛑 2. Catch Network/OS-level errors
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                return "RATE_LIMIT"
            return f"NETWORK_ERROR: {str(e)}"

    if all_candles:
        try:
            df = pd.DataFrame(all_candles, columns=['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
            df = df.set_index('Datetime').sort_index()
            df = df[~df.index.duplicated(keep='last')]
            df.to_csv(file_path)
            return "SUCCESS"
        except Exception:
            return "ERROR"

    return "NO_DATA"


# ==========================================
# 3. THE THREAD WORKER
# ==========================================
def threaded_worker(stock_item):
    stock_name, symbol = stock_item
    retries = 0

    with bouncer:
        while retries < 3:
            status = fetch_data_macro(symbol)
            print(f"DEBUG: {stock_name} returned -> [{status}]", flush=True)
            if status == "SUCCESS":
                print(f"✅ {stock_name} (Full 5-Year Sync).", flush=True)
                break
            elif status == "APPEND_SUCCESS":
                print(f"⚡ {stock_name} (New Candles Appended).", flush=True)
                break
            elif status == "ALREADY_SYNCED" or status == "NO_NEW_CANDLES":
                print(f"👍 {stock_name} (Up to date).", flush=True)
                break
            elif status == "NO_DATA":
                print(f"⏩ {stock_name} - No trade data found.", flush=True)
                break
            elif status == "RATE_LIMIT":
                retries += 1
                print(f"⏳ Rate-Limit hit on {stock_name} (Attempt {retries}/3). Flushing bucket for 60s...", flush=True)
                t.sleep(60)
            elif status.startswith("FYERS_REJECTED:") or status.startswith("NETWORK_ERROR:", flush=True):
                # 🛑 Print the exact hidden error to the terminal
                print(f"🛑 {stock_name} failed -> {status}")
                break
            else:
                print(f"❌ Error on {stock_name}. Skipping. (Status returned: {status})", flush=True)
                break

        if retries >= 3:
            print(f"🛑 {stock_name} abandoned after 3 rate-limit strikes.", flush=True)

# ==========================================
# 4. THE TURBO SYNC MANAGER
# ==========================================
# ==========================================
# 4. THE TURBO SYNC MANAGER
# ==========================================
# ==========================================
# 4. THE TURBO SYNC MANAGER
# ==========================================
def run_turbo_sync():
    all_stocks = fetch_live_fyers_symbols()
    to_process = []

    # 1. Load the blacklist into memory
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r") as f:
            blacklist = set(line.strip().upper() for line in f if line.strip())

    for name, sym in all_stocks.items():
        name_clean = str(name).strip().upper()
        sym_clean = str(sym).strip().upper()

        # 2. Skip dead stocks immediately
        if sym_clean in blacklist or name_clean in blacklist:
            print(f"🚫 Skipping Blacklisted: {name}")
            continue

            # 🛑 THE FIX: We removed the os.path check!
        # Send EVERY valid stock to the worker so the Delta Sync can do its job.
        to_process.append((name, sym))

    print(f"📊 Total BSE stocks loaded: {len(all_stocks)}")
    print(f"🏃 Remaining stocks to check/sync: {len(to_process)}")

    if not to_process:
        print("🎉 Cache is completely up to date! You can run scanner.py now.")
        return

    print("🚀 Firing up Turbo Sync (Throttled for Fyers Limits)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STOCKS) as executor:
        executor.map(threaded_worker, to_process)

    print("\n✅ Sync Process Complete.")


if __name__ == "__main__":
    print("🔐 Authenticating with Fyers...")
    fyers = get_fyers_session()

    if fyers:
        run_turbo_sync()