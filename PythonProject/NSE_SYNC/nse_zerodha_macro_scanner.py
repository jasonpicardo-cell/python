import os
import time as t
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
import threading

# Import your unified auth engine
from utils import get_zerodha_session, fetch_zerodha_nse_instruments

# ==========================================
# 1. SETUP PATHS & LIMITS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🛑 Exactly identical to Fyers paths to share the same database
CACHE_DIR = os.path.join(BASE_DIR, "../nse_data_cache")
BLACKLIST_FILE = os.path.join(BASE_DIR, "../blacklist.txt")
os.makedirs(CACHE_DIR, exist_ok=True)

# Zerodha allows 3 historical requests per second (180/min)
MAX_CONCURRENT_STOCKS = 2
bouncer = threading.Semaphore(MAX_CONCURRENT_STOCKS)
blacklist_lock = threading.Lock()
kite = None


# ==========================================
# 2. THE ZERODHA FETCHER (DELTA SYNC)
# ==========================================
def fetch_data_zerodha(tradingsymbol, instrument_token):
    # Make it smart: Check for both "Reliance.csv" and "RELIANCE.csv"
    title_case_path = os.path.join(CACHE_DIR, f"{tradingsymbol.capitalize()}.csv")
    upper_case_path = os.path.join(CACHE_DIR, f"{tradingsymbol.upper()}.csv")

    if os.path.exists(title_case_path):
        file_path = title_case_path
    elif os.path.exists(upper_case_path):
        file_path = upper_case_path
    else:
        # Default to Title Case if it's a brand new stock
        file_path = title_case_path

    today = datetime.now()

    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df_existing = pd.read_csv(file_path, index_col='Datetime', parse_dates=True).sort_index()

            if not df_existing.empty:
                last_date = df_existing.index[-1]
                start_date = last_date + timedelta(days=1)

                if start_date.date() > today.date():
                    return "ALREADY_SYNCED"
        except Exception as e:
            print(f"⚠️ Read error on {tradingsymbol}: {e}. Rebuilding...")
            start_date = today - timedelta(days=2000)
            df_existing = None
    else:
        start_date = today - timedelta(days=2000)
        df_existing = None

    # 🛑 THE CRUISE CONTROL: Throttle to respect Zerodha's 3 req/sec limit
    t.sleep(0.35)

    try:
        records = kite.historical_data(
            instrument_token=instrument_token,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=today.strftime('%Y-%m-%d'),
            interval="day",
            continuous=False
        )

        if not records:
            return "NO_NEW_CANDLES" if df_existing is not None else "NO_DATA"

        # Format Kite's output to match the Fyers CSV structure perfectly
        df_new = pd.DataFrame(records)
        df_new.rename(columns={
            'date': 'Datetime', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }, inplace=True)

        df_new['Datetime'] = pd.to_datetime(df_new['Datetime']).dt.tz_localize(None)
        df_new = df_new.set_index('Datetime')

        if 'oi' in df_new.columns:
            df_new.drop(columns=['oi'], inplace=True)

        # Append or Create
        if df_existing is not None and not df_existing.empty:
            df_combined = pd.concat([df_existing, df_new])
        else:
            df_combined = df_new

        df_combined = df_combined[~df_combined.index.duplicated(keep='last')].sort_index()
        df_combined.to_csv(file_path)

        return "APPEND_SUCCESS" if df_existing is not None else "SUCCESS"

    except Exception as e:
        err_str = str(e).lower()
        if "limit" in err_str or "429" in err_str:
            return "RATE_LIMIT"
        elif "permission" in err_str:
            return "NO_SUBSCRIPTION"
        return f"NETWORK_ERROR: {e}"


# ==========================================
# 3. THREAD WORKER
# ==========================================
def threaded_worker(stock_item):
    try:
        tradingsymbol, instrument_token = stock_item
        retries = 0

        with bouncer:
            while retries < 3:
                status = fetch_data_zerodha(tradingsymbol, instrument_token)

                if status == "SUCCESS":
                    print(f"✅ {tradingsymbol} (Full Sync).", flush=True)
                    break
                elif status == "APPEND_SUCCESS":
                    print(f"⚡ {tradingsymbol} (New Candles Appended).", flush=True)
                    break
                elif status == "ALREADY_SYNCED" or status == "NO_NEW_CANDLES":
                    print(f"👍 {tradingsymbol} (Up to date).", flush=True)
                    break
                elif status == "NO_DATA":
                    print(f"⏩ {tradingsymbol} - No trade data.", flush=True)
                    with blacklist_lock:
                        with open(BLACKLIST_FILE, "a") as f:
                            f.write(f"NSE:{tradingsymbol}-EQ\n")  # 🛑 Save Fyers format so Fyers skips it too!
                    break
                elif status == "RATE_LIMIT":
                    retries += 1
                    print(f"⏳ Rate-Limit on {tradingsymbol} (Attempt {retries}/3). Sleeping 60s...", flush=True)
                    t.sleep(60)
                elif status == "NO_SUBSCRIPTION":
                    print("🛑 FATAL: Historical API add-on subscription missing on Zerodha.", flush=True)
                    return
                else:
                    print(f"🛑 {tradingsymbol} failed -> {status}", flush=True)
                    break

            if retries >= 3:
                print(f"🛑 {tradingsymbol} abandoned after 3 rate-limit strikes.", flush=True)

    except Exception as e:
        print(f"🔥 FATAL THREAD CRASH on {stock_item[0]}: {e}", flush=True)


# ==========================================
# 4. NSE SYNC MANAGER
# ==========================================
# ==========================================
# 4. NSE SYNC MANAGER
# ==========================================
def run_nse_sync():
    instrument_map = fetch_zerodha_nse_instruments(kite)

    # 🛑 THE FIX: Tell the script to scan every single valid NSE Equity
    # instead of the hardcoded list of 5 stocks!
    nifty_universe = instrument_map.keys()

    to_process = []

    # 1. Load the shared blacklist
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, "r") as f:
            blacklist = set(line.strip().upper() for line in f if line.strip())

    for symbol in nifty_universe:
        sym_clean = symbol.strip().upper()

        # 2. Check if blacklisted
        if sym_clean in blacklist or f"NSE:{sym_clean}-EQ" in blacklist:
            print(f"🚫 Skipping Blacklisted: {sym_clean}")
            continue

        if sym_clean in instrument_map:
            to_process.append((sym_clean, instrument_map[sym_clean]))
        else:
            print(f"⚠️ {symbol} not found in Zerodha's active NSE instruments.")

    print(f"🏃 Remaining stocks to sync: {len(to_process)}")
    if not to_process:
        print("🎉 NSE Cache is completely up to date!")
        return

    print("🚀 Firing up Zerodha NSE Sync...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_STOCKS) as executor:
        executor.map(threaded_worker, to_process)


if __name__ == "__main__":
    print("🔐 Authenticating with Zerodha...")
    kite = get_zerodha_session()

    if kite:
        run_nse_sync()