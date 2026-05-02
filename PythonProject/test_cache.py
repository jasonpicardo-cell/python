import os
from utils import fetch_live_fyers_symbols

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

print("🔍 Fetching symbols...")
all_stocks = fetch_live_fyers_symbols()
print(f"➡️ 1. Total stocks downloaded: {len(all_stocks)}")

if len(all_stocks) > 0:
    # Grab the very first stock in the list
    first_name = list(all_stocks.keys())[0]
    first_sym = all_stocks[first_name]

    file_path = os.path.join(CACHE_DIR, f"{first_sym.replace(':', '_')}.csv")

    print(f"➡️ 2. Checking specific file: {file_path}")
    print(f"➡️ 3. Does Python think this file exists? {os.path.exists(file_path)}")

    # Let's see what is ACTUALLY in that folder according to Python
    folder_contents = os.listdir(CACHE_DIR)
    print(f"➡️ 4. Total files Python sees in this folder right now: {len(folder_contents)}")
else:
    print("❌ ERROR: Your symbol list is completely empty! The issue is in utils.py.")