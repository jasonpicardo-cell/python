import os
from datetime import datetime, timedelta

# Import your working authentication
from utils import get_fyers_session

print("🔐 Authenticating...")
fyers = get_fyers_session()

# We will test ACC using 5 different formats Fyers might use behind the scenes
formats_to_test = [
    "BSE:ACC-A",  # 1. Standard format
    "BSE:ACC",  # 2. Dropping the Group 'A' suffix
    "BSE_CM:ACC-A",  # 3. Capital Market prefix
    "BSE_CM:ACC",  # 4. CM prefix without suffix
    "BSE:ACC-EQ"  # 5. Using the NSE suffix on BSE (Fyers backend quirk)
]

end_date = datetime.now()
start_date = end_date - timedelta(days=10)

print("\n📡 Testing historical symbol formats for ACC...")

for sym in formats_to_test:
    data = {
        "symbol": sym,
        "resolution": "1D",
        "date_format": "1",
        "range_from": start_date.strftime('%Y-%m-%d'),
        "range_to": end_date.strftime('%Y-%m-%d'),
        "cont_flag": "1"
    }

    response = fyers.history(data=data)

    status = response.get('s')
    message = response.get('message', 'Got Data!')

    if status == "ok":
        print(f"✅ SUCCESS: [{sym}] works perfectly!")
    else:
        print(f"❌ FAILED:  [{sym}] -> {message}")