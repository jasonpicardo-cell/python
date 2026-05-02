import os
from fyers_apiv3 import fyersModel

# Internal cache and tracker variables
_properties_cache = {}
_last_modified_time = 0
_CONFIG_FILE = "config.properties"


def _load_properties():
    """Reads the file and updates the cache and timestamp."""
    global _last_modified_time

    if not os.path.exists(_CONFIG_FILE):
        print(f"Warning: {_CONFIG_FILE} not found.")
        return

    # 1. Record the exact time the file was last saved
    _last_modified_time = os.path.getmtime(_CONFIG_FILE)

    # 2. Clear the old data
    _properties_cache.clear()

    # 3. Read the fresh data
    with open(_CONFIG_FILE, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith(('#', '!')):
                if '=' in line:
                    key, value = line.split('=', 1)
                    _properties_cache[key.strip()] = value.strip()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_property(property_name):
    """Reads values from config.properties using an absolute path"""
    # Join the base directory with the filename
    config_path = os.path.join(BASE_DIR, "config.properties")

    try:
        with open(config_path, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == property_name:
                        return value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"❌ Error reading config at {config_path}: {e}")
    return None


# ==========================================
# 🛑 PASTE YOUR FYERS API CREDENTIALS HERE
# ==========================================
CLIENT_ID = get_property("fyers_client_id")  # Remember, v3 App IDs usually end in -100
SECRET_KEY = get_property("fyers_secret_key")
REDIRECT_URI = get_property("fyers_redirect_uri")  # Standard default
TOKEN_FILE = "fyers_token.txt"


def get_fyers_session():
    """
    Checks for an existing valid token. If none exists or it's expired,
    it guides the user through the login process to generate a new one.
    """
    # 1. Try to load an existing token
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as f:
            access_token = f.read().strip()
        if not CLIENT_ID or CLIENT_ID == "None":
            print("🚨 FATAL: client_id is None. Check your config.properties file name and key!")

        # Initialize session with the saved token
        fyers = fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access_token, log_path="")

        # Ping the API to see if the token is still alive (they expire daily)
        profile_response = fyers.get_profile()
        if profile_response.get('s') == 'ok':
            # Token is good! Return the session silently.
            return fyers
        else:
            print("⚠️ Token expired. Generating a new one...")

    # 2. If no token exists or it's expired, start the Login Flow
    session = fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )

    # Generate the login link
    auth_url = session.generate_authcode()

    print("\n" + "=" * 60)
    print("🔐 FYERS AUTHENTICATION REQUIRED")
    print("1. Click this link to log in:")
    print(f"\n{auth_url}\n")
    print("2. After logging in, you will be redirected to a new webpage.")
    print("3. Look at the URL of that page. It will look like this:")
    print("   https://trade.fyers.in/...&auth_code=YOUR_LONG_CODE_HERE&state=None")
    print("4. Copy ONLY the text for YOUR_LONG_CODE_HERE.")
    print("=" * 60 + "\n")

    # Wait for the user to paste the code
    auth_code = input("Paste your auth_code here: ").strip()

    # 3. Exchange the auth_code for an access_token
    session.set_token(auth_code)
    response = session.generate_token()

    if response.get('s') == 'ok':
        access_token = response['access_token']

        # Save the new token to the file so we don't have to do this again today
        with open(TOKEN_FILE, 'w') as f:
            f.write(access_token)

        print("\n✅ Authentication Successful! Token saved.")

        # Return the fully authorized Fyers object
        return fyersModel.FyersModel(client_id=CLIENT_ID, is_async=False, token=access_token, log_path="")
    else:
        print(f"\n❌ Authentication Failed: {response.get('message', response)}")
        # Exit the script entirely if auth fails, to prevent a cascade of errors
        exit()


# ---------------------------------------------------------
# You mentioned putting this in utils.py too, so here it is!
# ---------------------------------------------------------
def fetch_live_fyers_symbols():
    # 🛑 BSE CONFIGURATION 🛑
    url = "https://public.fyers.in/sym_details/BSE_CM.csv"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        print("⏳ Downloading BSE master list...")
        import requests
        import io
        import pandas as pd
        requests.packages.urllib3.disable_warnings()
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()

        # Grab StockName (Index 1) and Symbol (Index 9)
        df = pd.read_csv(io.StringIO(response.text), header=None, usecols=[1, 9], names=['StockName', 'Symbol'])

        df['Symbol'] = df['Symbol'].astype(str).str.strip()
        df['StockName'] = df['StockName'].astype(str).str.strip()

        # 🛑 THE FIX: Include ALL Equity Groups and ETFs.
        # Safely ignores -F (Fixed Income), -G (Govt Debt), and -INDEX
        allowed_suffixes = (
            '-A', '-B', '-T',
            '-X', '-XT', '-Z', '-ZP',
            '-M', '-MT', '-MS', '-P',
            '-B1', '-IF',
            '-E'  # ETFs
        )

        # Filter for all valid stock/ETF suffixes
        df_eq = df[df['Symbol'].str.endswith(allowed_suffixes)]

        print(f"✅ Filtered down to {len(df_eq)} tradable BSE Equities & ETFs.")
        return dict(zip(df_eq['StockName'], df_eq['Symbol']))

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}