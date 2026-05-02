import pandas as pd
import pandas_ta as ta
import schedule
import time
import webbrowser
import os
from datetime import datetime, timedelta
from dhanhq import DhanContext, dhanhq
from utils import get_property
import ssl
# ==========================================
# 1. API CONFIGURATION
# ==========================================
# Replace these with your actual Dhan credentials
client_id = get_property("dhan_client_id")
access_token = get_property("dhan_access_token")

try:
    # NEW SYNTAX: Create the context object first
    dhan_context = DhanContext(client_id, access_token)

    # Pass the context object to initialize the API
    dhan = dhanhq(dhan_context)
except Exception as e:
    print(f"Failed to connect to Dhan API. Check credentials. Error: {e}")


# ==========================================
# 2. DATA FETCHING (DHAN API)
# ==========================================
def fetch_data(security_id, timeframe="D"):
    """Fetches 1 year of historical data from Dhan and formats it for analysis."""
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=int(get_property('no_of_days')))).strftime('%Y-%m-%d')

    try:
        response = dhan.historical_daily_data(
            security_id=str(security_id),
            exchange_segment='NSE_EQ',
            instrument_type='EQUITY',
            expiry_code=0,
            from_date=from_date,
            to_date=to_date
        )

        if response.get('status') == 'success' and 'data' in response:
            data = response['data']

            # Handle Dhan's timestamp formatting
            time_key = 'timestamp' if 'timestamp' in data else 'start_Time'
            is_epoch = 'timestamp' in data

            df = pd.DataFrame({
                'Date': pd.to_datetime(data[time_key], unit='s' if is_epoch else None),
                'Open': data['open'],
                'High': data['high'],
                'Low': data['low'],
                'Close': data['close'],
                'Volume': data['volume']
            })

            df = df.sort_values('Date').reset_index(drop=True)

            # Resample for Weekly timeframe if requested
            if timeframe == 'W':
                df.set_index('Date', inplace=True)
                df = df.resample('W-FRI').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna().reset_index()

            return df
        else:
            print(f"API Rejected ID {security_id}. Raw response: {response}")
            return None
    except Exception as e:
        print(f"API Error fetching ID {security_id}: {e}")
        return None


# ==========================================
# 3. TECHNICAL ANALYSIS (DEMAND & SUPPLY)
# ==========================================
def check_zone(df):
    """Applies candlestick math to find institutional zones."""
    if df is None or len(df) < 21:
        return None  # Not enough data to calculate 20-SMA

    # Calculate Indicators
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Vol_SMA'] = ta.sma(df['Volume'], length=20)

    # Yesterday's variables (Index -2)
    prev_high, prev_low = df['High'].iloc[-2], df['Low'].iloc[-2]
    prev_open, prev_close = df['Open'].iloc[-2], df['Close'].iloc[-2]
    prev_atr = df['ATR'].iloc[-2]

    prev_range = prev_high - prev_low
    prev_body = abs(prev_close - prev_open)

    # Today's variables (Index -1)
    curr_high, curr_low = df['High'].iloc[-1], df['Low'].iloc[-1]
    curr_open, curr_close = df['Open'].iloc[-1], df['Close'].iloc[-1]
    curr_vol, curr_vol_sma = df['Volume'].iloc[-1], df['Vol_SMA'].iloc[-1]
    curr_atr = df['ATR'].iloc[-1]

    curr_range = curr_high - curr_low
    curr_body = abs(curr_close - curr_open)

    # Avoid division by zero on flat candles
    if prev_range == 0 or curr_range == 0:
        return None

    # Base Candle Condition
    is_base = (prev_range < prev_atr) and ((prev_body / prev_range) <= 0.5)

    # Bullish Demand Breakout
    is_demand_erc = (curr_close > curr_open) and ((curr_body / curr_range) >= 0.6) and (curr_range > curr_atr) and (
                curr_vol > curr_vol_sma)
    is_demand_breakout = curr_close > prev_high

    # Bearish Supply Breakout
    is_supply_erc = (curr_open > curr_close) and ((curr_body / curr_range) >= 0.6) and (curr_range > curr_atr) and (
                curr_vol > curr_vol_sma)
    is_supply_breakout = curr_close < prev_low

    if is_base and is_demand_erc and is_demand_breakout:
        return "🟢 DEMAND"
    elif is_base and is_supply_erc and is_supply_breakout:
        return "🔴 SUPPLY"

    return None


# ==========================================
# 4. HTML DASHBOARD GENERATOR
# ==========================================
def generate_html_popup(results):
    """Creates a local HTML file and opens it in the browser."""
    html_content = """
    <html>
    <head>
        <title>Scanner Results</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; background-color: #f4f4f9; }
            table { width: 100%; border-collapse: collapse; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #2c3e50; color: white; }
            .green { color: #27ae60; font-weight: bold; }
            .red { color: #c0392b; font-weight: bold; }
            .black { color: #333333; }
        </style>
    </head>
    <body>
        <h2>Institutional Zones & Macro Levels Detected</h2>
        <p>Last scanned at: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
        <table>
            <tr><th>Stock</th><th>Demand / Supply</th><th>Macro High / Low</th><th>Latest Price</th></tr>
    """

    for res in results:
        # Determine colors dynamically so blank "-" cells stay black
        zone_color = "green" if "DEMAND" in res['Zone'] else ("red" if "SUPPLY" in res['Zone'] else "black")
        hl_color = "green" if "High" in res['HighLow'] else ("red" if "Low" in res['HighLow'] else "black")

        html_content += "<tr>"
        html_content += f"<td><strong>{res['Stock']}</strong></td>"
        html_content += f"<td class='{zone_color}'>{res['Zone']}</td>"
        html_content += f"<td class='{hl_color}'>{res['HighLow']}</td>"
        html_content += f"<td>₹{res['Price']:.2f}</td>"
        html_content += "</tr>"

    html_content += "</table></body></html>"

    file_path = os.path.realpath("scanner_results.html")
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(html_content)

    import webbrowser
    webbrowser.open('file://' + file_path)


def fetch_live_dhan_ids():
    print("Downloading Dhan's master file in chunks (Takes ~10 seconds)...")

    # Using the standard Compact CSV
    url = get_property("dhan_stock_data_url")
    nse_dict = {}

    try:
        # Read the file in chunks of 100,000 rows
        for chunk in pd.read_csv(url, chunksize=100000, low_memory=False):

            # Clean column names (removes hidden spaces like "SEM_SEGMENT ")
            chunk.columns = chunk.columns.str.strip()

            if 'SEM_EXM_EXCH_ID' in chunk.columns:

                # Filter for NSE Equity
                # Segment = E (Equity), Series = EQ (Regular Equities)
                nse_chunk = chunk[
                    (chunk['SEM_EXM_EXCH_ID'] == 'NSE') &
                    (chunk['SEM_SEGMENT'] == 'E') &
                    (chunk['SEM_SERIES'] == 'EQ')
                    ]

                # Add to dictionary
                for _, row in nse_chunk.iterrows():
                    nse_dict[str(row['SEM_TRADING_SYMBOL'])] = str(row['SEM_SMST_SECURITY_ID'])

        print(f"Successfully extracted {len(nse_dict)} NSE stocks!")
        return nse_dict

    except Exception as e:
        print(f"❌ Failed to download master list: {e}")
        return {}

# --- To use it in your run_scanner() function: ---
# Replace your manual dictionary with this single line:
# all_stocks = fetch_live_dhan_ids()



# ==========================================
# 5. SCANNER ENGINE & SCHEDULER
# ==========================================
def run_scanner():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running market scan...")

    #Add your desired stocks and their Dhan Security IDs here
    all_stocks = {
        "RELIANCE": "2885",
        "TCS": "11536",
        "HDFCBANK": "1333",
        "INFY": "1594",
        "ICICIBANK": "4963"
    }

    #all_stocks = fetch_live_dhan_ids()
    print("----------------")
    print(len(all_stocks.items()))
    print(all_stocks.items())
    print("----------------")
    results = []
    timeframe = get_property("timeframe")
    print(f"✅ timeframe : {timeframe}")
    print(f"✅ no_of_days : {get_property('no_of_days')}")

    for stock_name, security_id in all_stocks.items():
        print(f"Scanning {stock_name}...")

        df = fetch_data(security_id, timeframe)  # Change to "W" for weekly

        if df is not None:
            print(f"[{stock_name}] Rows received: {len(df)}")
            zone = check_zone(df)
            hl_status = check_high_low(df)

            # Save the stock if it hits a Zone OR a High/Low
            if zone or hl_status != "-":
                results.append({
                    "Stock": stock_name,
                    "Zone": zone if zone else "-",
                    "HighLow": hl_status,
                    "Price": df['Close'].iloc[-1]
                })

        time.sleep(0.2) # API Rate Limit Protection

    if results:
        print(f"Found {len(results)} zones! Opening dashboard...")
        generate_html_popup(results)
    else:
        print("No valid zones found in this cycle.")


def check_high_low(df):
    """Checks if the current price is a 1, 2, 3, 4, or 5-year High/Low."""
    # We need at least 1 year of data (approx 252 trading days)
    if df is None or len(df) < 252:
        return "-"

    # Today's High and Low
    curr_high = df['High'].iloc[-1]
    curr_low = df['Low'].iloc[-1]

    # Trading days per year
    periods = {
        "5-Year": 1260,  # 252 * 5
        "4-Year": 1008,  # 252 * 4
        "3-Year": 756,  # 252 * 3
        "2-Year": 504,  # 252 * 2
        "1-Year": 252  # 252 * 1
    }

    # Check from 5-year down to 1-year
    for label, days in periods.items():
        # Ensure the stock has been listed long enough for this check
        if len(df) >= days:
            window_df = df.tail(days)

            # If today's high is the highest in the entire window
            if curr_high >= window_df['High'].max():
                return f"🚀 {label} High"

            # If today's low is the lowest in the entire window
            elif curr_low <= window_df['Low'].min():
                return f"🩸 {label} Low"

    return "-"  # Returns a dash if no major high/low is broken

# --- Start the Program ---
if __name__ == "__main__":
    print("Initializing Dhan Demand/Supply Scanner...")

    # Run it once immediately upon starting
    run_scanner()

    # Schedule it to run every 15 minutes thereafter
    schedule.every(15).minutes.do(run_scanner)

    print("Scanner is now active and running in the background. Press Ctrl+C to stop.")

    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)