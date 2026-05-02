import concurrent.futures
import os
import time
import webbrowser
from datetime import datetime, timedelta

import pandas as pd
from fyers_apiv3 import fyersModel


from utils import get_property, fetch_live_fyers_symbols

client_id = get_property("fyers_client_id")
try:
    with open("fyers_token.txt", "r") as f:
        access_token = f.read().strip()
except FileNotFoundError:
    print("❌ Token not found! Please run 'fyers_login.py' first.")
    exit()

fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")

def check_btst(df):
    """
    Evaluates momentum for Buy Today Sell Tomorrow (BTST).
    Requires at least 11 days of data to calculate volume averages.
    """
    # 1. Validation: Ensure we have enough data (today + 10 previous days)
    if df is None or len(df) < 11:
        return False

    # Get the last row (Today) and the second to last row (Yesterday)
    today = df.iloc[-1]
    yesterday = df.iloc[-2]

    # 2. PRICE MOMENTUM (The "Upward Trend" Filter)
    # The stock should be up at least 2.5% to 3% for the day
    price_change_pct = ((today['Close'] - yesterday['Close']) / yesterday['Close']) * 100
    is_bullish = price_change_pct >= 2.5

    # 3. CLOSING STRENGTH (The "No Sellers" Filter)
    # Stock should close near its daily high (upper wick should be very small)
    # Calculation: Where does the close sit within the High-Low range (0 to 1)
    daily_range = today['High'] - today['Low']
    if daily_range == 0: return False  # Skip stocks with no movement (circuit hit)

    relative_close = (today['Close'] - today['Low']) / daily_range
    is_closing_strong = relative_close >= 0.85  # Closed in the top 15% of the daily range

    # 4. VOLUME CONFIRMATION (The "Smart Money" Filter)
    # Today's volume should be significantly higher than the average of the last 10 days
    avg_volume_10d = df['Volume'].iloc[-11:-1].mean()
    is_volume_breakout = today['Volume'] > (avg_volume_10d * 1.5)  # 50% higher than average

    # --- FINAL LOGIC ---
    # A true BTST candidate must satisfy all three:
    # Price is up + Closing at Highs + Heavy Volume
    if is_bullish and is_closing_strong and is_volume_breakout:
        return True

    return False

def fetch_btst_data(fyers_symbol):
    """
    Fetches 15 days of 1D data in a single request.
    Optimized for speed and multi-threaded BTST scanning.
    """
    # 1. Calculate Date Range (Last 15 days is enough for BTST + 10d Vol Avg)
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d')

    # 2. Prepare the Request Payload
    data = {
        "symbol": fyers_symbol,
        "resolution": "1D",
        "date_format": "1",
        "range_from": from_date,
        "range_to": to_date,
        "cont_flag": "1"
    }

    try:
        # 3. Execute the API Call
        response = fyers.history(data=data)

        # 4. Handle Response
        if response.get('s') == 'ok' and response.get('candles'):
            # Convert the list of lists into a DataFrame
            df = pd.DataFrame(response['candles'], columns=[
                'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'
            ])

            # 5. Format and Clean Data
            # Convert Epoch timestamps to human-readable dates
            df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
            df.set_index('Datetime', inplace=True)

            # Ensure all price/volume columns are float/int for mathematical operations
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

            # Drop any rows that might have NaN values after conversion
            df.dropna(inplace=True)

            return df

        else:
            # Returns None if stock is illiquid, delisted, or symbol is wrong
            return None

    except Exception as e:
        # In multi-threading, we log the error but return None so the scanner continues
        # print(f"Error fetching {fyers_symbol}: {e}")
        return None

def run_btst_scanner():
    """Main execution engine for BTST scanning."""
    print(f"\n{'=' * 50}")
    print(f"🚀 BTST MOMENTUM SCANNER | Start Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'=' * 50}\n")

    # 1. Get the list of stocks (assuming this function is in your utils)
    all_stocks = fetch_live_fyers_symbols()
    stock_items = list(all_stocks.items())
    total_stocks = len(stock_items)

    results = []
    start_time = time.time()

    print(f"Checking {total_stocks} stocks for End-of-Day buying pressure...\n")

    # 2. Multi-threaded Execution
    # We use 10 workers because BTST data packets are very small.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10, thread_name_prefix="BTST-Worker") as executor:
        # Submit all tasks
        future_to_stock = {executor.submit(process_btst, item): item for item in stock_items}

        count = 0
        for future in concurrent.futures.as_completed(future_to_stock):
            count += 1

            # Silent progress heartbeat
            print(f"Progress: {count}/{total_stocks} stocks scanned...", end="\r", flush=True)

            try:
                res = future.result()
                if res:
                    # 'res' is the dictionary returned by process_btst
                    results.append(res)
            except Exception as e:
                # Catching errors here ensures one bad API response doesn't kill the whole scan
                pass

    # 3. Final Summary
    end_time = time.time()
    duration = round(end_time - start_time, 2)

    print(f"\n\n{'=' * 50}")
    print(f"✅ SCAN COMPLETE in {duration} seconds")
    print(f"📈 Total BTST Candidates Found: {len(results)}")
    print(f"{'=' * 50}\n")

    # 4. Actionable Output
    if results:
        # Sort results by the most promising (Volume_Change or Price Change)
        # Assuming your process_btst returns 'Volume_Change'
        results = sorted(results, key=lambda x: x.get('Volume_Change', 0), reverse=True)

        # Display in terminal for immediate action
        print(f"{'STOCK':<15} | {'PRICE':<10} | {'VOL SURGE':<10}")
        print("-" * 40)
        for r in results:
            print(f"{r['Stock']:<15} | ₹{r['Price']:<9.2f} | {r.get('Volume_Change', '-'):<10}x")

        # Launch the HTML dashboard for a cleaner view
        generate_html_popup(results)
    else:
        print("No stocks met the BTST momentum criteria today.")


def process_btst(stock_data):
    name, symbol = stock_data
    df = fetch_btst_data(symbol)  # Ensure this is also defined

    if df is not None and not df.empty:
        # Your check_btst logic here
        if check_btst(df):
            return {
                "Stock": name,
                "Price": df['Close'].iloc[-1],
                "Volume_Change": round(df['Volume'].iloc[-1] / df['Volume'].tail(10).mean(), 2)
            }
    return None


def generate_html_popup(results):
    """
    Generates a clean, BTST-focused dashboard and opens it in Safari.
    Expects results to contain: Stock, Price, Change_Pct, and Vol_Surge.
    """

    # Define the HTML file path
    file_path = os.path.abspath("btst_dashboard.html")

    # Start building the HTML String
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>BTST Momentum Dashboard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f7f6; padding: 20px; }}
            h2 {{ color: #2c3e50; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            th {{ background-color: #27ae60; color: white; padding: 15px; text-align: left; }}
            td {{ padding: 12px 15px; border-bottom: 1px solid #eee; }}
            tr:hover {{ background-color: #f1f1f1; }}
            .buy-tag {{ background-color: #27ae60; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
            .vol-high {{ color: #27ae60; font-weight: bold; }}
            .price {{ font-family: monospace; font-weight: bold; font-size: 1.1em; }}
        </style>
    </head>
    <body>
        <h2>🔥 BTST Momentum Candidates ({datetime.now().strftime('%d %b, %H:%M')})</h2>
        <table>
            <thead>
                <tr>
                    <th>Stock Name</th>
                    <th>Price (₹)</th>
                    <th>Day Change (%)</th>
                    <th>Volume Surge</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
    """

    # Add rows for each BTST candidate
    for res in results:
        # Extract values or set defaults to avoid KeyErrors
        stock = res.get('Stock', '-')
        price = res.get('Price', 0.0)
        change = res.get('Change_Pct', 0.0)
        vol_surge = res.get('Volume_Change', 0.0)

        html_content += f"""
                <tr>
                    <td><strong>{stock}</strong></td>
                    <td class="price">{price:.2f}</td>
                    <td style="color: {'green' if change > 0 else 'red'}">{change:+.2f}%</td>
                    <td class="vol-high">{vol_surge}x</td>
                    <td><span class="buy-tag">STRONG BUY</span></td>
                </tr>
        """

    # Closing tags
    html_content += """
            </tbody>
        </table>
        <p style="text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 20px;">
            Criteria: Price > 2.5%, Volume > 1.5x Avg, Closing in Top 15% of Day Range.
        </p>
    </body>
    </html>
    """

    # Write the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Open the file in the default browser (Safari on Mac)
    webbrowser.open(f"file://{file_path}")

if __name__ == "__main__":
    # Ensure you have your Fyers session initialized before calling this
    try:
        run_btst_scanner()
    except KeyboardInterrupt:
        print("\n\nScanner stopped by user.")