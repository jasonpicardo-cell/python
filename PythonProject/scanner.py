import os
import webbrowser
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ==========================================
# 1. SETUP PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

if not os.path.exists(CACHE_DIR):
    print("❌ Cache directory not found! Run macro_scanner.py first.")
    exit()


# ==========================================
# 2. TRADING LOGIC
# ==========================================
def check_zone(df):
    """Checks for institutional Demand or Supply Zones."""
    if df is None or len(df) < 5:
        return ""

    # ⚠️ PASTE YOUR EXISTING DEMAND/SUPPLY LOGIC HERE ⚠️
    # Example placeholder:
    # if condition: return "DEMAND"
    # elif condition: return "SUPPLY"

    return ""


def check_high_low(df):
    """Checks for Multi-Year Breakouts OR Proximity (within 5%)."""
    if df is None or df.empty:
        return "-"

    curr_close = df['Close'].iloc[-1]
    last_date = df.index[-1]
    buffer_percent = 0.05

    # We check years descending so the oldest breakout triggers first
    periods = {"5-Year": 5, "4-Year": 4, "3-Year": 3, "2-Year": 2, "1-Year": 1}

    for label, years in periods.items():
        # Using relativedelta ensures leap years are calculated perfectly
        cutoff_date = last_date - relativedelta(years=years)

        # Slice DataFrame to the specific timeframe
        timeframe_df = df[df.index >= cutoff_date]

        if not timeframe_df.empty:
            period_high = timeframe_df['High'].max()
            period_low = timeframe_df['Low'].min()

            if curr_close >= period_high:
                return f"🚀 {label} Breakout"
            elif curr_close <= period_low:
                return f"🩸 {label} Breakdown"
            elif curr_close >= (period_high * (1 - buffer_percent)):
                return f"👀 Near {label} High"
            elif curr_close <= (period_low * (1 + buffer_percent)):
                return f"⚠️ Near {label} Low"

    return "-"


# ==========================================
# 3. HTML DASHBOARD GENERATOR
# ==========================================
def generate_html_popup(results):
    file_path = os.path.abspath("macro_dashboard.html")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Macro Structure Dashboard</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background-color: #1a1a1a; color: #e0e0e0; padding: 25px; }}
            h2 {{ color: #ffffff; text-align: center; border-bottom: 2px solid #333; padding-bottom: 15px; }}
            table {{ width: 100%; border-collapse: collapse; background: #252525; border-radius: 10px; overflow: hidden; margin-top: 20px; }}
            th {{ background-color: #333; color: #888; padding: 18px; text-align: left; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }}
            td {{ padding: 15px; border-bottom: 1px solid #333; }}
            tr:hover {{ background-color: #2d2d2d; }}
            .zone-demand {{ background-color: #1e4620; color: #4caf50; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
            .zone-supply {{ background-color: #4a1c1c; color: #f44336; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
            .zone-none {{ color: #666; }}
            .level-text {{ font-weight: 500; color: #bbdefb; }}
            .price-text {{ font-family: "SF Mono", "Monaco", monospace; color: #ffd54f; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>🏛️ MACRO STRUCTURE & INSTITUTIONAL ZONES</h2>
        <p style="text-align: center; color: #888;">Scan Date: {datetime.now().strftime('%d %b %Y | %H:%M')}</p>
        <table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Institutional Zone</th>
                    <th>Macro High/Low Status</th>
                    <th>Current Price</th>
                </tr>
            </thead>
            <tbody>
    """

    for res in results:
        stock = res.get('Stock', '-')
        zone = res.get('Zone', '-')
        hl_status = res.get('HighLow', '-')
        price = res.get('Price', 0.0)

        if "DEMAND" in zone.upper():
            zone_html = f'<span class="zone-demand">{zone}</span>'
        elif "SUPPLY" in zone.upper():
            zone_html = f'<span class="zone-supply">{zone}</span>'
        else:
            zone_html = f'<span class="zone-none">-</span>'

        html_content += f"""
                <tr>
                    <td><strong>{stock}</strong></td>
                    <td>{zone_html}</td>
                    <td class="level-text">{hl_status}</td>
                    <td class="price-text">₹{price:.2f}</td>
                </tr>
        """

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    webbrowser.open(f"file://{file_path}")


# ==========================================
# 4. MAIN EXECUTION (LOCAL ONLY)
# ==========================================
def run_scanner():
    print(f"📂 Scanning cached BSE stocks in: {CACHE_DIR}")

    screened_results = []

    # Iterate directly through the files in your cache
    for filename in os.listdir(CACHE_DIR):

        # 🛑 Only read CSVs. Safely ignores Mac hidden files and our new '.skip' files!
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(CACHE_DIR, filename)

        # 🛑 THE FIX: Ignore 0-byte files completely
        if os.path.getsize(filepath) == 0:
            continue
        # Reconstruct Stock Name from filename (e.g., "BSE_360ONE-A.csv" -> "360ONE")
        # Handles BSE format perfectly
        try:
            stock_name = filename.split('_')[1].split('-')[0]
        except IndexError:
            stock_name = filename.replace('.csv', '')  # Fallback

        try:
            # Read the pristine local data
            df = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)

            # Send to analysis functions
            zone = check_zone(df)
            hl_status = check_high_low(df)

            # If it triggered a signal, print it and add it to the dashboard array
            if zone or hl_status != "-":
                print(f"✅ MACRO HIT: {stock_name:<12} | Zone: {zone:<10} | Level: {hl_status}")

                screened_results.append({
                    "Stock": stock_name,
                    "Zone": zone if zone else "-",
                    "HighLow": hl_status,
                    "Price": df['Close'].iloc[-1]
                })

        except Exception as e:
            print(f"⚠️ Error analyzing {filename}: {e}")

    # Launch Dashboard
    if screened_results:
        print(f"\n🎯 SCAN COMPLETE. FOUND {len(screened_results)} OPPORTUNITIES.")
        generate_html_popup(screened_results)
    else:
        print("\n📉 Scan complete. No stocks met the criteria today.")


if __name__ == "__main__":
    run_scanner()