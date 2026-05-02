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
def check_institutional_zone(price_df, macro_lookback_periods=150, imbalance_lookback_periods=20):
    if price_df is None or len(price_df) < 50:
        return ""

    df = price_df.copy()

    # STEP 1: CONTEXT
    recent_macro_data = df.tail(macro_lookback_periods)
    macro_resistance_level = recent_macro_data['High'].max()
    macro_support_level = recent_macro_data['Low'].min()
    macro_price_range = macro_resistance_level - macro_support_level

    if macro_price_range == 0:
        return ""

    discount_threshold = macro_support_level + (macro_price_range * 0.30)
    premium_threshold = macro_resistance_level - (macro_price_range * 0.30)

    latest_close = df['Close'].iloc[-1]
    latest_high = df['High'].iloc[-1]
    latest_low = df['Low'].iloc[-1]

    is_in_discount_zone = latest_close <= discount_threshold
    is_in_premium_zone = latest_close >= premium_threshold

    if not is_in_discount_zone and not is_in_premium_zone:
        return ""

    # STEP 2: FAIR VALUE GAPS
    df['candle_1_high'] = df['High'].shift(2)
    df['candle_1_low'] = df['Low'].shift(2)

    df['is_bullish_fvg'] = df['Low'] > df['candle_1_high']
    df['bull_fvg_upper_edge'] = df['Low']
    df['bull_fvg_lower_edge'] = df['candle_1_high']

    df['is_bearish_fvg'] = df['High'] < df['candle_1_low']
    df['bear_fvg_upper_edge'] = df['candle_1_low']
    df['bear_fvg_lower_edge'] = df['High']

    recent_imbalances = df.tail(imbalance_lookback_periods)

    # STEP 3: MITIGATION CHECK
    if is_in_discount_zone:
        bullish_signals = recent_imbalances[recent_imbalances['is_bullish_fvg']]
        for index, row in bullish_signals.iterrows():
            subsequent_price_action = df.loc[index:].iloc[1:]
            is_mitigated = not subsequent_price_action.empty and (
                        subsequent_price_action['Close'] < row['bull_fvg_lower_edge']).any()

            if not is_mitigated:
                if latest_low <= row['bull_fvg_upper_edge'] and latest_close >= (row['bull_fvg_lower_edge'] * 0.99):
                    return "DEMAND"

    if is_in_premium_zone:
        bearish_signals = recent_imbalances[recent_imbalances['is_bearish_fvg']]
        for index, row in bearish_signals.iterrows():
            subsequent_price_action = df.loc[index:].iloc[1:]
            is_mitigated = not subsequent_price_action.empty and (
                        subsequent_price_action['Close'] > row['bear_fvg_upper_edge']).any()

            if not is_mitigated:
                if latest_high >= row['bear_fvg_lower_edge'] and latest_close <= (row['bear_fvg_upper_edge'] * 1.01):
                    return "SUPPLY"

    return ""


def check_high_low(df):
    if df is None or df.empty:
        return "-"

    curr_close = df['Close'].iloc[-1]
    last_date = df.index[-1]
    buffer_percent = 0.05

    periods = {"5-Year": 5, "4-Year": 4, "3-Year": 3, "2-Year": 2, "1-Year": 1}

    for label, years in periods.items():
        cutoff_date = last_date - relativedelta(years=years)
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
def format_zone_html(zone_text):
    if "DEMAND" in zone_text.upper():
        return f'<span class="zone-demand">{zone_text}</span>'
    elif "SUPPLY" in zone_text.upper():
        return f'<span class="zone-supply">{zone_text}</span>'
    return f'<span class="zone-none">-</span>'


def build_html_content(title, data_list):
    """Generates the full HTML string with built-in JS for a specific dataset."""

    # Extract unique Macro High/Low statuses for the dropdown filter dynamically
    unique_macros = set(r.get('HighLow', '-') for r in data_list)
    dropdown_options = '<option value="ALL">Show All Statuses</option>'
    for m in sorted(unique_macros):
        if m != "-":
            dropdown_options += f'<option value="{m}">{m}</option>'
    dropdown_options += '<option value="-">- (No Status)</option>'

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; background-color: #1a1a1a; color: #e0e0e0; padding: 25px; }}
            h2 {{ color: #ffffff; text-align: center; border-bottom: 2px solid #333; padding-bottom: 15px; margin-bottom: 5px; }}
            table {{ width: 100%; border-collapse: collapse; background: #252525; border-radius: 10px; overflow: hidden; margin-top: 20px; }}
            th {{ background-color: #333; color: #888; padding: 18px; text-align: left; text-transform: uppercase; font-size: 12px; letter-spacing: 1px; user-select: none; }}
            td {{ padding: 15px; border-bottom: 1px solid #333; }}
            tr:hover {{ background-color: #2d2d2d; }}

            .sortable {{ cursor: pointer; transition: background 0.2s; }}
            .sortable:hover {{ background-color: #444; color: #fff; }}

            select {{ margin-top: 8px; padding: 6px; background: #222; color: #bbdefb; border: 1px solid #555; border-radius: 4px; font-weight: bold; width: 100%; cursor: pointer; }}

            .zone-demand {{ background-color: #1e4620; color: #4caf50; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
            .zone-supply {{ background-color: #4a1c1c; color: #f44336; padding: 5px 10px; border-radius: 4px; font-weight: bold; font-size: 13px; }}
            .zone-none {{ color: #666; }}
            .level-text {{ font-weight: 500; color: #bbdefb; }}
            .price-text {{ font-family: "SF Mono", "Monaco", monospace; color: #ffd54f; font-weight: bold; }}
            .confluence {{ border-left: 4px solid #ffd54f; }} 
        </style>
    </head>
    <body>
        <h2>🏛️ {title.upper()} ({len(data_list)} Setups)</h2>
        <p style="text-align: center; color: #888;">Scan Date: {datetime.now().strftime('%d %b %Y | %H:%M')}</p>

        <table id="dataTable">
            <thead>
                <tr>
                    <th class="sortable" onclick="sortTable(0, false)">Symbol ↕️</th>
                    <th class="sortable" onclick="sortTable(1, false)">Weekly Zone ↕️</th>
                    <th class="sortable" onclick="sortTable(2, false)">Daily Zone ↕️</th>
                    <th>
                        Macro High/Low Status<br>
                        <select id="macroFilter" onchange="filterTable()">
                            {dropdown_options}
                        </select>
                    </th>
                    <th class="sortable" onclick="sortTable(4, true)">Current Price ↕️</th>
                </tr>
            </thead>
            <tbody>
    """

    for res in data_list:
        stock = res.get('Stock', '-')
        weekly = res.get('WeeklyZone', '')
        daily = res.get('DailyZone', '')
        hl_status = res.get('HighLow', '-')
        price = res.get('Price', 0.0)

        row_class = "confluence" if (weekly and daily) else ""
        html += f"""
                <tr class="{row_class}">
                    <td><strong>{stock}</strong></td>
                    <td data-val="{weekly}">{format_zone_html(weekly)}</td>
                    <td data-val="{daily}">{format_zone_html(daily)}</td>
                    <td class="level-text">{hl_status}</td>
                    <td class="price-text">₹{price:.2f}</td>
                </tr>
        """

    if not data_list:
        html += "<tr><td colspan='5' style='text-align:center; color:#666;'>No setups found in this category today.</td></tr>"

    # Insert Javascript for Sorting and Filtering
    html += """
            </tbody>
        </table>

        <script>
            let sortDirections = [true, true, true, true, true]; // Track asc/desc for each column

            function sortTable(columnIndex, isNumeric) {
                const table = document.getElementById("dataTable");
                const tbody = table.querySelector("tbody");
                const rows = Array.from(tbody.querySelectorAll("tr"));

                // Toggle direction
                const ascending = sortDirections[columnIndex];
                sortDirections[columnIndex] = !ascending;

                rows.sort((rowA, rowB) => {
                    let cellA = rowA.cells[columnIndex].innerText.trim();
                    let cellB = rowB.cells[columnIndex].innerText.trim();

                    // Fallback to data-val if reading zones so it doesn't sort by HTML spans
                    if(rowA.cells[columnIndex].getAttribute('data-val')) cellA = rowA.cells[columnIndex].getAttribute('data-val');
                    if(rowB.cells[columnIndex].getAttribute('data-val')) cellB = rowB.cells[columnIndex].getAttribute('data-val');

                    if (isNumeric) {
                        // Strip currency symbols and commas
                        cellA = parseFloat(cellA.replace(/[^0-9.-]+/g,"")) || 0;
                        cellB = parseFloat(cellB.replace(/[^0-9.-]+/g,"")) || 0;
                        return ascending ? cellA - cellB : cellB - cellA;
                    } else {
                        // String comparison
                        if (cellA < cellB) return ascending ? -1 : 1;
                        if (cellA > cellB) return ascending ? 1 : -1;
                        return 0;
                    }
                });

                // Re-append sorted rows to tbody
                rows.forEach(row => tbody.appendChild(row));
            }

            function filterTable() {
                const filterValue = document.getElementById("macroFilter").value;
                const tbody = document.getElementById("dataTable").querySelector("tbody");
                const rows = tbody.querySelectorAll("tr");

                rows.forEach(row => {
                    if (filterValue === "ALL") {
                        row.style.display = ""; // Show all
                    } else {
                        // Macro status is always in column index 3
                        const cellValue = row.cells[3].innerText.trim();
                        if (cellValue === filterValue) {
                            row.style.display = "";
                        } else {
                            row.style.display = "none";
                        }
                    }
                });
            }
        </script>
    </body>
    </html>
    """
    return html


def generate_html_reports(results, nifty_tickers):
    # 1. Split Data
    nifty_results = [r for r in results if r['Stock'] in nifty_tickers]
    other_results = [r for r in results if r['Stock'] not in nifty_tickers]

    # 2. File Paths
    nifty_path = os.path.abspath("nifty_dashboard.html")
    others_path = os.path.abspath("others_dashboard.html")

    # 3. Generate HTML Content
    nifty_html = build_html_content("Nifty 750 SMC Dashboard", nifty_results)
    others_html = build_html_content("Other Stocks SMC Dashboard", other_results)

    # 4. Write to Disk
    with open(nifty_path, "w", encoding="utf-8") as f:
        f.write(nifty_html)
    with open(others_path, "w", encoding="utf-8") as f:
        f.write(others_html)

    # 5. Open in Browser
    webbrowser.open(f"file://{nifty_path}")
    webbrowser.open(f"file://{others_path}")


# ==========================================
# 4. MAIN EXECUTION (LOCAL ONLY)
# ==========================================
def load_nifty_tickers(filepath="nifty750.txt"):
    if not os.path.exists(filepath):
        print(f"⚠️ Notice: '{filepath}' not found. All stocks will be grouped into 'Others'.")
        return set()

    with open(filepath, 'r') as f:
        return set(line.strip().upper() for line in f if line.strip())


def run_scanner():
    print(f"📂 Scanning cached BSE/NSE stocks in: {CACHE_DIR}")

    nifty_750_set = load_nifty_tickers(os.path.join(BASE_DIR, "nifty750.txt"))
    screened_results = []

    for filename in os.listdir(CACHE_DIR):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(CACHE_DIR, filename)

        if os.path.getsize(filepath) == 0:
            continue

        try:
            stock_name = filename.split('_')[1].split('-')[0]
        except IndexError:
            stock_name = filename.replace('.csv', '')

        try:
            df_daily = pd.read_csv(filepath, index_col='Datetime', parse_dates=True)

            if df_daily.empty or len(df_daily) < 50:
                continue

            df_weekly = df_daily.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
            }).dropna()

            daily_zone = check_institutional_zone(df_daily, macro_lookback_periods=150, imbalance_lookback_periods=20)
            weekly_zone = check_institutional_zone(df_weekly, macro_lookback_periods=52, imbalance_lookback_periods=8)
            hl_status = check_high_low(df_daily)

            if daily_zone or weekly_zone or hl_status != "-":
                print(
                    f"✅ HIT: {stock_name:<12} | W-Zone: {weekly_zone or '-':<6} | D-Zone: {daily_zone or '-':<6} | Lvl: {hl_status}")

                screened_results.append({
                    "Stock": stock_name,
                    "WeeklyZone": weekly_zone,
                    "DailyZone": daily_zone,
                    "HighLow": hl_status,
                    "Price": df_daily['Close'].iloc[-1]
                })

        except Exception as e:
            print(f"⚠️ Error analyzing {filename}: {e}")

    if screened_results:
        print(f"\n🎯 SCAN COMPLETE. FOUND {len(screened_results)} OPPORTUNITIES.")
        generate_html_reports(screened_results, nifty_750_set)
    else:
        print("\n📉 Scan complete. No stocks met the criteria today.")


if __name__ == "__main__":
    run_scanner()