#!/usr/bin/env python3
"""
NSE Stocks — Price-Based DCF Analysis
======================================
Reads daily OHLCV CSVs from ../nse_data_cache, computes:
  • CAGRs (1Y / 3Y / 5Y)
  • Annualised volatility
  • 52-week range position
  • 14-day RSI
  • DCF intrinsic value  (earnings proxy + Gordon Growth terminal value)
  • Upside / downside vs current price

Outputs two sortable, filterable HTML reports:
  1. nifty750_dcf.html   – stocks in ../nifty750.txt
  2. other_stocks_dcf.html – remaining stocks
"""

import os, sys, glob, math, time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR          = "../nse_data_cache"
NIFTY750_FILE     = "../nifty750.txt"
OUTPUT_DIR        = "."          # HTML files written here

DISCOUNT_RATE     = 0.12         # Required rate of return (WACC proxy)
TERMINAL_GROWTH   = 0.07         # Long-term terminal growth (India GDP + inflation)
EARNINGS_YIELD    = 0.04         # Assumed earnings yield (=25× P/E — change as needed)
PROJECTION_YEARS  = 5            # DCF horizon

MIN_DATA_POINTS   = 60           # Skip stocks with too little history
MAX_GROWTH_CAP    = 0.55         # Cap unrealistic growth rates
MIN_GROWTH_FLOOR  = -0.25


# ─────────────────────────────────────────────
# ANALYSIS HELPERS
# ─────────────────────────────────────────────

def cagr(start_px, end_px, years: float):
    if not start_px or not end_px or start_px <= 0 or years <= 0:
        return None
    return (end_px / start_px) ** (1.0 / years) - 1.0


def dcf_intrinsic_value(current_price, growth_rate,
                        discount_rate=DISCOUNT_RATE,
                        terminal_growth=TERMINAL_GROWTH,
                        years=PROJECTION_YEARS,
                        earnings_yield=EARNINGS_YIELD):
    """
    Price-Based DCF
    ───────────────
    1. Base Earnings  = current_price × earnings_yield  (proxy for EPS / FCF per share)
    2. Project earnings for `years` at `growth_rate`
    3. Terminal value via Gordon Growth Model
    4. Discount all cash flows back at `discount_rate`

    This is an *approximation*. Real DCF requires actual financials.
    """
    if growth_rate is None or math.isnan(growth_rate):
        return None

    g = min(max(float(growth_rate), MIN_GROWTH_FLOOR), MAX_GROWTH_CAP)

    # Ensure terminal growth < discount rate
    tg = min(terminal_growth, discount_rate - 0.01)

    base_e = current_price * earnings_yield

    # PV of projected earnings
    pv = sum(
        base_e * (1 + g) ** yr / (1 + discount_rate) ** yr
        for yr in range(1, years + 1)
    )

    # Terminal value (perpetuity from year `years`)
    terminal_e  = base_e * (1 + g) ** years * (1 + tg)
    tv          = terminal_e / (discount_rate - tg)
    pv_terminal = tv / (1 + discount_rate) ** years

    return pv + pv_terminal


def rsi(series: pd.Series, period: int = 14) -> float | None:
    if len(series) < period + 5:
        return None
    delta   = series.diff()
    gain    = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss    = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs      = gain / loss.replace(0, np.nan)
    rsi_s   = 100 - 100 / (1 + rs)
    v       = rsi_s.iloc[-1]
    return float(v) if not math.isnan(v) else None


def analyze_stock(csv_path: str) -> dict | None:
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # Normalise datetime column (sometimes called 'Date')
        dt_col = next((c for c in df.columns if c.lower() in ("datetime", "date")), None)
        if dt_col is None:
            return None
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.rename(columns={dt_col: "Datetime"})
        df = df.sort_values("Datetime").dropna(subset=["Close", "Datetime"])
        df = df[df["Close"] > 0].reset_index(drop=True)

        if len(df) < MIN_DATA_POINTS:
            return None

        ticker       = Path(csv_path).stem.upper()
        today        = df["Datetime"].iloc[-1]
        current_px   = float(df["Close"].iloc[-1])

        def px_n_years_ago(n):
            target  = today - timedelta(days=int(n * 365.25))
            subset  = df[df["Datetime"] <= target]
            return float(subset["Close"].iloc[-1]) if len(subset) >= 5 else None

        c1  = cagr(px_n_years_ago(1),  current_px, 1)
        c3  = cagr(px_n_years_ago(3),  current_px, 3)
        c5  = cagr(px_n_years_ago(5),  current_px, 5)
        c10 = cagr(px_n_years_ago(10), current_px, 10)

        # Use best available CAGR for DCF (3Y preferred)
        dcf_growth = c3 if c3 is not None else (c5 if c5 is not None else c1)

        # Annualised volatility (last 252 trading days)
        rets  = df["Close"].pct_change().dropna()
        vol   = float(rets.tail(252).std() * math.sqrt(252)) if len(rets) >= 30 else None

        # 52-week range
        fy = df[df["Datetime"] >= today - timedelta(days=365)]
        h52 = float(fy["High"].max())  if len(fy) > 0 else float(df["High"].max())
        l52 = float(fy["Low"].min())   if len(fy) > 0 else float(df["Low"].min())
        r52 = h52 - l52
        pos52 = ((current_px - l52) / r52 * 100) if r52 > 0 else 50.0

        # Intrinsic value & upside
        iv     = dcf_intrinsic_value(current_px, dcf_growth)
        upside = ((iv / current_px) - 1) * 100 if iv else None

        # RSI
        rsi_val = rsi(df["Close"])

        # Avg daily volume (30-day)
        avg_vol = float(df["Volume"].tail(30).mean()) if "Volume" in df.columns else 0.0

        # All-time high
        ath = float(df["High"].max())

        # Data start date
        start_date = df["Datetime"].iloc[0].strftime("%Y-%m-%d")

        return dict(
            ticker        = ticker,
            current_price = current_px,
            last_date     = today.strftime("%Y-%m-%d"),
            start_date    = start_date,
            cagr_1y       = c1,
            cagr_3y       = c3,
            cagr_5y       = c5,
            cagr_10y      = c10,
            dcf_growth    = dcf_growth,
            volatility    = vol,
            high_52w      = h52,
            low_52w       = l52,
            pos_52w       = pos52,
            ath           = ath,
            intrinsic_val = iv,
            upside        = upside,
            rsi           = rsi_val,
            avg_volume    = avg_vol,
            data_points   = len(df),
        )
    except Exception as exc:
        print(f"  ✗ {Path(csv_path).stem}: {exc}")
        return None


# ─────────────────────────────────────────────
# HTML GENERATION
# ─────────────────────────────────────────────

def fmt_pct(v, decimals=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v * 100:+.{decimals}f}%"

def fmt_num(v, decimals=2, prefix="₹"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v >= 1_00_00_000:      # crores
        return f"{prefix}{v/1_00_00_000:.1f}Cr"
    if v >= 1_00_000:         # lakhs
        return f"{prefix}{v/1_00_000:.1f}L"
    return f"{prefix}{v:,.{decimals}f}"

def upside_class(up):
    if up is None or math.isnan(up):
        return "neutral"
    if up > 20:   return "strong-buy"
    if up > 5:    return "buy"
    if up > -5:   return "neutral"
    if up > -20:  return "sell"
    return "strong-sell"

def rsi_class(r):
    if r is None or math.isnan(r): return ""
    if r < 30: return "oversold"
    if r > 70: return "overbought"
    return ""

def cagr_class(c):
    if c is None or math.isnan(c): return ""
    if c > 0.20:  return "great"
    if c > 0.10:  return "good"
    if c > 0:     return "ok"
    return "bad"


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{
    --bg:        #0f1117;
    --surface:   #1a1d27;
    --surface2:  #222635;
    --border:    #2e3247;
    --text:      #e2e8f0;
    --muted:     #8892a4;
    --accent:    #6366f1;
    --green:     #10b981;
    --red:       #ef4444;
    --yellow:    #f59e0b;
    --blue:      #3b82f6;
    --orange:    #f97316;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html {{
    overflow-x: scroll;
    scrollbar-width: auto;
    scrollbar-color: #4b5563 #1a1d27;
  }}
  html::-webkit-scrollbar {{ height: 12px; }}
  html::-webkit-scrollbar-track {{ background: #1a1d27; }}
  html::-webkit-scrollbar-thumb {{ background: #4b5563; border-radius: 4px; }}
  html::-webkit-scrollbar-thumb:hover {{ background: #6b7280; }}
  body {{
    font-family: 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 13px;
    line-height: 1.5;
    min-width: max-content;
  }}

  /* ── HEADER ── */
  header {{
    background: linear-gradient(135deg, #1e2235 0%, #151823 100%);
    border-bottom: 1px solid var(--border);
    padding: 24px 32px 20px;
  }}
  header h1 {{
    font-size: 22px; font-weight: 700; color: #fff;
    display: flex; align-items: center; gap: 10px;
  }}
  header h1 span.badge {{
    background: var(--accent); color: #fff;
    font-size: 11px; padding: 2px 8px; border-radius: 20px;
    font-weight: 600; letter-spacing: .5px;
  }}
  header p {{ color: var(--muted); margin-top: 6px; font-size: 12px; }}
  .meta-row {{
    display: flex; flex-wrap: wrap; gap: 24px;
    margin-top: 14px;
  }}
  .meta-item {{ display: flex; flex-direction: column; }}
  .meta-item .label {{ font-size: 10px; text-transform: uppercase;
                       letter-spacing: .8px; color: var(--muted); }}
  .meta-item .value {{ font-size: 15px; font-weight: 600; color: #fff; margin-top: 2px; }}

  /* ── CONTROLS ── */
  .controls {{
    padding: 14px 32px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
  }}
  .controls input[type=text] {{
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--text); padding: 7px 12px; border-radius: 6px;
    font-size: 13px; width: 200px;
    outline: none; transition: border .2s;
  }}
  .controls input[type=text]:focus {{ border-color: var(--accent); }}
  .filter-btn {{
    background: var(--surface2); border: 1px solid var(--border);
    color: var(--muted); padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 12px; transition: all .2s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    border-color: var(--accent); color: var(--accent); background: #1e2040;
  }}
  .sort-hint {{ color: var(--muted); font-size: 11px; margin-left: auto; }}

  /* ── TABLE ── */
  .table-wrap {{
    overflow-x: scroll;
    padding: 0 32px 32px;
    scrollbar-width: auto;
    scrollbar-color: var(--accent) var(--surface2);
  }}
  .table-wrap::-webkit-scrollbar {{ height: 10px; }}
  .table-wrap::-webkit-scrollbar-track {{ background: var(--surface2); border-radius: 4px; }}
  .table-wrap::-webkit-scrollbar-thumb {{ background: var(--accent); border-radius: 4px; }}
  .table-wrap::-webkit-scrollbar-thumb:hover {{ background: #818cf8; }}
  table {{
    width: max-content; min-width: 100%;
    border-collapse: separate; border-spacing: 0;
    margin-top: 16px;
  }}
  thead th {{
    background: var(--surface);
    color: var(--muted);
    font-size: 10px; text-transform: uppercase; letter-spacing: .7px;
    padding: 10px 12px; white-space: nowrap;
    border-bottom: 2px solid var(--border);
    cursor: pointer; user-select: none;
    position: sticky; top: 0; z-index: 10;
    transition: color .2s;
  }}
  thead th:hover {{ color: var(--text); }}
  thead th.sorted-asc::after  {{ content: " ▲"; color: var(--accent); }}
  thead th.sorted-desc::after {{ content: " ▼"; color: var(--accent); }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: background .15s;
  }}
  tbody tr:hover {{ background: var(--surface2); }}
  tbody td {{
    padding: 9px 12px; white-space: nowrap;
    border-bottom: 1px solid #1e2132;
  }}

  .ticker-cell {{
    font-weight: 700; font-size: 13px; color: var(--accent);
    letter-spacing: .3px;
  }}
  .ticker-link {{
    color: var(--accent);
    text-decoration: none;
    border-bottom: 1px dashed transparent;
    transition: border-color .15s, color .15s;
  }}
  .ticker-link:hover {{
    color: #818cf8;
    border-bottom-color: #818cf8;
  }}
  .price-cell {{ font-weight: 600; }}

  /* Signal badges */
  .badge-strong-buy  {{ background: #064e3b; color: #34d399; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .badge-buy         {{ background: #14532d; color: #86efac; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .badge-neutral     {{ background: #374151; color: #9ca3af; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .badge-sell        {{ background: #7c2d12; color: #fdba74; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}
  .badge-strong-sell {{ background: #450a0a; color: #fca5a5; padding: 2px 9px; border-radius: 20px; font-size: 11px; font-weight: 600; }}

  /* Coloured numbers */
  .great {{ color: #34d399; }}
  .good  {{ color: #86efac; }}
  .ok    {{ color: #fde68a; }}
  .bad   {{ color: #f87171; }}
  .pos   {{ color: #34d399; }}
  .neg   {{ color: #f87171; }}

  /* RSI pill */
  .oversold   {{ color: #34d399; font-weight: 600; }}
  .overbought {{ color: #f87171; font-weight: 600; }}

  /* 52-W range bar */
  .range-wrap {{ display: flex; align-items: center; gap: 6px; min-width: 100px; }}
  .range-bar  {{ flex: 1; height: 4px; background: var(--border); border-radius: 2px; position: relative; }}
  .range-fill {{ height: 100%; background: var(--accent); border-radius: 2px; }}
  .range-pct  {{ font-size: 11px; color: var(--muted); width: 34px; text-align: right; }}

  /* Disclaimer */
  footer {{
    text-align: center;
    padding: 20px 32px;
    color: var(--muted);
    font-size: 11px;
    border-top: 1px solid var(--border);
    line-height: 1.8;
  }}

  /* Totals bar */
  .stats-bar {{
    display: flex; flex-wrap: wrap; gap: 0;
    padding: 0 32px;
    margin-top: 16px;
  }}
  .stat-box {{
    flex: 1; min-width: 120px;
    background: var(--surface); border: 1px solid var(--border);
    padding: 12px 16px;
  }}
  .stat-box:first-child {{ border-radius: 8px 0 0 8px; }}
  .stat-box:last-child  {{ border-radius: 0 8px 8px 0; }}
  .stat-box .s-label {{ font-size: 10px; text-transform: uppercase; letter-spacing: .7px; color: var(--muted); }}
  .stat-box .s-value {{ font-size: 18px; font-weight: 700; margin-top: 4px; }}

  .hidden {{ display: none !important; }}
</style>
</head>
<body>

<header>
  <h1>
    📊 {title}
    <span class="badge">DCF Analysis</span>
  </h1>
  <p>Price-based Discounted Cash Flow — Discount Rate {dr}% · Terminal Growth {tg}% · Earnings Yield {ey}% · {py}-Year Horizon</p>
  <div class="meta-row">
    <div class="meta-item"><span class="label">Generated</span><span class="value">{gen_date}</span></div>
    <div class="meta-item"><span class="label">Stocks Analysed</span><span class="value" id="totalCount">{count}</span></div>
    <div class="meta-item"><span class="label">Discount Rate</span><span class="value">{dr}%</span></div>
    <div class="meta-item"><span class="label">Terminal Growth</span><span class="value">{tg}%</span></div>
    <div class="meta-item"><span class="label">Earnings Yield</span><span class="value">{ey}%</span></div>
    <div class="meta-item"><span class="label">DCF Horizon</span><span class="value">{py} Years</span></div>
  </div>
</header>

<div class="stats-bar">
  <div class="stat-box">
    <div class="s-label">Strong Buy (>20% upside)</div>
    <div class="s-value great" id="cntSB">—</div>
  </div>
  <div class="stat-box">
    <div class="s-label">Buy (5–20% upside)</div>
    <div class="s-value good" id="cntB">—</div>
  </div>
  <div class="stat-box">
    <div class="s-label">Neutral (±5%)</div>
    <div class="s-value" id="cntN">—</div>
  </div>
  <div class="stat-box">
    <div class="s-label">Sell (5–20% overvalued)</div>
    <div class="s-value bad" id="cntS">—</div>
  </div>
  <div class="stat-box">
    <div class="s-label">Strong Sell (>20% overvalued)</div>
    <div class="s-value" style="color:#f87171" id="cntSS">—</div>
  </div>
</div>

<div class="controls">
  <input type="text" id="searchBox" placeholder="🔍  Search ticker…" oninput="filterTable()">
  <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
  <button class="filter-btn" onclick="setFilter('strong-buy',this)">Strong Buy</button>
  <button class="filter-btn" onclick="setFilter('buy',this)">Buy</button>
  <button class="filter-btn" onclick="setFilter('neutral',this)">Neutral</button>
  <button class="filter-btn" onclick="setFilter('sell',this)">Sell</button>
  <button class="filter-btn" onclick="setFilter('strong-sell',this)">Strong Sell</button>
  <span class="sort-hint">Click column headers to sort</span>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
  <th onclick="sortTable(0,'str')">Ticker</th>
  <th onclick="sortTable(1,'num')">Price (₹)</th>
  <th onclick="sortTable(2,'num')">Intrinsic Value</th>
  <th onclick="sortTable(3,'num')">Upside %</th>
  <th onclick="sortTable(4,'str')">Signal</th>
  <th onclick="sortTable(5,'num')">1Y CAGR</th>
  <th onclick="sortTable(6,'num')">3Y CAGR</th>
  <th onclick="sortTable(7,'num')">5Y CAGR</th>
  <th onclick="sortTable(8,'num')">10Y CAGR</th>
  <th onclick="sortTable(9,'num')">Volatility</th>
  <th onclick="sortTable(10,'num')">RSI (14)</th>
  <th onclick="sortTable(11,'num')">52W Range</th>
  <th onclick="sortTable(12,'num')">52W High</th>
  <th onclick="sortTable(13,'num')">52W Low</th>
  <th onclick="sortTable(14,'num')">ATH</th>
  <th onclick="sortTable(15,'num')">DCF Growth</th>
  <th onclick="sortTable(16,'str')">Last Date</th>
  <th onclick="sortTable(17,'num')">Avg Vol(30d)</th>
</tr>
</thead>
<tbody id="tableBody">
{rows}
</tbody>
</table>
</div>

<footer>
  ⚠️ <strong>DISCLAIMER:</strong> This analysis uses a price-based DCF proxy and is for <em>educational / research purposes only</em>.
  Intrinsic value estimates are derived from historical price CAGRs as an earnings growth proxy — not from actual company financials.<br>
  Assumed Earnings Yield: {ey}% (equivalent P/E ≈ {pe}×) · Discount Rate: {dr}% · Terminal Growth: {tg}%.
  <strong>Not investment advice. Always verify with fundamental analysis before making any investment decision.</strong>
</footer>

<script>
let currentFilter = 'all';
let sortCol = 3, sortDir = -1;  // default: sort by upside desc

function filterTable() {{
  const q    = document.getElementById('searchBox').value.toUpperCase().trim();
  const rows = document.querySelectorAll('#tableBody tr');
  rows.forEach(r => {{
    const ticker  = r.cells[0].textContent.toUpperCase();
    const signal  = r.dataset.signal || '';
    const matchQ  = !q || ticker.includes(q);
    const matchF  = currentFilter === 'all' || signal === currentFilter;
    r.classList.toggle('hidden', !(matchQ && matchF));
  }});
  updateVisibleCount();
}}

function setFilter(f, btn) {{
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterTable();
}}

function sortTable(col, type) {{
  const tbody = document.getElementById('tableBody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  if (sortCol === col) sortDir *= -1; else sortDir = -1;
  sortCol = col;

  rows.sort((a, b) => {{
    let av = a.cells[col].dataset.val ?? a.cells[col].textContent.trim();
    let bv = b.cells[col].dataset.val ?? b.cells[col].textContent.trim();
    if (type === 'num') {{
      av = parseFloat(av); bv = parseFloat(bv);
      if (isNaN(av)) av = -Infinity;
      if (isNaN(bv)) bv = -Infinity;
      return (av - bv) * sortDir;
    }}
    return av.localeCompare(bv) * sortDir;
  }});

  rows.forEach(r => tbody.appendChild(r));

  document.querySelectorAll('thead th').forEach((th, i) => {{
    th.classList.remove('sorted-asc','sorted-desc');
    if (i === sortCol) th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
  }});
}}

function updateVisibleCount() {{
  const visible = Array.from(document.querySelectorAll('#tableBody tr:not(.hidden)')).length;
  document.getElementById('totalCount').textContent = visible;
}}

function buildStats() {{
  const counts = {{}}; 
  ['strong-buy','buy','neutral','sell','strong-sell'].forEach(k => counts[k] = 0);
  document.querySelectorAll('#tableBody tr').forEach(r => {{
    const s = r.dataset.signal; if (s) counts[s] = (counts[s]||0) + 1;
  }});
  document.getElementById('cntSB').textContent = counts['strong-buy'];
  document.getElementById('cntB').textContent  = counts['buy'];
  document.getElementById('cntN').textContent  = counts['neutral'];
  document.getElementById('cntS').textContent  = counts['sell'];
  document.getElementById('cntSS').textContent = counts['strong-sell'];
}}

// Auto sort on load
window.addEventListener('DOMContentLoaded', () => {{
  sortTable(3, 'num');  // Sort by upside descending
  buildStats();
}});
</script>
</body>
</html>
"""


def build_row(s: dict) -> str:
    up   = s["upside"]
    sig  = upside_class(up)
    c3   = s["cagr_3y"]
    rsi_ = s["rsi"]

    # Upside display
    def fmt_up(v):
        if v is None or math.isnan(v): return "—", "0"
        cls = "pos" if v >= 0 else "neg"
        return f'<span class="{cls}">{v:+.1f}%</span>', f"{v:.2f}"

    # CAGR coloured
    def fmt_cagr(c):
        if c is None or math.isnan(c): return "—", "0"
        cc = cagr_class(c)
        return f'<span class="{cc}">{c*100:+.1f}%</span>', f"{c*100:.2f}"

    up_html,  up_val   = fmt_up(up)
    c1_html,  c1_val   = fmt_cagr(s["cagr_1y"])
    c3_html,  c3_val   = fmt_cagr(s["cagr_3y"])
    c5_html,  c5_val   = fmt_cagr(s["cagr_5y"])
    c10_html, c10_val  = fmt_cagr(s["cagr_10y"])
    cg_html,  cg_val   = fmt_cagr(s["dcf_growth"])

    # RSI
    rsi_v = s["rsi"]
    if rsi_v is None or math.isnan(rsi_v):
        rsi_html, rsi_data = "—", "0"
    else:
        rc = rsi_class(rsi_v)
        rsi_html = f'<span class="{rc}">{rsi_v:.1f}</span>'
        rsi_data = f"{rsi_v:.1f}"

    # Volatility
    vol = s["volatility"]
    vol_html = f"{vol*100:.1f}%" if vol else "—"
    vol_data = f"{vol*100:.2f}" if vol else "0"

    # 52W position bar
    pos = s["pos_52w"] or 0
    pos = min(max(pos, 0), 100)
    bar_html = (
        f'<div class="range-wrap">'
        f'<div class="range-bar"><div class="range-fill" style="width:{pos:.0f}%"></div></div>'
        f'<span class="range-pct">{pos:.0f}%</span>'
        f'</div>'
    )

    # Intrinsic value
    iv = s["intrinsic_val"]
    iv_str  = f"₹{iv:,.0f}" if iv else "—"
    iv_data = f"{iv:.2f}"   if iv else "0"

    # Volume
    avg_v = s["avg_volume"]
    if avg_v >= 1_00_000:
        vol_str = f"{avg_v/1_00_000:.1f}L"
    elif avg_v >= 1000:
        vol_str = f"{avg_v/1000:.1f}K"
    else:
        vol_str = f"{avg_v:.0f}" if avg_v else "—"

    signal_label = {
        "strong-buy":  "Strong Buy",
        "buy":         "Buy",
        "neutral":     "Neutral",
        "sell":        "Sell",
        "strong-sell": "Strong Sell",
    }.get(sig, "—")

    return (
        f'<tr data-signal="{sig}">'
        f'<td class="ticker-cell"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A{s["ticker"]}" '
        f'target="_blank" rel="noopener noreferrer" class="ticker-link">{s["ticker"]}</a></td>'
        f'<td class="price-cell" data-val="{s["current_price"]:.2f}">₹{s["current_price"]:,.2f}</td>'
        f'<td data-val="{iv_data}">{iv_str}</td>'
        f'<td data-val="{up_val}">{up_html}</td>'
        f'<td><span class="badge-{sig}">{signal_label}</span></td>'
        f'<td data-val="{c1_val}">{c1_html}</td>'
        f'<td data-val="{c3_val}">{c3_html}</td>'
        f'<td data-val="{c5_val}">{c5_html}</td>'
        f'<td data-val="{c10_val}">{c10_html}</td>'
        f'<td data-val="{vol_data}">{vol_html}</td>'
        f'<td data-val="{rsi_data}">{rsi_html}</td>'
        f'<td data-val="{pos:.1f}">{bar_html}</td>'
        f'<td data-val="{s["high_52w"]:.2f}">₹{s["high_52w"]:,.2f}</td>'
        f'<td data-val="{s["low_52w"]:.2f}">₹{s["low_52w"]:,.2f}</td>'
        f'<td data-val="{s["ath"]:.2f}">₹{s["ath"]:,.2f}</td>'
        f'<td data-val="{cg_val}">{cg_html}</td>'
        f'<td>{s["last_date"]}</td>'
        f'<td data-val="{avg_v:.0f}">{vol_str}</td>'
        f'</tr>'
    )


def generate_html(results: list[dict], title: str, out_path: str):
    rows_html = "\n".join(build_row(r) for r in results)
    html = HTML_TEMPLATE.format(
        title    = title,
        rows     = rows_html,
        count    = len(results),
        gen_date = datetime.now().strftime("%d %b %Y, %H:%M"),
        dr       = int(DISCOUNT_RATE * 100),
        tg       = int(TERMINAL_GROWTH * 100),
        ey       = int(EARNINGS_YIELD * 100),
        pe       = int(1 / EARNINGS_YIELD),
        py       = PROJECTION_YEARS,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ Written: {out_path}  ({len(results):,} stocks)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    t0 = time.time()

    # Collect all CSV paths
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(csv_files):,} CSV files in {DATA_DIR}")

    # Load Nifty750 list
    nifty750_tickers = set()
    if os.path.exists(NIFTY750_FILE):
        with open(NIFTY750_FILE) as f:
            for line in f:
                t = line.strip().upper()
                if t:
                    nifty750_tickers.add(t)
        print(f"Loaded {len(nifty750_tickers):,} tickers from {NIFTY750_FILE}")
    else:
        print(f"WARNING: {NIFTY750_FILE} not found — all stocks will go to 'other'")

    # Process all stocks
    nifty750_results = []
    other_results    = []
    errors           = 0

    for i, csv_path in enumerate(sorted(csv_files), 1):
        stem   = Path(csv_path).stem.upper()
        result = analyze_stock(csv_path)
        if result:
            if stem in nifty750_tickers:
                nifty750_results.append(result)
            else:
                other_results.append(result)
        else:
            errors += 1

        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i:,}/{len(csv_files):,}  "
                  f"({elapsed:.1f}s  ~{elapsed/i*len(csv_files):.0f}s total est.)")

    total = len(nifty750_results) + len(other_results)
    print(f"\nProcessed: {total:,} stocks  |  Errors/skipped: {errors}  |  "
          f"Nifty750: {len(nifty750_results)}  |  Other: {len(other_results)}")

    # Generate HTML files
    print("\nGenerating HTML reports…")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out1 = os.path.join(OUTPUT_DIR, "nifty750_dcf.html")
    out2 = os.path.join(OUTPUT_DIR, "other_stocks_dcf.html")

    generate_html(nifty750_results, "Nifty 750 — DCF Valuation",     out1)
    generate_html(other_results,    "Other NSE Stocks — DCF Valuation", out2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  → {out1}")
    print(f"  → {out2}")


if __name__ == "__main__":
    main()
