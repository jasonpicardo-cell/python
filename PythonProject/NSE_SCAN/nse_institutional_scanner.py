"""
╔══════════════════════════════════════════════════════════════════╗
║        NSE INSTITUTIONAL FOOTPRINT SCANNER — Python v3          ║
║  Signals: Accumulation · Block Trade · Breakout · Absorption     ║
╚══════════════════════════════════════════════════════════════════╝

Setup (one-time):
    pip install yfinance pandas numpy requests tqdm tabulate

Usage:
    python nse_institutional_scanner.py                  # scan all NSE stocks (daily)
    python nse_institutional_scanner.py --tf 60          # 60-min intraday scan
    python nse_institutional_scanner.py --top 500        # limit to top 500 by mktcap
    python nse_institutional_scanner.py --min-strength 2 # only show strong signals
    python nse_institutional_scanner.py --watchlist my_stocks.txt
"""

import os
import sys
import time
import argparse
import datetime
import requests
import io
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Optional pretty imports (graceful fallback if missing) ────────
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    class tqdm:
        def __init__(self, iterable, **kwargs):
            self._it = iterable
            total = kwargs.get("total", "?")
            desc  = kwargs.get("desc", "")
            print(f"{desc} ({total} stocks) — no tqdm, logging every 50...")
        def __iter__(self):
            for i, v in enumerate(self._it):
                if i % 50 == 0:
                    print(f"  processed {i} stocks...")
                yield v
        def __enter__(self): return self
        def __exit__(self, *a): pass

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("ERROR: yfinance not installed. Run:  pip install yfinance")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — tune these to adjust signal sensitivity
# ═══════════════════════════════════════════════════════════════════
CONFIG = {
    # Volume moving average window
    "vol_ma_len":       20,

    # Accumulation: bullish candle, close in top 40%, vol > multiplier × avg
    "accum_vol_mult":   1.8,

    # Block trade: vol spike ≥ multiplier × avg (any direction)
    "block_vol_mult":   3.0,

    # Breakout: close above N-bar high/low, vol > multiplier × avg
    "bo_lookback":      20,
    "bo_vol_mult":      2.0,

    # Absorption/iceberg: candle body ≤ pct of price + vol > mult × avg,
    # repeated for N consecutive bars
    "abs_body_pct":     0.5,
    "abs_vol_mult":     2.5,
    "abs_bars":         3,

    # Download settings
    "period":           "3mo",   # how much history to fetch (enough for vol MA)
    "interval":         "1d",    # daily default; change to "60m" for intraday
    "batch_size":       50,      # stocks per yfinance batch call
    "delay_between_batches": 1,  # seconds, to avoid rate-limits

    # Output
    "output_csv":       "institutional_signals.csv",
    "output_html":      "institutional_signals.html",
    "min_price":        10,      # filter out penny stocks
    "min_avg_vol":      50000,   # minimum average daily volume
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Fetch full NSE equity list
# ═══════════════════════════════════════════════════════════════════

NSE_EQUITY_URL = (
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
)
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


def fetch_nse_stock_list(top_n: int = None) -> list[str]:
    """
    Download the full NSE equity list and return tickers as
    yfinance-compatible symbols (appended with .NS).
    Falls back to a curated Nifty-500 list if NSE blocks the request.
    """
    print("📡 Fetching NSE equity list...")
    try:
        session = requests.Session()
        # NSE requires a session cookie — warm it up first
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        r = session.get(NSE_EQUITY_URL, headers=NSE_HEADERS, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Column is usually 'SYMBOL'
        sym_col = [c for c in df.columns if "SYMBOL" in c.upper()][0]
        symbols = df[sym_col].str.strip().tolist()
        print(f"  ✅ Found {len(symbols)} NSE-listed stocks")
    except Exception as e:
        print(f"  ⚠️  NSE fetch failed ({e}). Using fallback Nifty-500 list...")
        symbols = _nifty500_fallback()

    # Convert to yfinance format
    yf_symbols = [f"{s}.NS" for s in symbols if isinstance(s, str) and s.strip()]

    if top_n:
        yf_symbols = yf_symbols[:top_n]
        print(f"  → Limited to top {top_n} symbols")

    return yf_symbols


def load_watchlist(path: str) -> list[str]:
    """Load a user-supplied watchlist file (one symbol per line)."""
    with open(path) as f:
        raw = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]
    return [s if s.endswith(".NS") else f"{s}.NS" for s in raw]


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Download OHLCV data in batches
# ═══════════════════════════════════════════════════════════════════

def download_batch(symbols: list[str], period: str, interval: str) -> dict:
    """
    Download OHLCV for a batch of symbols using yfinance.
    Returns {symbol: DataFrame} mapping.
    """
    try:
        raw = yf.download(
            tickers    = symbols,
            period     = period,
            interval   = interval,
            group_by   = "ticker",
            auto_adjust= True,
            progress   = False,
            threads    = True,
        )
    except Exception:
        return {}

    result = {}
    if len(symbols) == 1:
        sym = symbols[0]
        if not raw.empty:
            result[sym] = raw
        return result

    for sym in symbols:
        try:
            df = raw[sym].dropna()
            if len(df) >= CONFIG["vol_ma_len"] + 5:
                result[sym] = df
        except Exception:
            pass
    return result


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Signal detection logic
# ═══════════════════════════════════════════════════════════════════

def detect_signals(df: pd.DataFrame) -> dict | None:
    """
    Run all 4 institutional footprint signals on a stock's OHLCV DataFrame.
    Returns a result dict if ANY signal fires on the latest bar, else None.
    """
    cfg = CONFIG
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        return None

    # Need enough bars for the longest lookback
    min_bars = max(cfg["vol_ma_len"], cfg["bo_lookback"]) + cfg["abs_bars"] + 5
    if len(df) < min_bars:
        return None

    # Work on a clean copy
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    d.columns = ["open", "high", "low", "close", "volume"]

    # Basic filters
    last_price = float(d["close"].iloc[-1])
    avg_vol    = float(d["volume"].rolling(cfg["vol_ma_len"]).mean().iloc[-1])

    if last_price < cfg["min_price"]:
        return None
    if avg_vol < cfg["min_avg_vol"]:
        return None

    # Rolling averages
    d["avg_vol"]   = d["volume"].rolling(cfg["vol_ma_len"]).mean()
    d["vol_ratio"] = d["volume"] / d["avg_vol"]

    # ── Signal 1: Accumulation ────────────────────────────────────
    upper40     = d["low"] + (d["high"] - d["low"]) * 0.6
    is_accum    = (
        (d["volume"] > d["avg_vol"] * cfg["accum_vol_mult"]) &
        (d["close"]  > d["open"]) &
        (d["close"]  >= upper40)
    )

    # ── Signal 2: Block Trade ─────────────────────────────────────
    is_block    = d["volume"] >= d["avg_vol"] * cfg["block_vol_mult"]

    # ── Signal 3: Breakout ────────────────────────────────────────
    hi_n        = d["high"].shift(1).rolling(cfg["bo_lookback"]).max()
    lo_n        = d["low"].shift(1).rolling(cfg["bo_lookback"]).min()

    is_bo_bull  = (d["close"] > hi_n) & (d["volume"] > d["avg_vol"] * cfg["bo_vol_mult"])
    is_bo_bear  = (d["close"] < lo_n) & (d["volume"] > d["avg_vol"] * cfg["bo_vol_mult"])
    is_breakout = is_bo_bull | is_bo_bear

    # ── Signal 4: Absorption / Iceberg ───────────────────────────
    body_pct      = (d["close"] - d["open"]).abs() / d["close"] * 100
    is_tight      = (body_pct <= cfg["abs_body_pct"]) & (d["volume"] > d["avg_vol"] * cfg["abs_vol_mult"])

    # Count consecutive tight candles
    abs_count = is_tight.astype(int)
    # Running consecutive count using cumsum trick
    cum = (~is_tight).cumsum()
    consec = abs_count.groupby(cum).cumsum()
    is_absorption = consec >= cfg["abs_bars"]

    # ── Evaluate on LATEST bar ────────────────────────────────────
    idx = -1
    sig_accum     = bool(is_accum.iloc[idx])
    sig_block     = bool(is_block.iloc[idx])
    sig_breakout  = bool(is_breakout.iloc[idx])
    sig_absorb    = bool(is_absorption.iloc[idx])
    bo_dir        = "↑" if bool(is_bo_bull.iloc[idx]) else ("↓" if bool(is_bo_bear.iloc[idx]) else "")

    strength = sum([sig_accum, sig_block, sig_breakout, sig_absorb])

    if strength == 0:
        return None

    vol_ratio = float(d["vol_ratio"].iloc[idx])
    prev_close = float(d["close"].iloc[-2]) if len(d) > 1 else last_price
    chg_pct    = (last_price - prev_close) / prev_close * 100

    return {
        "price":       round(last_price, 2),
        "chg_pct":     round(chg_pct, 2),
        "volume":      int(d["volume"].iloc[idx]),
        "avg_volume":  int(avg_vol),
        "vol_ratio":   round(vol_ratio, 2),
        "strength":    strength,
        "accumulation":sig_accum,
        "block_trade": sig_block,
        "breakout":    sig_breakout,
        "bo_dir":      bo_dir,
        "absorption":  sig_absorb,
        "signals":     _signal_summary(sig_accum, sig_block, sig_breakout, bo_dir, sig_absorb),
    }


def _signal_summary(accum, block, bo, bo_dir, absorb) -> str:
    parts = []
    if accum:  parts.append("ACCUM")
    if block:  parts.append("BLOCK")
    if bo:     parts.append(f"BO{bo_dir}")
    if absorb: parts.append("ABSORB")
    return " + ".join(parts)


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — Main scan loop
# ═══════════════════════════════════════════════════════════════════

def run_scan(symbols: list[str]) -> pd.DataFrame:
    cfg    = CONFIG
    bs     = cfg["batch_size"]
    period = cfg["period"]
    itvl   = cfg["interval"]

    results = []
    batches = [symbols[i:i+bs] for i in range(0, len(symbols), bs)]

    print(f"\n🔍 Scanning {len(symbols)} stocks in {len(batches)} batches "
          f"(interval={itvl}, period={period})...\n")

    for batch in tqdm(batches, desc="Scanning batches", total=len(batches)):
        data = download_batch(batch, period, itvl)

        for sym, df in data.items():
            res = detect_signals(df)
            if res:
                ticker = sym.replace(".NS", "")
                results.append({"ticker": ticker, **res})

        time.sleep(cfg["delay_between_batches"])

    if not results:
        print("\n⚠️  No institutional signals detected in this scan.")
        return pd.DataFrame()

    df_out = pd.DataFrame(results)
    df_out.sort_values(["strength", "vol_ratio"], ascending=False, inplace=True)
    df_out.reset_index(drop=True, inplace=True)
    return df_out


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Output: terminal table + CSV + HTML report
# ═══════════════════════════════════════════════════════════════════

STRENGTH_STARS = {1: "★☆☆☆", 2: "★★☆☆", 3: "★★★☆", 4: "★★★★"}

def print_results(df: pd.DataFrame, min_strength: int = 1):
    df_show = df[df["strength"] >= min_strength].copy()

    if df_show.empty:
        print(f"\n⚠️  No signals with strength ≥ {min_strength}")
        return

    print(f"\n{'═'*72}")
    print(f"  🏦  INSTITUTIONAL FOOTPRINT SCAN RESULTS  —  "
          f"{datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
    print(f"  Showing {len(df_show)} stocks with signal strength ≥ {min_strength}")
    print(f"{'═'*72}\n")

    display_cols = ["ticker", "price", "chg_pct", "vol_ratio", "strength", "signals"]
    rename = {
        "ticker":    "Ticker",
        "price":     "Price ₹",
        "chg_pct":   "Chg %",
        "vol_ratio": "Vol Ratio",
        "strength":  "Strength",
        "signals":   "Signals Fired",
    }

    table_df = df_show[display_cols].rename(columns=rename).copy()
    table_df["Strength"] = table_df["Strength"].map(STRENGTH_STARS)
    table_df["Chg %"]    = table_df["Chg %"].apply(lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%")

    if HAS_TABULATE:
        print(tabulate(table_df, headers="keys", tablefmt="rounded_outline",
                       showindex=False, numalign="right"))
    else:
        print(table_df.to_string(index=False))


def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print(f"\n💾 Results saved → {path}")


def save_html(df: pd.DataFrame, path: str):
    scan_time = datetime.datetime.now().strftime("%d %b %Y %H:%M IST")
    total     = len(df)
    strong    = len(df[df["strength"] >= 3])

    signal_counts = {
        "Accumulation": int(df["accumulation"].sum()),
        "Block Trade":  int(df["block_trade"].sum()),
        "Breakout":     int(df["breakout"].sum()),
        "Absorption":   int(df["absorption"].sum()),
    }

    rows_html = ""
    for _, r in df.iterrows():
        strength_bar = "●" * r["strength"] + "○" * (4 - r["strength"])
        chg_cls  = "pos" if r["chg_pct"] > 0 else "neg"
        chg_str  = f"+{r['chg_pct']:.2f}%" if r["chg_pct"] > 0 else f"{r['chg_pct']:.2f}%"
        sig_tags = ""
        if r["accumulation"]: sig_tags += '<span class="tag accum">ACCUM</span>'
        if r["block_trade"]:  sig_tags += '<span class="tag block">BLOCK</span>'
        if r["breakout"]:     sig_tags += f'<span class="tag bo">BO{r["bo_dir"]}</span>'
        if r["absorption"]:   sig_tags += '<span class="tag absorb">ABSORB</span>'

        rows_html += f"""
        <tr>
          <td class="sym">{r['ticker']}</td>
          <td>₹{r['price']:,.2f}</td>
          <td class="{chg_cls}">{chg_str}</td>
          <td>{r['vol_ratio']:.1f}×</td>
          <td class="strength">{strength_bar}</td>
          <td>{sig_tags}</td>
        </tr>"""

    sc_html = "".join(
        f'<div class="stat-card"><div class="stat-label">{k}</div>'
        f'<div class="stat-num">{v}</div></div>'
        for k, v in signal_counts.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NSE Institutional Footprint Scanner</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f1117; color: #e2e8f0; padding: 24px; }}
  h1   {{ font-size: 22px; font-weight: 600; color: #f8fafc; margin-bottom: 4px; }}
  .sub {{ font-size: 13px; color: #64748b; margin-bottom: 24px; }}
  .stats {{ display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }}
  .stat-card {{ background: #1e2330; border: 1px solid #2d3548; border-radius: 10px;
                padding: 14px 20px; min-width: 130px; }}
  .stat-label {{ font-size: 11px; color: #64748b; text-transform: uppercase;
                 letter-spacing: .06em; margin-bottom: 6px; }}
  .stat-num   {{ font-size: 26px; font-weight: 600; color: #f8fafc; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: #1e2330; color: #94a3b8; font-weight: 500; padding: 10px 14px;
        text-align: left; border-bottom: 1px solid #2d3548; position: sticky; top: 0; }}
  td {{ padding: 9px 14px; border-bottom: 1px solid #1a1f2e; }}
  tr:hover td {{ background: #1a2035; }}
  .sym  {{ font-weight: 600; color: #60a5fa; font-size: 14px; }}
  .pos  {{ color: #4ade80; }}
  .neg  {{ color: #f87171; }}
  .strength {{ font-size: 15px; letter-spacing: 2px; color: #f59e0b; }}
  .tag  {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 11px; font-weight: 600; margin-right: 4px; }}
  .accum  {{ background: #14532d; color: #4ade80; }}
  .block  {{ background: #7c2d12; color: #fb923c; }}
  .bo     {{ background: #1e3a5f; color: #60a5fa; }}
  .absorb {{ background: #3b0764; color: #c084fc; }}
</style>
</head>
<body>
  <h1>🏦 NSE Institutional Footprint Scanner</h1>
  <p class="sub">Scan time: {scan_time} &nbsp;|&nbsp; {total} signals found &nbsp;|&nbsp;
     {strong} high-conviction (★★★+)</p>
  <div class="stats">
    <div class="stat-card"><div class="stat-label">Total Signals</div>
      <div class="stat-num">{total}</div></div>
    <div class="stat-card"><div class="stat-label">High Conviction</div>
      <div class="stat-num">{strong}</div></div>
    {sc_html}
  </div>
  <table>
    <thead>
      <tr>
        <th>Ticker</th><th>Price</th><th>Change</th>
        <th>Vol Ratio</th><th>Strength</th><th>Signals</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
    print(f"🌐 HTML report saved → {path}")


# ═══════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="NSE Institutional Footprint Scanner")
    p.add_argument("--tf",           default="1d",  help="Timeframe: 1d | 60m | 15m | 5m")
    p.add_argument("--top",          type=int,      help="Limit to top N stocks")
    p.add_argument("--min-strength", type=int, default=1,
                   help="Minimum signal strength to display (1–4)")
    p.add_argument("--watchlist",    type=str,      help="Path to watchlist .txt file")
    p.add_argument("--period",       default="3mo", help="Historical period (3mo | 6mo | 1y)")
    p.add_argument("--no-html",      action="store_true", help="Skip HTML report generation")
    p.add_argument("--block-mult",   type=float,    help="Override block trade multiplier")
    p.add_argument("--accum-mult",   type=float,    help="Override accumulation multiplier")
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides to config
    CONFIG["interval"] = args.tf
    CONFIG["period"]   = args.period
    if args.block_mult: CONFIG["block_vol_mult"]  = args.block_mult
    if args.accum_mult: CONFIG["accum_vol_mult"]  = args.accum_mult

    # Adjust period for intraday
    if args.tf in ("5m", "15m", "30m", "60m"):
        CONFIG["period"] = "5d"
        print(f"ℹ️  Intraday mode ({args.tf}) — period forced to 5d")

    # Get symbols
    if args.watchlist:
        symbols = load_watchlist(args.watchlist)
        print(f"📋 Loaded {len(symbols)} symbols from {args.watchlist}")
    else:
        symbols = fetch_nse_stock_list(top_n=args.top)

    if not symbols:
        print("❌ No symbols to scan.")
        sys.exit(1)

    # Run scan
    start = time.time()
    df    = run_scan(symbols)
    elapsed = time.time() - start

    print(f"\n⏱️  Scan complete in {elapsed:.1f}s")

    if df.empty:
        sys.exit(0)

    # Filter & print
    print_results(df, min_strength=args.min_strength)

    # Save outputs
    save_csv(df, CONFIG["output_csv"])
    if not args.no_html:
        save_html(df, CONFIG["output_html"])

    # Summary
    print(f"\n{'─'*50}")
    print(f"  📊 Total signals found : {len(df)}")
    print(f"  🔥 High conviction (3+): {len(df[df['strength'] >= 3])}")
    print(f"  ✅ ACCUM  : {int(df['accumulation'].sum())}")
    print(f"  🟠 BLOCK  : {int(df['block_trade'].sum())}")
    print(f"  🔵 BO     : {int(df['breakout'].sum())}")
    print(f"  🟣 ABSORB : {int(df['absorption'].sum())}")
    print(f"{'─'*50}\n")


# ── Fallback Nifty-500 list ───────────────────────────────────────
def _nifty500_fallback() -> list[str]:
    """Returns a hardcoded list of Nifty-500 symbols as fallback."""
    return [
        "RELIANCE","TCS","HDFCBANK","INFY","HINDUNILVR","ICICIBANK","KOTAKBANK",
        "SBIN","BHARTIARTL","ASIANPAINT","ITC","LT","AXISBANK","DMART","SUNPHARMA",
        "TITAN","ULTRACEMCO","WIPRO","NESTLEIND","HCLTECH","MARUTI","M&M",
        "BAJFINANCE","TECHM","POWERGRID","NTPC","ONGC","JSWSTEEL","TATASTEEL",
        "COALINDIA","ADANIENT","ADANIPORTS","GRASIM","HDFCLIFE","BAJAJFINSV",
        "DIVISLAB","DRREDDY","CIPLA","EICHERMOT","HEROMOTOCO","BPCL","IOC",
        "TATAMOTORS","BRITANNIA","TATACONSUM","SBICARD","PIDILITIND","AMBUJACEM",
        "SHREECEM","INDUSINDBK","HINDALCO","VEDL","DLF","GODREJCP","MUTHOOTFIN",
        "HAVELLS","BERGEPAINT","COLPAL","MARICO","DABUR","MOTHERSON","LTIM",
        "MPHASIS","PERSISTENT","COFORGE","LTF","CHOLAFIN","BAJAJ-AUTO","BOSCHLTD",
        "SIEMENS","ABB","DIXON","TRENT","PAGEIND","POLYCAB","ZOMATO","NYKAA",
        "PAYTM","IRCTC","RVNL","IRFC","BEL","HAL","BHEL","NMDC","SAIL",
        "RECLTD","PFC","CANBK","BANKBARODA","PNB","UNIONBANK","FEDERALBNK",
        "IDFCFIRSTB","BANDHANBNK","RBLBANK","AUBANK","KARURVYSYA","CITYUNIONBANK",
        "EQUITASBNK","UJJIVANSFB","ESAFSFB","SURYODAY","FINPIPE","GRINDWELL",
        "ASTRAL","SUPREMEIND","PRINCEPIPE","FINOLEX","KPITTECH","TATAELXSI",
        "HEXAWARE","NIITTECH","ZENSAR","MASTEK","BIRLASOFT","SONATSOFTW",
        "INGERRAND","CUMMINSIND","THERMAX","KOEL","BHARAT_FORGE","MAHINDCIE",
        "TIINDIA","SUNDARMFIN","SHRIRAMFIN","MANAPPURAM","IIFLWAM","ANGELONE",
        "ICICIPRULI","HDFCAMC","MFSL","NIACL","GICRE","STARHEALTH",
        "MAXHEALTH","FORTIS","APOLLOHOSP","NARAYANHC","METROPOLIS","THYROCARE",
        "DRLAL","VIJAYA","KRSNAA","POLYMED","LONZA",
    ]


if __name__ == "__main__":
    main()
