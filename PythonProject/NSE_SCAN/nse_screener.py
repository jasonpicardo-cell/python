"""
NSE Stock Screener  v1
========================
Combines three complementary analyses into a unified score per stock:

  1. TREND TEMPLATE  — Minervini's 9-condition Stage-2 checklist
  2. RELATIVE STRENGTH — Price performance vs benchmark over 1M/3M/6M/1Y
  3. VCP (Volume Contraction Pattern) — Volume dry-up near highs

All computed from the same daily OHLCV CSVs used by the volume profile scanner.

Hardcoded paths
---------------
    Data dir   : ../nse_data_cache      (one CSV per stock)
    Nifty750   : ../nifty750.txt
    Benchmark  : ../nse_data_cache/NIFTY50.csv  (falls back to universe median)
    Output     : current directory

Run
---
    python nse_screener.py                        # default 1250-day lookback
    python nse_screener.py --lookback-days 500
    python nse_screener.py --lookback-days 0      # full history
    python nse_screener.py --csv                  # also save CSV results

Outputs
-------
    nifty750_screener.html
    others_screener.html
"""

import argparse, warnings
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Hardcoded paths ───────────────────────────────────────────────────────────
DATA_DIR   = Path("../nse_data_cache")
NIFTY750   = Path("../nifty750.txt")
OUTPUT_DIR = Path(".")

# Benchmark for RS calculation — tries these filenames in order
BENCHMARK_CANDIDATES = ["NIFTY50.csv", "NIFTY.csv", "NIFTY_50.csv", "^NSEI.csv"]

DEFAULT_LOOKBACK = 0   # daily rows; 0 = full history


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("datetime", "date", "time", "timestamp"): col_map[c] = "Date"
        elif cl == "open":   col_map[c] = "Open"
        elif cl == "high":   col_map[c] = "High"
        elif cl == "low":    col_map[c] = "Low"
        elif cl == "close":  col_map[c] = "Close"
        elif cl == "volume": col_map[c] = "Volume"
    return df.rename(columns=col_map)


def load_csv(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, parse_dates=[0])
    df = normalise_cols(df)
    required = {"High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")
    df = df.dropna(subset=list(required))
    df = df[df["Volume"] > 0]
    if "Date" in df.columns:
        df = df.sort_values("Date")
    return df.reset_index(drop=True)


def load_benchmark() -> pd.Series | None:
    """
    Returns a Close price Series indexed by Date, or None if not found.
    Falls back to None — RS will use universe-median method instead.
    """
    for name in BENCHMARK_CANDIDATES:
        p = DATA_DIR / name
        if p.exists():
            try:
                df = load_csv(p)
                print(f"  Benchmark : {name}")
                return df.set_index("Date")["Close"]
            except Exception:
                continue
    print("  Benchmark : not found — using universe-median RS")
    return None


def safe_ma(series: pd.Series, window: int) -> float:
    """Return the latest simple moving average, or NaN if not enough data."""
    if len(series) < window:
        return float("nan")
    return float(series.iloc[-window:].mean())


def pct_return(series: pd.Series, lookback: int) -> float:
    """% return over last `lookback` rows. NaN if not enough data."""
    if len(series) < lookback + 1:
        return float("nan")
    start = series.iloc[-(lookback + 1)]
    end   = series.iloc[-1]
    if start <= 0:
        return float("nan")
    return (end - start) / start * 100


# ─────────────────────────────────────────────────────────────────────────────
#  1. TREND TEMPLATE  (Minervini Stage-2 checklist)
# ─────────────────────────────────────────────────────────────────────────────
def compute_trend(df: pd.DataFrame) -> dict:
    """
    9 binary conditions → trend_score 0–9.
    Also returns individual flags and key MA values for display.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    price  = float(close.iloc[-1])

    ma50   = safe_ma(close, 50)
    ma150  = safe_ma(close, 150)
    ma200  = safe_ma(close, 200)
    ma200_20ago = float(close.iloc[-220:-20].mean()) if len(close) >= 220 else float("nan")

    # 52-week window = last 252 trading sessions
    w52 = 252
    high52 = float(high.iloc[-w52:].max()) if len(high) >= w52 else float(high.max())
    low52  = float(low.iloc[-w52:].min())  if len(low)  >= w52 else float(low.min())

    def chk(cond): return 1 if (cond and not np.isnan(cond)) else 0

    c1 = chk(price > ma150)                     # price above 150 MA
    c2 = chk(price > ma200)                     # price above 200 MA
    c3 = chk(ma150 > ma200)                     # 150 MA above 200 MA
    c4 = chk(ma200 > ma200_20ago)               # 200 MA trending up (vs 20 sessions ago)
    c5 = chk(ma50  > ma150)                     # 50 MA above 150 MA
    c6 = chk(ma50  > ma200)                     # 50 MA above 200 MA
    c7 = chk(price > ma50)                      # price above 50 MA
    c8 = chk(price >= high52 * 0.75)            # within 25% of 52W high
    c9 = chk(price >= low52  * 1.30)            # at least 30% above 52W low

    score = c1+c2+c3+c4+c5+c6+c7+c8+c9

    return {
        "Trend Score"  : score,
        "MA50"         : round(ma50,  2) if not np.isnan(ma50)  else None,
        "MA150"        : round(ma150, 2) if not np.isnan(ma150) else None,
        "MA200"        : round(ma200, 2) if not np.isnan(ma200) else None,
        "52W High"     : round(high52, 2),
        "52W Low"      : round(low52,  2),
        "T_c1": c1, "T_c2": c2, "T_c3": c3, "T_c4": c4, "T_c5": c5,
        "T_c6": c6, "T_c7": c7, "T_c8": c8, "T_c9": c9,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  2. RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────
# Lookback windows in trading days
RS_WINDOWS = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252}
RS_WEIGHTS = {"1M": 0.15, "3M": 0.20, "6M": 0.30, "1Y": 0.35}   # sum = 1.0


def compute_rs_raw(close: pd.Series, bench_close: pd.Series | None) -> dict:
    """
    Returns raw RS values per window.
    If benchmark provided: RS = stock_return - bench_return
    If no benchmark: RS = raw stock return (percentile ranked across universe later)
    """
    results = {}
    for label, days in RS_WINDOWS.items():
        stock_ret = pct_return(close, days)
        if bench_close is not None:
            # Align benchmark to same date range as stock
            bench_ret = pct_return(bench_close, days)
            rs = stock_ret - bench_ret if not (np.isnan(stock_ret) or np.isnan(bench_ret)) else float("nan")
        else:
            rs = stock_ret
        results[f"RS_{label}"] = round(rs, 3) if not np.isnan(rs) else None
    return results


def compute_rs_scores(df_results: pd.DataFrame) -> pd.Series:
    """
    Convert raw RS values to a 0–100 composite RS Score via percentile ranking.
    Weighted average across windows.
    """
    weighted_raw = pd.Series(0.0, index=df_results.index)
    weight_used  = pd.Series(0.0, index=df_results.index)

    for label, w in RS_WEIGHTS.items():
        col = f"RS_{label}"
        if col not in df_results.columns:
            continue
        vals = pd.to_numeric(df_results[col], errors="coerce")
        valid = vals.notna()
        weighted_raw[valid] += vals[valid] * w
        weight_used[valid]  += w

    # Normalise for rows with partial data
    has_data = weight_used > 0
    composite = pd.Series(float("nan"), index=df_results.index)
    composite[has_data] = weighted_raw[has_data] / weight_used[has_data]

    # Percentile rank 1–99
    ranked = composite.rank(pct=True, na_option="bottom") * 98 + 1
    return ranked.round(1)


# ─────────────────────────────────────────────────────────────────────────────
#  3. VCP — Volatility / Volume Contraction Pattern
# ─────────────────────────────────────────────────────────────────────────────
def compute_vcp(df: pd.DataFrame) -> dict:
    """
    Score 0–100 measuring how close the stock is to a VCP setup:

    Three components (equal weight):
      A. Volume dry-up  : recent 10-day avg vol vs 50-day avg vol
      B. Price proximity: how close current price is to recent 52W high
      C. Range tightening: recent 10-day avg daily range vs 50-day avg range

    Each component 0–100, final VCP = mean of three.
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── A. Volume dry-up ─────────────────────────────────────
    vol10 = float(volume.iloc[-10:].mean())  if len(volume) >= 10 else float("nan")
    vol50 = float(volume.iloc[-50:].mean())  if len(volume) >= 50 else float("nan")
    if np.isnan(vol10) or np.isnan(vol50) or vol50 == 0:
        vol_ratio = float("nan")
        vol_score = 50.0
    else:
        vol_ratio = vol10 / vol50
        # Score: ratio=0.0 → 100, ratio=0.5 → 75, ratio=1.0 → 50, ratio≥2.0 → 0
        vol_score = float(max(0, min(100, (1 - vol_ratio) * 100)))

    # ── B. Price proximity to 52W high ────────────────────────
    w52   = min(252, len(high))
    high52 = float(high.iloc[-w52:].max())
    price  = float(close.iloc[-1])
    if high52 > 0:
        pct_from_high = (high52 - price) / high52 * 100
        # 0% from high → 100, 25% from high → 0
        prox_score = float(max(0, min(100, (1 - pct_from_high / 25) * 100)))
    else:
        pct_from_high = float("nan")
        prox_score    = 50.0

    # ── C. Daily range contraction ────────────────────────────
    daily_range = (high - low) / close
    range10 = float(daily_range.iloc[-10:].mean()) if len(daily_range) >= 10 else float("nan")
    range50 = float(daily_range.iloc[-50:].mean()) if len(daily_range) >= 50 else float("nan")
    if np.isnan(range10) or np.isnan(range50) or range50 == 0:
        range_ratio = float("nan")
        range_score = 50.0
    else:
        range_ratio = range10 / range50
        # ratio=0.3 → 100, ratio=0.7 → 70, ratio=1.0 → 50, ratio≥1.5 → 0
        range_score = float(max(0, min(100, (1 - range_ratio) * 100)))

    vcp_score = round((vol_score + prox_score + range_score) / 3, 1)

    return {
        "VCP Score"     : vcp_score,
        "Vol Ratio"     : round(vol_ratio,    3) if not np.isnan(vol_ratio)    else None,
        "% From High"   : round(pct_from_high, 2) if not np.isnan(pct_from_high) else None,
        "Range Ratio"   : round(range_ratio,  3) if not np.isnan(range_ratio)  else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SCAN
# ─────────────────────────────────────────────────────────────────────────────
def scan(lookback_days: int = DEFAULT_LOOKBACK) -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {DATA_DIR.resolve()}")

    print(f"\n  NSE Screener  v1  —  Trend + RS + VCP")
    print(f"  {'─'*52}")
    print(f"  Data dir   : {DATA_DIR.resolve()}")
    print(f"  Stocks     : {len(csv_files)}")

    bench_close = load_benchmark()
    lookback_lbl = f"last {lookback_days} sessions" if lookback_days > 0 else "full history"
    print(f"  Lookback   : {lookback_lbl}\n")

    results, errors = [], []

    # Skip benchmark files themselves
    bench_stems = {Path(b).stem.upper() for b in BENCHMARK_CANDIDATES}

    for i, fpath in enumerate(csv_files, 1):
        symbol = fpath.stem.upper()
        if symbol in bench_stems:
            continue
        if i % 50 == 0 or i == len(csv_files):
            pct = i / len(csv_files) * 100
            bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
            print(f"  [{bar}] {pct:5.1f}%  {i}/{len(csv_files)}  {symbol:<20}", end="\r")

        try:
            df = load_csv(fpath)
            if len(df) < 60:
                continue

            # Apply lookback on daily rows
            if lookback_days > 0 and len(df) > lookback_days:
                df = df.iloc[-lookback_days:].reset_index(drop=True)

            price      = float(df["Close"].iloc[-1])
            last_date  = df["Date"].iloc[-1]
            first_date = df["Date"].iloc[0]

            if price <= 0:
                continue

            # ── Align benchmark to same date range ───────────
            bench_aligned = None
            if bench_close is not None:
                try:
                    bench_aligned = bench_close.reindex(df["Date"]).ffill()
                except Exception:
                    bench_aligned = None

            trend = compute_trend(df)
            rs    = compute_rs_raw(df["Close"], bench_aligned)
            vcp   = compute_vcp(df)

            row = {
                "Symbol"       : symbol,
                "Price"        : round(price, 2),
                "Data From"    : first_date.strftime("%Y-%m-%d") if hasattr(first_date, "strftime") else str(first_date),
                "Last Date"    : last_date.strftime("%Y-%m-%d")  if hasattr(last_date,  "strftime") else str(last_date),
                "Sessions"     : len(df),
            }
            row.update(trend)
            row.update(rs)
            row.update(vcp)
            results.append(row)

        except Exception as e:
            errors.append((symbol, str(e)))

    print(f"\n\n  ✓ {len(results)} stocks processed  |  {len(errors)} errors\n")
    if errors:
        for sym, err in errors[:10]:
            print(f"    ✗ {sym:<20} {err}")

    df_all = pd.DataFrame(results)
    if df_all.empty:
        return df_all

    # ── RS Score: percentile rank across full universe ────────
    df_all["RS Score"] = compute_rs_scores(df_all)

    # ── Composite Score ───────────────────────────────────────
    # Trend: 0–9 → normalise to 0–100
    trend_norm = (df_all["Trend Score"] / 9 * 100).round(1)
    rs_norm    = df_all["RS Score"]
    vcp_norm   = df_all["VCP Score"]

    df_all["Composite"] = (
        trend_norm * 0.40 +
        rs_norm    * 0.35 +
        vcp_norm   * 0.25
    ).round(1)

    return df_all.sort_values("Composite", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HTML OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
def fmt_val(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def trend_badges(row) -> str:
    """9 tiny coloured squares showing individual condition pass/fail."""
    labels = ["P>150","P>200","150>200","200↑","50>150","50>200","P>50","≤25%Hi","≥30%Lo"]
    squares = []
    for i, lbl in enumerate(labels, 1):
        val = row.get(f"T_c{i}", 0)
        colour = "#26d87c" if val else "#ff4d6d"
        squares.append(f'<span title="{lbl}" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:{colour};margin:1px"></span>')
    return "".join(squares)


def score_colour(score: float, thresholds=(40, 65, 80)) -> str:
    """Return CSS colour class based on score value."""
    if score >= thresholds[2]: return "s-high"
    if score >= thresholds[1]: return "s-mid"
    if score >= thresholds[0]: return "s-low"
    return "s-dim"


def build_rows(df: pd.DataFrame) -> str:
    parts = []
    for _, r in df.iterrows():
        ts   = r.get("Trend Score", 0) or 0
        rs   = r.get("RS Score",    0) or 0
        vcp  = r.get("VCP Score",   0) or 0
        comp = r.get("Composite",   0) or 0

        parts.append(
            f'<tr class="dr"'
            f' data-symbol="{r["Symbol"]}"'
            f' data-composite="{comp}"'
            f' data-trend="{ts}"'
            f' data-rs="{rs}"'
            f' data-vcp="{vcp}"'
            f' data-price="{r["Price"]}"'
            f' data-ma50="{r.get("MA50") or ""}"'
            f' data-ma150="{r.get("MA150") or ""}"'
            f' data-ma200="{r.get("MA200") or ""}"'
            f' data-high52="{r.get("52W High") or ""}"'
            f' data-low52="{r.get("52W Low") or ""}"'
            f' data-vol-ratio="{r.get("Vol Ratio") or ""}"'
            f' data-from-high="{r.get("% From High") or ""}"'
            f' data-range-ratio="{r.get("Range Ratio") or ""}"'
            f' data-rs1m="{r.get("RS_1M") or ""}"'
            f' data-rs3m="{r.get("RS_3M") or ""}"'
            f' data-rs6m="{r.get("RS_6M") or ""}"'
            f' data-rs1y="{r.get("RS_1Y") or ""}"'
            f' data-sessions="{r.get("Sessions") or ""}">'
            f'<td class="sym">{r["Symbol"]}</td>'
            f'<td>₹{fmt_val(r["Price"])}</td>'
            f'<td class="{score_colour(comp)}"><b>{fmt_val(comp, 1)}</b></td>'
            f'<td class="{score_colour(ts/9*100)}">{int(ts)}/9 {trend_badges(r)}</td>'
            f'<td class="{score_colour(rs)}">{fmt_val(rs, 1)}</td>'
            f'<td class="{score_colour(vcp)}">{fmt_val(vcp, 1)}</td>'
            f'<td>{fmt_val(r.get("RS_1M"))}</td>'
            f'<td>{fmt_val(r.get("RS_3M"))}</td>'
            f'<td>{fmt_val(r.get("RS_6M"))}</td>'
            f'<td>{fmt_val(r.get("RS_1Y"))}</td>'
            f'<td>{fmt_val(r.get("Vol Ratio"), 2)}</td>'
            f'<td>{fmt_val(r.get("% From High"), 1)}%</td>'
            f'<td>{fmt_val(r.get("Range Ratio"), 2)}</td>'
            f'<td>₹{fmt_val(r.get("MA50"))}</td>'
            f'<td>₹{fmt_val(r.get("MA150"))}</td>'
            f'<td>₹{fmt_val(r.get("MA200"))}</td>'
            f'<td>{r.get("Sessions", "—")}</td>'
            f'</tr>'
        )
    return "\n".join(parts)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne:wght@600;800&display=swap');
:root{{
  --bg:#0d0f14;--surface:#13161e;--border:#1f2433;--muted:#3a3f52;
  --text:#c8cfe0;--dim:#6b7590;--accent:#4fffb0;--accent2:#00b8ff;
  --up:#26d87c;--down:#ff4d6d;--warn:#ffbb33;--gold:#f0b429;
  --mono:'IBM Plex Mono',monospace;--sans:'Syne',sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:12.5px;min-height:100vh}}

header{{padding:22px 28px 16px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:14px}}
.title-block h1{{font-family:var(--sans);font-size:19px;font-weight:800;color:#fff;letter-spacing:-.5px}}
.title-block h1 span{{color:var(--accent)}}
.meta{{margin-top:4px;font-size:10.5px;color:var(--dim);letter-spacing:.4px}}
.stats{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}
.pill{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:5px 12px;font-size:11px;color:var(--dim)}}
.pill b{{color:var(--text);font-weight:600}}
.pill.g b{{color:var(--up)}}
.pill.a b{{color:var(--accent)}}
.pill.w b{{color:var(--warn)}}

/* ── Score legend ── */
.s-high{{color:var(--up)  ;font-weight:600}}
.s-mid {{color:var(--warn);font-weight:500}}
.s-low {{color:#e07b39}}
.s-dim {{color:var(--dim)}}

/* ── Controls ── */
.controls{{padding:12px 28px;display:flex;align-items:center;gap:18px;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--surface)}}
.ctrl-grp{{display:flex;align-items:center;gap:8px}}
.ctrl-label{{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:.7px;white-space:nowrap}}
.range-wrap{{display:flex;align-items:center;gap:6px}}
input[type=range]{{-webkit-appearance:none;width:110px;height:3px;background:var(--muted);border-radius:2px;outline:none}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:13px;height:13px;border-radius:50%;background:var(--accent);cursor:pointer}}
.range-val{{font-size:12px;color:var(--accent);min-width:26px;text-align:right}}
input[type=text]{{background:var(--bg);border:1px solid var(--muted);border-radius:5px;color:var(--text);font-family:var(--mono);font-size:12px;padding:4px 9px;width:140px;outline:none}}
input[type=text]:focus{{border-color:var(--accent2)}}
.search-box{{margin-left:auto;display:flex;align-items:center;gap:8px}}
.count-bar{{padding:6px 28px;font-size:11px;color:var(--dim);border-bottom:1px solid var(--border)}}
.count-bar b{{color:var(--accent)}}

/* ── Table ── */
.table-wrap{{overflow-x:auto;padding-bottom:40px}}
table{{width:100%;border-collapse:collapse;min-width:1300px}}
thead{{position:sticky;top:0;z-index:10;background:var(--surface)}}

/* Column group headers */
.col-group th{{padding:4px 14px;font-family:var(--sans);font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid var(--border);text-align:center}}
.cg-base {{color:#fff;background:#13161e}}
.cg-trend{{color:var(--accent2);background:#0f1620;border-left:2px solid var(--accent2)}}
.cg-rs   {{color:var(--gold);   background:#141208;border-left:2px solid var(--gold)}}
.cg-vcp  {{color:var(--warn);   background:#141108;border-left:2px solid var(--warn)}}
.cg-ma   {{color:var(--muted);  background:#111318;border-left:2px solid var(--muted)}}

th{{padding:8px 12px;font-family:var(--sans);font-size:9.5px;font-weight:600;letter-spacing:.6px;text-transform:uppercase;color:var(--dim);text-align:right;border-bottom:2px solid var(--border);cursor:pointer;user-select:none;white-space:nowrap;transition:color .15s}}
th:first-child{{text-align:left}}
th:hover{{color:var(--accent)}}
th.asc::after{{content:" ↑";color:var(--accent)}}
th.desc::after{{content:" ↓";color:var(--accent)}}
.bl{{border-left:2px solid var(--border)}}

tr.dr:hover td{{background:#181c28}}
tr.dr.hidden{{display:none}}
td{{padding:7px 12px;border-bottom:1px solid #181c26;text-align:right;white-space:nowrap}}
td:first-child{{text-align:left}}
.sym{{font-weight:600;color:#fff;letter-spacing:.3px;font-size:13px}}

.footer{{padding:12px 28px;font-size:10px;color:var(--muted);border-top:1px solid var(--border);text-align:center;letter-spacing:.4px;line-height:1.8}}
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>NSE Screener &mdash; <span>{group_label}</span></h1>
    <div class="meta">
      Generated {gen_date} &nbsp;|&nbsp; {total_stocks} stocks &nbsp;|&nbsp;
      Trend Template + Relative Strength + VCP &nbsp;|&nbsp; Lookback: {lookback_label}
    </div>
  </div>
  <div class="stats">
    <div class="pill"><b>{total_stocks}</b> total</div>
    <div class="pill g"><b id="cnt-perfect">—</b> perfect trend (9/9)</div>
    <div class="pill a"><b id="cnt-rs80">—</b> RS &gt; 80</div>
    <div class="pill w"><b id="cnt-vcp70">—</b> VCP &gt; 70</div>
    <div class="pill" style="border-color:#f0b429"><b id="cnt-all">—</b> all three strong</div>
  </div>
</header>

<div class="controls">
  <span class="ctrl-label">Min Trend</span>
  <div class="range-wrap">
    <input type="range" id="sl-trend" min="0" max="9" step="1" value="0">
    <span class="range-val" id="vl-trend">0</span><span style="font-size:10px;color:var(--dim)">/9</span>
  </div>

  <span class="ctrl-label">Min RS</span>
  <div class="range-wrap">
    <input type="range" id="sl-rs" min="0" max="99" step="1" value="0">
    <span class="range-val" id="vl-rs">0</span>
  </div>

  <span class="ctrl-label">Min VCP</span>
  <div class="range-wrap">
    <input type="range" id="sl-vcp" min="0" max="99" step="1" value="0">
    <span class="range-val" id="vl-vcp">0</span>
  </div>

  <span class="ctrl-label">Min Composite</span>
  <div class="range-wrap">
    <input type="range" id="sl-comp" min="0" max="99" step="1" value="0">
    <span class="range-val" id="vl-comp">0</span>
  </div>

  <div class="search-box">
    <span class="ctrl-label">Symbol</span>
    <input type="text" id="sym-search" placeholder="Filter symbol…">
  </div>
</div>

<div class="count-bar">Showing <b id="vis-count">—</b> / {total_stocks}</div>

<div class="table-wrap">
<table>
<thead>
<tr class="col-group">
  <th class="cg-base" colspan="3">Base</th>
  <th class="cg-trend bl" colspan="3">① Trend Template</th>
  <th class="cg-rs bl" colspan="4">② Relative Strength</th>
  <th class="cg-vcp bl" colspan="3">③ VCP</th>
  <th class="cg-ma bl" colspan="4">Moving Averages</th>
</tr>
<tr>
  <th data-col="symbol"     data-type="str">Symbol</th>
  <th data-col="price"      data-type="num">Price</th>
  <th data-col="composite"  data-type="num">Score</th>

  <th data-col="trend"      data-type="num" class="bl">Trend</th>
  <th data-col="rs"         data-type="num">RS</th>
  <th data-col="vcp"        data-type="num">VCP</th>

  <th data-col="rs1m"       data-type="num" class="bl">RS 1M%</th>
  <th data-col="rs3m"       data-type="num">RS 3M%</th>
  <th data-col="rs6m"       data-type="num">RS 6M%</th>
  <th data-col="rs1y"       data-type="num">RS 1Y%</th>

  <th data-col="volRatio"   data-type="num" class="bl">Vol Ratio</th>
  <th data-col="fromHigh"   data-type="num">% from Hi</th>
  <th data-col="rangeRatio" data-type="num">Rng Ratio</th>

  <th data-col="ma50"       data-type="num" class="bl">MA50</th>
  <th data-col="ma150"      data-type="num">MA150</th>
  <th data-col="ma200"      data-type="num">MA200</th>
  <th data-col="sessions"   data-type="num">Sessions</th>
</tr>
</thead>
<tbody id="tbody">
{rows}
</tbody>
</table>
</div>

<div class="footer">
  <b>Trend Template</b>: 9 Minervini Stage-2 conditions (price vs MA50/150/200, MA slopes, 52W range) &nbsp;·&nbsp;
  <b>RS Score</b>: percentile rank vs universe, weighted 1M×15% + 3M×20% + 6M×30% + 1Y×35% &nbsp;·&nbsp;
  <b>VCP</b>: Volume dry-up (vol10/vol50) + proximity to 52W high + daily range tightening<br>
  <b>Composite</b>: Trend×40% + RS×35% + VCP×25% &nbsp;·&nbsp;
  Trend badges (left→right): P&gt;150MA · P&gt;200MA · 150&gt;200MA · 200MA↑ · 50&gt;150MA · 50&gt;200MA · P&gt;50MA · ≤25% from 52W Hi · ≥30% above 52W Lo
</div>

<script>
const allRows = Array.from(document.querySelectorAll("tr.dr"));
const tbody   = document.getElementById("tbody");

const slTrend = document.getElementById("sl-trend");
const slRs    = document.getElementById("sl-rs");
const slVcp   = document.getElementById("sl-vcp");
const slComp  = document.getElementById("sl-comp");
const vlTrend = document.getElementById("vl-trend");
const vlRs    = document.getElementById("vl-rs");
const vlVcp   = document.getElementById("vl-vcp");
const vlComp  = document.getElementById("vl-comp");
const symSearch = document.getElementById("sym-search");

const cntPerfect = document.getElementById("cnt-perfect");
const cntRs80    = document.getElementById("cnt-rs80");
const cntVcp70   = document.getElementById("cnt-vcp70");
const cntAll     = document.getElementById("cnt-all");
const visCount   = document.getElementById("vis-count");

function applyFilters() {{
  const minTrend = parseInt(slTrend.value);
  const minRs    = parseFloat(slRs.value);
  const minVcp   = parseFloat(slVcp.value);
  const minComp  = parseFloat(slComp.value);
  const search   = symSearch.value.trim().toUpperCase();

  let vis=0, perf=0, rs80=0, vcp70=0, allThree=0;

  allRows.forEach(row => {{
    const trend = parseFloat(row.dataset.trend);
    const rs    = parseFloat(row.dataset.rs);
    const vcp   = parseFloat(row.dataset.vcp);
    const comp  = parseFloat(row.dataset.composite);
    const sym   = row.dataset.symbol || "";

    // Global stats
    if (trend === 9)  perf++;
    if (rs   >= 80)   rs80++;
    if (vcp  >= 70)   vcp70++;
    if (trend >= 7 && rs >= 70 && vcp >= 60) allThree++;

    const pass = trend >= minTrend && rs >= minRs && vcp >= minVcp &&
                 comp >= minComp && (!search || sym.includes(search));
    row.classList.toggle("hidden", !pass);
    if (pass) vis++;
  }});

  vlTrend.textContent = slTrend.value;
  vlRs.textContent    = slRs.value;
  vlVcp.textContent   = slVcp.value;
  vlComp.textContent  = slComp.value;

  cntPerfect.textContent = perf;
  cntRs80.textContent    = rs80;
  cntVcp70.textContent   = vcp70;
  cntAll.textContent     = allThree;
  visCount.textContent   = vis;
}}

[slTrend, slRs, slVcp, slComp].forEach(s => s.addEventListener("input", applyFilters));
symSearch.addEventListener("input", applyFilters);

// ── Sort ──────────────────────────────────────────────────
let sortCol = "", sortDir = 1;

document.querySelectorAll("th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col  = th.dataset.col;
    const type = th.dataset.type;
    if (sortCol === col) {{ sortDir *= -1; }}
    else {{ sortCol = col; sortDir = 1; }}

    document.querySelectorAll("th[data-col]").forEach(t => t.classList.remove("asc","desc"));
    th.classList.add(sortDir === 1 ? "asc" : "desc");

    const sorted = [...allRows].sort((a, b) => {{
      const av = a.dataset[col] ?? "";
      const bv = b.dataset[col] ?? "";
      if (type === "num") {{
        const an = parseFloat(av), bn = parseFloat(bv);
        if (isNaN(an) && isNaN(bn)) return 0;
        if (isNaN(an)) return 1;
        if (isNaN(bn)) return -1;
        return (an - bn) * sortDir;
      }}
      return av.localeCompare(bv) * sortDir;
    }});
    sorted.forEach(r => tbody.appendChild(r));
    applyFilters();
  }});
}});

applyFilters();
</script>
</body>
</html>
"""


def build_html(df: pd.DataFrame, title: str, group_label: str,
               lookback_days: int = DEFAULT_LOOKBACK) -> str:
    lbl = f"last {lookback_days} daily sessions" if lookback_days > 0 else "full history"
    return HTML.format(
        title         = title,
        group_label   = group_label,
        gen_date      = date.today().strftime("%d %b %Y"),
        total_stocks  = len(df),
        lookback_label= lbl,
        rows          = build_rows(df),
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def load_nifty750(path: Path) -> set:
    symbols = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(",")[0].split()[0].split(".")[0].upper()
            symbols.add(token)
    return symbols


def parse_args():
    p = argparse.ArgumentParser(description="NSE Screener — Trend + RS + VCP")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK,
                   help=f"Daily rows per stock (default {DEFAULT_LOOKBACK} ≈ 5 yr). 0 = all.")
    p.add_argument("--csv", action="store_true", help="Also save CSV results")
    return p.parse_args()


def main():
    args = parse_args()

    for p, label in [(DATA_DIR, "Data dir"), (NIFTY750, "Nifty750")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p.resolve()}")

    n750   = load_nifty750(NIFTY750)
    df_all = scan(lookback_days=args.lookback_days)

    if df_all.empty:
        print("  No results."); return

    df_n750   = df_all[df_all["Symbol"].isin(n750)].copy()
    df_others = df_all[~df_all["Symbol"].isin(n750)].copy()

    print(f"  Nifty750 : {len(df_n750)}   Others : {len(df_others)}\n")

    for df_g, title, label, fname in [
        (df_n750,   "NSE Screener — Nifty 750",    "Nifty 750",    "nifty750_screener.html"),
        (df_others, "NSE Screener — Other Stocks", "Other Stocks", "others_screener.html"),
    ]:
        if df_g.empty:
            print(f"  ⚠  No data for {label}"); continue

        html     = build_html(df_g, title, label, args.lookback_days)
        out_path = OUTPUT_DIR / fname
        out_path.write_text(html, encoding="utf-8")
        size_kb  = out_path.stat().st_size // 1024
        print(f"  ✓ {fname:<34} {len(df_g):>4} stocks  {size_kb} KB")

        if args.csv:
            csv_p = OUTPUT_DIR / fname.replace(".html", ".csv")
            # Drop individual condition columns from CSV (too wide)
            drop_cols = [c for c in df_g.columns if c.startswith("T_c")]
            df_g.drop(columns=drop_cols, errors="ignore").to_csv(csv_p, index=False)
            print(f"    ↳ CSV: {csv_p}")

    print(f"\n  Open nifty750_screener.html or others_screener.html in your browser.\n")


if __name__ == "__main__":
    main()
