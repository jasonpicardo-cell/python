"""
NSE Volume Profile Scanner  v4
================================
• Reads CSVs from  ../nse_data_cache
• Nifty750 list from  ../nifty750.txt
• Writes HTML files in current directory

Run:
    python volume_profile_scanner.py                           # daily, last 1250 sessions
    python volume_profile_scanner.py --timeframe weekly        # weekly candles
    python volume_profile_scanner.py --lookback-days 500       # last 500 daily sessions
    python volume_profile_scanner.py --lookback-days 0         # full history
    python volume_profile_scanner.py --bins 200
    python volume_profile_scanner.py --csv

Lookback note (always applied on daily rows before any aggregation):
    1250 days ≈ 5 years  →  ~250 weekly candles
     500 days ≈ 2 years  →  ~100 weekly candles
     250 days ≈ 1 year   →   ~52 weekly candles
       0      = entire CSV history

Special sessions (Diwali Mahurat, Budget on Sat/Sun) are included
correctly because weekly grouping uses ISO week number, not calendar
Mon–Fri windows.
"""

import argparse, warnings
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Hardcoded paths ───────────────────────────────────────────
DATA_DIR   = Path("../nse_data_cache")
NIFTY750   = Path("../nifty750.txt")
OUTPUT_DIR = Path(".")          # current directory

DEFAULT_BINS      = 150
DEFAULT_LOOKBACK  = 0         # trading days; 0 = full history
DEFAULT_TIMEFRAME = "daily"       # "daily" | "weekly"


# ────────────────────────────────────────────────────────────
#  VOLUME PROFILE
# ────────────────────────────────────────────────────────────
def compute_volume_profile(df: pd.DataFrame, bins: int):
    price_min = df["Low"].min()
    price_max = df["High"].max()
    if price_min >= price_max:
        return float(price_min), float(df["Volume"].sum()), pd.Series(dtype=float)

    edges   = np.linspace(price_min, price_max, bins + 1)
    mids    = (edges[:-1] + edges[1:]) / 2
    bin_vol = np.zeros(bins, dtype=np.float64)

    for lo, hi, vol in zip(df["Low"].values, df["High"].values,
                           df["Volume"].values.astype(float)):
        if vol == 0:
            continue
        i0   = max(0,    int(np.searchsorted(edges, lo, "left")))
        i1   = min(bins, int(np.searchsorted(edges, hi, "right")))
        span = max(1, i1 - i0)
        bin_vol[i0:i0+span] += vol / span

    poc_idx = int(np.argmax(bin_vol))
    return float(mids[poc_idx]), float(bin_vol[poc_idx]), pd.Series(bin_vol, index=mids)


def value_area(profile: pd.Series, pct: float = 0.70):
    if profile.empty:
        return float("nan"), float("nan")
    total   = profile.sum()
    target  = total * pct
    poc_pos = int(profile.values.argmax())
    accum   = profile.iloc[poc_pos]
    lo, hi  = poc_pos, poc_pos
    while accum < target:
        can_lo = lo > 0
        can_hi = hi < len(profile) - 1
        if not can_lo and not can_hi:
            break
        v_lo = profile.iloc[lo - 1] if can_lo else -1
        v_hi = profile.iloc[hi + 1] if can_hi else -1
        if v_lo >= v_hi:
            lo -= 1; accum += v_lo
        else:
            hi += 1; accum += v_hi
    return float(profile.index[lo]), float(profile.index[hi])


# ────────────────────────────────────────────────────────────
#  SCAN
# ────────────────────────────────────────────────────────────
def load_nifty750(path: Path) -> set:
    symbols = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split(",")[0].split()[0]
            token = token.split(".")[0].upper()
            symbols.add(token)
    return symbols


def normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("datetime","date","time","timestamp"): col_map[c] = "Date"
        elif cl == "open":   col_map[c] = "Open"
        elif cl == "high":   col_map[c] = "High"
        elif cl == "low":    col_map[c] = "Low"
        elif cl == "close":  col_map[c] = "Close"
        elif cl == "volume": col_map[c] = "Volume"
    return df.rename(columns=col_map)


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily OHLCV rows into weekly candles using ISO week number.

    Grouping key: (ISO year, ISO week)  — handles year-boundary weeks correctly
    and includes special trading sessions (Sat/Sun) in the same week they occur,
    unlike resample("W-FRI") which would discard or misplace them.

    Aggregation:
        Open   = first session open of the week
        High   = max High across all sessions
        Low    = min Low  across all sessions
        Close  = last session close of the week
        Volume = sum of all session volumes
        Date   = date of the last session (week-end anchor)
    """
    if "Date" not in df.columns or len(df) == 0:
        return df

    iso = df["Date"].dt.isocalendar()          # returns year/week/day columns
    df = df.copy()
    df["_iso_year"] = iso["year"].values
    df["_iso_week"] = iso["week"].values

    weekly = (
        df.groupby(["_iso_year", "_iso_week"], sort=True)
        .agg(
            Date   = ("Date",   "last"),    # last trading day of the week
            Open   = ("Open",   "first"),
            High   = ("High",   "max"),
            Low    = ("Low",    "min"),
            Close  = ("Close",  "last"),
            Volume = ("Volume", "sum"),
        )
        .reset_index(drop=True)
    )
    return weekly


def scan_directory(bins: int = DEFAULT_BINS, lookback_days: int = DEFAULT_LOOKBACK,
                   timeframe: str = DEFAULT_TIMEFRAME) -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in: {DATA_DIR.resolve()}")

    lookback_label = f"last {lookback_days} daily sessions" if lookback_days > 0 else "full history"
    tf_label       = timeframe.capitalize()
    print(f"\n  NSE Volume Profile Scanner v4")
    print(f"  Data dir  : {DATA_DIR.resolve()}")
    print(f"  Timeframe : {tf_label}")
    print(f"  Stocks    : {len(csv_files)}  |  Bins: {bins}  |  Lookback: {lookback_label}\n")

    results, errors = [], []

    for i, fpath in enumerate(csv_files, 1):
        symbol = fpath.stem.upper()
        if i % 50 == 0 or i == len(csv_files):
            pct = i / len(csv_files) * 100
            bar = "█" * int(pct/5) + "░" * (20-int(pct/5))
            print(f"  [{bar}] {pct:5.1f}%  {i}/{len(csv_files)}  {symbol:<20}", end="\r")
        try:
            df = pd.read_csv(fpath, parse_dates=[0])
            df = normalise_cols(df)
            required = {"High","Low","Close","Volume"}
            if not required.issubset(df.columns):
                errors.append((symbol, f"Missing: {required - set(df.columns)}")); continue
            df = df.dropna(subset=list(required))
            df = df[df["Volume"] > 0]
            if "Date" in df.columns:
                df = df.sort_values("Date")
            df = df.reset_index(drop=True)

            # ── Lookback: slice the most recent N daily rows ──
            # Applied on daily data first so --lookback-days always means
            # the same calendar window regardless of timeframe.
            if lookback_days > 0 and len(df) > lookback_days:
                df = df.iloc[-lookback_days:].reset_index(drop=True)

            # ── Weekly aggregation (ISO week grouping) ──────────────
            if timeframe == "weekly":
                df = resample_to_weekly(df)

            if len(df) < 5:
                continue

            last_row      = df.iloc[-1]
            current_price = float(last_row["Close"])
            first_date    = df["Date"].iloc[0]  if "Date" in df.columns else ""
            last_date     = df["Date"].iloc[-1] if "Date" in df.columns else ""
            if current_price <= 0:
                continue

            poc_price, poc_volume, profile = compute_volume_profile(df, bins)
            val, vah = value_area(profile)

            # Signed dist: positive = price ABOVE poc, negative = price BELOW poc
            dist_pct = round(((current_price - poc_price) / poc_price) * 100, 3)

            results.append({
                "Symbol"       : symbol,
                "Current Price": round(current_price, 2),
                "POC Price"    : round(poc_price, 2),
                "Dist %"       : dist_pct,                        # signed
                "Abs Dist %"   : round(abs(dist_pct), 3),
                "VAL"          : round(val, 2) if not np.isnan(val) else None,
                "VAH"          : round(vah, 2) if not np.isnan(vah) else None,
                "POC Volume"   : int(poc_volume),
                "Total Volume" : int(df["Volume"].sum()),
                "Candles"      : len(df),    # sessions (days or weeks depending on timeframe)
                "Data From"    : first_date.strftime("%Y-%m-%d") if hasattr(first_date,"strftime") else str(first_date),
                "Last Date"    : last_date.strftime("%Y-%m-%d")  if hasattr(last_date,"strftime")  else str(last_date),
            })
        except Exception as e:
            errors.append((symbol, str(e)))

    print(f"\n\n  ✓ {len(results)} processed  |  {len(errors)} errors\n")
    if errors:
        for sym, err in errors[:10]:
            print(f"    ✗ {sym:<20} {err}")

    return pd.DataFrame(results).sort_values("Dist %").reset_index(drop=True)


# ────────────────────────────────────────────────────────────
#  HTML
# ────────────────────────────────────────────────────────────
def fmt_vol(v):
    v = int(v)
    if v >= 1_000_000_000: return f"{v/1e9:.2f}B"
    if v >= 1_000_000:     return f"{v/1e6:.2f}M"
    if v >= 1_000:         return f"{v/1e3:.1f}K"
    return str(v)


def build_rows(df: pd.DataFrame) -> str:
    parts = []
    for _, r in df.iterrows():
        dist  = r["Dist %"]
        arrow = "▲" if dist >= 0 else "▼"
        val   = r["VAL"] if r["VAL"] is not None else "—"
        vah   = r["VAH"] if r["VAH"] is not None else "—"
        parts.append(
            f'<tr class="data-row"'
            f' data-symbol="{r["Symbol"]}"'
            f' data-dist="{dist}"'
            f' data-abs="{r["Abs Dist %"]}"'
            f' data-current-price="{r["Current Price"]}"'
            f' data-poc-price="{r["POC Price"]}"'
            f' data-val="{val}" data-vah="{vah}"'
            f' data-poc-vol="{r["POC Volume"]}"'
            f' data-total-vol="{r["Total Volume"]}"'
            f' data-candles="{r["Candles"]}"'
            f' data-data-from="{r["Data From"]}"'
            f' data-last-date="{r["Last Date"]}">'
            f'<td class="sym">{r["Symbol"]}</td>'
            f'<td>{r["Current Price"]}</td>'
            f'<td>{r["POC Price"]}</td>'
            f'<td class="dist-cell"><span class="arrow">{arrow}</span> {dist}%</td>'
            f'<td class="abs-cell"><span class="badge badge-hide">{r["Abs Dist %"]}%</span></td>'
            f'<td>{val}</td>'
            f'<td>{vah}</td>'
            f'<td>{fmt_vol(r["POC Volume"])}</td>'
            f'<td>{fmt_vol(r["Total Volume"])}</td>'
            f'<td>{r["Candles"]}</td>'
            f'<td>{r["Data From"]}</td>'
            f'<td>{r["Last Date"]}</td>'
            f'</tr>'
        )
    return "\n".join(parts)


HTML_TEMPLATE = """<!DOCTYPE html>
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
  --up:#26d87c;--down:#ff4d6d;--warn:#ffbb33;
  --mono:'IBM Plex Mono',monospace;--sans:'Syne',sans-serif;
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:var(--mono);font-size:13px;min-height:100vh}}

header{{padding:24px 32px 18px;border-bottom:1px solid var(--border);display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px}}
.title-block h1{{font-family:var(--sans);font-size:20px;font-weight:800;color:#fff;letter-spacing:-.5px}}
.title-block h1 span{{color:var(--accent)}}
.meta{{margin-top:4px;font-size:11px;color:var(--dim);letter-spacing:.4px}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;align-items:center}}
.pill{{background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 14px;font-size:11px;color:var(--dim)}}
.pill b{{color:var(--text);font-weight:600}}
.pill.hi b{{color:var(--up)}}
.pill.lo b{{color:var(--down)}}

/* ── Controls ── */
.controls{{
  padding:14px 32px;display:flex;align-items:center;gap:20px;
  flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--surface)
}}
.ctrl-label{{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.8px;white-space:nowrap}}

/* POC input box */
.poc-input-wrap{{
  display:flex;align-items:center;gap:8px;
  background:var(--bg);border:1px solid var(--muted);border-radius:6px;
  padding:6px 12px;
}}
.poc-input-wrap input[type=number]{{
  width:80px;background:transparent;border:none;
  color:var(--accent);font-family:var(--mono);font-size:15px;font-weight:600;
  text-align:center;outline:none;
}}
.poc-input-wrap .pct{{font-size:12px;color:var(--dim)}}
.poc-hint{{font-size:11px;color:var(--dim);max-width:360px;line-height:1.5}}
.poc-hint b{{color:var(--accent)}}

input[type=text]{{
  background:var(--bg);border:1px solid var(--muted);border-radius:5px;
  color:var(--text);font-family:var(--mono);font-size:12px;padding:5px 10px;
  width:150px;outline:none;
}}
input[type=text]:focus{{border-color:var(--accent2)}}
.search-box{{margin-left:auto;display:flex;align-items:center;gap:8px}}

.count-bar{{padding:7px 32px;font-size:11px;color:var(--dim);border-bottom:1px solid var(--border)}}
.count-bar b{{color:var(--accent)}}

/* ── Table ── */
.table-wrap{{overflow-x:auto;padding-bottom:40px}}
table{{width:100%;border-collapse:collapse;min-width:980px}}
thead{{position:sticky;top:0;z-index:10;background:var(--surface)}}
th{{
  padding:10px 14px;font-family:var(--sans);font-size:10px;font-weight:600;
  letter-spacing:.7px;text-transform:uppercase;color:var(--dim);text-align:right;
  border-bottom:1px solid var(--border);cursor:pointer;user-select:none;
  white-space:nowrap;transition:color .15s
}}
th:first-child{{text-align:left}}
th:hover{{color:var(--accent)}}
th.asc::after{{content:" ↑";color:var(--accent)}}
th.desc::after{{content:" ↓";color:var(--accent)}}

tr.data-row:hover td{{background:#181c28}}
tr.data-row.hidden{{display:none}}
td{{padding:8px 14px;border-bottom:1px solid #181c26;text-align:right;white-space:nowrap}}
td:first-child{{text-align:left}}
.sym{{font-weight:600;color:#fff;letter-spacing:.3px}}

/* dist cell colour states */
.dist-cell{{transition:color .2s}}
.dist-above {{color:var(--up)}}
.dist-below {{color:var(--down)}}
.dist-match {{color:var(--accent);font-weight:600}}
.dist-other {{color:var(--dim)}}

.badge{{
  display:inline-block;padding:1px 6px;border-radius:3px;
  font-size:10px;font-weight:600;letter-spacing:.3px;margin-left:4px
}}
.badge-pos  {{background:rgba(38,216,124,.12);color:var(--up);  border:1px solid rgba(38,216,124,.25)}}
.badge-neg  {{background:rgba(255,77,109,.12); color:var(--down);border:1px solid rgba(255,77,109,.25)}}
.badge-at   {{background:rgba(79,255,176,.12); color:var(--accent);border:1px solid rgba(79,255,176,.25)}}
.badge-hide {{background:transparent;color:var(--muted);border:1px solid var(--border)}}

.footer{{padding:14px 32px;font-size:10px;color:var(--muted);border-top:1px solid var(--border);text-align:center;letter-spacing:.5px}}
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>Volume Profile &mdash; <span>{group_label}</span></h1>
    <div class="meta">Generated {gen_date} &nbsp;|&nbsp; {total_stocks} stocks &nbsp;|&nbsp; {tf_badge} &nbsp;|&nbsp; Lookback: {lookback_label}</div>
  </div>
  <div class="stats">
    <div class="pill"><b>{total_stocks}</b> total</div>
    <div class="pill"><b>{avg_candles}</b> avg {candle_label}</div>
    <div class="pill hi"><b id="above-count">—</b> above POC</div>
    <div class="pill lo"><b id="below-count">—</b> below POC</div>
    <div class="pill"><b id="match-count">—</b> matching filter</div>
  </div>
</header>

<div class="controls">
  <span class="ctrl-label">POC Filter</span>

  <div class="poc-input-wrap">
    <input type="number" id="poc-input" value="1" step="0.1" min="-100" max="100">
    <span class="pct">%</span>
  </div>

  <span class="poc-hint">
    <b id="hint-text">Showing: 0% to +1%</b><br>
    Positive → stocks <em>above</em> POC (0 to +N%) &nbsp;|&nbsp; Negative → stocks <em>below</em> POC (−N% to 0)
  </span>

  <div class="search-box">
    <span class="ctrl-label">Symbol</span>
    <input type="text" id="sym-search" placeholder="Filter symbol…">
  </div>
</div>

<div class="count-bar">
  Showing <b id="vis-count">—</b> / {total_stocks} stocks
</div>

<div class="table-wrap">
<table>
<thead><tr>
  <th data-col="symbol"       data-type="str">Symbol</th>
  <th data-col="currentPrice" data-type="num">Current ₹</th>
  <th data-col="pocPrice"     data-type="num">POC ₹</th>
  <th data-col="dist"         data-type="num">Dist %</th>
  <th data-col="abs"          data-type="num">|Dist %|</th>
  <th data-col="val"          data-type="num">VAL ₹</th>
  <th data-col="vah"          data-type="num">VAH ₹</th>
  <th data-col="pocVol"       data-type="num">POC Vol</th>
  <th data-col="totalVol"     data-type="num">Total Vol</th>
  <th data-col="candles"      data-type="num">Candles</th>
  <th data-col="dataFrom"     data-type="str">Data From</th>
  <th data-col="lastDate"     data-type="str">Last Date</th>
</tr></thead>
<tbody id="tbody">
{rows}
</tbody>
</table>
</div>

<div class="footer">
  POC = Point of Control (highest-volume price level) &nbsp;·&nbsp;
  VAL / VAH = Value Area Low / High (70% of volume) &nbsp;·&nbsp;
  Dist % is signed: + means price is above POC, − means price is below POC
</div>

<script>
const allRows   = Array.from(document.querySelectorAll("tr.data-row"));
const tbody     = document.getElementById("tbody");
const pocInput  = document.getElementById("poc-input");
const symSearch = document.getElementById("sym-search");
const hintText  = document.getElementById("hint-text");

const aboveCount = document.getElementById("above-count");
const belowCount = document.getElementById("below-count");
const matchCount = document.getElementById("match-count");
const visCount   = document.getElementById("vis-count");

function applyFilters() {{
  const val    = parseFloat(pocInput.value);
  const search = symSearch.value.trim().toUpperCase();
  const isPos  = val >= 0;
  const thresh = Math.abs(val);

  // Describe the active filter in plain English
  if (isPos) {{
    hintText.textContent = thresh === 0
      ? "Showing: exactly at POC (0%)"
      : `Showing: 0% to +${{thresh.toFixed(1)}}%  (price above POC within ${{thresh.toFixed(1)}}%)`;
  }} else {{
    hintText.textContent = `Showing: −${{thresh.toFixed(1)}}% to 0%  (price below POC within ${{thresh.toFixed(1)}}%)`;
  }}

  let above=0, below=0, matched=0, visible=0;

  allRows.forEach(row => {{
    const dist = parseFloat(row.dataset.dist);
    const sym  = row.dataset.symbol || "";

    // Global stats (regardless of search)
    if (dist >= 0) above++; else below++;

    const passSearch = !search || sym.includes(search);

    // Filter logic
    let inRange;
    if (isPos) {{
      inRange = dist >= 0 && dist <= thresh;   // 0 to +N
    }} else {{
      inRange = dist < 0 && dist >= -thresh;   // -N to 0
    }}

    if (!passSearch || !inRange) {{
      row.classList.add("hidden");
      return;
    }}

    row.classList.remove("hidden");
    visible++;
    matched++;

    // Colour dist cell and abs badge
    const dc   = row.querySelector(".dist-cell");
    const absc = row.querySelector(".abs-cell");
    const bd   = absc ? absc.querySelector(".badge") : null;
    const absD = Math.abs(dist);

    if (absD <= 0.2) {{
      dc.className = "dist-cell dist-match";
      if (bd) {{ bd.className = "badge badge-at"; bd.textContent = "AT POC"; }}
    }} else if (dist > 0) {{
      dc.className = "dist-cell dist-above";
      if (bd) {{ bd.className = "badge badge-pos"; bd.textContent = `${{absD.toFixed(3)}}%`; }}
    }} else {{
      dc.className = "dist-cell dist-below";
      if (bd) {{ bd.className = "badge badge-neg"; bd.textContent = `${{absD.toFixed(3)}}%`; }}
    }}
  }});

  aboveCount.textContent = above;
  belowCount.textContent = below;
  matchCount.textContent = matched;
  visCount.textContent   = visible;
}}

pocInput.addEventListener("input", applyFilters);
symSearch.addEventListener("input", applyFilters);

// ── Sorting ──────────────────────────────────────────────
// Empty string so first click on ANY column always sorts ascending
let sortCol = "", sortDir = 1;

document.querySelectorAll("th[data-col]").forEach(th => {{
  th.addEventListener("click", () => {{
    const col  = th.dataset.col;
    const type = th.dataset.type;

    if (sortCol === col) {{
      sortDir *= -1;             // same column: flip asc ↔ desc
    }} else {{
      sortCol = col;
      sortDir = 1;               // new column: always start ascending
    }}

    document.querySelectorAll("th[data-col]").forEach(t => t.classList.remove("asc","desc"));
    th.classList.add(sortDir === 1 ? "asc" : "desc");

    // NOTE: data-current-price → dataset.currentPrice (browser converts hyphens to camelCase)
    //       data-poc-price     → dataset.pocPrice
    //       data-poc-vol       → dataset.pocVol  etc.
    const sorted = [...allRows].sort((a, b) => {{
      const av = a.dataset[col] ?? "";
      const bv = b.dataset[col] ?? "";
      if (type === "num") {{
        const an = parseFloat(av), bn = parseFloat(bv);
        if (isNaN(an) && isNaN(bn)) return 0;
        if (isNaN(an)) return 1 * sortDir;
        if (isNaN(bn)) return -1 * sortDir;
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
               lookback_days: int = DEFAULT_LOOKBACK,
               timeframe: str = DEFAULT_TIMEFRAME) -> str:
    candle_label   = "weeks" if timeframe == "weekly" else "sessions"
    avg_candles    = f"{int(df['Candles'].mean()):,}" if not df.empty else "0"
    lookback_label = (f"last {lookback_days} daily sessions → ~{lookback_days//5} weeks"
                      if lookback_days > 0 and timeframe == "weekly"
                      else f"last {lookback_days} daily sessions"
                      if lookback_days > 0 else "full history")
    tf_badge       = "Weekly candles" if timeframe == "weekly" else "Daily candles"
    return HTML_TEMPLATE.format(
        title          = title,
        group_label    = group_label,
        gen_date       = date.today().strftime("%d %b %Y"),
        total_stocks   = len(df),
        avg_candles    = avg_candles,
        candle_label   = candle_label,
        lookback_label = lookback_label,
        tf_badge       = tf_badge,
        rows           = build_rows(df),
    )


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="NSE Volume Profile Scanner v4")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS,
                   help=f"Price buckets (default {DEFAULT_BINS})")
    p.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK,
                   help=f"Most recent N daily rows per stock before aggregation "
                        f"(default {DEFAULT_LOOKBACK} ≈ 5 years). Use 0 for full history.")
    p.add_argument("--timeframe", choices=["daily", "weekly"], default=DEFAULT_TIMEFRAME,
                   help="Candle timeframe for volume profile (default: daily)")
    p.add_argument("--csv", action="store_true",
                   help="Also save results as CSV files")
    return p.parse_args()


def main():
    args = parse_args()

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data dir not found: {DATA_DIR.resolve()}")
    if not NIFTY750.exists():
        raise FileNotFoundError(f"Nifty750 file not found: {NIFTY750.resolve()}")

    n750   = load_nifty750(NIFTY750)
    print(f"  Nifty750 : {len(n750)} symbols loaded from {NIFTY750.resolve()}")

    df_all = scan_directory(bins=args.bins, lookback_days=args.lookback_days, timeframe=args.timeframe)
    if df_all.empty:
        print("  No results."); return

    df_n750   = df_all[df_all["Symbol"].isin(n750)].copy()
    df_others = df_all[~df_all["Symbol"].isin(n750)].copy()

    print(f"  Nifty750 matched : {len(df_n750)}")
    print(f"  Others           : {len(df_others)}\n")

    tf = args.timeframe                                      # "daily" | "weekly"
    for df_g, title, label, fname in [
        (df_n750,   f"NSE Volume Profile — Nifty 750 ({tf.capitalize()})",
                    f"Nifty 750 · {tf.capitalize()}",
                    f"nifty750_{tf}_vp.html"),
        (df_others, f"NSE Volume Profile — Other Stocks ({tf.capitalize()})",
                    f"Other Stocks · {tf.capitalize()}",
                    f"others_{tf}_vp.html"),
    ]:
        if df_g.empty:
            print(f"  ⚠  No data for {label}"); continue
        html     = build_html(df_g, title, label,
                              lookback_days=args.lookback_days,
                              timeframe=args.timeframe)
        out_path = OUTPUT_DIR / fname
        out_path.write_text(html, encoding="utf-8")
        size_kb  = out_path.stat().st_size // 1024
        print(f"  ✓ {fname:<28} {len(df_g):>4} stocks  {size_kb} KB")
        if args.csv:
            csv_p = OUTPUT_DIR / fname.replace(".html", ".csv")
            df_g.to_csv(csv_p, index=False)
            print(f"    ↳ CSV saved: {csv_p}")

    print(f"\n  Done. Open nifty750_{tf}_vp.html or others_{tf}_vp.html in your browser.\n")


if __name__ == "__main__":
    main()
