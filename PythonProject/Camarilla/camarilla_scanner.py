#!/usr/bin/env python3
"""
Camarilla Pivot Scanner — TradingView "Auto" timeframe, exact match
====================================================================

SOURCE CANDLE — official TradingView docs
  (https://www.tradingview.com/support/solutions/43000521824)

  Auto timeframe selection:
    chart ≤ 15 min        → Daily   source
    15 min < chart < 1D   → Weekly  source
    chart ≥ 1D (daily+)   → Monthly source  ← scanner uses daily CSVs = 1D chart

  Our scanner = daily CSV data = "1D chart equivalent"
  → Auto must use MONTHLY source (previous complete month's H/L/C)

  Manual overrides also available: Daily / Weekly / Monthly

FORMULA (TradingView official, verified):
  Range = prevHigh − prevLow
  S1 = prevClose − Range × 1.1 / 12
  S2 = prevClose − Range × 1.1 / 6
  S3 = prevClose − Range × 1.1 / 4
  S4 = prevClose − Range × 1.1 / 2
  R1 = prevClose + Range × 1.1 / 12
  R2 = prevClose + Range × 1.1 / 6
  R3 = prevClose + Range × 1.1 / 4
  R4 = prevClose + Range × 1.1 / 2

SOURCE CANDLE DEFINITIONS:
  Daily   → df.iloc[-1]  (last complete trading day in CSV)
  Weekly  → last complete week (Mon–Fri) before today
             H = max of week's daily highs
             L = min of week's daily lows
             C = last trading day close of that week
  Monthly → last complete calendar month before this month
             H = max of month's daily highs
             L = min of month's daily lows
             C = last trading day close of that month
"""

import os, glob, json, argparse, math
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

DATA_DIR    = "../nse_data_cache"
NIFTY_FILE  = "../nifty750.txt"
OUTPUT_HTML = "camarilla_scanner.html"

# ── Camarilla formula (TradingView-official) ───────────────────────────────────
def cam(h: float, l: float, c: float) -> dict:
    r = h - l
    return {
        "S1": c - r * 1.1 / 12,
        "S2": c - r * 1.1 / 6,
        "S3": c - r * 1.1 / 4,
        "S4": c - r * 1.1 / 2,
        "R1": c + r * 1.1 / 12,
        "R2": c + r * 1.1 / 6,
        "R3": c + r * 1.1 / 4,
        "R4": c + r * 1.1 / 2,
    }

def rounded_cam(h, l, c):
    p = cam(h, l, c)
    return {k: round(v, 2) for k, v in p.items()}

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_nifty750(f: str) -> set:
    try:
        return {ln.strip().upper() for ln in open(f) if ln.strip()}
    except FileNotFoundError:
        return set()

def load_csv(fp: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(fp)
        col = next((c for c in df.columns if c.lower() in ("datetime","date")), None)
        if col is None: return None
        df.rename(columns={col: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for c in ("Open","High","Low","Close","Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["High","Low","Close"], inplace=True)
        return df
    except Exception:
        return None

def prev_week_src(df: pd.DataFrame, today: pd.Timestamp) -> dict | None:
    """
    Previous COMPLETE week (Mon–Fri) before today.
    Aggregation: H=max, L=min, C=last day close.
    Matches TradingView 'Use Daily-based Values = ON'.
    """
    # Start of the current week (Monday)
    curr_week_start = today - pd.Timedelta(days=today.dayofweek)
    past = df[df["Date"] < curr_week_start]
    if len(past) == 0:
        return None
    # Previous week = Mon–Sun range ending on the most recent Friday
    prev_week_end   = curr_week_start - pd.Timedelta(days=1)          # last Sunday
    prev_week_start = prev_week_end  - pd.Timedelta(days=6)           # previous Monday
    wk = past[(past["Date"] >= prev_week_start) & (past["Date"] <= prev_week_end)]
    if len(wk) == 0:
        # Fallback: take the last 5 trading days before this week
        wk = past.tail(5)
    return {
        "src_high":  round(float(wk["High"].max()),   2),
        "src_low":   round(float(wk["Low"].min()),    2),
        "src_close": round(float(wk["Close"].iloc[-1]), 2),
        "src_date":  str(wk["Date"].iloc[-1].date()),
        "src_tf":    "Weekly",
    }

def prev_month_src(df: pd.DataFrame, today: pd.Timestamp) -> dict | None:
    """
    Previous COMPLETE calendar month.
    Aggregation: H=max, L=min, C=last trading day close.
    Matches TradingView 'Use Daily-based Values = ON' for monthly source.
    This is what TradingView Auto uses on a Daily chart.
    """
    first_of_month = today.replace(day=1)
    prev = df[df["Date"] < first_of_month]
    if len(prev) == 0:
        return None
    # All rows in the previous calendar month
    prev_month_end   = first_of_month - pd.Timedelta(days=1)
    prev_month_start = prev_month_end.replace(day=1)
    mo = prev[(prev["Date"] >= prev_month_start) & (prev["Date"] <= prev_month_end)]
    if len(mo) == 0:
        mo = prev.tail(22)   # fallback: ~1 month of trading days
    return {
        "src_high":  round(float(mo["High"].max()),    2),
        "src_low":   round(float(mo["Low"].min()),     2),
        "src_close": round(float(mo["Close"].iloc[-1]), 2),
        "src_date":  str(mo["Date"].iloc[-1].date()),
        "src_tf":    "Monthly",
    }

def prev_day_src(df: pd.DataFrame) -> dict | None:
    if len(df) < 1: return None
    last = df.iloc[-1]
    return {
        "src_high":  round(float(last["High"]),  2),
        "src_low":   round(float(last["Low"]),   2),
        "src_close": round(float(last["Close"]), 2),
        "src_date":  str(last["Date"].date()),
        "src_tf":    "Daily",
    }

def build_src(src_info: dict) -> dict:
    """Attach computed Camarilla levels to a source-candle dict."""
    p = rounded_cam(src_info["src_high"], src_info["src_low"], src_info["src_close"])
    return {**src_info, **p}

# ── Per-stock precomputation ───────────────────────────────────────────────────
def precompute(filepath: str, nifty750: set) -> dict | None:
    symbol = Path(filepath).stem.upper()
    df     = load_csv(filepath)
    if df is None or len(df) < 2:
        return None

    today = pd.Timestamp(datetime.now().date())

    # ── Daily source ───────────────────────────────────────────────────────────
    d_src = prev_day_src(df)
    if d_src is None: return None

    # ── Weekly source ──────────────────────────────────────────────────────────
    w_src = prev_week_src(df, today) or d_src

    # ── Monthly source (TradingView Auto on daily chart) ───────────────────────
    m_src = prev_month_src(df, today) or w_src

    current_price = round(float(df.iloc[-1]["Close"]), 2)
    last_date     = str(df.iloc[-1]["Date"].date())
    avg_vol = int(df["Volume"].tail(20).mean()) if len(df) >= 20 else int(df["Volume"].mean())

    return {
        "sym":   symbol,
        "n750":  symbol in nifty750,
        "price": current_price,
        "date":  last_date,
        "d":     build_src(d_src),
        "w":     build_src(w_src),
        "m":     build_src(m_src),   # ← used by Auto (TradingView daily chart)
        "avol":  avg_vol,
    }

def build_dataset(data_dir: str, nifty_file: str) -> list[dict]:
    nifty750  = load_nifty750(nifty_file)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    results   = []
    for i, fp in enumerate(csv_files, 1):
        if i % 100 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}]…", end="\r")
        rec = precompute(fp, nifty750)
        if rec: results.append(rec)
    print(f"\n  Done — {len(results)} stocks loaded")
    return results

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Camarilla Pivot Scanner — NSE</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#090c12;--surf:#0f1420;--brd:#1a2235;
  --acc:#00e5a0;--acc2:#3d9eff;--warn:#ff8c42;--red:#ff4d6a;
  --txt:#cdd5e8;--muted:#4e5f7a;--hdr:#07090f;
  --mono:'IBM Plex Mono',monospace;--sans:'Space Grotesk',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;scrollbar-color:var(--acc) var(--brd);scrollbar-width:thin}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);overflow-y:scroll}

/* ─ topbar ─ */
.tb{background:var(--hdr);border-bottom:1px solid var(--brd);padding:12px 24px;
  display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:300}
.logo{font-family:var(--mono);font-size:11px;letter-spacing:3px;color:var(--acc);text-transform:uppercase;white-space:nowrap}
.fill{flex:1}
.bd{display:inline-flex;align-items:center;gap:5px;font-family:var(--mono);font-size:10px;
  padding:3px 10px;border-radius:3px;letter-spacing:.8px;white-space:nowrap}
.bg{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.25);color:var(--acc)}
.bb{background:rgba(61,158,255,.08);border:1px solid rgba(61,158,255,.25);color:var(--acc2)}
.bw{background:rgba(255,140,66,.08);border:1px solid rgba(255,140,66,.25);color:var(--warn)}

/* ─ controls ─ */
.ctrl{padding:16px 24px 0;display:flex;flex-wrap:wrap;gap:10px;align-items:flex-end}
.cg{display:flex;flex-direction:column;gap:4px}
.cg label{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-family:var(--mono)}
select{background:var(--surf);color:var(--txt);border:1px solid var(--brd);border-radius:4px;
  padding:7px 26px 7px 10px;font-family:var(--mono);font-size:12px;cursor:pointer;outline:none;
  -webkit-appearance:none;appearance:none;min-width:130px;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='9' fill='%234e5f7a' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 8px center;transition:border-color .2s}
select:hover,select:focus{border-color:var(--acc)}
.btn{background:var(--acc);color:#000;border:none;border-radius:4px;padding:7px 18px;
  font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
  cursor:pointer;transition:opacity .15s,transform .1s;align-self:flex-end;white-space:nowrap}
.btn:hover{opacity:.82;transform:translateY(-1px)}
.btn:active{transform:translateY(0)}

/* ─ source info bar ─ */
.ibar{margin:12px 24px 0;border-radius:6px;padding:10px 16px;
  background:rgba(61,158,255,.05);border:1px solid rgba(61,158,255,.15);
  font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.9}
.ibar .hl{color:var(--acc)}.ibar .bl{color:var(--acc2)}.ibar b{color:var(--txt)}

/* ─ stats ─ */
.stats{padding:12px 24px 0;display:flex;gap:10px;flex-wrap:wrap}
.st{background:var(--surf);border:1px solid var(--brd);border-radius:6px;padding:9px 16px;min-width:100px}
.sl{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-family:var(--mono)}
.sv{font-family:var(--mono);font-size:19px;font-weight:700;color:var(--acc);margin-top:2px}
.sv.sm{font-size:11px;margin-top:5px;color:var(--txt);font-weight:400}

/* ─ search row ─ */
.sr{padding:10px 24px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.sr input[type=text]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:7px 11px;font-family:var(--mono);font-size:12px;outline:none;
  width:190px;transition:border-color .2s}
.sr input[type=text]:focus{border-color:var(--acc2)}
.cbl{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);cursor:pointer;user-select:none}
.cbl input{accent-color:var(--acc);width:13px;height:13px;cursor:pointer}
.cnt{background:rgba(61,158,255,.1);border:1px solid rgba(61,158,255,.2);color:var(--acc2);
  font-family:var(--mono);font-size:10px;padding:2px 9px;border-radius:3px;margin-left:auto}

/* ─ table ─ */
.tw{padding:10px 24px 0}
.ts{overflow-x:auto;overflow-y:auto;max-height:60vh;border:1px solid var(--brd);border-radius:8px;
  scrollbar-color:var(--acc2) var(--brd);scrollbar-width:thin}
table{border-collapse:collapse;width:max-content;min-width:100%;font-family:var(--mono);font-size:11.5px}
thead{position:sticky;top:0;z-index:10}
thead tr{background:var(--hdr);border-bottom:2px solid var(--acc)}
th{padding:9px 13px;text-align:left;font-size:9px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--muted);cursor:pointer;white-space:nowrap;user-select:none;border-right:1px solid var(--brd);
  transition:color .15s}
th:last-child{border-right:none}
th:hover{color:var(--acc)}
th.asc::after{content:" ▲";color:var(--acc);font-size:8px}
th.desc::after{content:" ▼";color:var(--acc);font-size:8px}
th.active-lv{color:var(--acc2) !important}
tbody tr{border-bottom:1px solid rgba(26,34,53,.8);transition:background .1s}
tbody tr:hover{background:#0d1525}
tbody tr.nr td:first-child{border-left:2px solid var(--acc)}
td{padding:7px 13px;white-space:nowrap;border-right:1px solid rgba(26,34,53,.6)}
td:last-child{border-right:none}
/* cell styles */
.csym{font-weight:700;color:#e8eef8;font-size:12.5px}
.cpr{color:#e8eef8;font-weight:600}
.clv{color:var(--acc2);font-weight:700}
.pos{color:#00e5a0}.neg{color:#ff4d6a}
.cs1{color:#b8daff}.cs2{color:#80bcff}.cs3{color:#4da0ff}.cs4{color:var(--warn)}
.cr1{color:#b8ffe4}.cr2{color:#7dffcc}.cr3{color:#33ffb2}.cr4{color:var(--red)}
.mu{color:var(--muted)}
.chk{color:var(--acc)}
/* divider columns */
th.dv,td.dv{background:rgba(61,158,255,.04);border-left:1px solid rgba(61,158,255,.12);
  border-right:1px solid rgba(61,158,255,.12);font-size:9px;color:var(--muted);text-align:center;
  padding:7px 5px;letter-spacing:.5px;cursor:default}
.nodata{padding:70px;text-align:center;color:var(--muted);font-family:var(--mono);font-size:13px;line-height:2}
/* footer */
.ft{padding:16px 24px 4px;font-size:10px;color:var(--muted);font-family:var(--mono);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;
  border-top:1px solid var(--brd);margin-top:12px}
</style>
</head>
<body>

<div class="tb">
  <div class="logo">⬡ Camarilla Scanner · NSE</div>
  <div class="fill"></div>
  <span class="bd bg" id="b-lv">S3 SUPPORT</span>
  <span class="bd bb" id="b-src">AUTO · MONTHLY</span>
  <span class="bd bw" id="b-pr">±2 %</span>
</div>

<div class="ctrl">
  <div class="cg">
    <label>Level</label>
    <select id="sl">
      <optgroup label="─ Support ─">
        <option value="S1">S1 · Mild support</option>
        <option value="S2">S2 · Minor support</option>
        <option value="S3" selected>S3 · Key reversal support</option>
        <option value="S4">S4 · Breakdown level</option>
      </optgroup>
      <optgroup label="─ Resistance ─">
        <option value="R1">R1 · Mild resistance</option>
        <option value="R2">R2 · Minor resistance</option>
        <option value="R3">R3 · Key reversal resistance</option>
        <option value="R4">R4 · Breakout level</option>
      </optgroup>
    </select>
  </div>

  <div class="cg">
    <label>Pivot Timeframe</label>
    <select id="st">
      <option value="m" selected>Auto (Monthly) — TradingView default on daily chart</option>
      <option value="w">Weekly — TradingView default on 30min–4hr chart</option>
      <option value="d">Daily — TradingView default on ≤15min chart</option>
    </select>
  </div>

  <div class="cg">
    <label>Proximity ±%</label>
    <select id="sp">
      <option value="0.5">±0.5 %</option>
      <option value="1">±1 %</option>
      <option value="2" selected>±2 %</option>
      <option value="3">±3 %</option>
      <option value="5">±5 %</option>
      <option value="10">±10 %</option>
    </select>
  </div>

  <button class="btn" onclick="scan()">▶ SCAN</button>
</div>

<div class="ibar" id="ibar">
  Loading…
</div>

<div class="stats">
  <div class="st"><div class="sl">Hits</div><div class="sv" id="st-h">—</div></div>
  <div class="st"><div class="sl">Nifty 750</div><div class="sv" id="st-n">—</div></div>
  <div class="st"><div class="sl">Others</div><div class="sv" id="st-o">—</div></div>
  <div class="st"><div class="sl">Universe</div><div class="sv" id="st-u">—</div></div>
  <div class="st"><div class="sl">Data generated</div><div class="sv sm">__GT__</div></div>
</div>

<div class="sr">
  <input type="text" id="q" placeholder="Search symbol…" oninput="render()">
  <label class="cbl"><input type="checkbox" id="cb7" onchange="render()"> Nifty 750 only</label>
  <span class="cnt" id="vc">— rows</span>
</div>

<div class="tw"><div class="ts" id="ts"></div></div>

<div class="ft">
  <span>Camarilla: Sn/Rn = prevClose ∓/± (prevH−prevL)×1.1/k · k: S1/R1=12, S2/R2=6, S3/R3=4, S4/R4=2 · Source: TradingView official docs</span>
  <span>Data dir: __DD__ · __GT__</span>
</div>

<script>
const S = __JSON__;   // all stocks, pre-computed

// ── Source-candle accessor ────────────────────────────────────────────────────
function srcOf(s, tf) {
  // tf: 'm'=Monthly (Auto on daily chart), 'w'=Weekly, 'd'=Daily
  return tf === 'w' ? s.w : tf === 'd' ? s.d : s.m;
}

// ── Camarilla (mirrors Python exactly) ───────────────────────────────────────
function cam(h, l, c) {
  const r = h - l;
  return {
    S1: c - r*1.1/12, S2: c - r*1.1/6,  S3: c - r*1.1/4,  S4: c - r*1.1/2,
    R1: c + r*1.1/12, R2: c + r*1.1/6,  R3: c + r*1.1/4,  R4: c + r*1.1/2,
  };
}

// ── State ─────────────────────────────────────────────────────────────────────
let rows = [], sc = 4, sd = 1;

// ── Columns ───────────────────────────────────────────────────────────────────
function cols(lv) {
  return [
    { k:'sym',       h:'Symbol',      fn: r=>`<td class="csym">${r.sym}</td>` },
    { k:'n750',      h:'N750',        fn: r=>`<td class="${r.n750?'chk':'mu'}">${r.n750?'✓':'–'}</td>` },
    { k:'price',     h:'Last Close',  fn: r=>`<td class="cpr">${r.price.toFixed(2)}</td>` },
    { k:'lvval',     h:lv+' Level',   fn: r=>`<td class="clv">${r.lv.toFixed(2)}</td>`, hcls:'active-lv' },
    { k:'dist',      h:'Dist %',      fn: r=>`<td class="${r.dist>=0?'pos':'neg'}">${r.dist>=0?'+':''}${r.dist.toFixed(2)}%</td>` },
    { k:'_d1', h:'S1─S4 / R1─R4', fn:r=>`<td class="dv">◀▶</td>`, dv:true },
    { k:'s1',        h:'S1',          fn: r=>`<td class="cs1">${r.s1.toFixed(2)}</td>` },
    { k:'s2',        h:'S2',          fn: r=>`<td class="cs2">${r.s2.toFixed(2)}</td>` },
    { k:'s3',        h:'S3',          fn: r=>`<td class="cs3">${r.s3.toFixed(2)}</td>` },
    { k:'s4',        h:'S4',          fn: r=>`<td class="cs4">${r.s4.toFixed(2)}</td>` },
    { k:'r1',        h:'R1',          fn: r=>`<td class="cr1">${r.r1.toFixed(2)}</td>` },
    { k:'r2',        h:'R2',          fn: r=>`<td class="cr2">${r.r2.toFixed(2)}</td>` },
    { k:'r3',        h:'R3',          fn: r=>`<td class="cr3">${r.r3.toFixed(2)}</td>` },
    { k:'r4',        h:'R4',          fn: r=>`<td class="cr4">${r.r4.toFixed(2)}</td>` },
    { k:'_d2', h:'Pivot Source', fn:r=>`<td class="dv">◀▶</td>`, dv:true },
    { k:'sh',        h:'Src High',    fn: r=>`<td class="mu">${r.sh.toFixed(2)}</td>` },
    { k:'sl2',       h:'Src Low',     fn: r=>`<td class="mu">${r.sl.toFixed(2)}</td>` },
    { k:'sc2',       h:'Src Close',   fn: r=>`<td class="mu">${r.sc.toFixed(2)}</td>` },
    { k:'sd2',       h:'Src Date',    fn: r=>`<td class="mu">${r.sd}</td>` },
    { k:'avol',      h:'AvgVol(20)',  fn: r=>`<td class="mu">${r.avol.toLocaleString()}</td>` },
    { k:'date',      h:'Last Date',   fn: r=>`<td class="mu">${r.date}</td>` },
  ];
}

// ── Scan ──────────────────────────────────────────────────────────────────────
function scan() {
  const lv   = document.getElementById('sl').value;
  const tf   = document.getElementById('st').value;
  const prox = parseFloat(document.getElementById('sp').value);

  const isR  = lv.startsWith('R');
  const tfLbl = tf==='m' ? 'AUTO · MONTHLY' : tf==='w' ? 'WEEKLY' : 'DAILY';
  const srcDescMap = {
    m: 'Auto (Daily chart) → <b>Monthly source</b>: previous complete month\'s H/L/C — matches TradingView "Pivot Timeframe = Auto" on a daily chart',
    w: 'Weekly source: previous complete week\'s H/L/C — matches TradingView "Pivot Timeframe = Weekly"',
    d: 'Daily source: last completed trading session\'s H/L/C — matches TradingView "Pivot Timeframe = Daily"',
  };
  const kMap = {S1:12,S2:6,S3:4,S4:2,R1:12,R2:6,R3:4,R4:2};

  document.getElementById('b-lv').textContent  = lv + (isR?' RESISTANCE':' SUPPORT');
  document.getElementById('b-lv').className    = 'bd ' + (isR?'bb':'bg');
  document.getElementById('b-src').textContent = tfLbl;
  document.getElementById('b-pr').textContent  = '±' + prox + ' %';
  document.getElementById('ibar').innerHTML =
    `<span class="bl">Source</span> · ${srcDescMap[tf]}<br>` +
    `<span class="hl">Formula</span> · ${lv} = prevClose ${isR?'+':'−'} (prevHigh−prevLow) × 1.1 / <b>${kMap[lv]}</b>`;

  rows = [];
  for (const s of S) {
    const src = srcOf(s, tf);
    // Recompute from source H/L/C (same formula as Python — ensures JS values = TV values)
    const p   = cam(src.src_high, src.src_low, src.src_close);
    const tgt = p[lv];
    if (!tgt) continue;

    const dist = (s.price - tgt) / Math.abs(tgt) * 100;
    if (Math.abs(dist) > prox) continue;

    rows.push({
      sym:s.sym, n750:s.n750, price:s.price, date:s.date, avol:s.avol,
      lv: tgt, dist: +dist.toFixed(3),
      s1:p.S1, s2:p.S2, s3:p.S3, s4:p.S4,
      r1:p.R1, r2:p.R2, r3:p.R3, r4:p.R4,
      sh:src.src_high, sl:src.src_low, sc:src.src_close, sd:src.src_date,
    });
  }

  sc=4; sd=1;
  rows.sort((a,b)=>Math.abs(a.dist)-Math.abs(b.dist));
  render();
}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {
  const lv    = document.getElementById('sl').value;
  const q     = document.getElementById('q').value.trim().toUpperCase();
  const on750 = document.getElementById('cb7').checked;
  const cs    = cols(lv);

  const vis = rows.filter(r=>{
    if (q && !r.sym.includes(q)) return false;
    if (on750 && !r.n750) return false;
    return true;
  });

  document.getElementById('st-h').textContent = vis.length;
  document.getElementById('st-n').textContent = vis.filter(r=>r.n750).length;
  document.getElementById('st-o').textContent = vis.filter(r=>!r.n750).length;
  document.getElementById('st-u').textContent = S.length;
  document.getElementById('vc').textContent   = vis.length + ' rows';

  if (vis.length === 0) {
    document.getElementById('ts').innerHTML =
      '<div class="nodata">No stocks found near this level.<br>Try increasing Proximity % or switching the Pivot Timeframe.</div>';
    return;
  }

  const ths = cs.map((c,i)=>{
    let cls = c.hcls || '';
    if (!c.dv && i===sc) cls += (sd===1?' asc':' desc');
    const di = c.dv ? '' : ` data-i="${i}"`;
    return `<th class="${cls.trim()}"${di}>${c.h}</th>`;
  }).join('');

  const trs = vis.map(r=>{
    const rc = r.n750?' class="nr"':'';
    return `<tr${rc}>${cs.map(c=>c.fn(r)).join('')}</tr>`;
  }).join('');

  document.getElementById('ts').innerHTML =
    `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;

  document.querySelectorAll('#ts th[data-i]').forEach(th=>{
    th.addEventListener('click',()=>{
      const i=+th.dataset.i;
      if(sc===i) sd*=-1; else{sc=i;sd=1;}
      const k=cs[i].k;
      rows.sort((a,b)=>
        (typeof a[k]==='number'?a[k]-b[k]:String(a[k]).localeCompare(String(b[k])))*sd);
      render();
    });
  });
}

document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('st-u').textContent = S.length;
  scan();
});
</script>
</body>
</html>
"""

def build_html(stocks: list[dict], data_dir: str) -> str:
    gt = datetime.now().strftime("%d %b %Y  %H:%M")
    return (HTML
            .replace("__JSON__", json.dumps(stocks, separators=(",",":")))
            .replace("__GT__",   gt)
            .replace("__DD__",   data_dir))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  default=DATA_DIR)
    ap.add_argument("--nifty", default=NIFTY_FILE)
    ap.add_argument("--out",   default=OUTPUT_HTML)
    a = ap.parse_args()
    print(f"[*] Data : {a.data}")
    stocks = build_dataset(a.data, a.nifty)
    if not stocks:
        print("[!] No stocks loaded"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Output: {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
