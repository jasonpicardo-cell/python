#!/usr/bin/env python3
"""
Pivot Point Scanner — NSE  (full feature set)
Formulas: TradingView official (https://www.tradingview.com/support/solutions/43000521824)

Features
--------
1. 200 DMA trend filter     — above/below 200-day SMA
2. Historical bounce count  — monthly touches of selected pivot level (6M / 12M)
3. Nifty index tier filter  — N50 / N100 / N200 / N500 / N750 / All
4. Multi-level scan         — Any Support / Any Resistance / Any Level
5. 52-week H/L proximity    — distance from 52W high and low
6. RS Rating (0–99)         — percentile rank of 12-month price return vs all stocks
"""

import os, glob, json, argparse, math
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR    = "../nse_data_cache"
N50_FILE    = "../nifty50.txt"
N100_FILE   = "../nifty100.txt"
N200_FILE   = "../nifty200.txt"
N500_FILE   = "../nifty500.txt"
N750_FILE   = "../nifty750.txt"
OUTPUT_HTML = "pivot_scanner.html"
HIST_MONTHS = 25          # monthly candles stored per stock for bounce calc

# ── Loaders ───────────────────────────────────────────────────────────────────
def load_set(fp):
    try:    return {ln.strip().upper() for ln in open(fp) if ln.strip()}
    except: return set()

def load_index_map(files):
    """Return {symbol: lowest_index_tier} — 50 < 100 < 200 < 500 < 750 < 0=other."""
    m = {}
    for tier, fp in sorted(files.items()):   # ascending: 50 first
        for sym in load_set(fp):
            if sym not in m:
                m[sym] = tier
    return m

def load_csv(fp):
    try:
        df = pd.read_csv(fp)
        col = next((c for c in df.columns if c.lower() in ("datetime","date")), None)
        if col is None: return None
        df.rename(columns={col:"Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        for c in ("Open","High","Low","Close","Volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["High","Low","Close"], inplace=True)
        return df
    except: return None

def r2(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 0.0
    return round(float(v), 2)

# ── Source-candle builders ────────────────────────────────────────────────────
def daily_src(df):
    if len(df) < 1: return None
    row = df.iloc[-1]
    return dict(pH=r2(row["High"]), pL=r2(row["Low"]),
                pC=r2(row["Close"]), pO=r2(row.get("Open", row["Close"])),
                cO=r2(row.get("Open", row["Close"])), dt=str(row["Date"].date()))

def weekly_src(df, today):
    """Last COMPLETED week (Mon-Fri) using W-FRI resample — robust against Monday holidays."""
    wsm  = today - pd.Timedelta(days=today.dayofweek)
    past = df[df["Date"] < wsm]
    if not len(past): return None
    weekly = (past.set_index("Date")
                  .resample("W-FRI")
                  .agg(pH=("High","max"), pL=("Low","min"),
                       pC=("Close","last"), pO=("Open","first"),
                       _n=("Close","count"))
                  .query("_n > 0").drop(columns=["_n"]).dropna().reset_index())
    if not len(weekly): return None
    pw  = weekly.iloc[-1]
    cur = df[df["Date"] >= wsm]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(pw.pC))
    return dict(pH=r2(float(pw.pH)), pL=r2(float(pw.pL)),
                pC=r2(float(pw.pC)), pO=r2(float(pw.pO)),
                cO=cO, dt=str(pw.Date.date()))

def monthly_src(df, today):
    """Last COMPLETED calendar month."""
    fom  = today.replace(day=1)
    prev = df[df["Date"] < fom]
    if not len(prev): return None
    me = fom - pd.Timedelta(days=1); ms = me.replace(day=1)
    mo = prev[(prev["Date"] >= ms) & (prev["Date"] <= me)]
    if not len(mo): mo = prev.tail(22)
    cur = df[df["Date"] >= fom]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(mo["Close"].iloc[-1]))
    return dict(pH=r2(float(mo["High"].max())), pL=r2(float(mo["Low"].min())),
                pC=r2(float(mo["Close"].iloc[-1])), pO=r2(float(mo["Open"].iloc[0])),
                cO=cO, dt=str(mo["Date"].iloc[-1].date()))

def quarterly_src(df, today):
    """Last COMPLETED calendar quarter (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec)."""
    q_start_month = ((today.month - 1) // 3) * 3 + 1
    curr_q_start  = today.replace(month=q_start_month, day=1)
    prev_q_end    = curr_q_start - pd.Timedelta(days=1)
    prev_q_start  = prev_q_end.replace(month=((prev_q_end.month - 1) // 3) * 3 + 1, day=1)
    qr = df[(df["Date"] >= prev_q_start) & (df["Date"] <= prev_q_end)]
    if not len(qr): return None
    cur = df[df["Date"] >= curr_q_start]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(qr["Close"].iloc[-1]))
    return dict(pH=r2(float(qr["High"].max())), pL=r2(float(qr["Low"].min())),
                pC=r2(float(qr["Close"].iloc[-1])), pO=r2(float(qr["Open"].iloc[0])),
                cO=cO, dt=str(qr["Date"].iloc[-1].date()))

def yearly_src(df, today):
    """Last COMPLETED calendar year."""
    curr_yr_start = today.replace(month=1, day=1)
    prev_yr_end   = curr_yr_start - pd.Timedelta(days=1)
    prev_yr_start = prev_yr_end.replace(month=1, day=1)
    yr = df[(df["Date"] >= prev_yr_start) & (df["Date"] <= prev_yr_end)]
    if not len(yr): return None
    cur = df[df["Date"] >= curr_yr_start]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(yr["Close"].iloc[-1]))
    return dict(pH=r2(float(yr["High"].max())), pL=r2(float(yr["Low"].min())),
                pC=r2(float(yr["Close"].iloc[-1])), pO=r2(float(yr["Open"].iloc[0])),
                cO=cO, dt=str(yr["Date"].iloc[-1].date()))

# ── Historical candles for JS bounce count ────────────────────────────────────
# Compact [pH, pL, pC, pO] arrays.  whist=weekly, mhist=monthly, qhist=quarterly.
def build_hists(df, today, n_months=HIST_MONTHS, n_weeks=60, n_quarters=12):
    fom = today.replace(day=1)
    wsm = today - pd.Timedelta(days=today.dayofweek)

    def _resamp(past, rule, extra_filter=""):
        r = (past.set_index("Date").resample(rule)
                 .agg(pH=("High","max"), pL=("Low","min"),
                      pC=("Close","last"), pO=("Open","first"),
                      _n=("Close","count"))
                 .dropna())
        if extra_filter: r = r.query(extra_filter)
        return r.drop(columns=["_n"]).reset_index()

    def _rows(df_r, n):
        return [[r2(float(row.pH)),r2(float(row.pL)),
                 r2(float(row.pC)),r2(float(row.pO))]
                for _,row in df_r.tail(n).iterrows()]

    past_m = df[df["Date"] < fom]
    mhist  = _rows(_resamp(past_m, "MS"), n_months) if len(past_m) else []

    past_w = df[df["Date"] < wsm]
    whist  = _rows(_resamp(past_w, "W-FRI", "_n > 0"), n_weeks) if len(past_w) else []

    # Quarterly: resample to quarter-start ("QS")
    qhist  = _rows(_resamp(past_m, "QS"), n_quarters) if len(past_m) else []

    return mhist, whist, qhist

# ── Per-stock stats ───────────────────────────────────────────────────────────
def stock_stats(df, today):
    last_close = float(df.iloc[-1]["Close"])

    # 200 DMA
    dma200 = float(df["Close"].tail(200).mean()) if len(df) >= 50 else last_close
    above200 = last_close > dma200

    # 52-week H/L
    yr = df[df["Date"] >= (today - pd.Timedelta(days=365))]
    w52h = float(yr["High"].max())  if len(yr) else float(df["High"].max())
    w52l = float(yr["Low"].min())   if len(yr) else float(df["Low"].min())

    # 12-month return (for RS rating, computed later)
    yr_ago = today - pd.Timedelta(days=365)
    past_yr = df[df["Date"] <= yr_ago]
    price_1y = float(past_yr.iloc[-1]["Close"]) if len(past_yr) else last_close
    ret12m = round((last_close - price_1y) / price_1y * 100, 2) if price_1y else 0.0

    # Avg volume
    avol = int(df["Volume"].tail(20).mean()) if len(df) >= 20 else int(df["Volume"].mean())

    return dict(dma200=r2(dma200), above200=above200,
                w52h=r2(w52h), w52l=r2(w52l),
                ret12m=ret12m, avol=avol)

# ── Per-stock precompute ──────────────────────────────────────────────────────
def precompute(fp, idx_map):
    sym = Path(fp).stem.upper()
    df  = load_csv(fp)
    if df is None or len(df) < 2: return None
    today = pd.Timestamp(datetime.now().date())
    ds = daily_src(df)
    if ds is None: return None
    ws = weekly_src(df, today)    or ds
    ms = monthly_src(df, today)   or ws
    qs = quarterly_src(df, today) or ms
    ys = yearly_src(df, today)    or qs
    st = stock_stats(df, today)
    mhist, whist, qhist = build_hists(df, today)
    return dict(sym=sym, idx=idx_map.get(sym, 0),
                price=r2(float(df.iloc[-1]["Close"])),
                date=str(df.iloc[-1]["Date"].date()),
                d=ds, w=ws, m=ms, q=qs, y=ys,
                mhist=mhist, whist=whist, qhist=qhist,
                **st, rs=0)

# ── RS Rating — two-pass percentile rank ─────────────────────────────────────
def assign_rs(stocks):
    rets = [s["ret12m"] for s in stocks]
    rets_s = sorted(rets)
    n = len(rets_s)
    for s in stocks:
        rank = sum(1 for x in rets_s if x < s["ret12m"])
        s["rs"] = round(rank / n * 99) if n else 0

# ── Build full dataset ────────────────────────────────────────────────────────
def build_dataset(data_dir, index_files):
    idx_map   = load_index_map(index_files)
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    stocks    = []
    for i, fp in enumerate(csv_files, 1):
        if i % 100 == 0 or i == len(csv_files):
            print(f"  [{i}/{len(csv_files)}]…", end="\r")
        rec = precompute(fp, idx_map)
        if rec: stocks.append(rec)
    print(f"\n  Done — {len(stocks)} stocks loaded")
    assign_rs(stocks)
    return stocks

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pivot Point Scanner — NSE</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080b10;--surf:#0d1220;--brd:#182034;
  --acc:#00e5a0;--a2:#3b9eff;--a3:#c47aff;
  --warn:#ff8c42;--red:#ff4060;--gold:#f5c518;
  --txt:#cdd5e8;--mu:#4a5c78;--hdr:#060810;
  --mono:'IBM Plex Mono',monospace;--sans:'Space Grotesk',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);overflow-y:scroll}

/* topbar */
.tb{background:var(--hdr);border-bottom:1px solid var(--brd);padding:11px 22px;
  display:flex;align-items:center;gap:9px;position:sticky;top:0;z-index:300}
.logo{font-family:var(--mono);font-size:11px;letter-spacing:3px;color:var(--acc);text-transform:uppercase;white-space:nowrap}
.fill{flex:1}
.bd{font-family:var(--mono);font-size:9px;padding:3px 9px;border-radius:3px;letter-spacing:.8px;white-space:nowrap}
.ba{background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.22);color:var(--acc)}
.bb{background:rgba(59,158,255,.07);border:1px solid rgba(59,158,255,.22);color:var(--a2)}
.bc{background:rgba(196,122,255,.07);border:1px solid rgba(196,122,255,.22);color:var(--a3)}
.bw{background:rgba(255,140,66,.07);border:1px solid rgba(255,140,66,.22);color:var(--warn)}

/* controls — two rows */
.ctrl{padding:14px 22px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end}
.ctrl-sep{width:100%;height:0;border-top:1px solid var(--brd);margin:6px 0 2px}
.cg{display:flex;flex-direction:column;gap:4px}
.cg label{font-size:9px;letter-spacing:1.8px;text-transform:uppercase;color:var(--mu);font-family:var(--mono)}
select,input[type=number]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:12px;
  cursor:pointer;outline:none;transition:border-color .2s}
select{padding-right:24px;-webkit-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='9' fill='%234a5c78' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 7px center}
select:hover,select:focus,input[type=number]:hover,input[type=number]:focus{border-color:var(--acc)}
input[type=number]{width:70px}
.prange{display:flex;gap:4px;align-items:center}
.prange span{color:var(--mu);font-size:11px}
.btn{background:var(--acc);color:#000;border:none;border-radius:4px;padding:6px 16px;
  font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:1.5px;
  text-transform:uppercase;cursor:pointer;align-self:flex-end;
  transition:opacity .15s,transform .1s;white-space:nowrap}
.btn:hover{opacity:.82;transform:translateY(-1px)}.btn:active{transform:translateY(0)}
.btn-out{background:var(--surf);color:var(--acc);border:1px solid rgba(0,229,160,.4)}
.btn-out:hover{background:rgba(0,229,160,.08)}

/* formula bar */
.fbar{margin:10px 22px 0;padding:8px 13px;border-radius:5px;
  background:rgba(59,158,255,.04);border:1px solid rgba(59,158,255,.12);
  font-family:var(--mono);font-size:10.5px;color:var(--mu);line-height:1.9}
.fbar b{color:var(--txt)}.fbar .ha{color:var(--acc)}.fbar .hb{color:var(--a2)}.fbar code{color:var(--gold);font-size:10px}

/* stats */
.stats{padding:10px 22px 0;display:flex;gap:8px;flex-wrap:wrap}
.st{background:var(--surf);border:1px solid var(--brd);border-radius:6px;padding:7px 13px;min-width:88px}
.sl{font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--mu);font-family:var(--mono)}
.sv{font-family:var(--mono);font-size:17px;font-weight:700;color:var(--acc);margin-top:2px}
.sv.sm{font-size:11px;margin-top:4px;color:var(--txt);font-weight:400}

/* search row */
.sr{padding:9px 22px 0;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.sr input[type=text]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:6px 10px;font-family:var(--mono);font-size:12px;
  outline:none;width:170px;transition:border-color .2s}
.sr input[type=text]:focus{border-color:var(--a2)}
.cbl{display:flex;align-items:center;gap:5px;font-size:11.5px;color:var(--mu);cursor:pointer;user-select:none}
.cbl input{accent-color:var(--acc);width:13px;height:13px;cursor:pointer}
.cnt{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.18);color:var(--a2);
  font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:3px;margin-left:auto}
#saved-lbl{font-family:var(--mono);font-size:9px;color:var(--mu);letter-spacing:1px;margin-left:3px}

/* table */
.tw{padding:9px 22px 0}
.ts{overflow-x:auto;overflow-y:auto;max-height:56vh;border:1px solid var(--brd);
  border-radius:8px;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
table{border-collapse:collapse;width:max-content;min-width:100%;font-family:var(--mono);font-size:11.5px}
thead{position:sticky;top:0;z-index:10}
thead tr{background:var(--hdr);border-bottom:2px solid var(--acc)}
th{padding:8px 11px;text-align:left;font-size:8.5px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--mu);cursor:pointer;white-space:nowrap;user-select:none;
  border-right:1px solid var(--brd);transition:color .15s}
th:last-child{border-right:none}
th:hover{color:var(--acc)}
th.asc::after{content:" ▲";color:var(--acc);font-size:7px}
th.desc::after{content:" ▼";color:var(--acc);font-size:7px}
th.hl{color:var(--a2) !important}
tbody tr{border-bottom:1px solid rgba(24,32,52,.9);transition:background .1s}
tbody tr:hover{background:#0c1422}
tbody tr.nr td:first-child{border-left:2px solid var(--acc)}
td{padding:6px 11px;white-space:nowrap;border-right:1px solid rgba(24,32,52,.6)}
td:last-child{border-right:none}
.csym{font-weight:700;color:#dde5f5;font-size:12.5px}
.csym a{color:inherit;text-decoration:none;display:flex;align-items:center;gap:5px;white-space:nowrap}
.csym a:hover{color:var(--acc)}.csym a:hover .tv-ico{opacity:1}
.tv-ico{opacity:.3;transition:opacity .15s;flex-shrink:0}
.cpr{color:#dde5f5;font-weight:600}.clv{color:var(--a2);font-weight:700}.cp{color:var(--gold)}
.pos{color:var(--acc)}.neg{color:var(--red)}
.cs1{color:#a0c8ff}.cs2{color:#6aaeff}.cs3{color:#3090ff}.cs4{color:var(--warn)}.cs5{color:var(--red)}
.cr1{color:#a0ffe0}.cr2{color:#60ffcc}.cr3{color:#00ffb0}.cr4{color:var(--red)}.cr5{color:#ff0060}
.mu{color:var(--mu)}
th.dv,td.dv{background:rgba(59,158,255,.03);border-left:1px solid rgba(59,158,255,.1);
  border-right:1px solid rgba(59,158,255,.1);font-size:8px;color:var(--mu);
  text-align:center;padding:6px 4px;cursor:default;letter-spacing:.3px}
/* index tier badges */
.ix50{color:var(--gold)}.ix100{color:var(--acc)}.ix200{color:var(--a2)}
.ix500{color:var(--a3)}.ix750{color:#80a0c0}.ix0{color:var(--mu)}
/* RS rating colours */
.rs-hi{color:var(--acc)}.rs-md{color:#80d0a0}.rs-lo{color:var(--warn)}.rs-vl{color:var(--red)}
/* 200 DMA */
.dma-up{color:var(--acc)}.dma-dn{color:var(--red)}
/* bounce badges */
.bc-hi{color:var(--acc);font-weight:700}.bc-md{color:#80d080}.bc-lo{color:var(--mu)}
.nodata{padding:60px;text-align:center;color:var(--mu);font-family:var(--mono);font-size:13px;line-height:2.2}
.ft{padding:12px 22px 4px;font-size:9.5px;color:var(--mu);font-family:var(--mono);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:5px;
  border-top:1px solid var(--brd);margin-top:10px}
</style>
</head>
<body>

<div class="tb">
  <div class="logo">◈ Pivot Point Scanner · NSE</div>
  <div class="fill"></div>
  <span class="bd bc" id="b-type">FIBONACCI</span>
  <span class="bd ba" id="b-lv">S2</span>
  <span class="bd bb" id="b-src">QUARTERLY</span>
  <span class="bd bw" id="b-pr">±2%</span>
</div>

<!-- ─ Row 1: Pivot controls ─ -->
<div class="ctrl">
  <div class="cg"><label>Pivot Type</label>
    <select id="sp-type" onchange="onTypeChange()">
      <option value="camarilla">Camarilla</option>
      <option value="traditional">Traditional</option>
      <option value="classic">Classic</option>
      <option value="fibonacci" selected>Fibonacci</option>
      <option value="woodie">Woodie</option>
      <option value="dm">DeMark (DM)</option>
      <option value="floor">Floor</option>
    </select>
  </div>
  <div class="cg"><label>Level</label><select id="sp-lvl" style="min-width:195px"></select></div>
  <div class="cg"><label>Pivot Timeframe</label>
    <select id="sp-tf" style="min-width:315px">
      <option value="d">Daily      · matches TradingView ≤15min chart (Auto)</option>
      <option value="w">Weekly     · matches TradingView 30min–4hr chart (Auto)</option>
      <option value="m">Monthly    · matches TradingView Daily chart (Auto)</option>
      <option value="q" selected>Quarterly  · matches TradingView <b>Weekly chart</b> (Auto) ← prev quarter H/L/C</option>
      <option value="y">Yearly     · matches TradingView Monthly chart (Auto)</option>
    </select>
  </div>
  <div class="cg"><label>Proximity ±%</label>
    <select id="sp-pr">
      <option value="0.5">±0.5%</option><option value="1">±1%</option>
      <option value="2" selected>±2%</option><option value="3">±3%</option>
      <option value="5">±5%</option><option value="10">±10%</option>
    </select>
  </div>

  <div class="ctrl-sep"></div>

  <!-- ─ Row 2: Market filters ─ -->
  <div class="cg"><label>Index</label>
    <select id="idx-flt">
      <option value="0">All stocks</option>
      <option value="50">Nifty 50</option>
      <option value="100">Nifty 100</option>
      <option value="200">Nifty 200</option>
      <option value="500">Nifty 500</option>
      <option value="750" selected>Nifty 750</option>
    </select>
  </div>
  <div class="cg"><label>Trend (200 DMA)</label>
    <select id="dma-flt">
      <option value="any">Any</option>
      <option value="above">Above 200 DMA ↑</option>
      <option value="below">Below 200 DMA ↓</option>
    </select>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="rs-flt">
      <option value="0">Any</option>
      <option value="50">≥ 50</option>
      <option value="70">≥ 70</option>
      <option value="80">≥ 80</option>
      <option value="90">≥ 90</option>
    </select>
  </div>
  <div class="cg"><label>Bounces 12M ≥</label>
    <select id="bc-flt">
      <option value="0">Any</option>
      <option value="1">≥ 1</option>
      <option value="2">≥ 2</option>
      <option value="3">≥ 3</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="pr-min" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="pr-max" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" onclick="scan()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
  <button class="btn btn-out" style="color:var(--mu);border-color:var(--mu)" onclick="resetPrefs()" title="Reset all filters to defaults">↺ RESET</button>
</div>

<div class="fbar" id="fbar">Loading…</div>

<div class="stats">
  <div class="st"><div class="sl">Hits</div><div class="sv" id="st-h">—</div></div>
  <div class="st"><div class="sl">Above 200 DMA</div><div class="sv" id="st-up">—</div></div>
  <div class="st"><div class="sl">Below 200 DMA</div><div class="sv" id="st-dn">—</div></div>
  <div class="st"><div class="sl">RS ≥ 70</div><div class="sv" id="st-rs">—</div></div>
  <div class="st"><div class="sl">Universe</div><div class="sv" id="st-u">—</div></div>
  <div class="st"><div class="sl">Generated</div><div class="sv sm">__GT__</div></div>
</div>

<div class="sr">
  <input type="text" id="q" placeholder="Search symbol…" oninput="render()">
  <label class="cbl"><input type="checkbox" id="cb-vol" onchange="render()"> Above avg vol</label>
  <label class="cbl"><input type="checkbox" id="cb-52h" onchange="render()"> Near 52W High</label>
  <label class="cbl"><input type="checkbox" id="cb-52l" onchange="render()"> Near 52W Low</label>
  <span class="cnt" id="vc">— rows</span>
  <span id="saved-lbl"></span>
</div>

<div class="tw"><div class="ts" id="ts"></div></div>

<div class="ft">
  <span>Formulas: TradingView official · Monthly source = TradingView default for daily charts · RS = percentile rank of 12M return</span>
  <span>Data: __DD__ · __GT__</span>
</div>

<script>
const STOCKS = __JSON__;

// ── Pivot type metadata ───────────────────────────────────────────────────────
const TYPE_META = {
  camarilla:   { label:'Camarilla',   levels:['S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'], hasP:false },
  traditional: { label:'Traditional', levels:['P','S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'], hasP:true },
  classic:     { label:'Classic',     levels:['P','S1','S2','S3','S4','R1','R2','R3','R4'], hasP:true },
  fibonacci:   { label:'Fibonacci',   levels:['P','S1','S2','S3','R1','R2','R3'], hasP:true },
  woodie:      { label:'Woodie',      levels:['P','S1','S2','S3','S4','R1','R2','R3','R4'], hasP:true },
  dm:          { label:'DeMark',      levels:['P','S1','R1'], hasP:true },
  floor:       { label:'Floor',       levels:['P','S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'], hasP:true },
};
const LV_LABEL = {
  P:'P — Pivot',
  S1:'S1',S2:'S2',S3:'S3',S4:'S4',S5:'S5',
  R1:'R1',R2:'R2',R3:'R3',R4:'R4',R5:'R5',
};
const LV_CSS = { P:'cp',S1:'cs1',S2:'cs2',S3:'cs3',S4:'cs4',S5:'cs5',R1:'cr1',R2:'cr2',R3:'cr3',R4:'cr4',R5:'cr5' };
const TYPE_DEF_LV = { fibonacci:'S2',traditional:'S1',classic:'S1',camarilla:'S3',woodie:'S1',dm:'S1',floor:'S1' };

const FBAR_HTML = {
  camarilla:   `<b>Camarilla</b> · <span class="ha">Sn = prevC ∓ Range×1.1/k</span> · k: 1=12,2=6,3=4,4=2 · <code>R5=(prevH/prevL)×prevC</code>`,
  traditional: `<b>Traditional</b> · <code>P=(H+L+C)/3</code> · R1=2P−L, S1=2P−H · up to R5/S5`,
  classic:     `<b>Classic</b> · <code>P=(H+L+C)/3</code> · R3=P+2×Range, S3=P−2×Range · R4=P+3×Range`,
  fibonacci:   `<b>Fibonacci</b> · <code>P=(H+L+C)/3</code> · R1=P+0.382×Range · R2=P+0.618×Range · R3=P+Range`,
  woodie:      `<b>Woodie</b> · <code>P=(H+L+2×<span class="ha">currOpen</span>)/4</code> · uses current period open`,
  dm:          `<b>DeMark</b> · X=H+L+2C(O=C) / 2H+L+C(C>O) / 2L+H+C(C&lt;O) · P=X/4, R1=X/2−L, S1=X/2−H`,
  floor:       `<b>Floor</b> (= Traditional) · <code>P=(H+L+C)/3</code>`,
};
const TF_SRC = {
  d: '<b>Daily</b> · prev trading session H/L/C · TradingView Auto on ≤15min charts',
  w: '<b>Weekly</b> · prev complete Mon–Fri week H/L/C · TradingView Auto on 30min–4hr charts',
  m: '<b>Monthly</b> · prev complete calendar month H/L/C · TradingView Auto on Daily charts',
  q: '<b>Quarterly</b> · prev complete quarter H/L/C · TradingView Auto on <b>Weekly charts</b> · Q1=Jan–Mar, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Oct–Dec',
  y: '<b>Yearly</b> · prev complete calendar year H/L/C · TradingView Auto on Monthly charts',
};

// ── Pivot formulas ────────────────────────────────────────────────────────────
function computePivots(type, pH, pL, pC, pO, cO) {
  const R = pH - pL;
  if (type==='traditional'||type==='floor') {
    const P=( pH+pL+pC)/3, R3=P*2+(pH-2*pL), S3=P*2-(2*pH-pL);
    return {P, R1:P*2-pL,S1:P*2-pH, R2:P+R,S2:P-R, R3,S3,
            R4:P*3+(pH-3*pL),S4:P*3-(3*pH-pL), R5:P*4+(pH-4*pL),S5:P*4-(4*pH-pL)};
  }
  if (type==='fibonacci') {
    const P=(pH+pL+pC)/3;
    return {P, R1:P+.382*R,S1:P-.382*R, R2:P+.618*R,S2:P-.618*R, R3:P+R,S3:P-R};
  }
  if (type==='woodie') {
    const P=(pH+pL+2*cO)/4, R3=pH+2*(P-pL), S3=pL-2*(pH-P);
    return {P, R1:2*P-pL,S1:2*P-pH, R2:P+R,S2:P-R, R3,S3, R4:R3+R,S4:S3-R};
  }
  if (type==='classic') {
    const P=(pH+pL+pC)/3;
    return {P, R1:2*P-pL,S1:2*P-pH, R2:P+R,S2:P-R, R3:P+2*R,S3:P-2*R, R4:P+3*R,S4:P-3*R};
  }
  if (type==='camarilla') {
    const R5=(pH/pL)*pC, S5=pC-(R5-pC);
    return {R1:pC+1.1*R/12,S1:pC-1.1*R/12, R2:pC+1.1*R/6,S2:pC-1.1*R/6,
            R3:pC+1.1*R/4,S3:pC-1.1*R/4, R4:pC+1.1*R/2,S4:pC-1.1*R/2, R5,S5};
  }
  if (type==='dm') {
    let X = pO===pC ? pH+pL+2*pC : pC>pO ? 2*pH+pL+pC : 2*pL+pH+pC;
    return {P:X/4, R1:X/2-pL, S1:X/2-pH};
  }
  return {};
}

// ── Bounce count ─────────────────────────────────────────────────────────────
// hist entries: compact [pH, pL, pC, pO]
// tf→hist: d→mhist (monthly approx), w→whist (weekly), m→mhist, q→qhist, y→qhist (quarterly)
// "periods" = number of periods to look back (6 or 12, meaning months/weeks/quarters)
function bounceCount(mhist, whist, qhist, tf, type, lv, periods) {
  const hist  = tf==='w' ? whist : (tf==='q'||tf==='y') ? qhist : mhist;
  // For weekly tf, 6 "periods" = 6 weeks; for quarterly 6 = 6 quarters
  const slice = hist.slice(-(periods + 1));
  const isS   = lv.startsWith('S') || lv === 'P';
  let n = 0;
  for (let i = 1; i < slice.length; i++) {
    const [pH, pL, pC, pO] = slice[i-1];
    const curr = slice[i];
    const p    = computePivots(type, pH, pL, pC, pO, curr[3]);
    const tgt  = p[lv];
    if (tgt === undefined) continue;
    if (curr[1] <= tgt && tgt <= curr[0]) {          // range touched level
      if (isS ? curr[2] >= tgt : curr[2] <= tgt) n++; // closed correct side
    }
  }
  return n;
}

// ── State ─────────────────────────────────────────────────────────────────────
let rows = [], sc = 4, sd = 1;

// ── Level dropdown ────────────────────────────────────────────────────────────
function onTypeChange(forceLv) {
  const type = document.getElementById('sp-type').value;
  const meta = TYPE_META[type];
  const sel  = document.getElementById('sp-lvl');
  const cur  = forceLv || sel.value;
  const def  = TYPE_DEF_LV[type] || meta.levels[0];
  sel.innerHTML = '';

  // "Any" options
  const anyOpts = [
    {v:'__ANY_S__', t:'★ Any Support level (nearest S)'},
    {v:'__ANY_R__', t:'★ Any Resistance level (nearest R)'},
    {v:'__ANY__',   t:'★ Any level (nearest S or R)'},
  ];
  const ag = document.createElement('optgroup'); ag.label = '── Multi-level scan ──';
  anyOpts.forEach(o => {
    const opt = document.createElement('option');
    opt.value = o.v; opt.textContent = o.t;
    if (cur === o.v) opt.selected = true;
    ag.appendChild(opt);
  });
  sel.appendChild(ag);

  const sups = meta.levels.filter(l => l==='P' || l.startsWith('S'));
  const ress = meta.levels.filter(l => l.startsWith('R'));
  [[sups,'── Support / Pivot ──'],[ress,'── Resistance ──']].forEach(([lvs, lbl]) => {
    if (!lvs.length) return;
    const g = document.createElement('optgroup'); g.label = lbl;
    lvs.forEach(l => {
      const o = document.createElement('option');
      o.value = l; o.textContent = LV_LABEL[l] || l;
      if (cur ? l===cur : l===def) o.selected = true;
      g.appendChild(o);
    });
    sel.appendChild(g);
  });
  updateFbar();
}

function updateFbar() {
  const type = document.getElementById('sp-type').value;
  const tf   = document.getElementById('sp-tf').value;
  document.getElementById('fbar').innerHTML =
    FBAR_HTML[type] + ' &nbsp;·&nbsp; <span class="hb">Source:</span> ' + TF_SRC[tf];
}

// ── Defaults & localStorage ───────────────────────────────────────────────────
const DEFAULTS = {
  type:'fibonacci', lv:'S2', tf:'q', pr:'2',
  idx:'750', dma:'any', rs:'0', bc:'0',
  prMin:'', prMax:'', cvol:false, c52h:false, c52l:false,
};
const PK = 'pivot_scanner_v3';

function applyPrefs(p) {
  if (p.type)  document.getElementById('sp-type').value = p.type;
  if (p.tf)    document.getElementById('sp-tf').value   = p.tf;
  if (p.pr)    document.getElementById('sp-pr').value   = p.pr;
  if (p.idx)   document.getElementById('idx-flt').value = p.idx;
  if (p.dma)   document.getElementById('dma-flt').value = p.dma;
  if (p.rs)    document.getElementById('rs-flt').value  = p.rs;
  if (p.bc)    document.getElementById('bc-flt').value  = p.bc;
  document.getElementById('pr-min').value  = p.prMin || '';
  document.getElementById('pr-max').value  = p.prMax || '';
  document.getElementById('cb-vol').checked = !!p.cvol;
  document.getElementById('cb-52h').checked = !!p.c52h;
  document.getElementById('cb-52l').checked = !!p.c52l;
  onTypeChange(p.lv || '');
}

function savePrefs() {
  try {
    localStorage.setItem(PK, JSON.stringify({
      type: document.getElementById('sp-type').value,
      lv:   document.getElementById('sp-lvl').value,
      tf:   document.getElementById('sp-tf').value,
      pr:   document.getElementById('sp-pr').value,
      idx:  document.getElementById('idx-flt').value,
      dma:  document.getElementById('dma-flt').value,
      rs:   document.getElementById('rs-flt').value,
      bc:   document.getElementById('bc-flt').value,
      prMin:document.getElementById('pr-min').value,
      prMax:document.getElementById('pr-max').value,
      cvol: document.getElementById('cb-vol').checked,
      c52h: document.getElementById('cb-52h').checked,
      c52l: document.getElementById('cb-52l').checked,
    }));
    const lbl = document.getElementById('saved-lbl');
    lbl.textContent = '✓ SAVED'; setTimeout(()=>lbl.textContent='', 1800);
  } catch(e){}
}

function loadPrefs() {
  try{ return JSON.parse(localStorage.getItem(PK)); }catch(e){ return null; }
}

function resetPrefs() {
  try{ localStorage.removeItem(PK); }catch(e){}
  applyPrefs(DEFAULTS);
  const lbl = document.getElementById('saved-lbl');
  lbl.textContent = '↺ RESET'; setTimeout(()=>lbl.textContent='', 1800);
  scan();
}

// ── Multi-level helper ────────────────────────────────────────────────────────
function lvelsForKey(type, key) {
  const meta = TYPE_META[type];
  if (key==='__ANY_S__') return meta.levels.filter(l=>l!=='P'&&l.startsWith('S'));
  if (key==='__ANY_R__') return meta.levels.filter(l=>l.startsWith('R'));
  if (key==='__ANY__')   return meta.levels.filter(l=>l!=='P');
  return [key];
}

// ── Main scan ─────────────────────────────────────────────────────────────────
function scan() {
  const type  = document.getElementById('sp-type').value;
  const lvKey = document.getElementById('sp-lvl').value;
  const tf    = document.getElementById('sp-tf').value;
  const prox  = parseFloat(document.getElementById('sp-pr').value);
  const idxF  = parseInt(document.getElementById('idx-flt').value);
  const dmaF  = document.getElementById('dma-flt').value;
  const rsF   = parseInt(document.getElementById('rs-flt').value);
  const bcF   = parseInt(document.getElementById('bc-flt').value);
  const prMin = parseFloat(document.getElementById('pr-min').value)||0;
  const prMax = parseFloat(document.getElementById('pr-max').value)||Infinity;
  const meta  = TYPE_META[type];
  const multi = lvKey.startsWith('__ANY');
  const lvels = lvelsForKey(type, lvKey);

  document.getElementById('b-type').textContent = meta.label.toUpperCase();
  document.getElementById('b-lv').textContent   = multi ? lvKey.replace(/__/g,'') : lvKey;
  document.getElementById('b-src').textContent =
    {d:'DAILY',w:'WEEKLY',m:'MONTHLY',q:'QUARTERLY',y:'YEARLY'}[tf] || tf.toUpperCase();
  document.getElementById('b-pr').textContent   = '±'+prox+'%';
  updateFbar(); savePrefs();

  rows = [];
  for (const s of STOCKS) {
    // Index filter
    if (idxF > 0 && (s.idx === 0 || s.idx > idxF)) continue;
    // Price filter
    if (s.price < prMin || s.price > prMax) continue;
    // DMA filter
    if (dmaF==='above' && !s.above200) continue;
    if (dmaF==='below' &&  s.above200) continue;
    // RS filter
    if (s.rs < rsF) continue;

    const src = {d:s.d, w:s.w, m:s.m, q:s.q, y:s.y}[tf] || s.m;
    const p   = computePivots(type, src.pH, src.pL, src.pC, src.pO, src.cO);

    // Find nearest matching level
    let bestTgt=null, bestDist=Infinity, bestLv=null;
    for (const lv of lvels) {
      const tgt = p[lv]; if (tgt===undefined) continue;
      const dist = (s.price - tgt) / Math.abs(tgt) * 100;
      if (Math.abs(dist) < Math.abs(bestDist)) { bestDist=dist; bestTgt=tgt; bestLv=lv; }
    }
    if (bestTgt===null || Math.abs(bestDist)>prox) continue;

    // Bounce counts (JS computed from hist)
    const b6  = bounceCount(s.mhist, s.whist, s.qhist, tf, type, bestLv, 6);
    const b12 = bounceCount(s.mhist, s.whist, s.qhist, tf, type, bestLv, 12);
    if (b12 < bcF) continue;

    rows.push({
      sym:s.sym, idx:s.idx, price:s.price, date:s.date, avol:s.avol,
      lv:+bestTgt.toFixed(2), dist:+bestDist.toFixed(3), lvKey:bestLv,
      p, sh:src.pH, sl2:src.pL, sc2:src.pC, so:src.pO, sd:src.dt,
      dma200:s.dma200, above200:s.above200,
      w52h:s.w52h, w52l:s.w52l, rs:s.rs, b6, b12, multi,
      levels:meta.levels, type,
    });
  }
  sc=4; sd=1;
  rows.sort((a,b)=>Math.abs(a.dist)-Math.abs(b.dist));
  render();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function distCell(dist, prox) {
  const f = Math.abs(dist)/prox;
  const bg = f<=.25?'rgba(0,229,160,.14)':f<=.55?'rgba(0,229,160,.06)':f<=.80?'rgba(245,200,66,.06)':'rgba(255,64,96,.06)';
  const cls = f<=.55?'pos':f<=.80?'':' neg';
  return `<td class="${cls}" style="background:${bg}">${dist>=0?'+':''}${dist.toFixed(2)}%</td>`;
}
function idxBadge(idx) {
  const map={50:['ix50','N50'],100:['ix100','N100'],200:['ix200','N200'],500:['ix500','N500'],750:['ix750','N750'],0:['ix0','—']};
  const [cls,lbl] = map[idx]||map[0];
  return `<td class="${cls}">${lbl}</td>`;
}
function rsCell(rs) {
  const cls = rs>=80?'rs-hi':rs>=60?'rs-md':rs>=40?'rs-lo':'rs-vl';
  return `<td class="${cls}">${rs}</td>`;
}
function dmaCell(price, dma200, above) {
  const pct = ((price-dma200)/dma200*100).toFixed(1);
  const cls = above?'dma-up':'dma-dn';
  const arrow = above?'▲':'▼';
  return `<td class="${cls}">${arrow} ${pct}%</td>`;
}
function w52Cell(price, ref, isHigh) {
  const pct = ((price-ref)/ref*100).toFixed(1);
  const near = Math.abs(pct) < 5;
  const style = near ? 'font-weight:700;color:var(--gold)' : 'color:var(--mu)';
  return `<td style="${style}">${pct>=0?'+':''}${pct}%</td>`;
}
function bcCell(n) {
  const cls = n>=3?'bc-hi':n>=2?'bc-md':'bc-lo';
  return `<td class="${cls}">${n}</td>`;
}
function fmtVol(v) {
  return v>=1e7?`<span style="color:var(--acc);font-size:9px">▲ </span>${(v/1e7).toFixed(1)}Cr`
       :v>=1e5?`${(v/1e5).toFixed(1)}L`:v.toLocaleString();
}

// ── Columns ───────────────────────────────────────────────────────────────────
function buildCols(type, lvKey, prox, multi) {
  const meta   = TYPE_META[type];
  const activeLv = multi ? null : lvKey;
  const lvList = meta.levels.filter(l=>l!==activeLv);

  const core = [
    { k:'sym',  h:'Symbol',      fn:r=>`<td class="csym"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${r.sym}" target="_blank" rel="noopener" title="Open on TradingView"><svg class="tv-ico" width="13" height="13" viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="6" fill="#131722"/><path d="M4 20h4v-8H4v8zm6 0h4V8h-4v12zm6 0h4v-5h-4v5z" fill="#2962FF"/></svg>${r.sym}</a></td>` },
    { k:'idx',  h:'Index',       fn:r=>idxBadge(r.idx) },
    { k:'price',h:'Close',       fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>` },
    { k:'lv',   h:(multi?'Hit Lv':lvKey)+' Level', fn:r=>`<td class="clv">${multi?`<span class="${LV_CSS[r.lvKey]||''}">${r.lvKey} </span>`:''}${r.lv.toFixed(2)}</td>`, hcls:'hl' },
    { k:'dist', h:'Dist %',      fn:r=>distCell(r.dist, prox) },
    { k:'rs',   h:'RS',          fn:r=>rsCell(r.rs) },
    { k:'above200', h:'vs 200DMA', fn:r=>dmaCell(r.price, r.dma200, r.above200) },
    { k:'b6',   h:'Bnc 6M',      fn:r=>bcCell(r.b6) },
    { k:'b12',  h:'Bnc 12M',     fn:r=>bcCell(r.b12) },
    { k:'w52h', h:'vs 52W Hi',   fn:r=>w52Cell(r.price, r.w52h, true) },
    { k:'w52l', h:'vs 52W Lo',   fn:r=>w52Cell(r.price, r.w52l, false) },
  ];

  const div1 = { k:'_d1', h:'All Levels', fn:r=>`<td class="dv">◀▶</td>`, dv:true };
  const lvCols = [];
  if (meta.hasP && activeLv !== 'P')
    lvCols.push({ k:'P', h:'P', fn:r=>`<td class="cp">${(r.p.P??0).toFixed(2)}</td>` });
  lvList.forEach(l => lvCols.push({
    k:l, h:l, fn:r=>`<td class="${LV_CSS[l]||'mu'}">${r.p[l]!==undefined?r.p[l].toFixed(2):'—'}</td>`,
  }));

  const div2 = { k:'_d2', h:'Src Data', fn:r=>`<td class="dv">◀▶</td>`, dv:true };
  const src = [
    { k:'sh',  h:'Src H',    fn:r=>`<td class="mu">${r.sh.toFixed(2)}</td>` },
    { k:'sl2', h:'Src L',    fn:r=>`<td class="mu">${r.sl2.toFixed(2)}</td>` },
    { k:'sc2', h:'Src C',    fn:r=>`<td class="mu">${r.sc2.toFixed(2)}</td>` },
    { k:'so',  h:'Src O',    fn:r=>`<td class="mu">${r.so.toFixed(2)}</td>` },
    { k:'sd',  h:'Src Date', fn:r=>`<td class="mu">${r.sd}</td>` },
    { k:'avol',h:'AvgVol20', fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>` },
    { k:'dma200',h:'200DMA', fn:r=>`<td class="mu">${r.dma200.toFixed(0)}</td>` },
    { k:'w52h_v',h:'52W Hi', fn:r=>`<td class="mu">${r.w52h.toFixed(2)}</td>` },
    { k:'w52l_v',h:'52W Lo', fn:r=>`<td class="mu">${r.w52l.toFixed(2)}</td>` },
    { k:'date',h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>` },
  ];
  return [...core, div1, ...lvCols, div2, ...src];
}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {
  const type  = document.getElementById('sp-type').value;
  const lvKey = document.getElementById('sp-lvl').value;
  const prox  = parseFloat(document.getElementById('sp-pr').value);
  const multi = lvKey.startsWith('__ANY');
  const q     = document.getElementById('q').value.trim().toUpperCase();
  const cvol  = document.getElementById('cb-vol').checked;
  const c52h  = document.getElementById('cb-52h').checked;
  const c52l  = document.getElementById('cb-52l').checked;
  const cs    = buildCols(type, lvKey, prox, multi);

  const vis = rows.filter(r => {
    if (q && !r.sym.includes(q)) return false;
    if (cvol && r.avol < 100000) return false;
    if (c52h && Math.abs((r.price-r.w52h)/r.w52h*100) > 5) return false;
    if (c52l && Math.abs((r.price-r.w52l)/r.w52l*100) > 5) return false;
    return true;
  });

  document.getElementById('st-h').textContent  = vis.length;
  document.getElementById('st-up').textContent = vis.filter(r=>r.above200).length;
  document.getElementById('st-dn').textContent = vis.filter(r=>!r.above200).length;
  document.getElementById('st-rs').textContent = vis.filter(r=>r.rs>=70).length;
  document.getElementById('st-u').textContent  = STOCKS.length;
  document.getElementById('vc').textContent    = vis.length + ' rows';

  if (!vis.length) {
    document.getElementById('ts').innerHTML =
      `<div class="nodata">No stocks found.<br>Try wider Proximity %, changing Index filter, or removing DMA/RS constraints.</div>`;
    return;
  }

  const ths = cs.map((c,i)=>{
    const cls=[c.hcls||'',c.dv?'dv':'',(!c.dv&&i===sc)?(sd===1?'asc':'desc'):''].filter(Boolean).join(' ');
    return `<th class="${cls}"${c.dv?'':` data-i="${i}"`}>${c.h}</th>`;
  }).join('');

  const trs = vis.map(r=>{
    const nrCls = r.idx>0&&r.idx<=750?' class="nr"':'';
    return `<tr${nrCls}>${cs.map(c=>c.fn(r)).join('')}</tr>`;
  }).join('');

  document.getElementById('ts').innerHTML =
    `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;

  document.querySelectorAll('#ts th[data-i]').forEach(th=>{
    th.addEventListener('click',()=>{
      const i=+th.dataset.i;
      if(sc===i)sd*=-1;else{sc=i;sd=1;}
      const k=cs[i].k;
      rows.sort((a,b)=>(typeof a[k]==='number'?a[k]-b[k]:String(a[k]).localeCompare(String(b[k])))*sd);
      render();
    });
  });
}

// ── Export CSV ────────────────────────────────────────────────────────────────
function exportCSV() {
  if (!rows.length) return;
  const type  = document.getElementById('sp-type').value;
  const lvKey = document.getElementById('sp-lvl').value;
  const multi = lvKey.startsWith('__ANY');
  const meta  = TYPE_META[type];
  const hdrs  = ['Symbol','Index','Close','Hit Level','Level Value','Dist%',
                 'RS','Above200DMA','vs200DMA%','Bnc6M','Bnc12M',
                 'vs52WHi%','vs52WLo%','P',...meta.levels,
                 'SrcH','SrcL','SrcC','SrcO','SrcDate','AvgVol20',
                 '200DMA','52WHi','52WLo','LastDate'];
  const lines=[hdrs.join(',')];
  rows.forEach(r=>{
    const d200pct=((r.price-r.dma200)/r.dma200*100).toFixed(1);
    const v52h=((r.price-r.w52h)/r.w52h*100).toFixed(1);
    const v52l=((r.price-r.w52l)/r.w52l*100).toFixed(1);
    lines.push([
      r.sym, r.idx||'Other', r.price.toFixed(2), r.lvKey, r.lv.toFixed(2),
      (r.dist>=0?'+':'')+r.dist.toFixed(2)+'%',
      r.rs, r.above200?'Y':'N', (d200pct>=0?'+':'')+d200pct+'%',
      r.b6, r.b12, (v52h>=0?'+':'')+v52h+'%', (v52l>=0?'+':'')+v52l+'%',
      r.p.P!==undefined?r.p.P.toFixed(2):'',
      ...meta.levels.map(l=>r.p[l]!==undefined?r.p[l].toFixed(2):''),
      r.sh,r.sl2,r.sc2,r.so,r.sd,r.avol,r.dma200.toFixed(0),r.w52h,r.w52l,r.date,
    ].join(','));
  });
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));
  a.download=`pivot_scan_${type}_${lvKey}_${new Date().toISOString().slice(0,10)}.csv`;
  a.click();
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('st-u').textContent = STOCKS.length;
  const p = loadPrefs();
  applyPrefs(p || DEFAULTS);
  scan();
});
</script>
</body>
</html>
"""

def build_html(stocks, data_dir):
    gt = datetime.now().strftime("%d %b %Y  %H:%M")
    return (HTML
            .replace("__JSON__", json.dumps(stocks, separators=(",",":")))
            .replace("__GT__",   gt)
            .replace("__DD__",   data_dir))

def main():
    ap = argparse.ArgumentParser(description="Pivot Point Scanner — NSE")
    ap.add_argument("--data",  default=DATA_DIR)
    ap.add_argument("--n50",   default=N50_FILE)
    ap.add_argument("--n100",  default=N100_FILE)
    ap.add_argument("--n200",  default=N200_FILE)
    ap.add_argument("--n500",  default=N500_FILE)
    ap.add_argument("--n750",  default=N750_FILE)
    ap.add_argument("--out",   default=OUTPUT_HTML)
    a = ap.parse_args()
    print(f"[*] Data : {a.data}")
    index_files = {50:a.n50, 100:a.n100, 200:a.n200, 500:a.n500, 750:a.n750}
    stocks = build_dataset(a.data, index_files)
    if not stocks:
        print("[!] No stocks loaded — check --data path"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Output : {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
