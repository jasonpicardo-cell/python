#!/usr/bin/env python3
"""
Pivot Point Scanner — NSE
All 7 types: Traditional, Fibonacci, Woodie, Classic, Camarilla, DeMark, Floor
Formulas: TradingView official docs (https://www.tradingview.com/support/solutions/43000521824)

Auto timeframe (TradingView-verified):
  chart >= 1D (daily CSVs) → Monthly source  (prev complete month H/L/C)
"""

import os, glob, json, argparse, math
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

DATA_DIR    = "../nse_data_cache"
NIFTY_FILE  = "../nifty750.txt"
OUTPUT_HTML = "pivot_scanner.html"

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_nifty750(f):
    try:   return {ln.strip().upper() for ln in open(f) if ln.strip()}
    except FileNotFoundError: return set()

def load_csv(fp):
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
    except: return None

def r2(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 0.0
    return round(float(v), 2)

# ── Source-candle builders ────────────────────────────────────────────────────
def daily_src(df):
    """Last completed trading day."""
    if len(df) < 1: return None
    last = df.iloc[-1]
    return dict(pH=r2(last["High"]), pL=r2(last["Low"]),
                pC=r2(last["Close"]), pO=r2(last.get("Open", last["Close"])),
                cO=r2(last.get("Open", last["Close"])),   # no future open available
                dt=str(last["Date"].date()))

def weekly_src(df, today):
    """Previous complete Mon–Fri week."""
    wsm = today - pd.Timedelta(days=today.dayofweek)  # Monday of this week
    prev = df[df["Date"] < wsm]
    if not len(prev): return None
    we  = wsm - pd.Timedelta(days=1)
    ws  = we  - pd.Timedelta(days=6)
    wk  = prev[(prev["Date"] >= ws) & (prev["Date"] <= we)]
    if not len(wk): wk = prev.tail(5)
    # current week open (for Woodie currOpen)
    cur = df[df["Date"] >= wsm]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(wk["Close"].iloc[-1])
    return dict(pH=r2(wk["High"].max()), pL=r2(wk["Low"].min()),
                pC=r2(wk["Close"].iloc[-1]), pO=r2(wk["Open"].iloc[0]),
                cO=cO, dt=str(wk["Date"].iloc[-1].date()))

def monthly_src(df, today):
    """Previous complete calendar month — TradingView Auto on daily chart."""
    fom = today.replace(day=1)                     # first of current month
    prev = df[df["Date"] < fom]
    if not len(prev): return None
    me  = fom - pd.Timedelta(days=1)
    ms  = me.replace(day=1)
    mo  = prev[(prev["Date"] >= ms) & (prev["Date"] <= me)]
    if not len(mo): mo = prev.tail(22)
    # current month open (for Woodie currOpen)
    cur = df[df["Date"] >= fom]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(mo["Close"].iloc[-1])
    return dict(pH=r2(mo["High"].max()), pL=r2(mo["Low"].min()),
                pC=r2(mo["Close"].iloc[-1]), pO=r2(mo["Open"].iloc[0]),
                cO=cO, dt=str(mo["Date"].iloc[-1].date()))

# ── Per-stock precompute ───────────────────────────────────────────────────────
def precompute(fp, nifty750):
    sym = Path(fp).stem.upper()
    df  = load_csv(fp)
    if df is None or len(df) < 2: return None
    today = pd.Timestamp(datetime.now().date())
    ds = daily_src(df)
    ws = weekly_src(df, today) or ds
    ms = monthly_src(df, today) or ws
    if ds is None: return None
    avol = int(df["Volume"].tail(20).mean()) if len(df) >= 20 else int(df["Volume"].mean())
    return dict(sym=sym, n750=sym in nifty750,
                price=r2(df.iloc[-1]["Close"]),
                date=str(df.iloc[-1]["Date"].date()),
                d=ds, w=ws, m=ms, avol=avol)

def build_dataset(data_dir, nifty_file):
    n750 = load_nifty750(nifty_file)
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    out = []
    for i, fp in enumerate(files, 1):
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}]…", end="\r")
        rec = precompute(fp, n750)
        if rec: out.append(rec)
    print(f"\n  Done — {len(out)} stocks")
    return out

# ── HTML ───────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pivot Point Scanner — NSE</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080b10;--surf:#0d1220;--brd:#182034;
  --acc:#00e5a0;--a2:#3b9eff;--a3:#c47aff;
  --warn:#ff8c42;--red:#ff4060;--gold:#f5c842;
  --txt:#cdd5e8;--mu:#4a5c78;--hdr:#060810;
  --mono:'IBM Plex Mono',monospace;
  --sans:'Space Grotesk',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);overflow-y:scroll}

/* ── topbar ── */
.tb{background:var(--hdr);border-bottom:1px solid var(--brd);padding:11px 22px;
  display:flex;align-items:center;gap:10px;position:sticky;top:0;z-index:300}
.logo{font-family:var(--mono);font-size:11px;letter-spacing:3px;color:var(--acc);
  text-transform:uppercase;white-space:nowrap}
.fill{flex:1}
.bd{font-family:var(--mono);font-size:9px;padding:3px 9px;border-radius:3px;
  letter-spacing:.8px;white-space:nowrap}
.ba{background:rgba(0,229,160,.07);border:1px solid rgba(0,229,160,.22);color:var(--acc)}
.bb{background:rgba(59,158,255,.07);border:1px solid rgba(59,158,255,.22);color:var(--a2)}
.bc{background:rgba(196,122,255,.07);border:1px solid rgba(196,122,255,.22);color:var(--a3)}
.bw{background:rgba(255,140,66,.07);border:1px solid rgba(255,140,66,.22);color:var(--warn)}

/* ── controls ── */
.ctrl{padding:16px 22px 0;display:flex;flex-wrap:wrap;gap:9px;align-items:flex-end}
.cg{display:flex;flex-direction:column;gap:4px}
.cg label{font-size:9px;letter-spacing:1.8px;text-transform:uppercase;
  color:var(--mu);font-family:var(--mono)}
select{background:var(--surf);color:var(--txt);border:1px solid var(--brd);border-radius:4px;
  padding:6px 24px 6px 9px;font-family:var(--mono);font-size:12px;cursor:pointer;
  outline:none;-webkit-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='9' fill='%234a5c78' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 7px center;
  transition:border-color .2s;min-width:120px}
select:hover,select:focus{border-color:var(--acc)}
#sp-type{min-width:160px}
#sp-lvl{min-width:200px}
.btn{background:var(--acc);color:#000;border:none;border-radius:4px;padding:6px 16px;
  font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:1.5px;
  text-transform:uppercase;cursor:pointer;align-self:flex-end;
  transition:opacity .15s,transform .1s;white-space:nowrap}
.btn:hover{opacity:.83;transform:translateY(-1px)}
.btn:active{transform:translateY(0)}

/* ── formula bar ── */
.fbar{margin:12px 22px 0;padding:9px 14px;border-radius:5px;
  background:rgba(59,158,255,.04);border:1px solid rgba(59,158,255,.12);
  font-family:var(--mono);font-size:10.5px;color:var(--mu);line-height:1.9}
.fbar b{color:var(--txt)} .fbar .ha{color:var(--acc)} .fbar .hb{color:var(--a2)}
.fbar .hc{color:var(--a3)} .fbar code{color:var(--gold);font-size:10px}

/* ── stats ── */
.stats{padding:11px 22px 0;display:flex;gap:9px;flex-wrap:wrap}
.st{background:var(--surf);border:1px solid var(--brd);border-radius:6px;
  padding:8px 14px;min-width:90px}
.sl{font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--mu);
  font-family:var(--mono)}
.sv{font-family:var(--mono);font-size:18px;font-weight:700;color:var(--acc);margin-top:2px}
.sv.sm{font-size:11px;margin-top:4px;color:var(--txt);font-weight:400}

/* ── search row ── */
.sr{padding:10px 22px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.sr input[type=text]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:6px 10px;font-family:var(--mono);font-size:12px;
  outline:none;width:180px;transition:border-color .2s}
.sr input[type=text]:focus{border-color:var(--a2)}
.cbl{display:flex;align-items:center;gap:6px;font-size:12px;color:var(--mu);
  cursor:pointer;user-select:none}
.cbl input{accent-color:var(--acc);width:13px;height:13px;cursor:pointer}
.cnt{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.18);color:var(--a2);
  font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:3px;margin-left:auto}

/* ── table ── */
.tw{padding:10px 22px 0}
.ts{overflow-x:auto;overflow-y:auto;max-height:59vh;border:1px solid var(--brd);
  border-radius:8px;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
table{border-collapse:collapse;width:max-content;min-width:100%;
  font-family:var(--mono);font-size:11.5px}
thead{position:sticky;top:0;z-index:10}
thead tr{background:var(--hdr);border-bottom:2px solid var(--acc)}
th{padding:8px 12px;text-align:left;font-size:8.5px;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--mu);cursor:pointer;white-space:nowrap;
  user-select:none;border-right:1px solid var(--brd);transition:color .15s}
th:last-child{border-right:none}
th:hover{color:var(--acc)}
th.asc::after{content:" ▲";color:var(--acc);font-size:7px}
th.desc::after{content:" ▼";color:var(--acc);font-size:7px}
th.hl{color:var(--a2) !important}

tbody tr{border-bottom:1px solid rgba(24,32,52,.9);transition:background .1s}
tbody tr:hover{background:#0c1422}
tbody tr.nr td:first-child{border-left:2px solid var(--acc)}
td{padding:7px 12px;white-space:nowrap;border-right:1px solid rgba(24,32,52,.7)}
td:last-child{border-right:none}

/* cell colours */
.csym{font-weight:700;color:#dde5f5;font-size:12.5px}
.csym a{color:inherit;text-decoration:none;display:flex;align-items:center;gap:6px;white-space:nowrap}
.csym a:hover{color:var(--acc)}
.csym a:hover .tv-ico{opacity:1}
.tv-ico{opacity:.35;transition:opacity .15s;flex-shrink:0}
.cpr{color:#dde5f5;font-weight:600}
.clv{color:var(--a2);font-weight:700}
.cp{color:var(--gold)}
.pos{color:var(--acc)}.neg{color:var(--red)}
.cs1{color:#a0c8ff}.cs2{color:#6aaeff}.cs3{color:#3090ff}.cs4{color:var(--warn)}.cs5{color:var(--red)}
.cr1{color:#a0ffe0}.cr2{color:#60ffcc}.cr3{color:#00ffb0}.cr4{color:var(--red)}.cr5{color:#ff0060}
.mu{color:var(--mu)}
.chk{color:var(--acc)}
th.dv,td.dv{background:rgba(59,158,255,.03);border-left:1px solid rgba(59,158,255,.1);
  border-right:1px solid rgba(59,158,255,.1);font-size:8px;color:var(--mu);
  text-align:center;padding:7px 4px;cursor:default;letter-spacing:.3px}
.nodata{padding:70px;text-align:center;color:var(--mu);font-family:var(--mono);
  font-size:13px;line-height:2.2}

/* footer */
.ft{padding:14px 22px 4px;font-size:9.5px;color:var(--mu);font-family:var(--mono);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:6px;
  border-top:1px solid var(--brd);margin-top:12px}
</style>
</head>
<body>

<div class="tb">
  <div class="logo">◈ Pivot Point Scanner · NSE</div>
  <div class="fill"></div>
  <span class="bd bc" id="b-type">CAMARILLA</span>
  <span class="bd ba" id="b-lv">S3</span>
  <span class="bd bb" id="b-src">AUTO · MONTHLY</span>
  <span class="bd bw" id="b-pr">±2%</span>
</div>

<div class="ctrl">
  <div class="cg">
    <label>Pivot Type</label>
    <select id="sp-type" onchange="onTypeChange()">
      <option value="camarilla">Camarilla</option>
      <option value="traditional">Traditional</option>
      <option value="classic">Classic</option>
      <option value="fibonacci">Fibonacci</option>
      <option value="woodie">Woodie</option>
      <option value="dm">DeMark (DM)</option>
      <option value="floor">Floor</option>
    </select>
  </div>
  <div class="cg">
    <label>Level</label>
    <select id="sp-lvl"></select>
  </div>
  <div class="cg">
    <label>Pivot Timeframe</label>
    <select id="sp-tf">
      <option value="m" selected>Auto (Monthly) — TradingView default on daily chart</option>
      <option value="w">Weekly — TradingView default on 30m–4h chart</option>
      <option value="d">Daily — TradingView default on ≤15m chart</option>
    </select>
  </div>
  <div class="cg">
    <label>Proximity ±%</label>
    <select id="sp-pr">
      <option value="0.5">±0.5%</option>
      <option value="1">±1%</option>
      <option value="2" selected>±2%</option>
      <option value="3">±3%</option>
      <option value="5">±5%</option>
      <option value="10">±10%</option>
    </select>
  </div>
  <button class="btn" onclick="scan()">▶ SCAN</button>
</div>

<div class="fbar" id="fbar">Loading…</div>

<div class="stats">
  <div class="st"><div class="sl">Hits</div><div class="sv" id="st-h">—</div></div>
  <div class="st"><div class="sl">Nifty 750</div><div class="sv" id="st-n">—</div></div>
  <div class="st"><div class="sl">Others</div><div class="sv" id="st-o">—</div></div>
  <div class="st"><div class="sl">Universe</div><div class="sv" id="st-u">—</div></div>
  <div class="st"><div class="sl">Generated</div><div class="sv sm">__GT__</div></div>
</div>

<div class="sr">
  <input type="text" id="q" placeholder="Search symbol…" oninput="render()">
  <label class="cbl">
    <input type="checkbox" id="cb7" checked onchange="render()"> Nifty 750 only
  </label>
  <span class="cnt" id="vc">— rows</span>
</div>

<div class="tw"><div class="ts" id="ts"></div></div>

<div class="ft">
  <span>Formulas: TradingView official · Auto = Monthly source for daily charts (1D+)</span>
  <span>Data: __DD__ · __GT__</span>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const STOCKS = __JSON__;

// ── Pivot type definitions ────────────────────────────────────────────────────
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
  P:'P — Pivot', S1:'S1 — Support 1', S2:'S2 — Support 2', S3:'S3 — Support 3',
  S4:'S4 — Support 4', S5:'S5 — Support 5',
  R1:'R1 — Resistance 1', R2:'R2 — Resistance 2', R3:'R3 — Resistance 3',
  R4:'R4 — Resistance 4', R5:'R5 — Resistance 5',
};

const LV_CSS = {
  P:'cp', S1:'cs1', S2:'cs2', S3:'cs3', S4:'cs4', S5:'cs5',
  R1:'cr1', R2:'cr2', R3:'cr3', R4:'cr4', R5:'cr5',
};

// ── TradingView-official formulas ─────────────────────────────────────────────
function computePivots(type, pH, pL, pC, pO, cO) {
  // pH/pL/pC/pO = prev period High/Low/Close/Open
  // cO = current period Open (Woodie uses this)
  const R = pH - pL;

  if (type === 'traditional' || type === 'floor') {
    const P  = (pH + pL + pC) / 3;
    const R3 = P*2 + (pH - 2*pL);
    const S3 = P*2 - (2*pH - pL);
    return { P,
      R1: P*2 - pL,           S1: P*2 - pH,
      R2: P + R,              S2: P - R,
      R3,                     S3,
      R4: P*3 + (pH - 3*pL),  S4: P*3 - (3*pH - pL),
      R5: P*4 + (pH - 4*pL),  S5: P*4 - (4*pH - pL),
    };
  }
  if (type === 'fibonacci') {
    const P = (pH + pL + pC) / 3;
    return { P,
      R1: P + 0.382*R,  S1: P - 0.382*R,
      R2: P + 0.618*R,  S2: P - 0.618*R,
      R3: P + R,        S3: P - R,
    };
  }
  if (type === 'woodie') {
    const P  = (pH + pL + 2*cO) / 4;   // uses current period Open
    const R3 = pH + 2*(P - pL);
    const S3 = pL - 2*(pH - P);
    return { P,
      R1: 2*P - pL,   S1: 2*P - pH,
      R2: P + R,      S2: P - R,
      R3,             S3,
      R4: R3 + R,     S4: S3 - R,
    };
  }
  if (type === 'classic') {
    const P = (pH + pL + pC) / 3;
    return { P,
      R1: 2*P - pL,    S1: 2*P - pH,
      R2: P + R,       S2: P - R,
      R3: P + 2*R,     S3: P - 2*R,
      R4: P + 3*R,     S4: P - 3*R,
    };
  }
  if (type === 'camarilla') {
    const R5 = (pH / pL) * pC;
    const S5 = pC - (R5 - pC);
    return {
      R1: pC + 1.1*R/12,  S1: pC - 1.1*R/12,
      R2: pC + 1.1*R/6,   S2: pC - 1.1*R/6,
      R3: pC + 1.1*R/4,   S3: pC - 1.1*R/4,
      R4: pC + 1.1*R/2,   S4: pC - 1.1*R/2,
      R5,                  S5,
    };
  }
  if (type === 'dm') {
    let X;
    if (pO === pC)      X = pH + pL + 2*pC;
    else if (pC > pO)   X = 2*pH + pL + pC;
    else                X = 2*pL + pH + pC;
    const P = X / 4;
    return { P, R1: X/2 - pL, S1: X/2 - pH };
  }
  return {};
}

// ── Formula descriptions ──────────────────────────────────────────────────────
const FBAR_HTML = {
  camarilla: `<b>Camarilla</b> &nbsp;<span class="ha">Sn = prevC ∓ (prevH−prevL)×1.1/k</span> &nbsp;<span class="hb">Rn = prevC ± (prevH−prevL)×1.1/k</span> &nbsp;· k: 1=12, 2=6, 3=4, 4=2 &nbsp;· <code>R5=(prevH/prevL)×prevC &nbsp; S5=prevC−(R5−prevC)</code>`,
  traditional:`<b>Traditional</b> &nbsp;<code>P=(H+L+C)/3</code> &nbsp;· R1=2P−L, S1=2P−H &nbsp;· R2=P+Range, S2=P−Range &nbsp;· R3=P×2+(H−2L), S3=P×2−(2H−L) &nbsp;· up to R5/S5`,
  classic:    `<b>Classic</b> &nbsp;<code>P=(H+L+C)/3</code> &nbsp;· R1=2P−L, S1=2P−H &nbsp;· R2=P+Range, S2=P−Range &nbsp;· R3=P+2×Range, S3=P−2×Range &nbsp;· R4=P+3×Range, S4=P−3×Range`,
  fibonacci:  `<b>Fibonacci</b> &nbsp;<code>P=(H+L+C)/3</code> &nbsp;· R1=P+0.382×Range, S1=P−0.382×Range &nbsp;· R2=P+0.618×Range, S2=P−0.618×Range &nbsp;· R3=P+Range, S3=P−Range`,
  woodie:     `<b>Woodie</b> &nbsp;<code>P=(H+L+2×<span class="ha">currOpen</span>)/4</code> &nbsp;· uses <span class="ha">current period's Open</span>, not prevClose &nbsp;· R1=2P−L, S1=2P−H &nbsp;· R3=H+2(P−L), S3=L−2(H−P)`,
  dm:         `<b>DeMark (DM)</b> &nbsp;Only P, S1, R1 &nbsp;· <code>X=H+L+2C</code> if O=C; &nbsp;<code>X=2H+L+C</code> if C>O; &nbsp;<code>X=2L+H+C</code> if C&lt;O &nbsp;· P=X/4, R1=X/2−L, S1=X/2−H`,
  floor:      `<b>Floor</b> (= Traditional) &nbsp;<code>P=(H+L+C)/3</code> &nbsp;· Same formula as Traditional &nbsp;· R1=2P−L … up to R5/S5`,
};

const TF_SRC = {
  m: 'Auto · <b>Monthly</b> source — prev complete month H/L/C — matches TradingView "Auto" on daily chart',
  w: '<b>Weekly</b> source — prev complete Mon–Fri week H/L/C',
  d: '<b>Daily</b> source — last completed trading session H/L/C',
};

// ── State ─────────────────────────────────────────────────────────────────────
let rows = [], sc = 4, sd = 1;

// ── Level dropdown update ─────────────────────────────────────────────────────
function onTypeChange() {
  const type   = document.getElementById('sp-type').value;
  const meta   = TYPE_META[type];
  const sel    = document.getElementById('sp-lvl');
  const curLv  = sel.value;
  sel.innerHTML = '';

  const sups = meta.levels.filter(l => l === 'P' || l.startsWith('S'));
  const ress = meta.levels.filter(l => l.startsWith('R'));

  if (sups.length) {
    const g = document.createElement('optgroup');
    g.label = '── Support / Pivot ──';
    sups.forEach(l => {
      const o = document.createElement('option');
      o.value = l; o.textContent = LV_LABEL[l] || l;
      if (l === curLv || (!sel.value && l === (meta.levels.includes('S3') ? 'S3' : meta.levels[0])))
        o.selected = true;
      g.appendChild(o);
    });
    sel.appendChild(g);
  }
  if (ress.length) {
    const g = document.createElement('optgroup');
    g.label = '── Resistance ──';
    ress.forEach(l => {
      const o = document.createElement('option');
      o.value = l; o.textContent = LV_LABEL[l] || l;
      g.appendChild(o);
    });
    sel.appendChild(g);
  }
  updateFbar();
}

function updateFbar() {
  const type = document.getElementById('sp-type').value;
  const tf   = document.getElementById('sp-tf').value;
  document.getElementById('fbar').innerHTML =
    FBAR_HTML[type] + ' &nbsp;·&nbsp; <span class="hb">Source:</span> ' + TF_SRC[tf];
}

// ── Main scan ─────────────────────────────────────────────────────────────────
function scan() {
  const type = document.getElementById('sp-type').value;
  const lv   = document.getElementById('sp-lvl').value;
  const tf   = document.getElementById('sp-tf').value;
  const prox = parseFloat(document.getElementById('sp-pr').value);
  const meta = TYPE_META[type];

  // Update badges
  document.getElementById('b-type').textContent = meta.label.toUpperCase();
  document.getElementById('b-lv').textContent   = lv;
  document.getElementById('b-src').textContent  = tf==='m' ? 'AUTO·MONTHLY' : tf==='w' ? 'WEEKLY' : 'DAILY';
  document.getElementById('b-pr').textContent   = '±'+prox+'%';
  updateFbar();

  rows = [];
  for (const s of STOCKS) {
    const src = tf==='d' ? s.d : tf==='w' ? s.w : s.m;
    const p   = computePivots(type, src.pH, src.pL, src.pC, src.pO, src.cO);
    const tgt = p[lv];
    if (tgt === undefined || tgt === null) continue;

    const dist = (s.price - tgt) / Math.abs(tgt) * 100;
    if (Math.abs(dist) > prox) continue;

    rows.push({ sym:s.sym, n750:s.n750, price:s.price, date:s.date, avol:s.avol,
      lv: tgt, dist: +dist.toFixed(3), p: p,
      sh:src.pH, sl2:src.pL, sc2:src.pC, so:src.pO, co:src.cO, sd:src.dt,
      type, lvKey:lv, levels:meta.levels,
    });
  }

  sc=4; sd=1;
  rows.sort((a,b) => Math.abs(a.dist) - Math.abs(b.dist));
  render();
}

// ── Column builder (dynamic per pivot type) ───────────────────────────────────
function buildCols(type, lv) {
  const meta   = TYPE_META[type];
  const lvList = meta.levels.filter(l => l !== lv);  // other levels

  const fixed = [
    { k:'sym',  h:'Symbol',     fn:r=>`<td class="csym"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${r.sym}" target="_blank" rel="noopener" title="Open ${r.sym} on TradingView"><svg class="tv-ico" width="14" height="14" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="28" height="28" rx="6" fill="#131722"/><path d="M4 20h4v-8H4v8zm6 0h4V8h-4v12zm6 0h4v-5h-4v5z" fill="#2962FF"/><path d="M20 10l4-4" stroke="#2962FF" stroke-width="2" stroke-linecap="round"/><path d="M20 10h4V6h-4v4z" fill="#2962FF"/></svg>${r.sym}</a></td>` },
    { k:'n750', h:'N750',       fn:r=>`<td class="${r.n750?'chk':'mu'}">${r.n750?'✓':'–'}</td>` },
    { k:'price',h:'Last Close', fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>` },
    { k:'lv',   h:lv+' Level',  fn:r=>`<td class="clv">${r.lv.toFixed(2)}</td>`, hcls:'hl' },
    { k:'dist', h:'Dist %',     fn:r=>`<td class="${r.dist>=0?'pos':'neg'}">${r.dist>=0?'+':''}${r.dist.toFixed(2)}%</td>` },
  ];

  // P column if applicable and not the selected level
  const pivotCols = [];
  if (meta.hasP && lv !== 'P') {
    pivotCols.push({ k:'P', h:'P (Pivot)', fn:r=>`<td class="cp">${(r.p.P??0).toFixed(2)}</td>` });
  }

  // All other S/R levels
  const otherCols = lvList.map(l => ({
    k: l, h: l,
    fn: r => {
      const v = r.p[l];
      return `<td class="${LV_CSS[l]||'mu'}">${v !== undefined ? v.toFixed(2) : '—'}</td>`;
    },
  }));

  const div1 = { k:'_d1', h:'All Levels', fn:r=>`<td class="dv">◀▶</td>`, dv:true };
  const div2 = { k:'_d2', h:'Src Data',   fn:r=>`<td class="dv">◀▶</td>`, dv:true };

  const srcCols = [
    { k:'sh',  h:'Src High',  fn:r=>`<td class="mu">${r.sh.toFixed(2)}</td>` },
    { k:'sl2', h:'Src Low',   fn:r=>`<td class="mu">${r.sl2.toFixed(2)}</td>` },
    { k:'sc2', h:'Src Close', fn:r=>`<td class="mu">${r.sc2.toFixed(2)}</td>` },
    { k:'so',  h:'Src Open',  fn:r=>`<td class="mu">${r.so.toFixed(2)}</td>` },
    { k:'sd',  h:'Src Date',  fn:r=>`<td class="mu">${r.sd}</td>` },
    { k:'avol',h:'AvgVol20',  fn:r=>`<td class="mu">${r.avol.toLocaleString()}</td>` },
    { k:'date',h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>` },
  ];

  return [...fixed, div1, ...pivotCols, ...otherCols, div2, ...srcCols];
}

// ── Render ────────────────────────────────────────────────────────────────────
function render() {
  const type = document.getElementById('sp-type').value;
  const lv   = document.getElementById('sp-lvl').value;
  const q    = document.getElementById('q').value.trim().toUpperCase();
  const n750 = document.getElementById('cb7').checked;
  const cs   = buildCols(type, lv);

  const vis = rows.filter(r => {
    if (q && !r.sym.includes(q)) return false;
    if (n750 && !r.n750) return false;
    return true;
  });

  document.getElementById('st-h').textContent = vis.length;
  document.getElementById('st-n').textContent = vis.filter(r=>r.n750).length;
  document.getElementById('st-o').textContent = vis.filter(r=>!r.n750).length;
  document.getElementById('st-u').textContent = STOCKS.length;
  document.getElementById('vc').textContent   = vis.length + ' rows';

  if (!vis.length) {
    document.getElementById('ts').innerHTML =
      `<div class="nodata">No stocks found near <b>${lv}</b> (${TYPE_META[type].label}) within ±${document.getElementById('sp-pr').value}%<br>Try a wider proximity or different timeframe.</div>`;
    return;
  }

  const ths = cs.map((c,i) => {
    const cls = [c.hcls||'', c.dv?'dv':'', (!c.dv&&i===sc)?(sd===1?'asc':'desc'):'']
      .filter(Boolean).join(' ');
    const di = c.dv ? '' : ` data-i="${i}"`;
    return `<th class="${cls}"${di}>${c.h}</th>`;
  }).join('');

  const trs = vis.map(r =>
    `<tr${r.n750?' class="nr"':''}>${cs.map(c=>c.fn(r)).join('')}</tr>`
  ).join('');

  document.getElementById('ts').innerHTML =
    `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;

  document.querySelectorAll('#ts th[data-i]').forEach(th => {
    th.addEventListener('click', () => {
      const i = +th.dataset.i;
      if (sc===i) sd*=-1; else{sc=i;sd=1;}
      const k = cs[i].k;
      rows.sort((a,b) =>
        (typeof a[k]==='number' ? a[k]-b[k] : String(a[k]).localeCompare(String(b[k])))*sd);
      render();
    });
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('st-u').textContent = STOCKS.length;
  onTypeChange();   // populate level dropdown
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
    ap.add_argument("--nifty", default=NIFTY_FILE)
    ap.add_argument("--out",   default=OUTPUT_HTML)
    a = ap.parse_args()
    print(f"[*] Data : {a.data}")
    stocks = build_dataset(a.data, a.nifty)
    if not stocks:
        print("[!] No stocks loaded — check --data path"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Output: {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
