#!/usr/bin/env python3
"""
NSE Strategy Scanner — Pivot Points · Price Action/SMC · Volume · Multi-Indicator
All 4 strategies computed from daily OHLC CSV data and embedded in a single HTML file.
"""

import os, glob, json, argparse, math
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR    = "../nse_data_cache"
N50_FILE    = "../nifty50.txt"; N100_FILE = "../nifty100.txt"
N200_FILE   = "../nifty200.txt"; N500_FILE = "../nifty500.txt"; N750_FILE = "../nifty750.txt"
OUTPUT_HTML = "pivot_scanner.html"
HIST_MONTHS = 25

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_set(fp):
    try:    return {ln.strip().upper() for ln in open(fp) if ln.strip()}
    except: return set()

def load_index_map(files):
    m = {}
    for tier, fp in sorted(files.items()):
        for sym in load_set(fp):
            if sym not in m: m[sym] = tier
    return m

def load_csv(fp):
    try:
        df = pd.read_csv(fp)
        col = next((c for c in df.columns if c.lower() in ("datetime","date")), None)
        if col is None: return None
        df.rename(columns={col:"Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True); df.reset_index(drop=True, inplace=True)
        for c in ("Open","High","Low","Close","Volume"):
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df.dropna(subset=["High","Low","Close"], inplace=True)
        return df
    except: return None

def r2(v):
    if v is None or (isinstance(v, float) and math.isnan(v)): return 0.0
    return round(float(v), 2)

# ── Source-candle builders (pivot) ────────────────────────────────────────────
def daily_src(df):
    if len(df) < 1: return None
    row = df.iloc[-1]
    return dict(pH=r2(row["High"]), pL=r2(row["Low"]),
                pC=r2(row["Close"]), pO=r2(row.get("Open", row["Close"])),
                cO=r2(row.get("Open", row["Close"])), dt=str(row["Date"].date()))

def weekly_src(df, today):
    wsm  = today - pd.Timedelta(days=today.dayofweek)
    past = df[df["Date"] < wsm]
    if not len(past): return None
    weekly = (past.set_index("Date").resample("W-FRI")
                  .agg(pH=("High","max"),pL=("Low","min"),pC=("Close","last"),
                       pO=("Open","first"),_n=("Close","count"))
                  .query("_n > 0").drop(columns=["_n"]).dropna().reset_index())
    if not len(weekly): return None
    pw = weekly.iloc[-1]
    cur = df[df["Date"] >= wsm]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(pw.pC))
    return dict(pH=r2(float(pw.pH)),pL=r2(float(pw.pL)),pC=r2(float(pw.pC)),
                pO=r2(float(pw.pO)),cO=cO,dt=str(pw.Date.date()))

def monthly_src(df, today):
    fom = today.replace(day=1); prev = df[df["Date"] < fom]
    if not len(prev): return None
    me = fom-pd.Timedelta(days=1); ms = me.replace(day=1)
    mo = prev[(prev["Date"]>=ms)&(prev["Date"]<=me)]
    if not len(mo): mo = prev.tail(22)
    cur = df[df["Date"] >= fom]
    cO  = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(mo["Close"].iloc[-1]))
    return dict(pH=r2(float(mo["High"].max())),pL=r2(float(mo["Low"].min())),
                pC=r2(float(mo["Close"].iloc[-1])),pO=r2(float(mo["Open"].iloc[0])),
                cO=cO,dt=str(mo["Date"].iloc[-1].date()))

def quarterly_src(df, today):
    qsm = ((today.month-1)//3)*3+1; cqs = today.replace(month=qsm,day=1)
    pqe = cqs-pd.Timedelta(days=1); pqsm = ((pqe.month-1)//3)*3+1
    pqs = pqe.replace(month=pqsm,day=1)
    qr = df[(df["Date"]>=pqs)&(df["Date"]<=pqe)]
    if not len(qr): return None
    cur = df[df["Date"]>=cqs]; cO = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(qr["Close"].iloc[-1]))
    return dict(pH=r2(float(qr["High"].max())),pL=r2(float(qr["Low"].min())),
                pC=r2(float(qr["Close"].iloc[-1])),pO=r2(float(qr["Open"].iloc[0])),
                cO=cO,dt=str(qr["Date"].iloc[-1].date()))

def yearly_src(df, today):
    cys = today.replace(month=1,day=1); pye = cys-pd.Timedelta(days=1)
    pys = pye.replace(month=1,day=1)
    yr = df[(df["Date"]>=pys)&(df["Date"]<=pye)]
    if not len(yr): return None
    cur = df[df["Date"]>=cys]; cO = r2(cur["Open"].iloc[0]) if len(cur) else r2(float(yr["Close"].iloc[-1]))
    return dict(pH=r2(float(yr["High"].max())),pL=r2(float(yr["Low"].min())),
                pC=r2(float(yr["Close"].iloc[-1])),pO=r2(float(yr["Open"].iloc[0])),
                cO=cO,dt=str(yr["Date"].iloc[-1].date()))

def ytd_src(df, today):
    jan1 = today.replace(month=1,day=1)
    wsm  = today - pd.Timedelta(days=today.dayofweek)
    last_fri = wsm - pd.Timedelta(days=3)
    ytd = df[(df["Date"]>=jan1)&(df["Date"]<=last_fri)]
    if not len(ytd): ytd = df[df["Date"]>=jan1]
    if not len(ytd): return None
    lfd = df[df["Date"]<=last_fri]
    pC  = r2(float(lfd["Close"].iloc[-1])) if len(lfd) else r2(float(ytd["Close"].iloc[-1]))
    cur = df[df["Date"]>=wsm]; cO = r2(cur["Open"].iloc[0]) if len(cur) else pC
    return dict(pH=r2(float(ytd["High"].max())),pL=r2(float(ytd["Low"].min())),pC=pC,
                pO=r2(float(ytd["Open"].iloc[0])),cO=cO,
                dt=str(lfd["Date"].iloc[-1].date()) if len(lfd) else str(ytd["Date"].iloc[-1].date()))

def build_hists(df, today, nm=HIST_MONTHS, nw=60, nq=12):
    fom = today.replace(day=1); wsm = today - pd.Timedelta(days=today.dayofweek)
    def _rs(past, rule, q=""):
        r = (past.set_index("Date").resample(rule)
                 .agg(pH=("High","max"),pL=("Low","min"),pC=("Close","last"),pO=("Open","first"),_n=("Close","count"))
                 .dropna())
        if q: r = r.query(q)
        return r.drop(columns=["_n"]).reset_index()
    def _rows(d,n): return [[r2(float(row.pH)),r2(float(row.pL)),r2(float(row.pC)),r2(float(row.pO))] for _,row in d.tail(n).iterrows()]
    pm = df[df["Date"]<fom]; pw = df[df["Date"]<wsm]
    return (_rows(_rs(pm,"MS"),nm) if len(pm) else [],
            _rows(_rs(pw,"W-FRI","_n > 0"),nw) if len(pw) else [],
            _rows(_rs(pm,"QS"),nq) if len(pm) else [])

# ── Strategy: Price Action / Smart Money Concepts ─────────────────────────────
def compute_smc(df):
    if len(df) < 30: return {}
    C = df["Close"].values.astype(float); O = df["Open"].values.astype(float)
    H = df["High"].values.astype(float);  L = df["Low"].values.astype(float)
    n = len(df); last = C[-1]

    # Trend
    s50  = C[-50:].mean() if n>=50 else C.mean()
    s200 = C[-200:].mean() if n>=200 else C.mean()
    trend = "bull" if last > s50 > s200 else "bear" if last < s50 < s200 else "range"

    # Order Blocks (last 120 bars)
    lb = min(120, n-5)
    bull_ob = bear_ob = None
    for i in range(n-2, n-lb, -1):
        if C[i] < O[i] and bull_ob is None:           # red candle
            fh = H[i+1:min(i+6,n)].max() if i+1<n else 0
            if fh > H[i]:                               # followed by up-move
                bull_ob = {"h":r2(H[i]),"l":r2(L[i]),"dt":str(df.iloc[i]["Date"].date())}
        if C[i] > O[i] and bear_ob is None:           # green candle
            fl = L[i+1:min(i+6,n)].min() if i+1<n else 1e9
            if fl < L[i]:                               # followed by down-move
                bear_ob = {"h":r2(H[i]),"l":r2(L[i]),"dt":str(df.iloc[i]["Date"].date())}
        if bull_ob and bear_ob: break

    # Fair Value Gaps (last 60 bars, unfilled)
    bull_fvg = []; bear_fvg = []
    for i in range(max(1,n-60), n-1):
        if i+1 >= n: break
        if H[i-1] < L[i+1]:                           # bullish FVG gap
            fh, fl = L[i+1], H[i-1]
            if last > fh:                               # unfilled (price still above)
                bull_fvg.append({"h":r2(fh),"l":r2(fl),"dt":str(df.iloc[i]["Date"].date())})
        if L[i-1] > H[i+1]:                           # bearish FVG gap
            fh, fl = L[i-1], H[i+1]
            if last < fl:                               # unfilled (price still below)
                bear_fvg.append({"h":r2(fh),"l":r2(fl),"dt":str(df.iloc[i]["Date"].date())})
    bull_fvg = bull_fvg[-3:]; bear_fvg = bear_fvg[-3:]

    # Break of Structure: compare recent 20 bars vs prior 20-60 bars
    prior_h = H[max(0,n-60):max(0,n-20)].max() if n>20 else H.max()
    prior_l = L[max(0,n-60):max(0,n-20)].min() if n>20 else L.min()
    bos_bull = bool(last > prior_h)
    bos_bear = bool(last < prior_l)

    # Change of Character (BOS opposite to prevailing trend)
    choch = None
    if trend == "bull" and bos_bear: choch = "bear"
    if trend == "bear" and bos_bull: choch = "bull"

    return dict(bull_ob=bull_ob, bear_ob=bear_ob,
                bull_fvg=bull_fvg, bear_fvg=bear_fvg,
                bos_bull=bos_bull, bos_bear=bos_bear,
                choch=choch, trend=trend)

# ── Strategy: Volume / Institutional Footprint ────────────────────────────────
def compute_vol(df, lookback=252, bins=50):
    if len(df) < 20: return {}
    nn = len(df)
    rn = min(lookback, nn)
    H = df["High"].values[-rn:].astype(float)
    L = df["Low"].values[-rn:].astype(float)
    V = df["Volume"].values[-rn:].astype(float)
    Ca = df["Close"].values.astype(float)
    Va = df["Volume"].values.astype(float)
    Ha = df["High"].values.astype(float)
    La = df["Low"].values.astype(float)

    # Volume Profile
    pmin, pmax = L.min(), H.max()
    if pmax <= pmin: pmax = pmin + 1
    edges = np.linspace(pmin, pmax, bins+1)
    vb = np.zeros(bins)
    for i in range(rn):
        rng = H[i]-L[i]
        if rng <= 0:
            b = min(np.searchsorted(edges[1:], H[i]), bins-1)
            vb[b] += V[i]; continue
        ov = np.maximum(0, np.minimum(H[i], edges[1:]) - np.maximum(L[i], edges[:-1]))
        vb += V[i] * ov / rng

    poc_bin = int(np.argmax(vb))
    poc = (edges[poc_bin] + edges[poc_bin+1]) / 2

    # Value Area (70%)
    tot = vb.sum(); tgt = tot * 0.70
    va_lo = va_hi = poc_bin; va_vol = vb[poc_bin]
    while va_vol < tgt:
        up = vb[va_hi+1] if va_hi+1 < bins else 0
        dn = vb[va_lo-1] if va_lo-1 >= 0   else 0
        if up >= dn and va_hi+1 < bins: va_hi += 1; va_vol += up
        elif va_lo-1 >= 0:              va_lo -= 1; va_vol += dn
        else: break
    vah = edges[va_hi+1]; val = edges[va_lo]

    # Volume ratio (today vs avg 20)
    avg20 = Va[-21:-1].mean() if nn>=21 else Va.mean()
    vr    = round(float(Va[-1]/avg20), 2) if avg20 > 0 else 1.0

    # OBV
    obv_d = np.where(Ca[1:]>Ca[:-1], Va[1:], np.where(Ca[1:]<Ca[:-1], -Va[1:], 0))
    obv   = np.concatenate([[0], np.cumsum(obv_d)])
    w = min(20, nn)
    p_up, o_up = Ca[-1]>Ca[-w], obv[-1]>obv[-w]
    obv_str = "up" if p_up and o_up else "down" if not p_up and not o_up \
              else "div_bull" if o_up else "div_bear"

    # Accumulation / Distribution
    rng_a = Ha - La; rng_a[rng_a==0] = 1
    mfm = ((Ca-La)-(Ha-Ca)) / rng_a
    ad  = np.cumsum(mfm * Va)
    ad_trend = "up" if ad[-1] > ad[-w] else "down"

    return dict(poc=r2(float(poc)), vah=r2(float(vah)), val=r2(float(val)),
                vr=vr, obv=obv_str, ad=ad_trend)

# ── Strategy: Multi-Indicator (Minervini + Weinstein) ────────────────────────
def compute_mi(df):
    if len(df) < 50: return {}
    C = df["Close"].values.astype(float)
    H = df["High"].values.astype(float)
    L = df["Low"].values.astype(float)
    n = len(df); last = C[-1]

    def sma(k): return float(C[-k:].mean()) if n>=k else None

    s50=sma(50); s150=sma(150); s200=sma(200)
    yr=min(252,n); hi52=float(H[-yr:].max()); lo52=float(L[-yr:].min())
    s200_20 = float(C[-(200+20):-200].mean()) if n>=220 else s200
    s200_up  = bool(s200 and s200_20 and s200 > s200_20)

    # Minervini Trend Template (8 conditions)
    mt = [
        bool(s150 and s200 and last>s150 and last>s200),       # 1 price > 150 & 200 SMA
        bool(s150 and s200 and s150>s200),                      # 2 150 SMA > 200 SMA
        s200_up,                                                  # 3 200 SMA trending up
        bool(s50 and s150 and s200 and s50>s150 and s50>s200), # 4 50 > 150 > 200
        bool(s50 and last>s50),                                  # 5 price > 50 SMA
        last >= lo52*1.25,                                        # 6 ≥ 25% above 52W low
        last >= hi52*0.75,                                        # 7 within 25% of 52W high
        False,                                                    # 8 RS ≥ 70 (set after RS pass)
    ]

    # Weinstein Stage (30W = 150 trading days)
    s30w = s150
    s30w_50 = float(C[-(150+50):-150].mean()) if n>=200 else s30w
    ma_up   = bool(s30w and s30w_50 and s30w > s30w_50)
    ma_dn   = bool(s30w and s30w_50 and s30w < s30w_50)
    p_above = bool(s30w and last > s30w)

    if p_above and ma_up:  stg=2; stgl="Stage 2 — Advancing ↑"
    elif p_above:          stg=3; stgl="Stage 3 — Topping →"
    elif not p_above and ma_dn: stg=4; stgl="Stage 4 — Declining ↓"
    else:                  stg=1; stgl="Stage 1 — Basing →"

    return dict(mts=sum(mt), mt=[int(c) for c in mt],
                stg=stg, stgl=stgl,
                s50=r2(s50) if s50 else 0,
                s150=r2(s150) if s150 else 0,
                s200=r2(s200) if s200 else 0,
                s30w=r2(s30w) if s30w else 0)

# ── Stock stats ───────────────────────────────────────────────────────────────
def stock_stats(df, today):
    last = float(df.iloc[-1]["Close"])
    dma200 = float(df["Close"].tail(200).mean()) if len(df)>=50 else last
    yr = df[df["Date"]>=(today-pd.Timedelta(days=365))]
    w52h = float(yr["High"].max()) if len(yr) else float(df["High"].max())
    w52l = float(yr["Low"].min())  if len(yr) else float(df["Low"].min())
    ya   = today - pd.Timedelta(days=365)
    py   = df[df["Date"]<=ya]
    p1y  = float(py.iloc[-1]["Close"]) if len(py) else last
    ret12m = round((last-p1y)/p1y*100, 2) if p1y else 0.0
    avol = int(df["Volume"].tail(20).mean()) if len(df)>=20 else int(df["Volume"].mean())
    return dict(dma200=r2(dma200), above200=last>dma200,
                w52h=r2(w52h), w52l=r2(w52l), ret12m=ret12m, avol=avol)

# ── Per-stock precompute ──────────────────────────────────────────────────────
def precompute(fp, idx_map):
    sym = Path(fp).stem.upper(); df = load_csv(fp)
    if df is None or len(df) < 2: return None
    today = pd.Timestamp(datetime.now().date())
    ds = daily_src(df)
    if ds is None: return None
    ws=weekly_src(df,today) or ds; ms=monthly_src(df,today) or ws
    qs=quarterly_src(df,today) or ms; ys=yearly_src(df,today) or qs
    yts=ytd_src(df,today) or ms
    st = stock_stats(df, today)
    mh, wh, qh = build_hists(df, today)
    smc = compute_smc(df)
    vol = compute_vol(df)
    mi  = compute_mi(df)
    return dict(sym=sym, idx=idx_map.get(sym,0),
                price=r2(float(df.iloc[-1]["Close"])),
                date=str(df.iloc[-1]["Date"].date()),
                d=ds,w=ws,m=ms,q=qs,y=ys,ytd=yts,
                mhist=mh,whist=wh,qhist=qh,
                smc=smc,vol=vol,mi=mi,**st,rs=0)

def assign_rs(stocks):
    rets=sorted(s["ret12m"] for s in stocks); n=len(rets)
    for s in stocks:
        s["rs"] = round(sum(1 for x in rets if x<s["ret12m"])/n*99) if n else 0
        if s.get("mi"):
            s["mi"]["mt"][7] = int(s["rs"]>=70)
            s["mi"]["mts"]   = sum(s["mi"]["mt"])

def build_dataset(data_dir, index_files):
    idx_map = load_index_map(index_files)
    files   = sorted(glob.glob(os.path.join(data_dir,"*.csv")))
    stocks  = []
    for i,fp in enumerate(files,1):
        if i%100==0 or i==len(files): print(f"  [{i}/{len(files)}]…",end="\r")
        rec = precompute(fp, idx_map)
        if rec: stocks.append(rec)
    print(f"\n  Done — {len(stocks)} stocks"); assign_rs(stocks); return stocks

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NSE Strategy Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080b10;--surf:#0d1220;--brd:#182034;
  --acc:#00e5a0;--a2:#3b9eff;--a3:#c47aff;--a4:#ff8c42;
  --warn:#ff8c42;--red:#ff4060;--gold:#f5c518;
  --txt:#cdd5e8;--mu:#4a5c78;--hdr:#060810;
  --mono:'IBM Plex Mono',monospace;--sans:'Space Grotesk',sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
body{background:var(--bg);color:var(--txt);font-family:var(--sans);overflow-y:scroll}

/* ── Tab nav ── */
.tabnav{background:var(--hdr);border-bottom:1px solid var(--brd);
  padding:0 22px;display:flex;align-items:flex-end;gap:2px;position:sticky;top:0;z-index:300}
.logo{font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--acc);
  text-transform:uppercase;white-space:nowrap;padding:14px 14px 14px 0;margin-right:10px;
  border-right:1px solid var(--brd)}
.tab-btn{background:transparent;border:none;border-bottom:2px solid transparent;
  color:var(--mu);font-family:var(--sans);font-size:12px;font-weight:500;
  padding:12px 18px 10px;cursor:pointer;transition:all .2s;white-space:nowrap}
.tab-btn:hover{color:var(--txt)}
.tab-btn.active{color:var(--acc);border-bottom-color:var(--acc)}
.tab-btn.t-smc.active{color:var(--a3);border-bottom-color:var(--a3)}
.tab-btn.t-vol.active{color:var(--a2);border-bottom-color:var(--a2)}
.tab-btn.t-mi.active{color:var(--gold);border-bottom-color:var(--gold)}

/* ── Controls ── */
.ctrl{padding:14px 22px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end}
.ctrl-sep{width:100%;height:1px;background:var(--brd);margin:4px 0 0}
.cg{display:flex;flex-direction:column;gap:4px}
.cg label{font-size:9px;letter-spacing:1.8px;text-transform:uppercase;color:var(--mu);font-family:var(--mono)}
select,input[type=number]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:6px 8px;font-family:var(--mono);font-size:12px;cursor:pointer;
  outline:none;transition:border-color .2s}
select{padding-right:24px;-webkit-appearance:none;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='9' height='9' fill='%234a5c78' viewBox='0 0 16 16'%3E%3Cpath d='M7.247 11.14L2.451 5.658C1.885 5.013 2.345 4 3.204 4h9.592a1 1 0 0 1 .753 1.659l-4.796 5.48a1 1 0 0 1-1.506 0z'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 7px center}
select:hover,select:focus,input[type=number]:hover,input[type=number]:focus{border-color:var(--acc)}
input[type=number]{width:72px}
.prange{display:flex;gap:4px;align-items:center}
.prange span{color:var(--mu);font-size:11px}
.btn{background:var(--acc);color:#000;border:none;border-radius:4px;padding:7px 18px;
  font-family:var(--mono);font-size:11px;font-weight:700;letter-spacing:1.5px;
  text-transform:uppercase;cursor:pointer;align-self:flex-end;transition:opacity .15s,transform .1s;white-space:nowrap}
.btn:hover{opacity:.82;transform:translateY(-1px)}.btn:active{transform:translateY(0)}
.btn-out{background:var(--surf);color:var(--acc);border:1px solid rgba(0,229,160,.4)}
.btn-out:hover{background:rgba(0,229,160,.08)}
.btn-rst{color:var(--mu);border-color:rgba(74,92,120,.5)}
.btn-rst:hover{background:rgba(74,92,120,.1);opacity:1}

/* ── Info bar ── */
.fbar{margin:10px 22px 0;padding:8px 13px;border-radius:5px;
  background:rgba(59,158,255,.04);border:1px solid rgba(59,158,255,.12);
  font-family:var(--mono);font-size:10.5px;color:var(--mu);line-height:1.9}
.fbar b{color:var(--txt)}.fbar .ha{color:var(--acc)}.fbar .hb{color:var(--a2)}
.fbar .hc{color:var(--a3)}.fbar .hd{color:var(--gold)}.fbar code{color:var(--gold)}

/* ── Stats ── */
.stats{padding:10px 22px 0;display:flex;gap:8px;flex-wrap:wrap}
.st{background:var(--surf);border:1px solid var(--brd);border-radius:6px;padding:7px 13px;min-width:88px}
.sl{font-size:8px;letter-spacing:2px;text-transform:uppercase;color:var(--mu);font-family:var(--mono)}
.sv{font-family:var(--mono);font-size:17px;font-weight:700;color:var(--acc);margin-top:2px}
.sv.sm{font-size:11px;margin-top:4px;color:var(--txt);font-weight:400}

/* ── Search row ── */
.sr{padding:9px 22px 0;display:flex;gap:9px;align-items:center;flex-wrap:wrap}
.sr input[type=text]{background:var(--surf);color:var(--txt);border:1px solid var(--brd);
  border-radius:4px;padding:6px 10px;font-family:var(--mono);font-size:12px;outline:none;
  width:170px;transition:border-color .2s}
.sr input[type=text]:focus{border-color:var(--a2)}
.cbl{display:flex;align-items:center;gap:5px;font-size:11.5px;color:var(--mu);cursor:pointer;user-select:none}
.cbl input{accent-color:var(--acc);width:13px;height:13px;cursor:pointer}
.cnt{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.18);color:var(--a2);
  font-family:var(--mono);font-size:10px;padding:2px 8px;border-radius:3px;margin-left:auto}
#saved-lbl{font-family:var(--mono);font-size:9px;color:var(--mu);letter-spacing:1px;margin-left:3px}

/* ── Table ── */
.tw{padding:9px 22px 0}
.ts{overflow-x:auto;overflow-y:auto;max-height:54vh;border:1px solid var(--brd);
  border-radius:8px;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
table{border-collapse:collapse;width:max-content;min-width:100%;font-family:var(--mono);font-size:11.5px}
thead{position:sticky;top:0;z-index:10}
thead tr{background:var(--hdr);border-bottom:2px solid var(--acc)}
th{padding:8px 11px;text-align:left;font-size:8.5px;letter-spacing:1.5px;text-transform:uppercase;
  color:var(--mu);cursor:pointer;white-space:nowrap;user-select:none;border-right:1px solid var(--brd);transition:color .15s}
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
.csym a{color:inherit;text-decoration:none;display:flex;align-items:center;gap:5px}
.csym a:hover{color:var(--acc)}.csym a:hover .tvi{opacity:1}
.tvi{opacity:.3;transition:opacity .15s;flex-shrink:0}
.cpr{color:#dde5f5;font-weight:600}.clv{color:var(--a2);font-weight:700}.cp{color:var(--gold)}
.pos{color:var(--acc)}.neg{color:var(--red)}
.cs1{color:#a0c8ff}.cs2{color:#6aaeff}.cs3{color:#3090ff}.cs4{color:var(--warn)}.cs5{color:var(--red)}
.cr1{color:#a0ffe0}.cr2{color:#60ffcc}.cr3{color:#00ffb0}.cr4{color:var(--red)}.cr5{color:#ff0060}
.mu{color:var(--mu)}
th.dv,td.dv{background:rgba(59,158,255,.03);border-left:1px solid rgba(59,158,255,.1);
  border-right:1px solid rgba(59,158,255,.1);font-size:8px;color:var(--mu);
  text-align:center;padding:6px 4px;cursor:default}
/* Index badges */
.ix50{color:var(--gold)}.ix100{color:var(--acc)}.ix200{color:var(--a2)}
.ix500{color:var(--a3)}.ix750{color:#80a0c0}.ix0{color:var(--mu)}
/* RS */
.rs-hi{color:var(--acc)}.rs-md{color:#80d0a0}.rs-lo{color:var(--warn)}.rs-vl{color:var(--red)}
/* DMA */
.dma-up{color:var(--acc)}.dma-dn{color:var(--red)}
/* Bounce */
.bc-hi{color:var(--acc);font-weight:700}.bc-md{color:#80d080}.bc-lo{color:var(--mu)}
/* SMC */
.smc-bull{color:var(--acc)}.smc-bear{color:var(--red)}.smc-range{color:var(--mu)}
.smc-ob{background:rgba(0,229,160,.08);border:1px solid rgba(0,229,160,.2);
  color:var(--acc);font-size:9px;padding:1px 5px;border-radius:3px}
.smc-ob.bear{background:rgba(255,64,96,.08);border-color:rgba(255,64,96,.2);color:var(--red)}
.smc-fvg{background:rgba(59,158,255,.08);border:1px solid rgba(59,158,255,.2);
  color:var(--a2);font-size:9px;padding:1px 5px;border-radius:3px}
.smc-bos{color:var(--acc);font-weight:700}.smc-choch{color:var(--gold);font-weight:700}
/* Volume */
.vol-poc{color:var(--gold);font-weight:700}
.vol-spike{color:var(--warn);font-weight:700}
.vol-div-bull{color:var(--acc)}.vol-div-bear{color:var(--red)}
/* MI */
.mt-full{color:var(--acc);font-weight:700}.mt-hi{color:#80d0a0}.mt-lo{color:var(--mu)}
.stg-2{color:var(--acc);font-weight:700}
.stg-1{color:var(--a2)}.stg-3{color:var(--warn)}.stg-4{color:var(--red)}
.mt-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin:0 1px}
.mt-on{background:var(--acc)}.mt-off{background:var(--brd)}

.nodata{padding:60px;text-align:center;color:var(--mu);font-family:var(--mono);font-size:13px;line-height:2.2}
.ft{padding:12px 22px 4px;font-size:9.5px;color:var(--mu);font-family:var(--mono);
  display:flex;justify-content:space-between;flex-wrap:wrap;gap:5px;
  border-top:1px solid var(--brd);margin-top:10px}
</style>
</head>
<body>

<!-- ═══ TAB NAV ═══════════════════════════════════════════════════════════ -->
<div class="tabnav">
  <div class="logo">◈ NSE Scanner</div>
  <button class="tab-btn t-piv active" data-tab="piv" onclick="switchTab('piv')">📊 Pivot Points</button>
  <button class="tab-btn t-smc" data-tab="smc" onclick="switchTab('smc')">🎯 Price Action / SMC</button>
  <button class="tab-btn t-vol" data-tab="vol" onclick="switchTab('vol')">📈 Volume / Institutional</button>
  <button class="tab-btn t-mi"  data-tab="mi"  onclick="switchTab('mi')">⚡ Multi-Indicator</button>
</div>

<!-- ═══ PIVOT CONTROLS ════════════════════════════════════════════════════ -->
<div id="ctrl-piv" class="ctrl">
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
  <div class="cg"><label>Level</label><select id="sp-lvl" style="min-width:190px"></select></div>
  <div class="cg"><label>Timeframe</label>
    <select id="sp-tf" style="min-width:320px">
      <option value="d">Daily      · prev trading session H/L/C</option>
      <option value="w">Weekly     · prev Mon–Fri week H/L/C (resampled from daily)</option>
      <option value="m">Monthly    · prev calendar month H/L/C (resampled from daily)</option>
      <option value="q">Quarterly  · prev quarter H/L/C  e.g. Q2 uses Q1 Jan–Mar</option>
      <option value="ytd">YTD      · Jan 1 of current year → last Friday H/L/C</option>
      <option value="y" selected>Yearly   · prev complete calendar year H/L/C (Jan 1 – Dec 31)</option>
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
  <div class="cg"><label>Index</label>
    <select id="idx-flt">
      <option value="0" selected>All stocks</option>
      <option value="50">Nifty 50</option><option value="100">Nifty 100</option>
      <option value="200">Nifty 200</option><option value="500">Nifty 500</option>
      <option value="750">Nifty 750</option>
    </select>
  </div>
  <div class="cg"><label>Trend (200 DMA)</label>
    <select id="dma-flt"><option value="any">Any</option>
      <option value="above">Above 200 DMA ↑</option><option value="below">Below 200 DMA ↓</option>
    </select>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="rs-flt"><option value="0">Any</option>
      <option value="50">≥ 50</option><option value="70">≥ 70</option>
      <option value="80">≥ 80</option><option value="90">≥ 90</option>
    </select>
  </div>
  <div class="cg"><label>Bounces 12M ≥</label>
    <select id="bc-flt"><option value="0">Any</option>
      <option value="1">≥ 1</option><option value="2">≥ 2</option><option value="3">≥ 3</option>
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
  <button class="btn btn-out btn-rst" onclick="resetPrefs()" title="Reset to defaults">↺ RESET</button>
</div>

<!-- ═══ SMC CONTROLS ══════════════════════════════════════════════════════ -->
<div id="ctrl-smc" class="ctrl" style="display:none">
  <div class="cg"><label>Signal Type</label>
    <select id="smc-sig" style="min-width:220px">
      <option value="all">All Signals</option>
      <option value="bull_ob" selected>Bullish Order Block — near support zone</option>
      <option value="bear_ob">Bearish Order Block — near resistance zone</option>
      <option value="bull_fvg">Bullish FVG — price near unfilled gap below</option>
      <option value="bear_fvg">Bearish FVG — price near unfilled gap above</option>
      <option value="bos_bull">BOS Bullish — broke above swing high</option>
      <option value="bos_bear">BOS Bearish — broke below swing low</option>
      <option value="choch_bull">CHoCH Bullish — trend reversal signal up</option>
      <option value="choch_bear">CHoCH Bearish — trend reversal signal down</option>
    </select>
  </div>
  <div class="cg"><label>Proximity ±%</label>
    <select id="smc-prox">
      <option value="1">±1%</option><option value="2" selected>±2%</option>
      <option value="3">±3%</option><option value="5">±5%</option>
    </select>
  </div>
  <div class="cg"><label>Index</label>
    <select id="smc-idx">
      <option value="0" selected>All</option><option value="50">N50</option><option value="100">N100</option>
      <option value="200">N200</option><option value="500">N500</option>
      <option value="750">N750</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="smc-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="smc-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:var(--a3);color:#000" onclick="scanSMC()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ VOLUME CONTROLS ═══════════════════════════════════════════════════ -->
<div id="ctrl-vol" class="ctrl" style="display:none">
  <div class="cg"><label>Signal Type</label>
    <select id="vol-sig" style="min-width:240px">
      <option value="near_poc" selected>Near POC — point of control</option>
      <option value="above_vah">Above VAH — value area breakout</option>
      <option value="below_val">Below VAL — value area breakdown</option>
      <option value="vol_spike">Volume Spike — unusual activity</option>
      <option value="obv_div_bull">OBV Bullish Divergence</option>
      <option value="obv_div_bear">OBV Bearish Divergence</option>
      <option value="ad_up">A/D Line Rising — accumulation</option>
      <option value="all">All Signals</option>
    </select>
  </div>
  <div class="cg"><label>Proximity ±%</label>
    <select id="vol-prox">
      <option value="1">±1%</option><option value="2" selected>±2%</option>
      <option value="3">±3%</option><option value="5">±5%</option>
    </select>
  </div>
  <div class="cg"><label>Vol Spike ≥</label>
    <select id="vol-spike">
      <option value="1.5">1.5× avg</option><option value="2" selected>2× avg</option>
      <option value="3">3× avg</option><option value="5">5× avg</option>
    </select>
  </div>
  <div class="cg"><label>Index</label>
    <select id="vol-idx">
      <option value="0" selected>All</option><option value="50">N50</option><option value="100">N100</option>
      <option value="200">N200</option><option value="500">N500</option>
      <option value="750">N750</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="vol-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="vol-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:var(--a2)" onclick="scanVol()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ MULTI-INDICATOR CONTROLS ══════════════════════════════════════════ -->
<div id="ctrl-mi" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <select id="mi-strat" style="min-width:220px">
      <option value="both" selected>Both — Minervini + Weinstein Stage 2</option>
      <option value="minervini">Minervini Trend Template only</option>
      <option value="weinstein">Weinstein Stage 2 only</option>
      <option value="any">Any strategy match</option>
    </select>
  </div>
  <div class="cg"><label>Min Minervini Score</label>
    <select id="mi-score">
      <option value="4">≥ 4 / 8</option><option value="5">≥ 5 / 8</option>
      <option value="6" selected>≥ 6 / 8</option>
      <option value="7">≥ 7 / 8</option><option value="8">8 / 8 — perfect</option>
    </select>
  </div>
  <div class="cg"><label>Weinstein Stage</label>
    <select id="mi-stg">
      <option value="2" selected>Stage 2 — Advancing ↑</option>
      <option value="1">Stage 1 — Basing →</option>
      <option value="0">Any stage</option>
    </select>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="mi-rs">
      <option value="0">Any</option><option value="50">≥ 50</option>
      <option value="70" selected>≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Index</label>
    <select id="mi-idx">
      <option value="0" selected>All</option><option value="50">N50</option><option value="100">N100</option>
      <option value="200">N200</option><option value="500">N500</option>
      <option value="750">N750</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="mi-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="mi-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:var(--gold);color:#000" onclick="scanMI()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ SHARED: INFO BAR · STATS · SEARCH · TABLE ══════════════════════════ -->
<div class="fbar" id="fbar">Select a tab and click ▶ SCAN</div>

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
  <span>All timeframes resampled from daily CSV data · SMC: Order Blocks, FVGs, BOS, CHoCH · Vol: Volume Profile, OBV, A/D · MI: Minervini 8-condition + Weinstein Stage</span>
  <span>Data: __DD__ · __GT__</span>
</div>

<script>
const S = __JSON__;

// ── Pivot helpers (unchanged) ──────────────────────────────────────────────
const TYPE_META = {
  camarilla:  {label:'Camarilla',  levels:['S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'],hasP:false},
  traditional:{label:'Traditional',levels:['P','S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'],hasP:true},
  classic:    {label:'Classic',    levels:['P','S1','S2','S3','S4','R1','R2','R3','R4'],hasP:true},
  fibonacci:  {label:'Fibonacci',  levels:['P','S1','S2','S3','R1','R2','R3'],hasP:true},
  woodie:     {label:'Woodie',     levels:['P','S1','S2','S3','S4','R1','R2','R3','R4'],hasP:true},
  dm:         {label:'DeMark',     levels:['P','S1','R1'],hasP:true},
  floor:      {label:'Floor',      levels:['P','S1','S2','S3','S4','S5','R1','R2','R3','R4','R5'],hasP:true},
};
const LV_LABEL={P:'P — Pivot',S1:'S1',S2:'S2',S3:'S3',S4:'S4',S5:'S5',R1:'R1',R2:'R2',R3:'R3',R4:'R4',R5:'R5'};
const LV_CSS={P:'cp',S1:'cs1',S2:'cs2',S3:'cs3',S4:'cs4',S5:'cs5',R1:'cr1',R2:'cr2',R3:'cr3',R4:'cr4',R5:'cr5'};
const TYPE_DEF={fibonacci:'S2',traditional:'S1',classic:'S1',camarilla:'S3',woodie:'S1',dm:'S1',floor:'S1'};
const TF_SRC={
  d:'<b>Daily</b> · source = last completed trading session H/L/C',
  w:'<b>Weekly</b> · source = prev Mon–Fri week, resampled from daily CSV',
  m:'<b>Monthly</b> · source = prev calendar month, resampled from daily CSV',
  q:'<b>Quarterly</b> · source = prev complete quarter · Q1=Jan–Mar, Q2=Apr–Jun, Q3=Jul–Sep, Q4=Oct–Dec',
  ytd:'<b>YTD</b> · source = Jan 1 current year → last Friday, resampled from daily CSV',
  y:'<b>Yearly</b> · source = prev complete calendar year (Jan 1 – Dec 31), resampled from daily CSV',
};

function computePivots(type,pH,pL,pC,pO,cO){
  const R=pH-pL;
  if(type==='traditional'||type==='floor'){
    const P=(pH+pL+pC)/3,R3=P*2+(pH-2*pL),S3=P*2-(2*pH-pL);
    return{P,R1:P*2-pL,S1:P*2-pH,R2:P+R,S2:P-R,R3,S3,R4:P*3+(pH-3*pL),S4:P*3-(3*pH-pL),R5:P*4+(pH-4*pL),S5:P*4-(4*pH-pL)};
  }
  if(type==='fibonacci'){const P=(pH+pL+pC)/3;return{P,R1:P+.382*R,S1:P-.382*R,R2:P+.618*R,S2:P-.618*R,R3:P+R,S3:P-R};}
  if(type==='woodie'){const P=(pH+pL+2*cO)/4,R3=pH+2*(P-pL),S3=pL-2*(pH-P);return{P,R1:2*P-pL,S1:2*P-pH,R2:P+R,S2:P-R,R3,S3,R4:R3+R,S4:S3-R};}
  if(type==='classic'){const P=(pH+pL+pC)/3;return{P,R1:2*P-pL,S1:2*P-pH,R2:P+R,S2:P-R,R3:P+2*R,S3:P-2*R,R4:P+3*R,S4:P-3*R};}
  if(type==='camarilla'){const R5=(pH/pL)*pC,S5=pC-(R5-pC);return{R1:pC+1.1*R/12,S1:pC-1.1*R/12,R2:pC+1.1*R/6,S2:pC-1.1*R/6,R3:pC+1.1*R/4,S3:pC-1.1*R/4,R4:pC+1.1*R/2,S4:pC-1.1*R/2,R5,S5};}
  if(type==='dm'){let X=pO===pC?pH+pL+2*pC:pC>pO?2*pH+pL+pC:2*pL+pH+pC;return{P:X/4,R1:X/2-pL,S1:X/2-pH};}
  return{};
}

function bounceCount(mh,wh,qh,tf,type,lv,periods){
  const h=tf==='w'?wh:(tf==='q'||tf==='y')?qh:mh;
  const sl=h.slice(-(periods+1));const isS=lv.startsWith('S')||lv==='P';
  let n=0;
  for(let i=1;i<sl.length;i++){
    const[pH,pL,pC,pO]=sl[i-1];const c=sl[i];
    const p=computePivots(type,pH,pL,pC,pO,c[3]);const t=p[lv];
    if(t===undefined)continue;
    if(c[1]<=t&&t<=c[0])if(isS?c[2]>=t:c[2]<=t)n++;
  }
  return n;
}

// ── State ──────────────────────────────────────────────────────────────────
let rows=[],sc=4,sd=1,currentTab='piv',lastRows=[];

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab){
  currentTab=tab;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  ['piv','smc','vol','mi'].forEach(t=>{
    const el=document.getElementById('ctrl-'+t);
    if(el) el.style.display=t===tab?'flex':'none';
  });
  document.getElementById('ts').innerHTML='<div class="nodata">Click ▶ SCAN to find stocks</div>';
  if(tab==='piv') scan();
  else if(tab==='smc') scanSMC();
  else if(tab==='vol') scanVol();
  else if(tab==='mi') scanMI();
}

// ── Level dropdown (pivot) ─────────────────────────────────────────────────
function onTypeChange(forceLv){
  const type=document.getElementById('sp-type').value;
  const meta=TYPE_META[type];const sel=document.getElementById('sp-lvl');
  const cur=forceLv||sel.value;const def=TYPE_DEF[type]||meta.levels[0];
  sel.innerHTML='';
  const ag=document.createElement('optgroup');ag.label='── Multi-level scan ──';
  [['__ANY_S__','★ Any Support (nearest S)'],['__ANY_R__','★ Any Resistance (nearest R)'],['__ANY__','★ Any Level']].forEach(([v,t])=>{
    const o=document.createElement('option');o.value=v;o.textContent=t;if(cur===v)o.selected=true;ag.appendChild(o);
  });
  sel.appendChild(ag);
  [[meta.levels.filter(l=>l==='P'||l.startsWith('S')),'── Support / Pivot ──'],
   [meta.levels.filter(l=>l.startsWith('R')),'── Resistance ──']].forEach(([lvs,lbl])=>{
    if(!lvs.length)return;
    const g=document.createElement('optgroup');g.label=lbl;
    lvs.forEach(l=>{const o=document.createElement('option');o.value=l;o.textContent=LV_LABEL[l]||l;
      if(cur?l===cur:l===def)o.selected=true;g.appendChild(o);});
    sel.appendChild(g);
  });
  updateFbar();
}
function updateFbar(){
  const type=document.getElementById('sp-type').value;
  const tf=document.getElementById('sp-tf').value;
  const FBAR={
    camarilla:`<b>Camarilla</b> · <span class="ha">Sn=prevC∓Range×1.1/k</span> · k:1=12,2=6,3=4,4=2`,
    traditional:`<b>Traditional</b> · <code>P=(H+L+C)/3</code> · R1=2P−L, S1=2P−H`,
    classic:`<b>Classic</b> · <code>P=(H+L+C)/3</code> · Rn/Sn=P±n×Range`,
    fibonacci:`<b>Fibonacci</b> · <code>P=(H+L+C)/3</code> · ×0.382, ×0.618, ×1.0`,
    woodie:`<b>Woodie</b> · <code>P=(H+L+2×currO)/4</code> · uses current period open`,
    dm:`<b>DeMark</b> · X depends on O vs C · P=X/4`,
    floor:`<b>Floor</b> (=Traditional) · <code>P=(H+L+C)/3</code>`,
  };
  document.getElementById('fbar').innerHTML=
    (FBAR[type]||'')+' &nbsp;·&nbsp; <span class="hb">Source:</span> '+(TF_SRC[tf]||tf);
}

// ── localStorage ───────────────────────────────────────────────────────────
const DEFAULTS={type:'fibonacci',lv:'S2',tf:'y',pr:'2',idx:'0',dma:'any',rs:'0',bc:'0',prMin:'',prMax:'',cvol:false,c52h:false,c52l:false};
const PK='nse_scanner_v1';
function applyPrefs(p){
  if(p.type) document.getElementById('sp-type').value=p.type;
  if(p.tf)   document.getElementById('sp-tf').value=p.tf;
  if(p.pr)   document.getElementById('sp-pr').value=p.pr;
  if(p.idx)  document.getElementById('idx-flt').value=p.idx;
  if(p.dma)  document.getElementById('dma-flt').value=p.dma;
  if(p.rs)   document.getElementById('rs-flt').value=p.rs;
  if(p.bc)   document.getElementById('bc-flt').value=p.bc;
  document.getElementById('pr-min').value=p.prMin||'';
  document.getElementById('pr-max').value=p.prMax||'';
  document.getElementById('cb-vol').checked=!!p.cvol;
  document.getElementById('cb-52h').checked=!!p.c52h;
  document.getElementById('cb-52l').checked=!!p.c52l;
  onTypeChange(p.lv||'');
}
function savePrefs(){
  try{
    localStorage.setItem(PK,JSON.stringify({
      type:document.getElementById('sp-type').value,
      lv:document.getElementById('sp-lvl').value,
      tf:document.getElementById('sp-tf').value,
      pr:document.getElementById('sp-pr').value,
      idx:document.getElementById('idx-flt').value,
      dma:document.getElementById('dma-flt').value,
      rs:document.getElementById('rs-flt').value,
      bc:document.getElementById('bc-flt').value,
      prMin:document.getElementById('pr-min').value,
      prMax:document.getElementById('pr-max').value,
      cvol:document.getElementById('cb-vol').checked,
      c52h:document.getElementById('cb-52h').checked,
      c52l:document.getElementById('cb-52l').checked,
    }));
    const l=document.getElementById('saved-lbl');
    l.textContent='✓ SAVED';setTimeout(()=>l.textContent='',1800);
  }catch(e){}
}
function loadPrefs(){try{return JSON.parse(localStorage.getItem(PK));}catch(e){return null;}}
function resetPrefs(){try{localStorage.removeItem(PK);}catch(e){}applyPrefs(DEFAULTS);const l=document.getElementById('saved-lbl');l.textContent='↺ RESET';setTimeout(()=>l.textContent='',1800);scan();}

// ── Shared helpers ─────────────────────────────────────────────────────────
function symCell(sym){return`<td class="csym"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${sym}" target="_blank" rel="noopener"><svg class="tvi" width="12" height="12" viewBox="0 0 28 28" fill="none"><rect width="28" height="28" rx="6" fill="#131722"/><path d="M4 20h4v-8H4v8zm6 0h4V8h-4v12zm6 0h4v-5h-4v5z" fill="#2962FF"/></svg>${sym}</a></td>`;}
function idxBadge(idx){const m={50:['ix50','N50'],100:['ix100','N100'],200:['ix200','N200'],500:['ix500','N500'],750:['ix750','N750'],0:['ix0','—']};const[c,l]=m[idx]||m[0];return`<td class="${c}">${l}</td>`;}
function rsCell(rs){const c=rs>=80?'rs-hi':rs>=60?'rs-md':rs>=40?'rs-lo':'rs-vl';return`<td class="${c}">${rs}</td>`;}
function dmaCell(price,dma,above){const p=((price-dma)/dma*100).toFixed(1);return`<td class="${above?'dma-up':'dma-dn'}">${above?'▲':'▼'} ${p}%</td>`;}
function distCell(dist,prox){const f=Math.abs(dist)/prox;const bg=f<=.25?'rgba(0,229,160,.14)':f<=.55?'rgba(0,229,160,.06)':f<=.80?'rgba(245,200,66,.06)':'rgba(255,64,96,.06)';return`<td class="${f<=.55?'pos':f>.80?'neg':''}" style="background:${bg}">${dist>=0?'+':''}${dist.toFixed(2)}%</td>`;}
function fmtVol(v){return v>=1e7?`<span style="color:var(--acc);font-size:9px">▲ </span>${(v/1e7).toFixed(1)}Cr`:v>=1e5?`${(v/1e5).toFixed(1)}L`:v.toLocaleString();}
function w52Cell(price,ref){const p=((price-ref)/ref*100).toFixed(1);const near=Math.abs(p)<5;return`<td style="${near?'font-weight:700;color:var(--gold)':'color:var(--mu)'}">${p>=0?'+':''}${p}%</td>`;}
function mtDots(mt){return mt.map((b,i)=>`<span class="mt-dot ${b?'mt-on':'mt-off'}" title="Condition ${i+1}: ${b?'✓':'✗'}"></span>`).join('');}

function rowFilter(r){
  const q=document.getElementById('q').value.trim().toUpperCase();
  if(q&&!r.sym.includes(q))return false;
  if(document.getElementById('cb-vol').checked&&r.avol<100000)return false;
  if(document.getElementById('cb-52h').checked&&Math.abs((r.price-r.w52h)/r.w52h*100)>5)return false;
  if(document.getElementById('cb-52l').checked&&Math.abs((r.price-r.w52l)/r.w52l*100)>5)return false;
  return true;
}
function updateStats(vis){
  document.getElementById('st-h').textContent=vis.length;
  document.getElementById('st-up').textContent=vis.filter(r=>r.above200).length;
  document.getElementById('st-dn').textContent=vis.filter(r=>!r.above200).length;
  document.getElementById('st-rs').textContent=vis.filter(r=>r.rs>=70).length;
  document.getElementById('st-u').textContent=S.length;
  document.getElementById('vc').textContent=vis.length+' rows';
}

// ── PIVOT SCAN ─────────────────────────────────────────────────────────────
function lvelsForKey(type,key){
  const m=TYPE_META[type];
  if(key==='__ANY_S__')return m.levels.filter(l=>l!=='P'&&l.startsWith('S'));
  if(key==='__ANY_R__')return m.levels.filter(l=>l.startsWith('R'));
  if(key==='__ANY__')  return m.levels.filter(l=>l!=='P');
  return[key];
}

function scan(){
  const type=document.getElementById('sp-type').value;
  const lvKey=document.getElementById('sp-lvl').value;
  const tf=document.getElementById('sp-tf').value;
  const prox=parseFloat(document.getElementById('sp-pr').value);
  const idxF=parseInt(document.getElementById('idx-flt').value);
  const dmaF=document.getElementById('dma-flt').value;
  const rsF=parseInt(document.getElementById('rs-flt').value);
  const bcF=parseInt(document.getElementById('bc-flt').value);
  const prMin=parseFloat(document.getElementById('pr-min').value)||0;
  const prMax=parseFloat(document.getElementById('pr-max').value)||Infinity;
  const meta=TYPE_META[type];const multi=lvKey.startsWith('__ANY');
  const lvels=lvelsForKey(type,lvKey);
  updateFbar();savePrefs();
  rows=[];
  for(const s of S){
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(dmaF==='above'&&!s.above200)continue;
    if(dmaF==='below'&&s.above200)continue;
    if(s.rs<rsF)continue;
    const src={d:s.d,w:s.w,m:s.m,q:s.q,ytd:s.ytd,y:s.y}[tf]||s.m;
    const p=computePivots(type,src.pH,src.pL,src.pC,src.pO,src.cO);
    let bestTgt=null,bestDist=Infinity,bestLv=null;
    for(const lv of lvels){const t=p[lv];if(t===undefined)continue;const d=(s.price-t)/Math.abs(t)*100;if(Math.abs(d)<Math.abs(bestDist)){bestDist=d;bestTgt=t;bestLv=lv;}}
    if(bestTgt===null||Math.abs(bestDist)>prox)continue;
    const b6=bounceCount(s.mhist,s.whist,s.qhist,tf,type,bestLv,6);
    const b12=bounceCount(s.mhist,s.whist,s.qhist,tf,type,bestLv,12);
    if(b12<bcF)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,above200:s.above200,rs:s.rs,
      dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      lv:+bestTgt.toFixed(2),dist:+bestDist.toFixed(3),lvKey:bestLv,p,multi,
      sh:src.pH,sl2:src.pL,sc2:src.pC,so:src.pO,sd:src.dt,b6,b12,_tab:'piv',type,meta});
  }
  sc=4;sd=1;rows.sort((a,b)=>Math.abs(a.dist)-Math.abs(b.dist));render();
}

// ── SMC SCAN ───────────────────────────────────────────────────────────────
function scanSMC(){
  const sig=document.getElementById('smc-sig').value;
  const prox=parseFloat(document.getElementById('smc-prox').value)/100;
  const idxF=parseInt(document.getElementById('smc-idx').value);
  const prMin=parseFloat(document.getElementById('smc-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('smc-pmax').value)||Infinity;

  const FBAR_LABELS={
    all:'All SMC signals — Order Blocks, FVGs, BOS, CHoCH',
    bull_ob:'<span class="ha">Bullish Order Block</span> — last bearish candle before an up-move; price near support zone',
    bear_ob:'<span class="hc">Bearish Order Block</span> — last bullish candle before a down-move; price near resistance zone',
    bull_fvg:'<span class="ha">Bullish FVG</span> — unfilled price gap (imbalance); price above gap, likely to revisit',
    bear_fvg:'<span class="hc">Bearish FVG</span> — unfilled price gap (imbalance); price below gap, likely to revisit',
    bos_bull:'<span class="ha">BOS Bullish</span> — Break of Structure upward; price broke above recent swing high → bullish continuation',
    bos_bear:'<span class="hc">BOS Bearish</span> — Break of Structure downward; price broke below recent swing low → bearish continuation',
    choch_bull:'<span class="hd">CHoCH Bullish</span> — Change of Character; bearish trend broke upward → potential reversal',
    choch_bear:'<span class="hd">CHoCH Bearish</span> — Change of Character; bullish trend broke downward → potential reversal',
  };
  document.getElementById('fbar').innerHTML='<b>Price Action / Smart Money Concepts</b> · '+FBAR_LABELS[sig];
  rows=[];
  for(const s of S){
    if(!s.smc||Object.keys(s.smc).length===0)continue;
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
    if(s.price<prMin||s.price>prMax)continue;
    const smc=s.smc;
    let matched=false,signal='',zone_h=0,zone_l=0;

    const near=(ref,prx=prox)=>Math.abs(s.price-ref)/ref<=prx;
    const nearZone=(h,l,prx=prox)=>{
      const mid=(h+l)/2;return Math.abs(s.price-mid)/mid<=prx||
        (s.price>=l*(1-prx)&&s.price<=h*(1+prx));
    };

    if(sig==='bull_ob'||sig==='all'){
      if(smc.bull_ob&&nearZone(smc.bull_ob.h,smc.bull_ob.l)){
        matched=true;signal='Bull OB';zone_h=smc.bull_ob.h;zone_l=smc.bull_ob.l;
      }
    }
    if(!matched&&(sig==='bear_ob'||sig==='all')){
      if(smc.bear_ob&&nearZone(smc.bear_ob.h,smc.bear_ob.l)){
        matched=true;signal='Bear OB';zone_h=smc.bear_ob.h;zone_l=smc.bear_ob.l;
      }
    }
    if(!matched&&(sig==='bull_fvg'||sig==='all')){
      const fvg=smc.bull_fvg&&smc.bull_fvg.find(f=>nearZone(f.h,f.l));
      if(fvg){matched=true;signal='Bull FVG';zone_h=fvg.h;zone_l=fvg.l;}
    }
    if(!matched&&(sig==='bear_fvg'||sig==='all')){
      const fvg=smc.bear_fvg&&smc.bear_fvg.find(f=>nearZone(f.h,f.l));
      if(fvg){matched=true;signal='Bear FVG';zone_h=fvg.h;zone_l=fvg.l;}
    }
    if(!matched&&(sig==='bos_bull'||sig==='all')&&smc.bos_bull){matched=true;signal='BOS ↑';}
    if(!matched&&(sig==='bos_bear'||sig==='all')&&smc.bos_bear){matched=true;signal='BOS ↓';}
    if(!matched&&(sig==='choch_bull'||sig==='all')&&smc.choch==='bull'){matched=true;signal='CHoCH ↑';}
    if(!matched&&(sig==='choch_bear'||sig==='all')&&smc.choch==='bear'){matched=true;signal='CHoCH ↓';}

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,above200:s.above200,rs:s.rs,
      dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      signal,zone_h,zone_l,smc:s.smc,_tab:'smc'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── VOLUME SCAN ────────────────────────────────────────────────────────────
function scanVol(){
  const sig=document.getElementById('vol-sig').value;
  const prox=parseFloat(document.getElementById('vol-prox').value)/100;
  const spk=parseFloat(document.getElementById('vol-spike').value);
  const idxF=parseInt(document.getElementById('vol-idx').value);
  const prMin=parseFloat(document.getElementById('vol-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('vol-pmax').value)||Infinity;

  const FBAR_VOL={
    near_poc:'<span class="hb">Near POC</span> — price near Point of Control (price level with highest traded volume)',
    above_vah:'<span class="ha">Above VAH</span> — price broke above Value Area High; bullish breakout signal',
    below_val:'<span class="hc">Below VAL</span> — price below Value Area Low; bearish breakdown signal',
    vol_spike:'<span class="hd">Volume Spike</span> — unusual volume ≥ '+spk+'× average; potential institutional activity',
    obv_div_bull:'<span class="ha">OBV Bullish Divergence</span> — OBV rising while price falling; hidden accumulation',
    obv_div_bear:'<span class="hc">OBV Bearish Divergence</span> — OBV falling while price rising; hidden distribution',
    ad_up:'<span class="ha">A/D Rising</span> — Accumulation/Distribution line trending up; institutional buying',
    all:'All volume signals — POC/VAH/VAL/Spike/OBV divergence',
  };
  document.getElementById('fbar').innerHTML='<b>Volume / Institutional Footprint</b> · '+(FBAR_VOL[sig]||sig);
  rows=[];
  for(const s of S){
    if(!s.vol||Object.keys(s.vol).length===0)continue;
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
    if(s.price<prMin||s.price>prMax)continue;
    const v=s.vol;
    const near=(ref)=>Math.abs(s.price-ref)/ref<=prox;
    let matched=false,vsig='';

    if(sig==='near_poc'||sig==='all'){if(v.poc&&near(v.poc)){matched=true;vsig='Near POC';}}
    if(!matched&&(sig==='above_vah'||sig==='all')){if(v.vah&&s.price>v.vah&&near(v.vah)){matched=true;vsig='Above VAH ▲';}}
    if(!matched&&(sig==='below_val'||sig==='all')){if(v.val&&s.price<v.val&&near(v.val)){matched=true;vsig='Below VAL ▼';}}
    if(!matched&&(sig==='vol_spike'||sig==='all')){if(v.vr>=spk){matched=true;vsig=`Vol ${v.vr}× spike`;}}
    if(!matched&&(sig==='obv_div_bull'||sig==='all')){if(v.obv==='div_bull'){matched=true;vsig='OBV div ↑';}}
    if(!matched&&(sig==='obv_div_bear'||sig==='all')){if(v.obv==='div_bear'){matched=true;vsig='OBV div ↓';}}
    if(!matched&&(sig==='ad_up'||sig==='all')){if(v.ad==='up'){matched=true;vsig='A/D rising';}}

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,above200:s.above200,rs:s.rs,
      dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      vsig,poc:v.poc,vah:v.vah,val:v.val,vr:v.vr,obv:v.obv,ad:v.ad,_tab:'vol'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── MULTI-INDICATOR SCAN ───────────────────────────────────────────────────
function scanMI(){
  const strat=document.getElementById('mi-strat').value;
  const minScore=parseInt(document.getElementById('mi-score').value);
  const stgReq=parseInt(document.getElementById('mi-stg').value);
  const rsMin=parseInt(document.getElementById('mi-rs').value);
  const idxF=parseInt(document.getElementById('mi-idx').value);
  const prMin=parseFloat(document.getElementById('mi-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('mi-pmax').value)||Infinity;

  const MT_LABELS=['Price > 150 & 200 SMA','150 SMA > 200 SMA','200 SMA rising','50 SMA > 150 & 200 SMA','Price > 50 SMA','≥ 25% above 52W low','Within 25% of 52W high','RS ≥ 70'];
  document.getElementById('fbar').innerHTML=
    '<b>Minervini Trend Template</b> · 8 conditions: '+ MT_LABELS.map((l,i)=>`<span style="color:var(--mu)">${i+1}.${l}</span>`).join(' · ')+'<br>'+
    '<b>Weinstein Stage Analysis</b> · Stage 2 (Advancing): price above rising 30-week (150-day) SMA';
  rows=[];
  for(const s of S){
    if(!s.mi||Object.keys(s.mi).length===0)continue;
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    const mi=s.mi;
    const mOk=(strat==='minervini'||strat==='both'||strat==='any')&&mi.mts>=minScore;
    const wOk=(strat==='weinstein'||strat==='both'||strat==='any')&&(stgReq===0||mi.stg===stgReq);
    let pass=false;
    if(strat==='both')      pass=mOk&&wOk;
    else if(strat==='any')  pass=mOk||wOk;
    else if(strat==='minervini') pass=mOk;
    else if(strat==='weinstein') pass=wOk;
    if(!pass)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,above200:s.above200,rs:s.rs,
      dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      mts:mi.mts,mt:mi.mt,stg:mi.stg,stgl:mi.stgl,
      s50:mi.s50,s150:mi.s150,s200:mi.s200,s30w:mi.s30w,_tab:'mi'});
  }
  sc=7;sd=-1;rows.sort((a,b)=>(b.mts-a.mts)||a.sym.localeCompare(b.sym));render();
}

// ── Render (tab-aware columns) ─────────────────────────────────────────────
function buildCols(tab,r0){
  const common=[
    {k:'sym',  h:'Symbol',     fn:r=>symCell(r.sym)},
    {k:'idx',  h:'Index',      fn:r=>idxBadge(r.idx)},
    {k:'price',h:'Close',      fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
    {k:'rs',   h:'RS',         fn:r=>rsCell(r.rs)},
    {k:'above200',h:'vs 200DMA',fn:r=>dmaCell(r.price,r.dma200,r.above200)},
    {k:'w52h', h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
    {k:'avol', h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
    {k:'date', h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
  ];

  if(tab==='piv'){
    const prox=parseFloat(document.getElementById('sp-pr').value);
    const type=r0?.type||'fibonacci';const meta=r0?.meta||TYPE_META[type];
    const lvKey=r0?.lvKey;const multi=r0?.multi;
    const lvList=(meta.levels||[]).filter(l=>l!==lvKey);
    return[
      {k:'sym',h:'Symbol',fn:r=>symCell(r.sym),ns:true},
      {k:'idx',h:'Index',fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'lv',h:(multi?'Hit Lv':lvKey)+' Level',fn:r=>`<td class="clv">${multi?`<span class="${LV_CSS[r.lvKey]||''}">${r.lvKey} </span>`:''  }${r.lv.toFixed(2)}</td>`,hcls:'hl'},
      {k:'dist',h:'Dist %',fn:r=>distCell(r.dist,prox)},
      {k:'rs',h:'RS',fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'b6',h:'Bnc 6M',fn:r=>`<td class="${r.b6>=3?'bc-hi':r.b6>=2?'bc-md':'bc-lo'}">${r.b6}</td>`},
      {k:'b12',h:'Bnc 12M',fn:r=>`<td class="${r.b12>=3?'bc-hi':r.b12>=2?'bc-md':'bc-lo'}">${r.b12}</td>`},
      {k:'_d1',h:'Levels',fn:r=>`<td class="dv">◀▶</td>`,dv:true},
      ...(meta.hasP&&lvKey!=='P'?[{k:'P',h:'P',fn:r=>`<td class="cp">${(r.p.P??0).toFixed(2)}</td>`}]:[]),
      ...lvList.map(l=>({k:l,h:l,fn:r=>`<td class="${LV_CSS[l]||'mu'}">${r.p[l]!==undefined?r.p[l].toFixed(2):'—'}</td>`})),
      {k:'_d2',h:'Src',fn:r=>`<td class="dv">◀▶</td>`,dv:true},
      {k:'sh',h:'Src H',fn:r=>`<td class="mu">${r.sh}</td>`},
      {k:'sl2',h:'Src L',fn:r=>`<td class="mu">${r.sl2}</td>`},
      {k:'sc2',h:'Src C',fn:r=>`<td class="mu">${r.sc2}</td>`},
      {k:'sd',h:'Src Date',fn:r=>`<td class="mu">${r.sd}</td>`},
      {k:'avol',h:'AvgVol20',fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }

  if(tab==='smc'){
    return[
      {k:'sym',h:'Symbol',fn:r=>symCell(r.sym)},
      {k:'idx',h:'Index',fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'signal',h:'Signal',fn:r=>{
        const isB=r.signal.includes('Bull')||r.signal.includes('↑');
        const cls=r.signal.includes('CHoCH')?'smc-choch':r.signal.includes('BOS')?'smc-bos':isB?'smc-bull':'smc-bear';
        return`<td><span class="${r.signal.includes('OB')?'smc-ob '+(isB?'':'bear'):r.signal.includes('FVG')?'smc-fvg':cls}">${r.signal}</span></td>`;
      }},
      {k:'zone_h',h:'Zone High',fn:r=>`<td class="${r.zone_h?'cpr':'mu'}">${r.zone_h||'—'}</td>`},
      {k:'zone_l',h:'Zone Low', fn:r=>`<td class="${r.zone_l?'cpr':'mu'}">${r.zone_l||'—'}</td>`},
      {k:'trend',h:'Trend',fn:r=>`<td class="${r.smc.trend==='bull'?'smc-bull':r.smc.trend==='bear'?'smc-bear':'smc-range'}">${r.smc.trend?.toUpperCase()||'—'}</td>`},
      {k:'bos_bull',h:'BOS',fn:r=>`<td>${r.smc.bos_bull?'<span class="smc-bull">BOS↑</span>':r.smc.bos_bear?'<span class="smc-bear">BOS↓</span>':'<span class="mu">—</span>'}</td>`},
      {k:'choch',h:'CHoCH',fn:r=>`<td>${r.smc.choch==='bull'?'<span class="smc-choch">CHoCH↑</span>':r.smc.choch==='bear'?'<span class="smc-choch">CHoCH↓</span>':'<span class="mu">—</span>'}</td>`},
      {k:'bull_ob_h',h:'Bull OB H',fn:r=>`<td class="mu">${r.smc.bull_ob?r.smc.bull_ob.h:'—'}</td>`},
      {k:'bull_ob_l',h:'Bull OB L',fn:r=>`<td class="mu">${r.smc.bull_ob?r.smc.bull_ob.l:'—'}</td>`},
      {k:'bear_ob_h',h:'Bear OB H',fn:r=>`<td class="mu">${r.smc.bear_ob?r.smc.bear_ob.h:'—'}</td>`},
      {k:'bear_ob_l',h:'Bear OB L',fn:r=>`<td class="mu">${r.smc.bear_ob?r.smc.bear_ob.l:'—'}</td>`},
      {k:'fvg',h:'FVGs',fn:r=>`<td class="mu">${(r.smc.bull_fvg||[]).length}↑ ${(r.smc.bear_fvg||[]).length}↓</td>`},
      ...common.slice(3),
    ];
  }

  if(tab==='vol'){
    return[
      {k:'sym',h:'Symbol',fn:r=>symCell(r.sym)},
      {k:'idx',h:'Index',fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'vsig',h:'Signal',fn:r=>`<td class="${r.vsig.includes('POC')?'vol-poc':r.vsig.includes('spike')?'vol-spike':r.vsig.includes('↑')||r.vsig.includes('div ↑')||r.vsig.includes('rising')?'pos':'neg'}">${r.vsig}</td>`},
      {k:'poc',h:'POC',fn:r=>`<td class="vol-poc">${r.poc||'—'}</td>`},
      {k:'vah',h:'VAH',fn:r=>`<td class="pos">${r.vah||'—'}</td>`},
      {k:'val',h:'VAL',fn:r=>`<td class="neg">${r.val||'—'}</td>`},
      {k:'vr',h:'Vol Ratio',fn:r=>`<td class="${r.vr>=3?'vol-spike':r.vr>=2?'vol-spike':r.vr>=1.5?'pos':''}">${r.vr}×</td>`},
      {k:'obv',h:'OBV',fn:r=>`<td class="${r.obv==='up'?'pos':r.obv==='down'?'neg':r.obv==='div_bull'?'vol-div-bull':'vol-div-bear'}">${r.obv==='up'?'↑ Up':r.obv==='down'?'↓ Down':r.obv==='div_bull'?'⚡ Div↑':'⚡ Div↓'}</td>`},
      {k:'ad',h:'A/D',fn:r=>`<td class="${r.ad==='up'?'pos':'neg'}">${r.ad==='up'?'↑ Accum':'↓ Distrib'}</td>`},
      ...common.slice(3),
    ];
  }

  if(tab==='mi'){
    return[
      {k:'sym',h:'Symbol',fn:r=>symCell(r.sym)},
      {k:'idx',h:'Index',fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'mts',h:'Score',fn:r=>`<td class="${r.mts===8?'mt-full':r.mts>=6?'mt-hi':'mt-lo'}" title="Minervini score ${r.mts}/8">${r.mts}/8</td>`},
      {k:'mt',h:'8 Conditions',fn:r=>`<td>${mtDots(r.mt)}</td>`},
      {k:'stg',h:'Weinstein',fn:r=>`<td class="stg-${r.stg}">${r.stgl}</td>`},
      {k:'rs',h:'RS',fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'s50',h:'50 SMA',fn:r=>`<td class="${r.price>r.s50?'pos':'neg'}">${r.s50}</td>`},
      {k:'s150',h:'150 SMA',fn:r=>`<td class="${r.price>r.s150?'pos':'neg'}">${r.s150}</td>`},
      {k:'s200',h:'200 SMA',fn:r=>`<td class="${r.price>r.s200?'pos':'neg'}">${r.s200}</td>`},
      {k:'s30w',h:'30W SMA',fn:r=>`<td class="${r.price>r.s30w?'pos':'neg'}">${r.s30w}</td>`},
      {k:'w52h',h:'vs 52WH',fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',h:'AvgVol20',fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  return common;
}

function render(){
  const vis=rows.filter(rowFilter);
  updateStats(vis);
  const tab=rows.length?rows[0]._tab:currentTab;
  const cs=buildCols(tab,rows[0]);
  if(!vis.length){document.getElementById('ts').innerHTML='<div class="nodata">No stocks found. Adjust filters and try again.</div>';return;}
  const ths=cs.map((c,i)=>{
    const cls=[c.hcls||'',c.dv?'dv':'',(!c.dv&&i===sc)?(sd===1?'asc':'desc'):''].filter(Boolean).join(' ');
    return`<th class="${cls}"${c.dv?'':` data-i="${i}"`}>${c.h}</th>`;
  }).join('');
  const trs=vis.map(r=>`<tr${r.idx>0&&r.idx<=750?' class="nr"':''}>${cs.map(c=>c.fn(r)).join('')}</tr>`).join('');
  document.getElementById('ts').innerHTML=`<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
  document.querySelectorAll('#ts th[data-i]').forEach(th=>{
    th.addEventListener('click',()=>{
      const i=+th.dataset.i;if(sc===i)sd*=-1;else{sc=i;sd=1;}
      const k=cs[i].k;
      rows.sort((a,b)=>(typeof a[k]==='number'?a[k]-b[k]:String(a[k]).localeCompare(String(b[k])))*sd);
      render();
    });
  });
}

// ── Export CSV ─────────────────────────────────────────────────────────────
function exportCSV(){
  if(!rows.length)return;
  const tab=rows[0]._tab;const vis=rows.filter(rowFilter);
  const hdrs={
    piv:['Symbol','Index','Close','Level','Dist%','RS','vs200DMA','Bnc6M','Bnc12M','SrcH','SrcL','SrcC','SrcDate','AvgVol20','LastDate'],
    smc:['Symbol','Index','Close','Signal','ZoneH','ZoneL','Trend','BOS_Bull','BOS_Bear','CHoCH','BullOB_H','BullOB_L','BearOB_H','BearOB_L','RS','vs200DMA','AvgVol20','LastDate'],
    vol:['Symbol','Index','Close','Signal','POC','VAH','VAL','VolRatio','OBV','AD','RS','vs200DMA','AvgVol20','LastDate'],
    mi: ['Symbol','Index','Close','MinerviniScore','WeinStage','StageLabel','RS','vs200DMA','SMA50','SMA150','SMA200','SMA30W','AvgVol20','LastDate'],
  };
  const cells={
    piv:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.lv,`${r.dist>=0?'+':''}${r.dist.toFixed(2)}%`,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.b6,r.b12,r.sh,r.sl2,r.sc2,r.sd,r.avol,r.date],
    smc:r=>[r.sym,r.idx||'Other',r.price,r.signal,r.zone_h,r.zone_l,r.smc.trend,r.smc.bos_bull?'Y':'N',r.smc.bos_bear?'Y':'N',r.smc.choch||'',r.smc.bull_ob?.h||'',r.smc.bull_ob?.l||'',r.smc.bear_ob?.h||'',r.smc.bear_ob?.l||'',r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    vol:r=>[r.sym,r.idx||'Other',r.price,r.vsig,r.poc,r.vah,r.val,r.vr,r.obv,r.ad,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    mi: r=>[r.sym,r.idx||'Other',r.price,`${r.mts}/8`,r.stg,r.stgl,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.s50,r.s150,r.s200,r.s30w,r.avol,r.date],
  };
  const lines=[(hdrs[tab]||hdrs.piv).join(','),...vis.map(r=>(cells[tab]||cells.piv)(r).join(','))];
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));
  a.download=`nse_scan_${tab}_${new Date().toISOString().slice(0,10)}.csv`;a.click();
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  document.getElementById('st-u').textContent=S.length;
  const p=loadPrefs();applyPrefs(p||DEFAULTS);
  scan();
});
</script>
</body>
</html>
"""

def build_html(stocks, data_dir):
    gt = datetime.now().strftime("%d %b %Y  %H:%M")
    return (HTML.replace("__JSON__", json.dumps(stocks, separators=(",",":")))
                .replace("__GT__", gt).replace("__DD__", data_dir))

def main():
    ap = argparse.ArgumentParser(description="NSE Strategy Scanner")
    ap.add_argument("--data",  default=DATA_DIR)
    ap.add_argument("--n50",   default=N50_FILE); ap.add_argument("--n100", default=N100_FILE)
    ap.add_argument("--n200",  default=N200_FILE); ap.add_argument("--n500", default=N500_FILE)
    ap.add_argument("--n750",  default=N750_FILE); ap.add_argument("--out",  default=OUTPUT_HTML)
    a = ap.parse_args()
    print(f"[*] Data: {a.data}")
    idx = {50:a.n50, 100:a.n100, 200:a.n200, 500:a.n500, 750:a.n750}
    stocks = build_dataset(a.data, idx)
    if not stocks: print("[!] No stocks loaded"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f: f.write(html)
    print(f"[+] Output: {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
