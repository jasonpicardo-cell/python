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
NFNO_FILE   = "../niftyfno.txt"   # Nifty F&O eligible stocks
OUTPUT_HTML = "index.html"
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

# ══════════════════════════════════════════════════════════════════════════════
# Advanced Strategy Computations
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Darvas Box ─────────────────────────────────────────────────────────────
def compute_darvas(df):
    """Nicolas Darvas: 52W-high stock consolidates in tight box → volume breakout."""
    if len(df) < 60: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    # Find last confirmed box: a high that holds for 3+ bars, then find base
    for i in range(n-4, max(0,n-120), -1):
        if all(H[i] >= H[j] for j in range(i+1, min(i+4,n))):  # box top confirmed
            top = H[i]
            bot = L[i:].min()
            rng = (top-bot)/top
            if rng < 0.20:  # tight box
                avg20 = V[-21:-1].mean() if n>21 else V.mean()
                brk   = bool(C[-1] > top and C[-2] <= top)
                in_b  = bool(C[-1] <= top * 1.005 and C[-1] >= bot * 0.995)
                return dict(top=r2(top), bot=r2(bot),
                            dt=str(df.iloc[i]["Date"].date()),
                            in_box=in_b, breakout=brk,
                            vol_ok=bool(V[-1] > avg20*1.5))
    return {}

# ── 2. VCP — Volatility Contraction Pattern ───────────────────────────────────
def compute_vcp(df):
    """Minervini VCP: 3+ progressively tighter contractions, declining volume."""
    if len(df) < 60: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    segs=3; seg=20
    widths=[]; vols=[]
    for i in range(segs,0,-1):
        s=n-i*seg; e=n-(i-1)*seg
        if s<0: continue
        sh=H[s:e].max(); sl=L[s:e].min()
        widths.append(round((sh-sl)/sl*100,1))
        vols.append(float(V[s:e].mean()))
    if len(widths)<2: return {}
    contracting = all(widths[i]>widths[i+1] for i in range(len(widths)-1))
    vol_dec     = all(vols[i]>vols[i+1]     for i in range(len(vols)-1))
    tight_rng   = round((H[-10:].max()-L[-10:].min())/L[-10:].min()*100,1)
    pivot       = r2(H[-10:].max())
    is_vcp      = contracting and vol_dec and tight_rng < 10
    return dict(widths=widths, tightest=tight_rng, vol_dec=vol_dec,
                contracting=contracting, pivot=pivot, is_vcp=is_vcp)

# ── 3. Wyckoff ────────────────────────────────────────────────────────────────
def compute_wyckoff(df):
    """Wyckoff phase + Spring / Upthrust detection."""
    if len(df) < 60: return {}
    C=df["Close"].values.astype(float); H=df["High"].values.astype(float)
    L=df["Low"].values.astype(float);  V=df["Volume"].values.astype(float)
    n=len(df); lb=min(60,n)
    s20=C[-20:].mean(); s50=C[-50:].mean() if n>=50 else s20
    last=C[-1]
    vr=V[-20:].mean(); vp=V[-40:-20].mean() if n>=40 else vr
    vol_up=bool(vr>vp)
    if   last<s20<s50 and vol_up:  phase="Markdown"
    elif last<s20 and not vol_up:  phase="Distribution"
    elif last>s20>s50:             phase="Markup"
    else:                          phase="Accumulation"
    # Spring: price dips below support then recovers
    sup=L[-lb:-5].min() if n>10 else L.min()
    spring=any(L[i]<sup*0.99 and C[i]>sup for i in range(n-5,n-1)) if n>5 else False
    res=H[-lb:-5].max() if n>10 else H.max()
    upthrust=any(H[i]>res*1.01 and C[i]<res for i in range(n-5,n-1)) if n>5 else False
    return dict(phase=phase, spring=bool(spring), upthrust=bool(upthrust),
                support=r2(sup), resistance=r2(res),
                vol_trend="Rising" if vol_up else "Falling")

# ── 4. Turtle Trading / Donchian Channels ─────────────────────────────────────
def compute_turtle(df):
    """Donchian 20/55-day breakout + 10-day exit (Turtle Trading rules)."""
    if len(df) < 60: return {}
    C=df["Close"].values.astype(float)
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    n=len(df)
    dc20h=r2(H[-21:-1].max()) if n>=21 else r2(H.max())
    dc20l=r2(L[-21:-1].min()) if n>=21 else r2(L.min())
    dc55h=r2(H[-56:-1].max()) if n>=56 else r2(H.max())
    dc55l=r2(L[-56:-1].min()) if n>=56 else r2(L.min())
    dc10l=r2(L[-11:-1].min()) if n>=11 else r2(L.min())
    last=C[-1]
    return dict(dc20h=dc20h,dc20l=dc20l,dc55h=dc55h,dc55l=dc55l,dc10l=dc10l,
                bo20=bool(last>dc20h), bo55=bool(last>dc55h),
                exit10=bool(last<dc10l))

# ── 5. Ichimoku Cloud ─────────────────────────────────────────────────────────
def compute_ichimoku(df):
    """Ichimoku: Tenkan/Kijun, cloud position, TK cross, Kumo breakout."""
    if len(df) < 60: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    def mid(p,off=0):
        i=n-1-off; s=i-p+1
        return (H[s:i+1].max()+L[s:i+1].min())/2 if s>=0 else None
    tenkan=mid(9); kijun=mid(26)
    sa26=(mid(9,26)+mid(26,26))/2 if (mid(9,26) and mid(26,26)) else None
    sb26=mid(52,26)
    last=C[-1]
    above=below=False; thick=0
    if sa26 and sb26:
        ct=max(sa26,sb26); cb=min(sa26,sb26)
        above=bool(last>ct); below=bool(last<cb)
        thick=round((ct-cb)/cb*100,1)
    # TK cross
    tk_bull=tk_bear=False
    if tenkan and kijun:
        tp=mid(9,1); kp=mid(26,1)
        if tp and kp:
            tk_bull=bool(tenkan>kijun and tp<=kp)
            tk_bear=bool(tenkan<kijun and tp>=kp)
    # Kumo breakout (crossed above cloud in last 3 bars)
    kbo=False
    if sa26 and sb26:
        ct=max(sa26,sb26)
        kbo=bool(above and n>=4 and any(C[i]<ct for i in range(n-4,n-1)))
    return dict(tenkan=r2(tenkan) if tenkan else 0,
                kijun=r2(kijun) if kijun else 0,
                span_a=r2(sa26) if sa26 else 0,
                span_b=r2(sb26) if sb26 else 0,
                above=above, below=below, thick=thick,
                tk_bull=tk_bull, tk_bear=tk_bear, kbo=kbo)

# ── 6. TD Sequential ──────────────────────────────────────────────────────────
def compute_td(df):
    """Tom DeMark: 9-bar Setup countdown signals exhaustion reversal."""
    if len(df) < 14: return {}
    C=df["Close"].values.astype(float); n=len(df)
    # Count consecutive closes vs close 4 bars ago
    sell_cnt=buy_cnt=0
    for i in range(n-1,max(3,n-14),-1):
        if C[i]>C[i-4]: sell_cnt+=1
        else: break
    if sell_cnt==0:
        for i in range(n-1,max(3,n-14),-1):
            if C[i]<C[i-4]: buy_cnt+=1
            else: break
    cnt    = sell_cnt if sell_cnt else buy_cnt
    dirn   = "sell" if sell_cnt else ("buy" if buy_cnt else "none")
    done   = cnt>=9
    sig    = ("buy_9" if done and dirn=="buy" else
              "sell_9" if done and dirn=="sell" else "none")
    return dict(count=cnt, dir=dirn, complete=done, signal=sig)

# ── 7. Supertrend ─────────────────────────────────────────────────────────────
def compute_supertrend(df, period=10, mult=3.0):
    """ATR-based Supertrend: direction flip = trend change signal."""
    if len(df) < period+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    tr=np.maximum(H[1:]-L[1:],np.maximum(abs(H[1:]-C[:-1]),abs(L[1:]-C[:-1])))
    tr=np.concatenate([[H[0]-L[0]],tr])
    atr=np.zeros(n); atr[:period]=tr[:period].mean()
    for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    hl2=(H+L)/2; ub=hl2+mult*atr; lb=hl2-mult*atr
    st=np.zeros(n); d=np.zeros(n); st[0]=ub[0]; d[0]=-1
    for i in range(1,n):
        if C[i]>st[i-1]: st[i]=max(lb[i],st[i-1]) if d[i-1]==1 else lb[i]; d[i]=1
        else: st[i]=min(ub[i],st[i-1]) if d[i-1]==-1 else ub[i]; d[i]=-1
    flipped=bool(d[-1]!=d[-2]) if n>=2 else False
    bull=bool(d[-1]==1); bear=bool(d[-1]==-1)
    flip_bull=bool(flipped and bull); flip_bear=bool(flipped and bear)
    consec=1
    for _i in range(len(d)-2,-1,-1):
        if d[_i]==d[-1]: consec+=1
        else: break
    return dict(value=r2(st[-1]),direction="up" if d[-1]==1 else "down",
                flipped=flipped,atr=r2(atr[-1]),
                bull=bull,bear=bear,flip_bull=flip_bull,flip_bear=flip_bear,
                st=r2(st[-1]),dist=round((C[-1]-st[-1])/st[-1]*100,2) if st[-1] else 0.0,
                consec=int(consec))

# ── 8. Elder Triple Screen ────────────────────────────────────────────────────
def compute_elder(df):
    """Elder: weekly MACD trend + daily Force Index timing."""
    if len(df) < 60: return {}
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    def ema(d,p):
        a=2/(p+1); r=np.zeros(len(d)); r[0]=d[0]
        for i in range(1,len(d)): r[i]=a*d[i]+(1-a)*r[i-1]
        return r
    # Screen 1 — weekly MACD (proxy: 60-day vs 130-day EMA histogram)
    fp=min(60,n//2); sp=min(130,n-1)
    macd=ema(C,fp)-ema(C,sp); sig=ema(macd,9)
    trend="bull" if macd[-1]>sig[-1] else "bear"
    rising=bool(macd[-1]>macd[min(5,n-1)])
    # Screen 2 — 2-day EMA of Force Index
    fi=np.concatenate([[0],(C[1:]-C[:-1])*V[1:]])
    fi2=ema(fi,2)
    signal="neutral"
    if trend=="bull" and fi2[-1]<0: signal="buy_setup"
    if trend=="bear" and fi2[-1]>0: signal="sell_setup"
    return dict(macd_trend=trend, macd_rising=rising,
                fi=r2(fi2[-1]),
                fi_trend="pos" if fi2[-1]>0 else "neg",
                signal=signal)


# ══════════════════════════════════════════════════════════════════════════════
# Tier-1 High-Edge Strategy Computations
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Golden Cross / Death Cross ─────────────────────────────────────────────
def compute_golden_cross(df):
    if len(df) < 210: return {}
    C = df["Close"].values.astype(float); n = len(df)
    s50  = float(C[-50:].mean()); s200 = float(C[-200:].mean())
    last = C[-1]
    cross = "none"; cross_dt = ""
    for i in range(1, min(11, n-200)):
        e=n-i+1; s=n-i
        ma50n  = float(C[e-50:e].mean())   if e>=50  else None
        ma200n = float(C[e-200:e].mean())  if e>=200 else None
        ma50p  = float(C[s-50:s].mean())   if s>=50  else None
        ma200p = float(C[s-200:s].mean())  if s>=200 else None
        if None in (ma50n,ma200n,ma50p,ma200p): continue
        if ma50p<=ma200p and ma50n>ma200n:
            cross="golden"; cross_dt=str(df.iloc[n-i]["Date"].date()); break
        if ma50p>=ma200p and ma50n<ma200n:
            cross="death";  cross_dt=str(df.iloc[n-i]["Date"].date()); break
    alignment = "bull" if s50>s200 else "bear"
    return dict(s50=r2(s50), s200=r2(s200), alignment=alignment,
                cross=cross, cross_dt=cross_dt,
                dist_50=round((last-s50)/s50*100,2),
                dist_200=round((last-s200)/s200*100,2),
                gap_pct=round((s50-s200)/s200*100,2))

# ── 2. RSI Divergence ─────────────────────────────────────────────────────────
def compute_rsi_div(df, period=14, lookback=60):
    if len(df) < period+lookback+5: return {}
    C = df["Close"].values.astype(float); n = len(df)
    delta=np.diff(C); gain=np.where(delta>0,delta,0); loss=np.where(delta<0,-delta,0)
    ag=np.zeros(n-1); al=np.zeros(n-1)
    ag[period-1]=gain[:period].mean(); al[period-1]=loss[:period].mean()
    for i in range(period,n-1):
        ag[i]=(ag[i-1]*(period-1)+gain[i])/period
        al[i]=(al[i-1]*(period-1)+loss[i])/period
    rs=np.where(al==0, np.inf, ag/np.where(al==0,1,al))
    rsi=np.where(np.isinf(rs),100,100-100/(1+rs))
    rsi=np.concatenate([[np.nan],rsi])
    cur=float(rsi[-1]) if not np.isnan(rsi[-1]) else 50.0
    seg=rsi[-lookback:]; pr=C[-lookback:]; valid=~np.isnan(seg)
    bull_div=bear_div=False
    if valid.sum()>10:
        lo=[i for i in range(2,len(seg)-2) if valid[i] and pr[i]<pr[i-1] and pr[i]<pr[i+1] and pr[i]<pr[i-2] and pr[i]<pr[i+2]]
        hi=[i for i in range(2,len(seg)-2) if valid[i] and pr[i]>pr[i-1] and pr[i]>pr[i+1] and pr[i]>pr[i-2] and pr[i]>pr[i+2]]
        if len(lo)>=2:
            i1,i2=lo[-2],lo[-1]
            if pr[i2]<pr[i1] and seg[i2]>seg[i1]: bull_div=True
        if len(hi)>=2:
            i1,i2=hi[-2],hi[-1]
            if pr[i2]>pr[i1] and seg[i2]<seg[i1]: bear_div=True
    prev=float(rsi[-2]) if len(rsi)>=2 and not np.isnan(rsi[-2]) else cur
    return dict(rsi=round(cur,1), bull_div=bool(bull_div), bear_div=bool(bear_div),
                overbought=bool(cur>=70), oversold=bool(cur<=30),
                above50=bool(cur>50), below50=bool(cur<50),
                cross50_bull=bool(cur>50 and prev<=50),   # just crossed above 50
                cross50_bear=bool(cur<50 and prev>=50),   # just crossed below 50
                bull_zone=bool(cur>=40 and cur<=80),       # healthy bull RSI zone
                rsi_trend=bool(cur>float(np.nanmean(rsi[-10:]))))  # rising vs 10-bar avg

# ── 3. Bollinger Band Squeeze ─────────────────────────────────────────────────
def compute_bb_squeeze(df, period=20, mult=2.0):
    if len(df) < period+50: return {}
    C=df["Close"].values.astype(float); n=len(df)
    ma  = np.array([C[i-period:i].mean() for i in range(period,n+1)])
    std = np.array([C[i-period:i].std()  for i in range(period,n+1)])
    upper=ma+mult*std; lower=ma-mult*std; bw=(upper-lower)/ma*100
    hist=bw[-126:] if len(bw)>=126 else bw
    thr=np.percentile(hist,25)                    # bottom 25% = squeeze
    in_sq=bool(bw[-1]<=thr)
    mom=float(C[-1]-C[-13]) if n>=13 else 0.0
    return dict(upper=r2(float(upper[-1])), lower=r2(float(lower[-1])),
                mid=r2(float(ma[-1])), bw=round(float(bw[-1]),2),
                in_squeeze=in_sq, momentum=round(mom,2),
                bias="bull" if mom>0 else "bear",
                above_mid=bool(C[-1]>float(ma[-1])))

# ── 4. 52-Week High Breakout ───────────────────────────────────────────────────
def compute_w52_breakout(df):
    if len(df) < 60: return {}
    C=df["Close"].values.astype(float)
    H=df["High"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df); yr=min(252,n)
    w52h=float(H[-yr-1:-1].max()) if n>yr else float(H[:-1].max())
    last=C[-1]; last_h=H[-1]
    avg20=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    breakout=bool(last_h>w52h); close_bo=bool(last>w52h)
    near=bool(last>=w52h*0.97 and not breakout)
    consol=bool((H[-10:].max()-C[-10:].min())/C[-10:].min()*100<5) if n>=10 else False
    return dict(w52h=r2(w52h), breakout=breakout, close_bo=close_bo, near=near,
                vol_surge=bool(V[-1]>=avg20*1.5),
                pct_from=round((last-w52h)/w52h*100,2), consol=consol)

# ── 5. NR7 / Inside Bar ────────────────────────────────────────────────────────
def compute_nr7(df):
    if len(df) < 10: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    ranges=H-L; today_rng=float(ranges[-1])
    is_nr7=bool(today_rng<=float(ranges[-7:].min())) if n>=7 else False
    is_nr4=bool(today_rng<=float(ranges[-4:].min())) if n>=4 else False
    is_inside=bool(H[-1]<H[-2] and L[-1]>L[-2]) if n>=2 else False
    atr14=float(ranges[-14:].mean()) if n>=14 else float(ranges.mean())
    return dict(is_nr7=is_nr7, is_nr4=is_nr4, is_inside=is_inside,
                compression=round(today_rng/atr14*100,1) if atr14>0 else 100.0,
                bias="bull" if float(C[-1])>float(C[-6]) else "bear" if n>=6 else "none",
                rng=r2(today_rng), atr14=r2(atr14))


# ══════════════════════════════════════════════════════════════════════════════
# Tier-2 Strategy Computations
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Cup & Handle ───────────────────────────────────────────────────────────
def compute_cup_handle(df):
    """William O'Neil Cup & Handle: U-shaped base + tight handle + volume breakout."""
    if len(df) < 80: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    handle_bars=15; lb=min(200,n-handle_bars)
    sh=H[-lb-handle_bars:-handle_bars]; sl=L[-lb-handle_bars:-handle_bars]
    if len(sh)<30: return {}
    # Left lip: highest high in the cup segment
    lip_i=int(np.argmax(sh)); left_lip=float(sh[lip_i])
    # Cup bottom: lowest low after the left lip
    bot_seg=sl[lip_i:]; cup_bot=float(bot_seg.min())
    depth=(left_lip-cup_bot)/left_lip*100
    if not (8<=depth<=55): return {}          # 8-55% cup depth
    # Recovery: right side must recover ≥80% of the depth
    rec_seg=sh[lip_i+int(np.argmin(bot_seg)):]; right_lip=float(rec_seg.max())
    recovery=(right_lip-cup_bot)/(left_lip-cup_bot)*100 if (left_lip-cup_bot)>0 else 0
    if recovery<75: return {}
    # Handle: last 15 bars — tight (range <12% of lip) and doesn't exceed lip
    hdl_h=H[-handle_bars:]; hdl_l=L[-handle_bars:]; hdl_c=C[-handle_bars:]
    hdl_rng=(hdl_h.max()-hdl_l.min())/left_lip*100
    in_handle=bool(hdl_rng<12 and hdl_h.max()<=left_lip*1.04)
    # Breakout
    last=C[-1]; avg20=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    breakout=bool(last>left_lip and float(V[-1])>avg20*1.2)
    near_bo=bool(last>=left_lip*0.97 and not breakout)
    return dict(left_lip=r2(left_lip), cup_bot=r2(cup_bot),
                depth=round(depth,1), recovery=round(recovery,1),
                in_handle=in_handle, hdl_rng=round(hdl_rng,1),
                breakout=breakout, near_bo=near_bo, valid=True)

# ── 2. MACD Divergence ────────────────────────────────────────────────────────
def compute_macd_div(df, fast=12, slow=26, sig=9, lookback=60):
    """MACD with bullish/bearish divergence + histogram zero-cross."""
    if len(df)<slow+sig+lookback+5: return {}
    C=df["Close"].values.astype(float); n=len(df)
    def ema(d,p):
        a=2/(p+1); r=np.zeros(len(d)); r[0]=d[0]
        for i in range(1,len(d)): r[i]=a*d[i]+(1-a)*r[i-1]
        return r
    ml=ema(C,fast)-ema(C,slow); sl2=ema(ml,sig); hist=ml-sl2
    cur_m=float(ml[-1]); cur_s=float(sl2[-1]); cur_h=float(hist[-1])
    pr=C[-lookback:]; ms=ml[-lookback:]; hs=hist[-lookback:]
    bull_div=bear_div=False
    lo=[i for i in range(2,len(pr)-2) if pr[i]<pr[i-1] and pr[i]<pr[i+1] and pr[i]<pr[i-2] and pr[i]<pr[i+2]]
    hi=[i for i in range(2,len(pr)-2) if pr[i]>pr[i-1] and pr[i]>pr[i+1] and pr[i]>pr[i-2] and pr[i]>pr[i+2]]
    if len(lo)>=2:
        i1,i2=lo[-2],lo[-1]
        if pr[i2]<pr[i1] and ms[i2]>ms[i1]: bull_div=True
    if len(hi)>=2:
        i1,i2=hi[-2],hi[-1]
        if pr[i2]>pr[i1] and ms[i2]<ms[i1]: bear_div=True
    flip_bull=bool(n>=2 and hist[-1]>0 and hist[-2]<=0)
    flip_bear=bool(n>=2 and hist[-1]<0 and hist[-2]>=0)
    # Histogram direction signals
    hist_rising  = bool(n>=3 and hist[-1]>hist[-2])           # hist increasing
    hist_pos_inc = bool(n>=3 and hist[-1]>0 and hist[-1]>hist[-2])   # positive & rising
    hist_neg_inc = bool(n>=3 and hist[-1]<0 and hist[-1]>hist[-2])   # negative but rising (early bull)
    hist_falling = bool(n>=3 and hist[-1]<hist[-2])           # hist decreasing
    zero_bull    = bool(n>=2 and cur_m>0 and float(ml[-2])<=0)  # MACD crossed zero line up
    zero_bear    = bool(n>=2 and cur_m<0 and float(ml[-2])>=0)  # MACD crossed zero line down
    return dict(macd=round(cur_m,3), signal=round(cur_s,3), hist=round(cur_h,3),
                trend="bull" if cur_m>cur_s else "bear",
                rising=bool(ml[-1]>ml[-min(5,n)]),
                bull_div=bool(bull_div), bear_div=bool(bear_div),
                flip_bull=flip_bull, flip_bear=flip_bear,
                hist_rising=hist_rising, hist_pos_inc=hist_pos_inc,
                hist_neg_inc=hist_neg_inc, hist_falling=hist_falling,
                zero_bull=zero_bull, zero_bear=zero_bear)

# ── 3. Mean Reversion Z-Score ─────────────────────────────────────────────────
def compute_zscore(df):
    """Z-score: how many std-devs is price from its moving average.
    Z < -2 = extreme oversold (mean reversion long).
    Z > +2 = extreme overbought (mean reversion short)."""
    if len(df)<25: return {}
    C=df["Close"].values.astype(float); n=len(df)
    def zs(p):
        if n<p: return None
        m=C[-p:].mean(); s=C[-p:].std()
        return round(float((C[-1]-m)/s),2) if s>0 else 0.0
    z20=zs(20); z50=zs(50); z200=zs(200)
    ma20=float(C[-20:].mean()) if n>=20 else float(C.mean())
    std20=float(C[-20:].std()) if n>=20 else 1.0
    pct_b=round((C[-1]-(ma20-2*std20))/(4*std20)*100,1) if std20>0 else 50.0
    return dict(z20=z20 or 0.0, z50=z50 or 0.0, z200=z200 or 0.0,
                oversold=bool(z20 is not None and z20<-2.0),
                overbought=bool(z20 is not None and z20>2.0),
                extreme_os=bool(z20 is not None and z20<-3.0),
                extreme_ob=bool(z20 is not None and z20>3.0),
                pct_b=pct_b, ma20=r2(ma20))

# ── 4. Relative Strength vs Nifty ─────────────────────────────────────────────
def compute_rs_nifty(df, nifty_df):
    """RS ratio = stock / Nifty. Rising ratio = outperforming the index."""
    if nifty_df is None or len(df)<30 or len(nifty_df)<30: return {}
    # Align dates
    merged=pd.merge(df[["Date","Close"]].rename(columns={"Close":"stk"}),
                    nifty_df[["Date","Close"]].rename(columns={"Close":"nif"}),
                    on="Date", how="inner").sort_values("Date")
    if len(merged)<20: return {}
    stk=merged["stk"].values.astype(float)
    nif=merged["nif"].values.astype(float)
    nif=np.where(nif==0,1,nif)
    rs=stk/nif; m=len(rs)
    cur=float(rs[-1]); rs20=float(rs[-21]) if m>=21 else float(rs[0])
    rs_trend="bull" if cur>rs20 else "bear"
    # RS making new 3-month high
    rs3m=rs[-63:] if m>=63 else rs
    rs_new_hi=bool(cur>=float(rs3m.max())*0.995)
    # RS vs 20-day MA of RS ratio
    rs_ma20=float(rs[-20:].mean()) if m>=20 else float(rs.mean())
    above_rs_ma=bool(cur>rs_ma20)
    chg20=round((cur-rs20)/rs20*100,2) if rs20>0 else 0.0
    # z-score of RS (how extreme is the outperformance)
    rs_std=float(rs[-20:].std()) if m>=20 else 1.0
    rs_z=round((cur-rs_ma20)/rs_std,2) if rs_std>0 else 0.0
    return dict(rs_trend=rs_trend, rs_new_hi=rs_new_hi,
                above_rs_ma=above_rs_ma, chg20=chg20, rs_z=rs_z)


# ══════════════════════════════════════════════════════════════════════════════
# Tier-3 Strategy Computations
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Fibonacci Retracement from Swing Points ────────────────────────────────
def compute_fib_retracement(df, lookback=120):
    """Find most recent swing high/low and compute Fib retracement levels.
    Near a Fib level = high-probability support/resistance."""
    if len(df) < 40: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    lb=min(lookback,n)
    sh=H[-lb:]; sl=L[-lb:]
    swing_h=float(sh.max()); swing_l=float(sl.min())
    sh_i=int(np.argmax(sh)); sl_i=int(np.argmin(sl))
    rng=swing_h-swing_l
    if rng<=0: return {}
    # Determine trend direction for retracement
    uptrend=sh_i>sl_i   # high came after low = uptrend, measure retracement down
    last=C[-1]
    FIBS=[0.0,0.236,0.382,0.5,0.618,0.786,1.0]
    FIB_NAMES=["0%","23.6%","38.2%","50%","61.8%","78.6%","100%"]
    if uptrend:
        levels={n:round(swing_h-f*rng,2) for f,n in zip(FIBS,FIB_NAMES)}
    else:
        levels={n:round(swing_l+f*rng,2) for f,n in zip(FIBS,FIB_NAMES)}
    # Find nearest level
    nearest_name=min(levels,key=lambda k:abs(levels[k]-last))
    nearest_val=levels[nearest_name]
    dist_pct=round((last-nearest_val)/nearest_val*100,2)
    near=bool(abs(dist_pct)<=2.0)
    # Key levels dict (compact)
    return dict(swing_h=r2(swing_h), swing_l=r2(swing_l),
                uptrend=uptrend,
                f236=r2(levels["23.6%"]), f382=r2(levels["38.2%"]),
                f500=r2(levels["50%"]),   f618=r2(levels["61.8%"]),
                f786=r2(levels["78.6%"]),
                nearest=nearest_name, nearest_val=r2(nearest_val),
                dist_pct=dist_pct, near=near)

# ── 2. Chandelier Exit ────────────────────────────────────────────────────────
def compute_chandelier(df, period=22, mult=3.0):
    """Chandelier Exit: ATR-based trailing stop.
    Long stop  = Highest Close(N) − mult × ATR(N)
    Short stop = Lowest  Close(N) + mult × ATR(N)"""
    if len(df)<period+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    tr=np.maximum(H[1:]-L[1:],np.maximum(abs(H[1:]-C[:-1]),abs(L[1:]-C[:-1])))
    tr=np.concatenate([[H[0]-L[0]],tr])
    atr=np.zeros(n); atr[:period]=tr[:period].mean()
    for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    hh=float(C[-period:].max()); ll=float(C[-period:].min())
    long_stop  = r2(hh - mult*atr[-1])
    short_stop = r2(ll + mult*atr[-1])
    last=C[-1]
    bull=bool(last>long_stop)
    bear=bool(last<short_stop)
    # Flip: direction changed in last 3 bars
    flip_bull=flip_bear=False
    if n>=4:
        for i in range(1,4):
            ph=float(C[-period-i:-i].max()) if n>=period+i else float(C.max())
            pl=float(C[-period-i:-i].min()) if n>=period+i else float(C.min())
            ps=r2(ph-mult*atr[-i])
            pb=r2(pl+mult*atr[-i])
            was_bull=C[-i]>ps; was_bear=C[-i]<pb
            if not was_bull and bull: flip_bull=True
            if not was_bear and bear: flip_bear=True
    return dict(long_stop=long_stop, short_stop=short_stop,
                bull=bull, bear=bear,
                flip_bull=bool(flip_bull), flip_bear=bool(flip_bear),
                atr=r2(atr[-1]))

# ── 3. Parabolic SAR ──────────────────────────────────────────────────────────
def compute_psar(df, af_start=0.02, af_step=0.02, af_max=0.20):
    """Parabolic SAR: accelerating trailing stop.
    Price above SAR = bull; price below SAR = bear."""
    if len(df)<10: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    # Initialise
    bull=C[1]>C[0]; sar=L[0] if bull else H[0]
    ep=H[1] if bull else L[1]; af=af_start
    sars=np.zeros(n); sars[0]=sar; dirs=np.zeros(n); dirs[0]=1 if bull else -1
    for i in range(1,n):
        if bull:
            sar=min(sar+af*(ep-sar), L[i-1], L[max(0,i-2)])
            if L[i]<sar:
                bull=False; sar=ep; ep=L[i]; af=af_start
            else:
                if H[i]>ep: ep=H[i]; af=min(af+af_step,af_max)
        else:
            sar=max(sar+af*(ep-sar), H[i-1], H[max(0,i-2)])
            if H[i]>sar:
                bull=True; sar=ep; ep=H[i]; af=af_start
            else:
                if L[i]<ep: ep=L[i]; af=min(af+af_step,af_max)
        sars[i]=sar; dirs[i]=1 if bull else -1
    last_sar=r2(float(sars[-1])); last_dir="bull" if bull else "bear"
    flip=bool(dirs[-1]!=dirs[-2]) if n>=2 else False
    dist=round((C[-1]-last_sar)/last_sar*100,2)
    return dict(sar=last_sar, direction=last_dir, flip=flip,
                dist_pct=dist, af=round(af,3))

# ── 4. ADX + DI Crossover ─────────────────────────────────────────────────────
def compute_adx(df, period=14):
    """ADX: trend strength (>25=strong). +DI/-DI crossover = trend direction change."""
    if len(df)<period*2+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    # True Range and Directional Movement
    tr=np.zeros(n); dm_p=np.zeros(n); dm_m=np.zeros(n)
    for i in range(1,n):
        tr[i]=max(H[i]-L[i],abs(H[i]-C[i-1]),abs(L[i]-C[i-1]))
        h_diff=H[i]-H[i-1]; l_diff=L[i-1]-L[i]
        dm_p[i]=h_diff if h_diff>l_diff and h_diff>0 else 0
        dm_m[i]=l_diff if l_diff>h_diff and l_diff>0 else 0
    # Wilder smoothing
    def wilder(arr,p):
        r=np.zeros(n); r[p]=arr[1:p+1].sum()
        for i in range(p+1,n): r[i]=r[i-1]-r[i-1]/p+arr[i]
        return r
    atr14=wilder(tr,period); dp14=wilder(dm_p,period); dm14=wilder(dm_m,period)
    safe=np.where(atr14==0,1,atr14)
    di_p=100*dp14/safe; di_m=100*dm14/safe
    safe_sum=np.where((di_p+di_m)==0, 1, di_p+di_m)
    dx=np.where((di_p+di_m)==0, 0, 100*np.abs(di_p-di_m)/safe_sum)
    adx=np.zeros(n); adx[period*2]=dx[period:period*2+1].mean()
    for i in range(period*2+1,n): adx[i]=(adx[i-1]*(period-1)+dx[i])/period
    cur_adx=round(float(adx[-1]),1)
    cur_dip=round(float(di_p[-1]),1); cur_dim=round(float(di_m[-1]),1)
    trending=bool(cur_adx>=25)
    strong=bool(cur_adx>=40)
    bull_trend=bool(cur_dip>cur_dim)
    # DI crossover (last 3 bars)
    di_cross_bull=di_cross_bear=False
    if n>=4:
        if di_p[-1]>di_m[-1] and di_p[-2]<=di_m[-2]: di_cross_bull=True
        if di_m[-1]>di_p[-1] and di_m[-2]<=di_p[-2]: di_cross_bear=True
    return dict(adx=cur_adx, di_plus=cur_dip, di_minus=cur_dim,
                trending=trending, strong=strong, bull_trend=bull_trend,
                di_cross_bull=bool(di_cross_bull),
                di_cross_bear=bool(di_cross_bear))

# ── 5. Stochastic %K/%D ───────────────────────────────────────────────────────
def compute_stochastic(df, k_period=14, d_period=3, slowing=3):
    """Stochastic: %K=(C-LL)/(HH-LL)*100, %D=SMA(%K).
    Overbought >80, Oversold <20. Crossovers in extreme zones = high-prob."""
    if len(df)<k_period+d_period+slowing+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    raw_k=np.zeros(n)
    for i in range(k_period-1,n):
        hh=H[i-k_period+1:i+1].max(); ll=L[i-k_period+1:i+1].min()
        raw_k[i]=(C[i]-ll)/(hh-ll)*100 if hh>ll else 50
    # Slow %K = SMA of raw_k
    slow_k=np.zeros(n)
    for i in range(k_period+slowing-2,n):
        slow_k[i]=raw_k[i-slowing+1:i+1].mean()
    # %D = SMA of slow_k
    slow_d=np.zeros(n)
    for i in range(k_period+slowing+d_period-3,n):
        slow_d[i]=slow_k[i-d_period+1:i+1].mean()
    ck=round(float(slow_k[-1]),1); cd=round(float(slow_d[-1]),1)
    ob=bool(ck>=80); os_=bool(ck<=20)
    # Crossovers
    bull_cross=bool(slow_k[-1]>slow_d[-1] and slow_k[-2]<=slow_d[-2] and ck<=40)
    bear_cross=bool(slow_k[-1]<slow_d[-1] and slow_k[-2]>=slow_d[-2] and ck>=60)
    return dict(k=ck, d=cd, overbought=ob, oversold=os_,
                bull_cross=bool(bull_cross), bear_cross=bool(bear_cross))

# ── 6. Williams %R ────────────────────────────────────────────────────────────
def compute_williams_r(df, period=14):
    """Williams %R: momentum oscillator -100 to 0.
    Near 0 = overbought. Near -100 = oversold. Failure swing = reversal."""
    if len(df)<period+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    wr=np.zeros(n)
    for i in range(period-1,n):
        hh=H[i-period+1:i+1].max(); ll=L[i-period+1:i+1].min()
        wr[i]=((hh-C[i])/(hh-ll)*-100) if hh>ll else -50
    cur=round(float(wr[-1]),1)
    ob=bool(cur>=-20); os_=bool(cur<=-80)
    # Momentum shift: was extreme, now moving away
    bull_exit=bool(wr[-2]<=-80 and wr[-1]>-80)  # exiting oversold
    bear_exit=bool(wr[-2]>=-20 and wr[-1]<-20)  # exiting overbought
    # 14-bar momentum direction
    trend="bull" if wr[-1]>wr[-min(14,n)] else "bear"
    return dict(wr=cur, overbought=ob, oversold=os_,
                bull_exit=bool(bull_exit), bear_exit=bool(bear_exit),
                trend=trend)


# ══════════════════════════════════════════════════════════════════════════════
# 🏅 India Pro — strategies from top Indian paid platforms
# (Chartink, StockEdge, Trendlyne)
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Candlestick Patterns ───────────────────────────────────────────────────
def compute_candles(df):
    """Comprehensive candlestick pattern scanner — 27 patterns from daily OHLC.
    Covers all major StockEdge, Chartink, and MunafaSutra candlestick scans."""
    if len(df)<5: return {}
    O=df["Open"].values.astype(float); H=df["High"].values.astype(float)
    L=df["Low"].values.astype(float); C=df["Close"].values.astype(float)
    n=len(df)

    def body(i): return abs(C[i]-O[i])
    def rng(i):  return max(H[i]-L[i], 0.0001)
    def sup(i):  return H[i]-max(C[i],O[i])   # upper shadow
    def sdn(i):  return min(C[i],O[i])-L[i]   # lower shadow
    def bull(i): return C[i]>O[i]
    def bear(i): return C[i]<O[i]
    def mid(i):  return (H[i]+L[i])/2.0

    atr5=float(np.mean([rng(-j) for j in range(1,6)]))

    # ── Single-bar ──────────────────────────────────────────────────────────
    b0=body(-1); r0=rng(-1); su=sup(-1); sd=sdn(-1)
    hammer        = bool(bull(-1) and b0/r0<0.35 and sd>2*b0 and su<0.5*b0)
    hanging_man   = bool(bear(-1) and b0/r0<0.35 and sd>2*b0 and su<0.5*b0)
    shooting_star = bool(bear(-1) and b0/r0<0.35 and su>2*b0 and sd<0.5*b0)
    inv_hammer    = bool(bull(-1) and b0/r0<0.35 and su>2*b0 and sd<0.5*b0)
    doji          = bool(b0/r0<0.1 and r0>atr5*0.3)
    # Marubozu: strong body (≥90% of range), minimal shadows — StockEdge scan
    marubozu_bull = bool(bull(-1) and b0/r0>=0.90)
    marubozu_bear = bool(bear(-1) and b0/r0>=0.90)
    # Spinning Top: small body, long shadows on BOTH sides — pure indecision
    spinning_top  = bool(b0/r0<0.3 and su>b0 and sd>b0 and r0>atr5*0.4)

    # ── Two-bar ─────────────────────────────────────────────────────────────
    if n>=2:
        b1=body(-2)
        # Classic reversals
        bull_eng  = bool(bull(-1) and bear(-2) and C[-1]>O[-2] and O[-1]<C[-2] and b0>b1)
        bear_eng  = bool(bear(-1) and bull(-2) and C[-1]<O[-2] and O[-1]>C[-2] and b0>b1)
        bull_har  = bool(bull(-1) and bear(-2) and O[-1]>C[-2] and C[-1]<O[-2] and b0<b1*0.5)
        bear_har  = bool(bear(-1) and bull(-2) and O[-1]<C[-2] and C[-1]>O[-2] and b0<b1*0.5)
        piercing  = bool(bear(-2) and bull(-1) and O[-1]<L[-2]
                         and C[-1]>(O[-2]+C[-2])/2 and C[-1]<O[-2])
        dark_cloud= bool(bull(-2) and bear(-1) and O[-1]>H[-2]
                         and C[-1]<(O[-2]+C[-2])/2 and C[-1]>O[-2])
        # Inside Bar: current bar completely inside prior bar — Chartink #1 scan
        inside_bar      = bool(H[-1]<H[-2] and L[-1]>L[-2])
        inside_bar_bull = bool(inside_bar and C[-1]>mid(-1))   # closed upper half → bullish bias
        inside_bar_bear = bool(inside_bar and C[-1]<mid(-1))   # closed lower half → bearish bias
        # Outside Bar: current bar engulfs prior bar (range wider both sides)
        outside_bar      = bool(H[-1]>H[-2] and L[-1]<L[-2])
        outside_bar_bull = bool(outside_bar and bull(-1))
        outside_bar_bear = bool(outside_bar and bear(-1))
        # Tweezer: two candles with same high (top) or same low (bottom) within 0.3%
        tol = 0.003
        tweezer_top    = bool(abs(H[-1]-H[-2])/max(H[-2],0.01)<tol and bear(-1))
        tweezer_bottom = bool(abs(L[-1]-L[-2])/max(L[-2],0.01)<tol and bull(-1))
        # Gap: today's open vs yesterday's close (true price gap)
        gap_pct   = float((O[-1]-C[-2])/max(C[-2],0.01)*100)
        gap_up    = bool(gap_pct > 1.0)    # gap up > 1%
        gap_down  = bool(gap_pct < -1.0)   # gap down < -1%
        gap_up3   = bool(gap_pct > 3.0)    # gap up > 3% (Chartink style)
        gap_down3 = bool(gap_pct < -3.0)   # gap down > 3%
    else:
        bull_eng=bear_eng=bull_har=bear_har=piercing=dark_cloud=False
        inside_bar=inside_bar_bull=inside_bar_bear=False
        outside_bar=outside_bar_bull=outside_bar_bear=False
        tweezer_top=tweezer_bottom=False
        gap_pct=0.0; gap_up=gap_down=gap_up3=gap_down3=False

    # Double Inside Bar: two consecutive inside bars — even tighter coil
    if n>=3:
        double_inside = bool(inside_bar and H[-2]<H[-3] and L[-2]>L[-3])
    else:
        double_inside = False

    # ── Three-bar ───────────────────────────────────────────────────────────
    if n>=3:
        morning_star = bool(bear(-3) and body(-2)/rng(-2)<0.35
                            and bull(-1) and C[-1]>(O[-3]+C[-3])/2)
        evening_star = bool(bull(-3) and body(-2)/rng(-2)<0.35
                            and bear(-1) and C[-1]<(O[-3]+C[-3])/2)
        tws = bool(all(bull(-j) for j in [1,2,3]) and C[-1]>C[-2]>C[-3]
                   and all(body(-j)/rng(-j)>0.55 for j in [1,2,3]))
        tbc = bool(all(bear(-j) for j in [1,2,3]) and C[-1]<C[-2]<C[-3]
                   and all(body(-j)/rng(-j)>0.55 for j in [1,2,3]))
    else:
        morning_star=evening_star=tws=tbc=False

    return dict(
        # Original 14
        hammer=bool(hammer), hanging_man=bool(hanging_man),
        shooting_star=bool(shooting_star), inv_hammer=bool(inv_hammer),
        doji=bool(doji),
        bull_eng=bool(bull_eng), bear_eng=bool(bear_eng),
        bull_har=bool(bull_har), bear_har=bool(bear_har),
        piercing=bool(piercing), dark_cloud=bool(dark_cloud),
        morning_star=bool(morning_star), evening_star=bool(evening_star),
        tws=bool(tws), tbc=bool(tbc),
        # New 13 — StockEdge / Chartink patterns
        marubozu_bull=bool(marubozu_bull), marubozu_bear=bool(marubozu_bear),
        spinning_top=bool(spinning_top),
        inside_bar=bool(inside_bar),
        inside_bar_bull=bool(inside_bar_bull), inside_bar_bear=bool(inside_bar_bear),
        double_inside=bool(double_inside),
        outside_bar_bull=bool(outside_bar_bull), outside_bar_bear=bool(outside_bar_bear),
        tweezer_top=bool(tweezer_top), tweezer_bottom=bool(tweezer_bottom),
        gap_up=bool(gap_up), gap_up3=bool(gap_up3),
        gap_down=bool(gap_down), gap_down3=bool(gap_down3),
        gap_pct=round(gap_pct,2),
    )

# ── 2. Multi-EMA Crossovers ───────────────────────────────────────────────────
def compute_ema_cross(df):
    """EMA 9/21/50 crossovers + EMA fan alignment (Chartink premium style)."""
    if len(df)<55: return {}
    C=df["Close"].values.astype(float); n=len(df)
    def ema_arr(p):
        a=2/(p+1); r=np.zeros(n); r[0]=C[0]
        for i in range(1,n): r[i]=a*C[i]+(1-a)*r[i-1]
        return r
    e9=ema_arr(9); e21=ema_arr(21); e20=ema_arr(20); e50=ema_arr(50)
    # Crossovers (last 3 bars)
    def cross_up(a,b): return bool(a[-1]>b[-1] and a[-2]<=b[-2])
    def cross_dn(a,b): return bool(a[-1]<b[-1] and a[-2]>=b[-2])
    def was_up(a,b,k=5): return any(a[-i]>b[-i] and a[-i-1]<=b[-i-1] for i in range(1,min(k,n-1)))
    c921_bull=cross_up(e9,e21); c921_bear=cross_dn(e9,e21)
    c2050_bull=cross_up(e20,e50); c2050_bear=cross_dn(e20,e50)
    # Alignment
    fan_bull=bool(C[-1]>e9[-1]>e21[-1]>e50[-1])
    fan_bear=bool(C[-1]<e9[-1]<e21[-1]<e50[-1])
    # Above all
    above_all=bool(C[-1]>e9[-1] and C[-1]>e21[-1] and C[-1]>e50[-1])
    recent_921=bool(was_up(e9,e21,8))  # 9 crossed above 21 in last 8 bars
    return dict(c921_bull=c921_bull, c921_bear=c921_bear,
                c2050_bull=c2050_bull, c2050_bear=c2050_bear,
                fan_bull=fan_bull, fan_bear=fan_bear,
                above_all=above_all, recent_921=bool(recent_921),
                e9=r2(e9[-1]), e21=r2(e21[-1]), e50=r2(e50[-1]))

# ── 3. Price Momentum Score ───────────────────────────────────────────────────
def compute_momentum_score(df):
    """Trendlyne-style composite momentum: weighted 1M+3M+6M+12M returns."""
    if len(df)<25: return {}
    C=df["Close"].values.astype(float); n=len(df)
    def ret(d): return round((C[-1]/C[-d-1]-1)*100,2) if n>d+1 else None
    r1m=ret(21); r3m=ret(63); r6m=ret(126); r12m=ret(252)
    wts=[(r1m,.15),(r3m,.25),(r6m,.30),(r12m,.30)]
    tw=sum(w for v,w in wts if v is not None)
    score=round(sum(v*w for v,w in wts if v is not None)/tw,2) if tw>0 else 0.0
    # Rate of Change (10-day, 21-day)
    roc10=round((C[-1]/C[-11]-1)*100,2) if n>11 else 0.0
    roc21=round((C[-1]/C[-22]-1)*100,2) if n>22 else 0.0
    return dict(r1m=r1m or 0.0, r3m=r3m or 0.0, r6m=r6m or 0.0, r12m=r12m or 0.0,
                score=score, roc10=roc10, roc21=roc21,
                positive_all=bool(all(v is not None and v>0 for v,_ in wts if v is not None)))

# ── 4. Sequential Higher Highs/Lows ──────────────────────────────────────────
def compute_sequential(df, n_bars=3):
    """3+ consecutive higher highs or lower lows — trend structure (ScanX/Chartink)."""
    if len(df)<n_bars+2: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    # Higher highs
    hh3=bool(all(H[-i]>H[-i-1] for i in range(1,n_bars+1)))
    hh5=bool(all(H[-i]>H[-i-1] for i in range(1,6))) if n>=7 else False
    # Higher lows
    hl3=bool(all(L[-i]>L[-i-1] for i in range(1,n_bars+1)))
    # Lower lows / highs
    ll3=bool(all(L[-i]<L[-i-1] for i in range(1,n_bars+1)))
    lh3=bool(all(H[-i]<H[-i-1] for i in range(1,n_bars+1)))
    # 3 consecutive higher closes
    hc3=bool(C[-1]>C[-2]>C[-3]>C[-4]) if n>=5 else False
    lc3=bool(C[-1]<C[-2]<C[-3]<C[-4]) if n>=5 else False
    # Volume confirmation for the higher close pattern
    V=df["Volume"].values.astype(float)
    avgv=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    vol_confirm=bool(float(V[-1])>avgv*1.2)
    return dict(hh3=hh3, hh5=hh5, hl3=hl3, ll3=ll3, lh3=lh3,
                hc3=hc3, lc3=lc3, vol_confirm=vol_confirm)

# ── 5. Volume Buildup (Accumulation Days) ────────────────────────────────────
def compute_volume_buildup(df):
    """StockEdge-style: 3+ rising-volume up-days = institutional accumulation."""
    if len(df)<10: return {}
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    avg20=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    # Count consecutive up-days with rising volume
    streak=0
    for i in range(1,8):
        if C[-i]>C[-i-1] and V[-i]>V[-i-1]: streak+=1
        else: break
    # Quiet accumulation: price flat but volume steadily above avg
    quiet_acc=bool(streak==0 and all(V[-i]>avg20*1.3 for i in range(1,4))
                   and abs(C[-1]-C[-4])/C[-4]<0.02)
    return dict(streak=streak, acc3=bool(streak>=3), acc5=bool(streak>=5),
                quiet_acc=bool(quiet_acc), avg20=r2(avg20))

# ── 6. 52W Consolidation Near High ───────────────────────────────────────────
def compute_52w_consol(df):
    """Price within 5% of 52W high in tight range (5-15 days) — pre-breakout coil."""
    if len(df)<30: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df); lb=min(252,n)
    w52h=float(H[-lb:].max())
    dist_pct=(w52h-C[-1])/w52h*100
    # Last 10 days range as % of 52W high
    last10_rng=(H[-10:].max()-L[-10:].min())/w52h*100 if n>=10 else 100.0
    last15_rng=(H[-15:].max()-L[-15:].min())/w52h*100 if n>=15 else 100.0
    avg20=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    vol_dry=float(V[-5:].mean())/avg20 if avg20>0 else 1.0
    coil10=bool(dist_pct<=5 and last10_rng<=5)
    coil15=bool(dist_pct<=7 and last15_rng<=7 and vol_dry<0.8)
    return dict(dist_pct=round(dist_pct,1), last10_rng=round(last10_rng,1),
                last15_rng=round(last15_rng,1), vol_dry=round(vol_dry,2),
                coil10=coil10, coil15=coil15, w52h=r2(w52h))

# ── 7. Price vs All Key MAs ───────────────────────────────────────────────────
def compute_ma_alignment(df):
    """Price above/below 20/50/100/200 SMA — maximum bullish/bearish alignment."""
    if len(df)<25: return {}
    C=df["Close"].values.astype(float); n=len(df)
    def sma(p): return round(float(C[-p:].mean()),2) if n>=p else None
    ma20=sma(20); ma50=sma(50); ma100=sma(100); ma200=sma(200)
    last=C[-1]
    above=[ma20,ma50,ma100,ma200].count(lambda m: m is not None and last>m
                                        if callable(lambda:None) else 0)
    # Count how many MAs price is above
    above_count=sum(1 for m in [ma20,ma50,ma100,ma200] if m is not None and last>m)
    max4=bool(above_count==4)
    above_20_50=bool(ma20 and ma50 and last>ma20 and last>ma50)
    above_all3=bool(ma20 and ma50 and ma100 and last>ma20 and last>ma50 and last>ma100)
    all_bull=bool(ma20 and ma50 and ma100 and ma200
                  and last>ma20 and ma20>ma50 and ma50>ma100 and ma100>ma200)
    return dict(ma20=ma20 or 0.0, ma50=ma50 or 0.0,
                ma100=ma100 or 0.0, ma200=ma200 or 0.0,
                above_count=above_count, max4=max4,
                above_all3=bool(above_all3), all_bull=bool(all_bull))


# ══════════════════════════════════════════════════════════════════════════════
# 📐 NOISELESS — Definedge / TradePoint style (P&F · Renko · 3-Line Break · RRG · EW)
# ══════════════════════════════════════════════════════════════════════════════

# ── Point & Figure ────────────────────────────────────────────────────────────
def compute_pnf(df, reversal=3):
    """Point & Figure (1% auto box, 3-box reversal). Detects 14 P&F signals."""
    if len(df)<30: return {}
    C=df["Close"].values.astype(float); n=len(df)
    price=float(C[-1])
    box=max(price*0.01, 0.01)  # 1% of price, min 0.01

    def snap(p): return int(p/box)*box

    cols=[]  # [{'type':'X'/'O','high':float,'low':float}]
    prev=snap(C[0])
    for i in range(1,n):
        curr=snap(C[i])
        if not cols:
            if curr>=prev+box: cols.append({'type':'X','high':curr,'low':prev})
            elif curr<=prev-box: cols.append({'type':'O','high':prev,'low':curr})
            else: prev=curr
            continue
        last=cols[-1]
        if last['type']=='X':
            if curr>=last['high']+box: last['high']=curr
            elif curr<=last['high']-reversal*box:
                cols.append({'type':'O','high':last['high']-box,'low':curr})
        else:
            if curr<=last['low']-box: last['low']=curr
            elif curr>=last['low']+reversal*box:
                cols.append({'type':'X','high':curr,'low':last['low']+box})
        prev=curr

    if len(cols)<3: return {}
    xc=[c for c in cols if c['type']=='X']; oc=[c for c in cols if c['type']=='O']
    cur=cols[-1]; in_x=cur['type']=='X'; in_o=not in_x

    # Double Top/Bottom
    dbl_top=bool(in_x and len(xc)>=2 and xc[-1]['high']>xc[-2]['high'])
    dbl_bot=bool(in_o and len(oc)>=2 and oc[-1]['low']<oc[-2]['low'])
    # Triple Top/Bottom (previous two X/O columns had same high/low)
    tpl_top=bool(in_x and len(xc)>=3 and xc[-1]['high']>xc[-2]['high']
                 and abs(xc[-2]['high']-xc[-3]['high'])<=box)
    tpl_bot=bool(in_o and len(oc)>=3 and oc[-1]['low']<oc[-2]['low']
                 and abs(oc[-2]['low']-oc[-3]['low'])<=box)
    # Catapult: 3 consecutive X/O columns each making new extremes
    bull_cat=bool(in_x and len(xc)>=3 and xc[-1]['high']>xc[-2]['high']>xc[-3]['high'])
    bear_cat=bool(in_o and len(oc)>=3 and oc[-1]['low']<oc[-2]['low']<oc[-3]['low'])
    # High Pole: long X column (≥5 boxes) followed by O retracing >50%
    high_pole=False
    if in_o and len(cols)>=2:
        px=[c for c in cols[:-1] if c['type']=='X']
        if px:
            xh=px[-1]; xboxes=(xh['high']-xh['low'])/box
            if xboxes>=5 and (cur['high']-cur['low'])>xboxes*box*0.5: high_pole=True
    # Low Pole: long O column followed by X recovering >50%
    low_pole=False
    if in_x and len(cols)>=2:
        po=[c for c in cols[:-1] if c['type']=='O']
        if po:
            oh=po[-1]; oboxes=(oh['high']-oh['low'])/box
            if oboxes>=5 and (cur['high']-cur['low'])>oboxes*box*0.5: low_pole=True
    # Ascending Triangle: X tops same, O bottoms rising
    asc_tri=bool(len(xc)>=2 and len(oc)>=2
                 and abs(xc[-1]['high']-xc[-2]['high'])<=box
                 and oc[-1]['low']>oc[-2]['low'])
    # Descending Triangle: O bottoms same, X tops falling
    desc_tri=bool(len(oc)>=2 and len(xc)>=2
                  and abs(oc[-1]['low']-oc[-2]['low'])<=box
                  and xc[-1]['high']<xc[-2]['high'])
    # Long Tail Reversal (column ≥8 boxes then reversal)
    lt_bull=bool(in_x and len(cols)>=2
                 and any(c['type']=='O' and (c['high']-c['low'])/box>=8
                         for c in cols[-3:-1]))
    lt_bear=bool(in_o and len(cols)>=2
                 and any(c['type']=='X' and (c['high']-c['low'])/box>=8
                         for c in cols[-3:-1]))
    curr_boxes=round((cur['high']-cur['low'])/box)
    return dict(in_x=bool(in_x),in_o=bool(in_o),
                dbl_top=bool(dbl_top),dbl_bot=bool(dbl_bot),
                tpl_top=bool(tpl_top),tpl_bot=bool(tpl_bot),
                bull_cat=bool(bull_cat),bear_cat=bool(bear_cat),
                high_pole=bool(high_pole),low_pole=bool(low_pole),
                asc_tri=bool(asc_tri),desc_tri=bool(desc_tri),
                lt_bull=bool(lt_bull),lt_bear=bool(lt_bear),
                curr_boxes=curr_boxes,n_cols=len(cols),box=round(box,2))

# ── Renko ─────────────────────────────────────────────────────────────────────
def compute_renko(df):
    """Renko chart (ATR-based brick size, 2-brick reversal). Definedge-style."""
    if len(df)<20: return {}
    C=df["Close"].values.astype(float)
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    n=len(df); price=float(C[-1])
    atr14=float(np.mean(H[-15:-1]-L[-15:-1])) if n>=15 else float(np.mean(H-L))
    brick=max(atr14, price*0.005)

    # bricks: [(direction 1/-1, open, close)]
    bricks=[]
    def add_bricks(d, start, end):
        nb=int(abs(end-start)/brick)
        for j in range(nb):
            o=start+d*j*brick; c=start+d*(j+1)*brick
            bricks.append((d,o,c))
        return start+d*nb*brick

    curr=round(C[0]/brick)*brick
    for i in range(1,n):
        p=C[i]
        if not bricks:
            if p>=curr+brick:
                curr=add_bricks(1,curr,p)
            elif p<=curr-brick:
                curr=add_bricks(-1,curr,p)
            continue
        ld,lo,lc=bricks[-1]
        if ld==1:
            if p>=lc+brick: curr=add_bricks(1,lc,p)
            elif p<=lc-2*brick: curr=add_bricks(-1,lc,p)
        else:
            if p<=lc-brick: curr=add_bricks(-1,lc,p)
            elif p>=lc+2*brick: curr=add_bricks(1,lc,p)

    if not bricks: return {}
    ld,lo,lc=bricks[-1]
    # Consecutive same-direction count
    consec=0
    for b in reversed(bricks):
        if b[0]==ld: consec+=1
        else: break
    just_rev=bool(len(bricks)>=2 and bricks[-1][0]!=bricks[-2][0])
    # Double Top/Bottom on Renko
    up_closes=[b[2] for b in bricks if b[0]==1]
    dn_closes=[b[2] for b in bricks if b[0]==-1]
    renko_dbl_top=bool(len(up_closes)>=2 and abs(up_closes[-1]-up_closes[-2])<=brick*1.5)
    renko_dbl_bot=bool(len(dn_closes)>=2 and abs(dn_closes[-1]-dn_closes[-2])<=brick*1.5)
    return dict(bull=bool(ld==1),bear=bool(ld==-1),
                just_rev=bool(just_rev),consec=consec,
                punch3=bool(consec>=3),punch5=bool(consec>=5),
                dbl_top=bool(renko_dbl_top),dbl_bot=bool(renko_dbl_bot),
                n_bricks=len(bricks),brick=round(brick,2))

# ── 3-Line Break ──────────────────────────────────────────────────────────────
def compute_3lb(df, n=3):
    """3-Line Break chart (Japanese noiseless method). Reversal = n lines."""
    if len(df)<12: return {}
    C=df["Close"].values.astype(float)
    # lines: [(dir 1/-1, open_price, close_price)]
    if C[1]>C[0]: lines=[(1,C[0],C[1])]
    elif C[1]<C[0]: lines=[(-1,C[0],C[1])]
    else: lines=[(1,C[0],C[1])]
    for price in C[2:]:
        ld,lo,lc=lines[-1]
        recent=lines[-n:] if len(lines)>=n else lines[:]
        max_c=max(l[2] for l in recent); min_c=min(l[2] for l in recent)
        if ld==1:
            if price>lc: lines.append((1,lc,price))
            elif price<min_c: lines.append((-1,lc,price))
        else:
            if price<lc: lines.append((-1,lc,price))
            elif price>max_c: lines.append((1,lc,price))
    if not lines: return {}
    ld,lo,lc=lines[-1]
    just_rev=bool(len(lines)>=2 and lines[-1][0]!=lines[-2][0])
    consec=0
    for l in reversed(lines):
        if l[0]==ld: consec+=1
        else: break
    return dict(bull=bool(ld==1),bear=bool(ld==-1),
                just_rev=bool(just_rev),
                rev_bull=bool(just_rev and ld==1),
                rev_bear=bool(just_rev and ld==-1),
                consec=consec,n_lines=len(lines))

# ── Relative Rotation Graph ───────────────────────────────────────────────────
def compute_rrg(df, nifty_df):
    """RRG: RS-Ratio and RS-Momentum vs Nifty index.
    Quadrants: Leading / Weakening / Lagging / Improving."""
    if nifty_df is None or len(df)<25: return {}
    merged=pd.merge(df[['Date','Close']].rename(columns={'Close':'s'}),
                    nifty_df[['Date','Close']].rename(columns={'Close':'n'}),
                    on='Date',how='inner').sort_values('Date')
    if len(merged)<20: return {}
    stk=merged['s'].values.astype(float)
    nif=np.where(merged['n'].values.astype(float)==0,1,merged['n'].values.astype(float))
    rs=stk/nif; m=len(rs)
    # RS-Ratio: normalized around 100 using 10-period SMA
    if m<12: return {}
    sma10=np.array([rs[i-10:i].mean() for i in range(10,m+1)])
    rs_aligned=rs[10:]
    rs_ratio=rs_aligned/sma10*100 if sma10.any() else rs_aligned
    if len(rs_ratio)<12: return {}
    # RS-Momentum: RS-Ratio normalized by its own 10-period SMA
    sma10b=np.array([rs_ratio[i-10:i].mean() for i in range(10,len(rs_ratio)+1)])
    rsr_aligned=rs_ratio[10:]
    rs_mom=rsr_aligned/sma10b*100 if sma10b.any() else rsr_aligned
    if not len(rs_mom): return {}
    rr=round(float(rs_ratio[-1]),2); rm=round(float(rs_mom[-1]),2)
    leading  =rr>100 and rm>100
    weakening=rr>100 and rm<=100
    improving=rr<=100 and rm>100
    lagging  =rr<=100 and rm<=100
    # Rotation direction (last 3 periods)
    k=min(3,len(rs_mom)-1)
    rot_improving=bool(float(rs_mom[-1])>float(rs_mom[-k-1]))
    quad=('leading' if leading else 'weakening' if weakening
          else 'improving' if improving else 'lagging')
    return dict(rr=rr,rm=rm,quad=quad,
                leading=bool(leading),weakening=bool(weakening),
                improving=bool(improving),lagging=bool(lagging),
                rot_improving=bool(rot_improving))

# ── Simplified Elliott Wave (zigzag approximation) ───────────────────────────
def compute_ew_simple(df, swing_pct=0.05):
    """Simplified EW: zigzag pivot detection + wave count + Fibonacci check.
    Approximation only — not equivalent to Strike's proprietary system."""
    if len(df)<60: return {}
    C=df["Close"].values.astype(float); n=len(df)
    # Detect swing pivots via local extrema (5-bar lookback)
    lb=5; pivots=[]
    for i in range(lb,n-lb):
        if all(C[i]>=C[j] for j in range(i-lb,i+lb+1) if j!=i):
            pivots.append((i,C[i],1))   # swing high
        elif all(C[i]<=C[j] for j in range(i-lb,i+lb+1) if j!=i):
            pivots.append((i,C[i],-1))  # swing low
    # Filter to alternating + min swing size
    filt=[]
    for p in pivots:
        if not filt or filt[-1][2]!=p[2]:
            filt.append(p)
        elif p[2]==1 and p[1]>filt[-1][1]: filt[-1]=p
        elif p[2]==-1 and p[1]<filt[-1][1]: filt[-1]=p
    # Remove pivots with swing < swing_pct
    clean=[filt[0]] if filt else []
    for p in filt[1:]:
        if abs(p[1]-clean[-1][1])/clean[-1][1]>=swing_pct:
            clean.append(p)
    if len(clean)<5: return dict(w5_up=False,w5_dn=False,wave3_ext=False,
                                  abc_done=False,in_impulse=False,n_pivots=len(clean))
    p=clean[-5:]
    # Impulse up: L H L H L (potential 5th wave about to start or just completed)
    w5_up=bool(p[0][2]==-1 and p[1][2]==1 and p[2][2]==-1 and p[3][2]==1 and p[4][2]==-1
               and p[1][1]>p[0][1] and p[3][1]>p[1][1]  # HH
               and p[2][1]>p[0][1]                        # W4 doesn't overlap W1
               and (p[3][1]-p[2][1])>(p[1][1]-p[0][1]))  # W3 > W1
    # Impulse down
    w5_dn=bool(p[0][2]==1 and p[1][2]==-1 and p[2][2]==1 and p[3][2]==-1 and p[4][2]==1
               and p[1][1]<p[0][1] and p[3][1]<p[1][1]
               and p[2][1]<p[0][1]
               and (p[2][1]-p[3][1])>(p[0][1]-p[1][1]))
    # Wave 3 extension check: W3 > 1.618 × W1
    wave3_ext=False
    if len(clean)>=4:
        q=clean[-4:]
        if q[0][2]==-1 and q[1][2]==1 and q[2][2]==-1 and q[3][2]==1:
            w1=q[1][1]-q[0][1]; w3=q[3][1]-q[2][1]
            if w1>0 and w3>=w1*1.618: wave3_ext=True
    # ABC correction: 3-pivot A-B-C correction after impulse
    abc_done=False
    if len(clean)>=3:
        q=clean[-3:]
        if q[0][2]==1 and q[1][2]==-1 and q[2][2]==1:
            wA=q[0][1]-q[1][1]; wC=q[1][1]-q[2][1] if q[2][1]<q[1][1] else 0
            if 0<wC<=wA*1.05: abc_done=True  # C ≤ A (typical ABC correction done)
    in_impulse=bool(w5_up or w5_dn or wave3_ext)
    return dict(w5_up=bool(w5_up),w5_dn=bool(w5_dn),
                wave3_ext=bool(wave3_ext),abc_done=bool(abc_done),
                in_impulse=bool(in_impulse),n_pivots=len(clean))


# ══════════════════════════════════════════════════════════════════════════════
# 🧪 EXPERIMENTAL — Volume Spread Analysis (Tom Williams / Wyckoff VSA)
# ══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════
# BREAKOUT PRO — 10 paid-scanner compute functions
# ═══════════════════════════════════════════════════════════════════════════

def compute_vcp(df):
    """VCP — Mark Minervini's Volatility Contraction Pattern.
    Successive weekly range contractions with drying volume near 52W high."""
    if len(df)<60: return {}
    import pandas as _pd
    df2=df.copy(); df2['Date']=_pd.to_datetime(df2['Date']); df2=df2.set_index('Date')
    w=df2.resample('W').agg({'High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    if len(w)<6: return {}
    H=w['High'].values; L=w['Low'].values; C=w['Close'].values; V=w['Volume'].values
    rngs=[(H[i]-L[i])/(H[i]+L[i])*200 for i in range(len(w))]
    # Count consecutive range contractions in last 8 weeks
    seg=rngs[-8:]; cons=0
    for i in range(1,len(seg)):
        if seg[i]<seg[i-1]: cons+=1
        else: cons=0
    contractions=cons
    avg_v=float(np.mean(V[:-3])) if len(V)>3 else float(np.mean(V))
    recent_v=float(np.mean(V[-3:]))
    vol_dry=bool(avg_v>0 and recent_v<avg_v*0.75)
    high52=float(max(H[-52:])) if len(H)>=52 else float(max(H))
    near_hi=bool(C[-1]>=high52*0.85)
    return dict(
        is_vcp=bool(contractions>=2 and vol_dry and near_hi),
        is_vcp3=bool(contractions>=3 and vol_dry and near_hi),
        contractions=int(contractions),
        vol_dry=vol_dry, near_hi=near_hi,
        tight_pct=round(rngs[-1],2) if rngs else 0
    )

def compute_pocket_pivot(df):
    """Pocket Pivot — Gil Morales / Chris Kacher.
    Up-day volume exceeds highest down-day volume in prior 10 sessions."""
    if len(df)<14: return {}
    C=df['Close'].values.astype(float); V=df['Volume'].values.astype(float)
    is_up=bool(C[-1]>C[-2])
    max_dn=max((V[i] for i in range(-11,-1) if C[i]<C[i-1]),default=0)
    vol=float(V[-1])
    return dict(
        is_pp=bool(is_up and max_dn>0 and vol>max_dn),
        is_pp_strong=bool(is_up and max_dn>0 and vol>max_dn*1.5),
        vol_ratio=round(vol/max_dn,2) if max_dn else 0
    )

def compute_3week_tight(df):
    """3-Week Tight — O'Neil. Weekly closes within ±1.5% for 3+ consecutive weeks."""
    if len(df)<20: return {}
    import pandas as _pd
    df2=df.copy(); df2['Date']=_pd.to_datetime(df2['Date']); df2=df2.set_index('Date')
    wc=df2.resample('W').agg({'Close':'last'}).dropna()['Close'].values
    if len(wc)<5: return {}
    def _rng(arr): m=min(arr); return (max(arr)-m)/m*100 if m else 999
    r3=_rng(wc[-3:]); r4=_rng(wc[-4:]); r5=_rng(wc[-5:])
    return dict(
        is_3wt=bool(r3<=1.5), is_4wt=bool(r4<=1.5), is_5wt=bool(r5<=1.5),
        tight_pct=round(r3,2),
        tight_weeks=5 if r5<=1.5 else 4 if r4<=1.5 else 3 if r3<=1.5 else 0
    )

def compute_rs_new_high(df, nifty_df):
    """RS Line New High — IBD/MarketSmith concept.
    Stock's RS line (price/Nifty50) making a 52-week high."""
    if nifty_df is None or len(df)<252 or len(nifty_df)<252: return {}
    try:
        import pandas as _pd
        sd=df.copy(); sd.index=_pd.to_datetime(sd['Datetime'])
        nd=nifty_df.copy(); nd.index=_pd.to_datetime(nd['Datetime'])
        common=sd.index.intersection(nd.index)
        if len(common)<60: return {}
        sc=sd.loc[common,'Close'].values[-252:].astype(float)
        nc=nd.loc[common,'Close'].values[-252:].astype(float)
        if nc[-1]==0: return {}
        rs=sc/nc
        curr=float(rs[-1]); hi=float(max(rs))
        vs_hi=round((curr/hi-1)*100,2) if hi else 0
        rs_trend=bool(len(rs)>=65 and rs[-1]>rs[-65])
        rs_bo=bool(len(rs)>=65 and curr>float(max(rs[-65:-5])))
        return dict(
            is_rs_new_high=bool(curr>=hi*0.99),
            rs_trending=rs_trend,
            rs_breakout=rs_bo,
            vs_52w_hi=vs_hi
        )
    except: return {}

def compute_ad_score(df, periods=65):
    """IBD A/D Rating — institutional accumulation vs distribution over 13 weeks."""
    if len(df)<periods+2: return {}
    C=df['Close'].values[-periods:].astype(float)
    V=df['Volume'].values[-periods:].astype(float)
    avg_v=float(np.mean(V))
    score=0
    for i in range(1,len(C)):
        if V[i]>=avg_v:
            score += (1 if C[i]>C[i-1] else -1 if C[i]<C[i-1] else 0)
    mx=periods-1
    n=score/mx if mx else 0
    rating='A' if n>0.35 else 'B' if n>0.1 else 'C' if n>-0.1 else 'D' if n>-0.35 else 'E'
    return dict(
        score=int(score), rating=rating,
        is_accum=bool(rating in('A','B')),
        is_distrib=bool(rating in('D','E')),
        rating_val=dict(A=5,B=4,C=3,D=2,E=1).get(rating,3)
    )

def compute_hvb(df):
    """High Volume Breakout — 52W high close + 2× average volume."""
    if len(df)<55: return {}
    C=df['Close'].values.astype(float); H=df['High'].values.astype(float)
    V=df['Volume'].values.astype(float)
    hi52=float(max(H[-252:-1])) if len(H)>=252 else float(max(H[:-1]))
    avg50=float(np.mean(V[-51:-1]))
    vr=round(float(V[-1])/avg50,2) if avg50 else 0
    new_hi=bool(C[-1]>hi52)
    return dict(
        hvb=bool(new_hi and vr>=2.0),
        hvb_strong=bool(new_hi and vr>=3.0),
        near_hvb=bool(C[-1]>=hi52*0.98 and vr>=2.0),
        new_hi=new_hi, vol_ratio=vr, hi52=round(hi52,2)
    )

def compute_vdu(df):
    """Volume Dry-Up — volume contracting below 50% of avg while price holds."""
    if len(df)<25: return {}
    V=df['Volume'].values.astype(float); C=df['Close'].values.astype(float)
    avg20=float(np.mean(V[-21:-1])); rec5=float(np.mean(V[-5:]))
    vr=round(rec5/avg20,2) if avg20 else 1.0
    hold=bool(C[-1]>=float(max(C[-20:]))*0.97)
    min40=float(min(V[-41:-1])) if len(V)>=41 else float(min(V[:-1]))
    return dict(
        vdu=bool(vr<0.5),
        vdu_bullish=bool(vr<0.5 and hold),
        vol_40d_low=bool(V[-1]<=min40*1.1),
        price_hold=hold, vol_ratio=vr
    )

def compute_ema_stack(df):
    """EMA Power Stack — 9/21/50/200 EMA bullishly aligned with rising slopes."""
    if len(df)<205: return {}
    C=df['Close'].values.astype(float)
    def ema(d,n):
        e=np.empty(len(d)); e[0]=d[0]; k=2/(n+1)
        for i in range(1,len(d)): e[i]=d[i]*k+e[i-1]*(1-k)
        return e
    e9=ema(C,9); e21=ema(C,21); e50=ema(C,50); e200=ema(C,200)
    p=C[-1]
    full=bool(p>e9[-1]>e21[-1]>e50[-1]>e200[-1])
    partial=bool(p>e21[-1]>e50[-1]>e200[-1])
    sl=bool(e9[-1]>e9[-5] and e21[-1]>e21[-5] and e50[-1]>e50[-5])
    return dict(
        full_stack=full, partial_stack=partial, power_stack=bool(full and sl),
        slopes_bull=sl,
        e9=r2(e9[-1]), e21=r2(e21[-1]), e50=r2(e50[-1]), e200=r2(e200[-1])
    )

def compute_inside_week(df):
    """Inside Week — weekly bar fully contained inside prior week. Compression signal."""
    if len(df)<15: return {}
    import pandas as _pd
    df2=df.copy(); df2['Date']=_pd.to_datetime(df2['Date']); df2=df2.set_index('Date')
    w=df2.resample('W').agg({'High':'max','Low':'min','Close':'last'}).dropna()
    if len(w)<5: return {}
    H=w['High'].values; L=w['Low'].values; C=w['Close'].values
    iw1=bool(H[-1]<H[-2] and L[-1]>L[-2])
    iw2=bool(iw1 and len(H)>=3 and H[-2]<H[-3] and L[-2]>L[-3])
    avg10c=float(np.mean(C[-10:])) if len(C)>=10 else float(C[-1])
    return dict(
        iw1=iw1, iw2=iw2,
        iw_uptrend=bool(iw1 and C[-1]>avg10c)
    )

def compute_trendline_bo(df, lb=60):
    """Descending trendline breakout — price crosses above falling peak line."""
    if len(df)<lb: return {}
    H=df['High'].values[-lb:].astype(float)
    C=df['Close'].values[-lb:].astype(float)
    peaks=[(i,H[i]) for i in range(2,len(H)-2) if H[i]==max(H[i-2:i+3])]
    if len(peaks)<3: return dict(tl_breakout=False,is_desc=False,tl_val=0,dist=0)
    p3=peaks[-3:]
    xs=np.array([p[0] for p in p3],dtype=float)
    ys=np.array([p[1] for p in p3],dtype=float)
    slope=float(np.polyfit(xs,ys,1)[0])
    n=len(H)-1
    tl=float(ys[-1]+slope*(n-xs[-1]))
    desc=bool(slope<-0.001)
    bo=bool(desc and C[-1]>tl)
    return dict(
        tl_breakout=bo, is_desc=desc,
        tl_val=r2(tl),
        dist=round((C[-1]-tl)/tl*100,2) if tl else 0
    )



# ══ Smart Rank compute functions ══════════════════════════════
import numpy as np

def compute_seasonality(df):
    """Historical monthly performance — Definedge Seasonality Scanner equivalent."""
    if len(df)<200: return {}
    try:
        import pandas as _pd, datetime as _dt
        df2=df.copy(); df2['Date']=_pd.to_datetime(df2['Date'])
        df2=df2.set_index('Date')
        mc=df2['Close'].resample('ME').last().dropna()
        if len(mc)<13: return {}
        mr=(mc.pct_change()*100).dropna()
        months={}
        for m in range(1,13):
            v=mr[mr.index.month==m]
            if len(v)>=3:
                months[m]=dict(avg=round(float(v.mean()),2),
                               hit=round(float((v>0).mean()*100),0),
                               n=int(len(v)))
            else:
                months[m]=dict(avg=0,hit=50,n=0)
        cm=_dt.date.today().month
        cm_data=months.get(cm,{})
        bm=max(range(1,13),key=lambda x:months.get(x,{}).get('avg',-999))
        wm=min(range(1,13),key=lambda x:months.get(x,{}).get('avg',999))
        return dict(months=months,cm=cm,
                    cm_avg=cm_data.get('avg',0),cm_hit=cm_data.get('hit',50),
                    bm=bm,bm_avg=months.get(bm,{}).get('avg',0),
                    wm=wm,wm_avg=months.get(wm,{}).get('avg',0))
    except: return {}

def compute_mtf_matrix(df):
    """MTF Matrix Score — Definedge concept. D/W/M trend checks, 1=bull 0=bear.
    Checks per timeframe: price>MA50, MA50>MA200, price makes HH, RSI>50, positive slope."""
    if len(df)<210: return {}
    try:
        import pandas as _pd
        df2=df.copy(); df2['Date']=_pd.to_datetime(df2['Date'])
        df2=df2.set_index('Date')
        def _ma(s,n): return s.rolling(n).mean()
        def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
        def _rsi(s,n=14):
            d=s.diff(); g=d.clip(lower=0).rolling(n).mean(); l=(-d.clip(upper=0)).rolling(n).mean()
            return 100-100/(1+g/(l+1e-10))
        def _score(close,hi,lo):
            ma50=_ma(close,50).iloc[-1]; ma200=_ma(close,200).iloc[-1]
            ema21=_ema(close,21).iloc[-1]; rsi=_rsi(close).iloc[-1]
            c=float(close.iloc[-1]); p5=float(close.iloc[-6]) if len(close)>=6 else c
            s=0
            if c>ma50: s+=1
            if ma50>ma200: s+=1
            if c>ema21: s+=1
            if rsi>50: s+=1
            if c>p5: s+=1
            return int(s)
        # Daily
        d_score=_score(df2['Close'],df2['High'],df2['Low'])
        # Weekly
        wdf=df2.resample('W-FRI').agg({'Close':'last','High':'max','Low':'min'}).dropna()
        w_score=_score(wdf['Close'],wdf['High'],wdf['Low']) if len(wdf)>=210 else 0
        # Monthly
        mdf=df2.resample('ME').agg({'Close':'last','High':'max','Low':'min'}).dropna()
        m_score=_score(mdf['Close'],mdf['High'],mdf['Low']) if len(mdf)>=25 else 0
        total=d_score+w_score+m_score
        return dict(d=d_score,w=w_score,m=m_score,total=total,
                    pct=round(total/15*100))
    except: return {}

def compute_ath(df):
    """All-Time High scanner — price at ATH or approaching ATH."""
    if len(df)<5: return {}
    C=df['Close'].values.astype(float); H=df['High'].values.astype(float)
    ath=float(max(H[:-1])) if len(H)>1 else float(H[0])
    c=float(C[-1])
    dist=round((c-ath)/ath*100,2) if ath else 0
    return dict(
        is_ath=bool(c>ath),
        near_ath=bool(c>=ath*0.98),
        ath=round(ath,2),
        dist_pct=dist,
        consec_hi=int(sum(1 for i in range(len(C)-1,-1,-1) if C[i]>=C[i-1]*0.998) if len(C)>1 else 0)
    )

def compute_triple_compress(df):
    """Triple Compression — VDU + NR7 + BB Squeeze simultaneously.
    Pre-breakout setup: 3 independent compression signals firing together."""
    if len(df)<22: return {}
    C=df['Close'].values.astype(float); V=df['Volume'].values.astype(float)
    H=df['High'].values.astype(float); L=df['Low'].values.astype(float)
    # NR7: narrowest range in 7 bars
    rng=[H[i]-L[i] for i in range(len(H))]
    nr7=bool(rng[-1]==min(rng[-7:]))
    # Volume Dry-Up: recent 5-day avg < 60% of 20-day avg
    avg20=float(np.mean(V[-21:-1])); rec5=float(np.mean(V[-5:]))
    vdu=bool(avg20>0 and rec5<avg20*0.6)
    # BB Squeeze: bandwidth < 20th percentile of last 60 bars
    def bband(c,n=20,k=2):
        mn=np.array([np.mean(c[max(0,i-n):i]) for i in range(n,len(c)+1)])
        sd=np.array([np.std(c[max(0,i-n):i]) for i in range(n,len(c)+1)])
        return 2*k*sd/(mn+1e-10)*100  # bandwidth %
    if len(C)>=80:
        bw=bband(C)
        bb_sq=bool(bw[-1]<=np.percentile(bw[-60:],20))
    else:
        bb_sq=False
    count=sum([nr7,vdu,bb_sq])
    return dict(nr7=nr7,vdu=vdu,bb_sq=bb_sq,
                tc2=bool(count>=2),tc3=bool(count==3),count=int(count))

def compute_raw_returns(df):
    """Raw multi-period returns for momentum score normalization (universe-wide)."""
    if len(df)<5: return dict(r1m=0,r3m=0,r6m=0,r12m=0)
    C=df['Close'].values.astype(float)
    def ret(n): return round((C[-1]-C[-n-1])/C[-n-1]*100,2) if len(C)>n else 0.0
    return dict(r1m=ret(21),r3m=ret(63),r6m=ret(126),r12m=ret(252))




def compute_donchian(df, p20=20, p55=55):
    """Donchian Channel — Richard Donchian / Turtle Trading System."""
    if len(df)<p55+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    ub20=float(max(H[-p20-1:-1])); lb20=float(min(L[-p20-1:-1])); mid20=(ub20+lb20)/2
    ub55=float(max(H[-p55-1:-1])); lb55=float(min(L[-p55-1:-1])); mid55=(ub55+lb55)/2
    ub10=float(max(H[-11:-1]));   lb10=float(min(L[-11:-1]))
    c=C[-1]; bw20=round((ub20-lb20)/mid20*100,2) if mid20 else 0
    # Bandwidth history for squeeze detection
    bw_hist=[(max(H[max(0,i-p20):i])-min(L[max(0,i-p20):i]))/(max(H[max(0,i-p20):i])+min(L[max(0,i-p20):i]))*200
             for i in range(p20,n) if (max(H[max(0,i-p20):i])+min(L[max(0,i-p20):i]))>0]
    squeeze=bool(len(bw_hist)>=20 and bw20<=float(np.percentile(bw_hist[-60:],25))) if bw_hist else False
    return dict(ub20=r2(ub20),lb20=r2(lb20),mid20=r2(mid20),bw20=bw20,
                ub55=r2(ub55),lb55=r2(lb55),mid55=r2(mid55),
                dc20_bull=bool(c>ub20), dc20_bear=bool(c<lb20),
                dc55_bull=bool(c>ub55), dc55_bear=bool(c<lb55),
                squeeze=squeeze,
                pullback_mid20=bool(abs(c-mid20)/mid20<0.008) if mid20 else False,
                pct_from_ub=round((c-ub20)/ub20*100,2) if ub20 else 0)

def compute_gmma(df):
    """Guppy Multiple Moving Average — Daryl Guppy. 12 EMAs in 2 groups."""
    if len(df)<65: return {}
    C=df["Close"].values.astype(float)
    def _ema(d,p):
        e=np.empty(len(d)); e[0]=d[0]; k=2/(p+1)
        for i in range(1,len(d)): e[i]=d[i]*k+e[i-1]*(1-k)
        return float(e[-1])
    st=[_ema(C,p) for p in [3,5,8,10,12,15]]
    lt=[_ema(C,p) for p in [30,35,40,45,50,60]]
    # Previous day for cross detection
    Cp=C[:-1]
    st_p=[_ema(Cp,p) for p in [3,5,8,10,12,15]]
    lt_p=[_ema(Cp,p) for p in [30,35,40,45,50,60]]
    sta=sum(st)/6; lta=sum(lt)/6; sta_p=sum(st_p)/6; lta_p=sum(lt_p)/6
    bull=bool(sta>lta); bear=bool(sta<lta)
    return dict(
        bull=bull, bear=bear,
        cross_bull=bool(bull and sta_p<=lta_p),   # fresh bull cross today
        cross_bear=bool(bear and sta_p>=lta_p),   # fresh bear cross today
        all_bull=bool(min(st)>max(lt)),            # short-term group fully above long-term
        all_bear=bool(max(st)<min(lt)),
        compress=bool(abs(sta-lta)/lta<0.015 if lta else False),  # groups converging
        st_avg=r2(sta), lt_avg=r2(lta),
        spread_pct=round((sta-lta)/lta*100,2) if lta else 0)

def compute_pivot_bounce(df):
    """Pivot level proximity — classic floor pivot levels vs current price."""
    if len(df)<3: return {}
    H=float(df["High"].values[-2]); L=float(df["Low"].values[-2])
    Cp=float(df["Close"].values[-2]); price=float(df["Close"].values[-1])
    P=(H+L+Cp)/3; R=H-L
    S1=2*P-H; S2=P-R; S3=L-2*(H-P)
    R1=2*P-L; R2=P+R; R3=H+2*(P-L)
    tol=P*0.007
    def near(lvl): return bool(abs(price-lvl)<=tol)
    return dict(P=r2(P),S1=r2(S1),S2=r2(S2),S3=r2(S3),R1=r2(R1),R2=r2(R2),R3=r2(R3),
                near_P=near(P), near_S1=near(S1), near_S2=near(S2), near_S3=near(S3),
                near_R1=near(R1), near_R2=near(R2), near_R3=near(R3),
                above_P=bool(price>P), above_R1=bool(price>R1), above_R2=bool(price>R2),
                below_S1=bool(price<S1), below_S2=bool(price<S2),
                at_sup=bool(near(S1) or near(S2)), at_res=bool(near(R1) or near(R2)))



def _ema_series(arr, period):
    """Simple EMA over a numpy array."""
    e=np.empty(len(arr)); e[0]=arr[0]; k=2/(period+1)
    for i in range(1,len(arr)): e[i]=arr[i]*k+e[i-1]*(1-k)
    return e

def _ema_squeeze_tf(closes, periods):
    """Compute EMA squeeze for one timeframe given close array + list of EMA periods.
    Returns dict with squeeze flag, tightness %, individual EMA values, and stack direction."""
    n=len(closes)
    valid=[p for p in periods if n>=max(p,5)]
    if len(valid)<3: return {}
    emas={p:float(_ema_series(closes,p)[-1]) for p in valid}
    vals=list(emas.values())
    avg=sum(vals)/len(vals)
    if avg==0: return {}
    tight_pct=round((max(vals)-min(vals))/avg*100,2)
    c=float(closes[-1])
    # Bull stack: shorter EMAs above longer EMAs (sorted by period ascending)
    sorted_periods=sorted(valid)
    sorted_vals=[emas[p] for p in sorted_periods]
    bull_stack=bool(all(sorted_vals[i]>=sorted_vals[i+1] for i in range(len(sorted_vals)-1)))
    bear_stack=bool(all(sorted_vals[i]<=sorted_vals[i+1] for i in range(len(sorted_vals)-1)))
    return dict(
        emas={str(p):r2(v) for p,v in emas.items()},
        tight_pct=tight_pct,
        squeeze=bool(tight_pct<=2.5),       # all EMAs within 2.5% of each other
        squeeze_tight=bool(tight_pct<=1.0), # ultra-tight
        bull_stack=bull_stack, bear_stack=bear_stack,
        price_above_all=bool(c>max(vals)),
        price_below_all=bool(c<min(vals)),
        n_emas=len(valid),
    )

def compute_ema_squeeze(df, _wdf=None, _mdf=None):
    """10/20/50/100/150/200 EMA squeeze — daily/weekly/monthly.
    Squeeze = all loaded EMAs within 2.5% of each other (tight coil before expansion)."""
    out={}
    C=df['Close'].values.astype(float)
    out['d']=_ema_squeeze_tf(C,[10,20,50,100,150,200])
    if _wdf is not None and len(_wdf)>=15:
        out['w']=_ema_squeeze_tf(_wdf['Close'].values.astype(float),[10,20,50,100])
    if _mdf is not None and len(_mdf)>=15:
        out['m']=_ema_squeeze_tf(_mdf['Close'].values.astype(float),[10,20,50])
    return out

def _macd_5_35_tf(closes):
    """MACD(5,35,5) — fast scalper variant. Returns cross signals for one timeframe."""
    n=len(closes)
    if n<40: return {}
    e5=_ema_series(closes,5); e35=_ema_series(closes,35)
    macd=e5-e35; sig=_ema_series(macd,5); hist=macd-sig
    cur_m=float(macd[-1]); cur_s=float(sig[-1]); cur_h=float(hist[-1])
    prev_m=float(macd[-2]); prev_s=float(sig[-2])
    cross_bull=bool(cur_m>cur_s and prev_m<=prev_s)
    cross_bear=bool(cur_m<cur_s and prev_m>=prev_s)
    return dict(macd=round(cur_m,3), signal=round(cur_s,3), hist=round(cur_h,3),
                 bull=bool(cur_m>cur_s), bear=bool(cur_m<cur_s),
                 cross_bull=cross_bull, cross_bear=cross_bear,
                 zero_bull=bool(cur_m>0 and prev_m<=0),
                 zero_bear=bool(cur_m<0 and prev_m>=0))

def compute_macd_5_35(df, _wdf=None, _mdf=None):
    """MACD 5/35/5 crossover — daily/weekly/monthly. Faster variant for swing entries."""
    out={}
    out['d']=_macd_5_35_tf(df['Close'].values.astype(float))
    if _wdf is not None and len(_wdf)>=40:
        out['w']=_macd_5_35_tf(_wdf['Close'].values.astype(float))
    if _mdf is not None and len(_mdf)>=40:
        out['m']=_macd_5_35_tf(_mdf['Close'].values.astype(float))
    return out



def compute_vsa(df):
    """Volume Spread Analysis: interplay of Volume, Spread (H−L), and Close position.
    The 'Big Volume Candle' strategy is the centrepiece."""
    if len(df)<25: return {}
    O=df["Open"].values.astype(float); H=df["High"].values.astype(float)
    L=df["Low"].values.astype(float); C=df["Close"].values.astype(float)
    V=df["Volume"].values.astype(float); n=len(df)
    spread=float(H[-1]-L[-1])
    if spread<=0: spread=0.0001
    close_pos=float((C[-1]-L[-1])/spread)        # 0.0(bottom) to 1.0(top)
    avg_sp=float(np.mean(H[-21:-1]-L[-21:-1])) if n>=21 else float(np.mean(H-L))
    avg_vol=float(V[-21:-1].mean()) if n>=21 else float(V.mean())
    sr=spread/max(avg_sp,0.0001); vr=float(V[-1])/max(avg_vol,1)
    bull=C[-1]>O[-1]; bear=C[-1]<O[-1]
    wide=sr>1.5; narrow=sr<0.7
    hi_vol=vr>1.5; ultra_vol=vr>2.5; lo_vol=vr<0.7
    max20v=float(V[-20:].max()) if n>=20 else float(V.max())
    # ── Signal Map ──────────────────────────────────────────────────────────
    # Big Volume Bull: ultra high vol + wide spread + close in upper 60% — institutions buying
    big_vol_bull=bool(ultra_vol and wide and close_pos>0.6)
    # Big Volume Bear: ultra high vol + wide spread + close in lower 40% — institutions selling
    big_vol_bear=bool(ultra_vol and wide and close_pos<0.4)
    # Big Volume Indecision: ultra high vol + close in middle (40-60%) = battle zone
    big_vol_indecision=bool(ultra_vol and 0.4<=close_pos<=0.6)
    # Effort to Rise: wide up bar + high vol + close upper half
    effort_up=bool(bull and wide and hi_vol and close_pos>0.5)
    # Effort to Fall: wide down bar + high vol + close lower half
    effort_dn=bool(bear and wide and hi_vol and close_pos<0.5)
    # Stopping Volume: ultra high vol + down bar but closes mid/upper = supply being absorbed
    stopping_vol=bool(ultra_vol and bear and close_pos>0.4)
    # Upthrust: wide spread, but closes near the LOW (bearish reversal at top)
    upthrust=bool(wide and bear and hi_vol and close_pos<0.35)
    # Spring / Shakeout: wide spread bar, close near the HIGH (bullish reversal at bottom)
    spring=bool(wide and bull and close_pos>0.65 and hi_vol)
    # Climactic Selling: highest vol in 20 days + wide down bar + close near low
    climax_sell=bool(V[-1]>=max20v*0.9 and wide and bear and close_pos<0.35)
    # Climactic Buying (Demand Absorption): highest vol + wide up bar + close near high
    climax_buy=bool(V[-1]>=max20v*0.9 and wide and bull and close_pos>0.65)
    # No Demand: narrow up bar + below avg vol (weak bounce, institutions not buying)
    no_demand=bool(bull and narrow and lo_vol)
    # No Supply: narrow down bar + below avg vol (sellers dried up, pre-breakout)
    no_supply=bool(bear and narrow and lo_vol)
    # Test for Supply: low vol down day after a high vol up day (supply tested & absent)
    test_supply=bool(lo_vol and bear and close_pos>0.5 and n>=2 and float(V[-2])>avg_vol*1.4)
    # Volume Trap: high vol but minimal price progress (equal battle, no directional edge)
    vol_trap=bool(hi_vol and abs(C[-1]-O[-1])/spread<0.2)
    # Pseudo Upthrust: stock makes new high intraday then closes near the low on high vol
    pseudo_ut=bool(hi_vol and close_pos<0.3 and H[-1]>float(np.max(H[-6:-1])))
    return dict(
        vol_ratio=round(vr,2), spread_ratio=round(sr,2),
        close_pos=round(close_pos*100,1),
        big_vol_bull=bool(big_vol_bull), big_vol_bear=bool(big_vol_bear),
        big_vol_indecision=bool(big_vol_indecision),
        effort_up=bool(effort_up), effort_dn=bool(effort_dn),
        stopping_vol=bool(stopping_vol), upthrust=bool(upthrust),
        spring=bool(spring),
        climax_sell=bool(climax_sell), climax_buy=bool(climax_buy),
        no_demand=bool(no_demand), no_supply=bool(no_supply),
        test_supply=bool(test_supply), vol_trap=bool(vol_trap),
        pseudo_ut=bool(pseudo_ut),
    )


# ══════════════════════════════════════════════════════════════════════════════
# India Pro additions — HMA · Keltner · MFI · CCI · Heikin Ashi · LinReg
# ══════════════════════════════════════════════════════════════════════════════

def _wma(arr, p):
    """Weighted moving average."""
    n=len(arr); out=np.full(n, np.nan); w=np.arange(1,p+1,dtype=float); ws=w.sum()
    for i in range(p-1,n): out[i]=np.dot(arr[i-p+1:i+1],w)/ws
    return out

def compute_hma(df, period=20):
    """Hull Moving Average — anti-lag MA: HMA(n)=WMA(2×WMA(n/2)−WMA(n),√n)."""
    if len(df)<period+10: return {}
    C=df["Close"].values.astype(float); n=len(df)
    half=max(period//2,2); sq=max(int(np.sqrt(period)),2)
    w1=_wma(C,half); w2=_wma(C,period)
    # Align w1 and w2 to same length (w2 is shorter)
    w1a=w1[len(w1)-len(w2):]    # trim w1 to match w2 length
    diff=2*w1a-w2
    valid=np.where(~np.isnan(diff))[0]
    if len(valid)<sq: return {}
    d_clean=diff[valid]
    hma_raw=_wma(d_clean,sq)
    if len(hma_raw)<2: return {}
    cur=float(hma_raw[-1]); prev=float(hma_raw[-2])
    prev2=float(hma_raw[-3]) if len(hma_raw)>=3 else prev
    bull=bool(C[-1]>cur and cur>prev)
    bear=bool(C[-1]<cur and cur<prev)
    just_flip_bull=bool(cur>prev and prev<=prev2)
    just_flip_bear=bool(cur<prev and prev>=prev2)
    return dict(hma=r2(cur),bull=bool(bull),bear=bool(bear),
                flip_bull=bool(just_flip_bull),flip_bear=bool(just_flip_bear),
                dist=round((C[-1]-cur)/cur*100,2) if cur else 0.0)

def compute_keltner(df, ema_p=20, atr_p=10, mult=2.0):
    """Keltner Channels: Upper=EMA(20)+2×ATR(10), Lower=EMA(20)−2×ATR(10)."""
    if len(df)<ema_p+atr_p+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    a=2/(ema_p+1); ema=np.zeros(n); ema[0]=C[0]
    for i in range(1,n): ema[i]=a*C[i]+(1-a)*ema[i-1]
    tr=np.maximum(H[1:]-L[1:],np.maximum(abs(H[1:]-C[:-1]),abs(L[1:]-C[:-1])))
    tr=np.concatenate([[H[0]-L[0]],tr])
    atr=np.zeros(n); atr[atr_p-1]=tr[:atr_p].mean()
    for i in range(atr_p,n): atr[i]=(atr[i-1]*(atr_p-1)+tr[i])/atr_p
    upper=ema+mult*atr; lower=ema-mult*atr
    cur_u=float(upper[-1]); cur_m=float(ema[-1]); cur_l=float(lower[-1])
    last=C[-1]
    above=bool(last>cur_u); below=bool(last<cur_l)
    # BB inside Keltner (TTM Squeeze upgrade)
    std20=float(C[-20:].std()) if n>=20 else float(C.std())
    bb_u=cur_m+2*std20; bb_l=cur_m-2*std20
    squeezed=bool(bb_u<cur_u and bb_l>cur_l)
    # Just crossed upper/lower
    cross_up=bool(last>cur_u and C[-2]<=float(upper[-2])) if n>=2 else False
    cross_dn=bool(last<cur_l and C[-2]>=float(lower[-2])) if n>=2 else False
    return dict(upper=r2(cur_u),mid=r2(cur_m),lower=r2(cur_l),
                above=bool(above),below=bool(below),squeezed=bool(squeezed),
                cross_up=bool(cross_up),cross_dn=bool(cross_dn),
                pct_b=round((last-cur_l)/(cur_u-cur_l)*100,1) if cur_u>cur_l else 50.0)

def compute_mfi(df, period=14):
    """Money Flow Index — volume-weighted RSI. MFI≤20=oversold, ≥80=overbought."""
    if len(df)<period+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); V=df["Volume"].values.astype(float)
    n=len(df)
    tp=(H+L+C)/3; mf=tp*V
    mfi_vals=np.zeros(n)
    for i in range(period,n):
        pos=sum(mf[j] for j in range(i-period+1,i+1) if tp[j]>tp[j-1])
        neg=sum(mf[j] for j in range(i-period+1,i+1) if tp[j]<tp[j-1])
        mfi_vals[i]=100-(100/(1+pos/max(neg,0.001)))
    cur=round(float(mfi_vals[-1]),1)
    ob=bool(cur>=80); os_=bool(cur<=20)
    # Divergence (20-bar)
    lb=min(20,n-1)
    pr_seg=C[-lb:]; mfi_seg=mfi_vals[-lb:]
    lo=[i for i in range(2,lb-2) if pr_seg[i]<pr_seg[i-1] and pr_seg[i]<pr_seg[i+1]]
    hi=[i for i in range(2,lb-2) if pr_seg[i]>pr_seg[i-1] and pr_seg[i]>pr_seg[i+1]]
    bull_div=bool(len(lo)>=2 and pr_seg[lo[-1]]<pr_seg[lo[-2]] and mfi_seg[lo[-1]]>mfi_seg[lo[-2]])
    bear_div=bool(len(hi)>=2 and pr_seg[hi[-1]]>pr_seg[hi[-2]] and mfi_seg[hi[-1]]<mfi_seg[hi[-2]])
    return dict(mfi=cur,overbought=bool(ob),oversold=bool(os_),
                bull_div=bool(bull_div),bear_div=bool(bear_div))

def compute_cci(df, period=20):
    """Commodity Channel Index — (TP-SMA)/(0.015×MeanDev). >100=OB, <-100=OS."""
    if len(df)<period+5: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    tp=(H+L+C)/3; cci_v=np.zeros(n)
    for i in range(period-1,n):
        sma=tp[i-period+1:i+1].mean()
        mad=np.mean(np.abs(tp[i-period+1:i+1]-sma))
        cci_v[i]=(tp[i]-sma)/(0.015*max(mad,0.0001))
    cur=round(float(cci_v[-1]),1)
    ob=bool(cur>=100); os_=bool(cur<=-100)
    zero_cross_up=bool(n>=2 and cci_v[-1]>0 and cci_v[-2]<=0)
    zero_cross_dn=bool(n>=2 and cci_v[-1]<0 and cci_v[-2]>=0)
    ob_exit=bool(n>=2 and cci_v[-1]<100 and cci_v[-2]>=100)
    os_exit=bool(n>=2 and cci_v[-1]>-100 and cci_v[-2]<=-100)
    return dict(cci=cur,overbought=bool(ob),oversold=bool(os_),
                zero_up=bool(zero_cross_up),zero_dn=bool(zero_cross_dn),
                ob_exit=bool(ob_exit),os_exit=bool(os_exit))

def compute_heikin_ashi(df):
    """Heikin Ashi smoothed candles. Consecutive HA bull/bear runs, HA doji, reversal."""
    if len(df)<5: return {}
    O=df["Open"].values.astype(float); H=df["High"].values.astype(float)
    L=df["Low"].values.astype(float); C=df["Close"].values.astype(float)
    n=len(df)
    ha_c=(O+H+L+C)/4
    ha_o=np.zeros(n); ha_o[0]=(O[0]+C[0])/2
    for i in range(1,n): ha_o[i]=(ha_o[i-1]+ha_c[i-1])/2
    ha_h=np.maximum(H,np.maximum(ha_o,ha_c))
    ha_l=np.minimum(L,np.minimum(ha_o,ha_c))
    # HA candle direction
    ha_bull=ha_c>ha_o; ha_bear=ha_c<ha_o
    # No lower shadow on HA bull = strong bull (HA_Low = HA_Open)
    strong_bull=bool(ha_bull[-1] and abs(ha_l[-1]-ha_o[-1])<(ha_h[-1]-ha_l[-1])*0.05)
    strong_bear=bool(ha_bear[-1] and abs(ha_h[-1]-ha_o[-1])<(ha_h[-1]-ha_l[-1])*0.05)
    # HA Doji: small body relative to range
    body=abs(ha_c[-1]-ha_o[-1]); rng=ha_h[-1]-ha_l[-1]
    ha_doji=bool(rng>0 and body/rng<0.1)
    # Consecutive same-direction HA candles
    consec_bull=0; consec_bear=0
    for i in range(n-1,-1,-1):
        if ha_bull[i]: consec_bull+=1
        else: break
    for i in range(n-1,-1,-1):
        if ha_bear[i]: consec_bear+=1
        else: break
    just_rev=bool(n>=2 and ha_bull[-1]!=ha_bull[-2])
    rev_bull=bool(just_rev and ha_bull[-1])
    rev_bear=bool(just_rev and ha_bear[-1])
    return dict(bull=bool(ha_bull[-1]),bear=bool(ha_bear[-1]),
                strong_bull=bool(strong_bull),strong_bear=bool(strong_bear),
                ha_doji=bool(ha_doji),consec_bull=int(consec_bull),
                consec_bear=int(consec_bear),
                rev_bull=bool(rev_bull),rev_bear=bool(rev_bear))

def compute_linreg(df, period=50):
    """Linear Regression Channel — LR line ± 2-sigma bands over N periods."""
    if len(df)<period+5: return {}
    C=df["Close"].values.astype(float); n=len(df)
    lb=min(period,n); seg=C[-lb:]
    x=np.arange(lb,dtype=float)
    slope,intercept=np.polyfit(x,seg,1)
    fitted=slope*x+intercept
    resid=seg-fitted; std=float(resid.std())
    curr_fit=float(fitted[-1])
    dev=(C[-1]-curr_fit)/std if std>0 else 0.0
    above_2s=bool(dev>2); below_2s=bool(dev<-2)
    above_chan=bool(C[-1]>curr_fit+2*std)
    below_chan=bool(C[-1]<curr_fit-2*std)
    slope_bull=bool(slope>0); slope_bear=bool(slope<0)
    return dict(slope=round(slope,3),curr_fit=r2(curr_fit),
                dev=round(dev,2),std=r2(std),
                above_2s=bool(above_2s),below_2s=bool(below_2s),
                above_chan=bool(above_chan),below_chan=bool(below_chan),
                slope_bull=bool(slope_bull),slope_bear=bool(slope_bear))

# ══════════════════════════════════════════════════════════════════════════════
# 📐 Noiseless addition — Kagi Chart
# ══════════════════════════════════════════════════════════════════════════════
def compute_kagi(df):
    """Kagi chart: Yang(up)/Yin(down) lines with ATR-based reversal threshold."""
    if len(df)<20: return {}
    C=df["Close"].values.astype(float)
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float); n=len(df)
    atr14=float(np.mean(H[-15:-1]-L[-15:-1])) if n>=15 else float(np.mean(H-L))
    rev=max(atr14*1.5, C[-1]*0.015)   # reversal threshold
    # lines: [(dir 1/-1, start_price, end_price)]
    lines=[]; curr_dir=0; curr_start=C[0]; curr_end=C[0]
    for p in C[1:]:
        if curr_dir==0:
            if p>curr_end+rev: curr_dir=1; curr_end=p
            elif p<curr_end-rev: curr_dir=-1; curr_end=p
        elif curr_dir==1:
            if p>curr_end: curr_end=p
            elif p<curr_end-rev:
                lines.append((1,curr_start,curr_end))
                curr_dir=-1; curr_start=curr_end; curr_end=p
        else:
            if p<curr_end: curr_end=p
            elif p>curr_end+rev:
                lines.append((-1,curr_start,curr_end))
                curr_dir=1; curr_start=curr_end; curr_end=p
    if curr_dir!=0: lines.append((curr_dir,curr_start,curr_end))
    if not lines: return {}
    ld=lines[-1][0]; just_rev=bool(len(lines)>=2 and lines[-1][0]!=lines[-2][0])
    consec=0
    for l in reversed(lines):
        if l[0]==ld: consec+=1
        else: break
    # Shoulder/Waist: Yang line that breaks above prior Yang high = Shoulder (bullish)
    yang=[l for l in lines if l[0]==1]; yin=[l for l in lines if l[0]==-1]
    shoulder=bool(ld==1 and len(yang)>=2 and yang[-1][2]>yang[-2][2])
    waist=bool(ld==-1 and len(yin)>=2 and yin[-1][2]<yin[-2][2])
    return dict(yang=bool(ld==1),yin=bool(ld==-1),just_rev=bool(just_rev),
                rev_yang=bool(just_rev and ld==1),rev_yin=bool(just_rev and ld==-1),
                shoulder=bool(shoulder),waist=bool(waist),
                consec=consec,n_lines=len(lines))

# ══════════════════════════════════════════════════════════════════════════════
# 🎨 Patterns tab — Harmonic Patterns + Chart Patterns
# ══════════════════════════════════════════════════════════════════════════════
def _zigzag_pivots(H, L, C, lb=4, swing_pct=0.03):
    """Shared zigzag pivot detection for pattern functions."""
    n=len(C); pivots=[]
    for i in range(lb,n-lb):
        if all(H[i]>=H[j] for j in range(i-lb,i+lb+1) if j!=i):
            pivots.append((i,H[i],1))
        elif all(L[i]<=L[j] for j in range(i-lb,i+lb+1) if j!=i):
            pivots.append((i,L[i],-1))
    filt=[]
    for p in pivots:
        if not filt or filt[-1][2]!=p[2]: filt.append(p)
        elif p[2]==1 and p[1]>filt[-1][1]: filt[-1]=p
        elif p[2]==-1 and p[1]<filt[-1][1]: filt[-1]=p
    clean=[filt[0]] if filt else []
    for p in filt[1:]:
        if abs(p[1]-clean[-1][1])/max(clean[-1][1],0.01)>=swing_pct:
            clean.append(p)
    return clean

def compute_harmonics(df, tol=0.10):
    """Harmonic price patterns using zigzag pivots + Fibonacci ratio validation.
    Detects: Gartley · Bat · Butterfly · Crab · Cypher · AB=CD."""
    if len(df)<60: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float)
    clean=_zigzag_pivots(H,L,C)
    results=dict(gartley_bull=False,gartley_bear=False,
                 bat_bull=False,bat_bear=False,
                 butterfly_bull=False,butterfly_bear=False,
                 crab_bull=False,crab_bear=False,
                 cypher_bull=False,cypher_bear=False,
                 abcd_bull=False,abcd_bear=False,
                 n_pivots=len(clean))
    if len(clean)<5: return results

    def chk(r,t): return abs(r-t)/max(t,0.001)<=tol
    def rng(r,lo,hi): return (lo*(1-tol))<=r<=(hi*(1+tol))

    for k in range(max(0,len(clean)-7), len(clean)-4):
        X,A,B,C_,D = clean[k],clean[k+1],clean[k+2],clean[k+3],clean[k+4]
        XA=abs(A[1]-X[1]); AB=abs(B[1]-A[1]); BC=abs(C_[1]-B[1]); CD=abs(D[1]-C_[1])
        AD=abs(D[1]-A[1])
        if XA<0.001 or AB<0.001 or BC<0.001: continue
        ab_xa=AB/XA; bc_ab=BC/AB; cd_bc=CD/BC if BC>0 else 0; ad_xa=AD/XA
        # Setup: XABCD alternating direction
        bull_d=(X[2]==1 and A[2]==-1 and B[2]==1 and C_[2]==-1)  # D completes at low → bounce up
        bear_d=(X[2]==-1 and A[2]==1 and B[2]==-1 and C_[2]==1)  # D at high → drop down
        if not(bull_d or bear_d): continue
        tag=True if bull_d else False  # True=bullish pattern
        # AB=CD
        if rng(cd_bc,0.886,1.272):
            if tag: results['abcd_bull']=True
            else: results['abcd_bear']=True
        # Gartley: AB/XA≈0.618, AD/XA≈0.786
        if chk(ab_xa,0.618) and rng(bc_ab,0.382,0.886) and rng(cd_bc,1.13,1.618) and chk(ad_xa,0.786):
            if tag: results['gartley_bull']=True
            else: results['gartley_bear']=True
        # Bat: AB/XA=0.382-0.5, AD/XA≈0.886
        if rng(ab_xa,0.382,0.50) and rng(bc_ab,0.382,0.886) and rng(cd_bc,1.618,2.618) and chk(ad_xa,0.886):
            if tag: results['bat_bull']=True
            else: results['bat_bear']=True
        # Butterfly: AB/XA≈0.786, AD/XA=1.272-1.618
        if chk(ab_xa,0.786) and rng(bc_ab,0.382,0.886) and rng(cd_bc,1.618,2.24) and rng(ad_xa,1.272,1.618):
            if tag: results['butterfly_bull']=True
            else: results['butterfly_bear']=True
        # Crab: AD/XA≈1.618
        if rng(ab_xa,0.382,0.618) and rng(bc_ab,0.382,0.886) and rng(cd_bc,2.618,3.618) and chk(ad_xa,1.618):
            if tag: results['crab_bull']=True
            else: results['crab_bear']=True
        # Cypher: AB/XA=0.382-0.618, BC/XA=1.13-1.414 (BC extends past XA), CD/XC≈0.786
        if len(clean)>k+4:
            XC=abs(C_[1]-X[1])
            if XC>0:
                bc_xa=abs(B[1]-C_[1])/XA if XA else 0
                cd_xc=CD/XC
                if rng(ab_xa,0.382,0.618) and rng(bc_xa,1.13,1.414) and chk(cd_xc,0.786):
                    if tag: results['cypher_bull']=True
                    else: results['cypher_bear']=True
    return results


def _detect_wm(H, L, C, kind, tol=0.04, lb=4, swing_pct=0.03):
    """Detect Double Bottom (W) or Double Top (M) pattern from OHLC arrays.
    kind: 'W' for double bottom, 'M' for double top.
    Returns dict with pattern flag, neckline level, breakout confirmation, depth %."""
    n=len(C)
    if n<20: return {}
    clean=_zigzag_pivots(H,L,C,lb=lb,swing_pct=swing_pct)
    c=float(C[-1])
    if kind=='W':
        lows=[(p[0],p[1]) for p in clean if p[2]==-1]
        highs=[(p[0],p[1]) for p in clean if p[2]==1]
        if len(lows)<2 or len(highs)<1: return dict(found=False)
        l1,l2=lows[-2],lows[-1]
        # Two lows at a similar level
        if abs(l1[1]-l2[1])/max(l1[1],0.01)>tol: return dict(found=False)
        # Middle peak between the two lows (the "neckline")
        mids=[h for h in highs if l1[0]<h[0]<l2[0]]
        if not mids: return dict(found=False)
        neck=max(m[1] for m in mids)
        depth=(neck-min(l1[1],l2[1]))/neck*100 if neck else 0
        if depth<3: return dict(found=False)  # too shallow to be meaningful
        return dict(found=True, neckline=r2(neck),
                     breakout=bool(c>neck), near_breakout=bool(c>=neck*0.98),
                     depth_pct=round(depth,2))
    else:  # 'M'
        highs=[(p[0],p[1]) for p in clean if p[2]==1]
        lows=[(p[0],p[1]) for p in clean if p[2]==-1]
        if len(highs)<2 or len(lows)<1: return dict(found=False)
        h1,h2=highs[-2],highs[-1]
        if abs(h1[1]-h2[1])/max(h1[1],0.01)>tol: return dict(found=False)
        mids=[l for l in lows if h1[0]<l[0]<h2[0]]
        if not mids: return dict(found=False)
        neck=min(m[1] for m in mids)
        depth=(max(h1[1],h2[1])-neck)/max(h1[1],h2[1])*100 if max(h1[1],h2[1]) else 0
        if depth<3: return dict(found=False)
        return dict(found=True, neckline=r2(neck),
                     breakdown=bool(c<neck), near_breakdown=bool(c<=neck*1.02),
                     depth_pct=round(depth,2))

def compute_double_patterns(df, _wdf=None, _mdf=None):
    """Double Bottom (W) and Double Top (M) — Daily / Weekly / Monthly."""
    H=df['High'].values.astype(float); L=df['Low'].values.astype(float); C=df['Close'].values.astype(float)
    out={'d':{'w':_detect_wm(H,L,C,'W'), 'm':_detect_wm(H,L,C,'M')}}
    if _wdf is not None and len(_wdf)>=20:
        wH=_wdf['High'].values.astype(float); wL=_wdf['Low'].values.astype(float); wC=_wdf['Close'].values.astype(float)
        out['w']={'w':_detect_wm(wH,wL,wC,'W',lb=2,swing_pct=0.04), 'm':_detect_wm(wH,wL,wC,'M',lb=2,swing_pct=0.04)}
    if _mdf is not None and len(_mdf)>=20:
        mH=_mdf['High'].values.astype(float); mL=_mdf['Low'].values.astype(float); mC=_mdf['Close'].values.astype(float)
        out['m']={'w':_detect_wm(mH,mL,mC,'W',lb=2,swing_pct=0.05), 'm':_detect_wm(mH,mL,mC,'M',lb=2,swing_pct=0.05)}
    return out


def compute_chart_patterns(df, tol=0.05):
    """Chart pattern detection: H&S, Inv H&S, Wedges, Flags, Double Top/Bottom."""
    if len(df)<50: return {}
    H=df["High"].values.astype(float); L=df["Low"].values.astype(float)
    C=df["Close"].values.astype(float); n=len(df)
    clean=_zigzag_pivots(H,L,C)
    hs=inv_hs=rise_wedge=fall_wedge=bull_flag=bear_flag=False
    dbl_top=dbl_bot=False

    if len(clean)>=5:
        highs=[(p[0],p[1]) for p in clean if p[2]==1]
        lows=[(p[0],p[1]) for p in clean if p[2]==-1]
        # Head & Shoulders (3 highs, middle is tallest, shoulders at similar level)
        if len(highs)>=3:
            LS,HD,RS=highs[-3],highs[-2],highs[-1]
            if HD[1]>LS[1] and HD[1]>RS[1] and abs(LS[1]-RS[1])/HD[1]<tol:
                hs=True
        # Inverse H&S
        if len(lows)>=3:
            LS,HD,RS=lows[-3],lows[-2],lows[-1]
            if HD[1]<LS[1] and HD[1]<RS[1] and abs(LS[1]-RS[1])/max(abs(HD[1]),0.01)<tol:
                inv_hs=True
        # Double Top (2 highs at same level)
        if len(highs)>=2:
            h1,h2=highs[-2],highs[-1]
            if abs(h1[1]-h2[1])/h1[1]<tol: dbl_top=True
        # Double Bottom (2 lows at same level)
        if len(lows)>=2:
            l1,l2=lows[-2],lows[-1]
            if abs(l1[1]-l2[1])/max(l1[1],0.01)<tol: dbl_bot=True
        # Wedges (converging trendlines from last 2 highs + 2 lows)
        if len(highs)>=2 and len(lows)>=2:
            rh,rl=highs[-2:],lows[-2:]
            hs_slope=(rh[1][1]-rh[0][1])/max(rh[0][1],0.01)
            ls_slope=(rl[1][1]-rl[0][1])/max(rl[0][1],0.01)
            converging=abs(hs_slope-ls_slope)>0.005 and hs_slope*ls_slope>0  # same direction
            if converging:
                if hs_slope>0 and ls_slope>0 and ls_slope>hs_slope: rise_wedge=True
                elif hs_slope<0 and ls_slope<0 and hs_slope>ls_slope: fall_wedge=True

    # Flags (last 25 bars: strong pole then tight consolidation)
    if n>=25:
        pole_h=float(np.max(C[-25:-10])); pole_l=float(np.min(C[-25:-10]))
        flag_h=float(np.max(C[-10:])); flag_l=float(np.min(C[-10:]))
        pole_mv=(pole_h-pole_l)/max(pole_l,0.01)
        flag_rng=(flag_h-flag_l)/max(flag_l,0.01)
        if pole_mv>0.07 and flag_rng<pole_mv*0.5:
            if C[-25]<C[-10]: bull_flag=True   # upward pole
        if pole_mv>0.07 and flag_rng<pole_mv*0.5:
            if C[-25]>C[-10]: bear_flag=True   # downward pole

    return dict(hs=bool(hs),inv_hs=bool(inv_hs),
                dbl_top=bool(dbl_top),dbl_bot=bool(dbl_bot),
                rise_wedge=bool(rise_wedge),fall_wedge=bool(fall_wedge),
                bull_flag=bool(bull_flag),bear_flag=bool(bear_flag),
                n_pivots=len(clean))

# ══════════════════════════════════════════════════════════════════════════════
# 📡 Market Breadth — universe-wide signals (computed after all stocks)
# ══════════════════════════════════════════════════════════════════════════════
def assign_breadth(stocks):
    """Compute market-wide breadth metrics — stored ONCE as global, not per-stock."""
    if not stocks: return
    N=len(stocks)
    def pct(cond):
        try: return round(sum(1 for s in stocks if cond(s))/N*100,1)
        except: return 0.0
    above200   = pct(lambda s: s.get('above200',False))
    stage2     = pct(lambda s: s.get('mi',{}).get('stg',0)==2)
    near52wh   = pct(lambda s: bool(s.get('w52h',0) and s.get('price',0)
                     and (s['w52h']-s['price'])/max(s['w52h'],0.01)<=0.05))
    pnf_x      = pct(lambda s: s.get('nl',{}).get('pnf',{}).get('in_x',False))
    renko_bull = pct(lambda s: s.get('nl',{}).get('renko',{}).get('bull',False))
    ha_bull    = pct(lambda s: s.get('ti',{}).get('d',{}).get('ha',{}).get('bull',False))
    rs70       = pct(lambda s: (s.get('rs',0) or 0)>=70)
    rs80       = pct(lambda s: (s.get('rs',0) or 0)>=80)
    ema_fan    = pct(lambda s: s.get('ti',{}).get('d',{}).get('ema',{}).get('fan_bull',False))
    new_highs  = pct(lambda s: s.get('t1',{}).get('w52',{}).get('near',False))
    health=round(above200*0.25+stage2*0.20+rs70*0.20+pnf_x*0.10+
                 renko_bull*0.10+ha_bull*0.08+near52wh*0.07,1)
    # Store on a sentinel stock-0 only; JS reads from global BREADTH
    b=dict(total=N,above200=above200,stage2=stage2,near52wh=near52wh,
           pnf_x=pnf_x,renko_bull=renko_bull,ha_bull=ha_bull,
           rs70=rs70,rs80=rs80,ema_fan=ema_fan,new_highs=new_highs,health=health)
    for s in stocks: s['_breadth_ref']=True   # marker only — breadth stored globally
    stocks[0]['_breadth']=b  # attach to first stock so build_html can find it


# ── OHLC timeframe aggregation helpers ───────────────────────────────────────
def weekly_agg(df):
    """Aggregate daily OHLC to weekly (week ending Friday)."""
    tmp=df.copy(); tmp['Date']=pd.to_datetime(tmp['Date'])
    try:
        agg=tmp.set_index('Date').resample('W-FRI').agg(
            Open=('Open','first'),High=('High','max'),
            Low=('Low','min'),Close=('Close','last'),Volume=('Volume','sum'))
    except Exception:
        agg=tmp.set_index('Date').resample('W').agg(
            Open=('Open','first'),High=('High','max'),
            Low=('Low','min'),Close=('Close','last'),Volume=('Volume','sum'))
    return agg.dropna(subset=['Open']).reset_index()

def monthly_agg(df):
    """Aggregate daily OHLC to monthly."""
    tmp=df.copy(); tmp['Date']=pd.to_datetime(tmp['Date'])
    for rule in ('ME','M','MS'):
        try:
            agg=tmp.set_index('Date').resample(rule).agg(
                Open=('Open','first'),High=('High','max'),
                Low=('Low','min'),Close=('Close','last'),Volume=('Volume','sum'))
            return agg.dropna(subset=['Open']).reset_index()
        except Exception:
            continue
    return df.tail(1).copy()


def compute_gaps(df):
    """Gap scanner — overnight gap vs prior close, fill status, continuation."""
    if len(df)<3: return {}
    C=df["Close"].values.astype(float); O=df["Open"].values.astype(float)
    H=df["High"].values.astype(float);  L=df["Low"].values.astype(float)
    gap_pct=round((O[-1]-C[-2])/max(C[-2],0.01)*100,2)
    gap_up=bool(gap_pct>=0.5); gap_dn=bool(gap_pct<=-0.5)
    filled=False
    if gap_up: filled=bool(L[-1]<=C[-2])
    elif gap_dn: filled=bool(H[-1]>=C[-2])
    cont=bool(gap_up and C[-1]>O[-1]); reject=bool(gap_up and C[-1]<O[-1])
    dn_cont=bool(gap_dn and C[-1]<O[-1])
    g3=len(df)>=4 and all((O[-(i+1)]-C[-(i+2)])/max(C[-(i+2)],0.01)*100>0.3 for i in range(2))
    return dict(pct=gap_pct,
                up1=bool(gap_pct>=1),up2=bool(gap_pct>=2),up3=bool(gap_pct>=3),
                dn1=bool(gap_pct<=-1),dn2=bool(gap_pct<=-2),dn3=bool(gap_pct<=-3),
                up=gap_up,dn=gap_dn,filled=filled,
                cont=cont,reject=reject,dn_cont=dn_cont,consec3=bool(g3))

def _ti_for(data_df):
    """Run all India Pro compute functions on any OHLC DataFrame."""
    return dict(
        candle=compute_candles(data_df),
        ema   =compute_ema_cross(data_df),
        mom   =compute_momentum_score(data_df),
        seq   =compute_sequential(data_df),
        volbld=compute_volume_buildup(data_df),
        consol=compute_52w_consol(data_df),
        ma    =compute_ma_alignment(data_df),
        # New additions
        ha    =compute_heikin_ashi(data_df),
        hma   =compute_hma(data_df),
        kelt  =compute_keltner(data_df),
        mfi   =compute_mfi(data_df),
        cci   =compute_cci(data_df),
        linreg=compute_linreg(data_df),
    )


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
def precompute(fp, idx_map, nifty_df=None):
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
    # Tier-1 high-edge strategies
    t1 = dict(
        gc  = compute_golden_cross(df),
        rsi = compute_rsi_div(df),
        bb  = compute_bb_squeeze(df),
        w52 = compute_w52_breakout(df),
        nr  = compute_nr7(df),
        stt = compute_supertrend(df),   # SuperTrend in Tier-1
    )
    # Tier-2 strategies
    t2 = dict(
        ch   = compute_cup_handle(df),
        macd = compute_macd_div(df),
        zs   = compute_zscore(df),
        rsn  = compute_rs_nifty(df, nifty_df),
    )
    # Tier-3 strategies
    t3 = dict(
        fib  = compute_fib_retracement(df),
        ce   = compute_chandelier(df),
        psar = compute_psar(df),
        adx  = compute_adx(df),
        stoch= compute_stochastic(df),
        wr   = compute_williams_r(df),
    )
    # 🏅 India Pro strategies — daily only (weekly/monthly computed on demand)
    ti = _ti_for(df)
    # 📐 Noiseless (Definedge: P&F, Renko, 3-Line Break, RRG, EW, Kagi)
    nl = dict(
        pnf   = compute_pnf(df),
        renko = compute_renko(df),
        lb3   = compute_3lb(df),
        rrg   = compute_rrg(df, nifty_df),
        ew    = compute_ew_simple(df),
        kagi  = compute_kagi(df),
    )
    # ── Pre-compute shared weekly/monthly DFs (used by patt.dbl, sr.season, sr.mtf, swing) ──
    _wdf=None; _mdf=None
    try:
        import pandas as _pdx
        _dft=df.copy(); _dft['Date']=_pdx.to_datetime(_dft['Date']); _dft=_dft.set_index('Date')
        _wdf=_dft.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
        _mdf=_dft.resample('ME').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    except: pass
    # 🎨 Patterns (Harmonic + Chart Patterns)
    patt = dict(
        harm = compute_harmonics(df),
        cp   = compute_chart_patterns(df),
        dbl  = compute_double_patterns(df,_wdf=_wdf,_mdf=_mdf),
    )
    # 🧪 Experimental (VSA)
    xp = dict(vsa=compute_vsa(df))
    # 🔢 Smart Rank raw data
    sr = dict(
        season=compute_seasonality(df),
        mtf   =compute_mtf_matrix(df),
        ath   =compute_ath(df),
        tc    =compute_triple_compress(df),
        raw   =compute_raw_returns(df),
    )
    # 🎯 Swing Setup — EMA Squeeze + MACD 5/35 (D/W/M)
    swing = dict(
        ema_sq = compute_ema_squeeze(df,_wdf=_wdf,_mdf=_mdf),
        macd535= compute_macd_5_35(df,_wdf=_wdf,_mdf=_mdf),
    )
    # 🏹 Breakout Pro
    bp = dict(
        vcp=compute_vcp(df),
        pp =compute_pocket_pivot(df),
        twt=compute_3week_tight(df),
        rsh=compute_rs_new_high(df,nifty_df),
        ads=compute_ad_score(df),
        hvb=compute_hvb(df),
        vdu=compute_vdu(df),
        ems=compute_ema_stack(df),
        iw =compute_inside_week(df),
        tlb=compute_trendline_bo(df),
    )
    # ③ Gap Scanner
    gap = compute_gaps(df)
    # ⑧ Sparkline — last 60 closes normalised to first=100
    _cl=df['Close'].values[-60:].astype(float)
    _base=float(_cl[0]) if len(_cl) and _cl[0] else 1.0
    spark=[round(c/_base*100,1) for c in _cl]
    adv = dict(
        darvas   = compute_darvas(df),
        vcp      = compute_vcp(df),
        wyckoff  = compute_wyckoff(df),
        turtle   = compute_turtle(df),
        ichi     = compute_ichimoku(df),
        td       = compute_td(df),
        st       = compute_supertrend(df),
        elder    = compute_elder(df),
        dc       = compute_donchian(df),
        gmma     = compute_gmma(df),
    )
    pb = compute_pivot_bounce(df)
    return dict(sym=sym, idx=idx_map.get(sym,0),
                price=r2(float(df.iloc[-1]["Close"])),
                date=str(df.iloc[-1]["Date"].date()),
                d=ds,w=ws,m=ms,q=qs,y=ys,ytd=yts,
                mh=mh,wh=wh,qh=qh,
                smc=smc,vol=vol,mi=mi,t1=t1,t2=t2,t3=t3,ti=ti,nl=nl,patt=patt,xp=xp,adv=adv,gap=gap,spark=spark,bp=bp,sr=sr,pb=pb,swing=swing,**st,rs=0)


def assign_smart_ranks(stocks):
    """Normalize multi-period momentum across the universe (0-100 score)."""
    if not stocks: return
    import math as _math
    fields=['r1m','r3m','r6m','r12m']
    # Collect raw returns
    vals={f:[s.get('sr',{}).get('raw',{}).get(f,0) for s in stocks] for f in fields}
    n=len(stocks)
    def _pct(arr,v):
        return round(sum(1 for x in arr if x<v)/n*100) if n else 50
    for s in stocks:
        raw=s.get('sr',{}).get('raw',{})
        # Multi-period momentum score: percentile rank in each period, weighted avg
        p1=_pct(vals['r1m'],raw.get('r1m',0))
        p3=_pct(vals['r3m'],raw.get('r3m',0))
        p6=_pct(vals['r6m'],raw.get('r6m',0))
        p12=_pct(vals['r12m'],raw.get('r12m',0))
        ms=round(p1*0.15+p3*0.20+p6*0.25+p12*0.30+s.get('rs',0)*0.10)
        if 'sr' not in s: s['sr']={}
        s['sr']['ms']=ms          # Momentum Score 0-100
        s['sr']['r1m']=raw.get('r1m',0)
        s['sr']['r3m']=raw.get('r3m',0)
        s['sr']['r6m']=raw.get('r6m',0)
        s['sr']['r12m']=raw.get('r12m',0)

def assign_rs(stocks):
    rets=sorted(s["ret12m"] for s in stocks); n=len(rets)
    for s in stocks:
        s["rs"] = round(sum(1 for x in rets if x<s["ret12m"])/n*99) if n else 0
        if s.get("mi"):
            s["mi"]["mt"][7] = int(s["rs"]>=70)
            s["mi"]["mts"]   = sum(s["mi"]["mt"])

def build_dataset(data_dir, index_files, nifty_path=None, fno_path=None, sector_path=None):
    idx_map = load_index_map(index_files)
    fno_set = load_set(fno_path) if fno_path else set()
    sector_map = {}
    if sector_path:
        try:
            import csv as _csv
            with open(sector_path,'r',encoding='utf-8') as f:
                for row in _csv.DictReader(f):
                    sym=(row.get('Symbol') or row.get('symbol') or '').strip().upper()
                    sec=(row.get('Sector') or row.get('Industry') or 'Other').strip()
                    if sym: sector_map[sym]=sec
            print(f"  Sector map: {len(sector_map)} stocks")
        except Exception as e: print(f"  Sector skipped: {e}")
    files   = sorted(glob.glob(os.path.join(data_dir,"*.csv")))
    # Load Nifty index data (optional — for RS vs Nifty)
    nifty_df = None
    if nifty_path and os.path.exists(nifty_path):
        nifty_df = load_csv(nifty_path)
        if nifty_df is not None:
            print(f"  Nifty data loaded: {len(nifty_df)} rows from {nifty_path}")
        else:
            print(f"  Warning: could not load Nifty data from {nifty_path}")
    stocks  = []
    for i,fp in enumerate(files,1):
        if i%100==0 or i==len(files): print(f"  [{i}/{len(files)}]…",end="\r")
        rec = precompute(fp, idx_map, nifty_df)
        if rec:
            rec['fno']=rec['sym'] in fno_set
            rec['sector']=sector_map.get(rec['sym'],'Other')
            stocks.append(rec)
    print(f"\n  Done — {len(stocks)} stocks"); assign_rs(stocks); assign_smart_ranks(stocks); assign_breadth(stocks); return stocks

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
  padding:0 22px;display:flex;align-items:flex-end;gap:2px;position:sticky;top:0;z-index:300;
  flex-wrap:wrap;flex-shrink:0}
@media(max-width:1100px){.tabnav{padding:0 10px}}
.logo{font-family:var(--mono);font-size:10px;letter-spacing:3px;color:var(--acc);
  text-transform:uppercase;white-space:nowrap;padding:14px 14px 14px 0;margin-right:10px;
  border-right:1px solid var(--brd)}
.tab-btn{flex-shrink:0;background:transparent;border:none;border-bottom:2px solid transparent;
  color:var(--mu);font-family:var(--sans);font-size:12px;font-weight:500;
  padding:12px 18px 10px;cursor:pointer;transition:all .2s;white-space:nowrap}
.tab-btn:hover{color:var(--txt)}
.tab-btn.active{color:var(--acc);border-bottom-color:var(--acc)}
.tab-btn.t-smc.active{color:var(--a3);border-bottom-color:var(--a3)}
.tab-btn.t-vol.active{color:var(--a2);border-bottom-color:var(--a2)}
.tab-btn.t-adv.active{color:#ff8c42;border-bottom-color:#ff8c42}
.tab-btn.t-t1.active{color:#c47aff;border-bottom-color:#c47aff}
.tab-btn.t-t2.active{color:#00e5a0;border-bottom-color:#00e5a0}
.tab-btn.t-t3.active{color:#f5c518;border-bottom-color:#f5c518}
.tab-btn.t-ti.active{color:#ff9500;border-bottom-color:#ff9500}
.tab-btn.t-nl.active{color:#e879f9;border-bottom-color:#e879f9}
.tab-btn.t-xp.active{color:#f59e0b;border-bottom-color:#f59e0b}
.tab-btn.t-patt.active{color:#06b6d4;border-bottom-color:#06b6d4}
.tab-btn.t-br.active{color:#84cc16;border-bottom-color:#84cc16}

/* GLOBAL FILTER BAR */
.global-bar{display:flex;align-items:center;gap:10px;padding:6px 22px;
  background:rgba(0,229,160,.04);border-bottom:1px solid rgba(0,229,160,.12);
  flex-wrap:wrap;overflow-x:auto;-webkit-overflow-scrolling:touch}
@media(max-width:680px){
  .global-bar{flex-wrap:nowrap;padding:6px 12px}
  .gb-sep{display:none}
  .gb-hint{display:none}
  .gb-label{display:none}
}
.gb-label{font-family:var(--mono);font-size:9px;letter-spacing:2px;
  text-transform:uppercase;color:var(--acc);font-weight:700}
.gb-sep{width:1px;height:18px;background:var(--brd)}
.gb-lbl{font-size:10px;color:var(--mu);font-family:var(--mono)}
.gb-sel{font-family:var(--mono);font-size:10px;padding:3px 8px;
  background:var(--surf);border:1px solid var(--brd);color:var(--txt);
  border-radius:4px;cursor:pointer}
.gb-sel:hover{border-color:var(--acc)}
.gb-hint{font-size:9px;color:var(--mu);font-family:var(--mono);
  letter-spacing:.5px;margin-left:auto}

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
.ts{overflow-x:auto;overflow-y:auto;min-height:200px;max-height:60vh;border:1px solid var(--brd);
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
/* HOME */
.tab-btn.t-home.active{color:#fff;border-bottom-color:#fff}
.tab-btn.t-help.active{color:#80a0c0;border-bottom-color:#80a0c0}
.home-wrap{padding:16px 22px 22px;overflow-y:auto;max-height:58vh;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
.home-title{font-family:var(--mono);font-size:11px;letter-spacing:3px;text-transform:uppercase;color:var(--mu);margin:18px 0 8px;padding-bottom:4px;border-bottom:1px solid var(--brd)}
.home-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:6px}
@media(max-width:900px){.home-grid{grid-template-columns:repeat(2,1fr)}}
.hcard{background:var(--surf);border:1px solid var(--brd);border-radius:8px;padding:12px 13px 10px;cursor:pointer;transition:all .18s;display:flex;flex-direction:column;gap:5px;position:relative}
.hcard:hover{border-color:var(--acc);transform:translateY(-2px);box-shadow:0 4px 18px rgba(0,229,160,.1)}
.hcard-top{display:flex;align-items:center;justify-content:space-between}
.hcard-emoji{font-size:18px}
.hcard-badge{font-size:8px;letter-spacing:1.5px;text-transform:uppercase;padding:2px 6px;border-radius:3px;font-family:var(--mono)}
.badge-t1{background:rgba(196,122,255,.12);color:var(--a3)}
.badge-t2{background:rgba(0,229,160,.1);color:var(--acc)}
.badge-t3{background:rgba(245,197,24,.1);color:var(--gold)}
.badge-mi{background:rgba(59,158,255,.1);color:var(--a2)}
.badge-adv{background:rgba(255,140,66,.1);color:var(--warn)}
.badge-piv{background:rgba(74,92,120,.15);color:var(--mu)}
.badge-ti{background:rgba(255,149,0,.1);color:#ff9500}
.hcard-name{font-family:var(--mono);font-size:12px;font-weight:700;color:#dde5f5}
.hcard-desc{font-size:10.5px;color:var(--mu);line-height:1.5;flex:1}
.hcard-stars{font-size:10px;color:var(--gold);letter-spacing:1px}
.hcard-btn{margin-top:4px;background:rgba(0,229,160,.1);border:1px solid rgba(0,229,160,.2);color:var(--acc);font-family:var(--mono);font-size:9px;letter-spacing:2px;text-transform:uppercase;padding:5px 10px;border-radius:4px;text-align:center;transition:all .15s}
.hcard:hover .hcard-btn{background:var(--acc);color:#000}
/* HELP */
.help-wrap{padding:10px 22px 22px;overflow-y:auto;max-height:58vh;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
.help-wrap details{border:1px solid var(--brd);border-radius:6px;margin-bottom:6px;overflow:hidden}
.help-wrap details summary{padding:10px 14px;cursor:pointer;font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:1px;background:var(--surf);list-style:none;display:flex;align-items:center;gap:8px;user-select:none}
.help-wrap details summary::-webkit-details-marker{display:none}
.help-wrap details[open] summary{border-bottom:1px solid var(--brd);color:var(--acc)}
.help-wrap details summary::before{content:'▶';font-size:8px;color:var(--mu);transition:transform .2s}
.help-wrap details[open] summary::before{transform:rotate(90deg);color:var(--acc)}
.help-body{padding:12px 16px;display:grid;grid-template-columns:1fr 1fr;gap:6px}
@media(max-width:800px){.help-body{grid-template-columns:1fr}}
.help-item{background:rgba(255,255,255,.02);border:1px solid var(--brd);border-radius:5px;padding:10px 12px}
.help-item-name{font-family:var(--mono);font-size:11px;font-weight:700;color:var(--txt);margin-bottom:4px}
.help-item-desc{font-size:10.5px;color:var(--mu);line-height:1.6}
.help-item-formula{font-family:var(--mono);font-size:9.5px;color:var(--a2);margin-top:4px;padding:3px 6px;background:rgba(59,158,255,.05);border-radius:3px}
/* INFO PANEL */
.fbar{cursor:pointer;position:relative}
.fbar::after{content:'ⓘ';position:absolute;right:10px;top:50%;transform:translateY(-50%);font-size:14px;color:var(--a2);opacity:.5}
.fbar:hover::after{opacity:1}
.info-panel{display:none;margin:0 22px 6px;padding:10px 14px;background:rgba(59,158,255,.04);border:1px solid rgba(59,158,255,.15);border-radius:0 0 6px 6px;font-size:11px;color:var(--txt);line-height:1.8;font-family:var(--sans)}
.info-panel.open{display:block}

/* AI BUTTONS & REF LINKS */
.help-ai{display:flex;align-items:center;gap:6px;margin-top:8px;flex-wrap:wrap}
.ai-btn{font-size:9px;font-family:var(--mono);letter-spacing:1px;padding:3px 9px;border-radius:4px;text-decoration:none;font-weight:600;border:1px solid;transition:all .15s;white-space:nowrap}
.ai-claude{color:#c47aff;border-color:rgba(196,122,255,.4);background:rgba(196,122,255,.06)}
.ai-claude:hover{background:rgba(196,122,255,.18)}
.ai-chatgpt{color:#10a37f;border-color:rgba(16,163,127,.4);background:rgba(16,163,127,.06)}
.ai-chatgpt:hover{background:rgba(16,163,127,.18)}
.ai-copilot{color:#3b9eff;border-color:rgba(59,158,255,.4);background:rgba(59,158,255,.06)}
.ai-copilot:hover{background:rgba(59,158,255,.18)}
.ai-gemini{color:#f5c518;border-color:rgba(245,197,24,.4);background:rgba(245,197,24,.06)}
.ai-gemini:hover{background:rgba(245,197,24,.18)}
/* TOAST */
.toast{position:fixed;bottom:24px;left:50%;transform:translateX(-50%);background:var(--surf);border:1px solid var(--acc);color:var(--acc);padding:9px 22px;border-radius:20px;font-family:var(--mono);font-size:10px;letter-spacing:.5px;z-index:9999;opacity:0;transition:opacity .3s;pointer-events:none;white-space:nowrap;box-shadow:0 4px 20px rgba(0,229,160,.15)}
.toast.show{opacity:1}
/* HELP MODAL */
.hm-overlay{position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:998;display:none}
.hm-overlay.open{display:block}
.hm-box{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);width:min(780px,93vw);max-height:84vh;overflow-y:auto;background:var(--surf);border:1px solid var(--brd);border-radius:10px;z-index:999;display:none;scrollbar-color:var(--a2) var(--brd);scrollbar-width:thin}
.hm-box.open{display:block}
.hm-hdr{position:sticky;top:0;background:var(--hdr);border-bottom:1px solid var(--brd);padding:12px 18px;display:flex;justify-content:space-between;align-items:center;z-index:1}
.hm-title{font-family:var(--mono);font-size:12px;font-weight:700;color:var(--acc)}
.hm-close{background:none;border:1px solid rgba(255,64,96,.3);color:var(--red);font-size:14px;padding:2px 9px;border-radius:4px;cursor:pointer}
.hm-close:hover{background:rgba(255,64,96,.12)}
.hm-body{padding:18px 20px}
/* INFO BUTTON */
.info-btn{background:rgba(59,158,255,.06);border:1px solid rgba(59,158,255,.25);color:var(--a2);font-size:11px;font-weight:700;width:24px;height:24px;border-radius:50%;cursor:pointer;flex-shrink:0;transition:all .15s;display:flex;align-items:center;justify-content:center}
.info-btn:hover{background:rgba(59,158,255,.2);border-color:var(--a2)}
.help-refs{display:flex;gap:6px;flex-wrap:wrap;margin-top:6px}
.ref-link{font-size:9px;font-family:var(--mono);color:var(--a2);text-decoration:none;padding:2px 7px;border:1px solid rgba(59,158,255,.2);border-radius:3px;background:rgba(59,158,255,.04)}
.ref-link:hover{background:rgba(59,158,255,.14);color:#fff}
@keyframes ldpulse{from{opacity:.5}to{opacity:1}}
.sd-ico{display:inline-flex;align-items:center;justify-content:center;
  width:13px;height:13px;border-radius:50%;font-size:8px;font-weight:700;font-style:italic;
  background:rgba(59,158,255,.12);color:var(--a2);border:1px solid rgba(59,158,255,.25);
  cursor:pointer;margin-right:5px;flex-shrink:0;transition:background .15s;
  user-select:none;vertical-align:middle;line-height:1}
.sd-ico:hover{background:rgba(59,158,255,.35);border-color:var(--a2)}

/* ④ Stock detail modal */
#sd-overlay{position:fixed;inset:0;background:rgba(0,0,0,.8);z-index:9990;display:none}
#sd-box{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:var(--bg);border:1px solid var(--brd);border-radius:12px;z-index:9991;width:min(94vw,840px);max-height:86vh;display:none;flex-direction:column;box-shadow:0 20px 60px rgba(0,0,0,.7)}
#sd-hdr{display:flex;align-items:center;justify-content:space-between;padding:14px 18px;border-bottom:1px solid var(--brd);flex-shrink:0}
#sd-body{overflow-y:auto;padding:16px 18px;flex:1}
/* gap + combo tabs */
.tab-btn.t-gap.active{color:#ff8c42;border-bottom-color:#ff8c42}
.tab-btn.t-combo.active{color:#c47aff;border-bottom-color:#c47aff}
.tab-btn.t-bp.active{color:#ff6b35;border-bottom-color:#ff6b35}
.tab-btn.t-sr.active{color:#00d4ff;border-bottom-color:#00d4ff}
.tab-btn.t-lookup.active{color:#a78bfa;border-bottom-color:#a78bfa}
.tab-btn.t-iview.active{color:#fbbf24;border-bottom-color:#fbbf24}
.tab-btn.t-toppicks.active{color:#34d399;border-bottom-color:#34d399}
.tab-btn.t-swing.active{color:#fb923c;border-bottom-color:#fb923c}
/* ⑥ sector grouping */
.gb-chk{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--mu);cursor:pointer;user-select:none;white-space:nowrap}
.gb-chk input{accent-color:var(--acc)}
.sec-hdr{background:rgba(59,158,255,.07);color:var(--a2);font-size:10px;font-weight:700;padding:5px 12px;letter-spacing:1px;text-transform:uppercase}
/* ⑩ mobile */
@media(max-width:820px){
  .tab-btn{font-size:10px;padding:7px 6px;white-space:nowrap}
  .home-grid{grid-template-columns:repeat(2,1fr)}
  .hcard-desc{display:none}
  #sd-box{width:96vw;max-height:90vh}
}

</style>
</head>
<body>
<!-- LOADING OVERLAY — shown while large JSON is being parsed by browser -->
<div id="_ld" style="position:fixed;inset:0;background:#080b10;z-index:99999;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:20px;font-family:monospace">
  <div style="color:#00e5a0;font-size:22px;font-weight:700;letter-spacing:2px">◈ NSE Strategy Scanner</div>
  <div style="color:#3b9eff;font-size:13px" id="_ld_msg">Loading market data…</div>
  <div style="color:#c8d4e8;font-size:11px" id="_ld_detail">Parsing __STOCK_COUNT__ stocks — please wait</div>
  <div style="width:240px;height:4px;background:#182034;border-radius:3px;overflow:hidden">
    <div id="_ld_bar" style="width:20%;height:100%;background:linear-gradient(90deg,#00e5a0,#3b9eff);border-radius:3px;animation:ldpulse 1s ease-in-out infinite alternate"></div>
  </div>
  <div style="color:#4a607a;font-size:10px;margin-top:8px">Click any tab after this screen disappears</div>
</div>


<!-- ═══ TAB NAV ═══════════════════════════════════════════════════════════ -->
<div class="tabnav">
  <div class="logo">◈ NSE Scanner</div>
  <button class="tab-btn t-home active" data-tab="home" onclick="switchTab('home')">🏠 Home</button>
  <button class="tab-btn t-piv" data-tab="piv" onclick="switchTab('piv')">📊 Pivot Points</button>
  <button class="tab-btn t-smc" data-tab="smc" onclick="switchTab('smc')">🎯 Price Action / SMC</button>
  <button class="tab-btn t-vol" data-tab="vol" onclick="switchTab('vol')">📈 Volume / Institutional</button>
  <button class="tab-btn t-mi"  data-tab="mi"  onclick="switchTab('mi')">⚡ Multi-Indicator</button>
  <button class="tab-btn t-adv" data-tab="adv" onclick="switchTab('adv')">🔬 Advanced</button>
  <button class="tab-btn t-t1"  data-tab="t1"  onclick="switchTab('t1')">⭐ Tier-1</button>
  <button class="tab-btn t-t2"  data-tab="t2"  onclick="switchTab('t2')">🏆 Tier-2</button>
  <button class="tab-btn t-t3"  data-tab="t3"  onclick="switchTab('t3')">🎖 Tier-3</button>
  <button class="tab-btn t-ti"  data-tab="ti"  onclick="switchTab('ti')">🏅 India Pro</button>
  <button class="tab-btn t-nl"  data-tab="nl"  onclick="switchTab('nl')">📐 Noiseless</button>
  <button class="tab-btn t-xp"  data-tab="xp"  onclick="switchTab('xp')">🧪 Experimental</button>
  <button class="tab-btn t-patt" data-tab="patt" onclick="switchTab('patt')">🎨 Patterns</button>
  <button class="tab-btn t-br"  data-tab="br"  onclick="switchTab('br')">📡 Breadth</button>
  <button class="tab-btn t-gap" data-tab="gap" onclick="switchTab('gap')">⚡ Gaps</button>
  <button class="tab-btn t-combo" data-tab="combo" onclick="switchTab('combo')">⚗️ Combiner</button>
  <button class="tab-btn t-bp" data-tab="bp" onclick="switchTab('bp')">🏹 Breakout Pro</button>
  <button class="tab-btn t-sr" data-tab="sr" onclick="switchTab('sr')">🔢 Smart Rank</button>
  <button class="tab-btn t-lookup" data-tab="lookup" onclick="switchTab('lookup')">🔍 Stock Lookup</button>
  <button class="tab-btn t-iview" data-tab="iview" onclick="switchTab('iview')">📋 Index View</button>
  <button class="tab-btn t-toppicks" data-tab="toppicks" onclick="switchTab('toppicks')">🏆 Top Picks</button>
  <button class="tab-btn t-swing" data-tab="swing" onclick="switchTab('swing')">🎯 Swing Setup</button>
  <button class="tab-btn t-help" data-tab="help" onclick="switchTab('help')">❓ Help</button>
</div>
<!-- ═══ GLOBAL FILTERS ════════════════════════════════════════════════════ -->
<div class="global-bar" id="global-bar">
  <span class="gb-label">🌐 Global</span>
  <div class="gb-sep"></div>
  <label class="gb-lbl">Index</label>
  <select id="global-idx" class="gb-sel">
    <option value="0" selected>All Stocks</option>
    <option value="fno">Nifty F&amp;O</option>
    <option value="50">Nifty 50</option>
    <option value="100">Nifty 100</option>
    <option value="200">Nifty 200</option>
    <option value="500">Nifty 500</option>
    <option value="750">Nifty 750</option>
  </select>
  <div class="gb-sep"></div>
  <span class="gb-hint">Applies to all scanners ↓</span>
  <div class="gb-sep"></div>
  <label class="gb-chk" title="Group results by sector"><input type="checkbox" id="global-sector-grp" onchange="render()"> Sector groups</label>
</div>

<!-- ═══ PIVOT CONTROLS ════════════════════════════════════════════════════ -->
<div id="ctrl-piv" class="ctrl">
  <div class="cg"><label>Pivot Type</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="sp-type" onchange="onTypeChange()">
      <option value="camarilla">Camarilla</option>
      <option value="traditional">Traditional</option>
      <option value="classic">Classic</option>
      <option value="fibonacci" selected>Fibonacci</option>
      <option value="woodie">Woodie</option>
      <option value="dm">DeMark (DM)</option>
      <option value="floor">Floor</option>
      <option value="cpr">CPR — Central Pivot Range</option>
    </select><button class="info-btn" onclick="showHelp('piv_type',document.getElementById('sp-type').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>Level</label><select id="sp-lvl" style="min-width:190px"></select></div>
  <div class="cg"><label>Bounce Filter</label>
    <select id="sp-bounce" style="min-width:200px">
      <option value="">No bounce filter</option>
      <option value="near_S1">Near S1 (within 0.7%)</option>
      <option value="near_S2">Near S2</option>
      <option value="near_P">Near Pivot P</option>
      <option value="near_R1">Near R1</option>
      <option value="near_R2">Near R2</option>
      <option value="above_P">Above Pivot — bullish</option>
      <option value="below_S1">Below S1 — bearish</option>
      <option value="at_sup">At Any Support (S1/S2)</option>
      <option value="at_res">At Any Resistance (R1/R2)</option>
    </select>
  </div>
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
  <button class="btn btn-out" style="color:var(--a2);border-color:rgba(59,158,255,.4)" onclick="shareLink()" title="Copy shareable link with current settings">🔗 SHARE</button>
</div>

<!-- ═══ SMC CONTROLS ══════════════════════════════════════════════════════ -->
<div id="ctrl-smc" class="ctrl" style="display:none">
  <div class="cg"><label>Signal Type</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="smc-sig" style="min-width:220px">
      <option value="all">All Signals</option>
      <option value="bull_ob" selected>Bullish Order Block — near support zone</option>
      <option value="bear_ob">Bearish Order Block — near resistance zone</option>
      <option value="bull_fvg">Bullish FVG — price near unfilled gap below</option>
      <option value="bear_fvg">Bearish FVG — price near unfilled gap above</option>
      <option value="bos_bull">BOS Bullish — broke above swing high</option>
      <option value="bos_bear">BOS Bearish — broke below swing low</option>
      <option value="choch_bull">CHoCH Bullish — trend reversal signal up</option>
      <option value="choch_bear">CHoCH Bearish — trend reversal signal down</option>
    </select><button class="info-btn" onclick="showHelp('smc',document.getElementById('smc-sig').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>Proximity ±%</label>
    <select id="smc-prox">
      <option value="1">±1%</option><option value="2" selected>±2%</option>
      <option value="3">±3%</option><option value="5">±5%</option>
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
    <div style="display:flex;gap:6px;align-items:center"><select id="vol-sig" style="min-width:240px">
      <option value="near_poc" selected>Near POC — point of control</option>
      <option value="above_vah">Above VAH — value area breakout</option>
      <option value="below_val">Below VAL — value area breakdown</option>
      <option value="vol_spike">Volume Spike — unusual activity</option>
      <option value="obv_div_bull">OBV Bullish Divergence</option>
      <option value="obv_div_bear">OBV Bearish Divergence</option>
      <option value="ad_up">A/D Line Rising — accumulation</option>
      <option value="all">All Signals</option>
    </select><button class="info-btn" onclick="showHelp('vol',document.getElementById('vol-sig').value)" title="Show strategy info">ℹ</button></div>
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
    <div style="display:flex;gap:6px;align-items:center"><select id="mi-strat" style="min-width:220px">
      <option value="both" selected>Both — Minervini + Weinstein Stage 2</option>
      <option value="minervini">Minervini Trend Template only</option>
      <option value="weinstein">Weinstein Stage 2 only</option>
      <option value="any">Any strategy match</option>
    </select><button class="info-btn" onclick="showHelp('mi',document.getElementById('mi-strat').value)" title="Show strategy info">ℹ</button></div>
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

<!-- ═══ ADVANCED STRATEGIES CONTROLS ══════════════════════════════════════ -->
<div id="ctrl-adv" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="adv-strat" style="min-width:260px" onchange="updateAdvInfo()">
      <option value="darvas">Darvas Box — 52W-high + tight consolidation breakout</option>
      <option value="vcp">VCP — Volatility Contraction Pattern (Minervini)</option>
      <option value="wyckoff_acc">Wyckoff — Accumulation phase</option>
      <option value="wyckoff_markup">Wyckoff — Markup phase (advancing)</option>
      <option value="wyckoff_spring">Wyckoff — Spring / Upthrust signal</option>
      <option value="turtle20">Turtle Trading — 20-day Donchian breakout</option>
      <option value="turtle55">Turtle Trading — 55-day Donchian breakout</option>
      <option value="ichi_above">Ichimoku — Price above cloud (bull trend)</option>
      <option value="ichi_tk_bull">Ichimoku — Bullish TK cross</option>
      <option value="ichi_kbo">Ichimoku — Kumo breakout (just crossed above cloud)</option>
      <option value="td_buy9">TD Sequential — Buy Setup 9 (exhaustion low)</option>
      <option value="td_sell9">TD Sequential — Sell Setup 9 (exhaustion high)</option>
      <option value="st_up">Supertrend — Direction UP (bullish)</option>
      <option value="st_flip_up">Supertrend — Just flipped UP (fresh signal)</option>
      <option value="elder_buy">Elder Triple Screen — Buy Setup</option>
      <option value="elder_sell">Elder Triple Screen — Sell Setup</option>
    </select><button class="info-btn" onclick="showHelp('adv',document.getElementById('adv-strat').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="adv-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="adv-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#ff8c42;color:#000" onclick="scanAdv()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>


<!-- ═══ TIER-1 STRATEGIES CONTROLS ═══════════════════════════════════════ -->
<div id="ctrl-t1" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="t1-strat" style="min-width:280px" onchange="updateT1Info()">
      <optgroup label="── Golden / Death Cross ──">
        <option value="gc_fresh_golden">Golden Cross — fresh (last 10 days)</option>
        <option value="gc_fresh_death">Death Cross — fresh (last 10 days)</option>
        <option value="gc_bull_zone">Bull Zone — 50 SMA above 200 SMA</option>
        <option value="gc_near_golden">Near Golden Cross — gap closing (within 1%)</option>
      </optgroup>
      <optgroup label="── 📐 Donchian Channel (Turtle System) ──">
        <option value="dc20_bull">Donchian 20-Day Breakout — close above upper band</option>
        <option value="dc20_bear">Donchian 20-Day Breakdown — close below lower band</option>
        <option value="dc55_bull">Donchian 55-Day Bull — Turtle System entry</option>
        <option value="dc55_bear">Donchian 55-Day Bear — Turtle short</option>
        <option value="dc_squeeze">Donchian Squeeze — bandwidth at 6-month low</option>
        <option value="dc_mid20">Donchian Mid-Band Pullback</option>
      </optgroup>
      <optgroup label="── 🌈 GMMA — Guppy Multiple Moving Average ──">
        <option value="gmma_cross_bull">GMMA Cross Bull — short-term just crossed above long-term today</option>
        <option value="gmma_cross_bear">GMMA Cross Bear — fresh bear crossover</option>
        <option value="gmma_bull">GMMA Bull — short-term group above long-term</option>
        <option value="gmma_all_bull">GMMA All-Bull — all 6 short-term above all 6 long-term</option>
        <option value="gmma_bear">GMMA Bear</option>
        <option value="gmma_compress">GMMA Compression — both groups converging</option>
      </optgroup>
      <optgroup label="── RSI Divergence ──">
        <option value="rsi_bull_div">RSI Bullish Divergence — hidden accumulation</option>
        <option value="rsi_bear_div">RSI Bearish Divergence — hidden distribution</option>
        <option value="rsi_oversold">RSI Oversold ≤ 30 — extreme selling exhaustion</option>
        <option value="rsi_overbought">RSI Overbought ≥ 70 — extreme buying exhaustion</option>
      </optgroup>
      <optgroup label="── 📊 RSI Trend Filter ──">
        <option value="rsi_above50">RSI Above 50 — bull momentum zone</option>
        <option value="rsi_below50">RSI Below 50 — bear momentum zone</option>
        <option value="rsi_cross50_bull">RSI Cross Above 50 — fresh momentum shift to bull</option>
        <option value="rsi_cross50_bear">RSI Cross Below 50 — fresh shift to bear</option>
        <option value="rsi_bull_zone">RSI Bull Zone (40–80) — healthy uptrend</option>
      </optgroup>
      <optgroup label="── Bollinger Band Squeeze ──">
        <option value="bb_squeeze_bull">BB Squeeze + Bullish Momentum — explosion up</option>
        <option value="bb_squeeze_bear">BB Squeeze + Bearish Momentum — explosion down</option>
        <option value="bb_squeeze_any">BB Squeeze (any direction) — watch for breakout</option>
      </optgroup>
      <optgroup label="── 52-Week High Breakout ──">
        <option value="w52_breakout">52W High Breakout + Volume surge</option>
        <option value="w52_close_above">52W High Close Breakout (close above 52W high)</option>
        <option value="w52_near">Near 52W High (within 3%) — pre-breakout watch</option>
        <option value="w52_consol_near">Near 52W High + Consolidating — coiling for breakout</option>
      </optgroup>
      <optgroup label="── NR7 / Inside Bar ──">
        <option value="nr7_bull">NR7 — Bull Bias (narrowest range of 7 bars, uptrend)</option>
        <option value="nr7_bear">NR7 — Bear Bias (narrowest range of 7 bars, downtrend)</option>
        <option value="nr4">NR4 — Narrowest of 4 bars (shorter compression)</option>
        <option value="inside_bar">Inside Bar — price coiling inside previous bar's range</option>
        <option value="nr7_inside">NR7 + Inside Bar — double compression signal</option>
      </optgroup>
      <optgroup label="── 📈 SuperTrend (Chartink #1 most-used indicator) ──">
        <option value="st_bull">SuperTrend Bull — price above rising ST line (7×3)</option>
        <option value="st_bear">SuperTrend Bear — price below ST line</option>
        <option value="st_flip_bull">SuperTrend Flip Bull — fresh bull signal today</option>
        <option value="st_flip_bear">SuperTrend Flip Bear — fresh bear signal today</option>
        <option value="st_consec5">SuperTrend 5+ Bull bars — sustained momentum</option>
        <option value="st_tight">SuperTrend Tight (&lt;1% dist) — near support</option>
      </optgroup>
    </select><button class="info-btn" onclick="showHelp('t1',document.getElementById('t1-strat').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>Min RSI</label>
    <select id="t1-rsi-min">
      <option value="0" selected>Any</option>
      <option value="40">≥ 40</option><option value="50">≥ 50</option>
    </select>
  </div>
  <div class="cg"><label>Max RSI</label>
    <select id="t1-rsi-max">
      <option value="100" selected>Any</option>
      <option value="70">≤ 70</option><option value="60">≤ 60</option>
    </select>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="t1-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="t1-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="t1-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#c47aff;color:#000" onclick="scanT1()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ TIER-2 STRATEGIES CONTROLS ═══════════════════════════════════════ -->
<div id="ctrl-t2" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="t2-strat" style="min-width:300px" onchange="updateT2Info()">
      <optgroup label="── Cup & Handle (O'Neil) ──">
        <option value="ch_breakout">Cup & Handle — Breakout + Volume</option>
        <option value="ch_near">Cup & Handle — Near breakout (within 3%)</option>
        <option value="ch_handle">Cup & Handle — In handle (watch zone)</option>
      </optgroup>
      <optgroup label="── MACD Divergence ──">
        <option value="macd_bull_div">MACD Bullish Divergence — hidden accumulation</option>
        <option value="macd_bear_div">MACD Bearish Divergence — hidden distribution</option>
        <option value="macd_flip_bull">MACD Histogram — flipped positive (just crossed 0)</option>
        <option value="macd_flip_bear">MACD Histogram — flipped negative (just crossed 0)</option>
        <option value="macd_bull_trend">MACD Bull Trend — MACD above signal line</option>
      </optgroup>
      <optgroup label="── 📊 MACD Histogram Direction ──">
        <option value="macd_hist_rising">Histogram Rising — momentum building (even if negative)</option>
        <option value="macd_hist_pos_inc">Histogram Positive + Rising — strongest bull confirmation</option>
        <option value="macd_hist_neg_inc">Histogram Negative + Rising — early bull divergence zone</option>
        <option value="macd_zero_bull">Zero-Line Cross Up — MACD crossed above 0</option>
        <option value="macd_zero_bear">Zero-Line Cross Down — MACD crossed below 0</option>
      </optgroup>
      <optgroup label="── Mean Reversion Z-Score ──">
        <option value="zs_oversold">Z-Score Oversold — price ≤ −2 std devs from MA20</option>
        <option value="zs_overbought">Z-Score Overbought — price ≥ +2 std devs from MA20</option>
        <option value="zs_extreme_os">Z-Score Extreme OS — price ≤ −3 std devs (rare reversal)</option>
        <option value="zs_extreme_ob">Z-Score Extreme OB — price ≥ +3 std devs (rare reversal)</option>
        <option value="zs_reverting">Z-Score Reverting — was extreme, now pulling back toward MA</option>
      </optgroup>
      <optgroup label="── RS vs Nifty (needs --nifty flag) ──">
        <option value="rsn_outperform">RS Outperforming Nifty — ratio in uptrend</option>
        <option value="rsn_new_hi">RS at New 3-Month High — accelerating leadership</option>
        <option value="rsn_underperform">RS Underperforming Nifty — ratio in downtrend</option>
      </optgroup>
    </select><button class="info-btn" onclick="showHelp('t2',document.getElementById('t2-strat').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="t2-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="t2-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="t2-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:var(--acc)" onclick="scanT2()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ TIER-3 STRATEGIES CONTROLS ═══════════════════════════════════════ -->
<div id="ctrl-t3" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center"><select id="t3-strat" style="min-width:310px" onchange="updateT3Info()">
      <optgroup label="── Fibonacci Retracement ──">
        <option value="fib_near_382">Near 38.2% Fibonacci level (±2%)</option>
        <option value="fib_near_500">Near 50.0% Fibonacci level (±2%)</option>
        <option value="fib_near_618">Near 61.8% Fibonacci level — golden ratio (±2%)</option>
        <option value="fib_near_786">Near 78.6% Fibonacci level (±2%)</option>
        <option value="fib_any">Near any Fibonacci level (±2%)</option>
      </optgroup>
      <optgroup label="── Chandelier Exit ──">
        <option value="ce_bull">Chandelier — Bullish (price above long stop)</option>
        <option value="ce_bear">Chandelier — Bearish (price below short stop)</option>
        <option value="ce_flip_bull">Chandelier — Just flipped Bullish (fresh signal)</option>
        <option value="ce_flip_bear">Chandelier — Just flipped Bearish (fresh signal)</option>
      </optgroup>
      <optgroup label="── Parabolic SAR ──">
        <option value="psar_bull">PSAR — Bullish (price above SAR dots)</option>
        <option value="psar_bear">PSAR — Bearish (price below SAR dots)</option>
        <option value="psar_flip_bull">PSAR — Just flipped Bullish (trend reversal up)</option>
        <option value="psar_flip_bear">PSAR — Just flipped Bearish (trend reversal down)</option>
      </optgroup>
      <optgroup label="── ADX + DI ──">
        <option value="adx_strong_bull">ADX Strong Bull Trend — ADX ≥25 + +DI &gt; -DI</option>
        <option value="adx_strong_bear">ADX Strong Bear Trend — ADX ≥25 + -DI &gt; +DI</option>
        <option value="adx_di_cross_bull">ADX DI Crossover — +DI just crossed above -DI</option>
        <option value="adx_di_cross_bear">ADX DI Crossover — -DI just crossed above +DI</option>
        <option value="adx_extreme">ADX Extreme Trend — ADX ≥ 40 (very strong move)</option>
      </optgroup>
      <optgroup label="── Stochastic %K/%D ──">
        <option value="stoch_bull_cross">Stochastic Bullish Cross — %K crossed above %D (oversold)</option>
        <option value="stoch_bear_cross">Stochastic Bearish Cross — %K crossed below %D (overbought)</option>
        <option value="stoch_oversold">Stochastic Oversold — %K ≤ 20</option>
        <option value="stoch_overbought">Stochastic Overbought — %K ≥ 80</option>
      </optgroup>
      <optgroup label="── Williams %R ──">
        <option value="wr_oversold">Williams %R Oversold — %R ≤ −80 (exhaustion)</option>
        <option value="wr_overbought">Williams %R Overbought — %R ≥ −20 (exhaustion)</option>
        <option value="wr_bull_exit">Williams %R Exiting Oversold — momentum turning up</option>
        <option value="wr_bear_exit">Williams %R Exiting Overbought — momentum turning down</option>
      </optgroup>
    </select><button class="info-btn" onclick="showHelp('t3',document.getElementById('t3-strat').value)" title="Show strategy info">ℹ</button></div>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="t3-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="t3-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="t3-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:var(--gold);color:#000" onclick="scanT3()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ INDIA PRO CONTROLS ═══════════════════════════════════════════════ -->
<div id="ctrl-ti" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center">
    <select id="ti-strat" style="min-width:320px" onchange="updateTIInfo()">
      <optgroup label="── 🕯 Candlestick Patterns (StockEdge / Chartink) ──">
        <option value="candle_hammer">Hammer — bullish reversal at bottom (long lower shadow)</option>
        <option value="candle_inv_hammer">Inverted Hammer — bullish reversal, long upper shadow</option>
        <option value="candle_bull_eng">Bullish Engulfing — current candle fully engulfs prior bear</option>
        <option value="candle_morning_star">Morning Star — 3-bar bullish reversal: bear→star→bull</option>
        <option value="candle_piercing">Piercing Line — opens below prior low, closes above 50%</option>
        <option value="candle_bull_har">Bullish Harami — small bull body inside large bear</option>
        <option value="candle_tws">Three White Soldiers — 3 consecutive bullish closes</option>
        <option value="candle_doji">Doji — indecision candle, body &lt;10% of range</option>
        <option value="candle_shooting_star">Shooting Star — bearish reversal, long upper shadow at top</option>
        <option value="candle_bear_eng">Bearish Engulfing — current bear candle engulfs prior bull</option>
        <option value="candle_dark_cloud">Dark Cloud Cover — opens above prior high, closes below 50%</option>
        <option value="candle_evening_star">Evening Star — 3-bar bearish reversal: bull→star→bear</option>
        <option value="candle_bear_har">Bearish Harami — small bear body inside large bull</option>
        <option value="candle_tbc">Three Black Crows — 3 consecutive bearish closes</option>
        <option value="candle_doji">Doji — indecision candle, body &lt;10% of range</option>
      </optgroup>
      <optgroup label="── 🕯 Inside Bar (Chartink #1 scan) ──">
        <option value="candle_inside_bar">Inside Bar — current bar entirely within prior bar (neutral)</option>
        <option value="candle_inside_bar_bull">Inside Bar + Bullish Close — closed upper half → bullish bias</option>
        <option value="candle_inside_bar_bear">Inside Bar + Bearish Close — closed lower half → bearish bias</option>
        <option value="candle_double_inside">Double Inside Bar — 2 consecutive inside bars (tightest coil)</option>
      </optgroup>
      <optgroup label="── 📊 StockEdge Premium Patterns ──">
        <option value="candle_marubozu_bull">White Marubozu — full bull body ≥90% range, no shadows (StockEdge)</option>
        <option value="candle_marubozu_bear">Black Marubozu — full bear body ≥90% range, no shadows (StockEdge)</option>
        <option value="candle_spinning_top">Spinning Top — small body, long shadows both sides (indecision)</option>
        <option value="candle_tweezer_bottom">Tweezer Bottom — two candles with identical/near-identical lows</option>
        <option value="candle_tweezer_top">Tweezer Top — two candles with identical/near-identical highs</option>
        <option value="candle_outside_bull">Bullish Outside Bar — engulfs prior bar range, closes up</option>
        <option value="candle_outside_bear">Bearish Outside Bar — engulfs prior bar range, closes down</option>
      </optgroup>
      <optgroup label="── 🚀 Gap Scans (Chartink most-used) ──">
        <option value="candle_gap_up">Gap Up &gt;1% — opened above yesterday's close by 1%+</option>
        <option value="candle_gap_up3">Gap Up &gt;3% — strong gap (Chartink style: 3%+)</option>
        <option value="candle_gap_down">Gap Down &lt;-1% — opened below yesterday's close by 1%+</option>
        <option value="candle_gap_down3">Gap Down &lt;-3% — strong gap down (Chartink style: 3%+)</option>
      </optgroup>
      <optgroup label="── 📈 EMA Crossovers (Chartink paid alerts) ──">
        <option value="ema_c921_bull">EMA 9 crossed above EMA 21 — short-term bull signal</option>
        <option value="ema_c921_bear">EMA 9 crossed below EMA 21 — short-term bear signal</option>
        <option value="ema_c2050_bull">EMA 20 crossed above EMA 50 — medium-term bull signal</option>
        <option value="ema_c2050_bear">EMA 20 crossed below EMA 50 — medium-term bear signal</option>
        <option value="ema_fan_bull">EMA Fan Bullish — price &gt; EMA9 &gt; EMA21 &gt; EMA50</option>
        <option value="ema_fan_bear">EMA Fan Bearish — price &lt; EMA9 &lt; EMA21 &lt; EMA50</option>
        <option value="ema_recent_921">EMA 9/21 Bull Cross (last 8 bars) — fresh momentum signal</option>
      </optgroup>
      <optgroup label="── 🚀 Price Momentum Score (Trendlyne style) ──">
        <option value="mom_high">Composite Momentum Score ≥ 20% — strongest movers</option>
        <option value="mom_positive_all">All timeframes positive — 1M, 3M, 6M, 12M all up</option>
        <option value="mom_roc10_bull">Rate of Change 10-day &gt; 5% — fast mover</option>
        <option value="mom_roc21_bull">Rate of Change 21-day &gt; 8% — strong monthly mover</option>
        <option value="mom_neg">Composite Momentum Negative — weakest / avoid longs</option>
      </optgroup>
      <optgroup label="── 📊 Sequential Higher Highs (ScanX / Chartink) ──">
        <option value="seq_hh3">3 Consecutive Higher Highs — uptrend building</option>
        <option value="seq_hh5">5 Consecutive Higher Highs — strong sustained uptrend</option>
        <option value="seq_hl3">3 Consecutive Higher Lows — buyers defending higher ground</option>
        <option value="seq_hc3_vol">3 Higher Closes + Volume Confirm — conviction up-move</option>
        <option value="seq_ll3">3 Consecutive Lower Lows — downtrend building</option>
      </optgroup>
      <optgroup label="── 📦 Volume Buildup / Accumulation (StockEdge) ──">
        <option value="vol_acc3">3+ Days Rising Volume + Price Up — accumulation streak</option>
        <option value="vol_acc5">5+ Days Rising Volume + Price Up — sustained accumulation</option>
        <option value="vol_quiet">Quiet Accumulation — flat price, volume &gt;1.3× avg for 3 days</option>
      </optgroup>
      <optgroup label="── 🎯 52W Consolidation / Coiling (Chartink) ──">
        <option value="consol_10">Tight Coil — within 5% of 52W high, 10-day range &lt;5%</option>
        <option value="consol_15">Extended Base — within 7% of 52W high, volume drying up</option>
      </optgroup>
      <optgroup label="── 🔝 Multi-MA Alignment (StockEdge price scans) ──">
        <option value="ma_max4">Price above all 4 MAs — 20, 50, 100, 200 SMA</option>
        <option value="ma_all_bull">Full MA Stack — price &gt; MA20 &gt; MA50 &gt; MA100 &gt; MA200</option>
        <option value="ma_above3">Price above 20, 50, 100 SMA (less strict)</option>
      </optgroup>
      <optgroup label="── 🕯 Heikin Ashi (Chartink most-popular chart type) ──">
        <option value="ha_bull">HA Bull Candle — current Heikin Ashi candle is green</option>
        <option value="ha_bear">HA Bear Candle — current Heikin Ashi candle is red</option>
        <option value="ha_strong_bull">HA Strong Bull — no lower shadow (pure momentum up)</option>
        <option value="ha_strong_bear">HA Strong Bear — no upper shadow (pure momentum down)</option>
        <option value="ha_doji">HA Doji — small body, indecision on smoothed chart</option>
        <option value="ha_rev_bull">HA Reversal Bullish — first green HA after red run</option>
        <option value="ha_rev_bear">HA Reversal Bearish — first red HA after green run</option>
        <option value="ha_consec5">HA 5+ Consecutive Bull — 5+ green HA candles (strong trend)</option>
      </optgroup>
      <optgroup label="── 📉 Hull MA — anti-lag moving average (ChartAlert / TradePoint) ──">
        <option value="hma_bull">HMA Bull — price above rising Hull MA(20)</option>
        <option value="hma_bear">HMA Bear — price below falling Hull MA(20)</option>
        <option value="hma_flip_bull">HMA Just Flipped Up — HMA slope changed from down to up</option>
        <option value="hma_flip_bear">HMA Just Flipped Down — HMA slope changed from up to down</option>
      </optgroup>
      <optgroup label="── 📏 Keltner Channels (ChartAlert / Chartink premium) ──">
        <option value="kelt_above">Above Upper Keltner — breakout above EMA20+2×ATR band</option>
        <option value="kelt_below">Below Lower Keltner — breakdown below EMA20−2×ATR band</option>
        <option value="kelt_cross_up">Keltner Cross Up — just crossed above upper band</option>
        <option value="kelt_cross_dn">Keltner Cross Down — just crossed below lower band</option>
        <option value="kelt_squeeze">Keltner Squeeze — BB inside Keltner (TTM squeeze active)</option>
      </optgroup>
      <optgroup label="── 💰 Money Flow Index (ChartAlert / Spider Software) ──">
        <option value="mfi_os">MFI Oversold ≤20 — high-volume sell exhaustion</option>
        <option value="mfi_ob">MFI Overbought ≥80 — high-volume buy exhaustion</option>
        <option value="mfi_bull_div">MFI Bullish Divergence — price falling, MFI rising</option>
        <option value="mfi_bear_div">MFI Bearish Divergence — price rising, MFI falling</option>
      </optgroup>
      <optgroup label="── 〰 CCI — Commodity Channel Index (ChartAlert / TradePoint) ──">
        <option value="cci_os">CCI Oversold ≤−100 — extreme weakness</option>
        <option value="cci_ob">CCI Overbought ≥+100 — extreme strength</option>
        <option value="cci_zero_up">CCI Zero Cross Up — just crossed above zero (bull signal)</option>
        <option value="cci_zero_dn">CCI Zero Cross Down — just crossed below zero (bear signal)</option>
        <option value="cci_os_exit">CCI Exit Oversold — recovering from ≤−100 (buy timing)</option>
        <option value="cci_ob_exit">CCI Exit Overbought — falling from ≥+100 (sell timing)</option>
      </optgroup>
      <optgroup label="── 📐 Linear Regression Channel (Spider / ChartAlert) ──">
        <option value="lr_above_chan">Above Upper Channel — price &gt;2σ above regression line</option>
        <option value="lr_below_chan">Below Lower Channel — price &lt;2σ below regression line</option>
        <option value="lr_slope_bull">LR Slope Positive — 50-day regression trending up</option>
        <option value="lr_slope_bear">LR Slope Negative — 50-day regression trending down</option>
        <option value="lr_mean_rev">LR Mean Reversion Setup — price at ±2σ extreme</option>
      </optgroup>
    </select>
    <button class="info-btn" onclick="showHelp('ti',document.getElementById('ti-strat').value)" title="Show strategy info">ℹ</button>
    </div>
  </div>

  <div class="cg"><label>Timeframe</label>
    <select id="ti-tf">
      <option value="d" selected>📅 Daily</option>
      <option value="w">📅 Weekly</option>
      <option value="m">📅 Monthly</option>
    </select>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="ti-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="ti-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="ti-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#ff9500;color:#000" onclick="scanTI()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ NOISELESS (P&F / Renko / 3-Line Break / RRG / EW) ════════════════ -->
<div id="ctrl-nl" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center">
    <select id="nl-strat" style="min-width:330px" onchange="updateNLInfo()">
      <optgroup label="── 📐 Point & Figure (Definedge TradePoint) ──">
        <option value="pnf_dbl_top">P&F Double Top Breakout — X column breaks prior X high</option>
        <option value="pnf_dbl_bot">P&F Double Bottom Breakdown — O column breaks prior O low</option>
        <option value="pnf_tpl_top">P&F Triple Top Breakout — breaks two prior matching highs</option>
        <option value="pnf_tpl_bot">P&F Triple Bottom Breakdown — breaks two prior matching lows</option>
        <option value="pnf_bull_cat">P&F Bullish Catapult — 3 consecutive X cols making higher highs</option>
        <option value="pnf_bear_cat">P&F Bearish Catapult — 3 consecutive O cols making lower lows</option>
        <option value="pnf_high_pole">P&F High Pole Warning — long X column, O retraces >50%</option>
        <option value="pnf_low_pole">P&F Low Pole Warning — long O column, X recovers >50%</option>
        <option value="pnf_asc_tri">P&F Ascending Triangle — equal X highs, rising O lows</option>
        <option value="pnf_desc_tri">P&F Descending Triangle — equal O lows, falling X highs</option>
        <option value="pnf_lt_bull">P&F Long-Tail Bull Reversal — giant O column then X reversal</option>
        <option value="pnf_lt_bear">P&F Long-Tail Bear Reversal — giant X column then O reversal</option>
        <option value="pnf_in_x">P&F In X Column (bull trend)</option>
        <option value="pnf_in_o">P&F In O Column (bear trend)</option>
      </optgroup>
      <optgroup label="── 🧱 Renko (ATR-based, Definedge style) ──">
        <option value="renko_bull">Renko Bull — current brick is UP</option>
        <option value="renko_bear">Renko Bear — current brick is DOWN</option>
        <option value="renko_rev_bull">Renko Reversed Bullish — just changed from DOWN to UP</option>
        <option value="renko_rev_bear">Renko Reversed Bearish — just changed from UP to DOWN</option>
        <option value="renko_3">Renko 3-Brick Run — 3 consecutive same-direction bricks</option>
        <option value="renko_5">Renko 5 ka Punch — 5 consecutive bricks (Definedge signature)</option>
        <option value="renko_dbl_top">Renko Double Top — two up-runs reaching same high level</option>
        <option value="renko_dbl_bot">Renko Double Bottom — two down-runs reaching same low level</option>
      </optgroup>
      <optgroup label="── 📊 3-Line Break (Japanese noiseless chart) ──">
        <option value="lb3_bull">3-Line Break Bull — current line is white (up)</option>
        <option value="lb3_bear">3-Line Break Bear — current line is black (down)</option>
        <option value="lb3_rev_bull">3-Line Break Reversal UP — just switched white after black lines</option>
        <option value="lb3_rev_bear">3-Line Break Reversal DOWN — just switched black after white lines</option>
        <option value="lb3_consec">3-Line Break 5+ Consecutive — 5+ lines same direction</option>
      </optgroup>
      <optgroup label="── 🔄 Relative Rotation Graph (Strike / RRG) — needs --nifty ──">
        <option value="rrg_leading">RRG Leading Quadrant — strong RS + improving momentum vs Nifty</option>
        <option value="rrg_weakening">RRG Weakening — strong RS but momentum fading (exit signal)</option>
        <option value="rrg_improving">RRG Improving — weak RS but momentum picking up (entry watch)</option>
        <option value="rrg_lagging">RRG Lagging — weak RS + weak momentum (avoid)</option>
      </optgroup>
      <optgroup label="── 🌊 Simplified Elliott Wave (approximation) ──">
        <option value="ew_w5_up">EW Wave 5 Up — potential 5th wave bullish impulse pattern</option>
        <option value="ew_w5_dn">EW Wave 5 Down — potential 5th wave bearish impulse pattern</option>
        <option value="ew_w3_ext">EW Wave 3 Extension — W3 ≥ 161.8% of W1 (strongest wave)</option>
        <option value="ew_abc">EW ABC Correction Done — 3-wave correction appears complete</option>
      </optgroup>
      <optgroup label="── 🎋 Kagi Chart (ChartAlert · Definedge TradePoint) ──">
        <option value="kagi_yang">Kagi Yang Line — rising line, bullish trend</option>
        <option value="kagi_yin">Kagi Yin Line — falling line, bearish trend</option>
        <option value="kagi_rev_yang">Kagi Reversal to Yang — just turned bullish</option>
        <option value="kagi_rev_yin">Kagi Reversal to Yin — just turned bearish</option>
        <option value="kagi_shoulder">Kagi Shoulder — Yang line exceeds prior Yang high (strong bull)</option>
        <option value="kagi_waist">Kagi Waist — Yin line breaks prior Yin low (strong bear)</option>
        <option value="kagi_consec">Kagi 3+ Consecutive Yang — sustained bull run</option>
      </optgroup>
    </select>
    <button class="info-btn" onclick="showHelp('nl',document.getElementById('nl-strat').value)" title="Strategy info">ℹ</button>
    </div>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="nl-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="nl-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="nl-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#e879f9;color:#000" onclick="scanNL()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ EXPERIMENTAL (VSA) ═══════════════════════════════════════════════ -->
<div id="ctrl-xp" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center">
    <select id="xp-strat" style="min-width:330px" onchange="updateXPInfo()">
      <optgroup label="── 📦 Big Volume Candle Strategy (VSA core) ──">
        <option value="vsa_big_bull">Big Volume Bull — ultra-high vol + wide spread + close upper 60% (institutional buy)</option>
        <option value="vsa_big_bear">Big Volume Bear — ultra-high vol + wide spread + close lower 40% (institutional sell)</option>
        <option value="vsa_big_indecision">Big Volume Indecision — ultra-high vol + close mid-range (battle zone)</option>
      </optgroup>
      <optgroup label="── 💪 Effort vs Result (VSA strength/weakness) ──">
        <option value="vsa_effort_up">Effort to Rise — wide up bar + high vol + upper close (accumulation)</option>
        <option value="vsa_effort_dn">Effort to Fall — wide down bar + high vol + lower close (distribution)</option>
        <option value="vsa_stopping">Stopping Volume — ultra-high vol down bar, close in upper half (supply absorbed)</option>
        <option value="vsa_climax_buy">Climactic Demand — peak vol day, wide up bar, close near high (supply exhausted)</option>
        <option value="vsa_climax_sell">Climactic Supply — peak vol day, wide down bar, close near low (demand exhausted)</option>
      </optgroup>
      <optgroup label="── 🔕 No Demand / No Supply (VSA quietness signals) ──">
        <option value="vsa_no_demand">No Demand — narrow up bar + below-avg vol (weak bounce, no institutional interest)</option>
        <option value="vsa_no_supply">No Supply — narrow down bar + below-avg vol (sellers gone, pre-breakout)</option>
        <option value="vsa_test_supply">Test for Supply — low vol down day after high vol up day (supply confirmed absent)</option>
      </optgroup>
      <optgroup label="── ⚡ Traps & Reversals ──">
        <option value="vsa_upthrust">Upthrust — wide spread, high vol, but closes near the LOW (false breakout up)</option>
        <option value="vsa_spring">Spring / Shakeout — wide spread, high vol, closes near the HIGH (false break down)</option>
        <option value="vsa_pseudo_ut">Pseudo Upthrust — new intraday high on high vol but closes in lower 30%</option>
        <option value="vsa_trap">Volume Trap — high vol, minimal price progress (buyers & sellers equally matched)</option>
      </optgroup>
    </select>
    <button class="info-btn" onclick="showHelp('xp',document.getElementById('xp-strat').value)" title="Strategy info">ℹ</button>
    </div>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="xp-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="xp-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="xp-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#f59e0b;color:#000" onclick="scanXP()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ PATTERNS (Harmonic + Chart Patterns) ════════════════════════════ -->
<div id="ctrl-patt" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <div style="display:flex;gap:6px;align-items:center">
    <select id="patt-strat" style="min-width:330px" onchange="updatePATTInfo()">
      <optgroup label="── 🎵 Harmonic Patterns (Gartley · Bat · Butterfly · Crab · Cypher) ──">
        <option value="harm_gartley_bull">Gartley Bullish — AB/XA≈0.618, AD/XA≈0.786 (most common)</option>
        <option value="harm_gartley_bear">Gartley Bearish</option>
        <option value="harm_bat_bull">Bat Bullish — AB/XA=0.382-0.5, AD/XA≈0.886 (tightest PRZ)</option>
        <option value="harm_bat_bear">Bat Bearish</option>
        <option value="harm_butterfly_bull">Butterfly Bullish — AD/XA=1.272-1.618 (extends past X)</option>
        <option value="harm_butterfly_bear">Butterfly Bearish</option>
        <option value="harm_crab_bull">Crab Bullish — AD/XA≈1.618 (deepest extension)</option>
        <option value="harm_crab_bear">Crab Bearish</option>
        <option value="harm_cypher_bull">Cypher Bullish — unique BC extension past XA</option>
        <option value="harm_cypher_bear">Cypher Bearish</option>
        <option value="harm_abcd_bull">AB=CD Bullish — CD≈AB in price (base harmonic)</option>
        <option value="harm_abcd_bear">AB=CD Bearish</option>
      </optgroup>
      <optgroup label="── 📐 Chart Patterns — Classic Geometric (Investar · ScanX) ──">
        <option value="cp_hs">Head &amp; Shoulders — 3 highs, middle tallest, equal shoulders</option>
        <option value="cp_inv_hs">Inverse H&amp;S — 3 lows, middle deepest (bullish reversal)</option>
        <option value="cp_dbl_top">Double Top — 2 equal highs at resistance</option>
        <option value="cp_dbl_bot">Double Bottom — 2 equal lows at support (bullish)</option>
        <option value="cp_rise_wedge">Rising Wedge — converging up-trendlines (bearish)</option>
        <option value="cp_fall_wedge">Falling Wedge — converging down-trendlines (bullish)</option>
        <option value="cp_bull_flag">Bull Flag — strong pole + tight downward consolidation</option>
        <option value="cp_bear_flag">Bear Flag — strong drop + tight upward consolidation</option>
      </optgroup>
      <optgroup label="── 〔W〕 Double Bottom — Daily/Weekly/Monthly (Neckline-Confirmed) ──">
        <option value="dbl_bot_d">Double Bottom (W) — Daily</option>
        <option value="dbl_bot_d_bo">Double Bottom (W) — Daily, Neckline Breakout</option>
        <option value="dbl_bot_w">Double Bottom (W) — Weekly</option>
        <option value="dbl_bot_w_bo">Double Bottom (W) — Weekly, Neckline Breakout</option>
        <option value="dbl_bot_m">Double Bottom (W) — Monthly</option>
        <option value="dbl_bot_m_bo">Double Bottom (W) — Monthly, Neckline Breakout</option>
      </optgroup>
      <optgroup label="── 〔M〕 Double Top — Daily/Weekly/Monthly (Neckline-Confirmed) ──">
        <option value="dbl_top_d">Double Top (M) — Daily</option>
        <option value="dbl_top_d_bo">Double Top (M) — Daily, Neckline Breakdown</option>
        <option value="dbl_top_w">Double Top (M) — Weekly</option>
        <option value="dbl_top_w_bo">Double Top (M) — Weekly, Neckline Breakdown</option>
        <option value="dbl_top_m">Double Top (M) — Monthly</option>
        <option value="dbl_top_m_bo">Double Top (M) — Monthly, Neckline Breakdown</option>
      </optgroup>
    </select>
    <button class="info-btn" onclick="showHelp('patt',document.getElementById('patt-strat').value)" title="Strategy info">ℹ</button>
    </div>
  </div>
  <div class="cg"><label>RS Rating ≥</label>
    <select id="patt-rs">
      <option value="0" selected>Any</option><option value="50">≥ 50</option>
      <option value="70">≥ 70</option><option value="80">≥ 80</option>
    </select>
  </div>
  <div class="cg"><label>Price Range ₹</label>
    <div class="prange">
      <input type="number" id="patt-pmin" placeholder="Min" min="0">
      <span>–</span>
      <input type="number" id="patt-pmax" placeholder="Max" min="0">
    </div>
  </div>
  <button class="btn" style="background:#06b6d4;color:#000" onclick="scanPATT()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ MARKET BREADTH ══════════════════════════════════════════════════ -->
<div id="ctrl-br" class="ctrl" style="display:none">
  <div class="cg"><label>Filter stocks by</label>
    <select id="br-filter" style="min-width:280px" onchange="renderBreadth()">
      <option value="all" selected>Market Overview — show all breadth metrics</option>
      <option value="above200">Stocks above 200 DMA (market health leaders)</option>
      <option value="stage2">Stocks in Weinstein Stage 2 (confirmed uptrend)</option>
      <option value="near52wh">Stocks near 52W high (within 5%)</option>
      <option value="pnf_x">Stocks in P&amp;F X Column (noiseless bull)</option>
      <option value="renko_bull">Stocks in Renko Bull bricks</option>
      <option value="ha_bull">Stocks with Heikin Ashi Bull candle</option>
      <option value="ema_fan">Stocks in EMA Fan Bull (price&gt;EMA9&gt;EMA21&gt;EMA50)</option>
      <option value="rs80">RS Rating ≥ 80 (top relative strength)</option>
    </select>
  </div>
  <button class="btn" style="background:#84cc16;color:#000" onclick="renderBreadth()">▶ SHOW</button>
</div>

<!-- ═══ BREAKOUT PRO ═══════════════════════════════════════════════════════ -->
<div id="ctrl-bp" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <select id="bp-strat" style="min-width:340px" onchange="updateBPInfo()">
      <optgroup label="── 🔄 VCP — Mark Minervini ──">
        <option value="vcp">VCP — 2+ contractions + volume dry-up + near 52W high</option>
        <option value="vcp3">VCP Tight — 3+ contractions (higher conviction)</option>
      </optgroup>
      <optgroup label="── 🎯 Pocket Pivot — Gil Morales ──">
        <option value="pp">Pocket Pivot — up-day volume &gt; highest down-day vol (10d)</option>
        <option value="pp_strong">Strong Pocket Pivot — 1.5× highest down-day volume</option>
      </optgroup>
      <optgroup label="── 📏 Tight Pattern — O'Neil ──">
        <option value="twt3">3-Week Tight — weekly closes within ±1.5%</option>
        <option value="twt4">4-Week Tight</option>
        <option value="twt5">5-Week Tight — strongest consolidation</option>
      </optgroup>
      <optgroup label="── 📈 RS Line — IBD / MarketSmith ──">
        <option value="rsh">RS New 52W High — strength line at yearly peak</option>
        <option value="rs_bo">RS Line Breakout — exits 3-month consolidation</option>
        <option value="rs_trend">RS Line Trending Up — 13-week positive slope</option>
      </optgroup>
      <optgroup label="── 🏆 A/D Score — IBD Rating ──">
        <option value="ads_a">A-Rating — heavy institutional accumulation (13W)</option>
        <option value="ads_ab">A or B — net accumulation phase</option>
        <option value="ads_de">D or E — distribution (short / avoid)</option>
      </optgroup>
      <optgroup label="── ⚡ High Volume Breakout ──">
        <option value="hvb">HVB — 52W high close + 2× average volume</option>
        <option value="hvb_strong">HVB Strong — 52W high + 3× average volume</option>
        <option value="near_hvb">Near-High HVB — within 2% of 52W high + 2× vol</option>
      </optgroup>
      <optgroup label="── 💧 Volume Dry-Up ──">
        <option value="vdu_bull">VDU Bullish — vol &lt;50% avg while price holds</option>
        <option value="vdu_40d">Vol at 40-Session Low — extreme drying</option>
      </optgroup>
      <optgroup label="── 💪 EMA Power Stack ──">
        <option value="ems_full">Full Stack — price &gt; 9 &gt; 21 &gt; 50 &gt; 200 EMA</option>
        <option value="ems_power">Power Stack — full stack + all EMAs rising</option>
        <option value="ems_part">Partial Stack — price &gt; 21 &gt; 50 &gt; 200 EMA</option>
      </optgroup>
      <optgroup label="── 📅 Inside Week ──">
        <option value="iw1">Inside Week — week fully inside prior week</option>
        <option value="iw2">Double Inside Week — 2 consecutive inside weeks</option>
        <option value="iw_up">Inside Week in Uptrend</option>
      </optgroup>
      <optgroup label="── 📐 Trendline ──">
        <option value="tlb">Descending Trendline Breakout</option>
      </optgroup>
    </select>
  </div>
  <div class="cg"><label>Min RS</label><select id="bp-rs"><option value="0" selected>Any</option><option value="50">≥ 50</option><option value="70">≥ 70</option><option value="80">≥ 80</option></select></div>
  <div class="cg"><label>Price ₹</label><div class="prange"><input type="number" id="bp-pmin" placeholder="Min" min="0"><span>–</span><input type="number" id="bp-pmax" placeholder="Max" min="0"></div></div>
  <button class="btn" style="background:#ff6b35;color:#000" onclick="scanBP()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>


<!-- ═══ SMART RANK ══════════════════════════════════════════════════════════ -->
<div id="ctrl-sr" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <select id="sr-strat" style="min-width:360px" onchange="updateSRInfo()">
      <optgroup label="── 🔢 Composite Momentum Score ──">
        <option value="ms_90">Momentum Score ≥ 90 — top 10% universe</option>
        <option value="ms_75">Momentum Score ≥ 75 — top quartile</option>
        <option value="ms_50">Momentum Score ≥ 50 — above average</option>
        <option value="ms_rising">Momentum Score Rising — outperforming universe</option>
      </optgroup>
      <optgroup label="── 📅 Seasonality (Definedge-style) ──">
        <option value="season_bull">Current Month Historically Bullish — avg return &gt;2%</option>
        <option value="season_strong">Current Month Strong — avg &gt;3% + hit rate &gt;65%</option>
        <option value="season_best">Best Month of Year for this stock</option>
        <option value="season_avoid">Current Month Historically Bearish — avg &lt;-1%</option>
      </optgroup>
      <optgroup label="── 📊 MTF Matrix (Definedge Matrix) ──">
        <option value="mtf_perfect">MTF Perfect — all 15 conditions bullish (D+W+M)</option>
        <option value="mtf_strong">MTF Strong — 12+ of 15 conditions bullish</option>
        <option value="mtf_7">MTF Aligned — 7+ conditions bullish</option>
        <option value="mtf_daily_max">Daily Max Score (5/5) — all daily checks pass</option>
      </optgroup>
      <optgroup label="── 🏔 All-Time High ──">
        <option value="ath">All-Time High Breakout — new ATH close</option>
        <option value="near_ath">Near ATH — within 2% of all-time high</option>
      </optgroup>
      <optgroup label="── 🗜 Triple Compression ──">
        <option value="tc3">Triple Compression — NR7 + VDU + BB Squeeze simultaneously</option>
        <option value="tc2">Double Compression — any 2 of 3 compression signals</option>
      </optgroup>
      <optgroup label="── 📦 Multi-Period RS (Definedge Smart RS) ──">
        <option value="mtfrs_90">Consistent RS Leader — top 10% in all 4 periods</option>
        <option value="mtfrs_75">Consistent RS Strong — top 25% across 1M/3M/6M/12M</option>
        <option value="r1m_top">1-Month Return Top 10%</option>
        <option value="r3m_top">3-Month Return Top 10%</option>
      </optgroup>
    </select>
  </div>
  <div class="cg"><label>Min RS</label><select id="sr-rs"><option value="0" selected>Any</option><option value="50">≥ 50</option><option value="70">≥ 70</option><option value="80">≥ 80</option></select></div>
  <div class="cg"><label>Price ₹</label><div class="prange"><input type="number" id="sr-pmin" placeholder="Min" min="0"><span>–</span><input type="number" id="sr-pmax" placeholder="Max" min="0"></div></div>
  <button class="btn" style="background:#00d4ff;color:#000" onclick="scanSR()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>


<!-- ═══ SWING SETUP ═══════════════════════════════════════════════════════════ -->
<div id="ctrl-swing" class="ctrl" style="display:none">
  <div class="cg"><label>Strategy</label>
    <select id="swing-strat" style="min-width:380px" onchange="updateSwingInfo()">
      <optgroup label="── 🗜 EMA Squeeze — 10/20/50/100/150/200 EMA Tight Coil ──">
        <option value="ema_sq_d">Daily EMA Squeeze — all EMAs within 2.5% (10-200)</option>
        <option value="ema_sq_d_tight">Daily Ultra-Tight (within 1%) — imminent breakout</option>
        <option value="ema_sq_w">Weekly EMA Squeeze — 10/20/50/100 EMA tight coil</option>
        <option value="ema_sq_m">Monthly EMA Squeeze — 10/20/50 EMA tight coil (long-term)</option>
        <option value="ema_sq_d_bull">Daily Squeeze + Bull Stack (coiling to break up)</option>
        <option value="ema_sq_w_bull">Weekly Squeeze + Bull Stack</option>
      </optgroup>
      <optgroup label="── 📊 MACD 5/35/5 Crossover — Fast Swing Variant ──">
        <option value="macd535_d_bull">Daily MACD 5/35 Bullish Cross — fresh today</option>
        <option value="macd535_d_bear">Daily MACD 5/35 Bearish Cross</option>
        <option value="macd535_w_bull">Weekly MACD 5/35 Bullish Cross — multi-week swing</option>
        <option value="macd535_w_bear">Weekly MACD 5/35 Bearish Cross</option>
        <option value="macd535_m_bull">Monthly MACD 5/35 Bullish Cross — positional</option>
        <option value="macd535_m_bear">Monthly MACD 5/35 Bearish Cross</option>
        <option value="macd535_d_zero_bull">Daily MACD 5/35 Zero-Line Cross Up</option>
        <option value="macd535_align">All Timeframes Aligned Bull (D+W+M MACD 5/35 bullish)</option>
      </optgroup>
      <optgroup label="── 🎯 Swing Trade Suggestion (1 Week – 3 Months) ──">
        <option value="swing_buy">Swing Buy Candidates — Score ≥60 + entry/stop/target</option>
        <option value="swing_strong">Strong Swing Buy — Score ≥75 + confluence</option>
        <option value="swing_breakout_ready">Breakout-Ready — EMA squeeze + MACD bull cross</option>
      </optgroup>
    </select>
  </div>
  <div class="cg"><label>Min RS</label><select id="swing-rs"><option value="0" selected>Any</option><option value="50">≥ 50</option><option value="70">≥ 70</option><option value="80">≥ 80</option></select></div>
  <div class="cg"><label>Price ₹</label><div class="prange"><input type="number" id="swing-pmin" placeholder="Min" min="0"><span>–</span><input type="number" id="swing-pmax" placeholder="Max" min="0"></div></div>
  <button class="btn" style="background:#fb923c;color:#000" onclick="scanSwing()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>


<!-- ═══ GAP SCANNER ════════════════════════════════════════════════════ -->
<div id="ctrl-gap" class="ctrl" style="display:none">
  <div class="cg"><label>Gap Type</label>
    <select id="gap-strat" style="min-width:310px" onchange="updateGapInfo()">
      <optgroup label="── 🟢 Gap Up ──">
        <option value="up1">Gap Up ≥1%</option>
        <option value="up2">Gap Up ≥2%</option>
        <option value="up3">Gap Up ≥3%</option>
        <option value="up_cont">Gap Up + Continuation (close > open)</option>
        <option value="up_reject">Gap Up Rejection (close &lt; open)</option>
        <option value="up_unfilled">Gap Up Unfilled</option>
      </optgroup>
      <optgroup label="── 🔴 Gap Down ──">
        <option value="dn1">Gap Down ≥1%</option>
        <option value="dn2">Gap Down ≥2%</option>
        <option value="dn3">Gap Down ≥3%</option>
        <option value="dn_cont">Gap Down + Continuation</option>
        <option value="dn_unfilled">Gap Down Unfilled</option>
      </optgroup>
      <optgroup label="── 📊 Multi-day ──">
        <option value="consec3">3 Consecutive Gap Ups</option>
      </optgroup>
    </select>
  </div>
  <div class="cg"><label>Min RS</label><select id="gap-rs"><option value="0" selected>Any</option><option value="50">≥50</option><option value="70">≥70</option></select></div>
  <div class="cg"><label>Price ₹</label><div class="prange"><input type="number" id="gap-pmin" placeholder="Min" min="0"><span>–</span><input type="number" id="gap-pmax" placeholder="Max" min="0"></div></div>
  <button class="btn" style="background:#ff8c42;color:#000" onclick="scanGap()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<!-- ═══ AND COMBINER ══════════════════════════════════════════════════════ -->
<div id="ctrl-combo" class="ctrl" style="display:none">
  <div style="font-family:var(--mono);font-size:10px;color:var(--mu);padding:0 0 8px;width:100%">⚗️ AND-combine up to 5 signals — only stocks matching ALL selected conditions appear</div>
  <div style="display:flex;flex-direction:column;gap:5px;width:100%">
    <div class="cg" style="flex-direction:row;align-items:center;gap:8px"><label style="min-width:24px;color:var(--mu);font-size:10px">C1</label><select id="combo-c1" style="min-width:280px" onchange="renderComboSummary()"><option value="">— skip —</option></select></div>
    <div class="cg" style="flex-direction:row;align-items:center;gap:8px"><label style="min-width:24px;color:var(--mu);font-size:10px">C2</label><select id="combo-c2" style="min-width:280px" onchange="renderComboSummary()"><option value="">— skip —</option></select></div>
    <div class="cg" style="flex-direction:row;align-items:center;gap:8px"><label style="min-width:24px;color:var(--mu);font-size:10px">C3</label><select id="combo-c3" style="min-width:280px" onchange="renderComboSummary()"><option value="">— skip —</option></select></div>
    <div class="cg" style="flex-direction:row;align-items:center;gap:8px"><label style="min-width:24px;color:var(--mu);font-size:10px">C4</label><select id="combo-c4" style="min-width:280px" onchange="renderComboSummary()"><option value="">— skip —</option></select></div>
    <div class="cg" style="flex-direction:row;align-items:center;gap:8px"><label style="min-width:24px;color:var(--mu);font-size:10px">C5</label><select id="combo-c5" style="min-width:280px" onchange="renderComboSummary()"><option value="">— skip —</option></select></div>
  </div>
  <div class="cg"><label>Min RS</label><select id="combo-rs"><option value="0" selected>Any</option><option value="50">≥50</option><option value="70">≥70</option><option value="80">≥80</option></select></div>
  <button class="btn" style="background:#c47aff;color:#000" onclick="scanCombo()">▶ SCAN</button>
  <button class="btn btn-out" onclick="exportCSV()">↓ CSV</button>
</div>

<div class="fbar" id="fbar" onclick="toggleInfo()" title="Click to expand/collapse details">Select a tab and click ▶ SCAN</div>
<div class="info-panel" id="info-panel"></div>

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
const BREADTH = __BREADTH_JSON__;

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
let rows=[],sc=4,sd=1,currentTab='home',lastRows=[];

// ── Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab){
  currentTab=tab;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  ['piv','smc','vol','mi','adv','t1','t2','t3','ti','nl','xp','patt','br','gap','combo','bp','sr','swing'].forEach(t=>{
    const el=document.getElementById('ctrl-'+t);
    if(el) el.style.display=t===tab?'flex':'none';
  });
  if(tab==='home'){
    document.getElementById('ts').innerHTML=renderHome();
    setFbar('🏠 <b>Quick Launch</b> · Click any card to instantly run the scan · Strategies ranked by professional success rate','');
    updateStats([]); return;
  }
  if(tab==='lookup'){
    document.getElementById('ts').innerHTML=renderLookup();
    setFbar('🔍 <b>Stock Lookup</b> · Enter a stock symbol to see every scanner signal it matches and a composite Buy/Sell score','');
    updateStats([]); return;
  }
  if(tab==='iview'){
    document.getElementById('ts').innerHTML=renderIndexView();
    document.getElementById('iv-idx').value=document.getElementById('global-idx').value||'0';
    scanIndexView();
    setFbar('📋 <b>Index View</b> · All stocks ranked by composite Buy/Sell score · Click ⓘ or row for signal breakdown','');
    updateStats([]); return;
  }
  if(tab==='toppicks'){
    document.getElementById('ts').innerHTML=renderTopPicks();
    scanTopPicks();
    setFbar('🏆 <b>Top Picks</b> · Universe-wide ranked board · Confluence-weighted score rewards stocks confirmed by multiple independent signal categories','');
    updateStats([]); return;
  }
  if(tab==='help'){
    document.getElementById('ts').innerHTML=renderHelp();
    setFbar('❓ <b>Help</b> · Complete reference for all 100+ scan signals · Click any section to expand','');
    updateStats([]); return;
  }
  document.getElementById('ts').innerHTML='<div class="nodata">Select options above — scan runs automatically</div>';
  if(tab==='piv') scan();
  else if(tab==='smc') scanSMC();
  else if(tab==='vol') scanVol();
  else if(tab==='mi') scanMI();
  else if(tab==='adv'){ updateAdvInfo(); scanAdv(); }
  else if(tab==='t1'){ updateT1Info(); scanT1(); }
  else if(tab==='t2'){ updateT2Info(); scanT2(); }
  else if(tab==='t3'){ updateT3Info(); scanT3(); }
  else if(tab==='ti'){ updateTIInfo(); scanTI(); }
  else if(tab==='nl'){ updateNLInfo(); scanNL(); }
  else if(tab==='xp'){ updateXPInfo(); scanXP(); }
  else if(tab==='patt'){ updatePATTInfo(); scanPATT(); }
  else if(tab==='br'){ renderBreadth(); }
  else if(tab==='gap'){ updateGapInfo(); scanGap(); }
  else if(tab==='combo'){ renderComboSummary(); }
  else if(tab==='bp'){ updateBPInfo(); scanBP(); }
  else if(tab==='sr'){ updateSRInfo(); scanSR(); }
  else if(tab==='swing'){ updateSwingInfo(); scanSwing(); }
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
  const detail=`${FBAR[type]||''} &nbsp;·&nbsp; <span class="hb">Source:</span> ${TF_SRC[tf]||tf} &nbsp;·&nbsp; <span style="color:var(--mu);font-size:10px">Click ⓘ for more detail</span>`;
  setFbar((FBAR[type]||'')+' &nbsp;·&nbsp; <span class="hb">Source:</span> '+(TF_SRC[tf]||tf), detail);
}

// ── Settings persistence — URL hash (primary) + localStorage (fallback) ───────
// URL hash = shareable across devices and users; each user's URL is independent.
// localStorage = same-device fallback when no hash is present.
// Multiple users on the same server are fully isolated — nothing is stored server-side.
const DEFAULTS={type:'fibonacci',lv:'S2',tf:'y',pr:'2',idx:'0',dma:'any',rs:'0',bc:'0',prMin:'',prMax:'',cvol:false,c52h:false,c52l:false};
const PK='nse_scanner_v1';

function _collectPrefs(){
  return{
    type:document.getElementById('sp-type').value,
    lv:  document.getElementById('sp-lvl').value,
    tf:  document.getElementById('sp-tf').value,
    pr:  document.getElementById('sp-pr').value,
    idx: document.getElementById('global-idx').value,
    dma: document.getElementById('dma-flt').value,
    rs:  document.getElementById('rs-flt').value,
    bc:  document.getElementById('bc-flt').value,
    prMin:document.getElementById('pr-min').value,
    prMax:document.getElementById('pr-max').value,
    cvol: document.getElementById('cb-vol').checked?'1':'',
    c52h: document.getElementById('cb-52h').checked?'1':'',
    c52l: document.getElementById('cb-52l').checked?'1':'',
  };
}

function applyPrefs(p){
  if(p.type) document.getElementById('sp-type').value=p.type;
  if(p.tf)   document.getElementById('sp-tf').value=p.tf;
  if(p.pr)   document.getElementById('sp-pr').value=p.pr;
  if(p.idx)  document.getElementById('global-idx').value=p.idx;
  if(p.dma)  document.getElementById('dma-flt').value=p.dma;
  if(p.rs)   document.getElementById('rs-flt').value=p.rs;
  if(p.bc)   document.getElementById('bc-flt').value=p.bc;
  document.getElementById('pr-min').value=p.prMin||'';
  document.getElementById('pr-max').value=p.prMax||'';
  document.getElementById('cb-vol').checked=p.cvol==='1'||p.cvol===true;
  document.getElementById('cb-52h').checked=p.c52h==='1'||p.c52h===true;
  document.getElementById('cb-52l').checked=p.c52l==='1'||p.c52l===true;
  onTypeChange(p.lv||'');
}

function savePrefs(){
  const p=_collectPrefs();
  // 1. Write to URL hash (no reload; shareable link)
  history.replaceState(null,'','#'+new URLSearchParams(p).toString());
  // 2. Also persist to localStorage as same-device fallback
  try{localStorage.setItem(PK,JSON.stringify(p));}catch(e){}
  const l=document.getElementById('saved-lbl');
  l.textContent='✓ SAVED';setTimeout(()=>l.textContent='',1800);
}

function loadPrefs(){
  // Priority 1: URL hash (shared/bookmarked link)
  if(window.location.hash.length>1){
    try{
      const p=Object.fromEntries(new URLSearchParams(window.location.hash.slice(1)));
      return Object.keys(p).length?p:null;
    }catch(e){}
  }
  // Priority 2: localStorage (previous session, same device)
  try{return JSON.parse(localStorage.getItem(PK));}catch(e){return null;}
}

function resetPrefs(){
  // Clear both URL hash and localStorage
  history.replaceState(null,'',window.location.pathname);
  try{localStorage.removeItem(PK);}catch(e){}
  applyPrefs(DEFAULTS);
  const l=document.getElementById('saved-lbl');
  l.textContent='↺ RESET';setTimeout(()=>l.textContent='',1800);
  scan();
}

function shareLink(){
  // Build a clean shareable URL with current settings encoded in hash
  const p=_collectPrefs();
  const url=window.location.origin+window.location.pathname+'#'+new URLSearchParams(p).toString();
  navigator.clipboard.writeText(url).then(()=>{
    const l=document.getElementById('saved-lbl');
    l.textContent='🔗 LINK COPIED';setTimeout(()=>l.textContent='',2500);
  }).catch(()=>{
    prompt('Copy this link:',url);
  });
}

// ── Shared helpers ─────────────────────────────────────────────────────────
function symCell(sym){
  const ico=`<span class="sd-ico" onmouseenter="_sdHover('${sym}')" title="Stock detail">i</span>`;
  const lnk=`<a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${sym}" target="_blank" rel="noopener">${sym}</a>`;
  return`<td class="csym" style="white-space:nowrap">${ico}${lnk}</td>`;
}
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


// ── Index filter helper (handles numeric N50/N100/... AND 'fno' filter) ──
function passesIdx(s){
  var idxV=document.getElementById('global-idx').value;
  if(idxV==='fno') return !!s.fno;
  var idxN=parseInt(idxV)||0;
  if(idxN===0) return true;
  return s.idx>0 && s.idx<=idxN;
}
function scan(){
  const type=document.getElementById('sp-type').value;
  const lvKey=document.getElementById('sp-lvl').value;
  const tf=document.getElementById('sp-tf').value;
  const prox=parseFloat(document.getElementById('sp-pr').value);
  const bounceF=document.getElementById('sp-bounce')?.value||'';
  
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
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(dmaF==='above'&&!s.above200)continue;
    if(dmaF==='below'&&s.above200)continue;
    if(s.rs<rsF)continue;
    const src={d:s.d,w:s.w,m:s.m,q:s.q,ytd:s.ytd,y:s.y}[tf]||s.m;
    const p=computePivots(type,src.pH,src.pL,src.pC,src.pO,src.cO);
    let bestTgt=null,bestDist=Infinity,bestLv=null;
    for(const lv of lvels){const t=p[lv];if(t===undefined)continue;const d=(s.price-t)/Math.abs(t)*100;if(Math.abs(d)<Math.abs(bestDist)){bestDist=d;bestTgt=t;bestLv=lv;}}
    if(bestTgt===null||Math.abs(bestDist)>prox)continue;
    const b6=bounceCount(s.mh,s.wh,s.qh,tf,type,bestLv,6);
    const b12=bounceCount(s.mh,s.wh,s.qh,tf,type,bestLv,12);
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
  setFbar('<b>Price Action / Smart Money Concepts</b> · '+FBAR_LABELS[sig]);
  rows=[];
  for(const s of S){
    if(!s.smc||Object.keys(s.smc).length===0)continue;
    if(!passesIdx(s))continue;
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
  setFbar('<b>Volume / Institutional Footprint</b> · '+(FBAR_VOL[sig]||sig));
  rows=[];
  for(const s of S){
    if(!s.vol||Object.keys(s.vol).length===0)continue;
    if(!passesIdx(s))continue;
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
  
  const prMin=parseFloat(document.getElementById('mi-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('mi-pmax').value)||Infinity;

  const MT_LABELS=['Price > 150 & 200 SMA','150 SMA > 200 SMA','200 SMA rising','50 SMA > 150 & 200 SMA','Price > 50 SMA','≥ 25% above 52W low','Within 25% of 52W high','RS ≥ 70'];
  setFbar(
    '<b>Minervini Trend Template</b> · 8 conditions: '+ MT_LABELS.map((l,i)=>`<span style="color:var(--mu)">${i+1}.${l}</span>`).join(' · ')+'<br>'+
    '<b>Weinstein Stage Analysis</b> · Stage 2 (Advancing): price above rising 30-week (150-day) SMA');
  rows=[];
  for(const s of S){
    if(!s.mi||Object.keys(s.mi).length===0)continue;
    if(!passesIdx(s))continue;
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

// ── ADVANCED STRATEGIES ───────────────────────────────────────────────────────
const ADV_INFO = {
  darvas:       '<b>Darvas Box</b> · Nicolas Darvas (made $2M in 18 months) · 52W-high stock consolidates in tight box (≤20% range) · Volume breakout above box top = entry signal',
  vcp:          '<b>VCP — Volatility Contraction Pattern</b> · Mark Minervini · 3+ progressively tighter contractions with declining volume · Final tight base (&lt;10%) + breakout = high-probability entry',
  wyckoff_acc:  '<b>Wyckoff Accumulation</b> · Richard Wyckoff · Price stabilising after decline, volume decreasing, 50SMA &lt; 200SMA · Institutions quietly accumulating; watch for spring or BOS',
  wyckoff_markup:'<b>Wyckoff Markup</b> · Price advancing, above both SMAs · Trending phase — buy pullbacks to support',
  wyckoff_spring:'<b>Wyckoff Spring / Upthrust</b> · Spring = fake breakdown below support that quickly reverses (bullish) · Upthrust = fake breakout above resistance that quickly reverses (bearish)',
  turtle20:     '<b>Turtle Trading 20-day</b> · Richard Dennis · Price closed above highest high of last 20 days · Short-term breakout entry; exit if price closes below 10-day low',
  turtle55:     '<b>Turtle Trading 55-day</b> · Stronger signal — price broke above 55-day high · Long-term breakout; used by professional trend-followers',
  ichi_above:   '<b>Ichimoku — Above Cloud</b> · Price above both Span A and Span B · Bullish bias confirmed; cloud provides support zone below',
  ichi_tk_bull: '<b>Ichimoku — TK Cross Bullish</b> · Tenkan-sen (9-period) crossed above Kijun-sen (26-period) · Medium-term buy signal, especially when above the cloud',
  ichi_kbo:     '<b>Ichimoku — Kumo Breakout</b> · Price just crossed above the cloud (Span A &amp; B) · Strongest Ichimoku signal; often precedes sustained moves',
  td_buy9:      '<b>TD Sequential — Buy 9</b> · Tom DeMark · 9 consecutive closes each below close 4 bars ago · Exhaustion of selling; potential reversal. Look for confluence with support',
  td_sell9:     '<b>TD Sequential — Sell 9</b> · 9 consecutive closes each above close 4 bars ago · Exhaustion of buying; potential reversal. Look for confluence with resistance',
  st_up:        '<b>Supertrend — Bullish</b> · ATR-based trailing indicator (period=10, mult=3) · Price above Supertrend line = uptrend confirmed',
  st_flip_up:   '<b>Supertrend — Fresh Flip Up</b> · Supertrend just flipped from bearish to bullish (today or yesterday) · Early trend-change signal with tight stop',
  elder_buy:    '<b>Elder Triple Screen — Buy Setup</b> · Dr. Alexander Elder · Screen 1: weekly MACD bullish (trend) · Screen 2: daily Force Index negative (pullback) · Enter long on next bar',
  elder_sell:   '<b>Elder Triple Screen — Sell Setup</b> · Screen 1: weekly MACD bearish · Screen 2: daily Force Index positive (rally) · Enter short on next bar',
};

function updateAdvInfo(){
  const s=document.getElementById('adv-strat').value;
  setFbar('<span style="color:var(--warn)">Advanced Strategies</span> · '+(ADV_INFO[s]||s));
}

function scanAdv(){
  const strat=document.getElementById('adv-strat').value;
  
  const prMin=parseFloat(document.getElementById('adv-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('adv-pmax').value)||Infinity;
  updateAdvInfo();
  rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(!s.adv)continue;
    const a=s.adv; let matched=false; let extra={};

    if(strat==='darvas')      { const d=a.darvas;   matched=d&&(d.breakout||d.in_box); extra={sig:d?.breakout?'Breakout!':'In Box',box_top:d?.top,box_bot:d?.bot,vol_ok:d?.vol_ok,box_dt:d?.dt}; }
    if(strat==='vcp')         { const v=a.vcp;      matched=v&&v.is_vcp; extra={sig:'VCP',widths:v?.widths?.join('→')+'%',tight:v?.tightest+'%',pivot:v?.pivot}; }
    if(strat==='wyckoff_acc') { const w=a.wyckoff;  matched=w&&w.phase==='Accumulation'; extra={sig:w?.phase,spring:w?.spring,sup:w?.support,res:w?.resistance,vol:w?.vol_trend}; }
    if(strat==='wyckoff_markup'){const w=a.wyckoff; matched=w&&w.phase==='Markup';       extra={sig:w?.phase,spring:w?.spring,sup:w?.support,res:w?.resistance,vol:w?.vol_trend}; }
    if(strat==='wyckoff_spring'){const w=a.wyckoff; matched=w&&(w.spring||w.upthrust);   extra={sig:w?.spring?'Spring ↑':'Upthrust ↓',phase:w?.phase,sup:w?.support,res:w?.resistance}; }
    if(strat==='turtle20')    { const t=a.turtle;   matched=t&&t.bo20; extra={sig:'20d BO',dc20h:t?.dc20h,dc20l:t?.dc20l,dc55h:t?.dc55h,dc10l:t?.dc10l}; }
    if(strat==='turtle55')    { const t=a.turtle;   matched=t&&t.bo55; extra={sig:'55d BO',dc55h:t?.dc55h,dc55l:t?.dc55l,dc20h:t?.dc20h,dc10l:t?.dc10l}; }
    if(strat==='ichi_above')  { const i=a.ichi;     matched=i&&i.above; extra={sig:'Above Cloud',span_a:i?.span_a,span_b:i?.span_b,tenkan:i?.tenkan,kijun:i?.kijun,thick:i?.thick+'%'}; }
    if(strat==='ichi_tk_bull'){ const i=a.ichi;     matched=i&&i.tk_bull; extra={sig:'TK Cross ↑',tenkan:i?.tenkan,kijun:i?.kijun,above:i?.above?'Yes':'No',thick:i?.thick+'%'}; }
    if(strat==='ichi_kbo')    { const i=a.ichi;     matched=i&&i.kbo;  extra={sig:'Kumo BO',span_a:i?.span_a,span_b:i?.span_b,tenkan:i?.tenkan,kijun:i?.kijun}
    const dc=a.dc||{};
    if(strat==='dc20_bull')  {matched=!!dc.dc20_bull;     sig='📐 DC20↑'; extra={ub:dc.ub20,pct:dc.pct_from_ub+'%'};}
    if(strat==='dc20_bear')  {matched=!!dc.dc20_bear;     sig='📐 DC20↓'; extra={lb:dc.lb20};}
    if(strat==='dc55_bull')  {matched=!!dc.dc55_bull;     sig='🐢 DC55↑'; extra={ub:dc.ub55};}
    if(strat==='dc55_bear')  {matched=!!dc.dc55_bear;     sig='🐢 DC55↓'; extra={lb:dc.lb55};}
    if(strat==='dc_squeeze') {matched=!!dc.squeeze;       sig='🗜 DC Sq';  extra={bw:dc.bw20+'%'};}
    if(strat==='dc_mid20')   {matched=!!dc.pullback_mid20;sig='📐 DC Mid'; extra={mid:dc.mid20};}
    const gmma=a.gmma||{};
    if(strat==='gmma_cross_bull'){matched=!!gmma.cross_bull;sig='🌈 GMMA X↑';  extra={spread:gmma.spread_pct+'%'};}
    if(strat==='gmma_cross_bear'){matched=!!gmma.cross_bear;sig='🌈 GMMA X↓';  extra={spread:gmma.spread_pct+'%'};}
    if(strat==='gmma_bull')     {matched=!!gmma.bull;       sig='🌈 GMMA Bull'; extra={spread:gmma.spread_pct+'%'};}
    if(strat==='gmma_all_bull') {matched=!!gmma.all_bull;   sig='🌈💪 GMMA AB'; extra={spread:gmma.spread_pct+'%'};}
    if(strat==='gmma_bear')     {matched=!!gmma.bear;       sig='🌈 GMMA Bear'; extra={};}
    if(strat==='gmma_compress') {matched=!!gmma.compress;   sig='🌈🗜 GMMA Cmp';extra={};}; }
    if(strat==='td_buy9')     { const t=a.td;       matched=t&&t.signal==='buy_9'; extra={sig:'Buy 9 ✓',count:t?.count,dir:t?.dir}; }
    if(strat==='td_sell9')    { const t=a.td;       matched=t&&t.signal==='sell_9'; extra={sig:'Sell 9 ✓',count:t?.count,dir:t?.dir}; }
    if(strat==='st_up')       { const t=a.st;       matched=t&&t.direction==='up'; extra={sig:'ST ↑',st_val:t?.value,atr:t?.atr,flipped:t?.flipped?'NEW':''}; }
    if(strat==='st_flip_up')  { const t=a.st;       matched=t&&t.direction==='up'&&t.flipped; extra={sig:'Flip ↑ NEW',st_val:t?.value,atr:t?.atr}; }
    if(strat==='elder_buy')   { const e=a.elder;    matched=e&&e.signal==='buy_setup'; extra={sig:'Buy Setup',macd:e?.macd_trend,rising:e?.macd_rising?'↑':'→',fi:e?.fi}; }
    if(strat==='elder_sell')  { const e=a.elder;    matched=e&&e.signal==='sell_setup'; extra={sig:'Sell Setup',macd:e?.macd_trend,fi:e?.fi}; }

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      strat,extra,_tab:'adv'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── TIER-1 STRATEGIES ─────────────────────────────────────────────────────────
const T1_INFO = {
  gc_fresh_golden:  '<b>Golden Cross</b> · 50 SMA crossed <b>above</b> 200 SMA in last 10 days · Most-watched institutional signal',
  gc_fresh_death:   '<b>Death Cross</b> · 50 SMA crossed <b>below</b> 200 SMA in last 10 days · Institutional selling trigger',
  gc_bull_zone:     '<b>Bull Zone</b> · 50 SMA currently above 200 SMA · Long-term uptrend backdrop',
  gc_near_golden:   '<b>Near Golden Cross</b> · 50/200 SMA gap closing (within 1%) · Cross imminent — enter early',
  rsi_bull_div:     '<b>RSI Bullish Divergence</b> · Price lower low, RSI higher low · Hidden accumulation — highest-probability reversal signal',
  rsi_bear_div:     '<b>RSI Bearish Divergence</b> · Price higher high, RSI lower high · Hidden distribution — exit or short signal',
  rsi_oversold:     '<b>RSI Oversold ≤ 30</b> · Extreme selling exhaustion · Combine with support for high-conviction long',
  rsi_overbought:   '<b>RSI Overbought ≥ 70</b> · Extreme buying exhaustion · Combine with resistance for exit or short',
  bb_squeeze_bull:  '<b>BB Squeeze + Bull Momentum</b> · Bands at tightest 25% + positive 12-bar momentum · Upside explosion imminent',
  bb_squeeze_bear:  '<b>BB Squeeze + Bear Momentum</b> · Bands tight + negative momentum · Downside explosion imminent',
  bb_squeeze_any:   '<b>BB Squeeze (any direction)</b> · Volatility historically low · Big directional move imminent — wait for break with volume',
  w52_breakout:     '<b>52W High Breakout + Volume</b> · Broke above 52W high with volume ≥ 1.5× avg · Strongest momentum signal',
  w52_close_above:  '<b>52W High Close Breakout</b> · Closing price above prior 52W high · Close above is more significant than intraday',
  w52_near:         '<b>Near 52W High (within 3%)</b> · Approaching 52W resistance · Watch list for breakout',
  w52_consol_near:  '<b>Near 52W High + Consolidating</b> · Within 3% of 52W high + last 10d range &lt;5% · Classic coiling pre-breakout',
  nr7_bull:         '<b>NR7 Bull</b> · Narrowest range of 7 bars + uptrend bias · Volatility expansion up imminent · Toby Crabel 70%+ directional accuracy',
  nr7_bear:         '<b>NR7 Bear</b> · Narrowest 7-bar range + downtrend bias · Volatility expansion down imminent',
  nr4:              '<b>NR4</b> · Narrowest range of 4 bars · Shorter compression — active trader setup',
  inside_bar:       '<b>Inside Bar</b> · High &lt; yesterday high AND Low &gt; yesterday low · Full consolidation — mother-bar breakout = entry',
  nr7_inside:       '<b>NR7 + Inside Bar</b> · Both compressions together · Double signal — highest-probability volatility expansion setup',
};

function updateT1Info(){
  const s=document.getElementById('t1-strat').value;
  setFbar('<span style="color:#c47aff">⭐ Tier-1 Strategy</span> · '+(T1_INFO[s]||s));
}

function scanT1(){
  const strat  = document.getElementById('t1-strat').value;
  const idxF   = parseInt(document.getElementById('global-idx').value);
  const rsMin  = parseInt(document.getElementById('t1-rs').value);
  const rsiMin = parseInt(document.getElementById('t1-rsi-min').value);
  const rsiMax = parseInt(document.getElementById('t1-rsi-max').value);
  const prMin  = parseFloat(document.getElementById('t1-pmin').value)||0;
  const prMax  = parseFloat(document.getElementById('t1-pmax').value)||Infinity;
  updateT1Info();
  rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.t1)continue;
    const t=s.t1; const gc=t.gc||{}; const ri=t.rsi||{}; const bb=t.bb||{}; const w=t.w52||{}; const nr=t.nr||{};
    if(ri.rsi!==undefined){if(ri.rsi<rsiMin||ri.rsi>rsiMax)continue;}
    let matched=false; let sig=''; let extra={};

    if(strat==='gc_fresh_golden')   { matched=gc.cross==='golden'; sig='🟡 Golden Cross'; extra={cross_dt:gc.cross_dt,s50:gc.s50,s200:gc.s200,gap:gc.gap_pct+'%',vs200:gc.dist_200+'%'}; }
    if(strat==='gc_fresh_death')    { matched=gc.cross==='death';  sig='☠ Death Cross';  extra={cross_dt:gc.cross_dt,s50:gc.s50,s200:gc.s200,gap:gc.gap_pct+'%',vs200:gc.dist_200+'%'}; }
    if(strat==='gc_bull_zone')      { matched=gc.alignment==='bull'; sig='📈 Bull Zone'; extra={s50:gc.s50,s200:gc.s200,gap:gc.gap_pct+'%',vs50:gc.dist_50+'%',vs200:gc.dist_200+'%'}; }
    if(strat==='gc_near_golden')    { matched=gc.alignment==='bear'&&typeof gc.gap_pct==='number'&&gc.gap_pct>=-1; sig='⚡ Near GC'; extra={gap:gc.gap_pct+'%',s50:gc.s50,s200:gc.s200}; }
    if(strat==='rsi_bull_div')   { matched=!!ri.bull_div;   sig='📗 RSI Bull Div'; extra={rsi:ri.rsi}; }
    if(strat==='rsi_bear_div')   { matched=!!ri.bear_div;   sig='📕 RSI Bear Div'; extra={rsi:ri.rsi}; }
    if(strat==='rsi_oversold')   { matched=!!ri.oversold;   sig='🟢 RSI OS ≤30';   extra={rsi:ri.rsi}; }
    if(strat==='rsi_overbought') { matched=!!ri.overbought; sig='🔴 RSI OB ≥70';   extra={rsi:ri.rsi}; }
    if(strat==='rsi_above50')    {matched=!!ri.above50;      sig='📊 RSI>50 '+ri.rsi; extra={rsi:ri.rsi};}
    if(strat==='rsi_below50')    {matched=!!ri.below50;      sig='📉 RSI<50 '+ri.rsi; extra={rsi:ri.rsi};}
    if(strat==='rsi_cross50_bull'){matched=!!ri.cross50_bull;sig='⚡ RSI↑50';          extra={rsi:ri.rsi};}
    if(strat==='rsi_cross50_bear'){matched=!!ri.cross50_bear;sig='⚡ RSI↓50';          extra={rsi:ri.rsi};}
    if(strat==='rsi_bull_zone')  {matched=!!ri.bull_zone;    sig='📊 RSI Zone '+ri.rsi;extra={rsi:ri.rsi};}
    if(strat==='bb_squeeze_bull') { matched=!!bb.in_squeeze&&bb.bias==='bull'; sig='🔵 BB Squeeze ↑'; extra={bw:bb.bw,mom:bb.momentum,upper:bb.upper,lower:bb.lower,mid:bb.mid}; }
    if(strat==='bb_squeeze_bear') { matched=!!bb.in_squeeze&&bb.bias==='bear'; sig='🔵 BB Squeeze ↓'; extra={bw:bb.bw,mom:bb.momentum,upper:bb.upper,lower:bb.lower,mid:bb.mid}; }
    if(strat==='bb_squeeze_any')  { matched=!!bb.in_squeeze;                   sig='🔵 BB Squeeze';   extra={bw:bb.bw,bias:bb.bias,upper:bb.upper,lower:bb.lower,mid:bb.mid}; }
    if(strat==='w52_breakout')    { matched=!!w.breakout&&!!w.vol_surge; sig='🚀 52W BO+Vol';  extra={w52h:w.w52h,pct:w.pct_from+'%'}; }
    if(strat==='w52_close_above') { matched=!!w.close_bo;                sig='🚀 52W Close BO'; extra={w52h:w.w52h,pct:w.pct_from+'%',vol:w.vol_surge?'Yes':'No'}; }
    if(strat==='w52_near')        { matched=!!w.near;                    sig='👀 Near 52W H';  extra={w52h:w.w52h,pct:w.pct_from+'%',consol:w.consol?'Yes':'No'}; }
    if(strat==='w52_consol_near') { matched=!!w.near&&!!w.consol;        sig='🎯 52W+Consol';  extra={w52h:w.w52h,pct:w.pct_from+'%'}; }
    if(strat==='nr7_bull')   { matched=!!nr.is_nr7&&nr.bias==='bull'; sig='📦 NR7 ↑'; extra={comp:nr.compression+'%',rng:nr.rng,atr14:nr.atr14,nr4:nr.is_nr4?'Yes':'No'}; }
    if(strat==='nr7_bear')   { matched=!!nr.is_nr7&&nr.bias==='bear'; sig='📦 NR7 ↓'; extra={comp:nr.compression+'%',rng:nr.rng,atr14:nr.atr14,nr4:nr.is_nr4?'Yes':'No'}; }
    if(strat==='nr4')        { matched=!!nr.is_nr4;                   sig='📦 NR4';   extra={comp:nr.compression+'%',rng:nr.rng,atr14:nr.atr14,bias:nr.bias}; }
    if(strat==='inside_bar') { matched=!!nr.is_inside;                sig='📦 Inside'; extra={rng:nr.rng,atr14:nr.atr14,bias:nr.bias,comp:nr.compression+'%'}; }
    if(strat==='nr7_inside') { matched=!!nr.is_nr7&&!!nr.is_inside;
    // ① SuperTrend
    const stt_=s.t1&&(s.t1.stt||s.t1.st)||{};
    if(strat==='st_bull')    {matched=!!stt_.bull;  sig='📈 ST Bull';    extra={st:stt_.st,dist:(stt_.dist||0)+'%',consec:stt_.consec};}
    if(strat==='st_bear')    {matched=!!stt_.bear;  sig='📉 ST Bear';    extra={st:stt_.st};}
    if(strat==='st_flip_bull'){matched=!!stt_.flip_bull; sig='⚡ ST Flip↑'; extra={st:stt_.st};}
    if(strat==='st_flip_bear'){matched=!!stt_.flip_bear; sig='⚡ ST Flip↓'; extra={st:stt_.st};}
    if(strat==='st_consec5') {matched=(stt_.consec||0)>=5; sig='🔥 ST '+(stt_.consec||0)+'× Bull'; extra={consec:stt_.consec};}
    if(strat==='st_tight')   {matched=!!stt_.bull&&Math.abs(stt_.dist||99)<1; sig='🎯 ST Tight'; extra={dist:(stt_.dist||0)+'%'};}  sig='🎯 NR7+IB'; extra={comp:nr.compression+'%',rng:nr.rng,atr14:nr.atr14,bias:nr.bias}; }

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      rsi:ri.rsi||0, sig, extra, strat, _tab:'t1'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── TIER-2 STRATEGIES ─────────────────────────────────────────────────────────
const T2_INFO = {
  ch_breakout:    '<b>Cup & Handle Breakout</b> · William O\'Neil · U-shaped base (8-55% deep) + tight handle + price breaks the cup lip with volume · Most big winners form this before major moves',
  ch_near:        '<b>Cup & Handle Near Breakout</b> · Within 3% of the cup lip · Watch list: stock coiling for breakout · Enter when it clears the lip with volume',
  ch_handle:      '<b>In Handle Zone</b> · Cup formed, now consolidating in handle (tight range &lt;12%) · Best time to build a position ahead of the breakout',
  macd_bull_div:  '<b>MACD Bullish Divergence</b> · Price makes lower low but MACD makes higher low · Smart money accumulating while weak hands sell · High-probability reversal',
  macd_bear_div:  '<b>MACD Bearish Divergence</b> · Price makes higher high but MACD makes lower high · Distribution occurring at highs · Exit longs or initiate shorts',
  macd_flip_bull: '<b>MACD Histogram Flipped +ve</b> · Histogram just crossed above zero · Early trend change signal · Combine with support for entry',
  macd_flip_bear: '<b>MACD Histogram Flipped -ve</b> · Histogram just crossed below zero · Early downtrend signal · Reduce longs or short',
  macd_bull_trend:'<b>MACD Bull Trend</b> · MACD line above signal line · Trend is up · Use as a backdrop filter to only take longs',
  zs_oversold:    '<b>Z-Score Oversold (−2σ)</b> · Price is 2 standard deviations below its 20-day MA · Statistically extreme — expect mean reversion up · Buy with a tight stop below recent low',
  zs_overbought:  '<b>Z-Score Overbought (+2σ)</b> · Price 2 std devs above 20-day MA · Statistically stretched — expect pullback · Exit longs, do not chase',
  zs_extreme_os:  '<b>Extreme Oversold (−3σ)</b> · Rare event — occurs &lt;1% of days · Panic selling exhaustion · High-conviction reversal buy (best combined with strong support)',
  zs_extreme_ob:  '<b>Extreme Overbought (+3σ)</b> · Rare parabolic extension · Very high risk for longs · Consider partial profit-taking',
  zs_reverting:   '<b>Z-Score Reverting</b> · Was extreme (|Z|≥2) last week, now pulling back toward MA · Timing the reversion — price is moving back to fair value',
  rsn_outperform: '<b>RS Outperforming Nifty</b> · Stock/Nifty ratio in uptrend · Fund managers prefer stocks that lead the index · Rising relative strength precedes big moves',
  rsn_new_hi:     '<b>RS at 3-Month High</b> · Stock is accelerating vs Nifty — hitting new RS highs · Most powerful leadership signal · O\'Neil: buy leaders, not laggards',
  rsn_underperform:'<b>RS Underperforming Nifty</b> · Stock/Nifty ratio in downtrend · Avoid or reduce these · Even if price is up, weakness vs index signals institutional selling',
};

function updateT2Info(){
  const s=document.getElementById('t2-strat').value;
  const isRsn=s.startsWith('rsn_');
  const rsNote=isRsn?' <span style="color:var(--warn);font-size:9px">· Requires --nifty flag when generating</span>':'';
  setFbar('<span style="color:var(--acc)">🏆 Tier-2 Strategy</span> · '+(T2_INFO[s]||s)+rsNote);
}

function scanT2(){
  const strat=document.getElementById('t2-strat').value;
  
  const rsMin=parseInt(document.getElementById('t2-rs').value);
  const prMin=parseFloat(document.getElementById('t2-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('t2-pmax').value)||Infinity;
  updateT2Info();
  rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.t2)continue;
    const t=s.t2; const ch=t.ch||{}; const md=t.macd||{}; const zs=t.zs||{}; const rn=t.rsn||{};
    let matched=false; let sig=''; let extra={};

    // Cup & Handle
    if(strat==='ch_breakout') { matched=!!ch.breakout&&!!ch.valid; sig='🏆 C&H Breakout'; extra={lip:ch.left_lip,depth:ch.depth+'%',recovery:ch.recovery+'%',hdl:ch.hdl_rng+'%'}; }
    if(strat==='ch_near')     { matched=!!ch.near_bo&&!!ch.valid; sig='👀 Near C&H BO'; extra={lip:ch.left_lip,depth:ch.depth+'%',pct_from:ch.left_lip?((s.price-ch.left_lip)/ch.left_lip*100).toFixed(1)+'%':'—'}; }
    if(strat==='ch_handle')   { matched=!!ch.in_handle&&!!ch.valid&&!ch.breakout; sig='🔄 In Handle'; extra={lip:ch.left_lip,depth:ch.depth+'%',hdl_rng:ch.hdl_rng+'%'}; }

    // MACD
    if(strat==='macd_bull_div')  { matched=!!md.bull_div;   sig='📗 MACD Bull Div'; extra={macd:md.macd,signal:md.signal,hist:md.hist,trend:md.trend}; }
    if(strat==='macd_bear_div')  { matched=!!md.bear_div;   sig='📕 MACD Bear Div'; extra={macd:md.macd,signal:md.signal,hist:md.hist,trend:md.trend}; }
    if(strat==='macd_flip_bull') { matched=!!md.flip_bull;  sig='🟢 MACD Flip +'; extra={macd:md.macd,hist:md.hist}; }
    if(strat==='macd_flip_bear') { matched=!!md.flip_bear;  sig='🔴 MACD Flip −'; extra={macd:md.macd,hist:md.hist}
    if(strat==='macd_hist_rising')  {matched=!!m2.hist_rising;  sig='📊 MACD H↑';   extra={hist:+(m2.hist||0).toFixed(3)};}
    if(strat==='macd_hist_pos_inc') {matched=!!m2.hist_pos_inc; sig='🟢 MACD H+↑';  extra={hist:+(m2.hist||0).toFixed(3)};}
    if(strat==='macd_hist_neg_inc') {matched=!!m2.hist_neg_inc; sig='⚡ MACD H-↑';  extra={hist:+(m2.hist||0).toFixed(3)};}
    if(strat==='macd_zero_bull')    {matched=!!m2.zero_bull;    sig='🟢 MACD Z↑';   extra={macd:+(m2.macd||0).toFixed(3)};}
    if(strat==='macd_zero_bear')    {matched=!!m2.zero_bear;    sig='🔴 MACD Z↓';   extra={macd:+(m2.macd||0).toFixed(3)};}; }
    if(strat==='macd_bull_trend'){ matched=md.trend==='bull'; sig='📈 MACD Bull'; extra={macd:md.macd,signal:md.signal,hist:md.hist,rising:md.rising?'Yes':'No'}; }

    // Z-Score
    if(strat==='zs_oversold')   { matched=!!zs.oversold&&!zs.extreme_os; sig='🟢 OS −2σ'; extra={z20:zs.z20,z50:zs.z50,pct_b:zs.pct_b,ma20:zs.ma20}; }
    if(strat==='zs_overbought') { matched=!!zs.overbought&&!zs.extreme_ob; sig='🔴 OB +2σ'; extra={z20:zs.z20,z50:zs.z50,pct_b:zs.pct_b,ma20:zs.ma20}; }
    if(strat==='zs_extreme_os') { matched=!!zs.extreme_os; sig='⚡ Extreme OS −3σ'; extra={z20:zs.z20,z50:zs.z50,z200:zs.z200,ma20:zs.ma20}; }
    if(strat==='zs_extreme_ob') { matched=!!zs.extreme_ob; sig='⚡ Extreme OB +3σ'; extra={z20:zs.z20,z50:zs.z50,z200:zs.z200,ma20:zs.ma20}; }
    if(strat==='zs_reverting')  {
      const wasEx=Math.abs(zs.z20||0)>=1&&Math.abs(zs.z20||0)<2;
      matched=wasEx;
      sig=(zs.z20||0)<0?'↗ Reverting Up':'↘ Reverting Dn';
      extra={z20:zs.z20,z50:zs.z50,ma20:zs.ma20};
    }

    // RS vs Nifty
    if(strat==='rsn_outperform')  { matched=rn.rs_trend==='bull'&&!!rn.above_rs_ma; sig='🚀 RS Leader'; extra={trend:rn.rs_trend,new_hi:rn.rs_new_hi?'Yes':'No',chg20:rn.chg20+'%',rs_z:rn.rs_z}; }
    if(strat==='rsn_new_hi')      { matched=!!rn.rs_new_hi; sig='🏅 RS New Hi'; extra={trend:rn.rs_trend,chg20:rn.chg20+'%',rs_z:rn.rs_z}; }
    if(strat==='rsn_underperform'){ matched=rn.rs_trend==='bear'&&!rn.above_rs_ma; sig='📉 RS Laggard'; extra={trend:rn.rs_trend,new_hi:rn.rs_new_hi?'Yes':'No',chg20:rn.chg20+'%'}; }

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig, extra, strat, _tab:'t2'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── TIER-3 STRATEGIES ─────────────────────────────────────────────────────────
const T3_INFO = {
  fib_near_382: '<b>Fibonacci 38.2%</b> · First key retracement — shallow pullback in strong trends · Best entry in a fast-moving stock',
  fib_near_500: '<b>Fibonacci 50%</b> · Mid-point retracement · Widely watched by traders · Strong support/resistance',
  fib_near_618: '<b>Fibonacci 61.8% — Golden Ratio</b> · Most significant Fib level · "The last stand" for bulls/bears · Deepest pullback before trend resumes',
  fib_near_786: '<b>Fibonacci 78.6%</b> · Deep retracement level · Near the start of the move — high risk but high reward if it holds',
  fib_any:      '<b>Near any Fibonacci level</b> · Price within ±2% of any Fib retracement level (23.6%, 38.2%, 50%, 61.8%, 78.6%)',
  ce_bull:      '<b>Chandelier Exit — Bullish</b> · Price above the long stop (Highest Close − 3×ATR) · Trend is up and confirmed · Trail stops using the Chandelier long stop level',
  ce_bear:      '<b>Chandelier Exit — Bearish</b> · Price below short stop (Lowest Close + 3×ATR) · Trend is down · Chandelier short stop acts as overhead resistance',
  ce_flip_bull: '<b>Chandelier Just Flipped Bullish</b> · Fresh signal — Chandelier flipped from bear to bull within last 3 bars · Early trend change entry',
  ce_flip_bear: '<b>Chandelier Just Flipped Bearish</b> · Fresh signal — flipped from bull to bear · Exit longs, consider shorts',
  psar_bull:    '<b>Parabolic SAR — Bullish</b> · Price above the SAR dots · Uptrend in force · SAR level = trailing stop for long positions',
  psar_bear:    '<b>Parabolic SAR — Bearish</b> · Price below SAR dots · Downtrend in force · Avoid longs',
  psar_flip_bull:'<b>PSAR Just Flipped Bullish</b> · SAR dots just moved from above to below price · Trend reversal up signal · Enter long with SAR as stop',
  psar_flip_bear:'<b>PSAR Just Flipped Bearish</b> · SAR dots just moved below to above price · Trend reversal down · Exit longs',
  adx_strong_bull:'<b>ADX Strong Bull</b> · ADX ≥ 25 (strong trend) + +DI above -DI · Trending stocks outperform in this condition · Momentum strategy entry',
  adx_strong_bear:'<b>ADX Strong Bear</b> · ADX ≥ 25 + -DI above +DI · Strong downtrend — avoid longs, wait for ADX to peak and fall',
  adx_di_cross_bull:'<b>ADX DI Bull Cross</b> · +DI just crossed above -DI · Trend is shifting to bullish · Early entry signal — confirm with ADX rising',
  adx_di_cross_bear:'<b>ADX DI Bear Cross</b> · -DI just crossed above +DI · Trend shifting to bearish · Exit signal for longs',
  adx_extreme:  '<b>ADX Extreme ≥ 40</b> · Exceptionally strong trend · Parabolic moves — momentum is intense · Trade with the trend, widen stops',
  stoch_bull_cross:'<b>Stochastic Bull Cross</b> · %K crossed above %D while below 40 · Classic oversold reversal signal · Best when price is at support',
  stoch_bear_cross:'<b>Stochastic Bear Cross</b> · %K crossed below %D while above 60 · Overbought reversal signal · Best at resistance',
  stoch_oversold: '<b>Stochastic Oversold ≤ 20</b> · %K in extreme oversold zone · Combined with support = high-conviction buy',
  stoch_overbought:'<b>Stochastic Overbought ≥ 80</b> · %K in extreme overbought · Combined with resistance = reduce exposure',
  wr_oversold:   '<b>Williams %R Oversold ≤ −80</b> · Extreme oversold — selling exhaustion · Fast oscillator — good for short-term bounce trades',
  wr_overbought: '<b>Williams %R Overbought ≥ −20</b> · Extreme overbought — buying exhaustion · Consider taking profits',
  wr_bull_exit:  '<b>Williams %R Exiting Oversold</b> · Was at −80, now crossing above − momentum turning positive · Entry signal for mean reversion bounce',
  wr_bear_exit:  '<b>Williams %R Exiting Overbought</b> · Was at −20, now crossing below − momentum turning negative · Exit or short signal',
};

function updateT3Info(){
  const s=document.getElementById('t3-strat').value;
  setFbar('<span style="color:var(--gold)">🎖 Tier-3 Strategy</span> · '+(T3_INFO[s]||s));
}

function scanT3(){
  const strat=document.getElementById('t3-strat').value;
  
  const rsMin=parseInt(document.getElementById('t3-rs').value);
  const prMin=parseFloat(document.getElementById('t3-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('t3-pmax').value)||Infinity;
  updateT3Info();
  rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.t3)continue;
    const t=s.t3;
    const fb=t.fib||{}; const ce=t.ce||{}; const ps=t.psar||{};
    const ax=t.adx||{}; const st=t.stoch||{}; const wr=t.wr||{};
    let matched=false; let sig=''; let extra={};

    // Fibonacci
    if(strat==='fib_near_382') { matched=!!fb.near&&fb.nearest==='38.2%'; sig='🔶 Fib 38.2%'; extra={level:fb.f382,dist:fb.dist_pct+'%',swing_h:fb.swing_h,swing_l:fb.swing_l,trend:fb.uptrend?'Up':'Down'}; }
    if(strat==='fib_near_500') { matched=!!fb.near&&fb.nearest==='50%';   sig='🔶 Fib 50%';   extra={level:fb.f500,dist:fb.dist_pct+'%',swing_h:fb.swing_h,swing_l:fb.swing_l}; }
    if(strat==='fib_near_618') { matched=!!fb.near&&fb.nearest==='61.8%'; sig='🔶 Fib 61.8%'; extra={level:fb.f618,dist:fb.dist_pct+'%',swing_h:fb.swing_h,swing_l:fb.swing_l,trend:fb.uptrend?'Up':'Down'}; }
    if(strat==='fib_near_786') { matched=!!fb.near&&fb.nearest==='78.6%'; sig='🔶 Fib 78.6%'; extra={level:fb.f786,dist:fb.dist_pct+'%',swing_h:fb.swing_h,swing_l:fb.swing_l}; }
    if(strat==='fib_any')      { matched=!!fb.near; sig=`🔶 Fib ${fb.nearest||''}`;            extra={level:fb.nearest_val,nearest:fb.nearest,dist:fb.dist_pct+'%',f382:fb.f382,f618:fb.f618}; }

    // Chandelier
    if(strat==='ce_bull')      { matched=!!ce.bull;      sig='🕯 CE Bull';    extra={long_stop:ce.long_stop,atr:ce.atr,dist:ce.long_stop?((s.price-ce.long_stop)/ce.long_stop*100).toFixed(1)+'%':'—'}; }
    if(strat==='ce_bear')      { matched=!!ce.bear;      sig='🕯 CE Bear';    extra={short_stop:ce.short_stop,atr:ce.atr}; }
    if(strat==='ce_flip_bull') { matched=!!ce.flip_bull; sig='🕯 CE Flip ↑'; extra={long_stop:ce.long_stop,atr:ce.atr}; }
    if(strat==='ce_flip_bear') { matched=!!ce.flip_bear; sig='🕯 CE Flip ↓'; extra={short_stop:ce.short_stop,atr:ce.atr}; }

    // Parabolic SAR
    if(strat==='psar_bull')      { matched=ps.direction==='bull'; sig='🔵 PSAR Bull';   extra={sar:ps.sar,dist:ps.dist_pct+'%',af:ps.af}; }
    if(strat==='psar_bear')      { matched=ps.direction==='bear'; sig='🔴 PSAR Bear';   extra={sar:ps.sar,dist:ps.dist_pct+'%',af:ps.af}; }
    if(strat==='psar_flip_bull') { matched=!!ps.flip&&ps.direction==='bull'; sig='⚡ PSAR Flip ↑'; extra={sar:ps.sar,af:ps.af}; }
    if(strat==='psar_flip_bear') { matched=!!ps.flip&&ps.direction==='bear'; sig='⚡ PSAR Flip ↓'; extra={sar:ps.sar,af:ps.af}; }

    // ADX
    if(strat==='adx_strong_bull')  { matched=!!ax.trending&&!!ax.bull_trend; sig='📊 ADX Bull';    extra={adx:ax.adx,di_plus:ax.di_plus,di_minus:ax.di_minus}; }
    if(strat==='adx_strong_bear')  { matched=!!ax.trending&&!ax.bull_trend;  sig='📊 ADX Bear';    extra={adx:ax.adx,di_plus:ax.di_plus,di_minus:ax.di_minus}; }
    if(strat==='adx_di_cross_bull'){ matched=!!ax.di_cross_bull; sig='↗ DI Cross ↑'; extra={adx:ax.adx,di_plus:ax.di_plus,di_minus:ax.di_minus}; }
    if(strat==='adx_di_cross_bear'){ matched=!!ax.di_cross_bear; sig='↘ DI Cross ↓'; extra={adx:ax.adx,di_plus:ax.di_plus,di_minus:ax.di_minus}; }
    if(strat==='adx_extreme')      { matched=!!ax.strong;        sig='💥 ADX Extreme'; extra={adx:ax.adx,di_plus:ax.di_plus,di_minus:ax.di_minus,bull:ax.bull_trend?'Yes':'No'}; }

    // Stochastic
    if(strat==='stoch_bull_cross') { matched=!!st.bull_cross;   sig='📈 Stoch Bull X'; extra={k:st.k,d:st.d}; }
    if(strat==='stoch_bear_cross') { matched=!!st.bear_cross;   sig='📉 Stoch Bear X'; extra={k:st.k,d:st.d}; }
    if(strat==='stoch_oversold')   { matched=!!st.oversold;     sig='🟢 Stoch OS';     extra={k:st.k,d:st.d}; }
    if(strat==='stoch_overbought') { matched=!!st.overbought;   sig='🔴 Stoch OB';     extra={k:st.k,d:st.d}; }

    // Williams %R
    if(strat==='wr_oversold')   { matched=!!wr.oversold;   sig='📗 WR OS';      extra={wr:wr.wr,trend:wr.trend}; }
    if(strat==='wr_overbought') { matched=!!wr.overbought; sig='📕 WR OB';      extra={wr:wr.wr,trend:wr.trend}; }
    if(strat==='wr_bull_exit')  { matched=!!wr.bull_exit;  sig='↗ WR Exit OS'; extra={wr:wr.wr}; }
    if(strat==='wr_bear_exit')  { matched=!!wr.bear_exit;  sig='↘ WR Exit OB'; extra={wr:wr.wr}; }

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig, extra, strat, _tab:'t3'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── HOME PAGE ─────────────────────────────────────────────────────────────────
const HOME_CARDS=[
  // Highest edge — trend following
  {tab:'mi',  strat:null,           emoji:'⚡', badge:'mi',  label:'Minervini + Stage 2',  desc:'All 8 Minervini conditions + Weinstein Stage 2 advancing',         stars:'★★★★★', section:0},
  // RSI>50, GMMA, Donchian, MACD Hist — most important additions
  {tab:'t1',  strat:'rsi_cross50_bull',  emoji:'📊',badge:'t1',  label:'RSI Cross Above 50',   desc:'RSI just crossed above 50 — earliest momentum shift to bull phase. Combines with Golden Cross for high-probability entry',       stars:'★★★★', section:0},
  {tab:'adv', strat:'gmma_cross_bull',   emoji:'🌈',badge:'adv', label:'GMMA Cross Bull',       desc:'All 6 short-term EMAs just crossed above all 6 long-term EMAs — Guppy trend confirmation. Zerodha Kite standard indicator',       stars:'★★★★★',section:0},
  {tab:'adv', strat:'dc55_bull',         emoji:'🐢',badge:'adv', label:'Donchian 55-Day Bull',  desc:'Price close above 55-day upper band — Richard Donchian\'s original Turtle System entry. Highest win-rate breakout system backtest',stars:'★★★★', section:0},
  {tab:'t2',  strat:'macd_hist_neg_inc', emoji:'⚡',badge:'t2',  label:'MACD Histogram Rising', desc:'Histogram negative but increasing — early bull divergence before the crossover. Earlier entry, same setup. Chartink paid filter',  stars:'★★★★', section:1},
  {tab:'t2',  strat:'macd_zero_bull',    emoji:'🟢',badge:'t2',  label:'MACD Zero-Line Cross',  desc:'MACD line crossed above zero — stronger than signal-line cross, confirms trend momentum has genuinely shifted to bull',             stars:'★★★★', section:1},
  {tab:'adv', strat:'dc20_bull',         emoji:'📐',badge:'adv', label:'Donchian 20-Day Breakout',desc:'Close above 20-day Donchian upper band — Chartink\'s most-used breakout filter. New 20-day high close = momentum confirmed',    stars:'★★★★', section:1},
  {tab:'adv', strat:'gmma_all_bull',     emoji:'🌈',badge:'adv', label:'GMMA All-Bull',         desc:'All 6 short-term EMAs above all 6 long-term EMAs — perfect alignment, Daryl Guppy\'s strongest bull signal. TradePoint standard', stars:'★★★★★',section:1},
  {tab:'t1',  strat:'rsi_above50',       emoji:'📊',badge:'t1',  label:'RSI Above 50',          desc:'RSI in bull momentum zone — simplest trend filter. Combine with any other signal to eliminate bear-phase noise',                  stars:'★★★',  section:1},
  {tab:'adv', strat:'dc_squeeze',        emoji:'🗜',badge:'adv', label:'Donchian Squeeze',       desc:'Donchian bandwidth at 6-month low — channel contracting to minimum before inevitable expansion. Turtle trading pre-breakout setup', stars:'★★★★', section:1},
  {tab:'t1',  strat:'gc_fresh_golden',emoji:'🟡',badge:'t1', label:'Golden Cross (Fresh)',  desc:'50 SMA just crossed above 200 SMA — most-watched institutional signal',stars:'★★★★★', section:0},
  {tab:'t1',  strat:'rsi_bull_div',  emoji:'📗', badge:'t1', label:'RSI Bull Divergence',   desc:'Price lower low + RSI higher low = hidden institutional accumulation',  stars:'★★★★★', section:0},
  {tab:'t1',  strat:'w52_breakout',  emoji:'🚀', badge:'t1', label:'52W High + Volume',     desc:'New 52-week high with volume surge — most winners break out this way',    stars:'★★★★★', section:0},
  {tab:'t1',  strat:'bb_squeeze_bull',emoji:'🔵',badge:'t1', label:'BB Squeeze Bull',       desc:'Bollinger bands at tightest 25% + bullish momentum = explosion up',       stars:'★★★★', section:0},
  {tab:'t2',  strat:'ch_breakout',   emoji:'🏆', badge:'t2', label:'Cup & Handle Breakout', desc:"O'Neil's U-shaped base + tight handle + price breaks lip with volume",     stars:'★★★★', section:0},
  {tab:'t1',  strat:'nr7_inside',    emoji:'🎯', badge:'t1', label:'NR7 + Inside Bar',      desc:'Double compression — narrowest range in 7 bars AND inside prev bar',       stars:'★★★★', section:0},
  {tab:'adv', strat:'vcp',           emoji:'📐', badge:'adv', label:'VCP Pattern',          desc:'Volatility contraction with declining volume — Minervini setup',           stars:'★★★★', section:0},
  // Momentum / timing
  {tab:'t3',  strat:'psar_flip_bull',emoji:'💠', badge:'t3', label:'PSAR Flip Bullish',     desc:'Parabolic SAR just flipped from bear to bull — early trend reversal entry', stars:'★★★★', section:1},
  {tab:'t2',  strat:'macd_flip_bull',emoji:'🟢', badge:'t2', label:'MACD Histogram Flip +', desc:'Histogram just crossed above zero — early trend change confirmed',          stars:'★★★★', section:1},
  {tab:'t3',  strat:'adx_di_cross_bull',emoji:'↗️',badge:'t3',label:'ADX DI Bull Cross',   desc:'+DI crossed above -DI with trend strength — institutional momentum',       stars:'★★★★', section:1},
  {tab:'adv', strat:'wyckoff_markup',emoji:'📈', badge:'adv', label:'Wyckoff Markup',       desc:'Advancing phase — institutions completed accumulation, mark-up begins',      stars:'★★★★', section:1},
  // Mean reversion / reversal
  {tab:'t2',  strat:'zs_extreme_os', emoji:'🔮', badge:'t2', label:'Z-Score −3σ (Extreme)', desc:'Price 3 std devs below 20-day MA — rare panic sell, high-conviction bounce', stars:'★★★★', section:2},
  {tab:'t1',  strat:'rsi_oversold',  emoji:'🟩', badge:'t1', label:'RSI Oversold ≤ 30',     desc:'Extreme selling exhaustion — best at support or pivot level',                stars:'★★★', section:2},
  {tab:'t3',  strat:'stoch_bull_cross',emoji:'📊',badge:'t3',label:'Stochastic Bull Cross', desc:'%K crossed above %D in oversold zone — momentum turning up',                stars:'★★★', section:2},
  {tab:'t3',  strat:'wr_bull_exit',  emoji:'↗', badge:'t3', label:'Williams %R Exit OS',   desc:'Was at extreme oversold −80, now recovering — timing mean reversion',       stars:'★★★', section:2},
  // 🏅 India Pro — original 6 cards
  {tab:'ti', strat:'candle_bull_eng',    emoji:'📗', badge:'ti', label:'Bullish Engulfing',      desc:'Current bull body engulfs prior bear — institutional buyers overwhelm sellers', stars:'★★★★', section:3},
  {tab:'ti', strat:'candle_morning_star',emoji:'🌅', badge:'ti', label:'Morning Star',            desc:"3-bar reversal: bear→star→bull — O'Neil's most reliable candlestick bottom",   stars:'★★★★', section:3},
  {tab:'ti', strat:'ema_fan_bull',       emoji:'🔥', badge:'ti', label:'EMA Fan Bullish',         desc:'Price > EMA9 > EMA21 > EMA50 — Chartink momentum leader alignment',            stars:'★★★★', section:3},
  {tab:'ti', strat:'ma_all_bull',        emoji:'🏆', badge:'ti', label:'Full MA Stack',           desc:'MA20>MA50>MA100>MA200 — StockEdge maximum bullish alignment',                  stars:'★★★★', section:3},
  {tab:'ti', strat:'consol_10',          emoji:'🎯', badge:'ti', label:'52W Tight Coil',          desc:'Within 5% of 52W high, 10-day range <5% — pre-breakout coil',                 stars:'★★★★', section:3},
  {tab:'ti', strat:'vol_acc3',           emoji:'📦', badge:'ti', label:'3-Day Accumulation',      desc:'3 up-days with rising volume — StockEdge institutional accumulation',          stars:'★★★',  section:3},
  // 🏅 India Pro — new indicator cards
  {tab:'ti', strat:'ha_strong_bull',     emoji:'🕯', badge:'ti', label:'HA Strong Bull',          desc:'Heikin Ashi bull candle with no lower shadow — pure uninterrupted momentum',   stars:'★★★★', section:3},
  {tab:'ti', strat:'hma_flip_bull',      emoji:'⚡', badge:'ti', label:'HMA Flipped Bullish',     desc:'Hull MA slope just changed from down to up — early trend reversal with minimum lag', stars:'★★★★', section:3},
  {tab:'ti', strat:'kelt_cross_up',      emoji:'📏', badge:'ti', label:'Keltner Breakout Up',     desc:'Price just crossed above EMA20+2×ATR — volatility expansion beginning',        stars:'★★★',  section:3},
  {tab:'ti', strat:'mfi_os',             emoji:'💧', badge:'ti', label:'MFI Oversold',            desc:'Money Flow Index ≤20 — volume-confirmed selling exhaustion, deeper signal than RSI', stars:'★★★★', section:3},
  // 🎨 Patterns
  {tab:'patt', strat:'harm_gartley_bull',emoji:'🎵', badge:'patt', label:'Gartley Bullish',       desc:'AB/XA≈0.618, AD/XA≈0.786 — most common harmonic, highest win-rate at PRZ',     stars:'★★★★', section:4},
  {tab:'patt', strat:'harm_bat_bull',    emoji:'🦇', badge:'patt', label:'Bat Bullish',           desc:'AB/XA=0.382-0.5, AD/XA≈0.886 — tightest PRZ, best risk:reward of all harmonics', stars:'★★★★', section:4},
  {tab:'patt', strat:'cp_inv_hs',        emoji:'⛰', badge:'patt', label:'Inverse H&S',           desc:'3 troughs, middle deepest — one of the most reliable bullish reversal patterns', stars:'★★★★', section:4},
  {tab:'patt', strat:'cp_bull_flag',     emoji:'🚩', badge:'patt', label:'Bull Flag',             desc:'Strong pole + tight consolidation — momentum continuation with best R:R',       stars:'★★★★', section:4},
  {tab:'patt', strat:'cp_fall_wedge',    emoji:'📐', badge:'patt', label:'Falling Wedge',        desc:'Converging downward trendlines — deceptive bearish look, actually bullish reversal', stars:'★★★', section:4},
  {tab:'patt', strat:'harm_abcd_bull',   emoji:'〰', badge:'patt', label:'AB=CD Bullish',        desc:'CD equals AB in price — the base harmonic present in all other patterns',       stars:'★★★',  section:4},
  // 📡 Market Breadth
  {tab:'br', strat:'all',               emoji:'📡', badge:'br',   label:'Market Health Dashboard',desc:'Universe-wide breadth: % above MA200, Stage 2, RS leaders — market phase gauge', stars:'★★★★★',section:5},
  {tab:'br', strat:'stage2',            emoji:'📈', badge:'br',   label:'All Stage 2 Stocks',     desc:'Weinstein Stage 2 confirmed across the full universe — the only stocks to be long', stars:'★★★★', section:5},
  {tab:'br', strat:'near52wh',          emoji:'🎯', badge:'br',   label:'Near 52W High',          desc:'Stocks within 5% of annual high — potential breakout candidates universe-wide',  stars:'★★★',  section:5},
  {tab:'br', strat:'rs80',              emoji:'🌟', badge:'br',   label:'RS ≥ 80 Leaders',        desc:'Top 20% relative strength — the leaders institutions are accumulating right now', stars:'★★★★', section:5},
  {tab:'patt',strat:'dbl_bot_w_bo',  emoji:'〔W〕',badge:'patt',label:'Double Bottom Weekly BO', desc:'Weekly W pattern confirmed - price closed above neckline. Multi-month base breakout, highest-conviction swing/position entry', stars:'★★★★★',section:4},
  {tab:'patt',strat:'dbl_top_w_bo',  emoji:'〔M〕',badge:'patt',label:'Double Top Weekly BD',     desc:'Weekly M pattern confirmed - price closed below neckline. Multi-month topping process, high-conviction bearish signal', stars:'★★★★', section:4},
  {tab:'patt',strat:'dbl_bot_d_bo',  emoji:'〔W〕',badge:'patt',label:'Double Bottom Daily BO',   desc:'Daily W pattern confirmed with neckline breakout - validated short-term reversal entry', stars:'★★★★', section:4},
  {tab:'patt',strat:'dbl_bot_m_bo',  emoji:'〔W〕',badge:'patt',label:'Double Bottom Monthly BO', desc:'Monthly W pattern confirmed - rare multi-year base breakout, holding period often measured in years', stars:'★★★★★',section:4},
  // 🏹 Breakout Pro
  {tab:'bp',strat:'vcp',     emoji:'🔄',badge:'bp',label:'VCP (Minervini)',     desc:'Volatility Contraction Pattern — 2+ weekly contractions + vol dry-up near 52W high. MarketSmith #1',stars:'★★★★★',section:6},
  {tab:'bp',strat:'pp',      emoji:'🎯',badge:'bp',label:'Pocket Pivot',        desc:'Up-day volume > highest down-day vol in 10 sessions. Gil Morales early-entry before breakout',       stars:'★★★★★',section:6},
  {tab:'bp',strat:'twt3',    emoji:'📏',badge:'bp',label:'3-Week Tight',        desc:"O\'Neil: weekly closes within ±1.5% for 3 weeks — institutional absorption before breakout",          stars:'★★★★', section:6},
  {tab:'bp',strat:'rsh',     emoji:'📈',badge:'bp',label:'RS New 52W High',     desc:'RS line (stock/Nifty) at 52W peak — outperforming market. IBD #1 leading indicator',                stars:'★★★★★',section:6},
  {tab:'bp',strat:'ads_ab',  emoji:'🏆',badge:'bp',label:'A/D Accumulation',   desc:'IBD A/D rating A or B — institutional accumulation over 13 weeks. StockEdge premium equivalent',    stars:'★★★★', section:6},
  {tab:'bp',strat:'hvb',     emoji:'⚡',badge:'bp',label:'High Volume BO',      desc:'52W high close + 2× avg volume — institutions confirm the breakout. Volume removes ambiguity',       stars:'★★★★', section:6},
  {tab:'bp',strat:'vdu_bull',emoji:'💧',badge:'bp',label:'Volume Dry-Up',       desc:'Volume < 50% of avg while price holds — supply exhausted, coiled for expansion. Chartink paid filter',stars:'★★★★',section:6},
  {tab:'bp',strat:'ems_power',emoji:'💪',badge:'bp',label:'EMA Power Stack',   desc:'Price > 9 > 21 > 50 > 200 EMA, all rising. Perfect bull alignment — TradePoint semi-algo signal',   stars:'★★★★', section:6},
  {tab:'bp',strat:'iw1',     emoji:'📅',badge:'bp',label:'Inside Week',         desc:'Weekly range inside prior week — weekly compression before directional expansion move',              stars:'★★★',  section:6},
  {tab:'bp',strat:'tlb',     emoji:'📐',badge:'bp',label:'Trendline Breakout',  desc:'Price closes above descending trendline across 3 peaks — bearish structure broken',                  stars:'★★★',  section:6},
  // 🔢 Smart Rank
  {tab:'sr',strat:'ms_90',      emoji:'🔢',badge:'sr',label:'Momentum Score ≥90', desc:'Top 10% of universe by composite momentum: 1M+3M+6M+12M returns weighted + RS rank. Trendlyne Momentum Score equivalent',stars:'★★★★★',section:7},
  {tab:'sr',strat:'ms_75',      emoji:'📊',badge:'sr',label:'Momentum Score ≥75', desc:'Top quartile of momentum universe — sustained multi-period outperformance vs all NSE stocks',               stars:'★★★★', section:7},
  {tab:'sr',strat:'season_bull',emoji:'📅',badge:'sr',label:'Seasonality Bullish', desc:'Current month historically positive for this stock (avg > 2%, 5+ years of data). Definedge Seasonality equivalent', stars:'★★★★',section:7},
  {tab:'sr',strat:'season_strong',emoji:'📅',badge:'sr',label:'Seasonality Strong', desc:'Current month avg > 3% + hit rate > 65% — most reliable seasonal setup. Only available on Definedge (\u20B92,499/mo)',stars:'★★★★★',section:7},
  {tab:'sr',strat:'mtf_perfect',emoji:'💯',badge:'sr',label:'MTF Perfect (15/15)', desc:'All daily + weekly + monthly trend conditions bullish simultaneously. Definedge Matrix equivalent — highest conviction',stars:'★★★★★',section:7},
  {tab:'sr',strat:'mtf_strong', emoji:'💪',badge:'sr',label:'MTF Strong (12+/15)', desc:'12+ of 15 multi-timeframe checks pass. Strong structural alignment across all 3 timeframes',               stars:'★★★★', section:7},
  {tab:'sr',strat:'ath',        emoji:'🏔',badge:'sr',label:'All-Time High',        desc:'New all-time high close — ATH breakouts with volume have the highest 6-month forward returns of any pattern',  stars:'★★★★★',section:7},
  {tab:'sr',strat:'tc3',        emoji:'🗜',badge:'sr',label:'Triple Compression',   desc:'NR7 + Volume Dry-Up + BB Squeeze all firing simultaneously — three-dimensional maximum coiling',            stars:'★★★★★',section:7},
  {tab:'sr',strat:'mtfrs_90',   emoji:'🌟',badge:'sr',label:'Consistent RS Leader', desc:'Top 10% by 1M, 3M, 6M AND 12M returns simultaneously. Definedge Smart RS Scanner equivalent',             stars:'★★★★★',section:7},
  {tab:'sr',strat:'tc2',        emoji:'🗜',badge:'sr',label:'Double Compression',   desc:'Any 2 of NR7/VDU/BB Squeeze — strong pre-breakout coiling signal',                                         stars:'★★★★', section:7},
  // 🎯 Swing Setup
  {tab:'swing',strat:'ema_sq_d_bull',  emoji:'🗜',badge:'swing',label:'Daily EMA Squeeze + Bull',  desc:'10/20/50/100/150/200 EMA all within 2.5%% AND bullishly stacked — coiling inside an uptrend, highest-probability continuation', stars:'★★★★★',section:8},
  {tab:'swing',strat:'macd535_d_bull', emoji:'📊',badge:'swing',label:'MACD 5/35 Bull Cross (D)', desc:'Fast 5-EMA crossed above slow 35-EMA on daily — faster swing-entry variant of standard MACD, catches moves 1-2 days earlier', stars:'★★★★', section:8},
  {tab:'swing',strat:'macd535_w_bull', emoji:'📊',badge:'swing',label:'MACD 5/35 Bull Cross (W)', desc:'Weekly MACD 5/35 bullish cross — signals a multi-week swing move beginning, ideal for 1-3 month holding periods', stars:'★★★★★',section:8},
  {tab:'swing',strat:'macd535_align',  emoji:'📊',badge:'swing',label:'D+W+M MACD 5/35 Aligned', desc:'Daily, Weekly AND Monthly MACD(5,35,5) all bullish simultaneously — full timeframe alignment, the strongest momentum confirmation', stars:'★★★★★',section:8},
  {tab:'swing',strat:'ema_sq_w',       emoji:'🗜',badge:'swing',label:'Weekly EMA Squeeze',       desc:'10/20/50/100-week EMAs converged within 2.5%% — multi-month consolidation resolving, breakouts tend to run for weeks', stars:'★★★★', section:8},
  {tab:'swing',strat:'swing_strong',   emoji:'🎯',badge:'swing',label:'Strong Swing Buy',          desc:'Score ≥75 with confluence across 4+ categories — highest-conviction swing setups with full Entry/Stop/Target plan for 1W-3M', stars:'★★★★★',section:8},
  {tab:'swing',strat:'swing_breakout_ready',emoji:'🎯',badge:'swing',label:'Breakout-Ready Setup', desc:'EMA squeeze (coiled) + fresh MACD 5/35 bull cross (trigger firing) — the textbook compression + trigger breakout entry', stars:'★★★★★',section:8},
  {tab:'swing',strat:'ema_sq_d_tight', emoji:'🗜',badge:'swing',label:'Ultra-Tight EMA Squeeze',   desc:'All 6 EMAs (10-200) within 1%% of each other — extremely rare, historically precedes the largest volatility expansions', stars:'★★★★★',section:8},
];

const SECTIONS=[
  '⭐ HIGHEST EDGE — TREND FOLLOWING',
  '🔥 MOMENTUM & TIMING SIGNALS',
  '🔄 MEAN REVERSION & REVERSAL',
  '🏅 INDIA PRO — CHARTINK · STOCKEDGE · TRENDLYNE',
  '🎨 PATTERNS — HARMONIC & CHART PATTERNS',
  '📡 MARKET BREADTH & UNIVERSE SIGNALS',
  '🏹 BREAKOUT PRO — PAID SCANNER STRATEGIES',
  '🔢 SMART RANK — AI COMPOSITE SCORING',
  '🎯 SWING SETUP — EMA SQUEEZE · MACD 5/35 · TRADE PLANS',
];
const BADGES={t1:'Tier-1',t2:'Tier-2',t3:'Tier-3',mi:'Multi-Ind',adv:'Advanced',piv:'Pivot',
              ti:'India Pro',nl:'Noiseless',xp:'Experimental',patt:'Patterns',br:'Breadth',gap:'Gaps',combo:'Combo',bp:'Breakout Pro',sr:'Smart Rank',swing:'Swing Setup'};

function renderHome(){
  let html=`<div class="home-wrap">`;
  for(let si=0;si<SECTIONS.length;si++){
    const cards=HOME_CARDS.filter(c=>c.section===si);
    if(!cards.length) continue;
    html+=`<div class="home-title">${SECTIONS[si]}</div><div class="home-grid">`;
    cards.forEach(c=>{
      html+=`<div class="hcard" onclick="launchScanner('${c.tab}','${c.strat||''}')">
        <div class="hcard-top"><span class="hcard-emoji">${c.emoji}</span>
          <span class="hcard-badge badge-${c.badge}">${BADGES[c.badge]||c.badge}</span></div>
        <div class="hcard-name">${c.label}</div>
        <div class="hcard-desc">${c.desc}</div>
        <div class="hcard-stars">${c.stars}</div>
        <div class="hcard-btn">▶ Launch</div>
      </div>`;
    });
    html+=`</div>`;
  }
  html+=`<div style="font-family:var(--mono);font-size:9px;color:var(--mu);margin-top:12px;padding-top:8px;border-top:1px solid var(--brd)">
    ★★★★★ = Highest probability · ★★★★ = Strong edge · ★★★ = Good confluence tool · 
    All strategies computable from daily OHLC data · Success rates based on professional backtesting research
  </div></div>`;
  return html;
}

function launchScanner(tab, strat){
  // Map tab → its strategy dropdown id
  const stratMaps={
    piv:'piv-type', adv:'adv-strat', t1:'t1-strat', t2:'t2-strat', t3:'t3-strat',
    ti:'ti-strat', nl:'nl-strat', xp:'xp-strat', patt:'patt-strat', br:'br-filter',
  };
  if(strat && stratMaps[tab]){
    const el=document.getElementById(stratMaps[tab]);
    if(el) el.value=strat;
  }
  if(tab==='mi'){
    const ms=document.getElementById('mi-strat'); if(ms) ms.value='both';
    const sc=document.getElementById('mi-score'); if(sc) sc.value='7';
  }
  switchTab(tab);
}

// ── HELP PAGE ─────────────────────────────────────────────────────────────────
// ── AI links — open chat + copy prompt to clipboard ───────────────────────
const AI_URLS={
  claude:'https://claude.ai/new',
  chatgpt:'https://chatgpt.com/',
  copilot:'https://copilot.microsoft.com/',
  gemini:'https://gemini.google.com/app',
};
function _copyText(text){
  // Works on both https:// AND file:// — modern API first, execCommand fallback
  if(navigator.clipboard&&window.isSecureContext){
    return navigator.clipboard.writeText(text).then(()=>true).catch(()=>_execCopy(text));
  }
  return Promise.resolve(_execCopy(text));
}
function _execCopy(text){
  const ta=document.createElement('textarea');
  ta.value=text; ta.style.cssText='position:fixed;left:-9999px;top:-9999px;opacity:0';
  document.body.appendChild(ta); ta.focus(); ta.select();
  try{ const ok=document.execCommand('copy'); document.body.removeChild(ta); return ok; }
  catch(e){ document.body.removeChild(ta); return false; }
}
function _showAIModal(prompt,serviceName,url){
  document.getElementById('_ai_modal')?.remove();
  const safe=prompt.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const el=document.createElement('div'); el.id='_ai_modal';
  el.style.cssText='position:fixed;inset:0;background:rgba(0,0,0,.78);z-index:9999;display:flex;align-items:center;justify-content:center;padding:16px';
  el.innerHTML=`<div style="background:var(--bg);border:1px solid var(--brd);border-radius:12px;padding:22px;max-width:640px;width:100%;max-height:82vh;display:flex;flex-direction:column;gap:12px;box-shadow:0 20px 60px rgba(0,0,0,.6)">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <span style="font-size:13px;font-weight:500;color:var(--txt)">◈ Prompt for ${serviceName}</span>
      <button onclick="document.getElementById('_ai_modal').remove()" style="background:none;border:none;color:var(--mu);font-size:20px;cursor:pointer;line-height:1">×</button>
    </div>
    <div style="font-size:10px;color:var(--mu);font-family:var(--mono)">Select all → Copy → Paste into ${serviceName}</div>
    <textarea id="_ai_ta" readonly style="flex:1;min-height:160px;max-height:300px;background:var(--surf);border:1px solid var(--brd);color:var(--txt);padding:12px;border-radius:8px;font-size:11px;line-height:1.6;resize:vertical">${safe}</textarea>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <button id="_ai_copy" onclick="(()=>{const t=document.getElementById('_ai_ta');t.select();document.execCommand('copy');const b=document.getElementById('_ai_copy');b.textContent='✓ Copied!';b.style.background='#22c55e';b.style.color='#000';setTimeout(()=>{b.textContent='Copy Prompt';b.style.background='';b.style.color='';},2200)})()"
        style="padding:8px 20px;background:var(--acc);color:#000;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:600">Copy Prompt</button>
      <a href="${url}" target="_blank" rel="noopener" style="padding:8px 18px;background:var(--surf);color:var(--txt);border:1px solid var(--brd);border-radius:6px;font-size:12px;text-decoration:none;display:flex;align-items:center;gap:5px">Open ${serviceName} ↗</a>
      <span style="margin-left:auto;font-size:10px;color:var(--mu);align-self:center">Ctrl+A → Ctrl+C if button fails</span>
    </div>
  </div>`;
  document.body.appendChild(el);
  setTimeout(()=>{const ta=document.getElementById('_ai_ta');if(ta){ta.select();ta.focus();}},80);
}
function openAI(service,b64prompt){
  const prompt=decodeURIComponent(escape(atob(b64prompt)));
  const svcName={claude:'Claude',chatgpt:'ChatGPT',copilot:'Copilot',gemini:'Gemini'}[service]||service;
  _copyText(prompt).then(ok=>{
    if(ok!==false){
      showToast(`✓ Prompt copied — opening ${svcName}. Paste with Ctrl+V`);
      setTimeout(()=>window.open(AI_URLS[service],'_blank'),450);
    } else {
      _showAIModal(prompt,svcName,AI_URLS[service]||AI_URLS.claude);
    }
  });
}
function showToast(msg){
  const t=document.getElementById('_toast');
  if(!t)return;
  t.textContent=msg; t.classList.add('show');
  clearTimeout(t._tmr); t._tmr=setTimeout(()=>t.classList.remove('show'),3200);
}
function aiLinks(topic,prompt){
  const b64=btoa(unescape(encodeURIComponent(prompt)));
  return `<div class="help-ai">
    <span style="font-size:9px;color:var(--mu);font-family:var(--mono);letter-spacing:1px">ASK AI ›</span>
    <button class="ai-btn ai-claude"  onclick="openAI('claude','${b64}')">◈ Claude</button>
    <button class="ai-btn ai-chatgpt" onclick="openAI('chatgpt','${b64}')">✦ ChatGPT</button>
    <button class="ai-btn ai-copilot" onclick="openAI('copilot','${b64}')">⊞ Copilot</button>
    <button class="ai-btn ai-gemini"  onclick="openAI('gemini','${b64}')">✦ Gemini</button>
  </div>`;
}
function refLinks(links){
  return `<div class="help-refs">${links.map(([l,u])=>`<a href="${u}" target="_blank" class="ref-link">${l}</a>`).join('')}</div>`;
}

// ── HELP POPUP ─────────────────────────────────────────────────────────────
const STRAT_HELP_KEY={
  piv_type:{fibonacci:0,traditional:1,camarilla:2,woodie:3,dm:4,floor:1,classic:1},
  smc:{bull_ob:0,bear_ob:0,bull_fvg:1,bear_fvg:1,bos_bull:2,bos_bear:2,choch_bull:2,choch_bear:2,all:0},
  vol:{near_poc:0,above_vah:0,below_val:0,vol_spike:1,obv_div_bull:1,obv_div_bear:1,ad_up:1,all:0},
  mi:{both:0,minervini:0,weinstein:1,any:0},
  adv:{darvas:0,vcp:1,wyckoff_acc:2,wyckoff_markup:2,wyckoff_spring:2,turtle20:3,turtle55:3,ichi_above:4,ichi_tk_bull:4,ichi_kbo:4,td_buy9:5,td_sell9:5,st_up:5,st_flip_up:5,elder_buy:5,elder_sell:5},
  t1:{gc_fresh_golden:0,gc_fresh_death:0,gc_bull_zone:0,gc_near_golden:0,above_50sma:0,below_50sma:0,above_200sma:0,below_200sma:0,rsi_bull_div:1,rsi_bear_div:1,rsi_oversold:1,rsi_overbought:1,bb_squeeze_bull:2,bb_squeeze_bear:2,bb_squeeze_any:2,bb_cross_upper:2,bb_cross_lower:2,bb_above_upper:2,bb_below_lower:2,w52_breakout:3,w52_close_above:3,w52_near:3,w52_consol_near:3,h30_bo:3,h90_bo:3,h30_near:3,nr7_bull:4,nr7_bear:4,nr4:4,inside_bar:4,nr7_inside:4,st_bull:5,st_bear:5,st_flip_bull:5,st_flip_bear:5,st_consec5:5,st_tight:5},
  t2:{ch_breakout:0,ch_near:0,ch_handle:0,macd_bull_div:1,macd_bear_div:1,macd_flip_bull:1,macd_flip_bear:1,macd_bull_trend:1,zs_oversold:2,zs_overbought:2,zs_extreme_os:2,zs_extreme_ob:2,zs_reverting:2,rsn_outperform:2,rsn_new_hi:2,rsn_underperform:2},
  t3:{fib_near_382:0,fib_near_500:0,fib_near_618:0,fib_near_786:0,fib_any:0,ce_bull:0,ce_bear:0,ce_flip_bull:0,ce_flip_bear:0,psar_bull:0,psar_bear:0,psar_flip_bull:0,psar_flip_bear:0,adx_strong_bull:1,adx_strong_bear:1,adx_di_cross_bull:1,adx_di_cross_bear:1,adx_extreme:1,stoch_bull_cross:2,stoch_bear_cross:2,stoch_oversold:2,stoch_overbought:2,wr_oversold:3,wr_overbought:3,wr_bull_exit:3,wr_bear_exit:3},
  ti:{
    candle_hammer:0,candle_inv_hammer:0,candle_bull_eng:0,candle_morning_star:0,candle_piercing:0,candle_bull_har:0,candle_tws:0,
    candle_doji:0,candle_shooting_star:0,candle_bear_eng:0,candle_dark_cloud:0,candle_evening_star:0,candle_bear_har:0,candle_tbc:0,
    candle_inside_bar:0,candle_inside_bar_bull:0,candle_inside_bar_bear:0,candle_double_inside:0,
    candle_marubozu_bull:0,candle_marubozu_bear:0,candle_spinning_top:0,
    candle_tweezer_bottom:0,candle_tweezer_top:0,
    candle_outside_bull:0,candle_outside_bear:0,
    candle_gap_up:0,candle_gap_up3:0,candle_gap_down:0,candle_gap_down3:0,
    ema_c921_bull:1,ema_c921_bear:1,ema_c2050_bull:1,ema_c2050_bear:1,ema_fan_bull:1,ema_fan_bear:1,ema_recent_921:1,
    mom_high:2,mom_positive_all:2,mom_roc10_bull:2,mom_roc21_bull:2,mom_neg:2,
    seq_hh3:3,seq_hh5:3,seq_hl3:3,seq_hc3_vol:3,seq_ll3:3,
    vol_acc3:4,vol_acc5:4,vol_quiet:4,
    consol_10:5,consol_15:5,
    ma_max4:6,ma_all_bull:6,ma_above3:6,
    // New India Pro indicators (section 9)
    ha_bull:7,ha_bear:7,ha_strong_bull:7,ha_strong_bear:7,ha_doji:7,ha_rev_bull:7,ha_rev_bear:7,ha_consec5:7,
    hma_bull:8,hma_bear:8,hma_flip_bull:8,hma_flip_bear:8,
    kelt_above:9,kelt_below:9,kelt_cross_up:9,kelt_cross_dn:9,kelt_squeeze:9,
    mfi_os:10,mfi_ob:10,mfi_bull_div:10,mfi_bear_div:10,
    cci_os:11,cci_ob:11,cci_zero_up:11,cci_zero_dn:11,cci_os_exit:11,cci_ob_exit:11,
    lr_above_chan:12,lr_below_chan:12,lr_slope_bull:12,lr_slope_bear:12,lr_mean_rev:12,
  },
  nl:{
    pnf_dbl_top:0,pnf_dbl_bot:0,pnf_tpl_top:0,pnf_tpl_bot:0,pnf_bull_cat:0,pnf_bear_cat:0,
    pnf_high_pole:0,pnf_low_pole:0,pnf_asc_tri:0,pnf_desc_tri:0,pnf_lt_bull:0,pnf_lt_bear:0,pnf_in_x:0,pnf_in_o:0,
    renko_bull:1,renko_bear:1,renko_rev_bull:1,renko_rev_bear:1,renko_3:1,renko_5:1,renko_dbl_top:1,renko_dbl_bot:1,
    lb3_bull:2,lb3_bear:2,lb3_rev_bull:2,lb3_rev_bear:2,lb3_consec:2,
    rrg_leading:3,rrg_weakening:3,rrg_improving:3,rrg_lagging:3,
    ew_w5_up:4,ew_w5_dn:4,ew_w3_ext:4,ew_abc:4,
    kagi_yang:5,kagi_yin:5,kagi_rev_yang:5,kagi_rev_yin:5,kagi_shoulder:5,kagi_waist:5,kagi_consec:5,
  },
  xp:{
    vsa_big_bull:0,vsa_big_bear:0,vsa_big_indecision:0,
    vsa_effort_up:1,vsa_effort_dn:1,
    vsa_stopping:1,vsa_climax_buy:1,vsa_climax_sell:1,
    vsa_no_demand:1,vsa_no_supply:1,vsa_test_supply:1,
    vsa_upthrust:1,vsa_spring:1,vsa_pseudo_ut:1,vsa_trap:1,
  },
  patt:{
    harm_gartley_bull:0,harm_gartley_bear:0,harm_bat_bull:0,harm_bat_bear:0,
    harm_butterfly_bull:0,harm_butterfly_bear:0,harm_crab_bull:0,harm_crab_bear:0,
    harm_cypher_bull:0,harm_cypher_bear:0,harm_abcd_bull:0,harm_abcd_bear:0,
    cp_hs:1,cp_inv_hs:1,cp_dbl_top:1,cp_dbl_bot:1,
    cp_rise_wedge:1,cp_fall_wedge:1,cp_bull_flag:1,cp_bear_flag:1,
    dbl_bot_d:2,dbl_bot_d_bo:2,dbl_bot_w:2,dbl_bot_w_bo:2,dbl_bot_m:2,dbl_bot_m_bo:2,
    dbl_top_d:3,dbl_top_d_bo:3,dbl_top_w:3,dbl_top_w_bo:3,dbl_top_m:3,dbl_top_m_bo:3,
  },
  br:{all:0,above200:0,stage2:0,near52wh:0,pnf_x:0,renko_bull:0,ha_bull:0,ema_fan:0,rs80:0},
  gap:{up1:0,up2:0,up3:0,up_cont:0,dn1:0,dn_cont:0},
  combo:{},
  bp:{vcp:0,vcp3:0,pp:0,pp_strong:0,twt3:0,rsh:0,ads_ab:0,hvb:0,vdu_bull:0,ems_power:0,iw1:0,tlb:0},
  sr:{ms_90:0,ms_75:0,ms_50:0,season_bull:0,season_strong:0,mtf_perfect:0,mtf_strong:0,ath:0,tc3:0,mtfrs_90:0},
  swing:{ema_sq_d:0,ema_sq_d_tight:0,ema_sq_w:0,ema_sq_m:0,ema_sq_d_bull:0,ema_sq_w_bull:0,
    macd535_d_bull:1,macd535_d_bear:1,macd535_w_bull:1,macd535_w_bear:1,macd535_m_bull:1,macd535_m_bear:1,macd535_d_zero_bull:1,macd535_align:1,
    swing_buy:2,swing_strong:2,swing_breakout_ready:2},
};
// Map tab → section index in Help H array
const TAB_SEC={
  piv_type:0,smc:1,vol:2,mi:3,adv:4,t1:5,t2:6,t3:7,
  ti:8,   // India Pro (original) — new indicators use item indices 7-12 within section 8
  nl:10,  // Noiseless
  xp:11,  // Experimental / VSA
  patt:12,// Patterns
  br:13,  // Market Breadth
  gap:14, combo:14, bp:15, sr:16, swing:8,
};

function showHelp(tab,stratVal){
  // Trigger renderHelp once to ensure H is in scope — store it on window
  if(!window._HS){ const dummy=renderHelp(); } // renderHelp sets window._HS
  const secIdx=TAB_SEC[tab];
  const itemMap=STRAT_HELP_KEY[tab]||{};
  const itemIdx=(itemMap[stratVal]!==undefined)?itemMap[stratVal]:0;
  const item=window._HS&&window._HS[secIdx]&&window._HS[secIdx].items[itemIdx];
  const overlay=document.getElementById('hm-overlay');
  const box=document.getElementById('hm-box');
  const title=document.getElementById('hm-title');
  const body=document.getElementById('hm-body');
  if(!overlay||!box)return;
  if(!item){
    title.textContent='Strategy Info';
    body.innerHTML=`<p style="color:var(--mu);font:11px var(--mono)">Detailed help for this strategy is in the <b>❓ Help</b> tab.</p>`;
  } else {
    title.textContent=item.n;
    body.innerHTML=`
      <div class="help-item-formula" style="margin-bottom:10px">${item.f}</div>
      <div style="font-size:11.5px;line-height:1.85;color:var(--txt)">${item.d}</div>
      ${item.links?refLinks(item.links):''}
      ${item.ai?aiLinks(item.n,item.ai):''}`;
  }
  overlay.classList.add('open'); box.classList.add('open');
}
function closeHelpModal(){
  document.getElementById('hm-overlay')?.classList.remove('open');
  document.getElementById('hm-box')?.classList.remove('open');
}
// ── Stock detail hover ──────────────────────────────────────────────────────
// Show on hover, close only via ESC or overlay click — no auto-close on leave
let _sdT=null;
function _sdHover(sym){
  clearTimeout(_sdT);
  // If already open just update; otherwise 150ms delay on first open
  if(document.getElementById('sd-box').style.display==='flex'){
    showStockDetail(sym);
  } else {
    _sdT=setTimeout(()=>showStockDetail(sym),150);
  }
}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape'){closeHelpModal();hideStockDetail();}
});


function renderHelp(){
  // Each item: n=name, f=formula, d=description, links=[[label,url]], ai=prompt
  const H=[
    {icon:'📊', title:'Pivot Points', items:[
      { n:'Fibonacci Pivot Points',
        f:'P=(H+L+C)/3 · R1=P+0.382×Range · R2=P+0.618×Range · R3=P+Range · S1=P−0.382×Range · S2=P−0.618×Range · S3=P−Range',
        d:'Fibonacci pivot points combine the classic floor-trader pivot formula with mathematical ratios discovered by Leonardo Fibonacci. The central pivot P is the simple average of the prior period\'s High, Low, and Close — the same as Traditional. Support and resistance levels are then derived by multiplying the period\'s trading range (High − Low) by Fibonacci ratios: 0.382, 0.618, and 1.000. These ratios emerge from the Fibonacci sequence, where consecutive numbers (1,1,2,3,5,8,13...) approach the Golden Ratio φ≈1.618. S2 at the 61.8% extension is the most-watched level because the Golden Ratio appears throughout nature and financial markets. When price approaches S2 from above, professional traders watch for reversal candlestick patterns, volume spikes, or RSI divergence. S1 at the 38.2% level is a shallow retracement level seen in fast-trending markets. R1 and R2 are resistance zones where profit-taking and short sellers emerge. The yearly timeframe matches TradingView\'s Auto pivot on weekly charts — thousands of traders are simultaneously watching these levels. Confluence of a Fibonacci pivot with a key EMA or order block significantly increases reversal probability. Backtesting on NSE large-caps shows S2 Fibonacci yearly holds as support with approximately 60-65% probability on initial tests.',
        links:[['Investopedia','https://www.investopedia.com/terms/f/fibonacciretracement.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:pivot_points'],['Wikipedia','https://en.wikipedia.org/wiki/Fibonacci_retracement']],
        ai:'Explain Fibonacci pivot points in detail for NSE stock trading. Include: how to calculate all levels, which levels are most significant and why, how to trade bounces at S2 and S3, what timeframe source gives levels matching TradingView weekly charts, the mathematical origin of 0.382 and 0.618 ratios, how confluence with other indicators improves reliability, historical examples on Nifty 50 stocks, and common mistakes traders make using Fibonacci pivots.'
      },
      { n:'Traditional Pivot Points',
        f:'P=(H+L+C)/3 · R1=2P−L · S1=2P−H · R2=P+(H−L) · S2=P−(H−L) · R3=2P+(H−2L) · S3=2P−(2H−L)',
        d:'Traditional pivot points are the oldest and most widely used pivot system, developed by floor traders at the Chicago Board of Trade before electronic trading existed. The central pivot P represents the "fair value" of the prior session — the balance point between buyers and sellers. R1 and S1 are the most closely watched levels, representing the first meaningful support/resistance after the pivot. The formula R1=2P−Low reflects where resistance forms if price retraces exactly the amount it rose above pivot. S2 and R2 capture the full prior range from pivot, targeting larger moves. S3 and R3 are extreme levels reached only in high-volatility sessions like earnings or macro events. Floor traders historically shouted levels referencing "pivot" or "R1" to execute without screens. Modern algorithmic trading systems place limit orders at Traditional pivot levels, creating self-fulfilling effects. When NSE opens above pivot, bias is bullish and S1 is first support; when it opens below, R1 is first resistance. The yearly Traditional pivot used by TradingView on weekly charts requires matching the exact H/L/C source period to align with institutional levels.',
        links:[['Investopedia','https://www.investopedia.com/terms/p/pivotpoint.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:pivot_points'],['Babypips','https://www.babypips.com/learn/forex/pivot-points']],
        ai:'Explain Traditional pivot points for NSE stock trading. How were they used by floor traders historically? Which levels are most important? How do algorithmic traders use these levels today? What is the difference between daily vs yearly source data? How do Traditional pivots compare to Fibonacci pivots in reliability? Provide specific examples of Traditional S1 and R1 levels working on Reliance or HDFC Bank over the past year.'
      },
      { n:'Camarilla Pivot Points',
        f:'R1=C+Range×1.1/12 · R2=C+Range×1.1/6 · R3=C+Range×1.1/4 · R4=C+Range×1.1/2 · S1=C−Range×1.1/12 · S2=C−Range×1.1/6 · S3=C−Range×1.1/4 · S4=C−Range×1.1/2',
        d:'Camarilla pivots were developed by Nick Stott in 1989 while trading bonds at Camarilla Capital Management, and the system was kept proprietary for years. The key innovation is using the previous period\'s closing price as the basis — not the (H+L+C)/3 midpoint. This makes Camarilla a mean-reversion system: price tends to revert toward the prior close rather than trending indefinitely away from it. The 1.1 multiplier was derived empirically from historical bond market data representing typical intraday noise. S3 and R3 are the critical mean-reversion levels — price reaching S3 has moved far from the prior close and has high probability of reverting. S4 and R4 are breakout levels: price breaking through S4 signals a genuine trend day has begun. Professional traders buy at S3 with a stop below S4, or sell at R3 with a stop above R4. R5 and S5 are extreme "capitulation-level" moves. The system performs best in range-bound markets and underperforms on strong trend days. On NSE Nifty 50 stocks, Camarilla S3 from the yearly source attracts significant institutional buying and selling interest.',
        links:[['Investopedia','https://www.investopedia.com/terms/c/camarilla.asp'],['TradingView','https://www.tradingview.com/scripts/camarilla/'],['Wikipedia','https://en.wikipedia.org/wiki/Camarilla_equation']],
        ai:'Explain the Camarilla pivot system developed by Nick Stott in 1989. How does using the closing price instead of (H+L+C)/3 change the levels? What is the logic behind S3/R3 for mean reversion vs S4/R4 for breakouts? When does Camarilla fail and why? How should traders combine Camarilla with other indicators? Provide a complete trading plan using Camarilla on NSE stocks with entry, stop, and target examples.'
      },
      { n:'DeMark Pivot Points',
        f:'If C<O: X=H+2L+C · If C>O: X=2H+L+C · If C=O: X=H+L+2C · P=X/4 · R1=X/2−L · S1=X/2−H',
        d:'DeMark pivots were developed by Tom DeMark, quantitative analyst whose clients include George Soros, Paul Tudor Jones, and Steven Cohen, and whose indicators are built into Bloomberg terminals. The system is minimalist — only three levels (P, S1, R1) — because DeMark believed more levels dilute focus. The conditional X formula adjusts directionally: when the close is below the open (bearish session), more weight goes to the low; when the close is above the open (bullish), more weight to the high. This makes DeMark\'s pivot inherently directional — it shifts with yesterday\'s sentiment. When price closes near the session high, the pivot shifts upward, signaling buyers were dominant. DeMark pivot\'s P level is the most reliable "directional bias" filter: open above P = bullish bias, below P = bearish. The tightest range of the seven pivot types makes DeMark conservative and precise. Professional quants use DeMark pivots as a first-pass directional filter before applying other systems. DeMark also created TD Sequential (also in this scanner) with the same philosophy of objective market structure analysis. For NSE weekly chart analysis, DeMark pivots on the yearly source provide clean first-indication support and resistance.',
        links:[['DeMark Analytics','https://demark.com'],['Investopedia','https://www.investopedia.com/terms/d/demark-pivot-point.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:pivot_points']],
        ai:'Explain Tom DeMark pivot points and how they differ from all other pivot types. Why does DeMark use a conditional formula based on close vs open? What is the significance of having only 3 levels instead of 7+? Who uses DeMark pivots professionally? How do DeMark pivots connect philosophically to TD Sequential? Provide practical trading examples of DeMark S1 and R1 acting as support/resistance on NSE large-cap stocks over the past 2 years.'
      },
    ]},
    {icon:'🎯', title:'Price Action / Smart Money Concepts', items:[
      { n:'Bullish Order Block',
        f:'Last bearish (red) candle before a significant upward impulse move',
        d:'Order blocks are a cornerstone of Smart Money Concepts (SMC), popularized by the Inner Circle Trader (ICT) methodology developed by Michael Huddleston. A bullish OB represents where large institutions placed bulk buy orders, creating an imbalance that drove price sharply upward. Large institutions must distribute millions of shares over a price range rather than all at once — that distribution zone becomes the order block. The last red candle before an impulsive up-move is specifically identified because it represents the final shakeout of retail sellers before institutional buying overwhelmed the market. When price returns to this zone later, unfilled institutional orders theoretically still wait there, ready to absorb selling. Traders look for two confirmations: a liquidity sweep below the OB zone (stop-loss grab), then a rejection pattern (engulfing candle or pin bar) when price touches the OB. The quality depends on how aggressive the resulting move was — a 2% daily candle from an OB carries more significance than a 0.5% move. OBs lose validity if price trades through them cleanly without reaction. In NSE markets, OBs appear near index rebalancing, quarterly earnings, and FII block deal periods. Combining an OB with a key Fibonacci or pivot level creates the highest-confidence confluence entry in the SMC framework.',
        links:[['ICT YouTube','https://www.youtube.com/c/InnerCircleTrader'],['Investopedia SMC','https://www.investopedia.com/terms/s/smart-money.asp'],['TradingView Scripts','https://www.tradingview.com/scripts/orderblocks/']],
        ai:'Explain Smart Money Concepts (SMC) order blocks in detail for NSE stock trading. How are bullish vs bearish order blocks identified? What is the institutional logic behind why price returns to order blocks? How do you distinguish high-quality from low-quality order blocks? What is the relationship between order blocks, liquidity sweeps, and fair value gaps? Provide a complete trade setup using a bullish order block on an NSE stock with precise entry, stop loss, and target levels.'
      },
      { n:'Fair Value Gap (FVG)',
        f:'Bullish FVG: candle[i−1].High < candle[i+1].Low · Bearish FVG: candle[i−1].Low > candle[i+1].High',
        d:'Fair Value Gaps are three-candle patterns where price moves so aggressively that no two-way trading occurs in a specific price range — an imbalance zone. In a bullish FVG, the high of the candle two bars ago is completely below the low of the candle ahead, meaning the middle candle leaped upward with no downside wicks bridging the gap. Markets tend to fill these imbalance zones over time as price seeks fair value — this is supported by both SMC theory and traditional gap theory in technical analysis. Unfilled bullish FVGs below current price act as support magnets; unfilled bearish FVGs above act as resistance magnets. Research suggests daily FVGs fill approximately 70-80% of the time within 90 days. Traders use FVGs as pullback entries: instead of chasing a move, they wait for price to retrace into the FVG zone. FVG size matters — very small FVGs (below 0.2% of price) are noise; large FVGs (above 1% of price) signal genuine institutional urgency. The scanner tracks the most recent 3 unfilled FVGs per stock in both directions. When a FVG overlaps with an order block it creates an "Optimal Trade Entry" zone — the highest-confidence SMC setup. In NSE markets, FVGs frequently form during earnings announcements and RBI policy decisions.',
        links:[['ICT FVG Guide','https://www.youtube.com/results?search_query=ICT+fair+value+gap'],['Babypips SMC','https://www.babypips.com/learn/forex/smart-money-concepts'],['TradingView','https://www.tradingview.com/scripts/fairvaluegap/']],
        ai:'Explain Fair Value Gaps (FVG) in Smart Money Concepts trading for NSE stocks. What is the precise mathematical definition of bullish vs bearish FVG? Why do markets fill these gaps and what percentage historically get filled? How should traders enter when price returns to an FVG? What is the difference between a FVG and a traditional price gap? When is an FVG considered invalid? Provide NSE examples where FVGs acted as strong support or resistance with specific stocks and dates.'
      },
    ]},
    {icon:'📈', title:'Volume / Institutional Footprint', items:[
      { n:'Volume Profile — POC, VAH, VAL',
        f:'POC = highest-volume price bucket · VAH/VAL = boundaries of 70% volume concentration (expanding from POC)',
        d:'Volume Profile is a histogram that distributes trading volume across price levels rather than time, revealing where the market has accepted or rejected price. The Point of Control (POC) is the price level with the most volume traded — the market\'s "fair value" or equilibrium price where most business was conducted. Price consistently gravitates toward the POC because institutions who traded there are motivated to defend their positions. The Value Area (70% of volume) derives from statistical theory — approximately 70% of data falls within one standard deviation in a normal distribution. VAH and VAL are decision points: breaking above VAH with conviction signals bullish sentiment shift, breaking below VAL signals bearish shift. When price is inside the Value Area, the market is balanced; when outside, it either returns (rejection) or continues away (breakout). The scanner approximates Volume Profile by distributing daily volume across 50 price bins using the High-Low range — a standard approximation when intraday tick data is unavailable. Market Profile theory (the academic predecessor) was developed by J. Peter Steidlmayer and the CBOT in 1985 for futures trading. On NSE large-cap stocks, the annual POC acts as a powerful equilibrium level where institutional portfolios were rebalanced. Price returning to the POC from above after a breakout gives high-probability re-entry opportunities.',
        links:[['Investopedia','https://www.investopedia.com/terms/v/volume-profile.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:volume_by_price'],['Wikipedia Market Profile','https://en.wikipedia.org/wiki/Market_profile']],
        ai:'Explain Volume Profile trading (POC, VAH, VAL) for NSE stocks. How is Volume Profile calculated from daily OHLC data without intraday ticks? How do traders use POC as support/resistance? What is the significance of the 70% value area? How does Volume Profile differ from On-Balance Volume? What trading strategies use VAH breakouts and VAL breakdowns? Provide examples of institutional price acceptance and rejection at POC levels on NSE large-cap stocks.'
      },
      { n:'OBV — On Balance Volume Divergence',
        f:'OBV[i] = OBV[i−1] + Volume (if Close>prev) or − Volume (if Close<prev) · Divergence: OBV direction ≠ price direction',
        d:'On-Balance Volume was developed by Joseph Granville and introduced in his 1963 book "New Key to Stock Market Profits." The fundamental premise: volume precedes price — institutions must accumulate shares before price rises, and this shows in OBV before price moves. The calculation adds the full day\'s volume when price closes up and subtracts when it closes down. Bullish OBV divergence (price falling, OBV rising) means buying volume is increasing even as price declines — classic institutional stealth buying. Bearish divergence (price rising, OBV falling) means institutions are distributing into retail buying strength. This divergence pattern was validated by multiple academic studies in the 1970s-1980s showing above-random prediction accuracy for price reversals. The most significant OBV divergences appear during the final phase of bear markets — price still grinding lower but delivery volumes quietly rising, signaling long-term institutional accumulation. The Accumulation/Distribution line (also in the scanner) is OBV\'s more sophisticated cousin, using the Money Flow Multiplier to weight sessions by where the close fell in the High-Low range. Combining OBV divergence with RSI divergence simultaneously is a powerful confluence signal. On NSE, OBV divergences are particularly visible in delivery percentage data during sector bottoms — often appearing 4-8 weeks before the price reversal confirms.',
        links:[['Investopedia OBV','https://www.investopedia.com/terms/o/onbalancevolume.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:on_balance_volume_obv'],['Wikipedia','https://en.wikipedia.org/wiki/On-balance_volume']],
        ai:'Explain OBV divergence for NSE stock trading. How does OBV detect institutional accumulation and distribution before price moves? What percentage of OBV divergences correctly predict reversals? How does OBV compare to the Accumulation/Distribution line? What are the most common causes of false OBV divergence? How should NSE traders combine OBV divergence with price action signals? Provide 3 historical examples of OBV divergence predicting major bottoms or tops in NSE sectors.'
      },
    ]},
    {icon:'⚡', title:'Multi-Indicator (Minervini + Weinstein)', items:[
      { n:'Minervini Trend Template — All 8 Conditions',
        f:'1.Price>150&200SMA · 2.150SMA>200SMA · 3.200SMA rising 1M · 4.50SMA>150&200SMA · 5.Price>50SMA · 6.≥25% above 52Wlow · 7.≤25% from 52Whigh · 8.RS≥70',
        d:'Mark Minervini won the US Investing Championship in 1997 with a 155% return and again in 2021 with a 334.8% return — one of the greatest verified trading records in history. He published the Trend Template in his 2013 book "Trade Like a Stock Market Wizard," which became required reading for growth stock traders globally. The eight conditions are designed as a progressive filter: each subsequent condition narrows the universe further, ensuring that only stocks in the strongest possible technical configuration are selected. Condition 1 confirms the stock is in a long-term uptrend above both major moving averages. Condition 2 (150>200 SMA) ensures the medium-term average is above the long-term — no "death crosses" in the background. Condition 3 (200 SMA rising) filters out stocks where the long-term trend is still turning rather than clearly established. Condition 4 (the "golden alignment") ensures all three major MAs are stacked in proper bullish order. Conditions 6 and 7 eliminate stocks near their annual lows (weak fundamentals) and those that have already had major runs and are far from recent highs. A perfect 8/8 score qualifies typically only 2-5% of NSE stocks at any time. Research shows stocks meeting 7-8 conditions significantly outperform the Nifty 50 on a risk-adjusted 12-month basis. The template functions as a "universe filter" — Minervini then applies fundamental analysis only to 8/8 stocks.',
        links:[['Minervini Official','https://www.minervini.com'],['Amazon Book','https://www.amazon.com/Trade-Like-Stock-Market-Wizard/dp/0071807225'],['Investopedia','https://www.investopedia.com/trading/minervini-trading-strategy/']],
        ai:'Explain Mark Minervini\'s Trend Template in complete detail for NSE trading. Why are all 8 conditions necessary and what does each filter out? How does the Trend Template connect to Weinstein Stage Analysis? What is Minervini\'s SEPA entry methodology beyond the Template? How should traders use the 8/8 score to prioritize watchlist stocks? What is the historical outperformance of 7-8 score stocks vs Nifty 50 in Indian markets? Include examples of specific NSE stocks that met all 8 conditions before major advances.'
      },
      { n:'Weinstein Stage Analysis',
        f:'Stage 1:Basing (flat 30W SMA) · Stage 2:Advancing (price above rising 30W SMA) · Stage 3:Topping · Stage 4:Declining (price below falling 30W SMA)',
        d:'Stan Weinstein published his Stage Analysis in "Secrets for Profiting in Bull and Bear Markets" (1988), formalizing a framework professional fund managers had used informally for decades. The 30-week Simple Moving Average (≈150 trading days) is the backbone — it represents half a year of trading, enough to smooth out noise while remaining responsive to genuine trend changes. Stage 1 (Basing) is smart money\'s quiet accumulation phase: price oscillates in a flat range while the 30W SMA bottoms and begins turning up. Stage 2 (Advancing) begins when price closes decisively above the flat/rising 30W SMA with expanding volume — Weinstein\'s buy signal. This is the only stage he recommends being long: the 30W SMA acts as a rising support floor that institutional investors defend with buy programs. Stage 3 (Topping) is distribution: institutions quietly sell to retail buyers attracted by positive momentum, causing the 30W SMA to flatten. Stage 4 (Declining) is the most dangerous: price below falling 30W SMA, any rallies are exit opportunities rather than buying opportunities. The framework is powerful because it aligns individual analysis with institutional investment behavior rather than retail sentiment cycles. In NSE markets, stocks transitioning from Stage 1 to Stage 2 in sectors experiencing FII inflows have historically generated the best risk-adjusted returns. The Stage 2 entry combined with RS Rating ≥ 80 represents Weinstein\'s highest-quality setup.',
        links:[['Weinstein Book','https://www.amazon.com/Secrets-Profiting-Bull-Bear-Markets/dp/1556236832'],['Investopedia','https://www.investopedia.com/terms/w/weinstein-stagge-analysis.asp'],['StockCharts','https://school.stockcharts.com']],
        ai:'Explain Weinstein Stage Analysis in complete detail for NSE trading. How do you precisely identify when a stock transitions from Stage 1 to Stage 2? What volume characteristics confirm a Stage 2 breakout? What is the "30-week line" and how do institutional investors use it? How do you identify Stage 3 topping behavior early? Provide examples of NSE stocks going through all 4 stages with specific dates, price levels, and what fundamental changes coincided with stage transitions.'
      },
    ]},
    {icon:'🔬', title:'Advanced Strategies', items:[
      { n:'Darvas Box',
        f:'Box top = high confirmed by 3+ subsequent bars not exceeding it · Box bottom = lowest low within box · Breakout = close above top with volume ≥ 1.5×avg',
        d:'Nicolas Darvas was a Hungarian professional dancer who turned $36,000 into $2,345,000 in 18 months during 1957-1958, documented in his 1960 book "How I Made $2,000,000 in the Stock Market." The Darvas Box emerges from his observation that stocks making new highs consolidate in tight boxes before the next leg up. A box forms when price makes a high, then fails to exceed that high for at least 3 consecutive bars — the high becomes the confirmed box top. The box bottom is the lowest point within the consolidation range. The key quality criterion is tightness: boxes where the High-Low range is below 15-20% of stock price indicate true institutional accumulation. The breakout occurs when price closes decisively above the box top with volume surge, confirming supply exhaustion and new demand entering. Stop loss is placed at the box bottom — a fall back through the bottom invalidates the trade thesis. Darvas received stock quotes via telegram while performing in European nightclubs, making his returns extraordinary given limited data access. Boxes work because they represent supply/demand equilibrium — when demand overcomes supply (breakout), price moves rapidly to the next equilibrium zone. In NSE mid and small-cap stocks, Darvas boxes forming alongside earnings growth acceleration and FII accumulation are among the most powerful setups.',
        links:[['Amazon Book','https://www.amazon.com/How-Made-2-000-000-Stock/dp/0818403969'],['Investopedia','https://www.investopedia.com/terms/d/darvas-box-theory.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Nicolas_Darvas']],
        ai:'Explain Darvas Box Theory for NSE trading. How did Nicolas Darvas discover this system while touring as a dancer? What are the exact criteria for a valid box? How tight must the consolidation be? What volume confirms a valid breakout vs false breakout? How should position sizing and stops be set? How does the modern NSE market differ from the 1950s US market where Darvas traded — does the pattern still work? Provide recent examples of Darvas Box breakouts on NSE stocks.'
      },
      { n:'VCP — Volatility Contraction Pattern',
        f:'3+ contractions, each smaller range and lower volume · Final contraction range < 10% of stock price · Pivot = high of last contraction',
        d:'The Volatility Contraction Pattern was developed and named by Mark Minervini, described in "Trade Like a Stock Market Wizard" as his most reliable entry pattern for growth stocks. VCP identifies the precise transition from consolidation to breakout with the highest continuation probability. The pattern requires three or more progressively tighter price contractions — for example, 25% → 12% → 6% — with each swing having lower amplitude AND lower volume. The contraction of both price range and volume is critical: it indicates sellers becoming exhausted and fewer shares available at current prices, while weak holders ("tourists") sell to strong hands ("residents"). The final pivot point — the highest point of the last contraction — is the precise buy entry level. Ideal VCP has 3-4 contractions; more than 5-6 suggests the base is "stale." Volume during the final contraction should dry up to below 50% of average daily volume — complete supply exhaustion. Breakout volume should surge at least 40-50% above average, confirming genuine demand overwhelming residual supply. VCP patterns typically take 6-16 weeks to develop fully. On NSE, VCP patterns alongside earnings acceleration in small/mid-caps are among the most powerful setups with strong follow-through statistics. Combining VCP with Minervini\'s Trend Template (8 conditions) eliminates weak patterns from stronger ones.',
        links:[['Minervini','https://www.minervini.com'],['Amazon','https://www.amazon.com/Trade-Like-Stock-Market-Wizard/dp/0071807225'],['Investopedia VCP','https://www.investopedia.com/terms/v/vcp.asp']],
        ai:'Explain Minervini\'s VCP pattern for NSE trading. What are the exact criteria for counting valid contractions? How do you measure contraction percentages from swing highs and lows? What does volume drying up indicate about supply/demand? How is the precise pivot entry point identified? What is the ideal stop-loss placement relative to the pivot? What is the average advance expected after a confirmed VCP breakout? Provide step-by-step analysis of a historical VCP on an NSE mid-cap stock.'
      },
    ]},
    {icon:'⭐', title:'Tier-1 Strategies', items:[
      { n:'Golden Cross / Death Cross',
        f:'Golden Cross: 50SMA crosses above 200SMA · Death Cross: 50SMA crosses below 200SMA · Both detected within last 10 days for "Fresh" signals',
        d:'The Golden Cross and Death Cross are the most widely watched technical signals in global finance, followed by institutional investors, retail traders, algorithmic systems, and financial media simultaneously. The 50-day SMA captures approximately 10 weeks of momentum; the 200-day captures approximately 40 weeks of trend. When the faster MA crosses above the slower, it mathematically means that medium-term momentum has shifted to exceed long-term momentum — a confirmation that recent positive action represents genuine trend change rather than a bounce. Ned Davis Research data shows that investing when the S&P 500\'s 50-day is above its 200-day and exiting otherwise has historically generated better risk-adjusted returns than buy-and-hold since 1900. Many large pension funds and systematic hedge funds have explicit mandates requiring equity exposure reduction when the 200-day MA is declining — this creates reflexive buying and selling pressure that amplifies these signals. The "Fresh" Golden Cross (last 10 days) is specifically valuable because large institutional reallocation processes are still early — funds that haven\'t repositioned are still in the buying process. The "Bull Zone" (ongoing 50>200) is valuable as a filter: many professional strategies only enter long positions when this condition is met. Death Cross can be particularly destructive in NSE\'s liquidity-constrained mid-cap stocks where institutional selling pressure faces limited absorption. Combining Golden Cross with RS Rating ≥ 70 and above-average volume on the cross day dramatically improves signal quality. Academic backtesting on NSE indices shows Golden Cross signals on weekly charts have strong predictive power for 3-6 month returns.',
        links:[['Investopedia Golden Cross','https://www.investopedia.com/terms/g/goldencross.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Golden_cross_(finance)'],['StockCharts MA','https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages']],
        ai:'Explain Golden Cross and Death Cross signals for NSE stock trading. What is the statistical win rate historically? Do they work differently in bull vs bear markets? How do algorithmic traders use these signals? What is the best way to combine Golden Cross with volume, RS Rating, and sector analysis? What are the biggest failure modes of Golden Cross signals in NSE markets? Provide backtesting of Golden Cross on Nifty 50 stocks over the last 10 years with win rates and average returns.'
      },
      { n:'RSI Divergence',
        f:'RSI = 100 − 100/(1 + RS), RS = Avg14gain/Avg14loss (Wilder smoothing) · Bullish div: price lower low + RSI higher low · Bearish div: price higher high + RSI lower high',
        d:'RSI was developed by J. Welles Wilder Jr. in his 1978 book "New Concepts in Technical Trading Systems," one of the most influential technical analysis books ever written. Wilder chose 14 periods based on his belief that markets operate on half-lunar cycle rhythms of 28 days — modern research has not confirmed this, but 14 periods has become so standard that self-fulfilling forces make it effective. RSI measures the magnitude of recent gains versus losses, normalized to 0-100. The most powerful RSI signal is divergence — not the overbought/oversold extremes. Bullish divergence (price lower low, RSI higher low) means each successive decline brings fewer new sellers and less downside momentum, even as price continues falling — institutions are quietly buying into weakness. This hidden accumulation typically occurs over 2-4 weeks before the visible reversal. The divergence is most reliable when RSI\'s second low is below 40 and when the divergence spans at least 10-15 trading sessions. False divergences occur frequently in strong downtrends — RSI can form bullish divergence multiple times before the actual bottom. Combining RSI divergence with a key support level (Fibonacci, pivot, or order block) dramatically reduces false signals. Academic research confirms RSI divergence has statistically significant predictive power for medium-term reversals on weekly charts. In NSE banking and IT sectors, RSI bullish divergence on weekly charts at yearly Fibonacci S2 or S3 levels has historically been among the highest-probability reversal signals.',
        links:[['Investopedia RSI','https://www.investopedia.com/terms/r/rsi.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Relative_strength_index'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi']],
        ai:'Explain RSI divergence in complete detail for NSE stock trading. How is Wilder\'s smoothed RSI different from simple RSI? What is the statistical reliability of bullish vs bearish RSI divergence? How do you avoid false divergences in strongly trending markets? How does RSI divergence compare to MACD divergence in reliability? What is the optimal RSI period for NSE daily vs weekly analysis? Provide historical examples of RSI divergence predicting major bottoms or tops on NSE Nifty or sector indices.'
      },
      { n:'Bollinger Band Squeeze',
        f:'Upper = 20SMA + 2σ · Lower = 20SMA − 2σ · BW = (Upper−Lower)/20SMA×100 · Squeeze = BW in bottom 25th percentile of prior 126-bar history',
        d:'Bollinger Bands were developed by John Bollinger CFA, CMT in the early 1980s, who observed that price volatility follows cycles — low volatility inevitably precedes high volatility. The "Squeeze" is Bollinger\'s specific signal when band width reaches its lowest level in 6 months (126 trading days), signaling that price is compressed and an explosive move is imminent. John Bollinger trademarked the phrase and spent years defending it from unauthorized use, reflecting the system\'s commercial value. The 20-period SMA and 2 standard deviation multiplier were chosen to contain approximately 95% of price action in a normal distribution — upper/lower band touches occur only during genuine volatility events. The squeeze alone does not indicate direction — it signals imminent large movement, which is why the scanner pairs it with 12-bar momentum for directional bias. The TTM Squeeze (John Carter\'s extension) adds Keltner Channels as a filter, firing only when Bollinger Bands are entirely inside Keltner Channels — producing fewer, higher-quality signals. In NSE markets, Bollinger Band squeezes develop over 4-8 week periods before major sector breakouts, especially in interest-rate-sensitive sectors before RBI policy meetings. The "momentum" measurement (12-bar price change) determines whether the eventual breakout is likely to be upward or downward, though false directional signals do occur. Professional volatility traders use BB squeeze to identify premium-selling opportunities in options markets — low volatility = cheap options, and buying straddles during the squeeze capitalizes on the inevitable expansion.',
        links:[['Bollinger Official','https://www.bollingerbands.com'],['Investopedia','https://www.investopedia.com/terms/b/bollingerbands.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:bollinger_bands'],['Wikipedia','https://en.wikipedia.org/wiki/Bollinger_Bands']],
        ai:'Explain Bollinger Band Squeeze in detail for NSE trading. How is band width calculated and what level defines a squeeze? What is the statistical relationship between squeeze duration and subsequent move size? How does the TTM Squeeze differ from standard Bollinger Squeeze? How do options traders use the Squeeze for straddle strategies? How do you identify breakout direction during a squeeze? Provide historical examples of Bollinger Band squeezes before major breakouts in NSE stocks or Nifty index.'
      },
      { n:'52-Week High Breakout + Volume',
        f:'Breakout: today\'s High > max(High, last 252 days excluding today) · Volume surge: today\'s Volume ≥ 1.5× 20-day average',
        d:'The 52-week high breakout is one of the most consistently documented phenomena in academic finance research. A seminal 2004 study by George and Hwang in the Journal of Finance showed that proximity to the 52-week high is among the strongest predictors of near-term price momentum — more powerful than traditional price momentum measures alone. The psychological mechanism is anchoring bias: investors compare current prices to the 52-week high as a mental reference point, creating persistent resistance. When demand finally overwhelms this supply and price breaks above the 52-week high, the sellers who anchored to that reference have exhausted their supply and the path clears for rapid advance. William O\'Neil documented that virtually every major market winner — Google, Amazon, Apple — made new all-time highs before their largest price advances, never buying near 52-week lows. The volume confirmation (1.5× average) distinguishes institutional demand from retail momentum chasing — institutions buying millions of shares produce characteristic volume spikes that precede sustained rallies. In NSE Nifty 50 stocks, 52-week breakouts coinciding with sector rotation (FII shifting from underweight to neutral) produce the most durable advances. The "consolidation" filter (last 10 days range < 5%) identifies stocks that have "coiled" below the 52-week high, where supply has been absorbed in tight range before the breakout. Combining 52-week high breakout with RS Rating ≥ 80 and above-average volume on the breakout day eliminates most false breakouts and produces a high-quality signal historically above 60% win rate for 20%+ advance.',
        links:[['CANSLIM','https://www.investors.com/ibd-university/can-slim/'],['George & Hwang 2004','https://scholar.google.com/scholar?q=george+hwang+52+week+high+momentum+2004'],['Investopedia','https://www.investopedia.com/terms/1/52weekhighlow.asp']],
        ai:'Explain the 52-week high breakout strategy for NSE stock trading. What does academic research show about buying near 52-week highs vs lows? What is the anchoring bias mechanism? What volume level confirms genuine institutional demand? How does O\'Neil\'s CANSLIM methodology use 52-week highs? What is the historical win rate of volume-confirmed 52-week breakouts on NSE? How do you set stop-losses for this strategy? Provide statistics and examples from NSE markets.'
      },
      { n:'NR7 / Inside Bar',
        f:'NR7: today\'s range (H−L) is narrowest of last 7 bars · Inside Bar: today\'s High < yesterday\'s High AND today\'s Low > yesterday\'s Low · Compression: today\'s range / 14-day ATR',
        d:'The NR7 pattern was documented by Toby Crabel in his 1990 book "Day Trading with Short-Term Price Patterns and Opening Range Breakout," which became so sought-after that second-hand copies sold for hundreds of dollars. Crabel\'s research on historical commodity and equity data showed that when a day\'s range is narrowest of the past 7 sessions, the probability of a larger-than-average range the following day increases significantly. The statistical logic is volatility mean reversion — volatility alternates between compression and expansion, and a 7-period extreme low tends to be followed by expansion. The specific "7" came from Crabel\'s empirical backtesting; NR4 produces more frequent but slightly less reliable signals, NR10+ produces rarer but more powerful signals. The Inside Bar represents complete price indecision — the current bar\'s entire range fits within the prior bar, indicating supply/demand equilibrium. This equilibrium resolves with a directional breakout, ideally in the direction of the underlying trend. The "ATR Compression %" shown in the scanner is the key quality metric: below 50% of ATR = extremely tight compression, best setups. Professional traders combine NR7 with trend alignment, proximity to support/resistance, and volume characteristics to filter high-quality setups. NR7 + Inside Bar (double compression) is the highest-confidence volatility expansion signal available from daily data. In NSE Nifty Bank options trading, NR7 days on the index frequently precede large directional moves — professional options traders use NR7 to identify ideal straddle entry days.',
        links:[['Toby Crabel Book','https://www.amazon.com/Day-Trading-Short-Term-Patterns-Opening/dp/0934380171'],['Investopedia Inside Bar','https://www.investopedia.com/terms/i/inside-days.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Candlestick_pattern']],
        ai:'Explain NR7 and Inside Bar patterns for NSE trading. Why did Crabel specifically use 7 periods? What is the statistical win rate and average range expansion following NR7 days? How does the Inside Bar relate to volatility compression? How should traders determine breakout direction? What is the best way to trade NR7 + Inside Bar double compression using Nifty options? Provide historical analysis of NR7 patterns on Nifty 50 or Bank Nifty before major moves.'
      },
    ]},
    {icon:'🏆', title:'Tier-2 Strategies', items:[
      { n:'Cup & Handle',
        f:'Cup: U-shape, depth 8-55%, recovery ≥75% · Handle: tight range <12% near cup lip, duration 5-20 days · Breakout: close above lip with volume ≥1.4×avg',
        d:'The Cup & Handle was discovered by William O\'Neil, founder of Investor\'s Business Daily, documented in "How to Make Money in Stocks" (1988). O\'Neil studied every major stock market winner from 1880-1990 and found the Cup & Handle pattern appeared repeatedly before the largest price advances. The cup represents institutional accumulation: price declines as weak holders sell (left side), reaches bottom where institutions begin buying (round bottom), then recovers as buying exceeds selling (right side). Ideal cup depth: 12-33% in normal markets, up to 50% in bear conditions. The U-shape (round bottom) is critical — V-shapes indicate panic without sufficient accumulation time. The handle is the final shakeout: it should slope slightly downward, duration 5-20 days, range below 12% of stock price, with volume drying up to signal complete supply exhaustion before the breakout. The breakout occurs when price closes above the cup lip (pivot point) on volume at least 40% above average. O\'Neil\'s research showed stocks breaking out of proper Cup & Handle patterns advance an average of 20-25% from the pivot before the next consolidation. False breakouts occur when volume is insufficient at the pivot break — proper volume surge is non-negotiable. In NSE markets, Cup & Handle patterns are most reliable in large-cap stocks with growing earnings, increasing institutional ownership, and RS Rating above 80. The pattern duration on NSE typically ranges from 7-26 weeks, with longer cups producing larger subsequent advances.',
        links:[['O\'Neil Book','https://www.amazon.com/How-Make-Money-Stocks-Winning/dp/0071614133'],['IBD CANSLIM','https://www.investors.com/ibd-university/can-slim/'],['Investopedia','https://www.investopedia.com/terms/c/cupandhandle.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Cup_and_handle']],
        ai:'Explain Cup and Handle in complete detail for NSE trading using O\'Neil criteria. What is ideal cup depth? How deep should the handle be? What confirms valid breakout vs false? What is the average advance after confirmed breakout historically? How does Cup & Handle connect to Weinstein Stage Analysis? What is the failure rate and what causes Cup & Handle breakouts to fail? Provide analysis of historical Cup & Handle breakouts on NSE stocks with price targets and outcomes.'
      },
      { n:'MACD Divergence & Histogram Flip',
        f:'MACD = EMA12 − EMA26 · Signal = EMA9(MACD) · Histogram = MACD − Signal · Bullish div: price lower low + MACD higher low · Flip: histogram crosses above/below zero',
        d:'MACD (Moving Average Convergence Divergence) was developed by Gerald Appel in the late 1970s and remains one of the top three most-used technical indicators globally. The 12/26/9 parameter combination was chosen by Appel through empirical optimization on equity markets. The histogram (MACD minus Signal) is often the most useful component — it shows the momentum of momentum: rising histogram means acceleration, falling histogram means deceleration. MACD divergence differs from basic MACD crossovers: it compares MACD peaks/troughs to corresponding price peaks/troughs, revealing hidden momentum changes not visible in price. Bullish divergence (price lower low, MACD higher low) indicates selling momentum weakening — smart money buying gradually supports the market, causing MACD to bottom before price. Academic research consistently shows MACD divergence has statistically significant predictive power for medium-term reversals, particularly on weekly charts. The Histogram Flip (crossing zero) is an early confirmation: when histogram crosses from negative to positive, the 12-period EMA has just crossed above the 26-period EMA. The NSE scanner detects flip events that occurred on the most recent bar, providing the earliest possible signal of momentum reversal. MACD divergence combined with RSI divergence simultaneously — "double divergence" — is considered one of the highest-probability reversal signals available from technical indicators. In NSE banking stocks, MACD divergence on weekly charts at major support levels (yearly Fibonacci or Camarilla S3) has historically produced 65%+ win-rate reversal trades.',
        links:[['Investopedia MACD','https://www.investopedia.com/terms/m/macd.asp'],['Wikipedia','https://en.wikipedia.org/wiki/MACD'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:moving_average_convergence_divergence_macd']],
        ai:'Explain MACD divergence and histogram flip for NSE trading. What is the difference between MACD line divergence and histogram divergence? How reliable is MACD divergence vs RSI divergence historically? When does MACD give false divergence signals and how to avoid them? How should the 12/26/9 parameters be adjusted for NSE weekly analysis? What is "double divergence" and why is it especially powerful? Provide examples of MACD divergence preceding major reversals in NSE sectors.'
      },
      { n:'Z-Score Mean Reversion',
        f:'Z = (Close − MA20) / StdDev20 · Extreme Oversold: Z < −3.0 (occurs <0.5% of sessions) · Oversold: Z < −2.0 · Overbought: Z > +2.0',
        d:'Z-score mean reversion is a quantitative strategy used by systematic hedge funds, statistical arbitrage desks at major banks, and algorithmic trading firms globally. The Z-score measures how many standard deviations the current price is from its 20-day moving average — a reading of +2 means price is statistically 2 standard deviations above recent mean, occurring only 2.5% of the time in a normal distribution. Mean reversion strategies rest on the empirical observation that asset prices oscillate around short-term moving averages, and extreme deviations self-correct. Z below −2.0 signals an unusually severe short-term decline — not a fundamental bear market, but a statistical outlier tending to retrace toward the mean within 5-20 days. Z below −3.0 (extreme oversold) is particularly powerful: research on global indices shows buying when Z < −3 and holding 10 days produces positive returns approximately 72% of the time with average gains of 3-5%. Bollinger %B (shown alongside Z-score) normalizes this to 0-100: below 0 means price is below the lower Bollinger Band, above 100 means above upper band. The key risk of mean reversion: "value traps" — a stock can be statistically oversold for fundamental reasons (deteriorating earnings, fraud, regulatory action) making further decline likely. This is why the scanner should be combined with 200-day SMA filter and minimum RS rating — statistically oversold stocks in long-term uptrends have much better mean reversion characteristics than those in downtrends. In NSE markets, Z-score extreme oversold signals have historically worked best in Nifty 50 index components during market-wide corrections rather than individual stock-specific declines.',
        links:[['Investopedia Mean Reversion','https://www.investopedia.com/terms/m/meanreversion.asp'],['Wikipedia Z-score','https://en.wikipedia.org/wiki/Standard_score'],['QuantifiedStrategies','https://www.quantifiedstrategies.com/mean-reversion-trading/']],
        ai:'Explain Z-score mean reversion for NSE stock trading. How is the Z-score calculated? What is the win rate of Z<-2 vs Z<-3 signals historically? How do you avoid value traps when using Z-score? How should position sizing be adjusted for mean reversion trades? What holding period maximizes returns on NSE Z-score signals? How does the Bollinger %B metric relate to Z-score? Provide backtesting results comparing Z-score mean reversion signals in Nifty 50 stocks.'
      },
    ]},
    {icon:'🎖', title:'Tier-3 Strategies', items:[
      { n:'Fibonacci Retracement from Swing Points',
        f:'Swing High → Low (or Low → High) · 23.6%=(H−L)×0.236 · 38.2%=×0.382 · 50%=midpoint · 61.8%=golden ratio · 78.6%=deep retracement',
        d:'Fibonacci retracements measure pullbacks within the most recent significant swing move — dynamic and context-dependent, unlike Fibonacci pivots which use fixed formulas on recent sessions. The 61.8% level comes from dividing any Fibonacci number by the one following it (55÷89=0.618); the 38.2% level from dividing by the number two positions forward (55÷144=0.382). The 50% level has no Fibonacci basis — it derives from Dow Theory and Gann Analysis — but has been empirically observed as a powerful equilibrium level. The 61.8% "Golden Ratio" is the most significant level because this ratio appears throughout nature (galaxy spirals, shell growth, leaf patterns) and financial markets have collectively decided to respect it as a major support zone. Professional traders draw Fibonacci from clear swing lows to highs on clean trends and look for confluence — when a Fibonacci level coincides with a pivot point, order block, or moving average, reaction probability increases significantly. The 38.2% retracement is typically the first meaningful support in a strong uptrend — shallow enough to retain momentum, deep enough to shake out weak hands. The 78.6% level (√0.618) is the "last line of defense" — a closing breach typically indicates a full reversal rather than retracement. The NSE scanner identifies the most recent swing high/low within 120 bars and computes all Fibonacci levels dynamically, including the trend direction. In NSE markets, the 61.8% Fibonacci of major Nifty corrections (2020 crash, 2022 correction) has historically attracted massive institutional buying. Fibonacci levels achieve their highest accuracy when 3+ technical factors converge at the same price zone.',
        links:[['Investopedia','https://www.investopedia.com/terms/f/fibonacciretracement.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Fibonacci_retracement'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:fibonacci_retracemen']],
        ai:'Explain Fibonacci retracement for NSE trading. How are the 23.6%, 38.2%, 50%, 61.8%, 78.6% levels mathematically derived? Why does 61.8% work so consistently in markets? How do you identify valid swing points for drawing Fibonacci? What is confluence and how does combining Fibonacci with pivots increase reliability? What is the difference between Fibonacci retracement and Fibonacci extension? Provide detailed examples of Fibonacci retracements on major NSE sector moves.'
      },
      { n:'ADX — Average Directional Index + DI System',
        f:'+DI = smoothed(+DM)/ATR×100 · −DI = smoothed(−DM)/ATR×100 · DX = |+DI−−DI|/(+DI+−DI)×100 · ADX = Wilder-smoothed DX (14 periods)',
        d:'ADX was developed by J. Welles Wilder Jr. and introduced in "New Concepts in Technical Trading Systems" (1978), alongside RSI, ATR, and Parabolic SAR — a remarkable set of indicators from a single source still universally used today. ADX measures trend strength without indicating direction — a high ADX means the market is trending strongly (either up or down); a low ADX means range-bound or choppy conditions. The +DI (Positive Directional Indicator) measures upward movement force; −DI measures downward force; when +DI exceeds −DI, the trend direction is bullish. Wilder\'s key thresholds: ADX below 20 = no meaningful trend (avoid trend-following strategies), ADX 20-25 = emerging trend, ADX 25-40 = established trend (trade confidently), ADX above 40 = extremely powerful trend (widen stops considerably). The DI Crossover is a trend direction change signal: +DI crossing above −DI indicates bulls gaining control; −DI crossing above +DI signals bears taking control. Without the ADX strength filter, DI crossovers produce excessive false signals in choppy markets — the ADX ≥25 filter dramatically reduces whipsaws. Professional trend-following funds historically used ADX as a "gating" condition: only apply trend system if ADX > threshold. In NSE sector ETF trading, ADX crossovers have been effective for identifying when beaten-down sectors begin institutional recovery. The ADX Extreme signal (≥40) on NSE large-caps signals a powerful trend that should be traded with momentum rather than contra-trend strategies.',
        links:[['Investopedia ADX','https://www.investopedia.com/terms/a/adx.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Average_directional_movement_index'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:average_directional_index_adx']],
        ai:'Explain ADX and the DI system for NSE trading. How is ADX calculated differently from +DI and -DI? What ADX level defines a meaningful trend? How should DI crossovers be used with ADX filters to reduce false signals? How do trend-following funds use ADX as a gating condition? What is the "ADX peak" signal and what does it indicate? Compare ADX to moving average crossovers for trend detection. Provide NSE examples of ADX extreme signals preceding sustained sector trends.'
      },
      { n:'Stochastic Oscillator',
        f:'%K = (Close − LowestLow14)/(HighestHigh14 − LowestLow14)×100 · Slow%K = 3-period MA of %K · %D = 3-period MA of Slow%K',
        d:'The Stochastic Oscillator was developed by George Lane in the late 1950s, one of the first computer-generated indicators, created on early mainframes for commodity analysis. Lane\'s core insight: in uptrends, closes tend near the daily high; in downtrends, near the low — measuring where the close falls within the recent range reveals momentum direction. The raw %K formula is called "fast stochastic" and is highly sensitive to noise; "slow stochastic" applies a 3-period MA before computing %D, significantly reducing noise while retaining signal sensitivity. The overbought zone (>80) and oversold zone (<20) represent extremes of momentum where statistical reversal probability increases. The bullish cross (%K crossing above %D) while both lines are below 40 combines oversold conditions with a momentum inflection — the scanner specifically requires the cross to occur below 40, not 80, for maximum reliability. George Lane himself warned against buying simply because Stochastic is oversold — in strong downtrends, the oscillator can remain oversold for weeks. "Triple divergence" (price making 3 lower lows while Stochastic forms progressively higher lows across 3 cycles) is considered by many analysts to be the single most powerful oscillator reversal signal. Stochastic performs best in range-bound markets where overbought/oversold levels have genuine reversal implications. In NSE futures and options trading, Stochastic crosses in the 15-minute timeframe using a 21-period setting (covering one full trading session) are popular for short-term entry timing.',
        links:[['Investopedia Stochastic','https://www.investopedia.com/terms/s/stochasticoscillator.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Stochastic_oscillator'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:stochastic_oscillator_fast_slow_and_full']],
        ai:'Explain the Stochastic Oscillator for NSE trading. What is the difference between fast, slow, and full stochastic? When is a stochastic buy signal valid vs invalid? How do you avoid false signals in trending markets? How do overbought/oversold thresholds need adjusting in different market regimes? Compare Stochastic to RSI and Williams %R for NSE trading applications. What is the statistical win rate of stochastic bullish crosses at key NSE support levels?'
      },
      { n:'Williams %R',
        f:'%R = (HighestHigh14 − Close)/(HighestHigh14 − LowestLow14) × (−100) · Overbought: > −20 · Oversold: < −80',
        d:'Williams %R was developed by Larry Williams, who famously turned $10,000 into $1.1 million in one year during the 1987 World Cup Championship of Futures Trading — one of the most documented short-term trading performances in history. The indicator is mathematically the inverse of Stochastic %K: while Stochastic measures from the bottom of the range, %R measures from the top. The −100 to 0 scale means readings near 0 (between 0 and −20) indicate the close is near the session high (overbought), while readings near −100 (between −80 and −100) indicate the close is near the session low (oversold). Williams intentionally designed the inverted scale to force traders to think differently, avoiding mechanical over-reliance. The most valuable signal is the "exit from oversold" (was below −80, now crossed above −80): this indicates the selling force that compressed price to extremes is losing momentum and a bounce is underway. As a 14-period fast oscillator, %R responds significantly more quickly to price changes than MACD or RSI, making it ideal for entry timing within a larger trend framework. Williams used %R as part of his "Ultimate Oscillator" system combining three timeframe Stochastics. Professional traders typically use %R for timing entries within trends established by slower indicators — for example, entering a long position on Williams %R oversold (−80) only when weekly MACD is in bull territory. In NSE derivatives trading, %R computed on 15-minute bars is particularly popular among F&O traders for timing positions at key intraday support/resistance levels.',
        links:[['Investopedia Williams %R','https://www.investopedia.com/terms/w/williamsr.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Williams_%25R'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:williams_r']],
        ai:'Explain Williams %R for NSE trading. How does it compare to Stochastic %K mathematically? Why did Williams use an inverted scale? What is the statistical reliability of the exit-from-oversold signal? How should Williams %R be combined with trend indicators? What period setting works best on NSE daily vs intraday charts? How did Larry Williams personally use this indicator in his 1987 World Cup strategy? Provide examples of Williams %R signals on NSE F&O stocks.'
      },
    ]},
    // ── 🏅 INDIA PRO (index 8) ─────────────────────────────────────────────
    {icon:'🏅', title:'India Pro — Chartink · StockEdge · Trendlyne', items:[
      { n:'Candlestick Patterns — Bullish Reversals',
        f:'Hammer: body/range<0.35, lower_shadow>2×body · Bullish Engulfing: C[-1]>O[-1] AND C[-2]<O[-2] AND bull body engulfs bear · Morning Star: bear→doji/star→bull closing >50% of bar 1',
        d:`Candlestick patterns are the most widely used technical signals in Indian markets, offered as premium scans on both Chartink (₹780/month) and StockEdge Club. They were originally developed by Japanese rice traders in the 18th century and formalized by Steve Nison in his 1991 book "Japanese Candlestick Charting Techniques." Each candlestick encodes four pieces of information (Open, High, Low, Close) into a single visual pattern that reveals the psychology of market participants during that session. The Hammer is the most reliable single-bar reversal pattern: a small real body at the top of the range with a lower shadow at least 2× the body length, indicating that sellers pushed price down aggressively but buyers rejected the lows and pushed it back up by close. Hammer reliability increases significantly when it appears at a key support level — Fibonacci pivot, order block, or weekly MA. The Bullish Engulfing two-bar pattern is the most institutionally significant reversal: the current bull candle's real body completely covers the prior bear candle, representing a wholesale shift in sentiment from selling to buying. The Morning Star three-bar pattern — large bear, small indecision star, large bull closing above 50% of the first bar — is the most complete reversal signal because it requires three consecutive sessions of transitioning sentiment, making false signals rare. The Three White Soldiers continuation pattern (three consecutive large-body bullish closes, each higher than the previous) is the most powerful bullish confirmation signal on daily charts. The Piercing Line forms when a bull candle opens below the prior day's low (gap down) then recovers to close above 50% of the prior bear candle's body — the sharp intraday reversal demonstrates strong buying conviction. On NSE, candlestick patterns are most reliable on stocks with high liquidity (Nifty 500 components) and should always be confirmed with volume — a hammer on 3× average volume is dramatically more reliable than one on below-average volume.`,
        links:[['Investopedia Candlesticks','https://www.investopedia.com/trading/candlestick-charting-what-is-it/'],['StockEdge Patterns','https://blog.stockedge.com/how-to-use-stockedge-price-scans/'],['Wikipedia','https://en.wikipedia.org/wiki/Candlestick_chart']],
        ai:'Explain bullish candlestick reversal patterns (hammer, engulfing, morning star, three white soldiers) for NSE stock trading. How reliable are each of these patterns statistically? Which patterns require volume confirmation? How should traders combine candlestick patterns with support levels and pivot points? What are the most common false signals in each pattern and how to avoid them? Provide historical examples of these patterns working on NSE large-cap stocks before significant rallies.'
      },
      { n:'EMA Crossovers — 9/21 and 20/50',
        f:'EMA(n) = α×Close + (1−α)×EMA_prev, α=2/(n+1) · Cross: EMA_fast[-1]>EMA_slow[-1] AND EMA_fast[-2]≤EMA_slow[-2] · Fan: Price>EMA9>EMA21>EMA50',
        d:`EMA (Exponential Moving Average) crossovers are the most popular technical alert on Chartink's paid platform, generating real-time alerts every minute for subscribers. Unlike Simple Moving Averages that weight all periods equally, EMAs apply exponentially decreasing weights so that the most recent prices have the greatest influence, making them significantly more responsive to current price action. The 9/21 EMA crossover is the short-term momentum signal: when the 9-period EMA crosses above the 21-period EMA, recent price momentum has decisively shifted upward and the short-term trend is turning bullish. This crossover is particularly popular among NSE intraday and swing traders because it responds quickly to price changes while still providing meaningful trend signals. The 20/50 EMA crossover is the medium-term momentum signal — slower and therefore more reliable, with fewer whipsaws than the 9/21 cross. When the 20-period EMA crosses above the 50-period EMA, institutions are committing to the uptrend and the signal typically precedes moves lasting several weeks to months. The "EMA Fan" pattern — where price is above EMA9, EMA9 is above EMA21, and EMA21 is above EMA50 — represents perfect bullish alignment and is one of the most powerful continuation signals for momentum traders. Trendlyne's momentum scan specifically identifies stocks in EMA fan formation as their "momentum leaders" category. The "Fresh 9/21 Bull Cross" (occurring within the last 8 bars) is particularly valuable because the move is still in its early stages, providing the best risk-reward entry window. In NSE markets, EMA crossovers have historically been most reliable when the broader Nifty index is also in an EMA fan configuration — this ensures that market-wide momentum supports individual stock moves.`,
        links:[['Chartink Scanner','https://chartink.com'],['Investopedia EMA','https://www.investopedia.com/terms/e/ema.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages']],
        ai:'Explain EMA crossover strategies (9/21, 20/50, EMA fan) for NSE stock trading as used on Chartink. What is the difference between SMA and EMA crossovers in terms of signal quality? How should traders filter EMA crossover signals to reduce whipsaws? What is the EMA fan pattern and why is it a powerful continuation signal? How do Chartink premium real-time alerts use EMA crossovers? Provide backtesting analysis of 9/21 EMA crossovers on NSE Nifty 50 stocks.'
      },
      { n:'Price Momentum Score — Trendlyne Style',
        f:'Score = r1M×0.15 + r3M×0.25 + r6M×0.30 + r12M×0.30 · ROC_N = (Close[-1]/Close[-N-1]−1)×100 · Positive all: all 4 timeframe returns > 0',
        d:`The composite Momentum Score is inspired by Trendlyne's momentum screening system, which is one of the most sophisticated publicly available momentum ranking tools for Indian markets. Price momentum — the tendency for stocks that have recently outperformed to continue outperforming — is one of the most empirically robust phenomena in financial markets, documented in academic research dating back to Jegadeesh and Titman's landmark 1993 Journal of Finance paper. The composite score weights four return horizons: 1-month (15%), 3-month (25%), 6-month (30%), and 12-month (30%). The higher weights on longer timeframes reflect the evidence that 6-12 month momentum is the most persistent and actionable signal, while 1-month momentum can be mean-reverting (the "short-term reversal" effect). A composite score above 20% means the stock has delivered exceptional risk-adjusted returns across all four measurement windows — a profile characteristic of the top 10-15% of NSE performers at any given time. The "All Positive" filter (all four return windows positive simultaneously) identifies stocks in sustained uptrends across every meaningful time horizon — these are the stocks where every buyer, regardless of entry point, is in profit. The Rate of Change (ROC) measures how fast recent momentum is accelerating: a 10-day ROC above 5% signals aggressive buying in the immediate term, often seen just before breakouts. Trendlyne's premium platform provides this momentum scoring for all 5000+ NSE/BSE listed stocks and allows sorting by composite score — a feature we approximate here using daily OHLC returns. Research consistently shows that buying the top momentum quintile and avoiding the bottom quintile produces significant alpha in Indian markets, with the best signals generated at the start of new market uptrends.`,
        links:[['Trendlyne','https://trendlyne.com'],['Investopedia Momentum','https://www.investopedia.com/terms/m/momentum.asp'],['Jegadeesh & Titman 1993','https://scholar.google.com/scholar?q=jegadeesh+titman+returns+buying+winners+1993']],
        ai:'Explain price momentum scoring for NSE stock screening as used by Trendlyne. How is composite momentum calculated across 1M, 3M, 6M, 12M return windows? What weighting scheme produces the best results? What is the Rate of Change indicator and how does it differ from simple returns? What does academic research show about momentum persistence in Indian markets? How does Trendlyne\'s momentum score compare to simple 12-month return ranking? Provide examples of high-momentum NSE stocks and their subsequent performance.'
      },
      { n:'Sequential Higher Highs / Higher Lows',
        f:'HH3: High[-1]>High[-2]>High[-3] · HL3: Low[-1]>Low[-2]>Low[-3] · HC3+Vol: Close[-1]>Close[-2]>Close[-3] AND Volume[-1]>avg20×1.2',
        d:`Sequential Higher Highs and Higher Lows analysis is a fundamental concept in Dow Theory (1902) that has been popularized in modern form by ScanX and Chartink as a quantitative filter for identifying stocks in confirmed uptrends. Charles Dow observed that a primary uptrend is defined by a series of Higher Highs (HH) and Higher Lows (HL) — each rally peak exceeds the prior peak, and each pullback finds support above the prior pullback's low. When both conditions hold simultaneously (HH+HL), the stock is in a healthy Stage 2 uptrend where buyers are consistently willing to pay progressively higher prices. The "3 Consecutive Higher Highs" (HH3) filter identifies stocks in immediate short-term uptrend momentum — the last three daily highs form a rising staircase. This is more specific and timely than traditional uptrend identification because it focuses on the most recent three sessions, capturing stocks actively being accumulated. The "5 Consecutive Higher Highs" (HH5) is an even stronger signal indicating a sustained, persistent trend that has been building for an entire trading week without a single down-day in terms of daily highs. The "3 Higher Closes with Volume Confirmation" is arguably the most valuable variant: consecutive higher closes indicate that closing prices (the most important price in each session) are rising, and the volume confirmation above average validates that institutional money is driving the move rather than thin-volume drift. In NSE markets, sequential higher highs are most often seen in stocks experiencing sector tailwinds — banking stocks during rate cut expectations, IT stocks during USD appreciation, or infrastructure stocks during government capex announcements.`,
        links:[['ScanX','https://scanx.trade'],['Chartink','https://chartink.com'],['Investopedia Dow Theory','https://www.investopedia.com/terms/d/dowtheory.asp']],
        ai:'Explain sequential higher highs and higher lows analysis for NSE trading as used by ScanX and Chartink. How does this connect to Dow Theory? What is the difference between HH3 and HH5 in terms of signal strength? How should volume be used to confirm sequential higher closes? What are common false signals in this approach and how to filter them? How does sequential structure analysis compare to moving average trend identification for NSE stocks?'
      },
      { n:'Volume Buildup / Accumulation Streaks',
        f:'Acc streak: each of last N days: Close[i]>Close[i-1] AND Volume[i]>Volume[i-1] · Quiet acc: |price_change_3d|<2% AND all(Volume[-i]>avg20×1.3) for i in [1,2,3]',
        d:`Volume Buildup scanning is one of StockEdge Club's signature features, used by over 1 million Indian retail investors to identify institutional accumulation before price breakouts. The core insight is that institutional investors — mutual funds, FIIs, insurance companies — must purchase shares over multiple sessions to build large positions without significantly moving the price. This accumulation process creates a characteristic pattern: consecutive days of rising price accompanied by rising volume, as demand progressively exceeds supply at each higher price level. A 3-day accumulation streak (three consecutive up-days each with higher volume than the prior day) is a reliable early signal that institutional demand is building. A 5-day accumulation streak is significantly rarer and much more powerful — five straight sessions of rising price and volume indicates sustained, programmatic institutional buying with strong conviction. The "Quiet Accumulation" pattern is even more sophisticated: it detects stocks where the price has moved very little (less than 2% in three days) but trading volume is persistently 30%+ above average. This pattern suggests that institutions are buying in size but carefully pacing their orders to avoid moving the market — exactly the kind of "stealth buying" that precedes major breakouts. StockEdge reports that high-delivery-percentage stocks combined with volume buildup streaks have historically been among the highest-quality pre-breakout setups in Indian markets. The delivery percentage (available from NSE bhav copy, not in the scanner's daily OHLC data) adds a crucial layer: when high delivery % accompanies a volume streak, it confirms genuine long-term buying rather than intraday speculative activity. Combining volume buildup with price consolidation near the 52-week high creates a very high-probability setup.`,
        links:[['StockEdge Scans','https://blog.stockedge.com/how-to-use-stockedge-volume-delivery-scans/'],['Investopedia Volume','https://www.investopedia.com/terms/v/volume.asp'],['StockEdge','https://web.stockedge.com/scan-groups']],
        ai:'Explain volume accumulation streaks and quiet accumulation for NSE stock scanning as used by StockEdge. How many consecutive up-volume days indicate genuine institutional buying? What is the difference between volume buildup and quiet accumulation? How does delivery percentage (NSE bhav copy data) enhance this signal? What is the historical success rate of 3-day vs 5-day accumulation streaks before price breakouts in NSE markets? How should traders combine volume buildup with price pattern analysis?'
      },
      { n:'52-Week High Consolidation / Tight Coil',
        f:'Coil10: dist%=(w52h−Close)/w52h×100 ≤5% AND (max_High_10d − min_Low_10d)/w52h×100 ≤5% · Coil15: dist%≤7% AND volume_5d_avg/vol_20d_avg<0.8',
        d:`The 52-Week High Consolidation scan is one of the most popular pre-breakout setups in Indian markets, offered as a premium filter on both Chartink and StockEdge. The pattern identifies stocks that are within 5-7% of their annual high but have entered a period of tight range consolidation — a "coil" — where the daily price range has contracted significantly. This pattern is the quantified version of the Volatility Contraction Pattern (VCP) and Darvas Box: after reaching new highs, the best stocks don't fall sharply — they consolidate tightly near the high, absorbing any remaining sellers, before resuming the uptrend. The "10-day tight coil" filter requires that the stock is within 5% of its 52-week high AND that the entire 10-day high-low range fits within a 5% band relative to the 52-week high. This extremely tight requirement ensures genuine supply/demand equilibrium — not a wide, sloppy consolidation but a genuine coil. The "15-day extended base" is a less strict but more common variant that adds the volume dimension: volume during the last 5 days should be below 80% of the 20-day average, indicating that the selling interest has completely dried up. William O'Neil's research showed that stocks in these tight bases near new highs had dramatically higher breakout follow-through rates than stocks breaking out from wide, volatile bases. Chartink's paid subscribers specifically scan for this pattern daily as it represents stocks where the "risk/reward is most favorable" — a very small stop (just below the recent low) with potentially large upside once the resistance is cleared. In NSE large-cap stocks, this setup has been particularly reliable for Nifty 50 constituents during sector rotation where FII inflows are building.`,
        links:[['Chartink','https://chartink.com'],['Investopedia Consolidation','https://www.investopedia.com/terms/c/consolidation.asp'],['IBD Chartink style','https://www.investors.com/ibd-university/can-slim/']],
        ai:'Explain the 52-week high consolidation / tight coil pattern for NSE trading as scanned by Chartink. What percentage from the 52W high qualifies as a valid consolidation? How tight should the price range be? What does volume drying up signal about supply? How is this pattern related to VCP and Darvas Box? What is the historical breakout success rate for tight coil setups within 5% of 52W high on NSE stocks? How should traders set entry and stop-loss for this setup?'
      },
      { n:'Multi-MA Alignment — All 4 Key Moving Averages',
        f:'Above all 4: Close>SMA20>SMA50>SMA100>SMA200 · above_count = count(Close > each SMA) 0-4 · Full stack: MA20>MA50>MA100>MA200 in descending order',
        d:`The "Price above All Key Moving Averages" scan is one of StockEdge's most popular price scan categories, representing the absolute strongest configuration a stock can be in from a technical standpoint. The four key Simple Moving Averages — 20 (1 month), 50 (2.5 months), 100 (5 months), and 200 (10 months) — represent distinct institutional time horizons. Short-term traders use 20-day MA, medium-term managers use 50-day MA, portfolio advisors use 100-day MA, and long-term institutional investors defend the 200-day MA. When all four MAs are in ascending order (MA20 > MA50 > MA100 > MA200) AND price is above all of them, every institutional holding period is profitable, and every major category of investor is supportive of the stock rather than looking to exit. This "Full MA Stack" is the most demanding technical filter in the entire scanner — typically only 5-10% of NSE-listed stocks pass this test at any given time. The "Above All 4 MAs" filter is the Minervini Trend Template compatible signal: it satisfies conditions 1-5 of Minervini's 8 conditions simultaneously. StockEdge reports this as one of their "price momentum" scans and it consistently identifies the strongest trending stocks in the market. The "above_count" metric (0-4) provides a useful ranking system: 4/4 is the strongest, while stocks at 3/4 (missing only 200-day) are often in the process of transitioning from Stage 1 to Stage 2. In NSE market bull phases (2020-2021, 2023-2024), approximately 40-50% of Nifty 50 stocks qualified for the full MA stack — serving as a market health indicator. During corrections, this number drops to 10-15%, signaling broad deterioration.`,
        links:[['StockEdge Price Scans','https://blog.stockedge.com/how-to-use-stockedge-price-scans/'],['Investopedia Moving Averages','https://www.investopedia.com/terms/m/movingaverage.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:moving_averages']],
        ai:'Explain the multi-MA alignment scan (price above 20/50/100/200 SMA) for NSE trading as used by StockEdge. Why are these four specific periods important for institutional investors? What percentage of NSE stocks typically qualify for all 4 simultaneously? How does this signal relate to Minervini\'s Trend Template? How should traders use the above_count (0-4) as a ranking system? What does the percentage of Nifty 50 stocks passing this filter tell you about market health? Provide historical analysis for Indian bull and bear markets.'
      },
    ]},
    // ── Section 9: India Pro new indicators ──────────────────────────────────
    {icon:'🏅', title:'India Pro — New Indicators (HA · HMA · Keltner · MFI · CCI · LinReg)', items:[
      { n:'Heikin Ashi Charts & Patterns',
        f:'HA_Close=(O+H+L+C)/4 · HA_Open=(prev_HA_Open+prev_HA_Close)/2 · HA_High=max(H,HA_Open,HA_Close) · HA_Low=min(L,HA_Open,HA_Close)',
        d:'Heikin Ashi (Japanese for "average bar") candles were developed as a noise-filtering alternative to standard candlesticks, using average prices rather than raw OHLC. The key insight is that HA candles smooth the price series in real-time without the lag of a moving average. A HA bull candle (close > open) with no lower shadow — where HA_Low = HA_Open — is the strongest possible signal: it means the smoothed price moved from its open to a higher close without touching below the open at any point, signifying pure uninterrupted buying pressure. Similarly, a HA bear candle with no upper shadow indicates pure selling. The HA Doji (small body) represents a transition moment in the smoothed trend — significantly more reliable than a standard Doji because it filters out the noise of individual session fluctuations. Consecutive HA bull candles (especially 5 or more) identify stocks in a sustained momentum move with minimal pullback on the smoothed chart. The first HA reversal — a single red candle after multiple greens — is an early warning that momentum may be waning, even if the standard daily chart looks fine. Chartink\'s Heikin Ashi scanner is consistently their most-used feature for swing traders. In NSE markets, HA consecutive bull streaks on the weekly chart are particularly reliable indicators of institutional accumulation phases. The smoothing effect means HA signals typically lag by 1-2 sessions vs raw candlesticks, but generate significantly fewer false signals.',
        links:[['Investopedia Heikin Ashi','https://www.investopedia.com/terms/h/heikinashi.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Candlestick_chart#Heikin_Ashi_technique'],['ChartAlert','https://www.chartalert.com']],
        ai:'Explain Heikin Ashi candles for NSE trading. How is HA calculated differently from standard OHLC? What does a HA bull candle with no lower shadow specifically indicate about price action? How do consecutive HA bull candles differ from a standard moving average signal? What is the HA Doji transition signal and how reliable is it? How should HA candles be used with other indicators for confirmation? Compare Heikin Ashi performance vs standard candlesticks on NSE swing trading strategies.'
      },
      { n:'Hull Moving Average (HMA)',
        f:'HMA(n) = WMA(2×WMA(n/2) − WMA(n), √n) · WMA weights: most recent bars have highest weight',
        d:'The Hull Moving Average was developed by Alan Hull in 2005 specifically to address the two most serious flaws in conventional moving averages: lag and noise. Standard EMAs and SMAs lag because they average past prices — the more periods, the more lag. Hull\'s elegant solution uses a double-weighted moving average subtraction: compute WMA(n/2) on the short window and WMA(n) on the full window, then double the short and subtract the long to get a "delagged" estimate. The final WMA(√n) step smooths the result to remove noise. The mathematical result is a moving average that responds almost as quickly as price itself while remaining significantly smoother than a short-period SMA. HMA slope direction changes are the primary signal: when the slope flips from negative to positive (flip_bull), it signals a trend reversal significantly earlier than any EMA crossover would. Price above a rising HMA is a bullish trend confirmation. ChartAlert includes HMA crossovers and slope changes as premium alerts. The difference between HMA and standard EMAs is most dramatic in choppy markets — EMAs generate excessive crossovers while HMA maintains a cleaner signal. For NSE positional trading (holding 2-8 weeks), the 20-period HMA provides excellent entry timing. The HMA(20) slope flip has been found to be one of the most reliable trend-change signals for NSE mid-cap stocks with good liquidity.',
        links:[['HullMA.com','https://alanhull.com/hull-moving-average'],['Investopedia','https://www.investopedia.com/terms/h/hull-moving-average.asp'],['ChartAlert','https://www.chartalert.com']],
        ai:'Explain Hull Moving Average for NSE trading. How does HMA solve the lag problem in standard moving averages? What is the mathematical formula and why does the WMA(√n) final step matter? How does HMA compare to EMA in terms of lag and smoothness? What period setting works best for NSE positional vs swing trading? How should HMA slope changes be used as entry/exit signals? Compare HMA performance vs EMA crossovers on NSE mid-cap stocks historically.'
      },
      { n:'Keltner Channels',
        f:'Middle = EMA(20) · Upper = EMA(20) + 2×ATR(10) · Lower = EMA(20) − 2×ATR(10) · ATR = Wilder-smoothed true range',
        d:'Keltner Channels were developed by Chester Keltner in the 1960s and significantly refined by Linda Bradford Raschke in the 1980s. Unlike Bollinger Bands which use standard deviation of price, Keltner Channels use Average True Range (ATR) — a volatility measure that also accounts for gaps. This makes Keltner Channels more stable and less "jumpy" than Bollinger Bands in trending markets. A price breakout above the upper Keltner Channel — especially on a closing basis — indicates that a genuine volatility expansion is underway, not just intraday spiking. The most powerful use of Keltner Channels is the "TTM Squeeze" (Tim Knight Squeeze): when Bollinger Bands contract INSIDE the Keltner Channel, it signals that volatility has compressed to extreme levels and a directional breakout is imminent. When BB expands back outside Keltner, the direction of the initial bar determines the squeeze resolution direction. This scanner already detects the BB-inside-Keltner condition in the Bollinger Squeeze scan (T1 tab). ChartAlert includes Keltner breakout alerts as a core premium feature. In NSE markets, Keltner Channel breakouts on the Nifty 50 index itself have reliably preceded 3-5% directional moves. The Keltner Channel width also serves as a market volatility gauge — when channels narrow (ATR contracting), a large move is building regardless of direction.',
        links:[['Investopedia Keltner','https://www.investopedia.com/terms/k/keltnerchannel.asp'],['ChartAlert','https://www.chartalert.com'],['Linda Raschke','https://www.lbrgroup.com']],
        ai:'Explain Keltner Channels for NSE trading. How do Keltner Channels differ from Bollinger Bands mathematically and practically? What is the TTM Squeeze and why does BB-inside-Keltner signal an impending breakout? How should Keltner breakouts be traded — close above upper band vs intraday? What multiplier setting works best for NSE stocks? How does the Keltner Channel width track volatility cycles? Compare Keltner Channel breakout reliability vs Bollinger Band breakouts on NSE large-caps.'
      },
      { n:'Money Flow Index (MFI)',
        f:'Typical Price = (H+L+C)/3 · Raw Money Flow = TP×Volume · Money Ratio = sum(pos MF, n)/sum(neg MF, n) · MFI = 100 − 100/(1+Money Ratio)',
        d:'The Money Flow Index was developed by Gene Quong and Avrum Soudack and is essentially a volume-weighted RSI — incorporating trading volume into the oscillator to distinguish between significant and insignificant price moves. A standard RSI oversold reading (≤30) tells you price fell sharply, but cannot distinguish between a fall on high volume (institutional selling) vs a fall on thin volume (temporary illiquidity). MFI below 20 on high relative volume is significantly more meaningful because it confirms that substantial capital was committed to the move, making the reversal more reliable. An MFI Bullish Divergence — price making new lows but MFI making higher lows — is particularly powerful because it suggests that despite lower prices, less volume is flowing into the downside each time, indicating institutional absorption of supply at lower prices. ChartAlert and Spider Software both feature MFI prominently in their premium technical scans. The volume weighting makes MFI especially valuable for NSE F&O stocks around derivatives expiry — extreme MFI readings near option expiry dates (3rd Thursday) often identify positioning extremes. A common institutional trading pattern: MFI reaches oversold (≤20) on weekly charts for Nifty 50 stocks, then crosses back above 20 = high-conviction buy signal with 70%+ historical win rate on NSE large-caps over the following 8 weeks.',
        links:[['Investopedia MFI','https://www.investopedia.com/terms/m/mfi.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:money_flow_index_mfi'],['Wikipedia','https://en.wikipedia.org/wiki/Money_flow_index']],
        ai:'Explain Money Flow Index for NSE trading. How does MFI differ from RSI by incorporating volume? Why is MFI oversold more meaningful than RSI oversold? How should MFI divergence signals be used? What is the difference between positive and negative money flow? How does MFI behave around NSE F&O expiry dates? What period setting and thresholds work best for NSE daily vs weekly charts? Provide historical analysis of MFI oversold signals on Nifty 50 stocks.'
      },
      { n:'Commodity Channel Index (CCI)',
        f:'Typical Price = (H+L+C)/3 · CCI = (TP − SMA_TP_N) / (0.015 × Mean_Absolute_Deviation_N) · Typical periods: 14 or 20',
        d:'The Commodity Channel Index was developed by Donald Lambert in 1980 and first described in Commodities magazine. The name reflects its agricultural futures origins, but CCI has since been applied universally across stocks, indices, and currencies. The 0.015 constant was specifically chosen by Lambert to ensure approximately 70-80% of CCI values fall within the ±100 range under normal conditions, making values outside ±100 statistically "unusual." Values above +100 indicate the price is significantly above its statistical average — not necessarily overbought in the traditional sense, but deviating meaningfully from the norm. CCI Zero-Line Crossovers are one of the cleanest signals: crossing from negative to positive territory indicates the trend has shifted from below-average to above-average pricing, a classic trend initiation signal. The "exit from oversold" signal (−100 to above −100) — analogous to Williams %R exit — is a high-reliability timing signal for entering mean reversion trades. CCI divergence (price making new highs but CCI making lower highs) signals distribution and is often seen at major market tops. ChartAlert features CCI prominently in their premium scan package for both NSE stocks and Nifty index trading. NSE traders commonly use the 14-period CCI on Nifty futures for intraday entry timing — CCI crossing above −100 from deep oversold has a strong historical win rate within 1-2 sessions on NSE F&O.',
        links:[['Investopedia CCI','https://www.investopedia.com/terms/c/commoditychannelindex.asp'],['StockCharts','https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci'],['Wikipedia','https://en.wikipedia.org/wiki/Commodity_channel_index']],
        ai:'Explain CCI for NSE trading. Why did Lambert choose 0.015 as the constant and what does it ensure statistically? What does CCI above +100 specifically tell you vs RSI overbought? How should CCI zero-line crossovers be traded differently from ±100 signals? How reliable is CCI divergence as a top/bottom indicator? What is the "CCI Woodie" variant? How do NSE F&O traders use CCI for intraday timing? Provide backtesting results for CCI exit-from-oversold signals on Nifty 50.'
      },
      { n:'Linear Regression Channel',
        f:'slope, intercept = polyfit(bar_indices[-N:], Close[-N:], degree=1) · fitted = slope×i + intercept · std_err = std(Close[-N:] − fitted) · dev = (Close_last − fitted_last) / std_err',
        d:'The Linear Regression Channel applies ordinary least squares (OLS) regression to price data — the same statistical technique used in academic finance, economics, and quantitative trading. Unlike a moving average which weights recent bars more heavily, the linear regression line minimizes the sum of squared deviations from ALL bars in the lookback window equally, finding the mathematically optimal line through the price series. The slope of the regression line is the single most unambiguous measure of trend direction available: a positive slope means price is trending up on average over the chosen lookback period; a negative slope means it is trending down, regardless of short-term noise. The standard error bands (±1σ, ±2σ) show how far price typically deviates from the trend line in both directions. Price at +2σ above the regression line means it is at a statistically unusual extreme — similar to a Z-score of 2, which occurs only about 5% of the time under normal conditions, suggesting potential mean reversion. Price at −2σ is often an excellent buying opportunity in trending stocks. Spider Software and ChartAlert both include Linear Regression Channels in their professional technical analysis toolkits. The 50-period LR channel (used in this scanner) provides a robust medium-term trend context. In NSE index trading, Nifty\'s relationship to its linear regression channel has been a reliable gauge of extended vs compressed conditions, with +2σ readings often preceding 1-3 week corrections.',
        links:[['Investopedia LR Channel','https://www.investopedia.com/terms/l/linearregression.asp'],['Spider Software','https://www.spider-software.com'],['Wikipedia','https://en.wikipedia.org/wiki/Ordinary_least_squares']],
        ai:'Explain Linear Regression Channel for NSE trading. How does OLS regression differ from a moving average in calculating the trend line? Why does the slope of the regression line provide the clearest trend direction signal? How should the ±2σ deviation bands be interpreted — mean reversion vs trend break? How does the linear regression channel compare to Bollinger Bands statistically? What lookback period works best for NSE positional trading? Provide examples of LR Channel +2σ readings preceding NSE corrections.'
      },
    ]},
    // ── Section 10: Noiseless tab ─────────────────────────────────────────────
    {icon:'📐', title:'Noiseless Charts — P&F · Renko · 3-Line Break · Kagi · RRG · Elliott Wave', items:[
      { n:'Point & Figure Charts — Definedge TradePoint',
        f:'Box size = 1% of price (auto) · Reversal = 3 boxes · X column = up price movement · O column = down movement · Double Top BO: current X > prior X high',
        d:'Point & Figure charts were introduced in the early 20th century by Victor de Villiers and Tom Dorsey and remain among the most sophisticated technical tools available. Unlike time-based charts, P&F plots only significant price moves (≥1 box), completely ignoring time — a stock that consolidates for 3 months creates no new P&F data, but a single large move creates multiple columns. This noise filtering makes P&F patterns far more statistically reliable than their candlestick equivalents. Definedge TradePoint is the definitive Indian platform for P&F analysis, built by Prashant Shah who has published extensively on P&F for NSE. The Double Top Breakout (current X column rising above the prior X column\'s high) is the most frequently cited bullish signal. The Triple Top Breakout — breaking above TWO prior X column highs at the same level — is rarer and significantly more powerful, as it represents a resistance level that has been tested three times before breaking. The Bullish Catapult (three consecutive X columns making higher highs) is the most sustained bullish P&F signal. High Pole Warning: when an unusually tall X column (≥5 boxes = significant run-up) is followed by an O column retracing more than 50%, it warns that the rally is potentially exhausted. The 1% auto box size ensures the P&F chart is properly scaled across all price levels.',
        links:[['Definedge TradePoint','https://www.definedge.com'],["Dorsey Wright",'https://dorseywrightassociates.com'],['Prashant Shah P&F book','https://www.definedge.com/books']],
        ai:'Explain Point & Figure charts for NSE trading as used by Definedge TradePoint. How do P&F charts filter time-based noise? What is the difference between Double Top and Triple Top Breakouts? How is the box size determined for NSE stocks? What is a Bullish Catapult pattern? Explain High Pole Warning and Low Pole Warning signals. How does P&F Relative Strength vs Nifty work? Provide historical examples of P&F breakout success rates on NSE Nifty 500 stocks.'
      },
      { n:'Renko Charts — Definedge 5-ka-Punch',
        f:'ATR-based brick size = ATR(14) · Reversal = 2 bricks in opposite direction · Green brick = close ≥ prior_close + brick_size · Red brick = close ≤ prior_close − brick_size',
        d:'Renko charts (from the Japanese word "renga" meaning brick) filter market noise by only recording a new brick when price moves by a minimum threshold — the brick size. Definedge TradePoint popularized ATR-based dynamic brick sizing for Indian markets, where stock prices vary from ₹10 to ₹10,000. The ATR-based approach ensures the brick size is proportional to actual volatility, making patterns comparable across different stocks and price levels. The "5 ka Punch" — five consecutive same-direction bricks — is Definedge\'s signature proprietary pattern name and is one of their most widely followed signals among Indian traders. It identifies stocks where price has moved decisively in one direction for five complete ATR-based moves without reversal, indicating strong institutional commitment to the trend. A fresh Renko reversal (brick direction just changed) is often the earliest possible trend-change signal available from daily data, as it requires a complete ATR-sized move in the new direction before confirming. Renko Double Tops and Double Bottoms — where two consecutive up-runs reach the same price level — identify significant resistance and support zones that have formed on the noise-filtered chart. The 2-brick reversal requirement means you need 2×ATR of movement in the opposite direction before a reversal is confirmed, filtering minor pullbacks. For NSE F&O traders, Renko charts on Nifty/Bank Nifty are extensively used for positional entry timing.',
        links:[['Definedge Renko','https://www.definedge.com/renko-charts-trading/'],['Investopedia Renko','https://www.investopedia.com/terms/r/renkochart.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Renko_chart']],
        ai:'Explain Renko charts for NSE trading as used by Definedge. How does ATR-based brick sizing work and why is it superior to fixed brick sizes? What is the 5-ka-Punch pattern and why is it significant? How should Renko reversals be traded for entry timing? How does Renko Double Top compare to standard chart Double Top in terms of reliability? How are Renko charts used by NSE F&O traders on Nifty index? Provide backtesting data on Renko bull signal performance vs standard MA crossover on NSE stocks.'
      },
      { n:'3-Line Break Chart',
        f:'New white line: Close > highest close of last 3 lines · New black line: Close < lowest close of last 3 lines · Reversal requires clearing all 3 prior lines',
        d:'3-Line Break charts originated in Japan and were introduced to Western traders by Steve Nison in his follow-up book "Beyond Candlesticks" (1994). The key characteristic is that reversals require significantly more price movement to confirm than standard charts — you must clear the high or low of ALL THREE prior lines before a new direction is established. This makes 3-Line Break reversal signals among the most reliable trend-change confirmations in technical analysis, as the noise threshold is much higher. A "New White Line" reversal from black to white — price exceeding the highest close of the last 3 black (down) lines — is considered a definitive bullish signal. The 3-line requirement can be thought of as requiring the bulls to "prove themselves" by erasing 3 consecutive bearish moves. Consecutive same-direction lines (5 or more) identify the strongest trends — stocks in 5+ consecutive white lines are in sustained institutional markup phases. Unlike Renko which uses a fixed box size, 3-Line Break box height varies with price action — bigger lines mean bigger moves were required to form them. For NSE positional trading, the weekly 3-Line Break chart is particularly powerful: a reversal on weekly 3-Line Break has very high false-signal rate because it requires overcoming weeks of trend momentum.',
        links:[['Steve Nison Beyond Candlesticks','https://www.amazon.com/Beyond-Candlesticks-Japanese-Charting-Techniques/dp/047100720X'],['Investopedia','https://www.investopedia.com/terms/t/three-line-break.asp'],['Wikipedia','https://en.wikipedia.org/wiki/Three-line_break']],
        ai:'Explain 3-Line Break charts for NSE trading. How does the 3-line reversal requirement filter noise? What is the mathematical relationship between 3-Line Break and Renko/P&F? When is a 3-Line Break reversal considered a confirmed trend change signal? How should traders use consecutive same-direction lines for trend strength assessment? Why is 3-Line Break applied to weekly data particularly powerful? Provide historical examples of 3-Line Break false reversal rates vs candlestick patterns on NSE stocks.'
      },
      { n:'Kagi Chart',
        f:'Yang line = rising price · Yin line = falling price · Reversal threshold = ATR(14) × 1.5 · Shoulder: Yang line exceeds prior Yang high · Waist: Yin line breaks prior Yin low',
        d:'Kagi charts are one of the oldest Japanese charting methods, predating candlesticks, and were originally used by rice traders in 17th century Osaka. A Kagi chart records price movement without time on the x-axis: a Yang (thick/rising) line continues upward as long as price advances; a Yin (thin/falling) line continues down as long as price falls. A reversal occurs only when price moves by the reversal threshold (ATR-based) in the opposite direction. The critical signal in Kagi is the "Shoulder" — when a new Yang line rises above the high of the prior Yang line. This signals that bulls have overcome the previous resistance level established during the last bullish phase. Conversely, the "Waist" — a Yin line breaking below the prior Yin line\'s low — signals bears have overcome prior support. ChartAlert and Definedge TradePoint both include Kagi as a premium noiseless chart type. The Kagi chart is more sensitive to reversals than P&F but less sensitive than Renko, making it a useful "middle ground" noiseless methodology. For NSE sector analysis, Kagi charts on sector indices (Nifty Bank, Nifty IT, Nifty Pharma) are effective for identifying when a sector has genuinely changed trend vs temporary price noise.',
        links:[['ChartAlert Kagi','https://www.chartalert.com'],['Wikipedia Kagi','https://en.wikipedia.org/wiki/Kagi_chart'],['Investopedia','https://www.investopedia.com/terms/k/kagi.asp']],
        ai:'Explain Kagi charts for NSE trading. How do Yang and Yin lines work differently from Renko bricks? What is the Shoulder signal and why is it more significant than a standard breakout? How does Kagi compare to P&F and Renko in terms of sensitivity? What reversal threshold setting works best for NSE stocks? How should Kagi Shoulder and Waist signals be combined with other technical confirmation? Provide historical comparison of Kagi vs P&F signal quality on NSE mid-cap stocks.'
      },
      { n:'RRG — Relative Rotation Graph',
        f:'RS = Close / Nifty_Close · RS_Ratio = RS / SMA(RS,10) × 100 · RS_Momentum = RS_Ratio / SMA(RS_Ratio,10) × 100 · Leading: both >100 · Lagging: both <100',
        d:'Relative Rotation Graphs were invented by Julius de Kempenaer and made famous by CNBC and Bloomberg as a visual tool for portfolio rotation analysis. The RRG plots each stock on a two-dimensional graph where the x-axis is RS-Ratio (relative strength level vs benchmark) and the y-axis is RS-Momentum (rate of change of relative strength). The four quadrants represent: Leading (top-right, both >100 = outperforming and accelerating), Weakening (bottom-right = still outperforming but decelerating), Lagging (bottom-left = underperforming and decelerating), and Improving (top-left = still underperforming but beginning to accelerate). The clockwise rotation of stocks through these quadrants over time is the central insight: stocks typically rotate Leading → Weakening → Lagging → Improving → Leading again. Strike by JST Investments uses RRG extensively for sector and stock selection in Indian markets. The most actionable signal is identifying stocks in the Improving quadrant before they move into Leading — this represents the early adoption window before general market recognition. Stocks in the Leading quadrant with still-rising RS-Momentum are the safest long positions. The RRG analysis requires the Nifty index CSV (pass --nifty to the scanner), as all calculations are relative to the Nifty 50 benchmark. NSE sector rotation analysis using RRG has been extremely effective for identifying which sectors FIIs are rotating into before the moves become obvious on price charts.',
        links:[['RRG Research','https://www.relativerotationgraphs.com'],['Julius de Kempenaer','https://www.relativerotationgraphs.com/education/'],['Strike India','https://strike.money']],
        ai:'Explain Relative Rotation Graphs for NSE trading as used by Strike/Indiacharts. How are RS-Ratio and RS-Momentum calculated vs Nifty? What does the clockwise rotation through the four quadrants mean practically? Which quadrant provides the best entry timing? How is RRG used for NSE sector rotation analysis? How does RRG compare to simple relative strength ranking? Provide examples of RRG Leading quadrant stocks preceding large NSE moves.'
      },
      { n:'Simplified Elliott Wave',
        f:'Zigzag pivots (5-bar local extrema) → alternating HH/LL → impulse: L-H-L-H-L pattern with W3>W1 and W4 not overlapping W1 · Fibonacci W3≥1.618×W1 = Wave 3 Extension',
        d:'Elliott Wave Theory was developed by Ralph Nelson Elliott in the 1930s and refined extensively by Robert Prechter. The theory states that markets move in fractal waves — 5-wave impulses in the direction of the trend and 3-wave corrections against the trend. This scanner uses a simplified algorithmic approximation: zigzag pivot detection followed by wave count validation using basic Elliott Wave rules. The Wave 3 Extension signal — where the third wave is at least 161.8% (the Golden Ratio) of Wave 1 — identifies the most powerful and fastest portion of an impulse move. Wave 3 in Elliott Wave cannot be the shortest wave by rule, and when it extends to 1.618×Wave 1, it often runs to 2.618×Wave1, providing significant follow-through. The ABC Correction Done signal identifies potential completion of a 3-wave corrective phase, suggesting the primary trend may resume. IMPORTANT: This is a simplified algorithmic approximation — not equivalent to the sophisticated proprietary Elliott Wave system used by Strike/JST Investments, which employs market analyst oversight and multi-timeframe rule validation. Use as confluence only. The pivot detection uses 5-bar local extrema, which on daily data identifies swings lasting approximately 1-3 weeks. A 3% minimum swing size ensures minor intraday volatility doesn\'t generate false pivots. Professional Elliott Wave counting requires context, degree analysis, and alternative count management that this algorithmic approach cannot provide.',
        links:[['Elliott Wave International','https://www.elliottwave.com'],['Investopedia EW','https://www.investopedia.com/terms/e/elliottwavetheory.asp'],['Strike India','https://strike.money']],
        ai:'Explain Elliott Wave Theory for NSE trading as approximated algorithmically. What are the key rules of impulse waves (wave 2, 3, 4 constraints)? What does a Wave 3 Extension (≥1.618×W1) indicate about price momentum? How is the ABC correction identified and what does completion signal? What are the limitations of algorithmic Elliott Wave counting vs professional analysis? How does Strike India use Elliott Wave for market phase identification? Provide examples of Wave 3 extensions on major NSE stocks preceding large moves.'
      },
    ]},
    // ── Section 11: Experimental (VSA) ───────────────────────────────────────
    {icon:'🧪', title:'Experimental — Volume Spread Analysis (VSA · Tom Williams)', items:[
      { n:'Big Volume Candle Strategy — VSA Core',
        f:'vol_ratio = Volume / avg_vol_20 · spread_ratio = (H−L) / avg_spread_20 · close_pos = (Close−Low) / (High−Low) · Big Vol Bull: vol_ratio≥2.5 AND spread≥1.5 AND close_pos>0.6',
        d:'Volume Spread Analysis (VSA) was developed by Tom Williams, based on the original work of Richard Wyckoff. Williams worked for decades analyzing the real-time order flow of professional trading syndicates and distilled the patterns into VSA methodology, published in his 1993 book "Master the Markets." The Big Volume Candle strategy is the centrepiece: it looks for the specific combination of ultra-high volume (≥2.5× average), wide price spread, AND close position within the bar\'s range. A Big Volume Bull candle — high volume, wide spread, close in the upper 60% of the range — means that despite large two-way activity, buyers decisively won the session. Institutions absorbed all available supply AND moved price up, a double signal of both demand and supply exhaustion. A Big Volume Bear candle — high volume, wide spread, close in the lower 40% — means sellers overwhelmed buyers, a distribution signal. The Big Volume Indecision candle — high volume but close in the middle 40-60% — represents a genuine battle with no winner yet; the next session\'s direction is critical. VSA is taught by Rakesh Jhunjhunwala\'s traders and is extensively used by high-net-worth NSE traders for timing entries and exits. The close position (0-100%) is the most important variable — on any day with significant volume, WHERE the close falls tells you who won the battle between supply and demand.',
        links:[['VSA Tom Williams','https://www.tradelikeapro.com/vsa-volume-spread-analysis-basics/'],['Wyckoff Analytics','https://www.wyckoffanalytics.com'],['Investopedia VSA','https://www.investopedia.com/articles/trading/07/tradingvsa.asp']],
        ai:'Explain Volume Spread Analysis and the Big Volume Candle strategy for NSE trading. How does VSA differ from standard volume analysis? What does close position within the bar range tell you about supply vs demand? How does the Big Volume Bull candle confirm institutional buying? What is the difference between stopping volume and effort-to-rise? How should VSA signals be combined with Wyckoff methodology? Provide historical examples of VSA big volume candle signals preceding major NSE stock moves.'
      },
      { n:'VSA — Upthrust, Spring, No Demand, No Supply',
        f:'Upthrust: wide spread + high vol + close_pos<0.35 · Spring: wide spread + high vol + close_pos>0.65 · No Demand: narrow spread + low vol + bull bar · No Supply: narrow spread + low vol + bear bar',
        d:'The Upthrust and Spring patterns are the most sophisticated VSA signals and are directly derived from Wyckoff\'s distribution and accumulation schematics. The Upthrust is a false breakout: price spikes to new highs on high volume, attracting momentum buyers, then reverses to close near the LOW of the session. All buyers who chased the high are immediately trapped. Tom Williams identified this as a classic institutional "take profit" operation — institutions sell into the buying frenzy created by the spike, then cover shorts at the close-near-low. The Spring (or Shakeout) is the mirror: price falls sharply to below support, triggering stop-losses of weak longs and attracting short sellers, then recovers immediately to close near the HIGH. Institutions buy aggressively at the dip below support, absorbing all the selling from panicking retail traders. No Demand is one of VSA\'s most underappreciated signals: a narrow spread UP bar on below-average volume means price rose, but with no institutional participation. Williams calls this "the market telling you it doesn\'t want to go up." No Supply is the pre-rally signal: price dips on low volume with a narrow range, indicating sellers have dried up completely. These four signals — Upthrust, Spring, No Demand, No Supply — form the core diagnostic toolkit for identifying institutional footprints at key price levels in NSE stocks.',
        links:[['Wyckoff Spring','https://www.wyckoffanalytics.com/the-wyckoff-spring/'],['Tom Williams VSA','https://www.tradelikeapro.com'],['VSA Indicators','https://www.tradersonline-mag.com/tutorials-trade/tools-indicators/volume-spread-analysis']],
        ai:'Explain the Upthrust, Spring, No Demand, and No Supply VSA signals for NSE trading. How does an Upthrust trap buyers and why does institutional selling cause it? What is the difference between a Spring and a regular bounce from support? Why does No Demand on a up bar with low volume indicate weakness rather than consolidation? How should these four signals be combined to identify Wyckoff accumulation and distribution? Provide examples of Spring and Upthrust signals on major NSE stocks near major turning points.'
      },
    ]},
    // ── Section 12: Patterns tab ──────────────────────────────────────────────
    {icon:'🎨', title:'Patterns — Harmonic Patterns · Chart Patterns', items:[
      { n:'Harmonic Patterns — Gartley, Bat, Butterfly, Crab, Cypher',
        f:'XABCD zigzag pivots · AB/XA, BC/AB, CD/BC, AD/XA must fall within Fibonacci tolerance (±10%) · Gartley: AB/XA≈0.618, AD/XA≈0.786 · Bat: AB/XA=0.382-0.5, AD/XA≈0.886 · Crab: AD/XA≈1.618',
        d:'Harmonic patterns are geometric price patterns based on precise Fibonacci ratios at key swing pivot points (XABCD), developed and systematized by Harold Gartley (1935) and extensively expanded by Scott Carney (2000s) in his book "Harmonic Trading." Each pattern defines specific Fibonacci relationships between the four price legs (XA, AB, BC, CD). The beauty of harmonic patterns is that the completion point D (the Potential Reversal Zone, PRZ) is identified BEFORE price reaches it, unlike most technical analysis which is reactive. PatternSurfer and HarmonicPattern.com charge ₹1,000-2,000/month in India specifically for this type of pattern recognition. The Gartley — the most common harmonic — has AB/XA approximately at the Golden Ratio (0.618) and the D completion point at 78.6% of the original XA move. This creates a specific price level where the reward-to-risk ratio is maximally favorable. The Bat pattern\'s D point at 88.6% of XA makes it the most aggressive (tightest stop) harmonic. The Crab, with D extending to 161.8% of XA, marks the most extreme price extensions. The Butterfly extends past the X point, often marking generational lows/highs. IMPORTANT: This scanner uses a simplified zigzag-based approximation with ±10% Fibonacci tolerance — results require confirmation with other signals. False patterns are common without volume confirmation at the PRZ.',
        links:[['HarmonicTrader.com','https://www.harmonictrader.com'],['PatternSurfer','https://www.patternsurfer.com'],['Investopedia Harmonics','https://www.investopedia.com/terms/h/harmonicpricepattern.asp']],
        ai:'Explain harmonic price patterns for NSE trading. How are the XABCD pivot points identified and what Fibonacci ratios define each pattern? What is the Potential Reversal Zone and how is it traded? How reliable are Gartley vs Bat vs Crab patterns statistically? How should PRZ confluence (multiple Fibonacci levels converging) be used to increase win rate? What is the difference between bullish and bearish harmonics? What are the limitations of algorithmic harmonic detection vs manual pattern drawing? Provide historical analysis of harmonic pattern win rates on NSE large-cap stocks.'
      },
      { n:'Chart Patterns — H&S, Wedges, Flags, Double Top/Bottom',
        f:'H&S: 3 peaks, center tallest, |LS−RS|/Head<5% · Wedge: converging trendlines from last 2 highs+lows · Bull Flag: strong pole + tight consolidation with pole_move>7%',
        d:'Classical chart patterns have been documented since the early 20th century by Richard Schabacker ("Technical Analysis and Stock Market Profits," 1932) and popularized by John Magee in "Technical Analysis of Stock Trends" (1948). These patterns represent recurring formations caused by the psychology of buyers and sellers at specific price levels. The Head & Shoulders is the most reliable reversal pattern in technical analysis: three peaks where the center (head) is highest and the two shoulders are at similar levels, with the neckline connecting the intervening lows. A confirmed close below the neckline initiates a bearish move with a target equal to the head height below the neckline. The Inverse H&S is the mirror: three troughs with the center (head) deepest, a break above the neckline targets the head height projected upward. The Rising Wedge is deceptively bullish-looking (both highs and lows are rising) but is actually bearish because the distance between support and resistance is narrowing, meaning buyers are losing momentum despite each successive high. The Falling Wedge is the opposite — despite falling, it resolves bullishly. Bull Flags are among the highest probability continuation patterns: a vertical "pole" move followed by a consolidation that drifts slightly against the trend, then resumes the original direction with the target equal to the pole length. This scanner uses 4-bar local extrema zigzag detection and 5% tolerance for pattern matching — confirmation with volume is essential before trading.',
        links:[['Investopedia H&S','https://www.investopedia.com/terms/h/head-shoulders.asp'],['Investopedia Bull Flag','https://www.investopedia.com/terms/b/bull-flag.asp'],['StockCharts Patterns','https://school.stockcharts.com/doku.php?id=chart_analysis:chart_patterns']],
        ai:'Explain classical chart patterns for NSE trading. What is the specific definition of a Head & Shoulders neckline and how is the price target calculated? Why does a Rising Wedge resolve bearishly despite making higher highs? How is a Bull Flag pole identified and what is the breakout target? How reliable are Double Tops vs Head & Shoulders as reversal signals? What volume patterns should accompany each pattern type for confirmation? How do these patterns perform on NSE mid-cap vs large-cap stocks historically?'
      },
      { n:'Double Bottom (W) — Daily / Weekly / Monthly with Neckline Confirmation',
        f:`Two zigzag lows within tol% of each other, with a peak (neckline) between them at least 3% above the lows. Breakout = Close > neckline.`,
        d:`The Double Bottom is the classic "W" reversal pattern: price falls to a low, bounces to an intermediate peak (the neckline), falls again to a similar low (within a tight tolerance band), then ideally breaks above the neckline to confirm the reversal. This scanner detects the W shape automatically using zigzag pivot detection on three independent timeframes. The DAILY version catches short-term reversals (days to weeks) - useful for swing entries. The WEEKLY version requires the W shape to form across weekly bars, representing a multi-month base - these breakouts tend to run for months once confirmed. The MONTHLY version is rare and represents a multi-year base; when a monthly W pattern breaks its neckline, it often marks the start of a multi-year bull phase. Each timeframe has a plain version (pattern visible, breakout not yet confirmed - useful for watching/alerts) and a "Neckline Breakout" version (pattern confirmed AND price has closed above the neckline - the actionable entry signal). The depth_pct field shows how deep the W is relative to the neckline - deeper patterns (10%+) tend to produce larger subsequent moves than shallow ones (3-5%).`,
        links:[['Double Bottom Pattern','https://www.investopedia.com/terms/d/doublebottom.asp']],
        ai:'How reliable is a Double Bottom pattern on NSE stocks compared to global markets? What is the typical price target after a neckline breakout, using the measured-move technique? How does pattern reliability differ between daily, weekly, and monthly Double Bottoms?'
      },
      { n:'Double Top (M) — Daily / Weekly / Monthly with Neckline Confirmation',
        f:`Two zigzag highs within tol% of each other, with a trough (neckline) between them at least 3% below the highs. Breakdown = Close < neckline.`,
        d:`The Double Top is the classic "M" reversal pattern: price rises to a high, pulls back to an intermediate trough (the neckline), rises again to a similar high (within a tight tolerance band), then ideally breaks below the neckline to confirm the bearish reversal. This scanner detects the M shape on three independent timeframes. The DAILY version catches short-term tops - useful for exiting longs or short-term short setups. The WEEKLY version represents a multi-month topping process; a confirmed weekly M pattern often precedes a multi-week-to-multi-month decline. The MONTHLY version is rare and represents a multi-year top - when confirmed, it can mark a major trend change lasting years. As with Double Bottoms, each timeframe has a plain "pattern visible" version and a "Neckline Breakdown" confirmed version. The depth_pct field shows how deep the M is - larger depths historically correspond to larger subsequent declines.`,
        links:[['Double Top Pattern','https://www.investopedia.com/terms/d/doubletop.asp']],
        ai:'How reliable is a Double Top pattern for predicting reversals on NSE large-cap vs mid-cap stocks? What is the typical decline magnitude after a confirmed neckline breakdown? Should I exit existing long positions on an unconfirmed Double Top, or wait for the neckline breakdown?'
      },
    ]},
    // ── Section 13: Market Breadth ────────────────────────────────────────────
    {icon:'📡', title:'Market Breadth — Universe-Wide Signals', items:[
      { n:'Market Breadth Dashboard — % Above MA, Stage 2, RS Leaders',
        f:'Health Score = %AboveMA200×0.25 + %Stage2×0.20 + %RS70×0.20 + %PnfX×0.10 + %RenkoBull×0.10 + %HaBull×0.08 + %Near52WH×0.07',
        d:'Market breadth analysis measures the health of the overall market by looking at how many individual stocks are participating in a trend, rather than just watching the index. A rising Nifty 50 on broad participation (70%+ stocks above MA200) is very different from a rising index driven by just 5-10 large-cap stocks dragging the index higher. This scanner computes breadth metrics across the full stock universe loaded from your data directory — a unique advantage that requires a full dataset. The "Percentage of stocks above MA200" is the classic breadth indicator: above 60% = healthy bull market; below 40% = bear market territory; below 20% = deep bear, potential generational buying opportunity. The Weinstein Stage 2 participation rate shows what fraction of stocks are in the primary advancing phase — this is the most actionable breadth metric for position traders. The RS ≥ 70 percentile count identifies how many stocks are in leadership territory. The Market Health Score is a proprietary composite of all breadth metrics weighted by their empirical reliability for NSE markets.',
        links:[['MarketBreadth.com','https://www.marketbreadth.com'],['Investopedia Breadth','https://www.investopedia.com/terms/m/market_breadth.asp']],
        ai:'Explain market breadth analysis for NSE trading. How does % stocks above MA200 indicate market health? What breadth level distinguishes a healthy bull market from a top-heavy market? How should Stage 2 participation rate be used as a market phase indicator?'
      },
    ]},
    {icon:'⚡', title:'Gap Scanner — Overnight Gap Signals', items:[
      { n:'Gap Up ≥1% / ≥2% / ≥3% — Significant opening gaps',
        f:'Gap% = (Open₀ − Close₋₁) / Close₋₁ × 100. Gap Up ≥1% | ≥2% | ≥3%',
        d:'An overnight gap occurs when the opening price is significantly different from the prior day\'s closing price due to news, earnings, block deals, or broad market moves. Gap Up ≥1% captures stocks that opened at least 1% above yesterday\'s close — these are among the most actively traded setups in NSE. Gaps on high-volume indices like Nifty FNO stocks tend to have stronger follow-through. The Gap Up ≥2% and ≥3% filters progressively narrow to higher-conviction moves. Gaps driven by institutional block deals before market open are especially powerful.',
        links:[['Chartink Gap Screener','https://chartink.com'],['NSE Bhav Copy','https://www.nseindia.com/market-data/live-equity-market']],
        ai:'Explain gap trading strategies for NSE. What gap sizes have the best follow-through on NSE? How does gap continuation differ from gap rejection? What is the difference between an exhaustion gap and a breakaway gap?'
      },
      { n:'Gap Up Continuation vs Rejection',
        f:'Continuation: Close > Open after gap up. Rejection: Close < Open after gap up.',
        d:'Not all gaps behave the same way. A Gap Up Continuation (close > open) means buyers held the gap and continued pushing prices higher intraday — this confirms institutional demand. A Gap Up Rejection (close < open after gap up) means sellers took control after the gap — the price could not sustain the opening euphoria. Gap rejections on stocks in distribution (IBD D/E rating) are strong shorting setups. Gap continuations on stocks with A/D accumulation ratings and strong RS are the highest-probability setups in NSE.',
        links:[],
        ai:'How do I trade gap up continuation vs gap up rejection patterns on NSE? What volume pattern confirms each? Which sectors gap up most reliably on NSE?'
      },
      { n:'Gap Unfilled — Demand Zone Confirmation',
        f:'Unfilled Gap: Low₀ > Close₋₁ (gap up) or High₀ < Close₋₁ (gap down) with no intraday fill',
        d:'An unfilled gap means price never returned to the prior close\'s level during the day. This confirms strong one-sided demand (for gap ups) — sellers at the prior close\'s price level were overwhelmed. Unfilled gaps on stocks at 52-week highs with volume 2× average are among the most powerful continuation signals. This is a paid Chartink filter — our scanner computes it natively from daily OHLC.',
        links:[], ai:'What is the significance of an unfilled gap in NSE trading? How long do gaps typically stay unfilled on NSE stocks? At what price levels do gap fills most commonly occur?'
      },
    ]},
    {icon:'⚗️', title:'AND Combiner — Multi-Signal Custom Scanner', items:[
      { n:'AND Combiner — Up to 5 Signals Simultaneously',
        f:'Result = Stock passes Condition1 AND Condition2 AND ... AND Condition5',
        d:'The AND Combiner is the most powerful tool in this scanner — equivalent to Chartink\'s premium paid plan and StockEdge\'s Combination Scans (₹999–2,499/month). Instead of scanning for a single signal, it finds stocks that simultaneously pass 2-5 independent conditions from ANY tab. For example: "SuperTrend Bull + Heikin Ashi Strong Bull + Volume Spike + Near 52W High + RS ≥ 70" — this combination would find stocks that are confirmed by five independent methods simultaneously. Each additional condition dramatically reduces the result set and increases the probability of follow-through. In a universe of 2,000 stocks: a single condition may return 100-150 results; combining 3 conditions returns 15-30; combining 5 conditions returns 3-8 highest-conviction setups. The combiner draws from all 19 tabs — Tier-1, Tier-2, Tier-3, India Pro, Noiseless, Patterns, Gap, Multi-Indicator, Breakout Pro, and Smart Rank signals.',
        links:[['StockEdge Combo','https://stockedge.com'],['Chartink Premium','https://chartink.com']],
        ai:'What are the most powerful AND combinations for NSE swing trading? Which combinations of indicators have the highest historical win rates? How many conditions should I combine to balance quality vs quantity of results?'
      },
    ]},
    {icon:'🏹', title:'Breakout Pro — Paid Scanner Strategies (Minervini · IBD · O\'Neil)', items:[
      { n:'VCP — Volatility Contraction Pattern (Mark Minervini)',
        f:'2+ successive weekly range contractions (each < prior) + volume declining + price within 15% of 52W high',
        d:'The Volatility Contraction Pattern is Mark Minervini\'s most powerful setup — the one he used to win the US Investing Championship with 220%+ annual returns. The stock forms successive contractions in its weekly price range: the first swing might be 8% wide, the second 5%, the third 3%. Each contraction should have lower volume than the previous, indicating that supply (selling pressure) is drying up. The final pivot before the breakout should be the tightest, with the lowest volume — this is the "T" (pivot point). A VCP breakout above the pivot on 2×+ average volume is the entry. This is only available on MarketSmith India (₹2,999/month). Our scanner detects VCPs from weekly OHLC — 2+ contractions with volume dry-up near the 52W high.',
        links:[['Minervini.com','https://www.minervini.com'],['MarketSmith India','https://marketsmithindia.com']],
        ai:'Explain Mark Minervini\'s VCP pattern in detail. What are the specific contraction ratios that define a valid VCP? How do you identify the pivot point? What volume patterns validate a VCP breakout on NSE stocks?'
      },
      { n:'Pocket Pivot (Gil Morales / Chris Kacher)',
        f:'Up-day Volume₀ > MAX(Volume of all down-days in prior 10 sessions)',
        d:'Developed by Gil Morales and Chris Kacher (former William O\'Neil portfolio managers). A pocket pivot occurs on an up-day when the volume exceeds the highest volume of any down-day in the prior 10 trading sessions. This signals that institutional accumulation is occurring BEFORE the formal price breakout — giving earlier, lower-risk entry. The stop loss is placed just below the pivot low, creating a very tight risk. Pocket pivots are especially powerful when they occur while the stock is still forming a base (not extended) and have strong RS. MarketSmith India exclusively provides pocket pivot detection. Our scanner computes it natively.',
        links:[['Gil Morales Book','https://www.amazon.com/Trade-Like-O-Neil-Disciples']],
        ai:'How does a pocket pivot differ from a regular volume surge? What is the success rate of pocket pivots on NSE compared to formal breakouts? Can pocket pivots occur in downtrends?'
      },
      { n:'3-Week Tight / 4-Week Tight / 5-Week Tight (O\'Neil)',
        f:'Weekly closing range: (MAX(closes₋₁..₋₃) − MIN) / MIN × 100 < 1.5%',
        d:'William O\'Neil identified that stocks in strong uptrends occasionally pause for 2-5 weeks in an extremely tight price range — weekly closes within ±1.5% of each other. This pattern forms when institutional holders refuse to sell (keeping supply off the market), while new buyers accumulate quietly. The result: price goes essentially nowhere for several weeks. The longer the tight area holds (5-week tight > 3-week tight), the stronger the eventual breakout. Available on StockEdge Premium and MarketSmith. The volume during tight weeks should also decline, confirming that sellers have stepped back.',
        links:[['O\'Neil How to Make Money','https://www.amazon.com/How-Make-Money-Stocks-William'],['MarketSmith India','https://marketsmithindia.com']],
        ai:'What is the difference between a 3-week tight and a flat base pattern? How tight should weekly closes be for a valid tight area? What breakout characteristics confirm a tight area pattern on NSE?'
      },
      { n:'RS Line New 52W High (IBD / MarketSmith)',
        f:'RS_line = Stock_Price / Nifty50_Price. is_new_high = RS_line[-1] >= MAX(RS_line[-252:]) × 0.99',
        d:'The Relative Strength line plots the ratio of the stock\'s price to the Nifty 50 index. When this line makes a NEW 52-week high — even before the stock price itself breaks out — it signals that the stock has been outperforming the broad market to reach a new relative peak. IBD research (William O\'Neil company) identifies this as the single most reliable LEADING indicator for monster stock moves. Stocks whose RS line makes new highs while the price is still forming a base are the ideal candidates for the next big move. Exclusive to MarketSmith India and IBD. Our scanner requires the Nifty50 CSV (--nifty flag) to compute RS lines.',
        links:[['IBD RS Rating','https://www.investors.com/how-to-invest/investors-corner/relative-strength-rating-stock-chart-analysis-picks-best-stocks-sp500-macd-deflect'],['MarketSmith India','https://marketsmithindia.com']],
        ai:'What is the RS Line (relative strength line) and how is it different from the RSI indicator? Why is the RS line making new highs before the stock price considered a leading indicator? How do I use the RS line to confirm a base breakout?'
      },
      { n:'IBD A/D Rating (Accumulation/Distribution Score)',
        f:'Score = Σ(+1 if up-day with volume≥avg, -1 if down-day with volume≥avg) over 65 trading days. A>35%, B>10%, C neutral, D<-10%, E<-35%',
        d:'IBD\'s Accumulation/Distribution rating grades stocks from A to E based on the net balance of buying vs selling pressure over the past 13 weeks. An A-rated stock has seen heavy institutional accumulation — more up-days on above-average volume than down-days. A B-rated stock shows moderate net buying. C is neutral. D and E indicate distribution (institutional selling). Stocks with A or B ratings before a breakout are significantly more likely to follow through. This rating is a paid feature on StockEdge and IBD. Our scanner computes the equivalent from daily OHLCV data.',
        links:[['IBD A/D Explained','https://www.investors.com/how-to-invest/investors-corner/accumulation-distribution-rating-stock-chart-analysis-picks'],['StockEdge','https://stockedge.com']],
        ai:'How is IBD\'s A/D rating different from the Accumulation/Distribution oscillator (Chaikin)? What A/D rating is required for a valid base breakout? How do I use A/D ratings to filter stocks before earnings?'
      },
      { n:'High Volume Breakout (HVB)',
        f:'HVB = Close > MAX(High[-252:-1]) AND Volume[-1] ≥ 2 × AVG(Volume[-51:-1])',
        d:'A High Volume Breakout is a 52-week high close accompanied by volume at least twice the 50-day average. Volume confirmation is the critical filter that separates institutional breakouts from retail noise. IBD studies show that low-volume new highs (below average volume) fail 60-70% of the time, while high-volume new highs (2×+ average) have a significantly better success rate as institutions are providing the buying power. The Strong HVB filter (3×+) further narrows to explosive institutional entries — these are often driven by block deals or mutual fund quarterly rebalancing.',
        links:[['O\'Neil Volume Rules','https://www.investors.com']],
        ai:'What volume threshold confirms a genuine 52-week breakout vs a false breakout on NSE? How does the 50-day average volume calculation work? Should I use absolute volume or relative volume for breakout confirmation?'
      },
      { n:'Volume Dry-Up (VDU) at Pivot',
        f:'VDU = AVG(Volume[-5:]) < 0.50 × AVG(Volume[-20:-1]) AND Close[-1] ≥ MAX(Close[-20:]) × 0.97',
        d:'Volume Dry-Up is a Chartink paid filter that identifies stocks where volume has contracted to below 50% of the 20-day average while price holds near recent highs. This signals that sellers have completely exhausted themselves — no one wants to sell at current prices — creating a potential energy situation for the next move. VDU combined with NR7 and BB Squeeze (Triple Compression) is one of the most reliable pre-breakout signals available from OHLCV data.',
        links:[['Chartink Premium','https://chartink.com']],
        ai:'Why does volume drying up at a pivot indicate accumulation rather than distribution? How long should volume dry-up last before a breakout? What is the typical volume expansion ratio at the breakout after a VDU setup?'
      },
      { n:'EMA Power Stack',
        f:'Full Stack: Price > EMA9 > EMA21 > EMA50 > EMA200. Power Stack: Full Stack + all 4 EMAs have positive 5-day slope',
        d:'The EMA Power Stack describes perfect bullish alignment across all major moving averages. Each EMA represents a different group of market participants: EMA9 (very short-term speculators), EMA21 (3-week swing traders), EMA50 (medium-term investors), EMA200 (long-term investors). When price is above all four AND all four are rising, every group is simultaneously profitable and bullish — there is no overhead resistance from any timeframe. TradePoint uses this as a core semi-algorithmic scanner signal.',
        links:[['TradePoint','https://www.definedgesecurities.com/products/tradepoint/']],
        ai:'What is the significance of all EMAs rising simultaneously vs just price being above all EMAs? How does the EMA Power Stack perform in trending vs ranging markets? What is the failure rate when only the partial stack (price > EMA21 > 50 > 200) is present?'
      },
      { n:'Inside Week',
        f:'Inside Week: High[-1W] < High[-2W] AND Low[-1W] > Low[-2W]',
        d:'An inside week occurs when the current week\'s entire price range (high to low) is contained within the prior week\'s range. This weekly compression signals a period of equilibrium between buyers and sellers — the market is coiling energy for a directional move. The direction of the eventual breakout (above the prior week\'s high or below its low) typically aligns with the prior trend. Double inside weeks (two consecutive weeks each inside the prior) are rarer and stronger compression signals. TradePoint\'s Pattern Finder identifies inside weeks across all NSE stocks.',
        links:[['TradePoint','https://www.definedgesecurities.com/products/tradepoint/']],
        ai:'What is the probability of an upward breakout from an inside week when it occurs in an uptrend? How does an inside week differ from a NR7 pattern? What is the typical breakout timing (days to weeks) after an inside week?'
      },
      { n:'Descending Trendline Breakout',
        f:'Trendline through last 3 peaks (linear regression on peak prices). Breakout: Close > trendline_value',
        d:'A descending trendline connects a series of lower highs — the bearish structure that defines a downtrend or consolidation. When price closes ABOVE this falling trendline, the sequence of lower highs is broken: the trend structure has changed. TradePoint\'s Pattern Finder scans for this across all stocks and timeframes. Our approximation uses the last 3 significant price peaks (local highs within ±2 bars) and fits a linear regression line through them to find the trendline value at today\'s date.',
        links:[['TradePoint Pattern Finder','https://www.definedgesecurities.com/products/tradepoint/']],
        ai:'How reliable is a descending trendline breakout as a trading signal? What volume is needed to confirm a trendline break? How do I draw valid trendlines — which highs to connect?'
      },
    ]},
    {icon:'🔢', title:'Smart Rank — AI Composite Scoring (Definedge · Trendlyne · IBD)', items:[
      { n:'Composite Momentum Score (0–100)',
        f:'MS = Rank(1M_return)×0.15 + Rank(3M_return)×0.20 + Rank(6M_return)×0.25 + Rank(12M_return)×0.30 + Rank(RS)×0.10',
        d:'The Composite Momentum Score is our implementation of Trendlyne\'s Momentum Score — computed purely from OHLCV data. Every stock in the universe is ranked by its return over 1 month, 3 months, 6 months, and 12 months separately. These percentile ranks are then weighted (shorter periods get lower weights, longer periods get higher weights) and combined with the RS percentile rank into a single 0–100 score. A score of 90 means the stock is in the top 10% of the universe by composite momentum. This is calculated in assign_smart_ranks() after all stocks are processed — it is a TRUE relative ranking, not an absolute indicator.',
        links:[['Trendlyne Momentum Score','https://trendlyne.com/score-details/']],
        ai:'How does a multi-period momentum score compare to single-period momentum for predicting future returns? What score threshold has been shown to have the best risk-adjusted returns? How does momentum score decay over time after initial computation?'
      },
      { n:'Seasonality Scanner (Historical Monthly Performance)',
        f:'Monthly avg return = MEAN(monthly_returns[month==current_month, years=5+]). Hit rate = % of years positive in that month',
        d:'The Seasonality Scanner is our implementation of Definedge\'s Seasonality Scanner (exclusive to their ₹2,499/month plan). For each stock and each calendar month, we compute the average percentage return across all instances of that month in the historical data. A stock that gains 4% on average in October with an 80% hit rate (8 of 10 years positive) is showing a strong seasonal pattern. The best seasonal setups combine: avg return > 3%, hit rate > 70%, and the current month is the stock\'s single best month of the year. Seasonal patterns are especially strong in NSE due to quarterly earnings cycles, FII rebalancing, F&O expiry effects, and budget season.',
        links:[['Definedge Seasonality','https://www.definedge.com/seasonality-scanner/']],
        ai:'Why do seasonal patterns exist in stock markets? Which months are historically strongest for NSE Nifty 50? How reliable are individual stock seasonal patterns compared to index seasonality? How many years of data are needed for reliable seasonal analysis?'
      },
      { n:'MTF Matrix Score (Daily + Weekly + Monthly, 0–15)',
        f:'5 checks × 3 timeframes: price>MA50 (+1), MA50>MA200 (+1), price>EMA21 (+1), RSI>50 (+1), price>5-period-ago (+1). Max=15.',
        d:'The MTF Matrix is our implementation of Definedge\'s Matrix feature — exclusive to RZone and TradePoint. For each timeframe (daily, weekly, monthly), five trend conditions are checked: is price above the 50-period MA? Is the 50-period MA above the 200-period MA? Is price above the 21-period EMA? Is RSI above 50? Is price higher than 5 periods ago? Each check scores +1. Maximum total = 15 (5 checks × 3 timeframes). A score of 15 means perfect bullish alignment on ALL three timeframes simultaneously — this is an extremely strong signal that every major group of market participants is bullish. Definedge extends this concept to 120 parameters across more chart types.',
        links:[['Definedge Matrix','https://www.definedgesecurities.com/user-manuals/rzone-user-guide/'],['RZone','https://www.definedgesecurities.com/faqs/rzone/']],
        ai:'Why is multi-timeframe analysis more reliable than single-timeframe analysis? What MTF Matrix score threshold has the best predictive power? How do daily, weekly, and monthly alignments contribute differently to trade outcomes?'
      },
      { n:'All-Time High Breakout',
        f:'ATH = Close[-1] > MAX(High[:-1]) — price exceeds entire price history in the loaded dataset',
        d:'An All-Time High breakout is conceptually the most powerful breakout type. There is NO overhead resistance — every single buyer who ever owned this stock at any price is in profit. This creates a pure momentum environment where the only question is how fast new buyers push the price higher before it finds new equilibrium. IBD research and StockTrendz AI backtesting on NSE data show ATH breakouts on above-average volume have the highest 6-month forward return probability of any technical pattern. The key condition is volume — ATH on low volume is a red flag; ATH on 2×+ average volume with strong RS confirms institutional involvement.',
        links:[['IBD ATH Research','https://www.investors.com']],
        ai:'What is the average 6-month return after an ATH breakout on NSE Nifty 500 stocks? How does volume at the ATH breakout affect subsequent performance? What is the psychological significance of an ATH breakout for market participants?'
      },
      { n:'Triple Compression (NR7 + VDU + BB Squeeze)',
        f:'NR7: range[-1] < MIN(ranges[-7:]). VDU: AVG(vol[-5:])<0.6×AVG(vol[-20:]). BB Squeeze: bandwidth[-1] < P20(bandwidth[-60:])',
        d:'Triple Compression identifies stocks where three independent compression mechanisms are simultaneously active: NR7 (the narrowest price range in 7 days), Volume Dry-Up (volume below 60% of 20-day average), and Bollinger Band Squeeze (bandwidth in the lowest 20th percentile of the last 60 sessions). Each of these alone signals potential energy; all three together represent maximum coiling in price range, volume, and volatility simultaneously. This combination has very few false signals and typically precedes explosive directional moves. The direction of the breakout usually aligns with the prior trend.',
        links:[],
        ai:'What is the typical magnitude of a move following a triple compression on NSE stocks? How many days after triple compression does the breakout typically occur? Should I trade in the direction of the prior trend or wait for the breakout direction to confirm?'
      },
      { n:'Consistent RS Leader (1M + 3M + 6M + 12M Top Quartile)',
        f:'Passes if: return_1M > P75(all_1M) AND return_3M > P75(all_3M) AND return_6M > P75(all_6M) AND return_12M > P75(all_12M)',
        d:'This is our implementation of Definedge\'s Smart RS Scanner. A "Consistent RS Leader" is a stock that ranks in the top 25% of the entire universe by return over ALL four time periods simultaneously: 1-month, 3-month, 6-month, and 12-month. This is a much stricter test than a single-period RS rank — it ensures the stock has been outperforming consistently across all timeframes, not just having a recent spike. Definedge charges for their multi-timeframe RS scanner; our version is computed natively. The Top 10% version (all four periods in top 10%) is even more restrictive and identifies the market\'s true sustained leaders.',
        links:[['Definedge Smart RS','https://www.definedgesecurities.com/blog/products/stock-selection-using-the-smart-rs-scanner/']],
        ai:'How does consistent multi-period RS leadership predict future performance? What percentage of stocks that are top-10% across all four periods maintain their leadership over the next 6 months? How does multi-period RS leadership compare to single-period RS rating?'
      },
    ]},

    {icon:'📐', title:'Donchian Channel & GMMA — Trend Systems', items:[
      { n:'Donchian 20-Day / 55-Day Breakout (Richard Donchian / Turtle Trading)',
        f:'Upper = MAX(High, 20 bars). Lower = MIN(Low, 20 bars). Bull = Close > Upper',
        d:`Richard Donchian built this channel in the 1950s; it became the Turtle Trading System in the 1980s. A close above the 20-day upper band = new 20-day high — the simplest mechanically backtested breakout signal. 55-day = Turtle System 2 slow entry, catches major trends. Donchian Squeeze (bandwidth at 6-month low) identifies the pre-breakout coiling state. This is Chartink's most-used breakout filter.`,
        links:[['Chartink','https://chartink.com'],['Turtle Rules','https://www.turtletrader.com']],
        ai:'Compare Donchian 20d vs 55d breakout success rates on NSE. How does bandwidth contraction predict breakout magnitude? What stop-loss rules did the Turtle Traders use with Donchian entries?'
      },
      { n:'GMMA Cross Bull — Guppy Multiple Moving Average (Daryl Guppy)',
        f:'Short-term: EMA(3,5,8,10,12,15). Long-term: EMA(30,35,40,45,50,60). Cross = short avg > long avg (fresh today)',
        d:`Daryl Guppy identified two dominant market participant groups: short-term traders (3-15 day EMAs) and long-term investors (30-60 day EMAs). When the short-term ribbon crosses above the long-term ribbon, both groups are aligned bullishly. "All-Bull" (zero ribbon overlap) is the highest conviction state. "GMMA Compression" (ribbons converging) signals major trend change pending. Available on Zerodha Kite and TradePoint — not on Chartink or StockEdge.`,
        links:[['TradePoint','https://www.definedgesecurities.com']],
        ai:'How does GMMA compare to a simple Golden Cross? What does GMMA compression predict? How is GMMA used with Renko charts for trend confirmation?'
      },
    ]},
    {icon:'📊', title:'RSI Trend Filter & MACD Histogram — Advanced Momentum', items:[
      { n:'RSI Cross Above 50 — Momentum Phase Shift',
        f:'cross50_bull = RSI(14)[-1] > 50 AND RSI(14)[-2] <= 50',
        d:`RSI 50 is the professional's momentum threshold — RSI crossing above 50 means average gains exceed average losses for the first time. This is earlier than waiting for RSI oversold recovery and more reliable than RSI overbought for trend entry. Stage 2 stocks consistently hold RSI above 50 on pullbacks. RSI Bull Zone (40-80) selects stocks in sustainable uptrends — stocks that drop below 40 are in distribution.`,
        links:[],
        ai:'Why is RSI 50 more important than RSI 30/70 for trend trading? How does RSI behave differently in bull vs bear markets on NSE? What is the win rate of RSI-cross-50-bull combined with price above 200 SMA?'
      },
      { n:'MACD Histogram Rising (Negative + Improving) — Early Bull Divergence',
        f:'hist_neg_inc = hist[-1] < 0 AND hist[-1] > hist[-2]',
        d:`MACD histogram rising from negative territory is an earlier entry signal than waiting for the histogram to cross zero. When the histogram is still negative but improving, selling pressure is diminishing before bulls take control. This is the early divergence signal — entering before the MACD crossover that the crowd uses. Chartink charges for this filter. Zero-Line Cross (MACD > 0) is the stronger confirmation: equivalent to a 12/26 EMA Golden Cross.`,
        links:[['MACD explained','https://www.investopedia.com/terms/m/macd.asp']],
        ai:'How much earlier does histogram divergence signal a trend change vs the signal-line crossover? What is the forward return after MACD zero-line cross on NSE midcaps?'
      },
    ]},
    {icon:'📍', title:'Pivot Bounce Scanner — Support & Resistance', items:[
      { n:'Pivot Bounce Filter — Price Near S1/S2/P/R1/R2 (0.7% tolerance)',
        f:'P=(H+L+C)/3. S1=2P-H. S2=P-R. R1=2P-L. R2=P+R where R=H-L. Near = |price-level|/level < 0.7%',
        d:`The Pivot tab now has a Bounce Filter dropdown — combine any pivot strategy with a price-level filter. "Near S1" finds stocks testing first support; "Above P" finds stocks that just crossed the pivot bullishly. These are the most-searched Chartink scans: stocks near support about to bounce. The 0.7% tolerance catches stocks within ±7 points on a ₹1000 stock — tight enough to be actionable.`,
        links:[['Floor Trader Pivots','https://www.investopedia.com/terms/p/pivotpoint.asp']],
        ai:'What is the bounce success rate at S1 vs S2 on NSE? How should I combine pivot bounces with candlestick patterns? Which pivot types (Fibonacci vs Classic) have better bounce reliability?'
      },
    ]},
    {icon:'🎯', title:'Swing Setup — EMA Squeeze · MACD 5/35 · Trade Plans', items:[
      { n:'EMA Squeeze (10/20/50/100/150/200) — Daily/Weekly/Monthly',
        f:`tight_pct = (MAX(EMA10..EMA200) - MIN(EMA10..EMA200)) / AVG(EMAs) * 100. Squeeze if tight_pct <= 2.5%, Ultra-Tight if <= 1%`,
        d:`When 6 EMAs spanning 10 to 200 periods all converge to within 2.5% of each other, every major group of market participants - from day traders watching the 10 EMA to position traders watching the 200 EMA - is looking at nearly the same price. This represents maximum equilibrium. Historically, the tighter this squeeze, the more explosive the eventual breakout, because the energy has been compressed across ALL timeframes simultaneously rather than just one. The Daily Squeeze captures short-term coiling (days to weeks). The Weekly Squeeze (10/20/50/100-week EMAs) captures multi-month consolidations that resolve into multi-week trends. The Monthly Squeeze (10/20/50-month EMAs) is rare and signals multi-year base patterns - when these resolve, the resulting trend often lasts for quarters or years. Bull Stack squeeze (EMAs tight AND shorter-above-longer) is the highest-probability continuation setup: the stock is pausing INSIDE an established uptrend rather than reversing.`,
        links:[['EMA Ribbon Concept','https://www.investopedia.com/terms/e/ema.asp']],
        ai:'Explain why converging EMAs across multiple timeframes (daily/weekly/monthly) predict larger breakouts than a single-timeframe squeeze. What is the typical breakout magnitude after a daily EMA squeeze versus a weekly EMA squeeze on NSE stocks? How long does an EMA squeeze typically last before resolving?'
      },
      { n:'MACD 5/35/5 Crossover — Fast Swing Variant (Daily/Weekly/Monthly)',
        f:`Fast EMA(5) - Slow EMA(35) = MACD line. Signal = EMA(5) of MACD line. Cross_bull = MACD crosses above Signal. Computed independently for daily, weekly, and monthly closes.`,
        d:`Standard MACD uses 12/26/9 periods, tuned for medium-term position trades. The 5/35/5 variant uses a much faster fast-EMA (5 vs 12) and a faster signal line (5 vs 9), making it react roughly 40% quicker to momentum shifts - at the cost of slightly more false signals. This makes it ideal for SWING trading specifically: the Daily 5/35/5 cross catches entries 1-2 days before the standard MACD would, while the Weekly 5/35/5 cross identifies multi-week swing moves perfect for a 1-3 month holding horizon. The Monthly 5/35/5 cross is positional - when it turns bullish, the underlying trend change often persists for 3-12 months. The "All Timeframes Aligned" signal requires Daily, Weekly AND Monthly MACD(5,35,5) to all be bullish simultaneously - this is the strongest possible momentum confirmation because short-term traders, swing traders, AND positional investors are all aligned in the same direction at the same time. The Zero-Line Cross (MACD line itself crossing above zero, not just the signal line) confirms that the 5-EMA has crossed above the 35-EMA in absolute terms - a stronger trend-change confirmation than the signal-line cross alone.`,
        links:[['MACD Explained','https://www.investopedia.com/terms/m/macd.asp']],
        ai:'How does a 5/35/5 MACD compare to the standard 12/26/9 MACD for swing trading on NSE? What is the false-signal rate of the faster variant versus the standard one? How should position size differ when entering on a daily versus weekly MACD 5/35 cross?'
      },
      { n:'Swing Trade Suggestion — Entry / Stop / Target for 1 Week to 3 Months',
        f:`Entry = current price. Stop = Pivot S1 (if below price) else price - 1.5*ATR. Target = Pivot R2 (if above price) else price + 2.5*ATR. R:R = (Target-Entry)/(Entry-Stop)`,
        d:`The Swing Trade Suggestion combines the composite Buy/Sell score (built from 148 signals across 14 categories - see Stock Lookup) with concrete trade levels. "Swing Buy Candidates" filters to score >= 60 (Buy or Strong Buy grade) and computes a suggested entry (the current price), a stop-loss (placed at the nearest pivot support level S1, or 1.5x ATR below price if no pivot support is nearby), and a target (the next pivot resistance R2, or 2.5x ATR above price). The resulting Risk:Reward ratio is shown - a ratio above 1.5 is generally considered favorable for swing trades. "Strong Swing Buy" raises the bar to score >= 75 AND confluence across 4+ signal categories (meaning the bullish case is confirmed by multiple independent methods - e.g. trend-following AND smart money concepts AND momentum - not just one). "Breakout-Ready" combines an EMA squeeze (the stock is coiled, compressed across timeframes) with a fresh MACD 5/35 bullish cross (the momentum trigger has just fired) - this compression-plus-trigger combination is the textbook breakout entry pattern used by professional swing traders. The 1-week to 3-month horizon is appropriate because: EMA squeezes typically resolve within 1-4 weeks, MACD 5/35 weekly crosses persist for 4-12 weeks, and the ATR-based stop/target levels are calibrated for swing (not intraday or long-term investment) volatility.`,
        links:[['Position Sizing & R:R','https://www.investopedia.com/articles/trading/04/091504.asp']],
        ai:'How should I size a swing trade position based on the Risk:Reward ratio and my account size? What R:R ratio is considered acceptable for a 1-3 month swing trade on NSE? How often should I review and adjust the stop-loss on an open swing position?'
      },
    ]},
  ];

  let html=`<div class="help-wrap">
    <div style="font-family:var(--mono);font-size:9px;color:var(--mu);margin-bottom:14px;padding:8px 12px;background:rgba(59,158,255,.04);border:1px solid rgba(59,158,255,.12);border-radius:5px">
      ❓ Detailed reference for every strategy in the scanner — 10+ sentence explanations, authoritative references, and AI chat shortcuts to learn more. Click any section to expand.
    </div>`;
  window._HS=H; // expose to showHelp() popup
  H.forEach(sec=>{
    const cnt=sec.items.length;
    html+=`<details><summary>${sec.icon} ${sec.title} <span style="color:var(--mu);font-size:9px;margin-left:auto">${cnt} strategies ▼</span></summary><div style="padding:10px 14px;display:flex;flex-direction:column;gap:14px">`;
    sec.items.forEach(item=>{
      const refs=item.links?refLinks(item.links):'';
      const ai=item.ai?aiLinks(item.n,item.ai):'';
      html+=`<div class="help-item" style="max-width:100%">
        <div class="help-item-name">${item.n}</div>
        <div class="help-item-formula">${item.f}</div>
        <div class="help-item-desc" style="font-size:11px;line-height:1.85;color:var(--txt);margin-top:6px">${item.d}</div>
        ${refs}${ai}
      </div>`;
    });
    html+=`</div></details>`;
  });

  html+=`<div style="font-family:var(--mono);font-size:9px;color:var(--mu);margin-top:10px;padding:8px 12px;background:rgba(255,140,66,.04);border:1px solid rgba(255,140,66,.12);border-radius:4px">
    ⚠️ Not buildable from daily OHLC: VWAP (needs intraday) · FII/DII buying (needs NSE bhav copy) · Options Max Pain/PCR (needs options chain) · 
    Use <code>--nifty</code> flag to enable RS vs Nifty · Regenerate index.html after running pivot_scanner.py
  </div></div>`;
  return html;
}



// ── INFO PANEL TOGGLE ─────────────────────────────────────────────────────────
let _infoOpen=false;
function toggleInfo(){
  _infoOpen=!_infoOpen;
  const p=document.getElementById('info-panel');
  if(!p) return;
  p.classList.toggle('open',_infoOpen);
  if(_infoOpen && !p.innerHTML){
    p.innerHTML=document.getElementById('fbar').innerHTML;
  }
}
function setFbar(html, detail){
  _infoOpen=false;
  const fb=document.getElementById('fbar');
  const ip=document.getElementById('info-panel');
  if(fb) fb.innerHTML=html;
  if(ip){ ip.innerHTML=detail||html; ip.classList.remove('open'); }
}

// ── AUTO-SCAN (debounced) ─────────────────────────────────────────────────────
let _aScanTimer=null;
function triggerAutoScan(){
  clearTimeout(_aScanTimer);
  _aScanTimer=setTimeout(()=>{
    const t=currentTab;
    if(t==='home'||t==='help') return;
    if(t==='piv') scan();
    else if(t==='smc') scanSMC();
    else if(t==='vol') scanVol();
    else if(t==='mi') scanMI();
    else if(t==='adv'){ updateAdvInfo(); scanAdv(); }
    else if(t==='t1'){ updateT1Info(); scanT1(); }
    else if(t==='t2'){ updateT2Info(); scanT2(); }
    else if(t==='t3'){ updateT3Info(); scanT3(); }
    else if(t==='ti'){ updateTIInfo(); scanTI(); }
    else if(t==='nl'){ updateNLInfo(); scanNL(); }
    else if(t==='xp'){ updateXPInfo(); scanXP(); }
    else if(t==='patt'){ updatePATTInfo(); scanPATT(); }
    else if(t==='br'){ renderBreadth(); }
    else if(t==='gap'){ scanGap(); }
    else if(t==='combo'){ scanCombo(); }
    else if(t==='bp'){ scanBP(); }
    else if(t==='sr'){ scanSR(); }
    else if(t==='swing'){ scanSwing(); }
  },350);
}
function initAutoScan(){
  // Global index dropdown triggers re-scan on any tab
  const gi=document.getElementById('global-idx');
  if(gi) gi.addEventListener('change',triggerAutoScan);
  // Attach change/input listeners to all ctrl panel form elements
  document.querySelectorAll('[id^="ctrl-"] select').forEach(el=>{
    el.addEventListener('change',triggerAutoScan);
  });
  document.querySelectorAll('[id^="ctrl-"] input[type=number]').forEach(el=>{
    el.addEventListener('input',triggerAutoScan);
  });
}

// ── 🏅 INDIA PRO ─────────────────────────────────────────────────────────────
const TI_INFO={
  candle_hammer:'<b>Hammer</b> · Chartink/StockEdge · Small body at top, long lower shadow (≥2× body) — sellers tried but buyers rejected the lows · Most reliable at support levels or pivot zones',
  candle_inv_hammer:'<b>Inverted Hammer</b> · StockEdge · Small body at bottom, long upper shadow — buyers attempted a rally · Often precedes reversal when found at support',
  candle_bull_eng:'<b>Bullish Engulfing</b> · Chartink · Current bull candle body fully covers prior bear candle — aggressive buying overwhelmed selling · Most reliable at key support',
  candle_morning_star:'<b>Morning Star</b> · StockEdge · Three-bar reversal: large bear → small indecision star → large bull closing above 50% of bar 1 · High-probability bottom',
  candle_piercing:'<b>Piercing Line</b> · Chartink · Bear day, then bull opens below prior low and closes above 50% of prior body · Strong reversal signal at support',
  candle_bull_har:'<b>Bullish Harami</b> · Small bull candle body contained inside prior large bear candle · Indecision — watch for follow-through on next bar',
  candle_tws:'<b>Three White Soldiers</b> · Three consecutive bullish candles each with large body (>55% of range) and progressively higher closes · Strongest bullish pattern',
  candle_doji:'<b>Doji</b> · Body <10% of range — perfect indecision between buyers and sellers · Critical at swing highs/lows as potential reversal warning',
  candle_shooting_star:'<b>Shooting Star</b> · Bear candle with long upper shadow (≥2× body) at the top of an uptrend · Sellers rejected the highs — bearish reversal',
  candle_bear_eng:'<b>Bearish Engulfing</b> · Current bear candle engulfs prior bull candle · Institutional selling signal especially at resistance levels',
  candle_dark_cloud:'<b>Dark Cloud Cover</b> · Bull day, then bear opens above prior high and closes below 50% · Distribution signal — exit longs on confirmation',
  candle_evening_star:'<b>Evening Star</b> · Three-bar reversal: large bull → small star → large bear closing below 50% of bar 1 · High-probability top at resistance',
  candle_bear_har:'<b>Bearish Harami</b> · Small bear candle inside prior large bull · Indecision at top — monitor for confirmation breakdown next session',
  candle_tbc:'<b>Three Black Crows</b> · Three consecutive large-body bearish candles with lower closes · Strongest bearish continuation pattern',
  // Inside Bar
  candle_inside_bar:'<b>Inside Bar</b> · Chartink #1 most-used scan · Current bar\'s High is below prior High AND Low is above prior Low — the entire bar is "inside" the prior bar · Represents perfect consolidation and supply/demand equilibrium · Breakout in either direction follows — use trend direction to determine bias',
  candle_inside_bar_bull:'<b>Inside Bar + Bullish Close</b> · Inside bar where the close is in the upper half of the bar\'s range · Buyers stepped in during consolidation · Higher probability of upside breakout — combine with uptrend or support level for conviction',
  candle_inside_bar_bear:'<b>Inside Bar + Bearish Close</b> · Inside bar where the close is in the lower half of the bar\'s range · Sellers maintained pressure during consolidation · Higher probability of downside breakout — most relevant in downtrends or at resistance',
  candle_double_inside:'<b>Double Inside Bar</b> · Two consecutive inside bars — the most compressed volatility pattern available from daily data · Extremely rare (~1-2% of sessions) · When this fires, the subsequent breakout is typically large and fast · Best combined with 52W consolidation for pre-breakout timing',
  // StockEdge premium
  candle_marubozu_bull:'<b>White Marubozu</b> · StockEdge Club scan · Bullish candle where the body is ≥90% of the entire High-Low range — virtually no upper or lower shadows · Opens at or near the low, closes at or near the high · Signals overwhelming, uninterrupted buying pressure with no seller resistance at any point in the session · Often seen at the start of strong trends',
  candle_marubozu_bear:'<b>Black Marubozu</b> · StockEdge Club scan · Bearish candle where the body is ≥90% of the High-Low range — no shadows · Opens at or near the high, closes at or near the low · Signals pure, relentless selling pressure throughout the session · Exit signal for longs; potential short entry at next resistance',
  candle_spinning_top:'<b>Spinning Top</b> · Small real body (< 30% of range) with long shadows on BOTH upper and lower sides · Represents complete indecision — bulls and bears fought for control, neither won · Most significant at swing highs (warns of exhaustion) or swing lows (warns of potential reversal) · Combine with volume — spinning top on high volume = institutional indecision worth watching',
  candle_tweezer_bottom:'<b>Tweezer Bottom</b> · Two consecutive candles that share the same (or very nearly the same) low · The market tested a specific support price twice and rejected it both times · Institutions are defending that price level · Best when the second candle is bullish and appears at a known support zone · Often marks the exact bottom of a short-term move',
  candle_tweezer_top:'<b>Tweezer Top</b> · Two consecutive candles sharing the same (or very nearly the same) high · Resistance was tested twice and rejected both times · Most reliable when the second candle is bearish and occurs at a known resistance level (prior high, pivot, order block) · A third rejection from the same level confirms distribution',
  candle_outside_bull:'<b>Bullish Outside Bar</b> · Current bar has a higher high AND lower low than the prior bar, AND closes bullishly · The bulls completely dominated the session, reversing the prior day\'s range and closing near the high · More aggressive than Bullish Engulfing because the range itself expands both ways · Signals strong momentum shift — particularly reliable when appearing after a downtrend',
  candle_outside_bear:'<b>Bearish Outside Bar</b> · Current bar engulfs prior bar range and closes bearishly · Bears completely overwhelmed bulls · Strong reversal or continuation signal depending on context · At resistance after an uptrend = high-conviction reversal; mid-trend downtrend = acceleration',
  // Gap scans
  candle_gap_up:'<b>Gap Up Opening (&gt;1%)</b> · Chartink most-popular scan · Today\'s opening price is more than 1% above yesterday\'s closing price · A true price gap shows that buy orders overnight overwhelmed sell orders · Gaps have strong statistical tendency to either fill (revert) or accelerate — the direction depends on the catalyst · Combine with volume to determine intent: gap + high volume = institutional, gap + low volume = retail FOMO',
  candle_gap_up3:'<b>Strong Gap Up (&gt;3%)</b> · Chartink style: the classic 3%+ gap up alert · Stock opened 3% or more above yesterday\'s close — a significant overnight move · Typically driven by earnings, results, corporate actions, or sector news · Gap + consolidation above the gap = bullish continuation; gap + immediate fill = potential mean reversion short',
  candle_gap_down:'<b>Gap Down Opening (&lt;-1%)</b> · Today\'s open is 1%+ below yesterday\'s close · Overnight selling overwhelmed buying — negative catalyst or fear · Gap downs can be mean-reversion opportunities (buy the dip) or continuation signals (sell the gap) · Always check the news catalyst: earnings miss or promoter selling = continuation; market-wide gap = possible bounce',
  candle_gap_down3:'<b>Strong Gap Down (&lt;-3%)</b> · Chartink 3%+ gap down alert · Aggressive overnight sell-off of 3%+ · High-information content event — typically result of significant negative news · Watch the first 30 minutes: if price recovers above the gap open, institutional buyers have stepped in; if it continues lower, the selling is genuine',
  ema_c921_bull:'<b>EMA 9/21 Bull Cross</b> · Chartink real-time alert · 9-period EMA just crossed above 21-period EMA · Short-term trend shift to bullish · Best when ADX ≥25',
  ema_c921_bear:'<b>EMA 9/21 Bear Cross</b> · 9 EMA just crossed below 21 EMA · Short-term trend turning bearish · Exit longs, reduce exposure',
  ema_c2050_bull:'<b>EMA 20/50 Bull Cross</b> · Chartink premium alert · Medium-term trend confirmed bullish · More reliable than 9/21, fewer whipsaws',
  ema_c2050_bear:'<b>EMA 20/50 Bear Cross</b> · Medium-term trend turning bearish · Portfolio-level signal to reduce equity exposure',
  ema_fan_bull:'<b>EMA Fan Bullish</b> · Price > EMA9 > EMA21 > EMA50 — all EMAs stacked in perfect bullish order · Trendlyne momentum hallmark · Strong trend confirmation',
  ema_fan_bear:'<b>EMA Fan Bearish</b> · Price < EMA9 < EMA21 < EMA50 — all EMAs in bearish stack · Strong downtrend — avoid longs',
  ema_recent_921:'<b>Fresh 9/21 Bull Cross</b> · EMA 9 crossed above 21 within last 8 bars — still early in the move · Best entry window for short-term momentum',
  mom_high:'<b>High Composite Momentum</b> · Trendlyne-style weighted score (1M×15%+3M×25%+6M×30%+12M×30%) ≥ 20% · Top-tier price performance across all time horizons',
  mom_positive_all:'<b>All-Timeframe Positive</b> · 1M, 3M, 6M, and 12M returns all positive — consistent performer across every lookback · Classic bull market leader',
  mom_roc10_bull:'<b>Fast Mover (ROC 10-day >5%)</b> · Chartink scan · 10-day Rate of Change above 5% · Strong recent momentum — buyers are aggressive',
  mom_roc21_bull:'<b>Monthly Mover (ROC 21-day >8%)</b> · 21-day ROC above 8% · Sustained buying pressure over the past month',
  mom_neg:'<b>Negative Composite Momentum</b> · Weighted score negative — stock is underperforming across all timeframes · Avoid long positions',
  seq_hh3:'<b>3 Higher Highs</b> · ScanX/Chartink · Three consecutive days where each high exceeds the prior day\'s high · Uptrend structure forming — institutional buying',
  seq_hh5:'<b>5 Higher Highs</b> · Five consecutive higher highs — sustained uptrend with consistent bid · Very reliable for momentum entry',
  seq_hl3:'<b>3 Higher Lows</b> · Three consecutive higher lows — buyers are defending progressively higher prices · Classic uptrend structure (HH+HL)',
  seq_hc3_vol:'<b>3 Higher Closes + Volume</b> · Three consecutive higher closes WITH volume above 20-day average · Best quality accumulation signal — institutions buying consistently',
  seq_ll3:'<b>3 Lower Lows</b> · Three consecutive lower lows — downtrend structure · Avoid longs, wait for reversal confirmation',
  vol_acc3:'<b>3-Day Accumulation Streak</b> · StockEdge signature · 3+ days of both rising price AND rising volume — classic institutional steady accumulation pattern',
  vol_acc5:'<b>5-Day Accumulation Streak</b> · Five consecutive up-days with rising volume — sustained institutional buying · Very high conviction setup',
  vol_quiet:'<b>Quiet Accumulation</b> · StockEdge Club · Price flat (±2%) but volume consistently 1.3× average for 3 days — institutions accumulating without moving price · Pre-breakout signal',
  consol_10:'<b>10-Day Tight Coil</b> · Chartink pre-breakout scan · Within 5% of 52W high AND last 10-day range is less than 5% of the high · Classic pre-breakout compression like VCP',
  consol_15:'<b>Extended Base (15-day)</b> · Within 7% of 52W high with volume drying up below 80% of avg — stock coiling for a move · Watch for high-volume breakout above the high',
  ma_max4:'<b>Above All 4 MAs</b> · StockEdge price scan · Price above 20, 50, 100, AND 200 SMA simultaneously · Maximum bullish alignment — stock is above all significant moving average levels',
  ma_all_bull:'<b>Full MA Stack Bullish</b> · Price > MA20 > MA50 > MA100 > MA200 — perfect stacking in descending order · The "holy grail" of MA alignment — Minervini Trend Template compatible',
  ma_above3:'<b>Above 20/50/100 SMA</b> · Less strict version · All three key MAs confirmed bullish · Good for mid-cap and small-cap where 200 SMA may lag',
};

function updateTIInfo(){
  const s=document.getElementById('ti-strat').value;
  setFbar(`<span style="color:#ff9500">🏅 India Pro</span> · ${TI_INFO[s]||s}`);
}

function scanTI(){
  const strat=document.getElementById('ti-strat').value;
  
  const rsMin=parseInt(document.getElementById('ti-rs').value);
  const prMin=parseFloat(document.getElementById('ti-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('ti-pmax').value)||Infinity;
  updateTIInfo();
  rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.ti)continue;
    const tiData=s.ti;
    const cd=tiData.candle||{}; const em=tiData.ema||{}; const mo=tiData.mom||{};
    const sq=tiData.seq||{}; const vb=tiData.volbld||{}; const co=tiData.consol||{};
    const ma=tiData.ma||{};
    let matched=false; let sig=''; let extra={};

    // Candlestick
    if(strat==='candle_hammer')     {matched=!!cd.hammer;      sig='🔨 Hammer';       extra={close:s.price};}
    if(strat==='candle_inv_hammer') {matched=!!cd.inv_hammer;  sig='🔼 Inv Hammer';   extra={close:s.price};}
    if(strat==='candle_bull_eng')   {matched=!!cd.bull_eng;    sig='📗 Bull Eng';     extra={close:s.price};}
    if(strat==='candle_morning_star'){matched=!!cd.morning_star;sig='🌅 Morn Star';   extra={close:s.price};}
    if(strat==='candle_piercing')   {matched=!!cd.piercing;    sig='🔵 Piercing';     extra={close:s.price};}
    if(strat==='candle_bull_har')   {matched=!!cd.bull_har;    sig='📘 Bull Harami';  extra={close:s.price};}
    if(strat==='candle_tws')        {matched=!!cd.tws;         sig='🕯🕯🕯 3W Sold';  extra={close:s.price};}
    if(strat==='candle_doji')       {matched=!!cd.doji;        sig='➕ Doji';         extra={close:s.price};}
    if(strat==='candle_shooting_star'){matched=!!cd.shooting_star;sig='🔻 Shoot Star';extra={close:s.price};}
    if(strat==='candle_bear_eng')   {matched=!!cd.bear_eng;    sig='📕 Bear Eng';     extra={close:s.price};}
    if(strat==='candle_dark_cloud') {matched=!!cd.dark_cloud;  sig='☁️ Dark Cloud';   extra={close:s.price};}
    if(strat==='candle_evening_star'){matched=!!cd.evening_star;sig='🌆 Eve Star';    extra={close:s.price};}
    if(strat==='candle_bear_har')   {matched=!!cd.bear_har;    sig='📙 Bear Harami';  extra={close:s.price};}
    if(strat==='candle_tbc')        {matched=!!cd.tbc;              sig='🕯🕯🕯 3B Crow';    extra={close:s.price};}

    // Inside Bar (Chartink #1 scan)
    if(strat==='candle_inside_bar')      {matched=!!cd.inside_bar;       sig='📦 Inside Bar';    extra={h:s.price,prev_h:s.w52h};}
    if(strat==='candle_inside_bar_bull') {matched=!!cd.inside_bar_bull;  sig='📦↑ IB Bull';     extra={close:s.price};}
    if(strat==='candle_inside_bar_bear') {matched=!!cd.inside_bar_bear;  sig='📦↓ IB Bear';     extra={close:s.price};}
    if(strat==='candle_double_inside')   {matched=!!cd.double_inside;    sig='📦📦 2x Inside';  extra={close:s.price};}

    // StockEdge premium patterns
    if(strat==='candle_marubozu_bull')   {matched=!!cd.marubozu_bull;    sig='💚 White Marub';  extra={close:s.price};}
    if(strat==='candle_marubozu_bear')   {matched=!!cd.marubozu_bear;    sig='🔴 Black Marub';  extra={close:s.price};}
    if(strat==='candle_spinning_top')    {matched=!!cd.spinning_top;     sig='🔄 Spin Top';     extra={close:s.price};}
    if(strat==='candle_tweezer_bottom')  {matched=!!cd.tweezer_bottom;   sig='🔻🔻 Twz Bot';   extra={close:s.price};}
    if(strat==='candle_tweezer_top')     {matched=!!cd.tweezer_top;      sig='🔺🔺 Twz Top';   extra={close:s.price};}
    if(strat==='candle_outside_bull')    {matched=!!cd.outside_bar_bull; sig='⬆ Outside Bull'; extra={close:s.price};}
    if(strat==='candle_outside_bear')    {matched=!!cd.outside_bar_bear; sig='⬇ Outside Bear'; extra={close:s.price};}

    // Gap scans (Chartink most-used)
    if(strat==='candle_gap_up')    {matched=!!cd.gap_up;    sig=`⬆ Gap +${cd.gap_pct}%`;  extra={gap:cd.gap_pct+'%'};}
    if(strat==='candle_gap_up3')   {matched=!!cd.gap_up3;   sig=`🚀 Gap +${cd.gap_pct}%`; extra={gap:cd.gap_pct+'%'};}
    if(strat==='candle_gap_down')  {matched=!!cd.gap_down;  sig=`⬇ Gap ${cd.gap_pct}%`;   extra={gap:cd.gap_pct+'%'};}
    if(strat==='candle_gap_down3') {matched=!!cd.gap_down3; sig=`💥 Gap ${cd.gap_pct}%`;  extra={gap:cd.gap_pct+'%'};}

    // EMA
    if(strat==='ema_c921_bull')  {matched=!!em.c921_bull;  sig='📈 EMA 9↑21'; extra={e9:em.e9,e21:em.e21,e50:em.e50};}
    if(strat==='ema_c921_bear')  {matched=!!em.c921_bear;  sig='📉 EMA 9↓21'; extra={e9:em.e9,e21:em.e21};}
    if(strat==='ema_c2050_bull') {matched=!!em.c2050_bull; sig='📈 EMA 20↑50'; extra={e21:em.e21,e50:em.e50};}
    if(strat==='ema_c2050_bear') {matched=!!em.c2050_bear; sig='📉 EMA 20↓50'; extra={e21:em.e21,e50:em.e50};}
    if(strat==='ema_fan_bull')   {matched=!!em.fan_bull;   sig='🚀 EMA Fan ↑'; extra={e9:em.e9,e21:em.e21,e50:em.e50};}
    if(strat==='ema_fan_bear')   {matched=!!em.fan_bear;   sig='📉 EMA Fan ↓'; extra={e9:em.e9,e21:em.e21,e50:em.e50};}
    if(strat==='ema_recent_921') {matched=!!em.recent_921; sig='⚡ Fresh 9↑21'; extra={e9:em.e9,e21:em.e21};}

    // Momentum
    if(strat==='mom_high')        {matched=(mo.score||0)>=20;       sig='🔥 High Mom';   extra={score:mo.score+'%',r1m:mo.r1m+'%',r3m:mo.r3m+'%',r6m:mo.r6m+'%'};}
    if(strat==='mom_positive_all'){matched=!!mo.positive_all;       sig='📈 All+ve';     extra={r1m:mo.r1m+'%',r3m:mo.r3m+'%',r6m:mo.r6m+'%',r12m:mo.r12m+'%'};}
    if(strat==='mom_roc10_bull')  {matched=(mo.roc10||0)>5;         sig='⚡ ROC10>5%';   extra={roc10:mo.roc10+'%',r1m:mo.r1m+'%'};}
    if(strat==='mom_roc21_bull')  {matched=(mo.roc21||0)>8;         sig='🚀 ROC21>8%';   extra={roc21:mo.roc21+'%',r3m:mo.r3m+'%'};}
    if(strat==='mom_neg')         {matched=(mo.score||0)<0;         sig='📉 Neg Mom';    extra={score:mo.score+'%',r3m:mo.r3m+'%',r12m:mo.r12m+'%'};}

    // Sequential
    if(strat==='seq_hh3')    {matched=!!sq.hh3; sig='↗ 3 Higher H';  extra={vol_conf:sq.vol_confirm?'Yes':'No'};}
    if(strat==='seq_hh5')    {matched=!!sq.hh5; sig='↗↗ 5 Higher H'; extra={};}
    if(strat==='seq_hl3')    {matched=!!sq.hl3; sig='↗ 3 Higher L';  extra={};}
    if(strat==='seq_hc3_vol'){matched=!!sq.hc3&&!!sq.vol_confirm; sig='📈 3HC+Vol'; extra={vol_confirm:'Yes'};}
    if(strat==='seq_ll3')    {matched=!!sq.ll3; sig='↘ 3 Lower L';   extra={};}

    // Volume buildup
    if(strat==='vol_acc3')  {matched=!!vb.acc3;    sig='📦 Acc 3d';    extra={streak:vb.streak+'d',avg20:vb.avg20};}
    if(strat==='vol_acc5')  {matched=!!vb.acc5;    sig='📦📦 Acc 5d';  extra={streak:vb.streak+'d',avg20:vb.avg20};}
    if(strat==='vol_quiet') {matched=!!vb.quiet_acc;sig='🔕 Quiet Acc';extra={avg20:vb.avg20};}

    // Consolidation
    if(strat==='consol_10') {matched=!!co.coil10; sig='🎯 Coil 10d'; extra={dist:co.dist_pct+'%',rng10:co.last10_rng+'%',w52h:co.w52h};}
    if(strat==='consol_15') {matched=!!co.coil15; sig='🎯 Coil 15d'; extra={dist:co.dist_pct+'%',vol_dry:co.vol_dry+'×',w52h:co.w52h};}

    // MA alignment
    if(strat==='ma_max4')   {matched=!!ma.max4;    sig='🔝 Above All4'; extra={ma20:ma.ma20,ma50:ma.ma50,ma100:ma.ma100,ma200:ma.ma200};}
    if(strat==='ma_all_bull'){matched=!!ma.all_bull;sig='🏆 MA Stack';  extra={ma20:ma.ma20,ma50:ma.ma50,ma100:ma.ma100,ma200:ma.ma200};}
    if(strat==='ma_above3') {matched=!!ma.above_all3;sig='📊 Above3MA'; extra={ma20:ma.ma20,ma50:ma.ma50,ma100:ma.ma100};}

    // Heikin Ashi
    const ha=tiData.ha||{};
    if(strat==='ha_bull')       {matched=!!ha.bull;        sig='🟢 HA Bull';      extra={consec:ha.consec_bull};}
    if(strat==='ha_bear')       {matched=!!ha.bear;        sig='🔴 HA Bear';      extra={consec:ha.consec_bear};}
    if(strat==='ha_strong_bull'){matched=!!ha.strong_bull; sig='💚 HA Str Bull';  extra={consec:ha.consec_bull};}
    if(strat==='ha_strong_bear'){matched=!!ha.strong_bear; sig='❤️ HA Str Bear';  extra={consec:ha.consec_bear};}
    if(strat==='ha_doji')       {matched=!!ha.ha_doji;     sig='➕ HA Doji';      extra={};}
    if(strat==='ha_rev_bull')   {matched=!!ha.rev_bull;    sig='⚡ HA Rev↑';     extra={};}
    if(strat==='ha_rev_bear')   {matched=!!ha.rev_bear;    sig='⚡ HA Rev↓';     extra={};}
    if(strat==='ha_consec5')    {matched=(ha.consec_bull||0)>=5; sig=`🔥 HA ${ha.consec_bull||0}× Bull`; extra={consec:ha.consec_bull};}

    // Hull MA
    const hm=tiData.hma||{};
    if(strat==='hma_bull')      {matched=!!hm.bull;       sig='📈 HMA Bull';    extra={hma:hm.hma,dist:hm.dist+'%'};}
    if(strat==='hma_bear')      {matched=!!hm.bear;       sig='📉 HMA Bear';    extra={hma:hm.hma,dist:hm.dist+'%'};}
    if(strat==='hma_flip_bull') {matched=!!hm.flip_bull;  sig='⚡ HMA Flip↑';  extra={hma:hm.hma};}
    if(strat==='hma_flip_bear') {matched=!!hm.flip_bear;  sig='⚡ HMA Flip↓';  extra={hma:hm.hma};}

    // Keltner
    const kt=tiData.kelt||{};
    if(strat==='kelt_above')    {matched=!!kt.above;      sig='🚀 Kelt Above';  extra={upper:kt.upper,pctb:kt.pct_b+'%'};}
    if(strat==='kelt_below')    {matched=!!kt.below;      sig='⬇ Kelt Below';  extra={lower:kt.lower,pctb:kt.pct_b+'%'};}
    if(strat==='kelt_cross_up') {matched=!!kt.cross_up;   sig='⚡ Kelt Cross↑'; extra={upper:kt.upper};}
    if(strat==='kelt_cross_dn') {matched=!!kt.cross_dn;   sig='⚡ Kelt Cross↓'; extra={lower:kt.lower};}
    if(strat==='kelt_squeeze')  {matched=!!kt.squeezed;   sig='🗜 Kelt Squeeze'; extra={upper:kt.upper,lower:kt.lower};}

    // MFI
    const mf=tiData.mfi||{};
    if(strat==='mfi_os')        {matched=!!mf.oversold;   sig='🟢 MFI OS';     extra={mfi:mf.mfi};}
    if(strat==='mfi_ob')        {matched=!!mf.overbought; sig='🔴 MFI OB';     extra={mfi:mf.mfi};}
    if(strat==='mfi_bull_div')  {matched=!!mf.bull_div;   sig='↗ MFI Bull Div'; extra={mfi:mf.mfi};}
    if(strat==='mfi_bear_div')  {matched=!!mf.bear_div;   sig='↘ MFI Bear Div'; extra={mfi:mf.mfi};}

    // CCI
    const cc=tiData.cci||{};
    if(strat==='cci_os')        {matched=!!cc.oversold;   sig='🟢 CCI OS';     extra={cci:cc.cci};}
    if(strat==='cci_ob')        {matched=!!cc.overbought; sig='🔴 CCI OB';     extra={cci:cc.cci};}
    if(strat==='cci_zero_up')   {matched=!!cc.zero_up;    sig='📈 CCI 0↑';     extra={cci:cc.cci};}
    if(strat==='cci_zero_dn')   {matched=!!cc.zero_dn;    sig='📉 CCI 0↓';     extra={cci:cc.cci};}
    if(strat==='cci_os_exit')   {matched=!!cc.os_exit;    sig='⚡ CCI OS Exit'; extra={cci:cc.cci};}
    if(strat==='cci_ob_exit')   {matched=!!cc.ob_exit;    sig='⚡ CCI OB Exit'; extra={cci:cc.cci};}

    // Linear Regression
    const lr=tiData.linreg||{};
    if(strat==='lr_above_chan')  {matched=!!lr.above_chan;   sig='📈 LR Above+2σ'; extra={dev:lr.dev,fit:lr.curr_fit};}
    if(strat==='lr_below_chan')  {matched=!!lr.below_chan;   sig='📉 LR Below-2σ'; extra={dev:lr.dev,fit:lr.curr_fit};}
    if(strat==='lr_slope_bull')  {matched=!!lr.slope_bull;  sig='↗ LR Slope↑';   extra={slope:lr.slope,dev:lr.dev};}
    if(strat==='lr_slope_bear')  {matched=!!lr.slope_bear;  sig='↘ LR Slope↓';   extra={slope:lr.slope,dev:lr.dev};}
    if(strat==='lr_mean_rev')    {matched=lr.dev&&Math.abs(lr.dev)>=2; sig='⚖ LR ±2σ';    extra={dev:lr.dev,fit:lr.curr_fit};}

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig, extra, strat, _tab:'ti'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── 📐 NOISELESS (Definedge P&F / Renko / 3-Line Break / RRG / EW) ─────────
const NL_INFO={
  pnf_dbl_top:'<b>P&F Double Top Breakout</b> · Current X column rises above the HIGH of the previous X column · The most classic P&F buy signal · Definedge TradePoint default alert · Use 1% box size on NSE stocks',
  pnf_dbl_bot:'<b>P&F Double Bottom Breakdown</b> · Current O column falls below the LOW of the previous O column · Classic P&F sell signal · Highly reliable in trending markets',
  pnf_tpl_top:'<b>P&F Triple Top Breakout</b> · Breaks above TWO previous X column highs (all at the same level) · Stronger and rarer than Double Top · Institutional resistance level has been decisively overcome',
  pnf_tpl_bot:'<b>P&F Triple Bottom Breakdown</b> · Breaks below two previous O column lows at the same level · Very strong sell signal · Often marks the start of a sustained downtrend',
  pnf_bull_cat:'<b>P&F Bullish Catapult</b> · Three consecutive X columns each making progressively higher highs · The "launch pad" formation — sustained demand overwhelms supply across multiple rallies',
  pnf_bear_cat:'<b>P&F Bearish Catapult</b> · Three O columns making progressively lower lows · Sustained selling momentum with no support found',
  pnf_high_pole:'<b>P&F High Pole Warning</b> · Unusually tall X column (≥5 boxes) followed by an O column that retraces more than 50% of it · Warning that bulls are exhausted — proceed with caution on longs',
  pnf_low_pole:'<b>P&F Low Pole Warning</b> · Unusually tall O column (≥5 boxes) followed by an X column recovering more than 50% · Bears are exhausted — potential bottoming signal',
  pnf_asc_tri:'<b>P&F Ascending Triangle</b> · X column tops at the same price level (resistance) while O column lows are rising (higher lows) · Classic accumulation pattern — breakout above resistance is likely',
  pnf_desc_tri:'<b>P&F Descending Triangle</b> · O column lows at the same level (support) while X column highs are falling (lower highs) · Distribution pattern — breakdown below support is likely',
  pnf_lt_bull:'<b>P&F Long-Tail Bull Reversal</b> · A very long O column (≥8 boxes = heavy selling) followed by an X reversal · The extreme selling was a capitulation event — often marks major lows',
  pnf_lt_bear:'<b>P&F Long-Tail Bear Reversal</b> · A very long X column (≥8 boxes = extreme buying) followed by an O reversal · Buying climax — often marks major highs',
  pnf_in_x:'<b>P&F In X Column</b> · Current P&F chart is in an X (rising) column · Bullish trend on noiseless chart basis · Definedge uses this as a primary trend filter',
  pnf_in_o:'<b>P&F In O Column</b> · Current P&F chart is in an O (falling) column · Bearish noiseless trend · Avoid longs when in O column',
  renko_bull:'<b>Renko Bull</b> · Most recent Renko brick is UP (green) · Trend is bullish on noiseless basis · ATR-based brick size filters minor price noise',
  renko_bear:'<b>Renko Bear</b> · Most recent Renko brick is DOWN (red) · Trend is bearish · Avoid longs',
  renko_rev_bull:'<b>Renko Reversed Bullish</b> · Just changed from DOWN to UP brick · Fresh trend reversal — early entry opportunity · Most reliable when after 3+ consecutive down bricks',
  renko_rev_bear:'<b>Renko Reversed Bearish</b> · Just switched from UP to DOWN · Trend reversal alert — exit longs',
  renko_3:'<b>Renko 3-Brick Run</b> · 3 consecutive same-direction bricks · Trend is establishing · Definedge Zone platform highlights 3-brick sequences as minimum trend confirmation',
  renko_5:'<b>Renko 5 ka Punch</b> · Definedge signature pattern — 5 consecutive same-direction bricks · "The Punch" signals a strong, sustained trend with significant institutional participation · High-conviction momentum signal',
  renko_dbl_top:'<b>Renko Double Top</b> · Two up-runs reached approximately the same high level · Resistance zone identified on the noiseless chart · Watch for breakdown or breakout',
  renko_dbl_bot:'<b>Renko Double Bottom</b> · Two down-runs reached the same low level · Support zone on noiseless chart · Potential reversal if held',
  lb3_bull:'<b>3-Line Break Bull</b> · Current chart is in a white (up) line · Bullish noiseless trend · Requires stronger evidence to reverse than a standard daily chart',
  lb3_bear:'<b>3-Line Break Bear</b> · Current chart is in a black (down) line · Bearish noiseless trend',
  lb3_rev_bull:'<b>3-Line Break Reversal UP</b> · Chart just turned white (up) after black lines — requires price to exceed the high of the last 3 black lines · Strong reversal confirmation signal',
  lb3_rev_bear:'<b>3-Line Break Reversal DOWN</b> · Chart just turned black (down) · Price fell below the low of the last 3 white lines · Strong bearish reversal signal',
  lb3_consec:'<b>3-Line Break 5+ Consecutive</b> · Five or more consecutive same-direction lines · Very strong noiseless trend with consistent price action',
  rrg_leading:'<b>RRG Leading Quadrant</b> · RS-Ratio > 100 AND RS-Momentum > 100 · Stock is outperforming Nifty AND accelerating · Top-right quadrant = institutional favorite · Best entry timing',
  rrg_weakening:'<b>RRG Weakening</b> · RS-Ratio > 100 (still outperforming) but RS-Momentum < 100 (slowing down) · Bottom-right quadrant · Still a leader but losing steam — consider reducing or exit on rotation to lagging',
  rrg_improving:'<b>RRG Improving</b> · RS-Ratio < 100 (underperforming) but RS-Momentum > 100 (picking up) · Top-left quadrant · Early rotation into this sector/stock — buying opportunity before it enters leading quadrant',
  rrg_lagging:'<b>RRG Lagging</b> · RS-Ratio < 100 AND RS-Momentum < 100 · Bottom-left quadrant = weakest stocks · Avoid longs; institutional money is leaving this stock',
  ew_w5_up:'<b>EW Wave 5 Bullish Impulse</b> · Approximation — zigzag pivot analysis shows a completed 5-wave structure with wave 3 strongest · Note: this is a simplified algorithmic detection, not a professional EW count · Use as confluence, not standalone signal',
  ew_w5_dn:'<b>EW Wave 5 Bearish Impulse</b> · Simplified detection of a 5-wave bearish impulse pattern · Wave 3 is the strongest wave down · Potential completion of a bearish impulse — watch for corrective bounce',
  ew_w3_ext:'<b>EW Wave 3 Extension</b> · Wave 3 is at least 161.8% of Wave 1 (Golden Ratio extension) · Extended wave 3s are the strongest, fastest moves in Elliott Wave · If wave 3 is still in progress, significant upside potential remains',
  ew_abc:'<b>EW ABC Correction Done</b> · Three-wave A-B-C correction pattern appears to be completing · C wave ≤ A wave length suggests correction is exhausted · Potential resumption of the primary trend',
};

function updateNLInfo(){
  const s=document.getElementById('nl-strat').value;
  const isRrg=s.startsWith('rrg_');
  const note=isRrg?' <span style="color:var(--warn);font-size:9px">· Requires --nifty flag</span>':'';
  const isEw=s.startsWith('ew_');
  const ewNote=isEw?' <span style="color:var(--mu);font-size:9px">· Simplified approximation</span>':'';
  setFbar(`<span style="color:#e879f9">📐 Noiseless</span> · ${NL_INFO[s]||s}${note}${ewNote}`);
}

function scanNL(){
  const strat=document.getElementById('nl-strat').value;
  
  const rsMin=parseInt(document.getElementById('nl-rs').value);
  const prMin=parseFloat(document.getElementById('nl-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('nl-pmax').value)||Infinity;
  updateNLInfo(); rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.nl)continue;
    const pnf=s.nl.pnf||{}; const rk=s.nl.renko||{}; const lb=s.nl.lb3||{};
    const rrg=s.nl.rrg||{}; const ew=s.nl.ew||{};
    let matched=false; let sig=''; let extra={};
    // P&F
    if(strat==='pnf_dbl_top') {matched=!!pnf.dbl_top;  sig='📈 P&F Dbl Top BO'; extra={cols:pnf.n_cols,box:pnf.box};}
    if(strat==='pnf_dbl_bot') {matched=!!pnf.dbl_bot;  sig='📉 P&F Dbl Bot BD'; extra={cols:pnf.n_cols,box:pnf.box};}
    if(strat==='pnf_tpl_top') {matched=!!pnf.tpl_top;  sig='📈📈 P&F Tpl Top';  extra={cols:pnf.n_cols,box:pnf.box};}
    if(strat==='pnf_tpl_bot') {matched=!!pnf.tpl_bot;  sig='📉📉 P&F Tpl Bot';  extra={cols:pnf.n_cols};}
    if(strat==='pnf_bull_cat'){matched=!!pnf.bull_cat; sig='🚀 P&F Bull Cat';   extra={cols:pnf.n_cols};}
    if(strat==='pnf_bear_cat'){matched=!!pnf.bear_cat; sig='⬇ P&F Bear Cat';   extra={cols:pnf.n_cols};}
    if(strat==='pnf_high_pole'){matched=!!pnf.high_pole;sig='⚠ P&F Hi Pole';   extra={boxes:pnf.curr_boxes};}
    if(strat==='pnf_low_pole') {matched=!!pnf.low_pole; sig='⚠ P&F Lo Pole';   extra={boxes:pnf.curr_boxes};}
    if(strat==='pnf_asc_tri') {matched=!!pnf.asc_tri;  sig='△ P&F Asc Tri';   extra={cols:pnf.n_cols};}
    if(strat==='pnf_desc_tri'){matched=!!pnf.desc_tri; sig='▽ P&F Desc Tri';  extra={cols:pnf.n_cols};}
    if(strat==='pnf_lt_bull') {matched=!!pnf.lt_bull;  sig='↗ P&F LT Bull';   extra={cols:pnf.n_cols};}
    if(strat==='pnf_lt_bear') {matched=!!pnf.lt_bear;  sig='↘ P&F LT Bear';   extra={cols:pnf.n_cols};}
    if(strat==='pnf_in_x')    {matched=!!pnf.in_x;     sig='✗ P&F In X';      extra={boxes:pnf.curr_boxes,cols:pnf.n_cols};}
    if(strat==='pnf_in_o')    {matched=!!pnf.in_o;     sig='○ P&F In O';      extra={boxes:pnf.curr_boxes,cols:pnf.n_cols};}
    // Renko
    if(strat==='renko_bull')    {matched=!!rk.bull;       sig='🟢 Renko Bull';   extra={consec:rk.consec,brick:rk.brick};}
    if(strat==='renko_bear')    {matched=!!rk.bear;       sig='🔴 Renko Bear';   extra={consec:rk.consec,brick:rk.brick};}
    if(strat==='renko_rev_bull'){matched=!!rk.just_rev&&!!rk.bull; sig='⚡ Renko Rev↑'; extra={consec:rk.consec};}
    if(strat==='renko_rev_bear'){matched=!!rk.just_rev&&!!rk.bear; sig='⚡ Renko Rev↓'; extra={consec:rk.consec};}
    if(strat==='renko_3')       {matched=!!rk.punch3;     sig='🔥 Renko 3';     extra={consec:rk.consec,brick:rk.brick};}
    if(strat==='renko_5')       {matched=!!rk.punch5;     sig='🥊 5 ka Punch';  extra={consec:rk.consec,brick:rk.brick};}
    if(strat==='renko_dbl_top') {matched=!!rk.dbl_top;    sig='🔝 Renko Dbl T'; extra={brick:rk.brick};}
    if(strat==='renko_dbl_bot') {matched=!!rk.dbl_bot;    sig='🔻 Renko Dbl B'; extra={brick:rk.brick};}
    // 3-Line Break
    if(strat==='lb3_bull')   {matched=!!lb.bull;       sig='📗 3LB Bull';     extra={lines:lb.n_lines,consec:lb.consec};}
    if(strat==='lb3_bear')   {matched=!!lb.bear;       sig='📕 3LB Bear';     extra={lines:lb.n_lines,consec:lb.consec};}
    if(strat==='lb3_rev_bull'){matched=!!lb.rev_bull;  sig='⚡ 3LB Rev↑';    extra={lines:lb.n_lines};}
    if(strat==='lb3_rev_bear'){matched=!!lb.rev_bear;  sig='⚡ 3LB Rev↓';    extra={lines:lb.n_lines};}
    if(strat==='lb3_consec') {matched=(lb.consec||0)>=5;sig=`🔁 3LB ${lb.consec||0}+`; extra={consec:lb.consec,lines:lb.n_lines};}
    // RRG
    if(strat==='rrg_leading')  {matched=!!rrg.leading;  sig='🌟 RRG Leading';  extra={rr:rrg.rr,rm:rrg.rm};}
    if(strat==='rrg_weakening'){matched=!!rrg.weakening;sig='⬇ RRG Weaken';   extra={rr:rrg.rr,rm:rrg.rm};}
    if(strat==='rrg_improving'){matched=!!rrg.improving;sig='↗ RRG Improv';   extra={rr:rrg.rr,rm:rrg.rm};}
    if(strat==='rrg_lagging')  {matched=!!rrg.lagging;  sig='📉 RRG Lagging'; extra={rr:rrg.rr,rm:rrg.rm};}
    // Elliott Wave
    if(strat==='ew_w5_up') {matched=!!ew.w5_up;    sig='🌊 EW W5 Up';    extra={pivots:ew.n_pivots};}
    if(strat==='ew_w5_dn') {matched=!!ew.w5_dn;    sig='🌊 EW W5 Down';  extra={pivots:ew.n_pivots};}
    if(strat==='ew_w3_ext'){matched=!!ew.wave3_ext; sig='⚡ EW W3 Ext';   extra={pivots:ew.n_pivots};}
    if(strat==='ew_abc')   {matched=!!ew.abc_done;  sig='🔁 EW ABC Done'; extra={pivots:ew.n_pivots};}

    // Kagi
    const kg=s.nl.kagi||{};
    if(strat==='kagi_yang')    {matched=!!kg.yang;     sig='🎋 Kagi Yang';    extra={consec:kg.consec,lines:kg.n_lines};}
    if(strat==='kagi_yin')     {matched=!!kg.yin;      sig='🎋 Kagi Yin';     extra={consec:kg.consec,lines:kg.n_lines};}
    if(strat==='kagi_rev_yang'){matched=!!kg.rev_yang; sig='⚡ Kagi→Yang';    extra={lines:kg.n_lines};}
    if(strat==='kagi_rev_yin') {matched=!!kg.rev_yin;  sig='⚡ Kagi→Yin';     extra={lines:kg.n_lines};}
    if(strat==='kagi_shoulder'){matched=!!kg.shoulder; sig='💪 Kagi Shoulder'; extra={consec:kg.consec};}
    if(strat==='kagi_waist')   {matched=!!kg.waist;    sig='⬇ Kagi Waist';   extra={consec:kg.consec};}
    if(strat==='kagi_consec')  {matched=(kg.consec||0)>=3&&!!kg.yang; sig=`🔥 Kagi ${kg.consec||0}× Yang`; extra={consec:kg.consec};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'nl'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── 🧪 EXPERIMENTAL — Volume Spread Analysis ──────────────────────────────
const XP_INFO={
  vsa_big_bull:'<b>Big Volume Bull Candle</b> · Ultra-high volume (≥2.5× avg) + wide spread + close in upper 60% of range · Classic VSA sign of institutional accumulation · Smart money absorbed all available supply AND pushed price up · Tom Williams called this "effort equals result" — the effort was justified',
  vsa_big_bear:'<b>Big Volume Bear Candle</b> · Ultra-high volume + wide spread + close in lower 40% · Institutional distribution — large players selling while retail buys the headline · Often seen at major tops after sustained rallies · The "smart money exit" signature',
  vsa_big_indecision:'<b>Big Volume Indecision</b> · Ultra-high volume but close lands in the middle of the range · Equal battle between supply and demand at high activity · Neither side won — next bar direction is critical · Often precedes violent directional resolution',
  vsa_effort_up:'<b>Effort to Rise</b> · Wide UP bar + high volume + close in upper half · Volume confirms the price rise — institutions are genuinely buying · Reliable bullish continuation signal when appearing in an uptrend · Strong divergence from weak bounces (no demand) in the same stock',
  vsa_effort_dn:'<b>Effort to Fall</b> · Wide DOWN bar + high volume + close in lower half · Supply is genuine and institutional · Bearish continuation signal in downtrends · Distinct from stopping volume where buyers absorb the selling',
  vsa_stopping:'<b>Stopping Volume</b> · Ultra-high volume + DOWN bar but close in upper 40-60% of range · Professional buyers stepped in and absorbed all the selling · Classic bottom formation — supply has been "stopped" · Most powerful when it appears at a key support or pivot level',
  vsa_climax_buy:'<b>Climactic Demand</b> · Highest volume day in last 20 sessions + wide UP bar + close near high · Peak buying intensity — every seller who wanted to sell has sold · Often the final stage of an accumulation · Bullish but watch for exhaustion if at overhead resistance',
  vsa_climax_sell:'<b>Climactic Supply</b> · Peak volume day + wide DOWN bar + close near low · Panic selling exhaustion · Every buyer who wanted to buy at this level has bought · Often marks a temporary or permanent bottom · Most reliable after an extended decline',
  vsa_no_demand:'<b>No Demand</b> · Narrow UP bar + below-average volume · The price rose but nobody significant bought · "Weak hands" pushing price up · Extremely bearish in context — usually precedes a decline · Tom Williams: "the market is telling you there is no professional interest at these prices"',
  vsa_no_supply:'<b>No Supply</b> · Narrow DOWN bar + below-average volume · A down day, but sellers have dried up · "The supply has been absorbed" — institutional buyers have been at work · Pre-breakout condition · One of VSA\'s most reliable pre-rally signals',
  vsa_test_supply:'<b>Test for Supply</b> · Low volume down day coming AFTER a high volume up day · Market is testing whether sellers return at current price · If this bar closes near its high (which we check), supply is absent · Bullish confirmation that the prior upthrust was genuine',
  vsa_upthrust:'<b>Upthrust (False Breakout)</b> · Wide spread + high volume + closes near the LOW · Price spiked up, attracted buyers, then reversed into the close · All the buyers who chased the high are now trapped · Classic Wyckoff distribution signal · Often appears at the top of a trading range',
  vsa_spring:'<b>Spring / Shakeout</b> · Wide spread + high volume + closes near the HIGH after going low · Price fell sharply to trap weak bulls into selling, then immediately recovered · Institutional "shake and bake" to collect stop-loss orders · Followed by the markup phase — one of Wyckoff\'s highest-conviction setups',
  vsa_pseudo_ut:'<b>Pseudo Upthrust</b> · New intraday high BUT closes in the lower 30% of the day\'s range on high volume · Tests the supply at new highs — finds sellers there · Signals that the breakout to new highs has failed · Exit signal for momentum longs; potential short entry',
  vsa_trap:'<b>Volume Trap</b> · High volume but price barely moved (close ≈ open, less than 20% of range) · Equal battle — supply exactly met demand · Market "trapped" in indecision · Usually resolves violently one way or the other · Wait for directional clarity before entering',
};

function updateXPInfo(){
  const s=document.getElementById('xp-strat').value;
  setFbar(`<span style="color:#f59e0b">🧪 Experimental</span> · ${XP_INFO[s]||s}`);
}

function scanXP(){
  const strat=document.getElementById('xp-strat').value;
  
  const rsMin=parseInt(document.getElementById('xp-rs').value);
  const prMin=parseFloat(document.getElementById('xp-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('xp-pmax').value)||Infinity;
  updateXPInfo(); rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.xp)continue;
    const v=s.xp.vsa||{};
    let matched=false; let sig=''; let extra={};
    if(strat==='vsa_big_bull')       {matched=!!v.big_vol_bull;    sig='🟢📦 Big Vol Bull'; extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_big_bear')       {matched=!!v.big_vol_bear;    sig='🔴📦 Big Vol Bear'; extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_big_indecision') {matched=!!v.big_vol_indecision;sig='⚡📦 Big Vol Ind'; extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_effort_up')      {matched=!!v.effort_up;       sig='💪 Effort Rise';   extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×'};}
    if(strat==='vsa_effort_dn')      {matched=!!v.effort_dn;       sig='⬇ Effort Fall';   extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×'};}
    if(strat==='vsa_stopping')       {matched=!!v.stopping_vol;    sig='🛑 Stopping Vol';  extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_climax_buy')     {matched=!!v.climax_buy;      sig='🌋 Climax Buy';    extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_climax_sell')    {matched=!!v.climax_sell;     sig='🌋 Climax Sell';   extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_no_demand')      {matched=!!v.no_demand;       sig='🔇 No Demand';     extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×'};}
    if(strat==='vsa_no_supply')      {matched=!!v.no_supply;       sig='🔕 No Supply';     extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×'};}
    if(strat==='vsa_test_supply')    {matched=!!v.test_supply;     sig='🔍 Test Supply';   extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_upthrust')       {matched=!!v.upthrust;        sig='⬆🔴 Upthrust';     extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_spring')         {matched=!!v.spring;          sig='⬇🟢 Spring';        extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_pseudo_ut')      {matched=!!v.pseudo_ut;       sig='🔺 Pseudo UT';     extra={vr:v.vol_ratio+'×',pos:v.close_pos+'%'};}
    if(strat==='vsa_trap')           {matched=!!v.vol_trap;        sig='🪤 Vol Trap';      extra={vr:v.vol_ratio+'×',sr:v.spread_ratio+'×'};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'xp'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── 🎨 PATTERNS (Harmonic + Chart Patterns) ──────────────────────────────
const PATT_INFO={
  harm_gartley_bull:'<b>Gartley Bullish</b> · Harold Gartley (1935) — most common harmonic pattern · AB/XA≈0.618 (golden ratio), AD/XA≈0.786 · D point is the Potential Reversal Zone (PRZ) · Bullish: D falls to PRZ then price reverses up · High win-rate when combined with volume confirmation',
  harm_gartley_bear:'<b>Gartley Bearish</b> · Inverse structure — D rises to PRZ then reverses down · Most reliable at known resistance levels · AB/XA≈0.618, AD/XA≈0.786',
  harm_bat_bull:'<b>Bat Bullish</b> · Scott Carney (2001) · AB/XA=0.382-0.5 (shallow retracement), AD/XA≈0.886 · The "shallow" AB retracement is key — distinguishes from Gartley · Tightest PRZ of all harmonics · Risk:reward often 1:3 or better',
  harm_bat_bear:'<b>Bat Bearish</b> · Shallow AB corrects to 0.382-0.5 of XA, then D rises to 0.886 of XA · Strong resistance at D',
  harm_butterfly_bull:'<b>Butterfly Bullish</b> · Scott Carney · AB/XA≈0.786, AD/XA=1.272-1.618 · D EXTENDS past X (goes further than the original move) · Appears at major price extensions — often marks generational lows',
  harm_butterfly_bear:'<b>Butterfly Bearish</b> · D extends above X — marks major tops · AD/XA=1.272-1.618 · Most dramatic pattern in harmonic trading',
  harm_crab_bull:'<b>Crab Bullish</b> · Extreme extension: AD/XA≈1.618 · The D point is the furthest from X of any harmonic pattern · Marks capitulation lows · When confirmed, bounces can be violent and rapid',
  harm_crab_bear:'<b>Crab Bearish</b> · AD/XA≈1.618 extension above X · Marks blow-off tops · Sharp reversals typical',
  harm_cypher_bull:'<b>Cypher Bullish</b> · Darren Oglesbee · BC extends past XA (unique to Cypher) · CD/XC≈0.786 · Different structure from other harmonics — BC goes beyond the original swing',
  harm_cypher_bear:'<b>Cypher Bearish</b> · BC extends above A beyond XA · CD/XC≈0.786 · Marks distribution at extended highs',
  harm_abcd_bull:'<b>AB=CD Bullish</b> · The base harmonic — all other patterns contain an AB=CD · CD leg equals AB in price (and ideally time) · D is the completion PRZ · Most frequent harmonic setup',
  harm_abcd_bear:'<b>AB=CD Bearish</b> · CD equals AB in price — D marks the top · Common at resistance levels',
  cp_hs:'<b>Head & Shoulders</b> · Most reliable chart pattern per technical analysis literature · Three peaks: left shoulder, head (tallest), right shoulder (at similar level to left) · Neckline connects the two lows · Break below neckline = bearish · Target = head height projected below neckline',
  cp_inv_hs:'<b>Inverse H&S</b> · Mirror of H&S — three troughs, middle deepest · Break above neckline = bullish reversal · One of the highest probability bullish continuation/reversal patterns in NSE large-caps',
  cp_dbl_top:'<b>Double Top</b> · Two equal-height peaks at resistance with a valley between · Price cannot break through resistance twice · Confirmation = close below the valley · Very common reversal pattern on Indian stocks at all-time highs',
  cp_dbl_bot:'<b>Double Bottom</b> · Two equal lows at support · "W" pattern · Confirmation = break above the peak between the two lows · Bullish reversal · Target = height of W projected above neckline',
  cp_rise_wedge:'<b>Rising Wedge</b> · Both trendlines slope up but converge — highs rising slower than lows · Despite looking bullish, it is bearish · Supply is tightening · Breakdown when price exits the lower trendline · Common in bear market rallies',
  cp_fall_wedge:'<b>Falling Wedge</b> · Both trendlines slope down but converge — lows falling slower than highs · Despite looking bearish, it is bullish · Demand is absorbing supply · Breakout above upper trendline · Common in healthy bull market corrections',
  cp_bull_flag:'<b>Bull Flag</b> · Strong vertical up-move (the pole) followed by a tight, slightly downward consolidation channel (the flag) · The consolidation shows supply exhaustion · Breakout from the flag = continuation of the original move · Target = pole length added to breakout point',
  cp_bear_flag:'<b>Bear Flag</b> · Strong down-move (pole) followed by tight upward consolidation · Bearish continuation · Distribution into the bounce · Breakdown from flag = continuation of decline',
  dbl_bot_d:'<b>Double Bottom (W) — Daily</b> · Two lows at a similar level separated by a peak (neckline) · The classic "W" reversal shape on the daily chart · Confirmation comes when price closes above the neckline',
  dbl_bot_d_bo:'<b>Double Bottom (W) — Daily, Neckline Breakout</b> · The W pattern is complete AND price has already closed above the neckline · Confirmed, actionable entry — the reversal is validated',
  dbl_bot_w:'<b>Double Bottom (W) — Weekly</b> · Two weekly lows at a similar level with a neckline peak between them · A weekly W pattern represents a multi-month base — breakouts tend to run for months',
  dbl_bot_w_bo:'<b>Double Bottom (W) — Weekly, Neckline Breakout</b> · Weekly W pattern confirmed with price closing above the weekly neckline · One of the highest-conviction swing/position entries',
  dbl_bot_m:'<b>Double Bottom (W) — Monthly</b> · Two monthly lows at a similar level — represents a multi-year base · Rare and significant when found',
  dbl_bot_m_bo:'<b>Double Bottom (W) — Monthly, Neckline Breakout</b> · Monthly W pattern confirmed — a major multi-year trend reversal, holding period often measured in years',
  dbl_top_d:'<b>Double Top (M) — Daily</b> · Two highs at a similar level separated by a trough (neckline) · The classic "M" reversal shape on the daily chart · Confirmation comes when price closes below the neckline',
  dbl_top_d_bo:'<b>Double Top (M) — Daily, Neckline Breakdown</b> · The M pattern is complete AND price has already closed below the neckline · Confirmed bearish reversal — consider exits or short setups',
  dbl_top_w:'<b>Double Top (M) — Weekly</b> · Two weekly highs at a similar level with a neckline trough between them · A weekly M pattern signals a multi-month top forming',
  dbl_top_w_bo:'<b>Double Top (M) — Weekly, Neckline Breakdown</b> · Weekly M pattern confirmed with price closing below the weekly neckline · High-conviction multi-week bearish signal',
  dbl_top_m:'<b>Double Top (M) — Monthly</b> · Two monthly highs at a similar level — represents a multi-year topping process · Rare and significant when found',
  dbl_top_m_bo:'<b>Double Top (M) — Monthly, Neckline Breakdown</b> · Monthly M pattern confirmed — a major multi-year trend reversal to the downside',
};
function updatePATTInfo(){
  const s=document.getElementById('patt-strat').value;
  const isHarm=s.startsWith('harm_');
  setFbar(`<span style="color:#06b6d4">🎨 Patterns</span> · ${PATT_INFO[s]||s}${isHarm?' <span style="color:var(--mu);font-size:9px">· Fibonacci tolerance ±10%</span>':''}`);
}
function scanPATT(){
  const strat=document.getElementById('patt-strat').value;
  
  const rsMin=parseInt(document.getElementById('patt-rs').value);
  const prMin=parseFloat(document.getElementById('patt-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('patt-pmax').value)||Infinity;
  updatePATTInfo(); rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.patt)continue;
    const h=s.patt.harm||{}; const cp=s.patt.cp||{};
    let matched=false; let sig=''; let extra={};
    if(strat==='harm_gartley_bull'){matched=!!h.gartley_bull;sig='🎵 Gartley↑'; extra={pivots:h.n_pivots};}
    if(strat==='harm_gartley_bear'){matched=!!h.gartley_bear;sig='🎵 Gartley↓'; extra={pivots:h.n_pivots};}
    if(strat==='harm_bat_bull')    {matched=!!h.bat_bull;    sig='🦇 Bat↑';     extra={pivots:h.n_pivots};}
    if(strat==='harm_bat_bear')    {matched=!!h.bat_bear;    sig='🦇 Bat↓';     extra={pivots:h.n_pivots};}
    if(strat==='harm_butterfly_bull'){matched=!!h.butterfly_bull;sig='🦋 Butterfly↑';extra={pivots:h.n_pivots};}
    if(strat==='harm_butterfly_bear'){matched=!!h.butterfly_bear;sig='🦋 Butterfly↓';extra={pivots:h.n_pivots};}
    if(strat==='harm_crab_bull')   {matched=!!h.crab_bull;   sig='🦀 Crab↑';   extra={pivots:h.n_pivots};}
    if(strat==='harm_crab_bear')   {matched=!!h.crab_bear;   sig='🦀 Crab↓';   extra={pivots:h.n_pivots};}
    if(strat==='harm_cypher_bull') {matched=!!h.cypher_bull; sig='⚡ Cypher↑'; extra={pivots:h.n_pivots};}
    if(strat==='harm_cypher_bear') {matched=!!h.cypher_bear; sig='⚡ Cypher↓'; extra={pivots:h.n_pivots};}
    if(strat==='harm_abcd_bull')   {matched=!!h.abcd_bull;   sig='〰 AB=CD↑';  extra={pivots:h.n_pivots};}
    if(strat==='harm_abcd_bear')   {matched=!!h.abcd_bear;   sig='〰 AB=CD↓';  extra={pivots:h.n_pivots};}
    if(strat==='cp_hs')        {matched=!!cp.hs;         sig='⛰ H&S';        extra={pivots:cp.n_pivots};}
    if(strat==='cp_inv_hs')    {matched=!!cp.inv_hs;     sig='⛰ Inv H&S';    extra={pivots:cp.n_pivots};}
    if(strat==='cp_dbl_top')   {matched=!!cp.dbl_top;    sig='🔝 Dbl Top';    extra={pivots:cp.n_pivots};}
    if(strat==='cp_dbl_bot')   {matched=!!cp.dbl_bot;    sig='⬇ Dbl Bot';    extra={pivots:cp.n_pivots};}
    if(strat==='cp_rise_wedge'){matched=!!cp.rise_wedge; sig='📐 Rise Wedge'; extra={pivots:cp.n_pivots};}
    if(strat==='cp_fall_wedge'){matched=!!cp.fall_wedge; sig='📐 Fall Wedge'; extra={pivots:cp.n_pivots};}
    if(strat==='cp_bull_flag') {matched=!!cp.bull_flag;  sig='🚩 Bull Flag';  extra={pivots:cp.n_pivots};}
    if(strat==='cp_bear_flag') {matched=!!cp.bear_flag;  sig='🚩 Bear Flag';  extra={pivots:cp.n_pivots};}
    // Double Bottom (W) / Double Top (M) — Daily/Weekly/Monthly with neckline confirmation
    const dbl=s.patt?.dbl||{}; const dd=dbl.d||{}, dw=dbl.w||{}, dm=dbl.m||{};
    const wW=dd.w||{}, wM=dd.m||{}, wwW=dw.w||{}, wwM=dw.m||{}, mwW=dm.w||{}, mwM=dm.m||{};
    if(strat==='dbl_bot_d')   {matched=!!wW.found;             sig='〔W〕 D Dbl Bot';     extra={neckline:wW.neckline,depth:wW.depth_pct+'%',breakout:wW.breakout?'✓':'—'};}
    if(strat==='dbl_bot_d_bo'){matched=!!wW.found&&!!wW.breakout; sig='〔W〕 D Dbl Bot BO'; extra={neckline:wW.neckline,depth:wW.depth_pct+'%'};}
    if(strat==='dbl_bot_w')   {matched=!!wwW.found;            sig='〔W〕 W Dbl Bot';     extra={neckline:wwW.neckline,depth:wwW.depth_pct+'%',breakout:wwW.breakout?'✓':'—'};}
    if(strat==='dbl_bot_w_bo'){matched=!!wwW.found&&!!wwW.breakout; sig='〔W〕 W Dbl Bot BO'; extra={neckline:wwW.neckline,depth:wwW.depth_pct+'%'};}
    if(strat==='dbl_bot_m')   {matched=!!mwW.found;            sig='〔W〕 M Dbl Bot';     extra={neckline:mwW.neckline,depth:mwW.depth_pct+'%',breakout:mwW.breakout?'✓':'—'};}
    if(strat==='dbl_bot_m_bo'){matched=!!mwW.found&&!!mwW.breakout; sig='〔W〕 M Dbl Bot BO'; extra={neckline:mwW.neckline,depth:mwW.depth_pct+'%'};}
    if(strat==='dbl_top_d')   {matched=!!wM.found;             sig='〔M〕 D Dbl Top';     extra={neckline:wM.neckline,depth:wM.depth_pct+'%',breakdown:wM.breakdown?'✓':'—'};}
    if(strat==='dbl_top_d_bo'){matched=!!wM.found&&!!wM.breakdown; sig='〔M〕 D Dbl Top BD'; extra={neckline:wM.neckline,depth:wM.depth_pct+'%'};}
    if(strat==='dbl_top_w')   {matched=!!wwM.found;            sig='〔M〕 W Dbl Top';     extra={neckline:wwM.neckline,depth:wwM.depth_pct+'%',breakdown:wwM.breakdown?'✓':'—'};}
    if(strat==='dbl_top_w_bo'){matched=!!wwM.found&&!!wwM.breakdown; sig='〔M〕 W Dbl Top BD'; extra={neckline:wwM.neckline,depth:wwM.depth_pct+'%'};}
    if(strat==='dbl_top_m')   {matched=!!mwM.found;            sig='〔M〕 M Dbl Top';     extra={neckline:mwM.neckline,depth:mwM.depth_pct+'%',breakdown:mwM.breakdown?'✓':'—'};}
    if(strat==='dbl_top_m_bo'){matched=!!mwM.found&&!!mwM.breakdown; sig='〔M〕 M Dbl Top BD'; extra={neckline:mwM.neckline,depth:mwM.depth_pct+'%'};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'patt'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ── 📡 MARKET BREADTH ────────────────────────────────────────────────────────
function renderBreadth(){
  const filter=document.getElementById('br-filter').value;
  const b=BREADTH||{};
  if(!b.total){
    document.getElementById('ts').innerHTML='<div class="nodata">No breadth data — regenerate scanner to compute breadth</div>';
    return;
  }
  if(filter==='all'){
    const health=b.health||0;
    const hColor=health>=60?'#84cc16':health>=40?'#f59e0b':'#ef4444';
    const bars=(vals)=>vals.map(([k,v,c])=>`
      <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:0.5px solid var(--brd)">
        <span style="width:160px;font-size:12px;color:var(--mu)">${k}</span>
        <div style="flex:1;background:var(--brd);border-radius:4px;height:8px;overflow:hidden">
          <div style="width:${v}%;height:100%;background:${c};border-radius:4px"></div>
        </div>
        <span style="width:44px;text-align:right;font-size:12px;font-weight:500;color:var(--txt)">${v}%</span>
      </div>`).join('');
    document.getElementById('ts').innerHTML=`
      <div style="padding:18px 20px">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px">
          <div style="font-size:13px;font-weight:500;color:var(--txt)">📡 Market Health Score</div>
          <div style="font-size:28px;font-weight:500;color:${hColor}">${health}</div>
          <div style="font-size:11px;color:var(--mu)">/100</div>
          <div style="font-size:10px;color:var(--mu);margin-left:8px">${b.total||0} stocks in universe</div>
        </div>
        <div style="max-width:560px">
          ${bars([
            ['Above 200 DMA',b.above200||0,'#3b82f6'],
            ['Weinstein Stage 2',b.stage2||0,'#22c55e'],
            ['Near 52W High (≤5%)',b.near52wh||0,'#a855f7'],
            ['P&F in X Column',b.pnf_x||0,'#e879f9'],
            ['Renko Bull Bricks',b.renko_bull||0,'#f59e0b'],
            ['Heikin Ashi Bull',b.ha_bull||0,'#06b6d4'],
            ['EMA Fan Bullish',b.ema_fan||0,'#ff9500'],
            ['RS Rating ≥ 70',b.rs70||0,'#1d9e75'],
            ['RS Rating ≥ 80',b.rs80||0,'#0f6e56'],
          ])}
        </div>
      </div>`;
    setFbar('📡 <b>Market Breadth</b> · Universe-wide health metrics · Higher = stronger market participation');
    return;
  }
  // Filter stocks matching the breadth condition
  const conds={
    above200: s=>s.above200,
    stage2:   s=>s.mi&&s.mi.stg===2,
    near52wh: s=>s.w52h&&(s.w52h-s.price)/s.w52h<=0.05,
    pnf_x:    s=>s.nl&&s.nl.pnf&&s.nl.pnf.in_x,
    renko_bull:s=>s.nl&&s.nl.renko&&s.nl.renko.bull,
    ha_bull:  s=>s.ti&&s.ti.d&&s.ti.d.ha&&s.ti.d.ha.bull,
    ema_fan:  s=>s.ti&&s.ti.d&&s.ti.d.ema&&s.ti.d.ema.fan_bull,
    rs80:     s=>s.rs>=80,
  };
  
  rows=S.filter(s=>{
    if(!passesIdx(s))return false;
    return conds[filter]?conds[filter](s):true;
  }).map(s=>({
    sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
    above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
    sig:'📡 Breadth',extra:{},strat:filter,_tab:'br'
  }));
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}


// ════════════════════════════════════════════════════════════════════════════
// ⑦ CPR in computePivots
// ════════════════════════════════════════════════════════════════════════════
// patched in the existing function via separate override:
(function(){
  var _orig=computePivots;
  computePivots=function(type,pH,pL,pC,pO,cO){
    if(type==='cpr'){
      var P=(pH+pL+pC)/3,BC=(pH+pL)/2,TC=2*P-BC,w=Math.abs(TC-BC);
      return{P:+P.toFixed(2),TC:+TC.toFixed(2),BC:+BC.toFixed(2),W:+w.toFixed(2),Wpct:+(w/P*100).toFixed(2),
             R1:+(TC+w).toFixed(2),R2:+(TC+2*w).toFixed(2),R3:+(TC+3*w).toFixed(2),
             S1:+(BC-w).toFixed(2),S2:+(BC-2*w).toFixed(2),S3:+(BC-3*w).toFixed(2)};
    }
    return _orig(type,pH,pL,pC,pO,cO);
  };
})();
// Add CPR to TYPE_META
if(window.TYPE_META) TYPE_META['cpr']={label:'CPR',levels:['TC','BC','P','R1','R2','S1','S2'],hasP:true};

// ════════════════════════════════════════════════════════════════════════════
// ① SuperTrend in scanT1 (patched after existing nr7_inside)
// ════════════════════════════════════════════════════════════════════════════
// handled inline in scanT1 via evalSignal approach

// ════════════════════════════════════════════════════════════════════════════
// ③ GAP SCANNER
// ════════════════════════════════════════════════════════════════════════════
function updateGapInfo(){setFbar('⚡ <b>Gap Scanner</b> — overnight gap vs prior close');}
function scanGap(){
  var strat=document.getElementById('gap-strat').value;
  var rsMin=parseInt(document.getElementById('gap-rs').value)||0;
  var prMin=parseFloat(document.getElementById('gap-pmin').value)||0;
  var prMax=parseFloat(document.getElementById('gap-pmax').value)||Infinity;
  updateGapInfo(); rows=[];
  for(var i=0;i<S.length;i++){
    var s=S[i];
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if((s.rs||0)<rsMin)continue;
    var g=s.gap; if(!g)continue;
    var matched=false,sig='',extra={};
    if(strat==='up1')       {matched=!!g.up1;   sig='🟢 Gap+'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='up2')       {matched=!!g.up2;   sig='🟢 Gap+'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='up3')       {matched=!!g.up3;   sig='🚀 Gap+'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='up_cont')   {matched=!!g.up&&!!g.cont;    sig='🟢↑ Gap+Cont';    extra={gap:g.pct+'%'};}
    if(strat==='up_reject') {matched=!!g.up&&!!g.reject;  sig='🔴 Gap Reject';   extra={gap:g.pct+'%'};}
    if(strat==='up_unfilled'){matched=!!g.up&&!g.filled;  sig='🟢 Unfilled';     extra={gap:g.pct+'%'};}
    if(strat==='dn1')       {matched=!!g.dn1;   sig='🔴 Gap'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='dn2')       {matched=!!g.dn2;   sig='🔴 Gap'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='dn3')       {matched=!!g.dn3;   sig='💥 Gap'+g.pct+'%'; extra={gap:g.pct+'%'};}
    if(strat==='dn_cont')   {matched=!!g.dn&&!!g.dn_cont; sig='🔴↓ Gap+Cont';    extra={gap:g.pct+'%'};}
    if(strat==='dn_unfilled'){matched=!!g.dn&&!g.filled;  sig='🔴 Unfilled';     extra={gap:g.pct+'%'};}
    if(strat==='consec3')   {matched=!!g.consec3&&!!g.up; sig='🚀 3×Gap↑';       extra={gap:g.pct+'%'};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig:sig,extra:extra,strat:strat,_tab:'gap',sector:s.sector||''});
  }
  sc=2;sd=1;rows.sort(function(a,b){return a.sym.localeCompare(b.sym);}); render();
}

// ════════════════════════════════════════════════════════════════════════════
// ② MULTI-SIGNAL AND COMBINER
// ════════════════════════════════════════════════════════════════════════════
var COMBO_SIGNALS=(function(){
  var G=[
    {g:'⭐ Tier-1',tab:'t1',ss:[['Golden Cross Bull Zone','gc_bull_zone'],['Fresh Golden Cross','gc_fresh_golden'],['RSI Bull Div','rsi_bull_div'],['RSI Oversold','rsi_oversold'],['BB Squeeze Bull','bb_squeeze_bull'],['52W Breakout','w52_breakout'],['52W Near High','w52_near'],['NR7 Bull','nr7_bull'],['ST Bull','st_bull'],['ST Flip Bull','st_flip_bull'],['ST Tight','st_tight']]},
    {g:'🏆 Tier-2',tab:'t2',ss:[['Cup&Handle BO','ch_breakout'],['MACD Bull Div','macd_bull_div'],['MACD Flip Bull','macd_flip_bull'],['Z-Score OS','zs_oversold'],['RS Outperform','rsn_outperform']]},
    {g:'🎖 Tier-3',tab:'t3',ss:[['Fib 61.8% Support','fib_near_618'],['ADX Strong Bull','adx_strong_bull'],['Stoch Bull Cross','stoch_bull_cross'],['Williams %R OS','wr_oversold'],['ParSAR Bull','psar_bull']]},
    {g:'⚡ Multi-Ind',tab:'mi',ss:[['Minervini+Weinstein','both'],['Minervini','minervini'],['Weinstein Stage 2','weinstein']]},
    {g:'🏅 India Pro',tab:'ti',ss:[['HA Strong Bull','ha_strong_bull'],['HA Bull','ha_bull'],['HMA Bull','hma_bull'],['HMA Flip Bull','hma_flip_bull'],['EMA Fan Bull','ema_fan_bull'],['MA Stack','ma_all_bull'],['Keltner Cross Up','kelt_cross_up'],['MFI Oversold','mfi_os'],['MFI Bull Div','mfi_bull_div'],['CCI OS Exit','cci_os_exit'],['LR Slope Bull','lr_slope_bull']]},
    {g:'📐 Noiseless',tab:'nl',ss:[['P&F X-Column','pnf_in_x'],['P&F Dbl Top','pnf_dbl_top'],['Renko Bull','renko_bull'],['Renko 5 Punch','renko_5'],['Kagi Shoulder','kagi_shoulder'],['3LB Bull Rev','lb3_rev_bull'],['RRG Leading','rrg_leading']]},
    {g:'🎨 Patterns',tab:'patt',ss:[['Inv H&S','cp_inv_hs'],['Bull Flag','cp_bull_flag'],['Falling Wedge','cp_fall_wedge'],['Gartley Bull','harm_gartley_bull'],['AB=CD Bull','harm_abcd_bull']]},
    {g:'⚡ Gaps',tab:'gap',ss:[['Gap Up ≥1%','up1'],['Gap Up Cont','up_cont'],['Gap Down ≥1%','dn1']]},
    {g:'📊 Meta',tab:'_meta',ss:[['Above 200 DMA','above200'],['RS ≥ 70','rs70'],['RS ≥ 80','rs80'],['Near 52W High','near52wh'],['FNO Stock','is_fno']]},
  ];
  var out=[];
  for(var gi=0;gi<G.length;gi++){var grp=G[gi]; for(var si=0;si<grp.ss.length;si++) out.push({label:grp.g+' › '+grp.ss[si][0],tab:grp.tab,strat:grp.ss[si][1]});}
  return out;
})();

function _populateComboDropdowns(){
  ['combo-c1','combo-c2','combo-c3','combo-c4','combo-c5'].forEach(function(id){
    var el=document.getElementById(id); if(!el)return;
    var cur=el.value;
    while(el.options.length>1) el.remove(1);
    COMBO_SIGNALS.forEach(function(sig){
      var o=document.createElement('option');
      o.value=sig.tab+':'+sig.strat; o.textContent=sig.label;
      el.appendChild(o);
    });
    if(cur) el.value=cur;
  });
}
function renderComboSummary(){
  var ids=['combo-c1','combo-c2','combo-c3','combo-c4','combo-c5'];
  var active=[];
  ids.forEach(function(id){var el=document.getElementById(id); if(el&&el.value){var s=COMBO_SIGNALS.find(function(x){return x.tab+':'+x.strat===el.value;}); if(s)active.push(s.label.split(' › ')[1]);}});
  setFbar('⚗️ <b>Combiner</b>'+(active.length?': '+active.join(' AND '):'— pick conditions above'));
}
function evalCondition(s,tab,strat){
  if(!strat)return true;
  if(tab==='_meta'){if(strat==='above200')return !!s.above200;if(strat==='rs70')return (s.rs||0)>=70;if(strat==='rs80')return (s.rs||0)>=80;if(strat==='near52wh')return s.w52h&&(s.w52h-s.price)/s.w52h<=0.05;if(strat==='is_fno')return !!s.fno;return false;}
  if(tab==='gap'){var g=s.gap||{};if(strat==='up1')return !!g.up1;if(strat==='up_cont')return !!g.up&&!!g.cont;if(strat==='dn1')return !!g.dn1;return false;}
  if(tab==='mi'){var m=s.mi||{};if(strat==='both')return (m.mts||0)>=6&&m.stg===2;if(strat==='minervini')return (m.mts||0)>=6;if(strat==='weinstein')return m.stg===2;return false;}
  if(tab==='t1'){var t=s.t1||{},gc=t.gc||{},rsi=t.rsi||{},bb=t.bb||{},w52=t.w52||{},nr=t.nr||{},stt=t.stt||t.st||{};
    var map={gc_bull_zone:gc.bull_zone,gc_fresh_golden:gc.fresh_golden,rsi_bull_div:rsi.bull_div,rsi_oversold:rsi.oversold,bb_squeeze_bull:bb.squeeze_bull,w52_breakout:w52.breakout,w52_near:w52.near,nr7_bull:nr.bias_bull,st_bull:stt.bull,st_flip_bull:stt.flip_bull,st_tight:stt.bull&&Math.abs(stt.dist||99)<1};
    return !!map[strat];}
  if(tab==='t2'){var t2=s.t2||{},ch=t2.ch||{},macd=t2.macd||{},zs=t2.zs||{},rsn=t2.rsn||{};
    var m2={ch_breakout:ch.breakout,macd_bull_div:macd.bull_div,macd_flip_bull:macd.flip_bull,zs_oversold:zs.oversold,rsn_outperform:rsn.outperform};return !!m2[strat];}
  if(tab==='t3'){var t3=s.t3||{},fib=t3.fib||{},adx=t3.adx||{},stch=t3.stoch||{},wr=t3.wr||{},psar=t3.psar||{};
    var m3={fib_near_618:fib.near_618,adx_strong_bull:adx.strong_bull,stoch_bull_cross:stch.bull_cross,wr_oversold:wr.oversold,psar_bull:psar.bull};return !!m3[strat];}
  if(tab==='ti'){var ti=s.ti||{},ha=ti.ha||{},hma=ti.hma||{},ema=ti.ema||{},ma=ti.ma||{},kelt=ti.kelt||{},mfi=ti.mfi||{},cci=ti.cci||{},lr=ti.linreg||{};
    var mti={ha_strong_bull:ha.strong_bull,ha_bull:ha.bull,hma_bull:hma.bull,hma_flip_bull:hma.flip_bull,ema_fan_bull:ema.fan_bull,ma_all_bull:ma.all_bull,kelt_cross_up:kelt.cross_up,mfi_os:mfi.oversold,mfi_bull_div:mfi.bull_div,cci_os_exit:cci.os_exit,lr_slope_bull:lr.slope_bull};return !!mti[strat];}
  if(tab==='nl'){var nl=s.nl||{},pnf=nl.pnf||{},renko=nl.renko||{},kagi=nl.kagi||{},lb3=nl.lb3||{},rrg=nl.rrg||{};
    var mnl={pnf_in_x:pnf.in_x,pnf_dbl_top:pnf.dbl_top,renko_bull:renko.bull,renko_5:renko.punch5,kagi_shoulder:kagi.shoulder,lb3_rev_bull:lb3.rev_bull,rrg_leading:rrg.quad==='Leading'};return !!mnl[strat];}
  if(tab==='patt'){var h=(s.patt||{}).harm||{},cp=(s.patt||{}).cp||{};
    var mp={cp_inv_hs:cp.inv_hs,cp_bull_flag:cp.bull_flag,cp_fall_wedge:cp.fall_wedge,harm_gartley_bull:h.gartley_bull,harm_abcd_bull:h.abcd_bull};return !!mp[strat];}
  return false;
}
function scanCombo(){
  var conds=['combo-c1','combo-c2','combo-c3','combo-c4','combo-c5'].map(function(id){
    var v=document.getElementById(id);v=v?v.value:'';
    if(!v)return null;var p=v.split(':');return{tab:p[0],strat:p[1]};
  }).filter(Boolean);
  if(!conds.length){setFbar('⚗️ Combiner — select at least one condition');return;}
  var rsMin=parseInt(document.getElementById('combo-rs').value)||0;
  renderComboSummary(); rows=[];
  for(var i=0;i<S.length;i++){
    var s=S[i];
    if(!passesIdx(s))continue;
    if((s.rs||0)<rsMin)continue;
    var ok=true;
    for(var ci=0;ci<conds.length;ci++){if(!evalCondition(s,conds[ci].tab,conds[ci].strat)){ok=false;break;}}
    if(!ok)continue;
    var sigs=conds.map(function(c){var f=COMBO_SIGNALS.find(function(x){return x.tab===c.tab&&x.strat===c.strat;});return f?f.label.split(' › ')[1]:c.strat;});
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,sig:sigs.join('+'),extra:{n:sigs.length},strat:'combo',_tab:'combo',sector:s.sector||''});
  }
  sc=2;sd=1;rows.sort(function(a,b){return a.sym.localeCompare(b.sym);});render();
}

// ════════════════════════════════════════════════════════════════════════════
// ④ STOCK DETAIL POP-UP
// ════════════════════════════════════════════════════════════════════════════
function showStockDetail(sym){
  var s=S.find(function(x){return x.sym===sym;});if(!s)return;
  var f2=function(v){return typeof v==='number'?v.toFixed(2):v||'—';};
  var pct=function(v,b){return b&&v?((v-b)/b*100).toFixed(1)+'%':'—';};
  var flag=function(v,y){return v?'<span style="color:#00e5a0">'+y+'</span>':''};
  var R=function(k,v){return '<tr><td style="color:var(--mu);padding:2px 8px;font-size:11px">'+k+'</td><td style="padding:2px 8px;font-size:11px">'+v+'</td></tr>';};
  var SEC=function(t,c){return '<div style="margin-bottom:12px"><div style="font-size:9px;font-weight:700;color:var(--acc);letter-spacing:1px;margin-bottom:4px;font-family:var(--mono)">'+t+'</div><table>'+c+'</table></div>';};
  var ti=s.ti||{},t1=s.t1||{},mi=s.mi||{};
  var ha=ti.ha||{},hma=ti.hma||{},kelt=ti.kelt||{},mfi=ti.mfi||{},cci=ti.cci||{},lr=ti.linreg||{};
  var stt=t1.stt||t1.st||{},gc=t1.gc||{},rsi=t1.rsi||{},w52=t1.w52||{};
  var nl=s.nl||{},pnf=nl.pnf||{},renko=nl.renko||{},kagi=nl.kagi||{};
  var gap=s.gap||{};
  // Sparkline SVG
  var spk=s.spark||[];var spkMin=spk.length?Math.min.apply(null,spk):0;var spkMax=spk.length?Math.max.apply(null,spk):1;
  var spkRng=spkMax-spkMin||1;var trend=spk.length>=2&&spk[spk.length-1]>spk[0];
  var pts=spk.map(function(v,i){return (i/(spk.length-1||1)*220).toFixed(1)+','+(36-(v-spkMin)/spkRng*32).toFixed(1);}).join(' ');
  var spkSvg=spk.length>1?'<svg viewBox="0 0 220 36" width="220" height="36" style="margin-bottom:8px;display:block"><polyline points="'+pts+'" fill="none" stroke="'+(trend?'#00e5a0':'#ef4444')+'" stroke-width="1.5"/></svg>':'';
  // Freshness
  var fr=dateFreshness(s.date);
  document.getElementById('sd-body').dataset.sym=sym;
  document.getElementById('ai-output').innerHTML='';
  document.getElementById('sd-body').innerHTML='<div style="display:flex;flex-wrap:wrap;gap:18px;padding:2px">'
    +'<div style="min-width:200px"><div style="font-size:22px;font-weight:700;color:var(--txt);margin-bottom:2px">'+sym+'</div>'
    +'<div style="font-size:11px;color:var(--mu);margin-bottom:8px">'+(s.sector||'Other')+' · '+(s.idx?'N'+s.idx:'Other')+(s.fno?' · <span style="color:#00e5a0">FNO</span>':'')+'</div>'
    +spkSvg
    +'<div style="font-size:10px;color:'+fr.color+';margin-bottom:10px">⏱ Data: '+fr.label+' ('+s.date+')</div>'
    +SEC('PRICE',R('Price','₹'+f2(s.price))+R('RS Rating',(s.rs||0)+'/99')+R('200 DMA',f2(s.dma200))+R('vs 200 DMA',pct(s.price,s.dma200))+R('52W High',f2(s.w52h))+R('52W Low',f2(s.w52l)))
    +'</div>'
    +'<div style="min-width:200px">'
    +SEC('SUPERTREND & T1',R('SuperTrend',stt.bull?'🟢 Bull ('+stt.consec+'×)':stt.bear?'🔴 Bear':'—')+R('ST Line',f2(stt.st))+R('ST Distance',(stt.dist||0)+'%')+R('Golden Cross',gc.bull_zone?'✓ Bull Zone':gc.near_golden?'⚡ Near':'')+R('RSI',rsi.bull_div?'🟢 Bull Div':rsi.bear_div?'🔴 Bear Div':'')+R('52W',w52.breakout?'🚀 Breakout':w52.near?'Near High':''))
    +SEC('INDIA PRO',R('Heikin Ashi',ha.bull?'🟢 Bull ('+ha.consec_bull+'×)':ha.bear?'🔴 Bear':'—')+R('HMA',hma.bull?'🟢 Bull':hma.bear?'🔴 Bear':'—')+R('Keltner',kelt.above?'Above Band':kelt.squeezed?'Squeezed':kelt.below?'Below Band':'')+R('MFI',mfi.mfi!==undefined?mfi.mfi.toFixed(0)+(mfi.oversold?' OS':mfi.overbought?' OB':''):'—')+R('CCI',cci.cci!==undefined?cci.cci.toFixed(0)+(cci.oversold?' OS':cci.overbought?' OB':''):'—'))
    +'</div>'
    +'<div style="min-width:180px">'
    +SEC('NOISELESS',R('P&F',pnf.in_x?'X Column':pnf.in_o?'O Column':'—')+R('Renko',renko.bull?'🟢 Bull':renko.bear?'🔴 Bear':'—')+R('Kagi',kagi.yang?'🟢 Yang ('+kagi.consec+'×)':kagi.yin?'🔴 Yin':'—')+R('Kagi Shoulder',flag(kagi.shoulder,'✓ Yes')))
    +SEC('MI',R('Stage',mi.stg?'Stage '+mi.stg:'—')+R('Score',(mi.mts||0)+'/8')+R('Minervini',flag((mi.mts||0)>=6,'✓'))+R('Weinstein',flag(mi.stg===2,'✓ Stage 2')))
    +SEC('GAP',R('Gap %',(gap.pct||0)+'%')+R('Status',gap.up?(gap.filled?'Filled':gap.cont?'Continuation':'Unfilled'):gap.dn?'Gap Down':'—'))
    +'</div></div>';
  document.getElementById('sd-overlay').style.display='block';
  document.getElementById('sd-box').style.display='flex';
}
function hideStockDetail(){
  document.getElementById('sd-overlay').style.display='none';
  document.getElementById('sd-box').style.display='none';
}

// ════════════════════════════════════════════════════════════════════════════
// ⑨ DATA FRESHNESS
// ════════════════════════════════════════════════════════════════════════════
function dateFreshness(dateStr){
  if(!dateStr)return{days:null,label:'?',color:'#888'};
  var d=Math.round((new Date()-new Date(dateStr))/864e5);
  return{days:d,label:d===0?'Today':d===1?'Yday':d+'d',color:d>5?'#ef4444':d>2?'#f59e0b':'#00e5a0'};
}


function dateFreshness(dateStr){
  if(!dateStr)return{days:null,label:'?',color:'#888'};
  const d=Math.round((new Date()-new Date(dateStr))/864e5);
  return{days:d,label:d===0?'Today':d===1?'Yday':d+'d ago',color:d>5?'#ef4444':d>2?'#f59e0b':'#00e5a0'};
}
function ageCell(d){const fr=dateFreshness(d);return`<td style="padding:3px 8px;color:${fr.color};font-size:9px;text-align:center;white-space:nowrap">${fr.label}</td>`;}
function sparkCell(sym){
  const spk=(S.find(s=>s.sym===sym)||{}).spark||[];
  if(spk.length<2)return'<td></td>';
  const mn=Math.min(...spk),mx=Math.max(...spk),rng=mx-mn||1,tr=spk[spk.length-1]>spk[0];
  const pts=spk.map((v,i)=>((i/(spk.length-1)*58)).toFixed(1)+','+(16-(v-mn)/rng*14).toFixed(1)).join(' ');
  return`<td style="padding:2px 6px"><svg viewBox="0 0 58 16" width="58" height="16"><polyline points="${pts}" fill="none" stroke="${tr?'#00e5a0':'#ef4444'}" stroke-width="1.2"/></svg></td>`;
}

// ════════════════════════════════════════════════════════════════════════════
// 🏹 BREAKOUT PRO — 10 paid-scanner strategies
// ════════════════════════════════════════════════════════════════════════════
const BP_INFO={
  vcp:'<b>VCP</b> · Mark Minervini · Volatility Contraction Pattern · 2+ weekly range contractions with drying volume, price near 52W high · MarketSmith India\'s #1 setup',
  vcp3:'<b>VCP Tight</b> · 3+ contractions · Higher conviction setup — 3 successive range compressions with declining volume · Breakout imminent',
  pp:'<b>Pocket Pivot</b> · Gil Morales\/Chris Kacher · Today\'s up-day volume exceeds highest down-day volume in prior 10 sessions · Early entry before breakout',
  pp_strong:'<b>Strong Pocket Pivot</b> · Volume 1.5× highest down-day vol · Institutional accumulation signal · High-conviction early entry',
  twt3:'<b>3-Week Tight</b> · O\'Neil · Weekly closes within ±1.5% for 3 weeks · Institutional absorption — supply being locked up',
  twt4:'<b>4-Week Tight</b> · Stronger version of 3-week tight · More weeks of tight action = stronger coiling',
  twt5:'<b>5-Week Tight</b> · Strongest tight-area pattern · Price coiled for 5 weeks — significant expansion potential',
  rsh:'<b>RS Line New 52W High</b> · IBD/MarketSmith · Stock\'s relative strength vs Nifty at yearly peak — outperforming the market at a new high · Most powerful leading indicator',
  rs_bo:'<b>RS Line Breakout</b> · RS line exiting 3-month consolidation — strength returning before price breaks out',
  rs_trend:'<b>RS Line Trending</b> · RS line slope positive over 13 weeks — sustained outperformance',
  ads_a:'<b>A-Rating (IBD A/D)</b> · Heavy institutional accumulation over 13 weeks · More up-volume days than down · Institutions quietly buying',
  ads_ab:'<b>A or B Rating</b> · Net accumulation phase · Institutional money flowing in · StockEdge premium equivalent',
  ads_de:'<b>D or E Rating</b> · Distribution phase · Institutional selling · Avoid or short candidates',
  hvb:'<b>High Volume Breakout</b> · 52W high close + 2× avg volume · Volume confirms the breakout — not a fake-out',
  hvb_strong:'<b>HVB Strong</b> · 52W high + 3× avg volume · Explosive breakout with massive institutional participation',
  near_hvb:'<b>Near-High HVB</b> · Within 2% of 52W high + 2× volume · Setting up for breakout — buy before the breakout',
  vdu_bull:'<b>Volume Dry-Up Bullish</b> · Volume < 50% of 20-day avg while price holds · Supply exhausted — fuel for next move · Chartink paid filter',
  vdu_40d:'<b>Vol 40-Session Low</b> · Volume at multi-month lows · Extreme drying — maximum coiling before expansion',
  ems_full:'<b>Full EMA Stack</b> · Price > 9 > 21 > 50 > 200 EMA · Perfect bull alignment · TradePoint semi-algo signal',
  ems_power:'<b>Power Stack</b> · Full stack + all EMAs rising · EMAs trending up and price above all — strongest bull configuration',
  ems_part:'<b>Partial EMA Stack</b> · Price > 21 > 50 > 200 EMA · Strong but not perfect alignment',
  iw1:'<b>Inside Week</b> · Current week\'s range inside prior week · Weekly compression before expansion — classic setup',
  iw2:'<b>Double Inside Week</b> · 2 consecutive inside weeks · Extreme compression — strong expansion signal',
  iw_up:'<b>Inside Week in Uptrend</b> · Inside week occurring in established uptrend · Lower risk entry',
  tlb:'<b>Descending Trendline Breakout</b> · Price closes above falling trendline drawn across last 3 peaks · Trend reversal signal',
};
function updateBPInfo(){const s=document.getElementById('bp-strat').value;setFbar('🏹 <b>Breakout Pro</b> · '+(BP_INFO[s]||s));}
function scanBP(){
  const strat=document.getElementById('bp-strat').value;
  const rsMin=parseInt(document.getElementById('bp-rs').value)||0;
  const prMin=parseFloat(document.getElementById('bp-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('bp-pmax').value)||Infinity;
  updateBPInfo(); rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if((s.rs||0)<rsMin)continue;
    const bp=s.bp||{};
    const vcp=bp.vcp||{},pp=bp.pp||{},twt=bp.twt||{},rsh=bp.rsh||{};
    const ads=bp.ads||{},hvb=bp.hvb||{},vdu=bp.vdu||{};
    const ems=bp.ems||{},iw=bp.iw||{},tlb=bp.tlb||{};
    let matched=false,sig='',extra={};
    if(strat==='vcp')      {matched=!!vcp.is_vcp;    sig='🔄 VCP';          extra={contractions:vcp.contractions,tight:vcp.tight_pct+'%'};}
    if(strat==='vcp3')     {matched=!!vcp.is_vcp3;   sig='🔄 VCP×3';        extra={contractions:vcp.contractions};}
    if(strat==='pp')       {matched=!!pp.is_pp;      sig='🎯 PktPiv';       extra={vol_ratio:pp.vol_ratio+'×'};}
    if(strat==='pp_strong'){matched=!!pp.is_pp_strong;sig='🎯💪 PktPiv+';   extra={vol_ratio:pp.vol_ratio+'×'};}
    if(strat==='twt3')     {matched=!!twt.is_3wt;    sig='📏 3W Tight';     extra={tight:twt.tight_pct+'%'};}
    if(strat==='twt4')     {matched=!!twt.is_4wt;    sig='📏 4W Tight';     extra={tight:twt.tight_pct+'%',weeks:twt.tight_weeks};}
    if(strat==='twt5')     {matched=!!twt.is_5wt;    sig='📏 5W Tight';     extra={tight:twt.tight_pct+'%',weeks:twt.tight_weeks};}
    if(strat==='rsh')      {matched=!!rsh.is_rs_new_high; sig='📈 RS High'; extra={vs_52w:rsh.vs_52w_hi+'%'};}
    if(strat==='rs_bo')    {matched=!!rsh.rs_breakout;    sig='📈 RS BO';   extra={};}
    if(strat==='rs_trend') {matched=!!rsh.rs_trending;    sig='📈 RS↑';     extra={};}
    if(strat==='ads_a')    {matched=ads.rating==='A';      sig='🏆 A-Rating'; extra={score:ads.score};}
    if(strat==='ads_ab')   {matched=!!ads.is_accum;        sig='🏆 '+(ads.rating||'?')+'-Rating'; extra={score:ads.score};}
    if(strat==='ads_de')   {matched=!!ads.is_distrib;      sig='🔴 '+(ads.rating||'?')+'-Rating'; extra={score:ads.score};}
    if(strat==='hvb')      {matched=!!hvb.hvb;         sig='⚡ HVB';       extra={vol:hvb.vol_ratio+'×'};}
    if(strat==='hvb_strong'){matched=!!hvb.hvb_strong; sig='⚡⚡ HVB×3';   extra={vol:hvb.vol_ratio+'×'};}
    if(strat==='near_hvb') {matched=!!hvb.near_hvb;    sig='⚡ Near HVB';  extra={vol:hvb.vol_ratio+'×'};}
    if(strat==='vdu_bull') {matched=!!vdu.vdu_bullish; sig='💧 VDU Bull';  extra={vol:vdu.vol_ratio+'×avg'};}
    if(strat==='vdu_40d')  {matched=!!vdu.vol_40d_low; sig='💧 40d Low';   extra={vol:vdu.vol_ratio+'×avg'};}
    if(strat==='ems_full') {matched=!!ems.full_stack;   sig='💪 EMA Full'; extra={e9:ems.e9,e21:ems.e21};}
    if(strat==='ems_power'){matched=!!ems.power_stack;  sig='💪💪 Power';  extra={e9:ems.e9,e50:ems.e50};}
    if(strat==='ems_part') {matched=!!ems.partial_stack;sig='💪 EMA Part'; extra={e21:ems.e21};}
    if(strat==='iw1')      {matched=!!iw.iw1;           sig='📅 IW';       extra={};}
    if(strat==='iw2')      {matched=!!iw.iw2;           sig='📅 IW×2';     extra={};}
    if(strat==='iw_up')    {matched=!!iw.iw_uptrend;    sig='📅 IW↑';      extra={};}
    if(strat==='tlb')      {matched=!!tlb.tl_breakout;  sig='📐 TL BO';    extra={dist:tlb.dist+'%',tl:tlb.tl_val};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'bp',sector:s.sector||''});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ════════════════════════════════════════════════════════════════════════════
// 🔢 SMART RANK — composite scoring (Definedge/Trendlyne equivalents)
// ════════════════════════════════════════════════════════════════════════════
const MONTH_NAMES=['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const SR_INFO={
  ms_90:'<b>Momentum Score ≥ 90</b> · Top 10% of universe by composite momentum · Weighted: 1M(15%) + 3M(20%) + 6M(25%) + 12M(30%) + RS(10%) · Trendlyne Momentum Score equivalent built from OHLCV',
  ms_75:'<b>Momentum Score ≥ 75</b> · Top quartile · Stocks showing sustained multi-timeframe outperformance vs the full universe',
  ms_50:'<b>Momentum Score ≥ 50</b> · Above-average momentum · Good for early-stage breakout candidates before they reach top decile',
  ms_rising:'<b>Momentum Rising</b> · 1-month return rank improving vs 6-month rank · Stock accelerating vs universe',
  season_bull:'<b>Seasonality Bullish</b> · Historical avg return in current month &gt; 2% across 5+ years · Definedge Seasonality Scanner equivalent',
  season_strong:'<b>Seasonality Strong</b> · Avg &gt; 3% AND hit rate &gt; 65% in current month · Most reliable seasonal setups',
  season_best:'<b>Best Month</b> · Current month is this stock\'s best-performing calendar month historically',
  season_avoid:'<b>Seasonality Bearish</b> · Historically negative in current month — avoid longs, consider hedges',
  mtf_perfect:'<b>MTF Perfect (15/15)</b> · All daily + weekly + monthly conditions simultaneously bullish · Definedge Matrix equivalent — highest-conviction multi-timeframe setup',
  mtf_strong:'<b>MTF Strong (12+/15)</b> · 12 or more of 15 trend conditions bullish across 3 timeframes · Strong structural alignment',
  mtf_7:'<b>MTF Aligned (7+/15)</b> · Majority of multi-timeframe checks pass · Emerging alignment',
  mtf_daily_max:'<b>Daily Max (5/5)</b> · All 5 daily trend checks pass: above MA50, MA50>MA200, above EMA21, RSI>50, higher than 5d ago',
  ath:'<b>All-Time High Breakout</b> · Price closes above ALL prior history · ATH breakouts on volume have the highest 6-month forward returns of any pattern (IBD/StockTrendz data)',
  near_ath:'<b>Near All-Time High</b> · Within 2% of ATH · Loading zone before potential ATH breakout',
  tc3:'<b>Triple Compression</b> · NR7 + Volume Dry-Up + BB Squeeze firing simultaneously · Three independent compression signals = maximum coiling, smallest range in all three dimensions',
  tc2:'<b>Double Compression</b> · Any two of NR7/VDU/BB Squeeze active · Still powerful pre-breakout signal',
  mtfrs_90:'<b>Consistent RS Leader</b> · Top 10% of universe by 1M, 3M, 6M, AND 12M returns simultaneously · Definedge Smart RS equivalent — the market\'s true leaders',
  mtfrs_75:'<b>Consistent RS Strong</b> · Top quartile across all four return periods · Sustained outperformance, not a 1-day wonder',
  r1m_top:'<b>1M Return Top 10%</b> · Highest 1-month momentum in the universe · Short-term leaders',
  r3m_top:'<b>3M Return Top 10%</b> · Strongest 3-month performers · Intermediate momentum leaders',
};
function updateSRInfo(){const s=document.getElementById('sr-strat').value;setFbar('🔢 <b>Smart Rank</b> · '+(SR_INFO[s]||s));}

function scanSR(){
  const strat=document.getElementById('sr-strat').value;
  const rsMin=parseInt(document.getElementById('sr-rs').value)||0;
  const prMin=parseFloat(document.getElementById('sr-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('sr-pmax').value)||Infinity;
  updateSRInfo(); rows=[];
  // Pre-sort by momentum score for top% calculations
  const all_ms=S.map(s=>s.sr?.ms||0);
  const n=all_ms.length;
  const pct=(val)=>Math.round(S.filter(s=>(s.sr?.ms||0)<val).length/n*100);
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if((s.rs||0)<rsMin)continue;
    const sr=s.sr||{};
    const ms=sr.ms||0;
    const season=sr.season||{};
    const mtf=sr.mtf||{};
    const ath=sr.ath||{};
    const tc=sr.tc||{};
    const r1m=sr.r1m||0; const r3m=sr.r3m||0;
    const r6m=sr.r6m||0; const r12m=sr.r12m||0;
    let matched=false,sig='',extra={};
    // Momentum Score
    if(strat==='ms_90') {matched=ms>=90; sig='🔢 MS:'+ms; extra={ms,r3m:r3m+'%',r12m:r12m+'%'};}
    if(strat==='ms_75') {matched=ms>=75; sig='🔢 MS:'+ms; extra={ms,r6m:r6m+'%'};}
    if(strat==='ms_50') {matched=ms>=50; sig='🔢 MS:'+ms; extra={ms};}
    if(strat==='ms_rising'){
      const sr_r1=S.filter(x=>(x.sr?.r1m||0)<r1m).length/n*100;
      const sr_r6=S.filter(x=>(x.sr?.r6m||0)<r6m).length/n*100;
      matched=sr_r1>sr_r6+10;
      sig='📈 Accel '+r1m.toFixed(1)+'%'; extra={r1m:r1m.toFixed(1)+'%',r6m:r6m.toFixed(1)+'%'};
    }
    // Seasonality
    const cm=season.cm||new Date().getMonth()+1;
    const cm_data=season.months?season.months[cm]:{};
    if(strat==='season_bull') {matched=(cm_data.avg||0)>2; sig='📅 '+MONTH_NAMES[cm]+' avg '+(cm_data.avg||0)+'%'; extra={avg:(cm_data.avg||0)+'%',hit:(cm_data.hit||0)+'%',n:cm_data.n||0};}
    if(strat==='season_strong'){matched=(cm_data.avg||0)>3&&(cm_data.hit||0)>65; sig='📅💪 '+(cm_data.avg||0)+'% hit '+(cm_data.hit||0)+'%'; extra={avg:(cm_data.avg||0)+'%',hit:(cm_data.hit||0)+'%'};}
    if(strat==='season_best') {matched=season.bm===cm; sig='📅★ Best month ('+MONTH_NAMES[cm]+')'; extra={avg:(season.bm_avg||0)+'%'};}
    if(strat==='season_avoid'){matched=(cm_data.avg||0)<-1; sig='📅🔴 '+(cm_data.avg||0)+'%'; extra={avg:(cm_data.avg||0)+'%',hit:(cm_data.hit||0)+'%'};}
    // MTF Matrix
    if(strat==='mtf_perfect') {matched=mtf.total===15; sig='💯 MTF 15/15'; extra={d:mtf.d+'/5',w:mtf.w+'/5',m:mtf.m+'/5'};}
    if(strat==='mtf_strong')  {matched=(mtf.total||0)>=12; sig='💪 MTF '+(mtf.total||0)+'/15'; extra={d:mtf.d+'/5',w:mtf.w+'/5',m:mtf.m+'/5'};}
    if(strat==='mtf_7')       {matched=(mtf.total||0)>=7; sig='📊 MTF '+(mtf.total||0)+'/15'; extra={d:mtf.d||0,w:mtf.w||0,m:mtf.m||0};}
    if(strat==='mtf_daily_max'){matched=(mtf.d||0)===5; sig='📊 D:5/5 MTF:'+(mtf.total||0); extra={w:mtf.w||0,m:mtf.m||0};}
    // ATH
    if(strat==='ath')      {matched=!!ath.is_ath;  sig='🏔 ATH'; extra={dist:ath.dist_pct+'%'};}
    if(strat==='near_ath') {matched=!!ath.near_ath&&!ath.is_ath; sig='🏔 Near ATH '+ath.dist_pct+'%'; extra={ath:ath.ath};}
    // Triple Compression
    if(strat==='tc3'){matched=!!tc.tc3; sig='🗜×3 '+(tc.nr7?'NR7 ':'')+( tc.vdu?'VDU ':'')+( tc.bb_sq?'BBS':''); extra={nr7:tc.nr7?'✓':'—',vdu:tc.vdu?'✓':'—',bbs:tc.bb_sq?'✓':'—'};}
    if(strat==='tc2'){matched=!!tc.tc2; sig='🗜×'+tc.count+' '+(tc.nr7?'NR7 ':'')+(tc.vdu?'VDU ':'')+(tc.bb_sq?'BBS':''); extra={count:tc.count};}
    // MTF RS
    const topPct10=n>0?all_ms.sort((a,b)=>a-b)[Math.floor(n*0.9)]:99;
    const r1m_all=S.map(s=>s.sr?.r1m||0).sort((a,b)=>a-b);
    const r3m_all=S.map(s=>s.sr?.r3m||0).sort((a,b)=>a-b);
    const top10_r1=r1m_all[Math.floor(n*0.9)]||0;
    const top10_r3=r3m_all[Math.floor(n*0.9)]||0;
    const r6m_all=S.map(s=>s.sr?.r6m||0).sort((a,b)=>a-b);
    const r12m_all=S.map(s=>s.sr?.r12m||0).sort((a,b)=>a-b);
    const top25_r1=r1m_all[Math.floor(n*0.75)]||0;
    const top25_r3=r3m_all[Math.floor(n*0.75)]||0;
    const top25_r6=r6m_all[Math.floor(n*0.75)]||0;
    const top25_r12=r12m_all[Math.floor(n*0.75)]||0;
    if(strat==='mtfrs_90') {matched=r1m>=top10_r1&&r3m>=top10_r3&&r6m>=r6m_all[Math.floor(n*0.9)]&&r12m>=r12m_all[Math.floor(n*0.9)]; sig='🌟 RS Leader'; extra={r1m:r1m.toFixed(1)+'%',r3m:r3m.toFixed(1)+'%'};}
    if(strat==='mtfrs_75') {matched=r1m>=top25_r1&&r3m>=top25_r3&&r6m>=top25_r6&&r12m>=top25_r12; sig='⭐ RS Consistent'; extra={r1m:r1m.toFixed(1)+'%',r12m:r12m.toFixed(1)+'%'};}
    if(strat==='r1m_top')  {matched=r1m>=top10_r1; sig='⚡ 1M '+r1m.toFixed(1)+'%'; extra={r1m:r1m.toFixed(1)+'%'};}
    if(strat==='r3m_top')  {matched=r3m>=top10_r3; sig='⚡ 3M '+r3m.toFixed(1)+'%'; extra={r3m:r3m.toFixed(1)+'%'};}
    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'sr',sector:s.sector||''});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

// ════════════════════════════════════════════════════════════════════════════
// 🤖 AI STOCK ANALYSIS — Claude API in the stock detail modal
// ════════════════════════════════════════════════════════════════════════════
async function aiAnalyzeStock(sym){
  const s=S.find(x=>x.sym===sym); if(!s)return;
  const btn=document.getElementById('ai-btn');
  const out=document.getElementById('ai-output');
  if(!btn||!out)return;
  btn.disabled=true; btn.textContent='⏳ Analysing...';
  out.innerHTML='<div style="color:var(--mu);font-size:11px;padding:8px">🤖 Claude is analysing this stock...</div>';
  const mi=s.mi||{}; const t1=s.t1||{}; const stt=t1.stt||t1.st||{};
  const bp=s.bp||{}; const sr=s.sr||{};
  const prompt=`You are a professional NSE equity analyst. Analyse this stock and give a concise 3-paragraph technical assessment.

Stock: ${sym}
Price: ₹${s.price} | RS Rating: ${s.rs}/99 | Index: ${s.idx?'Nifty '+s.idx:'Other'} | FNO: ${s.fno?'Yes':'No'}
Sector: ${s.sector||'Unknown'}

KEY SIGNALS:
- vs 200 DMA: ${s.dma200?((s.price-s.dma200)/s.dma200*100).toFixed(1)+'%':'N/A'}
- 52W High: ₹${s.w52h||'N/A'} (${s.w52h?((s.price-s.w52h)/s.w52h*100).toFixed(1)+'%':'N/A'})
- SuperTrend: ${stt.bull?'Bullish ('+stt.consec+'× bars, dist '+stt.dist+'%)':stt.bear?'Bearish':'N/A'}
- MI Score: ${mi.mts||0}/8 | Stage: ${mi.stg||'N/A'}
- Momentum Score: ${sr.ms||0}/100
- MTF Matrix: ${sr.mtf?.total||0}/15 (D:${sr.mtf?.d||0} W:${sr.mtf?.w||0} M:${sr.mtf?.m||0})
- Returns: 1M=${(sr.r1m||0).toFixed(1)}% 3M=${(sr.r3m||0).toFixed(1)}% 6M=${(sr.r6m||0).toFixed(1)}% 12M=${(sr.r12m||0).toFixed(1)}%
- VCP: ${bp.vcp?.is_vcp?'Yes ('+bp.vcp.contractions+' contractions)':'No'}
- Gap: ${s.gap?.pct||0}% ${s.gap?.up?'(up)':s.gap?.dn?'(down)':''}
- Seasonality: ${sr.season?.cm_avg||0}% avg this month, ${sr.season?.cm_hit||0}% hit rate
- ATH: ${sr.ath?.is_ath?'YES — at all-time high':sr.ath?.near_ath?'Near ATH ('+sr.ath?.dist_pct+'%)':'No'}

Format as 3 short paragraphs: (1) Overall assessment, (2) Key bullish/bearish signals, (3) Risk and setup quality. Be direct. Use ₹ for prices. Max 150 words total.`;
  try{
    const resp=await fetch('https://api.anthropic.com/v1/messages',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'claude-sonnet-4-20250514',max_tokens:300,
        messages:[{role:'user',content:prompt}]})
    });
    const data=await resp.json();
    const txt=data.content?.map(b=>b.text||'').join('')||'No response';
    out.innerHTML='<div style="font-size:11px;line-height:1.6;padding:8px;color:var(--txt);white-space:pre-wrap">'+txt+'</div>';
  }catch(e){
    out.innerHTML='<div style="color:var(--red);font-size:11px;padding:8px">⚠️ API error: '+e.message+'</div>';
  }
  btn.disabled=false; btn.textContent='🤖 AI Analysis';
}


// ════════════════════════════════════════════════════════════════════════════
// 🔍 SIGNAL SCORER — evaluates ~35 signals across all tabs, returns Buy/Sell score
// ════════════════════════════════════════════════════════════════════════════

// ════════════════════════════════════════════════════════════════════════════
// 🏆 TOP PICKS — universe-wide Strong Buy / Strong Sell board
// ════════════════════════════════════════════════════════════════════════════
function renderTopPicks(){
  return `<div style="padding:14px">
    <div style="display:flex;gap:10px;margin-bottom:12px;align-items:center;flex-wrap:wrap">
      <label style="font-size:12px;color:var(--mu)">Universe</label>
      <select id="tp-idx" style="padding:6px 10px;background:var(--bg2);border:1px solid var(--brd);
        border-radius:5px;color:var(--txt);font-size:12px" onchange="scanTopPicks()">
        <option value="0">All Stocks</option>
        <option value="fno">F&O Only</option>
        <option value="50">Nifty 50</option>
        <option value="100">Nifty 100</option>
        <option value="200">Nifty 200</option>
        <option value="500">Nifty 500</option>
        <option value="750">Nifty 750</option>
      </select>
      <label style="font-size:12px;color:var(--mu)">Show top</label>
      <select id="tp-n" style="padding:6px 10px;background:var(--bg2);border:1px solid var(--brd);
        border-radius:5px;color:var(--txt);font-size:12px" onchange="scanTopPicks()">
        <option value="10">10</option>
        <option value="15">15</option>
        <option value="20">20</option>
        <option value="30">30</option>
      </select>
      <span id="tp-count" style="font-size:11px;color:var(--mu);margin-left:auto"></span>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;min-width:0">
      <div>
        <div style="font-size:12px;font-weight:700;color:#00e5a0;margin-bottom:8px;font-family:var(--mono)">
          🟢 TOP BUYS — highest confluence-weighted score</div>
        <div id="tp-buy"></div>
      </div>
      <div>
        <div style="font-size:12px;font-weight:700;color:#ef4444;margin-bottom:8px;font-family:var(--mono)">
          🔴 TOP SELLS — lowest confluence-weighted score</div>
        <div id="tp-sell"></div>
      </div>
    </div>
    <div style="margin-top:14px;font-size:10px;color:var(--mu);padding:8px 12px;background:rgba(59,158,255,.04);
      border:1px solid rgba(59,158,255,.12);border-radius:5px">
      💡 Score combines 148 signals across 14 categories with a confluence bonus — stocks confirmed by
      multiple independent methods (e.g. Trend + SMC + Smart Rank together) rank higher than stocks with
      many signals from a single category. Click ⓘ on any symbol for the full scanner breakdown.
    </div>
  </div>`;
}

function _tpRow(s,sc,rank){
  const f2=v=>typeof v==='number'?v.toFixed(2):(v||'—');
  const pct=(v,b)=>b&&v?((v-b)/b*100).toFixed(1)+'%':'—';
  const dist=s.dma200?((s.price-s.dma200)/s.dma200*100):0;
  return `<div onclick="showScorePopup('${s.sym}')" style="display:flex;align-items:center;gap:10px;
    padding:8px 10px;background:var(--bg2);border:1px solid var(--brd);border-radius:6px;
    margin-bottom:5px;cursor:pointer">
    <span style="color:var(--mu);font-size:11px;width:18px;text-align:right">${rank}</span>
    <span class="sd-ico" onmouseenter="_sdHover('${s.sym}')" title="Stock detail" onclick="event.stopPropagation()">i</span>
    <a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${s.sym}" target="_blank" rel="noopener"
       onclick="event.stopPropagation()" style="font-weight:700;min-width:90px">${s.sym}</a>
    <span style="font-size:11px;color:var(--mu);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${s.sector||'—'}</span>
    <span style="font-size:12px">₹${f2(s.price)}</span>
    <span style="font-size:11px;color:${dist>=0?'#00e5a0':'#ef4444'};min-width:54px;text-align:right">${pct(s.price,s.dma200)}</span>
    <div style="width:50px;height:6px;background:var(--brd);border-radius:3px;overflow:hidden">
      <div style="width:${sc.score}%;height:100%;background:${sc.color}"></div>
    </div>
    <span style="color:${sc.color};font-weight:700;min-width:30px;text-align:right">${sc.score}</span>
  </div>`;
}

function scanTopPicks(){
  const idxSel=document.getElementById('tp-idx').value;
  const n=parseInt(document.getElementById('tp-n').value)||10;
  let list=S.slice();
  if(idxSel==='fno') list=list.filter(s=>s.fno);
  else if(idxSel!=='0') list=list.filter(s=>s.idx&&s.idx<=parseInt(idxSel));

  const scored=list.map(s=>({s,sc:scoreStock(s)}));
  scored.sort((a,b)=>b.sc.score-a.sc.score);

  const buys=scored.slice(0,n);
  const sells=scored.slice(-n).reverse();

  document.getElementById('tp-count').textContent=list.length+' stocks scanned';
  document.getElementById('tp-buy').innerHTML=buys.map((x,i)=>_tpRow(x.s,x.sc,i+1)).join('')
    ||'<div class="nodata">No stocks</div>';
  document.getElementById('tp-sell').innerHTML=sells.map((x,i)=>_tpRow(x.s,x.sc,i+1)).join('')
    ||'<div class="nodata">No stocks</div>';
}

const TAB_NAMES={piv:'Pivot Points',smc:'Price Action/SMC',vol:'Volume',mi:'Multi-Indicator',
  adv:'Advanced',t1:'Tier-1',t2:'Tier-2',t3:'Tier-3',ti:'India Pro',nl:'Noiseless',
  xp:'Experimental/VSA',patt:'Patterns',br:'Breadth',gap:'Gaps',combo:'Combiner',
  bp:'Breakout Pro',sr:'Smart Rank',swing:'Swing Setup'};
const SIGNAL_DEFS=[
  // ── Trend / Tier-1 ──
  {cat:'Trend',tab:'t1',label:'SuperTrend Bullish',          w:3, chk:s=>s.t1?.stt?.bull},
  {cat:'Trend',tab:'t1',label:'SuperTrend Bearish',          w:-3,chk:s=>s.t1?.stt?.bear},
  {cat:'Trend',tab:'t1',label:'SuperTrend Flip Bull (fresh)',w:4, chk:s=>s.t1?.stt?.flip_bull},
  {cat:'Trend',tab:'t1',label:'SuperTrend Flip Bear (fresh)',w:-4,chk:s=>s.t1?.stt?.flip_bear},
  {cat:'Trend',tab:'t1',label:'Golden Cross (50>200 SMA)',   w:3, chk:s=>s.t1?.gc?.bull_zone},
  {cat:'Trend',tab:'t1',label:'Fresh Golden Cross',          w:4, chk:s=>s.t1?.gc?.fresh_golden},
  {cat:'Trend',tab:'t1',label:'Fresh Death Cross',           w:-4,chk:s=>s.t1?.gc?.fresh_death},
  {cat:'Trend',tab:'t1',label:'Price Above 200 SMA',         w:1, chk:s=>s.above200},
  {cat:'Trend',tab:'t1',label:'Price Below 200 SMA',         w:-1,chk:s=>!s.above200&&s.dma200>0},
  {cat:'Trend',tab:'adv',label:'Donchian 20-Day Breakout',   w:3, chk:s=>s.adv?.dc?.dc20_bull},
  {cat:'Trend',tab:'adv',label:'Donchian 20-Day Breakdown',  w:-3,chk:s=>s.adv?.dc?.dc20_bear},
  {cat:'Trend',tab:'adv',label:'Donchian 55-Day Bull (Turtle)',w:4,chk:s=>s.adv?.dc?.dc55_bull},
  {cat:'Trend',tab:'adv',label:'GMMA Cross Bull (fresh)',    w:4, chk:s=>s.adv?.gmma?.cross_bull},
  {cat:'Trend',tab:'adv',label:'GMMA Cross Bear (fresh)',    w:-4,chk:s=>s.adv?.gmma?.cross_bear},
  {cat:'Trend',tab:'adv',label:'GMMA All-Bull (full stack)', w:3, chk:s=>s.adv?.gmma?.all_bull},
  {cat:'Trend',tab:'adv',label:'GMMA All-Bear',              w:-3,chk:s=>s.adv?.gmma?.all_bear},
  {cat:'Trend',tab:'adv',label:'Ichimoku Price Above Cloud', w:2, chk:s=>s.adv?.ichi?.above},
  {cat:'Trend',tab:'adv',label:'Ichimoku Kumo Breakout',     w:3, chk:s=>s.adv?.ichi?.kbo},
  {cat:'Trend',tab:'bp', label:'EMA Power Stack (full)',     w:3, chk:s=>s.bp?.ems?.full_stack},
  {cat:'Trend',tab:'bp', label:'EMA Power Stack (rising)',   w:4, chk:s=>s.bp?.ems?.power_stack},

  // ── Momentum ──
  {cat:'Momentum',tab:'t1',label:'RSI Bullish Divergence',   w:3, chk:s=>s.t1?.rsi?.bull_div},
  {cat:'Momentum',tab:'t1',label:'RSI Bearish Divergence',   w:-3,chk:s=>s.t1?.rsi?.bear_div},
  {cat:'Momentum',tab:'t1',label:'RSI Cross Above 50',       w:2, chk:s=>s.t1?.rsi?.cross50_bull},
  {cat:'Momentum',tab:'t1',label:'RSI Cross Below 50',       w:-2,chk:s=>s.t1?.rsi?.cross50_bear},
  {cat:'Momentum',tab:'t1',label:'RSI Overbought (≥70)',     w:-1,chk:s=>s.t1?.rsi?.overbought},
  {cat:'Momentum',tab:'t1',label:'RSI Oversold (≤30)',       w:1, chk:s=>s.t1?.rsi?.oversold},
  {cat:'Momentum',tab:'t1',label:'RSI Bull Zone (40-80)',    w:1, chk:s=>s.t1?.rsi?.bull_zone},
  {cat:'Momentum',tab:'t2',label:'MACD Bullish Flip',        w:3, chk:s=>s.t2?.macd?.flip_bull},
  {cat:'Momentum',tab:'t2',label:'MACD Bearish Flip',        w:-3,chk:s=>s.t2?.macd?.flip_bear},
  {cat:'Momentum',tab:'t2',label:'MACD Zero-Line Cross Up',  w:3, chk:s=>s.t2?.macd?.zero_bull},
  {cat:'Momentum',tab:'t2',label:'MACD Zero-Line Cross Down',w:-3,chk:s=>s.t2?.macd?.zero_bear},
  {cat:'Momentum',tab:'t2',label:'MACD Histogram Rising',    w:1, chk:s=>s.t2?.macd?.hist_rising},
  {cat:'Momentum',tab:'t2',label:'MACD Hist +ve & Rising',   w:2, chk:s=>s.t2?.macd?.hist_pos_inc},
  {cat:'Momentum',tab:'t2',label:'MACD Bull Divergence',     w:3, chk:s=>s.t2?.macd?.bull_div},
  {cat:'Momentum',tab:'t2',label:'MACD Bear Divergence',     w:-3,chk:s=>s.t2?.macd?.bear_div},
  {cat:'Momentum',tab:'t3',label:'ADX Strong Bull Trend',    w:2, chk:s=>s.t3?.adx?.strong&&s.t3?.adx?.bull_trend},
  {cat:'Momentum',tab:'t3',label:'Stochastic Bull Cross',    w:2, chk:s=>s.t3?.stoch?.bull_cross},
  {cat:'Momentum',tab:'t3',label:'Stochastic Oversold',      w:1, chk:s=>s.t3?.stoch?.oversold},
  {cat:'Momentum',tab:'t3',label:'Stochastic Overbought',    w:-1,chk:s=>s.t3?.stoch?.overbought},
  {cat:'Momentum',tab:'t3',label:'PSAR Flip Bullish',        w:3, chk:s=>s.t3?.psar?.flip&&s.t3?.psar?.direction==='bull'},
  {cat:'Momentum',tab:'t3',label:'PSAR Flip Bearish',        w:-3,chk:s=>s.t3?.psar?.flip&&s.t3?.psar?.direction==='bear'},

  // ── Volatility / Breakout ──
  {cat:'Volatility',tab:'t1',label:'BB Squeeze + Bull Momentum',w:3,chk:s=>s.t1?.bb?.squeeze_bull},
  {cat:'Volatility',tab:'t1',label:'BB Squeeze + Bear Momentum',w:-3,chk:s=>s.t1?.bb?.squeeze_bear},
  {cat:'Volatility',tab:'t1',label:'BB Cross Above Upper Band',w:2, chk:s=>s.t1?.bb?.cross_upper},
  {cat:'Volatility',tab:'t1',label:'BB Cross Below Lower Band',w:-2,chk:s=>s.t1?.bb?.cross_lower},
  {cat:'Volatility',tab:'t1',label:'52-Week High Breakout',  w:3, chk:s=>s.t1?.w52?.breakout},
  {cat:'Volatility',tab:'t1',label:'30-Day High Breakout',   w:2, chk:s=>s.t1?.w52?.h30_bo},
  {cat:'Volatility',tab:'t1',label:'90-Day High Breakout',   w:2, chk:s=>s.t1?.w52?.h90_bo},
  {cat:'Volatility',tab:'sr',label:'All-Time High Breakout', w:4, chk:s=>s.sr?.ath?.is_ath},
  {cat:'Volatility',tab:'sr',label:'Triple Compression (NR7+VDU+BBSq)',w:2,chk:s=>s.sr?.tc?.tc3},

  // ── Smart Money / Structure ──
  {cat:'SMC',tab:'smc',label:'Break of Structure Bull',      w:3, chk:s=>s.smc?.bos_bull},
  {cat:'SMC',tab:'smc',label:'Break of Structure Bear',      w:-3,chk:s=>s.smc?.bos_bear},
  {cat:'SMC',tab:'smc',label:'Change of Character (ChoCh)',  w:3, chk:s=>s.smc?.choch==='bull'},
  {cat:'SMC',tab:'smc',label:'Change of Character (Bear)',   w:-3,chk:s=>s.smc?.choch==='bear'},
  {cat:'SMC',tab:'smc',label:'Bullish Fair Value Gap',       w:2, chk:s=>s.smc?.bull_fvg},
  {cat:'SMC',tab:'smc',label:'Bearish Fair Value Gap',       w:-2,chk:s=>s.smc?.bear_fvg},
  {cat:'SMC',tab:'smc',label:'Bullish Order Block',          w:2, chk:s=>s.smc?.bull_ob},
  {cat:'SMC',tab:'smc',label:'Bearish Order Block',          w:-2,chk:s=>s.smc?.bear_ob},

  // ── Minervini / Weinstein ──
  {cat:'MultiIndicator',tab:'mi',label:'Minervini Trend Template (≥6/8)',w:4,chk:s=>(s.mi?.mts||0)>=6},
  {cat:'MultiIndicator',tab:'mi',label:'Weinstein Stage 2 (Advancing)',w:3,chk:s=>s.mi?.stg===2},
  {cat:'MultiIndicator',tab:'mi',label:'Weinstein Stage 4 (Declining)',w:-3,chk:s=>s.mi?.stg===4},

  // ── Breakout Pro ──
  {cat:'BreakoutPro',tab:'bp',label:'VCP (Volatility Contraction)',w:4,chk:s=>s.bp?.vcp?.is_vcp},
  {cat:'BreakoutPro',tab:'bp',label:'VCP Tight (3+ contractions)',w:5,chk:s=>s.bp?.vcp?.is_vcp3},
  {cat:'BreakoutPro',tab:'bp',label:'Pocket Pivot',           w:3, chk:s=>s.bp?.pp?.is_pp},
  {cat:'BreakoutPro',tab:'bp',label:'Strong Pocket Pivot',    w:4, chk:s=>s.bp?.pp?.is_pp_strong},
  {cat:'BreakoutPro',tab:'bp',label:'3-Week Tight',           w:2, chk:s=>s.bp?.twt?.is_3wt},
  {cat:'BreakoutPro',tab:'bp',label:'5-Week Tight',           w:3, chk:s=>s.bp?.twt?.is_5wt},
  {cat:'BreakoutPro',tab:'bp',label:'RS New 52W High',        w:4, chk:s=>s.bp?.rsh?.is_rs_new_high},
  {cat:'BreakoutPro',tab:'bp',label:'RS Line Breakout',       w:3, chk:s=>s.bp?.rsh?.rs_breakout},
  {cat:'BreakoutPro',tab:'bp',label:'A/D Rating A or B',      w:2, chk:s=>s.bp?.ads?.is_accum},
  {cat:'BreakoutPro',tab:'bp',label:'A/D Rating D or E',      w:-2,chk:s=>s.bp?.ads?.is_distrib},
  {cat:'BreakoutPro',tab:'bp',label:'High Volume Breakout',   w:3, chk:s=>s.bp?.hvb?.hvb},
  {cat:'BreakoutPro',tab:'bp',label:'Volume Dry-Up (Bullish)',w:2, chk:s=>s.bp?.vdu?.vdu_bullish},
  {cat:'BreakoutPro',tab:'bp',label:'Trendline Breakout',     w:2, chk:s=>s.bp?.tlb?.tl_breakout},

  // ── Smart Rank ──
  {cat:'SmartRank',tab:'sr',label:'Momentum Score ≥ 90',      w:4, chk:s=>(s.sr?.ms||0)>=90},
  {cat:'SmartRank',tab:'sr',label:'Momentum Score ≥ 75',      w:2, chk:s=>(s.sr?.ms||0)>=75&&(s.sr?.ms||0)<90},
  {cat:'SmartRank',tab:'sr',label:'Momentum Score < 25',      w:-2,chk:s=>(s.sr?.ms||0)<25},
  {cat:'SmartRank',tab:'sr',label:'MTF Matrix Perfect (15/15)',w:5,chk:s=>(s.sr?.mtf?.total||0)===15},
  {cat:'SmartRank',tab:'sr',label:'MTF Matrix Strong (≥12)',  w:3, chk:s=>(s.sr?.mtf?.total||0)>=12&&(s.sr?.mtf?.total||0)<15},
  {cat:'SmartRank',tab:'sr',label:'Consistent RS Leader',     w:4, chk:s=>{
    const r1=s.sr?.r1m||0,r3=s.sr?.r3m||0,r6=s.sr?.r6m||0,r12=s.sr?.r12m||0;
    return r1>0&&r3>0&&r6>0&&r12>0&&(s.rs||0)>=80;
  }},

  // ── Patterns ──
  {cat:'Patterns',tab:'patt',label:'Gartley Bullish (Harmonic)',w:3,chk:s=>s.patt?.harm?.gartley_bull},
  {cat:'Patterns',tab:'patt',label:'Gartley Bearish (Harmonic)',w:-3,chk:s=>s.patt?.harm?.gartley_bear},
  {cat:'Patterns',tab:'patt',label:'Double Bottom',           w:3, chk:s=>s.patt?.cp?.double_bottom},
  {cat:'Patterns',tab:'patt',label:'Double Top',              w:-3,chk:s=>s.patt?.cp?.double_top},


  // ── Double Bottom (W) / Double Top (M) — D/W/M ──
  {cat:'Patterns',tab:'patt',label:'Double Bottom (W) Daily — Neckline BO',w:3,chk:s=>s.patt?.dbl?.d?.w?.found&&s.patt?.dbl?.d?.w?.breakout},
  {cat:'Patterns',tab:'patt',label:'Double Bottom (W) Weekly — Neckline BO',w:4,chk:s=>s.patt?.dbl?.w?.w?.found&&s.patt?.dbl?.w?.w?.breakout},
  {cat:'Patterns',tab:'patt',label:'Double Bottom (W) Monthly — Neckline BO',w:5,chk:s=>s.patt?.dbl?.m?.w?.found&&s.patt?.dbl?.m?.w?.breakout},
  {cat:'Patterns',tab:'patt',label:'Double Top (M) Daily — Neckline BD',w:-3,chk:s=>s.patt?.dbl?.d?.m?.found&&s.patt?.dbl?.d?.m?.breakdown},
  {cat:'Patterns',tab:'patt',label:'Double Top (M) Weekly — Neckline BD',w:-4,chk:s=>s.patt?.dbl?.w?.m?.found&&s.patt?.dbl?.w?.m?.breakdown},
  {cat:'Patterns',tab:'patt',label:'Double Top (M) Monthly — Neckline BD',w:-5,chk:s=>s.patt?.dbl?.m?.m?.found&&s.patt?.dbl?.m?.m?.breakdown},

  // ── Gap ──
  {cat:'Gap',tab:'gap',label:'Gap Up Continuation',           w:2, chk:s=>s.gap?.up&&s.gap?.cont},
  {cat:'Gap',tab:'gap',label:'Gap Down Continuation',         w:-2,chk:s=>s.gap?.dn&&s.gap?.cont},

  // ── India Pro Candlesticks ──
  {cat:'Candles',tab:'ti',label:'Hammer',                     w:2, chk:s=>s.ti?.candle?.hammer},
  {cat:'Candles',tab:'ti',label:'Inverted Hammer',            w:1, chk:s=>s.ti?.candle?.inv_hammer},
  {cat:'Candles',tab:'ti',label:'Bullish Engulfing',          w:3, chk:s=>s.ti?.candle?.bull_eng},
  {cat:'Candles',tab:'ti',label:'Bearish Engulfing',          w:-3,chk:s=>s.ti?.candle?.bear_eng},
  {cat:'Candles',tab:'ti',label:'Morning Star',               w:3, chk:s=>s.ti?.candle?.morning_star},
  {cat:'Candles',tab:'ti',label:'Evening Star',               w:-3,chk:s=>s.ti?.candle?.evening_star},
  {cat:'Candles',tab:'ti',label:'Piercing Pattern',           w:2, chk:s=>s.ti?.candle?.piercing},
  {cat:'Candles',tab:'ti',label:'Bullish Harami',             w:1, chk:s=>s.ti?.candle?.bull_har},
  {cat:'Candles',tab:'ti',label:'Bearish Harami',             w:-1,chk:s=>s.ti?.candle?.bear_har},
  {cat:'Candles',tab:'ti',label:'Three White Soldiers',       w:3, chk:s=>s.ti?.candle?.tws},
  {cat:'Candles',tab:'ti',label:'Three Black Crows',          w:-3,chk:s=>s.ti?.candle?.tbc},
  {cat:'Candles',tab:'ti',label:'Doji',                       w:0, chk:s=>s.ti?.candle?.doji, info:true},
  {cat:'Candles',tab:'ti',label:'Shooting Star',              w:-2,chk:s=>s.ti?.candle?.shooting_star},
  {cat:'Candles',tab:'ti',label:'Marubozu Bull',              w:2, chk:s=>s.ti?.candle?.marubozu_bull},
  {cat:'Candles',tab:'ti',label:'Marubozu Bear',              w:-2,chk:s=>s.ti?.candle?.marubozu_bear},

  // ── Noiseless Charts ──
  {cat:'Noiseless',tab:'nl',label:'P&F X Column (Bull)',      w:2, chk:s=>s.nl?.pnf?.in_x},
  {cat:'Noiseless',tab:'nl',label:'P&F O Column (Bear)',      w:-2,chk:s=>s.nl?.pnf?.in_o},
  {cat:'Noiseless',tab:'nl',label:'P&F Double Top BO',        w:3, chk:s=>s.nl?.pnf?.dbl_top},
  {cat:'Noiseless',tab:'nl',label:'P&F Double Bottom BO',     w:-3,chk:s=>s.nl?.pnf?.dbl_bot},
  {cat:'Noiseless',tab:'nl',label:'P&F Bullish Catapult',     w:4, chk:s=>s.nl?.pnf?.bull_cat},
  {cat:'Noiseless',tab:'nl',label:'P&F Bearish Catapult',     w:-4,chk:s=>s.nl?.pnf?.bear_cat},
  {cat:'Noiseless',tab:'nl',label:'P&F High Pole',            w:-2,chk:s=>s.nl?.pnf?.high_pole},
  {cat:'Noiseless',tab:'nl',label:'Renko Bull',               w:2, chk:s=>s.nl?.renko?.bull},
  {cat:'Noiseless',tab:'nl',label:'Renko Bear',               w:-2,chk:s=>s.nl?.renko?.bear},
  {cat:'Noiseless',tab:'nl',label:'Renko Fresh Reversal',     w:3, chk:s=>s.nl?.renko?.just_rev&&s.nl?.renko?.bull},
  {cat:'Noiseless',tab:'nl',label:'Renko 3+ Punch Bull',      w:2, chk:s=>s.nl?.renko?.punch3&&s.nl?.renko?.bull},
  {cat:'Noiseless',tab:'nl',label:'Renko 5+ Punch Bull',      w:3, chk:s=>s.nl?.renko?.punch5&&s.nl?.renko?.bull},
  {cat:'Noiseless',tab:'nl',label:'Kagi Yang (Bull)',         w:2, chk:s=>s.nl?.kagi?.yang},
  {cat:'Noiseless',tab:'nl',label:'Kagi Yin (Bear)',          w:-2,chk:s=>s.nl?.kagi?.yin},
  {cat:'Noiseless',tab:'nl',label:'Kagi Fresh Reversal (Yang)',w:3,chk:s=>s.nl?.kagi?.just_rev&&s.nl?.kagi?.yang},
  {cat:'Noiseless',tab:'nl',label:'3-Line Break Bull Reversal',w:3,chk:s=>s.nl?.lb3?.bull_rev},
  {cat:'Noiseless',tab:'nl',label:'3-Line Break Bear Reversal',w:-3,chk:s=>s.nl?.lb3?.bear_rev},
  {cat:'Noiseless',tab:'nl',label:'RRG Leading Quadrant',     w:3, chk:s=>s.nl?.rrg?.quadrant==='leading'},
  {cat:'Noiseless',tab:'nl',label:'RRG Lagging Quadrant',     w:-2,chk:s=>s.nl?.rrg?.quadrant==='lagging'},

  // ── VSA (Experimental) ──
  {cat:'VSA',tab:'xp',label:'VSA Effort Up (Smart Money Buying)',w:2,chk:s=>s.xp?.vsa?.effort_up},
  {cat:'VSA',tab:'xp',label:'VSA Effort Down (Smart Money Selling)',w:-2,chk:s=>s.xp?.vsa?.effort_dn},
  {cat:'VSA',tab:'xp',label:'VSA High Volume Bull Bar',       w:2, chk:s=>s.xp?.vsa?.big_vol_bull},
  {cat:'VSA',tab:'xp',label:'VSA High Volume Bear Bar',       w:-2,chk:s=>s.xp?.vsa?.big_vol_bear},

  // ── Pivot Bounce ──
  {cat:'PivotBounce',tab:'piv',label:'Near Pivot S1 (Support)',w:1, chk:s=>s.pb?.near_S1},
  {cat:'PivotBounce',tab:'piv',label:'Near Pivot S2 (Support)',w:1, chk:s=>s.pb?.near_S2},
  {cat:'PivotBounce',tab:'piv',label:'Near Pivot R1 (Resistance)',w:-1,chk:s=>s.pb?.near_R1},
  {cat:'PivotBounce',tab:'piv',label:'Near Pivot R2 (Resistance)',w:-1,chk:s=>s.pb?.near_R2},
  {cat:'PivotBounce',tab:'piv',label:'Above Pivot Point (Bullish)',w:1,chk:s=>s.pb?.above_P},
  {cat:'PivotBounce',tab:'piv',label:'At Support Zone',       w:1, chk:s=>s.pb?.at_sup},
  {cat:'PivotBounce',tab:'piv',label:'At Resistance Zone',    w:-1,chk:s=>s.pb?.at_res},
  {cat:'PivotBounce',tab:'piv',label:'Below Support S1 (Bearish)',w:-1,chk:s=>s.pb?.below_S1},

  // ── Inside Week / Trendline (Breakout Pro extras) ──
  {cat:'BreakoutPro',tab:'bp',label:'Inside Week',            w:1, chk:s=>s.bp?.iw?.iw1},
  {cat:'BreakoutPro',tab:'bp',label:'Double Inside Week',     w:2, chk:s=>s.bp?.iw?.iw2},
  {cat:'BreakoutPro',tab:'bp',label:'Inside Week in Uptrend', w:2, chk:s=>s.bp?.iw?.iw_uptrend},

  // ── Advanced extras ──
  {cat:'Advanced',tab:'adv',label:'VCP (Volatility Contraction)',w:3,chk:s=>s.adv?.vcp?.is_vcp},
  {cat:'Advanced',tab:'adv',label:'Darvas Box Breakout',      w:3, chk:s=>s.adv?.darvas?.breakout},
  {cat:'Advanced',tab:'adv',label:'Wyckoff Spring (Accumulation)',w:3,chk:s=>s.adv?.wyckoff?.spring},
  {cat:'Advanced',tab:'adv',label:'Turtle Soup (False BO Reversal)',w:2,chk:s=>s.adv?.turtle?.bull_setup},
  {cat:'Advanced',tab:'adv',label:'TD Sequential Buy Setup 9',w:3, chk:s=>s.adv?.td?.buy9},
  {cat:'Advanced',tab:'adv',label:'TD Sequential Sell Setup 9',w:-3,chk:s=>s.adv?.td?.sell9},
  {cat:'Advanced',tab:'adv',label:'Elder Triple Screen Buy',  w:2, chk:s=>s.adv?.elder?.buy},
  {cat:'Advanced',tab:'adv',label:'Elder Triple Screen Sell', w:-2,chk:s=>s.adv?.elder?.sell},

  // ── Donchian extras ──
  {cat:'Trend',tab:'adv',label:'Donchian Squeeze (Pre-Breakout)',w:1,chk:s=>s.adv?.dc?.squeeze},
  {cat:'Trend',tab:'adv',label:'GMMA Compression (Trend Change Pending)',w:0,chk:s=>s.adv?.gmma?.compress,info:true},

  // ── Fib / Chandelier ──
  {cat:'Momentum',tab:'t3',label:'Near Fibonacci 61.8% Retracement',w:1,chk:s=>s.t3?.fib?.near&&s.t3?.fib?.nearest==='61.8%'},
  {cat:'Momentum',tab:'t3',label:'Chandelier Exit Bullish',   w:2, chk:s=>s.t3?.ce?.bull},
  {cat:'Momentum',tab:'t3',label:'Chandelier Exit Bearish',   w:-2,chk:s=>s.t3?.ce?.bear},
];

function scoreStock(s){
  let bullW=0, bearW=0; const matched=[];
  for(const d of SIGNAL_DEFS){
    let ok=false;
    try{ ok=!!d.chk(s); }catch(e){ ok=false; }
    if(ok){
      matched.push(d);
      if(d.w>0) bullW+=d.w; else if(d.w<0) bearW+=Math.abs(d.w);
    }
  }
  // ── Confluence bonus: reward signals spanning MULTIPLE categories over
  //    redundant signals piling up in one category (e.g. 5 candlestick hits
  //    shouldn't outweigh 1 hit each from Trend + SMC + Smart Rank + BreakoutPro) ──
  const bullCats=new Set(matched.filter(m=>m.w>0).map(m=>m.cat));
  const bearCats=new Set(matched.filter(m=>m.w<0).map(m=>m.cat));
  // +8% of bullW per distinct bullish category beyond the first (capped at +40%)
  const bullConfluence=Math.min(bullCats.size>0?(bullCats.size-1)*0.08:0, 0.40);
  const bearConfluence=Math.min(bearCats.size>0?(bearCats.size-1)*0.08:0, 0.40);
  const bullWAdj=bullW*(1+bullConfluence);
  const bearWAdj=bearW*(1+bearConfluence);

  const total=bullWAdj+bearWAdj;
  const score=total>0?Math.round(bullWAdj/total*100):50;
  let grade,color;
  if(score>=75){grade='Strong Buy';color='#00e5a0';}
  else if(score>=60){grade='Buy';color='#4ade80';}
  else if(score>=40){grade='Neutral';color='#94a3b8';}
  else if(score>=25){grade='Sell';color='#fb923c';}
  else{grade='Strong Sell';color='#ef4444';}
  return {score,grade,color,bullW,bearW,matched,
          bullCats:bullCats.size, bearCats:bearCats.size,
          confluenceBonus:Math.round(bullConfluence*100),
          bullSignals:matched.filter(m=>m.w>0),
          bearSignals:matched.filter(m=>m.w<0)};
}

// ════════════════════════════════════════════════════════════════════════════
// 🔍 STOCK LOOKUP — search a stock, see all matched signals + score
// ════════════════════════════════════════════════════════════════════════════
function renderLookup(){
  return `<div style="padding:14px;max-width:900px">
    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">
      <input id="lookup-input" type="text" placeholder="Enter stock symbol e.g. RELIANCE, TCS, INFY..."
        style="flex:1;min-width:240px;padding:10px 14px;background:var(--bg2);border:1px solid var(--brd);
        border-radius:6px;color:var(--txt);font-size:14px;font-family:var(--mono);text-transform:uppercase"
        onkeydown="if(event.key==='Enter')lookupStock()">
      <button class="btn" style="background:var(--acc);color:#000" onclick="lookupStock()">🔍 Search</button>
    </div>
    <div id="lookup-suggest" style="margin-bottom:10px"></div>
    <div id="lookup-result"></div>
  </div>`;
}


// ════════════════════════════════════════════════════════════════════════════
// 🎯 SWING SETUP — EMA Squeeze + MACD 5/35 (D/W/M) + Swing Trade Suggestions
// ════════════════════════════════════════════════════════════════════════════
const SWING_INFO={
  ema_sq_d:'<b>Daily EMA Squeeze</b> · 10/20/50/100/150/200 EMA all within 2.5% of each other on the DAILY chart · A tight coil across every major moving average — the calm before a directional expansion',
  ema_sq_d_tight:'<b>Daily Ultra-Tight Squeeze</b> · All loaded EMAs (10-200) within 1% of each other · Extremely rare — historically precedes the largest volatility expansions',
  ema_sq_w:'<b>Weekly EMA Squeeze</b> · 10/20/50/100-week EMAs converged within 2.5% · A multi-month consolidation resolving — breakouts from weekly squeeze tend to run for weeks',
  ema_sq_m:'<b>Monthly EMA Squeeze</b> · 10/20/50-month EMAs converged · Multi-year base — when this resolves, the move is typically a major trend change lasting quarters',
  ema_sq_d_bull:'<b>Daily Squeeze + Bull Stack</b> · EMAs are tight AND already arranged bullishly (shorter above longer) · Coiling INSIDE an uptrend — highest-probability continuation setup',
  ema_sq_w_bull:'<b>Weekly Squeeze + Bull Stack</b> · Weekly EMAs tight and bullishly stacked · Multi-week pause within an established uptrend',
  macd535_d_bull:'<b>Daily MACD 5/35/5 Bullish Cross</b> · Fast 5-EMA crossed above slow 35-EMA today on the daily chart · A faster variant of standard MACD(12,26,9) — catches swing entries 1-2 days earlier',
  macd535_d_bear:'<b>Daily MACD 5/35/5 Bearish Cross</b> · Fast EMA crossed below slow EMA — momentum turning down on daily timeframe',
  macd535_w_bull:'<b>Weekly MACD 5/35/5 Bullish Cross</b> · Weekly fast/slow EMA cross — signals a multi-week swing move beginning, ideal for 1-3 month holding periods',
  macd535_w_bear:'<b>Weekly MACD 5/35/5 Bearish Cross</b> · Weekly momentum turning down — consider exiting swing longs or avoiding new entries',
  macd535_m_bull:'<b>Monthly MACD 5/35/5 Bullish Cross</b> · Monthly cross — positional/investment-grade signal, holding period often 3-12 months',
  macd535_m_bear:'<b>Monthly MACD 5/35/5 Bearish Cross</b> · Monthly bearish cross — major trend deterioration, longer-term caution',
  macd535_d_zero_bull:'<b>Daily MACD 5/35 Zero-Line Cross Up</b> · The fast EMA crossed above the slow EMA AND the MACD line itself crossed above zero — stronger confirmation than signal-line cross alone',
  macd535_align:'<b>All-Timeframe MACD 5/35 Alignment</b> · Daily, Weekly AND Monthly MACD(5,35,5) are all bullish simultaneously · Full timeframe alignment — the strongest possible momentum confirmation across short, medium and long-term traders',
  swing_buy:'<b>Swing Buy Candidates</b> · Composite signal score ≥60 (Buy or Strong Buy grade) · Each result includes a suggested Entry (current price), Stop-Loss (below nearest pivot support / 1.5×ATR), and Target (next resistance / 2.5×ATR) for a 1-week to 3-month swing horizon',
  swing_strong:'<b>Strong Swing Buy</b> · Score ≥75 (Strong Buy) with confluence across 4+ signal categories · The highest-conviction swing setups in the universe right now, with full entry/stop/target plan',
  swing_breakout_ready:'<b>Breakout-Ready Setup</b> · Stock is BOTH in an EMA squeeze (coiled) AND has a fresh MACD 5/35 bullish cross (momentum trigger firing) · The combination of compression + trigger is the textbook breakout entry',
};
function updateSwingInfo(){const s=document.getElementById('swing-strat').value;setFbar('🎯 <b>Swing Setup</b> · '+(SWING_INFO[s]||s));}

function _swingTargets(s){
  // Suggest Entry/Stop/Target for a 1-week to 3-month swing using pivot levels + ATR fallback
  const price=s.price; const pb=s.pb||{}; const ce=s.t3?.ce||{};
  const atr=ce.atr||price*0.02; // fallback ~2% if ATR missing
  const stop = pb.S1 && pb.S1<price ? pb.S1 : price-1.5*atr;
  const target = pb.R2 && pb.R2>price ? pb.R2 : price+2.5*atr;
  const rr = (price-stop)>0 ? ((target-price)/(price-stop)).toFixed(2) : '—';
  return {entry:price, stop:Math.round(stop*100)/100, target:Math.round(target*100)/100, rr};
}

function scanSwing(){
  const strat=document.getElementById('swing-strat').value;
  const rsMin=parseInt(document.getElementById('swing-rs').value)||0;
  const prMin=parseFloat(document.getElementById('swing-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('swing-pmax').value)||Infinity;
  updateSwingInfo(); rows=[];
  for(const s of S){
    if(!passesIdx(s))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if((s.rs||0)<rsMin)continue;
    const sq=s.swing?.ema_sq||{}; const m5=s.swing?.macd535||{};
    const d=sq.d||{}, w=sq.w||{}, m=sq.m||{};
    const md=m5.d||{}, mw=m5.w||{}, mm=m5.m||{};
    let matched=false,sig='',extra={};

    if(strat==='ema_sq_d')        {matched=!!d.squeeze;                  sig='🗜 D-EMA Sq';     extra={tight:d.tight_pct+'%',n:d.n_emas};}
    if(strat==='ema_sq_d_tight')  {matched=!!d.squeeze_tight;            sig='🗜🗜 D-Ultra';     extra={tight:d.tight_pct+'%'};}
    if(strat==='ema_sq_w')        {matched=!!w.squeeze;                  sig='🗜 W-EMA Sq';     extra={tight:(w.tight_pct||0)+'%',n:w.n_emas};}
    if(strat==='ema_sq_m')        {matched=!!m.squeeze;                  sig='🗜 M-EMA Sq';     extra={tight:(m.tight_pct||0)+'%',n:m.n_emas};}
    if(strat==='ema_sq_d_bull')   {matched=!!d.squeeze&&!!d.bull_stack;  sig='🗜📈 D-Sq Bull';  extra={tight:d.tight_pct+'%'};}
    if(strat==='ema_sq_w_bull')   {matched=!!w.squeeze&&!!w.bull_stack;  sig='🗜📈 W-Sq Bull';  extra={tight:(w.tight_pct||0)+'%'};}

    if(strat==='macd535_d_bull')  {matched=!!md.cross_bull; sig='📊 D MACD5/35↑'; extra={hist:md.hist};}
    if(strat==='macd535_d_bear')  {matched=!!md.cross_bear; sig='📊 D MACD5/35↓'; extra={hist:md.hist};}
    if(strat==='macd535_w_bull')  {matched=!!mw.cross_bull; sig='📊 W MACD5/35↑'; extra={hist:mw.hist};}
    if(strat==='macd535_w_bear')  {matched=!!mw.cross_bear; sig='📊 W MACD5/35↓'; extra={hist:mw.hist};}
    if(strat==='macd535_m_bull')  {matched=!!mm.cross_bull; sig='📊 M MACD5/35↑'; extra={hist:mm.hist};}
    if(strat==='macd535_m_bear')  {matched=!!mm.cross_bear; sig='📊 M MACD5/35↓'; extra={hist:mm.hist};}
    if(strat==='macd535_d_zero_bull'){matched=!!md.zero_bull; sig='📊 D Zero↑'; extra={macd:md.macd};}
    if(strat==='macd535_align')   {matched=!!md.bull&&!!mw.bull&&!!mm.bull; sig='📊 D+W+M Align'; extra={d:md.bull?'✓':'—',w:mw.bull?'✓':'—',m:mm.bull?'✓':'—'};}

    if(strat==='swing_buy'||strat==='swing_strong'||strat==='swing_breakout_ready'){
      const sc=scoreStock(s);
      if(strat==='swing_buy')      matched=sc.score>=60;
      if(strat==='swing_strong')   matched=sc.score>=75&&sc.bullCats>=4;
      if(strat==='swing_breakout_ready') matched=!!d.squeeze&&!!md.cross_bull;
      if(matched){
        const tgt=_swingTargets(s);
        sig='🎯 '+sc.grade+' ('+sc.score+')';
        extra={entry:tgt.entry,stop:tgt.stop,target:tgt.target,rr:tgt.rr+'x',horizon:'1W-3M'};
      }
    }

    if(!matched)continue;
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig,extra,strat,_tab:'swing',sector:s.sector||''});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

function lookupStock(){
  const q=(document.getElementById('lookup-input').value||'').trim().toUpperCase();
  const sugEl=document.getElementById('lookup-suggest');
  const resEl=document.getElementById('lookup-result');
  if(!q){resEl.innerHTML='';sugEl.innerHTML='';return;}
  const exact=S.find(s=>s.sym===q);
  if(exact){ sugEl.innerHTML=''; renderLookupResult(exact); return; }
  // Suggestions: partial matches
  const matches=S.filter(s=>s.sym.includes(q)).slice(0,10);
  if(matches.length===0){
    resEl.innerHTML='<div class="nodata">No stock found matching "'+q+'"</div>';
    sugEl.innerHTML=''; return;
  }
  resEl.innerHTML='';
  sugEl.innerHTML='<div style="font-size:11px;color:var(--mu);margin-bottom:6px">Did you mean:</div>'
    +'<div style="display:flex;gap:6px;flex-wrap:wrap">'
    +matches.map(s=>`<button class="btn btn-out" style="padding:5px 12px;font-size:12px" onclick="document.getElementById('lookup-input').value='${s.sym}';lookupStock()">${s.sym}</button>`).join('')
    +'</div>';
}

function renderLookupResult(s){
  const sc=scoreStock(s);
  const resEl=document.getElementById('lookup-result');
  const f2=v=>typeof v==='number'?v.toFixed(2):(v||'—');
  const pct=(v,b)=>b&&v?((v-b)/b*100).toFixed(1)+'%':'—';

  // Group matched signals by category
  const cats={};
  for(const m of sc.matched){
    if(!cats[m.cat]) cats[m.cat]=[];
    cats[m.cat].push(m);
  }
  const catOrder=['Trend','Momentum','Volatility','SMC','MultiIndicator','BreakoutPro','SmartRank','Patterns','Gap'];
  const catLabels={Trend:'📈 Trend',Momentum:'⚡ Momentum',Volatility:'🗜 Volatility/Breakout',
    SMC:'🎯 Smart Money Concepts',MultiIndicator:'🏆 Minervini/Weinstein',
    BreakoutPro:'🏹 Breakout Pro',SmartRank:'🔢 Smart Rank',Patterns:'🎨 Patterns',Gap:'⚡ Gaps'};

  let catsHtml='';
  for(const cat of catOrder){
    const items=cats[cat]; if(!items||!items.length)continue;
    catsHtml+=`<div style="margin-bottom:10px">
      <div style="font-size:10px;font-weight:700;color:var(--acc);letter-spacing:1px;margin-bottom:4px;font-family:var(--mono)">${catLabels[cat]||cat}</div>
      <div style="display:flex;flex-direction:column;gap:3px">
        ${items.map(m=>{
          const clr=m.w>0?'#00e5a0':m.w<0?'#ef4444':'#94a3b8';
          const bg =m.w>0?'rgba(0,229,160,.06)':m.w<0?'rgba(239,68,68,.06)':'rgba(148,163,184,.06)';
          return `<div style="display:flex;align-items:center;gap:6px;font-size:12px;padding:4px 8px;
          background:${bg};border-radius:4px;border-left:2px solid ${clr}">
          <span style="color:${clr};font-weight:700;min-width:24px">${m.w>0?'+'+m.w:m.w<0?m.w:'•'}</span>
          <span style="flex:1">${m.label}</span>
          <span style="color:var(--mu);font-size:10px;cursor:pointer" onclick="switchTab('${m.tab}')">→ ${TAB_NAMES[m.tab]||m.tab}</span>
        </div>`;}).join('')}
      </div>
    </div>`;
  }
  if(!catsHtml) catsHtml='<div class="nodata">No active signals found for this stock currently.</div>';
  catsHtml='<div style="font-size:11px;font-weight:700;color:var(--txt);margin-bottom:10px">📋 Appears in '+sc.matched.length+' scanner'+(sc.matched.length===1?'':'s')+':</div>'+catsHtml;

  // Gauge SVG (semicircle)
  const angle=sc.score/100*180-90; // -90 to +90 degrees
  const rad=angle*Math.PI/180;
  const gx=100+70*Math.cos(rad), gy=90+70*Math.sin(rad);
  const gaugeSvg=`<svg width="200" height="110" viewBox="0 0 200 110">
    <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="var(--brd)" stroke-width="14"/>
    <path d="M 20 90 A 80 80 0 0 1 180 90" fill="none" stroke="${sc.color}" stroke-width="14"
      stroke-dasharray="${sc.score/100*251.2} 251.2"/>
    <line x1="100" y1="90" x2="${gx}" y2="${gy}" stroke="var(--txt)" stroke-width="2.5"/>
    <circle cx="100" cy="90" r="4" fill="var(--txt)"/>
    <text x="100" y="75" text-anchor="middle" font-size="24" font-weight="700" fill="${sc.color}" font-family="var(--mono)">${sc.score}</text>
    <text x="100" y="105" text-anchor="middle" font-size="12" font-weight="700" fill="${sc.color}">${sc.grade}</text>
  </svg>`;

  const stt=s.t1?.stt||{};
  resEl.innerHTML=`<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:16px;align-items:center;
    padding:14px;background:var(--bg2);border:1px solid var(--brd);border-radius:8px">
    <div style="min-width:160px">
      <div style="font-size:24px;font-weight:700;color:var(--txt)">${s.sym}</div>
      <div style="font-size:12px;color:var(--mu);margin:4px 0">${s.sector||'Other'} · ${s.idx?'Nifty '+s.idx:'Other'}${s.fno?' · <span style="color:#00e5a0">FNO</span>':''}</div>
      <div style="font-size:20px;color:var(--txt);font-weight:600">₹${f2(s.price)}</div>
      <div style="font-size:11px;color:var(--mu)">RS Rating: ${s.rs||0}/99 · vs 200DMA: ${pct(s.price,s.dma200)}</div>
      <div style="font-size:11px;color:var(--mu)">SuperTrend: ${stt.bull?'🟢 Bull':stt.bear?'🔴 Bear':'—'}${stt.consec?' ('+stt.consec+'×)':''}</div>
    </div>
    <div style="flex:1;display:flex;justify-content:center;min-width:200px">${gaugeSvg}</div>
    <div style="min-width:160px;text-align:right">
      <div style="font-size:11px;color:var(--mu)">Bullish weight: <span style="color:#00e5a0;font-weight:700">+${sc.bullW}</span> across <b style="color:var(--txt)">${sc.bullCats}</b> categories</div>
      <div style="font-size:11px;color:var(--mu)">Bearish weight: <span style="color:#ef4444;font-weight:700">-${sc.bearW}</span> across <b style="color:var(--txt)">${sc.bearCats}</b> categories</div>
      ${sc.confluenceBonus>0?'<div style="font-size:10px;color:var(--acc);margin-top:2px">🔗 Confluence bonus: +'+sc.confluenceBonus+'% (signals span multiple categories)</div>':''}
      <div style="font-size:11px;color:var(--mu);margin-top:6px">Appears in <b style="color:var(--txt)">${sc.matched.length}</b> scanners</div>
      <button class="btn btn-out" style="margin-top:8px;font-size:11px" onclick="showStockDetail('${s.sym}')">View Full Detail →</button>
    </div>
  </div>
  ${catsHtml}`;
}

// ════════════════════════════════════════════════════════════════════════════
// 📋 INDEX VIEW — ranked list of stocks by score for selected index
// ════════════════════════════════════════════════════════════════════════════
function renderIndexView(){
  return `<div style="padding:14px">
    <div style="display:flex;gap:10px;margin-bottom:12px;align-items:center;flex-wrap:wrap">
      <label style="font-size:12px;color:var(--mu)">Index</label>
      <select id="iv-idx" style="padding:6px 10px;background:var(--bg2);border:1px solid var(--brd);
        border-radius:5px;color:var(--txt);font-size:12px" onchange="scanIndexView()">
        <option value="0">All Stocks</option>
        <option value="fno">F&O Only</option>
        <option value="50">Nifty 50</option>
        <option value="100">Nifty 100</option>
        <option value="200">Nifty 200</option>
        <option value="500">Nifty 500</option>
        <option value="750">Nifty 750</option>
      </select>
      <label style="font-size:12px;color:var(--mu)">Sort</label>
      <select id="iv-sort" style="padding:6px 10px;background:var(--bg2);border:1px solid var(--brd);
        border-radius:5px;color:var(--txt);font-size:12px" onchange="scanIndexView()">
        <option value="score_desc">Score: High → Low (Best Buy)</option>
        <option value="score_asc">Score: Low → High (Best Sell)</option>
        <option value="rs_desc">RS Rating: High → Low</option>
        <option value="sym">Symbol A-Z</option>
      </select>
      <span id="iv-count" style="font-size:11px;color:var(--mu);margin-left:auto"></span>
    </div>
    <div id="iv-table"></div>
  </div>`;
}

function scanIndexView(){
  const idxSel=document.getElementById('iv-idx').value;
  const sortSel=document.getElementById('iv-sort').value;
  let list=S.slice();
  if(idxSel==='fno') list=list.filter(s=>s.fno);
  else if(idxSel!=='0') list=list.filter(s=>s.idx&&s.idx<=parseInt(idxSel));

  // Compute scores
  list=list.map(s=>({s,sc:scoreStock(s)}));

  if(sortSel==='score_desc') list.sort((a,b)=>b.sc.score-a.sc.score);
  else if(sortSel==='score_asc') list.sort((a,b)=>a.sc.score-b.sc.score);
  else if(sortSel==='rs_desc') list.sort((a,b)=>(b.s.rs||0)-(a.s.rs||0));
  else list.sort((a,b)=>a.s.sym.localeCompare(b.s.sym));

  document.getElementById('iv-count').textContent=list.length+' stocks';

  const rows=list.map(({s,sc})=>{
    const f2=v=>typeof v==='number'?v.toFixed(2):(v||'—');
    const pct=(v,b)=>b&&v?((v-b)/b*100).toFixed(1)+'%':'—';
    const dist=s.dma200?((s.price-s.dma200)/s.dma200*100):0;
    return `<tr style="cursor:pointer" onclick="showScorePopup('${s.sym}')">
      <td class="csym" style="white-space:nowrap"><span class="sd-ico" onmouseenter="_sdHover('${s.sym}')" title="Stock detail">i</span><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A${s.sym}" target="_blank" rel="noopener" onclick="event.stopPropagation()">${s.sym}</a></td>
      <td>${s.sector||'—'}</td>
      <td class="num">₹${f2(s.price)}</td>
      <td class="num" style="color:${dist>=0?'#00e5a0':'#ef4444'}">${pct(s.price,s.dma200)}</td>
      <td class="num">${s.rs||0}</td>
      <td class="num">
        <div style="display:flex;align-items:center;gap:6px;justify-content:flex-end">
          <div style="width:60px;height:6px;background:var(--brd);border-radius:3px;overflow:hidden">
            <div style="width:${sc.score}%;height:100%;background:${sc.color}"></div>
          </div>
          <span style="color:${sc.color};font-weight:700;min-width:32px;text-align:right">${sc.score}</span>
        </div>
      </td>
      <td style="color:${sc.color};font-weight:600;font-size:11px">${sc.grade}</td>
      <td class="mu" style="font-size:11px">${sc.matched.length} scanners</td>
    </tr>`;
  }).join('');

  document.getElementById('iv-table').innerHTML=`<table style="width:100%">
    <thead><tr>
      <th>Symbol</th><th>Sector</th><th class="num">Price</th><th class="num">vs 200DMA</th>
      <th class="num">RS</th><th class="num" style="min-width:120px">Score</th><th>Grade</th><th>Signals</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

// ════════════════════════════════════════════════════════════════════════════
// 📊 SCORE POPUP — used from Index View: shows score + matched signals in modal
// ════════════════════════════════════════════════════════════════════════════
function showScorePopup(sym){
  showStockDetail(sym);
  // Inject score breakdown into the AI output area as a quick summary
  const s=S.find(x=>x.sym===sym); if(!s)return;
  const sc=scoreStock(s);
  const cats={};
  for(const m of sc.matched){ if(!cats[m.cat]) cats[m.cat]=[]; cats[m.cat].push(m); }
  const catLabels={Trend:'📈 Trend',Momentum:'⚡ Momentum',Volatility:'🗜 Volatility',
    SMC:'🎯 SMC',MultiIndicator:'🏆 Minervini/Weinstein',BreakoutPro:'🏹 Breakout Pro',
    SmartRank:'🔢 Smart Rank',Patterns:'🎨 Patterns',Gap:'⚡ Gaps'};
  let html=`<div style="padding:10px;font-size:11px">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
      <span style="font-size:20px;font-weight:700;color:${sc.color}">${sc.score}</span>
      <span style="color:${sc.color};font-weight:600">${sc.grade}</span>
      <span style="color:var(--mu)">· Appears in ${sc.matched.length} scanners (${sc.bullSignals.length} bull / ${sc.bearSignals.length} bear)</span>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:4px">`;
  for(const cat in cats){
    html+=`<span style="background:rgba(59,158,255,.08);border:1px solid var(--brd);border-radius:4px;
      padding:2px 8px;font-size:10px" title="${cats[cat].map(m=>m.label).join(', ')}">${catLabels[cat]||cat}: ${cats[cat].length}</span>`;
  }
  html+='</div>';
  // Full scanner name list
  html+='<div style="margin-top:8px;display:flex;flex-direction:column;gap:2px;max-height:140px;overflow-y:auto">';
  for(const m of sc.matched){
    const clr=m.w>0?'#00e5a0':m.w<0?'#ef4444':'#94a3b8';
    html+='<div style="font-size:10px;color:var(--txt)"><span style="color:'+clr+';font-weight:700;display:inline-block;width:20px">'+(m.w>0?'+'+m.w:m.w<0?m.w:'•')+'</span>'+m.label+' <span style="color:var(--mu)">('+(TAB_NAMES[m.tab]||m.tab)+')</span></div>';
  }
  html+='</div></div>';
  document.getElementById('ai-output').innerHTML=html;
}

function buildCols(tab,r0){
  const common=[
    {k:'sym',  h:'Symbol',     fn:r=>symCell(r.sym)},
    {k:'idx',  h:'Index',      fn:r=>idxBadge(r.idx)},
    {k:'price',h:'Close',      fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
    {k:'rs',   h:'RS',         fn:r=>rsCell(r.rs)},
    {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
    {k:'w52h', h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
    {k:'avol', h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
    {k:'date', h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
    {k:'_spark',h:'Spark',fn:r=>sparkCell(r.sym),dv:true},
    {k:'_age',  h:'Age',  fn:r=>ageCell(r.date),  dv:true},
  ];

  if(tab==='piv'){
    const prox=parseFloat(document.getElementById('sp-pr').value);
    const type=r0?.type||'fibonacci';const meta=r0?.meta||TYPE_META[type];
    const lvKey=r0?.lvKey;const multi=r0?.multi;
    const lvList=(meta.levels||[]).filter(l=>l!==lvKey&&l!=='P');
    return[
      {k:'sym',h:'Symbol',fn:r=>symCell(r.sym),ns:true},
      {k:'idx',h:'Index',fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'lv',h:(multi?'Hit Lv':lvKey)+' Level',fn:r=>`<td class="clv">${multi?`<span class="${LV_CSS[r.lvKey]||''}">${r.lvKey} </span>`:''  }${r.lv.toFixed(2)}</td>`,hcls:'hl'},
      {k:'dist',h:'Dist %',fn:r=>distCell(r.dist,prox)},
      {k:'rs',h:'RS',fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
      {k:'trend',h:'Trend',sortFn:r=>r.smc?.trend==='bull'?1:r.smc?.trend==='bear'?-1:0,fn:r=>`<td class="${r.smc.trend==='bull'?'smc-bull':r.smc.trend==='bear'?'smc-bear':'smc-range'}">${r.smc.trend?.toUpperCase()||'—'}</td>`},
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
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'s50',h:'50 SMA',fn:r=>`<td class="${r.price>r.s50?'pos':'neg'}">${r.s50}</td>`},
      {k:'s150',h:'150 SMA',fn:r=>`<td class="${r.price>r.s150?'pos':'neg'}">${r.s150}</td>`},
      {k:'s200',h:'200 SMA',fn:r=>`<td class="${r.price>r.s200?'pos':'neg'}">${r.s200}</td>`},
      {k:'s30w',h:'30W SMA',fn:r=>`<td class="${r.price>r.s30w?'pos':'neg'}">${r.s30w}</td>`},
      {k:'w52h',h:'vs 52WH',fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',h:'AvgVol20',fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='adv'){
    // Dynamic extra columns based on strategy
    const extraCols = rows.length ? Object.keys(rows[0].extra||{}).map(k=>({
      k:'extra_'+k, h:k,
      fn: r=>{
        const v=r.extra?.[k];
        if(v===undefined||v===null||v==='')return`<td class="mu">—</td>`;
        if(typeof v==='boolean') return`<td class="${v?'pos':'neg'}">${v?'✓':'✗'}</td>`;
        if(k==='sig') return`<td style="color:var(--warn);font-weight:700">${v}</td>`;
        return`<td class="mu">${v}</td>`;
      }
    })) : [];
    return[
      {k:'sym',   h:'Symbol',    fn:r=>symCell(r.sym)},
      {k:'idx',   h:'Index',     fn:r=>idxBadge(r.idx)},
      {k:'price', h:'Close',     fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      ...extraCols,
      {k:'rs',    h:'RS',        fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',  h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',  h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',  h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='t1'){
    const extraCols=rows.length?Object.keys(rows[0].extra||{}).map(k=>({
      k:'ex_'+k,h:k,
      fn:r=>{const v=r.extra?.[k];
        if(v===undefined||v===null||v==='')return`<td class="mu">—</td>`;
        if(k==='rsi')return`<td class="${v<=30?'pos':v>=70?'neg':''}" style="font-weight:700">${v}</td>`;
        return`<td class="mu">${v}</td>`;}
    })):[];
    return[
      {k:'sym',  h:'Symbol',   fn:r=>symCell(r.sym)},
      {k:'idx',  h:'Index',    fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',    fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'sig',  h:'Signal',   fn:r=>`<td style="color:#c47aff;font-weight:700;font-size:12px">${r.sig}</td>`},
      {k:'rsi',  h:'RSI',      fn:r=>`<td class="${r.rsi<=30?'pos':r.rsi>=70?'neg':''}">${r.rsi}</td>`},
      ...extraCols,
      {k:'rs',       h:'RS',        fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',     h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',     h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',     h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='t2'){
    const extraCols=rows.length?Object.keys(rows[0].extra||{}).map(k=>({
      k:'e2_'+k,h:k,
      fn:r=>{const v=r.extra?.[k];
        if(v===undefined||v===null||v==='')return`<td class="mu">—</td>`;
        if(k==='z20'||k==='z50'||k==='z200'){
          const fv=parseFloat(v)||0;
          return`<td class="${fv<=-2?'pos':fv>=2?'neg':''}" style="font-weight:600">${v}</td>`;}
        return`<td class="mu">${v}</td>`;}
    })):[];
    return[
      {k:'sym',  h:'Symbol',   fn:r=>symCell(r.sym)},
      {k:'idx',  h:'Index',    fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',    fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'sig',  h:'Signal',   fn:r=>`<td style="color:var(--acc);font-weight:700;font-size:12px">${r.sig}</td>`},
      ...extraCols,
      {k:'rs',       h:'RS',        fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',     h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',     h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',     h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='t3'){
    const extraCols=rows.length?Object.keys(rows[0].extra||{}).map(k=>({
      k:'e3_'+k, h:k,
      fn:r=>{const v=r.extra?.[k];
        if(v===undefined||v===null||v==='')return`<td class="mu">—</td>`;
        if(k==='adx'){const fv=parseFloat(v)||0; return`<td class="${fv>=40?'pos':fv>=25?'':'mu'}" style="font-weight:600">${v}</td>`;}
        if(k==='k'||k==='d'){const fv=parseFloat(v)||0; return`<td class="${fv<=20?'pos':fv>=80?'neg':''}">${v}</td>`;}
        if(k==='wr'){const fv=parseFloat(v)||0; return`<td class="${fv<=-80?'pos':fv>=-20?'neg':''}">${v}</td>`;}
        return`<td class="mu">${v}</td>`;}
    })):[];
    return[
      {k:'sym',  h:'Symbol',   fn:r=>symCell(r.sym)},
      {k:'idx',  h:'Index',    fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',    fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'sig',  h:'Signal',   fn:r=>`<td style="color:var(--gold);font-weight:700;font-size:12px">${r.sig}</td>`},
      ...extraCols,
      {k:'rs',       h:'RS',        fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',     h:'vs 52WH',   fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',     h:'AvgVol20',  fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',     h:'Last Date', fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='ti'){
    const extraCols=rows.length?Object.keys(rows[0].extra||{}).map(k=>({
      k:'eti_'+k,h:k,fn:r=>`<td class="mu">${r.extra?.[k]??'—'}</td>`
    })):[];
    return[
      {k:'sym',  h:'Symbol',  fn:r=>symCell(r.sym)},
      {k:'idx',  h:'Index',   fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',   fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'sig',  h:'Pattern / Signal',fn:r=>`<td style="color:#ff9500;font-weight:700;font-size:12px">${r.sig}</td>`},
      ...extraCols,
      {k:'rs',      h:'RS',       fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',    h:'vs 52WH',  fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',    h:'AvgVol20', fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',    h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>`},
    ];
  }
  if(tab==='nl'||tab==='xp'||tab==='patt'||tab==='br'||tab==='gap'||tab==='combo'||tab==='bp'||tab==='sr'||tab==='swing'){
    const colorMap={nl:'#e879f9',xp:'#f59e0b',patt:'#06b6d4',br:'#84cc16'};
    const color=colorMap[tab]||'#888';
    const extraCols=rows.length?Object.keys(rows[0].extra||{}).map(k=>({
      k:'e_'+k,h:k,fn:r=>`<td class="mu">${r.extra?.[k]??'—'}</td>`
    })):[];
    return[
      {k:'sym',  h:'Symbol',  fn:r=>symCell(r.sym)},
      {k:'idx',  h:'Index',   fn:r=>idxBadge(r.idx)},
      {k:'price',h:'Close',   fn:r=>`<td class="cpr">${r.price.toFixed(2)}</td>`},
      {k:'sig',  h:'Signal',  fn:r=>`<td style="color:${color};font-weight:700;font-size:12px">${r.sig}</td>`},
      ...extraCols,
      {k:'rs',      h:'RS',       fn:r=>rsCell(r.rs)},
      {k:'above200',h:'vs 200DMA',sortFn:r=>r.dma200?(r.price-r.dma200)/r.dma200*100:0,fn:r=>dmaCell(r.price,r.dma200,r.above200)},
      {k:'w52h',    h:'vs 52WH',  fn:r=>w52Cell(r.price,r.w52h)},
      {k:'avol',    h:'AvgVol20', fn:r=>`<td class="mu">${fmtVol(r.avol)}</td>`},
      {k:'date',    h:'Last Date',fn:r=>`<td class="mu">${r.date}</td>`},
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
  const trs=vis.map(r=>`<tr>${cs.map(c=>c.fn(r)).join('')}</tr>`).join('');
  document.getElementById('ts').innerHTML=`<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
  document.querySelectorAll('#ts th[data-i]').forEach(th=>{
    th.addEventListener('click',()=>{
      const i=+th.dataset.i;if(sc===i)sd*=-1;else{sc=i;sd=1;}
      const col=cs[i],k=col.k;
      if(col.sortFn) rows.sort((a,b)=>(col.sortFn(a)-col.sortFn(b))*sd);
      else rows.sort((a,b)=>(typeof a[k]==='number'?a[k]-b[k]:String(a[k]).localeCompare(String(b[k])))*sd);
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
    adv:['Symbol','Index','Close','Strategy','Signal',...Object.keys((rows[0]?.extra)||{}).filter(k=>k!=='sig'),'RS','vs200DMA','AvgVol20','LastDate'],
    t1: ['Symbol','Index','Close','Signal','RSI',...Object.keys((rows[0]?.extra)||{}),'RS','vs200DMA','vs52WH','AvgVol20','LastDate'],
    t2: ['Symbol','Index','Close','Signal',...Object.keys((rows[0]?.extra)||{}),'RS','vs200DMA','vs52WH','AvgVol20','LastDate'],
    t3: ['Symbol','Index','Close','Signal',...Object.keys((rows[0]?.extra)||{}),'RS','vs200DMA','vs52WH','AvgVol20','LastDate'],
    ti: ['Symbol','Index','Close','Pattern/Signal',...Object.keys((rows[0]?.extra)||{}),'RS','vs200DMA','vs52WH','AvgVol20','LastDate'],
  };
  const cells={
    piv:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.lv,`${r.dist>=0?'+':''}${r.dist.toFixed(2)}%`,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.b6,r.b12,r.sh,r.sl2,r.sc2,r.sd,r.avol,r.date],
    smc:r=>[r.sym,r.idx||'Other',r.price,r.signal,r.zone_h,r.zone_l,r.smc.trend,r.smc.bos_bull?'Y':'N',r.smc.bos_bear?'Y':'N',r.smc.choch||'',r.smc.bull_ob?.h||'',r.smc.bull_ob?.l||'',r.smc.bear_ob?.h||'',r.smc.bear_ob?.l||'',r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    vol:r=>[r.sym,r.idx||'Other',r.price,r.vsig,r.poc,r.vah,r.val,r.vr,r.obv,r.ad,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    mi: r=>[r.sym,r.idx||'Other',r.price,`${r.mts}/8`,r.stg,r.stgl,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.s50,r.s150,r.s200,r.s30w,r.avol,r.date],
    adv:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.strat,r.extra?.sig||'',...Object.keys((rows[0]?.extra)||{}).filter(k=>k!=='sig').map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    t1: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,r.rsi,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,`${((r.price-r.w52h)/r.w52h*100).toFixed(1)}%`,r.avol,r.date],
    t2: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,`${((r.price-r.w52h)/r.w52h*100).toFixed(1)}%`,r.avol,r.date],
    t3: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,`${((r.price-r.w52h)/r.w52h*100).toFixed(1)}%`,r.avol,r.date],
    ti: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,`${((r.price-r.w52h)/r.w52h*100).toFixed(1)}%`,r.avol,r.date],
    nl: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    xp: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    patt: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.keys((rows[0]?.extra)||{}).map(k=>r.extra?.[k]??''),r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
    gap:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,r.extra.gap||'',r.rs,r.date],
    combo:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,r.rs,r.date],
    bp:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.values(r.extra||{}),r.rs,r.date],
    sr:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.values(r.extra||{}),r.rs,r.date],
    swing:r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,...Object.values(r.extra||{}),r.rs,r.date],
    br: r=>[r.sym,r.idx||'Other',r.price.toFixed(2),r.sig,r.rs,`${((r.price-r.dma200)/r.dma200*100).toFixed(1)}%`,r.avol,r.date],
  };
  const lines=[(hdrs[tab]||hdrs.piv).join(','),...vis.map(r=>(cells[tab]||cells.piv)(r).join(','))];
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([lines.join('\n')],{type:'text/csv'}));
  a.download=`nse_scan_${tab}_${new Date().toISOString().slice(0,10)}.csv`;a.click();
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',()=>{
  // Dismiss loading overlay
  const ld=document.getElementById('_ld');
  const bar=document.getElementById('_ld_bar');
  if(bar){bar.style.animation='none';bar.style.width='100%';bar.style.background='#00e5a0';}
  const ldMsg=document.getElementById('_ld_msg');
  if(ldMsg) ldMsg.textContent='Ready — '+S.length+' stocks loaded';
  if(ld) setTimeout(()=>{ld.style.transition='opacity .4s';ld.style.opacity='0';setTimeout(()=>{ld.style.display='none';},420);},500);
  try{
    const su=document.getElementById('st-u');
    if(su) su.textContent=S.length;
    const p=loadPrefs(); applyPrefs(p||DEFAULTS);
    initAutoScan();
    switchTab('home');
    _populateComboDropdowns();
  } catch(e){
    console.error('NSE Scanner init error:',e);
    const ts=document.getElementById('ts');
    if(ts) ts.innerHTML=`<div style="padding:24px;font-family:monospace;color:#ef4444">
      <b>⚠ Scanner init error</b><br><br>
      ${e.message}<br><br>
      <small>Open browser console (F12 → Console) for full details.</small>
    </div>`;
  }
});
</script>
<div class="hm-overlay" id="hm-overlay" onclick="closeHelpModal()"></div>
<div class="hm-box" id="hm-box">
  <div class="hm-hdr">
    <span class="hm-title" id="hm-title">Strategy Info</span>
    <button class="hm-close" onclick="closeHelpModal()">✕ Close</button>
  </div>
  <div class="hm-body" id="hm-body"></div>
</div>
<div class="toast" id="_toast"></div>
<!-- ④ Stock detail pop-up -->
<div id="sd-overlay" onclick="hideStockDetail()"></div>
<div id="sd-box">
  <div id="sd-hdr">
    <span style="font-size:13px;font-weight:600;color:var(--txt)">◈ Stock Detail</span>
    <div style="display:flex;gap:8px;align-items:center">
      <button id="ai-btn" onclick="aiAnalyzeStock(document.getElementById('sd-body').dataset.sym)"
        style="background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);color:#00d4ff;font-size:11px;padding:4px 10px;border-radius:5px;cursor:pointer">🤖 AI Analysis</button>
      <button onclick="hideStockDetail()" style="background:none;border:none;color:var(--mu);font-size:22px;cursor:pointer;line-height:1">×</button>
    </div>
  </div>
  <div id="ai-output" style="border-bottom:1px solid var(--brd);min-height:0;max-height:140px;overflow-y:auto"></div>
  <div id="sd-body"></div>
</div>
</body>
</html>
"""

class _NpEnc(json.JSONEncoder):
    """Serialize numpy scalars that json.dumps can't handle natively."""
    def default(self, o):
        if isinstance(o, np.bool_):    return bool(o)
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)

def build_html(stocks, data_dir):
    import json as _json
    gt = datetime.now().strftime("%d %b %Y  %H:%M")
    # Extract breadth data (stored on first stock by assign_breadth)
    breadth_data = stocks[0].get('_breadth', {}) if stocks else {}
    # Remove internal breadth keys from all stocks before serializing
    clean_stocks = []
    for s in stocks:
        cs = {k:v for k,v in s.items() if k not in ('_breadth','_breadth_ref')}
        clean_stocks.append(cs)
    return (HTML
            .replace("__JSON__", _json.dumps(clean_stocks, separators=(",",":"), cls=_NpEnc))
            .replace("__BREADTH_JSON__", _json.dumps(breadth_data, separators=(",",":"), cls=_NpEnc))
            .replace("__STOCK_COUNT__", str(len(stocks)))
            .replace("__GT__", gt).replace("__DD__", data_dir))

def main():
    ap = argparse.ArgumentParser(description="NSE Strategy Scanner")
    ap.add_argument("--data",  default=DATA_DIR)
    ap.add_argument("--n50",   default=N50_FILE); ap.add_argument("--n100", default=N100_FILE)
    ap.add_argument("--n200",  default=N200_FILE); ap.add_argument("--n500", default=N500_FILE)
    ap.add_argument("--n750",  default=N750_FILE); ap.add_argument("--out",  default=OUTPUT_HTML)
    ap.add_argument("--nifty", default=None,
                    help="Path to Nifty50 index CSV (enables RS vs Nifty strategy)")
    ap.add_argument("--nfno",  default=NFNO_FILE,
                    help="Path to Nifty F&O eligible symbols file (default: ../niftyfno.txt)")
    a = ap.parse_args()
    print(f"[*] Data: {a.data}")
    idx = {50:a.n50, 100:a.n100, 200:a.n200, 500:a.n500, 750:a.n750}
    ap.add_argument('--sector',   default=None, help='Path to sector map CSV (Symbol,Sector)')
    ap.add_argument('--delivery', default=None, help='Path to NSE bhav copy CSV with delivery volume (columns: Symbol, Deliverable_Vol, Total_Vol)')
    stocks = build_dataset(a.data, idx, nifty_path=a.nifty, fno_path=a.nfno, sector_path=getattr(a,'sector',None))
    if not stocks: print("[!] No stocks loaded"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f: f.write(html)
    print(f"[+] Output: {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
