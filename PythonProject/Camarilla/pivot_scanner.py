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
    return dict(value=r2(st[-1]), direction="up" if d[-1]==1 else "down",
                flipped=flipped, atr=r2(atr[-1]))

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
    return dict(rsi=round(cur,1), bull_div=bool(bull_div), bear_div=bool(bear_div),
                overbought=bool(cur>=70), oversold=bool(cur<=30))

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
    return dict(macd=round(cur_m,3), signal=round(cur_s,3), hist=round(cur_h,3),
                trend="bull" if cur_m>cur_s else "bear",
                rising=bool(ml[-1]>ml[-min(5,n)]),
                bull_div=bool(bull_div), bear_div=bool(bear_div),
                flip_bull=flip_bull, flip_bear=flip_bear)

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
    # 🏅 India Pro strategies — pre-computed for Daily / Weekly / Monthly
    _df_w = weekly_agg(df)
    _df_m = monthly_agg(df)
    ti = dict(
        d = _ti_for(df),
        w = _ti_for(_df_w),
        m = _ti_for(_df_m),
    )
    # Advanced strategies
    adv = dict(
        darvas   = compute_darvas(df),
        vcp      = compute_vcp(df),
        wyckoff  = compute_wyckoff(df),
        turtle   = compute_turtle(df),
        ichi     = compute_ichimoku(df),
        td       = compute_td(df),
        st       = compute_supertrend(df),
        elder    = compute_elder(df),
    )
    return dict(sym=sym, idx=idx_map.get(sym,0),
                price=r2(float(df.iloc[-1]["Close"])),
                date=str(df.iloc[-1]["Date"].date()),
                d=ds,w=ws,m=ms,q=qs,y=ys,ytd=yts,
                mhist=mh,whist=wh,qhist=qh,
                smc=smc,vol=vol,mi=mi,t1=t1,t2=t2,t3=t3,ti=ti,adv=adv,**st,rs=0)

def assign_rs(stocks):
    rets=sorted(s["ret12m"] for s in stocks); n=len(rets)
    for s in stocks:
        s["rs"] = round(sum(1 for x in rets if x<s["ret12m"])/n*99) if n else 0
        if s.get("mi"):
            s["mi"]["mt"][7] = int(s["rs"]>=70)
            s["mi"]["mts"]   = sum(s["mi"]["mt"])

def build_dataset(data_dir, index_files, nifty_path=None):
    idx_map = load_index_map(index_files)
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
.tab-btn.t-adv.active{color:#ff8c42;border-bottom-color:#ff8c42}
.tab-btn.t-t1.active{color:#c47aff;border-bottom-color:#c47aff}
.tab-btn.t-t2.active{color:#00e5a0;border-bottom-color:#00e5a0}
.tab-btn.t-t3.active{color:#f5c518;border-bottom-color:#f5c518}
.tab-btn.t-ti.active{color:#ff9500;border-bottom-color:#ff9500}

/* GLOBAL FILTER BAR */
.global-bar{display:flex;align-items:center;gap:10px;padding:6px 22px;
  background:rgba(0,229,160,.04);border-bottom:1px solid rgba(0,229,160,.12);
  flex-wrap:wrap}
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
</style>
</head>
<body>

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
  <button class="tab-btn t-help" data-tab="help" onclick="switchTab('help')">❓ Help</button>
</div>
<!-- ═══ GLOBAL FILTERS ════════════════════════════════════════════════════ -->
<div class="global-bar" id="global-bar">
  <span class="gb-label">🌐 Global</span>
  <div class="gb-sep"></div>
  <label class="gb-lbl">Index</label>
  <select id="global-idx" class="gb-sel">
    <option value="0" selected>All Stocks</option>
    <option value="50">Nifty 50</option>
    <option value="100">Nifty 100</option>
    <option value="200">Nifty 200</option>
    <option value="500">Nifty 500</option>
    <option value="750">Nifty 750</option>
  </select>
  <div class="gb-sep"></div>
  <span class="gb-hint">Applies to all scanners ↓</span>
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
    </select><button class="info-btn" onclick="showHelp('piv_type',document.getElementById('sp-type').value)" title="Show strategy info">ℹ</button></div>
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
      <optgroup label="── RSI Divergence ──">
        <option value="rsi_bull_div">RSI Bullish Divergence — hidden accumulation</option>
        <option value="rsi_bear_div">RSI Bearish Divergence — hidden distribution</option>
        <option value="rsi_oversold">RSI Oversold ≤ 30 — extreme selling exhaustion</option>
        <option value="rsi_overbought">RSI Overbought ≥ 70 — extreme buying exhaustion</option>
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
  ['piv','smc','vol','mi','adv','t1','t2','t3','ti'].forEach(t=>{
    const el=document.getElementById('ctrl-'+t);
    if(el) el.style.display=t===tab?'flex':'none';
  });
  if(tab==='home'){
    document.getElementById('ts').innerHTML=renderHome();
    setFbar('🏠 <b>Quick Launch</b> · Click any card to instantly run the scan · Strategies ranked by professional success rate','');
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
  const idxF=parseInt(document.getElementById('global-idx').value);
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
  const idxF=parseInt(document.getElementById('global-idx').value);
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
  const idxF=parseInt(document.getElementById('global-idx').value);
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
  const idxF=parseInt(document.getElementById('global-idx').value);
  const prMin=parseFloat(document.getElementById('mi-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('mi-pmax').value)||Infinity;

  const MT_LABELS=['Price > 150 & 200 SMA','150 SMA > 200 SMA','200 SMA rising','50 SMA > 150 & 200 SMA','Price > 50 SMA','≥ 25% above 52W low','Within 25% of 52W high','RS ≥ 70'];
  setFbar(
    '<b>Minervini Trend Template</b> · 8 conditions: '+ MT_LABELS.map((l,i)=>`<span style="color:var(--mu)">${i+1}.${l}</span>`).join(' · ')+'<br>'+
    '<b>Weinstein Stage Analysis</b> · Stage 2 (Advancing): price above rising 30-week (150-day) SMA');
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
  const idxF =parseInt(document.getElementById('global-idx').value);
  const prMin=parseFloat(document.getElementById('adv-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('adv-pmax').value)||Infinity;
  updateAdvInfo();
  rows=[];
  for(const s of S){
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
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
    if(strat==='ichi_kbo')    { const i=a.ichi;     matched=i&&i.kbo;  extra={sig:'Kumo BO',span_a:i?.span_a,span_b:i?.span_b,tenkan:i?.tenkan,kijun:i?.kijun}; }
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
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
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
    if(strat==='nr7_inside') { matched=!!nr.is_nr7&&!!nr.is_inside;  sig='🎯 NR7+IB'; extra={comp:nr.compression+'%',rng:nr.rng,atr14:nr.atr14,bias:nr.bias}; }

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
  const idxF =parseInt(document.getElementById('global-idx').value);
  const rsMin=parseInt(document.getElementById('t2-rs').value);
  const prMin=parseFloat(document.getElementById('t2-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('t2-pmax').value)||Infinity;
  updateT2Info();
  rows=[];
  for(const s of S){
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
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
    if(strat==='macd_flip_bear') { matched=!!md.flip_bear;  sig='🔴 MACD Flip −'; extra={macd:md.macd,hist:md.hist}; }
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
  const idxF =parseInt(document.getElementById('global-idx').value);
  const rsMin=parseInt(document.getElementById('t3-rs').value);
  const prMin=parseFloat(document.getElementById('t3-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('t3-pmax').value)||Infinity;
  updateT3Info();
  rows=[];
  for(const s of S){
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
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
  // 🏅 India Pro (Chartink / StockEdge / Trendlyne)
  {tab:'ti', strat:'candle_bull_eng',    emoji:'📗', badge:'ti', label:'Bullish Engulfing',      desc:'Current bull body engulfs prior bear — institutional buyers overwhelm sellers', stars:'★★★★', section:3},
  {tab:'ti', strat:'candle_morning_star',emoji:'🌅', badge:'ti', label:'Morning Star',            desc:"3-bar reversal: bear→star→bull — O'Neil's most reliable candlestick bottom",   stars:'★★★★', section:3},
  {tab:'ti', strat:'ema_fan_bull',       emoji:'🔥', badge:'ti', label:'EMA Fan Bullish',         desc:'Price > EMA9 > EMA21 > EMA50 — Chartink momentum leader alignment',            stars:'★★★★', section:3},
  {tab:'ti', strat:'ma_all_bull',        emoji:'🏆', badge:'ti', label:'Full MA Stack',           desc:'MA20>MA50>MA100>MA200 — StockEdge maximum bullish alignment',                  stars:'★★★★', section:3},
  {tab:'ti', strat:'consol_10',          emoji:'🎯', badge:'ti', label:'52W Tight Coil',          desc:'Within 5% of 52W high, 10-day range <5% — pre-breakout coil',                 stars:'★★★★', section:3},
  {tab:'ti', strat:'vol_acc3',           emoji:'📦', badge:'ti', label:'3-Day Accumulation',      desc:'3 up-days with rising volume — StockEdge institutional accumulation',          stars:'★★★',  section:3},
];

const SECTIONS=['⭐ HIGHEST EDGE — TREND FOLLOWING','🔥 MOMENTUM & TIMING SIGNALS','🔄 MEAN REVERSION & REVERSAL','🏅 INDIA PRO — CHARTINK · STOCKEDGE · TRENDLYNE'];
const BADGES={t1:'Tier-1',t2:'Tier-2',t3:'Tier-3',mi:'Multi-Ind',adv:'Advanced',piv:'Pivot',ti:'India Pro'};

function renderHome(){
  let html=`<div class="home-wrap">`;
  for(let si=0;si<3;si++){
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
  // Set dropdown value before switching tab
  const maps={adv:'adv-strat',t1:'t1-strat',t2:'t2-strat',t3:'t3-strat'};
  if(strat && maps[tab]){
    const el=document.getElementById(maps[tab]);
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
function openAI(service,b64prompt){
  const prompt=decodeURIComponent(escape(atob(b64prompt)));
  navigator.clipboard.writeText(prompt).then(()=>{
    showToast('✓ Prompt copied! Paste in '+(service==='chatgpt'?'ChatGPT':service.charAt(0).toUpperCase()+service.slice(1))+' (Ctrl+V / ⌘V)');
  }).catch(()=>showToast('Opening '+service+'…'));
  window.open(AI_URLS[service],'_blank');
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
  t1:{gc_fresh_golden:0,gc_fresh_death:0,gc_bull_zone:0,gc_near_golden:0,rsi_bull_div:1,rsi_bear_div:1,rsi_oversold:1,rsi_overbought:1,bb_squeeze_bull:2,bb_squeeze_bear:2,bb_squeeze_any:2,w52_breakout:3,w52_close_above:3,w52_near:3,w52_consol_near:3,nr7_bull:4,nr7_bear:4,nr4:4,inside_bar:4,nr7_inside:4},
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
  },
};
// Map tab → section index in H array inside renderHelp
const TAB_SEC={piv_type:0,smc:1,vol:2,mi:3,adv:4,t1:5,t2:6,t3:7,ti:8};

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
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeHelpModal();});


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
  const tf   =document.getElementById('ti-tf').value||'d';
  const idxF =parseInt(document.getElementById('global-idx').value);
  const rsMin=parseInt(document.getElementById('ti-rs').value);
  const prMin=parseFloat(document.getElementById('ti-pmin').value)||0;
  const prMax=parseFloat(document.getElementById('ti-pmax').value)||Infinity;
  updateTIInfo();
  const TF_LABEL={d:'Daily',w:'Weekly',m:'Monthly'};
  const tfLabel=TF_LABEL[tf]||'Daily';
  rows=[];
  for(const s of S){
    if(idxF>0&&(s.idx===0||s.idx>idxF))continue;
    if(s.price<prMin||s.price>prMax)continue;
    if(s.rs<rsMin)continue;
    if(!s.ti)continue;
    // Pick timeframe sub-dict; fall back to daily if unavailable
    const tiData=s.ti[tf]||s.ti['d']||{};
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

    if(!matched)continue;
    // Append timeframe label to signal so user knows which TF fired
    const tfTag=tf!=='d'?` [${tfLabel}]`:'';
    rows.push({sym:s.sym,idx:s.idx,price:s.price,date:s.date,avol:s.avol,
      above200:s.above200,rs:s.rs,dma200:s.dma200,w52h:s.w52h,w52l:s.w52l,
      sig:sig+tfTag, extra, strat, _tab:'ti'});
  }
  sc=2;sd=1;rows.sort((a,b)=>a.sym.localeCompare(b.sym));render();
}

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
      {k:'above200',h:'vs 200DMA',fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
      {k:'above200', h:'vs 200DMA', fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
      {k:'above200', h:'vs 200DMA', fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
      {k:'above200', h:'vs 200DMA', fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
      {k:'above200',h:'vs 200DMA',fn:r=>dmaCell(r.price,r.dma200,r.above200)},
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
  initAutoScan();
  switchTab('home');
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
    gt = datetime.now().strftime("%d %b %Y  %H:%M")
    return (HTML.replace("__JSON__", json.dumps(stocks, separators=(",",":"), cls=_NpEnc))
                .replace("__GT__", gt).replace("__DD__", data_dir))

def main():
    ap = argparse.ArgumentParser(description="NSE Strategy Scanner")
    ap.add_argument("--data",  default=DATA_DIR)
    ap.add_argument("--n50",   default=N50_FILE); ap.add_argument("--n100", default=N100_FILE)
    ap.add_argument("--n200",  default=N200_FILE); ap.add_argument("--n500", default=N500_FILE)
    ap.add_argument("--n750",  default=N750_FILE); ap.add_argument("--out",  default=OUTPUT_HTML)
    ap.add_argument("--nifty", default=None,
                    help="Path to Nifty50 index CSV (enables RS vs Nifty strategy)")
    a = ap.parse_args()
    print(f"[*] Data: {a.data}")
    idx = {50:a.n50, 100:a.n100, 200:a.n200, 500:a.n500, 750:a.n750}
    stocks = build_dataset(a.data, idx, nifty_path=a.nifty)
    if not stocks: print("[!] No stocks loaded"); return
    html = build_html(stocks, a.data)
    with open(a.out, "w", encoding="utf-8") as f: f.write(html)
    print(f"[+] Output: {a.out}  ({len(html)//1024} KB)")

if __name__ == "__main__":
    main()
