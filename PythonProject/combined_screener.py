"""
Combined Institutional Screener
================================
Merges Price-Action + Momentum analysis on Daily AND Weekly timeframes.

Price Action  : Demand/Supply Zones · RSI · SMA 20/50/200 · Volume · Breakout
Momentum      : ADX/DI · Supertrend · ROC 1M/3M/6M · RS vs Index · OBV · MACD(5/35/5)

Features
--------
• Daily / Weekly toggle with independent filters
• Clickable signal pills  (filter table for each view)
• MACD Buy / Sell filter buttons
• Min CMP input  (default ₹50 — hides penny stocks)
• Export visible stock names to clipboard
• Horizontal scroll on table AND full webpage
• Sortable columns

Outputs
-------
  nifty750_signals.html   — Nifty 750 stocks
  other_stocks_signals.html — All other stocks
"""

import os
import glob
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(BASE_DIR, "data_cache")
NIFTY750_FILE  = os.path.join(BASE_DIR, "nifty750.txt")
OUT_NIFTY750   = os.path.join(BASE_DIR, "nifty750_signals.html")
OUT_OTHERS     = os.path.join(BASE_DIR, "other_stocks_signals.html")

LOOKBACK       = 400    # daily candles loaded per stock
MIN_ROWS       = 60     # minimum daily candles required

# Price-action parameters
ZONE_TOLERANCE = 0.015
SWING_WINDOW   = 10
VOL_MA_PERIOD  = 20
RSI_PERIOD     = 14
SMA_SHORT      = 20
SMA_MED        = 50
SMA_LONG       = 200

# Momentum parameters
ADX_P          = 14
ST_P           = 10
ST_MULT        = 3.0
MFAST, MSLOW, MSIG = 5, 35, 5

_PFXS = ("BSE_", "NSE_", "BSE-", "NSE-")
_SFXS = ('-A','-B','-T','-X','-XT','-Z','-ZP','-M','-MT','-MS','-P','-B1','-IF','-E')

# ── Signal meta (PA signals) ──────────────────────────────────
PA_META = {
    "BREAKOUT":    ("🚀", "#ff6f00", "#0f0800"),
    "STRONG BUY":  ("🟢", "#00c853", "#000f06"),
    "BUY":         ("🟩", "#43a047", "#030d03"),
    "WATCH":       ("👁",  "#1976d2", "#00080f"),
    "NEUTRAL":     ("⬜", "#9e9e9e", "#111111"),
    "CAUTION":     ("🟡", "#f9a825", "#0d0a00"),
    "NEAR SUPPLY": ("🔶", "#e65100", "#0e0500"),
    "SELL":        ("🔴", "#e53935", "#0f0000"),
}
PA_ORDER = ["BREAKOUT","STRONG BUY","BUY","WATCH","NEUTRAL","CAUTION","NEAR SUPPLY","SELL"]

# ── Momentum signal meta ──────────────────────────────────────
MOM_META = {
    "STRONG MOMENTUM": ("🚀", "#ff8c00"),
    "MOMENTUM":        ("📈", "#00c853"),
    "WATCH":           ("👁",  "#1e88e5"),
    "NEUTRAL":         ("⬜", "#9e9e9e"),
    "WEAK":            ("📉", "#e53935"),
}


# ─────────────────────────────────────────────────────────────
# NORMALISATION & DATA LOADING
# ─────────────────────────────────────────────────────────────

def norm(raw: str) -> str:
    s = raw.strip().upper()
    for p in _PFXS:
        if s.startswith(p): s = s[len(p):]; break
    for x in _SFXS:
        if s.endswith(x):   s = s[:-len(x)]; break
    return s

def disp_name(filename: str) -> str:
    return norm(Path(filename).stem)

def load_nifty750():
    if not os.path.exists(NIFTY750_FILE): return set()
    out = set()
    with open(NIFTY750_FILE) as f:
        for line in f:
            r = line.strip()
            if r and not r.startswith('#'): out.add(norm(r))
    return out

def load_csv(path):
    try:
        df = pd.read_csv(path, parse_dates=['Datetime'])
        df.columns = [c.strip() for c in df.columns]
        df = df.sort_values('Datetime').reset_index(drop=True)
        df = df.dropna(subset=['Open','High','Low','Close'])
        for c in ['Open','High','Low','Close','Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=['Close'])
        df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception:
        return None

def find_index_csv():
    best = (0, None)
    for f in glob.glob(os.path.join(DATA_DIR, '*.csv')):
        s = Path(f).stem.upper()
        for kw in ('NIFTY50','NIFTY_50','NSE_NIFTY50','^NSEI','NIFTY','SENSEX'):
            if kw in s and len(kw) > best[0]:
                best = (len(kw), f)
    return best[1]

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.set_index('Datetime')
          .resample('W-FRI')
          .agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
          .dropna(subset=['Close'])
          .reset_index()
    )


# ─────────────────────────────────────────────────────────────
# PRICE-ACTION INDICATORS
# ─────────────────────────────────────────────────────────────

def rsi_calc(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_calc(df, period=14):
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low']  - df['Close'].shift()).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).mean()

def find_swing_lows(df, window):
    lows = []
    for i in range(window, len(df) - window):
        if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
            lows.append(i)
    return lows

def find_swing_highs(df, window):
    highs = []
    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
            highs.append(i)
    return highs

def build_demand_zones(df, swing_lows, vol_ma):
    zones = []
    for idx in swing_lows:
        row     = df.iloc[idx]
        avg_vol = vol_ma.iloc[idx]
        future  = df.iloc[idx+1:idx+11]
        bounce  = (future['Close'].max() - row['Low']) / row['Low'] if len(future) > 0 else 0
        if bounce >= 0.02:
            zl  = row['Low'] * (1 - ZONE_TOLERANCE)
            zh  = row['Low'] * (1 + ZONE_TOLERANCE * 2)
            rev = ((df['Low'] >= zl) & (df['Low'] <= zh)).sum()
            vs  = min(row['Volume'] / avg_vol, 3.0) if avg_vol > 0 else 1.0
            st  = max(0, min(100, bounce*50 + vs*15 - rev*5))
            zones.append({'zone_low': round(zl,2), 'zone_high': round(zh,2),
                          'pivot': round(row['Low'],2), 'date': row['Datetime'],
                          'vol_ok': row['Volume'] > avg_vol*1.3 if avg_vol>0 else False,
                          'bounce': round(bounce*100,2), 'revisits': rev,
                          'strength': round(st,1), 'idx': idx})
    zones.sort(key=lambda z: (z['idx'], z['strength']), reverse=True)
    return zones[:5]

def build_supply_zones(df, swing_highs, vol_ma):
    zones = []
    for idx in swing_highs:
        row     = df.iloc[idx]
        avg_vol = vol_ma.iloc[idx]
        future  = df.iloc[idx+1:idx+11]
        drop    = (row['High'] - future['Close'].min()) / row['High'] if len(future) > 0 else 0
        if drop >= 0.02:
            zl  = row['High'] * (1 - ZONE_TOLERANCE * 2)
            zh  = row['High'] * (1 + ZONE_TOLERANCE)
            rev = ((df['High'] >= zl) & (df['High'] <= zh)).sum()
            vs  = min(row['Volume'] / avg_vol, 3.0) if avg_vol > 0 else 1.0
            st  = max(0, min(100, drop*50 + vs*15 - rev*5))
            zones.append({'zone_low': round(zl,2), 'zone_high': round(zh,2),
                          'pivot': round(row['High'],2), 'date': row['Datetime'],
                          'vol_ok': row['Volume'] > avg_vol*1.3 if avg_vol>0 else False,
                          'drop': round(drop*100,2), 'revisits': rev,
                          'strength': round(st,1), 'idx': idx})
    zones.sort(key=lambda z: (z['idx'], z['strength']), reverse=True)
    return zones[:5]


# ─────────────────────────────────────────────────────────────
# PRICE-ACTION ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_pa(df: pd.DataFrame, is_weekly: bool = False) -> dict | None:
    """
    Price-action analysis on a trimmed, ready OHLCV dataframe.
    is_weekly: True when called on a weekly-resampled df.
    """
    sw_win        = 5  if is_weekly else SWING_WINDOW
    sma_long_p    = 100 if is_weekly else SMA_LONG
    ret_short_p   = 4  if is_weekly else 5
    ret_long_p    = 13 if is_weekly else 20
    bk_lookback_p = 20  # same bar count for both (20 candles / 20 weeks)

    n = len(df)
    if n < (25 if is_weekly else MIN_ROWS):
        return None

    df = df.copy()
    df['vol_ma']  = df['Volume'].rolling(VOL_MA_PERIOD).mean()
    df['sma20']   = df['Close'].rolling(SMA_SHORT).mean()
    df['sma50']   = df['Close'].rolling(SMA_MED).mean()
    df['sma_long']= df['Close'].rolling(sma_long_p).mean()
    df['rsi_v']   = rsi_calc(df['Close'], RSI_PERIOD)

    last     = df.iloc[-1]
    cmp      = float(last['Close'])
    vol_now  = float(last['Volume'])
    vol_ma_v = float(last['vol_ma'])  if not np.isnan(last['vol_ma'])  else 0
    vol_ratio= round(vol_now / vol_ma_v, 2) if vol_ma_v > 0 else 0.0

    rsi_val  = round(float(last['rsi_v']),  1) if not np.isnan(last['rsi_v'])  else None
    sma20_v  = float(last['sma20'])   if not np.isnan(last['sma20'])   else None
    sma50_v  = float(last['sma50'])   if not np.isnan(last['sma50'])   else None
    smal_v   = float(last['sma_long'])if not np.isnan(last['sma_long'])else None

    above_20  = cmp > sma20_v  if sma20_v  else False
    above_50  = cmp > sma50_v  if sma50_v  else False
    above_200 = cmp > smal_v   if smal_v   else False
    sma_align = sum([above_20, above_50, above_200])

    swing_lows  = find_swing_lows(df, sw_win)
    swing_highs = find_swing_highs(df, sw_win)
    demand_zones = build_demand_zones(df, swing_lows,  df['vol_ma'])
    supply_zones = build_supply_zones(df, swing_highs, df['vol_ma'])

    best_demand = None; in_demand = False; dist_demand = None
    for z in demand_zones:
        dist = (cmp - z['zone_high']) / cmp * 100
        if z['zone_low'] <= cmp <= z['zone_high'] * 1.03:
            in_demand = True
            if best_demand is None or z['strength'] > best_demand['strength']:
                best_demand = z; dist_demand = round(dist, 2)
        elif best_demand is None:
            best_demand = z; dist_demand = round(dist, 2)

    best_supply = None; in_supply = False; dist_supply = None
    for z in supply_zones:
        dist = (z['zone_low'] - cmp) / cmp * 100
        if z['zone_low'] <= cmp <= z['zone_high'] * 1.02:
            in_supply = True
            if best_supply is None or z['strength'] > best_supply['strength']:
                best_supply = z; dist_supply = round(dist, 2)
        elif best_supply is None:
            best_supply = z; dist_supply = round(dist, 2)

    recent_n  = df.tail(5)
    price_up5 = (float(recent_n['Close'].iloc[-1]) - float(recent_n['Close'].iloc[0])) / float(recent_n['Close'].iloc[0]) * 100
    vol_surge = vol_ratio >= 1.5

    def is_accum(row):
        rng = row['High'] - row['Low']
        body= abs(row['Close'] - row['Open'])
        return rng > 0 and body >= 0.4*rng and row['Close'] >= row['Low'] + 0.7*rng

    def is_distrib(row):
        rng = row['High'] - row['Low']
        body= abs(row['Close'] - row['Open'])
        return rng > 0 and body >= 0.4*rng and row['Close'] <= row['Low'] + 0.3*rng

    accum_bars   = sum(is_accum(recent_n.iloc[i])   for i in range(len(recent_n)))
    distrib_bars = sum(is_distrib(recent_n.iloc[i]) for i in range(len(recent_n)))

    ret_short = round(price_up5, 2)
    ret_long  = round((cmp / float(df['Close'].iloc[-ret_long_p-1]) - 1) * 100, 2) if n > ret_long_p else None

    tail52  = df.tail(252 if not is_weekly else 52)
    hi52    = round(float(tail52['High'].max()),  2)
    lo52    = round(float(tail52['Low'].min()),   2)
    pos52   = round((cmp - lo52) / (hi52 - lo52) * 100, 1) if hi52 != lo52 else 50.0

    bk_high = float(df['High'].iloc[-bk_lookback_p-1:-1].max()) if n >= bk_lookback_p+1 else cmp
    breakout = cmp > bk_high and vol_surge

    # ── Score ──────────────────────────────────────────────
    score = 0; reasons = []

    if in_demand:
        score += 30; reasons.append("✦ Price in Demand Zone")
    elif best_demand and dist_demand is not None and 0 <= dist_demand <= 5:
        score += 15; reasons.append("Near Demand Zone")

    if in_supply:
        score -= 25; reasons.append("⚠ Price in Supply Zone")

    score += sma_align * 10
    if sma_align == 3:   reasons.append("All SMAs Bullish")
    elif sma_align == 2: reasons.append("SMA 20/50 Bullish")

    if vol_surge and accum_bars >= 2:
        score += 20; reasons.append("✦ Volume + Accumulation")
    elif vol_surge:
        score += 10; reasons.append("Volume Surge")

    if accum_bars  >= 3: score += 15; reasons.append("Strong Accumulation")
    if distrib_bars >= 3: score -= 20; reasons.append("Distribution Pattern")

    if breakout:
        lbl = "20W Breakout" if is_weekly else "20D Breakout"
        score += 25; reasons.append(f"✦ {lbl}")

    if rsi_val:
        if   55 <= rsi_val <= 70: score += 10; reasons.append(f"RSI Momentum ({rsi_val})")
        elif rsi_val > 75:        score -= 10; reasons.append(f"RSI Overbought ({rsi_val})")
        elif rsi_val < 35:        score -=  5; reasons.append(f"RSI Weak ({rsi_val})")

    if ret_short and ret_short > 3:   score += 8
    if ret_long  and ret_long  > 8:   score += 7

    score = max(-100, min(100, score))

    if   score >= 60: signal = "STRONG BUY"
    elif score >= 30: signal = "BUY"
    elif score >= 10: signal = "WATCH"
    elif score <= -30: signal = "SELL"
    elif score <= -10: signal = "CAUTION"
    else:              signal = "NEUTRAL"

    if breakout and score >= 30:   signal = "BREAKOUT"
    if in_supply and score < 20:   signal = "NEAR SUPPLY"

    last_date = str(df.iloc[-1]['Datetime'].date()) if 'Datetime' in df.columns else ''

    return dict(
        cmp=round(cmp, 2), signal=signal, score=score,
        sma_align=sma_align,
        rsi=rsi_val, vol_ratio=vol_ratio,
        ret_short=ret_short, ret_long=ret_long,
        pos_52w=pos52, hi_52w=hi52, lo_52w=lo52,
        demand_zone=(f"{best_demand['zone_low']}–{best_demand['zone_high']}" if best_demand else "—"),
        demand_str=(best_demand['strength'] if best_demand else 0),
        dist_demand=dist_demand, in_demand=in_demand,
        supply_zone=(f"{best_supply['zone_low']}–{best_supply['zone_high']}" if best_supply else "—"),
        supply_str=(best_supply['strength'] if best_supply else 0),
        dist_supply=dist_supply, in_supply=in_supply,
        accum_bars=accum_bars, distrib_bars=distrib_bars,
        sma20=round(sma20_v, 2) if sma20_v else None,
        sma50=round(sma50_v, 2) if sma50_v else None,
        sma_long=round(smal_v, 2) if smal_v else None,
        breakout=breakout, reasons=reasons, last_date=last_date,
    )


# ─────────────────────────────────────────────────────────────
# MOMENTUM INDICATORS
# ─────────────────────────────────────────────────────────────

def _wilder_atr(h, l, c, period):
    n  = len(h)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    return pd.Series(tr).ewm(alpha=1.0/period, adjust=False).mean().values

def calc_adx(df):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(h)
    tr=np.zeros(n); pdm=np.zeros(n); ndm=np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
        up, dn = h[i]-h[i-1], l[i-1]-l[i]
        pdm[i] = up if up>dn and up>0 else 0.0
        ndm[i] = dn if dn>up and dn>0 else 0.0
    a = 1.0/ADX_P
    atr = pd.Series(tr).ewm(alpha=a, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr>0, 100*pd.Series(pdm).ewm(alpha=a,adjust=False).mean().values/atr, 0.)
        ndi = np.where(atr>0, 100*pd.Series(ndm).ewm(alpha=a,adjust=False).mean().values/atr, 0.)
        dx  = np.where((pdi+ndi)>0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0.)
    adx = pd.Series(dx).ewm(alpha=a, adjust=False).mean().values
    return float(adx[-1]), float(pdi[-1]), float(ndi[-1])

def calc_supertrend(df):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n   = len(h)
    atr = _wilder_atr(h, l, c, ST_P)
    hl2 = (h+l)/2.0
    ubr = hl2+ST_MULT*atr; lbr = hl2-ST_MULT*atr
    ub=ubr.copy(); lb=lbr.copy()
    direction=np.ones(n,dtype=int); st=np.zeros(n)
    for i in range(1, n):
        ub[i] = ubr[i] if ubr[i]<ub[i-1] or c[i-1]>ub[i-1] else ub[i-1]
        lb[i] = lbr[i] if lbr[i]>lb[i-1] or c[i-1]<lb[i-1] else lb[i-1]
        if   direction[i-1]==-1 and c[i]>ub[i]: direction[i]=1
        elif direction[i-1]== 1 and c[i]<lb[i]: direction[i]=-1
        else:                                    direction[i]=direction[i-1]
        st[i] = lb[i] if direction[i]==1 else ub[i]
    return float(st[-1]), int(direction[-1])

def calc_obv_trend(df):
    sign = np.sign(df['Close'].diff().fillna(0).values)
    obv  = (sign * df['Volume'].values).cumsum()
    s    = pd.Series(obv)
    return float((s.ewm(span=10,adjust=False).mean() - s.ewm(span=30,adjust=False).mean()).iloc[-1])

def calc_macd(close: pd.Series):
    ml = close.ewm(span=MFAST,adjust=False).mean() - close.ewm(span=MSLOW,adjust=False).mean()
    ms = ml.ewm(span=MSIG,adjust=False).mean()
    mh = ml - ms
    now, prev = float(ml.iloc[-1]), float(ml.iloc[-2])
    sn,  sp   = float(ms.iloc[-1]), float(ms.iloc[-2])
    if   prev<=sp and now>sn and now<0: t="BUY"
    elif sp<=prev and sn>now and now>0: t="SELL"
    else:                               t="NONE"
    return round(now,4), round(sn,4), round(float(mh.iloc[-1]),4), t


# ─────────────────────────────────────────────────────────────
# MOMENTUM ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_mom(df: pd.DataFrame, idx_close, p1: int, p3: int, p6: int) -> dict | None:
    if df is None or len(df) < max(40, p6+4):
        return None
    n     = len(df)
    close = df['Close']

    adx_v, pdi_v, ndi_v = calc_adx(df)
    st_v,  dir_v         = calc_supertrend(df)
    obv_mom              = calc_obv_trend(df)
    ml, ms_v, mh, macd_t = calc_macd(close)
    above_st             = float(close.iloc[-1]) > st_v

    def roc(p):
        return round((float(close.iloc[-1])/float(close.iloc[-p-1])-1)*100, 2) if n>p+1 else None

    r1=roc(p1); r3=roc(p3); r6=roc(p6)

    rs3=None
    if idx_close is not None and len(idx_close)==n and n>p3+1:
        try:
            ix_now  = float(idx_close.iloc[-1])
            ix_then = float(idx_close.iloc[-p3-1])
            if ix_then>0 and ix_now>0:
                rs3 = round((float(close.iloc[-1])/float(close.iloc[-p3-1])-1)*100 - (ix_now/ix_then-1)*100, 2)
        except Exception:
            pass

    # Momentum score
    sc=0; rsn=[]
    if   adx_v>=40: sc+=20; rsn.append(f"Strong Trend ADX {adx_v:.0f}")
    elif adx_v>=25: sc+=12; rsn.append(f"Trending ADX {adx_v:.0f}")
    elif adx_v>=15: sc+= 5
    if pdi_v>ndi_v: sc+=10; rsn.append(f"+DI>{ndi_v:.0f} Bullish")
    else:           sc-= 5; rsn.append(f"-DI>{pdi_v:.0f} Bearish")
    if above_st: sc+=10; rsn.append("Above Supertrend")
    else:        sc-= 8; rsn.append("Below Supertrend")
    if r1 is not None:
        if r1>5: sc+=8;  rsn.append(f"1M {r1:+.1f}%")
        elif r1>0: sc+=3
        elif r1<-5: sc-=5
    if r3 is not None:
        if r3>15: sc+=15; rsn.append(f"3M {r3:+.1f}%")
        elif r3>5: sc+=8; rsn.append(f"3M {r3:+.1f}%")
        elif r3>0: sc+=3
        elif r3<-10: sc-=8
    if r6 is not None:
        if r6>25: sc+=10; rsn.append(f"6M {r6:+.1f}%")
        elif r6>10: sc+=5
        elif r6<-15: sc-=5
    if rs3 is not None:
        if rs3>10: sc+=15; rsn.append(f"RS +{rs3:.1f}% vs Index")
        elif rs3>5: sc+=10; rsn.append(f"RS +{rs3:.1f}% vs Index")
        elif rs3>0: sc+=5
        elif rs3<-5: sc-=5
    if obv_mom>0: sc+=10; rsn.append("OBV Rising")
    else:         sc-= 3
    if macd_t=="BUY":  sc+=8;  rsn.append("MACD Crossover Buy")
    elif macd_t=="SELL": sc-=5

    sc=max(-50,min(100,sc))
    if   sc>=70: mom_sig="STRONG MOMENTUM"
    elif sc>=45: mom_sig="MOMENTUM"
    elif sc>=20: mom_sig="WATCH"
    elif sc>= 0: mom_sig="NEUTRAL"
    else:        mom_sig="WEAK"

    return dict(
        adx=round(adx_v,1), pdi=round(pdi_v,1), ndi=round(ndi_v,1),
        st_val=round(st_v,2), st_dir=dir_v, above_st=above_st,
        roc1=r1, roc3=r3, roc6=r6, rs3=rs3,
        obv_rising=(obv_mom>0),
        macd_line=ml, macd_sig=ms_v, macd_hist=mh, macd_type=macd_t,
        mom_score=sc, mom_signal=mom_sig, mom_reasons=rsn,
    )

def _empty_mom():
    return dict(adx=None, pdi=None, ndi=None, st_val=None, st_dir=1, above_st=None,
                roc1=None, roc3=None, roc6=None, rs3=None,
                obv_rising=None, macd_line=None, macd_sig=None, macd_hist=None,
                macd_type='NONE', mom_score=0, mom_signal="NEUTRAL", mom_reasons=[])


# ─────────────────────────────────────────────────────────────
# COMBINED ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyse_daily(df: pd.DataFrame, idx_df) -> dict | None:
    if df is None or len(df) < MIN_ROWS:
        return None
    df_d = df.tail(LOOKBACK).reset_index(drop=True).copy()

    ic = None
    if idx_df is not None:
        try:
            idx_s = idx_df.set_index('Datetime')['Close']
            ic    = idx_s.reindex(df_d['Datetime']).ffill().reset_index(drop=True)
        except Exception:
            pass

    pa  = analyse_pa(df_d, is_weekly=False)
    if pa is None:
        return None
    mom = analyse_mom(df_d, ic, p1=20, p3=63, p6=126) or _empty_mom()
    return {**pa, **mom}


def analyse_weekly(df: pd.DataFrame, idx_df) -> dict | None:
    if df is None or len(df) < MIN_ROWS:
        return None
    wdf = to_weekly(df.tail(LOOKBACK))

    ic = None
    if idx_df is not None:
        try:
            idx_s = idx_df.set_index('Datetime')['Close']
            idx_w = idx_s.resample('W-FRI').last().dropna()
            ic    = idx_w.reindex(wdf['Datetime']).ffill().reset_index(drop=True)
        except Exception:
            pass

    pa  = analyse_pa(wdf, is_weekly=True)
    if pa is None:
        return None
    mom = analyse_mom(wdf, ic, p1=4, p3=13, p6=26) or _empty_mom()
    return {**pa, **mom}


# ─────────────────────────────────────────────────────────────
# HTML CELL HELPERS
# ─────────────────────────────────────────────────────────────

def _pct(v, decimals=1):
    if v is None: return '<span style="color:#444">—</span>'
    c = "#00c853" if v>=0 else "#f85149"
    a = "▲" if v>=0 else "▼"
    return f'<span style="color:{c};font-weight:600">{a} {abs(v):.{decimals}f}%</span>'

def _score_bar(sc, lo=-100, hi=100):
    pct = max(0.0, min(100.0, (sc - lo) / (hi - lo) * 100.0))
    hue = max(0, min(120, int(pct * 1.2)))
    return (
        f'<div style="display:flex;align-items:center;gap:5px;min-width:100px">'
        f'<div style="flex:1;height:5px;background:#21262d;border-radius:3px">'
        f'<div style="width:{pct:.0f}%;height:100%;background:hsl({hue},75%,42%);border-radius:3px"></div></div>'
        f'<span style="font-size:10px;color:#8b949e;font-family:monospace;flex-shrink:0">{sc:+d}</span></div>'
    )

def _52w_bar(pos):
    c = "#00c853" if pos>=70 else "#d29922" if pos>=40 else "#f85149"
    return (
        f'<div style="display:flex;align-items:center;gap:5px">'
        f'<div style="width:70px;height:5px;background:#21262d;border-radius:3px;flex-shrink:0">'
        f'<div style="width:{pos:.0f}%;height:100%;background:{c};border-radius:3px"></div></div>'
        f'<span style="font-size:10px;color:#8b949e">{pos:.0f}%</span></div>'
    )

def _tbody_row(r: dict, is_weekly: bool) -> str:
    s       = r['signal']
    em, col, bg = PA_META.get(s, ("","#888","#111"))
    si      = PA_ORDER.index(s) if s in PA_ORDER else 99

    # Mom signal badge
    ms  = r.get('mom_signal','NEUTRAL')
    mc, mscol = MOM_META.get(ms, ("","#9e9e9e"))
    mom_badge = f'<span style="color:{mscol};font-size:10px;font-weight:600">{mc} {ms}</span>'

    # SMA labels
    cmp = r['cmp']
    sma_html = ""
    for lbl, val in [("S20", r.get('sma20')),
                     ("S50", r.get('sma50')),
                     ("SL",  r.get('sma_long'))]:
        display_lbl = "S200" if not is_weekly and lbl=="SL" else ("S100" if is_weekly and lbl=="SL" else lbl)
        if val:
            c = "#00c853" if cmp>val else "#f85149"
            sma_html += f'<span style="color:{c};font-size:11px;margin-right:4px">{display_lbl}</span>'

    # ADX / DI cell
    adx = r.get('adx'); pdi = r.get('pdi'); ndi = r.get('ndi')
    if adx is not None:
        ac,al = (("#00c853","Strong") if adx>=40 else ("#69db7c","Trending") if adx>=25
                  else ("#d29922","Weak") if adx>=15 else ("#6e7681","Ranging"))
        pc = "#00c853" if (pdi or 0)>(ndi or 0) else "#f85149"
        nc = "#f85149" if (pdi or 0)>(ndi or 0) else "#00c853"
        adx_html = (f'<b style="color:{ac}">{adx:.1f}</b>'
                    f'<span style="font-size:10px;color:{ac}"> {al}</span><br>'
                    f'<span style="font-size:11px">'
                    f'<span style="color:{pc}">+DI {pdi:.1f}</span>&nbsp;'
                    f'<span style="color:{nc}">-DI {ndi:.1f}</span></span>')
        adx_dv = adx
    else:
        adx_html = '<span style="color:#444">—</span>'
        adx_dv   = 0

    # Supertrend cell
    stv = r.get('st_val'); ab = r.get('above_st')
    if stv is not None and ab is not None:
        stc = "#00c853" if ab else "#f85149"
        stl = "🟢 ABOVE" if ab else "🔴 BELOW"
        pst = round((cmp-stv)/stv*100, 1) if stv > 0 else 0
        st_html = (f'<span style="color:{stc};font-weight:700">{stl}</span><br>'
                   f'<span style="font-size:11px;color:#6e7681">₹{stv:,.2f} ({pst:+.1f}%)</span>')
        st_dv = 1 if ab else -1
    else:
        st_html = '<span style="color:#444">—</span>'
        st_dv   = 0

    # ROC + RS cell
    r1=r.get('roc1'); r3=r.get('roc3'); r6=r.get('roc6'); rs3=r.get('rs3')
    roc_html = (
        f'<span style="font-size:10px;color:#8b949e">1M:</span>{_pct(r1)}&nbsp;'
        f'<span style="font-size:10px;color:#8b949e">3M:</span>{_pct(r3)}<br>'
        f'<span style="font-size:10px;color:#8b949e">6M:</span>{_pct(r6)}<br>'
        + (f'{_pct(rs3)}<span style="font-size:10px;color:#555"> vs Nifty</span>' if rs3 is not None else '<span style="color:#444">RS:—</span>')
    )
    roc_dv = r3 if r3 is not None else 0

    # OBV + MACD cell
    obv = r.get('obv_rising')
    if obv is not None:
        oc  = "#00c853" if obv else "#f85149"
        ol  = "▲ Rising" if obv else "▼ Falling"
        obv_html = f'<span style="color:{oc};font-size:11px;font-weight:600">{ol}</span>'
    else:
        obv_html = '<span style="color:#444">—</span>'

    mt=r.get('macd_type','NONE'); ml=r.get('macd_line',0)
    msv=r.get('macd_sig',0); mh=r.get('macd_hist',0)
    if mt != 'NONE' and ml is not None:
        mhc = "#00c853" if (mh or 0)>=0 else "#f85149"
        mha = "▲" if (mh or 0)>=0 else "▼"
        mb  = ('<span class="sig-badge" style="background:#00875a;padding:2px 6px">📈 BUY</span>' if mt=="BUY"
               else '<span class="sig-badge" style="background:#b71c1c;padding:2px 6px">📉 SELL</span>')
        macd_html = (f'<span style="font-size:10px;color:#8b949e">L:{ml:+.3f} S:{msv:+.3f}</span><br>'
                     f'<span style="color:{mhc};font-size:10px">{mha}{abs(mh or 0):.3f}</span>&nbsp;{mb}')
    elif ml is not None:
        macd_html = f'<span style="font-size:10px;color:#8b949e">L:{ml:+.3f}</span><br><span style="color:#444;font-size:10px">—</span>'
    else:
        macd_html = '<span style="color:#444">—</span>'
    macd_dv = 0 if mt=="NONE" else (1 if mt=="BUY" else -1)

    # Zone cells
    dem_cls = ' class="zone-active"' if r.get('in_demand') else ""
    sup_cls = ' class="zone-warn"'   if r.get('in_supply') else ""

    # Reasons: combine PA + momentum reasons
    rsn_pa  = r.get('reasons', [])
    rsn_mom = r.get('mom_reasons', [])
    rsn_all = rsn_pa + ([f"<span style='color:#888;font-size:10px'>{x}</span>" for x in rsn_mom] if rsn_mom else [])
    rsn_html = "<br>".join(rsn_all) or "—"

    dn = disp_name(r['name'])

    return f"""<tr style="background:{bg}" data-signal="{s}" data-macd-type="{mt}" data-cmp="{cmp}">
  <td class="sticky-col" data-val="{r['name']}"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A{dn}" target="_blank">{dn}</a></td>
  <td data-val="{cmp}" style="font-weight:700;white-space:nowrap">₹{cmp:,.2f}</td>
  <td data-val="{si}" style="white-space:nowrap;min-width:150px"><span class="sig-badge" style="background:{col}">{em} {s}</span><br>{mom_badge}</td>
  <td data-val="{r['score']}" style="min-width:120px">{_score_bar(r['score'],-100,100)}</td>
  <td data-val="{r.get('rsi') or 0}" style="white-space:nowrap">
    <span style="font-size:12px">{r.get('rsi') or '—'}</span><span style="color:#8b949e;font-size:10px"> RSI</span><br>
    <span style="font-size:12px">{r.get('vol_ratio','—')}×</span><span style="color:#8b949e;font-size:10px"> Vol</span>
  </td>
  <td data-val="{r.get('ret_short') or 0}">{_pct(r.get('ret_short'), decimals=2)}</td>
  <td data-val="{r.get('ret_long') or 0}">{_pct(r.get('ret_long'), decimals=2)}</td>
  <td data-val="{r.get('pos_52w',0)}" style="min-width:110px">{_52w_bar(r.get('pos_52w',0))}</td>
  <td{dem_cls} data-val="{r.get('demand_str',0)}" style="min-width:120px;font-size:12px">
    {r.get('demand_zone','—')}<br>
    <span style="font-size:10px;color:#8b949e">Str:{r.get('demand_str',0):.0f} Dist:{r.get('dist_demand','—')}%</span>
  </td>
  <td{sup_cls} data-val="{r.get('supply_str',0)}" style="min-width:120px;font-size:12px">
    {r.get('supply_zone','—')}<br>
    <span style="font-size:10px;color:#8b949e">Str:{r.get('supply_str',0):.0f} Dist:{r.get('dist_supply','—')}%</span>
  </td>
  <td data-val="{r.get('sma_align',0)}">{sma_html}<br><span style="font-size:10px;color:#8b949e">Acc:{r.get('accum_bars',0)} Dis:{r.get('distrib_bars',0)}</span></td>
  <td data-val="{adx_dv}" style="min-width:120px">{adx_html}</td>
  <td data-val="{st_dv}" style="min-width:120px">{st_html}</td>
  <td data-val="{roc_dv}" style="min-width:150px;white-space:nowrap">{roc_html}</td>
  <td data-val="{macd_dv}" style="min-width:150px">{obv_html}<br>{macd_html}</td>
  <td class="reasons" data-val="{len(rsn_pa)}">{rsn_html}</td>
  <td data-val="{r.get('last_date','')}" style="font-size:11px;color:#6e7681;white-space:nowrap">{r.get('last_date','')}</td>
</tr>"""


# ─────────────────────────────────────────────────────────────
# HTML BUILDER
# ─────────────────────────────────────────────────────────────

def build_html(rows: list, title: str, subtitle: str, generated_at: str) -> str:

    def skey(r, tf):
        m = r.get(tf)
        return (PA_ORDER.index(m['signal']) if m and m['signal'] in PA_ORDER else 99,
                -(m['score'] if m else 0))

    dr = sorted([r for r in rows if r.get('daily')],  key=lambda r: skey(r,'daily'))
    wr = sorted([r for r in rows if r.get('weekly')], key=lambda r: skey(r,'weekly'))

    def sig_counts(lst, tf):
        c = {}
        for r in lst:
            m = r.get(tf)
            if m: c[m['signal']] = c.get(m['signal'],0) + 1
        return c

    dc = sig_counts(dr, 'daily')
    wc = sig_counts(wr, 'weekly')

    def pill_set(cnt, view):
        total = sum(cnt.values())
        h = (f'<div class="pill active-pill" id="pill-all-{view}" '
             f'style="border-color:#58a6ff;color:#58a6ff" '
             f'onclick="setFilter(\'all\',this)">All <b>{total}</b></div>')
        for sig in PA_ORDER:
            if sig in cnt:
                em2, col2, _ = PA_META.get(sig, ("","#888",""))
                h += (f'<div class="pill" style="border-color:{col2};color:{col2}" '
                      f'onclick="setFilter(\'{sig}\',this)">{em2} {sig} <b>{cnt[sig]}</b></div>')
        return h

    def make_tbody(lst, tf, is_w):
        return "".join(
            _tbody_row({**r[tf], 'name': r['name'], 'ticker': r['ticker']}, is_weekly=is_w)
            for r in lst if r.get(tf)
        )

    td_d  = make_tbody(dr, 'daily', False)
    td_w  = make_tbody(wr, 'weekly', True)
    ps_d  = pill_set(dc, 'daily')
    ps_w  = pill_set(wc, 'weekly')

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --surface2:#21262d;
    --border:#30363d; --text:#c9d1d9; --muted:#8b949e;
    --accent:#58a6ff; --green:#3fb950; --red:#f85149;
    --orange:#d29922; --radius:8px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  html{{overflow-x:auto}}
  body{{
    font-family:'IBM Plex Sans',sans-serif;
    background:var(--bg);color:var(--text);
    min-height:100vh;min-width:1700px;padding:24px;
  }}

  /* ── Header ── */
  .header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:16px}}
  .header h1{{
    font-size:clamp(18px,3vw,28px);font-weight:700;
    background:linear-gradient(135deg,#58a6ff,#a371f7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }}
  .header .meta{{color:var(--muted);font-size:13px;margin-top:4px}}
  .badge{{background:var(--surface2);border:1px solid var(--border);border-radius:20px;padding:6px 14px;font-size:12px;font-family:'IBM Plex Mono',monospace;white-space:nowrap}}

  /* ── Legend ── */
  .legend{{
    display:flex;flex-wrap:wrap;gap:12px;align-items:center;
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--radius);padding:10px 16px;margin-bottom:16px;
    font-size:11px;color:var(--muted);
  }}
  .legend b{{color:var(--text);font-size:11px}}

  /* ── View toggle ── */
  .view-toggle{{display:flex;gap:0;margin-bottom:16px;border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;width:fit-content}}
  .view-btn{{background:var(--surface);border:none;color:var(--muted);padding:9px 26px;font-size:13px;cursor:pointer;font-family:'IBM Plex Mono',monospace;font-weight:600;transition:all .2s}}
  .view-btn.active{{background:var(--accent);color:#000}}
  .view-btn:hover:not(.active){{background:var(--surface2);color:var(--accent)}}

  /* ── Summary pills ── */
  .summary{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}}
  .pill{{border:1.5px solid;border-radius:20px;padding:4px 12px;font-size:11px;font-weight:600;font-family:'IBM Plex Mono',monospace;cursor:pointer;transition:all .2s;user-select:none}}
  .pill:hover{{opacity:.8;transform:translateY(-1px)}}
  .active-pill{{box-shadow:0 0 0 2px currentColor;background:rgba(255,255,255,.05)}}

  /* ── Controls ── */
  .controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}}
  .search-box{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:7px 13px;color:var(--text);font-size:13px;width:200px;font-family:'IBM Plex Sans',sans-serif}}
  .search-box:focus{{outline:none;border-color:var(--accent)}}
  .cmp-wrap{{display:flex;align-items:center;gap:6px;background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:5px 12px}}
  .cmp-wrap label{{font-size:12px;color:var(--muted);font-family:'IBM Plex Mono',monospace;white-space:nowrap}}
  .cmp-input{{background:transparent;border:none;color:var(--text);font-size:13px;font-family:'IBM Plex Mono',monospace;width:70px;outline:none}}
  .export-btn{{background:var(--surface2);border:1px solid var(--green);color:var(--green);border-radius:var(--radius);padding:7px 13px;font-size:12px;cursor:pointer;font-family:'IBM Plex Mono',monospace;transition:all .2s}}
  .export-btn:hover{{background:rgba(63,185,80,.1)}}
  .btn-sep{{width:1px;height:24px;background:var(--border);margin:0 4px}}
  .macd-buy-btn{{background:var(--surface);border:1.5px solid #00c853;color:#00c853;border-radius:var(--radius);padding:6px 12px;font-size:12px;cursor:pointer;font-family:'IBM Plex Mono',monospace;transition:all .2s}}
  .macd-buy-btn:hover,.macd-buy-btn.active{{background:rgba(0,200,83,.12);box-shadow:0 0 0 1.5px #00c853}}
  .macd-sell-btn{{background:var(--surface);border:1.5px solid #f85149;color:#f85149;border-radius:var(--radius);padding:6px 12px;font-size:12px;cursor:pointer;font-family:'IBM Plex Mono',monospace;transition:all .2s}}
  .macd-sell-btn:hover,.macd-sell-btn.active{{background:rgba(248,81,73,.12);box-shadow:0 0 0 1.5px #f85149}}
  .clear-btn{{background:var(--surface);border:1px solid var(--border);color:var(--muted);border-radius:var(--radius);padding:6px 12px;font-size:12px;cursor:pointer;font-family:'IBM Plex Mono',monospace;transition:all .2s}}
  .clear-btn:hover{{border-color:var(--accent);color:var(--accent)}}
  .ctrl-label{{font-size:10px;color:var(--muted);font-family:'IBM Plex Mono',monospace;text-transform:uppercase;letter-spacing:.5px}}

  /* ── Table ── */
  .table-wrap{{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border);background:var(--surface)}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  thead th{{
    background:var(--surface2);padding:10px 13px;text-align:left;
    font-size:10px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);
    white-space:nowrap;cursor:pointer;user-select:none;
    position:sticky;top:0;z-index:2;border-bottom:1px solid var(--border);transition:color .2s;
  }}
  thead th:first-child{{position:sticky;left:0;top:0;z-index:3}}
  thead th:hover{{color:var(--accent)}}
  thead th.sort-asc::after {{content:" ▲";color:var(--accent)}}
  thead th.sort-desc::after{{content:" ▼";color:var(--accent)}}
  tbody tr{{border-bottom:1px solid var(--border);transition:filter .15s}}
  tbody tr:hover{{filter:brightness(1.18)}}
  tbody tr:last-child{{border-bottom:none}}
  tbody td{{padding:9px 13px;vertical-align:middle;color:var(--text)}}
  .sticky-col{{position:sticky;left:0;z-index:1;font-family:'IBM Plex Mono',monospace;font-weight:600;font-size:12px;background:inherit;border-right:1px solid var(--border);min-width:85px}}
  .sticky-col a{{color:var(--accent);text-decoration:none}}
  .sticky-col a:hover{{text-decoration:underline}}
  .sig-badge{{color:#fff;border-radius:4px;padding:3px 8px;font-size:11px;font-weight:700;font-family:'IBM Plex Mono',monospace;white-space:nowrap}}
  .zone-active{{background:rgba(0,200,83,.07)!important}}
  .zone-warn  {{background:rgba(245,124,0,.07)!important}}
  .reasons{{font-size:11px;color:var(--muted);max-width:210px;line-height:1.5}}

  /* ── Footer ── */
  .footer{{margin-top:28px;text-align:center;font-size:11px;color:var(--muted);line-height:1.8}}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{title}</h1>
    <div class="meta">{subtitle}</div>
    <div class="meta" style="margin-top:4px">Generated: {generated_at} &nbsp;|&nbsp; {len(rows)} stocks analysed</div>
  </div>
  <div class="badge">📊 Combined Institutional Screener</div>
</div>

<div class="legend">
  <b>Price Action →</b>
  <span>🚀 Breakout · 🟢 Strong Buy · 🟩 Buy · 👁 Watch · ⬜ Neutral · 🟡 Caution · 🔶 Near Supply · 🔴 Sell</span>
  <span style="width:1px;height:16px;background:var(--border);flex-shrink:0"></span>
  <b>Momentum →</b>
  <span>ADX≥40 Strong · ADX≥25 Trending · Supertrend 🟢/🔴 · ROC 1M/3M/6M · RS vs Nifty · OBV · MACD(5/35/5)</span>
</div>

<div class="view-toggle">
  <button class="view-btn active" id="view-btn-daily"  onclick="switchView('daily')">📅 Daily</button>
  <button class="view-btn"        id="view-btn-weekly" onclick="switchView('weekly')">📆 Weekly</button>
</div>

<div id="summary-daily"  class="summary">{ps_d}</div>
<div id="summary-weekly" class="summary" style="display:none">{ps_w}</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="🔍 Search stock…" oninput="filterRows()">
  <div class="cmp-wrap">
    <label>Min CMP ₹</label>
    <input class="cmp-input" type="number" id="cmpMin" value="50" min="0" step="5" oninput="filterRows()">
  </div>
  <button class="export-btn" onclick="exportNames()">📋 Export Names</button>
  <div class="btn-sep"></div>
  <span class="ctrl-label">MACD</span>
  <button class="macd-buy-btn  macd-btn" onclick="setMacdFilter('BUY',this)">📈 Buy</button>
  <button class="macd-sell-btn macd-btn" onclick="setMacdFilter('SELL',this)">📉 Sell</button>
  <div class="btn-sep"></div>
  <button class="clear-btn" onclick="clearFilters()">✕ Clear All</button>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead><tr>
  <th onclick="sortTable(0)">Stock</th>
  <th onclick="sortTable(1)">CMP</th>
  <th onclick="sortTable(2)">Signal</th>
  <th onclick="sortTable(3)">PA Score</th>
  <th onclick="sortTable(4)">RSI / Vol</th>
  <th id="th-ret-short" onclick="sortTable(5)">5D Ret</th>
  <th id="th-ret-long"  onclick="sortTable(6)">20D Ret</th>
  <th onclick="sortTable(7)">52W Pos</th>
  <th onclick="sortTable(8)">Demand Zone ↑</th>
  <th onclick="sortTable(9)">Supply Zone ↓</th>
  <th id="th-sma" onclick="sortTable(10)">SMA / Pattern</th>
  <th onclick="sortTable(11)">ADX / DI</th>
  <th onclick="sortTable(12)">Supertrend</th>
  <th onclick="sortTable(13)">ROC + RS</th>
  <th onclick="sortTable(14)">OBV + MACD</th>
  <th onclick="sortTable(15)">Reasons</th>
  <th onclick="sortTable(16)">Last Date</th>
</tr></thead>
<tbody id="tbody-daily">{td_d}</tbody>
<tbody id="tbody-weekly" style="display:none">{td_w}</tbody>
</table>
</div>

<div class="footer">
  <b>Disclaimer:</b> For educational &amp; research purposes only. Not SEBI-registered advice.<br>
  Demand/Supply zones and all indicators are algorithmic approximations. Always do your own due diligence.
</div>

<script>
let currentView = 'daily';
let activeFilter = 'all';
let activeMacd   = 'all';
let sortCol = -1, sortDir = 1;

function getCurTbody() {{
  return document.getElementById('tbody-' + currentView);
}}

function switchView(v) {{
  currentView = v;
  ['daily','weekly'].forEach(x => {{
    document.getElementById('tbody-'    + x).style.display = x===v ? '' : 'none';
    document.getElementById('summary-' + x).style.display = x===v ? '' : 'none';
    document.getElementById('view-btn-'+ x).classList.toggle('active', x===v);
  }});
  // Update column headers for the timeframe
  const w = v === 'weekly';
  document.getElementById('th-ret-short').textContent = w ? '4W Ret'  : '5D Ret';
  document.getElementById('th-ret-long').textContent  = w ? '13W Ret' : '20D Ret';
  document.getElementById('th-sma').textContent       = w ? 'SMA(W) / Pattern' : 'SMA / Pattern';
  // Reset rows, filters, pills
  getCurTbody().querySelectorAll('tr').forEach(r => r.style.display = '');
  activeFilter = 'all'; activeMacd = 'all';
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active-pill'));
  const ap = document.getElementById('pill-all-' + v);
  if (ap) ap.classList.add('active-pill');
  document.querySelectorAll('.macd-btn').forEach(b => b.classList.remove('active'));
  filterRows();
}}

function filterRows() {{
  const q      = (document.getElementById('search').value || '').toLowerCase();
  const minCmp = parseFloat(document.getElementById('cmpMin').value) || 0;
  getCurTbody().querySelectorAll('tr').forEach(r => {{
    const name   = r.cells[0].textContent.toLowerCase();
    const signal = (r.dataset.signal || '').toLowerCase();
    const mt     = (r.dataset.macdType || '').toUpperCase();
    const cmp    = parseFloat(r.dataset.cmp) || 0;
    const mQ = !q || name.includes(q);
    const mF = activeFilter === 'all' || signal === activeFilter.toLowerCase();
    const mM = activeMacd   === 'all' || mt     === activeMacd;
    const mC = cmp >= minCmp;
    r.style.display = (mQ && mF && mM && mC) ? '' : 'none';
  }});
}}

function setFilter(f, el) {{
  activeFilter = f;
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active-pill'));
  el.classList.add('active-pill');
  activeMacd = 'all';
  document.querySelectorAll('.macd-btn').forEach(b => b.classList.remove('active'));
  filterRows();
}}

function setMacdFilter(type, btn) {{
  if (activeMacd === type) {{
    activeMacd = 'all'; btn.classList.remove('active');
  }} else {{
    activeMacd = type;
    document.querySelectorAll('.macd-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  }}
  activeFilter = 'all';
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active-pill'));
  const ap = document.getElementById('pill-all-' + currentView);
  if (ap) ap.classList.add('active-pill');
  filterRows();
}}

function clearFilters() {{
  activeFilter = 'all'; activeMacd = 'all';
  document.getElementById('search').value = '';
  document.getElementById('cmpMin').value = 50;
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active-pill'));
  const ap = document.getElementById('pill-all-' + currentView);
  if (ap) ap.classList.add('active-pill');
  document.querySelectorAll('.macd-btn').forEach(b => b.classList.remove('active'));
  getCurTbody().querySelectorAll('tr').forEach(r => r.style.display = '');
}}

function exportNames() {{
  const names = [];
  getCurTbody().querySelectorAll('tr').forEach(r => {{
    if (r.style.display !== 'none') {{
      const t = r.cells[0].textContent.trim();
      if (t) names.push(t);
    }}
  }});
  const csv = names.join(',');
  navigator.clipboard.writeText(csv)
    .then(() => {{
      const btn = document.querySelector('.export-btn');
      const orig = btn.textContent;
      btn.textContent = '✅ Copied ' + names.length + ' stocks!';
      setTimeout(() => btn.textContent = orig, 2500);
    }})
    .catch(() => prompt('Copy stock names:', csv));
}}

function sortTable(col) {{
  const tbody = getCurTbody();
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const ths   = document.querySelectorAll('thead th');
  if (sortCol === col) sortDir *= -1;
  else {{ sortCol = col; sortDir = 1; }}
  ths.forEach((th,i) => {{
    th.classList.remove('sort-asc','sort-desc');
    if (i === col) th.classList.add(sortDir===1 ? 'sort-asc' : 'sort-desc');
  }});
  rows.sort((a,b) => {{
    const av = a.cells[col]?.dataset.val ?? a.cells[col]?.textContent ?? '';
    const bv = b.cells[col]?.dataset.val ?? b.cells[col]?.textContent ?? '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an-bn)*sortDir;
    return av.localeCompare(bv)*sortDir;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Combined Institutional Screener")
    print("  Price Action + Momentum | Daily & Weekly")
    print("=" * 60)

    nifty750 = load_nifty750()
    print(f"  Nifty 750 list : {len(nifty750)} tickers")

    idx_path = find_index_csv()
    idx_df   = load_csv(idx_path) if idx_path else None
    if idx_path:
        print(f"  Index file     : {os.path.basename(idx_path)}")
    else:
        print("  ⚠  No index CSV found — RS vs Index will be blank")

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if idx_path:
        abs_idx   = os.path.abspath(idx_path)
        csv_files = [f for f in csv_files if os.path.abspath(f) != abs_idx]

    if not csv_files:
        print(f"\n  ERROR: No CSV files found in {DATA_DIR}")
        return
    print(f"  Stock CSVs     : {len(csv_files)} files\n")

    nifty750_rows, other_rows = [], []
    errors, skipped = [], []
    generated_at = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    for i, path in enumerate(csv_files, 1):
        filename = os.path.basename(path)
        raw_stem = Path(filename).stem
        ticker   = norm(raw_stem)
        display  = raw_stem.upper()

        if i % 50 == 0 or i == len(csv_files):
            print(f"  Processing {i}/{len(csv_files)} — {ticker:<22}", end="\r")

        df = load_csv(path)
        try:
            daily  = analyse_daily(df, idx_df)
            weekly = analyse_weekly(df, idx_df)
        except Exception as e:
            errors.append(f"{ticker}: {e}")
            continue

        if daily is None and weekly is None:
            skipped.append(ticker)
            continue

        row = {'name': display, 'ticker': ticker, 'daily': daily, 'weekly': weekly}
        if ticker in nifty750:
            nifty750_rows.append(row)
        else:
            other_rows.append(row)

    print(f"\n\n  ✓ Analysed  : {len(nifty750_rows)+len(other_rows)} stocks")
    print(f"    Nifty 750 : {len(nifty750_rows)}")
    print(f"    Other     : {len(other_rows)}")
    print(f"    Skipped   : {len(skipped)}")
    if errors:
        print(f"    Errors    : {len(errors)}")
        for e in errors[:5]: print(f"      {e}")

    subtitle = (
        "Price Action (Demand/Supply · RSI · SMA · Volume · Breakout) + "
        "Momentum (ADX · Supertrend · ROC · RS · OBV · MACD)  |  Daily & Weekly"
    )

    for rows, path, title in [
        (nifty750_rows, OUT_NIFTY750, "Nifty 750 — Institutional Screener"),
        (other_rows,    OUT_OTHERS,   "Other Stocks — Institutional Screener"),
    ]:
        html = build_html(rows if rows else [], title, subtitle, generated_at)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"  ✅  {path}  ({len(rows)} stocks)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
