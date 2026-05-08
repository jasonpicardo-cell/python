"""
Momentum Screener — Institutional Indicators
==============================================
Indicators computed on both Daily and Weekly timeframes:
  1. ADX (14) / +DI / -DI    — Trend strength and direction
  2. Supertrend (10, 3)       — Dynamic trend direction & dynamic S/R
  3. ROC 1M / 3M / 6M         — Price rate-of-change momentum
  4. RS vs Index              — Outperformance vs Nifty (3-month)
  5. OBV Trend                — On-Balance Volume momentum
  6. MACD (5/35/5)            — Crossover signals

Output files (same folder as this script):
  nifty750_momentum.html   — Nifty 750 stocks
  other_momentum.html      — All other stocks

Each HTML has:
  • Daily / Weekly toggle tabs
  • Clickable signal-count pills to filter the table
  • MACD Buy / MACD Sell filter buttons
  • Export visible stock names to clipboard
  • Full horizontal scroll (page + table)
  • Sortable columns (click any header)
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
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "bse_data_cache")
NIFTY750_FILE = os.path.join(BASE_DIR, "nifty750.txt")
OUT_NIFTY750  = os.path.join(BASE_DIR, "nifty750_momentum.html")
OUT_OTHERS    = os.path.join(BASE_DIR, "other_momentum.html")

LOOKBACK      = 400    # daily candles to load per stock
MIN_ROWS      = 100    # minimum daily candles to attempt analysis

# Indicator parameters
ADX_P         = 14
ST_P          = 10
ST_MULT       = 3.0
MFAST         = 5
MSLOW         = 35
MSIG          = 5

_PFXS = ("BSE_", "NSE_", "BSE-", "NSE-")
_SFXS = ('-A','-B','-T','-X','-XT','-Z','-ZP','-M','-MT','-MS','-P','-B1','-IF','-E')

SIGNAL_META = {
    "STRONG MOMENTUM": ("🚀", "#ff8c00", "#0f0900"),
    "MOMENTUM":        ("📈", "#00c853", "#000f06"),
    "WATCH":           ("👁",  "#1e88e5", "#00080f"),
    "NEUTRAL":         ("⬜", "#9e9e9e", "#111111"),
    "WEAK":            ("📉", "#e53935", "#0f0000"),
}
SIGNAL_ORDER = ["STRONG MOMENTUM", "MOMENTUM", "WATCH", "NEUTRAL", "WEAK"]


# ─────────────────────────────────────────────────────────────
# TICKER NORMALISATION
# ─────────────────────────────────────────────────────────────

def norm(raw: str) -> str:
    s = raw.strip().upper()
    for p in _PFXS:
        if s.startswith(p):
            s = s[len(p):]
            break
    for x in _SFXS:
        if s.endswith(x):
            s = s[:-len(x)]
            break
    return s

def disp_name(filename: str) -> str:
    """Clean display ticker from filename/stem."""
    return norm(Path(filename).stem)


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_nifty750():
    if not os.path.exists(NIFTY750_FILE):
        return set()
    out = set()
    with open(NIFTY750_FILE) as f:
        for line in f:
            r = line.strip()
            if r and not r.startswith('#'):
                out.add(norm(r))
    return out


def load_csv(path):
    try:
        df = pd.read_csv(path, parse_dates=['Datetime'])
        df.columns = [c.strip() for c in df.columns]
        df = df.sort_values('Datetime').reset_index(drop=True)
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close'])
        df['Volume'] = df['Volume'].fillna(0)
        return df
    except Exception:
        return None


def find_index_csv():
    """Auto-detect a Nifty50 / index CSV in DATA_DIR."""
    best = (0, None)
    for f in glob.glob(os.path.join(DATA_DIR, '*.csv')):
        s = Path(f).stem.upper()
        for kw in ('NIFTY50', 'NIFTY_50', 'NSE_NIFTY50', '^NSEI', 'NIFTY', 'SENSEX'):
            if kw in s and len(kw) > best[0]:
                best = (len(kw), f)
    return best[1]


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV → weekly (week ending Friday)."""
    return (
        df.set_index('Datetime')
          .resample('W-FRI')
          .agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                'Close': 'last', 'Volume': 'sum'})
          .dropna(subset=['Close'])
          .reset_index()
    )


# ─────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS
# ─────────────────────────────────────────────────────────────

def _wilder_atr(h, l, c, period):
    n = len(h)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    return pd.Series(tr).ewm(alpha=1.0 / period, adjust=False).mean().values


def calc_adx(df):
    """Returns (adx, +DI, -DI) scalars for the last bar."""
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(h)
    tr  = np.zeros(n)
    pdm = np.zeros(n)
    ndm = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        up, dn = h[i] - h[i-1], l[i-1] - l[i]
        pdm[i] = up if up > dn and up > 0 else 0.0
        ndm[i] = dn if dn > up and dn > 0 else 0.0
    a = 1.0 / ADX_P
    atr = pd.Series(tr).ewm(alpha=a, adjust=False).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr > 0, 100 * pd.Series(pdm).ewm(alpha=a, adjust=False).mean().values / atr, 0.0)
        ndi = np.where(atr > 0, 100 * pd.Series(ndm).ewm(alpha=a, adjust=False).mean().values / atr, 0.0)
        dx  = np.where((pdi + ndi) > 0, 100 * np.abs(pdi - ndi) / (pdi + ndi), 0.0)
    adx = pd.Series(dx).ewm(alpha=a, adjust=False).mean().values
    return float(adx[-1]), float(pdi[-1]), float(ndi[-1])


def calc_supertrend(df):
    """Returns (supertrend_level, direction) for the last bar. direction: 1=bullish, -1=bearish."""
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n   = len(h)
    atr = _wilder_atr(h, l, c, ST_P)
    hl2 = (h + l) / 2.0
    ubr = hl2 + ST_MULT * atr
    lbr = hl2 - ST_MULT * atr
    ub  = ubr.copy()
    lb  = lbr.copy()
    direction = np.ones(n, dtype=int)
    st = np.zeros(n)
    for i in range(1, n):
        ub[i] = ubr[i] if ubr[i] < ub[i-1] or c[i-1] > ub[i-1] else ub[i-1]
        lb[i] = lbr[i] if lbr[i] > lb[i-1] or c[i-1] < lb[i-1] else lb[i-1]
        if   direction[i-1] == -1 and c[i] > ub[i]: direction[i] = 1
        elif direction[i-1] ==  1 and c[i] < lb[i]: direction[i] = -1
        else:                                         direction[i] = direction[i-1]
        st[i] = lb[i] if direction[i] == 1 else ub[i]
    return float(st[-1]), int(direction[-1])


def calc_obv_trend(df):
    """OBV momentum: fast EMA minus slow EMA of OBV. Positive = rising."""
    sign = np.sign(df['Close'].diff().fillna(0).values)
    obv  = (sign * df['Volume'].values).cumsum()
    s    = pd.Series(obv)
    return float((s.ewm(span=10, adjust=False).mean() - s.ewm(span=30, adjust=False).mean()).iloc[-1])


def calc_macd(close: pd.Series):
    """MACD (MFAST/MSLOW/MSIG). Returns (line, signal, hist, type) for last bar."""
    ml = close.ewm(span=MFAST, adjust=False).mean() - close.ewm(span=MSLOW, adjust=False).mean()
    ms = ml.ewm(span=MSIG, adjust=False).mean()
    mh = ml - ms
    ml_now, ml_prev = float(ml.iloc[-1]), float(ml.iloc[-2])
    ms_now, ms_prev = float(ms.iloc[-1]), float(ms.iloc[-2])
    if   ml_prev <= ms_prev and ml_now > ms_now and ml_now < 0: t = "BUY"
    elif ms_prev <= ml_prev and ms_now > ml_now and ml_now > 0: t = "SELL"
    else:                                                         t = "NONE"
    return round(ml_now, 4), round(ms_now, 4), round(float(mh.iloc[-1]), 4), t


# ─────────────────────────────────────────────────────────────
# CORE ANALYSIS — single timeframe
# ─────────────────────────────────────────────────────────────

def analyse_tf(df: pd.DataFrame, idx_arr, p1: int, p3: int, p6: int):
    """
    Run all indicators on a (daily or weekly) OHLCV dataframe.
    idx_arr : pandas Series of index close prices aligned to df rows, or None.
    p1/p3/p6: lookback bars for 1M/3M/6M ROC (differs for daily vs weekly).
    Returns a dict, or None if not enough data.
    """
    if df is None or len(df) < max(40, p6 + 4):
        return None

    n     = len(df)
    close = df['Close']
    cmp   = round(float(close.iloc[-1]), 2)

    # ── Indicators ──────────────────────────────────────────
    adx_v, pdi_v, ndi_v = calc_adx(df)
    st_v,  dir_v         = calc_supertrend(df)
    obv_mom              = calc_obv_trend(df)
    ml, ms_v, mh, macd_t = calc_macd(close)
    above_st             = cmp > st_v

    # ROC
    def roc(p):
        return round((float(close.iloc[-1]) / float(close.iloc[-p - 1]) - 1) * 100, 2) if n > p + 1 else None

    r1 = roc(p1)
    r3 = roc(p3)
    r6 = roc(p6)

    # Relative Strength vs Index (3-month)
    rs3 = None
    if idx_arr is not None and len(idx_arr) == n and n > p3 + 1:
        try:
            ix_now  = float(idx_arr.iloc[-1])
            ix_then = float(idx_arr.iloc[-p3 - 1])
            if ix_then > 0 and ix_now > 0:
                stock_ret = (float(close.iloc[-1]) / float(close.iloc[-p3 - 1]) - 1) * 100
                index_ret = (ix_now / ix_then - 1) * 100
                rs3 = round(stock_ret - index_ret, 2)
        except Exception:
            pass

    # ── Scoring ─────────────────────────────────────────────
    sc  = 0
    rsn = []

    # ADX strength (0–20 pts)
    if   adx_v >= 40: sc += 20; rsn.append(f"Strong Trend ADX {adx_v:.0f}")
    elif adx_v >= 25: sc += 12; rsn.append(f"Trending ADX {adx_v:.0f}")
    elif adx_v >= 15: sc +=  5
    # else: ranging — no pts

    # Directional bias (±15 pts)
    if pdi_v > ndi_v: sc += 10; rsn.append(f"+DI {pdi_v:.0f} > -DI {ndi_v:.0f}")
    else:             sc -=  5; rsn.append(f"-DI {ndi_v:.0f} > +DI {pdi_v:.0f}")

    # Supertrend (±10 pts)
    if above_st: sc += 10; rsn.append("Above Supertrend")
    else:        sc -=  8; rsn.append("Below Supertrend")

    # ROC momentum (0–33 pts)
    if r1 is not None:
        if   r1 >  5: sc +=  8; rsn.append(f"1M ROC {r1:+.1f}%")
        elif r1 >  0: sc +=  3
        elif r1 < -5: sc -=  5

    if r3 is not None:
        if   r3 > 15: sc += 15; rsn.append(f"3M ROC {r3:+.1f}%")
        elif r3 >  5: sc +=  8; rsn.append(f"3M ROC {r3:+.1f}%")
        elif r3 >  0: sc +=  3
        elif r3 < -10: sc -= 8

    if r6 is not None:
        if   r6 > 25:  sc += 10; rsn.append(f"6M ROC {r6:+.1f}%")
        elif r6 > 10:  sc +=  5
        elif r6 < -15: sc -=  5

    # Relative Strength (0–15 pts)
    if rs3 is not None:
        if   rs3 > 10: sc += 15; rsn.append(f"RS +{rs3:.1f}% vs Index")
        elif rs3 >  5: sc += 10; rsn.append(f"RS +{rs3:.1f}% vs Index")
        elif rs3 >  0: sc +=  5
        elif rs3 < -5: sc -=  5

    # OBV (0–10 pts)
    if obv_mom > 0: sc += 10; rsn.append("OBV Rising")
    else:           sc -=  3

    # MACD bonus/penalty
    if   macd_t == "BUY":  sc +=  8; rsn.append("MACD Crossover Buy")
    elif macd_t == "SELL": sc -=  5

    sc = max(-50, min(100, sc))

    if   sc >= 70: sig = "STRONG MOMENTUM"
    elif sc >= 45: sig = "MOMENTUM"
    elif sc >= 20: sig = "WATCH"
    elif sc >=  0: sig = "NEUTRAL"
    else:          sig = "WEAK"

    last_date = str(df['Datetime'].iloc[-1].date()) if 'Datetime' in df.columns else ''

    return dict(
        cmp=cmp,
        adx=round(adx_v, 1), pdi=round(pdi_v, 1), ndi=round(ndi_v, 1),
        st_val=round(st_v, 2), st_dir=dir_v, above_st=above_st,
        roc1=r1, roc3=r3, roc6=r6,
        rs3=rs3,
        obv_rising=(obv_mom > 0),
        macd_line=ml, macd_sig=ms_v, macd_hist=mh, macd_type=macd_t,
        score=sc, signal=sig, reasons=rsn,
        last_date=last_date,
    )


def analyse(df: pd.DataFrame, idx_df):
    """
    Returns (daily_dict, weekly_dict) for a stock.
    Either may be None if there isn't enough data for that timeframe.
    """
    if df is None or len(df) < MIN_ROWS:
        return None, None

    df = df.tail(LOOKBACK).reset_index(drop=True).copy()

    # ── Align index to daily dates ───────────────────────────
    ic_d = None
    if idx_df is not None:
        try:
            idx_s = idx_df.set_index('Datetime')['Close']
            ic_d  = idx_s.reindex(df['Datetime']).ffill().reset_index(drop=True)
        except Exception:
            pass

    daily = analyse_tf(df, ic_d, p1=20, p3=63, p6=126)

    # ── Weekly resampling ────────────────────────────────────
    wdf  = to_weekly(df)
    ic_w = None
    if idx_df is not None:
        try:
            idx_s = idx_df.set_index('Datetime')['Close']
            idx_w = idx_s.resample('W-FRI').last().dropna()
            ic_w  = idx_w.reindex(wdf['Datetime']).ffill().reset_index(drop=True)
        except Exception:
            pass

    weekly = analyse_tf(wdf, ic_w, p1=4, p3=13, p6=26)

    return daily, weekly


# ─────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────

def _pct(v, suffix=""):
    """Coloured percentage badge."""
    if v is None:
        return '<span style="color:#444">—</span>'
    col   = "#00c853" if v >= 0 else "#f85149"
    arrow = "▲" if v >= 0 else "▼"
    return f'<span style="color:{col};font-weight:600">{arrow} {abs(v):.1f}%{suffix}</span>'


def _score_bar(sc):
    """Compact score bar, score range −50…100."""
    pct = max(0.0, min(100.0, (sc + 50) * 100.0 / 150.0))
    hue = max(0, min(120, int(pct * 1.2)))
    return (
        f'<div style="display:flex;align-items:center;gap:5px;min-width:110px">'
        f'<div style="flex:1;height:5px;background:#21262d;border-radius:3px">'
        f'<div style="width:{pct:.0f}%;height:100%;background:hsl({hue},75%,42%);border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-size:10px;color:#8b949e;font-family:monospace;flex-shrink:0">{sc:+d}</span>'
        f'</div>'
    )


def _tbody_row(r: dict) -> str:
    s   = r['signal']
    em, col, bg = SIGNAL_META.get(s, ("", "#888", "#111"))
    si  = SIGNAL_ORDER.index(s) if s in SIGNAL_ORDER else 99
    rsn = "<br>".join(r.get('reasons', []))

    # ADX
    adx = r['adx']; pdi = r['pdi']; ndi = r['ndi']
    if   adx >= 40: ac, al = "#00c853", "Strong"
    elif adx >= 25: ac, al = "#69db7c", "Trending"
    elif adx >= 15: ac, al = "#d29922", "Weak Trend"
    else:           ac, al = "#6e7681", "Ranging"
    pc = "#00c853" if pdi > ndi else "#f85149"
    nc = "#f85149" if pdi > ndi else "#00c853"

    # Supertrend
    cmp  = r['cmp'];  stv = r['st_val']; ab = r['above_st']
    stc  = "#00c853" if ab else "#f85149"
    stl  = "🟢 ABOVE" if ab else "🔴 BELOW"
    pst  = round((cmp - stv) / stv * 100, 1) if stv > 0 else 0.0

    # OBV
    obv = r.get('obv_rising', False)
    oc  = "#00c853" if obv else "#f85149"
    ol  = "▲ Rising" if obv else "▼ Falling"

    # MACD
    mt  = r.get('macd_type', 'NONE')
    ml  = r.get('macd_line', 0.0)
    msv = r.get('macd_sig',  0.0)
    mh  = r.get('macd_hist', 0.0)
    mhc = "#00c853" if mh >= 0 else "#f85149"
    mha = "▲" if mh >= 0 else "▼"
    if   mt == "BUY":  mb = '<span class="sig-badge" style="background:#00875a">📈 BUY</span>'
    elif mt == "SELL": mb = '<span class="sig-badge" style="background:#b71c1c">📉 SELL</span>'
    else:              mb = '<span style="color:#444;font-size:10px">—</span>'
    mdv = 0 if mt == "NONE" else (1 if mt == "BUY" else -1)

    # RS
    rs3 = r.get('rs3')
    rs_html = (
        f'{_pct(rs3)}<br><span style="font-size:10px;color:#555">vs Nifty</span>'
        if rs3 is not None else '<span style="color:#444">—</span>'
    )
    r1 = r.get('roc1'); r3 = r.get('roc3'); r6 = r.get('roc6')
    dn  = disp_name(r['name'])

    return f"""<tr style="background:{bg}" data-macd-type="{mt}" data-signal="{s}">
  <td class="sticky-col" data-val="{r['name']}"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A{dn}" target="_blank">{dn}</a></td>
  <td data-val="{cmp}" style="font-weight:700">₹{cmp:,.2f}</td>
  <td data-val="{si}" data-signal="{s}" style="white-space:nowrap"><span class="sig-badge" style="background:{col}">{em} {s}</span></td>
  <td data-val="{r['score']}">{_score_bar(r['score'])}</td>
  <td data-val="{adx}"><b style="color:{ac}">{adx:.1f}</b> <span style="font-size:10px;color:{ac}">{al}</span><br><span style="font-size:11px"><span style="color:{pc}">+DI {pdi:.1f}</span>&nbsp;<span style="color:{nc}">-DI {ndi:.1f}</span></span></td>
  <td data-val="{r1 if r1 is not None else -999}">{_pct(r1)}</td>
  <td data-val="{r3 if r3 is not None else -999}">{_pct(r3)}</td>
  <td data-val="{r6 if r6 is not None else -999}">{_pct(r6)}</td>
  <td data-val="{rs3 if rs3 is not None else -999}">{rs_html}</td>
  <td data-val="{1 if ab else -1}"><span style="color:{stc};font-weight:700">{stl}</span><br><span style="font-size:11px;color:#6e7681">₹{stv:,.2f}&nbsp;({pst:+.1f}%)</span></td>
  <td data-val="{1 if obv else -1}"><span style="color:{oc};font-weight:600">{ol}</span></td>
  <td data-val="{mdv}" style="font-size:11px;min-width:150px;white-space:nowrap"><span style="color:#8b949e">L:</span><b>{ml:+.4f}</b>&nbsp;<span style="color:#8b949e">S:</span><b>{msv:+.4f}</b><br><span style="color:{mhc}">{mha}&nbsp;{abs(mh):.4f}</span>&nbsp;{mb}</td>
  <td class="reasons" data-val="{len(r.get('reasons', []))}">{rsn or "—"}</td>
  <td data-val="{r.get('last_date', '')}" style="font-size:11px;color:#6e7681">{r.get('last_date', '')}</td>
</tr>"""


# ─────────────────────────────────────────────────────────────
# HTML BUILDER
# ─────────────────────────────────────────────────────────────

def build_html(rows: list, title: str, subtitle: str, generated_at: str) -> str:

    def sort_key(r, tf):
        m = r.get(tf)
        return (SIGNAL_ORDER.index(m['signal']) if m and m['signal'] in SIGNAL_ORDER else 99,
                -(m['score'] if m else 0))

    dr = sorted([r for r in rows if r.get('daily')],  key=lambda r: sort_key(r, 'daily'))
    wr = sorted([r for r in rows if r.get('weekly')], key=lambda r: sort_key(r, 'weekly'))

    def sig_counts(lst, tf):
        c = {}
        for r in lst:
            m = r.get(tf)
            if m:
                c[m['signal']] = c.get(m['signal'], 0) + 1
        return c

    dc = sig_counts(dr, 'daily')
    wc = sig_counts(wr, 'weekly')

    def pill_set(cnt, view):
        total = sum(cnt.values())
        h = (f'<div class="pill active-pill" id="pill-all-{view}" '
             f'style="border-color:#58a6ff;color:#58a6ff" '
             f'onclick="setFilter(\'all\',this)">All <b>{total}</b></div>')
        for sig in SIGNAL_ORDER:
            if sig in cnt:
                em2, col2, _ = SIGNAL_META.get(sig, ("", "#888", ""))
                h += (f'<div class="pill" style="border-color:{col2};color:{col2}" '
                      f'onclick="setFilter(\'{sig}\',this)">{em2} {sig} <b>{cnt[sig]}</b></div>')
        return h

    def make_tbody(lst, tf):
        return "".join(
            _tbody_row({**r[tf], 'name': r['name'], 'ticker': r['ticker']})
            for r in lst if r.get(tf)
        )

    td_d  = make_tbody(dr, 'daily')
    td_w  = make_tbody(wr, 'weekly')
    ps_d  = pill_set(dc, 'daily')
    ps_w  = pill_set(wc, 'weekly')

    n750 = len(rows)

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
    min-height:100vh;min-width:1500px;padding:24px;
  }}

  /* ── Header ── */
  .header{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:24px;flex-wrap:wrap;gap:16px}}
  .header h1{{
    font-size:clamp(18px,3vw,28px);font-weight:700;
    background:linear-gradient(135deg,#58a6ff,#a371f7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  }}
  .header .meta{{color:var(--muted);font-size:13px;margin-top:4px}}
  .badge{{
    background:var(--surface2);border:1px solid var(--border);
    border-radius:20px;padding:6px 14px;font-size:12px;
    font-family:'IBM Plex Mono',monospace;white-space:nowrap;
  }}

  /* ── View toggle ── */
  .view-toggle{{
    display:flex;gap:0;margin-bottom:20px;
    border:1px solid var(--border);border-radius:var(--radius);
    overflow:hidden;width:fit-content;
  }}
  .view-btn{{
    background:var(--surface);border:none;color:var(--muted);
    padding:10px 28px;font-size:13px;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;font-weight:600;
    transition:all .2s;letter-spacing:.3px;
  }}
  .view-btn.active{{background:var(--accent);color:#000}}
  .view-btn:hover:not(.active){{background:var(--surface2);color:var(--accent)}}

  /* ── Summary pills ── */
  .summary{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px}}
  .pill{{
    border:1.5px solid;border-radius:20px;
    padding:5px 14px;font-size:12px;font-weight:600;
    font-family:'IBM Plex Mono',monospace;
    cursor:pointer;transition:all .2s;user-select:none;
  }}
  .pill:hover{{opacity:.8;transform:translateY(-1px)}}
  .active-pill{{box-shadow:0 0 0 2px currentColor;background:rgba(255,255,255,.05)}}

  /* ── Controls ── */
  .controls{{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:16px}}
  .search-box{{
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--radius);padding:8px 14px;
    color:var(--text);font-size:14px;width:220px;
    font-family:'IBM Plex Sans',sans-serif;
  }}
  .search-box:focus{{outline:none;border-color:var(--accent)}}
  .export-btn{{
    background:var(--surface2);border:1px solid var(--green);
    color:var(--green);border-radius:var(--radius);
    padding:7px 14px;font-size:12px;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;transition:all .2s;
  }}
  .export-btn:hover{{background:rgba(63,185,80,.12)}}
  .macd-buy-btn{{
    background:var(--surface);border:1.5px solid #00c853;
    color:#00c853;border-radius:var(--radius);
    padding:7px 14px;font-size:12px;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;transition:all .2s;
  }}
  .macd-buy-btn:hover,.macd-buy-btn.active{{background:rgba(0,200,83,.12);box-shadow:0 0 0 1.5px #00c853}}
  .macd-sell-btn{{
    background:var(--surface);border:1.5px solid #f85149;
    color:#f85149;border-radius:var(--radius);
    padding:7px 14px;font-size:12px;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;transition:all .2s;
  }}
  .macd-sell-btn:hover,.macd-sell-btn.active{{background:rgba(248,81,73,.12);box-shadow:0 0 0 1.5px #f85149}}
  .btn-label{{
    font-size:10px;color:var(--muted);font-family:'IBM Plex Mono',monospace;
    text-transform:uppercase;letter-spacing:.5px;
    padding:0 4px 0 10px;border-left:1px solid var(--border);
  }}

  /* ── Table wrapper ── */
  .table-wrap{{
    overflow-x:auto;border-radius:var(--radius);
    border:1px solid var(--border);background:var(--surface);
  }}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  thead th{{
    background:var(--surface2);padding:11px 14px;text-align:left;
    font-size:11px;text-transform:uppercase;letter-spacing:.6px;
    color:var(--muted);white-space:nowrap;cursor:pointer;
    user-select:none;position:sticky;top:0;z-index:2;
    border-bottom:1px solid var(--border);transition:color .2s;
  }}
  thead th:first-child{{position:sticky;left:0;top:0;z-index:3}}
  thead th:hover{{color:var(--accent)}}
  thead th.sort-asc::after {{content:" ▲";color:var(--accent)}}
  thead th.sort-desc::after{{content:" ▼";color:var(--accent)}}
  tbody tr{{border-bottom:1px solid var(--border);transition:filter .15s}}
  tbody tr:hover{{filter:brightness(1.18)}}
  tbody tr:last-child{{border-bottom:none}}
  tbody td{{padding:10px 14px;vertical-align:middle;color:var(--text)}}
  .sticky-col{{
    position:sticky;left:0;z-index:1;
    font-family:'IBM Plex Mono',monospace;
    font-weight:600;font-size:12px;
    background:inherit;border-right:1px solid var(--border);
    min-width:90px;
  }}
  .sticky-col a{{color:var(--accent);text-decoration:none}}
  .sticky-col a:hover{{text-decoration:underline}}
  .sig-badge{{
    color:#fff;border-radius:4px;padding:3px 8px;
    font-size:11px;font-weight:700;
    font-family:'IBM Plex Mono',monospace;white-space:nowrap;
  }}
  .reasons{{font-size:11px;color:var(--muted);max-width:200px;line-height:1.5}}

  /* ── Legend ── */
  .legend{{
    display:flex;flex-wrap:wrap;gap:16px;
    background:var(--surface);border:1px solid var(--border);
    border-radius:var(--radius);padding:14px 18px;margin-bottom:20px;
    font-size:11px;color:var(--muted);
  }}
  .legend-item{{display:flex;flex-direction:column;gap:2px}}
  .legend-item b{{color:var(--text);font-size:12px}}

  /* ── Footer ── */
  .footer{{
    margin-top:32px;text-align:center;
    font-size:11px;color:var(--muted);line-height:1.8;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{title}</h1>
    <div class="meta">{subtitle}</div>
    <div class="meta" style="margin-top:4px">Generated: {generated_at} &nbsp;|&nbsp; {n750} stocks analysed</div>
  </div>
  <div class="badge">📊 Institutional Momentum Screener</div>
</div>

<div class="legend">
  <div class="legend-item"><b>ADX</b>≥40 Strong · ≥25 Trending · ≥15 Weak · &lt;15 Ranging</div>
  <div class="legend-item"><b>Supertrend</b>ATR(10)×3 · 🟢 price above = bullish · 🔴 below = bearish</div>
  <div class="legend-item"><b>ROC</b>Rate of change · 1M=20D · 3M=63D · 6M=126D (daily) | 4/13/26 bars (weekly)</div>
  <div class="legend-item"><b>RS vs Index</b>Stock 3M return minus Nifty 3M return</div>
  <div class="legend-item"><b>OBV</b>Fast EMA(10) − Slow EMA(30) of OBV · Rising = bullish accumulation</div>
  <div class="legend-item"><b>MACD</b>Fast 5 · Slow 35 · Signal 5 · Buy when line crosses above signal &amp; line &lt;0</div>
</div>

<div class="view-toggle">
  <button class="view-btn active" id="view-btn-daily"  onclick="switchView('daily')">📅 Daily</button>
  <button class="view-btn"        id="view-btn-weekly" onclick="switchView('weekly')">📆 Weekly</button>
</div>

<div id="summary-daily"  class="summary">{ps_d}</div>
<div id="summary-weekly" class="summary" style="display:none">{ps_w}</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="🔍 Search stock…" oninput="filterRows()">
  <button class="export-btn" onclick="exportNames()">📋 Export Stock Names</button>
  <span class="btn-label">MACD</span>
  <button class="macd-buy-btn  macd-btn" onclick="setMacdFilter('BUY',this)">📈 MACD Buy</button>
  <button class="macd-sell-btn macd-btn" onclick="setMacdFilter('SELL',this)">📉 MACD Sell</button>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead><tr>
  <th onclick="sortTable(0)">Stock</th>
  <th onclick="sortTable(1)">CMP</th>
  <th onclick="sortTable(2)">Signal</th>
  <th onclick="sortTable(3)">Score</th>
  <th onclick="sortTable(4)">ADX / DI</th>
  <th onclick="sortTable(5)">ROC 1M</th>
  <th onclick="sortTable(6)">ROC 3M</th>
  <th onclick="sortTable(7)">ROC 6M</th>
  <th onclick="sortTable(8)">RS vs Index</th>
  <th onclick="sortTable(9)">Supertrend</th>
  <th onclick="sortTable(10)">OBV Trend</th>
  <th onclick="sortTable(11)">MACD (5/35/5)</th>
  <th onclick="sortTable(12)">Reasons</th>
  <th onclick="sortTable(13)">Last Date</th>
</tr></thead>
<tbody id="tbody-daily">{td_d}</tbody>
<tbody id="tbody-weekly" style="display:none">{td_w}</tbody>
</table>
</div>

<div class="footer">
  <b>Disclaimer:</b> This tool is for educational &amp; research purposes only. Not SEBI-registered advice.<br>
  All indicators are algorithmic approximations. Always conduct your own due diligence before investing.
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
    document.getElementById('tbody-'    + x).style.display = x === v ? '' : 'none';
    document.getElementById('summary-' + x).style.display = x === v ? '' : 'none';
    document.getElementById('view-btn-'+ x).classList.toggle('active', x === v);
  }});
  // Reset all row visibility in the new view
  getCurTbody().querySelectorAll('tr').forEach(r => r.style.display = '');
  // Reset filters
  activeFilter = 'all';
  activeMacd   = 'all';
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active-pill'));
  const allPill = document.getElementById('pill-all-' + v);
  if (allPill) allPill.classList.add('active-pill');
  document.querySelectorAll('.macd-btn').forEach(b => b.classList.remove('active'));
  // Re-apply search if any
  filterRows();
}}

function filterRows() {{
  const q = document.getElementById('search').value.toLowerCase();
  getCurTbody().querySelectorAll('tr').forEach(r => {{
    const name   = r.cells[0].textContent.toLowerCase();
    const signal = (r.dataset.signal || '').toLowerCase();
    const mt     = (r.dataset.macdType || '').toUpperCase();
    const mQ = name.includes(q);
    const mF = activeFilter === 'all' || signal === activeFilter.toLowerCase();
    const mM = activeMacd   === 'all' || mt     === activeMacd;
    r.style.display = (mQ && mF && mM) ? '' : 'none';
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
    activeMacd = 'all';
    btn.classList.remove('active');
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
  ths.forEach((th, i) => {{
    th.classList.remove('sort-asc', 'sort-desc');
    if (i === col) th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
  }});
  rows.sort((a, b) => {{
    const av = a.cells[col]?.dataset.val ?? a.cells[col]?.textContent ?? '';
    const bv = b.cells[col]?.dataset.val ?? b.cells[col]?.textContent ?? '';
    const an = parseFloat(av), bn = parseFloat(bv);
    if (!isNaN(an) && !isNaN(bn)) return (an - bn) * sortDir;
    return av.localeCompare(bv) * sortDir;
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
    print("  Momentum Screener — Institutional Indicators")
    print("=" * 60)

    nifty750 = load_nifty750()
    print(f"  Nifty 750 list : {len(nifty750)} normalised tickers")

    idx_path = find_index_csv()
    idx_df   = load_csv(idx_path) if idx_path else None
    if idx_path:
        print(f"  Index file     : {os.path.basename(idx_path)}")
    else:
        print("  ⚠  No index CSV found — RS vs Index column will be blank")

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if idx_path:
        abs_idx = os.path.abspath(idx_path)
        csv_files = [f for f in csv_files if os.path.abspath(f) != abs_idx]

    if not csv_files:
        print(f"\n  ERROR: No CSV files in {DATA_DIR}")
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
            daily, weekly = analyse(df, idx_df)
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

    print(f"\n\n  ✓ Analysed  : {len(nifty750_rows) + len(other_rows)} stocks")
    print(f"    Nifty 750 : {len(nifty750_rows)}")
    print(f"    Other     : {len(other_rows)}")
    print(f"    Skipped   : {len(skipped)}")
    if errors:
        print(f"    Errors    : {len(errors)}")
        for e in errors[:5]:
            print(f"      {e}")

    subtitle = (
        "ADX · Supertrend · ROC 1M/3M/6M · RS vs Index · OBV · MACD (5/35/5)"
        "  |  Daily & Weekly Timeframes"
    )

    for rows, path, title in [
        (nifty750_rows, OUT_NIFTY750, "Nifty 750 — Momentum Screener"),
        (other_rows,    OUT_OTHERS,   "Other Stocks — Momentum Screener"),
    ]:
        html = build_html(rows if rows else [], title, subtitle, generated_at)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        status = f"({len(rows)} stocks)" if rows else "(empty)"
        print(f"  ✅  {path}  {status}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
