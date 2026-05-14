#!/usr/bin/env python3
"""
NSE Institutional Scanner
==========================
5 institutional-grade scanning methodologies — OHLCV data only.

Scanners
--------
  1. RS Rating       — IBD-style weighted relative strength (1–99 percentile)
  2. Stage Analysis  — Stan Weinstein Stage 1 / 2 / 3 / 4
  3. VCP             — Volatility Contraction Pattern (Minervini)
  4. Accumulation    — Smart-money accumulation / distribution score
  5. EMA Ribbon+ADX  — Multi-EMA alignment + ADX trend strength

Output
------
  nifty750_scanner.html   — Nifty 750 stocks (6 tabs)
  other_stocks_scanner.html — remaining NSE stocks (6 tabs)
"""

import os, sys, glob, math, time, json
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
DATA_DIR       = "../nse_data_cache"
NIFTY750_FILE  = "../nifty750.txt"
OUTPUT_DIR     = "."
MIN_ROWS       = 100          # minimum trading days required
TV_BASE        = "https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A"

# Weinstein Stage MA periods (trading days)
STAGE_MA_LONG  = 150          # 30-week MA
STAGE_MA_SHORT = 50           # 10-week MA

# ADX / EMA periods
ADX_PERIOD     = 14
EMA_PERIODS    = [8, 21, 55, 200]

# Accumulation lookback
ACCUM_PERIOD   = 50

# VCP
VCP_LOOKBACK   = 252          # max bars to scan for contractions


# ──────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────

def wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (used in ADX / ATR)."""
    result = series.copy() * np.nan
    result.iloc[period - 1] = series.iloc[:period].sum()
    for i in range(period, len(series)):
        result.iloc[i] = result.iloc[i - 1] - (result.iloc[i - 1] / period) + series.iloc[i]
    return result


def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        dt = next((c for c in df.columns if c.lower() in ("datetime", "date")), None)
        if dt is None:
            return None
        df[dt] = pd.to_datetime(df[dt], errors="coerce")
        df = df.rename(columns={dt: "Date"})
        df = (df.sort_values("Date")
                .dropna(subset=["Date", "Close"])
                .query("Close > 0")
                .reset_index(drop=True))
        return df if len(df) >= MIN_ROWS else None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────
# 1. RS RATING  (IBD weighted, cross-sectional percentile)
# ──────────────────────────────────────────────────────────

def rs_raw_score(df: pd.DataFrame) -> float | None:
    """
    Weighted 12-month return:
      40% × last-3M return  +
      20% × prior 3M (3–6M) +
      20% × prior 3M (6–9M) +
      20% × prior 3M (9–12M)
    """
    today = df["Date"].iloc[-1]
    cp    = df["Close"].iloc[-1]

    def px(n_days):
        t = today - timedelta(days=n_days)
        s = df[df["Date"] <= t]
        return float(s["Close"].iloc[-1]) if len(s) >= 5 else None

    p3  = px(63)
    p6  = px(126)
    p9  = px(189)
    p12 = px(252)

    parts = []
    weights = []
    for px_val, w in [(p3, 0.40), (p6, 0.20), (p9, 0.20), (p12, 0.20)]:
        if px_val and px_val > 0:
            parts.append((cp / px_val - 1) * w)
            weights.append(w)

    if not parts or sum(weights) < 0.40:
        return None
    # normalise if we don't have full 12M history
    return sum(parts) / sum(weights)


def assign_rs_ratings(stocks: list[dict]) -> None:
    """Compute cross-sectional percentile RS rating (1–99) in-place."""
    valid = [(i, s["rs_raw"]) for i, s in enumerate(stocks)
             if s["rs_raw"] is not None and not math.isnan(s["rs_raw"])]
    if not valid:
        return
    idxs, vals = zip(*valid)
    vals = np.array(vals, dtype=float)
    ranks = (vals.argsort().argsort() + 1) / len(vals) * 99
    ranks = np.clip(ranks.astype(int), 1, 99)
    for list_idx, rating in zip(idxs, ranks):
        stocks[list_idx]["rs_rating"] = int(rating)


# ──────────────────────────────────────────────────────────
# 2. STAGE ANALYSIS  (Stan Weinstein)
# ──────────────────────────────────────────────────────────

def stage_analysis(df: pd.DataFrame) -> dict:
    """
    Returns stage (1–4), MA150, MA50, MA150 slope, details string.

    Stage 2 criteria (ideal long):
      • Price > MA150
      • MA150 rising (positive slope over 4 weeks)
      • Price > MA50
      • MA50 > MA150

    Stage 4 (avoid / short):
      • Price < MA150
      • MA150 declining

    Stage 1 / 3 are transitional.
    """
    close = df["Close"]
    if len(close) < STAGE_MA_LONG + 10:
        return {"stage": 0, "stage_label": "N/A", "ma150": None, "ma50": None,
                "ma150_slope": None, "stage_detail": "Insufficient data"}

    ma150 = close.rolling(STAGE_MA_LONG).mean()
    ma50  = close.rolling(STAGE_MA_SHORT).mean()

    cur_price  = float(close.iloc[-1])
    cur_ma150  = float(ma150.iloc[-1])
    cur_ma50   = float(ma50.iloc[-1])

    # Slope of MA150 over last 20 bars (normalised)
    ma150_20ago = float(ma150.iloc[-21]) if len(ma150) > 21 else cur_ma150
    slope = (cur_ma150 - ma150_20ago) / ma150_20ago if ma150_20ago > 0 else 0.0

    above_ma150 = cur_price > cur_ma150
    ma150_rising = slope > 0.001
    ma150_falling = slope < -0.001
    above_ma50  = cur_price > cur_ma50
    ma50_above_ma150 = cur_ma50 > cur_ma150

    if above_ma150 and ma150_rising and above_ma50 and ma50_above_ma150:
        stage, label = 2, "Stage 2 ▲"
    elif above_ma150 and not ma150_rising and not ma150_falling:
        stage, label = 3, "Stage 3 ◆"
    elif not above_ma150 and ma150_falling:
        stage, label = 4, "Stage 4 ▼"
    else:
        stage, label = 1, "Stage 1 ●"

    # % above/below MA150
    pct_from_ma = (cur_price / cur_ma150 - 1) * 100

    detail = (f"Price {'above' if above_ma150 else 'below'} MA150 by {abs(pct_from_ma):.1f}% | "
              f"MA150 slope {slope*100:+.2f}%/20d | "
              f"MA50 {'>' if ma50_above_ma150 else '<'} MA150")

    return dict(
        stage        = stage,
        stage_label  = label,
        ma150        = round(cur_ma150, 2),
        ma50         = round(cur_ma50, 2),
        ma150_slope  = round(slope * 100, 3),
        pct_from_ma  = round(pct_from_ma, 2),
        stage_detail = detail,
    )


# ──────────────────────────────────────────────────────────
# 3. VCP — Volatility Contraction Pattern (Minervini)
# ──────────────────────────────────────────────────────────

def vcp_scan(df: pd.DataFrame) -> dict:
    """
    Detects VCP by finding a series of price swings where:
      • Each high-to-low depth < previous depth (contracting)
      • Volume dries up in later contractions
      • At least 2 contractions found
      • Current price within 10% of last pivot high (near breakout)

    Returns vcp_score (0–100), contraction count, tightness %, pivot high.
    """
    if len(df) < 60:
        return {"vcp_score": 0, "vcp_contractions": 0,
                "vcp_tightness": None, "vcp_pivot": None, "vcp_detail": "Insufficient data"}

    window = df.tail(min(VCP_LOOKBACK, len(df))).copy().reset_index(drop=True)
    highs  = window["High"].values
    lows   = window["Low"].values
    closes = window["Close"].values
    vols   = window["Volume"].values
    n      = len(window)

    # Find local swing highs (peak if higher than 10 bars either side)
    swing_highs = []
    for i in range(10, n - 5):
        if highs[i] == max(highs[max(0, i-10):i+11]):
            swing_highs.append(i)

    if len(swing_highs) < 2:
        return {"vcp_score": 0, "vcp_contractions": 0,
                "vcp_tightness": None, "vcp_pivot": None,
                "vcp_detail": "No swing highs found"}

    # Measure contraction between consecutive swing highs
    contractions = []
    for k in range(len(swing_highs) - 1):
        i1, i2 = swing_highs[k], swing_highs[k + 1]
        segment_lows  = lows[i1:i2 + 1]
        segment_vols  = vols[i1:i2 + 1]
        swing_low     = segment_lows.min()
        depth_pct     = (highs[i1] - swing_low) / highs[i1] * 100
        avg_vol       = float(np.mean(segment_vols)) if len(segment_vols) > 0 else 0
        contractions.append({
            "idx": i1,
            "high": highs[i1],
            "low":  swing_low,
            "depth": depth_pct,
            "avg_vol": avg_vol,
        })

    if len(contractions) < 2:
        return {"vcp_score": 0, "vcp_contractions": 0,
                "vcp_tightness": None, "vcp_pivot": None,
                "vcp_detail": "Not enough contractions"}

    # Check if depths are shrinking (VCP hallmark)
    valid_vcp = 0
    for k in range(1, len(contractions)):
        if contractions[k]["depth"] < contractions[k - 1]["depth"]:
            valid_vcp += 1

    # Volume declining in later contractions
    vol_declining = 0
    for k in range(1, len(contractions)):
        if contractions[k]["avg_vol"] < contractions[k - 1]["avg_vol"]:
            vol_declining += 1

    # Latest contraction tightness (< 10% = tight, < 5% = very tight)
    latest_depth  = contractions[-1]["depth"]
    pivot_high    = float(contractions[-1]["high"])
    cur_price     = float(closes[-1])
    pct_from_pivot = (cur_price / pivot_high - 1) * 100

    # Score
    score = 0
    score += min(valid_vcp * 15, 40)          # up to 40 pts for contracting depths
    score += min(vol_declining * 15, 30)       # up to 30 pts for drying volume
    if latest_depth < 5:   score += 20
    elif latest_depth < 10: score += 12
    elif latest_depth < 15: score += 5
    if -10 <= pct_from_pivot <= 2: score += 10  # near pivot = breakout candidate

    score = min(score, 100)

    detail = (f"{valid_vcp}/{len(contractions)-1} contractions shrinking | "
              f"Vol declining: {vol_declining}/{len(contractions)-1} | "
              f"Latest depth: {latest_depth:.1f}% | "
              f"Price vs pivot: {pct_from_pivot:+.1f}%")

    return dict(
        vcp_score        = score,
        vcp_contractions = valid_vcp,
        vcp_tightness    = round(latest_depth, 2),
        vcp_pivot        = round(pivot_high, 2),
        vcp_pct_pivot    = round(pct_from_pivot, 2),
        vcp_detail       = detail,
    )


# ──────────────────────────────────────────────────────────
# 4. ACCUMULATION / DISTRIBUTION SCORE
# ──────────────────────────────────────────────────────────

def accumulation_score(df: pd.DataFrame) -> dict:
    """
    Multi-factor smart-money accumulation score (0–100):

      A) Up-day volume ratio  — volume on up-days / total volume (50d)
      B) On-Balance Volume trend — OBV slope direction
      C) Chaikin Money Flow  — [(close-low)-(high-close)] / (high-low) × volume
      D) Large-vol up-days   — days with vol > 1.5× avg AND price up
    """
    if len(df) < ACCUM_PERIOD + 10:
        return {"accum_score": 0, "accum_label": "N/A",
                "up_vol_ratio": None, "cmf": None, "accum_detail": "Insufficient data"}

    recent = df.tail(ACCUM_PERIOD).copy().reset_index(drop=True)
    close  = recent["Close"].values
    high   = recent["High"].values
    low    = recent["Low"].values
    vol    = recent["Volume"].values.astype(float)
    n      = len(recent)

    # A) Up-day volume ratio
    price_diff  = np.diff(close, prepend=close[0])
    up_mask     = price_diff > 0
    up_vol      = vol[up_mask].sum()
    total_vol   = vol.sum()
    up_vol_ratio = up_vol / total_vol if total_vol > 0 else 0.5

    # B) OBV slope (normalised)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]
    obv_slope = (obv[-1] - obv[0]) / (abs(obv[0]) + 1e-9)
    obv_rising = obv_slope > 0

    # C) Chaikin Money Flow (21-day)
    hl = high - low
    hl = np.where(hl == 0, 1e-9, hl)
    mfm = ((close - low) - (high - close)) / hl        # money flow multiplier
    mfv = mfm * vol                                    # money flow volume
    cmf = mfv[-21:].sum() / (vol[-21:].sum() + 1e-9)  # CMF

    # D) Power days (large vol + price up)
    avg_vol     = vol.mean()
    power_days  = int(((vol > 1.5 * avg_vol) & (price_diff > 0)).sum())
    weak_days   = int(((vol > 1.5 * avg_vol) & (price_diff < 0)).sum())
    power_ratio = (power_days - weak_days) / max(power_days + weak_days, 1)

    # Composite score
    score = 0
    score += int(np.clip((up_vol_ratio - 0.5) * 200, 0, 30))   # 0–30
    score += 20 if obv_rising else 0                            # 20
    score += int(np.clip(cmf * 100, 0, 30))                     # 0–30
    score += int(np.clip(power_ratio * 20, 0, 20))              # 0–20
    score = min(score, 100)

    if score >= 70:   label = "Strong Accumulation"
    elif score >= 50: label = "Accumulation"
    elif score >= 35: label = "Neutral"
    elif score >= 20: label = "Distribution"
    else:             label = "Strong Distribution"

    detail = (f"Up-vol ratio: {up_vol_ratio:.2f} | "
              f"OBV: {'↑' if obv_rising else '↓'} | "
              f"CMF: {cmf:+.3f} | "
              f"Power days: {power_days} vs Weak: {weak_days}")

    return dict(
        accum_score   = score,
        accum_label   = label,
        up_vol_ratio  = round(float(up_vol_ratio), 3),
        cmf           = round(float(cmf), 4),
        power_days    = power_days,
        weak_days     = weak_days,
        accum_detail  = detail,
    )


# ──────────────────────────────────────────────────────────
# 5. EMA RIBBON + ADX
# ──────────────────────────────────────────────────────────

def ema_adx_analysis(df: pd.DataFrame) -> dict:
    """
    EMA Ribbon alignment + ADX trend strength.

    EMA alignment score: how many EMAs are in bull order (8>21>55>200)
    ADX > 25 = trending market (direction-agnostic)
    +DI > -DI confirms bullish trend
    """
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]

    if len(close) < 220:
        return {"adx": None, "plus_di": None, "minus_di": None,
                "ema_align_score": 0, "ema_align_label": "N/A",
                "ema8": None, "ema21": None, "ema55": None, "ema200": None,
                "trend_score": 0, "trend_detail": "Insufficient data"}

    # EMAs
    emas = {p: float(close.ewm(span=p, adjust=False).mean().iloc[-1])
            for p in EMA_PERIODS}

    e8, e21, e55, e200 = emas[8], emas[21], emas[55], emas[200]
    cur = float(close.iloc[-1])

    # Alignment: price > e8 > e21 > e55 > e200  = fully bullish
    checks = [cur > e8, e8 > e21, e21 > e55, e55 > e200]
    ema_align_score = sum(checks)           # 0–4
    align_labels = {0: "Full Bear 🔴", 1: "Weak Bear 🟠",
                    2: "Neutral 🟡",   3: "Weak Bull 🟢", 4: "Full Bull 💚"}
    ema_align_label = align_labels[ema_align_score]

    # ADX calculation (Wilder)
    prev_close = close.shift(1)
    tr   = pd.concat([high - low,
                      (high - prev_close).abs(),
                      (low  - prev_close).abs()], axis=1).max(axis=1)
    pdm  = (high - high.shift(1)).clip(lower=0)
    ndm  = (low.shift(1) - low).clip(lower=0)
    # zero out where the other direction is larger
    pdm  = pdm.where(pdm > ndm, 0.0)
    ndm  = ndm.where(ndm > pdm, 0.0)

    atr14  = wilder_smooth(tr.dropna(),  ADX_PERIOD)
    pdm14  = wilder_smooth(pdm.dropna(), ADX_PERIOD)
    ndm14  = wilder_smooth(ndm.dropna(), ADX_PERIOD)

    pdi    = 100 * pdm14 / atr14.replace(0, np.nan)
    ndi    = 100 * ndm14 / atr14.replace(0, np.nan)
    dx     = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    adx    = wilder_smooth(dx.dropna(), ADX_PERIOD)

    cur_adx  = float(adx.iloc[-1])   if not adx.empty  else 0.0
    cur_pdi  = float(pdi.iloc[-1])   if not pdi.empty  else 0.0
    cur_ndi  = float(ndi.iloc[-1])   if not ndi.empty  else 0.0

    # Trend score (0–100)
    t_score = 0
    t_score += ema_align_score * 12          # 0–48
    if cur_adx > 40:   t_score += 30
    elif cur_adx > 25: t_score += 20
    elif cur_adx > 15: t_score += 10
    if cur_pdi > cur_ndi: t_score += 22      # bullish directional bias
    t_score = min(t_score, 100)

    trend_str = ("Strong Trend" if cur_adx > 40 else
                 "Trending"     if cur_adx > 25 else
                 "Weak Trend"   if cur_adx > 15 else "Ranging")

    detail = (f"ADX: {cur_adx:.1f} ({trend_str}) | "
              f"+DI: {cur_pdi:.1f}  -DI: {cur_ndi:.1f} | "
              f"EMA align: {ema_align_score}/4 ({ema_align_label})")

    return dict(
        adx             = round(cur_adx, 2),
        plus_di         = round(cur_pdi, 2),
        minus_di        = round(cur_ndi, 2),
        ema_align_score = ema_align_score,
        ema_align_label = ema_align_label,
        ema8            = round(e8,   2),
        ema21           = round(e21,  2),
        ema55           = round(e55,  2),
        ema200          = round(e200, 2),
        trend_score     = t_score,
        trend_detail    = detail,
    )


# ──────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ──────────────────────────────────────────────────────────

def composite_score(s: dict) -> int:
    """
    Weighted composite institutional score (0–100):
      RS Rating     25 pts
      Stage         25 pts  (Stage2=25, Stage1=12, Stage3=6, Stage4=0)
      VCP           20 pts
      Accumulation  20 pts
      Trend (ADX)   10 pts
    """
    rs_pts    = int((s.get("rs_rating") or 0) / 99 * 25)
    stage_map = {2: 25, 1: 12, 3: 6, 4: 0, 0: 0}
    stage_pts = stage_map.get(s.get("stage", 0), 0)
    vcp_pts   = int((s.get("vcp_score") or 0) / 100 * 20)
    acc_pts   = int((s.get("accum_score") or 0) / 100 * 20)
    trend_pts = int((s.get("trend_score") or 0) / 100 * 10)
    return min(rs_pts + stage_pts + vcp_pts + acc_pts + trend_pts, 100)


# ──────────────────────────────────────────────────────────
# FULL STOCK ANALYSIS
# ──────────────────────────────────────────────────────────

def analyze_stock(csv_path: str) -> dict | None:
    df = load_csv(csv_path)
    if df is None:
        return None

    ticker    = Path(csv_path).stem.upper()
    cur_price = float(df["Close"].iloc[-1])
    last_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    # 52-week metrics
    today = df["Date"].iloc[-1]
    fy    = df[df["Date"] >= today - timedelta(days=365)]
    h52   = float(fy["High"].max())  if len(fy) > 0 else float(df["High"].max())
    l52   = float(fy["Low"].min())   if len(fy) > 0 else float(df["Low"].min())
    r52   = h52 - l52
    pos52 = (cur_price - l52) / r52 * 100 if r52 > 0 else 50.0

    # CAGR helpers
    def px_n_ago(days):
        s = df[df["Date"] <= today - timedelta(days=days)]
        return float(s["Close"].iloc[-1]) if len(s) >= 5 else None

    def cagr(p, yrs):
        return (cur_price / p) ** (1 / yrs) - 1 if p and p > 0 else None

    c1y = cagr(px_n_ago(252),  1)
    c3y = cagr(px_n_ago(756),  3)
    c5y = cagr(px_n_ago(1260), 5)

    # Volatility
    rets = df["Close"].pct_change().dropna()
    vol  = float(rets.tail(252).std() * math.sqrt(252)) if len(rets) >= 30 else None

    # Per-scanner metrics
    stage_res  = stage_analysis(df)
    vcp_res    = vcp_scan(df)
    accum_res  = accumulation_score(df)
    ema_res    = ema_adx_analysis(df)

    result = dict(
        ticker     = ticker,
        cur_price  = cur_price,
        last_date  = last_date,
        high_52w   = round(h52, 2),
        low_52w    = round(l52, 2),
        pos_52w    = round(pos52, 1),
        cagr_1y    = c1y,
        cagr_3y    = c3y,
        cagr_5y    = c5y,
        volatility = vol,
        rs_raw     = rs_raw_score(df),
        rs_rating  = None,   # filled after cross-sectional ranking
        **stage_res,
        **vcp_res,
        **accum_res,
        **ema_res,
    )
    return result


# ──────────────────────────────────────────────────────────
# HTML BUILDER
# ──────────────────────────────────────────────────────────

def ticker_link(t: str) -> str:
    return (f'<a href="{TV_BASE}{t}" target="_blank" rel="noopener noreferrer" '
            f'class="tlink">{t}</a>')

def pct_cell(v, good_pos=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '<span class="muted">—</span>', "null"
    pct = v * 100
    cls = ("pos" if pct > 0 else "neg") if good_pos else ("neg" if pct > 0 else "pos")
    return f'<span class="{cls}">{pct:+.1f}%</span>', f"{pct:.2f}"

def score_bar(score: int, color_fn=None) -> str:
    if score is None: return "—"
    w = min(max(score, 0), 100)
    if color_fn:
        col = color_fn(w)
    else:
        col = "#10b981" if w >= 70 else "#f59e0b" if w >= 40 else "#ef4444"
    return (f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:70px;height:6px;background:#1e2132;border-radius:3px">'
            f'<div style="width:{w}%;height:100%;background:{col};border-radius:3px"></div>'
            f'</div><span style="font-weight:600">{score}</span></div>')

def stage_badge(stage, label):
    colors = {
        2: ("background:#064e3b;color:#34d399", "▲ Stage 2"),
        1: ("background:#1e3a5f;color:#93c5fd", "● Stage 1"),
        3: ("background:#44370a;color:#fde68a", "◆ Stage 3"),
        4: ("background:#450a0a;color:#fca5a5", "▼ Stage 4"),
        0: ("background:#374151;color:#9ca3af", "N/A"),
    }
    style, text = colors.get(stage, colors[0])
    return f'<span style="padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;{style}">{label or text}</span>'

def accum_badge(label):
    m = {
        "Strong Accumulation": "background:#064e3b;color:#34d399",
        "Accumulation":        "background:#14532d;color:#86efac",
        "Neutral":             "background:#374151;color:#9ca3af",
        "Distribution":        "background:#7c2d12;color:#fdba74",
        "Strong Distribution": "background:#450a0a;color:#fca5a5",
    }
    style = m.get(label, m["Neutral"])
    return f'<span style="padding:2px 10px;border-radius:20px;font-size:11px;font-weight:700;{style}">{label}</span>'

def fmt_price(v):
    if v is None: return "—"
    return f"₹{v:,.2f}"

def rs_color(v):
    if v is None: return '<span class="muted">—</span>'
    cls = "great" if v >= 80 else "good" if v >= 60 else "ok" if v >= 40 else "bad"
    return f'<span class="{cls}" style="font-weight:700">{v}</span>'

def adx_color(v):
    if v is None: return '<span class="muted">—</span>'
    cls = "great" if v >= 40 else "good" if v >= 25 else "ok" if v >= 15 else "muted"
    return f'<span class="{cls}">{v:.1f}</span>'

def ema_align_badge(score, label):
    colors = {4: "#10b981", 3: "#6ee7b7", 2: "#fde68a", 1: "#fb923c", 0: "#f87171"}
    col = colors.get(score, "#9ca3af")
    return (f'<span style="color:{col};font-weight:600">'
            f'{"■"*score}{"□"*(4-score)} {label}</span>')


# ─────────── CSS + SHARED HTML ───────────

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600;700&display=swap');
:root{
  --bg:#0b0e18; --s1:#131724; --s2:#1a1f30; --s3:#222840;
  --border:#2a2f47; --text:#dde3f0; --muted:#5a6282;
  --accent:#7c6af7; --accent2:#5b4fe0;
  --green:#10b981; --red:#ef4444; --yellow:#f59e0b;
  --blue:#3b82f6; --cyan:#22d3ee;
}
*{box-sizing:border-box;margin:0;padding:0}
html{overflow-x:scroll;scrollbar-width:thin;scrollbar-color:#2a2f47 #0b0e18}
html::-webkit-scrollbar{height:10px}
html::-webkit-scrollbar-track{background:#0b0e18}
html::-webkit-scrollbar-thumb{background:#2a2f47;border-radius:4px}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
     font-size:13px;line-height:1.5;min-width:max-content}

/* ── HEADER ── */
.page-header{
  background:linear-gradient(135deg,#0f1220 0%,#1a1535 50%,#0f1220 100%);
  border-bottom:1px solid var(--border);
  padding:28px 36px 22px;
  position:relative; overflow:hidden;
}
.page-header::before{
  content:''; position:absolute; top:-60px; right:-60px;
  width:260px; height:260px; border-radius:50%;
  background:radial-gradient(circle,rgba(124,106,247,.15) 0%,transparent 70%);
}
.header-top{display:flex;align-items:flex-start;justify-content:space-between;gap:20px}
.header-title{display:flex;align-items:center;gap:12px}
.header-title h1{font-size:22px;font-weight:700;color:#fff;letter-spacing:-.3px}
.pill{display:inline-flex;align-items:center;padding:3px 11px;border-radius:20px;
      font-size:10px;font-weight:700;letter-spacing:.8px;text-transform:uppercase}
.pill-accent{background:rgba(124,106,247,.2);color:var(--accent);border:1px solid rgba(124,106,247,.3)}
.pill-green{background:rgba(16,185,129,.15);color:var(--green);border:1px solid rgba(16,185,129,.25)}
.header-meta{display:flex;flex-wrap:wrap;gap:20px;margin-top:16px}
.meta-box{display:flex;flex-direction:column}
.meta-box .ml{font-size:10px;text-transform:uppercase;letter-spacing:.7px;color:var(--muted)}
.meta-box .mv{font-size:16px;font-weight:700;color:#fff;margin-top:3px;
              font-family:'JetBrains Mono',monospace}

/* ── TABS ── */
.tab-bar{
  display:flex;gap:0;
  background:var(--s1);
  border-bottom:2px solid var(--border);
  padding:0 36px;
  overflow-x:auto;
  scrollbar-width:none;
}
.tab-bar::-webkit-scrollbar{display:none}
.tab-btn{
  padding:13px 22px;border:none;background:none;color:var(--muted);
  font-size:12px;font-weight:600;cursor:pointer;white-space:nowrap;
  border-bottom:2px solid transparent;margin-bottom:-2px;
  transition:all .2s;letter-spacing:.3px;text-transform:uppercase;
}
.tab-btn:hover{color:var(--text)}
.tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab-panel{display:none}
.tab-panel.active{display:block}

/* ── SUMMARY STRIP ── */
.summary-strip{
  display:flex;flex-wrap:wrap;gap:0;
  background:var(--s1);border-bottom:1px solid var(--border);
}
.sum-box{
  flex:1;min-width:130px;padding:14px 20px;
  border-right:1px solid var(--border);
}
.sum-box:last-child{border-right:none}
.sum-box .sl{font-size:9px;text-transform:uppercase;letter-spacing:.9px;color:var(--muted);margin-bottom:4px}
.sum-box .sv{font-size:20px;font-weight:700;font-family:'JetBrains Mono',monospace}

/* ── CONTROLS ── */
.controls{
  display:flex;flex-wrap:wrap;align-items:center;gap:10px;
  padding:12px 36px;
  background:var(--s2);border-bottom:1px solid var(--border);
}
.ctrl-search{
  background:var(--s3);border:1px solid var(--border);color:var(--text);
  padding:7px 13px;border-radius:6px;font-size:12px;width:180px;
  outline:none;font-family:'JetBrains Mono',monospace;
  transition:border .2s;
}
.ctrl-search:focus{border-color:var(--accent)}
.ctrl-btn{
  background:var(--s3);border:1px solid var(--border);color:var(--muted);
  padding:6px 14px;border-radius:6px;cursor:pointer;font-size:11px;
  font-weight:600;letter-spacing:.4px;transition:all .15s;
}
.ctrl-btn:hover,.ctrl-btn.on{border-color:var(--accent);color:var(--accent);background:rgba(124,106,247,.1)}
.ctrl-sep{width:1px;height:22px;background:var(--border)}
.ctrl-hint{color:var(--muted);font-size:11px;margin-left:auto}

/* ── TABLE ── */
.tbl-wrap{
  overflow-x:scroll;padding:0 36px 36px;
  scrollbar-width:auto;scrollbar-color:var(--accent) var(--s3);
}
.tbl-wrap::-webkit-scrollbar{height:10px}
.tbl-wrap::-webkit-scrollbar-track{background:var(--s3);border-radius:4px}
.tbl-wrap::-webkit-scrollbar-thumb{background:var(--accent);border-radius:4px}
.tbl-wrap::-webkit-scrollbar-thumb:hover{background:#9d8ff9}
table{width:max-content;min-width:100%;border-collapse:separate;
      border-spacing:0;margin-top:14px}
thead th{
  background:var(--s2);color:var(--muted);
  font-size:9px;text-transform:uppercase;letter-spacing:.9px;
  padding:10px 13px;white-space:nowrap;
  border-bottom:2px solid var(--border);
  cursor:pointer;user-select:none;
  position:sticky;top:0;z-index:10;
  transition:color .15s;font-family:'JetBrains Mono',monospace;
}
thead th:hover{color:var(--text)}
thead th.sa::after{content:" ▲";color:var(--accent)}
thead th.sd::after{content:" ▼";color:var(--accent)}
tbody tr{transition:background .1s}
tbody tr:hover{background:var(--s3)}
tbody td{padding:9px 13px;white-space:nowrap;border-bottom:1px solid #181d2e;
         font-family:'JetBrains Mono',monospace;font-size:12px}
tbody tr:last-child td{border-bottom:none}

/* ── COLOUR HELPERS ── */
.tlink{color:var(--accent);text-decoration:none;font-weight:700;
       border-bottom:1px dashed transparent;transition:all .15s}
.tlink:hover{color:#9d8ff9;border-bottom-color:#9d8ff9}
.pos{color:var(--green)}.neg{color:var(--red)}
.great{color:#34d399}.good{color:#86efac}.ok{color:#fde68a}.bad{color:#f87171}
.muted{color:var(--muted)}

/* ── RANGE BAR ── */
.rbar{display:flex;align-items:center;gap:6px;min-width:90px}
.rbar-track{flex:1;height:4px;background:var(--s3);border-radius:2px}
.rbar-fill{height:100%;background:var(--accent);border-radius:2px}
.rbar-pct{font-size:10px;color:var(--muted);width:32px;text-align:right}

/* ── FOOTER ── */
footer{text-align:center;padding:18px 36px;color:var(--muted);
       font-size:11px;border-top:1px solid var(--border);line-height:1.9}

.hidden{display:none!important}
</style>
"""

JS_COMMON = """
<script>
// ── Tab switching ──
function showTab(id, el) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  el.classList.add('active');
}

// ── Per-tab filter state ──
const tabFilters = {};
const tabSearch  = {};

function setFilter(tabId, f, btn) {
  tabFilters[tabId] = f;
  btn.closest('.controls').querySelectorAll('.ctrl-btn[data-filter]')
     .forEach(b => b.classList.toggle('on', b.dataset.filter === f));
  applyFilter(tabId);
}

function onSearch(tabId) {
  tabSearch[tabId] = document.getElementById('search_' + tabId).value.toUpperCase();
  applyFilter(tabId);
}

function applyFilter(tabId) {
  const f = tabFilters[tabId] || 'all';
  const q = tabSearch[tabId]  || '';
  document.querySelectorAll(`#${tabId} tbody tr`).forEach(r => {
    const ticker = r.cells[0]?.textContent.trim().toUpperCase() || '';
    const sig    = r.dataset.sig || '';
    const matchQ = !q || ticker.includes(q);
    const matchF = f === 'all' || sig === f;
    r.classList.toggle('hidden', !(matchQ && matchF));
  });
  updateCount(tabId);
}

function updateCount(tabId) {
  const vis = document.querySelectorAll(`#${tabId} tbody tr:not(.hidden)`).length;
  const el  = document.getElementById('cnt_' + tabId);
  if (el) el.textContent = vis;
}

// ── Sorting ──
const sortState = {};
function sortTab(tabId, col, type) {
  const key = tabId + '_' + col;
  if (sortState[key] === undefined) sortState[key] = -1;
  else sortState[key] *= -1;
  const dir   = sortState[key];
  const tbody = document.querySelector(`#${tabId} tbody`);
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  rows.sort((a, b) => {
    let av = a.cells[col]?.dataset.val ?? a.cells[col]?.textContent.trim() ?? '';
    let bv = b.cells[col]?.dataset.val ?? b.cells[col]?.textContent.trim() ?? '';
    if (type === 'num') {
      av = parseFloat(av); bv = parseFloat(bv);
      if (isNaN(av)) av = -Infinity; if (isNaN(bv)) bv = -Infinity;
      return (av - bv) * dir;
    }
    return av.localeCompare(bv) * dir;
  });
  rows.forEach(r => tbody.appendChild(r));
  document.querySelectorAll(`#${tabId} thead th`).forEach((th, i) => {
    th.classList.remove('sa','sd');
    if (i === col) th.classList.add(dir === 1 ? 'sa' : 'sd');
  });
}

// ── Stats badges ──
function buildStats(tabId, sigKey, countMap) {
  document.querySelectorAll(`#${tabId} tbody tr`).forEach(r => {
    const s = r.dataset[sigKey];
    if (s) countMap[s] = (countMap[s] || 0) + 1;
  });
  Object.entries(countMap).forEach(([k, v]) => {
    const el = document.getElementById(`${tabId}_${k}`);
    if (el) el.textContent = v;
  });
}

window.addEventListener('DOMContentLoaded', () => {
  // Default sort each tab by its primary score column (desc)
  const defaults = {
    'tab-overview': [3,'num'],
    'tab-rs':       [2,'num'],
    'tab-stage':    [3,'num'],
    'tab-vcp':      [3,'num'],
    'tab-accum':    [3,'num'],
    'tab-trend':    [3,'num'],
  };
  Object.entries(defaults).forEach(([t,[c,tp]]) => sortTab(t, c, tp));
  // Reset sort direction so next click toggles correctly
  Object.keys(defaults).forEach(t => {
    const [c] = defaults[t];
    const key = t + '_' + c;
    sortState[key] = -1;
  });
});
</script>
"""


def controls_html(tab_id: str, filter_btns: list[tuple]) -> str:
    btns = ''.join(
        f'<button class="ctrl-btn{" on" if i == 0 else ""}" '
        f'data-filter="{fval}" '
        f'onclick="setFilter(\'{tab_id}\',\'{fval}\',this)">{flabel}</button>'
        for i, (fval, flabel) in enumerate(filter_btns)
    )
    return f"""
<div class="controls">
  <input class="ctrl-search" id="search_{tab_id}" placeholder="🔍 Search ticker…"
         oninput="onSearch('{tab_id}')">
  {btns}
  <span class="ctrl-hint">Click headers to sort &nbsp;|&nbsp;
    Showing <b id="cnt_{tab_id}">—</b> stocks</span>
</div>"""


def summary_strip(boxes: list[tuple]) -> str:
    inner = ""
    for label, val, color in boxes:
        inner += (f'<div class="sum-box">'
                  f'<div class="sl">{label}</div>'
                  f'<div class="sv" style="color:{color}">{val}</div>'
                  f'</div>')
    return f'<div class="summary-strip">{inner}</div>'


# ──────────────────────────────────────────────────────────
# TAB BUILDERS
# ──────────────────────────────────────────────────────────

def overview_tab(stocks):
    rows = ""
    for s in stocks:
        comp   = s.get("composite_score", 0) or 0
        sig    = ("elite" if comp >= 80 else "strong" if comp >= 60 else
                  "watch" if comp >= 40 else "weak")
        c1h, c1d = pct_cell(s.get("cagr_1y"))
        c3h, c3d = pct_cell(s.get("cagr_3y"))
        stage   = s.get("stage", 0)
        pos     = min(max(s.get("pos_52w") or 50, 0), 100)
        bar     = (f'<div class="rbar"><div class="rbar-track">'
                   f'<div class="rbar-fill" style="width:{pos:.0f}%"></div>'
                   f'</div><span class="rbar-pct">{pos:.0f}%</span></div>')
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{comp}">{score_bar(comp)}</td>'
            f'<td data-val="{s.get("rs_rating") or 0}">{rs_color(s.get("rs_rating"))}</td>'
            f'<td data-val="{stage}">{stage_badge(stage, s.get("stage_label",""))}</td>'
            f'<td data-val="{s.get("vcp_score") or 0}">{score_bar(s.get("vcp_score") or 0)}</td>'
            f'<td data-val="{s.get("accum_score") or 0}">{score_bar(s.get("accum_score") or 0)}</td>'
            f'<td data-val="{s.get("trend_score") or 0}">{score_bar(s.get("trend_score") or 0)}</td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{c1d}">{c1h}</td>'
            f'<td data-val="{c3d}">{c3h}</td>'
            f'<td data-val="{pos:.1f}">{bar}</td>'
            f'<td data-val="{s.get("adx") or 0}">{adx_color(s.get("adx"))}</td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("elite","Elite ≥80"),("strong","Strong ≥60"),
               ("watch","Watch ≥40"),("weak","Weak <40")]

    # summary counts
    counts = {"elite":0,"strong":0,"watch":0,"weak":0}
    for s in stocks:
        c = s.get("composite_score",0) or 0
        k = "elite" if c>=80 else "strong" if c>=60 else "watch" if c>=40 else "weak"
        counts[k] += 1

    strip = summary_strip([
        ("Total Stocks",     len(stocks),       "#fff"),
        ("Elite (≥80)",      counts["elite"],   "#34d399"),
        ("Strong (≥60)",     counts["strong"],  "#86efac"),
        ("Watch (≥40)",      counts["watch"],   "#fde68a"),
        ("Weak (<40)",       counts["weak"],    "#f87171"),
    ])

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-overview',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-overview',1,'num')">Composite</th>
  <th onclick="sortTab('tab-overview',2,'num')">RS Rating</th>
  <th onclick="sortTab('tab-overview',3,'num')">Stage</th>
  <th onclick="sortTab('tab-overview',4,'num')">VCP</th>
  <th onclick="sortTab('tab-overview',5,'num')">Accumulation</th>
  <th onclick="sortTab('tab-overview',6,'num')">Trend</th>
  <th onclick="sortTab('tab-overview',7,'num')">Price</th>
  <th onclick="sortTab('tab-overview',8,'num')">1Y CAGR</th>
  <th onclick="sortTab('tab-overview',9,'num')">3Y CAGR</th>
  <th onclick="sortTab('tab-overview',10,'num')">52W Pos</th>
  <th onclick="sortTab('tab-overview',11,'num')">ADX</th>
  <th onclick="sortTab('tab-overview',12,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    return (strip + controls_html("tab-overview", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


def rs_tab(stocks):
    rows = ""
    for s in stocks:
        rs   = s.get("rs_rating")
        sig  = ("rs-elite" if (rs or 0) >= 80 else "rs-strong" if (rs or 0) >= 60 else
                "rs-avg"   if (rs or 0) >= 40 else "rs-weak")
        c1h, c1d = pct_cell(s.get("cagr_1y"))
        c3h, c3d = pct_cell(s.get("cagr_3y"))
        c5h, c5d = pct_cell(s.get("cagr_5y"))
        raw  = s.get("rs_raw")
        raw_s = f'<span class="{"pos" if (raw or 0)>0 else "neg"}">{(raw or 0)*100:+.1f}%</span>'
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{rs or 0}">{rs_color(rs)}</td>'
            f'<td data-val="{(raw or 0)*100:.2f}">{raw_s}</td>'
            f'<td data-val="{c1d}">{c1h}</td>'
            f'<td data-val="{c3d}">{c3h}</td>'
            f'<td data-val="{c5d}">{c5h}</td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{s.get("high_52w") or 0}">{fmt_price(s.get("high_52w"))}</td>'
            f'<td data-val="{s.get("low_52w") or 0}">{fmt_price(s.get("low_52w"))}</td>'
            f'<td data-val="{s.get("pos_52w") or 0}">{(s.get("pos_52w") or 0):.1f}%</td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("rs-elite","RS ≥80"),("rs-strong","RS 60–79"),
               ("rs-avg","RS 40–59"),("rs-weak","RS <40")]

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-rs',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-rs',1,'num')">RS Rating (1–99)</th>
  <th onclick="sortTab('tab-rs',2,'num')">Weighted 12M Return</th>
  <th onclick="sortTab('tab-rs',3,'num')">1Y CAGR</th>
  <th onclick="sortTab('tab-rs',4,'num')">3Y CAGR</th>
  <th onclick="sortTab('tab-rs',5,'num')">5Y CAGR</th>
  <th onclick="sortTab('tab-rs',6,'num')">Price</th>
  <th onclick="sortTab('tab-rs',7,'num')">52W High</th>
  <th onclick="sortTab('tab-rs',8,'num')">52W Low</th>
  <th onclick="sortTab('tab-rs',9,'num')">52W Position</th>
  <th onclick="sortTab('tab-rs',10,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    counts = {k:0 for k in ["rs-elite","rs-strong","rs-avg","rs-weak"]}
    for s in stocks:
        r = s.get("rs_rating") or 0
        k = "rs-elite" if r>=80 else "rs-strong" if r>=60 else "rs-avg" if r>=40 else "rs-weak"
        counts[k] += 1

    strip = summary_strip([
        ("RS ≥ 80 (Elite)",   counts["rs-elite"],  "#34d399"),
        ("RS 60–79",          counts["rs-strong"], "#86efac"),
        ("RS 40–59",          counts["rs-avg"],    "#fde68a"),
        ("RS < 40",           counts["rs-weak"],   "#f87171"),
    ])
    return (strip + controls_html("tab-rs", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


def stage_tab(stocks):
    rows = ""
    for s in stocks:
        stage = s.get("stage", 0)
        sig   = f"st{stage}"
        pma, pmad = (f'<span class="{"pos" if (s.get("pct_from_ma") or 0)>=0 else "neg"}">'
                     f'{s.get("pct_from_ma") or 0:+.1f}%</span>',
                     f'{s.get("pct_from_ma") or 0:.2f}')
        slope = s.get("ma150_slope") or 0
        slope_s = f'<span class="{"pos" if slope>=0 else "neg"}">{slope:+.3f}%</span>'
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{stage}">{stage_badge(stage, s.get("stage_label",""))}</td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{pmad}">{pma}</td>'
            f'<td data-val="{s.get("ma150") or 0}">{fmt_price(s.get("ma150"))}</td>'
            f'<td data-val="{s.get("ma50") or 0}">{fmt_price(s.get("ma50"))}</td>'
            f'<td data-val="{slope:.3f}">{slope_s}</td>'
            f'<td data-val="{s.get("rs_rating") or 0}">{rs_color(s.get("rs_rating"))}</td>'
            f'<td><span class="muted" style="font-size:10px">{s.get("stage_detail","")[:60]}</span></td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("st2","Stage 2"),("st1","Stage 1"),
               ("st3","Stage 3"),("st4","Stage 4")]

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-stage',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-stage',1,'num')">Stage</th>
  <th onclick="sortTab('tab-stage',2,'num')">Price</th>
  <th onclick="sortTab('tab-stage',3,'num')">% From MA150</th>
  <th onclick="sortTab('tab-stage',4,'num')">MA150</th>
  <th onclick="sortTab('tab-stage',5,'num')">MA50</th>
  <th onclick="sortTab('tab-stage',6,'num')">MA150 Slope/20d</th>
  <th onclick="sortTab('tab-stage',7,'num')">RS Rating</th>
  <th onclick="sortTab('tab-stage',8,'str')">Detail</th>
  <th onclick="sortTab('tab-stage',9,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    counts = {f"st{i}":0 for i in range(5)}
    for s in stocks: counts[f'st{s.get("stage",0)}'] = counts.get(f'st{s.get("stage",0)}',0)+1

    strip = summary_strip([
        ("Stage 2 ▲ (Long)",  counts["st2"], "#34d399"),
        ("Stage 1 ● (Base)",  counts["st1"], "#93c5fd"),
        ("Stage 3 ◆ (Top)",   counts["st3"], "#fde68a"),
        ("Stage 4 ▼ (Avoid)", counts["st4"], "#f87171"),
    ])
    return (strip + controls_html("tab-stage", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


def vcp_tab(stocks):
    rows = ""
    for s in stocks:
        vs  = s.get("vcp_score", 0) or 0
        sig = ("vcp-hot" if vs >= 70 else "vcp-ok" if vs >= 40 else "vcp-weak")
        vc  = s.get("vcp_contractions", 0)
        vt  = s.get("vcp_tightness")
        vp  = s.get("vcp_pivot")
        vpct = s.get("vcp_pct_pivot")
        tight_s = f"{vt:.1f}%" if vt is not None else "—"
        pivot_s = fmt_price(vp)
        ppct_s  = (f'<span class="{"pos" if (vpct or 0)>=0 else "neg"}">{(vpct or 0):+.1f}%</span>'
                   if vpct is not None else "—")
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{vs}">{score_bar(vs)}</td>'
            f'<td data-val="{vc}">{vc}</td>'
            f'<td data-val="{vt or 0}">{tight_s}</td>'
            f'<td data-val="{vp or 0}">{pivot_s}</td>'
            f'<td data-val="{vpct or -999}">{ppct_s}</td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{s.get("rs_rating") or 0}">{rs_color(s.get("rs_rating"))}</td>'
            f'<td data-val="{s.get("stage",0)}">{stage_badge(s.get("stage",0), s.get("stage_label",""))}</td>'
            f'<td><span class="muted" style="font-size:10px">{s.get("vcp_detail","")[:65]}</span></td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("vcp-hot","VCP ≥70"),("vcp-ok","VCP 40–69"),("vcp-weak","VCP <40")]

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-vcp',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-vcp',1,'num')">VCP Score</th>
  <th onclick="sortTab('tab-vcp',2,'num')">Contractions</th>
  <th onclick="sortTab('tab-vcp',3,'num')">Latest Depth %</th>
  <th onclick="sortTab('tab-vcp',4,'num')">Pivot High</th>
  <th onclick="sortTab('tab-vcp',5,'num')">Price vs Pivot</th>
  <th onclick="sortTab('tab-vcp',6,'num')">Current Price</th>
  <th onclick="sortTab('tab-vcp',7,'num')">RS Rating</th>
  <th onclick="sortTab('tab-vcp',8,'num')">Stage</th>
  <th onclick="sortTab('tab-vcp',9,'str')">Detail</th>
  <th onclick="sortTab('tab-vcp',10,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    hot  = sum(1 for s in stocks if (s.get("vcp_score") or 0) >= 70)
    ok   = sum(1 for s in stocks if 40 <= (s.get("vcp_score") or 0) < 70)
    weak = sum(1 for s in stocks if (s.get("vcp_score") or 0) < 40)
    strip = summary_strip([
        ("Hot Setup (≥70)",  hot,  "#34d399"),
        ("Forming (40–69)",  ok,   "#fde68a"),
        ("Weak / None (<40)",weak, "#f87171"),
    ])
    return (strip + controls_html("tab-vcp", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


def accum_tab(stocks):
    rows = ""
    for s in stocks:
        sc   = s.get("accum_score", 0) or 0
        lbl  = s.get("accum_label", "Neutral")
        sig  = lbl.lower().replace(" ", "-")
        uvr  = s.get("up_vol_ratio")
        cmf  = s.get("cmf")
        pd_  = s.get("power_days", 0)
        wd_  = s.get("weak_days", 0)
        uvr_s = f"{(uvr or 0):.2f}"
        cmf_s = (f'<span class="{"pos" if (cmf or 0)>=0 else "neg"}">{(cmf or 0):+.3f}</span>'
                 if cmf is not None else "—")
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{sc}">{score_bar(sc)}</td>'
            f'<td data-val="{sc}">{accum_badge(lbl)}</td>'
            f'<td data-val="{uvr or 0}">{uvr_s}</td>'
            f'<td data-val="{(cmf or 0):.4f}">{cmf_s}</td>'
            f'<td data-val="{pd_}"><span class="pos">{pd_}</span></td>'
            f'<td data-val="{wd_}"><span class="neg">{wd_}</span></td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{s.get("rs_rating") or 0}">{rs_color(s.get("rs_rating"))}</td>'
            f'<td><span class="muted" style="font-size:10px">{s.get("accum_detail","")[:65]}</span></td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("strong-accumulation","Strong Accum"),
               ("accumulation","Accumulation"),("neutral","Neutral"),
               ("distribution","Distribution"),("strong-distribution","Strong Dist")]

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-accum',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-accum',1,'num')">Accum Score</th>
  <th onclick="sortTab('tab-accum',2,'num')">Label</th>
  <th onclick="sortTab('tab-accum',3,'num')">Up-Vol Ratio</th>
  <th onclick="sortTab('tab-accum',4,'num')">CMF (21d)</th>
  <th onclick="sortTab('tab-accum',5,'num')">Power Days</th>
  <th onclick="sortTab('tab-accum',6,'num')">Weak Days</th>
  <th onclick="sortTab('tab-accum',7,'num')">Price</th>
  <th onclick="sortTab('tab-accum',8,'num')">RS Rating</th>
  <th onclick="sortTab('tab-accum',9,'str')">Detail</th>
  <th onclick="sortTab('tab-accum',10,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    sa  = sum(1 for s in stocks if (s.get("accum_score") or 0) >= 70)
    a   = sum(1 for s in stocks if 50 <= (s.get("accum_score") or 0) < 70)
    n   = sum(1 for s in stocks if 35 <= (s.get("accum_score") or 0) < 50)
    d   = sum(1 for s in stocks if (s.get("accum_score") or 0) < 35)
    strip = summary_strip([
        ("Strong Accum (≥70)", sa, "#34d399"),
        ("Accumulation (50–69)", a, "#86efac"),
        ("Neutral (35–49)",    n, "#fde68a"),
        ("Distribution (<35)", d, "#f87171"),
    ])
    return (strip + controls_html("tab-accum", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


def trend_tab(stocks):
    rows = ""
    for s in stocks:
        ts   = s.get("trend_score", 0) or 0
        adx  = s.get("adx")
        pdi  = s.get("plus_di")
        ndi  = s.get("minus_di")
        eas  = s.get("ema_align_score", 0)
        eal  = s.get("ema_align_label", "N/A")
        e8   = s.get("ema8")
        e21  = s.get("ema21")
        e55  = s.get("ema55")
        e200 = s.get("ema200")
        sig  = ("bull4" if eas==4 else "bull3" if eas==3 else
                "neutral2" if eas==2 else "bear1" if eas==1 else "bear0")

        pdi_s = f'<span class="pos">{pdi:.1f}</span>' if pdi else "—"
        ndi_s = f'<span class="neg">{ndi:.1f}</span>' if ndi else "—"
        rows += (
            f'<tr data-sig="{sig}">'
            f'<td>{ticker_link(s["ticker"])}</td>'
            f'<td data-val="{ts}">{score_bar(ts)}</td>'
            f'<td data-val="{adx or 0}">{adx_color(adx)}</td>'
            f'<td data-val="{eas}">{ema_align_badge(eas, eal)}</td>'
            f'<td data-val="{pdi or 0}">{pdi_s}</td>'
            f'<td data-val="{ndi or 0}">{ndi_s}</td>'
            f'<td data-val="{e8 or 0}">{fmt_price(e8)}</td>'
            f'<td data-val="{e21 or 0}">{fmt_price(e21)}</td>'
            f'<td data-val="{e55 or 0}">{fmt_price(e55)}</td>'
            f'<td data-val="{e200 or 0}">{fmt_price(e200)}</td>'
            f'<td data-val="{s["cur_price"]:.2f}">{fmt_price(s["cur_price"])}</td>'
            f'<td data-val="{s.get("rs_rating") or 0}">{rs_color(s.get("rs_rating"))}</td>'
            f'<td><span class="muted" style="font-size:10px">{s.get("trend_detail","")[:60]}</span></td>'
            f'<td>{s["last_date"]}</td>'
            f'</tr>\n'
        )

    filters = [("all","All"),("bull4","Full Bull (4/4)"),("bull3","Weak Bull (3/4)"),
               ("neutral2","Neutral (2/4)"),("bear1","Weak Bear (1/4)"),("bear0","Full Bear (0/4)")]

    table = f"""
<thead><tr>
  <th onclick="sortTab('tab-trend',0,'str')">Ticker</th>
  <th onclick="sortTab('tab-trend',1,'num')">Trend Score</th>
  <th onclick="sortTab('tab-trend',2,'num')">ADX (14)</th>
  <th onclick="sortTab('tab-trend',3,'num')">EMA Alignment</th>
  <th onclick="sortTab('tab-trend',4,'num')">+DI</th>
  <th onclick="sortTab('tab-trend',5,'num')">-DI</th>
  <th onclick="sortTab('tab-trend',6,'num')">EMA 8</th>
  <th onclick="sortTab('tab-trend',7,'num')">EMA 21</th>
  <th onclick="sortTab('tab-trend',8,'num')">EMA 55</th>
  <th onclick="sortTab('tab-trend',9,'num')">EMA 200</th>
  <th onclick="sortTab('tab-trend',10,'num')">Price</th>
  <th onclick="sortTab('tab-trend',11,'num')">RS Rating</th>
  <th onclick="sortTab('tab-trend',12,'str')">Detail</th>
  <th onclick="sortTab('tab-trend',13,'str')">Last Date</th>
</tr></thead>
<tbody>{rows}</tbody>"""

    bull4 = sum(1 for s in stocks if s.get("ema_align_score")==4)
    bull3 = sum(1 for s in stocks if s.get("ema_align_score")==3)
    neut  = sum(1 for s in stocks if s.get("ema_align_score")==2)
    bear  = sum(1 for s in stocks if (s.get("ema_align_score") or 0) <= 1)
    strip = summary_strip([
        ("Full Bull (4/4)",   bull4, "#34d399"),
        ("Weak Bull (3/4)",   bull3, "#86efac"),
        ("Neutral (2/4)",     neut,  "#fde68a"),
        ("Bear (0–1/4)",      bear,  "#f87171"),
        ("ADX>25 Trending",   sum(1 for s in stocks if (s.get("adx") or 0)>25), "#22d3ee"),
    ])
    return (strip + controls_html("tab-trend", filters) +
            f'<div class="tbl-wrap"><table>{table}</table></div>')


# ──────────────────────────────────────────────────────────
# FULL PAGE
# ──────────────────────────────────────────────────────────

def generate_page(stocks: list[dict], title: str, out_path: str, gen_date: str):
    ov   = overview_tab(stocks)
    rs   = rs_tab(stocks)
    st   = stage_tab(stocks)
    vc   = vcp_tab(stocks)
    ac   = accum_tab(stocks)
    tr   = trend_tab(stocks)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{title}</title>
{PAGE_CSS}
</head>
<body>

<div class="page-header">
  <div class="header-top">
    <div class="header-title">
      <h1>📡 {title}</h1>
      <span class="pill pill-accent">Institutional Scanner</span>
      <span class="pill pill-green">OHLCV Only</span>
    </div>
  </div>
  <div class="header-meta">
    <div class="meta-box"><span class="ml">Generated</span><span class="mv">{gen_date}</span></div>
    <div class="meta-box"><span class="ml">Stocks</span><span class="mv">{len(stocks):,}</span></div>
    <div class="meta-box"><span class="ml">RS Rating</span><span class="mv">IBD Weighted 12M</span></div>
    <div class="meta-box"><span class="ml">Stage MA</span><span class="mv">150d / 50d</span></div>
    <div class="meta-box"><span class="ml">ADX Period</span><span class="mv">{ADX_PERIOD}d Wilder</span></div>
    <div class="meta-box"><span class="ml">Accum Window</span><span class="mv">{ACCUM_PERIOD}d</span></div>
    <div class="meta-box"><span class="ml">EMA Ribbon</span><span class="mv">8 · 21 · 55 · 200</span></div>
  </div>
</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="showTab('tab-overview',this)">📊 Overview</button>
  <button class="tab-btn"        onclick="showTab('tab-rs',this)">📈 RS Rating</button>
  <button class="tab-btn"        onclick="showTab('tab-stage',this)">🏗 Stage Analysis</button>
  <button class="tab-btn"        onclick="showTab('tab-vcp',this)">🔍 VCP Setups</button>
  <button class="tab-btn"        onclick="showTab('tab-accum',this)">💰 Accumulation</button>
  <button class="tab-btn"        onclick="showTab('tab-trend',this)">📉 EMA / ADX</button>
</div>

<div id="tab-overview" class="tab-panel active">{ov}</div>
<div id="tab-rs"       class="tab-panel">{rs}</div>
<div id="tab-stage"    class="tab-panel">{st}</div>
<div id="tab-vcp"      class="tab-panel">{vc}</div>
<div id="tab-accum"    class="tab-panel">{ac}</div>
<div id="tab-trend"    class="tab-panel">{tr}</div>

<footer>
  ⚠️ <strong>Educational / Research Use Only.</strong>
  All metrics derived from OHLCV price &amp; volume data.
  RS Rating = IBD-style weighted 12-month return percentile rank.
  Stage = Weinstein 150d/50d MA methodology.
  VCP = Minervini Volatility Contraction Pattern.
  Accumulation = Up-volume ratio + OBV + CMF + power-day analysis.
  EMA/ADX = 8/21/55/200 EMA ribbon alignment + Wilder ADX(14).<br>
  <strong>Not investment advice. Always conduct your own due diligence.</strong>
</footer>

{JS_COMMON}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓ {out_path}  ({len(stocks):,} stocks, "
          f"{os.path.getsize(out_path)/1024:.0f} KB)")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        csv_files = glob.glob(os.path.join(DATA_DIR, "**", "*.csv"), recursive=True)
    if not csv_files:
        print(f"ERROR: No CSV files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(csv_files):,} CSV files")

    nifty750 = set()
    if os.path.exists(NIFTY750_FILE):
        with open(NIFTY750_FILE) as f:
            nifty750 = {l.strip().upper() for l in f if l.strip()}
        print(f"Loaded {len(nifty750):,} tickers from nifty750.txt")

    # ── Process all stocks ──
    all_stocks = []
    errors = 0
    for i, path in enumerate(sorted(csv_files), 1):
        r = analyze_stock(path)
        if r:
            all_stocks.append(r)
        else:
            errors += 1
        if i % 100 == 0:
            print(f"  {i:,}/{len(csv_files):,} processed  ({time.time()-t0:.1f}s)")

    print(f"\nProcessed: {len(all_stocks):,} ok  |  Skipped: {errors}")

    # ── Cross-sectional RS rating (needs all stocks) ──
    print("Computing RS ratings…")
    assign_rs_ratings(all_stocks)

    # ── Composite score ──
    for s in all_stocks:
        s["composite_score"] = composite_score(s)

    # ── Split into nifty750 vs others ──
    n750   = [s for s in all_stocks if s["ticker"] in nifty750]
    others = [s for s in all_stocks if s["ticker"] not in nifty750]
    print(f"Nifty750: {len(n750)}  |  Others: {len(others)}")

    # ── Generate HTML ──
    gen_date = datetime.now().strftime("%d %b %Y, %H:%M")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("\nGenerating HTML…")
    generate_page(n750,   "Nifty 750 — Institutional Scanner",
                  os.path.join(OUTPUT_DIR, "nifty750_scanner.html"),   gen_date)
    generate_page(others, "Other NSE Stocks — Institutional Scanner",
                  os.path.join(OUTPUT_DIR, "other_stocks_scanner.html"), gen_date)

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
