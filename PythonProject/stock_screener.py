"""
Institutional Price Action Strategy Screener
============================================
Signals:
  1. Demand Zone Accumulation  - Zones where institutions accumulated
  2. Supply Zone Distribution   - Zones where institutions distributed
  3. Price & Volume Confirmation - Volume-confirmed momentum signals
"""

import os
import glob
import json
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data_cache")
NIFTY750_FILE   = os.path.join(BASE_DIR, "nifty750.txt")
OUT_NIFTY750    = os.path.join(BASE_DIR, "nifty750_signals.html")
OUT_OTHERS      = os.path.join(BASE_DIR, "other_stocks_signals.html")

LOOKBACK        = 252          # 1 year of data used for zone detection
MIN_ROWS        = 60           # minimum candles required
ZONE_TOLERANCE  = 0.015        # ±1.5% tolerance around zones
SWING_WINDOW    = 10           # bars each side for swing detection
VOL_MA_PERIOD   = 20
RSI_PERIOD      = 14
SMA_SHORT       = 20
SMA_MED         = 50
SMA_LONG        = 200

# Exchange prefixes and series suffixes to strip during normalisation
_EXCHANGE_PREFIXES = ("BSE_", "NSE_", "BSE-", "NSE-")
_SERIES_SUFFIXES   = (
    '-A', '-B', '-T',
    '-X', '-XT', '-Z', '-ZP',
    '-M', '-MT', '-MS', '-P',
    '-B1', '-IF',
    '-E'  # ETFs
    # yfinance-style suffixes
)


# ─────────────────────────────────────────────
# TICKER NORMALISATION
# ─────────────────────────────────────────────

def normalise_ticker(raw: str) -> str:
    """
    Convert any filename / list entry into a bare uppercase ticker symbol.

    Examples
    --------
    BSE_AARTIIND-A.csv  →  AARTIIND
    NSE_RELIANCE-EQ     →  RELIANCE
    HDFCBANK.NS         →  HDFCBANK
    INFY                →  INFY
    AARTIIND            →  AARTIIND
    """
    s = raw.strip().upper()

    # Strip exchange prefix (case-insensitive already upper-cased)
    for pfx in _EXCHANGE_PREFIXES:
        if s.startswith(pfx):
            s = s[len(pfx):]
            break

    # Strip known series / exchange suffixes
    for sfx in _SERIES_SUFFIXES:
        if s.endswith(sfx):
            s = s[: -len(sfx)]
            break

    return s


def extract_display_name(filename: str) -> str:
    """Return a clean display ticker from a CSV filename like BSE_AARTIIND-A.csv"""
    stem = Path(filename).stem          # BSE_AARTIIND-A
    return normalise_ticker(stem)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_nifty750():
    """
    Load nifty750.txt and return a set of *normalised* ticker symbols so they
    can be matched against normalised CSV filenames regardless of exchange
    prefix or series suffix differences.
    """
    if not os.path.exists(NIFTY750_FILE):
        print(f"  ⚠  {NIFTY750_FILE} not found — all stocks will go to 'Other'")
        return set()
    tickers = set()
    with open(NIFTY750_FILE) as f:
        for line in f:
            raw = line.strip()
            if raw and not raw.startswith("#"):
                tickers.add(normalise_ticker(raw))
    return tickers


def load_csv(path):
    try:
        df = pd.read_csv(path, parse_dates=["Datetime"])
        df.columns = [c.strip() for c in df.columns]
        df = df.sort_values("Datetime").reset_index(drop=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        df["Volume"] = df["Volume"].fillna(0)
        return df
    except Exception:
        return None


def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df, period=14):
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─────────────────────────────────────────────
# ZONE DETECTION
# ─────────────────────────────────────────────

def find_swing_lows(df, window=SWING_WINDOW):
    lows = []
    for i in range(window, len(df) - window):
        if df["Low"].iloc[i] == df["Low"].iloc[i-window:i+window+1].min():
            lows.append(i)
    return lows


def find_swing_highs(df, window=SWING_WINDOW):
    highs = []
    for i in range(window, len(df) - window):
        if df["High"].iloc[i] == df["High"].iloc[i-window:i+window+1].max():
            highs.append(i)
    return highs


def build_demand_zones(df, swing_lows, vol_ma):
    """
    Demand Zone = price band around a swing low where:
      - Volume spike (≥1.3× avg) confirms institutional buying
      - Price bounced ≥2% from the low
      - Zone hasn't been breached more than once since formation
    Returns list of (zone_low, zone_high, strength) sorted by recency.
    """
    zones = []
    for idx in swing_lows:
        row     = df.iloc[idx]
        avg_vol = vol_ma.iloc[idx]
        vol_ok  = row["Volume"] > avg_vol * 1.3 if avg_vol > 0 else False

        # Price bounce check: next 10 candles must recover ≥2%
        future  = df.iloc[idx+1 : idx+11]
        bounce  = (future["Close"].max() - row["Low"]) / row["Low"] if len(future) > 0 else 0

        if bounce >= 0.02:
            zone_low  = row["Low"] * (1 - ZONE_TOLERANCE)
            zone_high = row["Low"] * (1 + ZONE_TOLERANCE * 2)

            # Count how many times price dipped into zone (fresh = stronger)
            revisits = ((df["Low"] >= zone_low) & (df["Low"] <= zone_high)).sum()

            # Volume strength multiplier
            vol_score = min(row["Volume"] / avg_vol, 3.0) if avg_vol > 0 else 1.0

            strength = (bounce * 50) + (vol_score * 15) - (revisits * 5)
            strength = max(0, min(100, strength))

            zones.append({
                "zone_low":  round(zone_low,  2),
                "zone_high": round(zone_high, 2),
                "pivot":     round(row["Low"], 2),
                "date":      row["Datetime"],
                "vol_ok":    vol_ok,
                "bounce":    round(bounce * 100, 2),
                "revisits":  revisits,
                "strength":  round(strength, 1),
                "idx":       idx
            })

    # Keep freshest & strongest zones (max 5)
    zones.sort(key=lambda z: (z["idx"], z["strength"]), reverse=True)
    return zones[:5]


def build_supply_zones(df, swing_highs, vol_ma):
    """
    Supply Zone = price band around a swing high where:
      - Volume spike confirms institutional distribution
      - Price dropped ≥2% from the high
    """
    zones = []
    for idx in swing_highs:
        row     = df.iloc[idx]
        avg_vol = vol_ma.iloc[idx]

        future  = df.iloc[idx+1 : idx+11]
        drop    = (row["High"] - future["Close"].min()) / row["High"] if len(future) > 0 else 0

        if drop >= 0.02:
            zone_low  = row["High"] * (1 - ZONE_TOLERANCE * 2)
            zone_high = row["High"] * (1 + ZONE_TOLERANCE)
            revisits  = ((df["High"] >= zone_low) & (df["High"] <= zone_high)).sum()

            vol_score = min(row["Volume"] / avg_vol, 3.0) if avg_vol > 0 else 1.0
            strength  = (drop * 50) + (vol_score * 15) - (revisits * 5)
            strength  = max(0, min(100, strength))

            zones.append({
                "zone_low":  round(zone_low,  2),
                "zone_high": round(zone_high, 2),
                "pivot":     round(row["High"], 2),
                "date":      row["Datetime"],
                "vol_ok":    row["Volume"] > avg_vol * 1.3 if avg_vol > 0 else False,
                "drop":      round(drop * 100, 2),
                "revisits":  revisits,
                "strength":  round(strength, 1),
                "idx":       idx
            })

    zones.sort(key=lambda z: (z["idx"], z["strength"]), reverse=True)
    return zones[:5]


# ─────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────

def analyse(df):
    if len(df) < MIN_ROWS:
        return None

    df = df.tail(LOOKBACK).reset_index(drop=True).copy()

    # ── Indicators ────────────────────────────
    df["vol_ma"]  = df["Volume"].rolling(VOL_MA_PERIOD).mean()
    df["sma20"]   = df["Close"].rolling(SMA_SHORT).mean()
    df["sma50"]   = df["Close"].rolling(SMA_MED).mean()
    df["sma200"]  = df["Close"].rolling(SMA_LONG).mean()
    df["rsi"]     = rsi(df["Close"], RSI_PERIOD)
    df["atr"]     = atr(df)

    last      = df.iloc[-1]
    cmp       = last["Close"]
    vol_now   = last["Volume"]
    vol_ma_v  = last["vol_ma"]
    vol_ratio = round(vol_now / vol_ma_v, 2) if vol_ma_v > 0 else 0

    rsi_val   = round(last["rsi"], 1) if not np.isnan(last["rsi"]) else None
    sma20_v   = last["sma20"]
    sma50_v   = last["sma50"]
    sma200_v  = last["sma200"]
    atr_v     = last["atr"]

    # ── Trend Structure ───────────────────────
    above_20  = cmp > sma20_v  if not np.isnan(sma20_v)  else False
    above_50  = cmp > sma50_v  if not np.isnan(sma50_v)  else False
    above_200 = cmp > sma200_v if not np.isnan(sma200_v) else False

    sma_align = sum([above_20, above_50, above_200])
    trend_tag = {3: "Strong Uptrend", 2: "Uptrend", 1: "Mixed", 0: "Downtrend"}[sma_align]

    # ── Swing Detection ───────────────────────
    swing_lows  = find_swing_lows(df)
    swing_highs = find_swing_highs(df)

    demand_zones = build_demand_zones(df, swing_lows,  df["vol_ma"])
    supply_zones = build_supply_zones(df, swing_highs, df["vol_ma"])

    # ── Best nearby demand zone ───────────────
    best_demand = None
    in_demand   = False
    dist_demand = None
    for z in demand_zones:
        dist = (cmp - z["zone_high"]) / cmp * 100
        if z["zone_low"] <= cmp <= z["zone_high"] * 1.03:
            in_demand = True
            if best_demand is None or z["strength"] > best_demand["strength"]:
                best_demand = z
                dist_demand = round(dist, 2)
        elif best_demand is None:
            best_demand = z
            dist_demand = round(dist, 2)

    # ── Best nearby supply zone ───────────────
    best_supply = None
    in_supply   = False
    dist_supply = None
    for z in supply_zones:
        dist = (z["zone_low"] - cmp) / cmp * 100
        if z["zone_low"] <= cmp <= z["zone_high"] * 1.02:
            in_supply = True
            if best_supply is None or z["strength"] > best_supply["strength"]:
                best_supply = z
                dist_supply = round(dist, 2)
        elif best_supply is None:
            best_supply = z
            dist_supply = round(dist, 2)

    # ── Volume–Price Analysis ─────────────────
    recent5 = df.tail(5)
    price_up5   = (recent5["Close"].iloc[-1] - recent5["Close"].iloc[0]) / recent5["Close"].iloc[0] * 100
    vol_surge   = vol_ratio >= 1.5

    # Accumulation candles: body > 60% of range, close in upper 30%
    def is_accum(row):
        rng  = row["High"] - row["Low"]
        body = abs(row["Close"] - row["Open"])
        upper_close = row["Close"] >= row["Low"] + 0.7 * rng
        return rng > 0 and body >= 0.4 * rng and upper_close

    def is_distrib(row):
        rng  = row["High"] - row["Low"]
        body = abs(row["Close"] - row["Open"])
        lower_close = row["Close"] <= row["Low"] + 0.3 * rng
        return rng > 0 and body >= 0.4 * rng and lower_close

    accum_bars  = sum(is_accum(recent5.iloc[i]) for i in range(len(recent5)))
    distrib_bars = sum(is_distrib(recent5.iloc[i]) for i in range(len(recent5)))

    # ── 20-day, 5-day returns ─────────────────
    ret_20d = round((cmp - df["Close"].iloc[-21]) / df["Close"].iloc[-21] * 100, 2) if len(df) >= 21 else None
    ret_5d  = round(price_up5, 2)

    # ── 52-week range ─────────────────────────
    tail52 = df.tail(252)
    hi52   = round(tail52["High"].max(), 2)
    lo52   = round(tail52["Low"].min(), 2)
    pos52  = round((cmp - lo52) / (hi52 - lo52) * 100, 1) if hi52 != lo52 else 50

    # ── Breakout Detection ────────────────────
    recent20_high = df["High"].iloc[-21:-1].max() if len(df) >= 21 else cmp
    breakout = cmp > recent20_high and vol_surge

    # ── SIGNAL GENERATION ─────────────────────
    score = 0
    reasons = []

    # Demand zone buy signals
    if in_demand:
        score += 30
        reasons.append("✦ Price in Demand Zone")
    elif best_demand and dist_demand is not None and 0 <= dist_demand <= 5:
        score += 15
        reasons.append("Near Demand Zone")

    # Supply zone sell signals
    if in_supply:
        score -= 25
        reasons.append("⚠ Price in Supply Zone")

    # Trend alignment
    score += sma_align * 10
    if sma_align == 3:
        reasons.append("All SMAs Bullish")
    elif sma_align == 2:
        reasons.append("SMA20/50 Bullish")

    # Volume confirmation
    if vol_surge and accum_bars >= 2:
        score += 20
        reasons.append("✦ Volume + Accumulation")
    elif vol_surge:
        score += 10
        reasons.append("Volume Surge")

    if accum_bars >= 3:
        score += 15
        reasons.append("Strong Accumulation")
    if distrib_bars >= 3:
        score -= 20
        reasons.append("Distribution Pattern")

    # Breakout
    if breakout:
        score += 25
        reasons.append("✦ 20D Breakout")

    # RSI filter
    if rsi_val:
        if 55 <= rsi_val <= 70:
            score += 10
            reasons.append(f"RSI Momentum ({rsi_val})")
        elif rsi_val > 75:
            score -= 10
            reasons.append(f"RSI Overbought ({rsi_val})")
        elif rsi_val < 35:
            score -= 5
            reasons.append(f"RSI Weak ({rsi_val})")

    # Momentum
    if ret_5d and ret_5d > 3:
        score += 8
    if ret_20d and ret_20d > 8:
        score += 7

    # ── Final Signal ──────────────────────────
    score = max(-100, min(100, score))
    if score >= 60:
        signal = "STRONG BUY"
    elif score >= 30:
        signal = "BUY"
    elif score >= 10:
        signal = "WATCH"
    elif score <= -30:
        signal = "SELL"
    elif score <= -10:
        signal = "CAUTION"
    else:
        signal = "NEUTRAL"

    if breakout and score >= 30:
        signal = "BREAKOUT"

    # Supply zone override
    if in_supply and score < 20:
        signal = "NEAR SUPPLY"

    return {
        "cmp":           round(cmp, 2),
        "signal":        signal,
        "score":         score,
        "trend":         trend_tag,
        "rsi":           rsi_val,
        "vol_ratio":     vol_ratio,
        "ret_5d":        ret_5d,
        "ret_20d":       ret_20d,
        "pos_52w":       pos52,
        "hi_52w":        hi52,
        "lo_52w":        lo52,
        "demand_zone":   f"{best_demand['zone_low']}–{best_demand['zone_high']}" if best_demand else "—",
        "demand_str":    best_demand["strength"] if best_demand else 0,
        "dist_demand":   dist_demand,
        "in_demand":     in_demand,
        "supply_zone":   f"{best_supply['zone_low']}–{best_supply['zone_high']}" if best_supply else "—",
        "supply_str":    best_supply["strength"] if best_supply else 0,
        "dist_supply":   dist_supply,
        "in_supply":     in_supply,
        "accum_bars":    accum_bars,
        "distrib_bars":  distrib_bars,
        "sma20":         round(sma20_v, 2) if not np.isnan(sma20_v) else None,
        "sma50":         round(sma50_v, 2) if not np.isnan(sma50_v) else None,
        "sma200":        round(sma200_v, 2) if not np.isnan(sma200_v) else None,
        "atr":           round(atr_v, 2) if not np.isnan(atr_v) else None,
        "reasons":       reasons,
        "breakout":      breakout,
        "last_date":     str(df.iloc[-1]["Datetime"].date()),
    }


# ─────────────────────────────────────────────
# HTML GENERATOR
# ─────────────────────────────────────────────

SIGNAL_META = {
    "STRONG BUY":  ("🟢", "#00c853", "#e8f5e9"),
    "BUY":         ("🟩", "#43a047", "#f1f8e9"),
    "BREAKOUT":    ("🚀", "#ff6f00", "#fff8e1"),
    "WATCH":       ("👁",  "#1976d2", "#e3f2fd"),
    "NEUTRAL":     ("⬜", "#757575", "#f5f5f5"),
    "CAUTION":     ("🟡", "#f57f17", "#fff9c4"),
    "NEAR SUPPLY": ("🔶", "#e65100", "#fbe9e7"),
    "SELL":        ("🔴", "#d32f2f", "#ffebee"),
}


def pct_badge(val):
    if val is None:
        return '<span class="na">—</span>'
    color = "#00c853" if val >= 0 else "#d32f2f"
    arrow = "▲" if val >= 0 else "▼"
    return f'<span style="color:{color};font-weight:600">{arrow} {abs(val):.2f}%</span>'


def score_bar(score):
    pct  = (score + 100) / 2   # map -100..100 → 0..100
    hue  = max(0, min(120, int(pct * 1.2)))
    return (
        f'<div class="score-wrap" title="Score: {score}">'
        f'<div class="score-bar" style="width:{pct:.0f}%;background:hsl({hue},80%,42%)"></div>'
        f'<span class="score-num">{score:+d}</span></div>'
    )


def build_html(rows, title, subtitle, generated_at):
    signal_order = ["BREAKOUT","STRONG BUY","BUY","WATCH","NEUTRAL","CAUTION","NEAR SUPPLY","SELL"]
    rows = sorted(rows, key=lambda r: (signal_order.index(r["signal"]) if r["signal"] in signal_order else 99, -r["score"]))

    # Summary counts
    counts = {}
    for r in rows:
        counts[r["signal"]] = counts.get(r["signal"], 0) + 1

    summary_html = ""
    for sig in signal_order:
        if sig in counts:
            em, col, _ = SIGNAL_META.get(sig, ("", "#888", "#eee"))
            summary_html += f'<div class="pill" style="border-color:{col};color:{col}">{em} {sig} <b>{counts[sig]}</b></div>'

    # Table rows
    tbody = ""
    for r in rows:
        s      = r["signal"]
        em, col, bg = SIGNAL_META.get(s, ("", "#888", "#fafafa"))
        rsn    = "<br>".join(r.get("reasons", []))
        demand_cls = ' class="zone-active"' if r.get("in_demand") else ""
        supply_cls = ' class="zone-warn"'   if r.get("in_supply") else ""
        sma_html = ""
        for label, key in [("S20","sma20"),("S50","sma50"),("S200","sma200")]:
            v = r.get(key)
            if v:
                ok = r["cmp"] > v
                c  = "#00c853" if ok else "#d32f2f"
                sma_html += f'<span style="color:{c};margin-right:4px">{label}</span>'

        tbody += f"""
<tr style="background:{bg}">
  <td class="sticky-col stock-name" data-val="{r['name']}"><a href="https://in.tradingview.com/chart/0dT5rHYi/?symbol=NSE%3A{extract_display_name(r['name'])}">{extract_display_name(r['name'])}</a></td>
  <td data-val="{r['cmp']}" style="font-weight:700">₹{r['cmp']:,.2f}</td>
  <td data-val="{signal_order.index(s)}" style="white-space:nowrap">
    <span class="sig-badge" style="background:{col}">{em} {s}</span>
  </td>
  <td data-val="{r['score']}">{score_bar(r['score'])}</td>
  <td data-val="{r.get('rsi') or 0}">{r.get('rsi') or '—'}</td>
  <td data-val="{r.get('vol_ratio') or 0}">{r.get('vol_ratio') or '—'}×</td>
  <td data-val="{r.get('ret_5d') or 0}">{pct_badge(r.get('ret_5d'))}</td>
  <td data-val="{r.get('ret_20d') or 0}">{pct_badge(r.get('ret_20d'))}</td>
  <td data-val="{r.get('pos_52w') or 0}">
    <div class="range-bar"><div class="range-fill" style="width:{r.get('pos_52w',0)}%"></div></div>
    <small>{r.get('pos_52w',0)}%</small>
  </td>
  <td{demand_cls} data-val="{r.get('demand_str') or 0}">{r.get('demand_zone','—')}<br><small>Str: {r.get('demand_str',0):.0f} | Dist: {r.get('dist_demand','—')}%</small></td>
  <td{supply_cls} data-val="{r.get('supply_str') or 0}">{r.get('supply_zone','—')}<br><small>Str: {r.get('supply_str',0):.0f} | Dist: {r.get('dist_supply','—')}%</small></td>
  <td data-val="{r.get('accum_bars') or 0}">{sma_html}<br><small>Acc:{r.get('accum_bars',0)} Dis:{r.get('distrib_bars',0)}</small></td>
  <td class="reasons" data-val="{len(r.get('reasons',[]))}">{rsn or '—'}</td>
  <td data-val="{r.get('last_date','')}" style="font-size:11px;color:#666">{r.get('last_date','')}</td>
</tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #21262d;
    --border:   #30363d;
    --text:     #1303FC;
    --muted:    #8b949e;
    --accent:   #58a6ff;
    --green:    #3fb950;
    --red:      #f85149;
    --orange:   #d29922;
    --radius:   8px;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    padding: 24px;
  }}

  /* ── Header ── */
  .header {{
    display: flex; align-items: flex-start; justify-content: space-between;
    margin-bottom: 32px; flex-wrap: wrap; gap: 16px;
  }}
  .header h1 {{
    font-size: clamp(20px,4vw,32px); font-weight: 700; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #58a6ff, #a371f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .header .meta {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .header .badge {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: 6px 14px; font-size: 13px;
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Summary pills ── */
  .summary {{
    display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 24px;
  }}
  .pill {{
    border: 1.5px solid; border-radius: 20px;
    padding: 5px 12px; font-size: 12px; font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
  }}

  /* ── Controls ── */
  .controls {{
    display: flex; gap: 12px; flex-wrap: wrap;
    align-items: center; margin-bottom: 16px;
  }}
  .search-box {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 8px 14px;
    color: var(--text); font-size: 14px; width: 220px;
    font-family: 'IBM Plex Sans', sans-serif;
  }}
  .search-box:focus {{ outline: none; border-color: var(--accent); }}
  .filter-btn {{
    background: var(--surface); border: 1px solid var(--border);
    color: var(--muted); border-radius: var(--radius);
    padding: 7px 14px; font-size: 12px; cursor: pointer;
    font-family: 'IBM Plex Mono', monospace; transition: all .2s;
  }}
  .filter-btn:hover, .filter-btn.active {{
    border-color: var(--accent); color: var(--accent);
    background: rgba(88,166,255,.08);
  }}

  /* ── Table wrapper ── */
  .table-wrap {{
    overflow-x: auto; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--surface);
  }}
  table {{
    width: 100%; border-collapse: collapse; font-size: 13px;
  }}
  thead th {{
    background: var(--surface2);
    padding: 12px 14px; text-align: left;
    font-size: 11px; text-transform: uppercase;
    letter-spacing: .6px; color: var(--muted);
    white-space: nowrap; cursor: pointer;
    user-select: none; position: sticky; top: 0; z-index: 2;
    border-bottom: 1px solid var(--border);
    transition: color .2s;
  }}
  thead th:hover {{ color: var(--accent); }}
  thead th.sort-asc::after  {{ content: " ▲"; color: var(--accent); }}
  thead th.sort-desc::after {{ content: " ▼"; color: var(--accent); }}

  tbody tr {{
    border-bottom: 1px solid var(--border);
    transition: filter .15s;
  }}
  tbody tr:hover {{ filter: brightness(1.12); }}
  tbody tr:last-child {{ border-bottom: none; }}
  tbody td {{
    padding: 10px 14px; vertical-align: middle;
    color: var(--text);
  }}

  /* ── Sticky first col ── */
  .sticky-col {{
    position: sticky; left: 0; z-index: 1;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600; font-size: 12px;
    background: inherit;
    border-right: 1px solid var(--border);
  }}

  /* ── Signal badge ── */
  .sig-badge {{
    color: #fff; border-radius: 4px;
    padding: 3px 8px; font-size: 11px; font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
  }}

  /* ── Score bar ── */
  .score-wrap {{
    display: flex; align-items: center; gap: 6px;
    min-width: 120px;
  }}
  .score-bar {{
    height: 6px; border-radius: 3px; flex-shrink: 0;
    transition: width .3s;
  }}
  .score-num {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: var(--muted);
  }}

  /* ── 52w range bar ── */
  .range-bar {{
    width: 80px; height: 5px; background: var(--surface2);
    border-radius: 3px; overflow: hidden; display: inline-block;
    vertical-align: middle; margin-right: 4px;
  }}
  .range-fill {{ height: 100%; background: var(--accent); border-radius: 3px; }}

  /* ── Zone cells ── */
  .zone-active {{ background: rgba(0,200,83,.07) !important; }}
  .zone-warn   {{ background: rgba(245,124,0,.07) !important; }}

  /* ── Reasons ── */
  .reasons {{ font-size: 11px; color: var(--muted); max-width: 220px; }}
  .na {{ color: var(--muted); }}

  /* ── Footer ── */
  .footer {{
    margin-top: 32px; text-align: center;
    font-size: 11px; color: var(--muted); line-height: 1.8;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>{title}</h1>
    <div class="meta">{subtitle}</div>
    <div class="meta" style="margin-top:4px">Generated: {generated_at} &nbsp;|&nbsp; {len(rows)} stocks analysed</div>
  </div>
  <div class="badge">Institutional Price Action Screener</div>
</div>

<div class="summary">{summary_html}</div>

<div class="controls">
  <input class="search-box" id="search" placeholder="🔍 Search stock…" oninput="filterRows()">
  <button class="filter-btn active" onclick="setFilter('all',this)">All</button>
  <button class="filter-btn" onclick="setFilter('BREAKOUT',this)">🚀 Breakout</button>
  <button class="filter-btn" onclick="setFilter('STRONG BUY',this)">🟢 Strong Buy</button>
  <button class="filter-btn" onclick="setFilter('BUY',this)">🟩 Buy</button>
  <button class="filter-btn" onclick="setFilter('WATCH',this)">👁 Watch</button>
  <button class="filter-btn" onclick="setFilter('SELL',this)">🔴 Sell</button>
</div>

<div class="table-wrap">
<table id="mainTable">
<thead>
<tr>
  <th onclick="sortTable(0)">Stock</th>
  <th onclick="sortTable(1)">CMP</th>
  <th onclick="sortTable(2)">Signal</th>
  <th onclick="sortTable(3)">Score</th>
  <th onclick="sortTable(4)">RSI</th>
  <th onclick="sortTable(5)">Vol Ratio</th>
  <th onclick="sortTable(6)">5D Ret</th>
  <th onclick="sortTable(7)">20D Ret</th>
  <th onclick="sortTable(8)">52W Pos</th>
  <th onclick="sortTable(9)">Demand Zone ↑</th>
  <th onclick="sortTable(10)">Supply Zone ↓</th>
  <th onclick="sortTable(11)">SMA / Pattern</th>
  <th onclick="sortTable(12)">Reasons</th>
  <th onclick="sortTable(13)">Last Data</th>
</tr>
</thead>
<tbody id="tableBody">
{tbody}
</tbody>
</table>
</div>

<div class="footer">
  <b>Disclaimer:</b> This tool is for educational &amp; research purposes only.<br>
  Not SEBI-registered investment advice. Always do your own due diligence.<br>
  Demand/Supply zones are algorithmic approximations of institutional price levels.
</div>

<script>
let activeFilter = 'all';
let sortCol = -1, sortDir = 1;

function filterRows() {{
  const q = document.getElementById('search').value.toLowerCase();
  const rows = document.querySelectorAll('#tableBody tr');
  rows.forEach(r => {{
    const name   = r.cells[0].textContent.toLowerCase();
    const signal = r.cells[2].textContent.toLowerCase();
    const matchQ = name.includes(q);
    const matchF = activeFilter === 'all' || signal.includes(activeFilter.toLowerCase());
    r.style.display = (matchQ && matchF) ? '' : 'none';
  }});
}}

function setFilter(f, btn) {{
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  filterRows();
}}

function sortTable(col) {{
  const tbody = document.getElementById('tableBody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  const ths   = document.querySelectorAll('thead th');

  if (sortCol === col) sortDir *= -1;
  else {{ sortCol = col; sortDir = 1; }}

  ths.forEach((th, i) => {{
    th.classList.remove('sort-asc','sort-desc');
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
    return html


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Institutional Price Action Screener")
    print("=" * 60)

    nifty750 = load_nifty750()
    if nifty750:
        print(f"  Loaded {len(nifty750)} normalised tickers from nifty750.txt")
        sample = sorted(list(nifty750))[:5]
        print(f"  Sample (normalised): {sample}")
    else:
        print("  ⚠  nifty750.txt not found — all stocks → 'Other' bucket")

    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"\n  ERROR: No CSV files found in '{DATA_DIR}/' directory.")
        return

    # ── Show filename → ticker mapping for first 5 files (diagnostic) ──
    print(f"  Found {len(csv_files)} CSV files in {DATA_DIR}/")
    print("  Filename → Normalised ticker (first 5 samples):")
    for p in sorted(csv_files)[:5]:
        fn  = os.path.basename(p)
        tok = normalise_ticker(Path(fn).stem)
        hit = "✓ Nifty750" if tok in nifty750 else "  Other"
        print(f"    {fn:<35} → {tok:<20} [{hit}]")
    print()

    nifty750_rows, other_rows = [], []
    errors, skipped = [], []
    generated_at = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    for i, path in enumerate(csv_files, 1):
        filename = os.path.basename(path)
        raw_stem = Path(filename).stem          # e.g. BSE_AARTIIND-A
        ticker   = normalise_ticker(raw_stem)   # e.g. AARTIIND  (for lookup)
        display  = raw_stem.upper()             # full name shown in HTML

        if i % 50 == 0 or i == len(csv_files):
            print(f"  Processing {i}/{len(csv_files)} — {ticker:<20}", end="\r")

        df = load_csv(path)
        if df is None or len(df) < MIN_ROWS:
            skipped.append(ticker)
            continue

        try:
            result = analyse(df)
            if result is None:
                skipped.append(ticker)
                continue
            result["name"]   = display   # shown in HTML table
            result["ticker"] = ticker    # normalised symbol used for lookup
            if ticker in nifty750:
                nifty750_rows.append(result)
            else:
                other_rows.append(result)
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    print(f"\n\n  ✓ Analysed: {len(nifty750_rows) + len(other_rows)} stocks")
    print(f"    Nifty 750 bucket : {len(nifty750_rows)}")
    print(f"    Other bucket     : {len(other_rows)}")
    print(f"    Skipped (< data) : {len(skipped)}")
    if errors:
        print(f"    Errors           : {len(errors)}")
        for e in errors[:5]:
            print(f"      {e}")

    # ── Write HTML outputs ────────────────────
    if nifty750_rows:
        html = build_html(
            nifty750_rows,
            title    = "Nifty 750 — Institutional Price Action Signals",
            subtitle = "Demand Zone Accumulation · Supply Zone Distribution · Volume-Price Momentum",
            generated_at = generated_at
        )
        with open(OUT_NIFTY750, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n  ✅  {OUT_NIFTY750}  ({len(nifty750_rows)} stocks)")
    else:
        print(f"\n  ⚠  No Nifty 750 stocks found — skipping {OUT_NIFTY750}")
        # Still write an empty file
        html = build_html([], "Nifty 750 — Signals", "No data matched nifty750.txt", generated_at)
        with open(OUT_NIFTY750, "w", encoding="utf-8") as f:
            f.write(html)

    if other_rows:
        html = build_html(
            other_rows,
            title    = "Other Stocks — Institutional Price Action Signals",
            subtitle = "Demand Zone Accumulation · Supply Zone Distribution · Volume-Price Momentum",
            generated_at = generated_at
        )
        with open(OUT_OTHERS, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  ✅  {OUT_OTHERS}  ({len(other_rows)} stocks)")
    else:
        print(f"  ⚠  No 'Other' stocks to write — skipping {OUT_OTHERS}")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
