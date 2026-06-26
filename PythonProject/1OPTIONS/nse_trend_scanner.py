"""
nse_trend_scanner.py
====================
Supertrend + EMA Crossover Scanner for NSE F&O stocks.

All calculations are purely CSV-based — no NSE API calls needed.
Results match TradingView defaults:
  Supertrend  : ATR period 7, multiplier 3.0
  EMA pairs   : 9/21 (short), 20/50 (medium), 50/200 (long / Golden Cross)

Signal strength (0–4 bullish):
  4  → Strong Buy  (all signals bullish)
  3  → Buy
  2  → Neutral
  1  → Sell
  0  → Strong Sell (all signals bearish)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from nse_pivot_scanner import (
    DATA_DIR, SYMBOL_LISTS, load_symbol_list,
    load_all_csv_symbols, _find_csv, _read_rows,
)

log = logging.getLogger(__name__)

_trend_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300   # 5 minutes


# ── Indicator maths ──────────────────────────────────────────────────────────

def _ema(closes: list[float], period: int) -> list[float | None]:
    """Exponential Moving Average — matches TradingView EMA."""
    n = len(closes)
    if n < period:
        return [None] * n
    k   = 2.0 / (period + 1)
    out: list[float | None] = [None] * n
    out[period - 1] = sum(closes[:period]) / period   # SMA seed
    for i in range(period, n):
        out[i] = closes[i] * k + out[i - 1] * (1 - k)  # type: ignore[operator]
    return out


def _atr(highs: list[float], lows: list[float], closes: list[float],
         period: int = 14) -> list[float | None]:
    """Average True Range (RMA smoothing) — matches TradingView ATR."""
    n = len(closes)
    tr: list[float] = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))
    out: list[float | None] = [None] * n
    if n < period:
        return out
    out[period - 1] = sum(tr[:period]) / period          # SMA seed
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period  # type: ignore[operator]
    return out


def _supertrend(highs: list[float], lows: list[float], closes: list[float],
                period: int = 7, mult: float = 3.0
                ) -> tuple[list[int | None], list[float | None]]:
    """
    Supertrend indicator — matches TradingView.
    Returns (direction[], value[]) where direction: 1=bullish, -1=bearish.
    """
    n   = len(closes)
    atr = _atr(highs, lows, closes, period)

    ub  = [0.0] * n    # final upper band
    lb  = [0.0] * n    # final lower band
    st  = [None] * n   # supertrend value
    di  = [None] * n   # direction

    for i in range(period, n):
        if atr[i] is None:
            continue
        hl2 = (highs[i] + lows[i]) / 2
        ub_basic = hl2 + mult * atr[i]
        lb_basic = hl2 - mult * atr[i]

        ub[i] = ub_basic if (ub_basic < ub[i-1] or closes[i-1] > ub[i-1]) else ub[i-1]
        lb[i] = lb_basic if (lb_basic > lb[i-1] or closes[i-1] < lb[i-1]) else lb[i-1]

        prev_di = di[i - 1]
        if prev_di is None:
            di[i] = 1 if closes[i] > ub[i] else -1
        elif closes[i] > ub[i - 1]:
            di[i] = 1
        elif closes[i] < lb[i - 1]:
            di[i] = -1
        else:
            di[i] = prev_di

        st[i] = lb[i] if di[i] == 1 else ub[i]

    return di, st   # type: ignore[return-value]


def _crossover_state(fast: list[float | None], slow: list[float | None],
                     lookback: int = 3) -> dict:
    """
    Returns the current crossover state of two EMA series.
    {
      direction : 'bullish' | 'bearish' | None
      fresh     : True if crossover happened within last `lookback` bars
      fast      : last fast value
      slow      : last slow value
    }
    """
    # Find the last index where both are valid
    last_fast = next((v for v in reversed(fast) if v is not None), None)
    last_slow = next((v for v in reversed(slow) if v is not None), None)

    if last_fast is None or last_slow is None:
        return {"direction": None, "fresh": False, "fast": None, "slow": None}

    direction = "bullish" if last_fast > last_slow else "bearish"

    # Check for fresh crossover: direction changed within lookback bars
    fresh = False
    valid_pairs = [(f, s) for f, s in zip(fast, slow) if f is not None and s is not None]
    if len(valid_pairs) >= lookback + 1:
        recent = valid_pairs[-(lookback + 1):]
        signs  = [1 if f > s else -1 for f, s in recent]
        # If sign changed anywhere in the recent window → fresh crossover
        for j in range(1, len(signs)):
            if signs[j] != signs[j - 1]:
                fresh = True
                break

    return {
        "direction": direction,
        "fresh":     fresh,
        "fast":      round(last_fast, 2),
        "slow":      round(last_slow, 2),
    }


# ── Per-stock scan ────────────────────────────────────────────────────────────

def scan_stock(symbol: str) -> dict | None:
    """
    Compute Supertrend + EMA signals for one stock.
    Returns None if insufficient data.
    """
    p = _find_csv(symbol)
    if not p:
        return None

    rows, err = _read_rows(p, n=220)    # 200+ rows for EMA-200
    if err or not rows:
        return None

    # Filter out today
    from datetime import date
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 30:                  # need at least 30 for short EMAs
        return None

    closes = [r["C"] for r in rows]
    highs  = [r["H"] for r in rows]
    lows   = [r["L"] for r in rows]

    price       = closes[-1]
    price_date  = rows[-1]["date"]

    # ── EMAs ─────────────────────────────────────────────────────────────
    ema9   = _ema(closes, 9)
    ema21  = _ema(closes, 21)
    ema20  = _ema(closes, 20)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200) if len(closes) >= 200 else [None] * len(closes)

    cross_9_21  = _crossover_state(ema9,  ema21)
    cross_20_50 = _crossover_state(ema20, ema50)
    cross_50_200 = _crossover_state(ema50, ema200) if ema200[-1] is not None else {
        "direction": None, "fresh": False, "fast": None, "slow": None
    }

    # ── Supertrend ───────────────────────────────────────────────────────
    st_dir, st_val = _supertrend(highs, lows, closes, period=7, mult=3.0)
    last_di  = next((v for v in reversed(st_dir)  if v is not None), None)
    last_st  = next((v for v in reversed(st_val)  if v is not None), None)

    # Fresh ST direction change in last 3 bars?
    valid_di = [v for v in st_dir if v is not None]
    st_fresh = (len(valid_di) >= 2 and valid_di[-1] != valid_di[-2])

    st_result = {
        "direction": last_di,
        "signal":    "Bullish" if last_di == 1 else ("Bearish" if last_di == -1 else None),
        "value":     round(last_st, 2) if last_st else None,
        "fresh":     st_fresh,
    }

    # ── Overall signal strength ──────────────────────────────────────────
    bullish_count = sum([
        1 if last_di == 1                              else 0,
        1 if cross_9_21.get("direction")  == "bullish" else 0,
        1 if cross_20_50.get("direction") == "bullish" else 0,
        1 if cross_50_200.get("direction") == "bullish" else 0,
    ])
    valid_signals = 4 if ema200[-1] is not None else 3

    strength_map = {
        (4, 4): ("Strong Buy",  4),
        (3, 4): ("Buy",         3),
        (2, 4): ("Neutral",     2),
        (1, 4): ("Sell",        1),
        (0, 4): ("Strong Sell", 0),
        (3, 3): ("Strong Buy",  4),
        (2, 3): ("Buy",         3),
        (1, 3): ("Neutral",     2),
        (0, 3): ("Sell",        1),
    }
    label, strength = strength_map.get((bullish_count, valid_signals), ("Neutral", 2))

    return {
        "symbol":       symbol,
        "price":        round(price, 2),
        "price_date":   price_date,
        "supertrend":   st_result,
        "ema9":         round(ema9[-1],  2) if ema9[-1]  is not None else None,
        "ema21":        round(ema21[-1], 2) if ema21[-1] is not None else None,
        "ema50":        round(ema50[-1], 2) if ema50[-1] is not None else None,
        "ema200":       round(ema200[-1],2) if ema200[-1] is not None else None,
        "cross_9_21":   cross_9_21,
        "cross_20_50":  cross_20_50,
        "cross_50_200": cross_50_200,
        "signal":       label,
        "strength":     strength,    # 0=Strong Sell … 4=Strong Buy
    }


# ── Full scanner ─────────────────────────────────────────────────────────────

def run_trend_scanner(source: str = "niftyfno", force: bool = False,
                      ttl: int = _DEFAULT_TTL) -> dict:
    """
    Run Supertrend / EMA scanner across all symbols in the chosen list.
    Pure CSV — no network calls, no NSE session needed.
    """
    key = source
    c   = _trend_cache
    if (not force and c["data"] and c["key"] == key
            and (time.time() - c["ts"]) < ttl):
        return c["data"]

    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {
            "stocks": [], "errors": [], "skipped": [], "count": 0, "total": 0,
            "source": source, "sym_label": sym_label,
            "as_of": time.strftime("%H:%M:%S"),
            "error": f"Symbol list empty or not found for source={source!r}",
        }

    stocks:  list[dict] = []
    errors:  list[dict] = []
    skipped: list[str]  = []

    for sym in symbols:
        try:
            result = scan_stock(sym)
            if result is None:
                skipped.append(sym)
            else:
                stocks.append(result)
        except Exception as exc:  # noqa: BLE001
            errors.append({"symbol": sym, "reason": str(exc)})

    data = {
        "stocks":    stocks,
        "errors":    errors,
        "skipped":   skipped,
        "count":     len(stocks),
        "total":     len(symbols),
        "source":    source,
        "sym_label": sym_label,
        "as_of":     time.strftime("%H:%M:%S"),
    }
    c["ts"] = time.time(); c["data"] = data; c["key"] = key
    return data
