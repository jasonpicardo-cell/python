"""
nse_indicators.py — shared technical indicator primitives
Imported by all new scanner modules to avoid duplication.
All implementations match TradingView defaults.
"""
from __future__ import annotations
import math


def sma(closes: list[float], period: int) -> list[float | None]:
    n, out = len(closes), [None] * len(closes)
    for i in range(period - 1, n):
        out[i] = sum(closes[i - period + 1:i + 1]) / period
    return out


def ema(closes: list[float], period: int) -> list[float | None]:
    n, k = len(closes), 2.0 / (period + 1)
    out: list[float | None] = [None] * n
    if n < period: return out
    out[period - 1] = sum(closes[:period]) / period
    for i in range(period, n):
        out[i] = closes[i] * k + out[i - 1] * (1 - k)  # type: ignore
    return out


def rma(values: list[float], period: int) -> list[float | None]:
    """Wilder's RMA (EMA with alpha=1/period, SMA seed)."""
    n, out = len(values), [None] * len(values)
    if n < period: return out
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period  # type: ignore
    return out


def stddev(closes: list[float], period: int) -> list[float | None]:
    n, out = len(closes), [None] * len(closes)
    for i in range(period - 1, n):
        sl = closes[i - period + 1:i + 1]
        m  = sum(sl) / period
        out[i] = math.sqrt(sum((x - m) ** 2 for x in sl) / period)
    return out


def atr(highs: list[float], lows: list[float], closes: list[float],
        period: int = 14) -> list[float | None]:
    n = len(closes)
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(highs[i] - lows[i],
                      abs(highs[i] - closes[i - 1]),
                      abs(lows[i]  - closes[i - 1])))
    return rma(tr, period)


def pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient of two equal-length series."""
    n = len(xs)
    if n < 2: return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    return round(num / (dx * dy), 4) if dx * dy else None
