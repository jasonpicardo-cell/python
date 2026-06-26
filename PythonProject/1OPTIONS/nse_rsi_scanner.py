"""
nse_rsi_scanner.py  — RSI-14 Scanner
Uses Wilder's RMA smoothing (matches TradingView RSI exactly).

Signal bands:
  Overbought  RSI ≥ 70
  Bullish     RSI 55–70
  Neutral     RSI 45–55
  Bearish     RSI 30–45
  Oversold    RSI ≤ 30

Also detects fresh cross of 50 midline (momentum shift).
"""
from __future__ import annotations
import logging, time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)
_rsi_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def _rma(values: list[float], period: int) -> list[float | None]:
    """Wilder's RMA (RMA = EMA with alpha=1/period, SMA seed)."""
    n, out = len(values), [None] * len(values)
    if n < period: return out
    out[period - 1] = sum(values[:period]) / period
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period  # type: ignore
    return out


def _rsi_series(closes: list[float], period: int = 14) -> list[float | None]:
    n = len(closes)
    if n < period + 2: return [None] * n
    gains  = [max(closes[i] - closes[i-1], 0.0) for i in range(1, n)]
    losses = [max(closes[i-1] - closes[i], 0.0) for i in range(1, n)]
    ag, al = _rma(gains, period), _rma(losses, period)
    rsi: list[float | None] = [None] * n
    for i in range(period - 1, n - 1):
        if ag[i] is None or al[i] is None: continue
        rsi[i + 1] = 100.0 if al[i] == 0 else 100 - 100 / (1 + ag[i] / al[i])  # type: ignore
    return rsi


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=40)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 17: return None

    closes = [r["C"] for r in rows]
    rsi_s  = _rsi_series(closes, 14)
    cur    = next((v for v in reversed(rsi_s) if v is not None), None)
    prev   = next((v for v in reversed(rsi_s[:-1]) if v is not None), None)
    if cur is None: return None

    # Signal
    if   cur >= 70: sig = "Overbought"
    elif cur >= 55: sig = "Bullish"
    elif cur >= 45: sig = "Neutral"
    elif cur >= 30: sig = "Bearish"
    else:           sig = "Oversold"

    # Fresh midline cross (last 3 bars)
    valid = [v for v in rsi_s if v is not None]
    fresh_bull = fresh_bear = False
    if len(valid) >= 4:
        fresh_bull = valid[-1] > 50 and any(v <= 50 for v in valid[-4:-1])
        fresh_bear = valid[-1] < 50 and any(v >= 50 for v in valid[-4:-1])

    return {"symbol": symbol, "price": round(closes[-1], 2),
            "price_date": rows[-1]["date"], "rsi": round(cur, 2),
            "rsi_prev": round(prev, 2) if prev else None,
            "signal": sig, "fresh_bull": fresh_bull, "fresh_bear": fresh_bear}


def run_rsi_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _rsi_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]
    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,
                "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),
                "error":f"Symbol list empty for {source!r}"}
    stocks, errors, skipped = [], [], []
    for sym in symbols:
        try:
            r = scan_stock(sym)
            (stocks if r else skipped).append(r or sym)
        except Exception as exc:
            errors.append({"symbol": sym, "reason": str(exc)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,
            "count":len(stocks),"total":len(symbols),
            "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
