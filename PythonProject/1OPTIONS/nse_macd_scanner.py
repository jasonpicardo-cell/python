"""
nse_macd_scanner.py — MACD Scanner (12/26/9, TradingView defaults)

MACD line   = EMA12 − EMA26
Signal line = EMA9 of MACD
Histogram   = MACD − Signal

Signals (checked in priority order):
  Bullish Crossover   MACD crossed above Signal in last 3 bars
  Bearish Crossover   MACD crossed below Signal in last 3 bars
  Zero Cross Up       MACD crossed above 0 in last 3 bars
  Zero Cross Down     MACD crossed below 0 in last 3 bars
  Above Zero          MACD > 0, Signal > 0
  Below Zero          MACD < 0, Signal < 0
  Neutral             mixed
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import ema

_macd_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def _crossed_above(a: list, b: list, lookback: int = 3) -> bool:
    valid = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(valid) < lookback + 1: return False
    recent = valid[-(lookback + 1):]
    return recent[-1][0] > recent[-1][1] and any(r[0] <= r[1] for r in recent[:-1])


def _crossed_below(a: list, b: list, lookback: int = 3) -> bool:
    valid = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(valid) < lookback + 1: return False
    recent = valid[-(lookback + 1):]
    return recent[-1][0] < recent[-1][1] and any(r[0] >= r[1] for r in recent[:-1])


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=70)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 35: return None

    closes  = [r["C"] for r in rows]
    e12     = ema(closes, 12)
    e26     = ema(closes, 26)

    macd_s: list[float | None] = [
        (a - b) if (a is not None and b is not None) else None
        for a, b in zip(e12, e26)
    ]
    valid_macd = [v for v in macd_s if v is not None]
    if len(valid_macd) < 9: return None

    sig_s = ema(valid_macd, 9)
    # Pad sig_s back to same length as macd_s
    sig_full: list[float | None] = [None] * len(macd_s)
    j = 0
    for i, v in enumerate(macd_s):
        if v is not None:
            sig_full[i] = sig_s[j]; j += 1

    cur_macd = next((v for v in reversed(macd_s) if v is not None), None)
    cur_sig  = next((v for v in reversed(sig_full) if v is not None), None)
    if cur_macd is None or cur_sig is None: return None

    hist     = round(cur_macd - cur_sig, 4)
    zero     = [0.0] * len(macd_s)

    if   _crossed_above(macd_s, sig_full):       signal = "Bullish Crossover"
    elif _crossed_below(macd_s, sig_full):        signal = "Bearish Crossover"
    elif _crossed_above(macd_s, zero):            signal = "Zero Cross Up"
    elif _crossed_below(macd_s, zero):            signal = "Zero Cross Down"
    elif cur_macd > 0 and cur_sig > 0:            signal = "Above Zero"
    elif cur_macd < 0 and cur_sig < 0:            signal = "Below Zero"
    else:                                          signal = "Neutral"

    return {"symbol": symbol, "price": round(closes[-1], 2), "price_date": rows[-1]["date"],
            "macd": round(cur_macd, 4), "signal_line": round(cur_sig, 4),
            "histogram": hist, "signal": signal}


def run_macd_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _macd_cache
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
        except Exception as e:
            errors.append({"symbol": sym, "reason": str(e)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),
            "total":len(symbols),"source":source,"sym_label":sym_label,
            "as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
