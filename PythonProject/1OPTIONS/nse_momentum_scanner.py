"""
nse_momentum_scanner.py  — Rate of Change (Momentum) Scanner
ROC = (close_today / close_N_days_ago - 1) × 100

Timeframes: 5D (short), 10D (medium), 20D (swing)

Signal:
  Strong Up    all three positive and strengthening
  Up           majority positive
  Flat         mixed / near-zero
  Down         majority negative
  Strong Down  all three negative
"""
from __future__ import annotations
import logging, time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)
_mom_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def _roc(closes: list[float], period: int) -> float | None:
    if len(closes) <= period or closes[-(period+1)] == 0: return None
    return round((closes[-1] / closes[-(period+1)] - 1) * 100, 2)


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=25)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 22: return None

    closes = [r["C"] for r in rows]
    r5, r10, r20 = _roc(closes, 5), _roc(closes, 10), _roc(closes, 20)

    vals  = [v for v in (r5, r10, r20) if v is not None]
    pos   = sum(1 for v in vals if v > 0)
    neg   = sum(1 for v in vals if v < 0)
    total = len(vals)

    if total == 0: return None
    if   pos == total and all(v > 1.0 for v in vals): signal = "Strong Up"
    elif pos >= total * 0.67:                          signal = "Up"
    elif neg == total and all(v < -1.0 for v in vals):signal = "Strong Down"
    elif neg >= total * 0.67:                          signal = "Down"
    else:                                              signal = "Flat"

    return {"symbol": symbol, "price": round(closes[-1], 2),
            "price_date": rows[-1]["date"],
            "roc5": r5, "roc10": r10, "roc20": r20, "signal": signal}


def run_momentum_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _mom_cache
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
