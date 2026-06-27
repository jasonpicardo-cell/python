"""
nse_rs_scanner.py — Relative Strength vs Nifty (20D)

RS = stock_return_20D / nifty_return_20D
RS > 1.5  Strong Outperform
RS > 1.0  Outperform
RS 0.8–1.0 In-line
RS < 0.8  Underperform
RS < 0.0  Negative divergence (stock down while Nifty up, or worse)
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_rs_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300
_PERIOD = 20


def _load_nifty_close() -> list[tuple[str, float]] | None:
    for name in ("NIFTY", "NIFTY50", "NIFTY_50", "^NSEI"):
        p = _find_csv(name)
        if p:
            rows, err = _read_rows(p, n=_PERIOD + 5)
            if not err and rows:
                today = date.today().isoformat()
                rows = [r for r in rows if r["date"] < today]
                if len(rows) >= _PERIOD + 1:
                    return [(r["date"], r["C"]) for r in rows]
    return None


def scan_stock(symbol: str, nifty_closes: list[tuple[str, float]]) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=_PERIOD + 5)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < _PERIOD + 1: return None

    stock_ret = (rows[-1]["C"] - rows[-(_PERIOD + 1)]["C"]) / rows[-(_PERIOD + 1)]["C"] * 100
    nifty_start = nifty_closes[-(_PERIOD + 1)][1]
    nifty_end   = nifty_closes[-1][1]
    if nifty_start == 0: return None
    nifty_ret   = (nifty_end - nifty_start) / nifty_start * 100

    rs = round(stock_ret / nifty_ret, 3) if nifty_ret != 0 else None
    if rs is None: return None

    if   rs >= 1.5:  signal = "Strong Outperform"
    elif rs >= 1.0:  signal = "Outperform"
    elif rs >= 0.8:  signal = "In-line"
    elif rs >= 0.0:  signal = "Underperform"
    else:            signal = "Negative Diverge"

    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"],
            "rs": rs, "stock_ret": round(stock_ret, 2),
            "nifty_ret": round(nifty_ret, 2), "signal": signal}


def run_rs_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _rs_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]
    nifty = _load_nifty_close()
    if not nifty:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,
                "as_of":time.strftime("%H:%M:%S"),"error":"Nifty CSV not found"}
    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,
                "sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty list"}
    stocks, errors, skipped = [], [], []
    for sym in symbols:
        try:
            r = scan_stock(sym, nifty)
            (stocks if r else skipped).append(r or sym)
        except Exception as e:
            errors.append({"symbol": sym, "reason": str(e)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),
            "total":len(symbols),"source":source,"sym_label":sym_label,
            "as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
