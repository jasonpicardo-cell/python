"""nse_williamsr_scanner.py — Williams %R (period 14)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_wr_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300

def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=20)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 15: return None
    window = rows[-14:]; c = rows[-1]["C"]
    hh = max(r["H"] for r in window); ll = min(r["L"] for r in window)
    wr = round(-100 * (hh - c) / (hh - ll) if hh != ll else -50, 2)
    # Prev value for turn detection
    w2 = rows[-2]["C"] if len(rows) >= 2 else c
    hw2 = max(r["H"] for r in rows[-15:-1]); lw2 = min(r["L"] for r in rows[-15:-1])
    wr_prev = -100 * (hw2 - w2) / (hw2 - lw2) if hw2 != lw2 else wr
    if   wr > -20:                          signal = "Overbought"
    elif wr < -80:                          signal = "Oversold"
    elif wr > -20 and wr_prev <= -20:       signal = "Turning Down"
    elif wr < -80 and wr_prev >= -80:       signal = "Turning Up"
    elif wr > -50:                          signal = "Bullish"
    else:                                   signal = "Bearish"
    return {"symbol": symbol, "price": round(c, 2), "price_date": rows[-1]["date"],
            "wr": wr, "signal": signal}

def run_williamsr_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c2 = _wr_cache
    if not force and c2["data"] and c2["key"] == key and (time.time()-c2["ts"]) < ttl:
        return c2["data"]
    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,
                "sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),"error":"Empty list"}
    stocks, errors, skipped = [], [], []
    for sym in symbols:
        try:
            r = scan_stock(sym); (stocks if r else skipped).append(r or sym)
        except Exception as e:
            errors.append({"symbol": sym, "reason": str(e)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),
            "total":len(symbols),"source":source,"sym_label":sym_label,
            "as_of":time.strftime("%H:%M:%S")}
    c2["ts"]=time.time(); c2["data"]=data; c2["key"]=key
    return data
