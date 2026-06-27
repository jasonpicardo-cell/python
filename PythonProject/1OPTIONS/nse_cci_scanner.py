"""nse_cci_scanner.py — Commodity Channel Index (period 20)"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_cci_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300

def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=30)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 20: return None
    w = rows[-20:]; tp = [(r["H"]+r["L"]+r["C"])/3 for r in w]
    sma_tp = sum(tp) / 20
    md = sum(abs(t - sma_tp) for t in tp) / 20
    cci = round((tp[-1] - sma_tp) / (0.015 * md) if md else 0, 2)
    cci_prev = 0
    if len(rows) >= 21:
        w2 = rows[-21:-1]; tp2 = [(r["H"]+r["L"]+r["C"])/3 for r in w2]
        s2 = sum(tp2)/20; m2 = sum(abs(t-s2) for t in tp2)/20
        cci_prev = (tp2[-1]-s2)/(0.015*m2) if m2 else 0
    if   cci >= 200:                                        signal = "Extreme Overbought"
    elif cci >= 100:                                        signal = "Overbought"
    elif cci <= -200:                                       signal = "Extreme Oversold"
    elif cci <= -100:                                       signal = "Oversold"
    elif cci > 0 and cci_prev <= 0:                         signal = "Zero Cross Up"
    elif cci < 0 and cci_prev >= 0:                         signal = "Zero Cross Down"
    elif cci > 0:                                           signal = "Bullish"
    else:                                                   signal = "Bearish"
    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"], "cci": cci, "signal": signal}

def run_cci_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _cci_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]
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
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
