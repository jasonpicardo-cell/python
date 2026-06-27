"""
nse_cpr_scanner.py — Central Pivot Range Scanner

TC = (P + H) / 2    BC = (P + L) / 2    P = (H+L+C)/3
CPR Width% = (TC - BC) / P * 100

Narrow CPR (< 0.5%)  → trending day expected
Wide CPR   (> 1.5%)  → sideways/volatile expected
Virgin CPR           → price never entered TC-BC range this session
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_cpr_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=5)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 2: return None

    prev = rows[-2]; cur = rows[-1]
    H, L, C = prev["H"], prev["L"], prev["C"]
    pivot = (H + L + C) / 3
    tc = (pivot + H) / 2; bc = (pivot + L) / 2
    width_pct = round((tc - bc) / pivot * 100, 3) if pivot else 0

    cur_price = cur["C"]; cur_h = cur["H"]; cur_l = cur["L"]
    in_cpr = cur_l <= tc and cur_h >= bc
    virgin = not in_cpr

    if   width_pct < 0.5:  day_type = "Narrow (Trend Day)"
    elif width_pct < 1.0:  day_type = "Normal"
    elif width_pct < 1.5:  day_type = "Wide"
    else:                   day_type = "Very Wide (Sideways)"

    if   cur_price > tc:    pos = "Above CPR"
    elif cur_price < bc:    pos = "Below CPR"
    else:                   pos = "Inside CPR"

    return {"symbol": symbol, "price": round(cur_price, 2), "price_date": cur["date"],
            "pivot": round(pivot, 2), "tc": round(tc, 2), "bc": round(bc, 2),
            "width_pct": width_pct, "virgin": virgin, "position": pos,
            "day_type": day_type}


def run_cpr_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _cpr_cache
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
