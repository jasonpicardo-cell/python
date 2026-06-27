"""
nse_darvas_scanner.py — Darvas Box Scanner
1. Stock makes a new N-day high (box top)
2. Next 3–10 bars stay below that high (consolidation)
3. Box bottom = lowest low of the consolidation
4. Breakout = price closes above box top
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_darvas_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300
_LOOKBACK = 50
_CONSOL_BARS = (3, 15)   # min/max consolidation bars

def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=_LOOKBACK + 10)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 20: return None

    price = rows[-1]["C"]
    # Find the most recent local high that started a consolidation
    for i in range(len(rows)-2, max(0, len(rows)-_LOOKBACK), -1):
        box_top = rows[i]["H"]
        # Must be a new high vs prior bars
        prior = [r["H"] for r in rows[max(0,i-10):i]]
        if not prior or box_top <= max(prior): continue
        # Consolidation: subsequent bars stay below box_top
        consol = rows[i+1:]
        if len(consol) < _CONSOL_BARS[0]: continue
        if len(consol) > _CONSOL_BARS[1]:  consol = consol[:_CONSOL_BARS[1]]
        if any(r["H"] > box_top for r in consol): continue  # box broken upward already
        box_bot = min(r["L"] for r in consol)
        # Current price position
        if   price > box_top:                          signal = "Breakout"
        elif price >= box_bot and price <= box_top:    signal = "In Box"
        elif price < box_bot:                          signal = "Box Broken"
        else:                                          continue
        return {"symbol": symbol, "price": round(price, 2), "price_date": rows[-1]["date"],
                "box_top": round(box_top, 2), "box_bot": round(box_bot, 2),
                "box_width_pct": round((box_top-box_bot)/box_top*100, 2),
                "signal": signal}
    return None   # no Darvas box found

def run_darvas_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _darvas_cache
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
