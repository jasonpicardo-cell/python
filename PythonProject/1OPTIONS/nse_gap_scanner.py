"""
nse_gap_scanner.py  — Overnight Gap Scanner
Compares latest session open against previous session close.

Gap types:
  Gap Up    gap_pct >=  0.5%
  Gap Down  gap_pct <= -0.5%
  Flat      within ±0.5%

Also checks if the gap was subsequently filled intraday.
"""
from __future__ import annotations
import logging, time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)
_gap_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=5)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 2: return None

    cur  = rows[-1]
    prev = rows[-2]
    if not prev["C"] or not cur["O"]: return None

    gap_pct   = round((cur["O"] - prev["C"]) / prev["C"] * 100, 2)
    close_pct = round((cur["C"] - prev["C"]) / prev["C"] * 100, 2)
    open_pct  = round((cur["O"] - prev["C"]) / prev["C"] * 100, 2)

    if   gap_pct >=  0.5: gap_type = "Gap Up"
    elif gap_pct <= -0.5: gap_type = "Gap Down"
    else:                 gap_type = "Flat"

    # Was the gap filled during the session?
    if   gap_type == "Gap Up":   gap_filled = cur["L"] <= prev["C"]
    elif gap_type == "Gap Down": gap_filled = cur["H"] >= prev["C"]
    else:                        gap_filled = True

    return {
        "symbol":     symbol,
        "price":      round(cur["C"], 2),
        "price_date": cur["date"],
        "open":       round(cur["O"], 2),
        "prev_close": round(prev["C"], 2),
        "gap_pct":    gap_pct,
        "close_pct":  close_pct,
        "gap_type":   gap_type,
        "gap_filled": gap_filled,
    }


def run_gap_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _gap_cache
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
