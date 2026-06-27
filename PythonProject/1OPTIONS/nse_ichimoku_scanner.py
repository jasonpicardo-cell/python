"""
nse_ichimoku_scanner.py — Ichimoku Cloud Scanner
Tenkan(9) Kijun(26) Senkou A/B (cloud = spans shifted 26 bars)
Needs 78 rows minimum for full computation.
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_ich_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300

def _midpt(rows, start, end) -> float:
    sl = rows[start:end]
    return (max(r["H"] for r in sl) + min(r["L"] for r in sl)) / 2

def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=90)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 78: return None
    n = len(rows)
    price  = rows[-1]["C"]
    tenkan = _midpt(rows, n-9,  n)
    kijun  = _midpt(rows, n-26, n)
    # Cloud today = spans projected 26 bars ago
    span_a = (_midpt(rows, n-35, n-26) + _midpt(rows, n-52, n-26)) / 2
    span_b = _midpt(rows, n-78, n-26)
    cloud_top = max(span_a, span_b); cloud_bot = min(span_a, span_b)
    chikou_above = price > rows[n-27]["C"] if n >= 27 else False
    above_cloud = price > cloud_top; below_cloud = price < cloud_bot
    tk_bull = tenkan > kijun
    if   above_cloud and tk_bull and chikou_above: signal = "Strong Bullish"
    elif above_cloud and tk_bull:                  signal = "Bullish"
    elif above_cloud:                              signal = "Weak Bullish"
    elif below_cloud and not tk_bull:              signal = "Bearish"
    elif below_cloud:                              signal = "Weak Bearish"
    else:                                          signal = "In Cloud"
    return {"symbol": symbol, "price": round(price, 2), "price_date": rows[-1]["date"],
            "tenkan": round(tenkan, 2), "kijun": round(kijun, 2),
            "span_a": round(span_a, 2), "span_b": round(span_b, 2),
            "cloud_top": round(cloud_top, 2), "cloud_bot": round(cloud_bot, 2),
            "signal": signal}

def run_ichimoku_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _ich_cache
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
