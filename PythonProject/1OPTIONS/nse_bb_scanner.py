"""
nse_bb_scanner.py — Bollinger Band Scanner
Period 20, StdDev multiplier 2.0 (TradingView defaults).

Signals:
  Squeeze       band-width percentile in bottom 20% (breakout imminent)
  Outside Upper price > upper band
  Upper Touch   price within 0.5% below upper band
  Outside Lower price < lower band
  Lower Touch   price within 0.5% above lower band
  Normal        within bands

%B = (price − lower) / (upper − lower) × 100
BW = (upper − lower) / middle × 100
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma, stddev

_bb_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=70)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 22: return None

    closes = [r["C"] for r in rows]
    mid = sma(closes, 20); std = stddev(closes, 20)

    cur_mid = next((v for v in reversed(mid) if v is not None), None)
    cur_std = next((v for v in reversed(std) if v is not None), None)
    if cur_mid is None or cur_std is None: return None

    price  = closes[-1]
    upper  = cur_mid + 2 * cur_std
    lower  = cur_mid - 2 * cur_std
    bw     = round((upper - lower) / cur_mid * 100, 2) if cur_mid else 0
    pct_b  = round((price - lower) / (upper - lower) * 100, 2) if (upper - lower) else 50

    # Band-width percentile over last 50 bars for squeeze detection
    bws = []
    for i in range(len(closes)):
        if mid[i] is not None and std[i] is not None and mid[i] > 0:
            bws.append((mid[i] + 2*std[i] - (mid[i] - 2*std[i])) / mid[i] * 100)  # type: ignore
    squeeze = bw <= sorted(bws)[int(len(bws) * 0.2)] if len(bws) >= 10 else False

    if   price > upper:                          signal = "Outside Upper"
    elif price >= upper * 0.995:                 signal = "Upper Touch"
    elif price < lower:                          signal = "Outside Lower"
    elif price <= lower * 1.005:                 signal = "Lower Touch"
    elif squeeze:                                signal = "Squeeze"
    else:                                        signal = "Normal"

    return {"symbol": symbol, "price": round(price, 2), "price_date": rows[-1]["date"],
            "upper": round(upper, 2), "middle": round(cur_mid, 2), "lower": round(lower, 2),
            "bw": bw, "pct_b": pct_b, "squeeze": squeeze, "signal": signal}


def run_bb_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _bb_cache
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
