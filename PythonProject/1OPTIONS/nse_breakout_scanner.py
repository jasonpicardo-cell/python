"""
nse_breakout_scanner.py — N-Day High/Low Breakout Scanner

A "breakout" occurs when today's close exceeds the highest high
of the last N trading days (excluding today).
A "breakdown" occurs when close falls below the lowest low.

Timeframes checked: 10D, 20D, 52W (252 trading days).
The highest-timeframe confirmed breakout is reported.

Also computes volume confirmation ratio (today vol / 20D avg).
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_bko_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=270)   # 252 + buffer
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 22: return None

    price  = rows[-1]["C"]
    history = rows[:-1]   # exclude today's bar for the lookback

    results: list[str] = []

    for period, label in [(10, "10D"), (20, "20D"), (252, "52W")]:
        lookback = history[-period:] if len(history) >= period else history
        hi = max(r["H"] for r in lookback)
        lo = min(r["L"] for r in lookback)
        if   price > hi: results.append(f"{label} Breakout")
        elif price < lo: results.append(f"{label} Breakdown")

    if not results: return None

    # Strongest signal = longest confirmed timeframe
    signal = results[-1]   # last in list = longest period

    # Volume confirmation
    vol_today = rows[-1].get("V", 0) or 0
    vols_20 = [r.get("V", 0) or 0 for r in history[-20:] if (r.get("V") or 0) > 0]
    avg_vol  = sum(vols_20) / len(vols_20) if vols_20 else 0
    vol_ratio = round(vol_today / avg_vol, 2) if avg_vol > 0 and vol_today > 0 else None

    # 52W high / low for context
    yr = history[-252:] if len(history) >= 252 else history
    hi_52w = max(r["H"] for r in yr)
    lo_52w = min(r["L"] for r in yr)
    pct_from_hi = round((price - hi_52w) / hi_52w * 100, 2)
    pct_from_lo = round((price - lo_52w) / lo_52w * 100, 2)

    return {"symbol": symbol, "price": round(price, 2), "price_date": rows[-1]["date"],
            "signal": signal, "all_signals": results,
            "hi_52w": round(hi_52w, 2), "lo_52w": round(lo_52w, 2),
            "pct_from_hi": pct_from_hi, "pct_from_lo": pct_from_lo,
            "vol_ratio": vol_ratio}


def run_breakout_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _bko_cache
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
