"""
nse_stoch_scanner.py — Stochastic Oscillator (14, 3, 3)
%K = (close - LL14) / (HH14 - LL14) * 100
%D = SMA3 of %K
Overbought ≥ 80, Oversold ≤ 20
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import sma

_stoch_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=30)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 17: return None

    k_raw = []
    for i in range(13, len(rows)):
        hh = max(r["H"] for r in rows[i-13:i+1])
        ll = min(r["L"] for r in rows[i-13:i+1])
        k_raw.append(100 * (rows[i]["C"] - ll) / (hh - ll) if hh != ll else 50.0)

    if len(k_raw) < 3: return None
    k_vals = [v for v in sma(k_raw, 3) if v is not None]
    if len(k_vals) < 3: return None

    cur_k = k_raw[-1]; d_vals = sma(k_raw, 3)
    cur_d = next((v for v in reversed(d_vals) if v is not None), None)
    if cur_d is None: return None

    cur_k = round(cur_k, 2); cur_d = round(cur_d, 2)

    # Fresh crossover (last 3 bars)
    fresh_bull = cur_k > cur_d and len(k_raw) >= 4 and k_raw[-2] <= (d_vals[-2] or k_raw[-2])
    fresh_bear = cur_k < cur_d and len(k_raw) >= 4 and k_raw[-2] >= (d_vals[-2] or k_raw[-2])

    if   cur_k >= 80 and cur_d >= 80:   signal = "Overbought"
    elif cur_k <= 20 and cur_d <= 20:   signal = "Oversold"
    elif fresh_bull and cur_k < 80:     signal = "Bull Cross"
    elif fresh_bear and cur_k > 20:     signal = "Bear Cross"
    elif cur_k > cur_d:                 signal = "Bullish"
    else:                               signal = "Bearish"

    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"],
            "k": cur_k, "d": cur_d, "signal": signal}


def run_stoch_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _stoch_cache
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
