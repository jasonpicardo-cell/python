"""
nse_adx_scanner.py — ADX / Trend Strength Scanner (period 14)

+DI = 100 × RMA(+DM, 14) / ATR14
-DI = 100 × RMA(-DM, 14) / ATR14
DX  = 100 × |+DI − -DI| / (+DI + -DI)
ADX = RMA(DX, 14)

Strength bands:
  Strong Trend  ADX ≥ 40
  Trending      ADX 25–40
  Weak Trend    ADX 15–25
  Ranging       ADX < 15

Direction: Bullish (+DI > -DI) or Bearish (+DI < -DI)
Fresh crossover detected in last 3 bars.
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list
from nse_indicators import rma, atr

_adx_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=60)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 30: return None

    highs  = [r["H"] for r in rows]
    lows   = [r["L"] for r in rows]
    closes = [r["C"] for r in rows]
    n = len(closes)

    # True Range and DM
    tr_s, pdm_s, ndm_s = [], [], []
    tr_s.append(highs[0] - lows[0])
    pdm_s.append(0.0); ndm_s.append(0.0)
    for i in range(1, n):
        tr_s.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
        up, dn = highs[i]-highs[i-1], lows[i-1]-lows[i]
        pdm_s.append(up if (up > dn and up > 0) else 0.0)
        ndm_s.append(dn if (dn > up and dn > 0) else 0.0)

    period = 14
    atr14  = rma(tr_s, period)
    pdi14  = rma(pdm_s, period)
    ndi14  = rma(ndm_s, period)

    di_p_s: list[float | None] = []
    di_n_s: list[float | None] = []
    dx_s:   list[float | None] = []

    for i in range(n):
        a = atr14[i]; pp = pdi14[i]; nn = ndi14[i]
        if a and a > 0:
            dp = 100 * pp / a; dn_v = 100 * nn / a  # type: ignore
            di_p_s.append(dp); di_n_s.append(dn_v)
            s = dp + dn_v
            dx_s.append(100 * abs(dp - dn_v) / s if s else 0.0)
        else:
            di_p_s.append(None); di_n_s.append(None); dx_s.append(None)

    valid_dx = [v for v in dx_s if v is not None]
    if len(valid_dx) < period: return None
    adx_valid = rma(valid_dx, period)

    cur_adx = next((v for v in reversed(adx_valid) if v is not None), None)
    cur_dip = next((v for v in reversed(di_p_s) if v is not None), None)
    cur_din = next((v for v in reversed(di_n_s) if v is not None), None)
    if cur_adx is None: return None

    if   cur_adx >= 40: strength = "Strong Trend"
    elif cur_adx >= 25: strength = "Trending"
    elif cur_adx >= 15: strength = "Weak Trend"
    else:               strength = "Ranging"

    direction = "Bullish" if (cur_dip and cur_din and cur_dip > cur_din) else "Bearish"

    # Fresh DI crossover in last 3 bars
    dp_vals = [v for v in di_p_s if v is not None]
    dn_vals = [v for v in di_n_s if v is not None]
    fresh = False
    if len(dp_vals) >= 4 and len(dn_vals) >= 4:
        signs = [1 if a > b else -1 for a, b in zip(dp_vals[-4:], dn_vals[-4:])]
        fresh = signs[-1] != signs[0]

    return {"symbol": symbol, "price": round(closes[-1], 2), "price_date": rows[-1]["date"],
            "adx": round(cur_adx, 2),
            "di_plus": round(cur_dip, 2) if cur_dip else None,
            "di_minus": round(cur_din, 2) if cur_din else None,
            "strength": strength, "direction": direction, "fresh": fresh}


def run_adx_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _adx_cache
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
