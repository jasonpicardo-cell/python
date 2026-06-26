"""
nse_correlation_scanner.py — 20-Day Rolling Correlation with Nifty

Computes Pearson correlation of each stock's daily returns
against Nifty 50 daily returns over the last 20 trading days.

Correlation bands:
  High         r ≥ 0.80   (moves closely with index)
  Moderate     r 0.50–0.80
  Low          r 0.20–0.50
  Independent  r < 0.20   (stock-specific driver)
  Negative     r < 0      (inverse to index)

The Nifty CSV is looked up as: NIFTY, NIFTY50, NIFTY_50, ^NSEI
(first match in nse_data_cache).
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list, DATA_DIR
from nse_indicators import pearson_r

_corr_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300
_PERIOD      = 20     # rolling days


def _load_nifty_returns() -> dict[str, float] | None:
    """Returns {date_str: return_pct} for NIFTY from CSV."""
    for name in ("NIFTY", "NIFTY50", "NIFTY_50", "^NSEI", "INDIA50", "NIFTY 50"):
        p = _find_csv(name)
        if p:
            rows, err = _read_rows(p, n=40)
            if not err and rows:
                today = date.today().isoformat()
                rows  = [r for r in rows if r["date"] < today]
                if len(rows) >= 2:
                    result = {}
                    for i in range(1, len(rows)):
                        if rows[i-1]["C"]:
                            result[rows[i]["date"]] = (rows[i]["C"] - rows[i-1]["C"]) / rows[i-1]["C"]
                    return result
    return None


def scan_stock(symbol: str, nifty_rets: dict[str, float]) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=30)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < _PERIOD + 1: return None

    # Compute daily returns
    stock_rets: dict[str, float] = {}
    for i in range(1, len(rows)):
        if rows[i-1]["C"]:
            stock_rets[rows[i]["date"]] = (rows[i]["C"] - rows[i-1]["C"]) / rows[i-1]["C"]

    # Align with Nifty on common dates (last _PERIOD)
    common = sorted(set(stock_rets) & set(nifty_rets))[-_PERIOD:]
    if len(common) < 10: return None

    xs = [stock_rets[d] for d in common]
    ys = [nifty_rets[d] for d in common]
    r  = pearson_r(xs, ys)
    if r is None: return None

    if   r >= 0.80: signal = "High"
    elif r >= 0.50: signal = "Moderate"
    elif r >= 0.20: signal = "Low"
    elif r >= 0.00: signal = "Independent"
    else:           signal = "Negative"

    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"],
            "correlation": r, "signal": signal,
            "days_used": len(common)}


def run_correlation_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _corr_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]

    nifty_rets = _load_nifty_returns()
    if nifty_rets is None:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,
                "source":source,"as_of":time.strftime("%H:%M:%S"),
                "error":"Nifty CSV not found — tried NIFTY, NIFTY50, NIFTY_50, ^NSEI"}

    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,
                "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),
                "error":f"Symbol list empty for {source!r}"}

    stocks, errors, skipped = [], [], []
    for sym in symbols:
        try:
            r = scan_stock(sym, nifty_rets)
            (stocks if r else skipped).append(r or sym)
        except Exception as e:
            errors.append({"symbol": sym, "reason": str(e)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,"count":len(stocks),
            "total":len(symbols),"source":source,"sym_label":sym_label,
            "as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
