"""
nse_sr_scanner.py — Support / Resistance Cluster Scanner

Algorithm:
  1. Find all local highs and lows from last 100 rows using a
     Williams-style fractal look (N bars on each side).
  2. Cluster nearby levels: any two levels within 0.5% of each
     other belong to the same cluster.
  3. Rank clusters by touch count (number of pivot points).
  4. Return top 3 resistance clusters and top 3 support clusters.
  5. Show distance of current price to the nearest S and R level.
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_sr_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300
_FRACTAL_N   = 3     # bars on each side for fractal detection
_CLUSTER_PCT = 0.005 # 0.5% cluster radius
_TOP_N       = 3     # top clusters to return


def _find_pivots(rows: list[dict], n: int = _FRACTAL_N) -> tuple[list[float], list[float]]:
    """Returns (highs, lows) as flat lists of pivot prices."""
    highs, lows = [], []
    for i in range(n, len(rows) - n):
        h = rows[i]["H"]; l = rows[i]["L"]
        if all(rows[i-k]["H"] <= h for k in range(1, n+1)) and \
           all(rows[i+k]["H"] <= h for k in range(1, n+1)):
            highs.append(h)
        if all(rows[i-k]["L"] >= l for k in range(1, n+1)) and \
           all(rows[i+k]["L"] >= l for k in range(1, n+1)):
            lows.append(l)
    return highs, lows


def _cluster(levels: list[float], pct: float = _CLUSTER_PCT) -> list[dict]:
    """Group nearby levels into clusters, return sorted by touch count desc."""
    if not levels: return []
    s = sorted(levels)
    groups: list[list[float]] = [[s[0]]]
    for v in s[1:]:
        if abs(v - groups[-1][-1]) / groups[-1][-1] <= pct:
            groups[-1].append(v)
        else:
            groups.append([v])
    clusters = [{"level": round(sum(g)/len(g), 2), "touches": len(g)} for g in groups]
    return sorted(clusters, key=lambda x: -x["touches"])


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=110)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 20: return None

    price = rows[-1]["C"]
    pivot_highs, pivot_lows = _find_pivots(rows)
    if not pivot_highs and not pivot_lows: return None

    res_clusters = [c for c in _cluster(pivot_highs) if c["level"] > price][:_TOP_N]
    sup_clusters = [c for c in _cluster(pivot_lows)  if c["level"] < price][:_TOP_N]

    nearest_r = min((c["level"] for c in res_clusters), default=None)
    nearest_s = max((c["level"] for c in sup_clusters), default=None)
    dist_r = round((nearest_r - price) / price * 100, 2) if nearest_r else None
    dist_s = round((price - nearest_s) / price * 100, 2) if nearest_s else None

    # Signal based on proximity to nearest S or R
    if   nearest_r and dist_r is not None and dist_r <= 1.0: signal = "Near Resistance"
    elif nearest_s and dist_s is not None and dist_s <= 1.0: signal = "Near Support"
    elif (nearest_r and dist_r is not None and dist_r <= 2.5) or \
         (nearest_s and dist_s is not None and dist_s <= 2.5): signal = "At Cluster"
    else:                                                       signal = "Mid Range"

    return {"symbol": symbol, "price": round(price, 2), "price_date": rows[-1]["date"],
            "resistance": res_clusters, "support": sup_clusters,
            "nearest_r": nearest_r, "nearest_s": nearest_s,
            "dist_r": dist_r, "dist_s": dist_s, "signal": signal}


def run_sr_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _sr_cache
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
