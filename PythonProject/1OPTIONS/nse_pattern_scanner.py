"""
nse_pattern_scanner.py — Classic Price Pattern Scanner

Detects from last 100 CSV rows:
  Double Top      two highs within 3%, separated by trough, price below neckline
  Double Bottom   two lows within 3%, separated by peak, price above neckline
  Head & Shoulders     three peaks, middle highest, neckline broken
  Inverse H&S          three troughs, middle lowest, neckline broken

Each pattern carries a bias (Bullish / Bearish).
"""
from __future__ import annotations
import time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

_pat_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300
_FRACTAL_N   = 3     # bars on each side for local extremes
_SIM_PCT     = 0.03  # 3% similarity threshold for double patterns


def _local_highs(rows, n=_FRACTAL_N) -> list[tuple[int, float]]:
    result = []
    for i in range(n, len(rows) - n):
        h = rows[i]["H"]
        if all(rows[i-k]["H"] <= h for k in range(1, n+1)) and \
           all(rows[i+k]["H"] <= h for k in range(1, n+1)):
            result.append((i, h))
    return result


def _local_lows(rows, n=_FRACTAL_N) -> list[tuple[int, float]]:
    result = []
    for i in range(n, len(rows) - n):
        l = rows[i]["L"]
        if all(rows[i-k]["L"] >= l for k in range(1, n+1)) and \
           all(rows[i+k]["L"] >= l for k in range(1, n+1)):
            result.append((i, l))
    return result


def detect_patterns(rows: list[dict]) -> list[dict]:
    """Returns list of {name, bias, confidence} dicts."""
    if len(rows) < 15: return []
    price    = rows[-1]["C"]
    patterns = []
    lh = _local_highs(rows)
    ll = _local_lows(rows)

    # ── Double Top ──────────────────────────────────────────────────────
    if len(lh) >= 2:
        h1i, h1 = lh[-2]; h2i, h2 = lh[-1]
        if abs(h1 - h2) / max(h1, h2) <= _SIM_PCT and h2i > h1i:
            # trough between them
            between_lows  = [r["L"] for r in rows[h1i:h2i+1]]
            neckline      = min(between_lows)
            if price < neckline:
                conf = round((1 - abs(h1-h2)/max(h1,h2)) * 100)
                patterns.append({"name":"Double Top","bias":"Bearish","confidence":conf,
                                  "key_level": round(neckline,2)})

    # ── Double Bottom ───────────────────────────────────────────────────
    if len(ll) >= 2:
        l1i, l1 = ll[-2]; l2i, l2 = ll[-1]
        if abs(l1 - l2) / min(l1, l2) <= _SIM_PCT and l2i > l1i:
            between_highs = [r["H"] for r in rows[l1i:l2i+1]]
            neckline      = max(between_highs)
            if price > neckline:
                conf = round((1 - abs(l1-l2)/min(l1,l2)) * 100)
                patterns.append({"name":"Double Bottom","bias":"Bullish","confidence":conf,
                                  "key_level": round(neckline,2)})

    # ── Head & Shoulders ────────────────────────────────────────────────
    if len(lh) >= 3:
        li, lv = lh[-3]; hi, hv = lh[-2]; ri, rv = lh[-1]
        if hv > lv and hv > rv and abs(lv-rv)/max(lv,rv) <= 0.05:
            # neckline = avg of troughs between shoulders
            tl = [r["L"] for r in rows[li:hi+1]]
            tr = [r["L"] for r in rows[hi:ri+1]]
            neckline = (min(tl) + min(tr)) / 2
            if price < neckline:
                patterns.append({"name":"Head & Shoulders","bias":"Bearish","confidence":75,
                                  "key_level": round(neckline,2)})

    # ── Inverse H&S ─────────────────────────────────────────────────────
    if len(ll) >= 3:
        li, lv = ll[-3]; hi, hv = ll[-2]; ri, rv = ll[-1]
        if hv < lv and hv < rv and abs(lv-rv)/min(lv,rv) <= 0.05:
            th = [r["H"] for r in rows[li:hi+1]]
            tr = [r["H"] for r in rows[hi:ri+1]]
            neckline = (max(th) + max(tr)) / 2
            if price > neckline:
                patterns.append({"name":"Inverse H&S","bias":"Bullish","confidence":75,
                                  "key_level": round(neckline,2)})

    return patterns


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=110)
    if err or not rows: return None
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if len(rows) < 15: return None

    patterns = detect_patterns(rows)
    if not patterns: return None

    biases = [pt["bias"] for pt in patterns]
    overall = "Bullish" if biases.count("Bullish") >= biases.count("Bearish") else "Bearish"

    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"],
            "patterns": patterns, "bias": overall}


def run_pattern_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _pat_cache
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
