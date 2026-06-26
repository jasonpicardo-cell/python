"""
nse_52week_scanner.py
=====================
52-Week High / Low Proximity Scanner.

For each stock reads the last 252 trading rows (≈1 year) from CSV
and computes position relative to the yearly high and low.

Position labels:
  Near High   – within 3% below the 52W high
  Near Low    – within 5% above the 52W low
  Mid Range   – everything else

Range Position (0–100%): where price sits inside the 52W band.
  100% = at the 52W high, 0% = at the 52W low.

All calculations are purely CSV-based — no NSE API calls needed.
"""
from __future__ import annotations

import logging
import time
from datetime import date

from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)

_wk_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300

# Thresholds for "near" classification
_NEAR_HIGH_PCT  = 3.0   # within 3% below 52W high
_NEAR_LOW_PCT   = 5.0   # within 5% above 52W low


def scan_stock(symbol: str) -> dict | None:
    """
    Compute 52-week proximity metrics for one stock.
    Returns None if CSV missing or < 50 rows of history.
    """
    p = _find_csv(symbol)
    if not p:
        return None

    rows, err = _read_rows(p, n=270)   # 252 + buffer for gaps/holidays
    if err or not rows:
        return None

    today_iso = date.today().isoformat()
    rows      = [r for r in rows if r["date"] < today_iso]

    if len(rows) < 50:
        return None     # too little history for a meaningful 52-week range

    year_rows  = rows[-252:]           # cap at 252 trading days
    last_row   = year_rows[-1]

    price      = last_row["C"]
    price_date = last_row["date"]

    high_52w   = max(r["H"] for r in year_rows)
    low_52w    = min(r["L"] for r in year_rows)

    pct_from_high = round((price - high_52w) / high_52w * 100, 2)   # ≤0
    pct_from_low  = round((price - low_52w)  / low_52w  * 100, 2)   # ≥0

    rng = high_52w - low_52w
    range_pos = round((price - low_52w) / rng * 100, 1) if rng > 0 else 50.0

    # Dates of the 52W extremes
    hi_idx    = max(range(len(year_rows)), key=lambda i: year_rows[i]["H"])
    lo_idx    = min(range(len(year_rows)), key=lambda i: year_rows[i]["L"])
    high_date = year_rows[hi_idx]["date"]
    low_date  = year_rows[lo_idx]["date"]

    # Position label
    if pct_from_high >= -_NEAR_HIGH_PCT:
        position = "Near High"
    elif pct_from_low <= _NEAR_LOW_PCT:
        position = "Near Low"
    else:
        position = "Mid Range"

    # Number of rows used (may be less than 252 for newer listings)
    rows_used = len(year_rows)

    return {
        "symbol":        symbol,
        "price":         round(price, 2),
        "price_date":    price_date,
        "high_52w":      round(high_52w, 2),
        "low_52w":       round(low_52w, 2),
        "pct_from_high": pct_from_high,
        "pct_from_low":  pct_from_low,
        "range_pos":     range_pos,
        "high_date":     high_date,
        "low_date":      low_date,
        "position":      position,
        "rows_used":     rows_used,
    }


def run_52week_scanner(source: str = "niftyfno",
                       force: bool = False,
                       ttl: int = _DEFAULT_TTL) -> dict:
    """Run 52-Week High/Low scanner across all symbols in the chosen list."""
    key = source
    c   = _wk_cache
    if not force and c["data"] and c["key"] == key and (time.time() - c["ts"]) < ttl:
        return c["data"]

    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks": [], "errors": [], "skipped": [], "count": 0, "total": 0,
                "source": source, "sym_label": sym_label,
                "as_of": time.strftime("%H:%M:%S"),
                "error": f"Symbol list empty for source={source!r}"}

    stocks:  list[dict] = []
    errors:  list[dict] = []
    skipped: list[str]  = []

    for sym in symbols:
        try:
            r = scan_stock(sym)
            if r is None:
                skipped.append(sym)
            else:
                stocks.append(r)
        except Exception as exc:   # noqa: BLE001
            errors.append({"symbol": sym, "reason": str(exc)})

    data = {
        "stocks":    stocks,
        "errors":    errors,
        "skipped":   skipped,
        "count":     len(stocks),
        "total":     len(symbols),
        "source":    source,
        "sym_label": sym_label,
        "as_of":     time.strftime("%H:%M:%S"),
    }
    c["ts"] = time.time(); c["data"] = data; c["key"] = key
    return data
