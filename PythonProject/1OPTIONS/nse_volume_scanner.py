"""
nse_volume_scanner.py
=====================
Volume Anomaly Scanner — compares latest session volume against
20-day rolling average to flag unusual activity.

Alert levels:
  Surge  – vol_ratio >= 3.0   🚀
  Spike  – vol_ratio >= 2.0   🔥
  Active – vol_ratio >= 1.5   ⚡
  Normal – vol_ratio 0.5–1.5
  Dry    – vol_ratio <  0.5   💀

All calculations are purely CSV-based — no NSE API calls needed.
"""
from __future__ import annotations

import logging
import time
from datetime import date

from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)

_vol_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300


def _fmt_vol(v: float) -> str:
    """Human-readable volume: 1234567 → '12.35L', 12345678 → '1.23Cr'"""
    if v >= 1_00_00_000:
        return f"{v/1_00_00_000:.2f}Cr"
    if v >= 1_00_000:
        return f"{v/1_00_000:.2f}L"
    if v >= 1_000:
        return f"{v/1_000:.1f}K"
    return str(int(v))


def scan_stock(symbol: str) -> dict | None:
    """
    Compute volume anomaly metrics for one stock.
    Returns None if CSV is missing or has insufficient data.
    """
    p = _find_csv(symbol)
    if not p:
        return None

    rows, err = _read_rows(p, n=30)   # 20 history + today + buffer
    if err or not rows:
        return None

    today_iso = date.today().isoformat()
    rows      = [r for r in rows if r["date"] < today_iso]

    if len(rows) < 5:
        return None

    today_row  = rows[-1]
    prev_row   = rows[-2]

    price      = today_row["C"]
    price_date = today_row["date"]
    price_chg  = round((price - prev_row["C"]) / prev_row["C"] * 100, 2) if prev_row["C"] else 0.0
    today_vol  = today_row.get("V", 0) or 0

    if today_vol == 0:
        return None     # no volume data in this CSV

    # 20-day average from the rows before today's session
    history   = [r.get("V", 0) or 0 for r in rows[:-1]]
    recent_20 = history[-20:]
    avg_vol   = sum(recent_20) / len(recent_20) if recent_20 else 0

    if avg_vol == 0:
        return None

    vol_ratio = round(today_vol / avg_vol, 2)

    if   vol_ratio >= 3.0:  alert = "Surge"
    elif vol_ratio >= 2.0:  alert = "Spike"
    elif vol_ratio >= 1.5:  alert = "Active"
    elif vol_ratio <  0.5:  alert = "Dry"
    else:                   alert = "Normal"

    return {
        "symbol":      symbol,
        "price":       round(price, 2),
        "price_date":  price_date,
        "price_chg":   price_chg,
        "today_vol":   int(today_vol),
        "avg_vol_20":  int(avg_vol),
        "vol_ratio":   vol_ratio,
        "today_vol_h": _fmt_vol(today_vol),
        "avg_vol_h":   _fmt_vol(avg_vol),
        "alert":       alert,
    }


def run_volume_scanner(source: str = "niftyfno",
                       force: bool = False,
                       ttl: int = _DEFAULT_TTL) -> dict:
    """Run Volume Anomaly scanner across all symbols in the chosen list."""
    key = source
    c   = _vol_cache
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
