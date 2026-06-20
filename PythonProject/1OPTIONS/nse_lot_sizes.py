#!/usr/bin/env python3
"""
nse_lot_sizes.py
==================

Fetches and caches NSE's official F&O lot-size list. Rather than hardcoding
lot sizes (which get revised periodically — see the index lot-size rebase
already baked into nse_strategy_engine.py), this pulls the live CSV NSE
itself publishes, so it's always current and covers every F&O stock, not
just the 4 indices.

Source: https://archives.nseindia.com/content/fo/fo_mktlots.csv
This is the same source nsepython uses. Notably it's on the `archives`
subdomain, not `www.nseindia.com` — in practice this tends to sit behind
much lighter (or no) bot protection than the main site, so a plain request
often works without the curl_cffi TLS-impersonation dance the option chain
needs. We still try with curl_cffi first for consistency/safety, falling
back to plain `requests` if that's unavailable.
"""

from __future__ import annotations

import csv
import io
import time

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    cffi_requests = None
    HAS_CURL_CFFI = False

import requests

LOT_SIZE_CSV_URL = "https://archives.nseindia.com/content/fo/fo_mktlots.csv"
CACHE_TTL_SECONDS = 6 * 3600  # lot sizes change rarely — recheck a few times a day at most

_cache: dict = {"data": None, "fetched_at": 0.0}


def _fetch_raw_csv() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,text/plain,*/*",
    }
    if HAS_CURL_CFFI:
        session = cffi_requests.Session(impersonate="chrome124")
        resp = session.get(LOT_SIZE_CSV_URL, headers=headers, timeout=10)
    else:
        resp = requests.get(LOT_SIZE_CSV_URL, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.text


def get_lot_sizes(force_refresh: bool = False) -> dict[str, int]:
    """Returns {SYMBOL: lot_size} for every NSE F&O instrument (indices +
    stocks). Cached in-process for CACHE_TTL_SECONDS."""
    now = time.time()
    if not force_refresh and _cache["data"] and (now - _cache["fetched_at"]) < CACHE_TTL_SECONDS:
        return _cache["data"]

    raw = _fetch_raw_csv()
    result: dict[str, int] = {}
    reader = csv.reader(io.StringIO(raw))
    for row in reader:
        if len(row) < 3:
            continue
        symbol = row[1].strip().upper()
        lot_str = row[2].strip()
        if not symbol or symbol == "SYMBOL" or not lot_str.isdigit():
            continue
        result[symbol] = int(lot_str)

    if not result:
        raise RuntimeError("Lot size CSV fetched but parsed to zero entries — NSE may have changed the format")
    if "NIFTY" not in result:
        raise RuntimeError(
            f"Lot size CSV parsed {len(result)} entries but 'NIFTY' wasn't among them — "
            f"the column layout has likely changed. Sample parsed keys: {list(result.keys())[:10]}"
        )

    _cache["data"] = result
    _cache["fetched_at"] = now
    return result


def get_lot_size(symbol: str, fallback: int = 1) -> int:
    try:
        sizes = get_lot_sizes()
        return sizes.get(symbol.upper(), fallback)
    except Exception:
        return fallback


def get_fno_symbol_list() -> list[str]:
    """All symbols currently in the F&O list (indices + stocks), sorted."""
    try:
        return sorted(get_lot_sizes().keys())
    except Exception:
        return []


if __name__ == "__main__":
    sizes = get_lot_sizes()
    print(f"Loaded {len(sizes)} F&O symbols")
    for sym in ("NIFTY", "BANKNIFTY", "RELIANCE", "TCS"):
        print(f"  {sym}: {sizes.get(sym, 'NOT FOUND')}")
