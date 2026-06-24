#!/usr/bin/env python3
"""
nse_fno_movers.py
==================

Fetch today's top F&O gainers and losers, updated automatically in the background.

Design:
- A daemon thread (start_background_worker) fetches data every REFRESH_INTERVAL
  seconds using the warmed NSESession from the main chain server.
- The /api/fno-movers endpoint just reads the cache — instant response.
- On first boot the cache is empty; data appears ~5s after the first
  successful option-chain fetch warms the session.

API strategy (same Referer as VIX — the one the warmed session expects):
  Primary:  equity-stockIndices?index=NIFTY%2050     (50 liquid stocks)
  Fallback: equity-stockIndices?index=NIFTY%20BANK   (bank stocks)
  Both filtered to F&O symbols via the lot-size list.
"""

from __future__ import annotations

import threading
import time
from typing import Any

REFRESH_INTERVAL = 90        # seconds between background fetches
_CACHE_TTL = 180             # treat cache as stale after 3 minutes

_cache: dict[str, Any] = {"ts": 0, "data": {}, "error": None}
_lock = threading.Lock()
_worker_thread: threading.Thread | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def start_background_worker(get_shared_fetcher, lot_size_fetcher) -> None:
    """
    Launch the background refresh thread.  Call once at server start.

    get_shared_fetcher: a zero-argument callable that returns the most
        recently warmed NSESession (from nse_chain_server._shared_fetcher).
        Returns None if no chain has been fetched yet.
    lot_size_fetcher: the nse_lot_sizes module (for filtering F&O symbols).
    """
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return  # already running

    def _run():
        while True:
            try:
                fetcher = get_shared_fetcher()
                if fetcher is not None and fetcher._warmed:
                    _fetch_and_cache(fetcher, lot_size_fetcher)
                # If not warmed yet, just wait — the chain server will warm it
            except Exception as e:  # noqa: BLE001
                with _lock:
                    _cache["error"] = str(e)
            time.sleep(REFRESH_INTERVAL)

    _worker_thread = threading.Thread(target=_run, daemon=True, name="nse-movers-bg")
    _worker_thread.start()


def get_movers() -> dict:
    """Return the cached movers (or error state). Non-blocking."""
    with _lock:
        if not _cache["data"] and not _cache["error"]:
            return {
                "gainers": [], "losers": [], "as_of": None,
                "source": "pending",
                "error": "Fetching — will appear within 90 s after the first option chain loads.",
            }
        if _cache["error"] and not _cache["data"]:
            return {
                "gainers": [], "losers": [], "as_of": None,
                "source": "error",
                "error": _cache["error"],
            }
        age = time.time() - _cache["ts"]
        source = "live" if age < REFRESH_INTERVAL + 10 else f"cached ({int(age)}s old)"
        return {**_cache["data"], "source": source}


# ──────────────────────────────────────────────────────────────────────────────
# Internal
# ──────────────────────────────────────────────────────────────────────────────

# Import the exact constants used by the option-chain code so our headers
# are byte-for-byte identical to what the warmed session already sent.
def _get_nse_constants():
    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
    return API_HEADERS, NSE_OC_PAGE


def _api_get(session, url: str) -> Any:
    """GET url with the same headers/referer that VIX and option-chain use."""
    API_HEADERS, NSE_OC_PAGE = _get_nse_constants()
    headers = dict(API_HEADERS)
    headers["Referer"] = NSE_OC_PAGE   # ← same referer the warmed session expects
    r = session.get(url, headers=headers, timeout=12)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} from {url.split('?')[0]}")
    return r.json()


def _fetch_and_cache(fetcher, lot_size_fetcher) -> None:
    try:
        fno_symbols = set(lot_size_fetcher.get_fno_symbol_list())
    except Exception:
        fno_symbols = set()

    data = None
    last_err = None

    # Try NIFTY 50, then NIFTY BANK as fallback
    for index in ("NIFTY%2050", "NIFTY%20BANK", "NIFTY%20500"):
        try:
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={index}"
            payload = _api_get(fetcher.session, url)
            items = payload.get("data", [])
            # Filter out the index row itself
            stocks = [it for it in items
                      if it.get("symbol") not in ("NIFTY 50", "NIFTY BANK", "NIFTY 500")
                      and it.get("lastPrice", 0) > 0]
            if stocks:
                data = _build_result(stocks, fno_symbols)
                break
        except Exception as e:
            last_err = e
            continue

    if data is None:
        # Also try the live-analysis-variations endpoint (may work on some sessions)
        try:
            gainers_raw = _api_get(fetcher.session,
                "https://www.nseindia.com/api/live-analysis-variations?index=gainers"
            ).get("data", [])
            losers_raw = _api_get(fetcher.session,
                "https://www.nseindia.com/api/live-analysis-variations?index=losers"
            ).get("data", [])
            if gainers_raw or losers_raw:
                data = _build_from_variations(gainers_raw, losers_raw, fno_symbols)
        except Exception as e:
            last_err = e

    with _lock:
        if data:
            _cache["ts"] = time.time()
            _cache["data"] = data
            _cache["error"] = None
        else:
            _cache["error"] = (
                f"All NSE endpoints failed ({last_err}). "
                "This usually clears itself — wait 90 s for the next retry."
            )


def _to_mover(item: dict, fno_symbols: set[str]) -> dict:
    sym = item.get("symbol", "").upper()
    return {
        "symbol": sym,
        "ltp": float(item.get("lastPrice") or item.get("ltp") or 0),
        "prev_close": float(item.get("previousClose") or item.get("previousPrice") or 0),
        "pct_change": float(item.get("pChange") or item.get("perChange") or 0),
        "volume": int(item.get("totalTradedVolume") or item.get("tradedQuantity") or 0),
        "is_fno": bool(not fno_symbols or sym in fno_symbols),
    }


def _build_result(stocks: list[dict], fno_symbols: set[str]) -> dict:
    movers = [_to_mover(s, fno_symbols) for s in stocks]
    if fno_symbols:
        movers = [m for m in movers if m["is_fno"]]
    gainers = sorted(movers, key=lambda m: m["pct_change"], reverse=True)[:5]
    losers = sorted(movers, key=lambda m: m["pct_change"])[:5]
    return {"gainers": gainers, "losers": losers, "as_of": time.strftime("%H:%M:%S")}


def _build_from_variations(gainers_raw, losers_raw, fno_symbols) -> dict:
    def process(raw, reverse):
        movers = [_to_mover(it, fno_symbols) for it in raw]
        if fno_symbols:
            movers = [m for m in movers if m["is_fno"]]
        return sorted(movers, key=lambda m: m["pct_change"], reverse=reverse)[:5]
    return {
        "gainers": process(gainers_raw, True),
        "losers": process(losers_raw, False),
        "as_of": time.strftime("%H:%M:%S"),
    }
