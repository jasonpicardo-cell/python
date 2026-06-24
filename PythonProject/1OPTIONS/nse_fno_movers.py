#!/usr/bin/env python3
"""
nse_fno_movers.py  — F&O Top Movers
======================================

Two complementary data sources, both using only endpoints we know work:

  1. allIndices  — the same call used for India VIX.  Contains every NSE
                   index including NIFTY BANK, NIFTY FIN SERVICE, NIFTY MID
                   SELECT, NIFTY IT, NIFTY AUTO, etc.  We keep the F&O ones.
                   This is the PRIMARY source for index options movers.

  2. Tracked spots — every time nse_chain_server fetches an option chain for
                     any symbol, it calls record_spot_price().  We store the
                     session-open price (first seen today) and the latest price,
                     then compute intraday % change.  This gives individual
                     stock movers for whatever symbols the user has loaded.

No separate "movers" NSE API call is made — they were unreliable (returning
JSON strings instead of dicts, or requiring page-specific cookies).  Instead
we derive movers from endpoints that already work.
"""

from __future__ import annotations

import threading
import time
from typing import Any

REFRESH_INTERVAL = 90   # seconds between background refreshes

_lock = threading.Lock()
_state: dict[str, Any] = {
    "ts": 0,
    "data": {},
    "error": None,
    # symbol → {date, open, current, ts}  — populated by record_spot_price()
    "tracked_spots": {},
}
_worker_thread: threading.Thread | None = None

# F&O index names as they appear in the allIndices response
_FNO_INDEX_NAMES = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FIN SERVICE": "FINNIFTY",
    "NIFTY MID SELECT": "MIDCPNIFTY",
    "NIFTY IT": "NIFTYIT",
    "NIFTY AUTO": "NIFTYAUTO",
    "NIFTY PHARMA": "NIFTYPHARMA",
    "NIFTY FMCG": "NIFTYFMCG",
    "NIFTY METAL": "NIFTYMETAL",
    "NIFTY REALTY": "NIFTYREALTY",
    "NIFTY ENERGY": "NIFTYENERGY",
}


# ── Public API ────────────────────────────────────────────────────────────────

def record_spot_price(symbol: str, spot: float) -> None:
    """
    Called by nse_chain_server after every successful option-chain fetch.
    Stores today's open price (first observation) and the latest price so
    we can compute intraday % change without hitting any extra NSE endpoint.
    """
    today = time.strftime("%Y-%m-%d")
    sym = symbol.upper()
    with _lock:
        tracked = _state["tracked_spots"]
        entry = tracked.get(sym)
        if entry is None or entry.get("date") != today:
            tracked[sym] = {"date": today, "open": spot, "current": spot, "ts": time.time()}
        else:
            entry["current"] = spot
            entry["ts"] = time.time()


def start_background_worker(get_shared_fetcher, lot_size_fetcher) -> None:
    """Launch the 90-second background refresh thread.  Call once at server start."""
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return

    def _run():
        while True:
            try:
                fetcher = get_shared_fetcher()
                if fetcher is not None and fetcher._warmed:
                    _fetch_and_cache(fetcher, lot_size_fetcher)
            except Exception as exc:  # noqa: BLE001
                with _lock:
                    _state["error"] = f"Background worker error: {exc}"
            time.sleep(REFRESH_INTERVAL)

    _worker_thread = threading.Thread(target=_run, daemon=True, name="nse-movers-bg")
    _worker_thread.start()


def get_movers() -> dict:
    """Return the latest movers data.  Non-blocking — reads the cache."""
    with _lock:
        if not _state["data"] and not _state["error"]:
            return {
                "gainers": [], "losers": [], "as_of": None,
                "source": "pending",
                "error": (
                    "Warming up — movers appear within 90 s after the first "
                    "option chain loads."
                ),
            }
        if _state["error"] and not _state["data"]:
            return {
                "gainers": [], "losers": [], "as_of": None,
                "source": "error",
                "error": _state["error"],
            }
        age = int(time.time() - _state["ts"])
        return {**_state["data"], "source": f"live ({age}s ago)"}


# ── Internal ──────────────────────────────────────────────────────────────────

def _safe_json(r) -> Any:
    """Parse JSON and validate it's a dict; raise clearly if not."""
    try:
        payload = r.json()
    except Exception as exc:
        raise RuntimeError(
            f"JSON parse failed (status={r.status_code}, "
            f"body={r.text[:80]!r})"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected JSON object, got {type(payload).__name__}: "
            f"{str(payload)[:80]!r}"
        )
    return payload


def _api_headers():
    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
    h = dict(API_HEADERS)
    h["Referer"] = NSE_OC_PAGE   # same referer the warmed session already used
    return h


def _fetch_index_movers(session) -> list[dict]:
    """
    Use allIndices (known working — same as VIX) to get F&O index movers.
    Returns a list of mover dicts suitable for _rank().
    """
    from nse_options_strategy import NSE_ALL_INDICES_API
    r = session.get(NSE_ALL_INDICES_API, headers=_api_headers(), timeout=12)
    if r.status_code != 200:
        raise RuntimeError(f"allIndices HTTP {r.status_code}")
    payload = _safe_json(r)
    rows = payload.get("data", [])
    movers = []
    for row in rows:
        idx_name = row.get("index") or row.get("indexSymbol") or ""
        symbol = _FNO_INDEX_NAMES.get(idx_name)
        if not symbol:
            continue
        pct = float(row.get("percentChange") or row.get("variation", 0))
        last = float(row.get("last") or row.get("lastPrice") or 0)
        movers.append({
            "symbol": symbol,
            "ltp": last,
            "prev_close": round(last / (1 + pct / 100), 2) if pct != -100 else 0,
            "pct_change": pct,
            "volume": 0,
            "is_fno": True,
        })
    return movers


def _fetch_stock_movers_from_tracked() -> list[dict]:
    """
    Derive stock movers from the option-chain spot prices the server has
    already collected today — no extra NSE call required.
    """
    today = time.strftime("%Y-%m-%d")
    movers = []
    with _lock:
        for sym, entry in _state["tracked_spots"].items():
            if entry.get("date") != today:
                continue
            op = entry.get("open") or 0
            cur = entry.get("current") or 0
            if op <= 0 or cur <= 0:
                continue
            # Skip index symbols already covered by allIndices
            if sym in _FNO_INDEX_NAMES.values() or sym in {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}:
                continue
            pct = (cur - op) / op * 100
            movers.append({
                "symbol": sym,
                "ltp": cur,
                "prev_close": round(op, 2),
                "pct_change": round(pct, 2),
                "volume": 0,
                "is_fno": True,
            })
    return movers


def _rank(movers: list[dict], top_n: int = 5) -> tuple[list, list]:
    gainers = sorted(movers, key=lambda m: m["pct_change"], reverse=True)[:top_n]
    losers  = sorted(movers, key=lambda m: m["pct_change"])[:top_n]
    return gainers, losers


def _fetch_and_cache(fetcher, lot_size_fetcher) -> None:
    all_movers: list[dict] = []
    errors: list[str] = []

    # Source 1: allIndices — F&O index movers (reliable, same session)
    try:
        all_movers.extend(_fetch_index_movers(fetcher.session))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"allIndices: {exc}")

    # Source 2: tracked spot prices from option chain fetches (zero extra calls)
    try:
        all_movers.extend(_fetch_stock_movers_from_tracked())
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tracked_spots: {exc}")

    if not all_movers:
        with _lock:
            _state["error"] = (
                "No mover data yet. "
                + (f"Errors: {'; '.join(errors)}. " if errors else "")
                + "Load at least one option chain so the session warms up."
            )
        return

    gainers, losers = _rank(all_movers)
    data = {
        "gainers": gainers,
        "losers": losers,
        "as_of": time.strftime("%H:%M:%S"),
        "note": "Index movers via allIndices; stock movers from loaded chains today." + (
            f" Partial errors: {'; '.join(errors)}" if errors else ""
        ),
    }
    with _lock:
        _state["ts"] = time.time()
        _state["data"] = data
        _state["error"] = None
