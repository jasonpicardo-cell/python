#!/usr/bin/env python3
"""
nse_chain_server.py
====================

A tiny local HTTP API that wraps nse_options_strategy.py's fetcher and
analysis logic, so the interactive HTML dashboard (nse_dashboard.html) can
pull live option-chain data and strategy reads via a normal browser fetch().

WHY THIS EXISTS
---------------
A webpage cannot call NSE's API directly: browsers enforce CORS, and
NSE doesn't allow cross-origin requests from arbitrary pages. The TLS-
impersonation trick (curl_cffi) that gets past NSE's bot detection only
works in a real Python process too — JS in a browser can't fake its own
TLS handshake. So this script runs ON YOUR MACHINE, does the actual NSE
fetch + analysis server-side (exactly like nse_options_strategy.py does),
and serves the result as JSON with permissive CORS headers so the
dashboard (opened as a local HTML file, a different "origin") can read it.

REQUIRES
--------
    nse_options_strategy.py, nse_strategy_engine.py, nse_lot_sizes.py,
    nse_history_store.py, and nse_alerts.py must all be in the SAME FOLDER
    as this file.

USAGE
-----
    pip install requests curl_cffi
    python3 nse_chain_server.py                 # serves on http://127.0.0.1:8765
    python3 nse_chain_server.py --port 9000      # custom port

Then open nse_dashboard.html in your browser (just double-click it).
On first run this also writes a template alert_config.json (Telegram
alerts) — see nse_alerts.py's docstring for setup. Historical snapshots
get written to ./history/ automatically, no setup needed.

ENDPOINTS
---------
    GET /api/chain?symbol=NIFTY&expiry=24-Jun-2026&band=12
        -> full JSON: spot, atm, pcr, max_pain, support/resistance walls,
           sentiment, OI build-up flags, the FULL strike chain (`strikes`,
           not band-limited — `band` only affects support/resistance/flag
           detection), `strategies` (sorted by POP), and `india_vix`.
           `symbol` can be ANY NSE F&O symbol now — index or stock, not
           just the original 4 indices. `expiry` is optional (defaults to
           nearest). `band` is optional (defaults to 12).

    GET /api/history?symbol=NIFTY&days=1
        -> {"points": [{"t":..., "spot":..., "pcr":..., "atm_iv":...,
           "max_pain":..., "support":..., "resistance":..., "india_vix":...}]}
           Persisted across server restarts and browser reloads.

    GET /api/vix
        -> {"india_vix": 13.28}

    GET /api/fno-symbols
        -> {"symbols": [...], "count": N} — every symbol currently in
           NSE's F&O list, fetched live (not hardcoded).

    GET /api/health
        -> {"status": "ok"} — quick check that the server is up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

try:
    from nse_options_strategy import (
        NSESession,
        NSEFetchError,
        parse_chain,
        find_atm_strike,
        infer_strike_gap,
        support_resistance,
        compute_pcr,
        compute_payout_distribution,
        classify_buildups,
        iv_skew_read,
        days_to_expiry,
        generate_strategies,
    )
    from nse_strategy_engine import build_strategy_list, LOT_SIZES
    import nse_lot_sizes
    import nse_history_store
    import nse_alerts
    import nse_paper_trades
    import nse_drafts
    import nse_span
    import nse_fno_movers
except ImportError as e:
    print(
        f"[ERROR] Could not import a required module ({e}) — make sure "
        "nse_options_strategy.py, nse_strategy_engine.py, nse_lot_sizes.py, "
        "nse_history_store.py, and nse_alerts.py are all in the SAME FOLDER "
        "as nse_chain_server.py.",
        file=sys.stderr,
    )
    sys.exit(1)

# Simple in-process guard so a misbehaving frontend (e.g. an over-eager
# auto-refresh) can't hammer NSE faster than its rate limit tolerates.
# Tuned for a 5s frontend poll: cache expires just under that interval so
# steady polling gets fresh data almost every time, while a burst of rapid
# requests (multiple tabs, quick expiry/symbol switching) still gets
# throttled. NOTE: polling NSE every 5s is meaningfully more aggressive
# than the original 30s default — if you see renewed 403s after running
# this for a while, that's NSE's rate limiting kicking back in; back off
# the frontend's auto-refresh interval if so.
MIN_SECONDS_BETWEEN_FETCHES = 3.0
CACHE_TTL_SECONDS = 4.0
_last_fetch_time = 0.0
_cache: dict[str, tuple[float, dict]] = {}

# A single module-level NSESession that stays warm across requests.
# The option-chain endpoint populates it on every successful fetch.
# The movers endpoint reuses it to avoid paying the 2-second warm-up
# cost on a session that's already live.
_shared_fetcher: NSESession | None = None
_shared_fetcher_ts: float = 0.0
_SHARED_FETCHER_MAX_AGE = 270  # seconds — NSE cookies typically expire in ~5 min

# India VIX is symbol-independent (same value regardless of which index/stock
# you're viewing) and changes slowly relative to the option chain — cache it
# separately with a longer TTL so we're not re-fetching it on every poll.
VIX_CACHE_TTL_SECONDS = 15.0
_vix_cache: dict = {"value": None, "fetched_at": 0.0}


def _get_india_vix(fetcher: NSESession) -> float | None:
    """Best-effort VIX fetch — returns None on failure rather than breaking
    the whole /api/chain response, since VIX is a nice-to-have overlay, not
    core to the dashboard."""
    now = time.time()
    if _vix_cache["value"] is not None and (now - _vix_cache["fetched_at"]) < VIX_CACHE_TTL_SECONDS:
        return _vix_cache["value"]
    try:
        vix = fetcher.get_india_vix()
        _vix_cache["value"] = vix
        _vix_cache["fetched_at"] = now
        return vix
    except NSEFetchError as e:
        print(f"[!] India VIX fetch failed (non-fatal, chain data unaffected): {e}")
        return _vix_cache["value"]  # serve last-known value if we have one, else None


def _compute_iv_rank(symbol: str, current_iv: float) -> dict | None:
    """Compute IV Rank (IVR) from the collected history.
    IVR = (current_IV - period_low) / (period_high - period_low) × 100
    Returns None if < 5 days of data (too few to be meaningful)."""
    try:
        records = nse_history_store.read_history(symbol, days=30)
        ivs = [r["atm_iv"] for r in records if r.get("atm_iv") and r["atm_iv"] > 0]
        if len(ivs) < 5:
            return None
        lo, hi = min(ivs), max(ivs)
        rank = round((current_iv - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0
        return {
            "rank_pct": rank,
            "period_days": len(set(r["t"] // 86400 for r in records)),
            "low": round(lo, 2),
            "high": round(hi, 2),
            "current": round(current_iv, 2),
        }
    except Exception as e:  # noqa: BLE001
        return None


def _build_response(symbol: str, expiry: str | None, band: int) -> dict:
    """Fetch (or reuse a cached) chain, run the full analysis pipeline, and
    shape everything into a single JSON-serializable dict for the frontend."""
    global _last_fetch_time

    now = time.time()
    cache_key = symbol
    cached = _cache.get(cache_key)
    fetcher = NSESession()  # cheap to construct — no I/O until a method is actually called
    raw = None
    if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
        raw = cached[1]
    else:
        wait = MIN_SECONDS_BETWEEN_FETCHES - (now - _last_fetch_time)
        if wait > 0:
            time.sleep(wait)
        raw = fetcher.get_option_chain(symbol)
        _last_fetch_time = time.time()
        _cache[cache_key] = (_last_fetch_time, raw)
        # Save the warmed session so other endpoints (movers, etc.) can reuse
        # it without paying the 2-second cold-start cost again
        global _shared_fetcher, _shared_fetcher_ts
        _shared_fetcher = fetcher
        _shared_fetcher_ts = time.time()

    snap = parse_chain(raw, symbol, expiry)
    strike_gap = infer_strike_gap(snap)
    atm = find_atm_strike(snap)
    support, resistance, nearby = support_resistance(snap, atm, band, strike_gap)
    pcr = compute_pcr(snap)
    payout_distribution = compute_payout_distribution(snap)
    max_pain = min(payout_distribution, key=lambda d: d[1])[0]
    flags = classify_buildups(nearby, atm)
    skew_note = iv_skew_read(nearby, atm)
    dte = days_to_expiry(snap.expiry)
    sentiment, ideas = generate_strategies(
        snap, atm, support, resistance, pcr, max_pain, flags, skew_note, dte
    )

    # ATM implied vol — representative volatility used as the terminal-price
    # distribution's sigma for every strategy's POP calculation.
    atm_strike_data = min(snap.strikes, key=lambda s: abs(s.strike - atm))
    atm_iv = (atm_strike_data.ce_iv + atm_strike_data.pe_iv) / 2.0 or 13.0

    # The raw payload already contains ALL expiries in one go (that's why we
    # cache per-symbol, not per-expiry) — so parsing several of them costs no
    # extra NSE call, just additional parse_chain passes over data already in
    # memory. We expose all of these to the frontend so the Strategy Builder
    # can let you pick ANY week for ANY leg (not just an auto-picked "next"
    # expiry), and reuse the same data for the pre-built calendar/diagonal
    # strategy templates below.
    MAX_EXPIRIES_EXPOSED = 8
    expiries_data: dict[str, dict] = {}
    for exp in snap.all_expiries[:MAX_EXPIRIES_EXPOSED]:
        try:
            exp_snap = snap if exp == snap.expiry else parse_chain(raw, symbol, exp)
            expiries_data[exp] = {
                "dte": days_to_expiry(exp),
                "strikes": [asdict(s) for s in exp_snap.strikes],
            }
        except NSEFetchError:
            continue  # an individual expiry failing to parse shouldn't break the whole response

    far_strikes, dte_far, far_expiry = None, None, None
    far_expiry_candidates = [e for e in snap.all_expiries if e != snap.expiry]
    if far_expiry_candidates:
        far_expiry = far_expiry_candidates[0]
        if far_expiry in expiries_data:
            far_strikes_raw = expiries_data[far_expiry]["strikes"]
            dte_far = expiries_data[far_expiry]["dte"]
            # build_strategy_list wants StrikeData objects, not dicts — reparse cheaply
            far_snap = parse_chain(raw, symbol, far_expiry)
            far_strikes = far_snap.strikes

    # Dynamic lot size from NSE's own published list — works for both
    # indices (overrides the hardcoded fallback if NSE's number differs)
    # and any F&O stock (which isn't in the hardcoded dict at all). Patch
    # it into the strategy engine's LOT_SIZES too, since build_strategy_list
    # reads from there internally for every margin/funds calculation.
    lot_size = nse_lot_sizes.get_lot_size(symbol, fallback=LOT_SIZES.get(symbol, 50))
    LOT_SIZES[symbol] = lot_size

    strategy_list = build_strategy_list(
        symbol=symbol,
        near_strikes=snap.strikes,   # full chain, not band-limited — strategy legs can reach further OTM than the display band
        atm=atm,
        strike_gap=strike_gap,
        spot=snap.underlying_value,
        dte_near=dte,
        atm_iv=atm_iv,
        far_strikes=far_strikes,
        dte_far=dte_far,
    )

    india_vix = _get_india_vix(fetcher)

    # ATM straddle premium — tracked over time for the straddle chart
    atm_row = next((s for s in snap.strikes if abs(s.strike - atm) < 1e-6), None)
    straddle_premium = round((atm_row.ce_ltp + atm_row.pe_ltp), 1) if atm_row else None

    # IV Rank from collected history (meaningful after ≥5 days of data)
    iv_rank = _compute_iv_rank(symbol, round(atm_iv, 2))

    response = {
        "symbol": snap.symbol,
        "expiry": snap.expiry,
        "all_expiries": snap.all_expiries,
        "underlying_value": snap.underlying_value,
        "timestamp": snap.timestamp,
        "dte": dte,
        "dte_far": dte_far,
        "far_expiry": far_expiry,
        "atm": atm,
        "atm_iv": round(atm_iv, 2),
        "atm_iv_rank": iv_rank,
        "straddle_premium": straddle_premium,
        "strike_gap": strike_gap,
        "lot_size": lot_size,
        "pcr": pcr,
        "max_pain": max_pain,
        "payout_distribution": payout_distribution,
        "support": asdict(support),
        "resistance": asdict(resistance),
        "sentiment": sentiment,
        "skew_note": skew_note,
        "ideas": [asdict(i) for i in ideas],
        "flags": flags,
        "strikes": [asdict(s) for s in snap.strikes],
        "expiries_data": expiries_data,
        "strategies": [asdict(s) for s in strategy_list],
        "india_vix": india_vix,
    }

    snapshot = {
        "spot": snap.underlying_value, "pcr": pcr, "atm_iv": round(atm_iv, 2),
        "max_pain": max_pain, "support": support.strike, "resistance": resistance.strike,
        "india_vix": india_vix, "straddle_premium": straddle_premium,
    }
    nse_history_store.append_snapshot(symbol, snapshot)
    try:
        nse_alerts.check_and_alert(symbol, snapshot)
    except Exception as e:
        print(f"[!] Alert check failed (non-fatal): {e}")

    return response


def _compute_brokerage_cost(legs: list[dict], lot_size: int, brokerage_per_order: float = 20.0) -> float:
    """Compute total NSE F&O options transaction costs for a set of legs.
    Same formula as the JS implementation — see the JS comment block for rate
    citations. Returns the total cost in rupees (positive number to subtract
    from P&L)."""
    buy_turnover = sell_turnover = 0.0
    num_orders = 0
    for leg in legs:
        if not leg.get("premium") or leg.get("instrument_type") == "FUTURES":
            num_orders += 1
            continue
        tv = leg["premium"] * leg.get("qty_lots", 1) * lot_size
        if leg.get("action") == "BUY":
            buy_turnover += tv
        else:
            sell_turnover += tv
        num_orders += 1
    total_turnover = buy_turnover + sell_turnover
    stt = sell_turnover * 0.000125
    exchange = total_turnover * 0.00053
    sebi = total_turnover * 0.000001
    stamp = buy_turnover * 0.00003
    brokerage = num_orders * brokerage_per_order
    gst = (brokerage + exchange) * 0.18
    return round(stt + exchange + sebi + stamp + brokerage + gst, 2)


def _compute_open_trade_pnl(trade: dict) -> float | None:
    """Looks up each leg's CURRENT live premium (reusing the same fetch/cache
    path as /api/chain — no extra NSE hits if that symbol's already been
    polled recently) and returns the trade's live P&L. Returns None if any
    leg's expiry is no longer in the live chain (most likely it's expired
    since the trade was opened) — the dashboard shows that as 'needs manual
    close' rather than guessing at a settlement value."""
    try:
        now = time.time()
        cached = _cache.get(trade["symbol"])
        if cached and (now - cached[0]) < CACHE_TTL_SECONDS:
            raw = cached[1]
        else:
            fetcher = NSESession()
            raw = fetcher.get_option_chain(trade["symbol"])
            global _last_fetch_time
            _last_fetch_time = time.time()
            _cache[trade["symbol"]] = (_last_fetch_time, raw)

        snap = parse_chain(raw, trade["symbol"])  # nearest expiry, just need underlying_value generally
        total = 0.0
        for leg in trade["legs"]:
            if leg.get("instrument_type") == "FUTURES":
                current_value = snap.underlying_value
            else:
                leg_snap = parse_chain(raw, trade["symbol"], leg["expiry"])
                strikes_by_value = {s.strike: s for s in leg_snap.strikes}
                if not strikes_by_value:
                    return None
                closest = min(strikes_by_value.keys(), key=lambda k: abs(k - leg["strike"]))
                sd = strikes_by_value[closest]
                current_value = sd.ce_ltp if leg["option_type"] == "CE" else sd.pe_ltp
            sign = 1 if leg["action"] == "BUY" else -1
            total += sign * (current_value - leg["premium"]) * leg["qty_lots"]
        return round(total * trade["lot_size"], 2)
    except NSEFetchError:
        return None
    except Exception as e:  # noqa: BLE001 - a pricing failure for one trade shouldn't break the list
        print(f"[!] Live P&L calc failed for trade {trade.get('id')}: {e}")
        return None


class Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request console spam; we print our own concise log line.
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/api/health":
            self._send_json({"status": "ok"})
            return

        if parsed.path == "/api/chain":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            expiry = qs.get("expiry", [None])[0]
            try:
                band = int(qs.get("band", ["12"])[0])
            except ValueError:
                band = 12

            # No longer restricted to the 4 index symbols — any NSE F&O
            # symbol works (confirmed the underlying endpoint is identical
            # for stocks). Just a basic sanity check on the format so an
            # obviously-malformed symbol fails fast with a clear message
            # rather than burning an NSE round-trip first.
            if not symbol or not symbol.replace("-", "").replace("&", "").isalnum():
                self._send_json({"error": f"'{symbol}' doesn't look like a valid NSE symbol."}, status=400)
                return

            print(f"[i] {self.command} /api/chain symbol={symbol} expiry={expiry} band={band}")
            try:
                data = _build_response(symbol, expiry, band)
                self._send_json(data)
            except NSEFetchError as e:
                print(f"[!] Fetch error: {e}")
                self._send_json({"error": str(e)}, status=502)
            except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI rather than hanging it
                print(f"[!] Unexpected error: {e}")
                self._send_json({"error": f"Unexpected server error: {e}"}, status=500)
            return

        if parsed.path == "/api/history":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                days = int(qs.get("days", ["1"])[0])
            except ValueError:
                days = 1
            try:
                records = nse_history_store.read_history(symbol, days=days)
                records = nse_history_store.downsample(records, max_points=500)
                self._send_json({"symbol": symbol, "days": days, "points": records})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"History read failed: {e}"}, status=500)
            return

        if parsed.path == "/api/vix":
            try:
                fetcher = NSESession()
                vix = _get_india_vix(fetcher)
                self._send_json({"india_vix": vix})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=502)
            return

        if parsed.path == "/api/fno-symbols":
            try:
                symbols = nse_lot_sizes.get_fno_symbol_list()
                self._send_json({"symbols": symbols, "count": len(symbols)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not fetch F&O symbol list: {e}"}, status=502)
            return

        if parsed.path == "/api/paper-trades":
            status_filter = qs.get("status", ["all"])[0]
            try:
                trades = nse_paper_trades.get_trades(status_filter)
                for t in trades:
                    if t["status"] == "open":
                        t["live_pnl"] = _compute_open_trade_pnl(t)
                stats = nse_paper_trades.summary_stats(nse_paper_trades.get_trades("all"))
                self._send_json({"trades": trades, "stats": stats})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not load paper trades: {e}"}, status=500)
            return

        if parsed.path == "/api/drafts":
            try:
                self._send_json({"drafts": nse_drafts.get_drafts()})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not load drafts: {e}"}, status=500)
            return

        if parsed.path == "/api/iv-rank":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                records = nse_history_store.read_history(symbol, days=30)
                ivs = [r["atm_iv"] for r in records if r.get("atm_iv") and r["atm_iv"] > 0]
                if len(ivs) < 5:
                    self._send_json({"symbol": symbol, "iv_rank": None,
                                     "msg": f"Need ≥5 data points, have {len(ivs)}"})
                    return
                lo, hi = min(ivs), max(ivs)
                # Return full IV history for charting (downsampled)
                iv_history = nse_history_store.downsample(
                    [{"t": r["t"], "iv": r["atm_iv"]} for r in records if r.get("atm_iv")], 300
                )
                self._send_json({
                    "symbol": symbol,
                    "iv_rank": {
                        "current": round(ivs[-1], 2) if ivs else None,
                        "low": round(lo, 2), "high": round(hi, 2),
                        "rank_pct": round((ivs[-1] - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0,
                        "period_days": len(set(r["t"] // 86400 for r in records)),
                        "samples": len(ivs),
                    },
                    "iv_history": iv_history,
                })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"IV rank calculation failed: {e}"}, status=500)
            return

        if parsed.path == "/api/fno-movers":
            self._send_json(nse_fno_movers.get_movers())
            return
            try:
                records = nse_history_store.read_history(symbol, days=1)
                straddle_pts = [{"t": r["t"], "v": r["straddle_premium"]}
                                 for r in records if r.get("straddle_premium")]
                straddle_pts = nse_history_store.downsample(straddle_pts, 300)
                self._send_json({"symbol": symbol, "points": straddle_pts})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Straddle history failed: {e}"}, status=500)
            return

        self._send_json({"error": f"Unknown route: {parsed.path}"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_raw = self.rfile.read(content_length) if content_length else b"{}"
            body = json.loads(body_raw) if body_raw else {}
        except Exception as e:  # noqa: BLE001
            self._send_json({"error": f"Invalid request body: {e}"}, status=400)
            return

        if parsed.path == "/api/paper-trade/open":
            try:
                symbol = body.get("symbol", "").upper()
                name = body.get("name", "Custom Strategy")
                legs = body.get("legs", [])
                lot_size = body.get("lot_size")
                if not symbol or not legs or not lot_size:
                    self._send_json({"error": "Need symbol, legs, and lot_size"}, status=400)
                    return
                trade = nse_paper_trades.open_trade(symbol, name, legs, lot_size)
                self._send_json({"trade": trade})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not open paper trade: {e}"}, status=500)
            return

        if parsed.path == "/api/paper-trade/close":
            try:
                trade_id = body.get("id")
                if not trade_id:
                    self._send_json({"error": "Need trade id"}, status=400)
                    return
                # recompute live P&L server-side at close time rather than
                # trusting whatever the client last displayed (avoids a
                # stale-quote mismatch between what was shown and what gets recorded)
                trades = nse_paper_trades.get_trades("open")
                trade = next((t for t in trades if t["id"] == trade_id), None)
                if not trade:
                    self._send_json({"error": "Trade not found or already closed"}, status=404)
                    return
                live_pnl = _compute_open_trade_pnl(trade)
                if live_pnl is None:
                    self._send_json({"error": "Could not price this trade right now (expiry may have passed) — try again or check manually"}, status=502)
                    return
                closed = nse_paper_trades.close_trade(trade_id, live_pnl, body.get("reason", "manual"))
                # deduct brokerage for BOTH entry (when trade was opened) and
                # exit (now) — two full round-legs of transaction costs
                entry_costs = _compute_brokerage_cost(trade["legs"], trade["lot_size"])
                exit_costs = _compute_brokerage_cost(trade["legs"], trade["lot_size"])
                total_costs = entry_costs + exit_costs
                net_pnl = live_pnl - total_costs
                # re-close with the cost-adjusted P&L
                closed = nse_paper_trades.close_trade.__func__ if False else None
                # re-read and update in place since close_trade already wrote the file
                all_trades = nse_paper_trades.load_trades()
                for t in all_trades:
                    if t["id"] == trade_id:
                        t["exit_pnl"] = round(net_pnl, 2)
                        t["brokerage_deducted"] = round(total_costs, 2)
                        closed = t
                        break
                nse_paper_trades.save_trades(all_trades)
                self._send_json({"trade": closed, "costs": {"entry": entry_costs, "exit": exit_costs, "total": total_costs}})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not close paper trade: {e}"}, status=500)
            return

        if parsed.path == "/api/paper-trade/delete":
            try:
                trade_id = body.get("id")
                deleted = nse_paper_trades.delete_trade(trade_id) if trade_id else False
                self._send_json({"deleted": deleted})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not delete paper trade: {e}"}, status=500)
            return

        if parsed.path == "/api/draft/save":
            try:
                name = body.get("name", "Untitled")
                symbol = body.get("symbol", "").upper()
                legs = body.get("legs", [])
                if not legs:
                    self._send_json({"error": "No legs to save"}, status=400)
                    return
                draft = nse_drafts.save_draft(name, symbol, legs)
                self._send_json({"draft": draft})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not save draft: {e}"}, status=500)
            return

        if parsed.path == "/api/draft/delete":
            try:
                draft_id = body.get("id")
                deleted = nse_drafts.delete_draft(draft_id) if draft_id else False
                self._send_json({"deleted": deleted})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not delete draft: {e}"}, status=500)
            return

        self._send_json({"error": f"Unknown route: {parsed.path}"}, status=404)


def main():
    ap = argparse.ArgumentParser(description="Local API server for the NSE options dashboard")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    nse_alerts.ensure_config_file_exists()
    alert_cfg = nse_alerts.load_config()
    alert_status = "ENABLED" if alert_cfg.get("enabled") and alert_cfg.get("telegram_token") else "disabled (edit alert_config.json to set up Telegram alerts)"

    # Start movers background worker — fetches F&O top gainers/losers every 90s
    # using the shared warmed session from the option chain fetcher.
    nse_fno_movers.start_background_worker(
        get_shared_fetcher=lambda: _shared_fetcher,
        lot_size_fetcher=nse_lot_sizes,
    )
    print("[i] F&O movers background worker started (updates every 90s after first chain fetch)")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[i] NSE chain server running at http://{args.host}:{args.port}")
    print(f"[i] Try it: http://{args.host}:{args.port}/api/chain?symbol=NIFTY")
    print(f"[i] Telegram alerts: {alert_status}")
    print(f"[i] History persists to ./{nse_history_store.HISTORY_DIR.name}/")
    print("[i] Now open nse_dashboard.html in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Stopped.")


if __name__ == "__main__":
    main()
