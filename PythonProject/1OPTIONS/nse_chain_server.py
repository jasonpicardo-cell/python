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
           Also includes: `iv_rank` (flat 0-100, from ./history — needs
           ≥5 days of data), `futures_price` + `futures_dte` (nearest-month
           futures for the basis card, best-effort, cached 30s).

    GET /api/oi-timeline?symbol=NIFTY
        -> {"strikes":[...], "times":["09:18",...], "ce":[[dOI,...]...],
           "pe":[[...]...]} — 3-min dOI-per-strike grid for the dashboard's
           OI Flow heatmap. Populated automatically while /api/chain is
           polled (piggybacks on chain fetches, no extra NSE calls); holds
           ~6.5h in memory, resets on server restart.

    POST /api/notify   {"message": "..."}
        -> relays a dashboard alert to Telegram via alert_config.json
           (rate-limited 1/5s; returns {"sent": bool, "reason"?}).

    GET /api/replay-dates?symbol=NIFTY          -> recorded session dates
    GET /api/replay-index?symbol=NIFTY&date=YYYY-MM-DD -> snapshot timestamps
    GET /api/replay-snap?symbol=NIFTY&date=...&i=N     -> one full snapshot
        Session recorder: one chain snapshot/min saved to ./replay/ while
        /api/chain is polled. Powers the dashboard's Replay mode.

    GET /api/ltp-history?symbol=NIFTY&strike=24000&side=CE
        -> {"points":[{"t":epoch,"ltp":...}]} — 1-min LTP samples of one
           strike/side, collected automatically while /api/chain is polled.
           Powers the per-leg mini price charts in the Builder.

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
import csv as _csv_mod
import io as _io_mod
import json
import sys
import itertools
import threading
from urllib.parse import quote as _urlquote
import time
import concurrent.futures
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

# Track intraday OHLC for each symbol (populated from chain fetches + allIndices)
_session_ohlc: dict[str, dict] = {}   # symbol → {open,high,low,close,prev_close,date}

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

# ── Server telemetry ─────────────────────────────────────────────────────────
_SERVER_START_TIME: float = time.time()

# ── 15-minute OI momentum snapshots ─────────────────────────────────────────
# Separate from the session baseline — records a periodic snapshot every
# 15 minutes so we can show "what changed in the last 15 minutes" per strike,
# which is far more actionable for scalping than the session total.
_oi_snapshots: dict[str, list] = {}
_OI_SNAPSHOT_INTERVAL = 900  # 15 minutes

def _record_oi_snapshot(symbol: str, strikes: list) -> None:
    now = time.time()
    sym = symbol.upper()
    snaps = _oi_snapshots.setdefault(sym, [])
    if snaps and (now - snaps[-1]["ts"]) < _OI_SNAPSHOT_INTERVAL:
        return
    snaps.append({
        "ts": now,
        "oi": {s.strike: {"ce": s.ce_oi or 0, "pe": s.pe_oi or 0} for s in strikes},
    })
    if len(snaps) > 20:   # keep up to 5 hours of snapshots
        snaps.pop(0)

def _get_15m_oi_delta(symbol: str, strikes: list) -> dict:
    """Return {strike: (ce_15m, pe_15m)} vs the snapshot ≥15 min ago."""
    sym  = symbol.upper()
    snaps = _oi_snapshots.get(sym, [])
    now  = time.time()
    ref  = next((s for s in reversed(snaps) if (now - s["ts"]) >= _OI_SNAPSHOT_INTERVAL), None)
    if not ref:
        return {}
    result: dict[int, tuple[int, int]] = {}
    for s in strikes:
        prev = ref["oi"].get(s.strike, {"ce": 0, "pe": 0})
        result[s.strike] = ((s.ce_oi or 0) - prev["ce"], (s.pe_oi or 0) - prev["pe"])
    return result


# ── OI flow timeline (heatmap feed) ──────────────────────────────────────────
# Finer-grained than the 15-min snapshots: one column every 3 minutes storing
# ΔOI per strike vs the previous column. Piggybacks on chain fetches (no extra
# NSE calls), served via GET /api/oi-timeline?symbol=NIFTY in the grid shape
# the dashboard's heatmap expects.
# ── Session replay recorder ─────────────────────────────────────────────────
# Writes one full chain snapshot per minute to ./replay/SYMBOL_YYYY-MM-DD.jsonl
# so the dashboard can scrub back through the day after close. ~20-60KB/min.
import os as _os
# ══════════════════════════════════════════════════════════════════════
# .env LOADER
# Reads a .env file sitting next to this script and puts the values into
# the environment BEFORE anything else reads them. Precedence stays sane:
# a variable already exported in the shell wins over the file, and command
# line flags win over both.
# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# SESSION BOUNDS — updated for the Closing Auction Session (CAS)
#
# From 3 August 2026, SEBI's CAS changed the end of the trading day:
#   * continuous trading in F&O-eligible CASH stocks ends 15:15, then those
#     stocks go into a 20-minute auction that sets their closing price
#   * EQUITY DERIVATIVES keep trading until 15:40 - ten minutes past the old
#     close - so index options are live after the cash market has stopped
#   * the Nifty close is now derived from CAS equilibrium prices, which is why
#     3 August saw a ~200-point move in the final minutes
#
# Index options themselves are NOT in the auction; they trade continuously to
# 15:40. So a dashboard that stops at 15:30 silently drops the last ten
# minutes of options trading from candles, OI flow, alerts and every archive
# the historical panels read - during a window where price discovery is
# unusually active because the underlying close is landing.
# ══════════════════════════════════════════════════════════════════════
SESSION_OPEN_MIN = 9 * 60 + 15         # 09:15 IST
SESSION_END_MIN = int(_os.environ.get("NSE_SESSION_END_MIN", 15 * 60 + 40))   # 15:40
CAS_CASH_CUTOFF_MIN = 15 * 60 + 15     # F&O cash stocks stop continuous trade
SESSION_LEN_MIN = SESSION_END_MIN - SESSION_OPEN_MIN                          # 385


def _in_session(mins: int) -> bool:
    """True for a minute inside the derivatives trading day (09:15-15:40)."""
    return SESSION_OPEN_MIN <= mins < SESSION_END_MIN


def _envbool_g(name: str, default: bool) -> bool:
    v = _os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


# Candidate names for the settings file, in priority order. ".env" is the
# convention, but a plain "env" is common on machines where a leading dot makes
# the file awkward to see or edit - and silently loading nothing because of a
# filename is a miserable thing to debug, so accept both and say which was used.
_ENV_NAMES = ("env", ".env", ".env.local", "env.txt", ".env.txt")


def _find_dotenv() -> str:
    here = _os.path.dirname(_os.path.abspath(__file__))
    found = [n for n in _ENV_NAMES if _os.path.exists(_os.path.join(here, n))]
    if len(found) > 1:
        # A silent trap: edit the wrong one and your change simply never takes
        # effect, with nothing to indicate why. Say which is being used.
        print(f"[env] WARNING: multiple settings files present ({', '.join(found)}). "
              f"Using '{found[0]}' - edits to the others are IGNORED. "
              f"Delete or rename the spares to avoid confusion.")
    return _os.path.join(here, found[0]) if found else ""


def _load_dotenv(path: str = None) -> int:
    path = path or _os.environ.get("NSE_ENV_FILE") or _find_dotenv()
    if not path or not _os.path.exists(path):
        here = _os.path.dirname(_os.path.abspath(__file__))
        print(f"[env] no settings file found in {here} "
              f"(looked for: {', '.join(_ENV_NAMES)})")
        return 0
    n = 0
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in _os.environ:      # shell wins over the file
                    _os.environ[k] = v
                    n += 1
    except Exception as e:  # noqa: BLE001
        print(f"[env] could not read {path}: {e}")
        return 0
    if n:
        print(f"[env] loaded {n} setting(s) from {_os.path.basename(path)}")
    return n


_load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# RUN SETTINGS — edit these if you start the server by pressing ▶ Run in an
# IDE (PyCharm/VS Code) instead of typing a command line.
#
# Command-line flags, when given, always WIN over these values, so the two
# ways of starting the server never fight each other. Environment variables
# (NSE_LAN, NSE_POLL, NSE_PORT, NSE_POLL_INTERVAL, NSE_CHAIN_TTL) sit in
# between: they override these defaults but yield to explicit flags.
# ══════════════════════════════════════════════════════════════════════
# Market-data adapter: "nse" (scrape, polled) or "arrow" (broker, streamed).
# Set DATA_SOURCE in .env; this is only the fallback when .env is absent.
RUN_DATA_SOURCE = _os.environ.get("DATA_SOURCE", "nse").strip().lower()

RUN_LAN = True            # True  -> reachable from other computers (binds 0.0.0.0)
                          # False -> this machine only (127.0.0.1)
RUN_PORT = 8765
RUN_POLL = "NIFTY,BANKNIFTY"   # symbols kept warm in the background; "" to disable.
                               # With polling on, NSE sees one request per symbol
                               # per cycle no matter how many browsers are open.
RUN_POLL_INTERVAL = 5.0   # seconds between background polls per symbol
RUN_TTL = 8.0             # seconds a cached chain counts as fresh
RUN_REFRESH_WEIGHTS = False  # fetch live index constituent weights at startup.
                             # Off by default because it costs an NSE round
                             # trip on every boot; turn it on after a quarterly
                             # rebalance (Mar/Jun/Sep/Dec) or when a
                             # constituent changes, then turn it back off.
RUN_RECONCILE = True       # periodically re-check today's candles for holes,
                           # fill them, and upgrade Yahoo-filled minutes to NSE
                           # data once the authoritative feed recovers
RUN_RECONCILE_INTERVAL = 300.0   # seconds between sweeps

RUN_KEEP_AWAKE = True     # stop the OS suspending this process when the
                          # machine is locked or idle (macOS caffeinate /
                          # Windows SetThreadExecutionState / systemd-inhibit)

# ── Alert engine (server-side detection, streamed to every browser) ──
# Tuned for Nifty. The two gates are ANDed, and the original 8%/3% pair was
# mismatched: 8% OI on a 400k-contract strike means 32,000 contracts inside
# two minutes (a news spike, not positioning), while 3% on a 40-rupee option
# is 1.2 points, which tick noise crosses constantly. Loosening OI and
# tightening premium lets real position building through and keeps jitter out.
RUN_ALERT_OI_PCT = 3.0     # min |open-interest change| %
RUN_ALERT_PREM_PCT = 6.0   # min |premium change| % (both must be crossed)
RUN_ALERT_NEAR_PCT = 1.5   # only strikes within this % of spot. Nifty moves
                           # ~0.6-1% a session, so +/-1.5% (about 15 strikes)
                           # covers what price can reach; wider drags in thin
                           # strikes whose small OI base trips the % gate cheaply
RUN_ALERT_WINDOW = 90.0    # seconds between the two compared snapshots
RUN_ALERT_COOLDOWN = 300.0 # per strike+side, seconds before it can fire again
RUN_ALERT_CLOSING = True   # also announce unwinding and short covering, not
                           # just freshly-opened positions. Closing flow at a
                           # level price is approaching is often the earlier
                           # signal - a wall being abandoned precedes the break.
RUN_ALERT_MIN_OI = 4000    # absolute contract floor, to reject THIN strikes
                           # only. A 3% move on a 150k-OI strike is just 4,500
                           # contracts, so a floor above ~9,000 cancels the
                           # percentage gate and the two filters fight each
                           # other into silence. 4,000 is about 53 lots.
RUN_ALERT_MAX = 6          # extra same-category events per cycle (all 4
                           # categories always get through regardless)

# ══════════════════════════════════════════════════════════════════════
# ALERT STREAM — the server detects alerts and PUSHES them to browsers.
#
# Previously each browser watched the chain itself and then argued over who
# should speak. That had too many moving parts: different refresh timings,
# per-machine toggles, claims taken by muted tabs. Now:
#   * the server compares consecutive chain snapshots and raises the events
#   * every connected browser holds an open SSE connection (so the server
#     knows exactly which systems are live, and how many tabs each has)
#   * for each alert the server elects ONE tab per machine as the speaker and
#     marks the payload accordingly, so audio never overlaps on a PC while
#     every separate computer still announces once
# ══════════════════════════════════════════════════════════════════════
_sse_clients: list = []          # [{"id","ip","q","ts"}]
_sse_lock = threading.Lock()
_flow_prev: dict = {}            # symbol -> {"ts":, "m": {strike: (ce_oi, pe_oi, ce_ltp, pe_ltp)}}
_flow_fired: dict = {}           # dedupe key -> ts
ALERT_CFG = {
    "oi_pct": float(_os.environ.get("NSE_ALERT_OI_PCT", RUN_ALERT_OI_PCT)),
    "prem_pct": float(_os.environ.get("NSE_ALERT_PREM_PCT", RUN_ALERT_PREM_PCT)),
    "near_pct": float(_os.environ.get("NSE_ALERT_NEAR_PCT", RUN_ALERT_NEAR_PCT)),
    "window": float(_os.environ.get("NSE_ALERT_WINDOW", RUN_ALERT_WINDOW)),
    "cooldown": float(_os.environ.get("NSE_ALERT_COOLDOWN", RUN_ALERT_COOLDOWN)),
    # ceiling on EXTRA same-category events; distinct categories always pass
    "max_per_cycle": int(_os.environ.get("NSE_ALERT_MAX", RUN_ALERT_MAX)),
    "min_oi": float(_os.environ.get("NSE_ALERT_MIN_OI", RUN_ALERT_MIN_OI)),
    # emit closing flow (unwinding / short covering) as well as fresh positions
    "closing_flow": _os.environ.get("NSE_ALERT_CLOSING", str(RUN_ALERT_CLOSING)).strip().lower()
                    in ("1", "true", "yes", "on"),
}

_sse_seq = itertools.count(1)

def _sse_register(ip: str):
    import queue as _q
    # monotonic counter, not a timestamp: two tabs connecting in the same
    # millisecond used to collide and both be elected speaker.
    cl = {"id": f"{ip}#{next(_sse_seq)}", "ip": ip, "q": _q.Queue(maxsize=50),
          "ts": time.time(), "seq": next(_sse_seq)}
    with _sse_lock:
        _sse_clients.append(cl)
    return cl

def _sse_unregister(cl):
    with _sse_lock:
        try:
            _sse_clients.remove(cl)
        except ValueError:
            pass

def _sse_broadcast(event: dict):
    """Send an event to every client, electing one speaker per machine."""
    with _sse_lock:
        by_ip = {}
        for cl in _sse_clients:
            by_ip.setdefault(cl["ip"], []).append(cl)
        # the oldest connection on each machine is that machine's speaker
        speakers = {ip: sorted(cls, key=lambda x: x["seq"])[0]["id"] for ip, cls in by_ip.items()}
        targets = list(_sse_clients)
    for cl in targets:
        payload = dict(event)
        payload["speak"] = (speakers.get(cl["ip"]) == cl["id"])
        payload["client_id"] = cl["id"]
        try:
            cl["q"].put_nowait(payload)
        except Exception:
            pass          # slow client: drop rather than block the detector

_m920: dict = {}          # symbol -> frozen levels for the day
_m920_fired: dict = {}    # "symbol|label" -> ts

def _ist_minutes(ts=None):
    g = time.gmtime((ts or time.time()) + 5 * 3600 + 1800)
    return g.tm_hour * 60 + g.tm_min, time.strftime("%Y-%m-%d", g)

def _m920_levels(symbol: str, data: dict):
    """Freeze the 09:20 reference lines once per day, per symbol."""
    mins, day = _ist_minutes()
    cur = _m920.get(symbol)
    if cur and cur["day"] == day:
        return cur
    if mins < 560:                     # before 09:20 IST
        return None
    spot, iv = data.get("underlying_value"), data.get("atm_iv")
    if not spot or not iv:
        return None
    sigma = spot * (iv / 100.0) * (1.0 / 365.0) ** 0.5
    lv = {"day": day, "spot920": spot, "sigma": sigma, "iv": iv,
          "EOR+1": spot + sigma, "EOR": spot + 0.5 * sigma,
          "EOS": spot - 0.5 * sigma, "EOS-1": spot - sigma}
    _m920[symbol] = lv
    print(f"[9:20] {symbol} frozen: spot {spot:.1f} sigma {sigma:.1f} "
          f"EOR {lv['EOR']:.0f} EOS {lv['EOS']:.0f}")
    return lv

def _m920_detect(symbol: str, data: dict):
    """Raise a touch alert when spot reaches a frozen 09:20 line.

    The fired-history lives HERE, not in each browser: previously every system
    kept its own cooldown map, so a machine that had already seen a touch
    stayed silent while another announced it — alerts appeared to work on some
    systems and not others.
    """
    lv = _m920_levels(symbol, data)
    spot = data.get("underlying_value")
    if not lv or not spot:
        return
    tol = max(4.0, spot * 0.0006)
    now = time.time()
    for lab in ("EOR+1", "EOR", "EOS", "EOS-1"):
        v = lv[lab]
        if abs(spot - v) > tol:
            continue
        key = f"{symbol}|{lab}"
        if now - _m920_fired.get(key, 0) < 600:
            continue
        _m920_fired[key] = now
        kind = "resistance extension" if lab == "EOR+1" else "resistance" if lab == "EOR" \
            else "support" if lab == "EOS" else "support extension"
        spoken = lab.replace("+1", " plus one").replace("-1", " minus one")
        _sse_broadcast({
            "type": "m920", "symbol": symbol, "ts": now, "label": lab,
            "level": round(v, 1), "spot": round(spot, 1), "kind": kind,
            "text": f"Price touching {spoken} at {int(round(v))}, the nine twenty {kind} line",
        })

def _sse_broadcast_ticks(event: dict) -> None:
    """Fan price ticks out to every connected tab.

    Deliberately different from _sse_broadcast: no speaker election (every
    browser wants prices, only one should speak alerts) and a full queue is
    skipped rather than waited on, so one slow laptop cannot stall the feed
    for everyone.
    """
    with _sse_lock:
        targets = list(_sse_clients)
    for cl in targets:
        try:
            cl["q"].put_nowait(event)
        except Exception:
            pass


def _flow_detect(symbol: str, data: dict):
    """Compare this snapshot with the last one and raise flow alerts."""
    try:
        strikes = data.get("strikes") or []
        spot = data.get("underlying_value")
        if not strikes or not spot:
            return
        now = time.time()
        snap = {"ts": now, "m": {s.get("strike"): (s.get("ce_oi") or 0, s.get("pe_oi") or 0,
                                                   s.get("ce_ltp") or 0, s.get("pe_ltp") or 0)
                                 for s in strikes if s.get("strike")}}
        prev = _flow_prev.get(symbol)
        if not prev:
            _flow_prev[symbol] = snap
            return
        if now - prev["ts"] < ALERT_CFG["window"]:
            return
        _flow_prev[symbol] = snap
        events = []
        for k, (ce, pe, cl_, pl) in snap["m"].items():
            if abs(k - spot) / spot * 100 > ALERT_CFG["near_pct"]:
                continue
            p = prev["m"].get(k)
            if not p:
                continue
            for side, oi_now, oi_old, l_now, l_old in (
                ("CE", ce, p[0], cl_, p[2]), ("PE", pe, p[1], pl, p[3])):
                if not oi_old or not l_old or not l_now:
                    continue
                oi_pct = (oi_now - oi_old) / oi_old * 100
                pr_pct = (l_now - l_old) / l_old * 100
                if abs(oi_pct) < ALERT_CFG["oi_pct"] or abs(pr_pct) < ALERT_CFG["prem_pct"]:
                    continue
                if abs(oi_now - oi_old) < ALERT_CFG.get("min_oi", 0):
                    continue          # too few contracts to be worth saying
                # Four flow types, not two. OI RISING is a new position being
                # opened; OI FALLING is an existing one being closed, and that
                # is the other half of the story - a wall being unwound as
                # price approaches it is often more actionable than fresh
                # writing somewhere far away. The client could ask for these
                # ("also unwinding/covering") but the server never sent them,
                # so the setting did nothing whenever the stream was live.
                buying = pr_pct > 0
                opening = oi_pct > 0
                if opening:
                    kind = f"fresh {'call' if side == 'CE' else 'put'} {'buying' if buying else 'selling'}"
                    bias = ("bullish" if buying else "bearish") if side == "CE" else ("bearish" if buying else "bullish")
                    cat = ("ce" if side == "CE" else "pe") + ("Buy" if buying else "Sell")
                else:
                    # premium up while OI falls = shorts buying back (covering)
                    # premium down while OI falls = longs giving up (unwinding)
                    covering = buying
                    side_word = "call" if side == "CE" else "put"
                    kind = f"{side_word} {'short covering' if covering else 'long unwinding'}"
                    if side == "CE":
                        bias = "bullish" if covering else "bearish"
                    else:
                        bias = "bearish" if covering else "bullish"
                    cat = "cover" if covering else "unwind"
                if cat in ("cover", "unwind") and not ALERT_CFG.get("closing_flow", True):
                    continue
                fk = f"{symbol}|{k}|{side}"
                if now - _flow_fired.get(fk, 0) < ALERT_CFG["cooldown"]:
                    continue
                events.append({"fk": fk, "cat": cat, "strike": k, "side": side, "kind": kind,
                               "bias": bias, "oi_pct": round(oi_pct, 1), "pr_pct": round(pr_pct, 1)})
        if not events:
            return
        # round-robin by category so calls are never crowded out by puts
        events.sort(key=lambda e: -abs(e["oi_pct"]))
        seen, ordered = set(), []
        for e in events:
            if e["cat"] not in seen:
                seen.add(e["cat"]); ordered.append(e)
        ordered += [e for e in events if e not in ordered]
        # Never drop a whole category: everything in the round-robin head (one
        # event per distinct category) is always emitted, and max_per_cycle
        # only limits the EXTRA same-category events after that. Previously a
        # cap of 3 silently killed the 4th category every cycle, which is why
        # one of call/put buying/selling could go permanently unheard.
        head = len(seen)
        limit = max(head, ALERT_CFG["max_per_cycle"])
        for e in ordered[:limit]:
            _flow_fired[e["fk"]] = now
            _sse_broadcast({
                "type": "flow", "symbol": symbol, "ts": now,
                "text": (f"{e['kind']} at {int(e['strike'])}, open interest up "
                         f"{abs(e['oi_pct']):.0f} percent, premium "
                         f"{'up' if e['pr_pct'] > 0 else 'down'} {abs(e['pr_pct']):.0f} percent. {e['bias']}"),
                **e})
    except Exception as ex:  # noqa: BLE001
        print(f"[alerts] detect failed: {ex}")

# ══════════════════════════════════════════════════════════════════════
# SHARED CHAIN CACHE — one NSE request serves every connected browser.
#
# Without this, five open tabs meant five identical NSE fetches every
# refresh: wasteful, slower, and a fast route to NSE rate-limiting or an
# IP block. Now the server owns the data:
#   * every /api/chain response is cached per (symbol, expiry, band)
#   * a request inside the freshness window is served from memory, with
#     no network call at all
#   * when the cache IS stale, ONE request does the fetch while the others
#     wait on the same lock and then read the result (request coalescing),
#     so a simultaneous refresh by ten clients still makes one NSE call
#   * an optional background poller keeps hot symbols warm on its own, so
#     clients almost always hit a warm cache
# ══════════════════════════════════════════════════════════════════════
_CHAIN_TTL = float(_os.environ.get("NSE_CHAIN_TTL", RUN_TTL))     # seconds a snapshot counts as fresh
_chain_cache: dict = {}          # key -> {"data":..., "ts": epoch, "hits": n}
_chain_locks: dict = {}          # key -> threading.Lock (one fetcher per key)
_chain_meta_lock = threading.Lock()
_cache_stats = {"served": 0, "from_cache": 0, "fetches": 0, "coalesced": 0, "clients": {}}

def _cache_key(symbol: str, expiry, band: int) -> str:
    return f"{symbol}|{expiry or 'near'}|{band}"

def _cached_chain(symbol: str, expiry, band: int, client: str = "?"):
    """Return a chain response, fetching from NSE only when genuinely needed.

    Returns (data, meta) where meta describes how the request was served so
    the UI can show whether it hit cache and how old the data is.
    """
    key = _cache_key(symbol, expiry, band)
    now = time.time()
    with _chain_meta_lock:
        _cache_stats["served"] += 1
        _cache_stats["clients"][client] = now
        ent = _chain_cache.get(key)
        fresh = ent and (now - ent["ts"]) < _CHAIN_TTL
        if fresh:
            ent["hits"] += 1
            _cache_stats["from_cache"] += 1
            return ent["data"], {"cache": "hit", "age": round(now - ent["ts"], 2),
                                 "ttl": _CHAIN_TTL, "shared_hits": ent["hits"]}
        lock = _chain_locks.get(key)
        if lock is None:
            lock = _chain_locks[key] = threading.Lock()
        already_fetching = lock.locked()
    if already_fetching:
        with _chain_meta_lock:
            _cache_stats["coalesced"] += 1
    # Only one thread per key gets through here; the rest queue and then find
    # the freshly-stored snapshot waiting for them.
    with lock:
        now = time.time()
        with _chain_meta_lock:
            ent = _chain_cache.get(key)
        if ent and (now - ent["ts"]) < _CHAIN_TTL:
            with _chain_meta_lock:
                ent["hits"] += 1
                _cache_stats["from_cache"] += 1
            return ent["data"], {"cache": "coalesced", "age": round(now - ent["ts"], 2),
                                 "ttl": _CHAIN_TTL, "shared_hits": ent["hits"]}
        data = _fetch_via_adapter(symbol, expiry, band)
        with _chain_meta_lock:
            _chain_cache[key] = {"data": data, "ts": time.time(), "hits": 0}
            _cache_stats["fetches"] += 1
        try:
            _write_replay_snapshot(symbol, data)
        except Exception:
            pass
        try:
            _flow_detect(symbol, data)      # server-side alert engine
        except Exception:
            pass
        try:
            _m920_detect(symbol, data)      # 09:20 line touches
        except Exception:
            pass
        return data, {"cache": "miss", "age": 0.0, "ttl": _CHAIN_TTL, "shared_hits": 0}

_YAHOO_SYMBOLS = {
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "MIDCPNIFTY": "NIFTY_MID_SELECT.NS",
    "SENSEX": "^BSESN",
}
_yahoo_cache: dict = {}
_yahoo_state: dict = {"session": None, "last_call": 0.0, "blocked_until": 0.0}


def _yahoo_intraday(symbol: str, interval_min: int = 1) -> list:
    """Fetch today's intraday series from Yahoo Finance as a backfill source.

    Yahoo rate-limits aggressively (HTTP 429) when requests arrive without a
    session cookie or too often from one IP. Three defences:
      * a real session with a cookie obtained from the fc.yahoo.com handshake,
        reused across calls rather than a bare request each time
      * a hard floor between calls, plus honouring Retry-After on a 429
      * a long success cache and a backoff window after a 429, so a failing
        endpoint is not hammered into a longer ban

    Honest limits, surfaced to the UI rather than hidden: it is a derived and
    slightly delayed feed, so prices will not match NSE tick for tick - good
    enough to draw the shape of the morning, not to compute a level from.
    Returns [(epoch_sec, price)] or [] on any failure.
    """
    ysym = _YAHOO_SYMBOLS.get(symbol.upper())
    if not ysym:
        return []
    key = (ysym, interval_min)
    now = time.time()
    hit = _yahoo_cache.get(key)
    # serve a cached series for 5 minutes; intraday backfill does not need
    # to be fresher than that and every extra call risks the 429
    if hit and now - hit[0] < 300:
        return hit[1]
    # after a 429, stay away for a while rather than retrying into a ban
    if now < _yahoo_state.get("blocked_until", 0):
        wait = int(_yahoo_state["blocked_until"] - now)
        print(f"[yahoo] backing off {wait}s after rate limit; using cached/local data")
        return hit[1] if hit else []
    # global floor between any two Yahoo calls
    gap = now - _yahoo_state.get("last_call", 0)
    if gap < 3.0:
        time.sleep(3.0 - gap)
    _yahoo_state["last_call"] = time.time()

    iv = "1m" if interval_min <= 1 else ("5m" if interval_min <= 5 else "15m")
    sess = _yahoo_state.get("session")
    if sess is None:
        try:
            import requests
            sess = requests.Session()
            sess.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/121.0 Safari/537.36",
                "Accept": "application/json,text/plain,*/*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://finance.yahoo.com/",
                "Connection": "keep-alive",
            })
            # this handshake sets the cookies the chart API expects; without
            # them Yahoo answers 429 almost immediately
            try:
                sess.get("https://fc.yahoo.com", timeout=8)
            except Exception:
                pass
            try:
                sess.get("https://finance.yahoo.com/quote/" + ysym, timeout=8)
            except Exception:
                pass
            _yahoo_state["session"] = sess
        except Exception as e:  # noqa: BLE001
            print(f"[yahoo] could not create session: {e}")
            return hit[1] if hit else []

    hosts = ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
    payload = None
    for host in hosts:
        url = (f"https://{host}/v8/finance/chart/{_urlquote(ysym)}"
               f"?interval={iv}&range=1d&includePrePost=false")
        try:
            r = sess.get(url, timeout=12)
        except Exception as e:  # noqa: BLE001
            print(f"[yahoo] {ysym} via {host}: {type(e).__name__} {str(e)[:60]}")
            continue
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            back = int(retry) if (retry or "").isdigit() else 900
            _yahoo_state["blocked_until"] = time.time() + back
            _yahoo_state["session"] = None          # rebuild cookies next time
            print(f"[yahoo] {ysym}: rate limited, backing off {back}s "
                  f"(chart will use NSE/archive data meanwhile)")
            return hit[1] if hit else []
        if r.status_code != 200:
            print(f"[yahoo] {ysym} via {host}: HTTP {r.status_code}")
            continue
        try:
            payload = r.json()
            break
        except Exception as e:  # noqa: BLE001
            print(f"[yahoo] {ysym}: bad json {e}")
    if payload is None:
        return hit[1] if hit else []

    try:
        res = (payload.get("chart", {}).get("result") or [None])[0]
        if not res:
            print(f"[yahoo] {ysym}: no result ({payload.get('chart', {}).get('error')})")
            return hit[1] if hit else []
        stamps = res.get("timestamp") or []
        quote = ((res.get("indicators", {}).get("quote") or [{}])[0]) or {}
        closes = quote.get("close") or []
        out = []
        today_yday = time.gmtime(time.time() + 5 * 3600 + 1800).tm_yday
        for ts, px in zip(stamps, closes):
            if px is None:
                continue
            g = time.gmtime(ts + 5 * 3600 + 1800)
            if g.tm_yday != today_yday:
                continue
            mins = g.tm_hour * 60 + g.tm_min
            if not (_in_session(mins)):
                continue
            out.append((float(ts), float(px)))
        if out:
            _yahoo_cache[key] = (time.time(), out)
            _yahoo_state["blocked_until"] = 0
            f0 = time.gmtime(out[0][0] + 5 * 3600 + 1800)
            l0 = time.gmtime(out[-1][0] + 5 * 3600 + 1800)
            print(f"[yahoo] {ysym}: {len(out)} points "
                  f"{f0.tm_hour:02d}:{f0.tm_min:02d}-{l0.tm_hour:02d}:{l0.tm_min:02d} IST")
        else:
            print(f"[yahoo] {ysym}: 200 but no session points in range")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[yahoo] parse failed: {e}")
        return hit[1] if hit else []

# ══════════════════════════════════════════════════════════════════════
# CANDLE RECONCILER
# Backfill has been on-demand only: at startup and when the button is pressed.
# Two things then go unnoticed. A brief network drop mid-session leaves a hole
# nobody sees, because the poller keeps running and nothing errors. And a
# minute filled from Yahoo stays Yahoo-filled even after NSE's feed recovers
# and could supply the authoritative price.
#
# This sweeps today's series every few minutes, fills holes, and UPGRADES
# provisional minutes to NSE data where it becomes available. It goes quiet
# once the session is complete from authoritative sources, so a clean day
# costs nothing, and it reuses the shared session rather than handshaking.
# ══════════════════════════════════════════════════════════════════════
_recon_state: dict = {"last": 0.0, "runs": 0, "filled": 0, "upgraded": 0,
                      "report": {}, "provisional": {}}


def _session_minutes(day: str = None) -> tuple:
    """(first, last) IST session minute indices - now 09:15-15:40 under CAS."""
    return (SESSION_OPEN_MIN, SESSION_END_MIN)


def _ist_min_of(ts: float) -> int:
    g = time.gmtime(ts + 5 * 3600 + 1800)
    return g.tm_hour * 60 + g.tm_min


def _load_archive(symbol: str, day: str) -> dict:
    """minute-epoch -> price from the on-disk tick archive."""
    out = {}
    p = _os.path.join(_TICK_DIR, f"{symbol}_{day}.csv")
    if not _os.path.exists(p):
        return out
    try:
        with open(p) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    try:
                        out[int(float(parts[0]) // 60)] = float(parts[1])
                    except Exception:
                        continue
    except Exception:
        pass
    return out


def _write_archive(symbol: str, day: str, series: dict) -> None:
    p = _os.path.join(_TICK_DIR, f"{symbol}_{day}.csv")
    try:
        _os.makedirs(_TICK_DIR, exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            for m in sorted(series):
                f.write(f"{m * 60},{series[m]}\n")
        _os.replace(tmp, p)            # atomic, so a crash cannot truncate it
    except Exception as e:  # noqa: BLE001
        print(f"[reconcile] archive write failed: {e}")


def _nse_full_day(symbol: str) -> dict:
    """NSE's own intraday series as minute -> price, or {} if unavailable."""
    idx_names = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                 "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MID SELECT"}
    name = idx_names.get(symbol.upper())
    f = _shared_fetcher
    if not name or f is None:
        return {}
    try:
        from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
        from urllib.parse import quote_plus as _qp
        h = dict(API_HEADERS)
        h["Referer"] = NSE_OC_PAGE
        nospace = name.replace(" ", "")
        for url in (f"https://www.nseindia.com/api/chart-databyindex?index={_qp(name)}&indices=true",
                    f"https://www.nseindia.com/api/chart-databyindex?index={_qp(nospace)}&indices=true"):
            r = f.session.get(url, headers=h, timeout=12)
            if r.status_code != 200:
                continue
            pl = r.json()
            rows = []
            if isinstance(pl, dict):
                for k in ("grapthData", "graphData", "data", "chartData"):
                    v = pl.get(k)
                    if isinstance(v, list) and v:
                        rows = v
                        break
            elif isinstance(pl, list):
                rows = pl
            out = {}
            today_yday = time.gmtime(time.time() + 5 * 3600 + 1800).tm_yday
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue
                try:
                    ts = float(row[0]) / 1000.0
                    px = float(row[4]) if len(row) >= 5 else float(row[1])
                except Exception:
                    continue
                g = time.gmtime(ts + 5 * 3600 + 1800)
                if g.tm_yday != today_yday:
                    continue
                mn = g.tm_hour * 60 + g.tm_min
                if not (_in_session(mn)):
                    continue
                out[int(ts // 60)] = px
            if out:
                return out
    except Exception as e:  # noqa: BLE001
        print(f"[reconcile] nse feed: {type(e).__name__} {str(e)[:60]}")
    return {}


def _reconcile_symbol(symbol: str) -> dict:
    """Fill holes and upgrade provisional minutes for one symbol."""
    day = time.strftime("%Y-%m-%d")
    series = _load_archive(symbol, day)
    prov = _recon_state["provisional"].setdefault(symbol, set())
    now_min = _ist_min_of(time.time())
    open_m, close_m = _session_minutes()
    if now_min < open_m:
        return {"status": "pre-open"}
    upto = min(now_min, close_m - 1)

    # which session minutes SHOULD exist by now
    day_mid = None
    for m in series:
        day_mid = m - (_ist_min_of(m * 60) - open_m)
        break
    if day_mid is None:
        # derive today's 09:15 minute index without any existing sample
        base = time.time()
        g = time.gmtime(base + 5 * 3600 + 1800)
        secs_today = g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec
        day_mid = int((base - secs_today + open_m * 60) // 60)
    expected = {day_mid + i for i in range(0, upto - open_m + 1)}
    missing = expected - set(series)
    upgradable = prov & set(series)

    if not missing and not upgradable:
        return {"status": "complete", "minutes": len(series),
                "expected": len(expected), "provisional": len(prov)}

    filled = upgraded = 0
    # Prefer the ACTIVE data source. Under Fyers the reconciler was still
    # asking NSE's chart feed to repair holes, which needs a warmed NSE
    # session the server no longer maintains - so repairs silently did nothing.
    nse = {}
    if RUN_DATA_SOURCE == "fyers":
        try:
            import nse_adapter_fyers as _fy
            for cd in _fy.fetch_candles(symbol, 1, 1):
                nse[int(cd["t"] // 60)] = cd["c"]
        except Exception as e:  # noqa: BLE001
            print(f"[reconcile] fyers history unavailable: {e}")
    if not nse:
        nse = _nse_full_day(symbol)
    if nse:
        for m in list(missing):
            if m in nse:
                series[m] = nse[m]
                filled += 1
                prov.discard(m)
        for m in list(upgradable):
            if m in nse and abs(nse[m] - series[m]) > 0.001:
                series[m] = nse[m]           # authoritative price wins
                upgraded += 1
                prov.discard(m)
            elif m in nse:
                prov.discard(m)              # same value, no longer provisional
        missing = expected - set(series)

    # anything NSE could not supply, try Yahoo - and remember it as provisional
    if missing and _envbool_g("NSE_YAHOO_BACKFILL", True):
        ypts = _yahoo_intraday(symbol, 1)
        for ts, px in ypts:
            m = int(ts // 60)
            if m in missing:
                series[m] = px
                prov.add(m)
                filled += 1
        missing = expected - set(series)

    if filled or upgraded:
        _write_archive(symbol, day, series)
    rep = {"status": "repaired" if (filled or upgraded) else "gaps-remain",
           "minutes": len(series), "expected": len(expected),
           "filled": filled, "upgraded": upgraded,
           "still_missing": len(missing), "provisional": len(prov)}
    if filled or upgraded:
        _srcname = "Fyers" if RUN_DATA_SOURCE == "fyers" else "NSE"
        print(f"[reconcile] {symbol}: +{filled} filled, {upgraded} upgraded to {_srcname}, "
              f"{len(missing)} still missing ({len(series)}/{len(expected)} minutes)")
    return rep


def _reconcile_loop(interval: float, symbols):
    while not _poller_stop.is_set():
        _poller_stop.wait(interval)
        if _poller_stop.is_set():
            break
        mn = _ist_min_of(time.time())
        if not (SESSION_OPEN_MIN + 5 <= mn <= SESSION_END_MIN + 10):
            continue                       # only during and just after the session
        for sym in symbols:
            try:
                _recon_state["report"][sym] = _reconcile_symbol(sym)
                r = _recon_state["report"][sym]
                _recon_state["filled"] += r.get("filled", 0)
                _recon_state["upgraded"] += r.get("upgraded", 0)
            except Exception as e:  # noqa: BLE001
                print(f"[reconcile] {sym}: {e}")
        _recon_state["runs"] += 1
        _recon_state["last"] = time.time()


def _start_reconciler(symbols, interval: float = 300.0):
    syms = [s.strip().upper() for s in symbols if s and s.strip()]
    if not syms:
        return
    threading.Thread(target=_reconcile_loop, args=(interval, syms), daemon=True).start()
    print(f"[i] Candle reconciler: {', '.join(syms)} every {int(interval)}s "
          f"- fills gaps and upgrades Yahoo-filled minutes to NSE data when it recovers")


def _dte_fractional(expiry_str: str) -> float:
    """Days to expiry as a FRACTION, measured to 15:30 IST on expiry day.

    The shared days_to_expiry() truncates to whole days, which has two
    consequences that matter for an intraday options dashboard:
      * on expiry day it returns 0 from 09:15 to 15:30, so T is pinned to a
        floor and every Greek is frozen for the whole session - theta stops
        decaying exactly when it decays fastest
      * on any other day it ignores the time, so the morning and the close
        share a T that differs by a full trading session

    Measuring to the 15:30 close in fractional days fixes both, and returns
    a small positive floor after expiry rather than zero so Black-Scholes
    does not divide by zero.
    """
    from datetime import datetime as _dtc, timedelta as _td
    for fmt in ("%d-%b-%Y", "%d-%b-%y", "%Y-%m-%d"):
        try:
            exp = _dtc.strptime(str(expiry_str), fmt)
        except ValueError:
            continue
        # expiry settles at 15:30 IST
        exp = exp.replace(hour=15, minute=30)
        # "now" in IST regardless of where the server runs
        now_ist = _dtc.utcfromtimestamp(time.time() + 5 * 3600 + 1800)
        secs = (exp - now_ist).total_seconds()
        return max(secs / 86400.0, 1.0 / (24 * 60))    # floor: one minute
    return 1.0


def _trading_dte(cal_days: float) -> float:
    """Convert calendar days to trading days for volatility scaling.

    Volatility accrues on trading days, not calendar days: a Friday-to-Tuesday
    weekly spans 4 calendar days but only ~3 sessions, and pricing it on 365
    overstates the expected-move band by roughly 11%. This applies the 252/365
    ratio, which is the standard approximation and close enough for intraday
    use without a full holiday calendar.
    """
    return cal_days * (252.0 / 365.0)


def _ist_hm_g(ts) -> str:
    g = time.gmtime(ts + 5 * 3600 + 1800)
    return f"{g.tm_hour:02d}:{g.tm_min:02d}"


def _attach_oi_rollups(symbol: str, data: dict) -> dict:
    """Add the rolling OI-change fields the NSE path computes.

    ce_oi_15m / pe_oi_15m and ce_d60 / pe_d60 are derived from the server's own
    OI timeline, not from the source. The adapter path returned before that
    happened, so a Fyers chain carried neither - the CSV export lost two
    columns and the custom OI-change alert had nothing to compare against.
    """
    try:
        # Its OWN store. The first version reused _oi_timeline, which already
        # holds a completely different shape ({times, ce, pe, _prev, _last_ts})
        # for the GEX heatmap and OI Build Map - writing per-strike lists into
        # it destroyed those panels and threw KeyError: '_last_ts' on the next
        # timeline write. A shared name is not a shared schema.
        tl = _oi_rollup_hist.setdefault(symbol, {})
        now = time.time()
        for s in data.get("strikes") or []:
            k = s.get("strike")
            if k is None:
                continue
            hist = tl.setdefault(k, [])
            hist.append((now, s.get("ce_oi") or 0, s.get("pe_oi") or 0))
            # keep roughly an hour, which covers both windows
            while hist and now - hist[0][0] > 3900:
                hist.pop(0)
            for mins, suffix in ((15, "oi_15m"), (60, "d60")):
                cutoff = now - mins * 60
                # Baseline = the most recent sample AT OR BEFORE the window
                # start. Taking the first sample after the cutoff picked the
                # entry just appended this tick, so the difference came out as
                # zero whenever no older sample sat inside the window.
                older = [h for h in hist if h[0] <= cutoff]
                base = older[-1] if older else (hist[0] if hist else None)
                if base:
                    s[f"ce_{suffix}"] = (s.get("ce_oi") or 0) - base[1]
                    s[f"pe_{suffix}"] = (s.get("pe_oi") or 0) - base[2]
    except Exception as e:  # noqa: BLE001
        print(f"[oi-rollup] {e}")
    return data


def _fyers_token_date() -> str:
    """IST date the saved Fyers token was generated, or '' if none."""
    import json as _json
    path = _os.environ.get("FYERS_TOKEN_FILE") or _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)), ".fyers_token")
    if not _os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            ts = _json.load(f).get("ts", 0)
        if not ts:
            return ""
        g = time.gmtime(ts + 5 * 3600 + 1800)
        return f"{g.tm_year:04d}-{g.tm_mon:02d}-{g.tm_mday:02d}"
    except Exception:
        return ""


def _ensure_fyers_token() -> bool:
    """Refresh the Fyers token at startup if it was not generated today.

    Fyers tokens die at about 06:00 IST, so one generated yesterday is always
    dead by the time the market opens - there is no case where yesterday's
    token survives into a trading session. Rather than starting, failing on the
    first chain request and falling back to NSE silently, this checks up front
    and runs the interactive login.

    The interactive part is the constraint: the login needs an auth code pasted
    from a browser, so it can only run where somebody is watching. If stdin is
    not a terminal - a service, a cron job, a startup item - blocking on input
    would hang the server forever with no visible cause, so in that case it
    prints what to do and carries on with the NSE fallback instead.
    """
    today = time.strftime("%Y-%m-%d", time.gmtime(time.time() + 5 * 3600 + 1800))
    tok_date = _fyers_token_date()
    if tok_date == today:
        print(f"[fyers] token generated today ({tok_date}) - no login needed")
        return True

    reason = "no saved token" if not tok_date else f"token is from {tok_date}, not today"
    print(f"[fyers] {reason}")

    if not _envbool_g("NSE_FYERS_AUTO_LOGIN", True):
        print("[fyers] auto-login disabled (NSE_FYERS_AUTO_LOGIN=false). "
              "Run: python3 fyers_login.py")
        return False

    if not sys.stdin.isatty():
        # Headless: say exactly what is wrong and what to do, rather than
        # blocking on an input nobody can provide.
        print("[fyers] cannot run the interactive login: stdin is not a terminal.")
        print("[fyers] Run this from a terminal first:  python3 fyers_login.py")
        print("[fyers] Starting with NSE as the data source for now.")
        return False

    script = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "fyers_login.py")
    if not _os.path.exists(script):
        print(f"[fyers] fyers_login.py not found next to the server - cannot auto-login")
        return False

    print("=" * 66)
    print("  Fyers login required (tokens expire daily, around 06:00 IST)")
    print("  A browser will open. Log in, then paste the redirected URL below.")
    print("=" * 66)
    try:
        import subprocess
        # stdin/stdout inherited so the prompt and paste work normally
        rc = subprocess.call([sys.executable, script])
    except KeyboardInterrupt:
        print("\n[fyers] login cancelled - starting with NSE instead")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[fyers] login failed to run: {e}")
        return False
    if rc != 0:
        print(f"[fyers] login exited with code {rc} - starting with NSE instead")
        return False
    if _fyers_token_date() == today:
        print("[fyers] token refreshed - continuing startup")
        return True
    print("[fyers] login finished but no fresh token was written - "
          "starting with NSE instead")
    return False


# ══════════════════════════════════════════════════════════════════════
# INDEX CONSTITUENT WEIGHTS
# These were hardcoded in the server. NSE rebalances quarterly and changes
# constituents outright, so a hardcoded table silently drifts from reality -
# and every point-contribution figure derived from it drifts with it, without
# anything looking wrong. They now live in index_weights.json, refreshable on
# demand, with the hardcoded table kept only as a last-resort fallback.
# ══════════════════════════════════════════════════════════════════════
_WEIGHTS_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                              "index_weights.json")
_weights_cache: dict = {}


def _load_index_weights() -> dict:
    """Read the weights file, or {} if absent/unreadable."""
    global _weights_cache
    if _weights_cache:
        return _weights_cache
    try:
        if _os.path.exists(_WEIGHTS_FILE):
            with open(_WEIGHTS_FILE) as f:
                data = json.load(f)
            _weights_cache = data
            upd = data.get("_updated", "?")
            n = sum(len(v.get("weights", {})) for k, v in data.items() if not k.startswith("_"))
            print(f"[weights] loaded {n} constituent weights from "
                  f"{_os.path.basename(_WEIGHTS_FILE)} (updated {upd})")
            return data
    except Exception as e:  # noqa: BLE001
        print(f"[weights] could not read {_WEIGHTS_FILE}: {e}")
    return {}


def _save_index_weights(data: dict) -> bool:
    try:
        data["_updated"] = time.strftime("%Y-%m-%d %H:%M")
        tmp = _WEIGHTS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        _os.replace(tmp, _WEIGHTS_FILE)          # atomic
        global _weights_cache
        _weights_cache = data
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[weights] write failed: {e}")
        return False


def _fetch_index_weights(symbol: str, fetcher=None) -> dict:
    """Derive live weights from NSE's index constituent feed.

    NSE does not publish the free-float weight directly on this endpoint, but
    it does publish each constituent's index point contribution and the index
    level - and weight is recoverable from those. Where that is unavailable we
    fall back to relative traded value, which tracks weight loosely enough to
    rank contributors but is NOT a substitute for the real figure, so the file
    records which method produced it.
    """
    idx_name = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                "FINNIFTY": "NIFTY FIN SERVICE",
                "MIDCPNIFTY": "NIFTY MID SELECT"}.get(symbol.upper())
    if not idx_name:
        return {}
    f = fetcher or _shared_fetcher
    if f is None:
        print(f"[weights] no NSE session - load an option chain first")
        return {}
    try:
        from nse_options_strategy import API_HEADERS
        from urllib.parse import quote_plus as _qp
        h = dict(API_HEADERS)
        h["Referer"] = "https://www.nseindia.com/market-data/live-equity-market"
        r = f.session.get(
            "https://www.nseindia.com/api/equity-stockIndices?index=" + _qp(idx_name),
            headers=h, timeout=15)
        if r.status_code != 200:
            print(f"[weights] {idx_name}: HTTP {r.status_code}")
            return {}
        rows = (r.json() or {}).get("data") or []
    except Exception as e:  # noqa: BLE001
        print(f"[weights] fetch failed: {e}")
        return {}
    if len(rows) < 5:
        print(f"[weights] {idx_name}: only {len(rows)} rows returned")
        return {}

    idx_row = next((x for x in rows if x.get("symbol") in (idx_name, symbol.upper())), None)
    members = [x for x in rows if x is not idx_row and x.get("symbol")]
    weights, method = {}, "traded-value proxy"
    # preferred: recover weight from each name's point contribution
    idx_val = float((idx_row or {}).get("lastPrice") or 0)
    got_pts = 0
    if idx_val:
        for m in members:
            try:
                pchg = float(m.get("pChange") or 0)
                pts = m.get("indexPoints") or m.get("pointchange")
                if pts is None or not pchg:
                    continue
                # pts = pchg/100 * weight/100 * index  ->  weight follows
                w = float(pts) / (pchg / 100.0) / idx_val * 100.0
                if 0 < w < 40:
                    weights[m["symbol"]] = round(w, 3)
                    got_pts += 1
            except Exception:
                continue
    if got_pts >= len(members) * 0.6:
        method = "point-contribution"
    else:
        weights = {}
        total = sum(float(m.get("totalTradedValue") or 0) for m in members) or 1.0
        for m in members:
            tv = float(m.get("totalTradedValue") or 0)
            if tv > 0:
                weights[m["symbol"]] = round(tv / total * 100, 3)
    if not weights:
        return {}
    # normalise to 100
    s = sum(weights.values()) or 1.0
    weights = {k: round(v / s * 100, 3) for k, v in weights.items()}
    print(f"[weights] {symbol}: {len(weights)} constituents via {method}")
    return {"weights": weights, "method": method, "count": len(weights),
            "fetched": time.strftime("%Y-%m-%d %H:%M")}


def _refresh_all_weights(symbols=("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")) -> dict:
    data = dict(_load_index_weights())
    changed = 0
    for sym in symbols:
        got = _fetch_index_weights(sym)
        if got.get("weights"):
            old = set((data.get(sym) or {}).get("weights", {}))
            new = set(got["weights"])
            if old and old != new:
                added, gone = new - old, old - new
                print(f"[weights] {sym} membership changed: "
                      f"+{sorted(added)[:5]} -{sorted(gone)[:5]}")
            data[sym] = got
            changed += 1
    if changed:
        _save_index_weights(data)
    return data


# ══════════════════════════════════════════════════════════════════════
# DISK RETENTION
# Nothing deleted files before this: the janitor pruned memory only, so the
# archives grew without limit. replay/ is ~32 MB a day for two symbols, which
# is 7.8 GB a year - the only folder where that matters.
#
# The retention differs per folder because the DATA differs:
#   replay/   the only record of what the chain looked like minute by minute,
#             and unreconstructable once gone. Claims on Trial reads across
#             dates, so this needs weeks, not days. Deleted last and kept
#             longest, because NSE will not sell it back to you.
#   ticks/    20 KB a day. Deleting it saves nothing and costs restart
#             recovery, so it is kept far longer than it needs to be.
#   bhavcopy/ only yesterday's file is ever read, and every file is
#             re-downloadable from NSE. The cheapest thing here to lose.
#
# Runs weekly rather than on the 5-minute janitor cycle: walking three
# directories is wasted work at that frequency, and a retention policy that
# fires 2,000 times a day is a policy nobody can reason about.
# ══════════════════════════════════════════════════════════════════════
RUN_CLEANUP = True                 # enable the weekly disk pass
RUN_CLEANUP_REPLAY_DAYS = 60       # option-chain snapshots (Claims on Trial)
RUN_CLEANUP_TICKS_DAYS = 365       # spot samples - tiny, kept generously
RUN_CLEANUP_BHAV_DAYS = 30         # end-of-day files, re-downloadable
RUN_CLEANUP_DRY_RUN = False        # log what WOULD go, delete nothing


def _file_day(name: str):
    """Extract a YYYY-MM-DD date from an archive filename, or None.

    Imports locally because this file has NO module-level datetime or re
    import - every one of the eight datetime imports here sits inside a
    function. Reaching for a name another function imported works until that
    function is not the one running, which is exactly how this failed the
    first time it was called from the endpoint rather than from the loop.
    """
    import re as _re
    from datetime import datetime as _dtc
    m = _re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if not m:
        return None
    try:
        return _dtc(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def _cleanup_dir(path: str, keep_days: int, pattern: str, dry: bool) -> dict:
    """Delete files in one directory older than keep_days.

    Dates come from the FILENAME, not the modification time: a file touched by
    a backup or a copy would otherwise look new and survive forever, and one
    restored from an archive would look old and be deleted immediately.
    """
    import glob as _glob
    from datetime import datetime as _dtc, timedelta as _td
    out = {"dir": _os.path.basename(path), "kept": 0, "removed": 0, "bytes": 0, "errors": 0}
    if not _os.path.isdir(path):
        return out
    cutoff = _dtc.today() - _td(days=keep_days)
    for f in _glob.glob(_os.path.join(path, pattern)):
        day = _file_day(_os.path.basename(f))
        if day is None or day.date() >= cutoff.date():
            out["kept"] += 1
            continue
        try:
            sz = _os.path.getsize(f)
            if not dry:
                _os.remove(f)
            out["removed"] += 1
            out["bytes"] += sz
        except Exception as e:  # noqa: BLE001
            out["errors"] += 1
            print(f"[cleanup] could not remove {f}: {e}")
    return out


def _run_cleanup(dry: bool = None) -> dict:
    dry = RUN_CLEANUP_DRY_RUN if dry is None else dry
    jobs = [
        (_REPLAY_DIR, int(_os.environ.get("NSE_KEEP_REPLAY_DAYS", RUN_CLEANUP_REPLAY_DAYS)), "*.jsonl"),
        (_TICK_DIR, int(_os.environ.get("NSE_KEEP_TICKS_DAYS", RUN_CLEANUP_TICKS_DAYS)), "*.csv"),
        (_BHAVCOPY_DIR, int(_os.environ.get("NSE_KEEP_BHAV_DAYS", RUN_CLEANUP_BHAV_DAYS)), "*"),
    ]
    res, freed, removed = [], 0, 0
    for path, days, pat in jobs:
        r = _cleanup_dir(path, days, pat, dry)
        r["keep_days"] = days
        res.append(r)
        freed += r["bytes"]
        removed += r["removed"]
    if removed:
        print(f"[cleanup]{' DRY RUN -' if dry else ''} removed {removed} file(s), "
              f"freed {freed / 1048576:.1f} MB")
        for r in res:
            if r["removed"]:
                print(f"[cleanup]   {r['dir']}: {r['removed']} older than "
                      f"{r['keep_days']}d ({r['bytes'] / 1048576:.1f} MB)")
    else:
        print(f"[cleanup] nothing older than the retention window")
    return {"dry_run": dry, "removed": removed, "freed_mb": round(freed / 1048576, 1),
            "detail": res, "ran": time.strftime("%Y-%m-%d %H:%M")}


_cleanup_state = {"last": 0.0, "result": None}


def _cleanup_loop():
    """Weekly, and once shortly after startup if a week has passed."""
    stamp = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".last_cleanup")
    try:
        if _os.path.exists(stamp):
            _cleanup_state["last"] = _os.path.getmtime(stamp)
    except Exception:
        pass
    WEEK = 7 * 86400
    while not _poller_stop.is_set():
        due = time.time() - _cleanup_state["last"] >= WEEK
        if due:
            try:
                _cleanup_state["result"] = _run_cleanup()
                _cleanup_state["last"] = time.time()
                with open(stamp, "w") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M"))
            except Exception as e:  # noqa: BLE001
                print(f"[cleanup] failed: {e}")
        _poller_stop.wait(3600)          # check hourly, act weekly


def _start_cleanup():
    if not _envbool_g("NSE_CLEANUP", RUN_CLEANUP):
        print("[i] Weekly disk cleanup DISABLED (NSE_CLEANUP=false)")
        return
    threading.Thread(target=_cleanup_loop, daemon=True).start()
    print(f"[i] Weekly disk cleanup: replay {RUN_CLEANUP_REPLAY_DAYS}d, "
          f"ticks {RUN_CLEANUP_TICKS_DAYS}d, bhavcopy {RUN_CLEANUP_BHAV_DAYS}d")


def _prune_state() -> None:
    # the rollup history is per-strike and per-symbol; prune it with the rest
    try:
        cutoff = time.time() - 4200
        for sym in list(_oi_rollup_hist):
            for k in list(_oi_rollup_hist[sym]):
                h = [x for x in _oi_rollup_hist[sym][k] if x[0] > cutoff]
                if h:
                    _oi_rollup_hist[sym][k] = h
                else:
                    del _oi_rollup_hist[sym][k]
            if not _oi_rollup_hist[sym]:
                del _oi_rollup_hist[sym]
    except Exception:
        pass

    """Bound the in-memory maps so a long-running server does not creep.

    None of these need history beyond the current session: caches are
    re-fetched, fired-maps only prevent immediate repeats, and per-day
    structures are meaningless once the date rolls. Without this a server left
    up for a week accumulates every strike it has ever seen.
    """
    now = time.time()
    try:
        with _chain_meta_lock:
            for k, v in list(_chain_cache.items()):
                if now - v.get("ts", 0) > 900:            # 15 min of staleness
                    _chain_cache.pop(k, None)
                    _chain_locks.pop(k, None)
        for d, ttl in ((_flow_fired, 3600), (_m920_fired, 86400)):
            for k, t in list(d.items()):
                if now - t > ttl:
                    d.pop(k, None)
        today = time.strftime("%Y-%m-%d")
        for k, v in list(_m920.items()):
            if v.get("day") and v["day"] != time.strftime("%a %b %d %Y") and v.get("day") != today:
                # keep only the current session's frozen levels
                if now - v.get("_ts", now) > 86400:
                    _m920.pop(k, None)
        for k, v in list(_flow_prev.items()):
            if now - v.get("ts", 0) > 1800:
                _flow_prev.pop(k, None)
        for name in ("_ltp_history", "_oi_timeline"):
            d = globals().get(name)
            if isinstance(d, dict):
                for k, v in list(d.items()):
                    if isinstance(v, list) and len(v) > 3000:
                        d[k] = v[-1500:]
                if len(d) > 400:
                    for k in list(d)[:len(d) - 400]:
                        d.pop(k, None)
        fut = globals().get("_futures_cache")
        if isinstance(fut, dict) and len(fut) > 50:
            for k in list(fut)[:len(fut) - 50]:
                fut.pop(k, None)
    except Exception as e:  # noqa: BLE001
        print(f"[janitor] {e}")


def _start_janitor() -> None:
    def loop():
        while True:
            time.sleep(300)
            _prune_state()
    threading.Thread(target=loop, daemon=True).start()
    print("[i] State janitor running (prunes stale caches every 5 min)")


def _fetch_via_adapter(symbol: str, expiry, band: int) -> dict:
    """Route the chain fetch to the configured adapter.

    Any adapter failure falls back to the built-in NSE path rather than
    taking the dashboard down: a broker outage should degrade the data,
    not the tool.
    """
    src = RUN_DATA_SOURCE
    if src == "fyers":
        try:
            import nse_adapter_fyers as _fy
            return _attach_oi_rollups(symbol, _fy.fetch_chain(symbol, expiry, band))
        except Exception as e:  # noqa: BLE001
            print(f"[adapter] fyers failed ({e}) - using NSE for this request")
    if src == "arrow":
        try:
            import nse_adapter_arrow as _arrow
            return _attach_oi_rollups(symbol, _arrow.fetch_chain(symbol, expiry, band))
        except Exception as e:  # noqa: BLE001
            print(f"[adapter] arrow failed ({e}) - using NSE for this request")
    return _build_response(symbol, expiry, band)


def _adapter_streams() -> bool:
    """True when the active adapter pushes its own data.

    When it does, the NSE background poller is pointless (and would keep
    scraping a source we are no longer using), so it is not started.
    """
    return RUN_DATA_SOURCE in ("arrow", "fyers")


_poller_stop = threading.Event()
_poller_symbols: list = []

# ══════════════════════════════════════════════════════════════════════
# KEEP-AWAKE
# A locked or idle machine will suspend this process, so polling, alert
# detection and the SSE stream all stall until someone touches the keyboard.
# Three defences, applied together because each covers a different case:
#   * macOS  - spawn `caffeinate` to block idle sleep AND App Nap. App Nap in
#              particular throttles a "background" process to ~1 wakeup/minute,
#              which is what makes the poller appear to freeze rather than die.
#   * Windows- SetThreadExecutionState tells the OS this process is doing
#              real work and must not be put to sleep.
#   * Linux  - systemd-inhibit when available.
# A watchdog then reports any gap larger than expected, so if the OS suspends
# us anyway the log says so instead of leaving you guessing.
# ══════════════════════════════════════════════════════════════════════
_caffeinate_proc = None


def _keep_awake(enable: bool = True) -> str:
    """Ask the OS not to suspend this process. Returns what was applied."""
    global _caffeinate_proc
    if not enable:
        return "disabled"
    plat = sys.platform
    try:
        if plat == "darwin":
            import subprocess
            # -i no idle sleep, -m no disk sleep, -s no system sleep on AC,
            # -w tie the caffeinate lifetime to THIS pid so it dies with us
            _caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-i", "-m", "-s", "-w", str(_os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return "macOS caffeinate (idle+disk+system sleep blocked)"
        if plat == "win32":
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            return "Windows SetThreadExecutionState (sleep blocked)"
        if plat.startswith("linux"):
            import shutil, subprocess
            if shutil.which("systemd-inhibit"):
                _caffeinate_proc = subprocess.Popen(
                    ["systemd-inhibit", "--what=idle:sleep", "--who=1OPTIONS",
                     "--why=market data poller", "sleep", "infinity"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return "systemd-inhibit (idle+sleep blocked)"
            return "no inhibitor found - install systemd or run with caffeine"
    except FileNotFoundError:
        return f"keep-awake helper not found on {plat}"
    except Exception as e:  # noqa: BLE001
        return f"keep-awake failed: {e}"
    return f"no keep-awake available for {plat}"


def _release_awake() -> None:
    global _caffeinate_proc
    try:
        if _caffeinate_proc:
            _caffeinate_proc.terminate()
            _caffeinate_proc = None
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # ES_CONTINUOUS
    except Exception:
        pass


def _watchdog_loop(expected: float):
    """Report suspensions instead of letting them pass silently.

    If the wall clock jumps much further than the sleep we asked for, the
    process was suspended - by a lock screen, a lid close, or the OS. Saying
    so in the log turns "the poller randomly stopped" into a diagnosable fact.
    """
    tick = 10.0
    last = time.time()
    while not _poller_stop.is_set():
        _poller_stop.wait(tick)
        now = time.time()
        drift = now - last - tick
        if drift > max(20.0, expected * 2):
            print(f"[watchdog] process was suspended for ~{drift:.0f}s "
                  f"(machine slept or was throttled) - resuming; "
                  f"chain cache and alert baselines will re-prime on the next tick")
            # a long gap makes the flow baselines meaningless: reset them so
            # the first comparison after waking is not against stale data
            try:
                _flow_prev.clear()
            except Exception:
                pass
        last = now


def _poller_loop(interval: float):
    """Keep hot symbols warm so browsers essentially always hit cache."""
    while not _poller_stop.is_set():
        for sym in list(_poller_symbols):
            if _poller_stop.is_set():
                break
            try:
                _cached_chain(sym, None, 12, client="poller")
            except Exception as e:  # noqa: BLE001
                print(f"[poller] {sym}: {e}")
        _poller_stop.wait(interval)

def _start_poller(symbols, interval: float):
    global _poller_symbols
    _poller_symbols = [s.upper() for s in symbols if s]
    if not _poller_symbols:
        return
    t = threading.Thread(target=_poller_loop, args=(interval,), daemon=True)
    t.start()
    threading.Thread(target=_watchdog_loop, args=(interval,), daemon=True).start()
    print(f"[i] Background poller: {', '.join(_poller_symbols)} every {interval}s "
          f"(browsers read the shared cache; NSE sees one request per symbol per cycle)")

_TICK_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "ticks")
_REPLAY_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "replay")
_REPLAY_INTERVAL = 60
_replay_due: dict = {}    # sym -> bool (set by chain fetch, consumed after response build)
_replay_last: dict = {}   # sym -> ts

def _record_replay_snapshot(symbol: str) -> None:
    sym = symbol.upper()
    if time.time() - _replay_last.get(sym, 0) >= _REPLAY_INTERVAL:
        _replay_due[sym] = True

def _write_replay_snapshot(symbol: str, response: dict) -> None:
    sym = symbol.upper()
    if not _replay_due.pop(sym, False):
        return
    _replay_last[sym] = time.time()
    try:
        _os.makedirs(_REPLAY_DIR, exist_ok=True)
        day = time.strftime("%Y-%m-%d")
        path = _os.path.join(_REPLAY_DIR, f"{sym}_{day}.jsonl")
        slim = {k: v for k, v in response.items() if k not in ("strategies",)}
        slim["_replay_ts"] = int(time.time())
        with open(path, "a") as f:
            f.write(json.dumps(slim, separators=(",", ":")) + "\n")
    except Exception as e:  # noqa: BLE001
        print(f"[!] Replay snapshot write failed (non-fatal): {e}")

def _replay_index(symbol: str, day: str) -> dict:
    path = _os.path.join(_REPLAY_DIR, f"{symbol.upper()}_{day}.jsonl")
    if not _os.path.exists(path):
        return {"symbol": symbol, "date": day, "timestamps": []}
    ts = []
    with open(path) as f:
        for line in f:
            try:
                ts.append(json.loads(line).get("_replay_ts", 0))
            except Exception:
                ts.append(0)
    return {"symbol": symbol, "date": day, "timestamps": ts}

def _replay_snapshot(symbol: str, day: str, idx: int) -> dict | None:
    path = _os.path.join(_REPLAY_DIR, f"{symbol.upper()}_{day}.jsonl")
    if not _os.path.exists(path):
        return None
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                try:
                    return json.loads(line)
                except Exception:
                    return None
    return None

def _replay_dates(symbol: str) -> list:
    if not _os.path.isdir(_REPLAY_DIR):
        return []
    pre = symbol.upper() + "_"
    return sorted(f[len(pre):-6] for f in _os.listdir(_REPLAY_DIR)
                  if f.startswith(pre) and f.endswith(".jsonl"))


# ── Rule-based intraday backtester over recorded replay data ────────────────
# StockMock-style: time-based entry/exit, ATM-offset or delta strike selection,
# per-leg SL / profit-lock, combined SL/target. Runs on ./replay/*.jsonl files
# recorded by this server (1 snapshot/min) — your own data, unlimited runs.
import math as _math

def _bt_norm_cdf(x):
    return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))

def _bt_delta(spot, strike, dte_days, iv_pct, opt_type):
    try:
        T = max(0.25, dte_days) / 365.0
        sig = max(0.005, iv_pct / 100.0)
        d1 = (_math.log(spot / strike) + (0.07 + sig * sig / 2) * T) / (sig * _math.sqrt(T))
        return _bt_norm_cdf(d1) if opt_type == "CE" else _bt_norm_cdf(d1) - 1.0
    except Exception:
        return None

def _bt_load_day(symbol, day):
    path = _os.path.join(_REPLAY_DIR, f"{symbol.upper()}_{day}.jsonl")
    if not _os.path.exists(path):
        return []
    snaps = []
    with open(path) as f:
        for line in f:
            try:
                snaps.append(json.loads(line))
            except Exception:
                pass
    return snaps

def _bt_hhmm(ts):
    return time.strftime("%H:%M", time.localtime(ts))

def _bt_resolve_strike(snap, leg):
    """strike_sel: 'ATM', 'ATM+2', 'ATM-1', or 'delta:0.25'."""
    sel = str(leg.get("strike_sel", "ATM")).upper()
    gap = snap.get("strike_gap") or 50
    atm = snap.get("atm")
    if sel.startswith("DELTA:"):
        try:
            target = abs(float(sel.split(":", 1)[1]))
        except ValueError:
            return None
        spot, dte = snap.get("underlying_value"), snap.get("dte", 1)
        best, bd = None, 1e9
        for s in snap.get("strikes", []):
            iv = s.get("ce_iv" if leg["type"] == "CE" else "pe_iv") or 0
            if iv <= 0.5:
                continue
            d = _bt_delta(spot, s["strike"], dte, iv, leg["type"])
            if d is None:
                continue
            diff = abs(abs(d) - target)
            if diff < bd:
                bd, best = diff, s["strike"]
        return best
    off = 0
    if sel.startswith("ATM+"):
        off = int(sel[4:] or 0)
    elif sel.startswith("ATM-"):
        off = -int(sel[4:] or 0)
    return (atm + off * gap) if atm else None

def _bt_ltp(snap, strike, opt_type):
    for s in snap.get("strikes", []):
        if s["strike"] == strike:
            return s.get("ce_ltp" if opt_type == "CE" else "pe_ltp")
    return None

def run_backtest(symbol, spec):
    days = spec.get("dates") or _replay_dates(symbol)
    entry_t, exit_t = spec.get("entry_time", "09:20"), spec.get("exit_time", "15:15")
    legs_spec = spec.get("legs", [])
    sl_pct = spec.get("sl_pct")            # per-leg stop, % of entry premium
    tgt_pct = spec.get("target_pct")       # per-leg profit lock, % of entry premium
    comb_sl = spec.get("combined_sl")      # ₹ per lot (positive number)
    comb_tgt = spec.get("combined_target") # ₹ per lot
    if not legs_spec:
        return {"error": "Need legs"}
    results = []
    for day in days:
        snaps = _bt_load_day(symbol, day)
        if len(snaps) < 3:
            continue
        lot = snaps[0].get("lot_size") or 50
        # entry snapshot = first at/after entry_time
        ei = next((i for i, s in enumerate(snaps) if _bt_hhmm(s.get("_replay_ts", 0)) >= entry_t), None)
        if ei is None or ei >= len(snaps) - 1:
            continue
        entry_snap = snaps[ei]
        legs = []
        ok = True
        for ls in legs_spec:
            k = _bt_resolve_strike(entry_snap, ls)
            p = _bt_ltp(entry_snap, k, ls["type"]) if k else None
            if not k or not p:
                ok = False
                break
            legs.append({"side": ls["side"], "type": ls["type"], "strike": k, "qty": int(ls.get("qty", 1)),
                         "entry": p, "exit": None, "exit_reason": None, "exit_time": None})
        if not ok:
            continue
        day_min_mtm = 0.0
        exit_reason = "time"
        # walk forward
        for i in range(ei + 1, len(snaps)):
            snap = snaps[i]
            hhmm = _bt_hhmm(snap.get("_replay_ts", 0))
            mtm = 0.0
            all_closed = True
            for leg in legs:
                if leg["exit"] is not None:
                    mtm += (leg["entry"] - leg["exit"] if leg["side"] == "SELL" else leg["exit"] - leg["entry"]) * leg["qty"]
                    continue
                ltp = _bt_ltp(snap, leg["strike"], leg["type"])
                if ltp is None:
                    all_closed = False
                    continue
                # per-leg SL / target
                if leg["side"] == "SELL":
                    if sl_pct and ltp >= leg["entry"] * (1 + sl_pct / 100.0):
                        leg["exit"], leg["exit_reason"], leg["exit_time"] = ltp, f"leg SL {sl_pct}%", hhmm
                    elif tgt_pct and ltp <= leg["entry"] * (1 - tgt_pct / 100.0):
                        leg["exit"], leg["exit_reason"], leg["exit_time"] = ltp, f"profit lock {tgt_pct}%", hhmm
                else:
                    if sl_pct and ltp <= leg["entry"] * (1 - sl_pct / 100.0):
                        leg["exit"], leg["exit_reason"], leg["exit_time"] = ltp, f"leg SL {sl_pct}%", hhmm
                    elif tgt_pct and ltp >= leg["entry"] * (1 + tgt_pct / 100.0):
                        leg["exit"], leg["exit_reason"], leg["exit_time"] = ltp, f"profit lock {tgt_pct}%", hhmm
                cur = leg["exit"] if leg["exit"] is not None else ltp
                if leg["exit"] is None:
                    all_closed = False
                mtm += (leg["entry"] - cur if leg["side"] == "SELL" else cur - leg["entry"]) * leg["qty"]
            mtm_rs = mtm * lot
            day_min_mtm = min(day_min_mtm, mtm_rs)
            # combined SL / target / time exit
            hit_comb_sl = comb_sl and mtm_rs <= -abs(comb_sl)
            hit_comb_tgt = comb_tgt and mtm_rs >= abs(comb_tgt)
            if hit_comb_sl or hit_comb_tgt or hhmm >= exit_t or all_closed:
                for leg in legs:
                    if leg["exit"] is None:
                        ltp = _bt_ltp(snap, leg["strike"], leg["type"])
                        leg["exit"] = ltp if ltp is not None else leg["entry"]
                        leg["exit_time"] = hhmm
                        leg["exit_reason"] = "combined SL" if hit_comb_sl else ("combined target" if hit_comb_tgt else "time exit")
                exit_reason = "combined SL" if hit_comb_sl else ("combined target" if hit_comb_tgt else ("all legs closed" if all_closed else "time exit"))
                break
        pnl = sum((l["entry"] - l["exit"] if l["side"] == "SELL" else l["exit"] - l["entry"]) * l["qty"] for l in legs if l["exit"] is not None) * lot
        results.append({"date": day, "pnl": round(pnl, 2), "max_dd": round(day_min_mtm, 2),
                        "exit_reason": exit_reason, "entry_time": _bt_hhmm(entry_snap.get("_replay_ts", 0)),
                        "legs": legs})
    if not results:
        return {"symbol": symbol, "days": [], "summary": {"note": "No usable recorded days for this spec — record more sessions first."}}
    pnls = [r["pnl"] for r in results]
    wins = [p for p in pnls if p > 0]
    summary = {
        "n_days": len(results), "total": round(sum(pnls), 2),
        "win_rate": round(len(wins) / len(results) * 100, 1),
        "avg": round(sum(pnls) / len(results), 2),
        "best": round(max(pnls), 2), "worst": round(min(pnls), 2),
        "max_dd": round(min(r["max_dd"] for r in results), 2),
    }
    return {"symbol": symbol, "days": results, "summary": summary}


# ── Measured levels from the replay archive (TPO, max-pain stats, ToD) ──────
def _replay_all_days(symbol):
    return [(d, _bt_load_day(symbol, d)) for d in _replay_dates(symbol)]

def compute_tpo_levels(symbol):
    """Per recorded day: time-at-price profile POC / VAH / VAL from 1-min spot
    samples, plus naked (unrevisited) status of each prior day's POC."""
    days = []
    for day, snaps in _replay_all_days(symbol):
        spots = [s.get("underlying_value") for s in snaps if s.get("underlying_value")]
        if len(spots) < 30:
            continue
        lo, hi = min(spots), max(spots)
        if hi <= lo:
            continue
        nbins = 40
        w = (hi - lo) / nbins
        bins = [0] * nbins
        for p in spots:
            bins[min(nbins - 1, int((p - lo) / w))] += 1
        poc_i = bins.index(max(bins))
        poc = lo + (poc_i + 0.5) * w
        # value area: expand around POC until 70% of samples covered
        total = sum(bins)
        covered = bins[poc_i]
        a = b = poc_i
        while covered < 0.7 * total and (a > 0 or b < nbins - 1):
            up = bins[b + 1] if b < nbins - 1 else -1
            dn = bins[a - 1] if a > 0 else -1
            if up >= dn:
                b += 1; covered += bins[b]
            else:
                a -= 1; covered += bins[a]
        days.append({"date": day, "poc": round(poc, 1), "vah": round(lo + (b + 1) * w, 1),
                     "val": round(lo + a * w, 1), "hi": round(hi, 1), "lo": round(lo, 1)})
    # naked POC: not touched by any LATER day's range
    for i, d in enumerate(days):
        naked = True
        for later in days[i + 1:]:
            if later["lo"] <= d["poc"] <= later["hi"]:
                naked = False
                break
        d["naked"] = naked and i < len(days) - 1   # today's own POC isn't "naked" yet
    return {"symbol": symbol, "days": days}

def compute_maxpain_stats(symbol):
    """Does price actually converge toward max pain intraday? Measured from
    your own recorded days: |spot−MP| early vs at close, per day."""
    rows = []
    for day, snaps in _replay_all_days(symbol):
        if len(snaps) < 30:
            continue
        early = next((s for s in snaps if _bt_hhmm(s.get("_replay_ts", 0)) >= "10:00"), snaps[0])
        last = snaps[-1]
        mp = early.get("max_pain")
        s0, s1 = early.get("underlying_value"), last.get("underlying_value")
        if not (mp and s0 and s1):
            continue
        d0, d1 = abs(s0 - mp), abs(s1 - mp)
        rows.append({"date": day, "max_pain": mp, "dist_10am": round(d0, 1),
                     "dist_close": round(d1, 1), "converged": d1 < d0,
                     "dte": early.get("dte")})
    if not rows:
        return {"symbol": symbol, "days": [], "summary": {"note": "No recorded days yet."}}
    conv = [r for r in rows if r["converged"]]
    expiry_rows = [r for r in rows if (r["dte"] or 9) <= 0]
    summary = {
        "n_days": len(rows),
        "converge_rate": round(len(conv) / len(rows) * 100, 1),
        "avg_move_toward": round(sum(r["dist_10am"] - r["dist_close"] for r in rows) / len(rows), 1),
        "expiry_days": len(expiry_rows),
        "expiry_converge_rate": round(len([r for r in expiry_rows if r["converged"]]) / len(expiry_rows) * 100, 1) if expiry_rows else None,
    }
    return {"symbol": symbol, "days": rows, "summary": summary}

def compute_tod_seasonality(symbol):
    """Average |Δspot| and range per 15-min bucket across recorded days —
    WHEN does this market actually move?"""
    from collections import defaultdict as _dd
    buckets = _dd(lambda: {"absmove": [], "rng": []})
    for day, snaps in _replay_all_days(symbol):
        by_bucket = _dd(list)
        for s in snaps:
            sp = s.get("underlying_value")
            if not sp:
                continue
            hhmm = _bt_hhmm(s.get("_replay_ts", 0))
            try:
                h, m = int(hhmm[:2]), int(hhmm[3:5])
            except ValueError:
                continue
            key = f"{h:02d}:{(m // 15) * 15:02d}"
            by_bucket[key].append(sp)
        for key, sps in by_bucket.items():
            if len(sps) >= 2:
                buckets[key]["absmove"].append(abs(sps[-1] - sps[0]))
                buckets[key]["rng"].append(max(sps) - min(sps))
    out = []
    for key in sorted(buckets):
        b = buckets[key]
        out.append({"bucket": key,
                    "avg_move": round(sum(b["absmove"]) / len(b["absmove"]), 1),
                    "avg_range": round(sum(b["rng"]) / len(b["rng"]), 1),
                    "n": len(b["absmove"])})
    return {"symbol": symbol, "buckets": out}


# ── Level touch-and-react statistics (grades round numbers on YOUR data) ────
def compute_level_stats(symbol):
    """For every recorded day: how did price react at round 500/1000 levels?
    Touch = within 0.04%% of the level; bounce = reversed >=0.1%% away against
    the approach direction within the next 15 samples; else break/absorb."""
    tiers = {"1000": [], "500": []}
    for day, snaps in _replay_all_days(symbol):
        spots = [s.get("underlying_value") for s in snaps if s.get("underlying_value")]
        if len(spots) < 40:
            continue
        for unit_name, unit in (("1000", 1000), ("500", 500)):
            seen = set()
            for i in range(3, len(spots) - 16):
                p = spots[i]
                lvl = round(p / unit) * unit
                if unit_name == "500" and lvl % 1000 == 0:
                    continue          # pure-500s only; 1000s counted in their own tier
                if lvl in seen or abs(p - lvl) / lvl > 0.0004:
                    continue
                approach = spots[i] - spots[i - 3]
                if abs(approach) < lvl * 0.0003:
                    continue          # drifting sideways, not an approach
                seen.add(lvl)
                fut = spots[i + 1:i + 16]
                moved_back = any((f - lvl) * (1 if approach < 0 else -1) > lvl * 0.001 for f in fut)
                moved_thru = any((f - lvl) * (1 if approach > 0 else -1) > lvl * 0.001 for f in fut)
                if moved_back and not moved_thru:
                    tiers[unit_name].append(1)
                elif moved_thru:
                    tiers[unit_name].append(0)
    out = {}
    for name, arr in tiers.items():
        out[name] = {"touches": len(arr), "bounce_rate": round(sum(arr) / len(arr) * 100, 1) if arr else None}
    return {"symbol": symbol, "tiers": out,
            "note": "bounce = reversed ≥0.1% against approach within ~15 min of a touch (±0.04%)"}


# ── Anchored VWAP over recorded intraday spot (volume-proxied) ──────────────
def compute_anchored_vwap(symbol, day=None):
    """Index has no true tick volume; we proxy 'activity' by straddle premium
    turnover if present, else equal-weight (=> anchored average price). Bands are
    +/- 1 and 2 standard deviations of price around the running VWAP."""
    days = _replay_dates(symbol)
    if not days:
        return {"symbol": symbol, "series": [], "note": "No recorded days yet."}
    day = day or days[-1]
    snaps = _bt_load_day(symbol, day)
    pts = []
    cum_pv = cum_w = cum_pv2 = 0.0
    for s in snaps:
        p = s.get("underlying_value")
        if not p:
            continue
        w = s.get("straddle_premium") or 1.0     # proxy weight
        cum_pv += p * w; cum_w += w; cum_pv2 += p * p * w
        vwap = cum_pv / cum_w
        var = max(0.0, cum_pv2 / cum_w - vwap * vwap)
        sd = var ** 0.5
        pts.append({"t": _bt_hhmm(s.get("_replay_ts", 0)), "price": round(p, 1),
                    "vwap": round(vwap, 1), "sd": round(sd, 1)})
    last = pts[-1] if pts else None
    return {"symbol": symbol, "day": day, "series": pts,
            "current": last, "weighted": any(s.get("straddle_premium") for s in snaps)}

# ── Gap & opening-range statistics from the replay archive ──────────────────
def compute_gap_orb_stats(symbol):
    days = _replay_dates(symbol)
    rows = []
    prev_close = None
    for day in days:
        snaps = _bt_load_day(symbol, day)
        spots = [(s.get("_replay_ts", 0), s.get("underlying_value")) for s in snaps if s.get("underlying_value")]
        if len(spots) < 20:
            prev_close = spots[-1][1] if spots else prev_close
            continue
        day_open = spots[0][1]
        day_close = spots[-1][1]
        day_high = max(p for _, p in spots)
        day_low = min(p for _, p in spots)
        # opening range = first 15 minutes
        t0 = spots[0][0]
        orb = [p for t, p in spots if t - t0 <= 900]
        orb_hi, orb_lo = (max(orb), min(orb)) if orb else (day_open, day_open)
        # did price break the ORB high/low later, and hold to close?
        after = [p for t, p in spots if t - t0 > 900]
        broke_hi = any(p > orb_hi for p in after)
        broke_lo = any(p < orb_lo for p in after)
        held_hi = broke_hi and day_close > orb_hi
        held_lo = broke_lo and day_close < orb_lo
        gap = None
        gap_filled = None
        if prev_close:
            gap = day_open - prev_close
            # gap fill = price traded back through prev_close during the day
            if abs(gap) > 0.0005 * prev_close:
                gap_filled = (day_low <= prev_close <= day_high)
        rows.append({"date": day, "gap": round(gap, 1) if gap is not None else None,
                     "gap_pct": round(gap / prev_close * 100, 2) if gap and prev_close else None,
                     "gap_filled": gap_filled,
                     "orb_hi": round(orb_hi, 1), "orb_lo": round(orb_lo, 1),
                     "orb_range": round(orb_hi - orb_lo, 1),
                     "broke_hi": broke_hi, "broke_lo": broke_lo,
                     "held_hi": held_hi, "held_lo": held_lo})
        prev_close = day_close
    valid = [r for r in rows if r["gap"] is not None]
    gaps = [r for r in valid if r["gap_pct"] is not None and abs(r["gap_pct"]) > 0.05]
    orb_breaks = [r for r in rows if r["broke_hi"] or r["broke_lo"]]
    orb_holds = [r for r in rows if r["held_hi"] or r["held_lo"]]
    summary = {
        "n_days": len(rows),
        "gap_days": len(gaps),
        "gap_fill_rate": round(sum(1 for r in gaps if r["gap_filled"]) / len(gaps) * 100, 1) if gaps else None,
        "orb_break_rate": round(len(orb_breaks) / len(rows) * 100, 1) if rows else None,
        "orb_hold_rate": round(len(orb_holds) / len(orb_breaks) * 100, 1) if orb_breaks else None,
        "avg_orb_range": round(sum(r["orb_range"] for r in rows) / len(rows), 1) if rows else None,
    }
    return {"symbol": symbol, "days": rows[-15:], "summary": summary}

# ── Calendar-effect seasonality (day-of-week, turn-of-month) ────────────────
def compute_calendar_seasonality(symbol):
    import datetime as _dt
    from collections import defaultdict as _dd
    dow = _dd(list)      # 0=Mon .. 4=Fri -> daily % moves
    tom = _dd(list)      # 'turn' (last 2 / first 3 trading days) vs 'mid'
    regime = _dd(list)   # 'low_vix' / 'high_vix' -> daily % moves
    for day in _replay_dates(symbol):
        snaps = _bt_load_day(symbol, day)
        spots = [s.get("underlying_value") for s in snaps if s.get("underlying_value")]
        if len(spots) < 20:
            continue
        move = (spots[-1] - spots[0]) / spots[0] * 100
        vix_vals = [s.get("india_vix") for s in snaps if s.get("india_vix")]
        vix = sum(vix_vals) / len(vix_vals) if vix_vals else None
        try:
            d = _dt.datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            continue
        dow[d.weekday()].append(move)
        dom = d.day
        bucket = "turn" if (dom <= 3 or dom >= 27) else "mid"
        tom[bucket].append(move)
        if vix is not None:
            regime["high_vix" if vix >= 15 else "low_vix"].append(move)
    def stats(arr):
        if not arr:
            return None
        avg = sum(arr) / len(arr)
        up = sum(1 for x in arr if x > 0)
        return {"n": len(arr), "avg_move": round(avg, 3), "up_rate": round(up / len(arr) * 100, 1)}
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {"symbol": symbol,
            "dow": [{"day": dow_names[k], **(stats(dow[k]) or {"n": 0})} for k in range(5)],
            "tom": {k: stats(v) for k, v in tom.items()},
            "regime": {k: stats(v) for k, v in regime.items()}}


# ── Confluence backtester: does a high-confluence zone actually predict a bounce? ──
def _snap_levels(s):
    """Reconstruct the confluence sources present in a replay snapshot.
    Returns list of (price, source_tier) where tier weights match the dashboard:
    positioning=3, crowd=2. (Geometric S9/planetary aren't stored, so this
    validates the DATA-BACKED confluence — the part that should carry signal.)"""
    out = []
    sup = s.get("support"); res = s.get("resistance")
    def strike_of(x):
        if isinstance(x, dict): return x.get("strike")
        return x
    if strike_of(sup): out.append((strike_of(sup), 3, "OC support"))
    if strike_of(res): out.append((strike_of(res), 3, "OC resistance"))
    if s.get("max_pain"): out.append((s["max_pain"], 3, "max pain"))
    # OI walls from strikes (heaviest CE / PE OI)
    strikes = s.get("strikes") or []
    if strikes:
        cw = max(strikes, key=lambda k: k.get("ce_oi", 0) or 0, default=None)
        pw = max(strikes, key=lambda k: k.get("pe_oi", 0) or 0, default=None)
        if cw: out.append((cw.get("strike"), 3, "call wall"))
        if pw: out.append((pw.get("strike"), 3, "put wall"))
    # crowd: round numbers near spot
    spot = s.get("underlying_value")
    if spot:
        for unit in (1000, 500):
            below = int(spot // unit) * unit
            out.append((below, 2, f"round {unit}"))
            out.append((below + unit, 2, f"round {unit}"))
    return [(p, w, lbl) for (p, w, lbl) in out if p]

def _cluster_zones(levels, tol_frac):
    """Cluster nearby levels; zone score = sum of weights."""
    if not levels: return []
    levels = sorted(levels, key=lambda x: x[0])
    zones = []
    for price, w, lbl in levels:
        if zones and abs(price - zones[-1]["center"]) / zones[-1]["center"] < tol_frac:
            z = zones[-1]
            z["members"].append((price, w, lbl)); z["score"] += w
            z["center"] = sum(p * ww for p, ww, _ in z["members"]) / sum(ww for _, ww, _ in z["members"])
        else:
            zones.append({"center": price, "members": [(price, w, lbl)], "score": w})
    return zones

def compute_confluence_backtest(symbol, min_score=3):
    """For each recorded snapshot, build confluence zones, then measure: when
    spot came within 0.1%% of a zone, did it bounce (reverse >=0.15%% away) or
    break (continue >=0.15%% through) over the next ~15 samples? Bucketed by the
    zone's confluence score, so we can see whether MORE confluence => MORE bounce."""
    buckets = {}   # score -> {"touch":n, "bounce":n, "excursion":[...]}
    for day in _replay_dates(symbol):
        snaps = _bt_load_day(symbol, day)
        spots = [s.get("underlying_value") for s in snaps if s.get("underlying_value")]
        if len(spots) < 30:
            continue
        # build zones once per day from a mid-session snapshot (levels are stable intraday)
        mid = snaps[len(snaps) // 2]
        zones = _cluster_zones(_snap_levels(mid), 0.002)
        zones = [z for z in zones if z["score"] >= min_score]
        for z in zones:
            lvl = z["center"]; sc = min(z["score"], 12)
            seen = False
            for i in range(3, len(spots) - 16):
                p = spots[i]
                if abs(p - lvl) / lvl > 0.001:
                    continue
                approach = spots[i] - spots[i - 3]
                if abs(approach) < lvl * 0.0003:
                    continue
                if seen:
                    continue
                seen = True
                fut = spots[i + 1:i + 16]
                bounced = any((f - lvl) * (1 if approach < 0 else -1) > lvl * 0.0015 for f in fut)
                broke = any((f - lvl) * (1 if approach > 0 else -1) > lvl * 0.0015 for f in fut)
                exc = max(abs(f - lvl) for f in fut) if fut else 0
                b = buckets.setdefault(sc, {"touch": 0, "bounce": 0, "break": 0, "exc": []})
                b["touch"] += 1
                if bounced and not broke: b["bounce"] += 1
                elif broke: b["break"] += 1
                b["exc"].append(exc)
    rows = []
    for sc in sorted(buckets):
        b = buckets[sc]
        n = b["touch"]
        rows.append({"score": sc, "touches": n,
                     "bounce_rate": round(b["bounce"] / n * 100, 1) if n else None,
                     "break_rate": round(b["break"] / n * 100, 1) if n else None,
                     "avg_excursion": round(sum(b["exc"]) / len(b["exc"]), 1) if b["exc"] else None})
    total = sum(r["touches"] for r in rows)
    # headline: does bounce rate rise with score? (correlation sign)
    scored = [(r["score"], r["bounce_rate"]) for r in rows if r["bounce_rate"] is not None and r["touches"] >= 3]
    trend = None
    if len(scored) >= 2:
        xs = [x for x, _ in scored]; ys = [y for _, y in scored]
        mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in scored)
        den = sum((x - mx) ** 2 for x in xs) or 1
        slope = num / den
        trend = "rising" if slope > 1 else "flat" if abs(slope) <= 1 else "falling"
    return {"symbol": symbol, "min_score": min_score, "buckets": rows,
            "total_touches": total, "trend": trend,
            "note": "bounce = reversed ≥0.15% against approach within ~15 samples of a touch (±0.1%); validates data-backed confluence only (geometric sources not stored in replay)"}

# ── Alert outcome journal: what did price do after each fired alert? ──
_ALERT_LOG = _os.path.join(_REPLAY_DIR, "_alert_journal.jsonl")
def log_alert_outcome(payload):
    try:
        _os.makedirs(_REPLAY_DIR, exist_ok=True)
        payload["_ts"] = int(time.time())
        with open(_ALERT_LOG, "a") as f:
            f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        return True
    except Exception as e:
        print(f"[!] alert-journal write failed: {e}")
        return False

def compute_alert_journal(symbol):
    if not _os.path.exists(_ALERT_LOG):
        return {"symbol": symbol, "entries": [], "summary": {}}
    entries = []
    with open(_ALERT_LOG) as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("symbol", "").upper() == symbol.upper():
                    entries.append(e)
            except Exception:
                pass
    resolved = [e for e in entries if e.get("outcome_pts") is not None]
    favorable = sum(1 for e in resolved if e.get("favorable"))
    summary = {
        "total": len(entries), "resolved": len(resolved),
        "favorable_rate": round(favorable / len(resolved) * 100, 1) if resolved else None,
        "avg_move_pts": round(sum(abs(e["outcome_pts"]) for e in resolved) / len(resolved), 1) if resolved else None,
    }
    return {"symbol": symbol, "entries": entries[-40:], "summary": summary}


# ── Results / earnings calendar (NSE event calendar, cached 1h) ──────────────
_results_cal_cache = {"ts": 0.0, "data": None}

def _fetch_results_calendar(fetcher):
    now = time.time()
    if _results_cal_cache["data"] is not None and now - _results_cal_cache["ts"] < 3600:
        return _results_cal_cache["data"]
    try:
        from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
        h = dict(API_HEADERS)
        h["Referer"] = NSE_OC_PAGE
        r = fetcher.session.get("https://www.nseindia.com/api/event-calendar", headers=h, timeout=10)
        if r.status_code != 200:
            raise NSEFetchError(f"event-calendar HTTP {r.status_code}")
        raw = r.json()
        events = []
        for e in (raw if isinstance(raw, list) else raw.get("data", [])):
            purpose = (e.get("purpose") or e.get("bm_purpose") or "")
            if "result" not in purpose.lower():
                continue
            events.append({"symbol": e.get("symbol"), "company": e.get("company") or e.get("sm_name"),
                           "date": e.get("date") or e.get("bm_date"), "purpose": purpose})
        events = events[:60]
        _results_cal_cache.update(ts=now, data=events)
        return events
    except Exception as e:  # noqa: BLE001
        print(f"[!] Results calendar fetch failed (non-fatal): {e}")
        return _results_cal_cache["data"] or []


_oi_timeline: dict = {}      # sym -> {times, ce:{k:[..]}, pe:{k:[..]}, _prev:{k:(ce,pe)}, _last_ts}
# separate from _oi_timeline above: sym -> {strike: [(ts, ce_oi, pe_oi), ...]}
# used only for the rolling 15m/60m OI deltas attached to adapter responses
_oi_rollup_hist: dict = {}
_OI_TIMELINE_INTERVAL = 180  # 3 minutes
_OI_TIMELINE_MAX_COLS = 130  # ~6.5h of samples

def _record_oi_timeline(symbol: str, strikes: list) -> None:
    now = time.time()
    sym = symbol.upper()
    tl = _oi_timeline.setdefault(sym, {"times": [], "ce": {}, "pe": {}, "_prev": {}, "_last_ts": 0.0})
    if (now - tl["_last_ts"]) < _OI_TIMELINE_INTERVAL:
        return
    tl["_last_ts"] = now
    tl["times"].append(time.strftime("%H:%M", time.localtime(now)))
    n = len(tl["times"])
    for s in strikes:
        k = s.strike
        ce_oi, pe_oi = (s.ce_oi or 0), (s.pe_oi or 0)
        prev_ce, prev_pe = tl["_prev"].get(k, (ce_oi, pe_oi))   # first sample = 0 delta
        tl["ce"].setdefault(k, [0] * (n - 1)).append(ce_oi - prev_ce)
        tl["pe"].setdefault(k, [0] * (n - 1)).append(pe_oi - prev_pe)
        tl["_prev"][k] = (ce_oi, pe_oi)
    # pad strikes missing from this sample, then trim width
    for d in (tl["ce"], tl["pe"]):
        for k in d:
            if len(d[k]) < n:
                d[k].append(0)
    if n > _OI_TIMELINE_MAX_COLS:
        drop = n - _OI_TIMELINE_MAX_COLS
        tl["times"] = tl["times"][drop:]
        for d in (tl["ce"], tl["pe"]):
            for k in d:
                d[k] = d[k][drop:]

# ── Strike LTP history (mini price charts per builder leg) ──────────────────
# 1-min LTP samples for every strike, piggybacked on chain fetches. Memory-
# bounded: ~100 strikes × 2 sides × 390 samples of (ts,ltp) per symbol.
_ltp_history: dict = {}      # sym -> {"CE": {strike: [(ts, ltp), ...]}, "PE": {...}, "_last_ts": 0}
_LTP_INTERVAL = 60           # 1 minute
_LTP_MAX_SAMPLES = 400

def _record_ltp_history(symbol: str, strikes: list) -> None:
    now = time.time()
    sym = symbol.upper()
    st = _ltp_history.setdefault(sym, {"CE": {}, "PE": {}, "_last_ts": 0.0})
    if (now - st["_last_ts"]) < _LTP_INTERVAL:
        return
    st["_last_ts"] = now
    for s in strikes:
        for side, ltp in (("CE", s.ce_ltp), ("PE", s.pe_ltp)):
            if not ltp:
                continue
            arr = st[side].setdefault(s.strike, [])
            arr.append((int(now), round(float(ltp), 2)))
            if len(arr) > _LTP_MAX_SAMPLES:
                arr.pop(0)

def _ltp_history_series(symbol: str, strike: float, side: str) -> dict:
    st = _ltp_history.get(symbol.upper(), {})
    arr = st.get(side.upper(), {}).get(strike) or st.get(side.upper(), {}).get(int(strike)) or []
    return {"symbol": symbol, "strike": strike, "side": side.upper(),
            "points": [{"t": t, "ltp": v} for t, v in arr]}


def _oi_timeline_grid(symbol: str) -> dict:
    """Shape the timeline for the dashboard heatmap: rows=strikes, cols=times."""
    tl = _oi_timeline.get(symbol.upper())
    if not tl or not tl["times"]:
        return {"symbol": symbol, "strikes": [], "times": [], "ce": [], "pe": []}
    strikes = sorted(set(tl["ce"]) | set(tl["pe"]))
    n = len(tl["times"])
    def grid(d):
        return [(d.get(k, []) + [0] * n)[:n] for k in strikes]
    return {"symbol": symbol, "strikes": strikes, "times": tl["times"],
            "ce": grid(tl["ce"]), "pe": grid(tl["pe"])}


# ── FII/DII cache ────────────────────────────────────────────────────────────
_fii_dii_cache: dict = {"data": None, "ts": 0.0}
_FII_DII_TTL = 300   # 5 minutes

# ── GIFT Nifty ───────────────────────────────────────────────────────────────
_gift_nifty_cache: dict = {"data": None, "ts": 0.0}
_GIFT_NIFTY_TTL = 60   # 1 minute

def _fetch_gift_nifty() -> dict:
    global _gift_nifty_cache
    now = time.time()
    if _gift_nifty_cache["data"] and (now - _gift_nifty_cache["ts"]) < _GIFT_NIFTY_TTL:
        return _gift_nifty_cache["data"]
    fetcher = _shared_fetcher
    if not fetcher:
        return {"error": "Session not warmed"}
    urls = [
        "https://www.nseindia.com/api/gift-nifty",
        "https://www.nseindia.com/api/getGIFTNifty",
    ]
    for url in urls:
        try:
            r = fetcher.session.get(url,
                headers={"Accept": "application/json, */*",
                         "Referer": "https://www.nseindia.com/"},
                timeout=8)
            if r.status_code != 200:
                continue
            raw = r.json()
            # Normalise various response shapes NSE might use
            if isinstance(raw, dict):
                price  = (raw.get("last") or raw.get("lastPrice") or
                          raw.get("close") or raw.get("value"))
                chg    = (raw.get("change") or raw.get("pChange") or 0)
                chgPct = raw.get("pChange") or raw.get("percentChange") or 0
            elif isinstance(raw, list) and raw:
                item   = raw[0]
                price  = item.get("last") or item.get("lastPrice")
                chg    = item.get("change") or 0
                chgPct = item.get("pChange") or 0
            else:
                continue
            if price:
                data = {"price": float(price), "change": float(chg),
                        "pChange": float(chgPct), "as_of": time.strftime("%H:%M")}
                _gift_nifty_cache = {"data": data, "ts": now}
                return data
        except Exception:
            continue
    stale = _gift_nifty_cache.get("data")
    return stale or {"error": "GIFT Nifty unavailable from NSE"}

# ── Multi-index spot cache (for relative strength) ───────────────────────────
_index_spot_cache: dict[str, float] = {}   # sym → latest spot

# ── Bhavcopy disk cache ──────────────────────────────────────────────────────
# NSE's archive CSVs are immutable once published, so caching them to disk
# is safe indefinitely — we never need to re-download a past date's file.
from pathlib import Path as _Path
_BHAVCOPY_DIR = _Path(__file__).parent / "nse_data_cache" / "bhavcopy"
_BHAVCOPY_DIR.mkdir(parents=True, exist_ok=True)

_BHAV_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "text/csv,*/*",
}

def _bhavcopy_for_date(target_date, fetcher_session) -> str | None:
    """Return bhavcopy CSV text for target_date from cache, or fetch + cache it."""
    from datetime import date as _d
    ddmmyyyy = target_date.strftime("%d%m%Y")
    cache_file = _BHAVCOPY_DIR / f"{ddmmyyyy}.csv"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    url = (f"https://archives.nseindia.com/content/indices/"
           f"ind_close_all_{ddmmyyyy}.csv")
    try:
        r = fetcher_session.get(url, headers=_BHAV_HEADERS, timeout=10)
        if r.status_code == 200 and r.text.strip():
            cache_file.write_text(r.text, encoding="utf-8")
            return r.text
    except Exception:
        pass
    return None

def _parse_bhavcopy_row(csv_text: str, name_aliases: list[str]) -> dict | None:
    """Extract O/H/L/C for one index from a bhavcopy CSV text blob."""
    reader = _csv_mod.DictReader(_io_mod.StringIO(csv_text))
    for row in reader:
        row_name = (row.get("Index Name") or "").strip().upper()
        if any(row_name == a.upper() for a in name_aliases):
            def _bv(*keys):
                for k in keys:
                    v = row.get(k, "")
                    if v:
                        try:
                            return float(str(v).replace(",", "").strip())
                        except ValueError:
                            continue
                return 0.0
            O = _bv("Open Index Value", "Open")
            H = _bv("High Index Value", "High")
            L = _bv("Low Index Value", "Low")
            C = _bv("Closing Index Value", "Close", "Close Index Value")
            if H > L > 0 and C > 0:
                return {"O": O or C, "H": H, "L": L, "C": C}
    return None


def fetch_live_stock(symbol: str, fetcher, headers) -> dict | None:
    """Fetch live OHLC + last price for ANY NSE equity via the warmed session —
    no caching, no local CSV. NSE's quote-equity endpoint returns intraday OHLC
    in priceInfo. Returns dict with price/O/H/L/C/prevClose or None on failure.

    This is 403 to a cold request but succeeds with a warmed session (the same
    session that fetches the option chain). Tries a couple of endpoints so it
    also covers non-F&O cash stocks.
    """
    sym = (symbol or "").strip().upper()
    if not sym or fetcher is None:
        return None
    sess = getattr(fetcher, "session", None)
    if sess is None:
        return None
    h = headers or {}

    # 1) quote-equity — richest: has priceInfo{lastPrice, open, intraDayHighLow,
    #    close, previousClose} plus weekHighLow.
    try:
        r = sess.get("https://www.nseindia.com/api/quote-equity",
                     params={"symbol": sym}, headers=h, timeout=12)
        if r.status_code == 200:
            j = r.json()
            pi = j.get("priceInfo") or {}
            idl = pi.get("intraDayHighLow") or {}
            last = float(pi.get("lastPrice") or 0)
            o = float(pi.get("open") or 0)
            hi = float(idl.get("max") or 0)
            lo = float(idl.get("min") or 0)
            prev = float(pi.get("previousClose") or 0)
            close_ = float(pi.get("close") or 0) or last
            if last > 0:
                # if intraday H/L missing (pre-open), fall back to last/prev band
                if hi <= 0:
                    hi = max(last, o, prev) or last
                if lo <= 0:
                    lo = min(x for x in (last, o, prev) if x > 0)
                chg = last - prev if prev > 0 else 0.0
                chg_pct = (chg / prev * 100) if prev > 0 else 0.0
                return {"symbol": sym, "price": round(last, 2),
                        "O": round(o or prev or last, 2), "H": round(hi, 2),
                        "L": round(lo, 2), "C": round(close_ or last, 2),
                        "prevClose": round(prev or close_ or last, 2),
                        "change": round(chg, 2), "pChange": round(chg_pct, 2),
                        "source": "nse-quote-equity", "live": True}
    except Exception as e:  # noqa: BLE001
        print(f"[i] quote-equity {sym} failed: {e}")

    # 2) fallback — equity trade-info sometimes succeeds when quote-equity is rate-limited
    try:
        r = sess.get("https://www.nseindia.com/api/quote-equity",
                     params={"symbol": sym, "section": "trade_info"}, headers=h, timeout=12)
        if r.status_code == 200:
            j = r.json()
            pi = (j.get("priceInfo") or {})
            last = float(pi.get("lastPrice") or 0)
            if last > 0:
                return {"symbol": sym, "price": round(last, 2),
                        "O": last, "H": last, "L": last, "C": last,
                        "prevClose": last, "source": "nse-trade-info", "live": True}
    except Exception:
        pass
    return None


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


# ── Futures price (basis / cost-of-carry) ────────────────────────────────────
# Nearest-month futures LTP via the quote-derivative API. Cached 30s per symbol;
# best-effort — returns (None, None) on any failure so /api/chain never breaks.
_futures_cache: dict = {}   # symbol -> (ts, (price, dte))
_FUTURES_TTL = 30.0

def _fetch_futures_price(fetcher: NSESession, symbol: str):
    sym = symbol.upper()
    now = time.time()
    cached = _futures_cache.get(sym)
    if cached and (now - cached[0]) < _FUTURES_TTL:
        return cached[1]
    try:
        from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
        from datetime import datetime as _dt
        h = dict(API_HEADERS)
        h["Referer"] = NSE_OC_PAGE
        url = f"https://www.nseindia.com/api/quote-derivative?symbol={sym}"
        r = fetcher.session.get(url, headers=h, timeout=10)
        if r.status_code != 200:
            raise NSEFetchError(f"quote-derivative HTTP {r.status_code}")
        data = r.json()
        futs = []
        for st in data.get("stocks", []):
            meta = st.get("metadata", {})
            if "Future" not in meta.get("instrumentType", ""):
                continue
            try:
                exp = _dt.strptime(meta.get("expiryDate", ""), "%d-%b-%Y")
            except ValueError:
                continue
            price = meta.get("lastPrice")
            if not price:
                continue
            # Open interest and volume matter more than price for signals:
            # futures OI paired with price direction gives the four-way read
            # (long buildup / short buildup / covering / unwinding) on a
            # SINGLE instrument, which is cleaner than inferring it across
            # sixty option strikes where flow is fragmented.
            mkt = st.get("marketDeptOrderBook", {}) or {}
            tinfo = mkt.get("tradeInfo", {}) or {}
            def _f(*keys):
                for src in (tinfo, meta, st):
                    for k in keys:
                        v = src.get(k)
                        if v not in (None, "", "-"):
                            try:
                                return float(str(v).replace(",", ""))
                            except Exception:
                                pass
                return None
            futs.append((exp, float(price), {
                "oi": _f("openInterest", "opnInterest"),
                "oi_chg": _f("changeinOpenInterest", "changeInOpenInterest"),
                "volume": _f("tradedVolume", "totalTradedVolume", "vmap"),
                "prev_close": _f("previousClose", "prevClose", "closePrice"),
                "change_pct": _f("pChange", "percentChange"),
            }))
        if not futs:
            raise NSEFetchError("no futures rows in quote-derivative payload")
        futs.sort(key=lambda x: x[0])
        exp, price, extra = futs[0]
        dte = max(0.0, (exp - _dt.now()).total_seconds() / 86400.0)  # fractional
        # next-month, for the rollover read
        nxt = futs[1] if len(futs) > 1 else None
        extra = dict(extra or {})
        extra["far_price"] = nxt[1] if nxt else None
        extra["far_oi"] = (nxt[2] or {}).get("oi") if nxt else None
        result = (price, dte, extra)
        _futures_cache[sym] = (now, result)
        return result
    except Exception as e:  # noqa: BLE001 — strictly best-effort
        print(f"[!] Futures price fetch failed (non-fatal): {e}")
        return cached[1] if cached else (None, None, {})


def _update_session_ohlc(symbol: str, spot: float) -> None:
    today = time.strftime("%Y-%m-%d")
    sym = symbol.upper()
    entry = _session_ohlc.get(sym)
    if entry is None or entry.get("date") != today:
        _session_ohlc[sym] = {"date": today, "open": spot, "high": spot, "low": spot, "close": spot, "prev_close": None}
    else:
        entry["high"] = max(entry["high"], spot)
        entry["low"] = min(entry["low"], spot)
        entry["close"] = spot


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


def _enrich_strikes_with_oi_delta(symbol: str, strikes: list) -> list[dict]:
    """Serialise strikes and add 15-minute OI momentum fields.

    ce_oi_chg / pe_oi_chg come directly from NSE's changeinOpenInterest
    (change vs previous day's closing OI) and are already in StrikeData.
    We add:
      ce_oi_15m / pe_oi_15m = change in the last 15 minutes (periodic snapshots)
    """
    deltas_15m = _get_15m_oi_delta(symbol, strikes)
    _record_oi_snapshot(symbol, strikes)
    _record_oi_timeline(symbol, strikes)   # 3-min heatmap feed (no extra NSE calls)
    _record_ltp_history(symbol, strikes)   # 1-min LTP series for leg sparklines
    _record_replay_snapshot(symbol)        # marks that a snapshot is due (written post-build)
    result = []
    for s in strikes:
        d = asdict(s)
        ce_15m, pe_15m = deltas_15m.get(s.strike, (0, 0))
        d["ce_oi_15m"] = ce_15m
        d["pe_oi_15m"] = pe_15m
        result.append(d)
    return result


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
    # Feed spot price into the movers tracker + session OHLC for pivot calculations
    try:
        nse_fno_movers.record_spot_price(symbol, snap.underlying_value)
    except Exception:  # noqa: BLE001
        pass
    _update_session_ohlc(symbol, snap.underlying_value)
    _index_spot_cache[symbol] = snap.underlying_value   # for relative strength
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
                "dte": round(_dte_fractional(exp), 4),
            "dte_days": days_to_expiry(exp),
            "dte_trading": round(_trading_dte(_dte_fractional(exp)), 4),
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

    # Nearest-month futures — basis / cost-of-carry card (best-effort, cached 30s)
    futures_price, futures_dte, futures_extra = _fetch_futures_price(fetcher, symbol)

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
        "iv_rank": iv_rank["rank_pct"] if iv_rank else None,   # flat field for the IV Rank stat card
        "futures_price": futures_price,
        "futures_dte": futures_dte,
        # Futures OI is the cleanest positioning read available: paired with
        # price direction it gives the four-way classification on a SINGLE
        # instrument, rather than inferring it across sixty option strikes
        # where the flow is fragmented and partly hedging.
        "futures_oi": (futures_extra or {}).get("oi"),
        "futures_oi_chg": (futures_extra or {}).get("oi_chg"),
        "futures_volume": (futures_extra or {}).get("volume"),
        "futures_prev_close": (futures_extra or {}).get("prev_close"),
        "futures_far_price": (futures_extra or {}).get("far_price"),
        "futures_far_oi": (futures_extra or {}).get("far_oi"),
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
        "strikes": _enrich_strikes_with_oi_delta(symbol, snap.strikes),
        "expiries_data": expiries_data,
        "strategies": [asdict(s) for s in strategy_list],
        "india_vix": india_vix,
        "session_ohlc": _session_ohlc.get(symbol.upper(), {}),
        "symbol": symbol,
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


def _ohlc_from_history(symbol: str) -> dict | None:
    """
    Read previous day's OHLC from the locally-stored history files.
    This is the most reliable source — it's derived from actual spot prices
    the server observed throughout that session.

    Tries the last 4 calendar days so weekends/holidays are handled automatically.
    Requires at least 5 observations to be considered a valid session.
    """
    from datetime import date, timedelta
    from nse_history_store import HISTORY_DIR
    import json as _json

    for delta in range(1, 8):  # look back up to 7 calendar days
        target = date.today() - timedelta(days=delta)
        path = HISTORY_DIR / f"{symbol.upper()}_{target.isoformat()}.jsonl"
        if not path.exists():
            continue
        spots = []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = _json.loads(line)
                        s = r.get("spot") or r.get("close")
                        if s and float(s) > 0:
                            spots.append(float(s))
                    except Exception:
                        continue
        except Exception:
            continue
        if len(spots) < 5:
            continue   # too few observations — probably a partial/corrupt file
        lo, hi = min(spots), max(spots)
        if hi == lo:
            continue   # H=L=C: server was running on a holiday — no price movement
        return {
            "symbol": symbol,
            "open": round(spots[0], 2),
            "high": round(hi, 2),
            "low":  round(lo, 2),
            "close": round(spots[-1], 2),
            "source": "history_daily",
            "date": target.isoformat(),
        }
    return None


def _fetch_prev_day_ohlc(symbol: str, verbose: bool = True) -> dict:
    """
    Fetch the previous trading day's OHLC from NSE APIs (with CSV as the
    fastest/most-reliable first check). When verbose=True (default), prints
    a one-line status for every source tried — this is intentional: NSE's
    API field names are not officially documented and have shifted before,
    so visible logging is how we diagnose failures in production rather
    than guessing blindly.

    Source priority:
      0. CSV (nse_data_cache/*.csv)   — official EOD, no network needed
      1. historical/indicesHistory    — explicit date-range EOD OHLC
      2. chart-databyindex             — daily candle array
      3. historical/indicesHistory     — retry with longer timeout
      4. allIndices                    — live snapshot, light-weight
      5. equity-stockIndices           — index row from constituent list
      6. no-data                       — user enters manually
    """
    def _log(msg: str) -> None:
        if verbose:
            print(f"[ohlc:{symbol}] {msg}")

    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
    from datetime import date, timedelta

    global _shared_fetcher, _shared_fetcher_ts
    fetcher = (
        _shared_fetcher
        if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
            and (time.time() - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE)
        else NSESession()
    )
    # If the fetcher is fresh, warm it now so NSE API calls succeed.
    if not getattr(fetcher, "_warmed", False):
        try:
            fetcher._warm_up()
            _shared_fetcher = fetcher
            _shared_fetcher_ts = time.time()
        except Exception:  # noqa: BLE001
            pass
    h = dict(API_HEADERS)
    h["Referer"] = NSE_OC_PAGE

    index_map = {
        "NIFTY":      ("NIFTY%2050",           "NIFTY 50"),
        "BANKNIFTY":  ("NIFTY%20BANK",         "NIFTY BANK"),
        "FINNIFTY":   ("NIFTY%20FIN%20SERVICE", "NIFTY FIN SERVICE"),
        "MIDCPNIFTY": ("NIFTY%20MID%20SELECT",  "NIFTY MID SELECT"),
        # SENSEX intentionally excluded — it is a BSE index, not NSE.
        # None of NSE's indicesHistory / chart-databyindex / allIndices /
        # equity-stockIndices APIs carry SENSEX data, so every lookup for it
        # was guaranteed to exhaust all 5 sources and land on "no prev-day
        # data". The four indices above are NSE's only F&O indices anyway
        # (matches INDEX_SYMBOLS in nse_options_strategy.py).
    }

    def _v(d, *keys):
        for k in keys:
            val = d.get(k)
            if val:
                try:
                    return float(str(val).replace(",", "").strip())
                except ValueError:
                    continue
        return 0.0

    def _ohlc(O, H, L, C, src):
        return {"symbol": symbol,
                "open": round(float(O), 2), "high": round(float(H), 2),
                "low":  round(float(L), 2), "close": round(float(C), 2),
                "source": src}

    # ── 0. CSV (official NSE EOD data, no network required) ─────────────
    # IMPORTANT: use EXACT filename match only — never the glob fallback in
    # _find_csv, which can return unrelated files (e.g. NIFTYBEES.csv when
    # searching for NIFTY, returning an ETF price of ~₹35 instead of ~24,000).
    try:
        from nse_pivot_scanner import DATA_DIR, _read_rows, _last_trading_row
        from datetime import date as _dt_date
        sym_up = symbol.upper()
        csv_path = None
        for _name in (sym_up, f"{sym_up}.NS", symbol, f"{symbol}.NS"):
            _p = DATA_DIR / f"{_name}.csv"
            if _p.exists():
                csv_path = _p
                break
        if not csv_path:
            _log(f"src0 CSV: no file found in {DATA_DIR} for {sym_up}")
        else:
            _rows, _err = _read_rows(csv_path, n=10)
            if _err or not _rows:
                _log(f"src0 CSV: {csv_path.name} read error: {_err}")
            else:
                _today = _dt_date.today().isoformat()
                _past  = [r for r in _rows if r["date"] < _today]
                _row   = _last_trading_row(_past) if _past else None
                if not _row:
                    _log(f"src0 CSV: {csv_path.name} has no rows before {_today}")
                else:
                    _h, _l, _c = _row.get("H", 0), _row.get("L", 0), _row.get("C", 0)
                    _min_val = 500 if symbol in index_map else 10
                    if _h > _l > 0 and _c > 0 and _l >= _min_val:
                        _log(f"src0 CSV: OK  H={_h} L={_l} C={_c} date={_row['date']}")
                        return _ohlc(_row.get("O", _c), _h, _l, _c, "csv_eod")
                    _log(f"src0 CSV: rejected H={_h} L={_l} C={_c} (min_val={_min_val})")
    except Exception as e:
        _log(f"src0 CSV: exception {e}")

    # ── 0.5 NSE bhavcopy archive ──────────────────────────────────────────────
    # Static archive CSV — no cookies needed, now cached to disk so the file
    # is only downloaded once per calendar day, not on every /api/ohlc call.
    if symbol in index_map:
        _, idx_name = index_map[symbol]
        name_aliases = {
            "NIFTY":      ["NIFTY 50", "Nifty 50"],
            "BANKNIFTY":  ["NIFTY BANK", "Nifty Bank"],
            "FINNIFTY":   ["NIFTY FIN SERVICE", "Nifty Fin Service",
                           "NIFTY FINANCIAL SERVICES", "Nifty Financial Services"],
            "MIDCPNIFTY": ["NIFTY MID SELECT", "Nifty Mid Select",
                           "NIFTY MIDCAP SELECT", "Nifty Midcap Select"],
        }.get(symbol, [idx_name])
        for _delta in range(1, 11):
            _target = date.today() - timedelta(days=_delta)
            try:
                csv_text = _bhavcopy_for_date(_target, fetcher.session)
                if not csv_text:
                    continue
                row = _parse_bhavcopy_row(csv_text, name_aliases)
                if not row:
                    _log(f"src0.5 bhavcopy {_target}: file OK but '{idx_name}' not found")
                    continue
                Ob, Hb, Lb, Cb = row["O"], row["H"], row["L"], row["C"]
                _log(f"src0.5 bhavcopy {_target}: OK  H={Hb} L={Lb} C={Cb}")
                return _ohlc(Ob, Hb, Lb, Cb, "bhavcopy_eod")
            except Exception as e:
                _log(f"src0.5 bhavcopy {_target}: exception {e}")
        _log(f"src0.5 bhavcopy: no usable file found in last 10 days for {idx_name}")

    # ── 1. historical/indicesHistory ────────────────────────────────────
    # Query date_to = yesterday so we NEVER include today's partial intraday data.
    # Use a 10-day window so extended holiday periods are always covered.
    # Retries on 503 since NSE's WAF often returns it transiently.
    if symbol in index_map:
        _, idx_name = index_map[symbol]
        idx_enc = idx_name.replace(" ", "%20")
        today     = date.today()
        date_to   = (today - timedelta(days=1)).strftime("%d-%m-%Y")   # yesterday
        date_from = (today - timedelta(days=10)).strftime("%d-%m-%Y")  # 10-day window
        url = (f"https://www.nseindia.com/api/historical/indicesHistory"
               f"?indexType={idx_enc}&from={date_from}&to={date_to}")
        r = None
        for _attempt in range(3):
            try:
                r = fetcher.session.get(url, headers=h, timeout=12)
                if r.status_code == 503 and _attempt < 2:
                    _log(f"src1 indicesHistory: HTTP 503 (attempt {_attempt+1}/3), retrying...")
                    time.sleep(1.5 * (_attempt + 1))
                    continue
                break
            except Exception as e:
                _log(f"src1 indicesHistory: request exception (attempt {_attempt+1}/3): {e}")
                time.sleep(1.0)
        try:
            if r is None or r.status_code != 200:
                _log(f"src1 indicesHistory: HTTP {r.status_code if r else 'no-response'} "
                     f"after retries — {r.text[:150] if r else ''}")
            else:
                all_rows = (r.json().get("data") or {}).get("indexCloseOnlineRecords", [])
                if not all_rows:
                    _log(f"src1 indicesHistory: 200 OK but 0 rows — response keys: {list(r.json().keys())}")
                trading_rows = [
                    row for row in all_rows
                    if _v(row, "EOD_HIGH_INDEX_VAL", "HIGH_INDEX_VAL", "HIGH_INDEX_VALUE") >
                       _v(row, "EOD_LOW_INDEX_VAL",  "LOW_INDEX_VAL",  "LOW_INDEX_VALUE")
                ]
                rows = trading_rows if trading_rows else all_rows
                if rows:
                    row = rows[-1]   # last actual completed trading day
                    O = _v(row, "EOD_OPEN_INDEX_VAL",  "OPEN_INDEX_VAL",  "OPEN_INDEX_VALUE")
                    H = _v(row, "EOD_HIGH_INDEX_VAL",  "HIGH_INDEX_VAL",  "HIGH_INDEX_VALUE")
                    L = _v(row, "EOD_LOW_INDEX_VAL",   "LOW_INDEX_VAL",   "LOW_INDEX_VALUE")
                    C = _v(row, "EOD_CLOSE_INDEX_VAL", "CLOSE_INDEX_VAL", "CLOSING_INDEX_VAL")
                    if H > L > 0 and C > 0:  # strict: reject flat/degenerate candles

                        _log(f"src1 indicesHistory: OK  H={H} L={L} C={C}")
                        return _ohlc(O or C, H, L, C, "daily_history")
                    _log(f"src1 indicesHistory: row found but H/L/C invalid: H={H} L={L} C={C} raw={row}")
        except Exception as e:  # noqa: BLE001
            _log(f"src1 indicesHistory: exception {e}")

    # ── 2. chart-databyindex ─────────────────────────────────────────────
    if symbol in index_map:
        idx_code, _ = index_map[symbol]
        try:
            url2 = f"https://www.nseindia.com/api/chart-databyindex?index={idx_code}&indices=true"
            r = fetcher.session.get(url2, headers=h, timeout=12)
            if r.status_code != 200:
                _log(f"src2 chart-databyindex: HTTP {r.status_code} — {r.text[:150]}")
            else:
                payload = r.json() if isinstance(r.json(), dict) else {}
                found = False
                for key in ("grapthData", "graphData"):
                    candles = payload.get(key, [])
                    if not candles:
                        continue
                    found = True
                    last_ts_ms = candles[-1][0] if isinstance(candles[-1], (list, tuple)) else 0
                    import datetime as _dt
                    last_date = _dt.datetime.fromtimestamp(last_ts_ms / 1000).date() if last_ts_ms else None
                    use_second_last = (last_date == date.today()) and len(candles) >= 2
                    candidates = [candles[-2]] if use_second_last else []
                    for candle in reversed(candles[:-1] if use_second_last else candles):
                        if isinstance(candle, (list, tuple)) and len(candle) >= 5:
                            cH, cL = float(candle[2]), float(candle[3])
                            if cH > cL:
                                candidates.insert(0, candle)
                                break
                    prev = candidates[0] if candidates else (candles[-2] if len(candles) >= 2 else candles[-1])
                    if isinstance(prev, (list, tuple)) and len(prev) >= 5:
                        O, H, L, C = float(prev[1]), float(prev[2]), float(prev[3]), float(prev[4])
                        if H > L > 0 and C > 0:
                            _log(f"src2 chart-databyindex[{key}]: OK  H={H} L={L} C={C}")
                            return _ohlc(O, H, L, C, "daily_chart")
                        _log(f"src2 chart-databyindex[{key}]: candle invalid H={H} L={L} C={C}")
                closes = payload.get("closePrice") or []
                highs  = payload.get("dayHigh")    or []
                lows   = payload.get("dayLow")     or []
                opens  = payload.get("openPrice")  or []
                if closes and highs and lows and len(closes) >= 2:
                    found = True
                    lv = lambda arr: float(arr[-2][1]) if len(arr) >= 2 else float(arr[-1][1])
                    H2, L2, C2 = lv(highs), lv(lows), lv(closes)
                    O2 = lv(opens) if opens else C2
                    if H2 > L2 > 0 and C2 > 0:
                        _log(f"src2 chart-databyindex[arrays]: OK  H={H2} L={L2} C={C2}")
                        return _ohlc(O2, H2, L2, C2, "daily_chart")
                    _log(f"src2 chart-databyindex[arrays]: invalid H={H2} L={L2} C={C2}")
                if not found:
                    _log(f"src2 chart-databyindex: 200 OK but no usable keys — response keys: {list(payload.keys())}")
        except Exception as e:  # noqa: BLE001
            _log(f"src2 chart-databyindex: exception {e}")

    # ── 3. equitiesHistory (index fallback) ─────────────────────────────
    # Extra fallback for NIFTY / BANKNIFTY etc. if the two index-specific
    # APIs above both failed (network blip, NSE rate-limit, etc.).
    # We never fall back to .jsonl recordings for these symbols (see below).
    if symbol in index_map:
        try:
            today     = date.today()
            date_to   = (today - timedelta(days=1)).strftime("%d-%m-%Y")
            date_from = (today - timedelta(days=10)).strftime("%d-%m-%Y")
            _, idx_name = index_map[symbol]
            idx_enc = idx_name.replace(" ", "%20")
            # Try the generic NSE historical API as a last-resort for index symbols
            url = (f"https://www.nseindia.com/api/historical/indicesHistory"
                   f"?indexType={idx_enc}&from={date_from}&to={date_to}")
            r2 = fetcher.session.get(url, headers=h, timeout=15)
            if r2.status_code == 200:
                all_rows = (r2.json().get("data") or {}).get("indexCloseOnlineRecords", [])
                trading = [
                    row for row in all_rows
                    if _v(row, "EOD_HIGH_INDEX_VAL", "HIGH_INDEX_VAL") >
                       _v(row, "EOD_LOW_INDEX_VAL",  "LOW_INDEX_VAL")
                ]
                rows = trading if trading else all_rows
                if rows:
                    row = rows[-1]
                    O3 = _v(row, "EOD_OPEN_INDEX_VAL",  "OPEN_INDEX_VAL")
                    H3 = _v(row, "EOD_HIGH_INDEX_VAL",  "HIGH_INDEX_VAL")
                    L3 = _v(row, "EOD_LOW_INDEX_VAL",   "LOW_INDEX_VAL")
                    C3 = _v(row, "EOD_CLOSE_INDEX_VAL", "CLOSE_INDEX_VAL", "CLOSING_INDEX_VAL")
                    if H3 > L3 > 0 and C3 > 0:  # strict: reject flat/degenerate candles
                        return _ohlc(O3 or C3, H3, L3, C3, "daily_history")
        except Exception:
            pass

    # ── 4. allIndices — REMOVED ───────────────────────────────────────────
    # Originally tried as a fallback, but allIndices' open/high/low/last
    # fields are the CURRENT session's live, still-developing values — not
    # the previous day's settled OHLC. While markets are open, "today's high
    # so far" keeps changing every few minutes, which made pivot levels look
    # subtly wrong/inconsistent throughout the day even though each fetch
    # "succeeded". Removed entirely: this source can never correctly answer
    # "what was yesterday's OHLC", only "what is today's OHLC so far".

    # ── 5. equity-stockIndices — REMOVED ─────────────────────────────────
    # Same flaw as source 4: dayHigh/dayLow/lastPrice are today's live
    # intraday values, not yesterday's settled close. Removed for the same
    # reason — using "today's range so far" as "previous day" data is
    # structurally wrong regardless of which NSE endpoint it comes from.

    # ── No NSE data yet — try history store as last resort ──────────────
    # .jsonl history writes are disabled (nse_history_store.append_snapshot
    # is a no-op). Previous-day OHLC is always sourced from NSE APIs or CSV.
    # _ohlc_from_history is therefore never called here.

    # ── Absolute fallback: prompt the user ───────────────────────────────
    _log("ALL 5 sources exhausted — falling back to manual entry. "
         "Run GET /api/ohlc-debug?symbol=" + symbol + " to see raw NSE responses.")
    se = _session_ohlc.get(symbol, {})
    return {
        "symbol": symbol,
        "open":  round(se.get("open")  or 0, 2),
        "high":  round(se.get("high")  or 0, 2),
        "low":   round(se.get("low")   or 0, 2),
        "close": round(se.get("close") or 0, 2),
        "source": "no_data",
        "warn": (
            "NSE history endpoints did not return OHLC data yet. "
            "Enter H / L / C manually in the pivot panel."
        ),
    }



_nifty_cache: dict[str, dict] = {}   # keyed by index symbol
_NIFTY_CACHE_TTL = 60

# NSE index names + fallback weights + allIndices name for spot price
# Weights are approximate free-float cap weights (updated quarterly; these reflect 2025 Q1-Q2)
_INDEX_CONFIG: dict[str, dict] = {
    "NIFTY": {
        "nse_names": ["NIFTY 50", "Nifty 50"],
        "all_indices_name": "NIFTY 50",
        "title": "Nifty 50",
        # Approximate free-float market-cap weights (%). Normalised at
        # calculation time, so the exact sum does not need to be 100.
        # NSE rebalances quarterly (Mar/Jun/Sep/Dec).
        "fallback_weights": [
            ("HDFCBANK", 11.5), ("RELIANCE", 9.2),  ("ICICIBANK", 8.5),
            ("INFY",      5.8), ("TCS",      4.3),  ("BHARTIARTL",4.0),
            ("ITC",       3.7), ("LT",       3.4),  ("KOTAKBANK", 3.2), ("AXISBANK", 3.0),
            ("SBIN",      2.9), ("BAJFINANCE",2.1), ("HINDUNILVR",2.0),
            ("ASIANPAINT",1.7), ("MARUTI",   1.7),  ("M&M",       1.6),
            ("SUNPHARMA", 1.6), ("TATAMOTORS",1.5), ("NTPC",      1.4),
            ("TITAN",     1.4), ("ULTRACEMCO",1.3), ("ONGC",      1.2),
            ("ADANIENT",  1.2), ("WIPRO",    1.1),  ("POWERGRID", 1.1),
            ("BAJAJFINSV",1.0), ("NESTLEIND",1.0),  ("COALINDIA", 1.0),
            ("JSWSTEEL",  0.95),("TATASTEEL",0.95), ("HCLTECH",   0.95),
            ("INDUSINDBK",0.9), ("GRASIM",   0.85), ("ADANIPORTS",0.85),
            ("TECHM",     0.85),("CIPLA",    0.85), ("DRREDDY",   0.75),
            ("EICHERMOT", 0.75),("BRITANNIA",0.75), ("APOLLOHOSP",0.75),
            ("DIVISLAB",  0.65),("HEROMOTOCO",0.65),("BAJAJ-AUTO",0.65),
            ("SBILIFE",   0.65),("HDFCLIFE", 0.65), ("SHRIRAMFIN",0.65),
            ("TATACONSUM",0.6), ("LTIM",     0.55), ("UPL",       0.5), ("BPCL",0.5),
        ],
    },
    "BANKNIFTY": {
        "nse_names": ["NIFTY BANK", "Nifty Bank"],
        "all_indices_name": "NIFTY BANK",
        "title": "Bank Nifty",
        "fallback_weights": [
            ("HDFCBANK", 28.8), ("ICICIBANK", 23.4), ("KOTAKBANK", 12.3),
            ("AXISBANK",  9.9), ("SBIN",       7.8), ("INDUSINDBK", 5.0),
            ("BANDHANBNK",3.3), ("FEDERALBNK", 2.7), ("IDFCFIRSTB", 2.4),
            ("PNB",       2.2), ("AUBANK",     1.5), ("CUB",        0.7),
        ],
    },
    "FINNIFTY": {
        "nse_names": ["NIFTY FIN SERVICE", "Nifty Financial Services"],
        "all_indices_name": "NIFTY FIN SERVICE",
        "title": "Fin Nifty",
        "fallback_weights": [
            ("HDFCBANK", 18.2), ("ICICIBANK", 16.5), ("KOTAKBANK",  8.3),
            ("AXISBANK",  7.0), ("SBIN",       6.0), ("BAJFINANCE", 5.6),
            ("BAJAJFINSV",4.9), ("HDFCLIFE",   4.1), ("SBILIFE",    3.8),
            ("ICICIPRULI",3.0), ("ICICIGI",    2.8), ("SHRIRAMFIN", 2.6),
            ("CHOLAFIN",  2.3), ("PFC",        2.0), ("RECLTD",     1.9),
            ("MUTHOOTFIN",1.7), ("LICHSGFIN",  1.4), ("PNBHOUSING", 1.2),
            ("INDUSINDBK",1.2), ("IDFCFIRSTB", 1.0),
        ],
    },
    "MIDCPNIFTY": {
        "nse_names": ["NIFTY MID SELECT", "Nifty Midcap Select"],
        "all_indices_name": "NIFTY MID SELECT",
        "title": "MidCap Select",
        "fallback_weights": [
            ("PERSISTENT", 6.5), ("ZOMATO",    6.0), ("POLYCAB",   5.2),
            ("JSWENERGY",  4.9), ("CANBK",     4.5), ("BHEL",      4.2),
            ("LICHSGFIN",  3.9), ("ABCAPITAL", 3.7), ("MRF",       3.5),
            ("MFSL",       3.1), ("INDHOTEL",  3.0), ("COFORGE",   2.9),
            ("AUROPHARMA", 2.7), ("PAGEIND",   2.5), ("BHARATFORG",2.4),
            ("GODREJPROP", 2.3), ("SUPREMEIND",2.2), ("INDUSTOWER",2.1),
            ("TATACOMM",   2.0), ("VOLTAS",    1.9), ("BALKRISIND",1.8),
            ("FEDERALBNK", 1.7), ("MAXHEALTH", 1.6), ("PIIND",     1.5),
            ("OBEROIRLTY", 1.4),
        ],
    },
}


def _fetch_prev_week_ohlc(symbol: str) -> dict | None:
    """Return the previous COMPLETE trading week's O/H/L/C from bhavcopy cache.

    'Previous complete week' = Mon–Fri that ended before today.
    O = Monday's open, H = week's highest H, L = week's lowest L, C = Friday's close.
    Uses cached bhavcopy files — no new network calls if the week is already cached.
    """
    from datetime import date, timedelta

    index_map_local = {
        "NIFTY":      ("NIFTY 50",          ["NIFTY 50", "Nifty 50"]),
        "BANKNIFTY":  ("NIFTY BANK",        ["NIFTY BANK", "Nifty Bank"]),
        "FINNIFTY":   ("NIFTY FIN SERVICE", ["NIFTY FIN SERVICE", "Nifty Fin Service",
                                              "NIFTY FINANCIAL SERVICES"]),
        "MIDCPNIFTY": ("NIFTY MID SELECT",  ["NIFTY MID SELECT", "Nifty Mid Select",
                                              "NIFTY MIDCAP SELECT"]),
    }
    if symbol not in index_map_local:
        return None
    _, aliases = index_map_local[symbol]

    global _shared_fetcher
    fetcher = _shared_fetcher
    if not fetcher:
        return None
    sess = fetcher.session

    today = date.today()
    # Walk back to find the most recent Friday that is at least 1 day before today
    friday = today - timedelta(days=1)
    while friday.weekday() != 4:   # 4 = Friday
        friday -= timedelta(days=1)
    monday = friday - timedelta(days=4)

    week_rows: list[dict] = []
    for d in (monday + timedelta(days=i) for i in range(5)):
        if d.weekday() >= 5:   # skip Saturday/Sunday
            continue
        csv_text = _bhavcopy_for_date(d, sess)
        if not csv_text:
            continue
        row = _parse_bhavcopy_row(csv_text, aliases)
        if row:
            week_rows.append({**row, "date": d})

    if not week_rows:
        return None

    O = week_rows[0]["O"]
    H = max(r["H"] for r in week_rows)
    L = min(r["L"] for r in week_rows)
    C = week_rows[-1]["C"]
    week_start = week_rows[0]["date"].strftime("%d %b")
    week_end   = week_rows[-1]["date"].strftime("%d %b")
    return {
        "symbol": symbol, "open": round(O, 2), "high": round(H, 2),
        "low": round(L, 2), "close": round(C, 2),
        "source": "bhavcopy_week",
        "date": f"{week_start}–{week_end}",
        "days": len(week_rows),
    }


def _build_constituents_result(sym: str, raw_stocks: list, source: str = "nse",
                               data_quality: str = "live") -> dict:
    """Shared result builder.

    Extracted so the broker path and the NSE paths produce IDENTICAL
    output - same weights, same normalisation, same fields. Two
    separate builders would drift, and the drift would show up as
    contribution figures that quietly disagree depending on which
    source happened to answer.
    """
    cfg = _INDEX_CONFIG.get(sym.upper()) or _INDEX_CONFIG["NIFTY"]
    # Index level: prefer a live spot, then the session cache, then derive it
    # from the constituents themselves. Needed because point contributions are
    # scaled by it, and a zero here would silently zero every figure.
    index_value = 0.0
    try:
        se = _session_ohlc.get(sym.upper()) or {}
        index_value = float(se.get("close") or se.get("last") or 0)
    except Exception:
        pass
    if not index_value:
        try:
            ld = globals().get("_last_chain") or {}
            if isinstance(ld, dict) and ld.get("symbol", "").upper() == sym.upper():
                index_value = float(ld.get("underlying_value") or 0)
        except Exception:
            pass
    if not index_value:
        index_value = {"NIFTY": 24500.0, "BANKNIFTY": 52000.0,
                       "FINNIFTY": 23500.0, "MIDCPNIFTY": 12500.0}.get(sym.upper(), 24500.0)
    # ── Build result ─────────────────────────────────────────────────
    # Prefer live weights from index_weights.json; the hardcoded table is a
    # last resort, and the response says which was used so a stale figure is
    # visible rather than assumed correct.
    _wf = _load_index_weights().get(sym.upper()) or {}
    _weight_map = dict(_wf.get("weights") or {}) or dict(cfg["fallback_weights"])
    _weight_src = ("file:" + (_wf.get("method") or "?")) if _wf.get("weights") else "hardcoded fallback"
    stocks = [
        s for s in raw_stocks
        if s.get("symbol") not in (cfg["nse_names"] + [sym, "NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY MID SELECT"])
        and float(s.get("lastPrice") or 0) > 0
    ]
    stocks.sort(key=lambda s: float(s.get("totalTradedValue") or 0), reverse=True)

    total_tv = sum(float(s.get("totalTradedValue") or 0) for s in stocks) or 1.0
    # Prefer the WEIGHT TABLE whenever we have one. Traded value is only a
    # proxy for weight and a poor one intraday - a heavily traded mid-cap can
    # out-turnover a larger constituent for a session without being anywhere
    # near it in index weight. The table (recovered from point contributions)
    # is the real figure, so it wins; traded value is the fallback for when
    # there is no table, not the other way round.
    use_fixed = bool(_weight_map) or total_tv < 100

    # Pre-compute raw weights, then NORMALISE to sum to exactly 100%.
    # This is the critical fix: hardcoded fallback_weights are approximate
    # and almost never sum to exactly 100 (e.g. NIFTY hardcoded weights
    # summed to 105.1%, inflating every pts_contributed by 5%). Normalising
    # ensures the sum of all pts_contributed ≈ actual index point change.
    raw_weights: dict[str, float] = {}
    for s in stocks:
        stock_sym = s.get("symbol", "")
        if use_fixed:
            raw_weights[stock_sym] = _weight_map.get(stock_sym, 0.5)
        else:
            raw_weights[stock_sym] = float(s.get("totalTradedValue") or 0) / total_tv * 100

    weight_total = sum(raw_weights.values()) or 1.0
    norm_weights = {k: v / weight_total * 100 for k, v in raw_weights.items()}

    result = []
    for rank, s in enumerate(stocks, 1):
        stock_sym = s.get("symbol", "")
        wt  = norm_weights.get(stock_sym, 0.0)
        pct = float(s.get("pChange") or 0)
        pts = round(pct / 100 * wt / 100 * index_value, 2)
        result.append({
            "rank": rank,
            "symbol": stock_sym,
            "ltp":        round(float(s.get("lastPrice") or 0), 2),
            "prev_close": round(float(s.get("previousClose") or 0), 2),
            "change":     round(float(s.get("change") or 0), 2),
            "pct_change": round(pct, 2),
            "volume":     int(float(s.get("totalTradedVolume") or 0)),
            "weight_est": round(wt, 2),
            "pts_contributed": pts,
            "pts_abs": abs(pts),
        })

    data = {
        "stocks": result,
        "index_value": index_value,
        "nifty_value": index_value,   # keep compat field
        "index_symbol": sym,
        "source": source,
        "weight_source": _weight_src,
        "weight_updated": (_wf.get("fetched") if _wf else None),
        "index_title": cfg["title"],
        "as_of": time.strftime("%H:%M:%S"),
        "data_quality": data_quality,   # "live" | "pre_open" | "fallback_quote"
        "note": (
            f"Weight from {'hardcoded approx' if use_fixed else 'session traded value'}. "
            f"Points contributed ≈ pChange × weight × {cfg['title']} value."
            + (" ⚠️ PRE-OPEN DATA — prices frozen at day's open, not live."
               if data_quality == "pre_open" else "")
        ),
    }
    _nifty_cache[sym] = {"ts": time.time(), "data": data}
    return data


_stock_symbols_cache: dict = {"ts": 0, "symbols": []}


def _fetch_nifty_constituents(symbol: str = "NIFTY") -> dict:
    """
    Fetch constituents for NIFTY, BANKNIFTY, FINNIFTY, or MIDCPNIFTY.

    Three strategies tried in order:
      1. equity-stockIndices bulk fetch — currently 404s on NSE's site
         (endpoint appears to have been moved/retired), kept in case it's
         restored or inconsistent by region.
      2. Per-stock option-chain fetch (live LTP) for the top ~10-12 weighted
         constituents — reuses get_option_chain(), the proven endpoint
         already used for NIFTY/BANKNIFTY chains all session. previousClose
         comes from the local CSV cache, no extra network call.
      3. market-data-pre-open — TRUE last resort. This is frozen auction
         data from 9:00-9:15 IST that never updates again that session.
         (Note: /api/quote-equity, originally strategy 2, was removed
         entirely — it sits behind a hard Akamai "Access Denied" 403 wall
         regardless of headers/cookies/referer, confirmed across every
         symbol and every attempt.)

    Results are cached per index symbol for 60 s.
    """
    # ── strategy 0: the active BROKER feed ────────────────────────────
    # Tried first, before any NSE scraping, when a broker source is selected.
    # The constituent list comes from index_weights.json and the broker quotes
    # every name in it, so this works when NSE is unreachable, unwarmed or
    # rate-limiting - which is exactly when the NSE strategies below fail and
    # the widget used to show "Server not reachable" despite the server being
    # perfectly healthy and the data sitting in the feed.
    if RUN_DATA_SOURCE == "fyers":
        try:
            import nse_adapter_fyers as _fy
            if _fy.is_configured():
                _cfg0 = _INDEX_CONFIG.get(symbol.upper()) if "_INDEX_CONFIG" in globals() else None
                _names = list((_load_index_weights().get(symbol.upper()) or {}).get("weights") or {})
                if not _names and _cfg0:
                    _names = [n for n, _w in _cfg0.get("fallback_weights", [])]
                if _names:
                    _q = _fy.constituent_quotes(_names)
                    if len(_q) >= max(3, len(_names) // 3):
                        print(f"[constituents] {symbol}: {len(_q)} names from the Fyers feed")
                        return _build_constituents_result(symbol, list(_q.values()),
                                                          source="fyers",
                                                          data_quality="live")
                    if _q:
                        print(f"[constituents] fyers returned only {len(_q)}/{len(_names)} "
                              f"- falling through to NSE")
        except Exception as e:  # noqa: BLE001
            print(f"[constituents] fyers path failed ({e}) - falling through to NSE")

    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE, NSE_ALL_INDICES_API

    sym = symbol.upper()
    cfg = _INDEX_CONFIG.get(sym, _INDEX_CONFIG["NIFTY"])

    now = time.time()
    cached = _nifty_cache.get(sym)
    if cached and (now - cached.get("ts", 0)) < _NIFTY_CACHE_TTL:
        return cached["data"]

    global _shared_fetcher, _shared_fetcher_ts
    fetcher = (
        _shared_fetcher
        if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
            and (now - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE)
        else NSESession()
    )
    if not getattr(fetcher, "_warmed", False):
        fetcher._warm_up()
        _shared_fetcher = fetcher
        _shared_fetcher_ts = time.time()

    # Note: a separate "extra equity-page warm-up" used to live here, added
    # specifically to support /api/quote-equity. That endpoint is hard-
    # blocked by NSE's Akamai edge (confirmed: HTTP 403 "Access Denied" for
    # every symbol regardless of headers/cookies/referer) and has been
    # removed entirely from this codebase — so the extra warm-up call is
    # gone too, since it served no remaining purpose.

    h = dict(API_HEADERS)
    h["Referer"] = NSE_OC_PAGE   # used for allIndices (same family as /api/chain — works)

    # equity-stockIndices and market-data-pre-open are normally accessed
    # from NSE's live-market-data pages, NOT the option-chain page. NSE's
    # WAF validates Referer per endpoint family — using the option-chain
    # referer here is a likely cause of Strategy 1 silently degrading/
    # failing every time, forcing a permanent fallback to frozen pre-open
    # data (which never updates after the 9:00–9:15 auction).
    h_market = dict(API_HEADERS)
    h_market["Referer"] = "https://www.nseindia.com/market-data/live-equity-market"

    # Get the index spot value from allIndices
    index_value = 24000.0
    try:
        r = fetcher.session.get(NSE_ALL_INDICES_API, headers=h, timeout=10)
        if r.status_code == 200:
            for row in r.json().get("data", []):
                if row.get("index") == cfg["all_indices_name"]:
                    index_value = float(row.get("last") or row.get("lastPrice") or 24000)
                    break
    except Exception:  # noqa: BLE001
        pass

    # ── Strategy 1: equity-stockIndices with params= ─────────────────
    raw_stocks: list[dict] = []
    data_quality = "live"   # "live" | "pre_open" | "fallback_quote"
    for idx_name in cfg["nse_names"]:
        try:
            r = fetcher.session.get(
                "https://www.nseindia.com/api/equity-stockIndices",
                params={"index": idx_name},
                headers=h_market, timeout=12,
            )
            if r.status_code == 200:
                raw_stocks = r.json().get("data", [])
                if raw_stocks:
                    print(f"[constituents:{sym}] strategy1 equity-stockIndices OK "
                          f"({len(raw_stocks)} rows, idx_name={idx_name!r})")
                    break
                else:
                    print(f"[constituents:{sym}] strategy1 equity-stockIndices "
                          f"200 OK but 0 rows for idx_name={idx_name!r} — "
                          f"response body: {r.text[:200]}")
            else:
                print(f"[constituents:{sym}] strategy1 equity-stockIndices "
                      f"HTTP {r.status_code} for idx_name={idx_name!r} — "
                      f"body: {r.text[:200]}")
        except Exception as e:  # noqa: BLE001
            print(f"[constituents:{sym}] strategy1 exception for {idx_name!r}: {e}")
            continue

    # ── Strategy 2 (was 3): per-stock option chain for live LTP ──────
    # quote-equity sits behind a stricter Akamai bot-protection layer than
    # the rest of NSE's site — even with correct cookies/referer it returns
    # a hard "Access Denied" 403 for every symbol, every time. That's not
    # fixable from a script; NSE's edge is blocking the endpoint itself.
    #
    # Instead, reuse get_option_chain() — the exact function that has
    # successfully fetched NIFTY/BANKNIFTY chains all session. It works
    # identically for individual F&O stocks (confirmed in its own
    # docstring) and returns underlyingValue = live LTP. previousClose
    # comes from the local CSV cache (nse_data_cache/*.csv) that already
    # powers all 41 scanners — no extra network call, already proven
    # reliable.
    if not raw_stocks:
        print(f"[constituents:{sym}] strategy1 FAILED for all idx_names — "
              f"trying strategy2 per-stock option-chain fetch (live LTP) "
              f"for {len(cfg['fallback_weights'])} stocks, parallelized")
        from nse_pivot_scanner import get_daily_ohlc as _csv_ohlc

        def _fetch_one(sym_code: str) -> dict | None:
            try:
                chain = fetcher.get_option_chain(sym_code)
                last_price = float(chain.get("records", {}).get("underlyingValue") or 0)
                if last_price <= 0:
                    print(f"[constituents:{sym}] strategy2 option-chain for "
                          f"{sym_code}: no underlyingValue in response")
                    return None
                prev_close = None
                try:
                    row, err = _csv_ohlc(sym_code)
                    if row and not err:
                        prev_close = row.get("C")
                except Exception:
                    pass
                change  = (last_price - prev_close) if prev_close else 0
                pchange = (change / prev_close * 100) if prev_close else 0
                return {
                    "symbol":        sym_code,
                    "lastPrice":     last_price,
                    "previousClose": prev_close or last_price,
                    "change":        round(change, 2),
                    "pChange":       round(pchange, 2),
                    "totalTradedValue":  0,
                    "totalTradedVolume": 0,
                }
            except Exception as e:  # noqa: BLE001
                print(f"[constituents:{sym}] strategy2 option-chain exception "
                      f"for {sym_code}: {e}")
                return None

        # 8 concurrent workers balances speed (50 stocks would take 30-60s+
        # sequentially) against not overwhelming NSE's rate limiting. The
        # shared requests.Session's connection pool is thread-safe for
        # concurrent reads — this is a standard, well-established pattern.
        symbols = [s for s, _ in cfg["fallback_weights"]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_fetch_one, symbols))
        raw_stocks = [r for r in results if r is not None]

        if raw_stocks:
            data_quality = "fallback_quote"
            print(f"[constituents:{sym}] strategy2 option-chain fetch OK "
                  f"({len(raw_stocks)}/{len(cfg['fallback_weights'])} rows, LIVE LTP)")

    # ── Strategy 3 (was 2): market-data-pre-open — TRUE LAST RESORT ──
    # WARNING: this is PRE-MARKET AUCTION data (9:00–9:15 IST), captured
    # once at the opening auction and NEVER updated again that session.
    # Only used if both the bulk live endpoint AND individual live quotes
    # failed — using frozen data is strictly worse than no data with a
    # clear warning, which is why this now runs last instead of first.
    if not raw_stocks:
        print(f"[constituents:{sym}] strategy2 also failed — "
              f"falling back to strategy3 PRE-OPEN data (prices will be frozen at day's open!)")
        pre_open_key = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY"}.get(sym)
        if pre_open_key:
            try:
                r = fetcher.session.get(
                    "https://www.nseindia.com/api/market-data-pre-open",
                    params={"key": pre_open_key},
                    headers=h_market, timeout=12,
                )
                if r.status_code == 200:
                    for item in r.json().get("data", []):
                        meta = item.get("metadata") or {}
                        if meta.get("symbol") and meta.get("lastPrice"):
                            raw_stocks.append({
                                "symbol":        meta.get("symbol"),
                                "lastPrice":     meta.get("lastPrice"),
                                "previousClose": meta.get("previousClose"),
                                "change":        meta.get("change"),
                                "pChange":       meta.get("pChange"),
                                "totalTradedValue":  meta.get("totalTradedValue") or 0,
                                "totalTradedVolume": meta.get("totalTradedVolume") or 0,
                            })
                    if raw_stocks:
                        data_quality = "pre_open"
                        print(f"[constituents:{sym}] strategy3 PRE-OPEN data used "
                              f"({len(raw_stocks)} rows) — prices are FROZEN at day's open")
                else:
                    print(f"[constituents:{sym}] strategy3 pre-open HTTP {r.status_code}")
            except Exception as e:  # noqa: BLE001
                print(f"[constituents:{sym}] strategy3 exception: {e}")


    # ── broker feed fallback ──────────────────────────────────────────
    # Under DATA_SOURCE=fyers the NSE session may be unwarmed or blocked, but
    # the constituent list is already known from index_weights.json and the
    # broker can quote every name in it. Previously choosing a broker source
    # silently broke this widget, even though the data was in the feed.
    if not raw_stocks and RUN_DATA_SOURCE == "fyers":
        try:
            import nse_adapter_fyers as _fy
            names = list((_load_index_weights().get(sym.upper()) or {}).get("weights") or {})
            if not names:
                names = [n for n, _w in cfg["fallback_weights"]]
            q = _fy.constituent_quotes(names)
            if q:
                raw_stocks = list(q.values())
                print(f"[constituents] {sym}: {len(raw_stocks)} names from the Fyers feed")
        except Exception as e:  # noqa: BLE001
            print(f"[constituents] fyers fallback failed: {e}")

    if not raw_stocks:
        raise RuntimeError(
            f"No constituent data for {sym}. "
            + ("Fyers returned nothing - check the token with /api/data-source."
               if RUN_DATA_SOURCE == "fyers"
               else "Load an option chain first to warm the NSE session."))

    return _build_constituents_result(sym, raw_stocks, source="nse",
                                     data_quality=locals().get("data_quality", "live"))


def _stock_search(q: str) -> dict:
    """
    Return NSE symbols/names matching the query string.
    Searches the fno list + NSE equity index data.
    Results cached for 5 min to avoid repeated NSE calls.
    """
    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE  # noqa: PLC0415

    # Build or refresh the symbol+name list
    now = time.time()
    sc  = _stock_symbols_cache
    if not sc["symbols"] or (now - sc["ts"]) > 300:
        symbols = []
        # Seed from fno list
        try:
            import nse_pivot_scanner as _ps  # noqa: PLC0415
            for sym in _ps.load_fno_symbols():
                symbols.append({"s": sym, "n": sym})
        except Exception:
            pass
        # Augment with NSE equity-stockIndices bulk fetch
        ftch = (
            _shared_fetcher
            if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
                and (now - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE)
            else None
        )
        if ftch:
            h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE
            existing = {d["s"] for d in symbols}
            for idx in ("NIFTY 500", "NIFTY MIDCAP 100"):
                try:
                    r = ftch.session.get(
                        "https://www.nseindia.com/api/equity-stockIndices",
                        params={"index": idx}, headers=h, timeout=10,
                    )
                    if r.status_code == 200:
                        for item in r.json().get("data", []):
                            sym  = (item.get("symbol") or "").strip().upper()
                            name = (item.get("meta", {}) or {}).get("companyName", sym)
                            if sym and sym not in existing:
                                symbols.append({"s": sym, "n": name})
                                existing.add(sym)
                except Exception:
                    pass
        sc["symbols"] = symbols
        sc["ts"]      = now

    if not q:
        return {"results": sc["symbols"][:50]}

    # Filter: match against symbol prefix first, then substring in name
    q_up = q.upper()
    prefix = [d for d in sc["symbols"] if d["s"].startswith(q_up)]
    others = [d for d in sc["symbols"] if not d["s"].startswith(q_up)
              and (q_up in d["s"] or q_up in d["n"].upper())]
    return {"results": (prefix + others)[:50]}


class Handler(BaseHTTPRequestHandler):
    # Quiet the default per-request console spam; we print our own concise log line.
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        # A browser that navigated away, refreshed, or closed a tab leaves a
        # dead socket behind. Writing to it raises BrokenPipeError, which is
        # normal client behaviour rather than a server fault - swallow it (and
        # remember it, so the caller's error handler does not try to reply on
        # the same dead socket and produce a second traceback).
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self._client_gone = True
        except Exception as e:  # noqa: BLE001
            self._client_gone = True
            print(f"[!] response write failed: {e}")

    # ── connection-loss handling ──────────────────────────────────────
    # ThreadingHTTPServer prints a full traceback for every client that hangs
    # up mid-response. With a dashboard that opens SSE streams and polls on a
    # timer, that is routine (tab closed, page refreshed, laptop slept) and
    # floods the console. These overrides turn it into a one-line debug note.
    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_error(self, fmt, *args):
        msg = fmt % args if args else fmt
        if any(s in str(msg) for s in ("Broken pipe", "Connection reset", "Errno 32", "Errno 54")):
            return
        print(f"[http] {msg}")

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

            # Basic format sanity so an obviously-malformed symbol fails fast.
            if not symbol or not symbol.replace("-", "").replace("&", "").isalnum():
                self._send_json({"error": f"'{symbol}' doesn't look like a valid NSE symbol."}, status=400)
                return

            # Option chains exist ONLY for F&O instruments (the indices + the
            # ~180 F&O stocks). A cash-only stock has no options, so reject it
            # up front with a clear message instead of a confusing NSE error.
            # If the F&O list can't be loaded (NSE unreachable), fall through
            # and let the fetch attempt decide, rather than blocking everything.
            try:
                fno_set = {s.upper() for s in nse_lot_sizes.get_fno_symbol_list()}
            except Exception:
                fno_set = set()
            if fno_set and symbol not in fno_set:
                self._send_json({
                    "error": f"'{symbol}' is not an F&O symbol — it has no option chain. "
                             f"Option chains are available only for NIFTY/BANKNIFTY/FINNIFTY and "
                             f"the ~180 F&O stocks. (For cash stocks, use the Gann/Levels/esoteric "
                             f"tabs, which work on any NSE stock.)",
                    "not_fno": True,
                }, status=400)
                return

            client = self.client_address[0] if self.client_address else "?"
            try:
                data, meta = _cached_chain(symbol, expiry, band, client=client)
                # tell the UI how this was served (cache hit vs live fetch)
                try:
                    data = dict(data)
                    data["_cache"] = meta
                except Exception:
                    pass
                if meta["cache"] != "miss":
                    print(f"[cache {meta['cache']}] /api/chain {symbol} age={meta['age']}s -> {client}")
                else:
                    print(f"[FETCH] /api/chain {symbol} exp={expiry} band={band} -> {client}")
                self._send_json(data)
            except NSEFetchError as e:
                print(f"[!] Fetch error: {e}")
                self._send_json({"error": str(e)}, status=502)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass          # client disconnected mid-response; nothing to report
            except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI rather than hanging it
                print(f"[!] Unexpected error: {e}")
                if not getattr(self, "_client_gone", False):
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

        if parsed.path == "/api/ohlc-debug":
            # Calls every NSE source directly and reports the RAW response shape
            # for each one. Use this to see exactly what NSE returns right now —
            # field names in NSE's APIs are not officially documented and can
            # change without notice, so this is the ground-truth diagnostic.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            results = {}
            try:
                from nse_options_strategy import API_HEADERS, NSE_OC_PAGE, NSE_ALL_INDICES_API
                from datetime import date, timedelta

                global _shared_fetcher, _shared_fetcher_ts
                fetcher = (
                    _shared_fetcher
                    if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
                        and (time.time() - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE)
                    else NSESession()
                )
                if not getattr(fetcher, "_warmed", False):
                    fetcher._warm_up()
                    _shared_fetcher = fetcher
                    _shared_fetcher_ts = time.time()
                h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE

                index_map = {
                    "NIFTY":      ("NIFTY%2050",            "NIFTY 50"),
                    "BANKNIFTY":  ("NIFTY%20BANK",          "NIFTY BANK"),
                    "FINNIFTY":   ("NIFTY%20FIN%20SERVICE", "NIFTY FIN SERVICE"),
                    "MIDCPNIFTY": ("NIFTY%20MID%20SELECT",  "NIFTY MID SELECT"),
                }
                if symbol not in index_map:
                    self._send_json({"error": f"{symbol} not in index_map", "valid": list(index_map)})
                    return
                idx_code, idx_name = index_map[symbol]
                idx_enc = idx_name.replace(" ", "%20")
                today = date.today()

                # ── Source 1: indicesHistory ──
                try:
                    url = (f"https://www.nseindia.com/api/historical/indicesHistory"
                           f"?indexType={idx_enc}"
                           f"&from={(today-timedelta(days=10)).strftime('%d-%m-%Y')}"
                           f"&to={(today-timedelta(days=1)).strftime('%d-%m-%Y')}")
                    r1 = fetcher.session.get(url, headers=h, timeout=12)
                    results["1_indicesHistory"] = {
                        "url": url, "status": r1.status_code,
                        "raw_keys": list(r1.json().keys()) if r1.status_code == 200 else None,
                        "data_sample": (r1.json().get("data", {}).get("indexCloseOnlineRecords", [])[-2:]
                                        if r1.status_code == 200 else None),
                        "body_snippet": r1.text[:300] if r1.status_code != 200 else None,
                    }
                except Exception as e:
                    results["1_indicesHistory"] = {"error": str(e)}

                # ── Source 2: chart-databyindex ──
                try:
                    url2 = f"https://www.nseindia.com/api/chart-databyindex?index={idx_code}&indices=true"
                    r2 = fetcher.session.get(url2, headers=h, timeout=12)
                    body2 = r2.json() if r2.status_code == 200 else None
                    results["2_chartDataByIndex"] = {
                        "url": url2, "status": r2.status_code,
                        "raw_keys": list(body2.keys()) if isinstance(body2, dict) else None,
                        "grapthData_sample": (body2.get("grapthData", [])[-2:]
                                               if isinstance(body2, dict) else None),
                        "graphData_sample": (body2.get("graphData", [])[-2:]
                                              if isinstance(body2, dict) else None),
                        "body_snippet": r2.text[:300] if r2.status_code != 200 else None,
                    }
                except Exception as e:
                    results["2_chartDataByIndex"] = {"error": str(e)}

                # ── Source 4: allIndices ──
                try:
                    r4 = fetcher.session.get(NSE_ALL_INDICES_API, headers=h, timeout=10)
                    matched_row = None
                    if r4.status_code == 200:
                        for row in r4.json().get("data", []):
                            name = (row.get("index") or row.get("indexSymbol") or "").upper()
                            if idx_name.upper() in name or name in idx_name.upper():
                                matched_row = row
                                break
                    results["4_allIndices"] = {
                        "url": NSE_ALL_INDICES_API, "status": r4.status_code,
                        "matched_row": matched_row,
                        "body_snippet": r4.text[:300] if r4.status_code != 200 else None,
                    }
                except Exception as e:
                    results["4_allIndices"] = {"error": str(e)}

                # ── Source 5: equity-stockIndices ──
                try:
                    r5 = fetcher.session.get(
                        "https://www.nseindia.com/api/equity-stockIndices",
                        params={"index": idx_name}, headers=h, timeout=10)
                    results["5_equityStockIndices"] = {
                        "status": r5.status_code,
                        "data_sample": r5.json().get("data", [])[:2] if r5.status_code == 200 else None,
                        "body_snippet": r5.text[:300] if r5.status_code != 200 else None,
                    }
                except Exception as e:
                    results["5_equityStockIndices"] = {"error": str(e)}

                # ── Final result the dashboard actually uses ──
                results["final_result"] = _fetch_prev_day_ohlc(symbol)
                self._send_json({"symbol": symbol, "sources": results})
            except Exception as e:
                self._send_json({"error": str(e), "partial_results": results})
            return

        if parsed.path == "/api/health":
            self._send_json(_build_health_response())
            return

        if parsed.path == "/api/fii-dii":
            try:
                self._send_json(_fetch_fii_dii())
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)})
            return

        if parsed.path == "/api/gift-nifty":
            try:
                self._send_json(_fetch_gift_nifty())
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)})
            return

        if parsed.path == "/api/relative-strength":
            so = _session_ohlc
            result = {}
            for sym, spot in _index_spot_cache.items():
                entry = so.get(sym, {})
                open_ = entry.get("open") or spot
                result[sym] = {
                    "spot": spot,
                    "open": open_,
                    "pct": round((spot - open_) / open_ * 100, 2) if open_ else 0,
                }
            self._send_json(result)
            return

        if parsed.path == "/api/weekly-ohlc":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                data = _fetch_prev_week_ohlc(symbol)
                if data:
                    self._send_json(data)
                else:
                    self._send_json({"error": f"No weekly OHLC available for {symbol}. "
                                              "Bhavcopy files may not be cached yet."})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)})
            return

        if parsed.path == "/api/refresh-weights":
            try:
                _weights_cache.clear()
                data = _refresh_all_weights()
                self._send_json({"ok": True,
                                 "updated": data.get("_updated"),
                                 "indices": {k: {"count": v.get("count"),
                                                 "method": v.get("method")}
                                             for k, v in data.items()
                                             if not k.startswith("_")}})
            except Exception as e:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(e)})
            return

        if parsed.path == "/api/cleanup":
            # ?dry=1 to see what would go without removing anything
            dry = qs.get("dry", ["0"])[0] in ("1", "true", "yes")
            try:
                self._send_json(_run_cleanup(dry=dry))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, status=500)
            return

        if parsed.path == "/api/candle-health":
            out = {"runs": _recon_state["runs"],
                   "last_run": _recon_state["last"],
                   "age_sec": round(time.time() - _recon_state["last"], 1) if _recon_state["last"] else None,
                   "total_filled": _recon_state["filled"],
                   "total_upgraded": _recon_state["upgraded"],
                   "symbols": _recon_state["report"]}
            self._send_json(out)
            return

        if parsed.path == "/api/candles-debug":
            # Tells you exactly WHY the chart is not filling: which sources were
            # tried, what each returned, and where it failed. Built because
            # three rounds of fixes to the backfill produced no visible change
            # and there was no way to see which stage was breaking.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = time.strftime("%Y-%m-%d")
            rep = {"symbol": symbol, "day": day, "checks": []}

            def add(name, ok, detail):
                rep["checks"].append({"source": name, "ok": bool(ok), "detail": str(detail)[:300]})

            # 1. shared NSE session
            f = _shared_fetcher
            add("shared_session",
                f is not None,
                f"exists={f is not None} warmed={getattr(f, '_warmed', None)}"
                + ("" if f is not None else " - no option chain has been loaded yet, so nothing can reach NSE"))

            # 2. NSE chart feed, tried live right now
            idx_names = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                         "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MID SELECT"}
            if symbol in idx_names and f is not None:
                try:
                    from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
                    from urllib.parse import quote_plus as _qp
                    h = dict(API_HEADERS)
                    h["Referer"] = NSE_OC_PAGE
                    u = ("https://www.nseindia.com/api/chart-databyindex?index="
                         + _qp(idx_names[symbol]) + "&indices=true")
                    r = f.session.get(u, headers=h, timeout=12)
                    body = (r.text or "")[:200]
                    n = 0
                    if r.status_code == 200:
                        try:
                            pl = r.json()
                            n = len(pl.get("grapthData") or pl.get("graphData") or pl.get("data") or [])
                        except Exception as e:  # noqa: BLE001
                            body = f"json parse failed: {e} | {body}"
                    add("nse_chart_feed", r.status_code == 200 and n > 0,
                        f"HTTP {r.status_code}, {n} raw points. url={u} body~={body}")
                except Exception as e:  # noqa: BLE001
                    add("nse_chart_feed", False, f"{type(e).__name__}: {e}")
            else:
                add("nse_chart_feed", False, "skipped - no warmed session or unknown symbol")

            # 3. Yahoo
            try:
                y = _yahoo_intraday(symbol, 1)
                add("yahoo", len(y) > 0, f"{len(y)} session points returned")
            except Exception as e:  # noqa: BLE001
                add("yahoo", False, f"{type(e).__name__}: {e}")

            # 4. local files
            rpath = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            apath = _os.path.join(_TICK_DIR, f"{symbol}_{day}.csv")
            for label, p in (("replay_file", rpath), ("tick_archive", apath)):
                if _os.path.exists(p):
                    try:
                        lines = sum(1 for _ in open(p))
                    except Exception:
                        lines = -1
                    add(label, lines > 0, f"{p} exists, {lines} lines")
                else:
                    add(label, False, f"{p} does not exist")

            # 5. what a real call produces right now
            try:
                import urllib.request as _u
                inner = f"http://127.0.0.1:{self.server.server_address[1]}/api/intraday-candles?symbol={symbol}&interval=5&force=1"
                with _u.urlopen(inner, timeout=25) as rr:
                    j = json.loads(rr.read().decode())
                add("actual_result", bool(j.get("candles")),
                    f"source={j.get('source')} coverage={j.get('coverage')} "
                    f"candles={j.get('count')} error={j.get('error')}")
            except Exception as e:  # noqa: BLE001
                add("actual_result", False, f"{type(e).__name__}: {e}")

            rep["verdict"] = ("Backfill sources are all failing - see the first check that reads ok=false"
                              if not any(x["ok"] for x in rep["checks"][:3])
                              else "At least one backfill source is working")
            self._send_json(rep)
            return

        if parsed.path == "/api/intraday-candles":
            # Intraday OHLC candles for the chart widget.
            # Source 1: NSE chart-databyindex (live intraday data for indices);
            #           rows are [ts_ms, price] or [ts_ms,O,H,L,C] and get
            #           aggregated into N-minute candles here.
            # Source 2: today's replay archive (1 spot sample/min) — works with
            #           no extra network call once the recorder has run.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                interval = max(1, min(60, int(qs.get("interval", ["5"])[0])))
            except Exception:
                interval = 5
            idx_names = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                         "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MID SELECT"}

            def _bucket(ticks, mins):
                out, cur = [], None
                for ts, px in sorted(ticks):
                    b = int(ts // (mins * 60)) * (mins * 60)
                    if cur is None or cur["t"] != b:
                        if cur:
                            out.append(cur)
                        # "n" = tick count in the bucket. Index feeds carry no
                        # traded volume, so this is the only activity weight
                        # available; the client uses it for a tick-weighted
                        # VWAP and labels it honestly as such.
                        cur = {"t": b, "o": px, "h": px, "l": px, "c": px, "n": 1}
                    else:
                        cur["h"] = max(cur["h"], px)
                        cur["l"] = min(cur["l"], px)
                        cur["c"] = px
                        cur["n"] = cur.get("n", 1) + 1
                if cur:
                    out.append(cur)
                # Chain opens: with one spot sample per minute a 1m bucket holds
                # a single tick, giving O==H==L==C — an invisible, colourless
                # candle. Setting each open to the previous close produces a
                # continuous, properly coloured series (standard practice when
                # building candles from a sparse tick stream).
                bucket2 = interval * 60
                for i in range(1, len(out)):
                    if out[i]["t"] - out[i - 1]["t"] > bucket2:
                        continue          # do not chain across a recording gap
                    po = out[i - 1]["c"]
                    out[i]["o"] = po
                    out[i]["h"] = max(out[i]["h"], po)
                    out[i]["l"] = min(out[i]["l"], po)
                return out

            # ── Gather ticks from EVERY available source and merge them ──
            # A server started at 09:30 must still show 09:15 onward, so we:
            #   1. pull NSE's own intraday series (full day from the open),
            #      warming the session ourselves if nothing has warmed it yet
            #   2. add today's replay samples (covers any NSE gap)
            #   3. add the on-disk tick archive (survives server restarts)
            # then de-duplicate to one price per minute.
            ticks, srcs = [], []
            # ?force=1 makes the endpoint re-warm a session and re-pull NSE's
            # full-day series even if the local archive already has coverage.
            # Used by the chart's "fill from 9:15" button, so a server started
            # late can recover the morning on demand rather than only at boot.
            force = qs.get("force", ["0"])[0] in ("1", "true", "yes")

            def _warm_fetcher():
                """Return a usable session, preferring the one already working.

                Still used under DATA_SOURCE=fyers. Fyers supplies option-chain
                and index data, but the stock scanner, watchlist, pivots and
                results calendar are EQUITY endpoints with no Fyers wiring, and
                turning the NSE warm-up off at startup silently broke all nine.
                Warming lazily keeps them working while costing nothing on a
                session where nothing asks for them.

                The shared fetcher is what serves the option chain, so if the
                dashboard is loading data at all, that session is good. Reuse
                it rather than performing a fresh handshake - a new warm-up can
                fail (rate limit, cookie churn) while the working session sits
                idle, which made the backfill look broken even though NSE was
                perfectly reachable.
                """
                global _shared_fetcher
                f = _shared_fetcher
                if f is not None:
                    return f              # reuse whatever is serving the chain
                try:
                    f = NSESession()
                    f._warm_up()
                    try:
                        f._warmed = True
                    except Exception:
                        pass
                    if _shared_fetcher is None:
                        _shared_fetcher = f
                    return f
                except Exception as e:  # noqa: BLE001
                    print(f"[candles] could not warm a session: {e}")
                    return None

            # 0 ── Fyers history: genuine OHLC with TRADED VOLUME for the whole
            # session regardless of when this server started. That removes the
            # backfill problem entirely rather than working around it, and it
            # is the only source here that carries real volume - NSE gives one
            # spot sample a minute, from which volume cannot be recovered.
            if RUN_DATA_SOURCE == "fyers":
                try:
                    import nse_adapter_fyers as _fy
                    fc = _fy.fetch_candles(symbol, interval, 1)
                    if fc:
                        today_yday = time.gmtime(time.time() + 5 * 3600 + 1800).tm_yday
                        for cd in fc:
                            g = time.gmtime(cd["t"] + 5 * 3600 + 1800)
                            # date AND session, not just session - a multi-day
                            # history response would otherwise draw yesterday's
                            # candles alongside today's
                            if g.tm_yday != today_yday:
                                continue
                            if not _in_session(g.tm_hour * 60 + g.tm_min):
                                continue
                            ticks.append((float(cd["t"]), float(cd["c"])))
                        srcs.append(f"fyers({len(fc)})")
                        # real candles supersede reconstruction entirely
                        def _today_session(cd):
                            g = time.gmtime(cd["t"] + 5 * 3600 + 1800)
                            return (g.tm_yday == today_yday
                                    and _in_session(g.tm_hour * 60 + g.tm_min))
                        out = [cd for cd in fc if _today_session(cd)]
                        # Measure gaps rather than assuming there are none.
                        # Fyers history normally covers the whole session, but a
                        # partial response would otherwise be reported as clean.
                        fgaps = []
                        if out:
                            bucket = interval * 60
                            for a, b in zip(out, out[1:]):
                                miss = int((b["t"] - a["t"]) / bucket) - 1
                                if miss > 0:
                                    fgaps.append({"from": a["t"], "to": b["t"],
                                                  "mins": miss * interval,
                                                  "from_px": a["c"], "to_px": b["o"]})
                        # ── keep the tick archive current under Fyers too ──
                        # The branch used to return before the archive write, so
                        # the on-disk history simply stopped growing - which the
                        # reconciler, the coverage indicator and tomorrow's
                        # restart all depend on.
                        try:
                            apath_f = _os.path.join(_TICK_DIR, f"{symbol}_{time.strftime('%Y-%m-%d')}.csv")
                            _os.makedirs(_TICK_DIR, exist_ok=True)
                            existing = {}
                            if _os.path.exists(apath_f):
                                for line in open(apath_f):
                                    p = line.strip().split(",")
                                    if len(p) >= 2:
                                        try:
                                            existing[int(float(p[0]) // 60)] = float(p[1])
                                        except Exception:
                                            pass
                            for cd in out:
                                existing[int(cd["t"] // 60)] = cd["c"]
                            tmp = apath_f + ".tmp"
                            with open(tmp, "w") as f:
                                for m in sorted(existing):
                                    f.write(f"{m * 60},{existing[m]}\n")
                            _os.replace(tmp, apath_f)
                        except Exception as e:  # noqa: BLE001
                            print(f"[fyers] archive write failed: {e}")
                        if out:
                            first = _ist_hm_g(out[0]["t"]); last = _ist_hm_g(out[-1]["t"])
                            self._send_json({"symbol": symbol, "interval": interval,
                                             "source": f"fyers({len(out)})",
                                             "coverage": f"{first}-{last}",
                                             "gaps": fgaps,
                                             "gap_mins": sum(g["mins"] for g in fgaps),
                                             "candles": out, "count": len(out),
                                             "has_volume": True})
                            return
                except Exception as e:  # noqa: BLE001
                    print(f"[fyers] candles unavailable, falling back: {e}")

            # 1 ── NSE intraday series (authoritative back-history for today)
            if symbol in idx_names:
                if force:
                    print(f"[candles] FORCED backfill for {symbol} - re-warming session")
                ftch = _warm_fetcher()
                if ftch is not None:
                    try:
                        from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
                        # '+' not %20 - the same encoding that made every
                        # equity-stockIndices name 404. quote() breaks this feed.
                        from urllib.parse import quote_plus as _q
                        h = dict(API_HEADERS)
                        h["Referer"] = NSE_OC_PAGE
                        name = idx_names[symbol]
                        # This feed has appeared under more than one spelling.
                        # Try each and keep the first that returns real points,
                        # recording the failures so an empty chart is explainable.
                        # NSE serves index intraday under several shapes. The
                        # "&indices=true" form wants the index name; the plain
                        # form wants a SYMBOL identifier. Sending the wrong one
                        # returns HTTP 200 with an EMPTY array rather than an
                        # error, which is why this failed silently for so long.
                        nospace = name.replace(" ", "")
                        urls = [
                            f"https://www.nseindia.com/api/chart-databyindex?index={_q(name)}&indices=true",
                            f"https://www.nseindia.com/api/chart-databyindex?index={_q(nospace)}&indices=true",
                            f"https://www.nseindia.com/api/chart-databyindex?index={_q(nospace.upper())}&indices=true",
                            f"https://www.nseindia.com/api/chart-databyindex-dynamic?index={_q(name)}&type=symbol",
                            # equity-style identifier used by some index pages
                            f"https://www.nseindia.com/api/chart-databyindex?index={_q(name)}",
                        ]
                        got, why = 0, []
                        for url in urls:
                            try:
                                r = ftch.session.get(url, headers=h, timeout=12)
                            except Exception as e:  # noqa: BLE001
                                why.append(f"{type(e).__name__}")
                                continue
                            if r.status_code != 200:
                                why.append(f"HTTP{r.status_code}")
                                continue
                            try:
                                pl = r.json()
                            except Exception:
                                why.append("badjson")
                                continue
                            # Accept any of the shapes NSE has used, and if a
                            # 200 arrives with none of them, print the actual
                            # keys - an "empty" that repeats forever means the
                            # data is there under a name we are not reading.
                            rows = []
                            if isinstance(pl, dict):
                                for k in ("grapthData", "graphData", "data",
                                          "chartData", "indexChart", "records"):
                                    v = pl.get(k)
                                    if isinstance(v, list) and v:
                                        rows = v
                                        break
                                    if isinstance(v, dict):
                                        for k2 in ("grapthData", "graphData", "data"):
                                            v2 = v.get(k2)
                                            if isinstance(v2, list) and v2:
                                                rows = v2
                                                break
                                    if rows:
                                        break
                            elif isinstance(pl, list):
                                rows = pl
                            if not rows:
                                shape = (f"keys={list(pl)[:8]}" if isinstance(pl, dict)
                                         else f"type={type(pl).__name__}")
                                body = (r.text or "")[:200].replace("\n", " ")
                                why.append("empty")
                                print(f"[candles] 200 but no series: {shape} | body~={body}")
                                continue
                            # Keep anything inside today's IST session; the old
                            # code compared against LOCAL midnight, which is wrong
                            # on any server not running in IST.
                            for row in rows:
                                if not isinstance(row, (list, tuple)) or len(row) < 2:
                                    continue
                                try:
                                    ts = float(row[0]) / 1000.0
                                    px = float(row[4]) if len(row) >= 5 else float(row[1])
                                except Exception:
                                    continue
                                g = time.gmtime(ts + 5 * 3600 + 1800)
                                mins = g.tm_hour * 60 + g.tm_min
                                if g.tm_yday != time.gmtime(time.time() + 5 * 3600 + 1800).tm_yday:
                                    continue
                                if not (_in_session(mins)):
                                    continue
                                ticks.append((ts, px))
                                got += 1
                            if got:
                                srcs.append(f"nse_chart({got})")
                                break
                            why.append("no-session-points")
                        if not got:
                            print(f"[candles] chart feed gave nothing for {name}: {', '.join(why[:4])}")
                            srcs.append("nse_chart(0)")
                    except Exception as e:  # noqa: BLE001
                        print(f"[candles] chart feed failed: {e}")

            # 2 ── today's replay snapshots
            day = time.strftime("%Y-%m-%d")
            rpath = _os.path.join(_REPLAY_DIR, symbol + "_" + day + ".jsonl")
            if _os.path.exists(rpath):
                got = 0
                try:
                    with open(rpath) as f:
                        for line in f:
                            try:
                                s = json.loads(line)
                                v, t = s.get("underlying_value"), s.get("_replay_ts")
                                if v and t:
                                    ticks.append((float(t), float(v)))
                                    got += 1
                            except Exception:
                                continue
                except Exception:
                    pass
                if got:
                    srcs.append(f"replay({got})")

            # 2b ── ANY earlier replay file for today, including one written by
            # a previous run of the server before it was restarted. The current
            # process only appends to today's file, but a crash-restart can
            # leave more than one archive shard around; sweep them all so a
            # mid-day restart does not orphan the morning.
            try:
                import glob as _glob
                for extra in sorted(_glob.glob(_os.path.join(_REPLAY_DIR, f"{symbol}_{day}*.jsonl"))):
                    if extra == rpath:
                        continue
                    got = 0
                    with open(extra) as f:
                        for line in f:
                            try:
                                s = json.loads(line)
                                v, t = s.get("underlying_value"), s.get("_replay_ts")
                                if v and t:
                                    ticks.append((float(t), float(v)))
                                    got += 1
                            except Exception:
                                continue
                    if got:
                        srcs.append(f"replay-shard({got})")
            except Exception:
                pass

            # 3 ── persistent tick archive (so a restart never loses back-history)
            apath = _os.path.join(_TICK_DIR, symbol + "_" + day + ".csv")
            if _os.path.exists(apath):
                got = 0
                try:
                    with open(apath) as f:
                        for line in f:
                            p = line.strip().split(",")
                            if len(p) >= 2:
                                try:
                                    ticks.append((float(p[0]), float(p[1])))
                                    got += 1
                                except Exception:
                                    continue
                except Exception:
                    pass
                if got:
                    srcs.append(f"archive({got})")

            # 4 ── Yahoo Finance backfill (no auth, no warm-up).
            # Only consulted when the local sources leave real holes, and its
            # points are dropped for any minute we already hold - NSE data
            # always wins where both exist, so this fills gaps rather than
            # overwriting authoritative prices.
            if symbol in _YAHOO_SYMBOLS and _envbool_g("NSE_YAHOO_BACKFILL", True):
                have_mins = {int(t // 60) for t, _ in ticks}
                need_fill = force or not have_mins
                if have_mins and not need_fill:
                    lo_m, hi_m = min(have_mins), max(have_mins)
                    missing = (hi_m - lo_m + 1) - len(have_mins)
                    # the morning is missing if our earliest sample is late
                    g0 = time.gmtime(lo_m * 60 + 5 * 3600 + 1800)
                    starts_late = (g0.tm_hour * 60 + g0.tm_min) > 560   # after 09:20
                    need_fill = missing > 2 or starts_late
                if need_fill:
                    ypts = _yahoo_intraday(symbol, interval)
                    added = 0
                    for ts, px in ypts:
                        if int(ts // 60) in have_mins:
                            continue          # never overwrite a real NSE sample
                        ticks.append((ts, px))
                        added += 1
                    if added:
                        srcs.append(f"yahoo({added})")

            # ── keep only REGULAR SESSION ticks (09:15:00-15:30:00 IST) ──
            # NSE's intraday feed also carries the pre-open call auction
            # (09:00-09:15), whose indicative prices are not tradable and
            # distort the first candle plus every level derived from it.
            # This filter is on wall-clock IST, so a server started late still
            # keeps the full 09:15-onward back-history it fetched.
            def _ist_parts(ts):
                # IST = UTC+5:30, computed without relying on the host timezone
                ist = time.gmtime(ts + 5 * 3600 + 1800)
                return ist.tm_hour * 60 + ist.tm_min, ist.tm_sec, ist.tm_wday
            MKT_OPEN, MKT_CLOSE = SESSION_OPEN_MIN, SESSION_END_MIN
            before = len(ticks)
            ticks = [(ts, px) for (ts, px) in ticks
                     if MKT_OPEN <= _ist_parts(ts)[0] < MKT_CLOSE]
            dropped = before - len(ticks)
            if dropped:
                srcs.append(f"-preopen({dropped})")

            # ── report and interpolate GAPS ───────────────────────────
            # A mid-session restart leaves a hole: the archive stops when the
            # old process died and resumes when the new one warms up. We do
            # NOT invent prices inside a hole - that would draw candles that
            # never traded. Instead the gaps are measured and returned, so the
            # chart can show them honestly and the UI can say what is missing.
            gaps = []
            if ticks:
                by_min_pre = {}
                for ts, px in sorted(ticks):
                    by_min_pre.setdefault(int(ts // 60), px)
                mins_sorted = sorted(by_min_pre)
                for a, b in zip(mins_sorted, mins_sorted[1:]):
                    if b - a > 2:                     # more than a 2-minute hole
                        gaps.append({"from": a * 60, "to": b * 60,
                                     "mins": b - a - 1,
                                     "from_px": by_min_pre[a], "to_px": by_min_pre[b]})

            # ── de-duplicate to one sample per minute (first writer wins) ──
            if ticks:
                by_min = {}
                for ts, px in sorted(ticks):
                    by_min.setdefault(int(ts // 60), (ts, px))
                ticks = sorted(by_min.values())
                # persist the merged series so tomorrow's restart-at-noon works too
                try:
                    _os.makedirs(_TICK_DIR, exist_ok=True)
                    with open(apath, "w") as f:
                        for ts, px in ticks:
                            f.write(f"{int(ts)},{px}\n")
                except Exception as e:  # noqa: BLE001
                    print(f"[candles] archive write failed: {e}")

            if not ticks:
                self._send_json({"error": "no intraday data yet - the NSE chart feed is unreachable and nothing has been recorded today",
                                 "candles": [], "symbol": symbol, "interval": interval})
                return
            candles = _bucket(ticks, interval)
            # report coverage in IST regardless of the server's own timezone
            def _ist_hm(ts):
                g = time.gmtime(ts + 5 * 3600 + 1800)
                return f"{g.tm_hour:02d}:{g.tm_min:02d}"
            first, last = _ist_hm(ticks[0][0]), _ist_hm(ticks[-1][0])
            self._send_json({"symbol": symbol, "interval": interval,
                             "source": "+".join(srcs) or "none",
                             "coverage": f"{first}-{last}",
                             "gaps": gaps,
                             "gap_mins": sum(g["mins"] for g in gaps),
                             "candles": candles, "count": len(candles),
                             "asOf": time.strftime("%H:%M:%S")})
            return

        if parsed.path == "/api/alert-stream":
            # Server-Sent Events: one long-lived connection per browser tab.
            # This IS the connected-systems registry — if the socket is open
            # the tab is live, and when it closes the tab disappears with no
            # heartbeat guesswork.
            ip = self.client_address[0] if self.client_address else "?"
            cl = _sse_register(ip)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                with _sse_lock:
                    same = sum(1 for x in _sse_clients if x["ip"] == ip)
                    total = len(_sse_clients)
                hello = {"type": "hello", "client_id": cl["id"], "ip": ip,
                         "tabs_here": same, "total_tabs": total}
                self.wfile.write(f"data: {json.dumps(hello)}\n\n".encode())
                self.wfile.flush()
                print(f"[sse] connected {cl['id']} ({same} tab(s) on {ip}, {total} total)")
                while True:
                    try:
                        ev = cl["q"].get(timeout=15)
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    except Exception:
                        self.wfile.write(b": keepalive\n\n")   # keeps proxies from closing
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                _sse_unregister(cl)
                with _sse_lock:
                    print(f"[sse] disconnected {cl['id']} ({len(_sse_clients)} remain)")
            return

        if parsed.path == "/api/audio-status":
            # Browsers POST/GET their real audio capability here so failures on
            # a remote machine are visible instead of invisible: is the voice
            # toggle on, does the browser expose speechSynthesis, have voices
            # loaded, and has the page received the user gesture Chrome
            # requires before it will speak at all.
            ip = self.client_address[0] if self.client_address else "?"
            cid = qs.get("id", [""])[0]
            st = {
                "ip": ip, "id": cid,
                "toggle": qs.get("toggle", ["?"])[0],
                "synth": qs.get("synth", ["?"])[0],
                "voices": qs.get("voices", ["0"])[0],
                "unlocked": qs.get("unlocked", ["?"])[0],
                "spoke_ok": qs.get("spoke", ["?"])[0],
                "ua": (self.headers.get("User-Agent") or "")[:60],
                "ts": time.time(),
            }
            with _sse_lock:
                global _audio_status
                try:
                    _audio_status
                except NameError:
                    _audio_status = {}
                if cid:
                    _audio_status[cid] = st
                for k, v in list(_audio_status.items()):
                    if time.time() - v["ts"] > 120:
                        _audio_status.pop(k, None)
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/audio-report":
            now = time.time()
            with _sse_lock:
                reg = dict(globals().get("_audio_status") or {})
                live = {x["id"]: x for x in _sse_clients}
                by_ip = {}
                for x in _sse_clients:
                    by_ip.setdefault(x["ip"], []).append(x)
                speakers = {ip: sorted(v, key=lambda z: z["seq"])[0]["id"] for ip, v in by_ip.items()}
            rows = []
            for cid, st in sorted(reg.items(), key=lambda kv: kv[1]["ip"]):
                ready = (st["toggle"] == "1" and st["synth"] == "1"
                         and int(st["voices"] or 0) > 0 and st["unlocked"] == "1")
                rows.append({**st, "age": round(now - st["ts"], 1),
                             "connected": cid in live,
                             "is_speaker": speakers.get(st["ip"]) == cid,
                             "ready": ready})
            self._send_json({"clients": rows, "systems": len(by_ip),
                             "speaker_per_machine": speakers})
            return

        if parsed.path == "/api/alert-emit":
            # Generic relay: a browser that detects something locally (a pivot
            # or level touch, a user-defined alert) hands it to the server,
            # which de-duplicates against ONE shared history and broadcasts it
            # to every system. Without this each browser kept its own fired-map
            # and systems drifted out of sync — some announcing, some silent.
            key = qs.get("key", [""])[0]
            text = qs.get("text", [""])[0]
            kind = qs.get("kind", ["level"])[0]
            # the index this alert belongs to, so every browser can apply its
            # own per-index filter. Without it, relayed level touches reached
            # machines that had muted that index.
            sym = (qs.get("symbol", [""])[0] or "").upper()
            try:
                cool = float(qs.get("cooldown", ["300"])[0])
            except Exception:
                cool = 300.0
            if not key or not text:
                self._send_json({"ok": False, "error": "key and text required"})
                return
            now = time.time()
            with _sse_lock:
                global _relay_fired
                try:
                    _relay_fired
                except NameError:
                    _relay_fired = {}
                for k, t in list(_relay_fired.items()):
                    if now - t > 3600:
                        _relay_fired.pop(k, None)
                last = _relay_fired.get(key, 0)
                fresh = now - last >= cool
                if fresh:
                    _relay_fired[key] = now
            if not fresh:
                self._send_json({"ok": True, "broadcast": False,
                                 "reason": "already announced", "age": round(now - last, 1)})
                return
            _sse_broadcast({"type": kind, "ts": now, "key": key, "text": text,
                            "symbol": sym or None})
            with _sse_lock:
                n = len(_sse_clients)
            print(f"[relay] {kind}: {text[:60]} -> {n} tab(s)")
            self._send_json({"ok": True, "broadcast": True, "sent_to": n})
            return

        if parsed.path == "/api/m920":
            # One source of truth for the 09:20 lines so every system draws
            # and alerts on exactly the same values.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            lv = _m920.get(symbol)
            self._send_json(lv or {"error": "not frozen yet (before 09:20 IST or no chain data)"})
            return

        if parsed.path == "/api/connected":
            now = time.time()
            with _sse_lock:
                by_ip = {}
                for x in _sse_clients:
                    by_ip.setdefault(x["ip"], []).append(
                        {"id": x["id"], "seq": x["seq"], "up_sec": round(now - x["ts"])})
                speakers = {ip: sorted(v, key=lambda z: z["seq"])[0]["id"]
                            for ip, v in by_ip.items()}
                total = len(_sse_clients)
            self._send_json({"systems": len(by_ip), "total_tabs": total,
                             "by_machine": by_ip, "speaker_per_machine": speakers})
            return

        if parsed.path == "/api/alert-test":
            # fire a synthetic alert down the stream so multi-machine audio can
            # be verified without waiting for real market flow
            _sse_broadcast({"type": "test", "ts": time.time(),
                            "text": "Alert stream test. This system is announcing alerts.",
                            "kind": "test alert", "bias": "neutral"})
            with _sse_lock:
                n = len(_sse_clients)
            self._send_json({"sent_to": n})
            return

        if parsed.path == "/api/tab-ping":
            # Browsers heartbeat here so the server knows how many tabs each
            # computer has open. Used for the "tabs on this PC" readout and to
            # let clients see whether they are alone or one of several.
            tab = qs.get("tab", [""])[0]
            ip = self.client_address[0] if self.client_address else "?"
            now = time.time()
            with _chain_meta_lock:
                global _tab_registry
                try:
                    _tab_registry
                except NameError:
                    _tab_registry = {}
                if tab:
                    _tab_registry[(ip, tab)] = now
                # forget tabs silent for 45s (closed or asleep)
                for k, ts in list(_tab_registry.items()):
                    if now - ts > 45:
                        _tab_registry.pop(k, None)
                mine = sorted(t for (i, t) in _tab_registry if i == ip)
                machines = {}
                for (i, t) in _tab_registry:
                    machines[i] = machines.get(i, 0) + 1
            self._send_json({"ip": ip, "tabs_here": len(mine), "tab_ids": mine,
                             "machines": machines, "total_tabs": sum(machines.values()),
                             "is_first_tab": bool(mine) and mine[0] == tab})
            return

        if parsed.path == "/api/alert-claim":
            # Multi-browser voice de-duplication.
            # With several tabs/computers open, every one of them would
            # otherwise speak the same alert at once. Each client asks to
            # "claim" an event key; exactly ONE gets granted inside the TTL
            # window and speaks, the rest stay silent but still show toasts.
            key = qs.get("key", [""])[0]
            try:
                ttl = float(qs.get("ttl", ["300"])[0])
            except Exception:
                ttl = 300.0
            who = qs.get("who", ["?"])[0]
            # scope=machine (default): one speaker PER COMPUTER, so several tabs
            #   on the same PC never overlap while a second PC still announces.
            # scope=global: exactly one speaker across the whole network.
            scope = qs.get("scope", ["machine"])[0]
            ip = self.client_address[0] if self.client_address else "?"
            if scope == "machine":
                key = f"{ip}::{key}"
            if not key:
                self._send_json({"granted": False, "error": "key required"})
                return
            now = time.time()
            with _chain_meta_lock:
                global _alert_claims
                try:
                    _alert_claims
                except NameError:
                    _alert_claims = {}
                for k, v in list(_alert_claims.items()):
                    if now - v["ts"] > max(v["ttl"], 60) * 2:
                        _alert_claims.pop(k, None)
                cur = _alert_claims.get(key)
                if cur and now - cur["ts"] < cur["ttl"]:
                    self._send_json({"granted": False, "owner": cur["who"],
                                     "age": round(now - cur["ts"], 1)})
                    return
                _alert_claims[key] = {"ts": now, "ttl": ttl, "who": who}
            self._send_json({"granted": True, "key": key, "ttl": ttl})
            return

        if parsed.path == "/api/data-source":
            info = {"source": RUN_DATA_SOURCE, "streams": _adapter_streams(),
                    "polling": bool(_poller_symbols)}
            if RUN_DATA_SOURCE == "fyers":
                try:
                    import nse_adapter_fyers as _fy
                    info["fyers"] = _fy.status()
                    info["fyers"]["rest_today"] = _fy.rest_usage()
                    info["configured"] = _fy.is_configured()
                except Exception as e:  # noqa: BLE001
                    info["fyers"] = {"last_error": str(e)}
            if RUN_DATA_SOURCE == "arrow":
                try:
                    import nse_adapter_arrow as _arrow
                    info["arrow"] = _arrow.status()
                    info["configured"] = _arrow.is_configured()
                except Exception as e:  # noqa: BLE001
                    info["arrow"] = {"last_error": str(e)}
            self._send_json(info)
            return

        if parsed.path == "/api/cache-stats":
            now = time.time()
            with _chain_meta_lock:
                entries = [{"key": k, "age": round(now - v["ts"], 1), "shared_hits": v["hits"]}
                           for k, v in _chain_cache.items()]
                clients = {ip: round(now - t, 1) for ip, t in _cache_stats["clients"].items()
                           if now - t < 300}
                s = dict(_cache_stats)
            served, cached = s["served"], s["from_cache"]
            self._send_json({
                "ttl": _CHAIN_TTL,
                "served": served, "from_cache": cached, "fetches": s["fetches"],
                "coalesced": s["coalesced"],
                "hit_rate": round(cached / served * 100, 1) if served else 0.0,
                "nse_calls_saved": max(0, served - s["fetches"]),
                "entries": sorted(entries, key=lambda x: x["age"]),
                "active_clients": clients,
                "poller": _poller_symbols,
                "tabs": (lambda: (lambda reg: {"total": len(reg),
                                               "by_machine": {i: sum(1 for (x, _) in reg if x == i)
                                                              for (i, _) in reg}})(
                    {k: v for k, v in (globals().get("_tab_registry") or {}).items()
                     if time.time() - v < 45}))(),
            })
            return

        if parsed.path == "/api/today-open":
            # Official TODAY open for an index (allIndices feed) — needed for
            # TradingView-parity Woodie pivots. The polled session open can be
            # minutes late if the server starts after 09:15; this is the real
            # opening print. Cached 5 minutes.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            names = {"NIFTY": "NIFTY 50", "BANKNIFTY": "NIFTY BANK",
                     "FINNIFTY": "NIFTY FIN SERVICE", "MIDCPNIFTY": "NIFTY MID SELECT"}
            try:
                global _today_open_cache
                try:
                    _today_open_cache
                except NameError:
                    _today_open_cache = {}
                ck = symbol + time.strftime("%Y-%m-%d")
                hit = _today_open_cache.get(ck)
                if hit and time.time() - hit[0] < 300:
                    self._send_json(hit[1])
                    return
                from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
                fetcher = _shared_fetcher if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)) else None
                if fetcher is None:
                    self._send_json({"error": "session not warmed — load a chain first", "open": None})
                    return
                h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE
                r = fetcher.session.get("https://www.nseindia.com/api/allIndices", headers=h, timeout=10)
                row = next((x for x in r.json().get("data", []) if x.get("index") == names.get(symbol)), None)
                if not row or not row.get("open"):
                    self._send_json({"error": "index not found in allIndices", "open": None})
                    return
                out = {"symbol": symbol, "open": float(row["open"]), "source": "allIndices_official",
                       "asOf": time.strftime("%H:%M:%S")}
                _today_open_cache[ck] = (time.time(), out)
                self._send_json(out)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "open": None})
            return

        if parsed.path == "/api/ohlc":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                ohlc = _fetch_prev_day_ohlc(symbol)
                self._send_json(ohlc)
            except Exception as e:  # noqa: BLE001
                se = _session_ohlc.get(symbol, {})
                self._send_json({"symbol": symbol,
                    "open": se.get("open") or 0, "high": se.get("high") or 0,
                    "low": se.get("low") or 0, "close": se.get("close") or 0,
                    "prev_close": None, "source": "session", "error": str(e)})
            return

        if parsed.path == "/api/drafts":
            try:
                self._send_json({"drafts": nse_drafts.get_drafts()})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Could not load drafts: {e}"}, status=500)
            return

        if parsed.path == "/api/ltp-history":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                strike = float(qs.get("strike", ["0"])[0])
            except ValueError:
                strike = 0.0
            side = (qs.get("side", ["CE"])[0]).upper()
            self._send_json(_ltp_history_series(symbol, strike, side))
            return

        if parsed.path == "/api/confluence-backtest":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            ms = int(qs.get("min_score", ["3"])[0])
            self._send_json(compute_confluence_backtest(symbol, ms))
            return

        if parsed.path == "/api/alert-journal":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_alert_journal(symbol))
            return

        if parsed.path == "/api/anchored-vwap":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("day", [None])[0]
            self._send_json(compute_anchored_vwap(symbol, day))
            return

        if parsed.path == "/api/gap-orb-stats":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_gap_orb_stats(symbol))
            return

        if parsed.path == "/api/calendar-seasonality":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_calendar_seasonality(symbol))
            return

        if parsed.path == "/api/level-stats":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_level_stats(symbol))
            return

        if parsed.path == "/api/tpo-levels":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_tpo_levels(symbol))
            return

        if parsed.path == "/api/maxpain-stats":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_maxpain_stats(symbol))
            return

        if parsed.path == "/api/tod-seasonality":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(compute_tod_seasonality(symbol))
            return

        if parsed.path == "/api/results-calendar":
            fetcher = _shared_fetcher or NSESession()
            self._send_json({"events": _fetch_results_calendar(fetcher)})
            return

        if parsed.path == "/api/replay-dates":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json({"symbol": symbol, "dates": _replay_dates(symbol)})
            return

        if parsed.path == "/api/option-candles":
            # Intraday candles for a single OPTION (one strike, CE or PE),
            # rebuilt from the replay archive - which already stores the whole
            # chain every minute, so this needs no extra NSE traffic.
            # Also returns the strike's OI series, because for an option the
            # OI trace is as informative as the price trace.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            side = (qs.get("side", ["CE"])[0]).upper()
            try:
                strike = float(qs.get("strike", ["0"])[0])
            except Exception:
                strike = 0.0
            try:
                interval = max(1, min(60, int(qs.get("interval", ["5"])[0])))
            except Exception:
                interval = 5
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            if not strike or side not in ("CE", "PE"):
                self._send_json({"error": "strike and side (CE|PE) required", "candles": []})
                return
            if not _os.path.exists(path):
                self._send_json({"error": "no recorded chain for that day yet", "candles": []})
                return
            try:
                ticks, ois, ivs = [], [], []
                with open(path) as f:
                    for line in f:
                        try:
                            s = json.loads(line)
                        except Exception:
                            continue
                        ts = s.get("_replay_ts")
                        if not ts:
                            continue
                        for st_ in (s.get("strikes") or []):
                            if st_.get("strike") != strike:
                                continue
                            ltp = st_.get("ce_ltp" if side == "CE" else "pe_ltp")
                            oi = st_.get("ce_oi" if side == "CE" else "pe_oi")
                            iv = st_.get("ce_iv" if side == "CE" else "pe_iv")
                            if ltp:
                                ticks.append((float(ts), float(ltp)))
                                ois.append((float(ts), float(oi or 0)))
                                ivs.append((float(ts), float(iv or 0)))
                            break
                if not ticks:
                    self._send_json({"error": f"{int(strike)} {side} not present in today's recording",
                                     "candles": []})
                    return
                # keep only regular-session samples (IST), same rule as the index
                def _ist_min(t):
                    g = time.gmtime(t + 5 * 3600 + 1800)
                    return g.tm_hour * 60 + g.tm_min
                ticks = [(t, v) for t, v in ticks if _in_session(_ist_min(t))]
                ois = [(t, v) for t, v in ois if _in_session(_ist_min(t))]
                out, cur = [], None
                for ts, px in sorted(ticks):
                    b = int(ts // (interval * 60)) * (interval * 60)
                    if cur is None or cur["t"] != b:
                        if cur:
                            out.append(cur)
                        cur = {"t": b, "o": px, "h": px, "l": px, "c": px, "n": 1}
                    else:
                        cur["h"] = max(cur["h"], px)
                        cur["l"] = min(cur["l"], px)
                        cur["c"] = px
                        cur["n"] += 1
                if cur:
                    out.append(cur)
                # Chain opens so a one-sample bucket is not a flat, invisible
                # bar - but ONLY across contiguous candles. Chaining across a
                # recording gap would stretch the first post-gap candle over
                # the entire missing move, inventing a range that never traded
                # and corrupting the day high/low computed from it.
                bucket = interval * 60
                for i in range(1, len(out)):
                    if out[i]["t"] - out[i - 1]["t"] > bucket:
                        continue          # a gap sits between: leave the open alone
                    po = out[i - 1]["c"]
                    out[i]["o"] = po
                    out[i]["h"] = max(out[i]["h"], po)
                    out[i]["l"] = min(out[i]["l"], po)
                oi_series = []
                seen = set()
                for ts, v in sorted(ois):
                    b = int(ts // (interval * 60)) * (interval * 60)
                    if b in seen:
                        oi_series[-1] = {"t": b, "oi": v}
                    else:
                        seen.add(b)
                        oi_series.append({"t": b, "oi": v})
                self._send_json({"symbol": symbol, "strike": strike, "side": side,
                                 "interval": interval, "candles": out, "count": len(out),
                                 "oi": oi_series,
                                 "iv_last": ivs[-1][1] if ivs else None})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "candles": []})
            return

        if parsed.path == "/api/eod-report":
            # End-of-day session review, rebuilt from the replay archive.
            # This is the honest version of an "accuracy tracker": rather than
            # asserting a hit rate, it reports what actually happened to the
            # levels the dashboard drew that morning - which were touched,
            # which held, which broke - so the claim can be checked instead of
            # believed. It reads only recorded data, so it costs NSE nothing.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            apath = _os.path.join(_TICK_DIR, f"{symbol}_{day}.csv")
            spots = []
            if _os.path.exists(apath):
                try:
                    for line in open(apath):
                        p = line.strip().split(",")
                        if len(p) >= 2:
                            spots.append((float(p[0]), float(p[1])))
                except Exception:
                    pass
            snaps = []
            if _os.path.exists(path):
                for line in open(path):
                    try:
                        s = json.loads(line)
                        if s.get("_replay_ts") and s.get("underlying_value"):
                            snaps.append(s)
                    except Exception:
                        continue
            if not spots and not snaps:
                self._send_json({"error": "nothing recorded for that day", "ok": False})
                return
            if not spots and snaps:
                spots = [(float(s["_replay_ts"]), float(s["underlying_value"])) for s in snaps]
            spots.sort()
            prices = [p for _, p in spots]
            o, hi, lo, c_ = prices[0], max(prices), min(prices), prices[-1]

            # the 09:20 lines as they were frozen that morning
            lines = {}
            m920 = _m920.get(symbol)
            if m920 and m920.get("day") == day:
                lines = {k: m920[k] for k in ("EOR+1", "EOR", "EOS", "EOS-1") if k in m920}
            elif snaps:
                # reconstruct from the 09:20 snapshot if the live one is gone
                for s in snaps:
                    g = time.gmtime(s["_replay_ts"] + 5 * 3600 + 1800)
                    if g.tm_hour * 60 + g.tm_min >= 560:
                        sp, iv = s.get("underlying_value"), s.get("atm_iv")
                        if sp and iv:
                            sg = sp * (iv / 100.0) * (1.0 / 365.0) ** 0.5
                            lines = {"EOR+1": sp + sg, "EOR": sp + 0.5 * sg,
                                     "EOS": sp - 0.5 * sg, "EOS-1": sp - sg}
                        break

            level_report = []
            for lab, v in sorted(lines.items(), key=lambda kv: -kv[1]):
                touched = any(abs(p - v) <= max(4.0, v * 0.0006) for p in prices)
                broke_up = max(prices) > v + max(8.0, v * 0.0012)
                broke_dn = min(prices) < v - max(8.0, v * 0.0012)
                is_res = lab.startswith("EOR")
                held = (touched and not broke_up) if is_res else (touched and not broke_dn)
                level_report.append({
                    "label": lab, "value": round(v, 1), "touched": touched,
                    "held": bool(held),
                    "verdict": "held" if held else ("broke" if touched else "never reached"),
                })

            # OI extremes and where the walls ended up
            wall_note = {}
            if snaps:
                last = snaps[-1]
                st = last.get("strikes") or []
                if st:
                    ce = max(st, key=lambda x: x.get("ce_oi") or 0)
                    pe = max(st, key=lambda x: x.get("pe_oi") or 0)
                    first = snaps[0]
                    fst = first.get("strikes") or []
                    ce0 = max(fst, key=lambda x: x.get("ce_oi") or 0) if fst else None
                    pe0 = max(fst, key=lambda x: x.get("pe_oi") or 0) if fst else None
                    wall_note = {
                        "call_wall_open": ce0.get("strike") if ce0 else None,
                        "call_wall_close": ce.get("strike"),
                        "put_wall_open": pe0.get("strike") if pe0 else None,
                        "put_wall_close": pe.get("strike"),
                        "pcr_open": first.get("pcr"), "pcr_close": last.get("pcr"),
                    }

            rng = hi - lo
            body = abs(c_ - o)
            self._send_json({
                "ok": True, "symbol": symbol, "day": day,
                "open": round(o, 2), "high": round(hi, 2), "low": round(lo, 2),
                "close": round(c_, 2), "range": round(rng, 2),
                "change_pct": round((c_ - o) / o * 100, 2) if o else 0,
                "trend_day": bool(rng and body / rng >= 0.6),
                "body_pct": round(body / rng * 100, 1) if rng else 0,
                "minutes": len(spots), "snapshots": len(snaps),
                "lines": level_report, "walls": wall_note,
            })
            return

        if parsed.path == "/api/pcr-history":
            # Session PCR curve from the replay archive.
            # Returns BOTH ratios, because they answer different questions:
            #   OI-PCR     = accumulated positioning (where money already sits)
            #   Volume-PCR = what is trading TODAY (where conviction is now)
            # The spread between them is where fresh intent shows up - a rising
            # volume-PCR against a flat OI-PCR means puts are being bought now,
            # long before the OI figure reflects it.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            try:
                step = max(1, min(60, int(qs.get("step", ["5"])[0])))
            except Exception:
                step = 5
            try:
                band = max(0.5, min(10.0, float(qs.get("band", ["3"])[0])))
            except Exception:
                band = 3.0
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            if not _os.path.exists(path):
                self._send_json({"error": "no recorded chain for that day yet", "rows": []})
                return
            try:
                rows, last_bucket = [], None
                for line in open(path):
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    ts, spot = s.get("_replay_ts"), s.get("underlying_value")
                    strikes = s.get("strikes") or []
                    if not ts or not spot or not strikes:
                        continue
                    g = time.gmtime(ts + 5 * 3600 + 1800)
                    mins = g.tm_hour * 60 + g.tm_min
                    if not (_in_session(mins)):
                        continue
                    bucket = int(ts // (step * 60)) * (step * 60)
                    if bucket == last_bucket:
                        continue
                    last_bucket = bucket
                    ce_oi = pe_oi = ce_vol = pe_vol = 0.0
                    n_ce = n_pe = 0
                    for st_ in strikes:
                        k = st_.get("strike")
                        if not k or abs(k - spot) / spot * 100 > band:
                            continue
                        ce_oi += float(st_.get("ce_oi") or 0)
                        pe_oi += float(st_.get("pe_oi") or 0)
                        ce_vol += float(st_.get("ce_volume") or 0)
                        pe_vol += float(st_.get("pe_volume") or 0)
                        n_ce += 1
                        n_pe += 1
                    if not ce_oi:
                        continue
                    pcr_oi = pe_oi / ce_oi
                    pcr_vol = (pe_vol / ce_vol) if ce_vol else None
                    # whole-chain PCR as the server itself reported it, when present
                    rows.append({
                        "t": bucket, "spot": round(spot, 2),
                        "pcr_oi": round(pcr_oi, 3),
                        "pcr_vol": round(pcr_vol, 3) if pcr_vol else None,
                        "pcr_reported": s.get("pcr"),
                        "ce_oi": int(ce_oi), "pe_oi": int(pe_oi),
                    })
                if not rows:
                    self._send_json({"error": "no in-session snapshots recorded yet", "rows": []})
                    return
                first, last = rows[0], rows[-1]
                spreads = [r["pcr_vol"] - r["pcr_oi"] for r in rows if r["pcr_vol"] is not None]
                self._send_json({
                    "symbol": symbol, "day": day, "step": step, "band": band,
                    "rows": rows, "count": len(rows),
                    "open_pcr": first["pcr_oi"], "last_pcr": last["pcr_oi"],
                    "chg": round(last["pcr_oi"] - first["pcr_oi"], 3),
                    "hi": round(max(r["pcr_oi"] for r in rows), 3),
                    "lo": round(min(r["pcr_oi"] for r in rows), 3),
                    "spread_now": round(spreads[-1], 3) if spreads else None,
                    "spread_max": round(max(spreads, key=abs), 3) if spreads else None,
                })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "rows": []})
            return

        if parsed.path == "/api/oi-history":
            # Per-strike OPEN INTEREST through the session, rebuilt from the
            # replay archive. The migration card samples two points (30/60 min
            # back); this returns the whole curve so a wall can be watched
            # building or dissolving strike by strike.
            #
            # Returns both the absolute OI level and the change since the
            # first recorded snapshot, because they answer different questions:
            # level says where the wall IS, change says where it is FORMING.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            side = (qs.get("side", ["BOTH"])[0]).upper()
            try:
                step = max(1, min(60, int(qs.get("step", ["5"])[0])))
            except Exception:
                step = 5
            try:
                band = max(0.5, min(10.0, float(qs.get("band", ["4"])[0])))
            except Exception:
                band = 4.0
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            if not _os.path.exists(path):
                self._send_json({"error": "no recorded chain for that day yet", "cols": []})
                return
            try:
                cols, last_bucket, base = [], None, {}
                for line in open(path):
                    try:
                        s = json.loads(line)
                    except Exception:
                        continue
                    ts, spot = s.get("_replay_ts"), s.get("underlying_value")
                    strikes = s.get("strikes") or []
                    if not ts or not spot or not strikes:
                        continue
                    # regular session only (IST), same rule as the candles
                    g = time.gmtime(ts + 5 * 3600 + 1800)
                    mins = g.tm_hour * 60 + g.tm_min
                    if not (_in_session(mins)):
                        continue
                    bucket = int(ts // (step * 60)) * (step * 60)
                    if bucket == last_bucket:
                        continue
                    last_bucket = bucket
                    ce, pe = {}, {}
                    for st_ in strikes:
                        k = st_.get("strike")
                        if not k or abs(k - spot) / spot * 100 > band:
                            continue
                        c_oi = float(st_.get("ce_oi") or 0)
                        p_oi = float(st_.get("pe_oi") or 0)
                        ce[k] = c_oi
                        pe[k] = p_oi
                        base.setdefault(("CE", k), c_oi)
                        base.setdefault(("PE", k), p_oi)
                    cols.append({"t": bucket, "spot": round(spot, 2), "ce": ce, "pe": pe})
                if not cols:
                    self._send_json({"error": "no in-session snapshots recorded yet", "cols": []})
                    return
                # per-strike summary: where it started, where it is, net build
                last = cols[-1]
                summary = []
                for (sd, k), start in base.items():
                    if side != "BOTH" and sd != side:
                        continue
                    now = (last["ce"] if sd == "CE" else last["pe"]).get(k)
                    if now is None:
                        continue
                    summary.append({"strike": k, "side": sd, "start": start, "now": now,
                                    "chg": now - start,
                                    "pct": round((now - start) / start * 100, 1) if start else 0})
                summary.sort(key=lambda x: -abs(x["chg"]))
                self._send_json({"symbol": symbol, "day": day, "step": step, "band": band,
                                 "cols": cols, "count": len(cols),
                                 "summary": summary[:12],
                                 "spot": last["spot"]})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "cols": []})
            return

        if parsed.path == "/api/gex-history":
            # Per-minute gamma structure from today's replay snapshots.
            # The archive already records the whole chain every minute, so the
            # heatmap and the flip time-series come free - no extra NSE load.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            try:
                step = max(1, int(qs.get("step", ["5"])[0]))       # minutes per column
            except Exception:
                step = 5
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            if not _os.path.exists(path):
                self._send_json({"error": "no replay data for that day", "cols": []})
                return
            try:
                import math as _m
                def _norm_cdf(x):
                    return 0.5 * (1.0 + _m.erf(x / _m.sqrt(2.0)))
                def _gamma(S, K, T, iv, r=0.07):
                    if S <= 0 or K <= 0 or T <= 0 or iv <= 0.01:
                        return 0.0
                    sq = iv * _m.sqrt(T)
                    d1 = (_m.log(S / K) + (r + iv * iv / 2) * T) / sq
                    pdf = _m.exp(-d1 * d1 / 2) / _m.sqrt(2 * _m.pi)
                    return pdf / (S * sq)

                cols, last_bucket = [], None
                with open(path) as f:
                    for line in f:
                        try:
                            s = json.loads(line)
                        except Exception:
                            continue
                        ts = s.get("_replay_ts")
                        spot = s.get("underlying_value")
                        strikes = s.get("strikes") or []
                        if not ts or not spot or not strikes:
                            continue
                        bucket = int(ts // (step * 60)) * (step * 60)
                        if bucket == last_bucket:
                            continue                     # one column per bucket
                        last_bucket = bucket
                        dte = max(0.25, s.get("dte") or 1)
                        T = dte / 365.0
                        lot = s.get("lot_size") or 75
                        per_strike, cum, flip = {}, 0.0, None
                        rows = []
                        for st_ in strikes:
                            k = st_.get("strike")
                            if not k or abs(k - spot) / spot > 0.05:
                                continue
                            gc = _gamma(spot, k, T, (st_.get("ce_iv") or 0) / 100.0)
                            gp = _gamma(spot, k, T, (st_.get("pe_iv") or 0) / 100.0)
                            # same naive dealer convention as the client GEX panel
                            gex = (gc * (st_.get("ce_oi") or 0) - gp * (st_.get("pe_oi") or 0)) * lot * spot * spot * 0.01
                            per_strike[k] = round(gex)
                            rows.append((k, gex))
                        rows.sort()
                        for k, g in rows:
                            prev = cum
                            cum += g
                            if flip is None and prev < 0 <= cum and k:
                                flip = k
                        king = max(per_strike.items(), key=lambda kv: abs(kv[1]))[0] if per_strike else None
                        cols.append({"t": bucket, "spot": round(spot, 2),
                                     "gex": per_strike, "total": round(cum),
                                     "flip": flip, "king": king})
                self._send_json({"symbol": symbol, "day": day, "step": step,
                                 "cols": cols, "count": len(cols)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "cols": []})
            return

        if parsed.path == "/api/oi-migration":
            # Intraday OI velocity from TODAY's replay snapshots: compares the
            # latest snapshot's per-strike OI against snapshots ~30 and ~60 min
            # earlier. Shows which walls are BUILDING (absorbing fresh writing)
            # and which are DECAYING — the live version of the static OI panel.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
            try:
                if not _os.path.exists(path):
                    self._send_json({"error": "no replay data yet today — snapshots record while the chain auto-refreshes", "rows": []})
                    return
                snaps = []
                with open(path) as f:
                    for line in f:
                        try:
                            snaps.append(json.loads(line))
                        except Exception:
                            continue
                if len(snaps) < 2:
                    self._send_json({"error": "need ≥2 snapshots (≈2 min of recording)", "rows": []})
                    return
                latest = snaps[-1]
                t_now = latest.get("_replay_ts", 0)
                def _closest(mins):
                    tgt = t_now - mins * 60
                    return min(snaps[:-1], key=lambda s: abs(s.get("_replay_ts", 0) - tgt))
                s30, s60 = _closest(30), _closest(60)
                def _oi_map(snap):
                    m = {}
                    for st in snap.get("strikes", []):
                        m[st.get("strike")] = (st.get("ce_oi") or 0, st.get("pe_oi") or 0)
                    return m
                mNow, m30, m60 = _oi_map(latest), _oi_map(s30), _oi_map(s60)
                spot = latest.get("underlying_value") or 0
                rows = []
                for k, (ce, pe) in mNow.items():
                    if k is None or not spot or abs(k - spot) / spot > 0.04:
                        continue
                    c30, p30 = m30.get(k, (ce, pe))
                    c60, p60 = m60.get(k, (ce, pe))
                    rows.append({"strike": k,
                                 "ce_oi": ce, "pe_oi": pe,
                                 "ce_d30": ce - c30, "pe_d30": pe - p30,
                                 "ce_d60": ce - c60, "pe_d60": pe - p60})
                rows.sort(key=lambda r: -(abs(r["ce_d60"]) + abs(r["pe_d60"])))
                mins30 = round((t_now - s30.get("_replay_ts", t_now)) / 60)
                mins60 = round((t_now - s60.get("_replay_ts", t_now)) / 60)
                self._send_json({"symbol": symbol, "spot": spot, "rows": rows[:14],
                                 "win30": mins30, "win60": mins60,
                                 "asOf": time.strftime("%H:%M:%S", time.localtime(t_now)),
                                 "nSnaps": len(snaps)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "rows": []})
            return

        if parsed.path == "/api/replay-ohlc":
            # Daily OHLC per recorded replay day, extracted from each day's
            # snapshots (open = first, close = last, H/L = min/max of spot).
            # This is the fuel for the client-side "Claims on Trial" tests —
            # the history grows automatically each day the dashboard runs.
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                out = []
                for day in _replay_dates(symbol):
                    path = _os.path.join(_REPLAY_DIR, f"{symbol}_{day}.jsonl")
                    vals, ts0 = [], None
                    try:
                        with open(path) as f:
                            for line in f:
                                try:
                                    s = json.loads(line)
                                    v = s.get("underlying_value")
                                    if v:
                                        vals.append(float(v))
                                except Exception:
                                    continue
                    except Exception:
                        continue
                    if len(vals) >= 3:
                        out.append({"day": day, "O": vals[0], "H": max(vals), "L": min(vals),
                                    "C": vals[-1], "n": len(vals)})
                self._send_json({"symbol": symbol, "days": out, "count": len(out)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "days": []})
            return

        if parsed.path == "/api/replay-index":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            self._send_json(_replay_index(symbol, day))
            return

        if parsed.path == "/api/replay-snap":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            day = qs.get("date", [time.strftime("%Y-%m-%d")])[0]
            try:
                idx = int(qs.get("i", ["0"])[0])
            except ValueError:
                idx = 0
            snap = _replay_snapshot(symbol, day, idx)
            if snap is None:
                self._send_json({"error": "snapshot not found"}, status=404)
            else:
                self._send_json(snap)
            return

        if parsed.path == "/api/oi-timeline":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            self._send_json(_oi_timeline_grid(symbol))
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

        # ── 17 new historical-data scanners (explicit blocks) ────────────────
        def _scan17(mod_name, run_fn, lbl):
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import importlib; m = importlib.import_module(mod_name)
                self._send_json(getattr(m, run_fn)(source=source, force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
        def _sym17(mod_name, lbl):
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import importlib; m = importlib.import_module(mod_name)
                if mod_name == "nse_beta_scanner":
                    nr = m._load_nifty_rets(); r = m.scan_stock(symbol, nr) if nr else None
                elif mod_name == "nse_momrank_scanner":
                    r = {"error": "Per-symbol rank not available — run full scan first"}
                else:
                    r = m.scan_stock(symbol)
                self._send_json(r if r else {"error": f"No {lbl} data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})

        if parsed.path == "/api/nr7-scanner":          _scan17("nse_nr7_scanner","run_nr7_scanner","NR7"); return
        if parsed.path == "/api/nr7-scanner/symbol":   _sym17("nse_nr7_scanner","NR7"); return
        if parsed.path == "/api/insidebar-scanner":    _scan17("nse_insidebar_scanner","run_insidebar_scanner","InsideBar"); return
        if parsed.path == "/api/insidebar-scanner/symbol": _sym17("nse_insidebar_scanner","InsideBar"); return
        if parsed.path == "/api/atrpct-scanner":       _scan17("nse_atrpct_scanner","run_atrpct_scanner","ATR%"); return
        if parsed.path == "/api/atrpct-scanner/symbol": _sym17("nse_atrpct_scanner","ATR%"); return
        if parsed.path == "/api/zscore-scanner":       _scan17("nse_zscore_scanner","run_zscore_scanner","ZScore"); return
        if parsed.path == "/api/zscore-scanner/symbol": _sym17("nse_zscore_scanner","ZScore"); return
        if parsed.path == "/api/consec-scanner":       _scan17("nse_consec_scanner","run_consec_scanner","Consecutive"); return
        if parsed.path == "/api/consec-scanner/symbol": _sym17("nse_consec_scanner","Consecutive"); return
        if parsed.path == "/api/madist-scanner":       _scan17("nse_madist_scanner","run_madist_scanner","MADist"); return
        if parsed.path == "/api/madist-scanner/symbol": _sym17("nse_madist_scanner","MADist"); return
        if parsed.path == "/api/roundnum-scanner":     _scan17("nse_roundnum_scanner","run_roundnum_scanner","RoundNum"); return
        if parsed.path == "/api/roundnum-scanner/symbol": _sym17("nse_roundnum_scanner","RoundNum"); return
        if parsed.path == "/api/sar-scanner":          _scan17("nse_sar_scanner","run_sar_scanner","SAR"); return
        if parsed.path == "/api/sar-scanner/symbol":   _sym17("nse_sar_scanner","SAR"); return
        if parsed.path == "/api/donchian-scanner":     _scan17("nse_donchian_scanner","run_donchian_scanner","Donchian"); return
        if parsed.path == "/api/donchian-scanner/symbol": _sym17("nse_donchian_scanner","Donchian"); return
        if parsed.path == "/api/aroon-scanner":        _scan17("nse_aroon_scanner","run_aroon_scanner","Aroon"); return
        if parsed.path == "/api/aroon-scanner/symbol": _sym17("nse_aroon_scanner","Aroon"); return
        if parsed.path == "/api/hv-scanner":           _scan17("nse_hv_scanner","run_hv_scanner","HV"); return
        if parsed.path == "/api/hv-scanner/symbol":    _sym17("nse_hv_scanner","HV"); return
        if parsed.path == "/api/stage-scanner":        _scan17("nse_stage_scanner","run_stage_scanner","Stage"); return
        if parsed.path == "/api/stage-scanner/symbol": _sym17("nse_stage_scanner","Stage"); return
        if parsed.path == "/api/beta-scanner":         _scan17("nse_beta_scanner","run_beta_scanner","Beta"); return
        if parsed.path == "/api/beta-scanner/symbol":  _sym17("nse_beta_scanner","Beta"); return
        if parsed.path == "/api/squeeze-scanner":      _scan17("nse_squeeze_scanner","run_squeeze_scanner","Squeeze"); return
        if parsed.path == "/api/squeeze-scanner/symbol": _sym17("nse_squeeze_scanner","Squeeze"); return
        if parsed.path == "/api/elder-scanner":        _scan17("nse_elder_scanner","run_elder_scanner","Elder"); return
        if parsed.path == "/api/elder-scanner/symbol": _sym17("nse_elder_scanner","Elder"); return
        if parsed.path == "/api/swing-scanner":        _scan17("nse_swing_scanner","run_swing_scanner","Swing"); return
        if parsed.path == "/api/swing-scanner/symbol": _sym17("nse_swing_scanner","Swing"); return
        if parsed.path == "/api/momrank-scanner":      _scan17("nse_momrank_scanner","run_momrank_scanner","MomRank"); return
        if parsed.path == "/api/momrank-scanner/symbol": _sym17("nse_momrank_scanner","MomRank"); return

        # ── RS / CPR / Stoch / WilliamsR / CCI / Ichimoku / Darvas / Confluence ──
        for _path, _mod, _run_fn, _scan_fn, _lbl in [
            ("/api/rs-scanner",         "nse_rs_scanner",          "run_rs_scanner",          "scan_stock",   "RS"),
            ("/api/cpr-scanner",        "nse_cpr_scanner",         "run_cpr_scanner",         "scan_stock",   "CPR"),
            ("/api/stoch-scanner",      "nse_stoch_scanner",       "run_stoch_scanner",       "scan_stock",   "Stochastic"),
            ("/api/williamsr-scanner",  "nse_williamsr_scanner",   "run_williamsr_scanner",   "scan_stock",   "Williams%R"),
            ("/api/cci-scanner",        "nse_cci_scanner",         "run_cci_scanner",         "scan_stock",   "CCI"),
            ("/api/ichimoku-scanner",   "nse_ichimoku_scanner",    "run_ichimoku_scanner",    "scan_stock",   "Ichimoku"),
            ("/api/darvas-scanner",     "nse_darvas_scanner",      "run_darvas_scanner",      "scan_stock",   "Darvas"),
            ("/api/confluence-scanner", "nse_confluence_scanner",  "run_confluence_scanner",  None,           "Confluence"),
        ]:
            if parsed.path == _path:
                source = qs.get("source",["niftyfno"])[0].lower()
                force  = qs.get("force", ["false"])[0].lower() == "true"
                try:
                    import importlib as _il; _m = _il.import_module(_mod)
                    self._send_json(getattr(_m, _run_fn)(source=source, force=force))
                except Exception as _e: self._send_json({"stocks":[],"error":str(_e),"count":0,"total":0})
                return
            if _scan_fn and parsed.path == _path + "/symbol":
                symbol = qs.get("symbol",[""])[0].upper().strip()
                if not symbol: self._send_json({"error":"symbol required"}); return
                try:
                    import importlib as _il; _m = _il.import_module(_mod)
                    if _mod == "nse_rs_scanner":
                        nr = _m._load_nifty_close(); r = _m.scan_stock(symbol, nr) if nr else None
                    else:
                        r = getattr(_m, _scan_fn)(symbol)
                    self._send_json(r if r else {"error":f"No {_lbl} data for {symbol}"})
                except Exception as _e: self._send_json({"error":str(_e),"symbol":symbol})
                return

        # ── Bollinger Band Scanner ────────────────────────────────────
        if parsed.path in ("/api/bb-scanner", "/api/bollinger-scanner"):
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_bb_scanner as _bb; self._send_json(_bb.run_bb_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/bb-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_bb_scanner as _bb; r = _bb.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No BB data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── MACD Scanner ───────────────────────────────────────────────
        if parsed.path == "/api/macd-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_macd_scanner as _mc; self._send_json(_mc.run_macd_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/macd-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_macd_scanner as _mc; r = _mc.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No MACD data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── ADX Scanner ───────────────────────────────────────────────
        if parsed.path == "/api/adx-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_adx_scanner as _ax; self._send_json(_ax.run_adx_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/adx-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_adx_scanner as _ax; r = _ax.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No ADX data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Breakout Scanner ──────────────────────────────────────────
        if parsed.path == "/api/breakout-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_breakout_scanner as _bk; self._send_json(_bk.run_breakout_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/breakout-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_breakout_scanner as _bk; r = _bk.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No breakout for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── S/R Cluster Scanner ───────────────────────────────────────
        if parsed.path == "/api/sr-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_sr_scanner as _sr; self._send_json(_sr.run_sr_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/sr-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_sr_scanner as _sr; r = _sr.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No S/R clusters for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Price Pattern Scanner ─────────────────────────────────────
        if parsed.path == "/api/pattern-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_pattern_scanner as _pp; self._send_json(_pp.run_pattern_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/pattern-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_pattern_scanner as _pp; r = _pp.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No patterns detected for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Correlation Scanner ───────────────────────────────────────
        if parsed.path == "/api/correlation-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            try:
                import nse_correlation_scanner as _cr; self._send_json(_cr.run_correlation_scanner(source=source,force=force))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/correlation-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_correlation_scanner as _cr
                import nse_correlation_scanner as _cr2
                nr = _cr2._load_nifty_returns()
                r  = _cr.scan_stock(symbol, nr) if nr else None
                self._send_json(r if r else {"error":"Nifty CSV not found or insufficient data"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── RSI Scanner ────────────────────────────────────────────────
        if parsed.path == "/api/rsi-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl",["300"])[0])
            try:
                import nse_rsi_scanner as _rs  # noqa: PLC0415
                self._send_json(_rs.run_rsi_scanner(source=source,force=force,ttl=ttl))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/rsi-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_rsi_scanner as _rs  # noqa: PLC0415
                r = _rs.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Gap Scanner ─────────────────────────────────────────────────
        if parsed.path == "/api/gap-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl",["300"])[0])
            try:
                import nse_gap_scanner as _gs  # noqa: PLC0415
                self._send_json(_gs.run_gap_scanner(source=source,force=force,ttl=ttl))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/gap-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_gap_scanner as _gs  # noqa: PLC0415
                r = _gs.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Momentum / ROC Scanner ──────────────────────────────────────
        if parsed.path == "/api/momentum-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl",["300"])[0])
            try:
                import nse_momentum_scanner as _ms  # noqa: PLC0415
                self._send_json(_ms.run_momentum_scanner(source=source,force=force,ttl=ttl))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/momentum-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_momentum_scanner as _ms  # noqa: PLC0415
                r = _ms.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No data for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        # ── Candlestick Pattern Scanner ─────────────────────────────────
        if parsed.path == "/api/candle-scanner":
            source = qs.get("source",["niftyfno"])[0].lower()
            force  = qs.get("force", ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl",["300"])[0])
            try:
                import nse_candle_scanner as _cs  # noqa: PLC0415
                self._send_json(_cs.run_candle_scanner(source=source,force=force,ttl=ttl))
            except Exception as e: self._send_json({"stocks":[],"error":str(e),"count":0,"total":0})
            return
        if parsed.path == "/api/candle-scanner/symbol":
            symbol = qs.get("symbol",[""])[0].upper().strip()
            if not symbol: self._send_json({"error":"symbol required"}); return
            try:
                import nse_candle_scanner as _cs  # noqa: PLC0415
                r = _cs.scan_stock(symbol)
                self._send_json(r if r else {"error":f"No pattern detected for {symbol}"})
            except Exception as e: self._send_json({"error":str(e),"symbol":symbol})
            return

        if parsed.path == "/api/volume-scanner":
            source = qs.get("source", ["niftyfno"])[0].lower()
            force  = qs.get("force",  ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl", ["300"])[0])
            try:
                import nse_volume_scanner as _vs  # noqa: PLC0415
                self._send_json(_vs.run_volume_scanner(source=source, force=force, ttl=ttl))
            except Exception as e:  # noqa: BLE001
                self._send_json({"stocks": [], "error": str(e), "count": 0, "total": 0})
            return

        if parsed.path == "/api/volume-scanner/symbol":
            symbol = qs.get("symbol", [""])[0].upper().strip()
            if not symbol:
                self._send_json({"error": "symbol parameter required"}); return
            try:
                import nse_volume_scanner as _vs  # noqa: PLC0415
                r = _vs.scan_stock(symbol)
                self._send_json(r if r else {"error": f"No volume data for {symbol}"})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "symbol": symbol})
            return

        if parsed.path == "/api/52week-scanner":
            source = qs.get("source", ["niftyfno"])[0].lower()
            force  = qs.get("force",  ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl", ["300"])[0])
            try:
                import nse_52week_scanner as _wk  # noqa: PLC0415
                self._send_json(_wk.run_52week_scanner(source=source, force=force, ttl=ttl))
            except Exception as e:  # noqa: BLE001
                self._send_json({"stocks": [], "error": str(e), "count": 0, "total": 0})
            return

        if parsed.path == "/api/52week-scanner/symbol":
            symbol = qs.get("symbol", [""])[0].upper().strip()
            if not symbol:
                self._send_json({"error": "symbol parameter required"}); return
            try:
                import nse_52week_scanner as _wk  # noqa: PLC0415
                r = _wk.scan_stock(symbol)
                self._send_json(r if r else {"error": f"Insufficient history for {symbol}"})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "symbol": symbol})
            return

        if parsed.path == "/api/trend-scanner/symbol":
            symbol = qs.get("symbol", [""])[0].upper().strip()
            if not symbol:
                self._send_json({"error": "symbol parameter required"})
                return
            try:
                import nse_trend_scanner as _ts  # noqa: PLC0415
                result = _ts.scan_stock(symbol)
                if result is None:
                    self._send_json({"error": f"No data for {symbol} — CSV missing or insufficient rows"})
                else:
                    self._send_json(result)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "symbol": symbol})
            return

        if parsed.path == "/api/trend-scanner":
            source = qs.get("source", ["niftyfno"])[0].lower()
            force  = qs.get("force",  ["false"])[0].lower() == "true"
            ttl    = int(qs.get("ttl", ["300"])[0])
            try:
                import nse_trend_scanner as _ts  # noqa: PLC0415
                self._send_json(_ts.run_trend_scanner(source=source, force=force, ttl=ttl))
            except Exception as e:  # noqa: BLE001
                self._send_json({"stocks": [], "error": str(e), "count": 0, "total": 0})
            return

        if parsed.path == "/api/pivot-scanner":
            mode       = qs.get("mode",       ["daily"])[0].lower()
            pivot_type = qs.get("pivot_type", ["fibonacci"])[0].lower()
            source     = qs.get("source",     ["fno"])[0].lower()
            live       = qs.get("live",       ["false"])[0].lower() == "true"
            force      = qs.get("force",      ["false"])[0].lower() == "true"
            ttl        = int(qs.get("ttl",    ["300"])[0])
            try:
                import nse_pivot_scanner as _ps  # noqa: PLC0415
                from nse_options_strategy import API_HEADERS, NSE_OC_PAGE  # noqa: PLC0415
                ftch = (_shared_fetcher if (
                    _shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
                    and (time.time() - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE
                ) else None)
                h = None
                if ftch:
                    h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE
                result = _ps.run_scanner(mode=mode, pivot_type=pivot_type, source=source,
                                         live=live, fetcher=ftch, headers=h,
                                         force=force, ttl=ttl)
                self._send_json(result)
            except Exception as e:  # noqa: BLE001
                self._send_json({"stocks": [], "error": str(e), "count": 0, "total": 0})
            return

        if parsed.path == "/api/pivot-scanner/symbols":
            try:
                import nse_pivot_scanner as _ps  # noqa: PLC0415
                syms = _ps.load_all_csv_symbols()
                self._send_json({"symbols": syms, "count": len(syms)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"symbols": [], "error": str(e)})
            return

        if parsed.path == "/api/pivot-scanner/symbol":
            symbol     = qs.get("symbol",     [""])[0].upper().strip()
            mode       = qs.get("mode",       ["daily"])[0].lower()
            pivot_type = qs.get("pivot_type", ["fibonacci"])[0].lower()
            if not symbol:
                self._send_json({"error": "symbol parameter required"})
                return
            try:
                import nse_pivot_scanner as _ps  # noqa: PLC0415
                self._send_json(_ps.lookup_symbol(symbol, mode=mode, pivot_type=pivot_type))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "symbol": symbol})
            return

        if parsed.path in ("/api/pivot-scanner/debug", "/api/pivot-scanner-debug"):
            symbol = qs.get("symbol", [""])[0].upper().strip()
            try:
                import nse_pivot_scanner as _ps  # noqa: PLC0415
                if symbol:
                    self._send_json(_ps.debug_symbol(symbol))
                else:
                    # Return global state + first 3 symbols as sample
                    syms = _ps.load_fno_symbols()
                    sample = [_ps.debug_symbol(s) for s in syms[:3]]
                    self._send_json({
                        "data_dir":        str(_ps.DATA_DIR),
                        "fno_file":        str(_ps.FNO_FILE),
                        "data_dir_exists": _ps.DATA_DIR.exists(),
                        "fno_file_exists": _ps.FNO_FILE.exists(),
                        "total_symbols":   len(syms),
                        "first_3_symbols": syms[:3],
                        "sample_debug":    sample,
                    })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)})
            return

        if parsed.path == "/api/strength-rank":
            # F&O strength ranking.
            #
            # This was first built on /api/equity-stockIndices, which now 404s
            # for every index name - the endpoint appears to have been retired.
            # Rebuilt on quote-equity instead, which the dashboard already uses
            # successfully for the stock watchlist, so it works wherever that
            # works. The trade-off is one request per symbol, so results are
            # cached for 60s and the universe is capped per call; the scan runs
            # in a thread pool to keep it to a few seconds rather than a minute.
            #
            # Score is disclosed, not proprietary:
            #   60% today's % change
            #   25% position within today's own range (holding vs fading)
            #   15% range-normalised move (2% in a quiet name > 2% in a wild one)
            try:
                top = max(5, min(50, int(qs.get("top", ["15"])[0])))
            except Exception:
                top = 15
            try:
                universe_cap = max(20, min(220, int(qs.get("cap", ["120"])[0])))
            except Exception:
                universe_cap = 120
            ftch = _shared_fetcher if (_shared_fetcher and getattr(_shared_fetcher, "_warmed", False)) else None
            if ftch is None:
                self._send_json({"error": "session not warmed - load an option chain first", "rows": []})
                return
            try:
                global _strength_cache
                try:
                    _strength_cache
                except NameError:
                    _strength_cache = {}
                ckey = f"rank|{universe_cap}"
                hit = _strength_cache.get(ckey)
                if hit and time.time() - hit[0] < 60:
                    self._send_json(hit[1])
                    return

                # ── universe: the F&O list, from whichever source has it ──
                syms, src = [], None
                try:
                    import nse_lot_sizes
                    syms = [s.upper() for s in (nse_lot_sizes.get_fno_symbol_list() or [])]
                    if syms:
                        src = "nse_lot_sizes"
                except Exception:
                    pass
                if not syms:
                    # fall back to the lot-size table's own keys
                    try:
                        import nse_lot_sizes
                        tbl = getattr(nse_lot_sizes, "LOT_SIZES", None) or getattr(nse_lot_sizes, "_LOTS", None)
                        if isinstance(tbl, dict):
                            syms = [k.upper() for k in tbl if k.upper() not in
                                    ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50")]
                            src = "lot-size table"
                    except Exception:
                        pass
                if not syms:
                    self._send_json({"error": "no F&O symbol list available - "
                                              "nse_lot_sizes.get_fno_symbol_list() returned nothing, "
                                              "so there is no universe to rank",
                                     "rows": []})
                    return
                syms = syms[:universe_cap]

                from nse_options_strategy import API_HEADERS, NSE_OC_PAGE
                h = dict(API_HEADERS)
                h["Referer"] = NSE_OC_PAGE

                def _one(sym):
                    try:
                        d = fetch_live_stock(sym, ftch, h)
                        if not d:
                            return None
                        ltp = float(d.get("price") or 0)
                        prev = float(d.get("prevClose") or 0)
                        hi = float(d.get("high") or 0)
                        lo = float(d.get("low") or 0)
                        opn = float(d.get("open") or 0)
                        if not ltp or not prev:
                            return None
                        pchg = (ltp - prev) / prev * 100
                        rng = hi - lo
                        pos = ((ltp - lo) / rng) if rng > 0 else 0.5
                        norm = (pchg / (rng / prev * 100)) if rng > 0 else 0
                        score = (pchg * 0.6
                                 + (pos - 0.5) * 100 * 0.25
                                 + max(-3.0, min(3.0, norm)) * 1.5)
                        return {"symbol": sym, "ltp": round(ltp, 2), "pchg": round(pchg, 2),
                                "open": round(opn, 2), "high": round(hi, 2), "low": round(lo, 2),
                                "prev": round(prev, 2), "range_pos": round(pos, 3),
                                "score": round(score, 2)}
                    except Exception:
                        return None

                from concurrent.futures import ThreadPoolExecutor
                rows = []
                with ThreadPoolExecutor(max_workers=8) as pool:
                    for res in pool.map(_one, syms):
                        if res:
                            rows.append(res)
                if not rows:
                    self._send_json({"error": f"no quotes returned for any of {len(syms)} symbols - "
                                              "the warmed session may have expired; reload an option chain",
                                     "rows": []})
                    return
                rows.sort(key=lambda x: -x["score"])
                out = {"index": f"F&O universe ({src})", "requested": "fno",
                       "count": len(rows), "scanned": len(syms),
                       "strong": rows[:top], "weak": rows[-top:][::-1],
                       "asOf": time.strftime("%H:%M:%S")}
                _strength_cache[ckey] = (time.time(), out)
                self._send_json(out)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "rows": []})
            return

        if parsed.path == "/api/watchlist-prices":
            # Batch live prices for the stock watchlist (TradingView-style rows).
            # ?symbols=RELIANCE,TCS,BALAMINES — live fetch per symbol via the
            # warmed session, NO caching. Capped to 20 symbols per request.
            raw_syms = qs.get("symbols", [""])[0]
            syms = [s.strip().upper() for s in raw_syms.split(",") if s.strip()][:20]
            if not syms:
                self._send_json({"error": "symbols parameter required"}, status=400)
                return
            ftch = (_shared_fetcher if (
                _shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
                and (time.time() - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE
            ) else None)
            if ftch is None:
                self._send_json({"error": "NSE session not warmed — load an option chain first.",
                                 "quotes": {}})
                return
            h = None
            try:
                from nse_options_strategy import API_HEADERS, NSE_OC_PAGE  # noqa: PLC0415
                h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE
            except Exception:
                h = None
            quotes = {}
            for s in syms:
                q = fetch_live_stock(s, ftch, h)
                if q:
                    quotes[s] = {"price": q["price"], "change": q.get("change", 0),
                                 "pChange": q.get("pChange", 0), "H": q["H"], "L": q["L"]}
            self._send_json({"quotes": quotes, "count": len(quotes),
                             "asOf": time.strftime("%H:%M:%S")})
            return

        if parsed.path == "/api/stock-search":
            q = qs.get("q", [""])[0].strip().upper()
            try:
                self._send_json(_stock_search(q))
            except Exception as e:  # noqa: BLE001
                self._send_json({"results": [], "error": str(e)})
            return

        if parsed.path == "/api/stock-pivot":
            sym        = qs.get("symbol", [""])[0].strip().upper()
            mode       = qs.get("mode",       ["daily"])[0].lower()
            pivot_type = qs.get("pivot_type", ["fibonacci"])[0].lower()
            if not sym:
                self._send_json({"error": "symbol required"}, status=400)
                return
            try:
                import nse_pivot_scanner as _ps  # noqa: PLC0415
                # Live-price fetcher (shared session if warmed).
                ftch = (_shared_fetcher if (
                    _shared_fetcher and getattr(_shared_fetcher, "_warmed", False)
                    and (time.time() - _shared_fetcher_ts) < _SHARED_FETCHER_MAX_AGE
                ) else None)
                h = None
                if ftch:
                    try:
                        from nse_options_strategy import API_HEADERS, NSE_OC_PAGE  # noqa: PLC0415
                        h = dict(API_HEADERS); h["Referer"] = NSE_OC_PAGE
                    except Exception:
                        h = None

                # PRIMARY: live NSE quote (no caching). Works for any equity —
                # index-constituent or not — via the warmed session.
                live = fetch_live_stock(sym, ftch, h) if ftch else None
                ohlc = None
                price = 0.0
                is_live = False
                data_src = ""
                if live:
                    ohlc = {"O": live["O"], "H": live["H"], "L": live["L"], "C": live["C"]}
                    price = live["price"]
                    is_live = True
                    data_src = live.get("source", "nse-live")
                else:
                    # FALLBACK: local CSV only if the user happens to have one.
                    ohlc_csv, ohlc_err = (_ps.get_weekly_ohlc(sym) if mode == "weekly" else _ps.get_daily_ohlc(sym))
                    if not ohlc_csv:
                        self._send_json({"error": (
                            f"No live data for {sym} (NSE session not warmed yet — "
                            f"load an option chain first to warm it), and no local CSV. "
                            f"Detail: {ohlc_err or 'n/a'}")})
                        return
                    ohlc = ohlc_csv
                    price = float(ohlc.get("C", 0))
                    data_src = "csv"

                pivots = _ps.compute_pivots(
                    H=float(ohlc.get("H", 0)), L=float(ohlc.get("L", 0)),
                    C=float(ohlc.get("C", 0)), O=float(ohlc.get("O") or ohlc.get("C", 0)),
                    pivot_type=pivot_type,
                )
                self._send_json({
                    "symbol": sym, "price": round(price, 2),
                    "live": is_live, "source": data_src, "pivots": pivots,
                    "ohlc": {k: round(float(v), 2) for k, v in ohlc.items() if k in ("H", "L", "C", "O")},
                    "prevClose": round(float(live["prevClose"]), 2) if live else None,
                    "mode": mode, "pivot_type": pivot_type,
                })
                return
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e), "symbol": sym})
            return

        if parsed.path == "/api/nifty-constituents":
            symbol = (qs.get("symbol", ["NIFTY"])[0]).upper()
            try:
                self._send_json(_fetch_nifty_constituents(symbol))
            except Exception as e:  # noqa: BLE001
                self._send_json({"stocks": [], "error": str(e)})
            return

        if parsed.path == "/api/fno-movers":
            self._send_json(nse_fno_movers.get_movers())
            return

        if parsed.path == "/api/straddle-history":
            try:
                records = nse_history_store.read_history(symbol, days=1)
                straddle_pts = [{"t": r["t"], "v": r["straddle_premium"]}
                                 for r in records if r.get("straddle_premium")]
                straddle_pts = nse_history_store.downsample(straddle_pts, 300)
                self._send_json({"symbol": symbol, "points": straddle_pts})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Straddle history failed: {e}"}, status=500)
            return

        # ── static files: lets other computers load the dashboard itself from
        #    this server (http://<host>:<port>/nse_dashboard.html) instead of
        #    needing a copy of the HTML on every machine.
        if parsed.path in ("/", "/index.html", "/nse_dashboard.html"):
            fname = "nse_dashboard.html"
            fpath = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), fname)
            if _os.path.exists(fpath):
                try:
                    with open(fpath, "rb") as f:
                        body = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                except Exception as e:  # noqa: BLE001
                    self._send_json({"error": f"Could not read {fname}: {e}"}, status=500)
            else:
                self._send_json({"error": f"{fname} not found next to the server script"}, status=404)
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

        if parsed.path == "/api/backtest":
            symbol = (body.get("symbol") or "NIFTY").upper()
            try:
                self._send_json(run_backtest(symbol, body))
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"Backtest failed: {e}"}, status=500)
            return

        if parsed.path == "/api/log-alert":
            ok = log_alert_outcome(body)
            self._send_json({"logged": bool(ok)})
            return

        if parsed.path == "/api/notify":
            # Relay a dashboard alert to Telegram using existing alert_config.json.
            # Rate-limited to 1 message per 5s to avoid Telegram flood limits.
            msg = str(body.get("message", "")).strip()[:500]
            if not msg:
                self._send_json({"error": "Need message"}, status=400)
                return
            try:
                cfg = nse_alerts.load_config()
                if not (cfg.get("enabled") and cfg.get("telegram_token") and cfg.get("chat_id")):
                    self._send_json({"sent": False, "reason": "Telegram disabled — edit alert_config.json (enabled, telegram_token, chat_id)"})
                    return
                now = time.time()
                if now - getattr(self.server, "_last_notify_ts", 0) < 5:
                    self._send_json({"sent": False, "reason": "rate-limited"})
                    return
                self.server._last_notify_ts = now
                ok = nse_alerts.send_telegram_message(cfg["telegram_token"], cfg["chat_id"], f"📊 Dashboard: {msg}")
                self._send_json({"sent": bool(ok)})
            except Exception as e:  # noqa: BLE001
                self._send_json({"sent": False, "reason": str(e)})
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

        if parsed.path == "/api/ai-suggest":
            prompt = body.get("prompt", "").strip()
            if not prompt:
                self._send_json({"error": "No prompt provided"}, status=400)
                return
            try:
                import anthropic  # noqa: PLC0415
                client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._send_json({"text": msg.content[0].text})
            except ImportError:
                self._send_json({"error": "anthropic not installed — run: pip install anthropic --break-system-packages"})
            except Exception as e:  # noqa: BLE001
                err = str(e)
                if any(x in err.lower() for x in ("api_key", "authentication", "auth_")):
                    err = "ANTHROPIC_API_KEY not set. Before starting the server: export ANTHROPIC_API_KEY=sk-ant-..."
                self._send_json({"error": err})
            return

        self._send_json({"error": f"Unknown route: {parsed.path}"}, status=404)


def _fetch_fii_dii() -> dict:
    """Fetch FII/DII provisional equity activity from NSE (5-minute cache)."""
    global _fii_dii_cache, _shared_fetcher
    now = time.time()
    if _fii_dii_cache["data"] and (now - _fii_dii_cache["ts"]) < _FII_DII_TTL:
        return _fii_dii_cache["data"]
    fetcher = _shared_fetcher
    if not fetcher:
        return {"error": "Session not warmed — fetch the chain first."}
    try:
        r = fetcher.session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers={"Accept": "application/json, */*", "Referer": "https://www.nseindia.com/"},
            timeout=12,
        )
        raw = r.json()
        # NSE returns a list of dicts: date, name, buyValue, sellValue, netValue
        if not isinstance(raw, list) or not raw:
            return {"error": f"Unexpected response shape from NSE"}
        # Most-recent first; take last 5 trading days
        records = []
        for row in raw[:10]:
            records.append({
                "date":      row.get("date", ""),
                "category":  row.get("name", ""),
                "buy":       row.get("buyValue", 0),
                "sell":      row.get("sellValue", 0),
                "net":       row.get("netValue", 0),
            })
        data = {"records": records, "as_of": time.strftime("%H:%M:%S")}
        _fii_dii_cache = {"data": data, "ts": now}
        return data
    except Exception as e:  # noqa: BLE001
        stale = _fii_dii_cache.get("data")
        return stale if stale else {"error": str(e)}


def _build_health_response() -> dict:
    """Return server + session health metrics."""
    age = time.time() - _shared_fetcher_ts if _shared_fetcher_ts else None
    return {
        "status": "ok",
        "session_warmed":       _shared_fetcher is not None and getattr(_shared_fetcher, "_warmed", False),
        "session_age_seconds":  round(age) if age is not None else None,
        "session_fresh":        (age is not None and age < 1500),  # <25 min
        "bhavcopy_cached":      len(list(_BHAVCOPY_DIR.glob("*.csv"))),
        "uptime_seconds":       round(time.time() - _SERVER_START_TIME),
    }


def _start_session_rewarm_thread() -> None:
    """Background thread that re-warms the NSE session every 25 minutes.
    NSE's cookies typically expire after ~30 minutes of inactivity — re-warming
    proactively prevents the silent degradation where all API calls start 403ing
    mid-session without any obvious error until the server is restarted.
    """
    def _loop() -> None:
        while True:
            time.sleep(25 * 60)
            global _shared_fetcher, _shared_fetcher_ts
            fetcher = _shared_fetcher
            if fetcher and getattr(fetcher, "_warmed", False):
                try:
                    fetcher._warm_up()
                    _shared_fetcher_ts = time.time()
                    print("[i] Background session re-warm: OK")
                except Exception as e:  # noqa: BLE001
                    print(f"[!] Background session re-warm failed (non-fatal): {e}")
    t = threading.Thread(target=_loop, daemon=True, name="nse-session-rewarm")
    t.start()


def main():
    ap = argparse.ArgumentParser(description="Local API server for the NSE options dashboard")
    env = _os.environ.get
    def _envbool(name, dflt):
        v = env(name)
        return dflt if v is None else v.strip().lower() in ("1", "true", "yes", "on")

    dflt_lan  = _envbool("NSE_LAN", RUN_LAN)
    dflt_host = "0.0.0.0" if dflt_lan else "127.0.0.1"
    ap.add_argument("--port", type=int, default=int(env("NSE_PORT", RUN_PORT)))
    ap.add_argument("--host", default=dflt_host,
                    help="0.0.0.0 to serve every machine on your LAN")
    ap.add_argument("--lan", action="store_true",
                    help="shorthand for --host 0.0.0.0 (multi-computer access)")
    ap.add_argument("--no-lan", action="store_true",
                    help="force this machine only, overriding RUN_LAN in the file")
    ap.add_argument("--poll", default=env("NSE_POLL", RUN_POLL),
                    help="comma-separated symbols to keep warm in the background, "
                         "e.g. --poll NIFTY,BANKNIFTY. Browsers then read the shared "
                         "cache and NSE sees one request per symbol per cycle.")
    ap.add_argument("--poll-interval", type=float,
                    default=float(env("NSE_POLL_INTERVAL", RUN_POLL_INTERVAL)),
                    help="seconds between background polls per symbol")
    ap.add_argument("--ttl", type=float, default=None,
                    help="seconds a cached chain counts as fresh")
    args = ap.parse_args()
    if args.lan:
        args.host = "0.0.0.0"
    if args.no_lan:
        args.host = "127.0.0.1"
    if args.ttl is not None:
        globals()["_CHAIN_TTL"] = args.ttl

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

    # Start session re-warm thread — prevents silent cookie expiry mid-session
    _start_session_rewarm_thread()

    # Background poller: keeps hot symbols warm so every browser hits cache.
    if RUN_DATA_SOURCE == "fyers":
        # Do this before anything else touches Fyers: connecting with a stale
        # token just fails, and the fallback to NSE would be silent.
        # Wrapped: a startup convenience must never be able to stop the
        # server from starting. The first version raised NameError and took
        # the whole process down - strictly worse than the stale token it
        # was meant to catch.
        try:
            ok_tok = _ensure_fyers_token()
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] token check failed ({type(e).__name__}: {e}) - continuing")
            ok_tok = False
        if not ok_tok:
            print("[fyers] proceeding without a valid token - option-chain "
                  "requests will fall back to NSE scraping")

    if _adapter_streams() and RUN_DATA_SOURCE == "fyers":
        print(f"[i] DATA_SOURCE=fyers: broker stream is the source - "
              f"NSE polling and cookie warm-up are DISABLED")
        try:
            import nse_adapter_fyers as _fy
            tv = _fy.token_validity()
            if not tv.get("valid"):
                # Say this ONCE, clearly, at startup - rather than letting the
                # first chain request fail with an auth error the user then has
                # to interpret. A token lasts until ~06:00 IST tomorrow, so this
                # is a once-a-morning step, not a once-per-restart step.
                print(f"[fyers] TOKEN NOT USABLE: {tv.get('reason')}")
                print(f"[fyers] Run:  python3 fyers_login.py")
                print(f"[fyers] Falling back to NSE scraping until then.")
            if _fy.is_configured() and tv.get("valid"):
                _fy.set_tick_sink(_sse_broadcast_ticks)
                _fy.connect()
                for sym in [s.strip().upper() for s in
                            _os.environ.get("FYERS_SYMBOLS", "NIFTY,BANKNIFTY").split(",") if s.strip()]:
                    try:
                        _fy.fetch_chain(sym, None, 12)     # prime + subscribe
                        print(f"[fyers] {sym} chain primed and subscribed")
                    except Exception as e:  # noqa: BLE001
                        print(f"[fyers] {sym} prime failed: {e}")
            elif not _fy.is_configured():
                print("[fyers] not configured - set FYERS_CLIENT_ID in your env "
                      "file and run fyers_login.py")
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] startup failed: {e}")
    elif _adapter_streams():
        print(f"[i] DATA_SOURCE={RUN_DATA_SOURCE}: broker stream is the source - "
              f"NSE polling and cookie warm-up are DISABLED")
        try:
            import nse_adapter_arrow as _arrow
            if _arrow.is_configured():
                # ticks flow: broker WebSocket -> adapter -> SSE -> browsers.
                # _sse_broadcast already elects one speaker per machine for
                # alerts; price ticks go to every tab (no election needed).
                _arrow.set_tick_sink(_sse_broadcast_ticks)
                _arrow.connect()
                for sym in [s.strip().upper() for s in
                            _os.environ.get("ARROW_SYMBOLS", "NIFTY,BANKNIFTY").split(",") if s.strip()]:
                    try:
                        _arrow.fetch_chain(sym, None, 12)   # prime + subscribe
                        print(f"[arrow] {sym} chain primed and subscribed")
                    except Exception as e:  # noqa: BLE001
                        print(f"[arrow] {sym} prime failed: {e}")
            else:
                print("[arrow] not configured - fill ARROW_APP_ID and credentials in .env")
        except Exception as e:  # noqa: BLE001
            print(f"[arrow] startup failed: {e}")
    elif args.poll:
        _start_poller([s.strip() for s in args.poll.split(",")], args.poll_interval)

    class _QuietThreadingHTTPServer(ThreadingHTTPServer):
        daemon_threads = True
        def handle_error(self, request, client_address):
            import sys as _s
            exc = _s.exc_info()[1]
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
                return                      # routine client disconnect
            super().handle_error(request, client_address)

    if _envbool("NSE_KEEP_AWAKE", RUN_KEEP_AWAKE):
        print(f"[i] Keep-awake: {_keep_awake(True)}")
        print("[i] Note: this prevents IDLE sleep. If you close a laptop lid or "
              "choose Sleep from the menu, the machine still sleeps - "
              "connect power and set the display to sleep rather than the system.")
    if _envbool("NSE_REFRESH_WEIGHTS", RUN_REFRESH_WEIGHTS):
        print("[weights] NSE_REFRESH_WEIGHTS is on - fetching live constituent weights")
        try:
            # The refresh needs a live NSE session. Build one here rather than
            # assuming a helper exists: an undefined name in a startup path is
            # a crash, and a startup convenience must never stop the server.
            global _shared_fetcher
            if _shared_fetcher is None:
                from nse_options_strategy import NSESession
                _f = NSESession()
                if hasattr(_f, "warm_up"):
                    _f.warm_up()
                _shared_fetcher = _f
            _refresh_all_weights()
        except Exception as e:  # noqa: BLE001
            print(f"[weights] refresh failed ({type(e).__name__}: {e}) "
                  f"- using whatever is on disk")
    else:
        _load_index_weights()
    _start_janitor()
    _start_cleanup()
    if _envbool("NSE_RECONCILE", RUN_RECONCILE):
        _start_reconciler([s.strip() for s in (args.poll or "NIFTY").split(",")],
                          float(_os.environ.get("NSE_RECONCILE_INTERVAL", RUN_RECONCILE_INTERVAL)))
    server = _QuietThreadingHTTPServer((args.host, args.port), Handler)
    # Build stamp: printed so you can confirm at a glance which version is
    # actually running. Several rounds of "still not working" turned out to be
    # an older file still in place, which no amount of code fixing can cure.
    _feat = []
    for name, present in (("yahoo-backfill", "_yahoo_intraday" in globals()),
                          ("candles-debug", True),
                          ("alert-stream", "_sse_broadcast" in globals()),
                          ("keep-awake", "_keep_awake" in globals())):
        _feat.append(f"{name}:{'yes' if present else 'NO'}")
    print(f"[i] Build 2026-08-21 · {' · '.join(_feat)}")
    print(f"[i] NSE chain server running at http://{args.host}:{args.port}")
    if args.host == "0.0.0.0":
        try:
            import socket as _sock
            s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80)); lan_ip = s.getsockname()[0]; s.close()
        except Exception:
            lan_ip = "<this-machine-ip>"
        print(f"[i] LAN mode: other computers open  http://{lan_ip}:{args.port}/nse_dashboard.html")
        print(f"[i] They must set the same address in the dashboard's Server URL box.")
    print(f"[i] Shared chain cache: TTL {_CHAIN_TTL}s — simultaneous browsers are coalesced "
          f"into ONE NSE request; check /api/cache-stats")
    if len(sys.argv) == 1:
        print(f"[i] Started with no arguments (IDE Run) — using the RUN_* settings near the "
              f"top of this file: LAN={'on' if args.host == '0.0.0.0' else 'off'}, "
              f"port={args.port}, poll={args.poll or 'off'}. Edit those lines to change.")
    print(f"[i] Try it: http://{args.host}:{args.port}/api/chain?symbol=NIFTY")
    print(f"[i] Telegram alerts: {alert_status}")
    print(f"[i] .jsonl history writes disabled — prev-day OHLC always fetched live from NSE")
    print(f"[i] Bhavcopy cache: {_BHAVCOPY_DIR}")
    print("[i] Now open nse_dashboard.html in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Stopped.")
    finally:
        # release the sleep inhibitor, otherwise the machine stays awake
        # after the server has gone
        _release_awake()
        try:
            _poller_stop.set()
        except Exception:
            pass


if __name__ == "__main__":
    main()
