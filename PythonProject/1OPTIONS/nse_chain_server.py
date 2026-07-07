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
import threading
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
            if price:
                futs.append((exp, float(price)))
        if not futs:
            raise NSEFetchError("no futures rows in quote-derivative payload")
        futs.sort(key=lambda x: x[0])
        exp, price = futs[0]
        dte = max(0, (exp - _dt.now()).days)
        result = (price, dte)
        _futures_cache[sym] = (now, result)
        return result
    except Exception as e:  # noqa: BLE001 — strictly best-effort
        print(f"[!] Futures price fetch failed (non-fatal): {e}")
        return cached[1] if cached else (None, None)


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

    # Nearest-month futures — basis / cost-of-carry card (best-effort, cached 30s)
    futures_price, futures_dte = _fetch_futures_price(fetcher, symbol)

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


    if not raw_stocks:
        raise RuntimeError(f"All data sources failed for {sym}. Warm the session by loading the option chain first.")

    # ── Build result ─────────────────────────────────────────────────
    _weight_map = dict(cfg["fallback_weights"])
    stocks = [
        s for s in raw_stocks
        if s.get("symbol") not in (cfg["nse_names"] + [sym, "NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY MID SELECT"])
        and float(s.get("lastPrice") or 0) > 0
    ]
    stocks.sort(key=lambda s: float(s.get("totalTradedValue") or 0), reverse=True)

    total_tv = sum(float(s.get("totalTradedValue") or 0) for s in stocks) or 1.0
    use_fixed = total_tv < 100   # individual fetch fallback

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
                _write_replay_snapshot(symbol, data)   # session recorder (1/min)
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
                # OHLC: CSV only. The old NSE quote-equity fallback that used
                # to live here is permanently removed — that endpoint sits
                # behind a hard Akamai "Access Denied" 403 wall regardless of
                # headers/cookies/referer (confirmed: every symbol, every
                # attempt, even with a correctly-warmed session). Attempting
                # it just wastes a request that can never succeed.
                ohlc = (_ps.get_weekly_ohlc(sym) if mode == "weekly" else _ps.get_daily_ohlc(sym))
                if not ohlc:
                    self._send_json({"error": f"No OHLC data for {sym}. CSV file missing from nse_data_cache."})
                    return
                pivots = _ps.compute_pivots(
                    H=float(ohlc.get("H", 0)), L=float(ohlc.get("L", 0)),
                    C=float(ohlc.get("C", 0)), O=float(ohlc.get("O") or ohlc.get("C", 0)),
                    pivot_type=pivot_type,
                )
                # Current price
                price = 0.0
                if ftch:
                    try:
                        prices = _ps.get_live_prices([sym], ftch, h)
                        price = prices.get(sym, 0.0)
                    except Exception:
                        pass
                if not price:
                    price = float(ohlc.get("C", 0))
                self._send_json({
                    "symbol": sym, "price": round(price, 2),
                    "live": bool(ftch), "pivots": pivots,
                    "ohlc": {k: round(float(v), 2) for k, v in ohlc.items() if k in ("H","L","C","O")},
                    "mode": mode, "pivot_type": pivot_type,
                })
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)})
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

    # Start session re-warm thread — prevents silent cookie expiry mid-session
    _start_session_rewarm_thread()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[i] NSE chain server running at http://{args.host}:{args.port}")
    print(f"[i] Try it: http://{args.host}:{args.port}/api/chain?symbol=NIFTY")
    print(f"[i] Telegram alerts: {alert_status}")
    print(f"[i] .jsonl history writes disabled — prev-day OHLC always fetched live from NSE")
    print(f"[i] Bhavcopy cache: {_BHAVCOPY_DIR}")
    print("[i] Now open nse_dashboard.html in your browser. Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[i] Stopped.")


if __name__ == "__main__":
    main()
