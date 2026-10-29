"""
Zerodha Kite Connect market-data adapter for 1OPTIONS.

A drop-in alternative to nse_adapter_fyers.py: the same public functions,
the same payload shape, the same tick format pushed to the browsers - so the
server, the alert engine and the dashboard run unchanged whichever broker is
behind them. Select it with DATA_SOURCE=zerodha.

How Kite differs from Fyers, and what that changes here
------------------------------------------------------
* Kite has NO option-chain endpoint. The chain is assembled from Kite's
  instrument list (downloaded once a day, cached on disk) and priced from the
  live stream; a REST quote (up to 500 contracts per call) only fills in what
  the stream has not delivered yet. With the stream healthy a chain refresh
  costs zero API calls.
* Subscriptions cost nothing. Kite's ticker takes instrument tokens that are
  already known from the instrument list, so subscribing, re-subscribing and
  reconnecting make no REST call at all (Fyers converts every symbol over
  HTTP first). The ticker also re-subscribes by itself after a reconnect.
* FULL mode carries everything on one subscription: price, volume, OI, the
  five-level book with order counts, and total buy / sell quantity. There is
  no separate depth subscription and no depth cap.
* Kite publishes no previous-day OI, so OI CHANGE is built here: each
  contract's OI is recorded through the session and saved at the close, the
  pre-open value (which IS yesterday's close) is captured when the server is
  up before 09:15, and anything still missing is back-filled once a day from
  Kite's daily candles (which carry OI) at a slow, rate-limited pace.
* Kite reports OI and volume in units (shares). They are converted to
  CONTRACTS at this boundary, as the Fyers adapter does, so every threshold
  and formula downstream keeps its meaning. The lot comes straight from the
  instrument list - the exchange's own figure.
* Index candles from Kite carry no volume. VWAP, the volume profile and
  PVSRA need it, so the near-month future's volume is laid onto the index
  candles minute by minute (ZERODHA_INDEX_VOLUME=none turns that off).

Auth
----
Kite access tokens expire at about 06:00 IST the next day and the login is
interactive. zerodha_login.py does it once a morning and saves the token to
.zerodha_token; nothing here can renew a token on its own, and it says so
plainly when one is needed.

Rate limits respected: quote 1/s, historical 3/s, everything else 10/s.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from kiteconnect import KiteConnect, KiteTicker
    SDK_AVAILABLE = True
    SDK_ERROR = ""
except Exception as _e:  # noqa: BLE001
    KiteConnect = KiteTicker = None   # type: ignore
    SDK_AVAILABLE = False
    SDK_ERROR = str(_e)

BROKER = "zerodha"
_TAG = "[zerodha]"

# index -> (index exchange, candidate index tradingsymbols, option exchange, option name)
_INDEX_SPEC = {
    "NIFTY": ("NSE", ("NIFTY 50",), "NFO", "NIFTY"),
    "BANKNIFTY": ("NSE", ("NIFTY BANK",), "NFO", "BANKNIFTY"),
    "FINNIFTY": ("NSE", ("NIFTY FIN SERVICE", "FINNIFTY"), "NFO", "FINNIFTY"),
    "MIDCPNIFTY": ("NSE", ("NIFTY MID SELECT", "NIFTY MIDCAP SELECT", "MIDCPNIFTY"), "NFO", "MIDCPNIFTY"),
    "SENSEX": ("BSE", ("SENSEX",), "BFO", "SENSEX"),
}
_VIX_CANDIDATES = ("INDIA VIX",)
# Last resort only, used when the instrument list cannot be read at all.
_KNOWN_INDEX_TOKENS = {"NSE:NIFTY 50": 256265, "NSE:NIFTY BANK": 260105, "NSE:INDIA VIX": 264969}

# ── session bounds, shared with the server ────────────────────────────
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_END_MIN = int(os.environ.get("NSE_SESSION_END_MIN", 15 * 60 + 40))
CAS_CASH_CUTOFF_MIN = 15 * 60 + 15


def _in_session(mins: int) -> bool:
    return SESSION_OPEN_MIN <= mins < SESSION_END_MIN


def market_open_now() -> bool:
    g = time.gmtime(time.time() + 19800)
    if g.tm_wday >= 5:
        return False
    return _in_session(g.tm_hour * 60 + g.tm_min)


def _ist_market_open(now: Optional[float] = None) -> bool:
    t = time.gmtime((now or time.time()) + 19800)
    m = t.tm_hour * 60 + t.tm_min
    return t.tm_wday < 5 and 555 <= m <= 930


def _ist_today(offset_days: int = 0) -> str:
    g = time.gmtime(time.time() + 19800 - offset_days * 86400)
    return f"{g.tm_year:04d}-{g.tm_mon:02d}-{g.tm_mday:02d}"


def _ist_min_now() -> int:
    g = time.gmtime(time.time() + 19800)
    return g.tm_hour * 60 + g.tm_min


# ── state ─────────────────────────────────────────────────────────────
_kite: Optional[Any] = None           # KiteConnect (REST)
_kws: Optional[Any] = None            # KiteTicker (stream)
_socket = None                        # alias the server's generic code may test
_lock = threading.Lock()
_sub_lock = threading.Lock()
_raw_last: Dict[str, Dict[str, Any]] = {}
_raw_rest: Dict[str, Any] = {}
_reconnects = {"n": 0, "last": 0.0}
_TICK_FRESH_S = float(os.environ.get("ZERODHA_TICK_FRESH_S", os.environ.get("FYERS_TICK_FRESH_S", "15")))
_FUT_REST_TTL = float(os.environ.get("ZERODHA_FUT_REST_TTL", "60"))
_CHAIN_REST_TTL = float(os.environ.get("ZERODHA_CHAIN_REST_TTL", "15"))
_MAX_TOKENS = int(os.environ.get("ZERODHA_MAX_TOKENS", "2900"))     # Kite: 3000 per connection
_ticks: Dict[str, Dict[str, Any]] = {}      # key "NFO:NIFTY26O2025000CE" -> latest tick
_depth: Dict[str, Dict[str, Any]] = {}
_depth_subs: set = set()                    # full-mode (book-carrying) subscriptions
_subscribed: set = set()
_tick_sink = None
_last_push = 0.0
_dirty: Dict[str, Dict[str, Any]] = {}
_qcache: Dict[str, Tuple[float, Dict[str, Any]]] = {}    # key -> (ts, normalised REST quote)
_flow: Dict[str, Dict[str, Any]] = {}
_status: Dict[str, Any] = {
    "connected": False, "streaming": False, "last_error": "",
    "ticks": 0, "subscribed": 0, "last_chain": 0.0, "token_age_h": None,
}

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zerodha_cache")


# ── configuration ─────────────────────────────────────────────────────
_ENV_NAMES = ("env", ".env", ".env.local", "env.txt", ".env.txt")
_env_cache: Optional[Dict[str, str]] = None


def _env_file_values() -> Dict[str, str]:
    global _env_cache
    if _env_cache is not None:
        return _env_cache
    _env_cache = {}
    here = os.path.dirname(os.path.abspath(__file__))
    override = os.environ.get("NSE_ENV_FILE")
    paths = [override] if override else [os.path.join(here, n) for n in _ENV_NAMES]
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        try:
            for line in open(p):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                _env_cache[k.strip()] = v.strip().strip('"').strip("'")
        except Exception:
            pass
        break
    return _env_cache


def _cfg(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    if v:
        return v.strip()
    return (_env_file_values().get(key) or default).strip()


def _token_path() -> str:
    return _cfg("ZERODHA_TOKEN_FILE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".zerodha_token")


def _access_token() -> str:
    tok = _cfg("ZERODHA_ACCESS_TOKEN")
    if tok:
        return tok
    path = _token_path()
    try:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            ts = data.get("ts", 0)
            _status["token_age_h"] = round((time.time() - ts) / 3600, 1) if ts else None
            return (data.get("access_token") or "").strip()
    except Exception:
        pass
    return ""


def is_configured() -> bool:
    return bool(_cfg("ZERODHA_API_KEY") and _access_token())


def token_validity() -> Dict[str, Any]:
    """Kite tokens expire at about 06:00 IST the day after they are issued."""
    path = _token_path()
    if _cfg("ZERODHA_ACCESS_TOKEN") and not os.path.exists(path):
        return {"present": True, "valid": True, "reason": "token set in the env file (age unknown)",
                "expires_in_h": None}
    if not os.path.exists(path):
        return {"present": False, "valid": False, "reason": "no saved token - run zerodha_login.py"}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"present": True, "valid": False, "reason": f"token file unreadable: {e}"}
    ts = data.get("ts", 0)
    if not ts:
        return {"present": True, "valid": True, "reason": "no timestamp recorded", "expires_in_h": None}
    issued = time.gmtime(ts + 19800)
    day_start = ts - (issued.tm_hour * 3600 + issued.tm_min * 60 + issued.tm_sec)
    expiry = day_start + 30 * 3600 if issued.tm_hour >= 6 else day_start + 6 * 3600
    left = (expiry - time.time()) / 3600
    return {"present": True, "valid": left > 0, "age_h": round((time.time() - ts) / 3600, 1),
            "expires_in_h": round(left, 1),
            "reason": "valid" if left > 0 else "expired - run zerodha_login.py to refresh"}


def status() -> Dict[str, Any]:
    with _lock:
        s = dict(_status)
    s["broker"] = BROKER
    s["sdk"] = SDK_AVAILABLE
    s["configured"] = is_configured()
    s["token"] = token_validity()
    s["login_script"] = "zerodha_login.py"
    if not SDK_AVAILABLE:
        s["last_error"] = s["last_error"] or f"kiteconnect not installed: {SDK_ERROR} - pip install kiteconnect"
    elif not is_configured():
        s["last_error"] = ("ZERODHA_API_KEY or access token missing - run zerodha_login.py "
                           "(tokens expire daily at about 06:00 IST)")
    return s


def _set_err(msg: str) -> None:
    with _lock:
        _status["last_error"] = msg
        _status["connected"] = False
    print(f"{_TAG} {msg}")


# ── REST: rate limits and usage ───────────────────────────────────────
# Kite: quote 1 request/s, historical 3/s, the rest 10/s. Each call waits its
# turn here instead of drawing a 429 that would then have to be retried.
_rl = {"quote": [0.0, 1.05], "historical": [0.0, 0.36], "other": [0.0, 0.12]}
_rl_lock = threading.Lock()
_rest_count: Dict[str, Any] = {"n": 0, "day": "", "by": {}}
_rest_errors: List[Dict[str, Any]] = []


def _throttle(kind: str) -> None:
    slot = _rl.get(kind) or _rl["other"]
    with _rl_lock:
        now = time.time()
        start = max(now, slot[0] + slot[1])
        slot[0] = start
    if start > now:
        time.sleep(start - now)


def _count_rest(kind: str) -> None:
    today = _ist_today()
    with _lock:
        if _rest_count["day"] != today:
            _rest_count.update({"n": 0, "day": today, "by": {}})
        _rest_count["n"] += 1
        by = _rest_count.setdefault("by", {})
        by[kind] = by.get(kind, 0) + 1
        try:
            _rest_attrib()
        except Exception:
            pass


def rest_usage() -> Dict[str, Any]:
    with _lock:
        out = dict(_rest_count)
        out["by"] = dict(_rest_count.get("by", {}))
        return out


def _rest(kind: str, fn, *args, **kwargs):
    """One REST call: throttled, counted, and an expired token reported as such."""
    _throttle("quote" if kind == "quote" else "historical" if kind.startswith("hist") else "other")
    _count_rest(kind)
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        _rest_errors.append({"ts": time.time(), "kind": kind, "err": f"{type(e).__name__}: {str(e)[:140]}"})
        del _rest_errors[:-30]
        if type(e).__name__ == "TokenException":
            _set_err(f"token rejected ({e}) - run zerodha_login.py")
            _conn.update(fail_ts=time.time(), fail_tok=_conn.get("tok") or "")
        raise


# ── instruments: the chain's skeleton, once a day ────────────────────
_inst: Dict[str, Any] = {"day": {}, "rows": {}, "fetched": {}, "busy": threading.Lock()}
# Kite publishes the day's instrument list at about 08:30 IST. A list taken
# before that is YESTERDAY's - a weekly series listed this morning would be
# missing from it all day - so one taken before 08:45 is provisional and is
# fetched again, once, after 08:45.
_INST_READY_MIN = 8 * 60 + 45


def _fetched_early(ts: float) -> bool:
    g = time.gmtime(ts + 19800)
    return g.tm_hour * 60 + g.tm_min < _INST_READY_MIN
_key2tok: Dict[str, int] = {}
_tok2key: Dict[int, str] = {}
_kind: Dict[str, str] = {}            # key -> IDX / EQ / OPT / FUT
_lot_of: Dict[str, int] = {}          # key -> lot size (derivatives)
_meta_of: Dict[str, Dict[str, Any]] = {}   # option key -> {under, typ, strike, expiry, lot, ts(tradingsymbol)}
_opts: Dict[Tuple[str, str], Dict[str, Dict[float, Dict[str, str]]]] = {}   # (exch, name) -> expiry -> strike -> {CE, PE}
_futs: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}                     # (exch, name) -> [(expiry, key)]
_idx_key: Dict[str, str] = {}         # NIFTY -> "NSE:NIFTY 50"
_eq_key: Dict[str, str] = {}          # RELIANCE -> "NSE:RELIANCE"
_WANT_NAMES = {spec[3] for spec in _INDEX_SPEC.values()}


def _compact(row: Dict[str, Any]) -> Dict[str, Any]:
    e = row.get("expiry")
    return {"t": int(row.get("instrument_token") or 0), "s": row.get("tradingsymbol") or "",
            "n": row.get("name") or "", "e": e.isoformat() if hasattr(e, "isoformat") else (e or ""),
            "k": float(row.get("strike") or 0), "lot": int(row.get("lot_size") or 0),
            "it": row.get("instrument_type") or "", "seg": row.get("segment") or "",
            "x": row.get("exchange") or ""}


def _keep(exch: str, r: Dict[str, Any]) -> bool:
    if exch in ("NSE", "BSE"):
        return r["seg"] == "INDICES" or (exch == "NSE" and r["it"] == "EQ")
    return r["n"] in _WANT_NAMES and r["it"] in ("CE", "PE", "FUT")


def _load_instruments(exch: str) -> List[Dict[str, Any]]:
    """Kite's instrument list for one exchange, cached on disk for the day.

    The NFO list is tens of thousands of rows; only the index derivatives
    this dashboard trades are kept, so the cache is small and loads instantly
    on a restart (no API call until tomorrow)."""
    today = _ist_today()
    path = os.path.join(_CACHE_DIR, f"instruments_{exch}_{today}.json")
    try:
        if os.path.exists(path):
            with open(path) as f:
                j = json.load(f)
            rows_, fetched = (j, os.path.getmtime(path)) if isinstance(j, list) else (j.get("rows") or [], j.get("fetched") or 0)
            if not (_fetched_early(fetched) and _ist_min_now() >= _INST_READY_MIN):
                _inst["fetched"][exch] = fetched
                return rows_
    except Exception:
        pass
    if not _kite:
        return []
    raw = _rest("instruments", _kite.instruments, exch)
    rows = [r for r in (_compact(x) for x in (raw or [])) if _keep(exch, r)]
    if rows:
        _inst["fetched"][exch] = time.time()
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"fetched": time.time(), "rows": rows}, f, separators=(",", ":"))
            os.replace(tmp, path)
            for old in os.listdir(_CACHE_DIR):          # yesterday's lists are dead weight
                if old.startswith(f"instruments_{exch}_") and old != os.path.basename(path):
                    try:
                        os.remove(os.path.join(_CACHE_DIR, old))
                    except Exception:
                        pass
        except Exception:
            pass
        print(f"{_TAG} instrument list {exch}: {len(rows)} relevant of {len(raw or [])} (cached for the day)")
    return rows


def _index_rows(rows: List[Dict[str, Any]]) -> None:
    for r in rows:
        key = f"{r['x']}:{r['s']}"
        _key2tok[key] = r["t"]
        _tok2key[r["t"]] = key
        if r["seg"] == "INDICES":
            _kind[key] = "IDX"
        elif r["it"] == "EQ":
            _kind[key] = "EQ"
            _eq_key[r["s"]] = key
        elif r["it"] in ("CE", "PE"):
            _kind[key] = "OPT"
            _lot_of[key] = r["lot"]
            _meta_of[key] = {"under": r["n"], "typ": r["it"], "strike": r["k"], "expiry": r["e"],
                             "lot": r["lot"], "ts": r["s"], "exch": r["x"]}
            node = _opts.setdefault((r["x"], r["n"]), {}).setdefault(r["e"], {}).setdefault(r["k"], {})
            node[r["it"]] = key
        elif r["it"] == "FUT":
            _kind[key] = "FUT"
            _lot_of[key] = r["lot"]
            lst = _futs.setdefault((r["x"], r["n"]), [])
            if (r["e"], key) not in lst:
                lst.append((r["e"], key))
                lst.sort()


def _ensure_instruments(exchanges: Tuple[str, ...]) -> None:
    today = _ist_today()
    late = _ist_min_now() >= _INST_READY_MIN
    stale = lambda x: (_inst["day"].get(x) != today or
                       (late and _fetched_early(_inst["fetched"].get(x) or time.time())))
    need = [x for x in exchanges if stale(x)]
    if not need:
        return
    with _inst["busy"]:
        for exch in need:
            if not stale(exch):
                continue
            rows = _load_instruments(exch)
            if not rows:
                continue
            if _inst["day"].get(exch):               # a new day: drop yesterday's contracts
                for k in [k for k in _opts if k[0] == exch]:
                    _opts.pop(k, None)
                for k in [k for k in _futs if k[0] == exch]:
                    _futs.pop(k, None)
            _index_rows(rows)
            _inst["day"][exch] = today
            _inst["rows"][exch] = len(rows)
    for sym, (ix, cands, _ox, _nm) in _INDEX_SPEC.items():
        if sym in _idx_key:
            continue
        over = _cfg(f"ZERODHA_INDEX_{sym}")
        for c in ((over,) if over else ()) + tuple(f"{ix}:{c}" for c in cands):
            if c in _key2tok:
                _idx_key[sym] = c
                break
    if "VIX" not in _idx_key:
        for c in _VIX_CANDIDATES:
            if f"NSE:{c}" in _key2tok:
                _idx_key["VIX"] = f"NSE:{c}"
    for key, tok in _KNOWN_INDEX_TOKENS.items():         # instrument list unreadable
        if key not in _key2tok:
            _key2tok[key] = tok
            _tok2key[tok] = key
            _kind[key] = "IDX"
    _idx_key.setdefault("NIFTY", "NSE:NIFTY 50")
    _idx_key.setdefault("BANKNIFTY", "NSE:NIFTY BANK")
    _idx_key.setdefault("VIX", "NSE:INDIA VIX")


def _exchanges_for(sym: str) -> Tuple[str, ...]:
    spec = _INDEX_SPEC.get(sym)
    if not spec:
        return ("NSE", "NFO")
    return tuple(dict.fromkeys(("NSE", spec[0], spec[2])))


# ── connection ────────────────────────────────────────────────────────
_conn = {"fail_ts": 0.0, "fail_tok": "", "tok": ""}


def _restart_stream() -> None:
    """A new token (zerodha_login.py was run again): retire the old stream
    and open one with the new token - no server restart needed."""
    global _kws, _socket
    old = _kws
    _kws = _socket = None
    if old is not None:
        try:
            _ws_call(old.close)
        except Exception:
            pass
    _start_socket()


def connect() -> bool:
    global _kite
    if not SDK_AVAILABLE:
        _set_err(f"kiteconnect not installed: {SDK_ERROR} - pip install kiteconnect")
        return False
    api_key, token = _cfg("ZERODHA_API_KEY"), _access_token()
    if not api_key or not token:
        _set_err("ZERODHA_API_KEY or access token missing - run zerodha_login.py")
        return False
    tv = token_validity()
    if tv.get("present") and not tv.get("valid"):
        _set_err(f"saved token {tv.get('reason')} (issued {tv.get('age_h')}h ago). Run: python3 zerodha_login.py")
        return False
    with _lock:
        if _status["connected"] and _kite is not None and token == _conn["tok"]:
            return True
    # After a failure, wait a minute before asking Kite again - otherwise a
    # bad token costs one profile call on EVERY chain request. A new token
    # (the login was re-run) is tried at once.
    if _conn["fail_ts"] and time.time() - _conn["fail_ts"] < 60 and token == _conn["fail_tok"]:
        return False
    renewed = bool(_conn["tok"]) and token != _conn["tok"]
    try:
        k = KiteConnect(api_key=api_key)
        k.set_access_token(token)
        _kite = k
        prof = _rest("profile", k.profile)
        name = (prof or {}).get("user_name") or (prof or {}).get("user_shortname") or ""
        with _lock:
            _status.update(connected=True, last_error="", token_rejected=False)
        _conn.update(fail_ts=0.0, fail_tok="", tok=token)
        print(f"{_TAG} authenticated{f' as {name}' if name else ''}"
              + (f" - token valid for another {tv['expires_in_h']}h" if tv.get("expires_in_h") is not None else ""))
        _ensure_instruments(("NSE", "NFO"))
        if renewed and _kws is not None:
            print(f"{_TAG} new token - reopening the market-data stream with it")
            _restart_stream()
        else:
            _start_socket()
        _start_oi_keeper()
        return True
    except Exception as e:  # noqa: BLE001
        _kite = None
        _conn.update(fail_ts=time.time(), fail_tok=token)
        _set_err(f"connect failed: {type(e).__name__} {e}. Tokens expire daily - run zerodha_login.py")
        return False


# ── live stream (KiteTicker, FULL mode) ──────────────────────────────
def set_tick_sink(fn) -> None:
    global _tick_sink
    _tick_sink = fn


def _push_ticks() -> None:
    global _last_push
    if _tick_sink is None:
        return
    hz = float(os.environ.get("ZERODHA_PUSH_HZ", os.environ.get("FYERS_PUSH_HZ", "4")))
    now = time.time()
    if now - _last_push < (1.0 / max(1.0, hz)):
        return
    with _lock:
        if not _dirty:
            return
        batch = dict(_dirty)
        _dirty.clear()
    _last_push = now
    try:
        _tick_sink({"type": "ticks", "ts": now, "n": len(batch), "t": batch})
    except Exception:
        pass


def _jsonable(t: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in t.items():
        if hasattr(v, "isoformat"):
            v = v.isoformat()
        out[k] = v
    return out


def _levels(rows) -> List[Dict[str, Any]]:
    out = []
    for r in (rows or [])[:5]:
        try:
            p = float(r.get("price") or 0)
        except Exception:
            p = 0.0
        if p > 0:
            out.append({"p": p, "q": float(r.get("quantity") or 0), "n": int(r.get("orders") or 0)})
    return out


def _norm(t: Dict[str, Any]) -> Dict[str, Any]:
    """A Kite tick OR a Kite REST quote, in the adapter's common fields."""
    d: Dict[str, Any] = {}
    if t.get("last_price") is not None:
        d["ltp"] = float(t["last_price"])
    vol = t.get("volume_traded", t.get("volume"))
    if vol is not None:
        d["volume"] = float(vol)
    if t.get("oi") is not None:
        d["oi"] = float(t["oi"])
    o = t.get("ohlc") or {}
    for a, b in (("open", "open"), ("high", "high"), ("low", "low"), ("close", "prev_close")):
        if o.get(a):
            d[b] = float(o[a])
    dp = t.get("depth") or {}
    b5, a5 = _levels(dp.get("buy")), _levels(dp.get("sell"))
    if b5:
        d["bid"], d["bid_qty"] = b5[0]["p"], b5[0]["q"]
    if a5:
        d["ask"], d["ask_qty"] = a5[0]["p"], a5[0]["q"]
    if b5 or a5:
        d["_b5"], d["_a5"] = b5, a5
    tb = t.get("total_buy_quantity", t.get("buy_quantity"))
    ts_ = t.get("total_sell_quantity", t.get("sell_quantity"))
    if tb is not None or ts_ is not None:
        d["tbq"], d["tsq"] = float(tb or 0), float(ts_ or 0)
    return d


# ── tick tape: every change to an option's (or index's) price, book top or
# volume, kept in memory for the Scalping tab's candles, tape and order flow.
# Rows are (seq, epoch_ms, ltp, bid, ask, bid_qty, ask_qty, volume); seq counts
# up per instrument, so a reader asks for "everything after seq N" and never
# misses or repeats a row. No broker call is involved - it is the tick stream.
import collections as _col_tape
_tape: Dict[str, Any] = {}
_TAPE_MAX = 6000


def _tape_add(key: str, now: float, tk: Dict[str, Any]) -> None:
    dq = _tape.get(key)
    if dq is None:
        dq = _tape[key] = _col_tape.deque(maxlen=_TAPE_MAX)
    body = (tk.get("ltp"), tk.get("bid"), tk.get("ask"), tk.get("bid_qty"), tk.get("ask_qty"), tk.get("volume"))
    if dq and dq[-1][2:] == body:
        return
    dq.append((dq[-1][0] + 1 if dq else 1, int(now * 1000)) + body)


def tape_since(key: str, after: int = 0, limit: int = 1500) -> Dict[str, Any]:
    """Rows newer than `after` (at most `limit`, the newest ones). `reset` is
    true when the caller's cursor is ahead of ours (the server restarted)."""
    key = key if ":" in key else "NFO:" + key
    with _lock:
        dq = _tape.get(key)
        if not dq:
            return {"cur": 0, "rows": [], "reset": bool(after)}
        last = dq[-1][0]
        reset = after > last
        if reset:
            after = 0
        out = []
        for r in reversed(dq):
            if r[0] <= after or len(out) >= limit:
                break
            out.append(r)
    out.reverse()
    return {"cur": last, "rows": [list(r) for r in out], "reset": reset}


def _on_ticks(_ws, ticks) -> None:
    now = time.time()
    push = False
    # the freshest exchange stamp in the batch measures the feed's delay
    # (the ticker builds it with fromtimestamp, so .timestamp() is the epoch)
    try:
        ex = max((t.get("exchange_timestamp").timestamp() for t in (ticks or [])
                  if hasattr(t.get("exchange_timestamp"), "timestamp")), default=0)
        if ex:
            _lat_note(ex, now)
    except Exception:
        pass
    for t in (ticks or []):
        try:
            key = _tok2key.get(int(t.get("instrument_token") or 0))
            if not key:
                continue
            try:
                _raw_last.setdefault(key, {})[str(t.get("mode") or "tick")] = (now, _jsonable(t))
            except Exception:
                pass
            d = _norm(t)
            if "ltp" not in d:
                continue
            b5, a5 = d.pop("_b5", None), d.pop("_a5", None)
            tbq, tsq = d.pop("tbq", None), d.pop("tsq", None)
            d["ts"] = now
            kind = _kind.get(key, "")
            with _lock:
                if b5 is not None or a5 is not None or tbq is not None:
                    prev = _depth.get(key) or {}
                    _depth[key] = {
                        "bids": b5 if b5 is not None else prev.get("bids") or [],
                        "asks": a5 if a5 is not None else prev.get("asks") or [],
                        "ts": now, "lts": now if (b5 or a5) else prev.get("lts"),
                        "tbq": tbq if tbq is not None else prev.get("tbq", 0.0),
                        "tsq": tsq if tsq is not None else prev.get("tsq", 0.0),
                        "tts": now if tbq is not None else prev.get("tts"),
                    }
                _ticks[key] = {**_ticks.get(key, {}), **d}
                if kind in ("OPT", "IDX"):
                    _tape_add(key, now, _ticks[key])
                _status["ticks"] += 1
                _status["streaming"] = True
                if d.get("oi"):
                    _oi_seen(key, d["oi"])
                if kind != "EQ":
                    try:
                        _flow_update(key, d)
                    except Exception:
                        pass
                    _dirty[key] = {k: v for k, v in _ticks[key].items()
                                   if k in ("ltp", "oi", "volume", "bid", "ask", "bid_qty", "ask_qty")}
                    push = True
        except Exception:
            continue
    if push:
        _push_ticks()


def _ws_call(fn, *args) -> None:
    """Run a ticker call on Twisted's reactor thread (the ticker is not
    thread-safe); directly when no reactor is running (tests, shutdown)."""
    try:
        from twisted.internet import reactor
        if reactor.running:
            reactor.callFromThread(fn, *args)
            return
    except Exception:
        pass
    fn(*args)


def _ws_ready() -> bool:
    """Is the ticker's socket open right now? Before the first handshake
    completes (and between a drop and the reconnect) the ticker has no
    socket object at all, and a subscribe raises "'NoneType' object has no
    attribute 'sendMessage'"."""
    try:
        return bool(_kws is not None and getattr(_kws, "ws", None) is not None and _kws.is_connected())
    except Exception:
        return False


_held = {"n": 0, "noted": 0.0}


def _send_subscribe(tokens: List[int], full: List[int]) -> None:
    """Send a subscription now if the stream is open; otherwise HOLD it.
    Nothing is lost by holding: every instrument is already on the
    subscription list, and the connect handler subscribes the whole list the
    moment the stream opens (the ticker itself re-subscribes after any
    reconnect)."""
    if not _kws or not tokens:
        return
    if not _ws_ready():
        _held["n"] += len(tokens)
        if time.time() - _held["noted"] > 60:
            _held["noted"] = time.time()
            _sock_note("info", f"{_held['n']} subscription(s) held until the stream connects - sent on connect")
        return

    def _do():
        if not _ws_ready():                      # dropped between the check and now
            return
        try:
            _kws.subscribe(tokens)
            if full:
                _kws.set_mode(_kws.MODE_FULL, full)
        except Exception as e:  # noqa: BLE001
            # held, not lost: the next connect subscribes the whole list
            _sock_note("info", f"subscribe deferred to the next connect ({type(e).__name__})")
    _ws_call(_do)


def _on_ws_connect(ws, _response) -> None:
    print(f"{_TAG} market-data stream connected")
    _sock_note("connect", f"connect #{_reconnects['n'] + 1}")
    _reconnects["n"] += 1
    _reconnects["last"] = time.time()
    _retry_n.clear()
    with _lock:
        _status["streaming"] = True
        keys = sorted(_subscribed)
    toks = [_key2tok[k] for k in keys if k in _key2tok]
    full = [_key2tok[k] for k in keys if k in _key2tok and _kind.get(k) != "EQ"]
    now = time.time()
    for k in keys:
        _sub_req[k] = now
    if _held["n"]:
        _sock_note("info", f"sending {len(toks)} subscription(s), including {_held['n']} held before the connect")
        _held["n"] = 0
    # on the FIRST connect the ticker has nothing to resubscribe; after a
    # reconnect it resubscribes on its own, and repeating it is harmless
    if toks:
        try:
            ws.subscribe(toks)
            if full:
                ws.set_mode(ws.MODE_FULL, full)
        except Exception as e:  # noqa: BLE001
            _sock_note("error", f"subscribe on connect failed: {e}")


def _token_rejected_by_stream(code, reason) -> bool:
    """Kite refuses the stream with HTTP 403 when the token is no longer
    valid. Retrying cannot fix that - the ticker would reconnect every minute
    for hours - so stop it and say what will."""
    txt = f"{code} {reason}"
    if "403" not in txt and "Forbidden" not in txt and "TokenException" not in txt:
        return False
    if not _status.get("token_rejected"):
        _status["token_rejected"] = True
        _set_err("the market-data stream was refused (403): the Kite token is no longer valid - "
                 "run zerodha_login.py (the server picks the new token up by itself)")
        try:
            if _kws:
                _ws_call(_kws.stop_retry)
        except Exception:
            pass
    return True


def _on_ws_close(_ws, code, reason) -> None:
    _sock_note("close", f"{code} {reason}")
    with _lock:
        _status["streaming"] = False
    _token_rejected_by_stream(code, reason)


def _on_ws_error(_ws, code, reason) -> None:
    _sock_note("error", f"{code} {reason}")
    _token_rejected_by_stream(code, reason)
    _errors.append({"ts": time.time(), "code": code, "text": str(reason)[:160]})
    del _errors[:-50]
    print(f"{_TAG} stream error: {code} {str(reason)[:160]}")


def _start_socket() -> None:
    global _kws, _socket
    if _kws is not None or not _kite:
        return
    try:
        kws = KiteTicker(_cfg("ZERODHA_API_KEY"), _access_token(), reconnect=True,
                         reconnect_max_tries=300, reconnect_max_delay=60)
        kws.on_ticks = _on_ticks
        kws.on_connect = _on_ws_connect
        kws.on_close = _on_ws_close
        kws.on_error = _on_ws_error
        kws.on_reconnect = lambda _w, n: _sock_note("info", f"reconnecting (attempt {n})")
        kws.on_noreconnect = lambda _w: _sock_note("error", "ticker gave up reconnecting")
        # first time: starts Twisted's reactor in its own thread; later (a new
        # token) the reactor is already running and the connect is scheduled
        # on it, the only thread allowed to touch the network objects
        _ws_call(kws.connect, True)
        _kws = kws
        _socket = kws
        if not _wd.get("thread"):
            _wd["thread"] = True
            threading.Thread(target=_watchdog_loop, daemon=True, name="zerodha-watchdog").start()
    except Exception as e:  # noqa: BLE001
        print(f"{_TAG} stream unavailable, REST quotes only: {e}")
        with _lock:
            _status["streaming"] = False


def _subscribe(keys: List[str]) -> None:
    """Subscribe instruments: FULL mode for indices and derivatives (price,
    OI, five-level book, totals), QUOTE mode for equities. No REST call."""
    with _sub_lock:
        new = [k for k in dict.fromkeys(keys) if k and k in _key2tok and k not in _subscribed
               and k not in _invalid_syms]
        if not new:
            return
        room = _MAX_TOKENS - len(_subscribed)
        if room <= 0:
            _sock_note("error", f"subscription limit reached ({_MAX_TOKENS}); {len(new)} not added")
            return
        new = new[:room]
        _subscribed.update(new)
        full = [k for k in new if _kind.get(k) != "EQ"]
        _depth_subs.update(k for k in full if _kind.get(k) in ("OPT", "FUT"))
        now = time.time()
        for k in new:
            _sub_req[k] = now
        with _lock:
            _status["subscribed"] = len(_subscribed)
    _send_subscribe([_key2tok[k] for k in new], [_key2tok[k] for k in full])


# ── aggressor flow (identical rule to the Fyers adapter) ─────────────
def _flow_update(sym: str, d: Dict[str, Any]) -> None:
    vol, ltp = d.get("volume"), d.get("ltp")
    if vol is None or ltp is None:
        return
    vol, ltp = float(vol), float(ltp)
    f = _flow.get(sym)
    if f is None:
        _flow[sym] = {"v": vol, "v0": vol, "ltp": ltp, "bid": d.get("bid"), "ask": d.get("ask"), "sgn": 0, "m": {}}
        return
    dv = vol - f["v"]
    if dv < 0:
        f.update({"v": vol, "v0": vol, "m": {}})
        return
    if dv > 0:
        bid, ask = f.get("bid"), f.get("ask")
        if ask and ltp >= float(ask):
            sg = 1
        elif bid and ltp <= float(bid):
            sg = -1
        elif ltp > f["ltp"]:
            sg = 1
        elif ltp < f["ltp"]:
            sg = -1
        else:
            sg = f["sgn"] or 0
        if sg:
            mnt = int(time.time() // 60)
            b = f["m"].setdefault(mnt, [0.0, 0.0])
            b[0 if sg > 0 else 1] += dv
            f["sgn"] = sg
            if len(f["m"]) > 420:
                for k in sorted(f["m"])[:-400]:
                    f["m"].pop(k, None)
    f["v"], f["ltp"] = vol, ltp
    if d.get("bid") is not None:
        f["bid"] = d.get("bid")
    if d.get("ask") is not None:
        f["ask"] = d.get("ask")


def _flow_read(sym: Optional[str]) -> Optional[Dict[str, Any]]:
    f = _flow.get(sym or "")
    if not f or not f["m"]:
        return None
    now = int(time.time() // 60)
    b5 = s5 = b15 = s15 = bd = sd = 0.0
    first = None
    for mnt, (b, s_) in list(f["m"].items()):
        first = mnt if first is None else min(first, mnt)
        bd += b; sd += s_
        if now - mnt < 15:
            b15 += b; s15 += s_
        if now - mnt < 5:
            b5 += b; s5 += s_
    traded = (f.get("v") or 0) - (f.get("v0") or 0)
    return {"b5": b5, "s5": s5, "b15": b15, "s15": s15, "bd": bd, "sd": sd,
            "since": first * 60, "coverage": round((bd + sd) / traded, 3) if traded > 0 else None}


# ── stream watchdog ───────────────────────────────────────────────────
# Kite answers a subscription with the instrument's first tick, so a
# subscription is confirmed by a tick arriving after it was requested. Every
# remedy here is free - re-subscribing and reconnecting make no REST call -
# but each is still spaced out, so a broker-side problem is not answered by
# hammering the connection.
_sub_req: Dict[str, float] = {}
_retry_n: Dict[str, int] = {}
_invalid_syms: set = set()
_sock_log: List[Any] = []
_errors: List[Dict[str, Any]] = []
_wd: Dict[str, Any] = {"retried": 0, "heals": 0, "forced_reconnects": 0, "last_heal": 0.0,
                       "last_force": 0.0, "last_pass": 0.0, "force_streak": 0}


def _sock_note(kind: str, text: Any) -> None:
    try:
        _sock_log.append((time.time(), kind, str(text)[:240]))
        del _sock_log[:-60]
    except Exception:
        pass


def _last_tick(keys) -> float:
    best = 0.0
    for k in keys:
        ts = (_ticks.get(k) or {}).get("ts") or 0.0
        if ts > best:
            best = ts
    return best


def _force_gap() -> float:
    return 300.0 * (2 ** min(3, _wd["force_streak"]))      # 5, 10, 20, 40 min


def _force_reconnect(why: str) -> bool:
    if not _kws:
        return False
    _sock_note("heal", f"reopening the stream: {why}")
    print(f"{_TAG} watchdog: reopening the market-data stream - {why}")
    _wd["forced_reconnects"] += 1
    _wd["force_streak"] += 1
    _wd["last_force"] = time.time()
    try:
        # _close, not close(): close() also stops the ticker's own retry loop.
        # The ticker reconnects and re-subscribes everything by itself.
        _ws_call(_kws._close, None, None)
        return True
    except Exception as e:  # noqa: BLE001
        _sock_note("error", f"close failed: {e}")
        return False


def _resubscribe(keys: List[str]) -> None:
    toks = [_key2tok[k] for k in keys if k in _key2tok]
    full = [_key2tok[k] for k in keys if k in _key2tok and _kind.get(k) != "EQ"]
    now = time.time()
    for k in keys:
        _sub_req[k] = now
    _send_subscribe(toks, full)


def _watchdog_pass(now: Optional[float] = None) -> Dict[str, Any]:
    now = now or time.time()
    _wd["last_pass"] = now
    acts: Dict[str, Any] = {}
    if not _kws:
        return acts
    with _lock:
        subs = [k for k in _subscribed if k in _sub_req]
    # 1. requested, never answered: ask again, backing off; after four tries
    #    the contract is treated as dead (an expired series, a bad token)
    lost = []
    for k in subs:
        t0 = _sub_req.get(k, now)
        wait = min(600, 25 * (2 ** _retry_n.get(k, 0)))
        if now - t0 > wait and ((_ticks.get(k) or {}).get("ts") or 0) < t0 - 1:
            if _retry_n.get(k, 0) >= 4:
                continue        # a strike nobody trades: stop asking, it is not broken
            lost.append(k)
    if lost and _ist_market_open(now):
        for k in lost:
            _retry_n[k] = _retry_n.get(k, 0) + 1
        _sock_note("heal", f"re-requesting {len(lost)} subscription(s) never answered: {', '.join(lost[:5])}")
        _resubscribe(lost)
        _wd["retried"] += len(lost)
        acts["retried"] = len(lost)
    if not _ist_market_open(now):
        return acts
    traded = [k for k in subs if _kind.get(k) in ("OPT", "FUT")]
    last_any = _last_tick(traded or subs)
    if now - last_any < 30:
        _wd["force_streak"] = 0
    if last_any and now - last_any > 90:
        if now - _wd["last_force"] > _force_gap():
            acts["forced"] = _force_reconnect(f"no ticks for {int(now - last_any)}s")
        return acts
    idx = [k for k in subs if _kind.get(k) == "IDX" and "VIX" not in k]
    if idx and now - min(_sub_req.get(k, now) for k in idx) > 60:
        li = _last_tick(idx)
        if now - li > 90:
            if now - _wd["last_heal"] > 600:
                _wd["heals"] += 1
                _wd["last_heal"] = now
                _sock_note("heal", f"index silent {int(now - li)}s while contracts tick - re-subscribing it")
                _resubscribe(idx)
                acts["resubscribed"] = len(idx)
            elif now - _wd["last_heal"] > 150 and now - _wd["last_force"] > _force_gap():
                acts["forced"] = _force_reconnect(f"index still silent {int(now - li)}s after re-subscribing")
    return acts


def _watchdog_loop() -> None:
    while True:
        time.sleep(15)
        try:
            _watchdog_pass()
        except Exception as e:  # noqa: BLE001
            print(f"{_TAG} watchdog error: {e}")


def refusals_recent(window_s: float = 600.0, now: Optional[float] = None) -> Dict[str, Any]:
    """Stream errors in the window, in the shape the feed check reads."""
    now = now or time.time()
    r = [x for x in _errors if now - x["ts"] < window_s]
    return {"px": 0, "dp": 0, "auth": sum(1 for x in r if "token" in x["text"].lower() or x.get("code") == 403),
            "total": len(r), "last_age_s": round(now - r[-1]["ts"], 1) if r else None,
            "last": ({"kind": "stream", "n": 0, "sample": [], "code": r[-1].get("code"), "text": r[-1]["text"]}
                     if r else None)}


def watchdog_state() -> Dict[str, Any]:
    now = time.time()
    with _lock:
        px = [k for k in _subscribed if k in _sub_req]
    dp = [k for k in px if k in _depth_subs]
    unconf = [k for k in px if now - _sub_req.get(k, now) > 25
              and ((_ticks.get(k) or {}).get("ts") or 0) < _sub_req.get(k, now) - 1]
    unconf_dp = [k for k in unconf if k in _depth_subs]
    age = lambda t: round(now - t, 1) if t else None
    return {
        "price_requested": len(px), "price_unconfirmed": len(unconf), "price_unconfirmed_sample": unconf[:8],
        "depth_requested": len(dp), "depth_unconfirmed": len(unconf_dp), "depth_unconfirmed_sample": unconf_dp[:8],
        "last_sf_age_s": age(_last_tick(px)),
        "last_index_age_s": age(_last_tick([k for k in px if _kind.get(k) == "IDX"])),
        "last_depth_age_s": age(max(((_depth.get(k) or {}).get("lts") or 0) for k in dp) if dp else 0),
        "invalid_symbols": sorted(_invalid_syms)[:20],
        "retried": _wd["retried"], "heals": _wd["heals"], "forced_reconnects": _wd["forced_reconnects"],
        "next_reopen_gap_s": int(_force_gap()),
        "refusals_10m": refusals_recent(600, now),
        "socket_lock_blocked_by": None,
        "watchdog_pass_age_s": age(_wd["last_pass"]),
        "log": [{"age_s": round(now - t, 1), "kind": k, "text": x} for t, k, x in _sock_log[-15:]][::-1],
    }


# ── OI change: Kite publishes no previous-day OI, so it is kept here ──
_oi_today: Dict[str, float] = {}
_oi_prev: Dict[str, Any] = {"day": None, "oi": {}, "complete": False, "loaded_for": ""}
_oi_dirty = {"v": False, "busy": False}
_oi_backfill_q: List[str] = []
_oi_tried: set = set()
_oi_preopen: Dict[str, Any] = {"day": "", "oi": {}}     # captured before 09:15 today


def _oi_seen(key: str, oi: float) -> None:
    """Called under _lock from the tick path: cheap, no I/O."""
    if _oi_today.get(key) != oi:
        _oi_today[key] = oi
        _oi_dirty["v"] = True
    # Before 09:15 nothing has traded, so the OI on the screen IS yesterday's
    # closing OI - the exact figure the change is measured from.
    if _ist_min_now() < SESSION_OPEN_MIN and time.gmtime(time.time() + 19800).tm_wday < 5:
        day = _ist_today()
        if _oi_preopen["day"] != day:
            _oi_preopen.update(day=day, oi={})
        _oi_preopen["oi"].setdefault(key, oi)


def _oi_load_prev() -> None:
    today = _ist_today()
    if _oi_prev["loaded_for"] == today:
        return
    _oi_prev.update({"day": None, "oi": {}, "complete": False, "loaded_for": today})
    _oi_tried.clear()                                    # a new day: every contract may be tried again
    try:
        files = sorted(f for f in os.listdir(_CACHE_DIR) if f.startswith("oi_close_") and f.endswith(".json"))
    except Exception:
        files = []
    prev = [f for f in files if f[9:19] < today]
    if prev:
        try:
            with open(os.path.join(_CACHE_DIR, prev[-1])) as f:
                j = json.load(f)
            _oi_prev.update({"day": prev[-1][9:19], "oi": dict(j.get("oi") or {}),
                             "complete": bool(j.get("complete"))})
        except Exception:
            pass
    for old in files[:-5]:                               # keep a week of closes
        try:
            os.remove(os.path.join(_CACHE_DIR, old))
        except Exception:
            pass


def _oi_save_today() -> None:
    if not _oi_dirty["v"]:
        return
    with _lock:
        snap = dict(_oi_today)
        _oi_dirty["v"] = False
    g = time.gmtime(time.time() + 19800)
    if g.tm_wday >= 5:
        return
    complete = _ist_min_now() >= 15 * 60 + 30
    path = os.path.join(_CACHE_DIR, f"oi_close_{_ist_today()}.json")
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        old = {}
        if os.path.exists(path):
            with open(path) as f:
                old = (json.load(f) or {}).get("oi") or {}
        old.update(snap)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"oi": old, "complete": complete, "saved": time.time()}, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        pass


def _save_prev_oi() -> None:
    if not _oi_prev.get("day"):
        return
    path = os.path.join(_CACHE_DIR, f"oi_close_{_oi_prev['day']}.json")
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"oi": _oi_prev["oi"], "complete": True, "saved": time.time()}, f, separators=(",", ":"))
        os.replace(tmp, path)
    except Exception:
        pass


def _oi_backfill_one(key: str) -> None:
    """Yesterday's closing OI from Kite's daily candle (OI-flagged)."""
    tok = _key2tok.get(key)
    if not tok or not _kite:
        return
    to = _ist_today(1)
    frm = _ist_today(10)
    rows = _rest("hist-oi", _kite.historical_data, tok, f"{frm} 00:00:00", f"{to} 23:59:59", "day", False, True)
    today = _ist_today()
    best = None
    for r in rows or []:
        d = r.get("date")
        ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        if ds < today and r.get("oi") is not None:
            best = (ds, float(r["oi"]))
    if best:
        if not _oi_prev.get("day"):
            _oi_prev["day"] = best[0]
        _oi_prev["oi"][key] = best[1]


def _oi_keeper_loop() -> None:
    n_since_save = 0
    last_save = 0.0
    while True:
        # 2 calls a second while there is a queue (Kite allows 3 historical
        # calls a second, shared with the charts); idle otherwise
        time.sleep(0.5 if _oi_backfill_q else 2)
        try:
            _oi_load_prev()
            q = None
            with _lock:
                while _oi_backfill_q:
                    k = _oi_backfill_q.pop(0)
                    have = k in _oi_prev["oi"] or (_oi_preopen["day"] == _ist_today() and k in _oi_preopen["oi"])
                    if not have or not _oi_prev.get("complete"):
                        if k not in _oi_tried:
                            q = k
                            break
            if q:
                _oi_tried.add(q)
                try:
                    _oi_backfill_one(q)
                    n_since_save += 1
                except Exception as e:  # noqa: BLE001
                    if "TokenException" in type(e).__name__:
                        time.sleep(300)          # wait for a fresh login; never give up for good
                if n_since_save >= 10 or not _oi_backfill_q:
                    _save_prev_oi()
                    n_since_save = 0
            if time.time() - last_save >= 60:
                last_save = time.time()
                _oi_save_today()
        except Exception:
            pass


def _start_oi_keeper() -> None:
    if _oi_dirty["busy"]:
        return
    _oi_dirty["busy"] = True
    threading.Thread(target=_oi_keeper_loop, daemon=True, name="zerodha-oi").start()


def _prev_oi(key: str) -> Optional[float]:
    _oi_load_prev()
    # the pre-open reading is the exchange's own closing figure: it wins
    if _oi_preopen["day"] == _ist_today() and key in _oi_preopen["oi"]:
        return float(_oi_preopen["oi"][key])
    v = _oi_prev["oi"].get(key)
    return float(v) if v is not None else None


def _queue_backfill(keys: List[str]) -> None:
    if os.environ.get("ZERODHA_OI_BACKFILL", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    with _lock:
        for k in keys:
            if k not in _oi_tried and k not in _oi_backfill_q and \
                    (k not in _oi_prev["oi"] or not _oi_prev.get("complete")):
                _oi_backfill_q.append(k)


# ── prices: stream first, one batched REST quote for the rest ────────
def _tick_usable(t: Dict[str, Any]) -> bool:
    if not t or not t.get("ltp"):
        return False
    g = time.gmtime(time.time() + 19800)
    mins = g.tm_hour * 60 + g.tm_min
    if not (g.tm_wday < 5 and 555 <= mins < 930):
        return True
    return (time.time() - float(t.get("ts", 0) or 0)) < _TICK_FRESH_S


def _snap(keys: List[str], ttl: float = None) -> Dict[str, Dict[str, Any]]:
    """Current fields per instrument: the live tick when it is usable, else a
    REST quote - all the missing ones in ONE call (Kite takes 500), reused
    for `ttl` seconds."""
    ttl = _CHAIN_REST_TTL if ttl is None else ttl
    if not _ist_market_open():
        ttl = max(ttl, 300.0)            # nothing changes while the market is shut
    out: Dict[str, Dict[str, Any]] = {}
    need = []
    now = time.time()
    for k in dict.fromkeys(keys):
        if not k:
            continue
        t = _ticks.get(k) or {}
        if _tick_usable(t):
            dd = dict(t)
            bk = _depth.get(k) or {}
            if bk.get("tbq") is not None:
                dd["tbq"], dd["tsq"] = bk.get("tbq"), bk.get("tsq")
            out[k] = dd
            continue
        hit = _qcache.get(k)
        if hit and now - hit[0] < ttl:
            out[k] = hit[1]
        else:
            need.append(k)
    if need and _kite:
        for i in range(0, len(need), 500):
            batch = need[i:i + 500]
            try:
                r = _rest("quote", _kite.quote, batch) or {}
            except Exception as e:  # noqa: BLE001
                print(f"{_TAG} quote failed: {e}")
                for k in batch:                          # keep the last good answer through a blip
                    if k in _qcache:
                        out[k] = _qcache[k][1]
                continue
            for k, row in r.items():
                d = _norm(row or {})
                d.pop("_b5", None); d.pop("_a5", None)
                d["ts_rest"] = now
                _qcache[k] = (now, d)
                out[k] = d
                if d.get("oi"):
                    with _lock:
                        _oi_seen(k, d["oi"])
                kd = _kind.get(k, "")
                if kd and f"quote:{kd}" not in _raw_rest:
                    _raw_rest[f"quote:{kd}"] = _jsonable(row or {})
            if len(_qcache) > 4000:
                for k in list(_qcache)[:1000]:
                    _qcache.pop(k, None)
    return out


def _spot(symbol: str) -> float:
    key = _idx_key.get(symbol.upper())
    if not key:
        return 0.0
    return float((_snap([key], ttl=3).get(key) or {}).get("ltp") or 0.0)


def india_vix() -> Optional[float]:
    key = _idx_key.get("VIX")
    if not key:
        return None
    v = (_snap([key], ttl=_FUT_REST_TTL).get(key) or {}).get("ltp")
    return round(float(v), 2) if v else None


# ── implied volatility (same Black-76-on-the-forward convention) ─────
_IV_R = 0.065


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _b76(F: float, K: float, T: float, v: float, call: bool) -> float:
    if v <= 0 or T <= 0:
        return max(0.0, (F - K) if call else (K - F))
    sd = v * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    df = math.exp(-_IV_R * T)
    return df * (F * _ncdf(d1) - K * _ncdf(d2)) if call else df * (K * _ncdf(-d2) - F * _ncdf(-d1))


def _b76_iv(price: float, F: float, K: float, T: float, call: bool) -> Optional[float]:
    intrinsic = max(0.0, (F - K) if call else (K - F)) * math.exp(-_IV_R * T)
    if price is None or price < 1.0 or price - intrinsic < 0.5 or T <= 0:
        return None
    lo, hi = 0.005, 3.0
    if _b76(F, K, T, hi, call) < price:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _b76(F, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def _opt_px(r: Dict[str, Any], p: str) -> Optional[float]:
    b, a, l = r.get(f"{p}_bid") or 0, r.get(f"{p}_ask") or 0, r.get(f"{p}_ltp") or 0
    if b > 0 and a > 0 and a >= b and (a - b) <= max(1.0, 0.05 * a):
        return 0.5 * (a + b)
    return l or None


def _solve_chain_iv(strikes: List[Dict[str, Any]], spot: float, dte_days: float) -> str:
    if not strikes or not spot:
        return "none"
    T = max(dte_days, 1.0 / 1440) / 365.0
    near = sorted(strikes, key=lambda r: abs(r["strike"] - spot))[:3]
    fw = []
    for r in near:
        c, p_ = _opt_px(r, "ce"), _opt_px(r, "pe")
        if c and p_:
            fw.append(r["strike"] + (c - p_) * math.exp(_IV_R * T))
    F = sorted(fw)[len(fw) // 2] if fw else spot
    if not (0.9 * spot < F < 1.1 * spot):
        F = spot
    n = 0
    for r in strikes:
        K = r["strike"]
        otm_call = K >= F
        p = "ce" if otm_call else "pe"
        v = _b76_iv(_opt_px(r, p), F, K, T, otm_call)
        if v is None:
            q = "pe" if otm_call else "ce"
            v = _b76_iv(_opt_px(r, q), F, K, T, not otm_call)
        if v is not None and 0.01 < v < 2.5:
            r["ce_iv"] = r["pe_iv"] = round(v * 100.0, 2)
            n += 1
    return "solved" if n else "none"


def _classify_buildups(strikes, atm, gap, threshold_pct=8.0):
    flags = []
    for s in strikes:
        ce_oi, pe_oi = s.get("ce_oi") or 0, s.get("pe_oi") or 0
        ce_chg, pe_chg = s.get("ce_oi_chg") or 0, s.get("pe_oi_chg") or 0
        if ce_oi > 0 and abs(ce_chg) / max(ce_oi, 1) * 100 >= threshold_pct:
            flags.append({"strike": s["strike"], "side": "CE", "oi_chg": ce_chg,
                          "label": ("Fresh call writing (resistance building)" if ce_chg > 0 else
                                    "Call unwinding (resistance weakening)")})
        if pe_oi > 0 and abs(pe_chg) / max(pe_oi, 1) * 100 >= threshold_pct:
            flags.append({"strike": s["strike"], "side": "PE", "oi_chg": pe_chg,
                          "label": ("Fresh put writing (support building)" if pe_chg > 0 else
                                    "Put unwinding (support weakening)")})
    return flags


def _infer_sentiment(strikes, atm, max_pain, pcr):
    bull = bear = 0.0
    for s in strikes:
        ce_chg, pe_chg = s.get("ce_oi_chg") or 0, s.get("pe_oi_chg") or 0
        if pe_chg > 0:
            bull += 0.25
        if ce_chg > 0:
            bear += 0.25
        if ce_chg < 0:
            bull += 0.15
        if pe_chg < 0:
            bear += 0.15
    if pcr:
        if pcr > 1.2:
            bull += 1
        elif pcr < 0.8:
            bear += 1
    if max_pain and atm:
        if max_pain > atm:
            bull += 0.5
        elif max_pain < atm:
            bear += 0.5
    if bull - bear >= 1.5:
        return "Mildly bullish to bullish"
    if bear - bull >= 1.5:
        return "Mildly bearish to bearish"
    return "Range-bound / no strong directional lean"


def _max_pain(strikes: List[Dict[str, Any]]) -> Optional[float]:
    if not strikes:
        return None
    best, best_pain = None, None
    for probe in strikes:
        k = probe["strike"]
        pain = 0.0
        for s in strikes:
            if s["strike"] < k:
                pain += (k - s["strike"]) * s.get("ce_oi", 0)
            elif s["strike"] > k:
                pain += (s["strike"] - k) * s.get("pe_oi", 0)
        if best_pain is None or pain < best_pain:
            best, best_pain = k, pain
    return best


# ── expiries ──────────────────────────────────────────────────────────
def _settle_epoch(iso: str) -> float:
    """15:30 IST on an expiry date (YYYY-MM-DD)."""
    import calendar
    y, m, d = (int(x) for x in iso[:10].split("-"))
    return calendar.timegm((y, m, d, 15, 30, 0)) - 19800


def _settle_of(ep: float) -> float:
    g = time.gmtime(ep + 19800)
    day_start = ep - (g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec)
    return day_start + 15 * 3600 + 30 * 60


def _exp_str(ep: float) -> str:
    return time.strftime("%d-%b-%Y", time.gmtime(ep + 19800))


def _dte_from(ep: Optional[float]) -> float:
    if not ep:
        return 1.0
    return max((_settle_of(ep) - time.time()) / 86400.0, 1.0 / (24 * 60))


def _match_expiry(want: str, isos: List[str]) -> Optional[str]:
    if not want:
        return None
    want = str(want).strip()
    import datetime as _dx
    for iso in isos:
        ep = _settle_epoch(iso)
        if want in (iso, _exp_str(ep), str(int(ep))):
            return iso
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d %b %Y", "%d%b%Y"):
        try:
            d_ = _dx.datetime.strptime(want, fmt).date().isoformat()
        except Exception:
            continue
        if d_ in isos:
            return d_
    try:                                                   # an epoch near one of ours
        e = float(want)
        for iso in isos:
            if abs(_settle_epoch(iso) - e) < 86400:
                return iso
    except Exception:
        pass
    return None


def _monthly_dte(exp_epochs: List[float], fut_iso: Optional[str] = None) -> Optional[float]:
    if fut_iso:
        return round(max((_settle_epoch(fut_iso) - time.time()) / 86400.0, 1.0 / 1440), 4)
    if not exp_epochs:
        return None
    return round(max((_settle_of(exp_epochs[0]) - time.time()) / 86400.0, 1.0 / 1440), 4)


def _session_from_ticks(key: Optional[str], snap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    t = snap or _ticks.get(key or "") or {}
    o, h, l, c = t.get("open"), t.get("high"), t.get("low"), t.get("ltp")
    if not (o and h and l and c):
        return {}
    return {"date": _ist_today(), "open": o, "high": h, "low": l, "close": c,
            "prev_close": t.get("prev_close"), "src": "exchange"}


# ── units ─────────────────────────────────────────────────────────────
def _shares_feed(vals: List[float], lot: int) -> bool:
    """Kite counts OI and volume in units. Confirmed from the data when the
    data can say (every figure a whole number of lots), so a change on Kite's
    side shows up as a correct number rather than one off by the lot."""
    forced = _cfg("ZERODHA_QTY_UNIT").lower()
    if forced in ("shares", "units"):
        return True
    if forced == "contracts":
        return False
    v = [int(round(x)) for x in vals if x and x > 0]
    if lot <= 1 or len(v) < 8:
        return True
    return sum(1 for x in v if x % lot == 0) / len(v) >= 0.9


# ── option chain ──────────────────────────────────────────────────────
def _futures_pair(sym: str) -> List[Tuple[str, str]]:
    spec = _INDEX_SPEC.get(sym)
    if not spec:
        return []
    lst = _futs.get((spec[2], spec[3])) or []
    now = time.time()
    live = [(e, k) for e, k in lst if _settle_epoch(e) > now]
    return live[:2]


def fetch_chain(symbol: str, expiry: Optional[str], band: int) -> Dict[str, Any]:
    """Return a chain response in the server's native shape (the same keys
    as the Fyers adapter). Raises RuntimeError when unavailable so the
    caller can fall back to NSE."""
    if not connect():
        raise RuntimeError(status().get("last_error") or "zerodha unavailable")
    sym = (symbol or "NIFTY").upper()
    spec = _INDEX_SPEC.get(sym)
    if not spec:
        raise RuntimeError(f"no Zerodha mapping for {sym}")
    _ensure_instruments(_exchanges_for(sym))
    ikey = _idx_key.get(sym)
    vkey = _idx_key.get("VIX")
    chains = _opts.get((spec[2], spec[3])) or {}
    now = time.time()
    all_isos = sorted(e for e in chains if e and _settle_epoch(e) > now - 86400)
    live_isos = [e for e in all_isos if _settle_epoch(e) > now]
    if not live_isos:
        raise RuntimeError(f"no live {sym} option expiries in the instrument list")
    # an explicit expiry (even today's settled one) is served if listed;
    # the default is the first that has not settled - after 15:30 on expiry
    # day the dead weekly is rolled past, as on the Fyers path
    chosen = _match_expiry(expiry, all_isos) if expiry else None
    chosen = chosen or live_isos[0]
    futs = _futures_pair(sym)
    fut_keys = [k for _e, k in futs]

    head = _snap([ikey, vkey] + fut_keys, ttl=3)
    spot = float((head.get(ikey) or {}).get("ltp") or 0.0)
    if not spot and futs:
        spot = float((head.get(fut_keys[0]) or {}).get("ltp") or 0.0)
    ladder = chains.get(chosen) or {}
    ks = sorted(ladder)
    if not ks:
        raise RuntimeError(f"{sym} {chosen}: no strikes listed")
    if not spot:
        spot = ks[len(ks) // 2]
    count = int(_cfg("ZERODHA_STRIKE_COUNT", _cfg("FYERS_STRIKE_COUNT", "25")) or 25)
    ci = min(range(len(ks)), key=lambda i: abs(ks[i] - spot))
    window = ks[max(0, ci - count): ci + count + 1]
    opt_keys = [ladder[k][s] for k in window for s in ("CE", "PE") if ladder[k].get(s)]

    # every contract on screen, plus the index, VIX and both futures, on the
    # stream - free on Kite, so the window is subscribed whole
    _subscribe([ikey, vkey] + fut_keys + opt_keys)
    q = _snap(opt_keys)

    by_strike: Dict[float, Dict[str, Any]] = {}
    for k in window:
        rec = by_strike.setdefault(k, {"strike": k})
        for side in ("CE", "PE"):
            key = ladder[k].get(side)
            if not key:
                continue
            d = q.get(key) or {}
            p = side.lower()
            oi = float(d.get("oi") or 0)
            po = _prev_oi(key)
            rec[f"{p}_ltp"] = float(d.get("ltp") or 0)
            rec[f"{p}_oi"] = oi
            rec[f"{p}_oi_chg"] = (oi - po) if (po is not None and oi) else 0.0
            rec[f"{p}_volume"] = float(d.get("volume") or 0)
            rec[f"{p}_bid"] = float(d.get("bid") or 0)
            rec[f"{p}_ask"] = float(d.get("ask") or 0)
            rec[f"{p}_bid_qty"] = d.get("bid_qty")
            rec[f"{p}_ask_qty"] = d.get("ask_qty")
            rec[f"{p}_iv"] = 0.0
            rec[f"{p}_prev_close"] = float(d.get("prev_close") or 0)
            rec[f"{p}_symbol"] = key
    strikes = [by_strike[k] for k in sorted(by_strike)]
    if not strikes:
        raise RuntimeError("option chain returned no strikes")

    # yesterday's closing OI for the strikes that matter, back-filled slowly
    near_keys = [ladder[k][s] for k in ks[max(0, ci - 10): ci + 11] for s in ("CE", "PE") if ladder[k].get(s)]
    _queue_backfill(fut_keys + near_keys)

    # ── units: OI / OI change / volume to CONTRACTS ───────────────────
    lot = next((_lot_of[k] for k in opt_keys if _lot_of.get(k)), 0) or _LOTS.get(sym, 65)
    qty_unit = "contracts"
    vals = [r.get(f"{p}_{f}") or 0 for r in strikes for p in ("ce", "pe") for f in ("volume", "oi")]
    if lot > 1 and _shares_feed(vals, lot):
        for r in strikes:
            for p in ("ce", "pe"):
                for fld in ("oi", "oi_chg", "volume"):
                    v = r.get(f"{p}_{fld}")
                    if v:
                        r[f"{p}_{fld}"] = round(float(v) / lot, 2)
        div = lot
    else:
        div = 1
    qty_per_contract = div if div > 1 else 1

    atm = min((s["strike"] for s in strikes), key=lambda k: abs(k - spot))
    exp_epochs = [_settle_epoch(e) for e in live_isos]
    expiries = [_exp_str(x) for x in exp_epochs]
    near_epoch = _settle_epoch(chosen)
    dte_frac = _dte_from(near_epoch)
    iv_src = _solve_chain_iv(strikes, spot, dte_frac)
    atm_row = next((s for s in strikes if s["strike"] == atm), {})
    ivs = [v for v in (atm_row.get("ce_iv"), atm_row.get("pe_iv")) if v]
    atm_iv = sum(ivs) / len(ivs) if ivs else 0.0

    # ── futures (near and next month), from the same snapshot ─────────
    fut: Dict[str, Any] = {}
    if futs:
        n_key = futs[0][1]
        nd = head.get(n_key) or {}
        if nd.get("ltp"):
            po = _prev_oi(n_key)
            foi = float(nd.get("oi") or 0) or None
            fut.update({
                "futures_price": float(nd["ltp"]),
                "futures_oi": foi,
                "futures_oi_chg": (foi - po) if (foi and po is not None) else None,
                "futures_symbol": n_key,
                "futures_prev_close": nd.get("prev_close") or None,
                "futures_volume": nd.get("volume") or None,
            })
        if len(futs) > 1:
            f_key = futs[1][1]
            fd = head.get(f_key) or {}
            if fd.get("ltp"):
                fut.update({"futures_far_price": float(fd["ltp"]),
                            "futures_far_oi": float(fd.get("oi") or 0) or None,
                            "futures_far_symbol": f_key})
        if div > 1:
            for fk in ("futures_oi", "futures_oi_chg", "futures_volume", "futures_far_oi"):
                if fut.get(fk):
                    fut[fk] = round(float(fut[fk]) / div, 2)

    tot_ce = sum(s.get("ce_oi", 0) for s in strikes)
    tot_pe = sum(s.get("pe_oi", 0) for s in strikes)
    gaps = sorted({round(b - a) for a, b in zip(sorted(by_strike), sorted(by_strike)[1:]) if b > a})
    strike_gap = gaps[0] if gaps else 50
    lo_b, hi_b = atm - band * strike_gap, atm + band * strike_gap
    nearby = [s for s in strikes if lo_b <= s["strike"] <= hi_b] or strikes
    resistance = max(nearby, key=lambda s: s.get("ce_oi", 0))
    support = max(nearby, key=lambda s: s.get("pe_oi", 0))
    isnap = head.get(ikey) or {}

    def _fq(fkey, field):
        if not fkey:
            return None
        return (head.get(fkey) or {}).get(field)

    out = {
        "symbol": sym,
        "underlying_value": spot,
        "support": dict(support),
        "flags": _classify_buildups(nearby, atm, strike_gap),
        "sentiment": _infer_sentiment(nearby, atm, _max_pain(strikes), (tot_pe / tot_ce) if tot_ce else None),
        "skew_note": "",
        "expiries_data": {},
        "gainers": [], "symbols": [], "ideas": [], "strategies": [], "payout_distribution": [],
        "resistance": dict(resistance),
        "nearby": nearby,
        "atm": atm,
        "atm_iv": round(atm_iv, 2),
        "iv_source": iv_src,
        "expiry": _exp_str(near_epoch),
        "all_expiries": expiries,
        "far_expiry": expiries[1] if len(expiries) > 1 else None,
        "dte_far": round(_dte_from(exp_epochs[1]), 4) if len(exp_epochs) > 1 else None,
        "strikes": strikes,
        "pcr": round(tot_pe / tot_ce, 3) if tot_ce else None,
        "total_ce_oi": tot_ce,
        "total_pe_oi": tot_pe,
        "max_pain": _max_pain(strikes),
        "strike_gap": strike_gap,
        "lot_size": lot,
        "oi_unit": "contracts",
        "qty_per_contract": qty_per_contract,
        "dte": round(dte_frac, 4),
        "dte_days": int(dte_frac),
        "dte_trading": round(dte_frac * (252.0 / 365.0), 4),
        "expiry_epoch": near_epoch,
        "futures_price": fut.get("futures_price"),
        "futures_oi": fut.get("futures_oi"),
        "futures_oi_chg": fut.get("futures_oi_chg"),
        "futures_prev_close": fut.get("futures_prev_close"),
        "futures_volume": fut.get("futures_volume"),
        "futures_far_price": fut.get("futures_far_price"),
        **{f"futures{suf}_{k2}": _fq(fut.get(fsk), k1)
           for suf, fsk in (("", "futures_symbol"), ("_far", "futures_far_symbol"))
           for k1, k2 in (("bid", "bid"), ("ask", "ask"), ("bid_qty", "bid_qty"), ("ask_qty", "ask_qty"))},
        "futures_far_oi": fut.get("futures_far_oi"),
        "futures_dte": _monthly_dte(exp_epochs, futs[0][0] if futs else None),
        "india_vix": round(float((head.get(vkey) or {}).get("ltp") or 0), 2) or None,
        "spot_prev_close": isnap.get("prev_close"),
        "session_ohlc": _session_from_ticks(ikey, isnap),
        "timestamp": time.strftime("%d-%b-%Y %H:%M:%S"),
        "_source": "zerodha",
        "_live": bool(_status.get("streaming")),
        "depth": {
            "index": _depth.get(ikey),
            "strikes": {f'{int(round(r["strike"]))}{p.upper()}': _depth[r[f"{p}_symbol"]]
                        for r in strikes for p in ("ce", "pe")
                        if r.get(f"{p}_symbol") and r[f"{p}_symbol"] in _depth},
            "futures": {"near": _depth.get(fut.get("futures_symbol")),
                        "far": _depth.get(fut.get("futures_far_symbol"))},
        },
        "_tokens": {r[f"{p}_symbol"]: {"sym": sym, "k": r["strike"], "side": p.upper()}
                    for r in strikes for p in ("ce", "pe") if r.get(f"{p}_symbol")},
        "flow": {
            "strikes": {k: v for k, v in
                        ((f'{int(round(r["strike"]))}{p.upper()}', _flow_read(r.get(f"{p}_symbol")))
                         for r in strikes for p in ("ce", "pe")) if v},
            "futures": {"near": _flow_read(fut.get("futures_symbol")),
                        "far": _flow_read(fut.get("futures_far_symbol"))},
        },
    }
    if ikey:
        out["_tokens"][ikey] = {"sym": sym, "spot": True}
    _g = time.gmtime(time.time() + 19800)
    _mm = _g.tm_hour * 60 + _g.tm_min
    out["session_state"] = ("closed" if _g.tm_wday >= 5 or _mm >= 930 or _mm < 540
                            else "pre-open" if _mm < 555 else "open")
    for _pre in ("futures", "futures_far"):
        _b, _a = out.get(f"{_pre}_bid"), out.get(f"{_pre}_ask")
        if _b and _a and float(_b) >= float(_a):
            for _k in ("bid", "ask", "bid_qty", "ask_qty"):
                out[f"{_pre}_{_k}"] = None
    for _which, _fs in (("near", fut.get("futures_symbol")), ("far", fut.get("futures_far_symbol"))):
        if _fs:
            out["_tokens"][_fs] = {"sym": sym, "fut": _which}
    with _lock:
        _status["last_chain"] = time.time()
    return out


_LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120, "SENSEX": 20}


# ── historical candles ────────────────────────────────────────────────
_history_cache: Dict[tuple, Any] = {}
_history_lock = threading.Lock()
_history_inflight: Dict[tuple, threading.Event] = {}
_KITE_IV = {1: "minute", 3: "3minute", 5: "5minute", 10: "10minute",
            15: "15minute", 30: "30minute", 60: "60minute"}


def _candle_ts(c: Dict[str, Any]) -> Optional[int]:
    d = c.get("date")
    try:
        if hasattr(d, "timestamp"):
            if getattr(d, "tzinfo", None) is None:        # naive = IST wall clock
                import calendar
                return calendar.timegm(d.timetuple()) - 19800
            return int(d.timestamp())
        import datetime as _dx
        dd = _dx.datetime.fromisoformat(str(d))
        if dd.tzinfo is None:
            import calendar
            return calendar.timegm(dd.timetuple()) - 19800
        return int(dd.timestamp())
    except Exception:
        return None


def _hist(token: int, frm: str, to: str, interval: str) -> List[Dict[str, Any]]:
    rows = _rest("historical", _kite.historical_data, token, frm, to, interval, False, False)
    out = []
    for c in rows or []:
        ts = _candle_ts(c)
        if ts is None:
            continue
        out.append({"t": ts, "o": float(c.get("open") or 0), "h": float(c.get("high") or 0),
                    "l": float(c.get("low") or 0), "c": float(c.get("close") or 0),
                    "v": float(c.get("volume") or 0)})
    return out


def fetch_candles(symbol: str, interval_min: int = 1, days: int = 1,
                  on_date: str = None) -> List[Dict[str, Any]]:
    """Intraday index candles from Kite history, with the near-month
    future's volume laid on (index candles carry none). Cached and coalesced
    exactly as on the Fyers path: completed candles never change."""
    if not connect():
        return []
    sym = symbol.upper()
    _ensure_instruments(_exchanges_for(sym))
    ikey = _idx_key.get(sym)
    tok = _key2tok.get(ikey or "")
    if not tok:
        return []
    key = (ikey, interval_min, days, on_date or "")
    ttl = 86400.0 if on_date else float(os.environ.get("ZERODHA_HISTORY_TTL", os.environ.get("FYERS_HISTORY_TTL", "60")))
    waiter = None
    with _history_lock:
        if len(_history_cache) > 40:
            _history_cache.clear()
        hit = _history_cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        ev = _history_inflight.get(key)
        if ev is not None:
            waiter = ev
        else:
            _history_inflight[key] = threading.Event()
    if waiter is not None:
        waiter.wait(timeout=30)
        with _history_lock:
            hit = _history_cache.get(key)
        return hit[1] if hit else []
    iv = _KITE_IV.get(interval_min, "minute")
    day = on_date or _ist_today(0)
    frm = f"{on_date or _ist_today(max(1, days))} 09:00:00"
    to = f"{day} 15:45:00"
    try:
        bars = _hist(tok, frm, to, iv)
        out = []
        for b in bars:
            g = time.gmtime(b["t"] + 19800)
            if f"{g.tm_year:04d}-{g.tm_mon:02d}-{g.tm_mday:02d}" != day:
                continue
            if not _in_session(g.tm_hour * 60 + g.tm_min):
                continue
            b["n"] = 1
            out.append(b)
        if out and not any(b["v"] for b in out) and \
                _cfg("ZERODHA_INDEX_VOLUME", "futures").lower() != "none":
            fkey = _future_for_day(sym, day)
            ftok = _key2tok.get(fkey or "")
            if ftok:
                try:
                    vol = {b["t"]: b["v"] for b in _hist(ftok, frm, to, iv)}
                    for b in out:
                        b["v"] = vol.get(b["t"], 0.0)
                except Exception as e:  # noqa: BLE001
                    print(f"{_TAG} futures volume for the index chart unavailable: {e}")
        with _history_lock:
            _history_cache[key] = (time.time() if out else 0.0, out)
        if on_date:
            print(f"{_TAG} history {on_date}: {len(out)} candle(s){' - EMPTY, will retry' if not out else ''}")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"{_TAG} history failed: {e}")
        return []
    finally:
        with _history_lock:
            _ev = _history_inflight.pop(key, None)
        if _ev:
            _ev.set()


def _future_for_day(sym: str, day: str) -> Optional[str]:
    spec = _INDEX_SPEC.get(sym)
    if not spec:
        return None
    for e, k in _futs.get((spec[2], spec[3])) or []:
        if e >= day:
            return k
    return None


_hist_range_cache: Dict[str, Any] = {}


def fetch_history_days(symbol: str, days: int = 60) -> List[Dict[str, Any]]:
    """Several sessions of 1-minute index bars. Kite serves up to 60 days of
    minute data per request, so 60 days is one call (more is two), cached in
    memory and on disk for the day."""
    days = max(5, min(95, int(days)))
    sym = symbol.upper()
    today = _ist_today()
    key = f"{sym}_{days}_{today}"
    if key in _hist_range_cache:
        return _hist_range_cache[key]
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "history")
    cpath = os.path.join(cdir, f"{key}.json")
    try:
        if os.path.exists(cpath):
            with open(cpath) as f:
                out = json.load(f)
            _hist_range_cache[key] = out
            return out
    except Exception:
        pass
    if not connect():
        return []
    _ensure_instruments(_exchanges_for(sym))
    tok = _key2tok.get(_idx_key.get(sym) or "")
    if not tok:
        return []
    out: List[Dict[str, Any]] = []
    try:
        start = days
        while start > 0:
            span = min(60, start)
            frm, to = _ist_today(start), _ist_today(max(0, start - span))
            for b in _hist(tok, f"{frm} 09:00:00", f"{to} 15:45:00", "minute"):
                g = time.gmtime(b["t"] + 19800)
                if _in_session(g.tm_hour * 60 + g.tm_min):
                    out.append({"t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]})
            start -= span
    except Exception as e:  # noqa: BLE001
        print(f"{_TAG} multi-day history failed: {e}")
        return []
    out = sorted({b["t"]: b for b in out}.values(), key=lambda b: b["t"])
    if out:
        _hist_range_cache[key] = out
        try:
            os.makedirs(cdir, exist_ok=True)
            tmp = cpath + ".tmp"
            with open(tmp, "w") as f:
                json.dump(out, f, separators=(",", ":"))
            os.replace(tmp, cpath)
        except Exception:
            pass
        print(f"{_TAG} multi-day history: {len(out)} minute bars over {days} days (cached for the day)")
    return out


# ── index constituents: streamed, REST only for the gaps ─────────────
_eq_stats: Dict[str, Any] = {"socket": 0, "rest": 0, "rest_calls": 0, "ts": 0.0}
_eq_rest_cache: Dict[str, Any] = {}


def _eq_row(name: str, d: Dict[str, Any], age: Optional[float] = None) -> Optional[Dict[str, Any]]:
    ltp, prev = float(d.get("ltp") or 0), float(d.get("prev_close") or 0)
    if not ltp:
        return None
    vol = float(d.get("volume") or 0)
    row = {"symbol": name, "lastPrice": ltp, "previousClose": prev,
           "change": round(ltp - prev, 2) if prev else 0.0,
           "pChange": round((ltp - prev) / prev * 100, 2) if prev else 0.0,
           "totalTradedVolume": vol, "totalTradedValue": vol * ltp,
           "dayHigh": float(d.get("high") or 0), "dayLow": float(d.get("low") or 0),
           "open": float(d.get("open") or 0)}
    if age is not None:
        row["_age"] = age
    return row


def constituent_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    if not connect() or not symbols:
        return {}
    _ensure_instruments(("NSE",))
    names = [s.strip().upper() for s in symbols if s and s.strip()]
    keys = {n: _eq_key.get(n) or f"NSE:{n}" for n in names}
    try:
        _subscribe([k for k in keys.values() if k in _key2tok])
    except Exception:
        pass
    out: Dict[str, Dict[str, Any]] = {}
    missing = []
    with _lock:
        for n, k in keys.items():
            t = _ticks.get(k) or {}
            row = _eq_row(n, t, round(time.time() - (t.get("ts") or 0), 1)) if t.get("prev_close") else None
            if row and k in _subscribed:
                out[n] = row
            else:
                missing.append(n)
    _eq_stats.update(socket=len(out), rest=len(missing), ts=time.time())
    now_ = time.time()
    ask = []
    for n in missing:
        hit = _eq_rest_cache.get(n)
        if hit and now_ - hit[0] < 30:
            out[n] = hit[1]
        else:
            ask.append(n)
    if ask and _kite:
        _eq_stats["rest_calls"] += (len(ask) + 499) // 500
        for i in range(0, len(ask), 500):
            batch = ask[i:i + 500]
            try:
                r = _rest("quote", _kite.quote, [keys[n] for n in batch]) or {}
            except Exception as e:  # noqa: BLE001
                print(f"{_TAG} constituent quotes failed: {e}")
                continue
            for n in batch:
                row = _eq_row(n, _norm(r.get(keys[n]) or {}))
                if row:
                    out[n] = row
                    _eq_rest_cache[n] = (time.time(), row)
    return out


# ── diagnostics, in the shapes /api/data-audit and friends read ──────
def depth_status(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    now = time.time()
    with _lock:
        out = {"subscribed": len(_depth_subs), "ladders": sum(1 for v in _depth.values() if v.get("lts")),
               "contracts": {}}
        for s_ in (symbols or []):
            v = _depth.get(s_) or {}
            out["contracts"][s_] = {
                "depth_subscribed": s_ in _depth_subs,
                "ladder_age_s": round(now - v["lts"], 1) if v.get("lts") else None,
                "levels": [len(v.get("bids") or []), len(v.get("asks") or [])],
                "bid5": sum(x["q"] for x in (v.get("bids") or [])),
                "ask5": sum(x["q"] for x in (v.get("asks") or [])),
                "total_buy": v.get("tbq"), "total_sell": v.get("tsq"),
            }
    return out


def _infer_lot_book(symbols) -> Optional[int]:
    vals = []
    with _lock:
        for s_ in symbols:
            bk = _depth.get(s_ or "") or {}
            for lv in (bk.get("bids") or []) + (bk.get("asks") or []):
                q = int(round(lv.get("q") or 0))
                if q > 0:
                    vals.append(q)
    if len(vals) < 20:
        return None
    g = 0
    for v in vals:
        g = math.gcd(g, v)
        if g < 15:
            return None
    return g if g <= 5000 else None


def audit_state(sym: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    ikey = _idx_key.get((sym or "").upper())
    toks = list((payload or {}).get("_tokens") or {})
    with _lock:
        tick_age = {t: now - (_ticks.get(t) or {}).get("ts", 0) for t in toks}
        dp = {t: _depth.get(t) or {} for t in toks}
        spot_age = now - (_ticks.get(ikey) or {}).get("ts", 0) if ikey else None
    opt_toks = [t for t in toks if _kind.get(t) == "OPT"]
    lot_master = next((_lot_of[t] for t in opt_toks if _lot_of.get(t)), None)
    return {
        "socket": {"streaming": _status.get("streaming"), "reconnects": _reconnects["n"],
                   "last_reconnect_age_s": round(now - _reconnects["last"], 1) if _reconnects["last"] else None},
        "spot_tick_age_s": round(spot_age, 1) if spot_age is not None and spot_age < 1e8 else None,
        "options": len(opt_toks),
        "price_subscribed": sum(1 for t in opt_toks if t in _subscribed),
        "ticking_60s": sum(1 for t in opt_toks if tick_age.get(t, 1e9) < 60),
        "depth_subscribed": sum(1 for t in toks if t in _depth_subs),
        "ladders_fresh_120s": sum(1 for t in toks if dp[t].get("lts") and now - dp[t]["lts"] < 120),
        "totals_fresh_120s": sum(1 for t in toks if dp[t].get("tts") and now - dp[t]["tts"] < 120),
        "flow_contracts": sum(1 for t in toks if _flow_read(t)),
        "flow_coverage": {t: (_flow_read(t) or {}).get("coverage") for t in toks if _kind.get(t) == "FUT"},
        "lot_from_data": lot_master,
        "lot_from_book": _infer_lot_book(opt_toks),
        "lot_from_master": lot_master, "master_error": None,
        "lot_table": _LOTS.get((sym or "").upper()),
        "raw_rest": dict(_raw_rest),
        "oi_change_base": {"previous_session": _oi_prev.get("day"), "contracts": len(_oi_prev.get("oi") or {}),
                           "complete": _oi_prev.get("complete"), "backfill_queue": len(_oi_backfill_q)},
        "watchdog": watchdog_state(),
    }


def raw_debug(match: str) -> Dict[str, Any]:
    now = time.time()
    m = (match or "").upper()
    out: Dict[str, Any] = {}
    hits = [(s, b) for s, b in list(_raw_last.items()) if not m or m in s.upper()]
    hits.sort(key=lambda sb: (_kind.get(sb[0]) != "FUT", sb[0] not in _depth_subs,
                              -max((ts for ts, _ in sb[1].values()), default=0)))
    for sym, by in hits:
        dp = _depth.get(sym) or {}
        out[sym] = {
            "raw": {t: {"age_s": round(now - ts, 1), "msg": msg} for t, (ts, msg) in by.items()},
            "stored": {
                "ladder_age_s": round(now - dp["lts"], 1) if dp.get("lts") else None,
                "bids": dp.get("bids"), "asks": dp.get("asks"),
                "bid5_sum": sum(x["q"] for x in (dp.get("bids") or [])),
                "ask5_sum": sum(x["q"] for x in (dp.get("asks") or [])),
                "total_buy": dp.get("tbq"), "total_sell": dp.get("tsq"),
                "tick": {k: v for k, v in (_ticks.get(sym) or {}).items()
                         if k in ("ltp", "bid", "ask", "bid_qty", "ask_qty", "volume", "oi")},
            },
            "subscribed": {"price": sym in _subscribed, "depth": sym in _depth_subs},
            "flow": _flow_read(sym),
        }
        if len(out) >= 12:
            break
    if len(hits) > len(out):
        out["_note"] = (f"showing {len(out)} of {len(hits)} matching contracts "
                        "(futures first, then book-carrying, newest); narrow the symbol to see others")
    return out


def depth_snapshot() -> Dict[str, Any]:
    with _lock:
        return {k: dict(v) for k, v in _depth.items()}


# ── orders (used only by the Scalping tab's engine, nse_trade.py) ─────
# Called directly rather than through _rest(): _rest paces calls 0.12 s apart
# behind everything else, and an exit must not queue behind a background call.
# They are still counted in the day's API usage.
def _order_call(kind: str, fn, *args, **kwargs):
    _count_rest(kind)
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # noqa: BLE001
        _rest_errors.append({"ts": time.time(), "kind": kind, "err": f"{type(e).__name__}: {str(e)[:140]}"})
        del _rest_errors[:-30]
        if type(e).__name__ == "TokenException":
            _set_err(f"token rejected ({e}) - run zerodha_login.py")
        raise


def place_order(tradingsymbol: str, side: str, qty: int, price: float, tag: str = "") -> str:
    """One marketable LIMIT, IOC, intraday (MIS) order on NFO. Returns the order id.
    A static IP registered in the Kite developer console is required for this call."""
    if not connect() or _kite is None:
        raise RuntimeError(status().get("last_error") or "zerodha unavailable")
    return str(_order_call(
        "order", _kite.place_order, variety="regular", exchange="NFO", tradingsymbol=tradingsymbol,
        transaction_type="BUY" if side.upper() == "BUY" else "SELL", quantity=int(qty), product="MIS",
        order_type="LIMIT", price=float(price), validity="IOC", tag=(tag or "nsescalp")[:20]))


def order_margin(orders: List[Dict[str, Any]]) -> Dict[str, float]:
    """Margin the broker would block for these orders together (a hedged short is
    cheaper than a naked one - the basket call knows). ONE API call; the page only
    makes it when you press 'Check margin', never on a timer."""
    if not connect() or _kite is None:
        raise RuntimeError(status().get("last_error") or "zerodha unavailable")
    params = [dict(exchange="NFO", tradingsymbol=o["tradingsymbol"], transaction_type="BUY" if str(o["side"]).upper() == "BUY" else "SELL",
                   variety="regular", product="MIS", order_type="LIMIT", quantity=int(o["qty"]), price=float(o["price"])) for o in orders]
    r = _order_call("margin", _kite.basket_order_margins, params, consider_positions=True) or {}
    tot = lambda k: float(((r.get(k) or {}).get("total")) or 0)          # noqa: E731
    return {"initial": tot("initial"), "required": tot("final") or tot("initial")}


def order_status(order_id: str) -> Dict[str, Any]:
    """Latest state of one order: status, filled quantity, average price, message."""
    if _kite is None:
        return {"status": "UNKNOWN", "msg": "not connected"}
    hist = _order_call("order_status", _kite.order_history, order_id) or []
    last = hist[-1] if hist else {}
    return {"status": str(last.get("status") or "UNKNOWN").upper(), "filled": int(last.get("filled_quantity") or 0),
            "avg": float(last.get("average_price") or 0), "msg": str(last.get("status_message") or "")}


def net_positions() -> List[Dict[str, Any]]:
    """The broker's intraday option positions (NFO, MIS)."""
    if _kite is None:
        return []
    net = (_order_call("positions", _kite.positions) or {}).get("net") or []
    return [r for r in net if r.get("exchange") == "NFO" and r.get("product") == "MIS"]


def market_depth(symbol: str) -> Dict[str, Any]:
    if not connect():
        return {}
    sym = symbol.upper()
    key = _idx_key.get(sym, symbol)
    try:
        r = _rest("quote", _kite.quote, [key]) or {}
        return r.get(key) or {}
    except Exception:
        return {}


def shutdown() -> None:
    global _kws, _socket
    try:
        if _kws:
            _ws_call(_kws.close)
    except Exception:
        pass
    _kws = _socket = None
    try:
        _oi_save_today()
    except Exception:
        pass
    with _lock:
        _status.update(streaming=False, connected=False)


# ── feed delay and API-call timeline (read by the Diagnostics tab) ────
# Delay: seconds between the exchange stamping an update and this server
# receiving it, sampled at most twice a second. Exchange stamps are whole
# seconds, so a healthy feed reads about 0.5 s on average; the clock check
# on the Diagnostics tab separates a slow feed from a wrong computer clock.
import collections as _collections

_lat = _collections.deque(maxlen=4000)          # (received epoch, delay s)
_lat_last = {"t": 0.0, "best": None}
# REST calls per minute today, and which browser / endpoint caused them.
_rest_min: Dict[int, int] = {}
_rest_who: Dict[str, int] = {}
_who = threading.local()                        # the server names the request being served


def _lat_note(ex_ts: float, now: Optional[float] = None) -> None:
    now = now or time.time()
    if not ex_ts or ex_ts < 1e9:
        return
    best = _lat_last["best"]
    if best is None or ex_ts > best:
        _lat_last["best"] = ex_ts
    if now - _lat_last["t"] >= 0.5:
        _lat.append((now, now - (_lat_last["best"] or ex_ts)))
        _lat_last.update(t=now, best=None)


def _rest_attrib() -> None:
    m = int(time.time() // 60)
    _rest_min[m] = _rest_min.get(m, 0) + 1
    if len(_rest_min) > 1500:
        for k in sorted(_rest_min)[:-1440]:
            _rest_min.pop(k, None)
    who = getattr(_who, "name", None) or _bg_label(threading.current_thread().name)
    _rest_who[who] = _rest_who.get(who, 0) + 1


_BG_NAMES = {"_oi_keeper_loop": "OI back-fill", "zerodha-oi": "OI back-fill", "_watchdog_loop": "stream watchdog",
             "fyers-watchdog": "stream watchdog", "zerodha-watchdog": "stream watchdog", "poller": "chain poller",
             "_run": "server background job", "MainThread": "server startup", "_master_load": "symbol master"}


def _bg_label(name: str) -> str:
    """'Thread-12 (_run)' -> 'background · server background job'."""
    target = name.split("(", 1)[1].rstrip(")") if "(" in name else name
    return f"background · {_BG_NAMES.get(target) or _BG_NAMES.get(name) or target}"


def latency_stats(window_s: float = 300.0) -> Dict[str, Any]:
    now = time.time()
    xs = sorted(d for t, d in list(_lat) if now - t <= window_s)
    per_min: Dict[int, List[float]] = {}
    for t, d in list(_lat):
        if now - t <= 1800:
            per_min.setdefault(int(t // 60), []).append(d)
    pct = lambda a, q: a[min(len(a) - 1, int(q * len(a)))] if a else None
    return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": xs[-1] if xs else None,
            "min": xs[0] if xs else None, "last_age_s": round(now - _lat[-1][0], 1) if _lat else None,
            "series": [[m * 60, round(sorted(v)[len(v) // 2], 2)] for m, v in sorted(per_min.items())]}


def rest_timeline() -> Dict[str, Any]:
    g = time.gmtime(time.time() + 19800)
    day0 = int((time.time() + 19800) // 86400 * 86400 - 19800)
    mins = [[m * 60, n] for m, n in sorted(_rest_min.items()) if m * 60 >= day0]
    top = sorted(_rest_who.items(), key=lambda kv: -kv[1])[:12]
    return {"limit_per_min": 600, "limit_label": "Kite 10 calls/s overall (quotes 1/s, history 3/s)", "minutes": mins, "peak": max((n for _m, n in mins), default=0), "by_client": top}


# ── diagnostics: everything the Diagnostics tab shows, from memory only ──
def _age(ts: Optional[float]) -> Optional[float]:
    return round(time.time() - ts, 1) if ts else None


def _fmt_age(a: Optional[float]) -> str:
    if a is None:
        return "—"
    return f"{a:.0f}s ago" if a < 120 else f"{a / 60:.0f} min ago" if a < 7200 else f"{a / 3600:.1f} h ago"


def diagnostics() -> Dict[str, Any]:
    """The broker's state in rows of (label, value, status) grouped into
    sections - status is ok / warn / bad / '' . Reads memory only: opening
    the Diagnostics tab costs no Kite call."""
    now = time.time()
    st = status()
    tv = st.get("token") or {}
    wd = watchdog_state()
    mkt = _ist_market_open(now)
    with _lock:
        last_any = max((t.get("ts") or 0 for k, t in _ticks.items() if _kind.get(k) in ("OPT", "FUT")), default=0)
        last_idx = max((t.get("ts") or 0 for k, t in _ticks.items() if _kind.get(k) == "IDX"), default=0)
        n_ticking = sum(1 for t in _ticks.values() if now - (t.get("ts") or 0) < 60)
        n_full = len(_depth_subs)
        n_eq = sum(1 for k in _subscribed if _kind.get(k) == "EQ")
    a_any, a_idx = _age(last_any), _age(last_idx)
    live = lambda a, lim: "ok" if a is not None and a < lim else ("bad" if mkt else "")
    rest = rest_usage()
    by = rest.get("by") or {}
    errs10 = [e for e in _rest_errors if now - e["ts"] < 600]
    rf = wd.get("refusals_10m") or {}
    sections = [
        {"title": "Login & token", "rows": [
            ["SDK (kiteconnect)", "installed" if SDK_AVAILABLE else f"missing - pip install kiteconnect", "ok" if SDK_AVAILABLE else "bad"],
            ["API key in env", "set" if _cfg("ZERODHA_API_KEY") else "missing", "ok" if _cfg("ZERODHA_API_KEY") else "bad"],
            ["Access token", tv.get("reason") or "—", "ok" if tv.get("valid") else "bad"],
            ["Token expires in", f"{tv['expires_in_h']} h" if tv.get("expires_in_h") is not None else "—",
             "warn" if (tv.get("expires_in_h") or 99) < 1 else ""],
            ["Authenticated", "yes" if st.get("connected") else "no", "ok" if st.get("connected") else "bad"],
            ["Stream refused (403)", "yes - run zerodha_login.py" if st.get("token_rejected") else "no",
             "bad" if st.get("token_rejected") else "ok"],
            ["Last error", st.get("last_error") or "none", "bad" if st.get("last_error") else "ok"],
        ]},
        {"title": "Live stream (KiteTicker)", "rows": [
            ["Streaming", "yes" if st.get("streaming") else "no", "ok" if st.get("streaming") else ("bad" if mkt else "warn")],
            ["Stream socket", "open" if _ws_ready() else ("not started" if _kws is None else "connecting / reconnecting"),
             "ok" if _ws_ready() else "warn"],
            ["Held until connect", f"{_held['n']} subscription(s)" if _held["n"] else "none", "warn" if _held["n"] else "ok"],
            ["Last contract tick", _fmt_age(a_any), live(a_any, 30)],
            ["Last index tick", _fmt_age(a_idx), live(a_idx, 15)],
            ["Instruments ticking (60s)", f"{n_ticking}", ""],
            ["Subscribed", f"{len(_subscribed)} of {_MAX_TOKENS} allowed ({n_full} full-mode with book, {n_eq} equities)",
             "warn" if len(_subscribed) > 0.9 * _MAX_TOKENS else "ok"],
            ["Never answered", f"{wd.get('price_unconfirmed', 0)}" + (f" ({', '.join(x.split(':')[-1] for x in wd.get('price_unconfirmed_sample') or [])})" if wd.get('price_unconfirmed') else ""),
             "warn" if wd.get("price_unconfirmed") else "ok"],
            ["Connects", f"{_reconnects['n']} (last {_fmt_age(_age(_reconnects['last']))})", ""],
            ["Watchdog", f"{wd.get('heals', 0)} re-subscribe(s), {wd.get('forced_reconnects', 0)} reopen(s), "
                         f"next reopen no sooner than {int(wd.get('next_reopen_gap_s', 300) / 60)} min apart", ""],
            ["Stream errors (10 min)", f"{rf.get('total', 0)}" + (f" - last: {((rf.get('last') or {}).get('text') or '')[:90]}" if rf.get("total") else ""),
             "warn" if rf.get("total") else "ok"],
            ["Ticks received", f"{st.get('ticks', 0):,}", ""],
        ]},
        {"title": "API usage today (Kite: quote 1/s, history 3/s, other 10/s - flat monthly plan)", "rows": [
            ["REST calls", f"{rest.get('n', 0):,}", ""],
            *[[f"  {k}", f"{v:,}", ""] for k, v in sorted(by.items(), key=lambda kv: -kv[1])],
            ["Errors (10 min)", f"{len(errs10)}" + (f" - last: {errs10[-1]['kind']} {errs10[-1]['err']}" if errs10 else ""),
             "warn" if errs10 else "ok"],
            ["Quote cache", f"{len(_qcache)} contracts (reused {_CHAIN_REST_TTL:.0f}s; 300s with the market shut)", ""],
            ["History cache", f"{len(_history_cache)} series (TTL {os.environ.get('ZERODHA_HISTORY_TTL', '60')}s)", ""],
            ["Constituents", f"{_eq_stats.get('socket', 0)} from the stream, {_eq_stats.get('rest', 0)} needed REST "
                             f"({_eq_stats.get('rest_calls', 0)} quote call(s) so far)", ""],
        ]},
        {"title": "Instrument lists (daily, cached on disk)", "rows": [
            *[[x, f"{_inst['rows'].get(x, 0)} rows · {_inst['day'].get(x) or '—'} · fetched "
                  f"{_fmt_age(_age(_inst['fetched'].get(x)))}"
                  + (" (provisional - before Kite's 08:30 update)" if _inst['fetched'].get(x) and _fetched_early(_inst['fetched'][x]) else ""),
               "ok" if _inst["day"].get(x) == _ist_today() else "warn"] for x in sorted(_inst["day"])],
            ["Index symbols", ", ".join(f"{k}={v.split(':', 1)[-1]}" for k, v in sorted(_idx_key.items())) or "—", ""],
        ]},
        {"title": "OI change baseline (Kite publishes no previous OI)", "rows": [
            ["Previous session", _oi_prev.get("day") or "—", "ok" if _oi_prev.get("day") else "warn"],
            ["Contracts with a previous close", f"{len(_oi_prev.get('oi') or {})}"
             + (" (complete)" if _oi_prev.get("complete") else " (partial)"), "ok" if _oi_prev.get("complete") else "warn"],
            ["Captured before 09:15 today", f"{len(_oi_preopen['oi']) if _oi_preopen['day'] == _ist_today() else 0}", ""],
            ["Back-fill queue", f"{len(_oi_backfill_q)} waiting, {len(_oi_tried)} done today", ""],
        ]},
    ]
    bad = sum(1 for s_ in sections for r in s_["rows"] if r[2] == "bad")
    warn = sum(1 for s_ in sections for r in s_["rows"] if r[2] == "warn")
    return {"id": "zerodha", "name": "Zerodha Kite Connect",
            "verdict": "bad" if bad else "warn" if warn else "ok",
            "summary": (f"{bad} problem(s)" if bad else f"{warn} warning(s)" if warn else "healthy")
                       + ("" if mkt else " · market closed"),
            "sections": sections,
            "log": wd.get("log") or []}
