"""
Fyers API v3 market-data adapter for 1OPTIONS.

Why this exists
---------------
The default source scrapes NSE's public option chain: no account needed, but
the OI it returns is several minutes stale, it needs a cookie warm-up that
breaks whenever NSE changes its front end, and there is no tick stream at all.
Fyers gives the same information from a broker feed - real time, with bid/ask
and market depth, and a WebSocket that pushes every price change.

What this adapter does and does not do
--------------------------------------
* Option chain comes from Fyers' REST `optionchain` call, which returns OI,
  LTP, bid and ask per strike. Fyers has no option-chain WebSocket, so the
  chain is polled while the SOCKET carries live prices for the contracts we
  subscribe to - the same split the Arrow adapter uses.
* Every response is normalised to the exact shape the rest of the server
  already consumes, so the cache, alert engine, SSE stream, trend engine and
  dashboard need no knowledge of which broker is behind it.
* Everything degrades rather than crashes. A missing SDK, an expired token, a
  dropped socket - each reports unavailable and the server falls back to NSE.

Auth note
---------
Fyers access tokens expire daily and the login flow is interactive (auth code
in a browser). The adapter therefore reads a token from .env or a token file
and tells you plainly when it needs refreshing, rather than pretending it can
renew one on its own.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from fyers_apiv3 import fyersModel
    from fyers_apiv3.FyersWebsocket import data_ws
    SDK_AVAILABLE = True
    SDK_ERROR = ""
except Exception as _e:  # noqa: BLE001
    SDK_AVAILABLE = False
    SDK_ERROR = str(_e)


# Fyers symbol names for the indices this dashboard trades.
FYERS_INDEX = {
    "NIFTY": "NSE:NIFTY50-INDEX",
    "BANKNIFTY": "NSE:NIFTYBANK-INDEX",
    "FINNIFTY": "NSE:FINNIFTY-INDEX",
    "MIDCPNIFTY": "NSE:MIDCPNIFTY-INDEX",
    "SENSEX": "BSE:SENSEX-INDEX",
}
INDIA_VIX = "NSE:INDIAVIX-INDEX"

# ── session bounds, shared with the server ────────────────────────────
# Under SEBI's Closing Auction Session (live since 3 August 2026) equity
# DERIVATIVES trade until 15:40, ten minutes past the old cash close, while
# F&O-eligible cash stocks stop continuous trading at 15:15 and go to auction.
# These read the same override the server uses, so a future timing change is a
# one-place edit rather than a divergence that silently truncates the last ten
# minutes of every session on one code path but not the other.
SESSION_OPEN_MIN = 9 * 60 + 15
SESSION_END_MIN = int(os.environ.get("NSE_SESSION_END_MIN", 15 * 60 + 40))
CAS_CASH_CUTOFF_MIN = 15 * 60 + 15


def _in_session(mins: int) -> bool:
    return SESSION_OPEN_MIN <= mins < SESSION_END_MIN


def market_open_now() -> bool:
    """Is the derivatives market trading right now (IST)?"""
    g = time.gmtime(time.time() + 5 * 3600 + 1800)
    if g.tm_wday >= 5:
        return False
    return _in_session(g.tm_hour * 60 + g.tm_min)

_client: Optional[Any] = None
_socket: Optional[Any] = None
_lock = threading.Lock()
# The SDK's subscribe() stores data_type on the socket object and reads it
# back while converting symbols - after an HTTP round trip to validate them.
# Two subscribes at once (two PCs, two indices, price and depth from the
# same chain build) could each run with the OTHER's type: price symbols
# subscribed as depth (no ticks - prices fall back to slow REST) or depth
# subscribed as price. Every subscribe goes through this lock.
_sub_lock = threading.Lock()
# last raw message per contract and type, exactly as the SDK delivered it,
# for /api/depth-debug - the ground truth to compare against a terminal
_raw_last: Dict[str, Dict[str, Any]] = {}
_raw_rest: Dict[str, Any] = {}          # last raw REST rows, for /api/data-audit
_reconnects = {"n": 0, "last": 0.0}
# Expiry -> broker epoch, learned once per (symbol, expiry). Fyers validates
# `timestamp` against its own list, so the first request for a non-nearest
# expiry has to ask without one and look it up; caching the answer means every
# later poll makes ONE option-chain call instead of two. REST calls are capped
# per second, per minute and per day, so doubling them on every poll is not free.
_exp_ts_cache: Dict[str, str] = {}
# A streamed price older than this is treated as stale and re-fetched.
_TICK_FRESH_S = float(os.environ.get("FYERS_TICK_FRESH_S", "15"))
# Futures REST quote is refreshed at most this often; price comes from ticks.
_FUT_REST_TTL = float(os.environ.get("FYERS_FUT_REST_TTL", "60"))
_fut_cache: Dict[str, Any] = {}
# Option-chain REST response reused for this long; prices come from ticks.
_CHAIN_REST_TTL = float(os.environ.get("FYERS_CHAIN_REST_TTL", "15"))
_oc_cache: Dict[str, Any] = {}
_ticks: Dict[str, Dict[str, Any]] = {}      # fyers symbol -> latest tick
_depth: Dict[str, Dict[str, Any]] = {}      # fyers symbol -> latest L2 book
_depth_subs: set = set()
_subscribed: set = set()
_tick_sink = None
_last_push = 0.0
_dirty: Dict[str, Dict[str, Any]] = {}

# ── aggressor flow: who is taking liquidity ──────────────────────────────
# Every trade is either a buyer lifting the offer or a seller hitting the
# bid, and that - not the resting bid/ask quantity - is what desks weigh.
# Resting size is passive and cancellable (deep in the money it is mostly
# market-maker inventory); a print is commitment. Classified here, on every
# raw tick rather than the 4 Hz batches the browsers get, with the quote that
# stood BEFORE the trade (the Lee-Ready rule): at or above the prior offer is
# a buy, at or below the prior bid a sell, in between by the tick test.
# Minute buckets for the session; nothing extra is requested from Fyers.
_flow: Dict[str, Dict[str, Any]] = {}       # fyers symbol -> {"v","ltp","bid","ask","sgn","m":{minute:[buy,sell]}}


def _flow_update(sym: str, d: Dict[str, Any]) -> None:
    vol, ltp = d.get("volume"), d.get("ltp")
    if vol is None or ltp is None:
        return
    try:
        vol, ltp = float(vol), float(ltp)
    except Exception:
        return
    f = _flow.get(sym)
    if f is None:
        _flow[sym] = {"v": vol, "v0": vol, "ltp": ltp, "bid": d.get("bid"), "ask": d.get("ask"), "sgn": 0, "m": {}}
        return
    dv = vol - f["v"]
    if dv < 0:                        # a new session's counter
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
    # list() of a dict is taken in one step under the GIL, so the socket
    # thread adding a minute mid-read cannot break the loop
    first = None
    for mnt, (b, s_) in list(f["m"].items()):
        first = mnt if first is None else min(first, mnt)
        bd += b; sd += s_
        if now - mnt < 15:
            b15 += b; s15 += s_
        if now - mnt < 5:
            b5 += b; s5 += s_
    # share of the volume traded since the server started that was classified
    traded = (f.get("v") or 0) - (f.get("v0") or 0)
    return {"b5": b5, "s5": s5, "b15": b15, "s15": s15, "bd": bd, "sd": sd,
            "since": first * 60, "coverage": round((bd + sd) / traded, 3) if traded > 0 else None}
_status: Dict[str, Any] = {
    "connected": False, "streaming": False, "last_error": "",
    "ticks": 0, "subscribed": 0, "last_chain": 0.0, "token_age_h": None,
}


# ── configuration ─────────────────────────────────────────────────────
_ENV_NAMES = ("env", ".env", ".env.local", "env.txt", ".env.txt")
_env_cache: Optional[Dict[str, str]] = None


def _env_file_values() -> Dict[str, str]:
    """Read the settings file directly.

    Normally the server has already loaded it into os.environ, but the adapter
    can be imported standalone (a test, a script), and it accepts a plain "env"
    as well as ".env" because a filename mismatch failing silently is a
    miserable thing to debug.
    """
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


def is_configured() -> bool:
    return bool(_cfg("FYERS_CLIENT_ID") and _access_token())


def _access_token() -> str:
    """Token from .env, or from a token file the login helper writes."""
    tok = _cfg("FYERS_ACCESS_TOKEN")
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


def token_validity() -> Dict[str, Any]:
    """Is the saved token still good, and for how long?

    Fyers tokens expire at roughly 06:00 IST the day after they are issued, so
    a token minted this morning survives every restart until tomorrow. Knowing
    that up front is the difference between "just start the server" and a
    confusing auth failure three restarts later.
    """
    path = _token_path()
    if not os.path.exists(path):
        return {"present": False, "valid": False,
                "reason": "no saved token - run fyers_login.py"}
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return {"present": True, "valid": False, "reason": f"token file unreadable: {e}"}
    ts = data.get("ts", 0)
    if not ts:
        return {"present": True, "valid": True, "reason": "no timestamp recorded",
                "expires_in_h": None}
    # next 06:00 IST after the token was issued
    issued_ist = time.gmtime(ts + 5 * 3600 + 1800)
    day_start = ts - (issued_ist.tm_hour * 3600 + issued_ist.tm_min * 60 + issued_ist.tm_sec)
    expiry = day_start + 24 * 3600 + 6 * 3600      # 06:00 IST tomorrow
    if issued_ist.tm_hour < 6:
        expiry = day_start + 6 * 3600              # issued before 06:00: expires today
    left = (expiry - time.time()) / 3600
    return {"present": True, "valid": left > 0,
            "age_h": round((time.time() - ts) / 3600, 1),
            "expires_in_h": round(left, 1),
            "reason": ("valid" if left > 0
                       else "expired - run fyers_login.py to refresh")}


def _token_path() -> str:
    return _cfg("FYERS_TOKEN_FILE") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".fyers_token")


def status() -> Dict[str, Any]:
    with _lock:
        s = dict(_status)
    s["sdk"] = SDK_AVAILABLE
    s["configured"] = is_configured()
    s["token"] = token_validity()
    if not SDK_AVAILABLE:
        s["last_error"] = s["last_error"] or f"fyers-apiv3 not installed: {SDK_ERROR}"
    elif not is_configured():
        s["last_error"] = ("FYERS_CLIENT_ID or access token missing - run "
                           "fyers_login.py to generate one (they expire daily)")
    return s


def _set_err(msg: str) -> None:
    with _lock:
        _status["last_error"] = msg
        _status["connected"] = False
    print(f"[fyers] {msg}")


# ── connection ────────────────────────────────────────────────────────
def connect() -> bool:
    global _client
    if not SDK_AVAILABLE:
        _set_err(f"fyers-apiv3 not installed: {SDK_ERROR}")
        return False
    client_id = _cfg("FYERS_CLIENT_ID")
    token = _access_token()
    if not client_id or not token:
        _set_err("client id or access token missing - see fyers_login.py")
        return False
    tv = token_validity()
    if tv.get("present") and not tv.get("valid"):
        _set_err(f"saved token {tv.get('reason')} (issued {tv.get('age_h')}h ago). "
                 f"Run: python3 fyers_login.py")
        return False
    if tv.get("expires_in_h") is not None:
        print(f"[fyers] token valid for another {tv['expires_in_h']}h "
              f"- restarts until then need no re-login")
    with _lock:
        if _status["connected"] and _client is not None:
            return True
    try:
        c = fyersModel.FyersModel(client_id=client_id, token=token,
                                  is_async=False, log_path="")
        prof = c.get_profile()
        if not isinstance(prof, dict) or prof.get("s") != "ok":
            # Fyers answers with s='error' and a message rather than raising,
            # and an expired token is by far the most common cause.
            msg = (prof or {}).get("message", "unknown")
            _set_err(f"profile check failed: {msg}. Tokens expire daily - "
                     f"run fyers_login.py to refresh")
            return False
        name = (prof.get("data") or {}).get("name", "")
        _client = c
        with _lock:
            _status.update(connected=True, last_error="")
        print(f"[fyers] authenticated{f' as {name}' if name else ''}")
        _start_socket()
        return True
    except Exception as e:  # noqa: BLE001
        _set_err(f"connect failed: {type(e).__name__} {e}")
        return False


# ── live stream ───────────────────────────────────────────────────────
def set_tick_sink(fn) -> None:
    """Register the callback that forwards ticks to connected browsers."""
    global _tick_sink
    _tick_sink = fn


def _push_ticks() -> None:
    """Forward accumulated ticks, throttled.

    Raw ticks arrive far faster than any UI can repaint, and faster than a
    browser can usefully receive them. Changes are coalesced and flushed on a
    fixed cadence instead of being relayed one by one.
    """
    global _last_push
    if _tick_sink is None:
        return
    hz = float(os.environ.get("FYERS_PUSH_HZ", "4"))
    now = time.time()
    if now - _last_push < (1.0 / max(1.0, hz)):
        return
    with _lock:
        if not _dirty:
            return
        batch, _dirty_local = dict(_dirty), None
        _dirty.clear()
    _last_push = now
    try:
        _tick_sink({"type": "ticks", "ts": now, "n": len(batch), "t": batch})
    except Exception:
        pass


def _on_message(msg) -> None:
    """Normalise a Fyers tick and stage it for the browsers."""
    try:
        if not isinstance(msg, dict):
            return
        sym = msg.get("symbol") or msg.get("s") or msg.get("symbol_name")
        if not sym:
            return
        try:
            _raw_last.setdefault(sym, {})[str(msg.get("type") or "?")] = (time.time(), dict(msg))
        except Exception:
            pass
        # ── DepthUpdate, as the v3 SDK actually delivers it ──────────
        # FLAT numbered fields - bid_price1..5, bid_size1..5, bid_order1..5
        # and the same for ask, type "dp" - with no ltp, no volume and no
        # totals (checked against the SDK's own map.json). The handler below
        # looked for "bids"/"asks" lists, which only the REST depth call
        # returns, and returned early on "no ltp" before even that - so every
        # five-level book the socket sent was thrown away, and the panels fell
        # back to the WHOLE-book totals from the price tick: every resting
        # order at every price, far-from-market ones included. That is not
        # the bid and offer a depth window shows.
        if msg.get("type") == "dp" or "bid_price1" in msg or "ask_price1" in msg:
            def _flat(side):
                out = []
                for i in range(1, 6):
                    p = msg.get(f"{side}_price{i}")
                    try:
                        p = float(p) if p is not None else 0.0
                    except Exception:
                        p = 0.0
                    if p > 0:
                        out.append({"p": p, "q": float(msg.get(f"{side}_size{i}") or 0),
                                    "n": int(msg.get(f"{side}_order{i}") or 0)})
                return out
            b5, a5 = _flat("bid"), _flat("ask")
            if b5 or a5:
                now_ = time.time()
                with _lock:
                    prev = _depth.get(sym) or {}
                    _depth[sym] = {"bids": b5, "asks": a5, "ts": now_, "lts": now_,
                                   "tbq": prev.get("tbq", 0.0), "tsq": prev.get("tsq", 0.0),
                                   "tts": prev.get("tts")}
            return
        d = {}
        for src, dst in (("ltp", "ltp"), ("last_price", "ltp"), ("lp", "ltp"),
                         ("vol_traded_today", "volume"), ("volume", "volume"),
                         ("open_interest", "oi"), ("oi", "oi"),
                         ("bid_price", "bid"), ("ask_price", "ask"),
                         ("bid_size", "bid_qty"), ("ask_size", "ask_qty"),
                         ("high_price", "high"), ("low_price", "low"),
                         ("open_price", "open"), ("prev_close_price", "prev_close")):
            v = msg.get(src)
            if v is not None:
                d[dst] = v
        if not d:
            return
        # ── Level-2 book, when this message carries one ─────────────
        # DepthUpdate arrives on the same socket as price ticks. Storing the
        # full ladder rather than just the top of book matters: the top level
        # is where spoofing lives, and depth several levels back is what
        # actually has to be consumed for price to move.
        bids = msg.get("bids") or msg.get("bid") or []
        asks = msg.get("asks") or msg.get("ask") or []
        if bids or asks:
            def _lvls(rows):
                out = []
                for r in (rows or [])[:5]:
                    if not isinstance(r, dict):
                        continue
                    p = r.get("price") or r.get("prc")
                    q = r.get("volume") or r.get("qty") or r.get("vol")
                    n = r.get("ord") or r.get("orders") or 0
                    if p:
                        out.append({"p": float(p), "q": float(q or 0), "n": int(n or 0)})
                return out
            b, a = _lvls(bids), _lvls(asks)
            if b or a:
                with _lock:
                    prev = _depth.get(sym) or {}
                    _depth[sym] = {
                        "bids": b, "asks": a, "ts": time.time(), "lts": time.time(),
                        "tbq": float(msg.get("tot_buy_qty") or msg.get("totalbuyqty") or prev.get("tbq") or 0),
                        "tsq": float(msg.get("tot_sell_qty") or msg.get("totalsellqty") or prev.get("tsq") or 0),
                    }

        # TOTAL buy and sell quantity, which Fyers sends on the ORDINARY tick
        # as well as on a depth update. This is the aggregate the bid/offer
        # panel actually wants, and it arrives without a DepthUpdate
        # subscription at all - so when the depth ladder is unavailable, or
        # the broker does not publish it for a contract, the totals are still
        # there. They were being read only inside the ladder branch and thrown
        # away otherwise.
        _tbq = msg.get("tot_buy_qty") or msg.get("totalbuyqty") or msg.get("tbq")
        _tsq = msg.get("tot_sell_qty") or msg.get("totalsellqty") or msg.get("tsq")
        if _tbq or _tsq:
            with _lock:
                prev = _depth.get(sym) or {}
                _depth[sym] = {
                    "bids": prev.get("bids") or [],
                    "asks": prev.get("asks") or [],
                    "lts": prev.get("lts"),           # when the LADDER last updated
                    "tts": time.time(),               # when the TOTALS last updated
                    "ts": time.time(),
                    "tbq": float(_tbq or 0),
                    "tsq": float(_tsq or 0),
                }

        d["ts"] = time.time()
        with _lock:
            try:
                _flow_update(sym, d)
            except Exception:
                pass
            _ticks[sym] = {**_ticks.get(sym, {}), **d}
            _status["ticks"] += 1
            _status["streaming"] = True
            _dirty[sym] = {k: v for k, v in _ticks[sym].items()
                           if k in ("ltp", "oi", "volume", "bid", "ask", "bid_qty", "ask_qty")}
        _push_ticks()
    except Exception:
        pass


def _on_socket_connect() -> None:
    """Every (re)connect, not just the first.

    On a reconnect the SDK clears its own subscription table and does NOT
    resubscribe, while our _subscribed / _depth_subs sets still listed every
    symbol - so nothing was ever requested again. Prices fell back to REST
    quotes (spending quota), and the depth ladders simply stopped: the books
    froze on their last values, which is how a futures bid/offer can sit
    unchanged for hours. Clear the sets and ask again for everything.
    """
    print("[fyers] market-data socket connected")
    _reconnects["n"] += 1
    _reconnects["last"] = time.time()
    with _lock:
        prev_px, prev_dp = sorted(_subscribed), list(_depth_subs)
        _subscribed.clear()
        _depth_subs.clear()
    if not prev_px and not prev_dp:
        return

    def _again():
        time.sleep(1.0)
        try:
            if prev_px:
                _subscribe(prev_px)
            if prev_dp:
                _subscribe_depth(prev_dp)
            print(f"[fyers] resubscribed {len(prev_px)} price + {len(prev_dp)} depth after reconnect")
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] resubscribe failed: {e}")
    threading.Thread(target=_again, daemon=True).start()


def _start_socket() -> None:
    global _socket
    if _socket is not None or not _client:
        return
    token = f"{_cfg('FYERS_CLIENT_ID')}:{_access_token()}"
    try:
        s = data_ws.FyersDataSocket(
            access_token=token,
            write_to_file=False,
            log_path="",
            litemode=False,          # full mode: we want OI, bid/ask, volume
            reconnect=True,
            on_message=_on_message,
            on_error=lambda m: print(f"[fyers] socket error: {str(m)[:120]}"),
            on_close=lambda m: print("[fyers] socket closed"),
            on_connect=_on_socket_connect,
        )
        threading.Thread(target=s.connect, daemon=True).start()
        _socket = s
        with _lock:
            _status["streaming"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[fyers] socket unavailable, REST polling only: {e}")
        with _lock:
            _status["streaming"] = False


def _subscribe_depth(symbols: List[str]) -> None:
    """Subscribe L2 depth for a SMALL set of contracts.

    Depth is far heavier than a price tick, and subscribing the whole chain
    would flood the socket for no benefit: the strikes anyone actually scalps
    are the handful around the money. Restricting it there keeps the stream
    responsive and the book meaningful.
    """
    if not _socket or not symbols:
        return
    # The SDK sends no depth for an index (it drops it silently), so do not
    # spend a slot on one. The caller orders the list by priority - futures
    # first, then strikes nearest the money - and the cap applies after that:
    # the futures used to sit LAST, behind 42 strikes, past a 40 cap.
    with _sub_lock:
        new = []
        for s_ in symbols:
            if s_ and s_ not in _depth_subs and not s_.endswith("-INDEX") and s_ not in new:
                new.append(s_)
        if not new:
            return
        # futures are few and always wanted; the cap applies to option books -
        # otherwise the indices fetched later (Midcap here) found it full
        cap = int(os.environ.get("FYERS_DEPTH_MAX", "100"))
        futs = [x for x in new if x.endswith("FUT")]
        opts = [x for x in new if not x.endswith("FUT")]
        room = max(0, cap - sum(1 for x in _depth_subs if not x.endswith("FUT")))
        new = futs + opts[:room]
        if not new:
            return
        try:
            _socket.subscribe(symbols=new, data_type="DepthUpdate")
            _depth_subs.update(new)
            print(f"[fyers] L2 depth subscribed for {len(new)} contract(s) ({len(_depth_subs)} total)")
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] depth subscribe failed: {e}")


def depth_status(symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    """What the depth feed is actually delivering, per contract."""
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


def audit_state(sym: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Facts about the feed behind one chain payload, for /api/data-audit."""
    now = time.time()
    ysym = FYERS_INDEX.get((sym or "").upper())
    toks = list((payload or {}).get("_tokens") or {})
    with _lock:
        tick_age = {t: now - (_ticks.get(t) or {}).get("ts", 0) for t in toks}
        dp = {t: _depth.get(t) or {} for t in toks}
        spot_age = now - (_ticks.get(ysym) or {}).get("ts", 0) if ysym else None
    opt_toks = [t for t in toks if t.endswith("CE") or t.endswith("PE")]
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
        "flow_coverage": {t: (_flow_read(t) or {}).get("coverage") for t in toks if t.endswith("FUT")},
        "lot_from_data": _lot_seen_cache.get((sym or "").upper()),
        "lot_from_book": _infer_lot_book(opt_toks),
        "lot_from_master": _master_lot(sym), "master_error": _master.get("err") or None,
        "lot_table": _LOTS.get((sym or "").upper()),
        "raw_rest": {k: v for k, v in _raw_rest.items()},
    }


def raw_debug(match: str) -> Dict[str, Any]:
    """Last raw SDK message(s) for contracts whose symbol contains `match`,
    with what the adapter stored from them."""
    now = time.time()
    m = (match or "").upper()
    out = {}
    hits = [(s, b) for s, b in list(_raw_last.items()) if not m or m in s.upper()]
    # Futures first, then depth-subscribed contracts, then most recently
    # updated — so a symbol query never hides the future behind 12 options.
    hits.sort(key=lambda sb: (not sb[0].endswith("FUT"), sb[0] not in _depth_subs,
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
                        "(futures first, then depth-subscribed, newest); narrow the symbol to see others")
    return out


def depth_snapshot() -> Dict[str, Any]:
    with _lock:
        return {k: dict(v) for k, v in _depth.items()}


def _subscribe(symbols: List[str]) -> None:
    if not _socket or not symbols:
        return
    with _sub_lock:
        new = [s for s in symbols if s not in _subscribed]
        if not new:
            return
        try:
            # Fyers caps symbols per connection; subscribe in chunks so a large
            # chain does not silently drop the tail.
            for i in range(0, len(new), 200):
                _socket.subscribe(symbols=new[i:i + 200], data_type="SymbolUpdate")
            _subscribed.update(new)
            with _lock:
                _status["subscribed"] = len(_subscribed)
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] subscribe failed: {e}")


# ── helpers ───────────────────────────────────────────────────────────
def _f(d: Dict[str, Any], *keys, default=0.0) -> float:
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "-"):
            try:
                return float(str(v).replace(",", ""))
            except Exception:
                continue
    return default


def _tick_usable(t: Dict[str, Any]) -> bool:
    """Is a streamed price good enough to use instead of a REST quote?

    In session, only if it is fresh: a stalled socket must not freeze prices.
    OUTSIDE the session ticks simply stop, so every price looks "stale" - yet
    the last tick IS the closing value and cannot change. Treating it as stale
    there would spend a REST call on every refresh, all evening, for a number
    that is already right.
    """
    if not t or not t.get("ltp"):
        return False
    g = time.gmtime(time.time() + 5 * 3600 + 1800)        # IST, host-independent
    mins = g.tm_hour * 60 + g.tm_min
    in_session = g.tm_wday < 5 and 555 <= mins < 930        # 09:15-15:30, Mon-Fri
    if not in_session:
        return True
    return (time.time() - float(t.get("ts", 0) or 0)) < _TICK_FRESH_S


def _spot(symbol: str) -> float:
    ysym = FYERS_INDEX.get(symbol.upper())
    if not ysym or not _client:
        return 0.0
    live = _ticks.get(ysym, {})
    # Use the streamed price only while it is FRESH. With no age check, a
    # stalled socket left the last tick in place indefinitely, and every chain
    # refresh reported that frozen price as spot - a quiet way for every index
    # to stop moving at once. A fresh tick costs nothing; a stale one falls
    # through to one REST quote, which is only spent while the stream is down.
    if _tick_usable(live):
        return float(live["ltp"])
    try:
        r = _count_rest("quotes") or _client.quotes({"symbols": ysym})
        for row in (r.get("d") or []):
            v = row.get("v") or {}
            return _f(v, "lp", "ltp", "last_price")
    except Exception:
        pass
    return 0.0


def india_vix() -> Optional[float]:
    live = _ticks.get(INDIA_VIX, {})
    # Fresh tick first, as with spot - a stale one would freeze VIX silently.
    if _tick_usable(live):
        return round(float(live["ltp"]), 2)
    if not _client:
        return None
    # REST fallback, cached: VIX moves slowly, and this call was made on EVERY
    # chain refresh whenever the stream lacked it - and it was not counted, so
    # it never showed up in the quota figures at all.
    hit = _fut_cache.get("__vix__")
    if hit and time.time() - hit[0] < _FUT_REST_TTL:
        return hit[1]
    try:
        r = _count_rest("quotes") or _client.quotes({"symbols": INDIA_VIX})
        for row in (r.get("d") or []):
            v = row.get("v") or {}
            val = _f(v, "lp", "ltp")
            out = round(val, 2) if val else None
            if out:
                _fut_cache["__vix__"] = (time.time(), out)
            return out
    except Exception:
        return hit[1] if hit else None
    return None


def _optionchain(req: Dict[str, Any]) -> Dict[str, Any]:
    """Fyers optionchain, reused for _CHAIN_REST_TTL seconds per request.

    The dashboard asks for the same index under several bands and expiries
    (the tile at band 4, the chain at band 12, a chosen expiry), and each was
    a separate REST call even though Fyers returns the SAME strike set for
    all of them - band is applied afterwards, locally. One response now serves
    every request for that symbol and expiry until it ages out.

    Freshness is unaffected where it matters: strike prices and OI are
    overlaid from the live stream on every build, so the cached response only
    supplies the slow fields (IV, OI change, the strike list itself).
    """
    key = f"{req.get('symbol')}|{req.get('timestamp', '')}|{req.get('strikecount')}"
    now = time.time()
    hit = _oc_cache.get(key)
    if hit and now - hit[0] < _CHAIN_REST_TTL:
        return hit[1]
    raw = _count_rest("optionchain") or _client.optionchain(data=req)
    if isinstance(raw, dict) and raw.get("s") == "ok":
        _oc_cache[key] = (now, raw)
        if len(_oc_cache) > 32:
            try:                      # threads share this dict; never let
                _oc_cache.pop(next(iter(_oc_cache)), None)   # eviction raise
            except Exception:
                pass
    return raw



# ── implied volatility (desk convention) ─────────────────────────────────
# Black-76 on the synthetic forward. The forward comes from put-call parity at
# the strikes nearest spot (F = K + C - P), which is what an index desk quotes
# against: it carries the dividend and carry the cash index does not, and it
# exists for weekly expiries that have no listed future. Each strike's vol is
# taken from its OUT-of-the-money option, which is the liquid one, and given
# to both sides - an ITM premium is mostly intrinsic, and inverting it turns
# one tick of spread into several vol points of noise.
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
    # a few-tick option has no information in its price: at 0.05 the vol
    # reads 30% or 3% depending on which side of a single tick it printed
    if price is None or price < 1.0 or price - intrinsic < 0.5 or T <= 0:
        return None
    lo, hi = 0.005, 3.0
    if _b76(F, K, T, hi, call) < price:
        return None
    for _ in range(60):                       # bisection: monotone in vol
        mid = 0.5 * (lo + hi)
        if _b76(F, K, T, mid, call) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-5:
            break
    return 0.5 * (lo + hi)


def _opt_px(r: Dict[str, Any], p: str) -> Optional[float]:
    """Mid when the quote is two-sided and tight, else last traded."""
    b, a, l = r.get(f"{p}_bid") or 0, r.get(f"{p}_ask") or 0, r.get(f"{p}_ltp") or 0
    if b > 0 and a > 0 and a >= b and (a - b) <= max(1.0, 0.05 * a):
        return 0.5 * (a + b)
    return l or None


def _solve_chain_iv(strikes: List[Dict[str, Any]], spot: float, dte_days: float) -> str:
    """Fill ce_iv / pe_iv (percent) in place when the source sent none.

    Returns "broker" when the chain already carried IV, "solved" when it was
    computed here, "none" when there was nothing to work with.
    """
    if not strikes or not spot:
        return "none"
    have = sum(1 for r in strikes if (r.get("ce_iv") or 0) > 0 or (r.get("pe_iv") or 0) > 0)
    if have >= max(3, len(strikes) // 4):
        return "broker"
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
        if v is None:                          # the OTM side is untradeable
            q = "pe" if otm_call else "ce"
            v = _b76_iv(_opt_px(r, q), F, K, T, not otm_call)
        if v is not None and 0.01 < v < 2.5:
            r["ce_iv"] = r["pe_iv"] = round(v * 100.0, 2)
            n += 1
    return "solved" if n else "none"

# ── option chain ──────────────────────────────────────────────────────
def fetch_chain(symbol: str, expiry: Optional[str], band: int) -> Dict[str, Any]:
    """Return a chain response in the server's native shape.

    Raises RuntimeError when unavailable so the caller can fall back to NSE.
    """
    if not connect():
        raise RuntimeError(status().get("last_error") or "fyers unavailable")
    sym = (symbol or "NIFTY").upper()
    ysym = FYERS_INDEX.get(sym)
    if not ysym:
        raise RuntimeError(f"no Fyers symbol mapping for {sym}")
    count = int(os.environ.get("FYERS_STRIKE_COUNT", "25"))
    try:
        # Fyers validates `timestamp` against ITS OWN expiry list - an epoch
        # computed from a date string is rejected even when it names the right
        # day, because it must match their value exactly. So: ask without a
        # timestamp first, which returns the nearest expiry AND the full
        # expiryData list, then re-ask with the matching epoch only if a
        # different expiry was requested. One extra call on a non-default
        # expiry, and none at all on the default.
        req = {"symbol": ysym, "strikecount": count}
        _ck = f"{ysym}|{str(expiry or '').strip()}"
        if expiry and _exp_ts_cache.get(_ck):
            req["timestamp"] = _exp_ts_cache[_ck]
        elif not expiry:
            _rl = _rolled.get(ysym)
            if _rl and _rl[2] == time.strftime("%Y-%m-%d"):
                req["timestamp"] = str(int(_rl[1]))      # today's expiry has settled
        raw = _optionchain(req)
        # ── after 15:30 on expiry day the default chain is a dead contract ──
        # Fyers keeps serving the settled weekly until the day changes, so
        # everything priced off it - IV (time to expiry is zero), deltas, the
        # 3-delta book, the strategy library, the next-day plan - was computed
        # on an expiry that no longer exists. Roll to the next one, and keep
        # asking for it directly for the rest of the day (one call, not two).
        if not expiry and "timestamp" not in req and isinstance(raw, dict) and raw.get("s") == "ok":
            _eps = sorted(x for x in (_exp_epoch_any(e) for e in ((raw.get("data") or {}).get("expiryData") or [])) if x)
            if len(_eps) > 1 and _settle_of(_eps[0]) <= time.time():
                _rolled[ysym] = (_eps[0], _eps[1], time.strftime("%Y-%m-%d"))
                print(f"[fyers] {sym}: today's expiry has settled - rolling the default chain to the next expiry")
                raw2 = _optionchain({"symbol": ysym, "strikecount": count, "timestamp": str(int(_eps[1]))})
                if isinstance(raw2, dict) and raw2.get("s") == "ok" and (raw2.get("data") or {}).get("optionsChain"):
                    raw = raw2
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"optionchain call failed: {e}") from e
    if not isinstance(raw, dict) or raw.get("s") != "ok":
        raise RuntimeError(f"optionchain: {(raw or {}).get('message', 'bad response')}")
    data = raw.get("data") or {}
    rows = data.get("optionsChain") or []
    # live index tick first; then the chain's own underlying row (the one
    # without an option_type). It fell back to the India VIX block before.
    spot = _spot(sym)
    if not spot:
        for r0 in rows:
            if isinstance(r0, dict) and not (r0.get("option_type") or r0.get("optionType")):
                spot = _f(r0, "ltp", "lp")
                if spot:
                    break

    by_strike: Dict[float, Dict[str, Any]] = {}
    tokens: List[str] = []
    try:                                   # newest raw rows, one of each kind
        _seen = set()
        for _r in rows:
            if not isinstance(_r, dict):
                continue
            _ot = str(_r.get("option_type") or "").upper() or "underlying"
            if _ot not in _seen:
                _seen.add(_ot)
                _raw_rest["chain:" + _ot] = dict(_r)
            if len(_seen) == 3:
                break
    except Exception:
        pass
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = _f(r, "strike_price", "strikePrice")
        opt = str(r.get("option_type") or r.get("optionType") or "").upper()
        if not k or opt not in ("CE", "PE"):
            continue          # the underlying row has no option_type
        fsym = r.get("symbol") or ""
        if fsym:
            tokens.append(fsym)
        live = _ticks.get(fsym, {}) if fsym else {}
        rec = by_strike.setdefault(k, {"strike": k})
        p = opt.lower()
        # a live tick supersedes the REST snapshot when we have one
        rec[f"{p}_ltp"] = live.get("ltp", _f(r, "ltp", "lp"))
        rec[f"{p}_oi"] = live.get("oi", _f(r, "oi", "openInterest"))
        # "oichp" is the change in PERCENT - it was the fallback here, so a
        # missing oich put a percentage where a quantity belongs
        rec[f"{p}_oi_chg"] = _f(r, "oich", "changeinOpenInterest", default=None)
        if rec[f"{p}_oi_chg"] is None:
            _po = _f(r, "prev_oi", default=None)
            rec[f"{p}_oi_chg"] = (rec[f"{p}_oi"] - _po) if (_po is not None and rec[f"{p}_oi"]) else 0.0
        rec[f"{p}_volume"] = live.get("volume", _f(r, "volume", "vol_traded_today"))
        rec[f"{p}_bid"] = live.get("bid", _f(r, "bid", "bid_price"))
        rec[f"{p}_ask"] = live.get("ask", _f(r, "ask", "ask_price"))
        # best-level quantities: the "x 1,950" beside each price
        rec[f"{p}_bid_qty"] = live.get("bid_qty", _f(r, "bid_qty", "bidQty", "bid_size", default=None))
        rec[f"{p}_ask_qty"] = live.get("ask_qty", _f(r, "ask_qty", "askQty", "ask_size", default=None))
        rec[f"{p}_iv"] = _f(r, "iv", "impliedVolatility")
        rec[f"{p}_prev_close"] = _f(r, "prev_close_price", "prevClose")
        # The v3 option chain gives the day's change (ltpch), not the previous
        # close. Without it the OI-build read (premium change with OI change)
        # never had a premium change to work with.
        # A contract with NO TRADE TODAY (every contract before 09:15) still
        # carries yesterday's last price and yesterday's change, so ltp-ltpch
        # is the close of the day BEFORE - 55 for a strike that closed at 207.
        # Its last price is its previous close.
        if not rec[f"{p}_prev_close"] and not (rec[f"{p}_volume"] or 0) and rec[f"{p}_ltp"]:
            rec[f"{p}_prev_close"] = rec[f"{p}_ltp"]
        if not rec[f"{p}_prev_close"] and r.get("ltpch") is not None:
            try:
                _lp = _f(r, "ltp", "lp")
                if _lp:
                    rec[f"{p}_prev_close"] = round(_lp - float(r.get("ltpch")), 2)
            except Exception:
                pass
        rec[f"{p}_symbol"] = fsym

    strikes = [by_strike[k] for k in sorted(by_strike)]
    if not strikes:
        raise RuntimeError("option chain returned no strikes")

    # ── units: Fyers counts SHARES, the rest of the system counts CONTRACTS ──
    # NSE's chain reports open interest, OI change and volume in contracts,
    # and every threshold, label and formula in the server and dashboard was
    # written for that: the 4,000-contract alert floor, "OI change >= 1,000
    # contracts", premium flow = LTP x volume x LOT, GEX = gamma x OI x LOT.
    # Fyers reports the same fields in shares (every value a multiple of the
    # lot), so on this feed each of those was off by the lot size - 65x on
    # Nifty: GEX and premium flow inflated 65-fold, alert floors 65x looser.
    # Converted here, at the boundary, once. Depth quantities (total bid /
    # offer, best-level size) stay in shares, as a depth window shows them.
    # Fyers always counts shares, so the conversion always applies; only the
    # LOT is in question. Read it off the data when the data can say (and
    # remember it for the session), else the lot-size table.
    #
    # Which evidence, in order. OPEN INTEREST IS NOT EVIDENCE: positions
    # opened before a lot-size revision stay in the old lot, so a contract's
    # OI mixes lots (65 and 75 share a divisor of 5) and the divisor of OI
    # read 5 - which turned every figure 13x too large. What reflects the
    # CURRENT lot is what trades now: the order book (every resting order is
    # whole lots) and today's volume.
    lot = _lot_size(sym)
    _bk = _infer_lot_book([r.get(f"{p}_symbol") for r in strikes for p in ("ce", "pe")])
    _ms = _master_lot(sym)
    if _ms and _bk and _bk % _ms != 0:     # the master must agree with the live book
        _ms = None
    lot_seen = _ms or _bk or _infer_lot(strikes) or _lot_seen_cache.get(sym)
    if lot_seen and lot_seen >= 15:
        _lot_seen_cache[sym] = lot_seen
        if lot < 5 or (lot_seen != lot and lot_seen % lot != 0):   # the table is wrong; the data is not
            if _lot_warned.get(sym) != (lot_seen, lot):
                _lot_warned[sym] = (lot_seen, lot)
                print(f"[fyers] {sym}: lot size from the feed is {lot_seen}, table says {lot} - using {lot_seen}")
            lot = lot_seen
    qty_unit = "shares"
    if lot > 1:
        for r in strikes:
            for p in ("ce", "pe"):
                for fld in ("oi", "oi_chg", "volume"):
                    v = r.get(f"{p}_{fld}")
                    if v:
                        r[f"{p}_{fld}"] = round(float(v) / lot, 2)
        qty_unit = "contracts"
    if not spot:
        best = min(strikes, key=lambda s: abs(s.get("ce_ltp", 0) - s.get("pe_ltp", 0)))
        spot = best["strike"]

    # subscribe the chain plus the index itself so ticks flow for both
    if tokens:
        _subscribe(sorted(set(tokens))[:400] + [ysym, INDIA_VIX])

    atm = min((s["strike"] for s in strikes), key=lambda k: abs(k - spot))
    atm_row = next((s for s in strikes if s["strike"] == atm), {})
    ivs = [v for v in (atm_row.get("ce_iv"), atm_row.get("pe_iv")) if v]
    atm_iv = sum(ivs) / len(ivs) if ivs else 0.0
    fut = {}

    tot_ce = sum(s.get("ce_oi", 0) for s in strikes)
    tot_pe = sum(s.get("pe_oi", 0) for s in strikes)
    # Fyers returns expiryData as epoch seconds, not a date string. Emitting
    # the raw epoch left every consumer to guess, and omitting dte entirely
    # broke every panel that prices an option - Coiled Spring, Scalp
    # Calculator, Premium Shock and Hero/Zero all read d.dte and silently fell
    # back to a default, producing plausible-looking but wrong numbers.
    def _exp_epoch(e):
        v = e.get("expiry") if isinstance(e, dict) else e
        try:
            return float(v)
        except Exception:
            return None

    # ── resolve a requested expiry against the broker's own list ──────
    # Done here because expiryData only exists after the first response. If
    # the caller asked for a specific expiry and it is not the one we were
    # given, re-request with the broker's matching epoch.
    if expiry:
        try:
            want = str(expiry).strip()
            _cands = []
            for e in (data.get("expiryData") or []):
                ep = _exp_epoch(e)
                if ep:
                    # formatted inline: _exp_str is defined further down this
                    # function, so calling it here raised "cannot access local
                    # variable" - silently caught, which meant a chosen
                    # expiry was ignored and the NEAREST chain served instead
                    _cands.append((ep, time.strftime("%d-%b-%Y",
                                                     time.gmtime(ep + 5 * 3600 + 1800))))
            match = None
            for ep, es in _cands:
                if es == want or str(int(ep)) == want:
                    match = ep
                    break
            if match is None:
                # tolerate other date spellings by comparing day/month/year
                import datetime as _dx
                for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y", "%d %b %Y"):
                    try:
                        d_ = _dx.datetime.strptime(want, fmt).date()
                    except Exception:
                        continue
                    for ep, _es in _cands:
                        g = time.gmtime(ep + 5 * 3600 + 1800)
                        if (g.tm_year, g.tm_mon, g.tm_mday) == (d_.year, d_.month, d_.day):
                            match = ep
                            break
                    break
            cur = _exp_epoch((data.get("expiryData") or [{}])[0]) if data.get("expiryData") else None
            if match:
                # The NEAREST expiry is what Fyers serves with no timestamp, so
                # store it as "" and keep asking that way: the tiles and the
                # background poller request the default, and pinning the epoch
                # here put the same data in two cache slots - two calls for one
                # answer. Only a non-nearest expiry needs its epoch remembered.
                _exp_ts_cache[f"{ysym}|{want}"] = "" if (cur and int(match) == int(cur)) else str(int(match))
            # Already served the requested expiry (cache hit): nothing to do.
            if match and req.get("timestamp") == str(int(match)):
                match = None
            if match and cur and int(match) != int(cur):
                raw2 = _optionchain({"symbol": ysym, "strikecount": count, "timestamp": str(int(match))})
                d2 = (raw2 or {}).get("data") or {}
                if d2.get("optionsChain"):
                    data = d2
                else:
                    print(f"[fyers] expiry {want} not served; using the nearest instead")
        except Exception as _ee:
            print(f"[fyers] expiry resolve skipped: {_ee}")

    exp_epochs = sorted(x for x in (_exp_epoch(e) for e in (data.get("expiryData") or [])) if x)
    def _exp_str(ep):
        g = time.gmtime(ep + 5 * 3600 + 1800)
        return time.strftime("%d-%b-%Y", g)
    expiries = [_exp_str(x) for x in exp_epochs]

    # Futures quote, moved down to here. It was called before exp_epochs was
    # assigned, so it raised UnboundLocalError on every single call - and
    # because the call sat inside a bare `except Exception: pass`, it failed
    # silently and the futures basis has simply never been populated on the
    # broker path. A swallowed exception is how a feature stays broken for
    # months without anyone seeing an error.
    try:
        fut = _futures_quote(sym, exp_epochs)
        if qty_unit == "contracts" and fut:
            fut = dict(fut)
            for fk in ("futures_oi", "futures_oi_chg", "futures_volume", "futures_far_oi"):
                if fut.get(fk):
                    fut[fk] = round(float(fut[fk]) / lot, 2)
        # The futures legs need an ORDINARY tick subscription. Total buy/sell
        # quantity rides on the regular tick, and they were only ever added to
        # the depth subscription - which the broker may not serve - so no tick
        # arrived for them, _depth stayed empty, and both rows showed a dash.
        # Placed here because `fut` does not exist until this line.
        try:
            _subscribe([x for x in ((fut or {}).get("futures_symbol"),
                                    (fut or {}).get("futures_far_symbol")) if x])
        except Exception as _fse:
            print(f"[fyers] futures tick subscribe skipped: {_fse}")
    except Exception as _fe:
        print(f"[fyers] futures quote failed: {_fe}")

    def _dte_from(ep):
        """Fractional days to 15:30 IST on expiry day.

        Fractional, not whole days: on expiry day a whole-day figure is 0 from
        09:15 to 15:30, which pins T to a floor and freezes every Greek for the
        entire session - exactly when theta moves fastest.
        """
        if not ep:
            return 1.0
        g = time.gmtime(ep + 5 * 3600 + 1800)
        day_start = ep - (g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec)
        settle = day_start + 15 * 3600 + 30 * 60          # 15:30 IST
        return max((settle - time.time()) / 86400.0, 1.0 / (24 * 60))

    near_epoch = None
    if expiry:
        for ep, s in zip(exp_epochs, expiries):
            if str(expiry) in (s, str(ep)):
                near_epoch = ep
                break
    if near_epoch is None and exp_epochs:
        # the first expiry that has not settled yet
        near_epoch = next((ep for ep in exp_epochs if _settle_of(ep) > time.time()), exp_epochs[0])
    dte_frac = _dte_from(near_epoch)

    # ── implied vol where the broker sends none ───────────────────────
    # The Fyers option chain carries prices and OI, not IV. Every panel that
    # prices an option - greeks, skew, the smile, GEX, portfolio risk, the
    # 3-delta picker, the risk reversal - read ce_iv/pe_iv as 0 and fell back
    # to a flat 12-13%, which flattens the smile and mis-states every delta.
    # Solved here from prices already in hand: no extra request.
    iv_src = _solve_chain_iv(strikes, spot, dte_frac)
    if iv_src == "solved":
        atm_row = next((s for s in strikes if s["strike"] == atm), {})
        ivs = [v for v in (atm_row.get("ce_iv"), atm_row.get("pe_iv")) if v]
        atm_iv = sum(ivs) / len(ivs) if ivs else atm_iv

    gaps = sorted({round(b - a) for a, b in zip(sorted(by_strike), sorted(by_strike)[1:]) if b > a})
    strike_gap = gaps[0] if gaps else 50

    # ── L2 depth subscription ─────────────────────────────────────────
    # Placed here, not up with the price subscription, because it needs BOTH
    # atm and strike_gap and those are computed above. Referencing them
    # earlier raised UnboundLocalError on every chain fetch, and because the
    # adapter catches and falls back to NSE, the dashboard kept working while
    # the broker path was dead - a failure that hides itself.
    if _status.get("streaming") and strikes:
        try:
            near_syms = []
            _strike_of = {r.get(f"{p}_symbol"): r["strike"] for r in strikes for p in ("ce", "pe")}
            for r in strikes:
                # +/- 10 strikes, not 5. A 0.90-delta option sits roughly
                # seven strikes from the money on a weekly Nifty, so the old
                # band stopped short of exactly the strikes whose book is
                # worth reading: deep ITM is where size is worked quietly,
                # because the extrinsic value is small and a large order moves
                # the premium far less than it would at the money.
                #
                # Ten strikes each side is about 60 extra subscriptions and
                # the socket carries it comfortably.
                if abs(r["strike"] - atm) <= strike_gap * int(
                        os.environ.get("FYERS_DEPTH_STRIKES", "10")):
                    for p in ("ce", "pe"):
                        fs = r.get(f"{p}_symbol")
                        if fs:
                            near_syms.append(fs)
            # include the futures legs: their book is what the bid/ask panel
            # reads for the current and next series
            # `fut` is the futures-quote result; `data` is the chain source
            # and never carries these keys. Reading them from `data` returned
            # undefined and the futures legs were silently never subscribed.
            fut_syms = [x for x in (fut.get("futures_symbol"),
                                    fut.get("futures_far_symbol")) if x]
            near_syms.sort(key=lambda fs_: abs(_strike_of.get(fs_, atm) - atm))
            _subscribe_depth(fut_syms + near_syms)
        except Exception as _e:
            print(f"[fyers] depth subscribe skipped: {_e}")

    # Support and resistance: the largest put-OI and call-OI strikes within the
    # band. The dashboard reads these as OBJECTS (support.strike), not numbers -
    # omitting them threw "Cannot read properties of undefined" and killed the
    # whole render pipeline, since one failure there stops every panel after it.
    lo_b, hi_b = atm - band * strike_gap, atm + band * strike_gap
    nearby = [s for s in strikes if lo_b <= s["strike"] <= hi_b] or strikes
    resistance = max(nearby, key=lambda s: s.get("ce_oi", 0))
    support = max(nearby, key=lambda s: s.get("pe_oi", 0))

    out = {
        "symbol": sym,
        "underlying_value": spot,
        "support": dict(support),
        # every field below is dereferenced by the dashboard WITHOUT a guard,
        # so omitting any one of them throws and stops the render pipeline
        "flags": _classify_buildups(nearby, atm, strike_gap),
        "sentiment": _infer_sentiment(nearby, atm, _max_pain(strikes),
                                      (tot_pe / tot_ce) if tot_ce else None),
        "skew_note": "",
        "expiries_data": {},
        "gainers": [],
        "symbols": [],
        "ideas": [],
        "strategies": [],
        "payout_distribution": [],
        "resistance": dict(resistance),
        "nearby": nearby,
        "atm": atm,
        "atm_iv": round(atm_iv, 2),
        "iv_source": iv_src,
        "expiry": (expiry if expiry in expiries else None) or (_exp_str(near_epoch) if near_epoch else (expiries[0] if expiries else None)),
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
        # OI / OI change / volume are in CONTRACTS on both feeds now; depth
        # quantities are shares. Ticks carry shares, so the dashboard divides
        # tick volume by this.
        "oi_unit": qty_unit,
        "qty_per_contract": lot if qty_unit == "contracts" else 1,
        # every option-pricing panel reads these
        "dte": round(dte_frac, 4),
        "dte_days": int(dte_frac),
        "dte_trading": round(dte_frac * (252.0 / 365.0), 4),
        "expiry_epoch": near_epoch,
        # futures basis and OI build-up feed the trend engine
        "futures_price": fut.get("futures_price"),
        "futures_oi": fut.get("futures_oi"),
        "futures_oi_chg": fut.get("futures_oi_chg"),
        "futures_prev_close": fut.get("futures_prev_close"),
        "futures_volume": fut.get("futures_volume"),
        "futures_far_price": fut.get("futures_far_price"),
        # best bid / offer, price and size, for both series - live from the
        # socket (the same ticks the price comes from)
        **{f"futures{suf}_{k2}": (_ticks.get(fut.get(fsk)) or {}).get(k1)
           for suf, fsk in (("", "futures_symbol"), ("_far", "futures_far_symbol"))
           for k1, k2 in (("bid", "bid"), ("ask", "ask"), ("bid_qty", "bid_qty"), ("ask_qty", "ask_qty"))},
        "futures_far_oi": fut.get("futures_far_oi"),
        # days to the FUTURES expiry (the monthly), not the weekly option
        # expiry: annualising the basis over a 1-day weekly turned a 0.26%
        # premium into "+91% a year"
        "futures_dte": _monthly_dte(exp_epochs),
        "india_vix": india_vix(),
        # The index's previous close, from the stream, so a tile can show the
        # DAY's change rather than the change since its own last refresh.
        "spot_prev_close": (_ticks.get(ysym) or {}).get("prev_close"),
        # the exchange's own day open / high / low for the index, from the
        # socket - the server's version only starts counting when it starts
        "session_ohlc": _session_from_ticks(ysym),
        "timestamp": time.strftime("%d-%b-%Y %H:%M:%S"),
        "_source": "fyers",
        "_live": bool(_status.get("streaming")),
        "depth": {
            "index": _depth.get(ysym),
            # Keyed with an INTEGER strike. _f() returns floats, so the key
            # came out as "22300.0CE" while every consumer asks for "22300CE" -
            # 102 strikes of depth arrived and none could ever be found. Nifty
            # and BankNifty strikes are whole numbers, so int() loses nothing.
            "strikes": {f'{int(round(r["strike"]))}{p.upper()}': _depth[r[f"{p}_symbol"]]
                        for r in strikes for p in ("ce", "pe")
                        if r.get(f"{p}_symbol") and r[f"{p}_symbol"] in _depth},
            # Futures books, keyed plainly. They were absent from the payload
            # entirely, so the current and next series could never show a
            # book however the client asked for it.
            "futures": {
                "near": _depth.get(fut.get("futures_symbol")),
                "far": _depth.get(fut.get("futures_far_symbol")),
            },
        },
        "_tokens": {r[f"{p}_symbol"]: {"sym": sym, "k": r["strike"], "side": p.upper()}
                    for r in strikes for p in ("ce", "pe") if r.get(f"{p}_symbol")},
        # aggressor flow per contract, keyed like depth ("22300CE"), plus the
        # two futures - buy/sell volume taken in the last 5 and 15 minutes
        # and over the session
        "flow": {
            "strikes": {k: v for k, v in
                        ((f'{int(round(r["strike"]))}{p.upper()}', _flow_read(r.get(f"{p}_symbol")))
                         for r in strikes for p in ("ce", "pe")) if v},
            "futures": {"near": _flow_read(fut.get("futures_symbol")),
                        "far": _flow_read(fut.get("futures_far_symbol"))},
        },
    }
    out["_tokens"][ysym] = {"sym": sym, "spot": True}
    # Session state, and no crossed quotes. In the 09:00-09:08 pre-open the
    # futures carry auction orders - a 23,371 bid against a 22,650 offer - and
    # options carry yesterday's last prices; neither is a market to read.
    _g = time.gmtime(time.time() + 19800)
    _mm = _g.tm_hour * 60 + _g.tm_min
    out["session_state"] = ("closed" if _g.tm_wday >= 5 or _mm >= 930 or _mm < 540
                            else "pre-open" if _mm < 555 else "open")
    for _pre in ("futures", "futures_far"):
        _b, _a = out.get(f"{_pre}_bid"), out.get(f"{_pre}_ask")
        if _b and _a and float(_b) >= float(_a):
            for _k in ("bid", "ask", "bid_qty", "ask_qty"):
                out[f"{_pre}_{_k}"] = None
    # futures ticks were streamed but had no token entry, so the browser
    # dropped them and the futures price only moved once per poll
    for _which, _fs in (("near", fut.get("futures_symbol")), ("far", fut.get("futures_far_symbol"))):
        if _fs:
            out["_tokens"][_fs] = {"sym": sym, "fut": _which}
    with _lock:
        _status["last_chain"] = time.time()
    return out



def _classify_buildups(strikes, atm, gap, threshold_pct=8.0):
    """Same shape and rule as the NSE source's classify_buildups().

    The dashboard reads data.flags without a guard, so an adapter that omits
    it throws and takes down the whole render pipeline. Descriptive, not
    predictive: it labels what a large OI change on each side conventionally
    implies, nothing more.
    """
    flags = []
    for s in strikes:
        ce_oi = s.get("ce_oi") or 0
        pe_oi = s.get("pe_oi") or 0
        ce_chg = s.get("ce_oi_chg") or 0
        pe_chg = s.get("pe_oi_chg") or 0
        if ce_oi > 0 and abs(ce_chg) / max(ce_oi, 1) * 100 >= threshold_pct:
            flags.append({"strike": s["strike"], "side": "CE", "oi_chg": ce_chg,
                          "label": ("Fresh call writing (resistance building)"
                                    if ce_chg > 0 else
                                    "Call unwinding (resistance weakening)")})
        if pe_oi > 0 and abs(pe_chg) / max(pe_oi, 1) * 100 >= threshold_pct:
            flags.append({"strike": s["strike"], "side": "PE", "oi_chg": pe_chg,
                          "label": ("Fresh put writing (support building)"
                                    if pe_chg > 0 else
                                    "Put unwinding (support weakening)")})
    return flags


def _infer_sentiment(strikes, atm, max_pain, pcr):
    """Same scoring rule as the NSE source, so the two agree.

    Descriptive, not predictive: it counts which side is building and which is
    unwinding, adds a half point for where max pain sits, and needs a clear
    margin before calling a direction at all. The dashboard reads this with
    .toLowerCase() and no guard, so omitting it throws.
    """
    bull = bear = 0.0
    for s in strikes:
        ce_chg = s.get("ce_oi_chg") or 0
        pe_chg = s.get("pe_oi_chg") or 0
        if pe_chg > 0:
            bull += 0.25          # put writing = support building
        if ce_chg > 0:
            bear += 0.25          # call writing = resistance building
        if ce_chg < 0:
            bull += 0.15          # call unwinding = resistance weakening
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

FUT_SUFFIX = {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY",
              "FINNIFTY": "FINNIFTY", "MIDCPNIFTY": "MIDCPNIFTY"}


def _futures_quote(symbol: str, exp_epochs: List[float]) -> Dict[str, Any]:
    """Near-month futures price, OI and previous close.

    The trend engine reads basis (futures minus spot) and the futures OI
    build-up: both are among the cleanest positioning signals available, and
    both were silently absent, so those inputs simply never contributed.
    Fyers option-chain symbols encode the expiry, so the future for the same
    series is derivable from the same date.
    """
    base = FUT_SUFFIX.get(symbol.upper())
    if not base or not _client or not exp_epochs:
        return {}
    # ── which contracts ──────────────────────────────────────────────
    # The current and next MONTHLY futures. These were derived from the first
    # two WEEKLY option expiries, so whenever both weeklies fell in the same
    # month (13 and 20 Oct, say) both legs resolved to the same contract and
    # "next month" silently showed the current month again.
    mons = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    syms: List[str] = []
    for ep in exp_epochs:
        g = time.gmtime(ep + 5 * 3600 + 1800)
        fsym = f"NSE:{base}{str(g.tm_year)[2:]}{mons[g.tm_mon - 1]}FUT"
        if fsym not in syms:
            syms.append(fsym)
        if len(syms) == 2:
            break
    if len(syms) == 1:
        # only one month among the listed expiries: derive the next calendar month
        g = time.gmtime(exp_epochs[0] + 5 * 3600 + 1800)
        y, m = g.tm_year + (1 if g.tm_mon == 12 else 0), (g.tm_mon % 12) + 1
        syms.append(f"NSE:{base}{str(y)[2:]}{mons[m - 1]}FUT")

    # ── one request, cached ──────────────────────────────────────────
    # Both contracts in ONE quotes call (comma-separated) instead of two, and
    # the result kept for _FUT_REST_TTL seconds rather than re-requested on
    # every chain refresh. Futures OI and previous close move slowly; the PRICE
    # moves fast, but that arrives on the live stream and is layered on below,
    # so caching costs no freshness where it matters.
    ckey = ",".join(syms)
    now = time.time()
    hit = _fut_cache.get(ckey)
    if hit and now - hit[0] < _FUT_REST_TTL:
        rows = hit[1]
    else:
        rows = {}
        try:
            r = _count_rest("quotes") or _client.quotes({"symbols": ckey})
            for row in (r.get("d") or []):
                v = row.get("v") or {}
                nm = row.get("n") or v.get("symbol") or v.get("short_name") or ""
                if nm in syms and _f(v, "lp", "ltp"):
                    rows[nm] = v
                    _raw_rest["quote:" + nm] = dict(v)
            # Some responses omit the symbol name; fall back to request order.
            if not rows:
                for k2, row in zip(syms, (r.get("d") or [])):
                    v = row.get("v") or {}
                    if _f(v, "lp", "ltp"):
                        rows[k2] = v
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] futures quote failed: {e}")
        if rows:
            _fut_cache[ckey] = (now, rows)
        elif hit:
            rows = hit[1]            # keep the last good answer through a blip

    def _leg(fsym):
        v = dict(rows.get(fsym) or {})
        # live stream overlay: price, OI and volume as fresh as the last tick
        t = _ticks.get(fsym) or {}
        if _tick_usable(t):
            if t.get("ltp"):
                v["lp"] = t["ltp"]
            if t.get("oi"):
                v["open_interest"] = t["oi"]
            if t.get("volume"):
                v["volume"] = t["volume"]
        return v

    out = {}
    near, far = _leg(syms[0]), _leg(syms[1]) if len(syms) > 1 else {}
    px = _f(near, "lp", "ltp")
    if px:
        out.update({
            "futures_price": px,
            "futures_oi": _f(near, "open_interest", "oi") or None,
            "futures_oi_chg": _f(near, "oich", "open_interest_change") or None,
            # The SYMBOL, so the client can look up this leg's depth book.
            "futures_symbol": syms[0],
            "futures_prev_close": _f(near, "prev_close_price", "prev_close") or None,
            "futures_volume": _f(near, "volume", "vol_traded_today") or None,
        })
    fpx = _f(far, "lp", "ltp")
    if fpx and len(syms) > 1:
        out["futures_far_price"] = fpx
        out["futures_far_oi"] = _f(far, "open_interest", "oi") or None
        out["futures_far_symbol"] = syms[1]
    return out


_rolled: Dict[str, Any] = {}       # index symbol -> (settled epoch, next epoch, day)


def _exp_epoch_any(e) -> Optional[float]:
    v = e.get("expiry") if isinstance(e, dict) else e
    try:
        return float(v)
    except Exception:
        return None


def _session_from_ticks(ysym: Optional[str]) -> Dict[str, Any]:
    t = _ticks.get(ysym or "") or {}
    o, h, l, c = t.get("open"), t.get("high"), t.get("low"), t.get("ltp")
    if not (o and h and l and c):
        return {}
    return {"date": time.strftime("%Y-%m-%d"), "open": o, "high": h, "low": l, "close": c,
            "prev_close": t.get("prev_close"), "src": "exchange"}


def _settle_of(ep: float) -> float:
    """15:30 IST on the expiry's own day, as an epoch."""
    g = time.gmtime(ep + 19800)
    day_start = ep - (g.tm_hour * 3600 + g.tm_min * 60 + g.tm_sec)
    return day_start + 15 * 3600 + 30 * 60


def _monthly_dte(exp_epochs: List[float]) -> Optional[float]:
    """Fractional days to the current monthly expiry (15:30 IST).

    The monthly is the last expiry of the month the nearest expiry falls in.
    Fyers lists a handful of upcoming expiries, which may stop short of the
    monthly, so the last Tuesday of that month (the index monthly expiry day)
    is the fallback; a listed expiry within a week of it - a holiday moving it
    a day earlier - is preferred.
    """
    if not exp_epochs:
        return None
    try:
        import calendar
        g = time.gmtime(exp_epochs[0] + 19800)
        y, m = g.tm_year, g.tm_mon
        last = calendar.monthrange(y, m)[1]
        d = last
        while calendar.weekday(y, m, d) != 1:          # Tuesday
            d -= 1
        lt = calendar.timegm((y, m, d, 15, 30, 0)) - 19800
        same = [e for e in exp_epochs if time.gmtime(e + 19800).tm_mon == m and abs(e - lt) < 7 * 86400]
        ep = max(same) if same else lt
        gg = time.gmtime(ep + 19800)
        settle = calendar.timegm((gg.tm_year, gg.tm_mon, gg.tm_mday, 15, 30, 0)) - 19800
        return round(max((settle - time.time()) / 86400.0, 1.0 / 1440), 4)
    except Exception:
        return None


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


# fallback only - the chain's own quantities decide (see _infer_lot)
_LOTS = {"NIFTY": 65, "BANKNIFTY": 30, "FINNIFTY": 60, "MIDCPNIFTY": 120, "SENSEX": 20}


_lot_seen_cache: Dict[str, int] = {}
_lot_warned: Dict[str, Any] = {}


def _infer_lot(strikes) -> Optional[int]:
    """The lot size, read off the quantities themselves.

    On a feed that counts shares every OI and volume figure is a whole number
    of lots, so their greatest common divisor IS the lot. Returns None when
    the numbers are not lot multiples (a contracts feed) or too few to tell.
    """
    import math as _m
    vals = []
    for r in strikes:
        for k in ("ce_volume", "pe_volume"):      # volume only - OI mixes old lots
            v = r.get(k)
            if v and v > 0:
                vals.append(int(round(v)))
    if len(vals) < 12:
        return None
    g = 0
    for v in vals:
        g = _m.gcd(g, v)
        if g == 1:
            return None
    return g if 15 <= g <= 5000 else None


def _infer_lot_book(symbols) -> Optional[int]:
    """The lot from the five-level books: every resting order is whole lots."""
    import math as _m
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
        g = _m.gcd(g, v)
        if g < 15:
            return None
    return g if g <= 5000 else None


# ── Fyers' own symbol master: the exchange lot per contract ──────────────
# A public CSV (no API quota), downloaded once a day in the background.
_MASTER_URL = os.environ.get("FYERS_SYMBOL_MASTER", "https://public.fyers.in/sym_details/NSE_FO.csv")
_master = {"day": "", "lots": {}, "busy": False, "err": ""}


def _master_load() -> None:
    if _master["busy"]:
        return
    _master["busy"] = True
    try:
        import csv
        import io
        import urllib.request
        with urllib.request.urlopen(_MASTER_URL, timeout=30) as r:
            text = r.read().decode("utf-8", "ignore")
        lots = {}
        for row in csv.reader(io.StringIO(text)):
            # Fytoken, details, instrument type, MIN LOT SIZE, tick, ISIN,
            # session, updated, expiry, SYMBOL TICKER, ... underlying symbol
            if len(row) < 10:
                continue
            tick = row[9].strip()
            if not tick.endswith("FUT"):
                continue
            try:
                lt = int(float(row[3]))
            except Exception:
                continue
            if 1 <= lt <= 100000:
                lots[tick] = lt
        if lots:
            _master.update({"day": time.strftime("%Y-%m-%d"), "lots": lots, "err": ""})
            print(f"[fyers] symbol master: lot sizes for {len(lots)} futures")
    except Exception as e:  # noqa: BLE001
        _master["err"] = str(e)[:120]
        _master["day"] = time.strftime("%Y-%m-%d")      # do not retry all day
    finally:
        _master["busy"] = False


def _master_lot(sym: str) -> Optional[int]:
    if _master["day"] != time.strftime("%Y-%m-%d") and not _master["busy"]:
        threading.Thread(target=_master_load, daemon=True).start()
        return None
    base = FUT_SUFFIX.get((sym or "").upper())
    if not base:
        return None
    for tick, lt in _master["lots"].items():
        if tick.startswith(f"NSE:{base}") and tick[len(f"NSE:{base}"):][:2].isdigit():
            return lt if lt >= 5 else None
    return None


def _lot_size(symbol: str) -> int:
    # nse_lot_sizes answers 1 when it cannot reach NSE (no NSE session on the
    # broker path) - and a lot of 1 made every per-lot figure in the system a
    # per-SHARE figure. Only a plausible answer is taken.
    try:
        import nse_lot_sizes
        v = nse_lot_sizes.get_lot_size(symbol)
        if v and int(v) >= 5:
            return int(v)
    except Exception:
        pass
    return _LOTS.get(symbol.upper(), 65)


# ── historical candles: real OHLC with VOLUME ─────────────────────────
_history_cache: Dict[tuple, Any] = {}
_history_lock = threading.Lock()
# Keys currently being fetched, so concurrent callers wait on the one call in
# flight rather than each starting their own. The chain path has had this from
# the start; history did not, and the gap only showed once a request became
# slow enough for the next poll to arrive before it finished.
_history_inflight: Dict[tuple, threading.Event] = {}
_rest_count = {"n": 0, "day": ""}


def _count_rest(kind: str) -> None:
    """Track REST usage so the quota is observable, not guessed at."""
    today = time.strftime("%Y-%m-%d")
    with _lock:
        if _rest_count["day"] != today:
            _rest_count.update({"n": 0, "day": today, "by": {}})
        _rest_count["n"] += 1
        # per-kind breakdown, so it is visible WHICH calls the quota goes on
        by = _rest_count.setdefault("by", {})
        by[kind] = by.get(kind, 0) + 1
        n = _rest_count["n"]
    if n in (1000, 5000, 10000, 25000, 50000, 75000, 90000):
        print(f"[fyers] REST calls today: {n:,} (limit ~100,000)")


def rest_usage() -> Dict[str, Any]:
    with _lock:
        out = dict(_rest_count)
        out["by"] = dict(_rest_count.get("by", {}))
        return out


def fetch_candles(symbol: str, interval_min: int = 1, days: int = 1,
                  on_date: str = None) -> List[Dict[str, Any]]:
    """Intraday candles from Fyers history.

    This is the piece NSE cannot give: genuine OHLC with TRADED VOLUME, rather
    than candles reconstructed from one spot sample per minute. It also covers
    the whole session regardless of when this server started, which removes the
    backfill problem entirely.
    """
    if not connect():
        return []
    ysym = FYERS_INDEX.get(symbol.upper())
    if not ysym:
        return []
    # Cache history hard. Without this, EVERY browser tab polling the chart
    # every 5 seconds hit Fyers directly - three tabs alone burned ~14,000
    # requests a session, and simultaneous polls could burst past the
    # per-second limit even while the daily total looked comfortable.
    # Completed candles never change, so re-fetching them is pure waste; the
    # live edge of the chart comes from the WebSocket, not from here.
    # on_date is part of the cache key: a past session is a different series
    # from today's, and sharing a key would serve one for the other.
    key = (ysym, interval_min, days, on_date or "")
    # A completed past session never changes, so it can be cached for the
    # whole run rather than for thirty seconds. Today's data still needs the
    # short TTL because its last candle is still forming.
    ttl = 86400.0 if on_date else float(os.environ.get("FYERS_HISTORY_TTL", "60"))
    # ── cache, then coalesce ──────────────────────────────────────────
    # A cold history call can take many seconds. The chart polls every five,
    # so without coalescing each poll started ANOTHER call for the same date:
    # threads accumulated, the Fyers per-minute history limit tripped, and the
    # server stopped answering anything at all. One caller fetches; the rest
    # wait on its result.
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
        # someone else is already fetching this exact series
        waiter.wait(timeout=30)
        with _history_lock:
            hit = _history_cache.get(key)
        return hit[1] if hit else []
    res = {1: "1", 3: "3", 5: "5", 15: "15", 30: "30", 60: "60"}.get(interval_min, "1")
    # Dates must be IST, not the server's local timezone - a machine running
    # in UTC would otherwise ask for the wrong day either side of midnight.
    def _ist_date(offset_days=0):
        g = time.gmtime(time.time() + 5 * 3600 + 1800 - offset_days * 86400)
        return f"{g.tm_year:04d}-{g.tm_mon:02d}-{g.tm_mday:02d}"

    today = on_date or _ist_date(0)
    # Ask for a couple of days so a holiday or a weekend still returns the last
    # session, then filter to the day we actually want. Requesting only today
    # returns nothing before the open, and requesting a range without filtering
    # drew YESTERDAY'S candles alongside today's - which is what this fixes.
    # For an explicit date ask for that day only; the +/- day padding exists
    # to survive weekends when asking for "today", and here it would drag in
    # the neighbouring session.
    frm = on_date or _ist_date(max(1, days))
    if on_date:
        _y, _m, _d = (int(x) for x in on_date.split("-"))
        # midday avoids any timezone rounding landing on the wrong day
        want_yday = time.gmtime(time.mktime((_y, _m, _d, 12, 0, 0, 0, 0, 0))).tm_yday
    else:
        want_yday = time.gmtime(time.time() + 5 * 3600 + 1800).tm_yday
    try:
        r = _count_rest("history") or _client.history({
            "symbol": ysym, "resolution": res, "date_format": "1",
            "range_from": frm, "range_to": today, "cont_flag": "1",
        })
        if not isinstance(r, dict) or r.get("s") != "ok":
            print(f"[fyers] history: {(r or {}).get('message', 'bad response')}")
            return []
        out, dropped = [], 0
        for c in (r.get("candles") or []):
            if len(c) < 6:
                continue
            ts = int(c[0])
            g = time.gmtime(ts + 5 * 3600 + 1800)
            if g.tm_yday != want_yday:
                dropped += 1
                continue                 # a previous session: not today's chart
            mins = g.tm_hour * 60 + g.tm_min
            if not _in_session(mins):
                dropped += 1
                continue                 # pre-open or past the 15:40 CAS close
            out.append({"t": ts, "o": float(c[1]), "h": float(c[2]),
                        "l": float(c[3]), "c": float(c[4]), "v": float(c[5]), "n": 1})
        if dropped:
            print(f"[fyers] history: kept {len(out)} of today's candles, "
                  f"dropped {dropped} from other sessions or outside market hours")
        # Never cache an EMPTY result for a day. A transient failure - a rate
        # limit, a dropped connection, a token that expired mid-session -
        # would otherwise be remembered for twenty-four hours, and that date
        # would stay blank no matter how many times it was reselected. Empty
        # results get the short TTL so the next attempt actually retries.
        with _history_lock:
            _history_cache[key] = (time.time() if out else 0.0, out)
            _ev = _history_inflight.pop(key, None)
        if _ev:
            _ev.set()
        if on_date:
            print(f"[fyers] history {on_date}: {len(out)} candle(s)"
                  f"{' — EMPTY, will retry' if not out else ''}")
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[fyers] history failed: {e}")
        return []
    finally:
        # Waiters must be released even when the fetch raised, or every other
        # caller blocks for the full timeout and the pile-up this was meant to
        # prevent happens anyway - just more slowly.
        with _history_lock:
            _ev2 = _history_inflight.pop(key, None)
        if _ev2:
            _ev2.set()


def constituent_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Live quotes for a list of NSE equity symbols.

    This is what lets the index-contributors widget work on a broker feed at
    all. Previously it could only be built by scraping NSE, so choosing a
    broker source silently broke it - the data was right there in the feed and
    simply was not being asked for.

    Fyers accepts a comma-separated symbol list, so fifty constituents cost a
    handful of calls rather than fifty.
    """
    if not connect() or not symbols:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    syms = [f"NSE:{s.strip().upper()}-EQ" for s in symbols if s and s.strip()]
    CHUNK = 40                      # keep each request comfortably small
    for i in range(0, len(syms), CHUNK):
        batch = syms[i:i + CHUNK]
        try:
            r = _count_rest("quotes") or _client.quotes({"symbols": ",".join(batch)})
        except Exception as e:  # noqa: BLE001
            print(f"[fyers] constituent quotes failed: {e}")
            continue
        if not isinstance(r, dict) or r.get("s") != "ok":
            print(f"[fyers] quotes: {(r or {}).get('message', 'bad response')}")
            continue
        for row in (r.get("d") or []):
            v = row.get("v") or {}
            name = (row.get("n") or v.get("symbol") or "")
            name = name.replace("NSE:", "").replace("-EQ", "")
            if not name:
                continue
            ltp = _f(v, "lp", "ltp", "last_price")
            prev = _f(v, "prev_close_price", "prev_close", "c")
            if not ltp:
                continue
            chg = ltp - prev if prev else 0.0
            out[name] = {
                "symbol": name,
                "lastPrice": ltp,
                "previousClose": prev,
                "change": round(chg, 2),
                "pChange": round(chg / prev * 100, 2) if prev else 0.0,
                "totalTradedVolume": _f(v, "volume", "vol_traded_today"),
                "totalTradedValue": _f(v, "volume", "vol_traded_today") * ltp,
                "dayHigh": _f(v, "high_price", "h"),
                "dayLow": _f(v, "low_price", "l"),
                "open": _f(v, "open_price", "o"),
            }
    return out


def market_depth(symbol: str) -> Dict[str, Any]:
    """Level-2 depth — the input a genuine order-flow read needs."""
    if not connect():
        return {}
    ysym = FYERS_INDEX.get(symbol.upper(), symbol)
    try:
        r = _count_rest("depth") or _client.depth({"symbol": ysym, "ohlcv_flag": "1"})
        return r.get("d", {}) if isinstance(r, dict) else {}
    except Exception:
        return {}


def shutdown() -> None:
    global _socket
    try:
        if _socket:
            _socket.close_connection()
    except Exception:
        pass
    _socket = None
    with _lock:
        _status.update(streaming=False, connected=False)
