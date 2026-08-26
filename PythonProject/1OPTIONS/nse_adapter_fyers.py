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

_client: Optional[Any] = None
_socket: Optional[Any] = None
_lock = threading.Lock()
_ticks: Dict[str, Dict[str, Any]] = {}      # fyers symbol -> latest tick
_subscribed: set = set()
_tick_sink = None
_last_push = 0.0
_dirty: Dict[str, Dict[str, Any]] = {}
_status: Dict[str, Any] = {
    "connected": False, "streaming": False, "last_error": "",
    "ticks": 0, "subscribed": 0, "last_chain": 0.0, "token_age_h": None,
}


# ── configuration ─────────────────────────────────────────────────────
_ENV_NAMES = (".env", "env", ".env.local", "env.txt", ".env.txt")
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
        d["ts"] = time.time()
        with _lock:
            _ticks[sym] = {**_ticks.get(sym, {}), **d}
            _status["ticks"] += 1
            _status["streaming"] = True
            _dirty[sym] = {k: v for k, v in _ticks[sym].items()
                           if k in ("ltp", "oi", "volume", "bid", "ask")}
        _push_ticks()
    except Exception:
        pass


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
            on_connect=lambda: print("[fyers] market-data socket connected"),
        )
        threading.Thread(target=s.connect, daemon=True).start()
        _socket = s
        with _lock:
            _status["streaming"] = True
    except Exception as e:  # noqa: BLE001
        print(f"[fyers] socket unavailable, REST polling only: {e}")
        with _lock:
            _status["streaming"] = False


def _subscribe(symbols: List[str]) -> None:
    if not _socket or not symbols:
        return
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


def _spot(symbol: str) -> float:
    ysym = FYERS_INDEX.get(symbol.upper())
    if not ysym or not _client:
        return 0.0
    live = _ticks.get(ysym, {})
    if live.get("ltp"):
        return float(live["ltp"])
    try:
        r = _client.quotes({"symbols": ysym})
        for row in (r.get("d") or []):
            v = row.get("v") or {}
            return _f(v, "lp", "ltp", "last_price")
    except Exception:
        pass
    return 0.0


def india_vix() -> Optional[float]:
    live = _ticks.get(INDIA_VIX, {})
    if live.get("ltp"):
        return round(float(live["ltp"]), 2)
    if not _client:
        return None
    try:
        r = _client.quotes({"symbols": INDIA_VIX})
        for row in (r.get("d") or []):
            v = row.get("v") or {}
            val = _f(v, "lp", "ltp")
            return round(val, 2) if val else None
    except Exception:
        return None
    return None


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
        req = {"symbol": ysym, "strikecount": count}
        if expiry:
            req["timestamp"] = expiry
        raw = _client.optionchain(data=req)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"optionchain call failed: {e}") from e
    if not isinstance(raw, dict) or raw.get("s") != "ok":
        raise RuntimeError(f"optionchain: {(raw or {}).get('message', 'bad response')}")
    data = raw.get("data") or {}
    rows = data.get("optionsChain") or []
    spot = _f(data, "callOi", default=0.0) and 0.0  # placeholder, real spot below
    spot = _spot(sym) or _f(data, "indiavixData", default=0.0)

    by_strike: Dict[float, Dict[str, Any]] = {}
    tokens: List[str] = []
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
        rec[f"{p}_oi_chg"] = _f(r, "oich", "oichp", "changeinOpenInterest")
        rec[f"{p}_volume"] = live.get("volume", _f(r, "volume", "vol_traded_today"))
        rec[f"{p}_bid"] = live.get("bid", _f(r, "bid", "bid_price"))
        rec[f"{p}_ask"] = live.get("ask", _f(r, "ask", "ask_price"))
        rec[f"{p}_iv"] = _f(r, "iv", "impliedVolatility")
        rec[f"{p}_prev_close"] = _f(r, "prev_close_price", "prevClose")
        rec[f"{p}_symbol"] = fsym

    strikes = [by_strike[k] for k in sorted(by_strike)]
    if not strikes:
        raise RuntimeError("option chain returned no strikes")
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
    tot_ce = sum(s.get("ce_oi", 0) for s in strikes)
    tot_pe = sum(s.get("pe_oi", 0) for s in strikes)
    expiries = [str(e.get("expiry") or e) for e in (data.get("expiryData") or [])]

    gaps = sorted({round(b - a) for a, b in zip(sorted(by_strike), sorted(by_strike)[1:]) if b > a})
    strike_gap = gaps[0] if gaps else 50

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
        "resistance": dict(resistance),
        "nearby": nearby,
        "atm": atm,
        "atm_iv": round(atm_iv, 2),
        "expiry": expiry or (expiries[0] if expiries else None),
        "all_expiries": expiries,
        "far_expiry": expiries[1] if len(expiries) > 1 else None,
        "strikes": strikes,
        "pcr": round(tot_pe / tot_ce, 3) if tot_ce else None,
        "total_ce_oi": tot_ce,
        "total_pe_oi": tot_pe,
        "max_pain": _max_pain(strikes),
        "strike_gap": strike_gap,
        "lot_size": _lot_size(sym),
        "india_vix": india_vix(),
        "timestamp": time.strftime("%d-%b-%Y %H:%M:%S"),
        "_source": "fyers",
        "_live": bool(_status.get("streaming")),
        "_tokens": {r[f"{p}_symbol"]: {"sym": sym, "k": r["strike"], "side": p.upper()}
                    for r in strikes for p in ("ce", "pe") if r.get(f"{p}_symbol")},
    }
    out["_tokens"][ysym] = {"sym": sym, "spot": True}
    with _lock:
        _status["last_chain"] = time.time()
    return out


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


_LOTS = {"NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65, "MIDCPNIFTY": 140, "SENSEX": 20}


def _lot_size(symbol: str) -> int:
    try:
        import nse_lot_sizes
        v = nse_lot_sizes.get_lot_size(symbol)
        if v:
            return int(v)
    except Exception:
        pass
    return _LOTS.get(symbol.upper(), 75)


# ── historical candles: real OHLC with VOLUME ─────────────────────────
def fetch_candles(symbol: str, interval_min: int = 1, days: int = 1) -> List[Dict[str, Any]]:
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
    res = {1: "1", 3: "3", 5: "5", 15: "15", 30: "30", 60: "60"}.get(interval_min, "1")
    today = time.strftime("%Y-%m-%d")
    frm = time.strftime("%Y-%m-%d", time.localtime(time.time() - days * 86400))
    try:
        r = _client.history({
            "symbol": ysym, "resolution": res, "date_format": "1",
            "range_from": frm, "range_to": today, "cont_flag": "1",
        })
        if not isinstance(r, dict) or r.get("s") != "ok":
            print(f"[fyers] history: {(r or {}).get('message', 'bad response')}")
            return []
        out = []
        for c in (r.get("candles") or []):
            if len(c) < 6:
                continue
            out.append({"t": int(c[0]), "o": float(c[1]), "h": float(c[2]),
                        "l": float(c[3]), "c": float(c[4]), "v": float(c[5]), "n": 1})
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[fyers] history failed: {e}")
        return []


def market_depth(symbol: str) -> Dict[str, Any]:
    """Level-2 depth — the input a genuine order-flow read needs."""
    if not connect():
        return {}
    ysym = FYERS_INDEX.get(symbol.upper(), symbol)
    try:
        r = _client.depth({"symbol": ysym, "ohlcv_flag": "1"})
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
