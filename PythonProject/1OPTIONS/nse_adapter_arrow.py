"""
Arrow (arrow.trade / iRage Broking) market-data adapter for 1OPTIONS.

Why this exists
---------------
The default source scrapes NSE's public option chain. That works without an
account, but the OI it returns is 3-5 minutes stale and it needs a cookie
warm-up that breaks whenever NSE changes its front end. This adapter reads
the same information from a broker feed instead: real-time, with Greeks and
traded volume included, and no scraping.

Design notes
------------
* Chain snapshots come from Arrow's REST option-chain call. Live prices and
  OI arrive over the WebSocket and are merged into the latest snapshot, so
  the served chain is always as fresh as the last tick rather than as fresh
  as the last poll.
* Output is normalised to the exact shape the rest of the server already
  consumes (the same keys _build_response produces), so nothing downstream -
  cache, alert engine, SSE stream, dashboard - needs to know which source is
  active.
* Everything degrades rather than crashes: if the SDK is missing, or login
  fails, or the socket drops, the adapter reports unavailable and the server
  falls back to the NSE source.
"""

from __future__ import annotations

import math
import os as _os
import os
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from pyarrow_client import ArrowClient
    from pyarrow_client.constants import Exchange, QuoteMode
    from pyarrow_client.sockets import ArrowStreams, DataMode
    SDK_AVAILABLE = True
    SDK_ERROR = ""
except Exception as _e:  # noqa: BLE001
    SDK_AVAILABLE = False
    SDK_ERROR = str(_e)


# ── module state ──────────────────────────────────────────────────────
_client: Optional["ArrowClient"] = None
_streams: Optional["ArrowStreams"] = None
_lock = threading.Lock()
_ticks: Dict[str, Dict[str, Any]] = {}      # token -> latest tick fields
_chain_cache: Dict[str, Dict[str, Any]] = {}  # symbol -> last REST snapshot
_status: Dict[str, Any] = {
    "connected": False, "streaming": False, "last_error": "",
    "last_chain": 0.0, "ticks": 0, "subscribed": 0,
}


def status() -> Dict[str, Any]:
    with _lock:
        s = dict(_status)
    s["sdk"] = SDK_AVAILABLE
    if not SDK_AVAILABLE:
        s["last_error"] = s["last_error"] or f"pyarrow-client not installed: {SDK_ERROR}"
    return s


def is_configured() -> bool:
    return bool(os.environ.get("ARROW_APP_ID"))


# ── authentication ────────────────────────────────────────────────────
def connect() -> bool:
    """Log in and open the market-data stream. Safe to call repeatedly."""
    global _client, _streams
    if not SDK_AVAILABLE:
        _set_err(f"pyarrow-client not installed: {SDK_ERROR}")
        return False
    app_id = os.environ.get("ARROW_APP_ID", "").strip()
    if not app_id:
        _set_err("ARROW_APP_ID is empty - fill in .env")
        return False
    with _lock:
        if _status["connected"] and _client is not None:
            return True
    try:
        client = ArrowClient(app_id=app_id, timeout=15)
        token = os.environ.get("ARROW_ACCESS_TOKEN", "").strip()
        user = os.environ.get("ARROW_USER_ID", "").strip()
        if user and os.environ.get("ARROW_PASSWORD", "").strip():
            # unattended path: nothing to paste each morning
            client.auto_login(
                user_id=user,
                password=os.environ["ARROW_PASSWORD"],
                api_secret=os.environ.get("ARROW_API_SECRET") or None,
                totp_secret=os.environ.get("ARROW_TOTP_SECRET") or None,
            )
            how = "auto_login (TOTP)"
        elif token:
            client.set_token(token)
            how = "pasted access token"
        else:
            _set_err("no credentials: set ARROW_USER_ID/PASSWORD/TOTP_SECRET "
                     "or ARROW_ACCESS_TOKEN in .env")
            return False
        who = ""
        try:
            who = (client.get_user_details() or {}).get("name", "")
        except Exception:
            pass
        _client = client
        with _lock:
            _status.update(connected=True, last_error="")
        print(f"[arrow] authenticated via {how}" + (f" as {who}" if who else ""))
        _start_stream()
        return True
    except Exception as e:  # noqa: BLE001
        _set_err(f"login failed: {e}")
        return False


def _set_err(msg: str) -> None:
    with _lock:
        _status["last_error"] = msg
        _status["connected"] = False
    print(f"[arrow] {msg}")


# ── live stream ───────────────────────────────────────────────────────
_tick_sink = None          # set by the server: callable(payload) -> None
_last_push = 0.0
_dirty: Dict[str, Dict[str, Any]] = {}


def set_tick_sink(fn) -> None:
    """Register the callback that forwards ticks to connected browsers."""
    global _tick_sink
    _tick_sink = fn


def _push_ticks() -> None:
    """Forward accumulated ticks to the browsers, at most N times a second.

    Raw ticks can arrive hundreds of times a second across a full chain.
    Sending each one to every browser would drown the SSE connection and the
    UI cannot repaint that fast anyway, so changes are coalesced and flushed
    on a fixed cadence - the same idea as the chart's animation-frame
    batching, applied to the network.
    """
    global _last_push
    if _tick_sink is None:
        return
    hz = float(_os.environ.get("ARROW_PUSH_HZ", "4"))
    now = time.time()
    if now - _last_push < (1.0 / max(1.0, hz)):
        return
    with _lock:
        if not _dirty:
            return
        batch = _dirty
        _dirty.clear()
    _last_push = now
    try:
        _tick_sink({"type": "ticks", "ts": now, "n": len(batch), "t": batch})
    except Exception:
        pass


def _on_tick(tick: Any) -> None:
    """Merge a MarketTick into the local map keyed by instrument token."""
    try:
        tok = str(getattr(tick, "token", "") or getattr(tick, "instrument_token", ""))
        if not tok:
            return
        d = {}
        for src, dst in (("ltp", "ltp"), ("oi", "oi"), ("volume", "volume"),
                         ("ltq", "ltq"), ("open", "open"), ("high", "high"),
                         ("low", "low"), ("close", "close"), ("avg_price", "avg_price")):
            v = getattr(tick, src, None)
            if v is not None:
                d[dst] = v
        d["ts"] = time.time()
        with _lock:
            _ticks[tok] = {**_ticks.get(tok, {}), **d}
            _status["ticks"] += 1
            _status["streaming"] = True
            # only forward what the UI actually redraws
            _dirty[tok] = {k: v for k, v in _ticks[tok].items()
                           if k in ("ltp", "oi", "volume")}
        _push_ticks()
    except Exception:
        pass


def _start_stream() -> None:
    global _streams
    if _streams is not None:
        return
    try:
        st = ArrowStreams(app_id=os.environ["ARROW_APP_ID"], token=_client.get_token())
        st.connect_data_stream(on_tick=_on_tick)
        _streams = st
        with _lock:
            _status["streaming"] = True
        print("[arrow] market-data stream connected (auto-reconnect enabled)")
    except Exception as e:  # noqa: BLE001
        # streaming is an optimisation; REST snapshots still work without it
        print(f"[arrow] stream unavailable, falling back to REST snapshots: {e}")
        with _lock:
            _status["streaming"] = False


def _subscribe(tokens: List[str]) -> None:
    if not _streams or not tokens:
        return
    try:
        _streams.subscribe_market_data(mode=DataMode.FULL, tokens=tokens)
        with _lock:
            _status["subscribed"] = len(tokens)
    except Exception as e:  # noqa: BLE001
        print(f"[arrow] subscribe failed: {e}")


# ── chain fetch + normalisation ───────────────────────────────────────
def _f(d: Dict[str, Any], *names, default=0.0) -> float:
    """Read the first present key from a broker payload, tolerantly."""
    for n in names:
        if n in d and d[n] not in (None, "", "-"):
            try:
                return float(str(d[n]).replace(",", ""))
            except Exception:
                continue
    return default


def fetch_chain(symbol: str, expiry: Optional[str], band: int) -> Dict[str, Any]:
    """Return a chain response in the server's native shape.

    Raises RuntimeError when unavailable so the caller can fall back.
    """
    if not connect():
        raise RuntimeError(status().get("last_error") or "arrow unavailable")
    sym = (symbol or "NIFTY").upper()
    count = int(os.environ.get("ARROW_STRIKE_COUNT", "20"))
    try:
        expiries = _expiries(sym)
        exp = expiry or (expiries[0] if expiries else None)
        raw = _client.get_option_chain(underlying=sym, exchange=Exchange.NFO,
                                       count=count, expiry=exp)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"option chain call failed: {e}") from e

    rows = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, (list, tuple)):
        raise RuntimeError("unexpected option-chain payload shape")

    # group the flat CE/PE rows into per-strike records
    by_strike: Dict[float, Dict[str, Any]] = {}
    tokens: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = _f(r, "strikePrice", "strike_price", "strike")
        if not k:
            continue
        side = str(r.get("optionType") or r.get("option_type")
                   or r.get("right") or "").upper()
        side = "CE" if side.startswith("C") else "PE" if side.startswith("P") else ""
        if not side:
            continue
        tok = str(r.get("token") or r.get("securityId") or r.get("security_id") or "")
        if tok:
            tokens.append(tok)
        live = _ticks.get(tok, {}) if tok else {}
        rec = by_strike.setdefault(k, {"strike": k})
        pre = side.lower()
        # live tick wins over the REST snapshot when we have one
        rec[f"{pre}_ltp"] = live.get("ltp", _f(r, "lastPrice", "ltp", "last_price"))
        rec[f"{pre}_oi"] = live.get("oi", _f(r, "openInterest", "oi", "open_interest"))
        rec[f"{pre}_volume"] = live.get("volume", _f(r, "volume", "tradedVolume", "totalTradedVolume"))
        rec[f"{pre}_iv"] = _f(r, "impliedVolatility", "iv", "implied_volatility")
        # dashboard reads *_oi_chg (not *_change_oi) - keep both spellings
        _chg = _f(r, "changeinOpenInterest", "changeInOI", "oi_change", "changeOi")
        rec[f"{pre}_oi_chg"] = _chg
        rec[f"{pre}_change_oi"] = _chg
        rec[f"{pre}_delta"] = _f(r, "delta")
        rec[f"{pre}_gamma"] = _f(r, "gamma")
        rec[f"{pre}_theta"] = _f(r, "theta")
        rec[f"{pre}_vega"] = _f(r, "vega")
        rec[f"{pre}_bid"] = _f(r, "bidPrice", "bid", "best_bid")
        rec[f"{pre}_ask"] = _f(r, "askPrice", "ask", "best_ask")
        rec[f"{pre}_token"] = tok

    strikes = [by_strike[k] for k in sorted(by_strike)]
    spot = _spot(sym)
    if not spot and strikes:                      # fall back to min |CE-PE| parity
        best = min(strikes, key=lambda s: abs(s.get("ce_ltp", 0) - s.get("pe_ltp", 0)))
        spot = best["strike"]

    if tokens:
        _subscribe(sorted(set(tokens))[:5000])

    atm = min((s["strike"] for s in strikes), key=lambda k: abs(k - spot)) if strikes else None
    atm_row = next((s for s in strikes if s["strike"] == atm), {})
    atm_iv = 0.0
    ivs = [atm_row.get("ce_iv", 0), atm_row.get("pe_iv", 0)]
    ivs = [v for v in ivs if v]
    if ivs:
        atm_iv = sum(ivs) / len(ivs)

    tot_ce_oi = sum(s.get("ce_oi", 0) for s in strikes)
    tot_pe_oi = sum(s.get("pe_oi", 0) for s in strikes)

    # ── fields the dashboard reads beyond the raw chain ──────────────
    ks = sorted(s["strike"] for s in strikes)
    gaps = [round(b - a2) for a2, b in zip(ks, ks[1:]) if b > a2]
    strike_gap = min(gaps) if gaps else 50
    idx_ohlc = _index_ohlc(sym)
    flags = _buildup_flags(strikes, spot, strike_gap)
    # The dashboard reads support/resistance as OBJECTS (support.strike), so
    # omitting them throws and kills the entire render pipeline - one failure
    # there stops every panel after it. Same shape as the NSE source: the
    # largest put-OI and call-OI strikes inside the band.
    _sg = strike_gap if "strike_gap" in dir() else 50
    lo_b, hi_b = (atm or spot) - band * _sg, (atm or spot) + band * _sg
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
        "expiry": exp,
        "all_expiries": expiries,
        "far_expiry": expiries[1] if len(expiries) > 1 else None,
        "strikes": strikes,
        "pcr": round(tot_pe_oi / tot_ce_oi, 3) if tot_ce_oi else None,
        "total_ce_oi": tot_ce_oi,
        "total_pe_oi": tot_pe_oi,
        "max_pain": _max_pain(strikes),
        "dte": _dte(exp),
        "timestamp": time.strftime("%d-%b-%Y %H:%M:%S"),
        "strike_gap": strike_gap,
        "lot_size": _lot_size(sym),
        "prev_close": idx_ohlc.get("prev_close"),
        "prev_open": idx_ohlc.get("prev_open"),
        "prev_high": idx_ohlc.get("prev_high"),
        "prev_low": idx_ohlc.get("prev_low"),
        "session_ohlc": idx_ohlc.get("session"),      # today's O/H/L/C (Woodie pivots)
        "india_vix": _india_vix(),
        "iv_rank": _iv_rank(sym, atm_iv),
        "flags": flags,                                # OI Flow "Top Buildup" table
        "expiries_data": {},                           # filled on demand per expiry
        "_source": "arrow",
        "_live": bool(_status.get("streaming")),
    }
    # token -> what it IS, so a browser receiving a bare token can apply it
    tmap = {}
    for s in strikes:
        for side in ("ce", "pe"):
            t = s.get(f"{side}_token")
            if t:
                tmap[t] = {"sym": sym, "k": s["strike"], "side": side.upper()}
    spot_tok = _index_token(sym)
    if spot_tok:
        tmap[spot_tok] = {"sym": sym, "spot": True}
        _subscribe([spot_tok])
    out["_tokens"] = tmap
    with _lock:
        _chain_cache[sym] = out
        _status["last_chain"] = time.time()
    return out


def _index_token(symbol: str) -> Optional[str]:
    """Instrument token of the underlying index, so spot streams too."""
    try:
        q = _client.get_quote(mode=QuoteMode.LTP, symbol=symbol, exchange=Exchange.INDEX)
        d = q.get("data", q) if isinstance(q, dict) else {}
        t = (d or {}).get("token") or (d or {}).get("instrument_token")
        return str(t) if t else None
    except Exception:
        return None


_LOTS = {"NIFTY": 75, "BANKNIFTY": 35, "FINNIFTY": 65, "MIDCPNIFTY": 140, "SENSEX": 20}
_iv_hist: Dict[str, List[float]] = {}


def _lot_size(symbol: str) -> int:
    """Lot size; the shared nse_lot_sizes module wins when importable."""
    try:
        import nse_lot_sizes
        v = nse_lot_sizes.get_lot_size(symbol)
        if v:
            return int(v)
    except Exception:
        pass
    return _LOTS.get(symbol.upper(), 75)


def _index_ohlc(symbol: str) -> Dict[str, Any]:
    """Previous-day OHLC (pivots) and today's developing OHLC (Woodie)."""
    out: Dict[str, Any] = {}
    try:
        q = _client.get_quote(mode=QuoteMode.OHLCV, symbol=symbol, exchange=Exchange.INDEX)
        d = q.get("data", q) if isinstance(q, dict) else {}
        if isinstance(d, dict):
            o = _f(d, "open", "openPrice")
            h = _f(d, "high", "dayHigh")
            l = _f(d, "low", "dayLow")
            c = _f(d, "ltp", "close", "lastPrice")
            pc = _f(d, "prevClose", "previousClose", "close_price")
            if o or h or l:
                out["session"] = {"open": o, "high": h, "low": l, "close": c}
            if pc:
                out["prev_close"] = pc
    except Exception:
        pass
    # previous session's H/L/O from daily candles - what pivots actually need
    try:
        to_ts = time.strftime("%Y-%m-%d")
        frm = time.strftime("%Y-%m-%d", time.localtime(time.time() - 12 * 86400))
        cd = _client.candle_data(exchange=Exchange.INDEX, token=symbol,
                                 interval="D", from_timestamp=frm, to_timestamp=to_ts)
        rows = cd.get("data", cd) if isinstance(cd, dict) else cd
        if isinstance(rows, (list, tuple)) and len(rows) >= 2:
            prev = rows[-2] if isinstance(rows[-1], dict) else rows[-2]
            if isinstance(prev, dict):
                out["prev_open"] = _f(prev, "open", "o")
                out["prev_high"] = _f(prev, "high", "h")
                out["prev_low"] = _f(prev, "low", "l")
                out.setdefault("prev_close", _f(prev, "close", "c"))
    except Exception:
        pass
    return out


def _india_vix() -> Optional[float]:
    try:
        q = _client.get_quote(mode=QuoteMode.LTP, symbol="INDIAVIX", exchange=Exchange.INDEX)
        d = q.get("data", q) if isinstance(q, dict) else {}
        v = _f(d if isinstance(d, dict) else {}, "ltp", "lastPrice")
        return round(v, 2) if v else None
    except Exception:
        return None


def _iv_rank(symbol: str, atm_iv: float) -> Optional[int]:
    """Session-local IV rank.

    A true rank needs a year of IV history, which the broker does not hand
    over in one call. This ranks today's ATM IV against what THIS server has
    observed since it started, and returns None until there is enough spread
    to be meaningful - better than showing a confident number built on three
    samples.
    """
    if not atm_iv:
        return None
    h = _iv_hist.setdefault(symbol, [])
    h.append(atm_iv)
    if len(h) > 3000:
        del h[:1000]
    if len(h) < 30:
        return None
    lo, hi = min(h), max(h)
    if hi - lo < 0.5:
        return None
    return int(round((atm_iv - lo) / (hi - lo) * 100))


def _buildup_flags(strikes: List[Dict[str, Any]], spot: float, gap: float) -> List[Dict[str, Any]]:
    """Per-strike OI build-up classification for the OI Flow table."""
    out = []
    if not spot:
        return out
    for s in strikes:
        if abs(s["strike"] - spot) > 6 * gap:
            continue
        for side in ("ce", "pe"):
            chg = s.get(f"{side}_oi_chg", 0)
            if not chg or abs(chg) < 1:
                continue
            ltp = s.get(f"{side}_ltp", 0)
            up = chg > 0
            if side == "ce":
                label = "Short Buildup" if up else "Short Covering"
            else:
                label = "Put Writing" if up else "Put Unwinding"
            out.append({"strike": s["strike"], "side": side.upper(),
                        "oi_chg": int(chg), "ltp": ltp, "label": label})
    out.sort(key=lambda x: -abs(x["oi_chg"]))
    return out[:20]


def _expiries(symbol: str) -> List[str]:
    try:
        yr = time.strftime("%Y")
        ex = _client.get_expiry_dates(symbol=symbol, year=yr)
        lst = ex.get("data", ex) if isinstance(ex, dict) else ex
        return sorted(str(x) for x in lst) if isinstance(lst, (list, tuple)) else []
    except Exception:
        return []


def _spot(symbol: str) -> float:
    try:
        q = _client.get_quote(mode=QuoteMode.LTP, symbol=symbol, exchange=Exchange.INDEX)
        d = q.get("data", q) if isinstance(q, dict) else {}
        return _f(d if isinstance(d, dict) else {}, "ltp", "lastPrice", "last_price")
    except Exception:
        return 0.0


def _max_pain(strikes: List[Dict[str, Any]]) -> Optional[float]:
    """Strike where total option-writer payout is smallest."""
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


def _dte(expiry: Optional[str]) -> Optional[int]:
    if not expiry:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            t = time.strptime(str(expiry), fmt)
            days = (time.mktime(t) - time.time()) / 86400.0
            return max(0, int(math.ceil(days)))
        except Exception:
            continue
    return None


def shutdown() -> None:
    global _streams
    try:
        if _streams:
            _streams.disconnect_all()
    except Exception:
        pass
    _streams = None
    with _lock:
        _status.update(streaming=False, connected=False)
