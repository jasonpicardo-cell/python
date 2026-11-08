"""Live read of the Pro guide's calibration (guide/pro_calib.json, written by pro_calibrate.py after each close):

  VER          realised variance so far today / the VIX-implied variance due by now on the calibrated intraday
               clock, read against your own median full-day value and the 11:00 tercile cut-offs (on the recorded
               days the 11:00 value ranked the full day with a rank correlation of ~0.9).
  IV clocks    the front expiry's ATM IV on the calendar clock (what the chain shows) and on the trading clock
               (the same total variance spread over the variance units actually left: sessions, nights, weekends).
  Event move   the front expiry's variance above what VIX's 30-day base rate accounts for, as a one-off move.

Information only: no trade call, no order, no broker call. Pure Python mirror of pro_calibrate's clock
(test_nse checks the two agree), so the server needs no numpy.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
IST = timezone(timedelta(hours=5, minutes=30))
SESSION, WINDOW = 375, 30
YEAR_S = 365 * 86400
_cache = {"path": None, "mtime": None, "data": None}


def calib(path: str = None):
    """The calibration dict, re-read when the file changes; None if it has not been generated."""
    path = path or os.path.join(HERE, "guide", "pro_calib.json")
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    if _cache["path"] != path or _cache["mtime"] != mt:
        try:
            with open(path, encoding="utf-8") as f:
                _cache.update(path=path, mtime=mt, data=json.load(f))
        except Exception:  # noqa: BLE001
            return _cache["data"] if _cache["path"] == path else None
    return _cache["data"]


def reload():
    _cache.update(path=None, mtime=None, data=None)


# ----------------------------------------------------------------------------- the clock (mirror of pro_calibrate)
def _u1(u5):
    if not u5:
        return [1.0] * SESSION
    return [float(u5[min(i // 5, len(u5) - 1)]) for i in range(SESSION)]


def frac_remaining(m, u5, averaging=True):
    u = _u1(u5)
    tot = sum(u)
    s = 0.0
    for i in range(max(int(m), 0), SESSION):
        w = 1.0 if (i < SESSION - WINDOW or not averaging) else (SESSION - i - 0.5) / WINDOW
        s += w * w * u[i]
    return s / tot


def _weights(clock):
    ow = (clock or {}).get("overnight_w") or 0.25
    ww = (clock or {}).get("weekend_w") or max(ow, 0.35)
    return ow, ww


def units_per_year(clock):
    ow, ww = _weights(clock)
    return 252 + 200 * ow + 52 * ww


def variance_units(now, expiry, clock):
    u5 = (clock or {}).get("u5")
    ow, ww = _weights(clock)
    m = (now.hour * 60 + now.minute) - 555
    exp_m = (expiry.hour * 60 + expiry.minute) - 555
    if now.date() == expiry.date():
        return frac_remaining(max(m, 0), u5) if m < SESSION else 0.0
    if now.date() > expiry.date():
        return 0.0
    units = frac_remaining(max(m, 0), u5, averaging=False) if 0 <= m < SESSION else (1.0 if m < 0 else 0.0)
    d = now.date()
    while True:
        nd = d + timedelta(days=1)
        while nd.weekday() >= 5:
            nd += timedelta(days=1)
        units += ww if (nd - d).days >= 3 else ow
        d = nd
        if d >= expiry.date():
            break
        units += 1.0
    full = frac_remaining(0, u5) if exp_m >= SESSION - 1 else frac_remaining(0, u5) - frac_remaining(max(exp_m, 0), u5)
    return units + full


# ----------------------------------------------------------------------------- the gauge
def realised_r2(px_by_min: dict, upto: int):
    """Sum of squared log returns between successive samples at session minutes < upto ({session minute: price})."""
    ks = sorted(k for k in px_by_min if 0 <= k < upto and px_by_min[k] and px_by_min[k] > 0)
    return sum(math.log(px_by_min[b] / px_by_min[a]) ** 2 for a, b in zip(ks, ks[1:])), len(ks)


def ver_read(px_by_min: dict, m_now: int, prev_vix: float, cal: dict):
    clock, vs = (cal or {}).get("clock"), (cal or {}).get("ver") or {}
    if not prev_vix or not clock or m_now < 15:
        return None
    u5 = clock.get("u5")
    imp = (prev_vix / 100.0) ** 2 / units_per_year(clock)
    m = min(m_now, SESSION)
    rv, n = realised_r2(px_by_min, m)
    if n < max(10, m * 0.6):
        return {"ok": False, "why": f"only {n} minute samples so far"}
    f = 1 - frac_remaining(m, u5, averaging=False)
    out = {"ok": True, "ver": rv / (imp * f), "minutes": m, "samples": n, "prev_vix": prev_vix,
           "median_close": vs.get("median_close"), "n_days": vs.get("n_days"), "rank_corr": vs.get("rank_corr")}
    if out["median_close"]:
        out["rel"] = out["ver"] / out["median_close"]
    cut = vs.get("ver11_cut")
    m11 = 105
    base = None
    if m >= m11:
        rv11, _ = realised_r2(px_by_min, m11)
        base = rv11 / (imp * (1 - frac_remaining(m11, u5, averaging=False)))
        out["ver11"] = base
    if cut and len(cut) == 2:
        x = base if base is not None else out["ver"]
        t = 0 if x <= cut[0] else (1 if x <= cut[1] else 2)
        out["tercile"] = t
        out["provisional"] = base is None
        rel = vs.get("close_rel_by_tercile") or []
        out["close_rel_hist"] = rel[t] if t < len(rel) else None
    # implied stdev of what is left of today's session on the calibrated clock, in index points per 1.0 of spot
    out["sd_left_frac"] = math.sqrt(imp * frac_remaining(m, u5))
    return out


def clocks_read(now: datetime, spot: float, atm_iv: float, expiry_epoch: float, vix: float, cal: dict, events=()):
    clock = (cal or {}).get("clock")
    if not (atm_iv and expiry_epoch and spot):
        return None
    exp = datetime.fromtimestamp(expiry_epoch, IST)
    T = (expiry_epoch - now.timestamp()) / YEAR_S
    if T <= 0:
        return None
    w = (atm_iv / 100.0) ** 2 * T
    u = variance_units(now, exp, clock)
    uy = units_per_year(clock)
    out = {"expiry": exp.strftime("%Y-%m-%d %H:%M"), "iv_calendar": atm_iv, "units_left": u, "days_cal": T * 365,
           "iv_trading": math.sqrt(w / (u / uy)) * 100 if u > 0 else None, "calibrated": bool(clock)}
    if vix and vix > 0:
        u30 = variance_units(now, now + timedelta(days=30), clock)
        base = (vix / 100.0) ** 2 * 30 / 365 / u30 if u30 > 0 else None
        if base:
            lump = w - base * u
            out["base_vol_trading"] = math.sqrt(base * uy) * 100
            out["event_move_pct"] = math.sqrt(lump) * 100 if lump > 0 else None
            out["event_move_pts"] = math.sqrt(lump) * spot if lump > 0 else None
    out["events"] = [{"date": str(d), "label": l} for d, l in events if now.date() <= d <= exp.date()]
    return out


def read_events(root=HERE):
    out = []
    p = os.path.join(root, "guide", "events.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    out.append((datetime.strptime(line[:10], "%Y-%m-%d").date(), line[10:].strip()))
                except ValueError:
                    pass
    return out


def gauge(now: datetime, spot: float, atm_iv: float, expiry_epoch: float, vix: float, prev_vix: float,
          px_by_min: dict, cal: dict = None, events=()):
    cal = cal if cal is not None else calib()
    m_now = (now.hour * 60 + now.minute) - 555
    out = {"generated": (cal or {}).get("generated"), "calibrated": bool(cal), "minute": m_now}
    out["ver"] = ver_read(px_by_min, m_now, prev_vix, cal) if 0 <= m_now else None
    out["clocks"] = clocks_read(now, spot, atm_iv, expiry_epoch, vix, cal, events)
    return out
