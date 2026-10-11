"""Move quality: is the current rally or fall real, or likely to reverse?

One implementation, used two ways:
  * LIVE      - the current leg of today's session is scored on nine pieces of
                evidence, and given a reversal probability.
  * CALIBRATE - every leg of every PAST recorded day is scored the same way
                at the moment it was in progress, and labelled by what price
                did next. Those labels turn the evidence into a probability.

Everything is read from the replay archive the server already writes (one
full option-chain snapshot a minute per symbol). No broker calls are made.

A "leg" is a zigzag swing on the minute series. A leg becomes a scored
"move" once it has travelled Q points (about 0.18% of spot). At each new
extreme past Q, 2Q, 3Q... a sample is taken; its label is decided by the
next 45 minutes:
    FAKE  price gives back 61.8% of the leg before extending another 50%
    REAL  it extends 50% first
    (neither within 45 minutes: not used)

Evidence, each scored +1 (supports the move), -1 (says it is fake) or 0:
    futures OI        rising = new positions behind the move; falling = covering
    option writers    for a rally, puts written below vs calls written above
    futures basis     futures leading or lagging spot over the leg
    option basis      ATM call minus put (put-call parity) against spot
    India VIX         falls on real rallies, rises on real falls
    efficiency        net move over the path travelled
    momentum          last five minutes against the leg's average pace
    acceptance        a broken day extreme held for ten minutes, or lost again
    VWAP / day type   with the trend day, or stretched past 2 sigma on a range day

Probability is naive Bayes over those nine, Laplace-smoothed, with the
evidence weight shrunk (the features are not independent, and naive Bayes
overstates confidence when they are not). It is reported only once at least
30 labelled samples, 10 of each kind, exist from days before the one shown.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

FEATURES = [
    ("foi", "Futures OI"),
    ("writers", "Option writers"),
    ("basis", "Futures basis"),
    ("synth", "Option-implied basis"),
    ("vix", "India VIX"),
    ("eff", "Efficiency"),
    ("mom", "Momentum (5 min)"),
    ("accept", "Acceptance"),
    ("vwap", "VWAP / day type"),
]
HORIZON_MIN = 45
SHRINK = 0.6          # evidence weight multiplier for correlated features

_lock = threading.Lock()
_series_cache: Dict[str, Tuple[Tuple, List[dict]]] = {}     # path -> (sig, series)
_series_off: Dict[str, Tuple[int, List[dict]]] = {}          # path -> (byte offset, rows) for growing files
_calib_cache: Dict[str, Tuple[Tuple, dict]] = {}             # symbol|day -> (sig, stats)
_calib_building: Dict[str, bool] = {}


# ── compact minute series ─────────────────────────────────────────────────
def _compact(s: dict) -> Optional[dict]:
    spot = s.get("underlying_value")
    if not spot:
        return None
    t = int(s.get("_replay_ts") or 0)
    gap = s.get("strike_gap") or 50
    ks = {}
    for r in s.get("strikes") or []:
        k = r.get("strike")
        if k is None or abs(k - spot) > gap * 10.5:
            continue
        ks[str(int(round(k)))] = [r.get("ce_oi") or 0, r.get("pe_oi") or 0,
                                  r.get("ce_ltp") or 0, r.get("pe_ltp") or 0]
    return {"t": t // 60 * 60, "spot": float(spot), "fut": s.get("futures_price"),
            "foi": s.get("futures_oi"), "vix": s.get("india_vix"), "gap": gap, "ks": ks}


def _parse_lines(lines) -> List[dict]:
    out = []
    for line in lines:
        try:
            c = _compact(json.loads(line))
        except Exception:
            continue
        if c and c["t"]:
            out.append(c)
    return out


def _dedupe(rows: List[dict]) -> List[dict]:
    by = {}
    for r in rows:
        by[r["t"]] = r            # last snapshot of each minute wins
    return [by[t] for t in sorted(by)]


def day_series(path: str) -> List[dict]:
    """Minute series for one replay file. Past days are parsed once and kept
    (memory, plus a small JSON beside the archive); today's growing file is
    read incrementally from where the last call stopped."""
    try:
        st = os.stat(path)
    except OSError:
        return []
    sig = (st.st_size, int(st.st_mtime))
    with _lock:
        hit = _series_cache.get(path)
        if hit and hit[0] == sig:
            return hit[1]
    disk = os.path.join(os.path.dirname(path), "_mq", os.path.basename(path) + f".{sig[0]}.json")
    if os.path.exists(disk):
        try:
            with open(disk) as fh:
                rows = json.load(fh)
            with _lock:
                _series_cache[path] = (sig, rows)
            return rows
        except Exception:
            pass
    off, rows = _series_off.get(path, (0, []))
    if off > st.st_size:
        off, rows = 0, []
    with open(path, "rb") as fh:
        fh.seek(off)
        chunk = fh.read()
    # only whole lines; a half-written last line waits for the next call
    cut = chunk.rfind(b"\n")
    if cut >= 0:
        new = _parse_lines(chunk[:cut + 1].decode("utf-8", "ignore").splitlines())
        rows = _dedupe(rows + new)
        off += cut + 1
    _series_off[path] = (off, rows)
    with _lock:
        _series_cache[path] = (sig, rows)
    # a past day never changes: persist its compact form
    today = time.strftime("%Y-%m-%d")
    if today not in os.path.basename(path):
        try:
            os.makedirs(os.path.dirname(disk), exist_ok=True)
            tmp = disk + f".{threading.get_ident()}.tmp"
            with open(tmp, "w") as fh:
                json.dump(rows, fh, separators=(",", ":"))
            os.replace(tmp, disk)
        except Exception:
            pass
    return rows


# ── helpers ───────────────────────────────────────────────────────────────
def _ist_min(t: int) -> int:
    g = time.gmtime(t + 19800)
    return g.tm_hour * 60 + g.tm_min


def _thresholds(series: List[dict], i: int) -> Tuple[float, float]:
    spot = series[i]["spot"]
    d = sorted(abs(series[k]["spot"] - series[k - 1]["spot"]) for k in range(1, i + 1)) or [1.0]
    med = d[len(d) // 2] or 1.0
    rev = max(0.0012 * spot, 3 * med)
    qual = max(0.0018 * spot, 5 * med)
    return rev, qual


def _zigzag(series: List[dict], upto: int, rev: float):
    """Legs as (start index, extreme index, direction) using data up to `upto`."""
    legs = []
    if upto < 1:
        return legs
    lo_i = hi_i = 0
    d = 0
    s = e = 0
    for i in range(1, upto + 1):
        p = series[i]["spot"]
        if d == 0:
            if p < series[lo_i]["spot"]:
                lo_i = i
            if p > series[hi_i]["spot"]:
                hi_i = i
            if p - series[lo_i]["spot"] >= rev:
                d, s, e = 1, lo_i, i
            elif series[hi_i]["spot"] - p >= rev:
                d, s, e = -1, hi_i, i
            continue
        if (p - series[e]["spot"]) * d > 0:
            e = i
        elif (series[e]["spot"] - p) * d >= rev:
            legs.append((s, e, d))
            s, e, d = e, i, -d
    if d:
        legs.append((s, e, d))
    return legs


def _prep(series: List[dict]) -> dict:
    """One pass: session TWAP and its sigma at every minute (equal weights -
    the index has no volume, and the same rule must hold on every day of the
    calibration), and the day type as it stood at every minute."""
    n = len(series)
    tw, sd, dt = [0.0] * n, [0.0] * n, [0] * n
    s1 = s2 = 0.0
    c = 0
    side = crosses = win = 0
    broken = False
    for k in range(n):
        x, m = series[k]["spot"], _ist_min(series[k]["t"])
        if m >= 555:
            s1 += x; s2 += x * x; c += 1
        if c >= 3:
            mu = s1 / c
            tw[k], sd[k] = mu, math.sqrt(max(0.0, s2 / c - mu * mu))
        else:
            tw[k], sd[k] = x, 0.0
        if m >= 560 and sd[k] > 0:
            dv = x - tw[k]
            sg = 1 if dv > 0.25 * sd[k] else -1 if dv < -0.25 * sd[k] else 0
            if sg:
                if m < 585:
                    if side and sg != side:
                        crosses += 1
                    side = win = sg
                else:
                    if side and sg != side and win and not crosses:
                        broken = True
                    if not broken:
                        side = sg
        dt[k] = 0 if (m < 585 or crosses or not win or broken) else win
    return {"tw": tw, "sd": sd, "dt": dt}


def features(series: List[dict], s: int, i: int, d: int, ctx: Optional[dict] = None) -> Dict[str, Tuple[int, str]]:
    ctx = ctx or _prep(series)
    a, b = series[s], series[i]
    spot = b["spot"]
    thr = max(1.0, 0.00004 * spot)
    out: Dict[str, Tuple[int, str]] = {}

    # futures OI
    if a.get("foi") and b.get("foi"):
        rel = (b["foi"] - a["foi"]) / a["foi"]
        v = 0 if abs(rel) < 0.001 else (1 if rel > 0 else -1)
        why = (f"{rel * 100:+.2f}% — " +
               ("new positions behind the move" if v > 0 else
                ("short covering" if d > 0 else "long unwinding") if v < 0 else "flat"))
        out["foi"] = (v, why)

    # option writers around the leg's starting price
    gap = a.get("gap") or 50
    above = [k for k in a["ks"] if a["spot"] < float(k) <= a["spot"] + 5 * gap]
    below = [k for k in a["ks"] if a["spot"] - 5 * gap <= float(k) < a["spot"]]
    if above and below:
        dce = sum((b["ks"].get(k, [0])[0] or 0) - a["ks"][k][0] for k in above)
        dpe = sum((b["ks"].get(k, [0, 0])[1] or 0) - a["ks"][k][1] for k in below)
        w = (dpe - dce) / (abs(dpe) + abs(dce) + 1.0)
        w = w if d > 0 else -w
        v = 1 if w > 0.25 else -1 if w < -0.25 else 0
        if d > 0:
            why = ("puts written below" if v > 0 else "calls written above — selling the rally" if v < 0
                   else "writers balanced")
        else:
            why = ("calls written above" if v > 0 else "puts written below — buying the fall" if v < 0
                   else "writers balanced")
        out["writers"] = (v, why)

    # futures basis
    if a.get("fut") and b.get("fut"):
        db = (b["fut"] - b["spot"]) - (a["fut"] - a["spot"])
        x = db * d
        v = 1 if x >= thr else -1 if x <= -thr else 0
        out["basis"] = (v, f"{db:+.1f} pts — futures {'leading' if v > 0 else 'lagging' if v < 0 else 'in step'}")

    # option-implied basis at the starting ATM strike
    if a["ks"]:
        K = min(a["ks"], key=lambda k: abs(float(k) - a["spot"]))
        r0, r1 = a["ks"].get(K), b["ks"].get(K)
        if r0 and r1 and r0[2] and r0[3] and r1[2] and r1[3]:
            sb0 = float(K) + r0[2] - r0[3] - a["spot"]
            sb1 = float(K) + r1[2] - r1[3] - b["spot"]
            x = (sb1 - sb0) * d
            v = 1 if x >= thr else -1 if x <= -thr else 0
            out["synth"] = (v, f"{sb1 - sb0:+.1f} pts — options {'confirming' if v > 0 else 'not confirming' if v < 0 else 'neutral'}")

    # VIX
    if a.get("vix") and b.get("vix"):
        dv = b["vix"] - a["vix"]
        if d > 0:
            v = 1 if dv <= -0.1 else -1 if dv >= 0.15 else 0
        else:
            v = 1 if dv >= 0.15 else -1 if dv <= -0.1 else 0
        out["vix"] = (v, f"{dv:+.2f} — " + ("fear easing" if dv < 0 else "fear rising" if dv > 0 else "flat"))

    # efficiency over the leg
    path = sum(abs(series[k]["spot"] - series[k - 1]["spot"]) for k in range(s + 1, i + 1))
    if path > 0 and i - s >= 3:
        er = abs(b["spot"] - a["spot"]) / path
        v = 1 if er >= 0.55 else -1 if er <= 0.3 else 0
        out["eff"] = (v, f"{er * 100:.0f}% — {'direct' if v > 0 else 'choppy' if v < 0 else 'mixed'}")

    # momentum: last five minutes against the leg's average pace
    if i - s >= 8 and i >= 5:
        pace = (b["spot"] - a["spot"]) * d / (i - s)
        prog = (b["spot"] - series[i - 5]["spot"]) * d
        if pace > 0:
            v = 1 if prog >= 0.8 * pace * 5 else -1 if prog <= 0.2 * pace * 5 else 0
            out["mom"] = (v, f"{prog:+.0f} pts in 5 min vs {pace * 5:.0f} average — "
                             f"{'still driving' if v > 0 else 'stalling' if v < 0 else 'slowing'}")

    # acceptance beyond the day's prior extreme
    if s >= 1:
        prior = [r["spot"] for r in series[:s + 1]]
        lvl = max(prior) if d > 0 else min(prior)
        crossed = [k for k in range(s, i + 1) if (series[k]["spot"] - lvl) * d > 0]
        if crossed:
            first = crossed[0]
            last10 = series[max(first, i - 9): i + 1]
            beyond = sum(1 for r in last10 if (r["spot"] - lvl) * d > 0)
            if (b["spot"] - lvl) * d <= 0:
                out["accept"] = (-1, f"broke the day's {'high' if d > 0 else 'low'} {lvl:.0f} and lost it")
            elif i - first >= 9 and beyond >= 8:
                out["accept"] = (1, f"held beyond {lvl:.0f} for {i - first + 1} min")
            else:
                out["accept"] = (0, f"beyond {lvl:.0f} for {i - first + 1} min — not yet accepted")

    # VWAP / day type
    tw, sd, td = ctx["tw"][i], ctx["sd"][i], ctx["dt"][i]
    if sd > 0:
        z = (b["spot"] - tw) / sd
        if td:
            v = 1 if td == d else -1
            why = f"trend day {'up' if td > 0 else 'down'} — move {'with' if v > 0 else 'AGAINST'} it ({z:+.1f}σ)"
        elif abs(z) >= 2 and (1 if z > 0 else -1) == d:
            v, why = -1, f"range day, stretched to {z:+.1f}σ from VWAP"
        else:
            v, why = 0, f"range day, {z:+.1f}σ from VWAP"
        out["vwap"] = (v, why)
    return out


def samples_for_day(series: List[dict]) -> List[dict]:
    """Labelled samples from one day, each judged only on data up to its own minute."""
    out = []
    n = len(series)
    if n < 30:
        return out
    taken = set()
    ctx = _prep(series)
    for i in range(20, n):
        if _ist_min(series[i]["t"]) > 915:
            break
        rev, qual = _thresholds(series, i)
        legs = _zigzag(series, i, rev)
        if not legs:
            continue
        s, e, d = legs[-1]
        if e != i:
            continue                      # sample only at a new extreme
        L = (series[i]["spot"] - series[s]["spot"]) * d
        step = int(L // qual)
        if step < 1 or (s, step) in taken:
            continue
        taken.add((s, step))
        P = series[i]["spot"]
        back = series[s]["spot"] + 0.382 * (P - series[s]["spot"])
        ext = P + 0.5 * L * d
        label = None
        for k in range(i + 1, min(n, i + 1 + HORIZON_MIN)):
            p = series[k]["spot"]
            if (p - back) * d <= 0:
                label = "fake"
                break
            if (p - ext) * d >= 0:
                label = "real"
                break
        if label is None:
            continue
        f = features(series, s, i, d, ctx)
        out.append({"label": label, "f": {k: v[0] for k, v in f.items()}})
    return out


def _replay_files(replay_dir: str, symbol: str) -> List[Tuple[str, str]]:
    pre = symbol.upper() + "_"
    try:
        names = sorted(f for f in os.listdir(replay_dir) if f.startswith(pre) and f.endswith(".jsonl"))
    except OSError:
        return []
    return [(f[len(pre):-6], os.path.join(replay_dir, f)) for f in names]


def _build_calibration(replay_dir: str, symbol: str, before_day: str, key: str, sig: tuple) -> None:
    try:
        counts = {fk: {"1": [0, 0], "0": [0, 0], "-1": [0, 0]} for fk, _ in FEATURES}
        nf = nr = 0
        days = 0
        for day, path in _replay_files(replay_dir, symbol):
            if day >= before_day:
                continue
            smp = samples_for_day(day_series(path))
            if smp:
                days += 1
            for sm in smp:
                j = 0 if sm["label"] == "fake" else 1
                if j == 0:
                    nf += 1
                else:
                    nr += 1
                for fk, _ in FEATURES:
                    v = str(sm["f"].get(fk, 0))
                    counts[fk][v][j] += 1
        stats = {"counts": counts, "fake": nf, "real": nr, "days": days, "built": int(time.time())}
        with _lock:
            _calib_cache[key] = (sig, stats)
    finally:
        _calib_building.pop(key, None)


def calibration(replay_dir: str, symbol: str, before_day: str) -> Optional[dict]:
    """Stats from days strictly before `before_day`. Built in the background
    the first time (parsing a month of chain snapshots takes a while) and
    then reused until a new past day appears."""
    files = [(d, p) for d, p in _replay_files(replay_dir, symbol) if d < before_day]
    sig = tuple((d, os.path.getsize(p)) for d, p in files)
    key = f"{symbol.upper()}|{before_day}"
    with _lock:
        hit = _calib_cache.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    if not _calib_building.get(key):
        _calib_building[key] = True
        threading.Thread(target=_build_calibration, args=(replay_dir, symbol, before_day, key, sig),
                         daemon=True).start()
    return hit[1] if hit else None


def probability(stats: Optional[dict], fv: Dict[str, int]) -> Optional[float]:
    if not stats:
        return None
    nf, nr = stats["fake"], stats["real"]
    if nf < 10 or nr < 10 or nf + nr < 30:
        return None
    logit = math.log(nf / nr)
    ev = 0.0
    for fk, _ in FEATURES:
        c = stats["counts"][fk][str(fv.get(fk, 0))]
        ev += math.log((c[0] + 1) / (nf + 3)) - math.log((c[1] + 1) / (nr + 3))
    p = 1 / (1 + math.exp(-(logit + SHRINK * ev)))
    return max(0.05, min(0.95, p))


def current(replay_dir: str, symbol: str, day: str, spot_now: Optional[float] = None) -> dict:
    path = os.path.join(replay_dir, f"{symbol.upper()}_{day}.jsonl")
    series = list(day_series(path))
    if not series:
        return {"symbol": symbol, "date": day, "error": "no replay recording for this day yet"}
    # the live price between minute snapshots, carried on the last snapshot
    if spot_now and spot_now > 0:
        now_t = int(time.time()) // 60 * 60
        last = dict(series[-1])
        last["spot"] = float(spot_now)
        if now_t > series[-1]["t"] and day == time.strftime("%Y-%m-%d"):
            last["t"] = now_t
            series.append(last)
        else:
            series[-1] = last
    i = len(series) - 1
    rev, qual = _thresholds(series, i)
    legs = _zigzag(series, i, rev)
    stats = calibration(replay_dir, symbol, day)
    calib = {"samples": (stats["fake"] + stats["real"]) if stats else 0,
             "fake": stats["fake"] if stats else 0, "real": stats["real"] if stats else 0,
             "days": stats["days"] if stats else 0,
             "building": bool(_calib_building.get(f"{symbol.upper()}|{day}"))}
    if not legs:
        return {"symbol": symbol, "date": day, "leg": None, "calib": calib}
    s, e, d = legs[-1]
    a, b = series[s], series[i]
    size = (b["spot"] - a["spot"]) * d
    # The evidence is read AT THE LEG'S EXTREME, not at this minute. That is
    # where every calibration sample was taken, so it is the only reading the
    # probability is valid for; what has happened since the extreme is shown
    # separately as the retracement so far.
    ctx = _prep(series)
    f = features(series, s, e, d, ctx)
    fv = {k: v[0] for k, v in f.items()}
    p = probability(stats, fv)
    ext_size = (series[e]["spot"] - a["spot"]) * d
    retr = ((series[e]["spot"] - b["spot"]) * d / ext_size) if ext_size > 0 else 0.0
    recent = []
    for (s2, e2, d2) in legs[-7:-1]:
        recent.append({"dir": d2, "start_t": series[s2]["t"], "end_t": series[e2]["t"],
                       "size": round((series[e2]["spot"] - series[s2]["spot"]) * d2, 1)})
    return {
        "symbol": symbol, "date": day,
        "leg": {"dir": d, "start_t": a["t"], "start_px": a["spot"], "ext_px": series[e]["spot"],
                "now_t": b["t"], "now_px": b["spot"], "size": round(size, 1),
                "ext_size": round((series[e]["spot"] - a["spot"]) * d, 1),
                "mins": (b["t"] - a["t"]) // 60, "qualified": ext_size >= qual, "qual": round(qual, 1),
                "retraced": round(retr, 3), "ext_t": series[e]["t"]},
        "features": [{"k": k, "name": name, "v": f[k][0], "why": f[k][1]} for k, name in FEATURES if k in f],
        "for": sum(1 for v in fv.values() if v > 0), "against": sum(1 for v in fv.values() if v < 0),
        "p_fake": None if p is None else round(p, 3),
        "calib": calib, "recent": recent,
    }
