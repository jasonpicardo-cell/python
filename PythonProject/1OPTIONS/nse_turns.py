"""Turn model: objective labels, feature log, fit and walk-forward validation.

The dashboard's turn light scores each closed bar for a BOTTOM (turn up) and
a TOP (turn down) from families of evidence (location, liquidity sweep,
momentum, flow, absorption, positioning, gamma, breadth, Nifty/BankNifty
divergence, VIX, time of day). Each family is ONE number in [0, 1], so
indicators built from the same closes (Squeeze, WaveTrend, RSI) count once.

This module does the parts that need history:

  label_bars()   the objective definition of a turn at a bar (no lookahead
                 in the features; the label itself looks forward by design)
  load_log()     the per-bar feature rows the dashboard logs to turn_log/
  fit()          logistic regression per side, walk-forward by day, with a
                 time-of-day-only baseline and a permutation null
  CLI            python nse_turns.py fit      -> turn_model.json + report
                 python nse_turns.py label --symbol NIFTY --date 2026-10-23
                 python nse_turns.py status

THE LABEL. At the close of bar t, a BOTTOM is a long that works:
  prior move    the high of the last PRE bars minus the low of the last NEAR
                bars is at least X  (there was a decline to turn from)
  near the low  close[t] is within S of that low      (it is a turn, not a
                                                       continuation)
  outcome       within H bars, price reaches close[t] + X BEFORE it reaches
                close[t] - S  (a bar touching both counts as a loss)
  X = max(0.30 x D, 0.0006 x spot) where D is the day's 1-sigma move from
  India VIX (spot x VIX / sqrt(252)); S = 0.4 x X (2.5 R); H = 30 minutes.
A TOP is the mirror. Rows without H minutes of bars after them are left out.

No broker or NSE call is made anywhere here: everything comes from files
the server and dashboard already write.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "turn_log")
WARM_DIR = os.path.join(HERE, "warm_cache")
MODEL_PATH = os.path.join(HERE, "turn_model.json")
REPORT_PATH = os.path.join(HERE, "turn_fit_report.json")

FAMILIES = ["loc", "sweep", "mom", "flow", "absorb", "pos", "gamma", "breadth", "smt", "vix", "time", "reg"]
# a family missing from an older log row means "not measured", which is NOT the
# same as 0: "reg" (1 = not a one-way trend) is neutral when absent
FAMILY_DEFAULT = {"reg": 1.0}
SIDES = {"bottom": 1, "top": -1}
LABEL = {"kX": 0.30, "minPct": 0.0006, "kS": 0.4, "H": 30, "PRE": 30, "NEAR": 15, "maxGap": 3}
# the hand-set starting weights the dashboard uses until a model is accepted
# "reg" = 1 when price is NOT running one way against the call; its +0.8 is
# paid out of the intercept (-3.8 = -3.0 - 0.8), so a normal market scores
# exactly as before and only a one-way trend takes points off
PRIOR = {"b": -3.8, "w": {"loc": 1.4, "sweep": 1.1, "mom": 0.9, "flow": 1.0, "absorb": 0.8, "pos": 0.6,
                          "gamma": 0.4, "breadth": 0.5, "smt": 0.8, "vix": 0.5, "time": 0.4, "reg": 0.8}}
MIN_TURNS = int(os.environ.get("NSE_TURN_MIN_EVENTS", "100"))
MIN_AUC = 0.55            # below this a "model" is noise, whatever the null says


# ── labels ──────────────────────────────────────────────────────────────
def label_bars(bars, D, iv=1, p=None):
    """bars: [{t,o,h,l,c}] of ONE session at one interval, sorted by t.
    D: the day's 1-sigma move in points. Returns a list aligned to bars of
    {"bottom": 0/1/None, "top": 0/1/None, "X", "S"}; None = not labelable."""
    P = dict(LABEL, **(p or {}))
    n = len(bars)
    out = [{"bottom": None, "top": None} for _ in range(n)]
    if n < 3 or not D or D <= 0:
        return out
    H = max(1, int(round(P["H"] / iv)))
    PRE = max(2, int(round(P["PRE"] / iv)))
    NEAR = max(2, int(round(P["NEAR"] / iv)))
    gap = 60 * iv * (P["maxGap"] + 1)
    for t in range(n):
        c = bars[t]["c"]
        X = max(P["kX"] * D, P["minPct"] * c)
        S = P["kS"] * X
        out[t]["X"], out[t]["S"] = X, S
        if t + H >= n:
            continue
        # the forward window must be contiguous (no missing stretch)
        if any(bars[k + 1]["t"] - bars[k]["t"] > gap for k in range(t, t + H)):
            continue
        a0, b0 = max(0, t - PRE + 1), max(0, t - NEAR + 1)
        hiPre = max(b["h"] for b in bars[a0:t + 1])
        loPre = min(b["l"] for b in bars[a0:t + 1])
        loNear = min(b["l"] for b in bars[b0:t + 1])
        hiNear = max(b["h"] for b in bars[b0:t + 1])
        for side, sg in SIDES.items():
            if sg > 0:
                setup = hiPre - loNear >= X and c - loNear <= S
            else:
                setup = hiNear - loPre >= X and hiNear - c <= S
            if not setup:
                out[t][side] = 0
                continue
            win, tgt, stp = 0, c + sg * X, c - sg * S
            for k in range(t + 1, t + H + 1):
                b = bars[k]
                hitT = b["h"] >= tgt if sg > 0 else b["l"] <= tgt
                hitS = b["l"] <= stp if sg > 0 else b["h"] >= stp
                if hitS:
                    break
                if hitT:
                    win = 1
                    break
            out[t][side] = win
    return out


def events(labels, iv=1):
    """Positive bars of one side close together are ONE turn: a bottom
    that could be bought on bars 32-37 is a single event, not six. Returns
    {side: [[first, last], ...]} of bar indices."""
    near = max(2, int(round(LABEL["NEAR"] / iv)))
    out = {s: [] for s in SIDES}
    for side in SIDES:
        for t, lb in enumerate(labels):
            if lb.get(side) != 1:
                continue
            ev = out[side]
            if ev and t - ev[-1][1] <= near:
                ev[-1][1] = t
            else:
                ev.append([t, t])
    return out


def turn_points(bars, labels, iv=1):
    """One point per turn event: the extreme it turned from, and the first
    bar a trade from it would have worked - for drawing on the chart."""
    near = max(2, int(round(LABEL["NEAR"] / iv)))
    pts = []
    for side, evs in events(labels, iv).items():
        sg = SIDES[side]
        for a, b in evs:
            lo = max(0, a - near + 1)
            k = (min(range(lo, b + 1), key=lambda i: bars[i]["l"]) if sg > 0
                 else max(range(lo, b + 1), key=lambda i: bars[i]["h"]))
            pts.append({"side": side, "t": bars[k]["t"], "price": bars[k]["l"] if sg > 0 else bars[k]["h"],
                        "first_t": bars[a]["t"], "last_t": bars[b]["t"]})
    return sorted(pts, key=lambda p: p["t"])


# ── the feature log ─────────────────────────────────────────────────────
def log_path(sym, day):
    return os.path.join(LOG_DIR, f"{sym.upper()}_{day}.jsonl")


def read_rows(path):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return rows


def _warm_bars(sym, day):
    try:
        with open(os.path.join(WARM_DIR, f"{sym}_{day}.json")) as f:
            return json.load(f).get("bars") or []
    except Exception:
        return []


def day_dataset(sym, day, rows=None):
    """Feature rows of one session joined to their labels. One series per
    chart interval; 1-minute warm-cache bars (from the broker, complete)
    replace the logged bars when the day has them."""
    rows = rows if rows is not None else read_rows(log_path(sym, day))
    out = []
    by_iv = {}
    for r in rows:
        if r.get("t") is None or r.get("c") is None:
            continue
        by_iv.setdefault(int(r.get("iv") or 1), {})[int(r["t"])] = r
    for iv, rmap in by_iv.items():
        ts = sorted(rmap)
        D = next((rmap[t].get("D") for t in reversed(ts) if rmap[t].get("D")), None)
        bars = None
        if iv == 1:
            wb = _warm_bars(sym, day)
            if len(wb) >= 300:
                bars = [{"t": int(b["t"]), "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]} for b in wb]
        if bars is None:
            bars = [{"t": t, "o": rmap[t].get("o", rmap[t]["c"]), "h": rmap[t].get("h", rmap[t]["c"]),
                     "l": rmap[t].get("l", rmap[t]["c"]), "c": rmap[t]["c"]} for t in ts]
        labs = label_bars(bars, D, iv)
        evid = {}
        for side, evs in events(labs, iv).items():
            for e, (a, b) in enumerate(evs):
                for i in range(a, b + 1):
                    evid[(side, i)] = f"{sym}|{day}|{iv}|{side}|{e}"
        idx = {b["t"]: i for i, b in enumerate(bars)}
        for t in ts:
            i = idx.get(t)
            if i is None:
                continue
            r = rmap[t]
            for side in SIDES:
                y = labs[i][side]
                f = ((r.get("f") or {}).get(side[0])) or {}
                if y is None or not f:
                    continue
                out.append({"day": day, "sym": sym, "t": t, "iv": iv, "side": side, "y": int(y),
                            "ev": evid.get((side, i)) if y else None,
                            "x": [float(FAMILY_DEFAULT.get(k, 0.0) if f.get(k) is None and k in FAMILY_DEFAULT
                                        else (f.get(k) or 0.0)) for k in FAMILIES]})
    return out


def day_turns(sym, day, iv=1):
    """The labelled turns of one logged session, for drawing on the chart."""
    rows = [r for r in read_rows(log_path(sym, day)) if int(r.get("iv") or 1) == iv and r.get("c") is not None]
    if not rows:
        return {"symbol": sym, "day": day, "iv": iv, "bars": 0, "points": [], "events": {s: 0 for s in SIDES}}
    rmap = {int(r["t"]): r for r in rows}
    ts = sorted(rmap)
    D = next((rmap[t].get("D") for t in reversed(ts) if rmap[t].get("D")), None)
    bars = None
    if iv == 1:
        wb = _warm_bars(sym, day)
        if len(wb) >= 300:
            bars = [{"t": int(b["t"]), "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]} for b in wb]
    if bars is None:
        bars = [{"t": t, "o": rmap[t].get("o", rmap[t]["c"]), "h": rmap[t].get("h", rmap[t]["c"]),
                 "l": rmap[t].get("l", rmap[t]["c"]), "c": rmap[t]["c"]} for t in ts]
    labs = label_bars(bars, D, iv)
    ev = events(labs, iv)
    return {"symbol": sym, "day": day, "iv": iv, "bars": len(bars), "D": D,
            "points": turn_points(bars, labs, iv), "events": {k: len(v) for k, v in ev.items()}}


def list_days(sym=None):
    if not os.path.isdir(LOG_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(LOG_DIR)):
        if not fn.endswith(".jsonl"):
            continue
        s, d = fn[:-6].rsplit("_", 1)
        if sym and s != sym.upper():
            continue
        out.append((s, d))
    return out


def load_dataset(sym=None):
    data = []
    for s, d in list_days(sym):
        data.extend(day_dataset(s, d))
    return data


# ── logistic regression (numpy if present, plain Python otherwise) ──────
try:
    import numpy as _np  # noqa: F401
except Exception:  # noqa: BLE001
    _np = None


def _sig(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(X, y, l2=1.0, iters=25):
    """Newton / IRLS with an L2 penalty (bias unpenalised). X: rows of
    features, y: 0/1. Returns (bias, weights)."""
    n = len(X)
    if not n:
        return PRIOR["b"], [PRIOR["w"][k] for k in FAMILIES]
    p = len(X[0]) + 1
    if _np is not None:
        # errstate: numpy 2.x on macOS (Accelerate BLAS) raises spurious
        # "divide by zero / overflow in matmul" warnings on valid input. The
        # fit does not rely on that: every step is checked for finite values
        # and halved until the penalised likelihood improves (a plain Newton
        # step can overshoot on lopsided data), and anything non-finite falls
        # back to the pure-Python path below.
        with _np.errstate(all="ignore"):
            A = _np.hstack([_np.ones((n, 1)), _np.asarray(X, dtype=float)])
            Y = _np.asarray(y, dtype=float)
            R = _np.eye(p) * l2
            R[0, 0] = 0.0
            pm = min(max(Y.mean(), 1e-6), 1 - 1e-6)
            w = _np.zeros(p)
            w[0] = math.log(pm / (1 - pm))                  # start at the base rate

            def nll(v):
                z = A @ v
                return float(_np.sum(_np.logaddexp(0.0, z) - Y * z) + 0.5 * l2 * float(v[1:] @ v[1:]))
            f0 = nll(w)
            ok = _np.isfinite(f0)
            for _ in range(iters if ok else 0):
                z = _np.clip(A @ w, -30, 30)
                mu = 1.0 / (1.0 + _np.exp(-z))
                g = A.T @ (mu - Y) + R @ w
                Hm = (A * (mu * (1 - mu))[:, None]).T @ A + R + 1e-9 * _np.eye(p)
                try:
                    step = _np.linalg.solve(Hm, g)
                except Exception:  # noqa: BLE001
                    ok = False
                    break
                if not _np.all(_np.isfinite(step)):
                    ok = False
                    break
                t, moved = 1.0, False
                while t >= 1e-4:
                    w2 = w - t * step
                    f2 = nll(w2)
                    if _np.isfinite(f2) and f2 <= f0 + 1e-12:
                        w, f0, moved = w2, f2, True
                        break
                    t *= 0.5
                if not moved or _np.max(_np.abs(t * step)) < 1e-7:
                    break
            if ok and _np.all(_np.isfinite(w)):
                return float(w[0]), [float(v) for v in w[1:]]
    w = [0.0] * p
    for _ in range(min(iters, 12)):
        g = [0.0] * p
        Hm = [[0.0] * p for _ in range(p)]
        for xi, yi in zip(X, y):
            a = [1.0] + list(xi)
            mu = _sig(sum(wj * aj for wj, aj in zip(w, a)))
            r = mu - yi
            s = mu * (1 - mu)
            for j in range(p):
                if a[j] == 0.0:
                    continue
                g[j] += r * a[j]
                sa = s * a[j]
                row = Hm[j]
                for k in range(p):
                    if a[k] != 0.0:
                        row[k] += sa * a[k]
        for j in range(1, p):
            g[j] += l2 * w[j]
            Hm[j][j] += l2
        for j in range(p):
            Hm[j][j] += 1e-9
        step = _solve(Hm, g)
        w = [wj - sj for wj, sj in zip(w, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return w[0], w[1:]


def _solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        if abs(M[c][c]) < 1e-15:
            continue
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-15 else 0.0 for i in range(n)]


def predict(b, w, X):
    return [_sig(b + sum(wi * xi for wi, xi in zip(w, x))) for x in X]


def auc(scores, y):
    """Mann-Whitney AUC with ties averaged; None if one class is missing."""
    pos = sum(1 for v in y if v)
    neg = len(y) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    sp = sum(r for r, v in zip(ranks, y) if v)
    return (sp - pos * (pos + 1) / 2.0) / (pos * neg)


# ── walk-forward evaluation ─────────────────────────────────────────────
def walk_forward(rows, cols=None, min_train_days=5, step_days=5, l2=1.0, shuffle_seed=None):
    """Expanding-window walk-forward by day: fit on all days before a block,
    predict the block. Returns (oos_scores, oos_y). cols restricts features
    (baselines). shuffle_seed permutes labels BY DAY BLOCK inside each
    training set (keeps the within-day structure) for the null."""
    days = sorted({r["day"] for r in rows})
    if len(days) <= min_train_days:
        return [], []
    sel = (lambda x: [x[FAMILIES.index(c)] for c in cols]) if cols else (lambda x: x)
    scores, ys = [], []
    k = min_train_days
    rng = random.Random(shuffle_seed) if shuffle_seed is not None else None
    while k < len(days):
        train_d, test_d = set(days[:k]), set(days[k:k + step_days])
        tr = [r for r in rows if r["day"] in train_d]
        te = [r for r in rows if r["day"] in test_d]
        ytr = [r["y"] for r in tr]
        if rng is not None:
            rng.shuffle(ytr)          # the null: same features, labels at random
        b, w = fit_logistic([sel(r["x"]) for r in tr], ytr, l2=l2)
        scores.extend(predict(b, w, [sel(r["x"]) for r in te]))
        ys.extend(r["y"] for r in te)
        k += step_days
    return scores, ys


def _calib(scores, y, bins=5):
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    out = []
    for b in range(bins):
        idx = order[b * len(order) // bins:(b + 1) * len(order) // bins]
        if idx:
            out.append({"bin": b + 1, "pred": round(sum(scores[i] for i in idx) / len(idx), 4),
                        "actual": round(sum(y[i] for i in idx) / len(idx), 4), "n": len(idx)})
    return out


def _top_lift(scores, y, frac=0.1):
    if not scores:
        return None
    base = sum(y) / len(y)
    k = max(1, int(len(scores) * frac))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    prec = sum(y[i] for i in top) / k
    return {"precision": round(prec, 4), "base": round(base, 4), "lift": round(prec / base, 2) if base else None,
            "threshold": round(min(scores[i] for i in top), 4)}


def fit(sym=None, perms=None, min_turns=MIN_TURNS, min_train_days=5, step_days=5, write=True, rows=None):
    rows = rows if rows is not None else load_dataset(sym)
    perms = perms if perms is not None else (20 if _np is not None else 3)
    report = {"fitted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "symbol": sym or "ALL", "families": FAMILIES,
              "label": LABEL, "min_turns": min_turns, "numpy": _np is not None, "sides": {}}
    model = {"version": 1, "fitted_at": report["fitted_at"], "families": FAMILIES, "label": LABEL,
             "sides": {}, "accepted": False}
    reasons = []
    for side in SIDES:
        R = [r for r in rows if r["side"] == side]
        days = sorted({r["day"] for r in R})
        npos = len({r.get("ev") or (r["day"], r["t"]) for r in R if r["y"]})     # EVENTS, not bars
        rep = {"rows": len(R), "days": len(days), "turns": npos, "positive_rows": sum(r["y"] for r in R),
               "base_rate": round(sum(r["y"] for r in R) / len(R), 4) if R else None}
        sc, yy = walk_forward(R, min_train_days=min_train_days, step_days=step_days)
        rep["oos_rows"] = len(yy)
        rep["auc"] = auc(sc, yy)
        st, yt = walk_forward(R, cols=["time"], min_train_days=min_train_days, step_days=step_days)
        rep["auc_time_only"] = auc(st, yt)
        sl, yl = walk_forward(R, cols=["loc"], min_train_days=min_train_days, step_days=step_days)
        rep["auc_location_only"] = auc(sl, yl)
        nulls = []
        for k in range(perms if len(yy) else 0):
            s0, y0 = walk_forward(R, min_train_days=min_train_days, step_days=step_days, shuffle_seed=1000 + k)
            a0 = auc(s0, y0)
            if a0 is not None:
                nulls.append(a0)
        nulls.sort()
        rep["null_auc_p95"] = nulls[int(0.95 * (len(nulls) - 1))] if nulls else None
        rep["null_runs"] = len(nulls)
        rep["calibration"] = _calib(sc, yy)
        rep["top_decile"] = _top_lift(sc, yy, 0.1)
        rep["top_quintile"] = _top_lift(sc, yy, 0.2)
        b, w = fit_logistic([r["x"] for r in R], [r["y"] for r in R]) if R else (PRIOR["b"], [PRIOR["w"][k] for k in FAMILIES])
        rep["weights"] = {k: round(v, 4) for k, v in zip(FAMILIES, w)}
        rep["bias"] = round(b, 4)
        ok = True
        why = []
        if npos < min_turns:
            ok = False; why.append(f"{npos} {side}s labelled, {min_turns} needed")
        if rep["auc"] is None:
            ok = False; why.append("not enough days for a walk-forward test")
        else:
            if rep["auc"] < MIN_AUC:
                ok = False; why.append(f"AUC {rep['auc']:.3f} below {MIN_AUC}")
            if rep["null_auc_p95"] is not None and rep["auc"] <= rep["null_auc_p95"]:
                ok = False; why.append(f"AUC {rep['auc']:.3f} not above the shuffled-label null ({rep['null_auc_p95']:.3f})")
            if rep["auc_time_only"] is not None and rep["auc"] < rep["auc_time_only"] + 0.02:
                ok = False; why.append(f"AUC {rep['auc']:.3f} not clearly above time-of-day alone ({rep['auc_time_only']:.3f})")
        rep["accepted"] = ok
        rep["why_not"] = why
        report["sides"][side] = rep
        model["sides"][side] = {"b": rep["bias"], "w": rep["weights"], "auc": rep["auc"],
                                "watch_p": (rep["top_quintile"] or {}).get("threshold"),
                                "base_rate": rep["base_rate"], "turns": npos, "accepted": ok}
        reasons += [f"{side}: {x}" for x in why]
    model["accepted"] = all(model["sides"][s]["accepted"] for s in SIDES) if model["sides"] else False
    model["why_not"] = reasons
    report["accepted"] = model["accepted"]
    if write:
        with open(REPORT_PATH, "w") as f:
            json.dump(report, f, indent=1)
        with open(MODEL_PATH, "w") as f:
            json.dump(model, f, indent=1)
    return model, report


def signals(sym=None, rows_by_day=None):
    """What the BUY / SELL calls the light actually made were worth, from the
    logged bars - usable from the first day, long before a model can be fitted.

    A call is a bar where the light's state became BUY or SELL (the entry is
    that bar's close). It is scored by the SAME objective as the turn labels: a
    win is +X reached before -S, within H minutes (X = max(0.3 x the day's 1-sigma
    move, 0.06% of spot), S = 0.4 X). A bar that touches both counts as a loss
    (no way to know the order). Neither reached = timeout. A call whose
    following bars are not all logged is skipped, not guessed."""
    out = {"n": 0, "win": 0, "loss": 0, "timeout": 0, "skipped": 0, "by_side": {}, "days": 0}
    src = rows_by_day if rows_by_day is not None else {(s, d): read_rows(log_path(s, d)) for s, d in list_days(sym)}
    H = LABEL["H"] * 60
    for (s, d), rows in sorted(src.items()):
        series = {}
        for r in rows:
            if r.get("t") is None or r.get("c") is None:
                continue
            series.setdefault(int(r.get("iv") or 1), {})[int(r["t"])] = r
        had = False
        for iv, rmap in series.items():
            ts = sorted(rmap)
            prev = None
            for i, t in enumerate(ts):
                r = rmap[t]
                st, dr = r.get("st"), r.get("dir")
                cur = (st, dr) if st in ("BUY", "SELL") and dr in (1, -1) else None
                fired = cur is not None and cur != prev
                prev = cur
                if not fired:
                    continue
                D, spot, entry = float(r.get("D") or 0), float(r.get("spot") or r["c"]), float(r["c"])
                if D <= 0:
                    out["skipped"] += 1
                    continue
                X = max(LABEL["kX"] * D, LABEL["minPct"] * spot)
                S = LABEL["kS"] * X
                fut = [rmap[u] for u in ts[i + 1:] if u - t <= H]
                need = max(1, int(H / 60 / iv * 0.8))
                res = None
                for b in fut:
                    up = float(b.get("h", b["c"])) >= entry + X if dr > 0 else float(b.get("l", b["c"])) <= entry - X
                    dn = float(b.get("l", b["c"])) <= entry - S if dr > 0 else float(b.get("h", b["c"])) >= entry + S
                    if dn:
                        res = "loss"
                        break
                    if up:
                        res = "win"
                        break
                if res is None:
                    if len(fut) < need:
                        out["skipped"] += 1
                        continue
                    res = "timeout"
                side = "bottom" if dr > 0 else "top"
                bs = out["by_side"].setdefault(side, {"n": 0, "win": 0, "loss": 0, "timeout": 0})
                bs["n"] += 1
                bs[res] += 1
                out["n"] += 1
                out[res] += 1
                had = True
        out["days"] += 1 if had else 0
    done = out["win"] + out["loss"]
    out["win_rate"] = round(out["win"] / done, 3) if done else None      # of the calls that resolved
    out["breakeven"] = round(LABEL["kS"] / (1 + LABEL["kS"]), 3)         # a 2.5 : 1 payoff needs 28.6%
    return out


def status(sym=None):
    """Progress toward a fit: days logged and turns labelled per side."""
    days = list_days(sym)
    ev = {s: set() for s in SIDES}
    rows = set()
    for s, d in days:
        for r in day_dataset(s, d):
            rows.add((r["sym"], r["iv"], r["t"]))
            if r["y"]:
                ev[r["side"]].add(r["ev"])
    sg = None
    try:
        sg = signals(sym)
    except Exception:
        pass
    return {"days": len(days), "rows": len(rows), "turns": {k: len(v) for k, v in ev.items()}, "min_turns": MIN_TURNS,
            "signals": sg}


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="turn model: label, fit, status")
    sub = ap.add_subparsers(dest="cmd")
    f = sub.add_parser("fit")
    f.add_argument("--symbol")
    f.add_argument("--perms", type=int)
    f.add_argument("--min-turns", type=int, default=MIN_TURNS)
    f.add_argument("--step-days", type=int, default=5)
    lb = sub.add_parser("label")
    lb.add_argument("--symbol", default="NIFTY")
    lb.add_argument("--date", required=True)
    st = sub.add_parser("status")
    st.add_argument("--symbol")
    sg = sub.add_parser("signals", help="what the BUY/SELL calls made so far were worth (same +X before -S objective)")
    sg.add_argument("--symbol")
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        model, rep = fit(a.symbol, perms=a.perms, min_turns=a.min_turns, step_days=a.step_days)
        for side, r in rep["sides"].items():
            print(f"{side:6s} days {r['days']:3d} rows {r['rows']:6d} turns {r['turns']:4d} base {r['base_rate']}"
                  f" | AUC {r['auc']} (time-only {r['auc_time_only']}, location-only {r['auc_location_only']},"
                  f" null p95 {r['null_auc_p95']}) top-decile {r['top_decile']}")
            if r["why_not"]:
                print("        not accepted: " + "; ".join(r["why_not"]))
        print(f"model {'ACCEPTED' if model['accepted'] else 'NOT accepted - the dashboard keeps the hand-set weights'}"
              f" -> {MODEL_PATH}")
    elif a.cmd == "label":
        ds = day_dataset(a.symbol.upper(), a.date)
        pos = [r for r in ds if r["y"]]
        print(f"{a.symbol} {a.date}: {len(ds)} labelled rows, {len(pos)} turns "
              f"({sum(1 for r in pos if r['side'] == 'bottom')} bottoms, {sum(1 for r in pos if r['side'] == 'top')} tops)")
    elif a.cmd == "signals":
        r = signals(a.symbol)
        print(f"{r['n']} resolved-or-timed-out call(s) over {r['days']} day(s), {r['skipped']} skipped (bars missing)")
        for side, v in r["by_side"].items():
            print(f"  {side:6s} n {v['n']:3d}  win {v['win']:3d}  loss {v['loss']:3d}  timeout {v['timeout']:3d}")
        if r["win_rate"] is not None:
            print(f"win rate of the calls that resolved: {r['win_rate']:.0%}  (break-even at this payoff: {r['breakeven']:.0%})")
        else:
            print("no resolved calls yet")
    elif a.cmd == "status":
        print(json.dumps(status(a.symbol), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    _main(sys.argv[1:])
