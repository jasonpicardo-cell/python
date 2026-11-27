"""Does a boxed strike (the strike grid's top-ranked x B / x S, _bfMax) matter to a scalper?

Replays book_ticks/NIFTY_<day>.csv.gz (every book change, the same tbq/tsq totals the live grid ranks) through
a twin of the dashboard's ranking (bf_max = _bfMax, limits read from _BFR in nse_dashboard.html), one sample
a second. An EVENT is the index coming within --near points of a strike from outside. From the event price,
the outcome is whichever comes first: a move of --move points back the way it came (REJECT: the level held,
a fade works) or --move points onward (BREAK: it ran through). Unresolved within the horizon = no outcome.

Boxed events: the strike is #1 on either side and has held #1 for --held seconds. Control: the same
approaches to strikes on show that carry no rank at all. If boxed rejects are no different from control,
the box is noise for scalping. Both rankings are run: the whole-book totals the grid uses (tbq/tsq) and the
5 best levels (b5/a5), which are harder to fake. Like the grid, both rank OTM + ATM cells only (CE at or above the
ATM strike, PE at or below); "all" ranks every cell, as the grid first did: ITM books are lopsided by structure,
so #1 sat mostly on ITM strikes 100-300 points away, where spot seldom goes.

Read with care: few sessions, approaches on one day are not independent, and the per-day lines show whether
one day carries the result. Pure Python, no broker calls.

    python3 rank_eval.py                      # all recorded days
    python3 rank_eval.py --near 10 --move 15 --held 120
"""
import argparse
import glob
import math
import os
import re

import nse_tick_recorder as rec

HERE = os.path.dirname(os.path.abspath(__file__))
HS = (60, 180, 300)          # outcome horizons, seconds
MODES = ("tot", "d5", "all")  # grid ranking (totals, OTM + ATM cells) / same on the 5 best levels / totals on every cell (the old grid)
DEPTH = 10                   # grid default: ATM +/- 10 strikes, 50s shown
OPEN_S, LAST_EVENT_S, CLOSE_S = (9 * 60 + 20) * 60, (15 * 60 + 15) * 60, (15 * 60 + 30) * 60


def _html():
    return open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()


def bfr_params(html=None):
    """_BFR (ranking limits) and _BFA.cellAge from the dashboard, so this stays in step with the grid."""
    html = html or _html()
    m = re.search(r"const _BFR = \{([^}]*)\}", html)
    out = {}
    for k, v in re.findall(r"(\w+):\s*([^,}]+)", m.group(1)):
        v = v.strip()
        lg = re.fullmatch(r"Math\.log\(([\d.]+)\)", v)
        out[k] = math.log(float(lg.group(1))) if lg else float(v)
    a = re.search(r"const _BFA = \{[^}]*cellAge:\s*([\d.]+)", html)
    out["cellAge"] = float(a.group(1))
    return out


def bf_max(rows, now, key, st, R):
    """Twin of _bfMax. rows: [(strike, {'ce': (b, x, sym, fresh), 'pe': (...)})] in grid order.
    st: {'ema': {sym: (v, t)}, 'held': {key|side: {strike: (rank, since)}}}.
    Returns {'ce': {strike: (rank, v, held)}, 'pe': {...}}."""
    m = {"ce": {}, "pe": {}}
    for p in ("ce", "pe"):
        c = []
        for k, sides in rows:
            b, x, sym, fresh = sides[p]
            if not (b and b > 0 and x and x > 0) or not fresh:
                continue
            ln = math.log(b / x)
            e = st["ema"].get(sym)
            v = e[0] + (ln - e[0]) * (1 - math.exp(-max(0.0, now - e[1]) / R["tau"])) if e else ln
            st["ema"][sym] = (v, now)
            c.append((k, v, b + x))
        qs = sorted(q for _, _, q in c)
        med = qs[len(qs) >> 1] if qs else 0
        hk = f"{key}|{p}"
        prev, held = st["held"].get(hk, {}), {}
        cand = [(k, v, abs(v) * (R["keep"] if k in prev else 1)) for k, v, q in c if q >= R["floor"] * med and abs(v) >= R["min"]]
        cand.sort(key=lambda e: -e[2])                      # stable, like Array.prototype.sort
        for i, (k, v, _) in enumerate(cand[:3]):
            rank = i + 1
            since = prev[k][1] if k in prev and prev[k][0] == rank else now
            held[k] = (rank, since)
            m[p][k] = (rank, v, now - since)
        st["held"][hk] = held
    return m


def _qty(levels):
    tot = 0
    for lv in (levels or "").split("|"):
        parts = lv.split(":")
        if len(parts) >= 2 and parts[1]:
            tot += float(parts[1])
    return tot


def _f(v):
    return float(v) if v not in (None, "") else None


def day_events(path, R, near, move, held_min):
    """One session: (events, spot range). Each event: dict(t, k, dir, mode, kind, lean, out={h: 'R'|'B'|None})."""
    # rows with no exchange time are test_nse.py's fake stream (spot 25012) that leaked into the real
    # files before the tests turned the recorder off; Kite always stamps real ticks
    rows = [r for r in rec.read_day(path) if r["ex_ms"]]
    opts = [r for r in rows if r["kind"] == "OPT"]
    if not opts:
        return [], 0.0
    day = os.path.basename(path).split("_")[1][:10]
    exps = sorted({r["expiry"] for r in opts if r["expiry"] >= day}) or sorted({r["expiry"] for r in opts})
    exp = exps[0]
    book = {"tot": {}, "d5": {}}          # (strike, typ) -> (b, x, t)
    spot, spots = None, []                # per-second spot samples [(t, spot)]
    states = {m: {"ema": {}, "held": {}} for m in MODES}
    arm = {}                              # strike -> armed (True when spot is outside near + move)
    events = []
    i, n = 0, len(rows)
    t0 = int(rows[0]["t_ms"]) // 1000
    t_end = int(rows[-1]["t_ms"]) // 1000
    for now in range(t0, t_end + 1):
        while i < n and int(rows[i]["t_ms"]) // 1000 <= now:
            r = rows[i]
            i += 1
            if r["kind"] == "IDX":
                v = _f(r["ltp"])
                if v:
                    spot = v
            elif r["kind"] == "OPT" and r["expiry"] == exp:
                k, typ, t = float(r["strike"]), r["typ"].lower(), int(r["t_ms"]) / 1000
                b, x = _f(r["tbq"]), _f(r["tsq"])
                if b is not None or x is not None:
                    ob = book["tot"].get((k, typ), (None, None, 0))
                    book["tot"][(k, typ)] = (b if b is not None else ob[0], x if x is not None else ob[1], t)
                if r["b5"] or r["a5"]:
                    book["d5"][(k, typ)] = (_qty(r["b5"]), _qty(r["a5"]), t)
        sec = (now + 19800) % 86400                    # IST seconds since midnight
        if spot is None or sec < OPEN_S - 300 or sec > CLOSE_S:
            continue
        spots.append((now, spot, sec))
        strikes = sorted({k for (k, typ) in book["tot"] if (k, "ce") in book["tot"] and (k, "pe") in book["tot"]})
        if not strikes:
            continue
        ai = min(range(len(strikes)), key=lambda j: abs(strikes[j] - spot))
        show = strikes[max(0, ai - DEPTH): ai + DEPTH + 1]
        snap = {}
        atm = show[min(range(len(show)), key=lambda j: abs(show[j] - spot))]
        for mode in MODES:
            bk = book["d5" if mode == "d5" else "tot"]
            grid = []
            for k in show:
                sides = {}
                for p in ("ce", "pe"):
                    e = bk.get((k, p))
                    ok = mode == "all" or (k >= atm if p == "ce" else k <= atm)    # the grid ranks OTM + ATM cells only
                    sides[p] = (e[0], e[1], f"{k}{p}", ok and now - e[2] <= R["cellAge"]) if e else (None, None, f"{k}{p}", False)
                grid.append((k, sides))
            mx = bf_max(grid, now, mode, states[mode], R)
            lv = {}
            for p in ("ce", "pe"):
                for k, (rank, v, h) in mx[p].items():
                    lean = (1 if v > 0 else -1) * (1 if p == "ce" else -1)   # CE buyers / PE sellers = bullish book
                    cur = lv.get(k)
                    if cur is None or rank < cur[0]:
                        lv[k] = (rank, h, lean)
                    elif rank == cur[0] and cur[2] != lean:
                        lv[k] = (rank, max(h, cur[1]), 0)                   # both sides #N, leaning opposite ways
            snap[mode] = lv
        # approaches
        if sec < OPEN_S or sec > LAST_EVENT_S:
            continue
        for k in show:
            dist = spot - k
            if abs(dist) > near + move:
                arm[k] = 1 if dist < 0 else -1          # remember the side it is approaching from
            elif abs(dist) <= near and arm.get(k):
                d = arm.pop(k)
                for mode in MODES:
                    r1 = snap[mode].get(k)
                    if r1 is None:
                        kind = "control"
                    elif r1[0] == 1 and r1[1] >= held_min:
                        kind = "box1"
                    else:
                        kind = "box23" if r1[0] > 1 else "box1_new"
                    events.append({"t": now, "idx": len(spots) - 1, "k": k, "dir": d, "s0": spot, "mode": mode, "kind": kind,
                                   "lean": r1[2] if r1 else 0})
    # outcomes
    for e in events:
        out = {}
        res, j0 = None, e["idx"]
        for j in range(j0 + 1, len(spots)):
            t, s, _ = spots[j]
            if t - e["t"] > max(HS):
                break
            mv = (s - e["s0"]) * e["dir"]
            if mv <= -move:
                res = ("R", t - e["t"])
                break
            if mv >= move:
                res = ("B", t - e["t"])
                break
        for h in HS:
            out[h] = res[0] if res and res[1] <= h else None
        e["out"] = out
    rng = (max(s for _, s, _ in spots) - min(s for _, s, _ in spots)) if spots else 0.0
    return events, rng


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def tally(evs, h):
    r = sum(1 for e in evs if e["out"][h] == "R")
    b = sum(1 for e in evs if e["out"][h] == "B")
    return r, b


def line(label, evs, h):
    r, b = tally(evs, h)
    n = r + b
    if not n:
        return f"  {label:<34} n=0"
    lo, hi = wilson(r, n)
    return f"  {label:<34} n={n:<4} reject {100 * r / n:5.1f}%  (95% {100 * lo:4.0f}-{100 * hi:3.0f}%)  of {len(evs)} approaches"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--near", type=float, default=10, help="points from the strike that count as reaching it (10)")
    ap.add_argument("--move", type=float, default=15, help="points that decide reject / break (15)")
    ap.add_argument("--held", type=float, default=120, help="seconds a strike must have held #1 (120)")
    ap.add_argument("--min-range", type=float, default=60, help="skip days the index moved less than this (60)")
    ap.add_argument("--files", nargs="*", help="book_ticks files (default: all NIFTY days)")
    a = ap.parse_args()
    R = bfr_params()
    files = a.files or sorted(glob.glob(os.path.join(HERE, "book_ticks", "NIFTY_*.csv.gz")))
    print(f"rank_eval: near {a.near:g} pts, move {a.move:g} pts, #1 held >= {a.held:g} s; ranking limits {R}")
    allev, used = [], []
    for f in files:
        evs, rng = day_events(f, R, a.near, a.move, a.held)
        name = os.path.basename(f)[6:16]
        if rng < a.min_range:
            print(f"  skip {name}: index range {rng:.0f} pts (no session or too quiet)")
            continue
        for e in evs:
            e["day"] = name
        allev += evs
        used.append(name)
    print(f"sessions used: {len(used)} ({', '.join(used) or 'none'})")
    if not allev:
        print("No events. Record more sessions (nse_tick_recorder.py runs inside the server).")
        return
    for mode, title in (("tot", "whole-book totals, OTM + ATM cells (what the grid ranks)"), ("d5", "5 best levels (b5/a5), OTM + ATM cells"),
                        ("all", "totals on every cell (the old grid: ITM books are lopsided by structure)")):
        ev = [e for e in allev if e["mode"] == mode]
        print(f"\n== ranking on {title} ==")
        for h in HS:
            print(f" horizon {h // 60} min  (outcome = first {a.move:g}-pt move from the event price; reject = back the way it came)")
            ctl = [e for e in ev if e["kind"] == "control"]
            b1 = [e for e in ev if e["kind"] == "box1"]
            print(line("control (no rank)", ctl, h))
            print(line("#1 held", b1, h))
            print(line("  #1 held, book leans against", [e for e in b1 if e["lean"] == -e["dir"]], h))
            print(line("  #1 held, book leans with", [e for e in b1 if e["lean"] == e["dir"]], h))
            print(line("#1 new (under held)", [e for e in ev if e["kind"] == "box1_new"], h))
            print(line("#2 / #3", [e for e in ev if e["kind"] == "box23"], h))
        h = 180
        print(" per day, 3 min: #1 held vs control reject %")
        for d in used:
            ctl = [e for e in ev if e["day"] == d and e["kind"] == "control"]
            b1 = [e for e in ev if e["day"] == d and e["kind"] == "box1"]
            rc, bc = tally(ctl, h)
            rb, bb = tally(b1, h)
            pc = f"{100 * rc / (rc + bc):5.1f}% (n={rc + bc})" if rc + bc else "   -  "
            pb = f"{100 * rb / (rb + bb):5.1f}% (n={rb + bb})" if rb + bb else "   -  "
            print(f"  {d}: #1 held {pb:<16} control {pc}")
    print("\n'leans against' = the box's book leans against the approach (spot rising into a bearish book: CE offers or PE bids"
          " heavy). A gap vs control that holds on most days and across horizons is worth acting on; one day or a CI"
          " spanning the control rate is not.")


if __name__ == "__main__":
    main()
