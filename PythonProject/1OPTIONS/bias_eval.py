"""Backtest the strike-grid bias on recorded chain snapshots (replay/<SYMBOL>_<date>.jsonl).

    python3 bias_eval.py                # NIFTY and BANKNIFTY, every replay day
    python3 bias_eval.py --sym NIFTY    # one index

For every snapshot it takes ATM +/- 5 strikes (round hundreds for NIFTY, as the grid shows) and asks
whether a read predicted the index over the next 1 / 3 / 5 / 10 snapshots (~1 minute apart).

Reads compared (all signed so + = bullish for the index; PE reads are flipped):
    level      ln(total bid / total offer)          the grid's old colouring
    shift      2-min change of that level / its session spread (EWMA)   what the grid shows now (_bfZ)
    prem_day   premium vs yesterday's close
    oi15       15-min OI buildup tag (LB +1, SC +.5, SB -1, LU -.5)
    flow5      5-min taker-flow imbalance (b5 - s5) / (b5 + s5)

Metric: within-day Spearman IC (rank correlation with the forward move), so a trending day does not
flatter an always-bullish read, plus the number of days on which it was positive. Then the hit rate of
the grid's own tiers. Thresholds are read from _BF in nse_dashboard.html, so the two never drift.

Caveat: a handful of days, overlapping horizons, and ~1-minute snapshots where the live grid sees ticks.
Treat ICs under ~0.05 as noise.
"""
import argparse
import glob
import json
import math
import os
import re
from datetime import datetime
from statistics import mean

HERE = os.path.dirname(os.path.abspath(__file__))
HS = (1, 3, 5, 10)


def bf_params():
    html = open(os.path.join(HERE, "nse_dashboard.html"), encoding="utf-8").read()
    m = re.search(r"const _BF = \{([^}]*)\}", html)
    return {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", m.group(1))}


BF = bf_params()


class Shift:
    """Python twin of _bfZ in nse_dashboard.html."""

    def __init__(self):
        self.s = {}

    def z(self, key, b, o, now):
        if not key or not (b and b > 0) or not (o and o > 0):
            return None
        L = math.log(b / o)
        s = self.s.get(key)
        if s is None or now - s["t"] > 1800 or now < s["t"]:
            s = self.s[key] = {"m": L, "v": 0.0, "t": now, "h": [(now, L)]}
        elif now - s["t"] >= 5:
            a = 1 - math.exp(-(now - s["t"]) / BF["tau"])
            dl = L - s["m"]
            s["m"] += a * dl
            s["v"] = (1 - a) * (s["v"] + a * dl * dl)
            s["t"] = now
            s["h"].append((now, L))
            while len(s["h"]) > 1 and now - s["h"][0][0] > BF["lb"] * 3:
                s["h"].pop(0)
        old = None
        for x in s["h"]:
            if now - x[0] >= BF["lb"] * 0.9:
                old = x
        if old is None or s["v"] <= 1e-9:
            return None
        return max(-3.0, min(3.0, (L - old[1]) / math.sqrt(s["v"])))


def tier(z):
    if z is None:
        return 0
    return 2 if abs(z) >= BF["strong"] else 1 if abs(z) >= BF["mod"] else 0


def _rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def ic(a, b):
    if len(a) < 20:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = mean(ra), mean(rb)
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra))
    sb = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / sa / sb if sa and sb else None


def load(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                t = datetime.strptime(d["timestamp"], "%d-%b-%Y %H:%M:%S")
            except Exception:
                continue
            if d.get("strikes") and d.get("underlying_value"):
                out.append((t, d))
    return out


def grid_rows(d, n=5):
    spot, nifty = d["underlying_value"], d.get("symbol") == "NIFTY"
    rows = sorted((r for r in d["strikes"] if r.get("ce_symbol") and r.get("pe_symbol") and not (nifty and r["strike"] % 100)),
                  key=lambda r: r["strike"])
    if not rows:
        return []
    ai = min(range(len(rows)), key=lambda i: abs(rows[i]["strike"] - spot))
    return rows[max(0, ai - n): ai + n + 1]


def day_samples(path):
    """Per snapshot: (per-side reads, forward moves in bps)."""
    S, sh, out = load(path), Shift(), []
    for i, (t, d) in enumerate(S):
        now, spot = t.timestamp(), d["underlying_value"]
        j15 = next((j for j in range(i, -1, -1) if (t - S[j][0]).total_seconds() >= 900), None)
        P15 = {r["strike"]: r for r in S[j15][1]["strikes"]} if j15 is not None else {}
        flow = (d.get("flow") or {}).get("strikes") or {}
        sides = []
        for r in grid_rows(d):
            for p, sg in (("ce", 1), ("pe", -1)):
                b, o = r.get(p + "_tbq"), r.get(p + "_tsq")
                x = {}
                if b and o and b > 0 and o > 0:
                    x["level"] = sg * math.log(b / o)
                z = sh.z(r.get(p + "_symbol"), b, o, now)
                if z is not None:
                    x["shift"] = sg * z
                ltp, pc = r.get(p + "_ltp") or 0, r.get(p + "_prev_close") or 0
                if ltp > 0 and pc > 0:
                    x["prem_day"] = sg * (ltp / pc - 1)
                q = P15.get(r["strike"])
                if q and ltp > 0 and (q.get(p + "_ltp") or 0) > 0:
                    dp, doi = ltp - q[p + "_ltp"], (r.get(p + "_oi") or 0) - (q.get(p + "_oi") or 0)
                    if dp and doi:
                        x["oi15"] = sg * (1 if dp > 0 else -1) * (1.0 if doi > 0 else 0.5)
                f = flow.get(f"{int(r['strike'])}{p.upper()}")
                if f and (f.get("b5") or 0) + (f.get("s5") or 0) > 0:
                    x["flow5"] = sg * (f["b5"] - f["s5"]) / (f["b5"] + f["s5"])
                sides.append(x)
        fwd = {h: (S[i + h][1]["underlying_value"] / spot - 1) * 1e4 for h in HS
               if i + h < len(S) and (S[i + h][0] - t).total_seconds() < h * 150}
        if t.hour * 60 + t.minute >= 9 * 60 + 25:        # skip the open: the shift needs its history first
            out.append((sides, fwd))
    return out


READS = ("level", "shift", "prem_day", "oi15", "flow5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", nargs="*", default=["NIFTY", "BANKNIFTY"])
    ap.add_argument("--dir", default=os.path.join(HERE, "replay"))
    a = ap.parse_args()
    files = [f for f in sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
             if os.path.basename(f).split("_")[0] in a.sym and os.path.getsize(f) > 1_000_000]
    if not files:
        print("no replay files"); return 1
    days = {os.path.basename(f)[:-6]: day_samples(f) for f in files}
    print("thresholds from nse_dashboard.html _BF:", BF)
    print("days:", ", ".join(f"{k} ({len(v)})" for k, v in days.items()))

    def agg(sides, k):
        v = [x[k] for x in sides if k in x]
        return sum(v) / len(v) if v else None

    print(f"\nwithin-day IC of the grid average  (mean over days, days positive)")
    print(f"{'read':10s}" + "".join(f"{f'{h} min':>16s}" for h in HS))
    for k in READS:
        line = f"{k:10s}"
        for h in HS:
            v = []
            for smp in days.values():
                pr = [(agg(s, k), f[h]) for s, f in smp if h in f]
                pr = [(x, y) for x, y in pr if x is not None]
                c = ic([x for x, _ in pr], [y for _, y in pr])
                if c is not None:
                    v.append(c)
            line += f"{(f'{mean(v):+.3f} {sum(c > 0 for c in v)}/{len(v)}' if v else 'n/a'):>16s}"
        print(line)

    print(f"\nhit rate of the shift arrows (direction of the index over the next h snapshots)")
    for label, pick in (("per side", lambda s: [x["shift"] for x in s if "shift" in x]),
                        ("grid avg", lambda s: [m] if (m := agg(s, "shift")) is not None else [])):
        for name, test in (("arrow ▲/▼", (lambda z: tier(z) >= 1) if label == "per side" else (lambda z: abs(z) >= BF["agg"])),
                           ("strong", (lambda z: tier(z) == 2) if label == "per side" else (lambda z: abs(z) >= 2 * BF["agg"])),
                           ("no arrow", (lambda z: tier(z) == 0) if label == "per side" else (lambda z: abs(z) < BF["agg"]))):
            cells = []
            for h in (1, 3, 5):
                pr = [(z, f[h]) for smp in days.values() for s, f in smp if h in f and f[h] for z in pick(s) if test(z)]
                if pr:
                    cells.append(f"{h}m {100 * mean(1.0 if (z > 0) == (y > 0) else 0.0 for z, y in pr):4.1f}% (n={len(pr)})")
            print(f"  {label:9s} {name:10s} " + "   ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
