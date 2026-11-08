"""Backtest the strike-grid bias and the desk setup on recorded chain snapshots (replay/<SYMBOL>_<date>.jsonl).

    python3 bias_eval.py                # NIFTY and BANKNIFTY, every replay day
    python3 bias_eval.py --sym NIFTY    # one index

Every snapshot (~1 minute apart) is scored on whether a read called the index's next 1 / 3 / 5 / 10 snapshots.

Reads (all signed so + = bullish for the index):
  grid (ATM +/- 5 strikes, as the grid shows; PE reads flipped)
    level      ln(total bid / total offer)                          the grid's old colouring
    shift      2-min change of that level / its session spread       the arrows (_bfZ)
    prem_day   premium vs yesterday's close
    oi15       15-min OI buildup tag (LB +1, SC +.5, SB -1, LU -.5)
    flow5      5-min taker-flow imbalance
    dflow      the same, delta-weighted (net customer delta bought)  flow desks
    writer     15-min OI added, delta-weighted: put writing +, call writing -   Indian OI desks
    wconf      5-min writer OI change, counted only when the taker side agrees (sellers while OI rises = writing,
               buyers while it falls = covering), delta-weighted. The Pro guide's "classify OI by aggressor" read.
    oitag      the grid's OI tag (oi_tag below: 15-min OI change x 15-min taker side), read as the direction it
               implies (CE BUY / COVER and PE WRITE / DUMP bullish). The grid shows the tag without colour or arrow.
               First six sessions: 45-49%, no edge (leans the wrong way, like oi15). Re-check as days accumulate.
  index
    fut_flow   5-min taker-flow imbalance on the near future         index desks (CVD)
    walls      position between the OI put wall and call wall, contrarian
    pin        distance to the gamma-weighted strike (pull toward it)        gamma desks
    fade15     minus the last 15-min move                                     mean reversion
  setup        CANDIDATE, not shown in the grid: long dealer gamma + 15-min move of at least fadeBps
               + grid-average shift of at least agg in the fade direction (the long-gamma "fade with
               flow confirmation" playbook). On the first six sessions: 38 calls, ~53% right, so the grid
               shows the gamma regime, walls and 15-min stretch as context only. Re-check as days accumulate,
               especially short-gamma (γ−) days, which the first sample did not contain.

n/a = the read's strongest calls were nearly all one direction (no balanced score possible).

Metric: BALANCED hit rate = mean of (hit rate of the bullish calls, hit rate of the bearish calls). It is not
flattered by a trending day the way a raw hit rate is (an always-bullish read wins on an up day), nor by the
within-day rank IC, which quietly removes each day's trend and so rewards fading it. Both are shown for
reference. Continuous reads are scored on their strongest 30% (by |value|) so they are comparable with
the arrows. Thresholds come from _BF in nse_dashboard.html, so the two never drift.

Caveat: a handful of days, overlapping horizons, ~1-minute snapshots where the live grid sees ticks, and
many reads tried. Treat anything within ~4 points of 50% as noise.
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


class SpotMove:
    """Python twin of _bfSpot: the index move over the last fadeMin seconds, in bps."""

    def __init__(self):
        self.s = {}

    def bps(self, key, spot, now):
        if not (spot and spot > 0):
            return None
        s = self.s.get(key)
        if s is None or now - s["t"] > 1800 or now < s["t"]:
            s = self.s[key] = {"t": now, "h": [(now, spot)]}
        elif now - s["t"] >= 5:
            s["t"] = now
            s["h"].append((now, spot))
            while len(s["h"]) > 1 and now - s["h"][0][0] > BF["fadeMin"] * 1.5:
                s["h"].pop(0)
        old = None
        for x in s["h"]:
            if now - x[0] >= BF["fadeMin"]:
                old = x
        return (spot / old[1] - 1) * 1e4 if old else None


def setup(shift_avg, mv15, gex_total):
    """Candidate desk setup (not in the grid): +1 fade long, -1 fade short, 0 none."""
    if shift_avg is None or mv15 is None or gex_total is None or gex_total < 0 or abs(mv15) < BF["fadeBps"]:
        return 0
    d = -1 if mv15 > 0 else 1
    return d if shift_avg * d >= BF["agg"] else 0


def tier(z):
    if z is None:
        return 0
    return 2 if abs(z) >= BF["strong"] else 1 if abs(z) >= BF["mod"] else 0


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _greeks(S, K, T, iv, call, r=0.07):
    """delta, gamma; T in years (dashboard: dte / 365)."""
    if T <= 0 or iv <= 0.01:
        return (1.0 if S > K else 0.0) - (0.0 if call else 1.0), 0.0
    sd = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / sd
    return (_ncdf(d1) if call else _ncdf(d1) - 1), math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) / (S * sd)


def gex_total(d):
    """Sign twin of _raw_computeGex: dealers long calls (+), long puts (-), every strike in the chain."""
    spot, T, tot = d["underlying_value"], max(1 / 1440, d.get("dte") or 1) / 365, 0.0
    for r in d["strikes"]:
        g_ce = _greeks(spot, r["strike"], T, (r.get("ce_iv") or 0) / 100, True)[1]
        g_pe = _greeks(spot, r["strike"], T, (r.get("pe_iv") or 0) / 100, False)[1]
        tot += g_ce * (r.get("ce_oi") or 0) - g_pe * (r.get("pe_oi") or 0)
    return tot


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


def _imb(b, s):
    return (b - s) / (b + s) if (b or 0) + (s or 0) > 0 else None


def oi_tag(oi, d_oi, b, s):
    """Python twin of _bfOi in nse_dashboard.html: WRITE / BUY / COVER / DUMP or None.
    OI must move at least oiMin of the open interest and the taker side must lean at least oiImb."""
    oi, d_oi, b, s = oi or 0, d_oi or 0, b or 0, s or 0
    if not oi > 0 or not b + s > 0 or abs(d_oi) < oi * BF["oiMin"]:
        return None
    im = (b - s) / (b + s)
    if abs(im) < BF["oiImb"]:
        return None
    return ("BUY" if im > 0 else "WRITE") if d_oi > 0 else ("COVER" if im > 0 else "DUMP")


TAG_SIGN = {"BUY": 1, "COVER": 1, "WRITE": -1, "DUMP": -1}   # for the option itself: + = premium-bullish


def day_samples(path):
    """Per snapshot: ({read: value}, {h: forward move in bps})."""
    S, sh, sm, out = load(path), Shift(), SpotMove(), []
    hist = []
    for i, (t, d) in enumerate(S):
        now, spot = t.timestamp(), d["underlying_value"]
        T = max(1 / 1440, d.get("dte") or 1) / 365
        j15 = next((j for j in range(i, -1, -1) if (t - S[j][0]).total_seconds() >= 900), None)
        P15 = {r["strike"]: r for r in S[j15][1]["strikes"]} if j15 is not None else {}
        j5 = next((j for j in range(i, -1, -1) if (t - S[j][0]).total_seconds() >= 300), None)
        P5 = {r["strike"]: r for r in S[j5][1]["strikes"]} if j5 is not None else {}
        flow = (d.get("flow") or {}).get("strikes") or {}
        acc = {}

        def add(k, num, den=1.0):
            if num is not None and math.isfinite(num):
                a = acc.setdefault(k, [0.0, 0.0]); a[0] += num; a[1] += den
        for r in grid_rows(d):
            iv = (r.get("ce_iv") or d.get("atm_iv") or 13) / 100
            for p, sg in (("ce", 1), ("pe", -1)):
                ad = abs(_greeks(spot, r["strike"], T, iv, sg > 0)[0])
                b, o = r.get(p + "_tbq"), r.get(p + "_tsq")
                if b and o and b > 0 and o > 0:
                    add("level", sg * math.log(b / o))
                z = sh.z(r.get(p + "_symbol"), b, o, now)
                if z is not None:
                    add("shift", sg * z)
                ltp, pc = r.get(p + "_ltp") or 0, r.get(p + "_prev_close") or 0
                if ltp > 0 and pc > 0:
                    add("prem_day", sg * (ltp / pc - 1))
                q = P15.get(r["strike"])
                if q and ltp > 0 and (q.get(p + "_ltp") or 0) > 0:
                    dp, doi = ltp - q[p + "_ltp"], (r.get(p + "_oi") or 0) - (q.get(p + "_oi") or 0)
                    if dp and doi:
                        add("oi15", sg * (1 if dp > 0 else -1) * (1.0 if doi > 0 else 0.5))
                f = flow.get(f"{int(r['strike'])}{p.upper()}")
                if f and (f.get("b5") or 0) + (f.get("s5") or 0) > 0:
                    add("flow5", sg * _imb(f["b5"], f["s5"]))
                    add("dflow", sg * ad * (f["b5"] - f["s5"]), ad * (f["b5"] + f["s5"]))
                o15 = r.get(p + "_oi_15m")
                if o15:
                    add("writer", -sg * ad * o15, ad * abs(o15))
                q5 = P5.get(r["strike"])
                if f and q5 and (f.get("b5") or 0) + (f.get("s5") or 0) > 0:
                    d5 = (r.get(p + "_oi") or 0) - (q5.get(p + "_oi") or 0)
                    if d5:
                        agree = (f["s5"] > f["b5"]) == (d5 > 0)
                        add("wconf", -sg * ad * d5 * agree, ad * abs(d5))
                tg = oi_tag(r.get(p + "_oi"), o15, (f or {}).get("b15"), (f or {}).get("s15"))
                if tg:
                    add("oitag", sg * TAG_SIGN[tg])
        F = {k: (v[0] / v[1] if v[1] else None) for k, v in acc.items()}
        fu = ((d.get("flow") or {}).get("futures") or {}).get("near")
        F["fut_flow"] = _imb(fu.get("b5") or 0, fu.get("s5") or 0) if fu else None
        allr = [r for r in d["strikes"] if r.get("ce_symbol")]
        up = [r for r in allr if r["strike"] > spot]
        dn = [r for r in allr if r["strike"] <= spot]
        if up and dn:
            cw = max(up, key=lambda r: r.get("ce_oi") or 0)["strike"]
            pw = max(dn, key=lambda r: r.get("pe_oi") or 0)["strike"]
            F["walls"] = 0.5 - (spot - pw) / (cw - pw) if cw > pw else None
        gw = gk = 0.0
        for r in allr:
            g = _greeks(spot, r["strike"], T, (r.get("ce_iv") or d.get("atm_iv") or 13) / 100, True)[1]
            w = g * ((r.get("ce_oi") or 0) + (r.get("pe_oi") or 0)); gw += w; gk += w * r["strike"]
        F["pin"] = (gk / gw - spot) if gw else None
        hist.append((now, spot))
        o15 = next((h for h in reversed(hist) if now - h[0] >= 900), None)
        F["fade15"] = -(spot / o15[1] - 1) * 1e4 if o15 else None
        F["setup"] = setup(F.get("shift"), sm.bps(d.get("symbol") or "", spot, now), gex_total(d)) or None
        fwd = {h: (S[i + h][1]["underlying_value"] / spot - 1) * 1e4 for h in HS
               if i + h < len(S) and (S[i + h][0] - t).total_seconds() < h * 150}
        if t.hour * 60 + t.minute >= 9 * 60 + 25:        # skip the open: the reads need their history first
            out.append((F, fwd))
    return out


READS = ("level", "prem_day", "oi15", "flow5", "dflow", "writer", "wconf", "oitag", "fut_flow", "walls", "pin", "fade15", "shift", "setup")


def balanced(pairs):
    """pairs of (signal, forward move); signal 0/None = no call. Returns (balanced %, long %, short %, n)."""
    L = [y > 0 for x, y in pairs if x and x > 0 and y]
    S_ = [y < 0 for x, y in pairs if x and x < 0 and y]
    if len(L) < 10 or len(S_) < 10:
        return None
    lh, sh = mean(L), mean(S_)
    return 100 * (lh + sh) / 2, 100 * lh, 100 * sh, len(L) + len(S_)


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
    allp = [(F, f) for smp in days.values() for F, f in smp]
    print("thresholds from nse_dashboard.html _BF:", BF)
    print("days:", ", ".join(f"{k} ({len(v)})" for k, v in days.items()))
    ups = [f[5] > 0 for _, f in allp if f.get(5)]
    print(f"index up over the next 5 snapshots {100 * mean(ups):.0f}% of the time (an always-bullish read's raw hit rate)")

    def cut(k):
        v = sorted(abs(F[k]) for F, _ in allp if F.get(k) is not None)
        return v[int(0.7 * (len(v) - 1))] if v and k != "setup" else 0.0

    print("\nBALANCED hit rate (avg of long-call and short-call hit rates)   [long% / short%, n]     within-day IC (days +)")
    print(f"{'read':10s}" + "".join(f"{f'{h} min':>27s}" for h in (1, 3, 5)) + f"{'5 min':>16s}")
    for k in READS:
        th, line = cut(k), f"{k:10s}"
        for h in (1, 3, 5):
            r = balanced([(F[k] if F.get(k) is not None and abs(F[k]) >= th else None, f.get(h)) for F, f in allp])
            line += f"{(f'{r[0]:4.1f}% [{r[1]:.0f}/{r[2]:.0f}, {r[3]}]' if r else 'n/a'):>27s}"
        v = []
        for smp in days.values():
            pr = [(F[k], f[5]) for F, f in smp if F.get(k) is not None and 5 in f]
            c = ic([x for x, _ in pr], [y for _, y in pr])
            if c is not None:
                v.append(c)
        print(line + f"{(f'{mean(v):+.2f} ({sum(c > 0 for c in v)}/{len(v)})' if v else 'n/a'):>16s}")

    print("\nsetup by day (5 min):")
    for dn, smp in days.items():
        pr = [(F.get("setup"), f.get(5)) for F, f in smp]
        n = sum(1 for x, y in pr if x and y)
        hit = [((x > 0) == (y > 0)) for x, y in pr if x and y]
        print(f"  {dn:22s} {n:3d} calls  {(f'{100 * mean(hit):.0f}% right' if hit else '')}")

    print("\nshift arrows, raw hit rate per strike side and grid average:")
    sides = []
    for smp_path in files:
        sh = Shift()
        S = load(smp_path)
        for i, (t, d) in enumerate(S):
            if i + 3 >= len(S) or t.hour * 60 + t.minute < 9 * 60 + 25:
                continue
            fw = S[i + 3][1]["underlying_value"] - d["underlying_value"]
            for r in grid_rows(d):
                for p, sg in (("ce", 1), ("pe", -1)):
                    z = sh.z(r.get(p + "_symbol"), r.get(p + "_tbq"), r.get(p + "_tsq"), t.timestamp())
                    if z is not None and tier(z) and fw:
                        sides.append(((sg * z > 0) == (fw > 0), tier(z)))
    for tr in (1, 2):
        v = [h for h, t_ in sides if t_ == tr]
        if v:
            print(f"  per side {'▲▲/▼▼' if tr == 2 else '▲/▼  '} 3 min: {100 * mean(v):.1f}% (n={len(v)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
