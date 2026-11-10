"""Direction-pane series for the Scalper tab: one point per index every 10 s, from the chain the server already
serves (no broker call), and rebuilt from replay/<SYM>_<day>.jsonl when a day has no recording.

Every read reuses bias_eval.py's twins (Shift, walls, gex_total, grid_rows, grid_pcr, composite), so the pane
plots exactly what bias_eval.py scores:

    bs    grid-average 2-min book shift (bullish +), the grid's "book shift"            backtested ~56%
    w     walls lean, +0.5 on the put wall, -0.5 on the call wall                       backtested ~56%
    oth   the other index's bs at the same moment (filled by pair())                   backtested ~60% (few days)
    dir   bias_eval.composite(bs, w, oth): -1..+1                                       backtested ~56%, 8/8 days
  context only (at or below 50% in bias_eval.py, drawn dimmed, not in dir):
    pcr   ln(put OI / call OI) over the grid strikes
    noi   put OI added minus call OI added today over the grid strikes (contracts)
    cvd   near-future cumulative taker delta since the open (buys - sells)
    gex   dealer gamma sign/size (bias_eval.gex_total convention: + = dealers long gamma)

    python3 nse_dirflow.py replay/NIFTY_2026-11-09.jsonl     # print the rebuilt series
"""
import json
import sys

import bias_eval as BE

TWIN = {"NIFTY": "BANKNIFTY", "BANKNIFTY": "NIFTY"}


class Track:
    """Per-index running state: the book-shift EWMA (one bias_eval.Shift per index, keyed by contract)."""

    def __init__(self):
        self.sh = BE.Shift()
        self.day = None


def _r(x, n=4):
    return None if x is None else round(x, n)


def point(d, ts, tr, shift_ts=None):
    """One pane point from a chain snapshot at epoch `ts`. None without strikes or spot. `shift_ts` is the clock the
    book shift runs on (bias_eval uses the snapshot's own timestamp field); `ts` by default."""
    spot = float(d.get("underlying_value") or 0)
    if not spot or not d.get("strikes"):
        return None
    rows = BE.grid_rows(d)
    zs = []
    for r in rows:
        for p, sg in (("ce", 1), ("pe", -1)):
            z = tr.sh.z(r.get(p + "_symbol"), r.get(p + "_tbq"), r.get(p + "_tsq"), ts if shift_ts is None else shift_ts)
            if z is not None:
                zs.append(sg * z)
    bs = sum(zs) / len(zs) if zs else None
    w = BE.walls(d, spot)[2]
    noi = None
    if rows and any(r.get("ce_oi_chg") is not None or r.get("pe_oi_chg") is not None for r in rows):
        noi = sum(r.get("pe_oi_chg") or 0 for r in rows) - sum(r.get("ce_oi_chg") or 0 for r in rows)
    fu = ((d.get("flow") or {}).get("futures") or {}).get("near") or {}
    cvd = (fu.get("bd") or 0) - (fu.get("sd") or 0) if fu.get("bd") is not None or fu.get("sd") is not None else None
    try:
        gex = BE.gex_total(d)
    except Exception:  # noqa: BLE001
        gex = None
    return {"ts": int(ts), "spot": round(spot, 2), "bs": _r(bs), "w": _r(w), "pcr": _r(BE.grid_pcr(d)),
            "noi": noi, "cvd": cvd, "gex": _r(gex, 8), "oth": None, "dir": _r(BE.composite(bs, w, None))}


def finish(pt, oth):
    """Attach the other index's book shift and recompute the composite."""
    pt["oth"] = _r(oth)
    pt["dir"] = _r(BE.composite(pt.get("bs"), pt.get("w"), pt["oth"]))
    return pt


def nearest(points, ts, within=90):
    """The point nearest `ts` (within `within` seconds) of a time-sorted list, else None."""
    best = None
    for p in reversed(points):
        if p["ts"] < ts - within:
            break
        if abs(p["ts"] - ts) <= within and (best is None or abs(p["ts"] - ts) < abs(best["ts"] - ts)):
            best = p
    return best


def pair(points, others):
    """Fill oth / dir of `points` from the other index's points (nearest within 90 s, as bias_eval.pair_other)."""
    j = 0
    for p in points:
        while j + 1 < len(others) and abs(others[j + 1]["ts"] - p["ts"]) <= abs(others[j]["ts"] - p["ts"]):
            j += 1
        o = others[j] if others and abs(others[j]["ts"] - p["ts"]) <= 90 else None
        finish(p, o.get("bs") if o else None)
    return points


def from_replay(path):
    """The series rebuilt from a replay file (about one point a minute). [] when the index never moved."""
    tr, out, spots = Track(), [], set()
    for t, d in BE.load(path):
        ts = d.get("_replay_ts") or t.timestamp()
        pt = point(d, float(ts), tr, shift_ts=t.timestamp())
        if pt:
            out.append(pt)
            spots.add(pt["spot"])
    return out if len(spots) > 1 else []


if __name__ == "__main__":
    for pth in sys.argv[1:]:
        for p_ in from_replay(pth):
            print(json.dumps(p_))
