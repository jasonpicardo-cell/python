"""Replay Auto Trade on recorded chain days (replay/<SYMBOL>_<date>.jsonl) through the REAL paper Engine and
AutoTrader, so what is measured is the code that trades, not a re-implementation of it.

    python3 auto_eval.py                         # NIFTY and BANKNIFTY, every replay day, the standard config set
    python3 auto_eval.py --sym NIFTY --tf 3      # one index, 3-minute candles
    python3 auto_eval.py --mode SELL             # option selling instead of buying
    python3 auto_eval.py --delta 0.55            # strikes by delta instead of the -300 / +300 offsets
    python3 auto_eval.py --only RIBBON           # only the configs whose name starts with this

How a day is replayed
  * Candles: 1-minute bars of the index price built from the snapshots (~1 a minute, so a bar is usually one
    price: ATR and the ribbon's bands run on close-to-close ranges). The previous replay day of the same index, if
    recorded, is loaded first as warm-up, as fetch_bars does live; without it the ribbon needs ~36 bars (~09:51).
  * Fills: the paper engine at each snapshot's option bid / ask, with its own charges. The auto loop is stepped at
    every snapshot and 4 and 8 s after it (entry retries); a signal is read up to a minute late.
  * Filters read the snapshot itself: dealer gamma (gex_total), book shift (grid_book totals, the Python twin of
    _bfZ), day type and VWAP from the candles, the other index from its own replay file of that day.
  * A file that stops before 15:15 is closed at its last snapshot ("cut short").

Read the numbers as a smoke test until there are dozens of full days: a handful of sessions, 1-minute snapshots
where the live loop sees ticks, many configs tried. Anything inside a few trades of zero is noise.
"""
import argparse
import calendar
import glob
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402
import nse_autotrade as A  # noqa: E402
import nse_trade as N  # noqa: E402

IST = 19800
BASE = {"mode": "BUY", "ce_off": -300, "pe_off": 300, "start": "09:20", "end": "15:15", "lots": 2,
        "max_trades": 50, "max_loss": 1e12, "ema_fast": 5, "ema_slow": 15}
WING = {"NIFTY": 300, "BANKNIFTY": 800}           # PREMIUM wing (and tent) per index, about the same % of spot
ALL_GATES = {"gate_day": True, "gate_gex": True, "gate_book": True, "gate_vwap": True, "gate_lead": True}
EXITS = {"flat_exit": True, "time_stop": 10, "scale_out": True}
CONFIGS = [
    ("RIBBON", {"strategy": "RIBBON"}),
    ("RIBBON +day type", {"strategy": "RIBBON", "gate_day": True}),
    ("RIBBON +gamma", {"strategy": "RIBBON", "gate_gex": True}),
    ("RIBBON +book shift", {"strategy": "RIBBON", "gate_book": True}),
    ("RIBBON +VWAP side", {"strategy": "RIBBON", "gate_vwap": True}),
    ("RIBBON +other index", {"strategy": "RIBBON", "gate_lead": True}),
    ("RIBBON +flat exit", {"strategy": "RIBBON", "flat_exit": True}),
    ("RIBBON +time stop 10", {"strategy": "RIBBON", "time_stop": 10}),
    ("RIBBON +scale-out", {"strategy": "RIBBON", "scale_out": True}),
    ("RIBBON +all filters", {"strategy": "RIBBON", **ALL_GATES}),
    ("RIBBON +all exits", {"strategy": "RIBBON", **EXITS}),
    ("EMA 5/15", {"strategy": "EMA"}),
    ("EMA +day +VWAP +flat", {"strategy": "EMA", "gate_day": True, "gate_vwap": True, "flat_exit": True}),
    ("ORB 15", {"strategy": "ORB", "orb_min": 15}),
    ("ORB 30", {"strategy": "ORB", "orb_min": 30}),
    ("ORB 15 +book shift", {"strategy": "ORB", "orb_min": 15, "gate_book": True}),
    ("VWAP sigma", {"strategy": "VWAP"}),
    ("PDH/PDL trap", {"strategy": "TRAP"}),
    ("PDH/PDL trap +gamma", {"strategy": "TRAP", "gate_gex": True}),
    ("RIBBON +RV/IV", {"strategy": "RIBBON", "gate_vol": True}),
    ("RIBBON +fut flow", {"strategy": "RIBBON", "gate_flow": True}),
    ("RIBBON +OI writers", {"strategy": "RIBBON", "gate_oi": True}),
    ("RIBBON +stop 1 trail 1.5", {"strategy": "RIBBON", "tm_sl_atr": 1.0, "tm_trail_atr": 1.5}),
    ("RIBBON +B/E 1 ATR", {"strategy": "RIBBON", "tm_sl_atr": 1.0, "tm_be_atr": 1.0}),
    ("RIBBON +stop +re-entry", {"strategy": "RIBBON", "tm_sl_atr": 1.0, "reentry": True}),
    ("RIBBON +profit lock 5k", {"strategy": "RIBBON", "lock_at": 5000, "lock_give": 50}),
    ("PREMIUM iron fly", {"strategy": "PREMIUM", "prem_kind": "FLY"}),
    ("PREMIUM straddle", {"strategy": "PREMIUM", "prem_kind": "STRADDLE"}),
    ("PREMIUM fly +RV/IV", {"strategy": "PREMIUM", "prem_kind": "FLY", "gate_vol": True}),
]


def load(path):
    """[(epoch, snapshot)] in time order."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
                t = datetime.strptime(d["timestamp"], "%d-%b-%Y %H:%M:%S")
            except Exception:  # noqa: BLE001
                continue
            if d.get("strikes") and d.get("underlying_value"):
                out.append((calendar.timegm(t.timetuple()) - IST, d))
    out.sort(key=lambda x: x[0])
    return out


def minute_bars(snaps):
    bars = {}
    for t, d in snaps:
        m = t - t % 60
        p = float(d["underlying_value"])
        b = bars.get(m)
        if b is None:
            bars[m] = {"t": m, "o": p, "h": p, "l": p, "c": p, "v": 0}
        else:
            b["h"], b["l"], b["c"] = max(b["h"], p), min(b["l"], p), p
    return [bars[k] for k in sorted(bars) if A.OPEN_MIN <= A.ist_min(k) < 15 * 60 + 30]


def resample(bars, tf):
    """tf-minute candles anchored at 09:15 IST, as Kite's chart."""
    if tf == 1:
        return bars
    out = {}
    for b in bars:
        m = A.ist_min(b["t"])
        k = b["t"] - ((m - A.OPEN_MIN) % tf) * 60
        o = out.get(k)
        if o is None:
            out[k] = dict(b, t=k)
        else:
            o["h"], o["l"], o["c"] = max(o["h"], b["h"]), min(o["l"], b["l"]), b["c"]
    return [out[k] for k in sorted(out)]


def _inst(sym):
    """The engine's key for a chain row's contract (the recorder stores it with or without the exchange)."""
    return None if not sym else sym if ":" in sym else "NFO:" + sym


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


class ReplayFeed:
    """The engine's feed, answered from the current snapshot."""

    def __init__(self, clk, sym):
        self.clk, self.sym, self.d, self.rows, self.inst = clk, sym, None, {}, {}

    def set(self, d):
        self.d = d
        self.rows = {r["strike"]: r for r in d["strikes"]}

    def _row(self, inst):
        k, typ = self.inst.get(inst, (None, None))
        r = self.rows.get(k)
        return (r, typ.lower()) if r and _inst(r.get(typ.lower() + "_symbol")) == inst else (None, None)

    def quote(self, inst):
        r, p = self._row(inst)
        if not r:
            return None
        ltp = r.get(p + "_ltp") or 0
        bid, ask = r.get(p + "_bid") or 0, r.get(p + "_ask") or 0
        if not (bid and ask) and ltp:
            bid, ask = max(0.05, ltp - 0.05), ltp + 0.05
        return {"bid": bid, "ask": ask, "ltp": ltp or (bid + ask) / 2, "ts": self.clk()} if bid and ask else None

    def meta(self, inst):
        k, typ = self.inst.get(inst, (None, None))
        if k is None:
            return None
        return dict(under=self.sym, typ=typ, strike=k, expiry=self.d.get("expiry"), lot=int(self.d.get("lot_size") or 1),
                    ts=inst.split(":")[-1])

    def spot(self, u):
        return self.d.get("underlying_value") if self.d and u == self.sym else None

    def depth(self, inst):
        q = self.quote(inst)
        return {"asks": [{"p": q["ask"], "q": 10 ** 6}], "bids": [{"p": q["bid"], "q": 10 ** 6}]} if q else {}

    def pick(self, sym, typ, off, delta=None):
        d, p = self.d, typ.lower()
        ks = sorted(k for k, r in self.rows.items() if not k % 100 and r.get(p + "_symbol"))
        if not ks:
            raise ValueError("no strikes in the snapshot")
        atm = min(ks, key=lambda k: abs(k - d["underlying_value"]))
        if delta is not None:
            kd = A.pick_by_delta(d, typ, delta)
            if kd is None:
                raise ValueError("no delta strike")
            off = kd - atm
        k = min(ks, key=lambda x: abs(x - (atm + off)))
        inst = _inst(self.rows[k][p + "_symbol"])
        self.inst[inst] = (k, typ)
        return inst, k, atm


def day_kind(bars):
    i0 = A.today_start(bars)
    k, d = A.day_type_series(bars[i0:])[-1] if bars else ("unknown", 0)
    return k if k != "trend" else ("trend up" if d > 0 else "trend down")


def run_day(sym, snaps, warm, other, cfg, tf=1):
    """-> (auto closes [{net, why, ...}], cut_short, entries). A scale-out is a close of its own. `warm` = previous session's 1-minute bars, `other` = the other
    index's 1-minute bars of the day (warm-up included) for the lead filter."""
    clk = Clock()
    feed = ReplayFeed(clk, sym)
    bars = resample(warm + minute_bars(snaps), tf)
    obars = resample(other, tf) if other else []

    def bars_fn(s, _tf):
        src = bars if s == sym else obars
        return [b for b in src if b["t"] <= clk.t]

    clk.t = snaps[0][0]                     # before the trader is built: it keys its trading day off the clock
    feed.set(snaps[0][1])
    eng = N.Engine(feed, None, N.Config(env={"NSE_TRADE_MODE": "paper"}), clock=clk, async_exits=False, sleep=lambda s: None)
    a = A.AutoTrader(eng, bars_fn, feed.pick, None, clock=clk,
                     gex_fn=lambda s: A.gex_total(feed.d), book_fn=lambda s: A.grid_book(feed.d), chain_fn=lambda s: feed.d,
                     flow_fn=lambda s: ((feed.d.get("flow") or {}).get("futures") or {}).get("near"))
    if cfg.get("strategy") == "PREMIUM" and "prem_wing" not in cfg:
        cfg = dict(cfg, prem_wing=WING.get(sym, 300))
    r = a.configure({**BASE, **cfg, "sym": sym, "tf": tf, "enabled": True})
    if not r.get("ok"):
        raise SystemExit(f"config refused: {r}")
    for t, d in snaps:
        feed.set(d)
        for dt in (0, 4, 8):
            clk.t = t + dt
            eng.step()                      # the engine's own watcher: stops, trails, structure stop / target
            a.step()
    cut = bool(a.holding())
    if cut:
        for inst, _p in a.holding():
            eng.exit(inst, 100, "auto: cut short (replay ended)", urgent=True, wait=True)
    return [x for x in eng.trades if x.get("src") == A.SRC], cut, a.entries


def files_for(d, syms):
    return {os.path.basename(f)[:-6]: f for f in sorted(glob.glob(os.path.join(d, "*.jsonl")))
            if os.path.basename(f).split("_")[0] in syms and os.path.getsize(f) > 1_000_000}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", nargs="*", default=["NIFTY", "BANKNIFTY"])
    ap.add_argument("--dir", default=os.path.join(HERE, "replay"))
    ap.add_argument("--tf", type=int, default=1, choices=(1, 2, 3, 5, 10, 15))
    ap.add_argument("--mode", default="BUY", choices=("BUY", "SELL"))
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    files = files_for(a.dir, set(a.sym) | {"NIFTY", "BANKNIFTY"})
    if not any(k.split("_")[0] in a.sym for k in files):
        print("no replay files")
        return 1
    snaps = {k: load(f) for k, f in files.items()}
    snaps = {k: v for k, v in snaps.items() if len(v) >= 30}
    mbars = {k: minute_bars(v) for k, v in snaps.items()}

    def prev_key(k):
        s, day = k.split("_", 1)
        older = sorted(x for x in mbars if x.split("_")[0] == s and x.split("_", 1)[1] < day)
        return older[-1] if older else None

    days = sorted((k for k in snaps if k.split("_")[0] in a.sym), key=lambda k: (k.split("_", 1)[1], k))
    extra = {"mode": a.mode}
    if a.delta:
        extra.update(strike="DELTA", delta=a.delta)
    print(f"Auto Trade replay  tf={a.tf}m  mode={a.mode}  strikes={'delta ' + str(a.delta) if a.delta else 'CE -300 / PE +300'}  lots={BASE['lots']}")
    info = []
    for k in days:
        pk = prev_key(k)
        warm = mbars[pk] if pk else []
        full = warm + mbars[k]
        s_, t0, t1 = snaps[k], A.ist_min(snaps[k][0][0]), A.ist_min(snaps[k][-1][0])
        info.append((k, f"{t0 // 60:02d}:{t0 % 60:02d}-{t1 // 60:02d}:{t1 % 60:02d}", len(s_), day_kind(full), "warm" if pk else "cold"))
    print("days: " + "; ".join(f"{k} {w} ({n} snaps, {kind}, {wm})" for k, w, n, kind, wm in info))
    kinds = {k: kind for k, _w, _n, kind, _wm in info}

    rows = []
    for name, cfg in CONFIGS:
        if a.only and not name.upper().startswith(a.only.upper()):
            continue
        per, cut_n, ent = {}, 0, 0
        for k in days:
            s, day = k.split("_", 1)
            pk = prev_key(k)
            ok_ = f"{'BANKNIFTY' if s == 'NIFTY' else 'NIFTY'}_{day}"
            other = ((mbars.get(prev_key(ok_)) or []) + mbars[ok_]) if ok_ in mbars else []
            tr, cut, n_ent = run_day(s, snaps[k], mbars[pk] if pk else [], other, {**cfg, **extra}, a.tf)
            per[k] = tr
            cut_n += cut
            ent += n_ent
        allt = [t for v in per.values() for t in v]
        net = sum(t["net"] for t in allt)
        win = sum(1 for t in allt if t["net"] > 0)
        gw, gl = sum(t["net"] for t in allt if t["net"] > 0), -sum(t["net"] for t in allt if t["net"] < 0)
        by_kind = {}
        for k, v in per.items():
            kk = "trend" if kinds[k].startswith("trend ") else "range" if kinds[k] in ("range", "trend-broken") else "other"
            by_kind[kk] = by_kind.get(kk, 0.0) + sum(t["net"] for t in v)
        rows.append((name, ent, len(allt), win, net, (gw / gl) if gl else None, by_kind, cut_n))
    print(f"\n{'config':24s}{'entries':>8s}{'win%':>6s}{'net Rs':>11s}{'PF':>6s}{'trend days':>12s}{'range days':>12s}{'other':>10s}")
    for name, ent, n, win, net, pf, bk, cut in rows:
        print(f"{name:24s}{ent:8d}{(f'{100 * win / n:.0f}' if n else '-'):>6s}{net:11,.0f}{(f'{pf:.2f}' if pf else '-'):>6s}"
              f"{bk.get('trend', 0):12,.0f}{bk.get('range', 0):12,.0f}{bk.get('other', 0):10,.0f}" + (f"   ({cut} cut short)" if cut else ""))
    print("\nwin% = share of closes (a scale-out counts as its own close) in profit. net = after the paper engine's charges, all lots. 'other' = day type unknown/forming (file starts after 9:20).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
