"""
nse_trade.py - the order engine behind the Scalping tab.

Long AND short, with no size or risk limits: each order goes out as pressed. A
short (sell a call or put to open) has no ceiling on its loss, so set a stop or
HEDGE it - the engine buys the far out-of-the-money wing first (so the margin
benefit is real and the loss is capped), then sells, and sells the wing back only
after the short is covered. Structures (straddle, strangle) open as one unit with a
combined stop/target in rupees. Conditional entries fire when a premium or the
index crosses a level.

Two modes
  PAPER (default)  fills against the live bid/ask, walking the 5-level book for
                   the size, with the estimated charges. No broker call at all.
  LIVE             NSE_TRADE_MODE=live AND the tab's ARM switch. Orders go out as
                   marketable LIMIT, IOC (never a market order, never resting):
                   buy at ask + a small buffer, sell at bid - a small buffer.

Server-side, whatever the page does: an order-rate cap, a kill switch, de-duplication
of double clicks, reduce-only exits, a fresh-quote check, and the entry window and
15:14 square-off.

Stops, targets, the trailing stop, the breakeven move, the partial target,
the underlying-level exits, the time stop and the 15:14 square-off all run in
the server's own loop on the live tick feed - they keep working if the browser
tab is closed or the laptop's screen sleeps. They need NO broker call until
they fire. If the feed goes stale the position says so: stops that cannot see
a price cannot protect you.

Nothing here talks to a broker directly: the server gives the engine a `feed`
(quotes, depth, option metadata, spot) and, for LIVE, a `broker`.
"""
import json
import math
import os
import threading
import time

IST = 19800
TICK = 0.05

# Zerodha, equity options (zerodha.com/charges): flat 20 per executed order;
# STT 0.15% on the SELL premium; NSE transaction 0.03553%; SEBI 10/crore;
# stamp 0.003% on the BUY side; GST 18% on brokerage + SEBI + transaction.
# An ESTIMATE - rates change; edit here if they do.
CHARGES = {"brokerage": 20.0, "stt_sell": 0.0015, "txn": 0.0003553, "sebi": 10.0 / 1e7,
           "stamp_buy": 0.00003, "ipft": 0.01 / 1e7, "gst": 0.18}


def _r(x, up=False):
    """To the 0.05 tick (up = never below x, for a buy limit)."""
    n = x / TICK
    n = math.ceil(n - 1e-9) if up else round(n)
    return round(max(TICK, n * TICK), 2)


def est_charges(buy_val, sell_val, orders):
    """Estimated total charges (rupees) for these executed orders."""
    C = CHARGES
    brok = C["brokerage"] * orders
    txn = (buy_val + sell_val) * C["txn"]
    sebi = (buy_val + sell_val) * C["sebi"]
    gst = (brok + txn + sebi) * C["gst"]
    return brok + txn + sebi + gst + sell_val * C["stt_sell"] + buy_val * C["stamp_buy"] + (buy_val + sell_val) * C["ipft"]


def ist_clock(now):
    g = time.gmtime(now + IST)
    return g.tm_hour * 60 + g.tm_min, g.tm_wday


def _hm(s, default):
    try:
        h, m = str(s).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default


class Config:
    """Deliberately small: the only settings are NSE_TRADE_MODE (paper | live) and an
    optional NSE_TRADE_PIN. There are no size or risk limits - what you press is what is
    sent. The fields below are the engine's fixed behaviour (0 = off); tests set them
    directly, nothing reads them from env."""
    def __init__(self, env=None):
        e = (env if env is not None else os.environ).get
        self.mode = "live" if str(e("NSE_TRADE_MODE") or "paper").strip().lower() == "live" else "paper"
        self.pin = (e("NSE_TRADE_PIN") or "").strip()
        # no limits
        self.max_lots = 0
        self.max_open_lots = 0
        self.max_value = 0
        self.day_loss = 0
        self.max_trades = 0
        self.loss_streak = 0
        self.cooldown_s = 0
        self.max_spread_pct = 0
        self.min_premium = 0
        self.allow_short = True
        self.max_short_lots = 0
        self.max_risk = 0
        self.day_target = 0
        self.giveback_pct = 0
        self.giveback_min = 0
        self.loss_flatten = True
        # fixed execution behaviour
        self.slip_pct = 1.5             # limit buffer beyond the bid/ask, % of price
        self.slip_min = 0.5
        self.ops = 5                    # orders per second (Zerodha allows 10)
        self.stale_s = 6                # a quote older than this is flagged NO PRICE
        self.entry_from = 9 * 60 + 15
        self.entry_to = 15 * 60 + 14
        self.flatten_at = 15 * 60 + 14  # MIS is squared off here, before Zerodha's own 15:20 square-off
        self.reconcile_s = 20

    def public(self):
        d = {k: v for k, v in self.__dict__.items() if k != "pin"}
        d["pin_required"] = bool(self.pin)
        return d


class Pos:
    """One open contract. side +1 = long, -1 = short; qty is always positive."""

    def __init__(self, inst, meta, qty, avg, now, src="manual", side=1, role="main", hedge_for=None, grp=None, tag=""):
        self.inst, self.meta = inst, meta
        self.qty, self.avg, self.opened = int(qty), float(avg), now
        self.cost = float(avg) * int(qty)
        self.src = src
        self.side = -1 if side < 0 else 1
        self.role, self.hedge_for, self.grp, self.tag = role, hedge_for, grp, tag
        self.hedge_armed = role != "hedge"          # a wing is released only after its short has opened and closed again
        self.sl = self.tp = self.tp1 = self.be_at = None
        self.tp1_pct = 50
        self.trail = None
        self.best = self.worst = float(avg)         # most favourable / most adverse price seen (max/min for a long, min/max for a short)
        self.u_stop = self.u_target = None
        self.u_dir = 0
        self.time_stop = None
        self.tp1_done = False
        self.exiting = False
        self.exit_note = ""
        self.retry_at = 0.0
        self.stale = False
        self.managed = True
        self.risk_ps = 0.0                          # initial stop distance per unit (for R multiples)
        self.slip = 0.0                             # rupees given up to slippage getting in
        self.sig = float(avg)                       # the price we were aiming at when we decided to enter
        self.fills = [(now, "BUY" if self.side > 0 else "SELL", int(qty), float(avg))]

    @property
    def peak(self):                                 # kept for older callers: the best price so far
        return self.best

    def eff_sl(self):
        if self.side > 0:
            v = [x for x in (self.sl, (self.best - self.trail) if self.trail else None) if x is not None]
            return max(v) if v else None
        v = [x for x in (self.sl, (self.best + self.trail) if self.trail else None) if x is not None]
        return min(v) if v else None

    def exits(self):
        return {"sl": self.sl, "tp": self.tp, "tp1": self.tp1, "tp1_pct": self.tp1_pct, "trail": self.trail,
                "be_at": self.be_at, "u_stop": self.u_stop, "u_target": self.u_target, "u_dir": self.u_dir,
                "time_stop": self.time_stop, "tp1_done": self.tp1_done}


class Engine:
    MAX_PENDING = 8

    def __init__(self, feed, broker=None, cfg=None, clock=time.time, journal_dir=None, async_exits=True, sleep=time.sleep):
        self.feed, self.broker, self.cfg = feed, broker, cfg or Config()
        self.clock, self.sleep = clock, sleep
        self.async_exits = async_exits
        self.lock = threading.RLock()
        self.jdir = journal_dir
        self.day = self._day()
        self.armed = False
        self.kill = False
        self.kill_why = ""
        self.halt = ""                      # set when the book may no longer match the broker
        self.pos = {}
        self.orders = []                    # recent order log (newest last)
        self.trades = []                    # closed round trips today
        self.gross = 0.0
        self.charges = 0.0
        self.streak = 0
        self.cool_until = 0.0
        self.day_peak = 0.0                 # best net P&L of the day so far (for the give-back rule)
        self.inflight = set()
        self.seen = {}                      # req_id -> (time, result)
        self.ops_t = []
        self.last_recon = 0.0
        self.warn = []
        self.pending = []                   # conditional entries waiting for their level
        self.pend_log = []                  # what happened to the last few of them
        self.groups = {}                    # structures (straddle, strangle): legs + combined stop / target
        self._seq = 0
        self._stop = threading.Event()
        self._thread = None

    # ── plumbing ──────────────────────────────────────────────────
    def _day(self):
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock() + IST))

    def live(self):
        return self.cfg.mode == "live"

    def _id(self, prefix):
        self._seq += 1
        return f"{prefix}{int(self.clock()) % 100000}{self._seq}"

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="trade-engine")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(0.25):
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                print(f"[trade] step failed: {type(e).__name__}: {e}")

    def log(self, ev, **kw):
        row = {"t": round(self.clock(), 3), "ev": ev, "mode": self.cfg.mode, **kw}
        if self.jdir:
            try:
                os.makedirs(self.jdir, exist_ok=True)
                with open(os.path.join(self.jdir, f"{self._day()}.jsonl"), "a") as f:
                    f.write(json.dumps(row, default=str, separators=(",", ":")) + "\n")
            except Exception:
                pass
        return row

    def _roll_day(self):
        d = self._day()
        if d != self.day:
            with self.lock:
                if d != self.day and not self.pos:
                    self.day = d
                    self.trades, self.orders = [], []
                    self.gross = self.charges = 0.0
                    self.streak, self.cool_until, self.day_peak = 0, 0.0, 0.0
                    self.pending, self.pend_log, self.groups = [], [], {}
                    self.kill, self.kill_why, self.armed = False, "", False
                    self.halt = ""

    def _ops_ok(self):
        now = self.clock()
        with self.lock:
            self.ops_t = [t for t in self.ops_t if now - t < 1.0]
            if len(self.ops_t) >= max(1, int(self.cfg.ops)):
                return False
            self.ops_t.append(now)
            return True

    def _quote(self, inst):
        q = self.feed.quote(inst)
        if not q:
            return None
        q = dict(q)
        q["age"] = self.clock() - float(q.get("ts") or 0)
        q["fresh"] = q["age"] <= self.cfg.stale_s
        return q

    # ── money ─────────────────────────────────────────────────────
    @staticmethod
    def _mark(p, q):
        """What closing this position would cost / return right now: a long sells
        at the bid, a short buys back at the ask."""
        q = q or {}
        px = (q.get("bid") if p.side > 0 else q.get("ask")) or q.get("ltp") or p.avg
        return px

    def _pnl(self, p, q=None):
        q = q if q is not None else (self.feed.quote(p.inst) or {})
        return (self._mark(p, q) - p.avg) * p.qty * p.side

    def unrealized(self):
        return sum(self._pnl(p) for p in self.pos.values())

    def net_day(self):
        return self.gross - self.charges + self.unrealized()

    # ── order placement (paper or live) ───────────────────────────
    def _walk(self, levels, qty):
        """(filled_qty, avg_price) taking `qty` from a list of {p,q} levels."""
        got, val = 0, 0.0
        for lv in levels or []:
            take = min(qty - got, int(lv.get("q") or 0))
            if take > 0:
                got += take
                val += take * float(lv["p"])
            if got >= qty:
                break
        return got, (val / got if got else 0.0)

    def _send(self, inst, side, qty, q, urgent=False, widen=1.0):
        """One marketable IOC order. -> dict(ok, qty, price, order_id, status, msg)"""
        meta = self.feed.meta(inst) or {}
        buf = max(self.cfg.slip_min, (q.get("ask") or q.get("ltp") or 0) * self.cfg.slip_pct / 100.0) * widen
        if side == "BUY":
            ref = q.get("ask") or (q.get("ltp") or 0) * (1.05 if urgent else 1.0)
            limit = _r(ref + buf, up=True)
        else:
            ref = q.get("bid") or (q.get("ltp") or 0) * (0.95 if urgent else 1.0)
            limit = max(TICK, _r(ref - buf))
        if not self._ops_ok():
            return {"ok": False, "qty": 0, "price": 0, "order_id": None, "status": "RATE", "msg": f"order-rate cap ({self.cfg.ops:g}/s)"}
        if not self.live():
            lv = self.feed.depth(inst) or {}
            levels = (lv.get("asks") if side == "BUY" else lv.get("bids")) or []
            if not levels:
                p0 = q.get("ask") if side == "BUY" else q.get("bid")
                if not p0:
                    return {"ok": False, "qty": 0, "price": 0, "order_id": None, "status": "NOQUOTE", "msg": "no book to fill against"}
                levels = [{"p": p0, "q": 10 ** 9}]
            ok_levels = [l for l in levels if (l["p"] <= limit if side == "BUY" else l["p"] >= limit)]
            got, avg = self._walk(ok_levels, qty)
            oid = f"P{int(self.clock() * 1000) % 10 ** 9}"
            if not got:
                return {"ok": False, "qty": 0, "price": 0, "order_id": oid, "status": "UNFILLED", "msg": f"nothing at or better than {limit}"}
            return {"ok": True, "qty": got, "price": round(avg, 4), "order_id": oid, "status": "COMPLETE", "msg": "paper fill"}
        if not self.broker:
            return {"ok": False, "qty": 0, "price": 0, "order_id": None, "status": "NOBROKER", "msg": "no broker connection"}
        try:
            oid = self.broker.place(meta.get("ts") or inst.split(":")[-1], side, qty, limit, tag="nsescalp")
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "qty": 0, "price": 0, "order_id": None, "status": "REJECTED", "msg": f"{type(e).__name__}: {e}"}
        # confirm: an IOC order is final within a moment
        last = {}
        for _ in range(16):
            self.sleep(0.25)
            try:
                last = self.broker.status(oid) or {}
            except Exception as e:  # noqa: BLE001
                last = {"status": "UNKNOWN", "msg": str(e)}
            if last.get("status") in ("COMPLETE", "REJECTED", "CANCELLED"):
                break
        st = last.get("status") or "UNKNOWN"
        fq = int(last.get("filled") or 0)
        if st in ("COMPLETE", "CANCELLED", "REJECTED") and fq > 0:
            return {"ok": True, "qty": fq, "price": float(last.get("avg") or limit), "order_id": oid, "status": st, "msg": last.get("msg") or ""}
        if st in ("CANCELLED", "REJECTED"):
            return {"ok": False, "qty": 0, "price": 0, "order_id": oid, "status": st, "msg": last.get("msg") or "not filled"}
        # still open / unknown after ~4 s: cannot say what happened
        self.halt = f"order {oid} ({side} {inst}) is {st} - check Kite, then press Reconcile"
        self.log("halt", why=self.halt)
        return {"ok": False, "qty": 0, "price": 0, "order_id": oid, "status": st, "msg": self.halt, "unknown": True}

    def _note_order(self, inst, side, qty, res, why="", act="open"):
        row = {"t": round(self.clock(), 2), "inst": inst, "side": side, "qty": qty, "filled": res.get("qty", 0), "act": act,
               "price": res.get("price"), "status": res.get("status"), "msg": res.get("msg", ""), "id": res.get("order_id"), "why": why}
        self.orders.append(row)
        del self.orders[:-60]
        self.log("order", **row)

    # ── entry ─────────────────────────────────────────────────────
    def _lots_of(self, p):
        return p.qty // max(1, int((p.meta or {}).get("lot") or 1))

    def _blocked(self, role="main"):
        """Why NO entry is allowed right now (whatever the contract), or None."""
        c, now = self.cfg, self.clock()
        mins, wd = ist_clock(now)
        if self.kill:
            return f"kill switch is ON ({self.kill_why or 'manual'})"
        if self.halt:
            return f"halted: {self.halt}"
        if self.live() and not self.armed:
            return "LIVE mode is not armed - press ARM"
        if wd >= 5 or not (c.entry_from <= mins < c.entry_to):
            return f"entries only {c.entry_from // 60:02d}:{c.entry_from % 60:02d}-{c.entry_to // 60:02d}:{c.entry_to % 60:02d} IST on weekdays"
        if role == "main":
            if now < self.cool_until:
                return f"cooling off after {c.loss_streak} losses in a row - {int(self.cool_until - now)}s left"
            if c.max_trades and len(self.trades) >= c.max_trades:
                return f"max trades for the day ({c.max_trades}) reached"
            net = self.net_day()
            if c.day_loss and net <= -c.day_loss:
                return f"daily loss limit (Rs {c.day_loss:,.0f}) reached"
            if c.day_target and net >= c.day_target:
                return f"daily target (Rs {c.day_target:,.0f}) reached - the day is locked in"
            if c.giveback_pct and self.day_peak >= c.giveback_min and net <= self.day_peak * (1 - c.giveback_pct / 100.0):
                return f"gave back {c.giveback_pct:g}% of the day's peak (Rs {self.day_peak:,.0f} -> Rs {net:,.0f}) - done for the day"
        return None

    def check_entry(self, inst, lots, req_id=None, side="BUY", role="main", exits=None, extra_lots=0, extra_short=0,
                    stop_ok=False, static_only=False):
        """None when an entry is allowed, else the reason. `static_only` skips the
        checks that depend on the moment (window, cooldown, day P&L, kill, arm)."""
        c = self.cfg
        if not static_only:
            why = self._blocked(role)
            if why:
                return why
        meta = self.feed.meta(inst)
        if not meta or meta.get("typ") not in ("CE", "PE"):
            return "not a listed index option"
        if not isinstance(lots, int) or isinstance(lots, bool) or lots < 1:
            return "lots must be a whole number >= 1"
        if role == "hedge":
            return None                     # the main entry already vetted the caps for it
        if c.max_lots and lots > c.max_lots:
            return f"{lots} lots exceeds the per-order cap of {c.max_lots}"
        open_lots = sum(self._lots_of(p) for p in self.pos.values())
        if c.max_open_lots and open_lots + lots + extra_lots > c.max_open_lots:
            return f"open lots would be {open_lots + lots + extra_lots}; cap is {c.max_open_lots}"
        have = self.pos.get(inst)
        if have and have.side != (1 if side == "BUY" else -1):
            return "you hold the opposite side of this contract - close it first (or use Flip)"
        if side == "SELL":
            if not c.allow_short:
                return "short selling is off"
            if c.max_short_lots and lots > c.max_short_lots:
                return f"{lots} lots exceeds the short cap of {c.max_short_lots}"
            short_lots = sum(self._lots_of(p) for p in self.pos.values() if p.side < 0 and p.role == "main")
            if c.max_short_lots and short_lots + lots + extra_short > c.max_short_lots:
                return f"short lots would be {short_lots + lots + extra_short}; the short cap is {c.max_short_lots}"
        if inst in self.inflight:
            return "an order for this contract is already in flight"
        return None

    def _stop_distance(self, side, q, exits):
        """Per-unit distance from the likely fill to the stop, or None if no stop is set."""
        ref = (q.get("ask") if side == "BUY" else q.get("bid")) or q.get("ltp") or 0
        if exits.get("sl"):
            return (ref - float(exits["sl"])) if side == "BUY" else (float(exits["sl"]) - ref)
        if exits.get("sl_pts"):
            return float(exits["sl_pts"])
        return None

    def buy(self, inst, lots, exits=None, req_id=None, src="manual", **kw):
        """-> dict(ok, msg, ...). `exits` = absolute levels / points, see _arm_exits."""
        return self._enter(inst, "BUY", lots, exits, req_id, src, **kw)

    def sell(self, inst, lots, exits=None, req_id=None, src="manual", **kw):
        """Sell to OPEN (a short). Needs a stop; see check_entry."""
        return self._enter(inst, "SELL", lots, exits, req_id, src, **kw)

    def _enter(self, inst, side, lots, exits=None, req_id=None, src="manual", tag="", role="main", hedge_for=None, grp=None,
               extra_lots=0, extra_short=0, stop_ok=False):
        self._roll_day()
        now = self.clock()
        exits = exits or {}
        sgn = 1 if side == "BUY" else -1
        with self.lock:
            if req_id and req_id in self.seen and now - self.seen[req_id][0] < 15:
                return dict(self.seen[req_id][1], dup=True)
            why = self.check_entry(inst, lots, req_id, side=side, role=role, exits=exits, extra_lots=extra_lots,
                                   extra_short=extra_short, stop_ok=stop_ok)
            if why:
                return self._done(req_id, {"ok": False, "msg": why})
            meta = self.feed.meta(inst)
            qty = lots * int(meta["lot"])
            q = self._quote(inst)
            if not q or not q.get("fresh"):
                return self._done(req_id, {"ok": False, "msg": "no fresh quote for this contract"})
            ask, bid = q.get("ask") or 0, q.get("bid") or 0
            if role == "hedge":
                if not ask:
                    return self._done(req_id, {"ok": False, "msg": "the hedge contract has no offer to buy"})
                ref = ask
            else:
                if not ask or not bid or ask < bid:
                    return self._done(req_id, {"ok": False, "msg": "no two-sided market (bid/ask missing)"})
                ref = ask if side == "BUY" else bid
                if ref < self.cfg.min_premium:
                    return self._done(req_id, {"ok": False, "msg": f"premium {ref} is below the minimum {self.cfg.min_premium}"})
                spr = (ask - bid) / ask * 100
                if self.cfg.max_spread_pct and spr > self.cfg.max_spread_pct:
                    return self._done(req_id, {"ok": False, "msg": f"spread {spr:.1f}% is wider than {self.cfg.max_spread_pct:g}%"})
            if self.cfg.max_value and ref * qty > self.cfg.max_value:
                return self._done(req_id, {"ok": False, "msg": f"order value Rs {ref * qty:,.0f} exceeds Rs {self.cfg.max_value:,.0f}"})
            if side == "SELL" and exits.get("sl") and float(exits["sl"]) <= ask + TICK:
                return self._done(req_id, {"ok": False, "msg": f"stop {float(exits['sl']):g} is inside the market (ask {ask:g}) - set it above"})
            if side == "BUY" and exits.get("sl") and float(exits["sl"]) >= bid - TICK and role == "main":
                return self._done(req_id, {"ok": False, "msg": f"stop {float(exits['sl']):g} is inside the market (bid {bid:g}) - set it below"})
            sd = self._stop_distance(side, q, exits) if role == "main" else None
            if sd is not None and sd <= 0:
                return self._done(req_id, {"ok": False, "msg": "the stop is on the wrong side of the price"})
            if sd and self.cfg.max_risk and sd * qty > self.cfg.max_risk:
                return self._done(req_id, {"ok": False, "msg": f"risk at the stop Rs {sd * qty:,.0f} exceeds the per-order limit Rs {self.cfg.max_risk:,.0f}"})
            self.inflight.add(inst)
        try:
            res = self._send(inst, side, qty, q)
        finally:
            with self.lock:
                self.inflight.discard(inst)
        with self.lock:
            self._note_order(inst, side, qty, res, src)
            if not res["ok"]:
                return self._done(req_id, {"ok": False, "msg": res["msg"] or res["status"], "status": res["status"]})
            fill = res["price"]
            slip = ((fill - ref) if side == "BUY" else (ref - fill)) * res["qty"]
            p = self.pos.get(inst)
            if p:
                p.cost += fill * res["qty"]
                p.qty += res["qty"]
                p.avg = p.cost / p.qty
                p.fills.append((now, side, res["qty"], fill))
                p.slip += slip
            else:
                p = Pos(inst, meta, res["qty"], fill, now, src, side=sgn, role=role, hedge_for=hedge_for, grp=grp, tag=tag)
                p.slip, p.sig = slip, ref
                self.pos[inst] = p
            self._arm_exits(p, exits)
            if role == "main" and p.sl:
                p.risk_ps = abs(p.avg - p.sl)
            if tag:
                p.tag = tag
            self.log("fill", inst=inst, side=side, qty=res["qty"], price=fill, exits=p.exits(), src=src, role=role, tag=tag)
            verb = "bought" if side == "BUY" else "sold"
            return self._done(req_id, {"ok": True, "msg": f"{verb} {res['qty']} @ {fill:.2f}", "qty": res["qty"],
                                       "price": fill, "partial": res["qty"] < qty, "inst": inst})

    def place(self, spec, req_id=None, src="manual"):
        """The single entry point the page uses. spec: inst, side, lots, exits, tag,
        optional hedge (a further-OTM contract of the same type), or legs for a structure."""
        side = "SELL" if str(spec.get("side") or "BUY").upper().startswith("S") else "BUY"
        lots, exits, tag = spec.get("lots"), spec.get("exits") or {}, str(spec.get("tag") or "")[:24]
        if spec.get("legs"):
            return self._enter_structure(spec["legs"], lots, exits, float(spec.get("g_sl") or 0), float(spec.get("g_tp") or 0),
                                         req_id, src, tag, str(spec.get("kind") or "")[:16])
        if side == "SELL" and spec.get("hedge"):
            return self._enter_hedged(spec["inst"], lots, exits, spec["hedge"], req_id, src, tag)
        return self._enter(spec["inst"], side, lots, exits, req_id, src, tag=tag)

    def _enter_hedged(self, inst, lots, exits, hedge, req_id, src, tag):
        """Buy the wing FIRST, then sell the short; if the short does not fill, sell the wing back."""
        self._roll_day()
        now = self.clock()
        with self.lock:
            if req_id and req_id in self.seen and now - self.seen[req_id][0] < 15:
                return dict(self.seen[req_id][1], dup=True)
            why = self.check_entry(inst, lots, side="SELL", exits=exits, extra_lots=lots if isinstance(lots, int) else 0)
            if not why and hedge in self.pos:
                why = "you already hold the hedge contract - pick a different wing"
            if not why:
                m, hm = self.feed.meta(inst) or {}, self.feed.meta(hedge) or {}
                if not hm or hm.get("typ") != m.get("typ") or hm.get("under") != m.get("under") or hm.get("expiry") != m.get("expiry"):
                    why = "the hedge must be the same index, expiry and call/put as the short"
                elif not ((hm.get("strike") > m.get("strike")) if m.get("typ") == "CE" else (hm.get("strike") < m.get("strike"))):
                    why = "the hedge must be further out of the money than the short"
                elif hm.get("lot") != m.get("lot"):
                    why = "the hedge has a different lot size"
            if why:
                return self._done(req_id, {"ok": False, "msg": why})
        hres = self._enter(hedge, "BUY", lots, {}, None, "hedge", tag=tag, role="hedge", hedge_for=inst)
        if not hres["ok"]:
            return self._done(req_id, {"ok": False, "msg": "hedge not bought: " + str(hres["msg"])})
        with self.lock:
            hp = self.pos.get(hedge)
            use = min(lots, (hp.qty // max(1, int(hp.meta.get("lot") or 1))) if hp else 0)
        if use < 1:
            self.exit(hedge, 100, "unwind hedge", urgent=True)
            return self._done(req_id, {"ok": False, "msg": "the hedge filled less than a lot - sold back, no short opened"})
        res = self._enter(inst, "SELL", use, exits, None, src, tag=tag)
        if not res["ok"]:
            un = self.exit(hedge, 100, "unwind hedge", urgent=True)
            return self._done(req_id, {"ok": False, "msg": f"short not opened ({res['msg']}); wing sold back" + ("" if un.get("ok") else " - SELL THE WING MANUALLY")})
        with self.lock:
            hp, sp = self.pos.get(hedge), self.pos.get(inst)
            if hp:
                hp.hedge_armed = True
            if sp:
                sp.hedge = hedge
        return self._done(req_id, {"ok": True, "msg": res["msg"] + f"  (hedged with {hedge.split(':')[-1]})", "qty": res["qty"], "price": res["price"], "inst": inst})

    def _enter_structure(self, legs, lots, exits, g_sl, g_tp, req_id, src, tag, kind):
        """Several legs as one unit (straddle, strangle ...): bought legs first, then sold;
        a failure part-way sells back what already filled. Stop/target are combined, in rupees."""
        self._roll_day()
        now = self.clock()
        if not isinstance(legs, list) or not 2 <= len(legs) <= 4:
            return self._done(req_id, {"ok": False, "msg": "a structure needs 2-4 legs"})
        norm = []
        for lg in legs:
            inst = str((lg or {}).get("inst") or "")
            sd = "SELL" if str((lg or {}).get("side") or "BUY").upper().startswith("S") else "BUY"
            norm.append((inst, sd))
        if len({i for i, _ in norm}) != len(norm):
            return self._done(req_id, {"ok": False, "msg": "the same contract appears twice"})
        n_short = sum(1 for _, s in norm if s == "SELL")
        with self.lock:
            if req_id and req_id in self.seen and now - self.seen[req_id][0] < 15:
                return dict(self.seen[req_id][1], dup=True)
            metas = [self.feed.meta(i) or {} for i, _ in norm]
            if len({(m.get("under"), m.get("expiry")) for m in metas}) != 1 or not all(m.get("typ") for m in metas):
                return self._done(req_id, {"ok": False, "msg": "all legs must be options on the same index and expiry"})
            for inst, sd in norm:
                why = self.check_entry(inst, lots, side=sd, exits=exits, extra_lots=(lots * (len(norm) - 1)) if isinstance(lots, int) else 0,
                                       extra_short=(lots * (n_short - 1)) if isinstance(lots, int) and n_short else 0, stop_ok=True)
                if why:
                    return self._done(req_id, {"ok": False, "msg": f"{inst.split(':')[-1]}: {why}"})
            gid = self._id("G")
        done = []
        for inst, sd in sorted(norm, key=lambda x: x[1] != "BUY"):          # buys first (margin), then sells
            r = self._enter(inst, sd, lots, exits, None, src, tag=tag, grp=gid, stop_ok=True)
            if not r["ok"]:
                for di in sorted(done, key=lambda i: self.pos[i].side if i in self.pos else 0):
                    self.exit(di, 100, "unwind structure", urgent=True)
                return self._done(req_id, {"ok": False, "msg": f"{inst.split(':')[-1]}: {r['msg']} - legs already filled were closed"})
            done.append(inst)
        with self.lock:
            self.groups[gid] = {"id": gid, "legs": list(done), "sl": g_sl or None, "tp": g_tp or None, "real": 0.0, "done": False,
                                "opened": now, "tag": tag, "kind": kind}
            self.log("group", id=gid, legs=done, sl=g_sl, tp=g_tp, kind=kind)
        return self._done(req_id, {"ok": True, "msg": f"{kind or 'structure'} open: {len(done)} legs", "group": gid, "legs": done})

    def _done(self, req_id, out):
        if req_id:
            self.seen[req_id] = (self.clock(), out)
            if len(self.seen) > 200:
                for k in sorted(self.seen, key=lambda k: self.seen[k][0])[:100]:
                    self.seen.pop(k, None)
        if not out.get("ok"):
            self.log("reject", msg=out.get("msg"))
        return out

    # ── conditional entries ───────────────────────────────────────
    def add_pending(self, spec, trig, ttl_s=1800):
        """Queue an entry that fires when a premium or the index crosses a level.
        The order itself is placed through the normal `place` path at that moment,
        so every guard runs then. Held in memory only: a restart drops them."""
        kind, op = str((trig or {}).get("kind") or ""), str((trig or {}).get("op") or "")
        try:
            level = float((trig or {}).get("level"))
        except (TypeError, ValueError):
            level = 0.0
        if kind not in ("px", "spot") or op not in ("ge", "le") or not level > 0:
            return {"ok": False, "msg": "trigger needs a kind (premium or spot), a direction and a level"}
        if kind == "px" and spec.get("legs"):
            return {"ok": False, "msg": "a structure can wait on the index level only, not a premium"}
        side = "SELL" if str(spec.get("side") or "BUY").upper().startswith("S") else "BUY"
        with self.lock:
            if len(self.pending) >= self.MAX_PENDING:
                return {"ok": False, "msg": f"{self.MAX_PENDING} conditional entries are already waiting"}
            if not spec.get("legs"):
                inst = str(spec.get("inst") or "")
                why = self.check_entry(inst, spec.get("lots"), side=side, exits=spec.get("exits") or {}, static_only=True,
                                       extra_lots=spec.get("lots") if (side == "SELL" and spec.get("hedge") and isinstance(spec.get("lots"), int)) else 0)
                if why:
                    return {"ok": False, "msg": why}
                meta = self.feed.meta(inst) or {}
            else:
                meta = self.feed.meta(str(spec["legs"][0].get("inst") or "")) or {}
            now = self.clock()
            pd = {"id": self._id("C"), "spec": dict(spec, side=side), "trig": {"kind": kind, "op": op, "level": level},
                  "under": meta.get("under"), "created": round(now, 1), "expires": now + max(30.0, min(float(ttl_s), 6 * 3600.0))}
            self.pending.append(pd)
            self.log("pending", id=pd["id"], spec=pd["spec"], trig=pd["trig"])
            return {"ok": True, "msg": "waiting", "id": pd["id"]}

    def cancel_pending(self, pid=None):
        with self.lock:
            n0 = len(self.pending)
            self.pending = [p for p in self.pending if pid and p["id"] != pid]
            self.log("pending_cancel", id=pid, n=n0 - len(self.pending))
            return {"ok": True, "msg": f"{n0 - len(self.pending)} cancelled"}

    def _pending_step(self, now):
        with self.lock:
            items = list(self.pending)
        for pd in items:
            if now >= pd["expires"]:
                self._pend_finish(pd, {"ok": False, "msg": "expired before the level was reached"})
                continue
            tr, spec = pd["trig"], pd["spec"]
            if tr["kind"] == "px":
                q = self._quote(spec["inst"])
                if not q or not q.get("fresh"):
                    continue
                px = q.get("ask") if spec["side"] == "BUY" else q.get("bid")
            else:
                px = self.feed.spot(pd["under"]) or 0
            if not px:
                continue
            if (px >= tr["level"]) if tr["op"] == "ge" else (px <= tr["level"]):
                with self.lock:
                    if pd not in self.pending:
                        continue
                    self.pending.remove(pd)
                job = lambda pd=pd: self._pend_finish(pd, self.place(pd["spec"], src="pending"), px=px)      # noqa: E731
                if self.async_exits:
                    threading.Thread(target=job, daemon=True).start()
                else:
                    job()

    def _pend_finish(self, pd, res, px=None):
        with self.lock:
            if pd in self.pending:
                self.pending.remove(pd)
            self.pend_log.append({"id": pd["id"], "t": round(self.clock(), 1), "ok": bool(res.get("ok")), "msg": res.get("msg"),
                                  "inst": (pd["spec"].get("inst") or "").split(":")[-1], "side": pd["spec"].get("side"), "px": px})
            del self.pend_log[:-12]
            self.log("pending_done", id=pd["id"], ok=bool(res.get("ok")), msg=res.get("msg"))

    # ── exits ─────────────────────────────────────────────────────
    def _arm_exits(self, p, e):
        """Exits given in POINTS from the average (sl_pts, tp_pts, tp1_pts, trail_pts, be_pts)
        or as absolute premiums (sl, tp, tp1); underlying levels (u_stop, u_target).
        For a short the stop is ABOVE the average and the targets BELOW it."""
        a, sg = p.avg, p.side

        def lvl(abs_k, pts_k, sign):
            if e.get(abs_k):
                return _r(float(e[abs_k]))
            if e.get(pts_k):
                return max(TICK, _r(a + sign * sg * float(e[pts_k])))
            return None
        for name, ak, pk, sn in (("sl", "sl", "sl_pts", -1), ("tp", "tp", "tp_pts", 1), ("tp1", "tp1", "tp1_pts", 1)):
            v = lvl(ak, pk, sn)
            if v is not None or any(k in e for k in (ak, pk)):
                setattr(p, name, v)
        if "trail_pts" in e:
            p.trail = float(e["trail_pts"]) or None
        if "be_pts" in e:
            p.be_at = _r(a + sg * float(e["be_pts"])) if e.get("be_pts") else None
        if "tp1_pct" in e:
            p.tp1_pct = max(1, min(100, int(e["tp1_pct"])))
        if "u_stop" in e or "u_target" in e:
            p.u_stop = float(e["u_stop"]) if e.get("u_stop") else None
            p.u_target = float(e["u_target"]) if e.get("u_target") else None
            p.u_dir = sg * (1 if p.meta.get("typ") == "CE" else -1)
        if "time_stop_s" in e:
            p.time_stop = float(e["time_stop_s"]) if e.get("time_stop_s") else None
        if p.sl is not None:
            q = self.feed.quote(p.inst) or {}
            if sg > 0:
                # a stop at or above the current bid would fire at once; keep it two
                # ticks under it (so break-even works once the price is above entry)
                ceil = (q.get("bid") or a) - TICK * 2
                if p.sl > ceil:
                    p.sl = _r(ceil)
            else:
                # a short's stop at or below the current ask would fire at once
                floor = (q.get("ask") or a) + TICK * 2
                if p.sl < floor:
                    p.sl = _r(floor, up=True)
        p.managed = True

    def modify(self, inst, e):
        with self.lock:
            p = self.pos.get(inst)
            if not p:
                return {"ok": False, "msg": "no such position"}
            if p.role == "hedge":
                return {"ok": False, "msg": "a hedge wing has no exits - it is sold once its short is covered"}
            if e.get("clear"):
                if p.side < 0:
                    return {"ok": False, "msg": "a short must keep its stop"}
                p.sl = p.tp = p.tp1 = p.be_at = p.trail = p.u_stop = p.u_target = p.time_stop = None
            self._arm_exits(p, e)
            if p.side < 0 and p.sl is None:
                return {"ok": False, "msg": "a short must keep its stop"}
            self.log("modify", inst=inst, exits=p.exits())
            return {"ok": True, "msg": "exits updated", "exits": p.exits()}

    def modify_group(self, gid, sl=None, tp=None):
        with self.lock:
            g = self.groups.get(gid)
            if not g:
                return {"ok": False, "msg": "no such structure"}
            if sl is not None:
                if sl <= 0 and any(self.pos[i].side < 0 and not self.pos[i].sl for i in g["legs"] if i in self.pos):
                    return {"ok": False, "msg": "a structure with a short leg must keep a combined stop"}
                g["sl"] = sl or None
            if tp is not None:
                g["tp"] = tp or None
            self.log("group_modify", id=gid, sl=g["sl"], tp=g["tp"])
            return {"ok": True, "msg": "structure exits updated"}

    def exit(self, inst, pct=100, why="manual", urgent=False, wait=True):
        """Close pct% of the position (whole lots; the last lot closes it): a long sells, a short buys back."""
        with self.lock:
            p = self.pos.get(inst)
            if not p:
                return {"ok": False, "msg": "no such position"}
            if p.exiting:
                return {"ok": False, "msg": "an exit is already in progress"}
            if p.role == "hedge" and p.hedge_for in self.pos:
                return {"ok": False, "msg": "close the short first - this wing is what keeps its margin and loss down"}
            lot = max(1, int(p.meta.get("lot") or 1))
            lots = max(1, p.qty // lot)
            qty = p.qty if pct >= 100 else min(p.qty, max(1, round(lots * pct / 100.0)) * lot)
            p.exiting = True
        job = lambda: self._exit_now(inst, qty, why, urgent)          # noqa: E731
        if self.async_exits and not wait:
            threading.Thread(target=job, daemon=True).start()
            return {"ok": True, "msg": "exit sent"}
        return job()

    def _exit_now(self, inst, qty, why, urgent):
        res = None
        with self.lock:
            p0 = self.pos.get(inst)
            side_x = "SELL" if (p0 is None or p0.side > 0) else "BUY"
        ref_x = 0.0
        try:
            for attempt in range(3):
                q = self._quote(inst) or {}
                ref_x = (q.get("bid") if side_x == "SELL" else q.get("ask")) or q.get("ltp") or 0
                if not ref_x:
                    res = {"ok": False, "qty": 0, "price": 0, "status": "NOQUOTE", "msg": "no quote to exit against", "order_id": None}
                    break
                res = self._send(inst, side_x, qty, q, urgent=urgent or attempt > 0, widen=2.0 ** attempt)
                with self.lock:
                    self._note_order(inst, side_x, qty, res, why, act="close")
                if res["ok"] or res.get("unknown"):
                    break
                if attempt < 2:
                    self.sleep(0.2)
        finally:
            with self.lock:
                p = self.pos.get(inst)
                if p:
                    p.exiting = False
        with self.lock:
            p = self.pos.get(inst)
            if not res or not res["ok"]:
                if p:
                    p.exit_note = f"exit failed: {res['msg'] if res else 'no attempt'}"
                    p.retry_at = self.clock() + 3.0
                self.log("exit_failed", inst=inst, why=why, msg=(res or {}).get("msg"))
                return {"ok": False, "msg": (res or {}).get("msg") or "exit failed"}
            if not p:
                return {"ok": True, "msg": "closed"}
            sold = min(res["qty"], p.qty)
            sg = p.side
            gross = (res["price"] - p.avg) * sold * sg
            p.qty -= sold
            p.cost = p.avg * p.qty
            p.fills.append((self.clock(), side_x, sold, res["price"]))
            long_val, exit_val = p.avg * sold, res["price"] * sold
            buy_val, sell_val = (long_val, exit_val) if sg > 0 else (exit_val, long_val)
            fee = est_charges(buy_val, sell_val, 2)
            self.gross += gross
            self.charges += fee
            net = gross - fee
            xslip = ((ref_x - res["price"]) if side_x == "SELL" else (res["price"] - ref_x)) * sold
            share = sold / float(sold + p.qty) if (sold + p.qty) else 1.0
            tr = {"t": round(self.clock(), 1), "inst": inst, "typ": p.meta.get("typ"), "strike": p.meta.get("strike"),
                  "under": p.meta.get("under"), "side": "long" if sg > 0 else "short", "role": p.role, "qty": sold,
                  "avg": round(p.avg, 2), "exit": round(res["price"], 2), "gross": round(gross, 2), "fee": round(fee, 2),
                  "net": round(net, 2), "why": why, "held_s": round(self.clock() - p.opened), "src": p.src,
                  "tag": p.tag or "", "grp": p.grp or "", "opened": round(p.opened),
                  "risk": round(p.risk_ps * sold, 2), "mfe": round(sg * (p.best - p.avg) * sold, 2),
                  "mae": round(sg * (p.avg - p.worst) * sold, 2), "slip": round(p.slip * share + xslip, 2)}
            p.slip -= p.slip * share
            self.trades.append(tr)
            del self.trades[:-400]
            if p.grp and p.grp in self.groups:
                self.groups[p.grp]["real"] += gross
            if net < 0:
                self.streak += 1
                if self.cfg.loss_streak and self.streak >= self.cfg.loss_streak:
                    self.cool_until = self.clock() + self.cfg.cooldown_s
                    self.log("cooldown", streak=self.streak)
                    self.streak = 0
            else:
                self.streak = 0
            if p.qty <= 0:
                self.pos.pop(inst, None)
            self.log("fill", inst=inst, side=side_x, qty=sold, price=res["price"], gross=round(gross, 2), net=round(net, 2), why=why)
            verb = "sold" if side_x == "SELL" else "covered"
            return {"ok": True, "msg": f"{verb} {sold} @ {res['price']:.2f}  ({net:+,.0f} net)", "net": round(net, 2)}

    def flatten(self, why="flatten", wait=True):
        out = []
        with self.lock:
            order = sorted(self.pos, key=lambda i: self.pos[i].role == "hedge")          # wings last
        for inst in order:
            p = self.pos.get(inst)
            if p and p.role == "hedge" and p.hedge_for in self.pos and not wait:
                continue                                                    # step() sells it once the short is covered
            out.append(self.exit(inst, 100, why, urgent=True, wait=wait))
        with self.lock:
            self.pending = []
        return {"ok": all(o.get("ok") for o in out) if out else True, "msg": f"{len(out)} position(s) closing", "results": out}

    def set_kill(self, on, flatten=False, why="manual"):
        with self.lock:
            self.kill = bool(on)
            self.kill_why = why if on else ""
            if on:
                self.armed = False
                self.pending = []
            self.log("kill", on=bool(on), flatten=bool(flatten), why=why)
        if on and flatten:
            self.flatten("kill switch", wait=False)
        return {"ok": True, "kill": self.kill}

    def set_armed(self, on):
        with self.lock:
            if on and (self.kill or self.halt):
                return {"ok": False, "msg": "cannot arm: " + (f"kill switch ({self.kill_why or 'manual'})" if self.kill else self.halt)}
            if on and not self.live():
                return {"ok": False, "msg": "server is in PAPER mode (set NSE_TRADE_MODE=live in env to allow live orders)"}
            if on and not self.broker:
                return {"ok": False, "msg": "no broker connection (DATA_SOURCE=zerodha and a valid token needed)"}
            self.armed = bool(on)
            self.log("arm", on=bool(on))
            return {"ok": True, "armed": self.armed}

    # ── the managed-exit loop ─────────────────────────────────────
    def step(self):
        self._roll_day()
        now = self.clock()
        mins, wd = ist_clock(now)
        c = self.cfg
        with self.lock:
            plist = sorted(self.pos.values(), key=lambda p: p.role == "hedge")           # wings after the shorts they protect
        self._pending_step(now)
        spot_cache = {}
        for p in plist:
            if p.exiting or now < p.retry_at:
                continue
            if p.role == "hedge":
                # a wing has no exits of its own: it goes only once its short has been covered
                if p.hedge_armed and p.hedge_for not in self.pos:
                    self.exit(p.inst, 100, "hedge released", wait=False)
                continue
            q = self._quote(p.inst)
            p.stale = not (q and q.get("fresh"))
            lot = max(1, int(p.meta.get("lot") or 1))
            sg = p.side
            # end of day first: it needs no price
            if wd < 5 and mins >= c.flatten_at:
                self.exit(p.inst, 100, "end-of-day square-off", urgent=True, wait=False)
                continue
            if p.time_stop and now - p.opened >= p.time_stop:
                self.exit(p.inst, 100, "time stop", wait=False)
                continue
            if p.stale or not q:
                continue
            px = (q.get("bid") if sg > 0 else q.get("ask")) or q.get("ltp")
            if not px:
                continue
            p.best = max(p.best, px) if sg > 0 else min(p.best, px)
            p.worst = min(p.worst, px) if sg > 0 else max(p.worst, px)
            if p.be_at and (p.best >= p.be_at if sg > 0 else p.best <= p.be_at) and (p.sl is None or (p.sl < p.avg if sg > 0 else p.sl > p.avg)):
                p.sl = _r(p.avg + sg * TICK)
                self.log("breakeven", inst=p.inst, sl=p.sl)
            sl = p.eff_sl()
            if sl is not None and ((px - sl) * sg <= 0):
                self.exit(p.inst, 100, f"stop {sl:.2f}", urgent=True, wait=False)
                continue
            if p.tp and (px - p.tp) * sg >= 0:
                self.exit(p.inst, 100, f"target {p.tp:.2f}", wait=False)
                continue
            if p.tp1 and not p.tp1_done and (px - p.tp1) * sg >= 0:
                p.tp1_done = True
                if p.qty > lot:
                    self.exit(p.inst, p.tp1_pct, f"partial target {p.tp1:.2f}", wait=False)
                    if p.sl is None or (p.sl < p.avg if sg > 0 else p.sl > p.avg):
                        p.sl = _r(p.avg + sg * TICK)         # the rest rides at no risk
                    continue
            if p.u_stop or p.u_target:
                u = spot_cache.get(p.meta.get("under"))
                if u is None:
                    u = spot_cache[p.meta.get("under")] = self.feed.spot(p.meta.get("under")) or 0
                if u:
                    d = p.u_dir or sg
                    if p.u_stop and (u - p.u_stop) * d <= 0:
                        self.exit(p.inst, 100, f"underlying stop {p.u_stop:g}", urgent=True, wait=False)
                        continue
                    if p.u_target and (u - p.u_target) * d >= 0:
                        self.exit(p.inst, 100, f"underlying target {p.u_target:g}", wait=False)
                        continue
        self._group_step(now)
        with self.lock:
            self.day_peak = max(self.day_peak, self.net_day()) if (self.pos or self.trades) else self.day_peak
        # the daily loss limit: no more entries, and (by default) out of everything
        if c.day_loss and self.pos and not self.kill and self.net_day() <= -c.day_loss:
            self.set_kill(True, flatten=c.loss_flatten, why=f"daily loss limit Rs {c.day_loss:,.0f}")
        # live: does the book still match the broker?
        if self.live() and self.broker and (self.pos or self.armed) and now - self.last_recon >= c.reconcile_s and not self.inflight \
                and not any(p.exiting for p in self.pos.values()):
            self.last_recon = now
            self.reconcile(adopt=False)

    def _group_step(self, now):
        with self.lock:
            groups = list(self.groups.values())
        for g in groups:
            with self.lock:
                legs = [self.pos[i] for i in g["legs"] if i in self.pos]
                if not legs:
                    self.groups.pop(g["id"], None)
                    continue
            if g["done"]:
                for p in legs:                                              # a leg whose exit failed: keep trying
                    if not p.exiting and now >= p.retry_at:
                        self.exit(p.inst, 100, "structure exit (retry)", urgent=True, wait=False)
                continue
            qs = [self._quote(p.inst) for p in legs]
            if any(not q or not q.get("fresh") for q in qs):
                continue
            pnl = g["real"] + sum(self._pnl(p, q) for p, q in zip(legs, qs))
            g["pnl"] = round(pnl, 2)
            why = None
            if g.get("sl") and pnl <= -g["sl"]:
                why = f"structure stop -Rs {g['sl']:,.0f}"
            elif g.get("tp") and pnl >= g["tp"]:
                why = f"structure target Rs {g['tp']:,.0f}"
            if why:
                g["done"] = True
                self.log("group_exit", id=g["id"], why=why, pnl=g["pnl"])
                for p in sorted(legs, key=lambda p: p.side):                   # shorts first
                    self.exit(p.inst, 100, why, urgent=True, wait=False)

    # ── live book vs broker ───────────────────────────────────────
    def reconcile(self, adopt=False):
        if not self.live() or not self.broker:
            return {"ok": True, "msg": "paper mode - nothing to reconcile"}
        try:
            rows = self.broker.positions() or []
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "msg": f"could not read broker positions: {e}"}
        theirs = {}
        for r in rows:
            q = int(r.get("quantity") or 0)
            if q:
                theirs[r["tradingsymbol"]] = q                              # negative = short
        mine = {(p.meta or {}).get("ts") or p.inst.split(":")[-1]: p.qty * p.side for p in self.pos.values()}
        diff = {k: (mine.get(k, 0), theirs.get(k, 0)) for k in set(mine) | set(theirs) if mine.get(k, 0) != theirs.get(k, 0)}
        with self.lock:
            if not diff:
                if self.halt.startswith("book differs") or self.halt.startswith("order "):
                    self.halt = ""
                return {"ok": True, "msg": "book matches the broker"}
            if adopt:
                keep = {}
                for k, q in theirs.items():
                    inst = "NFO:" + k
                    old = self.pos.get(inst)
                    meta = self.feed.meta(inst)
                    if q != 0 and meta:
                        sg = 1 if q > 0 else -1
                        if old and old.side != sg:
                            old = None
                        np_ = old or Pos(inst, meta, abs(q), float(next((r.get("average_price") for r in rows if r["tradingsymbol"] == k), 0) or 0),
                                         self.clock(), "adopted", side=sg)
                        np_.qty = abs(q)
                        np_.managed = bool(old)
                        keep[inst] = np_
                self.pos = keep
                self.halt = ""
                self.log("adopt", positions=list(keep))
                return {"ok": True, "msg": f"adopted {len(keep)} broker position(s); set their exits"}
            self.halt = "book differs from the broker " + ", ".join(f"{k}: engine {a} / broker {b}" for k, (a, b) in sorted(diff.items()))
            self.log("halt", why=self.halt)
        return {"ok": False, "msg": self.halt}

    # ── what the page shows ───────────────────────────────────────
    def state(self):
        self._roll_day()
        now = self.clock()
        mins, wd = ist_clock(now)
        with self.lock:
            rows = []
            for p in self.pos.values():
                q = self.feed.quote(p.inst) or {}
                bid, ltp, ask = q.get("bid") or 0, q.get("ltp") or 0, q.get("ask") or 0
                mark = self._mark(p, q)
                rows.append({
                    "inst": p.inst, "ts": p.meta.get("ts"), "under": p.meta.get("under"), "typ": p.meta.get("typ"),
                    "strike": p.meta.get("strike"), "expiry": p.meta.get("expiry"), "lot": p.meta.get("lot"),
                    "side": "long" if p.side > 0 else "short", "role": p.role, "hedge_for": p.hedge_for, "grp": p.grp, "tag": p.tag,
                    "qty": p.qty, "lots": p.qty // max(1, int(p.meta.get("lot") or 1)), "avg": round(p.avg, 2),
                    "ltp": ltp, "bid": bid, "ask": ask, "mark": round(mark, 2), "pnl": round((mark - p.avg) * p.qty * p.side, 2),
                    "peak": round(p.best, 2), "best": round(p.best, 2), "worst": round(p.worst, 2),
                    "mfe": round(p.side * (p.best - p.avg) * p.qty, 2), "mae": round(p.side * (p.avg - p.worst) * p.qty, 2),
                    "risk": round(p.risk_ps * p.qty, 2), "slip": round(p.slip, 2),
                    "sl": p.sl, "eff_sl": p.eff_sl(), "exits": p.exits(), "stale": p.stale,
                    "exiting": p.exiting, "note": p.exit_note, "src": p.src, "managed": p.managed, "opened": round(p.opened),
                    "held_s": round(now - p.opened), "age": round(now - float(q.get("ts") or 0), 1) if q else None,
                })
            unreal = sum(r["pnl"] for r in rows)
            wins = sum(1 for t in self.trades if t["net"] > 0)
            net = self.gross - self.charges + unreal
            groups = []
            for g in self.groups.values():
                legs = [r for r in rows if r["inst"] in g["legs"]]
                groups.append({"id": g["id"], "kind": g.get("kind"), "legs": g["legs"], "sl": g.get("sl"), "tp": g.get("tp"),
                               "pnl": round(g["real"] + sum(r["pnl"] for r in legs), 2), "done": g["done"], "tag": g.get("tag")})
            pend = [{"id": d["id"], "spec": d["spec"], "trig": d["trig"], "left_s": max(0, int(d["expires"] - now))} for d in self.pending]
            return {
                "enabled": True, "mode": self.cfg.mode, "armed": self.armed, "kill": self.kill, "kill_why": self.kill_why,
                "halt": self.halt, "now": round(now, 2), "day": self.day, "blocked": self._blocked(),
                "entries_open": wd < 5 and self.cfg.entry_from <= mins < self.cfg.entry_to,
                "cooldown_s": max(0, int(self.cool_until - now)),
                "limits": self.cfg.public(),
                "pnl": {"gross": round(self.gross, 2), "fees": round(self.charges, 2), "realized": round(self.gross - self.charges, 2),
                        "unrealized": round(unreal, 2), "net": round(net, 2), "peak": round(max(self.day_peak, net), 2),
                        "trades": len(self.trades), "wins": wins, "losses": len(self.trades) - wins, "streak": self.streak},
                "positions": rows, "orders": self.orders[-60:], "trades": list(self.trades),
                "groups": groups, "pending": pend, "pend_log": list(self.pend_log),
                "open_lots": sum(r["lots"] for r in rows),
            }
