"""Auto Trade for the Scalping tab: follows the Trend Target Ribbon, server side.

Runs in the server process, so closing or reloading the browser tab changes nothing.
Orders go through the same Engine as manual ones (paper by default, ARM needed for live).

Rules
  * Enabling never trades on the current trend: the first signal is the next FLIP.
  * Signals are read from CLOSED candles only (a flip never appears mid-bar and vanishes).
  * BUY mode:  bull -> buy CE, bear -> buy PE.   SELL mode: bear -> sell CE, bull -> sell PE.
  * A flip closes the auto position (CE bought is sold, CE sold is bought back) and opens the other.
  * Entries only from `start` to before `end`; at `end` the auto position is closed and
    nothing more is opened that day.
  * The contract is chosen when the entry is sent (ATM + offset, snapped to a listed strike)
    and the exit always uses the contract that was actually bought.
Positions are recognised by src="auto" on the engine's book, so nothing is lost if this
object is rebuilt.
"""
import math
import threading
import time

IST = 19800
SRC = "auto"

# Trend Target Ribbon [BOSWaves] defaults, as in the dashboard's TTR object
ALMA_LEN, ALMA_OFFSET, ALMA_SIGMA = 34, 0.85, 6.0
DEV_LEN, DEV_MULT = 34, 0.65
SLOPE_LEN, SLOPE_MIN = 3, 0.08
ATR_LEN = 14


def ttr_trend(bars):
    """Per-bar trend (+1 bull, -1 bear, 0 none yet) of the Trend Target Ribbon.
    Same maths as the dashboard's _ttrCompute (checked bar for bar in test_nse.py)."""
    n = len(bars)
    c = [b["c"] for b in bars]
    h = [b["h"] for b in bars]
    lo = [b["l"] for b in bars]
    m, sg = ALMA_OFFSET * (ALMA_LEN - 1), ALMA_LEN / ALMA_SIGMA
    wts = [math.exp(-((i - m) ** 2) / (2 * sg * sg)) for i in range(ALMA_LEN)]
    norm = sum(wts)
    alma = [None] * n
    dev = [None] * n
    atr = [None] * n
    for t in range(ALMA_LEN - 1, n):
        alma[t] = sum(c[t - (ALMA_LEN - 1 - i)] * wts[i] for i in range(ALMA_LEN)) / norm
    for t in range(DEV_LEN - 1, n):
        win = c[t - DEV_LEN + 1:t + 1]
        mu = sum(win) / DEV_LEN
        dev[t] = math.sqrt(sum((x - mu) ** 2 for x in win) / DEV_LEN)
    tr = [(h[0] - lo[0]) if i == 0 else max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(n)]
    for t in range(n):
        if t == ATR_LEN - 1:
            atr[t] = sum(tr[:ATR_LEN]) / ATR_LEN
        elif t >= ATR_LEN:
            atr[t] = (atr[t - 1] * (ATR_LEN - 1) + tr[t]) / ATR_LEN
    trend, out = 0, []
    for t in range(n):
        a, d, at = alma[t], dev[t], atr[t]
        a0 = alma[t - SLOPE_LEN] if t >= SLOPE_LEN else None
        slope = ((a - a0) / at if at > 0 else 0.0) if (a is not None and a0 is not None and at is not None) else None
        up = a + d * DEV_MULT if (a is not None and d is not None) else None
        dn = a - d * DEV_MULT if (a is not None and d is not None) else None
        bull = slope is not None and up is not None and slope > SLOPE_MIN and c[t] > up
        bear = slope is not None and dn is not None and slope < -SLOPE_MIN and c[t] < dn
        if trend != 1 and bull:
            trend = 1
        elif trend != -1 and bear:
            trend = -1
        out.append(trend)
    return out


def _ema(vals, period):
    """EMA seeded with the SMA of the first `period` closes, as the dashboard's _emaSeries does."""
    n = len(vals)
    out = [None] * n
    if n < 2:
        return out
    seed = min(period, n)
    prev = sum(vals[:seed]) / seed
    out[seed - 1] = prev
    k = 2.0 / (period + 1)
    for i in range(seed, n):
        prev = vals[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def ema_trend(bars, fast, slow):
    """Per-bar side of an EMA crossover (+1 fast above slow, -1 below, 0 until both exist; a tie keeps the side).
    Same rule as the dashboard's EMA crossover indicator."""
    pf, ps = min(fast, slow), max(fast, slow)
    c = [b["c"] for b in bars]
    f, s = _ema(c, pf), _ema(c, ps)
    side, out = 0, []
    for j in range(len(c)):
        if j >= ps - 1 and f[j] is not None and s[j] is not None:
            d = f[j] - s[j]
            side = 1 if d > 0 else -1 if d < 0 else side
        out.append(side)
    return out


def closed_bars(bars, tf_min, now):
    """Bars whose whole period has passed (the feed's last bar is usually still forming)."""
    return [b for b in bars if b["t"] + tf_min * 60 <= now]


def _hm(v, default):
    try:
        h, m = str(v).strip().split(":")
        return int(h) * 60 + int(m)
    except Exception:
        return default


def _fmt(mins):
    return f"{mins // 60:02d}:{mins % 60:02d}"


class AutoTrader:
    """bars_fn(sym, tf_min) -> bars [{t,o,h,l,c}] (t = epoch seconds, last may be forming)
    pick_fn(sym, typ, off) -> (inst, strike, atm) or raises ValueError
    strikes_fn(sym) -> {"atm": x, "step": s, "spot": y} or None"""

    SYMS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
    RETRY_S = 3.0           # a refused entry is retried this often ...
    RETRY_FOR_S = 60.0      # ... for this long, while the trend it was for still holds

    def __init__(self, engine, bars_fn, pick_fn, strikes_fn=None, clock=time.time):
        self.eng, self.bars_fn, self.pick_fn, self.strikes_fn, self.clock = engine, bars_fn, pick_fn, strikes_fn, clock
        self.lock = threading.RLock()
        self.act = threading.RLock()        # one decision at a time: the loop, START and EXIT never overlap
        self.cfg = {"sym": "NIFTY", "tf": 1, "mode": "BUY", "ce_off": -300, "pe_off": 300,
                    "start": 9 * 60 + 20, "end": 15 * 60 + 15, "lots": 2, "max_trades": 5, "max_loss": 5000,
                    "strategy": "RIBBON", "ema_fast": 5, "ema_slow": 15}
        self.enabled = False
        self._loss_noted = False
        self.entries = 0            # entries sent today (an entry and its exit are ONE trade)
        self.day = self._day()
        self.trend = 0              # trend of the last closed bar
        self.base = None            # trend when enabled / last acted on; a flip is a change from it
        self.bar_t = 0              # start time of the last closed bar read
        self.next_fetch = 0.0
        self.want = None            # {"trend", "bar", "since", "next"} an entry still to send
        self.done_today = False     # end time passed: no more entries today
        self.events = []
        self.msg = ""
        self._stop = threading.Event()
        self._thread = None

    # ── plumbing ──
    def _day(self):
        return time.strftime("%Y-%m-%d", time.gmtime(self.clock() + IST))

    def _clk(self):
        g = time.gmtime(self.clock() + IST)
        return g.tm_hour * 60 + g.tm_min, g.tm_wday

    def note(self, text):
        with self.lock:
            self.events.append({"t": round(self.clock()), "msg": text})
            del self.events[:-40]
            self.msg = text
        try:
            self.eng.log("auto", msg=text)
        except Exception:
            pass

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="auto-trade")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(1.0):
            try:
                self.step()
            except Exception as e:  # noqa: BLE001
                print(f"[auto] step failed: {type(e).__name__}: {e}")

    def holding(self):
        """The auto positions on the engine's book: [(inst, Pos)]."""
        with self.eng.lock:
            return [(i, p) for i, p in self.eng.pos.items() if p.src == SRC and p.role == "main"]

    # ── settings ──
    def configure(self, body):
        """Apply the tab's settings. -> {ok, msg}. Safe to call repeatedly with the same body."""
        act = str(body.get("action") or "")
        if act == "exit":
            return self.exit_now()
        if act == "start":
            body = dict(body, enabled=True)
        c = dict(self.cfg)
        try:
            sym = str(body.get("sym") or c["sym"]).upper()
            if sym not in self.SYMS:
                return {"ok": False, "msg": f"unknown index {sym}"}
            tf = int(body.get("tf") or c["tf"])
            if tf not in (1, 2, 3, 5, 10, 15):
                return {"ok": False, "msg": "timeframe must be 1, 2, 3, 5, 10 or 15 minutes"}
            mode = str(body.get("mode") or c["mode"]).upper()
            if mode not in ("BUY", "SELL"):
                return {"ok": False, "msg": "mode must be BUY or SELL"}
            lots = body.get("lots", c["lots"])
            if not isinstance(lots, int) or isinstance(lots, bool) or lots < 1:
                return {"ok": False, "msg": "lots must be a whole number >= 1"}
            ce_off, pe_off = float(body.get("ce_off", c["ce_off"])), float(body.get("pe_off", c["pe_off"]))
            if abs(ce_off) > 5000 or abs(pe_off) > 5000:
                return {"ok": False, "msg": "strike offset out of range"}
            max_trades, max_loss = body.get("max_trades", c["max_trades"]), float(body.get("max_loss", c["max_loss"]))
            if not isinstance(max_trades, int) or isinstance(max_trades, bool) or max_trades < 1:
                return {"ok": False, "msg": "max trades must be a whole number >= 1"}
            if max_loss <= 0:
                return {"ok": False, "msg": "max loss must be more than 0"}
            strategy = str(body.get("strategy") or c["strategy"]).upper()
            if strategy not in ("RIBBON", "EMA"):
                return {"ok": False, "msg": "strategy must be RIBBON or EMA"}
            ema_fast, ema_slow = body.get("ema_fast", c["ema_fast"]), body.get("ema_slow", c["ema_slow"])
            if not all(isinstance(x, int) and not isinstance(x, bool) for x in (ema_fast, ema_slow)) or not (1 <= ema_fast < ema_slow <= 200):
                return {"ok": False, "msg": "EMA periods must be whole numbers with fast < slow (1-200)"}
            start, end = _hm(body.get("start", _fmt(c["start"])), -1), _hm(body.get("end", _fmt(c["end"])), -1)
            if not (9 * 60 + 15 <= start < end <= 15 * 60 + 15):
                return {"ok": False, "msg": "times must satisfy 09:15 <= start < end <= 15:15"}
        except (TypeError, ValueError) as e:
            return {"ok": False, "msg": f"bad setting: {e}"}
        enable = bool(body.get("enabled")) if "enabled" in body else self.enabled
        with self.lock:
            if self.holding() and (sym != c["sym"] or tf != c["tf"]):
                return {"ok": False, "msg": "an auto position is open - close it before changing the index or timeframe"}
            changed = (sym, tf, strategy, ema_fast, ema_slow) != (c["sym"], c["tf"], c["strategy"], c["ema_fast"], c["ema_slow"])
            self.cfg = {"sym": sym, "tf": tf, "mode": mode, "ce_off": ce_off, "pe_off": pe_off, "start": start, "end": end, "lots": lots,
                        "max_trades": max_trades, "max_loss": max_loss,
                        "strategy": strategy, "ema_fast": ema_fast, "ema_slow": ema_slow}
            if enable and not self.enabled:
                self.enabled, self.done_today, self.want = True, False, None
                self.base, self.next_fetch = None, 0.0          # re-baselined on the next read: wait for a flip
                self.note(f"Auto Trade ON - {mode} {sym} {tf}m, {lots} lot(s), {_fmt(start)}-{_fmt(end)}; {self._sname()} - waiting for the next flip")
            elif not enable and self.enabled:
                self.enabled, self.want = False, None
                n = len(self.holding())
                self.note("Auto Trade OFF" + (f" - {n} auto position(s) stay open; close them yourself or at the square-off" if n else ""))
            elif changed:
                self.base, self.next_fetch, self.want = None, 0.0, None
        if act == "start":
            return self.start_now()
        return {"ok": True, "msg": self.msg}

    def exit_now(self):
        with self.act:
            return self._exit_now()

    def _exit_now(self):
        """Force-exit every auto position now. Auto Trade stays on but will not re-enter until the next flip (or START)."""
        hold = self.holding()
        if not hold:
            return {"ok": False, "msg": "no auto position is open"}
        with self.lock:
            self.want = None
        out = [self.eng.exit(i, 100, "auto: forced exit", urgent=True, wait=True) for i, _p in hold]
        self.note("forced exit: " + "; ".join(f"{i.split(':')[-1]} -> {r.get('msg')}" for (i, _p), r in zip(hold, out)))
        return {"ok": all(r.get("ok") for r in out), "msg": self.msg}

    def start_now(self):
        with self.act:
            return self._start_now()

    def _bars(self, c, now):
        """Closed candles, or None when the feed failed (then the caller backs off instead of retrying every second)."""
        try:
            return closed_bars(self.bars_fn(c["sym"], c["tf"]) or [], c["tf"], now)
        except Exception as e:  # noqa: BLE001
            self.next_fetch = now + 15
            self.msg = f"candle feed failed: {type(e).__name__}: {str(e)[:80]}"
            return None

    def _start_now(self):
        """START: enter on the CURRENT signal, but only if no auto trade is open or already being entered."""
        mins, wd = self._clk()
        c = self.cfg
        if self.holding():
            return {"ok": False, "msg": "an auto trade is already open - START only enters when none is"}
        if self.want:
            return {"ok": False, "msg": "an auto entry is already in progress"}
        if wd >= 5 or not (c["start"] <= mins < c["end"]):
            return {"ok": False, "msg": f"outside the trading window {_fmt(c['start'])}-{_fmt(c['end'])}"}
        if self.loss_hit():
            return {"ok": False, "msg": "daily max loss reached - no more entries today"}
        if self.entries >= c["max_trades"]:
            return {"ok": False, "msg": f"daily max trades ({c['max_trades']}) reached"}
        now = self.clock()
        bars = self._bars(c, now)
        if bars is None:
            return {"ok": False, "msg": self.msg}
        if len(bars) < ALMA_LEN + 2:
            return {"ok": False, "msg": f"not enough candle history yet ({len(bars)}/{ALMA_LEN + 2} bars)"}
        trend = (ema_trend(bars, c["ema_fast"], c["ema_slow"]) if c["strategy"] == "EMA" else ttr_trend(bars))[-1]
        if trend == 0:
            return {"ok": False, "msg": f"{self._sname()} has no direction yet"}
        with self.lock:
            self.bar_t, self.trend, self.base = bars[-1]["t"], trend, trend
            self.done_today = False
            self.want = {"trend": trend, "bar": bars[-1]["t"], "since": now, "next": 0.0, "n": 0}
        self.note(f"START: {self._sname()} is {'bullish' if trend > 0 else 'bearish'} - entering now")
        self._try_want(now, mins)
        return {"ok": True, "msg": self.msg}

    # ── the loop ──
    def step(self):
        with self.act:
            self._step()

    def _step(self):
        d = self._day()
        if d != self.day:
            with self.lock:
                self.day = d
                self.done_today, self.want, self.base, self.bar_t, self.next_fetch = False, None, None, 0, 0.0
                self.entries = 0
                if self.enabled:
                    self.enabled = False
                    self.note("new trading day - Auto Trade switched off, tick it again to start")
        if not (self.enabled or self.holding()):
            return
        self._loss_guard()
        if not self.enabled:
            return
        now = self.clock()
        mins, wd = self._clk()
        if wd >= 5:
            return
        c = self.cfg
        # end time: close what we hold, stop for the day
        if mins >= c["end"]:
            if not self.done_today:
                self.done_today, self.want = True, None
                for inst, _p in self.holding():
                    r = self.eng.exit(inst, 100, "auto: end time", urgent=True, wait=True)
                    self.note(f"end time {_fmt(c['end'])}: closing {inst.split(':')[-1]} -> {r.get('msg')}")
                if not self.holding():
                    self.note(f"end time {_fmt(c['end'])}: Auto Trade done for the day")
            else:
                for inst, p in self.holding():                      # a failed exit is retried until it works
                    if not p.exiting:
                        self.eng.exit(inst, 100, "auto: end time", urgent=True, wait=True)
            return
        if mins < 9 * 60 + 15:
            return
        self._read(now)
        self._try_want(now, mins)

    def pnl(self):
        """Today's auto P&L: closed auto trades (net of charges) plus the open auto positions marked at the exit price."""
        with self.eng.lock:
            real = sum(t["net"] for t in self.eng.trades if t.get("src") == SRC)
            return real + sum(self.eng._pnl(p) for p in self.eng.pos.values() if p.src == SRC)

    def loss_hit(self):
        return self.pnl() <= -self.cfg["max_loss"]

    def _loss_guard(self):
        """Daily max loss: out of every auto position at once - no waiting for a flip or the square-off - and no entries
        until the loss is back inside the limit (or the limit is raised)."""
        if not self.loss_hit():
            self._loss_noted = False
            return
        hold = self.holding()
        if hold and not self._loss_noted:
            self._loss_noted = True
            self.note(f"daily max loss Rs {self.cfg['max_loss']:,.0f} reached (P&L Rs {self.pnl():,.0f}) - exiting now, no more entries today")
        for inst, p in hold:
            if not p.exiting:                                       # a refused exit is retried every second
                self.eng.exit(inst, 100, "auto: daily max loss", urgent=True, wait=True)
        with self.lock:
            self.want = None

    def _read(self, now):
        c = self.cfg
        if now < self.next_fetch:
            return
        bars = self._bars(c, now)
        if bars is None:
            return
        if len(bars) < ALMA_LEN + 2:
            self.next_fetch = now + 10
            self.msg = f"waiting for candle history ({len(bars)}/{ALMA_LEN + 2} bars)"
            return
        last_t = bars[-1]["t"]
        # the next bar closes one period after the last one ended; ask again then (the feed is cached for 15 s anyway)
        due = last_t + 2 * c["tf"] * 60 + 1
        self.next_fetch = due if now < due else now + 5.0
        if last_t == self.bar_t and self.base is not None:
            return
        trend = (ema_trend(bars, c["ema_fast"], c["ema_slow"]) if c["strategy"] == "EMA" else ttr_trend(bars))[-1]
        with self.lock:
            self.bar_t, prev, self.trend = last_t, self.base, trend
            if prev is None:
                self.base = trend                                   # enabling never trades the current trend
                self.note(f"{self._sname()} is {'bullish' if trend > 0 else 'bearish' if trend < 0 else 'neutral'} - waiting for it to flip")
                return
            if trend == prev or trend == 0:
                return
            self.base = trend
        self._on_flip(trend, last_t, now)

    def _sname(self):
        c = self.cfg
        return f"EMA {c['ema_fast']}/{c['ema_slow']} cross" if c["strategy"] == "EMA" else "ribbon"

    def _wanted(self, trend):
        mode = self.cfg["mode"]
        if mode == "BUY":
            return ("CE", "BUY") if trend > 0 else ("PE", "BUY")
        return ("CE", "SELL") if trend < 0 else ("PE", "SELL")

    def _on_flip(self, trend, bar_t, now):
        name = "bullish" if trend > 0 else "bearish"
        self.note(f"{self._sname()} turned {name}")
        for inst, p in self.holding():
            r = self.eng.exit(inst, 100, f"auto: {self._sname()} turned {name}", urgent=True, wait=True)
            self.note(f"exit {inst.split(':')[-1]} -> {r.get('msg')}")
        mins, _wd = self._clk()
        if self.loss_hit():
            return
        if mins < self.cfg["start"]:
            self.note(f"before the start time {_fmt(self.cfg['start'])}: no entry")
            return
        with self.lock:
            self.want = {"trend": trend, "bar": bar_t, "since": now, "next": 0.0, "n": 0}

    def _try_want(self, now, mins):
        w = self.want
        if not w or now < w["next"]:
            return
        if self.loss_hit() or w["trend"] != self.base or mins < self.cfg["start"] or mins >= self.cfg["end"] or now - w["since"] > self.RETRY_FOR_S:
            if w["trend"] == self.base and now - w["since"] > self.RETRY_FOR_S:
                self.note("entry abandoned: still refused after 60 s")
            self.want = None
            return
        if self.holding():                                          # the old leg has not closed yet: do not stack
            w["next"] = now + self.RETRY_S
            for inst, p in self.holding():
                if not p.exiting:
                    self.eng.exit(inst, 100, "auto: flip", urgent=True, wait=True)
            return
        w["next"] = now + self.RETRY_S
        c = self.cfg
        if self.entries >= c["max_trades"]:
            self.want = None
            self.note(f"daily max trades ({c['max_trades']}) reached - no entry; open positions still exit on a flip")
            return
        typ, side = self._wanted(w["trend"])
        try:
            inst, strike, atm = self.pick_fn(c["sym"], typ, c["ce_off"] if typ == "CE" else c["pe_off"])
        except ValueError as e:
            self.msg = f"cannot pick a {typ}: {e}"
            return
        bare = inst.split(":")[-1]
        with self.eng.lock:
            other = [i for i, p in self.eng.pos.items() if i.split(":")[-1] == bare and p.src != SRC]
        if other:                                                   # a fill would merge into it and auto would lose track of its own lots
            self.msg = f"you already hold {bare} outside Auto Trade - not adding to it"
            return
        r = self.eng.place({"inst": inst, "side": side, "lots": c["lots"], "exits": {}, "tag": "auto"},
                           req_id=f"auto{self.day}{w['bar']}{typ}{w['n']}", src=SRC)
        if r.get("dup"):                                            # the same request answered twice: already counted
            self.want = None
            return
        if r.get("ok"):
            self.want = None
            self.entries += 1
            self.note(f"{side} {c['lots']} lot(s) {c['sym']} {int(strike)} {typ} (ATM {int(atm)}) -> {r.get('msg')}")
        else:
            w["n"] += 1                                             # a refusal is cached 15 s under its req_id: use a fresh one
            self.msg = f"{side} {typ} refused: {r.get('msg')}"

    # ── what the tab shows ──
    def state(self):
        with self.lock:
            c = self.cfg
            hold = [{"inst": i, "side": "long" if p.side > 0 else "short", "typ": p.meta.get("typ"), "strike": p.meta.get("strike"),
                     "lots": p.qty // max(1, int(p.meta.get("lot") or 1))} for i, p in self.holding()]
            out = {"enabled": self.enabled, "cfg": {**c, "start": _fmt(c["start"]), "end": _fmt(c["end"])},
                   "trend": self.trend, "waiting_flip": self.enabled and not hold and self.want is None and not self.done_today,
                   "holding": hold, "entries": self.entries, "pnl": round(self.pnl(), 2), "loss_hit": self.loss_hit(),
                   "pending_entry": bool(self.want), "done_today": self.done_today,
                   "msg": self.msg, "events": list(self.events[-12:])}
        try:
            info = self.strikes_fn(c["sym"]) if self.strikes_fn else None
        except Exception:
            info = None
        out["strikes"] = info
        return out
