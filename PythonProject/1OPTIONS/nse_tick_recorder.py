"""Order-book tick recorder: every change to the book of the options near the money, the near future and the index,
as the Kite stream delivers it, to book_ticks/<UNDER>_<YYYY-MM-DD>.csv.gz.

Why: the replay/ snapshots are a minute apart, too coarse for the short-horizon order-flow reads (order-flow
imbalance from best bid/ask changes, depth imbalance, queue changes) that bias_eval.py could not test. A few weeks
of this makes them testable.

Cost: no broker call. It taps the ticks the adapter already receives (FULL mode), so it records exactly what is
subscribed; it never subscribes anything. Rows go to an in-memory queue from the tick callback (append only) and a
background thread writes them, so a slow disk cannot stall the stream. A recorder failure never reaches the stream.

What is kept: NIFTY and BANKNIFTY (UNDERS) options whose strike is within SPAN points of the index (the strike grid
shows ATM +/- 5 round hundreds = 500 points), every future of those indices, and the index itself; 09:00-15:31 IST.
A row is written only when something in it changed.

Columns (CSV, header on the first line of each file):
  t_ms     receive time, epoch ms           ex_ms   exchange timestamp, epoch ms ('' if absent)
  inst     Kite key, e.g. NFO:NIFTY26N1024500CE
  kind     OPT / FUT / IDX                  strike, typ (CE/PE), expiry   options only
  ltp, vol, oi, bid, bq, ask, aq, tbq, tsq
  b5, a5   5-level depth, "price:qty:orders" joined by '|'

Files are gzip, flushed every FLUSH_S seconds with a sync flush, so a crash loses at most that much and the file stays
readable up to the last flush; read them with read_day(), which tolerates a truncated tail. A restart on the same day
appends a new gzip member, which every gzip reader accepts.

Retention: the server's daily post-close cleanup prunes book_ticks/ by age (see nse_chain_server._run_cleanup).
"""
import collections
import csv
import gzip
import io
import os
import threading
import time
import zlib
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "book_ticks")
UNDERS = ("NIFTY", "BANKNIFTY")
SPAN = 500                      # points either side of the index
OPEN_MIN, CLOSE_MIN = 9 * 60, 15 * 60 + 31
FLUSH_S = 5.0
QUEUE_MAX = 200_000             # a stalled disk drops the oldest rows rather than eating memory
COLS = ("t_ms", "ex_ms", "inst", "kind", "strike", "typ", "expiry", "ltp", "vol", "oi",
        "bid", "bq", "ask", "aq", "tbq", "tsq", "b5", "a5")
IST = timezone(timedelta(hours=5, minutes=30))


def _ist(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, IST)


def _num(v):
    if v is None:
        return ""
    f = float(v)
    return str(int(f)) if f == int(f) else repr(round(f, 4))


def _levels(lv) -> str:
    out = []
    for x in lv or []:
        try:
            out.append(f"{_num(x.get('p'))}:{_num(x.get('q'))}:{_num(x.get('n'))}")
        except Exception:
            continue
    return "|".join(out)


class Recorder:
    def __init__(self, root: str = DIR, unders=UNDERS, span: float = SPAN):
        self.root, self.unders, self.span = root, tuple(unders), float(span)
        self.q = collections.deque(maxlen=QUEUE_MAX)
        self.spot = {}                  # under -> last index ltp
        self.last = {}                  # inst -> last written payload (dedupe)
        self.files = {}                 # (under, day) -> gzip handle
        self.lock = threading.Lock()
        self.stats = {"rows": 0, "dropped": 0, "files": 0, "errors": 0, "last_error": ""}
        self.last_flush = 0.0
        self._thread = None
        self._stop = threading.Event()

    # ── tick side: cheap, never raises ──
    def note(self, now, ex_ts, inst, kind, under, d, b5=None, a5=None, tbq=None, tsq=None, meta=None):
        try:
            if under not in self.unders:
                return
            m = _ist(now)
            mins = m.hour * 60 + m.minute
            if mins < OPEN_MIN or mins >= CLOSE_MIN or m.weekday() >= 5:
                return
            if kind == "IDX":
                if d.get("ltp"):
                    self.spot[under] = float(d["ltp"])
            elif kind == "OPT":
                sp = self.spot.get(under)
                k = (meta or {}).get("strike")
                if sp is None or k is None or abs(float(k) - sp) > self.span:
                    return
            elif kind != "FUT":
                return
            meta = meta or {}
            payload = (_num(d.get("ltp")), _num(d.get("volume")), _num(d.get("oi")), _num(d.get("bid")), _num(d.get("bid_qty")),
                       _num(d.get("ask")), _num(d.get("ask_qty")), _num(tbq), _num(tsq), _levels(b5), _levels(a5))
            if self.last.get(inst) == payload:
                return
            self.last[inst] = payload
            if len(self.q) == self.q.maxlen:
                self.stats["dropped"] += 1
            self.q.append((m.strftime("%Y-%m-%d"), under, (str(int(now * 1000)), str(int(ex_ts * 1000)) if ex_ts else "", inst, kind,
                                                          _num(meta.get("strike")), meta.get("typ") or "", str(meta.get("expiry") or ""))
                           + payload))
        except Exception as e:  # noqa: BLE001
            self.stats["errors"] += 1
            self.stats["last_error"] = f"note: {type(e).__name__} {e}"

    # ── writer side ──
    def _handle(self, under, day):
        key = (under, day)
        fh = self.files.get(key)
        if fh is None:
            os.makedirs(self.root, exist_ok=True)
            path = os.path.join(self.root, f"{under}_{day}.csv.gz")
            new = not os.path.exists(path) or os.path.getsize(path) == 0
            fh = gzip.open(path, "at", encoding="utf-8", newline="", compresslevel=6)
            if new:
                fh.write(",".join(COLS) + "\n")
            self.files[key] = fh
            self.stats["files"] += 1
        return fh

    def drain(self, force_flush: bool = False, now: float = None):
        """Write everything queued; sync-flush every FLUSH_S; close files of past days and after the close."""
        now = time.time() if now is None else now
        with self.lock:
            touched = set()
            try:
                while self.q:
                    day, under, row = self.q.popleft()
                    fh = self._handle(under, day)
                    csv.writer(fh, lineterminator="\n").writerow(row)
                    self.stats["rows"] += 1
                    touched.add((under, day))
                if force_flush or now - self.last_flush >= FLUSH_S:
                    for fh in self.files.values():
                        fh.flush()
                    self.last_flush = now
                m = _ist(now)
                today, closed = m.strftime("%Y-%m-%d"), m.hour * 60 + m.minute >= CLOSE_MIN + 1
                for key in [k for k in self.files if k[1] != today or closed]:
                    self.files.pop(key).close()
                    if key[1] != today:
                        self.last.clear()
            except Exception as e:  # noqa: BLE001
                self.stats["errors"] += 1
                self.stats["last_error"] = f"write: {type(e).__name__} {e}"

    def close(self):
        self.drain(force_flush=True)
        with self.lock:
            for fh in self.files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self.files.clear()

    def start(self):
        if self._thread:
            return self

        def loop():
            while not self._stop.wait(1.0):
                self.drain()
            self.close()
        self._thread = threading.Thread(target=loop, name="book-ticks", daemon=True)
        self._thread.start()
        import atexit
        atexit.register(self.stop)
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.close()


def read_day(path: str):
    """Rows of one recorded file as dicts. Tolerates a truncated tail (a crash before the gzip trailer)."""
    raw = open(path, "rb").read()
    out, pos = [], 0
    while pos < len(raw):
        dec = zlib.decompressobj(31)
        try:
            out.append(dec.decompress(raw[pos:]))
            out.append(dec.flush())
        except zlib.error:
            break
        if not dec.eof:
            break                                   # truncated member: keep what decompressed
        pos = len(raw) - len(dec.unused_data)
    text = b"".join(out).decode("utf-8", "replace")
    lines = [ln for ln in text.split("\n") if ln and ln != ",".join(COLS)]
    if lines and not lines[-1].count(",") == len(COLS) - 1:
        lines.pop()                                 # a half-written last line
    return list(csv.DictReader(io.StringIO("\n".join([",".join(COLS)] + lines))))


_rec = None


def get() -> Recorder:
    """The process-wide recorder, started on first use. NSE_BOOK_TICKS=0 turns it off."""
    global _rec
    if _rec is None:
        off = os.environ.get("NSE_BOOK_TICKS", "1").strip().lower() in ("0", "false", "no", "off")
        _rec = False if off else Recorder().start()
    return _rec or None
