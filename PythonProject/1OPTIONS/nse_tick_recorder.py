"""Order-book tick recorder: every change to the book of the options near the money, the near future and the index,
as the Kite stream delivers it, to book_ticks/<UNDER>_<YYYY-MM-DD>.csv.gz.

Why: the replay/ snapshots are a minute apart, too coarse for the short-horizon order-flow reads (order-flow
imbalance from best bid/ask changes, depth imbalance, queue changes) that bias_eval.py could not test. A few weeks
of this makes them testable.

Cost: no broker call. It taps the ticks the adapter already receives (FULL mode), so it records exactly what is
subscribed; it never subscribes anything. Rows go to an in-memory queue from the tick callback (append only) and a
background thread writes them, so a slow disk cannot stall the stream. A recorder failure never reaches the stream.

What is kept: NIFTY (UNDERS) options whose strike is within SPAN points of the index (the strike grid
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

One writer per file: each process appends its own gzip stream, so two processes writing the same day's file interleave
their bytes and everything after that point is unreadable (Nov 2026: four sessions lost this way to a second server or
adapter process). A process takes an exclusive lock (book_ticks/.<UNDER>_<day>.lock) before it opens the day's file;
another process finding it held records nothing for that file (stats["locked_out"], last_error names the holder) and
retries every LOCK_RETRY_S, so it takes over when the holder exits. read_day() also skips a damaged stretch (a member
cut mid-block by a kill, or bytes interleaved by an older writer) and resumes at the next member.

Size: NIFTY only is roughly 5-20 MB a day, 1.5-5 GB a year.
Retention: the server's daily post-close cleanup prunes book_ticks/ by age, 730 days (see nse_chain_server._run_cleanup).
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
UNDERS = ("NIFTY",)               # NIFTY only: BANKNIFTY is not traded
SPAN = 500                      # points either side of the index
OPEN_MIN, CLOSE_MIN = 9 * 60, 15 * 60 + 31
FLUSH_S = 5.0
QUEUE_MAX = 200_000             # a stalled disk drops the oldest rows rather than eating memory
LOCK_RETRY_S = 30.0             # how often a locked-out process tries the day's lock again
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
        self.locks = {}                 # (under, day) -> open lock-file handle (held while the file is ours)
        self.lock_try = {}              # (under, day) -> last failed lock attempt
        self.lock = threading.Lock()
        self.stats = {"rows": 0, "dropped": 0, "files": 0, "errors": 0, "locked_out": 0, "last_error": ""}
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
    def _lock(self, key, now):
        """Exclusive lock on the day's file for this process; False while another process holds it."""
        if key in self.locks:
            return True
        if now - self.lock_try.get(key, -1e18) < LOCK_RETRY_S:
            return False
        os.makedirs(self.root, exist_ok=True)
        path = os.path.join(self.root, f".{key[0]}_{key[1]}.lock")
        fh = open(path, "a+", encoding="utf-8")
        try:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except ImportError:               # Windows
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            try:
                fh.seek(0)
                who = fh.read().strip()[:40] or "another process"
            except Exception:  # noqa: BLE001
                who = "another process"
            fh.close()
            self.lock_try[key] = now
            self.stats["last_error"] = f"{key[0]}_{key[1]}.csv.gz is being recorded by {who}; this process records nothing to it"
            return False
        try:
            fh.seek(0); fh.truncate(); fh.write(f"pid {os.getpid()}"); fh.flush()
        except Exception:  # noqa: BLE001
            pass
        self.locks[key] = fh
        self.lock_try.pop(key, None)
        return True

    def _unlock(self, key):
        fh = self.locks.pop(key, None)
        if fh is not None:
            try:
                fh.close()                    # closing the handle releases the lock
            except Exception:  # noqa: BLE001
                pass

    def _handle(self, under, day, now=None):
        key = (under, day)
        fh = self.files.get(key)
        if fh is None:
            if not self._lock(key, time.time() if now is None else now):
                return None
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
                    fh = self._handle(under, day, now)
                    if fh is None:
                        self.stats["locked_out"] += 1
                        continue
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
                    self._unlock(key)
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
            for key in list(self.locks):
                self._unlock(key)

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


_CHUNK = 1 << 16


def _member(raw: bytes, pos: int):
    """Decompress one gzip member from pos. Returns (text, next_pos, clean): next_pos is where the following member
    starts when clean. A broken member still yields what decoded before the break: the chunk that fails is fed
    again from a saved state in small pieces, since a failing decompress() call returns nothing of its own."""
    dec, out, p = zlib.decompressobj(31), [], pos
    while p < len(raw) and not dec.eof:
        chunk = raw[p:p + _CHUNK]
        saved = dec.copy()
        try:
            out.append(dec.decompress(chunk))
        except zlib.error:
            dec = saved
            for i in range(0, len(chunk), 256):
                piece, sv = chunk[i:i + 256], dec.copy()
                try:
                    out.append(dec.decompress(piece))
                except zlib.error:
                    dec = sv                          # the last good bytes, one at a time
                    for j in range(len(piece)):
                        try:
                            out.append(dec.decompress(piece[j:j + 1]))
                        except zlib.error:
                            break
                    break
                if dec.eof:
                    break
            return b"".join(out), None, False
        p += len(chunk)
    if dec.eof:
        return b"".join(out), p - len(dec.unused_data), True
    return b"".join(out), None, False


def _next_header(raw: bytes, start: int) -> int:
    """Offset of the next plausible gzip member header (magic, deflate, sane flags) at or after start, or -1."""
    i = raw.find(b"\x1f\x8b\x08", start)
    while i != -1 and i + 10 <= len(raw) and raw[i + 3] & 0xE0:
        i = raw.find(b"\x1f\x8b\x08", i + 1)
    return i


def read_day(path: str):
    """Rows of one recorded file as dicts. Tolerates a truncated tail (a crash before the gzip trailer), and a
    damaged stretch in the middle (a member cut by a kill, or interleaved by a second writer): it keeps what
    decoded before the damage and resumes at the next member that decodes."""
    raw = open(path, "rb").read()
    out, pos = [], 0
    while 0 <= pos < len(raw):
        text, nxt, clean = _member(raw, pos)
        out.append(text)
        if clean:
            pos = nxt
            continue
        pos = _next_header(raw, pos + 1)            # damaged: look for the next member that decodes
        if text:
            out.append(b"\n")                      # the damaged member's last line may be cut: keep it apart
    text = b"".join(out).decode("utf-8", "replace")
    head, n = ",".join(COLS), len(COLS) - 1
    lines = [ln for ln in text.split("\n") if ln and ln != head and ln.count(",") == n and ln[:13].isdigit()]
    return list(csv.DictReader(io.StringIO("\n".join([head] + lines))))


_rec = None


def get() -> Recorder:
    """The process-wide recorder, started on first use. NSE_BOOK_TICKS=0 turns it off."""
    global _rec
    if _rec is None:
        off = os.environ.get("NSE_BOOK_TICKS", "1").strip().lower() in ("0", "false", "no", "off")
        _rec = False if off else Recorder().start()
    return _rec or None
