"""Permanent, compact archive of the data the research tools need, kept apart from the caches the
post-close cleanup prunes.

  archive/index_daily.csv          date,index,open,high,low,close   (Nifty 50, Nifty Bank, India VIX, ...)
  archive/index_daily_missing.txt  weekdays NSE has no file for (holidays), so the backfill never asks twice
  archive/minutes/<SYM>_<YYYY>.csv t,o,h,l,c   1-minute index bars, one block of rows per session

Why: bhavcopy/ is a download cache, history/ keeps only today's 60-day broker cache, warm_cache/ 15 days,
ticks/ a year of spot samples. pro_calibrate.py (clock, VER, HAR, VRP, tails) and the historical-scenario
risk in nse_span.py read this archive, so its history only grows.

Sources: the NSE index close file (archives.nseindia.com ind_close_all_DDMMYYYY.csv, public, no broker call)
and the minute bars already on disk. Nothing here calls the broker.

  python3 nse_archive.py                       # archive what is on disk now
  python3 nse_archive.py --backfill 2019-01-01 # one-time: daily index history from NSE (about 1,900 files, ~15 min)
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
IST = timezone(timedelta(hours=5, minutes=30))
KEEP_INDICES = ("Nifty 50", "Nifty Bank", "India VIX", "Nifty Financial Services", "Nifty Midcap Select")
ALIASES = {"CNX Nifty": "Nifty 50", "CNX Bank": "Nifty Bank", "CNX Finance": "Nifty Financial Services"}   # names before Nov 2015
MIN_SYMS = ("NIFTY", "BANKNIFTY")
URL = "https://archives.nseindia.com/content/indices/ind_close_all_{}.csv"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Accept": "text/csv,*/*"}


def _dir(root):
    return os.path.join(root, "archive")


def _atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- daily index closes
def _f(v):
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


def parse_ind_close(text):
    """[(date, index, o, h, l, c)] for the kept indices from one ind_close_all file."""
    out = []
    for r in csv.reader(io.StringIO(text)):
        name = ALIASES.get(r[0].strip(), r[0].strip()) if r else ""
        if len(r) < 6 or name not in KEEP_INDICES:
            continue
        try:
            d = datetime.strptime(r[1].strip(), "%d-%m-%Y").date()
        except ValueError:
            continue
        o, h, l, c = (_f(x) for x in r[2:6])
        if c and c > 0:
            out.append((d, name, o or c, h or c, l or c, c))
    return out


def read_daily(root=HERE):
    """{(date, index): (o, h, l, c)}"""
    out = {}
    p = os.path.join(_dir(root), "index_daily.csv")
    if os.path.exists(p):
        for r in csv.reader(open(p, encoding="utf-8")):
            if len(r) == 6 and r[0] != "date":
                try:
                    out[(date.fromisoformat(r[0]), r[1])] = tuple(float(x) for x in r[2:])
                except ValueError:
                    continue
    return out


def write_daily(root, rows):
    """Merge [(date, index, o, h, l, c)] into the archive. Returns the number of new (date, index) rows."""
    cur = read_daily(root)
    n0 = len(cur)
    for d, ix, o, h, l, c in rows:
        cur[(d, ix)] = (o, h, l, c)
    if len(cur) == n0 and not rows:
        return 0
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(("date", "index", "open", "high", "low", "close"))
    for (d, ix) in sorted(cur):
        w.writerow((d.isoformat(), ix) + tuple(round(x, 4) for x in cur[(d, ix)]))
    _atomic_write(os.path.join(_dir(root), "index_daily.csv"), buf.getvalue())
    return len(cur) - n0


def archive_bhav_dir(root=HERE):
    """Copy the kept indices out of every cached bhavcopy file (the cache is a download cache, not a record)."""
    rows = []
    for f in glob.glob(os.path.join(root, "nse_data_cache", "bhavcopy", "*.csv")):
        try:
            rows += parse_ind_close(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:  # noqa: BLE001
            continue
    return write_daily(root, rows)


def drop_spikes(rows, jump=0.04, back=0.4):
    """Remove isolated bad prints: a close more than `jump` away from both neighbours that the next session fully
    undoes (|in + out| < back x the larger leg). NSE's own files carry a few: 2023-10-04 shows Nifty at 17,624 between
    two ~19,500 closes, and a Saturday and a Sunday file repeat a stale level. A real crash is not undone the next day."""
    import math
    out = list(rows)
    i = 1
    while i < len(out) - 1:
        a, b, c = out[i - 1][4], out[i][4], out[i + 1][4]
        rin, rout = math.log(b / a), math.log(c / b)
        if abs(rin) > jump and abs(rout) > jump and rin * rout < 0 and abs(rin + rout) < back * max(abs(rin), abs(rout)):
            del out[i]
            continue
        i += 1
    return out


def nifty_vix_daily(root=HERE, index="Nifty 50"):
    """[(date, o, h, l, c, vix_close or None)] for one index, oldest first, isolated bad prints removed."""
    cur = read_daily(root)
    vix = {d: v[3] for (d, ix), v in cur.items() if ix == "India VIX"}
    return drop_spikes([(d,) + cur[(d, ix)] + (vix.get(d),) for (d, ix) in sorted(cur) if ix == index])


def _missing_path(root):
    return os.path.join(_dir(root), "index_daily_missing.txt")


def _read_missing(root):
    p = _missing_path(root)
    if not os.path.exists(p):
        return set()
    out = set()
    for line in open(p):
        try:
            out.add(date.fromisoformat(line.strip()[:10]))
        except ValueError:
            pass
    return out


def _fetch(url, timeout=12):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        return e.code, ""


def backfill_daily(root=HERE, start=None, end=None, pause=0.35, fetch=_fetch, log=print, max_requests=None, today=None):
    """Download ind_close_all for every weekday in [start, end] the archive lacks. A 404 more than a week old is
    recorded as a holiday and never asked again; a recent one is retried next time (NSE publishes in the evening).
    Progress is saved every 50 files, so an interrupted run resumes where it stopped."""
    today = today or datetime.now(IST).date()
    end = min(end or today, today)
    have = {d for (d, ix) in read_daily(root) if ix == "Nifty 50"}
    miss = _read_missing(root)
    start = start or (min(have) if have else today - timedelta(days=30))
    todo, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in have and d not in miss:
            todo.append(d)
        d += timedelta(days=1)
    if max_requests is not None:
        todo = todo[-max_requests:]                 # newest first matters most for a capped daily catch-up
    got, holidays, errors, rows = 0, [], 0, []

    def flush():
        write_daily(root, rows)
        rows.clear()
        if holidays:
            os.makedirs(_dir(root), exist_ok=True)
            with open(_missing_path(root), "a") as f:
                for h in holidays:
                    f.write(h.isoformat() + "\n")
            holidays.clear()
    for i, d in enumerate(todo):
        try:
            st, txt = fetch(URL.format(d.strftime("%d%m%Y")))
        except Exception as e:  # noqa: BLE001
            st, txt = None, ""
            errors += 1
            if errors >= 8 and errors > got:
                log(f"[archive] backfill stopping: {type(e).__name__}: {e}")
                break
        if st == 200 and txt.strip():
            r = parse_ind_close(txt)
            if r:
                rows += r
                got += 1
        elif st == 404 and (today - d).days > 7:
            holidays.append(d)
        if (i + 1) % 50 == 0:
            flush()
            log(f"[archive] backfill {i + 1}/{len(todo)}: {got} days, through {d}")
        if pause and i + 1 < len(todo):
            time.sleep(pause)
    flush()
    return {"asked": len(todo), "got": got, "errors": errors}


# ----------------------------------------------------------------------------- minute bars
def _ist(ts):
    return datetime.fromtimestamp(ts, IST)


def _source_bars(root, sym):
    """{date: {minute_epoch: (o, h, l, c, priority)}} from the caches on disk, best source per minute."""
    days = {}

    def put(t, o, h, l, c, pri):
        t = int(t) // 60 * 60
        g = _ist(t)
        m = g.hour * 60 + g.minute
        if not 555 <= m < 930:
            return
        cur = days.setdefault(g.date(), {})
        if t not in cur or pri >= cur[t][4]:
            cur[t] = (o, h, l, c, pri)
    for f in glob.glob(os.path.join(root, "ticks", f"{sym}_*.csv")):
        for line in open(f):
            p = line.strip().split(",")
            if len(p) >= 2:
                try:
                    px = float(p[1])
                    put(float(p[0]), px, px, px, px, 1)
                except ValueError:
                    pass
    for f in glob.glob(os.path.join(root, "history", f"{sym}_*.json")) + glob.glob(os.path.join(root, "warm_cache", f"{sym}_*.json")):
        try:
            j = json.load(open(f))
            bars = j.get("bars", []) if isinstance(j, dict) else j
            pri = 3 if "warm_cache" in f else 2
            for b in bars:
                put(b["t"], b["o"], b.get("h", b["c"]), b.get("l", b["c"]), b["c"], pri)
        except Exception:  # noqa: BLE001
            continue
    return days


def read_minutes(root=HERE, sym="NIFTY"):
    """{date: {minute_epoch: (o, h, l, c)}} from the archive."""
    out = {}
    for f in sorted(glob.glob(os.path.join(_dir(root), "minutes", f"{sym}_*.csv"))):
        for r in csv.reader(open(f, encoding="utf-8")):
            if len(r) != 5 or r[0] == "t":
                continue
            try:
                t = int(r[0])
                out.setdefault(_ist(t).date(), {})[t] = tuple(float(x) for x in r[1:])
            except ValueError:
                continue
    return out


def archive_minutes(root=HERE, syms=MIN_SYMS, today=None):
    """Add every complete past session found in the caches to archive/minutes. A session already archived is
    replaced only by a fuller copy. Today's session is left until it is over. Returns {sym: sessions added}."""
    today = today or datetime.now(IST).date()
    now_m = datetime.now(IST).hour * 60 + datetime.now(IST).minute
    res = {}
    for sym in syms:
        src = _source_bars(root, sym)
        arch = read_minutes(root, sym)
        changed_years, added = set(), 0
        for d, bars in src.items():
            if d > today or (d == today and now_m < 935) or len(bars) < 300:
                continue
            if len(arch.get(d, {})) >= len(bars):
                continue
            arch[d] = {t: v[:4] for t, v in bars.items()}
            changed_years.add(d.year)
            added += 1
        for y in changed_years:
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow(("t", "o", "h", "l", "c"))
            for d in sorted(x for x in arch if x.year == y):
                for t in sorted(arch[d]):
                    w.writerow((t,) + tuple(round(x, 2) for x in arch[d][t]))
            _atomic_write(os.path.join(_dir(root), "minutes", f"{sym}_{y}.csv"), buf.getvalue())
        res[sym] = added
    return res


def run_daily(root=HERE, log=print, fetch=_fetch, backfill=True):
    """The post-close step: archive the caches, then catch up any missing daily closes from NSE (at most 15 files)."""
    out = {"minutes": archive_minutes(root), "bhav_rows": archive_bhav_dir(root)}
    if backfill:
        try:
            out["backfill"] = backfill_daily(root, start=datetime.now(IST).date() - timedelta(days=45), fetch=fetch,
                                             log=log, max_requests=15)
        except Exception as e:  # noqa: BLE001
            out["backfill"] = {"error": f"{type(e).__name__}: {e}"}
    log(f"[archive] {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--backfill", metavar="YYYY-MM-DD", help="download daily index closes from this date")
    ap.add_argument("--pause", type=float, default=0.35)
    a = ap.parse_args()
    print(f"[archive] minutes: {archive_minutes(a.root)}  bhavcopy rows added: {archive_bhav_dir(a.root)}")
    if a.backfill:
        print(f"[archive] backfill: {backfill_daily(a.root, start=date.fromisoformat(a.backfill), pause=a.pause)}")
    d = nifty_vix_daily(a.root)
    m = read_minutes(a.root, "NIFTY")
    print(f"[archive] daily Nifty: {len(d)} sessions" + (f" {d[0][0]} to {d[-1][0]}" if d else "")
          + f"; minute sessions NIFTY: {len(m)}" + (f" {min(m)} to {max(m)}" if m else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
