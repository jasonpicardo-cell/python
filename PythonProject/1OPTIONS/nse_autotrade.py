"""Auto Trade for the Scalping tab: a signal (Trend Ribbon, EMA crossover, opening-range break, VWAP sigma
playbook or a failed break of yesterday's high/low), optional desk filters, and optional exits. Server side.

Runs in the server process, so closing or reloading the browser tab changes nothing.
Orders go through the same Engine as manual ones (paper by default, ARM needed for live).

Rules
  * Enabling never trades the current signal: the first trade is the next CHANGE of it.
  * Signals are read from CLOSED candles only (a flip never appears mid-bar and vanishes).
  * BUY mode:  bull -> buy CE, bear -> buy PE.   SELL mode: bear -> sell CE, bull -> sell PE.
  * A flip closes the auto position (CE bought is sold, CE sold is bought back) and opens the other.
    The setup strategies (ORB, VWAP, TRAP) also go flat: their signal returns to 0 and the position is closed.
  * Filters (all off by default) only hold back an ENTRY; they never close a position. A held-back entry
    waits up to CONFIRM_BARS more candles for the filters to agree, then is skipped. START ignores them.
  * Exits (all off by default): flat exit (close back through the ribbon midline / slow EMA), time stop
    (no 1-ATR move in N candles), scale-out (half off at 1 ATR in favour). None of them re-enters: the next
    entry still needs a new signal.
  * Entries only from `start` to before `end`; at `end` the auto position is closed and
    nothing more is opened that day.
  * The contract is chosen when the entry is sent (ATM + offset, or the strike nearest a target delta,
    snapped to a listed round-hundred strike) and the exit always uses the contract that was actually bought.
  * PREMIUM is not directional: it sells an iron fly (or a straddle) at the pin strike on a range day with dealers
    long gamma (or on expiry day after 13:30), with a combined stop / target as a % of the credit, and closes it
    when the index leaves the tent (half the wing width from the centre).
  * Trade management (all off by default): stop / trail / breakeven given in index ATRs, turned into option points
    with the contract's delta and armed on the engine (so they are watched on every tick); one re-entry after a
    stop while the signal still holds; a day profit lock (after the auto P&L reaches X, giving back Y% of its
    peak closes everything and ends the day).
Positions are recognised by src="auto" on the engine's book, so nothing is lost if this
object is rebuilt (the entry reference used by the exits is, and those exits then stay idle).

The signal, filter and exit maths are plain functions so auto_eval.py replays them on recorded days.
"""
import math
import re
import threading
import time

IST = 19800
SRC = "auto"

# Trend Target Ribbon [BOSWaves] defaults, as in the dashboard's TTR object
ALMA_LEN, ALMA_OFFSET, ALMA_SIGMA = 34, 0.85, 6.0
DEV_LEN, DEV_MULT = 34, 0.65
SLOPE_LEN, SLOPE_MIN = 3, 0.08
ATR_LEN = 14

STRATEGIES = ("RIBBON", "EMA", "ORB", "VWAP", "TRAP", "PREMIUM")
CLASSIC = ("RIBBON", "EMA")                 # always in a direction once warmed up; 0 only before that
GATES = ("gate_day", "gate_gex", "gate_book", "gate_vwap", "gate_lead", "gate_vol", "gate_flow", "gate_oi")
FLOW_MIN = 0.10                             # futures 5-min taker imbalance needed with the trade
OI_MIN = 0.10                               # delta-weighted 15-min option-writer read needed with the trade
PIN_AFTER = 13 * 60 + 30                    # expiry-day pin mode starts here
PREM_COOL_MIN = 15                          # after a structure closes, PREMIUM waits this long before selling again
CONFIRM_BARS = 3                            # a filtered entry waits this many more candles, then is skipped
LEAD = {"NIFTY": "BANKNIFTY", "BANKNIFTY": "NIFTY", "FINNIFTY": "BANKNIFTY", "MIDCPNIFTY": "NIFTY"}
OPEN_MIN, DAY_WIN0, DAY_WIN1 = 9 * 60 + 15, 9 * 60 + 20, 9 * 60 + 45
TRAP_UNTIL = 14 * 60 + 30

# Python twin of _BF in nse_dashboard.html (the server reloads it from the HTML, so the two never drift)
BF = {"tau": 1680.0, "lb": 120.0, "mod": 0.6, "strong": 1.4, "agg": 0.35, "fadeMin": 900.0, "fadeBps": 15.0,
      "oiMin": 0.01, "oiImb": 0.15}


def bf_from_html(path):
    html = open(path, encoding="utf-8").read()
    m = re.search(r"const _BF = \{([^}]*)\}", html)
    return {k: float(v) for k, v in re.findall(r"(\w+):\s*([\d.]+)", m.group(1))} if m else {}


# ── time ──
def ist_min(t):
    g = time.gmtime(t + IST)
    return g.tm_hour * 60 + g.tm_min


def ist_date(t):
    return time.strftime("%Y-%m-%d", time.gmtime(t + IST))


def today_start(bars):
    """Index of the first bar of the last session in `bars` (None if empty)."""
    if not bars:
        return None
    d = ist_date(bars[-1]["t"])
    i = len(bars) - 1
    while i > 0 and ist_date(bars[i - 1]["t"]) == d:
        i -= 1
    return i


def prev_day_hl(bars, i0):
    """High and low of the session before bars[i0] (None, None when not loaded)."""
    if not i0:
        return None, None
    d = ist_date(bars[i0 - 1]["t"])
    pv = [b for b in bars[:i0] if ist_date(b["t"]) == d]
    return max(b["h"] for b in pv), min(b["l"] for b in pv)


# ── indicators ──
def atr_series(bars, n=ATR_LEN):
    """Wilder ATR, seeded with the SMA of the first n true ranges (as the ribbon's)."""
    h, lo, c = [b["h"] for b in bars], [b["l"] for b in bars], [b["c"] for b in bars]
    tr = [(h[0] - lo[0]) if i == 0 else max(h[i] - lo[i], abs(h[i] - c[i - 1]), abs(lo[i] - c[i - 1])) for i in range(len(bars))]
    atr = [None] * len(bars)
    for t in range(len(bars)):
        if t == n - 1:
            atr[t] = sum(tr[:n]) / n
        elif t >= n:
            atr[t] = (atr[t - 1] * (n - 1) + tr[t]) / n
    return atr


def _ttr(bars):
    """(trend per bar, ALMA midline per bar) of the Trend Target Ribbon."""
    n = len(bars)
    c = [b["c"] for b in bars]
    m, sg = ALMA_OFFSET * (ALMA_LEN - 1), ALMA_LEN / ALMA_SIGMA
    wts = [math.exp(-((i - m) ** 2) / (2 * sg * sg)) for i in range(ALMA_LEN)]
    norm = sum(wts)
    alma = [None] * n
    dev = [None] * n
    for t in range(ALMA_LEN - 1, n):
        alma[t] = sum(c[t - (ALMA_LEN - 1 - i)] * wts[i] for i in range(ALMA_LEN)) / norm
    for t in range(DEV_LEN - 1, n):
        win = c[t - DEV_LEN + 1:t + 1]
        mu = sum(win) / DEV_LEN
        dev[t] = math.sqrt(sum((x - mu) ** 2 for x in win) / DEV_LEN)
    atr = atr_series(bars)
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
    return out, alma


def ttr_trend(bars):
    """Per-bar trend (+1 bull, -1 bear, 0 none yet) of the Trend Target Ribbon.
    Same maths as the dashboard's _ttrCompute (checked bar for bar in test_nse.py)."""
    return _ttr(bars)[0]


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


def vwap_bands(cs):
    """Twin of the dashboard's _vwapBands for an index chart (no volume: every bar weighs 1).
    -> (vwap per bar, sigma per bar), cumulative over `cs` (one session)."""
    vw, sd = [], []
    if not cs:
        return vw, sd
    ref = (cs[0]["h"] + cs[0]["l"] + cs[0]["c"]) / 3
    sw = s1 = s2 = 0.0
    for b in cs:
        tp = (b["h"] + b["l"] + b["c"]) / 3 - ref
        sw += 1; s1 += tp; s2 += tp * tp
        m = s1 / sw
        vw.append(m + ref)
        sd.append(math.sqrt(max(0.0, s2 / sw - m * m)))
    return vw, sd


def day_type_series(cs, vw=None, sd=None):
    """Twin of the dashboard's _dayType, read after every bar of one session: [(kind, dir)].
    kind: unknown | forming | range | trend | trend-broken. A VWAP cross (0.25 sigma hysteresis) between 9:20 and
    9:45 makes a range day; none makes a trend day, which is 'broken' at its first cross after 9:45."""
    if vw is None:
        vw, sd = vwap_bands(cs)
    out = []
    if not cs:
        return out
    if ist_min(cs[0]["t"]) > DAY_WIN0:
        return [("unknown", 0)] * len(cs)
    side = crosses = win_side = 0
    done, later = False, None
    for i, b in enumerate(cs):
        m = ist_min(b["t"])
        if m >= DAY_WIN0:
            s_, dv = sd[i] or 0.0, b["c"] - vw[i]
            s = 1 if dv > 0.25 * s_ else -1 if dv < -0.25 * s_ else 0
            if s:
                if m < DAY_WIN1:
                    if side and s != side:
                        crosses += 1
                    side = win_side = s
                else:
                    done = True
                    if side and s != side and later is None:
                        later = b["t"]
                    if later is None:
                        side = s
        if not done:
            out.append(("forming", 0))
        elif not win_side or crosses >= 1:
            out.append(("range", 0))
        else:
            d = 1 if side >= 0 else -1
            out.append(("trend-broken" if later else "trend", d))
    return out


# ── setup strategies: per-bar wanted side (+1 / -1 / 0 = flat) ──
def orb_series(bars, orb_min=15, tf=1, until=11 * 60):
    """Opening-range break: the range is 09:15 + orb_min. A close beyond it goes that way (once per side per day,
    only before `until`); a close back through the range's middle goes flat."""
    out = [0] * len(bars)
    i0 = today_start(bars)
    if i0 is None or ist_min(bars[i0]["t"]) > OPEN_MIN:
        return out
    end = OPEN_MIN + orb_min
    rng = [b for b in bars[i0:] if ist_min(b["t"]) + tf <= end]
    if not rng:
        return out
    hi, lo = max(b["h"] for b in rng), min(b["l"] for b in rng)
    mid = (hi + lo) / 2
    side, fired = 0, set()
    for i in range(i0, len(bars)):
        b = bars[i]
        m = ist_min(b["t"])
        if m + tf <= end:
            continue
        c = b["c"]
        if (side > 0 and c < mid) or (side < 0 and c > mid):
            side = 0
        if not side and m < until:
            if c > hi and 1 not in fired:
                side = 1; fired.add(1)
            elif c < lo and -1 not in fired:
                side = -1; fired.add(-1)
        out[i] = side
    return out


def vwap_series(bars):
    """The VWAP sigma playbook, by day type (read at 9:45, as the chart's badge):
       trend day:  buy a pullback that holds +1 sigma (sell one that fails at -1 sigma); flat on a close through VWAP.
       range day (or a broken trend): fade a close back inside +-2 sigma toward VWAP; flat at VWAP, or 0.5 sigma
       beyond the band (stop).
    -> (sides per bar, style per bar: 'trend' | 'fade' | None)."""
    n = len(bars)
    out, sty = [0] * n, [None] * n
    i0 = today_start(bars)
    if i0 is None:
        return out, sty
    cs = bars[i0:]
    vw, sd = vwap_bands(cs)
    dts = day_type_series(cs, vw, sd)
    side, style = 0, None
    for j, b in enumerate(cs):
        kind, d = dts[j]
        v, s, c = vw[j], sd[j], b["c"]
        if side:
            if style == "trend":
                if (side > 0 and c < v) or (side < 0 and c > v):
                    side = 0
            elif (side > 0 and (c >= v or c < v - 2.5 * s)) or (side < 0 and (c <= v or c > v + 2.5 * s)):
                side = 0
        if not side and j > 0 and s > 0:
            pc, pv, ps = cs[j - 1]["c"], vw[j - 1], sd[j - 1]
            if kind == "trend":
                if d > 0 and b["l"] <= v + s and c > v + s:
                    side, style = 1, "trend"
                elif d < 0 and b["h"] >= v - s and c < v - s:
                    side, style = -1, "trend"
            elif kind in ("range", "trend-broken"):
                if pc > pv + 2 * ps and v < c < v + 2 * s:
                    side, style = -1, "fade"
                elif pc < pv - 2 * ps and v - 2 * s < c < v:
                    side, style = 1, "fade"
        out[i0 + j] = side
        sty[i0 + j] = style if side else None
    return out, sty


def trap_series(bars, until=TRAP_UNTIL):
    """Failed break of yesterday's high / low: a bar trades above yesterday's high but closes back below it (and
    above VWAP) -> short; the mirror at the low -> long. Once per side per day, before `until`. Flat on a close
    beyond the trap bar's extreme (stop) or at VWAP (target)."""
    out = [0] * len(bars)
    i0 = today_start(bars)
    ph, pl = prev_day_hl(bars, i0) if i0 else (None, None)
    if ph is None:
        return out
    cs = bars[i0:]
    vw, _sd = vwap_bands(cs)
    side, ext, fired = 0, None, set()
    for j, b in enumerate(cs):
        c, v = b["c"], vw[j]
        if (side < 0 and (c > ext or c <= v)) or (side > 0 and (c < ext or c >= v)):
            side = 0
        if not side and ist_min(b["t"]) < until:
            if -1 not in fired and b["h"] > ph and v < c < ph:
                side, ext = -1, b["h"]; fired.add(-1)
            elif 1 not in fired and b["l"] < pl and pl < c < v:
                side, ext = 1, b["l"]; fired.add(1)
        out[i0 + j] = side
    return out


def signal_series(cfg, bars):
    """-> (wanted side per bar, style of the last bar's signal: 'trend' or 'fade')."""
    s = cfg.get("strategy") or "RIBBON"
    if s == "EMA":
        return ema_trend(bars, cfg["ema_fast"], cfg["ema_slow"]), "trend"
    if s == "ORB":
        return orb_series(bars, cfg.get("orb_min", 15), cfg.get("tf", 1)), "trend"
    if s == "VWAP":
        sides, sty = vwap_series(bars)
        return sides, (sty[-1] if sty and sty[-1] else "trend")
    if s == "TRAP":
        return trap_series(bars), "fade"
    return ttr_trend(bars), "trend"


def momentum_faded(cfg, bars, side):
    """Flat exit for the trend followers: the last close is back through the ribbon's midline (ALMA) or the slow EMA."""
    if not bars or not side:
        return False
    c = bars[-1]["c"]
    if cfg.get("strategy") == "EMA":
        ref = _ema([b["c"] for b in bars], max(cfg["ema_fast"], cfg["ema_slow"]))[-1]
    else:
        ref = _ttr(bars)[1][-1]
    return ref is not None and (c - ref) * side < 0


# ── options maths (twins of bias_eval / the dashboard) ──
def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def greeks(S, K, T, iv, call, r=0.07):
    """(delta, gamma); T in years (dashboard: dte / 365), iv as a fraction."""
    if T <= 0 or iv <= 0.01:
        return (1.0 if S > K else 0.0) - (0.0 if call else 1.0), 0.0
    sd = iv * math.sqrt(T)
    d1 = (math.log(S / K) + (r + iv * iv / 2) * T) / sd
    return (_ncdf(d1) if call else _ncdf(d1) - 1), math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) / (S * sd)


def gex_total(d):
    """Sign twin of the dashboard's _raw_computeGex on a chain payload: + dealers long gamma (pinning),
    - dealers short gamma (moves extend). None without a usable chain."""
    spot = (d or {}).get("underlying_value")
    rows = (d or {}).get("strikes") or []
    if not spot or not rows:
        return None
    T, tot = max(1 / 1440, d.get("dte") or 1) / 365, 0.0
    for r in rows:
        g_ce = greeks(spot, r["strike"], T, (r.get("ce_iv") or 0) / 100, True)[1]
        g_pe = greeks(spot, r["strike"], T, (r.get("pe_iv") or 0) / 100, False)[1]
        tot += g_ce * (r.get("ce_oi") or 0) - g_pe * (r.get("pe_oi") or 0)
    return tot


def pick_by_delta(d, typ, target):
    """Round-hundred strike of a chain payload whose |delta| is nearest `target`, or None."""
    spot = (d or {}).get("underlying_value")
    if not spot:
        return None
    T, p, best = max(1 / 1440, d.get("dte") or 1) / 365, typ.lower(), None
    for r in (d.get("strikes") or []):
        k = r.get("strike")
        if not k or k % 100 or not r.get(p + "_symbol"):
            continue
        iv = (r.get(p + "_iv") or d.get("atm_iv") or 13) / 100
        gap = abs(abs(greeks(spot, k, T, iv, typ == "CE")[0]) - target)
        if best is None or gap < best[0]:
            best = (gap, k)
    return best[1] if best else None


def grid_book(d, n=5):
    """The strike grid's rows of a chain payload as book samples: [(key, sign, total bid, total offer)];
    sign +1 for calls, -1 for puts (a PE bid is bearish). ATM +-n round-hundred strikes."""
    spot = (d or {}).get("underlying_value")
    rows = sorted((r for r in ((d or {}).get("strikes") or []) if r.get("ce_symbol") and r.get("pe_symbol") and not r["strike"] % 100),
                  key=lambda r: r["strike"])
    if not spot or not rows:
        return []
    ai = min(range(len(rows)), key=lambda i: abs(rows[i]["strike"] - spot))
    out = []
    for r in rows[max(0, ai - n): ai + n + 1]:
        out.append((r["ce_symbol"], 1, r.get("ce_tbq"), r.get("ce_tsq")))
        out.append((r["pe_symbol"], -1, r.get("pe_tbq"), r.get("pe_tsq")))
    return out


class Shift:
    """Python twin of the dashboard's _bfZ: the 2-minute change of ln(total bid / total offer), z-scored."""

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


def realised_vol(bars, tf=1, window_min=15):
    """Annualised realised vol (%) of today's last `window_min` minutes of closes, or None (too few bars)."""
    i0 = today_start(bars)
    if i0 is None:
        return None
    c = [b["c"] for b in bars[i0:]]
    n = max(3, int(round(window_min / max(1, tf))))
    if len(c) < n + 1:
        return None
    r = [math.log(c[i] / c[i - 1]) for i in range(len(c) - n, len(c)) if c[i - 1] > 0 and c[i] > 0]
    if len(r) < 3:
        return None
    mu = sum(r) / len(r)
    sd = math.sqrt(sum((x - mu) ** 2 for x in r) / (len(r) - 1))
    return sd * math.sqrt(252 * 375 / max(1, tf)) * 100


def atm_iv(d):
    """ATM implied vol (%) of a chain payload, or None."""
    if not d:
        return None
    v = d.get("atm_iv")
    if v:
        return float(v)
    spot, rows = d.get("underlying_value"), d.get("strikes") or []
    if not spot or not rows:
        return None
    r = min(rows, key=lambda r: abs(r["strike"] - spot))
    ivs = [x for x in (r.get("ce_iv"), r.get("pe_iv")) if x]
    return sum(ivs) / len(ivs) if ivs else None


def _grid_rows(d, n=5):
    spot = (d or {}).get("underlying_value")
    rows = sorted((r for r in ((d or {}).get("strikes") or []) if r.get("ce_symbol") and r.get("pe_symbol") and not r["strike"] % 100),
                  key=lambda r: r["strike"])
    if not spot or not rows:
        return [], None
    ai = min(range(len(rows)), key=lambda i: abs(rows[i]["strike"] - spot))
    return rows[max(0, ai - n): ai + n + 1], rows[ai]["strike"]


def writer_read(d, n=5):
    """Option writers over the last 15 minutes, delta-weighted, ATM +-n: + puts being written (bullish),
    - calls being written (bearish), in [-1, 1]. None without 15-minute OI changes."""
    rows, _atm = _grid_rows(d, n)
    if not rows:
        return None
    spot, T = d["underlying_value"], max(1 / 1440, d.get("dte") or 1) / 365
    num = den = 0.0
    for r in rows:
        for p, sg in (("ce", 1), ("pe", -1)):
            o15 = r.get(p + "_oi_15m")
            if not o15:
                continue
            ad = abs(greeks(spot, r["strike"], T, (r.get(p + "_iv") or d.get("atm_iv") or 13) / 100, sg > 0)[0])
            num += -sg * ad * o15
            den += ad * abs(o15)
    return num / den if den else None


def flow_imbalance(f):
    """Futures taker flow over 5 minutes: (aggressive buys - sells) / total, or None."""
    if not f:
        return None
    b, s_ = f.get("b5") or 0, f.get("s5") or 0
    return (b - s_) / (b + s_) if b + s_ > 0 else None


def pin_strike(d, n=3):
    """The round-hundred strike within ATM +-n strikes holding the most open interest (calls + puts): where
    long-gamma dealers pin the index."""
    rows, atm = _grid_rows(d, n)
    if not rows:
        return None
    return max(rows, key=lambda r: ((r.get("ce_oi") or 0) + (r.get("pe_oi") or 0), -abs(r["strike"] - atm)))["strike"]


def option_delta(d, strike, typ):
    """|delta| of one contract from a chain payload (its own IV, else ATM IV), or None."""
    spot = (d or {}).get("underlying_value")
    if not spot:
        return None
    r = next((r for r in (d.get("strikes") or []) if r.get("strike") == strike), {})
    iv = (r.get(typ.lower() + "_iv") or d.get("atm_iv") or 0) / 100
    if iv <= 0.01:
        return None
    return abs(greeks(spot, strike, max(1 / 1440, d.get("dte") or 1) / 365, iv, typ == "CE")[0])


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
    pick_fn(sym, typ, off[, delta=x]) -> (inst, strike, atm) or raises ValueError (delta only in the DELTA strike mode)
    strikes_fn(sym) -> {"atm": x, "step": s, "spot": y} or None
    gex_fn(sym) -> dealer gamma total (+ long, - short) or None          (only read with the gamma filter on)
    book_fn(sym) -> [(key, sign, total bid, total offer)] grid samples    (only read with the book-shift filter on)
    chain_fn(sym) -> option-chain payload or None (memory): IV, OI writers, pin strike, deltas, PREMIUM
    flow_fn(sym) -> near-future taker flow {"b5", "s5", ...} or None      (only read with the futures-flow filter on)"""

    SYMS = ("NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY")
    RETRY_S = 3.0           # a refused entry is retried this often ...
    RETRY_FOR_S = 60.0      # ... for this long, while the trend it was for still holds

    def __init__(self, engine, bars_fn, pick_fn, strikes_fn=None, clock=time.time, gex_fn=None, book_fn=None, chain_fn=None, flow_fn=None):
        self.eng, self.bars_fn, self.pick_fn, self.strikes_fn, self.clock = engine, bars_fn, pick_fn, strikes_fn, clock
        self.gex_fn, self.book_fn, self.chain_fn, self.flow_fn = gex_fn, book_fn, chain_fn, flow_fn
        self.lock = threading.RLock()
        self.act = threading.RLock()        # one decision at a time: the loop, START and EXIT never overlap
        self.cfg = {"sym": "NIFTY", "tf": 1, "mode": "BUY", "ce_off": -300, "pe_off": 300,
                    "start": 9 * 60 + 20, "end": 15 * 60 + 15, "lots": 2, "max_trades": 5, "max_loss": 5000,
                    "strategy": "RIBBON", "ema_fast": 5, "ema_slow": 15, "orb_min": 15,
                    "gate_day": False, "gate_gex": False, "gate_book": False, "gate_vwap": False, "gate_lead": False,
                    "flat_exit": False, "time_stop": 0, "scale_out": False, "strike": "OFFSET", "delta": 0.5,
                    "gate_vol": False, "vol_ratio": 0.8, "gate_flow": False, "gate_oi": False,
                    "tm_sl_atr": 0.0, "tm_trail_atr": 0.0, "tm_be_atr": 0.0, "reentry": False, "lock_at": 0.0, "lock_give": 50.0,
                    "prem_kind": "FLY", "prem_center": "PIN", "prem_wing": 300, "prem_sl_pct": 50.0, "prem_tp_pct": 50.0, "prem_expiry": True}
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
        self.style = "trend"        # 'trend' or 'fade': what the last signal is (the filters read it)
        self.entry = None           # {"t", "px", "atr", "side", "scaled"} of the open auto trade, for the exits
        self.shift, self.book_z, self._book_t = Shift(), None, 0.0
        self.gex_last = None
        self.peak = 0.0             # best auto P&L today (profit lock)
        self.locked = False         # profit lock hit: no more entries today
        self._prem_bar = 0          # last candle PREMIUM tried to open on
        self._closing = None        # a close-everything reason still being worked (refused exits are retried each second)
        self._bars_last = None      # the closed candles of the last read
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

    def _exit_all(self, why, log=True):
        """Close every auto position, shorts first (a structure's wings keep its margin down until the shorts are out)."""
        for inst, p in sorted(self.holding(), key=lambda x: x[1].side):
            if not p.exiting:
                r = self.eng.exit(inst, 100, "auto: " + why, urgent=True, wait=True)
                if log:
                    self.note(f"exit {inst.split(':')[-1]} -> {r.get('msg')}")

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
            if strategy not in STRATEGIES:
                return {"ok": False, "msg": "strategy must be one of " + ", ".join(STRATEGIES)}
            orb_min = int(body.get("orb_min", c["orb_min"]))
            if orb_min not in (15, 30):
                return {"ok": False, "msg": "opening range must be 15 or 30 minutes"}
            flags = {k: bool(body.get(k, c[k])) for k in GATES + ("flat_exit", "scale_out")}
            time_stop = body.get("time_stop", c["time_stop"])
            if not isinstance(time_stop, int) or isinstance(time_stop, bool) or not 0 <= time_stop <= 120:
                return {"ok": False, "msg": "time stop must be 0 (off) to 120 candles"}
            strike = str(body.get("strike") or c["strike"]).upper()
            if strike not in ("OFFSET", "DELTA"):
                return {"ok": False, "msg": "strike choice must be OFFSET or DELTA"}
            delta = float(body.get("delta", c["delta"]))
            if not 0.05 <= delta <= 0.95:
                return {"ok": False, "msg": "target delta must be 0.05 to 0.95"}
            vol_ratio = float(body.get("vol_ratio", c["vol_ratio"]))
            if not 0.2 <= vol_ratio <= 3:
                return {"ok": False, "msg": "realised/implied vol ratio must be 0.2 to 3"}
            tm = {k: float(body.get(k, c[k])) for k in ("tm_sl_atr", "tm_trail_atr", "tm_be_atr")}
            if not all(0 <= v <= 10 for v in tm.values()):
                return {"ok": False, "msg": "stop / trail / breakeven must be 0 (off) to 10 ATR"}
            lock_at, lock_give = float(body.get("lock_at", c["lock_at"])), float(body.get("lock_give", c["lock_give"]))
            if lock_at < 0 or not 5 <= lock_give <= 95:
                return {"ok": False, "msg": "profit lock: amount 0 (off) or more, give-back 5-95%"}
            prem_kind, prem_center = str(body.get("prem_kind") or c["prem_kind"]).upper(), str(body.get("prem_center") or c["prem_center"]).upper()
            if prem_kind not in ("FLY", "STRADDLE") or prem_center not in ("PIN", "ATM"):
                return {"ok": False, "msg": "premium: structure FLY or STRADDLE, centre PIN or ATM"}
            prem_wing = body.get("prem_wing", c["prem_wing"])
            if not isinstance(prem_wing, int) or isinstance(prem_wing, bool) or prem_wing % 100 or not 100 <= prem_wing <= 3000:
                return {"ok": False, "msg": "premium wing must be 100-3000 points in hundreds"}
            prem_sl_pct, prem_tp_pct = float(body.get("prem_sl_pct", c["prem_sl_pct"])), float(body.get("prem_tp_pct", c["prem_tp_pct"]))
            if not (10 <= prem_sl_pct <= 300 and 10 <= prem_tp_pct <= 100):
                return {"ok": False, "msg": "premium stop 10-300% and target 10-100% of the credit"}
            flags["reentry"] = bool(body.get("reentry", c["reentry"]))
            flags["prem_expiry"] = bool(body.get("prem_expiry", c["prem_expiry"]))
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
            changed = (sym, tf, strategy, ema_fast, ema_slow, orb_min) != (c["sym"], c["tf"], c["strategy"], c["ema_fast"], c["ema_slow"], c["orb_min"])
            if sym != c["sym"]:
                self.shift, self.book_z = Shift(), None
            self.cfg = {"sym": sym, "tf": tf, "mode": mode, "ce_off": ce_off, "pe_off": pe_off, "start": start, "end": end, "lots": lots,
                        "max_trades": max_trades, "max_loss": max_loss,
                        "strategy": strategy, "ema_fast": ema_fast, "ema_slow": ema_slow, "orb_min": orb_min,
                        **flags, "time_stop": time_stop, "strike": strike, "delta": delta, "vol_ratio": vol_ratio, **tm,
                        "lock_at": lock_at, "lock_give": lock_give, "prem_kind": prem_kind, "prem_center": prem_center,
                        "prem_wing": prem_wing, "prem_sl_pct": prem_sl_pct, "prem_tp_pct": prem_tp_pct}
            if enable and not self.enabled:
                self.enabled, self.done_today, self.want = True, False, None
                self.base, self.next_fetch = None, 0.0          # re-baselined on the next read: wait for a flip
                self.note(f"Auto Trade ON - {mode} {sym} {tf}m, {lots} lot(s), {_fmt(start)}-{_fmt(end)}; {self._sname()} - " + ("waiting for its setup" if strategy == "PREMIUM" else "waiting for the next flip"))
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
        hold = sorted(hold, key=lambda x: x[1].side)                # shorts bought back before their wings are sold
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
        if self.locked:
            return {"ok": False, "msg": "profit lock hit - no more entries today"}
        if self.entries >= c["max_trades"]:
            return {"ok": False, "msg": f"daily max trades ({c['max_trades']}) reached"}
        now = self.clock()
        bars = self._bars(c, now)
        if c["strategy"] == "PREMIUM" and bars:
            with self.lock:
                self.enabled, self.done_today, self._bars_last, self.bar_t = True, False, bars, bars[-1]["t"]
            r = self._premium_open(bars, now, force=True)
            return {"ok": bool(r.get("ok")), "msg": self.msg}
        if bars is None:
            return {"ok": False, "msg": self.msg}
        if len(bars) < ALMA_LEN + 2:
            return {"ok": False, "msg": f"not enough candle history yet ({len(bars)}/{ALMA_LEN + 2} bars)"}
        series, style = signal_series(c, bars)
        trend = series[-1]
        if trend == 0:
            return {"ok": False, "msg": f"{self._sname()} has no {'direction' if c['strategy'] in CLASSIC else 'signal'} now"}
        with self.lock:
            self.bar_t, self.trend, self.base, self.style, self._bars_last = bars[-1]["t"], trend, trend, style, bars
            self.done_today = False
            self.want = {"trend": trend, "bar": bars[-1]["t"], "since": now, "next": 0.0, "n": 0, "force": True,
                         "style": style, "gate_until": now}
        self.note(f"START: {self._sname()} is {'bullish' if trend > 0 else 'bearish'} - entering now (filters skipped)")
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
                self.entries, self.entry, self.shift, self.book_z = 0, None, Shift(), None
                self.peak, self.locked, self._prem_bar, self._closing = 0.0, False, 0, None
                if self.enabled:
                    self.enabled = False
                    self.note("new trading day - Auto Trade switched off, tick it again to start")
        if not (self.enabled or self.holding()):
            return
        self._loss_guard()
        self._lock_guard()
        if self._closing:
            if self.holding():
                self._exit_all(self._closing, log=False)            # e.g. the order-rate cap refused part of it
            else:
                self._closing = None
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
                for inst, _p in sorted(self.holding(), key=lambda x: x[1].side):
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
        self._sample_book(now)
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
        self._exit_all("daily max loss", log=False)                # a refused exit is retried every second
        with self.lock:
            self.want = None

    def _lock_guard(self):
        """Day profit lock: once the auto P&L has reached lock_at, giving back lock_give % of its peak closes every
        auto position and ends the day's entries."""
        c = self.cfg
        if not c["lock_at"]:
            return
        pnl = self.pnl()
        self.peak = max(self.peak, pnl)
        if self.peak >= c["lock_at"] and pnl <= self.peak * (1 - c["lock_give"] / 100.0):
            if not self.locked:
                self.locked = True
                self.note(f"profit lock: P&L Rs {pnl:,.0f} gave back {c['lock_give']:.0f}% of the Rs {self.peak:,.0f} peak - closing, done for the day")
            with self.lock:
                self.want = None
            if self.holding():
                self._exit_all("profit lock", log=False)

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
        if c["strategy"] == "PREMIUM":
            with self.lock:
                self.bar_t, self._bars_last, self.trend = last_t, bars, 0
                if self.base is None:
                    self.base = 0
                    self.note("premium selling - waiting for its setup (range day, dealers long gamma)")
            self._premium(bars, now)
            return
        series, style = signal_series(c, bars)
        trend = series[-1]
        classic = c["strategy"] in CLASSIC
        with self.lock:
            self.bar_t, prev, self.trend, self._bars_last = last_t, self.base, trend, bars
            if prev is None:
                self.base = trend                                   # enabling never trades the current signal
                word = 'bullish' if trend > 0 else 'bearish' if trend < 0 else ('neutral' if classic else 'flat')
                self.note(f"{self._sname()} is {word} - waiting for {'it to flip' if classic else 'a fresh signal'}")
                return
            changed = trend != prev and not (trend == 0 and classic)  # a trend follower is never 'flat', only not warmed up
            if changed:
                self.base, self.style = trend, style
        if not changed:
            self._manage(bars)
            return
        if trend == 0:
            self._on_flat()
            return
        self._on_flip(trend, last_t, now)

    def _sname(self):
        c = self.cfg
        return {"EMA": f"EMA {c['ema_fast']}/{c['ema_slow']} cross", "ORB": f"ORB {c['orb_min']}m", "VWAP": "VWAP sigma",
                "TRAP": "PDH/PDL trap", "PREMIUM": f"premium {c['prem_kind'].lower()}"}.get(c["strategy"], "ribbon")

    def _wanted(self, trend):
        mode = self.cfg["mode"]
        if mode == "BUY":
            return ("CE", "BUY") if trend > 0 else ("PE", "BUY")
        return ("CE", "SELL") if trend < 0 else ("PE", "SELL")

    def _on_flip(self, trend, bar_t, now):
        name = "bullish" if trend > 0 else "bearish"
        self.note(f"{self._sname()} turned {name}")
        self._exit_all(f"{self._sname()} turned {name}")
        mins, _wd = self._clk()
        if self.loss_hit() or self.locked:
            return
        if mins < self.cfg["start"]:
            self.note(f"before the start time {_fmt(self.cfg['start'])}: no entry")
            return
        with self.lock:
            self.want = {"trend": trend, "bar": bar_t, "since": now, "next": 0.0, "n": 0, "style": self.style,
                         "gate_until": bar_t + (1 + CONFIRM_BARS) * self.cfg["tf"] * 60}

    def _on_flat(self):
        """A setup strategy's signal went back to 0: close the auto position, wait for the next signal."""
        with self.lock:
            self.want = None
        hold = self.holding()
        self.note(f"{self._sname()} signal over" + (" - exiting" if hold else ""))
        self._exit_all(f"{self._sname()} exit")
        self.entry = None

    # ── exits on the open auto trade (on each new closed candle) ──
    def _manage(self, bars):
        e, c = self.entry, self.cfg
        hold = self.holding()
        if e and not hold:
            self._maybe_reenter(bars)
            return
        if not e or not hold:
            return
        after = [b for b in bars if b["t"] > e["t"]]
        if not after:
            return
        sd, atr = e["side"], e["atr"] or 0.0
        mfe = max((b["h"] - e["px"]) if sd > 0 else (e["px"] - b["l"]) for b in after)
        why = None
        if c["flat_exit"] and c["strategy"] in CLASSIC and momentum_faded(c, bars, sd):
            why = "flat exit: momentum faded (close back through the " + ("slow EMA" if c["strategy"] == "EMA" else "ribbon midline") + ")"
        elif c["time_stop"] and len(after) >= c["time_stop"] and atr and mfe < atr:
            why = f"time stop: no 1-ATR move in {c['time_stop']} candles"
        if why:
            self.note(why)
            self._exit_all(why.split(":")[0])
            self.entry = None
            return
        if c["scale_out"] and not e["scaled"] and atr and mfe >= atr:
            e["scaled"] = True
            for inst, p in hold:
                lot = max(1, int(p.meta.get("lot") or 1))
                if p.qty // lot >= 2 and not p.exiting:
                    r = self.eng.exit(inst, 50, "auto: scale-out at 1 ATR", urgent=True, wait=True)
                    self.note(f"scale-out: half of {inst.split(':')[-1]} at 1 ATR in favour -> {r.get('msg')}")

    def _maybe_reenter(self, bars):
        """One re-entry per signal after the engine stopped the trade out, while the signal still points the same
        way and a candle closing after the stop closed with it (above the previous close for a long)."""
        e, c = self.entry, self.cfg
        if not (c["reentry"] and e and e.get("inst") and not e.get("re") and e["side"] == self.base and self.base) or self.want:
            return
        with self.eng.lock:
            last = next((t for t in reversed(self.eng.trades) if t.get("src") == SRC and t.get("inst") == e["inst"]), None)
        if not last or not str(last.get("why") or "").startswith("stop"):
            self.entry = None                                       # closed some other way: no re-entry
            return
        b = bars[-1]
        if len(bars) < 2 or b["t"] + c["tf"] * 60 <= last["t"] or (b["c"] - bars[-2]["c"]) * e["side"] <= 0:
            return
        e["re"] = True
        if self.loss_hit() or self.locked or self.done_today:
            return
        self.note(f"re-entry: stopped out, but the {self._sname()} still points {'up' if e['side'] > 0 else 'down'} and the last candle agrees")
        with self.lock:
            self.want = {"trend": e["side"], "bar": b["t"], "since": self.clock(), "next": 0.0, "n": 0, "style": self.style,
                         "gate_until": b["t"] + (1 + CONFIRM_BARS) * c["tf"] * 60, "re": True}

    def _tm_exits(self, typ, strike):
        """Stop / trail / breakeven in option points: index ATRs x the contract's delta (0.5 without a chain)."""
        c = self.cfg
        if not (c["tm_sl_atr"] or c["tm_trail_atr"] or c["tm_be_atr"]):
            return {}
        bl = self._bars_last or []
        atr = atr_series(bl)[-1] if bl else None
        if not atr:
            return {}
        dl = None
        try:
            dl = option_delta(self.chain_fn(c["sym"]) if self.chain_fn else None, strike, typ)
        except Exception:  # noqa: BLE001
            dl = None
        pts = atr * (dl or 0.5)
        out = {}
        for k, ek in (("tm_sl_atr", "sl_pts"), ("tm_trail_atr", "trail_pts"), ("tm_be_atr", "be_pts")):
            if c[k]:
                out[ek] = round(max(0.5, c[k] * pts), 2)
        return out

    # ── PREMIUM: an iron fly / straddle sold at the pin ──
    def _chain(self):
        try:
            return self.chain_fn(self.cfg["sym"]) if self.chain_fn else None
        except Exception:  # noqa: BLE001
            return None

    def _premium_setup(self, bars, d, mins):
        """-> (ok, why not, centre strike, expiry pin?)."""
        c = self.cfg
        if not d:
            return False, "no option chain", None, False
        g = gex_total(d)
        self.gex_last = g
        pin_mode = c["prem_expiry"] and (d.get("dte") if d.get("dte") is not None else 9) < 1 and mins >= PIN_AFTER
        if g is None or g < 0:
            return False, "dealers short gamma: moves extend", None, False
        if not pin_mode:
            i0 = today_start(bars)
            kind = day_type_series(bars[i0:])[-1][0] if bars else "unknown"
            if kind not in ("range", "trend-broken"):
                return False, "day type " + ("still forming (reads at 9:45)" if kind == "forming" else f"is {kind}: premium needs a range day"), None, False
        if c["gate_vol"]:
            iv, rv = atm_iv(d), realised_vol(bars, c["tf"])
            if not iv or rv is None:
                return False, "no IV / realised vol reading", None, False
            if rv / iv >= c["vol_ratio"]:
                return False, f"realised vol {rv:.1f}% is {rv / iv:.2f}x IV {iv:.1f}%: premium not rich enough", None, False
        if c["prem_center"] == "PIN" or pin_mode:
            k = pin_strike(d)
        else:
            rows, k = _grid_rows(d, 0)
        if k is None:
            return False, "no strike to centre on", None, False
        if abs(bars[-1]["c"] - k) >= c["prem_wing"] / 4.0:          # would open half-way to its own tent exit
            return False, f"centre {int(k)} is too far from the index {bars[-1]['c']:.0f}", None, False
        return True, "", k, pin_mode

    def _premium(self, bars, now):
        c, hold = self.cfg, self.holding()
        if hold:
            e = self.entry
            if e and e.get("center") is not None and abs(bars[-1]["c"] - e["center"]) >= c["prem_wing"] / 2.0:
                self.note(f"premium: index {bars[-1]['c']:.0f} left the tent ({e['center']:.0f} +- {c['prem_wing'] / 2:.0f}) - closing")
                self._closing = "premium: index left the tent"
                self._exit_all(self._closing)
                self.entry = None
            return
        self._premium_open(bars, now)

    def _premium_open(self, bars, now, force=False):
        c = self.cfg
        mins, _wd = self._clk()
        if self.holding():
            return {"ok": False}
        if not force and (self.done_today or self.loss_hit() or self.locked or self.entries >= c["max_trades"]
                          or mins < c["start"] or mins >= c["end"] or self._prem_bar == bars[-1]["t"]):
            return {"ok": False}
        if not force:
            with self.eng.lock:
                last = max((t["t"] for t in self.eng.trades if t.get("src") == SRC), default=0)
            if last and self.clock() - last < PREM_COOL_MIN * 60:
                self.msg = f"premium: waiting - {PREM_COOL_MIN} min pause after the last structure closed"
                return {"ok": False}
        self._prem_bar = bars[-1]["t"]
        d = self._chain()
        if force:                                                   # START: sell now, whatever the setup says
            pin_mode = False
            k = (pin_strike(d) if c["prem_center"] == "PIN" else _grid_rows(d, 0)[1]) if d else None
        else:
            ok, why, k, pin_mode = self._premium_setup(bars, d, mins)
            if not ok:
                self.msg = f"premium: waiting - {why}"
                return {"ok": False}
        if k is None:
            self.msg = "premium: no option chain to centre the structure on"
            return {"ok": False}
        try:
            _i, _k, atm = self.pick_fn(c["sym"], "CE", 0)
            off = k - atm
            legs = [(self.pick_fn(c["sym"], "CE", off), "SELL"), (self.pick_fn(c["sym"], "PE", off), "SELL")]
            if c["prem_kind"] == "FLY":
                legs += [(self.pick_fn(c["sym"], "CE", off + c["prem_wing"]), "BUY"), (self.pick_fn(c["sym"], "PE", off - c["prem_wing"]), "BUY")]
        except ValueError as e:
            self.msg = f"premium: cannot pick the legs: {e}"
            return {"ok": False}
        insts = [x[0][0] for x in legs]
        with self.eng.lock:
            other = [i for i in insts if i in self.eng.pos]
        if other:
            self.msg = f"premium: you already hold {other[0].split(':')[-1]} - not opening a structure on top of it"
            return {"ok": False}
        kind = "iron fly" if c["prem_kind"] == "FLY" else "short straddle"
        r = self.eng.place({"legs": [{"inst": x[0], "side": sd} for x, sd in legs], "lots": c["lots"], "exits": {}, "tag": "auto",
                            "kind": kind}, req_id=f"auto{self.day}{bars[-1]['t']}P{int(now)}", src=SRC)
        if not r.get("ok"):
            self.msg = f"premium: {kind} refused: {r.get('msg')}"
            return r
        self.entries += 1
        with self.eng.lock:
            ps = [self.eng.pos[i] for i in r.get("legs") or [] if i in self.eng.pos]
            credit = sum(-p.side * p.avg * p.qty for p in ps)
        sl, tp = credit * c["prem_sl_pct"] / 100.0, credit * c["prem_tp_pct"] / 100.0
        if credit <= 0:                                             # nothing to stop out against: never leave it unguarded
            self.note(f"premium: {kind} filled for no net credit (Rs {credit:,.0f}) - closing it")
            self._closing = "premium: no net credit"
            self._exit_all(self._closing)
            return {"ok": False, "msg": "no net credit"}
        if r.get("group"):
            self.eng.modify_group(r["group"], sl=sl, tp=tp)
        self.entry = {"t": bars[-1]["t"], "px": bars[-1]["c"], "atr": atr_series(bars)[-1], "side": 0, "scaled": True,
                      "center": float(k), "grp": r.get("group")}
        self.note(f"premium: sold {c['lots']} lot(s) {kind} at {int(k)}" + (" (expiry pin)" if pin_mode else "")
                  + (f" +-{c['prem_wing']}" if c["prem_kind"] == "FLY" else "")
                  + f", credit Rs {credit:,.0f}; stop -Rs {sl:,.0f}, target Rs {tp:,.0f}")
        return r

    # ── filters on an entry ──
    def _sample_book(self, now):
        """Feed the book-shift twin every 5 s while its filter is on (it needs ~2 minutes of history)."""
        if not (self.cfg["gate_book"] and self.book_fn) or now - self._book_t < 5:
            return
        self._book_t = now
        try:
            rows = self.book_fn(self.cfg["sym"]) or []
        except Exception:  # noqa: BLE001
            rows = []
        zs = []
        for key, sg, b, o in rows:
            z = self.shift.z(key, b, o, now)
            if z is not None:
                zs.append(sg * z)
        self.book_z = sum(zs) / len(zs) if zs else None

    def _gates(self, side, style, now):
        """-> (ok, why not). Only the filters switched on are read."""
        c, bars = self.cfg, self._bars_last or []
        i0 = today_start(bars)
        cs = bars[i0:] if i0 is not None else []
        if c["gate_day"] and c["strategy"] != "VWAP":            # the VWAP playbook reads the day type itself
            kind, d = day_type_series(cs)[-1] if cs else ("unknown", 0)
            if style == "trend":
                if kind != "trend":
                    return False, "day type " + ("still forming (reads at 9:45)" if kind == "forming" else f"is {kind}, not a trend day")
                if d != side:
                    return False, f"against the {'up' if d > 0 else 'down'} trend day"
            elif kind not in ("range", "trend-broken"):
                return False, f"day type is {kind}: a fade needs a range day"
        if c["gate_vwap"] and style == "trend" and cs:
            vw = vwap_bands(cs)[0][-1]
            if (cs[-1]["c"] - vw) * side <= 0:
                return False, f"close is {'below' if side > 0 else 'above'} VWAP"
        if c["gate_gex"]:
            try:
                g = self.gex_fn(c["sym"]) if self.gex_fn else None
            except Exception:  # noqa: BLE001
                g = None
            self.gex_last = g
            if g is None:
                return False, "no dealer-gamma reading (option chain not loaded)"
            if style == "trend" and g >= 0:
                return False, "dealers long gamma: moves get pinned"
            if style == "fade" and g < 0:
                return False, "dealers short gamma: moves extend"
        if c["gate_book"]:
            z = self.book_z
            if z is None:
                return False, "book shift not ready (needs ~2 min of order-book ticks)"
            if z * side < BF["agg"]:
                return False, f"book shift {z:+.2f} does not confirm (needs {'+' if side > 0 else '-'}{BF['agg']})"
        if c["gate_vol"] or c["gate_oi"]:
            d = self._chain()
            if c["gate_vol"]:
                iv, rv = atm_iv(d), realised_vol(bars, c["tf"])
                if not iv or rv is None:
                    return False, "no IV / realised vol reading"
                ratio = rv / iv
                if c["mode"] == "SELL" and ratio >= c["vol_ratio"]:
                    return False, f"realised vol {ratio:.2f}x IV: too lively to sell premium (needs < {c['vol_ratio']})"
                if c["mode"] == "BUY" and ratio < c["vol_ratio"]:
                    return False, f"realised vol {ratio:.2f}x IV: too quiet to buy premium (needs >= {c['vol_ratio']})"
            if c["gate_oi"]:
                w = writer_read(d)
                if w is None:
                    return False, "no 15-min OI change to read the writers"
                if w * side < OI_MIN:
                    return False, f"option writers {w:+.2f} do not confirm ({'put' if side > 0 else 'call'} writing needed)"
        if c["gate_flow"] and style == "trend":
            try:
                imb = flow_imbalance(self.flow_fn(c["sym"]) if self.flow_fn else None)
            except Exception:  # noqa: BLE001
                imb = None
            if imb is None:
                return False, "no futures taker flow yet"
            if imb * side < FLOW_MIN:
                return False, f"futures flow {imb:+.2f} does not confirm"
        if c["gate_lead"] and style == "trend":
            other = LEAD.get(c["sym"], "NIFTY")
            try:
                ob = closed_bars(self.bars_fn(other, c["tf"]) or [], c["tf"], now)
            except Exception:  # noqa: BLE001
                ob = []
            if len(ob) < ALMA_LEN + 2:
                return False, f"no {other} candles to confirm with"
            o_side = (ema_trend(ob, c["ema_fast"], c["ema_slow"]) if c["strategy"] == "EMA" else ttr_trend(ob))[-1]
            if o_side != side:
                return False, f"{other} {'ribbon' if c['strategy'] != 'EMA' else 'EMA cross'} does not agree"
        return True, ""

    def _try_want(self, now, mins):
        w = self.want
        if not w or now < w["next"]:
            return
        if self.loss_hit() or self.locked or w["trend"] != self.base or mins < self.cfg["start"] or mins >= self.cfg["end"] or now - w["since"] > self.RETRY_FOR_S:
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
        if not w.get("force"):
            ok, why = self._gates(w["trend"], w.get("style") or "trend", now)
            if not ok:
                if now >= w["gate_until"]:
                    self.note(f"entry skipped: {why}")
                    self.want = None
                else:
                    self.msg = f"waiting for confirmation: {why}"
                    w["since"] = now                                # the 60 s refusal window starts once the filters pass
                return
        typ, side = self._wanted(w["trend"])
        off = c["ce_off"] if typ == "CE" else c["pe_off"]
        try:
            if c["strike"] == "DELTA":
                inst, strike, atm = self.pick_fn(c["sym"], typ, off, delta=c["delta"])
            else:
                inst, strike, atm = self.pick_fn(c["sym"], typ, off)
        except ValueError as e:
            self.msg = f"cannot pick a {typ}: {e}"
            return
        bare = inst.split(":")[-1]
        with self.eng.lock:
            other = [i for i, p in self.eng.pos.items() if i.split(":")[-1] == bare and p.src != SRC]
        if other:                                                   # a fill would merge into it and auto would lose track of its own lots
            self.msg = f"you already hold {bare} outside Auto Trade - not adding to it"
            return
        r = self.eng.place({"inst": inst, "side": side, "lots": c["lots"], "exits": self._tm_exits(typ, strike), "tag": "auto"},
                           req_id=f"auto{self.day}{w['bar']}{typ}{w['n']}", src=SRC)
        if r.get("dup"):                                            # the same request answered twice: already counted
            self.want = None
            return
        if r.get("ok"):
            self.want = None
            self.entries += 1
            bl = self._bars_last or []
            self.entry = {"t": bl[-1]["t"], "px": bl[-1]["c"], "atr": atr_series(bl)[-1], "side": w["trend"], "scaled": False,
                          "inst": inst, "re": bool(w.get("re"))} if bl else None
            self.note(f"{side} {c['lots']} lot(s) {c['sym']} {int(strike)} {typ} (ATM {int(atm)}) -> {r.get('msg')}")
        else:
            w["n"] += 1                                             # a refusal is cached 15 s under its req_id: use a fresh one
            self.msg = f"{side} {typ} refused: {r.get('msg')}"

    # ── what the tab shows ──
    def _ctx(self):
        """Readings behind the filters, for the tab (memory only: no broker call)."""
        bl = self._bars_last or []
        i0 = today_start(bl)
        dt = day_type_series(bl[i0:])[-1] if bl else ("unknown", 0)
        return {"day": dt[0], "day_dir": dt[1], "book": None if self.book_z is None else round(self.book_z, 2),
                "gex": None if self.gex_last is None else (1 if self.gex_last >= 0 else -1), "style": self.style,
                "peak": round(self.peak, 2), "locked": self.locked}

    def state(self):
        with self.lock:
            c = self.cfg
            hold = [{"inst": i, "side": "long" if p.side > 0 else "short", "typ": p.meta.get("typ"), "strike": p.meta.get("strike"),
                     "lots": p.qty // max(1, int(p.meta.get("lot") or 1))} for i, p in self.holding()]
            out = {"enabled": self.enabled, "cfg": {**c, "start": _fmt(c["start"]), "end": _fmt(c["end"])},
                   "trend": self.trend, "waiting_flip": self.enabled and not hold and self.want is None and not self.done_today,
                   "holding": hold, "entries": self.entries, "pnl": round(self.pnl(), 2), "loss_hit": self.loss_hit(),
                   "pending_entry": bool(self.want), "done_today": self.done_today,
                   "ctx": self._ctx(),
                   "msg": self.msg, "events": list(self.events[-12:])}
        try:
            info = self.strikes_fn(c["sym"]) if self.strikes_fn else None
        except Exception:
            info = None
        out["strikes"] = info
        return out
