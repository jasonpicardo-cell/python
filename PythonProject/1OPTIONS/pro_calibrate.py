"""Calibrate the Pro guide (guide/nifty_pro_guide.html) on your own recorded data.

    python3 pro_calibrate.py                 # NIFTY, writes guide/pro_calib.js
    python3 pro_calibrate.py --out x.js      # somewhere else

No broker calls. Reads only what the server already saved:
  history/NIFTY_60_*.json, warm_cache/NIFTY_*.json, ticks/NIFTY_*.csv   1-minute index bars / prices
  nse_data_cache/bhavcopy/DDMMYYYY.csv                                  daily Nifty 50 OHLC and India VIX
  replay/NIFTY_<day>.jsonl                                              ~1-minute chain snapshots (bid/ask/OI/volume, futures, VIX)
  participant_oi/<day>.json, zerodha_cache/oi_close_<day>.json          participant OI, close OI per contract
  trade_log/<day>.jsonl                                                 fills (paper or live)
  guide/events.txt (optional)                                           "YYYY-MM-DD label" per line, for event attribution

Writes window.PRO_CALIB = {...} so the guide works from file:// without a server. Every study records its sample
size; the guide marks small samples as low confidence. Studies that lack data are reported as such, never faked.

Part 1 (measured market facts): variance clock (intraday u(m), overnight and weekend weights), realised vol by weekday,
variance-elapsed ratio, calendar-clock IV artifact, spot-vol beta vs skew slope, VRP (VIX vs later RV), HAR-RV fit,
risk-neutral vs empirical tails, book-shift (_bfZ) decay by horizon, aggressor-classified OI vs naive OI, fill markouts,
expiry pinning, participant-OI gamma ownership.
Part 2 (surface): synthetic forward, OTM Black-76 IVs, raw-SVI fit with arbitrage check, SABR (beta=1), Merton jump
diffusion, Bakshi-Kapadia-Madan implied skew/kurtosis, 25-delta RR/BF, event variance from weekly vs VIX.
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IST = timezone(timedelta(hours=5, minutes=30))
SESSION = 375                     # minutes 09:15-15:30
WINDOW = 30                       # settlement average 15:00-15:30
R = 0.065
YEAR_S = 365 * 86400


# ----------------------------------------------------------------------------- maths
def ncdf(x):
    """Normal CDF, vectorised (Abramowitz-Stegun 7.1.26 erf, |error| < 1.5e-7)."""
    x = np.asarray(x, dtype=float)
    z = np.abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-z * z)
    return 0.5 * (1 + np.sign(x) * y)


def npdf(x):
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / math.sqrt(2 * math.pi)


def black(F, K, T, vol, call):
    """Undiscounted Black-76 price. call: bool or bool array."""
    F, K, vol = np.asarray(F, float), np.asarray(K, float), np.maximum(np.asarray(vol, float), 1e-8)
    s = vol * math.sqrt(max(T, 1e-12))
    d1 = (np.log(F / K) + 0.5 * s * s) / s
    d2 = d1 - s
    c = F * ncdf(d1) - K * ncdf(d2)
    return np.where(call, c, c - F + K)


def black_iv(price, F, K, T, call, lo=0.005, hi=3.0, it=70):
    """Vectorised bisection for the Black-76 IV of undiscounted prices. NaN where the price is outside bounds."""
    price, K = np.asarray(price, float), np.asarray(K, float)
    call = np.broadcast_to(np.asarray(call), K.shape)
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    a, b = np.full(K.shape, lo), np.full(K.shape, hi)
    for _ in range(it):
        m = 0.5 * (a + b)
        hi_side = black(F, K, T, m, call) > price
        b, a = np.where(hi_side, m, b), np.where(hi_side, a, m)
    iv = 0.5 * (a + b)
    bad = (price <= intrinsic + 1e-9) | (price >= np.where(call, F, K)) | (iv <= lo * 1.01) | (iv >= hi * 0.99)
    return np.where(bad, np.nan, iv)


def nelder_mead(f, x0, step, iters=1500, tol=1e-12):
    n = len(x0)
    pts = [np.array(x0, float)] + [np.array(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts, vals = [pts[i] for i in o], [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol * (1 + abs(vals[0])):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


# ----------------------------------------------------------------------------- smile models
def svi_w(p, k):
    a, b, rho, m, s = p
    k = np.asarray(k, float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def svi_g(p, k):
    """Gatheral's butterfly-arbitrage density function; must be >= 0."""
    a, b, rho, m, s = p
    k = np.asarray(k, float)
    w = np.maximum(svi_w(p, k), 1e-12)
    r = np.sqrt((k - m) ** 2 + s * s)
    w1 = b * (rho + (k - m) / r)
    w2 = b * s * s / r ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def fit_svi(k, iv, T, wt=None):
    k, iv = np.asarray(k, float), np.asarray(iv, float)
    wt = np.ones_like(k) if wt is None else np.asarray(wt, float)
    w0 = float(np.interp(0, k, iv ** 2 * T)) if k.min() < 0 < k.max() else float(np.median(iv) ** 2 * T)
    ks = max(math.sqrt(w0), 1e-3)                                     # natural moneyness scale

    def unpack(x):
        return (x[0] * w0, abs(x[1]) * w0 / ks, math.tanh(x[2]), x[3] * ks, abs(x[4]) * ks + 1e-6)

    def obj(x):
        p = unpack(x)
        w = svi_w(p, k)
        if p[0] + p[1] * p[4] * math.sqrt(1 - p[2] ** 2) < 0 or np.any(w <= 0):
            return 1e6
        e = np.sqrt(w / T) - iv
        return float(np.sum(wt * e * e))
    best = None
    for rho0 in (-0.6, -0.2):
        for m0 in (0.0, 0.5):
            for s0 in (0.3, 1.0):
                x, v = nelder_mead(obj, [0.6, 0.3, math.atanh(rho0), m0, s0], [0.2, 0.2, 0.3, 0.3, 0.3], iters=2500)
                if best is None or v < best[1]:
                    best = (x, v)
    p = unpack(best[0])
    rmse = math.sqrt(best[1] / max(np.sum(wt), 1e-12))
    return [float(v) for v in p], rmse


def sabr_iv(F, K, T, alpha, rho, nu):
    """Hagan lognormal SABR with beta = 1."""
    K = np.asarray(K, float)
    z = nu / alpha * np.log(F / K)
    xz = np.log((np.sqrt(1 - 2 * rho * z + z * z) + z - rho) / (1 - rho))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.abs(z) < 1e-8, 1.0, z / xz)
    return alpha * ratio * (1 + (rho * nu * alpha / 4 + (2 - 3 * rho * rho) * nu * nu / 24) * T)


def fit_sabr(F, K, iv, T):
    a0 = float(np.interp(F, K, iv)) if K.min() < F < K.max() else float(np.median(iv))

    def unpack(x):
        return a0 * math.exp(x[0]), math.tanh(x[1]), math.exp(x[2])

    def obj(x):
        al, rho, nu = unpack(x)
        m = sabr_iv(F, K, T, al, rho, nu)
        if not np.all(np.isfinite(m)):
            return 1e6
        return float(np.sum((m - iv) ** 2))
    best = None
    for nu0 in (0.5, 2.0, 6.0):
        x, v = nelder_mead(obj, [0.0, math.atanh(-0.5), math.log(nu0)], [0.1, 0.3, 0.4])
        if best is None or v < best[1]:
            best = (x, v)
    al, rho, nu = unpack(best[0])
    return {"alpha": al, "rho": rho, "nu": nu}, math.sqrt(best[1] / len(K))


def merton_terms(T, sig, lam, mj, dj, nmax=40):
    kbar = math.exp(mj + dj * dj / 2) - 1
    out, p, n = [], math.exp(-lam * T), 0
    tot = 0.0
    while n <= nmax:
        mean = -lam * kbar * T - sig * sig * T / 2 + n * mj
        var = sig * sig * T + n * dj * dj
        out.append((p, mean, var))
        tot += p
        if tot > 1 - 1e-10:
            break
        n += 1
        p *= lam * T / n
    return out


def merton_price(F, K, T, sig, lam, mj, dj, call):
    K = np.asarray(K, float)
    px = np.zeros_like(K)
    for p, mean, var in merton_terms(T, sig, lam, mj, dj):
        sd = math.sqrt(var)
        Fn = F * math.exp(mean + var / 2)
        px = px + p * black(Fn, K, 1.0, sd, call)                   # total vol sd over "T = 1"
    return px


def merton_cdf(x, T, sig, lam, mj, dj):
    """P(ln(F_T/F) < x)."""
    x = np.asarray(x, float)
    return sum(p * ncdf((x - mean) / math.sqrt(var)) for p, mean, var in merton_terms(T, sig, lam, mj, dj))


def fit_merton(F, K, iv, T):
    call = K >= F
    mkt = black(F, K, T, iv, call)
    vega = np.maximum(F * npdf((np.log(F / K) + iv * iv * T / 2) / (iv * math.sqrt(T))) * math.sqrt(T), 1e-6)
    a0 = float(np.nanmin(iv))

    def unpack(x):
        return a0 * math.exp(x[0]), 30 / (1 + math.exp(-x[1])), -0.2 + 0.3 / (1 + math.exp(-x[2])), 0.002 + 0.15 / (1 + math.exp(-x[3]))

    def obj(x):
        s, lam, mj, dj = unpack(x)
        e = (merton_price(F, K, T, s, lam, mj, dj, call) - mkt) / vega
        return float(np.sum(e * e)) + 1e-6 * (lam / 30) ** 2       # tiny pull toward fewer jumps
    best = None
    for l0, m0 in ((-1.0, -0.5), (0.5, -1.0), (-2.0, 0.0)):
        x, v = nelder_mead(obj, [-0.1, l0, m0, -1.5], [0.15, 0.6, 0.6, 0.6], iters=1200)
        if best is None or v < best[1]:
            best = (x, v)
    s, lam, mj, dj = unpack(best[0])
    return {"sigma": s, "lam": lam, "muJ": mj, "deltaJ": dj}, math.sqrt(best[1] / len(K))


def bkm(F, T, ivf, n=1201, span=8.0):
    """Risk-neutral variance, skewness, kurtosis of ln(F_T/F) from an OTM strip (Bakshi-Kapadia-Madan), forward measure."""
    sd = max(float(ivf(0.0)) * math.sqrt(T), 1e-4)
    k = np.linspace(-span * sd, span * sd * 0.75, n)
    K = F * np.exp(k)
    v = np.array([ivf(x) for x in k])
    call = K >= F
    P = black(F, K, T, v, call)
    lk = np.log(K / F)
    Vw = np.where(call, 2 * (1 - lk), 2 * (1 + (-lk))) / K ** 2
    Ww = np.where(call, 6 * lk - 3 * lk ** 2, -(6 * (-lk) + 3 * lk ** 2)) / K ** 2
    Xw = np.where(call, 12 * lk ** 2 - 4 * lk ** 3, 12 * lk ** 2 + 4 * (-lk) ** 3) / K ** 2
    V, W, X = (float(np.trapezoid(w * P, K)) for w in (Vw, Ww, Xw))
    mu = -V / 2 - W / 6 - X / 24
    var = V - mu * mu
    return {"var": var, "skew": (W - 3 * mu * V + 2 * mu ** 3) / var ** 1.5,
            "kurt": (X - 4 * mu * W + 6 * mu * mu * V - 3 * mu ** 4) / var ** 2}


# ----------------------------------------------------------------------------- loaders
def _ist(ts):
    return datetime.fromtimestamp(ts, IST)


def load_minutes(root, sym="NIFTY"):
    """{date: np.array of 375 closes (NaN where missing), plus 'open'} from every 1-minute source, bars preferred."""
    days = {}

    def put(ts, o, c, pri):
        t = _ist(ts)
        m = (t.hour * 60 + t.minute) - (9 * 60 + 15)
        if not 0 <= m < SESSION:
            return
        d = days.setdefault(t.date(), {"c": np.full(SESSION, np.nan), "o": np.full(SESSION, np.nan), "p": np.zeros(SESSION)})
        if pri >= d["p"][m]:
            d["c"][m], d["o"][m], d["p"][m] = c, o, pri
    for f in glob.glob(os.path.join(root, "ticks", f"{sym}_*.csv")):
        for line in open(f):
            p = line.strip().split(",")
            if len(p) >= 2:
                try:
                    put(int(float(p[0])), float(p[1]), float(p[1]), 1)
                except ValueError:
                    pass
    try:                                                   # the permanent archive (nse_archive.py): sessions the caches no longer hold
        import nse_archive
        for d, bars in nse_archive.read_minutes(root, sym).items():
            for t, (o, h, l, c) in bars.items():
                put(t, o, c, 1.5)
    except Exception:  # noqa: BLE001
        pass
    for f in glob.glob(os.path.join(root, "history", f"{sym}_*.json")):
        try:
            for b in json.load(open(f)):
                put(b["t"], b["o"], b["c"], 2)
        except Exception:
            pass
    for f in glob.glob(os.path.join(root, "warm_cache", f"{sym}_*.json")):
        try:
            for b in json.load(open(f)).get("bars", []):
                put(b["t"], b["o"], b["c"], 3)
        except Exception:
            pass
    return {d: v for d, v in sorted(days.items()) if np.isfinite(v["c"]).sum() >= 300}


def load_daily(root):
    """[(date, open, high, low, close, vix)] from the archive (nse_archive.py) and the index bhavcopy files."""
    out = {}
    try:
        import nse_archive
        for row in nse_archive.nifty_vix_daily(root):
            out[row[0]] = row
    except Exception:  # noqa: BLE001
        pass
    for f in glob.glob(os.path.join(root, "nse_data_cache", "bhavcopy", "*.csv")):
        rows = {}
        try:
            for r in csv.reader(open(f, encoding="utf-8", errors="ignore")):
                if r and r[0] in ("Nifty 50", "India VIX"):
                    rows[r[0]] = r
        except Exception:
            continue
        n = rows.get("Nifty 50")
        if not n:
            continue
        try:
            d = datetime.strptime(n[1], "%d-%m-%Y").date()
            vix = float(rows["India VIX"][5]) if "India VIX" in rows else None
            out[d] = (d, float(n[2]), float(n[3]), float(n[4]), float(n[5]), vix)
        except Exception:
            continue
    return [out[d] for d in sorted(out)]


SNAP_KEYS = ("underlying_value", "futures_price", "india_vix", "expiry_epoch", "dte_trading", "atm_iv", "strikes", "timestamp")


def load_replay(root, sym="NIFTY"):
    """{date: [snapshot dicts]} with only the fields the studies need."""
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "replay", f"{sym}_*.jsonl"))):
        snaps = []
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
                t = datetime.strptime(d["timestamp"], "%d-%b-%Y %H:%M:%S").replace(tzinfo=IST)
            except Exception:
                continue
            if not d.get("strikes") or not d.get("expiry_epoch"):
                continue
            s = {k: d.get(k) for k in SNAP_KEYS}
            s["t"] = t
            snaps.append(s)
        if len(snaps) >= 30:
            out[snaps[0]["t"].date()] = snaps
    return out


# ----------------------------------------------------------------------------- chain -> surface
def _q(r, p):
    b, a = r.get(p + "_bid") or 0, r.get(p + "_ask") or 0
    return ((b + a) / 2, a - b) if b > 0 and a > 0 and a >= b else (None, None)


def snap_forward(s):
    """(F, T, D) from put-call parity at the 3 strikes where |C - P| is smallest."""
    T = (s["expiry_epoch"] - s["t"].timestamp()) / YEAR_S
    if T <= 0:
        return None
    D = math.exp(-R * T)
    both = []
    for r in s["strikes"]:
        c, p = _q(r, "ce")[0], _q(r, "pe")[0]
        if c is not None and p is not None:
            both.append((abs(c - p), r["strike"] + (c - p) / D))
    if len(both) < 3:
        return None
    both.sort()
    return float(np.median([f for _, f in both[:3]])), T, D


def chain_points(s, max_spread=0.25, min_mid=0.5):
    """Forward, T and OTM Black-76 IVs from one snapshot (puts below F, calls above). None if unusable."""
    fw = snap_forward(s)
    if not fw or fw[1] <= 0.5 / 365 / 24:
        return None
    F, T, D = fw
    K, mid, spr, gaps = [], [], [], []
    for r in s["strikes"]:
        k = r["strike"]
        c, cs = _q(r, "ce")
        p, ps = _q(r, "pe")
        if abs(k / F - 1) < 0.01 and c is not None and p is not None:
            ic = black_iv(np.array([c / D]), F, np.array([k]), T, True)[0]
            ip = black_iv(np.array([p / D]), F, np.array([k]), T, False)[0]
            if np.isfinite(ic) and np.isfinite(ip):
                gaps.append(abs(ic - ip) * 100)
        m, sp = (c, cs) if k >= F else (p, ps)
        if m is None or m < min_mid or sp / m > max_spread:
            continue
        K.append(k); mid.append(m / D); spr.append(sp / D)
    if len(K) < 6:
        return None
    K, mid, spr = np.array(K, float), np.array(mid), np.array(spr)
    call = K >= F
    iv = black_iv(mid, F, K, T, call)
    ok = np.isfinite(iv)
    if ok.sum() < 6:
        return None
    K, iv, spr = K[ok], iv[ok], spr[ok]
    vega = F * npdf((np.log(F / K) + iv ** 2 * T / 2) / (iv * math.sqrt(T))) * math.sqrt(T)
    return {"F": F, "T": T, "K": K, "iv": iv, "spread_vol": spr / np.maximum(vega, 1e-9), "pc_gap": float(np.mean(gaps)) if gaps else None}


def atm_iv_quick(s):
    """ATM IV (calendar clock) from the straddle at the strike nearest the parity forward."""
    fw = snap_forward(s)
    if not fw:
        return None, None, None
    F, T, D = fw
    rows = [r for r in s["strikes"] if _q(r, "ce")[0] is not None and _q(r, "pe")[0] is not None]
    r = min(rows, key=lambda x: abs(x["strike"] - F))
    st = (_q(r, "ce")[0] + _q(r, "pe")[0]) / D
    lo, hi = 0.005, 3.0
    for _ in range(60):
        m = (lo + hi) / 2
        v = float(black(F, r["strike"], T, m, True) + black(F, r["strike"], T, m, False))
        lo, hi = (lo, m) if v > st else (m, hi)
    return (lo + hi) / 2, F, T


def fit_surface(s, full=False):
    cp = chain_points(s)
    if not cp:
        return None
    F, T, K, iv = cp["F"], cp["T"], cp["K"], cp["iv"]
    k = np.log(K / F)
    o = np.argsort(k)
    k, K, iv = k[o], K[o], iv[o]
    wt = 1 / np.maximum(cp["spread_vol"][o], 0.002) ** 2
    wt = wt / wt.mean()
    p, rmse = fit_svi(k, iv, T, wt)
    ivf = lambda x: math.sqrt(max(float(svi_w(p, x)), 1e-12) / T)
    atm = ivf(0.0)
    kg = np.linspace(k.min() - 0.01, k.max() + 0.01, 401)
    g = svi_g(p, kg)
    # 25-delta strikes on the fitted smile
    vg = np.sqrt(np.maximum(svi_w(p, kg), 1e-12) / T)
    d1 = (-kg + vg * vg * T / 2) / (vg * math.sqrt(T))
    cd, pdl = ncdf(d1), ncdf(d1) - 1
    i25c, i25p = int(np.argmin(np.abs(cd - 0.25))), int(np.argmin(np.abs(pdl + 0.25)))
    slope = (ivf(0.005) - ivf(-0.005)) / 0.01                    # dσ/dk
    out = {"t": s["t"].strftime("%Y-%m-%d %H:%M"), "spot": s.get("underlying_value"), "F": F, "T": T,
           "dte_trading": s.get("dte_trading"), "vix": s.get("india_vix"), "expiry": datetime.fromtimestamp(s["expiry_epoch"], IST).strftime("%Y-%m-%d %H:%M"),
           "atm_iv": atm * 100, "svi": p, "svi_rmse": rmse * 100, "min_g": float(g.min()), "butterfly_arb_free": bool(g.min() > -1e-6),
           "rr25": (vg[i25c] - vg[i25p]) * 100, "bf25": ((vg[i25c] + vg[i25p]) / 2 - atm) * 100,
           "skew_slope": slope,                                   # dσ/dk = vol pts per 1% of moneyness
           "pc_gap": cp["pc_gap"], "n": int(len(K)), "k_range": [float(k.min()), float(k.max())]}
    try:
        out["bkm"] = bkm(F, T, ivf)
    except Exception as e:  # noqa: BLE001
        out["bkm"] = None
        out["bkm_err"] = str(e)
    if full:
        sab, sr = fit_sabr(F, K, iv, T)
        mer, mr = fit_merton(F, K, iv, T)
        Kc = F * np.exp(np.linspace(k.min() - 0.005, k.max() + 0.005, 61))
        mp = merton_price(F, Kc, T, mer["sigma"], mer["lam"], mer["muJ"], mer["deltaJ"], Kc >= F)
        miv = black_iv(mp, F, Kc, T, Kc >= F)
        out.update({"sabr": sab, "sabr_rmse": sr * 100, "merton": mer, "merton_rmse": mr * 100,
                    "points": {"K": K.tolist(), "iv": (iv * 100).tolist()},
                    "curve": {"K": Kc.tolist(), "svi": [ivf(math.log(x / F)) * 100 for x in Kc],
                              "sabr": (sabr_iv(F, Kc, T, sab["alpha"], sab["rho"], sab["nu"]) * 100).tolist(),
                              "merton": [None if not np.isfinite(v) else v * 100 for v in miv]}})
    return out


# ----------------------------------------------------------------------------- the variance clock
def day_r2(v):
    """Squared 1-minute log returns for one day, NaN where a minute is missing or the feed was frozen.
    The index feed often stops updating near 15:15 and the last bar then jumps to the official close
    (the 15:00-15:30 average). Returns (r2, last_live_minute, close_jump_bps)."""
    c, o = v["c"].copy(), v["o"]
    last = SESSION - 1
    j = last - 1
    while j > 0 and np.isfinite(c[j]) and np.isfinite(c[j - 1]) and c[j] == c[j - 1]:
        j -= 1
    live = j if j < last - 2 else last                     # a frozen run of 3+ minutes before the last bar
    jump = None
    if live < last and np.isfinite(c[last]) and np.isfinite(c[live]):
        jump = math.log(c[last] / c[live]) * 1e4
    if live < last:
        c[live + 1:] = np.nan
    prev = np.concatenate([[o[0] if np.isfinite(o[0]) else np.nan], c[:-1]])
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.log(c / prev)
    return r * r, live, jump


def study_clock(mins):
    days = sorted(mins)
    rows, gaps, jumps, lives = [], [], [], []
    prev_close, prev_day = None, None
    for d in days:
        r2, live, jump = day_r2(mins[d])
        rows.append(r2)
        lives.append(live)
        if jump is not None:
            jumps.append(jump)
        o, c = mins[d]["o"], mins[d]["c"]
        if prev_close and prev_day and np.isfinite(o[0]) and (d - prev_day).days <= 4:
            gaps.append(((d - prev_day).days, math.log(o[0] / prev_close) ** 2, d))
        prev_close, prev_day = (c[-1] if np.isfinite(c[-1]) else np.nanmax(c[np.isfinite(c)][-1:])), d
    if len(rows) < 5:
        return None
    M = np.array(rows)
    day_var = np.nansum(M, axis=1)
    keep = day_var <= np.quantile(day_var, 0.97)                       # drop the most extreme days from the shape
    cov = np.isfinite(M[keep]).mean(axis=0)
    prof = np.nanmean(np.where(np.isfinite(M[keep]), M[keep], np.nan), axis=0)
    first_bad = int(np.argmax(cov < 0.5)) if (cov < 0.5).any() else SESSION
    extrap = None
    if first_bad < SESSION:                                            # extrapolate the frozen tail from the 20 live minutes before it
        base = np.nanmean(prof[max(0, first_bad - 20):first_bad])
        prof[first_bad:] = base
        extrap = first_bad
    sess = float(np.nansum(prof))
    u1 = prof / np.nanmean(prof)
    u5 = np.array([u1[i:i + 5].mean() for i in range(0, SESSION, 5)])
    u15 = np.array([u1[i:i + 15].mean() for i in range(0, SESSION, 15)])
    on = [g for n, g, _ in gaps if n == 1]
    wk = [g for n, g, _ in gaps if n >= 3]
    ov_w = float(np.mean(on) / sess) if on else None
    wk_w = float(np.mean(wk) / sess) if wk else None
    hm = lambda m: f"{(555 + m) // 60}:{(555 + m) % 60:02d}"
    return {"n_days": len(rows), "days": [str(days[0]), str(days[-1])], "u5": u5.round(4).tolist(), "u15": u15.round(4).tolist(),
            "first_hour_share": float(prof[:60].sum() / sess), "first_15_share": float(prof[:15].sum() / sess),
            "last_30_share": float(prof[-30:].sum() / sess), "lunch_share": float(prof[150:255].sum() / sess),
            "overnight_w": ov_w, "weekend_w": wk_w, "n_overnight": len(on), "n_weekend": len(wk),
            "overnight_share": float(np.mean(on) / (sess + np.mean(on))) if on else None,
            "session_vol_ann": float(math.sqrt(sess * 252) * 100), "units_per_year": 252 + 200 * (ov_w or .25) + 52 * (wk_w or .35),
            "feed_freeze_from": hm(int(np.median(lives)) + 1) if np.median(lives) < SESSION - 1 else None,
            "extrapolated_from": hm(extrap) if extrap is not None else None,
            "close_jump": {"n": len(jumps), "mean_abs_bps": float(np.mean(np.abs(jumps))) if jumps else None,
                           "sd_bps": float(np.std(jumps)) if jumps else None}}


def clock_frac_remaining(m, u5, averaging=True):
    """Remaining settlement variance today from minute m, as a fraction of a session (with the 15:00-15:30 average)."""
    u = np.repeat(np.asarray(u5, float), 5)[:SESSION] if u5 is not None else np.ones(SESSION)
    tot = u.sum()
    s = np.arange(int(m), SESSION)
    w = np.where((s < SESSION - WINDOW) | (not averaging), 1.0, (SESSION - s - 0.5) / WINDOW)
    return float((w * w * u[s]).sum() / tot)


def units_per_year(clock):
    """Variance units in a year on the calibrated clock: 252 sessions + ~200 weekday nights + ~52 weekends."""
    ow = (clock or {}).get("overnight_w") or 0.25
    ww = (clock or {}).get("weekend_w") or max(ow, 0.35)
    return 252 + 200 * ow + 52 * ww


def variance_units(now, expiry, clock):
    """Variance to expiry in session units on the calibrated clock (weekdays as sessions; holidays unknown)."""
    u5 = clock.get("u5") if clock else None
    ow = (clock or {}).get("overnight_w") or 0.25
    ww = (clock or {}).get("weekend_w") or max(ow, 0.35)
    m = (now.hour * 60 + now.minute) - (9 * 60 + 15)
    exp_m = (expiry.hour * 60 + expiry.minute) - (9 * 60 + 15)
    if now.date() == expiry.date():
        return clock_frac_remaining(max(m, 0), u5) if m < SESSION else 0.0
    units = clock_frac_remaining(max(m, 0), u5, averaging=False) if 0 <= m < SESSION else (1.0 if m < 0 else 0.0)
    d = now.date()
    while True:
        nd = d + timedelta(days=1)
        while nd.weekday() >= 5:
            nd += timedelta(days=1)
        units += ww if (nd - d).days >= 3 else ow
        d = nd
        if d >= expiry.date():
            break
        units += 1.0
    full = clock_frac_remaining(0, u5) if exp_m >= SESSION - 1 else clock_frac_remaining(0, u5) - clock_frac_remaining(max(exp_m, 0), u5)
    return units + full


def study_weekday(mins):
    by = {}
    for d, v in mins.items():
        r2, live, _ = day_r2(v)
        c = v["c"][max(0, SESSION - WINDOW):live + 1]
        c = c[np.isfinite(c)]
        rng_ = (c.max() - c.min()) / c.mean() * 1e4 if len(c) > 5 else np.nan
        by.setdefault(d.strftime("%a"), []).append((math.sqrt(np.nansum(r2[1:]) * 252) * 100, rng_))
    order = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return {k: {"n": len(by[k]), "rv": float(np.mean([a for a, _ in by[k]])), "close_range_bps": float(np.nanmean([b for _, b in by[k]]))}
            for k in order if k in by}


def study_ver(mins, daily, clock):
    """Variance-elapsed ratio: realised session variance so far / (VIX-implied daily variance x clock fraction elapsed).
    VIX includes overnight variance and the risk premium, so the typical full-day value is well below 1; the guide
    therefore reads VER relative to your own median."""
    vix = {x[0]: x[5] for x in daily if x[5]}
    ds = sorted(vix)
    prev = {ds[i]: vix[ds[i - 1]] for i in range(1, len(ds))}
    u5 = clock["u5"] if clock else None
    rows = []
    for d, v in mins.items():
        if d not in prev:
            continue
        r2, live, _ = day_r2(v)
        imp = (prev[d] / 100) ** 2 / units_per_year(clock)
        f11 = 1 - clock_frac_remaining(105, u5, averaging=False)
        flive = 1 - clock_frac_remaining(live + 1, u5, averaging=False) if live < SESSION - 1 else 1.0
        rows.append((np.nansum(r2[:105]) / (imp * f11), np.nansum(r2[:live + 1]) / (imp * flive)))
    if len(rows) < 8:
        return {"n_days": len(rows)}
    a, b = np.array(rows).T
    med = float(np.median(b))
    terc = np.quantile(a, [1 / 3, 2 / 3])
    groups = [b[a <= terc[0]], b[(a > terc[0]) & (a <= terc[1])], b[a > terc[1]]]
    return {"n_days": len(rows), "rank_corr": spearman(a, b), "median_close": med, "ver11_cut": terc.tolist(),
            "ver11_cut_rel": (terc / med).tolist(),
            "close_by_tercile": [float(np.mean(g)) if len(g) else None for g in groups],
            "close_rel_by_tercile": [float(np.mean(g) / med) if len(g) else None for g in groups],
            "top_tercile_days_above_median": float(np.mean(groups[2] > med)) if len(groups[2]) else None,
            "bottom_tercile_days_below_median": float(np.mean(groups[0] < med)) if len(groups[0]) else None}


# ----------------------------------------------------------------------------- replay studies
def study_iv_clock_and_spotvol(replay, clock):
    buckets = {}
    uy = units_per_year(clock)
    pairs_dn, pairs_up = [], []
    n_snaps = 0
    for d, snaps in replay.items():
        series = []
        for s in snaps:
            iv, F, T = atm_iv_quick(s)
            if iv is None or not F:
                continue
            exp = datetime.fromtimestamp(s["expiry_epoch"], IST)
            units = variance_units(s["t"], exp, clock)
            ivt = math.sqrt(iv * iv * T / (units / uy)) if units > 0 else None
            m = (s["t"].hour * 60 + s["t"].minute) - (9 * 60 + 15)
            series.append((s["t"], m, iv * 100, ivt * 100 if ivt else None, F))
        n_snaps += len(series)
        if len(series) < 30:
            continue
        mc = np.mean([x[2] for x in series])
        mt = np.mean([x[3] for x in series if x[3]])
        for t, m, ic, it, F in series:
            if 0 <= m < SESSION:
                b = buckets.setdefault(m // 15, [[], []])
                b[0].append(ic - mc)
                if it:
                    b[1].append(it - mt)
        # spot-vol: 5-minute steps on the trading-clock ATM IV
        last = None
        for t, m, ic, it, F in series:
            if it is None or m < 10:
                continue
            if last is None or (t - last[0]).total_seconds() >= 290:
                if last is not None and (t - last[0]).total_seconds() <= 600:
                    dF = (math.log(F / last[4])) * 100
                    dv = it - last[3]
                    if abs(dF) > 0.02:
                        (pairs_dn if dF < 0 else pairs_up).append((dF, dv))
                last = (t, m, ic, it, F)
    ivc = {"bucket_min": 15, "calendar": [], "trading": [], "n_days": len(replay), "n_snaps": n_snaps}
    for b in range(SESSION // 15):
        x = buckets.get(b)
        ivc["calendar"].append(float(np.mean(x[0])) if x and x[0] else None)
        ivc["trading"].append(float(np.mean(x[1])) if x and x[1] else None)

    def beta(pairs):
        if len(pairs) < 10:
            return None, len(pairs)
        x, y = np.array(pairs).T
        return float(np.sum(x * y) / np.sum(x * x)), len(pairs)
    bd, nd = beta(pairs_dn)
    bu, nu = beta(pairs_up)
    return ivc, {"beta_down": bd, "beta_up": bu, "n_down": nd, "n_up": nu, "n_days": len(replay)}


def study_aggressor(replay, horizon=15, window=5, nstrikes=5):
    """Direction from the aggressor side (buyer- vs seller-initiated volume), weighted by |dOI|, against two naive
    OI readings: the price/OI quadrant and the 'writers' reading (call OI up = resistance, put OI up = support)."""
    sig_a, sig_n, sig_w, fwd = [], [], [], []
    for d, snaps in replay.items():
        S = [s for s in snaps if s.get("futures_price")]
        if len(S) < horizon + window + 5:
            continue
        maps = [{r["strike"]: r for r in s["strikes"]} for s in S]
        for i in range(window, len(S) - horizon):
            if (S[i + horizon]["t"] - S[i]["t"]).total_seconds() > horizon * 90:
                continue
            spot = S[i]["underlying_value"]
            A = N = Wr = wa = wn = ww = 0.0
            cur, old = maps[i], maps[i - window]
            for k in sorted(cur, key=lambda k: abs(k - spot))[:2 * nstrikes + 1]:
                for p, sg in (("ce", 1), ("pe", -1)):
                    buy = sell = 0.0
                    for j in range(i - window + 1, i + 1):
                        a, b = maps[j - 1].get(k), maps[j].get(k)
                        if not a or not b:
                            continue
                        dv = (b.get(p + "_volume") or 0) - (a.get(p + "_volume") or 0)
                        if dv <= 0:
                            continue
                        ltp, bid, ask, l0 = b.get(p + "_ltp") or 0, a.get(p + "_bid") or 0, a.get(p + "_ask") or 0, a.get(p + "_ltp") or 0
                        if ask and ltp >= ask:
                            buy += dv
                        elif bid and ltp <= bid:
                            sell += dv
                        elif l0 and ltp > l0:
                            buy += dv
                        elif l0 and ltp < l0:
                            sell += dv
                    o, c = old.get(k), cur.get(k)
                    if not o or not c:
                        continue
                    doi = (c.get(p + "_oi") or 0) - (o.get(p + "_oi") or 0)
                    if not doi:
                        continue
                    w = abs(doi)
                    if buy != sell:                       # long_open / short_cover = +, write_open / long_exit = - (for calls)
                        A += sg * (1 if buy > sell else -1) * w; wa += w
                    dp = (c.get(p + "_ltp") or 0) - (o.get(p + "_ltp") or 0)
                    if dp:
                        N += sg * (1 if dp > 0 else -1) * w; wn += w
                    Wr += -sg * (1 if doi > 0 else -1) * w; ww += w
            sig_a.append(A / wa if wa else np.nan)
            sig_n.append(N / wn if wn else np.nan)
            sig_w.append(Wr / ww if ww else np.nan)
            fwd.append(math.log(S[i + horizon]["futures_price"] / S[i]["futures_price"]) * 1e4)
    return {"horizon_snaps": horizon, "window_snaps": window, "n": int(np.sum(np.isfinite(sig_a))), "n_days": len(replay),
            "ic_aggressor": spearman(sig_a, fwd), "ic_price_quadrant": spearman(sig_n, fwd), "ic_writer_reading": spearman(sig_w, fwd)}


def study_book_shift(root, sym="NIFTY", hs=(1, 2, 3, 5, 10, 15)):
    sys.path.insert(0, root)
    try:
        import bias_eval as be
    except Exception as e:  # noqa: BLE001
        return {"error": f"bias_eval not importable: {e}"}
    be.HS = hs
    out, n_days = {}, 0
    pooled = {h: ([], []) for h in hs}
    for f in sorted(glob.glob(os.path.join(root, "replay", f"{sym}_*.jsonl"))):
        if os.path.getsize(f) < 1_000_000:
            continue
        n_days += 1
        for F, fw in be.day_samples(f):
            x = F.get("shift")
            for h in hs:
                if x is not None and fw.get(h) is not None:
                    pooled[h][0].append(x); pooled[h][1].append(fw[h])
    for h in hs:
        a, b = pooled[h]
        bal = be.balanced(list(zip(a, b))) if a else None
        out[str(h)] = {"ic": spearman(a, b), "n": len(a), "balanced": bal[0] if bal else None}
    ics = [(h, out[str(h)]["ic"]) for h in hs if out[str(h)]["ic"] is not None]
    peak = max(ics, key=lambda x: abs(x[1])) if ics else None
    half = next((h for h, v in ics if peak and h > peak[0] and abs(v) <= abs(peak[1]) / 2), None) if peak else None
    return {"n_days": n_days, "by_h": out, "peak_h": peak[0] if peak else None, "half_life_h": half}


def study_markouts(root, replay, sym="NIFTY"):
    fills = []
    for f in glob.glob(os.path.join(root, "trade_log", "*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("ev") == "fill" and str(r.get("inst", "")).startswith(f"NFO:{sym}") and r.get("price"):
                fills.append(r)
    res = {h: [] for h in (0, 1, 5, 15)}
    used, modes = 0, set()
    for r in fills:
        t = _ist(r["t"])
        snaps = replay.get(t.date())
        if not snaps:
            continue
        sg = 1 if r["side"] == "BUY" else -1
        side = "ce" if r["inst"].endswith("CE") else "pe"

        def mid_at(ts):
            s = min(snaps, key=lambda x: abs((x["t"] - ts).total_seconds()))
            if abs((s["t"] - ts).total_seconds()) > 90:
                return None
            for row in s["strikes"]:
                if row.get(side + "_symbol") == r["inst"] and (row.get(side + "_bid") or 0) > 0 and (row.get(side + "_ask") or 0) > 0:
                    return (row[side + "_bid"] + row[side + "_ask"]) / 2
            return None
        m0 = mid_at(t)
        if m0 is None:
            continue
        used += 1
        modes.add(r.get("mode"))
        res[0].append(sg * (m0 - r["price"]))
        for h in (1, 5, 15):
            mh = mid_at(t + timedelta(minutes=h))
            if mh is not None:
                res[h].append(sg * (mh - r["price"]))
    return {"n_fills_total": len(fills), "n_matched": used, "modes": sorted(m for m in modes if m),
            "markout_pts": {str(h): (float(np.mean(v)) if v else None) for h, v in res.items()},
            "n_by_h": {str(h): len(v) for h, v in res.items()}}


MONTHS = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "O": 10, "N": 11, "D": 12}


def parse_weekly(sym, ts):
    """NIFTY26N0924500CE -> (date(2026,11,9), 24500, 'CE'); monthly symbols return None."""
    s = ts.split(":")[-1]
    if not s.startswith(sym):
        return None
    r = s[len(sym):]
    try:
        yy, mo, dd = int(r[:2]), MONTHS.get(r[2]), int(r[3:5])
        if mo is None:
            return None
        return date(2000 + yy, mo, dd), float(r[5:-2]), r[-2:]
    except (ValueError, IndexError):
        return None


def study_pinning(root, mins, sym="NIFTY"):
    files = sorted(glob.glob(os.path.join(root, "zerodha_cache", "oi_close_*.json")))
    by_day = {}
    for f in files:
        try:
            by_day[datetime.strptime(os.path.basename(f)[9:19], "%Y-%m-%d").date()] = json.load(open(f)).get("oi", {})
        except Exception:
            continue
    rows = []
    for d, oi in by_day.items():
        exp = {}
        for ts, v in oi.items():
            p = parse_weekly(sym, ts)
            if p:
                exp.setdefault(p[0], {}).setdefault(p[1], 0.0)
                exp[p[0]][p[1]] += v or 0
        for e, strikes in exp.items():
            # OI known at the close before expiry day, expiry day bars available
            if e <= d or e not in mins or (e - d).days > 4:
                continue
            nxt = min((x for x in mins if x > d), default=None)
            if nxt != e:
                continue
            kmax = max(strikes, key=strikes.get)
            c = mins[e]["c"]
            at = lambda m: float(c[m]) if np.isfinite(c[m]) else float(np.nanmean(c[max(0, m - 3):m + 3]))
            rows.append({"expiry": str(e), "max_oi_strike": kmax, "dist_1230": abs(at(195) - kmax), "dist_1529": abs(at(SESSION - 1) - kmax)})
    return {"n_expiries": len(rows), "rows": rows,
            "pinned_share": float(np.mean([r["dist_1529"] < r["dist_1230"] for r in rows])) if rows else None}


def study_participant(root):
    files = sorted(glob.glob(os.path.join(root, "participant_oi", "*.json")))
    series = []
    latest = None
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        day = os.path.basename(f)[:-5]
        row = {"date": day}
        for c in ("Client", "Pro", "FII", "DII"):
            x = d.get(c) or {}
            row[c] = (x.get("opt_idx_call_long", 0) - x.get("opt_idx_call_short", 0)) + (x.get("opt_idx_put_long", 0) - x.get("opt_idx_put_short", 0))
        f_ = d.get("FII") or {}
        tot = (f_.get("fut_idx_long") or 0) + (f_.get("fut_idx_short") or 0)
        row["fii_fut_long_ratio"] = (f_.get("fut_idx_long") or 0) / tot if tot else None
        series.append(row)
        latest = (day, {c: [(d.get(c) or {}).get(k, 0) for k in ("opt_idx_call_long", "opt_idx_call_short", "opt_idx_put_long", "opt_idx_put_short")]
                        for c in ("Client", "Pro", "FII", "DII")})
    return {"n_days": len(series), "series": series[-30:], "latest_date": latest[0] if latest else None, "latest": latest[1] if latest else None}


# ----------------------------------------------------------------------------- daily studies
def daily_rv(daily):
    """Per-day variance proxy: overnight gap^2 + Rogers-Satchell, as annualised vol points."""
    out = []
    for i in range(1, len(daily)):
        d, o, h, l, c, _ = daily[i]
        if (d - daily[i - 1][0]).days > 6:                # a hole in the history, not an overnight gap
            continue
        pc = daily[i - 1][4]
        rs = math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l / o)
        out.append((d, math.sqrt(max(math.log(o / pc) ** 2 + rs, 1e-12) * 252) * 100))
    return out


def study_har(daily):
    rv = daily_rv(daily)
    if len(rv) < 40:
        return {"n": max(0, len(rv) - 22)}
    v = np.array([x[1] for x in rv])
    X, y = [], []
    for t in range(21, len(v) - 1):
        X.append([1, v[t], v[t - 4:t + 1].mean(), v[t - 21:t + 1].mean()]); y.append(v[t + 1])
    X, y = np.array(X), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):     # macOS Accelerate flags clean matmuls
        f = X @ beta
    q = lambda r, fc: float(np.mean(r ** 2 / fc ** 2 - np.log(r ** 2 / fc ** 2) - 1))
    vix = [x[5] for x in daily if x[5]]
    return {"n": int(len(y)), "c": float(beta[0]), "bd": float(beta[1]), "bw": float(beta[2]), "bm": float(beta[3]),
            "r2": float(1 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)),
            "qlike": q(y, np.maximum(f, 1)), "qlike_naive_m": q(y, X[:, 3]),
            "rmse": float(np.sqrt(np.mean((y - f) ** 2))),
            "last": {"d": float(v[-1]), "w": float(v[-5:].mean()), "m": float(v[-22:].mean()), "iv": vix[-1] if vix else None},
            "forecast": float(beta @ [1, v[-1], v[-5:].mean(), v[-22:].mean()]), "proxy": "daily overnight^2 + Rogers-Satchell"}


def study_vrp(daily, h=21):
    closes = np.array([x[4] for x in daily])
    vix = [x[5] for x in daily]
    r = np.diff(np.log(closes))
    rows = []
    for t in range(len(daily) - h):
        if vix[t]:
            rv = math.sqrt(np.sum(r[t:t + h] ** 2) / h * 252) * 100
            rows.append((vix[t], rv))
    if len(rows) < 10:
        return {"n": len(rows)}
    a, b = np.array(rows).T
    return {"n": len(rows), "horizon": h, "mean_vix": float(a.mean()), "mean_rv": float(b.mean()), "mean_vrp": float((a - b).mean()),
            "share_positive": float(np.mean(a > b)), "var_ratio": float(np.mean(a ** 2) / np.mean(b ** 2)),
            "note": "overlapping windows; a short sample is one regime"}


def study_tails(daily, surf):
    if not surf or not surf.get("svi"):
        return None
    F, T, p = surf["F"], surf["T"], surf["svi"]
    h = max(1, int(round(surf.get("dte_trading") or T * 252)))
    closes = np.array([x[4] for x in daily])
    rets = closes[h:] / closes[:-h] - 1 if len(closes) > h else np.array([])
    kg = np.linspace(-0.2, 0.2, 4001)
    K = F * np.exp(kg)
    P = black(F, K, T, np.sqrt(np.maximum(svi_w(p, kg), 1e-12) / T), False)
    cdf = np.gradient(P, K)
    sig = math.sqrt(max(svi_w(p, 0), 1e-12) / T)
    rows = []
    for x in (-4, -3, -2, -1, 1, 2, 3, 4):
        kx = math.log(1 + x / 100)
        rn = float(np.interp(kx, kg, cdf))
        ln = float(ncdf((kx + sig * sig * T / 2) / (sig * math.sqrt(T))))
        mer = surf.get("merton")
        mr = float(merton_cdf(kx, T, mer["sigma"], mer["lam"], mer["muJ"], mer["deltaJ"])) if mer else None
        if x > 0:
            rn, ln, mr = 1 - rn, 1 - ln, (1 - mr if mr is not None else None)
        emp = float(np.mean(rets <= x / 100) if x < 0 else np.mean(rets >= x / 100)) if len(rets) else None
        rows.append({"move": x, "rn_svi": rn, "lognormal_atm": ln, "merton": mr, "empirical": emp})
    return {"horizon_sessions": h, "n_windows": int(len(rets)), "rows": rows,
            "note": "empirical = overlapping windows over the daily history on disk; RN = fitted weekly surface"}


def study_scenarios(daily):
    """Every archived 1-session move as (date, Nifty log return, VIX ratio) for the guide's historical-replay tail
    (the same set nse_span.historical_scenarios gives the live risk preview)."""
    import nse_span
    sc = nse_span.historical_scenarios(daily)
    if not sc:
        return None
    return {"n": len(sc), "from": str(sc[0][0]), "to": str(sc[-1][0]), "d": [str(d) for d, _, _ in sc],
            "r": [round(r, 5) for _, r, _ in sc], "v": [round(v, 4) for _, _, v in sc]}


def read_events(root):
    out = []
    p = os.path.join(root, "guide", "events.txt")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    out.append((datetime.strptime(line[:10], "%Y-%m-%d").date(), line[10:].strip()))
                except ValueError:
                    pass
    return out


def study_events(surf, snap, clock, events):
    if not surf or not surf.get("vix"):
        return None
    t = snap["t"]
    exp = datetime.fromtimestamp(snap["expiry_epoch"], IST)
    w_week = surf["atm_iv"] ** 2 / 1e4 * surf["T"]
    u_week = variance_units(t, exp, clock)
    u30 = variance_units(t, t + timedelta(days=30), clock)
    base = (surf["vix"] / 100) ** 2 * 30 / 365 / u30
    lump = w_week - base * u_week
    inside = [{"date": str(d), "label": l} for d, l in events if t.date() <= d <= exp.date()]
    uy = units_per_year(clock)
    return {"weekly_total_var": w_week, "units_to_expiry": u_week, "base_vol_trading_clock": math.sqrt(base * uy) * 100,
            "lump_var": lump, "event_move_pct": math.sqrt(lump) * 100 if lump > 0 else None,
            "event_move_pts": math.sqrt(lump) * surf["F"] if lump > 0 else None,
            "events_listed": inside, "iv_trading_clock": math.sqrt(w_week / (u_week / uy)) * 100 if u_week > 0 else None,
            "note": "VIX (30-day) is the base; events inside VIX's own window inflate the base, so the lump is a lower bound"}


# ----------------------------------------------------------------------------- driver
def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def calibrate(root=HERE, sym="NIFTY", log=print):
    out = {"generated": datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"), "sym": sym, "errors": {}}

    def run(name, fn, *a):
        try:
            v = fn(*a)
            out[name] = v
            return v
        except Exception as e:  # noqa: BLE001
            out["errors"][name] = f"{type(e).__name__}: {e}"
            log(f"  {name}: FAILED {type(e).__name__}: {e}")
            return None
    mins = load_minutes(root, sym)
    daily = load_daily(root)
    replay = load_replay(root, sym)
    out["sources"] = {"minute_days": len(mins), "minute_range": [str(min(mins)), str(max(mins))] if mins else None,
                      "daily_days": len(daily), "daily_range": [str(daily[0][0]), str(daily[-1][0])] if daily else None,
                      "replay_days": len(replay), "replay_range": [str(min(replay)), str(max(replay))] if replay else None}
    log(f"  sources: {out['sources']}")
    clock = run("clock", study_clock, mins) if mins else None
    run("weekday", study_weekday, mins)
    run("ver", study_ver, mins, daily, clock)
    run("har", study_har, daily)
    run("vrp", study_vrp, daily)
    run("participant", study_participant, root)
    run("pinning", study_pinning, root, mins, sym)
    surf = None
    if replay:
        last_day = max(replay)
        snaps = replay[last_day]
        latest = next((s for s in reversed(snaps) if (s["t"].hour, s["t"].minute) <= (15, 25) and chain_points(s)), None)
        if latest:
            surf = run("surface_latest", fit_surface, latest, True)
            run("events", study_events, surf, latest, clock, read_events(root))
        series = []
        for d in sorted(replay):
            for hh, mm in ((10, 0), (12, 0), (14, 0), (15, 20)):
                s = min(replay[d], key=lambda x: abs((x["t"].hour * 60 + x["t"].minute) - (hh * 60 + mm)))
                if abs((s["t"].hour * 60 + s["t"].minute) - (hh * 60 + mm)) <= 10:
                    try:
                        f = fit_surface(s)
                    except Exception:  # noqa: BLE001
                        f = None
                    if f:
                        series.append({k: f[k] for k in ("t", "spot", "F", "atm_iv", "rr25", "bf25", "skew_slope", "svi_rmse", "butterfly_arb_free", "pc_gap", "vix")}
                                      | {"bkm_skew": (f.get("bkm") or {}).get("skew"), "bkm_kurt": (f.get("bkm") or {}).get("kurt")})
        out["surface_series"] = series
        ivc = run("ivclock_spotvol", study_iv_clock_and_spotvol, replay, clock)
        if ivc:
            out["ivclock"], out["spotvol"] = ivc
            del out["ivclock_spotvol"]
            if surf:
                sl = surf["skew_slope"]
                for k in ("down", "up"):
                    b = out["spotvol"].get("beta_" + k)
                    out["spotvol"]["ratio_" + k] = (b / sl) if (b is not None and sl) else None
                out["spotvol"]["skew_slope"] = sl
        run("aggressor", study_aggressor, replay)
        run("markouts", study_markouts, root, replay, sym)
    run("tails", study_tails, daily, surf)
    run("scenarios", study_scenarios, daily)
    run("book_shift", study_book_shift, root, sym)
    return _clean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="NIFTY")
    ap.add_argument("--root", default=HERE)
    ap.add_argument("--out", default=os.path.join(HERE, "guide", "pro_calib.js"))
    a = ap.parse_args()
    print(f"pro_calibrate: {a.sym} from {a.root}")
    res = calibrate(a.root, a.sym)
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("/* generated by pro_calibrate.py; re-run after more days are recorded */\nwindow.PRO_CALIB = ")
        json.dump(res, fh, separators=(",", ":"), allow_nan=False)
        fh.write(";\n")
    jp = os.path.splitext(a.out)[0] + ".json"                 # the same numbers for the server's live gauge (nse_pro_live.py)
    with open(jp + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(res, fh, separators=(",", ":"), allow_nan=False)
    os.replace(jp + ".tmp", jp)
    s = res.get("surface_latest") or {}
    print(f"  wrote {a.out} ({os.path.getsize(a.out) // 1024} KB)")
    if s:
        print(f"  surface {s.get('t')}: F {s.get('F'):.1f}  ATM IV {s.get('atm_iv'):.2f}%  RR25 {s.get('rr25'):.2f}  BF25 {s.get('bf25'):.2f}  "
              f"SVI rmse {s.get('svi_rmse'):.2f} vol  arb-free {s.get('butterfly_arb_free')}")
    if res["errors"]:
        print("  errors:", res["errors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
