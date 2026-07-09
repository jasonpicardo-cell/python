"""
key_figures.py
==============
A self-contained forecasting module for tracking a public figure through
three lenses at once:

    1. Vedic (sidereal, Lahiri) natal chart + live gochara transits
    2. Vimshottari Dasha timeline (Maha / Antar / Pratyantar)
    3. Pythagorean numerology cycles (life path + personal year/month/day)

Everything is JSON-serialisable via Figure.snapshot(), so it drops straight
into an HTML/JS dashboard tab ("Key Figures") behind a Python backend.

Configure any number of people in FIGURES. Nothing here is hardcoded to one
person; Trump is just the seeded default.

Dependencies: pyswisseph  (pip install pyswisseph)

NOTE ON EPISTEMICS: these outputs are interpretive overlays, not validated
predictors. Treat them as a sentiment/timing lens to overlay on price, never
a standalone signal.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Optional
from zoneinfo import ZoneInfo

import swisseph as swe

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

swe.set_sid_mode(swe.SIDM_LAHIRI)  # Chitrapaksha / Lahiri ayanamsa

# Vimshottari sequence: (dasha lord, years). Total = 120.
DASHA_SEQUENCE = [
    ("Ketu", 7), ("Venus", 20), ("Sun", 6), ("Moon", 10), ("Mars", 7),
    ("Rahu", 18), ("Jupiter", 16), ("Saturn", 19), ("Mercury", 17),
]
DASHA_TOTAL = 120

# 27 nakshatras, each 13°20'. Lord repeats the 9-planet cycle above.
NAKSHATRAS = [
    "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
    "Punarvasu", "Pushya", "Ashlesha", "Magha", "P.Phalguni", "U.Phalguni",
    "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha",
    "Mula", "P.Ashadha", "U.Ashadha", "Shravana", "Dhanishta", "Shatabhisha",
    "P.Bhadrapada", "U.Bhadrapada", "Revati",
]

RASHIS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

# Planets we compute (Vedic set: 7 grahas + nodes). Rahu = mean node; Ketu opposite.
PLANETS = {
    "Sun": swe.SUN, "Moon": swe.MOON, "Mars": swe.MARS, "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER, "Venus": swe.VENUS, "Saturn": swe.SATURN,
    "Rahu": swe.MEAN_NODE,
    # Western outers kept for mundane transit reads
    "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO,
}

# Aspect angles + orbs (degrees). Used for transit-to-natal hits.
ASPECTS = {
    "conjunction": (0, 6), "opposition": (180, 6), "trine": (120, 5),
    "square": (90, 5), "sextile": (60, 4),
}

NAK_ARC = 360.0 / 27.0          # 13.3333...
SIDEREAL_FLAGS = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _julday(when_utc: dt.datetime) -> float:
    """UTC datetime -> Julian day (UT)."""
    frac = (when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600)
    return swe.julday(when_utc.year, when_utc.month, when_utc.day, frac)


def _reduce(n: int, keep_master: bool = False) -> int:
    """Digit-sum reduction. Optionally preserve master numbers 11/22/33."""
    while n > 9:
        if keep_master and n in (11, 22, 33):
            return n
        n = sum(int(d) for d in str(n))
    return n


def _sign(deg: float) -> str:
    return RASHIS[int(deg // 30) % 12]


def _nakshatra(deg: float) -> tuple[str, int, float]:
    """Return (name, pada 1-4, fraction traversed within the nakshatra)."""
    idx = int(deg // NAK_ARC) % 27
    within = (deg % NAK_ARC) / NAK_ARC
    pada = int(within * 4) + 1
    return NAKSHATRAS[idx], pada, within


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

@dataclass
class Figure:
    name: str
    birth_local: dt.datetime          # naive local clock time at birthplace
    tz: str                           # IANA tz, e.g. "America/New_York"
    lat: float
    lon: float
    note: str = ""

    _birth_utc: dt.datetime = field(init=False)
    _natal: dict = field(init=False)

    def __post_init__(self):
        self._birth_utc = self.birth_local.replace(
            tzinfo=ZoneInfo(self.tz)
        ).astimezone(ZoneInfo("UTC"))
        self._natal = self._chart(self._birth_utc)

    # ---- chart primitives -------------------------------------------------

    def _chart(self, when_utc: dt.datetime) -> dict:
        jd = _julday(when_utc)
        out = {}
        for name, pid in PLANETS.items():
            lon = swe.calc_ut(jd, pid, SIDEREAL_FLAGS)[0][0]
            out[name] = lon % 360.0
        out["Ketu"] = (out["Rahu"] + 180.0) % 360.0
        # Ascendant (sidereal)
        cusps, ascmc = swe.houses_ex(
            jd, self.lat, self.lon, b"W", swe.FLG_SIDEREAL
        )
        out["Ascendant"] = ascmc[0] % 360.0
        out["ayanamsa"] = swe.get_ayanamsa_ut(jd)
        return out

    # ---- 1. Vimshottari dasha --------------------------------------------

    def dasha_timeline(self) -> list[dict]:
        """Full Maha->Antar->Pratyantar tree from birth, ~120 yrs forward."""
        moon = self._natal["Moon"]
        nak_idx = int(moon // NAK_ARC) % 27
        frac = (moon % NAK_ARC) / NAK_ARC

        start_lord_idx = nak_idx % 9
        # balance of the birth mahadasha
        first_lord, first_years = DASHA_SEQUENCE[start_lord_idx]
        balance_years = (1 - frac) * first_years

        timeline = []
        cursor = self._birth_utc
        seq_len = len(DASHA_SEQUENCE)

        for step in range(seq_len + 1):  # cover >120 yrs
            lord_idx = (start_lord_idx + step) % seq_len
            lord, full_years = DASHA_SEQUENCE[lord_idx]
            years = balance_years if step == 0 else full_years
            maha = {
                "lord": lord,
                "start": cursor,
                "end": _add_years(cursor, years),
                "antardashas": self._antardashas(lord, cursor, years),
            }
            timeline.append(maha)
            cursor = maha["end"]
        return timeline

    def _antardashas(self, maha_lord: str, start: dt.datetime,
                     maha_years: float) -> list[dict]:
        idx = [l for l, _ in DASHA_SEQUENCE].index(maha_lord)
        res, cursor = [], start
        for k in range(len(DASHA_SEQUENCE)):
            lord, yrs = DASHA_SEQUENCE[(idx + k) % len(DASHA_SEQUENCE)]
            span = maha_years * yrs / DASHA_TOTAL
            end = _add_years(cursor, span)
            res.append({
                "lord": lord, "start": cursor, "end": end,
                "pratyantardashas": self._pratyantar(maha_lord, lord, cursor, span),
            })
            cursor = end
        return res

    def _pratyantar(self, maha_lord, antar_lord, start, antar_years):
        idx = [l for l, _ in DASHA_SEQUENCE].index(antar_lord)
        res, cursor = [], start
        for k in range(len(DASHA_SEQUENCE)):
            lord, yrs = DASHA_SEQUENCE[(idx + k) % len(DASHA_SEQUENCE)]
            span = antar_years * yrs / DASHA_TOTAL
            end = _add_years(cursor, span)
            res.append({"lord": lord, "start": cursor, "end": end})
            cursor = end
        return res

    def active_dasha(self, when: Optional[dt.datetime] = None) -> dict:
        when = _utc(when)
        for maha in self.dasha_timeline():
            if maha["start"] <= when < maha["end"]:
                for antar in maha["antardashas"]:
                    if antar["start"] <= when < antar["end"]:
                        praty = next(
                            (p for p in antar["pratyantardashas"]
                             if p["start"] <= when < p["end"]), None)
                        return {
                            "maha": maha["lord"],
                            "antar": antar["lord"],
                            "pratyantar": praty["lord"] if praty else None,
                            "maha_ends": maha["end"],
                            "antar_ends": antar["end"],
                            "pratyantar_ends": praty["end"] if praty else None,
                        }
        return {}

    # ---- 2. Transits ------------------------------------------------------

    def transits(self, when: Optional[dt.datetime] = None,
                 movers=("Jupiter", "Saturn", "Rahu", "Ketu", "Mars",
                         "Uranus", "Neptune", "Pluto")) -> dict:
        when = _utc(when)
        now = self._chart(when)
        hits = []
        natal_points = {k: v for k, v in self._natal.items()
                        if k not in ("ayanamsa",)}
        for t in movers:
            tlon = now[t]
            for npoint, nlon in natal_points.items():
                sep = abs((tlon - nlon + 180) % 360 - 180)
                for asp, (angle, orb) in ASPECTS.items():
                    if abs(sep - angle) <= orb:
                        hits.append({
                            "transit": t, "aspect": asp, "natal": npoint,
                            "orb": round(abs(sep - angle), 2),
                            "transit_sign": _sign(tlon),
                            "natal_sign": _sign(nlon),
                        })
        hits.sort(key=lambda h: h["orb"])
        positions = {p: {"lon": round(now[p], 2), "sign": _sign(now[p]),
                         "nakshatra": _nakshatra(now[p])[0]}
                     for p in PLANETS} | {"Ketu": {
                         "lon": round(now["Ketu"], 2),
                         "sign": _sign(now["Ketu"]),
                         "nakshatra": _nakshatra(now["Ketu"])[0]}}
        return {"positions": positions, "aspects_to_natal": hits}

    # ---- 3. Numerology ----------------------------------------------------

    def numerology(self, when: Optional[dt.datetime] = None) -> dict:
        when = _utc(when)
        b = self.birth_local
        life_path = _reduce(
            sum(int(d) for d in f"{b.month:02d}{b.day:02d}{b.year}"),
            keep_master=True)
        py = _reduce(_reduce(b.month) + _reduce(b.day) + _reduce(when.year))
        pm = _reduce(py + _reduce(when.month))
        pd = _reduce(pm + _reduce(when.day))
        return {"life_path": life_path, "personal_year": py,
                "personal_month": pm, "personal_day": pd}

    # ---- forward scan: the actual "foresight" ----------------------------

    def upcoming_windows(self, days: int = 365,
                         when: Optional[dt.datetime] = None) -> dict:
        """
        Scan forward `days` and return dated inflection points across all
        three systems. These are the candidate 'windows' — dates where a
        period boundary or exact transit falls. Interpretation is yours.
        """
        start = _utc(when)
        end = start + dt.timedelta(days=days)

        # --- dasha sub-period changes in range -------------------------
        dasha_changes = []
        for maha in self.dasha_timeline():
            if maha["end"] < start or maha["start"] > end:
                continue
            for antar in maha["antardashas"]:
                if start <= antar["start"] <= end:
                    dasha_changes.append({
                        "date": antar["start"], "level": "antardasha",
                        "into": f"{maha['lord']}/{antar['lord']}"})
                for pr in antar["pratyantardashas"]:
                    if start <= pr["start"] <= end:
                        dasha_changes.append({
                            "date": pr["start"], "level": "pratyantar",
                            "into": f"{maha['lord']}/{antar['lord']}/{pr['lord']}"})
        dasha_changes.sort(key=lambda x: x["date"])

        # --- numerology inflections (personal-month rollovers) ---------
        num_changes, cur = [], start.replace(day=1)
        while cur <= end:
            n = self.numerology(cur)
            num_changes.append({"date": cur, "personal_month": n["personal_month"],
                                "personal_year": n["personal_year"]})
            # jump to first of next month
            cur = (cur.replace(day=28) + dt.timedelta(days=8)).replace(day=1)

        # --- slow-transit exact aspects to natal (daily scan) ----------
        transit_hits = self._scan_transit_perfections(start, end)

        return {
            "range": {"from": start.isoformat(), "to": end.isoformat()},
            "dasha_changes": _iso(dasha_changes),
            "numerology_months": _iso(num_changes),
            "transit_perfections": _iso(transit_hits),
        }

    def _scan_transit_perfections(self, start, end,
                                  movers=("Jupiter", "Saturn", "Rahu",
                                          "Ketu", "Mars")):
        """Find dates where a slow transit's aspect-orb to a natal point is
        minimised (i.e. the aspect perfects) within the window."""
        natal = {k: v for k, v in self._natal.items() if k != "ayanamsa"}
        prev = {}          # (transit, natal, aspect) -> (date, orb, closing?)
        results = []
        day = start
        step = dt.timedelta(days=1)
        tracking = {}
        while day <= end:
            chart = self._chart(day)
            for t in movers:
                tlon = chart[t]
                for npoint, nlon in natal.items():
                    sep = abs((tlon - nlon + 180) % 360 - 180)
                    for asp, (angle, orb) in ASPECTS.items():
                        d = abs(sep - angle)
                        key = (t, npoint, asp)
                        if d <= orb:
                            if key not in tracking or d < tracking[key][1]:
                                tracking[key] = (day, d)
                        else:
                            if key in tracking:
                                pk_date, pk_orb = tracking.pop(key)
                                results.append({
                                    "date": pk_date, "transit": t,
                                    "aspect": asp, "natal": npoint,
                                    "min_orb": round(pk_orb, 2)})
            day += step
        for key, (pk_date, pk_orb) in tracking.items():
            t, npoint, asp = key
            results.append({"date": pk_date, "transit": t, "aspect": asp,
                            "natal": npoint, "min_orb": round(pk_orb, 2)})
        results.sort(key=lambda x: x["date"])
        return results

    # ---- assembly ---------------------------------------------------------

    def snapshot(self, when: Optional[dt.datetime] = None) -> dict:
        when = _utc(when)
        natal = {k: (round(v, 2) if isinstance(v, float) else v)
                 for k, v in self._natal.items()}
        natal_readable = {
            p: {"lon": round(self._natal[p], 2), "sign": _sign(self._natal[p]),
                "nakshatra": _nakshatra(self._natal[p])[0],
                "pada": _nakshatra(self._natal[p])[1]}
            for p in list(PLANETS) + ["Ketu", "Ascendant"]
        }
        return {
            "name": self.name,
            "as_of": when.isoformat(),
            "birth_utc": self._birth_utc.isoformat(),
            "natal": natal_readable,
            "dasha": _iso(self.active_dasha(when)),
            "transits": self.transits(when),
            "numerology": self.numerology(when),
            "note": self.note,
        }


# --------------------------------------------------------------------------
# datetime utilities
# --------------------------------------------------------------------------

def _utc(when: Optional[dt.datetime]) -> dt.datetime:
    if when is None:
        return dt.datetime.now(ZoneInfo("UTC"))
    if when.tzinfo is None:
        return when.replace(tzinfo=ZoneInfo("UTC"))
    return when.astimezone(ZoneInfo("UTC"))


def _add_years(start: dt.datetime, years: float) -> dt.datetime:
    return start + dt.timedelta(days=years * 365.2425)


def _iso(obj):
    if isinstance(obj, dict):
        return {k: _iso(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_iso(v) for v in obj]
    if isinstance(obj, dt.datetime):
        return obj.isoformat()
    return obj


# --------------------------------------------------------------------------
# Seeded figures
# --------------------------------------------------------------------------

FIGURES = {
    "trump": Figure(
        name="Donald J. Trump",
        birth_local=dt.datetime(1946, 6, 14, 10, 54),
        tz="America/New_York",   # EDT in effect -> UTC-4
        lat=40.7009, lon=-73.7890,   # Jamaica, Queens NY
        note="Birth time 10:54 AM EDT (AA-rated, from birth certificate).",
    ),
}


if __name__ == "__main__":
    import json
    fig = FIGURES["trump"]
    print(json.dumps(fig.snapshot(), indent=2, default=str))
