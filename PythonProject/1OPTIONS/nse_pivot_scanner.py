"""
nse_pivot_scanner.py
====================
F&O Pivot Scanner — reads OHLC from ../nse_data_cache/*.csv files.

Key design decisions
---------------------
* Current price = CSV last close, always.  No live NSE fetch.
  (is_market_open() cannot detect public holidays without a holiday
  calendar, so live prices would be stale/wrong on holidays.)
* Each result carries ohlc_used so the caller can verify the source data.
* All errors are collected into the result dict instead of being silently
  swallowed so the UI can surface them.
"""

from __future__ import annotations

import csv
import datetime
import glob
import logging
import time
from datetime import date, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

_TOOL_DIR = Path(__file__).resolve().parent
DATA_DIR  = _TOOL_DIR.parent / "nse_data_cache"
FNO_FILE  = _TOOL_DIR.parent / "niftyfno.txt"   # kept for back-compat

# All supported symbol-list sources.
# "all" is a sentinel → load_symbol_list() returns every CSV in DATA_DIR.
SYMBOL_LISTS: dict[str, Path | None] = {
    "niftyfno": _TOOL_DIR.parent / "niftyfno.txt",
    "nifty50":  _TOOL_DIR.parent / "nifty50.txt",
    "nifty100": _TOOL_DIR.parent / "nifty100.txt",
    "nifty200": _TOOL_DIR.parent / "nifty200.txt",
    "nifty500": _TOOL_DIR.parent / "nifty500.txt",
    "nifty750": _TOOL_DIR.parent / "nifty750.txt",
    "all":      None,   # special: glob DATA_DIR
}

PIVOT_LEVELS = ("R5", "R4", "R3", "R2", "R1", "P", "S1", "S2", "S3", "S4", "S5")


# ── Pivot formulas (TradingView-matching) ────────────────────────────────────

def _pv_fibonacci(H, L, C, O):
    r = H - L;  P = (H + L + C) / 3
    return dict(P=P,
        R1=P+0.382*r, R2=P+0.618*r, R3=P+r,     R4=P+1.272*r, R5=P+1.618*r,
        S1=P-0.382*r, S2=P-0.618*r, S3=P-r,     S4=P-1.272*r, S5=P-1.618*r)

def _pv_traditional(H, L, C, O):
    r = H - L;  P = (H + L + C) / 3
    return dict(P=P,
        R1=2*P-L, R2=P+r,   R3=P+2*r, R4=P+3*r, R5=P+4*r,
        S1=2*P-H, S2=P-r,   S3=P-2*r, S4=P-3*r, S5=P-4*r)

def _pv_woodie(H, L, C, O):
    r = H - L;  P = (H + L + 2*C) / 4
    return dict(P=P,
        R1=2*P-L, R2=P+r,   R3=P+2*r, R4=P+3*r, R5=P+4*r,
        S1=2*P-H, S2=P-r,   S3=P-2*r, S4=P-3*r, S5=P-4*r)

def _pv_camarilla(H, L, C, O):
    r = H - L;  P = (H + L + C) / 3
    return dict(P=P,
        R1=C+1.1*r/12, R2=C+1.1*r/6, R3=C+1.1*r/4, R4=C+1.1*r/2, R5=C+1.1*r,
        S1=C-1.1*r/12, S2=C-1.1*r/6, S3=C-1.1*r/4, S4=C-1.1*r/2, S5=C-1.1*r)

def _pv_demark(H, L, C, O):
    X = H+2*L+C if C < O else (2*H+L+C if C > O else H+L+2*C)
    P = X / 4
    return dict(P=P, R1=X/2-L, S1=X/2-H,
                R2=None, R3=None, R4=None, R5=None,
                S2=None, S3=None, S4=None, S5=None)

_PV_FNS = {
    "fibonacci":   _pv_fibonacci,
    "traditional": _pv_traditional,
    "classic":     _pv_traditional,
    "woodie":      _pv_woodie,
    "camarilla":   _pv_camarilla,
    "demark":      _pv_demark,
}

def compute_pivots(H, L, C, O=None, pivot_type="fibonacci") -> dict:
    fn  = _PV_FNS.get(pivot_type, _pv_fibonacci)
    raw = fn(float(H), float(L), float(C), float(O or C))
    return {k: (round(float(v), 2) if v is not None else None) for k, v in raw.items()}


# ── Market status (display only — NOT used to gate price source) ─────────────

_IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def market_status() -> str:
    now = datetime.datetime.now(_IST)
    if now.weekday() >= 5:
        return "weekend"
    m = now.hour * 60 + now.minute
    if m < 9*60+15:   return "pre-market"
    if m <= 15*60+30: return "open"
    return "closed"


# ── Symbol list ──────────────────────────────────────────────────────────────

def load_symbol_list(source: str = "niftyfno") -> tuple[list[str], str]:
    """
    Load the symbol list for *source*.

    Returns (symbols, label) where label is a human-readable description.
    *source* must be one of the keys in SYMBOL_LISTS, or "fno" (alias for
    "niftyfno" kept for backward compatibility).

    "all" → returns every symbol that has a CSV file in DATA_DIR.
    """
    # back-compat alias
    if source == "fno":
        source = "niftyfno"

    if source == "all":
        syms  = load_all_csv_symbols()
        return syms, f"All CSVs in {DATA_DIR.name}/ ({len(syms)} stocks)"

    path = SYMBOL_LISTS.get(source)
    if path is None:
        # unknown key → fall back to niftyfno
        path = SYMBOL_LISTS["niftyfno"]
        source = "niftyfno"

    if not path.exists():
        return [], f"{path.name} not found at {path}"

    syms: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                syms.append(s)

    labels = {
        "niftyfno": "F&O Stocks",
        "nifty50":  "Nifty 50",
        "nifty100": "Nifty 100",
        "nifty200": "Nifty 200",
        "nifty500": "Nifty 500",
        "nifty750": "Nifty 750",
    }
    label = f"{labels.get(source, source)} ({len(syms)} stocks)"
    return syms, label


# kept for any external code that imported it directly
def load_fno_symbols() -> list[str]:
    syms, _ = load_symbol_list("niftyfno")
    return syms


# ── CSV helpers ──────────────────────────────────────────────────────────────

def _find_csv(symbol: str) -> Path | None:
    """
    Locate the CSV for *symbol* in DATA_DIR.
    Tries exact match first, then glob with case-insensitive fallback.
    """
    sym_up = symbol.upper()
    for name in (symbol, sym_up, f"{symbol}.NS", f"{sym_up}.NS"):
        p = DATA_DIR / f"{name}.csv"
        if p.exists():
            return p
    # glob fallback — handles case differences on case-sensitive filesystems
    pattern = str(DATA_DIR / f"{sym_up}*.csv")
    matches = glob.glob(pattern, recursive=False)
    if matches:
        return Path(matches[0])
    return None

def _to_iso(raw: str) -> str:
    """
    Normalise any common date string to YYYY-MM-DD.
    Handles: YYYY-MM-DD, DD-MM-YYYY, DD/MM/YYYY, YYYY/MM/DD,
             and timestamps like 'YYYY-MM-DD HH:MM:SS'.
    Falls back to the raw string if no format matches.
    """
    raw = raw.strip()[:19]   # trim possible timestamp suffix
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(raw[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return raw[:10]   # unknown format — return as-is

def _read_rows(path: Path, n: int = 30) -> tuple[list[dict], str | None]:
    """
    Read the last *n* data rows from a CSV.
    Returns (rows, error_string).  error_string is None on success.
    Accepts any encoding (tries utf-8-sig then latin-1 as fallback).
    Normalises the date column to YYYY-MM-DD via _to_iso().
    """
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        rows: list[dict] = []
        try:
            with open(path, newline="", encoding=enc) as f:
                raw_text = f.read()

            # Auto-detect column delimiter (comma or semicolon)
            delim = ";" if raw_text.count(";") > raw_text.count(",") else ","

            reader = csv.DictReader(raw_text.splitlines(), delimiter=delim)
            if not reader.fieldnames:
                continue

            # Normalise header names (strip spaces, lowercase for matching)
            headers = {h.strip(): h for h in reader.fieldnames}

            def _col(*candidates) -> str | None:
                for c in candidates:
                    if c in headers:       return headers[c]
                    if c.lower() in {k.lower(): k for k in headers}:
                        return {k.lower(): k for k in headers}[c.lower()]
                return None

            col_date   = _col("Datetime", "Date", "datetime", "date", "DATETIME", "DATE")
            col_open   = _col("Open",  "OPEN",  "open")
            col_high   = _col("High",  "HIGH",  "high")
            col_low    = _col("Low",   "LOW",   "low")
            col_close  = _col("Close", "CLOSE", "close")

            if not all([col_date, col_open, col_high, col_low, col_close]):
                return [], (
                    f"Missing columns. Found: {list(headers.keys())}. "
                    f"Need: Date/Datetime, Open, High, Low, Close."
                )

            parse_errors = 0
            for row in reader:
                try:
                    rows.append({
                        "date": _to_iso(row[col_date]),
                        "O": float(row[col_open]  or 0),
                        "H": float(row[col_high]  or 0),
                        "L": float(row[col_low]   or 0),
                        "C": float(row[col_close] or 0),
                    })
                except (ValueError, TypeError, KeyError):
                    parse_errors += 1

            if rows:
                return rows[-n:], None
            if parse_errors:
                return [], f"All {parse_errors} rows failed to parse (tried encoding={enc})"

        except Exception as exc:
            last_exc = str(exc)
            continue

    return [], f"Could not read {path.name}: {locals().get('last_exc', 'unknown error')}"


def get_daily_ohlc(symbol: str) -> tuple[dict | None, str | None]:
    """
    Return (ohlc_dict, error_string).
    ohlc_dict is the last completed row (date < today).
    """
    p = _find_csv(symbol)
    if not p:
        return None, f"CSV not found in {DATA_DIR}"

    rows, err = _read_rows(p, n=10)
    if err:
        return None, err
    if not rows:
        return None, f"No rows in {p.name}"

    today = date.today().isoformat()   # "2026-06-26"
    past  = [r for r in rows if r["date"] < today]

    if not past:
        return None, (
            f"All rows have date >= today ({today}). "
            f"Dates found: {[r['date'] for r in rows]}. "
            "Check that nse_data_cache contains historical (not future) data."
        )

    return past[-1], None


def get_weekly_ohlc(symbol: str) -> tuple[dict | None, str | None]:
    """
    Previous complete Mon–Fri week OHLC. Matches TradingView 'Weekly' pivot.
    Reads 100 rows to ensure we always reach the prior week even with holiday gaps.
    """
    p = _find_csv(symbol)
    if not p:
        return None, f"CSV not found in {DATA_DIR}"
    rows, err = _read_rows(p, n=100)
    if err:
        return None, err
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if not rows:
        return None, f"No historical rows before {today}"

    last_fri = date.today() - timedelta(days=1)
    while last_fri.weekday() != 4:
        last_fri -= timedelta(days=1)
    last_mon = last_fri - timedelta(days=4)

    week = [r for r in rows
            if last_mon.isoformat() <= r["date"] <= last_fri.isoformat()]
    if not week:
        return None, (
            f"No rows for prev week {last_mon} to {last_fri}. "
            f"CSV range: {rows[0]['date']} to {rows[-1]['date']}."
        )
    return {
        "date": f"{last_mon} to {last_fri}",
        "O": week[0]["O"], "H": max(r["H"] for r in week),
        "L": min(r["L"] for r in week), "C": week[-1]["C"],
    }, None


def get_monthly_ohlc(symbol: str) -> tuple[dict | None, str | None]:
    """Previous calendar month OHLC — matches TradingView 'Monthly' pivot."""
    p = _find_csv(symbol)
    if not p:
        return None, f"CSV not found in {DATA_DIR}"
    rows, err = _read_rows(p, n=60)
    if err:
        return None, err
    today = date.today().isoformat()
    rows  = [r for r in rows if r["date"] < today]
    if not rows:
        return None, f"No historical rows before {today}"

    prev_end    = date.today().replace(day=1) - timedelta(days=1)
    month_pfx   = prev_end.strftime("%Y-%m")
    month_rows  = [r for r in rows if r["date"].startswith(month_pfx)]
    if not month_rows:
        return None, (
            f"No rows for {month_pfx}. "
            f"CSV range: {rows[0]['date']} to {rows[-1]['date']}."
        )
    return {
        "date": month_pfx,
        "O": month_rows[0]["O"], "H": max(r["H"] for r in month_rows),
        "L": min(r["L"] for r in month_rows), "C": month_rows[-1]["C"],
    }, None


def get_quarterly_ohlc(symbol: str) -> tuple[dict | None, str | None]:
    """Previous calendar quarter OHLC — matches TradingView 'Quarterly' pivot."""
    p = _find_csv(symbol)
    if not p:
        return None, f"CSV not found in {DATA_DIR}"
    rows, err = _read_rows(p, n=200)
    if err:
        return None, err
    today     = date.today()
    today_iso = today.isoformat()
    rows      = [r for r in rows if r["date"] < today_iso]
    if not rows:
        return None, f"No historical rows before {today_iso}"

    q = (today.month - 1) // 3
    if q == 0:
        pq_s = date(today.year - 1, 10, 1);  pq_e = date(today.year - 1, 12, 31)
    else:
        pq_s = date(today.year, (q - 1) * 3 + 1, 1)
        pq_e = date(today.year, q * 3, 1) - timedelta(days=1)

    qrows = [r for r in rows if pq_s.isoformat() <= r["date"] <= pq_e.isoformat()]
    if not qrows:
        return None, (
            f"No rows for {pq_s} to {pq_e}. "
            f"CSV range: {rows[0]['date']} to {rows[-1]['date']}."
        )
    return {
        "date": f"{pq_s} to {pq_e}",
        "O": qrows[0]["O"], "H": max(r["H"] for r in qrows),
        "L": min(r["L"] for r in qrows), "C": qrows[-1]["C"],
    }, None


def get_yearly_ohlc(symbol: str) -> tuple[dict | None, str | None]:
    """
    Previous calendar year OHLC — matches TradingView 'Yearly' pivot.
    O = first trading day open, H/L = year extremes, C = last trading day close.
    """
    p = _find_csv(symbol)
    if not p:
        return None, f"CSV not found in {DATA_DIR}"
    rows, err = _read_rows(p, n=600)    # ~2.5 years of daily data
    if err:
        return None, err
    today     = date.today().isoformat()
    rows      = [r for r in rows if r["date"] < today]
    if not rows:
        return None, f"No historical rows before {today}"

    prev_year = date.today().year - 1
    yr_rows   = [r for r in rows if r["date"].startswith(str(prev_year))]
    if not yr_rows:
        return None, (
            f"No rows for {prev_year}. "
            f"CSV range: {rows[0]['date']} to {rows[-1]['date']}."
        )
    return {
        "date": f"{prev_year}-01-01 to {prev_year}-12-31",
        "O": yr_rows[0]["O"], "H": max(r["H"] for r in yr_rows),
        "L": min(r["L"] for r in yr_rows), "C": yr_rows[-1]["C"],
    }, None


# ── Near-level helpers ───────────────────────────────────────────────────────

_NEAR_PCT = 2.0

def near_levels(price: float, pivots: dict, pct: float = _NEAR_PCT) -> list[str]:
    """Return names of all pivot levels within pct% of price."""
    if not price:
        return []
    return [lv for lv in PIVOT_LEVELS
            if pivots.get(lv) is not None
            and abs(price - pivots[lv]) / pivots[lv] * 100 <= pct]

def closest_level(price: float, pivots: dict) -> tuple[str, float]:
    """Return (level_name, pct_distance) for the nearest pivot level."""
    if not price:
        return ("", 0.0)
    best, dist = "", float("inf")
    for lv in PIVOT_LEVELS:
        v = pivots.get(lv)
        if v:
            d = abs(price - v) / v * 100
            if d < dist:
                dist, best = d, lv
    return (best, round(dist, 2))


def _get_pivot_ohlc(symbol: str, mode: str) -> tuple[dict | None, str | None]:
    return {
        "daily":     get_daily_ohlc,
        "weekly":    get_weekly_ohlc,
        "monthly":   get_monthly_ohlc,
        "quarterly": get_quarterly_ohlc,
        "yearly":    get_yearly_ohlc,
    }.get(mode, get_daily_ohlc)(symbol)


def lookup_symbol(symbol: str, mode: str = "daily",
                  pivot_type: str = "fibonacci") -> dict:
    """
    Compute pivot data for ANY stock that has a CSV — NOT limited to niftyfno.txt.
    Used by the direct symbol-search UI.
    """
    symbol = symbol.upper().strip()
    result = _get_pivot_ohlc(symbol, mode)

    # _get_pivot_ohlc returns a 2-tuple (ohlc_dict | None, error_str | None)
    if not isinstance(result, tuple) or len(result) != 2:
        return {"error": f"Unexpected return from OHLC fetch: {type(result)}", "symbol": symbol}
    pivot_ohlc, pivot_err = result

    if pivot_ohlc is None:
        return {"error": pivot_err or "OHLC not found", "symbol": symbol, "mode": mode}

    H = float(pivot_ohlc.get("H") or 0)
    L = float(pivot_ohlc.get("L") or 0)
    C = float(pivot_ohlc.get("C") or 0)
    O = float(pivot_ohlc.get("O") or 0)

    if not (H > 0 and L > 0 and C > 0):
        return {"error": f"Zero/invalid OHLC H={H} L={L} C={C}", "symbol": symbol}

    # Current price = latest daily close (not the pivot-period close)
    if mode != "daily":
        day_result = get_daily_ohlc(symbol)
        day_ohlc   = day_result[0] if isinstance(day_result, tuple) else None
        price = round(float((day_ohlc.get("C") or C) if day_ohlc else C), 2)
    else:
        price = round(C, 2)

    pivots     = compute_pivots(H=H, L=L, C=C, O=O, pivot_type=pivot_type)
    cl, cl_pct = closest_level(price, pivots)
    return {
        "symbol":      symbol,
        "price":       price,
        "ohlc_used":   {
            "date": pivot_ohlc.get("date", ""),
            "O": round(O, 2), "H": round(H, 2),
            "L": round(L, 2), "C": round(C, 2),
        },
        "pivots":      pivots,
        "near":        near_levels(price, pivots),
        "closest":     cl,
        "closest_pct": cl_pct,
        "mode":        mode,
        "pivot_type":  pivot_type,
    }


_scan_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300   # 5-minute default scanner cache

def load_all_csv_symbols() -> list[str]:
    """List all stock symbols available as CSV files in DATA_DIR."""
    if not DATA_DIR.exists():
        return []
    syms = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        name = p.stem                              # filename without .csv
        if name.endswith(".NS"):
            name = name[:-3]                       # strip .NS suffix
        syms.append(name.upper())
    return syms


def run_scanner(mode="daily", pivot_type="fibonacci",
                source="fno",                      # "fno" = niftyfno.txt, "all" = all CSVs
                fetcher=None, headers=None, force=False, ttl=_DEFAULT_TTL) -> dict:
    """
    Run the pivot scanner.

    mode        – "daily" | "weekly"
    pivot_type  – fibonacci | traditional | classic | woodie | camarilla | demark
    source      – "fno"  → load symbols from niftyfno.txt (F&O universe)
                  "all"  → load symbols from all CSV files in nse_data_cache/

    Price = most recent daily close from CSV, regardless of mode.
    Weekly mode uses last-complete-week H/L/C for pivot levels,
    but the displayed price is still the latest trading day's close.
    """
    key = f"{mode}:{pivot_type}:{source}"
    c   = _scan_cache
    if (not force and c["data"] and c["key"] == key
            and (time.time() - c["ts"]) < ttl):
        return c["data"]

    if source == "all":
        symbols, sym_source = load_symbol_list("all")
    else:
        symbols, sym_source = load_symbol_list(source)

    if not symbols:
        path = SYMBOL_LISTS.get(source)
        return {
            "stocks": [], "errors": [], "skipped": [], "count": 0, "total": 0,
            "mode": mode, "pivot_type": pivot_type, "source": source,
            "as_of": time.strftime("%H:%M:%S"),
            "market_status": market_status(),
            "data_dir": str(DATA_DIR),
            "sym_source": sym_source,
            "error": (
                f"No CSV files found in {DATA_DIR}" if source == "all"
                else f"{path.name if path else source} not found or empty at {path or 'unknown'}"
            ),
        }

    stocks: list[dict] = []
    errors: list[dict] = []
    skipped: list[str] = []

    for sym in symbols:
        # ── Pivot OHLC — dispatched by mode ────────────────────────────
        pivot_ohlc, pivot_err = _get_pivot_ohlc(sym, mode)

        if pivot_ohlc is None:
            (skipped if pivot_err and "not found" in pivot_err else errors).append(
                {"symbol": sym, "reason": pivot_err or "no OHLC"}
            )
            continue

        H = pivot_ohlc.get("H", 0)
        L = pivot_ohlc.get("L", 0)
        C = pivot_ohlc.get("C", 0)
        O = pivot_ohlc.get("O", 0)

        if not (H > 0 and L > 0 and C > 0):
            errors.append({"symbol": sym,
                           "reason": f"Zero values in pivot OHLC: H={H} L={L} C={C}"})
            continue

        # ── Current price = latest daily close (never the pivot-period close) ─
        # In non-daily modes the pivot_ohlc["C"] is a historical period close.
        # Always show the most recent trading day's close as "current price."
        if mode != "daily":
            day_ohlc, _ = get_daily_ohlc(sym)
            price = round((day_ohlc["C"] if day_ohlc and day_ohlc.get("C") else C), 2)
        else:
            price = round(C, 2)

        pivots     = compute_pivots(H=H, L=L, C=C, O=O, pivot_type=pivot_type)
        cl, cl_pct = closest_level(price, pivots)

        stocks.append({
            "symbol":      sym,
            "price":       price,
            "ohlc_used":   {
                "date": pivot_ohlc.get("date", ""),
                "O": round(O, 2), "H": round(H, 2),
                "L": round(L, 2), "C": round(C, 2),
            },
            "pivots":      pivots,
            "near":        near_levels(price, pivots),
            "closest":     cl,
            "closest_pct": cl_pct,
        })

    data = {
        "stocks":        stocks,
        "errors":        errors,
        "skipped":       skipped,
        "count":         len(stocks),
        "total":         len(symbols),
        "mode":          mode,
        "pivot_type":    pivot_type,
        "source":        source,
        "sym_source":    sym_source,
        "as_of":         time.strftime("%H:%M:%S"),
        "market_status": market_status(),
        "data_dir":      str(DATA_DIR),
        "fno_file":      str(FNO_FILE),
    }
    c["ts"] = time.time(); c["data"] = data; c["key"] = key
    return data


# ── Debug helper (HTTP endpoint + CLI) ───────────────────────────────────────

def debug_symbol(symbol: str) -> dict:
    """
    Return a full diagnostic dict for one symbol.
    Called by GET /api/pivot-scanner/debug?symbol=X
    and by the diagnose() CLI helper below.
    """
    result: dict = {
        "symbol":   symbol,
        "data_dir": str(DATA_DIR),
        "fno_file": str(FNO_FILE),
        "data_dir_exists": DATA_DIR.exists(),
        "fno_file_exists": FNO_FILE.exists(),
    }

    csv_path = _find_csv(symbol)
    result["csv_path"] = str(csv_path) if csv_path else None
    result["csv_found"] = csv_path is not None

    if not csv_path:
        result["error"] = f"No CSV found for {symbol!r} in {DATA_DIR}"
        return result

    rows, err = _read_rows(csv_path, n=10)
    result["read_error"] = err
    result["rows_read"] = len(rows)
    result["last_5_rows"] = rows[-5:]

    daily, derr = get_daily_ohlc(symbol)
    result["daily_ohlc"]  = daily
    result["daily_error"] = derr

    weekly, werr = get_weekly_ohlc(symbol)
    result["weekly_ohlc"]  = weekly
    result["weekly_error"] = werr

    if daily and daily.get("H"):
        H, L, C, O = daily["H"], daily["L"], daily["C"], daily["O"]
        result["fibonacci_pivots"] = compute_pivots(H=H, L=L, C=C, O=O)
        result["price_in_scanner"] = round(C, 2)

    return result


def diagnose(symbol: str) -> None:
    """CLI diagnostic. Run: python3 -c \"from nse_pivot_scanner import diagnose; diagnose('RELIANCE')\" """
    import json, pprint
    d = debug_symbol(symbol)
    print(f"\n{'='*60}  PIVOT SCANNER DIAGNOSTIC  {'='*60}")
    for k, v in d.items():
        if k in ("last_5_rows",):
            print(f"\n{k}:")
            for r in (v or []):
                print(f"  {r}")
        elif k in ("fibonacci_pivots",):
            print(f"\n{k}:")
            for lv, val in (v or {}).items():
                if val is not None:
                    print(f"  {lv:4s} = {val:>10.2f}")
        else:
            print(f"  {k:22s}: {v}")
    print(f"{'='*140}\n")
