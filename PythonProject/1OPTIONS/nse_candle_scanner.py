"""
nse_candle_scanner.py  — Candlestick Pattern Scanner

Single-candle:  Doji, Hammer, Shooting Star, Spinning Top
Two-candle:     Bullish/Bearish Engulfing, Piercing Line, Dark Cloud Cover
Three-candle:   Morning Star, Evening Star

Each pattern carries a bias: Bullish | Bearish | Neutral
Overall bias for the stock = majority or strongest pattern detected.
"""
from __future__ import annotations
import logging, time
from datetime import date
from nse_pivot_scanner import _find_csv, _read_rows, load_symbol_list

log = logging.getLogger(__name__)
_candle_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 300

PATTERN_BIAS: dict[str, str] = {
    "Doji":              "Neutral",
    "Spinning Top":      "Neutral",
    "Hammer":            "Bullish",
    "Inverted Hammer":   "Bullish",
    "Piercing Line":     "Bullish",
    "Bullish Engulfing": "Bullish",
    "Morning Star":      "Bullish",
    "Shooting Star":     "Bearish",
    "Hanging Man":       "Bearish",
    "Dark Cloud Cover":  "Bearish",
    "Bearish Engulfing": "Bearish",
    "Evening Star":      "Bearish",
}


def _body(r: dict) -> float: return abs(r["C"] - r["O"])
def _rng(r: dict)  -> float: return r["H"] - r["L"]
def _bull(r: dict) -> bool:  return r["C"] >= r["O"]
def _bear(r: dict) -> bool:  return r["C"] <  r["O"]
def _upper(r: dict) -> float: return r["H"] - max(r["O"], r["C"])
def _lower(r: dict) -> float: return min(r["O"], r["C"]) - r["L"]


def detect_patterns(rows: list[dict]) -> list[str]:
    """rows[-1] = most recent candle. Returns detected pattern names."""
    if not rows: return []
    c = rows[-1]; bc = _body(c); rc = _rng(c)
    if rc == 0: return []
    uc = _upper(c); lc = _lower(c)
    patterns: list[str] = []

    # ── Single-candle ─────────────────────────────────────────────────
    body_ratio = bc / rc
    if body_ratio < 0.05:
        patterns.append("Doji")
    elif body_ratio < 0.25 and uc > bc and lc > bc:
        patterns.append("Spinning Top")
    elif bc > 0:
        if lc >= 2.0 * bc and uc <= 0.3 * bc:
            patterns.append("Hammer" if _bull(c) else "Hanging Man")
        if uc >= 2.0 * bc and lc <= 0.3 * bc:
            patterns.append("Shooting Star" if _bear(c) else "Inverted Hammer")

    if len(rows) < 2: return patterns

    # ── Two-candle ────────────────────────────────────────────────────
    p = rows[-2]; bp = _body(p)

    # Bullish Engulfing
    if _bear(p) and _bull(c) and bp > 0 and c["O"] < p["C"] and c["C"] > p["O"]:
        patterns.append("Bullish Engulfing")

    # Bearish Engulfing
    if _bull(p) and _bear(c) and bp > 0 and c["O"] > p["C"] and c["C"] < p["O"]:
        patterns.append("Bearish Engulfing")

    # Piercing Line
    if (_bear(p) and _bull(c) and bp > 0 and
            c["O"] < p["L"] and c["C"] > (p["O"] + p["C"]) / 2):
        patterns.append("Piercing Line")

    # Dark Cloud Cover
    if (_bull(p) and _bear(c) and bp > 0 and
            c["O"] > p["H"] and c["C"] < (p["O"] + p["C"]) / 2):
        patterns.append("Dark Cloud Cover")

    if len(rows) < 3: return patterns

    # ── Three-candle ──────────────────────────────────────────────────
    pp = rows[-3]; bpp = _body(pp)

    # Morning Star
    if (_bear(pp) and bpp > 0 and
            bp < 0.35 * bpp and              # small middle candle
            _bull(c) and bc > 0.5 * bpp and  # strong bullish close
            c["C"] > (pp["O"] + pp["C"]) / 2):
        patterns.append("Morning Star")

    # Evening Star
    if (_bull(pp) and bpp > 0 and
            bp < 0.35 * bpp and
            _bear(c) and bc > 0.5 * bpp and
            c["C"] < (pp["O"] + pp["C"]) / 2):
        patterns.append("Evening Star")

    return patterns


def scan_stock(symbol: str) -> dict | None:
    p = _find_csv(symbol)
    if not p: return None
    rows, err = _read_rows(p, n=6)
    if err or not rows: return None
    today = date.today().isoformat()
    rows = [r for r in rows if r["date"] < today]
    if len(rows) < 2: return None

    patterns = detect_patterns(rows[-3:])
    if not patterns: return None          # only include stocks with a detected pattern

    biases   = [PATTERN_BIAS.get(pat, "Neutral") for pat in patterns]
    bull_cnt = biases.count("Bullish")
    bear_cnt = biases.count("Bearish")
    if   bull_cnt > bear_cnt: overall = "Bullish"
    elif bear_cnt > bull_cnt: overall = "Bearish"
    else:                     overall = "Neutral"

    return {"symbol": symbol, "price": round(rows[-1]["C"], 2),
            "price_date": rows[-1]["date"],
            "patterns": patterns, "bias": overall}


def run_candle_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    key = source; c = _candle_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]
    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,
                "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),
                "error":f"Symbol list empty for {source!r}"}
    stocks, errors, skipped = [], [], []
    for sym in symbols:
        try:
            r = scan_stock(sym)
            (stocks if r else skipped).append(r or sym)
        except Exception as exc:
            errors.append({"symbol": sym, "reason": str(exc)})
    stocks = [s for s in stocks if isinstance(s, dict)]
    data = {"stocks":stocks,"errors":errors,"skipped":skipped,
            "count":len(stocks),"total":len(symbols),
            "source":source,"sym_label":sym_label,"as_of":time.strftime("%H:%M:%S")}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
