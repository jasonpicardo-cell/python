"""
nse_confluence_scanner.py — Multi-Signal Confluence Scanner

Reads cached results from existing scanners and counts how many
independent methods agree on direction for each stock.

Score: +1 bullish signal, -1 bearish, 0 neutral
Range -8 to +8. Display as bullish/bearish count and net score.
"""
from __future__ import annotations
import time

_conf_cache: dict = {"ts": 0.0, "data": None, "key": ""}
_DEFAULT_TTL = 120   # shorter TTL — reflects live scanner states


# Maps scanner module.run_fn → (is_bullish_fn, is_bearish_fn)
def _score_stock(sym: str, source: str) -> dict | None:
    """Collect signal for one symbol from all available scanner caches."""
    results: dict[str, str] = {}
    price = None; price_date = None

    def _try(mod_name: str, sig_key: str = "signal"):
        nonlocal price, price_date
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            cache = getattr(mod, "_" + mod_name.replace("nse_", "").replace("_scanner", "") + "_cache", None)
            if cache and cache.get("data"):
                for s in cache["data"].get("stocks", []):
                    if s.get("symbol") == sym:
                        if price is None:
                            price = s.get("price"); price_date = s.get("price_date")
                        return s.get(sig_key)
        except Exception:
            pass
        return None

    BULLISH_SIGS = {
        "nse_rsi_scanner":       {"Bullish", "Oversold"},
        "nse_macd_scanner":      {"Bullish Crossover", "Zero Cross Up", "Above Zero"},
        "nse_adx_scanner":       {},   # direction-based, handled separately
        "nse_bb_scanner":        {"Lower Touch", "Outside Lower"},
        "nse_momentum_scanner":  {"Strong Up", "Up"},
        "nse_breakout_scanner":  {"52W Breakout", "20D Breakout", "10D Breakout"},
        "nse_supertrend":        {"Bullish"},
        "nse_ichimoku_scanner":  {"Strong Bullish", "Bullish", "Weak Bullish"},
        "nse_stoch_scanner":     {"Oversold", "Bull Cross", "Bullish"},
        "nse_williamsr_scanner": {"Oversold", "Turning Up"},
        "nse_cci_scanner":       {"Oversold", "Zero Cross Up", "Bullish"},
    }
    BEARISH_SIGS = {
        "nse_rsi_scanner":       {"Bearish", "Overbought"},
        "nse_macd_scanner":      {"Bearish Crossover", "Zero Cross Down", "Below Zero"},
        "nse_bb_scanner":        {"Upper Touch", "Outside Upper"},
        "nse_momentum_scanner":  {"Strong Down", "Down"},
        "nse_breakout_scanner":  {"52W Breakdown", "20D Breakdown", "10D Breakdown"},
        "nse_ichimoku_scanner":  {"Strong Bearish", "Bearish", "Weak Bearish"},
        "nse_stoch_scanner":     {"Overbought", "Bear Cross", "Bearish"},
        "nse_williamsr_scanner": {"Overbought", "Turning Down"},
        "nse_cci_scanner":       {"Overbought", "Zero Cross Down", "Bearish"},
    }

    bull = 0; bear = 0
    details: dict[str, str] = {}

    for mod_name, bull_set in BULLISH_SIGS.items():
        sig = _try(mod_name)
        if sig:
            details[mod_name.replace("nse_","").replace("_scanner","")] = sig
            if sig in bull_set: bull += 1
            elif sig in BEARISH_SIGS.get(mod_name, set()): bear += 1

    # ADX direction separately
    adx_sig = _try("nse_adx_scanner", "direction")
    if adx_sig:
        if adx_sig == "Bullish": bull += 1
        elif adx_sig == "Bearish": bear += 1
        details["adx_dir"] = adx_sig

    # Supertrend from trend scanner
    try:
        import nse_trend_scanner as _ts
        if _ts._trend_cache.get("data"):
            for s in _ts._trend_cache["data"].get("stocks", []):
                if s.get("symbol") == sym:
                    if price is None: price = s.get("price"); price_date = s.get("price_date")
                    st = s.get("supertrend", {})
                    if st.get("signal") == "Bullish": bull += 1; details["supertrend"] = "Bullish"
                    elif st.get("signal") == "Bearish": bear += 1; details["supertrend"] = "Bearish"
    except Exception:
        pass

    if price is None: return None
    total = bull + bear
    score = bull - bear
    if   score >= 4:  label = "Strong Bull"
    elif score >= 2:  label = "Bull"
    elif score <= -4: label = "Strong Bear"
    elif score <= -2: label = "Bear"
    else:             label = "Neutral"

    return {"symbol": sym, "price": price, "price_date": price_date,
            "bull": bull, "bear": bear, "total": total,
            "score": score, "signal": label, "details": details}


def run_confluence_scanner(source="niftyfno", force=False, ttl=_DEFAULT_TTL) -> dict:
    from nse_pivot_scanner import load_symbol_list
    key = source; c = _conf_cache
    if not force and c["data"] and c["key"] == key and (time.time()-c["ts"]) < ttl:
        return c["data"]

    symbols, sym_label = load_symbol_list(source)
    if not symbols:
        return {"stocks":[],"errors":[],"skipped":[],"count":0,"total":0,"source":source,
                "sym_label":sym_label,"as_of":time.strftime("%H:%M:%S"),
                "error":"Run individual scanners first — confluence reads their caches."}

    stocks = [r for sym in symbols
              if (r := _score_stock(sym, source)) is not None and r["total"] >= 3]

    data = {"stocks":stocks,"errors":[],"skipped":[],"count":len(stocks),
            "total":len(symbols),"source":source,"sym_label":sym_label,
            "as_of":time.strftime("%H:%M:%S"),
            "note":"Only shows stocks with ≥3 signals loaded. Run other scanners first."}
    c["ts"]=time.time(); c["data"]=data; c["key"]=key
    return data
