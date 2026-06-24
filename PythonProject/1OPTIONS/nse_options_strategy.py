#!/usr/bin/env python3
"""
nse_options_strategy.py
========================

Live NSE option-chain puller + rule-based strategy suggester for index
options (NIFTY / BANKNIFTY / FINNIFTY).

WHAT IT DOES
------------
1. Opens a real browser-like session against nseindia.com (NSE blocks bare
   `requests.get()` calls — you need to warm up cookies on the option-chain
   page first, then hit the underlying JSON API with those cookies + a
   realistic header set).
2. Parses the chain: OI, change-in-OI, IV, LTP for calls & puts at every
   strike.
3. Computes PCR, max pain, support/resistance walls (by OI), and flags
   strikes showing fresh long/short build-up or unwinding.
4. Runs a transparent, rule-based read (NOT a black box) and prints/exports
   candidate structures for both intraday and positional trading, with the
   reasoning spelled out so you can sanity-check it against your own view.
5. Writes a dark-themed standalone HTML report, same family as your other
   scanner outputs.

IMPORTANT
---------
This is a positioning READ, not a signal generator or a recommendation
engine. OI tells you where existing option writers are positioned *right
now* — it does not predict direction. Treat the output as one input
alongside your own price-action / SMC / Wyckoff read, not a replacement
for it. Max pain, PCR, and OI walls are all lagging/contemporaneous
measures, and OI walls move intraday as writers adjust.

USAGE
-----
    pip install requests curl_cffi   # curl_cffi is required — see note below

    python3 nse_options_strategy.py --symbol NIFTY
    python3 nse_options_strategy.py --symbol BANKNIFTY --expiry 2026-06-24
    python3 nse_options_strategy.py --symbol NIFTY --band 10 --output report.html
    python3 nse_options_strategy.py --symbol NIFTY --debug   # see every request's status/body

    # Test against a previously saved JSON dump (no network needed):
    python3 nse_options_strategy.py --symbol NIFTY --offline saved_chain.json

WHY curl_cffi IS REQUIRED
--------------------------
NSE sits behind Akamai Bot Manager, which fingerprints the raw TLS handshake
(JA3/JA4) before it even reads your HTTP headers. Python's stock `requests`
library has a recognizable, catalogued TLS fingerprint that gets flagged as
non-browser traffic regardless of how convincing your User-Agent/headers are
— this is why even a homepage GET can come back as an Akamai "Access Denied"
page. `curl_cffi` wraps a libcurl build that replicates Chrome's actual TLS/
HTTP2 fingerprint, which is what gets you past this layer. The script will
run without it but will very likely fail.

NOTES ON THE NSE ENDPOINT
--------------------------
- NSE rate-limits aggressively. Don't poll faster than ~once every
  3-5 seconds, and definitely not in a tight loop.
- The session cookies expire after a few minutes of inactivity — this
  script re-warms the session on every run, which is the safe default for
  a script you run manually or via a scheduled job (cron / Task Scheduler).
- Valid `--symbol` values for indices: NIFTY, BANKNIFTY, FINNIFTY,
  MIDCPNIFTY. For individual equities the endpoint differs slightly
  (`option-chain-equities` instead of `option-chain-indices`) — see the
  `INDEX_SYMBOLS` set below if you want to extend this to stocks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import html as html_lib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# NSE sits behind Akamai, which fingerprints the raw TLS handshake (JA3/JA4) —
# not just HTTP headers — to identify non-browser clients. Python's `requests`
# (via urllib3/OpenSSL) has a recognizable TLS fingerprint that gets flagged
# regardless of how browser-like your headers look. curl_cffi wraps libcurl's
# browser-impersonation build, which replicates Chrome's actual TLS/HTTP2
# fingerprint and is the standard fix for this. We use it when available and
# fall back to plain `requests` (which will likely still get blocked) otherwise.
try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    cffi_requests = None
    HAS_CURL_CFFI = False

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}

# NSE's anti-bot layer fingerprints requests closely. A header set that's
# "close enough" (just User-Agent + Accept) gets served a decoy 404 instead
# of an honest block. These mirror what a real Chrome/Edge browser sends,
# based on what currently-working scrapers (e.g. nsepython) use in practice.
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

# Headers for full page loads (homepage, option-chain page) — mimics a
# top-level browser navigation.
PAGE_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,"
              "image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8,en-GB;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Sec-Ch-Ua": '"Chromium";v="126", "Not.A/Brand";v="8", "Google Chrome";v="126"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

# Headers for the XHR-style JSON API call — mimics what the page's own
# JavaScript sends when it fetches the chain (same-origin, cors, json accept).
API_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,en-IN;q=0.8,en-GB;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Sec-Ch-Ua": '"Chromium";v="126", "Not.A/Brand";v="8", "Google Chrome";v="126"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "X-Requested-With": "XMLHttpRequest",
}

NSE_HOME = "https://www.nseindia.com"
NSE_OC_PAGE = "https://www.nseindia.com/option-chain"
# NSE migrated off the old static /api/option-chain-indices route to this
# Next.js-based gateway (consistent with the v3.x app rebuild visible in the
# site's own meta tags). getSymbolDerivativesData returns ALL expiries and
# strikes in a single flat list (one row per CE or PE, not nested) — we
# recombine it into the old nested-by-strike shape in _normalize_nextapi_payload
# so the rest of the parsing code below doesn't need to change.
NSE_OC_API_INDEX = (
    "https://www.nseindia.com/api/NextApi/apiClient/GetQuoteApi"
    "?functionName=getSymbolDerivativesData&symbol={symbol}"
)
# Used for India VIX (and could expose other live indices if useful later).
# This is on the OLD /api/ namespace (not NextApi) — per community-documented
# usage (nsepython), unverified live in this build. If it 404s the way the
# old option-chain endpoint did, the fix is almost certainly the same kind
# of migration to a NextApi-style route — check with --debug.
NSE_ALL_INDICES_API = "https://www.nseindia.com/api/allIndices"

# Thresholds used by the rule-based read. Tune these if you find them too
# loose/tight for the underlying you're trading.
BUILDUP_OI_CHG_THRESHOLD_PCT = 8.0   # % of that strike's OI added/removed today to call it "fresh"
PCR_BULLISH = 1.20
PCR_BEARISH = 0.75
IV_SKEW_NOTABLE_PCT = 1.5            # absolute IV points difference to flag skew


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class StrikeData:
    strike: float
    ce_oi: int = 0
    ce_oi_chg: int = 0
    ce_iv: float = 0.0
    ce_ltp: float = 0.0
    ce_volume: int = 0
    ce_bid: float = 0.0
    ce_ask: float = 0.0
    pe_oi: int = 0
    pe_oi_chg: int = 0
    pe_iv: float = 0.0
    pe_ltp: float = 0.0
    pe_volume: int = 0
    pe_bid: float = 0.0
    pe_ask: float = 0.0


@dataclass
class ChainSnapshot:
    symbol: str
    expiry: str
    underlying_value: float
    timestamp: str
    strikes: list[StrikeData] = field(default_factory=list)
    all_expiries: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Step 1: NSE session + fetch
# --------------------------------------------------------------------------

class NSEFetchError(RuntimeError):
    pass


def _normalize_nextapi_payload(raw: dict) -> dict:
    """The new NextApi endpoint returns a flat list under `data` — one row
    per CE or PE (with an `optionType` field), plus top-level `underlyingValue`
    and `timestamp`. Recombine it into the old `{"records": {"data": [...],
    "expiryDates": [...], "underlyingValue": ..., "timestamp": ...}}` shape
    so parse_chain() and everything downstream needs no further changes."""
    if "data" not in raw:
        raise NSEFetchError(f"Unexpected NextApi response shape — no 'data' key. Keys: {list(raw.keys())}")

    # IMPORTANT: unlike the old API, `underlyingValue` (and likely `timestamp`)
    # are NOT top-level keys in this payload — they're repeated inside each
    # individual row, the same way they lived inside each old-style CE/PE
    # object. Scan rows for the first one that actually carries each field
    # rather than assuming raw['data'][0] has it (the 'XX' rows, when
    # present, may not) or that it's top-level at all.
    #
    # Also: this endpoint appears to return numeric-looking fields (strike
    # price, underlying value) as STRINGS rather than JSON numbers in some
    # responses — cast defensively wherever arithmetic happens on them.
    def _to_float(v):
        try:
            return float(v) if v not in (None, "") else 0.0
        except (TypeError, ValueError):
            return 0.0

    underlying_value = _to_float(raw.get("underlyingValue"))
    timestamp = raw.get("timestamp") or ""
    if not underlying_value or not timestamp:
        for entry in raw["data"]:
            if not underlying_value:
                uv = _to_float(entry.get("underlyingValue"))
                if uv:
                    underlying_value = uv
            if not timestamp and entry.get("timestamp"):
                timestamp = entry["timestamp"]
            if underlying_value and timestamp:
                break

    combined: dict[tuple, dict] = {}
    expiry_set: set[str] = set()
    for entry in raw["data"]:
        sp = _to_float(entry.get("strikePrice"))
        ed = entry.get("expiryDate")
        ot = entry.get("optionType")
        if not sp or not ed or ot not in ("CE", "PE"):
            continue
        expiry_set.add(ed)
        key = (sp, ed)
        if key not in combined:
            combined[key] = {"strikePrice": sp, "expiryDate": ed, "CE": None, "PE": None}
        combined[key][ot] = entry

    def _sort_key(d: str):
        try:
            return datetime.strptime(d, "%d-%b-%Y")
        except ValueError:
            return datetime.max  # unparseable dates sort last rather than crashing

    expiry_dates = sorted(expiry_set, key=_sort_key)

    # Sanity check: if we still couldn't find a spot price, or it falls
    # nowhere near the actual strikes on offer, something about the schema
    # has shifted again — fail loudly here rather than silently handing
    # back underlying_value=0 and letting find_atm_strike() quietly pick
    # the lowest strike in the chain as "ATM" (which is exactly what broke
    # the dashboard last time, with no error at any layer).
    if combined:
        all_strikes_seen = [k[0] for k in combined.keys()]
        lo_strike, hi_strike = min(all_strikes_seen), max(all_strikes_seen)
        margin = (hi_strike - lo_strike) * 0.5  # generous — spot should be well inside the chain, but don't be overly strict
        if not underlying_value or not (lo_strike - margin <= underlying_value <= hi_strike + margin):
            raise NSEFetchError(
                f"Could not reliably determine the spot price from NSE's response "
                f"(got underlyingValue={underlying_value!r}, but strikes range from "
                f"{lo_strike} to {hi_strike} — these don't line up). NSE likely changed "
                f"the response schema again. Run with --debug and inspect a single row "
                f"of the raw payload (e.g. via --dump-json) to find the new field name."
            )

    return {
        "records": {
            "data": list(combined.values()),
            "expiryDates": expiry_dates,
            "underlyingValue": underlying_value,
            "timestamp": timestamp,
        }
    }


class NSESession:
    """Handles the cookie warm-up handshake NSE requires before its JSON
    API will return real data instead of `{}` or a decoy 404."""

    def __init__(self, timeout: int = 10, max_retries: int = 4, debug: bool = False):
        if HAS_CURL_CFFI:
            self.session = cffi_requests.Session(impersonate="chrome124")
            self._backend = "curl_cffi (TLS-impersonating — recommended)"
        else:
            self.session = requests.Session()
            self._backend = "requests (no TLS impersonation)"
            print(
                "[!] curl_cffi not installed — falling back to plain `requests`.\n"
                "[!] NSE's Akamai protection fingerprints the TLS handshake itself, "
                "which plain `requests` cannot fake even with perfect headers.\n"
                "[!] This fetch will very likely 403 again. Fix: pip install curl_cffi\n"
            )
        self.timeout = timeout
        self.max_retries = max_retries
        self.debug = debug
        self._warmed = False

    def _log(self, msg: str) -> None:
        if self.debug:
            print(f"[debug] {msg}")

    def _snippet(self, resp: requests.Response) -> str:
        body = resp.text[:200].replace("\n", " ")
        return f"status={resp.status_code} content-type={resp.headers.get('content-type')} body~={body!r}"

    def _warm_up(self) -> None:
        self._log(f"Backend: {self._backend}")
        # Hit the homepage, then the option-chain page, in that order — NSE's
        # WAF checks for this referer/navigation chain and sets cookies along
        # the way. Both use full page-navigation headers, not API headers.
        r1 = self.session.get(NSE_HOME, headers=PAGE_HEADERS, timeout=self.timeout)
        self._log(f"GET {NSE_HOME} -> {self._snippet(r1)}")
        if r1.status_code != 200:
            raise NSEFetchError(f"Homepage warm-up failed: {self._snippet(r1)}")
        time.sleep(1.0)

        page_headers = dict(PAGE_HEADERS)
        page_headers["Referer"] = NSE_HOME
        r2 = self.session.get(NSE_OC_PAGE, headers=page_headers, timeout=self.timeout)
        self._log(f"GET {NSE_OC_PAGE} -> {self._snippet(r2)}")
        if r2.status_code != 200:
            raise NSEFetchError(f"Option-chain page warm-up failed: {self._snippet(r2)}")
        time.sleep(1.0)

        self._warmed = True

    def get_option_chain(self, symbol: str) -> dict:
        symbol = symbol.upper()
        # NOTE: previously restricted to INDEX_SYMBOLS only. Confirmed via
        # nsepython's source that getSymbolDerivativesData works identically
        # for individual F&O stocks (just pass the stock's symbol) — no
        # separate endpoint needed. We no longer hard-block other symbols
        # here; an invalid symbol will simply fail naturally against NSE's
        # API with a clear error from the retry loop below.

        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if not self._warmed:
                    self._log(f"Attempt {attempt}: warming up session...")
                    self._warm_up()

                api_headers = dict(API_HEADERS)
                api_headers["Referer"] = NSE_OC_PAGE
                url = NSE_OC_API_INDEX.format(symbol=symbol)
                resp = self.session.get(url, headers=api_headers, timeout=self.timeout)
                self._log(f"GET {url} -> {self._snippet(resp)}")

                if resp.status_code != 200:
                    raise NSEFetchError(
                        f"HTTP {resp.status_code} from NSE API "
                        f"(this is almost always anti-bot detection, not a missing "
                        f"endpoint — the route is correct). Body preview: "
                        f"{resp.text[:150]!r}"
                    )
                data = resp.json()
                if not data or "data" not in data:
                    raise NSEFetchError("Empty/invalid JSON — session likely not warmed correctly")
                return _normalize_nextapi_payload(data)
            except Exception as exc:  # noqa: BLE001 - retry on anything, report clearly
                last_err = exc
                self._warmed = False  # force a fresh handshake on retry
                self.session.cookies.clear()
                wait = 2.0 * attempt
                self._log(f"Attempt {attempt} failed ({exc}); retrying in {wait:.1f}s with a fresh session")
                time.sleep(wait)

        raise NSEFetchError(
            f"Failed to fetch NSE option chain for {symbol} after "
            f"{self.max_retries} attempts. Last error: {last_err}\n\n"
            "A 404/403 here means NSE's anti-bot layer flagged the request — the\n"
            "API route itself (api/option-chain-indices) is correct and current.\n"
            "Things to try:\n"
            "  1. Run with --debug to see the exact status/body at each step.\n"
            "  2. Open https://www.nseindia.com/option-chain in a real browser first\n"
            "     — if THAT also fails/redirects oddly, NSE may be geo/IP-blocking\n"
            "     your network entirely (common on some VPNs, cloud IPs, or certain\n"
            "     ISPs), not just the scripted request.\n"
            "  3. Wait 60-90 seconds before retrying — repeated quick attempts while\n"
            "     testing can trip a short-lived rate-limit block.\n"
            "  4. Try --offline against a manually saved chain in the meantime (see\n"
            "     --dump-json to capture one next time a fetch succeeds)."
        )

    def get_india_vix(self) -> float:
        """Fetches India VIX from NSE's allIndices endpoint, reusing the same
        warmed-up session as the option chain. Raises NSEFetchError with a
        clear message if the schema doesn't match what's expected (this
        endpoint is community-documented, not something we've verified live
        against the real site in this build — same caution as everything
        else fetched from NSE in this codebase)."""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if not self._warmed:
                    self._warm_up()
                api_headers = dict(API_HEADERS)
                api_headers["Referer"] = NSE_OC_PAGE
                resp = self.session.get(NSE_ALL_INDICES_API, headers=api_headers, timeout=self.timeout)
                self._log(f"GET {NSE_ALL_INDICES_API} -> {self._snippet(resp)}")
                if resp.status_code != 200:
                    raise NSEFetchError(f"HTTP {resp.status_code} fetching allIndices")
                data = resp.json()
                rows = data.get("data", [])
                vix_row = next((r for r in rows if r.get("index") == "INDIA VIX"), None)
                if not vix_row:
                    raise NSEFetchError(
                        f"'INDIA VIX' not found among {len(rows)} index rows — "
                        f"the allIndices schema may have changed."
                    )
                last_val = vix_row.get("last")
                if last_val is None:
                    raise NSEFetchError(f"INDIA VIX row found but has no 'last' field: {vix_row}")
                return float(last_val)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                self._warmed = False
                self.session.cookies.clear()
                time.sleep(1.5 * attempt)
        raise NSEFetchError(f"Failed to fetch India VIX after {self.max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Step 2: Parse
# --------------------------------------------------------------------------

_BS_R = 0.065   # risk-free rate — consistent with nse_strategy_engine.py


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(S: float, K: float, T: float, sigma: float, r: float, opt: str) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if opt == "CE" else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if opt == "CE":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def _bs_implied_vol(S: float, K: float, T: float, mkt_price: float, opt: str,
                    r: float = _BS_R) -> float:
    """Bisection BS IV solver.  Returns IV as a percentage (e.g. 15.2 = 15.2%),
    or 0.0 when IV can't be determined (price at or below intrinsic value)."""
    intrinsic = max(S - K, 0.0) if opt == "CE" else max(K - S, 0.0)
    if mkt_price <= max(intrinsic, 0.05):
        return 0.0
    lo, hi = 0.01, 5.0   # 1% – 500%
    for _ in range(60):
        mid = (lo + hi) / 2.0
        p = _bs_price(S, K, T, mid, r, opt)
        if p < mkt_price:
            lo = mid
        else:
            hi = mid
        if (hi - lo) < 1e-5:
            break
    iv = (lo + hi) / 2.0
    return round(iv * 100.0, 2)


def parse_chain(raw: dict, symbol: str, expiry_filter: Optional[str] = None) -> ChainSnapshot:
    records = raw.get("records", {})
    all_expiries = records.get("expiryDates", [])
    underlying_value = float(records.get("underlyingValue", 0.0))
    timestamp = records.get("timestamp", "")

    target_expiry = expiry_filter or (all_expiries[0] if all_expiries else None)
    if expiry_filter and expiry_filter not in all_expiries:
        raise NSEFetchError(
            f"Expiry '{expiry_filter}' not found. Available expiries: {all_expiries}"
        )

    by_strike: dict[float, StrikeData] = {}
    for row in records.get("data", []):
        if row.get("expiryDate") != target_expiry:
            continue
        strike = float(row["strikePrice"])
        sd = by_strike.setdefault(strike, StrikeData(strike=strike))

        ce = row.get("CE")
        if ce:
            sd.ce_oi = int(ce.get("openInterest", 0))
            sd.ce_oi_chg = int(ce.get("changeinOpenInterest", 0))
            sd.ce_iv = float(ce.get("impliedVolatility", 0.0) or 0.0)
            sd.ce_ltp = float(ce.get("lastPrice", 0.0) or 0.0)
            sd.ce_volume = int(ce.get("totalTradedVolume", 0) or 0)
            sd.ce_bid = float(ce.get("buyPrice1", ce.get("bidprice", 0.0)) or 0.0)
            sd.ce_ask = float(ce.get("sellPrice1", ce.get("askPrice", 0.0)) or 0.0)

        pe = row.get("PE")
        if pe:
            sd.pe_oi = int(pe.get("openInterest", 0))
            sd.pe_oi_chg = int(pe.get("changeinOpenInterest", 0))
            sd.pe_iv = float(pe.get("impliedVolatility", 0.0) or 0.0)
            sd.pe_ltp = float(pe.get("lastPrice", 0.0) or 0.0)
            sd.pe_volume = int(pe.get("totalTradedVolume", 0) or 0)
            sd.pe_bid = float(pe.get("buyPrice1", pe.get("bidprice", 0.0)) or 0.0)
            sd.pe_ask = float(pe.get("sellPrice1", pe.get("askPrice", 0.0)) or 0.0)

    strikes = sorted(by_strike.values(), key=lambda s: s.strike)
    if not strikes:
        raise NSEFetchError(f"No strikes parsed for expiry {target_expiry} — check the expiry value.")

    # IV fallback: NSE's NextApi returns impliedVolatility=0 for many strikes.
    # Compute from Black-Scholes when the API gives nothing — this ensures
    # the skew chart, vol surface, and Greeks always have usable IV data.
    dte_for_iv = max(days_to_expiry(target_expiry), 0)
    T_iv = max(dte_for_iv / 365.0, 1 / 365.0)
    for sd in strikes:
        if sd.ce_iv <= 0 and sd.ce_ltp > 0:
            sd.ce_iv = _bs_implied_vol(underlying_value, sd.strike, T_iv, sd.ce_ltp, "CE")
        if sd.pe_iv <= 0 and sd.pe_ltp > 0:
            sd.pe_iv = _bs_implied_vol(underlying_value, sd.strike, T_iv, sd.pe_ltp, "PE")

    return ChainSnapshot(
        symbol=symbol,
        expiry=target_expiry,
        underlying_value=underlying_value,
        timestamp=timestamp,
        strikes=strikes,
        all_expiries=all_expiries,
    )


# --------------------------------------------------------------------------
# Step 3: Analysis
# --------------------------------------------------------------------------

def find_atm_strike(snap: ChainSnapshot) -> float:
    return min(snap.strikes, key=lambda s: abs(s.strike - snap.underlying_value)).strike


def compute_pcr(snap: ChainSnapshot) -> float:
    total_ce = sum(s.ce_oi for s in snap.strikes)
    total_pe = sum(s.pe_oi for s in snap.strikes)
    return round(total_pe / total_ce, 3) if total_ce else 0.0


def compute_max_pain(snap: ChainSnapshot) -> float:
    """Strike at which option WRITERS (sellers) collectively lose the
    least — i.e. the strike where total intrinsic payout to buyers is
    minimized. Classic 'max pain' theory says price gravitates here into
    expiry, though this gets noisier the further out you are from expiry."""
    distribution = compute_payout_distribution(snap)
    return min(distribution, key=lambda d: d[1])[0]


def compute_payout_distribution(snap: ChainSnapshot) -> list[tuple[float, float]]:
    """The full per-strike writer-payout curve that compute_max_pain picks
    its single minimum from. Returns [(strike, total_payout_to_holders), ...]
    for every strike in the chain — this is the 'Gain/Pain by strike' data:
    LOW payout = strikes where option WRITERS are doing well (their 'gain'
    zone), HIGH payout = strikes where writers are hurting most (their
    'pain' zone). Same O(n²) computation compute_max_pain always did
    internally; this just returns the whole curve instead of only the argmin."""
    all_strikes = [s.strike for s in snap.strikes]
    distribution = []
    for candidate in all_strikes:
        total_payout = 0.0
        for s in snap.strikes:
            if candidate > s.strike:
                total_payout += (candidate - s.strike) * s.ce_oi  # ITM calls
            if candidate < s.strike:
                total_payout += (s.strike - candidate) * s.pe_oi  # ITM puts
        distribution.append((candidate, total_payout))
    return distribution


def support_resistance(snap: ChainSnapshot, atm: float, band: int, strike_gap: float):
    """Within `band` strikes either side of ATM, find the strike with the
    largest Call OI (resistance) and largest Put OI (support)."""
    lo = atm - band * strike_gap
    hi = atm + band * strike_gap
    nearby = [s for s in snap.strikes if lo <= s.strike <= hi]
    resistance = max(nearby, key=lambda s: s.ce_oi)
    support = max(nearby, key=lambda s: s.pe_oi)
    return support, resistance, nearby


def infer_strike_gap(snap: ChainSnapshot) -> float:
    diffs = sorted(set(round(b.strike - a.strike, 2)
                        for a, b in zip(snap.strikes, snap.strikes[1:])))
    return diffs[0] if diffs else 50.0


def classify_buildups(nearby: list[StrikeData], atm: float) -> list[dict]:
    """Flag strikes with a meaningfully large change in OI today, and
    label what that combination of side + OI-direction conventionally
    implies. This is descriptive (what's happening), not predictive."""
    flags = []
    for s in nearby:
        if s.ce_oi > 0 and abs(s.ce_oi_chg) / max(s.ce_oi, 1) * 100 >= BUILDUP_OI_CHG_THRESHOLD_PCT:
            label = "Fresh call writing (resistance building)" if s.ce_oi_chg > 0 else "Call unwinding (resistance weakening)"
            flags.append({"strike": s.strike, "side": "CE", "oi_chg": s.ce_oi_chg, "label": label})
        if s.pe_oi > 0 and abs(s.pe_oi_chg) / max(s.pe_oi, 1) * 100 >= BUILDUP_OI_CHG_THRESHOLD_PCT:
            label = "Fresh put writing (support building)" if s.pe_oi_chg > 0 else "Put unwinding (support weakening)"
            flags.append({"strike": s.strike, "side": "PE", "oi_chg": s.pe_oi_chg, "label": label})
    flags.sort(key=lambda f: abs(f["oi_chg"]), reverse=True)
    return flags


def iv_skew_read(nearby: list[StrikeData], atm: float) -> str:
    atm_strike = min(nearby, key=lambda s: abs(s.strike - atm))
    otm_calls = [s for s in nearby if s.strike > atm and s.ce_iv > 0]
    otm_puts = [s for s in nearby if s.strike < atm and s.pe_iv > 0]
    if not otm_calls or not otm_puts:
        return "Not enough OTM data either side to read skew."
    avg_call_iv = sum(s.ce_iv for s in otm_calls) / len(otm_calls)
    avg_put_iv = sum(s.pe_iv for s in otm_puts) / len(otm_puts)
    diff = avg_put_iv - avg_call_iv
    if diff >= IV_SKEW_NOTABLE_PCT:
        return f"Put skew: OTM puts pricier than OTM calls by {diff:.1f} IV pts — downside fear/hedging demand elevated."
    if diff <= -IV_SKEW_NOTABLE_PCT:
        return f"Call skew: OTM calls pricier than OTM puts by {abs(diff):.1f} IV pts — upside chase/hedging demand elevated."
    return f"Roughly flat skew (~{diff:+.1f} IV pts) — no strong directional hedging bias in IV."


def days_to_expiry(expiry_str: str) -> int:
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):
        try:
            exp_date = datetime.strptime(expiry_str, fmt)
            return max((exp_date.date() - datetime.now().date()).days, 0)
        except ValueError:
            continue
    return -1  # unknown format, caller should handle gracefully


# --------------------------------------------------------------------------
# Step 4: Strategy rules
# --------------------------------------------------------------------------

@dataclass
class StrategyIdea:
    name: str
    horizon: str       # "Intraday" | "Positional"
    structure: str
    rationale: str
    risk_note: str


def generate_strategies(
    snap: ChainSnapshot,
    atm: float,
    support: StrikeData,
    resistance: StrikeData,
    pcr: float,
    max_pain: float,
    flags: list[dict],
    skew_note: str,
    dte: int,
) -> tuple[str, list[StrategyIdea]]:

    ideas: list[StrategyIdea] = []
    range_width = resistance.strike - support.strike
    range_pct = (range_width / snap.underlying_value) * 100 if snap.underlying_value else 0

    fresh_call_writing = any(f["side"] == "CE" and f["oi_chg"] > 0 for f in flags)
    fresh_put_writing = any(f["side"] == "PE" and f["oi_chg"] > 0 for f in flags)
    call_unwind = any(f["side"] == "CE" and f["oi_chg"] < 0 for f in flags)
    put_unwind = any(f["side"] == "PE" and f["oi_chg"] < 0 for f in flags)

    # --- sentiment classification (transparent, rule-based) ---
    bullish_points = 0
    bearish_points = 0
    if pcr >= PCR_BULLISH:
        bullish_points += 1
    elif pcr <= PCR_BEARISH:
        bearish_points += 1
    if fresh_put_writing and not fresh_call_writing:
        bullish_points += 1
    if fresh_call_writing and not fresh_put_writing:
        bearish_points += 1
    if call_unwind and not put_unwind:
        bullish_points += 1
    if put_unwind and not call_unwind:
        bearish_points += 1
    if max_pain > atm:
        bullish_points += 0.5
    elif max_pain < atm:
        bearish_points += 0.5

    if bullish_points - bearish_points >= 1.5:
        sentiment = "Mildly bullish to bullish"
    elif bearish_points - bullish_points >= 1.5:
        sentiment = "Mildly bearish to bearish"
    else:
        sentiment = "Range-bound / no strong directional lean"

    # --- Intraday ideas ---
    if sentiment.startswith("Mildly bullish") or sentiment == "Mildly bullish to bullish":
        ideas.append(StrategyIdea(
            name="Intraday long bias",
            horizon="Intraday",
            structure=f"Buy ATM/slightly OTM CE near {atm:.0f}-{atm+ (resistance.strike-atm)*0.3:.0f}, "
                      f"or sell OTM PE around {support.strike:.0f} if you prefer defined premium collection.",
            rationale=f"PCR {pcr}, fresh put writing at {support.strike:.0f} and/or call unwinding above spot — "
                      f"writers positioning for upside or reducing resistance pressure.",
            risk_note=f"Hard stop on a close below {support.strike:.0f} (the support wall) — if it cracks, "
                      f"the bullish read is invalidated, not just 'pulling back'.",
        ))
    elif sentiment.startswith("Mildly bearish"):
        ideas.append(StrategyIdea(
            name="Intraday short bias",
            horizon="Intraday",
            structure=f"Buy ATM/slightly OTM PE near {atm:.0f}-{atm-(atm-support.strike)*0.3:.0f}, "
                      f"or sell OTM CE around {resistance.strike:.0f} if you prefer defined premium collection.",
            rationale=f"PCR {pcr}, fresh call writing at {resistance.strike:.0f} and/or put unwinding below spot — "
                      f"writers positioning for downside or reducing support pressure.",
            risk_note=f"Hard stop on a close above {resistance.strike:.0f} (the resistance wall) — if it breaks, "
                      f"the bearish read is invalidated.",
        ))
    else:
        ideas.append(StrategyIdea(
            name="Intraday range fade",
            horizon="Intraday",
            structure=f"Fade extremes between support {support.strike:.0f} and resistance {resistance.strike:.0f} — "
                      f"sell ATM-ish strangle/iron-fly style only if you're comfortable managing gamma into expiry, "
                      f"otherwise simply avoid directional premium-buying until one wall breaks.",
            rationale="No clear OI skew or fresh build-up dominance — chain suggests two-sided positioning, "
                      "i.e. the market itself hasn't picked a direction yet.",
            risk_note="Don't force a direction here. A breakout-with-volume through either wall (especially "
                      "with OI unwinding on that side) is your cue to flip from fading to following.",
        ))

    # --- Positional ideas ---
    if dte >= 0 and dte <= 2:
        ideas.append(StrategyIdea(
            name="Expiry-week positional",
            horizon="Positional",
            structure=f"With only {dte} day(s) to this expiry, prefer the NEXT expiry's chain for any "
                      f"positional (multi-day) structure — this week's chain is dominated by gamma/theta noise.",
            rationale="Near-dated OI walls shift fast into expiry and are unreliable for multi-day holds.",
            risk_note="Re-run this scanner against the next expiry (--expiry) before sizing a positional trade.",
        ))
    elif range_pct >= 2.5 and sentiment == "Range-bound / no strong directional lean":
        ideas.append(StrategyIdea(
            name="Iron Condor",
            horizon="Positional",
            structure=f"Short {resistance.strike:.0f} CE / Long {resistance.strike + (resistance.strike-atm)*0.5:.0f} CE "
                      f"  +  Short {support.strike:.0f} PE / Long {support.strike - (atm-support.strike)*0.5:.0f} PE.",
            rationale=f"Wide-ish OI-defined range (~{range_pct:.1f}% of spot) with no directional skew — "
                      f"classic premium-selling setup using the chain's own walls as your short strikes.",
            risk_note="Cap risk with the long legs (don't run naked strangles). Exit/adjust if spot tags either "
                      "short strike with fresh OI still building on that side — the wall may be about to move.",
        ))
    elif sentiment != "Range-bound / no strong directional lean":
        direction = "Bull" if "bullish" in sentiment.lower() else "Bear"
        ideas.append(StrategyIdea(
            name=f"{direction} Credit Spread",
            horizon="Positional",
            structure=(f"Sell {support.strike:.0f} PE / Buy {support.strike - (atm-support.strike)*0.6:.0f} PE"
                       if direction == "Bull" else
                       f"Sell {resistance.strike:.0f} CE / Buy {resistance.strike + (resistance.strike-atm)*0.6:.0f} CE"),
            rationale=f"{sentiment} read from OI positioning ({skew_note})",
            risk_note="Defined-risk credit spread sized so the short strike sits at the OI wall identified above — "
                      "if the wall gets taken out with fresh OI building beyond it, exit rather than average.",
        ))
    else:
        ideas.append(StrategyIdea(
            name="Iron Condor (tight range)",
            horizon="Positional",
            structure=f"Short strikes at the {support.strike:.0f}/{resistance.strike:.0f} OI walls, "
                      f"wings 1-2 strikes further out depending on premium received.",
            rationale="Range-bound chain read; selling the walls captures theta while the range holds.",
            risk_note="Range is narrow — size small, this can compress fast around any news/event.",
        ))

    return sentiment, ideas


# --------------------------------------------------------------------------
# Step 5: HTML report
# --------------------------------------------------------------------------

def render_html(
    snap: ChainSnapshot,
    atm: float,
    support: StrikeData,
    resistance: StrikeData,
    pcr: float,
    max_pain: float,
    flags: list[dict],
    skew_note: str,
    sentiment: str,
    ideas: list[StrategyIdea],
    nearby: list[StrikeData],
    dte: int,
) -> str:
    esc = html_lib.escape

    def fmt_oi(n: int) -> str:
        return f"{n:,}"

    def chg_class(n: int) -> str:
        return "pos" if n > 0 else ("neg" if n < 0 else "")

    rows = []
    for s in nearby:
        row_class = "atm-row" if abs(s.strike - atm) < 1e-6 else ""
        rows.append(f"""
        <tr class="{row_class}">
          <td class="num">{fmt_oi(s.ce_oi)}</td>
          <td class="num {chg_class(s.ce_oi_chg)}">{s.ce_oi_chg:+,}</td>
          <td class="num">{s.ce_iv:.1f}</td>
          <td class="num">{s.ce_ltp:.1f}</td>
          <td class="strike">{s.strike:.0f}</td>
          <td class="num">{s.pe_ltp:.1f}</td>
          <td class="num">{s.pe_iv:.1f}</td>
          <td class="num {chg_class(s.pe_oi_chg)}">{s.pe_oi_chg:+,}</td>
          <td class="num">{fmt_oi(s.pe_oi)}</td>
        </tr>""")

    flag_rows = "".join(
        f"""<tr>
              <td>{f['strike']:.0f}</td>
              <td>{f['side']}</td>
              <td class="{'pos' if f['oi_chg']>0 else 'neg'}">{f['oi_chg']:+,}</td>
              <td>{esc(f['label'])}</td>
            </tr>"""
        for f in flags[:10]
    ) or "<tr><td colspan='4' class='muted'>No strikes crossed the build-up threshold today.</td></tr>"

    idea_cards = "".join(f"""
      <div class="card idea">
        <div class="idea-head">
          <span class="badge {'badge-intra' if i.horizon=='Intraday' else 'badge-pos'}">{i.horizon}</span>
          <h3>{esc(i.name)}</h3>
        </div>
        <p><strong>Structure:</strong> {esc(i.structure)}</p>
        <p><strong>Why:</strong> {esc(i.rationale)}</p>
        <p class="risk"><strong>Risk note:</strong> {esc(i.risk_note)}</p>
      </div>"""
        for i in ideas
    )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{esc(snap.symbol)} Options Strategy Read — {esc(snap.expiry)}</title>
<style>
  :root {{
    --bg: #0b0e14; --panel: #131722; --panel2: #161b27; --border: #232838;
    --text: #e6e9f0; --muted: #8b93a7; --accent: #4f8cff;
    --pos: #2fbf71; --neg: #ef5b5b; --warn: #e8b339;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text); font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    margin: 0; padding: 24px; line-height: 1.5;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; color: var(--muted); margin: 0 0 20px; font-weight: 500; }}
  h3 {{ margin: 0; font-size: 15px; }}
  .meta {{ color: var(--muted); font-size: 12px; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }}
  .stat-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
  .stat-value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  .sentiment-card {{ grid-column: 1 / -1; background: var(--panel2); border-left: 3px solid var(--accent); }}
  .sentiment-card .stat-value {{ font-size: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--border); text-align: right; }}
  th {{ color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; }}
  td.strike {{ text-align: center; font-weight: 700; background: var(--panel2); }}
  tr.atm-row td {{ background: rgba(79,140,255,0.12); }}
  tr.atm-row td.strike {{ background: rgba(79,140,255,0.28); }}
  .num {{ font-variant-numeric: tabular-nums; }}
  .pos {{ color: var(--pos); }}
  .neg {{ color: var(--neg); }}
  .muted {{ color: var(--muted); text-align: left; }}
  .section {{ margin-bottom: 28px; }}
  .section-title {{ font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px; }}
  .idea-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
  .idea-head {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .badge {{ font-size: 10px; padding: 3px 8px; border-radius: 20px; text-transform: uppercase; letter-spacing: .04em; font-weight: 700; }}
  .badge-intra {{ background: rgba(232,179,57,0.18); color: var(--warn); }}
  .badge-pos {{ background: rgba(79,140,255,0.18); color: var(--accent); }}
  .idea p {{ margin: 6px 0; font-size: 13px; }}
  .idea .risk {{ color: var(--warn); }}
  .disclaimer {{ font-size: 11px; color: var(--muted); border-top: 1px solid var(--border); padding-top: 14px; margin-top: 30px; }}
  table.flags th, table.flags td {{ text-align: left; }}
  table.flags td:nth-child(1), table.flags td:nth-child(3) {{ text-align: right; }}
</style>
</head>
<body>
  <h1>{esc(snap.symbol)} Option Chain — Strategy Read</h1>
  <h2>Expiry: {esc(snap.expiry)} ({dte if dte>=0 else '?'} day(s) out) · Spot: {snap.underlying_value:,.2f}</h2>
  <div class="meta">NSE timestamp: {esc(snap.timestamp)} · Report generated: {generated_at}</div>

  <div class="grid">
    <div class="card"><div class="stat-label">Spot</div><div class="stat-value">{snap.underlying_value:,.2f}</div></div>
    <div class="card"><div class="stat-label">ATM Strike</div><div class="stat-value">{atm:.0f}</div></div>
    <div class="card"><div class="stat-label">PCR</div><div class="stat-value">{pcr}</div></div>
    <div class="card"><div class="stat-label">Max Pain</div><div class="stat-value">{max_pain:.0f}</div></div>
    <div class="card"><div class="stat-label">Support Wall</div><div class="stat-value pos">{support.strike:.0f}</div></div>
    <div class="card"><div class="stat-label">Resistance Wall</div><div class="stat-value neg">{resistance.strike:.0f}</div></div>
    <div class="card sentiment-card">
      <div class="stat-label">Rule-based sentiment read</div>
      <div class="stat-value">{esc(sentiment)}</div>
      <div style="color:var(--muted); font-size:12px; margin-top:6px;">{esc(skew_note)}</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Strategy Ideas</div>
    <div class="idea-grid">
      {idea_cards}
    </div>
  </div>

  <div class="section">
    <div class="section-title">Notable OI Build-ups Today (near ATM)</div>
    <table class="flags">
      <thead><tr><th>Strike</th><th>Side</th><th>ΔOI</th><th>Read</th></tr></thead>
      <tbody>{flag_rows}</tbody>
    </table>
  </div>

  <div class="section">
    <div class="section-title">Option Chain (±{len(nearby)//2} strikes around ATM)</div>
    <table>
      <thead><tr>
        <th>Call OI</th><th>Call ΔOI</th><th>Call IV</th><th>Call LTP</th>
        <th>Strike</th>
        <th>Put LTP</th><th>Put IV</th><th>Put ΔOI</th><th>Put OI</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>

  <div class="disclaimer">
    This report is a rule-based read of current option-chain positioning (OI, change in OI, IV, PCR, max pain).
    It is informational only, not investment advice or a trade recommendation — open interest reflects
    existing positioning, not a prediction of future price. Always size positions against your own risk
    management rules. Generated by nse_options_strategy.py.
  </div>
</body>
</html>"""


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(symbol: str, expiry: Optional[str], band: int, output: Path, offline_json: Optional[Path], dump_json: Optional[Path], debug: bool = False) -> Path:
    if offline_json:
        raw = json.loads(Path(offline_json).read_text())
        print(f"[i] Loaded offline chain from {offline_json}")
    else:
        print(f"[i] Connecting to NSE for {symbol}...")
        fetcher = NSESession(debug=debug)
        raw = fetcher.get_option_chain(symbol)
        print("[i] Fetched live chain.")
        if dump_json:
            Path(dump_json).write_text(json.dumps(raw))
            print(f"[i] Raw JSON cached to {dump_json}")

    snap = parse_chain(raw, symbol, expiry)
    strike_gap = infer_strike_gap(snap)
    atm = find_atm_strike(snap)
    support, resistance, nearby = support_resistance(snap, atm, band, strike_gap)
    pcr = compute_pcr(snap)
    max_pain = compute_max_pain(snap)
    flags = classify_buildups(nearby, atm)
    skew_note = iv_skew_read(nearby, atm)
    dte = days_to_expiry(snap.expiry)

    sentiment, ideas = generate_strategies(
        snap, atm, support, resistance, pcr, max_pain, flags, skew_note, dte
    )

    print(f"\n=== {symbol} — Expiry {snap.expiry} (spot {snap.underlying_value:,.2f}) ===")
    print(f"ATM: {atm:.0f} | PCR: {pcr} | Max Pain: {max_pain:.0f}")
    print(f"Support wall: {support.strike:.0f} (Put OI {support.pe_oi:,})")
    print(f"Resistance wall: {resistance.strike:.0f} (Call OI {resistance.ce_oi:,})")
    print(f"IV skew: {skew_note}")
    print(f"Sentiment read: {sentiment}\n")
    for idea in ideas:
        print(f"[{idea.horizon}] {idea.name}")
        print(f"  Structure: {idea.structure}")
        print(f"  Why: {idea.rationale}")
        print(f"  Risk: {idea.risk_note}\n")

    html_out = render_html(snap, atm, support, resistance, pcr, max_pain, flags, skew_note, sentiment, ideas, nearby, dte)
    output.write_text(html_out, encoding="utf-8")
    print(f"[i] HTML report written to {output}")
    return output


def main():
    ap = argparse.ArgumentParser(description="NSE option chain puller + rule-based strategy suggester")
    ap.add_argument("--symbol", default="NIFTY", choices=sorted(INDEX_SYMBOLS), help="Index symbol")
    ap.add_argument("--expiry", default=None, help="Expiry as shown by NSE, e.g. 24-Jun-2026. Defaults to nearest.")
    ap.add_argument("--band", type=int, default=8, help="Number of strikes either side of ATM to analyze")
    ap.add_argument("--output", default=None, help="Output HTML path")
    ap.add_argument("--offline", default=None, help="Path to a previously saved raw JSON dump (skip network)")
    ap.add_argument("--dump-json", default=None, help="Optionally cache the raw fetched JSON to this path")
    ap.add_argument("--debug", action="store_true", help="Print status code + body snippet for every request NSE makes")
    args = ap.parse_args()

    output = Path(args.output) if args.output else Path(
        f"{args.symbol.lower()}_options_strategy_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
    )

    try:
        run(
            symbol=args.symbol,
            expiry=args.expiry,
            band=args.band,
            output=output,
            offline_json=Path(args.offline) if args.offline else None,
            dump_json=Path(args.dump_json) if args.dump_json else None,
            debug=args.debug,
        )
    except NSEFetchError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
