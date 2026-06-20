#!/usr/bin/env python3
"""
nse_strategy_engine.py
========================

A quantitative options-strategy engine that builds a comprehensive list of
bullish/bearish/neutral strategies (plus cross-expiry "advanced" strategies
like calendar and diagonal spreads) from a LIVE option chain, and computes
for each one:

    - Probability of Profit (POP)   — via Black-Scholes risk-neutral pricing
    - Max profit / Max loss          — exact for same-expiry combos (computed
                                        from a fine numeric payoff grid), modeled
                                        for cross-expiry combos (Black-Scholes
                                        revaluation of the still-alive far leg)
    - Breakeven(s)
    - Funds needed                   — net premium outlay for debit strategies
    - Margin needed                  — exact-ish for defined-risk spreads
                                        (≈ max loss, per SEBI's post-2021 spread
                                        margin benefit), ESTIMATED for undefined-
                                        risk strategies (short straddle/strangle,
                                        naked legs) since real SPAN+exposure
                                        margin depends on volatility regime data
                                        this script doesn't have access to
    - A payoff curve (price -> P&L points) for charting

IMPORTANT — READ BEFORE TRUSTING THE NUMBERS
----------------------------------------------
- POP is a THEORETICAL risk-neutral probability derived from each strike's
  current implied volatility via Black-Scholes, NOT a guarantee or a
  backtested win rate. It assumes lognormal returns and constant IV — both
  are simplifications of how markets actually behave.
- Max profit/max loss/breakeven figures for same-expiry strategies (spreads,
  condors, straddles, backspreads, etc.) are computed directly from CURRENT
  live premiums and are exact given those premiums — they will differ from
  reality the moment premiums move.
- Calendar/diagonal spread economics assume the far leg's implied volatility
  stays constant between now and the near leg's expiry. In practice IV
  changes (often *expands* for the back month into binary events), so this
  is a simplification, not a forecast.
- Margin figures for DEFINED-RISK strategies use the common retail
  approximation margin ≈ max loss, which is broadly accurate post-SEBI's
  Feb 2021 spread-margin framework but is not a substitute for your broker's
  actual margin calculator.
- Margin figures for UNDEFINED-RISK strategies (short straddle/strangle,
  ratio backspreads, ladders) are ESTIMATES using a common retail heuristic
  (~12% of notional + premium collected). Real SPAN+exposure margin varies
  significantly with volatility and is NOT computed precisely here — always
  confirm with your broker before placing undefined-risk trades.
- This is informational output, not investment advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------
# Config — verify periodically, NSE rebases these every few quarters.
# Effective 27-Jan-2026 per NSE circular FAOP70616. If results look off,
# check https://www.nseindia.com for the latest contract specifications.
# --------------------------------------------------------------------------

LOT_SIZES = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
}

RISK_FREE_RATE = 0.065  # approx India short-term rate; only modestly affects POP/calendar pricing
UNDEFINED_RISK_MARGIN_PCT = 0.12  # retail heuristic: ~12% of notional contract value, see caveats above

# Strike-offset rules (in "number of strikes from ATM") used to pick legs for
# each template. All tunable — these are reasonable defaults, not the only
# valid construction. n=0 is ATM; positive n = OTM call / ITM put direction.
OFFSET_NEAR = 2   # "near the money" short strike for credit spreads
OFFSET_FAR = 5    # wing / long-leg strike for credit spreads, condors
OFFSET_WIDE = 8    # outer wing for strangles / wider condors
OFFSET_RATIO_LONG = 4  # long legs of a ratio backspread
OFFSET_DIAGONAL = 3   # far-leg strike offset for diagonal spreads


# --------------------------------------------------------------------------
# Black-Scholes pricer + risk-neutral probability
# --------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, sigma: float, r: float, option_type: str) -> float:
    """Theoretical option price. Falls back to intrinsic value at/after expiry
    or when sigma/T are degenerate (avoids division by zero)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if d1 is None:
        return max(S - K, 0.0) if option_type == "CE" else max(K - S, 0.0)
    if option_type == "CE":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def prob_above(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Risk-neutral probability that the underlying finishes ABOVE K at time T."""
    if K <= 0:
        return 1.0
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    _, d2 = _d1_d2(S, K, T, sigma, r)
    if d2 is None:
        return 1.0 if S > K else 0.0
    return norm_cdf(d2)


def norm_pdf(x: float) -> float:
    return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)


def greeks(S: float, K: float, T: float, sigma: float, r: float, option_type: str) -> dict:
    """Per-unit Greeks for a single option (not lot- or quantity-adjusted —
    that scaling happens when aggregating across a position)."""
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if d1 is None:
        delta = (1.0 if S > K else 0.0) if option_type == "CE" else (0.0 if S > K else -1.0)
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    pdf = norm_pdf(d1)
    delta = norm_cdf(d1) if option_type == "CE" else norm_cdf(d1) - 1.0
    gamma = pdf / (S * sigma * math.sqrt(T))
    vega = S * pdf * math.sqrt(T) / 100.0  # per 1% change in IV
    if option_type == "CE":
        theta_annual = -(S * pdf * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * norm_cdf(d2)
    else:
        theta_annual = -(S * pdf * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r * T) * norm_cdf(-d2)
    theta = theta_annual / 365.0  # per day
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


# --------------------------------------------------------------------------
# Leg representation
# --------------------------------------------------------------------------

@dataclass
class Leg:
    action: str          # 'BUY' or 'SELL'
    option_type: str     # 'CE' or 'PE' (ignored when instrument_type == 'FUTURES')
    strike: float
    premium: float        # entry premium per unit (live LTP at construction time); for FUTURES legs, the entry price
    expiry_tag: str = "near"   # 'near' or 'far' — which expiry this leg belongs to
    iv: float = 0.0        # entry IV (used to revalue 'far' legs at near-expiry)
    qty_lots: int = 1      # multiplier, e.g. 2 for the long legs of a ratio backspread
    instrument_type: str = "OPTION"   # 'OPTION' or 'FUTURES'

    @property
    def sign(self) -> int:
        return 1 if self.action == "BUY" else -1


@dataclass
class StrategyResult:
    name: str
    category: str          # 'Bullish' | 'Bearish' | 'Neutral'
    complexity: str         # 'Basic' | 'Advanced'
    description: str
    legs_desc: list[str]
    legs: list[dict]          # structured leg data for the frontend builder: action/option_type/strike/premium/iv/qty_lots/expiry_tag
    pop_pct: float
    max_profit: Optional[float]      # in rupees (already lot-multiplied); None if not computed
    max_profit_unlimited: bool
    max_loss: Optional[float]
    max_loss_unlimited: bool
    breakevens: list[float]
    funds_needed: float       # net debit outlay, rupees (0 for pure credit strategies)
    net_credit: float          # positive if credit received, negative if debit paid (rupees)
    margin_needed: float
    margin_is_estimate: bool
    payoff_curve: list[tuple]  # (price, pnl_rupees) sampled points for charting


# --------------------------------------------------------------------------
# Numeric payoff engine
# --------------------------------------------------------------------------

def _intrinsic(option_type: str, strike: float, S: float) -> float:
    return max(S - strike, 0.0) if option_type == "CE" else max(strike - S, 0.0)


def _leg_value_same_expiry(leg: "Leg", S: float) -> float:
    """A futures leg's 'value' at price S is just S itself (linear, no
    optionality) — its entry 'premium' is the price it was opened at, same
    role as an option's entry premium in the PnL formula below."""
    if leg.instrument_type == "FUTURES":
        return S
    return _intrinsic(leg.option_type, leg.strike, S)


def _payoff_same_expiry_per_unit(legs: list[Leg], S: float) -> float:
    total = 0.0
    for leg in legs:
        value = _leg_value_same_expiry(leg, S)
        leg_pnl = (value - leg.premium) if leg.action == "BUY" else (leg.premium - value)
        total += leg_pnl * leg.qty_lots
    return total


def _payoff_cross_expiry_per_unit(legs: list[Leg], S: float, T_gap_years: float, r: float) -> float:
    """PnL evaluated AT the near leg's expiry. 'near' legs are worth intrinsic
    value; 'far' legs still have T_gap_years of life left, valued via Black-
    Scholes at their entry IV (constant-IV assumption — see module docstring)."""
    total = 0.0
    for leg in legs:
        if leg.instrument_type == "FUTURES":
            value = S
        elif leg.expiry_tag == "near":
            value = _intrinsic(leg.option_type, leg.strike, S)
        else:
            value = bs_price(S, leg.strike, T_gap_years, leg.iv / 100.0, r, leg.option_type)
        leg_pnl = (value - leg.premium) if leg.action == "BUY" else (leg.premium - value)
        total += leg_pnl * leg.qty_lots
    return total


def _net_call_delta_sign(legs: list[Leg]) -> int:
    """Net count of long-minus-short calls (PLUS futures legs, which behave
    like an infinite-delta call/put combined — unbounded both ways) —
    determines whether payoff is truly unbounded as price -> infinity (only
    calls/futures can be unbounded upward on an index; pure puts are always
    floor-bounded at S=0)."""
    call_delta = sum(leg.sign * leg.qty_lots for leg in legs if leg.option_type == "CE" and leg.instrument_type == "OPTION")
    futures_delta = sum(leg.sign * leg.qty_lots for leg in legs if leg.instrument_type == "FUTURES")
    return call_delta + futures_delta


def analyze_strategy(
    legs: list[Leg],
    spot: float,
    lot_size: int,
    atm_iv: float,
    dte_near: int,
    dte_far: Optional[int] = None,
    r: float = RISK_FREE_RATE,
) -> tuple:
    """Returns (max_profit, max_profit_unlimited, max_loss, max_loss_unlimited,
    breakevens, pop_pct, payoff_curve_for_chart)."""

    is_cross_expiry = any(leg.expiry_tag == "far" for leg in legs)

    if is_cross_expiry:
        dte_far = dte_far or (dte_near + 25)
        T_gap = max(dte_far - dte_near, 1) / 365.0
        lo, hi, n_points = spot * 0.6, spot * 1.6, 800
        def payoff_fn(S):
            return _payoff_cross_expiry_per_unit(legs, S, T_gap, r)
    else:
        lo, hi, n_points = spot * 0.05, spot * 3.0, 2400
        def payoff_fn(S):
            return _payoff_same_expiry_per_unit(legs, S)

    step = (hi - lo) / (n_points - 1)
    grid = [lo + i * step for i in range(n_points)]
    # Payoff functions for these strategies are piecewise-linear with kinks
    # (peaks/troughs) exactly at each leg's strike — inject those explicitly
    # so max/min/breakeven aren't approximated by whatever uniform grid point
    # happens to land nearby.
    strike_points = sorted(set(leg.strike for leg in legs) | {spot})
    grid = sorted(set(grid) | set(strike_points))
    pnls = [payoff_fn(p) for p in grid]

    raw_max_profit = max(pnls)
    raw_max_loss = min(pnls)

    # Unbounded-upside detection only applies to same-expiry combos with a
    # net long/short call imbalance — calendars/diagonals are inherently
    # finite (both legs decay together; never flag them unlimited).
    if is_cross_expiry:
        max_profit_unlimited = False
        max_loss_unlimited = False
    else:
        net_calls = _net_call_delta_sign(legs)
        max_profit_unlimited = net_calls > 0
        max_loss_unlimited = net_calls < 0

    max_profit = None if max_profit_unlimited else raw_max_profit * lot_size
    max_loss = None if max_loss_unlimited else raw_max_loss * lot_size

    # Breakevens via sign-change detection + linear interpolation
    breakevens = []
    for i in range(len(grid) - 1):
        p1, pnl1 = grid[i], pnls[i]
        p2, pnl2 = grid[i + 1], pnls[i + 1]
        if pnl1 == 0:
            breakevens.append(p1)
        elif pnl1 * pnl2 < 0:
            frac = -pnl1 / (pnl2 - pnl1)
            breakevens.append(p1 + frac * (p2 - p1))
    # de-dupe breakevens that are essentially the same point
    deduped = []
    for b in breakevens:
        if not deduped or abs(b - deduped[-1]) > (spot * 0.001):
            deduped.append(b)
    breakevens = deduped

    # POP: integrate risk-neutral probability mass over profitable intervals,
    # using ATM IV as the representative volatility for the terminal price
    # distribution (a standard simplification — see module docstring).
    T = dte_near / 365.0
    sigma = atm_iv / 100.0
    boundaries = [0.0] + breakevens + [float("inf")]
    pop = 0.0
    for i in range(len(boundaries) - 1):
        seg_lo, seg_hi = boundaries[i], boundaries[i + 1]
        # sample the actual numeric curve at a representative point in this
        # segment to determine if it's a profit or loss zone
        if seg_hi == float("inf"):
            sample_price = grid[-1]
        elif seg_lo == 0.0:
            sample_price = grid[0]
        else:
            sample_price = (seg_lo + seg_hi) / 2.0
        is_profit = payoff_fn(sample_price) > 0
        if not is_profit:
            continue
        p_above_lo = 1.0 if seg_lo <= 0 else prob_above(spot, seg_lo, T, sigma, r)
        p_above_hi = 0.0 if seg_hi == float("inf") else prob_above(spot, seg_hi, T, sigma, r)
        pop += max(0.0, p_above_lo - p_above_hi)
    pop_pct = max(0.0, min(1.0, pop)) * 100.0

    # Downsample a clean curve for the frontend chart — centered window,
    # not the full wide detection range.
    chart_lo, chart_hi = spot * 0.85, spot * 1.15
    chart_n = 60
    chart_step = (chart_hi - chart_lo) / (chart_n - 1)
    chart_curve = []
    for i in range(chart_n):
        p = chart_lo + i * chart_step
        pnl = payoff_fn(p) * lot_size
        chart_curve.append((round(p, 1), round(pnl, 1)))

    return (
        max_profit, max_profit_unlimited,
        max_loss, max_loss_unlimited,
        breakevens, pop_pct, chart_curve,
    )


# --------------------------------------------------------------------------
# Strike lookup helper
# --------------------------------------------------------------------------

def _nearest_strike_data(strikes_by_value: dict, target: float):
    if target in strikes_by_value:
        return strikes_by_value[target]
    closest = min(strikes_by_value.keys(), key=lambda k: abs(k - target))
    return strikes_by_value[closest]


def _leg_from_chain(strikes_by_value: dict, target_strike: float, option_type: str, action: str,
                     expiry_tag: str = "near", qty_lots: int = 1) -> Leg:
    sd = _nearest_strike_data(strikes_by_value, target_strike)
    premium = sd.ce_ltp if option_type == "CE" else sd.pe_ltp
    iv = sd.ce_iv if option_type == "CE" else sd.pe_iv
    return Leg(action=action, option_type=option_type, strike=sd.strike, premium=premium,
               expiry_tag=expiry_tag, iv=iv, qty_lots=qty_lots)


def _leg_futures(spot: float, action: str, qty_lots: int = 1) -> Leg:
    """A futures leg, entered at the current SPOT price as an approximation
    of the actual futures price. Real index futures trade at a small
    premium/discount to spot (cost-of-carry basis) that narrows toward zero
    as expiry approaches — using spot directly is a simplification, not a
    live futures quote (this tool doesn't fetch the futures chain)."""
    return Leg(action=action, option_type="FUT", strike=spot, premium=spot,
               expiry_tag="near", iv=0.0, qty_lots=qty_lots, instrument_type="FUTURES")


def _fmt_strike(leg: Leg) -> str:
    if leg.instrument_type == "FUTURES":
        return f"{leg.action} FUTURES @ {leg.premium:.0f}"
    return f"{leg.action} {leg.strike:.0f} {leg.option_type}"


# --------------------------------------------------------------------------
# Strategy templates
# Each builder returns (legs, name, category, complexity, description,
# is_defined_risk) or None if the chain doesn't have enough usable strikes.
#
# is_defined_risk controls how MARGIN (not max loss — max loss is always
# computed honestly from the numeric engine regardless of this flag) gets
# approximated:
#   True  -> margin ≈ max loss. Valid for strategies where every short leg
#            is matched 1:1 by a long leg (verticals, iron condor/butterfly,
#            plain butterflies, condors, calendars, long straddle/strangle,
#            strap/strip, batman). SEBI's post-2021 spread-margin framework
#            reliably grants this benefit for genuinely matched positions.
#   False -> margin is an ESTIMATE (~12% of notional). Used for short
#            straddle/strangle, synthetic futures, and any UNEQUAL-ratio
#            structure (backspreads, ladders, ratio spreads, jade lizard,
#            broken-wing butterflies) — even when the numeric max loss is
#            technically finite, real broker RMS/SPAN systems generally do
#            NOT grant full matched-spread margin benefit to unequal-ratio
#            combos, so treating max-loss as the margin figure understates
#            real capital requirements. The 12% heuristic is deliberately
#            conservative (i.e. likely to overstate rather than understate).
# --------------------------------------------------------------------------

def _build_templates(strikes_by_value: dict, far_strikes_by_value: Optional[dict],
                      atm: float, gap: float, spot: float):
    T = []  # list of (legs, name, category, complexity, description, is_defined_risk)

    def K(n):
        return atm + n * gap

    # ================= BULLISH =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY")],
        "Long Call", "Bullish", "Basic",
        "Buy an ATM call. Profits from a strong upward move; loss capped at the premium paid.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "SELL")],
        "Bull Call Spread", "Bullish", "Basic",
        f"Buy ATM call, sell call {OFFSET_FAR} strikes higher. Defined risk/reward; cheaper than a naked long call.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "BUY")],
        "Bull Put Spread", "Bullish", "Basic",
        f"Sell put {OFFSET_NEAR} strikes below spot, buy put {OFFSET_FAR} strikes below. Credit received upfront; profits if price stays above the short strike.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_RATIO_LONG), "CE", "BUY", qty_lots=2)],
        "Call Ratio Backspread", "Bullish", "Advanced",
        f"Sell 1 ATM call, buy 2 calls {OFFSET_RATIO_LONG} strikes higher. Limited loss in the middle, unlimited profit on a strong rally, small credit/debit to enter.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "SELL")],
        "Bull Call Ladder", "Bullish", "Advanced",
        f"Buy ATM call, sell call {OFFSET_NEAR} strikes higher, sell another {OFFSET_FAR} strikes higher. "
        f"Reduces (or eliminates) the cost of a bull call spread, but exposes you to UNCAPPED loss if price "
        f"rallies hard past the upper strike — only do this if you're comfortable being net short calls above that level.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "SELL")],
        "Synthetic Long Future", "Bullish", "Advanced",
        "Buy ATM call, sell ATM put — replicates a long futures position synthetically. Unlimited profit AND unlimited loss; margin-heavy.",
        False,
    ))

    # ================= BEARISH =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "BUY")],
        "Long Put", "Bearish", "Basic",
        "Buy an ATM put. Profits from a strong downward move; loss capped at the premium paid (gain is capped too, since the index can't go below zero).",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "SELL")],
        "Bear Put Spread", "Bearish", "Basic",
        f"Buy ATM put, sell put {OFFSET_FAR} strikes lower. Defined risk/reward; cheaper than a naked long put.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "BUY")],
        "Bear Call Spread", "Bearish", "Basic",
        f"Sell call {OFFSET_NEAR} strikes above spot, buy call {OFFSET_FAR} strikes above. Credit received upfront; profits if price stays below the short strike.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_RATIO_LONG), "PE", "BUY", qty_lots=2)],
        "Put Ratio Backspread", "Bearish", "Advanced",
        f"Sell 1 ATM put, buy 2 puts {OFFSET_RATIO_LONG} strikes lower. Limited loss in the middle, large profit on a sharp fall, small credit/debit to enter.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "SELL")],
        "Bear Put Ladder", "Bearish", "Advanced",
        f"Buy ATM put, sell put {OFFSET_NEAR} strikes lower, sell another {OFFSET_FAR} strikes lower. "
        f"Reduces (or eliminates) the cost of a bear put spread, but exposes you to large loss if price "
        f"crashes past the lower strike — size accordingly.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "CE", "SELL")],
        "Synthetic Short Future", "Bearish", "Advanced",
        "Buy ATM put, sell ATM call — replicates a short futures position synthetically. Unlimited profit AND unlimited loss; margin-heavy.",
        False,
    ))

    # ================= NEUTRAL / RANGE (iron — mixed call+put) =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "BUY")],
        "Iron Condor", "Neutral", "Basic",
        f"Sell {OFFSET_NEAR}-strike-OTM put & call, buy {OFFSET_FAR}-strike-OTM wings on both sides. Defined risk; profits if price stays in the middle range.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(0), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "BUY")],
        "Iron Butterfly", "Neutral", "Basic",
        f"Sell ATM put & call, buy wings {OFFSET_FAR} strikes out on both sides. Higher credit than an iron condor, narrower profit zone.",
        True,
    ))

    # ================= NEUTRAL / RANGE (plain — single option type) =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "CE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "BUY")],
        "Long Call Butterfly", "Neutral", "Basic",
        f"Buy call {OFFSET_NEAR} strikes below ATM, sell 2 ATM calls, buy call {OFFSET_NEAR} strikes above ATM. "
        f"Low-cost, defined-risk bet that price pins close to ATM at expiry.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY")],
        "Long Put Butterfly", "Neutral", "Basic",
        f"Buy put {OFFSET_NEAR} strikes above ATM, sell 2 ATM puts, buy put {OFFSET_NEAR} strikes below ATM. "
        f"Same pin-risk bet as a call butterfly, priced via puts instead.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(0), "CE", "BUY", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL")],
        "Short Call Butterfly", "Neutral", "Advanced",
        f"Sell call {OFFSET_NEAR} strikes below ATM, buy 2 ATM calls, sell call {OFFSET_NEAR} strikes above ATM. "
        f"Inverse of the long butterfly — small credit, profits if price moves AWAY from ATM in either direction.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "BUY", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL")],
        "Short Put Butterfly", "Neutral", "Advanced",
        f"Sell put {OFFSET_NEAR} strikes above ATM, buy 2 ATM puts, sell put {OFFSET_NEAR} strikes below ATM. "
        f"Same away-from-ATM bet as a short call butterfly, priced via puts.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "BUY")],
        "Long Call Condor", "Neutral", "Advanced",
        f"Buy call {OFFSET_FAR} strikes below ATM, sell calls at {OFFSET_NEAR} strikes either side of ATM, buy call "
        f"{OFFSET_FAR} strikes above ATM. Like a butterfly but with a wider, flatter profit plateau instead of a single peak — "
        f"built entirely with calls.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_FAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "BUY")],
        "Long Put Condor", "Neutral", "Advanced",
        f"Same flat-top profit plateau as the call condor, built entirely with puts at the same four strikes.",
        True,
    ))

    # ================= NEUTRAL / VOLATILITY (long premium) =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "BUY")],
        "Long Straddle", "Neutral", "Basic",
        "Buy ATM call and put. Profits from a big move in EITHER direction; loss capped at total premium paid if price pins near ATM.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY")],
        "Long Strangle", "Neutral", "Basic",
        f"Buy call & put {OFFSET_NEAR} strikes out of the money each. Cheaper than a straddle, needs a bigger move to profit.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "SELL")],
        "Short Straddle", "Neutral", "Basic",
        "Sell ATM call and put. Highest premium collection, but UNDEFINED risk on both sides — needs active management.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_WIDE), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_WIDE), "PE", "SELL")],
        "Short Strangle", "Neutral", "Basic",
        f"Sell call & put {OFFSET_WIDE} strikes out of the money each. Wider profit zone than a straddle, still UNDEFINED risk.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(0), "PE", "BUY")],
        "Strap", "Bullish", "Advanced",
        "Buy 2 ATM calls + 1 ATM put. A long straddle with a bullish lean — profits more from an upside move than "
        "a downside one of the same size, while still benefiting from a volatility spike either way.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "BUY", qty_lots=2)],
        "Strip", "Bearish", "Advanced",
        "Buy 1 ATM call + 2 ATM puts. A long straddle with a bearish lean — profits more from a downside move "
        "than an upside one of the same size.",
        True,
    ))

    # ================= NEUTRAL — twin-peak / asymmetric advanced =================
    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "BUY")],
        "Jade Lizard", "Neutral", "Advanced",
        f"Sell ATM call, sell put {OFFSET_FAR} strikes below, buy call {OFFSET_FAR} strikes above as a cap. "
        f"If total credit exceeds the call-spread width, there's no upside risk at all — only downside to manage.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "PE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "PE", "BUY")],
        "Broken Wing Butterfly (Put)", "Neutral", "Advanced",
        f"Sell 2 ATM puts, buy 1 put {OFFSET_NEAR} strikes below and 1 put {OFFSET_FAR} strikes below (asymmetric wings). "
        f"Removes risk on the wider side at the cost of asymmetric exposure on the narrow side.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "CE", "BUY")],
        "Broken Wing Butterfly (Call)", "Neutral", "Advanced",
        f"Sell 2 ATM calls, buy 1 call {OFFSET_NEAR} strikes above and 1 call {OFFSET_FAR} strikes below (asymmetric wings). "
        f"Mirror of the put version, expressed with calls.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "BUY")],
        "Call Ratio Spread (Front)", "Neutral", "Advanced",
        f"Sell 2 calls near the money, buy 1 further ITM and 1 further OTM as protection. Income-focused, capped both sides.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY")],
        "Put Ratio Spread (Front)", "Neutral", "Advanced",
        f"Sell 2 puts near the money, buy 1 further ITM and 1 further OTM as protection. Mirror of the call ratio spread, priced via puts.",
        False,
    ))

    # ---------------- single-leg naked / cash-secured ----------------
    T.append((
        [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL")],
        "Naked Call", "Bearish", "Basic",
        f"Sell 1 call {OFFSET_NEAR} strikes OTM with no protection above. Simple income trade, but UNDEFINED risk if price rallies hard — the simplest bearish-income strategy and also one of the riskiest.",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL")],
        "Naked Put (Cash-Secured)", "Bullish", "Basic",
        f"Sell 1 put {OFFSET_NEAR} strikes OTM, setting aside enough cash/margin to buy the underlying if assigned. "
        f"Classic 'get paid to wait for a dip' income trade — large but floor-bounded risk if price craters.",
        False,
    ))

    # ---------------- guts ----------------
    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "BUY")],
        "Long Guts", "Neutral", "Advanced",
        f"Buy an ITM call ({OFFSET_NEAR} strikes below ATM) and an ITM put ({OFFSET_NEAR} strikes above ATM). "
        f"Like a long strangle but starting in-the-money on both legs — more expensive, smaller breakeven move needed, "
        f"never worth less than the difference between the two strikes at expiry.",
        True,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "PE", "SELL")],
        "Short Guts", "Neutral", "Advanced",
        f"Sell an ITM call and an ITM put ({OFFSET_NEAR} strikes either side of ATM). Collects more premium than a "
        f"short strangle, but UNDEFINED risk on both sides and a narrower profit zone since both legs start ITM.",
        False,
    ))

    # ---------------- arbitrage-style ----------------
    T.append((
        [_leg_from_chain(strikes_by_value, K(0), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "SELL"),
         _leg_from_chain(strikes_by_value, K(0), "PE", "SELL"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "PE", "BUY")],
        "Box Spread", "Neutral", "Advanced",
        f"A bull call spread + a bull put spread at the same two strikes. Combined payoff is FIXED regardless of "
        f"where price ends up — this isn't a directional or volatility bet, it's a synthetic lending/borrowing "
        f"structure. Only worth doing if the net price deviates from the strike-width's fair value (rare, and "
        f"transaction costs usually eat the edge for retail size).",
        True,
    ))

    # ---------------- futures-based (margin is always an estimate — see note in build_strategy_list) ----------------
    T.append((
        [_leg_futures(spot, "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL")],
        "Covered Call", "Neutral", "Advanced",
        f"Long futures + sell a call {OFFSET_NEAR} strikes OTM against it. Classic income overlay on a futures "
        f"position — collects premium, caps upside at the strike, but does NOT protect the downside (you still "
        f"carry full futures risk below). Uses spot as a proxy for the futures entry price (see disclaimers).",
        False,
    ))

    T.append((
        [_leg_futures(spot, "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY")],
        "Protective Put", "Bullish", "Advanced",
        f"Long futures + buy a put {OFFSET_NEAR} strikes OTM as insurance. Keeps unlimited upside while capping "
        f"downside loss below the put's strike — portfolio insurance, at the cost of the put's premium. "
        f"Uses spot as a proxy for the futures entry price (see disclaimers).",
        False,
    ))

    T.append((
        [_leg_from_chain(strikes_by_value, K(-OFFSET_FAR - OFFSET_NEAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR), "PE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(-OFFSET_FAR + OFFSET_NEAR), "PE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR - OFFSET_NEAR), "CE", "BUY"),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR), "CE", "SELL", qty_lots=2),
         _leg_from_chain(strikes_by_value, K(OFFSET_FAR + OFFSET_NEAR), "CE", "BUY")],
        "Batman", "Neutral", "Advanced",
        f"Two long butterflies — a put butterfly centered {OFFSET_FAR} strikes below ATM, a call butterfly centered "
        f"{OFFSET_FAR} strikes above ATM. Twin profit peaks either side of spot with a dip near ATM — for when you "
        f"expect a moderate move in EITHER direction but want to avoid paying up for a pin right at the money. "
        f"(Construction is one common version of this strategy; there's no single universal definition.)",
        True,
    ))

    # ================= ADVANCED — cross-expiry (calendar / diagonal) =================
    if far_strikes_by_value:
        T.append((
            [_leg_from_chain(strikes_by_value, K(0), "CE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(0), "CE", "BUY", "far")],
            "Call Calendar Spread", "Neutral", "Advanced",
            "Sell near-week ATM call, buy same-strike call in the next expiry. Profits from time decay on the short "
            "leg outpacing the long leg; best when price pins near the strike through the near expiry.",
            True,
        ))

        T.append((
            [_leg_from_chain(strikes_by_value, K(0), "PE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(0), "PE", "BUY", "far")],
            "Put Calendar Spread", "Neutral", "Advanced",
            "Sell near-week ATM put, buy same-strike put in the next expiry. Same time-decay logic as the call "
            "calendar, expressed via puts.",
            True,
        ))

        T.append((
            [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(OFFSET_DIAGONAL), "CE", "BUY", "far")],
            "Diagonal Call Spread", "Bullish", "Advanced",
            f"Sell near-week call {OFFSET_NEAR} strikes OTM, buy a further-OTM call ({OFFSET_DIAGONAL} strikes) "
            f"in the next expiry. Combines time decay with a mild bullish tilt.",
            True,
        ))

        T.append((
            [_leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(-OFFSET_DIAGONAL), "PE", "BUY", "far")],
            "Diagonal Put Spread", "Bearish", "Advanced",
            f"Sell near-week put {OFFSET_NEAR} strikes OTM, buy a further-OTM put ({OFFSET_DIAGONAL} strikes) "
            f"in the next expiry. Combines time decay with a mild bearish tilt.",
            True,
        ))

        T.append((
            [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(OFFSET_NEAR), "CE", "BUY", "far"),
             _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(-OFFSET_NEAR), "PE", "BUY", "far")],
            "Double Calendar", "Neutral", "Advanced",
            f"Call calendar + put calendar at strikes {OFFSET_NEAR} away on each side. Wider profit zone than a "
            f"single calendar, at roughly double the cost.",
            True,
        ))

        T.append((
            [_leg_from_chain(strikes_by_value, K(OFFSET_NEAR), "CE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(OFFSET_DIAGONAL), "CE", "BUY", "far"),
             _leg_from_chain(strikes_by_value, K(-OFFSET_NEAR), "PE", "SELL", "near"),
             _leg_from_chain(far_strikes_by_value, K(-OFFSET_DIAGONAL), "PE", "BUY", "far")],
            "Double Diagonal", "Neutral", "Advanced",
            f"Diagonal call spread + diagonal put spread combined — near-week strikes {OFFSET_NEAR} OTM each side, "
            f"far-week strikes {OFFSET_DIAGONAL} OTM each side. Wider profit zone than a double calendar since the "
            f"long legs sit further out, at a similar cost.",
            True,
        ))

    return T


# --------------------------------------------------------------------------
# Master builder
# --------------------------------------------------------------------------

def build_strategy_list(
    symbol: str,
    near_strikes: list,             # list of StrikeData-like objects (strike, ce_ltp, ce_iv, pe_ltp, pe_iv)
    atm: float,
    strike_gap: float,
    spot: float,
    dte_near: int,
    atm_iv: float,
    far_strikes: Optional[list] = None,
    dte_far: Optional[int] = None,
) -> list[StrategyResult]:
    lot_size = LOT_SIZES.get(symbol, 50)

    strikes_by_value = {s.strike: s for s in near_strikes}
    far_strikes_by_value = {s.strike: s for s in far_strikes} if far_strikes else None

    templates = _build_templates(strikes_by_value, far_strikes_by_value, atm, strike_gap, spot)

    results: list[StrategyResult] = []
    for legs, name, category, complexity, description, is_defined_risk in templates:
        # net premium: positive = credit received, negative = debit paid (per unit).
        # FUTURES legs are deliberately excluded here — opening a futures
        # position doesn't require paying its notional value upfront the
        # way buying/selling an option does, only posting margin (handled
        # separately below). Including it here would make "funds needed"
        # for Covered Call/Protective Put nonsensically report the full
        # notional value of a futures contract as required upfront cash.
        net_premium = sum(
            (leg.premium if leg.action == "SELL" else -leg.premium) * leg.qty_lots
            for leg in legs if leg.instrument_type == "OPTION"
        )
        net_credit_rupees = net_premium * lot_size
        has_futures_leg = any(leg.instrument_type == "FUTURES" for leg in legs)

        (max_profit, mp_unlim, max_loss, ml_unlim, breakevens, pop_pct, chart_curve) = analyze_strategy(
            legs, spot, lot_size, atm_iv, dte_near, dte_far,
        )

        funds_needed = max(0.0, -net_credit_rupees)  # only debit strategies need upfront funds

        # MARGIN — this is intentionally NOT just "use max loss whenever it's
        # finite". Two genuinely different situations look the same if you
        # only check whether max_loss is a finite number:
        #   1. A truly matched spread (every short leg offset 1:1 by a long
        #      leg) — here margin really does collapse to max loss, and SEBI's
        #      post-2021 framework reliably grants that benefit.
        #   2. An unequal-ratio structure (backspreads, ladders, jade lizard,
        #      broken-wing butterflies) — max loss is still a real, finite
        #      number, but broker RMS/SPAN systems generally do NOT extend
        #      full spread-margin treatment to these, because the "extra"
        #      short leg isn't cleanly hedged at every price point the way
        #      it is in a vertical or iron condor. Treating max-loss as
        #      margin here would understate real capital needs.
        # `is_defined_risk` (set per-template above) distinguishes these.
        # The unlimited-risk check is a hard safety override regardless of
        # how a template was tagged.
        if ml_unlim or not is_defined_risk or has_futures_leg:
            notional = spot * lot_size
            margin_needed = UNDEFINED_RISK_MARGIN_PCT * notional
            margin_is_estimate = True
        else:
            # Fully-hedged strategy: margin ≈ max loss. For a pure debit
            # spread this is mathematically identical to funds_needed
            # anyway (worst case = losing the whole premium), so this
            # naturally stays consistent with the funds figure above.
            margin_needed = abs(max_loss) if max_loss is not None else funds_needed
            margin_is_estimate = False

        results.append(StrategyResult(
            name=name,
            category=category,
            complexity=complexity,
            description=description,
            legs_desc=[_fmt_strike(leg) + (f" ({leg.expiry_tag})" if leg.expiry_tag == "far" else "") for leg in legs],
            legs=[{
                "action": leg.action, "option_type": leg.option_type, "strike": leg.strike,
                "premium": leg.premium, "iv": leg.iv, "qty_lots": leg.qty_lots,
                "expiry_tag": leg.expiry_tag, "instrument_type": leg.instrument_type,
            } for leg in legs],
            pop_pct=round(pop_pct, 1),
            max_profit=round(max_profit, 0) if max_profit is not None else None,
            max_profit_unlimited=mp_unlim,
            max_loss=round(max_loss, 0) if max_loss is not None else None,
            max_loss_unlimited=ml_unlim,
            breakevens=[round(b, 1) for b in breakevens],
            funds_needed=round(funds_needed, 0),
            net_credit=round(net_credit_rupees, 0),
            margin_needed=round(margin_needed, 0),
            margin_is_estimate=margin_is_estimate,
            payoff_curve=chart_curve,
        ))

    results.sort(key=lambda r: r.pop_pct, reverse=True)
    return results
