"""
trump_impact.py
===============
Companion to key_figures.py.

WHAT THIS DOES (and does NOT do)
--------------------------------
It does NOT predict what Trump will announce from planetary positions. Nothing
can. What it does, honestly:

  1. LIVE ISSUE TRACKER  — the menu of open US-India decision points. This is
     the closest thing to "what is he likely to announce": the pending items
     ARE the surface area. You maintain it; the tool structures it.

  2. IMPACT CLASSIFIER   — paste any real Trump headline, get an India-market
     playbook: direction, affected NSE sectors, transmission channel, a rough
     magnitude band calibrated to real reactions, and an options posture.

  3. ASTRO OVERLAY (thin, caveated) — merges key_figures.upcoming_windows()
     dates so you know which sessions to be flat / hedged into. Low confidence
     by construction; treat as "reduce size / widen stops," never as a signal.

MAGNITUDE CALIBRATION (real anchors, Feb 2026)
  - Deal / tariff cut 25%->18% (2 Feb 2026): Nifty +4.86% open, +2.5% close;
    ~Rs 20 lakh cr added; GIFT Nifty +800pts overnight. Led by Adani cx,
    ports, financials.
  - Escalation to 50% peak (Aug 2025): India worst major-EM performer, record
    FII outflows, INR Asia's weakest currency.
  - Solar CVD 126% (25 Feb 2026): sector-specific shock, solar names only.

Everything is JSON-serialisable. Trump/Truth-Social announcements land in US
hours = IST night => the dominant structure is a GIFT Nifty gap -> Nifty
gap-open. The playbooks are written around that.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Optional

# --------------------------------------------------------------------------
# Sector / instrument shorthands (NSE)
# --------------------------------------------------------------------------
IT      = "NIFTY IT (TCS, INFY, HCLTECH, WIPRO, TECHM)"
PHARMA  = "NIFTY PHARMA (SUNPHARMA, DRREDDY, CIPLA, AUROPHARMA)"
BANK    = "BANK NIFTY (HDFCBANK, ICICIBANK, SBIN, AXISBANK)"
OMC     = "OMCs / energy (IOC, BPCL, HPCL, ONGC, RELIANCE)"
METALS  = "NIFTY METAL (TATASTEEL, JSWSTEEL, HINDALCO)"
AUTO    = "NIFTY AUTO (component exporters: BHARATFORGE, MOTHERSON)"
TEXTILE = "Textiles / apparel / leather exporters"
GEMS    = "Gems, jewellery & diamonds (TITAN, KALYANKJIL)"
SOLAR   = "Solar / renewables (WAAREE, PREMIERENE, adani green)"
INFRA   = "Infra / ports (ADANIENT, ADANIPORTS, GMR)"
CHEM    = "Specialty chemicals (exporters)"
VIX     = "INDIA VIX / index gamma"
INR     = "USDINR (rupee)"

# --------------------------------------------------------------------------
# Impact profiles per announcement category
# direction: +1 bullish India, -1 bearish India, 0 = volatility/mixed
# sev_band: rough single-session index move by severity 1/2/3 (percent, abs)
# --------------------------------------------------------------------------

@dataclass
class Profile:
    category: str
    direction: int
    channel: str
    sectors_hit: list            # (sector, sign) sign +1 benefits / -1 hurt
    sev_band: tuple              # (mild, moderate, severe) abs % index move
    inr_effect: str
    options_posture: str
    keywords: list = field(default_factory=list)

PROFILES = {
    "tariff_hike": Profile(
        category="Tariff hike / punitive duty on India",
        direction=-1,
        channel="US-revenue earnings hit + FII outflow + INR weakness; risk-off gap-down",
        sectors_hit=[(IT, -1), (PHARMA, -1), (TEXTILE, -1), (GEMS, -1),
                     (AUTO, -1), (CHEM, -1), (INR, -1)],
        sev_band=(1.0, 3.0, 5.0),
        inr_effect="INR weakens (import bill up); partial FX cushion for IT/pharma exporters",
        options_posture=("Overnight gap-down risk. Bearish: Nifty/BankNifty put "
                         "spreads, long India VIX, put ratios on NIFTY IT. If already "
                         "long, hedge overnight — Trump posts in IST night, you can't react."),
        keywords=["tariff", "duty", "levy", "reciprocal", "punitive", "raise tariff",
                  "impose", "sanction india", "penalty"]),

    "tariff_cut": Profile(
        category="Tariff cut / trade deal / de-escalation",
        direction=+1,
        channel="Re-rating + FII re-entry + INR relief; violent short-covering gap-up",
        sectors_hit=[(INFRA, +1), (BANK, +1), (IT, +1), (TEXTILE, +1),
                     (GEMS, +1), (AUTO, +1), (INR, +1)],
        sev_band=(1.5, 3.0, 5.0),
        inr_effect="INR strengthens on flows/relief",
        options_posture=("Gap-up + IV crush. Sell puts / bull call spreads pre-open if "
                         "you can, but post-event IV collapse punishes long straddles. "
                         "Beware chasing the gap — Feb-2026 deal opened +4.9%, closed +2.5%."),
        keywords=["trade deal", "lower tariff", "reduce tariff", "cut tariff",
                  "agreement", "de-escalat", "remove tariff", "bta", "framework"]),

    "h1b_visa": Profile(
        category="H1B / visa / immigration restriction",
        direction=-1,
        channel="Onsite delivery cost up -> IT margin compression",
        sectors_hit=[(IT, -1)],
        sev_band=(0.5, 1.5, 3.0),
        inr_effect="Neutral-to-mild INR",
        options_posture="Bearish NIFTY IT specifically; index impact muted unless severe.",
        keywords=["h1b", "h-1b", "visa", "immigration", "green card", "opt", "wage rule"]),

    "russia_oil": Profile(
        category="Russia oil / secondary sanctions / energy sourcing",
        direction=-1,
        channel="Secondary-sanction threat = risk-off; OMC pricing/margin risk on sourcing shift",
        sectors_hit=[(OMC, -1), (INR, -1)],
        sev_band=(0.5, 2.0, 4.0),
        inr_effect="INR weakens if import bill / crude rises",
        options_posture="OMC-specific bearish + index risk-off; watch crude alongside.",
        keywords=["russian oil", "russia oil", "secondary sanction", "crude",
                  "venezuela", "energy purchase", "opec"]),

    "pharma_specific": Profile(
        category="Pharma tariff / US drug pricing action",
        direction=-1,
        channel="Indian generics = large share of US scripts; revenue at risk (often exempted, so check)",
        sectors_hit=[(PHARMA, -1)],
        sev_band=(0.5, 2.0, 4.0),
        inr_effect="Neutral",
        options_posture="NIFTY PHARMA directional. Note: pharma was EXEMPT in the Feb-26 deal — a reversal would be the surprise.",
        keywords=["pharma", "drug pricing", "generic", "medicine", "fda", "biosecure"]),

    "tech_positive": Profile(
        category="Tech / GPU / data-center / semiconductor cooperation",
        direction=+1,
        channel="Tech-transfer + capex tailwind (deal explicitly covers GPUs/data centers)",
        sectors_hit=[(IT, +1), (INFRA, +1)],
        sev_band=(0.5, 1.5, 3.0),
        inr_effect="Mild positive",
        options_posture="Bullish IT / data-center plays; usually a slower burn, not a gap.",
        keywords=["gpu", "data center", "semiconductor", "chip", "ai cooperation",
                  "technology partnership", "nvidia"]),

    "fed_dollar": Profile(
        category="Fed pressure / rate jawboning / dollar policy",
        direction=0,
        channel="Weaker USD/DXY -> EM inflows (India +); but policy-uncertainty vol",
        sectors_hit=[(BANK, +1), (INR, +1), (VIX, -1)],
        sev_band=(0.3, 1.0, 2.5),
        inr_effect="USD weakness supports INR & FII inflows",
        options_posture="Rate-sensitives (BankNifty, NBFC, realty) directional with DXY. Mixed sign — read the dollar.",
        keywords=["federal reserve", "fed", "powell", "rate cut", "interest rate",
                  "dollar", "dxy", "jerome"]),

    "volatility_generic": Profile(
        category="Generic geopolitical / market-moving post (unclassified)",
        direction=0,
        channel="Uncertainty spike; overnight gap either way",
        sectors_hit=[(VIX, -1)],
        sev_band=(0.3, 1.0, 2.5),
        inr_effect="Uncertain",
        options_posture="Long vol into known event windows; avoid naked short gamma overnight.",
        keywords=[]),
}

# --------------------------------------------------------------------------
# Live issue tracker — the honest "what might he announce" menu.
# Seed from current open US-India items; you update as things resolve.
# status: open | watch | resolved ; you set your own subjective p (0-1).
# --------------------------------------------------------------------------

@dataclass
class OpenIssue:
    name: str
    category: str            # must key into PROFILES
    status: str              # open | watch | resolved
    prob: float              # your subjective probability it moves near-term
    note: str = ""

LIVE_ISSUES = [
    OpenIssue("BTA (full bilateral trade agreement) next round", "tariff_cut",
              "open", 0.55,
              "Interim deal done Feb-26 at 18%; BTA still negotiating remaining lines. Positive-skew catalyst."),
    OpenIssue("Solar CVD 126% (crystalline PV cells/modules)", "tariff_hike",
              "watch", 0.40,
              "Preliminary CVD 25-Feb-26; final determination pending. Solar-name specific, not index-wide."),
    OpenIssue("Pharma tariff carve-out durability", "pharma_specific",
              "watch", 0.25,
              "Generics currently EXEMPT. Any reversal = downside surprise for NIFTY PHARMA."),
    OpenIssue("H1B / visa wage rule", "h1b_visa",
              "open", 0.35, "Recurrent IT-sector overhang; episodic headlines."),
    OpenIssue("Russian-oil compliance / secondary sanctions", "russia_oil",
              "watch", 0.30,
              "India committed to wind down Russian crude; slippage could re-trigger punitive threat."),
    OpenIssue("Digital-trade / data-localisation rules", "tariff_hike",
              "open", 0.20, "Non-tariff barrier item in the BTA; slow-moving."),
    OpenIssue("GPU / data-center tech cooperation rollout", "tech_positive",
              "open", 0.30, "Explicit deal component; positive drip catalyst for IT/infra."),
]

# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

_CUT_WORDS  = ("cut", "lower", "reduce", "drop", "slash", "remove", "deal",
               "agreement", "de-escalat", "ease", "rollback", "roll back",
               "exempt", "waive", "further reduc", "finalize", "finalise")
_HIKE_WORDS = ("hike", "raise", "impose", "higher", "increase", "punitive",
               "additional", "new tariff", "slap", "penalty", "threaten",
               "secondary sanction")

def classify(headline: str) -> str:
    """Keyword-match a headline to a category. Returns category key.

    Special handling: the bare word 'tariff' is direction-ambiguous, so tariff
    headlines are routed by surrounding cut/hike language before generic
    keyword scoring runs."""
    h = headline.lower()

    # --- tariff direction disambiguation (highest priority) -------------
    tariffish = any(w in h for w in ("tariff", "duty", "levy", "trade deal",
                                     "reciprocal", "bta", "trade agreement"))
    if tariffish:
        cut = any(w in h for w in _CUT_WORDS)
        hike = any(w in h for w in _HIKE_WORDS)
        if cut and not hike:
            return "tariff_cut"
        if hike and not cut:
            return "tariff_hike"
        # both or neither present -> fall through to scoring, but keep tariff
        # bias by nudging below

    # --- generic weighted keyword scoring -------------------------------
    best, best_hits = "volatility_generic", 0
    for key, prof in PROFILES.items():
        # longer keyword phrases count more (specificity)
        hits = sum(len(kw.split()) for kw in prof.keywords if kw in h)
        if hits > best_hits:
            best, best_hits = key, hits
    return best


def impact_profile(category: str, severity: int = 2,
                   headline: str = "") -> dict:
    """Return the India-market playbook for a category at severity 1/2/3."""
    prof = PROFILES.get(category, PROFILES["volatility_generic"])
    severity = max(1, min(3, severity))
    band = prof.sev_band[severity - 1]
    move = f"{'+' if prof.direction > 0 else '-' if prof.direction < 0 else '±'}{band:.1f}% Nifty gap (est.)"
    return {
        "headline": headline,
        "category": prof.category,
        "direction": {1: "BULLISH India", -1: "BEARISH India", 0: "VOL / MIXED"}[prof.direction],
        "est_index_move": move,
        "transmission": prof.channel,
        "sectors": [{"name": s, "sign": "benefits" if sg > 0 else "hurt"}
                    for s, sg in prof.sectors_hit],
        "inr": prof.inr_effect,
        "options_posture": prof.options_posture,
        "structure_note": ("Truth-Social post lands in IST night -> GIFT Nifty "
                           "gap -> Nifty gap-open. Position the night before, not at 9:15."),
    }


def analyse_headline(headline: str, severity: int = 2) -> dict:
    """One-shot: classify a real announcement and return its playbook."""
    cat = classify(headline)
    return impact_profile(cat, severity, headline)


def issue_board(issues: Optional[list] = None) -> list:
    """Sorted live menu of open items with their impact skew."""
    issues = issues or LIVE_ISSUES
    rows = []
    for it in issues:
        prof = PROFILES[it.category]
        rows.append({
            "issue": it.name, "status": it.status, "prob": it.prob,
            "skew": {1: "+", -1: "-", 0: "±"}[prof.direction],
            "category": prof.category, "note": it.note,
        })
    return sorted(rows, key=lambda r: (-r["prob"], r["status"]))


def overlay_astro_windows(figure, days: int = 120) -> dict:
    """
    Thin, clearly-caveated merge with key_figures windows. Returns dates the
    astro layer flags as high-friction, so you can choose to be flat/hedged
    into them. NOT a directional signal.
    """
    try:
        w = figure.upcoming_windows(days=days)
    except Exception as e:
        return {"error": f"key_figures not available: {e}"}
    slow = [t for t in w["transit_perfections"]
            if t["transit"] in ("Saturn", "Rahu", "Ketu", "Jupiter")
            and t["min_orb"] <= 1.0]
    return {
        "caveat": ("LOW CONFIDENCE. These are astro 'watch' dates only. Use to "
                   "reduce size / widen hedges into the session, never as a trade trigger."),
        "dasha_changes": w["dasha_changes"],
        "tight_slow_transits": slow[:15],
    }


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("=== LIVE ISSUE BOARD (the 'what might he announce' menu) ===")
    for r in issue_board():
        print(f"  [{r['skew']}] p={r['prob']:.2f} {r['status']:8} {r['issue']}")

    print("\n=== HEADLINE CLASSIFIER DEMOS ===")
    tests = [
        "Trump: reciprocal tariff on India to be raised over Russian oil",
        "US and India finalize BTA, tariffs to drop further",
        "New H-1B wage floor announced by administration",
        "Trump pressures Powell for immediate rate cut",
        "US GPU export cooperation with India expanded",
    ]
    for t in tests:
        p = analyse_headline(t, severity=2)
        print(f"\n  > {t}")
        print(f"    {p['category']}  |  {p['direction']}  |  {p['est_index_move']}")
        print(f"    posture: {p['options_posture'][:90]}...")
