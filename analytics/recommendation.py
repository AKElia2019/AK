"""
btc_dashboard.analytics.recommendation
Bias / setup / conviction / options-structure recommendation layer.

Pipeline order (per the project architecture):

    normalization → scoring → smoothing → regime → recommendation

Inputs are smoothed final scores per timeframe (1h, 4h), the 4h regime
assessment, and (optionally) ATM IV plus the primary timeframe driving
the trade. Outputs:

  • directional bias            (long / short / no_trade)
  • setup label                 (trend / squeeze / mean_reversion / none)
  • conviction tier             (high / medium / low / none)
  • 3-bullet explanation
  • options structure suggestion (strike-delta range, expiry window,
                                  size multiplier with IV haircut)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from analytics.regime import RegimeAssessment


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLDS  (easy to tune)
# ─────────────────────────────────────────────────────────────────────────────
GATE_BAND_LO: float = -50.0
GATE_BAND_HI: float = +50.0

LONG_4H_MIN:  float = 60.0
LONG_1H_MIN:  float = 50.0
SHORT_4H_MAX: float = -60.0
SHORT_1H_MAX: float = -50.0

# Conviction tiers (absolute score levels that BOTH timeframes must exceed)
HIGH_4H_ABS: float = 80.0
HIGH_1H_ABS: float = 70.0
MED_4H_ABS:  float = 70.0
MED_1H_ABS:  float = 60.0

# ── Options structure (absolute deltas in percent — e.g. 25 means 0.25Δ)
HIGH_DELTA_LOW:    float = 25.0
HIGH_DELTA_HIGH:   float = 35.0
SQUEEZE_DELTA_LOW: float = 5.0
SQUEEZE_DELTA_HIGH: float = 15.0
SPREAD_LONG_DELTA_LOW:  float = 25.0
SPREAD_LONG_DELTA_HIGH: float = 35.0
SPREAD_SHORT_DELTA_LOW:  float = 10.0
SPREAD_SHORT_DELTA_HIGH: float = 20.0

# ── Expiry windows (calendar days)
EXPIRY_1H_MIN: int = 5
EXPIRY_1H_MAX: int = 7
EXPIRY_4H_MIN: int = 7
EXPIRY_4H_MAX: int = 14

# ── IV size haircut bands
IV_HIGH_PCT:     float = 80.0     # ≥ 80% ATM IV → 0.50× size
IV_ELEVATED_PCT: float = 60.0     # ≥ 60% ATM IV → 0.75× size


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RecommendationInputs:
    score_1h: float                                  # smoothed final score [-100, +100]
    score_4h: float                                  # smoothed final score [-100, +100]
    regime_4h: Optional[RegimeAssessment] = None     # regime classification on 4h
    stable_1h: bool = True                           # smoothing-layer stability flag
    stable_4h: bool = True
    atm_iv_pct: Optional[float] = None               # ATM IV in %, drives size haircut
    primary_timeframe: str = "4h"                    # "1h" or "4h" — drives expiry window


@dataclass(frozen=True)
class OptionsSuggestion:
    structure: str                                   # "call" | "put" | "call_spread" | "put_spread" | "none"
    long_leg_delta: tuple[float, float]              # (low, high) absolute Δ in %
    short_leg_delta: Optional[tuple[float, float]]   # spreads only; None for single-leg
    expiry_days: tuple[int, int]                     # (min, max) calendar days
    size_multiplier: float                           # in [0.0, 1.0]; product of structure base × IV haircut
    notes: list[str]                                 # rationale lines


@dataclass(frozen=True)
class Recommendation:
    bias: str                  # "long" | "short" | "no_trade"
    setup: str                 # "trend" | "squeeze" | "mean_reversion" | "none"
    conviction: str            # "high" | "medium" | "low" | "none"
    explanation: list[str]     # 3 bullet points
    gatekeeper_passed: bool
    blocked_reasons: list[str]
    options_suggestion: Optional[OptionsSuggestion] = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sign(x: float) -> int:
    if x is None or not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


# ─────────────────────────────────────────────────────────────────────────────
# 1 — GATEKEEPER
# ─────────────────────────────────────────────────────────────────────────────
def gatekeeper(inputs: RecommendationInputs) -> tuple[bool, list[str]]:
    """Return (passed, blocked_reasons).

    Two block conditions:
      • Either timeframe's score is inside the [-50, +50] no-edge band.
      • The 1h and 4h scores disagree on direction (different signs).
    """
    reasons: list[str] = []

    if GATE_BAND_LO <= inputs.score_4h <= GATE_BAND_HI:
        reasons.append(
            f"4h score {inputs.score_4h:+.1f} inside the no-edge band "
            f"[{GATE_BAND_LO:+.0f}, {GATE_BAND_HI:+.0f}]"
        )
    if GATE_BAND_LO <= inputs.score_1h <= GATE_BAND_HI:
        reasons.append(
            f"1h score {inputs.score_1h:+.1f} inside the no-edge band "
            f"[{GATE_BAND_LO:+.0f}, {GATE_BAND_HI:+.0f}]"
        )

    s1, s4 = _sign(inputs.score_1h), _sign(inputs.score_4h)
    if s1 != 0 and s4 != 0 and s1 != s4:
        reasons.append(
            f"1h ({inputs.score_1h:+.1f}) and 4h ({inputs.score_4h:+.1f}) "
            f"disagree on direction"
        )

    return (len(reasons) == 0, reasons)


# ─────────────────────────────────────────────────────────────────────────────
# 2 — DIRECTION
# ─────────────────────────────────────────────────────────────────────────────
def determine_bias(inputs: RecommendationInputs) -> str:
    """Direction rule:
        long  if 4h > +60 AND 1h > +50
        short if 4h < -60 AND 1h < -50
        otherwise no_trade
    """
    if inputs.score_4h > LONG_4H_MIN and inputs.score_1h > LONG_1H_MIN:
        return "long"
    if inputs.score_4h < SHORT_4H_MAX and inputs.score_1h < SHORT_1H_MAX:
        return "short"
    return "no_trade"


# ─────────────────────────────────────────────────────────────────────────────
# 3 — SETUP
# ─────────────────────────────────────────────────────────────────────────────
def classify_setup(regime: Optional[RegimeAssessment]) -> str:
    """Map the regime label to a setup label.

    - regime "trend"          → "trend"
    - regime "squeeze"        → "squeeze"
    - regime "mean_reversion" → "mean_reversion"
    - regime "neutral" / None → "none"
    """
    if regime is None:
        return "none"
    label = str(regime.regime).lower()
    if label in ("trend", "squeeze", "mean_reversion"):
        return label
    return "none"


# ─────────────────────────────────────────────────────────────────────────────
# CONVICTION
# ─────────────────────────────────────────────────────────────────────────────
def determine_conviction(
    inputs: RecommendationInputs, bias: str, setup: str
) -> str:
    """Tier the bias from low / medium / high based on score depth, then
    apply two haircuts:
      • drop one tier if either timeframe is flagged unstable
      • drop one tier if setup is `mean_reversion` (contradicts a directional bias)
    """
    if bias == "no_trade":
        return "none"

    a4 = abs(inputs.score_4h)
    a1 = abs(inputs.score_1h)

    if a4 >= HIGH_4H_ABS and a1 >= HIGH_1H_ABS:
        tier = "high"
    elif a4 >= MED_4H_ABS and a1 >= MED_1H_ABS:
        tier = "medium"
    else:
        tier = "low"

    # Haircut #1: stability
    if not (inputs.stable_4h and inputs.stable_1h):
        tier = {"high": "medium", "medium": "low", "low": "low"}[tier]

    # Haircut #2: mean-reversion regime contradicts directional bias
    if setup == "mean_reversion":
        tier = {"high": "medium", "medium": "low", "low": "low"}[tier]

    return tier


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def _expiry_for_timeframe(timeframe: str) -> tuple[int, int]:
    tf = (timeframe or "").lower()
    if tf == "1h":
        return (EXPIRY_1H_MIN, EXPIRY_1H_MAX)
    return (EXPIRY_4H_MIN, EXPIRY_4H_MAX)  # default to the slower / safer window


def _iv_size_multiplier(atm_iv_pct: Optional[float]) -> tuple[float, str]:
    if atm_iv_pct is None or not np.isfinite(atm_iv_pct):
        return 1.0, "ATM IV unknown — no IV haircut applied."
    iv = float(atm_iv_pct)
    if iv >= IV_HIGH_PCT:
        return 0.50, f"ATM IV {iv:.1f}% ≥ {IV_HIGH_PCT:.0f}% — high — size halved."
    if iv >= IV_ELEVATED_PCT:
        return 0.75, f"ATM IV {iv:.1f}% ≥ {IV_ELEVATED_PCT:.0f}% — elevated — 25% size haircut."
    return 1.00, f"ATM IV {iv:.1f}% — normal regime — full size."


def select_options_structure(
    bias: str,
    setup: str,
    conviction: str,
    primary_timeframe: str = "4h",
    atm_iv_pct: Optional[float] = None,
) -> OptionsSuggestion:
    """Pick an options structure that matches bias × setup × conviction.

    Selection rules (evaluated in order):
      1. squeeze setup       → far-OTM single-leg ({SQ_LO}–{SQ_HI}Δ) for the burst
      2. conviction = high   → standard single-leg ({HI_LO}–{HI_HI}Δ)
      3. conviction = medium → debit spread (long {SP_LO_LO}–{SP_LO_HI}Δ /
                                            short {SP_SH_LO}–{SP_SH_HI}Δ)
      4. conviction = low    → same debit spread at half base size

    Then expiry is set from the primary timeframe (1h: 5–7D, 4h: 7–14D)
    and a size haircut is applied if ATM IV is elevated.

    Shorts mirror longs: every "call" becomes a "put" with the same |Δ| range.
    """
    expiry = _expiry_for_timeframe(primary_timeframe)
    iv_size, iv_note = _iv_size_multiplier(atm_iv_pct)

    if bias not in ("long", "short"):
        return OptionsSuggestion(
            structure="none",
            long_leg_delta=(0.0, 0.0),
            short_leg_delta=None,
            expiry_days=expiry,
            size_multiplier=0.0,
            notes=["No directional bias — no options structure."],
        )

    is_long = bias == "long"
    leg = "call" if is_long else "put"
    notes: list[str] = []

    # 1. Squeeze takes precedence regardless of conviction tier
    if setup == "squeeze":
        structure = leg
        long_leg = (SQUEEZE_DELTA_LOW, SQUEEZE_DELTA_HIGH)
        short_leg: Optional[tuple[float, float]] = None
        base_size = 1.0
        notes.append(
            f"Squeeze setup → far-OTM {leg} "
            f"({long_leg[0]:.0f}–{long_leg[1]:.0f}Δ) to capture the sharp burst."
        )

    # 2. High conviction → standard single-leg
    elif conviction == "high":
        structure = leg
        long_leg = (HIGH_DELTA_LOW, HIGH_DELTA_HIGH)
        short_leg = None
        base_size = 1.0
        notes.append(
            f"High conviction → directional {leg} "
            f"({long_leg[0]:.0f}–{long_leg[1]:.0f}Δ)."
        )

    # 3. Medium conviction → debit spread
    elif conviction == "medium":
        structure = f"{leg}_spread"
        long_leg = (SPREAD_LONG_DELTA_LOW, SPREAD_LONG_DELTA_HIGH)
        short_leg = (SPREAD_SHORT_DELTA_LOW, SPREAD_SHORT_DELTA_HIGH)
        base_size = 1.0
        notes.append(
            f"Medium conviction → {leg} debit spread "
            f"(long {long_leg[0]:.0f}–{long_leg[1]:.0f}Δ / "
            f"short {short_leg[0]:.0f}–{short_leg[1]:.0f}Δ)."
        )

    # 4. Low / fallback → smaller debit spread
    else:
        structure = f"{leg}_spread"
        long_leg = (SPREAD_LONG_DELTA_LOW, SPREAD_LONG_DELTA_HIGH)
        short_leg = (SPREAD_SHORT_DELTA_LOW, SPREAD_SHORT_DELTA_HIGH)
        base_size = 0.5
        notes.append(
            f"Low conviction → small {leg} debit spread "
            f"(long {long_leg[0]:.0f}–{long_leg[1]:.0f}Δ / "
            f"short {short_leg[0]:.0f}–{short_leg[1]:.0f}Δ) at half size."
        )

    notes.append(
        f"Expiry window {expiry[0]}–{expiry[1]} days "
        f"(driven by {primary_timeframe} timeframe)."
    )
    notes.append(iv_note)

    final_size = float(np.clip(base_size * iv_size, 0.0, 1.0))

    return OptionsSuggestion(
        structure=structure,
        long_leg_delta=long_leg,
        short_leg_delta=short_leg,
        expiry_days=expiry,
        size_multiplier=final_size,
        notes=notes,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4 — TOP-LEVEL OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
_SETUP_NOTE = {
    "trend":          "trend regime supports the move",
    "squeeze":        "squeeze regime — expect a sharp burst, then snap-back",
    "mean_reversion": "mean-reversion regime contradicts the directional bias — fade or reduce size",
    "none":           "no clear regime",
}


def evaluate_recommendation(inputs: RecommendationInputs) -> Recommendation:
    """Run the full pipeline and return a Recommendation."""

    # Step 1 — gatekeeper
    passed, reasons = gatekeeper(inputs)
    setup = classify_setup(inputs.regime_4h)

    if not passed:
        return Recommendation(
            bias="no_trade",
            setup=setup,
            conviction="none",
            explanation=[
                f"Gatekeeper blocked the trade ({len(reasons)} reason{'s' if len(reasons)!=1 else ''}).",
                f"Latest scores: 1h {inputs.score_1h:+.1f} · 4h {inputs.score_4h:+.1f}.",
                f"Primary reason: {reasons[0]}",
            ],
            gatekeeper_passed=False,
            blocked_reasons=list(reasons),
            options_suggestion=select_options_structure(
                bias="no_trade",
                setup=setup,
                conviction="none",
                primary_timeframe=inputs.primary_timeframe,
                atm_iv_pct=inputs.atm_iv_pct,
            ),
        )

    # Step 2 — direction
    bias = determine_bias(inputs)

    # Step 3 — setup is already computed above

    # Step 4 — conviction + explanation
    conviction = determine_conviction(inputs, bias, setup)

    if bias == "no_trade":
        explanation = [
            "Gates passed, but direction not strong enough: long needs 4h > 60 & 1h > 50; "
            "short needs 4h < -60 & 1h < -50.",
            f"Latest scores: 1h {inputs.score_1h:+.1f} · 4h {inputs.score_4h:+.1f}.",
            f"Setup: {setup} — wait for the 4h to break the directional threshold.",
        ]
    else:
        arrow = "▲" if bias == "long" else "▼"
        stab_note = (
            "both timeframes stable"
            if (inputs.stable_1h and inputs.stable_4h)
            else "stability haircut applied"
        )
        explanation = [
            f"{arrow} {bias.upper()} bias: 4h {inputs.score_4h:+.1f} and 1h {inputs.score_1h:+.1f} "
            f"both past directional thresholds.",
            f"Setup: {setup} — {_SETUP_NOTE.get(setup, _SETUP_NOTE['none'])}.",
            f"Conviction {conviction.upper()} ({stab_note}).",
        ]

    options = select_options_structure(
        bias=bias,
        setup=setup,
        conviction=conviction,
        primary_timeframe=inputs.primary_timeframe,
        atm_iv_pct=inputs.atm_iv_pct,
    )

    return Recommendation(
        bias=bias,
        setup=setup,
        conviction=conviction,
        explanation=explanation,
        gatekeeper_passed=True,
        blocked_reasons=[],
        options_suggestion=options,
    )
