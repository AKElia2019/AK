"""
btc_dashboard.analytics.position_sizing
Position sizing, stop placement, take-profit ladder, and exit-condition list.

Pipeline order (per the project architecture):

    normalization → scoring → smoothing → regime → recommendation → position_sizing

Inputs come from `Recommendation` (bias / setup / conviction) and a small
amount of market state (spot, daily σ or ATM IV, capital, stability flags,
nearest support/resistance). One snapshot in, one trade plan out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# RISK BANDS  (fraction of capital at risk per trade)
# ─────────────────────────────────────────────────────────────────────────────
RISK_HIGH_LO:    float = 0.0075   # 0.75%
RISK_HIGH_HI:    float = 0.0100   # 1.00%
RISK_MED_LO:     float = 0.0040   # 0.40%
RISK_MED_HI:     float = 0.0060   # 0.60%
RISK_SQUEEZE_LO: float = 0.0025   # 0.25%
RISK_SQUEEZE_HI: float = 0.0050   # 0.50%
RISK_LOW_LO:     float = 0.0010   # 0.10%
RISK_LOW_HI:     float = 0.0025   # 0.25%

# Hard cap on risk-per-trade after every adjustment (safety belt).
RISK_HARD_CAP:   float = 0.020    # 2.00%


# ─────────────────────────────────────────────────────────────────────────────
# STOP / TP MULTIPLIERS  (× 1σ daily move)
# ─────────────────────────────────────────────────────────────────────────────
STOP_K_HIGH:     float = 0.7
STOP_K_MEDIUM:   float = 0.6
STOP_K_LOW:      float = 0.5
STOP_K_SQUEEZE:  float = 0.5

TP1_K_DEFAULT:    float = 0.5
TP2_K_DEFAULT:    float = 1.2
RUNNER_K_DEFAULT: float = 2.0

TP1_K_SQUEEZE:    float = 0.4
TP2_K_SQUEEZE:    float = 0.9
RUNNER_K_SQUEEZE: float = 1.5

TP1_K_MR:    float = 0.4
TP2_K_MR:    float = 0.8
RUNNER_K_MR: float = 1.2


# ─────────────────────────────────────────────────────────────────────────────
# HAIRCUTS  (multiplicative reductions to risk %)
# ─────────────────────────────────────────────────────────────────────────────
IV_HIGH_PCT:     float = 80.0     # ≥ 80% ATM IV
IV_ELEVATED_PCT: float = 60.0     # ≥ 60% ATM IV
HAIRCUT_IV_HIGH:        float = 0.70
HAIRCUT_IV_ELEVATED:    float = 0.85
HAIRCUT_INSTABILITY:    float = 0.75
HAIRCUT_OPPOSING_LIQ:   float = 0.70

# Opposing liquidity is "close" when it sits within this many σ of entry.
OPPOSING_LIQ_THRESHOLD_SIGMA: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SizingInputs:
    """Snapshot of everything the sizer needs.

    `daily_sigma` (1σ daily $ move) wins if provided; otherwise it is derived
    from `atm_iv_pct` × spot × √(1/365). If neither is available the sizer
    falls back to 2% of spot as a defensive default.
    """

    # Recommendation context
    bias: str                   # "long" | "short" | "no_trade"
    setup: str                  # "trend" | "squeeze" | "mean_reversion" | "none"
    conviction: str             # "high" | "medium" | "low" | "none"

    # Market state
    capital: float              # account capital in USD
    spot: float                 # current price ($)
    daily_sigma: Optional[float] = None
    atm_iv_pct: Optional[float] = None

    # Quality / context flags
    stable_1h: bool = True
    stable_4h: bool = True
    nearest_support: Optional[float] = None     # $ price level below
    nearest_resistance: Optional[float] = None  # $ price level above


@dataclass(frozen=True)
class TradePlan:
    bias: str
    setup: str
    conviction: str

    # Levels
    entry: float
    stop: float
    tp1: float
    tp2: float
    runner: float

    # Risk
    risk_pct: float                # of capital, after haircuts
    risk_amount_usd: float         # USD at risk if stop hit
    position_size_usd: float       # notional in USD
    position_size_btc: float       # notional in BTC

    risk_reward_tp2: float         # reward / risk to TP2
    stop_distance: float           # |entry − stop| in USD

    haircuts: list[str]            # multiplicative haircuts applied
    exit_conditions: list[str]     # plain-language exit triggers
    notes: list[str]               # rationale & sizing transparency


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _midpoint(lo: float, hi: float) -> float:
    return (float(lo) + float(hi)) / 2.0


def _risk_band(conviction: str, setup: str) -> tuple[float, float]:
    """Pick the risk band per the rule table.

    Squeeze setup overrides conviction tier (always tighter risk for the
    burst-and-fade nature of squeezes).
    """
    if setup == "squeeze":
        return (RISK_SQUEEZE_LO, RISK_SQUEEZE_HI)
    if conviction == "high":
        return (RISK_HIGH_LO, RISK_HIGH_HI)
    if conviction == "medium":
        return (RISK_MED_LO, RISK_MED_HI)
    return (RISK_LOW_LO, RISK_LOW_HI)


def _stop_k(conviction: str, setup: str) -> float:
    if setup == "squeeze":
        return STOP_K_SQUEEZE
    if conviction == "high":
        return STOP_K_HIGH
    if conviction == "medium":
        return STOP_K_MEDIUM
    return STOP_K_LOW


def _tp_ks(setup: str) -> tuple[float, float, float]:
    if setup == "squeeze":
        return (TP1_K_SQUEEZE, TP2_K_SQUEEZE, RUNNER_K_SQUEEZE)
    if setup == "mean_reversion":
        return (TP1_K_MR, TP2_K_MR, RUNNER_K_MR)
    return (TP1_K_DEFAULT, TP2_K_DEFAULT, RUNNER_K_DEFAULT)


def _resolve_sigma(
    spot: float, daily_sigma: Optional[float], atm_iv_pct: Optional[float]
) -> tuple[float, str]:
    """Return (sigma, source_note). Falls back to a defensive 2% × spot."""
    if daily_sigma is not None and np.isfinite(daily_sigma) and daily_sigma > 0:
        return float(daily_sigma), "1σ daily provided directly"
    if (
        atm_iv_pct is not None
        and np.isfinite(atm_iv_pct)
        and atm_iv_pct > 0
        and spot > 0
    ):
        sig = float(spot) * (float(atm_iv_pct) / 100.0) * math.sqrt(1.0 / 365.0)
        return sig, f"derived from ATM IV {atm_iv_pct:.1f}%"
    return float(spot) * 0.02, "fallback: 2% of spot (no IV / σ available)"


def _opposing_liq_close(
    bias: str,
    spot: float,
    sigma: float,
    nearest_support: Optional[float],
    nearest_resistance: Optional[float],
) -> tuple[bool, str]:
    """For longs, the opposing wall is the nearest resistance above spot.
    For shorts, it is the nearest support below spot.
    Returns (is_close, note)."""
    if sigma <= 0:
        return False, ""
    if bias == "long" and nearest_resistance is not None and nearest_resistance > spot:
        gap = float(nearest_resistance) - float(spot)
        if gap < OPPOSING_LIQ_THRESHOLD_SIGMA * sigma:
            return True, (
                f"resistance ${nearest_resistance:,.0f} only "
                f"{gap/sigma:.2f}σ above entry"
            )
    if bias == "short" and nearest_support is not None and nearest_support < spot:
        gap = float(spot) - float(nearest_support)
        if gap < OPPOSING_LIQ_THRESHOLD_SIGMA * sigma:
            return True, (
                f"support ${nearest_support:,.0f} only "
                f"{gap/sigma:.2f}σ below entry"
            )
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# CORE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def stop_levels(
    bias: str, spot: float, sigma: float, conviction: str, setup: str
) -> tuple[float, float]:
    """Return (stop_price, stop_distance)."""
    sign = 1 if bias == "long" else -1
    k = _stop_k(conviction, setup)
    stop = float(spot) - sign * k * sigma
    return stop, abs(float(spot) - stop)


def tp_levels(
    bias: str, spot: float, sigma: float, setup: str
) -> tuple[float, float, float]:
    """Return (tp1, tp2, runner) prices."""
    sign = 1 if bias == "long" else -1
    k1, k2, kr = _tp_ks(setup)
    tp1 = float(spot) + sign * k1 * sigma
    tp2 = float(spot) + sign * k2 * sigma
    runner = float(spot) + sign * kr * sigma
    return tp1, tp2, runner


def exit_conditions_for(
    bias: str, setup: str, stop: float, runner: float, risk_amount_usd: float
) -> list[str]:
    """Plain-language list of triggers that should close the trade."""
    out = [
        f"Stop hit at ${stop:,.0f} — max loss ≈ ${risk_amount_usd:,.0f}.",
        f"Runner target ${runner:,.0f} reached; trail behind further structure.",
        "4h composite score crosses zero (regime / direction broken).",
        "4h stability flag flips off (signal no longer coherent).",
    ]
    if setup == "squeeze":
        out.append(
            "Time-stop: close if no meaningful move within 1–2 bars "
            "(squeezes resolve fast or not at all)."
        )
    elif setup == "mean_reversion":
        out.append("Close at the RN mean if reached before TP2 (target is the mean).")
    elif setup == "trend":
        out.append("Trail behind the most recent 1h higher-low / lower-high.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def build_trade_plan(inputs: SizingInputs) -> TradePlan:
    """Compute the full trade plan from a `SizingInputs` snapshot."""

    # Empty plan when there is no directional bias.
    if inputs.bias not in ("long", "short"):
        return TradePlan(
            bias=inputs.bias,
            setup=inputs.setup,
            conviction=inputs.conviction,
            entry=float(inputs.spot),
            stop=float(inputs.spot),
            tp1=float(inputs.spot),
            tp2=float(inputs.spot),
            runner=float(inputs.spot),
            risk_pct=0.0,
            risk_amount_usd=0.0,
            position_size_usd=0.0,
            position_size_btc=0.0,
            risk_reward_tp2=0.0,
            stop_distance=0.0,
            haircuts=[],
            exit_conditions=["No directional bias — stand aside."],
            notes=["No trade plan generated (bias = no_trade)."],
        )

    # Resolve 1σ daily move
    sigma, sigma_note = _resolve_sigma(
        spot=inputs.spot,
        daily_sigma=inputs.daily_sigma,
        atm_iv_pct=inputs.atm_iv_pct,
    )

    # Levels
    entry = float(inputs.spot)
    stop, stop_distance = stop_levels(
        inputs.bias, entry, sigma, inputs.conviction, inputs.setup
    )
    tp1, tp2, runner = tp_levels(inputs.bias, entry, sigma, inputs.setup)

    # Base risk band
    risk_lo, risk_hi = _risk_band(inputs.conviction, inputs.setup)
    risk_pct = _midpoint(risk_lo, risk_hi)

    # Haircuts
    haircuts: list[str] = []

    # IV
    if inputs.atm_iv_pct is not None and np.isfinite(inputs.atm_iv_pct):
        if inputs.atm_iv_pct >= IV_HIGH_PCT:
            risk_pct *= HAIRCUT_IV_HIGH
            haircuts.append(
                f"IV {inputs.atm_iv_pct:.1f}% ≥ {IV_HIGH_PCT:.0f}% → ×{HAIRCUT_IV_HIGH:.2f}"
            )
        elif inputs.atm_iv_pct >= IV_ELEVATED_PCT:
            risk_pct *= HAIRCUT_IV_ELEVATED
            haircuts.append(
                f"IV {inputs.atm_iv_pct:.1f}% ≥ {IV_ELEVATED_PCT:.0f}% → ×{HAIRCUT_IV_ELEVATED:.2f}"
            )

    # Instability
    if not (inputs.stable_1h and inputs.stable_4h):
        risk_pct *= HAIRCUT_INSTABILITY
        haircuts.append(f"signal instability → ×{HAIRCUT_INSTABILITY:.2f}")

    # Opposing liquidity
    opposing_close, opp_note = _opposing_liq_close(
        bias=inputs.bias,
        spot=entry,
        sigma=sigma,
        nearest_support=inputs.nearest_support,
        nearest_resistance=inputs.nearest_resistance,
    )
    if opposing_close:
        risk_pct *= HAIRCUT_OPPOSING_LIQ
        haircuts.append(
            f"opposing liquidity ({opp_note}) → ×{HAIRCUT_OPPOSING_LIQ:.2f}"
        )

    # Hard safety cap
    risk_pct = float(np.clip(risk_pct, 0.0, RISK_HARD_CAP))

    # Capital allocation
    risk_amount_usd = float(inputs.capital) * risk_pct
    if stop_distance > 0 and entry > 0:
        position_size_usd = risk_amount_usd * (entry / stop_distance)
        position_size_btc = position_size_usd / entry
    else:
        position_size_usd = 0.0
        position_size_btc = 0.0

    risk_reward_tp2 = (abs(tp2 - entry) / stop_distance) if stop_distance > 0 else 0.0

    # Exit conditions
    exits = exit_conditions_for(
        inputs.bias, inputs.setup, stop, runner, risk_amount_usd
    )

    # Transparency notes
    notes = [
        f"Risk band {risk_lo*100:.2f}–{risk_hi*100:.2f}% "
        f"(midpoint {(risk_lo+risk_hi)/2*100:.2f}%) → after haircuts {risk_pct*100:.3f}%.",
        f"1σ daily ≈ ${sigma:,.0f} ({sigma_note}); "
        f"stop {(stop_distance/sigma):.2f}σ; "
        f"TP1/TP2/runner at "
        f"{abs(tp1-entry)/sigma:.2f}/{abs(tp2-entry)/sigma:.2f}/{abs(runner-entry)/sigma:.2f}σ.",
        f"R/R to TP2: {risk_reward_tp2:.2f}x.",
    ]
    if not haircuts:
        notes.append("No haircuts applied.")

    return TradePlan(
        bias=inputs.bias,
        setup=inputs.setup,
        conviction=inputs.conviction,
        entry=float(entry),
        stop=float(stop),
        tp1=float(tp1),
        tp2=float(tp2),
        runner=float(runner),
        risk_pct=float(risk_pct),
        risk_amount_usd=float(risk_amount_usd),
        position_size_usd=float(position_size_usd),
        position_size_btc=float(position_size_btc),
        risk_reward_tp2=float(risk_reward_tp2),
        stop_distance=float(stop_distance),
        haircuts=haircuts,
        exit_conditions=exits,
        notes=notes,
    )
