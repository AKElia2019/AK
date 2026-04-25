"""
btc_dashboard.analytics.regime
Market-regime classifier.

Maps a snapshot of normalised signals + a few raw market-state values
to one of four regimes:

  - "trend"          low / negative GEX, aligned family scores, volume confirms
  - "mean_reversion" high positive GEX, price stretched vs RN mean
  - "squeeze"        extreme funding, crowded OI, RN diverging from spot
  - "neutral"        none of the above

Each regime is scored on a [0, 1] confidence scale. The label with the
highest confidence wins, unless every regime scores below
`NEUTRAL_THRESHOLD` — in that case the result is "neutral".

Pure classification — no smoothing, no time series. One snapshot in,
one regime out. Smoothing the inputs (e.g. via `analytics.smoothing`)
upstream is the recommended pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (easy-to-tune thresholds)
# ─────────────────────────────────────────────────────────────────────────────
NEUTRAL_THRESHOLD: float = 0.40
"""If max regime confidence is below this, the result is `neutral`."""

# Trend
TREND_FINAL_SAT: float = 50.0          # |final_score| at which alignment evidence saturates

# Mean reversion
MR_RN_Z_SAT: float = 2.0               # |z(spot − rn_mean)| at which stretch evidence saturates
MR_RN_PCT_SAT: float = 5.0             # fallback when rn_std is unavailable, in %

# Squeeze
SQ_FUNDING_BASE_PCT: float = 30.0      # annualised % at which "extreme funding" begins
SQ_FUNDING_SAT_PCT: float = 60.0       # annualised % at which evidence saturates to 1.0
SQ_OI_PCT_SAT: float = 3.0             # |OI %Δ| at which OI evidence saturates
SQ_RN_GAP_PCT_SAT: float = 3.0         # |rn_mean − spot| / spot at which RN-gap evidence saturates


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RegimeInputs:
    """Snapshot of features the classifier needs.

    Composite scores are in [-100, +100] (from `analytics.scoring`).
    Normalised signals are in [-1, +1] (from `analytics.normalization`).
    Raw market-state values keep their natural units.
    """

    # Composite (already weighted) scores from scoring.py
    final_score: float
    options_score: float = 0.0
    futures_score: float = 0.0
    spot_score: float = 0.0

    # Normalised signals
    gex_normalized: float = 0.0
    funding_normalized: float = 0.0
    volume_confirmation: float = 0.0
    rn_drift_normalized: float = 0.0

    # Raw market-state values
    rn_mean: Optional[float] = None
    spot: Optional[float] = None
    rn_std: Optional[float] = None
    funding_annualized_pct: float = 0.0
    oi_pct_change: float = 0.0
    spot_pct_change: float = 0.0


@dataclass(frozen=True)
class RegimeAssessment:
    regime: str                       # "trend" | "mean_reversion" | "squeeze" | "neutral"
    confidence: float                 # in [0, 1]
    direction: Optional[str]          # "bullish" | "bearish" | None
    scores: dict[str, float]          # confidence for every candidate regime
    rationale: dict[str, list[str]]   # evidence lines per candidate


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _sign(x: float) -> int:
    if x is None or not np.isfinite(x) or x == 0:
        return 0
    return 1 if x > 0 else -1


def _clamp01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return max(0.0, min(1.0, float(x)))


# ─────────────────────────────────────────────────────────────────────────────
# REGIME DETECTORS
# ─────────────────────────────────────────────────────────────────────────────
def detect_trend(x: RegimeInputs) -> tuple[float, list[str]]:
    """Trend regime: low/negative GEX, aligned family scores, volume confirms.

    Three pieces of evidence, averaged into a confidence in [0, 1]:
      1. GEX is non-positive (low or negative).
      2. The composite final score is aligned with each family's score.
      3. Volume confirms the direction of the final score.
    """
    # 1. GEX: −1 → 1.0, 0 → 0.5, +1 → 0.0
    gex_ev = _clamp01(0.5 - 0.5 * x.gex_normalized)

    # 2. Alignment between the final score and family scores
    final_dir = _sign(x.final_score)
    if final_dir == 0:
        aligned = 0
    else:
        aligned = sum(
            int(_sign(s) == final_dir)
            for s in (x.options_score, x.futures_score, x.spot_score)
        )
    final_mag = min(1.0, abs(x.final_score) / TREND_FINAL_SAT)
    align_ev = (aligned / 3.0) * final_mag

    # 3. Volume confirms the direction of the final score
    vol_ev = _clamp01(final_dir * x.volume_confirmation) if final_dir != 0 else 0.0

    confidence = float(np.mean([gex_ev, align_ev, vol_ev]))
    rationale = [
        f"GEX evidence {gex_ev:.2f} (gex_norm {x.gex_normalized:+.2f})",
        f"Alignment {aligned}/3 families · final {x.final_score:+.1f} (mag {final_mag:.2f})",
        f"Volume confirmation {vol_ev:.2f} (vol_conf {x.volume_confirmation:+.2f})",
    ]
    return confidence, rationale


def detect_mean_reversion(x: RegimeInputs) -> tuple[float, list[str]]:
    """Mean-reversion regime: high positive GEX and price stretched vs RN mean.

    Two pieces of evidence, averaged:
      1. GEX is positive (dealers long gamma → mean-reverting flow).
      2. Distance from spot to the RN mean — measured in σ if available,
         otherwise as a % of spot.
    """
    # 1. High GEX: +1 → 1.0, 0 → 0.0, −1 → 0.0
    gex_ev = _clamp01(x.gex_normalized)

    # 2. Stretch vs RN mean
    stretch_ev = 0.0
    if x.spot and x.spot > 0 and x.rn_mean is not None and np.isfinite(x.rn_mean):
        gap = abs(float(x.spot) - float(x.rn_mean))
        if x.rn_std and x.rn_std > 0:
            z = gap / float(x.rn_std)
            stretch_ev = _clamp01(z / MR_RN_Z_SAT)
            stretch_note = f"|z| {z:.2f}σ"
        else:
            pct = gap / float(x.spot) * 100.0
            stretch_ev = _clamp01(pct / MR_RN_PCT_SAT)
            stretch_note = f"|gap| {pct:.2f}%"
    else:
        stretch_note = "rn_mean / spot unavailable"

    confidence = float(np.mean([gex_ev, stretch_ev]))
    rationale = [
        f"GEX evidence {gex_ev:.2f} (positive GEX dampens vol)",
        f"Stretch evidence {stretch_ev:.2f} ({stretch_note})",
    ]
    return confidence, rationale


def detect_squeeze(x: RegimeInputs) -> tuple[float, list[str]]:
    """Squeeze regime: extreme funding, crowded OI, RN diverging from spot.

    Three pieces of evidence, averaged:
      1. Funding annualised |%| above SQ_FUNDING_BASE_PCT (saturates at SQ_FUNDING_SAT_PCT).
      2. OI movement; counts more strongly if it diverges from spot direction.
      3. RN mean diverged from spot, OR RN mean drifted sharply (use the larger).
    """
    # 1. Extreme funding
    a = abs(float(x.funding_annualized_pct))
    if a <= SQ_FUNDING_BASE_PCT:
        fund_ev = 0.0
    else:
        denom = max(SQ_FUNDING_SAT_PCT - SQ_FUNDING_BASE_PCT, 1e-9)
        fund_ev = _clamp01((a - SQ_FUNDING_BASE_PCT) / denom)

    # 2. Crowded OI / divergence
    oi_mag = _clamp01(abs(x.oi_pct_change) / SQ_OI_PCT_SAT)
    if _sign(x.oi_pct_change) != 0 and _sign(x.spot_pct_change) != 0:
        if _sign(x.oi_pct_change) != _sign(x.spot_pct_change):
            oi_ev = oi_mag                   # divergent — classic squeeze setup
        else:
            oi_ev = oi_mag * 0.5             # confirmation only, weaker squeeze
    else:
        oi_ev = oi_mag * 0.5

    # 3. RN divergence
    rn_gap_ev = 0.0
    if x.spot and x.spot > 0 and x.rn_mean is not None and np.isfinite(x.rn_mean):
        pct = abs(float(x.rn_mean) - float(x.spot)) / float(x.spot) * 100.0
        rn_gap_ev = _clamp01(pct / SQ_RN_GAP_PCT_SAT)
    rn_drift_ev = _clamp01(abs(x.rn_drift_normalized))
    rn_ev = max(rn_gap_ev, rn_drift_ev)

    confidence = float(np.mean([fund_ev, oi_ev, rn_ev]))
    rationale = [
        f"Funding evidence {fund_ev:.2f} ({x.funding_annualized_pct:+.1f}% ann)",
        f"OI evidence {oi_ev:.2f} (oi Δ {x.oi_pct_change:+.2f}% vs spot Δ {x.spot_pct_change:+.2f}%)",
        f"RN evidence {rn_ev:.2f} (gap {rn_gap_ev:.2f} · drift {rn_drift_ev:.2f})",
    ]
    return confidence, rationale


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
def classify_regime(
    inputs: RegimeInputs,
    neutral_threshold: float = NEUTRAL_THRESHOLD,
) -> RegimeAssessment:
    """Classify a snapshot into one of {trend, mean_reversion, squeeze, neutral}.

    Returns the regime label, its confidence in [0, 1], a directional bias
    (when applicable, derived from the sign of the final score), the
    confidence scores for every candidate regime, and per-regime rationale
    lines for transparency.
    """
    trend_conf,    trend_rat    = detect_trend(inputs)
    mr_conf,       mr_rat       = detect_mean_reversion(inputs)
    squeeze_conf,  squeeze_rat  = detect_squeeze(inputs)

    candidates: dict[str, float] = {
        "trend":          trend_conf,
        "mean_reversion": mr_conf,
        "squeeze":        squeeze_conf,
    }
    rationale: dict[str, list[str]] = {
        "trend":          trend_rat,
        "mean_reversion": mr_rat,
        "squeeze":        squeeze_rat,
    }

    # Pick the strongest candidate; fall back to neutral if all are weak.
    best_label = max(candidates, key=candidates.get)
    best_conf = candidates[best_label]

    if best_conf < neutral_threshold:
        regime = "neutral"
        confidence = float(1.0 - best_conf)  # how confident we are in "no regime"
        direction: Optional[str] = None
        rationale["neutral"] = [
            f"All regime scores below {neutral_threshold:.2f} (max {best_conf:.2f} for {best_label})"
        ]
        candidates["neutral"] = confidence
    else:
        regime = best_label
        confidence = float(best_conf)
        if regime == "mean_reversion":
            direction = None  # mean-reversion has no inherent direction at this layer
        else:
            d = _sign(inputs.final_score)
            direction = "bullish" if d > 0 else "bearish" if d < 0 else None

    return RegimeAssessment(
        regime=regime,
        confidence=_clamp01(confidence),
        direction=direction,
        scores={k: float(v) for k, v in candidates.items()},
        rationale=rationale,
    )
