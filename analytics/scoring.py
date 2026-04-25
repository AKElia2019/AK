"""
btc_dashboard.analytics.scoring
Weighted-score aggregator. Consumes normalised signals (each in [-1, +1])
and produces a composite score in [-100, +100] with per-family sub-scores.

Design
------
- Family weights live in `DEFAULT_FAMILY_WEIGHTS` and are easy to override
  via the `weights=` argument on `compute_composite`.
- Each family has its own `score_<family>()` function (modular handles)
  so family-specific logic can be added later without touching the API.
- Inside a family, signals are equal-weighted by default but can carry an
  intra-family `weight` field for callers that want to weight them.
- If a family has no signals, its weight is redistributed proportionally
  across the remaining families (so the composite isn't biased low).
- No smoothing, no EMA, no hysteresis. One snapshot in, one score out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_FAMILY_WEIGHTS: dict[str, float] = {
    "options":   0.40,
    "futures":   0.25,
    "spot":      0.15,
    "liquidity": 0.10,
    "flow":      0.10,
}


# ─────────────────────────────────────────────────────────────────────────────
# TYPES
# ─────────────────────────────────────────────────────────────────────────────
SignalLike = Union[float, int, Mapping[str, Any]]
"""A single signal: either a bare scalar in [-1, +1] or a dict carrying
`{"name", "value" (or "score"), "weight" (optional, default 1.0)}`."""

SignalsInput = Union[Sequence[SignalLike], Mapping[str, SignalLike]]
"""A family's signal collection: list of SignalLike, or {name: SignalLike}."""


@dataclass(frozen=True)
class SignalEntry:
    name: str
    value: float           # clipped to [-1, +1]
    weight: float          # raw intra-family weight (≥ 0)


@dataclass(frozen=True)
class FamilyScore:
    family: str
    score: float           # in [-100, +100]
    weight: float          # share of the final score (sums to 1.0 over active families)
    contribution: float    # score × weight (sums to final_score)
    n_signals: int
    signals: tuple[SignalEntry, ...]


@dataclass(frozen=True)
class CompositeScore:
    final_score: float                  # in [-100, +100]
    sub_scores: dict[str, FamilyScore]  # one entry per known family
    weights: dict[str, float]           # validated/normalised input weights


# ─────────────────────────────────────────────────────────────────────────────
# COERCION
# ─────────────────────────────────────────────────────────────────────────────
def _coerce_signal(item: SignalLike, default_name: str) -> SignalEntry:
    if isinstance(item, Mapping):
        name = str(item.get("name", default_name))
        raw = item.get("value", item.get("score", 0.0))
        try:
            value = float(raw)
        except Exception:
            value = 0.0
        try:
            weight = float(item.get("weight", 1.0))
        except Exception:
            weight = 1.0
    else:
        name = default_name
        try:
            value = float(item)
        except Exception:
            value = 0.0
        weight = 1.0

    if not np.isfinite(value):
        value = 0.0
    value = max(-1.0, min(1.0, value))
    if not np.isfinite(weight) or weight < 0:
        weight = 0.0
    return SignalEntry(name=name, value=value, weight=weight)


def _coerce_signals(signals: SignalsInput) -> tuple[SignalEntry, ...]:
    if signals is None:
        return ()
    if isinstance(signals, Mapping):
        out = []
        for name, item in signals.items():
            if isinstance(item, Mapping):
                d = dict(item)
                d.setdefault("name", name)
                out.append(_coerce_signal(d, str(name)))
            else:
                out.append(_coerce_signal(item, str(name)))
        return tuple(out)
    return tuple(_coerce_signal(s, f"signal_{i}") for i, s in enumerate(signals))


# ─────────────────────────────────────────────────────────────────────────────
# FAMILY SCORERS  (modular — one function per family, all delegate to the
# generic aggregator today; family-specific logic can be added in place
# later without breaking the API)
# ─────────────────────────────────────────────────────────────────────────────
def score_family(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    """Aggregate a family's signals into a score in [-100, +100].

    Weighted average of signal values × 100, where the weights are the
    intra-family weights from each signal. Signals with weight 0 are
    ignored. Empty input → (0.0, ()).
    """
    entries = _coerce_signals(signals)
    if not entries:
        return 0.0, ()
    total_w = sum(e.weight for e in entries)
    if total_w <= 0:
        return 0.0, entries
    avg = sum(e.value * e.weight for e in entries) / total_w
    return float(avg * 100.0), entries


def score_options(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    return score_family(signals)


def score_futures(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    return score_family(signals)


def score_spot(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    return score_family(signals)


def score_liquidity(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    return score_family(signals)


def score_flow(signals: SignalsInput) -> tuple[float, tuple[SignalEntry, ...]]:
    return score_family(signals)


_FAMILY_SCORERS = {
    "options":   score_options,
    "futures":   score_futures,
    "spot":      score_spot,
    "liquidity": score_liquidity,
    "flow":      score_flow,
}


# ─────────────────────────────────────────────────────────────────────────────
# WEIGHT VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
def validate_weights(
    weights: Mapping[str, float], tol: float = 1e-6
) -> dict[str, float]:
    """Validate and normalise family weights.

    - Each weight must be finite and ≥ 0.
    - Total must be > 0.
    - If the total is not 1.0 within `tol`, weights are renormalised so
      the caller doesn't need to balance the dict by hand.

    Raises ValueError on invalid input.
    """
    if not weights:
        raise ValueError("weights must be non-empty")
    out: dict[str, float] = {}
    for k, v in weights.items():
        try:
            vf = float(v)
        except Exception as exc:
            raise ValueError(f"weight for {k!r} is not numeric: {v!r}") from exc
        if not np.isfinite(vf) or vf < 0:
            raise ValueError(f"weight for {k!r} must be finite and ≥ 0; got {vf}")
        out[str(k)] = vf
    total = sum(out.values())
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    if abs(total - 1.0) > tol:
        out = {k: v / total for k, v in out.items()}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE
# ─────────────────────────────────────────────────────────────────────────────
def compute_composite(
    signals_by_family: Mapping[str, SignalsInput],
    weights: Mapping[str, float] | None = None,
) -> CompositeScore:
    """Compute the weighted composite score from per-family signal sets.

    Parameters
    ----------
    signals_by_family : Mapping[str, SignalsInput]
        e.g. ``{"options": [...], "futures": [...], "spot": [...],
                "liquidity": [...], "flow": [...]}``.
        Each family's payload is a list of SignalLike or a {name: value} dict.
    weights : Mapping[str, float], optional
        Family-weight overrides. Defaults to ``DEFAULT_FAMILY_WEIGHTS``.
        Auto-normalised if not summing to 1.0.

    Returns
    -------
    CompositeScore
        ``final_score`` in [-100, +100], with sub-scores per family.
    """
    w = validate_weights(weights or DEFAULT_FAMILY_WEIGHTS)

    # Step 1 — score each family independently.
    raw: dict[str, tuple[float, tuple[SignalEntry, ...]]] = {}
    active_weight = 0.0
    for fam, fam_w in w.items():
        scorer = _FAMILY_SCORERS.get(fam, score_family)
        score_val, entries = scorer(signals_by_family.get(fam, []))
        raw[fam] = (score_val, entries)
        if entries:
            active_weight += fam_w

    # Step 2 — assemble the composite. Families with no signals contribute 0
    # and their weight is redistributed across the families that DO have
    # signals so the headline score isn't artificially diluted.
    sub_scores: dict[str, FamilyScore] = {}
    final_score = 0.0
    for fam, (score_val, entries) in raw.items():
        original_w = w[fam]
        if entries and active_weight > 0:
            effective_w = original_w / active_weight
        else:
            effective_w = 0.0
        contribution = score_val * effective_w
        final_score += contribution
        sub_scores[fam] = FamilyScore(
            family=fam,
            score=float(np.clip(score_val, -100.0, 100.0)),
            weight=float(effective_w),
            contribution=float(contribution),
            n_signals=len(entries),
            signals=entries,
        )

    return CompositeScore(
        final_score=float(np.clip(final_score, -100.0, 100.0)),
        sub_scores=sub_scores,
        weights=dict(w),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def composite_to_dataframe(comp: CompositeScore) -> pd.DataFrame:
    """Tidy DataFrame view of a CompositeScore — one row per family.
    Columns: family, score, weight, contribution, n_signals."""
    rows = [
        {
            "family": fam,
            "score": fs.score,
            "weight": fs.weight,
            "contribution": fs.contribution,
            "n_signals": fs.n_signals,
        }
        for fam, fs in comp.sub_scores.items()
    ]
    return pd.DataFrame(rows)


def signals_to_dataframe(comp: CompositeScore) -> pd.DataFrame:
    """Long-form DataFrame view — one row per (family, signal).
    Columns: family, name, value, weight."""
    rows = []
    for fam, fs in comp.sub_scores.items():
        for s in fs.signals:
            rows.append(
                {
                    "family": fam,
                    "name": s.name,
                    "value": s.value,
                    "weight": s.weight,
                }
            )
    return pd.DataFrame(rows)
