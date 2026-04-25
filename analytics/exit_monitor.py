"""
btc_dashboard.analytics.exit_monitor
Risk-management exit monitor for open positions.

Implements the three exit conditions from the project scope:

  1. Score deterioration   — exit if the directional 4h score falls below
                              the absolute threshold (default 40).
  2. RN drift reversal     — exit if the risk-neutral mean drifts against
                              the position direction beyond a small band.
  3. Premium loss          — exit if the current option premium has fallen
                              by more than the configured % (default 50%).

This module is **pure logic**. It does not touch Streamlit, files, or the
network. The page layer is responsible for persisting positions (e.g. in
`st.session_state` or a database) and for fetching the live state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (tunable thresholds)
# ─────────────────────────────────────────────────────────────────────────────
SCORE_EXIT_ABS: float = 40.0
"""Directional 4h score level below which the trade should be exited."""

SCORE_WARN_BAND: float = 10.0
"""Within this band of the exit threshold, we raise a warning alert."""

RN_DRIFT_BAND_PCT: float = 0.5
"""Tolerance band on RN mean (as % of entry RN mean) before flagging
reversal. Below the band: warning. Above the band, in the wrong direction:
exit."""

PREMIUM_LOSS_EXIT: float = -0.50
"""Premium-loss threshold (as a fraction of entry premium) that triggers
an exit. -0.50 = -50%."""

PREMIUM_LOSS_WARN: float = -0.30
"""Warning threshold on premium loss."""


# ─────────────────────────────────────────────────────────────────────────────
# IO TYPES
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Position:
    """One open trade.

    Stored as a mutable dataclass so the UI can refresh `current_premium`
    in place. Pass the dataclass into `evaluate_exit` to get alerts.
    """

    id: str
    bias: str                       # "long" | "short"
    instrument: str                 # "call" | "put" | "call_spread" | "put_spread"

    # State at entry
    entry_score_4h: float           # in [-100, +100]
    entry_rn_mean: Optional[float]  # USD; may be None if no RN fit was available
    entry_premium: float            # USD per contract / structure
    entry_spot: float               # USD spot at entry
    opened_at: datetime

    # Current marks (refreshed by the caller from the live pipeline / inputs)
    current_premium: float = 0.0    # latest option premium (USD)
    notes: str = ""


@dataclass(frozen=True)
class ExitAlert:
    """One alert about a position."""

    level: str                      # "exit" | "warning" | "ok"
    rule: str                       # short-code: score | rn_drift | premium
    headline: str                   # one-line summary
    detail: str                     # longer rationale, includes numbers


@dataclass(frozen=True)
class PositionAssessment:
    """Result of evaluating one position."""

    position_id: str
    overall_level: str              # "exit" | "warning" | "ok"
    pnl_pct: float                  # premium pnl in % (current/entry − 1)
    alerts: list[ExitAlert] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE-STATE INPUT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LiveState:
    """Snapshot of live market state needed to evaluate exit conditions."""

    score_4h: float                 # current smoothed 4h composite score
    rn_mean: Optional[float]        # current RN mean (USD), None if unavailable
    spot: float                     # current BTC spot (USD)


# ─────────────────────────────────────────────────────────────────────────────
# RULE-BY-RULE EVALUATORS  (pure functions)
# ─────────────────────────────────────────────────────────────────────────────
def _eval_score_rule(pos: Position, live: LiveState) -> ExitAlert:
    """Score-deterioration rule.

    Long positions exit when score drops below +SCORE_EXIT_ABS.
    Short positions exit when score climbs above -SCORE_EXIT_ABS.
    Warning when within SCORE_WARN_BAND of the exit threshold.
    """
    is_long = pos.bias == "long"
    score = float(live.score_4h)

    if is_long:
        if score < SCORE_EXIT_ABS:
            return ExitAlert(
                level="exit",
                rule="score",
                headline=f"Score deteriorated → {score:+.0f}",
                detail=(f"4h score {score:+.0f} fell below the +{SCORE_EXIT_ABS:.0f} "
                        f"long-hold threshold. Directional edge has weakened."),
            )
        if score < SCORE_EXIT_ABS + SCORE_WARN_BAND:
            return ExitAlert(
                level="warning",
                rule="score",
                headline=f"Score weakening → {score:+.0f}",
                detail=(f"4h score {score:+.0f} is within {SCORE_WARN_BAND:.0f}pp "
                        f"of the +{SCORE_EXIT_ABS:.0f} long-hold threshold."),
            )
        return ExitAlert(
            level="ok",
            rule="score",
            headline=f"Score healthy {score:+.0f}",
            detail=f"Above +{SCORE_EXIT_ABS:.0f} long-hold threshold.",
        )

    # Short position
    if score > -SCORE_EXIT_ABS:
        return ExitAlert(
            level="exit",
            rule="score",
            headline=f"Score deteriorated → {score:+.0f}",
            detail=(f"4h score {score:+.0f} climbed above -{SCORE_EXIT_ABS:.0f} "
                    f"short-hold threshold. Directional edge has weakened."),
        )
    if score > -SCORE_EXIT_ABS - SCORE_WARN_BAND:
        return ExitAlert(
            level="warning",
            rule="score",
            headline=f"Score weakening → {score:+.0f}",
            detail=(f"4h score {score:+.0f} within {SCORE_WARN_BAND:.0f}pp of the "
                    f"-{SCORE_EXIT_ABS:.0f} short-hold threshold."),
        )
    return ExitAlert(
        level="ok",
        rule="score",
        headline=f"Score healthy {score:+.0f}",
        detail=f"Below -{SCORE_EXIT_ABS:.0f} short-hold threshold.",
    )


def _eval_rn_drift_rule(pos: Position, live: LiveState) -> Optional[ExitAlert]:
    """RN drift reversal rule.

    For longs: exit when RN mean has dropped below entry RN mean by more
    than RN_DRIFT_BAND_PCT% (i.e. the implied future shifted bearish).
    For shorts: exit when RN mean has risen above entry by the same band.
    Returns None when the rule cannot be evaluated (no entry RN, no live RN).
    """
    if pos.entry_rn_mean is None or live.rn_mean is None or pos.entry_rn_mean <= 0:
        return None

    drift_pct = (live.rn_mean - pos.entry_rn_mean) / pos.entry_rn_mean * 100.0
    is_long = pos.bias == "long"
    band = RN_DRIFT_BAND_PCT

    # Long: bad direction is "down". Short: bad direction is "up".
    bad_drift = -drift_pct if is_long else drift_pct

    if bad_drift > band * 2.0:
        return ExitAlert(
            level="exit",
            rule="rn_drift",
            headline=f"RN mean reversed → {drift_pct:+.2f}%",
            detail=(f"RN mean has moved {drift_pct:+.2f}% vs entry "
                    f"(${pos.entry_rn_mean:,.0f} → ${live.rn_mean:,.0f}). "
                    f"Implied direction has flipped against the position."),
        )
    if bad_drift > band:
        return ExitAlert(
            level="warning",
            rule="rn_drift",
            headline=f"RN drifting against → {drift_pct:+.2f}%",
            detail=(f"RN mean has moved {drift_pct:+.2f}% vs entry "
                    f"(${pos.entry_rn_mean:,.0f} → ${live.rn_mean:,.0f})."),
        )
    return ExitAlert(
        level="ok",
        rule="rn_drift",
        headline=f"RN drift {drift_pct:+.2f}%",
        detail="RN mean within tolerance band of entry.",
    )


def _eval_premium_rule(pos: Position) -> Optional[ExitAlert]:
    """Premium-loss rule. Returns None if no current premium has been set."""
    if pos.entry_premium <= 0 or pos.current_premium <= 0:
        return None
    pnl_pct = pos.current_premium / pos.entry_premium - 1.0
    if pnl_pct <= PREMIUM_LOSS_EXIT:
        return ExitAlert(
            level="exit",
            rule="premium",
            headline=f"Premium {pnl_pct*100:+.0f}%",
            detail=(f"Current premium ${pos.current_premium:,.2f} vs entry "
                    f"${pos.entry_premium:,.2f}. Loss exceeds "
                    f"{PREMIUM_LOSS_EXIT*100:.0f}% stop."),
        )
    if pnl_pct <= PREMIUM_LOSS_WARN:
        return ExitAlert(
            level="warning",
            rule="premium",
            headline=f"Premium {pnl_pct*100:+.0f}%",
            detail=(f"Premium drawdown {pnl_pct*100:+.0f}% — close to "
                    f"{PREMIUM_LOSS_EXIT*100:.0f}% exit threshold."),
        )
    return ExitAlert(
        level="ok",
        rule="premium",
        headline=f"Premium {pnl_pct*100:+.0f}%",
        detail=f"Premium drawdown is healthy.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_exit(pos: Position, live: LiveState) -> PositionAssessment:
    """Evaluate every applicable exit rule for one position. Returns the
    overall level (worst across rules) plus the list of individual alerts.
    """
    alerts: list[ExitAlert] = []

    score_alert = _eval_score_rule(pos, live)
    alerts.append(score_alert)

    rn_alert = _eval_rn_drift_rule(pos, live)
    if rn_alert is not None:
        alerts.append(rn_alert)

    prem_alert = _eval_premium_rule(pos)
    if prem_alert is not None:
        alerts.append(prem_alert)

    # Worst-of: any "exit" → exit; else any "warning" → warning; else ok
    if any(a.level == "exit" for a in alerts):
        overall = "exit"
    elif any(a.level == "warning" for a in alerts):
        overall = "warning"
    else:
        overall = "ok"

    pnl_pct = (
        (pos.current_premium / pos.entry_premium - 1.0) * 100.0
        if pos.entry_premium > 0 and pos.current_premium > 0
        else 0.0
    )

    return PositionAssessment(
        position_id=pos.id,
        overall_level=overall,
        pnl_pct=pnl_pct,
        alerts=alerts,
    )


def evaluate_portfolio(positions: list[Position], live: LiveState) -> list[PositionAssessment]:
    """Convenience wrapper: evaluate every position and return the list."""
    return [evaluate_exit(p, live) for p in positions]
