"""
btc_dashboard.analytics.smoothing
EMA smoothing + stability filtering for normalised signals and final scores.

Per-timeframe configuration
---------------------------
1h:
    signals     → EMA(span=6)
    final score → EMA(span=12)
    stability   → 4 consecutive bars

4h:
    signals     → EMA(span=4)
    final score → EMA(span=8)
    stability   → 2 consecutive bars

Inputs are *histories* (one row per timeframe bar). Outputs are the
smoothed histories plus per-series stability flags for the most recent bar.

A signal/score is considered "stable" when the last N bars of the smoothed
series all sit on the same side of zero AND each bar's magnitude is at
least `stability_threshold` (default 0.05 for signals, 5 for final score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SmoothingConfig:
    timeframe: str
    signal_ema_span: int
    final_ema_span: int
    stability_bars: int
    signal_stability_threshold: float = 0.05    # signals are in [-1, +1]
    final_stability_threshold: float = 5.0      # final score is in [-100, +100]


SMOOTHING_CONFIGS: dict[str, SmoothingConfig] = {
    "1h": SmoothingConfig(
        timeframe="1h",
        signal_ema_span=6,
        final_ema_span=12,
        stability_bars=4,
    ),
    "4h": SmoothingConfig(
        timeframe="4h",
        signal_ema_span=4,
        final_ema_span=8,
        stability_bars=2,
    ),
}


def get_config(timeframe: str) -> SmoothingConfig:
    """Return the smoothing config for a known timeframe (case-insensitive)."""
    key = str(timeframe).lower()
    if key not in SMOOTHING_CONFIGS:
        raise ValueError(
            f"unknown timeframe {timeframe!r}; "
            f"known: {sorted(SMOOTHING_CONFIGS)}"
        )
    return SMOOTHING_CONFIGS[key]


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SmoothingResult:
    timeframe: str
    config: SmoothingConfig
    smoothed_signals: pd.DataFrame      # full history, one column per signal
    smoothed_final: pd.Series           # full history of smoothed final score
    latest_signals: dict[str, float]    # latest smoothed value per signal
    latest_final: float                 # latest smoothed final score
    stability_flags: dict[str, bool]    # per-signal stability of latest bar
    final_stability: bool               # stability of the final score


# ─────────────────────────────────────────────────────────────────────────────
# CORE EMA
# ─────────────────────────────────────────────────────────────────────────────
def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponentially-weighted moving average with `span` (pandas convention,
    α = 2 / (span + 1), `adjust=False` for the recursive form most traders
    intuit). Preserves the input index and name."""
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    s = pd.Series(series, copy=False)
    out = s.ewm(span=span, adjust=False).mean()
    out.name = s.name
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SMOOTHERS
# ─────────────────────────────────────────────────────────────────────────────
def smooth_signals(history: pd.DataFrame, span: int) -> pd.DataFrame:
    """Column-wise EMA over a signal history.

    `history` is a DataFrame indexed by bar timestamp with one column per
    signal name (values in [-1, +1]). Returns a same-shape DataFrame of
    smoothed values.
    """
    if history is None or history.empty:
        return pd.DataFrame()
    df = history.copy()
    smoothed = pd.DataFrame(index=df.index)
    for col in df.columns:
        smoothed[col] = ema(df[col].astype("float64"), span)
    return smoothed


def smooth_final_score(history: pd.Series, span: int) -> pd.Series:
    """EMA over a final-score history (values in [-100, +100])."""
    if history is None or history.empty:
        return pd.Series(dtype="float64")
    return ema(history.astype("float64"), span)


# ─────────────────────────────────────────────────────────────────────────────
# STABILITY FILTER
# ─────────────────────────────────────────────────────────────────────────────
def stability_flag(
    series: pd.Series,
    n_bars: int,
    threshold: float = 0.0,
) -> bool:
    """Return True iff the last `n_bars` of `series` are all:
      • finite,
      • on the same side of zero (no sign flip),
      • and each |value| ≥ `threshold`.

    Returns False if the series is shorter than `n_bars`.
    """
    if series is None or n_bars <= 0:
        return False
    s = pd.Series(series, copy=False).dropna()
    if len(s) < n_bars:
        return False
    tail = s.iloc[-n_bars:].to_numpy(dtype="float64")
    if not np.all(np.isfinite(tail)):
        return False
    signs = np.sign(tail)
    if signs[-1] == 0:
        return False
    if not np.all(signs == signs[-1]):
        return False
    if np.any(np.abs(tail) < threshold):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# HIGH-LEVEL ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def apply_smoothing(
    signal_history: pd.DataFrame,
    final_history: pd.Series,
    timeframe: str,
    config: Optional[SmoothingConfig] = None,
) -> SmoothingResult:
    """Smooth a full history and compute stability flags for the most recent bar.

    Parameters
    ----------
    signal_history : DataFrame
        One row per bar, one column per normalised signal (values in [-1, +1]).
    final_history : Series
        One value per bar, final composite score (values in [-100, +100]).
    timeframe : "1h" | "4h"
        Selects EMA spans and stability bar count from `SMOOTHING_CONFIGS`.
    config : SmoothingConfig, optional
        Override the registered config for ad-hoc tuning.

    Returns
    -------
    SmoothingResult
    """
    cfg = config or get_config(timeframe)

    sig_smoothed = smooth_signals(signal_history, cfg.signal_ema_span)
    fin_smoothed = smooth_final_score(final_history, cfg.final_ema_span)

    # Latest values
    latest_signals: dict[str, float] = {}
    if not sig_smoothed.empty:
        last_row = sig_smoothed.iloc[-1]
        for col in sig_smoothed.columns:
            v = last_row[col]
            latest_signals[col] = float(v) if pd.notna(v) else 0.0

    latest_final = float(fin_smoothed.iloc[-1]) if not fin_smoothed.empty else 0.0
    if not np.isfinite(latest_final):
        latest_final = 0.0

    # Stability flags
    stability_flags: dict[str, bool] = {}
    for col in sig_smoothed.columns:
        stability_flags[col] = stability_flag(
            sig_smoothed[col],
            n_bars=cfg.stability_bars,
            threshold=cfg.signal_stability_threshold,
        )

    final_stab = stability_flag(
        fin_smoothed,
        n_bars=cfg.stability_bars,
        threshold=cfg.final_stability_threshold,
    )

    return SmoothingResult(
        timeframe=cfg.timeframe,
        config=cfg,
        smoothed_signals=sig_smoothed,
        smoothed_final=fin_smoothed,
        latest_signals=latest_signals,
        latest_final=latest_final,
        stability_flags=stability_flags,
        final_stability=final_stab,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def stability_to_dataframe(result: SmoothingResult) -> pd.DataFrame:
    """One row per signal: name, latest smoothed value, stable? flag.
    The final score is appended on the last row as `__final__`."""
    rows = [
        {
            "name": name,
            "value": result.latest_signals.get(name, 0.0),
            "stable": bool(flag),
        }
        for name, flag in result.stability_flags.items()
    ]
    rows.append(
        {
            "name": "__final__",
            "value": result.latest_final,
            "stable": bool(result.final_stability),
        }
    )
    return pd.DataFrame(rows)
