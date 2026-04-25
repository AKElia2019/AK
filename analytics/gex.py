"""
btc_dashboard.analytics.gex
Dealer gamma exposure (GEX) computed from an options chain.

Convention used here
--------------------
The standard market-maker assumption is that dealers are net long puts
(bought from retail) and net short calls (sold to retail). Under that
assumption, dealer gamma per option contract is:

    +γ × OI for puts        (dealers long → positive gamma)
    −γ × OI for calls       (dealers short → negative gamma)

Positive total GEX → dealers are net long gamma → hedging suppresses
volatility (mean-reverting flow). Negative GEX → dealers are net short
gamma → hedging amplifies moves (trend-accelerating flow).

GEX is reported in USD per 1% spot move:

    GEX_USD_per_pct = Σ ( ±γ × OI × spot² / 100 )

Callers should pass the chain DataFrame produced by
`data.options.fetch_deribit_option_chain` or equivalent — columns expected:
`type`, `strike`, `iv`, `open_interest`, `dte`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_SQRT_2PI = math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class GEXResult:
    gex_usd_per_pct: float       # signed total dealer gamma in USD/1% move
    gex_call_usd_per_pct: float  # call contribution (negative under convention)
    gex_put_usd_per_pct: float   # put contribution (positive under convention)
    by_strike: pd.DataFrame      # per-strike breakdown
    flip_strike: Optional[float] # strike where cumulative GEX crosses zero
    n_options: int


def _bs_gamma(spot: float, strike: float, t_years: float, iv_decimal: float, r: float) -> float:
    """Black-Scholes gamma per 1 underlying unit."""
    if (
        spot <= 0
        or strike <= 0
        or t_years <= 0
        or iv_decimal <= 0
        or not np.isfinite(spot)
        or not np.isfinite(strike)
        or not np.isfinite(iv_decimal)
    ):
        return 0.0
    d1 = (math.log(spot / strike) + (r + 0.5 * iv_decimal * iv_decimal) * t_years) / (
        iv_decimal * math.sqrt(t_years)
    )
    n_prime_d1 = math.exp(-0.5 * d1 * d1) / _SQRT_2PI
    return n_prime_d1 / (spot * iv_decimal * math.sqrt(t_years))


def compute_gex(
    chain: pd.DataFrame,
    spot: float,
    r: float = 0.05,
) -> Optional[GEXResult]:
    """Compute dealer gamma exposure across the chain.

    Returns None when the chain is empty or unusable.
    """
    if chain is None or chain.empty or spot <= 0:
        return None

    required = {"type", "strike", "iv", "open_interest", "dte"}
    if not required.issubset(chain.columns):
        return None

    df = chain.copy()
    df["type"] = df["type"].astype(str).str.upper()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce").fillna(0.0)
    df["dte"] = pd.to_numeric(df["dte"], errors="coerce")
    df = df.dropna(subset=["strike", "iv", "dte"])
    df = df[(df["iv"] > 0) & (df["dte"] > 0) & (df["strike"] > 0)]
    if df.empty:
        return None

    spot_sq_over_100 = (float(spot) ** 2) / 100.0

    rows = []
    total_call = 0.0
    total_put = 0.0
    for _, row in df.iterrows():
        t_years = float(row["dte"]) / 365.0
        iv_dec = float(row["iv"]) / 100.0
        gamma = _bs_gamma(float(spot), float(row["strike"]), t_years, iv_dec, r)
        contribution = gamma * float(row["open_interest"]) * spot_sq_over_100
        if row["type"] == "CALL":
            signed = -contribution     # dealers short calls
            total_call += signed
        elif row["type"] == "PUT":
            signed = +contribution     # dealers long puts
            total_put += signed
        else:
            continue
        rows.append(
            {
                "strike": float(row["strike"]),
                "type": row["type"],
                "gamma": gamma,
                "open_interest": float(row["open_interest"]),
                "gex_usd_per_pct": signed,
            }
        )
    if not rows:
        return None

    by_strike = (
        pd.DataFrame(rows)
        .groupby("strike", as_index=False)["gex_usd_per_pct"]
        .sum()
        .sort_values("strike")
        .reset_index(drop=True)
    )

    # GEX flip strike: where cumulative GEX (from low strikes up) crosses zero
    cumulative = by_strike["gex_usd_per_pct"].cumsum().to_numpy()
    flip_strike: Optional[float] = None
    for i in range(1, len(cumulative)):
        if cumulative[i - 1] == 0:
            continue
        if (cumulative[i - 1] < 0 and cumulative[i] >= 0) or (
            cumulative[i - 1] > 0 and cumulative[i] <= 0
        ):
            flip_strike = float(by_strike["strike"].iloc[i])
            break

    return GEXResult(
        gex_usd_per_pct=total_call + total_put,
        gex_call_usd_per_pct=total_call,
        gex_put_usd_per_pct=total_put,
        by_strike=by_strike,
        flip_strike=flip_strike,
        n_options=len(rows),
    )
