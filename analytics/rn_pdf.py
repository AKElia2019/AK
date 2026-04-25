"""
btc_dashboard.analytics.rn_pdf
Risk-neutral probability density via Breeden–Litzenberger.

Method
------
Breeden & Litzenberger (1978): the risk-neutral density of the underlying
at expiry equals the second derivative of the European call price with
respect to strike, scaled by e^(rT):

    f(K) = e^(rT) · ∂²C / ∂K²

We fit a smoothed cubic spline to call-mark prices vs. strike, take its
second derivative on a dense grid, smooth lightly with a Gaussian filter
(to suppress arbitrage-driven kinks), and renormalise to a probability
density.

Inputs
------
A chain DataFrame with at least these columns (lowercase, per
`data.options.OPTION_CHAIN_SCHEMA`):

    type        "CALL" | "PUT"
    strike      float
    mark        float — call mark price; on Deribit this is in BTC, so
                callers should multiply by spot before passing in if they
                want USD strikes returned. The function below treats
                `mark` as the dollar call premium directly. The pipeline
                module handles the BTC→USD conversion before calling.
    dte         days to expiry

Returns a dict (or None) with:

    K       1-D array of strike grid in USD
    pdf     1-D risk-neutral density on K (∫pdf dK = 1)
    cdf     1-D cumulative on K
    mean    float — RN expected price at expiry
    std     float — RN standard deviation
    skew    float — RN skewness
    mode    float — strike with maximum density
    p_above_spot  float — RN probability that price > current spot
    n       int — number of call strikes used
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

try:
    from scipy.interpolate import CubicSpline
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


def compute_oi_adjusted_pdf(
    chain: pd.DataFrame,
    base_pdf: dict,
    spot: float,
    oi_strength: float = 0.45,
    oi_smooth: float = 1.5,
) -> Optional[dict]:
    """Tilt a Breeden-Litzenberger RN density with the open-interest profile.

    The OI overlay smooths option open-interest across strikes, z-scores it,
    and applies a multiplicative tilt to the base RN density. The result is
    a *positioning-adjusted* implied distribution — not a true risk-neutral
    density, but a useful read on where the chain is *crowded* vs. the
    market-implied probability mass.

    Returns a dict with the same keys as `compute_rn_pdf` plus:
        oi_curve     smoothed OI evaluated on K
        oi_zone      strike with the largest OI cluster
        tilt         the multiplicative tilt vector applied to base_pdf
        strength     `oi_strength` (echoed)
    """
    if not _HAS_SCIPY or not base_pdf or chain is None or chain.empty:
        return None
    if "open_interest" not in chain.columns or "strike" not in chain.columns:
        return None

    K = np.asarray(base_pdf["K"], dtype=float)
    base = np.asarray(base_pdf["pdf"], dtype=float)
    if len(K) < 5 or len(base) != len(K):
        return None

    work = chain.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["open_interest"] = pd.to_numeric(work["open_interest"], errors="coerce")
    work = work.dropna(subset=["strike"])
    if work.empty:
        return None

    oi_by_strike = (
        work.groupby("strike")["open_interest"]
        .sum(min_count=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_index()
    )
    if oi_by_strike.empty or float(oi_by_strike.sum()) <= 0:
        return None

    strikes = oi_by_strike.index.to_numpy(dtype=float)
    oi_vals = oi_by_strike.to_numpy(dtype=float)

    oi_sigma = oi_smooth * max(1.0, 10.0 / max(len(strikes), 1))
    oi_smoothed = gaussian_filter1d(oi_vals, sigma=oi_sigma)
    oi_interp = np.interp(K, strikes, oi_smoothed, left=0.0, right=0.0)

    if not np.isfinite(oi_interp).any() or float(np.sum(oi_interp)) <= 0:
        return None

    mu = float(np.mean(oi_interp))
    sd = float(np.std(oi_interp))
    z = np.zeros_like(oi_interp) if sd <= 1e-12 else (oi_interp - mu) / sd
    z = np.clip(z, -3.0, 3.0)

    tilt = np.maximum(0.05, 1.0 + oi_strength * z)
    adj_density = np.maximum(base * tilt, 0.0)

    area = float(_trapz(adj_density, K))
    if not np.isfinite(area) or area <= 0:
        return None

    adj_pdf = adj_density / area
    dK = float(K[1] - K[0])
    adj_cdf = np.cumsum(adj_pdf) * dK
    if adj_cdf[-1] <= 0:
        return None
    adj_cdf = adj_cdf / adj_cdf[-1]

    mean = float(_trapz(K * adj_pdf, K))
    var = float(_trapz((K - mean) ** 2 * adj_pdf, K))
    std = math.sqrt(max(var, 0.0))
    skew = (
        float(_trapz(((K - mean) / (std + 1e-9)) ** 3 * adj_pdf, K))
        if std > 0 else 0.0
    )
    mode = float(K[int(np.argmax(adj_pdf))])
    oi_zone = float(K[int(np.argmax(oi_interp))])

    if spot < K[0]:
        p_above = 1.0
    elif spot > K[-1]:
        p_above = 0.0
    else:
        idx = int(np.searchsorted(K, float(spot)))
        idx = max(0, min(idx, len(adj_cdf) - 1))
        p_above = float(1.0 - adj_cdf[idx])

    return {
        "K": K,
        "pdf": adj_pdf,
        "cdf": adj_cdf,
        "mean": mean,
        "std": std,
        "skew": skew,
        "mode": mode,
        "oi_curve": oi_interp,
        "oi_zone": oi_zone,
        "tilt": tilt,
        "strength": float(oi_strength),
        "oi_smooth": float(oi_smooth),
        "p_above_spot": p_above,
        "dte": base_pdf.get("dte"),
        "n": base_pdf.get("n"),
    }


def compute_rn_pdf(
    chain: pd.DataFrame,
    spot: float,
    expiry: Optional[str] = None,
    r: float = 0.05,
    smooth_sigma: float = 1.5,
    n_grid: int = 500,
) -> Optional[dict]:
    """Fit a risk-neutral density from the call slice of the chain.

    If `expiry` is provided (matching the `expiry` or `instrument` column
    convention), only that expiry is used. Otherwise the nearest expiry
    by median DTE is selected automatically.

    Returns None when there aren't enough call strikes (need ≥ 4) or when
    scipy isn't available.
    """
    if not _HAS_SCIPY:
        return None
    if chain is None or chain.empty or spot <= 0:
        return None

    required = {"type", "strike", "mark", "dte"}
    if not required.issubset(chain.columns):
        return None

    work = chain.copy()
    work["type"] = work["type"].astype(str).str.upper().str.strip()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["mark"] = pd.to_numeric(work["mark"], errors="coerce")
    work["dte"] = pd.to_numeric(work["dte"], errors="coerce")

    # Pick expiry. If `expiry` column exists, prefer the explicitly named one,
    # else use the soonest non-zero DTE.
    if expiry is not None and "expiry" in work.columns:
        sub = work[work["expiry"] == expiry].copy()
        if sub.empty:
            sub = work
    else:
        sub = work
    if "expiry" in sub.columns and sub["expiry"].notna().any() and expiry is None:
        # Group by expiry, choose the first one with ≥4 valid call strikes
        chosen = None
        for exp_val, group in sub.groupby("expiry"):
            calls_g = group[group["type"] == "CALL"].dropna(subset=["mark", "strike"])
            if len(calls_g) >= 4:
                chosen = group
                break
        if chosen is not None:
            sub = chosen

    calls = (
        sub[sub["type"] == "CALL"]
        .dropna(subset=["mark", "strike"])
        .sort_values("strike")
        .drop_duplicates("strike")
    )
    if len(calls) < 4:
        return None

    K = calls["strike"].to_numpy(dtype=float)
    C = calls["mark"].to_numpy(dtype=float)

    dte_med = (
        float(calls["dte"].median()) if calls["dte"].notna().any() else 0.0
    )
    T = max(dte_med / 365.0, 1e-4)

    try:
        cs = CubicSpline(K, C, bc_type="natural", extrapolate=False)
    except Exception:
        return None

    span = K[-1] - K[0]
    if not np.isfinite(span) or span <= 0:
        return None

    # Build a dense grid; trim 5% off each end to avoid spline edge artefacts
    margin = span * 0.05
    left = K[0] + margin
    right = K[-1] - margin
    if right <= left:
        left, right = K[0], K[-1]

    Kg = np.linspace(left, right, n_grid)
    if len(Kg) < 2:
        return None
    dK = float(Kg[1] - Kg[0])

    sigma_filter = smooth_sigma * max(1.0, 10.0 / len(K))
    try:
        density = gaussian_filter1d(
            np.maximum(cs(Kg, 2), 0.0) * math.exp(r * T),
            sigma=sigma_filter,
        )
    except Exception:
        return None

    density = np.maximum(np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    area = float(_trapz(density, Kg))
    if not np.isfinite(area) or area <= 0:
        return None

    pdf = density / area
    cdf = np.cumsum(pdf) * dK
    if cdf[-1] <= 0:
        return None
    cdf = cdf / cdf[-1]

    mean = float(_trapz(Kg * pdf, Kg))
    var = float(_trapz((Kg - mean) ** 2 * pdf, Kg))
    std = math.sqrt(max(var, 0.0))
    skew = (
        float(_trapz(((Kg - mean) / (std + 1e-9)) ** 3 * pdf, Kg))
        if std > 0
        else 0.0
    )
    mode = float(Kg[int(np.argmax(pdf))])

    # P(price > spot) at expiry
    if spot < Kg[0]:
        p_above = 1.0
    elif spot > Kg[-1]:
        p_above = 0.0
    else:
        idx = int(np.searchsorted(Kg, float(spot)))
        idx = max(0, min(idx, len(cdf) - 1))
        p_above = float(1.0 - cdf[idx])

    return {
        "K": Kg,
        "pdf": pdf,
        "cdf": cdf,
        "mean": mean,
        "std": std,
        "skew": skew,
        "mode": mode,
        "p_above_spot": p_above,
        "dte": dte_med,
        "n": len(K),
    }
