"""
btc_dashboard.analytics.normalization
Map raw inputs to normalised signals in the closed interval [-1, +1].

Conventions
-----------
- Output sign: positive = bullish, negative = bearish.
- Output magnitude: 0 = neutral, ±1 = saturated extreme.
- No smoothing, no aggregation, no hysteresis. One raw observation in,
  one signal out. Smoothing belongs in a downstream layer.
- Scales (the value at which the signal saturates to ±1) are exposed as
  keyword arguments so callers can recalibrate without changing the math.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a scalar to [lo, hi]."""
    if x is None or not np.isfinite(x):
        return 0.0
    return float(max(lo, min(hi, x)))


def _linear(x: float, scale: float) -> float:
    """x / scale, clipped to [-1, +1]. Saturates linearly at ±scale."""
    if scale <= 0:
        return 0.0
    return _clip(x / scale)


def _tanh(x: float, scale: float) -> float:
    """tanh(x / scale). Smooth saturation; ~76% at ±scale, ~96% at ±2·scale."""
    if scale <= 0:
        return 0.0
    return float(np.tanh(x / scale))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ OPTIONS                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def normalize_rn_mean_vs_spot(
    rn_mean: float,
    spot: float,
    rn_std: Optional[float] = None,
    pct_scale: float = 5.0,
) -> float:
    """Risk-neutral mean vs spot.

    If `rn_std` is given, use a z-score (gap / 1σ). Otherwise fall back
    to percentage gap saturating at ±`pct_scale` percent.

    Positive output = market-implied mean above spot (bullish).
    """
    if spot is None or spot <= 0 or rn_mean is None or not np.isfinite(rn_mean):
        return 0.0
    gap = rn_mean - spot
    if rn_std and rn_std > 0:
        return _tanh(gap / rn_std, scale=1.0)
    pct = (gap / spot) * 100.0
    return _linear(pct, scale=pct_scale)


def normalize_rn_probability(p_above_spot: float) -> float:
    """Market-implied P(price > spot at expiry) → [-1, +1].
    Maps [0, 1] linearly to [-1, +1] via 2p − 1. p=0.50 → 0.
    """
    if p_above_spot is None or not np.isfinite(p_above_spot):
        return 0.0
    return _clip(2.0 * float(p_above_spot) - 1.0)


def normalize_rn_drift(
    rn_mean_curr: float,
    rn_mean_prev: float,
    spot: float,
    pct_scale: float = 1.0,
) -> float:
    """Drift in the RN mean between snapshots, normalised by spot.

    Positive = RN mean shifted up since previous snapshot (bullish flow).
    Saturates at ±`pct_scale` percent of spot.
    """
    if spot is None or spot <= 0:
        return 0.0
    if rn_mean_curr is None or rn_mean_prev is None:
        return 0.0
    drift_pct = ((rn_mean_curr - rn_mean_prev) / spot) * 100.0
    return _linear(drift_pct, scale=pct_scale)


def normalize_skew(skew: float, scale: float = 2.0) -> float:
    """Risk-neutral distribution skew.

    Positive skew (right-tail heavy) = market prices upside fatter → bullish.
    Negative skew (left-tail heavy) = market prices crash risk → bearish.
    Saturates at ±`scale` (typical RN skew range is ~±2).
    """
    if skew is None or not np.isfinite(skew):
        return 0.0
    return _linear(float(skew), scale=scale)


def normalize_gex(
    gex_usd_per_pct: float,
    gex_threshold_usd: float = 5.0e9,
) -> float:
    """Dealer net gamma exposure (in USD per 1% spot move).

    Positive GEX = dealers are long gamma; flows suppress volatility and
    create a stable / supportive regime → mildly bullish.
    Negative GEX = dealers are short gamma; flows amplify moves → fragile.

    Saturates at ±`gex_threshold_usd`. Callers should compute GEX upstream
    (sum over chain of γ × OI × spot² / 100, signed by dealer convention).
    """
    if gex_usd_per_pct is None or not np.isfinite(gex_usd_per_pct):
        return 0.0
    return _linear(float(gex_usd_per_pct), scale=gex_threshold_usd)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FUTURES                                                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def normalize_oi_vs_price_quadrant(
    spot_pct: float,
    oi_pct: float,
    spot_scale: float = 2.0,
    oi_scale: float = 1.0,
) -> float:
    """Quadrant of (Δspot, ΔOI) over a comparable window.

    Confirmation regime (same sign): new positions in the trend → trend score
    in the direction of spot, magnitude = √(|spot|·|oi|).
    Divergence regime (opposite sign): unwind / squeeze → half-magnitude
    contrarian score (still signed by spot direction so callers can see
    "rally on falling OI = weak rally").

    Inputs are percentages (e.g. spot_pct=0.8 means +0.8%).
    """
    if spot_pct is None or oi_pct is None:
        return 0.0
    s_norm = _linear(spot_pct, scale=spot_scale)
    o_norm = _linear(oi_pct, scale=oi_scale)
    if s_norm == 0.0 or o_norm == 0.0:
        return 0.0
    s_dir = math.copysign(1.0, s_norm)
    mag = math.sqrt(abs(s_norm) * abs(o_norm))
    if math.copysign(1.0, s_norm) == math.copysign(1.0, o_norm):
        return _clip(s_dir * mag)
    return _clip(-0.5 * s_dir * mag)


def normalize_funding(
    funding_rate_per_8h: float,
    extreme_ann_pct: float = 30.0,
    saturation_ann_pct: float = 60.0,
) -> float:
    """Perp funding rate.

    Below `extreme_ann_pct` (annualised %): mild flow signal in the same
    direction as the funding sign, saturating linearly.
    Above `extreme_ann_pct`: crowded trade → contrarian signal in the
    opposite direction, saturating linearly to `saturation_ann_pct`.

    `funding_rate_per_8h` is the decimal rate for one 8-hour period
    (Binance convention). Annualised% = rate × 3 × 365 × 100.
    """
    if funding_rate_per_8h is None or not np.isfinite(funding_rate_per_8h):
        return 0.0
    ann_pct = float(funding_rate_per_8h) * 3.0 * 365.0 * 100.0
    s = math.copysign(1.0, ann_pct) if ann_pct != 0 else 0.0
    a = abs(ann_pct)
    if a <= extreme_ann_pct:
        return _linear(ann_pct, scale=extreme_ann_pct)
    # Contrarian regime
    over = min(a - extreme_ann_pct, saturation_ann_pct - extreme_ann_pct)
    contr_mag = over / max(saturation_ann_pct - extreme_ann_pct, 1e-9)
    return _clip(-s * contr_mag)


def normalize_basis(basis_pct_annualised: float, scale_pct: float = 5.0) -> float:
    """Futures-vs-spot basis, annualised %.

    Positive basis (contango) = bullish carry; deep backwardation = stress.
    Saturates at ±`scale_pct` annualised percent.
    """
    if basis_pct_annualised is None or not np.isfinite(basis_pct_annualised):
        return 0.0
    return _linear(float(basis_pct_annualised), scale=scale_pct)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ SPOT                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def normalize_trend_structure(
    closes: Sequence[float],
    fast: int = 20,
    mid: int = 50,
    slow: int = 200,
) -> float:
    """SMA-stack alignment score.

    +1 when price > SMA(fast) > SMA(mid) > SMA(slow) (full bullish stack)
    −1 when the entire stack is inverted
    Intermediate = partial alignment (each of 4 alignments contributes 0.25).
    """
    s = pd.Series(list(closes), dtype="float64").dropna()
    if len(s) < slow:
        return 0.0
    price = float(s.iloc[-1])
    sma_f = float(s.tail(fast).mean())
    sma_m = float(s.tail(mid).mean())
    sma_s = float(s.tail(slow).mean())
    score = 0.0
    score += 0.25 if price > sma_f else -0.25
    score += 0.25 if sma_f > sma_m else -0.25
    score += 0.25 if sma_m > sma_s else -0.25
    score += 0.25 if price > sma_s else -0.25
    return _clip(score)


def normalize_volume_confirmation(
    closes: Sequence[float],
    volumes: Sequence[float],
    avg_window: int = 20,
) -> float:
    """Volume on the most recent bar relative to a rolling baseline,
    signed by the direction of the last bar's price change.

    +1 = strong volume in a rising bar (or strong volume in a falling bar
    inverted to negative). 0 = volume in line with baseline or no movement.
    """
    c = pd.Series(list(closes), dtype="float64").dropna()
    v = pd.Series(list(volumes), dtype="float64").dropna()
    n = min(len(c), len(v))
    if n < max(avg_window + 1, 2):
        return 0.0
    c = c.iloc[-n:]
    v = v.iloc[-n:]
    last_chg = float(c.iloc[-1]) - float(c.iloc[-2])
    direction = math.copysign(1.0, last_chg) if last_chg != 0 else 0.0
    base = float(v.iloc[-(avg_window + 1):-1].mean())
    if base <= 0:
        return 0.0
    rel = float(v.iloc[-1]) / base - 1.0  # e.g. 0.5 = 50% above baseline
    return _clip(direction * rel)


def normalize_vwap_distance(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    pct_scale: float = 1.0,
) -> float:
    """Distance of last close from session typical-price VWAP, in %.

    +1 = close is `pct_scale`% (or more) above VWAP (bullish premium).
    −1 = below VWAP by the same magnitude (bearish discount).
    """
    h = pd.Series(list(highs), dtype="float64")
    l = pd.Series(list(lows), dtype="float64")
    c = pd.Series(list(closes), dtype="float64")
    v = pd.Series(list(volumes), dtype="float64")
    n = min(len(h), len(l), len(c), len(v))
    if n < 2:
        return 0.0
    h, l, c, v = h.iloc[-n:], l.iloc[-n:], c.iloc[-n:], v.iloc[-n:]
    typ = (h + l + c) / 3.0
    denom = float(v.sum())
    if denom <= 0:
        return 0.0
    vwap = float((typ * v).sum() / denom)
    if vwap <= 0:
        return 0.0
    pct = (float(c.iloc[-1]) - vwap) / vwap * 100.0
    return _linear(pct, scale=pct_scale)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ LIQUIDITY                                                                ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def normalize_distance_to_liquidity(
    price: float,
    liquidity_levels: Sequence[Tuple[float, float]],
) -> float:
    """Asymmetry between distance to nearest support and nearest resistance.

    `liquidity_levels` is a sequence of (level_price, weight) pairs — typical
    inputs are large OI / book-cluster strikes. The function picks the nearest
    weighted level below and above `price` and returns:

        (d_above − d_below) / (d_above + d_below)

    +1 = right at support (next big level is far above) → bullish bounce setup
    −1 = right at resistance → bearish rejection setup
     0 = equidistant
    """
    if price is None or price <= 0 or not liquidity_levels:
        return 0.0
    below: Optional[Tuple[float, float]] = None
    above: Optional[Tuple[float, float]] = None
    for lvl, w in liquidity_levels:
        if lvl <= 0 or w <= 0:
            continue
        if lvl < price and (below is None or lvl > below[0]):
            below = (lvl, w)
        if lvl > price and (above is None or lvl < above[0]):
            above = (lvl, w)
    if below is None and above is None:
        return 0.0
    if below is None:
        return -1.0  # only resistance ahead
    if above is None:
        return 1.0   # only support behind
    d_below = (price - below[0]) / price
    d_above = (above[0] - price) / price
    if d_below + d_above <= 0:
        return 0.0
    return _clip((d_above - d_below) / (d_above + d_below))


def normalize_orderbook_imbalance(
    bids: Sequence[Tuple[float, float]],
    asks: Sequence[Tuple[float, float]],
    depth_levels: Optional[int] = None,
) -> float:
    """Top-of-book size imbalance over the first `depth_levels` levels.

    score = (Σ bid_size − Σ ask_size) / (Σ bid_size + Σ ask_size)

    +1 = pure bid-side; −1 = pure ask-side; 0 = balanced.
    """
    bb = list(bids)[:depth_levels] if depth_levels else list(bids)
    aa = list(asks)[:depth_levels] if depth_levels else list(asks)
    bid_sz = sum(float(s) for _p, s in bb if s and float(s) > 0)
    ask_sz = sum(float(s) for _p, s in aa if s and float(s) > 0)
    if bid_sz + ask_sz <= 0:
        return 0.0
    return _clip((bid_sz - ask_sz) / (bid_sz + ask_sz))


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║ FLOW                                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
def normalize_exchange_flows(
    net_inflow_btc: float,
    baseline_btc: float = 1000.0,
) -> float:
    """On-chain coins moving onto/off exchanges over a fixed window.

    Net INFLOW (positive `net_inflow_btc`) = coins moving onto exchanges
    = sell pressure → bearish.
    Net OUTFLOW (negative) = withdrawal to wallets / cold storage → bullish.

    Saturates at ±`baseline_btc` BTC per window.
    """
    if net_inflow_btc is None or not np.isfinite(net_inflow_btc):
        return 0.0
    return _linear(-float(net_inflow_btc), scale=baseline_btc)


def normalize_whale_activity(
    whale_buy_notional: float,
    whale_sell_notional: float,
) -> float:
    """Aggressor-side imbalance among whale prints (large trades).

    score = (buy − sell) / (buy + sell)

    Caller defines what qualifies as a "whale" trade upstream (e.g. notional
    above the 99th percentile of recent prints) and passes the two summed
    notionals here.

    +1 = pure whale buying; −1 = pure whale selling; 0 = balanced or no whales.
    """
    b = float(whale_buy_notional or 0.0)
    s = float(whale_sell_notional or 0.0)
    if b + s <= 0:
        return 0.0
    return _clip((b - s) / (b + s))


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY (introspection — handy for unit tests and UI tables)
# ─────────────────────────────────────────────────────────────────────────────
NORMALIZERS = {
    "options": [
        "normalize_rn_mean_vs_spot",
        "normalize_rn_probability",
        "normalize_rn_drift",
        "normalize_skew",
        "normalize_gex",
    ],
    "futures": [
        "normalize_oi_vs_price_quadrant",
        "normalize_funding",
        "normalize_basis",
    ],
    "spot": [
        "normalize_trend_structure",
        "normalize_volume_confirmation",
        "normalize_vwap_distance",
    ],
    "liquidity": [
        "normalize_distance_to_liquidity",
        "normalize_orderbook_imbalance",
    ],
    "flow": [
        "normalize_exchange_flows",
        "normalize_whale_activity",
    ],
}
