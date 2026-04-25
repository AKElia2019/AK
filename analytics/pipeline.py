"""
btc_dashboard.analytics.pipeline
End-to-end live signal pipeline.

    raw venue data
        →  per-signal normalization      (analytics.normalization)
        →  per-family aggregation         (analytics.scoring)
        →  composite score                (-100..+100)
        →  regime classification          (analytics.regime)
        →  recommendation                 (analytics.recommendation)
        →  trade plan                     (analytics.position_sizing)

This module is the wiring layer. It does not invent signal logic — every
math step is delegated to the modules above. Its job is fetching live data
from the connectors, building feature vectors per timeframe, and routing
the outputs into the existing analytics layers.

Failure mode: any connector that returns mock or empty data simply yields
a 0 (neutral) contribution to its signal. The pipeline never crashes — it
degrades gracefully and the resulting signal table flags the source.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from analytics.normalization import (
    normalize_basis,
    normalize_exchange_flows,
    normalize_funding,
    normalize_oi_vs_price_quadrant,
    normalize_orderbook_imbalance,
    normalize_rn_drift,
    normalize_rn_mean_vs_spot,
    normalize_rn_probability,
    normalize_skew,
    normalize_trend_structure,
    normalize_volume_confirmation,
    normalize_vwap_distance,
    normalize_whale_activity,
)
from analytics.scoring import (
    DEFAULT_FAMILY_WEIGHTS,
    CompositeScore,
    compute_composite,
)
from analytics.regime import RegimeAssessment, RegimeInputs, classify_regime
from analytics.recommendation import (
    Recommendation,
    RecommendationInputs,
    evaluate_recommendation,
)
from analytics.position_sizing import SizingInputs, TradePlan, build_trade_plan
from analytics.gex import GEXResult, compute_gex
from analytics.rn_pdf import compute_oi_adjusted_pdf, compute_rn_pdf
from analytics.smoothing import (
    SmoothingResult,
    apply_smoothing,
    get_config as _get_smoothing_config,
)

from data.spot import fetch_binance_spot_ticker
from data.futures import (
    fetch_binance_funding_history,
    fetch_binance_open_interest_hist,
    fetch_binance_perp_klines,
)
from data.options import fetch_deribit_option_chain
from data.coinglass import (
    fetch_coinglass_aggregated_oi,
    fetch_coinglass_funding_oi_weighted,
    fetch_coinglass_liquidations,
    fetch_coinglass_long_short_ratio,
)


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineResult:
    """Everything the UI layer needs to render."""

    # Market state
    spot: float
    atm_iv_pct: Optional[float]
    rn_mean: Optional[float]
    rn_p_above_spot: Optional[float]
    rn_curve: Optional[dict] = None    # full BL fit dict (K, pdf, cdf, std, mode, …)

    # OI-adjusted RN curve — same shape as `rn_curve` but density is tilted by
    # the open-interest profile across strikes.
    rn_oi_curve: Optional[dict] = None
    rn_oi_mean: Optional[float] = None
    rn_oi_p_above_spot: Optional[float] = None

    # Composite scores per timeframe (in [-100, +100], EMA-smoothed)
    score_1h: float
    score_4h: float
    score_1h_raw: float           # un-smoothed latest reading (for transparency)
    score_4h_raw: float
    composite_4h: CompositeScore

    # Per-bar history (DataFrames indexed by bar time, with both raw and
    # smoothed columns: composite, composite_smooth, options, futures,
    # spot, liquidity, flow, options_smooth, …)
    history_1h: pd.DataFrame
    history_4h: pd.DataFrame

    # GEX
    gex: Optional[GEXResult]
    gex_normalized: float  # in [-1, +1]

    # RN drift (signed, normalized) — computed from caller-supplied history
    rn_drift_normalized: float

    # Stability flags from the EMA-smoothing layer
    stable_1h: bool
    stable_4h: bool

    # Downstream
    regime: RegimeAssessment
    recommendation: Recommendation
    trade_plan: TradePlan

    # Transparency
    signal_table: pd.DataFrame
    sources: dict
    raw: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS  (pure)
# ─────────────────────────────────────────────────────────────────────────────
def _last_value(df: pd.DataFrame, col: str) -> Optional[float]:
    if df is None or df.empty or col not in df.columns:
        return None
    v = df[col].iloc[-1]
    return float(v) if pd.notna(v) else None


def _safe_pct_change(df: pd.DataFrame, col: str) -> float:
    if df is None or df.empty or col not in df.columns or len(df) < 2:
        return 0.0
    a = df[col].iloc[-1]
    b = df[col].iloc[-2]
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return 0.0
    return float((a - b) / b * 100.0)


def _source_of(df: pd.DataFrame) -> str:
    if df is None or df.empty or "_source" not in df.columns:
        return "missing"
    return str(df["_source"].iloc[-1])


def _sig(name: str, value: float, weight: float = 1.0) -> dict:
    return {"name": name, "value": float(value), "weight": float(weight)}


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONS-FAMILY SIGNALS
# ─────────────────────────────────────────────────────────────────────────────
def _compute_rn_stats(chain: pd.DataFrame, spot: float) -> dict:
    """Real Breeden–Litzenberger RN density from the option chain.

    Deribit option marks are quoted in BTC, so we convert to USD by
    multiplying by spot before fitting BL. Returns ATM IV, RN mean, RN
    std, RN skew, RN mode, and P(price > spot) at expiry. If the BL fit
    fails (insufficient strikes, scipy missing, mock chain), falls back
    to a lightweight wing-skew + ATM-IV proxy.
    """
    out = {
        "rn_mean": None, "rn_std": None, "p_above": None,
        "atm_iv": None, "skew": None, "rn_full": None,
    }
    if chain is None or chain.empty or spot <= 0:
        return out

    sub = chain.dropna(subset=["iv", "strike"]).copy()
    if sub.empty:
        return out

    # ATM IV (always computable)
    atm_idx = (sub["strike"] - spot).abs().idxmin()
    atm_strike = float(sub.loc[atm_idx, "strike"])
    atm_iv_pct = float(sub.loc[sub["strike"] == atm_strike, "iv"].mean())
    out["atm_iv"] = atm_iv_pct

    # Convert Deribit BTC-marks to USD before BL
    bl_input = sub.copy()
    if "mark" in bl_input.columns and "venue" in bl_input.columns:
        # Deribit chain: mark is in BTC → multiply by spot. Binance-options:
        # mark is already in USD. We tag by venue to decide.
        is_deribit = bl_input["venue"].astype(str).str.lower() == "deribit"
        bl_input.loc[is_deribit, "mark"] = (
            pd.to_numeric(bl_input.loc[is_deribit, "mark"], errors="coerce") * float(spot)
        )

    bl = compute_rn_pdf(bl_input, spot)
    if bl is not None:
        out["rn_mean"] = float(bl["mean"])
        out["rn_std"] = float(bl["std"])
        out["p_above"] = float(bl["p_above_spot"])
        out["skew"] = float(bl["skew"])  # already a unitless RN skew
        out["rn_full"] = bl

        # OI-adjusted overlay
        oi_adj = compute_oi_adjusted_pdf(bl_input, bl, spot)
        if oi_adj is not None:
            out["rn_oi_full"] = oi_adj
            out["rn_oi_mean"] = float(oi_adj["mean"])
            out["rn_oi_p_above"] = float(oi_adj["p_above_spot"])
        return out

    # Fallback: wing-skew proxy
    p_wing = sub[(sub["type"].astype(str).str.upper() == "PUT") & (sub["strike"] <= atm_strike * 0.92)]
    c_wing = sub[(sub["type"].astype(str).str.upper() == "CALL") & (sub["strike"] >= atm_strike * 1.08)]
    if not p_wing.empty and not c_wing.empty:
        out["skew"] = float(c_wing["iv"].mean() - p_wing["iv"].mean()) / 4.0
    return out


def _options_signals(chain: pd.DataFrame, spot: float, rn_stats: dict) -> list[dict]:
    out: list[dict] = []
    if chain is None or chain.empty or spot <= 0:
        return out

    # P/C OI ratio
    calls_oi = float(chain.loc[chain["type"].astype(str).str.upper() == "CALL", "open_interest"].fillna(0).sum())
    puts_oi  = float(chain.loc[chain["type"].astype(str).str.upper() == "PUT",  "open_interest"].fillna(0).sum())
    if calls_oi > 0:
        pcr = puts_oi / calls_oi
        # <1 bullish, >1 bearish. Map (1−pcr) to score, saturating at ±0.5
        score = max(-1.0, min(1.0, (1.0 - pcr)))
        out.append(_sig("Put/Call OI ratio", score, weight=1.0))

    # Skew (from rn_stats) — already scaled to ~±1 by the wing diff
    if rn_stats.get("skew") is not None:
        out.append(_sig("RN skew (wing proxy)", normalize_skew(rn_stats["skew"]), weight=1.5))

    # RN mean vs spot — None if we don't have a full BL fit yet
    if rn_stats.get("rn_mean") is not None:
        out.append(_sig(
            "RN mean vs spot",
            normalize_rn_mean_vs_spot(rn_stats["rn_mean"], spot),
            weight=1.5,
        ))

    # RN P(above spot)
    if rn_stats.get("p_above") is not None:
        out.append(_sig("RN P(above spot)", normalize_rn_probability(rn_stats["p_above"]), weight=1.0))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FLOW SIGNALS (latest snapshot only)
# ─────────────────────────────────────────────────────────────────────────────
def _flow_signals_latest(cg_ls: pd.DataFrame, cg_liq: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if cg_ls is not None and not cg_ls.empty:
        long_pct = _last_value(cg_ls, "long_pct") or 50.0
        short_pct = _last_value(cg_ls, "short_pct") or 50.0
        score = max(-1.0, min(1.0, (long_pct - short_pct) / 100.0))
        out.append(_sig("long_short_ratio", score))
    if cg_liq is not None and not cg_liq.empty:
        long_liq = _last_value(cg_liq, "long_liq_usd") or 0.0
        short_liq = _last_value(cg_liq, "short_liq_usd") or 0.0
        out.append(_sig("liquidation_imbalance",
                        normalize_whale_activity(short_liq, long_liq)))
    out.append(_sig("exchange_flows", normalize_exchange_flows(0.0)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-BAR SCORE HISTORY  (vectorized features → per-bar composite)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_score_history(
    timeframe: str,
    kl: pd.DataFrame,
    oi_hist: pd.DataFrame,
    funding: pd.DataFrame,
    chain: pd.DataFrame,
    spot: float,
    rn_stats: dict,
    cg_ls: pd.DataFrame,
    cg_liq: pd.DataFrame,
    history_bars: int = 80,
) -> pd.DataFrame:
    """Walk the kline history and compute the composite score per bar.

    Spot- and futures-family signals are recomputed at every bar from
    pandas-rolling features (SMA stack, vol baseline, VWAP, OI Δ, funding).
    Options- and flow-family signals are taken from the latest snapshot
    (they don't have venue-side history accessible at this layer) and
    held constant across the bar history. The output is then ready for
    EMA smoothing in `analytics.smoothing`.

    Returns a DataFrame indexed by bar timestamp with columns:
        composite, options, futures, spot, liquidity, flow
    """
    if kl is None or kl.empty or len(kl) < 5:
        return pd.DataFrame()

    df = kl.copy().reset_index(drop=True)
    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    vols = df["volume"].astype(float).replace(0.0, np.nan)
    times = pd.to_datetime(df["time"], utc=True)

    # ── Trend structure (SMA stack), vectorized ───────────────────────────
    sma20 = closes.rolling(20, min_periods=20).mean()
    sma50 = closes.rolling(50, min_periods=50).mean()
    sma200 = closes.rolling(200, min_periods=200).mean()

    def _aln(a: pd.Series, b: pd.Series) -> pd.Series:
        # +0.25 if a > b, −0.25 if a ≤ b, 0 if either is NaN
        out = pd.Series(0.0, index=a.index)
        valid = a.notna() & b.notna()
        out[valid & (a > b)] = 0.25
        out[valid & (a <= b)] = -0.25
        return out

    ts = (
        _aln(closes, sma20)
        + _aln(sma20, sma50)
        + _aln(sma50, sma200)
        + _aln(closes, sma200)
    ).clip(-1.0, 1.0)
    # Where SMA200 is missing, force 0
    ts = ts.where(sma200.notna(), 0.0)

    # ── Volume confirmation: relative-to-baseline × close direction ────────
    vol_baseline = vols.shift(1).rolling(20, min_periods=20).mean()
    direction = np.sign(closes.diff()).fillna(0.0)
    rel = (vols / vol_baseline - 1.0).fillna(0.0)
    vc = (direction * rel).clip(-1.0, 1.0).fillna(0.0)

    # ── VWAP distance over a rolling window ────────────────────────────────
    win = 24 if timeframe == "1h" else 6
    typ = (highs + lows + closes) / 3.0
    pv = typ * vols
    vol_sum = vols.rolling(win, min_periods=win).sum()
    pv_sum = pv.rolling(win, min_periods=win).sum()
    vwap = (pv_sum / vol_sum).where(vol_sum > 0)
    vw = (((closes - vwap) / vwap) * 100.0).fillna(0.0).clip(-1.0, 1.0)
    # `pct_scale` from the normalizer is 1% — equivalent to clipping at ±1

    # ── Spot pct change ────────────────────────────────────────────────────
    spot_pct = closes.pct_change().fillna(0.0) * 100.0

    # ── OI series aligned to klines ────────────────────────────────────────
    if oi_hist is not None and not oi_hist.empty:
        oi_idx = pd.to_datetime(oi_hist["time"], utc=True)
        oi_series = pd.Series(
            oi_hist["oi_base"].astype(float).values, index=oi_idx
        ).sort_index()
        oi_at_bar = (
            oi_series.reindex(oi_series.index.union(times))
            .sort_index()
            .ffill()
            .reindex(times)
        )
        oi_pct = oi_at_bar.pct_change().fillna(0.0).values * 100.0
    else:
        oi_pct = np.zeros(len(df))

    # ── OI quadrant per bar ────────────────────────────────────────────────
    sp = np.array(spot_pct.values, dtype=float)
    op = np.array(oi_pct, dtype=float)
    s_dir = np.sign(sp)
    o_dir = np.sign(op)
    s_norm = np.clip(sp / 2.0, -1.0, 1.0)
    o_norm = np.clip(op / 1.0, -1.0, 1.0)
    mag = np.sqrt(np.abs(s_norm) * np.abs(o_norm))
    same_sign = (s_dir == o_dir) & (s_dir != 0)
    oi_q = np.where(same_sign, s_dir * mag, -0.5 * s_dir * mag)
    oi_q = np.clip(np.nan_to_num(oi_q), -1.0, 1.0)

    # ── Funding at bar (forward-fill) ──────────────────────────────────────
    if funding is not None and not funding.empty:
        f_idx = pd.to_datetime(funding["time"], utc=True)
        f_series = pd.Series(
            funding["funding_rate"].astype(float).values, index=f_idx
        ).sort_index()
        f_at_bar = (
            f_series.reindex(f_series.index.union(times))
            .sort_index()
            .ffill()
            .reindex(times)
            .fillna(0.0)
            .values
        )
    else:
        f_at_bar = np.zeros(len(df))

    fund_sig = np.array([normalize_funding(float(x)) for x in f_at_bar])

    # ── Latest-snapshot signals (constant across bar history) ─────────────
    options_signals = _options_signals(chain, spot, rn_stats)
    flow_signals = _flow_signals_latest(cg_ls, cg_liq)
    liquidity_signals = [_sig("orderbook_imbalance", normalize_orderbook_imbalance([], []))]

    # ── Per-bar composite (only the last `history_bars` to bound cost) ────
    start = max(0, len(df) - history_bars)
    rows = []
    for i in range(start, len(df)):
        sigs = {
            "options": options_signals,
            "futures": [
                {"name": "oi_quadrant", "value": float(oi_q[i])},
                {"name": "funding", "value": float(fund_sig[i])},
                {"name": "basis", "value": 0.0},
            ],
            "spot": [
                {"name": "trend_structure", "value": float(ts.iloc[i])},
                {"name": "volume_confirmation", "value": float(vc.iloc[i])},
                {"name": "vwap_distance", "value": float(vw.iloc[i])},
            ],
            "liquidity": liquidity_signals,
            "flow": flow_signals,
        }
        comp = compute_composite(sigs, weights=DEFAULT_FAMILY_WEIGHTS)
        rows.append(
            {
                "time": times.iloc[i],
                "composite": comp.final_score,
                "options": comp.sub_scores["options"].score,
                "futures": comp.sub_scores["futures"].score,
                "spot": comp.sub_scores["spot"].score,
                "liquidity": comp.sub_scores["liquidity"].score,
                "flow": comp.sub_scores["flow"].score,
            }
        )

    out = pd.DataFrame(rows).set_index("time")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PER-TIMEFRAME COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _features_for_timeframe(
    timeframe: str,
    kl: pd.DataFrame,
    oi_hist: pd.DataFrame,
    funding: pd.DataFrame,
    cg_oi: pd.DataFrame,
    cg_funding: pd.DataFrame,
    cg_ls: pd.DataFrame,
    cg_liq: pd.DataFrame,
    chain: pd.DataFrame,
    spot: float,
    rn_stats: dict,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """Build the {family: [signal dicts]} payload for one timeframe."""

    sources: dict[str, str] = {}
    spot_pct = _safe_pct_change(kl, "close")
    oi_pct = _safe_pct_change(oi_hist, "oi_base")
    sources["binance_klines"] = _source_of(kl)
    sources["binance_oi"] = _source_of(oi_hist)
    sources["binance_funding"] = _source_of(funding)
    sources["coinglass_oi"] = _source_of(cg_oi)
    sources["coinglass_funding"] = _source_of(cg_funding)
    sources["coinglass_ls"] = _source_of(cg_ls)
    sources["coinglass_liq"] = _source_of(cg_liq)
    sources["deribit_chain"] = _source_of(chain)

    # ── Spot family ────────────────────────────────────────────────────────
    spot_signals: list[dict] = []
    if not kl.empty:
        closes = kl["close"].tolist()
        highs = kl["high"].tolist()
        lows = kl["low"].tolist()
        vols = kl["volume"].tolist()
        spot_signals.append(_sig("trend_structure", normalize_trend_structure(closes)))
        spot_signals.append(_sig("volume_confirmation", normalize_volume_confirmation(closes, vols)))
        spot_signals.append(_sig("vwap_distance", normalize_vwap_distance(highs, lows, closes, vols)))

    # ── Futures family ─────────────────────────────────────────────────────
    futures_signals: list[dict] = []
    futures_signals.append(_sig("oi_vs_price_quadrant",
                                normalize_oi_vs_price_quadrant(spot_pct, oi_pct)))

    last_funding = _last_value(funding, "funding_rate")
    if last_funding is not None:
        futures_signals.append(_sig("funding", normalize_funding(last_funding)))

    # Basis: we don't fetch a live basis curve yet → contribute 0
    futures_signals.append(_sig("basis", normalize_basis(0.0)))

    # ── Options family ─────────────────────────────────────────────────────
    options_signals = _options_signals(chain, spot, rn_stats)

    # ── Liquidity family ───────────────────────────────────────────────────
    # No live order-book fetch wired in this iteration — contribute 0.
    liquidity_signals = [_sig("orderbook_imbalance", normalize_orderbook_imbalance([], []))]

    # ── Flow family ────────────────────────────────────────────────────────
    flow_signals: list[dict] = []
    if not cg_ls.empty:
        # Long/short ratio as a directional flow proxy.
        long_pct = _last_value(cg_ls, "long_pct") or 50.0
        short_pct = _last_value(cg_ls, "short_pct") or 50.0
        # Centre on 50/50 → score = (long − short)/100 ∈ [-1, +1]
        score = max(-1.0, min(1.0, (long_pct - short_pct) / 100.0))
        flow_signals.append(_sig("long_short_ratio", score))

    if not cg_liq.empty:
        long_liq = _last_value(cg_liq, "long_liq_usd") or 0.0
        short_liq = _last_value(cg_liq, "short_liq_usd") or 0.0
        # When more longs are getting liquidated, that's bearish. Use
        # whale-activity normaliser with buy=short_liq (shorts profiting),
        # sell=long_liq (longs losing). Saturated by total notional.
        flow_signals.append(_sig("liquidation_imbalance",
                                 normalize_whale_activity(short_liq, long_liq)))

    # No on-chain flow source yet → contribute 0.
    flow_signals.append(_sig("exchange_flows", normalize_exchange_flows(0.0)))

    return (
        {
            "options": options_signals,
            "futures": futures_signals,
            "spot": spot_signals,
            "liquidity": liquidity_signals,
            "flow": flow_signals,
        },
        sources,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    capital: float = 100_000.0,
    primary_timeframe: str = "4h",
    nearest_support: Optional[float] = None,
    nearest_resistance: Optional[float] = None,
    prev_rn_mean: Optional[float] = None,
    history_bars: int = 80,
) -> PipelineResult:
    """Fetch live data, compute everything, and return a PipelineResult.

    Args:
        prev_rn_mean: Caller-supplied previous RN mean (typically pulled
            from session_state). When provided, drives the RN-drift signal
            via `normalize_rn_drift`. Pass `None` on the very first run.
        history_bars: How many of the most recent bars to score per
            timeframe before EMA smoothing. Lower = faster, less smoothing
            depth. 80 is a good default.
    """

    # ── Fetch raw inputs ──────────────────────────────────────────────────
    spot_df = fetch_binance_spot_ticker()
    spot = float(spot_df["price"].iloc[-1]) if not spot_df.empty else 0.0

    kl_1h = fetch_binance_perp_klines("1h", 200)
    kl_4h = fetch_binance_perp_klines("4h", 100)

    oi_1h = fetch_binance_open_interest_hist("1h", 50)
    oi_4h = fetch_binance_open_interest_hist("4h", 50)

    funding = fetch_binance_funding_history(8)

    chain = fetch_deribit_option_chain(currency="BTC", max_dte=14)

    cg_oi = fetch_coinglass_aggregated_oi("1h", 5)
    cg_funding = fetch_coinglass_funding_oi_weighted("1h", 5)
    cg_ls = fetch_coinglass_long_short_ratio("1h", 5)
    cg_liq = fetch_coinglass_liquidations("1h", 5)

    # ── Options / RN stats (shared across timeframes) ─────────────────────
    rn_stats = _compute_rn_stats(chain, spot)
    atm_iv = rn_stats.get("atm_iv")

    # ── Dealer gamma exposure ─────────────────────────────────────────────
    gex_result = compute_gex(chain, spot) if (chain is not None and not chain.empty) else None
    if gex_result is not None:
        # Saturation: ±5B USD/% is a strong threshold for BTC. Adjustable via
        # `normalize_gex` keyword if needed.
        from analytics.normalization import normalize_gex
        gex_norm = normalize_gex(gex_result.gex_usd_per_pct, gex_threshold_usd=5.0e9)
    else:
        gex_norm = 0.0

    # ── Per-timeframe scoring ─────────────────────────────────────────────
    sigs_1h, sources_1h = _features_for_timeframe(
        "1h", kl_1h, oi_1h, funding, cg_oi, cg_funding, cg_ls, cg_liq, chain, spot, rn_stats
    )
    sigs_4h, sources_4h = _features_for_timeframe(
        "4h", kl_4h, oi_4h, funding, cg_oi, cg_funding, cg_ls, cg_liq, chain, spot, rn_stats
    )

    composite_1h = compute_composite(sigs_1h, weights=DEFAULT_FAMILY_WEIGHTS)
    composite_4h = compute_composite(sigs_4h, weights=DEFAULT_FAMILY_WEIGHTS)

    # ── Per-bar history + EMA smoothing ───────────────────────────────────
    hist_1h = _compute_score_history(
        "1h", kl_1h, oi_1h, funding, chain, spot, rn_stats, cg_ls, cg_liq, history_bars
    )
    hist_4h = _compute_score_history(
        "4h", kl_4h, oi_4h, funding, chain, spot, rn_stats, cg_ls, cg_liq, history_bars
    )

    def _smooth_history(hist: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, float, float, bool]:
        """Apply EMA smoothing per `analytics.smoothing` and return:
        smoothed-augmented history, smoothed final score (latest), raw
        latest score, stability flag."""
        if hist is None or hist.empty:
            return pd.DataFrame(), 0.0, 0.0, False
        family_cols = ["options", "futures", "spot", "liquidity", "flow"]
        signal_history = hist[family_cols].copy()
        final_history = hist["composite"].copy()
        sm = apply_smoothing(signal_history, final_history, timeframe=tf)
        # Augment the history with smoothed columns for charting
        out = hist.copy()
        for col in family_cols:
            out[f"{col}_smooth"] = sm.smoothed_signals[col]
        out["composite_smooth"] = sm.smoothed_final
        return (
            out,
            float(sm.latest_final),
            float(final_history.iloc[-1]),
            bool(sm.final_stability),
        )

    history_1h_aug, score_1h, score_1h_raw, stable_1h = _smooth_history(hist_1h, "1h")
    history_4h_aug, score_4h, score_4h_raw, stable_4h = _smooth_history(hist_4h, "4h")

    # Fallback to raw composite if smoothing produced nothing (insufficient history)
    if score_4h == 0.0 and abs(composite_4h.final_score) > 0:
        score_4h = composite_4h.final_score
        score_4h_raw = composite_4h.final_score
    if score_1h == 0.0 and abs(composite_1h.final_score) > 0:
        score_1h = composite_1h.final_score
        score_1h_raw = composite_1h.final_score

    # ── RN drift (signal) — only when caller passes a previous RN mean ────
    rn_drift_norm = 0.0
    if (
        prev_rn_mean is not None
        and rn_stats.get("rn_mean") is not None
        and spot > 0
    ):
        rn_drift_norm = normalize_rn_drift(
            rn_mean_curr=float(rn_stats["rn_mean"]),
            rn_mean_prev=float(prev_rn_mean),
            spot=float(spot),
            pct_scale=1.0,
        )

    # ── Regime ────────────────────────────────────────────────────────────
    regime_inputs = RegimeInputs(
        final_score=score_4h,
        options_score=composite_4h.sub_scores["options"].score,
        futures_score=composite_4h.sub_scores["futures"].score,
        spot_score=composite_4h.sub_scores["spot"].score,
        gex_normalized=gex_norm,
        funding_normalized=normalize_funding(_last_value(funding, "funding_rate") or 0.0),
        volume_confirmation=next(
            (s["value"] for s in sigs_4h["spot"] if s["name"] == "volume_confirmation"),
            0.0,
        ),
        rn_drift_normalized=rn_drift_norm,
        rn_mean=rn_stats.get("rn_mean"),
        spot=spot,
        rn_std=rn_stats.get("rn_std"),
        funding_annualized_pct=(_last_value(funding, "funding_rate") or 0.0) * 3 * 365 * 100,
        oi_pct_change=_safe_pct_change(oi_4h, "oi_base"),
        spot_pct_change=_safe_pct_change(kl_4h, "close"),
    )
    regime = classify_regime(regime_inputs)

    # ── Recommendation ────────────────────────────────────────────────────
    rec = evaluate_recommendation(
        RecommendationInputs(
            score_1h=score_1h,
            score_4h=score_4h,
            regime_4h=regime,
            stable_1h=stable_1h,
            stable_4h=stable_4h,
            atm_iv_pct=atm_iv,
            primary_timeframe=primary_timeframe,
        )
    )

    # ── Position sizing / trade plan ──────────────────────────────────────
    plan = build_trade_plan(
        SizingInputs(
            bias=rec.bias,
            setup=rec.setup,
            conviction=rec.conviction,
            capital=capital,
            spot=spot,
            atm_iv_pct=atm_iv,
            stable_1h=stable_1h,
            stable_4h=stable_4h,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )
    )

    # ── Signal-table for transparency ─────────────────────────────────────
    rows = []
    for fam in ("options", "futures", "spot", "liquidity", "flow"):
        for s in sigs_4h.get(fam, []):
            rows.append(
                {
                    "family": fam,
                    "signal": s["name"],
                    "value": round(float(s["value"]), 3),
                    "weight": float(s.get("weight", 1.0)),
                }
            )
    signal_table = pd.DataFrame(rows)

    # ── Source summary ────────────────────────────────────────────────────
    sources = {**sources_1h, **{f"4h_{k}": v for k, v in sources_4h.items()}}

    return PipelineResult(
        spot=spot,
        atm_iv_pct=atm_iv,
        rn_mean=rn_stats.get("rn_mean"),
        rn_p_above_spot=rn_stats.get("p_above"),
        rn_curve=rn_stats.get("rn_full"),
        rn_oi_curve=rn_stats.get("rn_oi_full"),
        rn_oi_mean=rn_stats.get("rn_oi_mean"),
        rn_oi_p_above_spot=rn_stats.get("rn_oi_p_above"),
        score_1h=score_1h,
        score_4h=score_4h,
        score_1h_raw=score_1h_raw,
        score_4h_raw=score_4h_raw,
        composite_4h=composite_4h,
        history_1h=history_1h_aug,
        history_4h=history_4h_aug,
        gex=gex_result,
        gex_normalized=gex_norm,
        rn_drift_normalized=rn_drift_norm,
        stable_1h=stable_1h,
        stable_4h=stable_4h,
        regime=regime,
        recommendation=rec,
        trade_plan=plan,
        signal_table=signal_table,
        sources=sources,
        raw={
            "kl_1h": kl_1h, "kl_4h": kl_4h,
            "oi_1h": oi_1h, "oi_4h": oi_4h,
            "funding": funding, "chain": chain,
            "cg_oi": cg_oi, "cg_funding": cg_funding,
            "cg_ls": cg_ls, "cg_liq": cg_liq,
        },
    )
