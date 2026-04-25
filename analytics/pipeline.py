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
from analytics.rn_pdf import compute_rn_pdf

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

    # Composite scores per timeframe (in [-100, +100])
    score_1h: float
    score_4h: float
    composite_4h: CompositeScore

    # GEX
    gex: Optional[GEXResult]
    gex_normalized: float  # in [-1, +1]

    # Stability flags (placeholder — true smoothing/EMA across snapshots
    # will be added once the pipeline is invoked from a stateful caller).
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
) -> PipelineResult:
    """Fetch live data, compute everything, and return a PipelineResult."""

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

    score_1h = composite_1h.final_score
    score_4h = composite_4h.final_score

    # ── Stability flags (placeholder until a stateful caller adds smoothing) ──
    # For now: stable iff the score is far enough from zero to be meaningful.
    stable_1h = abs(score_1h) >= 5.0
    stable_4h = abs(score_4h) >= 5.0

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
        rn_drift_normalized=0.0,                  # RN drift requires snapshot history
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
        score_1h=score_1h,
        score_4h=score_4h,
        composite_4h=composite_4h,
        gex=gex_result,
        gex_normalized=gex_norm,
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
