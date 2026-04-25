"""
btc_dashboard.pages.overview
Decision-focused overview — the headline page of the dashboard.

Single-screen layout, top-to-bottom:

    1. Quadrant chart   (PRIMARY — score × conviction, current point)
    2. Key metrics row  (price, scores, regime, conviction, bias)
    3. Trade recommendation summary
    4. Kelly-based position sizing
    5. Reasoning  (3 bullets)

All upstream computations are imported from `analytics/`. The page never
re-derives a score, regime, or recommendation — it only routes those
outputs into the visual layer and applies the Kelly sizing layer
defined in this module (Kelly is page-local because no upstream module
owns it yet).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analytics.recommendation import (        # noqa: E402
    OptionsSuggestion,
    Recommendation,
    RecommendationInputs,
    evaluate_recommendation,
)
from analytics.position_sizing import (       # noqa: E402
    SizingInputs,
    TradePlan,
    build_trade_plan,
)
from analytics.regime import RegimeAssessment  # noqa: E402

# Live data connectors
from data.spot import fetch_binance_spot_ticker  # noqa: E402
from data.futures import (                       # noqa: E402
    fetch_binance_perp_klines,
    fetch_binance_funding_history,
    fetch_binance_open_interest_hist,
    fetch_deribit_open_interest,
)
from data.coinglass import (                     # noqa: E402
    fetch_coinglass_aggregated_oi,
    fetch_coinglass_funding_oi_weighted,
    fetch_coinglass_long_short_ratio,
    fetch_coinglass_liquidations,
    coinglass_status,
)

# End-to-end pipeline
from analytics.pipeline import run_pipeline, PipelineResult  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
GOLD  = "#C9A55A"
TEAL  = "#1A7A6B"
RED   = "#A83232"
AMBER = "#B8832A"
STONE = "#9C968A"
INK   = "#1C1A17"

GREEN_ZONE_STRONG = "rgba(26,122,107,0.12)"
GREEN_ZONE_WEAK   = "rgba(26,122,107,0.04)"
RED_ZONE_STRONG   = "rgba(168,50,50,0.12)"
RED_ZONE_WEAK     = "rgba(168,50,50,0.04)"


# Conviction tier → numeric Y position (0..100) for the quadrant chart
CONVICTION_Y = {"none": 5, "low": 25, "medium": 55, "high": 85}

# Caps
MAX_PORTFOLIO_PCT      = 10.0    # absolute hard cap on position size
MAX_RISK_PER_TRADE_PCT = 1.0     # absolute hard cap on risk/trade
LOW_CONVICTION_CAP_PCT = 2.5     # extra cap when conviction = "low"


# ─────────────────────────────────────────────────────────────────────────────
# KELLY  (page-local — does not exist upstream)
# ─────────────────────────────────────────────────────────────────────────────
def kelly_sizing(
    score_4h: float,
    conviction: str,
    bias: str,
    stable: bool,
    odds: float = 2.0,
) -> dict:
    """Kelly-based position sizing for the overview page.

    Kelly:                f* = edge / odds
    win_probability:      0.50 + score_4h / 200            (50% + score/2 pp)
    implied_probability:  1 / (odds + 1)                   (break-even from odds)
    edge:                 win_probability − implied_probability
    Fractional Kelly:     f_frac = f* × 0.25
    Final size:           f_frac, then apply caps in order:
                            • 0% if bias is no_trade
                            • clamp ≥ 0 (no shorts via this sizer; sign comes from bias)
                            • 2.5% cap when conviction = "low"
                            • 1% max risk-per-trade cap
                            • 10% absolute portfolio cap
                            • ×0.5 haircut when signal unstable
    """
    caps: list[str] = []

    if bias == "no_trade":
        return dict(
            edge=0.0,
            win_prob=0.5,
            implied_prob=1.0 / (float(odds) + 1.0) if odds and odds > 0 else 0.5,
            odds=float(odds) if odds and odds > 0 else 1.0,
            raw_kelly_pct=0.0,
            fractional_kelly_pct=0.0,
            final_position_pct=0.0,
            caps_applied=["no trade → size = 0%"],
        )

    o = float(odds) if odds and odds > 0 else 1.0
    win_prob = max(0.0, min(1.0, 0.50 + float(score_4h) / 200.0))
    implied_prob = 1.0 / (o + 1.0)
    edge = win_prob - implied_prob

    raw_kelly = edge / o
    frac_kelly = raw_kelly * 0.25
    final = frac_kelly

    # Caps & haircuts (order: clamp negative → conviction → risk → portfolio → stability)
    if final < 0.0:
        final = 0.0
        caps.append("Negative edge — sized to 0%")

    if conviction == "low" and final * 100.0 > LOW_CONVICTION_CAP_PCT:
        final = LOW_CONVICTION_CAP_PCT / 100.0
        caps.append(f"Low conviction → cap at {LOW_CONVICTION_CAP_PCT:.1f}%")

    if final * 100.0 > MAX_RISK_PER_TRADE_PCT:
        final = MAX_RISK_PER_TRADE_PCT / 100.0
        caps.append(f"Max risk per trade → cap at {MAX_RISK_PER_TRADE_PCT:.1f}%")

    if final * 100.0 > MAX_PORTFOLIO_PCT:
        final = MAX_PORTFOLIO_PCT / 100.0
        caps.append(f"Max portfolio → cap at {MAX_PORTFOLIO_PCT:.1f}%")

    if not stable:
        final *= 0.50
        caps.append("Unstable signal → ×0.5")

    return dict(
        edge=edge,
        win_prob=win_prob,
        implied_prob=implied_prob,
        odds=o,
        raw_kelly_pct=raw_kelly * 100.0,
        fractional_kelly_pct=frac_kelly * 100.0,
        final_position_pct=final * 100.0,
        caps_applied=caps,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUADRANT CHART
# ─────────────────────────────────────────────────────────────────────────────
def quadrant_chart(score_4h: float, conviction_tier: str, bias: str) -> go.Figure:
    conv_y = CONVICTION_Y.get(conviction_tier, 5)
    fig = go.Figure()

    # Background zones
    fig.add_shape(type="rect", x0=0,    x1=100, y0=50, y1=100,
                  fillcolor=GREEN_ZONE_STRONG, line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0,    x1=100, y0=0,  y1=50,
                  fillcolor=GREEN_ZONE_WEAK,   line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=-100, x1=0,   y0=50, y1=100,
                  fillcolor=RED_ZONE_STRONG,   line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=-100, x1=0,   y0=0,  y1=50,
                  fillcolor=RED_ZONE_WEAK,     line=dict(width=0), layer="below")

    # Quadrant labels
    fig.add_annotation(x=50,  y=94, text="STRONG LONG",
                       showarrow=False,
                       font=dict(size=14, color=TEAL,  family="DM Sans, sans-serif"))
    fig.add_annotation(x=50,  y=6,  text="Weak Long · Watch",
                       showarrow=False,
                       font=dict(size=11, color=STONE, family="DM Sans, sans-serif"))
    fig.add_annotation(x=-50, y=94, text="STRONG SHORT",
                       showarrow=False,
                       font=dict(size=14, color=RED,   family="DM Sans, sans-serif"))
    fig.add_annotation(x=-50, y=6,  text="Weak Short · Watch",
                       showarrow=False,
                       font=dict(size=11, color=STONE, family="DM Sans, sans-serif"))

    # Crosshairs
    fig.add_vline(x=0,  line=dict(color=STONE, width=1, dash="dot"))
    fig.add_hline(y=50, line=dict(color=STONE, width=1, dash="dot"))

    # Marker
    marker_color = TEAL if bias == "long" else RED if bias == "short" else STONE
    marker_size = 16 + conv_y * 0.45      # 16 → ~54 across the conviction range
    fig.add_trace(
        go.Scatter(
            x=[float(score_4h)],
            y=[conv_y],
            mode="markers",
            marker=dict(
                size=marker_size,
                color=marker_color,
                line=dict(color="#FFFFFF", width=2),
                opacity=0.95,
            ),
            hovertemplate=(
                f"4h Score: {score_4h:+.1f}<br>"
                f"Conviction: {conviction_tier.upper()} ({conv_y})"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=460,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=60, r=20, t=20, b=60),
        font=dict(family="DM Mono, monospace", color="#44403A"),
        xaxis=dict(
            title=dict(text="Score   (Bearish ←        → Bullish)",
                       font=dict(size=11, color=STONE)),
            range=[-100, 100], gridcolor="#EDEBE6",
            zeroline=False, tickvals=[-100, -50, 0, 50, 100],
        ),
        yaxis=dict(
            title=dict(text="Conviction   (Low ↓        ↑ High)",
                       font=dict(size=11, color=STONE)),
            range=[0, 100], gridcolor="#EDEBE6",
            zeroline=False, tickvals=[0, 25, 50, 75, 100],
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _label(text: str) -> str:
    return (
        f'<div style="font-family:DM Mono,monospace;font-size:9px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{GOLD};'
        f'margin:18px 0 6px 0;">{text}</div>'
    )


def _bias_color(bias: str) -> str:
    return TEAL if bias == "long" else RED if bias == "short" else STONE


def _fmt_money(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Running live pipeline…")
def _cached_pipeline(capital: float, primary_tf: str,
                     nearest_support: float, nearest_resistance: float) -> dict:
    """Cached wrapper. Returns a dict so the result is hashable for cache."""
    res: PipelineResult = run_pipeline(
        capital=capital,
        primary_timeframe=primary_tf,
        nearest_support=nearest_support if nearest_support > 0 else None,
        nearest_resistance=nearest_resistance if nearest_resistance > 0 else None,
    )
    # The PipelineResult is a frozen dataclass — we expose it directly.
    return {"result": res}


def _sidebar_inputs() -> dict:
    sb = st.sidebar

    sb.markdown("### Mode")
    live_mode = sb.toggle(
        "Live signal pipeline",
        value=True,
        help="Compute scores, regime and trade plan from live Binance/Deribit/Coinglass data. "
             "Turn off to drive everything from the sliders below.",
    )

    sb.markdown("### Scores")
    score_1h = sb.slider("1h composite", -100, 100, 68, 1)
    score_4h = sb.slider("4h composite", -100, 100, 72, 1)
    stable_1h = sb.checkbox("1h stable", value=True)
    stable_4h = sb.checkbox("4h stable", value=True)

    sb.markdown("### Regime (4h)")
    regime_label = sb.selectbox(
        "Regime label",
        ["trend", "squeeze", "mean_reversion", "neutral"],
        index=0,
    )
    regime_conf = sb.slider("Regime confidence", 0.0, 1.0, 0.65, 0.05)

    sb.markdown("### Market state")
    spot   = sb.number_input("Spot ($)",   min_value=1000.0, value=74_200.0, step=100.0)
    atm_iv = sb.number_input("ATM IV (%)", min_value=5.0,    value=55.0,     step=1.0)

    sb.markdown("### Liquidity")
    nearest_support    = sb.number_input("Nearest support ($)",    min_value=0.0, value=72_500.0, step=100.0)
    nearest_resistance = sb.number_input("Nearest resistance ($)", min_value=0.0, value=76_000.0, step=100.0)

    sb.markdown("### Account & horizon")
    capital    = sb.number_input("Capital ($)", min_value=100.0, value=100_000.0, step=1000.0)
    primary_tf = sb.radio("Primary timeframe", ["4h", "1h"], index=0, horizontal=True)
    odds       = sb.number_input("Kelly odds (R:R)", min_value=0.5, value=2.0, step=0.25)

    sb.markdown("---")
    insufficient = sb.checkbox("Force 'Insufficient Data' (failsafe demo)", value=False)

    return dict(
        live_mode=live_mode,
        score_1h=score_1h, score_4h=score_4h,
        stable_1h=stable_1h, stable_4h=stable_4h,
        regime_label=regime_label, regime_conf=regime_conf,
        spot=spot, atm_iv=atm_iv,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        capital=capital, primary_tf=primary_tf,
        odds=odds, insufficient=insufficient,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION RENDERS
# ─────────────────────────────────────────────────────────────────────────────
def _render_quadrant(score_4h: float, conviction: str, bias: str) -> None:
    fig = quadrant_chart(score_4h, conviction, bias)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_metrics_row(
    spot: float, score_1h: float, score_4h: float,
    regime: str, conviction: str, bias: str,
) -> None:
    cols = st.columns(6, gap="small")
    cols[0].metric("BTC Price", _fmt_money(spot))

    s1_color = TEAL if score_1h > 0 else RED if score_1h < 0 else STONE
    cols[1].metric("1h Score", f"{score_1h:+.0f}",
                   "bullish" if score_1h > 0 else "bearish" if score_1h < 0 else "flat")

    s4_color = TEAL if score_4h > 0 else RED if score_4h < 0 else STONE
    cols[2].metric("4h Score", f"{score_4h:+.0f}",
                   "bullish" if score_4h > 0 else "bearish" if score_4h < 0 else "flat")

    cols[3].metric("Regime",     regime.replace("_", " ").title())
    cols[4].metric("Conviction", conviction.title())

    bias_label = bias.replace("_", " ").upper()
    cols[5].metric("Bias", bias_label)


def _render_trade_summary(rec: Recommendation) -> None:
    st.markdown(_label("Trade Recommendation"), unsafe_allow_html=True)
    opt = rec.options_suggestion

    # Instrument family inferred from the structure label
    instrument_family = "Options" if (opt and opt.structure not in (None, "none")) else (
        "Futures / Spot" if rec.bias != "no_trade" else "—"
    )

    # Suggested trade text
    if opt is None or opt.structure == "none":
        suggested = "Stand aside" if rec.bias == "no_trade" else "—"
    else:
        long_lo, long_hi = opt.long_leg_delta
        exp_lo, exp_hi = opt.expiry_days
        if opt.short_leg_delta is None:
            suggested = (
                f"Buy {opt.structure} "
                f"{long_lo:.0f}–{long_hi:.0f}Δ · "
                f"{exp_lo}–{exp_hi}D expiry"
            )
        else:
            sh_lo, sh_hi = opt.short_leg_delta
            suggested = (
                f"Long {long_lo:.0f}–{long_hi:.0f}Δ · "
                f"short {sh_lo:.0f}–{sh_hi:.0f}Δ · "
                f"{exp_lo}–{exp_hi}D expiry"
            )

    cols = st.columns([1, 1, 1, 2])
    cols[0].metric("Bias",        rec.bias.replace("_", " ").upper())
    cols[1].metric("Setup",       rec.setup.replace("_", " ").title())
    cols[2].metric("Instrument",  instrument_family)
    cols[3].metric("Suggested trade", suggested)


def _render_kelly_block(kelly: dict, capital: float) -> None:
    st.markdown(_label("Position Sizing (Kelly)"), unsafe_allow_html=True)

    final_dollars = capital * kelly["final_position_pct"] / 100.0

    cols = st.columns(3)
    cols[0].metric(
        "Raw Kelly",
        f"{kelly['raw_kelly_pct']:.2f}%",
        f"edge {kelly['edge']*100:+.1f}pp · odds {kelly['odds']:.2f} "
        f"· P(win) {kelly['win_prob']*100:.0f}% vs implied {kelly['implied_prob']*100:.0f}%",
    )
    cols[1].metric(
        "Fractional Kelly",
        f"{kelly['fractional_kelly_pct']:.2f}%",
        "× 0.25 of raw",
    )
    cols[2].metric(
        "Final Position",
        f"{kelly['final_position_pct']:.2f}%",
        f"{_fmt_money(final_dollars)} of capital",
    )

    # Reason for adjustment
    if kelly["caps_applied"]:
        reasons = " · ".join(kelly["caps_applied"])
        st.markdown(
            f'<div style="background:#F8F7F4;border-left:2px solid {AMBER};'
            f'padding:8px 12px;font-size:12px;color:#44403A;line-height:1.6;'
            f'margin-top:6px;font-family:DM Mono,monospace;">'
            f'<strong>Reason for adjustment:</strong> {reasons}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("No caps or haircuts applied — fractional Kelly is the final size.")


def _render_options_state(pipe) -> None:
    """Live GEX + RN summary, sourced from the pipeline result."""
    st.markdown(_label("Options State"), unsafe_allow_html=True)
    cols = st.columns(4)

    # ATM IV
    iv = pipe.atm_iv_pct
    cols[0].metric("ATM IV", f"{iv:.1f}%" if iv is not None else "—")

    # RN mean vs spot
    rn = pipe.rn_mean
    if rn is not None and pipe.spot:
        gap_pct = (rn - pipe.spot) / pipe.spot * 100.0
        cols[1].metric("RN mean", _fmt_money(rn), f"{gap_pct:+.2f}% vs spot")
    else:
        cols[1].metric("RN mean", "—")

    # P(above spot)
    p_above = pipe.rn_p_above_spot
    if p_above is not None:
        cols[2].metric("P(above spot)", f"{p_above*100:.1f}%")
    else:
        cols[2].metric("P(above spot)", "—")

    # GEX
    gex = pipe.gex
    if gex is not None:
        gex_b = gex.gex_usd_per_pct / 1e9
        regime_label = (
            "long γ · stable" if gex.gex_usd_per_pct > 0 else
            "short γ · fragile"
        )
        cols[3].metric(
            "Dealer GEX",
            f"{gex_b:+.2f} B$/1%",
            f"{regime_label} · {gex.n_options} contracts",
        )
        if gex.flip_strike:
            st.caption(
                f"GEX flip strike ≈ ${gex.flip_strike:,.0f} · "
                f"normalized {pipe.gex_normalized:+.2f} (saturates at ±5B$/1%)"
            )
        else:
            st.caption(f"Normalized GEX {pipe.gex_normalized:+.2f} (saturates at ±5B$/1%)")
    else:
        cols[3].metric("Dealer GEX", "—")


def _render_levels(plan: TradePlan) -> None:
    """Stop / TP1 / TP2 / Runner row. Hidden when there is no directional bias."""
    if plan.bias == "no_trade":
        return
    st.markdown(_label("Levels"), unsafe_allow_html=True)
    cols = st.columns(5)
    cols[0].metric("Entry", _fmt_money(plan.entry))
    cols[1].metric(
        "Stop",
        _fmt_money(plan.stop),
        f"-${abs(plan.entry - plan.stop):,.0f}",
        delta_color="inverse",
    )
    cols[2].metric(
        "TP1",
        _fmt_money(plan.tp1),
        f"+${abs(plan.tp1 - plan.entry):,.0f}",
    )
    cols[3].metric(
        "TP2",
        _fmt_money(plan.tp2),
        f"+${abs(plan.tp2 - plan.entry):,.0f} · R/R {plan.risk_reward_tp2:.2f}x",
    )
    cols[4].metric(
        "Runner",
        _fmt_money(plan.runner),
        f"+${abs(plan.runner - plan.entry):,.0f}",
    )


def _render_reasoning(rec: Recommendation, regime: Optional[RegimeAssessment] = None) -> None:
    st.markdown(_label("Reasoning"), unsafe_allow_html=True)
    color = _bias_color(rec.bias)
    bullets = "<br>".join(f"&bull;&nbsp;&nbsp;{line}" for line in rec.explanation[:3])
    st.markdown(
        f'<div style="background:#F8F7F4;border-left:3px solid {color};'
        f'padding:14px 18px;font-size:13px;color:#44403A;line-height:1.7;">'
        f"{bullets}</div>",
        unsafe_allow_html=True,
    )

    # Regime evidence — collapsible so the headline view stays clean
    if regime is not None:
        with st.expander(
            f"Regime detail · {regime.regime.replace('_',' ').title()} "
            f"· confidence {regime.confidence*100:.0f}%"
        ):
            # Per-regime confidence scores
            score_rows = [
                {"Regime": k.replace("_", " ").title(), "Confidence": f"{v*100:.0f}%"}
                for k, v in regime.scores.items()
            ]
            if score_rows:
                st.dataframe(
                    pd.DataFrame(score_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            # Evidence bullets per candidate regime
            for label, lines in regime.rationale.items():
                st.markdown(f"**{label.replace('_',' ').title()}**")
                for line in lines:
                    st.markdown(f"• {line}")


def _render_insufficient_data(reason: str) -> None:
    st.markdown(
        f'<div style="background:#FAEAEA;border-left:3px solid {RED};'
        f'padding:14px 18px;font-size:13px;color:{RED};line-height:1.7;'
        f'font-family:DM Mono,monospace;letter-spacing:0.04em;">'
        f"<strong>INSUFFICIENT DATA</strong><br>{reason}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LIVE MARKET SNAPSHOT  (Binance + Deribit + Coinglass)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _live_snapshot() -> dict:
    """Pull a live snapshot from every connected venue.
    Cached for 30s so quick reruns don't hammer the APIs.
    Each field is None if the venue / endpoint is unreachable."""
    out: dict = {
        "binance_spot": None,
        "binance_perp": None,
        "binance_funding": None,
        "binance_oi": None,
        "deribit_oi": None,
        "coinglass_oi": None,
        "coinglass_funding": None,
        "coinglass_ls": None,
        "coinglass_liq": None,
        "coinglass_status": coinglass_status(),
    }
    try:
        df = fetch_binance_spot_ticker()
        if not df.empty:
            out["binance_spot"] = {
                "price": float(df["price"].iloc[-1]),
                "source": str(df["_source"].iloc[-1]),
            }
    except Exception as exc:
        log_msg = f"binance_spot ticker error: {exc}"
        out["binance_spot_error"] = log_msg

    try:
        kl = fetch_binance_perp_klines(interval="1h", limit=2)
        if not kl.empty and len(kl) >= 1:
            out["binance_perp"] = {
                "price": float(kl["close"].iloc[-1]),
                "source": str(kl["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        f = fetch_binance_funding_history(limit=1)
        if not f.empty:
            out["binance_funding"] = {
                "rate": float(f["funding_rate"].iloc[-1]),
                "source": str(f["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        oi = fetch_binance_open_interest_hist(period="1h", limit=2)
        if not oi.empty:
            out["binance_oi"] = {
                "oi_base": float(oi["oi_base"].iloc[-1]),
                "oi_usd": float(oi["oi_usd"].iloc[-1]),
                "source": str(oi["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        d_oi = fetch_deribit_open_interest()
        if not d_oi.empty:
            out["deribit_oi"] = {
                "oi_base": float(d_oi["oi_base"].iloc[-1]),
                "oi_usd": float(d_oi["oi_usd"].iloc[-1]),
                "source": str(d_oi["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        cg_oi = fetch_coinglass_aggregated_oi(interval="1h", limit=2)
        if not cg_oi.empty:
            out["coinglass_oi"] = {
                "oi_usd": float(cg_oi["oi_usd"].iloc[-1]),
                "change_pct": float(cg_oi["oi_change_pct"].iloc[-1]),
                "source": str(cg_oi["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        cg_f = fetch_coinglass_funding_oi_weighted(interval="1h", limit=1)
        if not cg_f.empty:
            out["coinglass_funding"] = {
                "rate": float(cg_f["funding_rate"].iloc[-1]),
                "source": str(cg_f["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        cg_ls = fetch_coinglass_long_short_ratio(interval="1h", limit=1)
        if not cg_ls.empty:
            out["coinglass_ls"] = {
                "long_pct": float(cg_ls["long_pct"].iloc[-1]),
                "short_pct": float(cg_ls["short_pct"].iloc[-1]),
                "ratio": float(cg_ls["ratio"].iloc[-1]),
                "source": str(cg_ls["_source"].iloc[-1]),
            }
    except Exception:
        pass

    try:
        cg_liq = fetch_coinglass_liquidations(interval="1h", limit=1)
        if not cg_liq.empty:
            out["coinglass_liq"] = {
                "long_liq_usd": float(cg_liq["long_liq_usd"].iloc[-1]),
                "short_liq_usd": float(cg_liq["short_liq_usd"].iloc[-1]),
                "source": str(cg_liq["_source"].iloc[-1]),
            }
    except Exception:
        pass

    return out


def _badge(source: Optional[str]) -> str:
    if source == "live":
        return f'<span style="color:{TEAL};font-size:9px;font-weight:600;letter-spacing:.08em;">● LIVE</span>'
    if source == "mock":
        return f'<span style="color:{AMBER};font-size:9px;font-weight:600;letter-spacing:.08em;">● MOCK</span>'
    return f'<span style="color:{STONE};font-size:9px;font-weight:600;letter-spacing:.08em;">● —</span>'


def _render_live_snapshot(snap: dict) -> None:
    st.markdown(_label("Live Market Snapshot"), unsafe_allow_html=True)

    # Row 1 — spot price + funding views + positioning (4 columns; perp tile dropped as redundant)
    c1, c2, c3, c4 = st.columns(4)

    bs = snap.get("binance_spot")
    c1.metric(
        "Binance spot",
        _fmt_money(bs["price"]) if bs else "—",
    )
    c1.markdown(_badge(bs["source"] if bs else None), unsafe_allow_html=True)

    bf = snap.get("binance_funding")
    if bf:
        ann = bf["rate"] * 3 * 365 * 100
        c2.metric(
            "Binance funding",
            f"{bf['rate']*100:+.4f}%/8h",
            f"{ann:+.1f}% ann",
        )
    else:
        c2.metric("Binance funding", "—")
    c2.markdown(_badge(bf["source"] if bf else None), unsafe_allow_html=True)

    cgf = snap.get("coinglass_funding")
    if cgf:
        ann = cgf["rate"] * 3 * 365 * 100
        c3.metric(
            "Aggregated funding",
            f"{cgf['rate']*100:+.4f}%/8h",
            f"{ann:+.1f}% ann · OI-wtd",
        )
    else:
        c3.metric("Aggregated funding", "—")
    c3.markdown(_badge(cgf["source"] if cgf else None), unsafe_allow_html=True)

    cgls = snap.get("coinglass_ls")
    if cgls:
        c4.metric(
            "Long / Short",
            f"{cgls['ratio']:.2f}x",
            f"L {cgls['long_pct']:.0f}% · S {cgls['short_pct']:.0f}%",
        )
    else:
        c4.metric("Long / Short", "—")
    c4.markdown(_badge(cgls["source"] if cgls else None), unsafe_allow_html=True)

    # Row 2 — open interest (Binance / Deribit / Coinglass) + liquidations
    d1, d2, d3, d4 = st.columns(4)

    boi = snap.get("binance_oi")
    if boi:
        d1.metric(
            "Binance OI",
            f"{boi['oi_base']:,.0f} BTC",
            _fmt_money(boi["oi_usd"]),
        )
    else:
        d1.metric("Binance OI", "—")
    d1.markdown(_badge(boi["source"] if boi else None), unsafe_allow_html=True)

    doi = snap.get("deribit_oi")
    if doi:
        d2.metric(
            "Deribit perp OI",
            f"{doi['oi_base']:,.0f} BTC",
            _fmt_money(doi["oi_usd"]),
        )
    else:
        d2.metric("Deribit perp OI", "—")
    d2.markdown(_badge(doi["source"] if doi else None), unsafe_allow_html=True)

    cgoi = snap.get("coinglass_oi")
    if cgoi:
        d3.metric(
            "Aggregated OI",
            _fmt_money(cgoi["oi_usd"]),
            f"{cgoi['change_pct']:+.2f}%",
        )
    else:
        d3.metric("Aggregated OI", "—")
    d3.markdown(_badge(cgoi["source"] if cgoi else None), unsafe_allow_html=True)

    cgliq = snap.get("coinglass_liq")
    if cgliq:
        d4.metric(
            "Liquidations 1h",
            _fmt_money(cgliq["long_liq_usd"] + cgliq["short_liq_usd"]),
            f"L {cgliq['long_liq_usd']/1e6:.1f}M · S {cgliq['short_liq_usd']/1e6:.1f}M",
        )
    else:
        d4.metric("Liquidations 1h", "—")
    d4.markdown(_badge(cgliq["source"] if cgliq else None), unsafe_allow_html=True)

    # Status caption
    cs = snap.get("coinglass_status", {})
    if not cs.get("has_api_key"):
        st.caption(
            "ℹ️ Coinglass API key not set — aggregated metrics use mock data. "
            "Set `COINGLASS_API_KEY` in DigitalOcean → App Settings → Environment Variables to enable live aggregations."
        )
    elif cs.get("error"):
        st.caption(f"⚠️ Coinglass auth set but call failed: {cs['error']}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA-MISSING DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _is_data_missing(inp: dict) -> tuple[bool, str]:
    if inp.get("insufficient"):
        return True, "Failsafe forced via sidebar."
    for key in ("score_1h", "score_4h", "spot"):
        v = inp.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return True, f"`{key}` is missing or non-finite."
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Overview · BTC Decision",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f'<h1 style="font-family:DM Sans,sans-serif;font-weight:300;'
        f'color:{INK};margin:0 0 4px 0;">Overview</h1>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{STONE};'
        f'margin-bottom:8px;">Decision interface · score × conviction quadrant · '
        f'Kelly-sized recommendation</div>',
        unsafe_allow_html=True,
    )

    # Pull a live snapshot first — used both for the snapshot panel and as
    # default values where the sidebar would otherwise show stale numbers.
    snap = _live_snapshot()
    _render_live_snapshot(snap)

    inp = _sidebar_inputs()

    # If the sidebar spot is the default 74_200 and live spot is available,
    # silently substitute live spot so the rest of the page reflects market.
    bs = snap.get("binance_spot")
    if bs and abs(float(inp["spot"]) - 74_200.0) < 1e-6:
        inp["spot"] = float(bs["price"])

    missing, why = _is_data_missing(inp)

    if missing:
        _render_insufficient_data(why)
        # Render the quadrant at the origin so the page still has visual structure.
        _render_quadrant(0.0, "none", "no_trade")
        _render_metrics_row(
            spot=inp.get("spot", float("nan")),
            score_1h=0.0, score_4h=0.0,
            regime="neutral", conviction="none", bias="no_trade",
        )
        st.markdown(_label("Trade Recommendation"), unsafe_allow_html=True)
        st.caption("Trade recommendation disabled — insufficient data.")
        st.markdown(_label("Position Sizing (Kelly)"), unsafe_allow_html=True)
        st.caption("Position size = 0% (failsafe).")
        return

    # ── Live mode: pipeline drives everything ─────────────────────────────
    pipeline_result: Optional[PipelineResult] = None
    if inp.get("live_mode"):
        try:
            pipe = _cached_pipeline(
                capital=float(inp["capital"]),
                primary_tf=str(inp["primary_tf"]),
                nearest_support=float(inp["nearest_support"]),
                nearest_resistance=float(inp["nearest_resistance"]),
            )
            pipeline_result = pipe["result"]
        except Exception as exc:
            st.warning(f"Live pipeline failed ({exc}). Falling back to slider inputs.")

    if pipeline_result is not None:
        rec = pipeline_result.recommendation
        plan = pipeline_result.trade_plan
        regime = pipeline_result.regime
        score_1h_used = pipeline_result.score_1h
        score_4h_used = pipeline_result.score_4h
        spot_used = pipeline_result.spot if pipeline_result.spot > 0 else float(inp["spot"])
        atm_iv_used = pipeline_result.atm_iv_pct
        stable_1h_used = pipeline_result.stable_1h
        stable_4h_used = pipeline_result.stable_4h
    else:
        # Manual / slider mode (failsafe or live mode disabled)
        regime = RegimeAssessment(
            regime=inp["regime_label"],
            confidence=float(inp["regime_conf"]),
            direction=None,
            scores={inp["regime_label"]: float(inp["regime_conf"])},
            rationale={inp["regime_label"]: ["Provided manually via sidebar"]},
        )
        rec = evaluate_recommendation(
            RecommendationInputs(
                score_1h=float(inp["score_1h"]),
                score_4h=float(inp["score_4h"]),
                regime_4h=regime,
                stable_1h=bool(inp["stable_1h"]),
                stable_4h=bool(inp["stable_4h"]),
                atm_iv_pct=float(inp["atm_iv"]),
                primary_timeframe=str(inp["primary_tf"]),
            )
        )
        plan = build_trade_plan(
            SizingInputs(
                bias=rec.bias,
                setup=rec.setup,
                conviction=rec.conviction,
                capital=float(inp["capital"]),
                spot=float(inp["spot"]),
                atm_iv_pct=float(inp["atm_iv"]),
                stable_1h=bool(inp["stable_1h"]),
                stable_4h=bool(inp["stable_4h"]),
                nearest_support=(float(inp["nearest_support"]) if inp["nearest_support"] > 0 else None),
                nearest_resistance=(float(inp["nearest_resistance"]) if inp["nearest_resistance"] > 0 else None),
            )
        )
        score_1h_used = float(inp["score_1h"])
        score_4h_used = float(inp["score_4h"])
        spot_used = float(inp["spot"])
        atm_iv_used = float(inp["atm_iv"])
        stable_1h_used = bool(inp["stable_1h"])
        stable_4h_used = bool(inp["stable_4h"])

    kelly = kelly_sizing(
        score_4h=score_4h_used,
        conviction=rec.conviction,
        bias=rec.bias,
        stable=bool(stable_1h_used and stable_4h_used),
        odds=float(inp["odds"]),
    )

    # 1. Quadrant — primary visual
    _render_quadrant(score_4h_used, rec.conviction, rec.bias)

    # 2. Key metrics row
    _render_metrics_row(
        spot=spot_used,
        score_1h=score_1h_used,
        score_4h=score_4h_used,
        regime=rec.setup if rec.setup != "none" else regime.regime,
        conviction=rec.conviction,
        bias=rec.bias,
    )

    # 3. Trade recommendation summary
    _render_trade_summary(rec)

    # 4. Kelly sizing
    _render_kelly_block(kelly, capital=float(inp["capital"]))

    # 5. Options-side state — only when the live pipeline supplied them
    if pipeline_result is not None:
        _render_options_state(pipeline_result)

    # 6. Levels (stop / TP ladder)
    _render_levels(plan)

    # 6. Reasoning + regime evidence
    _render_reasoning(rec, regime=regime)

    # 7. Live signal table (transparency) — only shown in live mode
    if pipeline_result is not None and not pipeline_result.signal_table.empty:
        st.markdown(_label("Live Signal Table"), unsafe_allow_html=True)
        st.caption(
            "Per-signal contributions feeding the 4h composite. "
            "Each `value` is normalised to [-1, +1]. "
            "Positive = bullish, negative = bearish, 0 = neutral / no data."
        )
        st.dataframe(
            pipeline_result.signal_table,
            use_container_width=True,
            hide_index=True,
        )


main()
