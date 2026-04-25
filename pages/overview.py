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
def _sidebar_inputs() -> dict:
    sb = st.sidebar
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


def _render_reasoning(rec: Recommendation) -> None:
    st.markdown(_label("Reasoning"), unsafe_allow_html=True)
    color = _bias_color(rec.bias)
    bullets = "<br>".join(f"&bull;&nbsp;&nbsp;{line}" for line in rec.explanation[:3])
    st.markdown(
        f'<div style="background:#F8F7F4;border-left:3px solid {color};'
        f'padding:14px 18px;font-size:13px;color:#44403A;line-height:1.7;">'
        f"{bullets}</div>",
        unsafe_allow_html=True,
    )


def _render_insufficient_data(reason: str) -> None:
    st.markdown(
        f'<div style="background:#FAEAEA;border-left:3px solid {RED};'
        f'padding:14px 18px;font-size:13px;color:{RED};line-height:1.7;'
        f'font-family:DM Mono,monospace;letter-spacing:0.04em;">'
        f"<strong>INSUFFICIENT DATA</strong><br>{reason}</div>",
        unsafe_allow_html=True,
    )


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

    inp = _sidebar_inputs()
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

    # Build a RegimeAssessment from sidebar so the upstream pipeline runs end-to-end.
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

    kelly = kelly_sizing(
        score_4h=float(inp["score_4h"]),
        conviction=rec.conviction,
        bias=rec.bias,
        stable=bool(inp["stable_1h"] and inp["stable_4h"]),
        odds=float(inp["odds"]),
    )

    # 1. Quadrant — primary visual
    _render_quadrant(float(inp["score_4h"]), rec.conviction, rec.bias)

    # 2. Key metrics row
    _render_metrics_row(
        spot=float(inp["spot"]),
        score_1h=float(inp["score_1h"]),
        score_4h=float(inp["score_4h"]),
        regime=rec.setup if rec.setup != "none" else inp["regime_label"],
        conviction=rec.conviction,
        bias=rec.bias,
    )

    # 3. Trade recommendation summary
    _render_trade_summary(rec)

    # 4. Kelly sizing
    _render_kelly_block(kelly, capital=float(inp["capital"]))

    # 5. Reasoning
    _render_reasoning(rec)


main()
