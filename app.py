"""
btc_dashboard · app.py
Landing page — Deribit-driven options-distribution overview.

Charts (top → bottom):
  1. RN PDF (pure BL) overlaid with OI-adjusted RN PDF, with annotations
     for both means and both P(above spot) values.
  2. Dealer GEX per strike (Deribit chain) with cumulative line + flip strike.
  3. Time series — RN-mean − spot gap in absolute USD, both base RN and
     OI-adjusted RN.
  4. Time series — P(above spot), both base RN and OI-adjusted RN.

Time-series (charts 3 & 4) build their history from `st.session_state`
across page refreshes — no DB. Two snapshots minimum to draw a line.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.pipeline import PipelineResult, run_pipeline
from charts.theme import (
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, page_title, section_label,
)
from config import settings
from utils.logger import get_logger


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE  (cached 30s)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Loading Deribit chain…")
def _pipe() -> dict:
    return {"result": run_pipeline()}


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY BUFFER  (per-session)
# ─────────────────────────────────────────────────────────────────────────────
_HIST_KEY = "landing_history"
_HIST_MAX = 200


def _record_snapshot(res: PipelineResult) -> None:
    """Append a new snapshot to the rolling history. Skips if essentially
    identical to the last one, so the buffer is meaningful even when the
    page is reloaded fast."""
    if res is None or res.spot is None or res.spot <= 0:
        return
    if res.rn_mean is None and res.rn_oi_mean is None:
        return

    now = datetime.now(timezone.utc)
    buf: list[dict] = st.session_state.setdefault(_HIST_KEY, [])

    new = {
        "ts": now,
        "spot": float(res.spot),
        "rn_mean": float(res.rn_mean) if res.rn_mean is not None else None,
        "rn_oi_mean": float(res.rn_oi_mean) if res.rn_oi_mean is not None else None,
        "p_above": (
            float(res.rn_p_above_spot) if res.rn_p_above_spot is not None else None
        ),
        "p_above_oi": (
            float(res.rn_oi_p_above_spot) if res.rn_oi_p_above_spot is not None else None
        ),
        "gap_abs": (
            float(res.rn_mean - res.spot) if res.rn_mean is not None else None
        ),
        "gap_oi_abs": (
            float(res.rn_oi_mean - res.spot) if res.rn_oi_mean is not None else None
        ),
    }

    if buf:
        last = buf[-1]
        same_spot = abs(last.get("spot", 0.0) - new["spot"]) < 1e-3
        same_rn = (
            (last.get("rn_mean") or 0.0) == (new["rn_mean"] or 0.0)
            and (last.get("rn_oi_mean") or 0.0) == (new["rn_oi_mean"] or 0.0)
        )
        if same_spot and same_rn:
            return  # nothing meaningfully changed

    buf.append(new)
    if len(buf) > _HIST_MAX:
        buf = buf[-_HIST_MAX:]
    st.session_state[_HIST_KEY] = buf


def _history_df() -> pd.DataFrame:
    buf = st.session_state.get(_HIST_KEY, [])
    if not buf:
        return pd.DataFrame()
    df = pd.DataFrame(buf)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _chart_rn_distributions(res: PipelineResult) -> None:
    rn = getattr(res, "rn_curve", None)
    rn_oi = getattr(res, "rn_oi_curve", None)
    if rn is None and rn_oi is None:
        st.caption("Risk-neutral density unavailable — Deribit chain insufficient.")
        return

    spot = float(res.spot or 0.0)
    fig = go.Figure()

    # Base RN curve (gold)
    if rn is not None:
        K = np.asarray(rn["K"], dtype=float)
        pdf = np.asarray(rn["pdf"], dtype=float)
        # Background shading around spot for the base curve only
        if spot > 0:
            for mask, color in (
                (K <= spot, "rgba(168,50,50,0.08)"),
                (K >= spot, "rgba(26,122,107,0.08)"),
            ):
                if mask.any():
                    xs = np.concatenate([[K[mask][0]], K[mask], [K[mask][-1]]])
                    ys = np.concatenate([[0], pdf[mask], [0]])
                    fig.add_trace(go.Scatter(
                        x=xs, y=ys, mode="none", fill="toself",
                        fillcolor=color, hoverinfo="skip", showlegend=False,
                    ))
        fig.add_trace(go.Scatter(
            x=K, y=pdf, mode="lines",
            name="RN density (BL)",
            line=dict(color=GOLD, width=2.4),
            hovertemplate="$%{x:,.0f}<br>density %{y:.6f}<extra>RN</extra>",
        ))

    # OI-adjusted RN curve (teal)
    if rn_oi is not None:
        K2 = np.asarray(rn_oi["K"], dtype=float)
        pdf2 = np.asarray(rn_oi["pdf"], dtype=float)
        fig.add_trace(go.Scatter(
            x=K2, y=pdf2, mode="lines",
            name="OI-adjusted RN density",
            line=dict(color=TEAL, width=2.0, dash="dash"),
            hovertemplate="$%{x:,.0f}<br>density %{y:.6f}<extra>OI-adj</extra>",
        ))

    # Reference verticals
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.6, dash="dot"),
                      annotation_text=f"  Spot ${spot:,.0f}",
                      annotation_font=dict(color=INK, size=10),
                      annotation_position="top")

    if rn is not None:
        rn_mean = float(rn["mean"])
        fig.add_vline(x=rn_mean, line=dict(color=GOLD, width=1.4, dash="dash"),
                      annotation_text=f"  RN mean ${rn_mean:,.0f}",
                      annotation_font=dict(color=GOLD, size=10),
                      annotation_position="top right")
    if rn_oi is not None:
        rn_oi_mean = float(rn_oi["mean"])
        fig.add_vline(x=rn_oi_mean, line=dict(color=TEAL, width=1.4, dash="dash"),
                      annotation_text=f"  OI-adj mean ${rn_oi_mean:,.0f}",
                      annotation_font=dict(color=TEAL, size=10),
                      annotation_position="bottom right")

    p_above = res.rn_p_above_spot
    p_above_oi = res.rn_oi_p_above_spot
    p_label_bits = []
    if p_above is not None:
        p_label_bits.append(f"P(above spot) RN {p_above*100:.1f}%")
    if p_above_oi is not None:
        p_label_bits.append(f"OI-adj {p_above_oi*100:.1f}%")
    title = " · ".join(p_label_bits) if p_label_bits else "Risk-Neutral Distribution"

    fig.update_layout(
        **base_layout(title=title, height=420),
        xaxis=dict(title="BTC at expiry ($)", gridcolor="#EDEBE6"),
        yaxis=dict(title="Probability density", gridcolor="#EDEBE6"),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_gex(res: PipelineResult) -> None:
    gex = getattr(res, "gex", None)
    if gex is None or gex.by_strike is None or gex.by_strike.empty:
        st.caption("Dealer GEX unavailable — Deribit chain insufficient.")
        return

    df = gex.by_strike.copy().sort_values("strike").reset_index(drop=True)
    bar_b = df["gex_usd_per_pct"] / 1e9
    cum_b = bar_b.cumsum()
    bar_colors = [TEAL if v >= 0 else RED for v in bar_b]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["strike"], y=bar_b,
        marker=dict(color=bar_colors, line=dict(width=0)),
        name="Per-strike GEX (B$/1%)",
        hovertemplate="Strike $%{x:,.0f}<br>%{y:+.3f} B$/1%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["strike"], y=cum_b, mode="lines",
        line=dict(color=GOLD, width=2.0),
        name="Cumulative GEX",
        yaxis="y2",
        hovertemplate="Strike $%{x:,.0f}<br>Cum %{y:+.3f} B$/1%<extra></extra>",
    ))

    if res.spot:
        fig.add_vline(x=float(res.spot), line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${res.spot:,.0f}",
                      annotation_font=dict(color=INK, size=10))
    if gex.flip_strike:
        fig.add_vline(x=gex.flip_strike, line=dict(color=AMBER, width=1.6, dash="dash"),
                      annotation_text=f" Flip ${gex.flip_strike:,.0f}",
                      annotation_font=dict(color=AMBER, size=10))
    fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))

    total_b = gex.gex_usd_per_pct / 1e9
    regime = "long γ · stable" if total_b > 0 else "short γ · fragile"
    fig.update_layout(
        **base_layout(
            title=f"Dealer GEX by strike · total {total_b:+.2f} B$/1% · {regime} · "
                  f"{gex.n_options} contracts",
            height=380,
        ),
        xaxis=dict(title="Strike ($)", gridcolor="#EDEBE6"),
        yaxis=dict(title="Per-strike GEX (B$/1%)", gridcolor="#EDEBE6"),
        yaxis2=dict(title="Cumulative GEX (B$/1%)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color=GOLD)),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_gap_history(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.caption("RN-mean gap history will populate after the next refresh.")
        return
    fig = go.Figure()
    if hist["gap_abs"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["ts"], y=hist["gap_abs"],
            mode="lines+markers",
            name="RN mean − Spot ($)",
            line=dict(color=GOLD, width=2.2),
            marker=dict(size=4, color=GOLD),
            hovertemplate="%{x|%d %b %H:%M}<br>RN gap %{y:+,.0f}<extra></extra>",
        ))
    if hist["gap_oi_abs"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["ts"], y=hist["gap_oi_abs"],
            mode="lines+markers",
            name="OI-adj mean − Spot ($)",
            line=dict(color=TEAL, width=2.0, dash="dash"),
            marker=dict(size=4, color=TEAL),
            hovertemplate="%{x|%d %b %H:%M}<br>OI-adj gap %{y:+,.0f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
    fig.update_layout(
        **base_layout(
            title=f"RN mean − spot gap (USD)  ·  {len(hist)} snapshots",
            height=320,
        ),
        xaxis=dict(title=None, gridcolor="#EDEBE6"),
        yaxis=dict(title="Gap ($)", gridcolor="#EDEBE6"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_p_above_history(hist: pd.DataFrame) -> None:
    if hist.empty:
        st.caption("P(above spot) history will populate after the next refresh.")
        return
    fig = go.Figure()
    if hist["p_above"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["ts"], y=hist["p_above"] * 100,
            mode="lines+markers",
            name="RN P(above spot)",
            line=dict(color=GOLD, width=2.2),
            marker=dict(size=4, color=GOLD),
            hovertemplate="%{x|%d %b %H:%M}<br>RN P %{y:.1f}%<extra></extra>",
        ))
    if hist["p_above_oi"].notna().any():
        fig.add_trace(go.Scatter(
            x=hist["ts"], y=hist["p_above_oi"] * 100,
            mode="lines+markers",
            name="OI-adj P(above spot)",
            line=dict(color=TEAL, width=2.0, dash="dash"),
            marker=dict(size=4, color=TEAL),
            hovertemplate="%{x|%d %b %H:%M}<br>OI-adj P %{y:.1f}%<extra></extra>",
        ))
    fig.add_hline(y=50, line=dict(color=STONE, width=1, dash="dot"),
                  annotation_text="50%", annotation_font=dict(color=STONE, size=9))
    fig.update_layout(
        **base_layout(
            title=f"P(above spot) history  ·  {len(hist)} snapshots",
            height=320,
        ),
        xaxis=dict(title=None, gridcolor="#EDEBE6"),
        yaxis=dict(title="Probability (%)", range=[0, 100], gridcolor="#EDEBE6"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar(res: PipelineResult) -> None:
    sb = st.sidebar
    sb.markdown("### Status")
    sb.caption(f"App: {settings.app_title}")
    sb.caption(f"Version: {settings.version}")
    env_color = TEAL if settings.environment == "production" else STONE
    sb.markdown(
        f'<div style="font-family:DM Mono,monospace;font-size:11px;'
        f'color:{env_color};">Env: {settings.environment}</div>',
        unsafe_allow_html=True,
    )
    sb.markdown("---")
    sb.markdown("### Snapshot")
    sb.caption(f"Spot: {fmt_money(res.spot)}")
    sb.caption(f"ATM IV: {res.atm_iv_pct:.1f}%" if res.atm_iv_pct is not None else "ATM IV: —")
    sb.caption(f"RN mean: {fmt_money(res.rn_mean)}")
    sb.caption(f"OI-adj mean: {fmt_money(res.rn_oi_mean)}")

    sb.markdown("---")
    sb.markdown("### History")
    n = len(st.session_state.get(_HIST_KEY, []))
    sb.caption(f"{n} snapshots stored (this session)")
    if sb.button("Clear history", use_container_width=True):
        st.session_state[_HIST_KEY] = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    log.info("Starting %s in %s mode", settings.app_title, settings.environment)

    page_title(
        settings.app_title,
        "Deribit-driven options distribution · GEX · drift over time",
    )

    try:
        res: PipelineResult = _pipe()["result"]
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    _sidebar(res)
    _record_snapshot(res)
    hist = _history_df()

    # 1) RN PDF + OI-adjusted overlay
    st.markdown(section_label("Risk-Neutral Distribution"), unsafe_allow_html=True)
    _chart_rn_distributions(res)

    # 2) GEX per strike
    st.markdown(section_label("Dealer GEX by Strike"), unsafe_allow_html=True)
    _chart_gex(res)

    # 3) RN-mean − spot gap history
    st.markdown(section_label("RN-Mean − Spot Gap (USD, over time)"),
                unsafe_allow_html=True)
    _chart_gap_history(hist)

    # 4) P(above spot) history
    st.markdown(section_label("P(above spot) History"), unsafe_allow_html=True)
    _chart_p_above_history(hist)

    st.caption(
        "All charts are driven by the live Deribit option chain. The OI-adjusted "
        "curve tilts the BL density by the smoothed open-interest profile across "
        "strikes — a positioning view, not a true risk-neutral density. "
        "Time-series charts build their history from this browser session — "
        "two refreshes minimum to draw a line."
    )


main()
