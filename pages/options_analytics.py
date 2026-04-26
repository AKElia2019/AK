"""
btc_dashboard.pages.options_analytics
Options Analytics deep-dive — IV smile, RN density, dealer GEX, chain table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from charts.theme import (   # noqa: E402
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, inject_global_css, page_title, section_label,
)
from analytics.pipeline import run_pipeline, PipelineResult  # noqa: E402


@st.cache_data(ttl=30, show_spinner="Running pipeline…")
def _pipe() -> dict:
    return {"result": run_pipeline()}


def _render_strip(res: PipelineResult) -> None:
    st.markdown(section_label("Options State"), unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("ATM IV",
                   f"{res.atm_iv_pct:.1f}%" if res.atm_iv_pct is not None else "—")
    rn_lbl = (f"{(res.rn_mean - res.spot)/res.spot*100:+.2f}% vs spot"
              if (res.rn_mean and res.spot) else None)
    cols[1].metric("RN mean", fmt_money(res.rn_mean), rn_lbl)
    cols[2].metric("P(above spot)",
                   f"{res.rn_p_above_spot*100:.1f}%" if res.rn_p_above_spot is not None else "—")
    if res.gex is not None:
        gex_b = res.gex.gex_usd_per_pct / 1e9
        regime = "long γ · stable" if gex_b > 0 else "short γ · fragile"
        cols[3].metric("Dealer GEX", f"{gex_b:+.2f} B$/1%", regime)
    else:
        cols[3].metric("Dealer GEX", "—")


def _render_rn_pdf(res: PipelineResult) -> None:
    rn = getattr(res, "rn_curve", None)
    if rn is None:
        st.caption("RN density unavailable — chain insufficient or scipy missing.")
        return
    K = np.asarray(rn["K"], dtype=float)
    pdf = np.asarray(rn["pdf"], dtype=float)
    mean, std, mode = float(rn["mean"]), float(rn["std"]), float(rn["mode"])
    spot = float(res.spot or 0.0)

    fig = go.Figure()
    if spot > 0:
        for mask, color in (
            (K <= spot, "rgba(168,50,50,0.10)"),
            (K >= spot, "rgba(26,122,107,0.10)"),
        ):
            if mask.any():
                xs = np.concatenate([[K[mask][0]], K[mask], [K[mask][-1]]])
                ys = np.concatenate([[0], pdf[mask], [0]])
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="none", fill="toself",
                                          fillcolor=color, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=K, y=pdf, mode="lines",
                             line=dict(color=GOLD, width=2.4), name="RN density"))
    if K[0] <= spot <= K[-1]:
        fig.add_vline(x=spot, line=dict(color=TEAL, width=1.6, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
                      annotation_font=dict(color=TEAL, size=10))
    if K[0] <= mean <= K[-1]:
        fig.add_vline(x=mean, line=dict(color=AMBER, width=1.6, dash="dash"),
                      annotation_text=f" RN mean ${mean:,.0f}",
                      annotation_font=dict(color=AMBER, size=10))
    if K[0] <= mode <= K[-1]:
        fig.add_vline(x=mode, line=dict(color=STONE, width=1.0, dash="dash"))
    fig.update_layout(**base_layout(title=f"Risk-Neutral Density · skew {rn.get('skew',0):+.2f}",
                                    height=340),
                      xaxis=dict(title="BTC at expiry ($)", gridcolor="#E5DCC9"),
                      yaxis=dict(title="Density", gridcolor="#E5DCC9"),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_gex(res: PipelineResult) -> None:
    if res.gex is None or res.gex.by_strike is None or res.gex.by_strike.empty:
        st.caption("GEX unavailable.")
        return
    df = res.gex.by_strike.copy().sort_values("strike").reset_index(drop=True)
    bar_b = df["gex_usd_per_pct"] / 1e9
    cum_b = bar_b.cumsum()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["strike"], y=bar_b,
                          marker=dict(color=[TEAL if v >= 0 else RED for v in bar_b]),
                          name="Per-strike"))
    fig.add_trace(go.Scatter(x=df["strike"], y=cum_b, mode="lines",
                              line=dict(color=GOLD, width=2.0),
                              name="Cumulative", yaxis="y2"))
    if res.spot:
        fig.add_vline(x=float(res.spot), line=dict(color=TEAL, width=1.4, dash="dot"))
    if res.gex.flip_strike:
        fig.add_vline(x=res.gex.flip_strike, line=dict(color=AMBER, width=1.4, dash="dash"),
                      annotation_text=f" Flip ${res.gex.flip_strike:,.0f}",
                      annotation_font=dict(color=AMBER, size=10))
    fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
    fig.update_layout(**base_layout(title=f"GEX by strike · total {bar_b.sum():+.2f} B$/1%",
                                    height=340),
                      xaxis=dict(title="Strike ($)", gridcolor="#E5DCC9"),
                      yaxis=dict(title="GEX (B$/1%)", gridcolor="#E5DCC9"),
                      yaxis2=dict(title="Cumulative", overlaying="y", side="right",
                                  showgrid=False),
                      bargap=0.05,
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_smile(res: PipelineResult) -> None:
    """Implied-vol smile from the Deribit chain (latest expiry by default)."""
    chain = res.raw.get("chain") if isinstance(res.raw, dict) else None
    if chain is None or chain.empty:
        st.caption("Chain unavailable.")
        return
    df = chain.copy()
    if "expiry" in df.columns and df["expiry"].notna().any():
        # Pick the soonest expiry with at least 6 rows
        best = None
        for exp_val, group in df.groupby("expiry"):
            if len(group) >= 6:
                best = (exp_val, group)
                break
        if best is None:
            return
        exp_val, sub = best
        try:
            exp_label = pd.Timestamp(exp_val).strftime("%d %b %y")
        except Exception:
            exp_label = str(exp_val)
    else:
        sub = df
        exp_label = "—"

    sub = sub.dropna(subset=["iv", "strike"])
    calls = sub[sub["type"].astype(str).str.upper() == "CALL"].sort_values("strike")
    puts = sub[sub["type"].astype(str).str.upper() == "PUT"].sort_values("strike")

    fig = go.Figure()
    if not calls.empty:
        fig.add_trace(go.Scatter(x=calls["strike"], y=calls["iv"], mode="lines+markers",
                                  name="Calls", line=dict(color=TEAL, width=2),
                                  marker=dict(size=5)))
    if not puts.empty:
        fig.add_trace(go.Scatter(x=puts["strike"], y=puts["iv"], mode="lines+markers",
                                  name="Puts", line=dict(color=RED, width=2),
                                  marker=dict(size=5)))
    if res.spot:
        fig.add_vline(x=float(res.spot), line=dict(color=GOLD, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${res.spot:,.0f}",
                      annotation_font=dict(color=GOLD, size=10))
    fig.update_layout(**base_layout(title=f"IV Smile · expiry {exp_label}",
                                    height=320),
                      xaxis=dict(title="Strike ($)", gridcolor="#E5DCC9"),
                      yaxis=dict(title="Implied Vol (%)", gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_chain_table(res: PipelineResult) -> None:
    chain = res.raw.get("chain") if isinstance(res.raw, dict) else None
    if chain is None or chain.empty:
        return
    df = chain[["instrument", "type", "strike", "dte", "iv", "mark", "open_interest", "volume"]].copy()
    df = df.dropna(subset=["strike"])
    df["strike"] = df["strike"].astype(float)
    df = df.sort_values(["dte", "strike"]).head(60)
    df["mark"] = df["mark"].round(4)
    df["iv"] = df["iv"].round(1)
    df["dte"] = df["dte"].round(1)
    st.markdown(section_label("Chain (first 60 rows)"), unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Options Analytics · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Options Analytics", "IV smile · RN density · dealer GEX · chain")
    res: PipelineResult = _pipe()["result"]
    _render_strip(res)
    cols = st.columns(2, gap="medium")
    with cols[0]:
        _render_rn_pdf(res)
    with cols[1]:
        _render_gex(res)
    _render_smile(res)
    _render_chain_table(res)


main()
