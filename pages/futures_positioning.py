"""
btc_dashboard.pages.futures_positioning
Futures Positioning deep-dive — OI history, funding, basis, OI/price quadrant.
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


def _render_metrics(res: PipelineResult) -> None:
    raw = res.raw if isinstance(res.raw, dict) else {}
    funding = raw.get("funding")
    oi_4h = raw.get("oi_4h")

    cols = st.columns(4)
    if funding is not None and not funding.empty:
        last = float(funding["funding_rate"].iloc[-1])
        ann = last * 3 * 365 * 100
        cols[0].metric("Binance funding", f"{last*100:+.4f}%/8h", f"{ann:+.1f}% ann")
    else:
        cols[0].metric("Binance funding", "—")

    cg_f = raw.get("cg_funding")
    if cg_f is not None and not cg_f.empty:
        last = float(cg_f["funding_rate"].iloc[-1])
        ann = last * 3 * 365 * 100
        cols[1].metric("Aggregated funding", f"{last*100:+.4f}%/8h", f"{ann:+.1f}% ann · OI-wtd")
    else:
        cols[1].metric("Aggregated funding", "—")

    if oi_4h is not None and not oi_4h.empty:
        oi_btc = float(oi_4h["oi_base"].iloc[-1])
        oi_usd = float(oi_4h["oi_usd"].iloc[-1])
        if len(oi_4h) >= 2:
            chg = (oi_4h["oi_base"].iloc[-1] / oi_4h["oi_base"].iloc[-2] - 1) * 100
            cols[2].metric("Binance OI", f"{oi_btc:,.0f} BTC", f"{chg:+.2f}% (4h)")
        else:
            cols[2].metric("Binance OI", f"{oi_btc:,.0f} BTC")
    else:
        cols[2].metric("Binance OI", "—")

    cg_oi = raw.get("cg_oi")
    if cg_oi is not None and not cg_oi.empty:
        last = float(cg_oi["oi_usd"].iloc[-1])
        chg = float(cg_oi["oi_change_pct"].iloc[-1])
        cols[3].metric("Aggregated OI", fmt_money(last), f"{chg:+.2f}%")
    else:
        cols[3].metric("Aggregated OI", "—")


def _render_funding_history(res: PipelineResult) -> None:
    raw = res.raw if isinstance(res.raw, dict) else {}
    funding = raw.get("funding")
    if funding is None or funding.empty:
        return
    df = funding.copy()
    df["ann_pct"] = df["funding_rate"] * 3 * 365 * 100
    fig = go.Figure(go.Scatter(
        x=df["time"], y=df["ann_pct"], mode="lines+markers",
        line=dict(color=GOLD, width=2),
        marker=dict(size=5),
        name="Annualised funding (%)",
    ))
    fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
    fig.add_hline(y=30, line=dict(color=RED, width=1, dash="dot"),
                  annotation_text=" extreme + (contrarian)",
                  annotation_font=dict(color=RED, size=9))
    fig.add_hline(y=-30, line=dict(color=TEAL, width=1, dash="dot"),
                  annotation_text=" extreme − (contrarian)",
                  annotation_font=dict(color=TEAL, size=9))
    fig.update_layout(**base_layout(title="Binance perp funding (annualised %)", height=300),
                      xaxis=dict(title=None, gridcolor="#E5DCC9"),
                      yaxis=dict(title="% annualised", gridcolor="#E5DCC9"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_oi_history(res: PipelineResult) -> None:
    raw = res.raw if isinstance(res.raw, dict) else {}
    oi_1h = raw.get("oi_1h")
    oi_4h = raw.get("oi_4h")
    if (oi_1h is None or oi_1h.empty) and (oi_4h is None or oi_4h.empty):
        return
    fig = go.Figure()
    if oi_1h is not None and not oi_1h.empty:
        fig.add_trace(go.Scatter(x=oi_1h["time"], y=oi_1h["oi_base"],
                                  mode="lines", name="OI 1h",
                                  line=dict(color=TEAL, width=1.6)))
    if oi_4h is not None and not oi_4h.empty:
        fig.add_trace(go.Scatter(x=oi_4h["time"], y=oi_4h["oi_base"],
                                  mode="lines", name="OI 4h",
                                  line=dict(color=GOLD, width=2.2)))
    fig.update_layout(**base_layout(title="Binance perp open interest (BTC)", height=300),
                      xaxis=dict(title=None, gridcolor="#E5DCC9"),
                      yaxis=dict(title="OI (BTC)", gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_quadrant_scatter(res: PipelineResult) -> None:
    """Plot Δprice vs ΔOI for the last 50 4h bars — visualises the quadrant signal."""
    raw = res.raw if isinstance(res.raw, dict) else {}
    kl = raw.get("kl_4h")
    oi = raw.get("oi_4h")
    if kl is None or kl.empty or oi is None or oi.empty:
        return
    closes = kl["close"].astype(float).reset_index(drop=True)
    times = pd.to_datetime(kl["time"], utc=True).reset_index(drop=True)
    spot_pct = closes.pct_change().fillna(0) * 100

    oi_idx = pd.to_datetime(oi["time"], utc=True)
    oi_series = pd.Series(oi["oi_base"].astype(float).values, index=oi_idx).sort_index()
    oi_at_bar = (
        oi_series.reindex(oi_series.index.union(times))
        .sort_index().ffill().reindex(times)
    )
    oi_pct = oi_at_bar.pct_change().fillna(0) * 100

    df = pd.DataFrame({"time": times, "spot_pct": spot_pct.values, "oi_pct": oi_pct.values}).tail(50).reset_index(drop=True)

    # Recency shading: oldest dot α=0.28, most recent α=1.0 — same fan logic
    # as the forward RN-mean overlay on the landing page.
    n = len(df)
    colors: list[str] = []
    for i, (s, o) in enumerate(zip(df["spot_pct"], df["oi_pct"])):
        alpha = 0.28 + 0.72 * (i / max(n - 1, 1))
        if s > 0 and o > 0:
            colors.append(f"rgba(74,111,165,{alpha:.2f})")    # steel · new longs
        elif s < 0 and o > 0:
            colors.append(f"rgba(163,90,72,{alpha:.2f})")     # terracotta · new shorts
        else:
            colors.append(f"rgba(182,146,72,{alpha:.2f})")    # brass · unwind / cover

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["spot_pct"], y=df["oi_pct"], mode="markers",
        marker=dict(size=9, color=colors, line=dict(width=0)),
        text=df["time"].dt.strftime("%d %b %H:%M"),
        hovertemplate="%{text}<br>ΔSpot %{x:+.2f}%<br>ΔOI %{y:+.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=STONE, width=1, dash="dot"))
    fig.add_annotation(x=2, y=2, text="New longs", showarrow=False,
                       font=dict(color=TEAL, size=11))
    fig.add_annotation(x=-2, y=2, text="New shorts", showarrow=False,
                       font=dict(color=RED, size=11))
    fig.add_annotation(x=2, y=-2, text="Short cover", showarrow=False,
                       font=dict(color=AMBER, size=11))
    fig.add_annotation(x=-2, y=-2, text="Long unwind", showarrow=False,
                       font=dict(color=AMBER, size=11))
    fig.update_layout(**base_layout(title="OI / price quadrant (last 50 × 4h bars)", height=340),
                      xaxis=dict(title="Δ Spot (%)", gridcolor="#E5DCC9", zeroline=False),
                      yaxis=dict(title="Δ OI (%)", gridcolor="#E5DCC9", zeroline=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Futures Positioning · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Futures Positioning", "Funding · OI · price-OI quadrant")
    res: PipelineResult = _pipe()["result"]
    st.markdown(section_label("Live Snapshot"), unsafe_allow_html=True)
    _render_metrics(res)
    cols = st.columns(2, gap="medium")
    with cols[0]:
        _render_funding_history(res)
    with cols[1]:
        _render_oi_history(res)
    _render_quadrant_scatter(res)


main()
