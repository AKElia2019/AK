"""
btc_dashboard.pages.signal_engine
Signal Engine deep-dive.

Per-family scores, per-signal contributions, family-weighted breakdown,
and the score history overlay.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from charts.theme import (   # noqa: E402
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, inject_global_css, page_title, section_label,
)
from analytics.pipeline import run_pipeline, PipelineResult  # noqa: E402
from analytics.scoring import DEFAULT_FAMILY_WEIGHTS         # noqa: E402


@st.cache_data(ttl=30, show_spinner="Running pipeline…")
def _pipe() -> dict:
    return {"result": run_pipeline()}


def main() -> None:
    st.set_page_config(page_title="Signal Engine · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Signal Engine",
               "Per-signal · per-family · weighted composite breakdown")

    res: PipelineResult = _pipe()["result"]

    # ── Family breakdown ──────────────────────────────────────────────────
    st.markdown(section_label("Family Composite (4h)"), unsafe_allow_html=True)
    fams = list(res.composite_4h.sub_scores.values())
    df = pd.DataFrame([
        {
            "Family": f.family.title(),
            "Score": round(f.score, 1),
            "Weight": round(f.weight * 100, 1),
            "Contribution": round(f.contribution, 1),
            "Signals": f.n_signals,
        }
        for f in fams
    ])

    cols = st.columns([2, 3])
    with cols[0]:
        st.dataframe(df, use_container_width=True, hide_index=True)

    with cols[1]:
        fig = go.Figure(go.Bar(
            x=df["Family"], y=df["Contribution"],
            marker=dict(color=[TEAL if v >= 0 else RED for v in df["Contribution"]],
                        line=dict(width=0)),
            hovertemplate="%{x}<br>contrib %{y:+.1f} pts<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color=STONE, width=1))
        fig.update_layout(**base_layout(title="Contribution to 4h composite", height=260),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Signal table ──────────────────────────────────────────────────────
    st.markdown(section_label("Signal Table"), unsafe_allow_html=True)
    if res.signal_table is not None and not res.signal_table.empty:
        def _row(row):
            v = float(row["value"])
            c = (f"color:{TEAL};font-weight:600;" if v > 0.4 else
                 f"color:{TEAL};" if v > 0.1 else
                 f"color:{RED};font-weight:600;" if v < -0.4 else
                 f"color:{RED};" if v < -0.1 else "")
            return ["", "", c, ""]
        st.dataframe(
            res.signal_table.style.apply(_row, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Signal table empty — pipeline returned no signals this cycle.")

    # ── Score history (raw + smoothed) ────────────────────────────────────
    if not res.history_4h.empty:
        st.markdown(section_label("Score History · 1h vs 4h"), unsafe_allow_html=True)
        fig = go.Figure()
        h4 = res.history_4h
        fig.add_trace(go.Scatter(
            x=h4.index, y=h4["composite"], mode="lines",
            name="4h raw", line=dict(color=STONE, width=1, dash="dot"), opacity=0.5,
        ))
        if "composite_smooth" in h4.columns:
            fig.add_trace(go.Scatter(
                x=h4.index, y=h4["composite_smooth"], mode="lines",
                name="4h smoothed", line=dict(color=GOLD, width=2.4),
            ))
        h1 = res.history_1h
        if not h1.empty and "composite_smooth" in h1.columns:
            fig.add_trace(go.Scatter(
                x=h1.index, y=h1["composite_smooth"], mode="lines",
                name="1h smoothed", line=dict(color=TEAL, width=1.6),
            ))
        for y, lbl, c in ((50, "long gate", TEAL), (-50, "short gate", RED)):
            fig.add_hline(y=y, line=dict(color=c, width=1, dash="dot"),
                          annotation_text=lbl, annotation_font=dict(color=c, size=9))
        fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
        fig.update_layout(
            **base_layout(height=320),
            xaxis=dict(title=None, gridcolor="#E5DCC9"),
            yaxis=dict(title="Score", range=[-100, 100], gridcolor="#E5DCC9"),
            legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Family weights reference ──────────────────────────────────────────
    st.markdown(section_label("Default Family Weights"), unsafe_allow_html=True)
    st.caption("Weights applied to each family before aggregation into the composite score.")
    st.dataframe(
        pd.DataFrame([
            {"Family": k.title(), "Weight": f"{v*100:.0f}%"}
            for k, v in DEFAULT_FAMILY_WEIGHTS.items()
        ]),
        use_container_width=True, hide_index=True,
    )


main()
