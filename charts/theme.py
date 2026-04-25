"""
btc_dashboard.charts.theme
Shared theme tokens + small helpers used by every page.
"""

from __future__ import annotations

import streamlit as st


GOLD  = "#C9A55A"
TEAL  = "#1A7A6B"
RED   = "#A83232"
AMBER = "#B8832A"
STONE = "#9C968A"
INK   = "#1C1A17"


def section_label(text: str) -> str:
    return (
        f'<div style="font-family:DM Mono,monospace;font-size:9px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{GOLD};'
        f'margin:18px 0 6px 0;">{text}</div>'
    )


def page_title(title: str, tagline: str) -> None:
    st.markdown(
        f'<h1 style="font-family:DM Sans,sans-serif;font-weight:300;'
        f'color:{INK};margin:0 0 4px 0;">{title}</h1>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{STONE};'
        f'margin-bottom:14px;">{tagline}</div>',
        unsafe_allow_html=True,
    )


def fmt_money(x) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except Exception:
        return "—"
    if not v == v:  # NaN
        return "—"
    return f"${v:,.0f}"


def base_layout(title: str = None, height: int = 320) -> dict:
    """Plotly layout dict — light theme matching the rest of the app.

    Returns paper / plot bg, font, height, margin, and title. Callers are
    expected to supply their own `xaxis=` / `yaxis=` (and any secondary
    axes) so the helper never collides with caller kwargs when expanded
    via `**base_layout(...)`.
    """
    return dict(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="DM Mono, monospace", color="#44403A", size=11),
        height=height,
        margin=dict(l=58, r=20, t=30 if title else 12, b=44),
        title=(
            dict(text=title, font=dict(size=12, color=INK), x=0.0, xanchor="left")
            if title else None
        ),
    )
