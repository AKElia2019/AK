"""
btc_dashboard.pages.flow
Flow deep-dive — recent trade tape · long/short ratio · liquidations.
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
from data.flows import (   # noqa: E402
    fetch_binance_spot_trades,
    fetch_binance_perp_trades,
    fetch_deribit_trades,
    bucket_flow,
)
from data.coinglass import (   # noqa: E402
    fetch_coinglass_long_short_ratio,
    fetch_coinglass_liquidations,
)


@st.cache_data(ttl=15, show_spinner=False)
def _trades(venue: str, limit: int = 500) -> pd.DataFrame:
    if venue == "binance_spot":
        return fetch_binance_spot_trades("BTCUSDT", limit)
    if venue == "binance_perp":
        return fetch_binance_perp_trades("BTCUSDT", limit)
    if venue == "deribit":
        return fetch_deribit_trades("BTC-PERPETUAL", limit)
    return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def _ls_ratio() -> pd.DataFrame:
    return fetch_coinglass_long_short_ratio(interval="1h", limit=72)


@st.cache_data(ttl=30, show_spinner=False)
def _liq() -> pd.DataFrame:
    return fetch_coinglass_liquidations(interval="1h", limit=72)


def _render_aggressor_flow(trades: pd.DataFrame, freq: str = "1min") -> None:
    if trades is None or trades.empty:
        st.caption("No trades available.")
        return
    buckets = bucket_flow(trades, freq=freq)
    if buckets.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=buckets["bucket"], y=buckets["buy_notional"],
        marker=dict(color=TEAL, opacity=0.85, line=dict(width=0)),
        name="Buy aggressor ($)",
    ))
    fig.add_trace(go.Bar(
        x=buckets["bucket"], y=-buckets["sell_notional"],
        marker=dict(color=RED, opacity=0.85, line=dict(width=0)),
        name="Sell aggressor ($)",
    ))
    fig.add_trace(go.Scatter(
        x=buckets["bucket"], y=buckets["net_notional"],
        mode="lines+markers",
        line=dict(color=GOLD, width=2),
        marker=dict(size=4),
        name="Net",
    ))
    fig.add_hline(y=0, line=dict(color=STONE, width=1))
    fig.update_layout(**base_layout(title=f"Aggressor flow · {freq} buckets", height=320),
                      barmode="relative",
                      xaxis=dict(title=None, gridcolor="#E5DCC9"),
                      yaxis=dict(title="Notional ($)", gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_recent_tape(trades: pd.DataFrame, n: int = 30) -> None:
    if trades is None or trades.empty:
        return
    df = trades.tail(n).iloc[::-1].copy()
    df["price"] = df["price"].astype(float).round(2)
    df["size"] = df["size"].astype(float).round(4)
    df["notional"] = df["notional"].astype(float).round(0)
    df = df[["time", "side", "price", "size", "notional"]]

    def _row_style(row):
        c = TEAL if row["side"] == "buy" else RED
        return ["", f"color:{c};font-weight:600;", f"color:{c};", f"color:{c};", ""]

    st.dataframe(
        df.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _render_ls_ratio() -> None:
    df = _ls_ratio()
    if df is None or df.empty:
        st.caption("Long/short ratio unavailable.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["long_pct"], mode="lines",
        line=dict(color=TEAL, width=2),
        name="Long %",
    ))
    fig.add_trace(go.Scatter(
        x=df["time"], y=df["short_pct"], mode="lines",
        line=dict(color=RED, width=2),
        name="Short %",
    ))
    fig.add_hline(y=50, line=dict(color=STONE, width=1, dash="dot"))
    src = str(df["_source"].iloc[-1])
    fig.update_layout(**base_layout(title=f"Long / Short account ratio (Coinglass · {src})",
                                    height=300),
                      xaxis=dict(title=None, gridcolor="#E5DCC9"),
                      yaxis=dict(title="% of accounts", range=[0, 100], gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_liquidations() -> None:
    df = _liq()
    if df is None or df.empty:
        st.caption("Liquidations history unavailable.")
        return
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["time"], y=df["long_liq_usd"],
        marker=dict(color=RED, opacity=0.8, line=dict(width=0)),
        name="Long liquidations ($)",
    ))
    fig.add_trace(go.Bar(
        x=df["time"], y=-df["short_liq_usd"],
        marker=dict(color=TEAL, opacity=0.8, line=dict(width=0)),
        name="Short liquidations ($)",
    ))
    fig.add_hline(y=0, line=dict(color=STONE, width=1))
    src = str(df["_source"].iloc[-1])
    fig.update_layout(**base_layout(title=f"Liquidations (Coinglass · {src})",
                                    height=300),
                      barmode="relative",
                      xaxis=dict(title=None, gridcolor="#E5DCC9"),
                      yaxis=dict(title="USD notional", gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Flow · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Flow", "Aggressor tape · long/short ratio · liquidations")

    sb = st.sidebar
    venue = sb.selectbox("Venue (trade tape)", ["binance_perp", "binance_spot", "deribit"], index=0)
    freq = sb.selectbox("Aggregation bucket", ["30s", "1min", "5min"], index=1)
    n_tape = sb.slider("Tape rows", 10, 100, 30, 5)

    trades = _trades(venue, limit=500)

    cols = st.columns(2, gap="medium")
    with cols[0]:
        st.markdown(section_label("Aggressor Flow"), unsafe_allow_html=True)
        _render_aggressor_flow(trades, freq=freq)
    with cols[1]:
        st.markdown(section_label("Recent Trades (tape)"), unsafe_allow_html=True)
        _render_recent_tape(trades, n=n_tape)

    cols2 = st.columns(2, gap="medium")
    with cols2[0]:
        st.markdown(section_label("Long / Short Ratio"), unsafe_allow_html=True)
        _render_ls_ratio()
    with cols2[1]:
        st.markdown(section_label("Liquidations"), unsafe_allow_html=True)
        _render_liquidations()


main()
