"""
btc_dashboard.pages.spot_volume
Spot / Volume deep-dive — price candle + SMA stack + VWAP + volume bars.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from charts.theme import (   # noqa: E402
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, inject_global_css, page_title, section_label,
)
from data.futures import fetch_binance_perp_klines  # noqa: E402


@st.cache_data(ttl=30, show_spinner=False)
def _klines(interval: str, limit: int) -> pd.DataFrame:
    return fetch_binance_perp_klines(interval=interval, limit=limit)


def _render_chart(interval: str, limit: int, sma_short: int = 20,
                   sma_mid: int = 50, sma_long: int = 200, vwap_window: int = 24) -> None:
    df = _klines(interval, limit).copy()
    if df.empty or len(df) < 5:
        st.caption(f"No data for {interval}.")
        return

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    opens = df["open"].astype(float)
    vols = df["volume"].astype(float).replace(0.0, np.nan)
    times = pd.to_datetime(df["time"], utc=True)

    typ = (highs + lows + closes) / 3.0
    pv = typ * vols
    vwap = pv.rolling(vwap_window, min_periods=vwap_window).sum() / vols.rolling(vwap_window, min_periods=vwap_window).sum()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=TEAL, decreasing_line_color=RED,
        showlegend=False, name="Price",
    ), row=1, col=1)

    for window, color, label in (
        (sma_short, GOLD, f"SMA{sma_short}"),
        (sma_mid, AMBER, f"SMA{sma_mid}"),
        (sma_long, STONE, f"SMA{sma_long}"),
    ):
        sma = closes.rolling(window, min_periods=window).mean()
        fig.add_trace(go.Scatter(x=times, y=sma, mode="lines", name=label,
                                   line=dict(color=color, width=1.4)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=vwap, mode="lines", name=f"VWAP({vwap_window})",
                               line=dict(color="#a78bfa", width=1.6, dash="dot")),
                  row=1, col=1)

    bar_colors = [TEAL if c >= o else RED for c, o in zip(closes, opens)]
    fig.add_trace(go.Bar(x=times, y=df["volume"].astype(float),
                          marker=dict(color=bar_colors, line=dict(width=0)),
                          showlegend=False, name="Volume"),
                  row=2, col=1)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#44403A", size=11),
        height=560, margin=dict(l=58, r=20, t=30, b=44),
        title=dict(
            text=f"BTC perp · {interval} · last {len(df)} bars",
            font=dict(size=12, color=INK), x=0.0, xanchor="left",
        ),
        legend=dict(orientation="h", yanchor="top", y=1.06, xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor="#E5DCC9"),
        xaxis2=dict(gridcolor="#E5DCC9", title=None),
        yaxis=dict(gridcolor="#E5DCC9", title="Price ($)"),
        yaxis2=dict(gridcolor="#E5DCC9", title="Volume"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _render_volume_profile(interval: str = "1h", limit: int = 200, buckets: int = 30) -> None:
    df = _klines(interval, limit)
    if df.empty:
        return
    typical = (df["high"].astype(float) + df["low"].astype(float) + df["close"].astype(float)) / 3.0
    bins = np.linspace(typical.min(), typical.max(), buckets + 1)
    df_b = pd.DataFrame({"price": typical, "volume": df["volume"].astype(float)})
    df_b["bucket"] = pd.cut(df_b["price"], bins=bins, include_lowest=True)
    profile = df_b.groupby("bucket", observed=True)["volume"].sum().reset_index()
    profile["mid"] = profile["bucket"].apply(lambda b: (b.left + b.right) / 2 if hasattr(b, "left") else 0)

    fig = go.Figure(go.Bar(
        x=profile["volume"], y=profile["mid"], orientation="h",
        marker=dict(color=GOLD, opacity=0.7, line=dict(width=0)),
        hovertemplate="$%{y:,.0f}<br>volume %{x:,.0f}<extra></extra>",
        showlegend=False,
    ))
    last_close = float(df["close"].astype(float).iloc[-1])
    fig.add_hline(y=last_close, line=dict(color=TEAL, width=1.6, dash="dot"),
                  annotation_text=f" Spot ${last_close:,.0f}",
                  annotation_font=dict(color=TEAL, size=10))
    fig.update_layout(**base_layout(title=f"Volume profile · {interval} · {buckets} buckets", height=420),
                      xaxis=dict(title="Volume (BTC)", gridcolor="#E5DCC9"),
                      yaxis=dict(title="Price ($)", gridcolor="#E5DCC9"))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="Spot / Volume · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Spot / Volume", "Price candles · SMA stack · VWAP · volume profile")

    sb = st.sidebar
    interval = sb.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], index=1)
    limit = sb.slider("Bars", 50, 500, 200, 10)
    vwap_window = sb.slider("VWAP window (bars)", 5, 96, 24, 1)

    st.markdown(section_label("Price + SMA + VWAP + Volume"), unsafe_allow_html=True)
    _render_chart(interval, limit, vwap_window=vwap_window)

    st.markdown(section_label("Volume Profile"), unsafe_allow_html=True)
    _render_volume_profile(interval, limit)


main()
