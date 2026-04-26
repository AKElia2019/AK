"""
btc_dashboard.pages.liquidity
Liquidity deep-dive — order book depth + spread comparison per venue.
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
    base_layout, inject_global_css, page_title, section_label,
)
from data.liquidity import (   # noqa: E402
    fetch_binance_spot_order_book,
    fetch_binance_perp_order_book,
    fetch_deribit_order_book,
    fetch_coinbase_order_book,
    top_of_book,
)


@st.cache_data(ttl=15, show_spinner=False)
def _book(venue: str) -> pd.DataFrame:
    if venue == "binance_spot":
        return fetch_binance_spot_order_book("BTCUSDT", 100)
    if venue == "binance_perp":
        return fetch_binance_perp_order_book("BTCUSDT", 100)
    if venue == "deribit":
        return fetch_deribit_order_book("BTC-PERPETUAL", 100)
    if venue == "coinbase":
        return fetch_coinbase_order_book("BTC-USD")
    return pd.DataFrame()


def _depth_chart(book: pd.DataFrame, venue: str) -> None:
    if book is None or book.empty:
        st.caption(f"{venue}: no book.")
        return
    bids = book[book["side"] == "bid"].sort_values("price", ascending=False).reset_index(drop=True)
    asks = book[book["side"] == "ask"].sort_values("price", ascending=True).reset_index(drop=True)

    bid_cum = bids["size"].cumsum()
    ask_cum = asks["size"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bids["price"], y=bid_cum, mode="lines",
        line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(26,122,107,0.15)",
        name="Bids (cumulative)",
    ))
    fig.add_trace(go.Scatter(
        x=asks["price"], y=ask_cum, mode="lines",
        line=dict(color=RED, width=2),
        fill="tozeroy", fillcolor="rgba(168,50,50,0.15)",
        name="Asks (cumulative)",
    ))

    if not bids.empty and not asks.empty:
        mid = (bids["price"].iloc[0] + asks["price"].iloc[0]) / 2
        fig.add_vline(x=mid, line=dict(color=GOLD, width=1.4, dash="dot"),
                      annotation_text=f" Mid ${mid:,.0f}",
                      annotation_font=dict(color=GOLD, size=10))

    fig.update_layout(**base_layout(title=f"{venue} · cumulative depth", height=320),
                      xaxis=dict(title="Price ($)", gridcolor="#E5DCC9"),
                      yaxis=dict(title="Size (BTC)", gridcolor="#E5DCC9"),
                      legend=dict(orientation="h", yanchor="top", y=-0.18,
                                  xanchor="center", x=0.5))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _depth_buckets(book: pd.DataFrame) -> dict:
    if book is None or book.empty:
        return {"d05": (0.0, 0.0), "d10": (0.0, 0.0), "d20": (0.0, 0.0), "mid": 0.0}
    bids = book[book["side"] == "bid"].sort_values("price", ascending=False)
    asks = book[book["side"] == "ask"].sort_values("price", ascending=True)
    if bids.empty or asks.empty:
        return {"d05": (0.0, 0.0), "d10": (0.0, 0.0), "d20": (0.0, 0.0), "mid": 0.0}
    mid = (float(bids["price"].iloc[0]) + float(asks["price"].iloc[0])) / 2
    out = {"mid": mid}
    for label, pct in (("d05", 0.005), ("d10", 0.010), ("d20", 0.020)):
        bid_sum = float(bids[bids["price"] >= mid * (1 - pct)]["size"].sum())
        ask_sum = float(asks[asks["price"] <= mid * (1 + pct)]["size"].sum())
        out[label] = (bid_sum, ask_sum)
    return out


def _render_book_summary(books_by_venue: dict) -> None:
    st.markdown(section_label("Top of Book"), unsafe_allow_html=True)
    rows = []
    for venue, book in books_by_venue.items():
        if book is None or book.empty:
            continue
        tob = top_of_book(book)
        if tob.empty:
            continue
        r = tob.iloc[0]
        d = _depth_buckets(book)
        rows.append({
            "Venue": venue,
            "Best bid": f"${r['best_bid']:,.2f}",
            "Best ask": f"${r['best_ask']:,.2f}",
            "Mid": f"${r['mid']:,.2f}",
            "Spread (bps)": f"{r['spread_bps']:.2f}",
            "Bid ±0.5%": f"{d['d05'][0]:,.1f}",
            "Ask ±0.5%": f"{d['d05'][1]:,.1f}",
            "Bid ±1.0%": f"{d['d10'][0]:,.1f}",
            "Ask ±1.0%": f"{d['d10'][1]:,.1f}",
            "Source": str(r.get("_source", "—")),
        })
    if not rows:
        st.caption("No order books available.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="Liquidity · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Liquidity",
               "L2 depth · top-of-book · spread comparison across venues")

    sb = st.sidebar
    sb.markdown("### Venues")
    show_binance_spot = sb.checkbox("Binance spot", value=True)
    show_binance_perp = sb.checkbox("Binance perp", value=True)
    show_deribit = sb.checkbox("Deribit perp", value=True)
    show_coinbase = sb.checkbox("Coinbase spot", value=False)

    selected = []
    if show_binance_spot:
        selected.append("binance_spot")
    if show_binance_perp:
        selected.append("binance_perp")
    if show_deribit:
        selected.append("deribit")
    if show_coinbase:
        selected.append("coinbase")

    books = {v: _book(v) for v in selected}
    _render_book_summary(books)

    st.markdown(section_label("Cumulative Depth"), unsafe_allow_html=True)
    cols = st.columns(2, gap="medium")
    for i, (venue, book) in enumerate(books.items()):
        with cols[i % 2]:
            _depth_chart(book, venue)


main()
