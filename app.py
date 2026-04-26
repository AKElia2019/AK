"""
btc_dashboard · app.py
Landing page — Deribit-driven options view + spot / volume context.

A sidebar dropdown lists every BTC expiry currently on Deribit. The
options charts recompute against the selected expiry only:

  1. RN PDF (BL) overlaid with OI-adjusted RN PDF + means & P(above spot).
  2. Dealer GEX per strike (filtered to the selected expiry).
  3. Spot price candles + SMA stack + VWAP + volume bars (perp · 1h · 200 bars).
  4. Volume profile (1h · 30 buckets).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from analytics.gex import GEXResult, compute_gex
from analytics.pipeline import PipelineResult, run_pipeline
from analytics.rn_pdf import compute_oi_adjusted_pdf, compute_rn_pdf
from charts.theme import (
    GOLD, TEAL, RED, AMBER, STONE, INK,
    BULL, BEAR, BRAND, BRASS, GRID,
    base_layout, fmt_money, inject_global_css, page_title, section_label,
)
from config import settings
from data.futures import fetch_binance_perp_klines
from utils.logger import get_logger


log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE  (cached 30s) — used only to fetch the full chain + spot
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Loading Deribit chain…")
def _pipe() -> dict:
    return {"result": run_pipeline()}


# ─────────────────────────────────────────────────────────────────────────────
# PER-EXPIRY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────
def _available_expiries(chain: Optional[pd.DataFrame]) -> list:
    if chain is None or chain.empty or "expiry" not in chain.columns:
        return []
    raw = chain["expiry"].dropna().unique().tolist()
    try:
        return sorted(raw, key=lambda x: pd.Timestamp(x))
    except Exception:
        return raw


def _format_expiry_label(expiry, chain: pd.DataFrame) -> str:
    """Format an expiry value for the dropdown: '02 May 2026 · 7d'."""
    try:
        ts = pd.Timestamp(expiry)
        date_part = ts.strftime("%d %b %Y")
    except Exception:
        date_part = str(expiry)

    if chain is not None and "dte" in chain.columns:
        sub = chain[chain["expiry"] == expiry]
        if not sub.empty:
            dte = float(sub["dte"].median())
            return f"{date_part} · {dte:.0f}d"
    return date_part


def _filter_chain(chain: pd.DataFrame, expiry) -> pd.DataFrame:
    if chain is None or chain.empty or "expiry" not in chain.columns:
        return pd.DataFrame()
    return chain[chain["expiry"] == expiry].copy()


def _btc_marks_to_usd(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Deribit option marks are quoted in BTC; multiply by spot to get USD
    so the BL fit returns a USD-denominated density."""
    if chain is None or chain.empty:
        return chain
    out = chain.copy()
    if "venue" in out.columns and "mark" in out.columns:
        is_deribit = out["venue"].astype(str).str.lower() == "deribit"
        out.loc[is_deribit, "mark"] = (
            pd.to_numeric(out.loc[is_deribit, "mark"], errors="coerce") * float(spot)
        )
    return out


def _compute_for_expiry(chain_for_expiry: pd.DataFrame, spot: float):
    """Return (rn_curve, rn_oi_curve, gex) computed from the filtered chain.
    Each may be None if the slice is too thin / scipy missing / etc."""
    if chain_for_expiry is None or chain_for_expiry.empty or spot <= 0:
        return None, None, None

    bl_input = _btc_marks_to_usd(chain_for_expiry, spot)
    rn = compute_rn_pdf(bl_input, spot)
    rn_oi = compute_oi_adjusted_pdf(bl_input, rn, spot) if rn is not None else None
    gex = compute_gex(chain_for_expiry, spot)
    return rn, rn_oi, gex


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _chart_rn_distributions(rn: Optional[dict], rn_oi: Optional[dict],
                            spot: float, expiry_label: str) -> None:
    if rn is None and rn_oi is None:
        st.caption(f"Risk-neutral density unavailable for {expiry_label} "
                   "— chain too thin or scipy missing.")
        return

    fig = go.Figure()

    if rn is not None:
        K = np.asarray(rn["K"], dtype=float)
        pdf = np.asarray(rn["pdf"], dtype=float)
        if spot > 0:
            for mask, color in (
                (K <= spot, "rgba(163,90,72,0.10)"),    # terracotta · downside
                (K >= spot, "rgba(107,139,104,0.12)"),  # sage · upside
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

    if rn_oi is not None:
        K2 = np.asarray(rn_oi["K"], dtype=float)
        pdf2 = np.asarray(rn_oi["pdf"], dtype=float)
        fig.add_trace(go.Scatter(
            x=K2, y=pdf2, mode="lines",
            name="OI-adjusted RN density",
            line=dict(color=TEAL, width=2.0, dash="dash"),
            hovertemplate="$%{x:,.0f}<br>density %{y:.6f}<extra>OI-adj</extra>",
        ))

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

    p_above = rn["p_above_spot"] if rn is not None else None
    p_above_oi = rn_oi["p_above_spot"] if rn_oi is not None else None
    bits = []
    if p_above is not None:
        bits.append(f"P(above spot) RN {p_above*100:.1f}%")
    if p_above_oi is not None:
        bits.append(f"OI-adj {p_above_oi*100:.1f}%")
    title = f"{expiry_label}  ·  " + (" · ".join(bits) if bits else "Risk-Neutral Distribution")

    fig.update_layout(
        **base_layout(title=title, height=420),
        xaxis=dict(title="BTC at expiry ($)", gridcolor=GRID),
        yaxis=dict(title="Probability density", gridcolor=GRID),
        legend=dict(orientation="h", yanchor="top", y=-0.14, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_gex(gex: Optional[GEXResult], spot: float, expiry_label: str) -> None:
    if gex is None or gex.by_strike is None or gex.by_strike.empty:
        st.caption(f"Dealer GEX unavailable for {expiry_label}.")
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

    if spot:
        fig.add_vline(x=float(spot), line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
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
            title=(f"{expiry_label}  ·  Dealer GEX  ·  total {total_b:+.2f} B$/1%  ·  "
                   f"{regime}  ·  {gex.n_options} contracts"),
            height=380,
        ),
        xaxis=dict(title="Strike ($)", gridcolor=GRID),
        yaxis=dict(title="Per-strike GEX (B$/1%)", gridcolor=GRID),
        yaxis2=dict(title="Cumulative GEX (B$/1%)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color=GOLD)),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SPOT / VOLUME  (perp klines)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _klines(interval: str, limit: int) -> pd.DataFrame:
    return fetch_binance_perp_klines(interval=interval, limit=limit)


def _expiry_means(chain: Optional[pd.DataFrame], spot: float) -> list[dict]:
    """Compute RN-mean and OI-adjusted RN-mean for every upcoming Deribit
    expiry. Returns a list of dicts sorted by ascending DTE."""
    if chain is None or chain.empty or "expiry" not in chain.columns or spot <= 0:
        return []
    out: list[dict] = []
    for exp in _available_expiries(chain):
        try:
            ts = pd.Timestamp(exp)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            continue
        sub = _filter_chain(chain, exp)
        if sub.empty:
            continue
        if "dte" in sub.columns and sub["dte"].notna().any():
            dte = float(sub["dte"].median())
        else:
            dte = (ts - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0
        if dte <= 0:
            continue
        rn, rn_oi, _gex = _compute_for_expiry(sub, spot)
        if rn is None and rn_oi is None:
            continue
        out.append({
            "expiry": ts,
            "dte": dte,
            "rn_mean": float(rn["mean"]) if rn is not None else None,
            "oi_mean": float(rn_oi["mean"]) if rn_oi is not None else None,
        })
    out.sort(key=lambda r: r["dte"])
    return out


def _chart_spot_volume(chain: Optional[pd.DataFrame], spot: float,
                       interval: str = "1h", limit: int = 200,
                       sma_short: int = 20, sma_mid: int = 50,
                       sma_long: int = 200, vwap_window: int = 24) -> None:
    df = _klines(interval, limit).copy()
    if df.empty or len(df) < 5:
        st.caption(f"No spot / volume data for {interval}.")
        return

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    opens = df["open"].astype(float)
    vols = df["volume"].astype(float).replace(0.0, np.nan)
    times = pd.to_datetime(df["time"], utc=True)

    typ = (highs + lows + closes) / 3.0
    pv = typ * vols
    vwap = (pv.rolling(vwap_window, min_periods=vwap_window).sum()
            / vols.rolling(vwap_window, min_periods=vwap_window).sum())

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.72, 0.28], vertical_spacing=0.04,
    )

    fig.add_trace(go.Candlestick(
        x=times, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color=BULL, decreasing_line_color=BEAR,
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
    fig.add_trace(go.Scatter(x=times, y=vwap, mode="lines",
                             name=f"VWAP({vwap_window})",
                             line=dict(color=TEAL, width=1.6, dash="dot")),
                  row=1, col=1)

    bar_colors = [BULL if c >= o else BEAR for c, o in zip(closes, opens)]
    fig.add_trace(go.Bar(x=times, y=df["volume"].astype(float),
                         marker=dict(color=bar_colors, line=dict(width=0)),
                         showlegend=False, name="Volume"),
                  row=2, col=1)

    # ── Forward fan: RN mean + OI-adjusted mean for every upcoming expiry ─
    means = _expiry_means(chain, spot)
    t_now = times.iloc[-1]
    t_max_axis = times.iloc[-1]
    if means:
        n = len(means)
        for i, row_d in enumerate(means):
            # Opacity: 1.0 (soonest) → 0.28 (furthest)
            alpha = 1.0 - 0.72 * (i / max(n - 1, 1))
            navy_c = f"rgba(31,58,95,{alpha:.2f})"
            steel_c = f"rgba(74,111,165,{alpha:.2f})"
            t_exp = row_d["expiry"]
            label = f"{int(round(row_d['dte']))}d"

            if row_d["rn_mean"] is not None:
                fig.add_trace(go.Scatter(
                    x=[t_now, t_exp], y=[row_d["rn_mean"], row_d["rn_mean"]],
                    mode="lines+markers+text",
                    line=dict(color=navy_c, width=1.4, dash="dot"),
                    marker=dict(size=[0, 7], color=navy_c, symbol="diamond"),
                    text=["", f"  {label}"],
                    textfont=dict(color=navy_c, size=9,
                                  family="JetBrains Mono"),
                    textposition="middle right",
                    showlegend=(i == 0),
                    name="RN mean · forward expiries",
                    hovertemplate=(f"Expiry {t_exp:%d %b %Y}<br>"
                                   f"RN mean ${row_d['rn_mean']:,.0f}"
                                   f"<extra>{label}</extra>"),
                ), row=1, col=1)

            if row_d["oi_mean"] is not None:
                fig.add_trace(go.Scatter(
                    x=[t_now, t_exp], y=[row_d["oi_mean"], row_d["oi_mean"]],
                    mode="lines+markers",
                    line=dict(color=steel_c, width=1.2, dash="dash"),
                    marker=dict(size=[0, 6], color=steel_c,
                                symbol="circle-open"),
                    showlegend=(i == 0),
                    name="OI-adj mean · forward expiries",
                    hovertemplate=(f"Expiry {t_exp:%d %b %Y}<br>"
                                   f"OI-adj mean ${row_d['oi_mean']:,.0f}"
                                   f"<extra>{label}</extra>"),
                ), row=1, col=1)
        t_max_axis = max(row_d["expiry"] for row_d in means)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color=INK, size=11),
        height=560, margin=dict(l=58, r=20, t=30, b=44),
        title=dict(
            text=f"BTC perp · {interval} · last {len(df)} bars · forward RN-mean fan",
            font=dict(size=12, color=INK), x=0.0, xanchor="left",
        ),
        legend=dict(orientation="h", yanchor="top", y=1.06,
                    xanchor="left", x=0),
        xaxis=dict(rangeslider=dict(visible=False), gridcolor=GRID,
                   range=[times.iloc[0], t_max_axis]),
        xaxis2=dict(gridcolor=GRID, title=None,
                    range=[times.iloc[0], t_max_axis]),
        yaxis=dict(gridcolor=GRID, title="Price ($)"),
        yaxis2=dict(gridcolor=GRID, title="Volume"),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


def _chart_volume_profile(interval: str = "1h", limit: int = 200,
                          buckets: int = 30) -> None:
    df = _klines(interval, limit)
    if df.empty:
        st.caption("Volume profile unavailable.")
        return
    typical = (df["high"].astype(float)
               + df["low"].astype(float)
               + df["close"].astype(float)) / 3.0
    bins = np.linspace(typical.min(), typical.max(), buckets + 1)
    df_b = pd.DataFrame({"price": typical,
                         "volume": df["volume"].astype(float)})
    df_b["bucket"] = pd.cut(df_b["price"], bins=bins, include_lowest=True)
    profile = (df_b.groupby("bucket", observed=True)["volume"]
               .sum().reset_index())
    profile["mid"] = profile["bucket"].apply(
        lambda b: (b.left + b.right) / 2 if hasattr(b, "left") else 0
    )

    fig = go.Figure(go.Bar(
        x=profile["volume"], y=profile["mid"], orientation="h",
        marker=dict(color=GOLD, opacity=0.7, line=dict(width=0)),
        hovertemplate="$%{y:,.0f}<br>volume %{x:,.0f}<extra></extra>",
        showlegend=False,
    ))
    last_close = float(df["close"].astype(float).iloc[-1])
    fig.add_hline(y=last_close, line=dict(color=INK, width=1.4, dash="dot"),
                  annotation_text=f" Spot ${last_close:,.0f}",
                  annotation_font=dict(color=INK, size=10))
    fig.update_layout(
        **base_layout(title=f"Volume profile · {interval} · {buckets} buckets",
                      height=420),
        xaxis=dict(title="Volume (BTC)", gridcolor=GRID),
        yaxis=dict(title="Price ($)", gridcolor=GRID),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar(spot: float, atm_iv: Optional[float]) -> None:
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
    sb.caption(f"Spot: {fmt_money(spot)}")
    sb.caption(f"ATM IV: {atm_iv:.1f}%" if atm_iv is not None else "ATM IV: —")


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
    inject_global_css()

    log.info("Starting %s in %s mode", settings.app_title, settings.environment)

    page_title(
        settings.app_title,
        "Deribit options · per-expiry RN distribution · GEX · drift over time",
    )

    try:
        res: PipelineResult = _pipe()["result"]
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    chain = res.raw.get("chain") if isinstance(res.raw, dict) else None
    spot = float(res.spot or 0.0)
    atm_iv = res.atm_iv_pct

    # ── Expiry dropdown (top of main pane, prominent) ─────────────────────
    expiries = _available_expiries(chain)
    if not expiries:
        st.error(
            "No Deribit expiries available — the chain returned empty / mock. "
            "Check that the deployment can reach Deribit."
        )
        return

    cols = st.columns([2, 5])
    with cols[0]:
        selected = st.selectbox(
            "Expiry",
            expiries,
            index=0,
            format_func=lambda e: _format_expiry_label(e, chain),
            key="landing_expiry",
            help="The two options charts below recompute against the selected Deribit expiry.",
        )
    expiry_label = _format_expiry_label(selected, chain)

    # Sidebar
    _sidebar(spot, atm_iv)

    # ── Compute per-expiry curves ────────────────────────────────────────
    chain_sel = _filter_chain(chain, selected)
    rn, rn_oi, gex = _compute_for_expiry(chain_sel, spot)

    # Per-expiry summary metrics
    cols2 = st.columns(3)
    cols2[0].metric("Spot", fmt_money(spot))
    cols2[1].metric(
        "RN mean",
        fmt_money(rn["mean"]) if rn is not None else "—",
        f"P(above spot) {rn['p_above_spot']*100:.1f}%" if rn is not None else None,
        delta_color="off",
    )
    cols2[2].metric(
        "OI-adj mean",
        fmt_money(rn_oi["mean"]) if rn_oi is not None else "—",
        f"P(above spot) {rn_oi['p_above_spot']*100:.1f}%" if rn_oi is not None else None,
        delta_color="off",
    )

    # 1) RN PDF + OI-adjusted overlay
    st.markdown(section_label("Risk-Neutral Distribution"), unsafe_allow_html=True)
    _chart_rn_distributions(rn, rn_oi, spot, expiry_label)

    # 2) GEX per strike
    st.markdown(section_label("Dealer GEX by Strike"), unsafe_allow_html=True)
    _chart_gex(gex, spot, expiry_label)

    # 3) Spot · SMA · VWAP · Volume + forward RN-mean fan
    st.markdown(section_label("Price + SMA + VWAP + Volume"),
                unsafe_allow_html=True)
    _chart_spot_volume(chain=chain, spot=spot,
                       interval="1h", limit=200, vwap_window=24)

    # 4) Volume profile
    st.markdown(section_label("Volume Profile"), unsafe_allow_html=True)
    _chart_volume_profile(interval="1h", limit=200, buckets=30)

    st.caption(
        "Options charts are filtered to the selected Deribit expiry. The "
        "OI-adjusted curve tilts the BL density by the smoothed open-interest "
        "profile — a positioning view, not a true risk-neutral density. "
        "Spot chart shows BTC perp · 1h · last 200 bars; the forward overlay "
        "draws RN mean (solid · diamond) and OI-adjusted mean (dashed · "
        "open circle) for every upcoming Deribit expiry — darker shade = "
        "nearer expiry, lighter = further out."
    )


main()
