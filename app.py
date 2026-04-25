"""
btc_dashboard · app.py
Landing page — Deribit-driven options-distribution overview, per-expiry.

A sidebar dropdown lists every BTC expiry currently on Deribit. All four
charts recompute against the selected expiry only:

  1. RN PDF (BL) overlaid with OI-adjusted RN PDF + means & P(above spot).
  2. Dealer GEX per strike (filtered to the selected expiry).
  3. Time series — RN-mean − spot gap in absolute USD (RN + OI-adjusted),
     keyed by expiry.
  4. Time series — P(above spot), RN + OI-adjusted, keyed by expiry.

Time-series charts build their history per expiry from `st.session_state`
across page refreshes. Two snapshots minimum to draw a line.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.gex import GEXResult, compute_gex
from analytics.pipeline import PipelineResult, run_pipeline
from analytics.rn_pdf import compute_oi_adjusted_pdf, compute_rn_pdf
from charts.theme import (
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, page_title, section_label,
)
from config import settings
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
# HISTORY BUFFER  (per-session, keyed by expiry)
# ─────────────────────────────────────────────────────────────────────────────
_HIST_KEY = "landing_history_by_expiry"
_HIST_MAX = 200


def _expiry_key(expiry) -> str:
    try:
        return pd.Timestamp(expiry).isoformat()
    except Exception:
        return str(expiry)


def _record_snapshot(expiry, spot: float,
                     rn: Optional[dict], rn_oi: Optional[dict]) -> None:
    if spot <= 0 or (rn is None and rn_oi is None):
        return
    key = _expiry_key(expiry)
    store: dict = st.session_state.setdefault(_HIST_KEY, {})
    buf: list[dict] = store.setdefault(key, [])

    rn_mean = float(rn["mean"]) if rn is not None else None
    rn_oi_mean = float(rn_oi["mean"]) if rn_oi is not None else None
    p_above = float(rn["p_above_spot"]) if rn is not None else None
    p_above_oi = (
        float(rn_oi["p_above_spot"]) if rn_oi is not None else None
    )

    new = {
        "ts": datetime.now(timezone.utc),
        "spot": float(spot),
        "rn_mean": rn_mean,
        "rn_oi_mean": rn_oi_mean,
        "p_above": p_above,
        "p_above_oi": p_above_oi,
        "gap_abs": (rn_mean - spot) if rn_mean is not None else None,
        "gap_oi_abs": (rn_oi_mean - spot) if rn_oi_mean is not None else None,
    }

    if buf:
        last = buf[-1]
        same_spot = abs(last.get("spot", 0.0) - new["spot"]) < 1e-3
        same_rn = (last.get("rn_mean") or 0.0) == (new["rn_mean"] or 0.0)
        same_oi = (last.get("rn_oi_mean") or 0.0) == (new["rn_oi_mean"] or 0.0)
        if same_spot and same_rn and same_oi:
            return

    buf.append(new)
    if len(buf) > _HIST_MAX:
        buf = buf[-_HIST_MAX:]
    store[key] = buf
    st.session_state[_HIST_KEY] = store


def _history_df(expiry) -> pd.DataFrame:
    store = st.session_state.get(_HIST_KEY, {})
    buf = store.get(_expiry_key(expiry), [])
    if not buf:
        return pd.DataFrame()
    df = pd.DataFrame(buf)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values("ts").reset_index(drop=True)


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
        xaxis=dict(title="BTC at expiry ($)", gridcolor="#EDEBE6"),
        yaxis=dict(title="Probability density", gridcolor="#EDEBE6"),
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
        xaxis=dict(title="Strike ($)", gridcolor="#EDEBE6"),
        yaxis=dict(title="Per-strike GEX (B$/1%)", gridcolor="#EDEBE6"),
        yaxis2=dict(title="Cumulative GEX (B$/1%)", overlaying="y", side="right",
                    showgrid=False, tickfont=dict(color=GOLD)),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_gap_history(hist: pd.DataFrame, expiry_label: str) -> None:
    if hist.empty:
        st.caption(f"RN-mean gap history will populate after the next refresh ({expiry_label}).")
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
            title=f"{expiry_label}  ·  RN mean − spot gap (USD)  ·  {len(hist)} snapshots",
            height=320,
        ),
        xaxis=dict(title=None, gridcolor="#EDEBE6"),
        yaxis=dict(title="Gap ($)", gridcolor="#EDEBE6"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _chart_p_above_history(hist: pd.DataFrame, expiry_label: str) -> None:
    if hist.empty:
        st.caption(f"P(above spot) history will populate after the next refresh ({expiry_label}).")
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
            title=f"{expiry_label}  ·  P(above spot)  ·  {len(hist)} snapshots",
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
def _sidebar(spot: float, atm_iv: Optional[float], chain: pd.DataFrame,
             selected_expiry, expiry_key: str) -> None:
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

    sb.markdown("---")
    sb.markdown("### History (per expiry)")
    store = st.session_state.get(_HIST_KEY, {})
    n = len(store.get(expiry_key, []))
    sb.caption(f"{n} snapshots stored for selected expiry")
    if sb.button("Clear history (this expiry)", use_container_width=True):
        store[expiry_key] = []
        st.session_state[_HIST_KEY] = store
        st.rerun()
    if sb.button("Clear ALL expiries", use_container_width=True):
        st.session_state[_HIST_KEY] = {}
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
            help="All four charts below recompute against the selected Deribit expiry.",
        )
    expiry_label = _format_expiry_label(selected, chain)
    expiry_key = _expiry_key(selected)

    # Sidebar
    _sidebar(spot, atm_iv, chain, selected, expiry_key)

    # ── Compute per-expiry curves ────────────────────────────────────────
    chain_sel = _filter_chain(chain, selected)
    rn, rn_oi, gex = _compute_for_expiry(chain_sel, spot)

    # Persist a snapshot for the selected expiry's history series
    _record_snapshot(selected, spot, rn, rn_oi)
    hist = _history_df(selected)

    # Per-expiry summary metrics
    n_contracts = int(len(chain_sel))
    cols2 = st.columns(4)
    cols2[0].metric("Spot", fmt_money(spot))
    cols2[1].metric("Chain rows", f"{n_contracts}")
    cols2[2].metric("RN mean", fmt_money(rn["mean"]) if rn is not None else "—")
    cols2[3].metric(
        "OI-adj mean",
        fmt_money(rn_oi["mean"]) if rn_oi is not None else "—",
    )

    # 1) RN PDF + OI-adjusted overlay
    st.markdown(section_label("Risk-Neutral Distribution"), unsafe_allow_html=True)
    _chart_rn_distributions(rn, rn_oi, spot, expiry_label)

    # 2) GEX per strike
    st.markdown(section_label("Dealer GEX by Strike"), unsafe_allow_html=True)
    _chart_gex(gex, spot, expiry_label)

    # 3) RN-mean − spot gap history
    st.markdown(section_label("RN-Mean − Spot Gap (USD, over time)"),
                unsafe_allow_html=True)
    _chart_gap_history(hist, expiry_label)

    # 4) P(above spot) history
    st.markdown(section_label("P(above spot) History"), unsafe_allow_html=True)
    _chart_p_above_history(hist, expiry_label)

    st.caption(
        "All four charts are filtered to the selected Deribit expiry. The "
        "OI-adjusted curve tilts the BL density by the smoothed open-interest "
        "profile — a positioning view, not a true risk-neutral density. "
        "Time-series charts build per-expiry history from this browser session — "
        "two refreshes minimum to draw a line. Switch expiry to see its own series."
    )


main()
