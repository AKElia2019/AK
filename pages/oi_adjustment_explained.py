"""
btc_dashboard.pages.oi_adjustment_explained
Step-by-step walk-through of how the OI-adjusted RN density is computed.

Each of the 8 steps shows:
  1) the formula
  2) the result (a number, a table, or both)
  3) a Plotly chart that visualises that specific step

Sidebar lets you change `oi_strength` and `oi_smooth` and watch every
step react in place.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from analytics.pipeline import PipelineResult, run_pipeline   # noqa: E402
from analytics.rn_pdf import compute_rn_pdf                  # noqa: E402
from charts.theme import (                                    # noqa: E402
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, inject_global_css, page_title, section_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Loading Deribit chain…")
def _pipe() -> dict:
    return {"result": run_pipeline()}


def _format_expiry(e, chain: pd.DataFrame) -> str:
    try:
        date_part = pd.Timestamp(e).strftime("%d %b %Y")
    except Exception:
        date_part = str(e)
    if chain is not None and "dte" in chain.columns:
        sub = chain[chain["expiry"] == e]
        if not sub.empty:
            return f"{date_part} · {float(sub['dte'].median()):.0f}d"
    return date_part


def _trapz(y, x):
    return np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _line_chart(x, y, title, xlabel, ylabel,
                color=GOLD, dash=None, height=300,
                spot: Optional[float] = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2.2, dash=dash),
        hovertemplate="$%{x:,.0f}<br>%{y:.6f}<extra></extra>",
        showlegend=False,
    ))
    if spot is not None:
        fig.add_vline(x=float(spot), line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
                      annotation_font=dict(color=INK, size=10))
    fig.update_layout(
        **base_layout(title=title, height=height),
        xaxis=dict(title=xlabel, gridcolor="#E5DCC9"),
        yaxis=dict(title=ylabel, gridcolor="#E5DCC9"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# THE STEP-BY-STEP RECOMPUTATION
# Mirrors `analytics.rn_pdf.compute_oi_adjusted_pdf` exactly, but exposes
# every intermediate vector so each step can be charted.
# ─────────────────────────────────────────────────────────────────────────────
def _btc_marks_to_usd(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """Deribit option marks are quoted in BTC; convert to USD before BL."""
    if chain is None or chain.empty:
        return chain
    out = chain.copy()
    if "venue" in out.columns and "mark" in out.columns:
        is_deribit = out["venue"].astype(str).str.lower() == "deribit"
        out.loc[is_deribit, "mark"] = (
            pd.to_numeric(out.loc[is_deribit, "mark"], errors="coerce") * float(spot)
        )
    return out


def _step_aggregate_oi(chain: pd.DataFrame) -> Optional[pd.Series]:
    if chain is None or chain.empty or "open_interest" not in chain.columns:
        return None
    work = chain.copy()
    work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
    work["open_interest"] = pd.to_numeric(work["open_interest"], errors="coerce")
    work = work.dropna(subset=["strike"])
    if work.empty:
        return None
    s = (
        work.groupby("strike")["open_interest"]
        .sum(min_count=1)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .sort_index()
    )
    return s if float(s.sum()) > 0 else None


def _step_smooth_resample(oi_by_strike: pd.Series, K: np.ndarray,
                          oi_smooth: float) -> Optional[np.ndarray]:
    if not _HAS_SCIPY:
        return None
    strikes = oi_by_strike.index.to_numpy(dtype=float)
    vals = oi_by_strike.to_numpy(dtype=float)
    sigma = oi_smooth * max(1.0, 10.0 / max(len(strikes), 1))
    smoothed = gaussian_filter1d(vals, sigma=sigma)
    return np.interp(K, strikes, smoothed, left=0.0, right=0.0)


def _step_zscore(oi_on_K: np.ndarray) -> np.ndarray:
    mu = float(np.mean(oi_on_K))
    sd = float(np.std(oi_on_K))
    if sd <= 1e-12:
        return np.zeros_like(oi_on_K)
    return np.clip((oi_on_K - mu) / sd, -3.0, 3.0)


def _step_tilt(z: np.ndarray, oi_strength: float) -> np.ndarray:
    return np.maximum(0.05, 1.0 + oi_strength * z)


def _step_apply_and_normalise(base_pdf: np.ndarray, tilt: np.ndarray,
                              K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (un-normalised tilted density, normalised pdf, area).
    Caller can show both pre- and post-normalisation curves."""
    raw = np.maximum(base_pdf * tilt, 0.0)
    area = float(_trapz(raw, K))
    if area <= 0:
        return raw, np.zeros_like(raw), 0.0
    return raw, raw / area, area


def _step_moments(K: np.ndarray, pdf: np.ndarray, spot: float) -> dict:
    mean = float(_trapz(K * pdf, K))
    var = float(_trapz((K - mean) ** 2 * pdf, K))
    std = math.sqrt(max(var, 0.0))
    dK = float(K[1] - K[0])
    cdf = np.cumsum(pdf) * dK
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    if spot < K[0]:
        p_above = 1.0
    elif spot > K[-1]:
        p_above = 0.0
    else:
        idx = int(np.searchsorted(K, float(spot)))
        idx = max(0, min(idx, len(cdf) - 1))
        p_above = float(1.0 - cdf[idx])
    return {"mean": mean, "std": std, "p_above": p_above, "cdf": cdf}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="OI-Adjusted RN · Step by Step",
                       page_icon="₿", layout="wide")
    inject_global_css()
    page_title(
        "OI-Adjusted RN — Step by Step",
        "How the BL density gets tilted by open interest, one step at a time",
    )

    if not _HAS_SCIPY:
        st.error("scipy is unavailable — install scipy and reload.")
        return

    # ── Sidebar — expiry + tunable knobs
    sb = st.sidebar
    sb.markdown("### Inputs")

    try:
        res: PipelineResult = _pipe()["result"]
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return

    chain = res.raw.get("chain") if isinstance(res.raw, dict) else None
    spot = float(res.spot or 0.0)

    if chain is None or chain.empty or "expiry" not in chain.columns:
        st.error("No chain available — Deribit feed empty.")
        return

    expiries = sorted(chain["expiry"].dropna().unique().tolist(),
                       key=lambda x: pd.Timestamp(x))
    selected = sb.selectbox(
        "Expiry", expiries, index=0,
        format_func=lambda e: _format_expiry(e, chain),
    )
    sb.markdown("---")
    sb.markdown("### Tilt knobs")
    oi_strength = sb.slider(
        "OI strength λ", 0.0, 1.50, 0.45, 0.05,
        help="Multiplicative tilt strength. 0 = no tilt; 1.5 = aggressive.",
    )
    oi_smooth = sb.slider(
        "Smoothing σ", 0.5, 5.0, 1.5, 0.5,
        help="Gaussian-filter σ on the OI vector before z-scoring.",
    )

    # ── Filter chain to the selected expiry
    sub = chain[chain["expiry"] == selected].copy()
    if sub.empty:
        st.error(f"No rows for expiry {selected}.")
        return

    # ── Step 0 — recompute the BASE BL fit so we have K and base pdf
    bl = compute_rn_pdf(_btc_marks_to_usd(sub, spot), spot)
    if bl is None:
        st.error(
            "Base Breeden-Litzenberger fit failed for this expiry — "
            "need at least 4 call strikes with valid marks."
        )
        return
    K = np.asarray(bl["K"], dtype=float)
    base_pdf = np.asarray(bl["pdf"], dtype=float)

    expiry_label = _format_expiry(selected, chain)
    st.caption(
        f"Selected expiry: **{expiry_label}**  ·  "
        f"spot **${spot:,.0f}**  ·  base BL strikes used: **{bl['n']}**"
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1 — Inputs
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 1 · Inputs"), unsafe_allow_html=True)
    st.markdown(
        """
- **Base RN density** `f_RN(K)` from Breeden-Litzenberger (gold curve below).
- **Strike grid** `K` — the same 500-point grid the BL fit produced.
- **Open interest per strike** — calls + puts summed by strike.
- **Spot** for the P(above spot) calculation only.
- Two knobs: `oi_strength = λ` and `oi_smooth = σ`.
        """
    )
    st.plotly_chart(
        _line_chart(K, base_pdf,
                    title="Base BL risk-neutral density f_RN(K)",
                    xlabel="Strike K ($)", ylabel="Density",
                    color=GOLD, height=280, spot=spot),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2 — Aggregate OI by strike
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 2 · Aggregate OI by strike"),
                unsafe_allow_html=True)
    st.markdown(
        "$$\\text{oi\\_by\\_strike}[K_i] = \\sum_{\\text{contracts at } K_i} \\text{OI}$$  "
        "Calls and puts are summed together — both contribute to dealer-side gamma."
    )
    oi_by_strike = _step_aggregate_oi(sub)
    if oi_by_strike is None or oi_by_strike.empty:
        st.error("No open interest in this expiry.")
        return
    cols = st.columns([2, 3])
    with cols[0]:
        st.metric("Strikes with OI", f"{len(oi_by_strike)}")
        st.metric("Total OI (contracts)", f"{float(oi_by_strike.sum()):,.0f}")
        st.dataframe(
            oi_by_strike.rename("open_interest").reset_index().head(20),
            use_container_width=True, hide_index=True,
        )
    with cols[1]:
        fig = go.Figure(go.Bar(
            x=oi_by_strike.index, y=oi_by_strike.values,
            marker=dict(color=AMBER, line=dict(width=0)),
            hovertemplate="Strike $%{x:,.0f}<br>OI %{y:,.0f}<extra></extra>",
        ))
        if spot > 0:
            fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"),
                          annotation_text=f" Spot ${spot:,.0f}",
                          annotation_font=dict(color=INK, size=10))
        fig.update_layout(
            **base_layout(title="Raw OI per strike (calls + puts)", height=320),
            xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
            yaxis=dict(title="Open interest (contracts)", gridcolor="#E5DCC9"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3 — Smooth and resample
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 3 · Smooth + resample onto the BL grid"),
                unsafe_allow_html=True)
    sigma = oi_smooth * max(1.0, 10.0 / max(len(oi_by_strike), 1))
    st.markdown(
        f"""
$$\\text{{oi\\_sigma}} = \\sigma \\cdot \\max(1,\\;10/n)
= {oi_smooth:.2f} \\cdot \\max(1, 10/{len(oi_by_strike)}) = {sigma:.3f}$$

Gaussian-smooth the discrete OI vector with that effective σ, then linearly
resample onto the BL strike grid `K` (so every subsequent step is on the
same 500-point grid).
        """
    )
    oi_on_K = _step_smooth_resample(oi_by_strike, K, oi_smooth)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=oi_by_strike.index, y=oi_by_strike.values,
        marker=dict(color=AMBER, opacity=0.45, line=dict(width=0)),
        name="Raw OI per strike",
        hovertemplate="$%{x:,.0f}<br>OI %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=K, y=oi_on_K, mode="lines",
        line=dict(color=AMBER, width=2.4),
        name="Smoothed + resampled OI",
        hovertemplate="$%{x:,.0f}<br>%{y:,.1f}<extra></extra>",
    ))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
                      annotation_font=dict(color=INK, size=10))
    fig.update_layout(
        **base_layout(title="Raw vs smoothed OI on the BL grid", height=320),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="OI (contracts)", gridcolor="#E5DCC9"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4 — Z-score
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 4 · Z-score"), unsafe_allow_html=True)
    mu = float(np.mean(oi_on_K))
    sd = float(np.std(oi_on_K))
    z = _step_zscore(oi_on_K)
    st.markdown(
        f"""
$$z(K) = \\mathrm{{clip}}\\!\\left(\\frac{{\\text{{oi\\_on\\_K}}(K)-\\mu}}{{\\sigma}},\\; -3,\\; +3\\right)$$

For this expiry: μ = **{mu:,.1f}**, σ = **{sd:,.1f}** (clipping at ±3 caps any
single dominant cluster).
        """
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K, y=z, mode="lines",
        line=dict(color=TEAL, width=2),
        fill="tozeroy", fillcolor="rgba(26,122,107,0.15)",
        hovertemplate="$%{x:,.0f}<br>z %{y:+.2f}<extra></extra>",
        showlegend=False,
    ))
    for level, label, c in ((1, "+1σ crowded", TEAL),
                             (-1, "-1σ thin", RED),
                             (0, "average OI", STONE)):
        fig.add_hline(y=level, line=dict(color=c, width=1, dash="dot"),
                      annotation_text=f" {label}",
                      annotation_font=dict(color=c, size=9))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"))
    fig.update_layout(
        **base_layout(title="z(K) — how crowded each strike is", height=300),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="z-score (clipped to ±3)",
                   range=[-3.2, 3.2], gridcolor="#E5DCC9"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5 — Tilt vector
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 5 · Build the multiplicative tilt"),
                unsafe_allow_html=True)
    tilt = _step_tilt(z, oi_strength)
    st.markdown(
        f"""
$$\\text{{tilt}}(K) = \\max\\!\\left(0.05,\\; 1 + \\lambda \\cdot z(K)\\right)
\\quad\\text{{with }}\\lambda = {oi_strength:.2f}$$

A strike with average OI keeps tilt ≈ 1. A maximally crowded strike (z ≈ +3)
gets tilt ≈ **{1 + oi_strength*3:.2f}**. An empty strike (z ≈ −3) gets tilt
≈ **{max(0.05, 1 - oi_strength*3):.2f}**. The 0.05 floor keeps the density
positive everywhere.
        """
    )
    cols = st.columns(3)
    cols[0].metric("Min tilt", f"{tilt.min():.3f}")
    cols[1].metric("Max tilt", f"{tilt.max():.3f}")
    cols[2].metric("Mean tilt (≈ 1)", f"{tilt.mean():.3f}")
    fig = go.Figure(go.Scatter(
        x=K, y=tilt, mode="lines",
        line=dict(color=AMBER, width=2.2),
        hovertemplate="$%{x:,.0f}<br>tilt %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_hline(y=1.0, line=dict(color=STONE, width=1, dash="dot"),
                  annotation_text=" 1.0 (no change)",
                  annotation_font=dict(color=STONE, size=9))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"))
    fig.update_layout(
        **base_layout(title="tilt(K) = max(0.05, 1 + λ · z(K))", height=300),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="Multiplier", gridcolor="#E5DCC9"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6 — Apply tilt (pre-normalisation)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 6 · Apply tilt to base RN"),
                unsafe_allow_html=True)
    raw_tilted, _adj_pdf_unused, area = _step_apply_and_normalise(base_pdf, tilt, K)
    st.markdown(
        f"""
$$\\widetilde{{f}}(K) = \\max\\!\\left(0,\\; f_{{RN}}(K)\\cdot \\text{{tilt}}(K)\\right)$$

This is **un-normalised** — the area under the curve is no longer 1.0.
Trapezoidal area ≈ **{area:.4f}** (would be 1.0 if no tilt). The next step
fixes this.
        """
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K, y=base_pdf, mode="lines",
        line=dict(color=GOLD, width=2),
        name="f_RN(K)",
    ))
    fig.add_trace(go.Scatter(
        x=K, y=raw_tilted, mode="lines",
        line=dict(color=TEAL, width=2, dash="dash"),
        name="f_RN · tilt (un-normalised)",
    ))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"))
    fig.update_layout(
        **base_layout(title="Before vs after tilt (un-normalised)", height=320),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="Density (un-normalised)", gridcolor="#E5DCC9"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 7 — Renormalise
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 7 · Renormalise"), unsafe_allow_html=True)
    if area <= 0:
        st.error("Tilted density has zero area — cannot normalise.")
        return
    adj_pdf = raw_tilted / area
    st.markdown(
        f"""
$$f_{{RN,\\,OI}}(K) = \\frac{{\\widetilde{{f}}(K)}}{{\\int \\widetilde{{f}}(K)\\,dK}}
= \\frac{{\\widetilde{{f}}(K)}}{{{area:.4f}}}$$

Now the OI-adjusted density integrates to 1 and is a proper probability density.
        """
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K, y=base_pdf, mode="lines",
        line=dict(color=GOLD, width=2.2),
        name="Base RN density",
    ))
    fig.add_trace(go.Scatter(
        x=K, y=adj_pdf, mode="lines",
        line=dict(color=TEAL, width=2.2, dash="dash"),
        name="OI-adjusted density",
    ))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
                      annotation_font=dict(color=INK, size=10))
    fig.update_layout(
        **base_layout(title="Normalised: base vs OI-adjusted", height=320),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="Probability density", gridcolor="#E5DCC9"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # STEP 8 — Compute moments
    # ─────────────────────────────────────────────────────────────────────
    st.markdown(section_label("Step 8 · Compute mean and P(above spot)"),
                unsafe_allow_html=True)
    base_moments = _step_moments(K, base_pdf, spot)
    adj_moments = _step_moments(K, adj_pdf, spot)
    st.markdown(
        r"""
$$\text{OI-adj mean} = \int K \cdot f_{RN,\,OI}(K)\,dK$$

$$P(\text{price} > \text{spot}) = 1 - F(\text{spot})
\quad\text{where } F\text{ is the CDF of the adjusted density.}$$
        """
    )
    cols = st.columns(4)
    cols[0].metric("RN mean (base)", fmt_money(base_moments["mean"]))
    cols[1].metric("OI-adj mean", fmt_money(adj_moments["mean"]),
                   f"{adj_moments['mean'] - base_moments['mean']:+,.0f}")
    cols[2].metric("RN P(above spot)", f"{base_moments['p_above']*100:.1f}%")
    cols[3].metric("OI-adj P(above spot)",
                   f"{adj_moments['p_above']*100:.1f}%",
                   f"{(adj_moments['p_above'] - base_moments['p_above'])*100:+.1f}pp")

    # CDF chart for visual P(above spot)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=K, y=base_moments["cdf"] * 100, mode="lines",
        line=dict(color=GOLD, width=2),
        name="Base RN CDF",
    ))
    fig.add_trace(go.Scatter(
        x=K, y=adj_moments["cdf"] * 100, mode="lines",
        line=dict(color=TEAL, width=2, dash="dash"),
        name="OI-adj CDF",
    ))
    fig.add_hline(y=50, line=dict(color=STONE, width=1, dash="dot"),
                  annotation_text="50%", annotation_font=dict(color=STONE, size=9))
    if spot > 0:
        fig.add_vline(x=spot, line=dict(color=INK, width=1.4, dash="dot"),
                      annotation_text=f" Spot ${spot:,.0f}",
                      annotation_font=dict(color=INK, size=10))
    fig.add_vline(x=base_moments["mean"], line=dict(color=GOLD, width=1.2, dash="dash"))
    fig.add_vline(x=adj_moments["mean"], line=dict(color=TEAL, width=1.2, dash="dash"))
    fig.update_layout(
        **base_layout(title="CDFs — where 50% line crosses spot tells you P(below)",
                      height=320),
        xaxis=dict(title="Strike K ($)", gridcolor="#E5DCC9"),
        yaxis=dict(title="Cumulative probability (%)",
                   range=[0, 100], gridcolor="#E5DCC9"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ─────────────────────────────────────────────────────────────────────
    # CAVEAT
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "The OI-adjusted curve is a positioning-tilted density, not a true "
        "risk-neutral density (a true RN density comes only from option prices, "
        "which is what the base BL gives you). Read it as a positioning compass "
        "alongside the pure RN curve — it answers 'where would the implied "
        "centre of gravity land if mass were reweighted toward strikes where "
        "people have parked their open interest?', not 'what's the actual "
        "risk-neutral expectation under a martingale measure?'"
    )


main()
