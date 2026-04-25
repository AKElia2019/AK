"""
btc_dashboard.pages.positions
Open-position monitor — logs trades, evaluates the three exit conditions
on each refresh, surfaces alerts.

Persistence: positions live in `st.session_state` (per-session, MVP). When
ready, swap to a Postgres-backed store without changing this page's API.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analytics.exit_monitor import (    # noqa: E402
    ExitAlert,
    LiveState,
    Position,
    PositionAssessment,
    evaluate_portfolio,
)
from analytics.pipeline import PipelineResult, run_pipeline  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# THEME (mirror of overview)
# ─────────────────────────────────────────────────────────────────────────────
GOLD  = "#C9A55A"
TEAL  = "#1A7A6B"
RED   = "#A83232"
AMBER = "#B8832A"
STONE = "#9C968A"
INK   = "#1C1A17"


def _label(text: str) -> str:
    return (
        f'<div style="font-family:DM Mono,monospace;font-size:9px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{GOLD};'
        f'margin:18px 0 6px 0;">{text}</div>'
    )


def _level_color(level: str) -> str:
    return {"exit": RED, "warning": AMBER, "ok": TEAL}.get(level, STONE)


def _level_glyph(level: str) -> str:
    return {"exit": "🔴", "warning": "🟡", "ok": "🟢"}.get(level, "⚪")


# ─────────────────────────────────────────────────────────────────────────────
# SESSION-STATE STORE
# ─────────────────────────────────────────────────────────────────────────────
_POS_KEY = "open_positions"


def _store() -> list[Position]:
    return st.session_state.setdefault(_POS_KEY, [])


def _add(pos: Position) -> None:
    _store().append(pos)


def _remove(pos_id: str) -> None:
    st.session_state[_POS_KEY] = [p for p in _store() if p.id != pos_id]


def _update(pos_id: str, **fields) -> None:
    for p in _store():
        if p.id == pos_id:
            for k, v in fields.items():
                setattr(p, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PIPELINE  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _cached_pipeline() -> dict:
    return {"result": run_pipeline()}


def _live_state() -> tuple[Optional[PipelineResult], Optional[LiveState]]:
    try:
        pipe = _cached_pipeline()["result"]
    except Exception as exc:
        st.warning(f"Live pipeline failed ({exc}). Exit checks will be skipped.")
        return None, None
    live = LiveState(
        score_4h=float(pipe.score_4h),
        rn_mean=pipe.rn_mean,
        spot=float(pipe.spot or 0.0),
    )
    return pipe, live


# ─────────────────────────────────────────────────────────────────────────────
# UI — ADD-POSITION FORM
# ─────────────────────────────────────────────────────────────────────────────
def _render_add_form(pipe: Optional[PipelineResult]) -> None:
    st.markdown(_label("Open a New Position"), unsafe_allow_html=True)

    # Sensible defaults from the live pipeline (if available)
    default_score_4h = float(pipe.score_4h) if pipe else 0.0
    default_rn_mean = float(pipe.rn_mean) if (pipe and pipe.rn_mean) else 0.0
    default_spot = float(pipe.spot) if (pipe and pipe.spot) else 65_000.0

    with st.form("add_position_form", clear_on_submit=True):
        cols = st.columns([1, 1, 1, 1, 1])
        bias = cols[0].selectbox("Bias", ["long", "short"], key="form_bias")
        instrument = cols[1].selectbox(
            "Instrument",
            ["call", "put", "call_spread", "put_spread"],
            key="form_instrument",
        )
        entry_score = cols[2].number_input(
            "Entry 4h score",
            min_value=-100.0,
            max_value=100.0,
            value=float(default_score_4h),
            step=1.0,
            key="form_score",
        )
        entry_premium = cols[3].number_input(
            "Entry premium ($)",
            min_value=0.0,
            value=500.0,
            step=10.0,
            key="form_prem",
        )
        entry_spot = cols[4].number_input(
            "Entry spot ($)",
            min_value=0.0,
            value=float(default_spot),
            step=100.0,
            key="form_spot",
        )

        cols2 = st.columns([1, 3])
        entry_rn = cols2[0].number_input(
            "Entry RN mean ($, 0 = unknown)",
            min_value=0.0,
            value=float(default_rn_mean),
            step=100.0,
            key="form_rn",
        )
        notes = cols2[1].text_input("Notes (optional)", key="form_notes")

        submitted = st.form_submit_button("Add position", type="primary")
        if submitted:
            new_pos = Position(
                id=str(uuid.uuid4())[:8],
                bias=bias,
                instrument=instrument,
                entry_score_4h=float(entry_score),
                entry_rn_mean=(float(entry_rn) if entry_rn > 0 else None),
                entry_premium=float(entry_premium),
                entry_spot=float(entry_spot),
                opened_at=datetime.now(tz=timezone.utc),
                current_premium=float(entry_premium),
                notes=str(notes),
            )
            _add(new_pos)
            st.success(f"Added {bias} {instrument} (id {new_pos.id}).")


# ─────────────────────────────────────────────────────────────────────────────
# UI — POSITIONS TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _render_positions(positions: list[Position],
                      assessments: list[PositionAssessment],
                      live: Optional[LiveState]) -> None:
    if not positions:
        st.info(
            "No open positions yet. Use the form above to log a trade — "
            "the monitor will then evaluate the three exit conditions on every refresh."
        )
        return

    st.markdown(_label("Open Positions"), unsafe_allow_html=True)
    by_id = {a.position_id: a for a in assessments}

    for pos in positions:
        a = by_id.get(pos.id)
        level = a.overall_level if a else "ok"
        col = _level_color(level)
        glyph = _level_glyph(level)

        # Position header card
        opened = pos.opened_at.astimezone(tz=timezone.utc).strftime("%d %b %H:%M UTC")
        pnl = f"{a.pnl_pct:+.1f}%" if a else "—"
        bias_arrow = "▲" if pos.bias == "long" else "▼"
        title_html = (
            f'<div style="border-left:3px solid {col};background:#FFFFFF;'
            f'padding:12px 16px;margin-top:6px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;font-family:DM Sans,sans-serif;">'
            f'<div>{glyph}&nbsp;&nbsp;<b>{bias_arrow} {pos.bias.upper()} '
            f'{pos.instrument.replace("_"," ").title()}</b> '
            f'<span style="color:{STONE};font-size:12px;font-family:DM Mono,monospace;">'
            f'· id {pos.id} · opened {opened}</span></div>'
            f'<div style="font-family:DM Mono,monospace;font-size:13px;color:{col};'
            f'font-weight:600;">{level.upper()} · P&L {pnl}</div>'
            f'</div></div>'
        )
        st.markdown(title_html, unsafe_allow_html=True)

        # Two-column body: state | actions
        body_cols = st.columns([3, 1])
        with body_cols[0]:
            metric_cols = st.columns(5)
            metric_cols[0].metric(
                "Entry score", f"{pos.entry_score_4h:+.0f}",
                f"now {live.score_4h:+.0f}" if live else None,
            )
            metric_cols[1].metric("Entry spot", f"${pos.entry_spot:,.0f}")
            metric_cols[2].metric(
                "Entry RN mean",
                f"${pos.entry_rn_mean:,.0f}" if pos.entry_rn_mean else "—",
                f"now ${live.rn_mean:,.0f}" if (live and live.rn_mean) else None,
            )
            metric_cols[3].metric("Entry premium", f"${pos.entry_premium:,.2f}")
            metric_cols[4].metric("Current premium", f"${pos.current_premium:,.2f}",
                                  f"{a.pnl_pct:+.1f}%" if a else None)

            # Alert lines (one row per rule)
            if a is not None:
                alert_rows = []
                for al in a.alerts:
                    alert_rows.append({
                        "Rule": al.rule,
                        "Level": al.level.upper(),
                        "Signal": al.headline,
                        "Detail": al.detail,
                    })
                if alert_rows:
                    df = pd.DataFrame(alert_rows)

                    def _row_style(row):
                        c = _level_color(row["Level"].lower())
                        return [
                            "",
                            f"color:{c};font-weight:700;",
                            f"color:{c};",
                            "",
                        ]

                    st.dataframe(
                        df.style.apply(_row_style, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

            if pos.notes:
                st.caption(f"Notes: {pos.notes}")

        with body_cols[1]:
            new_prem = st.number_input(
                "Update current premium",
                min_value=0.0,
                value=float(pos.current_premium),
                step=10.0,
                key=f"prem_{pos.id}",
            )
            if new_prem != pos.current_premium:
                _update(pos.id, current_premium=float(new_prem))
            if st.button("Close position", key=f"close_{pos.id}",
                         use_container_width=True):
                _remove(pos.id)
                st.rerun()

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI — PORTFOLIO HEADER
# ─────────────────────────────────────────────────────────────────────────────
def _render_header(positions: list[Position],
                   assessments: list[PositionAssessment]) -> None:
    n_open = len(positions)
    n_exit = sum(1 for a in assessments if a.overall_level == "exit")
    n_warn = sum(1 for a in assessments if a.overall_level == "warning")
    n_ok = sum(1 for a in assessments if a.overall_level == "ok")

    cols = st.columns(4)
    cols[0].metric("Open positions", n_open)
    cols[1].metric("🔴 Exit signals", n_exit, delta_color="inverse")
    cols[2].metric("🟡 Warnings", n_warn)
    cols[3].metric("🟢 Healthy", n_ok)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Positions · BTC Risk Monitor",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f'<h1 style="font-family:DM Sans,sans-serif;font-weight:300;'
        f'color:{INK};margin:0 0 4px 0;">Positions</h1>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{STONE};'
        f'margin-bottom:14px;">Risk-management exit monitor · score · RN drift · premium</div>',
        unsafe_allow_html=True,
    )

    # Sidebar — quick legend + thresholds
    sb = st.sidebar
    sb.markdown("### Exit conditions")
    sb.caption(
        "🔴 **EXIT** when any one of:\n"
        "- 4h score crosses -40 / +40 against position\n"
        "- RN mean drifts > 1% against position\n"
        "- Premium loss exceeds -50%"
    )
    sb.caption(
        "🟡 **WARNING** when any one of:\n"
        "- score within 10pp of threshold\n"
        "- RN mean drifts > 0.5% against\n"
        "- premium loss past -30%"
    )
    sb.markdown("---")
    sb.markdown("### Persistence")
    sb.caption(
        "Positions are stored in your browser session. "
        "They will be cleared on page refresh / new browser tab. "
        "Wire to Postgres to persist across sessions."
    )

    pipe, live = _live_state()

    # Live state strip
    if live is not None:
        st.markdown(_label("Live State (4h)"), unsafe_allow_html=True)
        cols = st.columns(3)
        cols[0].metric("Score 4h", f"{live.score_4h:+.0f}")
        cols[1].metric(
            "RN mean",
            f"${live.rn_mean:,.0f}" if live.rn_mean is not None else "—",
        )
        cols[2].metric("Spot", f"${live.spot:,.0f}" if live.spot else "—")

    _render_add_form(pipe)

    positions = _store()
    assessments = (
        evaluate_portfolio(positions, live) if (live is not None and positions) else []
    )

    _render_header(positions, assessments)
    _render_positions(positions, assessments, live)


main()
