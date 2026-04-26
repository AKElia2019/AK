"""
btc_dashboard.pages.positions
Trade Journal — log trades by strategy, develop statistics over time.

Per-session persistence via `st.session_state`. The data model is
identical to a DB-backed schema, so swapping in Postgres later is a
mechanical change.
"""

from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analytics.journal import (        # noqa: E402
    INSTRUMENTS, OPT_TYPES, DIRECTIONS,
    Strategy, Trade,
    by_instrument_breakdown,
    compute_strategy_stats,
    trade_margin_usd, trade_pnl_pct, trade_pnl_usd, trade_r_multiple,
)
from analytics.pipeline import run_pipeline   # noqa: E402
from charts.theme import (                     # noqa: E402
    GOLD, TEAL, RED, AMBER, STONE, INK,
    base_layout, fmt_money, inject_global_css, page_title, section_label,
)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STORE
# ─────────────────────────────────────────────────────────────────────────────
_S_KEY = "journal_strategies"
_T_KEY = "journal_trades"
_ACTIVE_KEY = "journal_active_strategy_id"


def _strategies() -> list[Strategy]:
    return st.session_state.setdefault(_S_KEY, [])


def _trades() -> list[Trade]:
    return st.session_state.setdefault(_T_KEY, [])


def _active_id() -> Optional[str]:
    return st.session_state.get(_ACTIVE_KEY)


def _set_active(strategy_id: str) -> None:
    st.session_state[_ACTIVE_KEY] = strategy_id


def _ensure_default_strategy() -> Strategy:
    """First-run convenience — create a 'Default Playbook' strategy."""
    strategies = _strategies()
    if strategies:
        if not _active_id() or all(s.id != _active_id() for s in strategies):
            _set_active(strategies[0].id)
        return next(s for s in strategies if s.id == _active_id())

    default = Strategy(
        id=str(uuid.uuid4())[:8],
        name="Default Playbook",
        description="Untitled — edit me.",
        rules=(
            "Gatekeeper: 4h score outside [-50, +50]; 1h and 4h must agree.\n"
            "Direction: long if 4h > 60 and 1h > 50; short mirrored.\n"
            "Setup: trend / squeeze / mean reversion → from regime classifier.\n"
            "Sizing: 0.75–1% high · 0.4–0.6% medium · 0.25–0.5% squeeze."
        ),
        capital=100_000.0,
    )
    strategies.append(default)
    _set_active(default.id)
    return default


def _add_strategy(s: Strategy) -> None:
    _strategies().append(s)


def _remove_strategy(strategy_id: str) -> None:
    st.session_state[_S_KEY] = [s for s in _strategies() if s.id != strategy_id]
    st.session_state[_T_KEY] = [t for t in _trades() if t.strategy_id != strategy_id]
    if _active_id() == strategy_id:
        remaining = _strategies()
        st.session_state[_ACTIVE_KEY] = remaining[0].id if remaining else None


def _add_trade(t: Trade) -> None:
    _trades().append(t)


def _remove_trade(trade_id: str) -> None:
    st.session_state[_T_KEY] = [t for t in _trades() if t.id != trade_id]


def _close_trade(trade_id: str, *, close_price: Optional[float] = None,
                 close_premium: Optional[float] = None,
                 status: str = "closed_manual") -> None:
    for t in _trades():
        if t.id != trade_id:
            continue
        t.close_at = datetime.now(tz=timezone.utc)
        t.status = status
        if t.instrument == "options":
            t.close_premium = float(close_premium) if close_premium is not None else None
        else:
            t.close_price = float(close_price) if close_price is not None else None
        return


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PIPELINE  (used to prefill entry score / RN mean / spot)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def _pipe() -> dict:
    return {"result": run_pipeline()}


def _live() -> dict:
    """Best-effort live snapshot. Returns {} on failure."""
    try:
        r = _pipe()["result"]
        return {
            "spot": float(r.spot or 0.0),
            "score_4h": float(r.score_4h),
            "rn_mean": (float(r.rn_mean) if r.rn_mean is not None else None),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# UI — STRATEGY MANAGER
# ─────────────────────────────────────────────────────────────────────────────
def _render_strategy_manager() -> Strategy:
    """Top-of-page strategy selector + edit / new / delete actions."""
    active = _ensure_default_strategy()
    strategies = _strategies()

    st.markdown(section_label("Strategy"), unsafe_allow_html=True)

    cols = st.columns([3, 1, 1])
    with cols[0]:
        names = [f"{s.name} (id {s.id})" for s in strategies]
        idx = next((i for i, s in enumerate(strategies) if s.id == active.id), 0)
        choice = st.selectbox(
            "Active strategy",
            range(len(strategies)),
            index=idx,
            format_func=lambda i: names[i],
            label_visibility="collapsed",
        )
        selected = strategies[choice]
        if selected.id != active.id:
            _set_active(selected.id)
            st.rerun()
    with cols[1]:
        if st.button("➕ New strategy", use_container_width=True):
            st.session_state["journal_show_new_form"] = True
    with cols[2]:
        if st.button("🗑 Delete", use_container_width=True,
                     help="Delete this strategy and all its trades."):
            st.session_state["journal_confirm_delete"] = active.id

    if st.session_state.get("journal_confirm_delete") == active.id:
        st.warning(f"Delete strategy **{active.name}** and all its trades?")
        c = st.columns([1, 1, 4])
        if c[0].button("Yes, delete", type="primary"):
            _remove_strategy(active.id)
            st.session_state.pop("journal_confirm_delete", None)
            st.rerun()
        if c[1].button("Cancel"):
            st.session_state.pop("journal_confirm_delete", None)
            st.rerun()

    if st.session_state.get("journal_show_new_form"):
        with st.form("new_strategy_form", clear_on_submit=True):
            st.markdown("**Define a new strategy**")
            name = st.text_input("Name", placeholder="e.g. Trend long, high conviction")
            desc = st.text_input("One-line description")
            rules = st.text_area(
                "Strategy logic / rules",
                placeholder="Describe entry triggers, sizing, exit rules…",
                height=140,
            )
            capital = st.number_input("Starting capital ($)",
                                       min_value=100.0, value=100_000.0, step=1000.0)
            cols2 = st.columns([1, 1, 6])
            create = cols2[0].form_submit_button("Create", type="primary")
            cancel = cols2[1].form_submit_button("Cancel")
            if create:
                new = Strategy(
                    id=str(uuid.uuid4())[:8],
                    name=name or "Untitled",
                    description=desc or "",
                    rules=rules or "",
                    capital=float(capital),
                )
                _add_strategy(new)
                _set_active(new.id)
                st.session_state["journal_show_new_form"] = False
                st.success(f"Created '{new.name}'.")
                st.rerun()
            if cancel:
                st.session_state["journal_show_new_form"] = False
                st.rerun()

    # Edit current strategy in an expander
    with st.expander(f"Edit · {active.name}"):
        new_name = st.text_input("Name", value=active.name, key=f"edit_name_{active.id}")
        new_desc = st.text_input("Description", value=active.description, key=f"edit_desc_{active.id}")
        new_rules = st.text_area("Rules", value=active.rules, height=140, key=f"edit_rules_{active.id}")
        new_capital = st.number_input("Capital ($)", min_value=100.0,
                                       value=float(active.capital), step=1000.0,
                                       key=f"edit_cap_{active.id}")
        if st.button("Save changes", key=f"save_{active.id}"):
            active.name = new_name
            active.description = new_desc
            active.rules = new_rules
            active.capital = float(new_capital)
            st.success("Saved.")
            st.rerun()

    return active


# ─────────────────────────────────────────────────────────────────────────────
# UI — NEW-TRADE FORM
# ─────────────────────────────────────────────────────────────────────────────
def _render_new_trade_form(strategy: Strategy, live: dict) -> None:
    st.markdown(section_label("Log a Trade"), unsafe_allow_html=True)

    # Instrument first — determines the rest of the form
    instrument = st.radio(
        "Instrument",
        options=list(INSTRUMENTS),
        format_func=lambda x: x.title(),
        horizontal=True,
        key="new_trade_instrument",
    )

    with st.form(f"new_trade_form_{instrument}", clear_on_submit=True):
        cols = st.columns([1, 1, 1, 1])
        direction = cols[0].selectbox("Direction", DIRECTIONS, key="ntf_dir")

        if instrument == "options":
            option_type = cols[1].selectbox("Option type", OPT_TYPES, key="ntf_otype")
            strike = cols[2].number_input("Strike ($)", min_value=0.0,
                                           value=float(live.get("spot") or 65_000.0),
                                           step=100.0, key="ntf_strike")
            expiry_d = cols[3].date_input(
                "Expiry", value=date.today(), key="ntf_expiry"
            )

            row2 = st.columns([1, 1, 1, 1])
            contracts = row2[0].number_input("Contracts", min_value=0.001,
                                              value=1.0, step=0.5, key="ntf_contracts")
            open_premium = row2[1].number_input("Open premium ($/contract)",
                                                 min_value=0.0, value=500.0,
                                                 step=10.0, key="ntf_op_prem")
            tp_premium = row2[2].number_input("TP premium ($, 0 = none)",
                                                min_value=0.0, value=0.0,
                                                step=10.0, key="ntf_tp_prem")
            sl_premium = row2[3].number_input("SL premium ($, 0 = none)",
                                                min_value=0.0, value=0.0,
                                                step=10.0, key="ntf_sl_prem")

            open_price = float(live.get("spot") or 0.0)
            size = float(contracts)
            leverage = 1.0
            tp = float(tp_premium) if tp_premium > 0 else None
            sl = float(sl_premium) if sl_premium > 0 else None

        elif instrument == "futures":
            open_price = cols[1].number_input("Open price ($)", min_value=0.0,
                                                value=float(live.get("spot") or 65_000.0),
                                                step=10.0, key="ntf_open")
            size = cols[2].number_input("Size (BTC)", min_value=0.0001,
                                         value=0.10, step=0.01, key="ntf_size")
            leverage = cols[3].number_input("Leverage (x)", min_value=1.0,
                                             value=5.0, step=1.0, max_value=100.0,
                                             key="ntf_lev")

            row2 = st.columns([1, 1, 2])
            tp_in = row2[0].number_input("TP price ($, 0 = none)", min_value=0.0,
                                          value=0.0, step=10.0, key="ntf_tp")
            sl_in = row2[1].number_input("SL price ($, 0 = none)", min_value=0.0,
                                          value=0.0, step=10.0, key="ntf_sl")
            tp = float(tp_in) if tp_in > 0 else None
            sl = float(sl_in) if sl_in > 0 else None
            option_type = None
            strike = None
            expiry_d = None
            open_premium = None

        else:  # spot
            open_price = cols[1].number_input("Open price ($)", min_value=0.0,
                                                value=float(live.get("spot") or 65_000.0),
                                                step=10.0, key="ntf_open")
            size = cols[2].number_input("Size (BTC)", min_value=0.0001,
                                         value=0.10, step=0.01, key="ntf_size")
            cols[3].caption("Leverage = 1.0 (spot)")
            leverage = 1.0

            row2 = st.columns([1, 1, 2])
            tp_in = row2[0].number_input("TP price ($, 0 = none)", min_value=0.0,
                                          value=0.0, step=10.0, key="ntf_tp")
            sl_in = row2[1].number_input("SL price ($, 0 = none)", min_value=0.0,
                                          value=0.0, step=10.0, key="ntf_sl")
            tp = float(tp_in) if tp_in > 0 else None
            sl = float(sl_in) if sl_in > 0 else None
            option_type = None
            strike = None
            expiry_d = None
            open_premium = None

        # Journal context (optional)
        st.markdown("**Journal context (optional)**")
        ctx_cols = st.columns([1, 1, 1, 3])
        bias = ctx_cols[0].selectbox("Bias", ["", "long", "short", "no_trade"],
                                       index=0, key="ntf_bias")
        setup = ctx_cols[1].selectbox("Setup", ["", "trend", "squeeze",
                                                  "mean_reversion", "none"],
                                        index=0, key="ntf_setup")
        conviction = ctx_cols[2].selectbox("Conviction", ["", "high", "medium", "low"],
                                             index=0, key="ntf_conv")
        notes = ctx_cols[3].text_input("Notes", key="ntf_notes")

        # Auto-fill from live pipeline
        autofill = st.checkbox("Auto-fill entry score / RN mean from live pipeline",
                                value=True, key="ntf_autofill")

        submit_cols = st.columns([1, 1, 4])
        save = submit_cols[0].form_submit_button("Log trade", type="primary")
        if save:
            entry_score = float(live.get("score_4h", 0.0)) if autofill else None
            entry_rn = (
                float(live["rn_mean"]) if (autofill and live.get("rn_mean") is not None)
                else None
            )
            new = Trade(
                id=str(uuid.uuid4())[:8],
                strategy_id=strategy.id,
                instrument=instrument,
                direction=direction,
                open_at=datetime.now(tz=timezone.utc),
                open_price=float(open_price),
                size=float(size),
                leverage=float(leverage),
                tp=tp,
                sl=sl,
                option_type=option_type,
                strike=float(strike) if strike else None,
                expiry=(datetime.combine(expiry_d, time(8, 0), tzinfo=timezone.utc)
                        if expiry_d else None),
                open_premium=(float(open_premium) if open_premium is not None else None),
                bias=bias or None,
                setup=setup or None,
                conviction=conviction or None,
                entry_score_4h=entry_score,
                entry_rn_mean=entry_rn,
                notes=notes,
            )
            _add_trade(new)
            st.success(f"Trade logged · id {new.id}")


# ─────────────────────────────────────────────────────────────────────────────
# UI — TRADES TABLE (open + closed)
# ─────────────────────────────────────────────────────────────────────────────
def _render_open_trades(strategy: Strategy, live: dict) -> None:
    open_trades = [t for t in _trades() if t.strategy_id == strategy.id and t.status == "open"]
    st.markdown(section_label(f"Open Trades ({len(open_trades)})"),
                unsafe_allow_html=True)
    if not open_trades:
        st.caption("No open trades for this strategy.")
        return

    spot = live.get("spot") or 0.0
    for t in open_trades:
        instrument_lbl = {"futures": "FUT", "spot": "SPOT", "options": "OPT"}[t.instrument]
        dir_arrow = "▲" if t.direction == "long" else "▼"
        opened = t.open_at.strftime("%d %b %H:%M UTC")

        # Live unrealised P&L
        live_pnl = trade_pnl_usd(t, mark=spot if t.instrument != "options" else None)
        pnl_color = TEAL if live_pnl > 0 else RED if live_pnl < 0 else STONE

        title = (
            f'<div style="border-left:3px solid {GOLD};background:#FFFFFF;'
            f'padding:12px 16px;margin-top:6px;'
            f'display:flex;justify-content:space-between;align-items:center;">'
            f'<div style="font-family:DM Sans,sans-serif;">{instrument_lbl} · '
            f'{dir_arrow} <b>{t.direction.upper()}</b> · '
            f'<span style="color:{STONE};font-size:12px;font-family:DM Mono,monospace;">'
            f'id {t.id} · opened {opened}</span></div>'
            f'<div style="font-family:DM Mono,monospace;font-size:13px;color:{pnl_color};font-weight:600;">'
            f'unrealised {live_pnl:+,.0f} $</div></div>'
        )
        st.markdown(title, unsafe_allow_html=True)

        cols = st.columns([3, 1])
        with cols[0]:
            mc = st.columns(5)
            if t.instrument == "options":
                mc[0].metric("Type", f"{(t.option_type or '').upper()}")
                mc[1].metric("Strike", fmt_money(t.strike))
                mc[2].metric("Contracts", f"{t.size:g}")
                mc[3].metric("Open premium", f"${t.open_premium:,.2f}" if t.open_premium else "—")
                mc[4].metric("Expiry", t.expiry.strftime("%d %b %y") if t.expiry else "—")
            else:
                mc[0].metric("Open", fmt_money(t.open_price))
                mc[1].metric("Size (BTC)", f"{t.size:g}")
                mc[2].metric("Leverage", f"{t.leverage:g}x")
                mc[3].metric("TP", fmt_money(t.tp) if t.tp else "—")
                mc[4].metric("SL", fmt_money(t.sl) if t.sl else "—")
            ctx_bits = []
            if t.bias: ctx_bits.append(f"bias **{t.bias}**")
            if t.setup: ctx_bits.append(f"setup **{t.setup}**")
            if t.conviction: ctx_bits.append(f"conviction **{t.conviction}**")
            if t.entry_score_4h is not None: ctx_bits.append(f"entry score **{t.entry_score_4h:+.0f}**")
            if t.entry_rn_mean: ctx_bits.append(f"entry RN **${t.entry_rn_mean:,.0f}**")
            if ctx_bits:
                st.caption(" · ".join(ctx_bits))
            if t.notes:
                st.caption(f"Notes: {t.notes}")

        with cols[1]:
            with st.form(f"close_{t.id}", clear_on_submit=False):
                if t.instrument == "options":
                    close_prem = st.number_input(
                        "Close premium",
                        min_value=0.0,
                        value=float(t.open_premium or 0.0),
                        step=10.0,
                        key=f"cp_{t.id}",
                    )
                    close_price = None
                else:
                    close_price = st.number_input(
                        "Close price",
                        min_value=0.0,
                        value=float(spot if spot else t.open_price),
                        step=10.0,
                        key=f"cp_{t.id}",
                    )
                    close_prem = None
                status = st.selectbox(
                    "Reason",
                    ["closed_tp", "closed_sl", "closed_manual"],
                    index=2,
                    key=f"cs_{t.id}",
                )
                close_btn = st.form_submit_button("Close trade")
                delete_btn = st.form_submit_button("Delete (no record)")
                if close_btn:
                    _close_trade(t.id, close_price=close_price,
                                  close_premium=close_prem, status=status)
                    st.rerun()
                if delete_btn:
                    _remove_trade(t.id)
                    st.rerun()

        st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)


def _render_closed_trades(strategy: Strategy) -> None:
    closed = [t for t in _trades() if t.strategy_id == strategy.id and t.status != "open"]
    st.markdown(section_label(f"Closed Trades ({len(closed)})"),
                unsafe_allow_html=True)
    if not closed:
        st.caption("No closed trades yet.")
        return

    rows = []
    for t in sorted(closed, key=lambda x: x.close_at or x.open_at, reverse=True):
        pnl_usd = trade_pnl_usd(t)
        pnl_pct = trade_pnl_pct(t) * 100
        r = trade_r_multiple(t)
        rows.append({
            "id": t.id,
            "instrument": t.instrument,
            "dir": t.direction,
            "opened": t.open_at.strftime("%d %b %H:%M"),
            "closed": (t.close_at.strftime("%d %b %H:%M") if t.close_at else "—"),
            "open": (
                f"${t.open_premium:,.2f}"
                if t.instrument == "options" and t.open_premium is not None
                else fmt_money(t.open_price)
            ),
            "close": (
                f"${t.close_premium:,.2f}"
                if t.instrument == "options" and t.close_premium is not None
                else fmt_money(t.close_price)
            ),
            "size": f"{t.size:g}",
            "lev": f"{t.leverage:g}x" if t.instrument != "options" else "—",
            "P&L $": f"{pnl_usd:+,.0f}",
            "P&L %": f"{pnl_pct:+.2f}%",
            "R": (f"{r:+.2f}" if r is not None else "—"),
            "status": t.status.replace("closed_", "").upper(),
        })

    df = pd.DataFrame(rows)

    def _row_style(row):
        try:
            v = float(row["P&L $"].replace(",", "").replace("+", ""))
        except Exception:
            v = 0.0
        c = TEAL if v > 0 else RED if v < 0 else STONE
        return [
            "", "", "", "", "", "", "", "", "",
            f"color:{c};font-weight:600;",
            f"color:{c};",
            "",
            "",
        ]

    st.dataframe(df.style.apply(_row_style, axis=1),
                 use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI — STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def _render_stats(strategy: Strategy) -> None:
    stats = compute_strategy_stats(strategy, _trades())
    st.markdown(section_label("Strategy Statistics"), unsafe_allow_html=True)

    cols = st.columns(6)
    cols[0].metric("Trades", stats.n_trades, f"{stats.n_open} open · {stats.n_closed} closed")
    cols[1].metric("Win rate", f"{stats.win_rate*100:.1f}%",
                    f"{stats.n_wins}W / {stats.n_losses}L")
    cols[2].metric("Total P&L", f"${stats.total_pnl_usd:+,.0f}",
                    f"avg ${stats.avg_pnl_usd:+,.0f}")
    cols[3].metric(
        "Profit factor",
        f"{stats.profit_factor:.2f}" if stats.profit_factor != float("inf") else "∞",
    )
    cols[4].metric("Avg R", f"{stats.avg_r_multiple:+.2f}" if stats.avg_r_multiple else "—")
    cols[5].metric(
        "Best / worst",
        f"${stats.best_trade_usd:+,.0f}",
        f"worst ${stats.worst_trade_usd:+,.0f}",
    )

    cols2 = st.columns(4)
    cols2[0].metric("Avg win", f"${stats.avg_win_usd:+,.0f}")
    cols2[1].metric("Avg loss", f"${stats.avg_loss_usd:+,.0f}")
    cols2[2].metric("Capital", fmt_money(strategy.capital))
    cols2[3].metric("Net return", f"{(stats.total_pnl_usd/strategy.capital*100):+.2f}%"
                                    if strategy.capital > 0 else "—")

    if stats.pnl_curve:
        xs = [ts for ts, _ in stats.pnl_curve]
        ys = [v for _, v in stats.pnl_curve]
        fig = go.Figure(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            line=dict(color=GOLD, width=2.4),
            marker=dict(size=5, color=GOLD),
            fill="tozeroy",
            fillcolor=("rgba(26,122,107,0.10)" if (ys and ys[-1] >= 0)
                       else "rgba(168,50,50,0.10)"),
            hovertemplate="%{x|%d %b %H:%M}<br>Cum P&L %{y:+,.0f}<extra></extra>",
            showlegend=False,
        ))
        fig.add_hline(y=0, line=dict(color=STONE, width=1, dash="dot"))
        fig.update_layout(
            **base_layout(title="Cumulative P&L (closed trades)", height=320),
            xaxis=dict(title=None, gridcolor="#E5DCC9"),
            yaxis=dict(title="P&L ($)", gridcolor="#E5DCC9"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Per-instrument breakdown
    st.markdown(section_label("Per-Instrument"), unsafe_allow_html=True)
    bd = by_instrument_breakdown(strategy, _trades())
    rows = [
        {
            "Instrument": k.title(),
            "Closed trades": v["n"],
            "Wins": v["wins"],
            "Win rate": f"{v['win_rate']*100:.1f}%" if v["n"] else "—",
            "Total P&L": f"${v['total_pnl_usd']:+,.0f}",
            "Avg P&L": f"${v['avg_pnl_usd']:+,.0f}" if v["n"] else "—",
        }
        for k, v in bd.items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI — SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar(live: dict) -> None:
    sb = st.sidebar
    sb.markdown("### Live state")
    sb.caption(f"Spot: {fmt_money(live.get('spot'))}")
    sb.caption(f"Score 4h: {live.get('score_4h', 0):+.0f}")
    rn = live.get("rn_mean")
    sb.caption(f"RN mean: {fmt_money(rn) if rn is not None else '—'}")

    sb.markdown("---")
    sb.markdown("### Persistence")
    sb.caption(
        "Strategies & trades live in this browser session only. Wire to "
        "Postgres for cross-session persistence."
    )
    n_s = len(_strategies())
    n_t = len(_trades())
    sb.caption(f"{n_s} strategies · {n_t} trades")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Trade Journal · BTC",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_global_css()
    page_title("Trade Journal",
               "Log trades by strategy · futures · spot · options · build statistics")

    live = _live()
    _render_sidebar(live)
    active = _render_strategy_manager()

    # Show the strategy's playbook (rules) for context
    if active.description or active.rules:
        with st.expander(f"Playbook · {active.name}"):
            if active.description:
                st.markdown(f"_{active.description}_")
            if active.rules:
                st.markdown(active.rules)

    # Tabs: log, open, closed, stats
    tabs = st.tabs(["📝 Log a trade", "📂 Open", "📁 Closed", "📊 Statistics"])
    with tabs[0]:
        _render_new_trade_form(active, live)
    with tabs[1]:
        _render_open_trades(active, live)
    with tabs[2]:
        _render_closed_trades(active)
    with tabs[3]:
        _render_stats(active)


main()
