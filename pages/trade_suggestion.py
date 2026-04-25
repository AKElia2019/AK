"""
btc_dashboard.pages.trade_suggestion
Streamlit page that surfaces the live trade suggestion.

Renders, in order:

    Bias · Setup · Conviction        (hero metrics)
    Instrument · Suggested trade     (clean text + delta/expiry)
    Risk · Position size             (numeric metrics)
    Stop / TP1 / TP2 / Runner        (levels table)
    Explanation                      (3 bullets)

The page is self-contained: inputs come from the left sidebar, the
analytics layers (`recommendation`, `position_sizing`, `regime`) do all
the work, and the main pane is read-only output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `analytics/` is importable
# when Streamlit launches this file as a top-level page script.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analytics.recommendation import (        # noqa: E402
    OptionsSuggestion,
    Recommendation,
    RecommendationInputs,
    evaluate_recommendation,
)
from analytics.position_sizing import (       # noqa: E402
    SizingInputs,
    TradePlan,
    build_trade_plan,
)
from analytics.regime import RegimeAssessment  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
GOLD  = "#C9A55A"
TEAL  = "#1A7A6B"
RED   = "#A83232"
AMBER = "#B8832A"
STONE = "#9C968A"
INK   = "#1C1A17"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _sidebar_inputs() -> dict:
    sb = st.sidebar

    sb.markdown("### Scores")
    score_1h = sb.slider("1h composite",  -100, 100, 65, 1)
    score_4h = sb.slider("4h composite",  -100, 100, 72, 1)
    stable_1h = sb.checkbox("1h stable", value=True)
    stable_4h = sb.checkbox("4h stable", value=True)

    sb.markdown("### Regime (4h)")
    regime_label = sb.selectbox(
        "Regime label",
        ["trend", "squeeze", "mean_reversion", "neutral"],
        index=0,
    )
    regime_conf = sb.slider("Regime confidence", 0.0, 1.0, 0.65, 0.05)

    sb.markdown("### Market state")
    spot   = sb.number_input("Spot ($)",      min_value=1000.0, value=65000.0, step=100.0)
    atm_iv = sb.number_input("ATM IV (%)",    min_value=5.0,    value=55.0,    step=1.0)

    sb.markdown("### Liquidity context")
    nearest_support    = sb.number_input("Nearest support ($)",    min_value=0.0, value=63000.0, step=100.0)
    nearest_resistance = sb.number_input("Nearest resistance ($)", min_value=0.0, value=68000.0, step=100.0)

    sb.markdown("### Account & horizon")
    capital    = sb.number_input("Capital ($)", min_value=100.0, value=100_000.0, step=1000.0)
    primary_tf = sb.radio("Primary timeframe", ["4h", "1h"], index=0, horizontal=True)

    return dict(
        score_1h=score_1h, score_4h=score_4h,
        stable_1h=stable_1h, stable_4h=stable_4h,
        regime_label=regime_label, regime_conf=regime_conf,
        spot=spot, atm_iv=atm_iv,
        capital=capital, primary_tf=primary_tf,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _label(text: str) -> str:
    return (
        f'<div style="font-family:DM Mono,monospace;font-size:9px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{GOLD};'
        f'margin:18px 0 6px 0;">{text}</div>'
    )


def _hero(bias: str, setup: str, conviction: str) -> None:
    color = TEAL if bias == "long" else RED if bias == "short" else STONE
    arrow = "▲" if bias == "long" else "▼" if bias == "short" else "—"

    cols = st.columns(3, gap="medium")
    with cols[0]:
        st.markdown(
            f'<div style="text-align:center;padding:18px 8px;'
            f'border-left:3px solid {color};">'
            f'<div style="font-family:DM Mono,monospace;font-size:9px;'
            f'letter-spacing:.22em;text-transform:uppercase;color:{STONE};">Bias</div>'
            f'<div style="font-family:DM Sans,sans-serif;font-size:34px;'
            f'font-weight:300;color:{color};line-height:1.0;margin-top:8px;">'
            f'{arrow}&nbsp;{bias.replace("_"," ").upper()}</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div style="text-align:center;padding:18px 8px;'
            f'border-left:3px solid {GOLD};">'
            f'<div style="font-family:DM Mono,monospace;font-size:9px;'
            f'letter-spacing:.22em;text-transform:uppercase;color:{STONE};">Setup</div>'
            f'<div style="font-family:DM Sans,sans-serif;font-size:24px;'
            f'font-weight:400;color:{INK};line-height:1.0;margin-top:14px;">'
            f'{setup.replace("_"," ").title()}</div></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div style="text-align:center;padding:18px 8px;'
            f'border-left:3px solid {AMBER};">'
            f'<div style="font-family:DM Mono,monospace;font-size:9px;'
            f'letter-spacing:.22em;text-transform:uppercase;color:{STONE};">Conviction</div>'
            f'<div style="font-family:DM Sans,sans-serif;font-size:24px;'
            f'font-weight:400;color:{INK};line-height:1.0;margin-top:14px;">'
            f'{conviction.upper()}</div></div>',
            unsafe_allow_html=True,
        )


def _suggested_trade_text(opt: OptionsSuggestion | None, bias: str) -> str:
    if opt is None or opt.structure == "none":
        return "—"
    long_lo, long_hi = opt.long_leg_delta
    exp_lo, exp_hi = opt.expiry_days
    if opt.short_leg_delta is None:
        return (
            f"Buy {opt.structure} "
            f"{long_lo:.0f}–{long_hi:.0f}Δ · "
            f"{exp_lo}–{exp_hi}D expiry"
        )
    sh_lo, sh_hi = opt.short_leg_delta
    return (
        f"Long {long_lo:.0f}–{long_hi:.0f}Δ · "
        f"short {sh_lo:.0f}–{sh_hi:.0f}Δ · "
        f"{exp_lo}–{exp_hi}D expiry"
    )


def _instrument_section(opt: OptionsSuggestion | None, bias: str) -> None:
    st.markdown(_label("Suggested Trade"), unsafe_allow_html=True)
    cols = st.columns([1, 2])
    structure = (opt.structure if opt else "none").replace("_", " ").title()
    cols[0].metric("Instrument", structure if structure != "None" else "—")
    cols[1].metric("Suggested trade", _suggested_trade_text(opt, bias))


def _risk_section(plan: TradePlan, opt: OptionsSuggestion | None) -> None:
    st.markdown(_label("Risk & Sizing"), unsafe_allow_html=True)
    cols = st.columns(4)
    if plan.bias == "no_trade":
        cols[0].metric("Risk", "0.00%")
        cols[1].metric("Position size", "—")
        cols[2].metric("Options size", "—")
        cols[3].metric("R/R to TP2", "—")
        return

    cols[0].metric(
        "Risk",
        f"{plan.risk_pct*100:.2f}%",
        f"${plan.risk_amount_usd:,.0f} of capital",
    )
    cols[1].metric(
        "Position size",
        f"{plan.position_size_btc:.4f} BTC",
        f"${plan.position_size_usd:,.0f} notional",
    )
    if opt and opt.structure != "none":
        cols[2].metric(
            "Options size",
            f"{opt.size_multiplier:.2f}×",
            "after IV haircut",
        )
    else:
        cols[2].metric("Options size", "—")
    cols[3].metric("R/R to TP2", f"{plan.risk_reward_tp2:.2f}x")


def _levels_section(plan: TradePlan) -> None:
    st.markdown(_label("Levels"), unsafe_allow_html=True)
    if plan.bias == "no_trade":
        st.caption("No directional bias — no levels.")
        return

    rows = [
        ("Entry",  plan.entry,  0.0),
        ("Stop",   plan.stop,   abs(plan.entry  - plan.stop)),
        ("TP1",    plan.tp1,    abs(plan.tp1    - plan.entry)),
        ("TP2",    plan.tp2,    abs(plan.tp2    - plan.entry)),
        ("Runner", plan.runner, abs(plan.runner - plan.entry)),
    ]
    df = pd.DataFrame(
        [
            {
                "Level":    name,
                "Price":    f"${px:,.0f}",
                "Distance": "—" if d == 0 else f"${d:,.0f}",
            }
            for name, px, d in rows
        ]
    )

    def _row_style(row):
        if row["Level"] == "Stop":
            c = f"color:{RED};font-weight:600;"
        elif row["Level"] in ("TP1", "TP2"):
            c = f"color:{TEAL};"
        elif row["Level"] == "Runner":
            c = f"color:{GOLD};font-weight:500;"
        elif row["Level"] == "Entry":
            c = f"color:{INK};font-weight:600;"
        else:
            c = ""
        return [c, c, ""]

    st.dataframe(
        df.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )


def _explanation_section(
    rec: Recommendation, plan: TradePlan, opt: OptionsSuggestion | None
) -> None:
    st.markdown(_label("Explanation"), unsafe_allow_html=True)
    color = TEAL if rec.bias == "long" else RED if rec.bias == "short" else STONE
    body = "<br>".join(f"&bull;&nbsp;&nbsp;{line}" for line in rec.explanation)
    st.markdown(
        f'<div style="background:#F8F7F4;border-left:3px solid {color};'
        f'padding:14px 18px;font-size:13px;color:#44403A;line-height:1.7;">'
        f"{body}</div>",
        unsafe_allow_html=True,
    )

    # Optional, collapsed-by-default detail panes — they keep the main view clean.
    if not rec.gatekeeper_passed and rec.blocked_reasons:
        with st.expander("Why the gatekeeper blocked this trade"):
            for r in rec.blocked_reasons:
                st.markdown(f"• {r}")

    if plan.bias != "no_trade":
        with st.expander("Sizing detail"):
            for n in plan.notes:
                st.markdown(f"• {n}")
            if plan.haircuts:
                st.markdown("**Haircuts applied**")
                for h in plan.haircuts:
                    st.markdown(f"• {h}")
        with st.expander("Exit conditions"):
            for e in plan.exit_conditions:
                st.markdown(f"• {e}")
        if opt is not None and opt.structure != "none" and opt.notes:
            with st.expander("Options structure rationale"):
                for n in opt.notes:
                    st.markdown(f"• {n}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Trade Suggestion · BTC",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        f'<h1 style="font-family:DM Sans,sans-serif;font-weight:300;'
        f'color:{INK};margin:0 0 4px 0;">Trade Suggestion</h1>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{STONE};'
        f'margin-bottom:8px;">Live recommendation from the systematic pipeline · '
        f'edit the sidebar to explore.</div>',
        unsafe_allow_html=True,
    )

    inp = _sidebar_inputs()

    # Synthesize a RegimeAssessment from the sidebar so we can drive the
    # recommendation pipeline end-to-end without real upstream data.
    regime = RegimeAssessment(
        regime=inp["regime_label"],
        confidence=float(inp["regime_conf"]),
        direction=None,
        scores={inp["regime_label"]: float(inp["regime_conf"])},
        rationale={inp["regime_label"]: ["Provided manually via sidebar"]},
    )

    rec = evaluate_recommendation(
        RecommendationInputs(
            score_1h=float(inp["score_1h"]),
            score_4h=float(inp["score_4h"]),
            regime_4h=regime,
            stable_1h=bool(inp["stable_1h"]),
            stable_4h=bool(inp["stable_4h"]),
            atm_iv_pct=float(inp["atm_iv"]),
            primary_timeframe=str(inp["primary_tf"]),
        )
    )

    plan = build_trade_plan(
        SizingInputs(
            bias=rec.bias,
            setup=rec.setup,
            conviction=rec.conviction,
            capital=float(inp["capital"]),
            spot=float(inp["spot"]),
            atm_iv_pct=float(inp["atm_iv"]),
            stable_1h=bool(inp["stable_1h"]),
            stable_4h=bool(inp["stable_4h"]),
            nearest_support=(float(inp["nearest_support"]) if inp["nearest_support"] > 0 else None),
            nearest_resistance=(float(inp["nearest_resistance"]) if inp["nearest_resistance"] > 0 else None),
        )
    )

    # Output
    _hero(rec.bias, rec.setup, rec.conviction)
    _instrument_section(rec.options_suggestion, rec.bias)
    _risk_section(plan, rec.options_suggestion)
    _levels_section(plan)
    _explanation_section(rec, plan, rec.options_suggestion)


main()
