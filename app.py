"""
btc_dashboard · app.py
Entry point for the Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Theme tokens (mirror the rest of the app) ──────────────────────────────
GOLD  = "#C9A55A"
TEAL  = "#1A7A6B"
RED   = "#A83232"
STONE = "#9C968A"
INK   = "#1C1A17"


def _hero() -> None:
    """Headline block — title + one-line tagline, brand-aligned."""
    st.markdown(
        f'<h1 style="font-family:DM Sans,sans-serif;font-weight:300;'
        f'color:{INK};margin:0 0 6px 0;font-size:42px;">{settings.app_title}</h1>'
        f'<div style="font-family:DM Mono,monospace;font-size:11px;'
        f'letter-spacing:.18em;text-transform:uppercase;color:{GOLD};'
        f'margin-bottom:6px;">Decision interface · BTC futures + options</div>'
        f'<div style="font-family:DM Sans,sans-serif;font-size:14px;'
        f'color:#5e594f;line-height:1.6;max-width:760px;margin-bottom:28px;">'
        f"Live Binance, Deribit and Coinglass feeds drive a systematic score, "
        f"regime classifier and Kelly-sized trade plan. Everything you need to "
        f"go from market snapshot to trade ticket in one screen."
        f"</div>",
        unsafe_allow_html=True,
    )


def _card(
    icon: str,
    title: str,
    body: str,
    page_path: str,
    accent: str,
) -> None:
    """Render a quick-link card. Uses st.page_link when available so clicks
    route through Streamlit's native navigation."""
    st.markdown(
        f'<div style="border:1px solid #EDEBE6;border-left:3px solid {accent};'
        f'background:#FFFFFF;padding:18px 20px;border-radius:2px;'
        f'box-shadow:0 1px 0 rgba(0,0,0,0.02);">'
        f'<div style="font-family:DM Mono,monospace;font-size:9px;'
        f'letter-spacing:.22em;text-transform:uppercase;color:{accent};'
        f'margin-bottom:8px;">{icon} &nbsp; {title}</div>'
        f'<div style="font-family:DM Sans,sans-serif;font-size:13px;'
        f'color:#44403A;line-height:1.6;min-height:62px;margin-bottom:10px;">'
        f"{body}</div></div>",
        unsafe_allow_html=True,
    )
    if hasattr(st, "page_link"):
        st.page_link(page_path, label=f"Open  →", icon=None)
    else:
        st.caption(f"Click **{title}** in the left sidebar to open.")


def _quick_links() -> None:
    """Two-column grid of quick links to the live pages."""
    cols = st.columns(2, gap="medium")
    with cols[0]:
        _card(
            icon="◆",
            title="Overview",
            body=(
                "Decision-focused headline page. Live market snapshot, "
                "score × conviction quadrant, Kelly-sized position, and "
                "the 3-bullet rationale behind it."
            ),
            page_path="pages/overview.py",
            accent=GOLD,
        )
    with cols[1]:
        _card(
            icon="◇",
            title="Trade Suggestion",
            body=(
                "Bias · setup · instrument · suggested options structure · "
                "stop / TP1 / TP2 / runner levels · 3-bullet explanation. "
                "All driven by the same upstream pipeline."
            ),
            page_path="pages/trade_suggestion.py",
            accent=TEAL,
        )


def _sidebar_status() -> None:
    with st.sidebar:
        st.markdown("### Status")
        st.caption(f"App: {settings.app_title}")
        st.caption(f"Version: {settings.version}")
        env_color = TEAL if settings.environment == "production" else STONE
        st.markdown(
            f'<div style="font-family:DM Mono,monospace;font-size:11px;'
            f'color:{env_color};">Env: {settings.environment}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown("### Data feeds")
        st.caption("Binance · Deribit · Coinglass (set COINGLASS_API_KEY for aggregated feeds)")


def main() -> None:
    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    log.info("Starting %s in %s mode", settings.app_title, settings.environment)

    _hero()
    _quick_links()
    _sidebar_status()


main()
