"""
btc_dashboard.pages.spot_volume — DEPRECATED.
Spot / volume charts now live on the landing page (app.py).
Delete this file from the repo on the next push.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from charts.theme import inject_global_css, page_title  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Spot / Volume · BTC", page_icon="₿", layout="wide")
    inject_global_css()
    page_title("Spot / Volume",
               "Moved — the spot price candles, SMA stack, VWAP and "
               "volume profile now live on the landing page (Overview).")
    st.info(
        "This page has been retired. The same charts are now rendered on the "
        "landing page below the GEX-by-strike section."
    )


main()
