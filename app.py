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


def main() -> None:
    st.set_page_config(
        page_title=settings.app_title,
        page_icon=settings.app_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    log.info("Starting %s in %s mode", settings.app_title, settings.environment)

    st.title(settings.app_title)
    st.caption(
        f"Environment: {settings.environment} · "
        f"Modular BTC analytics dashboard · scaffolding only"
    )

    st.info(
        "This is the project scaffold. Pages, analytics, and data loaders "
        "are placeholders — implement them under `pages/`, `analytics/`, "
        "`data/`, `charts/`, and `utils/`."
    )

    with st.sidebar:
        st.markdown("### Navigation")
        st.caption("Pages will appear here once they are added under `pages/`.")
        st.markdown("---")
        st.markdown("### Status")
        st.caption(f"App: {settings.app_title}")
        st.caption(f"Version: {settings.version}")
        st.caption(f"Env: {settings.environment}")


if __name__ == "__main__":
    main()
