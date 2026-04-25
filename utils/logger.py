"""
btc_dashboard.utils.logger
Centralised logger factory. Use:

    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("hello")

Console output is always on. File output is opt-in via `LOG_TO_FILE=true`.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import settings


_LOG_FORMAT = "%(asctime)s · %(levelname)-7s · %(name)s · %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured: bool = False


def _configure_root() -> None:
    """One-time configuration of the root logger."""
    global _configured
    if _configured:
        return

    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Clear any handlers Streamlit / Python may have attached to avoid dupes.
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler — always on
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    # File handler — opt-in
    if settings.log_to_file:
        file_h = RotatingFileHandler(
            settings.log_file,
            maxBytes=2_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_h.setFormatter(fmt)
        root.addHandler(file_h)

    _configured = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger by name (defaults to the root logger)."""
    _configure_root()
    return logging.getLogger(name if name else "btc_dashboard")
