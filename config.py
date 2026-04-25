"""
btc_dashboard · config.py
Centralised configuration and runtime settings.

Reads environment variables from the OS or a `.env` file (loaded via
`python-dotenv` when available). Never commit real keys — use `.env.example`
as the template and keep `.env` out of version control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Optional: load .env if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # python-dotenv is optional — fall back to plain os.environ.
    pass


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Application-level settings.

    All values are placeholders. Replace via environment variables or .env.
    """

    # ── Branding / runtime ────────────────────────────────────────────────
    app_title: str = "BTC Analytics Dashboard"
    app_icon: str = "₿"
    version: str = "0.1.0"
    environment: str = field(
        default_factory=lambda: os.getenv("APP_ENV", "development")
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_to_file: bool = field(
        default_factory=lambda: os.getenv("LOG_TO_FILE", "false").lower() == "true"
    )
    log_file: Path = LOG_DIR / "btc_dashboard.log"

    # ── External APIs (placeholders) ──────────────────────────────────────
    binance_base_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.com")
    )
    binance_fapi_url: str = field(
        default_factory=lambda: os.getenv("BINANCE_FAPI_URL", "https://fapi.binance.com")
    )
    deribit_base_url: str = field(
        default_factory=lambda: os.getenv("DERIBIT_BASE_URL", "https://www.deribit.com")
    )

    binance_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("BINANCE_API_KEY")
    )
    binance_api_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("BINANCE_API_SECRET")
    )
    deribit_client_id: Optional[str] = field(
        default_factory=lambda: os.getenv("DERIBIT_CLIENT_ID")
    )
    deribit_client_secret: Optional[str] = field(
        default_factory=lambda: os.getenv("DERIBIT_CLIENT_SECRET")
    )

    # ── HTTP defaults ─────────────────────────────────────────────────────
    request_timeout_seconds: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "10"))
    )
    request_max_retries: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_MAX_RETRIES", "2"))
    )

    # ── Cache (placeholder) ───────────────────────────────────────────────
    cache_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "30"))
    )

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


# Single shared instance imported across the app.
settings = Settings()
