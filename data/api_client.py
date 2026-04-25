"""
btc_dashboard.data.api_client
Base HTTP client for external market-data APIs.

This is the abstract foundation. Concrete clients (BinanceClient, DeribitClient,
…) should subclass `BaseAPIClient` and add endpoint methods. No endpoints are
implemented yet — this is scaffolding.
"""

from __future__ import annotations

from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


class APIError(Exception):
    """Raised when an API call fails or returns an error payload."""


class BaseAPIClient:
    """Lightweight base class for REST API clients.

    Subclasses set `base_url` and define methods that call `self._get(...)`
    or `self._post(...)`. This class handles:
      - shared `requests.Session`
      - default timeout & retries
      - authentication hook (override `_auth_headers()`)
      - basic error normalisation
    """

    base_url: str = ""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ) -> None:
        if base_url:
            self.base_url = base_url
        if not self.base_url:
            raise ValueError("base_url must be set on subclass or constructor.")

        self.timeout = timeout or settings.request_timeout_seconds
        self.max_retries = max_retries or settings.request_max_retries

        self.session = requests.Session()
        self.session.headers.update({"Accept-Encoding": "gzip"})
        self.session.mount(
            "https://",
            HTTPAdapter(
                pool_connections=1,
                pool_maxsize=16,
                max_retries=self.max_retries,
            ),
        )

    # ── Hooks subclasses can override ──────────────────────────────────────
    def _auth_headers(self) -> dict[str, str]:
        """Return per-request auth headers. Default: none."""
        return {}

    def _build_url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.base_url.rstrip('/')}/{path}"

    # ── Core request helpers ───────────────────────────────────────────────
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Any:
        url = self._build_url(path)
        headers = {**self._auth_headers(), **(extra_headers or {})}
        log.debug("HTTP %s %s params=%s", method, url, params)
        try:
            resp = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            log.warning("HTTP error: %s %s — %s", method, url, exc)
            raise APIError(f"network error: {exc}") from exc

        if resp.status_code >= 400:
            log.warning("HTTP %s %s → %s", method, url, resp.status_code)
            raise APIError(f"{method} {url} → HTTP {resp.status_code}: {resp.text[:200]}")

        try:
            return resp.json()
        except ValueError:
            return resp.text

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def _post(
        self,
        path: str,
        params: Optional[dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
    ) -> Any:
        return self._request("POST", path, params=params, data=data, json=json)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "BaseAPIClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
