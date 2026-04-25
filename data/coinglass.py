"""
btc_dashboard.data.coinglass
Coinglass connectors — cross-exchange aggregated futures metrics.

Coinglass aggregates open interest, funding rates, liquidations and
long/short positioning across CEX venues. It complements the per-venue
Binance / Deribit feeds with a single market-wide view.

Authentication
--------------
Requires COINGLASS_API_KEY (set in `.env` locally, or in DigitalOcean's
encrypted env-var UI in production). Without a key, every fetcher falls
back to deterministic mock data (`_source = "mock"`).

Public API
----------
- fetch_coinglass_btc_price()                -> PRICE_SCHEMA
- fetch_coinglass_aggregated_oi(interval)    -> AGG_OI_SCHEMA
- fetch_coinglass_funding_oi_weighted(int)   -> AGG_FUNDING_SCHEMA
- fetch_coinglass_long_short_ratio(int)      -> LONG_SHORT_SCHEMA
- fetch_coinglass_liquidations(interval)     -> LIQUIDATIONS_SCHEMA
- coinglass_status()                         -> dict (auth + connectivity)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from config import settings
from data.api_client import APIError, BaseAPIClient
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Schema:
    name: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    description: str


PRICE_SCHEMA = Schema(
    name="coinglass_btc_price",
    columns=("time", "venue", "symbol", "price", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "price": "float64",
        "_source": "string",
    },
    description="Coinglass-published BTC index price snapshot.",
)

AGG_OI_SCHEMA = Schema(
    name="coinglass_aggregated_oi",
    columns=(
        "time", "venue", "symbol", "oi_usd", "oi_change_pct", "_source",
    ),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "oi_usd": "float64",
        "oi_change_pct": "float64",
        "_source": "string",
    },
    description="Aggregated open-interest history across CEX futures venues.",
)

AGG_FUNDING_SCHEMA = Schema(
    name="coinglass_oi_weighted_funding",
    columns=("time", "venue", "symbol", "funding_rate", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "funding_rate": "float64",
        "_source": "string",
    },
    description="Open-interest weighted funding rate across CEX perps.",
)

LONG_SHORT_SCHEMA = Schema(
    name="coinglass_long_short_ratio",
    columns=("time", "venue", "symbol", "long_pct", "short_pct", "ratio", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "long_pct": "float64",
        "short_pct": "float64",
        "ratio": "float64",
        "_source": "string",
    },
    description="Long vs short account-ratio history.",
)

LIQUIDATIONS_SCHEMA = Schema(
    name="coinglass_liquidations",
    columns=(
        "time", "venue", "symbol", "long_liq_usd", "short_liq_usd",
        "total_liq_usd", "_source",
    ),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "long_liq_usd": "float64",
        "short_liq_usd": "float64",
        "total_liq_usd": "float64",
        "_source": "string",
    },
    description="Aggregated liquidation pulse across CEX venues.",
)


def _conform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    for col in schema.columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype=schema.dtypes.get(col, "object"))
    return df[list(schema.columns)]


# ─────────────────────────────────────────────────────────────────────────────
# CLIENT
# ─────────────────────────────────────────────────────────────────────────────
COINGLASS_BASE_URL = os.getenv(
    "COINGLASS_BASE_URL", "https://open-api-v4.coinglass.com"
)


class CoinglassClient(BaseAPIClient):
    """Auth via the `CG-API-KEY` header on every request."""

    base_url = COINGLASS_BASE_URL

    def _auth_headers(self) -> dict[str, str]:
        key = getattr(settings, "coinglass_api_key", None) or os.getenv("COINGLASS_API_KEY")
        if not key:
            return {}
        return {"CG-API-KEY": str(key)}


def _has_api_key() -> bool:
    key = getattr(settings, "coinglass_api_key", None) or os.getenv("COINGLASS_API_KEY")
    return bool(key)


def _unwrap(payload: Any) -> Any:
    """Coinglass wraps every response in {"code": "0", "msg": "...", "data": [...]}.
    Returns the `data` field, or raises APIError on a non-zero `code`."""
    if not isinstance(payload, dict):
        return payload
    code = str(payload.get("code", "0"))
    if code not in ("0", "00000", ""):
        raise APIError(f"Coinglass error code={code}: {payload.get('msg')}")
    return payload.get("data", payload)


# ─────────────────────────────────────────────────────────────────────────────
# FETCHERS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_coinglass_btc_price() -> pd.DataFrame:
    if not _has_api_key():
        log.info("Coinglass key missing — using mock BTC price.")
        return _mock_price()
    try:
        with CoinglassClient() as cli:
            payload = cli._get("/api/index/btc-price")
        data = _unwrap(payload)
        if isinstance(data, list) and data:
            row = data[0]
        elif isinstance(data, dict):
            row = data
        else:
            raise APIError("empty response")
        price = float(row.get("price") or row.get("indexPrice") or 0.0)
        ts = row.get("time") or row.get("timestamp")
        if ts:
            t = pd.to_datetime(int(ts), unit="ms", utc=True)
        else:
            t = pd.Timestamp.now(tz="UTC")
        df = pd.DataFrame(
            [{"time": t, "venue": "coinglass", "symbol": "BTC",
              "price": price, "_source": "live"}]
        )
        return _conform(df, PRICE_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinglass_btc_price failed (%s) — using mock", exc)
        return _mock_price()


def fetch_coinglass_aggregated_oi(
    interval: str = "1h", limit: int = 100, symbol: str = "BTC"
) -> pd.DataFrame:
    if not _has_api_key():
        return _mock_oi(symbol, interval, limit)
    try:
        with CoinglassClient() as cli:
            payload = cli._get(
                "/api/futures/open-interest/aggregated-history",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        data = _unwrap(payload)
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        prev_oi: Optional[float] = None
        for r in data:
            ts = int(r.get("time") or r.get("t") or 0)
            oi_usd = float(r.get("openInterest") or r.get("oi") or 0.0)
            change = (
                ((oi_usd - prev_oi) / prev_oi * 100.0)
                if (prev_oi and prev_oi > 0)
                else 0.0
            )
            prev_oi = oi_usd
            rows.append(
                {
                    "time": pd.to_datetime(ts, unit="ms", utc=True),
                    "venue": "coinglass",
                    "symbol": symbol,
                    "oi_usd": oi_usd,
                    "oi_change_pct": change,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), AGG_OI_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinglass_aggregated_oi failed (%s) — using mock", exc)
        return _mock_oi(symbol, interval, limit)


def fetch_coinglass_funding_oi_weighted(
    interval: str = "1h", limit: int = 100, symbol: str = "BTC"
) -> pd.DataFrame:
    if not _has_api_key():
        return _mock_funding(symbol, interval, limit)
    try:
        with CoinglassClient() as cli:
            payload = cli._get(
                "/api/futures/funding-rate/oi-weight-history",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        data = _unwrap(payload)
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for r in data:
            ts = int(r.get("time") or r.get("t") or 0)
            rate = float(r.get("fundingRate") or r.get("rate") or 0.0)
            rows.append(
                {
                    "time": pd.to_datetime(ts, unit="ms", utc=True),
                    "venue": "coinglass",
                    "symbol": symbol,
                    "funding_rate": rate,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), AGG_FUNDING_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinglass_funding_oi_weighted failed (%s) — using mock", exc)
        return _mock_funding(symbol, interval, limit)


def fetch_coinglass_long_short_ratio(
    interval: str = "1h", limit: int = 100, symbol: str = "BTC"
) -> pd.DataFrame:
    if not _has_api_key():
        return _mock_ls(symbol, interval, limit)
    try:
        with CoinglassClient() as cli:
            payload = cli._get(
                "/api/futures/long-short-account-ratio/history",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        data = _unwrap(payload)
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for r in data:
            ts = int(r.get("time") or r.get("t") or 0)
            long_pct = float(r.get("longAccount") or r.get("longRatio") or 0.0)
            short_pct = float(r.get("shortAccount") or r.get("shortRatio") or 0.0)
            ratio = (long_pct / short_pct) if short_pct > 0 else 0.0
            rows.append(
                {
                    "time": pd.to_datetime(ts, unit="ms", utc=True),
                    "venue": "coinglass",
                    "symbol": symbol,
                    "long_pct": long_pct,
                    "short_pct": short_pct,
                    "ratio": ratio,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), LONG_SHORT_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinglass_long_short_ratio failed (%s) — using mock", exc)
        return _mock_ls(symbol, interval, limit)


def fetch_coinglass_liquidations(
    interval: str = "1h", limit: int = 100, symbol: str = "BTC"
) -> pd.DataFrame:
    if not _has_api_key():
        return _mock_liq(symbol, interval, limit)
    try:
        with CoinglassClient() as cli:
            payload = cli._get(
                "/api/futures/liquidation/aggregated-history",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        data = _unwrap(payload)
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for r in data:
            ts = int(r.get("time") or r.get("t") or 0)
            long_liq = float(r.get("longLiquidationUsd") or r.get("longLiq") or 0.0)
            short_liq = float(r.get("shortLiquidationUsd") or r.get("shortLiq") or 0.0)
            rows.append(
                {
                    "time": pd.to_datetime(ts, unit="ms", utc=True),
                    "venue": "coinglass",
                    "symbol": symbol,
                    "long_liq_usd": long_liq,
                    "short_liq_usd": short_liq,
                    "total_liq_usd": long_liq + short_liq,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), LIQUIDATIONS_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinglass_liquidations failed (%s) — using mock", exc)
        return _mock_liq(symbol, interval, limit)


# ─────────────────────────────────────────────────────────────────────────────
# STATUS PROBE  (handy for debugging in the UI)
# ─────────────────────────────────────────────────────────────────────────────
def coinglass_status() -> dict:
    """Return a small status object: whether the key is set and whether
    a basic price call succeeds."""
    has_key = _has_api_key()
    out = {"has_api_key": has_key, "live": False, "error": None}
    if not has_key:
        return out
    try:
        with CoinglassClient() as cli:
            payload = cli._get("/api/index/btc-price")
        _unwrap(payload)
        out["live"] = True
    except Exception as exc:
        out["error"] = str(exc)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_price() -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash("coinglass-price") & 0xFFFFFFFF)
    df = pd.DataFrame(
        [
            {
                "time": pd.Timestamp.now(tz="UTC"),
                "venue": "coinglass",
                "symbol": "BTC",
                "price": float(65_000.0 * (1 + rng.normal(0, 0.001))),
                "_source": "mock",
            }
        ]
    )
    return _conform(df, PRICE_SCHEMA)


def _mock_series_times(interval: str, limit: int) -> list[pd.Timestamp]:
    secs = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 3600)
    end = datetime.now(tz=timezone.utc).replace(microsecond=0)
    return [pd.Timestamp(end - timedelta(seconds=secs * (limit - i - 1)), tz="UTC") for i in range(limit)]


def _mock_oi(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash(("cg-oi", symbol, interval)) & 0xFFFFFFFF)
    times = _mock_series_times(interval, limit)
    base = 25e9 + np.cumsum(rng.normal(0, 5e7, size=limit))
    pct = np.zeros(limit)
    for i in range(1, limit):
        pct[i] = (base[i] - base[i - 1]) / base[i - 1] * 100.0
    rows = [
        {
            "time": times[i],
            "venue": "coinglass",
            "symbol": symbol,
            "oi_usd": float(base[i]),
            "oi_change_pct": float(pct[i]),
            "_source": "mock",
        }
        for i in range(limit)
    ]
    return _conform(pd.DataFrame(rows), AGG_OI_SCHEMA)


def _mock_funding(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash(("cg-fund", symbol, interval)) & 0xFFFFFFFF)
    times = _mock_series_times(interval, limit)
    rates = rng.normal(0.0001, 0.00012, size=limit)
    rows = [
        {
            "time": times[i],
            "venue": "coinglass",
            "symbol": symbol,
            "funding_rate": float(rates[i]),
            "_source": "mock",
        }
        for i in range(limit)
    ]
    return _conform(pd.DataFrame(rows), AGG_FUNDING_SCHEMA)


def _mock_ls(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash(("cg-ls", symbol, interval)) & 0xFFFFFFFF)
    times = _mock_series_times(interval, limit)
    longs = np.clip(rng.normal(52, 4, size=limit), 30, 70)
    shorts = 100 - longs
    rows = [
        {
            "time": times[i],
            "venue": "coinglass",
            "symbol": symbol,
            "long_pct": float(longs[i]),
            "short_pct": float(shorts[i]),
            "ratio": float(longs[i] / shorts[i]) if shorts[i] > 0 else 0.0,
            "_source": "mock",
        }
        for i in range(limit)
    ]
    return _conform(pd.DataFrame(rows), LONG_SHORT_SCHEMA)


def _mock_liq(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash(("cg-liq", symbol, interval)) & 0xFFFFFFFF)
    times = _mock_series_times(interval, limit)
    long_liq = np.abs(rng.normal(2e6, 4e6, size=limit))
    short_liq = np.abs(rng.normal(2e6, 4e6, size=limit))
    rows = [
        {
            "time": times[i],
            "venue": "coinglass",
            "symbol": symbol,
            "long_liq_usd": float(long_liq[i]),
            "short_liq_usd": float(short_liq[i]),
            "total_liq_usd": float(long_liq[i] + short_liq[i]),
            "_source": "mock",
        }
        for i in range(limit)
    ]
    return _conform(pd.DataFrame(rows), LIQUIDATIONS_SCHEMA)
