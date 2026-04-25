"""
btc_dashboard.data.futures
Futures / perpetual swap connectors. One DataFrame per venue — no cross-venue
merging. Includes klines, funding rate, and open interest history.

Public API
----------
- fetch_binance_perp_klines(interval, limit)         -> FUTURES_KLINES_SCHEMA
- fetch_deribit_perp_klines(resolution, limit)       -> FUTURES_KLINES_SCHEMA
- fetch_binance_funding_history(limit)               -> FUNDING_RATE_SCHEMA
- fetch_deribit_funding_history(limit)               -> FUNDING_RATE_SCHEMA
- fetch_binance_open_interest_hist(period, limit)    -> OPEN_INTEREST_SCHEMA
- fetch_deribit_open_interest(symbol)                -> OPEN_INTEREST_SCHEMA (single-row snapshot)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

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


FUTURES_KLINES_SCHEMA = Schema(
    name="futures_klines",
    columns=(
        "time", "venue", "symbol", "open", "high", "low", "close",
        "volume", "quote_volume", "_source",
    ),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "open": "float64",
        "high": "float64",
        "low": "float64",
        "close": "float64",
        "volume": "float64",
        "quote_volume": "float64",
        "_source": "string",
    },
    description="Perp/futures OHLCV per venue/symbol.",
)

FUNDING_RATE_SCHEMA = Schema(
    name="funding_rate",
    columns=("time", "venue", "symbol", "funding_rate", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "funding_rate": "float64",      # decimal per period (Binance: per 8h)
        "_source": "string",
    },
    description="Periodic funding rate per venue. Decimal (0.0001 = 1bp).",
)

OPEN_INTEREST_SCHEMA = Schema(
    name="open_interest",
    columns=("time", "venue", "symbol", "oi_base", "oi_usd", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "oi_base": "float64",   # contracts in base asset (BTC)
        "oi_usd": "float64",    # USD notional
        "_source": "string",
    },
    description="Open interest snapshot or history per venue/symbol.",
)


def _conform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    for col in schema.columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype=schema.dtypes.get(col, "object"))
    return df[list(schema.columns)]


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
class BinanceFuturesClient(BaseAPIClient):
    base_url = settings.binance_fapi_url


class DeribitClient(BaseAPIClient):
    base_url = settings.deribit_base_url


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance_perp_klines(
    interval: str = "1h", limit: int = 100, symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    try:
        with BinanceFuturesClient() as cli:
            data = cli._get(
                "/fapi/v1/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for k in data:
            rows.append(
                {
                    "time": pd.to_datetime(k[6], unit="ms", utc=True),
                    "venue": "binance",
                    "symbol": symbol,
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[7]),
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), FUTURES_KLINES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_perp_klines failed (%s) — using mock", exc)
        return _mock_futures_klines("binance", symbol, interval, limit)


def fetch_binance_funding_history(limit: int = 30, symbol: str = "BTCUSDT") -> pd.DataFrame:
    try:
        with BinanceFuturesClient() as cli:
            data = cli._get(
                "/fapi/v1/fundingRate",
                params={"symbol": symbol, "limit": limit},
            )
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for r in data:
            rows.append(
                {
                    "time": pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True),
                    "venue": "binance",
                    "symbol": symbol,
                    "funding_rate": float(r["fundingRate"]),
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), FUNDING_RATE_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_funding_history failed (%s) — using mock", exc)
        return _mock_funding("binance", symbol, limit)


def fetch_binance_open_interest_hist(
    period: str = "1h", limit: int = 100, symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    try:
        with BinanceFuturesClient() as cli:
            data = cli._get(
                "/futures/data/openInterestHist",
                params={"symbol": symbol, "period": period, "limit": limit},
            )
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for r in data:
            rows.append(
                {
                    "time": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                    "venue": "binance",
                    "symbol": symbol,
                    "oi_base": float(r["sumOpenInterest"]),
                    "oi_usd": float(r["sumOpenInterestValue"]),
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), OPEN_INTEREST_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_open_interest_hist failed (%s) — using mock", exc)
        return _mock_oi("binance", symbol, period, limit)


# ─────────────────────────────────────────────────────────────────────────────
# DERIBIT
# ─────────────────────────────────────────────────────────────────────────────
_DERIBIT_RES = {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "1D"}


def fetch_deribit_perp_klines(
    resolution: str = "1h", limit: int = 100, symbol: str = "BTC-PERPETUAL"
) -> pd.DataFrame:
    try:
        res = _DERIBIT_RES.get(resolution, "60")
        end_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        per_sec = {"1": 60, "5": 300, "15": 900, "60": 3600, "240": 14400, "1D": 86400}[res]
        start_ts = end_ts - per_sec * limit * 1000
        with DeribitClient() as cli:
            payload = cli._get(
                "/api/v2/public/get_tradingview_chart_data",
                params={
                    "instrument_name": symbol,
                    "resolution": res,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                },
            )
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        ticks = result.get("ticks") or []
        if not ticks:
            raise APIError("empty response")
        df = pd.DataFrame(
            {
                "time": pd.to_datetime(ticks, unit="ms", utc=True),
                "venue": "deribit",
                "symbol": symbol,
                "open": result.get("open", []),
                "high": result.get("high", []),
                "low": result.get("low", []),
                "close": result.get("close", []),
                "volume": result.get("volume", []),
            }
        )
        df["quote_volume"] = df["volume"].astype(float) * df["close"].astype(float)
        df["_source"] = "live"
        return _conform(df, FUTURES_KLINES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_deribit_perp_klines failed (%s) — using mock", exc)
        return _mock_futures_klines("deribit", symbol, resolution, limit)


def fetch_deribit_funding_history(limit: int = 30, symbol: str = "BTC-PERPETUAL") -> pd.DataFrame:
    """Deribit funding is computed continuously; this returns the historical
    funding-rate samples (8h-equivalent) reported by Deribit's API."""
    try:
        end_ts = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        start_ts = end_ts - 8 * 3600 * 1000 * limit
        with DeribitClient() as cli:
            payload = cli._get(
                "/api/v2/public/get_funding_rate_history",
                params={
                    "instrument_name": symbol,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                },
            )
        result = payload.get("result", []) if isinstance(payload, dict) else []
        if not result:
            raise APIError("empty response")
        rows = []
        for r in result[-limit:]:
            rows.append(
                {
                    "time": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                    "venue": "deribit",
                    "symbol": symbol,
                    "funding_rate": float(r.get("interest_8h", 0.0)),
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), FUNDING_RATE_SCHEMA)
    except Exception as exc:
        log.warning("fetch_deribit_funding_history failed (%s) — using mock", exc)
        return _mock_funding("deribit", symbol, limit)


def fetch_deribit_open_interest(symbol: str = "BTC-PERPETUAL") -> pd.DataFrame:
    try:
        with DeribitClient() as cli:
            payload = cli._get("/api/v2/public/ticker", params={"instrument_name": symbol})
        r = payload.get("result", {}) if isinstance(payload, dict) else {}
        if not r:
            raise APIError("empty response")
        oi_base = float(r.get("open_interest", 0.0))
        last = float(r.get("last_price") or r.get("mark_price") or 0.0)
        row = {
            "time": pd.Timestamp.now(tz="UTC"),
            "venue": "deribit",
            "symbol": symbol,
            "oi_base": oi_base,
            "oi_usd": oi_base * last,
            "_source": "live",
        }
        return _conform(pd.DataFrame([row]), OPEN_INTEREST_SCHEMA)
    except Exception as exc:
        log.warning("fetch_deribit_open_interest failed (%s) — using mock", exc)
        return _mock_oi("deribit", symbol, "snap", 1)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def load_perp_klines(
    venue: str, interval: str = "1h", limit: int = 100, symbol: Optional[str] = None
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance":
        return fetch_binance_perp_klines(interval=interval, limit=limit, symbol=symbol or "BTCUSDT")
    if venue == "deribit":
        return fetch_deribit_perp_klines(resolution=interval, limit=limit, symbol=symbol or "BTC-PERPETUAL")
    raise ValueError(f"unknown venue: {venue}")


def load_funding_history(
    venue: str, limit: int = 30, symbol: Optional[str] = None
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance":
        return fetch_binance_funding_history(limit=limit, symbol=symbol or "BTCUSDT")
    if venue == "deribit":
        return fetch_deribit_funding_history(limit=limit, symbol=symbol or "BTC-PERPETUAL")
    raise ValueError(f"unknown venue: {venue}")


def load_open_interest(
    venue: str, period: str = "1h", limit: int = 100, symbol: Optional[str] = None
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance":
        return fetch_binance_open_interest_hist(period=period, limit=limit, symbol=symbol or "BTCUSDT")
    if venue == "deribit":
        return fetch_deribit_open_interest(symbol=symbol or "BTC-PERPETUAL")
    raise ValueError(f"unknown venue: {venue}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_futures_klines(venue: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    secs = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}.get(interval, 3600)
    rng = np.random.default_rng(seed=hash((venue, symbol, interval, "fut")) & 0xFFFFFFFF)
    end = datetime.now(tz=timezone.utc).replace(microsecond=0)
    times = [end - timedelta(seconds=secs * (limit - i - 1)) for i in range(limit)]
    base = 65000.0
    rets = rng.normal(0, 0.005, size=limit)
    closes = base * np.exp(np.cumsum(rets))
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, size=limit)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, size=limit)))
    vol = np.abs(rng.normal(800, 250, size=limit))
    rows = [
        {
            "time": pd.Timestamp(t),
            "venue": venue,
            "symbol": symbol,
            "open": float(opens[i]),
            "high": float(highs[i]),
            "low": float(lows[i]),
            "close": float(closes[i]),
            "volume": float(vol[i]),
            "quote_volume": float(vol[i] * closes[i]),
            "_source": "mock",
        }
        for i, t in enumerate(times)
    ]
    return _conform(pd.DataFrame(rows), FUTURES_KLINES_SCHEMA)


def _mock_funding(venue: str, symbol: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash((venue, symbol, "fund")) & 0xFFFFFFFF)
    end = datetime.now(tz=timezone.utc).replace(microsecond=0)
    times = [end - timedelta(hours=8 * (limit - i - 1)) for i in range(limit)]
    rates = rng.normal(0.0001, 0.00015, size=limit)
    rows = [
        {
            "time": pd.Timestamp(t),
            "venue": venue,
            "symbol": symbol,
            "funding_rate": float(rates[i]),
            "_source": "mock",
        }
        for i, t in enumerate(times)
    ]
    return _conform(pd.DataFrame(rows), FUNDING_RATE_SCHEMA)


def _mock_oi(venue: str, symbol: str, period: str, limit: int) -> pd.DataFrame:
    secs = {"5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400, "snap": 0}.get(period, 3600)
    rng = np.random.default_rng(seed=hash((venue, symbol, "oi")) & 0xFFFFFFFF)
    end = datetime.now(tz=timezone.utc).replace(microsecond=0)
    if limit <= 1:
        oi_base = float(rng.normal(80000, 5000))
        row = {
            "time": pd.Timestamp(end),
            "venue": venue,
            "symbol": symbol,
            "oi_base": oi_base,
            "oi_usd": oi_base * 65000.0,
            "_source": "mock",
        }
        return _conform(pd.DataFrame([row]), OPEN_INTEREST_SCHEMA)
    times = [end - timedelta(seconds=secs * (limit - i - 1)) for i in range(limit)]
    base = 80000.0 + np.cumsum(rng.normal(0, 200, size=limit))
    rows = [
        {
            "time": pd.Timestamp(t),
            "venue": venue,
            "symbol": symbol,
            "oi_base": float(base[i]),
            "oi_usd": float(base[i] * 65000.0),
            "_source": "mock",
        }
        for i, t in enumerate(times)
    ]
    return _conform(pd.DataFrame(rows), OPEN_INTEREST_SCHEMA)
