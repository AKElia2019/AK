"""
btc_dashboard.data.spot
Spot-market connectors. One DataFrame per venue — no cross-venue merging.

Public API
----------
- fetch_binance_spot_klines(interval, limit)  -> SPOT_KLINES_SCHEMA
- fetch_coinbase_spot_klines(granularity, limit) -> SPOT_KLINES_SCHEMA
- fetch_binance_spot_ticker()                 -> SPOT_TICKER_SCHEMA
- fetch_coinbase_spot_ticker()                -> SPOT_TICKER_SCHEMA
- load_spot_klines(venue, ...)                -> dispatches per venue

If the network call fails, the connector falls back to deterministic mock
data with a `_source = "mock"` marker so the caller can see the fallback.
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


SPOT_KLINES_SCHEMA = Schema(
    name="spot_klines",
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
    description="OHLCV candles per venue/symbol. One row per candle close.",
)

SPOT_TICKER_SCHEMA = Schema(
    name="spot_ticker",
    columns=("time", "venue", "symbol", "price", "bid", "ask", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "price": "float64",
        "bid": "float64",
        "ask": "float64",
        "_source": "string",
    },
    description="Latest top-of-book snapshot per venue/symbol.",
)


def _empty(schema: Schema) -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype=schema.dtypes.get(c, "object")) for c in schema.columns})
    return df


def _conform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    for col in schema.columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype=schema.dtypes.get(col, "object"))
    return df[list(schema.columns)]


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
class BinanceSpotClient(BaseAPIClient):
    base_url = settings.binance_base_url


class CoinbaseSpotClient(BaseAPIClient):
    base_url = "https://api.exchange.coinbase.com"


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance_spot_klines(
    interval: str = "1h", limit: int = 100, symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    try:
        with BinanceSpotClient() as cli:
            data = cli._get(
                "/api/v3/klines",
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
        return _conform(pd.DataFrame(rows), SPOT_KLINES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_spot_klines failed (%s) — using mock", exc)
        return _mock_klines(venue="binance", symbol=symbol, interval=interval, limit=limit)


def fetch_binance_spot_ticker(symbol: str = "BTCUSDT") -> pd.DataFrame:
    try:
        with BinanceSpotClient() as cli:
            book = cli._get("/api/v3/ticker/bookTicker", params={"symbol": symbol})
        row = {
            "time": pd.Timestamp.now(tz="UTC"),
            "venue": "binance",
            "symbol": symbol,
            "price": (float(book["bidPrice"]) + float(book["askPrice"])) / 2.0,
            "bid": float(book["bidPrice"]),
            "ask": float(book["askPrice"]),
            "_source": "live",
        }
        return _conform(pd.DataFrame([row]), SPOT_TICKER_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_spot_ticker failed (%s) — using mock", exc)
        return _mock_ticker(venue="binance", symbol=symbol)


# ─────────────────────────────────────────────────────────────────────────────
# COINBASE
# ─────────────────────────────────────────────────────────────────────────────
_COINBASE_GRANULARITY = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400}


def fetch_coinbase_spot_klines(
    granularity: str = "1h", limit: int = 100, symbol: str = "BTC-USD"
) -> pd.DataFrame:
    try:
        seconds = _COINBASE_GRANULARITY.get(granularity, 3600)
        end = datetime.now(tz=timezone.utc)
        start = end - timedelta(seconds=seconds * limit)
        with CoinbaseSpotClient() as cli:
            data = cli._get(
                f"/products/{symbol}/candles",
                params={
                    "granularity": seconds,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        # Coinbase: [time, low, high, open, close, volume]
        for k in data:
            rows.append(
                {
                    "time": pd.to_datetime(int(k[0]), unit="s", utc=True),
                    "venue": "coinbase",
                    "symbol": symbol,
                    "open": float(k[3]),
                    "high": float(k[2]),
                    "low": float(k[1]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "quote_volume": float(k[5]) * float(k[4]),
                    "_source": "live",
                }
            )
        df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
        return _conform(df, SPOT_KLINES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinbase_spot_klines failed (%s) — using mock", exc)
        return _mock_klines(venue="coinbase", symbol=symbol, interval=granularity, limit=limit)


def fetch_coinbase_spot_ticker(symbol: str = "BTC-USD") -> pd.DataFrame:
    try:
        with CoinbaseSpotClient() as cli:
            t = cli._get(f"/products/{symbol}/ticker")
        row = {
            "time": pd.Timestamp.now(tz="UTC"),
            "venue": "coinbase",
            "symbol": symbol,
            "price": float(t["price"]),
            "bid": float(t["bid"]),
            "ask": float(t["ask"]),
            "_source": "live",
        }
        return _conform(pd.DataFrame([row]), SPOT_TICKER_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinbase_spot_ticker failed (%s) — using mock", exc)
        return _mock_ticker(venue="coinbase", symbol=symbol)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def load_spot_klines(
    venue: str, interval: str = "1h", limit: int = 100, symbol: Optional[str] = None
) -> pd.DataFrame:
    """Return klines for a single venue. No cross-venue merging."""
    venue = venue.lower()
    if venue == "binance":
        return fetch_binance_spot_klines(interval=interval, limit=limit, symbol=symbol or "BTCUSDT")
    if venue == "coinbase":
        return fetch_coinbase_spot_klines(granularity=interval, limit=limit, symbol=symbol or "BTC-USD")
    raise ValueError(f"unknown venue: {venue}")


def load_spot_ticker(venue: str, symbol: Optional[str] = None) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance":
        return fetch_binance_spot_ticker(symbol=symbol or "BTCUSDT")
    if venue == "coinbase":
        return fetch_coinbase_spot_ticker(symbol=symbol or "BTC-USD")
    raise ValueError(f"unknown venue: {venue}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_klines(venue: str, symbol: str, interval: str, limit: int) -> pd.DataFrame:
    secs = _COINBASE_GRANULARITY.get(interval, 3600)
    rng = np.random.default_rng(seed=hash((venue, symbol, interval)) & 0xFFFFFFFF)
    end = datetime.now(tz=timezone.utc).replace(microsecond=0)
    times = [end - timedelta(seconds=secs * (limit - i - 1)) for i in range(limit)]
    base = 65000.0
    rets = rng.normal(0, 0.005, size=limit)
    closes = base * np.exp(np.cumsum(rets))
    opens = np.roll(closes, 1)
    opens[0] = base
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.002, size=limit)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.002, size=limit)))
    vol = np.abs(rng.normal(50, 20, size=limit))
    rows = [
        {
            "time": pd.Timestamp(t, tz="UTC"),
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
    return _conform(pd.DataFrame(rows), SPOT_KLINES_SCHEMA)


def _mock_ticker(venue: str, symbol: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash((venue, symbol)) & 0xFFFFFFFF)
    mid = 65000.0 * (1 + rng.normal(0, 0.001))
    spread = mid * 0.0002
    row = {
        "time": pd.Timestamp.now(tz="UTC"),
        "venue": venue,
        "symbol": symbol,
        "price": float(mid),
        "bid": float(mid - spread / 2),
        "ask": float(mid + spread / 2),
        "_source": "mock",
    }
    return _conform(pd.DataFrame([row]), SPOT_TICKER_SCHEMA)
