"""
btc_dashboard.data.flows
Trade-flow connectors. Recent prints / aggressor side per venue.
One DataFrame per venue — no cross-venue merging.

Public API
----------
- fetch_binance_spot_trades(symbol, limit)   -> TRADES_SCHEMA
- fetch_binance_perp_trades(symbol, limit)   -> TRADES_SCHEMA
- fetch_deribit_trades(symbol, count)        -> TRADES_SCHEMA
- fetch_coinbase_trades(symbol, limit)       -> TRADES_SCHEMA
- bucket_flow(trades_df, freq)               -> FLOW_BUCKETS_SCHEMA  (pure shaping)
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


TRADES_SCHEMA = Schema(
    name="trades",
    columns=(
        "time", "venue", "symbol", "trade_id", "price", "size",
        "side", "notional", "_source",
    ),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "trade_id": "string",
        "price": "float64",
        "size": "float64",       # base asset (BTC) units where the venue allows
        "side": "string",        # "buy" | "sell" — aggressor side
        "notional": "float64",   # USD notional = price * size
        "_source": "string",
    },
    description="Recent prints with aggressor side per venue. No cross-venue merging.",
)

FLOW_BUCKETS_SCHEMA = Schema(
    name="flow_buckets",
    columns=(
        "bucket", "venue", "symbol", "buy_notional", "sell_notional",
        "net_notional", "n_trades", "_source",
    ),
    dtypes={
        "bucket": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "buy_notional": "float64",
        "sell_notional": "float64",
        "net_notional": "float64",
        "n_trades": "int64",
        "_source": "string",
    },
    description="Per-bucket buy/sell aggregation. Pure data shaping — no signal logic.",
)


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


class BinanceFuturesClient(BaseAPIClient):
    base_url = settings.binance_fapi_url


class DeribitClient(BaseAPIClient):
    base_url = settings.deribit_base_url


class CoinbaseClient(BaseAPIClient):
    base_url = "https://api.exchange.coinbase.com"


# ─────────────────────────────────────────────────────────────────────────────
# FETCHERS
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance_spot_trades(symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    try:
        with BinanceSpotClient() as cli:
            data = cli._get("/api/v3/trades", params={"symbol": symbol, "limit": min(limit, 1000)})
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for t in data:
            price = float(t["price"]); size = float(t["qty"])
            # Binance spot: isBuyerMaker=true → seller was the aggressor → side="sell"
            side = "sell" if t.get("isBuyerMaker") else "buy"
            rows.append(
                {
                    "time": pd.to_datetime(int(t["time"]), unit="ms", utc=True),
                    "venue": "binance_spot",
                    "symbol": symbol,
                    "trade_id": str(t.get("id")),
                    "price": price,
                    "size": size,
                    "side": side,
                    "notional": price * size,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), TRADES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_spot_trades failed (%s) — using mock", exc)
        return _mock_trades("binance_spot", symbol, limit)


def fetch_binance_perp_trades(symbol: str = "BTCUSDT", limit: int = 500) -> pd.DataFrame:
    try:
        with BinanceFuturesClient() as cli:
            data = cli._get("/fapi/v1/trades", params={"symbol": symbol, "limit": min(limit, 1000)})
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for t in data:
            price = float(t["price"]); size = float(t["qty"])
            side = "sell" if t.get("isBuyerMaker") else "buy"
            rows.append(
                {
                    "time": pd.to_datetime(int(t["time"]), unit="ms", utc=True),
                    "venue": "binance_perp",
                    "symbol": symbol,
                    "trade_id": str(t.get("id")),
                    "price": price,
                    "size": size,
                    "side": side,
                    "notional": price * size,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), TRADES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_binance_perp_trades failed (%s) — using mock", exc)
        return _mock_trades("binance_perp", symbol, limit)


def fetch_deribit_trades(symbol: str = "BTC-PERPETUAL", count: int = 500) -> pd.DataFrame:
    try:
        with DeribitClient() as cli:
            payload = cli._get(
                "/api/v2/public/get_last_trades_by_instrument",
                params={"instrument_name": symbol, "count": min(count, 1000)},
            )
        result = payload.get("result", {}) if isinstance(payload, dict) else {}
        trades = result.get("trades") or []
        if not trades:
            raise APIError("empty response")
        rows = []
        for t in trades:
            price = float(t.get("price", 0))
            size = float(t.get("amount", 0)) / max(price, 1e-9)  # Deribit reports USD amount
            side = str(t.get("direction", "")).lower()  # "buy" | "sell"
            rows.append(
                {
                    "time": pd.to_datetime(int(t["timestamp"]), unit="ms", utc=True),
                    "venue": "deribit",
                    "symbol": symbol,
                    "trade_id": str(t.get("trade_id")),
                    "price": price,
                    "size": size,
                    "side": side,
                    "notional": float(t.get("amount", 0)),
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), TRADES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_deribit_trades failed (%s) — using mock", exc)
        return _mock_trades("deribit", symbol, count)


def fetch_coinbase_trades(symbol: str = "BTC-USD", limit: int = 500) -> pd.DataFrame:
    try:
        with CoinbaseClient() as cli:
            data = cli._get(f"/products/{symbol}/trades", params={"limit": min(limit, 1000)})
        if not isinstance(data, list) or not data:
            raise APIError("empty response")
        rows = []
        for t in data:
            price = float(t["price"]); size = float(t["size"])
            # Coinbase "side" is the maker side; aggressor is the opposite.
            maker = str(t.get("side", "")).lower()
            side = "buy" if maker == "sell" else "sell"
            rows.append(
                {
                    "time": pd.to_datetime(t["time"], utc=True),
                    "venue": "coinbase",
                    "symbol": symbol,
                    "trade_id": str(t.get("trade_id")),
                    "price": price,
                    "size": size,
                    "side": side,
                    "notional": price * size,
                    "_source": "live",
                }
            )
        return _conform(pd.DataFrame(rows), TRADES_SCHEMA)
    except Exception as exc:
        log.warning("fetch_coinbase_trades failed (%s) — using mock", exc)
        return _mock_trades("coinbase", symbol, limit)


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED VIEW (pure shaping — no signal logic)
# ─────────────────────────────────────────────────────────────────────────────
def bucket_flow(trades: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    """Aggregate trades into time buckets per venue/symbol.
    `freq` accepts any pandas offset alias (e.g. '1min', '5min', '1h')."""
    if trades is None or trades.empty:
        return _conform(pd.DataFrame(), FLOW_BUCKETS_SCHEMA)

    df = trades.copy()
    df["bucket"] = df["time"].dt.floor(freq)
    grouped = df.groupby(["bucket", "venue", "symbol"])
    out = grouped.apply(
        lambda g: pd.Series(
            {
                "buy_notional": float(g.loc[g["side"] == "buy", "notional"].sum()),
                "sell_notional": float(g.loc[g["side"] == "sell", "notional"].sum()),
                "n_trades": int(len(g)),
                "_source": str(g["_source"].iloc[0]),
            }
        ),
        include_groups=False,
    ).reset_index()
    out["net_notional"] = out["buy_notional"] - out["sell_notional"]
    return _conform(out, FLOW_BUCKETS_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def load_trades(
    venue: str, symbol: Optional[str] = None, limit: int = 500
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance_spot":
        return fetch_binance_spot_trades(symbol or "BTCUSDT", limit)
    if venue == "binance_perp":
        return fetch_binance_perp_trades(symbol or "BTCUSDT", limit)
    if venue == "deribit":
        return fetch_deribit_trades(symbol or "BTC-PERPETUAL", limit)
    if venue == "coinbase":
        return fetch_coinbase_trades(symbol or "BTC-USD", limit)
    raise ValueError(f"unknown venue: {venue}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_trades(venue: str, symbol: str, limit: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash((venue, symbol, "trades")) & 0xFFFFFFFF)
    end = datetime.now(tz=timezone.utc)
    times = [end - timedelta(seconds=float(rng.exponential(2.0)) * (limit - i)) for i in range(limit)]
    times = sorted(times)
    base = 65000.0
    drift = np.cumsum(rng.normal(0, 0.0005, size=limit))
    prices = base * np.exp(drift)
    sizes = np.abs(rng.normal(0.05, 0.05, size=limit)) + 0.001
    sides = rng.choice(["buy", "sell"], size=limit, p=[0.51, 0.49])
    rows = [
        {
            "time": pd.Timestamp(t, tz="UTC"),
            "venue": venue,
            "symbol": symbol,
            "trade_id": f"mock-{i}",
            "price": float(prices[i]),
            "size": float(sizes[i]),
            "side": str(sides[i]),
            "notional": float(prices[i] * sizes[i]),
            "_source": "mock",
        }
        for i, t in enumerate(times)
    ]
    return _conform(pd.DataFrame(rows), TRADES_SCHEMA)
