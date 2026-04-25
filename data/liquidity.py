"""
btc_dashboard.data.liquidity
Order-book / liquidity connectors. One DataFrame per venue — no cross-venue
merging. Returns the raw L2 book and a separate top-of-book snapshot.

Public API
----------
- fetch_binance_spot_order_book(symbol, depth)   -> ORDER_BOOK_SCHEMA
- fetch_binance_perp_order_book(symbol, depth)   -> ORDER_BOOK_SCHEMA
- fetch_deribit_order_book(symbol, depth)        -> ORDER_BOOK_SCHEMA
- fetch_coinbase_order_book(symbol, level)       -> ORDER_BOOK_SCHEMA
- top_of_book(order_book_df)                     -> TOP_OF_BOOK_SCHEMA
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


ORDER_BOOK_SCHEMA = Schema(
    name="order_book_l2",
    columns=("time", "venue", "symbol", "side", "level", "price", "size", "_source"),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "side": "string",      # "bid" | "ask"
        "level": "int64",      # 0 = best, 1 = next, …
        "price": "float64",
        "size": "float64",     # in base asset (BTC) where venue allows
        "_source": "string",
    },
    description="Resting L2 order-book levels per venue/symbol/side.",
)

TOP_OF_BOOK_SCHEMA = Schema(
    name="top_of_book",
    columns=(
        "time", "venue", "symbol", "best_bid", "best_ask",
        "mid", "spread", "spread_bps", "_source",
    ),
    dtypes={
        "time": "datetime64[ns, UTC]",
        "venue": "string",
        "symbol": "string",
        "best_bid": "float64",
        "best_ask": "float64",
        "mid": "float64",
        "spread": "float64",
        "spread_bps": "float64",
        "_source": "string",
    },
    description="Single-row top-of-book derived from ORDER_BOOK_SCHEMA.",
)


def _conform(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    for col in schema.columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype=schema.dtypes.get(col, "object"))
    return df[list(schema.columns)]


def _book_from_levels(
    venue: str, symbol: str, bids: list, asks: list, source: str = "live"
) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for i, (p, s) in enumerate(bids):
        rows.append(
            {
                "time": now, "venue": venue, "symbol": symbol,
                "side": "bid", "level": i,
                "price": float(p), "size": float(s), "_source": source,
            }
        )
    for i, (p, s) in enumerate(asks):
        rows.append(
            {
                "time": now, "venue": venue, "symbol": symbol,
                "side": "ask", "level": i,
                "price": float(p), "size": float(s), "_source": source,
            }
        )
    return _conform(pd.DataFrame(rows), ORDER_BOOK_SCHEMA)


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
def fetch_binance_spot_order_book(symbol: str = "BTCUSDT", depth: int = 100) -> pd.DataFrame:
    try:
        with BinanceSpotClient() as cli:
            data = cli._get("/api/v3/depth", params={"symbol": symbol, "limit": depth})
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids or not asks:
            raise APIError("empty book")
        return _book_from_levels("binance_spot", symbol, bids, asks, "live")
    except Exception as exc:
        log.warning("fetch_binance_spot_order_book failed (%s) — using mock", exc)
        return _mock_book("binance_spot", symbol, depth)


def fetch_binance_perp_order_book(symbol: str = "BTCUSDT", depth: int = 100) -> pd.DataFrame:
    try:
        with BinanceFuturesClient() as cli:
            data = cli._get("/fapi/v1/depth", params={"symbol": symbol, "limit": depth})
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        if not bids or not asks:
            raise APIError("empty book")
        return _book_from_levels("binance_perp", symbol, bids, asks, "live")
    except Exception as exc:
        log.warning("fetch_binance_perp_order_book failed (%s) — using mock", exc)
        return _mock_book("binance_perp", symbol, depth)


def fetch_deribit_order_book(symbol: str = "BTC-PERPETUAL", depth: int = 100) -> pd.DataFrame:
    try:
        with DeribitClient() as cli:
            payload = cli._get(
                "/api/v2/public/get_order_book",
                params={"instrument_name": symbol, "depth": depth},
            )
        r = payload.get("result", {}) if isinstance(payload, dict) else {}
        bids = r.get("bids") or []
        asks = r.get("asks") or []
        if not bids or not asks:
            raise APIError("empty book")
        return _book_from_levels("deribit", symbol, bids, asks, "live")
    except Exception as exc:
        log.warning("fetch_deribit_order_book failed (%s) — using mock", exc)
        return _mock_book("deribit", symbol, depth)


def fetch_coinbase_order_book(symbol: str = "BTC-USD", level: int = 2) -> pd.DataFrame:
    try:
        with CoinbaseClient() as cli:
            data = cli._get(f"/products/{symbol}/book", params={"level": level})
        bids = data.get("bids") or []
        asks = data.get("asks") or []
        # Coinbase entries are [price, size, num_orders] — drop the third.
        bids = [(p, s) for (p, s, *_rest) in bids]
        asks = [(p, s) for (p, s, *_rest) in asks]
        if not bids or not asks:
            raise APIError("empty book")
        return _book_from_levels("coinbase", symbol, bids, asks, "live")
    except Exception as exc:
        log.warning("fetch_coinbase_order_book failed (%s) — using mock", exc)
        return _mock_book("coinbase", symbol, 50)


# ─────────────────────────────────────────────────────────────────────────────
# DERIVED VIEW
# ─────────────────────────────────────────────────────────────────────────────
def top_of_book(book: pd.DataFrame) -> pd.DataFrame:
    """Reduce an order-book frame to a single-row top-of-book per venue/symbol.
    Pure data shaping — no signal computation."""
    if book is None or book.empty:
        return _conform(pd.DataFrame(), TOP_OF_BOOK_SCHEMA)
    rows = []
    for (venue, symbol), sub in book.groupby(["venue", "symbol"]):
        bids = sub[sub["side"] == "bid"].sort_values("level")
        asks = sub[sub["side"] == "ask"].sort_values("level")
        if bids.empty or asks.empty:
            continue
        bb = float(bids["price"].iloc[0])
        ba = float(asks["price"].iloc[0])
        mid = (bb + ba) / 2.0
        spread = ba - bb
        rows.append(
            {
                "time": sub["time"].max(),
                "venue": venue,
                "symbol": symbol,
                "best_bid": bb,
                "best_ask": ba,
                "mid": mid,
                "spread": spread,
                "spread_bps": (spread / mid) * 10000.0 if mid > 0 else 0.0,
                "_source": str(sub["_source"].iloc[0]),
            }
        )
    return _conform(pd.DataFrame(rows), TOP_OF_BOOK_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def load_order_book(
    venue: str, symbol: Optional[str] = None, depth: int = 100
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "binance_spot":
        return fetch_binance_spot_order_book(symbol or "BTCUSDT", depth)
    if venue == "binance_perp":
        return fetch_binance_perp_order_book(symbol or "BTCUSDT", depth)
    if venue == "deribit":
        return fetch_deribit_order_book(symbol or "BTC-PERPETUAL", depth)
    if venue == "coinbase":
        return fetch_coinbase_order_book(symbol or "BTC-USD")
    raise ValueError(f"unknown venue: {venue}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_book(venue: str, symbol: str, depth: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash((venue, symbol, "book")) & 0xFFFFFFFF)
    mid = 65000.0 * (1 + rng.normal(0, 0.001))
    tick = 0.5
    bids, asks = [], []
    for i in range(depth):
        bid_p = mid - tick * (i + 1)
        ask_p = mid + tick * (i + 1)
        size = float(np.abs(rng.normal(2.0, 0.8)) * (1 - i / (depth + 5)))
        bids.append((bid_p, max(0.01, size)))
        asks.append((ask_p, max(0.01, size)))
    return _book_from_levels(venue, symbol, bids, asks, "mock")
