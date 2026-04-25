"""
btc_dashboard.data.options
Options-chain connectors. One DataFrame per venue — no cross-venue merging.

Public API
----------
- fetch_deribit_option_chain(currency)  -> OPTION_CHAIN_SCHEMA
- fetch_binance_option_chain(symbol)    -> OPTION_CHAIN_SCHEMA
- load_option_chain(venue, ...)         -> dispatches per venue
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
# SCHEMA
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Schema:
    name: str
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    description: str


OPTION_CHAIN_SCHEMA = Schema(
    name="option_chain",
    columns=(
        "venue", "instrument", "underlying", "type", "strike", "expiry",
        "dte", "bid", "ask", "mark", "iv", "open_interest", "volume",
        "_source",
    ),
    dtypes={
        "venue": "string",
        "instrument": "string",
        "underlying": "string",
        "type": "string",            # "CALL" | "PUT"
        "strike": "float64",
        "expiry": "datetime64[ns, UTC]",
        "dte": "float64",
        "bid": "float64",            # in base asset (e.g. BTC) for Deribit; quote for Binance
        "ask": "float64",
        "mark": "float64",
        "iv": "float64",             # implied vol in percent (e.g. 65.4)
        "open_interest": "float64",
        "volume": "float64",
        "_source": "string",
    },
    description="Per-instrument option-chain snapshot. Quote conventions are venue-specific — do not aggregate naively.",
)


def _conform(df: pd.DataFrame) -> pd.DataFrame:
    for col in OPTION_CHAIN_SCHEMA.columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype=OPTION_CHAIN_SCHEMA.dtypes.get(col, "object"))
    return df[list(OPTION_CHAIN_SCHEMA.columns)]


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
class DeribitOptionsClient(BaseAPIClient):
    base_url = settings.deribit_base_url


class BinanceOptionsClient(BaseAPIClient):
    base_url = "https://eapi.binance.com"


# ─────────────────────────────────────────────────────────────────────────────
# DERIBIT
# ─────────────────────────────────────────────────────────────────────────────
def fetch_deribit_option_chain(currency: str = "BTC", max_dte: int = 90) -> pd.DataFrame:
    try:
        with DeribitOptionsClient() as cli:
            inst_payload = cli._get(
                "/api/v2/public/get_instruments",
                params={"currency": currency, "kind": "option", "expired": "false"},
            )
            book_payload = cli._get(
                "/api/v2/public/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
            )
        instruments = inst_payload.get("result") or []
        book = book_payload.get("result") or []
        if not instruments or not book:
            raise APIError("empty response")
        by_name = {b["instrument_name"]: b for b in book}

        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(days=max_dte)
        rows = []
        for inst in instruments:
            exp_ms = int(inst.get("expiration_timestamp", 0))
            exp_dt = datetime.fromtimestamp(exp_ms / 1000, tz=timezone.utc)
            if exp_dt > cutoff:
                continue
            name = inst["instrument_name"]
            b = by_name.get(name, {})
            rows.append(
                {
                    "venue": "deribit",
                    "instrument": name,
                    "underlying": currency,
                    "type": str(inst.get("option_type", "")).upper(),
                    "strike": float(inst.get("strike", 0)),
                    "expiry": pd.Timestamp(exp_dt),
                    "dte": max(0.0, (exp_dt - now).total_seconds() / 86400.0),
                    "bid": float(b.get("bid_price") or 0.0),
                    "ask": float(b.get("ask_price") or 0.0),
                    "mark": float(b.get("mark_price") or 0.0),
                    "iv": float(b.get("mark_iv") or 0.0),
                    "open_interest": float(b.get("open_interest") or 0.0),
                    "volume": float(b.get("volume") or 0.0),
                    "_source": "live",
                }
            )
        if not rows:
            raise APIError("no rows after filtering")
        return _conform(pd.DataFrame(rows).sort_values(["dte", "strike"]).reset_index(drop=True))
    except Exception as exc:
        log.warning("fetch_deribit_option_chain failed (%s) — using mock", exc)
        return _mock_chain(venue="deribit", underlying=currency)


# ─────────────────────────────────────────────────────────────────────────────
# BINANCE  (European-style options on eapi.binance.com)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_binance_option_chain(underlying: str = "BTCUSDT", max_dte: int = 90) -> pd.DataFrame:
    try:
        with BinanceOptionsClient() as cli:
            info = cli._get("/eapi/v1/exchangeInfo")
            mark = cli._get("/eapi/v1/mark")
        symbols = (info.get("optionSymbols") if isinstance(info, dict) else None) or []
        marks = mark if isinstance(mark, list) else []
        marks_by_sym = {m.get("symbol"): m for m in marks}

        now = datetime.now(tz=timezone.utc)
        cutoff = now + timedelta(days=max_dte)
        rows = []
        for s in symbols:
            if s.get("underlying") != underlying:
                continue
            try:
                strike = float(s.get("strikePrice", 0))
                exp_ms = int(s.get("expiryDate", 0))
            except Exception:
                continue
            exp_dt = datetime.fromtimestamp(exp_ms / 1000, tz=timezone.utc)
            if exp_dt > cutoff:
                continue
            sym = s.get("symbol")
            m = marks_by_sym.get(sym, {})
            rows.append(
                {
                    "venue": "binance",
                    "instrument": sym,
                    "underlying": underlying,
                    "type": str(s.get("side", "")).upper(),
                    "strike": strike,
                    "expiry": pd.Timestamp(exp_dt),
                    "dte": max(0.0, (exp_dt - now).total_seconds() / 86400.0),
                    "bid": float(m.get("bidIV") or 0.0) * 0.0,  # bidIV present, not bid price
                    "ask": float(m.get("askIV") or 0.0) * 0.0,
                    "mark": float(m.get("markPrice") or 0.0),
                    "iv": float(m.get("markIV") or 0.0) * 100.0,  # decimal → %
                    "open_interest": 0.0,
                    "volume": 0.0,
                    "_source": "live",
                }
            )
        if not rows:
            raise APIError("no rows after filtering")
        return _conform(pd.DataFrame(rows).sort_values(["dte", "strike"]).reset_index(drop=True))
    except Exception as exc:
        log.warning("fetch_binance_option_chain failed (%s) — using mock", exc)
        return _mock_chain(venue="binance", underlying=underlying)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
def load_option_chain(
    venue: str, underlying: Optional[str] = None, max_dte: int = 90
) -> pd.DataFrame:
    venue = venue.lower()
    if venue == "deribit":
        return fetch_deribit_option_chain(currency=underlying or "BTC", max_dte=max_dte)
    if venue == "binance":
        return fetch_binance_option_chain(underlying=underlying or "BTCUSDT", max_dte=max_dte)
    raise ValueError(f"unknown venue: {venue}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def _mock_chain(venue: str, underlying: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed=hash((venue, underlying, "chain")) & 0xFFFFFFFF)
    spot = 65000.0
    now = datetime.now(tz=timezone.utc)
    expiries = [now + timedelta(days=d) for d in [1, 7, 14, 30, 60]]
    strikes = [spot * (1 + p) for p in np.arange(-0.20, 0.21, 0.05)]
    rows = []
    for exp_dt in expiries:
        dte = (exp_dt - now).total_seconds() / 86400.0
        atm_iv = 60 + 8 * np.exp(-dte / 30)  # short-dated rich
        for k in strikes:
            for opt in ("CALL", "PUT"):
                moneyness = (k - spot) / spot
                iv = atm_iv + 8 * abs(moneyness)  # smile
                mark = max(0.0001, 0.04 * np.exp(-(moneyness ** 2) * 30) * (dte / 30 + 0.2))
                rows.append(
                    {
                        "venue": venue,
                        "instrument": f"MOCK-{underlying}-{int(k)}-{opt[0]}-{int(dte)}d",
                        "underlying": underlying,
                        "type": opt,
                        "strike": float(k),
                        "expiry": pd.Timestamp(exp_dt),
                        "dte": float(dte),
                        "bid": float(mark * (1 - 0.02)),
                        "ask": float(mark * (1 + 0.02)),
                        "mark": float(mark),
                        "iv": float(iv),
                        "open_interest": float(rng.uniform(10, 300)),
                        "volume": float(rng.uniform(0, 50)),
                        "_source": "mock",
                    }
                )
    return _conform(pd.DataFrame(rows).sort_values(["dte", "strike"]).reset_index(drop=True))
