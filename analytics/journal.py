"""
btc_dashboard.analytics.journal
Trade journal — strategies, trades, P&L, statistics.

Pure logic. The page layer (pages/positions.py) handles persistence and UI.

Concepts
--------
A *Strategy* is a named playbook with rules text and starting capital.
A *Trade* belongs to one strategy and to one of three instruments:

    - futures   leveraged perp (size in BTC + leverage multiplier)
    - spot      unleveraged or 1× margin (size in BTC)
    - options   bought / sold premium (contracts × premium per contract)

P&L is computed in USD. For futures and spot we use price moves on size;
for options we use premium moves on contracts. Each trade carries a
status (open / closed-tp / closed-sl / closed-manual) and the close
price/premium.

Statistics over a strategy's closed trades:

    n_trades, n_wins, n_losses, win_rate, total_pnl_usd, avg_pnl_usd,
    avg_win, avg_loss, profit_factor, expectancy, best_trade, worst_trade,
    average_r_multiple
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────
INSTRUMENTS = ("futures", "spot", "options")
OPT_TYPES = ("call", "put")
DIRECTIONS = ("long", "short")
TRADE_STATUSES = ("open", "closed_tp", "closed_sl", "closed_manual")


@dataclass
class Strategy:
    id: str
    name: str
    description: str = ""
    rules: str = ""                    # plain-text playbook
    capital: float = 100_000.0         # starting equity ($)
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class Trade:
    id: str
    strategy_id: str
    instrument: str                    # "futures" | "spot" | "options"
    direction: str                     # "long" | "short"

    # Entry
    open_at: datetime
    open_price: float                  # underlying price at open (futures/spot/options reference)
    size: float                        # BTC for futures/spot · contracts for options
    leverage: float = 1.0              # 1.0 for spot, X for futures, 1.0 for options

    # Risk levels (in price terms — futures/spot) or premium terms (options)
    tp: Optional[float] = None
    sl: Optional[float] = None

    # Options-only
    option_type: Optional[str] = None  # "call" | "put"
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    open_premium: Optional[float] = None     # USD per contract at open
    close_premium: Optional[float] = None    # USD per contract at close

    # Closure
    close_at: Optional[datetime] = None
    close_price: Optional[float] = None
    status: str = "open"

    # Journal context
    bias: Optional[str] = None
    setup: Optional[str] = None
    conviction: Optional[str] = None
    entry_score_4h: Optional[float] = None
    entry_rn_mean: Optional[float] = None
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# P&L
# ─────────────────────────────────────────────────────────────────────────────
def trade_pnl_usd(trade: Trade, mark: Optional[float] = None) -> float:
    """Return P&L in USD.

    `mark` is optional. For closed trades, P&L is computed from `close_price`
    (or `close_premium` for options). For open trades, pass `mark` to get a
    live unrealised P&L.
    """
    sign = +1 if trade.direction == "long" else -1

    if trade.instrument == "options":
        open_p = float(trade.open_premium or 0.0)
        close_p = (
            float(trade.close_premium)
            if trade.close_premium is not None
            else (float(mark) if mark is not None else open_p)
        )
        contracts = float(trade.size)
        return sign * (close_p - open_p) * contracts

    # futures / spot
    open_x = float(trade.open_price or 0.0)
    close_x = (
        float(trade.close_price)
        if trade.close_price is not None
        else (float(mark) if mark is not None else open_x)
    )
    return sign * (close_x - open_x) * float(trade.size)


def trade_margin_usd(trade: Trade) -> float:
    """Approximate USD margin posted at entry. Used to express returns as a
    percentage of capital-at-risk."""
    if trade.instrument == "options":
        return float((trade.open_premium or 0.0) * trade.size)
    notional = float(trade.open_price) * float(trade.size)
    if trade.leverage and trade.leverage > 0:
        return notional / float(trade.leverage)
    return notional


def trade_pnl_pct(trade: Trade, mark: Optional[float] = None) -> float:
    """P&L as a fraction of margin. 0.10 = +10%."""
    margin = trade_margin_usd(trade)
    if margin <= 0:
        return 0.0
    return trade_pnl_usd(trade, mark=mark) / margin


def trade_r_multiple(trade: Trade) -> Optional[float]:
    """Realised R: P&L divided by initial planned risk (entry → SL distance).
    Only defined when an SL was set and the trade is closed."""
    if trade.status == "open" or trade.sl is None:
        return None

    if trade.instrument == "options":
        open_p = float(trade.open_premium or 0.0)
        risk = abs(float(trade.sl) - open_p) * float(trade.size)
    else:
        risk = abs(float(trade.sl) - float(trade.open_price)) * float(trade.size)

    if risk <= 0:
        return None
    return trade_pnl_usd(trade) / risk


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyStats:
    n_trades: int
    n_open: int
    n_closed: int
    n_wins: int
    n_losses: int
    win_rate: float                # 0..1, over closed trades
    total_pnl_usd: float
    avg_pnl_usd: float
    avg_win_usd: float
    avg_loss_usd: float             # negative
    best_trade_usd: float
    worst_trade_usd: float
    profit_factor: float           # sum_wins / |sum_losses|
    expectancy_usd: float           # avg_pnl per trade
    avg_r_multiple: Optional[float]
    pnl_curve: list[tuple[datetime, float]]   # cumulative pnl after each closed trade


def compute_strategy_stats(strategy: Strategy, trades: list[Trade]) -> StrategyStats:
    """Aggregate closed-trade statistics. Open trades are counted but not
    included in P&L sums (their final pnl isn't known yet)."""
    own = [t for t in trades if t.strategy_id == strategy.id]
    closed = [t for t in own if t.status != "open"]
    closed_sorted = sorted(
        closed,
        key=lambda t: t.close_at or t.open_at,
    )

    pnls = [trade_pnl_usd(t) for t in closed_sorted]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    n_wins = len(wins)
    n_losses = len(losses)
    n_closed = len(closed_sorted)

    total = float(sum(pnls)) if pnls else 0.0
    avg_pnl = (total / n_closed) if n_closed else 0.0
    avg_win = (sum(wins) / n_wins) if n_wins else 0.0
    avg_loss = (sum(losses) / n_losses) if n_losses else 0.0
    best = max(pnls) if pnls else 0.0
    worst = min(pnls) if pnls else 0.0

    sum_w = float(sum(wins))
    sum_l = float(abs(sum(losses)))
    profit_factor = (sum_w / sum_l) if sum_l > 0 else float("inf") if sum_w > 0 else 0.0
    win_rate = (n_wins / n_closed) if n_closed else 0.0

    rs = [trade_r_multiple(t) for t in closed_sorted]
    rs = [r for r in rs if r is not None]
    avg_r = (sum(rs) / len(rs)) if rs else None

    cumulative = 0.0
    curve: list[tuple[datetime, float]] = []
    for t, p in zip(closed_sorted, pnls):
        cumulative += float(p)
        ts = t.close_at or t.open_at
        curve.append((ts, cumulative))

    return StrategyStats(
        n_trades=len(own),
        n_open=len([t for t in own if t.status == "open"]),
        n_closed=n_closed,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        total_pnl_usd=total,
        avg_pnl_usd=avg_pnl,
        avg_win_usd=avg_win,
        avg_loss_usd=avg_loss,
        best_trade_usd=best,
        worst_trade_usd=worst,
        profit_factor=profit_factor,
        expectancy_usd=avg_pnl,
        avg_r_multiple=avg_r,
        pnl_curve=curve,
    )


def by_instrument_breakdown(strategy: Strategy, trades: list[Trade]) -> dict[str, dict]:
    """Per-instrument summary: count, win rate, total P&L, avg P&L."""
    own = [t for t in trades if t.strategy_id == strategy.id]
    out: dict[str, dict] = {}
    for inst in INSTRUMENTS:
        sub = [t for t in own if t.instrument == inst and t.status != "open"]
        pnls = [trade_pnl_usd(t) for t in sub]
        n = len(sub)
        n_wins = sum(1 for p in pnls if p > 0)
        out[inst] = {
            "n": n,
            "wins": n_wins,
            "win_rate": (n_wins / n) if n else 0.0,
            "total_pnl_usd": float(sum(pnls)) if pnls else 0.0,
            "avg_pnl_usd": (float(sum(pnls)) / n) if n else 0.0,
        }
    return out
