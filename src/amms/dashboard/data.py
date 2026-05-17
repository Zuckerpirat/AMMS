"""Portfolio data for the dashboard.

Pulls live data from the broker when credentials are configured, otherwise
returns a deterministic demo snapshot so the dashboard is usable out of the
box. Reads equity history from the local sqlite snapshots table.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from amms.broker import AlpacaClient
from amms.broker.alpaca import Account, Position
from amms.config import ConfigError, load_settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PositionView:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    equity: float
    cash: float
    buying_power: float
    day_change_abs: float
    day_change_pct: float
    positions: list[PositionView] = field(default_factory=list)
    equity_history: list[tuple[str, float]] = field(default_factory=list)
    is_demo: bool = False
    demo_reason: str = ""

    @property
    def positions_count(self) -> int:
        return len(self.positions)

    @property
    def unrealized_pl(self) -> float:
        return sum(p.unrealized_pl for p in self.positions)

    @property
    def top_winner(self) -> PositionView | None:
        winners = [p for p in self.positions if p.unrealized_pl > 0]
        return max(winners, key=lambda p: p.unrealized_pl, default=None)

    @property
    def top_loser(self) -> PositionView | None:
        losers = [p for p in self.positions if p.unrealized_pl < 0]
        return min(losers, key=lambda p: p.unrealized_pl, default=None)


def _position_view(p: Position) -> PositionView:
    cost = p.avg_entry_price * p.qty
    pct = (p.unrealized_pl / cost * 100.0) if cost else 0.0
    return PositionView(
        symbol=p.symbol,
        qty=p.qty,
        avg_entry_price=p.avg_entry_price,
        market_value=p.market_value,
        unrealized_pl=p.unrealized_pl,
        unrealized_pl_pct=pct,
    )


def _read_equity_history(db_path: Path, days: int = 30) -> list[tuple[str, float]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.execute(
                "SELECT ts, equity FROM equity_snapshots ORDER BY ts DESC LIMIT ?",
                (days,),
            )
            rows = cursor.fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        log.warning("equity_snapshots read failed", exc_info=True)
        return []
    return [(ts, float(eq)) for ts, eq in reversed(rows)]


def _day_change(history: list[tuple[str, float]], current_equity: float) -> tuple[float, float]:
    if len(history) >= 2:
        prev = history[-2][1]
    elif history:
        prev = history[-1][1]
    else:
        return 0.0, 0.0
    abs_change = current_equity - prev
    pct = (abs_change / prev * 100.0) if prev else 0.0
    return abs_change, pct


def _live_snapshot(db_path: Path) -> PortfolioSnapshot:
    settings = load_settings()
    with AlpacaClient(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        settings.alpaca_base_url,
    ) as client:
        account: Account = client.get_account()
        positions = [_position_view(p) for p in client.get_positions()]
    history = _read_equity_history(db_path)
    if not history:
        history = [(date.today().isoformat(), account.equity)]
    day_abs, day_pct = _day_change(history, account.equity)
    return PortfolioSnapshot(
        equity=account.equity,
        cash=account.cash,
        buying_power=account.buying_power,
        day_change_abs=day_abs,
        day_change_pct=day_pct,
        positions=positions,
        equity_history=history,
        is_demo=False,
    )


def _demo_snapshot(reason: str) -> PortfolioSnapshot:
    today = date.today()
    base = 100_000.0
    history: list[tuple[str, float]] = []
    for i in range(30):
        d = today - timedelta(days=29 - i)
        wave = math.sin(i / 5.0) * 1500
        drift = i * 180
        history.append((d.isoformat(), base + drift + wave))
    equity = history[-1][1]
    day_abs = history[-1][1] - history[-2][1]
    day_pct = day_abs / history[-2][1] * 100
    positions = [
        PositionView("AAPL", 25, 175.20, 4690.0, 310.0, 7.07),
        PositionView("MSFT", 12, 410.00, 5160.0, 240.0, 4.88),
        PositionView("NVDA", 8, 880.00, 7280.0, 240.0, 3.41),
        PositionView("TSLA", 15, 245.00, 3525.0, -150.0, -4.08),
        PositionView("GOOGL", 18, 170.00, 3150.0, 90.0, 2.94),
    ]
    return PortfolioSnapshot(
        equity=equity,
        cash=18_450.0,
        buying_power=36_900.0,
        day_change_abs=day_abs,
        day_change_pct=day_pct,
        positions=positions,
        equity_history=history,
        is_demo=True,
        demo_reason=reason,
    )


def get_portfolio(db_path: Path) -> PortfolioSnapshot:
    """Return the current portfolio snapshot, falling back to demo data."""
    try:
        return _live_snapshot(db_path)
    except ConfigError as e:
        return _demo_snapshot(f"Alpaca-Konfiguration fehlt: {e}")
    except Exception as e:  # broker/network failure — still want the UI to work
        log.warning("live snapshot failed, using demo", exc_info=True)
        return _demo_snapshot(f"Live-Daten nicht verfügbar: {e}")
