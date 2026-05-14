from __future__ import annotations

from datetime import date as Date
from pathlib import Path

from amms import db
from amms.backtest import BacktestConfig, run_intraday_backtest
from amms.data.bars import Bar
from amms.data.bars import upsert_bars as upsert_bars_fn
from amms.risk import RiskConfig
from amms.strategy.base import Signal


class _BuyOnceStrategy:
    name = "buy_once"
    _emitted = False

    @property
    def lookback(self) -> int:
        return 1

    def evaluate(self, symbol, bars):
        if self._emitted:
            return Signal(symbol, "hold", "", bars[-1].close)
        self._emitted = True
        return Signal(symbol, "buy", "test", bars[-1].close, score=1.0)


def test_intraday_backtest_steps_per_bar(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    # Four 5Min bars on the same day, prices 10, 11, 12, 13.
    rows = [
        Bar("AAPL", "5Min", "2025-01-02T14:30:00Z", 10, 10, 10, 10, 100),
        Bar("AAPL", "5Min", "2025-01-02T14:35:00Z", 11, 11, 11, 11, 100),
        Bar("AAPL", "5Min", "2025-01-02T14:40:00Z", 12, 12, 12, 12, 100),
        Bar("AAPL", "5Min", "2025-01-02T14:45:00Z", 13, 13, 13, 13, 100),
    ]
    upsert_bars_fn(conn, rows)

    cfg = BacktestConfig(
        start=Date(2025, 1, 1),
        end=Date(2025, 1, 31),
        symbols=("AAPL",),
        initial_equity=1000,
        risk=RiskConfig(max_position_pct=0.5),
        strategy=_BuyOnceStrategy(),
        timeframe="5Min",
    )
    result = run_intraday_backtest(cfg, conn)
    conn.close()

    # Equity curve should have one point per bar timestamp (4), not just 1
    # per calendar date — that's the whole reason we have this engine.
    assert len(result.equity_curve) == 4
    # And the buy should have filled at the bar after the buy signal (open=11).
    assert any(t.price == 11.0 for t in result.trades if t.side == "buy")
