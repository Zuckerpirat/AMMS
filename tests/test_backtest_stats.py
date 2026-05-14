from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from amms.backtest.engine import BacktestConfig, BacktestResult, Portfolio, Trade
from amms.backtest.stats import (
    compute_stats,
    write_equity_curve_csv,
    write_trades_csv,
)
from amms.risk import RiskConfig
from amms.strategy import SmaCross


def _config(initial: float = 100_000.0) -> BacktestConfig:
    return BacktestConfig(
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
        symbols=("AAPL",),
        initial_equity=initial,
        risk=RiskConfig(),
        strategy=SmaCross(),
    )


def _result(trades, curve, initial=100_000.0) -> BacktestResult:
    return BacktestResult(
        config=_config(initial),
        portfolio=Portfolio(cash=initial),
        trades=trades,
        equity_curve=curve,
    )


def test_compute_stats_flat_equity_is_zero_return() -> None:
    result = _result([], [("2025-01-01", 100_000.0), ("2025-01-02", 100_000.0)])
    stats = compute_stats(result)
    assert stats.total_return_pct == pytest.approx(0.0)
    assert stats.max_drawdown_pct == pytest.approx(0.0)
    assert stats.win_rate == 0.0


def test_compute_stats_pairs_buys_and_sells_fifo_for_win_rate() -> None:
    trades = [
        Trade("2025-01-02", "AAPL", "buy", 1, 100.0, ""),
        Trade("2025-01-05", "AAPL", "sell", 1, 110.0, ""),  # win
        Trade("2025-01-06", "AAPL", "buy", 1, 110.0, ""),
        Trade("2025-01-09", "AAPL", "sell", 1, 100.0, ""),  # loss
        Trade("2025-01-10", "AAPL", "buy", 1, 100.0, ""),
        Trade("2025-01-13", "AAPL", "sell", 1, 120.0, ""),  # win
    ]
    stats = compute_stats(_result(trades, [("2025-01-13", 100_000.0)]))
    assert stats.closed_round_trips == 3
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.num_buys == 3
    assert stats.num_sells == 3


def test_compute_stats_max_drawdown_from_peak() -> None:
    curve = [
        ("2025-01-01", 100_000.0),
        ("2025-01-02", 110_000.0),
        ("2025-01-03", 88_000.0),  # 20% drawdown from peak 110k
        ("2025-01-04", 95_000.0),
    ]
    stats = compute_stats(_result([], curve))
    assert stats.max_drawdown_pct == pytest.approx(-20.0)


def test_compute_stats_total_return() -> None:
    curve = [
        ("2025-01-01", 100_000.0),
        ("2025-01-02", 125_000.0),
    ]
    stats = compute_stats(_result([], curve))
    assert stats.total_return_pct == pytest.approx(25.0)
    assert stats.final_equity == pytest.approx(125_000.0)


def test_write_trades_csv(tmp_path: Path) -> None:
    trades = [
        Trade("2025-01-02", "AAPL", "buy", 10, 100.0, "test"),
        Trade("2025-01-05", "AAPL", "sell", 10, 110.0, "test"),
    ]
    out = tmp_path / "trades.csv"
    n = write_trades_csv(trades, out)
    assert n == 2
    with out.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["date", "symbol", "side", "qty", "price", "reason"]
    assert rows[1][:5] == ["2025-01-02", "AAPL", "buy", "10", "100.0000"]


def test_write_equity_curve_csv(tmp_path: Path) -> None:
    out = tmp_path / "curve.csv"
    n = write_equity_curve_csv([("2025-01-01", 100_000.0), ("2025-01-02", 101_000.5)], out)
    assert n == 2
    with out.open() as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["date", "equity"]
    assert rows[1] == ["2025-01-01", "100000.00"]
    assert rows[2] == ["2025-01-02", "101000.50"]
