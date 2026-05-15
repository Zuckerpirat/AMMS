from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from amms.backtest.engine import BacktestConfig, BacktestResult, Portfolio, Trade
from amms.backtest.stats import (
    BacktestStats,
    ExtendedStats,
    compute_extended_stats,
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


def _make_result_with_trades(equity_curve, trades_def):
    """Helper: build BacktestResult from a list of (symbol, side, price, qty) tuples."""
    trades = [
        Trade(date="2026-01-01", symbol=sym, side=side, price=price, qty=qty, reason="test")
        for sym, side, price, qty in trades_def
    ]
    return _result(trades, equity_curve)


def test_profit_factor_mixed():
    """2 wins ($200 each), 1 loss ($100) → profit_factor = 2.0"""
    result = _make_result_with_trades(
        equity_curve=[("d1", 100_000), ("d2", 101_000), ("d3", 99_000)],
        trades_def=[
            ("AAPL", "buy",  100.0, 10),
            ("AAPL", "sell", 120.0, 10),  # +200 win
            ("MSFT", "buy",  100.0, 10),
            ("MSFT", "sell",  90.0, 10),  # -100 loss
        ],
    )
    stats = compute_stats(result)
    assert stats.closed_round_trips == 2
    assert stats.win_rate == 0.5
    assert abs(stats.profit_factor - 2.0) < 0.01
    assert abs(stats.avg_win - 200.0) < 0.01
    assert abs(stats.avg_loss - 100.0) < 0.01


def test_sharpe_computed_from_equity_curve():
    equity = [
        ("2026-01-01", 100_000),
        ("2026-01-02", 101_000),
        ("2026-01-03", 99_500),
        ("2026-01-04", 102_000),
        ("2026-01-05", 100_800),
    ]
    result = _make_result_with_trades(equity_curve=equity, trades_def=[])
    stats = compute_stats(result)
    assert isinstance(stats.sharpe, float)


def test_extended_stats_returns_instance():
    equity = [("2026-01-01", 100_000), ("2026-01-05", 105_000), ("2026-01-10", 102_000)]
    trades = [
        Trade("2026-01-02", "AAPL", "buy",  10, 100.0, ""),
        Trade("2026-01-04", "AAPL", "sell", 10, 105.0, ""),  # win
        Trade("2026-01-06", "AAPL", "buy",  10, 105.0, ""),
        Trade("2026-01-09", "AAPL", "sell", 10,  95.0, ""),  # loss
    ]
    result = _result(trades, equity)
    base = compute_stats(result)
    ext = compute_extended_stats(result, base)
    assert isinstance(ext, ExtendedStats)


def test_extended_stats_consec_wins_losses():
    equity = [("2026-01-01", 100_000), ("2026-01-10", 110_000)]
    trades = [
        Trade("d1", "AAPL", "buy",  1, 100.0, ""),
        Trade("d2", "AAPL", "sell", 1, 110.0, ""),  # win
        Trade("d3", "AAPL", "buy",  1, 100.0, ""),
        Trade("d4", "AAPL", "sell", 1, 110.0, ""),  # win
        Trade("d5", "MSFT", "buy",  1, 100.0, ""),
        Trade("d6", "MSFT", "sell", 1,  90.0, ""),  # loss
    ]
    result = _result(trades, equity)
    base = compute_stats(result)
    ext = compute_extended_stats(result, base)
    assert ext.max_consec_wins == 2
    assert ext.max_consec_losses == 1


def test_extended_stats_calmar_positive_return():
    equity = [
        ("2026-01-01", 100_000),
        ("2026-01-02", 105_000),
        ("2026-01-03", 100_000),  # 5% dd from peak
        ("2026-01-04", 110_000),
    ]
    result = _result([], equity)
    base = compute_stats(result)
    ext = compute_extended_stats(result, base)
    # Calmar = annualized_return / max_dd; max_dd > 0, return > 0
    assert isinstance(ext.calmar_ratio, float)


def test_extended_stats_payoff_ratio():
    equity = [("2026-01-01", 100_000), ("2026-01-10", 105_000)]
    trades = [
        Trade("d1", "AAPL", "buy",  1, 100.0, ""),
        Trade("d2", "AAPL", "sell", 1, 120.0, ""),  # +20 win
        Trade("d3", "MSFT", "buy",  1, 100.0, ""),
        Trade("d4", "MSFT", "sell", 1,  95.0, ""),  # -5 loss
    ]
    result = _result(trades, equity)
    base = compute_stats(result)
    ext = compute_extended_stats(result, base)
    # avg_win=20, avg_loss=5 → payoff=4.0
    assert ext.payoff_ratio == pytest.approx(4.0, abs=0.01)


def test_extended_stats_no_trades():
    equity = [("2026-01-01", 100_000), ("2026-01-02", 100_000)]
    result = _result([], equity)
    base = compute_stats(result)
    ext = compute_extended_stats(result, base)
    assert ext.max_consec_wins == 0
    assert ext.max_consec_losses == 0
    assert ext.expectancy == 0.0


def test_new_stats_fields_present():
    stats = BacktestStats(
        initial_equity=100_000,
        final_equity=110_000,
        total_return_pct=10.0,
        num_trades=10,
        num_buys=5,
        num_sells=5,
        closed_round_trips=5,
        win_rate=0.6,
        max_drawdown_pct=-5.0,
        sharpe=1.5,
        profit_factor=2.0,
        avg_win=300.0,
        avg_loss=150.0,
    )
    assert stats.sharpe == 1.5
    assert stats.profit_factor == 2.0
    assert stats.avg_win == 300.0
    assert stats.avg_loss == 150.0
