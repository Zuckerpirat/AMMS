from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from amms import db
from amms.backtest import BacktestConfig, run_backtest
from amms.backtest.engine import Trade
from amms.data.bars import Bar
from amms.data.bars import upsert_bars as upsert_bars_fn
from amms.risk import RiskConfig
from amms.strategy import SmaCross


def _seed_bars(conn, symbol: str, opens_closes: list[tuple[str, float, float]]) -> None:
    bars = [
        Bar(
            symbol=symbol,
            timeframe="1Day",
            ts=f"{d}T05:00:00Z",
            open=o,
            high=max(o, c),
            low=min(o, c),
            close=c,
            volume=100,
        )
        for d, o, c in opens_closes
    ]
    upsert_bars_fn(conn, bars)


def _config(
    symbols=("AAPL",),
    start="2025-01-01",
    end="2025-01-31",
    initial_equity=10_000.0,
    fast=3,
    slow=5,
) -> BacktestConfig:
    return BacktestConfig(
        start=date.fromisoformat(start),
        end=date.fromisoformat(end),
        symbols=tuple(symbols),
        initial_equity=initial_equity,
        risk=RiskConfig(
            max_open_positions=5,
            max_position_pct=0.5,  # generous so the test actually trades
            daily_loss_pct=-0.99,
        ),
        strategy=SmaCross(fast=fast, slow=slow),
    )


def test_run_backtest_raises_when_no_bars(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    with pytest.raises(ValueError, match="No bars"):
        run_backtest(_config(), conn)
    conn.close()


def test_run_backtest_buys_on_upward_crossover(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    # Flat then spike → SMA(3) crosses above SMA(5) on the 7th close.
    # Fill of the queued buy happens on the 8th bar's open.
    data = [
        ("2025-01-02", 1.0, 1.0),
        ("2025-01-03", 1.0, 1.0),
        ("2025-01-06", 1.0, 1.0),
        ("2025-01-07", 1.0, 1.0),
        ("2025-01-08", 1.0, 1.0),
        ("2025-01-09", 1.0, 1.0),
        ("2025-01-10", 10.0, 10.0),  # crossover signal at close
        ("2025-01-13", 11.0, 12.0),  # fill at 11.0 open
        ("2025-01-14", 12.0, 13.0),
    ]
    _seed_bars(conn, "AAPL", data)
    result = run_backtest(_config(), conn)
    conn.close()

    buys = [t for t in result.trades if t.side == "buy"]
    assert len(buys) == 1
    assert buys[0].date == "2025-01-13"
    assert buys[0].price == pytest.approx(11.0)
    assert "AAPL" in result.portfolio.positions
    # Final equity should reflect mark-to-market at the last close.
    final_eq = result.equity_curve[-1][1]
    qty = buys[0].qty
    expected = result.config.initial_equity - qty * 11.0 + qty * 13.0
    assert final_eq == pytest.approx(expected)


def test_run_backtest_round_trip_buy_then_sell(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    # Flat, spike up (buy signal), then crash (sell signal).
    data = [
        ("2025-01-02", 1.0, 1.0),
        ("2025-01-03", 1.0, 1.0),
        ("2025-01-06", 1.0, 1.0),
        ("2025-01-07", 1.0, 1.0),
        ("2025-01-08", 1.0, 1.0),
        ("2025-01-09", 1.0, 1.0),
        ("2025-01-10", 10.0, 10.0),  # buy signal at close
        ("2025-01-13", 10.0, 10.0),  # buy fill at 10.0 open
        ("2025-01-14", 10.0, 10.0),
        ("2025-01-15", 10.0, 10.0),
        ("2025-01-16", 10.0, 10.0),
        ("2025-01-17", 10.0, 0.5),  # crash; SMA crosses back down
        ("2025-01-20", 0.4, 0.4),  # sell fill at 0.4 open
    ]
    _seed_bars(conn, "AAPL", data)
    result = run_backtest(_config(), conn)
    conn.close()

    sides = [t.side for t in result.trades]
    assert sides == ["buy", "sell"]
    buy = result.trades[0]
    sell = result.trades[1]
    assert buy.price == pytest.approx(10.0)
    assert sell.price == pytest.approx(0.4)
    assert "AAPL" not in result.portfolio.positions
    # We lost money on the round trip; final equity reflects this.
    assert result.portfolio.cash < result.config.initial_equity


def test_run_backtest_skips_buy_when_cash_too_low(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    data = [
        ("2025-01-02", 1.0, 1.0),
        ("2025-01-03", 1.0, 1.0),
        ("2025-01-06", 1.0, 1.0),
        ("2025-01-07", 1.0, 1.0),
        ("2025-01-08", 1.0, 1.0),
        ("2025-01-09", 1.0, 1.0),
        ("2025-01-10", 10.0, 10.0),
        ("2025-01-13", 10.0, 10.0),
    ]
    _seed_bars(conn, "AAPL", data)
    # 1$ of cash can never afford a 10$ share with 50% sizing.
    result = run_backtest(_config(initial_equity=1.0), conn)
    conn.close()
    assert result.trades == []
    assert result.portfolio.positions == {}


def test_equity_curve_has_one_point_per_bar_date(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "x.sqlite")
    db.migrate(conn)
    data = [
        ("2025-01-02", 1.0, 1.0),
        ("2025-01-03", 1.0, 1.0),
        ("2025-01-06", 1.0, 1.0),
    ]
    _seed_bars(conn, "AAPL", data)
    result = run_backtest(_config(), conn)
    conn.close()
    assert len(result.equity_curve) == 3
    assert [d for d, _ in result.equity_curve] == ["2025-01-02", "2025-01-03", "2025-01-06"]


def test_trade_dataclass_is_immutable() -> None:
    t = Trade(date="2025-01-01", symbol="AAPL", side="buy", qty=1, price=1.0, reason="x")
    with pytest.raises(AttributeError):
        t.price = 2.0  # type: ignore[misc]
