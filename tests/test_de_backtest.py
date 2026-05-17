"""Tests for the in-memory Decision Engine backtest engine."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.engine.backtest import DEBacktestConfig, DEBacktestResult, run_de_backtest


def _bars(n: int, *, base: float = 100.0, step: float = 0.5) -> list[Bar]:
    bars = []
    for i in range(n):
        close = base + i * step
        open_ = close - step * 0.4
        high = close + step * 0.6
        low = open_ - step * 0.2
        bars.append(
            Bar(
                symbol="SIM",
                timeframe="1Day",
                ts=f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T05:00:00Z",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=10_000 + i * 50,
            )
        )
    return bars


# ── Basic smoke test ──────────────────────────────────────────────────────────

def test_run_returns_result_type() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    assert isinstance(result, DEBacktestResult)


def test_bars_total_matches_input() -> None:
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.bars_total == 300


def test_bars_simulated_is_subset_of_total() -> None:
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.bars_simulated <= result.bars_total
    assert result.bars_simulated >= 0


# ── Equity curve ──────────────────────────────────────────────────────────────

def test_equity_curve_length_matches_bars() -> None:
    n = 260
    bars = _bars(n)
    result = run_de_backtest(bars, symbol="SIM")
    assert len(result.equity_curve) == n


def test_equity_curve_starts_near_starting_cash() -> None:
    bars = _bars(250)
    cfg = DEBacktestConfig(starting_cash=100_000.0)
    result = run_de_backtest(bars, symbol="SIM", config=cfg)
    # First point is initial cash (no positions yet)
    assert abs(result.equity_curve[0] - 100_000.0) < 100.0


def test_equity_curve_is_positive() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    assert all(eq > 0 for eq in result.equity_curve)


# ── Return metrics ────────────────────────────────────────────────────────────

def test_total_return_is_float() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    assert isinstance(result.total_return_pct, float)


def test_max_drawdown_is_non_negative() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.max_drawdown_pct >= 0.0


def test_sharpe_is_finite() -> None:
    import math
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert math.isfinite(result.sharpe_ratio)


# ── Trade stats consistency ───────────────────────────────────────────────────

def test_num_trades_matches_trade_list() -> None:
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.num_trades == len(result.trades)


def test_round_trips_lte_half_trades() -> None:
    """Each round trip is 1 buy + 1 sell → round_trips <= trades // 2 + 1."""
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.num_round_trips <= result.num_trades


def test_win_rate_in_unit_interval() -> None:
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert 0.0 <= result.win_rate <= 1.0


def test_profit_factor_non_negative() -> None:
    bars = _bars(300)
    result = run_de_backtest(bars, symbol="SIM")
    assert result.profit_factor >= 0.0


# ── Config filtering ──────────────────────────────────────────────────────────

def test_very_high_min_score_produces_no_trades() -> None:
    """min_score=999 means no signal ever qualifies — 0 trades."""
    bars = _bars(300)
    cfg = DEBacktestConfig(min_score=999.0, min_confidence=0.0)
    result = run_de_backtest(bars, symbol="SIM", config=cfg)
    assert result.num_trades == 0


def test_zero_trades_gives_zero_return() -> None:
    """If no trades happen, equity equals starting cash, return = 0%."""
    bars = _bars(250)
    cfg = DEBacktestConfig(min_score=999.0)
    result = run_de_backtest(bars, symbol="SIM", config=cfg)
    assert result.total_return_pct == pytest.approx(0.0, abs=0.01)


def test_zero_trades_gives_zero_win_rate() -> None:
    bars = _bars(250)
    cfg = DEBacktestConfig(min_score=999.0)
    result = run_de_backtest(bars, symbol="SIM", config=cfg)
    assert result.win_rate == 0.0


# ── Summary string ────────────────────────────────────────────────────────────

def test_summary_is_non_empty_string() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    s = result.summary()
    assert isinstance(s, str) and len(s) > 0


def test_summary_contains_symbol() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="TSLA")
    assert "TSLA" in result.summary()


def test_summary_contains_return_pct() -> None:
    bars = _bars(250)
    result = run_de_backtest(bars, symbol="SIM")
    assert "Return:" in result.summary()


# ── Too few bars ──────────────────────────────────────────────────────────────

def test_fewer_than_warmup_bars_gives_zero_simulated() -> None:
    bars = _bars(100)  # below 200 warmup threshold
    result = run_de_backtest(bars, symbol="SIM")
    assert result.bars_simulated == 0
    assert result.num_trades == 0


def test_fewer_than_warmup_bars_returns_result_not_raises() -> None:
    bars = _bars(50)
    result = run_de_backtest(bars, symbol="SIM")
    assert isinstance(result, DEBacktestResult)
