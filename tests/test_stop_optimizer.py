from __future__ import annotations

import pytest

from amms.analysis.stop_optimizer import StopSuggestion, suggest_stop
from amms.data.bars import Bar


def _bar(i: int, close: float, high_add: float = 1.5, low_sub: float = 1.5) -> Bar:
    return Bar(
        "X", "1D",
        f"2026-01-{1 + i % 28:02d}",
        close, close + high_add, close - low_sub, close, 1000.0
    )


def _bars(n: int = 25, price: float = 100.0) -> list[Bar]:
    return [_bar(i, price + i * 0.1) for i in range(n)]


def test_suggest_stop_empty_bars() -> None:
    s = suggest_stop("X", [])
    assert s.atr_pct is None
    assert "No bar data" in s.recommendation


def test_suggest_stop_insufficient_history() -> None:
    bars = _bars(5)  # fewer than 15 needed for ATR-14
    s = suggest_stop("X", bars)
    assert s.atr_pct is None
    assert "Insufficient" in s.recommendation


def test_suggest_stop_returns_percentages() -> None:
    bars = _bars(25)
    s = suggest_stop("X", bars)
    assert s.atr_pct is not None
    assert s.stop_tight_pct is not None
    assert s.stop_balanced_pct is not None
    assert s.stop_wide_pct is not None
    # balanced = 1.5 × tight, wide = 2 × tight
    assert s.stop_balanced_pct == pytest.approx(s.stop_tight_pct * 1.5, rel=0.05)
    assert s.stop_wide_pct == pytest.approx(s.stop_tight_pct * 2.0, rel=0.05)


def test_stop_wide_greater_than_tight() -> None:
    bars = _bars(25)
    s = suggest_stop("X", bars)
    if s.stop_tight_pct and s.stop_wide_pct:
        assert s.stop_wide_pct > s.stop_tight_pct


def test_suggest_stop_includes_recommendation() -> None:
    bars = _bars(25)
    s = suggest_stop("X", bars)
    assert isinstance(s.recommendation, str)
    assert len(s.recommendation) > 0


def test_suggest_stop_current_price() -> None:
    bars = _bars(25, price=200.0)
    s = suggest_stop("X", bars)
    assert s.current_price == pytest.approx(bars[-1].close)


def test_suggest_stop_high_vol_wide_recommendation() -> None:
    # High volatility bars (big high-low swings)
    bars = [_bar(i, 100.0 + i * 0.1, high_add=5.0, low_sub=5.0) for i in range(25)]
    s = suggest_stop("X", bars)
    if s.atr_pct and s.atr_pct > 2.5:
        assert "wide" in s.recommendation.lower() or "high" in s.recommendation.lower()
