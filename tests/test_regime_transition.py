"""Tests for amms.analysis.regime_transition."""

from __future__ import annotations

import pytest

from amms.analysis.regime_transition import RegimeSnapshot, RegimeTransitionReport, analyze


class _Bar:
    def __init__(self, close: float, spread: float = 1.0):
        self.close = close
        self.high = close + spread
        self.low = close - spread


def _uptrend(n: int = 70) -> list[_Bar]:
    price = 100.0
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price += 0.5
    return bars


def _downtrend(n: int = 70) -> list[_Bar]:
    price = 200.0
    bars = []
    for _ in range(n):
        bars.append(_Bar(price))
        price -= 0.5
    return bars


def _flat(n: int = 70, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price)] * n


def _transition_bars(n: int = 80) -> list[_Bar]:
    """Bull first half, then sharp drop → bear."""
    bars = []
    price = 100.0
    for i in range(n):
        if i < n // 2:
            price += 1.0
        else:
            price -= 2.0
        bars.append(_Bar(max(price, 1.0)))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        bars = _flat(30)
        assert analyze(bars) is None

    def test_returns_result_enough_bars(self):
        bars = _flat(60)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, RegimeTransitionReport)


class TestCurrentRegime:
    def test_regime_valid(self):
        for bars in [_uptrend(70), _downtrend(70), _flat(70)]:
            result = analyze(bars)
            if result:
                assert result.current_regime in {"bull", "neutral", "bear"}

    def test_uptrend_bullish_regime(self):
        bars = _uptrend(100)
        result = analyze(bars)
        assert result is not None
        assert result.current_regime in {"bull", "neutral"}

    def test_downtrend_bearish_regime(self):
        bars = _downtrend(100)
        result = analyze(bars)
        assert result is not None
        assert result.current_regime in {"bear", "neutral"}


class TestVote:
    def test_vote_in_range(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert -4 <= result.current_vote <= 4


class TestHistory:
    def test_history_not_empty(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert len(result.history) > 0

    def test_history_items_are_snapshots(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        for s in result.history:
            assert isinstance(s, RegimeSnapshot)

    def test_history_regimes_valid(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        for s in result.history:
            assert s.regime in {"bull", "neutral", "bear"}


class TestTransition:
    def test_transition_detected_is_bool(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.transition_detected, bool)

    def test_direction_valid(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert "stable" in result.transition_direction or "bull" in result.transition_direction or "bear" in result.transition_direction or "neutral" in result.transition_direction

    def test_stable_when_no_change(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        # Flat = no strong directional signal → should be stable
        assert result.transition_direction in {"stable", "to_neutral"}


class TestIndicators:
    def test_rsi_in_range(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.rsi <= 100.0

    def test_current_price_positive(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert result.current_price > 0

    def test_atr_percentile_in_range(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.atr_percentile <= 100.0


class TestMetadata:
    def test_bars_used_correct(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 70

    def test_symbol_stored(self):
        bars = _flat(70)
        result = analyze(bars, symbol="QQQ")
        assert result is not None
        assert result.symbol == "QQQ"


class TestVerdict:
    def test_verdict_present(self):
        bars = _flat(70)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_regime(self):
        bars = _uptrend(70)
        result = analyze(bars)
        assert result is not None
        assert "regime" in result.verdict.lower()
