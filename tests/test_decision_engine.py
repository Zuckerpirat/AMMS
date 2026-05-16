"""Tests for amms.engine.decision (Central Decision Engine)."""

from __future__ import annotations

import pytest

from amms.engine.decision import DecisionReport, ModuleResult, CategoryScore, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float, volume: float = 1000.0):
        self.open   = open_
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume


def _up_bars(n: int = 200, start: float = 100.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = price + step
        bars.append(_Bar(o, c + 0.3, o - 0.2, c, 1000.0))
        price = c
    return bars


def _down_bars(n: int = 200, start: float = 200.0, step: float = 0.5) -> list[_Bar]:
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = max(1.0, price - step)
        bars.append(_Bar(o, o + 0.2, c - 0.3, c, 1000.0))
        price = c
    return bars


def _flat_bars(n: int = 200, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price, price + 0.5, price - 0.5, price, 500.0) for _ in range(n)]


class TestEdgeCases:
    def test_empty_returns_none(self):
        assert analyze([]) is None

    def test_too_few_bars_returns_none(self):
        assert analyze(_up_bars(30)) is None

    def test_sufficient_bars_returns_report(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert isinstance(result, DecisionReport)

    def test_no_attrs_returns_none(self):
        class _Bad:
            pass
        assert analyze([_Bad()] * 200) is None


class TestAction:
    def test_action_valid_values(self):
        valid = {"strong_buy", "buy", "hold", "sell", "strong_sell"}
        for bars in [_up_bars(200), _down_bars(200), _flat_bars(200)]:
            result = analyze(bars)
            if result:
                assert result.action in valid

    def test_uptrend_not_sell(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert result.action not in {"strong_sell", "sell"}

    def test_downtrend_not_buy(self):
        result = analyze(_down_bars(200))
        assert result is not None
        assert result.action not in {"strong_buy", "buy"}


class TestScores:
    def test_composite_score_in_range(self):
        for bars in [_up_bars(200), _down_bars(200), _flat_bars(200)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.composite_score <= 100.0

    def test_confidence_in_range(self):
        for bars in [_up_bars(200), _down_bars(200)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.confidence <= 1.0

    def test_uptrend_positive_score(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert result.composite_score > 0

    def test_downtrend_negative_score(self):
        result = analyze(_down_bars(200))
        assert result is not None
        assert result.composite_score < 0


class TestCategories:
    def test_categories_present(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert len(result.categories) > 0
        for cs in result.categories.values():
            assert isinstance(cs, CategoryScore)

    def test_category_scores_in_range(self):
        result = analyze(_up_bars(200))
        assert result is not None
        for cs in result.categories.values():
            assert -100.0 <= cs.score <= 100.0

    def test_category_module_count_positive(self):
        result = analyze(_up_bars(200))
        assert result is not None
        for cs in result.categories.values():
            assert cs.module_count > 0


class TestModules:
    def test_modules_run_positive(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert result.modules_run > 0

    def test_modules_list_matches_count(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert len(result.modules) == result.modules_run

    def test_module_scores_in_range(self):
        result = analyze(_up_bars(200))
        assert result is not None
        for m in result.modules:
            assert -100.0 <= m.score <= 100.0
            assert isinstance(m, ModuleResult)

    def test_modules_failed_non_negative(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert result.modules_failed >= 0


class TestRiskGate:
    def test_risk_blocked_is_bool(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert isinstance(result.risk_blocked, bool)

    def test_risk_reason_string(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert isinstance(result.risk_reason, str)

    def test_high_min_confidence_blocks(self):
        # Setting min_confidence=1.01 (impossible) should always block
        result = analyze(_up_bars(200), min_confidence=1.01)
        assert result is not None
        assert result.risk_blocked is True
        assert result.action == "hold"

    def test_risk_veto_blocks(self):
        def veto(_score, _conf):
            return "drawdown limit hit"
        result = analyze(_up_bars(200), risk_veto=veto)
        assert result is not None
        assert result.risk_blocked is True
        assert "drawdown limit hit" in result.risk_reason
        assert result.action == "hold"

    def test_risk_veto_none_allows(self):
        def veto(_score, _conf):
            return None
        result = analyze(_up_bars(200), risk_veto=veto)
        assert result is not None
        # Veto returning None should not block (other gates still apply)
        assert "External veto" not in result.risk_reason


class TestReasoning:
    def test_reasoning_list_present(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert isinstance(result.reasoning, list)
        assert len(result.reasoning) > 0

    def test_reasoning_strings(self):
        result = analyze(_up_bars(200))
        assert result is not None
        for r in result.reasoning:
            assert isinstance(r, str)


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(200), symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_bars_used(self):
        bars = _up_bars(200)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 200

    def test_verdict_present(self):
        result = analyze(_up_bars(200))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_symbol(self):
        result = analyze(_up_bars(200), symbol="TSLA")
        assert result is not None
        assert "TSLA" in result.verdict
