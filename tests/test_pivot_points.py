"""Tests for amms.analysis.pivot_points."""

from __future__ import annotations

import pytest

from amms.analysis.pivot_points import PivotLevel, PivotPointReport, compute


H, L, C = 110.0, 90.0, 100.0  # range = 20, PP = 100.0


class TestEdgeCases:
    def test_returns_none_zero_high(self):
        assert compute(0, L, C, 100.0) is None

    def test_returns_none_invalid(self):
        assert compute(None, L, C, 100.0) is None

    def test_returns_none_high_lt_low(self):
        assert compute(80.0, 100.0, 90.0, 95.0) is None

    def test_returns_result_classic(self):
        result = compute(H, L, C, 100.0, method="classic")
        assert result is not None
        assert isinstance(result, PivotPointReport)

    def test_returns_result_fibonacci(self):
        result = compute(H, L, C, 100.0, method="fibonacci")
        assert result is not None

    def test_returns_result_camarilla(self):
        result = compute(H, L, C, 100.0, method="camarilla")
        assert result is not None


class TestClassicLevels:
    def test_pivot_is_hlc3(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        assert result.pivot == pytest.approx((H + L + C) / 3, abs=0.01)

    def test_classic_has_7_levels(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        assert len(result.levels) == 7

    def test_r1_formula(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        pp = (H + L + C) / 3
        r1 = 2 * pp - L
        r1_level = next(lv for lv in result.levels if lv.name == "R1")
        assert r1_level.price == pytest.approx(r1, abs=0.01)

    def test_s1_formula(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        pp = (H + L + C) / 3
        s1 = 2 * pp - H
        s1_level = next(lv for lv in result.levels if lv.name == "S1")
        assert s1_level.price == pytest.approx(s1, abs=0.01)

    def test_resistances_above_pivot(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        pp = result.pivot
        for lv in result.levels:
            if lv.kind == "R":
                assert lv.price > pp

    def test_supports_below_pivot(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        pp = result.pivot
        for lv in result.levels:
            if lv.kind == "S":
                assert lv.price < pp

    def test_levels_sorted_ascending(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        prices = [lv.price for lv in result.levels]
        assert prices == sorted(prices)


class TestFibonacciLevels:
    def test_fibonacci_has_7_levels(self):
        result = compute(H, L, C, 100.0, method="fibonacci")
        assert result is not None
        assert len(result.levels) == 7

    def test_fib_r1_formula(self):
        result = compute(H, L, C, 100.0, method="fibonacci")
        assert result is not None
        pp = (H + L + C) / 3
        r1 = pp + 0.382 * (H - L)
        r1_level = next(lv for lv in result.levels if lv.name == "R1")
        assert r1_level.price == pytest.approx(r1, abs=0.01)


class TestCamarilla:
    def test_camarilla_has_9_levels(self):
        result = compute(H, L, C, 100.0, method="camarilla")
        assert result is not None
        assert len(result.levels) == 9

    def test_camarilla_r4_formula(self):
        result = compute(H, L, C, 100.0, method="camarilla")
        assert result is not None
        r4 = C + (H - L) * 1.1 / 2
        r4_level = next(lv for lv in result.levels if lv.name == "R4")
        assert r4_level.price == pytest.approx(r4, abs=0.01)


class TestPriceZone:
    def test_above_pivot_is_bullish(self):
        pp = (H + L + C) / 3
        result = compute(H, L, C, pp + 5.0)
        assert result is not None
        assert "bullish" in result.current_zone.lower()

    def test_below_pivot_is_bearish(self):
        pp = (H + L + C) / 3
        result = compute(H, L, C, pp - 5.0)
        assert result is not None
        assert "bearish" in result.current_zone.lower()

    def test_nearest_resistance_above_price(self):
        result = compute(H, L, C, 95.0)
        assert result is not None
        if result.nearest_resistance is not None:
            assert result.nearest_resistance > 95.0

    def test_nearest_support_below_price(self):
        result = compute(H, L, C, 105.0)
        assert result is not None
        if result.nearest_support is not None:
            assert result.nearest_support < 105.0


class TestMetadata:
    def test_symbol_stored(self):
        result = compute(H, L, C, 100.0, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_method_stored(self):
        result = compute(H, L, C, 100.0, method="fibonacci")
        assert result is not None
        assert result.method == "fibonacci"

    def test_current_price_stored(self):
        result = compute(H, L, C, 103.0)
        assert result is not None
        assert result.current_price == pytest.approx(103.0, abs=0.01)

    def test_verdict_present(self):
        result = compute(H, L, C, 100.0)
        assert result is not None
        assert len(result.verdict) > 10
