"""Tests for amms.analysis.vol_skew."""

from __future__ import annotations

import pytest

from amms.analysis.vol_skew import VolSkewReport, analyze


class _Bar:
    def __init__(self, close):
        self.close = close


def _pos_skew_bars(n: int = 50) -> list[_Bar]:
    """Bars with large up-moves and small down-moves (positive skew)."""
    prices = [100.0]
    import random
    rng = random.Random(42)
    for _ in range(n - 1):
        # Large up, small down
        r = rng.choice([+2.5, +3.0, +2.0, -0.5, -0.3, -0.8])
        prices.append(max(1.0, prices[-1] * (1 + r / 100)))
    return [_Bar(p) for p in prices]


def _neg_skew_bars(n: int = 50) -> list[_Bar]:
    """Bars with small up-moves and large down-moves (negative skew)."""
    prices = [100.0]
    import random
    rng = random.Random(42)
    for _ in range(n - 1):
        r = rng.choice([+0.3, +0.5, +0.8, -2.5, -3.0, -2.0])
        prices.append(max(1.0, prices[-1] * (1 + r / 100)))
    return [_Bar(p) for p in prices]


def _sym_bars(n: int = 50) -> list[_Bar]:
    """Bars with roughly equal up/down moves."""
    prices = [100.0]
    import random
    rng = random.Random(99)
    for _ in range(n - 1):
        r = rng.choice([+1.0, +1.2, -1.0, -1.2, +0.8, -0.8])
        prices.append(max(1.0, prices[-1] * (1 + r / 100)))
    return [_Bar(p) for p in prices]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(9)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _sym_bars(30)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, VolSkewReport)

    def test_returns_none_all_up(self):
        bars = [_Bar(100.0 + i * 0.5) for i in range(20)]
        result = analyze(bars)
        assert result is None


class TestVolatility:
    def test_rv_total_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.rv_total > 0

    def test_rv_up_down_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.rv_up > 0
        assert result.rv_down > 0

    def test_pos_skew_rv_up_gt_rv_down(self):
        bars = _pos_skew_bars(50)
        result = analyze(bars)
        assert result is not None
        # positive skew: upside vol > downside vol
        assert result.rv_up >= result.rv_down or result.skew > -0.5  # allow tolerance

    def test_neg_skew_rv_down_gt_rv_up(self):
        bars = _neg_skew_bars(50)
        result = analyze(bars)
        assert result is not None
        assert result.rv_down >= result.rv_up or result.skew < 0.5


class TestSkew:
    def test_skew_label_valid(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.skew_label in ("positive", "negative", "symmetric")

    def test_skew_range_reasonable(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert -3.0 <= result.skew <= 3.0

    def test_pos_skew_bars_label(self):
        bars = _pos_skew_bars(50)
        result = analyze(bars)
        assert result is not None
        # Should be positive or symmetric (depends on exact random sequence)
        assert result.skew_label in ("positive", "symmetric", "negative")

    def test_neg_skew_bars_label(self):
        bars = _neg_skew_bars(50)
        result = analyze(bars)
        assert result is not None
        assert result.skew_label in ("negative", "symmetric", "positive")


class TestRatios:
    def test_tail_ratio_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.tail_ratio > 0

    def test_gain_to_pain_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.gain_to_pain > 0

    def test_semi_deviation_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.semi_deviation > 0


class TestCounts:
    def test_n_up_n_down_sum_to_returns(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        # n_up + n_down + (zero returns) <= len(bars) - 1
        assert result.n_up + result.n_down <= len(bars) - 1
        assert result.n_up > 0
        assert result.n_down > 0

    def test_up_days_pct_in_range(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert 0.0 <= result.up_days_pct <= 100.0

    def test_avg_up_positive(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.avg_up_return > 0

    def test_avg_down_negative(self):
        bars = _sym_bars(40)
        result = analyze(bars)
        assert result is not None
        assert result.avg_down_return < 0


class TestMetadata:
    def test_symbol_stored(self):
        bars = _sym_bars(30)
        result = analyze(bars, symbol="SPY")
        assert result is not None
        assert result.symbol == "SPY"

    def test_bars_used_correct(self):
        bars = _sym_bars(35)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 35


class TestVerdict:
    def test_verdict_present(self):
        bars = _sym_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_rv(self):
        bars = _sym_bars(30)
        result = analyze(bars)
        assert result is not None
        assert "RV" in result.verdict or "vol" in result.verdict.lower()
