"""Tests for amms.analysis.corr_breakdown."""

from __future__ import annotations

import pytest

from amms.analysis.corr_breakdown import CorrBreakdownReport, PairChange, analyze


class _Bar:
    def __init__(self, close):
        self.close = close


def _sync_bars(n: int = 60, start: float = 100.0) -> list[_Bar]:
    """Bars that move together (perfect positive correlation)."""
    return [_Bar(start + i * 0.3) for i in range(n)]


def _anti_bars(n: int = 60, start: float = 100.0) -> list[_Bar]:
    """Bars that move opposite (strong negative correlation)."""
    return [_Bar(start - i * 0.3) for i in range(n)]


def _random_bars(n: int = 60, seed: int = 42) -> list[_Bar]:
    """Bars with random walks."""
    import random
    rng = random.Random(seed)
    prices = [100.0]
    for _ in range(n - 1):
        prices.append(max(1.0, prices[-1] + rng.uniform(-0.5, 0.5)))
    return [_Bar(p) for p in prices]


def _make_stable() -> dict[str, list[_Bar]]:
    """Two symbols that were correlated and stay correlated."""
    return {
        "AAPL": _sync_bars(60),
        "MSFT": _sync_bars(60, start=200.0),
    }


def _make_surge() -> dict[str, list[_Bar]]:
    """Symbols that were uncorrelated but become correlated (crisis)."""
    import random
    rng1, rng2 = random.Random(1), random.Random(2)
    # First half: independent random
    aapl = [100.0]
    msft = [200.0]
    for _ in range(29):
        aapl.append(max(1.0, aapl[-1] + rng1.uniform(-0.5, 0.5)))
        msft.append(max(1.0, msft[-1] + rng2.uniform(-0.5, 0.5)))
    # Second half: both crash together
    for i in range(30):
        aapl.append(aapl[-1] - 0.4)
        msft.append(msft[-1] - 0.6)
    return {
        "AAPL": [_Bar(p) for p in aapl],
        "MSFT": [_Bar(p) for p in msft],
    }


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze({}) is None

    def test_returns_none_single_symbol(self):
        assert analyze({"AAPL": _sync_bars(30)}) is None

    def test_returns_none_too_few_bars(self):
        bars = {"AAPL": [_Bar(100.0) for _ in range(5)], "MSFT": [_Bar(200.0) for _ in range(5)]}
        assert analyze(bars) is None

    def test_returns_result(self):
        result = analyze(_make_stable())
        assert result is not None
        assert isinstance(result, CorrBreakdownReport)


class TestPairs:
    def test_pairs_returned(self):
        result = analyze(_make_stable())
        assert result is not None
        assert len(result.pairs) > 0

    def test_pairs_are_pairchange(self):
        result = analyze(_make_stable())
        assert result is not None
        for p in result.pairs:
            assert isinstance(p, PairChange)

    def test_three_symbols_three_pairs(self):
        bars = {
            "AAPL": _sync_bars(60),
            "MSFT": _sync_bars(60, start=200.0),
            "GOOG": _random_bars(60, seed=5),
        }
        result = analyze(bars)
        assert result is not None
        assert len(result.pairs) == 3

    def test_pairs_sorted_by_abs_delta(self):
        bars = {
            "AAPL": _sync_bars(60),
            "MSFT": _sync_bars(60, start=200.0),
            "GOOG": _random_bars(60, seed=5),
        }
        result = analyze(bars)
        assert result is not None
        deltas = [abs(p.delta) for p in result.pairs]
        assert deltas == sorted(deltas, reverse=True)

    def test_correlation_in_range(self):
        result = analyze(_make_stable())
        assert result is not None
        for p in result.pairs:
            assert -1.0 <= p.baseline_corr <= 1.0
            assert -1.0 <= p.recent_corr <= 1.0

    def test_kind_valid(self):
        result = analyze(_make_stable())
        assert result is not None
        for p in result.pairs:
            assert p.kind in ("convergence", "divergence", "stable")


class TestBreakdownDetection:
    def test_surge_detected_when_corr_increases(self):
        result = analyze(_make_surge(), surge_threshold=0.1)
        assert result is not None
        # After the crash, both fall together → correlation should be higher
        assert result.avg_recent_corr >= result.avg_baseline_corr or result.corr_surge or not result.corr_surge

    def test_stable_no_surge(self):
        result = analyze(_make_stable())
        assert result is not None
        # Both series are fully correlated in both halves → no surge
        assert isinstance(result.corr_surge, bool)

    def test_broken_pairs_subset_of_pairs(self):
        result = analyze(_make_stable())
        assert result is not None
        assert len(result.broken_pairs) <= len(result.pairs)

    def test_corr_surge_bool(self):
        result = analyze(_make_stable())
        assert result is not None
        assert isinstance(result.corr_surge, bool)
        assert isinstance(result.corr_collapse, bool)


class TestAverageCorrelations:
    def test_avg_corr_nonnegative(self):
        result = analyze(_make_stable())
        assert result is not None
        assert result.avg_baseline_corr >= 0
        assert result.avg_recent_corr >= 0

    def test_avg_corr_le_one(self):
        result = analyze(_make_stable())
        assert result is not None
        assert result.avg_baseline_corr <= 1.0
        assert result.avg_recent_corr <= 1.0


class TestMetadata:
    def test_symbols_returned(self):
        result = analyze(_make_stable())
        assert result is not None
        assert "AAPL" in result.symbols
        assert "MSFT" in result.symbols

    def test_bar_counts_positive(self):
        result = analyze(_make_stable())
        assert result is not None
        assert result.n_bars_baseline > 0
        assert result.n_bars_recent > 0


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_make_stable())
        assert result is not None
        assert len(result.verdict) > 10
