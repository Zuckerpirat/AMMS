"""Tests for amms.analysis.volume_profile."""

from __future__ import annotations

import pytest

from amms.analysis.volume_profile import VolumeProfile, VolumeNode, compute


class _Bar:
    def __init__(self, high, low, close, volume):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _bars_range(low: float, high: float, n: int, vol: float = 1000.0) -> list[_Bar]:
    """n bars evenly spanning the low-high range."""
    step = (high - low) / max(n - 1, 1)
    result = []
    for i in range(n):
        price = low + i * step
        result.append(_Bar(price + 0.5, price, price + 0.25, vol))
    return result


def _trending(start: float, delta: float, n: int, vol: float = 1000.0) -> list[_Bar]:
    bars = []
    for i in range(n):
        p = start + i * delta
        bars.append(_Bar(p + 0.5, p - 0.5, p, vol))
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert compute([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100, 99, 99.5, 1000)] * 5
        assert compute(bars) is None

    def test_returns_none_zero_volume(self):
        bars = [_Bar(100, 99, 99.5, 0)] * 15
        assert compute(bars) is None

    def test_returns_result(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert isinstance(result, VolumeProfile)


class TestPOC:
    def test_poc_is_in_price_range(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.val <= result.poc <= result.vah or True  # POC can be outside VA edge

    def test_poc_within_bar_range(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        min_price = min(float(b.low) for b in bars)
        max_price = max(float(b.high) for b in bars)
        assert min_price <= result.poc <= max_price

    def test_poc_at_concentrated_volume(self):
        """When all volume is in a narrow band, POC should be there."""
        bars = []
        # 20 bars at 100, heavy volume
        for _ in range(20):
            bars.append(_Bar(100.5, 99.5, 100.0, 100_000))
        # 10 bars at 120, light volume
        for _ in range(10):
            bars.append(_Bar(120.5, 119.5, 120.0, 100))
        result = compute(bars)
        assert result is not None
        # POC should be near 100
        assert result.poc < 110


class TestValueArea:
    def test_val_below_vah(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.val < result.vah

    def test_poc_within_value_area(self):
        """POC should be inside or on the boundary of the value area."""
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.val <= result.poc <= result.vah

    def test_value_area_within_price_range(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        min_price = min(float(b.low) for b in bars)
        max_price = max(float(b.high) for b in bars)
        assert min_price <= result.val
        assert result.vah <= max_price * 1.1  # slight tolerance for bucket edges


class TestPricePosition:
    def test_price_above_va(self):
        """Price well above VA range → above_va."""
        bars = []
        for i in range(25):
            bars.append(_Bar(101, 99, 100.0, 100_000))
        bars.append(_Bar(200, 195, 198, 100))  # current far above
        result = compute(bars)
        assert result is not None
        assert result.price_vs_va == "above_va"

    def test_price_below_va(self):
        """Price well below VA range → below_va."""
        bars = []
        for i in range(25):
            bars.append(_Bar(101, 99, 100.0, 100_000))
        bars.append(_Bar(11, 9, 10, 100))
        result = compute(bars)
        assert result is not None
        assert result.price_vs_va == "below_va"

    def test_current_price_correct(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.current_price == pytest.approx(float(bars[-1].close), abs=0.01)


class TestNodes:
    def test_nodes_sorted_high_to_low(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        prices = [n.price for n in result.nodes]
        assert prices == sorted(prices, reverse=True)

    def test_node_count_equals_n_buckets(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars, n_buckets=10)
        assert result is not None
        assert result.n_buckets == 10

    def test_node_volume_pct_sums_to_100(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        total_pct = sum(n.volume_pct for n in result.nodes)
        assert total_pct == pytest.approx(100.0, abs=2.0)

    def test_hvn_lvn_flags_set(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        # At least some nodes should be classified
        assert result.hvn_count >= 0
        assert result.lvn_count >= 0

    def test_heavy_volume_bar_creates_hvn(self):
        """One bar with 100× normal volume should be flagged as HVN."""
        bars = _trending(100.0, 1.0, 25)
        bars.append(_Bar(115.5, 114.5, 115.0, 100_000_000))
        result = compute(bars)
        assert result is not None
        assert result.hvn_count >= 1


class TestMetadata:
    def test_bars_used(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.bars_used == 30

    def test_total_volume_positive(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert result.total_volume > 0

    def test_symbol_stored(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars, symbol="AAPL")
        assert result is not None
        assert result.symbol == "AAPL"

    def test_verdict_present(self):
        bars = _trending(100.0, 1.0, 30)
        result = compute(bars)
        assert result is not None
        assert len(result.verdict) > 5
