"""Tests for Bollinger Bands and volume spike features."""

from __future__ import annotations

from amms.data.bars import Bar
from amms.features.bollinger import BollingerBands, bollinger, volume_spike


def _bar(close: float, i: int = 0, volume: float = 1000.0) -> Bar:
    return Bar("X", "1D", f"2026-01-{(i % 28) + 1:02d}", close, close + 0.5, close - 0.5, close, volume)


def _bars(closes: list[float], volumes: list[float] | None = None) -> list[Bar]:
    if volumes is None:
        volumes = [1000.0] * len(closes)
    return [_bar(c, i, v) for i, (c, v) in enumerate(zip(closes, volumes))]


class TestBollinger:
    def test_returns_none_if_insufficient_bars(self) -> None:
        bars = _bars([100.0] * 10)
        assert bollinger(bars, n=20) is None

    def test_returns_none_for_exactly_n_minus_1_bars(self) -> None:
        bars = _bars([100.0] * 19)
        assert bollinger(bars, n=20) is None

    def test_returns_value_for_exactly_n_bars(self) -> None:
        bars = _bars([float(i + 100) for i in range(20)])
        result = bollinger(bars, n=20)
        assert result is not None
        assert isinstance(result, BollingerBands)

    def test_zero_std_returns_equal_bands(self) -> None:
        bars = _bars([100.0] * 20)
        result = bollinger(bars, n=20)
        assert result is not None
        assert result.upper == result.middle == result.lower
        assert result.pct_b == 0.5
        assert result.bandwidth == 0.0

    def test_upper_greater_than_lower(self) -> None:
        closes = [float(100 + (i % 5)) for i in range(20)]
        result = bollinger(_bars(closes), n=20)
        assert result is not None
        assert result.upper > result.lower

    def test_middle_is_mean(self) -> None:
        closes = [float(100 + i) for i in range(20)]
        result = bollinger(_bars(closes), n=20)
        assert result is not None
        expected_mean = sum(closes) / len(closes)
        assert abs(result.middle - expected_mean) < 0.01

    def test_pct_b_at_upper_is_one(self) -> None:
        """Price at upper band: pct_b ~= 1."""
        closes = [100.0] * 19
        # Last price at upper band (mean + 2 std)
        import statistics
        mean = statistics.fmean(closes)
        std = statistics.stdev(closes + [mean + 2 * statistics.stdev([float(100 + (i % 3)) for i in range(20)])])
        # Simpler: use controlled data
        data = [float(100 + (i % 3)) for i in range(20)]
        m = statistics.fmean(data)
        s = statistics.stdev(data)
        upper = m + 2 * s
        data[-1] = upper
        result = bollinger(_bars(data), n=20)
        assert result is not None
        assert result.pct_b > 0.9

    def test_pct_b_at_lower_is_zero(self) -> None:
        """Price at lower band: pct_b ~= 0."""
        import statistics
        data = [float(100 + (i % 3)) for i in range(20)]
        m = statistics.fmean(data)
        s = statistics.stdev(data)
        lower = m - 2 * s
        data[-1] = lower
        result = bollinger(_bars(data), n=20)
        assert result is not None
        assert result.pct_b < 0.1

    def test_raises_if_n_less_than_2(self) -> None:
        bars = _bars([100.0] * 20)
        try:
            bollinger(bars, n=1)
            assert False, "should have raised"
        except ValueError:
            pass

    def test_bandwidth_nonzero_for_volatile_data(self) -> None:
        closes = [float(90 + (i % 20)) for i in range(20)]
        result = bollinger(_bars(closes), n=20)
        assert result is not None
        assert result.bandwidth > 0

    def test_uses_last_n_bars(self) -> None:
        """Extra bars before the last n should not affect result."""
        closes_20 = [float(100 + i) for i in range(20)]
        closes_extra = [50.0] * 5 + closes_20
        r1 = bollinger(_bars(closes_20), n=20)
        r2 = bollinger(_bars(closes_extra), n=20)
        assert r1 is not None and r2 is not None
        assert r1.middle == r2.middle


class TestVolumeSpike:
    def test_returns_none_if_insufficient_bars(self) -> None:
        bars = _bars([100.0] * 20)
        assert volume_spike(bars, n=20) is None

    def test_returns_ratio_for_sufficient_bars(self) -> None:
        vols = [1000.0] * 21
        bars = _bars([100.0] * 21, volumes=vols)
        result = volume_spike(bars, n=20)
        assert result is not None
        assert result == 1.0

    def test_spike_above_threshold(self) -> None:
        vols = [1000.0] * 20 + [5000.0]
        bars = _bars([100.0] * 21, volumes=vols)
        result = volume_spike(bars, n=20)
        assert result is not None
        assert result > 2.0

    def test_normal_volume_near_one(self) -> None:
        vols = [1000.0] * 20 + [900.0]
        bars = _bars([100.0] * 21, volumes=vols)
        result = volume_spike(bars, n=20)
        assert result is not None
        assert abs(result - 0.9) < 0.05

    def test_zero_avg_volume_returns_none(self) -> None:
        vols = [0.0] * 20 + [1000.0]
        bars = _bars([100.0] * 21, volumes=vols)
        result = volume_spike(bars, n=20)
        assert result is None
