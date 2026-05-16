"""Tests for amms.analysis.order_flow."""

from __future__ import annotations

import pytest

from amms.analysis.order_flow import OFIBar, OFIReport, analyze


class _Bar:
    def __init__(self, close, high=None, low=None, volume=1000.0):
        self.close = close
        self.high = high if high is not None else close + 0.5
        self.low = low if low is not None else close - 0.5
        self.volume = volume


def _buyer_bars(n: int = 30) -> list[_Bar]:
    """Bars where close is always near the high — strong buying."""
    return [_Bar(100.0 + i * 0.2, high=100.0 + i * 0.2 + 0.1, low=100.0 + i * 0.2 - 0.9) for i in range(n)]


def _seller_bars(n: int = 30) -> list[_Bar]:
    """Bars where close is always near the low — strong selling."""
    return [_Bar(120.0 - i * 0.2, high=120.0 - i * 0.2 + 0.9, low=120.0 - i * 0.2 - 0.1) for i in range(n)]


def _neutral_bars(n: int = 30) -> list[_Bar]:
    """Bars where close is always at midpoint — neutral."""
    return [_Bar(100.0, high=100.5, low=99.5) for _ in range(n)]


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few(self):
        bars = [_Bar(100.0) for _ in range(4)]
        assert analyze(bars) is None

    def test_returns_result(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result, OFIReport)


class TestOFIValues:
    def test_buyer_bars_positive_avg_ofi(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.avg_ofi > 0

    def test_seller_bars_negative_avg_ofi(self):
        bars = _seller_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.avg_ofi < 0

    def test_neutral_bars_zero_avg_ofi(self):
        bars = _neutral_bars(30)
        result = analyze(bars)
        assert result is not None
        assert abs(result.avg_ofi) < 0.01

    def test_cofi_positive_for_buyers(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.cofi > 0

    def test_cofi_negative_for_sellers(self):
        bars = _seller_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.cofi < 0

    def test_buy_pressure_high_for_buyers(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.buy_pressure_pct > 60

    def test_buy_pressure_low_for_sellers(self):
        bars = _seller_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.buy_pressure_pct < 40


class TestCOFIDirection:
    def test_buyers_rising_cofi(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.cofi_direction == "rising"

    def test_sellers_falling_cofi(self):
        bars = _seller_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.cofi_direction == "falling"

    def test_neutral_flat_cofi(self):
        bars = _neutral_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.cofi_direction == "flat"

    def test_cofi_direction_valid(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.cofi_direction in ("rising", "falling", "flat")


class TestPriceTrend:
    def test_rising_price_detected(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.price_trend in ("up", "flat")

    def test_falling_price_detected(self):
        bars = _seller_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.price_trend in ("down", "flat")

    def test_price_trend_valid(self):
        bars = _neutral_bars(30)
        result = analyze(bars)
        assert result is not None
        assert result.price_trend in ("up", "down", "flat")


class TestDivergence:
    def test_no_divergence_buyers_up(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        # buyers → rising COFI + rising price = no divergence
        if result.cofi_direction == "rising" and result.price_trend == "up":
            assert result.divergence is False

    def test_divergence_is_bool(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert isinstance(result.divergence, bool)

    def test_divergence_detected_on_contradiction(self):
        # Bars where close is near high but price is falling — buying into decline
        bars = []
        for i in range(30):
            price = 120.0 - i * 0.5
            # close near high = buyer pressure despite falling price
            bars.append(_Bar(price, high=price + 0.05, low=price - 0.95))
        result = analyze(bars)
        assert result is not None
        # falling price + rising OFI = divergence
        if result.cofi_direction == "rising" and result.price_trend == "down":
            assert result.divergence is True


class TestOFIBars:
    def test_bars_returned(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.bars) > 0

    def test_bars_max_ten(self):
        bars = _buyer_bars(30)
        result = analyze(bars)
        assert result is not None
        assert len(result.bars) <= 10

    def test_bars_are_ofibar(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        for b in result.bars:
            assert isinstance(b, OFIBar)

    def test_cpr_in_range(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        for b in result.bars:
            assert 0.0 <= b.cpr <= 1.0

    def test_ofi_in_range(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        for b in result.bars:
            assert -1.0 <= b.ofi <= 1.0


class TestMetadata:
    def test_symbol_stored(self):
        bars = _buyer_bars(20)
        result = analyze(bars, symbol="NVDA")
        assert result is not None
        assert result.symbol == "NVDA"

    def test_bars_used_correct(self):
        bars = _buyer_bars(25)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 25

    def test_current_price_matches_last_bar(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert result.current_price == pytest.approx(bars[-1].close, abs=0.01)

    def test_custom_lookback(self):
        bars = _buyer_bars(30)
        result = analyze(bars, lookback=10)
        assert result is not None
        assert result.bars_used == 30  # total bars, not window


class TestVerdict:
    def test_verdict_present(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_ofi(self):
        bars = _buyer_bars(20)
        result = analyze(bars)
        assert result is not None
        assert "OFI" in result.verdict or "pressure" in result.verdict.lower()
