"""Tests for amms.analysis.order_flow_imbalance."""

from __future__ import annotations

import pytest

from amms.analysis.order_flow_imbalance import OFIBar, OFIReport, analyze


class _Bar:
    def __init__(self, open_: float, high: float, low: float, close: float, volume: float = 1000.0):
        self.open   = open_
        self.high   = high
        self.low    = low
        self.close  = close
        self.volume = volume


def _up_bars(n: int = 60, start: float = 100.0, step: float = 1.0) -> list[_Bar]:
    """Consistent up bars — should produce strong buy signal."""
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = price + step
        bars.append(_Bar(o, c + 0.2, o - 0.1, c, 1000.0))
        price = c
    return bars


def _down_bars(n: int = 60, start: float = 200.0, step: float = 1.0) -> list[_Bar]:
    """Consistent down bars — should produce strong sell signal."""
    bars = []
    price = start
    for _ in range(n):
        o = price
        c = max(price - step, 1.0)
        bars.append(_Bar(o, o + 0.1, c - 0.2, c, 1000.0))
        price = c
    return bars


def _flat_bars(n: int = 60, price: float = 100.0) -> list[_Bar]:
    """Doji-like flat bars — neutral signal."""
    return [_Bar(price, price + 0.5, price - 0.5, price, 500.0) for _ in range(n)]


def _mixed_bars(n: int = 60) -> list[_Bar]:
    bars = []
    price = 100.0
    for i in range(n):
        if i % 2 == 0:
            bars.append(_Bar(price, price + 1.0, price - 0.1, price + 1.0, 1000.0))
            price += 1.0
        else:
            bars.append(_Bar(price, price + 0.1, price - 1.0, price - 1.0, 1000.0))
            price -= 1.0
    return bars


class TestEdgeCases:
    def test_returns_none_empty(self):
        assert analyze([]) is None

    def test_returns_none_too_few_bars(self):
        assert analyze(_up_bars(10)) is None

    def test_returns_result_enough_bars(self):
        result = analyze(_up_bars(55))
        assert result is not None
        assert isinstance(result, OFIReport)

    def test_no_open_attr_returns_none(self):
        class _NoOpen:
            high = low = close = volume = 1.0
        assert analyze([_NoOpen()] * 60) is None

    def test_no_volume_still_works(self):
        class _NoVol:
            open = high = low = close = 100.0
        result = analyze([_NoVol()] * 60)
        assert result is not None


class TestSignal:
    def test_up_bars_buy_or_strong_buy(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert result.signal in {"buy", "strong_buy"}

    def test_down_bars_sell_or_strong_sell(self):
        result = analyze(_down_bars(60))
        assert result is not None
        assert result.signal in {"sell", "strong_sell"}

    def test_flat_bars_neutral_or_near(self):
        result = analyze(_flat_bars(60))
        assert result is not None
        assert result.signal in {"neutral", "buy", "sell"}

    def test_signal_valid_values(self):
        valid = {"strong_buy", "buy", "neutral", "sell", "strong_sell"}
        for bars in [_up_bars(60), _down_bars(60), _flat_bars(60), _mixed_bars(60)]:
            result = analyze(bars)
            if result:
                assert result.signal in valid


class TestScore:
    def test_score_in_range(self):
        for bars in [_up_bars(60), _down_bars(60), _flat_bars(60)]:
            result = analyze(bars)
            if result:
                assert -100.0 <= result.score <= 100.0

    def test_up_bars_positive_score(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert result.score > 0

    def test_down_bars_negative_score(self):
        result = analyze(_down_bars(60))
        assert result is not None
        assert result.score < 0

    def test_cumulative_ofi_norm_in_range(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert -100.0 <= result.cumulative_ofi_norm <= 100.0


class TestBuyVolume:
    def test_up_bars_high_buy_pct(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert result.buy_pct > 50.0

    def test_down_bars_low_buy_pct(self):
        result = analyze(_down_bars(60))
        assert result is not None
        assert result.buy_pct < 50.0

    def test_buy_pct_in_range(self):
        for bars in [_up_bars(60), _down_bars(60), _flat_bars(60)]:
            result = analyze(bars)
            if result:
                assert 0.0 <= result.buy_pct <= 100.0

    def test_total_vols_non_negative(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert result.total_buy_vol >= 0
        assert result.total_sell_vol >= 0


class TestDivergence:
    def test_up_price_up_ofi_no_divergence(self):
        result = analyze(_up_bars(60))
        assert result is not None
        # price up, OFI up → no divergence
        if result.price_direction == "up" and result.ofi_direction == "up":
            assert not result.divergence

    def test_divergence_bool(self):
        for bars in [_up_bars(60), _down_bars(60)]:
            result = analyze(bars)
            if result:
                assert isinstance(result.divergence, bool)

    def test_price_direction_valid(self):
        for bars in [_up_bars(60), _down_bars(60), _flat_bars(60)]:
            result = analyze(bars)
            if result:
                assert result.price_direction in {"up", "down", "flat"}

    def test_ofi_direction_valid(self):
        for bars in [_up_bars(60), _down_bars(60), _flat_bars(60)]:
            result = analyze(bars)
            if result:
                assert result.ofi_direction in {"up", "down", "flat"}


class TestByBar:
    def test_by_bar_length_matches_lookback(self):
        result = analyze(_up_bars(60), lookback=30)
        assert result is not None
        assert len(result.by_bar) == 30

    def test_by_bar_are_ofi_bar(self):
        result = analyze(_up_bars(60))
        assert result is not None
        for b in result.by_bar:
            assert isinstance(b, OFIBar)

    def test_by_bar_buy_vol_non_negative(self):
        result = analyze(_up_bars(60))
        assert result is not None
        for b in result.by_bar:
            assert b.buy_vol >= 0
            assert b.sell_vol >= 0


class TestOFIRatio:
    def test_ofi_ratio_bounded(self):
        for bars in [_up_bars(80), _down_bars(80)]:
            result = analyze(bars)
            if result:
                assert -3.0 <= result.ofi_ratio <= 3.0

    def test_recent_ofi_positive_for_up_bars(self):
        result = analyze(_up_bars(80))
        assert result is not None
        assert result.recent_ofi >= 0


class TestMetadata:
    def test_symbol_stored(self):
        result = analyze(_up_bars(60), symbol="TSLA")
        assert result is not None
        assert result.symbol == "TSLA"

    def test_bars_used_correct(self):
        bars = _up_bars(70)
        result = analyze(bars)
        assert result is not None
        assert result.bars_used == 70


class TestVerdict:
    def test_verdict_present(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert len(result.verdict) > 10

    def test_verdict_mentions_ofi(self):
        result = analyze(_up_bars(60))
        assert result is not None
        assert "OFI" in result.verdict or "ofi" in result.verdict.lower()
