"""Tests for amms.analysis.regime."""

from __future__ import annotations

import pytest

from amms.analysis.regime import MarketRegime, _FALLBACK, detect_regime


class _Bar:
    def __init__(self, close):
        self.close = close


class _MockData:
    """Stub data client for testing."""

    def __init__(self, spy_bars=None, vixy_bars=None, raise_spy=False):
        self._spy = spy_bars or []
        self._vixy = vixy_bars or []
        self._raise_spy = raise_spy

    def get_bars(self, symbol, limit=210):
        if symbol == "SPY":
            if self._raise_spy:
                raise RuntimeError("network error")
            return self._spy[-limit:] if self._spy else []
        if symbol == "VIXY":
            return self._vixy[-limit:] if self._vixy else []
        return []


def _bull_spy(n: int = 210) -> list[_Bar]:
    """SPY well above both SMA-50 and SMA-200."""
    return [_Bar(100.0 + i * 0.1) for i in range(n)]


def _bear_spy(n: int = 210) -> list[_Bar]:
    """SPY well below both SMAs (downtrend)."""
    return [_Bar(200.0 - i * 0.5) for i in range(n)]


def _flat_spy(n: int = 210, price: float = 100.0) -> list[_Bar]:
    """SPY roughly flat."""
    return [_Bar(price + (0.5 if i % 2 == 0 else -0.5)) for i in range(n)]


def _stressed_vixy() -> list[_Bar]:
    """VIXY with +7% 1-day move (stress)."""
    return [_Bar(20.0), _Bar(21.4)]  # +7%


def _calm_vixy() -> list[_Bar]:
    return [_Bar(15.0), _Bar(15.1)]  # +0.7%


class TestFallback:
    def test_fallback_on_no_data(self):
        data = _MockData()
        result = detect_regime(data)
        assert result.label == "neutral"
        assert result.confidence == 0.0

    def test_fallback_on_spy_error(self):
        data = _MockData(raise_spy=True)
        result = detect_regime(data)
        assert result.label == _FALLBACK.label
        assert result.confidence == _FALLBACK.confidence

    def test_fallback_on_empty_spy(self):
        data = _MockData(spy_bars=[])
        result = detect_regime(data)
        assert result == _FALLBACK


class TestRegimeLabel:
    def test_bull_regime_with_uptrend(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert result.label == "bull"

    def test_bear_regime_with_downtrend(self):
        data = _MockData(spy_bars=_bear_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert result.label == "bear"

    def test_label_valid(self):
        data = _MockData(spy_bars=_flat_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert result.label in ("bull", "neutral", "bear")

    def test_stressed_vixy_triggers_bear(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_stressed_vixy())
        result = detect_regime(data)
        # Stressed VIXY adds 2 bear counts — should override moderate bull
        assert result.label in ("bear", "neutral")


class TestProperties:
    def test_is_bull(self):
        regime = MarketRegime(label="bull", confidence=0.8, reason="test")
        assert regime.is_bull is True
        assert regime.is_bear is False

    def test_is_bear(self):
        regime = MarketRegime(label="bear", confidence=0.8, reason="test")
        assert regime.is_bear is True
        assert regime.is_bull is False

    def test_risk_multiplier_bull(self):
        regime = MarketRegime(label="bull", confidence=0.8, reason="test")
        assert regime.risk_multiplier == 1.0

    def test_risk_multiplier_neutral(self):
        regime = MarketRegime(label="neutral", confidence=0.5, reason="test")
        assert regime.risk_multiplier == 0.75

    def test_risk_multiplier_bear(self):
        regime = MarketRegime(label="bear", confidence=0.8, reason="test")
        assert regime.risk_multiplier == 0.5


class TestConfidence:
    def test_confidence_in_range(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert 0.0 <= result.confidence <= 1.0

    def test_bull_confidence_positive(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert result.confidence > 0.0


class TestReason:
    def test_reason_nonempty(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert len(result.reason) > 5

    def test_reason_mentions_spy(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert "SPY" in result.reason


class TestSMAs:
    def test_spy_vs_sma_populated(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        assert result.spy_vs_sma50 is not None
        assert result.spy_vs_sma200 is not None

    def test_bull_spy_above_sma50(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_calm_vixy())
        result = detect_regime(data)
        if result.spy_vs_sma50 is not None:
            assert result.spy_vs_sma50 > 0

    def test_vixy_1d_pct_set_when_bars_available(self):
        data = _MockData(spy_bars=_bull_spy(), vixy_bars=_stressed_vixy())
        result = detect_regime(data)
        assert result.vixy_1d_pct is not None
        assert result.vixy_1d_pct > 5.0
