import pytest
from unittest.mock import MagicMock
from bot.strategy.base import Signal
from bot.strategy.combiner import StrategySignalCombiner


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")


def _mock_strategy(signals: list[Signal]) -> MagicMock:
    s = MagicMock()
    s.generate_signals.return_value = signals
    return s


def _buy(symbol: str, reason: str = "test", confidence: float = 0.8) -> Signal:
    return Signal(symbol=symbol, side="buy", reason=reason, confidence=confidence)


# --- min_agreement = 1 ---

def test_single_strategy_triggers_with_agreement_1():
    strat = _mock_strategy([_buy("AAPL")])
    combiner = StrategySignalCombiner([strat], min_agreement=1)

    signals = combiner.generate_signals(["AAPL"])

    assert len(signals) == 1
    assert signals[0].symbol == "AAPL"
    assert signals[0].side == "buy"


def test_no_signal_when_no_strategy_fires():
    strat = _mock_strategy([])
    combiner = StrategySignalCombiner([strat], min_agreement=1)

    signals = combiner.generate_signals(["AAPL"])
    assert signals == []


# --- min_agreement = 2 ---

def test_both_strategies_agree_triggers():
    s1 = _mock_strategy([_buy("AAPL", reason="momentum breakout")])
    s2 = _mock_strategy([_buy("AAPL", reason="RSI crossover")])
    combiner = StrategySignalCombiner([s1, s2], min_agreement=2)

    signals = combiner.generate_signals(["AAPL"])

    assert len(signals) == 1
    assert "momentum breakout" in signals[0].reason
    assert "RSI crossover" in signals[0].reason


def test_only_one_strategy_agrees_with_min_2_no_signal():
    s1 = _mock_strategy([_buy("AAPL")])
    s2 = _mock_strategy([])           # second strategy silent
    combiner = StrategySignalCombiner([s1, s2], min_agreement=2)

    signals = combiner.generate_signals(["AAPL"])
    assert signals == []


def test_different_symbols_each_strategy():
    s1 = _mock_strategy([_buy("AAPL")])
    s2 = _mock_strategy([_buy("MSFT")])
    combiner = StrategySignalCombiner([s1, s2], min_agreement=2)

    # Both agree on different symbols → neither meets min_agreement=2
    signals = combiner.generate_signals(["AAPL", "MSFT"])
    assert signals == []


def test_confidence_is_averaged():
    s1 = _mock_strategy([_buy("AAPL", confidence=0.6)])
    s2 = _mock_strategy([_buy("AAPL", confidence=0.8)])
    combiner = StrategySignalCombiner([s1, s2], min_agreement=2)

    signals = combiner.generate_signals(["AAPL"])
    assert len(signals) == 1
    assert abs(signals[0].confidence - 0.7) < 0.001


def test_agreement_tag_in_reason():
    s1 = _mock_strategy([_buy("NVDA")])
    s2 = _mock_strategy([_buy("NVDA")])
    combiner = StrategySignalCombiner([s1, s2], min_agreement=1)

    signals = combiner.generate_signals(["NVDA"])
    assert "[2/2" in signals[0].reason
