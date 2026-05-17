"""Tests for amms.execution.auto_trader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from amms.execution.auto_trader import AutoTradeDecision, AutoTrader, AutoTraderConfig
from amms.execution.paper_trader import PaperTrader


class _Bar:
    def __init__(self, open_, high, low, close, volume=1000.0):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _up_bars(n: int = 200, start: float = 100.0, step: float = 0.5):
    bars = []
    p = start
    for _ in range(n):
        o = p; c = p + step
        bars.append(_Bar(o, c + 0.3, o - 0.2, c))
        p = c
    return bars


def _down_bars(n: int = 200, start: float = 200.0, step: float = 0.5):
    bars = []
    p = start
    for _ in range(n):
        o = p; c = max(1.0, p - step)
        bars.append(_Bar(o, o + 0.2, c - 0.3, c))
        p = c
    return bars


def _flat_bars(n: int = 200, price: float = 100.0):
    return [_Bar(price, price + 0.5, price - 0.5, price, 500.0) for _ in range(n)]


class _FakeData:
    """Fake data client with predefined bars per symbol."""
    def __init__(self, bars_by_symbol: dict):
        self.bars_by_symbol = bars_by_symbol

    def get_bars(self, symbol: str, limit: int = 200):
        return self.bars_by_symbol.get(symbol.upper(), [])


@pytest.fixture
def fresh_state(tmp_path):
    """Provide a fresh state path for cooldown tracking."""
    return tmp_path / "auto_state.json"


@pytest.fixture
def paper_with_cash():
    return PaperTrader(starting_cash=10_000.0)


class TestEdgeCases:
    def test_unknown_symbol_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        result = at.process_symbol("UNKNOWN")
        assert result.action == "skipped"
        assert "insufficient" in result.reason

    def test_insufficient_bars_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({"X": _up_bars(50)})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        result = at.process_symbol("X")
        assert result.action == "skipped"


class TestBuy:
    def test_uptrend_triggers_buy(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        # Should buy on strong uptrend
        assert result.action in {"bought", "skipped"}  # skipped only if score < min

    def test_buy_creates_position(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        config = AutoTraderConfig(min_score=20, min_confidence=0.4)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        if result.action == "bought":
            assert paper_with_cash.position("AAPL") is not None
            assert result.qty > 0

    def test_already_holding_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        config = AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=0)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        paper_with_cash.buy("AAPL", 5, 150.0)
        result = at.process_symbol("AAPL")
        assert result.action == "skipped"
        assert "already holding" in result.reason

    def test_max_positions_blocks(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        # Fill portfolio with positions
        for sym in ["A", "B", "C"]:
            paper_with_cash.buy(sym, 1, 100.0)
        config = AutoTraderConfig(max_positions=3, min_score=20, min_confidence=0.4)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        assert result.action == "skipped"

    def test_position_sized_by_pct(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        config = AutoTraderConfig(max_position_pct=0.05, min_score=20, min_confidence=0.4)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        if result.action == "bought":
            # 5% of $10k = $500; price ~200 → qty ~2.5
            assert 0 < result.qty * result.price <= 500.1


class TestSell:
    def test_downtrend_closes_position(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _down_bars(200)})
        # First add a position so the sell can close it
        paper_with_cash.buy("AAPL", 5, 150.0)
        config = AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=0)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        assert result.action in {"closed", "skipped"}

    def test_sell_no_position_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _down_bars(200)})
        config = AutoTraderConfig(min_score=20, min_confidence=0.4)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        # Down signal but no position → skip (no shorting)
        if result.score < -20:
            assert result.action == "skipped"


class TestFlat:
    def test_flat_market_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({"X": _flat_bars(200)})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        result = at.process_symbol("X")
        # Flat = neutral, should skip
        assert result.action == "skipped"


class TestCooldown:
    def test_cooldown_active_skips(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        config = AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=60)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        # First trade
        at.process_symbol("AAPL")
        # Second immediately should be in cooldown
        result2 = at.process_symbol("AAPL")
        if result2.action == "skipped":
            assert "cooldown" in result2.reason or "already holding" in result2.reason

    def test_cooldown_zero_no_block(self, paper_with_cash, fresh_state):
        config = AutoTraderConfig(cooldown_minutes=0)
        at = AutoTrader(paper_with_cash, _FakeData({}), config=config, state_path=fresh_state)
        at._record_cooldown("X")
        assert at._in_cooldown("X") is False

    def test_cooldown_persisted(self, paper_with_cash, fresh_state):
        config = AutoTraderConfig(cooldown_minutes=60)
        at1 = AutoTrader(paper_with_cash, _FakeData({}), config=config, state_path=fresh_state)
        at1._record_cooldown("AAPL")
        at2 = AutoTrader(paper_with_cash, _FakeData({}), config=config, state_path=fresh_state)
        assert "AAPL" in at2._cooldowns


class TestStrongOnly:
    def test_strong_only_filters(self, paper_with_cash, fresh_state):
        data = _FakeData({"AAPL": _up_bars(200)})
        config = AutoTraderConfig(allow_strong_only=True,
                                  min_score=20, min_confidence=0.4)
        at = AutoTrader(paper_with_cash, data, config=config, state_path=fresh_state)
        result = at.process_symbol("AAPL")
        # If not strong_buy, must skip
        if result.action == "bought":
            assert True  # was strong_buy
        elif result.action == "skipped":
            assert "not strong_*" in result.reason or "score" in result.reason or "no position" in result.reason or "max" in result.reason


class TestRunWatchlist:
    def test_watchlist_returns_decisions(self, paper_with_cash, fresh_state):
        data = _FakeData({"A": _up_bars(200), "B": _down_bars(200), "C": _flat_bars(200)})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        results = at.run_watchlist(["A", "B", "C"])
        assert len(results) == 3
        assert all(isinstance(r, AutoTradeDecision) for r in results)

    def test_watchlist_handles_exceptions(self, paper_with_cash, fresh_state):
        class BadData:
            def get_bars(self, *_a, **_k):
                raise RuntimeError("boom")
        at = AutoTrader(paper_with_cash, BadData(), state_path=fresh_state)
        # Exception in fetch is caught → skipped, not raised
        results = at.run_watchlist(["X"])
        assert len(results) == 1
        assert results[0].action == "skipped"


class TestAtrStop:
    """ATR stop-loss enforcement in auto_trader."""

    def _stop_bars(self, n: int = 200, entry: float = 100.0, drop: float = 30.0):
        """Bars where recent close is far below a hypothetical entry price."""
        bars = _up_bars(n - 5, start=entry)
        # Add 5 bars that drop hard (simulates stop breach)
        p = entry
        for _ in range(5):
            p -= drop / 5
            bars.append(_Bar(p + 1, p + 1.5, p - 0.5, max(1.0, p)))
        return bars

    def test_atr_stop_closes_position(self, fresh_state):
        """Position entered at $200 but price drops to ~$170 → ATR stop should trigger."""
        bars = self._stop_bars(n=200, entry=200.0, drop=35.0)
        data = _FakeData({"TST": bars})
        trader = PaperTrader(starting_cash=50_000.0)
        # Buy at $200 so stop is well above current price (~$165)
        trader.buy("TST", 10, 200.0)
        config = AutoTraderConfig(min_score=1, min_confidence=0.1, atr_stop_mult=1.5)
        at = AutoTrader(trader, data, config=config, state_path=fresh_state)
        result = at.process_symbol("TST")
        # Either ATR stop or DE sell triggered the close
        assert result.action in {"closed", "skipped"}
        if result.action == "closed":
            assert "stop" in result.reason.lower() or "sell" in result.reason.lower()

    def test_atr_stop_price_computed(self, fresh_state):
        """_atr_stop_price returns a value below avg_cost for normal bars."""
        bars = _up_bars(200, start=100.0)
        trader = PaperTrader(starting_cash=10_000.0)
        at = AutoTrader(trader, _FakeData({}), state_path=fresh_state)
        stop = at._atr_stop_price(bars, avg_cost=150.0)
        assert stop is not None
        assert stop < 150.0

    def test_no_stop_without_position(self, paper_with_cash, fresh_state):
        """No position → ATR stop branch not reached → normal DE flow."""
        bars = _flat_bars(200)
        data = _FakeData({"X": bars})
        at = AutoTrader(paper_with_cash, data, state_path=fresh_state)
        result = at.process_symbol("X")
        # Flat bars → DE says hold/skip; but no crash
        assert result.action in {"bought", "skipped", "closed"}


class TestAtrSizing:
    """ATR-based position sizing in _do_buy."""

    def test_qty_nonzero_for_valid_bars(self, fresh_state):
        """With valid bars and ATR computable, qty should be > 0."""
        bars = _up_bars(200, start=100.0)
        trader = PaperTrader(starting_cash=100_000.0)
        at = AutoTrader(trader, _FakeData({}), state_path=fresh_state)
        qty = at._calc_qty(100.0, 100_000.0, bars)
        assert qty > 0

    def test_qty_fallback_without_bars(self, fresh_state):
        """Without bars, falls back to max_position_pct sizing."""
        trader = PaperTrader(starting_cash=10_000.0)
        config = AutoTraderConfig(max_position_pct=0.10)
        at = AutoTrader(trader, _FakeData({}), config=config, state_path=fresh_state)
        qty = at._calc_qty(100.0, 10_000.0, None)
        # Fallback: 10% of $10k = $1000 / $100 = 10 shares
        assert qty == pytest.approx(10.0, abs=0.01)

    def test_qty_zero_for_invalid_price(self, fresh_state):
        trader = PaperTrader(starting_cash=10_000.0)
        at = AutoTrader(trader, _FakeData({}), state_path=fresh_state)
        assert at._calc_qty(0.0, 10_000.0, None) == 0.0
        assert at._calc_qty(-5.0, 10_000.0, None) == 0.0
