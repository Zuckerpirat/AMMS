"""Regression tests for the five CRITICAL fixes from the Opus 4.7 MAX review.

C1: Risk Guard not wired into Auto-Trader → killswitch was cosmetic
C2: assert_live_allowed() never called → live guard was advisory
C3: Cooldown blocked sells → couldn't exit losing positions
C4: cooldown_after_kill_minutes was dead config → no auto-recovery
H3 (bonus): Realized P&L did not include commission
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from amms.execution.auto_trader import AutoTrader, AutoTraderConfig, AutoTradeDecision
from amms.execution.paper_trader import PaperTrader
from amms.execution.risk_guard import RiskConfig, RiskGuard


class _Bar:
    def __init__(self, open_, high, low, close, volume=1000.0):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def _up_bars(n: int = 200, start: float = 100.0, step: float = 0.5):
    bars, p = [], start
    for _ in range(n):
        o, c = p, p + step
        bars.append(_Bar(o, c + 0.3, o - 0.2, c))
        p = c
    return bars


def _down_bars(n: int = 200, start: float = 200.0, step: float = 0.5):
    bars, p = [], start
    for _ in range(n):
        o, c = p, max(1.0, p - step)
        bars.append(_Bar(o, o + 0.2, c - 0.3, c))
        p = c
    return bars


class _FakeData:
    def __init__(self, bars_by_symbol):
        self.bars = bars_by_symbol

    def get_bars(self, symbol, limit=200):
        return self.bars.get(symbol.upper(), [])


@pytest.fixture
def trader():
    return PaperTrader(starting_cash=10_000.0)


@pytest.fixture
def risk_guard(trader, tmp_path):
    return RiskGuard(trader, state_path=tmp_path / "risk.json")


@pytest.fixture
def at_with_guard(trader, risk_guard, tmp_path):
    """AutoTrader wired to a RiskGuard."""
    data = _FakeData({"AAPL": _up_bars(200), "BAD": _down_bars(200)})
    config = AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=0)
    return AutoTrader(
        trader, data, config=config,
        state_path=tmp_path / "at_state.json",
        risk_guard=risk_guard,
    )


# ── C1: Risk Guard IS wired into Auto-Trader ──────────────────────────────────


class TestC1_RiskGuardWired:
    def test_killswitch_blocks_buy_decision(self, at_with_guard, risk_guard):
        risk_guard.arm_killswitch("test arm")
        result = at_with_guard.process_symbol("AAPL")
        assert result.action == "skipped"
        assert "killswitch" in result.reason

    def test_killswitch_blocks_before_data_fetch(self, trader, risk_guard, tmp_path):
        # If killswitch fires first, we should not even fetch bars.
        class _NeverCalledData:
            def get_bars(self, *a, **kw):
                raise AssertionError("Data fetch should be skipped when killswitch armed")
        at = AutoTrader(
            trader, _NeverCalledData(),
            config=AutoTraderConfig(cooldown_minutes=0),
            state_path=tmp_path / "at.json",
            risk_guard=risk_guard,
        )
        risk_guard.arm_killswitch("test")
        result = at.process_symbol("AAPL")
        assert result.action == "skipped"

    def test_drawdown_veto_blocks(self, trader, tmp_path):
        # Build up peak, then crash equity, then RiskGuard auto-arms
        # killswitch via the Decision Engine's risk_veto path.
        guard = RiskGuard(
            trader, RiskConfig(max_drawdown_pct=0.05),
            state_path=tmp_path / "r.json",
        )
        trader.cash = 20_000.0
        guard.update_peak()
        trader.cash = 18_000.0   # 10% drawdown — exceeds 5% limit
        # Direct check arms killswitch; subsequent veto blocks all actions.
        veto = guard.make_veto()
        reason = veto(50.0, 0.7)
        assert reason is not None
        assert guard.state.killswitch_armed is True

    def test_no_guard_means_no_block(self, trader, tmp_path):
        # AutoTrader without a risk guard works normally (backward compat).
        data = _FakeData({"AAPL": _up_bars(200)})
        at = AutoTrader(
            trader, data,
            config=AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=0),
            state_path=tmp_path / "at.json",
            # risk_guard not passed → None
        )
        result = at.process_symbol("AAPL")
        # Without a guard, no killswitch-related skip
        assert "killswitch" not in result.reason


# ── C2: Live guard enforced when broker is initialized with live URL ────────


class _FakeSettings:
    def __init__(self, base_url, key="K", secret="S"):
        self.alpaca_base_url = base_url
        self.alpaca_api_key = key
        self.alpaca_api_secret = secret


class TestC2_LiveGuardEnforced:
    def test_paper_url_does_not_require_live_ack(self, monkeypatch):
        """Paper URL must work without AMMS_LIVE_ACKNOWLEDGED."""
        monkeypatch.delenv("AMMS_LIVE_ACKNOWLEDGED", raising=False)
        from amms.execution.alpaca_broker import AlpacaPaperBroker
        # Paper URL goes through (Alpaca client lib refuses non-paper anyway)
        broker = AlpacaPaperBroker.from_settings(
            _FakeSettings("https://paper-api.alpaca.markets")
        )
        assert broker.name == "alpaca-paper"
        broker.close()

    def test_live_url_blocked_without_ack(self, monkeypatch):
        """Live URL must raise without acknowledgement."""
        monkeypatch.delenv("AMMS_LIVE_ACKNOWLEDGED", raising=False)
        from amms.execution.alpaca_broker import AlpacaPaperBroker
        from amms.execution.live_guard import LiveTradingNotAllowed
        with pytest.raises(LiveTradingNotAllowed):
            AlpacaPaperBroker.from_settings(
                _FakeSettings("https://api.alpaca.markets")
            )

    def test_live_url_blocked_with_only_ack_no_caps(self, monkeypatch):
        monkeypatch.setenv("AMMS_LIVE_ACKNOWLEDGED", "I_UNDERSTAND_REAL_MONEY")
        monkeypatch.delenv("AMMS_LIVE_MAX_ORDER_USD", raising=False)
        from amms.execution.alpaca_broker import AlpacaPaperBroker
        from amms.execution.live_guard import LiveTradingNotAllowed
        with pytest.raises(LiveTradingNotAllowed):
            AlpacaPaperBroker.from_settings(
                _FakeSettings("https://api.alpaca.markets")
            )


# ── C3: Cooldown does NOT block sells (only buys) ──────────────────────────


class TestC3_CooldownOnlyBlocksBuys:
    def test_cooldown_active_buy_skipped(self, trader, tmp_path):
        data = _FakeData({"AAPL": _up_bars(200)})
        at = AutoTrader(
            trader, data,
            config=AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=60),
            state_path=tmp_path / "at.json",
        )
        at._record_cooldown("AAPL")
        result = at.process_symbol("AAPL")
        assert result.action == "skipped"
        assert "cooldown" in result.reason

    def test_cooldown_active_sell_still_proceeds(self, trader, tmp_path):
        """If we hold a position and the engine says sell, cooldown must NOT block."""
        data = _FakeData({"BAD": _down_bars(200)})
        # Open a position first
        trader.buy("BAD", 5, 150.0)
        at = AutoTrader(
            trader, data,
            config=AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=60),
            state_path=tmp_path / "at.json",
        )
        at._record_cooldown("BAD")
        result = at.process_symbol("BAD")
        # Sell path should be reachable even with cooldown active.
        # Result could be "closed" if engine fires sell, or "skipped" with
        # a reason other than "cooldown" (e.g. score below threshold).
        if result.action == "skipped":
            assert "cooldown" not in result.reason, (
                f"Cooldown should not block sells. Got reason: {result.reason}"
            )


# ── C4: cooldown_after_kill_minutes auto-disarms auto-armed killswitch ─────


class TestC4_AutoDisarm:
    def test_manual_kill_not_auto_disarmed(self, trader, tmp_path):
        guard = RiskGuard(
            trader, RiskConfig(cooldown_after_kill_minutes=1),
            state_path=tmp_path / "r.json",
        )
        guard.arm_killswitch("manual", auto=False)
        # Simulate time passage
        guard.state.killswitch_armed_at = (
            datetime.now(timezone.utc) - timedelta(minutes=10)
        ).isoformat()
        guard._maybe_auto_disarm()
        assert guard.state.killswitch_armed is True, "Manual kill must NOT auto-disarm"

    def test_auto_kill_disarmed_after_cooldown(self, trader, tmp_path):
        guard = RiskGuard(
            trader, RiskConfig(cooldown_after_kill_minutes=1),
            state_path=tmp_path / "r.json",
        )
        guard.arm_killswitch("auto drawdown trigger", auto=True)
        # Simulate enough time passing
        guard.state.killswitch_armed_at = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
        guard._maybe_auto_disarm()
        assert guard.state.killswitch_armed is False

    def test_auto_kill_not_disarmed_before_cooldown(self, trader, tmp_path):
        guard = RiskGuard(
            trader, RiskConfig(cooldown_after_kill_minutes=60),
            state_path=tmp_path / "r.json",
        )
        guard.arm_killswitch("auto", auto=True)
        # Only 1 minute elapsed — should not yet disarm
        guard.state.killswitch_armed_at = (
            datetime.now(timezone.utc) - timedelta(minutes=1)
        ).isoformat()
        guard._maybe_auto_disarm()
        assert guard.state.killswitch_armed is True

    def test_drawdown_arms_with_auto_flag(self, trader, tmp_path):
        guard = RiskGuard(
            trader, RiskConfig(max_drawdown_pct=0.05),
            state_path=tmp_path / "r.json",
        )
        trader.cash = 20_000.0
        guard.update_peak()
        trader.cash = 18_000.0
        guard.check()
        assert guard.state.killswitch_armed is True
        assert guard.state.killswitch_auto is True, "Drawdown trigger must set auto=True"


# ── H3 (bonus): Realized P&L now subtracts commission ──────────────────────


class TestH3_RealizedPnLIncludesCommission:
    def test_realized_pnl_minus_commission(self):
        trader = PaperTrader(starting_cash=10_000.0, commission=0.001)
        trader.buy("AAPL", 10, 100.0)   # commission $1
        trader.sell("AAPL", 10, 110.0)  # commission $1.10
        snap = trader.snapshot()
        # Profit before commission: (110-100)*10 = 100
        # Commission on sell: 10*110*0.001 = 1.10
        # Realized P&L should be 100 - 1.10 = 98.90
        assert abs(snap.total_realized_pnl - 98.90) < 0.01

    def test_zero_commission_unchanged(self):
        trader = PaperTrader(starting_cash=10_000.0, commission=0.0)
        trader.buy("AAPL", 10, 100.0)
        trader.sell("AAPL", 10, 110.0)
        snap = trader.snapshot()
        assert abs(snap.total_realized_pnl - 100.0) < 0.01


# ── Bonus: Thread-safety against concurrent process_symbol calls ─────────


class TestConcurrentProcess:
    def test_concurrent_calls_serialized(self, trader, tmp_path):
        """Two threads calling process_symbol() simultaneously must serialize."""
        call_count = [0]
        in_flight = [0]
        max_concurrent = [0]
        lock_for_test = threading.Lock()

        class _SlowData:
            def get_bars(self, symbol, limit=200):
                with lock_for_test:
                    in_flight[0] += 1
                    if in_flight[0] > max_concurrent[0]:
                        max_concurrent[0] = in_flight[0]
                # Simulate slow API
                time.sleep(0.05)
                with lock_for_test:
                    in_flight[0] -= 1
                    call_count[0] += 1
                return _up_bars(200)

        at = AutoTrader(
            trader, _SlowData(),
            config=AutoTraderConfig(min_score=20, min_confidence=0.4, cooldown_minutes=0),
            state_path=tmp_path / "at.json",
        )

        threads = [
            threading.Thread(target=at.process_symbol, args=("AAPL",))
            for _ in range(5)
        ]
        for t in threads: t.start()
        for t in threads: t.join(timeout=5.0)

        assert max_concurrent[0] == 1, (
            f"process_symbol must serialize, but saw {max_concurrent[0]} concurrent calls"
        )
        assert call_count[0] == 5
