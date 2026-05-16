"""Tests for amms.execution.risk_guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from amms.execution.paper_trader import PaperTrader
from amms.execution.risk_guard import RiskConfig, RiskGuard


@pytest.fixture
def trader():
    return PaperTrader(starting_cash=10_000.0)


@pytest.fixture
def guard(trader, tmp_path):
    return RiskGuard(trader, state_path=tmp_path / "risk.json")


class TestKillswitch:
    def test_default_disarmed(self, guard):
        assert guard.state.killswitch_armed is False

    def test_arm_blocks(self, guard):
        guard.arm_killswitch("test reason")
        assert guard.check() is not None
        assert "killswitch" in guard.check()

    def test_disarm_unblocks(self, guard):
        guard.arm_killswitch("test")
        guard.disarm_killswitch()
        assert guard.state.killswitch_armed is False

    def test_arm_persisted(self, trader, tmp_path):
        path = tmp_path / "r.json"
        g1 = RiskGuard(trader, state_path=path)
        g1.arm_killswitch("persist")
        g2 = RiskGuard(trader, state_path=path)
        assert g2.state.killswitch_armed is True


class TestDrawdown:
    def test_no_drawdown_passes(self, guard):
        guard.update_peak()
        assert guard.check() is None

    def test_drawdown_triggers_kill(self, trader, tmp_path):
        guard = RiskGuard(trader, RiskConfig(max_drawdown_pct=0.10),
                          state_path=tmp_path / "r.json")
        # Mark a high peak, then simulate equity drop
        trader.cash = 20_000.0
        guard.update_peak()
        trader.cash = 17_000.0   # 15% drawdown
        reason = guard.check()
        assert reason is not None
        assert "drawdown" in reason
        assert guard.state.killswitch_armed is True


class TestDailyLoss:
    def test_no_session_no_block(self, guard):
        assert guard.check() is None

    def test_daily_loss_triggers(self, trader, tmp_path):
        guard = RiskGuard(trader, RiskConfig(max_daily_loss_pct=0.02),
                          state_path=tmp_path / "r.json")
        guard.mark_session_start()
        # Simulate equity drop by withdrawing cash directly
        trader.cash -= 500.0  # 5% loss
        reason = guard.check()
        assert reason is not None
        assert "daily" in reason


class TestExposure:
    def test_buy_blocked_when_over_exposed(self, trader, tmp_path):
        guard = RiskGuard(trader, RiskConfig(max_gross_exposure_pct=0.10),
                          state_path=tmp_path / "r.json")
        # Buy: $1500 position on $10k = 15% exposure
        trader.buy("AAPL", 10, 150.0)
        reason = guard.check(side="buy")
        assert reason is not None
        assert "exposure" in reason

    def test_sell_not_blocked_by_exposure(self, trader, tmp_path):
        guard = RiskGuard(trader, RiskConfig(max_gross_exposure_pct=0.10),
                          state_path=tmp_path / "r.json")
        trader.buy("AAPL", 10, 150.0)
        # Sell side bypasses exposure cap
        reason = guard.check(side="sell")
        # Could still be blocked by other reasons but not "exposure"
        assert reason is None or "exposure" not in reason


class TestVetoCallable:
    def test_veto_returns_callable(self, guard):
        veto = guard.make_veto()
        assert callable(veto)

    def test_veto_returns_none_when_safe(self, guard):
        veto = guard.make_veto()
        assert veto(50.0, 0.7) is None

    def test_veto_returns_reason_when_killed(self, guard):
        guard.arm_killswitch("test")
        veto = guard.make_veto()
        assert veto(50.0, 0.7) is not None

    def test_veto_translates_sign_to_side(self, guard):
        # Positive score → buy side, negative → sell side
        veto = guard.make_veto()
        # Both should be safe at default config
        assert veto(50.0, 0.7) is None
        assert veto(-50.0, 0.7) is None


class TestStatus:
    def test_status_returns_dict(self, guard):
        s = guard.status()
        assert isinstance(s, dict)
        for key in ("equity", "peak_equity", "drawdown_pct", "killswitch", "limits"):
            assert key in s

    def test_status_killswitch_reflects_state(self, guard):
        guard.arm_killswitch("test")
        s = guard.status()
        assert s["killswitch"] is True


class TestConfigDisabled:
    def test_disabled_returns_no_veto(self, trader, tmp_path):
        guard = RiskGuard(trader, RiskConfig(enabled=False),
                          state_path=tmp_path / "r.json")
        guard.arm_killswitch("should not matter")
        assert guard.check() is None
