from __future__ import annotations

import pytest

from amms.risk.rules import RiskConfig, check_buy, position_size


def test_position_size_basic() -> None:
    assert position_size(equity=100_000, price=200, max_position_pct=0.02) == 10
    assert position_size(equity=100_000, price=199, max_position_pct=0.02) == 10
    assert position_size(equity=100_000, price=1, max_position_pct=0.02) == 2000


def test_position_size_zero_for_unrealistic_inputs() -> None:
    assert position_size(equity=0, price=100, max_position_pct=0.02) == 0
    assert position_size(equity=10_000, price=0, max_position_pct=0.02) == 0
    assert position_size(equity=10_000, price=1_000_000, max_position_pct=0.02) == 0


def test_risk_config_validates() -> None:
    with pytest.raises(ValueError, match="max_position_pct"):
        RiskConfig(max_position_pct=1.5)
    with pytest.raises(ValueError, match="daily_loss_pct"):
        RiskConfig(daily_loss_pct=0.05)
    with pytest.raises(ValueError, match="max_open_positions"):
        RiskConfig(max_open_positions=0)
    with pytest.raises(ValueError, match="max_buys_per_tick"):
        RiskConfig(max_buys_per_tick=0)


def test_risk_config_accepts_none_max_buys_per_tick() -> None:
    cfg = RiskConfig(max_buys_per_tick=None)
    assert cfg.max_buys_per_tick is None


def test_risk_config_rejects_negative_min_hold_days() -> None:
    with pytest.raises(ValueError, match="min_hold_days"):
        RiskConfig(min_hold_days=-1)


def test_risk_config_min_hold_days_defaults_to_zero() -> None:
    assert RiskConfig().min_hold_days == 0


def _cfg(**kwargs) -> RiskConfig:
    return RiskConfig(**{
        "max_open_positions": 5,
        "max_position_pct": 0.02,
        "daily_loss_pct": -0.03,
        **kwargs,
    })


def test_check_buy_allowed_typical() -> None:
    d = check_buy(
        equity=100_000,
        price=200,
        cash=50_000,
        open_positions=0,
        daily_pnl_pct=0.0,
        already_holds=False,
        config=_cfg(),
    )
    assert d.allowed is True
    assert d.qty == 10
    assert "buy 10" in d.reason


def test_check_buy_blocks_when_daily_loss_hit() -> None:
    d = check_buy(
        equity=100_000, price=200, cash=50_000, open_positions=0,
        daily_pnl_pct=-0.05, already_holds=False, config=_cfg(),
    )
    assert d.allowed is False
    assert "loss cap" in d.reason


def test_check_buy_blocks_when_already_holds() -> None:
    d = check_buy(
        equity=100_000, price=200, cash=50_000, open_positions=1,
        daily_pnl_pct=0.0, already_holds=True, config=_cfg(),
    )
    assert d.allowed is False
    assert "already long" in d.reason


def test_check_buy_blocks_at_max_positions() -> None:
    d = check_buy(
        equity=100_000, price=200, cash=50_000, open_positions=5,
        daily_pnl_pct=0.0, already_holds=False, config=_cfg(max_open_positions=5),
    )
    assert d.allowed is False
    assert "max" in d.reason


def test_check_buy_blocks_when_size_zero() -> None:
    d = check_buy(
        equity=100, price=200, cash=100, open_positions=0,
        daily_pnl_pct=0.0, already_holds=False, config=_cfg(),
    )
    assert d.allowed is False
    assert "sized to 0" in d.reason


def test_check_buy_blocks_when_cash_too_low() -> None:
    d = check_buy(
        equity=100_000, price=200, cash=500, open_positions=0,
        daily_pnl_pct=0.0, already_holds=False, config=_cfg(),
    )
    assert d.allowed is False
    assert "insufficient cash" in d.reason
