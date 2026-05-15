"""Tests for volatility-adjusted position sizing."""

from __future__ import annotations

from amms.risk.rules import RiskConfig, check_buy, position_size


def test_position_size_basic():
    qty = position_size(100_000, 200.0, 0.02)
    # 100000 * 0.02 = 2000 / 200 = 10 shares
    assert qty == 10


def test_position_size_with_low_atr_no_reduction():
    # Low ATR: vol_dollars = (100000 * 0.01) / 0.5 * 200 = 400_000 — above target
    qty = position_size(100_000, 200.0, 0.02, atr=0.5)
    assert qty == 10  # unchanged


def test_position_size_with_high_atr_reduces():
    # vol budget: qty = (equity * target_risk_pct) / atr
    #           = (100000 * 0.01) / 200 = 5 shares
    # max_pct budget: 100000 * 0.02 / 200 = 10 shares
    # vol cap wins → 5 shares
    qty = position_size(100_000, 200.0, 0.02, atr=200.0)
    assert qty == 5


def test_position_size_zero_equity():
    assert position_size(0, 200.0, 0.02) == 0


def test_position_size_zero_price():
    assert position_size(100_000, 0.0, 0.02) == 0


def test_check_buy_passes_atr():
    config = RiskConfig()
    # Normal case — atr is passed and shouldn't break check_buy
    decision = check_buy(
        equity=100_000,
        price=200.0,
        cash=50_000,
        open_positions=1,
        daily_pnl_pct=0.0,
        already_holds=False,
        config=config,
        atr=2.0,
    )
    assert decision.allowed
    assert decision.qty > 0


def test_check_buy_with_none_atr():
    config = RiskConfig()
    decision = check_buy(
        equity=100_000,
        price=200.0,
        cash=50_000,
        open_positions=1,
        daily_pnl_pct=0.0,
        already_holds=False,
        config=config,
        atr=None,
    )
    assert decision.allowed
