"""Tests for sector exposure cap in the risk layer."""

from __future__ import annotations

from amms.risk.rules import RiskConfig, check_sector_cap


class _Pos:
    def __init__(self, symbol: str, market_value: float) -> None:
        self.symbol = symbol
        self.market_value = market_value


def test_cap_disabled_when_zero():
    config = RiskConfig(max_sector_pct=0.0)
    result = check_sector_cap(
        symbol="NVDA",
        positions=[_Pos("AAPL", 50_000), _Pos("MSFT", 50_000)],
        total_equity=100_000,
        config=config,
    )
    assert result is None


def test_cap_allows_buy_when_under_limit():
    config = RiskConfig(max_sector_pct=0.5)
    # Technology is at 20% of equity
    result = check_sector_cap(
        symbol="NVDA",  # Technology
        positions=[_Pos("AAPL", 20_000)],  # Technology at 20%
        total_equity=100_000,
        config=config,
    )
    assert result is None


def test_cap_blocks_buy_when_at_limit():
    config = RiskConfig(max_sector_pct=0.4)
    # Technology already at 45% — should block another Technology buy
    result = check_sector_cap(
        symbol="MSFT",  # Technology
        positions=[_Pos("AAPL", 45_000)],  # Technology at 45%
        total_equity=100_000,
        config=config,
    )
    assert result is not None
    assert not result.allowed
    assert "sector cap" in result.reason
    assert "Technology" in result.reason


def test_cap_allows_different_sector():
    config = RiskConfig(max_sector_pct=0.4)
    # Technology at 45% — but we're buying a Financial, which is at 0%
    result = check_sector_cap(
        symbol="JPM",  # Financials
        positions=[_Pos("AAPL", 45_000)],  # Technology at 45%
        total_equity=100_000,
        config=config,
    )
    assert result is None


def test_cap_no_positions():
    config = RiskConfig(max_sector_pct=0.3)
    result = check_sector_cap(
        symbol="AAPL",
        positions=[],
        total_equity=100_000,
        config=config,
    )
    assert result is None


def test_cap_zero_equity_safe():
    config = RiskConfig(max_sector_pct=0.3)
    result = check_sector_cap(
        symbol="AAPL",
        positions=[_Pos("MSFT", 5_000)],
        total_equity=0,
        config=config,
    )
    assert result is None


def test_cap_exact_boundary_blocked():
    config = RiskConfig(max_sector_pct=0.40)
    # Exactly at 40% → blocked (>= check)
    result = check_sector_cap(
        symbol="AMD",  # Technology
        positions=[_Pos("AAPL", 40_000)],  # Technology at 40%
        total_equity=100_000,
        config=config,
    )
    assert result is not None
    assert not result.allowed


def test_risk_config_validates_max_sector_pct():
    import pytest

    with pytest.raises(ValueError, match="max_sector_pct"):
        RiskConfig(max_sector_pct=1.5)

    with pytest.raises(ValueError, match="max_sector_pct"):
        RiskConfig(max_sector_pct=-0.1)

    # 0 and 1 are valid edge values
    c = RiskConfig(max_sector_pct=0.0)
    assert c.max_sector_pct == 0.0
    c2 = RiskConfig(max_sector_pct=1.0)
    assert c2.max_sector_pct == 1.0
