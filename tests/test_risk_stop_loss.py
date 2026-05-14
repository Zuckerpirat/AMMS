from __future__ import annotations

from dataclasses import dataclass

import pytest

from amms.risk.rules import (
    STOP_LOSS_REASON_PREFIX,
    RiskConfig,
    check_stop_losses,
)


@dataclass(frozen=True)
class _Position:
    """Minimal Alpaca-Position shape for tests."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float


def _pos(symbol: str, qty: float, entry: float, current: float) -> _Position:
    return _Position(
        symbol=symbol,
        qty=qty,
        avg_entry_price=entry,
        market_value=qty * current,
    )


def test_disabled_when_both_caps_are_zero() -> None:
    config = RiskConfig()  # stop_loss_pct=0, trailing_stop_pct=0
    triggers = check_stop_losses(
        positions=[_pos("AAPL", 10, 100.0, 50.0)],  # -50% loss
        config=config,
    )
    assert triggers == []


def test_fixed_stop_triggers_when_loss_exceeds_cap() -> None:
    config = RiskConfig(stop_loss_pct=0.05)
    # NVDA down 6% — over the 5% cap → trigger
    # AAPL down 4% — under the cap → no trigger
    triggers = check_stop_losses(
        positions=[
            _pos("NVDA", 5, 500.0, 470.0),  # -6%
            _pos("AAPL", 10, 100.0, 96.0),  # -4%
        ],
        config=config,
    )
    assert {t.symbol for t in triggers} == {"NVDA"}
    assert triggers[0].kind == "fixed"
    assert triggers[0].reason.startswith(STOP_LOSS_REASON_PREFIX)
    assert triggers[0].loss_pct == pytest.approx(-0.06)


def test_fixed_stop_does_not_trigger_on_winner() -> None:
    config = RiskConfig(stop_loss_pct=0.05)
    triggers = check_stop_losses(
        positions=[_pos("NVDA", 5, 500.0, 600.0)],  # +20%
        config=config,
    )
    assert triggers == []


def test_skip_zero_qty_or_entry() -> None:
    config = RiskConfig(stop_loss_pct=0.01)
    triggers = check_stop_losses(
        positions=[
            _pos("FOO", 0, 100.0, 1.0),  # qty 0
            _pos("BAR", 1, 0.0, 1.0),  # entry 0
        ],
        config=config,
    )
    assert triggers == []


def test_trailing_stop_triggers_after_drop_from_high() -> None:
    config = RiskConfig(trailing_stop_pct=0.10)
    # Entry 100, peaked at 150, now at 130 → 13% drop from high → trigger
    triggers = check_stop_losses(
        positions=[_pos("MSTR", 1, 100.0, 130.0)],
        config=config,
        high_water_marks={"MSTR": 150.0},
    )
    assert len(triggers) == 1
    assert triggers[0].kind == "trailing"
    assert triggers[0].symbol == "MSTR"


def test_trailing_stop_does_not_trigger_within_band() -> None:
    config = RiskConfig(trailing_stop_pct=0.10)
    # Peaked at 150, now at 145 → only 3.3% drop → no trigger
    triggers = check_stop_losses(
        positions=[_pos("MSTR", 1, 100.0, 145.0)],
        config=config,
        high_water_marks={"MSTR": 150.0},
    )
    assert triggers == []


def test_fixed_wins_over_trailing_when_both_would_fire() -> None:
    config = RiskConfig(stop_loss_pct=0.05, trailing_stop_pct=0.10)
    # Entry 100, peaked 150, now 80 → fixed (-20%) AND trailing (-46%) both fire
    # Only one trigger should be emitted, marked as "fixed".
    triggers = check_stop_losses(
        positions=[_pos("X", 1, 100.0, 80.0)],
        config=config,
        high_water_marks={"X": 150.0},
    )
    assert len(triggers) == 1
    assert triggers[0].kind == "fixed"


def test_invalid_stop_loss_pct_raises() -> None:
    with pytest.raises(ValueError):
        RiskConfig(stop_loss_pct=-0.01)
    with pytest.raises(ValueError):
        RiskConfig(stop_loss_pct=1.0)
    with pytest.raises(ValueError):
        RiskConfig(trailing_stop_pct=1.5)


def test_multiple_positions_get_independent_decisions() -> None:
    config = RiskConfig(stop_loss_pct=0.05)
    triggers = check_stop_losses(
        positions=[
            _pos("AAA", 10, 100.0, 90.0),  # -10% → trigger
            _pos("BBB", 10, 100.0, 99.0),  # -1% → no trigger
            _pos("CCC", 10, 100.0, 80.0),  # -20% → trigger
        ],
        config=config,
    )
    assert {t.symbol for t in triggers} == {"AAA", "CCC"}
