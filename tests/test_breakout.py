from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.strategy.breakout import Breakout


def _bars(values):
    return [
        Bar("X", "1Day", f"d{i}", v, v, v, v, 100)
        for i, v in enumerate(values)
    ]


def test_buy_on_new_high() -> None:
    closes = [10.0] * 20 + [15.0]
    s = Breakout(entry_window=20, exit_window=10).evaluate("X", _bars(closes))
    assert s.kind == "buy"


def test_sell_on_new_low() -> None:
    closes = [10.0] * 20 + [5.0]
    s = Breakout(entry_window=20, exit_window=10).evaluate("X", _bars(closes))
    assert s.kind == "sell"


def test_hold_in_range() -> None:
    closes = [10.0] * 20 + [10.0]
    s = Breakout(entry_window=20, exit_window=10).evaluate("X", _bars(closes))
    assert s.kind == "hold"


def test_hold_when_history_too_short() -> None:
    closes = [10.0] * 5
    s = Breakout(entry_window=20, exit_window=10).evaluate("X", _bars(closes))
    assert s.kind == "hold"


def test_rejects_tiny_windows() -> None:
    with pytest.raises(ValueError):
        Breakout(entry_window=1, exit_window=1)


def test_register_by_name() -> None:
    from amms.strategy import build_strategy

    assert build_strategy("breakout", {}).name == "breakout"
