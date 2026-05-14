from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.strategy.mean_reversion import MeanReversion


def _bars(closes):
    return [Bar("X", "1Day", f"d{i}", c, c, c, c, 100) for i, c in enumerate(closes)]


def test_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError):
        MeanReversion(z_buy=0.5, z_exit=-0.5)
    with pytest.raises(ValueError):
        MeanReversion(window=1)


def test_buy_when_price_far_below_mean() -> None:
    # 20 closes at 100, last close at 80 → ~3 stdev below mean of 100.
    closes = [100.0] * 20 + [80.0]
    s = MeanReversion(window=20, z_buy=-1.5).evaluate("X", _bars(closes))
    assert s.kind == "buy"


def test_sell_when_price_returns_to_mean() -> None:
    closes = list(range(80, 100)) + [100]  # mean ≈ 90, last 100 → high z
    s = MeanReversion(window=20).evaluate("X", _bars(closes))
    assert s.kind == "sell"


def test_hold_when_history_short() -> None:
    s = MeanReversion(window=20).evaluate("X", _bars([1, 2, 3]))
    assert s.kind == "hold"


def test_register_by_name() -> None:
    from amms.strategy import build_strategy

    s = build_strategy("mean_reversion", {})
    assert isinstance(s, MeanReversion)
