from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.filters import UniverseFilter


def _bar(close: float, *, volume: float = 100_000.0) -> Bar:
    return Bar(
        symbol="X",
        timeframe="1Day",
        ts="2025-01-01T00:00:00Z",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=volume,
    )


def test_inert_filter_passes_anything() -> None:
    f = UniverseFilter()
    passes, reason = f.passes([_bar(0.01, volume=1)])
    assert passes is True
    assert reason is None


def test_empty_bars_fail() -> None:
    passes, reason = UniverseFilter().passes([])
    assert passes is False
    assert "no bars" in reason


def test_min_price_blocks_sub_dollar() -> None:
    f = UniverseFilter(min_price=1.0)
    passes, reason = f.passes([_bar(0.50)])
    assert passes is False
    assert "min" in reason


def test_max_price_blocks_expensive() -> None:
    f = UniverseFilter(max_price=100.0)
    passes, reason = f.passes([_bar(150.0)])
    assert passes is False
    assert "max" in reason


def test_min_adv_blocks_illiquid() -> None:
    f = UniverseFilter(min_avg_dollar_volume=1_000_000, adv_lookback=5)
    bars = [_bar(2.0, volume=1_000) for _ in range(10)]  # ADV = $2,000
    passes, reason = f.passes(bars)
    assert passes is False
    assert "ADV" in reason


def test_min_adv_passes_liquid() -> None:
    f = UniverseFilter(min_avg_dollar_volume=100_000, adv_lookback=5)
    bars = [_bar(100.0, volume=10_000) for _ in range(10)]  # ADV = $1,000,000
    passes, reason = f.passes(bars)
    assert passes is True
    assert reason is None


def test_adv_check_requires_lookback_bars() -> None:
    f = UniverseFilter(min_avg_dollar_volume=100, adv_lookback=20)
    passes, reason = f.passes([_bar(10.0) for _ in range(5)])
    assert passes is False
    assert "ADV" in reason


def test_rejects_negative_min_price() -> None:
    with pytest.raises(ValueError, match="min_price"):
        UniverseFilter(min_price=-1)


def test_rejects_max_below_min() -> None:
    with pytest.raises(ValueError, match="max_price"):
        UniverseFilter(min_price=10, max_price=5)


def test_rejects_zero_adv_lookback() -> None:
    with pytest.raises(ValueError, match="adv_lookback"):
        UniverseFilter(adv_lookback=0)


def test_full_penny_stock_profile_blocks_dust() -> None:
    f = UniverseFilter(min_price=1.0, min_avg_dollar_volume=1_000_000, adv_lookback=20)
    # 25 bars at $0.10, 100k volume → price fails before ADV is even checked.
    passes, reason = f.passes([_bar(0.10, volume=100_000) for _ in range(25)])
    assert passes is False
    assert "price" in reason


def test_passes_asset_disabled_by_default() -> None:
    f = UniverseFilter()
    passes, reason = f.passes_asset(None)
    assert passes is True
    assert reason is None


def test_passes_asset_requires_active_tradable() -> None:
    f = UniverseFilter(require_tradable=True)
    p, _ = f.passes_asset({"status": "active", "tradable": True})
    assert p is True
    p, r = f.passes_asset({"status": "inactive", "tradable": True})
    assert p is False and "inactive" in r
    p, r = f.passes_asset({"status": "active", "tradable": False})
    assert p is False and "tradable" in r
    p, r = f.passes_asset(None)
    assert p is False and "unavailable" in r
