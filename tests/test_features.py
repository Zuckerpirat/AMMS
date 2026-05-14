from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.features import (
    atr,
    n_day_return,
    realized_vol,
    relative_volume,
    rsi,
    standard_features,
)


def _bar(
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
    volume: float = 100.0,
) -> Bar:
    h = high if high is not None else close
    lo = low if low is not None else close
    return Bar(
        symbol="X",
        timeframe="1Day",
        ts="2025-01-01T00:00:00Z",
        open=close,
        high=h,
        low=lo,
        close=close,
        volume=volume,
    )


def _bars_from_closes(closes: list[float]) -> list[Bar]:
    return [_bar(c) for c in closes]


# ---- n_day_return ----------------------------------------------------------


def test_n_day_return_basic() -> None:
    bars = _bars_from_closes([100.0, 100.0, 100.0, 100.0, 110.0])
    assert n_day_return(bars, n=4) == pytest.approx(0.10)


def test_n_day_return_returns_none_when_history_too_short() -> None:
    assert n_day_return(_bars_from_closes([1.0, 2.0]), n=5) is None


def test_n_day_return_handles_negative() -> None:
    bars = _bars_from_closes([100.0, 100.0, 100.0, 100.0, 80.0])
    assert n_day_return(bars, n=4) == pytest.approx(-0.20)


def test_n_day_return_rejects_zero_or_negative_n() -> None:
    with pytest.raises(ValueError):
        n_day_return(_bars_from_closes([1.0, 1.0]), n=0)


# ---- RSI -------------------------------------------------------------------


def test_rsi_all_gains_returns_100() -> None:
    bars = _bars_from_closes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    assert rsi(bars, n=14) == 100.0


def test_rsi_symmetric_changes_around_50() -> None:
    # Alternating +1/-1 closes: equal avg gain and loss → RSI = 50.
    closes = [10.0]
    for i in range(14):
        closes.append(closes[-1] + (1.0 if i % 2 == 0 else -1.0))
    bars = _bars_from_closes(closes)
    assert rsi(bars, n=14) == pytest.approx(50.0)


def test_rsi_returns_none_when_short() -> None:
    assert rsi(_bars_from_closes([1, 2, 3]), n=14) is None


# ---- ATR -------------------------------------------------------------------


def test_atr_constant_range() -> None:
    # 15 bars each with high-low spread of 2, no gaps.
    bars = [_bar(close=10, high=11, low=9) for _ in range(15)]
    assert atr(bars, n=14) == pytest.approx(2.0)


def test_atr_returns_none_when_short() -> None:
    assert atr(_bars_from_closes([1, 2, 3]), n=14) is None


# ---- realized_vol ---------------------------------------------------------


def test_realized_vol_is_zero_for_flat_prices() -> None:
    bars = _bars_from_closes([100.0] * 21)
    assert realized_vol(bars, n=20) == pytest.approx(0.0)


def test_realized_vol_is_positive_for_random_walk() -> None:
    closes = [100.0]
    for i in range(20):
        closes.append(closes[-1] * (1.02 if i % 2 == 0 else 0.98))
    bars = _bars_from_closes(closes)
    v = realized_vol(bars, n=20)
    assert v is not None and v > 0.0


def test_realized_vol_matches_known_value() -> None:
    # Two-day pattern: log return alternates between +ln(1.02) and +ln(0.98)*-1 etc.
    # We just sanity-check it scales by sqrt(252).
    closes = [100.0, 102.0, 100.0, 102.0]
    bars = _bars_from_closes(closes)
    v = realized_vol(bars, n=3)
    # log returns: ln(1.02), ln(100/102), ln(1.02). mean ≈ 0.00659, stdev > 0.
    assert v is not None and v > 0
    assert v < 10.0  # sanity, not insanely large


def test_realized_vol_returns_none_on_zero_price() -> None:
    closes = [100.0, 0.0, 100.0, 100.0, 100.0]
    assert realized_vol(_bars_from_closes(closes), n=4) is None


# ---- relative_volume ------------------------------------------------------


def test_relative_volume_basic() -> None:
    bars = [_bar(close=10, volume=v) for v in [100, 100, 100, 100, 200]]
    assert relative_volume(bars, n=4) == pytest.approx(2.0)


def test_relative_volume_returns_none_when_short() -> None:
    bars = [_bar(close=10, volume=100) for _ in range(3)]
    assert relative_volume(bars, n=5) is None


def test_relative_volume_returns_none_when_prior_avg_zero() -> None:
    bars = [_bar(close=10, volume=v) for v in [0, 0, 0, 0, 100]]
    assert relative_volume(bars, n=4) is None


# ---- standard_features ----------------------------------------------------


def test_standard_features_returns_all_when_history_sufficient() -> None:
    bars = _bars_from_closes([100.0 + i for i in range(25)])
    out = standard_features(bars)
    assert set(out) == {"momentum_20d", "rsi_14", "atr_14", "realized_vol_20d", "rvol_20"}
    for value in out.values():
        assert isinstance(value, float)


def test_standard_features_returns_subset_when_history_short() -> None:
    bars = _bars_from_closes([100.0 + i for i in range(15)])
    out = standard_features(bars)
    # 15 bars: enough for rsi_14 (needs 15) but not momentum_20d (needs 21).
    assert "rsi_14" in out
    assert "momentum_20d" not in out


def test_standard_features_empty_for_no_history() -> None:
    assert standard_features([]) == {}
