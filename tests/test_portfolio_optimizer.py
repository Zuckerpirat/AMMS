from __future__ import annotations

import pytest

from amms.analysis.portfolio_optimizer import AllocationResult, format_allocation, optimize
from amms.data.bars import Bar


def _bar(sym: str, close: float, i: int) -> Bar:
    return Bar(sym, "1D", f"2026-01-{1 + i % 28:02d}", close, close + 1, close - 1, close, 1000)


class _FakeData:
    def get_bars(self, symbol: str, *, limit: int = 60) -> list[Bar]:
        prices = [100.0 + i * 0.5 + hash(symbol) % 5 for i in range(65)][-limit:]
        return [_bar(symbol, p, i) for i, p in enumerate(prices)]


def test_equal_weight_sums_to_one() -> None:
    syms = ["AAPL", "MSFT", "NVDA"]
    results = optimize(syms, _FakeData(), mode="equal_weight")
    assert len(results) == 3
    total = sum(r.weight for r in results)
    assert total == pytest.approx(1.0)


def test_equal_weight_each_is_1_over_n() -> None:
    syms = ["A", "B", "C", "D"]
    results = optimize(syms, _FakeData(), mode="equal_weight")
    for r in results:
        assert r.weight == pytest.approx(0.25)


def test_momentum_weights_sum_to_one() -> None:
    syms = ["AAPL", "MSFT"]
    results = optimize(syms, _FakeData(), mode="momentum")
    total = sum(r.weight for r in results)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_inverse_vol_weights_sum_to_one() -> None:
    syms = ["AAPL", "MSFT", "TSLA"]
    results = optimize(syms, _FakeData(), mode="inverse_vol")
    total = sum(r.weight for r in results)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_empty_symbols_returns_empty() -> None:
    assert optimize([], _FakeData()) == []


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        optimize(["AAPL"], _FakeData(), mode="magic")


def test_format_allocation_basic() -> None:
    results = [
        AllocationResult("AAPL", 0.6, 60.0, "equal_weight"),
        AllocationResult("MSFT", 0.4, 40.0, "equal_weight"),
    ]
    out = format_allocation(results, equity=100_000)
    assert "AAPL" in out
    assert "60.0%" in out
    assert "$60,000" in out


def test_format_allocation_empty() -> None:
    assert format_allocation([]) == "No allocation computed."


def test_weight_pct_matches_weight() -> None:
    syms = ["A", "B"]
    results = optimize(syms, _FakeData(), mode="equal_weight")
    for r in results:
        assert r.weight_pct == pytest.approx(r.weight * 100)
