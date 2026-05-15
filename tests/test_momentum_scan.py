from __future__ import annotations

import pytest

from amms.analysis.momentum_scan import ScanResult, scan
from amms.data.bars import Bar


def _bar(symbol: str, close: float, i: int) -> Bar:
    return Bar(symbol, "1D", f"2026-01-{1 + i % 28:02d}", close, close + 1, close - 1, close, 1000)


class _FakeData:
    def __init__(self, prices_by_sym: dict[str, list[float]]) -> None:
        self._data = prices_by_sym

    def get_bars(self, symbol: str, *, limit: int = 60) -> list[Bar]:
        prices = self._data.get(symbol, [])[-limit:]
        return [_bar(symbol, p, i) for i, p in enumerate(prices)]


def _trending_up(n: int = 65, start: float = 100.0) -> list[float]:
    return [start + i * 0.3 for i in range(n)]


def _flat(n: int = 65, price: float = 100.0) -> list[float]:
    return [price] * n


def _trending_down(n: int = 65, start: float = 150.0) -> list[float]:
    return [start - i * 0.3 for i in range(n)]


def test_scan_empty_symbols() -> None:
    data = _FakeData({})
    results = scan([], data)
    assert results == []


def test_scan_returns_sorted_by_score() -> None:
    data = _FakeData({
        "AAPL": _trending_up(),
        "MSFT": _flat(),
        "TSLA": _trending_down(),
    })
    results = scan(["AAPL", "MSFT", "TSLA"], data, top_n=3)
    assert len(results) > 0
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_scan_respects_top_n() -> None:
    syms = [f"S{i}" for i in range(10)]
    data = _FakeData({s: _trending_up() for s in syms})
    results = scan(syms, data, top_n=3)
    assert len(results) <= 3


def test_scan_score_range() -> None:
    data = _FakeData({"AAPL": _trending_up()})
    results = scan(["AAPL"], data)
    assert len(results) == 1
    assert 0 <= results[0].score <= 100


def test_scan_missing_data_skipped() -> None:
    data = _FakeData({"AAPL": _trending_up(), "MISSING": []})
    results = scan(["AAPL", "MISSING"], data)
    syms = [r.symbol for r in results]
    assert "AAPL" in syms
    assert "MISSING" not in syms


def test_scan_result_has_fields() -> None:
    data = _FakeData({"AAPL": _trending_up()})
    results = scan(["AAPL"], data)
    r = results[0]
    assert r.symbol == "AAPL"
    assert r.ema_trend in ("strong_bull", "weak_bull", "bear", "unknown")
    assert isinstance(r.reason, str)


def test_trending_up_scores_higher_than_trending_down() -> None:
    data = _FakeData({
        "UP": _trending_up(),
        "DOWN": _trending_down(),
    })
    results = scan(["UP", "DOWN"], data, top_n=2)
    scores = {r.symbol: r.score for r in results}
    if "UP" in scores and "DOWN" in scores:
        assert scores["UP"] >= scores["DOWN"]


def test_scan_no_data_client_error() -> None:
    class _Broken:
        def get_bars(self, sym, *, limit=60):
            raise RuntimeError("fail")
    results = scan(["AAPL"], _Broken())
    assert results == []
