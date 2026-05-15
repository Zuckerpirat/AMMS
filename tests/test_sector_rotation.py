from __future__ import annotations

from amms.analysis.sector_rotation import SECTOR_ETFS, SectorMomentum, detect_rotation
from amms.data.bars import Bar


def _bar(sym: str, close: float, i: int) -> Bar:
    return Bar(sym, "1D", f"2026-01-{1 + i % 28:02d}", close, close + 1, close - 1, close, 1000)


class _FakeData:
    def __init__(self, trend: float = 0.3) -> None:
        self._trend = trend

    def get_bars(self, symbol: str, *, limit: int = 25) -> list[Bar]:
        # SPY: steady trend; sectors: slightly better or worse
        if symbol == "SPY":
            trend = self._trend
        elif symbol in ("XLK", "XLF"):
            trend = self._trend + 0.5  # outperform
        else:
            trend = self._trend - 0.3  # underperform
        prices = [100.0 + i * trend for i in range(limit)]
        return [_bar(symbol, p, i) for i, p in enumerate(prices)]


class _EmptyData:
    def get_bars(self, symbol: str, *, limit: int = 25) -> list[Bar]:
        return []


def test_detect_rotation_returns_all_sectors() -> None:
    data = _FakeData()
    results = detect_rotation(data)
    assert len(results) == len(SECTOR_ETFS)


def test_detect_rotation_sorted_by_vs_spy() -> None:
    data = _FakeData()
    results = detect_rotation(data)
    for i in range(len(results) - 1):
        a = results[i].vs_spy
        b = results[i + 1].vs_spy
        if a is not None and b is not None:
            assert a >= b


def test_outperformer_marked_as_in() -> None:
    data = _FakeData()
    results = detect_rotation(data)
    xlk = next((r for r in results if r.etf == "XLK"), None)
    assert xlk is not None
    assert xlk.trend == "in"


def test_underperformer_marked_as_out() -> None:
    data = _FakeData()
    results = detect_rotation(data)
    out_sectors = [r for r in results if r.trend == "out"]
    assert len(out_sectors) > 0


def test_empty_data_returns_unknowns() -> None:
    results = detect_rotation(_EmptyData())
    assert all(r.trend == "unknown" for r in results)


def test_sector_momentum_dataclass() -> None:
    sm = SectorMomentum("Tech", "XLK", 5.0, 2.0, "in")
    assert sm.sector == "Tech"
    assert sm.etf == "XLK"
    assert sm.trend == "in"


def test_sector_etfs_has_11_entries() -> None:
    assert len(SECTOR_ETFS) == 11
