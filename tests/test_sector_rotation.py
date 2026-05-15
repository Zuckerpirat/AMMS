from __future__ import annotations

from amms.analysis.sector_rotation import SECTOR_ETFS, SectorHeatRow, SectorMomentum, detect_rotation, sector_heatmap
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


# ── sector_heatmap() tests ────────────────────────────────────────────────────

class _HeatData:
    """Returns 70 bars with different trends per ETF."""
    def get_bars(self, symbol: str, *, limit: int = 70) -> list[Bar]:
        if symbol in ("XLK", "XLF"):
            trend = 0.5   # hot
        elif symbol in ("XLU", "XLRE"):
            trend = -0.3  # cold
        else:
            trend = 0.1
        prices = [100.0 + i * trend for i in range(limit)]
        return [_bar(symbol, p, i) for i, p in enumerate(prices)]


def test_sector_heatmap_returns_all_sectors() -> None:
    rows = sector_heatmap(_HeatData())
    assert len(rows) == len(SECTOR_ETFS)


def test_sector_heatmap_sorted_by_composite() -> None:
    rows = sector_heatmap(_HeatData())
    for i in range(len(rows) - 1):
        assert rows[i].composite_score >= rows[i + 1].composite_score


def test_sector_heatmap_row_fields() -> None:
    rows = sector_heatmap(_HeatData())
    for row in rows:
        assert isinstance(row, SectorHeatRow)
        assert row.sector in SECTOR_ETFS
        assert row.etf in SECTOR_ETFS.values()
        assert row.trend_5d in {"hot", "warm", "cool", "cold", "flat", "n/a"}
        assert row.trend_20d in {"hot", "warm", "cool", "cold", "flat", "n/a"}


def test_sector_heatmap_outperformer_at_top() -> None:
    rows = sector_heatmap(_HeatData())
    top_etfs = {r.etf for r in rows[:3]}
    # XLK and XLF should be near the top
    assert "XLK" in top_etfs or "XLF" in top_etfs


def test_sector_heatmap_empty_data_returns_flat() -> None:
    rows = sector_heatmap(_EmptyData())
    assert all(r.mom_5d is None for r in rows)
    assert all(r.composite_score == 0.0 for r in rows)
