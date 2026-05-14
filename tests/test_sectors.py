from __future__ import annotations

from amms.data.sectors import group_by_sector, sector_for


def test_sector_for_known_ticker() -> None:
    assert sector_for("AAPL") == "Technology"
    assert sector_for("aapl") == "Technology"
    assert sector_for("TSLA") == "Consumer Discretionary"
    assert sector_for("JPM") == "Financials"


def test_sector_for_unknown_ticker_falls_back() -> None:
    assert sector_for("ZZZZ") == "Unclassified"


def test_group_by_sector_aggregates_values() -> None:
    out = group_by_sector(
        [("AAPL", 1000.0), ("MSFT", 2000.0), ("JPM", 500.0), ("XYZ", 100.0)]
    )
    assert out["Technology"] == 3000.0
    assert out["Financials"] == 500.0
    assert out["Unclassified"] == 100.0
