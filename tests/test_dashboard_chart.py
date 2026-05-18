from __future__ import annotations

from datetime import date, timedelta

from amms.dashboard.chart import build_equity_chart


def _hist(values: list[float]) -> list[tuple[str, float]]:
    today = date.today()
    return [
        ((today - timedelta(days=len(values) - 1 - i)).isoformat(), v)
        for i, v in enumerate(values)
    ]


def test_empty_history_returns_chart_without_data() -> None:
    chart = build_equity_chart([])
    assert not chart.has_data
    assert chart.series == {}
    assert chart.x_ticks == []


def test_single_point_history_returns_no_data() -> None:
    chart = build_equity_chart(_hist([100.0]))
    assert not chart.has_data


def test_chart_has_three_series() -> None:
    chart = build_equity_chart(_hist([100.0, 102.0, 101.0, 105.0, 110.0]))
    assert chart.has_data
    assert set(chart.series.keys()) == {"equity", "pct", "abs"}


def test_pct_series_starts_at_zero_relative() -> None:
    chart = build_equity_chart(_hist([100.0, 110.0]))
    pct = chart.series["pct"]
    assert pct.last_label.startswith("+")
    assert "10,0" in pct.last_label


def test_abs_series_shows_signed_delta() -> None:
    chart = build_equity_chart(_hist([100.0, 90.0]))
    abs_series = chart.series["abs"]
    assert abs_series.last_label.startswith("−")
    assert not abs_series.is_positive


def test_equity_series_shows_absolute_value() -> None:
    chart = build_equity_chart(_hist([1000.0, 1500.0]))
    eq = chart.series["equity"]
    assert "1.500" in eq.last_label or "$1.500" in eq.last_label


def test_x_ticks_use_german_date_format() -> None:
    chart = build_equity_chart(_hist([100.0, 101.0, 102.0, 103.0, 104.0]))
    assert chart.x_ticks, "should produce at least one x tick"
    sample = chart.x_ticks[0][1]
    assert sample.endswith(".")  # DD.MM. format


def test_flat_history_does_not_divide_by_zero() -> None:
    chart = build_equity_chart(_hist([100.0, 100.0, 100.0]))
    assert chart.has_data
    assert chart.series["pct"].last_label is not None
