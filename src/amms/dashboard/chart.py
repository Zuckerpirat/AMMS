"""Equity chart geometry. Builds three series (absolute equity, % delta,
absolute delta) plus axis tick positions for SVG rendering.

Lives in the dashboard layer; pure function, no IO, easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Series:
    key: str
    polyline: str
    area: str
    y_ticks: list[tuple[float, str]] = field(default_factory=list)
    last_label: str = ""
    is_positive: bool = True


@dataclass
class Chart:
    width: int = 600
    height: int = 220
    pad_left: int = 56
    pad_right: int = 56
    pad_top: int = 20
    pad_bottom: int = 28
    x_ticks: list[tuple[float, str]] = field(default_factory=list)
    series: dict[str, Series] = field(default_factory=dict)
    has_data: bool = False

    @property
    def plot_w(self) -> int:
        return self.width - self.pad_left - self.pad_right

    @property
    def plot_h(self) -> int:
        return self.height - self.pad_top - self.pad_bottom

    @property
    def plot_right_x(self) -> int:
        return self.width - self.pad_right

    @property
    def plot_bottom_y(self) -> float:
        return self.pad_top + self.plot_h


def _de_int(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".")


def _fmt_equity(v: float) -> str:
    return f"${_de_int(v)}"


def _fmt_pct(v: float) -> str:
    sign = "+" if v >= 0 else "−"
    return f"{sign}{abs(v):.1f}%".replace(".", ",")


def _fmt_abs(v: float) -> str:
    sign = "+" if v >= 0 else "−"
    return f"{sign}${_de_int(abs(v))}"


FORMATTERS = {"equity": _fmt_equity, "pct": _fmt_pct, "abs": _fmt_abs}


def _build_series(key: str, values: list[float], chart: Chart) -> Series:
    n = len(values)
    lo, hi = min(values), max(values)
    if hi == lo:
        delta = max(abs(lo) * 0.01, 1.0)
        hi = lo + delta
    span = hi - lo
    pad = span * 0.10
    lo, hi = lo - pad, hi + pad
    span = hi - lo

    points: list[tuple[float, float]] = []
    for i, v in enumerate(values):
        x = chart.pad_left + (i / (n - 1)) * chart.plot_w
        y = chart.pad_top + chart.plot_h - ((v - lo) / span) * chart.plot_h
        points.append((x, y))

    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    bottom = chart.plot_bottom_y
    area_segments = " ".join(f"L {x:.1f},{y:.1f}" for x, y in points)
    area = (
        f"M {points[0][0]:.1f},{bottom:.1f} "
        f"L {points[0][0]:.1f},{points[0][1]:.1f} "
        f"{area_segments} "
        f"L {points[-1][0]:.1f},{bottom:.1f} Z"
    )

    fmt = FORMATTERS[key]
    y_ticks: list[tuple[float, str]] = []
    n_ticks = 4
    for i in range(n_ticks + 1):
        t_val = lo + (i / n_ticks) * span
        y_pos = chart.pad_top + chart.plot_h - ((t_val - lo) / span) * chart.plot_h
        y_ticks.append((y_pos, fmt(t_val)))

    return Series(
        key=key,
        polyline=polyline,
        area=area,
        y_ticks=y_ticks,
        last_label=fmt(values[-1]),
        is_positive=values[-1] >= values[0],
    )


def build_equity_chart(history: list[tuple[str, float]]) -> Chart:
    chart = Chart()
    if len(history) < 2:
        return chart
    chart.has_data = True

    values = [v for _, v in history]
    base = values[0]
    series_data: dict[str, list[float]] = {
        "equity": values,
        "pct": [(v - base) / base * 100 if base else 0.0 for v in values],
        "abs": [v - base for v in values],
    }
    for key, vals in series_data.items():
        chart.series[key] = _build_series(key, vals, chart)

    n = len(history)
    n_x = min(5, n)
    seen: set[int] = set()
    for i in range(n_x):
        idx = round(i * (n - 1) / max(n_x - 1, 1))
        if idx in seen:
            continue
        seen.add(idx)
        x = chart.pad_left + (idx / (n - 1)) * chart.plot_w
        raw = history[idx][0]
        try:
            label = date.fromisoformat(raw).strftime("%d.%m.")
        except ValueError:
            label = str(raw)
        chart.x_ticks.append((x, label))

    return chart
