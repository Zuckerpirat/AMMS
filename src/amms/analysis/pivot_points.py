"""Pivot point calculator.

Computes pivot points from the previous period's high, low, and close.
Three methods are supported:

1. Classic (Standard):
   PP = (H + L + C) / 3
   R1 = 2*PP - L,  S1 = 2*PP - H
   R2 = PP + (H-L), S2 = PP - (H-L)
   R3 = H + 2*(PP-L), S3 = L - 2*(H-PP)

2. Fibonacci:
   PP = (H + L + C) / 3
   R1 = PP + 0.382*(H-L), R2 = PP + 0.618*(H-L), R3 = PP + 1.0*(H-L)
   S1 = PP - 0.382*(H-L), S2 = PP - 0.618*(H-L), S3 = PP - 1.0*(H-L)

3. Camarilla:
   PP = (H + L + C) / 3
   R1 = C + (H-L)*1.1/12,  R2 = C + (H-L)*1.1/6
   R3 = C + (H-L)*1.1/4,   R4 = C + (H-L)*1.1/2
   S1 = C - (H-L)*1.1/12,  S2 = C - (H-L)*1.1/6
   S3 = C - (H-L)*1.1/4,   S4 = C - (H-L)*1.1/2

Also identifies which pivot zone the current price is in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PivotLevel:
    name: str
    price: float
    kind: str    # "R" (resistance) / "S" (support) / "PP" (pivot)


@dataclass(frozen=True)
class PivotPointReport:
    symbol: str
    method: str               # "classic" / "fibonacci" / "camarilla"
    pivot: float
    levels: list[PivotLevel]  # all support/resistance levels
    current_price: float
    nearest_resistance: float | None
    nearest_support: float | None
    current_zone: str         # description of where price is
    period_high: float
    period_low: float
    period_close: float
    verdict: str


def _classic(h: float, l: float, c: float) -> list[PivotLevel]:
    pp = (h + l + c) / 3
    r = h - l
    return [
        PivotLevel("PP", pp, "PP"),
        PivotLevel("R1", 2 * pp - l, "R"),
        PivotLevel("R2", pp + r, "R"),
        PivotLevel("R3", h + 2 * (pp - l), "R"),
        PivotLevel("S1", 2 * pp - h, "S"),
        PivotLevel("S2", pp - r, "S"),
        PivotLevel("S3", l - 2 * (h - pp), "S"),
    ]


def _fibonacci(h: float, l: float, c: float) -> list[PivotLevel]:
    pp = (h + l + c) / 3
    r = h - l
    return [
        PivotLevel("PP", pp, "PP"),
        PivotLevel("R1", pp + 0.382 * r, "R"),
        PivotLevel("R2", pp + 0.618 * r, "R"),
        PivotLevel("R3", pp + 1.000 * r, "R"),
        PivotLevel("S1", pp - 0.382 * r, "S"),
        PivotLevel("S2", pp - 0.618 * r, "S"),
        PivotLevel("S3", pp - 1.000 * r, "S"),
    ]


def _camarilla(h: float, l: float, c: float) -> list[PivotLevel]:
    pp = (h + l + c) / 3
    r = h - l
    return [
        PivotLevel("PP", pp, "PP"),
        PivotLevel("R1", c + r * 1.1 / 12, "R"),
        PivotLevel("R2", c + r * 1.1 / 6, "R"),
        PivotLevel("R3", c + r * 1.1 / 4, "R"),
        PivotLevel("R4", c + r * 1.1 / 2, "R"),
        PivotLevel("S1", c - r * 1.1 / 12, "S"),
        PivotLevel("S2", c - r * 1.1 / 6, "S"),
        PivotLevel("S3", c - r * 1.1 / 4, "S"),
        PivotLevel("S4", c - r * 1.1 / 2, "S"),
    ]


METHODS = {"classic": _classic, "fibonacci": _fibonacci, "camarilla": _camarilla}


def compute(
    period_high: float,
    period_low: float,
    period_close: float,
    current_price: float,
    *,
    symbol: str = "",
    method: str = "classic",
) -> PivotPointReport | None:
    """Compute pivot points from period OHLC.

    period_high/low/close: previous session's H, L, C (daily/weekly/monthly).
    current_price: today's last price.
    method: "classic", "fibonacci", or "camarilla".
    Returns None for invalid input.
    """
    try:
        h, l, c = float(period_high), float(period_low), float(period_close)
        cp = float(current_price)
    except (TypeError, ValueError):
        return None

    if h < l or h <= 0 or l <= 0 or c <= 0 or cp <= 0:
        return None

    calc = METHODS.get(method, _classic)
    levels = calc(h, l, c)
    pp = next(lv.price for lv in levels if lv.kind == "PP")

    # Sort all levels by price
    sorted_levels = sorted(levels, key=lambda lv: lv.price)

    # Find nearest support and resistance to current price
    supports = sorted([lv for lv in levels if lv.kind == "S"], key=lambda x: x.price, reverse=True)
    resistances = sorted([lv for lv in levels if lv.kind == "R"], key=lambda x: x.price)

    nearest_sup = next((s.price for s in supports if s.price < cp), None)
    nearest_res = next((r.price for r in resistances if r.price > cp), None)

    # Current zone
    if cp > max(lv.price for lv in levels if lv.kind == "R"):
        zone = "above all resistance — very strong"
    elif cp < min(lv.price for lv in levels if lv.kind == "S"):
        zone = "below all support — very weak"
    elif cp > pp:
        zone = "above pivot — bullish bias"
    else:
        zone = "below pivot — bearish bias"

    res_str = f"{nearest_res:.2f}" if nearest_res else "none"
    sup_str = f"{nearest_sup:.2f}" if nearest_sup else "none"
    verdict = (
        f"Price {cp:.2f} is {zone}. "
        f"Nearest resistance: {res_str}, nearest support: {sup_str}."
    )

    return PivotPointReport(
        symbol=symbol,
        method=method,
        pivot=round(pp, 2),
        levels=[PivotLevel(lv.name, round(lv.price, 2), lv.kind) for lv in sorted_levels],
        current_price=round(cp, 2),
        nearest_resistance=round(nearest_res, 2) if nearest_res else None,
        nearest_support=round(nearest_sup, 2) if nearest_sup else None,
        current_zone=zone,
        period_high=round(h, 2),
        period_low=round(l, 2),
        period_close=round(c, 2),
        verdict=verdict,
    )
