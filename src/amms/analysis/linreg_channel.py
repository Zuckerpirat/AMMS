"""Linear Regression Channel Analyser.

Fits a least-squares linear regression line to price history and computes
±1σ and ±2σ channel bands from the regression residuals. Measures trend
angle, R², and price deviation from the fitted line.

Outputs:
  - Slope (price change per bar) and annualised slope %
  - R² (goodness of fit — how "clean" the trend is)
  - Regression midline + upper/lower ±1σ/±2σ bands at current bar
  - Price position within channel
  - Residual z-score (how many std devs from the regression line)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LRCBand:
    sigma: float    # -2, -1, 0, +1, +2
    price: float


@dataclass(frozen=True)
class LRCReport:
    symbol: str
    period: int

    # Regression parameters
    slope: float           # price change per bar
    slope_pct_annual: float  # annualised slope as % of starting price
    intercept: float
    r_squared: float       # 0-1, fit quality

    # Current bar bands
    regression_line: float  # fitted price at current bar
    bands: list[LRCBand]    # -2σ, -1σ, 0, +1σ, +2σ

    # Price position
    current_price: float
    residual: float          # price - fitted (positive = above line)
    residual_z: float        # residual / residual_std
    position_label: str      # "above_upper", "upper_half", "on_line", "lower_half", "below_lower"

    # Trend quality
    trend_direction: str    # "up", "down", "flat"
    r_squared_label: str    # "strong", "moderate", "weak"

    bars_used: int
    verdict: str


def _linear_regression(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if ys else 0.0, 0.0

    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(xs[i] * ys[i] for i in range(n))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n, 0.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n

    # R²
    y_mean = sum_y / n
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((ys[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    r2 = max(0.0, min(1.0, r2))

    return slope, intercept, r2


def analyze(
    bars: list,
    *,
    symbol: str = "",
    period: int = 50,
    annualize: int = 252,
) -> LRCReport | None:
    """Fit linear regression channel to price history.

    bars: bar objects with .close attribute.
    period: number of bars to fit the regression on.
    annualize: trading days per year for slope annualisation.
    """
    if not bars or len(bars) < max(period, 10):
        return None

    try:
        closes = [float(b.close) for b in bars]
    except (AttributeError, TypeError, ValueError):
        return None

    current = closes[-1]
    if current <= 0:
        return None

    # Use last `period` bars
    subset = closes[-period:]
    n = len(subset)
    xs = list(range(n))

    slope, intercept, r2 = _linear_regression(xs, subset)

    # Residuals
    fitted = [slope * x + intercept for x in xs]
    residuals = [subset[i] - fitted[i] for i in range(n)]
    residual_std = math.sqrt(sum(r ** 2 for r in residuals) / n) if n > 0 else 1e-9

    # Current bar (x = n-1)
    cur_fitted = slope * (n - 1) + intercept
    cur_residual = current - cur_fitted
    residual_z = cur_residual / residual_std if residual_std > 0 else 0.0

    # Bands at current position
    bands = [
        LRCBand(sigma=-2.0, price=round(cur_fitted - 2 * residual_std, 4)),
        LRCBand(sigma=-1.0, price=round(cur_fitted - residual_std, 4)),
        LRCBand(sigma=0.0,  price=round(cur_fitted, 4)),
        LRCBand(sigma=1.0,  price=round(cur_fitted + residual_std, 4)),
        LRCBand(sigma=2.0,  price=round(cur_fitted + 2 * residual_std, 4)),
    ]

    # Position label
    if residual_z > 2.0:
        pos_label = "above_upper"
    elif residual_z > 0.5:
        pos_label = "upper_half"
    elif residual_z < -2.0:
        pos_label = "below_lower"
    elif residual_z < -0.5:
        pos_label = "lower_half"
    else:
        pos_label = "on_line"

    # Slope annualised
    if subset[0] > 0:
        slope_pct_annual = slope * annualize / subset[0] * 100.0
    else:
        slope_pct_annual = 0.0

    # Trend direction
    if slope > residual_std * 0.05:
        trend_dir = "up"
    elif slope < -residual_std * 0.05:
        trend_dir = "down"
    else:
        trend_dir = "flat"

    # R² label
    if r2 >= 0.7:
        r2_label = "strong"
    elif r2 >= 0.4:
        r2_label = "moderate"
    else:
        r2_label = "weak"

    # Verdict
    verdict = (
        f"LinReg channel ({period} bars): slope {slope_pct_annual:+.1f}%/yr, "
        f"R²={r2:.2f} ({r2_label} fit), "
        f"price {pos_label.replace('_', ' ')} (z={residual_z:+.2f}σ)."
    )

    return LRCReport(
        symbol=symbol,
        period=period,
        slope=round(slope, 4),
        slope_pct_annual=round(slope_pct_annual, 2),
        intercept=round(intercept, 4),
        r_squared=round(r2, 4),
        regression_line=round(cur_fitted, 4),
        bands=bands,
        current_price=round(current, 4),
        residual=round(cur_residual, 4),
        residual_z=round(residual_z, 3),
        position_label=pos_label,
        trend_direction=trend_dir,
        r_squared_label=r2_label,
        bars_used=len(bars),
        verdict=verdict,
    )
