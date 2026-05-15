"""Portfolio stress test.

Simulates what happens to the current portfolio under various historical
or hypothetical stress scenarios:

  - 2008 Crisis    : -50% market drop over 6 months
  - 2020 COVID     : -35% in 1 month, then V-shaped recovery
  - Dot-com crash  : -78% over 2.5 years (tech heavy)
  - Flash crash    : -10% intraday then recovers
  - Rising rates   : bonds down 20%, growth stocks -30%
  - Custom shock   : user-defined % drop

Each scenario assumes the portfolio moves proportionally to the market
(beta = 1.0 for simplicity). Beta-adjusted version can be added later.

Returns estimated portfolio loss, new equity, and per-position impact.
"""

from __future__ import annotations

from dataclasses import dataclass

SCENARIOS: dict[str, float] = {
    "2008_crisis":       -0.50,  # S&P 500 peak-to-trough
    "2020_covid":        -0.35,  # max drawdown in Feb-Mar 2020
    "dotcom_crash":      -0.49,  # S&P 500 Mar 2000 - Oct 2002
    "flash_crash_2010":  -0.10,  # May 6, 2010 intraday
    "rising_rates":      -0.25,  # estimated broad market impact
    "mild_correction":   -0.10,  # typical 10% correction
    "severe_bear":       -0.40,  # generic severe bear market
}


@dataclass(frozen=True)
class StressPosition:
    symbol: str
    current_value: float
    stressed_value: float
    loss: float
    loss_pct: float


@dataclass(frozen=True)
class StressResult:
    scenario: str
    shock_pct: float          # the applied shock (e.g. -0.35 = -35%)
    initial_total_mv: float
    stressed_total_mv: float
    total_loss: float
    total_loss_pct: float
    positions: list[StressPosition]
    verdict: str              # "low_risk" | "moderate" | "severe" | "critical"


def stress_test(
    broker,
    scenario: str = "2008_crisis",
    *,
    custom_shock_pct: float | None = None,
) -> StressResult | None:
    """Apply a stress scenario to current positions.

    scenario: one of SCENARIOS keys, or "custom" to use custom_shock_pct
    custom_shock_pct: e.g. -0.20 for a 20% drop (used when scenario="custom")
    """
    if scenario == "custom":
        if custom_shock_pct is None:
            return None
        shock = custom_shock_pct
        label = f"custom_{abs(int(custom_shock_pct * 100))}pct"
    else:
        shock = SCENARIOS.get(scenario)
        if shock is None:
            return None
        label = scenario

    try:
        positions = broker.get_positions()
    except Exception:
        return None

    stressed_positions: list[StressPosition] = []
    total_mv = 0.0
    total_stressed = 0.0

    for p in positions:
        try:
            mv = float(p.market_value)
            stressed = mv * (1 + shock)
            loss = stressed - mv
            loss_pct = shock * 100
            stressed_positions.append(StressPosition(
                symbol=p.symbol,
                current_value=round(mv, 2),
                stressed_value=round(stressed, 2),
                loss=round(loss, 2),
                loss_pct=round(loss_pct, 1),
            ))
            total_mv += mv
            total_stressed += stressed
        except (TypeError, ValueError):
            continue

    if total_mv == 0:
        return None

    total_loss = total_stressed - total_mv
    total_loss_pct = shock * 100

    if abs(total_loss_pct) < 15:
        verdict = "low_risk"
    elif abs(total_loss_pct) < 25:
        verdict = "moderate"
    elif abs(total_loss_pct) < 40:
        verdict = "severe"
    else:
        verdict = "critical"

    return StressResult(
        scenario=label,
        shock_pct=round(shock * 100, 1),
        initial_total_mv=round(total_mv, 2),
        stressed_total_mv=round(total_stressed, 2),
        total_loss=round(total_loss, 2),
        total_loss_pct=round(total_loss_pct, 1),
        positions=stressed_positions,
        verdict=verdict,
    )
