"""Portfolio performance attribution.

Breaks down portfolio P&L by position contribution:
  - Absolute P&L contribution ($ and %)
  - Relative weight in portfolio
  - Contribution to total return (weighted P&L)

Used by /attribution command to show which positions are driving
(or dragging) overall portfolio performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttributionRow:
    symbol: str
    market_value: float
    weight_pct: float         # % of portfolio market value
    unrealized_pnl: float     # $ P&L
    unrealized_pnl_pct: float  # % P&L on position
    contribution_pct: float   # weight × return = contribution to total portfolio return


@dataclass(frozen=True)
class AttributionReport:
    rows: list[AttributionRow]           # sorted by contribution_pct descending
    total_market_value: float
    total_unrealized_pnl: float
    total_return_pct: float              # weighted sum of contributions
    top_contributor: str | None
    top_detractor: str | None


def compute(broker) -> AttributionReport:
    """Compute performance attribution for all open positions."""
    try:
        positions = broker.get_positions()
    except Exception as e:
        logger.warning("failed to get positions: %s", e)
        positions = []

    rows: list[AttributionRow] = []
    total_mv = 0.0
    total_pnl = 0.0

    for p in positions:
        try:
            mv = float(p.market_value)
            pnl = float(p.unrealized_pl)
            qty = float(p.qty)
            entry = float(p.avg_entry_price)
            cost_basis = qty * entry
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            total_mv += mv
            total_pnl += pnl
            rows.append(AttributionRow(
                symbol=p.symbol,
                market_value=mv,
                weight_pct=0.0,  # filled below
                unrealized_pnl=pnl,
                unrealized_pnl_pct=round(pnl_pct, 2),
                contribution_pct=0.0,  # filled below
            ))
        except (TypeError, ValueError, AttributeError):
            continue

    # Compute weights and contributions
    final_rows: list[AttributionRow] = []
    for r in rows:
        weight = (r.market_value / total_mv * 100) if total_mv > 0 else 0.0
        contribution = weight / 100 * r.unrealized_pnl_pct
        final_rows.append(AttributionRow(
            symbol=r.symbol,
            market_value=r.market_value,
            weight_pct=round(weight, 2),
            unrealized_pnl=r.unrealized_pnl,
            unrealized_pnl_pct=r.unrealized_pnl_pct,
            contribution_pct=round(contribution, 3),
        ))

    final_rows.sort(key=lambda x: x.contribution_pct, reverse=True)

    total_return_pct = sum(r.contribution_pct for r in final_rows)
    top_contributor = final_rows[0].symbol if final_rows and final_rows[0].contribution_pct > 0 else None
    top_detractor = final_rows[-1].symbol if final_rows and final_rows[-1].contribution_pct < 0 else None

    return AttributionReport(
        rows=final_rows,
        total_market_value=round(total_mv, 2),
        total_unrealized_pnl=round(total_pnl, 2),
        total_return_pct=round(total_return_pct, 3),
        top_contributor=top_contributor,
        top_detractor=top_detractor,
    )
