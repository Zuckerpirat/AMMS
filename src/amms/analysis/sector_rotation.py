"""Sector rotation detector.

Uses SPDR sector ETFs to detect which sectors are gaining or losing
relative momentum vs the broad market (SPY). This is a macro-level
analysis that feeds into mode/regime decisions.

ETF mapping (SPDR sector funds):
  XLK  — Technology
  XLV  — Health Care
  XLF  — Financials
  XLY  — Consumer Discretionary
  XLP  — Consumer Staples
  XLE  — Energy
  XLI  — Industrials
  XLB  — Materials
  XLRE — Real Estate
  XLU  — Utilities
  XLC  — Communication Services

Rotation is detected by comparing 20-day momentum of each sector ETF
to SPY momentum. Sectors outperforming SPY are "rotating in".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SECTOR_ETFS: dict[str, str] = {
    "Technology":               "XLK",
    "Health Care":              "XLV",
    "Financials":               "XLF",
    "Consumer Discr.":          "XLY",
    "Consumer Staples":         "XLP",
    "Energy":                   "XLE",
    "Industrials":              "XLI",
    "Materials":                "XLB",
    "Real Estate":              "XLRE",
    "Utilities":                "XLU",
    "Communication Svcs":       "XLC",
}


@dataclass(frozen=True)
class SectorMomentum:
    sector: str
    etf: str
    momentum_20d: float | None   # 20-day % return
    vs_spy: float | None         # outperformance vs SPY (positive = rotating in)
    trend: str                   # "in" | "out" | "neutral" | "unknown"


def detect_rotation(data, *, n: int = 20) -> list[SectorMomentum]:
    """Compute sector momentum relative to SPY.

    Returns list sorted by vs_spy descending (best performers first).
    """
    from amms.features.momentum import n_day_return

    # Get SPY momentum as baseline
    spy_mom = None
    try:
        spy_bars = data.get_bars("SPY", limit=n + 5)
        spy_mom = n_day_return(spy_bars, n)
    except Exception:
        pass

    results: list[SectorMomentum] = []
    for sector, etf in SECTOR_ETFS.items():
        try:
            bars = data.get_bars(etf, limit=n + 5)
            mom = n_day_return(bars, n)
        except Exception:
            mom = None

        if mom is None:
            results.append(SectorMomentum(sector, etf, None, None, "unknown"))
            continue

        vs_spy = None
        trend = "neutral"
        if spy_mom is not None:
            vs_spy = (mom - spy_mom) * 100
            if vs_spy > 1.0:
                trend = "in"
            elif vs_spy < -1.0:
                trend = "out"

        results.append(SectorMomentum(
            sector=sector,
            etf=etf,
            momentum_20d=mom * 100,
            vs_spy=vs_spy,
            trend=trend,
        ))

    results.sort(
        key=lambda x: (x.vs_spy is not None, x.vs_spy or 0.0),
        reverse=True,
    )
    return results
