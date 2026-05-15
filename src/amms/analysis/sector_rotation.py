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


@dataclass(frozen=True)
class SectorHeatRow:
    sector: str
    etf: str
    mom_5d: float | None     # 5-day return %
    mom_20d: float | None    # 20-day return %
    mom_60d: float | None    # 60-day return %
    trend_5d: str            # "hot" | "warm" | "cool" | "cold" | "n/a"
    trend_20d: str
    composite_score: float   # weighted average, higher = stronger momentum


def sector_heatmap(data) -> list[SectorHeatRow]:
    """Compute multi-timeframe sector momentum heatmap.

    Returns rows sorted by composite_score descending.
    Composite score: 0.2 × 5d + 0.5 × 20d + 0.3 × 60d
    """
    from amms.features.momentum import n_day_return

    def classify(pct: float | None) -> str:
        if pct is None:
            return "n/a"
        if pct > 3.0:
            return "hot"
        if pct > 0.5:
            return "warm"
        if pct < -3.0:
            return "cold"
        if pct < -0.5:
            return "cool"
        return "flat"

    rows: list[SectorHeatRow] = []
    for sector, etf in SECTOR_ETFS.items():
        try:
            bars = data.get_bars(etf, limit=70)
        except Exception:
            bars = []

        from amms.features.momentum import n_day_return as ndr
        m5 = ndr(bars, 5)
        m5 = m5 * 100 if m5 is not None else None
        m20 = ndr(bars, 20)
        m20 = m20 * 100 if m20 is not None else None
        m60 = ndr(bars, 60)
        m60 = m60 * 100 if m60 is not None else None

        score_parts = []
        weights = []
        if m5 is not None:
            score_parts.append(m5 * 0.2)
            weights.append(0.2)
        if m20 is not None:
            score_parts.append(m20 * 0.5)
            weights.append(0.5)
        if m60 is not None:
            score_parts.append(m60 * 0.3)
            weights.append(0.3)

        composite = sum(score_parts) / sum(weights) if weights else 0.0

        rows.append(SectorHeatRow(
            sector=sector,
            etf=etf,
            mom_5d=round(m5, 2) if m5 is not None else None,
            mom_20d=round(m20, 2) if m20 is not None else None,
            mom_60d=round(m60, 2) if m60 is not None else None,
            trend_5d=classify(m5),
            trend_20d=classify(m20),
            composite_score=round(composite, 2),
        ))

    rows.sort(key=lambda r: r.composite_score, reverse=True)
    return rows
