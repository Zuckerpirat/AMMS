"""Static ticker → GICS-sector mapping for US equities.

A simple, hand-curated table for the watchlist of tickers the bot
actually trades. Unknown tickers fall into "Unclassified". Adding a
new ticker = adding a line. CLAUDE.md's "estimate sector exposure"
goal is built on top of this without taking a new external dependency.
"""

from __future__ import annotations

from typing import Iterable

_SECTORS: dict[str, str] = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "AVGO": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "ORCL": "Technology",
    "CRM": "Technology",
    "ADBE": "Technology",
    "MU": "Technology",
    "AMAT": "Technology",
    "TSM": "Technology",
    "QCOM": "Technology",
    "TXN": "Technology",
    "SNDK": "Technology",
    "ASML": "Technology",
    "PLTR": "Technology",
    "POET": "Technology",
    "NBIS": "Technology",
    # Communication Services
    "GOOG": "Communication Services",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "DIS": "Communication Services",
    "T": "Communication Services",
    "VZ": "Communication Services",
    "RDDT": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "RIVN": "Consumer Discretionary",
    "LCID": "Consumer Discretionary",
    "GME": "Consumer Discretionary",
    "AMC": "Consumer Discretionary",
    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "BRK.B": "Financials",
    "V": "Financials",
    "MA": "Financials",
    "SOFI": "Financials",
    "HOOD": "Financials",
    "COIN": "Financials",
    # Healthcare
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "UNH": "Healthcare",
    # Consumer Staples
    "WMT": "Consumer Staples",
    "COST": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    # Industrials
    "BA": "Industrials",
    "GE": "Industrials",
    # Crypto / Mining
    "MARA": "Crypto-adjacent",
    "RIOT": "Crypto-adjacent",
    # ETFs
    "SPY": "Broad ETF",
    "QQQ": "Broad ETF",
    "IWM": "Broad ETF",
    "VOO": "Broad ETF",
    "VTI": "Broad ETF",
    "DIA": "Broad ETF",
    "ARKK": "Thematic ETF",
    "TQQQ": "Leveraged ETF",
    "SQQQ": "Leveraged ETF",
    "VIXY": "Volatility",
}


UNCLASSIFIED = "Unclassified"


def sector_for(symbol: str) -> str:
    return _SECTORS.get(symbol.upper(), UNCLASSIFIED)


def group_by_sector(
    positions: Iterable[tuple[str, float]],
) -> dict[str, float]:
    """Aggregate (symbol, value) pairs into a {sector: total_value} map."""
    out: dict[str, float] = {}
    for sym, value in positions:
        sector = sector_for(sym)
        out[sector] = out.get(sector, 0.0) + float(value)
    return out
