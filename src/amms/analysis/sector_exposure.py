"""Portfolio sector exposure analysis.

Maps portfolio positions to GICS sectors using a built-in symbol→sector
lookup table (covers ~500 common US stocks and ETFs).

Computes:
  - Portfolio weight per sector
  - Number of positions per sector
  - Sector HHI (concentration)
  - Overweight / underweight vs S&P 500 benchmark weights
  - Risk flags: single sector > 40%, no defensive exposure, etc.

S&P 500 approximate sector weights (as of 2026):
  Technology:             30%
  Financials:             13%
  Health Care:            12%
  Consumer Discretionary: 10%
  Communication Services:  9%
  Industrials:             8%
  Consumer Staples:        6%
  Energy:                  4%
  Utilities:               2%
  Materials:               2%
  Real Estate:             2%
"""

from __future__ import annotations

from dataclasses import dataclass

# Approximate S&P 500 benchmark sector weights (pct)
BENCHMARK_WEIGHTS: dict[str, float] = {
    "Technology": 30.0,
    "Financials": 13.0,
    "Health Care": 12.0,
    "Consumer Discretionary": 10.0,
    "Communication Services": 9.0,
    "Industrials": 8.0,
    "Consumer Staples": 6.0,
    "Energy": 4.0,
    "Utilities": 2.0,
    "Materials": 2.0,
    "Real Estate": 2.0,
    "Unknown": 0.0,
}

# Symbol → Sector mapping (common US stocks + ETFs)
SYMBOL_SECTOR: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "INTC": "Technology", "AVGO": "Technology",
    "QCOM": "Technology", "TXN": "Technology", "MU": "Technology",
    "AMAT": "Technology", "LRCX": "Technology", "KLAC": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "SAP": "Technology",
    "ADBE": "Technology", "NOW": "Technology", "INTU": "Technology",
    "PANW": "Technology", "CRWD": "Technology", "ZS": "Technology",
    "SNPS": "Technology", "CDNS": "Technology", "MRVL": "Technology",
    "HPQ": "Technology", "DELL": "Technology", "IBM": "Technology",
    "XLK": "Technology",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "AXP": "Financials", "BLK": "Financials", "SCHW": "Financials",
    "COF": "Financials", "USB": "Financials", "TFC": "Financials",
    "PNC": "Financials", "BK": "Financials", "STT": "Financials",
    "V": "Financials", "MA": "Financials", "PYPL": "Financials",
    "SQ": "Financials", "XLF": "Financials",
    # Health Care
    "JNJ": "Health Care", "UNH": "Health Care", "LLY": "Health Care",
    "PFE": "Health Care", "MRK": "Health Care", "ABBV": "Health Care",
    "ABT": "Health Care", "TMO": "Health Care", "DHR": "Health Care",
    "BMY": "Health Care", "AMGN": "Health Care", "GILD": "Health Care",
    "VRTX": "Health Care", "REGN": "Health Care", "BIIB": "Health Care",
    "ISRG": "Health Care", "SYK": "Health Care", "MDT": "Health Care",
    "CVS": "Health Care", "HUM": "Health Care", "CI": "Health Care",
    "XLV": "Health Care",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "SBUX": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "F": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "ABNB": "Consumer Discretionary",
    "CMG": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "TGT": "Consumer Discretionary", "ETSY": "Consumer Discretionary",
    "XLY": "Consumer Discretionary",
    # Communication Services
    "META": "Communication Services", "GOOGL": "Communication Services",
    "GOOG": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "CMCSA": "Communication Services",
    "T": "Communication Services", "VZ": "Communication Services",
    "SNAP": "Communication Services", "PINS": "Communication Services",
    "RDDT": "Communication Services", "SPOT": "Communication Services",
    "EA": "Communication Services", "TTWO": "Communication Services",
    "XLC": "Communication Services",
    # Industrials
    "CAT": "Industrials", "HON": "Industrials", "GE": "Industrials",
    "UNP": "Industrials", "UPS": "Industrials", "RTX": "Industrials",
    "BA": "Industrials", "LMT": "Industrials", "NOC": "Industrials",
    "GD": "Industrials", "DE": "Industrials", "MMM": "Industrials",
    "EMR": "Industrials", "ETN": "Industrials", "PH": "Industrials",
    "FDX": "Industrials", "XLI": "Industrials",
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "GIS": "Consumer Staples", "KHC": "Consumer Staples", "SYY": "Consumer Staples",
    "XLP": "Consumer Staples",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "EOG": "Energy",
    "SLB": "Energy", "MPC": "Energy", "VLO": "Energy", "PSX": "Energy",
    "OXY": "Energy", "PXD": "Energy", "HAL": "Energy", "DVN": "Energy",
    "XLE": "Energy",
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "AEP": "Utilities", "D": "Utilities", "EXC": "Utilities",
    "PCG": "Utilities", "XEL": "Utilities", "ED": "Utilities",
    "XLU": "Utilities",
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials", "DOW": "Materials",
    "DD": "Materials", "ALB": "Materials", "XLB": "Materials",
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    "CCI": "Real Estate", "SPG": "Real Estate", "O": "Real Estate",
    "WELL": "Real Estate", "DLR": "Real Estate", "XLRE": "Real Estate",
    # Broad market ETFs
    "SPY": "Broad Market", "QQQ": "Broad Market", "IWM": "Broad Market",
    "DIA": "Broad Market", "VOO": "Broad Market", "VTI": "Broad Market",
}


@dataclass(frozen=True)
class SectorWeight:
    sector: str
    weight_pct: float
    n_positions: int
    benchmark_weight_pct: float
    active_weight_pct: float    # portfolio - benchmark
    symbols: list[str]


@dataclass(frozen=True)
class SectorExposureReport:
    sectors: list[SectorWeight]   # sorted by weight desc
    portfolio_hhi: float          # sector-level HHI
    dominant_sector: str
    unknown_pct: float            # % in unknown/unmapped symbols
    risk_flags: list[str]
    n_positions: int
    n_sectors: int
    verdict: str


def analyze(broker, *, min_position_value: float = 0.0) -> SectorExposureReport | None:
    """Compute sector exposure from current portfolio positions.

    broker: broker interface with get_positions() returning objects
            with .symbol and .market_value attributes

    Returns None if no positions.
    """
    try:
        positions = broker.get_positions()
    except Exception:
        return None

    if not positions:
        return None

    # Gather market values
    pos_list: list[tuple[str, float]] = []
    for pos in positions:
        try:
            mv = float(pos.market_value)
            if mv > min_position_value:
                pos_list.append((pos.symbol, mv))
        except Exception:
            pass

    if not pos_list:
        return None

    total = sum(mv for _, mv in pos_list)
    if total <= 0:
        return None

    # Map to sectors
    sector_data: dict[str, tuple[float, list[str]]] = {}  # sector → (total_value, symbols)
    for sym, mv in pos_list:
        sector = SYMBOL_SECTOR.get(sym.upper(), "Unknown")
        if sector not in sector_data:
            sector_data[sector] = (0.0, [])
        val, syms = sector_data[sector]
        sector_data[sector] = (val + mv, syms + [sym])

    # Build sector weights
    sector_weights: list[SectorWeight] = []
    for sector, (val, syms) in sector_data.items():
        weight = val / total * 100
        benchmark = BENCHMARK_WEIGHTS.get(sector, 0.0)
        sector_weights.append(SectorWeight(
            sector=sector,
            weight_pct=round(weight, 2),
            n_positions=len(syms),
            benchmark_weight_pct=benchmark,
            active_weight_pct=round(weight - benchmark, 2),
            symbols=sorted(syms),
        ))

    sector_weights.sort(key=lambda s: -s.weight_pct)

    # HHI at sector level
    weights_frac = [s.weight_pct / 100 for s in sector_weights]
    hhi = sum(w ** 2 for w in weights_frac)

    dominant = sector_weights[0].sector if sector_weights else "Unknown"
    unknown_pct = next((s.weight_pct for s in sector_weights if s.sector == "Unknown"), 0.0)

    # Risk flags
    flags: list[str] = []
    for s in sector_weights:
        if s.weight_pct > 40 and s.sector not in ("Broad Market", "Unknown"):
            flags.append(f"{s.sector} is {s.weight_pct:.1f}%% of portfolio (overconcentrated)")
    if hhi > 0.30:
        flags.append(f"Sector HHI {hhi:.3f} — very concentrated")
    defensive = sum(
        s.weight_pct for s in sector_weights
        if s.sector in ("Health Care", "Consumer Staples", "Utilities")
    )
    if defensive < 5 and unknown_pct < 50:
        flags.append("Less than 5%% in defensive sectors (Health Care / Staples / Utilities)")
    if unknown_pct > 30:
        flags.append(f"{unknown_pct:.1f}%% of portfolio in unmapped symbols")

    # Verdict
    if hhi > 0.4:
        verdict = "Highly concentrated in few sectors — significant sector risk"
    elif hhi > 0.25:
        verdict = "Moderate sector concentration"
    elif flags:
        verdict = "Sector exposure warnings present"
    else:
        verdict = "Reasonably diversified across sectors"

    n_sectors = sum(1 for s in sector_weights if s.sector not in ("Unknown", "Broad Market"))

    return SectorExposureReport(
        sectors=sector_weights,
        portfolio_hhi=round(hhi, 4),
        dominant_sector=dominant,
        unknown_pct=round(unknown_pct, 2),
        risk_flags=flags,
        n_positions=len(pos_list),
        n_sectors=n_sectors,
        verdict=verdict,
    )
