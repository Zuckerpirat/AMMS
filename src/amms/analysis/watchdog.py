"""Daily risk watchdog.

Generates a comprehensive risk summary covering:
  - Circuit breaker state
  - Market regime
  - Open P&L and drawdown
  - Per-symbol warning flags (Bollinger extremes, RSI extremes, high ADX)
  - Sector rotation summary
  - Recommended actions

Used by the /watchdog Telegram command and can be scheduled daily.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WatchdogWarning:
    symbol: str
    level: str    # "info" | "warning" | "critical"
    message: str


@dataclass(frozen=True)
class WatchdogReport:
    circuit_open: bool
    circuit_reason: str
    regime: str                 # "bull" | "neutral" | "bear" | "unknown"
    regime_risk_multiplier: float
    total_open_pnl: float
    total_open_pnl_pct: float
    position_count: int
    warnings: list[WatchdogWarning]
    top_rotating_in: list[str]   # sector ETFs rotating in
    summary: str


def generate(
    broker,
    *,
    conn=None,
    data=None,
) -> WatchdogReport:
    """Generate a full watchdog report.

    broker: must support get_positions(), get_account()
    conn: SQLite connection for circuit breaker and regime state
    data: market data client for technical signals
    """
    # --- Circuit breaker ---
    circuit_open = False
    circuit_reason = "ok"
    if conn is not None:
        try:
            from amms.risk.circuit_breaker import is_open, load_state
            state = load_state(conn)
            circuit_open = state.tripped
            if state.tripped:
                circuit_reason = state.reason or "tripped"
        except Exception as e:
            logger.debug("circuit breaker check failed: %s", e)

    # --- Market regime ---
    regime = "unknown"
    risk_multiplier = 0.75
    if conn is not None and data is not None:
        try:
            from amms.analysis.regime import detect_regime
            r = detect_regime(data)
            regime = r.label
            risk_multiplier = r.risk_multiplier
        except Exception as e:
            logger.debug("regime check failed: %s", e)

    # --- Positions ---
    total_open_pnl = 0.0
    total_mv = 0.0
    position_count = 0
    positions = []
    try:
        positions = broker.get_positions()
        for p in positions:
            total_open_pnl += float(p.unrealized_pl)
            total_mv += float(p.market_value)
            position_count += 1
    except Exception as e:
        logger.debug("position fetch failed: %s", e)

    total_open_pnl_pct = (total_open_pnl / total_mv * 100) if total_mv > 0 else 0.0

    # --- Per-symbol technical warnings ---
    warnings: list[WatchdogWarning] = []
    if data is not None:
        for p in positions[:10]:
            sym = p.symbol
            try:
                bars = data.get_bars(sym, limit=50)
            except Exception:
                continue
            _check_symbol_warnings(sym, bars, warnings)

    # --- Sector rotation ---
    top_rotating_in: list[str] = []
    if data is not None:
        try:
            from amms.analysis.sector_rotation import detect_rotation
            sectors = detect_rotation(data, n=20)
            top_rotating_in = [s.etf for s in sectors[:3] if s.trend == "in"]
        except Exception:
            pass

    # --- Circuit breaker warning ---
    if circuit_open:
        warnings.insert(0, WatchdogWarning(
            symbol="SYSTEM",
            level="critical",
            message=f"Circuit breaker OPEN: {circuit_reason}",
        ))

    if regime == "bear":
        warnings.insert(0, WatchdogWarning(
            symbol="MACRO",
            level="warning",
            message="Bear market regime: risk multiplier reduced to 0.5×",
        ))

    # --- Summary text ---
    warn_count = len([w for w in warnings if w.level in ("warning", "critical")])
    summary_parts = [
        f"Regime: {regime} ({risk_multiplier:.2f}× risk)",
        f"Positions: {position_count}",
        f"Open P&L: {total_open_pnl:+.2f} ({total_open_pnl_pct:+.1f}%)",
        f"Warnings: {warn_count}",
        f"Circuit: {'OPEN' if circuit_open else 'closed'}",
    ]
    summary = " | ".join(summary_parts)

    return WatchdogReport(
        circuit_open=circuit_open,
        circuit_reason=circuit_reason,
        regime=regime,
        regime_risk_multiplier=risk_multiplier,
        total_open_pnl=round(total_open_pnl, 2),
        total_open_pnl_pct=round(total_open_pnl_pct, 2),
        position_count=position_count,
        warnings=warnings,
        top_rotating_in=top_rotating_in,
        summary=summary,
    )


def _check_symbol_warnings(sym: str, bars: list[Bar], warnings: list[WatchdogWarning]) -> None:
    from amms.data.bars import Bar  # noqa: F401 — confirm type

    # Bollinger extreme
    try:
        from amms.features.bollinger import bollinger
        bb = bollinger(bars, 20)
        if bb is not None:
            if bb.pct_b > 1.1:
                warnings.append(WatchdogWarning(sym, "warning", f"Price above upper Bollinger band (%B {bb.pct_b:.2f})"))
            elif bb.pct_b < -0.1:
                warnings.append(WatchdogWarning(sym, "warning", f"Price below lower Bollinger band (%B {bb.pct_b:.2f})"))
    except Exception:
        pass

    # RSI extreme
    try:
        from amms.features.momentum import rsi
        rsi_val = rsi(bars, 14)
        if rsi_val is not None:
            if rsi_val > 80:
                warnings.append(WatchdogWarning(sym, "warning", f"RSI extreme overbought ({rsi_val:.1f})"))
            elif rsi_val < 20:
                warnings.append(WatchdogWarning(sym, "warning", f"RSI extreme oversold ({rsi_val:.1f})"))
    except Exception:
        pass

    # ADX very strong trend (caution: may be near exhaustion)
    try:
        from amms.features.adx import adx
        adx_result = adx(bars, 14)
        if adx_result is not None and adx_result.trend_strength in ("very_strong", "extreme"):
            warnings.append(WatchdogWarning(
                sym, "info",
                f"ADX {adx_result.adx:.1f} ({adx_result.trend_strength}): strong {adx_result.direction} trend"
            ))
    except Exception:
        pass
