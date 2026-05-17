"""Daily portfolio and market summary report.

Generates a concise nightly report covering:
  - Main paper portfolio performance (P&L, return, positions)
  - Meme sandbox summary
  - Top DE signals from the watchlist
  - Macro regime (if data available)
  - Risk guard status

Intended for: scheduled nightly delivery via Telegram, or on-demand
via /dailyreport command.

Usage:
    from amms.reporting.daily_report import generate_daily_report
    text = generate_daily_report(trader, meme_portfolio, data_client, symbols)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_MAX_SCAN_SYMBOLS = 10
_DE_BARS = 200


def generate_daily_report(
    trader,
    meme_portfolio=None,
    data=None,
    symbols: list[str] | None = None,
    *,
    mode: str = "swing",
) -> str:
    """Build the full daily report string.

    Args:
        trader: PaperTrader instance (main portfolio)
        meme_portfolio: MemePortfolio instance (optional)
        data: data client with get_bars() (optional)
        symbols: watchlist to scan for DE signals (optional)
        mode: DE trading mode for signal scan

    Returns:
        Formatted multi-section report string.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections: list[str] = [f"📊 Daily Report — {now}"]

    # ── Main portfolio ──────────────────────────────────────────────────
    try:
        prices: dict[str, float] = {}
        if data is not None and hasattr(trader, "positions"):
            for sym in list(trader.positions.keys()):
                try:
                    bars = data.get_bars(sym, limit=2)
                    if bars:
                        prices[sym] = float(bars[-1].close)
                except Exception:
                    pass

        snap = trader.snapshot(prices) if prices else trader.snapshot()
        ret_arrow = "▲" if snap.total_return_pct >= 0 else "▼"
        section = [
            "── Main Portfolio ──",
            f"  Value:    ${snap.portfolio_value:>12,.2f}",
            f"  Return:   {ret_arrow} {snap.total_return_pct:+.2f}%",
            f"  Cash:     ${snap.cash:>12,.2f}",
            f"  P&L:      ${snap.total_realized_pnl:>+12,.2f} realized  "
            f"${snap.total_unrealized_pnl:>+,.2f} unrealized",
            f"  Trades:   {snap.trade_count}",
        ]
        if snap.positions:
            section.append(f"  Positions ({len(snap.positions)}):")
            for sym, pd in sorted(snap.positions.items()):
                pnl_arrow = "▲" if pd["unrealized_pnl"] >= 0 else "▼"
                section.append(
                    f"    {sym:<8} qty={pd['qty']:.2f}  "
                    f"MV=${pd['market_value']:,.0f}  "
                    f"P&L {pnl_arrow}{pd['unrealized_pnl']:+,.0f} ({pd['pnl_pct']:+.1f}%)"
                )
        else:
            section.append("  No open positions.")
        sections.append("\n".join(section))
    except Exception as exc:
        logger.warning("daily_report: main portfolio section failed: %s", exc)
        sections.append("── Main Portfolio ──\n  (unavailable)")

    # ── Meme sandbox ───────────────────────────────────────────────────
    if meme_portfolio is not None:
        try:
            snap_m = meme_portfolio.snapshot()
            combined = meme_portfolio._combined_value()
            alloc_pct = snap_m.portfolio_value / combined * 100 if combined > 0 else 0.0
            meme_arrow = "▲" if snap_m.total_return_pct >= 0 else "▼"
            section = [
                "── Meme Sandbox ──",
                f"  Value:    ${snap_m.portfolio_value:>12,.2f}  "
                f"({alloc_pct:.1f}% of combined)",
                f"  Return:   {meme_arrow} {snap_m.total_return_pct:+.2f}%",
                f"  Positions: {len(snap_m.positions)} / {meme_portfolio.config.max_positions}",
            ]
            sections.append("\n".join(section))
        except Exception as exc:
            logger.warning("daily_report: meme section failed: %s", exc)

    # ── DE signal scan ─────────────────────────────────────────────────
    if data is not None and symbols:
        try:
            from amms.engine.decision import analyze as de_analyze
            scan_syms = [s.upper() for s in symbols[:_MAX_SCAN_SYMBOLS]]
            signal_rows: list[tuple[float, str, str]] = []  # (score, action, sym)

            for sym in scan_syms:
                try:
                    bars = data.get_bars(sym, limit=_DE_BARS)
                    if not bars or len(bars) < 120:
                        continue
                    report = de_analyze(bars, symbol=sym, min_confidence=0.50, mode=mode)
                    if report is None:
                        continue
                    signal_rows.append((report.composite_score, report.action, sym))
                except Exception:
                    pass

            if signal_rows:
                signal_rows.sort(key=lambda x: x[0], reverse=True)
                section = [f"── DE Signal Scan (mode={mode}, {len(scan_syms)} symbols) ──"]
                action_icons = {
                    "strong_buy":  "🟢 S.BUY",
                    "buy":         "🟩 BUY  ",
                    "hold":        "⬜ HOLD ",
                    "sell":        "🟥 SELL ",
                    "strong_sell": "🔴 S.SELL",
                }
                for score, action, sym in signal_rows:
                    icon = action_icons.get(action, action)
                    section.append(f"  {sym:<8} {icon}  score {score:+.0f}")
                sections.append("\n".join(section))
        except Exception as exc:
            logger.warning("daily_report: DE scan section failed: %s", exc)

    # ── Macro regime ───────────────────────────────────────────────────
    if data is not None:
        try:
            from amms.data.macro import compute_regime
            regime = compute_regime(data)
            sections.append(
                f"── Macro Regime ──\n"
                f"  Level: {regime.level.upper()}\n"
                f"  {regime.reason}"
            )
        except Exception:
            pass

    return "\n\n".join(sections)
