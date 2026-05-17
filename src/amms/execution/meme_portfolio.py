"""Meme Mode Sandbox Portfolio.

CLAUDE.md requirement: Mode 3 (Meme/Experimental) MUST remain sandboxed
with separate capital allocation, separate risk limits, separate
performance tracking, and separate trade journal.

This module wraps PaperTrader with:
  - A dedicated state file (separate from main paper portfolio)
  - Hard-coded conservative risk limits regardless of AutoTraderConfig
  - A "MEME:" prefix on all trade reasons for clear audit trail
  - A fixed capital cap — meme allocation is never more than a configurable
    fraction of total (main + meme) portfolio value

The sandbox is intentionally restrictive:
  max_position_pct  : 3%   (not 10% like main)
  max_positions     : 5    (not 10)
  max_allocation_pct: 10%  of combined portfolio (meme total cannot exceed this)
  cooldown_minutes  : 120  (not 60 — meme trades are even higher risk)

Usage:
    from amms.execution.meme_portfolio import MemePortfolio
    mp = MemePortfolio.load(main_trader=paper_trader)
    mp.buy("GME", qty=5, price=20.0, reason="WSB spike")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_MEME_STATE_FILE = Path(
    os.environ.get("AMMS_MEME_STATE", "meme_portfolio.json")
)


@dataclass(frozen=True)
class MemeConfig:
    max_position_pct: float = 0.03     # 3% of meme portfolio per symbol
    max_positions: int = 5
    max_allocation_pct: float = 0.10   # meme total <= 10% of combined portfolio
    cooldown_minutes: int = 120
    starting_cash: float = 5_000.0     # initial meme sandbox allocation


class MemePortfolio:
    """Sandboxed meme-mode paper trader with strict limits.

    Always uses a separate state file from the main paper portfolio.
    All trades are prefixed with 'MEME:' in the reason field for
    audit-trail clarity.
    """

    def __init__(self, config: MemeConfig | None = None, state_path: Path = _MEME_STATE_FILE,
                 main_trader=None):
        from amms.execution.paper_trader import PaperTrader
        self.config = config or MemeConfig()
        self.state_path = state_path
        self.main_trader = main_trader  # reference to check combined portfolio value
        if state_path.exists():
            self._trader = PaperTrader.load(path=state_path)
        else:
            self._trader = PaperTrader(starting_cash=self.config.starting_cash)
        self.name = "meme-sandbox"

    @classmethod
    def load(cls, main_trader=None, config: MemeConfig | None = None,
             state_path: Path = _MEME_STATE_FILE) -> "MemePortfolio":
        return cls(config=config, state_path=state_path, main_trader=main_trader)

    # ── Guard: check combined exposure ───────────────────────────────────────

    def _combined_value(self) -> float:
        """Main portfolio value + meme portfolio value."""
        snap_meme = self._trader.snapshot()
        if self.main_trader is None:
            return snap_meme.portfolio_value
        try:
            snap_main = self.main_trader.snapshot()
            return snap_main.portfolio_value + snap_meme.portfolio_value
        except Exception:
            return snap_meme.portfolio_value

    def _allocation_check(self) -> str | None:
        """Return a reason string if meme allocation would be exceeded, else None."""
        snap = self._trader.snapshot()
        combined = self._combined_value()
        if combined <= 0:
            return None
        meme_pct = snap.portfolio_value / combined
        if meme_pct > self.config.max_allocation_pct:
            return (
                f"meme sandbox at {meme_pct:.1%} of combined portfolio "
                f"(limit {self.config.max_allocation_pct:.0%})"
            )
        return None

    # ── Trading interface ─────────────────────────────────────────────────────

    def buy(self, symbol: str, qty: float, price: float, reason: str = "") -> object | None:
        symbol = symbol.upper()

        # Cap check
        cap_reason = self._allocation_check()
        if cap_reason:
            logger.warning("MEME buy blocked: %s", cap_reason)
            return None

        # Position count
        snap = self._trader.snapshot()
        if len(snap.positions) >= self.config.max_positions:
            logger.warning("MEME max positions (%d) reached", self.config.max_positions)
            return None

        # Size cap
        max_dollar = snap.portfolio_value * self.config.max_position_pct
        if qty * price > max_dollar:
            qty = round(max_dollar / price, 4) if price > 0 else 0.0

        if qty <= 0:
            return None

        meme_reason = f"MEME: {reason}".strip()
        trade = self._trader.buy(symbol, qty, price, reason=meme_reason)
        if trade:
            self._trader.save(self.state_path)
            logger.info("MEME-SANDBOX BUY %s × %.4f @ %.2f", symbol, qty, price)
        return trade

    def sell(self, symbol: str, qty: float, price: float, reason: str = "") -> object | None:
        meme_reason = f"MEME: {reason}".strip()
        trade = self._trader.sell(symbol, qty, price, reason=meme_reason)
        if trade:
            self._trader.save(self.state_path)
            logger.info("MEME-SANDBOX SELL %s × %.4f @ %.2f", symbol, qty, price)
        return trade

    def close_position(self, symbol: str, price: float, reason: str = "") -> object | None:
        meme_reason = f"MEME: {reason}".strip()
        trade = self._trader.close_position(symbol, price, reason=meme_reason)
        if trade:
            self._trader.save(self.state_path)
        return trade

    def position(self, symbol: str):
        return self._trader.position(symbol)

    def snapshot(self, prices=None):
        return self._trader.snapshot(prices)

    def recent_trades(self, n: int = 10):
        return self._trader.recent_trades(n)

    def save(self) -> None:
        self._trader.save(self.state_path)

    # ── Status ────────────────────────────────────────────────────────────────

    def status_summary(self) -> str:
        snap = self._trader.snapshot()
        combined = self._combined_value()
        alloc_pct = snap.portfolio_value / combined * 100 if combined > 0 else 0.0
        lines = [
            "── Meme Sandbox Portfolio ──",
            f"  Cash:      ${snap.cash:>10,.2f}",
            f"  Positions: {len(snap.positions)}  (max {self.config.max_positions})",
            f"  Value:     ${snap.portfolio_value:>10,.2f}  ({alloc_pct:.1f}% of combined)",
            f"  Max alloc: {self.config.max_allocation_pct:.0%} of combined portfolio",
            f"  Per-pos:   {self.config.max_position_pct:.0%} of meme portfolio",
            f"  Return:    {snap.total_return_pct:+.2f}%",
        ]
        if snap.positions:
            lines += ["", "  Open positions:"]
            for sym, pd in sorted(snap.positions.items()):
                lines.append(
                    f"    {sym:<8} {pd['qty']:.4f}  MV=${pd['market_value']:,.2f}  "
                    f"P&L {pd['unrealized_pnl']:+,.2f}"
                )
        return "\n".join(lines)
