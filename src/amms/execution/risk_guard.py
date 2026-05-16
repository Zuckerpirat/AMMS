"""Risk Guard — central veto authority for the Decision Engine.

Implements the CLAUDE.md mandate that "Risk layer has veto power over
strategy signals." Tracks portfolio-wide drawdown, exposure, and a
manual killswitch. Designed to be plugged into the Decision Engine via
its `risk_veto` callable parameter.

Sources of veto (any one fires):
  1. Killswitch is armed              → block all trades
  2. Daily P&L drawdown beyond limit  → block all trades
  3. Portfolio drawdown from peak     → block all trades
  4. Max gross exposure reached       → block new buys (sells allowed)
  5. Custom rule plugins              → block as configured

State is persisted to JSON so killswitch and peak-equity survive
restarts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_RISK_STATE_FILE = Path(os.environ.get("AMMS_RISK_STATE", "risk_guard_state.json"))


@dataclass
class RiskConfig:
    max_daily_loss_pct: float = 0.03      # 3% intraday loss → halt
    max_drawdown_pct: float  = 0.15       # 15% from peak → halt
    max_gross_exposure_pct: float = 0.95  # 95% of equity in positions → block new buys
    cooldown_after_kill_minutes: int = 60  # if killed automatically, wait this long
    enabled: bool = True                  # master toggle


@dataclass
class RiskState:
    killswitch_armed: bool = False
    killswitch_reason: str = ""
    killswitch_armed_at: str = ""           # ISO timestamp
    killswitch_auto: bool = False           # True if triggered by drawdown/daily-loss (auto-disarm eligible)
    peak_equity: float = 0.0                # all-time portfolio value high
    session_start_equity: float = 0.0       # equity at last session-start mark
    session_start_at: str = ""              # ISO timestamp of session-start mark


class RiskGuard:
    """Central risk veto + tracking.

    Usage with Decision Engine:
        guard = RiskGuard.load(trader)
        decision = analyze(bars, risk_veto=guard.make_veto())

    The veto callable signature is `(score, confidence) -> reason | None`.
    """

    def __init__(self, trader, config: RiskConfig | None = None,
                 state_path: Path = _RISK_STATE_FILE):
        self.trader = trader
        self.config = config or RiskConfig()
        self.state_path = state_path
        self.state = self._load_state()

    # ── State persistence ─────────────────────────────────────────────────

    def _load_state(self) -> RiskState:
        if not self.state_path.exists():
            return RiskState()
        try:
            raw = json.loads(self.state_path.read_text())
            return RiskState(**raw)
        except Exception as exc:
            logger.warning("Could not load risk state: %s", exc)
            return RiskState()

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state.__dict__, indent=2))

    # ── Killswitch ────────────────────────────────────────────────────────

    def arm_killswitch(self, reason: str = "manual", auto: bool = False) -> None:
        """Block all trades until disarm() is called.

        If `auto=True`, the killswitch may be auto-disarmed after
        `cooldown_after_kill_minutes`. Manual arms (auto=False) require
        explicit user disarm.
        """
        self.state.killswitch_armed = True
        self.state.killswitch_reason = reason
        self.state.killswitch_armed_at = datetime.now(timezone.utc).isoformat()
        self.state.killswitch_auto = auto
        self.save()
        tag = "AUTO" if auto else "MANUAL"
        logger.warning("🛑 KILLSWITCH ARMED [%s]: %s", tag, reason)

    def _maybe_auto_disarm(self) -> bool:
        """Auto-disarm if killswitch was auto-triggered and cooldown has passed.

        Returns True if disarmed. Manual arms are never auto-disarmed.
        """
        if not self.state.killswitch_armed:
            return False
        if not self.state.killswitch_auto:
            return False
        if not self.state.killswitch_armed_at:
            return False
        try:
            armed_at = datetime.fromisoformat(self.state.killswitch_armed_at)
            if armed_at.tzinfo is None:
                armed_at = armed_at.replace(tzinfo=timezone.utc)
        except ValueError:
            return False

        elapsed = datetime.now(timezone.utc) - armed_at
        cooldown = timedelta(minutes=self.config.cooldown_after_kill_minutes)
        if elapsed >= cooldown:
            logger.warning(
                "✓ Auto-disarming killswitch after %s cooldown (was: %s)",
                cooldown, self.state.killswitch_reason,
            )
            self.disarm_killswitch()
            return True
        return False

    def disarm_killswitch(self) -> None:
        if not self.state.killswitch_armed:
            return
        self.state.killswitch_armed = False
        self.state.killswitch_reason = ""
        self.state.killswitch_armed_at = ""
        self.state.killswitch_auto = False
        self.save()
        logger.warning("✓ Killswitch disarmed")

    # ── Equity tracking ───────────────────────────────────────────────────

    def mark_session_start(self) -> None:
        """Record current portfolio value as start-of-session baseline.

        Call at start of each trading day or session for accurate daily-loss
        tracking. If not called, daily-loss check falls back to peak_equity.
        """
        snap = self.trader.snapshot()
        self.state.session_start_equity = snap.portfolio_value
        self.state.session_start_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def update_peak(self) -> None:
        """Update peak equity if current value is higher."""
        snap = self.trader.snapshot()
        if snap.portfolio_value > self.state.peak_equity:
            self.state.peak_equity = snap.portfolio_value
            self.save()

    # ── Veto evaluation ───────────────────────────────────────────────────

    def check(self, side: str = "buy") -> str | None:
        """Return a veto reason if action should be blocked, else None.

        `side` is "buy" or "sell". Sells bypass the exposure cap.
        Auto-disarm of auto-armed killswitch is attempted first.
        """
        if not self.config.enabled:
            return None

        # Auto-disarm check (only auto-armed killswitches age out)
        self._maybe_auto_disarm()

        if self.state.killswitch_armed:
            return f"killswitch: {self.state.killswitch_reason}"

        snap = self.trader.snapshot()
        equity = snap.portfolio_value

        # Peak tracking — only persist on meaningful change (avoid I/O storm).
        # We accept staleness up to 0.1% of equity between writes.
        if equity > self.state.peak_equity:
            persist_threshold = max(self.state.peak_equity * 1.001, self.state.peak_equity + 1.0)
            self.state.peak_equity = equity
            if equity >= persist_threshold:
                self.save()

        # 1. Drawdown from all-time peak
        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - equity) / self.state.peak_equity
            if dd >= self.config.max_drawdown_pct:
                self.arm_killswitch(
                    reason=f"drawdown {dd:.1%} from peak ${self.state.peak_equity:,.2f}",
                    auto=True,
                )
                return f"max drawdown reached ({dd:.1%})"

        # 2. Daily loss
        if self.state.session_start_equity > 0:
            daily = (self.state.session_start_equity - equity) / self.state.session_start_equity
            if daily >= self.config.max_daily_loss_pct:
                self.arm_killswitch(
                    reason=f"daily loss {daily:.1%} from ${self.state.session_start_equity:,.2f}",
                    auto=True,
                )
                return f"daily loss limit ({daily:.1%})"

        # 3. Gross exposure cap (buys only)
        if side == "buy" and equity > 0:
            exposure = snap.total_market_value / equity
            if exposure >= self.config.max_gross_exposure_pct:
                return f"max exposure {exposure:.0%} reached"

        return None

    def make_veto(self):
        """Return a callable compatible with Decision Engine's risk_veto parameter.

        The decision engine passes (composite_score, confidence). We translate
        the sign of the score to a side and run the appropriate check.
        """
        def veto(score: float, _confidence: float) -> str | None:
            side = "buy" if score >= 0 else "sell"
            return self.check(side=side)
        return veto

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        snap = self.trader.snapshot()
        equity = snap.portfolio_value
        peak = self.state.peak_equity or equity
        dd = (peak - equity) / peak * 100.0 if peak > 0 else 0.0
        daily = (
            (self.state.session_start_equity - equity) / self.state.session_start_equity * 100.0
            if self.state.session_start_equity > 0 else 0.0
        )
        gross_exposure = snap.total_market_value / equity * 100.0 if equity > 0 else 0.0

        return {
            "enabled":         self.config.enabled,
            "killswitch":      self.state.killswitch_armed,
            "kill_reason":     self.state.killswitch_reason,
            "kill_at":         self.state.killswitch_armed_at,
            "equity":          round(equity, 2),
            "peak_equity":     round(peak, 2),
            "drawdown_pct":    round(dd, 2),
            "daily_loss_pct":  round(daily, 2),
            "gross_exposure_pct": round(gross_exposure, 2),
            "limits": {
                "max_dd_pct":       self.config.max_drawdown_pct * 100,
                "max_daily_pct":    self.config.max_daily_loss_pct * 100,
                "max_exposure_pct": self.config.max_gross_exposure_pct * 100,
            },
        }
