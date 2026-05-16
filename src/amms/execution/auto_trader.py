"""Auto-Trader: runs the Decision Engine on a watchlist and executes paper trades.

Wiring:
  Watchlist  →  Decision Engine  →  Risk Filter  →  Paper Trader

Safety guards (all configurable):
  - max_position_pct   : max % of portfolio per single symbol
  - max_positions      : max number of concurrent positions
  - cooldown_minutes   : minimum gap between trades for the same symbol
  - min_confidence     : skip if Decision Engine confidence below this
  - min_score          : skip if |score| below this
  - allow_strong_only  : only act on strong_buy / strong_sell signals

State is persisted via the PaperTrader's JSON file plus a separate
cooldown ledger so restarts don't immediately re-trade.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_COOLDOWN_FILE = Path(os.environ.get("AMMS_AUTOTRADER_STATE", "auto_trader_state.json"))


@dataclass
class AutoTradeDecision:
    symbol: str
    action: str             # "bought" / "sold" / "closed" / "skipped"
    score: float
    confidence: float
    qty: float
    price: float
    reason: str             # explanation


@dataclass
class AutoTraderConfig:
    max_position_pct: float = 0.10        # 10% of portfolio per symbol
    max_positions: int = 10
    cooldown_minutes: int = 60
    min_confidence: float = 0.60
    min_score: float = 35.0
    allow_strong_only: bool = False       # if True, only act on strong_* signals
    # When sell triggered but no position is held, skip (no shorting in paper)
    enable_close_on_sell: bool = True     # close existing long if sell signal


class AutoTrader:
    """Runs decisions for a list of symbols, executes paper trades."""

    def __init__(self, paper_trader, data_client, config: AutoTraderConfig | None = None,
                 state_path: Path = _COOLDOWN_FILE, risk_guard=None):
        self.trader = paper_trader
        self.data = data_client
        self.config = config or AutoTraderConfig()
        self.state_path = state_path
        self.risk_guard = risk_guard           # optional RiskGuard instance
        self._cooldowns: dict[str, str] = self._load_state()  # symbol → ISO timestamp
        # Prevent concurrent processing of the same symbol (manual + scheduler)
        self._process_lock = threading.Lock()

    # ── State persistence ──────────────────────────────────────────────────

    def _load_state(self) -> dict[str, str]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text()).get("cooldowns", {})
        except Exception as exc:
            logger.warning("Could not load auto-trader state: %s", exc)
            return {}

    def _save_state(self) -> None:
        state = {"cooldowns": self._cooldowns}
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def _in_cooldown(self, symbol: str) -> bool:
        ts = self._cooldowns.get(symbol)
        if not ts:
            return False
        try:
            last = datetime.fromisoformat(ts)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
        except Exception:
            return False
        return datetime.now(timezone.utc) - last < timedelta(minutes=self.config.cooldown_minutes)

    def _record_cooldown(self, symbol: str) -> None:
        self._cooldowns[symbol] = datetime.now(timezone.utc).isoformat()
        self._save_state()

    # ── Core decision flow ─────────────────────────────────────────────────

    def _fetch_bars(self, symbol: str, limit: int = 200):
        try:
            return self.data.get_bars(symbol, limit=limit)
        except Exception as exc:
            logger.warning("Could not fetch bars for %s: %s", symbol, exc)
            return None

    def _calc_qty(self, price: float, portfolio_value: float) -> float:
        """How many shares to buy given position size limit."""
        if price <= 0:
            return 0.0
        max_dollar = portfolio_value * self.config.max_position_pct
        return round(max_dollar / price, 4)

    def process_symbol(self, symbol: str) -> AutoTradeDecision:
        """Run the full decision + execution pipeline for one symbol.

        Thread-safe: only one process_symbol() call executes at a time
        per AutoTrader instance, preventing race conditions when both
        the scheduler and a manual /autorun trigger the same symbol.
        """
        symbol = symbol.upper()

        with self._process_lock:
            return self._process_symbol_locked(symbol)

    def _process_symbol_locked(self, symbol: str) -> AutoTradeDecision:
        # 0. Risk Guard hard veto BEFORE any work — killswitch must block
        #    everything immediately, including reading bars from data API.
        if self.risk_guard is not None and self.risk_guard.state.killswitch_armed:
            return AutoTradeDecision(
                symbol, "skipped", 0.0, 0.0, 0.0, 0.0,
                reason=f"killswitch armed: {self.risk_guard.state.killswitch_reason}",
            )

        # 1. Cooldown — DO NOT early-return; cooldown only blocks BUYs.
        #    Sells (risk reduction) must always be possible.
        cooldown_active = self._in_cooldown(symbol)

        # 2. Fetch bars
        bars = self._fetch_bars(symbol)
        if not bars or len(bars) < 120:
            return AutoTradeDecision(symbol, "skipped", 0.0, 0.0, 0.0, 0.0,
                                     reason="insufficient bar data")

        # 3. Run Decision Engine with risk veto wired in
        from amms.engine.decision import analyze as decide_analyze
        risk_veto = self.risk_guard.make_veto() if self.risk_guard is not None else None
        decision = decide_analyze(
            bars,
            symbol=symbol,
            min_confidence=self.config.min_confidence,
            risk_veto=risk_veto,
        )
        if decision is None:
            return AutoTradeDecision(symbol, "skipped", 0.0, 0.0, 0.0, 0.0,
                                     reason="decision engine returned None")

        price = float(bars[-1].close)
        cur_pos = self.trader.position(symbol)
        snap = self.trader.snapshot()

        # 4. Filter on signal strength
        if abs(decision.composite_score) < self.config.min_score:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason=f"score {decision.composite_score:.0f} below min {self.config.min_score}")

        if decision.risk_blocked:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason=f"risk gate: {decision.risk_reason}")

        if self.config.allow_strong_only and decision.action not in {"strong_buy", "strong_sell"}:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason=f"action {decision.action} not strong_*")

        # 5. Act — cooldown gates BUYs only; sells always proceed.
        if decision.action in {"buy", "strong_buy"}:
            if cooldown_active:
                return AutoTradeDecision(
                    symbol, "skipped",
                    decision.composite_score, decision.confidence,
                    0.0, price,
                    reason="cooldown active (buy blocked, sells still allowed)",
                )
            return self._do_buy(symbol, decision, price, cur_pos, snap)

        if decision.action in {"sell", "strong_sell"}:
            return self._do_sell(symbol, decision, price, cur_pos)

        return AutoTradeDecision(symbol, "skipped",
                                 decision.composite_score, decision.confidence,
                                 0.0, price,
                                 reason=f"hold action")

    def _do_buy(self, symbol, decision, price, cur_pos, snap) -> AutoTradeDecision:
        if cur_pos is not None:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     cur_pos.qty, price,
                                     reason="already holding position")

        if len(snap.positions) >= self.config.max_positions:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason=f"max positions ({self.config.max_positions}) reached")

        qty = self._calc_qty(price, snap.portfolio_value)
        if qty <= 0:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason="computed qty <= 0")

        reason = (f"Engine {decision.action} | score {decision.composite_score:+.0f} "
                  f"conf {decision.confidence:.0%}")
        trade = self.trader.buy(symbol, qty, price, reason=reason)
        if trade is None:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     qty, price,
                                     reason="insufficient cash")
        self.trader.save()
        self._record_cooldown(symbol)
        return AutoTradeDecision(symbol, "bought",
                                 decision.composite_score, decision.confidence,
                                 qty, price, reason=reason)

    def _do_sell(self, symbol, decision, price, cur_pos) -> AutoTradeDecision:
        if cur_pos is None:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason="no position to sell (no shorting in paper)")

        if not self.config.enable_close_on_sell:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason="close_on_sell disabled")

        reason = (f"Engine {decision.action} | score {decision.composite_score:+.0f} "
                  f"conf {decision.confidence:.0%}")
        trade = self.trader.close_position(symbol, price, reason=reason)
        if trade is None:
            return AutoTradeDecision(symbol, "skipped",
                                     decision.composite_score, decision.confidence,
                                     0.0, price,
                                     reason="close_position returned None")
        self.trader.save()
        self._record_cooldown(symbol)
        return AutoTradeDecision(symbol, "closed",
                                 decision.composite_score, decision.confidence,
                                 trade.qty, price, reason=reason)

    def run_watchlist(self, symbols: list[str]) -> list[AutoTradeDecision]:
        """Run process_symbol for each symbol. Returns all decisions."""
        results = []
        for sym in symbols:
            try:
                results.append(self.process_symbol(sym))
            except Exception as exc:
                logger.exception("Auto-trader error on %s: %s", sym, exc)
                results.append(AutoTradeDecision(sym, "skipped", 0.0, 0.0, 0.0, 0.0,
                                                 reason=f"exception: {exc}"))
        return results
