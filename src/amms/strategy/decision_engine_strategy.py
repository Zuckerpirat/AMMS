"""DecisionEngineStrategy — wraps amms.engine.decision as a Strategy.

Bridges the Central Decision Engine (16-indicator composite) into the
Strategy protocol so it can be used by:
  - The BacktestEngine (/backtest command, walk-forward analysis)
  - The Scheduler / Executor (run_tick)
  - The CLI (amms run --strategy decision_engine)

The engine needs at minimum 120 bars; we request 200 so all long-period
indicators (200-day MA, etc.) are warm on first evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar
from amms.strategy.base import Signal

_DE_LOOKBACK = 200   # bars Decision Engine needs to warm all indicators


@dataclass(frozen=True)
class DecisionEngineStrategy:
    """Multi-indicator strategy backed by the Central Decision Engine.

    Parameters
    ----------
    min_confidence:
        Signals with engine confidence below this are treated as hold.
    min_score:
        Signals with |composite_score| below this are treated as hold.
        Score is 0–100; 35 filters noise without over-filtering.
    allow_strong_only:
        When True, only "strong_buy" / "strong_sell" actions are acted on.
        Useful for conservative paper-trading modes.
    """

    min_confidence: float = 0.60
    min_score: float = 35.0
    allow_strong_only: bool = False
    mode: str = "swing"           # "conservative" | "swing" | "meme" | "event"
    name: str = "decision_engine"

    @property
    def lookback(self) -> int:
        return _DE_LOOKBACK

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0

        if len(bars) < 120:
            return Signal(
                symbol, "hold",
                f"need 120 bars, have {len(bars)}",
                price,
            )

        from amms.engine.decision import analyze
        report = analyze(
            bars,
            symbol=symbol,
            min_confidence=self.min_confidence,
            mode=self.mode,
        )

        if report is None:
            return Signal(symbol, "hold", "decision engine returned no result", price)

        if report.risk_blocked:
            return Signal(
                symbol, "hold",
                f"risk gate: {report.risk_reason}",
                price,
                score=report.composite_score,
            )

        if abs(report.composite_score) < self.min_score:
            return Signal(
                symbol, "hold",
                f"score {report.composite_score:+.0f} below min {self.min_score:.0f}",
                price,
                score=report.composite_score,
            )

        if self.allow_strong_only and report.action not in {"strong_buy", "strong_sell"}:
            return Signal(
                symbol, "hold",
                f"action {report.action!r} not strong (allow_strong_only=True)",
                price,
                score=report.composite_score,
            )

        reason = (
            f"DE {report.action} | score {report.composite_score:+.0f} "
            f"conf {report.confidence:.0%} | {report.verdict}"
        )

        if report.action in {"buy", "strong_buy"}:
            return Signal(symbol, "buy", reason, price, score=report.composite_score)

        if report.action in {"sell", "strong_sell"}:
            return Signal(symbol, "sell", reason, price, score=report.composite_score)

        return Signal(symbol, "hold", reason, price, score=report.composite_score)
