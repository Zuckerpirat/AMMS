from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar
from amms.features import n_day_return, realized_vol, relative_volume, rsi
from amms.strategy.base import Signal

_SENTIMENT_OVERLAY: dict[str, float] = {}


def set_sentiment_overlay(scores: dict[str, float]) -> None:
    """Replace the process-wide sentiment overlay used by CompositeStrategy.

    Pass an empty dict to clear. Designed for the scheduler to push fresh
    Reddit sentiment scores between ticks without threading state through
    the strategy interface.
    """
    _SENTIMENT_OVERLAY.clear()
    _SENTIMENT_OVERLAY.update(scores)


def get_sentiment_overlay() -> dict[str, float]:
    return dict(_SENTIMENT_OVERLAY)


@dataclass(frozen=True)
class CompositeStrategy:
    """Multi-feature long-only screen.

    A symbol passes all filters to earn a buy:
      - N-day return >= momentum_min  (trending up)
      - RSI <= rsi_max               (not overbought)
      - realized vol <= vol_max      (not too jumpy)
      - relative volume >= rvol_min  (today's volume is meaningfully higher
                                      than the recent average)

    The signal carries a score = momentum * rvol / max(vol, 0.01). Higher
    score means stronger setup. Used by ranking layers later.

    Exit (sell) fires when the N-day return reverses past ``momentum_sell``.
    """

    momentum_n: int = 20
    momentum_min: float = 0.05
    momentum_sell: float = -0.02

    rsi_n: int = 14
    rsi_max: float = 70.0

    vol_n: int = 20
    vol_max: float = 0.40

    rvol_n: int = 20
    rvol_min: float = 1.2

    # Sentiment bonus multiplier. Score is multiplied by (1 + sentiment_weight * s)
    # where s in [-1, 1] from the overlay. 0 disables.
    sentiment_weight: float = 0.0
    sentiment_min: float = -1.0  # set above -1 to require a minimum sentiment

    name: str = "composite"

    def __post_init__(self) -> None:
        for field_name in ("momentum_n", "rsi_n", "vol_n", "rvol_n"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0, got {value}")
        if self.momentum_sell >= self.momentum_min:
            raise ValueError(
                f"momentum_sell ({self.momentum_sell}) must be < "
                f"momentum_min ({self.momentum_min})"
            )

    @property
    def lookback(self) -> int:
        return max(self.momentum_n, self.rsi_n, self.vol_n, self.rvol_n) + 1

    def evaluate(self, symbol: str, bars: list[Bar]) -> Signal:
        price = bars[-1].close if bars else 0.0
        if len(bars) < self.lookback:
            return Signal(
                symbol,
                "hold",
                f"need {self.lookback} bars, have {len(bars)}",
                price,
            )

        momentum = n_day_return(bars, self.momentum_n)
        r = rsi(bars, self.rsi_n)
        vol = realized_vol(bars, self.vol_n)
        rvol = relative_volume(bars, self.rvol_n)

        if momentum is None or r is None or vol is None or rvol is None:
            return Signal(symbol, "hold", "missing feature inputs", price)

        if momentum <= self.momentum_sell:
            return Signal(
                symbol,
                "sell",
                f"momentum {momentum:+.2%} <= sell threshold {self.momentum_sell:+.2%}",
                price,
                score=0.0,
            )

        reasons: list[str] = []
        if momentum < self.momentum_min:
            reasons.append(f"momentum {momentum:+.2%} < {self.momentum_min:+.2%}")
        if r > self.rsi_max:
            reasons.append(f"RSI {r:.1f} > {self.rsi_max:.1f}")
        if vol > self.vol_max:
            reasons.append(f"realized vol {vol:.2%} > {self.vol_max:.2%}")
        if rvol < self.rvol_min:
            reasons.append(f"rvol {rvol:.2f} < {self.rvol_min:.2f}")

        if reasons:
            return Signal(symbol, "hold", "; ".join(reasons), price)

        score = (momentum * rvol) / max(vol, 0.01)
        sentiment = _SENTIMENT_OVERLAY.get(symbol, 0.0)
        if sentiment < self.sentiment_min:
            return Signal(
                symbol,
                "hold",
                f"sentiment {sentiment:+.2f} < min {self.sentiment_min:+.2f}",
                price,
            )
        if self.sentiment_weight != 0:
            score *= 1 + self.sentiment_weight * sentiment
        reason = (
            f"composite ok (mom={momentum:+.2%}, rsi={r:.1f}, "
            f"vol={vol:.2%}, rvol={rvol:.2f}, sentiment={sentiment:+.2f}, "
            f"score={score:.2f})"
        )
        return Signal(symbol, "buy", reason, price, score=score)
