from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_open_positions: int = 5
    max_position_pct: float = 0.02
    daily_loss_pct: float = -0.03
    blackout_minutes_after_open: int = 5
    blackout_minutes_before_close: int = 5
    max_buys_per_tick: int | None = None
    min_hold_days: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.max_position_pct <= 1:
            raise ValueError(f"max_position_pct must be in (0, 1], got {self.max_position_pct}")
        if self.daily_loss_pct >= 0:
            raise ValueError(f"daily_loss_pct must be negative, got {self.daily_loss_pct}")
        if self.max_open_positions <= 0:
            raise ValueError(f"max_open_positions must be > 0, got {self.max_open_positions}")
        if self.max_buys_per_tick is not None and self.max_buys_per_tick <= 0:
            raise ValueError(
                f"max_buys_per_tick must be > 0 or None, got {self.max_buys_per_tick}"
            )
        if self.min_hold_days < 0:
            raise ValueError(f"min_hold_days must be >= 0, got {self.min_hold_days}")


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    qty: int
    reason: str


def position_size(equity: float, price: float, max_position_pct: float) -> int:
    """Whole shares for a position not exceeding ``max_position_pct`` of equity."""
    if equity <= 0 or price <= 0:
        return 0
    target_dollars = equity * max_position_pct
    return max(0, int(target_dollars // price))


def check_buy(
    *,
    equity: float,
    price: float,
    cash: float,
    open_positions: int,
    daily_pnl_pct: float,
    already_holds: bool,
    config: RiskConfig,
) -> RiskDecision:
    """Decide whether a BUY is allowed and at what size.

    Pure function. Caller does the side-effects.
    """
    if daily_pnl_pct <= config.daily_loss_pct:
        return RiskDecision(False, 0, f"daily loss cap hit ({daily_pnl_pct:.2%})")
    if already_holds:
        return RiskDecision(False, 0, "already long this symbol")
    if open_positions >= config.max_open_positions:
        return RiskDecision(False, 0, f"max open positions ({open_positions})")
    qty = position_size(equity, price, config.max_position_pct)
    if qty <= 0:
        return RiskDecision(False, 0, f"sized to 0 shares at ${price:.2f}")
    cost = qty * price
    if cost > cash:
        return RiskDecision(False, 0, f"insufficient cash: need ${cost:.2f}, have ${cash:.2f}")
    return RiskDecision(True, qty, f"buy {qty} @ ${price:.2f} (~${cost:.2f})")
