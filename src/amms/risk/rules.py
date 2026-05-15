from __future__ import annotations

from dataclasses import dataclass

from amms.data.sectors import group_by_sector, sector_for


@dataclass(frozen=True)
class RiskConfig:
    max_open_positions: int = 5
    max_position_pct: float = 0.02
    daily_loss_pct: float = -0.03
    blackout_minutes_after_open: int = 5
    blackout_minutes_before_close: int = 5
    max_buys_per_tick: int | None = None
    min_hold_days: int = 0
    pdt_min_equity: float = 25_000.0
    pdt_max_day_trades: int = 3
    # When set, flatten all open positions when fewer than this many minutes
    # remain until market close. Useful for day-trading profiles. 0 = off.
    force_close_minutes_before_close: int = 0
    # Per-position stop-loss as a positive fraction (e.g. 0.05 = 5%). When the
    # unrealized loss vs. entry exceeds this, the position is force-sold on the
    # next tick — bypassing min_hold_days. 0 (default) = disabled.
    stop_loss_pct: float = 0.0
    # Per-position trailing stop as a positive fraction (e.g. 0.10 = 10%). Sell
    # when the price drops this much from the position's recorded high-water
    # mark. Requires the executor to track per-symbol highs. 0 = disabled.
    trailing_stop_pct: float = 0.0
    # Maximum share of the portfolio any single GICS sector may occupy as a
    # fraction (e.g. 0.40 = 40%). 0 (default) = disabled.
    max_sector_pct: float = 0.0

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
        if self.pdt_min_equity < 0:
            raise ValueError(f"pdt_min_equity must be >= 0, got {self.pdt_min_equity}")
        if self.pdt_max_day_trades < 0:
            raise ValueError(f"pdt_max_day_trades must be >= 0, got {self.pdt_max_day_trades}")
        if self.force_close_minutes_before_close < 0:
            raise ValueError(
                f"force_close_minutes_before_close must be >= 0, got "
                f"{self.force_close_minutes_before_close}"
            )
        if not 0 <= self.stop_loss_pct < 1:
            raise ValueError(
                f"stop_loss_pct must be in [0, 1), got {self.stop_loss_pct}"
            )
        if not 0 <= self.trailing_stop_pct < 1:
            raise ValueError(
                f"trailing_stop_pct must be in [0, 1), got {self.trailing_stop_pct}"
            )
        if not 0 <= self.max_sector_pct <= 1:
            raise ValueError(
                f"max_sector_pct must be in [0, 1], got {self.max_sector_pct}"
            )


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    qty: int
    reason: str


def position_size(
    equity: float,
    price: float,
    max_position_pct: float,
    *,
    atr: float | None = None,
    target_risk_pct: float = 0.01,
) -> int:
    """Whole shares, at most ``max_position_pct`` of equity.

    When ``atr`` is supplied (from the last 14-bar ATR of the instrument) the
    allocation is also capped by a volatility budget: the position is sized so
    that one ATR of adverse move equals ``target_risk_pct`` of equity
    (default 1%).  This prevents over-sizing volatile instruments.
    """
    if equity <= 0 or price <= 0:
        return 0
    target_dollars = equity * max_position_pct
    if atr and atr > 0:
        vol_dollars = (equity * target_risk_pct) / atr * price
        target_dollars = min(target_dollars, vol_dollars)
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
    atr: float | None = None,
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
    qty = position_size(equity, price, config.max_position_pct, atr=atr)
    if qty <= 0:
        return RiskDecision(False, 0, f"sized to 0 shares at ${price:.2f}")
    cost = qty * price
    if cost > cash:
        return RiskDecision(False, 0, f"insufficient cash: need ${cost:.2f}, have ${cash:.2f}")
    return RiskDecision(True, qty, f"buy {qty} @ ${price:.2f} (~${cost:.2f})")


# Prefix used to mark synthetic SELL signals generated by the stop-loss layer.
# The executor checks for this prefix and bypasses min_hold_days so a stop can
# always exit, even on a freshly-opened position.
STOP_LOSS_REASON_PREFIX = "stop_loss:"


@dataclass(frozen=True)
class StopLossTrigger:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    loss_pct: float  # negative; e.g. -0.06 = 6% below entry
    reason: str  # human-readable, prefixed with STOP_LOSS_REASON_PREFIX
    kind: str  # "fixed" or "trailing"


def check_stop_losses(
    *,
    positions: list,
    config: RiskConfig,
    high_water_marks: dict[str, float] | None = None,
) -> list[StopLossTrigger]:
    """Return positions that have breached the stop-loss or trailing-stop.

    Pure function. Caller (the executor) translates triggers into SELL signals
    and passes them through the normal sell pipeline.

    ``positions``: iterable of objects with ``.symbol``, ``.qty``,
    ``.avg_entry_price``, ``.market_value`` (Alpaca ``Position`` shape).
    ``high_water_marks``: per-symbol best price seen since entry; used for
    trailing stops. Pass ``None`` or an empty dict to skip trailing.
    """
    triggers: list[StopLossTrigger] = []
    if config.stop_loss_pct <= 0 and config.trailing_stop_pct <= 0:
        return triggers

    for pos in positions:
        qty = float(getattr(pos, "qty", 0) or 0)
        entry = float(getattr(pos, "avg_entry_price", 0) or 0)
        market_value = float(getattr(pos, "market_value", 0) or 0)
        if qty <= 0 or entry <= 0:
            continue
        current_price = market_value / qty if qty else 0.0
        if current_price <= 0:
            continue

        loss_pct = (current_price - entry) / entry  # negative when down

        # Fixed stop-loss vs entry price.
        if config.stop_loss_pct > 0 and loss_pct <= -config.stop_loss_pct:
            triggers.append(
                StopLossTrigger(
                    symbol=pos.symbol,
                    qty=qty,
                    avg_entry_price=entry,
                    current_price=current_price,
                    loss_pct=loss_pct,
                    reason=(
                        f"{STOP_LOSS_REASON_PREFIX} {loss_pct:+.2%} vs entry "
                        f"${entry:.2f} (cap -{config.stop_loss_pct:.0%})"
                    ),
                    kind="fixed",
                )
            )
            continue  # one trigger per symbol; fixed stop wins over trailing

        # Trailing stop vs high-water mark.
        if config.trailing_stop_pct > 0 and high_water_marks:
            high = high_water_marks.get(pos.symbol)
            if high and high > 0:
                drop_from_high = (current_price - high) / high
                if drop_from_high <= -config.trailing_stop_pct:
                    triggers.append(
                        StopLossTrigger(
                            symbol=pos.symbol,
                            qty=qty,
                            avg_entry_price=entry,
                            current_price=current_price,
                            loss_pct=loss_pct,
                            reason=(
                                f"{STOP_LOSS_REASON_PREFIX} trailing "
                                f"{drop_from_high:+.2%} from high ${high:.2f} "
                                f"(cap -{config.trailing_stop_pct:.0%})"
                            ),
                            kind="trailing",
                        )
                    )

    return triggers


def check_sector_cap(
    *,
    symbol: str,
    positions: list,
    total_equity: float,
    config: RiskConfig,
) -> RiskDecision | None:
    """Return a blocking RiskDecision when buying ``symbol`` would push its
    GICS sector beyond ``config.max_sector_pct`` of ``total_equity``.

    Returns ``None`` when the cap is disabled or the buy is safe.

    ``positions``: iterable of objects with ``.symbol`` and ``.market_value``.
    """
    if config.max_sector_pct <= 0 or total_equity <= 0:
        return None

    target_sector = sector_for(symbol)

    sector_exposure = group_by_sector(
        [(p.symbol, float(getattr(p, "market_value", 0) or 0)) for p in positions]
    )
    current = sector_exposure.get(target_sector, 0.0)
    current_pct = current / total_equity

    if current_pct >= config.max_sector_pct:
        return RiskDecision(
            False,
            0,
            (
                f"sector cap: {target_sector} already at "
                f"{current_pct:.1%} of equity "
                f"(cap {config.max_sector_pct:.0%})"
            ),
        )
    return None
