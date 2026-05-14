from __future__ import annotations

from dataclasses import dataclass

from amms.data.bars import Bar


@dataclass(frozen=True)
class UniverseFilter:
    """Pre-trade gate applied before BUY orders are sized.

    Used to keep the bot out of illiquid or absurdly-priced names without
    having to manually prune the watchlist. SELL orders are never filtered —
    if you somehow ended up holding something that no longer passes, you must
    still be able to exit.

    All defaults are inert (no filtering). Set thresholds in config.yaml to
    activate. For penny-stock trading, the common settings are
    ``min_price: 1.0`` and ``min_avg_dollar_volume: 1000000``.
    """

    min_price: float = 0.0
    max_price: float | None = None
    min_avg_dollar_volume: float = 0.0
    adv_lookback: int = 20
    require_tradable: bool = False

    def __post_init__(self) -> None:
        if self.min_price < 0:
            raise ValueError(f"min_price must be >= 0, got {self.min_price}")
        if self.max_price is not None and self.max_price <= self.min_price:
            raise ValueError(
                f"max_price ({self.max_price}) must be > min_price ({self.min_price})"
            )
        if self.min_avg_dollar_volume < 0:
            raise ValueError(
                f"min_avg_dollar_volume must be >= 0, got {self.min_avg_dollar_volume}"
            )
        if self.adv_lookback <= 0:
            raise ValueError(f"adv_lookback must be > 0, got {self.adv_lookback}")

    def passes(self, bars: list[Bar]) -> tuple[bool, str | None]:
        """Return ``(True, None)`` if the symbol qualifies, else ``(False, reason)``."""
        if not bars:
            return False, "no bars available"

        last_close = bars[-1].close
        if last_close < self.min_price:
            return False, f"price ${last_close:.2f} < min ${self.min_price:.2f}"
        if self.max_price is not None and last_close > self.max_price:
            return False, f"price ${last_close:.2f} > max ${self.max_price:.2f}"

        if self.min_avg_dollar_volume > 0:
            if len(bars) < self.adv_lookback:
                return (
                    False,
                    f"need {self.adv_lookback} bars for ADV, have {len(bars)}",
                )
            recent = bars[-self.adv_lookback :]
            dollar_volumes = [b.close * b.volume for b in recent]
            adv = sum(dollar_volumes) / len(dollar_volumes)
            if adv < self.min_avg_dollar_volume:
                return (
                    False,
                    f"ADV ${adv:,.0f} < min ${self.min_avg_dollar_volume:,.0f}",
                )

        return True, None

    def passes_asset(self, asset: dict | None) -> tuple[bool, str | None]:
        """Optional check against an Alpaca asset payload. Returns pass when
        require_tradable is off or the payload reports tradable + active."""
        if not self.require_tradable:
            return True, None
        if asset is None:
            return False, "asset metadata unavailable"
        if asset.get("status") != "active":
            return False, f"asset status={asset.get('status')!r}"
        if not asset.get("tradable", False):
            return False, "asset not tradable on Alpaca"
        return True, None
