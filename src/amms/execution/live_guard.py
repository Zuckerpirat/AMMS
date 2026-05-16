"""Live-Trading Acknowledgement Guard.

The bot has TWO physical safety layers preventing accidental live trading:

  Layer 1 (config.py):  PAPER_HOST_MARKER guard refuses any non-paper URL.
  Layer 2 (this file):  LiveGuard requires three independent conditions
                        before live trading can be enabled.

To enable live trading a human must:

  1. Set ALPACA_BASE_URL to the live endpoint (api.alpaca.markets)
     — AND deliberately edit config.py to allow non-paper URLs
       (we will NOT make that one-line change for the user)
  2. Set environment variable AMMS_LIVE_ACKNOWLEDGED=I_UNDERSTAND_REAL_MONEY
  3. Set environment variable AMMS_LIVE_MAX_ORDER_USD (per-order $ cap)
  4. Set environment variable AMMS_LIVE_MAX_DAILY_USD (per-day $ cap)

If any condition is missing, every call to `assert_live_allowed()`
raises RuntimeError. The auto-trader and broker switch call this
before any live order is even attempted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

LIVE_ACK_PHRASE = "I_UNDERSTAND_REAL_MONEY"


@dataclass(frozen=True)
class LiveLimits:
    enabled: bool
    max_order_usd: float
    max_daily_usd: float
    reason_if_disabled: str = ""


class LiveTradingNotAllowed(RuntimeError):
    """Raised when something tries to do live trading without all guards passed."""


def check_live_allowed() -> LiveLimits:
    """Return LiveLimits indicating whether live trading is allowed.

    Never raises. The auto-trader uses `assert_live_allowed()` for the
    enforcing version.
    """
    ack = os.environ.get("AMMS_LIVE_ACKNOWLEDGED", "").strip()
    if ack != LIVE_ACK_PHRASE:
        return LiveLimits(
            enabled=False,
            max_order_usd=0.0,
            max_daily_usd=0.0,
            reason_if_disabled=(
                f"AMMS_LIVE_ACKNOWLEDGED must equal '{LIVE_ACK_PHRASE}'"
            ),
        )

    try:
        per_order = float(os.environ.get("AMMS_LIVE_MAX_ORDER_USD", "0"))
    except ValueError:
        per_order = 0.0
    try:
        per_day = float(os.environ.get("AMMS_LIVE_MAX_DAILY_USD", "0"))
    except ValueError:
        per_day = 0.0

    if per_order <= 0:
        return LiveLimits(
            enabled=False, max_order_usd=0.0, max_daily_usd=0.0,
            reason_if_disabled="AMMS_LIVE_MAX_ORDER_USD must be a positive number",
        )
    if per_day <= 0:
        return LiveLimits(
            enabled=False, max_order_usd=per_order, max_daily_usd=0.0,
            reason_if_disabled="AMMS_LIVE_MAX_DAILY_USD must be a positive number",
        )

    return LiveLimits(enabled=True, max_order_usd=per_order, max_daily_usd=per_day)


def assert_live_allowed() -> LiveLimits:
    """Raise LiveTradingNotAllowed unless every safety condition passes."""
    limits = check_live_allowed()
    if not limits.enabled:
        raise LiveTradingNotAllowed(
            f"Live trading blocked: {limits.reason_if_disabled}"
        )
    return limits


def is_live_mode_url(url: str) -> bool:
    """Return True if the URL looks like the live (non-paper) Alpaca endpoint."""
    return "paper-api" not in url.lower() and "api.alpaca.markets" in url.lower()
