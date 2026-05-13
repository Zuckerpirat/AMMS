from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClockStatus:
    """Snapshot of the Alpaca market clock."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime


def parse_alpaca_dt(value: str) -> datetime:
    """Parse Alpaca's ISO-8601 timestamps. Python 3.11+ handles the offset natively."""
    return datetime.fromisoformat(value)
