"""Periodic WSB-driven watchlist expansion.

Runs the WSBScanner on a schedule, takes the top tickers that meet a
sentiment + mentions threshold, and returns them as a set of extra
symbols the scheduler merges into the active watchlist.

Discovery vs. scanner: ``WSBScanner`` is the raw data layer. This module
adds opinion (filters, refresh cadence, dedup) and exposes a tiny stateful
helper the scheduler calls each tick.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from amms.data.wsb_scanner import WSBScanner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WSBDiscoveryConfig:
    """Knobs for auto-watchlist expansion. Disabled by default."""

    enabled: bool = False
    top_n: int = 5
    min_mentions: int = 5
    min_sentiment: float = 0.0  # require non-bearish chatter
    refresh_hours: float = 24.0  # rescan at most this often
    subreddits: tuple[str, ...] = ("wallstreetbets",)
    time_filter: str = "day"  # day | week | month — Reddit's `t` parameter

    def __post_init__(self) -> None:
        if self.top_n <= 0:
            raise ValueError(f"wsb_discovery.top_n must be > 0, got {self.top_n}")
        if self.min_mentions <= 0:
            raise ValueError(
                f"wsb_discovery.min_mentions must be > 0, got {self.min_mentions}"
            )
        if not -1.0 <= self.min_sentiment <= 1.0:
            raise ValueError(
                f"wsb_discovery.min_sentiment must be in [-1, 1], "
                f"got {self.min_sentiment}"
            )
        if self.refresh_hours <= 0:
            raise ValueError(
                f"wsb_discovery.refresh_hours must be > 0, "
                f"got {self.refresh_hours}"
            )
        valid_filters = {"hour", "day", "week", "month", "year", "all"}
        if self.time_filter not in valid_filters:
            raise ValueError(
                f"wsb_discovery.time_filter must be one of {sorted(valid_filters)}, "
                f"got {self.time_filter!r}"
            )


@dataclass
class DiscoveryState:
    """Mutable state kept by the scheduler between ticks."""

    last_refresh_ts: float = 0.0
    extras: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DiscoveryDelta:
    """Returned by ``maybe_refresh`` so the caller can announce changes."""

    refreshed: bool
    added: frozenset[str]
    removed: frozenset[str]
    extras: frozenset[str]


def maybe_refresh(
    state: DiscoveryState,
    config: WSBDiscoveryConfig,
    *,
    static_watchlist: set[str],
    now_seconds: float,
    scanner: WSBScanner | None = None,
) -> DiscoveryDelta:
    """Possibly refresh the discovered extras.

    No-op when discovery is disabled, when Reddit creds are missing, or when
    the refresh window has not elapsed. Otherwise runs the scanner, applies
    the mentions/sentiment filters, drops anything already on the static
    watchlist, and updates ``state`` in place.

    Returns a ``DiscoveryDelta`` describing what changed so the scheduler can
    notify the user about additions/removals.
    """
    no_change = DiscoveryDelta(
        refreshed=False,
        added=frozenset(),
        removed=frozenset(),
        extras=state.extras,
    )

    if not config.enabled:
        return no_change

    elapsed_hours = (now_seconds - state.last_refresh_ts) / 3600.0
    if elapsed_hours < config.refresh_hours:
        return no_change

    if not os.environ.get("REDDIT_CLIENT_ID", "").strip():
        logger.info("wsb-discovery skipped: REDDIT_CLIENT_ID not set")
        state.last_refresh_ts = now_seconds  # don't hammer log every tick
        return no_change

    state.last_refresh_ts = now_seconds

    owns_scanner = scanner is None
    scanner = scanner or WSBScanner(subreddits=config.subreddits)
    try:
        results = scanner.scan(
            limit_per_sub=100,
            time_filter=config.time_filter,
            min_mentions=config.min_mentions,
            top_n=None,  # filter manually so sentiment cap can shrink the list
        )
    except Exception:
        logger.warning("wsb-discovery scan failed", exc_info=True)
        return DiscoveryDelta(
            refreshed=False, added=frozenset(), removed=frozenset(),
            extras=state.extras,
        )
    finally:
        if owns_scanner:
            scanner.close()

    static_upper = {s.upper() for s in static_watchlist}
    filtered = [
        t for t in results
        if t.avg_sentiment >= config.min_sentiment
        and t.symbol.upper() not in static_upper
    ]
    new_extras = frozenset(t.symbol.upper() for t in filtered[: config.top_n])

    added = new_extras - state.extras
    removed = state.extras - new_extras
    state.extras = new_extras

    return DiscoveryDelta(
        refreshed=True, added=added, removed=removed, extras=new_extras
    )


def format_delta_message(delta: DiscoveryDelta) -> str:
    """Render a Telegram-friendly summary when the discovered set changed."""
    parts = []
    if delta.added:
        parts.append("➕ added: " + ", ".join(sorted(delta.added)))
    if delta.removed:
        parts.append("➖ removed: " + ", ".join(sorted(delta.removed)))
    if not parts:
        return f"WSB watchlist unchanged ({len(delta.extras)} extras)"
    parts.insert(0, "WSB watchlist updated")
    if delta.extras:
        parts.append("now: " + ", ".join(sorted(delta.extras)))
    return "\n".join(parts)
