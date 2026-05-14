from __future__ import annotations

import pytest

from amms.data.wsb_discovery import (
    DiscoveryState,
    WSBDiscoveryConfig,
    format_delta_message,
    maybe_refresh,
)
from amms.data.wsb_scanner import TrendingTicker


class _StubScanner:
    """Stands in for WSBScanner — returns canned TrendingTicker results."""

    def __init__(self, results: list[TrendingTicker]) -> None:
        self._results = results
        self.scan_calls = 0

    def scan(self, **_kw) -> list[TrendingTicker]:
        self.scan_calls += 1
        return self._results

    def close(self) -> None:
        pass


def _ticker(symbol: str, mentions: int = 10, avg: float = 0.5) -> TrendingTicker:
    return TrendingTicker(
        symbol=symbol, mentions=mentions, avg_sentiment=avg,
        bullish_posts=mentions, bearish_posts=0,
    )


def test_disabled_returns_no_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    state = DiscoveryState()
    config = WSBDiscoveryConfig(enabled=False)
    delta = maybe_refresh(
        state, config,
        static_watchlist={"AAPL"},
        now_seconds=1000.0,
        scanner=_StubScanner([_ticker("NVDA")]),
    )
    assert delta.refreshed is False
    assert delta.extras == frozenset()


def test_missing_reddit_creds_no_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    state = DiscoveryState()
    config = WSBDiscoveryConfig(enabled=True, refresh_hours=24)
    now = 1_700_000_000.0  # realistic unix time so the "elapsed" check passes
    delta = maybe_refresh(
        state, config,
        static_watchlist=set(),
        now_seconds=now,
        scanner=_StubScanner([_ticker("NVDA")]),
    )
    assert delta.refreshed is False
    # last_refresh_ts is bumped so we don't log every tick
    assert state.last_refresh_ts == now


def test_within_refresh_window_skips_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    config = WSBDiscoveryConfig(enabled=True, refresh_hours=24)
    state = DiscoveryState(last_refresh_ts=1000.0, extras=frozenset({"NVDA"}))
    scanner = _StubScanner([_ticker("PLTR")])
    delta = maybe_refresh(
        state, config,
        static_watchlist=set(),
        now_seconds=1000.0 + 1 * 3600,  # only 1h elapsed
        scanner=scanner,
    )
    assert scanner.scan_calls == 0
    assert delta.refreshed is False
    assert delta.extras == frozenset({"NVDA"})


def test_refresh_runs_when_window_elapsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    config = WSBDiscoveryConfig(enabled=True, top_n=3, min_mentions=5,
                                min_sentiment=0.0, refresh_hours=24)
    scanner = _StubScanner([
        _ticker("NVDA", mentions=42, avg=0.7),
        _ticker("PLTR", mentions=25, avg=0.4),
        _ticker("GME", mentions=15, avg=-0.5),  # bearish, filtered out
        _ticker("SOFI", mentions=10, avg=0.2),
        _ticker("MARA", mentions=8, avg=0.1),
        _ticker("RIOT", mentions=6, avg=0.05),  # over top_n
    ])
    state = DiscoveryState()
    delta = maybe_refresh(
        state, config,
        static_watchlist=set(),
        now_seconds=1_700_000_000.0,
        scanner=scanner,
    )
    assert scanner.scan_calls == 1
    assert delta.refreshed is True
    # GME excluded by sentiment filter; top 3 of remaining
    assert delta.extras == frozenset({"NVDA", "PLTR", "SOFI"})


def test_static_watchlist_tickers_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    config = WSBDiscoveryConfig(enabled=True, top_n=5, refresh_hours=24)
    scanner = _StubScanner([
        _ticker("NVDA", mentions=42),
        _ticker("AAPL", mentions=30),  # already on static list
        _ticker("PLTR", mentions=25),
    ])
    state = DiscoveryState()
    delta = maybe_refresh(
        state, config,
        static_watchlist={"AAPL"},
        now_seconds=1_700_000_000.0,
        scanner=scanner,
    )
    assert "AAPL" not in delta.extras
    assert delta.extras == frozenset({"NVDA", "PLTR"})


def test_added_and_removed_diff_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REDDIT_CLIENT_ID", "x")
    config = WSBDiscoveryConfig(enabled=True, top_n=5, refresh_hours=24)
    state = DiscoveryState(extras=frozenset({"NVDA", "PLTR", "GME"}))
    scanner = _StubScanner([
        _ticker("NVDA", mentions=42),
        _ticker("PLTR", mentions=25),
        _ticker("SOFI", mentions=10),  # new
    ])
    # advance past refresh window
    state.last_refresh_ts = 0.0
    delta = maybe_refresh(
        state, config,
        static_watchlist=set(),
        now_seconds=24 * 3600 + 1,
        scanner=scanner,
    )
    assert delta.added == frozenset({"SOFI"})
    assert delta.removed == frozenset({"GME"})
    assert delta.extras == frozenset({"NVDA", "PLTR", "SOFI"})


def test_format_delta_message_handles_added_only() -> None:
    from amms.data.wsb_discovery import DiscoveryDelta

    msg = format_delta_message(DiscoveryDelta(
        refreshed=True,
        added=frozenset({"NVDA", "PLTR"}),
        removed=frozenset(),
        extras=frozenset({"NVDA", "PLTR"}),
    ))
    assert "added" in msg
    assert "NVDA" in msg
    assert "PLTR" in msg


def test_format_delta_message_no_changes() -> None:
    from amms.data.wsb_discovery import DiscoveryDelta

    msg = format_delta_message(DiscoveryDelta(
        refreshed=True, added=frozenset(), removed=frozenset(),
        extras=frozenset({"NVDA"}),
    ))
    assert "unchanged" in msg


def test_invalid_config_values_are_rejected() -> None:
    with pytest.raises(ValueError):
        WSBDiscoveryConfig(top_n=0)
    with pytest.raises(ValueError):
        WSBDiscoveryConfig(min_mentions=0)
    with pytest.raises(ValueError):
        WSBDiscoveryConfig(min_sentiment=2.0)
    with pytest.raises(ValueError):
        WSBDiscoveryConfig(refresh_hours=0)
    with pytest.raises(ValueError):
        WSBDiscoveryConfig(time_filter="quarter")
