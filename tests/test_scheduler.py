from __future__ import annotations

import threading
from pathlib import Path

import httpx
import respx

from amms.config import (
    AppConfig,
    SchedulerConfig,
    Settings,
    StrategyConfig,
)
from amms.notifier import NullNotifier
from amms.risk import RiskConfig
from amms.scheduler import (
    LoopState,
    _announce_tick,
    _handle_closed,
    _maybe_refresh_sentiment,
    run_loop,
)

PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"


class RecordingNotifier:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        alpaca_api_key="k",
        alpaca_api_secret="s",
        alpaca_base_url=PAPER_URL,
        alpaca_data_url=DATA_URL,
        db_path=tmp_path / "amms.sqlite",
        log_level="INFO",
    )


def _config() -> AppConfig:
    return AppConfig(
        watchlist=("AAPL",),
        strategy=StrategyConfig(name="sma_cross", params={"fast": 3, "slow": 5}),
        risk=RiskConfig(
            max_open_positions=5,
            max_position_pct=0.02,
            daily_loss_pct=-0.03,
        ),
        scheduler=SchedulerConfig(tick_seconds=60),
    )


def _bars_payload(symbol: str, closes: list[float]) -> dict:
    return {
        "bars": {
            symbol.upper(): [
                {
                    "t": f"2025-01-{i + 1:02d}T05:00:00Z",
                    "o": c, "h": c, "l": c, "c": c, "v": 100,
                }
                for i, c in enumerate(closes)
            ]
        },
        "next_page_token": None,
    }


@respx.mock
def test_run_loop_executes_a_tick_then_stops(tmp_path: Path) -> None:
    respx.get(f"{PAPER_URL}/v2/clock").mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": "2026-05-13T14:00:00-04:00",
                "is_open": True,
                "next_open": "2026-05-14T09:30:00-04:00",
                "next_close": "2026-05-13T16:00:00-04:00",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/account").mock(
        return_value=httpx.Response(
            200,
            json={
                "equity": "100000",
                "cash": "100000",
                "buying_power": "100000",
                "status": "ACTIVE",
            },
        )
    )
    respx.get(f"{PAPER_URL}/v2/positions").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{PAPER_URL}/v2/orders").mock(return_value=httpx.Response(200, json=[]))
    respx.get(f"{DATA_URL}/v2/stocks/bars").mock(
        return_value=httpx.Response(200, json=_bars_payload("AAPL", [1] * 6 + [10]))
    )

    notifier = RecordingNotifier()
    stop = threading.Event()

    # Stop the loop as soon as the bot tries to sleep between ticks. Doing
    # it via Event.set means the next wait() returns truthy and the loop
    # exits cleanly without a real sleep.
    original_wait = stop.wait

    def wait_then_stop(timeout: float | None = None) -> bool:
        stop.set()
        return original_wait(timeout)

    stop.wait = wait_then_stop  # type: ignore[method-assign]

    run_loop(
        _settings(tmp_path),
        _config(),
        execute=False,
        notifier=notifier,
        stop=stop,
        install_signal_handlers=False,
    )

    assert any("started" in m for m in notifier.messages)
    assert any("stopped" in m for m in notifier.messages)


@respx.mock
def test_run_loop_idles_when_market_closed(tmp_path: Path) -> None:
    clock_route = respx.get(f"{PAPER_URL}/v2/clock").mock(
        return_value=httpx.Response(
            200,
            json={
                "timestamp": "2026-05-13T22:00:00-04:00",
                "is_open": False,
                "next_open": "2026-05-14T09:30:00-04:00",
                "next_close": "2026-05-14T16:00:00-04:00",
            },
        )
    )
    orders_post = respx.post(f"{PAPER_URL}/v2/orders")

    notifier = RecordingNotifier()
    stop = threading.Event()
    original_wait = stop.wait

    def wait_then_stop(timeout: float | None = None) -> bool:
        stop.set()
        return original_wait(timeout)

    stop.wait = wait_then_stop  # type: ignore[method-assign]

    run_loop(
        _settings(tmp_path),
        _config(),
        execute=True,
        notifier=notifier,
        stop=stop,
        install_signal_handlers=False,
    )

    assert clock_route.called
    assert not orders_post.called
    assert any("started" in m for m in notifier.messages)


def test_in_force_close_window_disabled_when_zero() -> None:
    from datetime import UTC, datetime, timedelta

    from amms.clock import ClockStatus
    from amms.scheduler import _in_force_close_window

    now = datetime(2026, 5, 13, 19, 50, tzinfo=UTC)
    clock = ClockStatus(
        timestamp=now,
        is_open=True,
        next_open=now + timedelta(days=1),
        next_close=now + timedelta(minutes=10),
    )
    assert _in_force_close_window(clock, 0) is False
    assert _in_force_close_window(clock, 15) is True
    assert _in_force_close_window(clock, 5) is False


def test_announce_tick_is_silent_with_null_notifier() -> None:
    from amms.executor import TickResult

    notifier = NullNotifier()
    _announce_tick(notifier, TickResult(placed_order_ids=["x"]))  # must not raise


def test_handle_closed_emits_summary_only_once_per_day(tmp_path: Path) -> None:
    from datetime import datetime

    from amms import db
    from amms.clock import ClockStatus

    @respx.mock
    def _run() -> RecordingNotifier:
        respx.get(f"{PAPER_URL}/v2/account").mock(
            return_value=httpx.Response(
                200,
                json={
                    "equity": "100000",
                    "cash": "100000",
                    "buying_power": "100000",
                    "status": "ACTIVE",
                },
            )
        )
        respx.get(f"{PAPER_URL}/v2/positions").mock(
            return_value=httpx.Response(200, json=[])
        )
        from amms.broker import AlpacaClient

        notifier = RecordingNotifier()
        state = LoopState(last_saw_open=True)
        clock = ClockStatus(
            timestamp=datetime.fromisoformat("2026-05-13T22:00:00-04:00"),
            is_open=False,
            next_open=datetime.fromisoformat("2026-05-14T09:30:00-04:00"),
            next_close=datetime.fromisoformat("2026-05-14T16:00:00-04:00"),
        )
        with AlpacaClient("k", "s", PAPER_URL) as broker:
            conn = db.connect(tmp_path / "x.sqlite")
            db.migrate(conn)
            _handle_closed(broker, conn, notifier, clock, state)
            # Subsequent closed ticks on the same date should not re-send.
            _handle_closed(broker, conn, notifier, clock, state)
            conn.close()
        return notifier

    notifier = _run()
    summaries = [m for m in notifier.messages if "daily summary" in m]
    assert len(summaries) == 1


@respx.mock
def test_maybe_refresh_sentiment_uses_apewisdom_without_reddit_creds(
    monkeypatch,
) -> None:
    """Without REDDIT_CLIENT_ID, fall back to ApeWisdom and normalize
    mention counts into a 0..1 overlay."""
    from amms.strategy.composite import get_sentiment_overlay, set_sentiment_overlay

    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    set_sentiment_overlay({})

    payload = {
        "results": [
            {"ticker": "NVDA", "mentions": 1000},
            {"ticker": "GME", "mentions": 100},
            {"ticker": "OFFLIST", "mentions": 50},
        ]
    }
    respx.get("https://apewisdom.io/api/v1.0/filter/wallstreetbets").mock(
        return_value=httpx.Response(200, json=payload)
    )

    state = LoopState()
    # Must exceed SENTIMENT_REFRESH_SECONDS (3600) since last refresh = 0.
    _maybe_refresh_sentiment(state, watchlist={"NVDA", "GME"}, now_seconds=10_000.0)
    overlay = get_sentiment_overlay()
    # Watchlist filter excludes OFFLIST.
    assert set(overlay) == {"NVDA", "GME"}
    # 1000 mentions ≈ log10(1001)/3 ≈ 1.0 (clipped); 100 ≈ log10(101)/3 ≈ 0.67.
    assert overlay["NVDA"] == 1.0
    assert 0.5 < overlay["GME"] < 0.8
