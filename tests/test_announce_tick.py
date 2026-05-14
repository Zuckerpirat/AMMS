from __future__ import annotations

from amms.executor import TickResult
from amms.notifier import NullNotifier
from amms.scheduler import _announce_tick
from amms.strategy import Signal


class _Recorder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def _signal(symbol: str, kind: str, *, price: float = 100.0, score: float = 0.0,
            reason: str = "test") -> Signal:
    return Signal(symbol=symbol, kind=kind, reason=reason, price=price, score=score)


def test_never_mode_sends_nothing_even_with_orders() -> None:
    rec = _Recorder()
    result = TickResult(
        signals=[_signal("NVDA", "buy")],
        placed_order_ids=["o1"],
    )
    _announce_tick(rec, result, mode="never", execute=True)
    assert rec.messages == []


def test_orders_only_silent_in_dry_run() -> None:
    rec = _Recorder()
    result = TickResult(signals=[_signal("NVDA", "buy")])  # signal, no order
    _announce_tick(rec, result, mode="orders_only", execute=False)
    assert rec.messages == []


def test_orders_only_speaks_when_orders_placed() -> None:
    rec = _Recorder()
    result = TickResult(
        signals=[_signal("NVDA", "buy")], placed_order_ids=["o1"]
    )
    _announce_tick(rec, result, mode="orders_only", execute=True)
    assert len(rec.messages) == 1
    assert "1 buy" in rec.messages[0]


def test_decisions_mode_speaks_in_dry_run_when_signal_exists() -> None:
    rec = _Recorder()
    result = TickResult(signals=[_signal("NVDA", "buy", price=487.30, score=0.62)])
    _announce_tick(rec, result, mode="decisions", execute=False)
    assert len(rec.messages) == 1
    msg = rec.messages[0]
    assert "dry-run" in msg
    assert "NVDA" in msg
    assert "BUY" in msg


def test_decisions_mode_silent_when_nothing_happens() -> None:
    rec = _Recorder()
    result = TickResult()  # no signals, no orders
    _announce_tick(rec, result, mode="decisions", execute=False)
    assert rec.messages == []


def test_always_mode_sends_even_when_nothing_happens() -> None:
    rec = _Recorder()
    result = TickResult()
    _announce_tick(rec, result, mode="always", execute=False)
    assert len(rec.messages) == 1
    assert "no signals" in rec.messages[0]


def test_blocked_signals_appear_in_message() -> None:
    rec = _Recorder()
    result = TickResult(
        signals=[_signal("NVDA", "buy")],
        blocked=[("AAPL", "already long this symbol"),
                 ("AMD", "universe filter: price < $1")],
    )
    _announce_tick(rec, result, mode="decisions", execute=False)
    msg = rec.messages[0]
    assert "AAPL" in msg
    assert "already long" in msg
    assert "AMD" in msg


def test_buys_are_ranked_by_score_descending() -> None:
    rec = _Recorder()
    result = TickResult(signals=[
        _signal("LOW", "buy", score=0.1),
        _signal("HIGH", "buy", score=0.9),
        _signal("MID", "buy", score=0.5),
    ])
    _announce_tick(rec, result, mode="decisions", execute=False)
    msg = rec.messages[0]
    # Highest score appears before lowest score in the message body
    assert msg.index("HIGH") < msg.index("MID") < msg.index("LOW")


def test_message_caps_to_max_lines() -> None:
    rec = _Recorder()
    # 20 buys — should be capped to MAX_DECISION_LINES (8)
    result = TickResult(signals=[
        _signal(f"S{i:02d}", "buy", score=float(i)) for i in range(20)
    ])
    _announce_tick(rec, result, mode="decisions", execute=False)
    msg = rec.messages[0]
    # header + 8 decision lines = 9 lines max
    assert len(msg.split("\n")) <= 9


def test_null_notifier_short_circuits() -> None:
    null = NullNotifier()
    result = TickResult(
        signals=[_signal("NVDA", "buy")], placed_order_ids=["o1"]
    )
    # must not raise
    _announce_tick(null, result, mode="always", execute=True)


def test_invalid_tick_notify_value_rejected_at_config_load() -> None:
    import pytest

    from amms.config import ConfigError, SchedulerConfig

    with pytest.raises(ConfigError):
        SchedulerConfig(tick_notify="loud")
