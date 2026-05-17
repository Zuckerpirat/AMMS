"""Tests for DecisionEngineStrategy — the bridge between the Decision Engine
and the Strategy/Backtest protocol.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from amms.data.bars import Bar
from amms.strategy import DecisionEngineStrategy, build_strategy


def _make_bars(n: int, *, base: float = 100.0, step: float = 0.5) -> list[Bar]:
    """Generate `n` daily bars with a gentle uptrend."""
    bars = []
    for i in range(n):
        close = base + i * step
        open_ = close - step * 0.3
        high = close + step * 0.5
        low = open_ - step * 0.3
        bars.append(
            Bar(
                symbol="TEST",
                timeframe="1Day",
                ts=f"2025-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}T05:00:00Z",
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=10_000 + i * 100,
            )
        )
    return bars


# ── Lookback & insufficient bars ─────────────────────────────────────────────

def test_lookback_is_200() -> None:
    s = DecisionEngineStrategy()
    assert s.lookback == 200


def test_hold_when_bars_below_minimum() -> None:
    s = DecisionEngineStrategy()
    bars = _make_bars(50)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"
    assert "need 120" in sig.reason


def test_hold_when_empty_bars() -> None:
    s = DecisionEngineStrategy()
    sig = s.evaluate("TEST", [])
    assert sig.kind == "hold"


# ── Signal shape with enough bars ────────────────────────────────────────────

def test_returns_valid_signal_kind_with_200_bars() -> None:
    s = DecisionEngineStrategy(min_score=0.0, min_confidence=0.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind in {"buy", "sell", "hold"}
    assert sig.symbol == "TEST"
    assert isinstance(sig.price, float) and sig.price > 0
    assert isinstance(sig.score, float)


def test_signal_reason_contains_action_and_score() -> None:
    s = DecisionEngineStrategy(min_score=0.0, min_confidence=0.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    # Reason should always contain DE prefix or threshold-skip text
    assert "DE " in sig.reason or "score" in sig.reason or "need" in sig.reason


# ── Threshold filtering ───────────────────────────────────────────────────────

def test_high_min_score_forces_hold() -> None:
    """A min_score of 999 should force hold on any realistic signal."""
    s = DecisionEngineStrategy(min_score=999.0, min_confidence=0.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"
    assert "below min" in sig.reason


def test_high_min_confidence_forces_hold() -> None:
    """A min_confidence of 1.0 should force hold (confidence never reaches 1)."""
    s = DecisionEngineStrategy(min_score=0.0, min_confidence=1.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    # The decision engine's min_confidence=1.0 will cause analyze() to return
    # None or a report marked as low-confidence hold.
    assert sig.kind == "hold"


# ── allow_strong_only ─────────────────────────────────────────────────────────

def test_allow_strong_only_blocks_regular_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    """When allow_strong_only=True, a 'buy' (not strong_buy) is blocked."""
    from amms.engine import decision as de_module

    class _FakeReport:
        action = "buy"
        composite_score = 60.0
        confidence = 0.75
        risk_blocked = False
        risk_reason = ""
        verdict = "bullish"

    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: _FakeReport())

    s = DecisionEngineStrategy(min_score=10.0, allow_strong_only=True)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"
    assert "not strong" in sig.reason


def test_allow_strong_only_passes_strong_buy(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.engine import decision as de_module

    class _FakeReport:
        action = "strong_buy"
        composite_score = 80.0
        confidence = 0.85
        risk_blocked = False
        risk_reason = ""
        verdict = "strongly bullish"

    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: _FakeReport())

    s = DecisionEngineStrategy(min_score=10.0, allow_strong_only=True)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "buy"


# ── Risk gate ─────────────────────────────────────────────────────────────────

def test_risk_blocked_returns_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.engine import decision as de_module

    class _FakeReport:
        action = "buy"
        composite_score = 70.0
        confidence = 0.80
        risk_blocked = True
        risk_reason = "killswitch armed"
        verdict = "blocked"

    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: _FakeReport())

    s = DecisionEngineStrategy(min_score=10.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"
    assert "risk gate" in sig.reason
    assert "killswitch" in sig.reason


# ── Sell signal path ──────────────────────────────────────────────────────────

def test_sell_signal_mapped_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.engine import decision as de_module

    class _FakeReport:
        action = "strong_sell"
        composite_score = -72.0
        confidence = 0.78
        risk_blocked = False
        risk_reason = ""
        verdict = "bearish"

    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: _FakeReport())

    s = DecisionEngineStrategy(min_score=10.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "sell"
    assert sig.score < 0


# ── Hold action ───────────────────────────────────────────────────────────────

def test_hold_action_returned_as_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.engine import decision as de_module

    class _FakeReport:
        action = "hold"
        composite_score = 10.0
        confidence = 0.55
        risk_blocked = False
        risk_reason = ""
        verdict = "neutral"

    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: _FakeReport())

    s = DecisionEngineStrategy(min_score=5.0)
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"


# ── analyze() returning None ──────────────────────────────────────────────────

def test_none_from_analyze_returns_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    from amms.engine import decision as de_module
    monkeypatch.setattr(de_module, "analyze", lambda *a, **kw: None)

    s = DecisionEngineStrategy()
    bars = _make_bars(200)
    sig = s.evaluate("TEST", bars)
    assert sig.kind == "hold"
    assert "no result" in sig.reason


# ── Registry integration ──────────────────────────────────────────────────────

def test_registered_under_decision_engine_name() -> None:
    s = build_strategy("decision_engine", {})
    assert isinstance(s, DecisionEngineStrategy)
    assert s.name == "decision_engine"


def test_registered_strategy_accepts_params() -> None:
    s = build_strategy("decision_engine", {"min_score": 50.0, "allow_strong_only": True})
    assert isinstance(s, DecisionEngineStrategy)
    assert s.min_score == 50.0
    assert s.allow_strong_only is True
