"""Tests for amms.analysis.signal_outcome."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from amms.analysis.signal_outcome import (
    OutcomeStats,
    SignalOutcomeReport,
    _find_outcome_price,
    compute_signal_outcomes,
    format_outcome_report,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE de_signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            mode TEXT NOT NULL,
            action TEXT NOT NULL,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            horizon TEXT,
            price REAL,
            macro_level TEXT
        )
    """)
    conn.commit()
    return conn


def _ts(days_ago: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def _insert_signal(conn, *, symbol="AAPL", mode="swing", action="buy",
                   score=60.0, confidence=0.70, price=100.0, days_ago=10):
    conn.execute(
        "INSERT INTO de_signal_history "
        "(ts, symbol, mode, action, score, confidence, horizon, price, macro_level) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (_ts(days_ago), symbol, mode, action, score, confidence, "", price, "calm"),
    )
    conn.commit()


class _FakeBar:
    def __init__(self, ts: str, close: float):
        self.ts = ts
        self.close = close


class _FakeData:
    """Returns simple bars: price rises by 0.5 per day from a base."""
    def __init__(self, bars: list[_FakeBar] | None = None):
        self._bars = bars

    def get_bars(self, symbol, *, limit=400):
        if self._bars is not None:
            return self._bars
        # Default: 400 bars, price rises from 90 → 289
        bars = []
        base = datetime.now(timezone.utc) - timedelta(days=400)
        for i in range(400):
            dt = base + timedelta(days=i)
            bars.append(_FakeBar(dt.isoformat(), 90.0 + i * 0.5))
        return bars


# ── _find_outcome_price ───────────────────────────────────────────────────────

class TestFindOutcomePrice:
    def test_finds_price_after_signal(self):
        base = datetime.now(timezone.utc) - timedelta(days=20)
        bars = [_FakeBar((base + timedelta(days=i)).isoformat(), 100.0 + i)
                for i in range(30)]
        sig_ts = (base + timedelta(days=5)).isoformat()
        p = _find_outcome_price(bars, sig_ts, lookback_days=5)
        assert p is not None
        assert p >= 110.0  # bars at day 10+

    def test_returns_latest_if_target_in_future(self):
        base = datetime.now(timezone.utc) - timedelta(days=5)
        bars = [_FakeBar((base + timedelta(days=i)).isoformat(), 50.0 + i)
                for i in range(5)]
        sig_ts = datetime.now(timezone.utc).isoformat()
        p = _find_outcome_price(bars, sig_ts, lookback_days=10)
        # Should return latest bar close
        assert p == float(bars[-1].close)

    def test_returns_none_for_empty_bars(self):
        assert _find_outcome_price([], _ts(10), 5) is None

    def test_invalid_ts_returns_none(self):
        bars = [_FakeBar(_ts(1), 100.0)]
        assert _find_outcome_price(bars, "not-a-date", 5) is None


# ── OutcomeStats ──────────────────────────────────────────────────────────────

class TestOutcomeStats:
    def test_accuracy_zero_when_no_samples(self):
        s = OutcomeStats(mode="swing", action="buy")
        assert s.accuracy_pct == 0.0

    def test_accuracy_calculation(self):
        s = OutcomeStats(mode="swing", action="buy", total=4, correct=3)
        assert s.accuracy_pct == 75.0


# ── compute_signal_outcomes ───────────────────────────────────────────────────

class TestComputeSignalOutcomes:
    def test_no_conn_returns_empty(self):
        report = compute_signal_outcomes(None, _FakeData())
        assert report.total_evaluated == 0
        assert report.stats == []

    def test_no_data_client_returns_empty(self):
        report = compute_signal_outcomes(sqlite3.connect(":memory:"), None)
        assert report.total_evaluated == 0

    def test_no_signals_returns_empty(self, db):
        report = compute_signal_outcomes(db, _FakeData())
        assert report.total_evaluated == 0

    def test_evaluates_old_buy_signal_correctly(self, db):
        # Signal at price=100, bars go up → should be correct
        _insert_signal(db, symbol="AAPL", action="buy", price=100.0, days_ago=10)
        report = compute_signal_outcomes(db, _FakeData(), lookback_days=5, min_age_days=2)
        assert report.total_evaluated == 1
        assert len(report.stats) == 1
        s = report.stats[0]
        assert s.action == "buy"
        assert s.total == 1
        # Price went up (bars trend upward) → correct
        assert s.correct == 1

    def test_evaluates_old_sell_signal(self, db):
        # Signal at price=200 (above our uptrending data range ~90-289)
        # After lookback, price ~200+5*0.5 > 200 → incorrect (price went up, not down)
        base = datetime.now(timezone.utc) - timedelta(days=400)
        bars = [_FakeBar((base + timedelta(days=i)).isoformat(), 100.0 + i * 0.1)
                for i in range(400)]
        data = _FakeData(bars=bars)
        # Signal was at price=250 (higher than all bars) 10 days ago
        _insert_signal(db, symbol="AAPL", action="sell", price=250.0, days_ago=10)
        report = compute_signal_outcomes(db, data, lookback_days=5, min_age_days=2)
        assert report.total_evaluated == 1
        s = report.stats[0]
        assert s.action == "sell"

    def test_skips_too_new_signals(self, db):
        # Signal only 1 day old, min_age=2 → should be skipped
        _insert_signal(db, symbol="AAPL", action="buy", price=100.0, days_ago=1)
        report = compute_signal_outcomes(db, _FakeData(), min_age_days=2)
        assert report.total_evaluated == 0
        assert report.total_skipped == 0  # skipped due to age filter before evaluation

    def test_skips_hold_signals(self, db):
        # Hold signals are not directional
        _insert_signal(db, symbol="AAPL", action="hold", price=100.0, days_ago=10)
        report = compute_signal_outcomes(db, _FakeData(), min_age_days=2)
        assert report.total_evaluated == 0

    def test_skips_signal_without_price(self, db):
        db.execute(
            "INSERT INTO de_signal_history "
            "(ts, symbol, mode, action, score, confidence, horizon, price, macro_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (_ts(10), "AAPL", "swing", "buy", 60.0, 0.7, "", None, "calm"),
        )
        db.commit()
        report = compute_signal_outcomes(db, _FakeData(), min_age_days=2)
        assert report.total_evaluated == 0

    def test_multiple_modes(self, db):
        _insert_signal(db, symbol="AAPL", mode="swing",  action="buy", price=100.0, days_ago=10)
        _insert_signal(db, symbol="AAPL", mode="meme",   action="buy", price=100.0, days_ago=10)
        _insert_signal(db, symbol="AAPL", mode="conservative", action="buy", price=100.0, days_ago=10)
        report = compute_signal_outcomes(db, _FakeData(), lookback_days=5, min_age_days=2)
        assert report.total_evaluated == 3
        modes = {s.mode for s in report.stats}
        assert "swing" in modes
        assert "meme" in modes
        assert "conservative" in modes

    def test_report_lookback_days_preserved(self, db):
        report = compute_signal_outcomes(db, _FakeData(), lookback_days=14)
        assert report.lookback_days == 14


# ── format_outcome_report ─────────────────────────────────────────────────────

class TestFormatOutcomeReport:
    def test_empty_report_gives_no_signals_message(self):
        report = SignalOutcomeReport()
        out = format_outcome_report(report)
        assert "No signal" in out

    def test_report_with_stats(self, db):
        _insert_signal(db, symbol="AAPL", mode="swing", action="buy", price=100.0, days_ago=10)
        report = compute_signal_outcomes(db, _FakeData(), lookback_days=5, min_age_days=2)
        out = format_outcome_report(report)
        assert "swing" in out
        assert "buy" in out
        assert "acc=" in out

    def test_report_includes_summary_line(self, db):
        _insert_signal(db, symbol="AAPL", action="buy", price=100.0, days_ago=10)
        report = compute_signal_outcomes(db, _FakeData(), lookback_days=5, min_age_days=2)
        out = format_outcome_report(report)
        assert "Evaluated" in out
