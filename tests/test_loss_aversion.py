"""Tests for amms.analysis.loss_aversion."""

from __future__ import annotations

import sqlite3

import pytest

from amms.analysis.loss_aversion import LossAversionReport, compute


def _make_db(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE trades (
            pnl_pct REAL, entered_at TEXT, closed_at TEXT, status TEXT
        )
    """)
    conn.executemany("INSERT INTO trades VALUES (?, ?, ?, 'closed')", rows)
    conn.commit()
    return conn


def _ts(day: int, hour: int, minute: int = 0) -> str:
    return f"2024-03-{day:02d} {hour:02d}:{minute:02d}:00"


def _biased_rows() -> list[tuple]:
    """Losers held 4× longer than winners — strong disposition effect."""
    rows = []
    # Winners: held 10 minutes
    for i in range(15):
        rows.append((2.0, _ts(1 + i % 27, 10, 0), _ts(1 + i % 27, 10, 10)))
    # Losers: held 60 minutes
    for i in range(15):
        rows.append((-1.5, _ts(1 + i % 27, 11, 0), _ts(1 + i % 27, 12, 0)))
    return rows


def _balanced_rows() -> list[tuple]:
    """Equal hold times for winners and losers."""
    rows = []
    for i in range(15):
        rows.append((2.0, _ts(1 + i % 27, 10, 0), _ts(1 + i % 27, 10, 30)))
    for i in range(15):
        rows.append((-1.5, _ts(1 + i % 27, 11, 0), _ts(1 + i % 27, 11, 30)))
    return rows


def _high_loss_rows() -> list[tuple]:
    """Avg loss 3× avg win — high loss aversion."""
    rows = []
    for i in range(10):
        rows.append((1.0, _ts(1, 10, 0), _ts(1, 10, 30)))
    for i in range(10):
        rows.append((-3.0, _ts(2, 10, 0), _ts(2, 10, 30)))
    return rows


class TestEdgeCases:
    def test_returns_none_empty_db(self):
        conn = _make_db([])
        assert compute(conn) is None

    def test_returns_none_too_few(self):
        rows = [(1.0, _ts(1, 10), _ts(1, 10, 30)) for _ in range(4)]
        conn = _make_db(rows)
        assert compute(conn) is None

    def test_returns_none_only_winners(self):
        rows = [(1.0, _ts(1, 10, 0), _ts(1, 10, 30)) for _ in range(10)]
        conn = _make_db(rows)
        assert compute(conn) is None

    def test_returns_none_only_losers(self):
        rows = [(-1.0, _ts(1, 10, 0), _ts(1, 10, 30)) for _ in range(10)]
        conn = _make_db(rows)
        assert compute(conn) is None

    def test_returns_result(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert isinstance(result, LossAversionReport)


class TestDispositionEffect:
    def test_biased_hold_ratio_gt_one(self):
        conn = _make_db(_biased_rows())
        result = compute(conn)
        assert result is not None
        assert result.hold_ratio > 1.0

    def test_biased_disposition_detected(self):
        conn = _make_db(_biased_rows())
        result = compute(conn)
        assert result is not None
        assert result.disposition_effect is True
        assert result.disposition_strength in ("strong", "moderate")

    def test_balanced_no_disposition(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        # Hold ratio should be ≈ 1.0
        assert result.hold_ratio == pytest.approx(1.0, abs=0.1)
        assert result.disposition_strength == "none"

    def test_disposition_strength_valid(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.disposition_strength in ("strong", "moderate", "mild", "none")

    def test_hold_times_positive(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.median_hold_winner_min >= 0
        assert result.median_hold_loser_min >= 0


class TestLossAversion:
    def test_high_loss_multiplier(self):
        conn = _make_db(_high_loss_rows())
        result = compute(conn)
        assert result is not None
        assert result.loss_multiplier > 2.0

    def test_high_loss_aversion_detected(self):
        conn = _make_db(_high_loss_rows())
        result = compute(conn)
        assert result is not None
        assert result.loss_aversion_level in ("excessive", "high")

    def test_balanced_loss_aversion_normal(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        # avg win 2.0 / avg loss 1.5 → mult ≈ 0.75 → "low" or "normal"
        assert result.loss_aversion_level in ("low", "normal")

    def test_loss_aversion_level_valid(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.loss_aversion_level in ("excessive", "high", "normal", "low")

    def test_avg_win_positive(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.avg_win_pct > 0

    def test_avg_loss_positive(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.avg_loss_pct > 0


class TestPrematureExit:
    def test_premature_exit_rate_in_range(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert 0.0 <= result.premature_exit_rate <= 100.0

    def test_avg_winning_trade_pct_positive(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert result.avg_winning_trade_pct > 0


class TestCounts:
    def test_counts_correct(self):
        conn = _make_db(_biased_rows())
        result = compute(conn)
        assert result is not None
        assert result.n_winners == 15
        assert result.n_losers == 15
        assert result.n_trades == 30


class TestVerdict:
    def test_verdict_present(self):
        conn = _make_db(_balanced_rows())
        result = compute(conn)
        assert result is not None
        assert len(result.verdict) > 10

    def test_biased_verdict_mentions_disposition(self):
        conn = _make_db(_biased_rows())
        result = compute(conn)
        assert result is not None
        assert "disposition" in result.verdict.lower() or "bias" in result.verdict.lower()
