"""Tests for amms.analysis.drawdown_heatmap."""

from __future__ import annotations

import pytest

from amms.data.bars import Bar
from amms.analysis.drawdown_heatmap import DrawdownRow, DrawdownHeatmap, analyze


def _bar(sym: str, close: float, i: int = 0) -> Bar:
    return Bar(sym, "1Day", f"2026-01-{1 + i % 28:02d}T10:00:00Z",
               close, close + 1, close - 1, close, 10_000)


def _bars(sym: str, prices: list[float]) -> list[Bar]:
    return [_bar(sym, p, i) for i, p in enumerate(prices)]


class TestAnalyze:
    def test_returns_none_empty(self):
        assert analyze({}) is None

    def test_returns_none_insufficient_bars(self):
        result = analyze({"AAPL": [_bar("AAPL", 100.0)]})
        assert result is None

    def test_new_high_status(self):
        """Monotonically rising prices → new_high."""
        prices = [100.0 + i for i in range(20)]
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        assert result.rows[0].status == "new_high"

    def test_deepening_status(self):
        """Sharp drop at end → deepening."""
        prices = [100.0] * 15 + [95.0, 93.0, 91.0, 89.0, 87.0]
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        assert result.rows[0].status == "deepening"

    def test_drawdown_pct_correct(self):
        """Peak=110, current=100 → drawdown=-9.09%%."""
        prices = [100.0] * 9 + [110.0] + [100.0] * 10
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        row = result.rows[0]
        assert row.peak_price == pytest.approx(110.0)
        expected_dd = (100.0 - 110.0) / 110.0 * 100
        assert row.drawdown_pct == pytest.approx(expected_dd, abs=0.1)

    def test_max_drawdown_le_current_drawdown(self):
        """Max drawdown should be >= current drawdown (more negative or equal)."""
        prices = [100.0] * 5 + [110.0] + [80.0] + [95.0] * 5
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        row = result.rows[0]
        assert row.max_drawdown_pct <= row.drawdown_pct

    def test_recovery_pct_100_at_new_high(self):
        prices = [100.0 + i for i in range(20)]
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        assert result.rows[0].recovery_pct == pytest.approx(100.0, abs=0.1)

    def test_recovery_pct_partial(self):
        """After a dip, partial recovery gives 0<recovery<100."""
        # Peak=110, worst=80, current=95
        prices = [100.0] * 5 + [110.0] + [80.0] + [95.0]
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        row = result.rows[0]
        assert 0.0 < row.recovery_pct < 100.0

    def test_multiple_symbols(self):
        prices_good = [100.0 + i for i in range(20)]
        prices_bad = [100.0] * 10 + [90.0] * 10
        result = analyze({
            "AAPL": _bars("AAPL", prices_good),
            "TSLA": _bars("TSLA", prices_bad),
        })
        assert result is not None
        assert len(result.rows) == 2

    def test_worst_symbol_identified(self):
        prices_good = [100.0 + i for i in range(20)]
        prices_bad = [100.0] * 10 + [80.0] * 10
        result = analyze({
            "AAPL": _bars("AAPL", prices_good),
            "TSLA": _bars("TSLA", prices_bad),
        })
        assert result is not None
        assert result.worst_symbol == "TSLA"

    def test_best_symbol_at_new_high(self):
        prices_rising = [100.0 + i for i in range(20)]
        prices_flat = [100.0] * 10 + [90.0] * 10
        result = analyze({
            "AAPL": _bars("AAPL", prices_rising),
            "TSLA": _bars("TSLA", prices_flat),
        })
        assert result is not None
        assert result.best_symbol == "AAPL"

    def test_n_at_new_high(self):
        prices_a = [100.0 + i for i in range(20)]
        prices_b = [100.0 + i for i in range(20)]
        prices_c = [100.0] * 10 + [85.0] * 10
        result = analyze({
            "A": _bars("A", prices_a),
            "B": _bars("B", prices_b),
            "C": _bars("C", prices_c),
        })
        assert result is not None
        assert result.n_at_new_high == 2

    def test_n_deepening(self):
        prices_drop = [100.0] * 10 + [99.0, 97.0, 95.0, 93.0, 91.0]
        result = analyze({"AAPL": _bars("AAPL", prices_drop)})
        assert result is not None
        assert result.n_deepening >= 1

    def test_avg_drawdown_is_average(self):
        """avg_drawdown_pct should equal average of individual drawdowns."""
        prices_a = [100.0 + i for i in range(20)]  # new high → dd=0
        prices_b = [100.0] * 10 + [90.0] * 10
        result = analyze({
            "A": _bars("A", prices_a),
            "B": _bars("B", prices_b),
        })
        assert result is not None
        manual_avg = sum(r.drawdown_pct for r in result.rows) / len(result.rows)
        assert result.avg_drawdown_pct == pytest.approx(manual_avg, abs=0.01)

    def test_rows_sorted_by_drawdown(self):
        """Rows should be sorted worst drawdown first (most negative)."""
        prices_a = [100.0 + i for i in range(20)]
        prices_b = [100.0] * 10 + [80.0] * 10
        result = analyze({
            "A": _bars("A", prices_a),
            "B": _bars("B", prices_b),
        })
        assert result is not None
        dds = [r.drawdown_pct for r in result.rows]
        assert dds == sorted(dds)

    def test_lookback_respected(self):
        """With lookback=5, only last 5 bars matter."""
        # All rising for 50 bars, then big drop in last 5
        prices = [100.0 + i for i in range(50)] + [80.0, 79.0, 78.0, 77.0, 76.0]
        result = analyze({"AAPL": _bars("AAPL", prices)}, lookback=5)
        assert result is not None
        row = result.rows[0]
        # Within last 5 bars peak is 80.0, current is 76.0
        assert row.bars_used == 5

    def test_bars_since_peak(self):
        """Peak at bar 10, 10 more bars follow → bars_since_peak=10."""
        prices = [100.0] * 10 + [120.0] + [100.0] * 10
        result = analyze({"AAPL": _bars("AAPL", prices)})
        assert result is not None
        assert result.rows[0].bars_since_peak == 10
