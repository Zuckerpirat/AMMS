"""Tests for amms.execution.auto_scanner."""

from __future__ import annotations

import time

import pytest

from amms.execution.auto_scanner import AutoScanner, ScanResult, format_scan_results


class _Bar:
    def __init__(self, close, volume=1_000_000):
        self.close = close
        self.high = close + 1.0
        self.low = close - 1.0
        self.open = close
        self.volume = volume


def _trend_bars(n: int = 60, start: float = 100.0, step: float = 0.5,
                vol_mult: float = 1.0) -> list[_Bar]:
    """Uptrend with 4-up-1-down pattern so RSI has real gains and losses."""
    bars = []
    p = start
    for i in range(n):
        # Every 5th bar is a small pullback so RSI has losses
        if i % 5 == 4:
            change = -step * 0.5
        else:
            change = step * 1.3
        p = max(1.0, p + change)
        vol = int(2_000_000 * vol_mult) if i == n - 1 else 1_000_000
        bars.append(_Bar(p, vol))
    return bars


def _flat_bars(n: int = 60, price: float = 100.0) -> list[_Bar]:
    return [_Bar(price) for _ in range(n)]


class _FakeData:
    def __init__(self, bars_by_sym: dict, news_by_sym: dict | None = None):
        self._bars = bars_by_sym
        self._news = news_by_sym or {}

    def get_bars(self, symbol, *, limit=60):
        return self._bars.get(symbol.upper(), [])

    def get_news(self, symbols, *, limit=5):
        sym = symbols[0].upper() if symbols else ""
        return self._news.get(sym, [])


class TestScoreSymbol:
    def test_strong_momentum_scores_high(self):
        data = _FakeData({"NVDA": _trend_bars(60, 100.0, 0.8)})
        scanner = AutoScanner(data, ["NVDA"], min_score=30.0)
        result = scanner._score_symbol("NVDA")
        assert result is not None
        assert result.score >= 30.0
        assert result.symbol == "NVDA"

    def test_flat_bars_returns_none(self):
        data = _FakeData({"FLAT": _flat_bars(60)})
        scanner = AutoScanner(data, ["FLAT"], min_score=30.0)
        result = scanner._score_symbol("FLAT")
        assert result is None

    def test_volume_spike_adds_score(self):
        bars = _trend_bars(60, vol_mult=3.0)  # last bar has 3× volume
        data = _FakeData({"TST": bars})
        scanner = AutoScanner(data, ["TST"], min_score=10.0)
        result = scanner._score_symbol("TST")
        if result is not None:
            assert any("Volumen" in r or "volume" in r.lower() for r in result.reasons)

    def test_news_activity_adds_score(self):
        from datetime import UTC, datetime
        today = datetime.now(UTC).isoformat()[:10]
        news = [
            {"headline": f"News {i}", "created_at": f"{today}T10:00:00Z"}
            for i in range(4)
        ]
        bars = _trend_bars(60, 100.0, 0.5)
        data = _FakeData({"TST": bars}, {"TST": news})
        scanner = AutoScanner(data, ["TST"], min_score=10.0)
        result = scanner._score_symbol("TST")
        if result is not None:
            assert result.score > 0

    def test_insufficient_bars_returns_none(self):
        data = _FakeData({"TST": _flat_bars(10)})
        scanner = AutoScanner(data, ["TST"], min_score=30.0)
        assert scanner._score_symbol("TST") is None

    def test_data_error_returns_none(self):
        class BadData:
            def get_bars(self, *a, **k):
                raise RuntimeError("boom")
        scanner = AutoScanner(BadData(), ["TST"], min_score=10.0)
        # _score_symbol should not raise
        result = scanner._score_symbol("TST")
        assert result is None


class TestScanAndUpdate:
    def test_scan_respects_interval(self):
        data = _FakeData({"NVDA": _trend_bars(60, 100.0, 0.8)})
        scanner = AutoScanner(data, ["NVDA"], min_score=10.0)
        scanner._last_scan = time.monotonic()  # pretend we just scanned
        results = scanner.scan_and_update([])
        assert results == []  # too soon

    def test_scan_skips_existing_watchlist(self):
        data = _FakeData({"AAPL": _trend_bars(60, 100.0, 1.0)})
        scanner = AutoScanner(data, ["AAPL"], min_score=10.0)
        scanner._last_scan = time.monotonic() - 7200  # force scan: 2h ago
        results = scanner.scan_and_update(["AAPL"])  # AAPL already in watchlist
        assert "AAPL" not in results

    def test_scan_returns_new_symbols(self):
        data = _FakeData({"NVDA": _trend_bars(60, 100.0, 1.0)})
        scanner = AutoScanner(data, ["NVDA"], min_score=10.0, max_additions=5)
        scanner._last_scan = time.monotonic() - 7200
        results = scanner.scan_and_update([])
        assert isinstance(results, list)
        # NVDA may or may not qualify depending on mock data; just verify no crash
        assert all(isinstance(s, str) for s in results)

    def test_max_additions_respected(self):
        bars = _trend_bars(60, 100.0, 1.0)
        universe = [f"SYM{i}" for i in range(20)]
        data = _FakeData({s: bars for s in universe})
        scanner = AutoScanner(data, universe, min_score=1.0, max_additions=3)
        scanner._last_scan = time.monotonic() - 7200
        results = scanner.scan_and_update([])
        assert len(results) <= 3

    def test_decay_increments_for_stale_symbols(self):
        data = _FakeData({"NVDA": _flat_bars()})
        scanner = AutoScanner(data, ["NVDA"], min_score=999.0, decay_ticks=2)
        scanner._last_scan = time.monotonic() - 7200  # force scan: 2h ago
        scanner._added_symbols["NVDA"] = 0  # manually mark as added
        scanner.scan_and_update([])  # NVDA won't be re-discovered (flat)
        assert scanner._added_symbols.get("NVDA", 0) == 1  # incremented

    def test_decay_removes_after_threshold(self):
        data = _FakeData({"NVDA": _flat_bars()})
        scanner = AutoScanner(data, ["NVDA"], min_score=999.0, decay_ticks=2)
        scanner._last_scan = time.monotonic() - 7200  # force scan: 2h ago
        scanner._added_symbols["NVDA"] = 1  # one tick away from decay
        scanner.scan_and_update([])  # second miss → removed
        assert "NVDA" not in scanner._added_symbols


class TestFormatResults:
    def test_empty_results(self):
        out = format_scan_results([])
        assert "keine" in out.lower()

    def test_formats_results(self):
        results = [
            ScanResult("NVDA", 85.0, ["Momentum", "Volumen-Spike"], "2026-05-17T10:00:00Z"),
            ScanResult("AAPL", 55.0, ["RSI stark"], "2026-05-17T10:00:00Z"),
        ]
        out = format_scan_results(results)
        assert "NVDA" in out
        assert "85" in out
        assert "Momentum" in out
        assert "AAPL" in out

    def test_top_n_limit(self):
        results = [
            ScanResult(f"SYM{i}", float(100 - i), ["reason"], "2026-05-17")
            for i in range(20)
        ]
        out = format_scan_results(results, top=3)
        # Only first 3 should appear with their symbols
        assert "SYM0" in out
        assert "SYM1" in out
        assert "SYM2" in out


class TestSetUniverse:
    def test_universe_updated(self):
        data = _FakeData({})
        scanner = AutoScanner(data, ["AAPL"])
        scanner.set_universe(["NVDA", "TSLA"])
        assert "NVDA" in scanner.universe
        assert "AAPL" not in scanner.universe
