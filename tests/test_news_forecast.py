"""Tests for amms.analysis.news_forecast."""

from __future__ import annotations

import json
import sqlite3

import pytest

from amms.analysis.news_forecast import (
    NewsForecast,
    _input_hash,
    _unknown,
    forecast_from_news,
    format_forecast,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _articles(n: int = 3):
    return [
        {
            "headline": f"NVDA headline {i}: Strong AI chip demand",
            "summary": f"Nvidia reports record datacenter revenue in Q{i+1} driven by AI workloads.",
            "url": f"https://example.com/{i}",
            "created_at": "2026-05-17T10:00:00Z",
            "symbols": ["NVDA"],
        }
        for i in range(n)
    ]


class TestUnknownFallback:
    def test_no_articles_returns_unknown(self):
        fc = forecast_from_news("NVDA", [])
        assert fc.direction == "unknown"
        assert fc.confidence == 0.0
        assert fc.article_count == 0

    def test_no_api_key_returns_unknown(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fc = forecast_from_news("NVDA", _articles(3))
        assert fc.direction == "unknown"
        assert fc.confidence == 0.0
        assert "api" in fc.summary.lower() or "key" in fc.summary.lower() or "verfügbar" in fc.summary.lower()

    def test_unknown_has_zero_magnitude(self):
        fc = _unknown("AAPL", 0, "test")
        assert fc.magnitude == 0.0


class TestInputHash:
    def test_same_articles_same_hash(self):
        a = _articles(3)
        assert _input_hash(a) == _input_hash(a)

    def test_different_count_different_hash(self):
        assert _input_hash(_articles(2)) != _input_hash(_articles(4))

    def test_hash_length(self):
        assert len(_input_hash(_articles(1))) == 16


class TestCaching:
    def test_cached_result_loaded(self, monkeypatch):
        conn = _conn()
        from amms.analysis.news_forecast import _ensure_table, _save_cache, _load_cache
        _ensure_table(conn)
        fc = NewsForecast(
            symbol="NVDA",
            direction="up",
            magnitude=7.5,
            horizon="1w",
            confidence=0.8,
            catalysts=["AI demand", "Earnings beat"],
            risks=["Supply chain"],
            summary="NVDA likely up 7-8% on AI momentum",
            article_count=3,
        )
        arts = _articles(3)
        digest = _input_hash(arts)
        _save_cache(conn, "NVDA", "2026-05-17", digest, fc)

        loaded = _load_cache(conn, "NVDA", "2026-05-17", digest)
        assert loaded is not None
        assert loaded.direction == "up"
        assert loaded.magnitude == pytest.approx(7.5)
        assert loaded.cached is True

    def test_wrong_hash_misses(self):
        conn = _conn()
        from amms.analysis.news_forecast import _ensure_table, _load_cache
        _ensure_table(conn)
        result = _load_cache(conn, "NVDA", "2026-05-17", "wronghash12345a")
        assert result is None

    def test_wrong_date_misses(self):
        conn = _conn()
        from amms.analysis.news_forecast import _ensure_table, _save_cache, _load_cache
        _ensure_table(conn)
        fc = NewsForecast("X", "sideways", 0.0, "1d", 0.2, [], [], "ok", 1)
        arts = _articles(1)
        digest = _input_hash(arts)
        _save_cache(conn, "X", "2026-05-16", digest, fc)
        assert _load_cache(conn, "X", "2026-05-17", digest) is None


class TestFormatForecast:
    def _make(self, direction="up", magnitude=5.0, confidence=0.75):
        return NewsForecast(
            symbol="NVDA",
            direction=direction,
            magnitude=magnitude,
            horizon="1w",
            confidence=confidence,
            catalysts=["AI chip demand surge", "Blackwell GPU sold out"],
            risks=["China export controls"],
            summary="NVDA likely up 5% within 1 week due to AI demand",
            article_count=4,
        )

    def test_format_up(self):
        out = format_forecast(self._make("up", 5.0))
        assert "📈" in out or "▲" in out
        assert "NVDA" in out
        assert "5.0" in out
        assert "1 Woche" in out or "1w" in out
        assert "AI chip demand" in out
        assert "China" in out

    def test_format_down(self):
        out = format_forecast(self._make("down", 3.5))
        assert "📉" in out or "▼" in out
        assert "3.5" in out

    def test_format_sideways(self):
        out = format_forecast(self._make("sideways", 0.0))
        assert "➡" in out or "Seitwärts" in out

    def test_format_unknown(self):
        out = format_forecast(self._make("unknown", 0.0, 0.0))
        assert isinstance(out, str) and len(out) > 10

    def test_format_no_catalysts_no_crash(self):
        fc = NewsForecast("X", "up", 2.0, "1d", 0.5, [], [], "ok", 1)
        out = format_forecast(fc)
        assert isinstance(out, str)

    def test_format_disclaimer_present(self):
        out = format_forecast(self._make())
        assert "kein Anlageberater" in out.lower() or "ki-prognose" in out.lower()

    def test_format_cached_label(self):
        fc = NewsForecast("X", "up", 3.0, "1w", 0.6, [], [], "ok", 2, cached=True)
        out = format_forecast(fc)
        assert "cache" in out.lower() or "zwischen" in out.lower()

    def test_format_confidence_display(self):
        fc = self._make(confidence=0.75)
        out = format_forecast(fc)
        assert "75%" in out or "0.75" in out

    def test_horizon_1d_displayed(self):
        fc = NewsForecast("X", "up", 3.0, "1d", 0.5, [], [], "ok", 2)
        out = format_forecast(fc)
        assert "1 Tag" in out or "1d" in out

    def test_horizon_1m_displayed(self):
        fc = NewsForecast("X", "down", 10.0, "1m", 0.4, [], [], "ok", 2)
        out = format_forecast(fc)
        assert "1 Monat" in out or "1m" in out


class TestNewsForecastFields:
    def test_dataclass_fields(self):
        fc = NewsForecast(
            symbol="TEST",
            direction="sideways",
            magnitude=0.5,
            horizon="1d",
            confidence=0.3,
            catalysts=[],
            risks=[],
            summary="flat",
            article_count=1,
        )
        assert fc.symbol == "TEST"
        assert fc.cached is False
        assert fc.article_count == 1
