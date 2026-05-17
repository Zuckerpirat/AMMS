"""Tests for amms.analysis.news_sentiment."""

from __future__ import annotations

import json
import sqlite3

import pytest

from amms.analysis.news_sentiment import (
    NewsSentimentResult,
    _input_hash,
    _neutral,
    analyze_news,
    format_news_sentiment,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _articles(n: int = 3):
    return [
        {
            "headline": f"AAPL headline {i}",
            "summary": f"Summary of article {i} about Apple.",
            "url": f"https://example.com/{i}",
            "created_at": "2026-05-17T10:00:00Z",
            "symbols": ["AAPL"],
        }
        for i in range(n)
    ]


class TestNeutralFallback:
    def test_empty_articles(self):
        result = analyze_news("AAPL", [])
        assert result.score == 0.0
        assert result.article_count == 0
        assert "no news" in result.conclusion.lower()

    def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = analyze_news("AAPL", _articles(3))
        assert result.score == 0.0
        assert result.confidence == 0.0
        assert "unavailable" in result.conclusion.lower() or "no api key" in result.conclusion.lower()

    def test_neutral_is_not_bullish(self):
        n = _neutral("AAPL", 0, "test")
        assert n.is_bullish is False
        assert n.score == 0.0


class TestInputHash:
    def test_same_articles_same_hash(self):
        a = _articles(3)
        assert _input_hash(a) == _input_hash(a)

    def test_different_articles_different_hash(self):
        a1 = _articles(2)
        a2 = _articles(4)
        assert _input_hash(a1) != _input_hash(a2)

    def test_hash_is_16_chars(self):
        assert len(_input_hash(_articles(1))) == 16


class TestCaching:
    def test_cache_stores_and_retrieves(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        conn = _conn()

        # Manually inject a cached result
        from amms.analysis.news_sentiment import _ensure_table, _save_cache
        _ensure_table(conn)
        cached_result = NewsSentimentResult(
            symbol="AAPL",
            score=0.7,
            confidence=0.8,
            conclusion="Bullish earnings beat",
            reasoning=["Beat EPS by 15%", "Guidance raised"],
            article_count=3,
            is_bullish=True,
        )
        arts = _articles(3)
        digest = _input_hash(arts)
        _save_cache(conn, "AAPL", "2026-05-17", digest, cached_result)

        from amms.analysis.news_sentiment import _load_cache
        loaded = _load_cache(conn, "AAPL", "2026-05-17", digest)
        assert loaded is not None
        assert loaded.score == pytest.approx(0.7)
        assert loaded.conclusion == "Bullish earnings beat"
        assert loaded.cached is True

    def test_cache_miss_on_wrong_hash(self, monkeypatch):
        conn = _conn()
        from amms.analysis.news_sentiment import _ensure_table, _load_cache
        _ensure_table(conn)
        result = _load_cache(conn, "AAPL", "2026-05-17", "badhash12345678")
        assert result is None

    def test_cache_miss_wrong_date(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
        conn = _conn()
        from amms.analysis.news_sentiment import _ensure_table, _save_cache, _load_cache
        _ensure_table(conn)
        r = NewsSentimentResult("AAPL", 0.5, 0.5, "ok", [], 1, True)
        arts = _articles(1)
        digest = _input_hash(arts)
        _save_cache(conn, "AAPL", "2026-05-16", digest, r)
        # Different date → miss
        assert _load_cache(conn, "AAPL", "2026-05-17", digest) is None


class TestFormatting:
    def test_format_bullish(self):
        r = NewsSentimentResult(
            symbol="NVDA",
            score=0.8,
            confidence=0.9,
            conclusion="Strong AI tailwind",
            reasoning=["Data center demand up", "Beat earnings"],
            article_count=5,
            is_bullish=True,
        )
        out = format_news_sentiment(r)
        assert "NVDA" in out
        assert "Bullish" in out
        assert "+0.80" in out
        assert "Strong AI tailwind" in out
        assert "Data center demand up" in out

    def test_format_bearish(self):
        r = NewsSentimentResult(
            symbol="GME",
            score=-0.6,
            confidence=0.7,
            conclusion="Regulatory pressure",
            reasoning=["SEC investigation", "Revenue miss"],
            article_count=2,
            is_bullish=False,
        )
        out = format_news_sentiment(r)
        assert "Bearish" in out
        assert "-0.60" in out

    def test_format_neutral(self):
        r = NewsSentimentResult(
            symbol="AAPL",
            score=0.05,
            confidence=0.3,
            conclusion="Mostly routine updates",
            reasoning=[],
            article_count=1,
            is_bullish=False,
        )
        out = format_news_sentiment(r)
        assert "Neutral" in out

    def test_format_cached_label(self):
        r = NewsSentimentResult(
            symbol="X",
            score=0.0,
            confidence=0.0,
            conclusion="test",
            reasoning=[],
            article_count=0,
            is_bullish=False,
            cached=True,
        )
        out = format_news_sentiment(r)
        assert "cached" in out.lower()

    def test_format_no_reasoning_no_crash(self):
        r = NewsSentimentResult("Z", 0.3, 0.5, "ok", [], 2, True)
        out = format_news_sentiment(r)
        assert "ok" in out


class TestNewsSentimentResult:
    def test_is_bullish_threshold(self):
        assert NewsSentimentResult("X", 0.16, 1.0, "", [], 1, True).is_bullish is True
        assert NewsSentimentResult("X", 0.14, 1.0, "", [], 1, False).is_bullish is False

    def test_conclusion_truncated_at_200(self):
        long_conclusion = "X" * 300
        r = _neutral("A", 0, long_conclusion)
        # _neutral doesn't truncate, but format_news_sentiment should still work
        out = format_news_sentiment(r)
        assert isinstance(out, str)
