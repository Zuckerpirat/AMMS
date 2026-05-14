from __future__ import annotations

import httpx
import respx

from amms.data.wsb_scanner import WSBScanner, format_summary


class _FakeCollector:
    """In-memory stand-in for RedditSentimentCollector. Returns canned posts."""

    def __init__(self, posts_by_sub: dict[str, list[dict]]) -> None:
        self._posts_by_sub = posts_by_sub
        self.calls: list[tuple[str, str]] = []

    def fetch_top(
        self, subreddit: str, *, limit: int = 100, time_filter: str = "day"
    ) -> list[dict]:
        self.calls.append((subreddit, time_filter))
        return self._posts_by_sub.get(subreddit, [])

    def close(self) -> None:
        pass


def _post(title: str, selftext: str = "") -> dict:
    return {"title": title, "selftext": selftext}


def test_scan_ranks_by_mentions_then_sentiment() -> None:
    posts = {
        "wallstreetbets": [
            _post("$NVDA to the moon rocket 🚀", "bullish breakout"),
            _post("NVDA calls printing", "diamond hands"),
            _post("NVDA squeeze incoming"),
            _post("$GME bagholder dump", "crash bearish"),
            _post("GME tanking"),
            _post("PLTR rally", "winner"),
        ],
    }
    collector = _FakeCollector(posts)
    scanner = WSBScanner(collector=collector, subreddits=("wallstreetbets",))
    results = scanner.scan(min_mentions=2, top_n=10)

    symbols = [r.symbol for r in results]
    assert symbols[0] == "NVDA"  # 3 mentions, very bullish
    assert "GME" in symbols  # 2 mentions, bearish
    assert "PLTR" not in symbols  # only 1 mention — below threshold


def test_min_mentions_filter_drops_one_offs() -> None:
    posts = {
        "wallstreetbets": [
            _post("AAPL is fine"),
            _post("MSFT is fine"),
            _post("GOOG is fine"),
            _post("NVDA rocket"),
            _post("NVDA moon"),
            _post("NVDA squeeze"),
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = scanner.scan(min_mentions=3, top_n=10)
    assert {r.symbol for r in results} == {"NVDA"}


def test_stop_tickers_are_filtered() -> None:
    posts = {
        "wallstreetbets": [
            _post("YOLO $USD CEO IPO ETF SEC"),
            _post("YOLO $USD CEO IPO ETF SEC again"),
            _post("YOLO $USD CEO IPO ETF SEC third time"),
            _post("NVDA rocket"),
            _post("NVDA moon"),
            _post("NVDA squeeze"),
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = scanner.scan(min_mentions=2, top_n=10)
    symbols = {r.symbol for r in results}
    assert "NVDA" in symbols
    assert "USD" not in symbols
    assert "CEO" not in symbols
    assert "YOLO" not in symbols
    assert "ETF" not in symbols


def test_label_classifies_sentiment_buckets() -> None:
    posts = {
        "wallstreetbets": [
            _post("NVDA moon rocket bullish 🚀", "rally breakout"),
            _post("NVDA calls tendies winning"),
            _post("NVDA squeeze diamond hands"),
            _post("GME crash dump bearish bagholder"),
            _post("GME tanking selloff"),
            _post("GME bankrupt rugpull"),
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = {r.symbol: r for r in scanner.scan(min_mentions=2)}
    assert results["NVDA"].label == "bullish"
    assert results["GME"].label == "bearish"


def test_format_summary_handles_empty() -> None:
    assert "keine Treffer" in format_summary([])


def test_format_summary_renders_rankings() -> None:
    posts = {
        "wallstreetbets": [
            _post("NVDA rocket"),
            _post("NVDA moon"),
            _post("NVDA squeeze"),
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = scanner.scan(min_mentions=2)
    text = format_summary(results)
    assert "NVDA" in text
    assert "WSB Trending Tickers" in text


def test_scanner_visits_each_subreddit_once() -> None:
    collector = _FakeCollector({})
    scanner = WSBScanner(
        collector=collector, subreddits=("wallstreetbets", "stocks", "pennystocks")
    )
    scanner.scan(top_n=5)
    assert {sub for sub, _ in collector.calls} == {
        "wallstreetbets",
        "stocks",
        "pennystocks",
    }


def test_top_n_limits_result_length() -> None:
    posts = {
        "wallstreetbets": [
            _post(f"{sym} rocket moon")
            for sym in ["AAA", "BBB", "CCC", "DDD", "EEE"]
            for _ in range(3)  # 3 mentions each
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = scanner.scan(min_mentions=2, top_n=2)
    assert len(results) == 2


@respx.mock
def test_scan_uses_apewisdom_when_no_collector_and_no_reddit_creds(
    monkeypatch,
) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    payload = {
        "results": [
            {"ticker": "GME", "mentions": 120, "upvotes": 500},
            {"ticker": "NVDA", "mentions": 80, "upvotes": 300},
            {"ticker": "RARE", "mentions": 1, "upvotes": 5},
        ]
    }
    respx.get("https://apewisdom.io/api/v1.0/filter/wallstreetbets").mock(
        return_value=httpx.Response(200, json=payload)
    )
    scanner = WSBScanner()  # no collector → uses ApeWisdom
    results = scanner.scan(min_mentions=2, top_n=10)
    symbols = [r.symbol for r in results]
    assert symbols[0] == "GME"
    assert "NVDA" in symbols
    assert "RARE" not in symbols  # below min_mentions


def test_format_summary_renders_prices_when_provided() -> None:
    posts = {
        "wallstreetbets": [
            _post("NVDA rocket"),
            _post("NVDA moon"),
            _post("NVDA squeeze"),
        ],
    }
    scanner = WSBScanner(
        collector=_FakeCollector(posts), subreddits=("wallstreetbets",)
    )
    results = scanner.scan(min_mentions=2)
    prices = {"NVDA": {"price": 875.12, "prev_close": 850.0, "change_pct": 2.96}}
    text = format_summary(results, prices=prices)
    assert "$875.12" in text
    assert "+2.96%" in text


@respx.mock
def test_apewisdom_failure_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    respx.get("https://apewisdom.io/api/v1.0/filter/wallstreetbets").mock(
        return_value=httpx.Response(500)
    )
    scanner = WSBScanner()
    results = scanner.scan(min_mentions=1)
    assert results == []
