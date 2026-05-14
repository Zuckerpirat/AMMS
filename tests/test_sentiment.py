from __future__ import annotations

import httpx
import respx

from amms.features.sentiment import (
    RedditSentimentCollector,
    extract_tickers,
    score_text,
)


def test_score_text_positive() -> None:
    assert score_text("AAPL to the moon 🚀") > 0.5


def test_score_text_negative() -> None:
    assert score_text("AAPL is a scam and will crash") < -0.5


def test_score_text_neutral_for_unknown_words() -> None:
    assert score_text("the company is fine") == 0.0


def test_score_text_empty_returns_zero() -> None:
    assert score_text("") == 0.0
    assert score_text(None) == 0.0  # type: ignore[arg-type]


def test_extract_tickers_filters_stopwords() -> None:
    tickers = extract_tickers("AAPL is great. A WSB CEO loves IPO names like NVDA.")
    assert "AAPL" in tickers
    assert "NVDA" in tickers
    assert "WSB" not in tickers
    assert "CEO" not in tickers


def test_extract_tickers_respects_watchlist() -> None:
    out = extract_tickers(
        "AAPL is great but TSLA is too.",
        watchlist={"AAPL"},
    )
    assert out == ["AAPL"]


@respx.mock
def test_collector_aggregate_counts_mentions_and_averages_scores() -> None:
    listing = {
        "data": {
            "children": [
                {"data": {"title": "AAPL to the moon 🚀", "selftext": ""}},
                {"data": {"title": "AAPL crash", "selftext": "AAPL scam"}},
                {"data": {"title": "TSLA breakout", "selftext": ""}},
            ]
        }
    }
    respx.get("https://www.reddit.com/r/wallstreetbets/hot.json").mock(
        return_value=httpx.Response(200, json=listing)
    )

    coll = RedditSentimentCollector(client_id="", client_secret="")  # no auth → public API
    out = coll.aggregate(
        subreddits=("wallstreetbets",),
        watchlist={"AAPL", "TSLA"},
        limit_per_sub=10,
    )
    assert set(out) == {"AAPL", "TSLA"}
    # AAPL appears 3 times (once in post 1 title, twice in post 2 title+body).
    assert out["AAPL"].mentions == 3
    assert out["TSLA"].mentions == 1
    assert -1.0 <= out["AAPL"].avg_score <= 1.0


@respx.mock
def test_collector_returns_empty_on_http_failure() -> None:
    respx.get("https://www.reddit.com/r/wallstreetbets/hot.json").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )
    coll = RedditSentimentCollector(client_id="", client_secret="")
    out = coll.aggregate(subreddits=("wallstreetbets",))
    assert out == {}
