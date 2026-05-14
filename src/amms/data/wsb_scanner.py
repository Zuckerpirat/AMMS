"""WSB Auto-Discovery: scan r/wallstreetbets and surface trending tickers.

This module is a thin orchestrator on top of ``RedditSentimentCollector``.
It pulls top posts from one or more subreddits, extracts ticker mentions,
and ranks them by mention count + average sentiment.

The scanner is *discovery only* — it produces a ranked list. Whether the
trader acts on it (manually expanding the watchlist, or feeding it into the
strategy) is a separate decision.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

from amms.features.sentiment import (
    ApeWisdomCollector,
    RedditSentimentCollector,
    extract_tickers,
    score_text,
)

logger = logging.getLogger(__name__)

DEFAULT_SUBS = ("wallstreetbets", "stocks", "pennystocks")


@dataclass(frozen=True)
class TrendingTicker:
    symbol: str
    mentions: int
    avg_sentiment: float
    bullish_posts: int
    bearish_posts: int

    @property
    def bullish_ratio(self) -> float:
        total = self.bullish_posts + self.bearish_posts
        return self.bullish_posts / total if total else 0.0

    @property
    def label(self) -> str:
        if self.avg_sentiment >= 0.2:
            return "bullish"
        if self.avg_sentiment <= -0.2:
            return "bearish"
        return "mixed"


class WSBScanner:
    """Scan WSB for trending tickers (mentions + sentiment).

    Uses ApeWisdom by default (no API key needed). Falls back to
    RedditSentimentCollector if REDDIT_CLIENT_ID is set in env.

    Usage:
        with WSBScanner() as scanner:
            top = scanner.scan(min_mentions=3, top_n=20)
    """

    def __init__(
        self,
        *,
        collector: RedditSentimentCollector | None = None,
        subreddits: tuple[str, ...] = DEFAULT_SUBS,
    ) -> None:
        self._owns_collector = collector is None
        self._collector = collector or RedditSentimentCollector()
        self._subreddits = subreddits
        # Use ApeWisdom only when no collector was injected AND no Reddit keys set.
        self._use_apewisdom = collector is None and not bool(
            __import__("os").environ.get("REDDIT_CLIENT_ID")
        )

    def close(self) -> None:
        if self._owns_collector:
            self._collector.close()

    def __enter__(self) -> WSBScanner:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def scan(
        self,
        *,
        limit_per_sub: int = 100,
        time_filter: str = "day",
        min_mentions: int = 3,
        top_n: int | None = 20,
    ) -> list[TrendingTicker]:
        """Return tickers ranked by WSB mentions.

        Uses ApeWisdom (no key needed) unless REDDIT_CLIENT_ID is set,
        in which case the full Reddit text-scraping path runs instead.
        """
        if not self._use_apewisdom:
            return self._scan_reddit(
                limit_per_sub=limit_per_sub,
                time_filter=time_filter,
                min_mentions=min_mentions,
                top_n=top_n,
            )
        return self._scan_apewisdom(min_mentions=min_mentions, top_n=top_n)

    def _scan_apewisdom(
        self,
        *,
        min_mentions: int,
        top_n: int | None,
    ) -> list[TrendingTicker]:
        with ApeWisdomCollector() as ape:
            raw = ape.fetch_trending(filter="wallstreetbets")

        results: list[TrendingTicker] = []
        for item in raw:
            sym = (item.get("ticker") or "").upper()
            if not sym:
                continue
            mentions = int(item.get("mentions") or 0)
            if mentions < min_mentions:
                continue
            results.append(
                TrendingTicker(
                    symbol=sym,
                    mentions=mentions,
                    avg_sentiment=0.0,
                    bullish_posts=0,
                    bearish_posts=0,
                )
            )

        results.sort(key=lambda t: t.mentions, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results

    def _scan_reddit(
        self,
        *,
        limit_per_sub: int,
        time_filter: str,
        min_mentions: int,
        top_n: int | None,
    ) -> list[TrendingTicker]:
        mentions: Counter[str] = Counter()
        score_sums: dict[str, float] = {}
        bullish: Counter[str] = Counter()
        bearish: Counter[str] = Counter()

        for sub in self._subreddits:
            posts = self._collector.fetch_top(
                sub, limit=limit_per_sub, time_filter=time_filter
            )
            for post in posts:
                text = " ".join(
                    str(post.get(k, "") or "") for k in ("title", "selftext")
                )
                if not text:
                    continue
                post_score = score_text(text)
                tickers_in_post = set(extract_tickers(text, watchlist=None))
                for sym in tickers_in_post:
                    mentions[sym] += 1
                    score_sums[sym] = score_sums.get(sym, 0.0) + post_score
                    if post_score >= 0.1:
                        bullish[sym] += 1
                    elif post_score <= -0.1:
                        bearish[sym] += 1

        results = [
            TrendingTicker(
                symbol=sym,
                mentions=mentions[sym],
                avg_sentiment=score_sums[sym] / mentions[sym],
                bullish_posts=bullish[sym],
                bearish_posts=bearish[sym],
            )
            for sym in mentions
            if mentions[sym] >= min_mentions
        ]
        results.sort(key=lambda t: (t.mentions, t.avg_sentiment), reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results


def format_summary(
    results: list[TrendingTicker],
    *,
    prices: dict[str, dict[str, float]] | None = None,
) -> str:
    """Render a compact text summary suitable for Telegram or stdout.

    If ``prices`` is provided ({symbol: {price, change_pct}}), the current
    price and daily change are appended per line.
    """
    if not results:
        return "WSB-Scan: keine Treffer (Mindesterwähnungen unterschritten)."
    lines = ["WSB Trending Tickers:"]
    for i, t in enumerate(results, start=1):
        line = f"{i:>2}. {t.symbol:<6} {t.mentions:>3}x"
        if prices and t.symbol in prices:
            p = prices[t.symbol]
            d_arrow = "▲" if p["change_pct"] >= 0 else "▼"
            line += f"  ${p['price']:.2f} 1d {d_arrow}{p['change_pct']:+.2f}%"
            wk = p.get("change_pct_week")
            if wk is not None and p.get("week_ago_close"):
                w_arrow = "▲" if wk >= 0 else "▼"
                line += f"  1w {w_arrow}{wk:+.2f}%"
        # Only show sentiment columns when we actually have signal there.
        if t.avg_sentiment != 0 or t.bullish_posts or t.bearish_posts:
            line += (
                f"  ({t.label}, score={t.avg_sentiment:+.2f}, "
                f"+{t.bullish_posts}/-{t.bearish_posts})"
            )
        lines.append(line)
    return "\n".join(lines)
