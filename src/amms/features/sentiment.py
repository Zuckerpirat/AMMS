"""Lightweight built-in sentiment scorer + Reddit collector.

The scorer is a small bag-of-words; not as good as VADER but adds zero
external dependencies. Real VADER (``vaderSentiment`` package) is a drop-in
replacement; swap ``score_text`` for vader's compound score when ready.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Tiny lexicon. Each word maps to a polarity in [-1, 1]. Pulled from common
# WSB/penny-stock idioms; intentionally conservative.
_POSITIVE = {
    "moon": 1.0, "rocket": 1.0, "rip": 0.8, "rally": 0.8, "breakout": 0.7,
    "buy": 0.5, "long": 0.5, "bullish": 0.9, "calls": 0.4, "yolo": 0.6,
    "squeeze": 0.7, "rip!": 1.0, "🚀": 1.0, "🌝": 1.0, "💎": 0.8, "🙌": 0.5,
    "diamond": 0.6, "hold": 0.3, "winning": 0.7, "winner": 0.6, "alpha": 0.5,
    "tendies": 0.7, "moass": 0.9, "pumping": 0.6, "surge": 0.7,
}
_NEGATIVE = {
    "dump": -0.8, "crash": -1.0, "bagholder": -0.6, "puts": -0.5,
    "bearish": -0.9, "short": -0.5, "scam": -1.0, "fraud": -1.0,
    "rugpull": -1.0, "ratio": -0.4, "rip": -0.0,  # ambiguous, leave to context
    "loss": -0.5, "loser": -0.5, "halt": -0.5, "halted": -0.6,
    "delist": -0.9, "bk": -0.9, "bankrupt": -1.0, "drilling": -0.6,
    "tank": -0.7, "tanking": -0.8, "bleed": -0.5, "selloff": -0.7, "💀": -0.7,
}
_LEXICON = {**_POSITIVE, **_NEGATIVE}

_TOKEN_RE = re.compile(r"[A-Za-z!\$🚀🌝💎🙌💀]+")
_TICKER_RE = re.compile(r"\$?\b([A-Z]{1,5})\b")
# Known noise — common acronyms / slang on WSB that look like tickers but
# almost never refer to the actual listed company. Conservative on purpose:
# anything plausibly traded (AI, EV, AGI...) is left out so we don't filter
# real signal.
_STOP_TICKERS = {
    "A", "I", "DD", "WSB", "USA", "CEO", "CFO", "IPO", "ETF", "YOLO",
    "USD", "EUR", "CAD", "GBP", "JPY", "CNY",
    "ATH", "ATL", "IMO", "IRL", "TLDR", "FOMO", "FUD", "FYI", "EOD", "EOM",
    "BTW", "OMG", "WTF", "LOL", "IIRC", "AFAIK", "NSFW", "ELI5",
    "UK", "EU", "DOJ", "SEC", "FED", "FOMC", "CPI", "GDP", "PPI",
    "MOASS", "DRS", "APE", "OG", "GG", "EOY", "YTD", "MTD", "QTD",
    "PR", "ER", "FY", "Q1", "Q2", "Q3", "Q4", "AH", "PM", "AM",
    "NYSE", "OTC", "ITM", "OTM", "ATM",
    "PUT", "CALL", "BUY", "SELL", "HOLD", "LONG", "SHORT",
    "USDT", "USDC", "COVID",
}


def score_text(text: str) -> float:
    """Score text in [-1, 1]. Positive = bullish, negative = bearish."""
    if not text:
        return 0.0
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    matched = [_LEXICON[t] for t in tokens if t in _LEXICON]
    if not matched:
        return 0.0
    raw = sum(matched) / max(len(matched), 1)
    return max(-1.0, min(1.0, raw))


def extract_tickers(text: str, *, watchlist: set[str] | None = None) -> list[str]:
    """Find ticker-shaped tokens in text. With ``watchlist`` set, only return
    those that the bot actually cares about (cuts down on noise dramatically).
    """
    out: list[str] = []
    for match in _TICKER_RE.findall(text or ""):
        sym = match.upper()
        if sym in _STOP_TICKERS:
            continue
        if watchlist is not None and sym not in watchlist:
            continue
        out.append(sym)
    return out


@dataclass(frozen=True)
class SentimentResult:
    symbol: str
    mentions: int
    avg_score: float


class RedditSentimentCollector:
    """Pulls recent hot/new posts from a subreddit and aggregates per-ticker
    sentiment. Uses Reddit's JSON listing endpoint with the user's PRAW-style
    credentials passed via env (we do OAuth client_credentials, not a logged-in
    user). Designed to fail silently and return an empty dict on any error.
    """

    DEFAULT_SUBS = ("wallstreetbets", "pennystocks", "stocks")

    def __init__(
        self,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self._ua = user_agent or os.environ.get("REDDIT_USER_AGENT", "amms/0.1")
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout, headers={"User-Agent": self._ua}
        )
        self._token: str | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RedditSentimentCollector:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _auth(self) -> str | None:
        if self._token is not None:
            return self._token
        if not self._client_id or not self._client_secret:
            return None
        try:
            resp = self._client.post(
                "https://www.reddit.com/api/v1/access_token",
                data={"grant_type": "client_credentials"},
                auth=(self._client_id, self._client_secret),
            )
            resp.raise_for_status()
            self._token = resp.json().get("access_token")
        except Exception:
            logger.warning("reddit auth failed", exc_info=True)
            return None
        return self._token

    def fetch_hot(self, subreddit: str, *, limit: int = 50) -> list[dict]:
        return self._fetch_listing(subreddit, listing="hot", limit=limit)

    def fetch_top(
        self, subreddit: str, *, limit: int = 100, time_filter: str = "day"
    ) -> list[dict]:
        """Fetch top posts of the given time window (day/week/month/year/all)."""
        return self._fetch_listing(
            subreddit, listing="top", limit=limit, time_filter=time_filter
        )

    def _fetch_listing(
        self,
        subreddit: str,
        *,
        listing: str,
        limit: int,
        time_filter: str | None = None,
    ) -> list[dict]:
        token = self._auth()
        if token:
            url = f"https://oauth.reddit.com/r/{subreddit}/{listing}"
            headers = {"Authorization": f"bearer {token}"}
        else:
            url = f"https://www.reddit.com/r/{subreddit}/{listing}.json"
            headers = {}
        params: dict[str, str | int] = {"limit": limit}
        if time_filter is not None:
            params["t"] = time_filter
        try:
            resp = self._client.get(url, params=params, headers=headers)
            resp.raise_for_status()
        except Exception:
            logger.warning("reddit fetch %s failed", listing, exc_info=True)
            return []
        data = resp.json().get("data", {}).get("children", [])
        return [child.get("data", {}) for child in data]

    def aggregate(
        self,
        subreddits: tuple[str, ...] | None = None,
        *,
        watchlist: set[str] | None = None,
        limit_per_sub: int = 50,
    ) -> dict[str, SentimentResult]:
        subreddits = subreddits or self.DEFAULT_SUBS
        mentions: Counter[str] = Counter()
        score_sums: dict[str, float] = {}
        for sub in subreddits:
            for post in self.fetch_hot(sub, limit=limit_per_sub):
                text = " ".join(
                    str(post.get(k, "") or "") for k in ("title", "selftext")
                )
                if not text:
                    continue
                post_score = score_text(text)
                for sym in extract_tickers(text, watchlist=watchlist):
                    mentions[sym] += 1
                    score_sums[sym] = score_sums.get(sym, 0.0) + post_score
        return {
            sym: SentimentResult(
                symbol=sym,
                mentions=mentions[sym],
                avg_score=score_sums[sym] / mentions[sym],
            )
            for sym in mentions
        }
