"""ISIN lookup for US equities.

Strategy:
1. Curated static mapping for the most common tickers (zero-latency hot path).
2. Financial Times public search API for everything else.
3. In-memory cache for the bot's lifetime so each ticker is fetched at
   most once.

The FT endpoint returns rich security data including ISINs and is the
same source yfinance uses internally. No API key required. Failures
fall back to "" so the UI just omits the column for that ticker.
"""

from __future__ import annotations

import logging
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

_FT_SEARCH_URL = "https://markets.ft.com/data/searchapi/searchsecurities"

# Curated mapping. Mostly mega-caps, ETFs, and recurrent WSB favorites.
# Used as a fast first-line cache so the bot never hits the network for
# these. ISINs are immutable so this never goes stale.
_STATIC_ISINS: dict[str, str] = {
    # Mega-cap tech
    "AAPL": "US0378331005",
    "MSFT": "US5949181045",
    "GOOGL": "US02079K3059",
    "GOOG": "US02079K1079",
    "AMZN": "US0231351067",
    "META": "US30303M1027",
    "NVDA": "US67066G1040",
    "TSLA": "US88160R1014",
    "AVGO": "US11135F1012",
    "AMD": "US0079031078",
    "INTC": "US4581401001",
    "ORCL": "US68389X1054",
    "CRM": "US79466L3024",
    "ADBE": "US00724F1012",
    "NFLX": "US64110L1061",
    # Semis
    "MU": "US5951121038",
    "AMAT": "US0382221051",
    "TSM": "US8740391003",
    "QCOM": "US7475251036",
    "TXN": "US8825081040",
    "SNDK": "US80369C1053",
    # WSB favorites
    "GME": "US36467W1099",
    "AMC": "US00165C3025",
    "PLTR": "US69608A1088",
    "SOFI": "US83406F1021",
    "RIVN": "US76954A1034",
    "LCID": "US5494981039",
    "MARA": "US5657881067",
    "RIOT": "US7672921050",
    "COIN": "US19260Q1076",
    "HOOD": "US7846335016",
    "RDDT": "US7561751096",
    "POET": "CA73083A1075",
    "NBIS": "NL0015000G94",
    # ETFs
    "SPY": "US78462F1030",
    "QQQ": "US46090E1038",
    "IWM": "US4642876555",
    "VOO": "US9229083632",
    "ARKK": "US00214Q1040",
    "TQQQ": "US74347X8314",
    "SQQQ": "US74347B3839",
}


class IsinLookup:
    """Resolve tickers to ISINs.

    First checks the static table; falls back to the Financial Times
    public search API. Per-process cache prevents repeated network
    calls for the same ticker.
    """

    def __init__(
        self,
        *,
        timeout: float = 8.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._cache: dict[str, str] = dict(_STATIC_ISINS)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0 Safari/537.36"
                ),
                "Accept": "application/json",
            },
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> IsinLookup:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def lookup(self, symbols: Iterable[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for raw in symbols:
            sym = raw.upper()
            if sym in self._cache:
                out[sym] = self._cache[sym]
                continue
            isin = self._fetch_from_ft(sym)
            self._cache[sym] = isin
            out[sym] = isin
        return out

    def _fetch_from_ft(self, symbol: str) -> str:
        """Query the Financial Times search API for a single ticker."""
        try:
            resp = self._client.get(
                _FT_SEARCH_URL, params={"query": symbol}
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            logger.warning("ft isin lookup failed for %s", symbol, exc_info=True)
            return ""

        securities = (data.get("data") or {}).get("security") or []
        # Prefer exact ticker match on US/Nasdaq/NYSE listings.
        for item in securities:
            if not isinstance(item, dict):
                continue
            ticker = (item.get("symbol") or "").upper()
            isin = item.get("isin") or ""
            if ticker == symbol and isin:
                return str(isin)
        # Fallback: take the first hit with an ISIN.
        for item in securities:
            if isinstance(item, dict) and item.get("isin"):
                return str(item["isin"])
        return ""
