"""ISIN lookup via OpenFIGI with in-memory cache.

OpenFIGI is Bloomberg's free ticker-to-ISIN mapping service. No API key
is required for low-volume use. ISINs are immutable, so we cache them
in-process for the lifetime of the bot — first hit calls OpenFIGI, all
subsequent hits are instant.
"""

from __future__ import annotations

import logging
from typing import Iterable

import httpx

logger = logging.getLogger(__name__)

_OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"


class IsinLookup:
    """Resolve US-equity tickers to ISIN strings via OpenFIGI.

    Lookups are cached in a dict for the lifetime of the instance.
    Failures are also cached as empty strings so we don't hammer the
    API for unknown tickers.
    """

    def __init__(
        self,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._cache: dict[str, str] = {}
        self._timeout = timeout
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "amms/1.0",
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
        """Return {symbol: isin} for the given tickers.

        Symbols already in the cache are returned immediately. Unknown
        symbols are batched into a single OpenFIGI request. Symbols
        without a matching ISIN map to "".
        """
        symbols = [s.upper() for s in symbols]
        missing = [s for s in symbols if s not in self._cache]
        if missing:
            self._fetch(missing)
        return {s: self._cache.get(s, "") for s in symbols}

    def _fetch(self, symbols: list[str]) -> None:
        payload = [
            {"idType": "TICKER", "idValue": sym, "exchCode": "US"}
            for sym in symbols
        ]
        try:
            resp = self._client.post(_OPENFIGI_URL, json=payload)
            resp.raise_for_status()
            results = resp.json()
        except Exception:
            logger.warning("openfigi lookup failed", exc_info=True)
            # Cache misses as empty so we don't retry every scan.
            for sym in symbols:
                self._cache.setdefault(sym, "")
            return

        for sym, entry in zip(symbols, results, strict=False):
            isin = ""
            if isinstance(entry, dict):
                for item in entry.get("data") or []:
                    candidate = item.get("isin") if isinstance(item, dict) else None
                    if candidate:
                        isin = str(candidate)
                        break
            self._cache[sym] = isin
