"""In-memory bar data cache with TTL.

Wraps any data client that has get_bars(symbol, limit=N) and caches
results per (symbol, limit) for a configurable TTL (default 5 minutes).

Avoids redundant API calls when multiple Telegram commands run in quick
succession for the same symbol (e.g., /tradeplan calls /desizer logic
and both fetch bars).

Thread-safe via a simple lock.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _CacheEntry:
    bars: list
    fetched_at: float  # time.monotonic()


class BarCache:
    """Caching wrapper around any bar data client.

    Usage:
        raw_client = AlpacaDataClient(...)
        cached = BarCache(raw_client, ttl_seconds=300)
        bars = cached.get_bars("AAPL", limit=200)  # cached on second call
    """

    def __init__(self, client, ttl_seconds: int = 300):
        self._client = client
        self.ttl = max(10, int(ttl_seconds))
        self._cache: dict[tuple[str, int], _CacheEntry] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get_bars(self, symbol: str, *, limit: int = 200) -> list:
        key = (symbol.upper(), limit)
        now = time.monotonic()

        with self._lock:
            entry = self._cache.get(key)
            if entry is not None and (now - entry.fetched_at) < self.ttl:
                self._hits += 1
                return entry.bars

        # Cache miss or expired — fetch fresh
        self._misses += 1
        bars = self._client.get_bars(symbol, limit=limit)

        with self._lock:
            self._cache[key] = _CacheEntry(bars=bars, fetched_at=time.monotonic())

        return bars

    def invalidate(self, symbol: str | None = None) -> int:
        """Invalidate cache for one symbol or all symbols.

        Returns number of entries removed.
        """
        with self._lock:
            if symbol is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            sym_upper = symbol.upper()
            to_del = [k for k in self._cache if k[0] == sym_upper]
            for k in to_del:
                del self._cache[k]
            return len(to_del)

    def stats(self) -> dict:
        with self._lock:
            n = len(self._cache)
        total = self._hits + self._misses
        hit_rate = self._hits / total * 100.0 if total > 0 else 0.0
        return {
            "entries": n,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 1),
            "ttl_seconds": self.ttl,
        }

    # Proxy all other methods to the underlying client
    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)
