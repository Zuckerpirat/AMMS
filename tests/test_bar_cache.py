"""Tests for amms.data.bar_cache."""

from __future__ import annotations

import time

import pytest

from amms.data.bar_cache import BarCache


class _FakeClient:
    """Counting fake client — tracks how many times get_bars is called."""

    def __init__(self, bars=None):
        self.calls = 0
        self._bars = bars or [object() for _ in range(10)]
        self.extra_attr = "hello"

    def get_bars(self, symbol, *, limit=200):
        self.calls += 1
        return self._bars


class TestBarCache:
    def test_first_call_hits_client(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        assert client.calls == 1

    def test_second_call_uses_cache(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("AAPL", limit=100)
        assert client.calls == 1

    def test_different_limit_is_separate_cache_key(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("AAPL", limit=200)
        assert client.calls == 2

    def test_different_symbol_is_separate_cache_key(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("MSFT", limit=100)
        assert client.calls == 2

    def test_symbol_normalized_to_upper(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("aapl", limit=100)
        cache.get_bars("AAPL", limit=100)
        assert client.calls == 1

    def test_ttl_expiry_refetches(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        # Manually manipulate the cache entry to simulate TTL expiry
        cache.get_bars("AAPL", limit=100)
        with cache._lock:
            cache._cache[("AAPL", 100)].fetched_at -= 61  # expired
        cache.get_bars("AAPL", limit=100)
        assert client.calls == 2

    def test_returns_same_bars_object_from_cache(self):
        sentinel = [object()]
        client = _FakeClient(bars=sentinel)
        cache = BarCache(client, ttl_seconds=60)
        r1 = cache.get_bars("AAPL", limit=100)
        r2 = cache.get_bars("AAPL", limit=100)
        assert r1 is r2

    def test_invalidate_single_symbol(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("MSFT", limit=100)
        removed = cache.invalidate("AAPL")
        assert removed == 1
        cache.get_bars("AAPL", limit=100)
        assert client.calls == 3  # AAPL refetched

    def test_invalidate_all(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("MSFT", limit=100)
        removed = cache.invalidate()
        assert removed == 2
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("MSFT", limit=100)
        assert client.calls == 4

    def test_stats_tracks_hits_and_misses(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)  # miss
        cache.get_bars("AAPL", limit=100)  # hit
        cache.get_bars("AAPL", limit=100)  # hit
        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == pytest.approx(66.7, abs=0.2)

    def test_stats_entries_count(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        cache.get_bars("AAPL", limit=100)
        cache.get_bars("MSFT", limit=200)
        assert cache.stats()["entries"] == 2

    def test_proxies_other_attributes(self):
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)
        assert cache.extra_attr == "hello"

    def test_minimum_ttl_enforced(self):
        cache = BarCache(_FakeClient(), ttl_seconds=5)
        assert cache.ttl == 10  # min is 10

    def test_thread_safe_concurrent_calls(self):
        import threading
        client = _FakeClient()
        cache = BarCache(client, ttl_seconds=60)

        def fetch():
            cache.get_bars("AAPL", limit=100)

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # At most 1 extra call due to race on first fetch; subsequent all cached
        assert client.calls <= 2
