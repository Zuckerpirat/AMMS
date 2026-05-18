from __future__ import annotations

from unittest.mock import patch

from amms.dashboard import indices


def _yahoo_payload(price: float = 4500.0, prev: float = 4400.0) -> dict:
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "regularMarketPrice": price,
                        "chartPreviousClose": prev,
                    },
                    "indicators": {
                        "quote": [
                            {"close": [4400.0, 4450.0, 4480.0, 4500.0]}
                        ]
                    },
                }
            ]
        }
    }


def setup_function(_):
    indices.clear_cache()


def test_unknown_index_returns_error_quote() -> None:
    q = indices.get_index("does_not_exist")
    assert q.error
    assert q.price == 0.0


def test_fetch_parses_yahoo_payload() -> None:
    with patch.object(indices, "_fetch_yahoo", return_value=_yahoo_payload()):
        q = indices.get_index("sp500")
    assert q.symbol == "^GSPC"
    assert q.price == 4500.0
    assert q.day_change_abs == 100.0
    assert round(q.day_change_pct, 2) == round(100.0 / 4400.0 * 100, 2)
    assert q.sparkline_points
    assert q.error == ""


def test_cache_avoids_second_fetch() -> None:
    call_count = {"n": 0}

    def fake_fetch(symbol: str):
        call_count["n"] += 1
        return _yahoo_payload(price=4500.0 + call_count["n"])

    with patch.object(indices, "_fetch_yahoo", side_effect=fake_fetch):
        first = indices.get_index("sp500")
        second = indices.get_index("sp500")
    assert call_count["n"] == 1
    assert first.price == second.price


def test_fetch_failure_returns_stale_when_available() -> None:
    with patch.object(indices, "_fetch_yahoo", return_value=_yahoo_payload()):
        fresh = indices.get_index("sp500", ttl=0.0)
    # Force expiry, next call should hit the network and fail.
    with patch.object(indices, "_fetch_yahoo", side_effect=RuntimeError("boom")):
        stale = indices.get_index("sp500", ttl=0.0)
    assert stale.is_stale
    assert stale.price == fresh.price


def test_fetch_failure_with_no_cache_returns_error_quote() -> None:
    with patch.object(indices, "_fetch_yahoo", side_effect=RuntimeError("nope")):
        q = indices.get_index("dax")
    assert q.error
    assert q.price == 0.0
    assert not q.is_stale
