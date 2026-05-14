from __future__ import annotations

import httpx
import respx

from amms.data.edgar import (
    COMPANY_FACTS_URL,
    SUBMISSIONS_URL,
    TICKER_INDEX_URL,
    EdgarClient,
)


_TICKER_INDEX = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
}


@respx.mock
def test_resolve_cik() -> None:
    respx.get(TICKER_INDEX_URL).mock(return_value=httpx.Response(200, json=_TICKER_INDEX))
    with EdgarClient() as c:
        assert c.resolve_cik("aapl") == 320193
        assert c.resolve_cik("MSFT") == 789019
        assert c.resolve_cik("ZZZZ") is None


@respx.mock
def test_recent_filings_filters_by_form() -> None:
    respx.get(TICKER_INDEX_URL).mock(return_value=httpx.Response(200, json=_TICKER_INDEX))
    respx.get(SUBMISSIONS_URL.format(cik=320193)).mock(
        return_value=httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "accessionNumber": ["a-1", "a-2", "a-3"],
                        "form": ["10-K", "10-Q", "10-Q"],
                        "filingDate": ["2025-09-01", "2025-06-30", "2025-03-31"],
                    }
                }
            },
        )
    )
    with EdgarClient() as c:
        all_filings = c.recent_filings("AAPL")
        assert len(all_filings) == 3
        only_10q = c.recent_filings("AAPL", form="10-Q")
        assert {f.accession_number for f in only_10q} == {"a-2", "a-3"}


@respx.mock
def test_latest_value_picks_most_recent_end_date() -> None:
    respx.get(TICKER_INDEX_URL).mock(return_value=httpx.Response(200, json=_TICKER_INDEX))
    respx.get(COMPANY_FACTS_URL.format(cik=320193)).mock(
        return_value=httpx.Response(
            200,
            json={
                "facts": {
                    "us-gaap": {
                        "CommonStockSharesOutstanding": {
                            "units": {
                                "shares": [
                                    {"end": "2024-09-30", "val": 15_000_000_000},
                                    {"end": "2025-09-30", "val": 14_500_000_000},
                                    {"end": "2025-06-30", "val": 14_700_000_000},
                                ]
                            }
                        }
                    }
                }
            },
        )
    )
    with EdgarClient() as c:
        v = c.latest_value("AAPL", "CommonStockSharesOutstanding")
    assert v == 14_500_000_000


@respx.mock
def test_latest_value_returns_none_when_missing() -> None:
    respx.get(TICKER_INDEX_URL).mock(return_value=httpx.Response(200, json=_TICKER_INDEX))
    respx.get(COMPANY_FACTS_URL.format(cik=320193)).mock(
        return_value=httpx.Response(200, json={"facts": {"us-gaap": {}}})
    )
    with EdgarClient() as c:
        assert c.latest_value("AAPL", "Bogus") is None
