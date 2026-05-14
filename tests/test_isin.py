from __future__ import annotations

import httpx
import respx

from amms.data.isin import IsinLookup


def test_lookup_returns_known_isin_from_static_table() -> None:
    with IsinLookup() as lookup:
        out = lookup.lookup(["NVDA", "MSFT", "AAPL"])
    assert out["NVDA"] == "US67066G1040"
    assert out["MSFT"] == "US5949181045"
    assert out["AAPL"] == "US0378331005"


def test_lookup_normalizes_lowercase_input() -> None:
    with IsinLookup() as lookup:
        out = lookup.lookup(["nvda"])
    assert out == {"NVDA": "US67066G1040"}


@respx.mock
def test_unknown_ticker_falls_back_to_ft_search() -> None:
    payload = {
        "data": {
            "security": [
                {"symbol": "FAKE", "isin": "US1234567890"},
            ]
        }
    }
    respx.get("https://markets.ft.com/data/searchapi/searchsecurities").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["FAKE"])
    assert out == {"FAKE": "US1234567890"}


@respx.mock
def test_ft_lookup_prefers_exact_ticker_match() -> None:
    payload = {
        "data": {
            "security": [
                {"symbol": "FAKEX", "isin": "US0000000000"},
                {"symbol": "FAKE", "isin": "US1111111111"},
            ]
        }
    }
    respx.get("https://markets.ft.com/data/searchapi/searchsecurities").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["FAKE"])
    assert out == {"FAKE": "US1111111111"}


@respx.mock
def test_ft_lookup_returns_empty_on_no_match() -> None:
    respx.get("https://markets.ft.com/data/searchapi/searchsecurities").mock(
        return_value=httpx.Response(200, json={"data": {"security": []}})
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["UNKNOWNX"])
    assert out == {"UNKNOWNX": ""}


@respx.mock
def test_ft_lookup_returns_empty_on_http_failure() -> None:
    respx.get("https://markets.ft.com/data/searchapi/searchsecurities").mock(
        return_value=httpx.Response(500)
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["UNKNOWNX"])
    assert out == {"UNKNOWNX": ""}


@respx.mock
def test_caches_results_so_only_one_request_per_ticker() -> None:
    route = respx.get(
        "https://markets.ft.com/data/searchapi/searchsecurities"
    ).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"security": [{"symbol": "FAKE", "isin": "US1111111111"}]}},
        )
    )
    with IsinLookup() as lookup:
        lookup.lookup(["FAKE"])
        lookup.lookup(["FAKE"])
    assert route.call_count == 1
