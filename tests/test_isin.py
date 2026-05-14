from __future__ import annotations

import httpx
import respx

from amms.data.isin import IsinLookup


@respx.mock
def test_lookup_returns_isins_from_openfigi() -> None:
    payload = [
        {"data": [{"isin": "US67066G1040", "ticker": "NVDA"}]},
        {"data": [{"isin": "US5949181045", "ticker": "MSFT"}]},
    ]
    respx.post("https://api.openfigi.com/v3/mapping").mock(
        return_value=httpx.Response(200, json=payload)
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["NVDA", "MSFT"])
    assert out == {"NVDA": "US67066G1040", "MSFT": "US5949181045"}


@respx.mock
def test_lookup_caches_results_between_calls() -> None:
    route = respx.post("https://api.openfigi.com/v3/mapping").mock(
        return_value=httpx.Response(200, json=[{"data": [{"isin": "US67066G1040"}]}])
    )
    with IsinLookup() as lookup:
        lookup.lookup(["NVDA"])
        lookup.lookup(["NVDA"])  # cached → no second request
    assert route.call_count == 1


@respx.mock
def test_lookup_handles_missing_data_field() -> None:
    respx.post("https://api.openfigi.com/v3/mapping").mock(
        return_value=httpx.Response(200, json=[{"warning": "No match."}])
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["ZZZZ"])
    assert out == {"ZZZZ": ""}


@respx.mock
def test_lookup_returns_empty_on_http_failure() -> None:
    respx.post("https://api.openfigi.com/v3/mapping").mock(
        return_value=httpx.Response(500)
    )
    with IsinLookup() as lookup:
        out = lookup.lookup(["NVDA"])
    assert out == {"NVDA": ""}
