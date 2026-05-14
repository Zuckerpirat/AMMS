"""Thin SEC EDGAR client for fundamentals + filings.

EDGAR is free, no API key required, but mandates a descriptive
``User-Agent`` header identifying the requester. We send
``amms/0.1 (contact: see project repo)``; override via
``EDGAR_USER_AGENT`` if you need a different identifier.

Only the minimum surface the bot needs:
    - resolve ticker → CIK
    - latest filings of a given form type
    - the most recent reported value of a basic concept (e.g. shares
      outstanding) from XBRL company-facts.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_UA = "amms/0.1 (https://github.com/zuckerpirat/amms)"
TICKER_INDEX_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


@dataclass(frozen=True)
class Filing:
    accession_number: str
    form: str
    filed: str  # ISO date


class EdgarClient:
    def __init__(
        self,
        *,
        user_agent: str | None = None,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._ua = user_agent or os.environ.get("EDGAR_USER_AGENT", _DEFAULT_UA)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={"User-Agent": self._ua, "Accept": "application/json"},
        )
        self._ticker_to_cik: dict[str, int] | None = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request(self, url: str) -> Any:
        resp = self._client.get(url)
        resp.raise_for_status()
        return resp.json()

    def resolve_cik(self, ticker: str) -> int | None:
        """Map ``ticker`` to its CIK using the SEC's ticker index."""
        if self._ticker_to_cik is None:
            data = self._request(TICKER_INDEX_URL)
            self._ticker_to_cik = {
                row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
            }
        return self._ticker_to_cik.get(ticker.upper())

    def recent_filings(
        self, ticker: str, *, form: str | None = None, limit: int = 10
    ) -> list[Filing]:
        cik = self.resolve_cik(ticker)
        if cik is None:
            return []
        data = self._request(SUBMISSIONS_URL.format(cik=cik))
        recent = data.get("filings", {}).get("recent", {})
        accs = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        rows: list[Filing] = []
        for acc, f, d in zip(accs, forms, dates, strict=False):
            if form is not None and f != form:
                continue
            rows.append(Filing(accession_number=acc, form=f, filed=d))
            if len(rows) >= limit:
                break
        return rows

    def latest_value(self, ticker: str, concept: str) -> float | None:
        """Return the most recent reported value for ``concept`` (e.g.
        ``CommonStockSharesOutstanding``). Picks the latest end date across all
        units. Returns None if the concept isn't present.
        """
        cik = self.resolve_cik(ticker)
        if cik is None:
            return None
        try:
            data = self._request(COMPANY_FACTS_URL.format(cik=cik))
        except httpx.HTTPStatusError:
            return None
        facts = data.get("facts", {}).get("us-gaap", {}).get(concept, {})
        if not facts:
            return None
        units = facts.get("units", {})
        latest: tuple[str, float] | None = None
        for _unit, rows in units.items():
            for row in rows:
                end = row.get("end")
                value = row.get("val")
                if end is None or value is None:
                    continue
                if latest is None or end > latest[0]:
                    latest = (end, float(value))
        return latest[1] if latest else None
