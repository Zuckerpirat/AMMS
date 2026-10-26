"""Yahoo Finance market-data client (no API key required).

Minimal client mimicking the Alpaca MarketDataClient interface used by
the AutoTrader (``get_bars(symbol, limit=…)`` returning a list of
``Bar`` objects). Uses the ``yfinance`` package which handles Yahoo's
cookie/crumb handshake automatically.

Intended for the standalone simulate mode so the bot runs without any
Alpaca account.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from amms.data.bars import Bar

logger = logging.getLogger(__name__)

# Map our timeframe tokens to yfinance's (interval, period).
_TF_MAP = {
    "1Min": ("1m", "5d"),
    "5Min": ("5m", "60d"),
    "15Min": ("15m", "60d"),
    "30Min": ("30m", "60d"),
    "1Hour": ("60m", "730d"),
    "1Day": ("1d", "2y"),
}


class YahooDataClient:
    """Pulls OHLCV bars from Yahoo Finance via the yfinance library."""

    def __init__(self, timeframe: str = "15Min"):
        if timeframe not in _TF_MAP:
            raise ValueError(f"unsupported timeframe: {timeframe}")
        self.timeframe = timeframe
        self._interval, self._period = _TF_MAP[timeframe]

    def close(self) -> None:
        return None

    def __enter__(self) -> YahooDataClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_bars(self, symbol: str, limit: int = 200) -> list[Bar]:
        """Return the most recent ``limit`` OHLCV bars for ``symbol``."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed — add it to pyproject.toml")
            return []

        symbol_u = symbol.upper()
        try:
            hist = yf.Ticker(symbol_u).history(
                period=self._period, interval=self._interval, auto_adjust=False
            )
        except Exception as exc:
            logger.warning("yahoo fetch failed for %s: %s", symbol_u, exc)
            return []

        if hist is None or hist.empty:
            return []

        bars: list[Bar] = []
        for ts, row in hist.iterrows():
            o, h, l, c, v = row["Open"], row["High"], row["Low"], row["Close"], row["Volume"]
            try:
                iso = ts.to_pydatetime().astimezone(timezone.utc).isoformat()
            except Exception:
                iso = datetime.now(timezone.utc).isoformat()
            bars.append(Bar(
                symbol=symbol_u, timeframe=self.timeframe, ts=iso,
                open=float(o), high=float(h), low=float(l),
                close=float(c), volume=float(v),
            ))
        return bars[-limit:] if len(bars) > limit else bars

    def get_price(self, symbol: str) -> float | None:
        bars = self.get_bars(symbol, limit=1)
        return bars[-1].close if bars else None
