"""Broad Market Scanner — scannt den gesamten NASDAQ/NYSE eigenständig.

Statt nur einer vordefinierten Watchlist scannt dieser Scanner:
  1. Alle aktiven US-Aktien auf NASDAQ, NYSE, ARCA (~5000 Symbole)
  2. Findet Mover: Aktien mit starken Kursveränderungen + Volumen heute
  3. Findet News-Katalysatoren: Aktien mit Breaking News
  4. Findet Penny Stocks mit Pump-Mustern (Preis $0.50–$10)
  5. Kombiniert mit WSB Social-Daten

Läuft stündlich. Alle gefundenen Kandidaten werden an den AutoScanner
übergeben, der sie nach Score filtert und zur Watchlist hinzufügt.
"""

from __future__ import annotations

import logging
import re
import time
import threading
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Bekannte Symbole die nie gehandelt werden sollen (ETFs, Indizes, etc.)
_BLOCKLIST = {
    "SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "VXX", "UVXY",
    "SQQQ", "TQQQ", "SPXU", "SPXL", "LABD", "LABU", "TNA", "TZA",
}

# Ticker-Regex für News-Extraktion
_TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")

# Häufige englische Wörter die wie Ticker aussehen (False positives)
_COMMON_WORDS = {
    "A", "I", "IN", "ON", "TO", "AT", "OF", "OR", "IT", "IS", "AS",
    "BE", "BY", "DO", "GO", "IF", "MY", "NO", "SO", "UP", "US", "WE",
    "AND", "ARE", "BUT", "CAN", "FOR", "HAS", "HIM", "HIS", "HOW",
    "NEW", "NOT", "NOW", "OUT", "OUR", "THE", "TOO", "TWO", "WAS",
    "WHO", "WHY", "YOU", "ALL", "ANY", "DUE", "END", "FEW", "GET",
    "GOT", "HAD", "HER", "HIGH", "ITS", "LET", "LOW", "MAN", "MAY",
    "OLD", "OWN", "SAY", "SET", "SHE", "USE", "WAY", "YET", "ALSO",
    "BACK", "BEEN", "BOTH", "CAME", "COME", "DOWN", "EACH", "EVEN",
    "FROM", "GIVE", "HAVE", "HERE", "JUST", "KNOW", "LIKE", "LONG",
    "MADE", "MAKE", "MORE", "MOST", "MUCH", "MUST", "NAME", "NEXT",
    "ONLY", "OVER", "PART", "SAID", "SAME", "SAYS", "SEEN", "SHOW",
    "SOME", "SUCH", "TAKE", "THAN", "THAT", "THEM", "THEN", "THEY",
    "THIS", "THUS", "TIME", "UPON", "USED", "VERY", "WELL", "WENT",
    "WERE", "WHAT", "WHEN", "WITH", "YEAR", "YOUR",
    "CEO", "CFO", "COO", "IPO", "ETF", "SEC", "FDA", "FED", "GDP",
    "EPS", "P&L", "ROI", "APR", "YTD", "TTM", "ESG", "AI", "ML",
    "NYSE", "NASDAQ", "USA", "USD", "EUR", "GBP",
}


@dataclass
class MarketMover:
    symbol: str
    price: float
    change_pct: float      # heute in %
    volume_ratio: float    # vs. durchschnitt (>1 = erhöht)
    catalysts: list[str]   # was das Signal auslöst
    score: float = 0.0


class BroadMarketScanner:
    """Scannt den gesamten US-Markt nach Handelskandidaten.

    Parameters
    ----------
    data_client:
        MarketDataClient mit get_assets(), get_snapshots_bulk(), get_broad_news()
    wsb_client:
        Optional WSBScanner für Social-Daten
    min_change_pct:
        Mindest-Tagesrendite für Mover (default: 3%)
    min_volume_ratio:
        Mindest-Volumen vs. Durchschnitt (default: 1.5×)
    max_price:
        Maximaler Kurs — None = kein Limit, 10.0 = nur Penny Stocks
    penny_mode:
        True = bevorzuge Aktien unter $10 mit starkem Momentum
    scan_interval:
        Sekunden zwischen zwei Scans (default: 3600 = 1 Stunde)
    """

    def __init__(
        self,
        data_client,
        *,
        wsb_client=None,
        min_change_pct: float = 3.0,
        min_volume_ratio: float = 1.5,
        max_price: float | None = None,
        penny_mode: bool = True,
        scan_interval: int = 3600,
    ):
        self._data = data_client
        self._wsb = wsb_client
        self.min_change_pct = min_change_pct
        self.min_volume_ratio = min_volume_ratio
        self.max_price = max_price
        self.penny_mode = penny_mode
        self.scan_interval = scan_interval

        self._lock = threading.Lock()
        self._last_scan: float = 0.0
        self._last_results: list[MarketMover] = []

    def scan(self, *, force: bool = False) -> list[MarketMover]:
        """Führe Markt-Scan durch. Gibt [] zurück wenn Intervall noch nicht erreicht."""
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_scan < self.scan_interval:
                return list(self._last_results)
            self._last_scan = now

        results = self._do_scan()
        with self._lock:
            self._last_results = results
        return results

    def last_results(self) -> list[MarketMover]:
        with self._lock:
            return list(self._last_results)

    # ── Private ──────────────────────────────────────────────────────────

    def _do_scan(self) -> list[MarketMover]:
        logger.info("BroadMarketScanner: starting full market scan")

        # 1. WSB Social Data (schnell, parallel nutzbar)
        wsb_mentions: dict[str, int] = {}
        if self._wsb is not None:
            try:
                trending = self._wsb.scan(min_mentions=30, top_n=50)
                wsb_mentions = {t.symbol: t.mentions for t in trending}
                logger.info("BroadMarketScanner: WSB %d trending symbols", len(wsb_mentions))
            except Exception as exc:
                logger.debug("WSB scan failed: %s", exc)

        # 2. News-Katalysatoren — tickers aus Breaking News extrahieren
        news_tickers: dict[str, list[str]] = {}  # symbol → headline list
        try:
            articles = self._data.get_broad_news(limit=100)
            for article in articles:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                # Tickers aus article.symbols (direkt von Alpaca getaggt)
                for sym in article.get("symbols", []):
                    sym = sym.upper()
                    if sym not in _BLOCKLIST and sym not in _COMMON_WORDS:
                        news_tickers.setdefault(sym, [])
                        if headline not in news_tickers[sym]:
                            news_tickers[sym].append(headline[:80])
                # Tickers aus Headline extrahieren (Backup)
                for match in _TICKER_RE.finditer(f"{headline} {summary}"):
                    sym = match.group(1)
                    if len(sym) >= 2 and sym not in _BLOCKLIST and sym not in _COMMON_WORDS:
                        news_tickers.setdefault(sym, [])
            logger.info("BroadMarketScanner: %d tickers found in news", len(news_tickers))
        except Exception as exc:
            logger.debug("News scan failed: %s", exc)

        # 3. Alle aktiven Assets holen (gecacht 6h)
        all_assets = []
        try:
            all_assets = self._data.get_assets()
            logger.info("BroadMarketScanner: %d tradeable assets", len(all_assets))
        except Exception as exc:
            logger.warning("Asset list failed: %s", exc)

        all_symbols = [a["symbol"] for a in all_assets if a["symbol"] not in _BLOCKLIST]
        # News- und WSB-Symbole die nicht in der Asset-Liste sind hinzufügen
        extra = set(news_tickers.keys()) | set(wsb_mentions.keys())
        for sym in extra:
            if sym not in all_symbols and sym not in _BLOCKLIST:
                all_symbols.append(sym)

        if not all_symbols:
            logger.warning("BroadMarketScanner: no symbols to scan")
            return []

        # 4. Bulk Snapshots — Preis + Tagesveränderung für alle Symbole
        logger.info("BroadMarketScanner: fetching snapshots for %d symbols", len(all_symbols))
        snapshots = self._data.get_snapshots_bulk(all_symbols, batch_size=200)
        logger.info("BroadMarketScanner: got %d snapshots", len(snapshots))

        # 5. Kandidaten filtern und bewerten
        candidates: list[MarketMover] = []
        for sym, snap in snapshots.items():
            sym = sym.upper()
            if sym in _BLOCKLIST:
                continue

            daily_bar = snap.get("dailyBar") or {}
            prev_daily = snap.get("prevDailyBar") or {}
            latest_trade = snap.get("latestTrade") or {}

            price = float(latest_trade.get("p") or daily_bar.get("c") or 0.0)
            if price <= 0:
                continue

            # Preis-Filter
            if self.max_price and price > self.max_price:
                continue
            if price < 0.50:  # delisted / zu billig
                continue

            # Tages-Rendite
            prev_close = float(prev_daily.get("c") or 0.0)
            today_close = float(daily_bar.get("c") or price)
            if prev_close > 0:
                change_pct = (today_close - prev_close) / prev_close * 100.0
            else:
                change_pct = 0.0

            # Volumen-Verhältnis (heute vs. gestern als Proxy)
            vol_today = float(daily_bar.get("v") or 0.0)
            vol_prev = float(prev_daily.get("v") or 0.0)
            vol_ratio = (vol_today / vol_prev) if vol_prev > 0 else 1.0

            # Score berechnen
            score, catalysts = self._score(
                sym, price, change_pct, vol_ratio,
                wsb_mentions.get(sym, 0),
                news_tickers.get(sym, []),
            )

            if score >= 20.0:
                candidates.append(MarketMover(
                    symbol=sym,
                    price=round(price, 4),
                    change_pct=round(change_pct, 2),
                    volume_ratio=round(vol_ratio, 2),
                    catalysts=catalysts,
                    score=round(score, 1),
                ))

        # Nach Score sortieren
        candidates.sort(key=lambda m: m.score, reverse=True)
        top = candidates[:50]  # Top 50 Kandidaten zurückgeben

        logger.info(
            "BroadMarketScanner: %d candidates (top: %s)",
            len(top),
            ", ".join(f"{m.symbol}+{m.change_pct:.1f}%" for m in top[:5]),
        )
        return top

    def _score(
        self,
        symbol: str,
        price: float,
        change_pct: float,
        vol_ratio: float,
        wsb_mentions: int,
        news_headlines: list[str],
    ) -> tuple[float, list[str]]:
        score = 0.0
        catalysts: list[str] = []

        # Kurs-Momentum
        if change_pct >= 15.0:
            score += 45.0
            catalysts.append(f"Kurs +{change_pct:.1f}% heute 🚀")
        elif change_pct >= 8.0:
            score += 30.0
            catalysts.append(f"Kurs +{change_pct:.1f}% heute 🔥")
        elif change_pct >= 3.0:
            score += 15.0
            catalysts.append(f"Kurs +{change_pct:.1f}% heute")
        elif change_pct <= -8.0:
            # Stark gefallene Aktien: Rebound-Potential
            score += 10.0
            catalysts.append(f"Rebound-Kandidat: {change_pct:.1f}%")

        # Volumen-Spike
        if vol_ratio >= 5.0:
            score += 35.0
            catalysts.append(f"Volumen {vol_ratio:.1f}× Normalwert 🔥")
        elif vol_ratio >= 3.0:
            score += 20.0
            catalysts.append(f"Volumen {vol_ratio:.1f}× Normalwert")
        elif vol_ratio >= 1.5:
            score += 10.0
            catalysts.append(f"Erhöhtes Volumen {vol_ratio:.1f}×")

        # Penny Stock Bonus (höheres Upside-Potential)
        if self.penny_mode and 0.50 <= price <= 10.0:
            score += 15.0
            catalysts.append(f"Penny Stock ${price:.2f}")
        elif price <= 20.0:
            score += 5.0

        # News-Katalysatoren
        if len(news_headlines) >= 3:
            score += 30.0
            catalysts.append(f"Breaking News: {news_headlines[0][:60]}")
        elif len(news_headlines) >= 1:
            score += 15.0
            catalysts.append(f"News: {news_headlines[0][:60]}")

        # WSB Social Momentum
        if wsb_mentions >= 500:
            score += 35.0
            catalysts.append(f"WSB: {wsb_mentions}× Erwähnungen 🔥🔥")
        elif wsb_mentions >= 200:
            score += 20.0
            catalysts.append(f"WSB: {wsb_mentions}× trending 🔥")
        elif wsb_mentions >= 100:
            score += 12.0
            catalysts.append(f"WSB: {wsb_mentions}× erwähnt")
        elif wsb_mentions >= 30:
            score += 6.0
            catalysts.append(f"WSB: {wsb_mentions}×")

        return score, catalysts
