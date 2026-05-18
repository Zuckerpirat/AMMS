"""Auto-Scanner: entdeckt neue Handelskandidaten automatisch.

Läuft im Hintergrund des Schedulers (jede Stunde) und scannt einen
konfigurierbaren Aktienuniversum nach Signalen:

  1. Momentum-Breakout  — RSI(14) > 60 + Kurs > MA50 + 20T-Rendite > 5%
  2. Volumen-Spike      — Volumen heute > 2× Durchschnitt der letzten 20T
  3. News-Aktivität     — ≥2 Artikel in letzten 24h (Alpaca News API)

Ergebnis: Liste neuer Symbole die zur Watchlist hinzugefügt werden sollen.
Symbole werden automatisch entfernt wenn sie mehrere Ticks lang kein
Signal mehr zeigen (Decay-Mechanismus).

Thread-safe. Alle Fehler werden geloggt, nie geworfen.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

# Mindest-Intervall zwischen zwei Scans (Sekunden)
_MIN_SCAN_INTERVAL = 3600  # 1 Stunde


@dataclass
class ScanResult:
    symbol: str
    score: float          # 0–100 (höher = stärkeres Signal)
    reasons: list[str]    # warum das Symbol entdeckt wurde
    discovered_at: str    # ISO timestamp


class AutoScanner:
    """Scannt ein Symbol-Universum stündlich nach Handelskandidaten.

    Parameters
    ----------
    data_client:
        Muss get_bars(symbol, limit=N) und optional get_news([symbol]) haben.
    universe:
        Liste von Symbolen die gescannt werden (z.B. S&P 500 Top 100).
    max_additions:
        Wie viele neue Symbole pro Scan maximal hinzugefügt werden.
    min_score:
        Minimum-Score (0–100) damit ein Symbol qualifiziert.
    decay_ticks:
        Nach wie vielen Ticks ohne Signal wird ein Symbol wieder entfernt.
    """

    def __init__(
        self,
        data_client,
        universe: list[str],
        *,
        max_additions: int = 5,
        min_score: float = 25.0,
        decay_ticks: int = 6,
        wsb_client=None,
        wsb_min_mentions: int = 50,
    ):
        self.data = data_client
        self.universe = [s.upper() for s in universe]
        self.max_additions = max_additions
        self.min_score = min_score
        self.decay_ticks = decay_ticks
        self._wsb_client = wsb_client
        self._wsb_min_mentions = wsb_min_mentions

        self._lock = threading.Lock()
        self._last_scan: float = 0.0
        # symbol → ticks since last signal (for decay)
        self._added_symbols: dict[str, int] = {}
        # all discovered results from last scan
        self._last_results: list[ScanResult] = []

    def set_universe(self, universe: list[str]) -> None:
        with self._lock:
            self.universe = [s.upper() for s in universe]

    def last_scan_results(self) -> list[ScanResult]:
        with self._lock:
            return list(self._last_results)

    def scan_and_update(self, current_watchlist: list[str]) -> list[str]:
        """Scan universe, return list of newly discovered symbols.

        Only runs if enough time has passed since the last scan.
        Returns [] if scan interval not reached yet.
        """
        now = time.monotonic()
        with self._lock:
            if now - self._last_scan < _MIN_SCAN_INTERVAL:
                return []
            self._last_scan = now
            universe = list(self.universe)

        watchlist_set = {s.upper() for s in current_watchlist}

        # Fetch WSB trending data and extend universe with hot symbols
        wsb_mentions: dict[str, int] = {}
        if self._wsb_client is not None:
            try:
                trending = self._wsb_client.scan(
                    min_mentions=self._wsb_min_mentions, top_n=30
                )
                wsb_mentions = {t.symbol: t.mentions for t in trending}
                # Auto-add WSB symbols not in universe
                for sym, mentions in wsb_mentions.items():
                    if mentions >= 100 and sym not in universe:
                        universe.append(sym)
                logger.info(
                    "WSB scan: %d trending symbols (top: %s)",
                    len(wsb_mentions),
                    ", ".join(
                        f"{s}={m}" for s, m in
                        sorted(wsb_mentions.items(), key=lambda x: -x[1])[:5]
                    ),
                )
            except Exception as exc:
                logger.debug("WSB scan failed in auto_scanner: %s", exc)

        # Scan all universe symbols not already in watchlist
        candidates: list[ScanResult] = []
        for sym in universe:
            if sym in watchlist_set:
                continue
            try:
                result = self._score_symbol(sym, wsb_mentions=wsb_mentions)
                if result is not None and result.score >= self.min_score:
                    candidates.append(result)
            except Exception as exc:
                logger.debug("Scanner error for %s: %s", sym, exc)

        # Sort by score descending, take top N
        candidates.sort(key=lambda r: r.score, reverse=True)
        new_symbols = [r.symbol for r in candidates[:self.max_additions]]

        with self._lock:
            self._last_results = candidates
            # Track newly added symbols for decay
            for sym in new_symbols:
                self._added_symbols[sym] = 0
            # Increment decay counter for symbols not re-discovered
            discovered_set = {r.symbol for r in candidates}
            to_remove = []
            for sym, ticks in list(self._added_symbols.items()):
                if sym not in discovered_set:
                    self._added_symbols[sym] = ticks + 1
                    if self._added_symbols[sym] >= self.decay_ticks:
                        to_remove.append(sym)
                        logger.info("Auto-scanner decay: removing %s after %d stale ticks", sym, self.decay_ticks)
                else:
                    self._added_symbols[sym] = 0  # reset decay
            for sym in to_remove:
                del self._added_symbols[sym]

        if new_symbols:
            logger.info("Auto-scanner found %d candidates: %s", len(new_symbols), new_symbols)

        return new_symbols

    def symbols_to_remove(self) -> list[str]:
        """Return symbols that have decayed and should be removed from watchlist."""
        with self._lock:
            return [
                sym for sym, ticks in self._added_symbols.items()
                if ticks >= self.decay_ticks
            ]

    def _score_symbol(
        self, symbol: str, wsb_mentions: dict[str, int] | None = None
    ) -> ScanResult | None:
        """Score a single symbol on momentum, volume, and news signals."""
        try:
            bars = self.data.get_bars(symbol, limit=60)
        except Exception:
            return None

        if not bars or len(bars) < 25:
            return None

        score = 0.0
        reasons: list[str] = []

        closes = [float(b.close) for b in bars]
        volumes = [float(b.volume) for b in bars] if hasattr(bars[0], "volume") else []
        cur = closes[-1]

        # ── Signal 1: Momentum-Breakout ────────────────────────────────
        # RSI(14)
        gains, losses = [], []
        for i in range(1, min(15, len(closes))):
            diff = closes[-i] - closes[-(i + 1)]
            (gains if diff >= 0 else losses).append(abs(diff))
        rsi = None
        if gains and losses:
            avg_g = sum(gains) / len(gains)
            avg_l = sum(losses) / len(losses)
            if avg_l > 0:
                rsi = 100 - 100 / (1 + avg_g / avg_l)

        # MA50
        ma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sum(closes) / len(closes)

        # 20-day return
        ret_20d = (cur / closes[-21] - 1.0) * 100.0 if len(closes) > 20 and closes[-21] > 0 else 0.0

        momentum_ok = (
            rsi is not None and rsi > 58 and
            cur > ma50 and
            ret_20d > 4.0
        )
        if momentum_ok:
            score += 40.0
            reasons.append(f"Momentum: RSI {rsi:.0f}, +{ret_20d:.1f}% (20T), über MA50")

        # Partial credit for strong RSI alone
        elif rsi is not None and rsi > 65:
            score += 15.0
            reasons.append(f"RSI stark: {rsi:.0f}")

        # ── Signal 2: Volumen-Spike ────────────────────────────────────
        if len(volumes) >= 21:
            avg_vol = sum(volumes[-21:-1]) / 20
            cur_vol = volumes[-1]
            if avg_vol > 0:
                vol_ratio = cur_vol / avg_vol
                if vol_ratio >= 2.5:
                    score += 35.0
                    reasons.append(f"Volumen-Spike: {vol_ratio:.1f}× Durchschnitt")
                elif vol_ratio >= 1.5:
                    score += 15.0
                    reasons.append(f"Erhöhtes Volumen: {vol_ratio:.1f}×")

        # ── Signal 3: News-Aktivität ───────────────────────────────────
        if hasattr(self.data, "get_news"):
            try:
                articles = self.data.get_news([symbol], limit=5)
                # Count articles from last 24h
                cutoff = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
                recent = [
                    a for a in articles
                    if str(a.get("created_at", "")) >= cutoff[:10]
                ]
                if len(recent) >= 3:
                    score += 25.0
                    reasons.append(f"News-Aktivität: {len(recent)} Artikel (24h)")
                elif len(recent) >= 1:
                    score += 10.0
                    reasons.append(f"News: {len(recent)} Artikel (24h)")
            except Exception:
                pass

        # ── Signal 4: WSB Social Momentum ─────────────────────────────
        mentions = (wsb_mentions or {}).get(symbol, 0)
        if mentions >= 500:
            score += 40.0
            reasons.append(f"WSB Hype: {mentions}× Erwähnungen 🔥🔥")
        elif mentions >= 200:
            score += 25.0
            reasons.append(f"WSB trending: {mentions}× Erwähnungen 🔥")
        elif mentions >= 100:
            score += 15.0
            reasons.append(f"WSB aktiv: {mentions}× Erwähnungen")
        elif mentions >= 50:
            score += 8.0
            reasons.append(f"WSB: {mentions}× Erwähnungen")

        if score < self.min_score:
            return None

        return ScanResult(
            symbol=symbol,
            score=round(score, 1),
            reasons=reasons,
            discovered_at=datetime.now(UTC).isoformat(),
        )


def format_scan_results(results: list[ScanResult], *, top: int = 10) -> str:
    """Format auto-scanner results for Telegram display."""
    if not results:
        return "Auto-Scanner: keine neuen Kandidaten gefunden."

    lines = [f"🔍 Auto-Scanner — {len(results)} Kandidaten"]
    for r in results[:top]:
        lines.append(f"\n  {r.symbol}  Score: {r.score:.0f}/100")
        for reason in r.reasons:
            lines.append(f"    ✦ {reason}")
    return "\n".join(lines)


# Default S&P 500 / Nasdaq scan universe (top liquid names)
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
    "AVGO", "ORCL", "NFLX", "ADBE", "CRM", "INTC", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "MPWR", "SMCI",
    "JPM", "BAC", "GS", "MS", "WFC", "V", "MA", "PYPL",
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "AMGN", "GILD",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "BA", "CAT", "HON", "GE", "RTX", "LMT",
    "COST", "WMT", "HD", "TGT", "AMZN",
    "DIS", "CMCSA", "NFLX", "SPOT",
    "SPY", "QQQ", "IWM",
]
