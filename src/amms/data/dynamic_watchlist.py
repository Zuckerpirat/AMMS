"""User-managed watchlist additions, persisted as JSON.

The static watchlist lives in config.yaml. WSB Auto-Discovery adds extras
on a schedule. This module is the third layer: tickers the user adds by
hand from Telegram (/add) or the CLI (`amms add SYM`). Persisted to disk
so they survive container restarts.

Stored as a tiny JSON file next to the SQLite DB. No locking — the file
is rewritten in full on every change and the cost is negligible.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _watchlist_path(db_path: Path) -> Path:
    return Path(db_path).parent / "dynamic_watchlist.json"


def _load_raw(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("dynamic_watchlist: corrupt file, treating as empty",
                       exc_info=True)
        return set()
    if not isinstance(data, list):
        return set()
    return {str(s).upper() for s in data if isinstance(s, str)}


def _save_raw(path: Path, symbols: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = sorted(symbols)
    path.write_text(json.dumps(payload), encoding="utf-8")


def normalize_symbol(symbol: str) -> str:
    """Uppercase + validate. Raises ValueError on bad input."""
    sym = (symbol or "").strip().upper()
    if not _TICKER_RE.match(sym):
        raise ValueError(
            f"Not a valid ticker: {symbol!r}. Use 1-5 uppercase letters (e.g. NVDA)."
        )
    return sym


def load(db_path: Path) -> set[str]:
    """Read the persisted set of user-added symbols."""
    return _load_raw(_watchlist_path(db_path))


def add(db_path: Path, symbol: str, *, blocked: set[str] | None = None) -> tuple[set[str], str]:
    """Add ``symbol`` to the dynamic watchlist.

    Returns (new_set, status_message). Status is suitable for /add reply.
    ``blocked`` is an optional set of symbols already on the static config
    watchlist - we report a friendly "already on static watchlist" instead
    of silently no-oping.
    """
    sym = normalize_symbol(symbol)
    path = _watchlist_path(db_path)
    current = _load_raw(path)
    if blocked and sym in {s.upper() for s in blocked}:
        return current, f"{sym} is already on the static watchlist (config.yaml)."
    if sym in current:
        return current, f"{sym} already in dynamic watchlist."
    current.add(sym)
    _save_raw(path, current)
    return current, f"Added {sym}. Dynamic watchlist now has {len(current)} symbol(s)."


def remove(db_path: Path, symbol: str) -> tuple[set[str], str]:
    """Remove ``symbol`` from the dynamic watchlist."""
    sym = normalize_symbol(symbol)
    path = _watchlist_path(db_path)
    current = _load_raw(path)
    if sym not in current:
        return current, f"{sym} not in dynamic watchlist."
    current.remove(sym)
    _save_raw(path, current)
    return current, f"Removed {sym}. Dynamic watchlist now has {len(current)} symbol(s)."


def clear(db_path: Path) -> str:
    """Empty the dynamic watchlist."""
    path = _watchlist_path(db_path)
    if path.exists():
        path.unlink()
    return "Dynamic watchlist cleared."


def format_summary(
    static_watchlist: tuple[str, ...] | list[str],
    wsb_extras: set[str] | frozenset[str],
    user_extras: set[str],
) -> str:
    """Render a Telegram-friendly multi-source watchlist view."""
    static_upper = [s.upper() for s in static_watchlist]
    lines = [
        f"Effective watchlist ({len(static_upper) + len(wsb_extras) + len(user_extras)} total):",
        f"📌 Static (config): {', '.join(static_upper) if static_upper else '(none)'}",
    ]
    if wsb_extras:
        lines.append(f"🔥 WSB-discovered: {', '.join(sorted(wsb_extras))}")
    if user_extras:
        lines.append(f"✋ User-added:     {', '.join(sorted(user_extras))}")
    if not wsb_extras and not user_extras:
        lines.append("(no dynamic additions; use /add SYM to add one)")
    return "\n".join(lines)
