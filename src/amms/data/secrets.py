"""Runtime secrets store — set API keys via Telegram without editing files.

Secrets are stored in the SQLite DB (table: runtime_secrets) and loaded
into os.environ on startup so the rest of the bot finds them normally.

Security model:
  Values are stored as base64(XOR(value, key)) where key is derived from
  a machine-local secret (AMMS_SECRET_SEED env var, or a generated per-DB
  seed stored in the DB itself). This is NOT strong cryptography — it
  prevents casual shoulder-surfing in SQLite GUI tools but does NOT protect
  against someone who has full read access to the DB file and can read env vars.
  For a locally-run personal trading bot this is an acceptable tradeoff.

Supported secret names (canonical → env var):
  anthropic_key  → ANTHROPIC_API_KEY
  reddit_id      → REDDIT_CLIENT_ID
  reddit_secret  → REDDIT_CLIENT_SECRET
  alpaca_key     → ALPACA_API_KEY
  alpaca_secret  → ALPACA_API_SECRET
  telegram_token → TELEGRAM_BOT_TOKEN
  telegram_chat  → TELEGRAM_CHAT_ID
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import sqlite3
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Maps user-facing short names to environment variable names
_KNOWN: dict[str, str] = {
    "anthropic_key": "ANTHROPIC_API_KEY",
    "reddit_id": "REDDIT_CLIENT_ID",
    "reddit_secret": "REDDIT_CLIENT_SECRET",
    "alpaca_key": "ALPACA_API_KEY",
    "alpaca_secret": "ALPACA_API_SECRET",
    "telegram_token": "TELEGRAM_BOT_TOKEN",
    "telegram_chat": "TELEGRAM_CHAT_ID",
}

# Reverse map: env var name → short name (for display)
_ENV_TO_SHORT: dict[str, str] = {v: k for k, v in _KNOWN.items()}

# Secret names that should be shown with partial masking even in admin output
_SENSITIVE = {"ANTHROPIC_API_KEY", "ALPACA_API_KEY", "ALPACA_API_SECRET",
              "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "TELEGRAM_BOT_TOKEN",
              "TELEGRAM_CHAT_ID"}


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runtime_secrets (
            name        TEXT PRIMARY KEY,   -- env var name (e.g. ANTHROPIC_API_KEY)
            value_b64   TEXT NOT NULL,      -- obfuscated value
            short_name  TEXT NOT NULL,      -- user-facing alias
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS _secret_seed (
            id    INTEGER PRIMARY KEY CHECK (id = 1),
            seed  TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _get_seed(conn: sqlite3.Connection) -> str:
    """Return (or generate) a per-database obfuscation seed."""
    row = conn.execute("SELECT seed FROM _secret_seed WHERE id = 1").fetchone()
    if row:
        return row["seed"]
    import secrets as _sec
    seed = _sec.token_hex(32)
    conn.execute("INSERT INTO _secret_seed(id, seed) VALUES (1, ?)", (seed,))
    conn.commit()
    return seed


def _obfuscate(value: str, seed: str) -> str:
    """XOR-based obfuscation → base64. Not cryptographically strong."""
    key = hashlib.sha256((seed + "amms-secrets-v1").encode()).digest()
    b = value.encode()
    out = bytes(b[i] ^ key[i % len(key)] for i in range(len(b)))
    return base64.b64encode(out).decode()


def _deobfuscate(value_b64: str, seed: str) -> str:
    key = hashlib.sha256((seed + "amms-secrets-v1").encode()).digest()
    b = base64.b64decode(value_b64)
    out = bytes(b[i] ^ key[i % len(key)] for i in range(len(b)))
    return out.decode()


def _mask(value: str) -> str:
    """Show first 4 + last 4 characters with *** in between."""
    if len(value) <= 10:
        return "***"
    return value[:4] + "***" + value[-4:]


def resolve_name(name: str) -> tuple[str, str] | None:
    """Map short name or env var name to (env_var, short_name). Returns None if unknown."""
    n = name.strip().lower()
    if n in _KNOWN:
        return _KNOWN[n], n
    upper = name.strip().upper()
    if upper in _ENV_TO_SHORT:
        return upper, _ENV_TO_SHORT[upper]
    return None


def set_secret(conn: sqlite3.Connection, name: str, value: str) -> tuple[bool, str]:
    """Store a secret and apply it to os.environ immediately.

    Returns (success, message).
    """
    resolved = resolve_name(name)
    if resolved is None:
        known = ", ".join(sorted(_KNOWN.keys()))
        return False, f"Unbekannter Key-Name '{name}'. Bekannte Namen: {known}"

    env_var, short = resolved
    value = value.strip()
    if not value:
        return False, "Wert darf nicht leer sein."

    _ensure_table(conn)
    seed = _get_seed(conn)
    obf = _obfuscate(value, seed)
    now = datetime.now(UTC).isoformat()

    conn.execute(
        """
        INSERT INTO runtime_secrets(name, value_b64, short_name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            value_b64  = excluded.value_b64,
            updated_at = excluded.updated_at
        """,
        (env_var, obf, short, now, now),
    )
    conn.commit()

    # Apply immediately
    os.environ[env_var] = value
    logger.info("Runtime secret '%s' set and applied to env.", env_var)
    return True, f"✓ {short} ({env_var}) gesetzt und aktiv."


def delete_secret(conn: sqlite3.Connection, name: str) -> tuple[bool, str]:
    """Remove a secret from DB and unset from os.environ."""
    resolved = resolve_name(name)
    if resolved is None:
        return False, f"Unbekannter Key-Name '{name}'."

    env_var, short = resolved
    _ensure_table(conn)
    cur = conn.execute("DELETE FROM runtime_secrets WHERE name = ?", (env_var,))
    conn.commit()

    if cur.rowcount == 0:
        return False, f"'{short}' war nicht gespeichert."

    # Unset from environment
    os.environ.pop(env_var, None)
    logger.info("Runtime secret '%s' deleted.", env_var)
    return True, f"✓ {short} gelöscht."


def load_all(conn: sqlite3.Connection) -> int:
    """Load all stored secrets into os.environ. Call on bot startup.

    Returns number of secrets loaded.
    """
    try:
        _ensure_table(conn)
        seed = _get_seed(conn)
        rows = conn.execute(
            "SELECT name, value_b64 FROM runtime_secrets"
        ).fetchall()
        count = 0
        for row in rows:
            try:
                value = _deobfuscate(row["value_b64"], seed)
                # Only set if not already overridden by actual env (env takes priority)
                if not os.environ.get(row["name"], "").strip():
                    os.environ[row["name"]] = value
                    count += 1
            except Exception as exc:
                logger.warning("Could not load secret '%s': %s", row["name"], exc)
        if count:
            logger.info("Loaded %d runtime secret(s) from DB.", count)
        return count
    except Exception as exc:
        logger.warning("Could not load runtime secrets: %s", exc)
        return 0


def list_secrets(conn: sqlite3.Connection) -> list[dict]:
    """Return a list of stored secrets (values masked for display)."""
    try:
        _ensure_table(conn)
        seed = _get_seed(conn)
        rows = conn.execute(
            "SELECT name, value_b64, short_name, updated_at FROM runtime_secrets ORDER BY name"
        ).fetchall()
        result = []
        for row in rows:
            try:
                value = _deobfuscate(row["value_b64"], seed)
                masked = _mask(value) if row["name"] in _SENSITIVE else value
            except Exception:
                masked = "***"
            # Check if active in env
            active = os.environ.get(row["name"], "") == value if row["name"] not in _SENSITIVE else (
                bool(os.environ.get(row["name"]))
            )
            result.append({
                "short_name": row["short_name"],
                "env_var": row["name"],
                "masked": masked,
                "updated_at": str(row["updated_at"])[:10],
                "active_in_env": active,
            })
        return result
    except Exception as exc:
        logger.warning("Could not list secrets: %s", exc)
        return []


def known_names() -> dict[str, str]:
    """Return {short_name: env_var} for all supported secrets."""
    return dict(_KNOWN)
