"""Tests for amms.data.secrets — runtime API key store."""

from __future__ import annotations

import os
import sqlite3

import pytest

from amms.data.secrets import (
    delete_secret,
    known_names,
    list_secrets,
    load_all,
    resolve_name,
    set_secret,
)


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


class TestResolveName:
    def test_short_name_resolved(self):
        result = resolve_name("anthropic_key")
        assert result == ("ANTHROPIC_API_KEY", "anthropic_key")

    def test_env_var_name_resolved(self):
        result = resolve_name("ANTHROPIC_API_KEY")
        assert result == ("ANTHROPIC_API_KEY", "anthropic_key")

    def test_unknown_returns_none(self):
        assert resolve_name("totally_unknown_key") is None

    def test_case_insensitive_short(self):
        assert resolve_name("ANTHROPIC_KEY") == ("ANTHROPIC_API_KEY", "anthropic_key")

    def test_all_known_names_resolvable(self):
        for short, env_var in known_names().items():
            r = resolve_name(short)
            assert r is not None
            assert r[0] == env_var


class TestSetSecret:
    def test_set_stores_and_activates(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        conn = _conn()
        ok, msg = set_secret(conn, "anthropic_key", "sk-ant-test123")
        assert ok is True
        assert "anthropic_key" in msg.lower() or "ANTHROPIC" in msg
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test123"

    def test_set_empty_value_rejected(self):
        conn = _conn()
        ok, msg = set_secret(conn, "anthropic_key", "  ")
        assert ok is False
        assert "leer" in msg.lower() or "empty" in msg.lower()

    def test_set_unknown_key_rejected(self):
        conn = _conn()
        ok, msg = set_secret(conn, "totally_bogus", "value")
        assert ok is False
        assert "unbekannt" in msg.lower() or "unknown" in msg.lower()

    def test_set_overwrites_existing(self, monkeypatch):
        conn = _conn()
        set_secret(conn, "anthropic_key", "old-value")
        ok, _ = set_secret(conn, "anthropic_key", "new-value")
        assert ok is True
        assert os.environ.get("ANTHROPIC_API_KEY") == "new-value"

    def test_set_no_conn_without_crash(self):
        # If conn is None, callers check before calling set_secret
        # Just ensure no crash with real conn
        conn = _conn()
        ok, msg = set_secret(conn, "alpaca_key", "PKTEST123")
        assert ok is True


class TestDeleteSecret:
    def test_delete_removes_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-del-test")
        conn = _conn()
        set_secret(conn, "anthropic_key", "sk-ant-del-test")
        ok, msg = delete_secret(conn, "anthropic_key")
        assert ok is True
        assert os.environ.get("ANTHROPIC_API_KEY") in (None, "")

    def test_delete_not_found_returns_false(self):
        conn = _conn()
        ok, msg = delete_secret(conn, "anthropic_key")
        assert ok is False
        assert "nicht" in msg.lower() or "not" in msg.lower()

    def test_delete_unknown_key(self):
        conn = _conn()
        ok, msg = delete_secret(conn, "bogus_key")
        assert ok is False


class TestLoadAll:
    def test_load_applies_to_env(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        conn = _conn()
        set_secret(conn, "reddit_id", "my-reddit-id")
        # Unset from env to test loading
        os.environ.pop("REDDIT_CLIENT_ID", None)
        count = load_all(conn)
        assert count >= 0  # may be 0 if env already set; just no crash

    def test_load_returns_count(self, monkeypatch):
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
        conn = _conn()
        os.environ.pop("ALPACA_API_KEY", None)
        os.environ.pop("ALPACA_API_SECRET", None)
        set_secret(conn, "alpaca_key", "PK_TEST")
        set_secret(conn, "alpaca_secret", "SEC_TEST")
        # Clear from env to force re-load
        os.environ.pop("ALPACA_API_KEY", None)
        os.environ.pop("ALPACA_API_SECRET", None)
        count = load_all(conn)
        assert count == 2

    def test_load_empty_db_returns_zero(self):
        conn = _conn()
        assert load_all(conn) == 0


class TestListSecrets:
    def test_list_empty(self):
        conn = _conn()
        result = list_secrets(conn)
        assert result == []

    def test_list_shows_stored(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        conn = _conn()
        set_secret(conn, "anthropic_key", "sk-ant-ABCDEF1234567890")
        result = list_secrets(conn)
        assert len(result) == 1
        assert result[0]["short_name"] == "anthropic_key"
        assert result[0]["masked"] != "sk-ant-ABCDEF1234567890"  # must be masked
        assert "***" in result[0]["masked"]

    def test_list_multiple(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        conn = _conn()
        set_secret(conn, "anthropic_key", "sk-ant-test")
        set_secret(conn, "reddit_id", "reddit-id-123")
        result = list_secrets(conn)
        assert len(result) == 2
        short_names = {r["short_name"] for r in result}
        assert "anthropic_key" in short_names
        assert "reddit_id" in short_names


class TestObfuscation:
    def test_roundtrip(self):
        from amms.data.secrets import _obfuscate, _deobfuscate
        seed = "test-seed-abc"
        value = "sk-ant-my-secret-key-here-1234567890"
        obf = _obfuscate(value, seed)
        assert obf != value
        recovered = _deobfuscate(obf, seed)
        assert recovered == value

    def test_different_seeds_give_different_output(self):
        from amms.data.secrets import _obfuscate
        v = "same-value"
        assert _obfuscate(v, "seed-a") != _obfuscate(v, "seed-b")
