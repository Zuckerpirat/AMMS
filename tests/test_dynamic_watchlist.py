from __future__ import annotations

from pathlib import Path

import pytest

from amms.data.dynamic_watchlist import (
    _watchlist_path,
    add,
    clear,
    format_summary,
    load,
    normalize_symbol,
    remove,
)


def test_normalize_symbol_uppercases() -> None:
    assert normalize_symbol("nvda") == "NVDA"
    assert normalize_symbol("  pltr  ") == "PLTR"


def test_normalize_symbol_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        normalize_symbol("")
    with pytest.raises(ValueError):
        normalize_symbol("TOOLONG")
    with pytest.raises(ValueError):
        normalize_symbol("NV-DA")
    with pytest.raises(ValueError):
        normalize_symbol("123")


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    assert load(db) == set()


def test_add_persists_and_returns_status(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    new_set, msg = add(db, "nvda")
    assert "NVDA" in new_set
    assert "Added NVDA" in msg
    # Re-read from disk
    assert load(db) == {"NVDA"}


def test_add_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    add(db, "NVDA")
    new_set, msg = add(db, "NVDA")
    assert new_set == {"NVDA"}
    assert "already" in msg.lower()


def test_add_blocked_by_static_watchlist(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    new_set, msg = add(db, "AAPL", blocked={"AAPL", "MSFT"})
    assert new_set == set()  # nothing was actually persisted
    assert "static" in msg.lower()


def test_remove_works(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    add(db, "NVDA")
    add(db, "PLTR")
    new_set, msg = remove(db, "NVDA")
    assert new_set == {"PLTR"}
    assert "Removed NVDA" in msg


def test_remove_no_op_when_missing(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    add(db, "NVDA")
    new_set, msg = remove(db, "PLTR")
    assert new_set == {"NVDA"}
    assert "not in" in msg


def test_clear_removes_file(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    add(db, "NVDA")
    assert _watchlist_path(db).exists()
    clear(db)
    assert not _watchlist_path(db).exists()
    assert load(db) == set()


def test_load_handles_corrupt_file(tmp_path: Path) -> None:
    db = tmp_path / "amms.sqlite"
    path = _watchlist_path(db)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json {{{", encoding="utf-8")
    assert load(db) == set()


def test_format_summary_with_all_three_layers() -> None:
    out = format_summary(
        static_watchlist=("AAPL", "MSFT"),
        wsb_extras={"NVDA", "PLTR"},
        user_extras={"GME"},
    )
    assert "AAPL" in out
    assert "NVDA" in out
    assert "GME" in out
    assert "Static" in out
    assert "WSB" in out
    assert "User" in out


def test_format_summary_no_dynamic() -> None:
    out = format_summary(
        static_watchlist=("AAPL",),
        wsb_extras=set(),
        user_extras=set(),
    )
    assert "no dynamic" in out.lower()
