from __future__ import annotations

from pathlib import Path

from amms.dashboard.layout import (
    Layout,
    add_widget,
    default_layout,
    load_layout,
    move_widget,
    remove_widget,
    resize_widget,
    save_layout,
)


def test_default_layout_has_expected_widgets() -> None:
    layout = default_layout()
    types = [w.type for w in layout.widgets]
    assert "equity" in types
    assert "positions_table" in types
    assert all(w.id for w in layout.widgets)


def test_layout_round_trip(tmp_path: Path) -> None:
    layout = default_layout()
    path = tmp_path / "layout.json"
    save_layout(path, layout)
    loaded = load_layout(path)
    assert [w.type for w in loaded.widgets] == [w.type for w in layout.widgets]


def test_load_missing_falls_back_to_default(tmp_path: Path) -> None:
    layout = load_layout(tmp_path / "missing.json")
    assert layout.widgets, "default layout should be non-empty"


def test_add_remove_widget(tmp_path: Path) -> None:
    layout = Layout(widgets=[])
    add_widget(layout, "equity")
    add_widget(layout, "cash")
    assert [w.type for w in layout.widgets] == ["equity", "cash"]
    target_id = layout.widgets[0].id
    remove_widget(layout, target_id)
    assert [w.type for w in layout.widgets] == ["cash"]


def test_unknown_widget_type_ignored() -> None:
    layout = Layout(widgets=[])
    add_widget(layout, "does_not_exist")
    assert layout.widgets == []


def test_move_widget_clamps_at_bounds() -> None:
    layout = default_layout()
    first_id = layout.widgets[0].id
    move_widget(layout, first_id, -1)
    assert layout.widgets[0].id == first_id


def test_resize_widget() -> None:
    layout = default_layout()
    target = layout.widgets[0]
    resize_widget(layout, target.id, "lg")
    assert next(w for w in layout.widgets if w.id == target.id).size == "lg"


def test_resize_widget_rejects_invalid_size() -> None:
    layout = default_layout()
    target = layout.widgets[0]
    original = target.size
    resize_widget(layout, target.id, "huge")
    assert next(w for w in layout.widgets if w.id == target.id).size == original


def test_corrupted_file_falls_back_to_default(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{ not json")
    layout = load_layout(path)
    assert layout.widgets, "should recover with default layout"
