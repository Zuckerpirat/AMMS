"""Dashboard layout persistence.

A layout is an ordered list of widget instances. Each instance has a stable
id, a widget type (from the registry), and a chosen size. The layout is
stored as JSON so the user keeps their setup between sessions.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from amms.dashboard.widgets import SIZES, WIDGET_REGISTRY, is_known


@dataclass
class WidgetInstance:
    id: str
    type: str
    size: str

    @staticmethod
    def new(widget_type: str, size: str | None = None) -> WidgetInstance:
        spec = WIDGET_REGISTRY[widget_type]
        return WidgetInstance(
            id=uuid.uuid4().hex[:8],
            type=widget_type,
            size=size or spec.default_size,
        )


@dataclass
class Layout:
    widgets: list[WidgetInstance] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"widgets": [asdict(w) for w in self.widgets]}, indent=2)

    @staticmethod
    def from_json(text: str) -> Layout:
        data = json.loads(text)
        widgets: list[WidgetInstance] = []
        for raw in data.get("widgets", []):
            wtype = raw.get("type")
            if not isinstance(wtype, str) or not is_known(wtype):
                continue
            size = raw.get("size")
            if size not in SIZES:
                size = WIDGET_REGISTRY[wtype].default_size
            wid = raw.get("id") or uuid.uuid4().hex[:8]
            widgets.append(WidgetInstance(id=wid, type=wtype, size=size))
        return Layout(widgets=widgets)


def default_layout() -> Layout:
    return Layout(
        widgets=[
            WidgetInstance.new("equity"),
            WidgetInstance.new("day_change"),
            WidgetInstance.new("cash"),
            WidgetInstance.new("buying_power"),
            WidgetInstance.new("equity_sparkline"),
            WidgetInstance.new("positions_table"),
        ]
    )


def load_layout(path: Path) -> Layout:
    if not path.exists():
        return default_layout()
    try:
        return Layout.from_json(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_layout()


def save_layout(path: Path, layout: Layout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(layout.to_json(), encoding="utf-8")


def add_widget(layout: Layout, widget_type: str) -> Layout:
    if not is_known(widget_type):
        return layout
    layout.widgets.append(WidgetInstance.new(widget_type))
    return layout


def remove_widget(layout: Layout, widget_id: str) -> Layout:
    layout.widgets = [w for w in layout.widgets if w.id != widget_id]
    return layout


def move_widget(layout: Layout, widget_id: str, direction: int) -> Layout:
    idx = next((i for i, w in enumerate(layout.widgets) if w.id == widget_id), -1)
    if idx < 0:
        return layout
    new_idx = max(0, min(len(layout.widgets) - 1, idx + direction))
    if new_idx == idx:
        return layout
    layout.widgets[idx], layout.widgets[new_idx] = (
        layout.widgets[new_idx],
        layout.widgets[idx],
    )
    return layout


def resize_widget(layout: Layout, widget_id: str, size: str) -> Layout:
    if size not in SIZES:
        return layout
    for w in layout.widgets:
        if w.id == widget_id:
            w.size = size
    return layout
