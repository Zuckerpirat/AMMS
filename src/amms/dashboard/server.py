"""FastAPI server for the AMMS dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from amms.dashboard.chart import build_equity_chart
from amms.dashboard.data import get_portfolio
from amms.dashboard.indices import get_index
from amms.dashboard.layout import (
    add_widget,
    load_layout,
    move_widget,
    remove_widget,
    reorder_widget,
    resize_widget,
    save_layout,
)
from amms.dashboard.widgets import SIZES, WIDGET_REGISTRY

PKG_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PKG_DIR / "templates"
STATIC_DIR = PKG_DIR / "static"


def _needed_index_keys(layout) -> set[str]:
    return {
        w.type.removeprefix("index_")
        for w in layout.widgets
        if w.type.startswith("index_")
    }


def _format_currency(value: float) -> str:
    return f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _format_signed(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{_format_currency(abs(value))}"


def _format_pct(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.2f}%".replace(".", ",")


def create_app(layout_path: Path, db_path: Path, refresh_ms: int = 1000) -> FastAPI:
    app = FastAPI(title="AMMS Dashboard", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.filters["currency"] = _format_currency
    templates.env.filters["signed"] = _format_signed
    templates.env.filters["pct"] = _format_pct
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    def _context(edit: bool) -> dict:
        layout = load_layout(layout_path)
        portfolio = get_portfolio(db_path)
        indices = {key: get_index(key) for key in _needed_index_keys(layout)}
        return {
            "layout": layout,
            "portfolio": portfolio,
            "registry": WIDGET_REGISTRY,
            "available_widgets": list(WIDGET_REGISTRY.values()),
            "edit_mode": edit,
            "sizes": SIZES,
            "chart": build_equity_chart(portfolio.equity_history),
            "indices": indices,
            "refresh_ms": refresh_ms,
        }

    @app.get("/")
    def index(request: Request, edit: int = 0):
        return templates.TemplateResponse(request, "index.html", _context(bool(edit)))

    @app.get("/api/grid")
    def api_grid(request: Request, edit: int = 0):
        return templates.TemplateResponse(request, "_grid.html", _context(bool(edit)))

    @app.post("/layout/add")
    def layout_add(widget_type: str = Form(...)):
        layout = load_layout(layout_path)
        add_widget(layout, widget_type)
        save_layout(layout_path, layout)
        return RedirectResponse("/?edit=1", status_code=303)

    @app.post("/layout/remove")
    def layout_remove(widget_id: str = Form(...)):
        layout = load_layout(layout_path)
        remove_widget(layout, widget_id)
        save_layout(layout_path, layout)
        return RedirectResponse("/?edit=1", status_code=303)

    @app.post("/layout/move")
    def layout_move(widget_id: str = Form(...), direction: int = Form(...)):
        layout = load_layout(layout_path)
        move_widget(layout, widget_id, direction)
        save_layout(layout_path, layout)
        return RedirectResponse("/?edit=1", status_code=303)

    @app.post("/layout/resize")
    def layout_resize(widget_id: str = Form(...), size: str = Form(...)):
        layout = load_layout(layout_path)
        resize_widget(layout, widget_id, size)
        save_layout(layout_path, layout)
        return RedirectResponse("/?edit=1", status_code=303)

    @app.post("/layout/reorder")
    def layout_reorder(widget_id: str = Form(...), new_index: int = Form(...)):
        layout = load_layout(layout_path)
        reorder_widget(layout, widget_id, new_index)
        save_layout(layout_path, layout)
        return Response(status_code=204)

    @app.post("/layout/reset")
    def layout_reset():
        from amms.dashboard.layout import default_layout
        save_layout(layout_path, default_layout())
        return RedirectResponse("/", status_code=303)

    return app


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    layout_path: Path | None = None,
    db_path: Path | None = None,
    refresh_ms: int = 1000,
) -> None:
    import uvicorn

    layout_path = layout_path or (Path.home() / ".amms" / "dashboard_layout.json")
    db_path = db_path or (Path.home() / ".amms" / "amms.db")
    app = create_app(layout_path=layout_path, db_path=db_path, refresh_ms=refresh_ms)
    uvicorn.run(app, host=host, port=port, log_level="info")
