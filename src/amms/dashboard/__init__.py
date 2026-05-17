"""Local web dashboard for AMMS.

Reporting-layer module. Reads from broker + sqlite, renders portfolio
overview. Does not generate signals or trade.

`server` is imported lazily so the rest of the package stays usable without
the optional `dashboard` extras installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amms.dashboard.server import create_app as create_app  # noqa: F401
    from amms.dashboard.server import run as run  # noqa: F401


def __getattr__(name: str) -> Any:
    if name in {"create_app", "run"}:
        from amms.dashboard import server
        return getattr(server, name)
    raise AttributeError(f"module 'amms.dashboard' has no attribute {name!r}")
