"""Minimal in-process metrics + Prometheus-compatible HTTP exposition.

Avoids the prometheus_client dependency. The bot is a single-process,
single-writer service so a tiny atomic counter set is enough. Endpoint
serves ``/metrics`` in Prometheus text format.

Use:
    from amms.metrics import metrics, start_metrics_server
    server = start_metrics_server(port=9100)
    metrics.inc("ticks_total")
    metrics.observe("equity_dollars", account.equity)
    ...
    server.shutdown()
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from http.server import BaseHTTPRequestHandler, HTTPServer

logger = logging.getLogger(__name__)


class _Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, float] = {}
        self._gauges: dict[str, float] = {}
        self._labeled: dict[str, dict[tuple[tuple[str, str], ...], float]] = {}

    def inc(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + amount

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = float(value)

    def labeled_inc(
        self, name: str, labels: dict[str, str], amount: float = 1.0
    ) -> None:
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._labeled.setdefault(name, {})
            series[key] = series.get(key, 0.0) + amount

    def snapshot(self) -> tuple[dict[str, float], dict[str, float], dict[str, dict]]:
        with self._lock:
            return (
                dict(self._counters),
                dict(self._gauges),
                {k: dict(v) for k, v in self._labeled.items()},
            )

    def render(self) -> str:
        counters, gauges, labeled = self.snapshot()
        lines: list[str] = []
        for name, value in sorted(counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, value in sorted(gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        for name, series in sorted(labeled.items()):
            lines.append(f"# TYPE {name} counter")
            for key, value in sorted(series.items()):
                label_str = ",".join(f'{k}="{v}"' for k, v in key)
                lines.append(f"{name}{{{label_str}}} {value}")
        lines.append("")
        return "\n".join(lines)


metrics = _Metrics()


class _MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/metrics", "/"):
            self.send_response(404)
            self.end_headers()
            return
        body = metrics.render().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Iterable[object]) -> None:  # noqa: A002, N802
        # Quiet by default; the bot's own logging carries enough signal.
        return


def start_metrics_server(*, host: str = "0.0.0.0", port: int = 9100) -> HTTPServer:
    """Start an HTTP server in a daemon thread. Returns the server so callers
    can ``server.shutdown()`` on stop."""
    server = HTTPServer((host, port), _MetricsHandler)
    thread = threading.Thread(
        target=server.serve_forever, name="amms-metrics", daemon=True
    )
    thread.start()
    logger.info("metrics server listening on %s:%d", host, port)
    return server
