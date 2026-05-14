from __future__ import annotations

import urllib.request

from amms.metrics import _Metrics, metrics, start_metrics_server


def test_counter_and_gauge_render() -> None:
    m = _Metrics()
    m.inc("ticks_total")
    m.inc("ticks_total")
    m.observe("equity_dollars", 100000.0)
    out = m.render()
    assert "ticks_total 2" in out
    assert "equity_dollars 100000.0" in out
    assert "# TYPE ticks_total counter" in out
    assert "# TYPE equity_dollars gauge" in out


def test_labeled_counters_render() -> None:
    m = _Metrics()
    m.labeled_inc("orders_total", {"side": "buy"})
    m.labeled_inc("orders_total", {"side": "buy"})
    m.labeled_inc("orders_total", {"side": "sell"})
    out = m.render()
    assert 'orders_total{side="buy"} 2' in out
    assert 'orders_total{side="sell"} 1' in out


def test_metrics_server_serves_endpoint() -> None:
    metrics.inc("amms_test_counter")
    server = start_metrics_server(port=0)
    port = server.server_port
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
            body = resp.read().decode()
        assert "amms_test_counter" in body
    finally:
        server.shutdown()


def test_healthz_endpoint() -> None:
    server = start_metrics_server(port=0)
    port = server.server_port
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as resp:
            assert resp.read() == b"ok\n"
            assert resp.status == 200
    finally:
        server.shutdown()
