"""Import and exercise the FastAPI prototype using an isolated SQLite store."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROUTES = {
    ("GET", "/"),
    ("POST", "/graph/nodes"),
    ("POST", "/graph/links"),
    ("GET", "/graph"),
    ("POST", "/telemetry/events"),
    ("GET", "/telemetry/recent"),
    ("GET", "/metrics/current"),
    ("GET", "/interventions/suggestions"),
    ("POST", "/bridge/analyze"),
    ("POST", "/import/obsidian"),
    ("POST", "/import/pdf"),
    ("POST", "/import/csv"),
}

with tempfile.TemporaryDirectory(prefix="dsm-server-smoke-") as temporary:
    os.environ["DSM_DB_PATH"] = str(Path(temporary) / "smoke.db")
    spec = importlib.util.spec_from_file_location("dynamic_sentience_server", ROOT / "server main.py")
    if spec is None or spec.loader is None:
        raise AssertionError("could not load server module")
    server = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = server
    spec.loader.exec_module(server)

    declared = {
        (method, route.path)
        for route in server.app.routes
        for method in (route.methods or set())
        if (method, route.path) in EXPECTED_ROUTES
    }
    if declared != EXPECTED_ROUTES:
        raise AssertionError(f"route mismatch: expected {EXPECTED_ROUTES - declared}, extra {declared - EXPECTED_ROUTES}")

    status = asyncio.run(server.root())
    if status["status"] != "operational":
        raise AssertionError("health function did not return operational")

    node_a = server.NodeData(id="a", name="Current claim", content="current evidence")
    node_b = server.NodeData(id="b", name="Outcome", content="held-out test")
    asyncio.run(server.add_node(node_a))
    asyncio.run(server.add_node(node_b))
    asyncio.run(server.add_link(server.LinkData(source_id="a", target_id="b", weight=0.9)))
    asyncio.run(
        server.log_telemetry(
            [server.TelemetryEvent(event_type="observation", node_id="a", metrics={"processing_time": 0.2})]
        )
    )

    graph = asyncio.run(server.get_graph())
    if len(graph.nodes) != 2 or len(graph.links) != 1:
        raise AssertionError("graph round-trip failed")
    metrics = asyncio.run(server.get_current_metrics())
    if not 0.1 <= metrics.phi_star <= 1.0:
        raise AssertionError("metric calculation returned an invalid coherence proxy")
    bridge = asyncio.run(
        server.bridge_analyze(server.BridgeKernelRequest(graph_snapshot=graph, telemetry_window=[]))
    )
    if bridge.kernel_version != "demo-open-v0.1":
        raise AssertionError("bridge response version changed")

print("PASS FastAPI import, route contract, SQLite round-trip, metrics, and bridge smoke")
