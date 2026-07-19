"""Import and exercise the FastAPI prototype using an isolated SQLite store."""

from __future__ import annotations

import asyncio
import importlib.util
import json
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
    # Running this file directly makes ``tests/`` sys.path[0]. Temporarily add
    # the repository root so the server can import neighboring modules exactly
    # as it does during normal startup.
    root_path = str(ROOT)
    sys.path.insert(0, root_path)
    try:
        spec.loader.exec_module(server)
    finally:
        sys.path.remove(root_path)

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

    for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        allowed, _, reason = server.mutation_authorization(peer, None, None)
        if not allowed or reason != "loopback_peer":
            raise AssertionError(f"loopback mutation unexpectedly denied: {peer}")
    if server.mutation_authorization("203.0.113.7", None, None)[:2] != (False, 403):
        raise AssertionError("remote mutation was not disabled by default")
    if server.mutation_authorization("203.0.113.7", "Bearer wrong", "secret")[:2] != (False, 401):
        raise AssertionError("incorrect remote bearer token was accepted")
    if server.mutation_authorization("203.0.113.7", "Bearer secret", "secret")[0] is not True:
        raise AssertionError("configured remote bearer token was rejected")

    async def asgi_status(peer: str, token: str | None = None) -> tuple[int, bytes]:
        sent = []
        delivered = False

        async def receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": json.dumps({"id": "remote", "name": "Remote"}).encode(),
                    "more_body": False,
                }
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        headers = [(b"content-type", b"application/json")]
        if token is not None:
            headers.append((b"authorization", f"Bearer {token}".encode()))
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/graph/nodes",
            "raw_path": b"/graph/nodes",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 49152),
            "server": ("127.0.0.1", 8000),
        }
        await server.app(scope, receive, send)
        response_start = next(message for message in sent if message["type"] == "http.response.start")
        body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
        return response_start["status"], body

    os.environ.pop("DSM_MUTATION_TOKEN", None)
    remote_status, remote_body = asyncio.run(asgi_status("203.0.113.7"))
    if remote_status != 403 or b"remote_mutation_disabled" not in remote_body:
        raise AssertionError("middleware did not reject an unauthenticated remote mutation")
    os.environ["DSM_MUTATION_TOKEN"] = "runtime-secret"
    authorized_status, _ = asyncio.run(asgi_status("203.0.113.7", "runtime-secret"))
    if authorized_status != 200:
        raise AssertionError("middleware rejected an authenticated remote mutation")
    os.environ.pop("DSM_MUTATION_TOKEN", None)

print("PASS FastAPI runtime, local mutation default, remote bearer boundary, and bridge smoke")
