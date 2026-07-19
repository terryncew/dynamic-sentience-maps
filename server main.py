"""Dynamic Sentience Maps FastAPI server.

Open-core graph storage, telemetry, observable metric estimators, and the
declared bridge contract. The estimators are demonstration proxies; see
CLAIM_BOUNDARY.md before using them in a decision system.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dynamic Sentience Maps API",
    description="Open-core telemetry and graph processing for living knowledge structures",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NodeData(BaseModel):
    id: str
    name: str
    content: Optional[str] = None
    node_type: str = "concept"
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class LinkData(BaseModel):
    source_id: str
    target_id: str
    link_type: str = "semantic"
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphData(BaseModel):
    nodes: List[NodeData]
    links: List[LinkData]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TelemetryEvent(BaseModel):
    timestamp: datetime = Field(default_factory=datetime.now)
    event_type: str
    node_id: Optional[str] = None
    metrics: Dict[str, float]
    context: Dict[str, Any] = Field(default_factory=dict)


class CoherenceMetrics(BaseModel):
    phi_star: float = Field(..., description="Phi-star coherence proxy")
    kappa: float = Field(..., description="Kappa stress proxy")
    epsilon: float = Field(..., description="Epsilon entropy-leak proxy")
    i_capacity: float = Field(..., description="Information-capacity proxy")
    timestamp: datetime = Field(default_factory=datetime.now)


class InterventionSuggestion(BaseModel):
    intervention_type: str
    target_nodes: List[str]
    priority: float
    description: str
    estimated_impact: Dict[str, float]


class BridgeKernelRequest(BaseModel):
    graph_snapshot: GraphData
    telemetry_window: List[TelemetryEvent]
    analysis_params: Dict[str, Any] = Field(default_factory=dict)


class BridgeKernelResponse(BaseModel):
    metrics: CoherenceMetrics
    interventions: List[InterventionSuggestion]
    kernel_version: str = "demo-open-v0.1"


def _model_dump(model: BaseModel) -> Dict[str, Any]:
    """Return a plain mapping under Pydantic 1 or 2."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


class GraphStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.init_database()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init_database(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    content TEXT,
                    node_type TEXT DEFAULT 'concept',
                    embedding TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS links (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    link_type TEXT DEFAULT 'semantic',
                    weight REAL NOT NULL DEFAULT 1.0,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                );

                CREATE TABLE IF NOT EXISTS telemetry (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    node_id TEXT,
                    metrics TEXT NOT NULL,
                    context TEXT,
                    FOREIGN KEY (node_id) REFERENCES nodes(id)
                );
                """
            )

    def add_node(self, node: NodeData) -> str:
        now = datetime.now().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO nodes
                    (id, name, content, node_type, embedding, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    content = excluded.content,
                    node_type = excluded.node_type,
                    embedding = excluded.embedding,
                    metadata = excluded.metadata,
                    updated_at = excluded.updated_at
                """,
                (
                    node.id,
                    node.name,
                    node.content,
                    node.node_type,
                    json.dumps(node.embedding) if node.embedding is not None else None,
                    json.dumps(node.metadata),
                    node.created_at.isoformat(),
                    now,
                ),
            )
        return node.id

    def add_link(self, link: LinkData) -> str:
        link_id = str(uuid.uuid4())
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO links
                    (id, source_id, target_id, link_type, weight, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    link.source_id,
                    link.target_id,
                    link.link_type,
                    link.weight,
                    json.dumps(link.metadata),
                    datetime.now().isoformat(),
                ),
            )
        return link_id

    def get_graph(self, limit: int = 1000) -> GraphData:
        safe_limit = max(1, min(limit, 10_000))
        with self.connect() as connection:
            node_rows = connection.execute(
                """
                SELECT id, name, content, node_type, embedding, metadata, created_at
                FROM nodes ORDER BY created_at LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
            link_rows = connection.execute(
                """
                SELECT source_id, target_id, link_type, weight, metadata
                FROM links ORDER BY created_at LIMIT ?
                """,
                (safe_limit * 2,),
            ).fetchall()

        nodes = [
            NodeData(
                id=row[0],
                name=row[1],
                content=row[2],
                node_type=row[3],
                embedding=json.loads(row[4]) if row[4] else None,
                metadata=json.loads(row[5]) if row[5] else {},
                created_at=datetime.fromisoformat(row[6]),
            )
            for row in node_rows
        ]
        links = [
            LinkData(
                source_id=row[0],
                target_id=row[1],
                link_type=row[2],
                weight=row[3],
                metadata=json.loads(row[4]) if row[4] else {},
            )
            for row in link_rows
        ]
        return GraphData(nodes=nodes, links=links)

    def add_telemetry(self, events: List[TelemetryEvent]) -> int:
        rows = [
            (
                str(uuid.uuid4()),
                event.timestamp.isoformat(),
                event.event_type,
                event.node_id,
                json.dumps(event.metrics),
                json.dumps(event.context),
            )
            for event in events
        ]
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO telemetry (id, timestamp, event_type, node_id, metrics, context)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def get_recent_telemetry(self, hours: int = 1, limit: int = 1000) -> List[TelemetryEvent]:
        safe_hours = max(1, min(hours, 24 * 365))
        safe_limit = max(1, min(limit, 10_000))
        cutoff = (datetime.now() - timedelta(hours=safe_hours)).isoformat()
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT timestamp, event_type, node_id, metrics, context
                FROM telemetry
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (cutoff, safe_limit),
            ).fetchall()
        return [
            TelemetryEvent(
                timestamp=datetime.fromisoformat(row[0]),
                event_type=row[1],
                node_id=row[2],
                metrics=json.loads(row[3]),
                context=json.loads(row[4]) if row[4] else {},
            )
            for row in rows
        ]


class OpenMetricsEngine:
    """Open demonstration estimators for the four public readings."""

    def __init__(self) -> None:
        self.baseline_capacity = 1000.0
        self.stress_history: List[float] = []
        self.entropy_history: List[float] = []

    def calculate_stress(self, graph: GraphData, telemetry: List[TelemetryEvent]) -> float:
        if not graph.nodes:
            return 0.0
        degrees: Dict[str, int] = {}
        for link in graph.links:
            degrees[link.source_id] = degrees.get(link.source_id, 0) + 1
            degrees[link.target_id] = degrees.get(link.target_id, 0) + 1
        degree_stress = sum(min(degree**1.5, 10) for degree in degrees.values()) / len(graph.nodes)
        recent = [event for event in telemetry if event.timestamp > datetime.now() - timedelta(minutes=5)]
        telemetry_stress = sum(event.metrics.get("processing_time", 0.0) for event in recent) / max(len(recent), 1)
        stress = min(degree_stress * 0.7 + telemetry_stress * 0.3, 10.0)
        self.stress_history.append(stress)
        return stress

    def calculate_entropy_leak(self, graph: GraphData, telemetry: List[TelemetryEvent]) -> float:
        if len(graph.nodes) < 2:
            self.entropy_history.append(0.0)
            return 0.0
        weak_connection_entropy = len([link for link in graph.links if link.weight < 0.3]) / max(len(graph.links), 1)
        contradiction_entropy = len([event for event in telemetry if event.event_type == "contradiction"]) / max(len(telemetry), 1)
        entropy = min(weak_connection_entropy * 0.6 + contradiction_entropy * 0.4, 1.0)
        self.entropy_history.append(entropy)
        return entropy

    def calculate_capacity(self, graph: GraphData) -> float:
        if not graph.nodes:
            return 0.0
        average_degree = sum(
            len([link for link in graph.links if link.source_id == node.id or link.target_id == node.id])
            for node in graph.nodes
        ) / len(graph.nodes)
        average_content_length = sum(len(node.content or "") for node in graph.nodes) / len(graph.nodes)
        content_factor = min(average_content_length / 100, 2.0)
        return self.baseline_capacity * (1 + average_degree * 0.1) * content_factor

    def calculate_coherence(self, stress: float, entropy: float, capacity: float) -> float:
        normalized_capacity = min(capacity / self.baseline_capacity, 2.0)
        coherence = normalized_capacity / (1 + stress * 0.3 + entropy * 2.0)
        if len(self.stress_history) > 1 and len(self.entropy_history) > 1:
            stress_stability = 1.0 - abs(self.stress_history[-1] - self.stress_history[-2]) / 10.0
            entropy_stability = 1.0 - abs(self.entropy_history[-1] - self.entropy_history[-2])
            coherence *= (stress_stability + entropy_stability) / 2
        return max(0.1, min(coherence, 1.0))

    def analyze(self, graph: GraphData, telemetry: List[TelemetryEvent]) -> CoherenceMetrics:
        stress = self.calculate_stress(graph, telemetry)
        entropy = self.calculate_entropy_leak(graph, telemetry)
        capacity = self.calculate_capacity(graph)
        coherence = self.calculate_coherence(stress, entropy, capacity)
        return CoherenceMetrics(
            phi_star=coherence,
            kappa=stress,
            epsilon=entropy,
            i_capacity=capacity,
        )


default_db_path = Path(os.environ.get("DSM_DB_PATH", Path(__file__).with_name("dsm_graph.db")))
graph_store = GraphStore(default_db_path)
metrics_engine = OpenMetricsEngine()


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "service": "Dynamic Sentience Maps API",
        "version": "0.1.0",
        "status": "operational",
        "features": ["graph_storage", "telemetry", "open_metrics", "bridge_api"],
    }


@app.post("/graph/nodes", response_model=dict)
async def add_node(node: NodeData) -> Dict[str, str]:
    try:
        node_id = graph_store.add_node(node)
        logger.info("Added node: %s - %s", node_id, node.name)
        return {"node_id": node_id, "status": "created"}
    except Exception as exc:
        logger.exception("Error adding node")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/graph/links", response_model=dict)
async def add_link(link: LinkData) -> Dict[str, str]:
    try:
        link_id = graph_store.add_link(link)
        logger.info("Added link: %s -> %s", link.source_id, link.target_id)
        return {"link_id": link_id, "status": "created"}
    except Exception as exc:
        logger.exception("Error adding link")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph", response_model=GraphData)
async def get_graph(limit: int = 1000) -> GraphData:
    try:
        return graph_store.get_graph(limit)
    except Exception as exc:
        logger.exception("Error retrieving graph")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/telemetry/events")
async def log_telemetry(events: List[TelemetryEvent]) -> Dict[str, Any]:
    try:
        count = graph_store.add_telemetry(events)
        return {"events_logged": count, "status": "success"}
    except Exception as exc:
        logger.exception("Error logging telemetry")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/telemetry/recent", response_model=List[TelemetryEvent])
async def get_recent_telemetry(hours: int = 1, limit: int = 1000) -> List[TelemetryEvent]:
    try:
        return graph_store.get_recent_telemetry(hours, limit)
    except Exception as exc:
        logger.exception("Error retrieving telemetry")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/metrics/current", response_model=CoherenceMetrics)
async def get_current_metrics() -> CoherenceMetrics:
    try:
        graph = graph_store.get_graph()
        telemetry = graph_store.get_recent_telemetry()
        return metrics_engine.analyze(graph, telemetry)
    except Exception as exc:
        logger.exception("Error calculating metrics")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/interventions/suggestions", response_model=List[InterventionSuggestion])
async def get_intervention_suggestions() -> List[InterventionSuggestion]:
    try:
        graph = graph_store.get_graph()
        degrees: Dict[str, int] = {}
        for link in graph.links:
            degrees[link.source_id] = degrees.get(link.source_id, 0) + 1
            degrees[link.target_id] = degrees.get(link.target_id, 0) + 1

        suggestions: List[InterventionSuggestion] = []
        hotspots = [node_id for node_id, degree in degrees.items() if degree > 5]
        if hotspots:
            suggestions.append(
                InterventionSuggestion(
                    intervention_type="split_hotspot",
                    target_nodes=hotspots[:3],
                    priority=0.8,
                    description="Split high-connectivity nodes to reduce processing bottlenecks",
                    estimated_impact={"stress_reduction": 0.3, "coherence_gain": 0.15},
                )
            )

        weak_links = [link for link in graph.links if link.weight < 0.3]
        if weak_links and len(weak_links) > len(graph.links) * 0.2:
            weak_targets = sorted(
                {link.source_id for link in weak_links[:5]} | {link.target_id for link in weak_links[:5]}
            )
            suggestions.append(
                InterventionSuggestion(
                    intervention_type="strengthen_links",
                    target_nodes=weak_targets,
                    priority=0.6,
                    description="Strengthen weak semantic connections to reduce entropy leak",
                    estimated_impact={"entropy_reduction": 0.2, "coherence_gain": 0.1},
                )
            )

        suggestions.append(
            InterventionSuggestion(
                intervention_type="reorder_queue",
                target_nodes=[],
                priority=0.4,
                description="Optimize processing queue based on current stress patterns",
                estimated_impact={"stress_reduction": 0.15, "throughput_gain": 0.1},
            )
        )
        return suggestions
    except Exception as exc:
        logger.exception("Error generating interventions")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/bridge/analyze", response_model=BridgeKernelResponse)
async def bridge_analyze(request: BridgeKernelRequest) -> BridgeKernelResponse:
    try:
        metrics = metrics_engine.analyze(request.graph_snapshot, request.telemetry_window)
        interventions = [
            InterventionSuggestion(
                intervention_type="context_refresh",
                target_nodes=[node.id for node in request.graph_snapshot.nodes[:2]],
                priority=0.9,
                description="Refresh context windows showing high drift",
                estimated_impact={"coherence_gain": 0.25, "entropy_reduction": 0.4},
            )
        ]
        return BridgeKernelResponse(
            metrics=metrics,
            interventions=interventions,
            kernel_version="demo-open-v0.1",
        )
    except Exception as exc:
        logger.exception("Bridge API error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/import/obsidian")
async def import_obsidian(file: UploadFile = File(...)) -> Dict[str, str]:
    del file
    return {"status": "not_implemented", "message": "Obsidian import coming in Phase 1"}


@app.post("/import/pdf")
async def import_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    del file
    return {"status": "not_implemented", "message": "PDF import coming in Phase 1"}


@app.post("/import/csv")
async def import_csv(file: UploadFile = File(...)) -> Dict[str, str]:
    del file
    return {"status": "not_implemented", "message": "CSV import coming in Phase 1"}


if __name__ == "__main__":
    logger.info("Starting Dynamic Sentience Maps server")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
