“””
Dynamic Sentience Maps - FastAPI Server
Core server for graph processing, telemetry, and Bridge API
“””

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import asyncio
import numpy as np
import json
import sqlite3
import uuid
from datetime import datetime, timedelta
import math
from dataclasses import dataclass
from pathlib import Path
import logging
import uvicorn

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

app = FastAPI(
title=“Dynamic Sentience Maps API”,
description=“Open-core telemetry and graph processing for living mind-maps”,
version=“0.1.0”
)

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[”*”],
allow_headers=[”*”],
)

# ============================================================================

# Data Models

# ============================================================================

class NodeData(BaseModel):
id: str
name: str
content: Optional[str] = None
node_type: str = “concept”
embedding: Optional[List[float]] = None
metadata: Dict[str, Any] = Field(default_factory=dict)
created_at: datetime = Field(default_factory=datetime.now)

class LinkData(BaseModel):
source_id: str
target_id: str
link_type: str = “semantic”
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
phi_star: float = Field(…, description=“Φ* coherence measure”)
kappa: float = Field(…, description=“κ stress measure”)
epsilon: float = Field(…, description=“ε entropy leak”)
i_capacity: float = Field(…, description=“I_c information capacity”)
timestamp: datetime = Field(default_factory=datetime.now)

class InterventionSuggestion(BaseModel):
intervention_type: str
target_nodes: List[str]
priority: float
description: str
estimated_impact: Dict[str, float]

# Bridge API Models

class BridgeKernelRequest(BaseModel):
graph_snapshot: GraphData
telemetry_window: List[TelemetryEvent]
analysis_params: Dict[str, Any] = Field(default_factory=dict)

class BridgeKernelResponse(BaseModel):
metrics: CoherenceMetrics
interventions: List[InterventionSuggestion]
kernel_version: str = “demo-open-v0.1”

# ============================================================================

# Core Graph Store

# ============================================================================

class GraphStore:
def **init**(self, db_path: str = “dsm_graph.db”):
self.db_path = db_path
self.init_database()

```
def init_database(self):
    """Initialize SQLite database with schema"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Nodes table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            content TEXT,
            node_type TEXT DEFAULT 'concept',
            embedding TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Links table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS links (
            id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            link_type TEXT DEFAULT 'semantic',
            weight REAL DEFAULT 1.0,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_id) REFERENCES nodes(id),
            FOREIGN KEY (target_id) REFERENCES nodes(id)
        )
    """)
    
    # Telemetry table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id TEXT PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            node_id TEXT,
            metrics TEXT NOT NULL,
            context TEXT,
            FOREIGN KEY (node_id) REFERENCES nodes(id)
        )
    """)
    
    conn.commit()
    conn.close()

def add_node(self, node: NodeData) -> str:
    """Add node to graph store"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    node_id = node.id if node.id else str(uuid.uuid4())
    
    cursor.execute("""
        INSERT OR REPLACE INTO nodes 
        (id, name, content, node_type, embedding, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        node_id,
        node.name,
        node.content,
        node.node_type,
        json.dumps(node.embedding) if node.embedding else None,
        json.dumps(node.metadata)
    ))
    
    conn.commit()
    conn.close()
    return node_id

def add_link(self, link: LinkData) -> str:
    """Add link to graph store"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    link_id = str(uuid.uuid4())
    
    cursor.execute("""
        INSERT INTO links 
        (id, source_id, target_id, link_type, weight, metadata)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        link_id,
        link.source_id,
        link.target_id,
        link.link_type,
        link.weight,
        json.dumps(link.metadata)
    ))
    
    conn.commit()
    conn.close()
    return link_id

def get_graph(self, limit: int = 1000) -> GraphData:
    """Retrieve graph data"""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Get nodes
    cursor.execute("SELECT * FROM nodes LIMIT ?", (limit,))
    node_rows = cursor.fetchall()
    nodes = [
        NodeData(
            id=row[0],
            name=row[1],
            content=row[2],
            node_type=row[3],
            embedding=json.loads(row[4]) if row[4] else None,
            metadata=json.loads(row[5]) if row[5] else {}
        )
        for row in node_rows
    ]
    
    # Get links
    cursor.execute("SELECT * FROM links LIMIT ?", (limit * 2,))
    link_rows = cursor.fetchall()
    links = [
        LinkData(
            source_id=row[1],
            target_id=row[2],
            link_type=row[3],
            weight=row[4],
            metadata=json.loads(row[6]) if row[6] else {}
        )
        for row in link_rows
    ]
    
    conn.close()
    return GraphData(nodes=nodes, links=links)
```

# ============================================================================

# Open Metrics Estimators (Demo Implementation)

# ============================================================================

class OpenMetricsEngine:
“”“Open-source demo estimators for κ/ε/I_c/Φ*”””

```
def __init__(self):
    self.baseline_capacity = 1000.0
    self.stress_history = []
    self.entropy_history = []

def calculate_stress(self, graph: GraphData, telemetry: List[TelemetryEvent]) -> float:
    """Calculate κ (stress) from graph topology and telemetry"""
    if not graph.nodes:
        return 0.0
    
    # Simple stress calculation based on node degree and processing load
    node_degrees = {}
    for link in graph.links:
        node_degrees[link.source_id] = node_degrees.get(link.source_id, 0) + 1
        node_degrees[link.target_id] = node_degrees.get(link.target_id, 0) + 1
    
    # Calculate stress from high-degree nodes and recent telemetry
    degree_stress = sum(min(degree ** 1.5, 10) for degree in node_degrees.values()) / len(graph.nodes)
    
    # Add telemetry-based stress
    recent_events = [e for e in telemetry if e.timestamp > datetime.now() - timedelta(minutes=5)]
    telemetry_stress = sum(e.metrics.get('processing_time', 0) for e in recent_events) / max(len(recent_events), 1)
    
    total_stress = degree_stress * 0.7 + telemetry_stress * 0.3
    self.stress_history.append(total_stress)
    
    return min(total_stress, 10.0)  # Cap at 10

def calculate_entropy_leak(self, graph: GraphData, telemetry: List[TelemetryEvent]) -> float:
    """Calculate ε (entropy leak) from information drift"""
    if not graph.nodes or len(graph.nodes) < 2:
        return 0.0
    
    # Simulate entropy leak based on weak connections and contradictions
    weak_links = [l for l in graph.links if l.weight < 0.3]
    weak_connection_entropy = len(weak_links) / max(len(graph.links), 1)
    
    # Add contradiction detection from telemetry
    contradiction_events = [e for e in telemetry if e.event_type == 'contradiction']
    contradiction_entropy = len(contradiction_events) / max(len(telemetry), 1)
    
    total_entropy = weak_connection_entropy * 0.6 + contradiction_entropy * 0.4
    self.entropy_history.append(total_entropy)
    
    return min(total_entropy, 1.0)  # Cap at 1.0

def calculate_capacity(self, graph: GraphData) -> float:
    """Calculate I_c (information capacity)"""
    if not graph.nodes:
        return 0.0
    
    # Estimate capacity from graph connectivity and node richness
    avg_degree = sum(len([l for l in graph.links if l.source_id == node.id or l.target_id == node.id]) 
                     for node in graph.nodes) / len(graph.nodes)
    
    # Factor in content richness
    avg_content_length = sum(len(node.content or "") for node in graph.nodes) / len(graph.nodes)
    content_factor = min(avg_content_length / 100, 2.0)  # Normalize to 0-2 range
    
    capacity = self.baseline_capacity * (1 + avg_degree * 0.1) * content_factor
    return capacity

def calculate_coherence(self, stress: float, entropy: float, capacity: float) -> float:
    """Calculate Φ* (coherence) from other metrics"""
    # Coherence decreases with stress and entropy, increases with capacity
    normalized_capacity = min(capacity / self.baseline_capacity, 2.0)
    
    # Use sigmoid-like function for smooth coherence calculation
    coherence = normalized_capacity / (1 + stress * 0.3 + entropy * 2.0)
    
    # Apply smoothing based on history
    if len(self.stress_history) > 1:
        stress_stability = 1.0 - abs(self.stress_history[-1] - self.stress_history[-2]) / 10.0
        entropy_stability = 1.0 - abs(self.entropy_history[-1] - self.entropy_history[-2])
        coherence *= (stress_stability + entropy_stability) / 2
    
    return max(0.1, min(coherence, 1.0))  # Clamp to 0.1-1.0 range
```

# ============================================================================

# Global Instances

# ============================================================================

graph_store = GraphStore()
metrics_engine = OpenMetricsEngine()

# ============================================================================

# API Endpoints

# ============================================================================

@app.get(”/”)
async def root():
“”“Health check and API info”””
return {
“service”: “Dynamic Sentience Maps API”,
“version”: “0.1.0”,
“status”: “operational”,
“features”: [“graph_storage”, “telemetry”, “open_metrics”, “bridge_api”]
}

@app.post(”/graph/nodes”, response_model=dict)
async def add_node(node: NodeData):
“”“Add a new node to the graph”””
try:
node_id = graph_store.add_node(node)
logger.info(f”Added node: {node_id} - {node.name}”)
return {“node_id”: node_id, “status”: “created”}
except Exception as e:
logger.error(f”Error adding node: {str(e)}”)
raise HTTPException(status_code=500, detail=str(e))

@app.post(”/graph/links”, response_model=dict)
async def add_link(link: LinkData):
“”“Add a new link to the graph”””
try:
link_id = graph_store.add_link(link)
logger.info(f”Added link: {link.source_id} -> {link.target_id}”)
return {“link_id”: link_id, “status”: “created”}
except Exception as e:
logger.error(f”Error adding link: {str(e)}”)
raise HTTPException(status_code=500, detail=str(e))

@app.get(”/graph”, response_model=GraphData)
async def get_graph(limit: int = 1000):
“”“Retrieve the current graph state”””
try:
graph_data = graph_store.get_graph(limit)
logger.info(f”Retrieved graph: {len(graph_data.nodes)} nodes, {len(graph_data.links)} links”)
return graph_data
except Exception as e:
logger.error(f”Error retrieving graph: {str(e)}”)
raise HTTPException(status_code=500, detail=str(e))

@app.post(”/telemetry/events”)
async def log_telemetry(events: List[TelemetryEvent]):
“”“Log telemetry events”””
try:
conn = sqlite3.connect(graph_store.db_path)
cursor = conn.cursor()

```
    for event in events:
        event_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO telemetry (id, timestamp, event_type, node_id, metrics, context)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            event.timestamp,
            event.event_type,
            event.node_id,
            json.dumps(event.metrics),
            json.dumps(event.context)
        ))
    
    conn.commit()
    conn.close()
    
    logger.info(f"Logged {len(events)} telemetry events")
    return {"events_logged": len(events), "status": "success"}
except Exception as e:
    logger.error(f"Error logging telemetry: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/telemetry/recent”, response_model=List[TelemetryEvent])
async def get_recent_telemetry(hours: int = 1, limit: int = 1000):
“”“Retrieve recent telemetry events”””
try:
conn = sqlite3.connect(graph_store.db_path)
cursor = conn.cursor()

```
    cutoff_time = datetime.now() - timedelta(hours=hours)
    cursor.execute("""
        SELECT * FROM telemetry 
        WHERE timestamp > ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (cutoff_time, limit))
    
    rows = cursor.fetchall()
    events = [
        TelemetryEvent(
            timestamp=datetime.fromisoformat(row[1]),
            event_type=row[2],
            node_id=row[3],
            metrics=json.loads(row[4]),
            context=json.loads(row[5]) if row[5] else {}
        )
        for row in rows
    ]
    
    conn.close()
    return events
except Exception as e:
    logger.error(f"Error retrieving telemetry: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/metrics/current”, response_model=CoherenceMetrics)
async def get_current_metrics():
“”“Calculate and return current system metrics”””
try:
# Get current graph and telemetry
graph_data = graph_store.get_graph()

```
    conn = sqlite3.connect(graph_store.db_path)
    cursor = conn.cursor()
    
    # Get recent telemetry (last hour)
    cutoff_time = datetime.now() - timedelta(hours=1)
    cursor.execute("""
        SELECT * FROM telemetry 
        WHERE timestamp > ? 
        ORDER BY timestamp DESC 
        LIMIT 1000
    """, (cutoff_time,))
    
    telemetry_rows = cursor.fetchall()
    telemetry_events = [
        TelemetryEvent(
            timestamp=datetime.fromisoformat(row[1]),
            event_type=row[2],
            node_id=row[3],
            metrics=json.loads(row[4]),
            context=json.loads(row[5]) if row[5] else {}
        )
        for row in telemetry_rows
    ]
    
    conn.close()
    
    # Calculate metrics using open estimators
    stress = metrics_engine.calculate_stress(graph_data, telemetry_events)
    entropy = metrics_engine.calculate_entropy_leak(graph_data, telemetry_events)
    capacity = metrics_engine.calculate_capacity(graph_data)
    coherence = metrics_engine.calculate_coherence(stress, entropy, capacity)
    
    metrics = CoherenceMetrics(
        phi_star=coherence,
        kappa=stress,
        epsilon=entropy,
        i_capacity=capacity
    )
    
    logger.info(f"Calculated metrics: Φ*={coherence:.3f}, κ={stress:.2f}, ε={entropy:.3f}, I_c={capacity:.0f}")
    return metrics
    
except Exception as e:
    logger.error(f"Error calculating metrics: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/interventions/suggestions”, response_model=List[InterventionSuggestion])
async def get_intervention_suggestions():
“”“Generate intervention suggestions based on current system state”””
try:
# Get current metrics and graph
graph_data = graph_store.get_graph()

```
    # Simple intervention logic (demo implementation)
    suggestions = []
    
    # High-degree nodes (potential hotspots)
    node_degrees = {}
    for link in graph_data.links:
        node_degrees[link.source_id] = node_degrees.get(link.source_id, 0) + 1
        node_degrees[link.target_id] = node_degrees.get(link.target_id, 0) + 1
    
    high_degree_nodes = [node_id for node_id, degree in node_degrees.items() if degree > 5]
    if high_degree_nodes:
        suggestions.append(InterventionSuggestion(
            intervention_type="split_hotspot",
            target_nodes=high_degree_nodes[:3],  # Top 3
            priority=0.8,
            description="Split high-connectivity nodes to reduce processing bottlenecks",
            estimated_impact={"stress_reduction": 0.3, "coherence_gain": 0.15}
        ))
    
    # Weak links (entropy sources)
    weak_links = [l for l in graph_data.links if l.weight < 0.3]
    if len(weak_links) > len(graph_data.links) * 0.2:  # >20% weak links
        weak_targets = list(set([l.source_id for l in weak_links[:5]] + [l.target_id for l in weak_links[:5]]))
        suggestions.append(InterventionSuggestion(
            intervention_type="strengthen_links",
            target_nodes=weak_targets,
            priority=0.6,
            description="Strengthen weak semantic connections to reduce entropy leak",
            estimated_impact={"entropy_reduction": 0.2, "coherence_gain": 0.1}
        ))
    
    # Queue reordering suggestion (always available)
    suggestions.append(InterventionSuggestion(
        intervention_type="reorder_queue",
        target_nodes=[],
        priority=0.4,
        description="Optimize processing queue based on current stress patterns",
        estimated_impact={"stress_reduction": 0.15, "throughput_gain": 0.1}
    ))
    
    return suggestions
    
except Exception as e:
    logger.error(f"Error generating interventions: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# ============================================================================

# Bridge API (for private kernel integration)

# ============================================================================

@app.post(”/bridge/analyze”, response_model=BridgeKernelResponse)
async def bridge_analyze(request: BridgeKernelRequest):
“”“Bridge API endpoint for private kernel analysis”””
try:
# This is the demo implementation using open estimators
# In production, this would call the private compiled kernel

```
    logger.info("Bridge API called - using demo open estimators")
    
    # Use open metrics engine
    stress = metrics_engine.calculate_stress(request.graph_snapshot, request.telemetry_window)
    entropy = metrics_engine.calculate_entropy_leak(request.graph_snapshot, request.telemetry_window)
    capacity = metrics_engine.calculate_capacity(request.graph_snapshot)
    coherence = metrics_engine.calculate_coherence(stress, entropy, capacity)
    
    metrics = CoherenceMetrics(
        phi_star=coherence,
        kappa=stress,
        epsilon=entropy,
        i_capacity=capacity
    )
    
    # Generate enhanced interventions (simulated)
    interventions = [
        InterventionSuggestion(
            intervention_type="context_refresh",
            target_nodes=[node.id for node in request.graph_snapshot.nodes[:2]],
            priority=0.9,
            description="Refresh context windows showing high drift (KL divergence > 0.3)",
            estimated_impact={"coherence_gain": 0.25, "entropy_reduction": 0.4}
        )
    ]
    
    response = BridgeKernelResponse(
        metrics=metrics,
        interventions=interventions,
        kernel_version="demo-open-v0.1"
    )
    
    logger.info(f"Bridge analysis complete: Φ*={coherence:.3f}")
    return response
    
except Exception as e:
    logger.error(f"Bridge API error: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# ============================================================================

# Import Endpoints (Stubs for MVP)

# ============================================================================

@app.post(”/import/obsidian”)
async def import_obsidian(file: UploadFile = File(…)):
“”“Import Obsidian vault (markdown files)”””
# TODO: Implement Obsidian import logic
return {“status”: “not_implemented”, “message”: “Obsidian import coming in Phase 1”}

@app.post(”/import/pdf”)
async def import_pdf(file: UploadFile = File(…)):
“”“Import PDF document”””
# TODO: Implement PDF text extraction and graph generation
return {“status”: “not_implemented”, “message”: “PDF import coming in Phase 1”}

@app.post(”/import/csv”)
async def import_csv(file: UploadFile = File(…)):
“”“Import CSV data”””
# TODO: Implement CSV parsing and node/link generation
return {“status”: “not_implemented”, “message”: “CSV import coming in Phase 1”}

# ============================================================================

# Development Server

# ============================================================================

if **name** == “**main**”:
logger.info(“Starting Dynamic Sentience Maps server…”)
uvicorn.run(
“main:app”,
host=“0.0.0.0”,
port=8000,
reload=True,
log_level=“info”
