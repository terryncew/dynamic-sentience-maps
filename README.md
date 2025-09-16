[![OpenLine-compatible](https://img.shields.io/static/v1?label=OpenLine&message=compatible%20v0.1&color=1f6feb)](https://github.com/terryncew/openline-core)
![Schema check](https://github.com/terryncew/openline-core/actions/workflows/validate.yml/badge.svg)
**Live hub:** https://terryncew.github.io/openline-hub/

# Dynamic Sentience Maps

**Open-core telemetry and visualization for living knowledge graphs**

🧠 **Studio**: Interactive mind-map interface with real-time coherence metrics  
⚡ **Telemetry**: Live monitoring of κ (stress), ε (entropy leak), I_c (capacity), and Φ* coherence  
🔌 **Bridge API**: Drop-in support for private kernels without exposing IP

## Demo

Open `apps/studio/index.html` in your browser to see the interactive visualization in action. The demo includes:

- **Force-directed knowledge graph** with semantic clustering
- **Real-time coherence dial** showing Φ* metrics
- **Stress visualization** highlighting overloaded nodes
- **Live telemetry charts** tracking system health
- **Intervention suggestions** for maintaining optimal flow

## Architecture

This is an **open-core** system designed to protect intellectual property while enabling ecosystem growth:

### Open Components

- 📊 **Studio interface**: Full-featured graph visualization and interaction
- 🔄 **Import adapters**: Obsidian, Notion, PDF, Git, CSV processors
- 📈 **Telemetry system**: Event capture, aggregation, and visualization
- 🧮 **Demo estimators**: Open-source approximations for research/demos
- 🔗 **Graph processing**: Storage, embeddings, layouts, clustering

### Private Components (Bridge API)

- 🧠 **Production kernel**: High-fidelity coherence calculations
- ⚡ **Optimized estimators**: Proprietary stress/entropy algorithms
- 🎯 **Advanced interventions**: Sophisticated graph optimization

## Quick Start

### Studio Demo

```bash
# Just open in browser - no setup required
open apps/studio/index.html
```

### Development Server

```bash
# Install dependencies
pip install fastapi uvicorn numpy sqlite3

# Start server
cd server
python main.py

# Visit http://localhost:8000/docs for API documentation
```

### Import Your Data

The system supports multiple data sources:

- **Obsidian**: Import your vault’s markdown files and link structure
- **PDF Documents**: Extract text and generate semantic connections
- **CSV Data**: Process structured data into knowledge nodes
- **Git Repositories**: Analyze code structure and documentation

## Core Concepts

### Telemetry Metrics

- **κ (Kappa - Stress)**: Processing bottlenecks, queue depth, token rate pressure
- **ε (Epsilon - Entropy Leak)**: Information drift, contradiction rate, retrieval mismatch
- **I_c (Information Capacity)**: Baseline mutual information across stable connections
- **Φ* (Phi-star - Coherence)**: Integrated measure of system-wide information flow

### Intervention System

The platform suggests non-destructive optimizations:

- **Queue reordering** for stress hotspots
- **Context refresh** when drift is detected
- **Link strengthening** for weak semantic connections
- **Node splitting** for overloaded concepts

## Project Structure

```
dynamic-sentience-maps/
├── apps/
│   └── studio/              # React/D3 visualization interface
├── server/                  # FastAPI backend services
├── packages/
│   ├── graph-core/          # Graph storage and processing
│   ├── metrics-open/        # Open-source metric estimators
│   ├── adapters/           # Data import/export adapters
│   └── plugins/            # Optional extensions (RAG, MI probes)
├── specs/
│   ├── telemetry.md        # Event schema documentation
│   └── bridge-api.md       # Private kernel interface spec
└── docs/                   # Architecture and usage guides
```

## Roadmap

### Phase 1: Foundation (Weeks 1-2)

- ✅ Core visualization with D3/React
- ✅ Basic telemetry capture and display
- 🔄 Import adapters for common formats
- 🔄 SQLite graph storage with embeddings

### Phase 2: Intelligence (Weeks 3-6)

- 📋 Stress/entropy simulation and analysis
- 📋 Intervention suggestion engine
- 📋 Neo4j/Memgraph integration
- 📋 Mutual information probes

### Phase 3: Production (Weeks 7-8)

- 📋 Bridge API client implementation
- 📋 Export capabilities (PNG/SVG/PDF reports)
- 📋 Security hardening and ACL system
- 📋 Performance optimization

## Security & Privacy

- **Local-first**: SQLite + local vector storage by default
- **Redaction**: Automatic PII detection and filtering at ingest
- **Aggregate telemetry**: No raw data export without explicit consent
- **Boundary ACLs**: Fine-grained access control for sensitive nodes

## Contributing

We welcome contributions! Please see:

- `CONTRIBUTING.md` for development guidelines
- `CODE_OF_CONDUCT.md` for community standards
- GitHub Issues for bug reports and feature requests

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-username/dynamic-sentience-maps.git
cd dynamic-sentience-maps

# Install dependencies
make setup

# Start development environment
make dev
```

## License

- **Studio & Packages**: Apache-2.0 (ecosystem-friendly, commercial use allowed)
- **Server**: AGPL-3.0 (open-source, prevents proprietary SaaS clones)
- **Private Bridge Kernel**: Separate commercial license

## Links

- 📖 [Documentation](./docs/)
- 🐛 [Report Issues](https://github.com/your-username/dynamic-sentience-maps/issues)
- 💬 [Discussions](https://github.com/your-username/dynamic-sentience-maps/discussions)
- 🌐 [Live Demo](https://your-username.github.io/dynamic-sentience-maps/)

-----

**Dynamic Sentience Maps** - Making knowledge systems more transparent, coherent, and aligned with human understanding.

*Built with ❤️ for researchers, knowledge workers, and AI safety enthusiasts.*
