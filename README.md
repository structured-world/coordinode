# CoordiNode

[![CI](https://github.com/structured-world/coordinode/actions/workflows/ci.yml/badge.svg)](https://github.com/structured-world/coordinode/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**The graph-native hybrid retrieval engine for AI and GraphRAG.**

Graph + Vector + Full-Text retrieval in a single transactional engine.

Built in Rust. Zero GC. Single binary. OpenCypher-compatible.

---

## The Problem

Building relationship-aware AI today requires duct tape. You need Neo4j for the graph, Pinecone for vectors, and Elasticsearch for text. That means 3 systems, glue code, data sync pipelines, and no transactional consistency across your data.

## The Solution: One Engine, One Query

CoordiNode unifies graph traversal, vector similarity search, and full-text retrieval in one engine with one query language (OpenCypher-compatible) and one transaction model (MVCC, Snapshot Isolation).

### The Magic Moment

Traverse a knowledge graph, filter by semantic similarity, rank by text match — one query, one transaction:

```cypher
MATCH (topic:Concept {name: "machine learning"})-[:RELATED_TO*1..3]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $question_vector) < 0.4
  AND text_match(doc.body, "transformer attention mechanism")
RETURN doc.title,
       vector_distance(doc.embedding, $question_vector) AS relevance,
       text_score(doc.body, "transformer attention mechanism") AS text_rank
ORDER BY relevance LIMIT 10
```

Today this requires Neo4j + Pinecone + Elasticsearch + custom glue code. With CoordiNode — one query.

---

## Is CoordiNode Right for You? (v0.3-alpha)

### Use this today if:

- You are building **GraphRAG**, knowledge retrieval, or relationship-heavy AI apps
- You need **graph + vector + text** queries in a single transaction (no glue code)
- You want to replace a fragile multi-database stack with a single binary
- You need features no other graph DB offers: **vectors on edges**, spatial queries, encrypted search (SSE, programmatic API), time-travel queries, EXPLAIN SUGGEST

### Do not use this yet if:

- You need a 100% drop-in replacement for a mature Neo4j Enterprise deployment
- Your application relies on APOC procedures, Neo4j Browser/Bloom, or GDS
- You need native Bolt protocol for existing Neo4j drivers (planned for v1.2; gRPC and REST available now, GraphQL planned)
- You need production-grade multi-node clustering today (single-node is stable; Raft clustering is in active development for v0.4)

---

## Who This Is For

**GraphRAG and enterprise knowledge retrieval** — traverse knowledge graphs, filter by semantic similarity, rank by text relevance. One engine replaces Neo4j + vector DB + search engine.

**Fraud detection and threat intelligence** — detect fraud rings through shared-device graphs with behavioral embedding similarity. Correlate attack patterns across MITRE ATT&CK with vector + text search on indicators.

**Recommendations and social discovery** — traverse social graphs, find items semantically similar to user preferences. Edge properties (ratings, timestamps) filterable in the same query.

<details>
<summary>See example queries for each use case</summary>

### Fraud Ring Detection

```cypher
MATCH (suspect:Account {flagged: true})-[:SHARES_DEVICE*1..3]-(connected:Account)
WHERE vector_distance(suspect.tx_embedding, connected.tx_embedding) < 0.15
  AND connected.flagged = false
RETURN connected.id, connected.holder_name,
       vector_distance(suspect.tx_embedding, connected.tx_embedding) AS similarity
ORDER BY similarity LIMIT 50
```

### Semantic Recommendation

```cypher
MATCH (me:User {id: $userId})-[:FOLLOWS*1..2]->(friend)
MATCH (friend)-[:PURCHASED]->(item:Product)
WHERE NOT (me)-[:PURCHASED]->(item)
  AND vector_distance(item.embedding, $user_taste_vector) < 0.3
RETURN DISTINCT item.name, item.category,
       vector_distance(item.embedding, $user_taste_vector) AS match_score
ORDER BY match_score LIMIT 20
```

### Threat Intelligence

```cypher
MATCH (malware:Indicator {hash: $sample_hash})-[:USES]->(technique:AttackTechnique)
MATCH (technique)<-[:USES]-(similar:Indicator)
WHERE vector_distance(similar.behavior_embedding, malware.behavior_embedding) < 0.2
  AND text_match(similar.description, $ioc_keywords)
RETURN similar.name, technique.mitre_id,
       vector_distance(similar.behavior_embedding, malware.behavior_embedding) AS similarity
ORDER BY similarity LIMIT 25
```

</details>

---

## What Works Today (v0.3-alpha)

| Capability | Status | Details |
|-----------|--------|---------|
| OpenCypher read + write | **Stable** | MATCH, CREATE, MERGE, DELETE, SET, REMOVE, WITH, UNWIND |
| MVCC transactions | **Stable** | Snapshot Isolation, write conflict detection (OCC) |
| HNSW vector search | **Stable** | Up to 65536 dims, SQ8 quantization, cosine/L2/dot/L1 |
| Full-text search | **Stable** | BM25, fuzzy, phrase, 23+ languages, CJK via feature flags |
| Hybrid graph+vector+text | **Stable** | Compound WHERE predicates split into optimized pipeline; `hybrid_score(node, query [,weights])` opinionated blend helper (default 0.65·vector + 0.35·text) |
| B-tree indexes | **Stable** | Single, compound, unique, partial, TTL, sparse |
| Edge properties | **Stable** | CREATE with props, WHERE filter, inline pattern filter |
| gRPC API | **Stable** | Port 7080, tonic-based, all services |
| Operational HTTP | **Stable** | Port 7084: /metrics, /health, /ready |
| Encrypted search (SSE) | **Stable** | AES-256-GCM + HMAC-SHA256 equality search (programmatic API; Cypher DDL planned) |
| Time-travel queries | **Stable** | AS OF TIMESTAMP, 7-day retention |
| Query advisor | **Stable** | EXPLAIN SUGGEST with 5 detectors, N+1 detection |
| Spatial queries | **Stable** | `point()`, `point.distance()` (Haversine), WHERE filter |
| Document properties | **Stable** | Nested DOCUMENT type, dot-notation access, 3 schema modes |
| Document ↔ graph transformations | **Stable** | `DETACH DOCUMENT` promotes a nested property to a node + edge atomically; `ATTACH DOCUMENT` demotes a node back into a nested DOCUMENT property; optional `TRANSFER EDGES`, `ON CONFLICT REPLACE`, `ON REMAINING FAIL` |
| REST API | **Stable** | HTTP/JSON on port 7081 via gRPC-to-REST transcoding |
| Read/write concerns | **Stable** | local, majority, linearizable, causal sessions |

| Planned | Target | Notes |
|---------|--------|-------|
| GraphQL API | v0.3.1 | Auto-generated schema on port 7083 |
| 3-node Raft clustering | v0.4 | Free in CE (no per-node licensing) |
| Bolt protocol | v1.2 | Neo4j drivers connect without code changes |

## What Makes CoordiNode Different

| | CoordiNode | Neo4j CE | MongoDB | SurrealDB | Pinecone |
|---|:---:|:---:|:---:|:---:|:---:|
| Graph + vector + text in one query | **Yes** | No | Partial ($graphLookup + Atlas) | Partial (no FTS in graph) | No graph |
| Nested document properties | **Yes** (dot-notation) | No (flat props) | Yes | Yes | No |
| Vector search on edges | **Yes** | No | No | No | N/A |
| Spatial queries | **Yes** | Via APOC | Yes | No | No |
| Encrypted search | **Yes** | No | No | No | No |
| Time-travel queries | **Yes** | No | No | No | No |
| Built-in query advisor | **Yes** | No | Yes (explain) | No | No |
| OpenCypher queries | **Yes** | Yes | No | SurrealQL | No |
| Language | Rust (zero GC) | Java (JVM) | C++ | Rust | Cloud-only |
| License | AGPL-3.0 | GPL-3.0 | SSPL | BSL-1.1 | Proprietary |

## Full-Text Search: 23+ Languages

Built-in stemming for: Arabic, Armenian, Danish, Dutch, English, Finnish, French, German, Greek, Hungarian, Italian, Norwegian, Portuguese, Romanian, Russian, Spanish, Swedish, Tamil, Turkish, Ukrainian (20 languages via Snowball).

CJK (Chinese, Japanese, Korean) via feature flags: `cjk-zh`, `cjk-ja`, `cjk-ko`.

Auto-detection of document language with per-field analyzer configuration.

## Quick Start

```bash
# Option 1: Docker
git clone https://github.com/structured-world/coordinode.git
cd coordinode
docker compose up -d
curl http://localhost:7084/health

# Option 2: Build from source
cargo build --release
./target/release/coordinode serve --addr [::]:7080
curl http://localhost:7084/health
```

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for a complete 5-minute tutorial with sample data.

## Python SDK

```bash
pip install coordinode                               # core gRPC client
pip install langchain-coordinode                     # LangChain GraphStore
pip install llama-index-graph-stores-coordinode      # LlamaIndex PropertyGraphStore
```

Source: [structured-world/coordinode-python](https://github.com/structured-world/coordinode-python)

## Architecture

```
                    ┌─────────────────────────────────────────┐
  gRPC :7080 ──────▶│                                         │
  REST :7081 ──────▶│           CoordiNode Server             │
  Metrics :7084 ───▶│                                         │
                    │         (single Rust binary)            │
                    │                                         │
                    ├─────────────────────────────────────────┤
                    │  OpenCypher Parser + Query Planner      │
                    │  ┌──────┐ ┌──────┐ ┌───────┐ ┌───────┐  │
                    │  │Graph │ │Vector│ │  FTS  │ │Spatial│  │
                    │  │Engine│ │ HNSW │ │Tantivy│ │  S2   │  │
                    │  └──┬───┘ └──┬───┘ └──┬────┘ └───┬───┘  │
                    │     └────────┴────────┴──────────┘      │
                    │           LSM Storage Engine            │
                    └─────────────────────────────────────────┘
```

10 Rust crates, ~119K lines of code.

## Documentation

- [Quick Start](docs/QUICKSTART.md) — from zero to hybrid query in 5 minutes
- [Cypher Extensions](docs/cypher/extensions.md) — vector, full-text, spatial, time-travel, encrypted search syntax
- [Compatibility](docs/COMPATIBILITY.md) — Neo4j ecosystem compatibility matrix

## Known Limitations (alpha)

- **Single-node only** — clustering in development (v0.4). Single-node handles 100K-1M nodes comfortably.
- **No Bolt protocol** — use gRPC or REST. Bolt planned for v1.2.
- **No APOC/GDS** — common Cypher works; Neo4j-specific procedure libraries are not supported.
- **HNSW index is in-memory** — vector indexes reside in RAM. At 1M vectors x 384 dims = ~1.5GB RAM.
- **No benchmarks published yet** — we are working on reproducible benchmark suite.

## License

AGPL-3.0-only — genuine open source with SaaS protection.

Enterprise Edition (EE) for horizontal sharding, multi-tenancy, CRUSH placement, and geo-distribution. Contact: enterprise@sw.foundation

## Support the Project

CoordiNode is built by the [Structured World Foundation](https://sw.foundation) — a small team building the infrastructure layer for AI-native applications.

If you believe graph + vector + text should live in one engine under a genuine open-source license, consider sponsoring:

- [GitHub Sponsors](https://github.com/sponsors/structured-world)
- [Open Collective](https://opencollective.com/structured-world)

<div align="center">

![USDT TRC-20 Donation QR Code](assets/usdt-qr.svg)

USDT (TRC-20): `TFDsezHa1cBkoeZT5q2T49Wp66K8t2DmdA`

</div>

Sponsorship accelerates: Raft clustering (v0.4), Bolt protocol for Neo4j driver compatibility (v1.2), and the Enterprise Edition for horizontal scaling.

## Building from Source

```bash
git clone https://github.com/structured-world/coordinode.git
cd coordinode
cargo build --release
cargo test --workspace

# With CJK full-text support
cargo build --release --features cjk-zh,cjk-ja,cjk-ko
```

Requires Rust 1.90+.
