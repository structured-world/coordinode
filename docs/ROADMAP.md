---
description: "What CoordiNode ships today, what is being built next, and what is further out: query engine, vector search, replication, Cypher DDL, Bolt protocol and SQL over the same engine."
---

# CoordiNode Roadmap

Public roadmap in Now / Next / Later format.

For feature requests and bug reports, use [GitHub Issues](https://github.com/structured-world/coordinode/issues).

---

## Now (v0.5.x, current release)

These features are implemented, tested, and available today.

**Query Engine**
- OpenCypher read + write (MATCH, CREATE, MERGE, DELETE, SET, REMOVE, WITH, UNWIND, OPTIONAL MATCH)
- Variable-length path queries (`*1..N`), shortest path
- Aggregation (count, sum, avg, min, max, collect, percentile)
- MVCC transactions with Snapshot Isolation and optimistic conflict detection
- Time-travel queries (AS OF TIMESTAMP, 7-day retention)
- EXPLAIN with cost estimation

**Vector Search**
- HNSW index up to 65,536 dimensions
- SQ8 scalar quantization (4x memory reduction) and RaBitQ binary codes with exact rerank
- Distance metrics: cosine, L2, dot product, L1
- Vector search over relationship properties as well as node properties
- Hybrid graph traversal + vector filter in a single query
- `maxsim_score()` for ColBERT-style late-interaction relevance

**Full-Text Search**
- BM25 scoring with fuzzy, phrase, and wildcard queries
- 23+ built-in languages (Snowball stemmers)
- CJK support (Chinese, Japanese, Korean) via feature flags
- Per-field analyzer configuration

**Spatial**
- `point({latitude, longitude})` constructor
- `point.distance()` with Haversine formula
- Spatial predicates in WHERE clauses

**Document Properties**
- Nested DOCUMENT type (arbitrary JSON/MessagePack depth)
- Dot-notation property access (`n.config.network.ssid`)
- Three schema modes: STRICT, VALIDATED, FLEXIBLE

**Indexes**
- B-tree: single-field, compound, unique, partial, sparse
- TTL index with automatic background expiration
- Online index build (zero-downtime)

**Security**
- Searchable symmetric encryption (AES-256-GCM + HMAC-SHA256)
- Equality search on encrypted fields

**Operations**
- Built-in query advisor: EXPLAIN SUGGEST with 5 detectors + N+1 pattern detection
- Prometheus metrics, structured JSON logging, OTLP tracing
- Backup/restore (JSON, Cypher, binary formats)
- Docker image, embedded library mode (`coordinode-embed`)

**Document Operations**
- Path-targeted partial updates (`SET n.config.ssid = "home"` without read-modify-write)
- Array operators (push, pull, addToSet, increment) as merge operands
- Graph-document transformations: `DETACH DOCUMENT` promotes a nested property to a node and edge, `ATTACH DOCUMENT` demotes it back

**Schema and Index DDL**
- `CREATE LABEL` with typed properties, `CREATE EDGE TYPE`, and `CREATE VECTOR INDEX ... OPTIONS {m, ef_construction}` as Cypher statements
- `CREATE ENCRYPTED INDEX` with `encrypted_match()` in `WHERE`

**Entity Resolution**
- `MERGE NODES (a, b) INTO a` collapses duplicates in one MVCC transaction, with property merge rules, edge re-pointing and duplicate-edge handling

**Triggers**
- `CREATE / DROP / SHOW / ALTER TRIGGER` as first-class clauses, replicated through Raft, with cascade limits and per-trigger error policy

**Temporal**
- Bitemporal edge types: multiple `(valid_from, valid_to)` versions per pair, with time-slice predicates pushed into the scan

**Replication and Cluster**
- Multi-node Raft replication with leader election, snapshot transfer and log compaction, included in the open-source edition
- Follower reads with staleness tracking, and causal-consistency sessions
- Mutual TLS between nodes over a pure-Rust stack, with zstd compression on the replication transport
- Background scrub, with damaged segments rebuilt from a healthy replica
- Fault-injection test suite: partition matrix, crash and clock skew, linearizability checking over a live workload

**API**
- gRPC on port 7080 (native, all services)
- REST/JSON via gRPC-to-REST transcoding (port 7081, via structured-proxy)
- Operational HTTP on port 7084 (/metrics, /health, /ready)
- Parameter binding in gRPC/REST queries

---

## Next

Features in active development or planned for the next few releases.
When a feature is completed, the corresponding documentation will be updated.

**GraphQL API**
- GraphQL with auto-generated schema from graph model (SDL generation implemented, server wiring in progress)
  - *Closes:* COMPATIBILITY.md "GraphQL → Planned"

**WebSocket Subscriptions**
- Live query subscriptions via WebSocket (port 7083)
  - *Closes:* COMPATIBILITY.md "WebSocket → Planned"

**Trigger Firing**
- BEFORE and AFTER COMMIT execution for the trigger definitions the DDL already accepts
- Static cycle detection at definition time, and an auto-disable circuit breaker

**Late-Interaction Retrieval**
- Centroid-based candidate selection for `maxsim_score()`, replacing the brute-force pass over candidates

**SQL**
- PostgreSQL wire protocol over the same query IR, so SQL and Cypher reach one engine and one transaction model

---

## Later

Long-term vision. Timeline depends on community interest and sponsorship.

**Neo4j Compatibility**
- Bolt protocol (v4.3–v5.8) — existing Neo4j drivers connect without code changes
- 130+ OpenCypher functions (string, math, temporal, list, map)
- Neo4j procedures (`db.*`, `dbms.*`)
- LOAD CSV, neo4j-admin import compatibility
- Constraints (node key, existence, property type)

**Horizontal Scaling (Enterprise Edition)**
- Multi-group Raft with hash/range sharding
- CRUSH-like placement with failure domain awareness
- Scatter-gather query engine with predicate push-down
- Cross-shard 2PC transactions with HLC timestamps
- Erasure coding for storage efficiency

**Advanced Features**
- CDC (Change Data Capture) to NATS, Kafka, webhooks
- Materialized graph views with incremental refresh
- Graph analytics (PageRank, community detection, centrality)
- Visual graph explorer (Vue 3 + WebGL)
- Kubernetes operator with rolling upgrades
- Multi-tenancy with mClock QoS per tenant

---

## Contributing

We welcome contributions at any level — from bug reports to feature implementations.

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
