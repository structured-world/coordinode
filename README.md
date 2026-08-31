# CoordiNode

[![CI](https://github.com/structured-world/coordinode/actions/workflows/ci.yml/badge.svg)](https://github.com/structured-world/coordinode/actions/workflows/ci.yml)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**The graph-native hybrid retrieval engine for AI and GraphRAG.**

Graph + Vector + Full-Text retrieval in a single transactional engine.

Built in Rust. Zero GC. Single binary. OpenCypher-compatible.

---

## The Problem

Relationship-aware AI usually ends up spread across systems: a graph database for traversal, a vector store for embeddings, a search cluster for text. Each is good at its own job, and the cost lands between them: sync pipelines, duplicated identifiers, drift between stores, and no single transaction that covers a write touching all three. Retrieval quality then depends on how fresh the least-fresh copy happens to be.

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

Across a split stack this is three round trips and a join in application code. Here it is one query, planned and executed as one pipeline, inside one snapshot.

---

## Is CoordiNode Right for You? (v0.5.x)

### Use this today if:

- You are building **GraphRAG**, knowledge retrieval, or relationship-heavy AI apps
- You need **graph + vector + text** queries in a single transaction (no glue code)
- You want to replace a fragile multi-database stack with a single binary
- You want these in one engine: native vector search over both nodes and relationships, integrated with graph traversal and transactional query execution; spatial predicates; encrypted equality search; time-travel and bitemporal edges; a query advisor that reads plans back to you

### Do not use this yet if:

- You need a 100% drop-in replacement for a mature Neo4j Enterprise deployment
- Your application relies on APOC procedures, Neo4j Browser/Bloom, or GDS
- You need native Bolt protocol for existing Neo4j drivers (planned; gRPC and REST available now, GraphQL planned)
- You need a cluster with years of production mileage behind it. Raft replication, follower reads and mTLS between nodes ship today and are covered by a fault-injection suite, but the deployment history is short

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

## What Works Today (v0.5.x)

| Capability | Status | Details |
|-----------|--------|---------|
| OpenCypher read + write | **Stable** | MATCH, CREATE, MERGE, DELETE, SET, REMOVE, WITH, UNWIND |
| MVCC transactions | **Stable** | Snapshot Isolation, write conflict detection (OCC) |
| HNSW vector search | **Stable** | Up to 65536 dims, cosine/L2/dot/L1, on node and relationship properties alike |
| Vector compression | **Stable** | SQ8 scalar quantization and RaBitQ binary codes with exact rerank |
| Late-interaction scoring | **Beta** | `maxsim_score()` for ColBERT-style multi-vector relevance, brute force over candidates in this release |
| Full-text search | **Stable** | BM25, fuzzy, phrase, 23+ languages, CJK via feature flags |
| Hybrid graph+vector+text | **Stable** | Compound WHERE predicates split into optimized pipeline; `hybrid_score(node, query [,weights])` opinionated blend helper (default 0.65·vector + 0.35·text) |
| B-tree indexes | **Stable** | Single, compound, unique, partial, TTL, sparse |
| Edge properties | **Stable** | CREATE with props, WHERE filter, inline pattern filter |
| gRPC API | **Stable** | Port 7080, tonic-based, all services |
| Operational HTTP | **Stable** | Port 7084: /metrics, /health, /ready |
| Encrypted search (SSE) | **Stable** | AES-256-GCM + HMAC-SHA256 equality search over encrypted fields, via `CREATE ENCRYPTED INDEX` and `encrypted_match()` |
| Time-travel queries | **Stable** | AS OF TIMESTAMP, 7-day retention |
| Query advisor | **Stable** | EXPLAIN SUGGEST with 5 detectors, N+1 detection |
| Spatial queries | **Stable** | `point()`, `point.distance()` (Haversine), WHERE filter |
| Document properties | **Stable** | Nested DOCUMENT type, dot-notation access, 3 schema modes |
| Document ↔ graph transformations | **Stable** | `DETACH DOCUMENT` promotes a nested property to a node + edge atomically; `ATTACH DOCUMENT` demotes a node back into a nested DOCUMENT property; optional `TRANSFER EDGES`, `ON CONFLICT REPLACE`, `ON REMAINING FAIL` |
| Native entity-resolution | **Stable** | `MERGE NODES (a, b) INTO a` collapses two matched nodes in a single MVCC transaction — property merge (`KEEP FIRST` / `KEEP LAST` / `COALESCE` / `SET <exprs>`), edge re-pointing with `TRANSFER EDGES`, and duplicate-edge handling (`KEEP BOTH` / `MERGE PROPERTIES` / `KEEP TARGET`). Replaces Neo4j's APOC `mergeNodes()` with cluster-safe semantics |
| Trigger DDL | **Stable** (front-end) | `CREATE / DROP / SHOW / ALTER TRIGGER` — replicated through Raft, schema-partition storage, index keyed by `(label_or_edge_type, event)` for O(matching) lookup at 1M-trigger scale. Per-trigger `CASCADE_LIMIT` / `CASCADE_FANOUT` overrides and `ON ERROR { PROPAGATE \| RETRY n WITH BACKOFF ms \| DEAD_LETTER }`. Native first-class clause, not a plugin — replaces Neo4j APOC triggers which break in clusters. BEFORE / AFTER COMMIT firing lands in a follow-up release |
| Bitemporal edges | **Stable** | `CREATE EDGE TYPE … TEMPORAL` declares an edge type whose instances carry a `(valid_from, valid_to)` interval. Multiple versions coexist per `(src, tgt)` pair. Helpers: `temporal_active_at(r, t)`, `temporal_overlaps(r, t0, t1)`. Planner pushes time-slice predicates into a bounded prefix scan |
| REST API | **Stable** | HTTP/JSON on port 7081 via gRPC-to-REST transcoding |
| Read/write concerns | **Stable** | local, majority, linearizable, causal sessions |
| Raft replication | **Stable** | Multi-node replicated writes, leader election, snapshot transfer, log compaction. Included in CE with no per-node licensing |
| Follower reads | **Stable** | Reads served from replicas under the requested consistency level |
| Inter-node TLS and mTLS | **Stable** | Pure-Rust rustls stack, no OpenSSL and no C dependency |
| Inter-node compression | **Stable** | zstd on the replication transport, level configurable |
| Consistency test suite | **Stable** | In-process fault injection: partition matrix, crash and clock skew, linearizability checking over a live workload |
| Scrub and repair | **Stable** | Background checksum verification, damaged segments rebuilt from a healthy replica |
| Embedded engine | **Stable** | `coordinode-embed` runs the full engine in-process, no server; also exposed to Python as `coordinode-embedded` |

| Planned | Notes |
|---------|-------|
| GraphQL API | Auto-generated schema, SDL generation already in tree |
| Bolt protocol | Neo4j drivers connect without code changes |
| SQL over the same engine | PostgreSQL wire protocol against the shared query IR |

## What Makes CoordiNode Different

Other engines combine some of these. What we optimise for is the combination holding under one transaction, one planner and one storage engine:

- **One planner over all three modalities.** A compound `WHERE` mixing traversal, vector distance and text match is split into one pipeline and costed as a whole, rather than executed as three lookups joined by your application.
- **One snapshot.** Graph edges, embeddings and the text index move together under MVCC snapshot isolation, so a retrieval never reads a half-applied write.
- **Vector search over relationships as well as nodes**, integrated with traversal and transactional execution.
- **Rust with no garbage collector.** Tail latency is a design constraint, not a tuning exercise: no JVM pauses, no stop-the-world.
- **Pure Rust with no FFI.** `cargo build` produces the whole engine, compression and TLS included. No OpenSSL, no C storage engine, nothing to reconcile with your base image.
- **AGPL-3.0 with clustering included.** Replication is not held back for a paid tier.
- **Operations answered in-engine:** entity resolution (`MERGE NODES`), document promotion and demotion (`DETACH` / `ATTACH DOCUMENT`) and triggers are native clauses, not plugins that break once you cluster.

For measured comparisons rather than claims, see the [benchmarks](https://docs.coordinode.com/benchmarks/): CoordiNode and the systems we compare against run on the same host, results are JSON-recorded with a hardware fingerprint and commit SHA, and CoordiNode's numbers are regenerated by CI on every push.

Coming from Neo4j? [docs/cypher/compatibility.md](docs/cypher/compatibility.md) lists clause by clause what carries over, what is spelled differently, and what is missing.

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
pip install coordinode-embedded                      # in-process engine, no server
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

20 Rust crates, ~232K lines of code. A multi-node deployment runs the same binary on every node: replication and routing are built in, with no separate router or coordinator process to operate.

## Documentation

- [Quick Start](docs/QUICKSTART.md): from zero to a hybrid query in 5 minutes
- [Cypher Extensions](docs/cypher/extensions.md): vector, full-text, spatial, time-travel and encrypted-search syntax
- [Neo4j compatibility](docs/cypher/compatibility.md): clause-by-clause matrix, including what has no equivalent here
- [Configuration](docs/guide/configuration.md): every tunable, its default, and whether it needs a restart
- [Benchmarks](https://docs.coordinode.com/benchmarks/): per-modality results, reproducible, same hardware for every system
- [Embedded mode](docs/guide/embedded.md): the engine as a library, with no server process

## Known Limitations

- **No Bolt protocol.** Use gRPC or REST. Bolt is planned, so existing Neo4j drivers do not connect yet.
- **No APOC or GDS.** Common Cypher works, and the operations people reach APOC for most often (entity resolution, triggers) are native clauses here. Neo4j-specific procedure libraries are not supported.
- **Vector indexes are held in memory.** At 1M vectors of 384 dimensions that is roughly 1.5 GB before compression; SQ8 and RaBitQ bring it down substantially in exchange for a rerank pass.
- **The cluster is young.** Replication, follower reads and mTLS are covered by an automated fault-injection suite, but nothing substitutes for production years. Treat multi-node deployments accordingly.
- **Horizontal sharding is an Enterprise feature.** CE replicates the full dataset to every node, which is the right shape until a dataset outgrows one machine.

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

Sponsorship accelerates: the Bolt protocol so existing Neo4j drivers connect unchanged, SQL over the same engine, and hardening the cluster path that shipped in this release.

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
