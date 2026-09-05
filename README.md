# CoordiNode: graph, vector and full-text database in one engine

[![CI](https://github.com/structured-world/coordinode/actions/workflows/ci.yml/badge.svg)](https://github.com/structured-world/coordinode/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/structured-world/coordinode?label=release)](https://github.com/structured-world/coordinode/releases/latest)
[![PyPI](https://img.shields.io/pypi/v/coordinode?label=pypi)](https://pypi.org/project/coordinode/)
[![Docker](https://img.shields.io/badge/ghcr.io-coordinode-blue)](https://github.com/structured-world/coordinode/pkgs/container/coordinode)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

**The graph-native hybrid retrieval engine for AI and GraphRAG.** A graph database, a vector database and a full-text search engine, running as one transactional engine with one query language.

Built in Rust. Zero GC. Single binary. OpenCypher-compatible. Clustering included in the open-source edition.

- **One query** mixes graph traversal, vector similarity and BM25 text relevance, planned and executed as one pipeline.
- **One snapshot**: MVCC snapshot isolation across edges, embeddings and the text index, so a retrieval never reads a half-applied write.
- **One binary**: no JVM, no C dependencies, no separate router. The same executable runs standalone or as a Raft-replicated cluster.

---

## Quick start

Run the latest release from the container registry:

```bash
docker run -d --name coordinode \
  -p 7080:7080 -p 7081:7081 -p 7084:7084 \
  -v coordinode-data:/data \
  ghcr.io/structured-world/coordinode:latest
curl http://localhost:7084/health
```

Or install a package from the [latest release](https://github.com/structured-world/coordinode/releases/latest): RPMs for Fedora 42 to 44, DEBs for Debian and Ubuntu, and static binaries for linux-amd64 and linux-arm64.

From source (Rust 1.90+; the toolchain is pinned in `rust-toolchain.toml`):

```bash
git clone https://github.com/structured-world/coordinode.git
cd coordinode
cargo build --release
./target/release/coordinode serve --addr [::]:7080

# Optional CJK full-text analyzers
cargo build --release --features cjk-zh,cjk-ja,cjk-ko
```

Ports: 7080 gRPC, 7081 REST/JSON, 7084 metrics and health. The [Quick Start guide](https://docs.coordinode.com/QUICKSTART) goes from an empty database to a hybrid query in five minutes.

## One query, three modalities

Traverse a knowledge graph, filter by semantic similarity, rank by text match, all inside one transaction:

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

Across a split stack (graph database, vector store, search cluster) this is three round trips and a join in application code, with retrieval quality bounded by the least-fresh copy. Here it is one query, planned and costed as a whole, inside one snapshot.

## Who it is for, and where it stops

**Use CoordiNode today for**

- **GraphRAG and knowledge retrieval**: traverse a knowledge graph, filter by embedding similarity, rank by text relevance, in one statement.
- **Fraud detection and threat intelligence**: shared-device graphs with behavioural embedding similarity; attack patterns correlated across MITRE ATT&CK with vector and text search over indicators.
- **Recommendations and social discovery**: social-graph traversal plus semantic similarity to user preferences, with edge properties (ratings, timestamps) filterable in the same query.
- **Replacing a multi-database stack** (graph + vector + search) with a single binary and a single transaction model.

**Not yet, if you need**

- A drop-in replacement for a mature Neo4j Enterprise deployment, or APOC procedures, Neo4j Browser/Bloom, GDS.
- The Bolt protocol for existing Neo4j drivers. gRPC and REST are available now; Bolt and GraphQL are planned.
- A cluster with years of production mileage. Raft replication, follower reads and mTLS ship today and are covered by a fault-injection suite, but the deployment history is short.
- Vector indexes larger than memory. At 1M vectors of 384 dimensions that is roughly 1.5 GB before compression; SQ8 and RaBitQ bring it down substantially in exchange for a rerank pass.
- Horizontal sharding. The Community Edition replicates the full dataset to every node, which is the right shape until a dataset outgrows one machine; sharding is an Enterprise feature.

<details>
<summary>Example queries for each use case</summary>

### Fraud ring detection

```cypher
MATCH (suspect:Account {flagged: true})-[:SHARES_DEVICE*1..3]-(connected:Account)
WHERE vector_distance(suspect.tx_embedding, connected.tx_embedding) < 0.15
  AND connected.flagged = false
RETURN connected.id, connected.holder_name,
       vector_distance(suspect.tx_embedding, connected.tx_embedding) AS similarity
ORDER BY similarity LIMIT 50
```

### Semantic recommendation

```cypher
MATCH (me:User {id: $userId})-[:FOLLOWS*1..2]->(friend)
MATCH (friend)-[:PURCHASED]->(item:Product)
WHERE NOT (me)-[:PURCHASED]->(item)
  AND vector_distance(item.embedding, $user_taste_vector) < 0.3
RETURN DISTINCT item.name, item.category,
       vector_distance(item.embedding, $user_taste_vector) AS match_score
ORDER BY match_score LIMIT 20
```

### Threat intelligence

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

## Feature status

| Capability | Status | Details |
|-----------|--------|---------|
| OpenCypher read + write | **Stable** | MATCH, CREATE, MERGE, DELETE, SET, REMOVE, WITH, UNWIND |
| MVCC transactions | **Stable** | Snapshot isolation, write conflict detection (OCC) |
| HNSW vector search | **Stable** | Up to 65536 dims; cosine, L2, dot, L1; on node and relationship properties alike |
| Vector compression | **Stable** | SQ8 scalar quantization and RaBitQ binary codes with exact rerank |
| Late-interaction scoring | **Beta** | `maxsim_score()` for ColBERT-style multi-vector relevance, brute force over candidates |
| Full-text search | **Stable** | BM25, fuzzy, phrase; 20 languages via Snowball stemming, CJK via feature flags, per-field analyzers with language auto-detection |
| Hybrid graph + vector + text | **Stable** | Compound WHERE split into one optimised pipeline; `hybrid_score(node, query [, weights])` blend helper |
| B-tree indexes | **Stable** | Single, compound, unique, partial, TTL, sparse |
| Edge properties | **Stable** | CREATE with props, WHERE filter, inline pattern filter |
| Encrypted search (SSE) | **Stable** | AES-256-GCM + HMAC-SHA256 equality search over encrypted fields: `CREATE ENCRYPTED INDEX`, `encrypted_match()` |
| Time-travel queries | **Stable** | AS OF TIMESTAMP, 7-day retention |
| Bitemporal edges | **Stable** | `CREATE EDGE TYPE ... TEMPORAL`: versioned `(valid_from, valid_to)` intervals, `temporal_active_at()`, `temporal_overlaps()` |
| Query advisor | **Stable** | EXPLAIN SUGGEST with 5 detectors, N+1 detection |
| Spatial queries | **Stable** | `point()`, `point.distance()` (Haversine), WHERE filter |
| Document properties | **Stable** | Nested DOCUMENT type, dot-notation access, 3 schema modes |
| Document to graph and back | **Stable** | `DETACH DOCUMENT` promotes a nested property to a node and edge; `ATTACH DOCUMENT` demotes it back; atomic |
| Native entity resolution | **Stable** | `MERGE NODES (a, b) INTO a` in one transaction, with property and edge merge policies; replaces APOC `mergeNodes()` |
| Triggers | **Stable** (DDL) | `CREATE / DROP / SHOW / ALTER TRIGGER`, replicated through Raft, cascade limits and error policies; BEFORE / AFTER COMMIT firing lands in a follow-up release |
| gRPC API | **Stable** | Port 7080, all services |
| REST API | **Stable** | Port 7081, HTTP/JSON via gRPC-to-REST transcoding |
| Operational HTTP | **Stable** | Port 7084: /metrics, /health, /ready |
| Read/write concerns | **Stable** | local, majority, linearizable, causal sessions |
| Raft replication | **Stable** | Multi-node writes, leader election, snapshot transfer, log compaction; in the Community Edition, no per-node licensing |
| Follower reads | **Stable** | Reads served from replicas under the requested consistency level |
| Inter-node TLS and mTLS | **Stable** | Pure-Rust rustls stack, no OpenSSL |
| Inter-node compression | **Stable** | zstd on the replication transport, level configurable |
| Consistency test suite | **Stable** | In-process fault injection: partition matrix, crash and clock skew, linearizability checking over a live workload |
| Scrub and repair | **Stable** | Background checksum verification, damaged segments rebuilt from a healthy replica |
| Embedded engine | **Stable** | `coordinode-embed` runs the full engine in-process; exposed to Python as `coordinode-embedded` |

| Planned | Notes |
|---------|-------|
| Bolt protocol | Neo4j drivers connect without code changes |
| GraphQL API | Auto-generated schema; SDL generation already in tree |
| SQL over the same engine | PostgreSQL wire protocol against the shared query IR |

Syntax for every extension above is in the [Cypher extensions reference](https://docs.coordinode.com/cypher/extensions).

## What makes CoordiNode different

Other engines combine some of these. What CoordiNode optimises for is the combination holding under one transaction, one planner and one storage engine:

- **One planner over all three modalities.** A compound `WHERE` mixing traversal, vector distance and text match is split into one pipeline and costed as a whole, not executed as three lookups joined by your application.
- **One snapshot.** Graph edges, embeddings and the text index move together under MVCC snapshot isolation.
- **Vector search over relationships as well as nodes**, integrated with traversal and transactional execution.
- **Rust with no garbage collector.** Tail latency is a design constraint, not a tuning exercise: no JVM pauses, no stop-the-world.
- **Pure Rust with no FFI.** `cargo build` produces the whole engine, compression and TLS included. No OpenSSL, no C storage engine, nothing to reconcile with your base image.
- **AGPL-3.0 with clustering included.** Replication is not held back for a paid tier.
- **Operations answered in-engine:** entity resolution, document promotion and demotion, and triggers are native clauses, not plugins that break once you cluster.

For measured comparisons rather than claims, see the [benchmarks](https://docs.coordinode.com/benchmarks/): CoordiNode and the systems it is compared against run on the same host, results are JSON-recorded with a hardware fingerprint and commit SHA, and CoordiNode's numbers are regenerated by CI on every push.

**Coming from Neo4j?** The [compatibility matrix](https://docs.coordinode.com/cypher/compatibility) lists clause by clause what carries over, what is spelled differently, and what is missing.

## Python SDK

```bash
pip install coordinode                               # gRPC client
pip install coordinode-embedded                      # in-process engine, no server
pip install langchain-coordinode                     # LangChain GraphStore
pip install llama-index-graph-stores-coordinode      # LlamaIndex PropertyGraphStore
```

Source: [structured-world/coordinode-python](https://github.com/structured-world/coordinode-python).

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

20 Rust crates, about 232K lines of code. A multi-node deployment runs the same binary on every node: replication and routing are built in, with no separate router or coordinator process to operate.

## Documentation

- [Quick Start](https://docs.coordinode.com/QUICKSTART): from zero to a hybrid query in five minutes
- [Guide](https://docs.coordinode.com/guide/): concepts, deployment, operations
- [Cypher extensions](https://docs.coordinode.com/cypher/extensions): vector, full-text, spatial, time-travel and encrypted-search syntax
- [Neo4j compatibility](https://docs.coordinode.com/cypher/compatibility): clause-by-clause matrix
- [Configuration](https://docs.coordinode.com/guide/configuration): every tunable, its default, and whether it needs a restart
- [Embedded mode](https://docs.coordinode.com/guide/embedded): the engine as a library, with no server process
- [Benchmarks](https://docs.coordinode.com/benchmarks/): per-modality results, reproducible, same hardware for every system

## Contributing

Bug reports, features and documentation are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are accepted under the [Contributor License Agreement](CLA.md). Security issues go to the address in [SECURITY.md](SECURITY.md).

## License

Copyright (C) 2026 Dmitry Prudnikov.

CoordiNode Community Edition is licensed under **AGPL-3.0-only**: genuine open source with SaaS protection. See [LICENSE](LICENSE) and [COPYRIGHT](COPYRIGHT).

The same code base is also available under a commercial licence as the Enterprise Edition (horizontal sharding, multi-tenancy, CRUSH placement, geo-distribution), for deployments that cannot meet the AGPL terms. Contact: enterprise@sw.foundation. The commercial licence never narrows the Community Edition: everything published here stays under AGPL-3.0-only, in full.

## Support the project

CoordiNode is developed by [Dmitry Prudnikov](https://github.com/polaz) and contributors. Donations go directly to the maintainer and fund development time.

If you believe graph + vector + text should live in one engine under a genuine open-source license, you can support the work:

<div align="center">

![USDT TRC-20 Donation QR Code](assets/usdt-qr.svg)

USDT (TRC-20), maintainer's personal wallet: `TFDsezHa1cBkoeZT5q2T49Wp66K8t2DmdA`

</div>

Sponsorship accelerates: the Bolt protocol so existing Neo4j drivers connect unchanged, SQL over the same engine, and hardening the cluster path that shipped in this release.
