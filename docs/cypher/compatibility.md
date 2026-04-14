---
description: "Neo4j compatibility matrix for CoordiNode — what OpenCypher features work, what's planned, and what CoordiNode adds."
---

# Neo4j Compatibility

CoordiNode is OpenCypher-compatible and supports common Neo4j workloads via the gRPC API. This page tracks what works today and what is planned.

::: info Full Bolt protocol compatibility is planned for v1.2
The Neo4j wire protocol (Bolt) is not yet implemented. In this release, use the native gRPC API or any gRPC-compatible client.
:::

## Query Language (OpenCypher)

| Feature | Status | Notes |
|---------|--------|-------|
| `MATCH` / `OPTIONAL MATCH` | ✅ Supported | Full pattern matching |
| `WHERE` (comparisons, boolean, IS NULL) | ✅ Supported | All standard operators |
| `RETURN` / `ORDER BY` / `LIMIT` / `SKIP` | ✅ Supported | With expressions and aliases |
| `CREATE` / `MERGE` / `DELETE` / `DETACH DELETE` | ✅ Supported | Full write operations |
| `SET` / `REMOVE` | ✅ Supported | Properties and labels |
| `WITH` / `UNWIND` | ✅ Supported | Pipeline and list expansion |
| Variable-length paths `*1..N` | ✅ Supported | Configurable; default max 10 hops |
| Shortest path | ✅ Supported | BFS-based |
| `CASE WHEN` | ✅ Supported | Simple and searched forms |
| Aggregations | ✅ Supported | count, sum, avg, min, max, collect, stDev, percentile |
| `x IN [...]` | ✅ Supported | List membership |
| `STARTS WITH` / `ENDS WITH` / `CONTAINS` | ✅ Supported | Case-sensitive |
| Pattern predicates in WHERE | ✅ Supported | `WHERE (a)-[:R]->(b)` |
| Map projection `n { .prop }` | ✅ Supported | |
| `UPSERT MATCH` (atomic) | 🔷 Extension | Not in Neo4j Cypher |
| `AS OF TIMESTAMP` | 🔷 Extension | Time-travel reads |
| `vector_distance()` / `vector_similarity()` | 🔷 Extension | Native vector search |
| `text_match()` / `text_score()` | 🔷 Extension | Native full-text search |
| `point()` / `point.distance()` | ✅ / 🔷 | point() is standard; Haversine implementation |
| `EXPLAIN` (logical plan) | ✅ Supported | Via `CypherService.Explain` |
| `EXPLAIN SUGGEST` | 🔷 Extension | Query advisor |
| `CALL procedures` | ⚠️ Partial | Only `db.advisor.suggestions()` in this release |
| `CALL {} subqueries` | 📋 Planned | |
| `FOREACH` | 📋 Planned | v1.0 milestone |
| `LOAD CSV` | 📋 Planned | v1.2 milestone |

## Scalar Functions

Most Neo4j scalar functions are not yet implemented. See [Functions reference](./functions#not-yet-implemented) for the full list.

| Function category | Status |
|------------------|--------|
| `coalesce`, `toString`, `size`, `now`, `type`, `labels` | ✅ Supported |
| `toInteger`, `toFloat`, `toLower`, `toUpper` | 📋 Returns null (planned) |
| `length`, `abs`, `ceil`, `floor`, `round`, `sqrt` | 📋 Returns null (planned) |
| `trim`, `substring`, `replace`, `split` | 📋 Returns null (planned) |
| `head`, `tail`, `last`, `range`, `reverse` | 📋 Returns null (planned) |
| `id`, `elementId`, `properties`, `keys` | 📋 Returns null (planned) |

## Data Types

| Type | Status | Notes |
|------|--------|-------|
| String, Integer, Float, Boolean, Null | ✅ Supported | Standard types |
| List | ✅ Supported | Homogeneous or mixed lists |
| Map | ✅ Supported | String-keyed maps |
| Point (spatial WGS84) | ✅ Supported | Lat/lon coordinates |
| DateTime / Timestamp | ✅ Supported | Microsecond precision, UTC |
| Duration | 📋 Planned | |
| Vector | 🔷 Extension | Up to 65536 dimensions |
| Blob | 🔷 Extension | Content-addressed binary storage |

## Indexes

| Type | Status | Notes |
|------|--------|-------|
| B-tree (single property) | ✅ Supported | Standard performance index |
| Composite (multi-property) | 📋 Planned | |
| Unique constraint | ✅ Supported | Enforced at commit time |
| Sparse (skip nulls) | ✅ Supported | `CREATE SPARSE INDEX` |
| Partial (filtered) | ✅ Supported | `CREATE INDEX ... WHERE predicate` |
| Text index (full-text BM25) | ✅ Supported | 30+ languages, fuzzy, phrase |
| Vector index (HNSW) | 🔷 Extension | Approximate nearest neighbor |
| Vector index (Flat/exact) | 🔷 Extension | Exact NN for small datasets |
| Encrypted index (SSE) | 🔷 Extension | Searchable symmetric encryption |
| TTL index | 🔷 Extension | Automatic expiration (planned DDL) |
| Point index (spatial R-tree) | 📋 Planned | v1.0 milestone |
| Lookup index | 📋 Planned | Neo4j-specific |

## Protocols

| Protocol | Status | Notes |
|----------|--------|-------|
| gRPC (port 7080) | ✅ Active | Native high-performance API |
| HTTP/REST (port 7081) | 📋 Planned | JSON transcoding of gRPC endpoints |
| Bolt (Neo4j wire protocol, port 7082) | 📋 Planned v1.2 | Neo4j drivers will connect without code changes |
| WebSocket (port 7083) | 📋 Planned | Subscriptions and live queries |

## Drivers

| Driver | Status | Notes |
|--------|--------|-------|
| Any gRPC client | ✅ Supported | Proto definitions at github.com/structured-world/coordinode-proto-ce |
| Generated Rust client | ✅ Supported | From proto |
| Neo4j Python driver | 📋 Planned v1.2 | Via Bolt protocol |
| Neo4j JavaScript driver | 📋 Planned v1.2 | Via Bolt protocol |
| Neo4j Java driver | 📋 Planned v1.2 | Via Bolt protocol |
| Neo4j Go driver | 📋 Planned v1.2 | Via Bolt protocol |

## CoordiNode vs Neo4j

### CoordiNode has, Neo4j doesn't:

- **Native vector search** — HNSW indexes on nodes AND edges. No plugin required.
- **Encrypted search (SSE)** — query encrypted fields without decryption. Programmatic API in this release; Cypher DDL planned.
- **Time-travel queries** (`AS OF TIMESTAMP`) — read historical snapshots within 7-day retention.
- **Partial indexes** — index with a WHERE predicate. Only matching nodes are indexed.
- **TTL indexes** — automatic node/subtree expiration. Background reaper every 60s.
- **Built-in query advisor** (`EXPLAIN SUGGEST`) — detects missing indexes, unbounded traversals, Cartesian products, and KNN patterns without vector indexes.
- **Free 3-node HA clustering** — Raft consensus. Neo4j charges per-node for equivalent clustering.
- **UPSERT MATCH** — atomic upsert without TOCTOU race conditions.
- **Document properties** — nested maps with dot-notation access and concurrent-safe array mutations.
- **Vectors on edges** — SIMILAR edges can store embeddings.
- **SET ON VIOLATION SKIP** — skip constraint-violating nodes without aborting the query.

### Neo4j has, CoordiNode doesn't (yet):

- Bolt protocol (📋 planned v1.2)
- APOC procedures library (partial support 📋 planned)
- LOAD CSV (📋 planned v1.2)
- GDS (Graph Data Science) library
- Neo4j Browser / Bloom visualization
- Cypher subqueries (`CALL {} syntax`) (📋 planned)
- Most scalar functions (toLower, toInteger, abs, etc.) (📋 planned)
- HTTP/JSON API (📋 planned v1.1)
- WebSocket subscriptions (📋 planned)
- Composite indexes
- Cypher duration type
