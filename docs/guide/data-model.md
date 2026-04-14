# Data Model

CoordiNode uses a **property graph** extended with vector, spatial, time-series, and document capabilities — all within a single unified model.

## Core Primitives

### Nodes (Vertices)

A **node** represents an entity. It has:

- One or more **labels** (type tags): `:Person`, `:Document`, `:Sensor`
- A map of **properties** (typed key-value pairs)
- An optional **embedding** property (dense float vector for similarity search)

```cypher
CREATE (p:Person {name: "Alice", age: 30, embedding: $vec})
```

### Edges (Relationships)

An **edge** connects two nodes directionally. It has:

- Exactly one **relationship type**: `KNOWS`, `RELATED_TO`, `ABOUT`
- A map of **properties** (weight, timestamp, confidence, …)

```cypher
CREATE (a)-[:KNOWS {since: 2024}]->(b)
```

Edges are first-class citizens — they can be traversed, filtered, and returned like nodes.

### Properties

CoordiNode supports all standard property types:

| Type | Example | Notes |
|------|---------|-------|
| String | `"Alice"` | UTF-8 |
| Integer | `42` | i64 |
| Float | `3.14` | f64 |
| Boolean | `true` |  |
| List | `[1, 2, 3]` | Homogeneous or mixed |
| Map | `{x: 1, y: 2}` | Nested document |
| Bytes | `$blob` | Arbitrary binary |
| Vector | `[0.1, 0.2, ...]` | Dense float array |
| DateTime | `datetime("2024-01-01")` | ISO 8601, timezone-aware |

## Labels and Modalities

Labels control how CoordiNode stores and indexes a node's data. A node can carry multiple labels, each triggering a different storage mode:

| Label pattern | Storage mode | Use case |
|---------------|-------------|---------|
| Standard label (`:Person`) | Graph record | Entities, relationships |
| `:VECTOR` label | HNSW index | Embedding similarity search |
| `:DOCUMENT` label | Tantivy index | Full-text search |
| `:TIMESERIES` label | Bucketed LSM | Sensor data, events, metrics |
| `:GEO` label | R*-tree index | Points, polygons, spatial queries |
| `:BLOB` label | BlobStore | Large binary objects |

A node can have multiple modality labels at once:

```cypher
CREATE (d:Document:VECTOR:DOCUMENT {
  title: "Attention Is All You Need",
  body: "We propose a new simple network architecture...",
  embedding: $vec
})
```

This single node is simultaneously reachable by graph traversal, vector similarity, and full-text queries.

## Identifiers

Every node and edge gets an immutable system-assigned `id` (64-bit integer) on creation. You can read it with `id(n)` in Cypher:

```cypher
MATCH (n:Person {name: "Alice"}) RETURN id(n)
```

## Indexes

CoordiNode maintains indexes automatically when you create them:

```cypher
-- Exact property index (B-tree in LSM)
CREATE INDEX person_email ON Person(email)

-- Vector index (HNSW, in-memory at alpha)
CREATE VECTOR INDEX doc_embedding ON Document(embedding)
  OPTIONS {dimensions: 384, metric: "cosine"}

-- Full-text index (Tantivy)
CREATE TEXT INDEX doc_body ON Document(body)
```

`EXPLAIN SUGGEST` tells you which indexes to create for a given query:

```bash
curl -X POST http://localhost:7081/v1/query/cypher/explain \
  -d '{"query": "MATCH (u:User) WHERE u.email = $e RETURN u"}'
```

## Transactions and MVCC

All reads and writes are wrapped in **MVCC (Multi-Version Concurrency Control)** transactions with Snapshot Isolation. See [MVCC Transactions](./transactions) for details.

## Next Step

- [MVCC Transactions](./transactions) — how reads, writes, and conflicts work
- [Hybrid Retrieval](./hybrid-retrieval) — combining graph + vector + text in one query
- [Quick Start](../QUICKSTART) — hands-on example
