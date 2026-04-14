# Data Model

CoordiNode uses a **property graph** extended with vector, spatial, time-series, and document capabilities — all within a single unified model.

## Core Primitives

### Nodes (Vertices)

A **node** represents an entity. It has:

- One or more **labels** (type tags): `:Person`, `:Document`, `:Sensor`
- A map of **properties** (typed key-value pairs)

```cypher
CREATE (p:Person {name: "Alice", age: 30})
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

Labels are arbitrary user-defined strings — `:Person`, `:Article`, `:Sensor`. CoordiNode does not have reserved "special" labels for storage modes.

A node becomes **multi-modal** through its **property types**. A node with a vector-typed property participates in vector search; a node with a text property participates in full-text search — regardless of its label name.

```cypher
-- One node: graph + vector + full-text, defined by its properties
CREATE (d:Document {
  title: "Attention Is All You Need",
  body: "We propose a new simple network architecture...",
  embedding: [0.1, 0.2, ...]   -- stored as a Vector property → HNSW index
})
```

| Property type | Storage | Use case |
|--------------|---------|---------|
| String, Int, Float, Bool | Graph record | Standard entities |
| Vector (`[f32, ...]`) | HNSW index | Embedding similarity search |
| Geo point | Spatial index | Points, distance queries |
| Map | Nested document | Config, structured data |
| Bytes/Blob | BlobStore | Large binary objects |

## Indexes

CoordiNode maintains indexes when you create them:

```cypher
-- Exact property index (B-tree in LSM) — implemented
CREATE INDEX :Person(email)

-- Full-text index (Tantivy) — implemented
CREATE TEXT INDEX doc_body ON :Document(body)
  OPTIONS {analyzer: "english"}
```

> **Note:** Vector index DDL (`CREATE VECTOR INDEX`) is currently available via the programmatic API (`Database::create_vector_index()`). Cypher DDL syntax is planned.

`EXPLAIN SUGGEST` analyzes a query and recommends missing indexes. Available via gRPC or, when running Docker, via the REST proxy on port 7081:

```bash
curl -X POST http://localhost:7081/v1/query/cypher/explain \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (u:User) WHERE u.email = $e RETURN u", "parameters": {}}'
```

## Transactions and MVCC

All reads and writes are wrapped in **MVCC (Multi-Version Concurrency Control)** transactions with Snapshot Isolation. See [MVCC Transactions](./transactions) for details.

## Next Step

- [MVCC Transactions](./transactions) — how reads, writes, and conflicts work
- [Hybrid Retrieval](./hybrid-retrieval) — combining graph + vector + text in one query
- [Quick Start](../QUICKSTART) — hands-on example
