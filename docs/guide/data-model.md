---
description: "How CoordiNode models data: nodes, relationships and labels, with vector, spatial and nested-document properties in one property graph, plus the three schema modes that govern validation."
---

# Data Model

CoordiNode uses a **property graph** extended with vector, spatial, and document capabilities, all within a single unified model.

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

A declared property type nests arrays at most 32 deep (an array of arrays of
integers is two). A label or edge type declaring a deeper one is refused when
it is defined, over gRPC and through the embedded API alike.

### Property Names

Stored records carry a compact integer id for each property name, not the
name itself. The first write that uses a name binds it to the next free id,
and the binding never changes afterwards:

- **Any write can introduce a name.** Maps and dynamic keys work the same as
  names written out in the query. A statement's new names are bound in one
  step before its data is written, so a record never refers to an id that has
  no name.
- **Bindings are shared.** They replicate with the data to every member of a
  cluster, travel inside snapshots, backups and dumps, and survive a restart
  or a leader change unchanged. Concurrent writers that introduce the same
  name get the same id.
- **A failed write can leave a binding behind.** If a statement fails after its
  names were bound, the names stay bound and the next write that uses them
  reuses their ids. Nothing else is affected.
- **A damaged dictionary is refused, not rebuilt.** If the stored names are
  missing or inconsistent while records still use them, the database refuses
  to open instead of starting with an empty dictionary, which would read every
  stored property as absent. Restore the store from a backup.

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

### Blobs

A blob is stored through `BlobService.UploadBlob`, the S3 gateway, or
`Database::upload_blob` in an embedded database. Its bytes are split into
chunks of up to 256KB, each identified by its SHA-256 digest and length, and
identical chunks are stored once, however many blobs hold them.

Every upload is its own blob with its own id, even when another blob holds the
same bytes: deleting one leaves the other whole. An S3 key is one blob too;
putting an object replaces what the key held. An upload that names a client
operation (`BlobChunk.operation` on its first message) takes effect at most
once and names the same blob on every attempt: a retry after a lost answer is
refused with `OPERATION_SETTLED`, its `blob_id` metadata naming the blob the
operation stored.

The upload is answered only once its chunks and the blob naming them are
durable on the member that took it, so an answered upload is one a crash
keeps. A delete is durable before its answer in the same way. A download reads
the blob as it stood when the download began, chunk by chunk, however slowly
the client consumes it, and a chunk whose bytes do not match its digest is
refused, never served.

Deleting a blob ends its hold on its chunks; the space comes back later, from
the server's own collector, with nothing for a client to scan, compact or
restart. A chunk another blob still holds stays. One nothing holds any more is
kept for a grace period (24 hours by default, never less than the MVCC
retention window) and for as long as a reader, retained history or an upload
in progress still needs it; then its bytes are deleted and the storage
compacted, which frees their space. `DeleteBlobResponse.chunks_deleted`
is therefore `0`: the delete itself frees nothing, and a delete is not space
released. An upload that was never published, such as one cut off by a crash,
is collected the same way once it is an hour old. Where collection stands,
including what is waiting, what holds it and how much it has reclaimed, is on
the [ops port](configuration.md#ports) as `GET /blob-gc`; see
[Orphan blob collection](configuration.md#orphan-blob-collection).

A store written by an earlier engine format is refused at open, untouched.

Blob payload is not yet carried to other members of a replicated group: on a
member whose group has other members (voters or learners), blob uploads and
deletes are refused with `FAILED_PRECONDITION` and reason
`BLOB_NOT_REPLICATED` (metadata `members`), before anything is written.
Reads are served as before.

## Tables and Keys

A table is a label with declared, typed columns. It is created with the same
statement from Cypher and from SQL (over the PostgreSQL wire protocol):

```sql
CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING, email STRING)
```

`UNIQUE` on a column is not supported yet: a `CREATE TABLE` that declares it
is refused and creates nothing. On a `ROW` table, make a column unique with a
named constraint instead:

```cypher
CREATE CONSTRAINT account_email FOR (a:Account) REQUIRE a.email IS UNIQUE
```

A table that has a constraint cannot be dropped until its constraints are.

Every row is a node, and like every node it gets its own
[`elementId`](../cypher/identity.md), issued by the database. The key you
declare does not replace that identifier; it is a second way to find the row,
backed by an index the table always has.

- **The key is unique.** Inserting a row whose key a row of the table already
  holds fails with `key already exists` and writes nothing; the error names the
  row that holds the key. The check runs inside the inserting transaction, so
  two concurrent inserts of one key leave exactly one row, and the other
  transaction gets the same error. Over gRPC this is `ALREADY_EXISTS` with
  reason `DUPLICATE_KEY` and the `table`, `key` and `element_id` in the error
  details; over the PostgreSQL wire it is SQLSTATE `23505` (`unique_violation`).
- **The key cannot change.** An update that sets or removes a key column is
  refused (`KEY_IMMUTABLE` over gRPC, SQLSTATE `42P10` over the PostgreSQL
  wire). To give a row a new key, delete it and insert a new row.
- **Lookups by key use the index.** A query that fixes every key column with
  an equality (`WHERE id = 1`, `MATCH (a:Account {id: 1})`) reads one index
  entry instead of scanning the table.
- **A key can span columns.** Marking several columns `PRIMARY KEY` declares
  one composite key over all of them, in the order they are declared.
- **Key column types.** Keys can be `BOOL`, integer, `FLOAT`, `STRING`,
  `TIMESTAMP` and binary columns. A key value cannot be NULL or NaN; `-0.0` and
  `0.0` are the same key.
- **A table without a declared key** is keyed by its rows' `elementId`: every
  insert adds a new row, and a row is found by its `elementId`.
- **Deleting frees the key.** Deleting a row, expiring it through a TTL, or
  `DROP TABLE` (which removes every row of the table) frees its keys for new
  rows.

## Indexes

CoordiNode maintains indexes when you create them:

```cypher
-- Exact property index (B-tree); UNIQUE enforces one node per value and is
-- the uniqueness constraint person_email, owning the index
CREATE UNIQUE INDEX person_email ON :Person(email)

-- Full-text index (Tantivy)
CREATE TEXT INDEX doc_body ON :Document(body)
  OPTIONS {analyzer: "english"}

-- Vector index (HNSW)
CREATE VECTOR INDEX doc_embedding ON :Document(embedding)
  OPTIONS { m: 16, ef_construction: 200, metric: "cosine", dimensions: 384 }
```

A B-tree index entry is written in the same transaction as the node it
indexes, so it commits, replicates and rolls back with the data. The
CREATE INDEX section of the [Cypher reference](../cypher/reference.md) covers
how unique indexes, lists and NULL behave, and the two maintenance profiles:
`RESOLVED` logs the index entries, `DERIVED` logs the change and every member
derives the entries from it.

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
