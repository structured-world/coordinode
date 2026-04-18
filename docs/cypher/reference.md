---
description: "OpenCypher language reference for CoordiNode — clauses, operators, data types, patterns, EXPLAIN SUGGEST."
---

# Language Reference

## Clauses

### Read Clauses

#### MATCH ✅

Matches graph patterns against stored nodes and edges.

```cypher
MATCH (n:User)
MATCH (a:User)-[:KNOWS]->(b:User)
MATCH (a)-[:LIKES*1..3]->(b)              -- variable-length: 1 to 3 hops
MATCH (a)-[:LIKES*]->(b)                  -- variable-length: 1 to max (capped at 10)
MATCH (a)-[r:LIKES|RATED]->(b)            -- multiple rel types (any of)
MATCH (a:User {email: $email})            -- inline property filter
MATCH p = (a)-[:KNOWS]->(b)              -- path variable (currently no path functions)
```

#### OPTIONAL MATCH ✅

Like MATCH but produces `null` for unmatched parts instead of dropping the row.

```cypher
MATCH (u:User {id: $id})
OPTIONAL MATCH (u)-[:CREATED]->(post:Post)
RETURN u.name, post.title
```

#### WHERE ✅

Filters rows. Supports all comparison, logical, and string operators. Can include pattern predicates.

```cypher
WHERE n.age > 18 AND n.active = true
WHERE n.email IS NOT NULL
WHERE n.name STARTS WITH 'A'
WHERE n.name ENDS WITH 'son'
WHERE n.bio CONTAINS 'engineer'
WHERE n.role IN ['admin', 'editor']
WHERE (a)-[:KNOWS]->(b)                   -- pattern predicate
WHERE NOT (a)-[:BLOCKED]->(b)             -- negated pattern predicate
WHERE n.score > 0 XOR n.featured = true
```

#### RETURN ✅

Projects the result set.

```cypher
RETURN n.name, n.age
RETURN DISTINCT n.city
RETURN n.name AS name, n.age AS age
RETURN *                                  -- all bound variables
RETURN n { .name, .age }                  -- map projection
RETURN n { .name, tags: collect(t.name) } -- map projection with expression
```

#### WITH ✅

Passes results between query parts, enabling filtering and transformation mid-query.

```cypher
MATCH (u:User)-[:PURCHASED]->(p:Product)
WITH u, count(p) AS purchases
WHERE purchases > 5
RETURN u.name, purchases
ORDER BY purchases DESC
```

#### UNWIND ✅

Expands a list into individual rows.

```cypher
UNWIND [1, 2, 3] AS x RETURN x
UNWIND $ids AS id MATCH (n {id: id}) RETURN n
```

#### ORDER BY ✅

Sorts the result. Supports multiple sort keys, ASC (default) and DESC.

```cypher
RETURN n.name ORDER BY n.name
RETURN n.name, n.age ORDER BY n.age DESC, n.name ASC
```

#### SKIP and LIMIT ✅

Pagination. Both accept integer expressions or parameters.

```cypher
RETURN n SKIP 20 LIMIT 10
RETURN n SKIP $offset LIMIT $page_size
```

---

### Write Clauses

#### CREATE ✅

Creates nodes or relationships.

```cypher
CREATE (n:User {name: 'Alice', email: 'alice@example.com'})
CREATE (a)-[:KNOWS {since: 2024}]->(b)
CREATE (a:User {name: 'Bob'})-[:FOLLOWS]->(b:User {name: 'Carol'})
```

#### MERGE ✅

Matches a pattern; creates it if it doesn't exist. Supports `ON MATCH` and `ON CREATE` actions.

```cypher
MERGE (u:User {email: $email})
ON CREATE SET u.created = now(), u.login_count = 1
ON MATCH  SET u.login_count = u.login_count + 1
RETURN u
```

#### UPSERT MATCH 🔷

CoordiNode extension. Atomic match-or-create with separate ON MATCH and ON CREATE branches. Avoids the TOCTOU race condition in MERGE.

```cypher
UPSERT MATCH (u:User {email: $email})
ON MATCH SET u.login_count = u.login_count + 1
ON CREATE CREATE (u:User {email: $email, login_count: 1, created: now()})
```

#### MERGE ALL 🔷

CoordiNode extension. Creates relationships between all matching `src × tgt` pairs (Cartesian product semantics).

```cypher
MATCH (a:Tag {name: 'rust'}), (b:Article)
WHERE b.tags CONTAINS 'rust'
MERGE ALL (a)-[:TAGS]->(b)
```

#### DELETE / DETACH DELETE ✅

Deletes nodes or relationships. `DETACH DELETE` removes all connected edges before deleting the node.

```cypher
MATCH (n:Temp) DELETE n                   -- fails if node has edges
MATCH (n:Temp) DETACH DELETE n            -- removes edges first
MATCH (a)-[r:KNOWS]->(b) DELETE r         -- delete relationship only
```

#### DETACH DOCUMENT ✅ 🔷

CoordiNode extension. Promotes a nested DOCUMENT property to a separate graph
node + edge atomically. See
[Cypher Extensions — DETACH DOCUMENT](extensions.md#detach-document) for
semantics, `TRANSFER EDGES`, default edge-type derivation, and error cases.

```cypher
MATCH (n:User)
DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)
```

#### ATTACH DOCUMENT ✅ 🔷

CoordiNode extension. The inverse of `DETACH DOCUMENT`: demote a graph node
back into a nested DOCUMENT property on another node. See
[Cypher Extensions — ATTACH DOCUMENT](extensions.md#attach-document) for
`TRANSFER EDGES`, `ON CONFLICT REPLACE`, `ON REMAINING FAIL`, and error cases.

```cypher
ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address
```

#### SET ✅

Sets properties and labels.

```cypher
SET n.name = 'Alice'
SET n.config.network.ssid = 'home'        -- nested document property
SET n += {name: 'Alice', age: 30}         -- merge (add/update, no delete)
SET n = {name: 'Alice'}                   -- replace all properties
SET n:Premium                             -- add label
SET n.updated = now() ON VIOLATION SKIP   -- skip nodes that violate schema
```

#### REMOVE ✅

Removes properties and labels.

```cypher
REMOVE n.temp_flag
REMOVE n.config.old_key                   -- nested document key
REMOVE n:Pending                          -- remove label
```

---

### DDL Clauses

#### CREATE INDEX / DROP INDEX ✅

B-tree index on a single property. Optional: `UNIQUE`, `SPARSE` (skip nulls), `WHERE` predicate (partial index).

```cypher
CREATE INDEX user_email ON :User(email)
CREATE UNIQUE INDEX user_email ON :User(email)
CREATE SPARSE INDEX article_deleted ON :Article(deleted_at)
CREATE INDEX active_users ON :User(name) WHERE active = true
DROP INDEX user_email
```

#### CREATE VECTOR INDEX / DROP VECTOR INDEX ✅

HNSW approximate nearest-neighbor index.

```cypher
CREATE VECTOR INDEX product_emb ON :Product(embedding)
  OPTIONS { m: 16, ef_construction: 200, metric: "cosine", dimensions: 384 }
DROP VECTOR INDEX product_emb
```

Supported metrics: `cosine` (default), `euclidean`, `dot`.

#### CREATE TEXT INDEX / DROP TEXT INDEX ✅

Full-text BM25 index using tantivy. Single-property or multi-property.

```cypher
-- Simple syntax
CREATE TEXT INDEX doc_body ON :Document(body) LANGUAGE "english"

-- Multi-property with per-field analyzers
CREATE TEXT INDEX article_idx ON :Article {
  title:   { analyzer: "english" },
  body:    { analyzer: "auto_detect" },
  summary: { analyzer: "english" }
} DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang_field"

DROP TEXT INDEX doc_body
```

Supported language values: `english`, `russian`, `french`, `german`, `spanish`, `chinese_jieba`, `japanese_lindera`, `none`, and [30+ more](./extensions#supported-languages).

#### CREATE ENCRYPTED INDEX / DROP ENCRYPTED INDEX ✅

Searchable symmetric encryption (SSE) index. Clients query via HMAC tokens without exposing plaintext.

```cypher
CREATE ENCRYPTED INDEX patient_ssn ON :Patient(ssn)
DROP ENCRYPTED INDEX patient_ssn
```

#### ALTER LABEL ✅

Change the schema mode of a label.

```cypher
ALTER LABEL Config SET SCHEMA FLEXIBLE
ALTER LABEL User   SET SCHEMA STRICT
ALTER LABEL Event  SET SCHEMA VALIDATED
```

| Mode | Behavior |
|------|----------|
| `STRICT` | Only declared properties, all interned. Rejects unknown properties |
| `VALIDATED` | Declared properties interned; unknown properties stored in overflow map |
| `FLEXIBLE` | All properties accepted, all interned. No schema enforcement |

---

### Time-Travel Clause

#### AS OF TIMESTAMP 🔷

Read data as it was at a specific timestamp (microsecond precision). Retention window: 7 days.

```cypher
MATCH (u:User {id: 42})
RETURN u.name, u.balance
AS OF TIMESTAMP '2026-03-15T10:00:00Z'
```

---

### Query Advisor

#### EXPLAIN SUGGEST 🔷

Analyzes a query's logical plan and returns actionable optimization suggestions.

```cypher
EXPLAIN SUGGEST
MATCH (u:User)
WHERE u.email = $email
RETURN u
```

**Response** — list of suggestions, each with:

| Field | Type | Description |
|-------|------|-------------|
| `kind` | string | `CREATE INDEX`, `ADD DEPTH BOUND`, `ADD JOIN`, `CREATE VECTOR INDEX`, `ADD PRE-FILTER`, `BATCH REWRITE` |
| `severity` | string | `CRITICAL`, `WARNING`, `INFO` |
| `explanation` | string | Human-readable description of the problem |
| `ddl` | string? | Ready-to-run DDL statement (when applicable) |
| `rewritten_query` | string? | Suggested query rewrite (when applicable) |

**Built-in detectors:**

| Detector | Severity | Trigger |
|----------|----------|---------|
| `MissingIndex` | CRITICAL | Full label scan on a filtered property with no index |
| `UnboundedTraversal` | WARNING | Variable-length path `*..` without upper bound |
| `CartesianProduct` | WARNING | Two disconnected MATCH patterns (cross-join) |
| `KnnWithoutIndex` | WARNING | `ORDER BY vector_distance(...) LIMIT n` without a vector index |
| `VectorWithoutPreFilter` | INFO | Vector scan without graph narrowing — suggest pre-filtering with MATCH |

**Example response output:**

```
kind:        CREATE INDEX
severity:    CRITICAL
explanation: Full label scan on User.email — filtering User nodes by 'email'
             without an index requires scanning all User nodes
ddl:         CREATE INDEX user_email ON User(email)
```

#### EXPLAIN (logical plan) ✅

Returns the logical plan for a query without executing it. Available via the `CypherService.Explain` RPC.

---

### CALL (Procedures) ✅ partial

Calls a named procedure. Only `db.advisor.suggestions` is currently available.

```cypher
-- Run suggestions for last query in session (or pass query text as param)
CALL db.advisor.suggestions() YIELD id, severity, kind, explanation, ddl
RETURN id, severity, kind, explanation, ddl
ORDER BY severity DESC
```

---

## Operators

### Comparison

| Operator | Description | Notes |
|----------|-------------|-------|
| `=` | Equality | Three-valued: `null = null` → `null` |
| `<>` | Inequality | |
| `<`, `<=`, `>`, `>=` | Ordered comparison | Works on Int, Float, String, Bool, Timestamp |

### Arithmetic

| Operator | Description |
|----------|-------------|
| `+` | Addition (numbers), concatenation (strings) |
| `-` | Subtraction |
| `*` | Multiplication |
| `/` | Division. Integer ÷ integer = integer |
| `%` | Modulo |

### Logical

| Operator | Description |
|----------|-------------|
| `AND` | Logical and (three-valued) |
| `OR` | Logical or (three-valued) |
| `NOT` | Logical not |
| `XOR` | Exclusive or |

### String Predicates

| Operator | Example |
|----------|---------|
| `STARTS WITH` | `n.name STARTS WITH 'Al'` |
| `ENDS WITH` | `n.name ENDS WITH 'son'` |
| `CONTAINS` | `n.name CONTAINS 'ali'` |

Case-sensitive. For case-insensitive search, use `text_match` with a text index.

### List and Null Predicates

| Operator | Example |
|----------|---------|
| `IN` | `n.role IN ['admin', 'editor']` |
| `IS NULL` | `n.deleted_at IS NULL` |
| `IS NOT NULL` | `n.email IS NOT NULL` |

---

## Data Types

| Type | Cypher literal | CoordiNode extension |
|------|---------------|---------------------|
| Integer | `42`, `-7` | |
| Float | `3.14`, `-0.5e2` | |
| Boolean | `true`, `false` | |
| String | `'hello'`, `"world"` | |
| Null | `null` | |
| List | `[1, 2, 3]` | |
| Map | `{key: value}` | |
| Point (WGS84) | `point({latitude: 0.0, longitude: 0.0})` | 🔷 |
| Timestamp | `'2026-03-15T10:00:00Z'` | Microsecond precision |
| Vector | `[0.1, 0.2, ...]` | 🔷 Up to 65536 dims |
| Blob | — | 🔷 Via BlobService |

---

## Graph Patterns

### Node Pattern

```
(variable:Label {prop: value})
```

- `variable` — optional. Binds the matched node for later use
- `:Label` — optional label filter. Multiple labels not yet supported
- `{prop: value}` — inline property equality filter

### Relationship Pattern

```
-[variable:TYPE {prop: value}]->
<-[variable:TYPE]-
-[variable:TYPE|OTHER_TYPE]-
-[variable:TYPE*min..max]->
```

- Direction: `->` outgoing, `<-` incoming, `-` undirected
- Multiple rel types: `-[:A|B|C]->` — matches any of A, B, C
- Variable-length: `*1..5` — 1 to 5 hops. `*` — 1 to 10 (default max)

### Pattern Predicates

Patterns in WHERE context evaluate to boolean:

```cypher
WHERE (a)-[:KNOWS]->(b)           -- true if at least one path exists
WHERE NOT (a)-[:BLOCKED]->(b)     -- true if no path exists
```

---

## CASE WHEN ✅

Simple CASE (equality test) and searched CASE (boolean conditions):

```cypher
-- Simple CASE
RETURN CASE n.role
  WHEN 'admin' THEN 'Administrator'
  WHEN 'editor' THEN 'Editor'
  ELSE 'User'
END AS display_role

-- Searched CASE
RETURN CASE
  WHEN n.score >= 90 THEN 'A'
  WHEN n.score >= 80 THEN 'B'
  ELSE 'C'
END AS grade
```

---

## Optimizer Hints

Per-query optimizer hints in comment syntax:

```cypher
/*+ vector_consistency('snapshot') */
MATCH (n:Product)
WHERE vector_distance(n.embedding, $q) < 0.3
RETURN n
```

| Hint | Values | Default |
|------|--------|---------|
| `vector_consistency` | `'current'`, `'snapshot'`, `'exact'` | follows `read_consistency` |
| `read_consistency` | `'current'`, `'snapshot'`, `'exact'` | `'current'` single-modality, auto-promoted to `'snapshot'` for cross-modality |

### `read_consistency`

Governs whether graph, vector, full-text, document, and time-series reads inside a single query resolve against the **same HLC timestamp** `T`. Every modality on the serving shard waits until every write with `commit_ts ≤ T` has been applied to every index before the read dispatches. Orthogonal to `read_concern` (which governs replication durability).

| Mode | Behaviour |
|------|-----------|
| `current` | Each modality reads its latest state independently. No watermark wait. Lowest latency. Default for single-modality reads. |
| `snapshot` | All modalities align at a single HLC `T` via `MaxAssignedWatermark::wait_for(T, read_timeout)`. HNSW post-filtered, tantivy segment-filtered by `commit_ts ≤ T`. **Auto-selected** when a query touches >1 modality. |
| `exact` | As `snapshot`, plus HNSW is bypassed (brute-force scan with MVCC filter). 100% recall, 10–100× slower vector path; use for audit / correctness-critical reads. |

**What counts as a "modality" for the auto-promotion rule:**

| Query shape | Modalities | Promoted? |
|-------------|-----------|-----------|
| `MATCH (n:T) WHERE n.prop = ... RETURN n` | graph | No — count = 1 |
| `MATCH (n:T) RETURN n ORDER BY vector_distance(n.emb, $q) LIMIT k` | vector | No — count = 1 (pure vector KNN, `NodeScan` is the row source, not a distinct modality) |
| `MATCH (n:T) WHERE text_match(n.body, 'q') RETURN n` | text | No — count = 1 |
| `MATCH (a)-[:R]->(b) WHERE vector_distance(b.emb, $q) < 0.3 RETURN a, b` | graph + vector | Yes — multi-hop traversal is genuine graph work |
| `... WHERE text_match(...) AND vector_distance(...) < ... ...` | text + vector | Yes |
| `... RETURN rrf_score([n.emb, n.body], {vector: $qv, text: $qt}) ...` | vector + text (via `RankFuse`) | Yes |

A bare `NodeScan`/`IndexScan` is treated as a carrier, not a distinct modality — so pure vector KNN and pure text search stay on the fast `current` default. Multi-hop graph patterns (`Traverse`, `ShortestPath`) count as genuine graph modality. Use an explicit `/*+ read_consistency('snapshot') */` hint to opt in to snapshot semantics on a single-modality query.

```cypher
-- Cross-modality auto-promotion: this query touches graph + text + vector,
-- so the planner picks read_consistency='snapshot' automatically.
MATCH (c:Chunk)
WHERE text_match(c.body, 'rust')
  AND vector_distance(c.embedding, $q) < 0.5
RETURN c

-- Explicit override of an auto-decision — e.g. accept potential
-- cross-modality skew to shave latency:
MATCH (c:Chunk)
WHERE text_match(c.body, 'rust')
  AND vector_distance(c.embedding, $q) < 0.5
RETURN c /*+ read_consistency('current') */

-- Force snapshot on a single-modality query (rare; usually for causal reads):
MATCH (n:Product)
WHERE vector_distance(n.embedding, $q) < 0.3
RETURN n /*+ read_consistency('snapshot') */
```

### `vector_consistency` as narrower override

When `read_consistency` and `vector_consistency` are both set, `vector_consistency` wins for the vector modality only — every other modality still follows `read_consistency`. This lets power users pin the vector path to `exact` (brute-force) while keeping FTS + graph aligned via snapshot:

```cypher
MATCH (c:Chunk)
WHERE text_match(c.body, 'rust')
  AND vector_distance(c.embedding, $q) < 0.5
RETURN c /*+ vector_consistency('exact') */
-- planner still auto-promotes read_consistency='snapshot'; vector goes exact,
-- text + graph align at the same HLC T.
```

### Timeout behaviour

Under `snapshot` or `exact` the executor blocks on the shard's applied-watermark. If the applier is stuck (replication lag, network partition), the wait times out after `read_timeout` (default 2s) and the query returns an error naming the mode, the target `commit_ts`, and a retry/fallback hint. Callers that prefer latency over cross-modality alignment can retry with `/*+ read_consistency('current') */`.
