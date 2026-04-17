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
| `vector_consistency` | `'snapshot'`, `'eventual'` | `'snapshot'` |
