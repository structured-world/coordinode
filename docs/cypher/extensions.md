---
description: "CoordiNode Cypher extensions — vector search, full-text search, spatial queries, document operations, encrypted search, and time-travel reads."
---

# CoordiNode Extensions

CoordiNode extends OpenCypher with native support for vector search, full-text search, spatial queries, time-travel reads, document operations, and encrypted search. All extensions compose with standard Cypher — any combination works in a single query.

## Vector Search

### Schema (DDL) 📋

::: info Not yet available
Schema DDL (`CREATE LABEL`, `CREATE EDGE_TYPE`) is planned. In this release, schemas are created via the programmatic API (`LabelSchema`). Nodes can be created without pre-declared schemas.
:::

```cypher
-- Planned DDL syntax:
CREATE LABEL Product (
  name        STRING NOT NULL,
  description STRING,
  embedding   VECTOR(384, cosine)     -- dimensions, distance metric
)

-- Supported metrics: cosine, l2 (euclidean), dot, l1 (manhattan)
CREATE LABEL Image (
  pixels VECTOR(2048, l2)
)

-- Vectors on edges (unique to CoordiNode)
CREATE EDGE_TYPE SIMILAR (
  score           FLOAT,
  joint_embedding VECTOR(768, cosine)
)
```

### Vector Index ✅

```cypher
-- HNSW index (approximate nearest neighbor)
CREATE VECTOR INDEX product_embedding ON :Product(embedding)
  OPTIONS { m: 16, ef_construction: 200, metric: "cosine", dimensions: 384 }

-- Flat brute-force (exact, best for < 100K vectors)
CREATE VECTOR INDEX product_flat ON :Product(embedding)
  OPTIONS { metric: "cosine", dimensions: 384 }

DROP VECTOR INDEX product_embedding
```

| Option | Default | Description |
|--------|---------|-------------|
| `m` | 16 | Bi-directional links per HNSW node |
| `ef_construction` | 200 | Dynamic list size during build |
| `metric` | `"cosine"` | Distance metric: `cosine`, `euclidean`, `dot` |
| `dimensions` | — | Vector dimensionality (required) |

### Vector Query Functions ✅

```cypher
-- Euclidean (L2) distance — KNN pattern
MATCH (p:Product)
RETURN p.name, vector_distance(p.embedding, $query_vector) AS dist
ORDER BY dist LIMIT 10

-- Cosine similarity filter
MATCH (p:Product)
WHERE vector_similarity(p.embedding, $query_vector) > 0.7
RETURN p.name

-- Dot product
MATCH (a:Article)
WHERE vector_dot(a.embedding, $q) > 0.8
RETURN a.title

-- Manhattan (L1) distance
MATCH (n:Point)
RETURN vector_manhattan(n.coords, $target) AS l1_dist
```

### Vector Consistency Hint 🔷

Override the consistency mode for a single query:

```cypher
/*+ vector_consistency('eventual') */
MATCH (p:Product)
RETURN p.name, vector_distance(p.embedding, $query_vector) AS dist
ORDER BY dist LIMIT 10
```

| Mode | Description |
|------|-------------|
| `snapshot` | Consistent snapshot read (default). Reflects all committed writes |
| `eventual` | May read slightly stale index. Lower latency on hot workloads |

### Graph + Vector Combination ✅

```cypher
-- Traverse graph, then filter by vector similarity
MATCH (user:User {id: $uid})-[:PURCHASED]->(bought:Product)
MATCH (similar:Product)
WHERE similar <> bought
  AND vector_distance(similar.embedding, bought.embedding) < 0.2
RETURN DISTINCT similar.name,
       min(vector_distance(similar.embedding, bought.embedding)) AS score
ORDER BY score LIMIT 10
```

---

## Full-Text Search

### Text Index ✅

```cypher
-- Single property with language
CREATE TEXT INDEX doc_body ON :Document(body) LANGUAGE "english"

-- Multi-property with per-field analyzers and per-node language override
CREATE TEXT INDEX article_idx ON :Article {
  title:   { analyzer: "english" },
  body:    { analyzer: "auto_detect" },
  summary: { analyzer: "english" }
} DEFAULT LANGUAGE "english" LANGUAGE OVERRIDE "lang"

DROP TEXT INDEX doc_body
```

### Query Syntax ✅

```cypher
-- BM25 text search with score
MATCH (doc:Document)
WHERE text_match(doc.body, "distributed consensus algorithm")
RETURN doc.title, text_score(doc.body, "distributed consensus algorithm") AS relevance
ORDER BY relevance DESC LIMIT 10

-- Fuzzy: Levenshtein distance ≤ 2
WHERE text_match(doc.body, "konsensus~2")

-- Phrase: exact word sequence
WHERE text_match(doc.body, '"raft consensus"')

-- Boolean operators
WHERE text_match(doc.body, "raft AND (consensus OR paxos) NOT zookeeper")

-- Prefix wildcard
WHERE text_match(doc.body, "distribut*")

-- Per-term boosting
WHERE text_match(doc.body, "name^3 OR description^1")
```

### Supported Languages {#supported-languages}

**Single-word stemming (30+ languages):** Arabic, Armenian, Basque, Catalan, Danish, Dutch, English, Finnish, French, German, Greek, Hindi, Hungarian, Indonesian, Irish, Italian, Lithuanian, Nepali, Norwegian, Portuguese, Romanian, Russian, Serbian, Spanish, Swedish, Tamil, Turkish, Ukrainian, Yiddish.

**CJK (tokenizer feature flags):** Chinese (jieba-rs), Japanese (lindera), Korean (lindera).

**Special:** `auto_detect` — detects language per node from the LANGUAGE OVERRIDE field; `none` — whitespace tokenization only.

### Graph + Vector + Full-Text ✅

```cypher
-- All three retrieval modes in one query
MATCH (topic:Concept)-[:RELATED_TO*1..2]->(related)
MATCH (related)<-[:ABOUT]-(doc:Document)
WHERE vector_distance(doc.embedding, $query_vec) < 0.4
  AND text_match(doc.body, "attention mechanism")
RETURN doc.title,
       vector_distance(doc.embedding, $query_vec)        AS semantic_score,
       text_score(doc.body, "attention mechanism")       AS text_score
ORDER BY semantic_score LIMIT 10
```

---

## Spatial Queries

Spatial support uses WGS84 lat/lon coordinates with Haversine great-circle distance.

### Spatial Functions ✅

```cypher
-- Create a point literal
point({latitude: 40.7128, longitude: -74.0060})

-- Distance in meters (Haversine)
point.distance(r.location, point({latitude: 40.7128, longitude: -74.0060}))
```

### Spatial Queries ✅

```cypher
-- Restaurants within 2 km
MATCH (r:Restaurant)
WHERE point.distance(r.location, point({latitude: 40.7128, longitude: -74.0060})) < 2000
RETURN r.name, point.distance(r.location, point({latitude: 40.7128, longitude: -74.0060})) AS dist_m
ORDER BY dist_m

-- Social graph + spatial: places recommended by friends nearby
MATCH (me:User {id: $uid})-[:FOLLOWS]->(friend)-[rev:REVIEWED]->(place:Restaurant)
WHERE point.distance(place.location, $my_location) < 5000
  AND rev.rating >= 4
RETURN place.name, avg(rev.rating) AS score, count(friend) AS endorsements
ORDER BY score DESC LIMIT 10
```

::: info Spatial index
Point R-tree index is planned (v1.0 milestone). Currently, spatial queries perform a full scan with distance filter. For small datasets (< 100K nodes), performance is acceptable; for large datasets, pre-filter by label and bounding box manually until the spatial index is available.
:::

---

## Time-Travel Queries

Read historical data as it was at any point within the 7-day retention window.

### AS OF TIMESTAMP 🔷

```cypher
-- Read data at a specific timestamp
MATCH (u:User {id: 42})
RETURN u.name, u.email
AS OF TIMESTAMP '2026-03-15T10:00:00Z'

-- ISO 8601 format, UTC
-- Microsecond precision: '2026-03-15T10:00:00.123456Z'
```

The `AS OF TIMESTAMP` clause applies to the entire query. All MATCH patterns read from the MVCC snapshot at the given timestamp.

**Retention:** 7 days by default. Queries beyond the retention window return an error.

---

## Encrypted Search (SSE)

Searchable symmetric encryption — query encrypted fields without the server seeing plaintext. Uses HMAC-SHA256 tokens.

### Encrypted Index ✅

```cypher
CREATE ENCRYPTED INDEX patient_ssn ON :Patient(ssn)
DROP ENCRYPTED INDEX patient_ssn
```

### Encrypted Query ✅

```cypher
-- $encrypted_token is an HMAC-SHA256 token computed by the client
MATCH (p:Patient)
WHERE encrypted_match(p.ssn, $encrypted_token)
RETURN p.id, p.name
```

The server stores and compares HMAC tokens — it never sees the plaintext values. The client library (`coordinode-client`) provides `encrypt_field(key, plaintext)` to generate tokens.

---

## Document Operations

Nested document properties use MessagePack encoding with merge-operator-based mutations — no read-modify-write cycle, no OCC conflicts.

### Nested Document Properties ✅

```cypher
-- Create node with nested document property
CREATE (device:Device {
  name: 'Router-A',
  config: {
    network:  { ssid: 'office', channel: 6 },
    security: { protocol: 'WPA3', key_rotation: 3600 }
  }
})

-- Dot-notation access at any depth
MATCH (d:Device {name: 'Router-A'})
RETURN d.config.network.ssid, d.config.security.protocol

-- Partial update via dot-notation (O(1) merge operator — no full property read)
MATCH (d:Device {name: 'Router-A'})
SET d.config.network.ssid = 'home'

-- Remove nested key
MATCH (d:Device {name: 'Router-A'})
REMOVE d.config.security.key_rotation
```

### Array Mutation Functions ✅ 🔷

Concurrent-safe array mutations using LSM merge operators. Multiple writers can append/remove simultaneously without conflicts.

```cypher
-- Append to array
MATCH (n:Bag) SET doc_push(n.data.items, 'new_item')

-- Remove from array (all matching values)
MATCH (n:Bag) SET doc_pull(n.data.items, 'old_item')

-- Append only if not already present (set semantics)
MATCH (n:Bag) SET doc_add_to_set(n.data.tags, 'unique_tag')

-- Atomic increment / decrement
MATCH (n:Stats) SET doc_inc(n.data.view_count, 1)
MATCH (n:Stats) SET doc_inc(n.data.score, -0.5)
```

| Function | Syntax | Behavior |
|----------|--------|---------|
| `doc_push` | `SET doc_push(n.path, value)` | Append value to array |
| `doc_pull` | `SET doc_pull(n.path, value)` | Remove all occurrences of value |
| `doc_add_to_set` | `SET doc_add_to_set(n.path, value)` | Append if not present |
| `doc_inc` | `SET doc_inc(n.path, delta)` | Add delta to numeric value |

---

## Atomic Operations

### UPSERT MATCH 🔷

Atomic match-or-create. Avoids the TOCTOU race condition inherent in `MERGE`.

```cypher
UPSERT MATCH (u:User {email: $email})
ON MATCH  SET u.login_count = u.login_count + 1, u.last_login = now()
ON CREATE CREATE (u:User {email: $email, login_count: 1, created: now()})
```

### MERGE ALL 🔷

Creates relationships between ALL matching source-target pairs.

```cypher
-- Link all matching tags to all matching articles
MATCH (tag:Tag), (article:Article)
WHERE article.content CONTAINS tag.name
MERGE ALL (tag)-[:APPEARS_IN]->(article)
```

### SET ON VIOLATION SKIP 🔷

Skip nodes that would violate schema constraints, continue with the rest.

```cypher
-- Update users; silently skip those whose email would violate unique constraint
MATCH (u:User)
WHERE u.legacy = true
SET u.email = u.username + '@migrated.example.com' ON VIOLATION SKIP
```

---

## Computed Properties 📋

::: info Planned
Computed properties (DECAY, TTL) are defined in the schema API. Cypher DDL syntax is planned. They can be queried via standard property access in RETURN and WHERE once declared.
:::

```cypher
-- DECAY: value interpolated over time toward a target
-- Declared via: COMPUTED DECAY(initial: 1.0, target: 0.0, duration: '7d', anchor: created_at)
MATCH (m:Memory)
WHERE m.relevance > 0.5
RETURN m.content, m.relevance

-- TTL: auto-delete nodes/subtrees after a duration
-- Background reaper: every 60s, up to 1000 deletions per batch

-- Combining decay with vector search
MATCH (a:Article)
WHERE vector_similarity(a.embedding, $query) * a._recency > 0.5
RETURN a.title
```
