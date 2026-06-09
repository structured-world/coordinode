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
| `quantization` | `"none"` | In-RAM codec: `none`, `sq8`, `rabitq`, `rabitq-2bit`, `rabitq-3bit`, `rabitq-4bit` |
| `online_during_build` | `"block"` | Reader behaviour while the index backfills (see below) |

#### Online-during-build policy

`CREATE VECTOR INDEX` returns immediately after persisting the definition;
the HNSW graph is populated by a background backfill thread. Queries that
arrive before the backfill finishes are governed by `online_during_build`:

| Value | Behaviour |
|-------|-----------|
| `"block"` (default) | Reader polls the persisted state up to 30 seconds, then proceeds when the index reaches `Ready`. Matches the legacy synchronous semantic for callers that just want "do the right thing". |
| `"partial-recall"` | Reader hits the partial HNSW graph immediately. Recall improves as the backfill writes more vectors; useful when search latency matters more than completeness. |
| `"offline"` | Reader gets an error so it can pick a fallback path (e.g. brute force, alternative index, or queueing). |

A backfill that aborts (panic, write error) lands the index in the `Failed`
state and every reader sees an error regardless of policy. A subsequent
engine reopen rebuilds the HNSW from on-disk vectors and resets the state
to `Ready` automatically.

The CREATE response row includes a `state` field (`"building" | "ready" |
"failed"`) so clients can poll for completion or surface the status in a
control plane.

#### Graph predicate pushdown

When a query combines a vector top-K sort with a sibling label or simple
property filter, the planner pushes the predicate down into the HNSW
traversal so the search prunes non-matching candidates while it walks the
graph instead of returning a wider top-K that the executor then filters
post-hoc. Two examples:

```cypher
-- Label-only pushdown: only Item nodes are considered by HNSW.
MATCH (n:Item)
WHERE vector_similarity(n.embedding, $q) > 0.7
RETURN n ORDER BY vector_distance(n.embedding, $q) LIMIT 10;

-- Label + property pushdown: HNSW skips any candidate whose
-- `category` is not "electronics".
MATCH (n:Item)
WHERE n.category = 'electronics'
RETURN n ORDER BY vector_distance(n.embedding, $q) LIMIT 10;
```

What pushes down today:

- `:Label` from the MATCH pattern → `LabelEq(label)`.
- `var.prop = literal` (or `literal = var.prop`) leaves connected by
  top-level `AND` → `PropertyEq { property, value }`.

What does NOT push down (stays as a post-filter):

- Numeric range / inequality (`>`, `<=`, `BETWEEN`).
- `IS NULL` / `IS NOT NULL`.
- `OR`-branches (only top-level `AND` is decomposed).
- Cross-variable predicates (`n.x = m.y`).
- Parameter literals on the literal side (`$param`); deferred to a
  later optimisation that resolves params at plan time.

Pushdown is transparent: it never changes the result set, only the
search-time cost. The post-filter still runs and would catch any
mis-pushdown as a correctness backstop.

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

### DETACH DOCUMENT ✅ 🔷

Promote a nested document property to a separate graph node + edge in a single
atomic transaction. Useful when previously-embedded data outgrows its container
and needs its own relationships.

```cypher
-- Simple form: `n.address` → a new :Address node, linked back to n
MATCH (n:User {id: $uid})
DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)
RETURN a

-- Multi-segment path: promote a nested key
MATCH (n:User)
DETACH DOCUMENT n.meta.shipping AS (s:ShippingAddress)-[:HAS_SHIPPING]->(n)

-- Re-point existing edges onto the new node in the same transaction
MATCH (n:User {id: $uid})
DETACH DOCUMENT n.address AS (a:Address)-[:HAS_ADDRESS]->(n)
  TRANSFER EDGES ON n TO a WHERE type(r) IN ['SHIPS_TO', 'LIVES_AT']
```

**Semantics (single MVCC transaction):**

1. Read the DOCUMENT value at the given path.
2. `CREATE` the target node with the document's top-level keys as properties
   (shallow — nested maps/arrays remain as DOCUMENT on the new node).
3. `CREATE` the connecting edge. The canonical form
   `(a:Label)-[:TYPE]->(n)` stores the edge as `target → source`; the mirror
   form `(n)<-[:TYPE]-(a:Label)` is equivalent.
4. Remove the source property via a document merge operand — O(1) write, no
   read-modify-write.
5. If `TRANSFER EDGES` is specified, each matching edge on the source node is
   atomically re-pointed onto the new target via posting-list merge operators
   (no OCC conflicts, even on high-degree vertices).

**Default edge type:** if the relationship pattern omits a type (e.g. `-[]->`),
the engine derives `HAS_<UPPER_SNAKE(last_path_segment)>` — so
`n.sensorConfig` defaults to `HAS_SENSOR_CONFIG`.

**TRANSFER EDGES WHERE:** supports `type(r) IN [...]` (list of string literals)
or `type(r) = '...'` (single type). More complex predicates are rejected.

**Errors:**
- property does not exist on the source node
- property value is `null` or not a DOCUMENT/MAP
- source variable not bound by a prior `MATCH`

### ATTACH DOCUMENT ✅ 🔷

The inverse of `DETACH DOCUMENT`: demote a graph node back into a nested
DOCUMENT property on another node, atomically and in a single transaction.

```cypher
-- Simple form: the whole Address node becomes u.address
ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address

-- Nested target path
ATTACH (a:Shipping)-[:HAS_SHIPPING]->(u:User) INTO u.meta.shipping

-- Transfer out-of-band edges before the source node is deleted
ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address
  TRANSFER EDGES ON a TO u WHERE type(r) = 'SHIPS_TO'

-- Overwrite an existing target property
ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address
  ON CONFLICT REPLACE

-- Abort if any edges on the source would otherwise be cascade-deleted
ATTACH (a:Address)-[:HAS_ADDRESS]->(u:User) INTO u.address
  TRANSFER EDGES ON a TO u WHERE type(r) IN ['SHIPS_TO', 'LIVES_AT']
  ON REMAINING FAIL
```

**Semantics (single MVCC transaction):**

1. Match the inline pattern — builds its own `MATCH` (no prior `MATCH` needed).
2. Read all of the source node's properties and package them as a DOCUMENT map.
3. Write the DOCUMENT onto the target's property path via a `DocDelta::SetPath`
   merge operand — O(1) write, no read-modify-write. Single-segment target
   paths replace `props[root]` wholesale; multi-segment paths navigate into
   the existing DOCUMENT.
4. Delete the connecting edge (both adjacency halves + edge properties).
5. If `TRANSFER EDGES` is given, selected edges are re-pointed from source
   to target via posting-list merges before the source is removed.
6. Cascade-delete the source node and any untransferred edges — unless
   `ON REMAINING FAIL` was specified, in which case the query aborts when
   any untransferred edges remain.

**Options:**

| Clause | Default | With clause |
|--------|---------|-------------|
| `ON CONFLICT` | error if `target.path` already exists | `ON CONFLICT REPLACE` overwrites |
| `ON REMAINING` | cascade-delete remaining edges | `ON REMAINING FAIL` errors if any remain |
| `TRANSFER EDGES WHERE` | none — no edges are moved | supports `type(r) IN [...]` / `type(r) = '...'` |

**Errors:**
- source or target node not found
- target property exists and `ON CONFLICT REPLACE` was not specified
- `TRANSFER EDGES` predicate uses an unsupported shape
- `ON REMAINING FAIL` with untransferred edges

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

### MERGE NODES ✅ 🔷

Native node-merge for entity resolution and deduplication. Collapses two
matched nodes into one in a single atomic transaction — property merge,
edge re-pointing, and source deletion happen together. Equivalent in intent
to Neo4j APOC's `apoc.refactor.mergeNodes()` but without the plugin and with
correct behaviour under replication.

```cypher
-- Default: surviving node `a` keeps its properties on collision.
MATCH (a:User {email: 'alice@example.com'}),
      (b:User {email: 'alice@example.org'})
MERGE NODES (a, b) INTO a
  TRANSFER EDGES FROM b TO a
```

Property conflict resolution (`ON CONFLICT`):

| Strategy | Effect |
|----------|--------|
| `KEEP FIRST` (default) | Target's value wins. Source fills only missing keys. |
| `KEEP LAST` | Source's value overwrites target. |
| `COALESCE` | Source fills `NULL` / missing target keys only. |
| `SET <exprs>` | Per-property expressions referencing `a` and `b`. |

Duplicate-edge handling (`ON DUPLICATE`, requires `TRANSFER EDGES`): applied
when `target↔peer` and `source↔peer` exist with the same edge type and
direction.

| Strategy | Effect |
|----------|--------|
| `KEEP BOTH` (default) | Both edges preserved (parallel edges). |
| `MERGE PROPERTIES` | Single edge; edge facets coalesced (non-null from source fills null/missing on target). |
| `KEEP TARGET` | Target's edge wins; source's edge dropped. |

```cypher
-- Entity enrichment: fill missing fields from a duplicate record,
-- consolidate edges, merge edge properties on collision.
MATCH (a:Person {ssn: $ssn}), (b:Person {ssn: $ssn})
WHERE id(a) <> id(b)
MERGE NODES (a, b) INTO a
  ON CONFLICT COALESCE
  TRANSFER EDGES FROM b TO a
  ON DUPLICATE MERGE PROPERTIES
```

Idempotent: re-running with a non-surviving node already gone is a no-op
(the `MATCH` simply binds zero rows).

**Schema enforcement:** when the target's label is in `STRICT` mode, a merge
that would introduce an undeclared property is rejected before any mutation
commits. `VALIDATED` mode rejects type mismatches on declared properties but
allows source-only props into the `extra` overflow map. `FLEXIBLE` accepts
the merge unconditionally.

### Native Triggers ✅ 🔷

CoordiNode extension. Triggers are a first-class Cypher clause, not a
plugin — definitions persist in the schema partition, replicate via Raft,
and survive backups. Neo4j's equivalent (APOC) ships as a separate JAR and
breaks under clustering (eventually-consistent propagation, no failover
guarantees, doesn't survive `neo4j-admin restore`).

```cypher
-- Audit log without writing a single line of application code
CREATE TRIGGER audit
  ON :User CREATE | UPDATE | DELETE
  AFTER COMMIT
  EXECUTE
    CREATE (e:AuditEntry {
      action: $event,
      node_id: $after.id,
      ts: datetime(),
      before: $before,
      after: $after
    })
  ON ERROR RETRY 3 WITH BACKOFF 1000

-- Validation that rejects bad writes synchronously
CREATE TRIGGER reject_anonymous_user
  ON :User CREATE
  BEFORE COMMIT
  EXECUTE
    MATCH (u:User {id: $after.id})
    WHERE u.email IS NULL
    SET u.__rejected__ = true
  ON ERROR PROPAGATE

SHOW TRIGGERS
ALTER TRIGGER audit DISABLE        -- pause without losing the definition
ALTER TRIGGER audit ENABLE
ALTER TRIGGER audit SET EXECUTE …  -- replace body without re-registering
DROP TRIGGER audit
```

**Targets — node labels and edge types:**

```cypher
-- Node trigger: fires when nodes with the given label are mutated.
CREATE TRIGGER user_audit ON :User CREATE BEFORE COMMIT EXECUTE ...

-- Edge trigger: fires when edges of the given type are mutated.
CREATE TRIGGER follow_audit ON [:FOLLOWS] CREATE BEFORE COMMIT EXECUTE ...
```

Edge triggers and node triggers occupy separate index namespaces:
`:User` and `[:User]` never collide even when they share a name.

**Events:**

| Event | Node — fires on | Edge — fires on |
|-------|-----------------|-----------------|
| `CREATE` | `CREATE (n:Label ...)`, `MERGE (n:Label)` create branch, `UPSERT MATCH ... ON CREATE`, `DETACH DOCUMENT` (promoted node) | `CREATE (a)-[:TYPE ...]->(b)`, `MERGE (a)-[:TYPE]->(b)` create branch, `UPSERT MATCH ... ON CREATE`, `DETACH DOCUMENT` (connecting edge) |
| `UPDATE` | `SET n.prop = ...`, `REMOVE n.prop`, `REMOVE n:Label`, `MERGE NODES (a, b) INTO target` (target's merged record) (one firing per node per statement, regardless of how many items) | `SET r.prop = ...` (one firing per matched edge per statement; `SET r += {...}` is a no-op in the current executor and does NOT fire) |
| `DELETE` | `DELETE n` / `DETACH DELETE n`, source node of `ATTACH DOCUMENT`, non-surviving node of `MERGE NODES` | `DELETE r`, every edge removed by `DETACH DELETE` on an endpoint, every orphan edge removed by `ATTACH DOCUMENT` source cleanup, every orphan edge on `MERGE NODES` non-survivor |

**Trigger body parameters:**

| Parameter | Node trigger | Edge trigger |
|-----------|--------------|--------------|
| `$event`   | `"CREATE" \| "UPDATE" \| "DELETE"` | same |
| `$before`  | Pre-mutation prop map (or `NULL` for CREATE) | Pre-mutation edge prop map (or `NULL` for CREATE) |
| `$after`   | Post-mutation prop map (or `NULL` for DELETE) | Post-mutation edge prop map (or `NULL` for DELETE) |
| `$node`    | NodeId of the affected node | — |
| `$src`     | — | Source NodeId |
| `$tgt`     | — | Target NodeId |
| `$edge_type` | — | Edge type name (`"FOLLOWS"`, ...) |

For temporal edges, `DELETE r` fires the DELETE trigger once per stored
version of the matched `(src, tgt)` pair — each firing's `$before` is
that version's property map. `SET r.x = ...` fires once for the matched
version (keyed on `valid_from`).

**Execution model:**

| Timing | Where it runs | Failure default | Failure mode |
|--------|---------------|-----------------|--------------|
| `BEFORE COMMIT` | Raft leader, synchronous within the mutation's proposal | `PROPAGATE` | Aborts the originating transaction (caller sees the error) |
| `AFTER COMMIT` | Oplog consumer pool, any cluster node | `RETRY 3 WITH BACKOFF 1000` | Durable retry queue → dead-letter partition on exhaustion |

**Cycle protection (4 layers):**

- **L1** `CASCADE_LIMIT` — cumulative cascade depth across all triggers
  triggered by one originating user mutation. Per-trigger override; cluster
  default 10. Trips with the trigger chain attached for diagnostics.
- **L2** `CASCADE_FANOUT` — per-trigger fire count within one cascade root.
  Cluster default 100. Catches wide-but-shallow runaways (one trigger
  re-firing per row of a batch).
- **L3** Static cycle detection at `CREATE TRIGGER` *(planned)*.
  DFS over the `trigger.target_label → labels written by trigger body`
  graph; default warns, `WITH CYCLE_CHECK STRICT` rejects.
- **L4** Per-trigger auto-disable circuit breaker *(planned)*.
  When `trigger_errors_per_minute` or `cascade_overflow_count` exceeds
  thresholds, the trigger is automatically disabled until an operator
  re-enables it.

**Error handling — `ON ERROR { PROPAGATE | RETRY n [WITH BACKOFF ms] | DEAD_LETTER }`.**
Silent failure is impossible: every failed event lands in either
`trigger_pending:<name>:<seq>` (in-flight retry) or
`trigger_failures:<name>:<seq>` (dead-letter), inspectable via
`SHOW TRIGGER FAILURES` (planned).

**Replaces:** Neo4j APOC trigger plugin. CoordiNode's triggers are
native, replicated, and cluster-safe; CoordiNode's `CASCADE_LIMIT` /
`CASCADE_FANOUT` defaults catch runaways APOC silently lets through.

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
