# Temporal Edges (Bitemporal)

Temporal edges let you store **multiple versions of the same relationship** between the same two nodes, each tagged with a validity interval `(valid_from, valid_to)`. CoordiNode keeps every version on disk, returns all of them on a normal `MATCH`, and lets you filter to a point or window in time with helper functions.

Temporal edges are **bitemporal end-to-end**: the valid-time axis (`valid_from` / `valid_to` on the edge, *when was this fact true in the modeled world*) and the system-time axis (MVCC commit-timestamp, *when did the database know this fact*) are both queryable and compose by stacking. A valid-time predicate inside `AS OF TIMESTAMP <commit_ts>` is the textbook bitemporal pattern — "what did the database believe at commit-time `ts` was true at valid-time `T`" — and resolves correctly: the snapshot reader applies the historical adjacency state, and the per-version edgeprop scan runs against the state visible at that commit.

This is the model behind:

- **Regulated industries** answering "what did the graph know at time T" for audit / compliance (MiFID II, SOX, HIPAA, GDPR).
- **Digital twins** that need to reconstruct topology as it was during an incident.
- **AI agents and GraphRAG** that ground retrieval to the state of the world at a document's publication time.
- **Anything with history** — employment, ownership, lease, prescription, role assignment — where overwriting yesterday's truth is the wrong thing to do.

## Declaring a temporal edge type

A regular edge type is created implicitly the first time you write an edge of that type. A **temporal** edge type must be declared first, so the storage layer knows to store one entry per version instead of one entry per pair:

```cypher
CREATE EDGE TYPE WORKS_AT TEMPORAL WITH (
  role: STRING,
  valid_from: TIMESTAMP NOT NULL,
  valid_to: TIMESTAMP
)
```

- `TEMPORAL` switches the storage layout from "one edgeprop per `(src, tgt)`" to "one edgeprop per `(src, tgt, valid_from)`". The adjacency posting still tracks existence — temporal-ness is a property of the edge type, not of the graph topology.
- The `WITH (...)` block is optional and declares user-visible properties. The example uses `valid_from: TIMESTAMP NOT NULL` to document the contract; the engine enforces presence of `valid_from` on every write regardless of whether it appears in `WITH`.
- Supported property types in `WITH`: `STRING`, `INT`, `FLOAT`, `BOOL`, `TIMESTAMP`, `BLOB`, `MAP`, `GEO`, `BINARY`, `DOCUMENT`.

Once a type is declared, you cannot flip its `TEMPORAL` flag — the storage layout would no longer match the existing data. Drop and re-create the type if you really need to convert.

## Writing temporal edges

Every `CREATE` of a temporal edge **must** provide a `valid_from` epoch-ms timestamp:

```cypher
MATCH (a:Person {name: 'Alice'}), (c:Company {name: 'Acme'})
CREATE (a)-[:WORKS_AT {
  valid_from: 1577836800000,  // 2020-01-01
  valid_to:   1688083200000,  // 2023-06-30
  role: 'SWE'
}]->(c)
```

Omitting `valid_from` is rejected at write time. `valid_to` is optional — leaving it out (or setting it `NULL`) means "still active, no known end date". This is the typical state for the current version of an ongoing relationship.

To add a new version of the same edge, just `CREATE` again with a different `valid_from`:

```cypher
MATCH (a:Person {name: 'Alice'}), (c:Company {name: 'Google'})
CREATE (a)-[:WORKS_AT {valid_from: 1688169600000, role: 'Staff'}]->(c)
```

Versions are keyed by `(type, src, tgt, valid_from)` — two versions cannot start at the same instant on the same pair.

## Closing a version (`SET r.valid_to`)

The canonical way to "end" an ongoing temporal version is to set its `valid_to` in place:

```cypher
MATCH (a:Person {name: 'Alice'})-[r:WORKS_AT]->(c:Company {name: 'Google'})
WHERE r.valid_to IS NULL                  // pick the open version
SET r.valid_to = 1735603200000            // 2024-12-31
```

The version stays in the graph (a plain `MATCH` still returns it) and just answers `false` to `temporal_active_at(r, t)` for `t >= valid_to`. The matched row carries `r.valid_from`, which keys the per-version edgeprop entry — `SET` updates exactly that entry without creating a new version.

Backdated entries (e.g. importing historical data) are equivalent — `CREATE` with `valid_to` already populated.

## Erasing a version (`DELETE r`)

`DELETE r` on a temporal edge is a **hard delete** of every version of the matched `(src, tgt)` pair, plus the adjacency posting entry. There is no per-version delete in Cypher — the model treats "the edge" as the logical pair, and version history is data, not the entity itself.

```cypher
MATCH (a:Person {name: 'Pat'})-[r:ASSIGNED]->(p:Project {name: 'Apollo'})
DELETE r                                  // every ASSIGNED version vanishes
```

Use `DELETE` for GDPR right-to-erase, mistaken inserts, or test cleanup. For everyday "this version ended", use `SET r.valid_to`.

## Reading: enumeration vs. point-in-time

A bare `MATCH` returns **every stored version**:

```cypher
MATCH (a:Person {name: 'Alice'})-[r:WORKS_AT]->(c:Company)
RETURN c.name, r.role, r.valid_from, r.valid_to
```

For Alice's three jobs that's three rows. `r.valid_from` is always populated, even if the property happened to be omitted from the stored value — readers see it reconstructed from the storage key.

To restrict to a point in time, use **`temporal_active_at(r, t)`**:

```cypher
MATCH (a:Person {name: 'Alice'})-[r:WORKS_AT]->(c:Company)
WHERE temporal_active_at(r, 1710460800000)   // 2024-03-15
RETURN c.name AS employer
```

`temporal_active_at(r, t)` returns true iff `r.valid_from <= t AND (r.valid_to IS NULL OR r.valid_to > t)`. Exactly one version of an ongoing relationship is active at any instant (assuming you don't have overlapping versions in your data).

To restrict to a window, use **`temporal_overlaps(r, t_start, t_end)`**:

```cypher
MATCH (a:Person {name: 'Alice'})-[r:WORKS_AT]->(c:Company)
WHERE temporal_overlaps(r, 1672531200000, 1704067200000)   // calendar year 2023
RETURN c.name, r.role
```

`temporal_overlaps(r, t0, t1)` returns true iff the version's validity interval overlaps `[t0, t1)`: `valid_from < t_end AND (valid_to IS NULL OR valid_to > t_start)`. Multiple versions can match a window.

## Performance and EXPLAIN

The planner pushes a literal `temporal_active_at(r, T)` predicate down into the traversal so the per-version edgeprop prefix scan can stop early instead of materializing every stored version. You can see this in `EXPLAIN`:

```
Project
  Filter(temporal_active_at(r, 1700000000000))
    Traverse(a -[r:WORKS_AT]-> b)
      temporal_filter(r=r, valid_from<=1700000000000, valid_to>1700000000000)
      NodeScan(a)
```

The `temporal_filter` block under `Traverse` is the push-down: only versions with `valid_from <= T` are read from storage, and `valid_to > T` is checked at decode. The outer `Filter` is kept as a safety net — both must agree before a row is emitted.

The push-down currently triggers on `temporal_active_at(r, <int_literal>)`. Parameter expressions (`$t`) fall back to scanning every version and filtering above the traversal — correct, but slower for pairs with many versions. Bound parameters are likely to be supported in a future release; until then, inline literals when push-down matters.

## Modeling guidelines

- **Pick one `valid_from` convention per type and stick with it.** Epoch milliseconds (`INT`) is the engine-native form and is what `temporal_active_at` / `temporal_overlaps` compare against. If you take input as ISO-8601, convert once at the application boundary.
- **Don't overlap versions in your data.** The engine accepts overlapping versions (two rows with overlapping `[valid_from, valid_to)` on the same pair), and `temporal_active_at` will return true for both. Application invariants like "exactly one open version per pair" are not enforced by the engine.
- **Reserve `DELETE` for true erasure.** Closing a version is `SET r.valid_to`. Hard-deleting history because of a typo loses the audit trail; create a corrective version instead.
- **A non-temporal edge type with `valid_from` / `valid_to` properties is NOT the same thing.** Non-temporal edges have one row per `(src, tgt)` — a second `CREATE` overwrites the first. Temporal edges keep both. If you want history, declare `TEMPORAL`.

## Mutating the validity timeline

- **`SET r.valid_to = <new>`** — close an open version, or shift the close time of an already-closed version. Updates the matched row in place.
- **`SET r.valid_to = null`** — re-open a closed version. Common workflow when a correction reverses a previous close.
- **`SET r.valid_from = <new>` is rejected.** `valid_from` is part of the storage key; mutating it would either leak a phantom version at the old key or split the value-key invariant. Use `DELETE r` + `CREATE` to re-key a version.
- **Reserved property names** — `__src__`, `__tgt__`, `__type__` are engine-internal row metadata. Declaring them in `WITH (...)` or assigning to them via `SET` is rejected.

## Interval invariants enforced at write time

- `valid_from` must be `INT` or `TIMESTAMP` (epoch milliseconds). `NULL` or any other type is rejected at `CREATE`.
- `valid_to`, if present, must be strictly greater than `valid_from`. Zero-duration versions (`valid_to == valid_from`) and inverted intervals (`valid_to < valid_from`) are rejected.
- The engine **does not** enforce non-overlapping open versions. Two rows whose `[valid_from, valid_to)` intervals overlap on the same `(src, tgt)` pair are accepted; both will return true to `temporal_active_at` during the overlap. Application code is responsible for the "exactly one open version per pair" invariant if that is the desired semantics.

## MERGE is not supported on temporal edge types

`MERGE (a)-[:T_TEMPORAL]->(b)` is rejected. MERGE's "match an existing edge OR create" model assumes a single (src, tgt) edge per type, which does not fit a per-version world. Use `CREATE` to add a new version, and `MATCH + SET / DELETE` to update or remove an existing one.

## Limitations

- **Parallel traversal** is bypassed for temporal queries; large fan-out runs sequentially.
- **No `AS OF TIMESTAMP $t` sugar** that rewrites to `temporal_active_at` yet; use the helper functions directly.
- **Adjacency posting tracks existence only.** A traversal that doesn't bind the edge variable (`MATCH (a)-->(b) RETURN a, b`) sees the pair once, not once per version. Reference `r` to materialize per-version rows.
- **Variable-length paths and edge mutation.** `MATCH (a)-[r:T*1..3]-(b) DELETE r` (or `SET r.x`) acts on the **last hop** of the matched path, not all of them. For multi-hop edits, prefer explicit single-hop patterns.
