---
description: "Node identity in CoordiNode — id() vs elementId(), encoding, stability guarantees, and migration from Neo4j."
---

# Node Identity

Every node in CoordiNode has a 64-bit identifier assigned at creation time. Two Cypher functions expose it:

| Function | Returns | Use it when |
|----------|---------|-------------|
| `elementId(n)` | 13-char string | New code. Stable, opaque, safe for serialization, durable across cluster restarts |
| `id(n)` | 64-bit integer | Only for Neo4j v4 driver compatibility |

For new applications: prefer `elementId`. The integer form exists because the Neo4j v4 wire protocol still surfaces an integer node id field that some drivers serialize into application-level types; CoordiNode returns the same raw `u64` there so those drivers keep working unchanged.

## elementId encoding

`elementId(n)` returns a 13-character string built from the underlying 64-bit identifier using **Crockford base32**. The encoding has a few useful properties:

- **Fixed length.** Always exactly 13 characters. No padding character, no leading-zero compression.
- **Case-insensitive on input.** Both `elementId(n) = "A1B2C3D4E5F6G"` and `elementId(n) = "a1b2c3d4e5f6g"` resolve the same node. The canonical form CoordiNode returns is uppercase.
- **Crockford normalisations.** The characters `I` and `L` are normalised to `1`, and `O` is normalised to `0`. This is a deliberate property of the Crockford alphabet — it lets humans transcribe IDs verbally without ambiguity. So `01ARZ3NDEKTSV` and `OLARZ3NDEKTSV` parse to the same node.
- **Sort-stable within a shard.** When two nodes are created on the same shard, the one allocated first has a lexicographically smaller `elementId` (when stripped of the shard prefix). This is a consequence of the underlying `u64` being monotone per shard.
- **Roundtrip-safe.** Every valid `elementId` decodes to exactly one `u64`, and every `u64` encodes to exactly one canonical 13-character string.

The 64-bit space is split into two windows:

```
┌──────────────────────┬──────────────────────────────────────────────┐
│ shard hint (20 bits) │ per-shard sequence (44 bits)                 │
├──────────────────────┴──────────────────────────────────────────────┤
│  up to 1,048,576 shards │ up to 17 trillion node creates per shard  │
└─────────────────────────────────────────────────────────────────────┘
```

In **CoordiNode CE** the shard-hint window is always zero (CE is single-shard). The hint exists so that nodes minted on different shards in a multi-shard deployment do not collide on the sequence space — it's a hint, not a routing decision (routing goes through the schema's placement policy, not the hint).

You should **not** parse `elementId` strings to extract this layout. Treat them as opaque tokens. The shape is documented for capacity planning, not for application use.

## When the ID changes (and when it doesn't)

`elementId(n)` is **immutable for the lifetime of the node**. The same node returns the same ID across:

- restarts of the database
- backups and restores
- cluster failover
- schema changes (`ALTER LABEL`, adding properties)

It does **not** change when:

- properties are added, removed, or modified
- the node gains or loses labels (in mixed-label graphs)
- the node is involved in a transaction that rolls back
- the database is upgraded between minor versions

It **does** change when:

- the node is `DELETE`d and a new one is `CREATE`d in its place — these are different nodes, with different IDs
- you restore from a backup that predates the node's creation — the node simply doesn't exist in that timeline

## Comparison with Neo4j

Neo4j 4.x and earlier exposed an integer `id(n)`. From Neo4j 5.x onward, the recommended form became `elementId(n)`, returning a string. The Neo4j string format is `"<dbid>:<uuid>:<localId>"` — a structured composite — whereas CoordiNode's form is a flat 13-character token.

The implication for migration:

- Application code that uses `elementId(n)` as an opaque key (storing it, comparing it, putting it in URLs) is portable. The token shape differs but the contract — string in, string out, equality matters — is the same.
- Application code that **parses** Neo4j's `elementId` is not portable. CoordiNode's encoding is flat. If you need the per-database or per-shard signal, look at the underlying schema rather than the ID surface.
- Neo4j's integer `id(n)` was officially deprecated in 5.x for reuse-after-delete reasons. CoordiNode does not reuse IDs, but the `id()` surface is kept narrowly for v4 driver compatibility and may be removed in a future major release.

## Examples

Stable references in your application code:

```cypher
MATCH (u:User {email: $email})
RETURN elementId(u) AS user_id
```

Then later, fetch by that ID directly without re-running the lookup:

```cypher
MATCH (u) WHERE elementId(u) = $user_id
RETURN u
```

Cross-referencing nodes by ID inside a single query:

```cypher
MATCH (parent:Article)-[:HAS_COMMENT]->(c:Comment)
WHERE elementId(parent) = $article_id
RETURN c
```

The integer form, only when driving from a Neo4j v4 driver:

```cypher
MATCH (n) WHERE id(n) = $legacy_id RETURN n
```

## See also

- [Functions reference](./functions) — full function list
- [Reference](./reference) — Cypher language coverage
