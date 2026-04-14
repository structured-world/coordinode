# MVCC Transactions

CoordiNode uses **Multi-Version Concurrency Control (MVCC)** with Snapshot Isolation. Every read sees a consistent point-in-time snapshot; writers never block readers.

## Key Properties

| Property | Behavior |
|----------|---------|
| **Isolation** | Snapshot Isolation — readers see a consistent snapshot of the database at transaction start |
| **Concurrency** | Optimistic — conflicts detected at commit, not at lock acquisition |
| **Writes** | Raft-replicated — a write is durable only after the leader commits it to the majority |
| **Timestamps** | Hybrid Logical Clock (HLC) — decentralized, monotonically increasing, no central oracle |

## Read Transactions

A read transaction opens a snapshot at the current HLC timestamp and keeps that view for its lifetime. Concurrent writes do not affect in-flight reads.

```cypher
-- This always sees a consistent graph, even while writers are active
MATCH (a:Person)-[:KNOWS]->(b:Person)
WHERE a.name = "Alice"
RETURN b.name
```

## Write Transactions

CoordiNode uses **Optimistic Concurrency Control (OCC)**:

1. Client buffers writes locally
2. At commit, the server checks for conflicts (another writer modified the same keys since the transaction's snapshot timestamp)
3. If no conflict → commit (Raft-replicated), timestamp advances
4. If conflict → abort, client retries

The retry is transparent for single-statement writes via the REST/gRPC API.

## Multi-Statement Transactions

Multi-statement explicit transactions are available via the **embedded API** (`coordinode-embed`) and via **gRPC** (native clients). A dedicated REST transaction endpoint is not yet implemented.

For the embedded API:

```rust
// Each execute_cypher call runs in its own auto-committed transaction.
// For multi-statement atomicity, use the batch Cypher approach:
db.execute_cypher("
  CREATE (alice:Person {name: 'Alice'})
  CREATE (bob:Person {name: 'Bob'})
  CREATE (alice)-[:KNOWS]->(bob)
")?;
```

## Time-Travel Queries

MVCC retains older versions of data for the configured retention window (default: 1 hour). You can query historical snapshots:

```cypher
MATCH (n:Person {name: "Alice"})
RETURN n.age
AS OF TIMESTAMP '2024-06-01T12:00:00Z'
```

This is useful for auditing, debugging, and point-in-time recovery.

## Conflict Semantics

Two concurrent write transactions conflict if they modify the **same node or edge**. Conflict resolution is first-writer-wins: the first transaction to commit succeeds; the second is aborted and must retry.

Posting-list operations (adding/removing edges on a node) use **merge operators** — they are commutative and never conflict with each other, only with DELETE on the same node.

## Durability

A write is durable once the Raft leader has written it to a quorum (majority) of replicas. The default is `RF=1` (single node, no replication) for the alpha release. Raft replication (`RF=3`) is available via the cluster configuration.

## Next Step

- [Hybrid Retrieval](./hybrid-retrieval) — multi-modal queries in a single transaction
- [Data Model](./data-model) — nodes, edges, labels, and indexes
