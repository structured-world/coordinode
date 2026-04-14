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

For multi-statement transactions, use the transaction API:

```bash
# Begin
curl -X POST http://localhost:7081/v1/tx/begin
# → {"txId": "tx-abc123", "snapshotTs": "..."}

# Execute statements within the transaction
curl -X POST http://localhost:7081/v1/tx/tx-abc123/query \
  -d '{"query": "CREATE (n:Person {name: $name})", "parameters": {"name": "Bob"}}'

# Commit
curl -X POST http://localhost:7081/v1/tx/tx-abc123/commit
```

If you do not commit within the transaction timeout (default: 30 s), it is automatically rolled back.

## Time-Travel Queries

MVCC retains older versions of data for the configured retention window (default: 1 hour). You can query historical snapshots:

```cypher
MATCH (n:Person {name: "Alice"})
RETURN n.age
AT TIMESTAMP datetime("2024-06-01T12:00:00Z")
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
