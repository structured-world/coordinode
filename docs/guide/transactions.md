---
description: "CoordiNode's transaction model: MVCC snapshot isolation, optimistic conflict detection, write concerns from W0 to majority, read concerns, causal sessions and how replication factor relates to each."
---

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

Every statement reads the snapshot pinned when the transaction began (repeatable read), sees its own uncommitted writes, and buffers them until commit. Commit validates the write set against concurrent committers (first committer wins), assigns one commit timestamp, and applies every buffered mutation as a single atomic proposal at exactly that timestamp: a snapshot read at the commit timestamp sees the whole transaction, a read one tick earlier sees none of it, in embedded and cluster mode alike. Rollback discards the buffer; nothing was durable, so there is nothing to undo.

For the embedded API:

```rust
let tx = db.begin_transaction();
db.execute_in_transaction(tx, "CREATE (alice:Person {name: 'Alice'})", None)?;
db.execute_in_transaction(tx, "CREATE (bob:Person {name: 'Bob'})", None)?;
db.execute_in_transaction(
    tx,
    "MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}) CREATE (a)-[:KNOWS]->(b)",
    None,
)?;
let receipt = db.commit_transaction(tx)?;
// receipt.commit_ts: the HLC commit timestamp every write of this
// transaction landed at. Because commit timestamps are storage sequence
// numbers, it is the snapshot anchor for `AS OF TIMESTAMP <commit_ts>` (sees
// this commit and nothing later) and a change-stream position: a changed-keys
// scan from `commit_ts` includes this commit, a consumer that has processed
// it resumes from `commit_ts + 1`.
// receipt.applied_index: the committed Raft index in cluster mode, `None`
// embedded (there is no Raft log to index).
```

A statement error aborts the transaction; later statements and the commit fail with an unknown-transaction error, and the client restarts from `begin_transaction`. Idle transactions are rolled back after `interactive_idle_timeout` (default 30 s) and buffered writes are capped by `max_interactive_txn_bytes` (default 256 MiB) because an open transaction pins an MVCC snapshot and leader memory.

Over gRPC the same receipt is `CommitTransactionResponse { applied_index, commit_ts }`, and the multiplexed session stream acknowledges a commit with `Committed { applied_index, commit_ts }`. Pass `commit_ts` as `ReadConcern.at_timestamp` (level `SNAPSHOT`) to read exactly the state this commit produced.

## Time-Travel Queries

MVCC retains older versions of data for the configured retention window (`retention_window_secs`, default seven days, in every deployment mode including embedded). You can query historical snapshots in two ways.

**Cypher in-line syntax** for ad-hoc queries:

```cypher
MATCH (n:Person {name: "Alice"})
RETURN n.age
AS OF TIMESTAMP '2024-06-01T12:00:00Z'
```

**gRPC `ReadConcern.at_timestamp`** for programmatic pinning. Set `level = SNAPSHOT` and `at_timestamp = <HLC microseconds since Unix epoch>` on `ExecuteCypherRequest.read_concern`. The query then reads as of that exact HLC timestamp:

```python
client.execute_cypher(
    query="MATCH (a:Account {id: $id}) RETURN a.balance",
    parameters={"id": account_id},
    read_concern=ReadConcern(
        level=ReadConcernLevel.SNAPSHOT,
        at_timestamp=1_700_000_000_000_000,  # microseconds since epoch
    ),
)
```

Constraints:

- `at_timestamp` is only valid with `level = SNAPSHOT`. Combining it with `MAJORITY` / `LOCAL` / `LINEARIZABLE` returns `FAILED_PRECONDITION`.
- `at_timestamp` and `after_index` are mutually exclusive — pinning to a specific HLC and waiting for a Raft index are contradictory. The server returns `InvalidArgument` if both are non-zero.
- Reads older than the MVCC retention horizon return `OUT_OF_RANGE` with `ErrorInfo.reason = OUTSIDE_RETENTION` and `oldest_readable_ts` in the metadata: the earliest timestamp the same read succeeds at. The horizon is `now - retention_window_secs` (default seven days), held further back only while a live snapshot pin or a registered CDC / backup consumer still needs older history. The refusal is deliberate: history below the horizon may already be collected, and a read there could otherwise answer with a newer version or nothing. The same applies to `AS OF TIMESTAMP` in Cypher.

Typical use: auditing, debugging, time-aligned analytics across multiple queries that must observe the same database state.

In the embedded API the window is `StorageConfig::retention_window_secs` at open (`Database::open` uses the default), tunable at runtime with `Database::set_retention_window`; `Database::oldest_readable_timestamp` reports the current horizon, and a read below it fails with `DatabaseError::OutsideRetention` (`ReadConcern.at_timestamp`) or `ExecutionError::OutsideRetention` (`AS OF TIMESTAMP`).

What the window costs: the engine keeps every table a compaction consumed until the horizon passes the compaction, so history inside the window is served from the tree versions that were current at each point. Storage held by the window is therefore proportional to the write and compaction volume of the window, not to the number of rewritten keys. Size the window to the time travel you actually need.

## Conflict Semantics

Two concurrent write transactions conflict if they modify the **same node or edge**. Conflict resolution is first-writer-wins: the first transaction to commit succeeds; the second is aborted and must retry.

Posting-list operations (adding/removing edges on a node) use **merge operators** — they are commutative and never conflict with each other, only with DELETE on the same node.

## Durability

A write is durable once the Raft leader has written it to the level requested by the client's `WriteConcern.level`. The full ladder, ordered from least to most durable:

| Level | ACK after | Survives | Use for |
|-------|-----------|----------|---------|
| `W0` | request received | nothing (fire-and-forget) | non-critical metrics |
| `MEMORY` | RAM-only write (~1µs) | nothing before drain | hot counters, session state |
| `CACHE` | RAM + NVMe cache (~100µs) | process crash, not power loss before drain | analytics events, throughput-sensitive non-critical data |
| `W1` (default) | leader WAL fsync | leader crash if pre-replication | typical reads-mostly workloads |
| `MAJORITY` | Raft quorum (`RF/2 + 1` replicas) | single-replica failure | source-of-truth data, production writes |

`MEMORY` and `CACHE` use a background drain thread that batches volatile writes into Raft proposals asynchronously. The trade-off: ~1000× lower latency in exchange for losing in-flight writes on a leader crash before the drain completes. **Never select `MEMORY` or `CACHE` for data that you cannot reconstruct or afford to lose.**

The orthogonal `journal: true` flag forces a WAL fsync regardless of level (with `W0` it silently upgrades to `W1`). Use when you want fsync durability but cannot wait for replication — e.g., a single-node embedded deployment.

The default `WriteConcern` is `W1` (leader-only acknowledgement). Choose `MAJORITY` explicitly for writes that must survive a single-replica failure.

Replication factor (`RF`) is a deployment-time choice, independent of `WriteConcern`. The relationship between the two:

- **`RF` is the total number of replicas** holding a copy of each shard's log.
- **`MAJORITY` waits for `⌊RF / 2⌋ + 1` replicas** to acknowledge the entry before responding to the client.
- Raft tolerates `⌊RF / 2⌋` failed replicas while preserving liveness. So `RF=3` tolerates 1 failure (majority = 2 of 3), `RF=5` tolerates 2 failures, `RF=2` tolerates **zero** failures (majority = 2, cannot lose any) — which makes `RF=2` strictly worse than `RF=1` for cluster mode.

A single-node deployment runs at `RF=1`, where `MAJORITY` is functionally equivalent to `W1` because there is only one replica to hear from. Multi-node deployments choose `RF=3` as the smallest quorum-tolerant size, and that is where the distinction between `W1` and `MAJORITY` starts to matter: `W1` acknowledges once the leader has the write durably, `MAJORITY` waits for the write to survive the loss of the leader.

## Next Step

- [Hybrid Retrieval](./hybrid-retrieval) — multi-modal queries in a single transaction
- [Data Model](./data-model) — nodes, edges, labels, and indexes
