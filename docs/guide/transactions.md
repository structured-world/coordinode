---
description: "CoordiNode's transaction model: MVCC snapshot isolation, optimistic conflict detection, write concerns as two axes (how many members, how durably each), read concerns, causal sessions and how replication factor relates to each."
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

A read at a timestamp you name (`AS OF TIMESTAMP`, `ReadConcern.at_timestamp`) is answered at that timestamp and no other. When a commit covered by it has not finished landing, the read waits for it rather than answering from a state missing a write its own timestamp includes. The wait is bounded (2 seconds); when it expires the read is refused and names the commit it was waiting for. It is never quietly moved to a timestamp that happens to be ready, because that would answer a different question than the one asked.

For the same reason such a read is not answered from an index that holds only the current state. Vector and full-text indexes are of that kind: an index built from today's data no longer contains a node deleted since the timestamp, and it matches and ranks by today's values. A read at a named timestamp that would use one is refused with `FAILED_PRECONDITION` / `INDEX_NOT_HISTORICAL` (embedded: `ExecutionError::IndexNotHistorical`), naming the index. A vector search can be asked with `/*+ vector_consistency('exact') */` instead, which evaluates every vector of the snapshot and is exact at any timestamp in the retention window; full-text search has no such alternative. A vector top-k on a label without a vector index is always evaluated this way.

A transaction is told about the conflict whether the other one has finished or is still landing. A commit registers the keys it is about to write before it validates them, so the second transaction is refused while the first is still being applied rather than being allowed through to overwrite it. For the same reason a transaction's snapshot stops below the oldest commit still in flight. A single statement may read at a fresh timestamp that covers a commit still landing; it then cannot see that commit, and its write is checked against everything from that commit on, so if the commit touched the same key the statement is refused as a conflict rather than replacing an update it never observed.

Posting-list operations (adding/removing edges on a node) use **merge operators** — they are commutative and never conflict with each other, only with DELETE on the same node.

### Writing at the version you read

A record's **version** is the timestamp of the commit that last wrote it. It comes back three ways, all the same number:

- with the node you read (`Node.version` over gRPC, `Database::node_version` embedded), so a read gives you the value and the version together;
- from the write that produced it (`QueryStats.commit_ts` on a statement that commits on its own, `CypherResult::commit_ts()` embedded), so after writing you already hold the version to condition your next write on, with no read in between;
- from the commit of an interactive transaction (`CommitTransactionResponse.commit_ts`), which is the timestamp every statement in it landed at. A statement *inside* such a transaction reports no timestamp of its own, because it has not committed yet; a read reports none either, having committed nothing.

A transaction can be committed on the condition that a node is still at the version you read. Over gRPC that is `CommitTransactionRequest.expect`; embedded it is `Database::expect_node_version` on the open transaction. An unset version means the node must **not** exist, which is the create-if-absent form of the same condition.

When the version has moved, the whole transaction is refused with `ABORTED` and `reason = REVISION_MISMATCH`, and the error metadata carries `expected_version` and `current_version`. Nothing of the transaction is applied. The current version travels with the refusal because your next move is computed from it: retry against what is there, merge, or stop. The server attaches no retry advice for this reason, unlike a write conflict: you named a version, so the decision is yours, and repeating a write whose premise was just disproved is rarely the right one.

Two uses follow from the same primitive.

**Read-modify-write without a lock.** Read the node and its version, compute, write on the condition of that version. First-writer-wins is a compare-and-set here, so the loser is told rather than silently overwritten.

**A fenced claim.** A lease or a claim is a record; holding it means having written it. A write done under the claim does two things: it states the claim record's version, and it writes the claim record too. The version catches a takeover that has already committed; writing the record puts the claim in the transaction's write set, so a takeover racing that very commit loses to first-writer-wins instead of interleaving with it. Either way the previous holder's write is refused on the claim rather than on the data, so the work it would have done never reaches the records the claim protects.

**Retrying after an outcome you never heard.** If a commit's reply is lost, retry the same work against the version you originally read. If the first attempt landed, the version moved and the retry is refused, which is how you learn it succeeded; if it did not, the version is unchanged and the retry does the work. The blind retry, which applies the write twice, is what this avoids.

Adjacency and counters carry no version. Their rows are folded from operands rather than replaced, so no single commit last wrote one, and a conditional write over them would be a condition on nothing. Asking for their version is refused rather than answered with "nothing is there", and so is conditioning a write on one. Edges added and removed, and counters incremented, are ordered by their merge operator and need no version to be safe under concurrency.

### Invariant refusals

Writing the same key is not the only way two transactions can be incompatible. Two of them can write entirely different keys and still, together, break a condition each of them checked on its own: an edge attached to a node the other is deleting, or a MERGE that created a relationship because it saw none while the other erased the last one. First-writer-wins cannot see either case, because there is no shared key to see it on.

So every mutation states the conditions its result depends on, and the server checks and reserves them before anything is applied. When one of those conditions no longer holds, the commit is refused with gRPC `ABORTED` and `reason = INVARIANT_REFUSED`, distinct from `TRANSACTION_CONFLICT` so that the cause is not misreported as a contended key. Nothing of the refused transaction is applied. The advised retry delay is zero, as for a conflict: re-running the transaction re-reads the state the condition is evaluated against, and waiting changes nothing. In the embedded API this is `DatabaseError::InvariantRefused`.

Conditions that agree do not exclude each other. Any number of transactions may attach edges to one node at the same time: they all state that the node keeps its identity, which is compatible with itself, so a popular node does not serialise the writes that reference it. Only a mutation that destroys the identity excludes them.

## Durability

A write concern is two independent parameters, as in MongoDB. `w` says how many members of the replica group must hold the write before the caller is answered; `journal` says what state each of those members holds it in.

**`w`: how many members**

| `w` | ACK after | Survives | Use for |
|-----|-----------|----------|---------|
| `acks: 0` | nothing is awaited (fire-and-forget) | what any committed write survives; the caller is simply not told whether it committed | non-critical metrics |
| `acks: 1` | the leader holds the write | leader crash only if the write was replicated in time | throughput-sensitive writes that can be lost with their leader |
| `acks: N` | `N` members hold the write, the leader included | the loss of `N - 1` members | a fixed guarantee that does not move with the group size |
| `mode: MAJORITY` (default) | `⌊RF / 2⌋ + 1` members hold the write | the loss of a minority | source-of-truth data, production writes |

A number is always a count of members and never a mode, so `acks: 3` means three members in a group of any size. Asking for more members than the group has is rejected with `INVALID_ARGUMENT`.

**`journal`: what state each counted member holds the write in**

| `journal` | The member counts once the write is | Survives | Latency |
|-----------|-------------------------------------|----------|---------|
| `JOURNAL` (default) | fsynced in its Raft log | process crash and power failure | the fsync |
| `CACHE` | in RAM plus its NVMe write cache, drained to the log later | process crash, not power loss before the drain | ~100µs |
| `MEMORY` | in RAM, drained to the log later | nothing before the drain | ~1µs |

A write concern decides when the caller is answered, never whether the write is replicated. Every write, `acks: 0` included, goes through the Raft log and is applied on every replica in the same order; no concern writes to one node only.

`CACHE` and `MEMORY` use a background drain thread that batches volatile writes into Raft proposals asynchronously. The trade-off: ~1000× lower latency in exchange for losing in-flight writes on a leader crash before the drain completes. **Never select `CACHE` or `MEMORY` for data that you cannot reconstruct or afford to lose.** A volatile state cannot be confirmed across members, so these two are accepted only with `acks: 0` or `acks: 1`; combining them with a larger `w` is rejected with `INVALID_ARGUMENT`.

Neither axis is ever silently changed to satisfy the other. `acks: 0` is answered at once whatever `journal` says, and a `journal` the server cannot honour for the requested `w` is refused rather than rewritten.

The default is `w: MAJORITY, journal: JOURNAL`, in the embedded library and over every protocol alike: an acknowledged write survives the loss of a minority. Choose a weaker concern explicitly, per request, for writes you can afford to lose.

**After a crash.** A process that is killed (`kill -9`, power loss, a panic that aborts) leaves the journal segment it was writing without the footer a clean shutdown adds. The next start recovers it on its own: every entry that was fsynced, and therefore acknowledged, is kept; a write that was cut off mid-way was never acknowledged and is dropped; the segment is then sealed and the database opens. The log records it at `WARN` as `oplog: sealed a segment left open by an unclean shutdown`, with the number of entries kept and bytes dropped. Nothing needs to be deleted by hand. A damaged entry inside a segment that *was* sealed is different: that is corruption, not an interrupted write, and it is reported rather than cut away.

Replication factor (`RF`) is a deployment-time choice, independent of the write concern. The relationship between the two:

- **`RF` is the total number of replicas** holding a copy of each shard's log.
- **`MAJORITY` waits for `⌊RF / 2⌋ + 1` replicas** to acknowledge the entry before responding to the client.
- Raft tolerates `⌊RF / 2⌋` failed replicas while preserving liveness. So `RF=3` tolerates 1 failure (majority = 2 of 3), `RF=5` tolerates 2 failures, `RF=2` tolerates **zero** failures (majority = 2, cannot lose any), which makes `RF=2` strictly worse than `RF=1` for cluster mode.

A single-node deployment runs at `RF=1`, where `MAJORITY` is functionally equivalent to `acks: 1` because there is only one replica to hear from. Multi-node deployments choose `RF=3` as the smallest quorum-tolerant size, and that is where the distinction between `acks: 1` and `MAJORITY` starts to matter: `acks: 1` acknowledges once the leader has the write, `MAJORITY` waits for the write to survive the loss of the leader.

## Next Step

- [Hybrid Retrieval](./hybrid-retrieval) — multi-modal queries in a single transaction
- [Data Model](./data-model) — nodes, edges, labels, and indexes
