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
let tx = db.begin_transaction()?;
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

**The handle.** `BeginTransaction` (and the session's `Begun`) returns the transaction as an opaque byte string; the embedded API returns a `TransactionHandle`. It names the database, the member that holds the transaction and the leadership term it was begun in, so a handle of another database, of an earlier term or of a restarted server names nothing. Send it back unchanged; bytes that are not a handle are refused with `INVALID_ARGUMENT`.

**Who holds it.** The leader holds every transaction, and a client reaches it through any member. BEGIN sent to a follower opens the transaction at the leader; a statement, commit or rollback sent to any member runs at the member that holds the transaction, and its answer comes back through the member the client reached. Commands for one transaction may go to different members, one after another, and still run in it. A transaction belongs to its holder, not to the connection: closing a session stream does not roll anything back, and another stream, to the same member or another, continues the transaction by its handle, with the same snapshot, its own writes and the settings fixed at BEGIN. During an election, when no member leads, BEGIN is refused with `FAILED_PRECONDITION`, `reason = NOT_LEADER` and no `leader_id`; retry once a leader is elected. When the leader changes before the commit, the transaction ends: commands for it are refused with `ABORTED`, `reason = TRANSACTION_ABORTED`, `end = LEADER_CHANGED`. Nothing of it was applied; run it again from BEGIN. The same member becoming leader again later does not bring the old transaction back.

**Moving to the leader.** Every answer of a member that does not lead names the leader in the `x-coordinode-leader` response header: its node id, or `none` during an election. The address members reach each other at is internal and is not disclosed. A command that ran at another member also reports `x-coordinode-hops: 1`. The client keeps working where it is; one that can reach the leader directly saves a hop by moving there, and one that cannot (a remote site, a network where only some members are exposed) never has to. The leader's own answers carry no hint.

**Ordering.** Commands for one transaction run one at a time. One that arrives while another runs waits for its turn in a queue on the holder; past 64 waiting commands the next is refused with `ABORTED`, `reason = TRANSACTION_BUSY`, and the transaction stays as it is. A session transaction begun `ORDERED` numbers its statements from 1 (`Execute.nonce`) and they run in that order whatever order they arrive in; a missing statement is waited for up to the drain timeout (`Begin.drain_timeout_ms`, the session's setting, else 5 s), after which the transaction is aborted (`end = DRAIN_TIMED_OUT`). A commit with `last_nonce` waits until the chain up to it has run. In either ordering a statement repeated with a nonce already used is not applied again: the same statement is refused with `ALREADY_EXISTS`, `reason = STATEMENT_ALREADY_ADMITTED` (its rows are not kept), a different one under that nonce with `INVALID_ARGUMENT`, `reason = STATEMENT_NONCE_REUSED`.

**Ending.** A statement error aborts the transaction; later statements and the commit are refused with `TRANSACTION_ABORTED` (`end = STATEMENT_FAILED`), and the client restarts from BEGIN. Idle transactions are rolled back after `interactive_idle_timeout` (default 30 s, `end = IDLE_EXPIRED`) and buffered writes are capped by `max_interactive_txn_bytes` (default 256 MiB) because an open transaction pins an MVCC snapshot and leader memory. A handle naming nothing open, or a transaction of another caller, is refused with `NOT_FOUND`, `reason = UNKNOWN_TRANSACTION`. `SHOW TRANSACTIONS` lists the transactions a member holds, with the session attached to each or none when no stream is.

Over gRPC the same receipt is `CommitTransactionResponse { applied_index, commit_ts }`, and the multiplexed session stream acknowledges a commit with `Committed { applied_index, commit_ts }`. Pass `commit_ts` as `ReadConcern.at_timestamp` (level `SNAPSHOT`) to read exactly the state this commit produced.

## Time-Travel Queries

MVCC retains older versions of data for the configured retention window (`retention_window_secs`, default five minutes, in every deployment mode including embedded). You can query historical snapshots in two ways.

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
- Reads older than the MVCC retention horizon return `OUT_OF_RANGE` with `ErrorInfo.reason = OUTSIDE_RETENTION` and `oldest_readable_ts` in the metadata: the earliest timestamp the same read succeeds at. The horizon is `now - retention_window_secs` (default five minutes), held further back only while a live snapshot pin still needs older history. The refusal is deliberate: history below the horizon may already be collected, and a read there could otherwise answer with a newer version or nothing. The same applies to `AS OF TIMESTAMP` in Cypher.

Typical use: auditing, debugging, time-aligned analytics across multiple queries that must observe the same database state.

In the embedded API the window is `StorageConfig::retention_window_secs` at open (`Database::open` uses the default), tunable at runtime with `Database::set_retention_window`; `Database::oldest_readable_timestamp` reports the current horizon, and a read below it fails with `DatabaseError::OutsideRetention` (`ReadConcern.at_timestamp`) or `ExecutionError::OutsideRetention` (`AS OF TIMESTAMP`).

What the window costs: the engine keeps every table a compaction consumed until the horizon passes the compaction, so history inside the window is served from the tree versions that were current at each point. Storage held by the window is therefore proportional to the write and compaction volume of the window, not to the number of rewritten keys. Size the window to the time travel you actually need.

## Conflict Semantics

Two concurrent write transactions conflict if they modify the **same node or edge**. Conflict resolution is first-writer-wins: the first transaction to commit succeeds; the second is aborted and must retry.

A single statement run without `BEGIN` is retried by the server itself: a refused commit applied nothing, so the statement runs again from a fresh snapshot, by default up to six times over about 60 ms (`autocommit_retry_attempts`, `autocommit_retry_backoff_ms`; `Database::set_autocommit_retry` when embedded), before the conflict is returned. This covers contention with commits the client never issued, such as background index maintenance. A statement inside an explicit transaction, one pinned to a named timestamp and a cursor page are not retried: the transaction's earlier statements, the named snapshot and the cursor's position belong to the caller.

A read at a timestamp you name (`AS OF TIMESTAMP`, `ReadConcern.at_timestamp`) is answered at that timestamp and no other. When a commit covered by it has not finished landing, the read waits for it rather than answering from a state missing a write its own timestamp includes. The wait is bounded (2 seconds); when it expires the read is refused and names the commit it was waiting for. It is never quietly moved to a timestamp that happens to be ready, because that would answer a different question than the one asked.

Vector and full-text indexes hold the current state, and such a read still goes through them: the nodes written after the timestamp are found from the stored version history, left out of the index's answer and read from the snapshot instead, so a node deleted since is found, one created since is not, and text is matched and vectors ranked as they were then. Full-text scores use the corpus as it stood at the timestamp. The cost grows with the number of nodes written since the timestamp.

Searches at the current state follow the same rule for writes the indexes have not caught up with yet: the indexes are maintained in the background from committed writes, and a search never waits for them; the nodes those writes touched, and inside a transaction its own uncommitted writes, are read from the store and evaluated exactly beside the index.

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

A unique index that is still being built is protected the same way. A transaction that takes one of its values has its commit read the stored nodes the build has not reached, and a node there holding the value refuses the commit as a duplicate naming that node (gRPC `ALREADY_EXISTS` / `DUPLICATE_KEY`; in the embedded API `ExecutionError::UniqueViolation`). That read is bounded by `index_build_unique_read_limit`; past it the commit is refused as unresolved, not as a duplicate (gRPC `UNAVAILABLE` / `UNIQUENESS_UNRESOLVED` with a retry delay; `ExecutionError::UniquenessUnresolved`), and the same transaction succeeds once the build is done. An interactive transaction is judged at its commit, as a single statement is.

The exclusion holds in both orders. An edge attached to a node another transaction is deleting is refused when the deletion committed first, and the deletion is refused when the edge committed first: the deleting transaction decided on the edges it saw, and an edge that reached the node after it read them is one it never accounted for. This applies to `DETACH DELETE` too, for an edge of any type, including a type that did not exist when it read the node, so no edge is ever left pointing at a node that is gone.

## Retrying a Write Safely

A write whose reply is lost (a dropped connection, a timeout, a client restart, a change of leader) may or may not have committed. Running it again blindly can apply it twice; giving it up can lose it. Numbering the write removes the guess: a write that carries an **operation reference** takes effect at most once, however often and wherever it is sent again, and the database tells you how it stands.

**Numbering.** A logical client (an *issuer*, 16 bytes you choose once and keep across restarts) is enrolled once per incarnation with `OperationService.EnrollIssuer` (`Database::enroll_issuer` embedded). The reply names the incarnation, the first number to use and the deployment. The client then numbers its writes itself, without a round trip: each new write takes the next number, and a retry keeps the number of the write it repeats. An `OperationRef` is the issuer, incarnation, number and deployment. To survive its own crash, a client persists the reference before sending the write, and the next number with it, and continues with `ResumeIssuer` after restarting. Numbers are an identity, not an order: writes with different numbers may run concurrently and finish in any order.

**Carrying it.** The reference goes on `ExecuteCypherRequest.operation` for a statement that commits on its own, on `CommitTransactionRequest.operation` for an interactive transaction, on the session's `Execute.operation` and `Commit.operation`, and on the typed `CreateNode`, `CreateNodesBatch` (one reference for the whole batch) and `CreateEdge`. Embedded, it is `StatementOptions::operation` and `Database::commit_transaction_as`. A statement inside a transaction carries none: the transaction's reference is named on its commit, and a reference on such a statement is refused with `INVALID_ARGUMENT`.

**A retry.** When the write already took effect, the retry runs nothing and is refused with `FAILED_PRECONDITION`, `reason = OPERATION_SETTLED`. The metadata carries `outcome` (`COMMITTED`, `ABORTED`, `UNKNOWN` or `RETIRED`), `commit_ts` for a committed write, and `intent_differs = "true"` when the number was used before for a different write (other query text or parameters), which is a client bug rather than a retry. A commit repeated after its reply was lost answers the same way, even though its transaction handle is gone: it does not need the old buffer or a replay of its statements.

**Finding out.** `ResolveOperation` reports how an operation stands: `COMMITTED` with its commit timestamp, `ABORTED`, `UNKNOWN` or `RETIRED`. `UNKNOWN` means nothing is recorded *yet*; an attempt may still be on its way, so it is not proof that the write never happened. Resolving with `exclude = true` gives that proof: an operation no attempt has committed is excluded through the log, so the answer is final, `COMMITTED` or `ABORTED`, and an attempt arriving later applies nothing. Exclude before giving the write up or running the work again under a new reference.

**Volatile journals.** A write under `journal: MEMORY` or `CACHE` carrying a reference records its operation in the same log entry as its effects, like any write: the two are durable together or lost together. A write lost with the unsynced end of the log leaves its operation unrecorded, so resolving with `exclude = true` proves it never takes effect, and a retry under the same reference applies once.

**Releasing outcomes.** The database keeps each outcome until you release it. `AcknowledgeOperations` releases the outcomes of chosen numbers you no longer need, in any order, without waiting for the numbers below them; `RetireOperations` releases every number below a bound, including numbers never used. A released number reads as `RETIRED` and admits nothing again. Acknowledging never declares a number nothing is recorded of complete. Each incarnation admits a bounded window of numbers above its retired prefix (`operation_admission_window`, 65 536 by default); a write numbered past it is refused with `OPERATION_SETTLED` and outcome `UNKNOWN`, runs nothing, and is admitted once outcomes below it are released. Release what you no longer need to keep the window open.

**Where the reference is valid.** A reference belongs to the deployment that enrolled it. Another database refuses it with `INVALID_ARGUMENT`, `reason = FOREIGN_OPERATION`, rather than treat it as a new write. A standalone database that grows into a replicated group keeps its deployment, so references issued before the growth stay valid, and any member resolves them after a change of leader; a call that changes records (enrollment, a bound write, an exclusion, an acknowledgement, a retirement) sent to a follower is carried to the leader. A store restored from a `raft-snapshot` backup answers the outcomes the backup holds, but admits nothing new under the incarnations it held: enroll a new one. A `json`, `cypher` or `binary` backup carries the data but no operation records, so the store it is restored into is a new deployment: references issued before the backup are refused there with `FOREIGN_OPERATION`, and its clients enroll again.

**Scope.** Operation references identify writes; they do not authenticate the client. Any client allowed to write to the database can resolve, acknowledge or retire any issuer's operations, so use operation identity within a trusted set of clients until the database authenticates its callers.

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

| `journal` | The member counts once the write's log entry is | Survives | Latency |
|-----------|-------------------------------------|----------|---------|
| `JOURNAL` (default) | fsynced | process crash and power failure | the fsync |
| `CACHE` | handed to the operating system, synced later | process crash, not power loss before the sync | a write call |
| `MEMORY` | appended in the process, synced later | nothing before the sync | an in-memory append |

A write concern decides when the caller is answered, never whether or how the write is applied. Every write, `acks: 0` included, goes through the log and is applied on every replica in the same order; no concern writes to one node only.

`CACHE` and `MEMORY` defer the sync of the write's log entry, not the write itself, as MongoDB's `j: false` and PostgreSQL's `synchronous_commit = off` do. The write is validated, logged and applied exactly as under `JOURNAL`, schema changes and bound operations included; only the answer stops waiting for the sync. On a standalone node or an embedded database it is committed and visible to `local` reads the moment the statement returns, and the journal syncs it every `volatile_flush_interval_ms` (100 ms by default), sooner when a `JOURNAL` write or a table flush syncs everything before it. Reads under `majority`, `snapshot` or `linearizable` see it once it is synced: they read the latest state a crash cannot take back, without waiting or forcing a sync, and a read at a named timestamp that covers it is refused with `READ_NOT_READY` until then. In a replicated group the leader answers once its log holds the entry, before the sync, and the write commits once a majority holds it synced, as every write does. A crash before the sync loses the unsynced end of the log as a whole, in log order: never part of a write, and never a write whose successor survived. See [volatile writes](./configuration#volatile-writes). **Never select `CACHE` or `MEMORY` for data that you cannot reconstruct or afford to lose.** A volatile state cannot be confirmed across members, so these two are accepted only with `acks: 0` or `acks: 1`; combining them with a larger `w` is rejected with `INVALID_ARGUMENT`.

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
