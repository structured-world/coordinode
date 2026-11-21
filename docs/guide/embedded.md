---
description: "Run CoordiNode inside your Rust process with coordinode-embed: graph, vector and full-text queries without a separate database server."
---

# Embedded (Rust)

`coordinode-embed` runs CoordiNode in-process, as a library: no separate
database server and no network hop between your code and the engine. The
same directory format and Cypher dialect as the server, opened by your
process.

## Installation

The library is on crates.io as
[`coordinode-embed`](https://crates.io/crates/coordinode-embed), with the CE
packages it builds on (`coordinode-core`, `coordinode-storage` and the rest of
its closure), all released together under one version:

```toml
[dependencies]
coordinode-embed = "0.9"
# Query parameters and result values are coordinode_core types.
coordinode-core = "0.9"
# Only for Database::open_with_config (storage endpoints, limits, retention).
coordinode-storage = "0.9"
```

- **Compiler and targets.** Build with current stable Rust. Every release is
  installed from the registry and run on Linux with stable Rust before it is
  called ready; the workspace itself builds and tests on Linux, macOS and
  Windows with stable Rust. The declared minimum moves up when the code needs
  a newer compiler; older compilers are not tested.
- **Features.** The default build is a standalone database (see
  [Build features](#build-features)). `replication` adds membership in a
  replicated group; the io_uring storage backend is on by default on Linux.
- **What the registry checks run.** A registry release passes when the
  [quickstart example](#quickstart) and a broader consumer build from the
  published packages and run standalone, and the same consumer compiles with
  `replication`. The broader consumer covers unique identity under concurrent
  writers, revision and lease-epoch fencing, an atomic audit and outbox,
  rollback and reopen, an increment retried after a process exit and across a
  checkpoint and repair, a physical backup that settles it in the same
  history, a logical restore into a new domain, and the hard limit. A process exit is not a power loss. Compiling with a feature is not
  the same as running it: run your own scenario with the features you enable
  on the build you ship.

Other artifacts are separate: the Python package `coordinode-embedded`
(`pip install coordinode-embedded`) embeds the same engine for Python, and
the server binary runs it as a network service. A source build from the
[public repository](https://github.com/structured-world/coordinode) is a
different artifact from the registry packages: pin an exact reviewed
revision, never follow `main`, and qualify it separately. The workspace builds
with stable Rust; a source build with `replication` also uses the pinned proto
submodule and its generation tooling, which the registry packages carry
already generated.

The application owns its `Cargo.lock` and build profile. Check the features
your dependency graph actually selects: another dependency can unify features
the engine itself does not request.

The CE license is AGPL-3.0-only. See the repository's `LICENSE`, `COPYRIGHT`
and `NOTICES`; dependency licenses remain their own. Releases after 0.9.0
attach to their GitHub release a license inventory and a CycloneDX SBOM of
the graph the registry check installed, per toolchain, feature set and target.

## Quickstart

[`examples/embedded-quickstart`](https://github.com/structured-world/coordinode/tree/main/examples/embedded-quickstart)
is a complete application: its own `Cargo.toml` with the registry packages
and a `src/main.rs`. It opens a database on disk, runs parameterized
statements, commits a two-statement transaction, rolls another back, handles a
duplicate as a typed error, closes the database and reopens the directory to
check that the committed rows are there and the rolled-back ones are not.

```sh
cargo run -- my-quickstart-data
```

The directory belongs to the program: the first run creates the database
there, each later run adds its own rows under the next run number, and
nothing is ever deleted. Give it a directory of its own.

## Opening a database

`Database::open(path)` opens the database in `path`, creating it when the
directory is empty or absent: one durable endpoint with the default storage
settings. Statements run with `execute_cypher` and, with parameters,
`execute_cypher_with_params`; parameters are `coordinode_core` values, never
text spliced into the statement.

```rust
use std::collections::HashMap;
use std::path::Path;

use coordinode_core::graph::types::Value;
use coordinode_embed::{Database, DatabaseError};

fn find_person(dir: &Path, name: &str) -> Result<usize, DatabaseError> {
    let mut db = Database::open(dir)?;
    // Safe to repeat on every open: an equal constraint already in place is kept.
    db.execute_cypher(
        "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
    )?;
    let params = HashMap::from([("name".to_string(), Value::String(name.to_string()))]);
    db.execute_cypher_with_params("MERGE (:Person {name: $name})", params.clone())?;
    let rows = db.execute_cypher_with_params(
        "MATCH (p:Person {name: $name}) RETURN p.name AS name",
        params,
    )?;
    Ok(rows.len())
}
```

`Database::open_with_config(config)` opens from an explicit
`coordinode_storage` `StorageConfig`: several endpoints (one per disk, with
media, durability and tier), each endpoint's hard limit, the time-travel
retention window, the free-space reserve. The first endpoint's directory is
the database's primary directory.

```rust
use std::path::Path;

use coordinode_embed::Database;
use coordinode_storage::engine::config::{
    Durability, EndpointConfig, Media, StorageConfig, Tier,
};

fn open_capped(dir: &Path) -> Result<Database, Box<dyn std::error::Error>> {
    let endpoint = EndpointConfig::new("data", dir, Media::Ssd, Durability::Durable, Tier::Warm)
        // Every file of the database under `dir` stays within 1 GiB.
        .with_hard_limit_bytes(1 << 30);
    // Refuses an inconsistent topology (a reserve outside its limit, two
    // endpoints on one directory) by name.
    let mut config = StorageConfig::try_with_endpoints(vec![endpoint])?;
    // Time-travel reads reach one day back.
    config.retention_window_secs = 24 * 3600;
    // Writes stop while the file system has less than 512 MiB free.
    config.min_free_bytes = 512 << 20;
    config.resume_free_bytes = 1 << 30;
    Ok(Database::open_with_config(config)?)
}
```

The storage settings and their defaults are listed in
[Configuration](./configuration#storage-topology); the embedded host sets the
same fields on `StorageConfig` that the server reads from its config file.
`Database::open_in_memory()` opens a database that lives in memory only, for
tests and demos: nothing reaches the disk and nothing survives the process.

## Transactions and errors

A statement run on its own commits on its own. An interactive transaction
runs several statements against one snapshot and commits them atomically, or
rolls them all back:

```rust
use coordinode_embed::{Database, DatabaseError};

fn transfer(db: &Database, keep: bool) -> Result<(), DatabaseError> {
    let tx = db.begin_transaction()?;
    db.execute_in_transaction(tx, "CREATE (:Account {id: 'a', balance: 90})", None)?;
    db.execute_in_transaction(tx, "CREATE (:Account {id: 'b', balance: 10})", None)?;
    if keep {
        // Both accounts land at one commit timestamp, or neither does.
        let receipt = db.commit_transaction(tx)?;
        println!("committed at {}", receipt.commit_ts.as_raw());
    } else {
        // Nothing the transaction wrote is kept.
        db.rollback_transaction(tx)?;
    }
    Ok(())
}
```

Every call fails with a `DatabaseError` whose variant says what happened and
whose fields carry what a caller decides on: the transaction and why it
ended, the versions a conditioned write expected and found, the constraint
and node a write broke, the leader to retry at, the oldest readable
timestamp, an operation's outcome. Match on the variant; the rendered text is
for people. A refused write changes nothing.

```rust
use coordinode_embed::{Database, DatabaseError};

/// Creates the account, or reports that one with this email exists.
fn sign_up(db: &mut Database, email: &str) -> Result<bool, DatabaseError> {
    db.execute_cypher(
        "CREATE CONSTRAINT account_email IF NOT EXISTS \
         FOR (a:Account) REQUIRE a.email IS UNIQUE",
    )?;
    let params = std::collections::HashMap::from([(
        "email".to_string(),
        coordinode_core::graph::types::Value::String(email.to_string()),
    )]);
    match db.execute_cypher_with_params("CREATE (:Account {email: $email})", params) {
        Ok(_) => Ok(true),
        Err(DatabaseError::UniqueViolation { .. }) => Ok(false),
        Err(other) => Err(other),
    }
}
```

A refusal is one variant wherever it was found: a conflict is
`TransactionConflict` for an interactive commit and for a statement that
commits on its own. A storage failure no variant classifies is
`DatabaseError::Storage`: its message, and `storage_cause()` returns the
engine's error. A failure the caller cannot act on beyond reporting it is
`Other`. [Transactions](./transactions) covers the full interactive
transaction contract, conflicts, retrying a write safely under an operation
identity, and resolving a write whose outcome is unknown.

## Closing, reopening and durability

A `Database` is closed by dropping it. Dropping it stops and joins its
background work (index builds, the journal flusher, flushes and compactions),
syncs what was written under a volatile write concern, flushes the memtables
and releases the directory. It blocks while it does: from async code, drop it
in `tokio::task::spawn_blocking`, not on a runtime worker.

- **One handle per directory.** A directory a live `Database` holds is not
  opened a second time: the second open fails with `DatabaseError::Storage`
  naming the locked store, and the first handle goes on working. Share one
  `Database` within the process (`execute_cypher_shared` and the transaction
  methods take `&self`).
- **Open transactions.** A transaction that was not committed when its
  database was dropped is gone: nothing it wrote was durable, so nothing is
  undone.
- **Acknowledged writes.** The default write concern of `Database::open` is
  `journal: journal`: a write, or a transaction's commit, is answered only
  after its journal entry was synced to the medium (`F_FULLFSYNC` on macOS,
  `fdatasync` on Linux, `FlushFileBuffers` on Windows), and only then applied
  to the memtables. The directory of a new journal segment is synced before
  the first entry in it is (not on Windows, which has no directory sync). A
  transaction's effects are one journal entry, so they land together or not
  at all. It survives a process crash and a power failure. Weaker journal
  settings trade that for latency; see [Durability](./transactions#durability).
- **Crash recovery.** After a kill or power loss the next open recovers by
  itself: every acknowledged entry is replayed, a write cut off before it was
  acknowledged is dropped, and the database opens. Nothing is deleted by hand.

Reopen the directory with `Database::open` (or `open_with_config` with the
configuration it was opened with) once the earlier handle is dropped. A
member of a replicated group closes differently: `ReplicatedDatabase::shutdown`
hands leadership over, stops consensus and the listener, and releases the
store before it returns (see [Replication](#replication)).

## Disk space

Two independent controls bound what the database writes, and they answer
different questions:

| Control | Set with | Measures | What it guarantees |
|---------|----------|----------|--------------------|
| Free-space reserve | `StorageConfig::min_free_bytes` / `resume_free_bytes`; live with `db.engine().space().set_reserve(min, resume)` | Free space of the whole file system, other programs' files included | Writes stop before the disk itself fills, so the store keeps room to recover |
| Hard limit | `EndpointConfig::with_hard_limit_bytes` (and `with_recovery_reserve_bytes`) per endpoint, at least 8 MiB; live with `Database::set_hard_limit` | Space every file of the database under the endpoint's directory takes, as the file system allocates it | The database's own files never take more than the limit, whatever else is on the disk |

Under either control a write that does not fit is refused before it reaches
the journal (`CapacityExhausted` for the hard limit, a "no space" storage
error for the reserve), reads go on, and writes are admitted again once room
returns. The hard limit counts the journal, tables and indexes, full-text
segments, metadata, memtables waiting to be flushed, temporary files of
flushes and compactions, and local checkpoints; before refusing, it flushes
and retires the journal the tables cover. A limit below what the database
holds deletes nothing: the database opens or keeps running, serves reads and
refuses writes until it fits again. A replicated member is held to its own
limit the same way: a full group refuses writes through the leader and through
a follower alike, keeps its consensus running, and every member keeps
answering reads, at read concern majority too. What each control counts, the reserve
inside the hard limit, and its limits (files outside the endpoint directory,
backups written elsewhere, a store received from a replicated group) are in
[Hard limit: total database footprint](./configuration#hard-limit-total-database-footprint).
The engine cannot hold unrelated processes to a size: a host that needs a
system-wide cap also sets a file-system quota.

## Change-stream consumers

A consumer that reads the database's history (an outbox relay, a projection,
a backup reading state deltas) registers to record where it stands. Each
registration chooses its retention policy; there is no default:

- **`Ordinary`** records the consumer's position only and keeps nothing for
  it: the journal's own retention (and the MVCC retention window) decides how
  far back history reaches. Between reads the consumer holds back no
  reclamation and costs the store nothing. A position the store no longer
  holds is reported as `RetentionLost` with the position and where history
  starts now, never skipped and never replaced by the newest entry; the
  registration stays, and the consumer rebuilds its state from a complete
  read of the store and then acknowledges a position the store holds.
- **`Strict`** keeps what the consumer has not acknowledged until the
  consumer is cancelled. Missing heartbeats do not end it. It is for a
  consumer that must never miss history (a backup, an exact replica): the
  history it holds counts toward the store's limits, so at a hard limit the
  store refuses new writes rather than drop it. A new `Strict` registration
  is refused while the store has no room for more history.

A log consumer (`ConsumerKind::OplogEvents`) reads the journal; a state or
snapshot consumer (`LsmStateDelta`, `MvccSnapshotPin`) the MVCC history. A
`Strict` position is held from the registration on, across restarts, and a
registration at a position already gone is refused with `RetentionLost`:

```rust,no_run
use coordinode_embed::Database;
use coordinode_embed::consumer::{
    ConsumerKind, ConsumerRegistration, ConsumerRetentionPolicy, InitialSeqno, TopologyScope,
};

# fn main() -> Result<(), coordinode_embed::DatabaseError> {
let db = Database::open("/var/lib/myapp/graph")?;
let outbox = db.register_consumer(ConsumerRegistration {
    consumer_id: "outbox-relay".into(),
    kind: ConsumerKind::OplogEvents,
    scope: TopologyScope::Shard(0),
    initial_seqno: InitialSeqno::FromNow,
    retention: ConsumerRetentionPolicy::Strict,
})?;
// Before reading: where the consumer stands, refused if that history is gone.
let from = db.consumer_position(&outbox)?;
// ...deliver what follows `from`, then acknowledge up to where it was handed
// over; what lies below is no longer kept for this consumer.
db.acknowledge_consumer(&outbox, from)?;
# Ok(())
# }
```

After a restart the consumer takes its registration up again with
`resume_consumer(id, incarnation)`; the incarnation is in its handle
(`RegisteredHandle::incarnation`). A cancelled registration keeps refusing
its handle with the reason and last acknowledged position
(`RegistryError::Terminated`, with where the history started when it ended),
and registering the same id again starts a new incarnation: a resync. Its
listing shows what it followed (`follows`: the ended incarnation, why it
ended, what it had acknowledged) and where it started (`started_at`);
`follows.missed(started_at)` is the range of positions no incarnation was
given, `None` when the resync started at the last acknowledgement. A resync
never starts anywhere on its own: the consumer chooses `At`, `FromNow` or
`FromEarliestRetained`. `cancel_consumer` ends one; `consumers_list` shows every
registration with its policy, state, position, how long the oldest journal
entry it has not acknowledged has waited and the journal bytes from its
position (the MVCC history is not measured per consumer, and shows as
unavailable, never as zero), and whether it is the one holding the history
back. `consumer_retention` sums it up per space (journal, MVCC): the lowest
`Strict` position, the journal bytes kept from it counted once however many
consumers share them, which registrations hold it, and why a new `Strict`
registration would be refused now. A `Strict` registration is new history to
keep, so it is refused as `Backpressure` with its reason while compaction is
behind (`WritePressure`), the volume is below its free-space reserve
(`FreeSpace`), or an endpoint's hard limit has no room left for client
writes, only the reserve recovery needs (`Footprint`); retry once it eases.
Releasing one consumer can free nothing when another, a live reader or
recovery still needs the same history.

A database opened over registrations of a retention policy this version no
longer offers refuses to open its registry with an error naming the record,
rather than reuse them under another meaning.

The registry runs on a thread of the database and stops when it is dropped;
no async runtime is needed. A database kept in memory, and a member of a
replicated group (whose history is the group's log, not a journal of its
own), refuse registrations as `Unsupported`.

### Reading changes as events

`read_changes` hands a reader what was committed after its cursor as logical
events, unit by unit (a unit is one committed transaction). Each event is a
`Put` (the record or properties written, as the payload), a `Patch` (a
document delta) or a `Remove` of one target: a node, a relationship's
presence, a relationship's properties, or one version or discriminated
instance of a relationship. A put states what was written, not whether the
target was new; a node or relationship is never read back to tell. The
reverse side of a relationship, index entries and history rows are not
events of their own.

A `ChangeInterest` names the labels whose nodes the reader follows and the
relationship types it follows (`None` takes all of that kind, an empty list
none), and whether it follows blob objects (`blobs`): what each owner holds
(an object from `upload_blob`, an S3 object, a node's large property), as a
`Put` of `OwnedTarget::BlobOwner` with the owner record as its payload and
a `Remove` when it goes; `BlobId::owner_key` names an uploaded object's
owner. A blob's bytes are not in the event. A node event reaches the reader when the node carries one of its
labels before or after the change, with its `delivery`: `Enter` (it joins the
scope, with the record written), `Apply`, or `Leave` (deleted, relabelled or
retyped out of the scope). The labels before a deletion or relabelling come
from the transaction itself, so a removal reaches its readers after the
deleted record is gone. A reader of several labels gets each event once.

A label a type implies is followed through the nodes of every type implying
it: a node's labels in an event include its type's implied ones, as the type
implied them when the change was committed, so a reader reading old changes
after `ALTER NODE TYPE ... SET IMPLIES` routes them as they were. The change
itself is one `ScopeChange` event per type whose implied labels it moves (a
type implying the altered one included), with an `OwnedTarget::Scope` naming
the type and the activation that put the change in force: `Enter` for a
reader of a label the type comes to imply, `Leave` for one it stops
implying, `Apply` for a reader of the type itself. No node event is written
for the type's nodes. A reader keeping state by label reconciles the type's
nodes once per activation, or rebuilds that scope; every activation has its
own identity, a change back to an earlier declaration included.

Dropping a label (`DROP TABLE`) delivers a `Drop` of `OwnedTarget::Label`,
naming the identity the label had, to its readers as `Leave`, whether or not
nodes of it were left; nodes the same transaction removes are removals of
their own. A label declared again under the same name is another label: its
readers do not receive the old one's drop.

Each unit comes whole, with its position and commit timestamp, the events
the interest takes and `projected`, the number of events the unit has in
all; a unit with none of the reader's events is in the batch too. An event's
`id` (unit position and ordinal) is the same for every reader and every read
of it, so a reader that delivers twice can drop the repeat by it. The
cursor is opaque bytes to save with what the reader built (`to_bytes`,
`ChangeCursor::from_bytes`); it binds the history, the event interpretation
and the interest, and holds nothing of the journal. Its `position` is the
journal position a registered consumer acknowledges.

```rust,no_run
use coordinode_core::budget::QueryBudget;
use coordinode_embed::Database;
use coordinode_embed::changes::{ChangeCursor, ChangeInterest};

# fn main() -> Result<(), coordinode_embed::DatabaseError> {
let db = Database::open("/var/lib/myapp/graph")?;
let interest = ChangeInterest {
    labels: Some(vec!["Person".into()]),
    edge_types: Some(vec!["KNOWS".into()]),
    blobs: false,
};
let mut cursor = db.change_cursor(&interest)?;
let budget = QueryBudget::new(64 << 20);
loop {
    let page = db.read_changes(&cursor, 256, &budget)?;
    for unit in &page.batch().units {
        for event in &unit.events {
            // ...apply `event` to the projection.
            let _ = (event.id, event.action, event.delivery);
        }
    }
    let next = page.batch().cursor.clone();
    if next == cursor {
        break;
    }
    cursor = next;
    // ...store `cursor.to_bytes()` with what was applied, in one write.
}
let saved = cursor.to_bytes();
let _resumed = ChangeCursor::from_bytes(&saved).expect("a saved cursor");
# Ok(())
# }
```

Only units this database applied and made durable are read. A cursor whose
position the journal no longer holds is `RetentionLost`, never moved forward
on its own; a cursor of another database or of another event interpretation
is refused (`ChangeCursor`); a unit the feed cannot read, such as a change
recorded without the context that says what it means, is
`ChangeFeedUnsupported`; a first entry that does not fit the budget alone is
`Budget`. On each the cursor stays valid. A member of a replicated group
reads the group's log up to what it applied.

A feed is of the namespace the database serves: its cursor names that
namespace, and the transactions of other namespaces sharing the store's log
are passed over without a trace in the batch. A cursor of another namespace
is another stream; one carried to a database of another namespace reads
nothing of it. A transaction writes one namespace; a log unit spanning
several, which no transaction produces, is `ChangeFeedUnsupported`.

A transaction with more events than one read holds is delivered in
fragments: each `UnitChanges` covers the ordinals `first..through` of its
transaction, `completes_unit()` tells the last one, and the cursor continues
inside the transaction. Compare cursors, not positions, to tell whether a
read moved.

### Keeping a projection

A projection kept in a database (the same one or another) can commit its
effects and its progress through the feed together, so a crash leaves either
the previous complete projection or the new one, never part of a
transaction. `start_projection` records where it starts; `apply_changes`
applies a batch in one transaction of the destination: the closure runs for
each transaction of the batch with the host's writes for it, and the
projection's progress moves past the batch in the same commit.

```rust,no_run
use coordinode_core::budget::QueryBudget;
use coordinode_embed::Database;
use coordinode_embed::changes::{ChangeInterest, OwnedTarget};

# fn main() -> Result<(), coordinode_embed::DatabaseError> {
let source = Database::open("/var/lib/myapp/graph")?;
let projection = Database::open("/var/lib/myapp/projection")?;
let start = source.change_cursor(&ChangeInterest::default())?;
let progress = projection.start_projection("people", &start)?;
let budget = QueryBudget::new(64 << 20);
let page = source.read_changes(&progress.received, 256, &budget)?;
let progress = projection.apply_changes("people", page.batch(), |tx, unit| {
    for event in &unit.events {
        if let OwnedTarget::Node(_) = &event.target {
            // ...the projection's writes for this event.
            tx.execute("CREATE (:Seen)", None)?;
        }
    }
    Ok(())
})?;
// The next read starts here, in this process or after a restart.
println!("next read from {:?}", progress.received);
# Ok(())
# }
```

The progress has two positions. `received` is where the next read starts.
`applied` is past every change that has taken effect. They differ only while
a transaction arrives in fragments: each fragment is staged in the
destination as it is received, and the batch carrying the last one runs the
closure for every staged fragment in order and then for the last, all in its
one transaction. Until then the projection's state shows nothing of that
transaction. A batch must be read from the projection's `received` cursor;
one read from anywhere else, such as the same batch applied again, is
refused as `ProjectionOutOfStep` with nothing applied. A failed closure or
commit applies nothing, and the batch is applied again from the same cursor.
A projection's progress is its own: it does not hold the source's journal,
which follows its consumers' acknowledgements. A `ScopeChange` event moves a
type's nodes in or out of label scopes with no event per node: a closure
returning `Ok` for it claims the projection reconciled them; one that cannot
returns an error, and the projection stays before the change until it is
rebuilt.

### Rebuilding after `RetentionLost`

A consumer whose position is gone rebuilds its projection from a fixed cut
of the database over one row scope (`CutScope::rows`: a partition and a key
prefix): a read view of the scope pinned below a bound, and the log
continuation that joins it. A cut recovers rows only, not the whole
database, a columnar table or an index; the index and schema partitions are
refused as scopes. `state_cut` takes the cut, waiting up to its timeout for
the commits below the bound to finish and be durable (`CutNotReady`
otherwise). `state_cut_scan` pages through the view, each page starting
where the last ended; `state_cut_tail` reads the continuation from where the
last read ended and hands over only the units the view does not already
hold, each with its own commit timestamp and its changes inside the scope,
so a non-idempotent change (a counter increment) is applied once; a range
delete is clipped to the scope. It reads only entries the database applied
and made durable, so a write whose outcome is still open is not handed
over. Both take a row or entry count and a memory budget (`QueryBudget`):
a read keeps within it, and what it hands over stays charged to it until
the page or batch is dropped; a first row or entry that does not fit the
budget alone is refused as `Budget`, without moving the continuation. A
position the log no longer holds while the copy runs is `RetentionLost`:
take a new cut. An effect inside the scope the cut cannot carry is
`CutUnsupported`, and the continuation does not move past it.

A cut is of the namespace the database serves: its view holds that
namespace's rows and its continuation that namespace's units, even where
another namespace holds rows under the same keys. A database of another
namespace refuses to page through it, and a resume point reached in one
namespace is refused in another.

Copy the view into a new generation of the destination, apply the
continuation, and publish the generation together with the consumer's
progress in one destination write, only once `StateCut::joined` returns
where reading resumes: publishing earlier would certify a prefix the view
already passed. Building a new generation rather than updating the old one
in place is what removes the rows deleted at the source. The resume point
names the history it counts entries of (the database's own journal, or its
group's log) and the scope: `resume_position` gives back the log position
for the same history and scope and refuses it for another. The cut pins its
view until it is dropped. A cut of a database closed since, or whose scope
was repaired meanwhile, is `CutLost`, and a cut read against another
database is refused.

```rust,no_run
use std::time::Duration;
use coordinode_core::budget::QueryBudget;
use coordinode_embed::Database;
use coordinode_embed::consumer::CutScope;
use coordinode_storage::engine::partition::Partition;

# fn main() -> Result<(), coordinode_embed::DatabaseError> {
let db = Database::open("/var/lib/myapp/graph")?;
let scope = CutScope::rows(Partition::Node, b"node:".to_vec())?;
let mut cut = db.state_cut(scope.clone(), Duration::from_secs(10))?;
let budget = QueryBudget::new(64 << 20);
let mut after: Option<Vec<u8>> = None;
loop {
    let page = db.state_cut_scan(&cut, after.as_deref(), 1024, &budget)?;
    let Some(last) = page.last_key() else { break };
    after = Some(last.to_vec());
    // ...write `page.rows()` into the destination's new generation.
}
let resume = loop {
    let batch = db.state_cut_tail(&mut cut, 256, &budget)?;
    // ...apply `batch.units()` to the new generation.
    if let Some(resume) = batch.joined() {
        break resume.clone();
    }
};
let progress = db.resume_position(&resume, &scope)?;
// ...publish the new generation and `progress` in one destination write.
# let _ = progress;
# Ok(())
# }
```

A member of a replicated group (with the `replication` feature) takes its
cut of the group's log the same way: the view is below the group's closed
bound, and the continuation reads the log this member applied.

## Blobs

Large binary objects are stored by id beside the graph. Each upload is its own
blob, and identical chunks of different blobs are stored once:

```rust
use coordinode_embed::{Database, DatabaseError};

fn store_and_read(db: &Database, bytes: &[u8]) -> Result<Vec<u8>, DatabaseError> {
    let uploaded = db.upload_blob(bytes, None)?;
    let Some(reader) = db.open_blob(&uploaded.id)? else {
        return Ok(Vec::new());
    };
    // The reader holds the state it opened at until it is dropped: a delete
    // meanwhile does not take the chunks from under it.
    let mut read = Vec::new();
    for index in 0..reader.record().blob.chunk_count() {
        read.extend(reader.chunk(index)?);
    }
    db.delete_blob(&uploaded.id)?;
    Ok(read)
}
```

With a client operation (`Some(&descriptor)`), an upload takes effect at most
once and names the same blob on every attempt (`BlobId::for_operation`); a
retry of an operation that already stored its blob is refused with
`DatabaseError::OperationSettled`. A replicated member whose group holds
other members refuses blob writes with `DatabaseError::Unsupported`: blob
bytes are held only on the member that writes them.

Deleting a blob ends its hold on its chunks. The database collects the chunks
nothing holds any more on its own thread, woken by the deletion: after a grace
period (24 hours by default, never less than the retention window), and once
no reader, retained history or upload in progress needs them, their bytes are
deleted and the storage compacted, which frees their space. A chunk another
blob shares stays. An upload never published, such as one interrupted by a
crash, is collected an hour after it began. Nothing has to be called for any
of this; the same collector also answers on demand:

```rust
use std::time::Duration;

use coordinode_embed::{Database, DatabaseError};
use coordinode_embed::db::GcPolicy;

fn collect_now(db: &Database) -> Result<u64, DatabaseError> {
    // Keep released chunks for two hours instead of a day.
    db.set_blob_gc_policy(GcPolicy {
        grace: Duration::from_secs(2 * 60 * 60),
        ..db.blob_gc_policy()
    });
    // A pass run now is the one the collector runs by itself: it never
    // shortens the grace or passes a reader, history or an upload.
    let pass = db.run_blob_gc(false)?;
    let status = db.blob_gc_status()?;
    println!(
        "{} retired this pass; {} candidates waiting, {} bytes on disk",
        pass.retired, status.candidates, status.blob_disk_bytes
    );
    Ok(status.reclaimed_bytes)
}
```

`blob_gc_status` reports what waits (released chunks, unpublished uploads,
retired copies awaiting deletion), when the oldest of each began, whether a
compaction still owes space back, what has been reclaimed (payload bytes
deleted, and bytes the storage shrank by), and why collection is stopped if it
is: a record that does not decode makes ownership unknown, and nothing is
collected until it is gone. A member of a group that holds other members
collects nothing. A hard-linked checkpoint keeps the files it shares until it
is pruned.

## Backup and recovery

There are two kinds of backup, and they recover different things.

| | Physical backup | Logical export and restore |
|---|---|---|
| Made with | `Database::physical_backup(target)` | `coordinode_embed::backup::export` (binary, JSON or Cypher), then `Database::restore` into an empty database |
| Holds | The whole database as of the call: data, schema, indexes, operation records | Nodes, edges and schema at one snapshot, with their node ids |
| Opened as | `Database::open(target)`, the same database | A new database, a different domain |
| An operation committed before the backup | Settled: its retry runs nothing | Refused with `ForeignOperation`: never run again, never answered |

```rust
use std::path::Path;

use coordinode_embed::{Database, DatabaseError};

fn back_up(db: &Database, backups: &Path) -> Result<(), DatabaseError> {
    // `target` must not exist yet; put it on another device than the database.
    db.physical_backup(&backups.join("2026-10-26"))?;
    Ok(())
}
```

A physical backup is taken while writes go on. On the same file system its
files are hard links to the database's own, so it costs little space but
shares the device; elsewhere they are copies. What the database holds in
memory is written into the backup, not flushed into the database first, so
a backup outside the endpoint directory grows nothing under it: a database
at its hard limit, refusing writes, still writes its backup to another
directory. A statement that writes `STORAGE COLUMNAR` rows, or creates or
drops such a table, lands in steps no single commit gathers; a backup
waits for one under way, as for a repair rebuilding a table, and holds it
whole (after ten seconds it is refused instead and can be taken again).
A backup is refused when the catalog and the tables disagree, as a drop
whose transaction was rolled back after it removed the table's rows
leaves them: drop the table again, then take the backup. Backups outside
the endpoint directory are not counted against the hard limit. The backup
directory is a
complete database in one directory, whatever endpoint layout the original
had, and opens with plain `Database::open`.

Recovering from a physical backup means running it in place of the lost
database. Operations the backup holds are settled there exactly as in the
original. An operation committed after the backup is unknown to it and runs
once when retried: the backup does not have that write. Never run a backup
beside the database it was taken from: both would answer the same descriptors
from histories that have diverged. Results acknowledged after the backup
(`acknowledge_operation`) are back to settled in it; what the application does
about effects it already performed outside the database after the backup is
its own recovery policy.

The history temporal types keep past the retention horizon (see `AS OF
TIMESTAMP` in the Cypher extensions) belongs to the database's deployment.
A physical backup opens as the same deployment and answers the same reads
past the horizon; a database refuses to install a snapshot recording another
deployment's history, or history without the deployment that recorded it.

A logical export is for moving data into a new database, on another version
or with another layout. The restored database issues descriptors of its own
and refuses the old ones. See [Transactions](./transactions) for operation
identity and resolving a write whose outcome is unknown.

`Database::checkpoint` is different again: it keeps repair bases under the
database's own directory, for `verify_and_repair` to rebuild a damaged
partition or `STORAGE COLUMNAR` table from, and protects against local
corruption only, not the loss of the device. A tree whose open fails
structurally (a lost manifest) is repaired from its tables at open instead of
failing the database; when that repair had to drop damaged data, the next
`verify_and_repair` rebuilds the tree from the latest checkpoint plus the
journal, and reports it as `corrupt_partitions` / `corrupt_tables` with
nothing repaired when no checkpoint exists.

A partition or table is rebuilt with the history its checkpoint copy holds
(older versions and deletions keep their timestamps, and the journal is
replayed at each entry's commit timestamp), and only when the checkpoint and
the journal prove its state. Otherwise it is left as the open repair left it
and listed in `refused_partitions` or `refused_tables` with the reason: part of
the journal after the checkpoint is gone, the checkpoint holds no copy of the
table, or the table was dropped, re-created or altered after the checkpoint.
The report is then not clean, and the data is restored from an off-device
backup.

The deployment's records (the `deployment` partition under `system/`) are
settled before anything else the open reads, since they hold where every
other partition's files live, the database's own records and the commit
ceilings of its log. When that partition is owed a rebuild (its open repair
dropped data, a rebuild of it was cut short, or the open's verification of a
database with a checkpoint finds a corrupt block), the open rebuilds it from
the latest checkpoint and the journal itself, before the other partitions
are replayed. The checkpoint is copied for this, never changed. The open
fails with `DeploymentRecoveryRefused` instead, leaving the directory, its
checkpoints and its journal as they are, when the database's own records in
the partition cannot be shown intact (a repair dropped data whose range
could hold them), when the journal after the checkpoint is incomplete, or
when the checkpoint's copy is itself damaged. Without a checkpoint, a
database whose own records are intact opens and reports the partition for
repair as above.

## Build features

- **Default (standalone).** No consensus, inter-node transport, listener or
  async runtime is compiled. The `Database` calls are synchronous.
- **`replication`.** A database that is a member of a replicated group, in
  `coordinode_embed::replication`. The host supplies a multi-threaded Tokio
  runtime; the member serves its peers on a listener of its own (consensus,
  snapshot and segment transfer, under mutual TLS node identity), never a
  client protocol. A build with the feature still runs standalone until it is
  given peers.
- **`io-uring`, on by default.** The storage engine opens on one shared
  io_uring ring instead of synchronous `pread`/`pwrite`, on Linux 5.6 and
  later. Elsewhere it is a no-op; a kernel without io_uring makes the engine
  log a warning and use synchronous I/O. It comes with the storage crate's
  default features.

Enabling a feature changes the build, not the store format: a store written by
a standalone build becomes the member a group is formed around in place (see
below).

### Analyzer upgrades

This release pins `structured-stemmers` 1.2.0, generated from Snowball main
at [`6389c79f`](https://github.com/snowballstem/snowball/commit/6389c79f91cd67ed2de3ebdcf84ec8c2d2d23c11).
Its algorithms differ from the previous Git dependency: for example, English
`added` now stems to `add` instead of `ad`. When upgrading existing data, rebuild
full-text indexes and regenerate encrypted stem tokens from their source text
before serving them with this analyzer. Do not mix analyzer revisions within an
index generation or assume that old postings become compatible by reopening
the database. Package installation does not perform this data migration.

## SQL admission and host stack

SQL execution and SQL fingerprinting share the same admission path. The host
thread must provide at least a 2 MiB stack. Before constructing an AST, the
frontend rejects input larger than 1 MiB, more than 4096 non-whitespace tokens,
or more than 64 structural tokens. Structural tokens are keywords and
punctuation other than commas and semicolons; non-keyword identifiers and scalar
string/number literals consume width without adding structural cost. Quoted
literal and comment contents are interpreted by the SQL lexer, not counted as
SQL operators. The parser additionally enforces 16 recursive entries.

These are independent safety ceilings, not a promise that every expression
within the token count is supported. Long flat operator chains need the
structural bound even without parentheses, because formatting and destruction
also traverse the AST. A wide simple projection has a different shape. Exceeding
a ceiling returns an explicit error before execution, without applying a prefix
of a statement. Both source and registry packages use these same bounds, without
native stack switching or an extra thread per request.

## More operations

```rust
use coordinode_embed::Database;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::{IssuerId, Resolve};

fn time_travel_and_safe_retry(db: &Database) -> Result<(), Box<dyn std::error::Error>> {
    // Time travel is bounded by the MVCC retention window (default five
    // minutes, `StorageConfig::retention_window_secs`). Tune it at runtime; a
    // read older than `oldest_readable_timestamp()` fails with
    // `OutsideRetention` instead of answering from collected history.
    db.set_retention_window(std::time::Duration::from_secs(24 * 3600));
    let _horizon = db.oldest_readable_timestamp();

    // A write that takes effect at most once however it is retried: number
    // it with an enrolled issuer, persist the descriptor (`to_bytes`) before
    // running it, and after a lost reply or a restart resolve the same
    // descriptor. An issuer that ran out of numbers refuses `next` with
    // `IssuerExhausted`.
    let issuer = db.enroll_issuer(IssuerId(*b"payments-worker1"))?;
    let op = issuer.next()?;
    db.execute_cypher_shared_with(
        "CREATE (:Payment {amount: 10})",
        None,
        None,
        &StatementOptions { operation: Some(op), ..Default::default() },
    )?;
    let _outcome = db.resolve_operation(&op, Resolve::Lookup)?; // Committed { .. }
    db.acknowledge_operation(&op)?; // release the outcome once it is no longer needed
    Ok(())
}
```

See [Transactions](./transactions) for safe retries with operation identity
and what the retention window costs in storage.

## Declaring the Schema

An application declares its labels, edge types and constraints as one package, typically at every open. `Database::publish_schema` checks the whole package against one view of the catalog, with the package itself visible to it (a constraint may name a label the same package defines, in any order), and publishes every change in one catalog commit, or nothing.

```rust
use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::{
    Database, DatabaseError, EdgeTypeDeclaration, Intent, LabelDeclaration, PackageConstraint,
    SchemaPackage,
};

fn declare_schema(db: &Database) -> Result<bool, DatabaseError> {
    let mut user = LabelSchema::new_node_id("User");
    user.set_mode(SchemaMode::Strict);
    user.add_property(PropertyDef::new("email", PropertyType::String));
    let package = SchemaPackage {
        labels: vec![LabelDeclaration { schema: user, intent: Intent::Ensure }],
        edge_types: vec![EdgeTypeDeclaration {
            schema: EdgeTypeSchema::new("FOLLOWS"),
            intent: Intent::Ensure,
        }],
        constraints: vec![PackageConstraint {
            name: Some("user_email".into()),
            label: "User".into(),
            properties: vec!["email".into()],
            kind: ConstraintKind::Unique,
            scope: None,
            on_duplicate_rename: None,
        }],
    };
    let published = db.publish_schema(package)?;
    // Bounded: work outwaited goes on, and the answer says it is pending.
    let readiness = db.await_schema(&published, std::time::Duration::from_secs(30))?;
    Ok(readiness.all_ready())
}
```

- **Ensure, change, omit.** `Intent::Ensure` creates an absent object, keeps an equal one and refuses a different one with `CatalogObjectDiffers`, which carries the revision a change would name. `Intent::Change { expected_revision }` changes the definition at that revision and is refused with `CatalogRevisionMismatch` at any other. An object the package does not mention is never dropped.
- **Nothing to do, nothing done.** A package equal to what is in place commits nothing, scans nothing and starts no build; every object comes back `Retained`. Opening an application whose schema did not change costs one catalog read.
- **Acceptance is not readiness.** A definition that could refuse a stored node (STRICT, VALIDATED, or a NOT NULL property) is published as a candidate and proved over the stored nodes in the background; the rules in force before stay in force until the proof puts it in force, and every write meanwhile must satisfy both. A uniqueness is proved by its index build. `PublishedObject::operation` names that work; `await_schema` waits for it with a bound and reports `Ready`, `Pending` or `Failed`. A failed proof withdraws its own candidate or constraint only; the rest of the package stays published. Both kinds of work survive a restart.
- **Lost replies.** `Database::publish_schema_as(package, &descriptor)` publishes as a client operation: a retry of a publication that committed is answered with `OperationSettled` and its outcome instead of publishing again.
- **Bounded.** A package carries at most `MAX_SCHEMA_PACKAGE_DECLARATIONS` (4096) declarations; a larger one is refused, never split into several commits.

`define_label`, `create_label_schema`, `define_edge_type` and `create_edge_type_schema` publish one object through the same core. A single-object call that publishes a candidate waits for its proof within the statement wait (`index_build_statement_wait_secs`), as `CREATE CONSTRAINT` waits for its build. Over gRPC the package is `SchemaService.PublishSchema`, awaited with `AwaitSchema`.

## What Is and Isn't Available

The embedded API exposes the full query engine (graph, vector, full-text, time-series) with the same Cypher dialect as the network server. Only network-facing features (gRPC server, REST transcoding, Prometheus scrape endpoint) are absent; replication between embedded members is available behind the `replication` feature (see below).

## Replication

The same database can run as a member of a replicated group inside your process. Three things are separate:

- **Build capability.** Replication is compiled in only with the `replication` feature. Without it the crate pulls in no consensus, network or async runtime code at all.
- **Operating mode.** A build with the feature still runs a single member until you give it peers. A member alone leads its group at once, takes writes through its own log and serves nothing on the network.
- **History.** A store written without replication becomes the member a group is formed around, in place: no export, no import. Members that join start empty and receive everything from the leader. A member that already holds data of its own is refused as a joiner. Growth goes one way: once a store is a member, its history is the group's log, and `Database::open` (with or without the feature) refuses it, writing nothing, with a `DatabaseError::Storage` whose `storage_cause()` is `StorageError::GroupMemberStore`; it opens only as a member, through `ReplicatedDatabase`.

```toml
[dependencies]
coordinode-embed = { version = "0.9", features = ["replication"] }
coordinode-storage = "0.9"
```

```rust
use std::net::SocketAddr;
use std::sync::Arc;

use coordinode_embed::replication::{
    Founding, MemberOptions, PeerIdentity, PeerTimeouts, Peering, ReplicatedDatabase,
    ReplicationError, SecurityContext,
};
use coordinode_storage::engine::config::StorageConfig;

/// One member, no peers: it leads alone and opens no listener. Called on a
/// multi-threaded Tokio runtime.
async fn alone(storage: &StorageConfig) -> Result<(), ReplicationError> {
    let db = ReplicatedDatabase::open(storage, MemberOptions::new(1), None).await?;
    db.database().write().execute_cypher("CREATE (:Person {name: 'Alice'})")?;
    db.shutdown().await
}

/// A member that forms a group with peers: it serves them on `listen`, and
/// its node identity (mutual TLS) comes from `security`: this member's
/// certificate and the group's CA. Member 2 joins with the enrollment this
/// member hands out; the leader adds it as a learner and promotes it to a
/// voter once it caught up.
async fn form_group(
    storage: &StorageConfig,
    security: Arc<SecurityContext>,
    listen: SocketAddr,
    identity_of_2: PeerIdentity,
) -> Result<(), ReplicationError> {
    let mut options = MemberOptions::new(1);
    options.peering = Some(Peering {
        security,
        advertise: "10.0.0.1:7080".into(),
        founding: Founding::Form { deployment: None },
        timeouts: PeerTimeouts::default(),
    });
    let leader = ReplicatedDatabase::open(storage, options, Some(listen)).await?;
    // Handed to member 2, which opens with `Founding::Join(enrollment)`.
    let _enrollment = leader.enrollment("10.0.0.1:7080");
    leader.add_learner(2, "10.0.0.2:7080", identity_of_2).await?;
    leader.promote(2).await?;
    leader.shutdown().await
}
```

An already open `Database` becomes a member on the same store with `ReplicatedDatabase::from_database`: what it committed is all there under the replicated handle.

Behaviour every host can rely on:

- Only the leader applies writes. A write made through `database()` on another member, whatever its write concern, is refused with `NotLeader` naming the leader when it is known, and lands on no member. A restore is refused the same way: it runs on the leader, and its records, edges, schemas and the indexes built over them reach every member. To have a write run at the leader instead, send it through the member's ingress (below).
- `leadership()` streams leadership changes; `transfer_leadership(id)` hands leadership to a named member (for example before a laptop sleeps) and returns once that member leads.
- A member cut off from its group asks the others whether they would vote before it starts an election, so its return does not depose a leader the rest still follow.
- `shutdown()` stops the member, its listener and its background work, and closes the store before it returns, so the same directory can be opened again at once. Dropping the handle stops the background work too, without waiting for it.
- A member that opens with `Founding::Join` starts with no namespace of its own. Its listener serves the group from the open, and `ReplicatedDatabase::open` returns once the leader has added the member and its namespace has arrived with the group's state; a host that adds members from the same process opens the joiner and calls `add_learner` on the leader concurrently. A command a peer routes to a member before then is refused with `DatabaseError::NotServing`. A store in which a namespace was created, even an empty one (opening a `Database` on it creates one), is refused as a joiner with `RaftNodeError::JoinWithLocalData`.
- A member keeps its vote twice, in its consensus log store and in a second copy (`consensus-vote.N` files at the root of its store), both synced before a vote is answered. One lost or damaged copy is restored from the other at open. A member that lost every copy does not open: `ReplicatedDatabase::open` fails with `ReplicationError::Consensus(RaftNodeError::PromiseUnqualified(_))` and its data stays in its directory, because a member that forgot a vote could vote twice in one term. A store opened under another node id or group than the one its vote was given as is refused before anything opens on it: `RaftNodeError::Identity` when it forms or rejoins a group, `RaftNodeError::JoinWithLocalData` when it joins one, since a joining member needs an empty store.
- Such a member comes back only as a new member: open a new node id, under its own identity, on an empty store as a joiner, and call `replace(old, new, addr, identity)` on the leader. The new member joins as a learner; once it has caught up, one membership change makes it a voter and removes the old one, whose identity is retired. A call that failed or was cut short is resumed by calling again with the same arguments, and a repeated call adds no second member. A call that finds the group changed while the new member caught up acts on nothing it was not asked about: it fails with `ReplicationError::Consensus(RaftNodeError::Identity(_))` when the new node id is a member under another identity or the old node id now belongs to another member, and with `ReplicationError::Consensus(RaftNodeError::Membership(_))` when the voters changed under it (calling again resumes). Each refusal comes before the call's next membership change; steps it already committed stay and are not undone. It returns `Ok` only once the membership showing the replacement is committed. The old directory is left as it was for the host to inspect and delete. A replacement needs the quorum: a group that lost a quorum of votes at once stays unavailable and is recovered from a backup.

### Working through any member

A host can send every command through the member it opened, whichever member leads. `ReplicatedDatabase::execute` runs a statement here when this member may serve it and at the leader otherwise; the leader evaluates, validates and commits it, and the answer (columns, rows, what it wrote, its commit timestamp) is the one the leader gave. A follower never applies a write it evaluated itself. Commands travel over the member's inter-node listener, under the same node identity as consensus.

```rust
use coordinode_embed::DatabaseError;
use coordinode_embed::replication::ReplicatedDatabase;
use coordinode_embed::replication::ingress::{Placement, Statement};

async fn through_any_member(db: &ReplicatedDatabase) -> Result<(), DatabaseError> {
    // On any member: runs at the leader if this member does not lead.
    let _answer = db.execute(Statement::new("CREATE (:Person {name: 'Bob'})")).await?;

    // Leader-local: runs only if this member leads, never routed.
    let _refused = db
        .execute(Statement {
            placement: Placement::LeaderLocal,
            ..Statement::new("CREATE (:Lease {holder: 1})")
        })
        .await;
    Ok(())
}
```

- **Where it ran.** Every ingress call answers a `Routed` value: the answer, `forwarded_to` (the member that ran it, when not this one) and `leader` (the member this one knew to lead when the answer came back, `None` when it knew of none). The leader is advisory and costs no extra round trip; a host that cares can move its traffic there.
- **Settings.** `Statement::requested` names the read concern, read preference, write concern, causal position and deadline of the statement; an unnamed one is the executing member's. The same combinations a server refuses are refused here before anything runs (`SettingsRefused`). A read the default preference sends to the leader is answered there; name `ReadPreference::Nearest` to read this member's copy.
- **Transactions.** `db.ingress()` begins, runs, commits and rolls back interactive transactions under the ID the host drew (`coordinode_embed::db::draw_transaction_id`). The transaction's first statement opens it: set its `start` (`TransactionStart`: concerns, timeout outcome, operation) and the leader opens the transaction and runs the statement in one call, with no separate begin before it; `ingress().begin` is for a transaction whose first command is not a statement. Every command of a transaction runs at the leader that holds it, so a transaction begun through one member continues through another. A start naming an ID already open is refused with `TransactionIdInUse`. When the lead moves the transaction ends: the new leader answers `UnknownTransaction`, and the old member leading again does not revive it; run the transaction again under a new ID. A transaction left idle past `set_interactive_idle_timeout` ends at the leader by itself, with the timeout outcome its start named: rolled back (a later command answers `TransactionAborted` with `IdleExpired`), or committed under the operation the start carried, which then resolves to that commit through any member.
- **Operation records.** `ingress().operation` enrolls issuers and resolves, acknowledges and retires operations at the leader, whose log decides them.
- **Leader-local.** `Placement::LeaderLocal` keeps a command on this member: it runs only while this member leads and otherwise fails with `NotLeader` naming the leader. A host whose writes must originate at the leader (a singleton holding a lease there) uses it, so a deposed writer's work cannot reach the new leader by another road.
- **No leader.** While no member is known to lead, a write or a transaction command fails with `NotLeader { leader_id: None }`; nothing waits for the election or applies locally. A member that still names a leader it can no longer reach fails the call with `PeerUnreachable` when it could not send it, and otherwise with `PeerOutcomeUnknown`, since a transport failure cannot prove the command never ran.
- **Failures on the way.** `PeerUnreachable` means the command was not sent (for example a member that has just joined does not know the leader's address yet), so it did not run and can be sent again; the ingress itself sends such a command again to the leader it knows then, up to three times within the statement's deadline (pauses of 25, 50 and 100 ms), before returning it. A member reached by a routed command it cannot run refuses it with `NotLeader` naming the leader it knows, and never passes it on again. `PeerOutcomeUnknown` means it was sent and its answer was lost: a write may have committed, so resolve it by its operation rather than running it again. See [Transactions and errors](#transactions-and-errors).

A command one member of a replicated group passes to another (`replication::routing::CommandRouter`) fails with the error the executing member raised, field for field, as if it had run here; only the engine's cause stays on that member. Two variants belong to the hop: `PeerUnreachable` means nothing was sent, or the member refused the command before reading it, so it did not run; `PeerOutcomeUnknown` means the command was sent and its answer was lost (`PeerFailure::Transport`, `Deadline` or `Protocol`), so it may have run. Neither is a refusal or an abort: resolve a write of unknown outcome by its operation rather than running it again.

Election timing comes in two profiles: `ElectionTiming::RACK` (heartbeat 150 ms, failover within a second) for servers next to each other, and `ElectionTiming::PERSONAL_FLEET` (heartbeat 2 s, election after 20 to 40 s) for members on laptops, phones and home links. Set it in `MemberOptions::timing` and give every member of a group the same one. `MemberOptions::preference = ElectionPreference::Last` makes a member lead only when no other can, and hand leadership back once another member has caught up. The server exposes the same choices as `election_profile` and `election_preference`; see [Configuration](./configuration).

## Testing with the Embedded Engine

Use `tempfile::tempdir()` for isolated per-test databases that clean up automatically on drop:

```rust
use coordinode_embed::Database;

#[test]
fn test_find_related_concepts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher(
        "CREATE (:Concept {name: 'ML'})-[:RELATED_TO]->(:Concept {name: 'AI'})"
    ).expect("create");

    let rows = db
        .execute_cypher("MATCH (a:Concept)-[:RELATED_TO]->(b) RETURN b.name")
        .expect("match");

    assert_eq!(rows.len(), 1);
}
```

Add `tempfile` to `[dev-dependencies]`:

```toml
[dev-dependencies]
tempfile = "3"
```

## Next Step

See [Quick Start](../QUICKSTART) for a full example using the Docker server, or [Data Model](./data-model) to understand the graph + vector model.
