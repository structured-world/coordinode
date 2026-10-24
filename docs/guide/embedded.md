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

- **Compiler and targets.** The packages declare Rust 1.92 as their minimum.
  Every release is installed from the registry and run on Linux with Rust
  1.92.0 and 1.99.0 before it is called ready; the workspace itself builds and
  tests on Linux, macOS and Windows.
- **Features.** The default build is a standalone database (see
  [Build features](#build-features)). `replication` adds membership in a
  replicated group; `io-uring` adds the io_uring storage backend on Linux.
- **What the registry checks run.** A registry release passes when the
  [quickstart example](#quickstart) and a broader consumer (crash recovery,
  operation identity) build from the published packages and run standalone,
  and the same consumer compiles with `replication`. Compiling with a feature
  is not the same as running it: run your own scenario with the features you
  enable on the build you ship.

Other artifacts are separate: the Python package `coordinode-embedded`
(`pip install coordinode-embedded`) embeds the same engine for Python, and
the server binary runs it as a network service. A source build from the
[public repository](https://github.com/structured-world/coordinode) is a
different artifact from the registry packages: pin an exact reviewed
revision, never follow `main`, and qualify it separately. The workspace pins
Rust 1.99.0; a source build with `replication` also uses the pinned proto
submodule and its generation tooling, which the registry packages carry
already generated.

The application owns its `Cargo.lock` and build profile. Check the features
your dependency graph actually selects: another dependency can unify features
the engine itself does not request.

The CE license is AGPL-3.0-only. See the repository's `LICENSE`, `COPYRIGHT`
and `NOTICES`; dependency licenses remain their own.

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
- **Acknowledged writes.** Under the default write concern, a write is
  acknowledged once its journal entry is fsynced: it survives a process crash
  and a power failure. Weaker journal settings trade that for latency; see
  [Durability](./transactions#durability).
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
| Hard limit | `EndpointConfig::with_hard_limit_bytes` (and `with_recovery_reserve_bytes`) per endpoint; live with `Database::set_hard_limit` | Space every file of the database under the endpoint's directory takes, as the file system allocates it | The database's own files never take more than the limit, whatever else is on the disk |

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

Deleting a blob directly removes its metadata but does not reclaim
unreferenced application chunks; do not count it as space released.

## Build features

- **Default (standalone).** No consensus, inter-node transport, listener or
  async runtime is compiled. The `Database` calls are synchronous.
- **`replication`.** A database that is a member of a replicated group, in
  `coordinode_embed::replication`. The host supplies a multi-threaded Tokio
  runtime; the member serves its peers on a listener of its own (consensus,
  snapshot and segment transfer, under mutual TLS node identity), never a
  client protocol. A build with the feature still runs standalone until it is
  given peers.
- **`io-uring`.** The storage engine opens on one shared io_uring ring instead
  of synchronous `pread`/`pwrite`, on Linux 5.6 and later. Elsewhere it is a
  no-op; a kernel without io_uring makes the engine log a warning and use the
  default backend. The registry checks do not run it; it is built and tested
  inside the workspace on Linux.

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
