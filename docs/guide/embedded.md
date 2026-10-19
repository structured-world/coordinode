---
description: "Run the full CoordiNode engine inside your Rust process with coordinode-embed: graph, vector and full-text queries with no server, no network hop and no external dependency."
---

# Embedded (Rust)

`coordinode-embed` lets you run CoordiNode in-process — no separate server, no network hop, no external dependency. Ideal for desktop applications, test harnesses, and edge deployments.

## Add the Dependency

```toml
[dependencies]
coordinode-embed = "0.3"
```

## Minimal Example

```rust
use coordinode_embed::Database;

fn main() -> anyhow::Result<()> {
    let mut db = Database::open("/tmp/my-graph")?;

    // Execute Cypher directly — no gRPC
    db.execute_cypher("CREATE (n:Person {name: 'Alice'}) RETURN n")?;

    let rows = db.execute_cypher("MATCH (n:Person) RETURN n.name")?;
    println!("{rows:?}");
    Ok(())
}
```

## API Reference

```rust
use coordinode_embed::{Database, DatabaseError};
use coordinode_core::graph::types::Value;
use std::collections::HashMap;

// Open (creates directory if absent)
let mut db = Database::open("/path/to/data")?;

// Execute Cypher — no parameters
let rows = db.execute_cypher("MATCH (n:Person) RETURN n.name")?;

// Execute with parameters
let mut params = HashMap::new();
params.insert("name".to_string(), Value::String("Alice".into()));
let rows = db.execute_cypher_with_params(
    "MATCH (n:Person {name: $name}) RETURN n",
    params,
)?;

// Interactive multi-statement transaction: every statement reads the
// snapshot pinned at begin; commit applies all writes atomically and
// returns a CommitReceipt (commit_ts = the seqno every write landed at:
// the `AS OF TIMESTAMP` anchor, and a change scan from it includes this
// commit; applied_index = None in embedded mode).
let tx = db.begin_transaction();
db.execute_in_transaction(tx, "CREATE (n:Person {name: 'Bob'})", None)?;
let receipt = db.commit_transaction(tx)?;
let as_of = receipt.commit_ts.as_raw();

// Time travel is bounded by the MVCC retention window (default five minutes,
// `StorageConfig::retention_window_secs` via `Database::open_with_config`).
// Tune it at runtime; a read older than `oldest_readable_timestamp()` fails
// with `OutsideRetention` instead of answering from collected history.
db.set_retention_window(std::time::Duration::from_secs(24 * 3600));
let horizon = db.oldest_readable_timestamp();

// A write that takes effect at most once however it is retried: number it
// with an enrolled issuer, persist the descriptor (`to_bytes`) before running
// it, and after a lost reply or a restart resolve the same descriptor.
use coordinode_embed::db::StatementOptions;
use coordinode_embed::operation::{IssuerId, Resolve};
let issuer = db.enroll_issuer(IssuerId(*b"payments-worker1"))?;
let op = issuer.next()?;
db.execute_cypher_shared_with(
    "CREATE (:Payment {amount: 10})",
    None,
    None,
    &StatementOptions { operation: Some(op), ..Default::default() },
)?;
let outcome = db.resolve_operation(&op, Resolve::Lookup)?; // Committed { .. }
db.acknowledge_operation(&op)?; // release the outcome once it is no longer needed
```

See [Transactions](./transactions) for the full interactive-transaction contract, safe retries with operation identity, and what the retention window costs in storage.

## Declaring the Schema

An application declares its labels, edge types and constraints as one package, typically at every open. `Database::publish_schema` checks the whole package against one view of the catalog, with the package itself visible to it (a constraint may name a label the same package defines, in any order), and publishes every change in one catalog commit, or nothing.

```rust
use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::{
    EdgeTypeDeclaration, Intent, LabelDeclaration, PackageConstraint, SchemaPackage,
};

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
assert!(readiness.all_ready());
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
- **History.** A store written without replication becomes the member a group is formed around, in place: no export, no import. Members that join start empty and receive everything from the leader. A member that already holds data of its own is refused as a joiner. Growth goes one way: once a store is a member, its history is the group's log, and `Database::open` (with or without the feature) refuses it with `StorageError::GroupMemberStore`, writing nothing; it opens only as a member, through `ReplicatedDatabase`.

```toml
[dependencies]
coordinode-embed = { version = "0.7", features = ["replication"] }
```

```rust
use coordinode_embed::replication::{
    Founding, MemberOptions, PeerTimeouts, Peering, ReplicatedDatabase,
};

// One member, no peers: leads alone, opens no listener.
let db = ReplicatedDatabase::open(&storage, MemberOptions::new(1), None).await?;
db.database().write().execute_cypher("CREATE (:Person {name: 'Alice'})")?;

// A member that forms a group with peers: it serves them on `listen`, and
// its node identity (mutual TLS) comes from `security`.
let mut options = MemberOptions::new(1);
options.peering = Some(Peering {
    security,                            // this member's certificate and the group's CA
    advertise: "10.0.0.1:7080".into(),   // the address peers dial
    founding: Founding::Form { deployment: None },
    timeouts: PeerTimeouts::default(),
});
let leader = ReplicatedDatabase::open(&storage, options, Some(listen)).await?;

// A new member joins with the enrollment the leader hands out, then the
// leader adds it as a learner and promotes it to a voter once it caught up.
let enrollment = leader.enrollment("10.0.0.1:7080").expect("enrollment");
// ... on member 2: founding = Founding::Join(enrollment) ...
leader.add_learner(2, "10.0.0.2:7080", identity_of_2).await?;
leader.promote(2).await?;
```

An already open `Database` becomes a member on the same store with `ReplicatedDatabase::from_database`: what it committed is all there under the replicated handle.

Behaviour every host can rely on:

- Only the leader applies writes. A write on another member, whatever its write concern, is refused with `NotLeader` naming the leader when it is known, and lands on no member. A restore is refused the same way: it runs on the leader, and its records, edges, schemas and the indexes built over them reach every member.
- `leadership()` streams leadership changes; `transfer_leadership(id)` hands leadership to a named member (for example before a laptop sleeps) and returns once that member leads.
- A member cut off from its group asks the others whether they would vote before it starts an election, so its return does not depose a leader the rest still follow.
- `shutdown()` stops the member, its listener and its background work, and closes the store before it returns, so the same directory can be opened again at once. Dropping the handle stops the background work too, without waiting for it.

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
