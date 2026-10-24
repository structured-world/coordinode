//! Runs outside the source workspace against prepared or published packages.
//! This checks the actual installed library, including a process exit without
//! database destruction. It supplements, rather than replaces, the source suite.

use coordinode_core::graph::{node::NodeId, types::Value};
use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::backup::restore::RestoreOptions;
use coordinode_embed::backup::{BackupFormat, export};
use coordinode_embed::db::{StatementOptions, TransactionEnd};
use coordinode_embed::operation::{IssuerId, OperationDescriptor, OperationOutcome, Resolve};
use coordinode_embed::{
    Database, DatabaseError, EdgeTypeDeclaration, Intent, LabelDeclaration, PackageConstraint,
    SchemaPackage,
};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn schema() -> SchemaPackage {
    let mut label = LabelSchema::new_node_id("Inventory");
    label.set_mode(SchemaMode::Strict);
    label.add_property(PropertyDef::new("key", PropertyType::String));
    label.add_property(PropertyDef::new("epoch", PropertyType::Int));
    SchemaPackage {
        labels: vec![LabelDeclaration {
            schema: label,
            intent: Intent::Ensure,
        }],
        edge_types: vec![EdgeTypeDeclaration {
            schema: EdgeTypeSchema::new("DEPENDS_ON"),
            intent: Intent::Ensure,
        }],
        constraints: vec![PackageConstraint {
            name: Some("inventory_key".into()),
            label: "Inventory".into(),
            properties: vec!["key".into()],
            kind: ConstraintKind::Unique,
            scope: None,
            on_duplicate_rename: None,
        }],
    }
}

fn install_schema(db: &Database) -> Result<()> {
    let publication = db.publish_schema(schema())?;
    assert!(
        db.await_schema(&publication, Duration::from_secs(30))?
            .all_ready()
    );
    Ok(())
}

fn basic(path: &Path) -> Result<()> {
    let mut db = Database::open(path)?;
    install_schema(&db)?;

    // Inventory, relation, audit and outbox effects publish together. A separate
    // read must not see buffered effects before their transaction commits.
    let tx = db.begin_transaction()?;
    let rows = db.execute_in_transaction(tx,
        "CREATE (a:Inventory {key:'host', epoch:1})-[:DEPENDS_ON]->(b:Inventory {key:'disk', epoch:1}) RETURN id(a) AS id", None)?;
    let raw = rows[0]
        .get("id")
        .and_then(Value::as_int)
        .ok_or("missing node id")?;
    let node = NodeId::from_raw(u64::try_from(raw)?);
    db.execute_in_transaction(tx, "CREATE (:Audit {action:'created'})", None)?;
    db.execute_in_transaction(tx, "CREATE (:Outbox {action:'provision'})", None)?;
    assert!(
        db.execute_cypher("MATCH (n:Inventory) RETURN n")?
            .is_empty()
    );
    db.commit_transaction(tx)?;
    let links = db.execute_cypher(
        "MATCH (a:Inventory)-[:DEPENDS_ON]->(b:Inventory) RETURN a.key AS source, b.key AS target",
    )?;
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].get("source"), Some(&Value::String("host".into())));
    assert_eq!(links[0].get("target"), Some(&Value::String("disk".into())));

    // A declared unique identity must be enforced by the installed artifact.
    assert!(matches!(
        db.execute_cypher("CREATE (:Inventory {key:'host', epoch:1})"),
        Err(DatabaseError::UniqueViolation { .. })
    ));

    // A caller's stale revision cannot overwrite a newer epoch or publish its
    // audit record. Compare the versions returned by the actual store.
    let old = db.node_version(node)?.ok_or("missing initial version")?;
    db.execute_cypher("MATCH (n:Inventory {key:'host'}) SET n.epoch = 2")?;
    let current = db.node_version(node)?.ok_or("missing new version")?;
    assert_ne!(old, current);
    let stale = db.begin_transaction()?;
    db.execute_in_transaction(
        stale,
        "MATCH (n:Inventory {key:'host'}) SET n.epoch = 3",
        None,
    )?;
    db.execute_in_transaction(stale, "CREATE (:StaleAudit)", None)?;
    db.expect_node_version(stale, node, Some(old))?;
    match db.commit_transaction(stale) {
        Err(DatabaseError::RevisionMismatch {
            expected,
            current: actual,
            ..
        }) => {
            assert_eq!(expected, Some(old));
            assert_eq!(actual, Some(current));
        }
        other => panic!("expected revision refusal, got {other:?}"),
    }
    assert!(
        db.execute_cypher("MATCH (n:StaleAudit) RETURN n")?
            .is_empty()
    );

    let discard = db.begin_transaction()?;
    db.execute_in_transaction(discard, "CREATE (:Discarded)", None)?;
    db.rollback_transaction(discard)?;
    // Rollback retains its explicit terminal reason for this handle.
    let ended = db.commit_transaction(discard);
    assert!(
        matches!(
            ended,
            Err(DatabaseError::TransactionAborted {
                id,
                end: TransactionEnd::RolledBack,
            }) if id == discard
        ),
        "unexpected rollback termination: {ended:?}"
    );
    drop(db);

    let mut db = Database::open(path)?;
    install_schema(&db)?;
    assert_eq!(db.execute_cypher("MATCH (n:Inventory) RETURN n")?.len(), 2);
    assert_eq!(db.execute_cypher("MATCH (n:Audit) RETURN n")?.len(), 1);
    assert_eq!(db.execute_cypher("MATCH (n:Outbox) RETURN n")?.len(), 1);
    assert!(
        db.execute_cypher("MATCH (n:Discarded) RETURN n")?
            .is_empty()
    );
    let rows = db.execute_cypher("MATCH (n:Inventory {key:'host'}) RETURN n.epoch AS epoch")?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("epoch"), Some(&Value::Int(2)));

    // The installed SQL path has the same bounded Rust-only frontend and must
    // remain usable after refusing a statement before any storage effect.
    db.execute_sql("CREATE TABLE SqlProbe (id BIGINT, name VARCHAR(64))")?;
    db.execute_sql("INSERT INTO SqlProbe (id, name) VALUES (1, 'original')")?;
    let excessive = format!(
        "UPDATE SqlProbe SET name = 'changed' WHERE {}",
        vec!["id = 1"; 128].join(" AND ")
    );
    let error = db
        .execute_sql(&excessive)
        .expect_err("refuse excessive SQL structure");
    assert!(error.to_string().contains("SQL structural token limit"));
    assert_eq!(
        db.execute_sql("SELECT name FROM SqlProbe WHERE id = 1")?[0].get("name"),
        Some(&Value::String("original".into()))
    );
    Ok(())
}

// A non-idempotent effect: a retry that ran again would count twice.
const EFFECT: &str =
    "MATCH (c:Counter {key:'jobs'}) SET c.n = c.n + 1 CREATE (:Delivered {key:'once'})";

fn jobs(db: &mut Database) -> Result<i64> {
    let rows = db.execute_cypher("MATCH (c:Counter {key:'jobs'}) RETURN c.n AS n")?;
    rows.first()
        .and_then(|row| row.get("n"))
        .and_then(Value::as_int)
        .ok_or_else(|| "missing job counter".into())
}

fn run_operation(db: &Database, op: OperationDescriptor) -> std::result::Result<(), DatabaseError> {
    db.execute_cypher_shared_with(
        EFFECT,
        None,
        None,
        &StatementOptions {
            operation: Some(op),
            ..Default::default()
        },
    )
    .map(drop)
}

fn exit_after_commit(root: &Path) -> Result<()> {
    let mut db = Database::open(root.join("crash-db"))?;
    db.execute_cypher("CREATE (:Counter {key:'jobs', n:0})")?;
    let issuer = db.enroll_issuer(IssuerId(*b"registry-checker"))?;
    let op = issuer.next()?;
    // The host persists its retry descriptor before sending the operation.
    let mut descriptor = std::fs::File::create(root.join("operation"))?;
    descriptor.write_all(&op.to_bytes())?;
    descriptor.sync_all()?;
    // The reply is lost: the process ends right after the commit, without
    // Drop or any graceful database shutdown.
    run_operation(&db, op)?;
    std::process::exit(0);
}

fn assert_settled(result: std::result::Result<(), DatabaseError>) {
    assert!(
        matches!(
            result,
            Err(DatabaseError::OperationSettled {
                outcome: OperationOutcome::Committed { .. },
                intent_differs: false,
                ..
            })
        ),
        "a retry of a committed operation must run nothing: {result:?}"
    );
}

/// The same history, across a process exit, a checkpoint and the repair that
/// rebuilds from it: the persisted descriptor resolves as committed and its
/// retry counts nothing twice.
fn recovered_operation(root: &Path) -> Result<OperationDescriptor> {
    let status = std::process::Command::new(std::env::current_exe()?)
        .arg(root)
        .arg("exit-after-commit")
        .status()?;
    assert!(
        status.success(),
        "child failed before its acknowledged commit"
    );
    let bytes = std::fs::read(root.join("operation"))?;
    let op = OperationDescriptor::from_bytes(&bytes).ok_or("invalid saved descriptor")?;
    let mut db = Database::open(root.join("crash-db"))?;
    assert!(matches!(
        db.resolve_operation(&op, Resolve::Lookup)?,
        OperationOutcome::Committed { .. }
    ));
    assert_settled(run_operation(&db, op));
    assert_eq!(jobs(&mut db)?, 1);
    assert_eq!(db.execute_cypher("MATCH (n:Delivered) RETURN n")?.len(), 1);

    db.checkpoint(1)?;
    assert!(db.verify_and_repair()?.is_clean());
    drop(db);
    let mut db = Database::open(root.join("crash-db"))?;
    assert_settled(run_operation(&db, op));
    assert_eq!(jobs(&mut db)?, 1, "the retried increment counts once");
    Ok(op)
}

/// A logical backup restored into a new store is a new domain: the data is
/// there, and a descriptor issued before the backup is refused rather than run
/// again or answered with an outcome the restored store never recorded.
fn logical_restore(root: &Path, op: OperationDescriptor) -> Result<()> {
    let source = Database::open(root.join("crash-db"))?;
    let interner = source.interner()?;
    for format in [BackupFormat::Binary, BackupFormat::Json] {
        let mut dump = Vec::new();
        let snapshot = source.engine().snapshot();
        match format {
            BackupFormat::Binary => {
                export::export_binary(source.engine(), &interner, 1, &snapshot, &mut dump)?;
            }
            _ => {
                export::export_json(source.engine(), &interner, 1, &snapshot, &mut dump)?;
            }
        }
        let mut restored = Database::open(root.join(format!("restored-{format:?}")))?;
        restored.restore(format, &dump, &RestoreOptions::default())?;
        assert_eq!(jobs(&mut restored)?, 1, "{format:?}");
        assert!(
            matches!(
                run_operation(&restored, op),
                Err(DatabaseError::ForeignOperation { .. })
            ),
            "{format:?}: an old descriptor must not run in a new domain"
        );
        assert_eq!(jobs(&mut restored)?, 1, "{format:?}");
    }
    Ok(())
}

/// Independent writers racing on one declared identity: exactly one wins and
/// every other is refused typed, whatever their interleaving.
fn concurrent_identity(path: &Path) -> Result<()> {
    let db = Database::open(path)?;
    install_schema(&db)?;
    let db = Arc::new(db);
    let writers: Vec<_> = (0..8)
        .map(|_| {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                db.execute_cypher_shared_with(
                    "CREATE (:Inventory {key:'contended', epoch:1})",
                    None,
                    None,
                    &StatementOptions::default(),
                )
                .map(drop)
            })
        })
        .collect();
    let mut won = 0;
    for writer in writers {
        match writer.join().map_err(|_| "a writer panicked")? {
            Ok(()) => won += 1,
            Err(
                DatabaseError::UniqueViolation { .. } | DatabaseError::TransactionConflict { .. },
            ) => {}
            Err(other) => return Err(format!("unexpected refusal: {other:?}").into()),
        }
    }
    assert_eq!(won, 1, "exactly one writer takes a unique identity");
    let mut db = Arc::try_unwrap(db).map_err(|_| "a writer kept the database")?;
    assert_eq!(
        db.execute_cypher("MATCH (n:Inventory {key:'contended'}) RETURN n")?
            .len(),
        1
    );
    Ok(())
}

/// A worker whose lease was taken over is fenced by the lease's epoch: its
/// commit conditioned on the version it read is refused, and nothing of its
/// work lands.
fn lease_fencing(path: &Path) -> Result<()> {
    let mut db = Database::open(path)?;
    let rows =
        db.execute_cypher("CREATE (l:Lease {name:'sweeper', epoch:1}) RETURN id(l) AS id")?;
    let lease = NodeId::from_raw(u64::try_from(
        rows[0]
            .get("id")
            .and_then(Value::as_int)
            .ok_or("missing lease id")?,
    )?);
    let held = db.node_version(lease)?.ok_or("missing lease version")?;
    // Another worker takes the lease over.
    db.execute_cypher("MATCH (l:Lease {name:'sweeper'}) SET l.epoch = 2")?;
    let stale = db.begin_transaction()?;
    db.execute_in_transaction(stale, "CREATE (:Swept {worker:1})", None)?;
    db.expect_node_version(stale, lease, Some(held))?;
    assert!(matches!(
        db.commit_transaction(stale),
        Err(DatabaseError::RevisionMismatch { .. })
    ));
    assert!(db.execute_cypher("MATCH (n:Swept) RETURN n")?.is_empty());
    Ok(())
}

fn payload(seed: u64, len: usize) -> String {
    use std::fmt::Write as _;
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut out = String::with_capacity(len + 16);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let _ = write!(out, "{state:016x}");
    }
    out
}

/// The owned-footprint limit as an installed consumer meets it: writes are
/// refused typed once the database's files reach it, reads go on, nothing is
/// deleted, and a raised limit admits writes again; the data survives reopen.
fn hard_limit(path: &Path) -> Result<()> {
    const LIMIT: u64 = 16 << 20;
    let config = || -> Result<StorageConfig> {
        let endpoint =
            EndpointConfig::new("data", path, Media::Ssd, Durability::Durable, Tier::Warm)
                .with_hard_limit_bytes(LIMIT);
        Ok(StorageConfig::try_with_endpoints(vec![endpoint])?)
    };
    let mut db = Database::open_with_config(config()?)?;
    let mut accepted = 0i64;
    let mut refused = false;
    for key in 0..4096u64 {
        let statement = format!(
            "CREATE (:Filler {{key:{key}, payload:'{}'}})",
            payload(key, 32 * 1024)
        );
        match db.execute_cypher(&statement) {
            Ok(_) => accepted += 1,
            Err(DatabaseError::CapacityExhausted { .. }) => {
                refused = true;
                break;
            }
            Err(other) => return Err(format!("unexpected refusal: {other:?}").into()),
        }
    }
    assert!(refused, "the limit never refused a write");
    assert!(accepted > 0);
    let count = |db: &mut Database| -> Result<i64> {
        db.execute_cypher("MATCH (n:Filler) RETURN count(n) AS n")?[0]
            .get("n")
            .and_then(Value::as_int)
            .ok_or_else(|| "missing count".into())
    };
    assert_eq!(count(&mut db)?, accepted, "a full database answers reads");

    db.set_hard_limit("data", 2 * LIMIT, None)?;
    // A compaction the full store held back may take the new room for a
    // moment; it gives it back when it ends.
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        match db.execute_cypher("CREATE (:Filler {key:-1})") {
            Ok(_) => break,
            Err(DatabaseError::CapacityExhausted { .. })
                if std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(other) => return Err(format!("a raised limit must admit writes: {other:?}").into()),
        }
    }
    drop(db);
    let mut db = Database::open_with_config(config()?)?;
    assert_eq!(
        count(&mut db)?,
        accepted + 1,
        "every acknowledged write survives reopen"
    );
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let root = std::path::PathBuf::from(args.next().ok_or("missing owned test directory")?);
    if let Some(mode) = args.next() {
        assert_eq!(mode, "exit-after-commit");
        return exit_after_commit(&root);
    }
    basic(&root.join("basic-db"))?;
    concurrent_identity(&root.join("identity-db"))?;
    lease_fencing(&root.join("lease-db"))?;
    let op = recovered_operation(&root)?;
    logical_restore(&root, op)?;
    hard_limit(&root.join("capped-db"))?;
    println!(
        "Installed consumer: schema, graph, unique identity under concurrent writers, atomic \
         audit/outbox, CAS and lease fencing, rollback, reopen, an increment retried after a \
         process exit and across checkpoint and repair, logical restore into a new domain, \
         and the owned-footprint limit passed"
    );
    Ok(())
}
