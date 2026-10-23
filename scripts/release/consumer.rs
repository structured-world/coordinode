//! Runs outside the source workspace against prepared or published packages.
//! This checks the actual installed library, including a process exit without
//! database destruction. It supplements, rather than replaces, the source suite.

use coordinode_core::graph::{node::NodeId, types::Value};
use coordinode_core::schema::definition::{
    ConstraintKind, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType, SchemaMode,
};
use coordinode_embed::db::{StatementOptions, TransactionEnd};
use coordinode_embed::operation::{IssuerId, OperationDescriptor, OperationOutcome, Resolve};
use coordinode_embed::{
    Database, DatabaseError, EdgeTypeDeclaration, Intent, LabelDeclaration, PackageConstraint,
    SchemaPackage,
};
use std::io::Write;
use std::path::Path;
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

const EFFECT: &str = "CREATE (:Delivered {key:'once'})";

fn exit_after_commit(root: &Path) -> Result<()> {
    let db = Database::open(root.join("crash-db"))?;
    let issuer = db.enroll_issuer(IssuerId(*b"registry-checker"))?;
    let op = issuer.next()?;
    // The host persists its retry descriptor before sending the operation.
    let mut descriptor = std::fs::File::create(root.join("operation"))?;
    descriptor.write_all(&op.to_bytes())?;
    descriptor.sync_all()?;
    db.execute_cypher_shared_with(
        EFFECT,
        None,
        None,
        &StatementOptions {
            operation: Some(op),
            ..Default::default()
        },
    )?;
    // Deliberately bypass Drop and any graceful database shutdown.
    std::process::exit(0);
}

fn recovered_operation(root: &Path) -> Result<()> {
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
    assert!(matches!(
        db.execute_cypher_shared_with(
            EFFECT,
            None,
            None,
            &StatementOptions {
                operation: Some(op),
                ..Default::default()
            }
        ),
        Err(DatabaseError::OperationSettled {
            outcome: OperationOutcome::Committed { .. },
            intent_differs: false,
            ..
        })
    ));
    assert_eq!(db.execute_cypher("MATCH (n:Delivered) RETURN n")?.len(), 1);
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
    recovered_operation(&root)?;
    println!(
        "Installed consumer: schema, graph, unique identity, atomic audit/outbox, CAS, rollback, reopen and durable retry passed"
    );
    Ok(())
}
