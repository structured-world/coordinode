//! Node constraint DDL end to end: `CREATE CONSTRAINT` / `DROP CONSTRAINT`
//! through Cypher, the writes they refuse on every path, their activation
//! against stored data and against writers in flight, and their survival
//! across a restart.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use coordinode_core::graph::types::Value;
use coordinode_core::schema::definition::{ConstraintKind, ConstraintState, PropertyType};
use coordinode_core::txn::proposal::{
    Mutation, ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_embed::Database;
use coordinode_embed::db::DatabaseError;
use coordinode_modality::{LocalSchemaStore, SchemaStore as _};
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::{CatalogObject, ExecutionError};

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// Whether `err` refuses a change because a catalog object of kind `object`
/// named `name` already exists.
fn names_existing(err: &DatabaseError, object: CatalogObject, name: &str) -> bool {
    matches!(
        err,
        DatabaseError::Execution(ExecutionError::CatalogObjectExists { object: o, name: n })
            if *o == object && n == name
    )
}

/// The state of constraint `name` on `label` as stored.
fn stored_state(db: &Database, label: &str, name: &str) -> Option<ConstraintState> {
    LocalSchemaStore::new(db.engine())
        .load_label(label)
        .expect("load schema")
        .and_then(|s| s.constraint(name).map(|c| c.state))
}

/// A pipeline that, while `refuse` is set, refuses every proposal deleting
/// an index definition and applies everything else: a catalog commit that
/// fails after the statement decided to make it.
struct RefuseDefinitionDeletes {
    inner: coordinode_raft::proposal::OwnedLocalProposalPipeline,
    refuse: Arc<AtomicBool>,
}

impl ProposalPipeline for RefuseDefinitionDeletes {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        let deletes_definition = proposal
            .mutations
            .iter()
            .any(|m| matches!(m, Mutation::Delete { key, .. } if key.starts_with(b"schema:idx:")));
        if deletes_definition && self.refuse.load(Ordering::SeqCst) {
            return Err(ProposalError::Storage("definition delete refused".into()));
        }
        self.inner.propose_and_wait(proposal)
    }
}

/// A database whose index-definition deletes are refused while the returned
/// flag is set.
fn open_db_refusing_definition_deletes() -> (Database, Arc<AtomicBool>, tempfile::TempDir) {
    use coordinode_core::txn::timestamp::TimestampOracle;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine =
        Arc::new(StorageEngine::open_with_oracle(&config, oracle.clone()).expect("engine"));
    let refuse = Arc::new(AtomicBool::new(false));
    let pipeline: Arc<dyn ProposalPipeline> = Arc::new(RefuseDefinitionDeletes {
        inner: coordinode_raft::proposal::OwnedLocalProposalPipeline::new(&engine),
        refuse: Arc::clone(&refuse),
    });
    let db = Database::from_engine(dir.path(), engine, oracle, pipeline).expect("open db");
    (db, refuse, dir)
}

fn count(db: &mut Database, query: &str) -> i64 {
    let rows = db.execute_cypher(query).expect("count query");
    match rows[0].get("c") {
        Some(Value::Int(n)) => *n,
        other => panic!("expected an integer count, got {other:?}"),
    }
}

/// The constraint violation `result` must be, as (constraint, kind, property).
fn expect_violation(result: Result<Vec<Row>, DatabaseError>) -> (String, ConstraintKind, String) {
    match result {
        Err(DatabaseError::Execution(ExecutionError::ConstraintViolation {
            constraint,
            kind,
            property,
            ..
        })) => (constraint, *kind, property),
        other => panic!("expected a constraint violation, got {other:?}"),
    }
}

fn expect_unique_violation(result: Result<Vec<Row>, DatabaseError>) -> String {
    match result {
        Err(DatabaseError::Execution(ExecutionError::UniqueViolation { index, .. })) => index,
        other => panic!("expected a unique violation, got {other:?}"),
    }
}

// ── NOT NULL ─────────────────────────────────────────────────────────

/// A node created without a required property is refused and nothing of the
/// statement is stored.
#[test]
fn not_null_refuses_a_create_without_the_property() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("create constraint");

    let (constraint, kind, property) =
        expect_violation(db.execute_cypher("CREATE (u:User {name: 'a'})"));
    assert_eq!(constraint, "user_email");
    assert_eq!(kind, ConstraintKind::NotNull);
    assert_eq!(property, "email");
    // An explicit null is as missing as an absent property.
    expect_violation(db.execute_cypher("CREATE (u:User {name: 'b', email: null})"));
    assert_eq!(count(&mut db, "MATCH (u:User) RETURN count(u) AS c"), 0);

    db.execute_cypher("CREATE (u:User {name: 'c', email: 'c@x'})")
        .expect("a node with the property is accepted");
    assert_eq!(count(&mut db, "MATCH (u:User) RETURN count(u) AS c"), 1);
}

/// SET to null and REMOVE of a required property are refused, and the stored
/// value stays.
#[test]
fn not_null_refuses_set_null_and_remove() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("create constraint");
    db.execute_cypher("CREATE (u:User {name: 'a', email: 'a@x'})")
        .expect("create");

    expect_violation(db.execute_cypher("MATCH (u:User) SET u.email = null"));
    expect_violation(db.execute_cypher("MATCH (u:User) REMOVE u.email"));
    expect_violation(db.execute_cypher("MATCH (u:User) SET u = {name: 'only'}"));
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:User) WHERE u.email = 'a@x' RETURN count(u) AS c"
        ),
        1
    );
    // Writes that keep the property are untouched by the constraint.
    db.execute_cypher("MATCH (u:User) SET u.email = 'b@x', u.age = 3")
        .expect("a write keeping the property");
}

/// A constraint judges the node as the transaction leaves it, not each step
/// of building it: a node created bare and filled in by the same statement,
/// or by a later statement of the same transaction, is accepted, and one
/// left without the property at commit is refused.
#[test]
fn a_node_completed_before_commit_satisfies_the_constraint() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("create constraint");

    db.execute_cypher("CREATE (u:User) SET u.email = 'a@x'")
        .expect("created and completed in one statement");
    db.execute_cypher("UNWIND [{email: 'b@x'}, {email: 'c@x'}] AS r CREATE (u:User) SET u = r")
        .expect("the batch-create shape");

    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (u:User {name: 'd'})", None)
        .expect("incomplete inside the transaction");
    db.execute_in_transaction(txn, "MATCH (u:User {name: 'd'}) SET u.email = 'd@x'", None)
        .expect("completed by a later statement");
    db.commit_transaction(txn).expect("complete at commit");

    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (u:User {name: 'e'})", None)
        .expect("incomplete inside the transaction");
    let refused = db
        .commit_transaction(txn)
        .expect_err("still incomplete at commit");
    assert!(refused.to_string().contains("violated"), "{refused}");

    expect_violation(db.execute_cypher("CREATE (u:User) SET u.name = 'f'"));
    assert_eq!(count(&mut db, "MATCH (u:User) RETURN count(u) AS c"), 4);
}

/// MERGE creating a node is a write like any other.
#[test]
fn not_null_applies_to_merge() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("create constraint");
    expect_violation(db.execute_cypher("MERGE (u:User {name: 'm'})"));
    db.execute_cypher("MERGE (u:User {name: 'm', email: 'm@x'})")
        .expect("a merge carrying the property");
    assert_eq!(count(&mut db, "MATCH (u:User) RETURN count(u) AS c"), 1);
}

/// A constraint holds in every schema mode, on properties the schema never
/// declared: a VALIDATED label keeps them in its overflow map, a label
/// without a schema is FLEXIBLE.
#[test]
fn not_null_holds_on_undeclared_properties_in_every_mode() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("ALTER LABEL Device SET SCHEMA VALIDATED")
        .expect("validated");
    db.execute_cypher("CREATE CONSTRAINT FOR (d:Device) REQUIRE d.serial IS NOT NULL")
        .expect("constraint on a validated label");
    expect_violation(db.execute_cypher("CREATE (d:Device {model: 'x'})"));
    db.execute_cypher("CREATE (d:Device {model: 'x', serial: 's1'})")
        .expect("an overflow property satisfies it");

    db.execute_cypher("CREATE CONSTRAINT FOR (l:Log) REQUIRE l.at IS NOT NULL")
        .expect("constraint on a label without a schema");
    expect_violation(db.execute_cypher("CREATE (l:Log {msg: 'x'})"));
    // The schema the constraint created keeps the label flexible: any other
    // property is still accepted.
    db.execute_cypher("CREATE (l:Log {msg: 'x', at: 1, anything: true})")
        .expect("flexible label");
}

/// Enabling a constraint that a stored node already breaks fails, names the
/// node, and leaves no constraint behind.
#[test]
fn creation_is_refused_when_a_stored_node_breaks_it() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (u:User {name: 'old'})")
        .expect("create");
    let err = db
        .execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect_err("a stored node lacks the property");
    let message = err.to_string();
    assert!(message.contains("user_email"), "{message}");
    assert!(message.contains("breaks it"), "{message}");

    // Nothing was enabled: writes without the property still go through and
    // the name is free.
    db.execute_cypher("CREATE (u:User {name: 'new'})")
        .expect("no constraint in force");
    let rows = db
        .execute_cypher("DROP CONSTRAINT user_email IF EXISTS")
        .expect("drop if exists");
    assert_eq!(rows[0].get("dropped"), Some(&Value::Bool(false)));
}

// ── Type ─────────────────────────────────────────────────────────────

/// A type constraint refuses a value of another type and does not require
/// the property.
#[test]
fn type_constraint_refuses_values_of_another_type() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT item_qty FOR (i:Item) REQUIRE i.qty IS :: INTEGER")
        .expect("create constraint");

    let (constraint, kind, property) =
        expect_violation(db.execute_cypher("CREATE (i:Item {qty: 'many'})"));
    assert_eq!(constraint, "item_qty");
    assert_eq!(kind, ConstraintKind::Type(PropertyType::Int));
    assert_eq!(property, "qty");
    expect_violation(db.execute_cypher("CREATE (i:Item {qty: 1.5})"));

    db.execute_cypher("CREATE (i:Item {qty: 3})")
        .expect("an integer");
    db.execute_cypher("CREATE (i:Item {name: 'no qty'})")
        .expect("a type constraint does not require the property");
    expect_violation(db.execute_cypher("MATCH (i:Item {qty: 3}) SET i.qty = 'three'"));
    assert_eq!(count(&mut db, "MATCH (i:Item) RETURN count(i) AS c"), 2);
}

// ── UNIQUE / NODE KEY ────────────────────────────────────────────────

/// A uniqueness constraint refuses a second holder of a value through the
/// index it owns, and leaves nodes without the value unconstrained.
#[test]
fn unique_constraint_refuses_a_second_holder() {
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("create constraint");
    assert_eq!(rows[0].get("created"), Some(&Value::Bool(true)));

    db.execute_cypher("CREATE (u:User {email: 'a@x'})")
        .expect("first holder");
    let index = expect_unique_violation(db.execute_cypher("CREATE (u:User {email: 'a@x'})"));
    assert_eq!(index, "user_email");
    db.execute_cypher("CREATE (u:User {name: 'no email'})")
        .expect("missing value");
    db.execute_cypher("CREATE (u:User {name: 'no email either'})")
        .expect("two nodes without the value do not collide");
    assert_eq!(count(&mut db, "MATCH (u:User) RETURN count(u) AS c"), 3);
}

/// Stored duplicates refuse the constraint, and neither it nor its index is
/// left behind.
#[test]
fn unique_creation_is_refused_on_stored_duplicates() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:User {email: 'a@x'})")
        .expect("duplicates");
    expect_unique_violation(
        db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE"),
    );
    // The name is free for an index and for a constraint.
    db.execute_cypher("CREATE INDEX user_email ON :User(email)")
        .expect("no index of that name remains");
    db.execute_cypher("DROP INDEX user_email").expect("drop");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("no constraint of that name remains");
}

/// A node key requires every property and refuses a second holder of the
/// combination; one shared column alone is fine.
#[test]
fn node_key_requires_every_property_and_a_distinct_combination() {
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE (p.first, p.last) IS NODE KEY",
    )
    .expect("create constraint");

    let (constraint, kind, property) =
        expect_violation(db.execute_cypher("CREATE (p:Person {first: 'Ada'})"));
    assert_eq!(constraint, "person_key");
    assert_eq!(kind, ConstraintKind::NodeKey);
    assert_eq!(property, "last");

    db.execute_cypher("CREATE (p:Person {first: 'Ada', last: 'Lovelace'})")
        .expect("first key");
    db.execute_cypher("CREATE (p:Person {first: 'Ada', last: 'Byron'})")
        .expect("one shared column is a different key");
    let index = expect_unique_violation(
        db.execute_cypher("CREATE (p:Person {first: 'Ada', last: 'Lovelace'})"),
    );
    assert_eq!(index, "person_key");
    expect_violation(db.execute_cypher("MATCH (p:Person {last: 'Byron'}) REMOVE p.first"));
    assert_eq!(count(&mut db, "MATCH (p:Person) RETURN count(p) AS c"), 2);
}

// ── DROP / names ─────────────────────────────────────────────────────

/// Dropping a constraint lifts it and removes the index it owned.
#[test]
fn drop_constraint_lifts_it_and_its_index() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("unique");
    db.execute_cypher("CREATE CONSTRAINT user_name FOR (u:User) REQUIRE u.name IS NOT NULL")
        .expect("not null");
    db.execute_cypher("CREATE (u:User {name: 'a', email: 'a@x'})")
        .expect("create");

    let rows = db
        .execute_cypher("DROP CONSTRAINT user_email")
        .expect("drop unique");
    assert_eq!(rows[0].get("dropped"), Some(&Value::Bool(true)));
    db.execute_cypher("CREATE (u:User {name: 'b', email: 'a@x'})")
        .expect("uniqueness lifted");
    db.execute_cypher("CREATE INDEX user_email ON :User(email)")
        .expect("the owned index went with the constraint");

    db.execute_cypher("DROP CONSTRAINT user_name")
        .expect("drop not null");
    db.execute_cypher("CREATE (u:User {email: 'c@x'})")
        .expect("presence lifted");

    let err = db
        .execute_cypher("DROP CONSTRAINT user_name")
        .expect_err("already dropped");
    assert!(err.to_string().contains("not found"), "{err}");
}

/// Creating and dropping a uniqueness constraint leaves the type definition
/// of the property it covered as it was: its type, requiredness and default
/// survive both, and only the constraint and its index come and go.
#[test]
fn dropping_a_uniqueness_keeps_the_property_definition() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef};
    let (mut db, _dir) = open_db();
    let mut user = LabelSchema::new_node_id("User");
    let email = PropertyDef::new("email", PropertyType::String)
        .not_null()
        .with_default(Value::String("none@x".into()));
    user.add_property(email.clone());
    db.create_label_schema(user).expect("type");
    let defined = |db: &Database| {
        LocalSchemaStore::new(db.engine())
            .load_label("User")
            .expect("load schema")
            .expect("the type stays")
            .get_property("email")
            .cloned()
    };

    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("unique");
    assert_eq!(defined(&db).as_ref(), Some(&email), "creation keeps it");
    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect("drop");
    assert_eq!(defined(&db).as_ref(), Some(&email), "drop keeps it");
    assert!(indexes_on(&db, "User").is_empty());
}

/// A uniqueness constraint dropped and created again under the same name is
/// judged by the data as it is at the second creation: a duplicate written
/// in between refuses it and leaves the name free, and once it is gone the
/// new constraint indexes the current values only, with nothing left over
/// from the values its predecessor indexed.
#[test]
fn a_constraint_recreated_under_its_name_follows_the_current_data() {
    use super::helpers::index_named;
    let (mut db, _dir) = open_db();
    let create = "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE";
    db.execute_cypher(create).expect("create");
    db.execute_cypher("CREATE (:User {name: 'a', email: 'old@x'})")
        .expect("indexed under the first constraint");

    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect("drop");
    db.execute_cypher("MATCH (u:User {name: 'a'}) SET u.email = 'new@x'")
        .expect("change the indexed value while unconstrained");
    db.execute_cypher("CREATE (:User {name: 'b', email: 'new@x'})")
        .expect("a duplicate while unconstrained");

    let err = db
        .execute_cypher(create)
        .expect_err("the duplicate refuses it");
    assert!(
        err.to_string().contains("unique constraint violated"),
        "{err}"
    );
    assert_eq!(stored_state(&db, "User", "user_email"), None);
    assert!(
        index_named(db.engine(), "user_email").is_none(),
        "the refused creation leaves no index behind"
    );

    db.execute_cypher("MATCH (u:User {name: 'b'}) DELETE u")
        .expect("remove the duplicate");
    db.execute_cypher(create).expect("the name is free again");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );

    expect_unique_violation(db.execute_cypher("CREATE (:User {name: 'c', email: 'new@x'})"));
    db.execute_cypher("CREATE (:User {name: 'd', email: 'old@x'})")
        .expect("the value the first constraint indexed is not held by anyone");
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:User {email: 'new@x'}) RETURN count(u) AS c"
        ),
        1
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:User {email: 'old@x'}) RETURN count(u) AS c"
        ),
        1
    );
}

/// Two statements creating a constraint under the same name at the same
/// time, on different labels: exactly one wins, the other is refused as a
/// name clash or a concurrent change, and only the winner's label holds the
/// constraint. Repeated so that the two commits actually overlap.
#[test]
fn concurrent_creates_under_one_name_have_exactly_one_winner() {
    let (db, _dir) = open_db();
    for round in 0..32 {
        let name = format!("required_{round}");
        let statements = [
            format!("CREATE CONSTRAINT {name} FOR (n:Left) REQUIRE n.p{round} IS NOT NULL"),
            format!("CREATE CONSTRAINT {name} FOR (n:Right) REQUIRE n.p{round} IS NOT NULL"),
        ];
        let start = std::sync::Barrier::new(2);
        let outcomes: Vec<Result<(), DatabaseError>> = std::thread::scope(|s| {
            let handles: Vec<_> = statements
                .iter()
                .map(|statement| {
                    let (db, start) = (&db, &start);
                    s.spawn(move || {
                        start.wait();
                        db.execute_cypher_shared(statement, None, None, None, None)
                            .map(|_| ())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("the statement thread"))
                .collect()
        });

        let winners: Vec<&str> = outcomes
            .iter()
            .zip(["Left", "Right"])
            .filter(|(outcome, _)| outcome.is_ok())
            .map(|(_, label)| label)
            .collect();
        assert_eq!(winners.len(), 1, "round {round}: {outcomes:?}");
        for outcome in &outcomes {
            if let Err(err) = outcome {
                assert!(
                    names_existing(err, CatalogObject::Constraint, &name)
                        || matches!(err, DatabaseError::Execution(ExecutionError::Conflict(_))),
                    "round {round}: the loser is refused for the clash, got {err:?}"
                );
            }
        }
        for label in ["Left", "Right"] {
            let expected = (label == winners[0]).then_some(ConstraintState::Active);
            assert_eq!(
                stored_state(&db, label, &name),
                expected,
                "round {round}, label {label}"
            );
        }
    }
}

/// The index a constraint owns can only go with the constraint.
#[test]
fn drop_index_refuses_an_index_a_constraint_owns() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("create constraint");
    let err = db
        .execute_cypher("DROP INDEX user_email")
        .expect_err("owned index");
    assert!(err.to_string().contains("belongs to constraint"), "{err}");
    db.execute_cypher("CREATE (u:User {email: 'a@x'})")
        .expect("create");
    expect_unique_violation(db.execute_cypher("CREATE (u:User {email: 'a@x'})"));
}

/// Names are unique across labels and shared with indexes; IF NOT EXISTS
/// turns a repeat into a no-op, and an unnamed constraint gets a name from
/// what it constrains.
#[test]
fn names_and_if_not_exists() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT c1 FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("create");

    let err = db
        .execute_cypher("CREATE CONSTRAINT c1 FOR (o:Org) REQUIRE o.name IS NOT NULL")
        .expect_err("name taken on another label");
    assert!(
        names_existing(&err, CatalogObject::Constraint, "c1"),
        "{err}"
    );
    let rows = db
        .execute_cypher("CREATE CONSTRAINT c1 IF NOT EXISTS FOR (o:Org) REQUIRE o.name IS NOT NULL")
        .expect("if not exists");
    assert_eq!(rows[0].get("created"), Some(&Value::Bool(false)));

    let err = db
        .execute_cypher("CREATE CONSTRAINT c2 FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect_err("equivalent constraint");
    // The equivalent constraint already in place is the one named.
    assert!(
        names_existing(&err, CatalogObject::Constraint, "c1"),
        "{err}"
    );
    let rows = db
        .execute_cypher("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("equivalent with if not exists");
    assert_eq!(rows[0].get("constraint"), Some(&Value::String("c1".into())));

    let err = db
        .execute_cypher("CREATE INDEX c1 ON :User(name)")
        .expect_err("indexes and constraints share names");
    assert!(
        names_existing(&err, CatalogObject::Constraint, "c1"),
        "{err}"
    );

    let rows = db
        .execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.age IS :: INTEGER")
        .expect("unnamed");
    assert_eq!(
        rows[0].get("constraint"),
        Some(&Value::String("User_age_type".into()))
    );
    db.execute_cypher("DROP CONSTRAINT User_age_type")
        .expect("drop by the derived name");
    let rows = db
        .execute_cypher("DROP CONSTRAINT missing IF EXISTS")
        .expect("if exists");
    assert_eq!(rows[0].get("dropped"), Some(&Value::Bool(false)));
}

/// A COLUMNAR table's rows bypass the transaction a constraint binds, so it
/// cannot take one.
#[test]
fn columnar_tables_refuse_constraints() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE TABLE Trade (id BIGINT PRIMARY KEY, sym STRING) STORAGE COLUMNAR")
        .expect("columnar table");
    let err = db
        .execute_cypher("CREATE CONSTRAINT FOR (t:Trade) REQUIRE t.sym IS NOT NULL")
        .expect_err("columnar");
    assert!(err.to_string().contains("COLUMNAR"), "{err}");
}

// ── Temporal, restart, concurrency ───────────────────────────────────

/// Every version of a temporal node is held to the constraint.
#[test]
fn constraints_hold_on_temporal_versions() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE NODE TYPE Event TEMPORAL")
        .expect("temporal label");
    db.execute_cypher("ALTER LABEL Event SET SCHEMA FLEXIBLE")
        .expect("flexible");
    db.execute_cypher("CREATE CONSTRAINT FOR (e:Event) REQUIRE e.kind IS NOT NULL")
        .expect("create constraint");
    expect_violation(db.execute_cypher("CREATE (e:Event {valid_from: 1})"));
    db.execute_cypher("CREATE (e:Event {valid_from: 1, kind: 'start'})")
        .expect("a version carrying the property");
    expect_violation(db.execute_cypher("MATCH (e:Event) SET e.kind = null"));
}

/// A constraint is part of the stored schema: it is in force after a
/// restart.
#[test]
fn constraints_survive_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.email IS NOT NULL")
            .expect("not null");
        db.execute_cypher("CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE")
            .expect("unique");
        db.execute_cypher("CREATE (u:User {email: 'a@x'})")
            .expect("create");
    }
    let mut db = Database::open(dir.path()).expect("reopen");
    expect_violation(db.execute_cypher("CREATE (u:User {name: 'x'})"));
    expect_unique_violation(db.execute_cypher("CREATE (u:User {email: 'a@x'})"));
}

/// An index definition this build cannot read refuses the open, naming the
/// record. The open used to skip it with a log line and serve the database
/// without that index: a directory written by a development build with the
/// earlier name-keyed catalog lost its unique indexes that way, and a
/// duplicate the constraint forbids was then accepted.
#[test]
fn an_unreadable_index_definition_refuses_the_open() {
    use coordinode_storage::engine::partition::Partition;
    use coordinode_storage::error::StorageError;

    let dir = tempfile::tempdir().expect("tempdir");
    let planted = {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE CONSTRAINT user_key FOR (u:User) REQUIRE u.email IS UNIQUE")
            .expect("unique");
        db.execute_cypher("CREATE (u:User {email: 'a@x'})")
            .expect("create");
        // The definition record in the shape the earlier catalog stored it:
        // a MessagePack array opening with the index name.
        let old_shape =
            rmp_serde::to_vec(&("user_key", "User", vec!["email"], true)).expect("encode");
        let key = db
            .engine()
            .prefix_scan(Partition::Schema, b"schema:idx:")
            .expect("scan")
            .next()
            .expect("the constraint's index definition")
            .into_inner()
            .expect("read")
            .0
            .to_vec();
        db.engine()
            .put(Partition::Schema, &key, &old_shape)
            .expect("plant");
        db.persist().expect("persist");
        key
    };

    match Database::open(dir.path()) {
        Err(DatabaseError::Storage(StorageError::UnreadableCatalog { kind, key, .. })) => {
            assert_eq!(kind, "index definition");
            assert_eq!(key, coordinode_storage::error::printable_key(&planted));
        }
        Err(other) => panic!("refused for another reason: {other}"),
        Ok(_) => panic!("opened without the unique index it cannot read"),
    }
}

/// A writer that validated under the schema before the constraint and
/// commits after it cannot slip a breaking node in: the constraint, whose
/// scan could not see the uncommitted node, is enabled, and the writer's
/// commit is refused because the schema it validated under is gone.
#[test]
fn a_writer_in_flight_cannot_slip_past_activation() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (u:User {email: 'seed@x'})")
        .expect("seed");
    let txn = db.begin_transaction();
    db.execute_in_transaction(txn, "CREATE (u:User {name: 'late'})", None)
        .expect("a write staged before the constraint");

    db.execute_cypher("CREATE CONSTRAINT FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("the stored nodes satisfy it");
    let refused = db
        .commit_transaction(txn)
        .expect_err("the write validated under the old schema");
    assert!(
        matches!(refused, DatabaseError::TransactionConflict { .. })
            || refused.to_string().contains("no longer holds"),
        "{refused}"
    );
    assert_eq!(
        count(
            &mut db,
            "MATCH (u:User) WHERE u.email IS NULL RETURN count(u) AS c"
        ),
        0
    );
    expect_violation(db.execute_cypher("CREATE (u:User {name: 'after'})"));
}

// ── Validation lifecycle ─────────────────────────────────────────────

/// An interrupted validation (a crash after the constraint was published and
/// before its backfill finished) is finished when the database opens again;
/// stored data that breaks it withdraws the constraint with its index and
/// name in one commit, since the interrupted statement never acknowledged
/// it, and never leaves it reading as active.
#[test]
fn an_interrupted_validation_over_duplicates_is_withdrawn_on_open() {
    use coordinode_core::schema::definition::{
        LabelSchema, NodeConstraint, SchemaMode, encode_constraint_name_key,
    };
    use coordinode_query::index::IndexDescriptor;
    use coordinode_storage::engine::partition::Partition;

    use super::helpers::{admit_index, index_named};

    let dir = tempfile::tempdir().expect("tempdir");
    {
        // What the first commit of CREATE CONSTRAINT leaves over data that
        // breaks it: the constraint as validating, its index as building,
        // the name taken.
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (:User {email: 'same'}), (:User {email: 'same'})")
            .expect("duplicates");
        let mut schema = LabelSchema::new_node_id("User");
        schema.set_mode(SchemaMode::Flexible);
        schema.add_constraint(NodeConstraint {
            name: "user_email".into(),
            properties: vec!["email".into()],
            kind: ConstraintKind::Unique,
            state: ConstraintState::Validating,
            scope: None,
        });
        LocalSchemaStore::new(db.engine())
            .save_label(&schema)
            .expect("plant the schema");
        admit_index(
            db.engine(),
            IndexDescriptor::compound("user_email", "User", vec!["email".into()])
                .unique()
                .sparse()
                .owned_by("user_email"),
        );
        db.engine()
            .put(
                Partition::Schema,
                &encode_constraint_name_key("user_email"),
                b"User",
            )
            .expect("plant the name");
    }

    let mut db = Database::open(dir.path()).expect("reopen");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        None,
        "the refused constraint is withdrawn, never left reading as active"
    );
    assert!(
        index_named(db.engine(), "user_email").is_none(),
        "its index went with it"
    );
    db.execute_cypher("CREATE (:User {email: 'same'})")
        .expect("no uniqueness is left behind");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("the name is free");
}

/// A uniqueness constraint whose validation finished is stored as active.
#[test]
fn a_created_uniqueness_constraint_ends_active() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("seed");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("create");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );
}

/// Run `statement` on another thread while the backfill of the constraint
/// it creates is held at its start by an older open transaction; `meanwhile`
/// runs once the constraint is published as validating, then the older
/// transaction ends and the build goes on. Returns the statement's outcome.
fn create_while_held(
    db: &Database,
    statement: &str,
    name: &str,
    meanwhile: impl FnOnce(&Database),
) -> Result<(), String> {
    let held = db.begin_transaction();
    std::thread::scope(|s| {
        let build = s.spawn(|| db.execute_cypher_shared(statement, None, None, None, None));
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while stored_state(db, "User", name) != Some(ConstraintState::Validating) {
            assert!(
                std::time::Instant::now() < deadline,
                "the constraint was never published as validating"
            );
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        meanwhile(db);
        db.rollback_transaction(held)
            .expect("end the older transaction");
        build
            .join()
            .expect("the build thread")
            .map(|_| ())
            .map_err(|e| e.to_string())
    })
}

/// Another change of the label's schema landing while a uniqueness
/// constraint is validated survives the constraint's activation, and both
/// hold afterwards.
#[test]
fn a_schema_change_during_validation_survives_activation() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x', name: 'a'})")
        .expect("seed");
    create_while_held(
        &db,
        "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE",
        "user_email",
        |db| {
            db.execute_cypher_shared(
                "CREATE CONSTRAINT user_name FOR (u:User) REQUIRE u.name IS NOT NULL",
                None,
                None,
                None,
                None,
            )
            .expect("a change landing during the validation");
        },
    )
    .expect("the uniqueness constraint activates");

    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );
    assert_eq!(
        stored_state(&db, "User", "user_name"),
        Some(ConstraintState::Active)
    );
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x', name: 'b'})"));
    expect_violation(db.execute_cypher("CREATE (:User {email: 'c@x'})"));
}

/// A constraint dropped while it is being validated is gone for good: the
/// build stops without writing under the name, and nothing of either is
/// left.
#[test]
fn a_constraint_dropped_during_validation_leaves_nothing() {
    use super::helpers::index_named;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("seed");
    let outcome = create_while_held(
        &db,
        "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE",
        "user_email",
        |db| {
            db.execute_cypher_shared("DROP CONSTRAINT user_email", None, None, None, None)
                .expect("drop during the validation");
        },
    );
    assert!(outcome.is_err(), "the superseded build fails: {outcome:?}");

    assert_eq!(stored_state(&db, "User", "user_email"), None);
    assert!(index_named(db.engine(), "user_email").is_none());
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("no uniqueness is left behind");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.name IS NOT NULL")
        .expect_err("stored nodes lack a name");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("the name is free");
}

/// A build that fails over stored duplicates and whose withdrawal is then
/// refused keeps the index it published maintained and enforced: in force
/// here exactly as it is stored, not dropped from memory ahead of a commit
/// that never landed.
#[test]
fn a_refused_withdrawal_keeps_the_published_index_enforced() {
    use super::helpers::index_named;
    let (mut db, refuse, _dir) = open_db_refusing_definition_deletes();
    db.execute_cypher("CREATE (:User {email: 'same'}), (:User {email: 'same'})")
        .expect("duplicates");
    refuse.store(true, Ordering::SeqCst);
    let err = db
        .execute_cypher("CREATE UNIQUE INDEX user_email ON :User(email)")
        .expect_err("duplicates");
    assert!(err.to_string().contains("was not withdrawn"), "{err}");
    assert!(
        index_named(db.engine(), "user_email").is_some(),
        "the definition is still stored"
    );
    db.execute_cypher("CREATE (:User {email: 'new'})")
        .expect("first holder");
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'new'})"));

    refuse.store(false, Ordering::SeqCst);
    let err = db
        .execute_cypher("DROP INDEX user_email")
        .expect_err("the index belongs to its constraint");
    assert!(
        err.to_string()
            .contains("index 'user_email' belongs to constraint 'user_email'"),
        "{err}"
    );
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'new'})"));
    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect("drop");
    db.execute_cypher("CREATE (:User {email: 'new'})")
        .expect("withdrawn now");
}

/// A DROP CONSTRAINT whose commit is refused changes nothing: the
/// constraint and its index stay in force.
#[test]
fn a_refused_drop_keeps_the_constraint_in_force() {
    let (mut db, refuse, _dir) = open_db_refusing_definition_deletes();
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("create");
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("first holder");

    refuse.store(true, Ordering::SeqCst);
    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect_err("the commit is refused");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x'})"));

    refuse.store(false, Ordering::SeqCst);
    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect("drop");
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("lifted");
}

/// A node set to a document under a nested path breaks a type constraint on
/// that property as the transaction leaves it, while a later plain SET in
/// the same statement that restores the type is accepted.
#[test]
fn a_nested_write_is_judged_by_the_state_it_leaves() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE CONSTRAINT item_qty FOR (i:Item) REQUIRE i.qty IS :: INTEGER")
        .expect("create");
    db.execute_cypher("CREATE (:Item {name: 'a', qty: 1})")
        .expect("seed");
    expect_violation(db.execute_cypher("MATCH (i:Item) SET i.qty.unit = 'kg'"));
    db.execute_cypher("MATCH (i:Item) SET i.qty.unit = 'kg', i.qty = 2")
        .expect("restored before the commit");
    assert_eq!(
        count(
            &mut db,
            "MATCH (i:Item) WHERE i.qty = 2 RETURN count(i) AS c"
        ),
        1
    );
}

/// A build that fails over stored duplicates withdraws only its own
/// constraint: another constraint of the label created while it was being
/// validated stays active, and the failed one's name is free again.
#[test]
fn a_failed_validation_withdraws_only_its_own_constraint() {
    use super::helpers::index_named;
    let (mut db, _dir) = open_db();
    db.execute_cypher(
        "CREATE (:User {email: 'same', name: 'a'}), (:User {email: 'same', name: 'b'})",
    )
    .expect("duplicates");
    let outcome = create_while_held(
        &db,
        "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE",
        "user_email",
        |db| {
            db.execute_cypher_shared(
                "CREATE CONSTRAINT user_name FOR (u:User) REQUIRE u.name IS NOT NULL",
                None,
                None,
                None,
                None,
            )
            .expect("another constraint during the validation");
        },
    );
    let err = outcome.expect_err("the stored duplicates refuse it");
    assert!(err.contains("unique constraint violated"), "{err}");

    assert_eq!(stored_state(&db, "User", "user_email"), None);
    assert_eq!(
        stored_state(&db, "User", "user_name"),
        Some(ConstraintState::Active),
        "the constraint created meanwhile survives the withdrawal"
    );
    assert!(index_named(db.engine(), "user_email").is_none());
    expect_violation(db.execute_cypher("CREATE (:User {email: 'x'})"));
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.name IS NOT NULL")
        .expect_err("an equivalent constraint exists under another name");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS NOT NULL")
        .expect("the failed name is free");
}

/// Set by the parent of the interrupted-build test; the child does nothing
/// without it.
const INTERRUPTED_CHILD_DIR: &str = "COORDINODE_CONSTRAINT_INTERRUPTED_CHILD_DIR";

/// Child half of `a_kill_between_publication_and_activation_leaves_it_validating`:
/// a uniqueness constraint published by its first commit, its backfill held
/// by an older open transaction, and death before the commit that makes it
/// active.
#[test]
fn interrupted_child_publishes_a_constraint_then_aborts() {
    let Some(dir) = std::env::var_os(INTERRUPTED_CHILD_DIR) else {
        return;
    };
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("seed");
    let db = &db;
    let _held = db.begin_transaction();
    std::thread::scope(|s| {
        s.spawn(|| {
            let _ = db.execute_cypher_shared(
                "CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE",
                None,
                None,
                None,
                None,
            );
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while stored_state(db, "User", "user_email") != Some(ConstraintState::Validating) {
            assert!(std::time::Instant::now() < deadline, "never published");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        std::process::abort();
    });
}

/// A process killed between the commit that publishes a uniqueness
/// constraint and the one that activates it leaves, after the journal
/// replay, a build the next open finishes: the stored node that predates
/// the constraint is indexed, the constraint becomes active only then, and
/// a duplicate of that stored value is refused.
#[test]
fn a_kill_between_publication_and_activation_is_finished_on_open() {
    use super::helpers::index_named;
    use coordinode_query::index::IndexState;
    let dir = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::constraints::interrupted_child_publishes_a_constraint_then_aborts",
            "--nocapture",
        ])
        .env(INTERRUPTED_CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(6),
            "the child must die by abort after the publication, got {status:?}"
        );
    }
    #[cfg(not(unix))]
    assert!(!status.success(), "the child must die, got {status:?}");

    let mut db = Database::open(dir.path()).expect("reopen after the kill");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );
    let def = index_named(db.engine(), "user_email").expect("the index is published");
    assert_eq!(def.state, IndexState::Ready);
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x'})"));
    let rows = db
        .execute_cypher(
            "CREATE CONSTRAINT user_email IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE",
        )
        .expect("in place");
    assert_eq!(rows[0].get("created"), Some(&Value::Bool(false)));
}

/// Set by the parent of the replay test; the child does nothing without it.
const REPLAY_CHILD_DIR: &str = "COORDINODE_CONSTRAINT_REPLAY_CHILD_DIR";

/// Child half of `constraint_ddl_survives_a_kill_and_replay`: constraint DDL
/// and a write, then death without a single destructor, as under SIGKILL.
#[test]
fn replay_child_creates_constraints_then_aborts() {
    let Some(dir) = std::env::var_os(REPLAY_CHILD_DIR) else {
        return;
    };
    let mut db = Database::open(std::path::Path::new(&dir)).expect("open in child");
    db.execute_cypher("CREATE CONSTRAINT user_email FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("unique");
    db.execute_cypher("CREATE CONSTRAINT user_name FOR (u:User) REQUIRE u.name IS NOT NULL")
        .expect("not null");
    db.execute_cypher("CREATE CONSTRAINT user_age FOR (u:User) REQUIRE u.age IS UNIQUE")
        .expect("dropped below");
    db.execute_cypher("CREATE (:User {email: 'a@x', name: 'a', age: 1})")
        .expect("write");
    db.execute_cypher("DROP CONSTRAINT user_age").expect("drop");
    std::process::abort();
}

/// Constraint DDL acknowledged before a kill is what the replayed database
/// holds: each surviving constraint in force with its state, name and
/// index, the dropped one gone with its index, and none of it half applied.
#[test]
fn constraint_ddl_survives_a_kill_and_replay() {
    use super::helpers::index_named;
    let dir = tempfile::tempdir().expect("tempdir");
    let status = std::process::Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "integration::constraints::replay_child_creates_constraints_then_aborts",
            "--nocapture",
        ])
        .env(REPLAY_CHILD_DIR, dir.path())
        .status()
        .expect("run the child");
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(
            status.signal(),
            Some(6),
            "the child must die by abort after its DDL, got {status:?}"
        );
    }
    #[cfg(not(unix))]
    assert!(!status.success(), "the child must die, got {status:?}");

    let mut db = Database::open(dir.path()).expect("reopen after the kill");
    assert_eq!(
        stored_state(&db, "User", "user_email"),
        Some(ConstraintState::Active)
    );
    assert_eq!(
        stored_state(&db, "User", "user_name"),
        Some(ConstraintState::Active)
    );
    assert_eq!(stored_state(&db, "User", "user_age"), None);
    assert!(
        index_named(db.engine(), "user_age").is_none(),
        "the dropped constraint's index is gone"
    );
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x', name: 'b'})"));
    expect_violation(db.execute_cypher("CREATE (:User {email: 'b@x'})"));
    db.execute_cypher("CREATE (:User {email: 'c@x', name: 'c', age: 1})")
        .expect("the dropped uniqueness holds no more");
    db.execute_cypher("DROP CONSTRAINT user_email")
        .expect("the name survived the replay");
}

// ── Unique indexes ───────────────────────────────────────────────────

/// The unique indexes on `label`, by name.
fn unique_indexes_on(db: &Database, label: &str) -> Vec<String> {
    use coordinode_query::index::ops::list_index_definitions;
    let mut names: Vec<String> = list_index_definitions(db.engine())
        .expect("list indexes")
        .into_iter()
        .filter(|d| d.label == label && d.unique)
        .filter_map(|d| d.descriptor.name)
        .collect();
    names.sort();
    names
}

/// The owner recorded on index `name`.
fn index_owner(db: &Database, name: &str) -> Option<String> {
    super::helpers::index_named(db.engine(), name)
        .expect("the index is defined")
        .descriptor
        .owner
}

/// The indexes defined on `label`, by name.
fn indexes_on(db: &Database, label: &str) -> Vec<String> {
    use coordinode_query::index::ops::list_index_definitions;
    list_index_definitions(db.engine())
        .expect("list indexes")
        .into_iter()
        .filter(|d| d.label == label)
        .filter_map(|d| d.descriptor.name)
        .collect()
}

/// A type definition never declares uniqueness: UNIQUE inline in CREATE
/// NODE TYPE is refused with nothing written, and a type created through
/// Cypher or the embedded API, required properties included, builds no
/// index and declares no constraint. Uniqueness is a named constraint.
#[test]
fn a_type_definition_declares_no_uniqueness_and_builds_no_index() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef};
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE NODE TYPE User WITH (email: string UNIQUE)")
        .expect_err("UNIQUE is not a property modifier");
    assert!(
        LocalSchemaStore::new(db.engine())
            .load_label("User")
            .expect("load schema")
            .is_none(),
        "the refused definition left nothing"
    );

    db.execute_cypher("CREATE NODE TYPE User WITH (email: string NOT NULL, name: string)")
        .expect("type");
    let mut account = LabelSchema::new_node_id("Account");
    account.add_property(PropertyDef::new("email", PropertyType::String).not_null());
    db.create_label_schema(account).expect("embedded type");

    assert!(db.constraints().expect("constraints").is_empty());
    assert!(indexes_on(&db, "User").is_empty());
    assert!(indexes_on(&db, "Account").is_empty());
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:User {email: 'a@x'})")
        .expect("no uniqueness without a constraint");
    db.execute_cypher("CREATE (:Account {email: 'a@x'}), (:Account {email: 'a@x'})")
        .expect("no uniqueness without a constraint");
}

/// `CREATE UNIQUE INDEX` declares the same invariant a uniqueness
/// constraint does: it is listed as a constraint owning the index, and a
/// constraint asked for over the same property finds it.
#[test]
fn create_unique_index_declares_a_constraint_owning_it() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX user_email ON :User(email)")
        .expect("unique index");

    let listed = db.constraints().expect("constraints");
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0].constraint.name, "user_email");
    assert_eq!(listed[0].constraint.state, ConstraintState::Active);
    assert_eq!(listed[0].backing_index.as_deref(), Some("user_email"));

    db.execute_cypher("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect("an equivalent constraint is found");
    assert_eq!(
        unique_indexes_on(&db, "User"),
        ["user_email"],
        "no second index"
    );
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("first");
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x'})"));
}

/// The options a `CREATE UNIQUE INDEX` states shape the index its
/// constraint owns: SPARSE leaves nodes without the value out, and a stated
/// maintenance profile is the index's.
#[test]
fn create_unique_index_keeps_its_stated_options() {
    use coordinode_query::index::IndexProfile;
    let (mut db, _dir) = open_db();
    let rows = db
        .execute_cypher(
            "CREATE UNIQUE SPARSE INDEX user_email ON :User(email) \
             OPTIONS {maintenance: 'derived'}",
        )
        .expect("unique index");
    assert_eq!(rows[0].get("unique"), Some(&Value::Bool(true)));
    assert_eq!(rows[0].get("sparse"), Some(&Value::Bool(true)));
    assert_eq!(
        rows[0].get("maintenance"),
        Some(&Value::String("DERIVED".into()))
    );
    let def = super::helpers::index_named(db.engine(), "user_email").expect("defined");
    assert!(def.sparse && def.unique);
    assert_eq!(def.maintenance.profile, IndexProfile::Derived);
    assert_eq!(def.owner.as_deref(), Some("user_email"));
}

/// A partial unique index declares a uniqueness among the nodes its filter
/// admits: it is listed as a constraint with that scope, owning the index,
/// so the catalog shows every uniqueness the engine enforces. Only nodes in
/// scope are compared; the index goes with the constraint, never alone; a
/// constraint over the same property for every node is a different one.
#[test]
fn a_partial_unique_index_is_a_scoped_constraint_owning_it() {
    use coordinode_query::index::definition::PartialFilter;
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE UNIQUE INDEX active_email ON :User(email) WHERE n.active = true")
        .expect("partial unique index");
    let listed = db.constraints().expect("constraints");
    assert_eq!(listed.len(), 1, "{listed:?}");
    assert_eq!(listed[0].constraint.name, "active_email");
    assert_eq!(listed[0].constraint.kind, ConstraintKind::Unique);
    assert_eq!(listed[0].constraint.state, ConstraintState::Active);
    assert_eq!(
        listed[0].constraint.scope,
        Some(PartialFilter::PropertyEqualsBool {
            property: "active".into(),
            value: true,
        })
    );
    assert_eq!(listed[0].backing_index.as_deref(), Some("active_email"));
    assert_eq!(
        index_owner(&db, "active_email").as_deref(),
        Some("active_email")
    );

    db.execute_cypher("CREATE (:User {email: 'a@x', active: true})")
        .expect("first active");
    db.execute_cypher("CREATE (:User {email: 'a@x', active: false})")
        .expect("an inactive node is outside the scope");
    expect_unique_violation(db.execute_cypher("CREATE (:User {email: 'a@x', active: true})"));

    let drop_index = db
        .execute_cypher("DROP INDEX active_email")
        .expect_err("the index belongs to its constraint");
    assert!(
        drop_index.to_string().contains("drop the constraint"),
        "{drop_index}"
    );
    // Every node, not only the active ones: a different requirement.
    db.execute_cypher("CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.email IS UNIQUE")
        .expect_err("the stored inactive duplicate breaks the label-wide uniqueness");
    db.execute_cypher("DROP CONSTRAINT active_email")
        .expect("drop with its index");
    assert!(db.constraints().expect("constraints").is_empty());
    assert!(super::helpers::index_named(db.engine(), "active_email").is_none());
    db.execute_cypher("CREATE (:User {email: 'a@x', active: true})")
        .expect("nothing constrains the value any more");
}

/// A partial index scoped by a predicate it cannot hold is refused, never
/// built over every node instead.
#[test]
fn a_partial_index_with_an_unsupported_predicate_is_refused() {
    let (mut db, _dir) = open_db();
    db.execute_cypher("CREATE (:User {email: 'a@x', age: 3})")
        .expect("seed");
    for statement in [
        "CREATE UNIQUE INDEX adult_email ON :User(email) WHERE n.age > 17",
        "CREATE INDEX adult_email ON :User(email) WHERE n.age > 17",
    ] {
        let error = db.execute_cypher(statement).expect_err(statement);
        assert!(error.to_string().contains("WHERE supports only"), "{error}");
    }
    assert!(db.constraints().expect("constraints").is_empty());
    assert!(super::helpers::index_named(db.engine(), "adult_email").is_none());
}
