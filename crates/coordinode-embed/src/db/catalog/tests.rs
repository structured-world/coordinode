use super::*;
use coordinode_core::schema::definition::{ConstraintState, PropertyDef, PropertyType, SchemaMode};

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn user_definition() -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("User");
    schema.set_mode(SchemaMode::Flexible);
    schema.add_property(PropertyDef::new("email", PropertyType::String));
    schema.add_property(PropertyDef::new("age", PropertyType::Int));
    schema
}

fn declare(name: &str, property: &str, kind: ConstraintKind) -> ConstraintDeclaration {
    ConstraintDeclaration {
        name: Some(name.into()),
        label: "User".into(),
        properties: vec![property.into()],
        kind,
        if_not_exists: false,
    }
}

fn constraint_names(db: &Database) -> Vec<String> {
    db.constraints()
        .expect("list constraints")
        .into_iter()
        .map(|c| c.constraint.name)
        .collect()
}

fn catalog_error(e: DatabaseError) -> ExecutionError {
    match e {
        DatabaseError::Execution(e) => e,
        other => panic!("expected an execution error, got {other:?}"),
    }
}

/// A created uniqueness constraint is listed active, bound to the index it
/// owns; dropping it removes both, and a second drop finds nothing.
#[test]
fn a_constraint_is_listed_with_its_index_and_dropped_with_it() {
    let (db, _dir) = open();
    db.define_label(user_definition()).expect("define");

    let created = db
        .create_constraint(declare("user_email", "email", ConstraintKind::Unique))
        .expect("create");
    assert_eq!(created.label, "User");
    assert_eq!(created.constraint.state, ConstraintState::Active);
    assert_eq!(created.backing_index.as_deref(), Some("user_email"));
    assert_eq!(db.constraints().expect("list"), vec![created]);

    let not_null = db
        .create_constraint(declare("user_age", "age", ConstraintKind::NotNull))
        .expect("create NOT NULL");
    assert_eq!(not_null.backing_index, None, "NOT NULL owns no index");

    assert!(db.drop_constraint("user_email", false).expect("drop"));
    assert_eq!(constraint_names(&db), vec!["user_age".to_string()]);
    assert!(
        db.index_registry.get("user_email").is_none(),
        "the owned index goes with the constraint"
    );
    assert!(matches!(
        catalog_error(db.drop_constraint("user_email", false).unwrap_err()),
        ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Constraint,
            ..
        }
    ));
    assert!(!db.drop_constraint("user_email", true).expect("IF EXISTS"));
}

/// Replacing a label's type facts keeps every constraint it has, with its
/// state and the exact index it owns, at the next revision.
#[test]
fn replacing_a_definition_keeps_its_constraints_and_their_indexes() {
    let (db, _dir) = open();
    let first = db.define_label(user_definition()).expect("define");
    db.create_constraint(declare("user_email", "email", ConstraintKind::Unique))
        .expect("unique");
    db.create_constraint(declare("user_age", "age", ConstraintKind::NotNull))
        .expect("not null");
    let before = db.constraints().expect("list");

    let mut replacement = user_definition();
    replacement.add_property(PropertyDef::new("bio", PropertyType::String));
    let revision = db.create_label_schema(replacement).expect("replace");

    assert!(revision > first);
    assert_eq!(db.constraints().expect("list"), before);
    let schema = db
        .label_schemas()
        .expect("labels")
        .into_iter()
        .find(|s| s.name == "User")
        .expect("User");
    assert!(schema.get_property("bio").is_some());
    assert_eq!(schema.schema_revision, revision);
}

/// A definition replaced while a constraint is created concurrently does not
/// publish over it: the replacement read the definition the constraint's
/// commit changed, so its own commit is refused and the constraint stays.
#[test]
fn a_replacement_racing_a_constraint_creation_does_not_lose_it() {
    let (db, _dir) = open();
    db.define_label(user_definition()).expect("define");

    let mut txn = db.begin_catalog_txn();
    let mut replacement = user_definition();
    db.stage_label_definition(&mut txn, &mut replacement, ExistingDefinition::Replace)
        .expect("stage the replacement");
    // NOT NULL: a uniqueness constraint's backfill would wait for the staged
    // transaction above to end, which is the point of that wait.
    db.create_constraint(declare("user_age", "age", ConstraintKind::NotNull))
        .expect("the concurrent constraint commits first");

    assert!(matches!(
        catalog_error(db.commit_catalog_txn(txn).unwrap_err()),
        ExecutionError::Conflict(_)
    ));
    assert_eq!(constraint_names(&db), vec!["user_age".to_string()]);
}

/// The reverse race: a constraint dropped while a replacement that read it
/// is in flight is not brought back by that replacement.
#[test]
fn a_replacement_racing_a_constraint_drop_does_not_bring_it_back() {
    let (db, _dir) = open();
    db.define_label(user_definition()).expect("define");
    db.create_constraint(declare("user_age", "age", ConstraintKind::NotNull))
        .expect("create");

    let mut txn = db.begin_catalog_txn();
    let mut replacement = user_definition();
    db.stage_label_definition(&mut txn, &mut replacement, ExistingDefinition::Replace)
        .expect("stage the replacement");
    assert!(
        replacement.constraint("user_age").is_some(),
        "the staged replacement carries the constraint it read"
    );
    assert!(db.drop_constraint("user_age", false).expect("drop"));

    assert!(matches!(
        catalog_error(db.commit_catalog_txn(txn).unwrap_err()),
        ExecutionError::Conflict(_)
    ));
    assert!(constraint_names(&db).is_empty());
}

/// A replacement a kept constraint cannot hold under is refused as a whole:
/// the previous definition and its constraints stay as they were.
#[test]
fn a_replacement_its_constraints_cannot_hold_under_is_refused_atomically() {
    let (db, _dir) = open();
    db.define_label(user_definition()).expect("define");
    db.create_constraint(declare(
        "user_email_type",
        "email",
        ConstraintKind::Type(PropertyType::String),
    ))
    .expect("type constraint");
    let user = |db: &Database| {
        db.label_schemas()
            .expect("labels")
            .into_iter()
            .find(|s| s.name == "User")
            .expect("User")
    };
    let before_schema = user(&db);
    let before = db.constraints().expect("list");

    let mut replacement = user_definition();
    replacement.add_property(PropertyDef::new("email", PropertyType::Int));
    assert!(matches!(
        catalog_error(db.create_label_schema(replacement).unwrap_err()),
        ExecutionError::CatalogRefused(_)
    ));

    assert_eq!(user(&db), before_schema);
    assert_eq!(db.constraints().expect("list"), before);
}

/// A replacement the stored nodes break is not published, and the previous
/// definition with its constraints stays in force.
#[test]
fn a_replacement_stored_nodes_break_leaves_the_previous_definition() {
    let (mut db, _dir) = open();
    db.define_label(user_definition()).expect("define");
    db.create_constraint(declare("user_email", "email", ConstraintKind::Unique))
        .expect("unique");
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("a node without an age");
    let before_schema = db
        .label_schemas()
        .expect("labels")
        .into_iter()
        .find(|s| s.name == "User")
        .expect("User");
    let before = db.constraints().expect("list");

    let mut replacement = user_definition();
    replacement.add_property(PropertyDef::new("age", PropertyType::Int).not_null());
    assert!(matches!(
        catalog_error(db.create_label_schema(replacement).unwrap_err()),
        ExecutionError::SchemaViolation(_)
    ));

    let after_schema = db
        .label_schemas()
        .expect("labels")
        .into_iter()
        .find(|s| s.name == "User")
        .expect("User");
    assert_eq!(after_schema, before_schema);
    assert_eq!(db.constraints().expect("list"), before);
}

/// A type definition never declares uniqueness or any other constraint, and
/// defining a label twice is refused rather than replacing the first.
#[test]
fn a_definition_carrying_constraints_or_defined_twice_is_refused() {
    let (db, _dir) = open();

    let mut flagged = user_definition();
    let mut email = PropertyDef::new("email", PropertyType::String);
    email.unique = true;
    flagged.add_property(email);
    assert!(matches!(
        catalog_error(db.define_label(flagged).unwrap_err()),
        ExecutionError::CatalogRefused(_)
    ));

    let mut carrying = user_definition();
    carrying.add_constraint(NodeConstraint {
        name: "smuggled".into(),
        properties: vec!["email".into()],
        kind: ConstraintKind::Unique,
        state: ConstraintState::Active,
    });
    assert!(matches!(
        catalog_error(db.create_label_schema(carrying).unwrap_err()),
        ExecutionError::CatalogRefused(_)
    ));
    assert!(
        db.label_schemas().expect("labels").is_empty(),
        "neither refused definition was published"
    );

    db.define_label(user_definition()).expect("define");
    assert!(matches!(
        catalog_error(db.define_label(user_definition()).unwrap_err()),
        ExecutionError::CatalogObjectExists {
            object: CatalogObject::Label,
            ..
        }
    ));
}

/// A constraint that could never hold under the label's definition is
/// refused at creation, through the same check a replacement runs.
#[test]
fn a_constraint_the_definition_contradicts_is_refused() {
    let (db, _dir) = open();
    db.define_label(user_definition()).expect("define");
    assert!(matches!(
        catalog_error(
            db.create_constraint(declare(
                "user_age_type",
                "age",
                ConstraintKind::Type(PropertyType::String),
            ))
            .unwrap_err()
        ),
        ExecutionError::CatalogRefused(_)
    ));
    assert!(constraint_names(&db).is_empty());
}

/// Defining an edge type the catalog already knows is refused.
#[test]
fn an_edge_type_is_defined_once() {
    let (db, _dir) = open();
    db.define_edge_type(EdgeTypeSchema::new("KNOWS"))
        .expect("define");
    assert!(matches!(
        catalog_error(
            db.define_edge_type(EdgeTypeSchema::new("KNOWS"))
                .unwrap_err()
        ),
        ExecutionError::CatalogObjectExists {
            object: CatalogObject::EdgeType,
            ..
        }
    ));
    assert_eq!(
        db.edge_type_names().expect("names"),
        vec!["KNOWS".to_string()]
    );
}
