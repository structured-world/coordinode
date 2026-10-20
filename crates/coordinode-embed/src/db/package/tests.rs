use core::time::Duration;

use coordinode_core::schema::definition::{
    ConstraintKind, ConstraintState, EdgeTypeSchema, LabelSchema, PropertyDef, PropertyType,
    SchemaMode,
};
use coordinode_query::executor::runner::CatalogObject;

use super::{
    Disposition, EdgeTypeDeclaration, Intent, LabelDeclaration, MAX_SCHEMA_PACKAGE_DECLARATIONS,
    PackageConstraint, Readiness, SchemaPackage,
};
use crate::db::{Database, DatabaseError};

const WAIT: Duration = Duration::from_secs(30);

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn user(extra: Option<&str>) -> LabelSchema {
    let mut schema = LabelSchema::new_node_id("User");
    schema.set_mode(SchemaMode::Flexible);
    schema.add_property(PropertyDef::new("email", PropertyType::String));
    schema.add_property(PropertyDef::new("handle", PropertyType::String));
    if let Some(name) = extra {
        schema.add_property(PropertyDef::new(name, PropertyType::Int));
    }
    schema
}

fn constraint(name: &str, property: &str, kind: ConstraintKind) -> PackageConstraint {
    PackageConstraint {
        name: Some(name.into()),
        label: "User".into(),
        properties: vec![property.into()],
        kind,
        scope: None,
        on_duplicate_rename: None,
    }
}

/// A label, an edge type and two uniqueness constraints on the label, the
/// shape a host declares at every open.
fn package() -> SchemaPackage {
    SchemaPackage {
        // Constraints first: a constraint finds its label in the package
        // whatever the declaration order.
        constraints: vec![
            constraint("user_email", "email", ConstraintKind::Unique),
            constraint("user_handle", "handle", ConstraintKind::Unique),
        ],
        labels: vec![LabelDeclaration {
            schema: user(None),
            intent: Intent::Ensure,
        }],
        edge_types: vec![EdgeTypeDeclaration {
            schema: EdgeTypeSchema::new("FOLLOWS"),
            intent: Intent::Ensure,
        }],
    }
}

/// A fresh package publishes everything in one commit: one revision of the
/// label carrying both of its new constraints, each validating with its
/// build started, all ready after the wait.
#[test]
fn a_fresh_package_publishes_every_declaration_in_one_revision() {
    let (db, _dir) = open();
    let published = db.publish_schema(package()).expect("publish");
    assert!(published.committed);
    let dispositions: Vec<_> = published
        .objects
        .iter()
        .map(|o| (o.object, o.name.as_str(), o.disposition, o.revision))
        .collect();
    assert_eq!(
        dispositions,
        vec![
            (CatalogObject::Label, "User", Disposition::Created, 1),
            (CatalogObject::EdgeType, "FOLLOWS", Disposition::Created, 1),
            (
                CatalogObject::Constraint,
                "user_email",
                Disposition::Created,
                1
            ),
            (
                CatalogObject::Constraint,
                "user_handle",
                Disposition::Created,
                1
            ),
        ]
    );
    for object in &published.objects[2..] {
        assert_eq!(object.state, Some(ConstraintState::Validating));
        assert!(object.operation.is_some(), "{object:?}");
    }

    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(readiness.all_ready(), "{readiness:?}");
    let schema = db
        .label_schemas()
        .unwrap()
        .into_iter()
        .find(|s| s.name == "User")
        .unwrap();
    // Acceptance is one revision; each build activates its constraint in a
    // revision of its own.
    assert_eq!(schema.schema_revision, 3, "each build activates its own");
    assert!(
        schema
            .constraints()
            .iter()
            .all(|c| c.state == ConstraintState::Active)
    );
    assert_eq!(db.edge_type_schemas().unwrap().len(), 1);
}

/// The same package again changes nothing: no commit, no revision, no
/// build. Every object is reported retained with its current revision and
/// state.
#[test]
fn an_unchanged_package_commits_nothing() {
    let (db, _dir) = open();
    let first = db.publish_schema(package()).expect("publish");
    db.await_schema(&first, WAIT).expect("await");
    let builds = db.index_build_status().unwrap().len();

    let again = db.publish_schema(package()).expect("publish again");
    assert!(!again.committed);
    assert!(
        again
            .objects
            .iter()
            .all(|o| o.disposition == Disposition::Retained),
        "{again:?}"
    );
    let revisions: Vec<_> = again.objects.iter().map(|o| o.revision).collect();
    assert_eq!(revisions, vec![3, 1, 3, 3]);
    assert!(
        again.objects[2..]
            .iter()
            .all(|o| o.state == Some(ConstraintState::Active))
    );
    assert_eq!(db.index_build_status().unwrap().len(), builds);
    assert!(db.await_schema(&again, WAIT).unwrap().all_ready());
}

/// A declaration differing from the definition in place, without asking
/// for the change, refuses the whole package: the new edge type it also
/// declares is not published either.
#[test]
fn a_differing_declaration_refuses_the_whole_package() {
    let (db, _dir) = open();
    db.publish_schema(package()).expect("publish");

    let mut changed = package();
    changed.labels[0].schema = user(Some("age"));
    changed.edge_types.push(EdgeTypeDeclaration {
        schema: EdgeTypeSchema::new("BLOCKS"),
        intent: Intent::Ensure,
    });
    let refused = db.publish_schema(changed).unwrap_err();
    assert!(
        matches!(
            refused,
            DatabaseError::CatalogObjectDiffers {
                object: CatalogObject::Label,
                ref name,
                ..
            } if name == "User"
        ),
        "{refused:?}"
    );
    let edges: Vec<_> = db
        .edge_type_schemas()
        .unwrap()
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert_eq!(edges, vec!["FOLLOWS".to_string()]);
}

/// A change names the revision it was read at: at that revision it is
/// published as the next one and keeps the label's constraints; at another
/// it is refused with the revision the label is at.
#[test]
fn a_change_is_published_only_at_the_expected_revision() {
    let (db, _dir) = open();
    let first = db.publish_schema(package()).expect("publish");
    db.await_schema(&first, WAIT).expect("await");

    let stale = SchemaPackage {
        labels: vec![LabelDeclaration {
            schema: user(Some("age")),
            intent: Intent::Change {
                expected_revision: 1,
            },
        }],
        ..SchemaPackage::default()
    };
    let refused = db.publish_schema(stale).unwrap_err();
    assert!(
        matches!(
            refused,
            DatabaseError::CatalogRevisionMismatch {
                expected: 1,
                current: Some(3),
                ..
            }
        ),
        "{refused:?}"
    );

    let change = SchemaPackage {
        labels: vec![LabelDeclaration {
            schema: user(Some("age")),
            intent: Intent::Change {
                expected_revision: 3,
            },
        }],
        ..SchemaPackage::default()
    };
    let changed = db.publish_schema(change).expect("change");
    assert!(changed.committed);
    assert_eq!(changed.objects[0].disposition, Disposition::Changed);
    assert_eq!(changed.objects[0].revision, 4);
    let schema = db
        .label_schemas()
        .unwrap()
        .into_iter()
        .find(|s| s.name == "User")
        .unwrap();
    assert!(schema.properties.contains_key("age"));
    assert_eq!(schema.constraints().len(), 2, "omission drops nothing");
}

/// A change of an object that does not exist is refused: there is no
/// revision it could be at.
#[test]
fn a_change_of_an_absent_label_is_refused() {
    let (db, _dir) = open();
    let change = SchemaPackage {
        labels: vec![LabelDeclaration {
            schema: user(None),
            intent: Intent::Change {
                expected_revision: 1,
            },
        }],
        ..SchemaPackage::default()
    };
    assert!(matches!(
        db.publish_schema(change).unwrap_err(),
        DatabaseError::CatalogRevisionMismatch { current: None, .. }
    ));
    assert!(db.label_schemas().unwrap().is_empty());
}

/// A constraint whose name holds another requirement is a conflict, not
/// the constraint asked for.
#[test]
fn a_constraint_name_holding_another_requirement_is_a_conflict() {
    let (db, _dir) = open();
    db.publish_schema(package()).expect("publish");
    let mut other = package();
    other.constraints[0].properties = vec!["handle".into()];
    other.constraints.truncate(1);
    assert!(matches!(
        db.publish_schema(other).unwrap_err(),
        DatabaseError::CatalogObjectDiffers {
            object: CatalogObject::Constraint,
            ..
        }
    ));
}

/// A package naming an object twice, or larger than the bound, is refused
/// before anything is read or written.
#[test]
fn a_malformed_package_is_refused_before_publication() {
    let (db, _dir) = open();
    let mut twice = package();
    twice
        .constraints
        .push(constraint("user_email", "handle", ConstraintKind::Unique));
    assert!(matches!(
        db.publish_schema(twice).unwrap_err(),
        DatabaseError::CatalogRefused(ref why) if why.contains("twice")
    ));

    let oversize = SchemaPackage {
        edge_types: (0..=MAX_SCHEMA_PACKAGE_DECLARATIONS)
            .map(|i| EdgeTypeDeclaration {
                schema: EdgeTypeSchema::new(format!("E{i}")),
                intent: Intent::Ensure,
            })
            .collect(),
        ..SchemaPackage::default()
    };
    assert!(matches!(
        db.publish_schema(oversize).unwrap_err(),
        DatabaseError::CatalogRefused(ref why) if why.contains("at most")
    ));
    assert!(db.label_schemas().unwrap().is_empty());
    assert!(db.edge_type_schemas().unwrap().is_empty());
}

/// Stored duplicates fail the build of a uniqueness after publication: the
/// wait reports it failed and the constraint is withdrawn, while the rest
/// of the package stays published.
#[test]
fn a_failed_build_leaves_the_rest_of_the_package_published() {
    let (mut db, _dir) = open();
    db.execute_cypher(
        "CREATE (:User {email: 'a@x', handle: 'a'}), (:User {email: 'a@x', handle: 'b'})",
    )
    .expect("seed");
    let published = db.publish_schema(package()).expect("publish");
    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(
        matches!(
            readiness.of(CatalogObject::Constraint, "user_email"),
            Some(Readiness::Failed(_))
        ),
        "{readiness:?}"
    );
    assert_eq!(
        readiness.of(CatalogObject::Constraint, "user_handle"),
        Some(&Readiness::Ready)
    );
    let schema = db
        .label_schemas()
        .unwrap()
        .into_iter()
        .find(|s| s.name == "User")
        .unwrap();
    let names: Vec<_> = schema
        .constraints()
        .iter()
        .map(|c| c.name.as_str())
        .collect();
    assert_eq!(names, vec!["user_handle"]);
    assert_eq!(db.edge_type_schemas().unwrap().len(), 1);
}

/// A presence constraint is proved over the stored nodes in the background,
/// enforced on writes meanwhile: one the stored nodes break is withdrawn
/// by its proof, with its name released, and the rest of the package stays.
#[test]
fn a_presence_constraint_broken_by_stored_nodes_is_withdrawn() {
    let (mut db, _dir) = open();
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("seed");
    let mut broken = package();
    broken.constraints.push(constraint(
        "user_handle_present",
        "handle",
        ConstraintKind::NotNull,
    ));
    let published = db
        .publish_schema(broken)
        .expect("acceptance is not readiness");
    let presence = published
        .objects
        .iter()
        .find(|o| o.name == "user_handle_present")
        .expect("published");
    assert_eq!(presence.state, Some(ConstraintState::Validating));
    assert!(presence.operation.is_some());

    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(
        matches!(
            readiness.of(CatalogObject::Constraint, "user_handle_present"),
            Some(Readiness::Failed(why)) if why.contains("breaks it")
        ),
        "{readiness:?}"
    );
    let names: Vec<_> = db
        .constraints()
        .unwrap()
        .into_iter()
        .map(|c| c.constraint.name)
        .collect();
    assert!(!names.contains(&"user_handle_present".to_string()));
    assert_eq!(db.edge_type_schemas().unwrap().len(), 1);
    // The name is free again.
    db.execute_cypher("CREATE (:User {email: 'b@x', handle: 'b'})")
        .expect("write");
}

/// One certification proves every label a package publishes a candidate
/// for, each on its own outcome: the label the stored nodes satisfy is put
/// in force, the one they break withdrawn, neither rolling back the other.
#[test]
fn one_certification_settles_each_label_on_its_own() {
    let (mut db, _dir) = open();
    db.execute_cypher("CREATE (:User {email: 'a@x'}), (:Org {name: 'o', loose: 1})")
        .expect("seed");
    let mut org = LabelSchema::new_node_id("Org");
    org.set_mode(SchemaMode::Strict);
    org.add_property(PropertyDef::new("name", PropertyType::String));
    let mut package = strict_user();
    package.labels.push(LabelDeclaration {
        schema: org,
        intent: Intent::Ensure,
    });
    let published = db.publish_schema(package).expect("publish");
    assert_eq!(
        published.objects[0].operation, published.objects[1].operation,
        "one operation for both"
    );
    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert_eq!(
        readiness.of(CatalogObject::Label, "User"),
        Some(&Readiness::Ready)
    );
    assert!(matches!(
        readiness.of(CatalogObject::Label, "Org"),
        Some(Readiness::Failed(_))
    ));
    let schemas = db.label_schemas().unwrap();
    let mode = |name: &str| schemas.iter().find(|s| s.name == name).unwrap().mode;
    assert_eq!(mode("User"), SchemaMode::Strict);
    assert_eq!(mode("Org"), SchemaMode::Flexible);
}

/// A STRICT `User` declaring only `email`.
fn strict_user() -> SchemaPackage {
    let mut schema = LabelSchema::new_node_id("User");
    schema.set_mode(SchemaMode::Strict);
    schema.add_property(PropertyDef::new("email", PropertyType::String));
    SchemaPackage {
        labels: vec![LabelDeclaration {
            schema,
            intent: Intent::Ensure,
        }],
        ..SchemaPackage::default()
    }
}

fn user_schema(db: &Database) -> LabelSchema {
    db.label_schemas()
        .unwrap()
        .into_iter()
        .find(|s| s.name == "User")
        .expect("User")
}

/// A definition that could refuse stored nodes is published as a candidate
/// and proved in the background; stored nodes satisfying it put it in
/// force, after which it holds every write. Declaring it again changes
/// nothing.
#[test]
fn a_definition_the_stored_nodes_satisfy_is_put_in_force() {
    let (mut db, _dir) = open();
    db.execute_cypher("CREATE (:User {email: 'a@x'})")
        .expect("seed");
    let published = db.publish_schema(strict_user()).expect("publish");
    assert!(published.committed);
    let label = &published.objects[0];
    assert_eq!(label.disposition, Disposition::Created);
    assert!(label.operation.is_some(), "{label:?}");

    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(readiness.all_ready(), "{readiness:?}");
    let schema = user_schema(&db);
    assert_eq!(schema.mode, SchemaMode::Strict);
    assert!(schema.candidate().is_none());
    assert!(
        db.execute_cypher("CREATE (:User {email: 'b@x', nick: 'b'})")
            .is_err(),
        "STRICT is in force: an undeclared property is refused"
    );

    let again = db.publish_schema(strict_user()).expect("again");
    assert!(!again.committed);
    assert_eq!(again.objects[0].disposition, Disposition::Retained);
    assert_eq!(again.objects[0].operation, None);
}

/// A definition stored nodes break is withdrawn by its proof: the rules in
/// force before stay, and declaring it again is a conflict, not a retry
/// that recreates the withdrawn candidate.
#[test]
fn a_definition_stored_nodes_break_is_withdrawn() {
    let (mut db, _dir) = open();
    db.execute_cypher("CREATE (:User {email: 'a@x', nick: 'a'})")
        .expect("seed");
    let published = db.publish_schema(strict_user()).expect("publish");
    assert!(published.committed, "acceptance is not readiness");

    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(
        matches!(
            readiness.of(CatalogObject::Label, "User"),
            Some(Readiness::Failed(why)) if why.contains("breaks it")
        ),
        "{readiness:?}"
    );
    let schema = user_schema(&db);
    assert_eq!(schema.mode, SchemaMode::Flexible, "the rules before stay");
    assert!(schema.candidate().is_none());
    db.execute_cypher("CREATE (:User {nick: 'b'})")
        .expect("nothing more than before holds writes");

    assert!(matches!(
        db.publish_schema(strict_user()).unwrap_err(),
        DatabaseError::CatalogObjectDiffers {
            object: CatalogObject::Label,
            ..
        }
    ));
}

/// Keep the next certification waiting: the executor runs one job at a
/// time, and the build of a uniqueness on another label takes that seat and
/// waits for a transaction the caller opened before calling, until it ends.
fn occupy_the_only_seat(db: &Database) {
    db.set_index_build_config(coordinode_query::index::IndexBuildConfig {
        max_running: 1,
        ..db.index_build_config()
    });
    let blocker = SchemaPackage {
        constraints: vec![PackageConstraint {
            name: Some("org_name".into()),
            label: "Org".into(),
            properties: vec!["name".into()],
            kind: ConstraintKind::Unique,
            scope: None,
            on_duplicate_rename: None,
        }],
        ..SchemaPackage::default()
    };
    db.publish_schema(blocker).expect("the blocking build");
    // Seated and waiting before the caller publishes what must queue behind
    // it: a build accepted but not yet seated could lose the seat to it.
    let deadline = std::time::Instant::now() + WAIT;
    while !db
        .index_build_status()
        .expect("builds")
        .iter()
        .any(|s| s.phase == Some(coordinode_query::index::BuildPhase::AwaitingOlderTransactions))
    {
        assert!(
            std::time::Instant::now() < deadline,
            "the blocking build never took the seat"
        );
        std::thread::sleep(Duration::from_millis(5));
    }
}

/// While a candidate is proved, a write is held to it as well as to the
/// rules in force: the proof scans what was stored before, never what is
/// written after. A conforming write lands, and the proof puts the
/// candidate in force once it runs.
#[test]
fn a_write_during_certification_is_held_to_the_candidate() {
    let (db, _dir) = open();
    let older = db.begin_catalog_txn();
    occupy_the_only_seat(&db);
    let published = db.publish_schema(strict_user()).expect("publish");
    let pending = db.await_schema(&published, Duration::ZERO).expect("look");
    assert_eq!(
        pending.of(CatalogObject::Label, "User"),
        Some(&Readiness::Pending)
    );
    assert_eq!(user_schema(&db).mode, SchemaMode::Flexible);

    let refused = db
        .execute_cypher_shared(
            "CREATE (:User {email: 'a@x', nick: 'a'})",
            None,
            None,
            None,
            None,
        )
        .expect_err("the candidate refuses an undeclared property");
    assert!(refused.to_string().contains("being certified"), "{refused}");
    db.execute_cypher_shared("CREATE (:User {email: 'b@x'})", None, None, None, None)
        .expect("a write the candidate admits");

    drop(older);
    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(readiness.all_ready(), "{readiness:?}");
    assert_eq!(user_schema(&db).mode, SchemaMode::Strict);
}

/// A certification the closing process never finished is taken up by the
/// next opening, which puts the candidate in force: acceptance survives the
/// restart as the same operation, never a second candidate.
#[test]
fn an_interrupted_certification_finishes_after_a_restart() {
    let dir = tempfile::tempdir().expect("tempdir");
    let operation = {
        let db = Database::open(dir.path()).expect("open");
        // Never ended: the blocking build, and the proof queued behind it,
        // wait until the process closes.
        db.begin_transaction().expect("begin");
        occupy_the_only_seat(&db);
        let published = db.publish_schema(strict_user()).expect("publish");
        let pending = db.await_schema(&published, Duration::ZERO).expect("look");
        assert_eq!(
            pending.of(CatalogObject::Label, "User"),
            Some(&Readiness::Pending)
        );
        published.objects[0].operation.expect("operation")
    };

    let db = Database::open(dir.path()).expect("reopen");
    let schema = user_schema(&db);
    assert_eq!(schema.mode, SchemaMode::Strict, "put in force on reopen");
    assert!(schema.candidate().is_none());
    let again = db.publish_schema(strict_user()).expect("again");
    assert!(!again.committed);
    assert!(
        matches!(
            db.index_builds
                .wait(operation, Some(Duration::ZERO))
                .unwrap(),
            Some(coordinode_query::index::IndexBuildOutcome::Published { .. })
        ),
        "the record carries the outcome of the same operation"
    );
}

/// A publication under a client operation takes effect once: retrying it
/// after a lost reply answers with the operation's outcome instead of a
/// revision conflict, another package under the same number is told
/// apart, and an unchanged package records nothing under its operation.
#[test]
fn a_publication_retried_under_its_operation_answers_with_its_outcome() {
    use crate::operation::{IssuerId, OperationOutcome, Resolve};
    let (db, _dir) = open();
    let first = db.publish_schema(package()).expect("publish");
    db.await_schema(&first, WAIT).expect("await");
    let issuer = db.enroll_issuer(IssuerId([0x41; 16])).expect("enroll");

    let mut change = package();
    change.labels[0].schema = user(Some("age"));
    change.labels[0].intent = Intent::Change {
        expected_revision: user_schema(&db).schema_revision,
    };
    change.constraints.clear();
    change.edge_types.clear();
    let descriptor = issuer.next().expect("descriptor");
    let changed = db
        .publish_schema_as(change.clone(), &descriptor)
        .expect("change");
    assert!(changed.committed);

    // The reply was lost: the same call again changes nothing and says how
    // the first one ended.
    match db.publish_schema_as(change.clone(), &descriptor) {
        Err(DatabaseError::OperationSettled {
            outcome,
            intent_differs,
            ..
        }) => {
            assert!(
                matches!(outcome, OperationOutcome::Committed { .. }),
                "{outcome:?}"
            );
            assert!(!intent_differs);
        }
        other => panic!("expected the settled operation, got {other:?}"),
    }
    let mut other = change;
    other.labels[0].schema = user(Some("level"));
    other.labels[0].intent = Intent::Change {
        expected_revision: changed.objects[0].revision,
    };
    assert!(matches!(
        db.publish_schema_as(other, &descriptor),
        Err(DatabaseError::OperationSettled {
            intent_differs: true,
            ..
        })
    ));

    let unchanged = issuer.next().expect("descriptor");
    let mut same = package();
    same.labels[0].schema = user(Some("age"));
    assert!(
        !db.publish_schema_as(same, &unchanged)
            .expect("no-op")
            .committed
    );
    assert_eq!(
        db.resolve_operation(&unchanged, Resolve::Lookup)
            .expect("resolve"),
        OperationOutcome::Unknown,
        "nothing was committed under the operation"
    );
}

/// Equal packages published at once converge on one set of objects: each
/// call either publishes, keeps what the other published, or meets a
/// retriable conflict; the catalog ends with one label, one constraint
/// and one build for it, never a hidden duplicate. A different package
/// published at once over the same names is refused, never merged.
#[test]
fn concurrent_publications_converge_or_conflict() {
    let (db, _dir) = open();
    let db = std::sync::Arc::new(db);
    let outcomes: Vec<_> = (0..4)
        .map(|i| {
            let db = std::sync::Arc::clone(&db);
            std::thread::spawn(move || {
                let mut package = package();
                if i == 3 {
                    package.labels[0].schema = user(Some("age"));
                }
                db.publish_schema(package)
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|t| t.join().expect("thread"))
        .collect();
    for (i, outcome) in outcomes.iter().enumerate() {
        match outcome {
            Ok(_) => {}
            Err(
                DatabaseError::TransactionConflict { .. }
                | DatabaseError::InvariantRefused { .. }
                | DatabaseError::CatalogObjectDiffers { .. },
            ) => {}
            Err(other) => panic!("publication {i}: {other:?}"),
        }
    }
    let settled = db.publish_schema(package());
    let settled = match settled {
        Ok(p) => p,
        // The differing package won: the equal one now differs from it.
        Err(DatabaseError::CatalogObjectDiffers { .. }) => {
            let mut other = package();
            other.labels[0].schema = user(Some("age"));
            db.publish_schema(other)
                .expect("the winner's package is in place")
        }
        Err(e) => panic!("{e:?}"),
    };
    db.await_schema(&settled, WAIT).expect("await");
    assert_eq!(db.label_schemas().unwrap().len(), 1);
    let constraints = db.constraints().unwrap();
    let names: Vec<_> = constraints
        .iter()
        .map(|c| c.constraint.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec!["user_email", "user_handle"],
        "no duplicate constraint"
    );
    let builds: Vec<_> = db
        .index_build_status()
        .unwrap()
        .into_iter()
        // Every store keeps its built-in label lookup besides.
        .filter(|b| !b.index.as_ref().is_some_and(|i| i.builtin))
        .collect();
    assert_eq!(builds.len(), 2, "one build per constraint: {builds:?}");
}

/// A change of a label whose candidate is still being proved is refused,
/// naming the operation to wait for or cancel; cancelling withdraws it.
#[test]
fn a_label_being_certified_cannot_be_changed_until_it_ends() {
    let (db, _dir) = open();
    let older = db.begin_catalog_txn();
    occupy_the_only_seat(&db);
    let published = db.publish_schema(strict_user()).expect("publish");
    let operation = published.objects[0].operation.expect("operation");

    let mut change = strict_user();
    change.labels[0]
        .schema
        .add_property(PropertyDef::new("age", PropertyType::Int));
    change.labels[0].intent = Intent::Change {
        expected_revision: published.objects[0].revision,
    };
    assert!(matches!(
        db.publish_schema(change).unwrap_err(),
        DatabaseError::CatalogRefused(ref why) if why.contains("still being certified")
    ));

    assert!(db.cancel_index_build(operation).expect("cancel"));
    drop(older);
    let readiness = db.await_schema(&published, WAIT).expect("await");
    assert!(matches!(
        readiness.of(CatalogObject::Label, "User"),
        Some(Readiness::Failed(_))
    ));
    let schema = user_schema(&db);
    assert!(schema.candidate().is_none());
    assert_eq!(schema.mode, SchemaMode::Flexible);
}
