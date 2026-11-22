use std::sync::Arc;

use coordinode_core::access::MissingRight;
use coordinode_core::txn::command::Principal;

use super::*;
use crate::db::StatementOptions;

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn id(subject: &str) -> PrincipalId {
    PrincipalId::new(subject, "idp").expect("a valid principal")
}

/// `subject` acting in the default namespace.
fn acting(subject: &str) -> ActingContext {
    ActingContext::Principal(Principal {
        subject: Arc::from(subject),
        trust_scope: Arc::from("idp"),
        namespace: Arc::from("default"),
    })
}

fn grant(namespace: &str, subject: &str, role: NamespaceRole) -> AccessChange {
    AccessChange::Grant {
        namespace: namespace.into(),
        grantee: GranteeName::Principal(id(subject)),
        role,
    }
}

/// `query` run as `subject`.
fn run_as(db: &Database, subject: &str, query: &str) -> Result<(), DatabaseError> {
    db.execute_cypher_shared_with(
        query,
        None,
        None,
        &StatementOptions {
            actor: Some(acting(subject)),
            ..StatementOptions::default()
        },
    )
    .map(drop)
}

fn refused_for(result: Result<(), DatabaseError>, subject: &str, missing: MissingRight) {
    match result {
        Err(DatabaseError::AccessRefused(refusal)) => {
            assert_eq!(refusal.principal, id(subject));
            assert_eq!(refusal.missing, missing);
        }
        other => panic!("expected an access refusal of {subject}, got {other:?}"),
    }
}

fn default_namespace(db: &Database) -> coordinode_core::namespace::NamespaceId {
    coordinode_storage::engine::metadata::namespace_by_name(db.engine.deployment(), "default")
        .expect("catalog")
        .expect("the default namespace")
}

/// A principal without admin on a namespace cannot change its grants: the
/// change is refused where the log applies it, naming the admin right it
/// lacked, and nothing changes. The host grants it admin; from then on it
/// grants others.
#[test]
fn only_a_namespace_admin_changes_its_grants() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);

    refused_for(
        db.change_access(
            &acting("alice"),
            grant("default", "bob", NamespaceRole::Writer),
        ),
        "alice",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Admin,
        },
    );
    refused_for(
        run_as(&db, "bob", "CREATE (:Doc)"),
        "bob",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Writer,
        },
    );

    db.change_access(
        &ActingContext::Trusted,
        grant("default", "alice", NamespaceRole::Admin),
    )
    .expect("the host grants admin");
    db.change_access(
        &acting("alice"),
        grant("default", "bob", NamespaceRole::Writer),
    )
    .expect("the admin grants writer");
    run_as(&db, "bob", "CREATE (:Doc)").expect("the writer writes");
}

/// A reader reads and is refused a write; a writer is refused a schema
/// change, which needs admin.
#[test]
fn each_role_grants_what_it_names_and_no_more() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "rita", NamespaceRole::Reader),
    )
    .expect("grant reader");
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "will", NamespaceRole::Writer),
    )
    .expect("grant writer");

    run_as(&db, "rita", "MATCH (n) RETURN n").expect("the reader reads");
    refused_for(
        run_as(&db, "rita", "CREATE (:Doc)"),
        "rita",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Writer,
        },
    );
    run_as(&db, "will", "CREATE (:Doc)").expect("the writer writes");
    refused_for(
        run_as(&db, "will", "CREATE INDEX doc_n ON :Doc(n)"),
        "will",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Admin,
        },
    );
}

/// A revocation takes effect on a session that is already open: the write
/// after it is refused, the one before it stands.
#[test]
fn a_revocation_refuses_the_next_write_of_an_open_session() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "bob", NamespaceRole::Writer),
    )
    .expect("grant");
    run_as(&db, "bob", "CREATE (:Doc {n: 1})").expect("the write before");

    db.change_access(
        &ActingContext::Trusted,
        AccessChange::Revoke {
            namespaces: vec!["default".into()],
            grantee: GranteeName::Principal(id("bob")),
        },
    )
    .expect("revoke");
    refused_for(
        run_as(&db, "bob", "CREATE (:Doc {n: 2})"),
        "bob",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Writer,
        },
    );
    let rows = db
        .execute_cypher_shared("MATCH (d:Doc) RETURN d.n", None, None, None, None)
        .expect("count")
        .rows;
    assert_eq!(rows.len(), 1, "only the write before the revocation stands");
}

/// A transfer moves the right only while its source holds it: the first
/// moves it, a repeat from the old holder writes nothing, and only the new
/// holder writes from then on.
#[test]
fn a_transfer_moves_the_right_once() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "p", NamespaceRole::Writer),
    )
    .expect("grant");
    let transfer = || AccessChange::Transfer {
        namespace: "default".into(),
        role: NamespaceRole::Writer,
        from: id("p"),
        to: id("q"),
    };
    db.change_access(&ActingContext::Trusted, transfer())
        .expect("the first transfer");
    db.change_access(&ActingContext::Trusted, transfer())
        .expect("a repeat is not an error: it writes nothing");

    run_as(&db, "q", "CREATE (:Doc)").expect("the new holder writes");
    refused_for(
        run_as(&db, "p", "CREATE (:Doc)"),
        "p",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Writer,
        },
    );
}

/// A namespace statement acts for its principal: creating needs the right to
/// create namespaces, and the creator is admin of what it created, so it
/// drops it; a writer of a namespace does not drop it.
#[test]
fn a_namespace_statement_acts_for_its_principal() {
    let (db, _dir) = open();
    refused_for(
        run_as(&db, "carol", "CREATE NAMESPACE carols"),
        "carol",
        MissingRight::CreateNamespace,
    );
    assert!(
        db.namespaces()
            .expect("namespaces")
            .iter()
            .all(|n| n.name != "carols"),
        "the refused creation created nothing"
    );

    db.change_access(
        &ActingContext::Trusted,
        AccessChange::DeploymentRights {
            principal: id("carol"),
            create_namespace: Some(true),
            admin: None,
        },
    )
    .expect("grant the right to create namespaces");
    run_as(&db, "carol", "CREATE NAMESPACE carols").expect("carol creates");

    db.change_access(
        &ActingContext::Trusted,
        grant("carols", "dave", NamespaceRole::Writer),
    )
    .expect("grant dave writer on carols");
    let carols =
        coordinode_storage::engine::metadata::namespace_by_name(db.engine.deployment(), "carols")
            .expect("catalog")
            .expect("created");
    refused_for(
        run_as(&db, "dave", "DROP NAMESPACE carols"),
        "dave",
        MissingRight::Namespace {
            namespace: carols,
            role: NamespaceRole::Admin,
        },
    );
    run_as(&db, "carol", "DROP NAMESPACE carols").expect("the creator, its admin, drops it");
}

/// A principal bound to another namespace than the handle's changes
/// nothing through it.
#[test]
fn a_principal_of_another_namespace_changes_nothing_here() {
    let (db, _dir) = open();
    let elsewhere = ActingContext::Principal(Principal {
        subject: Arc::from("alice"),
        trust_scope: Arc::from("idp"),
        namespace: Arc::from("other"),
    });
    let result = db.change_access(&elsewhere, grant("default", "bob", NamespaceRole::Reader));
    assert!(
        matches!(result, Err(DatabaseError::NamespaceMismatch { .. })),
        "{result:?}"
    );
}
