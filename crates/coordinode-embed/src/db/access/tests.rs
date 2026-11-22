use std::sync::Arc;

use coordinode_core::access::{MissingRight, NamespaceRole};
use coordinode_core::txn::command::Principal;
use coordinode_core::txn::proposal::GranteeName;

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
    // Without any right bob is not admitted to the namespace at all.
    refused_for(
        run_as(&db, "bob", "CREATE (:Doc)"),
        "bob",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
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
    // Every right went, so bob is not admitted to the namespace at all.
    refused_for(
        run_as(&db, "bob", "CREATE (:Doc {n: 2})"),
        "bob",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
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
    // The right left p whole: it holds nothing there any more.
    refused_for(
        run_as(&db, "p", "CREATE (:Doc)"),
        "p",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
        },
    );
}

/// A namespace statement acts for its principal: creating needs the right to
/// create namespaces, and the creator is admin of what it created, so it
/// drops it; a writer of a namespace does not drop it.
#[test]
fn a_namespace_statement_acts_for_its_principal() {
    let (db, _dir) = open();
    // The statements run in a session of the default namespace, which each
    // principal reads; what is checked is the catalog right the statement
    // needs beyond that.
    for subject in ["carol", "dave"] {
        db.change_access(
            &ActingContext::Trusted,
            grant("default", subject, NamespaceRole::Reader),
        )
        .expect("grant reader on the session's namespace");
    }
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

/// A principal with no right on the namespace reads nothing of it: the read
/// is refused before it runs, naming the reader role it lacks.
#[test]
fn a_principal_without_a_right_reads_nothing() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.execute_cypher_shared("CREATE (:Secret {v: 1})", None, None, None, None)
        .expect("the host writes");
    refused_for(
        run_as(&db, "eve", "MATCH (s:Secret) RETURN s.v"),
        "eve",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
        },
    );
}

/// A revocation stops the reads of a transaction already open: the
/// statement after it is refused, whatever the transaction read before.
#[test]
fn a_revocation_stops_the_reads_of_an_open_transaction() {
    use crate::db::TransactionOptions;
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "rita", NamespaceRole::Reader),
    )
    .expect("grant reader");
    let rita = acting("rita");
    let as_rita = StatementOptions {
        actor: Some(rita.clone()),
        ..StatementOptions::default()
    };
    let tx = db
        .begin_transaction_with(&TransactionOptions {
            actor: Some(rita),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    db.execute_in_transaction_with(tx, "MATCH (n) RETURN n", None, &as_rita)
        .expect("the read before the revocation");

    db.change_access(
        &ActingContext::Trusted,
        AccessChange::Revoke {
            namespaces: vec!["default".into()],
            grantee: GranteeName::Principal(id("rita")),
        },
    )
    .expect("revoke");
    refused_for(
        db.execute_in_transaction_with(tx, "MATCH (n) RETURN n", None, &as_rita)
            .map(drop),
        "rita",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
        },
    );
}

/// The grant statements name principals and roles exactly: one subject
/// under two issuers is two principals, a slash in either part is data, and
/// two roles of one name in two homes are two roles. A right granted on a
/// third namespace to the role of one home reaches that role's holder and
/// nobody else.
#[test]
fn grant_statements_name_principals_and_roles_exactly() {
    use coordinode_storage::engine::access::effective_role;
    let (db, _dir) = open();
    let host = |query: &str| {
        db.execute_cypher_shared(query, None, None, None, None)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    };
    for name in ["newsroom", "sports", "shared"] {
        host(&format!("CREATE NAMESPACE {name}"));
    }
    host("CREATE ROLE publisher IN NAMESPACE newsroom");
    host("CREATE ROLE publisher IN NAMESPACE sports");
    host("GRANT writer ON NAMESPACE shared TO ROLE publisher IN NAMESPACE newsroom");
    host(
        "GRANT ROLE publisher IN NAMESPACE newsroom \
         TO PRINCIPAL 'editor/1' IN 'https://a.example/team'",
    );
    host(
        "GRANT ROLE publisher IN NAMESPACE sports \
         TO PRINCIPAL 'editor/1' IN 'https://b.example/team'",
    );

    let deployment = db.engine.deployment();
    let shared = coordinode_storage::engine::metadata::namespace_by_name(deployment, "shared")
        .unwrap()
        .unwrap();
    let of_a = PrincipalId::new("editor/1", "https://a.example/team").unwrap();
    let of_b = PrincipalId::new("editor/1", "https://b.example/team").unwrap();
    assert_eq!(
        effective_role(deployment, &of_a, shared).unwrap(),
        Some(NamespaceRole::Writer),
        "the holder of newsroom's publisher writes shared"
    );
    assert_eq!(
        effective_role(deployment, &of_b, shared).unwrap(),
        None,
        "the same subject of another issuer, holding sports' publisher, has nothing there"
    );

    // A statement acts for its principal: an admin of the default namespace
    // grants on it, a principal without admin there is refused.
    host("GRANT admin ON NAMESPACE default TO PRINCIPAL 'alice' IN 'idp'");
    run_as(
        &db,
        "alice",
        "GRANT reader ON NAMESPACE default TO PRINCIPAL 'bob' IN 'idp'",
    )
    .expect("the admin grants by statement");
    let ns = default_namespace(&db);
    assert_eq!(
        effective_role(deployment, &id("bob"), ns).unwrap(),
        Some(NamespaceRole::Reader)
    );
    refused_for(
        run_as(
            &db,
            "bob",
            "GRANT writer ON NAMESPACE default TO PRINCIPAL 'bob' IN 'idp'",
        ),
        "bob",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Admin,
        },
    );
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
