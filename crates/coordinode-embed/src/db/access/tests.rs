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

/// A statement on the deployment's catalog is decided by the right it needs,
/// not by a right on the session's namespace: a principal holding only the
/// right to create namespaces creates and lists them, and a principal that is
/// only in an exclusive role's eligible set takes the role. A reader of the
/// session's namespace lists nothing, and a principal without any role on a
/// namespace is refused alike whether it exists or not.
#[test]
fn a_catalog_statement_needs_its_own_right_and_no_other() {
    let (db, _dir) = open();
    db.change_access(
        &ActingContext::Trusted,
        AccessChange::DeploymentRights {
            principal: id("carol"),
            create_namespace: Some(true),
            admin: None,
        },
    )
    .expect("grant the right to create namespaces");
    run_as(&db, "carol", "CREATE NAMESPACE carols").expect("carol creates, with no data rights");
    run_as(&db, "carol", "SHOW NAMESPACES").expect("and lists the catalog");
    refused_for(
        run_as(&db, "carol", "MATCH (n) RETURN n"),
        "carol",
        MissingRight::Namespace {
            namespace: default_namespace(&db),
            role: NamespaceRole::Reader,
        },
    );

    db.change_access(
        &ActingContext::Trusted,
        grant("default", "rita", NamespaceRole::Reader),
    )
    .expect("grant reader");
    refused_for(
        run_as(&db, "rita", "SHOW NAMESPACES"),
        "rita",
        MissingRight::CreateNamespace,
    );
    // Rita holds no role on carols: an existing namespace and a missing one
    // are refused the same.
    for name in ["carols", "nowhere"] {
        refused_for(
            run_as(
                &db,
                "rita",
                &format!("ALTER NAMESPACE {name} SET STORAGE (compression: 'lz4')"),
            ),
            "rita",
            MissingRight::DeploymentAdmin,
        );
        refused_for(
            run_as(&db, "rita", &format!("DROP NAMESPACE {name}")),
            "rita",
            MissingRight::DeploymentAdmin,
        );
    }

    db.execute_cypher_shared(
        "CREATE ROLE coordinator IN NAMESPACE default \
         EXCLUSIVE AMONG (PRINCIPAL 'm1' IN 'idp', PRINCIPAL 'm2' IN 'idp') \
         HELD BY PRINCIPAL 'm1' IN 'idp'",
        None,
        None,
        None,
        None,
    )
    .expect("the host creates the role");
    run_as(
        &db,
        "m2",
        "TRANSFER ROLE coordinator IN NAMESPACE default \
         FROM PRINCIPAL 'm1' IN 'idp' TO PRINCIPAL 'm2' IN 'idp'",
    )
    .expect("an eligible principal takes the role with no other right");
    let role = db
        .role(&ActingContext::Trusted, "default", "coordinator")
        .expect("read")
        .expect("the role");
    assert_eq!(role.holders, vec![id("m2")]);
}

/// A role's eligible set is its home namespace's admin's alone: the admin of
/// another namespace the role holds a grant on has no say over who may hold
/// it, and the set stays as it was. The home's admin changes it.
#[test]
fn only_the_homes_admin_changes_a_roles_eligible_set() {
    let (db, _dir) = open();
    db.create_namespace("shared").expect("a second namespace");
    let host = |query: &str| {
        db.execute_cypher_shared(query, None, None, None, None)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    };
    host(
        "CREATE ROLE coordinator IN NAMESPACE default \
         EXCLUSIVE AMONG (PRINCIPAL 'm1' IN 'idp', PRINCIPAL 'm2' IN 'idp') \
         HELD BY PRINCIPAL 'm1' IN 'idp'",
    );
    host("GRANT writer ON NAMESPACE shared TO ROLE coordinator IN NAMESPACE default");
    db.change_access(
        &ActingContext::Trusted,
        grant("shared", "sam", NamespaceRole::Admin),
    )
    .expect("sam administers shared");
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "hana", NamespaceRole::Admin),
    )
    .expect("hana administers the role's home");

    let set_m2 =
        "ALTER ROLE coordinator IN NAMESPACE default SET ELIGIBLE (PRINCIPAL 'm2' IN 'idp')";
    refused_for(
        run_as(&db, "sam", set_m2),
        "sam",
        MissingRight::Namespace {
            namespace: default_namespace(&db),
            role: NamespaceRole::Admin,
        },
    );
    let role = |db: &Database| {
        db.role(&ActingContext::Trusted, "default", "coordinator")
            .expect("read")
            .expect("the role")
    };
    assert_eq!(role(&db).eligible, Some(vec![id("m1"), id("m2")]));

    run_as(&db, "hana", set_m2).expect("the home's admin changes the set");
    let after = role(&db);
    assert_eq!(after.eligible, Some(vec![id("m2")]));
    assert!(after.holders.is_empty(), "m1 left the set and the role");
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

/// A principal of an exclusive role's eligible set reads the role's holder
/// and the timestamp it last changed at with no right on any namespace's
/// data; after a transfer it reads the new holder. A principal neither
/// eligible, nor holding it, nor its home's admin is refused, and a role
/// that does not exist is refused the same way.
#[test]
fn an_eligible_principal_reads_its_roles_holder_without_data_rights() {
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.change_access(
        &ActingContext::Trusted,
        AccessChange::CreateRole {
            home: "default".into(),
            name: "coordinator".into(),
            eligible: Some(vec![id("m1"), id("m2")]),
            holder: Some(id("m1")),
        },
    )
    .expect("create the role");

    let seen = db
        .role(&acting("m2"), "default", "coordinator")
        .expect("an eligible principal reads it")
        .expect("the role exists");
    assert_eq!(seen.holders, vec![id("m1")]);
    let before = seen.changed_ts;

    db.change_access(
        &acting("m2"),
        AccessChange::TransferRole {
            home: "default".into(),
            name: "coordinator".into(),
            from: id("m1"),
            to: id("m2"),
        },
    )
    .expect("an eligible principal takes the role over");
    let after = db
        .role(&acting("m1"), "default", "coordinator")
        .expect("the former holder, still eligible, reads it")
        .expect("the role exists");
    assert_eq!(after.holders, vec![id("m2")]);
    assert!(
        after.changed_ts > before,
        "the transfer moved its timestamp"
    );

    for (subject, role) in [("outsider", "coordinator"), ("outsider", "nothing")] {
        match db.role(&acting(subject), "default", role) {
            Err(DatabaseError::AccessRefused(refusal)) => assert_eq!(
                refusal.missing,
                MissingRight::Namespace {
                    namespace: ns,
                    role: NamespaceRole::Admin,
                },
                "{role}"
            ),
            other => panic!("{subject} on {role}: expected a refusal, got {other:?}"),
        }
    }
}

/// The grants on a namespace are read by a principal holding a right
/// there, each with the timestamp of the change that made it; a principal
/// without one is refused, and a namespace that does not exist is refused
/// to it without saying so.
#[test]
fn the_grants_on_a_namespace_are_read_by_its_right_holders() {
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

    let grants = db
        .namespace_grants(&acting("rita"), "default")
        .expect("a reader reads the grants");
    let mut seen: Vec<_> = grants.iter().map(|g| (g.grantee.clone(), g.role)).collect();
    seen.sort_by_key(|(_, role)| *role);
    assert_eq!(
        seen,
        vec![
            (GranteeName::Principal(id("rita")), NamespaceRole::Reader),
            (GranteeName::Principal(id("will")), NamespaceRole::Writer),
        ]
    );
    assert!(grants.iter().all(|g| g.changed_ts > 0));

    refused_for(
        db.namespace_grants(&acting("eve"), "default").map(drop),
        "eve",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Reader,
        },
    );
    refused_for(
        db.namespace_grants(&acting("rita"), "nowhere").map(drop),
        "rita",
        MissingRight::DeploymentAdmin,
    );
}

/// Whether a whole-deployment admin is issued is read by a principal
/// holding a right on a namespace it reaches: none at first, then the one
/// granted, with the timestamp of the grant.
#[test]
fn a_right_holder_reads_whether_a_deployment_admin_is_issued() {
    let (db, _dir) = open();
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "rita", NamespaceRole::Reader),
    )
    .expect("grant reader");
    assert!(
        db.deployment_admins(&acting("rita"), "default")
            .expect("read")
            .is_empty(),
        "no deployment admin is issued"
    );

    db.change_access(
        &ActingContext::Trusted,
        AccessChange::DeploymentRights {
            principal: id("root"),
            create_namespace: None,
            admin: Some(true),
        },
    )
    .expect("issue a deployment admin");
    let admins = db
        .deployment_admins(&acting("rita"), "default")
        .expect("read");
    assert_eq!(admins.len(), 1);
    assert_eq!(admins[0].principal, id("root"));
    assert!(admins[0].changed_ts > 0);

    assert!(matches!(
        db.deployment_admins(&acting("eve"), "default"),
        Err(DatabaseError::AccessRefused(_))
    ));
}

/// A principal holding only the right to create namespaces, with no session
/// on any namespace's data, creates one in catalog scope, becomes its admin
/// at once, lists the deployment's namespaces and drops what it created. It
/// does not drop a namespace it has no admin on, and a principal without the
/// right creates nothing and lists nothing; neither learns from the refusal
/// whether a name is taken.
#[test]
fn a_creator_works_in_catalog_scope_without_a_data_session() {
    use coordinode_storage::engine::access::effective_role;
    let (db, _dir) = open();
    db.change_access(
        &ActingContext::Trusted,
        AccessChange::DeploymentRights {
            principal: id("carol"),
            create_namespace: Some(true),
            admin: None,
        },
    )
    .expect("grant the right to create namespaces");
    db.create_namespace("theirs").expect("the host creates one");

    let carol = db.catalog_scope(id("carol"));
    carol.create_namespace("carols").expect("carol creates");
    let deployment = db.engine.deployment();
    let carols = coordinode_storage::engine::metadata::namespace_by_name(deployment, "carols")
        .unwrap()
        .unwrap();
    assert_eq!(
        effective_role(deployment, &id("carol"), carols).unwrap(),
        Some(NamespaceRole::Admin),
        "the creator is admin of what it created"
    );
    let names: Vec<_> = carol
        .namespaces()
        .expect("carol lists")
        .into_iter()
        .map(|n| n.name)
        .collect();
    assert!(names.contains(&"carols".to_owned()) && names.contains(&"theirs".to_owned()));

    // Carol holds no role on theirs: refused as for a missing name.
    for name in ["theirs", "nowhere"] {
        refused_for(
            carol.drop_namespace(name),
            "carol",
            MissingRight::DeploymentAdmin,
        );
    }
    carol
        .drop_namespace("carols")
        .expect("carol drops what it created");

    let eve = db.catalog_scope(id("eve"));
    // A taken name and a free one are refused alike.
    for name in ["theirs", "free"] {
        refused_for(
            eve.create_namespace(name),
            "eve",
            MissingRight::CreateNamespace,
        );
    }
    refused_for(
        eve.namespaces().map(drop),
        "eve",
        MissingRight::CreateNamespace,
    );
}

/// A repairing build that loses its issuer's rights before it finishes makes
/// no further repair: the build is left as an interrupted statement leaves it,
/// the issuer's admin is revoked, and the build resumed on the next process
/// renames no node, since every repair is applied in the issuer's name.
#[test]
fn a_repairing_build_whose_issuer_lost_the_grant_repairs_nothing() {
    use coordinode_modality::{
        BuildFailure, DuplicateRepair, IndexBuildRecord, IndexDescriptor, IndexState,
        IndexStore as _, LocalIndexStore,
    };
    let (db, _dir) = open();
    db.execute_cypher_shared(
        "CREATE (:User {email: 'same@x'}), (:User {email: 'same@x'})",
        None,
        None,
        None,
        None,
    )
    .expect("the host writes two users of one email");
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "ada", NamespaceRole::Admin),
    )
    .expect("grant admin");

    // ada's build, admitted and not yet run, as a statement cut off by a
    // stop leaves it.
    let store = LocalIndexStore::new(&db.engine);
    let mut descriptor =
        IndexDescriptor::compound("user_email", "User", vec!["email".into()]).unique();
    descriptor.state = IndexState::Building {
        written: 0,
        estimated_total: 0,
    };
    let mut txn = coordinode_storage::engine::transaction::Transaction::new(
        &db.engine,
        None,
        coordinode_core::txn::timestamp::Timestamp::ZERO,
        None,
    );
    let def = store
        .publish_definition_txn(&mut txn, descriptor)
        .expect("publish index");
    store
        .put_build_txn(
            &mut txn,
            &IndexBuildRecord::accepted(def.id, def.generation, BuildFailure::Withdraw).repairing(
                Some(DuplicateRepair {
                    property: "email".into(),
                    issuer: Some(id("ada")),
                }),
            ),
            None,
        )
        .expect("admit build");
    drop(txn);

    db.change_access(
        &ActingContext::Trusted,
        AccessChange::Revoke {
            namespaces: vec!["default".into()],
            grantee: GranteeName::Principal(id("ada")),
        },
    )
    .expect("revoke");
    // The build ends however its repair is refused; what matters is that
    // nothing was renamed in ada's name.
    let _ = db.resume_interrupted_index_builds();

    let emails: Vec<_> = db
        .execute_cypher_shared("MATCH (u:User) RETURN u.email AS e", None, None, None, None)
        .expect("read as the host")
        .rows
        .into_iter()
        .map(|row| row.get("e").cloned())
        .collect();
    assert_eq!(
        emails,
        vec![
            Some(coordinode_core::graph::types::Value::String(
                "same@x".into()
            ));
            2
        ],
        "no repair was applied after the revocation"
    );
    let repaired: u64 = store
        .list_builds()
        .unwrap()
        .into_iter()
        .filter(|build| build.generation == def.generation)
        .map(|build| build.repaired)
        .sum();
    assert_eq!(repaired, 0);
}

/// A unique index built with ON DUPLICATE RENAME by a principal records that
/// principal on the build, so every repair, here and on a member resuming
/// the build, runs in its name; a principal without admin is refused before
/// any build is accepted.
#[test]
fn a_repairing_build_records_the_principal_that_asked_for_it() {
    use coordinode_modality::{IndexStore as _, LocalIndexStore};
    let (db, _dir) = open();
    let ns = default_namespace(&db);
    db.execute_cypher_shared(
        "CREATE (:User {email: 'same@x'}), (:User {email: 'same@x'})",
        None,
        None,
        None,
        None,
    )
    .expect("the host writes two users of one email");
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "will", NamespaceRole::Writer),
    )
    .expect("grant writer");
    refused_for(
        run_as(
            &db,
            "will",
            "CREATE UNIQUE INDEX user_email ON :User(email) ON DUPLICATE RENAME email",
        ),
        "will",
        MissingRight::Namespace {
            namespace: ns,
            role: NamespaceRole::Admin,
        },
    );
    let store = LocalIndexStore::new(&db.engine);
    // The store's own builds of its built-in indexes are there from the
    // start; the refused statement added none.
    let accepted = store.list_builds().unwrap();
    assert!(
        accepted.iter().all(|build| build.on_duplicate.is_none()),
        "no repairing build was accepted: {accepted:?}"
    );

    db.change_access(
        &ActingContext::Trusted,
        grant("default", "ada", NamespaceRole::Admin),
    )
    .expect("grant admin");
    run_as(
        &db,
        "ada",
        "CREATE UNIQUE INDEX user_email ON :User(email) ON DUPLICATE RENAME email",
    )
    .expect("the admin builds, repairing the duplicate");
    let repairing: Vec<_> = store
        .list_builds()
        .unwrap()
        .into_iter()
        .filter(|build| build.on_duplicate.is_some())
        .collect();
    assert_eq!(repairing.len(), 1, "{repairing:?}");
    assert_eq!(
        repairing[0]
            .on_duplicate
            .as_ref()
            .and_then(|repair| repair.issuer.clone()),
        Some(id("ada"))
    );
    assert_eq!(repairing[0].repaired, 1, "the repair ran in ada's name");
}

/// A catalog change and a data write in one transaction commit in one log
/// entry, at one timestamp: removing a principal from a role's eligible set
/// and deleting what only it could read is one act. When the change is
/// refused, the data write is refused with it.
#[test]
fn a_catalog_change_and_a_data_write_commit_as_one() {
    use crate::db::TransactionOptions;
    let (db, _dir) = open();
    let host = |query: &str| {
        db.execute_cypher_shared(query, None, None, None, None)
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    };
    host(
        "CREATE ROLE coordinator IN NAMESPACE default \
         EXCLUSIVE AMONG (PRINCIPAL 'm1' IN 'idp', PRINCIPAL 'm2' IN 'idp') \
         HELD BY PRINCIPAL 'm2' IN 'idp'",
    );
    host("CREATE (:Secret {owner: 'm2'}), (:Kept {owner: 'm2'})");

    let tx = db.begin_transaction().expect("begin");
    db.execute_in_transaction(
        tx,
        "ALTER ROLE coordinator IN NAMESPACE default SET ELIGIBLE (PRINCIPAL 'm1' IN 'idp')",
        None,
    )
    .expect("the change, staged");
    db.execute_in_transaction(tx, "MATCH (s:Secret) DELETE s", None)
        .expect("the deletion, staged");
    let receipt = db.commit_transaction(tx).expect("commit");

    let role = db
        .role(&ActingContext::Trusted, "default", "coordinator")
        .expect("read")
        .expect("the role");
    assert!(role.holders.is_empty(), "m2 left the set and the role");
    assert_eq!(role.eligible, Some(vec![id("m1")]));
    assert_eq!(
        role.changed_ts,
        receipt.commit_ts.as_raw(),
        "the change landed at the transaction's commit"
    );
    let secrets = db
        .execute_cypher_shared("MATCH (s:Secret) RETURN s", None, None, None, None)
        .unwrap()
        .rows;
    assert!(secrets.is_empty(), "the deletion landed with it");

    // A writer without admin: the change is refused, and the deletion with it.
    db.change_access(
        &ActingContext::Trusted,
        grant("default", "eve", NamespaceRole::Writer),
    )
    .expect("grant writer");
    let tx = db
        .begin_transaction_with(&TransactionOptions {
            actor: Some(acting("eve")),
            ..TransactionOptions::trusted()
        })
        .expect("begin");
    let as_eve = StatementOptions {
        actor: Some(acting("eve")),
        ..StatementOptions::default()
    };
    db.execute_in_transaction_with(
        tx,
        "ALTER ROLE coordinator IN NAMESPACE default SET ELIGIBLE ()",
        None,
        &as_eve,
    )
    .expect("staged");
    db.execute_in_transaction_with(tx, "MATCH (k:Kept) DELETE k", None, &as_eve)
        .expect("staged");
    let refused = db.commit_transaction_with(
        tx,
        &crate::db::CommitOptions {
            actor: Some(&acting("eve")),
            ..crate::db::CommitOptions::default()
        },
    );
    assert!(
        matches!(refused, Err(DatabaseError::AccessRefused(_))),
        "{refused:?}"
    );
    let kept = db
        .execute_cypher_shared("MATCH (k:Kept) RETURN k", None, None, None, None)
        .unwrap()
        .rows;
    assert_eq!(kept.len(), 1, "the refused unit deleted nothing");
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
