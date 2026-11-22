use std::sync::Arc;

use coordinode_core::access::{AccessRefusal, MissingRight, NamespaceRole, PrincipalId};
use coordinode_core::txn::command::{ActingContext, Principal};
use coordinode_core::txn::operation::{IssuerId, OperationOutcome};
use coordinode_core::txn::proposal::{AccessChange, GranteeName};

use crate::Database;
use crate::db::{DatabaseError, StatementOptions};
use crate::operation::Resolve;

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn id(subject: &str) -> PrincipalId {
    PrincipalId::new(subject, "idp").expect("a valid principal")
}

fn acting(subject: &str) -> ActingContext {
    ActingContext::Principal(Principal {
        subject: Arc::from(subject),
        trust_scope: Arc::from("idp"),
        namespace: Arc::from("default"),
    })
}

/// `result` is the refusal of `subject` for `issuer`, the one answer to a
/// foreign, ownerless or missing issuer alike.
fn refused<T: std::fmt::Debug>(result: Result<T, DatabaseError>, subject: &str, issuer: IssuerId) {
    match result {
        Err(DatabaseError::AccessRefused(AccessRefusal { principal, missing })) => {
            assert_eq!(principal, id(subject));
            assert_eq!(missing, MissingRight::Issuer(issuer));
        }
        other => panic!("expected {subject} refused for the issuer, got {other:?}"),
    }
}

/// An issuer belongs to the principal whose enrollment created it. Another
/// principal in the same process is refused every control action on it,
/// with nothing changed and no outcome told: re-enrollment, resume, lookup,
/// exclusion, acknowledgement, retirement, and a write carrying one of its
/// descriptors. The owner keeps using it throughout.
#[test]
fn a_foreign_principal_reaches_nothing_of_an_issuer() {
    let (db, _dir) = open();
    for subject in ["alice", "bob"] {
        db.change_access(
            &ActingContext::Trusted,
            AccessChange::Grant {
                namespace: "default".into(),
                grantee: GranteeName::Principal(id(subject)),
                role: NamespaceRole::Writer,
            },
        )
        .expect("grant writer");
    }
    let issuer = IssuerId([0x7a; 16]);
    let enrolled = db
        .enroll_issuer_as(&acting("alice"), issuer)
        .expect("alice enrolls");
    let written = enrolled.next().expect("a descriptor");
    db.execute_cypher_shared_with(
        "CREATE (:Owned {k: 1})",
        None,
        None,
        &StatementOptions {
            actor: Some(acting("alice")),
            operation: Some(written),
            ..StatementOptions::default()
        },
    )
    .expect("alice writes under her operation");
    let pending = enrolled.next().expect("a second descriptor");

    let bob = acting("bob");
    refused(db.enroll_issuer_as(&bob, issuer), "bob", issuer);
    refused(
        db.resume_issuer_as(&bob, issuer, enrolled.incarnation(), 1),
        "bob",
        issuer,
    );
    refused(
        db.resolve_operation_as(&bob, &written, Resolve::Lookup),
        "bob",
        issuer,
    );
    refused(
        db.resolve_operation_as(&bob, &pending, Resolve::Exclude),
        "bob",
        issuer,
    );
    refused(db.acknowledge_operation_as(&bob, &written), "bob", issuer);
    refused(db.retire_operations_as(&bob, &pending), "bob", issuer);
    refused(
        db.execute_cypher_shared_with(
            "CREATE (:Stolen {k: 1})",
            None,
            None,
            &StatementOptions {
                actor: Some(bob.clone()),
                operation: Some(pending),
                ..StatementOptions::default()
            },
        ),
        "bob",
        issuer,
    );

    // Nothing of it changed: alice's incarnation stays open, her pending
    // operation stays unresolved and the committed one stays committed.
    let alice = acting("alice");
    assert_eq!(
        db.resolve_operation_as(&alice, &pending, Resolve::Lookup)
            .expect("alice looks it up"),
        OperationOutcome::Unknown,
        "bob's exclusion did not exclude it"
    );
    assert!(matches!(
        db.resolve_operation_as(&alice, &written, Resolve::Lookup)
            .expect("alice looks it up"),
        OperationOutcome::Committed { .. }
    ));
    let resumed = db
        .resume_issuer_as(&alice, issuer, enrolled.incarnation(), 1)
        .expect("alice resumes her open incarnation");
    assert_eq!(resumed.incarnation(), enrolled.incarnation());
    let stolen = db
        .execute_cypher_shared("MATCH (s:Stolen) RETURN s", None, None, None, None)
        .expect("count")
        .rows;
    assert!(
        stolen.is_empty(),
        "bob's write under alice's operation did nothing"
    );
}

/// An issuer the explicitly trusted host enrolled belongs to no principal:
/// a principal reaches nothing of it, and an issuer that was never enrolled
/// is refused the same way. The host keeps reaching it.
#[test]
fn an_ownerless_issuer_is_the_hosts_alone() {
    let (db, _dir) = open();
    let issuer = IssuerId([0x7b; 16]);
    let enrolled = db.enroll_issuer(issuer).expect("the host enrolls");
    let op = enrolled.next().expect("a descriptor");

    let alice = acting("alice");
    refused(db.enroll_issuer_as(&alice, issuer), "alice", issuer);
    refused(
        db.resolve_operation_as(&alice, &op, Resolve::Lookup),
        "alice",
        issuer,
    );
    let never = IssuerId([0x7c; 16]);
    refused(db.resume_issuer_as(&alice, never, 1, 1), "alice", never);

    assert_eq!(
        db.resolve_operation(&op, Resolve::Lookup)
            .expect("the host looks it up"),
        OperationOutcome::Unknown
    );
    db.enroll_issuer(issuer).expect("the host enrolls it again");
}
