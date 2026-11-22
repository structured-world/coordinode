//! The embedded library built without `tenancy`: one store whose access is
//! decided by who can open its files.

#![cfg(not(feature = "tenancy"))]
#![allow(clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use coordinode_core::txn::command::{ActingContext, Principal};
use coordinode_core::txn::timestamp::TimestampOracle;
use coordinode_embed::db::StatementOptions;
use coordinode_embed::{Database, DatabaseError};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::Deployment;

fn config(dir: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A directory of one namespace opens and serves its data, again after a
/// restart; one holding several is refused, naming them, before anything is
/// read or written.
#[test]
fn a_store_of_one_namespace_opens_and_one_of_several_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("tempdir");
    {
        let db = Database::open(dir.path()).expect("a fresh store opens with its one namespace");
        db.execute_cypher_shared("CREATE (:Kept {k: 1})", None, None, None, None)
            .expect("write");
    }
    {
        let db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher_shared("MATCH (n:Kept) RETURN n.k", None, None, None, None)
            .expect("read")
            .rows;
        assert_eq!(rows.len(), 1, "the store serves what it holds");
    }

    // A second namespace, as a build with namespaces would leave it.
    {
        let deployment =
            Deployment::open_embedded(&config(dir.path()), Arc::new(TimestampOracle::new()))
                .expect("open the deployment");
        deployment
            .create_namespace("other")
            .expect("a second namespace");
    }
    let refused = match Database::open(dir.path()) {
        Ok(_) => panic!("a store of two namespaces opened"),
        Err(e) => e.to_string(),
    };
    assert!(
        refused.contains("default") && refused.contains("other"),
        "the refusal names the namespaces: {refused}"
    );
}

/// A call acting for a principal, and a statement on the catalog of
/// namespaces or grants, are refused rather than run as the trusted host:
/// this build checks no principal and has no such catalog. Nothing changes.
#[test]
fn principals_and_catalog_statements_are_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let alice = ActingContext::Principal(Principal {
        subject: Arc::from("alice"),
        trust_scope: Arc::from("idp"),
        namespace: Arc::from("default"),
    });
    let refused = db.execute_cypher_shared_with(
        "CREATE (:Written {k: 1})",
        None,
        None,
        &StatementOptions {
            actor: Some(alice),
            ..StatementOptions::default()
        },
    );
    assert!(
        matches!(refused, Err(DatabaseError::WithoutTenancy(_))),
        "{refused:?}"
    );
    for statement in [
        "CREATE NAMESPACE other",
        "SHOW NAMESPACES",
        "GRANT writer ON NAMESPACE default TO PRINCIPAL 'alice' IN 'idp'",
    ] {
        let refused = db.execute_cypher_shared(statement, None, None, None, None);
        assert!(
            matches!(refused, Err(DatabaseError::WithoutTenancy(_))),
            "{statement}: {refused:?}"
        );
    }
    let rows = db
        .execute_cypher_shared("MATCH (n:Written) RETURN n", None, None, None, None)
        .expect("read")
        .rows;
    assert!(rows.is_empty(), "the refused write changed nothing");
}
