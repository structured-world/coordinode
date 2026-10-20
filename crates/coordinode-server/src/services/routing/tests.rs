use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use coordinode_core::budget::CancelFlag;
use coordinode_core::group::DeploymentId;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal, TransactionId};
use coordinode_core::txn::operation::{IssuerId, OperationDescriptor, OperationId};
use coordinode_embed::DatabaseError;
use coordinode_embed::operation::Resolve;
use coordinode_embed::replication::routing::{
    AnswerWriter, CommandExecutor, CommandRouter, CommandRoutingHandler, MAX_ROUTED_MESSAGE,
    RoutedBegin, RoutedCommit, RoutedOperationCall, RoutedRollback, RoutedSettings,
    RoutedStatement,
};
use tonic_types::StatusExt;

use super::ServerCommandExecutor;
use crate::services::cypher::db_error_to_status;
use crate::services::statement::tests::standalone;

/// The same executor twice: called on this member, and served to a router
/// over the routing service, as another member reaches it.
async fn both_ways() -> (Arc<ServerCommandExecutor>, CommandRouter, tempfile::TempDir) {
    let (executor, dir) = standalone();
    let executor = Arc::new(ServerCommandExecutor::new(executor));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("address");
    let incoming = tonic::transport::server::TcpIncoming::from(listener);
    let served = Arc::clone(&executor);
    tokio::spawn(async move {
        tonic::transport::Server::builder()
            .add_service(CommandRoutingHandler::new(served).into_service(MAX_ROUTED_MESSAGE))
            .serve_with_incoming(incoming)
            .await
            .expect("serve");
    });
    let channel = tonic::transport::Endpoint::from_shared(format!("http://{addr}"))
        .expect("endpoint")
        .connect()
        .await
        .expect("connect");
    (executor, CommandRouter::new(1, channel), dir)
}

fn statement(query: &str) -> RoutedStatement {
    RoutedStatement {
        actor: ActingContext::Trusted,
        query: query.to_owned(),
        params: HashMap::new(),
        settings: RoutedSettings::default(),
        transaction: None,
        operation: None,
        deadline: Some(Duration::from_secs(30)),
        source: None,
    }
}

/// Run `query` on the executor directly, as its own member does.
async fn locally(executor: &Arc<ServerCommandExecutor>, query: &str) -> DatabaseError {
    let executor = Arc::clone(executor);
    let statement = statement(query);
    tokio::task::spawn_blocking(move || {
        executor
            .execute(
                statement,
                &CancelFlag::new(),
                &AdmissionSignal::new(|| {}),
                &mut AnswerWriter::unread(),
            )
            .expect_err("refused locally")
    })
    .await
    .expect("joined")
}

/// The error and the status a client of either member sees are the same.
fn assert_same(what: &str, local: DatabaseError, routed: DatabaseError) {
    assert_eq!(format!("{routed:?}"), format!("{local:?}"), "{what}");
    let (local, routed) = (db_error_to_status(local), db_error_to_status(routed));
    assert_eq!(routed.code(), local.code(), "{what}");
    assert_eq!(routed.message(), local.message(), "{what}");
    let info = |s: &tonic::Status| {
        s.get_error_details()
            .error_info()
            .map(|i| (i.reason.clone(), i.domain.clone(), i.metadata.clone()))
    };
    assert_eq!(info(&routed), info(&local), "{what}");
}

/// A statement refused by the engine reaches the ingress as the refusal the
/// executing member raised: the same variant and fields, and the same status
/// once projected for a client.
#[tokio::test(flavor = "multi_thread")]
async fn a_statement_fails_the_same_way_on_both_paths() {
    let (executor, router, _dir) = both_ways().await;
    let setup = [
        "CREATE CONSTRAINT u_k FOR (n:U) REQUIRE n.k IS UNIQUE",
        "CREATE (:U {k: 1})",
    ];
    for query in setup {
        router
            .execute(statement(query), &AdmissionSignal::new(|| {}))
            .await
            .unwrap_or_else(|e| panic!("{query}: {e}"));
    }

    for query in [
        "RETURN 1/0",
        "RETURN 9223372036854775807 + 1",
        "RETURN lenght('x')",
        "MATC (n) RETURN n",
        "CALL db.nosuch()",
        "CREATE (:U {k: 1})",
        "CREATE CONSTRAINT u_k FOR (n:U) REQUIRE n.k IS UNIQUE",
        "SET vector_consistency = 'exact'",
    ] {
        let local = locally(&executor, query).await;
        let routed = router
            .execute(statement(query), &AdmissionSignal::new(|| {}))
            .await
            .expect_err(query);
        assert_same(query, local, routed);
    }
}

/// A read that pages is answered page by page at the executing member, and
/// the ingress gets every row once, as the member's own client would: more
/// rows than one page holds, none lost or repeated at a page boundary.
#[tokio::test(flavor = "multi_thread")]
async fn a_paged_read_answers_every_row_across_pages() {
    let (executor, router, _dir) = both_ways().await;
    router
        .execute(
            statement("UNWIND range(1, 2500) AS k CREATE (:Paged {k: k})"),
            &AdmissionSignal::new(|| {}),
        )
        .await
        .expect("seeded");
    let query = "MATCH (n:Paged) RETURN n.k AS k";
    assert!(executor.executor.database().read().keyset_pageable(query));

    let answer = router
        .execute(statement(query), &AdmissionSignal::new(|| {}))
        .await
        .expect("read");
    assert_eq!(answer.columns, ["k".to_string()]);
    let mut keys: Vec<i64> = answer
        .rows
        .iter()
        .map(|row| match row.as_slice() {
            [coordinode_core::graph::types::Value::Int(k)] => *k,
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, (1..=2500).collect::<Vec<i64>>());
}

/// Transaction control refused by the executing member reaches the ingress
/// as the same refusal.
#[tokio::test(flavor = "multi_thread")]
async fn transaction_control_fails_the_same_way_on_both_paths() {
    let (executor, router, _dir) = both_ways().await;
    let id = TransactionId::from_raw([0x5a; 16]);
    let begin = || RoutedBegin {
        transaction: id,
        actor: ActingContext::Trusted,
        read_concern: None,
        write_concern: None,
        timeout_outcome: None,
        operation: None,
    };
    router.begin(begin()).await.expect("begun");

    let blocking = |f: Box<dyn FnOnce() -> DatabaseError + Send>| async move {
        tokio::task::spawn_blocking(f).await.expect("joined")
    };

    let local = {
        let executor = Arc::clone(&executor);
        let begin = begin();
        blocking(Box::new(move || executor.begin(begin).expect_err("taken"))).await
    };
    let routed = router.begin(begin()).await.expect_err("taken");
    assert!(matches!(routed, DatabaseError::TransactionIdInUse(t) if t == id));
    assert_same("begin under a taken id", local, routed);

    let unknown = TransactionId::from_raw([0x77; 16]);
    let commit = || RoutedCommit {
        actor: ActingContext::Trusted,
        transaction: unknown,
        operation: None,
        deadline: None,
        expect: Vec::new(),
    };
    let local = {
        let executor = Arc::clone(&executor);
        let commit = commit();
        blocking(Box::new(move || {
            executor.commit(commit).expect_err("unknown")
        }))
        .await
    };
    let routed = router.commit(commit()).await.expect_err("unknown");
    assert!(matches!(routed, DatabaseError::UnknownTransaction(t) if t == unknown));
    assert_same("commit of an unknown transaction", local, routed);

    let rollback = || RoutedRollback {
        actor: ActingContext::Trusted,
        transaction: unknown,
        abandoned: false,
    };
    let local = {
        let executor = Arc::clone(&executor);
        let rollback = rollback();
        blocking(Box::new(move || {
            executor.rollback(rollback).expect_err("unknown")
        }))
        .await
    };
    let routed = router.rollback(rollback()).await.expect_err("unknown");
    assert_same("rollback of an unknown transaction", local, routed);
}

/// A call on operation records refused by the executing member reaches the
/// ingress as the same refusal, with the deployment it names.
#[tokio::test(flavor = "multi_thread")]
async fn an_operation_call_fails_the_same_way_on_both_paths() {
    let (executor, router, _dir) = both_ways().await;
    let foreign = OperationDescriptor {
        op: OperationId {
            issuer: IssuerId([3; 16]),
            incarnation: 1,
            sequence: 1,
        },
        deployment: DeploymentId([0xee; 16]),
    };
    let call = || RoutedOperationCall::Resolve(foreign, Resolve::Lookup);
    let local = {
        let executor = Arc::clone(&executor);
        let call = call();
        tokio::task::spawn_blocking(move || {
            executor
                .operation(ActingContext::Trusted, call)
                .expect_err("foreign")
        })
        .await
        .expect("joined")
    };
    let routed = router
        .operation(&ActingContext::Trusted, call())
        .await
        .expect_err("foreign");
    assert!(
        matches!(routed, DatabaseError::ForeignOperation { descriptor } if descriptor == foreign.deployment),
        "{routed:?}"
    );
    assert_same("resolution under another deployment", local, routed);
}
