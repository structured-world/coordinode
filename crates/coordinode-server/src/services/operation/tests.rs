//! Client operation identity over the gRPC services: the reference a write
//! carries, the refusal of a retry, and the operation service.

use std::sync::Arc;

use coordinode_embed::Database;
use parking_lot::RwLock;
use tonic::{Code, Request};
use tonic_types::StatusExt;

use super::*;
use crate::proto::graph;
use crate::proto::graph::graph_service_server::GraphService;
use crate::proto::operation::operation_service_server::OperationService;
use crate::proto::query;
use crate::proto::query::cypher_service_server::CypherService;
use crate::services::cypher::CypherServiceImpl;
use crate::services::graph::GraphServiceImpl;

const ISSUER: [u8; 16] = [0x61; 16];

struct Services {
    operations: OperationServiceImpl,
    cypher: CypherServiceImpl,
    graph: GraphServiceImpl,
    /// What a session's transactions run through.
    session: crate::services::session::DatabaseCursorEngine,
    _dir: tempfile::TempDir,
}

fn services() -> Services {
    let dir = tempfile::tempdir().unwrap();
    let database = Arc::new(RwLock::new(Database::open(dir.path()).unwrap()));
    let executor = StatementExecutor::new(Arc::clone(&database));
    Services {
        operations: OperationServiceImpl::new(executor.clone()),
        session: crate::services::session::DatabaseCursorEngine::from_executor(executor.clone()),
        cypher: CypherServiceImpl::from_executor(executor),
        graph: GraphServiceImpl::new(database),
        _dir: dir,
    }
}

/// An enrolled incarnation's first reference and its deployment.
async fn enroll(s: &Services) -> wire::OperationRef {
    let enrolled = s
        .operations
        .enroll_issuer(Request::new(wire::EnrollIssuerRequest {
            issuer: ISSUER.to_vec(),
        }))
        .await
        .unwrap()
        .into_inner();
    wire::OperationRef {
        issuer: ISSUER.to_vec(),
        incarnation: enrolled.incarnation,
        sequence: enrolled.next_sequence,
        deployment: enrolled.deployment,
    }
}

fn statement(
    query: &str,
    operation: Option<wire::OperationRef>,
) -> Request<query::ExecuteCypherRequest> {
    Request::new(query::ExecuteCypherRequest {
        query: query.to_string(),
        operation,
        ..Default::default()
    })
}

/// The reason and metadata a refusal carries.
fn reason(status: &Status) -> (String, std::collections::HashMap<String, String>) {
    let info = status
        .get_details_error_info()
        .expect("a refusal carries ErrorInfo");
    (info.reason, info.metadata)
}

async fn resolve(
    s: &Services,
    operation: &wire::OperationRef,
    exclude: bool,
) -> wire::ResolveOperationResponse {
    s.operations
        .resolve_operation(Request::new(wire::ResolveOperationRequest {
            operation: Some(operation.clone()),
            exclude,
        }))
        .await
        .unwrap()
        .into_inner()
}

/// A reference survives the trip to a descriptor and back, and a malformed
/// one is refused naming the field, never admitted as something else.
#[test]
fn a_reference_converts_and_a_malformed_one_is_refused() {
    let reference = wire::OperationRef {
        issuer: ISSUER.to_vec(),
        incarnation: 2,
        sequence: 9,
        deployment: vec![3; 16],
    };
    let descriptor = descriptor_from_proto("operation", &reference).unwrap();
    assert_eq!(descriptor_to_proto(&descriptor), reference);

    for (broken, field) in [
        (
            wire::OperationRef {
                issuer: vec![1; 15],
                ..reference.clone()
            },
            "operation.issuer",
        ),
        (
            wire::OperationRef {
                deployment: Vec::new(),
                ..reference.clone()
            },
            "operation.deployment",
        ),
    ] {
        let refused = descriptor_from_proto("operation", &broken).unwrap_err();
        assert_eq!(refused.code(), Code::InvalidArgument);
        assert_eq!(reason(&refused).1["field"], field);
    }
}

/// A unary write retried with its reference runs once; the retry is refused
/// with the outcome the first attempt left, and the operation resolves to
/// the same commit. A statement inside a transaction names no operation.
#[tokio::test(flavor = "multi_thread")]
async fn a_unary_write_retried_runs_once() {
    let s = services();
    let op = enroll(&s).await;
    let stats = s
        .cypher
        .execute_cypher(statement("CREATE (:Paid {k: 1})", Some(op.clone())))
        .await
        .unwrap()
        .into_inner()
        .stats
        .unwrap();

    let refused = s
        .cypher
        .execute_cypher(statement("CREATE (:Paid {k: 1})", Some(op.clone())))
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::FailedPrecondition);
    let (why, metadata) = reason(&refused);
    assert_eq!(why, "OPERATION_SETTLED");
    assert_eq!(metadata["outcome"], "COMMITTED");
    assert_eq!(metadata["intent_differs"], "false");
    assert_eq!(metadata["commit_ts"], stats.commit_ts.to_string());

    let resolved = resolve(&s, &op, false).await;
    assert_eq!(resolved.state(), wire::OperationState::Committed);
    assert_eq!(resolved.commit_ts, stats.commit_ts);

    let rows = s
        .cypher
        .execute_cypher(statement("MATCH (n:Paid) RETURN n", None))
        .await
        .unwrap()
        .into_inner()
        .rows;
    assert_eq!(rows.len(), 1);
}

/// A session commit repeated after a lost reply answers with the operation's
/// outcome and writes nothing again.
#[tokio::test(flavor = "multi_thread")]
async fn a_repeated_commit_answers_with_the_outcome() {
    use coordinode_core::txn::command::AdmissionSignal;
    use coordinode_session::CursorEngine;

    let s = services();
    let op = super::optional_descriptor("operation", Some(&enroll(&s).await))
        .unwrap()
        .expect("a descriptor");
    let txn = coordinode_embed::db::draw_transaction_id();
    s.session
        .begin_transaction(
            txn,
            &coordinode_session::ConnectionSettings::default(),
            None,
        )
        .unwrap();
    s.session
        .open_cursor(
            "CREATE (:Purchase {k: 1})",
            std::collections::HashMap::new(),
            Some(&coordinode_session::InTransaction {
                id: txn,
                admission: AdmissionSignal::new(|| {}),
                start: None,
            }),
            &coordinode_session::ConnectionSettings::default(),
            None,
            None,
            &coordinode_core::budget::CancelFlag::new(),
        )
        .unwrap();
    let commit = || {
        s.session
            .commit_transaction(txn, Some(&op), &[], &AdmissionSignal::new(|| {}))
            .map_err(|e| crate::services::session::status(e.0))
    };
    let receipt = commit().unwrap();
    let refused = commit().unwrap_err();
    let (why, metadata) = reason(&refused);
    assert_eq!(why, "OPERATION_SETTLED");
    assert_eq!(metadata["outcome"], "COMMITTED");
    assert_eq!(
        metadata["commit_ts"],
        receipt.commit_ts.as_raw().to_string()
    );
}

/// A typed creation carries its operation like a statement does; a batch
/// names one operation for the whole batch, never one per entry.
#[tokio::test(flavor = "multi_thread")]
async fn a_typed_creation_retried_creates_once() {
    let s = services();
    let op = enroll(&s).await;
    let create = || {
        Request::new(graph::CreateNodeRequest {
            labels: vec!["Typed".to_string()],
            properties: std::collections::HashMap::new(),
            operation: Some(op.clone()),
        })
    };
    s.graph.create_node(create()).await.unwrap();
    let refused = s.graph.create_node(create()).await.unwrap_err();
    assert_eq!(reason(&refused).0, "OPERATION_SETTLED");
    let rows = s
        .cypher
        .execute_cypher(statement("MATCH (n:Typed) RETURN n", None))
        .await
        .unwrap()
        .into_inner()
        .rows;
    assert_eq!(rows.len(), 1);

    let refused = s
        .graph
        .create_nodes_batch(Request::new(graph::CreateNodesBatchRequest {
            nodes: vec![graph::CreateNodeRequest {
                labels: vec!["Typed".to_string()],
                properties: std::collections::HashMap::new(),
                operation: Some(op.clone()),
            }],
            operation: None,
        }))
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert_eq!(reason(&refused).1["field"], "nodes[0].operation");
}

/// The operation service: an unused number is excluded and stays aborted,
/// an acknowledged outcome reads as retired, a retired prefix admits
/// nothing, a resumed issuer continues its numbers, and a reference of
/// another deployment is refused rather than admitted.
#[tokio::test(flavor = "multi_thread")]
async fn the_operation_service_resolves_acknowledges_and_retires() {
    let s = services();
    let first = enroll(&s).await;
    let next = |sequence| wire::OperationRef {
        sequence,
        ..first.clone()
    };
    s.cypher
        .execute_cypher(statement("CREATE (:A)", Some(first.clone())))
        .await
        .unwrap();

    let unused = next(first.sequence + 1);
    assert_eq!(
        resolve(&s, &unused, false).await.state(),
        wire::OperationState::Unknown
    );
    assert_eq!(
        resolve(&s, &unused, true).await.state(),
        wire::OperationState::Aborted
    );
    let refused = s
        .cypher
        .execute_cypher(statement("CREATE (:A)", Some(unused.clone())))
        .await
        .unwrap_err();
    assert_eq!(reason(&refused).1["outcome"], "ABORTED");

    s.operations
        .acknowledge_operations(Request::new(wire::AcknowledgeOperationsRequest {
            issuer: ISSUER.to_vec(),
            incarnation: first.incarnation,
            sequences: vec![first.sequence],
            deployment: first.deployment.clone(),
        }))
        .await
        .unwrap();
    assert_eq!(
        resolve(&s, &first, false).await.state(),
        wire::OperationState::Retired
    );

    let resumed = s
        .operations
        .resume_issuer(Request::new(wire::ResumeIssuerRequest {
            issuer: ISSUER.to_vec(),
            incarnation: first.incarnation,
            next_sequence: first.sequence + 2,
        }))
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resumed.next_sequence, first.sequence + 2);
    assert_eq!(resumed.deployment, first.deployment);

    s.operations
        .retire_operations(Request::new(wire::RetireOperationsRequest {
            issuer: ISSUER.to_vec(),
            incarnation: first.incarnation,
            below: first.sequence + 3,
            deployment: first.deployment.clone(),
        }))
        .await
        .unwrap();
    let never_used = next(first.sequence + 2);
    assert_eq!(
        resolve(&s, &never_used, false).await.state(),
        wire::OperationState::Retired
    );

    let foreign = wire::OperationRef {
        deployment: vec![0xee; 16],
        ..next(first.sequence + 5)
    };
    let refused = s
        .cypher
        .execute_cypher(statement("CREATE (:A)", Some(foreign)))
        .await
        .unwrap_err();
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert_eq!(reason(&refused).0, "FOREIGN_OPERATION");
}
