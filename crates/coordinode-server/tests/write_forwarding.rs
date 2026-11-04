//! A write sent to a follower reaches the leader anyway.
//!
//! Only the leader can replicate a write, but making every client discover
//! which node that is turns a leadership change into an application problem.
//! A node that knows who leads passes the request along instead, so a client
//! that opened one connection and never thinks about topology keeps working
//! across an election. The response says how many nodes handled it, for a
//! client that does care.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

// Each test binary uses part of the shared members.
#[allow(dead_code)]
#[path = "support/members.rs"]
mod members;

use std::time::Duration;

use coordinode_server::proto::{operation, query};

use members::leader_and_follower;

#[tokio::test(flavor = "multi_thread")]
async fn a_write_sent_to_a_follower_is_carried_to_the_leader() {
    let ((n1, _p1), (_n2, p2)) = leader_and_follower().await;

    // The client talks to the FOLLOWER and never learns there is a leader.
    let mut client = query::cypher_service_client::CypherServiceClient::connect(format!(
        "http://127.0.0.1:{p2}"
    ))
    .await
    .unwrap();

    let response = client
        .execute_cypher(query::ExecuteCypherRequest {
            query: "CREATE (:Forwarded {tag: 'via-follower'})".to_string(),
            ..Default::default()
        })
        .await
        .expect("a write to a follower must reach the leader");

    assert_eq!(
        response
            .metadata()
            .get("x-coordinode-hops")
            .map(|v| v.to_str().unwrap_or_default()),
        Some("1"),
        "the response should say the write was carried one hop"
    );

    // It really landed, and on the leader: this is the state Raft replicated,
    // not something the follower accepted locally.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = n1
        .db
        .write()
        .execute_cypher("MATCH (n:Forwarded) RETURN n.tag AS tag")
        .unwrap();
    assert_eq!(
        rows.len(),
        1,
        "the forwarded write is missing on the leader"
    );
    assert_eq!(
        rows[0].get("tag").and_then(|v| v.as_str()),
        Some("via-follower")
    );
}

/// A client that only knows a follower numbers its writes all the same: the
/// enrollment, a bound write and its retry, and an exclusion all reach the
/// leader, and the write takes effect once.
#[tokio::test(flavor = "multi_thread")]
async fn operation_identity_works_through_a_follower() {
    use tonic_types::StatusExt;

    let ((n1, p1), (_n2, p2)) = leader_and_follower().await;

    let follower = format!("http://127.0.0.1:{p2}");
    let mut operations =
        operation::operation_service_client::OperationServiceClient::connect(follower.clone())
            .await
            .unwrap();
    let mut cypher = query::cypher_service_client::CypherServiceClient::connect(follower)
        .await
        .unwrap();

    let enrolled = operations
        .enroll_issuer(operation::EnrollIssuerRequest {
            issuer: vec![0x52; 16],
        })
        .await
        .expect("an enrollment through a follower reaches the leader")
        .into_inner();
    let reference = |sequence| operation::OperationRef {
        issuer: vec![0x52; 16],
        incarnation: enrolled.incarnation,
        sequence,
        deployment: enrolled.deployment.clone(),
    };
    let write = |sequence| query::ExecuteCypherRequest {
        query: "CREATE (:Bound {k: 1})".to_string(),
        operation: Some(reference(sequence)),
        ..Default::default()
    };

    cypher
        .execute_cypher(write(enrolled.next_sequence))
        .await
        .expect("a bound write through a follower");
    let refused = cypher
        .execute_cypher(write(enrolled.next_sequence))
        .await
        .expect_err("its retry runs nothing");
    let info = refused.get_details_error_info().expect("ErrorInfo");
    assert_eq!(info.reason, "OPERATION_SETTLED");
    assert_eq!(info.metadata["outcome"], "COMMITTED");

    let excluded = operations
        .resolve_operation(operation::ResolveOperationRequest {
            operation: Some(reference(enrolled.next_sequence + 1)),
            exclude: true,
        })
        .await
        .expect("an exclusion through a follower reaches the leader")
        .into_inner();
    assert_eq!(excluded.state(), operation::OperationState::Aborted);

    // Releasing the committed outcome and retiring the numbers, also through
    // the follower, take effect at the leader: it answers the operation as
    // retired.
    operations
        .acknowledge_operations(operation::AcknowledgeOperationsRequest {
            issuer: vec![0x52; 16],
            incarnation: enrolled.incarnation,
            sequences: vec![enrolled.next_sequence],
            deployment: enrolled.deployment.clone(),
        })
        .await
        .expect("a release through a follower reaches the leader");
    operations
        .retire_operations(operation::RetireOperationsRequest {
            issuer: vec![0x52; 16],
            incarnation: enrolled.incarnation,
            below: enrolled.next_sequence + 2,
            deployment: enrolled.deployment.clone(),
        })
        .await
        .expect("a retirement through a follower reaches the leader");
    let at_leader = operation::operation_service_client::OperationServiceClient::connect(format!(
        "http://127.0.0.1:{p1}"
    ))
    .await
    .unwrap()
    .resolve_operation(operation::ResolveOperationRequest {
        operation: Some(reference(enrolled.next_sequence)),
        exclude: false,
    })
    .await
    .expect("a lookup at the leader")
    .into_inner();
    assert_eq!(at_leader.state(), operation::OperationState::Retired);

    tokio::time::sleep(Duration::from_millis(500)).await;
    let rows = n1
        .db
        .write()
        .execute_cypher("MATCH (n:Bound) RETURN n")
        .unwrap();
    assert_eq!(rows.len(), 1, "the write took effect once");
}
