//! CypherService gRPC integration tests.
//!
//! These tests exercise `CypherService` endpoints against a real `coordinode`
//! binary spawned in a temp directory.
//!
//! ## Test matrix
//!
//! | Test | Gap | Scenario |
//! |------|-----|---------|
//! | `g088_causal_write_without_majority_rejected` | G088 | Write + after_index > 0 + no write_concern → FAILED_PRECONDITION |
//! | `g088_causal_write_with_w1_rejected` | G088 | Write + after_index > 0 + W1 → FAILED_PRECONDITION |
//! | `g088_causal_write_with_majority_accepted` | G088 | Write + after_index > 0 + MAJORITY → OK |
//! | `g088_causal_read_without_majority_accepted` | G088 | Read + after_index > 0 + no write_concern → OK (gate skipped) |
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test cypher
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::query::ExecuteCypherRequest;
use coordinode_integration::proto::replication::{ReadConcern, WriteConcern, WriteConcernLevel};

// ── G088: write-concern validation in causal sessions ─────────────────────────

/// Causal write without write_concern is rejected end-to-end (gRPC path).
///
/// Full production path:
///   gRPC client → CypherService::execute_cypher()
///     → after_index > 0, is_write() == true, write_concern == None
///       → FAILED_PRECONDITION
#[tokio::test]
async fn g088_causal_write_without_majority_rejected() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cypher_client().await;

    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (n:G088Test {x: 1})".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: Some(ReadConcern {
                level: 2, // MAJORITY
                after_index: 1,
            }),
            write_concern: None, // omitted → treated as UNSPECIFIED (w:1)
        })
        .await;

    let status = result.expect_err("causal write without write_concern must be rejected");
    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "expected FAILED_PRECONDITION, got {:?}: {}",
        status.code(),
        status.message()
    );
    assert!(
        status.message().contains("MAJORITY") || status.message().contains("causal"),
        "error must mention MAJORITY or causal, got: {}",
        status.message()
    );
}

/// Causal write with WriteConcern=W1 is rejected end-to-end (gRPC path).
///
/// W1 is insufficient for causal sessions — the write may never replicate,
/// making the returned applied_index a dangling dependency.
#[tokio::test]
async fn g088_causal_write_with_w1_rejected() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cypher_client().await;

    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MERGE (n:G088Test {x: 2})".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: Some(ReadConcern {
                level: 2, // MAJORITY
                after_index: 5,
            }),
            write_concern: Some(WriteConcern {
                level: WriteConcernLevel::W1 as i32,
                timeout_ms: 0,
                journal: false,
            }),
        })
        .await;

    let status = result.expect_err("causal write with W1 must be rejected");
    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "expected FAILED_PRECONDITION, got {:?}: {}",
        status.code(),
        status.message()
    );
}

/// Causal write with WriteConcern=MAJORITY succeeds end-to-end (gRPC path).
///
/// MAJORITY is the minimum required durability — the server accepts and
/// executes the write normally in standalone mode.
#[tokio::test]
async fn g088_causal_write_with_majority_accepted() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cypher_client().await;

    // after_index: 1 — triggers G088 gate (after_index > 0) but is immediately
    // satisfied because standalone Raft has applied_index >= 1 after election.
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (n:G088Test {x: 3})".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: Some(ReadConcern {
                level: 2, // MAJORITY
                after_index: 1,
            }),
            write_concern: Some(WriteConcern {
                level: WriteConcernLevel::Majority as i32,
                timeout_ms: 0,
                journal: false,
            }),
        })
        .await;

    assert!(
        result.is_ok(),
        "causal write with MAJORITY must succeed in standalone, got: {:?}",
        result.err()
    );
}

/// Read queries in causal sessions do not require write_concern (gRPC path).
///
/// The write-concern gate is only entered for mutating statements. MATCH
/// queries bypass the gate regardless of write_concern value.
#[tokio::test]
async fn g088_causal_read_without_majority_accepted() {
    let server = CoordinodeProcess::start().await;
    let mut client = server.cypher_client().await;

    // after_index: 1 — triggers G088 gate check (after_index > 0) and is
    // immediately satisfied on standalone (applied_index >= 1 after election).
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:G088Test) RETURN n".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: Some(ReadConcern {
                level: 2, // MAJORITY
                after_index: 1,
            }),
            write_concern: None, // read-only — write_concern irrelevant
        })
        .await;

    assert!(
        result.is_ok(),
        "causal read without write_concern must succeed, got: {:?}",
        result.err()
    );
}
