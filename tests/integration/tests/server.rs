//! Server startup and response-header integration tests (R150).
//!
//! These tests exercise the multi-protocol handler, `NodeInfoLayer` response
//! headers, and CE/EE mode validation against a real `coordinode` binary.
//!
//! ## Test matrix
//!
//! | Test | Gap | Scenario |
//! |------|-----|---------|
//! | `node_info_headers_present_in_grpc_response` | R150 | `x-coordinode-node/hops/load` injected by NodeInfoLayer |
//! | `x_coordinode_node_value_matches_default_node_id` | R150 | node_id=1 (default) reflected in `x-coordinode-node` header |
//! | `ee_mode_compute_rejected_at_startup` | R150 | `--mode=compute` exits non-zero with "coordinode-ee" message |
//! | `ee_mode_storage_rejected_at_startup` | R150 | `--mode=storage` exits non-zero with "coordinode-ee" message |
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test server
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_integration::harness::{binary_path, CoordinodeProcess};
use coordinode_integration::proto::common::property_value::Value as PvKind;
use coordinode_integration::proto::query::ExecuteCypherRequest;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Execute a Cypher query and return the raw tonic Response (including metadata).
async fn cypher_raw(
    proc: &CoordinodeProcess,
    query: &str,
) -> tonic::Response<coordinode_integration::proto::query::ExecuteCypherResponse> {
    let mut client = proc.cypher_client().await;
    client
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await
        .expect("execute_cypher must succeed")
}

// ── NodeInfoLayer response-header tests ───────────────────────────────────────

/// NodeInfoLayer must inject `x-coordinode-node`, `x-coordinode-hops`, and
/// `x-coordinode-load` headers into every gRPC response.
///
/// These headers are added by the tower `NodeInfoLayer` middleware that wraps
/// the main tonic router. They appear as HTTP/2 initial metadata on the client.
///
/// CE invariants tested:
/// - All three headers are present on every response
/// - `x-coordinode-hops` is "0" (CE has no routing — always local)
/// - `x-coordinode-load` is "0" (load tracking deferred to R151)
/// - EE-only header `x-coordinode-shard-hint` must NOT be present
#[tokio::test]
async fn node_info_headers_present_in_grpc_response() {
    let server = CoordinodeProcess::start().await;
    let response = cypher_raw(&server, "MATCH (n) RETURN n LIMIT 0").await;
    let meta = response.metadata();

    // All three CE headers must be present
    assert!(
        meta.get("x-coordinode-node").is_some(),
        "x-coordinode-node header must be present, got metadata keys: {:?}",
        meta.keys().collect::<Vec<_>>()
    );
    assert!(
        meta.get("x-coordinode-hops").is_some(),
        "x-coordinode-hops header must be present"
    );
    assert!(
        meta.get("x-coordinode-load").is_some(),
        "x-coordinode-load header must be present"
    );

    // CE hops must be 0 (no routing layer)
    let hops = meta.get("x-coordinode-hops").unwrap().to_str().unwrap();
    assert_eq!(hops, "0", "CE must always report 0 hops (local execution)");

    // CE load must be 0 (tracking deferred to R151)
    let load = meta.get("x-coordinode-load").unwrap().to_str().unwrap();
    assert_eq!(load, "0", "CE load tracking not yet implemented (R151)");

    // EE-only shard-hint must NOT be present in CE binary
    assert!(
        meta.get("x-coordinode-shard-hint").is_none(),
        "x-coordinode-shard-hint is EE-only — must not appear in CE responses"
    );
}

/// The `x-coordinode-node` header must contain the numeric node ID.
///
/// Default single-node start uses node_id=1 (hardcoded default).
/// This test verifies the value is a valid decimal integer and equals "1".
///
/// Implementation path:
///   NodeInfoLayer::new(node_id=1) → inserts node_id.to_string() into headers
///   → appears as x-coordinode-node: "1" in client initial metadata
#[tokio::test]
async fn x_coordinode_node_value_matches_default_node_id() {
    let server = CoordinodeProcess::start().await;
    let response = cypher_raw(&server, "RETURN 1 AS n").await;
    let meta = response.metadata();

    let node_val = meta
        .get("x-coordinode-node")
        .expect("x-coordinode-node must be present")
        .to_str()
        .expect("must be valid ASCII");

    // Must be a valid u64
    let node_id: u64 = node_val
        .parse()
        .unwrap_or_else(|_| panic!("x-coordinode-node must be a number, got '{node_val}'"));

    // Default single-node startup uses node_id=1
    assert_eq!(
        node_id, 1,
        "default single-node startup must report node_id=1, got {node_id}"
    );
}

/// NodeInfoLayer headers must be injected on WRITE responses too — not just reads.
///
/// Ensures the middleware wraps the full router, not just specific service handlers.
#[tokio::test]
async fn node_info_headers_present_on_write_response() {
    let server = CoordinodeProcess::start().await;
    let response = cypher_raw(
        &server,
        "CREATE (n:HeaderTest {id: 'r150'}) RETURN n.id AS id",
    )
    .await;
    let meta = response.metadata();

    assert!(
        meta.get("x-coordinode-node").is_some(),
        "x-coordinode-node must be present on write responses too"
    );
    assert!(
        meta.get("x-coordinode-hops").is_some(),
        "x-coordinode-hops must be present on write responses too"
    );
}

// ── CE/EE mode validation tests ───────────────────────────────────────────────

/// `--mode=compute` must cause immediate startup failure in the CE binary.
///
/// EE modes (compute, storage) are not available in CE. The binary must
/// exit with a non-zero status code and print a message containing
/// "coordinode-ee" so operators know to use the EE binary.
///
/// This prevents silent degradation where CE silently ignores an EE flag
/// and starts in an unintended configuration.
#[test]
fn ee_mode_compute_rejected_at_startup() {
    let output = std::process::Command::new(binary_path())
        .args(["serve", "--mode", "compute", "--ops-addr", "[::1]:0"])
        .output()
        .expect("failed to spawn coordinode binary");

    assert!(
        !output.status.success(),
        "coordinode --mode=compute must exit non-zero in CE, got: {:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("coordinode-ee"),
        "error message must mention 'coordinode-ee' so operators know to use EE binary. \
         Got stderr: {stderr}"
    );
}

/// `--mode=storage` must cause immediate startup failure in the CE binary.
///
/// Same invariant as compute — both EE modes must be cleanly rejected.
#[test]
fn ee_mode_storage_rejected_at_startup() {
    let output = std::process::Command::new(binary_path())
        .args(["serve", "--mode", "storage", "--ops-addr", "[::1]:0"])
        .output()
        .expect("failed to spawn coordinode binary");

    assert!(
        !output.status.success(),
        "coordinode --mode=storage must exit non-zero in CE, got: {:?}",
        output.status
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("coordinode-ee"),
        "error message must mention 'coordinode-ee'. Got stderr: {stderr}"
    );
}

/// `--mode=full` must start normally (default mode, CE-supported).
///
/// Regression guard: ensures we don't accidentally reject the default mode.
#[tokio::test]
async fn mode_full_starts_and_accepts_queries() {
    // CoordinodeProcess::start() uses default flags — mode=full implicitly.
    // We verify the server is functional and accepts gRPC queries.
    let server = CoordinodeProcess::start().await;
    let response = cypher_raw(&server, "RETURN 42 AS answer").await;

    let resp = response.into_inner();
    assert_eq!(resp.columns, vec!["answer"]);
    assert_eq!(resp.rows.len(), 1);

    let val = &resp.rows[0].values[0];
    assert!(
        matches!(val.value, Some(PvKind::IntValue(42))),
        "expected IntValue(42), got: {val:?}"
    );
}
