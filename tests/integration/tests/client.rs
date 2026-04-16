//! coordinode-client causal session API integration tests (G089).
//!
//! These tests exercise the `coordinode-client` Rust driver's causal session
//! API against a real `coordinode` binary. They verify that:
//!
//!   - `execute_causal_write` returns a non-zero `CausalToken` in standalone
//!     mode (applied_index from QueryStats).
//!
//!   **Note on standalone mode:** the server runs as a single-node Raft
//!   instance. `applied_index` in standalone mode reflects the Raft oplog
//!   position and is therefore > 0 after any committed write.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test client
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_client::{CausalToken, CoordinodeClient, Value};
use coordinode_integration::harness::CoordinodeProcess;

// ── G089: causal session API round-trip ───────────────────────────────────────

/// `execute_causal_write` sends write with MAJORITY concern and returns a
/// `CausalToken`.  In standalone mode applied_index is non-zero after the
/// first write because RaftNode::single_node() is a real Raft group (one voter).
///
/// This test verifies that:
///   - The write succeeds and returns rows normally.
///   - The CausalToken value is > 0 (Raft assigned a log index).
///   - A subsequent causal read (fenced by the token) succeeds and returns the
///     written node.
#[tokio::test]
async fn g089_causal_write_returns_token_and_causal_read_sees_write() {
    let server = CoordinodeProcess::start().await;

    let mut client = CoordinodeClient::connect(server.endpoint())
        .await
        .expect("connect coordinode-client");

    // Causal write: CREATE a node with a unique marker.
    let (rows, token) = client
        .execute_causal_write(
            "CREATE (n:G089Test {marker: 'causal_round_trip'}) RETURN n.marker AS m",
        )
        .await
        .expect("causal write must succeed");

    // The write returns the projected column.
    assert_eq!(rows.len(), 1, "expected 1 row from CREATE...RETURN");
    let marker_val = rows[0].get("m").expect("column 'm' must be present");
    assert_eq!(
        *marker_val,
        Value::String("causal_round_trip".into()),
        "returned marker must match"
    );

    // In a single-node Raft the applied_index is always > 0 after commit.
    assert!(
        token.as_u64() > 0,
        "CausalToken must be non-zero in standalone Raft mode, got {token:?}"
    );

    // Causal read: fenced by the token — must observe the write we just made.
    let read_rows = client
        .execute_causal_read(
            "MATCH (n:G089Test {marker: 'causal_round_trip'}) RETURN n.marker AS m",
            token,
        )
        .await
        .expect("causal read must succeed");

    assert_eq!(
        read_rows.len(),
        1,
        "causal read must find the node written before the token"
    );
    assert_eq!(
        read_rows[0].get("m").expect("column 'm'"),
        &Value::String("causal_round_trip".into()),
    );
}

/// `execute_causal_write_with_params` exercises the parameterised variant.
/// Verifies that parameters are correctly forwarded through the causal write path.
#[tokio::test]
async fn g089_causal_write_with_params_round_trip() {
    let server = CoordinodeProcess::start().await;

    let mut client = CoordinodeClient::connect(server.endpoint())
        .await
        .expect("connect coordinode-client");

    let mut params = std::collections::HashMap::new();
    params.insert("id".to_string(), Value::Int(8901));

    let (rows, token) = client
        .execute_causal_write_with_params(
            "CREATE (n:G089Param {id: $id}) RETURN n.id AS id",
            params,
        )
        .await
        .expect("causal write with params must succeed");

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("id").expect("column 'id'"), &Value::Int(8901));
    assert!(token.as_u64() > 0, "token must be non-zero");

    // Read back with params.
    let mut read_params = std::collections::HashMap::new();
    read_params.insert("id".to_string(), Value::Int(8901));

    let read_rows = client
        .execute_causal_read_with_params(
            "MATCH (n:G089Param {id: $id}) RETURN n.id AS id",
            read_params,
            token,
        )
        .await
        .expect("causal read with params must succeed");

    assert_eq!(read_rows.len(), 1);
    assert_eq!(
        read_rows[0].get("id").expect("column 'id'"),
        &Value::Int(8901)
    );
}

/// Zero token (standalone-mode sentinel) accepted by causal read without error.
///
/// When a CausalToken(0) is passed, the server performs the read immediately
/// (no fence). This is the graceful degradation path for code that runs in
/// both standalone and cluster configurations.
#[tokio::test]
async fn g089_zero_token_causal_read_is_accepted() {
    let server = CoordinodeProcess::start().await;

    let mut client = CoordinodeClient::connect(server.endpoint())
        .await
        .expect("connect coordinode-client");

    // Zero token: equivalent to a non-causal read — should succeed immediately.
    let rows = client
        .execute_causal_read("MATCH (n) RETURN n LIMIT 0", CausalToken::from(0))
        .await
        .expect("zero-token causal read must succeed");

    // LIMIT 0 always returns empty rows regardless.
    assert!(rows.is_empty(), "LIMIT 0 must return no rows");
}
