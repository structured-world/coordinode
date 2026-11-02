//! coordinode-client causal session API integration tests.
//!
//! These tests exercise the `coordinode-client` Rust driver's causal session
//! API against a real `coordinode` binary. They verify that:
//!
//!   - `execute_causal_write` returns the write's commit time as its
//!     `CausalToken`, from the statement's receipt.
//!   - A causal read bounded by it observes the write, and returns a token
//!     no earlier than the one it was given (monotonic reads).
//!   - A bound the server cannot meet is refused as not ready.
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

// ── Causal session API round-trip ─────────────────────────────────────────────

/// `execute_causal_write` sends the write with MAJORITY concern and returns
/// its commit time as a `CausalToken`; a causal read bounded by it returns
/// the written node.
#[tokio::test]
async fn causal_write_returns_token_and_causal_read_sees_write() {
    let server = CoordinodeProcess::start().await;

    // Causal write: CREATE a node with a unique marker.
    let mut client = causal_client(&server).await;
    let (rows, token) = client
        .execute_causal_write(
            "CREATE (n:CausalTest {marker: 'causal_round_trip'}) RETURN n.marker AS m",
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

    // The token is the write's commit timestamp, never zero.
    assert!(
        token.as_u64() > 0,
        "CausalToken must be the write's commit time, got {token:?}"
    );

    // Causal read: bounded by the token, must observe the write we just made.
    let (read_rows, after_read) = client
        .execute_causal_read(
            "MATCH (n:CausalTest {marker: 'causal_round_trip'}) RETURN n.marker AS m",
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
    // Monotonic reads: the session's token never goes back, and the read's
    // own view is at or after the write it observed.
    assert!(
        after_read >= token,
        "a causal read never lowers the session's token: {after_read:?} < {token:?}"
    );
}

/// The session's token after a read is that read's view, which the next
/// causal read is bounded by: a second read observes everything the first
/// one saw.
#[tokio::test]
async fn causal_reads_are_monotonic() {
    let server = CoordinodeProcess::start().await;
    let mut client = causal_client(&server).await;

    let (_, written) = client
        .execute_causal_write("CREATE (n:Monotonic {v: 1})")
        .await
        .expect("write");
    let (first, after_first) = client
        .execute_causal_read("MATCH (n:Monotonic) RETURN count(n) AS c", written)
        .await
        .expect("first read");
    let (second, after_second) = client
        .execute_causal_read("MATCH (n:Monotonic) RETURN count(n) AS c", after_first)
        .await
        .expect("second read");
    assert_eq!(first, second, "the second read sees what the first saw");
    assert!(after_second >= after_first, "the token never goes back");
}

/// A token past anything this server committed (another database's clock,
/// a fabricated value) is refused as not ready: the read is never answered
/// from before its bound.
#[tokio::test]
async fn a_token_from_another_clock_is_refused_as_not_ready() {
    let server = CoordinodeProcess::start().await;
    let mut client = CoordinodeClient::connect(server.endpoint())
        .await
        .expect("connect coordinode-client");
    let err = client
        .execute_causal_read("MATCH (n) RETURN n LIMIT 1", CausalToken::from(u64::MAX))
        .await
        .expect_err("a bound never met here");
    let coordinode_client::ClientError::Grpc(status) = err else {
        panic!("expected a server status, got {err:?}");
    };
    assert_eq!(status.code(), tonic::Code::Unavailable, "{status:?}");
    assert!(
        status.message().contains("not ready") || status.message().contains("not met"),
        "{status:?}"
    );
}

/// A client that lets causal reads wait for the member to catch up with
/// their bound.
async fn causal_client(server: &CoordinodeProcess) -> CoordinodeClient {
    CoordinodeClient::builder(server.endpoint())
        .causal_wait(std::time::Duration::from_secs(5))
        .build()
        .await
        .expect("connect coordinode-client")
}

/// `execute_causal_write_with_params` exercises the parameterised variant.
/// Verifies that parameters are correctly forwarded through the causal write path.
#[tokio::test]
async fn causal_write_with_params_round_trip() {
    let server = CoordinodeProcess::start().await;
    let mut client = causal_client(&server).await;

    let mut params = std::collections::HashMap::new();
    params.insert("id".to_string(), Value::Int(8901));

    let (rows, token) = client
        .execute_causal_write_with_params(
            "CREATE (n:CausalParam {id: $id}) RETURN n.id AS id",
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

    let (read_rows, _) = client
        .execute_causal_read_with_params(
            "MATCH (n:CausalParam {id: $id}) RETURN n.id AS id",
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

/// The default token bounds nothing: the first read of a session that has
/// observed nothing is served at once, and returns a token of its own.
#[tokio::test]
async fn zero_token_causal_read_is_accepted() {
    let server = CoordinodeProcess::start().await;

    let mut client = CoordinodeClient::connect(server.endpoint())
        .await
        .expect("connect coordinode-client");

    let (rows, token) = client
        .execute_causal_read("MATCH (n) RETURN n LIMIT 0", CausalToken::default())
        .await
        .expect("zero-token causal read must succeed");

    // LIMIT 0 always returns empty rows regardless.
    assert!(rows.is_empty(), "LIMIT 0 must return no rows");
    assert!(token.as_u64() > 0, "the read reports the time it read at");
}
