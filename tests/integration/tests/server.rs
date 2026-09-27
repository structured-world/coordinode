//! Server startup and response-header integration tests.
//!
//! These tests exercise the multi-protocol handler, `NodeInfoLayer` response
//! headers, and CE/EE mode validation against a real `coordinode` binary.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test server
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use coordinode_integration::harness::{CoordinodeProcess, binary_path};
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
            transaction_id: 0,
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
/// - `x-coordinode-hops` is "0" for a request this node served itself
/// - `x-coordinode-load` is "0": the server does not track load yet
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

    // A single node serves the request itself: no forwarding hop.
    let hops = meta.get("x-coordinode-hops").unwrap().to_str().unwrap();
    assert_eq!(hops, "0", "a locally served request must report 0 hops");

    // Load is not tracked yet, so the header is always 0.
    let load = meta.get("x-coordinode-load").unwrap().to_str().unwrap();
    assert_eq!(load, "0", "load is not tracked, the header must be 0");

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

/// A server whose gRPC port is taken must refuse to start before it opens
/// storage: the data directory stays untouched and the error names the
/// address, instead of the store being opened (and possibly recovered) only
/// to be abandoned when the late bind fails.
#[test]
fn busy_grpc_port_fails_the_start_before_storage_opens() {
    let taken = std::net::TcpListener::bind("[::1]:0").expect("hold a port");
    let addr = taken.local_addr().expect("held address");
    let root = tempfile::tempdir().expect("tempdir");
    let data_dir = root.path().join("data");

    let output = std::process::Command::new(binary_path())
        .args(["serve", "--ops-addr", "[::1]:0", "--addr"])
        .arg(addr.to_string())
        .arg("--data")
        .arg(&data_dir)
        .output()
        .expect("failed to spawn coordinode binary");

    assert!(
        !output.status.success(),
        "a taken gRPC port must fail the start, got: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(&addr.to_string()),
        "the error must name the taken address {addr}. Got stderr: {stderr}"
    );
    assert!(
        !data_dir.exists(),
        "storage must not be opened when the gRPC port is taken"
    );
}

/// Start `serve` with every listener on a free port except `flag`, which gets
/// a port held here, and return the exit status, stderr, the held address and
/// whether the data directory was created. A server still running after 30 s
/// has not refused the start: it is killed and the test fails on that.
fn serve_with_one_port_taken(flag: &str) -> (std::process::ExitStatus, String, String, bool) {
    use std::io::Read;
    let taken = std::net::TcpListener::bind("[::1]:0").expect("hold a port");
    let held = taken.local_addr().expect("held address").to_string();
    let root = tempfile::tempdir().expect("tempdir");
    let data_dir = root.path().join("data");

    let mut cmd = std::process::Command::new(binary_path());
    cmd.arg("serve");
    for listener in ["--addr", "--ops-addr", "--rest-addr"] {
        cmd.arg(listener);
        if listener == flag {
            cmd.arg(&held);
        } else {
            cmd.arg("[::1]:0");
        }
    }
    let mut child = cmd
        .arg("--data")
        .arg(&data_dir)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn coordinode binary");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the server") {
            break status;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("the server kept running with {flag} {held} taken instead of refusing to start");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .expect("piped stderr")
        .read_to_string(&mut stderr)
        .expect("read stderr");
    (status, stderr, held, data_dir.exists())
}

/// A taken ops port fails the start like a taken gRPC port. A node running
/// without its ops listener has no /ready of its own, and a health check
/// against that port would get its answer from whatever holds it.
#[test]
fn busy_ops_port_fails_the_start_before_storage_opens() {
    let (status, stderr, held, opened) = serve_with_one_port_taken("--ops-addr");
    assert!(!status.success(), "a taken ops port must fail the start");
    assert!(
        stderr.contains(&held),
        "the error must name the taken address {held}. Got stderr: {stderr}"
    );
    assert!(
        !opened,
        "storage must not be opened when the ops port is taken"
    );
}

/// A taken REST port fails the start: a node that silently serves without its
/// REST API looks healthy to every probe while clients of that API get
/// nothing, or reach whatever else holds the port.
#[test]
fn busy_rest_port_fails_the_start_before_storage_opens() {
    let (status, stderr, held, opened) = serve_with_one_port_taken("--rest-addr");
    assert!(!status.success(), "a taken REST port must fail the start");
    assert!(
        stderr.contains(&held),
        "the error must name the taken address {held}. Got stderr: {stderr}"
    );
    assert!(
        !opened,
        "storage must not be opened when the REST port is taken"
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
