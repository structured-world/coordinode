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
//! | `bug5_match_set_persists_across_queries` | Bug5 | MATCH+SET change must be visible in subsequent MATCH RETURN (no index) |
//! | `bug5_match_set_persists_with_btree_index` | Bug5 | MATCH+SET change must be visible in subsequent MATCH RETURN (with B-tree index) |
//! | `bug_latent_set_after_delete_must_not_show_unwritten_value` | LatentBug | DELETE+SET: RETURN must not show value when write was not applied (out_row.insert outside if-let block) |
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
use coordinode_integration::proto::common::{property_value::Value as PvKind, PropertyValue};
use coordinode_integration::proto::query::{ExecuteCypherRequest, Row};
use coordinode_integration::proto::replication::{ReadConcern, WriteConcern, WriteConcernLevel};

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Execute a Cypher query and return rows as column-name → PropertyValue maps.
async fn cypher_q(proc: &CoordinodeProcess, query: &str) -> Vec<HashMap<String, PropertyValue>> {
    let mut client = proc.cypher_client().await;
    let resp = client
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await
        .expect("execute_cypher must succeed")
        .into_inner();

    let columns = resp.columns;
    resp.rows
        .into_iter()
        .map(|Row { values }| {
            columns
                .iter()
                .zip(values)
                .map(|(col, v)| (col.clone(), v))
                .collect::<HashMap<_, _>>()
        })
        .collect()
}

/// Extract a string value from a row by column name.
fn str_val(row: &HashMap<String, PropertyValue>, col: &str) -> Option<String> {
    match row.get(col)?.value.as_ref()? {
        PvKind::StringValue(s) => Some(s.clone()),
        _ => None,
    }
}

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

// ── Bug5: MATCH+SET property change must persist across query boundaries ──────

/// Regression test (Bug5): MATCH+SET must be visible in a subsequent query.
///
/// Exact reproduction of the production bug report (v0.3.14/0.3.15):
///   Step 1: MERGE (p:Project {id: "x"}) SET p.status = "active"  → creates node
///   Step 2: MATCH (p:Project {id: "x"}) SET p.status = "removed" RETURN p.status → "removed" ✓
///   Step 3: MATCH (p:Project {id: "x"}) RETURN p.status → must return "removed" (bug: returned "active")
///
/// Uses NodeScan path (no B-tree index).
#[tokio::test]
async fn bug5_match_set_persists_across_queries() {
    let server = CoordinodeProcess::start().await;

    // Step 1: create node via MERGE+SET.
    cypher_q(
        &server,
        "MERGE (p:Bug5Test {id: 'x'}) SET p.status = 'active'",
    )
    .await;

    // Step 2: update via MATCH+SET — must return "removed" within the same query.
    let step2 = cypher_q(
        &server,
        "MATCH (p:Bug5Test {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s",
    )
    .await;
    assert_eq!(step2.len(), 1, "step2 must return exactly one row");
    assert_eq!(
        str_val(&step2[0], "s").as_deref(),
        Some("removed"),
        "MATCH+SET must be visible within the same query (step 2)"
    );

    // Step 3: new query — the change must persist.
    let step3 = cypher_q(&server, "MATCH (p:Bug5Test {id: 'x'}) RETURN p.status AS s").await;
    assert_eq!(step3.len(), 1, "step3 must return exactly one row");
    assert_eq!(
        str_val(&step3[0], "s").as_deref(),
        Some("removed"),
        "MATCH+SET must persist across query boundaries: \
         expected 'removed', got {:?}",
        str_val(&step3[0], "s")
    );
}

// ── Latent bug: RETURN must not show value when write was not applied ─────────

/// Regression test (latent): RETURN must not expose a SET value that was never written.
///
/// Exact reproduction path:
///   1. CREATE a node.
///   2. In one query: MATCH (n) DELETE n SET n.status = 'ghost' RETURN n.status
///      - DELETE puts a tombstone (None) in the MVCC write buffer.
///      - SET: `ctx.mvcc_get(Partition::Node, &key)` returns None (tombstone hit).
///      - The `if let Some(bytes) = ctx.mvcc_get(...)` block is skipped → no `mvcc_put`.
///      - Bug: `out_row.insert(..., val)` at runner.rs is OUTSIDE that block → RETURN
///        shows 'ghost' even though the write was never applied to storage.
///      - Fix: `out_row.insert` must live inside the `if let Some` block.
///   3. MATCH the node → must return 0 rows (node is deleted, 'ghost' was never written).
///
/// Invariant tested: RETURN must only reflect writes that were actually committed to storage.
#[tokio::test]
async fn bug_latent_set_after_delete_must_not_show_unwritten_value() {
    let server = CoordinodeProcess::start().await;

    // Step 1: create node.
    cypher_q(
        &server,
        "MERGE (n:LatentBug {id: 'x'}) SET n.status = 'original'",
    )
    .await;

    // Step 2: DELETE then SET in the same query.
    // DELETE puts a tombstone; SET's mvcc_get returns None → write is skipped.
    // Bug: out_row.insert still runs → RETURN shows 'ghost' (unwritten value).
    let mut client = server.cypher_client().await;
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (n:LatentBug {id: 'x'}) \
                    DELETE n \
                    SET n.status = 'ghost' \
                    RETURN n.status AS s"
                .to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await;

    // Either the query errors (acceptable — SET on deleted node is invalid) or it
    // succeeds but must NOT return 'ghost' in any row.
    if let Ok(resp) = result {
        let resp = resp.into_inner();
        let columns = resp.columns;
        let rows: Vec<HashMap<String, PropertyValue>> = resp
            .rows
            .into_iter()
            .map(|Row { values }| {
                columns
                    .iter()
                    .zip(values)
                    .map(|(col, v)| (col.clone(), v))
                    .collect::<HashMap<_, _>>()
            })
            .collect();

        for row in &rows {
            assert_ne!(
                str_val(row, "s").as_deref(),
                Some("ghost"),
                "RETURN must not show 'ghost' — SET was not applied (node was deleted)",
            );
        }
    }
    // If the query returned an error that is also acceptable: node deleted, SET invalid.

    // Step 3: node must not exist — DELETE must have been applied.
    let after = cypher_q(
        &server,
        "MATCH (n:LatentBug {id: 'x'}) RETURN n.status AS s",
    )
    .await;
    assert_eq!(
        after.len(),
        0,
        "node must be gone after DELETE — expected 0 rows, got {:?}",
        after.iter().map(|r| str_val(r, "s")).collect::<Vec<_>>()
    );
}

/// Regression test (Bug5): MATCH+SET must persist when a B-tree index is present.
///
/// Same as `bug5_match_set_persists_across_queries` but with an index on :Bug5Idx(id),
/// forcing the executor to use the IndexScan path instead of NodeScan.
/// The B-tree index write in `on_property_changed` (Partition::Idx) must not
/// interfere with the Node partition write buffered in the MVCC write buffer.
#[tokio::test]
async fn bug5_match_set_persists_with_btree_index() {
    let server = CoordinodeProcess::start().await;

    // Create a B-tree index to force the IndexScan path.
    cypher_q(&server, "CREATE INDEX idx_bug5_id ON :Bug5Idx(id)").await;

    // Step 1: create node (index entry is also created).
    cypher_q(
        &server,
        "MERGE (p:Bug5Idx {id: 'x'}) SET p.status = 'active'",
    )
    .await;

    // Step 2: update via MATCH+SET using the index.
    let step2 = cypher_q(
        &server,
        "MATCH (p:Bug5Idx {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s",
    )
    .await;
    assert_eq!(step2.len(), 1, "step2 must return exactly one row");
    assert_eq!(
        str_val(&step2[0], "s").as_deref(),
        Some("removed"),
        "MATCH+SET (IndexScan) must be visible within the same query (step 2)"
    );

    // Step 3: new query via IndexScan — the change must persist.
    let step3 = cypher_q(&server, "MATCH (p:Bug5Idx {id: 'x'}) RETURN p.status AS s").await;
    assert_eq!(step3.len(), 1, "step3 must return exactly one row");
    assert_eq!(
        str_val(&step3[0], "s").as_deref(),
        Some("removed"),
        "MATCH+SET (IndexScan) must persist across query boundaries: \
         expected 'removed', got {:?}",
        str_val(&step3[0], "s")
    );
}
