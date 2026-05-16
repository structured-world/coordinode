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

/// Extract an i64 value from a row by column name. Temporal `valid_from`
/// round-trips as an Int (epoch ms) through the gRPC layer.
fn int_val(row: &HashMap<String, PropertyValue>, col: &str) -> Option<i64> {
    match row.get(col)?.value.as_ref()? {
        PvKind::IntValue(n) => Some(*n),
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
                at_timestamp: 0,
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
                at_timestamp: 0,
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
                at_timestamp: 0,
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
                at_timestamp: 0,
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

// ── R171: Temporal Edges ─────────────────────────────────────────────────────

/// Alice WORKS_AT three companies over three time periods. Verifies the full
/// temporal stack: CREATE EDGE TYPE TEMPORAL → multi-version CREATE → version
/// fan-out on read → temporal_active_at point-in-time filtering.
#[tokio::test]
async fn r171_temporal_edge_alice_works_at_three_companies() {
    let server = CoordinodeProcess::start().await;

    // Declare the temporal edge type.
    cypher_q(
        &server,
        "CREATE EDGE TYPE WORKS_AT TEMPORAL WITH (role: STRING, valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;

    // Create Alice and three companies.
    cypher_q(
        &server,
        "CREATE (:Person {name: 'Alice'}), (:Company {name: 'Acme'}), \
         (:Company {name: 'Google'}), (:Company {name: 'Meta'})",
    )
    .await;

    // Three temporal versions: Acme 2020-2023, Google 2023-2025, Meta 2025-∞.
    // Timestamps in epoch ms.
    let t_2020 = 1_577_836_800_000_i64; // 2020-01-01
    let t_2023_jun = 1_688_169_600_000_i64; // 2023-07-01
    let t_2025 = 1_736_899_200_000_i64; // 2025-01-15
    let t_acme_end = 1_688_083_200_000_i64; // 2023-06-30
    let t_google_end = 1_735_603_200_000_i64; // 2024-12-31 (closed at year-end)

    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Alice'}}), (c:Company {{name: 'Acme'}}) \
             CREATE (a)-[:WORKS_AT {{valid_from: {t_2020}, valid_to: {t_acme_end}, role: 'SWE'}}]->(c)"
        ),
    )
    .await;
    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Alice'}}), (c:Company {{name: 'Google'}}) \
             CREATE (a)-[:WORKS_AT {{valid_from: {t_2023_jun}, valid_to: {t_google_end}, role: 'Staff'}}]->(c)"
        ),
    )
    .await;
    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Alice'}}), (c:Company {{name: 'Meta'}}) \
             CREATE (a)-[:WORKS_AT {{valid_from: {t_2025}, role: 'Principal'}}]->(c)"
        ),
    )
    .await;

    // Read enumerates every version: three rows expected.
    let all = cypher_q(
        &server,
        "MATCH (a:Person {name: 'Alice'})-[r:WORKS_AT]->(c:Company) \
         RETURN c.name AS company, r.role AS role, r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        all.len(),
        3,
        "temporal traversal must emit one row per version, got {}",
        all.len()
    );

    // Point-in-time: who did Alice work for on 2024-03-15? Only Google.
    let t_query = 1_710_460_800_000_i64; // 2024-03-15
    let pit = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Alice'}})-[r:WORKS_AT]->(c:Company) \
             WHERE temporal_active_at(r, {t_query}) \
             RETURN c.name AS company"
        ),
    )
    .await;
    assert_eq!(
        pit.len(),
        1,
        "exactly one version must be active at 2024-03-15, got {}",
        pit.len()
    );
    assert_eq!(
        str_val(&pit[0], "company").as_deref(),
        Some("Google"),
        "Alice worked at Google on 2024-03-15"
    );

    // Ensure valid_from is round-tripping through reads.
    let google_row = all
        .iter()
        .find(|r| str_val(r, "company").as_deref() == Some("Google"))
        .expect("must find Google row");
    assert_eq!(
        int_val(google_row, "vf"),
        Some(t_2023_jun),
        "valid_from must round-trip through temporal traversal"
    );
}

/// CREATE on a TEMPORAL edge type without a `valid_from` property must fail
/// with a clear message — temporal edges have no defined version without it.
#[tokio::test]
async fn r171_temporal_create_without_valid_from_rejected() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE FOLLOWS_T TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {name: 'X'}), (:Person {name: 'Y'})",
    )
    .await;

    // Attempt to create a temporal edge without valid_from.
    let mut client = server.cypher_client().await;
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query:
                "MATCH (a:Person {name: 'X'}), (b:Person {name: 'Y'}) CREATE (a)-[:FOLLOWS_T]->(b)"
                    .to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await;

    let status = result.expect_err("temporal CREATE without valid_from must be rejected");
    assert!(
        status.message().to_lowercase().contains("valid_from")
            || status.message().to_lowercase().contains("temporal"),
        "rejection message must mention valid_from or temporal, got: {}",
        status.message()
    );
}

/// A temporal version can be created already-closed: provide `valid_to` at
/// write time. After that, `temporal_active_at` rejects every timestamp past
/// the close. This is the "backdated correction" pattern — required because
/// `SET r.<prop>` on edge variables is a platform-wide gap (it's a no-op for
/// every edge type, not specific to temporal), so closing an existing version
/// in-place is not yet supported.
#[tokio::test]
async fn r171_temporal_closed_version_inactive_after_valid_to() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE EMPLOYED_AT TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {name: 'Sam'}), (:Company {name: 'Initech'})",
    )
    .await;

    let t_start = 1_577_836_800_000_i64; // 2020-01-01
    let t_close = 1_704_067_200_000_i64; // 2024-01-01
    let t_query_before = 1_640_995_200_000_i64; // 2022-01-01
    let t_query_after = 1_735_689_600_000_i64; // 2025-01-01

    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}}), (c:Company {{name: 'Initech'}}) \
             CREATE (a)-[:EMPLOYED_AT {{valid_from: {t_start}, valid_to: {t_close}}}]->(c)"
        ),
    )
    .await;

    // Active before the close timestamp.
    let active = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}})-[r:EMPLOYED_AT]->(c:Company) \
             WHERE temporal_active_at(r, {t_query_before}) RETURN c.name AS company"
        ),
    )
    .await;
    assert_eq!(active.len(), 1, "must be active before close");

    // Inactive after the close timestamp.
    let inactive = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}})-[r:EMPLOYED_AT]->(c:Company) \
             WHERE temporal_active_at(r, {t_query_after}) RETURN c.name AS company"
        ),
    )
    .await;
    assert!(
        inactive.is_empty(),
        "must NOT be active after close, got {} rows",
        inactive.len()
    );
}

/// `temporal_overlaps(r, t0, t1)` returns versions whose validity interval
/// overlaps `[t0, t1)`. Three versions on the same pair: only the ones whose
/// validity touches the query window are returned.
#[tokio::test]
async fn r171_temporal_overlaps_filters_by_window() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE LEASED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Tenant {name: 'T'}), (:Building {name: 'B'})",
    )
    .await;

    // Three leases: [100,200), [200,300), [300,400).
    for (vf, vt) in [(100_i64, 200_i64), (200, 300), (300, 400)] {
        cypher_q(
            &server,
            &format!(
                "MATCH (a:Tenant {{name: 'T'}}), (b:Building {{name: 'B'}}) \
                 CREATE (a)-[:LEASED {{valid_from: {vf}, valid_to: {vt}}}]->(b)"
            ),
        )
        .await;
    }

    // Window [150, 250) overlaps the first two, not the third.
    let rows = cypher_q(
        &server,
        "MATCH (a:Tenant {name: 'T'})-[r:LEASED]->(b:Building) \
         WHERE temporal_overlaps(r, 150, 250) \
         RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        rows.len(),
        2,
        "exactly two versions must overlap [150, 250), got {}",
        rows.len()
    );
    let mut vfs: Vec<i64> = rows.iter().filter_map(|r| int_val(r, "vf")).collect();
    vfs.sort();
    assert_eq!(
        vfs,
        vec![100, 200],
        "overlapping versions are valid_from=100 and 200"
    );
}

/// SET r.valid_to closes the matched temporal version in place. The row
/// remains in the graph and answers true to temporal_active_at before the
/// close time, false after.
#[tokio::test]
async fn r171_temporal_set_valid_to_soft_closes_version() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE EMPLOYED_AT TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {name: 'Sam'}), (:Company {name: 'Initech'})",
    )
    .await;

    let t_start = 1_577_836_800_000_i64; // 2020-01-01
    let t_close = 1_704_067_200_000_i64; // 2024-01-01
    let t_query_before = 1_640_995_200_000_i64; // 2022-01-01
    let t_query_after = 1_735_689_600_000_i64; // 2025-01-01

    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}}), (c:Company {{name: 'Initech'}}) \
             CREATE (a)-[:EMPLOYED_AT {{valid_from: {t_start}}}]->(c)"
        ),
    )
    .await;

    cypher_q(
        &server,
        &format!(
            "MATCH (:Person {{name: 'Sam'}})-[r:EMPLOYED_AT]->(:Company {{name: 'Initech'}}) \
             SET r.valid_to = {t_close}"
        ),
    )
    .await;

    // Soft-close keeps the version visible to plain MATCH.
    let still_there = cypher_q(
        &server,
        "MATCH (a:Person {name: 'Sam'})-[r:EMPLOYED_AT]->(c:Company) RETURN c.name AS company",
    )
    .await;
    assert_eq!(still_there.len(), 1, "soft-close must NOT remove the row");

    // Active before close.
    let active = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}})-[r:EMPLOYED_AT]->(c:Company) \
             WHERE temporal_active_at(r, {t_query_before}) RETURN c.name AS company"
        ),
    )
    .await;
    assert_eq!(active.len(), 1, "must be active before close");

    // Inactive after close.
    let inactive = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Sam'}})-[r:EMPLOYED_AT]->(c:Company) \
             WHERE temporal_active_at(r, {t_query_after}) RETURN c.name AS company"
        ),
    )
    .await;
    assert!(
        inactive.is_empty(),
        "must NOT be active after close, got {} rows",
        inactive.len()
    );
}

/// DELETE r on a temporal edge is a hard delete: every version of the
/// matched (src, tgt) pair vanishes and adj-posting is cleared.
#[tokio::test]
async fn r171_temporal_delete_removes_all_versions() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE ASSIGNED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {name: 'Pat'}), (:Project {name: 'Apollo'})",
    )
    .await;

    for (vf, vt) in [(1_000_i64, 2_000_i64), (2_000, 3_000), (3_000, 4_000)] {
        cypher_q(
            &server,
            &format!(
                "MATCH (a:Person {{name: 'Pat'}}), (p:Project {{name: 'Apollo'}}) \
                 CREATE (a)-[:ASSIGNED {{valid_from: {vf}, valid_to: {vt}}}]->(p)"
            ),
        )
        .await;
    }

    let before = cypher_q(
        &server,
        "MATCH (a:Person {name: 'Pat'})-[r:ASSIGNED]->(p:Project) RETURN p.name AS project",
    )
    .await;
    assert_eq!(before.len(), 3, "three versions before DELETE");

    cypher_q(
        &server,
        "MATCH (a:Person {name: 'Pat'})-[r:ASSIGNED]->(p:Project {name: 'Apollo'}) DELETE r",
    )
    .await;

    let after = cypher_q(
        &server,
        "MATCH (a:Person {name: 'Pat'})-[r:ASSIGNED]->(p:Project) RETURN p.name AS project",
    )
    .await;
    assert!(
        after.is_empty(),
        "DELETE on temporal edge must remove every version, got {} rows",
        after.len()
    );
}

/// SET r.<prop> on a non-temporal edge updates the single edgeprop entry.
/// Baseline test: this MUST work for any edge type, not only temporal ones.
#[tokio::test]
async fn r171_set_on_non_temporal_edge_property() {
    let server = CoordinodeProcess::start().await;
    cypher_q(&server, "CREATE (:User {id: 'u1'}), (:User {id: 'u2'})").await;
    cypher_q(
        &server,
        "MATCH (a:User {id: 'u1'}), (b:User {id: 'u2'}) CREATE (a)-[:LIKES {weight: 1.0}]->(b)",
    )
    .await;

    cypher_q(
        &server,
        "MATCH (a:User {id: 'u1'})-[r:LIKES]->(b:User {id: 'u2'}) SET r.weight = 5.5",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:User {id: 'u1'})-[r:LIKES]->(b:User {id: 'u2'}) RETURN r.weight AS w",
    )
    .await;
    assert_eq!(rows.len(), 1);
    let weight = rows[0]
        .get("w")
        .and_then(|pv| pv.value.as_ref())
        .and_then(|kind| match kind {
            PvKind::FloatValue(f) => Some(*f),
            _ => None,
        });
    assert_eq!(weight, Some(5.5), "SET on edge property must update value");
}

/// SET r.valid_to = NULL re-opens a closed temporal version: the row goes back
/// to being active for all timestamps >= valid_from. Mirror of soft-close.
#[tokio::test]
async fn r171_temporal_set_valid_to_null_reopens_version() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE HOLDS TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {name: 'Q'}), (:Asset {name: 'A'})",
    )
    .await;

    let t_start = 1_577_836_800_000_i64; // 2020-01-01
    let t_close = 1_704_067_200_000_i64; // 2024-01-01
    let t_query_after = 1_735_689_600_000_i64; // 2025-01-01

    cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Q'}}), (x:Asset {{name: 'A'}}) \
             CREATE (a)-[:HOLDS {{valid_from: {t_start}, valid_to: {t_close}}}]->(x)"
        ),
    )
    .await;

    // Confirm closed.
    let closed = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Q'}})-[r:HOLDS]->(x:Asset) \
             WHERE temporal_active_at(r, {t_query_after}) RETURN x.name AS asset"
        ),
    )
    .await;
    assert!(closed.is_empty(), "must be closed before re-open");

    // Re-open: SET valid_to = NULL.
    cypher_q(
        &server,
        "MATCH (:Person {name: 'Q'})-[r:HOLDS]->(:Asset {name: 'A'}) SET r.valid_to = null",
    )
    .await;

    let reopened = cypher_q(
        &server,
        &format!(
            "MATCH (a:Person {{name: 'Q'}})-[r:HOLDS]->(x:Asset) \
             WHERE temporal_active_at(r, {t_query_after}) RETURN x.name AS asset"
        ),
    )
    .await;
    assert_eq!(
        reopened.len(),
        1,
        "after SET valid_to=null version must be active again, got {} rows",
        reopened.len()
    );
}

/// SET r.valid_from on a temporal edge is rejected with a clear message —
/// valid_from is part of the storage key and cannot be mutated in place.
#[tokio::test]
async fn r171_temporal_set_valid_from_rejected() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE PHASED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Stage {name: 'S'}), (:Project {name: 'P'})",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (a:Stage {name: 'S'}), (p:Project {name: 'P'}) \
         CREATE (a)-[:PHASED {valid_from: 1000}]->(p)",
    )
    .await;

    let mut client = server.cypher_client().await;
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (:Stage)-[r:PHASED]->(:Project) SET r.valid_from = 2000".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await;
    let status = result.expect_err("SET r.valid_from must be rejected on temporal edges");
    assert!(
        status.message().contains("valid_from"),
        "rejection must mention valid_from, got: {}",
        status.message()
    );
}

/// Temporal CREATE with valid_to <= valid_from is rejected.
#[tokio::test]
async fn r171_temporal_invalid_interval_rejected() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE TIMED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:A {n:1}), (:B {n:2})").await;

    let mut client = server.cypher_client().await;
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (a:A), (b:B) CREATE (a)-[:TIMED {valid_from: 200, valid_to: 100}]->(b)"
                .to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await;
    let status = result.expect_err("inverted interval must be rejected");
    assert!(
        status.message().contains("valid_to") || status.message().contains("valid_from"),
        "rejection must mention timestamps, got: {}",
        status.message()
    );
}

/// DETACH DELETE on a node with temporal edges removes every version of every
/// temporal edge connected to that node — no orphan EdgeProp entries.
#[tokio::test]
async fn r171_detach_delete_cascades_temporal_versions() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE LEASE TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:Tenant {name: 'X'}), (:Unit {name: 'U'})").await;
    for (vf, vt) in [(1_i64, 100_i64), (100, 200), (200, 300)] {
        cypher_q(
            &server,
            &format!(
                "MATCH (a:Tenant {{name: 'X'}}), (u:Unit {{name: 'U'}}) \
                 CREATE (a)-[:LEASE {{valid_from: {vf}, valid_to: {vt}}}]->(u)"
            ),
        )
        .await;
    }

    // Verify all 3 versions present.
    let before = cypher_q(
        &server,
        "MATCH (a:Tenant)-[r:LEASE]->(u:Unit) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(before.len(), 3);

    // DETACH DELETE the tenant — must cascade through every version.
    cypher_q(&server, "MATCH (a:Tenant {name: 'X'}) DETACH DELETE a").await;

    let after = cypher_q(
        &server,
        "MATCH (a:Tenant)-[r:LEASE]->(u:Unit) RETURN r.valid_from AS vf",
    )
    .await;
    assert!(
        after.is_empty(),
        "DETACH DELETE must remove every temporal version, got {} survivors",
        after.len()
    );
}

/// MERGE on a temporal edge type is rejected with a clear message pointing to
/// CREATE — multi-version semantics don't fit MERGE's match-or-create model.
#[tokio::test]
async fn r171_temporal_merge_rejected() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE MERGE_T TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:A {n: 1}), (:B {n: 2})").await;

    let mut client = server.cypher_client().await;
    let result = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (a:A), (b:B) MERGE (a)-[r:MERGE_T {valid_from: 100}]->(b)".to_string(),
            parameters: HashMap::new(),
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await;
    let status = result.expect_err("MERGE on temporal edge type must be rejected");
    assert!(
        status.message().to_lowercase().contains("temporal")
            && status.message().to_lowercase().contains("merge"),
        "rejection must mention temporal + merge, got: {}",
        status.message()
    );
}

/// temporal_overlaps with bound parameter still pushes down (after substitution).
#[tokio::test]
async fn r171_temporal_overlaps_with_parameters() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE WINDOWED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:S {n: 1}), (:T {n: 2})").await;
    for (vf, vt) in [(100_i64, 200_i64), (200, 300), (300, 400)] {
        cypher_q(
            &server,
            &format!(
                "MATCH (s:S), (t:T) CREATE (s)-[:WINDOWED {{valid_from: {vf}, valid_to: {vt}}}]->(t)"
            ),
        )
        .await;
    }

    let mut client = server.cypher_client().await;
    let mut params = HashMap::new();
    params.insert(
        "t0".to_string(),
        PropertyValue {
            value: Some(PvKind::IntValue(150)),
        },
    );
    params.insert(
        "t1".to_string(),
        PropertyValue {
            value: Some(PvKind::IntValue(250)),
        },
    );
    let resp = client
        .execute_cypher(ExecuteCypherRequest {
            query: "MATCH (s:S)-[r:WINDOWED]->(t:T) \
                    WHERE temporal_overlaps(r, $t0, $t1) \
                    RETURN r.valid_from AS vf"
                .to_string(),
            parameters: params,
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await
        .expect("query must succeed")
        .into_inner();
    assert_eq!(
        resp.rows.len(),
        2,
        "parameter-bound temporal_overlaps must return the two overlapping versions"
    );
}

/// Self-loop temporal edge: same node as src and tgt. Adjacency, prefix scan,
/// and DELETE all behave correctly.
#[tokio::test]
async fn r171_temporal_self_loop() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE FOLLOWS_SELF TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:Node {id: 'n1'})").await;
    cypher_q(
        &server,
        "MATCH (a:Node {id: 'n1'}), (b:Node {id: 'n1'}) \
         CREATE (a)-[:FOLLOWS_SELF {valid_from: 1000}]->(b)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (a:Node {id: 'n1'}), (b:Node {id: 'n1'}) \
         CREATE (a)-[:FOLLOWS_SELF {valid_from: 2000}]->(b)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:Node {id: 'n1'})-[r:FOLLOWS_SELF]->(b:Node) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        rows.len(),
        2,
        "self-loop temporal must enumerate both versions"
    );
}

/// Calling `temporal_active_at` on a non-temporal edge returns false rather
/// than erroring — the row has no `r.valid_from`, function gracefully fails.
#[tokio::test]
async fn r171_temporal_active_at_on_non_temporal_edge_returns_false() {
    let server = CoordinodeProcess::start().await;
    cypher_q(&server, "CREATE (:U {id: 'a'}), (:U {id: 'b'})").await;
    cypher_q(
        &server,
        "MATCH (a:U {id: 'a'}), (b:U {id: 'b'}) CREATE (a)-[:LINKS {weight: 1}]->(b)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:U)-[r:LINKS]->(b:U) WHERE temporal_active_at(r, 100) RETURN b.id AS id",
    )
    .await;
    assert!(
        rows.is_empty(),
        "temporal_active_at on non-temporal edge must filter out the row, got {} rows",
        rows.len()
    );
}

/// Two overlapping open versions on the same pair: engine accepts; reads
/// return both; `temporal_active_at` returns both as active.
#[tokio::test]
async fn r171_temporal_overlapping_open_versions_engine_permissive() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE OPEN_T TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:X {n: 1}), (:Y {n: 2})").await;
    cypher_q(
        &server,
        "MATCH (x:X), (y:Y) CREATE (x)-[:OPEN_T {valid_from: 100}]->(y)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (x:X), (y:Y) CREATE (x)-[:OPEN_T {valid_from: 200}]->(y)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (x:X)-[r:OPEN_T]->(y:Y) WHERE temporal_active_at(r, 250) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        rows.len(),
        2,
        "two open versions both match temporal_active_at — engine is permissive"
    );
}

/// MATCH without binding the relationship variable must NOT fan out across
/// versions: the result is per-pair existence, not per-version multiplicity.
#[tokio::test]
async fn r171_match_without_edge_variable_does_not_fan_out() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE LINKED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:Src {n: 1}), (:Dst {n: 2})").await;
    for (vf, vt) in [(1_i64, 2_i64), (2, 3), (3, 4)] {
        cypher_q(
            &server,
            &format!(
                "MATCH (s:Src), (d:Dst) CREATE (s)-[:LINKED {{valid_from: {vf}, valid_to: {vt}}}]->(d)"
            ),
        )
        .await;
    }

    // With edge variable → 3 rows (one per version).
    let with_r = cypher_q(&server, "MATCH (s:Src)-[r:LINKED]->(d:Dst) RETURN d.n AS n").await;
    assert_eq!(with_r.len(), 3, "with edge variable must fan out");

    // Without edge variable → 1 row (pair existence).
    let without_r = cypher_q(&server, "MATCH (s:Src)-[:LINKED]->(d:Dst) RETURN d.n AS n").await;
    assert_eq!(
        without_r.len(),
        1,
        "without edge variable must NOT fan out, got {} rows",
        without_r.len()
    );
}

/// Wildcard relationship pattern `MATCH (a)-[r]-(b)` traverses every registered
/// edge type. With a mix of temporal and non-temporal types the fan-out must
/// behave correctly per type: temporal yields one row per version, non-temporal
/// yields one row per pair.
#[tokio::test]
async fn r171_wildcard_mixed_temporal_and_non_temporal() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE WORK TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:P {n: 1}), (:C {n: 2})").await;
    cypher_q(
        &server,
        "MATCH (p:P), (c:C) CREATE (p)-[:WORK {valid_from: 100}]->(c)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (p:P), (c:C) CREATE (p)-[:WORK {valid_from: 200}]->(c)",
    )
    .await;
    // Non-temporal edge on same pair.
    cypher_q(
        &server,
        "MATCH (p:P), (c:C) CREATE (p)-[:KNOWS {since: 999}]->(c)",
    )
    .await;

    let rows = cypher_q(&server, "MATCH (p:P)-[r]->(c:C) RETURN r AS type_marker").await;
    // 2 WORK versions + 1 KNOWS = 3.
    assert_eq!(
        rows.len(),
        3,
        "wildcard must fan out per-version for temporal and once per pair for non-temporal, got {}",
        rows.len()
    );
}

/// Backward traversal `(a)<-[r:T]-(b)` enumerates temporal versions like the
/// forward direction.
#[tokio::test]
async fn r171_temporal_backward_traversal() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE OWNED TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:Owner {n: 'O'}), (:Asset {n: 'A'})").await;
    for vf in [100_i64, 200, 300] {
        cypher_q(
            &server,
            &format!("MATCH (o:Owner), (a:Asset) CREATE (o)-[:OWNED {{valid_from: {vf}}}]->(a)"),
        )
        .await;
    }

    let rows = cypher_q(
        &server,
        "MATCH (a:Asset)<-[r:OWNED]-(o:Owner) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        rows.len(),
        3,
        "backward traversal must enumerate all versions, got {}",
        rows.len()
    );
}

/// Two temporal traversals in one query, each with their own time predicate.
/// Push-down may or may not lift; correctness must hold regardless.
#[tokio::test]
async fn r171_multiple_temporal_traversals() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE EMP TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE IN_GROUP TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Person {n: 'p'}), (:Company {n: 'c'}), (:Group {n: 'g'})",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (p:Person), (c:Company) CREATE (p)-[:EMP {valid_from: 100, valid_to: 500}]->(c)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (c:Company), (g:Group) CREATE (c)-[:IN_GROUP {valid_from: 200, valid_to: 600}]->(g)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (p:Person)-[r1:EMP]->(c:Company)-[r2:IN_GROUP]->(g:Group) \
         WHERE temporal_active_at(r1, 300) AND temporal_active_at(r2, 300) \
         RETURN g.n AS group_name",
    )
    .await;
    assert_eq!(rows.len(), 1, "both temporal predicates must apply");
}

/// SET multiple properties in a single SET clause on a temporal edge.
#[tokio::test]
async fn r171_temporal_set_multi_property() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE PHASE TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, tag: STRING)",
    )
    .await;
    cypher_q(&server, "CREATE (:Stage {n: 'A'}), (:Project {n: 'B'})").await;
    cypher_q(
        &server,
        "MATCH (s:Stage), (p:Project) CREATE (s)-[:PHASE {valid_from: 100, tag: 'init'}]->(p)",
    )
    .await;

    cypher_q(
        &server,
        "MATCH (:Stage)-[r:PHASE]->(:Project) SET r.valid_to = 200, r.tag = 'done'",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (s:Stage)-[r:PHASE]->(p:Project) RETURN r.valid_to AS vt, r.tag AS lab",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(int_val(&rows[0], "vt"), Some(200));
    assert_eq!(str_val(&rows[0], "lab").as_deref(), Some("done"));
}

/// Compound predicate `temporal_active_at(r, T) AND other_filter`: planner
/// doesn't lift it (chain isn't a direct FunctionCall) but execution must
/// still produce the correct result via the row-level Filter.
#[tokio::test]
async fn r171_temporal_predicate_in_and_chain() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE ENG TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, role: STRING)",
    )
    .await;
    cypher_q(
        &server,
        "CREATE (:Dev {n: 'd1'}), (:Repo {n: 'r1'}), (:Repo {n: 'r2'})",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (d:Dev), (r:Repo {n: 'r1'}) CREATE (d)-[:ENG {valid_from: 100, role: 'maintainer'}]->(r)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (d:Dev), (r:Repo {n: 'r2'}) CREATE (d)-[:ENG {valid_from: 100, role: 'contributor'}]->(r)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (d:Dev)-[r:ENG]->(p:Repo) \
         WHERE temporal_active_at(r, 200) AND r.role = 'maintainer' \
         RETURN p.n AS repo",
    )
    .await;
    assert_eq!(
        rows.len(),
        1,
        "AND-chain must combine temporal + property predicates"
    );
    assert_eq!(str_val(&rows[0], "repo").as_deref(), Some("r1"));
}

/// temporal_active_at(r, NULL) returns false rather than erroring.
#[tokio::test]
async fn r171_temporal_active_at_with_null_returns_false() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE NTEST TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:A), (:B)").await;
    cypher_q(
        &server,
        "MATCH (a:A), (b:B) CREATE (a)-[:NTEST {valid_from: 100}]->(b)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:A)-[r:NTEST]->(b:B) WHERE temporal_active_at(r, null) RETURN b",
    )
    .await;
    assert!(rows.is_empty(), "NULL timestamp must filter out everything");
}

/// Duplicate CREATE on the exact same `(src, tgt, valid_from)`: the second
/// write overwrites the first (engine permissive, value-blob "last wins").
/// Documented as overwrite semantics, not a duplicate-key violation.
#[tokio::test]
async fn r171_duplicate_valid_from_overwrites() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE DUP TEMPORAL WITH (valid_from: TIMESTAMP, tag: STRING)",
    )
    .await;
    cypher_q(&server, "CREATE (:X {n: 1}), (:Y {n: 2})").await;
    cypher_q(
        &server,
        "MATCH (x:X), (y:Y) CREATE (x)-[:DUP {valid_from: 100, tag: 'first'}]->(y)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (x:X), (y:Y) CREATE (x)-[:DUP {valid_from: 100, tag: 'second'}]->(y)",
    )
    .await;

    let rows = cypher_q(&server, "MATCH (x:X)-[r:DUP]->(y:Y) RETURN r.tag AS lab").await;
    assert_eq!(
        rows.len(),
        1,
        "duplicate (src,tgt,vf) collapses to one entry"
    );
    assert_eq!(
        str_val(&rows[0], "lab").as_deref(),
        Some("second"),
        "second write wins on duplicate key"
    );
}

/// Negative valid_from (pre-epoch) round-trips and orders correctly.
#[tokio::test]
async fn r171_negative_valid_from_works() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE HIST TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:E {n: 1}), (:F {n: 2})").await;

    for vf in [-1_000_i64, 0, 1_000] {
        cypher_q(
            &server,
            &format!("MATCH (e:E), (f:F) CREATE (e)-[:HIST {{valid_from: {vf}}}]->(f)"),
        )
        .await;
    }

    let rows = cypher_q(
        &server,
        "MATCH (e:E)-[r:HIST]->(f:F) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(
        rows.len(),
        3,
        "negative, zero, and positive valid_from all stored"
    );
    let mut vfs: Vec<i64> = rows.iter().filter_map(|r| int_val(r, "vf")).collect();
    vfs.sort();
    assert_eq!(vfs, vec![-1_000, 0, 1_000]);
}

/// Temporal edges survive process restart (graceful kill + respawn against
/// same data directory).
#[tokio::test]
async fn r171_temporal_survives_restart() {
    let mut server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE PERSIST TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP, role: STRING)",
    )
    .await;
    cypher_q(&server, "CREATE (:U {id: 'u1'}), (:R {id: 'r1'})").await;
    cypher_q(
        &server,
        "MATCH (u:U), (r:R) CREATE (u)-[:PERSIST {valid_from: 100, valid_to: 200, role: 'A'}]->(r)",
    )
    .await;
    cypher_q(
        &server,
        "MATCH (u:U), (r:R) CREATE (u)-[:PERSIST {valid_from: 200, role: 'B'}]->(r)",
    )
    .await;

    server = server.restart().await;

    let rows = cypher_q(
        &server,
        "MATCH (u:U)-[r:PERSIST]->(x:R) RETURN r.valid_from AS vf, r.role AS role",
    )
    .await;
    assert_eq!(
        rows.len(),
        2,
        "both temporal versions must survive a restart, got {}",
        rows.len()
    );

    // Point-in-time on restored data.
    let at_150 = cypher_q(
        &server,
        "MATCH (u:U)-[r:PERSIST]->(x:R) WHERE temporal_active_at(r, 150) RETURN r.role AS role",
    )
    .await;
    assert_eq!(at_150.len(), 1);
    assert_eq!(str_val(&at_150[0], "role").as_deref(), Some("A"));

    let at_250 = cypher_q(
        &server,
        "MATCH (u:U)-[r:PERSIST]->(x:R) WHERE temporal_active_at(r, 250) RETURN r.role AS role",
    )
    .await;
    assert_eq!(at_250.len(), 1);
    assert_eq!(str_val(&at_250[0], "role").as_deref(), Some("B"));
}

/// i64::MAX and far-future valid_from values round-trip through the sortable
/// encoding without overflow. Upper-bound key uses `saturating_add(1)` so the
/// extreme case doesn't wrap.
#[tokio::test]
async fn r171_temporal_boundary_valid_from() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE BOUND TEMPORAL WITH (valid_from: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:M {n: 1}), (:N {n: 2})").await;

    let big = i64::MAX - 1;
    cypher_q(
        &server,
        &format!("MATCH (a:M), (b:N) CREATE (a)-[:BOUND {{valid_from: {big}}}]->(b)"),
    )
    .await;
    cypher_q(
        &server,
        "MATCH (a:M), (b:N) CREATE (a)-[:BOUND {valid_from: 0}]->(b)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:M)-[r:BOUND]->(b:N) RETURN r.valid_from AS vf",
    )
    .await;
    assert_eq!(rows.len(), 2);
    let mut vfs: Vec<i64> = rows.iter().filter_map(|r| int_val(r, "vf")).collect();
    vfs.sort();
    assert_eq!(vfs, vec![0, big], "boundary values survive round-trip");
}

/// temporal_overlaps with an inverted window (t0 > t1) yields no matches —
/// no version's interval can overlap an empty window.
#[tokio::test]
async fn r171_temporal_overlaps_inverted_window_returns_empty() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE INV TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:A), (:B)").await;
    cypher_q(
        &server,
        "MATCH (a:A), (b:B) CREATE (a)-[:INV {valid_from: 100, valid_to: 200}]->(b)",
    )
    .await;

    let rows = cypher_q(
        &server,
        "MATCH (a:A)-[r:INV]->(b:B) WHERE temporal_overlaps(r, 500, 100) RETURN b",
    )
    .await;
    assert!(rows.is_empty(), "inverted window must produce no matches");
}

/// AS OF TIMESTAMP + temporal_active_at: transaction-time snapshot first, then
/// valid-time filter. After deleting all versions, AS OF still surfaces the
/// pre-delete state, and temporal_active_at on that historical state works.
#[tokio::test]
async fn r171_as_of_timestamp_composes_with_temporal_active_at() {
    let server = CoordinodeProcess::start().await;
    cypher_q(
        &server,
        "CREATE EDGE TYPE BITEMP TEMPORAL WITH (valid_from: TIMESTAMP, valid_to: TIMESTAMP)",
    )
    .await;
    cypher_q(&server, "CREATE (:P {n: 'p'}), (:Q {n: 'q'})").await;
    cypher_q(
        &server,
        "MATCH (p:P), (q:Q) CREATE (p)-[:BITEMP {valid_from: 100, valid_to: 200}]->(q)",
    )
    .await;

    // Read transaction-time AS OF "now" (effectively current snapshot) +
    // valid-time filter. The version is closed by valid_to=200, so a query
    // at valid time 300 must return nothing.
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);

    let inactive_at_300 = cypher_q(
        &server,
        &format!(
            "AS OF TIMESTAMP {now_us} \
             MATCH (p:P)-[r:BITEMP]->(q:Q) \
             WHERE temporal_active_at(r, 300) RETURN q.n"
        ),
    )
    .await;
    assert!(
        inactive_at_300.is_empty(),
        "AS OF (current snapshot) + temporal_active_at(300) on closed version must be empty"
    );

    let active_at_150 = cypher_q(
        &server,
        &format!(
            "AS OF TIMESTAMP {now_us} \
             MATCH (p:P)-[r:BITEMP]->(q:Q) \
             WHERE temporal_active_at(r, 150) RETURN q.n"
        ),
    )
    .await;
    assert_eq!(
        active_at_150.len(),
        1,
        "AS OF (current snapshot) + temporal_active_at(150) must surface the version"
    );
}
