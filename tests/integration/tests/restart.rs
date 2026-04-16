//! Standalone restart regression tests.
//!
//! Each test exercises a bug that was reported against the gRPC/standalone
//! code path (NOT reproducible via coordinode-embed because the embed API
//! bypasses proto serialisation for schema creation).
//!
//! ## Test matrix
//!
//! | Test | Bug | Scenario |
//! |------|-----|---------|
//! | `bug2_vector_write_before_restart_also_works` | Bug2 | VECTOR schema created via gRPC gets dimensions=0. Writing a vector fails with "dimension mismatch: expected 0, got N". |
//! | `bug2_vector_zero_dimensions_survives_restart` | Bug2 | After restart, vector write still fails with the same dimension mismatch. |
//! | `bug1_merge_unique_no_restart` | Bug1 | MERGE on existing unique node must not throw a unique constraint error. |
//! | `bug1_merge_unique_after_restart` | Bug1 | Same, but the node was created before a restart. |
//! | `bug3_hnsw_flexible_rebuilt_after_restart` | Bug3 | After restart, vector similarity queries on Flexible-mode labels return correct results. |
//! | `bug4_flexible_match_invisible_after_restart` | Bug4 | After restart, MATCH with property filter returns 0 in FLEXIBLE mode even though the node exists (unique constraint rejects duplicate CREATE). Label scan also returns 0. |
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_integration::harness::CoordinodeProcess;
use coordinode_integration::proto::common::{
    property_value::Value as PvKind, PropertyValue, Vector,
};
use coordinode_integration::proto::graph::{
    CreateLabelRequest, PropertyDefinition, PropertyType, SchemaMode,
};
use coordinode_integration::proto::query::{ExecuteCypherRequest, Row};
use std::collections::HashMap;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Execute a Cypher query and return rows as column-name → PropertyValue maps.
async fn cypher(
    proc: &CoordinodeProcess,
    query: &str,
    params: HashMap<String, PropertyValue>,
) -> Result<Vec<HashMap<String, PropertyValue>>, tonic::Status> {
    let mut client = proc.cypher_client().await;
    let resp = client
        .execute_cypher(ExecuteCypherRequest {
            query: query.to_string(),
            parameters: params,
            // PRIMARY / LOCAL defaults — sufficient for regression tests.
            read_preference: 0,
            read_concern: None,
            write_concern: None,
        })
        .await?
        .into_inner();

    let columns = resp.columns;
    let rows = resp
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
    Ok(rows)
}

/// Execute a Cypher query with no parameters.
async fn cypher_q(
    proc: &CoordinodeProcess,
    query: &str,
) -> Result<Vec<HashMap<String, PropertyValue>>, tonic::Status> {
    cypher(proc, query, HashMap::new()).await
}

/// Build a `PropertyValue` wrapping a `Vec<f32>` (VECTOR).
fn pv_vector(values: Vec<f32>) -> PropertyValue {
    PropertyValue {
        value: Some(PvKind::VectorValue(Vector { values })),
    }
}

/// Build a `PropertyValue` wrapping a string.
fn pv_string(s: &str) -> PropertyValue {
    PropertyValue {
        value: Some(PvKind::StringValue(s.to_string())),
    }
}

// ── Bug2: vector dimension mismatch ──────────────────────────────────────────

/// Bug2 — first write fails (no restart needed).
///
/// Root cause: `proto_type_to_property_type(PROPERTY_TYPE_VECTOR=7)` hardcodes
/// `dimensions = 0`. Validation then rejects any vector write because
/// `vec.len() != 0`.
///
/// Fix: treat `dimensions = 0` as "unset/any" sentinel — skip the check.
#[tokio::test]
async fn bug2_vector_write_before_restart_also_works() {
    let proc = CoordinodeProcess::start().await;

    // Create label with VECTOR property via gRPC SchemaService.
    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "VecNode".to_string(),
        properties: vec![PropertyDefinition {
            name: "emb".to_string(),
            r#type: PropertyType::Vector as i32,
            required: false,
            unique: false,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Strict as i32,
    })
    .await
    .expect("create_label");

    // Write a 4-dimensional vector — must succeed even though schema has dimensions=0.
    let mut params = HashMap::new();
    params.insert("vec".to_string(), pv_vector(vec![0.1, 0.2, 0.3, 0.4]));
    let result = cypher(&proc, "CREATE (n:VecNode {emb: $vec}) RETURN n", params).await;
    assert!(
        result.is_ok(),
        "Bug2: vector write must succeed when schema has dimensions=0. Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 created node");
}

/// Bug2 — vector write fails AFTER restart.
///
/// The dimensions=0 schema persists to msgpack on disk. After restart, the
/// schema is reloaded with dimensions=0 and the same validation failure occurs.
#[tokio::test]
async fn bug2_vector_zero_dimensions_survives_restart() {
    let proc = CoordinodeProcess::start().await;

    // Step 1: create schema + write initial vector BEFORE restart.
    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "VecRestart".to_string(),
        properties: vec![PropertyDefinition {
            name: "emb".to_string(),
            r#type: PropertyType::Vector as i32,
            required: false,
            unique: false,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Strict as i32,
    })
    .await
    .expect("create_label");

    let mut params = HashMap::new();
    params.insert("vec".to_string(), pv_vector(vec![1.0, 0.0, 0.0]));
    cypher(&proc, "CREATE (n:VecRestart {emb: $vec})", params)
        .await
        .expect("first write before restart");

    // Step 2: restart the process (same data dir).
    let proc = proc.restart().await;

    // Step 3: write another vector after restart — must NOT fail with dimension mismatch.
    let mut params = HashMap::new();
    params.insert("vec".to_string(), pv_vector(vec![0.0, 1.0, 0.0]));
    let result = cypher(&proc, "CREATE (n:VecRestart {emb: $vec})", params).await;
    assert!(
        result.is_ok(),
        "Bug2: vector write after restart must succeed. Got: {:?}",
        result.err()
    );

    // Step 4: verify both nodes are readable.
    let rows = cypher_q(&proc, "MATCH (n:VecRestart) RETURN n")
        .await
        .expect("match after restart");
    assert_eq!(rows.len(), 2, "both nodes must survive restart");
}

// ── Bug1: MERGE unique constraint ─────────────────────────────────────────────

/// Bug1 — MERGE on an existing unique node throws a unique constraint error
/// instead of performing an ON MATCH update.
#[tokio::test]
async fn bug1_merge_unique_no_restart() {
    let proc = CoordinodeProcess::start().await;

    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "UniqueNode".to_string(),
        properties: vec![PropertyDefinition {
            name: "id".to_string(),
            r#type: PropertyType::String as i32,
            required: true,
            unique: true,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Strict as i32,
    })
    .await
    .expect("create_label");

    // Create the initial node.
    let mut params = HashMap::new();
    params.insert("val".to_string(), pv_string("node-x1"));
    cypher(&proc, "CREATE (n:UniqueNode {id: $val})", params)
        .await
        .expect("create node");

    // MERGE on the same key must not raise a unique constraint violation.
    let mut params = HashMap::new();
    params.insert("val".to_string(), pv_string("node-x1"));
    let result = cypher(
        &proc,
        "MERGE (n:UniqueNode {id: $val}) ON MATCH SET n.id = $val RETURN n.id",
        params,
    )
    .await;
    assert!(
        result.is_ok(),
        "Bug1: MERGE on existing unique node must succeed (ON MATCH). Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 matched node");
}

/// Bug1 — same as above but after a restart (tests schema reload path).
#[tokio::test]
async fn bug1_merge_unique_after_restart() {
    let proc = CoordinodeProcess::start().await;

    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "UniqueRestart".to_string(),
        properties: vec![PropertyDefinition {
            name: "id".to_string(),
            r#type: PropertyType::String as i32,
            required: true,
            unique: true,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Strict as i32,
    })
    .await
    .expect("create_label");

    // Create node before restart.
    let mut params = HashMap::new();
    params.insert("val".to_string(), pv_string("restart-key-1"));
    cypher(&proc, "CREATE (n:UniqueRestart {id: $val})", params)
        .await
        .expect("create before restart");

    // Restart.
    let proc = proc.restart().await;

    // MERGE after restart must not throw unique constraint.
    let mut params = HashMap::new();
    params.insert("val".to_string(), pv_string("restart-key-1"));
    let result = cypher(
        &proc,
        "MERGE (n:UniqueRestart {id: $val}) ON MATCH SET n.id = $val RETURN n.id",
        params,
    )
    .await;
    assert!(
        result.is_ok(),
        "Bug1: MERGE on existing unique node after restart must succeed. Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 matched node");
}

// ── Bug3: HNSW not rebuilt for Flexible mode after restart ────────────────────

/// Bug3 — after restart, vector similarity search on Flexible-mode labels
/// returns no results.
///
/// Root cause (diagnosed via this test): the gRPC server path uses
/// `LocalProposalPipeline` which has no WAL. When the process was killed via
/// SIGKILL the LSM memtable was never flushed to SST files → nodes disappeared
/// entirely after restart. The symptom looked like an HNSW rebuild failure,
/// but the data itself was gone.
///
/// Fix: added SIGTERM handler (`serve_with_shutdown`) to the server so that
/// graceful shutdown flushes memtables via `StorageEngine::Drop`. The test
/// harness now sends SIGTERM and waits for the process to exit before
/// restarting.
#[tokio::test]
async fn bug3_hnsw_flexible_rebuilt_after_restart() {
    let proc = CoordinodeProcess::start().await;

    // Create a Flexible-mode label with a VECTOR property.
    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "FlexVec".to_string(),
        properties: vec![PropertyDefinition {
            name: "emb".to_string(),
            r#type: PropertyType::Vector as i32,
            required: false,
            unique: false,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Flexible as i32,
    })
    .await
    .expect("create_label");

    // Insert nodes with vectors before restart.
    for i in 0u32..5 {
        let mut params = HashMap::new();
        let v = i as f32 / 4.0;
        params.insert("vec".to_string(), pv_vector(vec![v, 1.0 - v, 0.5, 0.5]));
        cypher(&proc, "CREATE (n:FlexVec {emb: $vec})", params)
            .await
            .expect("create flex node");
    }

    // Verify vector search works before restart.
    let mut params = HashMap::new();
    params.insert("qvec".to_string(), pv_vector(vec![0.0, 1.0, 0.5, 0.5]));
    let rows_before = cypher(
        &proc,
        "MATCH (n:FlexVec) RETURN vector_similarity(n.emb, $qvec) AS score ORDER BY score DESC LIMIT 3",
        params,
    )
    .await
    .expect("vector search before restart");
    assert!(
        !rows_before.is_empty(),
        "Bug3: vector search must return results before restart"
    );

    // Restart.
    let proc = proc.restart().await;

    // Diagnostic: verify nodes still exist after restart.
    let count_rows = cypher_q(&proc, "MATCH (n:FlexVec) RETURN count(n) AS cnt")
        .await
        .expect("count after restart");
    let node_count = count_rows
        .first()
        .and_then(|r| r.get("cnt"))
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        node_count,
        coordinode_integration::proto::common::PropertyValue {
            value: Some(coordinode_integration::proto::common::property_value::Value::IntValue(5))
        },
        "Bug3 diagnostic: 5 nodes must still exist after restart, got {node_count:?}"
    );

    // Diagnostic: verify n.emb is readable after restart.
    let emb_rows = cypher_q(&proc, "MATCH (n:FlexVec) RETURN n.emb LIMIT 1")
        .await
        .expect("emb read after restart");
    assert!(
        !emb_rows.is_empty(),
        "Bug3 diagnostic: MATCH FlexVec must return at least 1 row after restart"
    );

    // Vector search must still return results after restart.
    let mut params = HashMap::new();
    params.insert("qvec".to_string(), pv_vector(vec![0.0, 1.0, 0.5, 0.5]));
    let rows_after = cypher(
        &proc,
        "MATCH (n:FlexVec) RETURN vector_similarity(n.emb, $qvec) AS score ORDER BY score DESC LIMIT 3",
        params,
    )
    .await
    .expect("vector search after restart");
    assert!(
        !rows_after.is_empty(),
        "Bug3: vector search must return results after restart (HNSW must be rebuilt). Got 0 rows.\
         \nNote: 5 nodes exist (checked above), n.emb readable (checked above).\
         \nFailing in VectorTopK or vector_similarity evaluation."
    );
    assert_eq!(
        rows_after.len(),
        rows_before.len(),
        "result count must be same before and after restart"
    );
}

// ── Bug4: MATCH invisible after restart in FLEXIBLE mode ──────────────────────

// ── SIGKILL restart: crash recovery (no graceful shutdown) ───────────────────

/// Verify that CoordiNode survives an unclean shutdown (SIGKILL) and restarts
/// without crashing.
///
/// Root cause of the crash: after an unclean shutdown the Raft oplog segment
/// file exists on disk but the LSM key `raft:oplog:last_log_id` was not
/// flushed before process death.  On restart, `LogStore::open()` saw
/// `last_log_id = None`, openraft treated the log as empty, called
/// `initialize()`, and tried to create `oplog-0000.bin` with `create_new`
/// semantics — failing with EEXIST (os error 17).
///
/// Fix: `LogStore::open()` now reconstructs `last_log_id` from existing oplog
/// segment files when the LSM key is absent.
///
/// This test exercises the end-to-end path: write data → SIGKILL → restart.
/// Because SIGKILL may lose in-flight memtable data, we don't assert on the
/// node count after restart — only that the server starts and accepts queries.
#[tokio::test]
async fn bug5_sigkill_restart_survives_without_crash() {
    let proc = CoordinodeProcess::start().await;

    // Write several nodes to ensure the Raft log has entries fsynced.
    for i in 0u32..5 {
        let mut params = HashMap::new();
        params.insert("i".to_string(), pv_string(&format!("crash-{i}")));
        let _ = cypher(&proc, "CREATE (n:CrashTest {id: $i})", params).await;
    }

    // Unclean shutdown — SIGKILL, no graceful flush.
    let proc = proc.restart_unclean().await;

    // The server must start and accept a query (not crash with EEXIST).
    // We don't assert node count because memtable may not have been flushed.
    let result = cypher_q(&proc, "MATCH (n:CrashTest) RETURN count(n) AS cnt").await;
    assert!(
        result.is_ok(),
        "crash recovery: server must start after SIGKILL and accept queries; \
         got error: {:?}",
        result.err()
    );
}

/// Bug4 — after restart, `MATCH (n:Label {prop: $val})` returns 0 results in
/// FLEXIBLE schema mode, even though the node demonstrably exists.
///
/// Evidence of existence:
///   - `CREATE (n:Label {prop: $val})` → unique constraint violation (index sees it)
///   - `MERGE (n:Label {prop: $val})` → succeeds and returns the node (MERGE uses
///     the unique-index lookup path)
///
/// Full label scan `MATCH (n:Label) DETACH DELETE n` also returns 0 → indicates
/// the label-level node roster is not rebuilt on restart, leaving nodes
/// "constraint-visible" but "query-invisible".
///
/// Workaround (cxbr-common): use `MERGE … ON MATCH SET … ON CREATE SET …`
/// instead of `MATCH … SET` / `CREATE`.
///
/// Root cause hypothesis: the in-memory label scan index is not repopulated from
/// LSM on server startup in FLEXIBLE mode (STRICT mode is unaffected because
/// Bug1/Bug2 are already fixed there). The unique constraint index is stored
/// durably in LSM and rebuilt correctly; the label scan uses a different path.
#[tokio::test]
async fn bug4_flexible_match_invisible_after_restart() {
    let proc = CoordinodeProcess::start().await;

    // 1. Create a FLEXIBLE label with one unique declared property.
    let mut sc = proc.schema_client().await;
    sc.create_label(CreateLabelRequest {
        name: "FlexPersist".to_string(),
        properties: vec![PropertyDefinition {
            name: "key".to_string(),
            r#type: PropertyType::String as i32,
            required: false,
            unique: true,
        }],
        computed_properties: vec![],
        schema_mode: SchemaMode::Flexible as i32,
    })
    .await
    .expect("create_label");

    // 2. Create a node with an extra (non-schema) property — exercises FLEXIBLE path.
    let mut params = HashMap::new();
    params.insert("k".to_string(), pv_string("fp-key-1"));
    cypher(
        &proc,
        "CREATE (s:FlexPersist {key: $k, extra: 'data'})",
        params,
    )
    .await
    .expect("create node before restart");

    // Sanity: node is visible before restart.
    let mut params = HashMap::new();
    params.insert("k".to_string(), pv_string("fp-key-1"));
    let before = cypher(
        &proc,
        "MATCH (s:FlexPersist {key: $k}) RETURN count(s) AS cnt",
        params,
    )
    .await
    .expect("match before restart");
    assert_eq!(
        before.first().and_then(|r| r.get("cnt")).cloned(),
        Some(PropertyValue {
            value: Some(PvKind::IntValue(1))
        }),
        "sanity: node must be visible before restart"
    );

    // 3. Restart.
    let proc = proc.restart().await;

    // 4. MATCH with property filter must still find the node.
    let mut params = HashMap::new();
    params.insert("k".to_string(), pv_string("fp-key-1"));
    let match_rows = cypher(
        &proc,
        "MATCH (s:FlexPersist {key: $k}) RETURN count(s) AS cnt",
        params,
    )
    .await
    .expect("match with property filter after restart");
    assert_eq!(
        match_rows.first().and_then(|r| r.get("cnt")).cloned(),
        Some(PropertyValue {
            value: Some(PvKind::IntValue(1))
        }),
        "Bug4: MATCH with property filter must find node after restart in FLEXIBLE mode. \
         Got: {:?}",
        match_rows
    );

    // 5. Full label scan must also find the node.
    let scan_rows = cypher_q(&proc, "MATCH (s:FlexPersist) RETURN count(s) AS cnt")
        .await
        .expect("label scan after restart");
    assert_eq!(
        scan_rows.first().and_then(|r| r.get("cnt")).cloned(),
        Some(PropertyValue {
            value: Some(PvKind::IntValue(1))
        }),
        "Bug4: full label scan must find node after restart. Got: {:?}",
        scan_rows
    );

    // 6. CREATE with the same unique key must fail — proves the node truly exists.
    let mut params = HashMap::new();
    params.insert("k".to_string(), pv_string("fp-key-1"));
    let dup = cypher(&proc, "CREATE (s:FlexPersist {key: $k})", params).await;
    assert!(
        dup.is_err(),
        "Bug4: CREATE with same unique key must fail after restart (unique constraint). \
         Got: Ok — node is query-invisible AND constraint-invisible (data lost entirely)"
    );
}
