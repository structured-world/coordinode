//! Standalone restart regression tests.
//!
//! Each test covers the gRPC/standalone path, which the embedded API cannot
//! reproduce because it bypasses proto serialisation for schema creation:
//! vector writes against a gRPC-declared VECTOR property, MERGE on a unique
//! key, vector search on a FLEXIBLE label, MATCH visibility in FLEXIBLE mode,
//! and startup after an unclean shutdown, each before and after a restart.
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
    PropertyValue, Vector, property_value::Value as PvKind,
};
use coordinode_integration::proto::query::{ExecuteCypherRequest, Row};
use coordinode_integration::proto::v2::graph::{
    ConstraintKind, CreateConstraintRequest, CreateLabelRequest, PropertyDefinition, PropertyType,
    ScalarType, SchemaMode, VectorType, create_constraint_request, property_type,
};
use std::collections::HashMap;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// A stored property of type `t`.
fn property(name: &str, t: property_type::Type, required: bool) -> PropertyDefinition {
    PropertyDefinition {
        name: name.to_string(),
        r#type: Some(PropertyType { r#type: Some(t) }),
        required,
        default_value: None,
    }
}

fn string_type() -> property_type::Type {
    property_type::Type::Scalar(ScalarType::String as i32)
}

/// A vector type with no fixed length: values of any length are accepted.
fn any_vector_type() -> property_type::Type {
    property_type::Type::Vector(VectorType {
        dimensions: 0,
        metric: 0,
    })
}

/// Define label `name` with `properties` in `mode`.
async fn create_label(
    proc: &CoordinodeProcess,
    name: &str,
    properties: Vec<PropertyDefinition>,
    mode: SchemaMode,
) {
    proc.schema_client()
        .await
        .create_label(CreateLabelRequest {
            name: name.to_string(),
            properties,
            computed_properties: vec![],
            schema_mode: mode as i32,
            temporal: false,
        })
        .await
        .expect("create_label");
}

/// Require `property` of `label` to be unique, through a named constraint.
async fn require_unique(proc: &CoordinodeProcess, label: &str, property: &str) {
    proc.schema_client()
        .await
        .create_constraint(CreateConstraintRequest {
            name: String::new(),
            target: Some(create_constraint_request::Target::Label(label.to_string())),
            properties: vec![property.to_string()],
            kind: ConstraintKind::Unique as i32,
            property_type: None,
            if_not_exists: false,
            wait: None,
            on_duplicate_rename: String::new(),
            scope: None,
        })
        .await
        .expect("create_constraint");
}

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
            transaction_id: 0,
            ..Default::default()
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

// ── Vector property declared over gRPC ────────────────────────────────────────

/// A VECTOR property declared over gRPC without dimensions records 0
/// ("unset"), and a vector of any length must be accepted.
#[tokio::test]
async fn vector_write_to_grpc_declared_property_succeeds() {
    let proc = CoordinodeProcess::start().await;

    create_label(
        &proc,
        "VecNode",
        vec![property("emb", any_vector_type(), false)],
        SchemaMode::Strict,
    )
    .await;

    // Write a 4-dimensional vector — must succeed even though schema has dimensions=0.
    let mut params = HashMap::new();
    params.insert("vec".to_string(), pv_vector(vec![0.1, 0.2, 0.3, 0.4]));
    let result = cypher(&proc, "CREATE (n:VecNode {emb: $vec}) RETURN n", params).await;
    assert!(
        result.is_ok(),
        "vector write must succeed when schema has dimensions=0. Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 created node");
}

/// The dimensions=0 schema persists on disk; after a restart it is reloaded
/// as-is and a vector write must still be accepted.
#[tokio::test]
async fn vector_zero_dimensions_survives_restart() {
    let proc = CoordinodeProcess::start().await;

    // Step 1: create schema + write initial vector BEFORE restart.
    create_label(
        &proc,
        "VecRestart",
        vec![property("emb", any_vector_type(), false)],
        SchemaMode::Strict,
    )
    .await;

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
        "vector write after restart must succeed. Got: {:?}",
        result.err()
    );

    // Step 4: verify both nodes are readable.
    let rows = cypher_q(&proc, "MATCH (n:VecRestart) RETURN n")
        .await
        .expect("match after restart");
    assert_eq!(rows.len(), 2, "both nodes must survive restart");
}

/// A vector written before a crash is there after it: the write was
/// acknowledged, so every property of it is in the log and comes back with
/// the node, not only the node itself.
#[tokio::test]
async fn a_vector_property_survives_a_crash() {
    let proc = CoordinodeProcess::start().await;
    create_label(
        &proc,
        "VecCrash",
        vec![property("emb", any_vector_type(), false)],
        SchemaMode::Strict,
    )
    .await;
    let mut params = HashMap::new();
    params.insert("vec".to_string(), pv_vector(vec![1.0, 0.0, 0.0]));
    cypher(&proc, "CREATE (n:VecCrash {emb: $vec})", params)
        .await
        .expect("write before the kill");

    let proc = proc.restart_unclean().await;

    let rows = cypher_q(&proc, "MATCH (n:VecCrash) RETURN n.emb AS emb")
        .await
        .expect("read after the restart");
    assert_eq!(rows.len(), 1, "the node survives");
    match &rows[0]["emb"].value {
        Some(PvKind::VectorValue(v)) => assert_eq!(v.values, [1.0, 0.0, 0.0]),
        other => panic!("the vector did not survive the crash: {other:?}"),
    }
}

/// Property names a statement learns only while it runs, from the keys of a
/// map parameter, read back after a crash like any other: each name is
/// registered before the write that uses it, not repaired after it.
#[tokio::test]
async fn properties_named_by_a_map_survive_a_crash() {
    let proc = CoordinodeProcess::start().await;
    cypher_q(&proc, "CREATE (n:DynCrash {k: 1})")
        .await
        .expect("create before the kill");
    let entries = HashMap::from([
        ("colour".to_string(), pv_string("red")),
        ("size".to_string(), pv_string("large")),
    ]);
    let params = HashMap::from([(
        "m".to_string(),
        PropertyValue {
            value: Some(PvKind::MapValue(
                coordinode_integration::proto::common::PropertyMap { entries },
            )),
        },
    )]);
    cypher(&proc, "MATCH (n:DynCrash) SET n += $m", params)
        .await
        .expect("set from a map before the kill");

    let proc = proc.restart_unclean().await;

    let rows = cypher_q(
        &proc,
        "MATCH (n:DynCrash) RETURN n.colour AS colour, n.size AS size",
    )
    .await
    .expect("read after the restart");
    assert_eq!(rows.len(), 1, "the node survives");
    assert_eq!(
        rows[0]["colour"],
        pv_string("red"),
        "colour after the crash"
    );
    assert_eq!(rows[0]["size"], pv_string("large"), "size after the crash");
}

// ── MERGE on a unique key ─────────────────────────────────────────────────────

/// MERGE on an existing unique node must take the ON MATCH branch, not raise
/// a unique constraint violation.
#[tokio::test]
async fn merge_on_existing_unique_key_matches() {
    let proc = CoordinodeProcess::start().await;

    create_label(
        &proc,
        "UniqueNode",
        vec![property("id", string_type(), true)],
        SchemaMode::Strict,
    )
    .await;
    require_unique(&proc, "UniqueNode", "id").await;

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
        "MERGE on existing unique node must succeed (ON MATCH). Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 matched node");
}

/// Same as above across a restart, through the schema reload path.
#[tokio::test]
async fn merge_on_existing_unique_key_matches_after_restart() {
    let proc = CoordinodeProcess::start().await;

    create_label(
        &proc,
        "UniqueRestart",
        vec![property("id", string_type(), true)],
        SchemaMode::Strict,
    )
    .await;
    require_unique(&proc, "UniqueRestart", "id").await;

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
        "MERGE on existing unique node after restart must succeed. Got: {:?}",
        result.err()
    );
    assert_eq!(result.unwrap().len(), 1, "should return 1 matched node");
}

// ── Vector search on a FLEXIBLE label across a restart ────────────────────────

/// Vector similarity search on a FLEXIBLE label must return the same number
/// of results after a graceful restart as before it: the nodes survive and
/// their vectors are readable and searchable again.
#[tokio::test]
async fn flexible_vector_search_survives_restart() {
    let proc = CoordinodeProcess::start().await;

    // Create a Flexible-mode label with a VECTOR property.
    create_label(
        &proc,
        "FlexVec",
        vec![property("emb", any_vector_type(), false)],
        SchemaMode::Flexible,
    )
    .await;

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
        "vector search must return results before restart"
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
        "5 nodes must still exist after restart, got {node_count:?}"
    );

    // Diagnostic: verify n.emb is readable after restart.
    let emb_rows = cypher_q(&proc, "MATCH (n:FlexVec) RETURN n.emb LIMIT 1")
        .await
        .expect("emb read after restart");
    assert!(
        !emb_rows.is_empty(),
        "MATCH FlexVec must return at least 1 row after restart"
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
        "vector search must return results after restart (HNSW must be rebuilt). Got 0 rows.\
         \nNote: 5 nodes exist (checked above), n.emb readable (checked above).\
         \nFailing in VectorTopK or vector_similarity evaluation."
    );
    assert_eq!(
        rows_after.len(),
        rows_before.len(),
        "result count must be same before and after restart"
    );
}

// ── SIGKILL restart: crash recovery (no graceful shutdown) ───────────────────

/// A server killed with SIGKILL must start again and accept queries.
///
/// After an unclean shutdown the Raft oplog segment can be on disk while the
/// LSM key `raft:oplog:last_log_id` is not, so `LogStore::open()` has to
/// recover the last log id from the segment files; treating the log as empty
/// would re-initialise it and fail creating a segment that already exists.
///
/// The node count is not asserted: whether the last writes reached the log
/// before the kill is not under the test's control.
#[tokio::test]
async fn sigkill_restart_survives_without_crash() {
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

/// A server killed mid-life must never hand out a NodeId that a live node
/// already carries. The identifiers it issued before the kill are in use by
/// committed nodes, so a node created after the restart has to get a new one
/// and leave every existing node in place (a reissued id overwrites a node).
#[tokio::test]
async fn a_crash_never_reissues_a_live_node_id() {
    let proc = CoordinodeProcess::start().await;
    for i in 0u32..5 {
        let mut params = HashMap::new();
        params.insert("i".to_string(), pv_string(&format!("before-{i}")));
        cypher(&proc, "CREATE (n:IdReuse {v: $i})", params)
            .await
            .expect("create before the kill");
    }

    let proc = proc.restart_unclean().await;

    let mut params = HashMap::new();
    params.insert("i".to_string(), pv_string("after"));
    cypher(&proc, "CREATE (n:IdReuse {v: $i})", params)
        .await
        .expect("create after the restart");

    let rows = cypher_q(&proc, "MATCH (n:IdReuse) RETURN id(n) AS nid, n.v AS v")
        .await
        .expect("read back");
    let mut values: Vec<String> = rows
        .iter()
        .map(|row| match &row["v"].value {
            Some(PvKind::StringValue(s)) => s.clone(),
            other => panic!("unexpected value {other:?}"),
        })
        .collect();
    values.sort();
    assert_eq!(
        values,
        [
            "after", "before-0", "before-1", "before-2", "before-3", "before-4"
        ],
        "every node written before the kill survives and the new one is added"
    );
    let mut ids: Vec<i64> = rows
        .iter()
        .map(|row| match row["nid"].value {
            Some(PvKind::IntValue(id)) => id,
            ref other => panic!("unexpected id {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 6, "six distinct ids");
}

// ── MATCH visibility in FLEXIBLE mode across a restart ────────────────────────

/// After a restart, a node on a FLEXIBLE label must be visible to every read
/// path at once: MATCH with a property filter, a full label scan, and the
/// unique constraint (a duplicate CREATE is refused). A node the constraint
/// sees but MATCH does not is the failure this guards.
#[tokio::test]
async fn flexible_match_visible_after_restart() {
    let proc = CoordinodeProcess::start().await;

    // 1. Create a FLEXIBLE label with one unique declared property.
    create_label(
        &proc,
        "FlexPersist",
        vec![property("key", string_type(), false)],
        SchemaMode::Flexible,
    )
    .await;
    require_unique(&proc, "FlexPersist", "key").await;

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
        "MATCH with property filter must find node after restart in FLEXIBLE mode. \
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
        "full label scan must find node after restart. Got: {:?}",
        scan_rows
    );

    // 6. CREATE with the same unique key must fail — proves the node truly exists.
    let mut params = HashMap::new();
    params.insert("k".to_string(), pv_string("fp-key-1"));
    let dup = cypher(&proc, "CREATE (s:FlexPersist {key: $k})", params).await;
    assert!(
        dup.is_err(),
        "CREATE with same unique key must fail after restart (unique constraint). \
         Got: Ok: node is query-invisible AND constraint-invisible (data lost entirely)"
    );
}

/// A unique constraint holds across a crash: the index entry an acknowledged
/// write made is recovered with the write, so a duplicate is still refused.
///
/// A clean stop flushes every tree and cannot tell an entry that was part of
/// the write from one written beside it.
#[tokio::test]
async fn a_unique_constraint_survives_a_crash() {
    let proc = CoordinodeProcess::start().await;
    create_label(
        &proc,
        "CrashUnique",
        vec![property("key", string_type(), false)],
        SchemaMode::Flexible,
    )
    .await;
    require_unique(&proc, "CrashUnique", "key").await;
    cypher_q(&proc, "CREATE (:CrashUnique {key: 'k1'})")
        .await
        .expect("create before the crash");

    let proc = proc.restart_unclean().await;

    let rows = cypher_q(&proc, "MATCH (s:CrashUnique) RETURN count(s) AS cnt")
        .await
        .expect("scan after the crash");
    assert_eq!(
        rows.first().and_then(|r| r.get("cnt")).cloned(),
        Some(PropertyValue {
            value: Some(PvKind::IntValue(1))
        }),
        "the acknowledged node survives the crash"
    );
    let dup = cypher_q(&proc, "CREATE (:CrashUnique {key: 'k1'})").await;
    assert!(
        dup.is_err(),
        "a duplicate of a key written before the crash must be refused"
    );
}

// ── Acknowledged writes across restarts ───────────────────────────────────────

/// A config that turns over everything a restart recovers from as fast as it
/// can: a Raft snapshot (and the log purge after it) every few entries, small
/// log segments, frequent checkpoints, and the slowest table codec, so flushes
/// are long enough for a stop to land in the middle of one.
const CHURN_CONFIG: &str = "\
storage:
  endpoints: []
  compression:
    hot: { codec: zstd, level: 22 }
    cold: { codec: zstd, level: 22 }
  oplog:
    segment_max_entries: 16
raft_snapshot_entries: 8
raft_snapshot_min_interval_secs: 1
checkpoint_interval_secs: 2
";

/// Write sequence number `seq` of `stream` at majority with the journal, the
/// strongest acknowledgement a client can ask for.
async fn write_acked(
    client: &mut coordinode_integration::proto::query::cypher_service_client::CypherServiceClient<
        tonic::transport::Channel,
    >,
    stream: &str,
    seq: i64,
) -> Result<(), tonic::Status> {
    use coordinode_integration::proto::replication::{
        Journal, WriteConcern, WriteConcernMode, write_concern::W,
    };
    let mut parameters = HashMap::new();
    parameters.insert("stream".to_string(), pv_string(stream));
    parameters.insert(
        "seq".to_string(),
        PropertyValue {
            value: Some(PvKind::IntValue(seq)),
        },
    );
    client
        .execute_cypher(ExecuteCypherRequest {
            query: "CREATE (:Acked {stream: $stream, seq: $seq})".to_string(),
            parameters,
            read_preference: 0,
            read_concern: None,
            write_concern: Some(WriteConcern {
                w: Some(W::Mode(WriteConcernMode::Majority as i32)),
                journal: Journal::Journal as i32,
                timeout_ms: 0,
            }),
            transaction_id: 0,
            ..Default::default()
        })
        .await
        .map(drop)
}

/// The sequence numbers of `stream` the server holds, found by a label scan
/// (no index can hide a node from it).
async fn stored_seqs(proc: &CoordinodeProcess, stream: &str) -> std::collections::BTreeSet<i64> {
    let mut params = HashMap::new();
    params.insert("stream".to_string(), pv_string(stream));
    cypher(
        proc,
        "MATCH (n:Acked) WITH n WHERE n.stream = $stream RETURN n.seq AS seq",
        params,
    )
    .await
    .expect("read the stream back")
    .iter()
    .map(|row| match row["seq"].value {
        Some(PvKind::IntValue(seq)) => seq,
        ref other => panic!("unexpected seq {other:?}"),
    })
    .collect()
}

/// Every acknowledged sequence number must be stored; a missing one is an
/// acknowledged write the server lost.
fn assert_all_stored(acked: &[i64], stored: &std::collections::BTreeSet<i64>, after: &str) {
    let lost: Vec<i64> = acked
        .iter()
        .copied()
        .filter(|seq| !stored.contains(seq))
        .collect();
    assert!(
        lost.is_empty(),
        "acknowledged writes lost after {after}: {lost:?} (acked {}, stored {})",
        acked.len(),
        stored.len()
    );
}

/// Writes acknowledged at majority with the journal survive every way a
/// server process stops: a clean stop, a kill between writes, and a kill
/// while a write stream is in flight, with Raft snapshots, log purges and
/// checkpoints turning over all the time. A stream observed to lose
/// acknowledged sequence numbers across a container restart is what this
/// guards.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn acknowledged_writes_survive_every_kind_of_stop() {
    let stream = "acked-stream";
    let mut proc = CoordinodeProcess::start_with_config(CHURN_CONFIG).await;
    let mut acked: Vec<i64> = Vec::new();
    let mut next: i64 = 0;

    // Writes between stops: clean stop, then a kill.
    for (round, unclean) in [(0, false), (1, true)] {
        let mut client = proc.cypher_client().await;
        for _ in 0..30 {
            write_acked(&mut client, stream, next)
                .await
                .unwrap_or_else(|e| panic!("write {next} in round {round}: {e}"));
            acked.push(next);
            next += 1;
        }
        drop(client);
        proc = if unclean {
            proc.restart_unclean().await
        } else {
            let proc = proc.restart().await;
            proc.wait_for_leader(std::time::Duration::from_secs(15))
                .await;
            proc
        };
        let stored = stored_seqs(&proc, stream).await;
        assert_all_stored(&acked, &stored, &format!("round {round}"));
    }

    // A kill while a stream is in flight: whatever was acknowledged before
    // the kill must be there after it.
    for round in 0..2 {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let count = std::sync::Arc::new(AtomicUsize::new(0));
        let mut client = proc.cypher_client().await;
        let writer = {
            let count = std::sync::Arc::clone(&count);
            let first = next;
            tokio::spawn(async move {
                let mut written = Vec::new();
                let mut seq = first;
                while write_acked(&mut client, stream, seq).await.is_ok() {
                    written.push(seq);
                    count.store(written.len(), Ordering::Release);
                    seq += 1;
                }
                written
            })
        };
        while count.load(Ordering::Acquire) < 30 + round * 7 {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        proc = proc.restart_unclean().await;
        let written = writer.await.expect("the writer ends at the kill");
        // The write in flight at the kill may have committed unacknowledged;
        // the next round starts past its number.
        next = written.last().map_or(next, |last| last + 1) + 1;
        acked.extend(written);
        let stored = stored_seqs(&proc, stream).await;
        assert_all_stored(
            &acked,
            &stored,
            &format!("an in-flight kill, round {round}"),
        );
    }
}
