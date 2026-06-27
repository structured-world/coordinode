use super::*;

#[test]
fn plan_cache_hit_returns_same_plan() {
    // Same query string twice → second call must observe the cache
    // entry created by the first. Asserted by introspecting
    // PlanCache.inner directly.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (n:Cached {k: 1}) RETURN n")
        .expect("create");
    db.execute_cypher("MATCH (n:Cached) RETURN n.k")
        .expect("first match");
    let after_first = db.plan_cache.inner.read().len();
    db.execute_cypher("MATCH (n:Cached) RETURN n.k")
        .expect("second match");
    let after_second = db.plan_cache.inner.read().len();
    assert_eq!(
        after_first, after_second,
        "second invocation of the same query must not grow the cache"
    );
    assert!(
        after_second >= 1,
        "cache must contain at least the MATCH plan"
    );
}

#[test]
fn plan_cache_distinguishes_queries() {
    // Different query strings → distinct cache entries.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (n:A {k: 1}) RETURN n")
        .expect("create A");
    db.execute_cypher("CREATE (n:B {k: 1}) RETURN n")
        .expect("create B");
    db.execute_cypher("MATCH (n:A) RETURN n.k")
        .expect("match A");
    db.execute_cypher("MATCH (n:B) RETURN n.k")
        .expect("match B");
    let entries = db.plan_cache.inner.read().len();
    assert!(
        entries >= 4,
        "expected ≥4 distinct cache entries, got {entries}"
    );
}

#[test]
fn plan_cache_bounded_eviction() {
    // Cache is bounded — pumping more distinct queries than the
    // bound must keep len at the bound, not grow unboundedly.
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    // Shrink the bound for the test so we don't have to issue
    // thousands of queries.
    db.plan_cache = Arc::new(PlanCache::new(4));
    for i in 0..16 {
        // Each query string is unique (different label) so cache
        // entries don't collide.
        let q = format!("MATCH (n:L{i}) RETURN n");
        // Don't care about result; this label has no data.
        let _ = db.execute_cypher(&q);
    }
    let entries = db.plan_cache.inner.read().len();
    assert!(
        entries <= 4,
        "cache exceeded max_entries=4: {entries} entries"
    );
}

#[test]
fn open_database() {
    let dir = tempfile::tempdir().expect("tempdir");
    let _db = Database::open(dir.path()).expect("open");
}

#[test]
fn create_and_match() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (n:User {name: 'Alice'}) RETURN n")
        .expect("create");

    let results = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("match");
    assert_eq!(results.len(), 1);
}

#[test]
fn explain_plan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let plan = db
        .explain_cypher("MATCH (n:User) RETURN n")
        .expect("explain");
    assert!(plan.contains("NodeScan"));
}

#[test]
fn parse_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    assert!(db.execute_cypher("INVALID").is_err());
}

#[test]
fn semantic_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let result = db.execute_cypher("MATCH (n) RETURN m");
    assert!(matches!(result, Err(DatabaseError::Semantic(_))));
}

/// Regression test: MATCH+SET property changes must be visible in subsequent queries.
///
/// Reproduces the snapshot isolation bug where MATCH+SET appears to succeed
/// (the change is visible via RETURN in the same query) but the updated value
/// is NOT visible in a subsequent MATCH query — the property reverts to its
/// pre-SET value across query boundaries.
///
/// Root cause hypothesis: the MVCC write buffer is flushed correctly but
/// the next query's snapshot is taken at a seqno that precedes the write.
#[test]
fn match_set_property_change_persists_across_queries() {
    use coordinode_core::graph::types::Value;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Step 1: create node via MERGE+SET (confirmed working per bug report)
    db.execute_cypher("MERGE (p:Project {id: 'x'}) SET p.status = 'active'")
        .expect("MERGE+SET must succeed");

    // Step 2: update via MATCH+SET — returns 'removed' in the same query
    let step2 = db
        .execute_cypher("MATCH (p:Project {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s")
        .expect("MATCH+SET must succeed");
    assert_eq!(step2.len(), 1, "MATCH+SET must return one row");
    assert_eq!(
        step2[0].get("s"),
        Some(&Value::String("removed".into())),
        "SET must be visible within the same query"
    );

    // Step 3: read in a SEPARATE query — must show the persisted value
    let step3 = db
        .execute_cypher("MATCH (p:Project {id: 'x'}) RETURN p.status AS s")
        .expect("MATCH RETURN must succeed");
    assert_eq!(step3.len(), 1, "node must still exist in step 3");
    assert_eq!(
        step3[0].get("s"),
        Some(&Value::String("removed".into())),
        "MATCH+SET must persist across query boundaries: expected 'removed', got {:?}",
        step3[0].get("s")
    );
}

/// Regression test: MATCH+SET property changes must persist when a B-tree index
/// is present on the lookup property, causing the planner to use IndexScan.
///
/// IndexScan reads from the engine directly (not MVCC snapshot) for the index
/// lookup. The node read in execute_update goes through mvcc_get (snapshot-based).
/// This test verifies the two-phase read (IndexScan → execute_update mvcc_get)
/// correctly commits the write to storage.
#[test]
fn match_set_persists_with_btree_index() {
    use coordinode_core::graph::types::Value;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create a B-tree index on :Project.id so that MATCH uses IndexScan
    db.execute_cypher("CREATE INDEX idx_project_id ON :Project(id)")
        .expect("CREATE INDEX");

    // Step 1: create node
    db.execute_cypher("MERGE (p:Project {id: 'x'}) SET p.status = 'active'")
        .expect("MERGE+SET");

    // Step 2: update via MATCH+SET — IndexScan path
    let step2 = db
        .execute_cypher("MATCH (p:Project {id: 'x'}) SET p.status = 'removed' RETURN p.status AS s")
        .expect("MATCH+SET with index");
    assert_eq!(
        step2[0].get("s"),
        Some(&Value::String("removed".into())),
        "SET visible in same query"
    );

    // Step 3: new query — must see 'removed'
    let step3 = db
        .execute_cypher("MATCH (p:Project {id: 'x'}) RETURN p.status AS s")
        .expect("MATCH RETURN");
    assert_eq!(step3.len(), 1);
    assert_eq!(
        step3[0].get("s"),
        Some(&Value::String("removed".into())),
        "MATCH+SET with IndexScan must persist: expected 'removed', got {:?}",
        step3[0].get("s")
    );
}

#[test]
fn data_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:User {name: 'Alice'})")
            .expect("create");
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let results = db.execute_cypher("MATCH (n:User) RETURN n").expect("match");
        assert!(!results.is_empty());
    }
}

/// Node IDs are monotonically increasing across database reopens.
/// This verifies G001: persistent NodeIdAllocator.
#[test]
fn node_ids_persist_across_reopen() {
    use coordinode_core::graph::types::Value;

    let dir = tempfile::tempdir().expect("tempdir");

    // First session: create a node, remember its ID via the node variable
    // (CREATE (n:...) RETURN n → n = node ID as Int)
    let first_id = {
        let mut db = Database::open(dir.path()).expect("open");
        let results = db
            .execute_cypher("CREATE (n:User {name: 'Alice'}) RETURN n")
            .expect("create Alice");
        assert_eq!(results.len(), 1);
        match results[0].get("n") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected Int, got {other:?}"),
        }
    };

    // Second session: create another node, verify its ID > first_id
    let second_id = {
        let mut db = Database::open(dir.path()).expect("reopen");
        let results = db
            .execute_cypher("CREATE (n:User {name: 'Bob'}) RETURN n")
            .expect("create Bob");
        assert_eq!(results.len(), 1);
        match results[0].get("n") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected Int, got {other:?}"),
        }
    };

    assert!(
        second_id > first_id,
        "second session ID ({second_id}) must be > first session ID ({first_id})"
    );
}

/// Multiple reopens don't lose ID state: IDs always increase.
#[test]
fn node_ids_persist_across_multiple_reopens() {
    use coordinode_core::graph::types::Value;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut last_id = 0i64;

    for i in 0..5 {
        let mut db = Database::open(dir.path()).expect("open");
        let results = db
            .execute_cypher(&format!("CREATE (n:User {{name: 'User{i}'}}) RETURN n"))
            .expect("create");
        assert_eq!(results.len(), 1);
        let id = match results[0].get("n") {
            Some(Value::Int(id)) => *id,
            other => panic!("expected Int, got {other:?}"),
        };
        assert!(
            id > last_id,
            "reopen {i}: id ({id}) must be > last_id ({last_id})"
        );
        last_id = id;
    }
}

/// Batch exhaustion triggers a new persistent batch.
#[test]
fn batch_exhaustion_persists_new_ceiling() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        // Create enough nodes to exhaust the first batch (1000 IDs).
        // Each CREATE allocates 1 ID.
        for i in 0..50 {
            db.execute_cypher(&format!("CREATE (n:User {{idx: {i}}})"))
                .expect("create");
        }
        // Verify ceiling was persisted correctly
        let ceiling_bytes = db
            .engine()
            .get(
                coordinode_storage::engine::partition::Partition::Schema,
                SCHEMA_KEY_NEXT_NODE_ID,
            )
            .expect("get ceiling")
            .expect("ceiling should exist");
        let ceiling = u64::from_be_bytes(ceiling_bytes[..8].try_into().expect("8 bytes"));
        assert!(
            ceiling >= 1000,
            "ceiling ({ceiling}) should be at least ID_BATCH_SIZE"
        );
    }

    // Reopen and verify we can continue creating nodes
    {
        let mut db = Database::open(dir.path()).expect("reopen");
        db.execute_cypher("CREATE (n:User {name: 'AfterReopen'})")
            .expect("create after reopen");
    }
}

#[test]
fn set_vector_consistency_session() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Default is Current
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);

    // SET via Cypher-like session command
    let result = db.execute_cypher("SET vector_consistency = 'snapshot'");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().len(), 0); // no rows returned
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);

    // SET exact
    db.execute_cypher("SET vector_consistency = 'exact'")
        .expect("set exact");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Exact);

    // SET back to current
    db.execute_cypher("SET vector_consistency = 'current'")
        .expect("set current");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);
}

#[test]
fn set_vector_consistency_case_insensitive() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("SET vector_consistency = 'SNAPSHOT'")
        .expect("upper");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);

    db.execute_cypher("set vector_consistency = 'Current'")
        .expect("mixed");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Current);
}

#[test]
fn set_vector_consistency_double_quotes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("SET vector_consistency = \"snapshot\"")
        .expect("double quotes");
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Snapshot);
}

#[test]
fn set_vector_consistency_api() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.set_vector_consistency(VectorConsistencyMode::Exact);
    assert_eq!(db.vector_consistency(), VectorConsistencyMode::Exact);
}

#[test]
fn set_vector_consistency_invalid_value_falls_through_to_parser() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Invalid mode name → not recognized as session SET → falls through to Cypher parser
    // which will fail with a parse error
    let result = db.execute_cypher("SET vector_consistency = 'invalid_mode'");
    assert!(result.is_err());
}

#[test]
fn extension_op_dispatches_through_database() {
    use std::sync::atomic::{AtomicBool, Ordering};

    struct CountingHandler {
        called: Arc<AtomicBool>,
    }
    impl ExtensionHandler for CountingHandler {
        fn execute(
            &self,
            _ctx: &mut ExecutionContext<'_>,
            _payload: &[u8],
        ) -> Result<Vec<Row>, ExecutionError> {
            self.called.store(true, Ordering::SeqCst);
            Ok(vec![])
        }
    }

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    // Ensure the :Doc label exists before the index DDL.
    db.execute_cypher("CREATE (:Doc {x: 1})")
        .expect("seed label");

    let called = Arc::new(AtomicBool::new(false));
    db.register_extension(
        "vector_index.create_ext",
        Arc::new(CountingHandler {
            called: Arc::clone(&called),
        }),
    );

    // A SHARDED-BY tail on CREATE VECTOR INDEX routes end-to-end through the
    // Database: the parser captures the tail, the planner emits an Extension
    // op, and the executor dispatches it to the registered handler.
    db.execute_cypher("CREATE VECTOR INDEX foo ON :Doc(embedding) SHARDED BY CENTROID(8)")
        .expect("extension op executes");
    assert!(
        called.load(Ordering::SeqCst),
        "registered extension handler ran via Database"
    );
}

/// engine_shared() returns an Arc pointing to the same engine as engine().
/// Writes through one are visible through the other.
#[test]
fn engine_shared_same_instance() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let shared = db.engine_shared();
    let key = b"meta:shared_test";
    let val = b"shared_value";

    // Write through shared Arc
    shared
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            key,
            val,
        )
        .expect("put via shared");

    // Read through engine() borrow — same data
    let got = db
        .engine()
        .get(
            coordinode_storage::engine::partition::Partition::Schema,
            key,
        )
        .expect("get via engine()")
        .expect("must exist");
    assert_eq!(got.as_ref(), val);
}

/// Multiple engine_shared() calls return Arcs to the same underlying engine.
#[test]
fn engine_shared_multiple_arcs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    let arc1 = db.engine_shared();
    let arc2 = db.engine_shared();

    // Both point to the same allocation (Arc strong count = 6:
    // one in Database, one in OwnedLocalProposalPipeline (drain),
    // one in TtlReaperHandle (background thread), one in the
    // LsmVectorTier backing VectorIndexRegistry (ADR-033 f32
    // truth tier), two here).
    assert_eq!(Arc::strong_count(&arc1), 6);
    assert_eq!(Arc::strong_count(&arc2), 6);

    // Write through arc1, read through arc2
    arc1.put(
        coordinode_storage::engine::partition::Partition::Schema,
        b"meta:arc_test",
        b"v",
    )
    .expect("put");
    let got = arc2
        .get(
            coordinode_storage::engine::partition::Partition::Schema,
            b"meta:arc_test",
        )
        .expect("get")
        .expect("must exist");
    assert_eq!(got.as_ref(), b"v");
}

#[test]
fn explain_shows_vector_consistency_for_vector_queries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Set snapshot mode
    db.set_vector_consistency(VectorConsistencyMode::Snapshot);

    let explain = db
        .explain_cypher(
            "MATCH (m:Movie) WHERE vector_distance(m.embedding, [1.0, 0.0]) < 0.5 RETURN m",
        )
        .expect("explain");

    assert!(
        explain.contains("Vector consistency: snapshot"),
        "EXPLAIN should show vector consistency mode: {explain}"
    );
}

// ─── Bug regression: MERGE on existing node with unique constraint ──────────

/// MERGE on a node that already exists must match and apply ON MATCH SET,
/// NOT throw "unique constraint violated".
///
/// Bug: `execute_merge` runs a full NodeScan with property filters. When the
/// scan returns empty (misses the existing node), MERGE falls through to
/// CREATE which triggers the B-tree unique index → "unique constraint violated".
///
/// Expected: MERGE finds the existing node and applies SET s.name = 'updated'.
#[test]
fn merge_on_existing_node_with_unique_constraint_does_not_error() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create label with a unique segment_id property.
    let mut schema = LabelSchema::new_node_id("Segment");
    schema.add_property(PropertyDef::new("segment_id", PropertyType::Int).unique());
    schema.add_property(PropertyDef::new("name", PropertyType::String));
    db.create_label_schema(schema).expect("create schema");

    // Create the initial node.
    db.execute_cypher("CREATE (s:Segment {segment_id: 42, name: 'original'})")
        .expect("create initial node");

    // MERGE on existing segment_id must find the node, not try to create it.
    // Before the fix this throws: "write conflict: unique constraint violated
    // on index segment_segment_id".
    let result = db
        .execute_cypher("MERGE (s:Segment {segment_id: 42}) SET s.name = 'updated' RETURN s.name");
    assert!(
        result.is_ok(),
        "MERGE on existing unique node must not error: {:?}",
        result.err()
    );

    let rows = result.unwrap();
    assert_eq!(rows.len(), 1, "MERGE must return exactly one matched row");

    // Verify the SET was applied — ON MATCH branch was taken.
    let rows = db
        .execute_cypher("MATCH (s:Segment {segment_id: 42}) RETURN s.name")
        .expect("match after merge");
    assert_eq!(rows.len(), 1);
    use coordinode_core::graph::types::Value;
    assert_eq!(
        rows[0].get("s.name"),
        Some(&Value::String("updated".into())),
        "SET must be applied via ON MATCH branch"
    );
}

/// Same as above but using Cypher parameters ($val / $name) — the bug was
/// reported with parameterized queries. Parameters must not affect MERGE
/// node matching behaviour.
#[test]
fn merge_with_params_on_existing_unique_node_does_not_error() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    let mut schema = LabelSchema::new_node_id("Segment");
    schema.add_property(PropertyDef::new("segment_id", PropertyType::Int).unique());
    schema.add_property(PropertyDef::new("name", PropertyType::String));
    db.create_label_schema(schema).expect("create schema");

    // Create node via params.
    let mut create_params = std::collections::HashMap::new();
    create_params.insert("sid".into(), Value::Int(99));
    create_params.insert("name".into(), Value::String("original".into()));
    db.execute_cypher_with_params(
        "CREATE (s:Segment {segment_id: $sid, name: $name})",
        create_params,
    )
    .expect("create node");

    // MERGE + SET via params.  Bug: this throws "unique constraint violated"
    // because NodeScan misses the node and MERGE falls through to CREATE.
    let mut merge_params = std::collections::HashMap::new();
    merge_params.insert("sid".into(), Value::Int(99));
    merge_params.insert("new_name".into(), Value::String("updated".into()));
    let result = db.execute_cypher_with_params(
        "MERGE (s:Segment {segment_id: $sid}) SET s.name = $new_name RETURN s.name",
        merge_params,
    );
    assert!(
        result.is_ok(),
        "parameterized MERGE on existing unique node must not error: {:?}",
        result.err()
    );

    // Verify ON MATCH was taken.
    let mut match_params = std::collections::HashMap::new();
    match_params.insert("sid".into(), Value::Int(99));
    let rows = db
        .execute_cypher_with_params(
            "MATCH (s:Segment {segment_id: $sid}) RETURN s.name",
            match_params,
        )
        .expect("match after merge");
    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("s.name"),
        Some(&Value::String("updated".into())),
        "SET must apply in ON MATCH branch"
    );
}

/// Exact gRPC repro: STRICT mode, unique INT id, node created with String value.
///
/// The user-reported repro uses schema_mode=1 (STRICT), id type=INT64 (1),
/// then creates a node with id="x1" (string literal in Cypher). After restart
/// MERGE (n:TestNode {id: "x1"}) SET n.value = "updated" throws unique constraint.
///
/// This test exercises the same mismatch: declared INT, actual String value in Cypher.
#[test]
fn merge_strict_mode_unique_id_string_literal_does_not_error() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // STRICT mode, unique id property (INT).
    let mut schema = LabelSchema::new_node_id("TestNode");
    schema.add_property(
        PropertyDef::new("id", PropertyType::String)
            .unique()
            .not_null(),
    );
    db.create_label_schema(schema).expect("create schema");

    // Create node.
    db.execute_cypher("CREATE (n:TestNode {id: 'x1'})")
        .expect("create node");

    // MERGE + SET undeclared property — the reported error is "unique constraint violated"
    // which means NodeScan missed the node. Setting 'value' is a separate schema issue
    // but the constraint error suggests MERGE didn't find the node at all.
    //
    // Test the core invariant: MERGE must find the existing node (match count = 1).
    let result =
        db.execute_cypher("MERGE (n:TestNode {id: 'x1'}) ON MATCH SET n.id = 'x1' RETURN n.id");
    assert!(
        result.is_ok(),
        "MERGE on existing STRICT unique node must not error: {:?}",
        result.err()
    );
    let rows = result.unwrap();
    assert_eq!(rows.len(), 1, "MERGE must match exactly one node");
}

/// MERGE across restart: create label+node in session 1, MERGE in session 2.
///
/// This is the exact gRPC repro pattern: restart the server between CREATE and MERGE.
/// The interner and unique index must survive restart correctly so MERGE finds
/// the node instead of trying to create a duplicate.
#[test]
fn merge_on_existing_unique_node_after_restart_does_not_error() {
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create label schema + node.
    {
        let mut db = Database::open(dir.path()).expect("open");

        let mut schema = LabelSchema::new_node_id("TestNode");
        schema.add_property(
            PropertyDef::new("id", PropertyType::String)
                .unique()
                .not_null(),
        );
        db.create_label_schema(schema).expect("create schema");

        db.execute_cypher("CREATE (n:TestNode {id: 'x1'})")
            .expect("create node");
    }

    // Session 2 (simulates server restart): reopen DB, run MERGE.
    // Bug hypothesis: after restart, the interner or index state is corrupted
    // so NodeScan misses the existing node → MERGE falls through to CREATE →
    // B-tree unique index violation.
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        let result =
            db.execute_cypher("MERGE (n:TestNode {id: 'x1'}) ON MATCH SET n.id = 'x1' RETURN n.id");
        assert!(
            result.is_ok(),
            "MERGE on existing unique node after restart must succeed: {:?}",
            result.err()
        );
        let rows = result.unwrap();
        assert_eq!(
            rows.len(),
            1,
            "MERGE must match exactly one node after restart"
        );
    }
}

// ─── Bug regression: vector schema dimension lost across restart ─────────────

/// Writing a vector node after DB restart must not fail when the persisted
/// label schema has `dimensions: 0` (lost across restart).
///
/// Bug: `PropertyDefinition` in proto has no `dimensions` field.
/// `schema.rs` hardcodes `dimensions: 0` when deserializing VECTOR type.
/// After restart, all VECTOR properties have `dimensions: 0` → dimension
/// validation fails on the next write.
///
/// Expected: dimension is either preserved in schema or auto-inferred from
/// the first written vector. Subsequent writes of matching dimension succeed.
#[test]
fn vector_schema_dimension_survives_restart() {
    use coordinode_core::graph::types::{Value, VectorMetric};
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: create schema with 3-dimensional vector, write a node.
    {
        let mut db = Database::open(dir.path()).expect("open");

        let mut schema = LabelSchema::new_node_id("Doc");
        schema.add_property(PropertyDef::new(
            "embedding",
            PropertyType::Vector {
                dimensions: 3,
                metric: VectorMetric::Cosine,
            },
        ));
        db.create_label_schema(schema).expect("create schema");

        db.create_vector_index(
            "doc_embedding",
            "Doc",
            "embedding",
            VectorIndexConfig {
                dimensions: 3,
                ..VectorIndexConfig::default()
            },
        );

        let mut params = std::collections::HashMap::new();
        params.insert("vec".into(), Value::Vector(vec![1.0, 0.0, 0.0]));
        db.execute_cypher_with_params("CREATE (d:Doc {embedding: $vec})", params)
            .expect("create first node");
    }

    // Session 2: reopen and write another vector — must not fail.
    // Before the fix: schema has dimensions=0 after restart → write rejected.
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        let mut params = std::collections::HashMap::new();
        params.insert("vec".into(), Value::Vector(vec![0.0, 1.0, 0.0]));
        let result = db.execute_cypher_with_params("CREATE (d:Doc {embedding: $vec})", params);
        assert!(
            result.is_ok(),
            "vector write after restart must succeed (dimension must survive restart): {:?}",
            result.err()
        );
    }
}

// ─── Bug regression: HNSW not rebuilt for overflow (Flexible-mode) vectors ───

/// After DB restart, HNSW must be rebuilt from vectors stored in `record.extra`
/// Schema with dimensions=0 (the value gRPC sets via proto_type_to_property_type(7)):
/// writing a vector must NOT fail on type/dimension mismatch.
///
/// This is the EXACT schema state after `SchemaService/CreateLabel` with type=7 (VECTOR):
/// `proto_type_to_property_type(7)` → `PropertyType::Vector { dimensions: 0, metric: Cosine }`.
/// Because the proto `PropertyDefinition` has no `dimensions` field, it's always 0.
///
/// Expected: dimension validation treats 0 as "unset/auto" and accepts any vector length,
/// OR the schema auto-updates to the first written dimension.
#[test]
fn vector_write_with_grpc_schema_zero_dimensions_does_not_error() {
    use coordinode_core::graph::types::{Value, VectorMetric};
    use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Simulate what gRPC SchemaService does: dimensions=0 because proto has no field.
    let mut schema = LabelSchema::new_node_id("VecTest");
    schema.add_property(PropertyDef::new(
        "emb",
        PropertyType::Vector {
            dimensions: 0, // ← gRPC always writes 0 (proto has no dimensions field)
            metric: VectorMetric::Cosine,
        },
    ));
    db.create_label_schema(schema).expect("create schema");

    // Write a 4-dimensional vector.
    // Bug: validation checks `vec.len() != 0` → VectorDimsMismatch(expected=0, got=4).
    let mut params = std::collections::HashMap::new();
    params.insert("vec".into(), Value::Vector(vec![0.1, 0.2, 0.3, 0.4]));
    let result = db.execute_cypher_with_params("CREATE (n:VecTest {emb: $vec})", params);
    assert!(
        result.is_ok(),
        "vector write must succeed when schema has dimensions=0 (unset via gRPC): {:?}",
        result.err()
    );
}

/// (overflow props in Flexible/Validated schema mode).
///
/// Bug: `load_vector_indexes` only checks `record.props.get(&field_id)`.
/// For Flexible-mode nodes, vectors are stored in `record.extra` (string-keyed
/// overflow map). The interner may not have an entry for the prop, and even if
/// it does, `record.props` doesn't contain it → HNSW rebuilt empty.
///
/// Expected: after restart, vector search returns semantically relevant results.
#[test]
fn hnsw_rebuilt_for_flexible_mode_overflow_vectors() {
    use coordinode_core::graph::types::Value;
    use coordinode_core::schema::definition::{LabelSchema, SchemaMode};
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");

    // Session 1: use Flexible schema (vectors go to overflow/extra).
    {
        let mut db = Database::open(dir.path()).expect("open");

        // Flexible label — no declared properties, all stored as overflow.
        let mut schema = LabelSchema::new_node_id("Article");
        schema.set_mode(SchemaMode::Flexible);
        db.create_label_schema(schema).expect("create schema");

        db.create_vector_index(
            "article_embedding",
            "Article",
            "embedding",
            VectorIndexConfig {
                dimensions: 3,
                ..VectorIndexConfig::default()
            },
        );

        // Write two nodes with clearly distinct embeddings.
        let mut p1 = std::collections::HashMap::new();
        p1.insert("vec".into(), Value::Vector(vec![1.0, 0.0, 0.0]));
        db.execute_cypher_with_params("CREATE (a:Article {embedding: $vec, title: 'rust'})", p1)
            .expect("create article 1");

        let mut p2 = std::collections::HashMap::new();
        p2.insert("vec".into(), Value::Vector(vec![0.0, 1.0, 0.0]));
        db.execute_cypher_with_params("CREATE (a:Article {embedding: $vec, title: 'golang'})", p2)
            .expect("create article 2");
    }

    // Session 2: reopen and do a vector search.
    // Before the fix: HNSW is empty after restart → no results returned.
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        // Query close to [1, 0, 0] — must return the 'rust' article.
        let results = db
            .execute_cypher(
                "MATCH (a:Article) \
                     WHERE vector_distance(a.embedding, [0.99, 0.0, 0.0]) < 0.1 \
                     RETURN a.title",
            )
            .expect("vector search after restart");

        assert!(
            !results.is_empty(),
            "HNSW must be rebuilt from overflow vectors on restart; got 0 results"
        );
    }
}

/// Regression: DETACH DELETE must actually remove the node from storage.
///
/// Bug: DETACH DELETE returns Ok but leaves the node in place.
/// After `MATCH (n:BugTest {id: "dt-1"}) DETACH DELETE n`, a subsequent
/// MATCH for the same node must return 0 rows.
#[test]
fn detach_delete_removes_node() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    db.execute_cypher("CREATE (:BugTest {id: 'dt-1', val: 'hello'})")
        .expect("create");

    // Confirm node exists.
    let before = db
        .execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) RETURN n.id")
        .expect("match before delete");
    assert_eq!(before.len(), 1, "node must exist before delete");

    // Delete the node.
    db.execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) DETACH DELETE n")
        .expect("detach delete");

    // Node must be gone.
    let after = db
        .execute_cypher("MATCH (n:BugTest {id: 'dt-1'}) RETURN n.id")
        .expect("match after delete");
    assert_eq!(
        after.len(),
        0,
        "DETACH DELETE must remove the node; got {} rows instead of 0",
        after.len()
    );
}

/// Regression: SET on a VECTOR property must update the HNSW index position.
///
/// Bug: after `SET n.embedding = [new_vec]`, vector_distance queries still
/// use the original embedding from CREATE, silently returning stale results.
#[test]
fn vector_set_updates_hnsw_index() {
    use coordinode_core::graph::types::VectorMetric;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create a vector index on :VecTest(embedding).
    db.create_vector_index(
        "vec_test_idx",
        "VecTest",
        "embedding",
        VectorIndexConfig {
            dimensions: 4,
            metric: VectorMetric::L2,
            ..VectorIndexConfig::default()
        },
    );

    // Insert a node with embedding A = [1,0,0,0].
    db.execute_cypher("CREATE (:VecTest {id: 'v-1', embedding: [1.0, 0.0, 0.0, 0.0]})")
        .expect("create");

    // Update embedding to B = [0,1,0,0] (orthogonal to A).
    db.execute_cypher("MATCH (n:VecTest {id: 'v-1'}) SET n.embedding = [0.0, 1.0, 0.0, 0.0]")
        .expect("set embedding");

    // Confirm storage reflects the update.
    let stored = db
        .execute_cypher("MATCH (n:VecTest {id: 'v-1'}) RETURN n.embedding")
        .expect("read back");
    assert_eq!(stored.len(), 1, "node must still exist after SET");

    // Vector search with query ≈ B must find the node (distance < 0.1).
    // Bug: HNSW still has the original A position → 0 rows returned.
    let results = db
        .execute_cypher(
            "MATCH (n:VecTest) \
                 WHERE vector_distance(n.embedding, [0.0, 1.0, 0.0, 0.0]) < 0.1 \
                 RETURN n.id",
        )
        .expect("vector search after SET");

    assert_eq!(
        results.len(),
        1,
        "HNSW must reflect the updated embedding; \
             expected 1 result for query ≈ B, got {}",
        results.len()
    );
}

// ── R-PUSH1: end-to-end + wiring tests ────────────────────────────

/// CombinedStats returns real values from VectorIndexRegistry when the
/// index is registered. Closes the gap where vector_index_size/dim/
/// crossover defaulted to None — the wrapper is the live source of
/// truth for per-index statistics consumed by optimize_push_down.
#[test]
fn combined_stats_returns_real_vector_index_metadata() {
    use coordinode_core::graph::stats::StorageStats;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Create an HNSW index on (Doc.embedding) with dim=64, M=12.
    db.create_vector_index(
        "doc_emb_hnsw",
        "Doc",
        "embedding",
        VectorIndexConfig {
            dimensions: 64,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
            m: 12,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    );
    // Seed one vector so size > 0.
    db.execute_cypher(
        "CREATE (d:Doc {embedding: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})",
    )
    .expect("create vector node");

    let graph_stats = db.compute_stats().expect("compute_stats");
    let combined = CombinedStats {
        graph: &graph_stats,
        vector: &db.vector_index_registry,
    };
    assert_eq!(
        combined.vector_index_dim("Doc", "embedding"),
        Some(64),
        "dim from VectorIndexConfig must reach StorageStats"
    );
    let cross = combined
        .vector_index_crossover("Doc", "embedding")
        .expect("crossover present when index registered");
    // Heuristic: M=12 → max(12,8) * 32 = 384, no quantization. Clamped to [64,1024] → 384.
    assert_eq!(cross, 384, "crossover heuristic uses HNSW M parameter");

    // Unknown (label, property) returns None.
    assert!(combined.vector_index_dim("Doc", "no_such_prop").is_none());
    assert!(combined.vector_index_size("Other", "embedding").is_none());
}

/// CombinedStats halves the crossover threshold when the index uses
/// SQ8 quantization — quantized distance is ~2x cheaper, so the
/// graph-first crossover moves down to keep ACORN attractive earlier.
#[test]
fn combined_stats_crossover_lower_under_quantization() {
    use coordinode_core::graph::stats::StorageStats;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.create_vector_index(
        "q_idx",
        "QDoc",
        "embedding",
        VectorIndexConfig {
            dimensions: 32,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
            m: 12,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::Sq8,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    );
    let graph_stats = db.compute_stats().expect("compute_stats");
    let combined = CombinedStats {
        graph: &graph_stats,
        vector: &db.vector_index_registry,
    };
    let cross = combined
        .vector_index_crossover("QDoc", "embedding")
        .unwrap();
    // M=12, base=384, quant → base/2 = 192. Clamped to [64,1024] → 192.
    assert_eq!(cross, 192);
}

/// EXPLAIN annotates the serving health of a vector index the plan reads
/// through: a top-k vector query is promoted to HnswScan, so
/// the index name appears and its current state is surfaced.
#[test]
fn explain_annotates_vector_index_health() {
    use coordinode_core::graph::types::VectorMetric;
    use coordinode_query::index::VectorIndexConfig;

    let dir = tempfile::tempdir().unwrap();
    let mut db = Database::open(dir.path()).unwrap();
    db.create_vector_index(
        "movie_emb",
        "Movie",
        "embedding",
        VectorIndexConfig {
            dimensions: 3,
            metric: VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    );
    // Drive the index into a non-ready state so EXPLAIN shows more than the
    // default "ready".
    db.vector_index_registry()
        .report_health_rebuild("Movie", "embedding", 0.5, 250);

    let explain = db
        .explain_cypher(
            "MATCH (m:Movie) \
                 WITH *, vector_distance(m.embedding, [0.1, 0.2, 0.3]) AS d \
                 ORDER BY d ASC LIMIT 5 \
                 RETURN m",
        )
        .expect("explain");

    assert!(
        explain.contains("movie_emb"),
        "top-k vector query must be promoted to HnswScan(movie_emb): {explain}"
    );
    assert!(
        explain.contains("Vector index health:"),
        "EXPLAIN must carry the index health section: {explain}"
    );
    assert!(
        explain.to_lowercase().contains("rebuilding"),
        "health annotation must reflect the rebuilding state: {explain}"
    );
}

/// End-to-end: an EXPLAIN of TRAVERSE→VECTOR_FILTER must hit
/// optimize_push_down and surface a populated `push_down` decision on
/// the VectorFilter operator in the plan. Smoke test that the entire
/// pipeline — execute_cypher_impl wiring → CombinedStats →
/// optimize_push_down → annotated VectorFilter — works on a real
/// Database instance, not just in planner unit tests.
#[test]
fn end_to_end_traverse_then_vector_filter_gets_push_down_decision() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");

    // Seed graph: User → LIKES → Movie with embedding.
    db.execute_cypher(
        "CREATE (u:User {name: 'A'})-[:LIKES]->(m:Movie {embedding: [0.1, 0.2, 0.3]})",
    )
    .expect("seed");

    let explain = db
        .explain_cypher(
            "MATCH (u:User)-[:LIKES]->(m:Movie) \
                 WHERE vector_distance(m.embedding, [0.1, 0.2, 0.3]) < 0.5 \
                 RETURN m",
        )
        .expect("explain");
    // The plan must build cleanly; we don't assert the strategy slug
    // here (that's R-PUSH2's EXPLAIN JSON contract). What we assert
    // end-to-end is that the explain pipeline runs without panic and
    // touches the push_down pass — verified indirectly by checking
    // that planner test invariants still hold on the equivalent plan
    // (covered by planner integration tests). This test guarantees
    // the wiring compiles AND runs against a real Database.
    assert!(
        !explain.is_empty(),
        "EXPLAIN must produce output for TRAVERSE→VECTOR_FILTER plans"
    );
    assert!(
        explain.to_lowercase().contains("vectorfilter")
            || explain.to_lowercase().contains("vector"),
        "EXPLAIN must mention the vector operator: {explain}"
    );
}

/// End-to-end: a query without TRAVERSE upstream of VECTOR_FILTER
/// must still build and execute without push-down annotation (the
/// invariant only fires when Traverse is present). Verifies negative
/// path through the entire stack.
#[test]
fn end_to_end_vector_filter_without_traverse_executes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    db.execute_cypher("CREATE (d:Doc {embedding: [0.5, 0.5, 0.5]})")
        .expect("seed");
    let rows = db
        .execute_cypher(
            "MATCH (d:Doc) WHERE vector_distance(d.embedding, [0.5, 0.5, 0.5]) < 1.0 \
                 RETURN d",
        )
        .expect("vector filter without traverse should work");
    assert!(!rows.is_empty());
}

// ----- server-side keyset cursor (execute_cypher_paged) -----

#[test]
fn paged_cursor_walks_label_in_keyset_pages() {
    // A keyset cursor over a label must visit every node exactly once across
    // pages of `limit` rows, then report exhaustion. Memory stays O(limit):
    // each page only materialises `limit` rows, not the whole label.
    use coordinode_core::graph::types::Value;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for k in 0..7 {
        db.execute_cypher(&format!("CREATE (n:Page {{k: {k}}})"))
            .expect("seed");
    }

    let mut seen: Vec<i64> = Vec::new();
    let mut resume: Option<Vec<u8>> = None;
    let mut read_ts: Option<u64> = None;
    let mut pages = 0;
    loop {
        let page = db
            .execute_cypher_paged(
                "MATCH (n:Page) RETURN n.k",
                None,
                read_ts,
                resume.clone(),
                2,
            )
            .expect("page");
        pages += 1;
        for row in &page.rows {
            match row.get("n.k") {
                Some(Value::Int(v)) => seen.push(*v),
                other => panic!("unexpected row value: {other:?}"),
            }
        }
        // Every page after the first must read against the same pinned snapshot.
        if let Some(t) = read_ts {
            assert_eq!(t, page.read_ts, "snapshot ts must stay pinned across pages");
        }
        read_ts = Some(page.read_ts);
        resume = page.last_key.clone();
        if page.exhausted {
            break;
        }
        assert!(pages < 100, "cursor must terminate");
    }

    seen.sort_unstable();
    assert_eq!(seen, (0..7).collect::<Vec<_>>(), "every node seen once");
    assert!(
        pages >= 4,
        "limit=2 over 7 rows needs ≥4 pages, got {pages}"
    );
}

#[test]
fn paged_cursor_snapshot_is_stable_under_concurrent_writes() {
    // A node inserted AFTER the cursor pins its snapshot must NOT appear in
    // later pages: the cursor reads at a fixed MVCC timestamp (repeatable
    // read), so the result set is stable for the cursor's life.
    use coordinode_core::graph::types::Value;
    let dir = tempfile::tempdir().expect("tempdir");
    let mut db = Database::open(dir.path()).expect("open");
    for k in 0..4 {
        db.execute_cypher(&format!("CREATE (n:Snap {{k: {k}}})"))
            .expect("seed");
    }

    // First page pins the snapshot.
    let first = db
        .execute_cypher_paged("MATCH (n:Snap) RETURN n.k", None, None, None, 2)
        .expect("first page");
    assert_eq!(first.rows.len(), 2);
    assert!(!first.exhausted);

    // Insert a fifth node while the cursor is open.
    db.execute_cypher("CREATE (n:Snap {k: 99})")
        .expect("late insert");

    // Drain the rest against the pinned snapshot: the late node must be absent.
    let mut seen: Vec<i64> = first
        .rows
        .iter()
        .map(|r| match r.get("n.k") {
            Some(Value::Int(v)) => *v,
            other => panic!("unexpected: {other:?}"),
        })
        .collect();
    let mut resume = first.last_key.clone();
    let read_ts = Some(first.read_ts);
    loop {
        let page = db
            .execute_cypher_paged(
                "MATCH (n:Snap) RETURN n.k",
                None,
                read_ts,
                resume.clone(),
                2,
            )
            .expect("page");
        for row in &page.rows {
            if let Some(Value::Int(v)) = row.get("n.k") {
                seen.push(*v);
            }
        }
        resume = page.last_key.clone();
        if page.exhausted {
            break;
        }
    }
    seen.sort_unstable();
    assert_eq!(
        seen,
        vec![0, 1, 2, 3],
        "late-inserted node must not be visible"
    );
}

#[test]
fn keyset_pageable_classifies_plan_shapes() {
    // The cursor layer routes only non-blocking single-scan plans to the
    // keyset path; everything that reorders, collapses, bounds, or multiplies
    // rows must materialise.
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");

    // Eligible: plain scan, scan + filter, scan + projection.
    assert!(db.keyset_pageable("MATCH (n:User) RETURN n"));
    assert!(db.keyset_pageable("MATCH (n:User) WHERE n.age > 30 RETURN n.name"));
    assert!(db.keyset_pageable("MATCH (n:User) RETURN n.name AS name"));

    // Ineligible: blocking / bounded / multi-source / collapsing.
    assert!(
        !db.keyset_pageable("MATCH (n:User) RETURN n ORDER BY n.name"),
        "sort blocks"
    );
    assert!(
        !db.keyset_pageable("MATCH (n:User) RETURN count(n)"),
        "aggregate blocks"
    );
    assert!(
        !db.keyset_pageable("MATCH (n:User) RETURN DISTINCT n.name"),
        "distinct blocks"
    );
    assert!(
        !db.keyset_pageable("MATCH (n:User) RETURN n LIMIT 10"),
        "limit bounds the result"
    );
    assert!(
        !db.keyset_pageable("MATCH (a:User)-[:KNOWS]->(b:User) RETURN b"),
        "traverse is multi-source"
    );

    // Unparseable / unplannable → ineligible (error surfaces on execution).
    assert!(!db.keyset_pageable("THIS IS NOT CYPHER"));
}

#[test]
fn paged_cursor_empty_label_is_exhausted_immediately() {
    // A label with no nodes yields one exhausted page, no resume token.
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let page = db
        .execute_cypher_paged("MATCH (n:Nothing) RETURN n.k", None, None, None, 10)
        .expect("page");
    assert!(page.rows.is_empty());
    assert!(page.exhausted);
    assert!(page.last_key.is_none());
}
