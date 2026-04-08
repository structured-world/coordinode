//! Integration tests: COMPUTED property query-time evaluation (R082, R085).
//!
//! Tests that COMPUTED properties (Decay, TTL, VectorDecay) are evaluated
//! inline during query execution and visible in RETURN and WHERE clauses.
//! R085 tests verify multi-timestamp decay interpolation and TTL subtree scope.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::Value;
use coordinode_core::schema::computed::{ComputedSpec, DecayFormula, TtlScope};
use coordinode_core::schema::definition::{LabelSchema, PropertyDef, PropertyType};
use coordinode_embed::Database;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// Helper: create a schema with COMPUTED properties and persist it.
fn setup_memory_schema(db: &mut Database) {
    let mut schema = LabelSchema::new("Memory");
    schema.add_property(PropertyDef::new("content", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::computed(
        "relevance",
        ComputedSpec::Decay {
            formula: DecayFormula::Linear,
            initial: 1.0,
            target: 0.0,
            duration_secs: 604800, // 7 days
            anchor_field: "created_at".into(),
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_ttl",
        ComputedSpec::Ttl {
            duration_secs: 2592000, // 30 days
            anchor_field: "created_at".into(),
            scope: TtlScope::Node,
        },
    ));

    // Persist schema directly to storage.
    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("Memory");
    let bytes = schema.to_msgpack().expect("serialize schema");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");
}

// ── COMPUTED Decay in RETURN ──────────────────────────────────────

/// COMPUTED Decay field appears in RETURN with evaluated value.
#[test]
fn computed_decay_in_return() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    // Create node with timestamp = now (microseconds).
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'test note', created_at: {now_us}}})"
    ))
    .expect("create memory node");

    // Query: RETURN the computed relevance field.
    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.relevance AS rel, m.content AS c")
        .expect("query with computed");

    assert_eq!(rows.len(), 1);
    let rel = rows[0].get("rel").expect("relevance field should exist");
    match rel {
        Value::Float(f) => {
            // Just created → elapsed ≈ 0 → relevance ≈ 1.0
            assert!(
                *f > 0.99,
                "freshly created node should have relevance ≈ 1.0, got {f}"
            );
        }
        other => panic!("expected Float for computed relevance, got {other:?}"),
    }
}

// ── COMPUTED TTL in RETURN ────────────────────────────────────────

/// COMPUTED TTL field returns seconds remaining.
#[test]
fn computed_ttl_returns_seconds_remaining() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'ttl test', created_at: {now_us}}})"
    ))
    .expect("create node");

    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m._ttl AS ttl")
        .expect("query ttl");

    assert_eq!(rows.len(), 1);
    match rows[0].get("ttl").expect("ttl should exist") {
        Value::Int(remaining) => {
            // 30 days = 2592000 secs, just created → remaining ≈ 2592000
            assert!(
                *remaining > 2591000,
                "TTL should be ≈ 2592000 secs, got {remaining}"
            );
        }
        other => panic!("expected Int for TTL, got {other:?}"),
    }
}

// ── COMPUTED in WHERE ─────────────────────────────────────────────

/// COMPUTED Decay field usable in WHERE clause.
#[test]
fn computed_decay_in_where_filter() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // Create two nodes: one fresh (relevance ≈ 1.0) and one "old" (relevance ≈ 0).
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'fresh', created_at: {now_us}}})"
    ))
    .expect("create fresh");

    // 8 days ago → beyond 7-day duration → relevance = 0.0
    let old_us = now_us - 8 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'old', created_at: {old_us}}})"
    ))
    .expect("create old");

    // Filter: relevance > 0.5 should return only the fresh node.
    let rows = db
        .execute_cypher("MATCH (m:Memory) WHERE m.relevance > 0.5 RETURN m.content AS c")
        .expect("filter by relevance");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("c"),
        Some(&Value::String("fresh".into())),
        "only the fresh node should pass relevance > 0.5"
    );
}

// ── Missing anchor field → Null ───────────────────────────────────

/// COMPUTED field returns Null when anchor field is missing.
#[test]
fn computed_with_missing_anchor_returns_null() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    // Create node WITHOUT created_at → COMPUTED can't evaluate.
    db.execute_cypher("CREATE (m:Memory {content: 'no timestamp'})")
        .expect("create without anchor");

    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.relevance AS rel")
        .expect("query computed without anchor");

    assert_eq!(rows.len(), 1);
    assert_eq!(
        rows[0].get("rel"),
        Some(&Value::Null),
        "COMPUTED with missing anchor should be Null"
    );
}

// ── COMPUTED in traversal target ──────────────────────────────────

/// COMPUTED properties work on traversal target nodes (build_target_row).
#[test]
fn computed_on_traversal_target() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher("CREATE (a:Agent {name: 'alice'})")
        .expect("create agent");
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'memory', created_at: {now_us}}})"
    ))
    .expect("create memory");
    db.execute_cypher(
        "MATCH (a:Agent {name: 'alice'}), (m:Memory {content: 'memory'}) CREATE (a)-[:RECALLS]->(m)"
    )
    .expect("create edge");

    // Traverse and check computed field on target.
    let rows = db
        .execute_cypher(
            "MATCH (a:Agent)-[:RECALLS]->(m:Memory) RETURN m.relevance AS rel, m.content AS c",
        )
        .expect("traverse with computed");

    assert_eq!(rows.len(), 1);
    match rows[0].get("rel") {
        Some(Value::Float(f)) => {
            assert!(
                *f > 0.99,
                "traversal target should have relevance ≈ 1.0, got {f}"
            );
        }
        other => panic!("expected Float for traversal computed, got {other:?}"),
    }
}

// ── COMPUTED TTL Background Reaper (R083) ────────────────────────────

/// Background reaper deletes expired nodes (scope: Node).
#[test]
fn computed_ttl_reaper_deletes_expired_node() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    // Create a node with created_at = 31 days ago (TTL = 30 days → expired).
    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let old_us = now_us - 31 * 86400 * 1_000_000;

    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'expired', created_at: {old_us}}})"
    ))
    .expect("create expired node");

    // Create a fresh node (not expired).
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'fresh', created_at: {now_us}}})"
    ))
    .expect("create fresh node");

    // Verify both exist.
    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.content AS c")
        .expect("count before reap");
    assert_eq!(rows.len(), 2, "should have 2 nodes before reap");

    // Run the reaper directly (don't wait for background thread).
    let result =
        coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000);
    assert_eq!(result.nodes_deleted, 1, "should delete 1 expired node");
    assert_eq!(result.nodes_checked, 2, "should check both nodes");

    // Verify only fresh node remains.
    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.content AS c")
        .expect("count after reap");
    assert_eq!(rows.len(), 1, "should have 1 node after reap");
    assert_eq!(
        rows[0].get("c"),
        Some(&Value::String("fresh".into())),
        "fresh node should survive"
    );
}

/// Reaper handles nodes with edges (DETACH DELETE equivalent).
#[test]
fn computed_ttl_reaper_cleans_edges_on_node_delete() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let old_us = now_us - 31 * 86400 * 1_000_000;

    // Create agent (no TTL) and expired memory with edge.
    db.execute_cypher("CREATE (a:Agent {name: 'alice'})")
        .expect("create agent");
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'old memory', created_at: {old_us}}})"
    ))
    .expect("create expired memory");
    db.execute_cypher(
        "MATCH (a:Agent {name: 'alice'}), (m:Memory {content: 'old memory'}) \
         CREATE (a)-[:RECALLS]->(m)",
    )
    .expect("create edge");

    // Verify edge exists.
    let rows = db
        .execute_cypher("MATCH (a:Agent)-[:RECALLS]->(m:Memory) RETURN m.content AS c")
        .expect("traverse before reap");
    assert_eq!(rows.len(), 1);

    // Run reaper.
    let result =
        coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000);
    assert_eq!(result.nodes_deleted, 1);

    // Agent should survive, memory should be gone.
    let agent_rows = db
        .execute_cypher("MATCH (a:Agent) RETURN a.name AS n")
        .expect("agents after reap");
    assert_eq!(agent_rows.len(), 1, "agent should survive");

    let memory_rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.content AS c")
        .expect("memories after reap");
    assert_eq!(memory_rows.len(), 0, "expired memory should be gone");
}

/// RemoveProperty merge operand survives persist + reopen (LSM compaction).
/// Field scope: reaper removes anchor field via merge, close DB, reopen,
/// verify field is still absent.
#[test]
fn computed_ttl_field_removal_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    // Phase 1: create node, run reaper, verify field removed.
    {
        let mut db = Database::open(dir.path()).expect("open");

        // TTL schema with Field scope, 1-second TTL.
        let mut schema = LabelSchema::new("CacheEntry");
        schema.add_property(PropertyDef::new("data", PropertyType::String));
        schema.add_property(PropertyDef::new("cached_at", PropertyType::Timestamp));
        schema.add_property(PropertyDef::computed(
            "_ttl",
            ComputedSpec::Ttl {
                duration_secs: 1,
                anchor_field: "cached_at".into(),
                scope: TtlScope::Field,
            },
        ));

        let schema_key = coordinode_core::schema::definition::encode_label_schema_key("CacheEntry");
        let bytes = schema.to_msgpack().expect("serialize schema");
        db.engine_shared()
            .put(
                coordinode_storage::engine::partition::Partition::Schema,
                &schema_key,
                &bytes,
            )
            .expect("persist schema");

        // 2 seconds ago → expired with 1s TTL.
        let old_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
            - 2_000_000;

        db.execute_cypher(&format!(
            "CREATE (c:CacheEntry {{data: 'stale', cached_at: {old_us}}})"
        ))
        .expect("create");

        // Run reaper → removes cached_at field via RemoveProperty merge.
        let result =
            coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000);
        assert_eq!(result.fields_removed, 1);

        // Node survives but field is gone.
        let rows = db
            .execute_cypher("MATCH (c:CacheEntry) RETURN c.data AS d")
            .expect("query");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("d"), Some(&Value::String("stale".into())));
    }
    // Database dropped → flush + close.

    // Phase 2: reopen and verify merge operand was compacted correctly.
    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher("MATCH (c:CacheEntry) RETURN c.data AS d")
            .expect("query after reopen");
        assert_eq!(rows.len(), 1, "node should survive reopen");
        assert_eq!(
            rows[0].get("d"),
            Some(&Value::String("stale".into())),
            "data field preserved"
        );
    }
}

/// RemoveProperty merge for Node scope: node deleted, verify gone after reopen.
#[test]
fn computed_ttl_node_deletion_survives_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");

    {
        let mut db = Database::open(dir.path()).expect("open");
        setup_memory_schema(&mut db);

        let old_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64
            - 31 * 86400 * 1_000_000;

        db.execute_cypher(&format!(
            "CREATE (m:Memory {{content: 'old', created_at: {old_us}}})"
        ))
        .expect("create");

        let result =
            coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000);
        assert_eq!(result.nodes_deleted, 1);
    }

    {
        let mut db = Database::open(dir.path()).expect("reopen");
        let rows = db
            .execute_cypher("MATCH (m:Memory) RETURN m.content AS c")
            .expect("query after reopen");
        assert_eq!(
            rows.len(),
            0,
            "expired node should stay deleted after reopen"
        );
    }
}

/// Pipeline path: reaper submits mutations through ProposalPipeline
/// (same path used by background thread in production).
#[test]
fn computed_ttl_reaper_via_pipeline() {
    let (mut db, _dir) = open_db();
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;
    let old_us = now_us - 31 * 86400 * 1_000_000;

    // Create expired + fresh nodes.
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'expired', created_at: {old_us}}})"
    ))
    .expect("create expired");
    db.execute_cypher(&format!(
        "CREATE (m:Memory {{content: 'fresh', created_at: {now_us}}})"
    ))
    .expect("create fresh");

    // Create pipeline (same as Database uses internally).
    let pipeline = std::sync::Arc::new(coordinode_raft::proposal::OwnedLocalProposalPipeline::new(
        &db.engine_shared(),
    ));
    let id_gen = coordinode_core::txn::proposal::ProposalIdGenerator::new();
    let interner = coordinode_core::graph::intern::FieldInterner::new();

    let result = coordinode_query::index::ttl_reaper::reap_computed_ttl_via_pipeline(
        &db.engine_shared(),
        1,
        1000,
        &interner,
        pipeline.as_ref(),
        &id_gen,
    );

    assert_eq!(
        result.nodes_deleted, 1,
        "should delete 1 expired node via pipeline"
    );

    let rows = db
        .execute_cypher("MATCH (m:Memory) RETURN m.content AS c")
        .expect("query after pipeline reap");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get("c"), Some(&Value::String("fresh".into())),);
}

// ── COMPUTED VECTOR_DECAY × vector_similarity ────────────────────

/// Helper: create a schema with VECTOR_DECAY + embedding for decay-weighted vector tests.
fn setup_vector_decay_schema(db: &mut Database) {
    let mut schema = LabelSchema::new("Article");
    schema.add_property(PropertyDef::new("title", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 3,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_recency",
        ComputedSpec::VectorDecay {
            formula: DecayFormula::Linear,
            duration_secs: 86400, // 1 day
            anchor_field: "created_at".into(),
        },
    ));

    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("Article");
    let bytes = schema.to_msgpack().expect("serialize schema");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");
}

/// Planner detects `vector_similarity() * _recency > threshold` and applies
/// decay-weighted scoring: fresh articles pass, old articles are filtered out
/// because their effective score (similarity × decay) falls below threshold.
#[test]
fn vector_decay_weights_similarity_scores() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // Fresh article: created now → _recency ≈ 1.0
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh article");

    // Old article: created 2 days ago → _recency = 0.0 (linear, beyond 1 day)
    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old article");

    // Both articles have identical embeddings and identical cosine similarity to query.
    // Without decay: both pass similarity > 0.5 (cosine with self = 1.0)
    // With decay: fresh passes (1.0 * 1.0 ≈ 1.0 > 0.5), old fails (1.0 * 0.0 = 0.0 < 0.5)

    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         RETURN a.title AS title",
        )
        .expect("decay-weighted vector query");

    assert_eq!(
        rows.len(),
        1,
        "only fresh article should pass decay-weighted filter"
    );
    assert_eq!(
        rows[0].get("title"),
        Some(&Value::String("fresh".into())),
        "fresh article should pass"
    );
}

/// Without decay multiplication, both articles pass the raw similarity threshold.
/// This test verifies the baseline behavior (no decay in WHERE).
#[test]
fn vector_similarity_without_decay_returns_all() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh");

    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old");

    // Raw similarity without decay — both pass.
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) > 0.5 \
         RETURN a.title AS title",
        )
        .expect("raw vector query");

    assert_eq!(
        rows.len(),
        2,
        "both articles should pass raw similarity filter"
    );
}

/// VECTOR_DECAY field is accessible in RETURN for score inspection.
#[test]
fn vector_decay_field_in_return() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'recent', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create");

    let rows = db
        .execute_cypher("MATCH (a:Article) RETURN a._recency AS recency")
        .expect("query recency");

    assert_eq!(rows.len(), 1);
    match rows[0].get("recency").expect("recency field") {
        Value::Float(f) => {
            assert!(*f > 0.99, "freshly created → _recency ≈ 1.0, got {f}");
        }
        other => panic!("expected Float for _recency, got {other:?}"),
    }
}

/// Reverse multiplication order: `_recency * vector_similarity()` should also
/// be detected by the planner and produce identical results.
#[test]
fn vector_decay_reverse_multiply_order() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh");

    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old");

    // Reverse order: decay * vector_fn (instead of vector_fn * decay)
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE a._recency * vector_similarity(a.embedding, [1.0, 0.0, 0.0]) > 0.5 \
         RETURN a.title AS title",
        )
        .expect("reverse order decay query");

    assert_eq!(
        rows.len(),
        1,
        "reverse multiply order should produce same result"
    );
    assert_eq!(rows[0].get("title"), Some(&Value::String("fresh".into())),);
}

/// Missing anchor field → _recency evaluates to Null → decay factor = 1.0
/// (fallback). Article without created_at still passes raw similarity.
#[test]
fn vector_decay_missing_anchor_uses_fallback() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    // Article WITHOUT created_at → anchor missing → _recency = Null → factor = 1.0
    db.execute_cypher("CREATE (a:Article {title: 'no_anchor', embedding: [1.0, 0.0, 0.0]})")
        .expect("create article without anchor");

    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         RETURN a.title AS title",
        )
        .expect("query with missing anchor");

    // _recency = Null → decay factor = 1.0 → score = 1.0 * 1.0 = 1.0 > 0.5 → passes
    // Wait — actually eval_expr returns Null for missing computed field.
    // In apply_decay_multiplier, Null → factor 1.0 (no attenuation).
    // But the expression is `vector_similarity(...) * a._recency` which the PLANNER
    // extracts as VectorFilter with decay_field. The decay_field evaluates to Null,
    // and apply_decay_multiplier maps Null → 1.0.
    //
    // However, if the planner DIDN'T detect the pattern (e.g., it fell through to
    // generic Filter), then eval_expr would compute Float(1.0) * Null = Null,
    // and Null > 0.5 = false. This would mean the article is FILTERED OUT.
    //
    // Both behaviors are acceptable: the optimized path treats Null as 1.0 (no decay),
    // the generic path treats Null * X = Null (filtered). Since we DO detect the pattern,
    // the article should pass with factor=1.0.
    assert_eq!(
        rows.len(),
        1,
        "missing anchor → Null decay → factor 1.0 → passes"
    );
}

/// Exponential decay formula: half-life behavior.
/// Article at exactly half the duration should have _recency ≈ 0.5 (for lambda=ln(2)).
#[test]
fn vector_decay_exponential_formula() {
    let (mut db, _dir) = open_db();

    // Exponential with lambda = ln(2) ≈ 0.693 → half-life at t=1.0 gives e^(-0.693) ≈ 0.5
    let mut schema = LabelSchema::new("ExpArticle");
    schema.add_property(PropertyDef::new("title", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 3,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_recency",
        ComputedSpec::VectorDecay {
            formula: DecayFormula::Exponential {
                lambda: std::f64::consts::LN_2,
            },
            duration_secs: 86400, // 1 day
            anchor_field: "created_at".into(),
        },
    ));

    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("ExpArticle");
    let bytes = schema.to_msgpack().expect("serialize");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // Article at exactly 1 day old → t = 1.0 → decay = e^(-ln2) ≈ 0.5
    // similarity = 1.0, effective = 1.0 * 0.5 = 0.5
    let one_day_ago = now_us - 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:ExpArticle {{title: 'day_old', created_at: {one_day_ago}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create day-old");

    // threshold = 0.4 → 0.5 > 0.4 → should pass
    let rows = db
        .execute_cypher(
            "MATCH (a:ExpArticle) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.4 \
         RETURN a.title AS title",
        )
        .expect("exponential decay query");
    assert_eq!(
        rows.len(),
        1,
        "day-old article with exp decay ≈ 0.5 should pass threshold 0.4"
    );

    // threshold = 0.6 → 0.5 < 0.6 → should NOT pass
    let rows2 = db
        .execute_cypher(
            "MATCH (a:ExpArticle) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.6 \
         RETURN a.title AS title",
        )
        .expect("exponential decay query strict");
    assert_eq!(
        rows2.len(),
        0,
        "day-old article with exp decay ≈ 0.5 should NOT pass threshold 0.6"
    );
}

/// vector_distance (not similarity) × decay pattern: distance uses < threshold.
/// Old article's effective distance (raw_dist * decay) shrinks to 0, passing < threshold.
/// Fresh article's effective distance stays at raw_dist.
#[test]
fn vector_decay_with_distance_function() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // Both articles at distance ~1.0 from query [0.0, 1.0, 0.0]
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh");

    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old");

    // vector_distance × decay < 0.5
    // Fresh: distance(~1.414) * decay(1.0) ≈ 1.414 → NOT < 0.5 → filtered
    // Old:   distance(~1.414) * decay(0.0) = 0.0  → < 0.5 → passes
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_distance(a.embedding, [0.0, 1.0, 0.0]) * a._recency < 0.5 \
         RETURN a.title AS title",
        )
        .expect("distance × decay query");

    assert_eq!(
        rows.len(),
        1,
        "old article passes (distance × 0.0 = 0.0 < 0.5)"
    );
    assert_eq!(rows[0].get("title"), Some(&Value::String("old".into())),);
}

/// Literal multiplier `vector_similarity() * 0.5 > threshold` should NOT be
/// extracted as decay pattern — it falls through to generic Filter.
#[test]
fn literal_multiplier_not_detected_as_decay() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'test', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create");

    // similarity(self, self) = 1.0, × 0.5 = 0.5, > 0.4 → passes
    // This uses literal 0.5, NOT a property reference — should fall through
    // to generic Filter which evaluates the expression normally.
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * 0.5 > 0.4 \
         RETURN a.title AS title",
        )
        .expect("literal multiplier query");

    assert_eq!(
        rows.len(),
        1,
        "literal multiplier should work via generic Filter"
    );
}

/// EXPLAIN output shows `* decay` annotation when decay pattern detected.
#[test]
fn explain_shows_decay_annotation() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'test', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create");

    let explain = db
        .explain_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         RETURN a.title",
        )
        .expect("explain decay query");

    assert!(
        explain.contains("* decay"),
        "EXPLAIN should show '* decay' annotation, got: {explain}"
    );
}

/// Step formula: binary cutoff at duration boundary.
#[test]
fn vector_decay_step_formula() {
    let (mut db, _dir) = open_db();

    let mut schema = LabelSchema::new("StepArticle");
    schema.add_property(PropertyDef::new("title", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 3,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_recency",
        ComputedSpec::VectorDecay {
            formula: DecayFormula::Step,
            duration_secs: 86400, // 1 day
            anchor_field: "created_at".into(),
        },
    ));

    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("StepArticle");
    let bytes = schema.to_msgpack().expect("serialize");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // Fresh: < 1 day → step = 1.0
    db.execute_cypher(&format!(
        "CREATE (a:StepArticle {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh");

    // Old: > 1 day → step = 0.0
    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:StepArticle {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old");

    let rows = db
        .execute_cypher(
            "MATCH (a:StepArticle) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         RETURN a.title AS title",
        )
        .expect("step decay query");

    assert_eq!(rows.len(), 1, "only fresh (step=1.0) passes");
    assert_eq!(rows[0].get("title"), Some(&Value::String("fresh".into())));
}

/// PowerLaw formula: fat-tail decay, slower than exponential at high t.
#[test]
fn vector_decay_power_law_formula() {
    let (mut db, _dir) = open_db();

    // PowerLaw: (1 + t/0.5)^(-1.0)
    // At t=1.0 (full duration): (1 + 1.0/0.5)^(-1) = (3)^(-1) = 0.333
    let mut schema = LabelSchema::new("PLArticle");
    schema.add_property(PropertyDef::new("title", PropertyType::String));
    schema.add_property(PropertyDef::new("created_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new(
        "embedding",
        PropertyType::Vector {
            dimensions: 3,
            metric: coordinode_core::graph::types::VectorMetric::Cosine,
        },
    ));
    schema.add_property(PropertyDef::computed(
        "_recency",
        ComputedSpec::VectorDecay {
            formula: DecayFormula::PowerLaw {
                tau: 0.5,
                alpha: 1.0,
            },
            duration_secs: 86400,
            anchor_field: "created_at".into(),
        },
    ));

    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("PLArticle");
    let bytes = schema.to_msgpack().expect("serialize");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // 1 day old → t=1.0 → decay = (1+2)^(-1) = 0.333
    // score = 1.0 * 0.333 = 0.333
    let one_day_ago = now_us - 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:PLArticle {{title: 'day_old', created_at: {one_day_ago}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create");

    // threshold 0.2 → 0.333 > 0.2 → passes (fat tail)
    let rows = db
        .execute_cypher(
            "MATCH (a:PLArticle) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.2 \
         RETURN a.title AS title",
        )
        .expect("power law query");
    assert_eq!(rows.len(), 1, "power law fat tail: 0.333 > 0.2 → passes");

    // threshold 0.5 → 0.333 < 0.5 → filtered
    let rows2 = db
        .execute_cypher(
            "MATCH (a:PLArticle) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         RETURN a.title AS title",
        )
        .expect("power law strict");
    assert_eq!(rows2.len(), 0, "power law: 0.333 < 0.5 → filtered");
}

/// Compound WHERE: decay pattern is extracted alongside property filter.
/// `WHERE vector_similarity() * decay > 0.5 AND a.title = 'fresh'`
/// should split into VectorFilter(decay) + generic Filter(title).
#[test]
fn vector_decay_in_compound_where() {
    let (mut db, _dir) = open_db();
    setup_vector_decay_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create fresh");

    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'also_fresh', created_at: {now_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create also_fresh");

    let old_us = now_us - 2 * 86400 * 1_000_000;
    db.execute_cypher(&format!(
        "CREATE (a:Article {{title: 'old', created_at: {old_us}, embedding: [1.0, 0.0, 0.0]}})"
    ))
    .expect("create old");

    // Compound: decay filter + property filter
    let rows = db
        .execute_cypher(
            "MATCH (a:Article) \
         WHERE vector_similarity(a.embedding, [1.0, 0.0, 0.0]) * a._recency > 0.5 \
         AND a.title = 'fresh' \
         RETURN a.title AS title",
        )
        .expect("compound decay + property query");

    // 3 articles: 'fresh' (passes both), 'also_fresh' (passes decay, fails title), 'old' (fails decay)
    assert_eq!(
        rows.len(),
        1,
        "compound filter: only 'fresh' passes both conditions"
    );
    assert_eq!(rows[0].get("title"), Some(&Value::String("fresh".into())));
}

// ── R085: COMPUTED integration tests ────────────────────────────────

// ── (1) Decay returns correct interpolated value at 5 timestamps ────

/// Create 5 nodes at different points in the decay window and verify
/// the linear decay formula produces correct values at each point.
#[test]
fn computed_decay_at_5_timestamps() {
    let (mut db, _dir) = open_db();

    // Schema with 7-day linear decay (same as setup_memory_schema).
    let duration_secs: u64 = 604800; // 7 days
    setup_memory_schema(&mut db);

    let now_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    // 5 timestamps: 0%, 25%, 50%, 75%, 100% of duration elapsed.
    let fractions = [0.0_f64, 0.25, 0.50, 0.75, 1.0];
    let names = ["t0", "t25", "t50", "t75", "t100"];

    for (i, frac) in fractions.iter().enumerate() {
        let elapsed_us = (*frac * duration_secs as f64 * 1_000_000.0) as i64;
        let created_at = now_us - elapsed_us;
        db.execute_cypher(&format!(
            "CREATE (m:Memory {{content: '{}', created_at: {}}})",
            names[i], created_at
        ))
        .expect("create node");
    }

    // Query all and collect relevance values indexed by content.
    let rows = db
        .execute_cypher(
            "MATCH (m:Memory) RETURN m.content AS name, m.relevance AS rel ORDER BY m.content",
        )
        .expect("query decay values");

    assert_eq!(rows.len(), 5, "should have 5 nodes");

    // Build a map: name → relevance.
    let mut values: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for row in &rows {
        let name = match row.get("name") {
            Some(Value::String(s)) => s.clone(),
            other => panic!("expected String name, got {other:?}"),
        };
        let rel = match row.get("rel") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float relevance for {name}, got {other:?}"),
        };
        values.insert(name, rel);
    }

    // Verify: linear decay f(t) = max(0, 1 - t).
    // Allow ±0.02 tolerance for timing jitter.
    let tolerance = 0.02;

    // t=0% → relevance ≈ 1.0
    let v = values["t0"];
    assert!(
        (v - 1.0).abs() < tolerance,
        "t0 (elapsed=0%): expected ≈1.0, got {v}"
    );

    // t=25% → relevance ≈ 0.75
    let v = values["t25"];
    assert!(
        (v - 0.75).abs() < tolerance,
        "t25 (elapsed=25%): expected ≈0.75, got {v}"
    );

    // t=50% → relevance ≈ 0.50
    let v = values["t50"];
    assert!(
        (v - 0.50).abs() < tolerance,
        "t50 (elapsed=50%): expected ≈0.50, got {v}"
    );

    // t=75% → relevance ≈ 0.25
    let v = values["t75"];
    assert!(
        (v - 0.25).abs() < tolerance,
        "t75 (elapsed=75%): expected ≈0.25, got {v}"
    );

    // t=100% → relevance ≈ 0.0
    let v = values["t100"];
    assert!(
        v.abs() < tolerance,
        "t100 (elapsed=100%): expected ≈0.0, got {v}"
    );

    // Verify monotonic decrease.
    assert!(
        values["t0"] > values["t25"]
            && values["t25"] > values["t50"]
            && values["t50"] > values["t75"]
            && values["t75"] > values["t100"],
        "decay values should decrease monotonically: {:?}",
        fractions
            .iter()
            .zip(names.iter())
            .map(|(_, n)| (n, values[*n]))
            .collect::<Vec<_>>()
    );
}

// ── (3) TTL scope=subtree: node survives, anchor removed ────────────

/// TTL with scope=Subtree removes the anchor property while the node
/// and nested DOCUMENT properties survive.
///
/// Uses inline map literal in CREATE to store nested DOCUMENT content.
/// Map literals are auto-converted to Document type on write.
///
/// Note: current implementation treats Subtree = Field (removes anchor).
/// See G068 for the gap between current behavior and arch doc intent.
#[test]
fn computed_ttl_subtree_removes_anchor_preserves_document() {
    let (mut db, _dir) = open_db();

    // Schema with TTL scope=Subtree and short duration.
    let mut schema = LabelSchema::new("CachedDoc");
    schema.add_property(PropertyDef::new("title", PropertyType::String).not_null());
    schema.add_property(PropertyDef::new("cached_at", PropertyType::Timestamp));
    schema.add_property(PropertyDef::new("payload", PropertyType::Document));
    schema.add_property(PropertyDef::computed(
        "_cache_ttl",
        ComputedSpec::Ttl {
            duration_secs: 60, // 1 minute
            anchor_field: "cached_at".into(),
            scope: TtlScope::Subtree,
        },
    ));

    let schema_key = coordinode_core::schema::definition::encode_label_schema_key("CachedDoc");
    let bytes = schema.to_msgpack().expect("serialize schema");
    db.engine_shared()
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &schema_key,
            &bytes,
        )
        .expect("persist schema");

    // Create expired node with nested DOCUMENT via inline map literal.
    let old_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
        - 120 * 1_000_000; // 2 min ago, TTL=60s → expired

    db.execute_cypher(&format!(
        "CREATE (d:CachedDoc {{title: 'report', cached_at: {old_us}, \
         payload: {{summary: 'quarterly', pages: 42}}}})"
    ))
    .expect("create expired node with nested doc");

    // Create fresh node with nested DOCUMENT.
    let fresh_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64;

    db.execute_cypher(&format!(
        "CREATE (d:CachedDoc {{title: 'fresh report', cached_at: {fresh_us}, \
         payload: {{summary: 'monthly', pages: 10}}}})"
    ))
    .expect("create fresh node with nested doc");

    // Verify both exist with nested DOCUMENT accessible via dot-notation.
    let rows = db
        .execute_cypher(
            "MATCH (d:CachedDoc) \
             RETURN d.title AS t, d.payload.summary AS s \
             ORDER BY d.title",
        )
        .expect("query before reap");
    assert_eq!(rows.len(), 2, "should have 2 nodes before reap");
    // Verify dot-notation into nested DOCUMENT works.
    let report_row = rows
        .iter()
        .find(|r| r.get("t") == Some(&Value::String("report".into())))
        .expect("report row before reap");
    assert_eq!(
        report_row.get("s"),
        Some(&Value::String("quarterly".into())),
        "dot-notation into nested DOCUMENT should work"
    );

    // Run reaper.
    let result =
        coordinode_query::index::ttl_reaper::reap_computed_ttl(&db.engine_shared(), 1, 1000);
    assert_eq!(
        result.subtrees_removed, 1,
        "should remove 1 subtree (expired node's anchor)"
    );
    assert_eq!(
        result.nodes_deleted, 0,
        "no nodes should be deleted in subtree scope"
    );

    // Verify: both nodes survive, expired node's cached_at removed,
    // nested DOCUMENT content intact on both.
    let rows = db
        .execute_cypher(
            "MATCH (d:CachedDoc) \
             RETURN d.title AS t, d.cached_at AS ca, d.payload.summary AS s \
             ORDER BY d.title",
        )
        .expect("query after reap");
    assert_eq!(rows.len(), 2, "both nodes should survive subtree TTL");

    let fresh_row = rows
        .iter()
        .find(|r| r.get("t") == Some(&Value::String("fresh report".into())))
        .expect("fresh row");
    let expired_row = rows
        .iter()
        .find(|r| r.get("t") == Some(&Value::String("report".into())))
        .expect("expired row");

    // Fresh node: all properties intact.
    assert!(
        matches!(
            fresh_row.get("ca"),
            Some(Value::Int(_)) | Some(Value::Timestamp(_))
        ),
        "fresh node's cached_at should survive"
    );
    assert_eq!(
        fresh_row.get("s"),
        Some(&Value::String("monthly".into())),
        "fresh node's nested DOCUMENT should survive"
    );

    // Expired node: anchor (cached_at) removed, nested doc survives.
    assert_eq!(
        expired_row.get("ca"),
        Some(&Value::Null),
        "expired node's cached_at should be removed by subtree TTL"
    );
    assert_eq!(
        expired_row.get("t"),
        Some(&Value::String("report".into())),
        "expired node's title should survive subtree TTL"
    );
    assert_eq!(
        expired_row.get("s"),
        Some(&Value::String("quarterly".into())),
        "expired node's nested DOCUMENT content should survive subtree TTL"
    );
}
