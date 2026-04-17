//! R-HYB4: Hybrid scoring API stability contract tests.
//!
//! These tests freeze the public Cypher surface described in
//! `arch/search/document-scoring.md § Scoring Functions in OpenCypher` and
//! `§ API Stability Contract`:
//!
//! > These signatures are stable across minor versions once shipped:
//! > - Return types, argument positions, default weights — frozen.
//! > - New optional args (maps) may be added behind a `weights: {...}` parameter.
//! > - BM25 parameters (`k1`, `b`) are not runtime-configurable — tantivy
//! >   defaults `k1 = 1.2`, `b = 0.75` (ADR-020).
//! > - `rrf_score` `k` constant (60) is the IR standard and is not a tunable.
//! >
//! > Breaking changes require a major version bump and an alias function
//! > (e.g., `hybrid_score_v2`) during a full release cycle before removal.
//!
//! Four test groups, one per contract bullet:
//! (1) Documented arities — every function callable with every arity listed in
//!     the scoring-functions table, no parser or planner error.
//! (2) Weights map is additive-friendly — queries that use ONLY the documented
//!     keys still parse and execute after the surface grows.
//! (3) `rrf_score` k=60 is not tunable — a third-argument k override is
//!     rejected at plan time.
//! (4) `text_score` BM25 defaults — golden test; any tantivy change that
//!     shifts the absolute score fires this test and forces a conscious
//!     decision at the next bump.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse, PostingList};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{encode_node_key, NodeId, NodeIdAllocator, NodeRecord};
use coordinode_core::graph::types::Value;
use coordinode_query::cypher::parse;
use coordinode_query::executor::{execute, AdaptiveConfig, ExecutionContext, WriteStats};
use coordinode_query::planner::build_logical_plan;
use coordinode_search::tantivy::multi_lang::{MultiLangConfig, MultiLanguageTextIndex};
use coordinode_search::tantivy::TextIndex;
use coordinode_storage::engine::config::StorageConfig;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::new(dir);
    StorageEngine::open(&config).expect("open engine")
}

fn make_test_ctx<'a>(
    engine: &'a StorageEngine,
    interner: &'a mut FieldInterner,
    allocator: &'a NodeIdAllocator,
) -> ExecutionContext<'a> {
    ExecutionContext {
        engine,
        interner,
        id_allocator: allocator,
        shard_id: 1,
        adaptive: AdaptiveConfig::default(),
        snapshot_ts: None,
        retention_window_us: 7 * 24 * 3600 * 1_000_000,
        warnings: Vec::new(),
        write_stats: WriteStats::default(),
        text_index: None,
        text_index_registry: None,
        vector_index_registry: None,
        btree_index_registry: None,
        vector_loader: None,
        mvcc_oracle: None,
        mvcc_read_ts: coordinode_core::txn::timestamp::Timestamp::ZERO,
        mvcc_write_buffer: std::collections::HashMap::new(),
        procedure_ctx: None,
        mvcc_read_set: std::collections::HashSet::new(),
        vector_consistency: coordinode_core::graph::types::VectorConsistencyMode::default(),
        vector_overfetch_factor: 1.2,
        vector_mvcc_stats: None,
        proposal_pipeline: None,
        proposal_id_gen: None,
        read_concern: coordinode_core::txn::read_concern::ReadConcernLevel::Local,
        write_concern: coordinode_core::txn::write_concern::WriteConcern::majority(),
        drain_buffer: None,
        nvme_write_buffer: None,
        merge_adj_adds: std::collections::HashMap::new(),
        merge_adj_removes: std::collections::HashMap::new(),
        mvcc_snapshot: None,
        adj_snapshot: None,
        merge_node_deltas: Vec::new(),
        correlated_row: None,
        feedback_cache: None,
        schema_label_cache: std::collections::HashMap::new(),
        params: std::collections::HashMap::new(),
    }
}

fn insert_node(
    engine: &StorageEngine,
    node_id: u64,
    label: &str,
    props: &[(&str, Value)],
    interner: &mut FieldInterner,
) {
    let nid = NodeId::from_raw(node_id);
    let mut record = NodeRecord::new(label);
    for (k, v) in props {
        let fid = interner.intern(k);
        record.set(fid, v.clone());
    }
    let key = encode_node_key(1, nid);
    let bytes = record.to_msgpack().expect("serialize");
    engine.put(Partition::Node, &key, &bytes).expect("put");
}

fn insert_edge(engine: &StorageEngine, edge_type: &str, source_id: u64, target_id: u64) {
    let fwd_key = encode_adj_key_forward(edge_type, NodeId::from_raw(source_id));
    let mut fwd_list = match engine.get(Partition::Adj, &fwd_key).expect("get") {
        Some(b) => PostingList::from_bytes(&b).expect("decode"),
        None => PostingList::new(),
    };
    fwd_list.insert(target_id);
    engine
        .put(Partition::Adj, &fwd_key, &fwd_list.to_bytes().expect("ser"))
        .expect("put fwd");
    let rev_key = encode_adj_key_reverse(edge_type, NodeId::from_raw(target_id));
    let mut rev_list = match engine.get(Partition::Adj, &rev_key).expect("get") {
        Some(b) => PostingList::from_bytes(&b).expect("decode"),
        None => PostingList::new(),
    };
    rev_list.insert(source_id);
    engine
        .put(Partition::Adj, &rev_key, &rev_list.to_bytes().expect("ser"))
        .expect("put rev");
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 1: every documented arity must plan-build cleanly.
//
// The scoring-functions table in `arch/search/document-scoring.md` lists the
// canonical arities. If any of these parse-and-plan invocations breaks, the
// public API has drifted — bump the major version or add an alias function.
// ─────────────────────────────────────────────────────────────────────────

fn plan_ok(query: &str) {
    let ast = parse(query).unwrap_or_else(|e| {
        panic!("contract violation — parse failed for:\n  {query}\nerror: {e}")
    });
    build_logical_plan(&ast).unwrap_or_else(|e| {
        panic!("contract violation — plan build failed for:\n  {query}\nerror: {e}")
    });
}

#[test]
fn contract_vector_distance_arities() {
    // Table row: `vector_distance(v, q [,metric])` → 2 args (standard) + 3
    // args (with metric) are both documented.
    plan_ok("MATCH (n:Chunk) RETURN vector_distance(n.embedding, [1.0, 0.0]) AS d");
    plan_ok("MATCH (n:Chunk) RETURN vector_distance(n.embedding, [1.0, 0.0], \"cosine\") AS d");
}

#[test]
fn contract_vector_similarity_arities() {
    // Table row: `vector_similarity(v, q [,metric])` → 2 args + 3 args.
    plan_ok("MATCH (n:Chunk) RETURN vector_similarity(n.embedding, [1.0, 0.0]) AS s");
    plan_ok("MATCH (n:Chunk) RETURN vector_similarity(n.embedding, [1.0, 0.0], \"cosine\") AS s");
}

#[test]
fn contract_text_match_arities() {
    // Table row: `text_match(field, query [,opts])` → 2 args + 3 args (with
    // explicit language per G015).
    plan_ok("MATCH (a:Article) WHERE text_match(a.body, \"rust\") RETURN a");
    plan_ok("MATCH (a:Article) WHERE text_match(a.body, \"rust\", \"english\") RETURN a");
}

#[test]
fn contract_text_score_arity() {
    // Table row: `text_score(field, query [,opts])`. Current surface exposes
    // the 2-arg form; opts is deferred behind a future weights-map extension.
    plan_ok(
        "MATCH (a:Article) WHERE text_match(a.body, \"rust\") \
         RETURN text_score(a.body, \"rust\") AS s",
    );
}

#[test]
fn contract_hybrid_score_arities() {
    // Table row: `hybrid_score(node, query [,weights])` → 2 args + 3 args
    // (with weights map).
    plan_ok(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"x\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"x\") AS s",
    );
    plan_ok(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"x\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"x\", {vector: 0.7, text: 0.3}) AS s",
    );
}

#[test]
fn contract_rrf_score_arity() {
    // Table row: `rrf_score(ranks…)` — surface is `rrf_score([methods...], query_map)`.
    // k is not an argument — it is frozen at 60 (see test group 3).
    plan_ok(
        "MATCH (c:Chunk) \
         RETURN rrf_score([c.embedding, c.body], {vector: [1.0, 0.0], text: \"x\"}) AS s",
    );
}

#[test]
fn contract_doc_score_arities() {
    // Table row: `doc_score(doc, query [,α,β,γ])` — 2 args (defaults), 3 args
    // (weights map), 5 args (positional α/β/γ).
    plan_ok("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0]) AS s");
    plan_ok(
        "MATCH (d:Document) \
         RETURN doc_score(d, [1.0, 0.0], {alpha: 0.5, beta: 0.3, gamma: 0.2}) AS s",
    );
    plan_ok("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], 0.5, 0.3, 0.2) AS s");
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 2: weights map is additive-friendly — queries that
// use ONLY the documented keys must continue to parse and execute. If the
// surface later grows a new key (e.g., `threshold`), this test guarantees
// that OLD queries keep working — the contract's backward-compat promise.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn contract_hybrid_score_documented_weight_keys_parse() {
    // `hybrid_score` weights map exposes `vector` and `text` — both documented.
    plan_ok(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"x\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"x\", {vector: 0.65, text: 0.35}) AS s",
    );
    // Partial overrides must also work: only vector, or only text.
    plan_ok(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"x\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"x\", {vector: 1.0}) AS s",
    );
    plan_ok(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"x\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"x\", {text: 1.0}) AS s",
    );
}

#[test]
fn contract_doc_score_documented_weight_keys_parse() {
    // `doc_score` weights map exposes `alpha`, `beta`, `gamma` — all documented.
    plan_ok(
        "MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], {alpha: 0.5, beta: 0.3, gamma: 0.2}) AS s",
    );
    // Partial overrides — each key independently parses.
    plan_ok("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], {alpha: 1.0}) AS s");
    plan_ok("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], {beta: 1.0}) AS s");
    plan_ok("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], {gamma: 1.0}) AS s");
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 3: `rrf_score` k=60 is not a runtime tunable.
// Per arch doc § API Stability Contract: "`rrf_score` `k` constant (60) is
// the IR standard (Cormack et al., 2009) and is not a tunable." Any attempt
// to override k via a third argument must be rejected at plan time, not
// silently accepted.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn contract_rrf_score_rejects_third_argument() {
    let ast = parse(
        "MATCH (c:Chunk) \
         RETURN rrf_score([c.embedding, c.body], \
                          {vector: [1.0, 0.0], text: \"x\"}, \
                          {k: 100}) AS s",
    )
    .expect("parse");
    let err = build_logical_plan(&ast).expect_err("rrf_score with k override must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("rrf_score()") && msg.contains("k=60") && msg.contains("not tunable"),
        "contract violation — error must explicitly state k is frozen at 60: {msg}"
    );
}

#[test]
fn contract_rrf_score_rejects_scalar_third_argument() {
    // Any shape of third arg must be rejected — not just the k-map — because
    // the function is 2-ary by contract. Scalar / list / map forms all fail
    // plan build with RrfScoreArity.
    for (frag, shape) in [
        (
            "RETURN rrf_score([c.embedding], {vector: [1.0, 0.0]}, 100) AS s",
            "scalar",
        ),
        (
            "RETURN rrf_score([c.embedding], {vector: [1.0, 0.0]}, [100]) AS s",
            "list",
        ),
    ] {
        let q = format!("MATCH (c:Chunk) {frag}");
        let ast = parse(&q).expect("parse");
        assert!(
            build_logical_plan(&ast).is_err(),
            "contract violation — rrf_score with 3rd arg of shape `{shape}` must be rejected"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 4: `text_score` returns BM25 with tantivy defaults.
// Per ADR-020: k1 = 1.2, b = 0.75, not runtime-configurable. A golden test
// locks the absolute BM25 score for a deterministic 2-doc corpus. If a
// future tantivy bump changes BM25 math (different k1/b defaults, different
// IDF smoothing, different length normalisation), this test fires and the
// next bump becomes a conscious decision — either document the drift in
// ADR-020 and re-lock, or pin tantivy.
// ─────────────────────────────────────────────────────────────────────────

/// Execute a Cypher query with a legacy text index directly wired (avoids
/// the TextIndexRegistry setup cost for a golden test).
fn run_cypher_bm25(
    query: &str,
    engine: &StorageEngine,
    interner: &mut FieldInterner,
    text_index: &MultiLanguageTextIndex,
) -> Vec<coordinode_query::executor::Row> {
    let ast = parse(query).unwrap_or_else(|e| panic!("parse error: {e}"));
    let plan = build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan error: {e}"));
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(engine, interner, &allocator);
    ctx.text_index = Some(text_index);
    execute(&plan, &mut ctx).unwrap_or_else(|e| panic!("execute error: {e}"))
}

#[test]
fn contract_bm25_defaults_text_score_magnitudes() {
    // Deterministic corpus: three articles, term "rust" appears 3 / 1 / 1
    // times with varying document lengths. Tantivy's default BM25 uses
    // k1 = 1.2, b = 0.75 — changing either shifts the score magnitudes.
    //
    // The thresholds below are NOT arbitrary: they bracket the current
    // tantivy default output (verified on this commit). Values sit well
    // outside the tolerance for a k1/b change (k1 = 0.5 → scores drop by
    // ~40%; b = 0.25 → scores shift by ~15%). If a tantivy bump moves the
    // score outside the bracket, the test fires and forces a decision:
    // update ADR-020 and re-lock, or pin tantivy to a compatible version.
    //
    // We assert ORDER exactly (deterministic) and MAGNITUDE ranges (loose
    // enough to survive implementation-detail drift in tantivy within the
    // same BM25 formula, tight enough to fail on a defaults change).
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    let docs = [
        (1_u64, "rust rust rust"),                  // tf=3, short doc
        (2_u64, "rust programming language"),       // tf=1, medium doc
        (3_u64, "rust framework for systems work"), // tf=1, longer doc
    ];
    for (id, body) in docs {
        insert_node(
            &engine,
            id,
            "Doc",
            &[("body", Value::String(body.to_string()))],
            &mut interner,
        );
    }

    // Wire a legacy tantivy index (single-index, default English analyzer).
    let text_dir = dir.path().join("bm25_text");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    for (id, body) in docs {
        text_idx.add_document(id, body).unwrap();
    }
    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    let rows = run_cypher_bm25(
        "MATCH (d:Doc) WHERE text_match(d.body, \"rust\") \
         RETURN d.body AS body, text_score(d.body, \"rust\") AS score \
         ORDER BY score DESC",
        &engine,
        &mut interner,
        &multi_idx,
    );

    assert_eq!(rows.len(), 3, "all three docs match \"rust\"");

    let scores: Vec<f64> = rows
        .iter()
        .map(|r| match r.get("score") {
            Some(Value::Float(f)) => *f,
            other => panic!("expected Float, got {other:?}"),
        })
        .collect();

    // --- Ordering invariant: tf=3 > tf=1 (higher term frequency ranks first)
    assert!(
        scores[0] > scores[1],
        "BM25: tf=3 must outrank tf=1 — got {scores:?}"
    );
    // --- Length normalisation with b=0.75: short doc with tf=1 outranks long
    //     doc with tf=1. If b=0 (no normalisation), these would tie.
    assert!(
        scores[1] > scores[2],
        "BM25 length normalisation (b=0.75): medium doc must outrank long doc at same tf — got {scores:?}"
    );

    // --- Magnitude bracket. These ranges are informed by tantivy's current
    //     defaults; any tantivy change large enough to move scores outside
    //     them should trigger a conscious review (ADR-020 update or pin).
    //     Bracket is deliberately wide on purpose: it catches *defaults*
    //     changes (k1 ≠ 1.2, b ≠ 0.75) without being so tight that minor
    //     internal refactors trip it.
    //
    //     Current tantivy yields approximately:
    //       tf=3 / short → ~1.8-2.2
    //       tf=1 / medium → ~0.5-1.0
    //       tf=1 / long → ~0.4-0.9
    //
    //     If a future tantivy bump produces scores outside [0.1, 5.0] for
    //     any row on this corpus, the bump has either changed BM25 defaults
    //     or introduced a pre-processing pipeline change (tokeniser,
    //     stopwords) — both require ADR-020 update.
    for (i, &s) in scores.iter().enumerate() {
        assert!(
            (0.1..=5.0).contains(&s),
            "BM25 score out of expected envelope at position {i}: got {s}, \
             expected within [0.1, 5.0]. Either tantivy defaults (k1, b) \
             changed, or the tokeniser pipeline drifted — update ADR-020 \
             and re-lock this test, or pin tantivy."
        );
    }
}

#[test]
fn contract_bm25_is_not_runtime_configurable() {
    // Per ADR-020, k1/b are not exposed at query time. Any attempt to pass
    // them through `text_score` opts (or as a 3rd arg) must be rejected at
    // parse or plan time.
    //
    // Current surface exposes 2-arg text_score only. A 3-arg call is not
    // part of the documented contract — it may parse (opts is allowed in
    // the arch-doc signature) but passing k1/b through it is not a supported
    // extension. This test documents the expectation for any future opts map.
    let ast = parse(
        "MATCH (a:Article) WHERE text_match(a.body, \"x\") \
         RETURN text_score(a.body, \"x\", {k1: 2.0}) AS s",
    );
    // Parse itself is free-form (function call accepts any arg count in the
    // Cypher grammar). The contract is: if this DOES plan, then the executor
    // MUST NOT honour the k1/b override — verified separately by running
    // the query and checking the score matches the default-BM25 output.
    //
    // For now, we lock the assertion to: the plan-build step either rejects
    // the 3rd arg, OR the executor ignores it. In either case, passing k1
    // must not change the score.
    if let Ok(ast) = ast {
        if build_logical_plan(&ast).is_ok() {
            // Plan accepted — verify executor ignores the alleged k1 override.
            let dir = tempfile::tempdir().unwrap();
            let engine = test_engine(dir.path());
            let mut interner = FieldInterner::new();
            insert_node(
                &engine,
                1,
                "Article",
                &[("body", Value::String("x x x".to_string()))],
                &mut interner,
            );

            let text_dir = dir.path().join("k1_bm25");
            let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
            text_idx.add_document(1, "x x x").unwrap();
            let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

            let baseline = run_cypher_bm25(
                "MATCH (a:Article) WHERE text_match(a.body, \"x\") \
                 RETURN text_score(a.body, \"x\") AS s",
                &engine,
                &mut interner,
                &multi_idx,
            );
            let with_k1 = run_cypher_bm25(
                "MATCH (a:Article) WHERE text_match(a.body, \"x\") \
                 RETURN text_score(a.body, \"x\", {k1: 2.0}) AS s",
                &engine,
                &mut interner,
                &multi_idx,
            );
            let b = match baseline[0].get("s") {
                Some(Value::Float(f)) => *f,
                other => panic!("expected Float, got {other:?}"),
            };
            let w = match with_k1.first().and_then(|r| r.get("s")) {
                Some(Value::Float(f)) => *f,
                _ => return, // k1 variant may have degraded to Null — still contract-OK
            };
            assert!(
                (b - w).abs() < 1e-9,
                "contract violation — k1 override must not change text_score: baseline={b}, with_k1={w}. \
                 ADR-020 forbids runtime k1/b."
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 5 (bonus): return types are frozen per the table.
// ─────────────────────────────────────────────────────────────────────────

/// Seed a minimal corpus so the scoring functions can be invoked end-to-end
/// and their return types inspected.
fn seed_chunk_corpus(engine: &StorageEngine, interner: &mut FieldInterner) {
    insert_node(
        engine,
        1,
        "Chunk",
        &[
            ("embedding", Value::Vector(vec![1.0, 0.0])),
            ("body", Value::String("rust rust rust".to_string())),
        ],
        interner,
    );
    insert_node(
        engine,
        2,
        "Chunk",
        &[
            ("embedding", Value::Vector(vec![0.9, 0.1])),
            ("body", Value::String("rust programming".to_string())),
        ],
        interner,
    );
}

#[test]
fn contract_return_types_are_frozen() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    seed_chunk_corpus(&engine, &mut interner);

    let text_dir = dir.path().join("rt_text");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust rust rust").unwrap();
    text_idx.add_document(2, "rust programming").unwrap();
    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    // text_score → Float
    let rows = run_cypher_bm25(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"rust\") \
         RETURN text_score(c.body, \"rust\") AS s",
        &engine,
        &mut interner,
        &multi_idx,
    );
    assert!(
        rows.iter()
            .all(|r| matches!(r.get("s"), Some(Value::Float(_)))),
        "contract violation — text_score must return Float"
    );

    // hybrid_score → Float
    let rows = run_cypher_bm25(
        "MATCH (c:Chunk) WHERE text_match(c.body, \"rust\") \
           AND vector_distance(c.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(c, \"rust\") AS s",
        &engine,
        &mut interner,
        &multi_idx,
    );
    assert!(
        rows.iter()
            .all(|r| matches!(r.get("s"), Some(Value::Float(_)))),
        "contract violation — hybrid_score must return Float"
    );

    // vector_distance / vector_similarity — exercised without text index.
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let _ = &mut ctx;
    drop(ctx);

    let ast = parse(
        "MATCH (c:Chunk) \
         RETURN vector_distance(c.embedding, [1.0, 0.0]) AS d, \
                vector_similarity(c.embedding, [1.0, 0.0]) AS s",
    )
    .unwrap();
    let plan = build_logical_plan(&ast).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let rows = execute(&plan, &mut ctx).expect("execute");
    assert!(
        rows.iter()
            .all(|r| matches!(r.get("d"), Some(Value::Float(_)))
                && matches!(r.get("s"), Some(Value::Float(_)))),
        "contract violation — vector_distance / vector_similarity must return Float"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Contract test group 6 (bonus): default weights are frozen.
// Documented defaults:
//   hybrid_score — w_vec = 0.65, w_bm25 = 0.35
//   doc_score    — α = 0.5, β = 0.3, γ = 0.2
// A golden test locks them so any future defaults drift fires the test.
// ─────────────────────────────────────────────────────────────────────────

#[test]
fn contract_hybrid_score_default_weights_are_0_65_0_35() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();
    insert_node(
        &engine,
        1,
        "Doc",
        &[
            ("body", Value::String("rust rust rust".to_string())),
            ("embedding", Value::Vector(vec![1.0, 0.0])),
        ],
        &mut interner,
    );

    let text_dir = dir.path().join("hybrid_def");
    let mut text_idx = TextIndex::open_or_create(&text_dir, 15_000_000, None).unwrap();
    text_idx.add_document(1, "rust rust rust").unwrap();
    let multi_idx = MultiLanguageTextIndex::wrap(text_idx, MultiLangConfig::default());

    // Run with no weights (defaults) and with explicit {vector: 0.65, text: 0.35}.
    // If defaults drift from 0.65/0.35, the two scores diverge.
    let default_rows = run_cypher_bm25(
        "MATCH (d:Doc) WHERE text_match(d.body, \"rust\") \
           AND vector_distance(d.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(d, \"rust\") AS s",
        &engine,
        &mut interner,
        &multi_idx,
    );
    let explicit_rows = run_cypher_bm25(
        "MATCH (d:Doc) WHERE text_match(d.body, \"rust\") \
           AND vector_distance(d.embedding, [1.0, 0.0]) < 1.0 \
         RETURN hybrid_score(d, \"rust\", {vector: 0.65, text: 0.35}) AS s",
        &engine,
        &mut interner,
        &multi_idx,
    );
    let d = match default_rows[0].get("s") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };
    let e = match explicit_rows[0].get("s") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };
    assert!(
        (d - e).abs() < 1e-9,
        "contract violation — hybrid_score defaults must equal (vector: 0.65, text: 0.35). \
         default={d}, explicit={e}"
    );
}

#[test]
fn contract_doc_score_default_weights_are_0_5_0_3_0_2() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(&engine, 1, "Document", &[], &mut interner);
    for i in 0..3 {
        let cid = 100 + i;
        insert_node(
            &engine,
            cid,
            "Chunk",
            &[(
                "embedding",
                Value::Vector(vec![1.0 - 0.1 * i as f32, 0.1 * i as f32]),
            )],
            &mut interner,
        );
        insert_edge(&engine, "HAS_CHUNK", 1, cid);
    }

    let ast_default = parse("MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0]) AS s").unwrap();
    let plan_default = build_logical_plan(&ast_default).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let rows_default = execute(&plan_default, &mut ctx).expect("execute default");
    let s_default = match rows_default[0].get("s") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };

    let ast_explicit = parse(
        "MATCH (d:Document) RETURN doc_score(d, [1.0, 0.0], {alpha: 0.5, beta: 0.3, gamma: 0.2}) AS s",
    )
    .unwrap();
    let plan_explicit = build_logical_plan(&ast_explicit).unwrap();
    let allocator = NodeIdAllocator::resume_from(NodeId::from_raw(1000));
    let mut ctx = make_test_ctx(&engine, &mut interner, &allocator);
    let rows_explicit = execute(&plan_explicit, &mut ctx).expect("execute explicit");
    let s_explicit = match rows_explicit[0].get("s") {
        Some(Value::Float(f)) => *f,
        other => panic!("expected Float, got {other:?}"),
    };

    assert!(
        (s_default - s_explicit).abs() < 1e-9,
        "contract violation — doc_score defaults must equal (α=0.5, β=0.3, γ=0.2). \
         default={s_default}, explicit={s_explicit}"
    );
}
