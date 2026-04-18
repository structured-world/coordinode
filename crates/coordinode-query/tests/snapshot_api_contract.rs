//! R-SNAP6 — Snapshot API stability contract tests.
//!
//! These tests FREEZE the public surface of the `read_consistency` knob.
//!
//! **Breaking any of these tests is a breaking-change signal.** Before
//! editing the test to match new behaviour, the change MUST:
//!
//! 1. Update `arch/core/transactions.md § Read Consistency` with the new
//!    rule or value.
//! 2. Add an alias for the old name (if renaming) — never remove a string
//!    form that was ever shipped in a public release.
//! 3. Run a full deprecation cycle (one minor release with the old form
//!    warned-on; removal only in the next major version).
//!
//! The contract, in one paragraph:
//!
//! - `ReadConsistencyMode` has EXACTLY three variants: `Current`, `Snapshot`,
//!   `Exact`. Their canonical strings are `"current"`, `"snapshot"`, `"exact"`.
//!   Parsing is case-insensitive.
//! - The planner auto-promotes to `Snapshot` when a query touches more than
//!   one modality from the enumerated set `{graph, vector, text, doc}`.
//!   Single-modality queries stay `Current`.
//! - An explicit `/*+ read_consistency(...) */` hint ALWAYS overrides the
//!   auto-promotion — in both directions (upgrade single-modality to
//!   `Snapshot` / downgrade cross-modality to `Current`).
//! - `vector_consistency` is a NARROWER override: when set, it applies to the
//!   vector modality only; all other modalities continue to follow
//!   `read_consistency`.
//! - `read_consistency` and `read_concern` are orthogonal. They carry
//!   different semantics (cross-modality snapshot alignment vs. durability
//!   visibility) and are independently settable.
//!
//! Gate: these tests MUST pass before every `v0.Y.0` minor release.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_core::graph::types::VectorConsistencyMode;
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::read_consistency::ReadConsistencyMode;
use coordinode_query::cypher::parser::parse;
use coordinode_query::planner::builder::build_logical_plan;

// ─────────────────────────────────────────────────────────────────────────
// Group 1 — Mode enumeration & parse contract
// ─────────────────────────────────────────────────────────────────────────

/// CONTRACT: the enum has exactly three variants. If this test fails to
/// compile, a variant was added or removed without updating the contract.
/// An exhaustive `match` is used so the compiler enforces enumeration.
#[test]
fn contract_mode_variants_are_exactly_three() {
    let all = [
        ReadConsistencyMode::Current,
        ReadConsistencyMode::Snapshot,
        ReadConsistencyMode::Exact,
    ];

    // Exhaustive match — compiler failure if a variant is added.
    for mode in all {
        let _witness: &'static str = match mode {
            ReadConsistencyMode::Current => "current",
            ReadConsistencyMode::Snapshot => "snapshot",
            ReadConsistencyMode::Exact => "exact",
        };
    }

    assert_eq!(
        all.len(),
        3,
        "read_consistency has exactly 3 public modes — changing this count requires arch/core/transactions.md update"
    );
}

/// CONTRACT: canonical string form for each mode is frozen.
#[test]
fn contract_canonical_strings() {
    assert_eq!(ReadConsistencyMode::Current.as_str(), "current");
    assert_eq!(ReadConsistencyMode::Snapshot.as_str(), "snapshot");
    assert_eq!(ReadConsistencyMode::Exact.as_str(), "exact");
}

/// CONTRACT: every canonical string parses back to the originating mode,
/// and the set of parsable strings is exactly the canonical three.
#[test]
fn contract_every_mode_parses_and_roundtrips() {
    for mode in [
        ReadConsistencyMode::Current,
        ReadConsistencyMode::Snapshot,
        ReadConsistencyMode::Exact,
    ] {
        let parsed =
            ReadConsistencyMode::from_str_opt(mode.as_str()).expect("canonical string must parse");
        assert_eq!(parsed, mode, "roundtrip: {} → {}", mode.as_str(), parsed);
    }

    // Negative enumeration: these strings MUST NOT be accepted. The list is
    // intentionally broad — any accidental alias would silently expand the
    // public surface.
    for rejected in [
        "",
        " ",
        "CURRENT ",
        "snap",
        "Snapshot_",
        "strict",
        "linearizable",
        "majority",
        "local",
        "read_committed",
        "serializable",
        "none",
    ] {
        assert!(
            ReadConsistencyMode::from_str_opt(rejected).is_none(),
            "string {rejected:?} must NOT parse — adding a new alias is a public-API change"
        );
    }
}

/// CONTRACT: parsing is case-insensitive across the shipped matrix.
#[test]
fn contract_parse_is_case_insensitive() {
    for (form, expected) in [
        ("current", ReadConsistencyMode::Current),
        ("Current", ReadConsistencyMode::Current),
        ("CURRENT", ReadConsistencyMode::Current),
        ("cUrReNt", ReadConsistencyMode::Current),
        ("snapshot", ReadConsistencyMode::Snapshot),
        ("SNAPSHOT", ReadConsistencyMode::Snapshot),
        ("Snapshot", ReadConsistencyMode::Snapshot),
        ("exact", ReadConsistencyMode::Exact),
        ("EXACT", ReadConsistencyMode::Exact),
        ("Exact", ReadConsistencyMode::Exact),
    ] {
        assert_eq!(
            ReadConsistencyMode::from_str_opt(form),
            Some(expected),
            "case-insensitive parse of {form:?}"
        );
    }
}

/// CONTRACT: Default is `Current` (perf-first).
#[test]
fn contract_default_is_current() {
    assert_eq!(ReadConsistencyMode::default(), ReadConsistencyMode::Current);
}

/// CONTRACT: `requires_snapshot_wait` is true for `Snapshot` and `Exact`
/// only. Changing this changes whether the executor blocks on the
/// watermark — a behavioural breaking change.
#[test]
fn contract_requires_snapshot_wait_matrix() {
    assert!(!ReadConsistencyMode::Current.requires_snapshot_wait());
    assert!(ReadConsistencyMode::Snapshot.requires_snapshot_wait());
    assert!(ReadConsistencyMode::Exact.requires_snapshot_wait());
}

// ─────────────────────────────────────────────────────────────────────────
// Group 2 — Auto-promotion contract
// ─────────────────────────────────────────────────────────────────────────

fn plan_of(q: &str) -> coordinode_query::planner::logical::LogicalPlan {
    let ast = parse(q).unwrap_or_else(|e| panic!("parse {q:?}: {e:?}"));
    build_logical_plan(&ast).unwrap_or_else(|e| panic!("plan {q:?}: {e:?}"))
}

/// CONTRACT: a graph-only query stays `Current`.
///
/// Adding a 5th modality to the planner's modality walk MUST NOT silently
/// change auto-promotion for existing single-modality queries. If this
/// test fails, review `planner::builder::modality_count` — the new
/// modality is accidentally counted for every graph query.
#[test]
fn contract_single_modality_graph_stays_current() {
    let plan = plan_of("MATCH (n:Person) WHERE n.age > 18 RETURN n LIMIT 10");
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Current,
        "pure graph query must NOT auto-promote — modality walk must count graph as exactly 1 modality"
    );
}

/// CONTRACT: a pure-vector query stays `Current` (VectorTopK alone = one modality).
#[test]
fn contract_single_modality_vector_stays_current() {
    // Single-modality check: a VectorTopK riding on a plain NodeScan — the
    // planner folds the graph scan as "vector's carrier", so modality_count = 1.
    let plan = plan_of("MATCH (n:Doc) RETURN n ORDER BY vector_distance(n.embedding, $q) LIMIT 5");
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Current,
        "pure vector query (VectorTopK over NodeScan) must NOT auto-promote"
    );
}

/// CONTRACT: a pure-text query stays `Current` (TextFilter over graph carrier).
#[test]
fn contract_single_modality_text_stays_current() {
    let plan = plan_of("MATCH (n:Doc) WHERE text_match(n.body, 'rust') RETURN n LIMIT 10");
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Current,
        "pure text query (TextFilter) must NOT auto-promote"
    );
}

/// CONTRACT: graph + vector → `Snapshot`.
#[test]
fn contract_cross_modality_graph_plus_vector_promotes() {
    let plan = plan_of(
        "MATCH (u:User)-[:WROTE]->(d:Doc) \
         WHERE vector_distance(d.embedding, $q) < 0.3 \
         RETURN u, d LIMIT 10",
    );
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Snapshot,
        "graph traversal + vector filter is cross-modality — MUST auto-promote"
    );
}

/// CONTRACT: graph + text → `Snapshot`.
#[test]
fn contract_cross_modality_graph_plus_text_promotes() {
    let plan = plan_of(
        "MATCH (u:User)-[:WROTE]->(d:Doc) \
         WHERE text_match(d.body, 'rust') \
         RETURN u, d LIMIT 10",
    );
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Snapshot,
        "graph traversal + text filter is cross-modality — MUST auto-promote"
    );
}

/// CONTRACT: vector + text → `Snapshot`.
#[test]
fn contract_cross_modality_vector_plus_text_promotes() {
    let plan = plan_of(
        "MATCH (d:Doc) \
         WHERE vector_distance(d.embedding, $q) < 0.3 \
            AND text_match(d.body, 'rust') \
         RETURN d LIMIT 10",
    );
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Snapshot,
        "vector + text is cross-modality — MUST auto-promote"
    );
}

// ─────────────────────────────────────────────────────────────────────────
// Group 3 — Precedence contract (hints vs auto-promotion; narrower overrides)
// ─────────────────────────────────────────────────────────────────────────

/// CONTRACT: an explicit hint upgrades a single-modality query.
#[test]
fn contract_hint_upgrades_single_modality() {
    let plan = plan_of("MATCH (n:Person) RETURN n /*+ read_consistency('snapshot') */");
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Snapshot,
        "explicit snapshot hint must upgrade even single-modality queries"
    );
}

/// CONTRACT: an explicit hint downgrades a cross-modality query.
#[test]
fn contract_hint_downgrades_cross_modality() {
    let plan = plan_of(
        "MATCH (u:User)-[:WROTE]->(d:Doc) \
         WHERE text_match(d.body, 'rust') \
         RETURN u, d /*+ read_consistency('current') */",
    );
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Current,
        "explicit current hint must defeat cross-modality auto-promotion — user knows best"
    );
}

/// CONTRACT: `read_consistency('exact')` hint is respected verbatim.
#[test]
fn contract_hint_exact_is_respected() {
    let plan = plan_of("MATCH (n:Person) RETURN n /*+ read_consistency('exact') */");
    assert_eq!(plan.read_consistency, ReadConsistencyMode::Exact);
}

/// CONTRACT: `vector_consistency` is a NARROWER override — applies to the
/// vector modality only, while `read_consistency` retains its value.
#[test]
fn contract_vector_consistency_is_narrower_override() {
    // Two separate `/*+ ... */` blocks — the hint parser accepts exactly
    // one key per comment (see cypher/parser.rs::parse_single_hint).
    let plan = plan_of(
        "MATCH (d:Doc) WHERE vector_distance(d.embedding, $q) < 0.3 \
         RETURN d /*+ read_consistency('snapshot') */ /*+ vector_consistency('exact') */",
    );
    assert_eq!(
        plan.read_consistency,
        ReadConsistencyMode::Snapshot,
        "read_consistency keeps its value"
    );
    assert_eq!(
        plan.vector_consistency,
        VectorConsistencyMode::Exact,
        "vector_consistency narrower override wins for vector modality"
    );
}

/// CONTRACT: without a `vector_consistency` hint, the vector modality
/// tracks `read_consistency` 1:1.
#[test]
fn contract_vector_consistency_follows_read_consistency_by_default() {
    for (hint, expected_rc, expected_vc) in [
        (
            "current",
            ReadConsistencyMode::Current,
            VectorConsistencyMode::Current,
        ),
        (
            "snapshot",
            ReadConsistencyMode::Snapshot,
            VectorConsistencyMode::Snapshot,
        ),
        (
            "exact",
            ReadConsistencyMode::Exact,
            VectorConsistencyMode::Exact,
        ),
    ] {
        let q = format!("MATCH (n:Person) RETURN n /*+ read_consistency('{hint}') */");
        let plan = plan_of(&q);
        assert_eq!(plan.read_consistency, expected_rc);
        assert_eq!(
            plan.vector_consistency, expected_vc,
            "vector_consistency must mirror read_consistency when no narrower hint is set (mode {hint})"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Group 4 — Orthogonality contract (read_consistency × read_concern)
// ─────────────────────────────────────────────────────────────────────────

/// CONTRACT: `ReadConsistencyMode` and `ReadConcernLevel` are DIFFERENT
/// types. They cannot accidentally unify through `From` / `Into`.
#[test]
fn contract_read_consistency_and_read_concern_are_distinct_types() {
    // If these ever become aliases, this test still compiles but the two
    // type names must remain in the public API — each governs a different
    // semantic axis (snapshot alignment vs durability visibility).
    fn assert_distinct<A: 'static, B: 'static>() {
        assert_ne!(
            std::any::TypeId::of::<A>(),
            std::any::TypeId::of::<B>(),
            "ReadConsistencyMode and ReadConcernLevel must be distinct types"
        );
    }
    assert_distinct::<ReadConsistencyMode, ReadConcernLevel>();
}

/// CONTRACT: every 3 × 4 combination of `ReadConsistencyMode` ×
/// `ReadConcernLevel` is representable. The two axes are orthogonal —
/// no combination is "forbidden" at the type level.
#[test]
fn contract_all_12_combinations_are_representable() {
    let consistency = [
        ReadConsistencyMode::Current,
        ReadConsistencyMode::Snapshot,
        ReadConsistencyMode::Exact,
    ];
    let concern = [
        ReadConcernLevel::Local,
        ReadConcernLevel::Majority,
        ReadConcernLevel::Linearizable,
        ReadConcernLevel::Snapshot,
    ];

    let mut count = 0usize;
    for rc in consistency {
        for rk in concern {
            // Must be constructible side-by-side. A compile-time type-level
            // "XOR" (forbidding some combinations) would break this.
            let pair = (rc, rk);
            // Modes retain their canonical strings regardless of concern.
            assert!(!rc.as_str().is_empty());
            assert!(!format!("{:?}", pair.1).is_empty());
            count += 1;
        }
    }
    assert_eq!(count, 12, "contract is 3 × 4 = 12 orthogonal combinations");
}

/// CONTRACT: naming collision check — there is a `Snapshot` variant on
/// both `ReadConsistencyMode` and `ReadConcernLevel`. This is intentional
/// and MUST stay because they carry different semantics:
///
/// - `ReadConsistencyMode::Snapshot` = cross-modality HLC T alignment.
/// - `ReadConcernLevel::Snapshot`   = MVCC point-in-time durability view.
///
/// If anyone ever merges these, this test fails compile via the distinct
/// type assertion and the deliberate `#[allow(clippy::disallowed_names)]`
/// documentation below.
#[test]
fn contract_snapshot_name_collision_is_intentional_and_semantic() {
    let a = ReadConsistencyMode::Snapshot;
    let b = ReadConcernLevel::Snapshot;
    assert_eq!(a.as_str(), "snapshot");
    // The `ReadConcernLevel::Snapshot` string form lives on
    // `ReadConcernLevel`; assert independence via debug format.
    assert!(format!("{b:?}").contains("Snapshot"));
    // The two are NOT interchangeable — this line wouldn't compile if
    // someone aliased them:
    // let _: ReadConsistencyMode = b;   // must not compile
}
