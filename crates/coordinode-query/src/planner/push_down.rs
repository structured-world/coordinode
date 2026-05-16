//! Graph predicate push-down into vector scan (R-PUSH1).
//!
//! Implements the planner-level rule from `arch/core/query-engine.md`
//! § Graph Predicate Push-Down into Vector Scan: when a plan contains a
//! `TRAVERSE` stage producing a candidate set `C` followed by a
//! `VECTOR_FILTER` over property `p`, the planner picks one of three
//! strategies and annotates the `VectorFilter` operator with the decision.
//!
//! The selection is **deterministic given identical statistics** — not a
//! runtime heuristic. The invariant is contract-tested: no plan may place a
//! `TRAVERSE` directly before an unfiltered `VECTOR_FILTER`.
//!
//! Physical execution of each strategy lives elsewhere (HNSW scoped search,
//! ACORN traversal, brute-force) — this module only decides which strategy
//! to use.

use std::collections::HashMap;

/// The three push-down strategies the planner may pick.
///
/// Stable wire identity (part of EXPLAIN JSON contract, R-PUSH2/R-PUSH4):
/// new variants may be added in future versions, but existing variants are
/// never renamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PushDownStrategy {
    /// Traverse first, then brute-force vector distance on the candidate set.
    /// Optimal when `|C|` is below the HNSW crossover threshold (default 200
    /// for edges, 500 for nodes).
    GraphFirst,
    /// ACORN-style filtered HNSW: expanded `ef_search` with 2-hop tunneling.
    /// Optimal when `|C| / |V|` falls in the [5%, 30%] selectivity band.
    AcornFiltered,
    /// Unfiltered HNSW top-K, then graph predicate as post-filter.
    /// Optimal when vector selectivity is below 1% or `|C|` is very loose.
    VectorFirst,
}

impl PushDownStrategy {
    /// Stable wire string. Reserved enum value; never renamed.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::GraphFirst => "graph_first",
            Self::AcornFiltered => "acorn_filtered",
            Self::VectorFirst => "vector_first",
        }
    }
}

impl std::fmt::Display for PushDownStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

/// Stable reason slugs explaining why a strategy was picked.
///
/// Part of the EXPLAIN JSON contract (R-PUSH2/R-PUSH4): callers may rely on
/// these exact slug strings; new slugs may be added but existing slugs are
/// never renamed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PushDownReason {
    /// `|C|` below the per-index crossover threshold; brute-force on `C` is
    /// cheaper than building an HNSW search frontier.
    CandidateSetBelowCrossover,
    /// `|C| / |V|` falls in the [5%, 30%] band where ACORN-filtered HNSW
    /// dominates both brute-force and unfiltered HNSW.
    SelectivityInAcornBand,
    /// Graph pattern produces a loose (`|C| / |V| > 30%`) candidate set;
    /// HNSW top-K with cheap graph post-filter is faster than ACORN navigation.
    LooseGraphPattern,
    /// Vector predicate is highly selective (`vector_sel < 1%`); HNSW finds
    /// the answer faster than any graph-first approach.
    VectorHighlySelective,
    /// None of the explicit conditions matched; ACORN-filtered is the safe
    /// default (covers the middle of the parameter space).
    FallbackDefault,
}

impl PushDownReason {
    /// Stable wire string. Reserved enum value; never renamed.
    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::CandidateSetBelowCrossover => "candidate_set_below_crossover",
            Self::SelectivityInAcornBand => "selectivity_in_acorn_band",
            Self::LooseGraphPattern => "loose_graph_pattern",
            Self::VectorHighlySelective => "vector_highly_selective",
            Self::FallbackDefault => "fallback_default",
        }
    }
}

impl std::fmt::Display for PushDownReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_wire_str())
    }
}

/// Output of strategy selection: chosen strategy + reasoning + cost
/// comparison (for EXPLAIN / Query Advisor).
///
/// `PartialEq` is derived (cost values use `f64`, so the impl is partial in
/// the strict mathematical sense — `NaN != NaN`). Two decisions are
/// considered equal when every scalar field matches bit-for-bit and the
/// cost map is identical key-set + value-set.
#[derive(Debug, Clone, PartialEq)]
pub struct PushDownDecision {
    /// The strategy the planner picked.
    pub strategy: PushDownStrategy,
    /// Estimated `|C|` — the candidate set size produced by the upstream
    /// TRAVERSE stage.
    pub estimated_candidates: usize,
    /// Estimated `|C| / |V|` — the candidate set size as a fraction of the
    /// vector index size. Bounded to `[0.0, 1.0]`.
    pub estimated_selectivity: f64,
    /// The per-index crossover threshold (`|C|` below this triggers
    /// graph-first).
    pub crossover_threshold: usize,
    /// Why this strategy was picked.
    pub reason: PushDownReason,
    /// Cost of the chosen strategy in abstract cost units. Not milliseconds.
    pub cost_chosen: f64,
    /// Costs of all three strategies, for Query Advisor / EXPLAIN debugging.
    pub cost_alternatives: HashMap<PushDownStrategy, f64>,
}

/// Per-vector-index parameters used by the cost model.
///
/// These are properties of the HNSW index itself, cached on the
/// `SegmentDescriptor` / `IndexDefinition` after build, not recomputed per
/// query. R-PUSH1 cost model treats them as constants for the lifetime of
/// the index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorIndexParams {
    /// Number of vectors in the index (`|V|`).
    pub size: usize,
    /// Vector dimensionality.
    pub dim: u32,
    /// HNSW `M` parameter (max connections per node, ACORN uses `M × 2`).
    pub m: u32,
    /// HNSW `ef_search` parameter (default 200 per arch). Search frontier
    /// size; ACORN may expand it adaptively but the base is the index
    /// default.
    pub ef_search: usize,
    /// Per-index crossover threshold: `|C|` below this triggers graph-first.
    /// Per arch: default 200 for edges, 500 for nodes. Computed once at
    /// build time from `m`, `dim`, and quantisation; cached on the index.
    pub crossover_threshold: usize,
}

impl VectorIndexParams {
    /// Defaults for a node-typed HNSW index (no measured statistics
    /// available). The crossover is 500 per arch.
    pub fn default_node(size: usize, dim: u32) -> Self {
        Self {
            size,
            dim,
            m: 16,
            ef_search: 200,
            crossover_threshold: 500,
        }
    }

    /// Defaults for an edge-typed HNSW index. The crossover is 200 per arch.
    pub fn default_edge(size: usize, dim: u32) -> Self {
        Self {
            size,
            dim,
            m: 16,
            ef_search: 200,
            crossover_threshold: 200,
        }
    }
}

/// Cost-model "f32 ops" baseline — abstract unit, normalised to 1.0 for a
/// single 4-byte distance contribution. Concrete latency is not modelled —
/// the planner compares costs ordinally, not absolutely.
const F32_OPS: f64 = 1.0;

/// Cost-model "graph verify cost" baseline — abstract unit for verifying
/// that a node satisfies the graph pattern (lookup + compare). Order-of-
/// magnitude higher than a single distance op because it touches storage.
const GRAPH_VERIFY_COST: f64 = 32.0;

/// `cost = |C| × vector_dim × f32_ops`
///
/// Brute-force compute distance for every member of the candidate set `C`
/// against the query vector.
pub fn cost_graph_first(c: usize, vector_dim: u32) -> f64 {
    (c as f64) * (vector_dim as f64) * F32_OPS
}

/// `cost = ef_search × α(selectivity) × vector_dim × f32_ops`
///
/// ACORN-filtered HNSW: navigate the expanded HNSW graph (`M × 2` neighbours)
/// with adaptive `ef_search` scaled by α to compensate for filter rejections.
pub fn cost_acorn_filtered(ef_search: usize, alpha_sel: f64, vector_dim: u32) -> f64 {
    (ef_search as f64) * alpha_sel * (vector_dim as f64) * F32_OPS
}

/// `cost = log2(N) × vector_dim × f32_ops + K × graph_verify_cost`
///
/// Unfiltered HNSW top-K, then verify graph predicate per result.
pub fn cost_vector_first(n_total: usize, vector_dim: u32, k: usize) -> f64 {
    let n = n_total.max(2);
    (n as f64).log2() * (vector_dim as f64) * F32_OPS + (k as f64) * GRAPH_VERIFY_COST
}

/// α(selectivity) for ACORN cost model.
///
/// Per `arch/search/vector.md` § HNSW Filtering: α ≈ 1 at 30% selectivity,
/// ≈ 4 at 5% selectivity. Linear interpolation in the [5%, 30%] band.
///
/// Outside the band: clamped to the endpoint values (1.0 above 30%, 4.0
/// below 5%). This means cost outside the band stays defined but reflects
/// the cost ACORN would pay if it were used there — selection logic in
/// [`select_push_down_strategy`] ensures ACORN is only picked inside the
/// band; the cost is only computed outside the band for the
/// `cost_alternatives` comparison.
pub fn alpha_from_selectivity(selectivity: f64) -> f64 {
    if selectivity >= 0.30 {
        1.0
    } else if selectivity <= 0.05 {
        4.0
    } else {
        // Linear interp: at 0.30 → 1.0, at 0.05 → 4.0.
        // Slope: (4.0 - 1.0) / (0.05 - 0.30) = 3.0 / -0.25 = -12.0
        1.0 + (-12.0) * (selectivity - 0.30)
    }
}

/// The selectivity band where ACORN-filtered HNSW dominates.
const ACORN_BAND_LOW: f64 = 0.05;
const ACORN_BAND_HIGH: f64 = 0.30;

/// Threshold below which vector predicate selectivity is considered "highly
/// selective" — HNSW top-K + graph verify wins over any graph-first variant.
const VECTOR_HIGHLY_SELECTIVE_THRESHOLD: f64 = 0.01;

/// Implement the strategy-selection rule from
/// `arch/core/query-engine.md` § Graph Predicate Push-Down:
///
/// ```text
/// if |C| < crossover(p):                        -> Graph-first
/// elif |C| / |V| in [0.05, 0.30]:               -> ACORN-filtered HNSW
/// elif vector_sel < 0.01:                       -> Vector-first
/// elif |C| / |V| > 0.30:                        -> Vector-first
/// else:                                         -> ACORN-filtered (fallback)
/// ```
///
/// Arguments:
/// - `estimated_candidates`: `|C|` from the upstream TRAVERSE stage
/// - `index`: per-index parameters (size, dim, M, ef_search, crossover)
/// - `vector_selectivity`: fraction of vectors expected to pass the
///   distance threshold (estimated from threshold + distribution stats; if
///   unknown, callers should pass `1.0` to neutralise the
///   "highly_selective" branch)
/// - `top_k`: K for vector-first cost (typically the LIMIT or a default
///   like 100 if no LIMIT)
pub fn select_push_down_strategy(
    estimated_candidates: usize,
    index: VectorIndexParams,
    vector_selectivity: f64,
    top_k: usize,
) -> PushDownDecision {
    let sel_ratio = if index.size == 0 {
        0.0
    } else {
        (estimated_candidates as f64 / index.size as f64).clamp(0.0, 1.0)
    };
    let alpha = alpha_from_selectivity(sel_ratio);

    let cost_gf = cost_graph_first(estimated_candidates, index.dim);
    let cost_af = cost_acorn_filtered(index.ef_search, alpha, index.dim);
    let cost_vf = cost_vector_first(index.size, index.dim, top_k);

    let (strategy, reason) = if estimated_candidates < index.crossover_threshold {
        (
            PushDownStrategy::GraphFirst,
            PushDownReason::CandidateSetBelowCrossover,
        )
    } else if (ACORN_BAND_LOW..=ACORN_BAND_HIGH).contains(&sel_ratio) {
        (
            PushDownStrategy::AcornFiltered,
            PushDownReason::SelectivityInAcornBand,
        )
    } else if vector_selectivity < VECTOR_HIGHLY_SELECTIVE_THRESHOLD {
        (
            PushDownStrategy::VectorFirst,
            PushDownReason::VectorHighlySelective,
        )
    } else if sel_ratio > ACORN_BAND_HIGH {
        (
            PushDownStrategy::VectorFirst,
            PushDownReason::LooseGraphPattern,
        )
    } else {
        (
            PushDownStrategy::AcornFiltered,
            PushDownReason::FallbackDefault,
        )
    };

    let cost_chosen = match strategy {
        PushDownStrategy::GraphFirst => cost_gf,
        PushDownStrategy::AcornFiltered => cost_af,
        PushDownStrategy::VectorFirst => cost_vf,
    };

    let mut cost_alternatives = HashMap::with_capacity(3);
    cost_alternatives.insert(PushDownStrategy::GraphFirst, cost_gf);
    cost_alternatives.insert(PushDownStrategy::AcornFiltered, cost_af);
    cost_alternatives.insert(PushDownStrategy::VectorFirst, cost_vf);

    PushDownDecision {
        strategy,
        estimated_candidates,
        estimated_selectivity: sel_ratio,
        crossover_threshold: index.crossover_threshold,
        reason,
        cost_chosen,
        cost_alternatives,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn idx_default(size: usize, dim: u32, crossover: usize) -> VectorIndexParams {
        VectorIndexParams {
            size,
            dim,
            m: 16,
            ef_search: 200,
            crossover_threshold: crossover,
        }
    }

    // ── α(selectivity) ────────────────────────────────────────────────

    #[test]
    fn alpha_above_30_pct_is_one() {
        assert_eq!(alpha_from_selectivity(0.30), 1.0);
        assert_eq!(alpha_from_selectivity(0.50), 1.0);
        assert_eq!(alpha_from_selectivity(1.0), 1.0);
    }

    #[test]
    fn alpha_at_or_below_5_pct_is_four() {
        assert_eq!(alpha_from_selectivity(0.05), 4.0);
        assert_eq!(alpha_from_selectivity(0.01), 4.0);
        assert_eq!(alpha_from_selectivity(0.0), 4.0);
    }

    #[test]
    fn alpha_interpolates_inside_band() {
        // Midpoint of band: (0.30 + 0.05) / 2 = 0.175
        // Slope = -12, so α(0.175) = 1.0 + (-12) × (0.175 - 0.30) = 1.0 + 1.5 = 2.5
        let mid = alpha_from_selectivity(0.175);
        assert!(
            (mid - 2.5).abs() < 1e-9,
            "α at band midpoint should be 2.5, got {mid}"
        );
    }

    // ── cost functions ────────────────────────────────────────────────

    #[test]
    fn cost_graph_first_scales_linearly_with_c() {
        let dim = 128_u32;
        assert_eq!(cost_graph_first(0, dim), 0.0);
        assert_eq!(cost_graph_first(100, dim), 100.0 * 128.0);
        assert_eq!(cost_graph_first(1_000, dim), 1_000.0 * 128.0);
    }

    #[test]
    fn cost_vector_first_logarithmic_in_index_size() {
        let dim = 128_u32;
        let k = 10;
        let c_small = cost_vector_first(1_000, dim, k);
        let c_large = cost_vector_first(1_000_000, dim, k);
        // 1M is 1000× larger than 1K, but log2 grows ~10× only.
        // Vector-first must scale sub-linearly with index size.
        assert!(c_large < 20.0 * c_small);
    }

    #[test]
    fn cost_acorn_grows_at_low_selectivity() {
        let dim = 128_u32;
        let ef = 200;
        let high = cost_acorn_filtered(ef, alpha_from_selectivity(0.30), dim);
        let low = cost_acorn_filtered(ef, alpha_from_selectivity(0.05), dim);
        // At 5% selectivity α=4, at 30% α=1 → 4× more cost.
        assert!((low - 4.0 * high).abs() < 1e-9);
    }

    // ── strategy selection ────────────────────────────────────────────

    #[test]
    fn graph_first_when_candidate_set_below_crossover() {
        // |C| = 50, crossover = 200 → graph_first
        let d = select_push_down_strategy(50, idx_default(1_000_000, 128, 200), 0.5, 10);
        assert_eq!(d.strategy, PushDownStrategy::GraphFirst);
        assert_eq!(d.reason, PushDownReason::CandidateSetBelowCrossover);
        assert_eq!(d.estimated_candidates, 50);
        assert!(d
            .cost_alternatives
            .contains_key(&PushDownStrategy::GraphFirst));
        assert!(d
            .cost_alternatives
            .contains_key(&PushDownStrategy::AcornFiltered));
        assert!(d
            .cost_alternatives
            .contains_key(&PushDownStrategy::VectorFirst));
    }

    #[test]
    fn acorn_filtered_in_5_to_30_pct_band() {
        // |C| = 100K, |V| = 1M, selectivity = 0.10 → ACORN band [0.05, 0.30]
        let d = select_push_down_strategy(100_000, idx_default(1_000_000, 128, 200), 0.5, 10);
        assert_eq!(d.strategy, PushDownStrategy::AcornFiltered);
        assert_eq!(d.reason, PushDownReason::SelectivityInAcornBand);
        let sel = d.estimated_selectivity;
        assert!((sel - 0.10).abs() < 1e-9);
    }

    #[test]
    fn vector_first_when_vector_predicate_highly_selective() {
        // |C| / |V| = 0.40 (above ACORN band, would fall to loose_graph),
        // BUT vector_selectivity = 0.005 < 1% → highly selective wins.
        // Need |C| above ACORN_BAND_HIGH (0.30 of 1M = 300K) AND above crossover.
        // |C| = 400K, |V| = 1M → sel_ratio = 0.40 (outside band).
        // But order matters: first the band check, then vector_highly_selective.
        // sel_ratio 0.40 NOT in [0.05, 0.30] → falls through to vector_sel check.
        let d = select_push_down_strategy(400_000, idx_default(1_000_000, 128, 200), 0.005, 10);
        assert_eq!(d.strategy, PushDownStrategy::VectorFirst);
        assert_eq!(d.reason, PushDownReason::VectorHighlySelective);
    }

    #[test]
    fn vector_first_when_graph_pattern_loose() {
        // |C| / |V| = 0.50 (above ACORN band), vector_sel = 0.50 (not highly selective)
        // → loose_graph_pattern → vector-first
        let d = select_push_down_strategy(500_000, idx_default(1_000_000, 128, 200), 0.50, 10);
        assert_eq!(d.strategy, PushDownStrategy::VectorFirst);
        assert_eq!(d.reason, PushDownReason::LooseGraphPattern);
    }

    #[test]
    fn boundary_at_crossover_threshold_exact() {
        // |C| = 200 (exactly crossover) → not strictly less, so falls to next branch.
        // sel = 200 / 1M = 0.0002 → below ACORN band, vector_sel = 0.5 → fallback.
        // Per rule: |C| < crossover means strictly less. At equality, falls through.
        let d = select_push_down_strategy(200, idx_default(1_000_000, 128, 200), 0.5, 10);
        // sel_ratio = 0.0002 — below 5%, NOT in band → vector_sel not <1% → not loose
        // → fallback to AcornFiltered.
        assert_eq!(d.strategy, PushDownStrategy::AcornFiltered);
        assert_eq!(d.reason, PushDownReason::FallbackDefault);
    }

    #[test]
    fn empty_index_does_not_panic() {
        // |V| = 0 → selectivity defaults to 0, |C| any → graph-first (below
        // crossover) or fallback. Should not divide by zero.
        let d = select_push_down_strategy(50, idx_default(0, 128, 200), 0.5, 10);
        assert_eq!(d.strategy, PushDownStrategy::GraphFirst);
        assert_eq!(d.estimated_selectivity, 0.0);
    }

    #[test]
    fn cost_alternatives_includes_all_three_strategies() {
        let d = select_push_down_strategy(100, idx_default(1_000, 64, 200), 0.5, 10);
        assert_eq!(d.cost_alternatives.len(), 3);
        // All costs must be finite and non-negative.
        for &cost in d.cost_alternatives.values() {
            assert!(cost.is_finite());
            assert!(cost >= 0.0);
        }
    }

    // ── wire-stability of enum strings ────────────────────────────────

    #[test]
    fn strategy_wire_strings_are_stable() {
        assert_eq!(PushDownStrategy::GraphFirst.as_wire_str(), "graph_first");
        assert_eq!(
            PushDownStrategy::AcornFiltered.as_wire_str(),
            "acorn_filtered"
        );
        assert_eq!(PushDownStrategy::VectorFirst.as_wire_str(), "vector_first");
    }

    #[test]
    fn reason_wire_strings_are_stable() {
        assert_eq!(
            PushDownReason::CandidateSetBelowCrossover.as_wire_str(),
            "candidate_set_below_crossover"
        );
        assert_eq!(
            PushDownReason::SelectivityInAcornBand.as_wire_str(),
            "selectivity_in_acorn_band"
        );
        assert_eq!(
            PushDownReason::LooseGraphPattern.as_wire_str(),
            "loose_graph_pattern"
        );
        assert_eq!(
            PushDownReason::VectorHighlySelective.as_wire_str(),
            "vector_highly_selective"
        );
        assert_eq!(
            PushDownReason::FallbackDefault.as_wire_str(),
            "fallback_default"
        );
    }
}
