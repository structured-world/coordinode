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

// ── EXPLAIN push_down JSON block emission (R-PUSH2) ────────────────

/// The block carries every contract field with the stable slug strings.
#[test]
fn explain_json_has_full_contract_shape() {
    let d = select_push_down_strategy(150, idx_default(1_000_000, 128, 200), 0.5, 10);
    let json = d.to_explain_json();
    for needle in [
        "\"stage\": \"VECTOR_FILTER\"",
        "\"strategy\":",
        "\"estimated_candidates\": 150",
        "\"estimated_selectivity_vs_index\":",
        "\"crossover_threshold\": 200",
        "\"reason\":",
        "\"cost_units_chosen\":",
        "\"cost_units_alternatives\":",
        // all three strategies considered, in the alternatives map
        "\"graph_first\":",
        "\"acorn_filtered\":",
        "\"vector_first\":",
    ] {
        assert!(
            json.contains(needle),
            "push_down JSON missing {needle}:\n{json}"
        );
    }
}

/// (1) 150 candidates below crossover → graph_first / candidate_set_below_crossover.
#[test]
fn explain_json_graph_first_below_crossover() {
    let json =
        select_push_down_strategy(150, idx_default(1_000_000, 128, 200), 0.5, 10).to_explain_json();
    assert!(json.contains("\"strategy\": \"graph_first\""));
    assert!(json.contains("\"reason\": \"candidate_set_below_crossover\""));
}

/// (2) 400K candidates, 0.005 vector selectivity → vector_first / vector_highly_selective.
#[test]
fn explain_json_vector_first_highly_selective() {
    let json = select_push_down_strategy(400_000, idx_default(1_000_000, 128, 200), 0.005, 10)
        .to_explain_json();
    assert!(json.contains("\"strategy\": \"vector_first\""));
    assert!(json.contains("\"reason\": \"vector_highly_selective\""));
}

/// (3) 100K candidates, 10% selectivity → acorn_filtered / selectivity_in_acorn_band.
#[test]
fn explain_json_acorn_filtered_in_band() {
    let json = select_push_down_strategy(100_000, idx_default(1_000_000, 128, 200), 0.5, 10)
        .to_explain_json();
    assert!(json.contains("\"strategy\": \"acorn_filtered\""));
    assert!(json.contains("\"reason\": \"selectivity_in_acorn_band\""));
}

/// (4) candidate set at crossover with low vector selectivity → fallback default.
#[test]
fn explain_json_fallback_default() {
    let json =
        select_push_down_strategy(200, idx_default(1_000_000, 128, 200), 0.5, 10).to_explain_json();
    assert!(json.contains("\"strategy\": \"acorn_filtered\""));
    assert!(json.contains("\"reason\": \"fallback_default\""));
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

// ── EXPLAIN push_down JSON contract: parse + version boundary (R-PUSH4) ──

/// The hand-rendered EXPLAIN block must be well-formed JSON whose every
/// frozen field is present with the contract-mandated JSON type and the
/// `strategy` / `reason` values are members of the frozen slug sets. This is
/// stronger than the `contains`-based shape tests above: it proves the block
/// parses as real JSON (a consumer can `JSON.parse` it) and pins each field's
/// type, not just its presence as a substring.
#[test]
fn explain_json_parses_with_frozen_field_types() {
    let d = select_push_down_strategy(150, idx_default(1_000_000, 128, 200), 0.5, 10);
    let v: serde_json::Value = serde_json::from_str(&d.to_explain_json())
        .expect("EXPLAIN push_down block must be valid JSON");
    let obj = v.as_object().expect("push_down block is a JSON object");

    assert_eq!(obj["stage"], "VECTOR_FILTER");
    assert!(
        ["graph_first", "acorn_filtered", "vector_first"]
            .contains(&obj["strategy"].as_str().expect("strategy is a string")),
        "strategy must be a frozen slug, got {:?}",
        obj["strategy"]
    );
    assert!(
        obj["estimated_candidates"].is_u64(),
        "estimated_candidates must be an integer"
    );
    assert!(
        obj["estimated_selectivity_vs_index"].is_number(),
        "estimated_selectivity_vs_index must be a number"
    );
    assert!(
        obj["crossover_threshold"].is_u64(),
        "crossover_threshold must be an integer"
    );
    assert!(
        [
            "candidate_set_below_crossover",
            "selectivity_in_acorn_band",
            "loose_graph_pattern",
            "vector_highly_selective",
            "fallback_default",
        ]
        .contains(&obj["reason"].as_str().expect("reason is a string")),
        "reason must be a frozen slug, got {:?}",
        obj["reason"]
    );
    assert!(
        obj["cost_units_chosen"].is_number(),
        "cost_units_chosen must be a number"
    );
    let alts = obj["cost_units_alternatives"]
        .as_object()
        .expect("cost_units_alternatives is an object");
    for k in ["graph_first", "acorn_filtered", "vector_first"] {
        assert!(
            alts[k].is_number(),
            "cost_units_alternatives.{k} must be a number"
        );
    }
}

/// Version-boundary forward compatibility: a future server may ADD fields to
/// the `push_down` block. A consumer pinned to the current contract must still
/// read every frozen field unchanged when an unknown field is present — adding
/// a field is non-breaking, renaming/removing one is breaking. This simulates
/// the boundary by injecting an unknown field into the emitted block and
/// re-reading the frozen fields.
#[test]
fn explain_json_tolerates_added_fields_across_version_boundary() {
    let d = select_push_down_strategy(150, idx_default(1_000_000, 128, 200), 0.5, 10);
    let mut v: serde_json::Value = serde_json::from_str(&d.to_explain_json()).expect("valid JSON");
    // A future minor version adds a field the current consumer does not know.
    v.as_object_mut()
        .expect("object")
        .insert("future_field_v2".to_string(), serde_json::json!({"x": 1}));
    let round_tripped = serde_json::to_string(&v).expect("re-serialize");
    let reparsed: serde_json::Value = serde_json::from_str(&round_tripped).expect("re-parse");

    // Every frozen field a current consumer relies on is still readable.
    assert_eq!(reparsed["stage"], "VECTOR_FILTER");
    assert_eq!(reparsed["strategy"], d.strategy.as_wire_str());
    assert_eq!(reparsed["reason"], d.reason.as_wire_str());
    assert_eq!(
        reparsed["estimated_candidates"].as_u64(),
        Some(d.estimated_candidates as u64)
    );
    assert_eq!(
        reparsed["crossover_threshold"].as_u64(),
        Some(d.crossover_threshold as u64)
    );
    assert!(reparsed["cost_units_alternatives"]["graph_first"].is_number());
}
