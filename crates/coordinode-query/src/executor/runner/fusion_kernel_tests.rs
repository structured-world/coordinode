use super::*;
use std::collections::BTreeMap;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn min_max_normalises_column_to_unit_range() {
    let col = vec![Some(1.0), Some(2.0), Some(4.0), None];
    let n = min_max_normalise(&col);
    assert_eq!(n.len(), 4);
    assert!(approx_eq(n[0].unwrap(), 0.0, 1e-9));
    // (2-1)/(4-1) = 1/3
    assert!(approx_eq(n[1].unwrap(), 1.0 / 3.0, 1e-9));
    assert!(approx_eq(n[2].unwrap(), 1.0, 1e-9));
    assert!(n[3].is_none());
}

#[test]
fn min_max_degenerate_range_returns_all_none() {
    let col = vec![Some(5.0), Some(5.0), None];
    let n = min_max_normalise(&col);
    assert!(n.iter().all(|c| c.is_none()));
}

#[test]
fn zscore_normalises_to_zero_mean_unit_stddev() {
    // For {1, 2, 3, 4}: mean = 2.5, σ = sqrt((1.25 + 0.25 + 0.25 + 1.25)/4) = sqrt(0.75 + 0.5) wait
    // Wait, do it: deviations [-1.5, -0.5, 0.5, 1.5], squared [2.25, 0.25, 0.25, 2.25], sum 5, /4 = 1.25, sqrt ≈ 1.1180339887
    let col = vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)];
    let n = zscore_normalise(&col);
    let sigma = (5.0_f64 / 4.0).sqrt();
    assert!(approx_eq(n[0].unwrap(), -1.5 / sigma, 1e-9));
    assert!(approx_eq(n[1].unwrap(), -0.5 / sigma, 1e-9));
    assert!(approx_eq(n[2].unwrap(), 0.5 / sigma, 1e-9));
    assert!(approx_eq(n[3].unwrap(), 1.5 / sigma, 1e-9));
}

#[test]
fn zscore_zero_sigma_returns_all_none() {
    let col = vec![Some(7.0), Some(7.0), Some(7.0)];
    let n = zscore_normalise(&col);
    assert!(n.iter().all(|c| c.is_none()));
}

#[test]
fn zscore_single_sample_returns_all_none() {
    let col = vec![Some(1.0), None, None];
    let n = zscore_normalise(&col);
    assert!(n.iter().all(|c| c.is_none()));
}

#[test]
fn fuse_raw_scores_convex_combination_weighted_sum() {
    // 3 rows, 2 methods (1 vector + 1 text).
    // Vector column: [0.0, 0.5, 1.0] (cosine similarities after the sign convention)
    // Text column  : [10.0, 5.0, 0.0] (BM25)
    // weights: vector 0.6, text 0.4.
    // Min-max:
    //   vector → [0, 0.5, 1.0]      (already unit-range)
    //   text   → [1.0, 0.5, 0.0]    ((10-0)/10, (5-0)/10, 0)
    // Fused = 0.6 * v + 0.4 * t.
    let raw = vec![
        vec![Some(0.0), Some(0.5), Some(1.0)],
        vec![Some(10.0), Some(5.0), Some(0.0)],
    ];
    let kinds = vec![
        RankFuseMethodKind::VectorBruteForce,
        RankFuseMethodKind::TextBm25 {
            label: "Doc".into(),
            property: "body".into(),
        },
    ];
    let mut weights: BTreeMap<String, f64> = BTreeMap::new();
    weights.insert("vector".into(), 0.6);
    weights.insert("text".into(), 0.4);

    let fused = fuse_raw_scores(&raw, &kinds, &weights, false);
    assert!(approx_eq(fused[0], 0.6 * 0.0 + 0.4 * 1.0, 1e-9));
    assert!(approx_eq(fused[1], 0.6 * 0.5 + 0.4 * 0.5, 1e-9));
    assert!(approx_eq(fused[2], 0.6 * 1.0 + 0.4 * 0.0, 1e-9));
}

#[test]
fn fuse_raw_scores_dbsf_weighted_zscore_sum() {
    // Single column with known z-scores; verify the weight scales correctly.
    let raw = vec![vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)]];
    let kinds = vec![RankFuseMethodKind::VectorBruteForce];
    let mut weights: BTreeMap<String, f64> = BTreeMap::new();
    weights.insert("vector".into(), 1.0);

    let fused = fuse_raw_scores(&raw, &kinds, &weights, true);
    let sigma = (5.0_f64 / 4.0).sqrt();
    assert!(approx_eq(fused[0], -1.5 / sigma, 1e-9));
    assert!(approx_eq(fused[3], 1.5 / sigma, 1e-9));
}

#[test]
fn fuse_raw_scores_missing_weight_drops_method() {
    let raw = vec![vec![Some(10.0), Some(0.0)]];
    let kinds = vec![RankFuseMethodKind::VectorBruteForce];
    // weights map is empty for "vector" — method drops out, fused stays 0.
    let weights: BTreeMap<String, f64> = BTreeMap::new();

    let fused = fuse_raw_scores(&raw, &kinds, &weights, false);
    assert_eq!(fused, vec![0.0, 0.0]);
}
