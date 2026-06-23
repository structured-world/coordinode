use super::*;

fn vec_eq(a: f32, b: f32, eps: f32) -> bool {
    (a - b).abs() <= eps
}

#[test]
fn empty_inputs_score_zero() {
    let q = vec![vec![1.0, 0.0]];
    let d = vec![vec![1.0, 0.0]];
    assert_eq!(maxsim(&[], &q), 0.0);
    assert_eq!(maxsim(&d, &[]), 0.0);
    assert_eq!(maxsim(&[], &[]), 0.0);
}

#[test]
fn mismatched_dim_scores_zero() {
    let q = vec![vec![1.0, 0.0]];
    let d_bad = vec![vec![1.0, 0.0, 0.0]];
    assert_eq!(maxsim(&d_bad, &q), 0.0);
    let q_mixed = vec![vec![1.0, 0.0], vec![1.0]];
    let d = vec![vec![1.0, 0.0]];
    assert_eq!(maxsim(&d, &q_mixed), 0.0);
}

#[test]
fn zero_dim_rows_score_zero() {
    let q: Vec<Vec<f32>> = vec![vec![]];
    let d: Vec<Vec<f32>> = vec![vec![]];
    assert_eq!(maxsim(&d, &q), 0.0);
}

#[test]
fn single_pair_equals_dot_product() {
    let q = vec![vec![1.0, 2.0, 3.0]];
    let d = vec![vec![4.0, 5.0, 6.0]];
    // Single q, single d: max = dot. Sum across 1 q-token = dot.
    let expected = 1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0;
    assert!(vec_eq(maxsim(&d, &q), expected, 1e-5));
}

#[test]
fn max_picks_best_doc_token_per_query() {
    // Hand-computed example.
    // q0 = [1, 0], q1 = [0, 1]
    // d0 = [1, 0], d1 = [0, 1], d2 = [0.5, 0.5]
    // q0 . d0 = 1, q0 . d1 = 0, q0 . d2 = 0.5 -> max = 1
    // q1 . d0 = 0, q1 . d1 = 1, q1 . d2 = 0.5 -> max = 1
    // total = 2
    let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let d = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
    assert!(vec_eq(maxsim(&d, &q), 2.0, 1e-5));
}

#[test]
fn per_query_token_decomposition_matches_sum() {
    let q = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let d = vec![
        vec![0.9, 0.1, 0.0],
        vec![0.1, 0.9, 0.0],
        vec![0.0, 0.1, 0.9],
    ];
    let per_q = maxsim_per_query_token(&d, &q);
    assert_eq!(per_q.len(), 3);
    let total = maxsim(&d, &q);
    let sum: f32 = per_q.iter().sum();
    assert!(vec_eq(total, sum, 1e-5));
}

#[test]
fn cosine_semantics_on_normalised_rows() {
    // Unit-norm rows; dot product equals cosine similarity.
    let q = vec![vec![1.0, 0.0]];
    let d = vec![vec![0.6, 0.8]]; // 0.6^2 + 0.8^2 = 1.0, unit norm
                                  // cos angle between q and d = 0.6
    assert!(vec_eq(maxsim(&d, &q), 0.6, 1e-5));
}

#[test]
fn larger_random_doc_score_finite_and_correct() {
    // Synthetic but deterministic input; verify the kernel's sum
    // matches a naive triple-loop reference implementation.
    let dim = 8;
    let q: Vec<Vec<f32>> = (0..4)
        .map(|i| (0..dim).map(|j| ((i + j) as f32) * 0.1).collect())
        .collect();
    let d: Vec<Vec<f32>> = (0..10)
        .map(|i| (0..dim).map(|j| ((i * 2 + j) as f32) * 0.05).collect())
        .collect();

    // Reference (naive triple loop) without going through dot_product.
    let mut reference = 0.0f32;
    for q_row in &q {
        let mut best = f32::NEG_INFINITY;
        for d_row in &d {
            let s: f32 = q_row.iter().zip(d_row.iter()).map(|(x, y)| x * y).sum();
            if s > best {
                best = s;
            }
        }
        reference += best;
    }

    let got = maxsim(&d, &q);
    assert!(
        vec_eq(got, reference, 1e-4),
        "maxsim={got} reference={reference}"
    );
    assert!(got.is_finite());
}
