use super::*;

const EPSILON: f32 = 1e-5;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// --- Dot Product ---

#[test]
fn dot_product_basic() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    // 1*4 + 2*5 + 3*6 = 32
    assert!(approx_eq(dot_product(&a, &b), 32.0));
}

#[test]
fn dot_product_zero_vector() {
    let a = [1.0, 2.0, 3.0];
    let z = [0.0, 0.0, 0.0];
    assert!(approx_eq(dot_product(&a, &z), 0.0));
}

#[test]
fn dot_product_orthogonal() {
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    assert!(approx_eq(dot_product(&a, &b), 0.0));
}

#[test]
fn dot_product_self() {
    let a = [3.0, 4.0];
    // 3^2 + 4^2 = 25
    assert!(approx_eq(dot_product(&a, &a), 25.0));
}

// --- L2 Distance ---

#[test]
fn l2_identical_vectors() {
    let a = [1.0, 2.0, 3.0];
    assert!(approx_eq(euclidean_distance(&a, &a), 0.0));
}

#[test]
fn l2_known_distance() {
    let a = [0.0, 0.0];
    let b = [3.0, 4.0];
    // sqrt(9 + 16) = 5
    assert!(approx_eq(euclidean_distance(&a, &b), 5.0));
}

#[test]
fn l2_symmetry() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];
    assert!(approx_eq(
        euclidean_distance(&a, &b),
        euclidean_distance(&b, &a)
    ));
}

// --- Cosine Similarity ---

#[test]
fn cosine_identical() {
    let a = [1.0, 2.0, 3.0];
    assert!(approx_eq(cosine_similarity(&a, &a), 1.0));
}

#[test]
fn cosine_opposite() {
    let a = [1.0, 0.0];
    let b = [-1.0, 0.0];
    assert!(approx_eq(cosine_similarity(&a, &b), -1.0));
}

#[test]
fn cosine_orthogonal() {
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];
    assert!(approx_eq(cosine_similarity(&a, &b), 0.0));
}

#[test]
fn cosine_zero_vector() {
    let a = [1.0, 2.0];
    let z = [0.0, 0.0];
    assert!(approx_eq(cosine_similarity(&a, &z), 0.0));
}

#[test]
fn cosine_scale_invariant() {
    let a = [1.0, 2.0, 3.0];
    let b = [2.0, 4.0, 6.0]; // 2x a
    assert!(approx_eq(cosine_similarity(&a, &b), 1.0));
}

/// Finite vectors whose dot and squared norms leave the f32 range score as
/// their direction says: `[1e20, 1e20]` against itself is 1, not the NaN
/// `inf / inf` gives, through every entry point.
#[test]
fn cosine_of_large_finite_vectors_is_their_direction() {
    let a = [1e20f32, 1e20];
    let b = [2e20f32, 2e20];
    let opposite = [-1e20f32, -1e20];
    let wide: Vec<f32> = vec![1e19; 1500]; // each square fits, the sum does not
    for (x, y, want) in [
        (&a[..], &a[..], 1.0),
        (&a[..], &b[..], 1.0),
        (&a[..], &opposite[..], -1.0),
        (&wide[..], &wide[..], 1.0),
        (
            &[f32::MAX, f32::MAX][..],
            &[f32::MAX, 0.0][..],
            core::f32::consts::FRAC_1_SQRT_2,
        ),
    ] {
        let got = cosine_similarity(x, y);
        assert!(approx_eq(got, want), "cosine {got}, want {want}");
        let got = cosine_similarity_with_query_norm(x, y, norm_l2(x));
        assert!(approx_eq(got, want), "with query norm {got}, want {want}");
        let got = cosine_similarity_with_both_norms(x, y, norm_l2(x), norm_l2(y));
        assert!(approx_eq(got, want), "with both norms {got}, want {want}");
    }
}

/// Cosine does not depend on scale: vectors far below 1 keep their angle
/// instead of being taken for zero vectors, down to subnormal components.
#[test]
fn cosine_is_scale_invariant_down_to_tiny_vectors() {
    let a = [1.0f32, 2.0, 3.0];
    let b = [3.0f32, -1.0, 2.0];
    let want = cosine_similarity(&a, &b);
    for scale in [1e-4f32, 1e-10, 1e-20, 1e-30, 1e-40] {
        let sa: Vec<f32> = a.iter().map(|x| x * scale).collect();
        let sb: Vec<f32> = b.iter().map(|x| x * scale).collect();
        let got = cosine_similarity(&sa, &sb);
        assert!(
            (got - want).abs() < 1e-3,
            "scale {scale}: cosine {got}, want {want}"
        );
    }
    // A zero vector still has no direction.
    assert_eq!(cosine_similarity(&[1e-30, 0.0], &[0.0, 0.0]), 0.0);
}

/// The norm of a finite vector is finite whenever it fits in f32, however
/// its squares overflow or underflow.
#[test]
fn norm_of_extreme_finite_vectors() {
    assert!((norm_l2(&[3e20, 4e20]) / 5e20 - 1.0).abs() < 1e-6);
    assert!((norm_l2(&[3e-25, 4e-25]) / 5e-25 - 1.0).abs() < 1e-6);
}

#[test]
fn cosine_distance_range() {
    let a = [1.0, 0.0];
    let b = [-1.0, 0.0];
    let d = cosine_distance(&a, &b);
    assert!((0.0..=2.0).contains(&d));
    assert!(approx_eq(d, 2.0)); // opposite vectors
}

// --- Manhattan Distance ---

#[test]
fn l1_identical() {
    let a = [1.0, 2.0, 3.0];
    assert!(approx_eq(manhattan_distance(&a, &a), 0.0));
}

#[test]
fn l1_known() {
    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 6.0, 8.0];
    // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
    assert!(approx_eq(manhattan_distance(&a, &b), 12.0));
}

#[test]
fn l1_symmetry() {
    let a = [1.0, 2.0];
    let b = [3.0, 5.0];
    assert!(approx_eq(
        manhattan_distance(&a, &b),
        manhattan_distance(&b, &a)
    ));
}

// --- Dispatcher ---

#[test]
fn distance_dispatcher() {
    let a = [1.0, 0.0];
    let b = [0.0, 1.0];

    let cos = distance(&a, &b, VectorMetric::Cosine);
    assert!(approx_eq(cos, 0.0)); // orthogonal

    let l2 = distance(&a, &b, VectorMetric::L2);
    assert!(approx_eq(l2, std::f32::consts::SQRT_2));

    let dot = distance(&a, &b, VectorMetric::DotProduct);
    assert!(approx_eq(dot, 0.0));

    let l1 = distance(&a, &b, VectorMetric::L1);
    assert!(approx_eq(l1, 2.0));
}

// --- High-dimensional ---

#[test]
fn high_dimensional_384() {
    let a: Vec<f32> = (0..384).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..384).map(|i| (i as f32).cos()).collect();

    let l2 = euclidean_distance(&a, &b);
    assert!(l2 > 0.0);
    assert!(l2.is_finite());

    let cos = cosine_similarity(&a, &b);
    assert!(cos.is_finite());
    assert!((-1.0..=1.0).contains(&cos));

    let l1 = manhattan_distance(&a, &b);
    assert!(l1 > 0.0);
    assert!(l1.is_finite());
}

#[test]
fn high_dimensional_768() {
    let a: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..768).map(|i| (i as f32 * 0.01).cos()).collect();

    let l2 = euclidean_distance(&a, &b);
    assert!(l2 > 0.0);
    assert!(l2.is_finite());
}

// --- Edge cases ---

#[test]
fn single_dimension() {
    let a = [3.0];
    let b = [7.0];
    assert!(approx_eq(euclidean_distance(&a, &b), 4.0));
    assert!(approx_eq(manhattan_distance(&a, &b), 4.0));
    assert!(approx_eq(cosine_similarity(&a, &b), 1.0)); // same direction
}

#[test]
fn negative_values() {
    let a = [-1.0, -2.0, -3.0];
    let b = [-4.0, -5.0, -6.0];
    let dot = dot_product(&a, &b);
    // (-1)(-4) + (-2)(-5) + (-3)(-6) = 4 + 10 + 18 = 32
    assert!(approx_eq(dot, 32.0));
}

#[test]
fn norm_l2_unit_vector() {
    let a = [0.6, 0.8]; // 0.36 + 0.64 = 1.0
    assert!(approx_eq(norm_l2(&a), 1.0));
}

// --- SIMD vs scalar consistency ---

#[test]
fn scalar_matches_for_non_simd_sizes() {
    // Sizes not divisible by SIMD width (8 for AVX2, 4 for NEON)
    let a: Vec<f32> = (0..13).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..13).map(|i| (i * 2) as f32).collect();

    let dot_result = dot_product(&a, &b);
    let scalar = dot_scalar(&a, &b);
    assert!(
        approx_eq(dot_result, scalar),
        "dot: {dot_result} vs scalar: {scalar}"
    );

    let l2_result = euclidean_distance_squared(&a, &b);
    let l2_s = l2_squared_scalar(&a, &b);
    assert!(
        approx_eq(l2_result, l2_s),
        "l2: {l2_result} vs scalar: {l2_s}"
    );

    let l1_result = manhattan_distance(&a, &b);
    let l1_s = l1_scalar(&a, &b);
    assert!(
        approx_eq(l1_result, l1_s),
        "l1: {l1_result} vs scalar: {l1_s}"
    );
}

#[test]
fn bound_kernels_match_scalar_across_dims() {
    // The bound function pointer must agree with the scalar reference
    // for SIMD-width multiples AND remainder tails. Relative tolerance:
    // SIMD multi-accumulator summation order legitimately differs from
    // sequential scalar order by a few ULPs at larger dims.
    fn rel_eq(x: f32, y: f32) -> bool {
        (x - y).abs() <= 1e-5 * x.abs().max(y.abs()).max(1.0)
    }
    for dim in [1usize, 7, 8, 16, 31, 100, 128, 1024] {
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.31 - 3.0).collect();
        let b: Vec<f32> = (0..dim).map(|i| (i as f32) * -0.17 + 1.5).collect();

        assert!(
            rel_eq(dot_product(&a, &b), dot_scalar(&a, &b)),
            "dot mismatch at dim {dim}"
        );
        assert!(
            rel_eq(
                euclidean_distance_squared(&a, &b),
                l2_squared_scalar(&a, &b)
            ),
            "l2 mismatch at dim {dim}"
        );
        assert!(
            rel_eq(manhattan_distance(&a, &b), l1_scalar(&a, &b)),
            "l1 mismatch at dim {dim}"
        );
    }
}
