//! Distance metrics for vector similarity search.
//!
//! Four metrics supported: Cosine, Euclidean (L2), Dot Product, Manhattan (L1).
//! Scalar implementations with SIMD acceleration where available.
//!
//! Architecture note: SIMD is used for hot-path distance computation.
//! Always provide scalar fallback for portability.

use coordinode_core::graph::types::VectorMetric;
use tracing::info;

/// Detect and log available SIMD capabilities.
///
/// Call once at startup to log which instruction sets will be used
/// for vector distance computation.
pub fn log_simd_capabilities() {
    #[cfg(target_arch = "x86_64")]
    {
        let avx2 = is_x86_feature_detected!("avx2");
        let fma = is_x86_feature_detected!("fma");
        let sse4_2 = is_x86_feature_detected!("sse4.2");

        if avx2 && fma {
            info!(
                arch = "x86_64",
                avx2 = true,
                fma = true,
                "SIMD: using AVX2+FMA for vector distance (8-wide f32)"
            );
        } else if sse4_2 {
            info!(
                arch = "x86_64",
                avx2 = false,
                sse4_2 = true,
                "SIMD: AVX2 not available, using scalar fallback (SSE4.2 detected but not used)"
            );
        } else {
            info!(
                arch = "x86_64",
                avx2 = false,
                sse4_2 = false,
                "SIMD: no SIMD acceleration, using scalar fallback"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on aarch64
        info!(
            arch = "aarch64",
            neon = true,
            "SIMD: using NEON for vector distance (4-wide f32)"
        );
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        info!(
            arch = std::env::consts::ARCH,
            "SIMD: no SIMD support for this architecture, using scalar fallback"
        );
    }
}

/// Compute distance between two vectors using the specified metric.
///
/// Lower values = more similar for L2 and L1.
/// Higher values = more similar for Cosine and DotProduct.
pub fn distance(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    match metric {
        VectorMetric::Cosine => cosine_similarity(a, b),
        VectorMetric::L2 => euclidean_distance(a, b),
        VectorMetric::DotProduct => dot_product(a, b),
        VectorMetric::L1 => manhattan_distance(a, b),
    }
}

/// Cosine similarity: dot(a,b) / (||a|| * ||b||).
/// Returns value in [-1, 1]. Higher = more similar.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = norm_l2(a);
    cosine_similarity_with_query_norm(a, b, norm_a)
}

/// Cosine similarity where ‖a‖₂ is already known.
///
/// HNSW search calls the same distance function 100-1000+ times per query,
/// always with the same `a` (the query vector). Without this entry point
/// every call recomputes `dot(a, a).sqrt()` — that's a third of the FLOPs
/// for the metric. The search loop caches the query norm once and passes
/// it here; recall is bit-identical to [`cosine_similarity`].
#[inline]
pub fn cosine_similarity_with_query_norm(a: &[f32], b: &[f32], a_norm_l2: f32) -> f32 {
    let dot = dot_product_inner(a, b);
    let denom = a_norm_l2 * norm_l2(b);
    if denom < f32::EPSILON {
        return 0.0; // Zero vectors have no direction
    }
    dot / denom
}

/// Cosine distance: 1 - cosine_similarity. Range [0, 2]. Lower = more similar.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Euclidean (L2) distance: sqrt(sum((a_i - b_i)^2)).
/// Lower = more similar.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Squared Euclidean distance (avoids sqrt for comparison-only use).
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    // Runtime SIMD detection — binary works on ALL CPUs
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return unsafe { l2_squared_neon(a, b) };
    }

    #[allow(unreachable_code)]
    l2_squared_scalar(a, b)
}

/// Dot product: sum(a_i * b_i). Higher = more similar (for normalized vectors).
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    dot_product_inner(a, b)
}

/// Manhattan (L1) distance: sum(|a_i - b_i|). Lower = more similar.
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l1_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l1_neon(a, b) };
    }

    #[allow(unreachable_code)]
    l1_scalar(a, b)
}

/// L2 norm: sqrt(sum(a_i^2)).
pub fn norm_l2(a: &[f32]) -> f32 {
    dot_product_inner(a, a).sqrt()
}

// --- Scalar implementations ---

fn dot_product_inner(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_neon(a, b) };
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn l1_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

// --- SIMD: x86_64 AVX2 ---

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }
        let mut result = hsum256_ps(sum);
        // Handle remaining elements
        for i in (chunks * 8)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }
        let mut result = hsum256_ps(sum);
        for i in (chunks * 8)..a.len() {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l1_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let sign_mask = _mm256_set1_ps(-0.0f32);
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            sum = _mm256_add_ps(sum, abs_diff);
        }
        let mut result = hsum256_ps(sum);
        for i in (chunks * 8)..a.len() {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

// --- SIMD: aarch64 NEON ---

#[cfg(target_arch = "aarch64")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, va, vb);
        }
        let mut result = vaddvq_f32(sum);
        for i in (chunks * 4)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn l2_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }
        let mut result = vaddvq_f32(sum);
        for i in (chunks * 4)..a.len() {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn l1_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            let abs_diff = vabsq_f32(diff);
            sum = vaddq_f32(sum, abs_diff);
        }
        let mut result = vaddvq_f32(sum);
        for i in (chunks * 4)..a.len() {
            result += (a[i] - b[i]).abs();
        }
        result
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
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
}
