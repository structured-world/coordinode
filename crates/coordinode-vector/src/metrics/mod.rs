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
        let avx512f = is_x86_feature_detected!("avx512f");
        let avx2 = is_x86_feature_detected!("avx2");
        let fma = is_x86_feature_detected!("fma");
        let sse4_2 = is_x86_feature_detected!("sse4.2");

        if avx512f {
            info!(
                arch = "x86_64",
                avx512f = true,
                avx2 = avx2,
                fma = fma,
                "SIMD: using AVX-512F multi-accumulator kernel (16-wide f32, 32 dim/iter)"
            );
        } else if avx2 && fma {
            info!(
                arch = "x86_64",
                avx2 = true,
                fma = true,
                "SIMD: using AVX2+FMA multi-accumulator kernel (8-wide f32, 32 dim/iter)"
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
        if std::arch::is_aarch64_feature_detected!("neon") {
            info!(
                arch = "aarch64",
                neon = true,
                "SIMD: using NEON multi-accumulator kernel (4-wide f32, 16 dim/iter)"
            );
        } else {
            info!(
                arch = "aarch64",
                neon = false,
                "SIMD: NEON not detected (unusual on aarch64), scalar fallback"
            );
        }
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

/// Cosine similarity where BOTH norms are already known.
///
/// `cosine_similarity_with_query_norm` still has to walk `b` twice — once
/// for `dot(a, b)`, once for `norm_l2(b)`. In HNSW search the second pass
/// is pure waste when the node vector's norm was precomputed at index
/// time (e.g. stored on `RaBitQCode.norm` for cosine workloads). This
/// entry point skips it and saves D add+mul ops per neighbour visit. For
/// glove M=16 ef=200 that's ~10% of the total search cycles measured via
/// `perf record`.
#[inline]
pub fn cosine_similarity_with_both_norms(
    a: &[f32],
    b: &[f32],
    a_norm_l2: f32,
    b_norm_l2: f32,
) -> f32 {
    let dot = dot_product_inner(a, b);
    let denom = a_norm_l2 * b_norm_l2;
    if denom < f32::EPSILON {
        return 0.0;
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
///
/// Kernel dispatch (runtime — same binary covers every CPU of the
/// target architecture):
///
/// | CPU feature                | Kernel           | f32/iter |
/// |----------------------------|------------------|----------|
/// | x86_64 AVX-512F (Zen4+, SPR) | `l2_squared_avx512` | 64 |
/// | x86_64 AVX2 + FMA          | `l2_squared_avx2_mt` | 32 |
/// | aarch64 NEON               | `l2_squared_neon_mt` | 16 |
/// | scalar fallback            | `l2_squared_scalar`  | 1  |
///
/// `#[inline]` is mandatory — the HNSW search hot loop calls this
/// thousands of times per query, and the CPUID branch must hoist OUT
/// of the loop (LLVM constant-folds `is_x86_feature_detected!` per
/// monomorphization site when inlined). Without inline the call
/// boundary plus per-call CPUID-cache load costs ~10 cycles each visit.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { l2_squared_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_squared_avx2_mt(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { l2_squared_neon_mt(a, b) };
        }
    }

    #[allow(unreachable_code)]
    l2_squared_scalar(a, b)
}

/// Dot product: sum(a_i * b_i). Higher = more similar (for normalized vectors).
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    dot_product_inner(a, b)
}

/// Manhattan (L1) distance: sum(|a_i - b_i|). Lower = more similar.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l1_avx2_mt(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { l1_neon_mt(a, b) };
        }
    }

    #[allow(unreachable_code)]
    l1_scalar(a, b)
}

/// L2 norm: sqrt(sum(a_i^2)).
#[inline]
pub fn norm_l2(a: &[f32]) -> f32 {
    dot_product_inner(a, a).sqrt()
}

// --- Scalar implementations ---

#[inline]
fn dot_product_inner(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { dot_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2_mt(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { dot_neon_mt(a, b) };
        }
    }

    #[allow(unreachable_code)]
    dot_scalar(a, b)
}

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
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

// --- SIMD: x86_64 AVX2 + FMA (single accumulator) ---
//
// Empirically validated kernel shape for Skylake / Coffee Lake on
// SIFT-128 (i9-9900K bench host):
//
// | Variant                              | M=16 | M=24 | M=32 |
// |--------------------------------------|------|------|------|
// | **single-acc FMA (this kernel)**     | 2251 | 1976 | 1808 |
// | multi-acc-4 FMA (commit 857216f)     | 2065 | 1807 | 1659 |
// | mul+add hnswlib-shape (commit 87f461) | 2009 | 1767 | 1622 |
//
// The single-acc FMA kernel wins because rustc + LLVM scheduling
// on this shape generates better µop dispatch than the multi-acc
// or hnswlib-port variants. Theoretical wider-ILP arguments don't
// hold at d=128: the inner loop is so short that reduction +
// horizontal-sum overhead beats any FMA-throughput gain.
//
// For the wider picture: profiling the bench on ro showed the
// distance kernel is ~2% of query time at the observed QPS — graph
// traversal (visited-set, neighbour iteration, candidate queue)
// dominates. Bigger QPS wins live there, not here.

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_mt(a: &[f32], b: &[f32]) -> f32 {
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
        for i in (chunks * 8)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_squared_avx2_mt(a: &[f32], b: &[f32]) -> f32 {
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
unsafe fn l1_avx2_mt(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        // Single accumulator — matches the AVX2 L2/dot shape (validated
        // empirical winner on Skylake; multi-acc regressed there).
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

// --- SIMD: x86_64 AVX-512F (16-wide, runtime-detected) ---
//
// AVX-512F is present on: Intel Sapphire Rapids+, Granite Rapids, Xeon
// Scalable 4th gen+; AMD Zen4+ (Ryzen 7000 / Epyc Genoa). NOT on the
// i9-9900K bench host — these paths are dead-code there at runtime,
// but compile + monomorphize once so any future hardware upgrade picks
// them up without rebuild. Two accumulators × 16 lanes = 32 f32/iter,
// matching the AVX2 path's chunk size so the tail logic stays uniform.

// AVX-512: hnswlib's L2SqrSIMD16ExtAVX512 also uses separate mul+add
// (the commented-out `_mm512_fmadd_ps` line in their source is an
// explicit choice). Single accumulator, 16 dim/iter — at 16-wide,
// the inner loop already covers d=128 in 8 iters with low chain
// pressure. Sapphire Rapids / Zen4 have wider FMA throughput but
// still benefit from the cleaner critical path of mul+add over FMA
// when D is small.

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm512_setzero_ps();
        let chunks = a.len() / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            sum = _mm512_fmadd_ps(va, vb, sum);
        }
        let mut result = _mm512_reduce_add_ps(sum);
        for i in (chunks * 16)..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn l2_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm512_setzero_ps();
        let chunks = a.len() / 16;
        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let d = _mm512_sub_ps(va, vb);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(d, d));
        }
        let mut result = _mm512_reduce_add_ps(sum);
        for i in (chunks * 16)..a.len() {
            let d = a[i] - b[i];
            result += d * d;
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

// --- SIMD: aarch64 NEON (multi-accumulator) ---
//
// NEON is mandatory in ARMv8 so the runtime probe always succeeds on
// real aarch64 hardware. We still gate behind `is_aarch64_feature_detected!`
// for symmetry with the x86_64 path — keeps every binary "runtime
// dispatch, not compile-time" per project policy.
//
// 4 independent accumulators × 4 lanes = 16 f32 per main iter. Apple
// M-series and AWS Graviton 3 / 4 have multiple FMA units; this layout
// keeps them all busy. On smaller cores (Cortex-A53) the extra parallel
// chains have no negative effect — they just retire as fast as they're
// issued.

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot_neon_mt(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        let main_chunks = a.len() / 16;
        for i in 0..main_chunks {
            let base = i * 16;
            let a0 = vld1q_f32(a.as_ptr().add(base));
            let b0 = vld1q_f32(b.as_ptr().add(base));
            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            let b1 = vld1q_f32(b.as_ptr().add(base + 4));
            let a2 = vld1q_f32(a.as_ptr().add(base + 8));
            let b2 = vld1q_f32(b.as_ptr().add(base + 8));
            let a3 = vld1q_f32(a.as_ptr().add(base + 12));
            let b3 = vld1q_f32(b.as_ptr().add(base + 12));

            s0 = vfmaq_f32(s0, a0, b0);
            s1 = vfmaq_f32(s1, a1, b1);
            s2 = vfmaq_f32(s2, a2, b2);
            s3 = vfmaq_f32(s3, a3, b3);
        }

        let mut tail_off = main_chunks * 16;
        while tail_off + 4 <= a.len() {
            let va = vld1q_f32(a.as_ptr().add(tail_off));
            let vb = vld1q_f32(b.as_ptr().add(tail_off));
            s0 = vfmaq_f32(s0, va, vb);
            tail_off += 4;
        }

        let sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut result = vaddvq_f32(sum);
        for i in tail_off..a.len() {
            result += a[i] * b[i];
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn l2_squared_neon_mt(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        let main_chunks = a.len() / 16;
        for i in 0..main_chunks {
            let base = i * 16;
            let a0 = vld1q_f32(a.as_ptr().add(base));
            let b0 = vld1q_f32(b.as_ptr().add(base));
            let a1 = vld1q_f32(a.as_ptr().add(base + 4));
            let b1 = vld1q_f32(b.as_ptr().add(base + 4));
            let a2 = vld1q_f32(a.as_ptr().add(base + 8));
            let b2 = vld1q_f32(b.as_ptr().add(base + 8));
            let a3 = vld1q_f32(a.as_ptr().add(base + 12));
            let b3 = vld1q_f32(b.as_ptr().add(base + 12));

            let d0 = vsubq_f32(a0, b0);
            let d1 = vsubq_f32(a1, b1);
            let d2 = vsubq_f32(a2, b2);
            let d3 = vsubq_f32(a3, b3);

            s0 = vfmaq_f32(s0, d0, d0);
            s1 = vfmaq_f32(s1, d1, d1);
            s2 = vfmaq_f32(s2, d2, d2);
            s3 = vfmaq_f32(s3, d3, d3);
        }

        let mut tail_off = main_chunks * 16;
        while tail_off + 4 <= a.len() {
            let va = vld1q_f32(a.as_ptr().add(tail_off));
            let vb = vld1q_f32(b.as_ptr().add(tail_off));
            let d = vsubq_f32(va, vb);
            s0 = vfmaq_f32(s0, d, d);
            tail_off += 4;
        }

        let sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut result = vaddvq_f32(sum);
        for i in tail_off..a.len() {
            let d = a[i] - b[i];
            result += d * d;
        }
        result
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_neon_mt(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let mut s2 = vdupq_n_f32(0.0);
        let mut s3 = vdupq_n_f32(0.0);

        let main_chunks = a.len() / 16;
        for i in 0..main_chunks {
            let base = i * 16;
            let d0 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(base)),
                vld1q_f32(b.as_ptr().add(base)),
            );
            let d1 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(base + 4)),
                vld1q_f32(b.as_ptr().add(base + 4)),
            );
            let d2 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(base + 8)),
                vld1q_f32(b.as_ptr().add(base + 8)),
            );
            let d3 = vsubq_f32(
                vld1q_f32(a.as_ptr().add(base + 12)),
                vld1q_f32(b.as_ptr().add(base + 12)),
            );

            s0 = vaddq_f32(s0, vabsq_f32(d0));
            s1 = vaddq_f32(s1, vabsq_f32(d1));
            s2 = vaddq_f32(s2, vabsq_f32(d2));
            s3 = vaddq_f32(s3, vabsq_f32(d3));
        }

        let mut tail_off = main_chunks * 16;
        while tail_off + 4 <= a.len() {
            let d = vsubq_f32(
                vld1q_f32(a.as_ptr().add(tail_off)),
                vld1q_f32(b.as_ptr().add(tail_off)),
            );
            s0 = vaddq_f32(s0, vabsq_f32(d));
            tail_off += 4;
        }

        let sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
        let mut result = vaddvq_f32(sum);
        for i in tail_off..a.len() {
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
