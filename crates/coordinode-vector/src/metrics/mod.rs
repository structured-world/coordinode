//! Distance metrics for vector similarity search.
//!
//! Four metrics supported: Cosine, Euclidean (L2), Dot Product, Manhattan (L1).
//! Scalar implementations with SIMD acceleration where available.
//!
//! Architecture note: SIMD is used for hot-path distance computation.
//! Always provide scalar fallback for portability.

use coordinode_core::graph::types::VectorMetric;
use tracing::info;

pub mod maxsim;
pub use maxsim::{maxsim, maxsim_per_query_token};

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
/// On x86_64 dispatch is a one-time bound function pointer (see
/// `KernelSlot`): one relaxed load + predicted indirect call, no
/// per-call feature detection. On aarch64 the NEON detect macro
/// constant-folds away and the kernel inlines directly.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: kernel resolved against this CPU's detected features.
        return unsafe { (L2_KERNEL.get(resolve_l2_kernel))(a, b) };
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
        // SAFETY: kernel resolved against this CPU's detected features.
        return unsafe { (L1_KERNEL.get(resolve_l1_kernel))(a, b) };
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

// --- Diagnostic distance-call counter (bench-only feature) ---

#[cfg(feature = "dist-counters")]
std::thread_local! {
    static DIST_CALLS: core::cell::Cell<u64> = const { core::cell::Cell::new(0) };
}

/// Read and reset this thread's distance-kernel invocation counter.
/// Only meaningful with the `dist-counters` feature; counts every
/// `dot_product_inner` entry (dot, cosine, norm all funnel through it).
#[cfg(feature = "dist-counters")]
pub fn take_dist_calls() -> u64 {
    DIST_CALLS.with(|c| c.replace(0))
}

#[inline]
fn count_dist_call() {
    #[cfg(feature = "dist-counters")]
    DIST_CALLS.with(|c| c.set(c.get() + 1));
}

// --- One-time kernel binding (x86_64 only) ---
//
// On x86_64 `is_x86_feature_detected!` compiles to a cached atomic
// load plus a branch, and the dispatchers below paid it up to three
// times per distance call (avx512f, avx2, fma) — measurable on a
// ~30ns kernel called thousands of times per HNSW query, while the
// `#[target_feature]` kernel cannot inline into the caller anyway.
// hnswlib avoids this by resolving the kernel function pointer once;
// same idea here, process-wide: first call resolves, every later call
// is one relaxed load + a perfectly-predicted indirect call. CPU
// features cannot change at runtime, so a benign Relaxed race is fine
// (worst case two threads resolve to the identical pointer).
//
// aarch64 deliberately keeps direct dispatch: NEON is in the target
// baseline, the detect macro constant-folds away and the kernel
// inlines into the caller — a bound pointer BLOCKS that inlining
// (measured +3-5% hnsw_search regression when tried).

#[cfg(target_arch = "x86_64")]
type SimKernel = unsafe fn(&[f32], &[f32]) -> f32;

#[cfg(target_arch = "x86_64")]
struct KernelSlot(core::sync::atomic::AtomicPtr<()>);

#[cfg(target_arch = "x86_64")]
impl KernelSlot {
    const fn new() -> Self {
        Self(core::sync::atomic::AtomicPtr::new(core::ptr::null_mut()))
    }

    #[inline]
    fn get(&self, resolve: fn() -> SimKernel) -> SimKernel {
        use core::sync::atomic::Ordering;
        let p = self.0.load(Ordering::Relaxed);
        if !p.is_null() {
            // SAFETY: the slot only ever stores pointers produced from a
            // `SimKernel` below; the transmute reverses that exact cast.
            return unsafe { core::mem::transmute::<*mut (), SimKernel>(p) };
        }
        let k = resolve();
        self.0.store(k as *mut (), Ordering::Relaxed);
        k
    }
}

#[cfg(target_arch = "x86_64")]
static DOT_KERNEL: KernelSlot = KernelSlot::new();
#[cfg(target_arch = "x86_64")]
static L2_KERNEL: KernelSlot = KernelSlot::new();
#[cfg(target_arch = "x86_64")]
static L1_KERNEL: KernelSlot = KernelSlot::new();

#[cfg(target_arch = "x86_64")]
fn resolve_dot_kernel() -> SimKernel {
    if is_x86_feature_detected!("avx512f") {
        return dot_avx512;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return dot_avx2_mt;
    }
    dot_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_l2_kernel() -> SimKernel {
    if is_x86_feature_detected!("avx512f") {
        return l2_squared_avx512;
    }
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return l2_squared_avx2_mt;
    }
    l2_squared_scalar
}

#[cfg(target_arch = "x86_64")]
fn resolve_l1_kernel() -> SimKernel {
    if is_x86_feature_detected!("avx2") {
        return l1_avx2_mt;
    }
    l1_scalar
}

// --- Scalar implementations ---

#[inline]
fn dot_product_inner(a: &[f32], b: &[f32]) -> f32 {
    count_dist_call();
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: the kernel was resolved against this CPU's detected
        // features; features do not change for the process lifetime.
        return unsafe { (DOT_KERNEL.get(resolve_dot_kernel))(a, b) };
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
mod tests;
