//! Isolated distance-kernel microbench — measures EXACTLY the inner
//! kernel cost, no HNSW graph / candidate management noise.
//!
//! Compares three SIMD strategies for L2-squared at d=128 (SIFT) and
//! d=1024 (text-embedding-large). Used to pick the right kernel
//! shape per architecture — multi-accumulator FMA wins at large D
//! (deep dep chain) but loses at small D (reduction overhead beats
//! the parallelism win).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use std::hint::black_box;

// Synthetic deterministic data — no allocation in the inner timing loop.
fn synth(seed: u64, dim: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (0..dim)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u32 << 24) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

// ──────────────────────────── L2-squared variants ──────────────────────

#[inline(always)]
fn l2_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// Variant A: production kernel (multi-acc 4× FMA).
fn l2_prod(a: &[f32], b: &[f32]) -> f32 {
    coordinode_vector::metrics::euclidean_distance_squared(a, b)
}

// Variant B: single-acc + FMA. Tight loop, low reduction overhead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(
    unused_unsafe,
    reason = "kept for forward-compat with edition 2024 unsafe_op_in_unsafe_fn"
)]
unsafe fn l2_single_fma_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;
        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let d = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(d, d, sum);
        }
        let mut r = hsum256(sum);
        for i in (chunks * 8)..a.len() {
            let d = a[i] - b[i];
            r += d * d;
        }
        r
    }
}

// Variant C: hnswlib-style — single-acc, separate mul+add (not FMA),
// unrolled by 2 inner blocks. Matches L2SqrSIMD16ExtAVX verbatim.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    unused_unsafe,
    reason = "kept for forward-compat with edition 2024 unsafe_op_in_unsafe_fn"
)]
unsafe fn l2_hnswlib_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let mut sum = _mm256_setzero_ps();
        // Process 16 dim per outer iter; tail handled below.
        let outer = a.len() / 16;
        for i in 0..outer {
            let base = i * 16;
            let v1 = _mm256_loadu_ps(a.as_ptr().add(base));
            let v2 = _mm256_loadu_ps(b.as_ptr().add(base));
            let d = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));

            let v1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
            let v2 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
            let d = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(d, d));
        }
        let mut r = hsum256(sum);
        for i in (outer * 16)..a.len() {
            let d = a[i] - b[i];
            r += d * d;
        }
        r
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(
    unused_unsafe,
    reason = "kept for forward-compat with edition 2024 unsafe_op_in_unsafe_fn"
)]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    unsafe {
        let hi = _mm256_extractf128_ps(v, 1);
        let lo = _mm256_castps256_ps128(v);
        let s = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(s);
        let s2 = _mm_add_ps(s, shuf);
        let shuf2 = _mm_movehl_ps(shuf, s2);
        let s3 = _mm_add_ss(s2, shuf2);
        _mm_cvtss_f32(s3)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(
    unused_unsafe,
    reason = "kept for forward-compat with edition 2024 unsafe_op_in_unsafe_fn"
)]
unsafe fn l2_single_fma_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;
        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let d = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, d, d);
        }
        let mut r = vaddvq_f32(sum);
        for i in (chunks * 4)..a.len() {
            let d = a[i] - b[i];
            r += d * d;
        }
        r
    }
}

// NEON hnswlib-equivalent: single accumulator, mul + add not FMA,
// unrolled by 4 (process 16 dim/iter to match SSE/AVX shape).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(
    unused_unsafe,
    reason = "kept for forward-compat with edition 2024 unsafe_op_in_unsafe_fn"
)]
unsafe fn l2_hnswlib_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    unsafe {
        let mut sum = vdupq_n_f32(0.0);
        let outer = a.len() / 16;
        for i in 0..outer {
            let base = i * 16;
            for off in [0, 4, 8, 12] {
                let v1 = vld1q_f32(a.as_ptr().add(base + off));
                let v2 = vld1q_f32(b.as_ptr().add(base + off));
                let d = vsubq_f32(v1, v2);
                sum = vaddq_f32(sum, vmulq_f32(d, d));
            }
        }
        let mut r = vaddvq_f32(sum);
        for i in (outer * 16)..a.len() {
            let d = a[i] - b[i];
            r += d * d;
        }
        r
    }
}

// ──────────────────────────── Bench harness ───────────────────────────

fn bench_l2(c: &mut Criterion) {
    for &dim in &[128usize, 1024usize] {
        let mut g = c.benchmark_group(format!("l2_squared_d{dim}"));
        let a: Vec<f32> = synth(1, dim);
        let b: Vec<f32> = synth(2, dim);

        // Force the compiler not to elide the call across iterations.
        g.bench_function(BenchmarkId::new("scalar", dim), |bch| {
            bch.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(l2_scalar(black_box(&a), black_box(&b))),
                BatchSize::SmallInput,
            )
        });
        g.bench_function(BenchmarkId::new("prod_multi_acc_fma", dim), |bch| {
            bch.iter_batched(
                || (a.clone(), b.clone()),
                |(a, b)| black_box(l2_prod(black_box(&a), black_box(&b))),
                BatchSize::SmallInput,
            )
        });
        #[cfg(target_arch = "x86_64")]
        {
            g.bench_function(BenchmarkId::new("single_fma_avx2", dim), |bch| {
                bch.iter_batched(
                    || (a.clone(), b.clone()),
                    |(a, b)| black_box(unsafe { l2_single_fma_avx2(black_box(&a), black_box(&b)) }),
                    BatchSize::SmallInput,
                )
            });
            g.bench_function(BenchmarkId::new("hnswlib_avx2", dim), |bch| {
                bch.iter_batched(
                    || (a.clone(), b.clone()),
                    |(a, b)| black_box(unsafe { l2_hnswlib_avx2(black_box(&a), black_box(&b)) }),
                    BatchSize::SmallInput,
                )
            });
        }
        #[cfg(target_arch = "aarch64")]
        {
            g.bench_function(BenchmarkId::new("single_fma_neon", dim), |bch| {
                bch.iter_batched(
                    || (a.clone(), b.clone()),
                    |(a, b)| black_box(unsafe { l2_single_fma_neon(black_box(&a), black_box(&b)) }),
                    BatchSize::SmallInput,
                )
            });
            g.bench_function(BenchmarkId::new("hnswlib_neon", dim), |bch| {
                bch.iter_batched(
                    || (a.clone(), b.clone()),
                    |(a, b)| black_box(unsafe { l2_hnswlib_neon(black_box(&a), black_box(&b)) }),
                    BatchSize::SmallInput,
                )
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench_l2);
criterion_main!(benches);
