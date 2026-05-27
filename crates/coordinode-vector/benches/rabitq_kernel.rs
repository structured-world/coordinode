//! Criterion benches for the RaBitQ popcount distance kernel.
//!
//! Validates the speed claim in [ADR-032](../../arch/DECISIONS.md):
//!   ~10× faster than SQ8 dequant+dot, ~50× faster than f32 AVX2 dot at D=1024.
//!
//! Each iteration scans all `N` database entries against a fixed query,
//! mirroring the inner loop of an HNSW search where the query is checked
//! against ~ef_search × M neighbours back-to-back. Encoding / calibration
//! happens once outside the timed region — only the per-distance work is
//! measured.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use coordinode_vector::metrics;
use coordinode_vector::quantize::popcount::{xor_popcount, xor_popcount_scalar};
use coordinode_vector::quantize::rabitq::{RaBitQCode, RaBitQParams};
use coordinode_vector::quantize::Sq8Params;

const D: usize = 1024;
const N: usize = 1000;

/// Deterministic pseudo-random vectors. Uses a self-contained xorshift64*
/// so the bench is reproducible without a `rand` dep.
fn make_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut s = seed.max(1);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u32 << 24) as f32 - 0.5
    };
    (0..n)
        .map(|_| {
            let raw: Vec<f32> = (0..dim).map(|_| next()).collect();
            // Normalise so cosine workloads see realistic vectors.
            let norm = metrics::norm_l2(&raw).max(f32::EPSILON);
            raw.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

fn bench_popcount_only(c: &mut Criterion) {
    let words = D / 64;
    let a: Vec<u64> = (0..words)
        .map(|i| 0xDEAD_BEEF_CAFE_F00Du64.wrapping_mul(i as u64 + 1))
        .collect();
    let b: Vec<u64> = (0..words)
        .map(|i| 0xA5A5_5A5A_3C3C_C3C3u64.wrapping_mul(i as u64 + 1))
        .collect();

    let mut group = c.benchmark_group("rabitq/popcount_bits1024");
    group.throughput(Throughput::Elements(1));

    group.bench_function("scalar", |bencher| {
        bencher.iter(|| black_box(xor_popcount_scalar(black_box(&a), black_box(&b))));
    });

    group.bench_function("dispatched", |bencher| {
        bencher.iter(|| black_box(xor_popcount(black_box(&a), black_box(&b))));
    });

    group.finish();
}

fn bench_full_scan(c: &mut Criterion) {
    let vectors = make_vectors(N + 1, D, 0xC0FFEE);
    let query = &vectors[0];
    let db: &[Vec<f32>] = &vectors[1..];

    // RaBitQ pre-encode (once).
    let params = RaBitQParams::calibrate(D as u32, 0xC0DE);
    let query_code = params.encode(query);
    let db_codes: Vec<RaBitQCode> = db.iter().map(|v| params.encode(v)).collect();

    // SQ8 pre-calibrate + quantize (once).
    let refs: Vec<&[f32]> = db.iter().map(|v| v.as_slice()).collect();
    let sq8 = Sq8Params::calibrate(&refs).expect("sq8 calibrate");
    let db_sq8: Vec<Vec<u8>> = db.iter().map(|v| sq8.quantize(v)).collect();
    let query_norm = metrics::norm_l2(query);

    let mut group = c.benchmark_group("rabitq/full_scan_1024d_1000docs");
    group.throughput(Throughput::Elements(N as u64));

    // RaBitQ — XOR + popcount only, no per-vector scalar ops on the hot path.
    group.bench_function(
        BenchmarkId::from_parameter("rabitq_cosine_dist"),
        |bencher| {
            bencher.iter(|| {
                let mut acc = 0.0f32;
                for code in &db_codes {
                    acc += params.estimate_cosine_distance(black_box(&query_code), black_box(code));
                }
                black_box(acc)
            });
        },
    );

    // SQ8 — dequant each vector then cosine. Realistic per-call cost.
    group.bench_function(BenchmarkId::from_parameter("sq8_cosine_dist"), |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for q in &db_sq8 {
                let dequantized = sq8.dequantize(black_box(q));
                acc += 1.0
                    - metrics::cosine_similarity_with_query_norm(
                        black_box(query),
                        &dequantized,
                        query_norm,
                    );
            }
            black_box(acc)
        });
    });

    // f32 cosine — the unoptimised reference path.
    group.bench_function(BenchmarkId::from_parameter("f32_cosine_dist"), |bencher| {
        bencher.iter(|| {
            let mut acc = 0.0f32;
            for v in db {
                acc += 1.0
                    - metrics::cosine_similarity_with_query_norm(
                        black_box(query),
                        black_box(v),
                        query_norm,
                    );
            }
            black_box(acc)
        });
    });

    group.finish();
}

criterion_group!(benches, bench_popcount_only, bench_full_scan);
criterion_main!(benches);
