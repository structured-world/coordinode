//! Whether the f32 truth vectors should stay in the shared block cache.
//!
//! A KV-separated tree no longer caches the values it reads from blob files
//! by default: large values are mostly read once and would evict the index
//! and filter blocks of every tree. The vector truth tier reads differently:
//! each search fetches the f32 originals of its candidates for the rerank,
//! and the candidates of different searches overlap on a hot set. This
//! measures that read (a `multi_get` of a search's candidates, drawn mostly
//! from a hot tenth of the vectors) with the values left out of the cache and
//! with them cached, after the same warm-up.
//!
//! Run: cargo bench -p coordinode-storage --bench blob_cache_fill

#![allow(clippy::expect_used, clippy::panic, clippy::print_stdout)]

use std::time::{Duration, Instant};

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use lsm_tree::config::BlobCacheFill;

/// Vectors stored.
const VECTORS: u64 = 50_000;
/// Dimensions of each: 3 KiB of f32, separated into blob files.
const DIMS: usize = 768;
/// Candidates a search reranks.
const CANDIDATES: usize = 40;
/// Searches measured per mode, after as many to warm up.
const SEARCHES: usize = 5_000;
/// Rounds of the two modes, alternating.
const ROUNDS: usize = 3;

fn key(id: u64) -> Vec<u8> {
    format!("vec:1:1:{id:012}").into_bytes()
}

/// A deterministic stream of pseudo-random u64s.
struct Draws(u64);

impl Draws {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A candidate id: nine in ten from the hot tenth of the vectors.
    fn candidate(&mut self) -> u64 {
        let draw = self.next();
        if draw.is_multiple_of(10) {
            draw % VECTORS
        } else {
            draw % (VECTORS / 10)
        }
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    let rank = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[rank]
}

fn searches(engine: &StorageEngine, draws: &mut Draws, n: usize) -> Vec<Duration> {
    (0..n)
        .map(|_| {
            let keys: Vec<Vec<u8>> = (0..CANDIDATES).map(|_| key(draws.candidate())).collect();
            let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
            let start = Instant::now();
            let values = engine
                .multi_get(Partition::VectorF32, &refs)
                .expect("multi_get");
            let took = start.elapsed();
            assert!(values.iter().all(Option::is_some));
            took
        })
        .collect()
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Ssd,
        Durability::Durable,
        Tier::Warm,
    )]);
    config.block_cache_bytes = 512 * 1024 * 1024;
    {
        let engine = StorageEngine::open(&config).expect("open");
        let mut draws = Draws(0x9e37_79b9_7f4a_7c15);
        for id in 0..VECTORS {
            let vector: Vec<u8> = (0..DIMS)
                .flat_map(|_| ((draws.next() % 2000) as f32 / 1000.0 - 1.0).to_le_bytes())
                .collect();
            engine
                .put(Partition::VectorF32, &key(id), &vector)
                .expect("put");
        }
        engine.persist().expect("flush");
        engine.major_compact(Partition::VectorF32).expect("compact");
    }

    // The modes alternate over several rounds, each on a freshly opened
    // engine: the order they run in and the values an earlier round left in
    // the block cache do not decide the comparison.
    let rounds = [BlobCacheFill::Never, BlobCacheFill::Always]
        .into_iter()
        .cycle()
        .take(2 * ROUNDS);
    for fill in rounds {
        let engine = StorageEngine::open(&config).expect("reopen");
        match engine.tree(Partition::VectorF32).expect("tree") {
            lsm_tree::AnyTree::Blob(tree) => tree
                .update_runtime_config(|runtime| runtime.blob_cache_fill = fill)
                .expect("set the fill"),
            lsm_tree::AnyTree::Standard(_) => panic!("the truth tier separates its values"),
        }
        let mut draws = Draws(0x2545_f491_4f6c_dd1d);
        searches(&engine, &mut draws, SEARCHES);
        let mut samples = searches(&engine, &mut draws, SEARCHES);
        samples.sort_unstable();
        let total: Duration = samples.iter().sum();
        println!(
            "{:<8} multi_get of {CANDIDATES}: mean={:>9.3?} p50={:>9.3?} p99={:>9.3?} p999={:>9.3?}",
            format!("{fill:?}"),
            total / samples.len() as u32,
            percentile(&samples, 0.50),
            percentile(&samples, 0.99),
            percentile(&samples, 0.999),
        );
    }
}
