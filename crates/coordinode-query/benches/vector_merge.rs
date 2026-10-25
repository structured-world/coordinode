//! The cost of the distributed top-k merge: the same population searched
//! as one index and as partitions whose answers merge into one ranking (by
//! exact score, equal scores by id, a replicated id kept once). Latency
//! distributions (p50, p99, p999) for the single index, and for a query
//! routed to one partition and to every partition.
//!
//! The partitioning is the router's job and not measured here: a router
//! that sends every vector by the sign of its first two coordinates stands
//! in for the placement intelligence.

#![allow(clippy::expect_used, clippy::print_stdout)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use coordinode_cluster::{PartitionSet, VectorShardRouter};
use coordinode_core::budget::Unmetered;
use coordinode_core::graph::types::VectorMetric;
use coordinode_query::index::{
    GenerationId, IndexDescriptor, IndexId, VectorIndexConfig, VectorIndexRegistry,
};

const DIM: usize = 128;
const N: u64 = 20_000;
const K: usize = 10;
/// Timed searches per variant: enough for a p999.
const RUNS: usize = 2_000;

/// Components in `[-1, 1)`, deterministic per seed.
fn synth(seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..DIM)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            let u = (s >> 40) as f32 / (1u32 << 24) as f32;
            2.0 * u - 1.0
        })
        .collect()
}

/// Four partitions by the signs of the first two coordinates; a query goes
/// to its own partition or, `fan_out`, to all of them.
struct Quadrants {
    fan_out: bool,
}

impl Quadrants {
    fn quadrant(v: &[f32]) -> u32 {
        u32::from(v[0] >= 0.0) * 2 + u32::from(v[1] >= 0.0)
    }
}

impl VectorShardRouter for Quadrants {
    fn assign(&self, v: &[f32]) -> PartitionSet {
        let mut set = PartitionSet::new();
        set.push(Self::quadrant(v));
        set
    }

    fn route(&self, q: &[f32], _top_m: usize) -> PartitionSet {
        let mut set = PartitionSet::new();
        if self.fan_out {
            for p in 0..4 {
                set.push(p);
            }
        } else {
            set.push(Self::quadrant(q));
        }
        set
    }

    fn n_partitions(&self) -> usize {
        4
    }
}

fn definition() -> coordinode_query::index::IndexDefinition {
    IndexDescriptor::hnsw(
        "v_e",
        "V",
        "e",
        VectorIndexConfig {
            dimensions: DIM as u32,
            metric: VectorMetric::L2,
            m: 16,
            ef_construction: 200,
            quantization: coordinode_vector::hnsw::QuantizationCodec::None,
            offload_vectors: false,
            ef_search: None,
            rerank_candidates: None,
        },
    )
    .bind(IndexId::from_raw(1), GenerationId::from_raw(1))
}

fn report(label: &str, mut samples: Vec<Duration>) {
    samples.sort_unstable();
    let at = |q: usize| samples[(samples.len() - 1) * q / 1000];
    println!(
        "{label}: n={} p50={:?} p99={:?} p999={:?} max={:?}",
        samples.len(),
        at(500),
        at(990),
        at(999),
        samples[samples.len() - 1],
    );
}

fn measure(label: &str, registry: &VectorIndexRegistry, queries: &[Vec<f32>]) {
    let mut checksum = 0.0f64;
    let samples = (0..RUNS)
        .map(|i| {
            let start = Instant::now();
            let hits = registry
                .search("V", "e", &queries[i % queries.len()], K, &mut Unmetered)
                .expect("an evaluable query")
                .expect("the index");
            let took = start.elapsed();
            checksum += hits.iter().map(|h| h.score).sum::<f64>();
            took
        })
        .collect();
    report(label, samples);
    println!("{label}: checksum {checksum:.17e}");
}

fn main() {
    let population = || (0..N).map(|i| (i, synth(i + 10)));
    let queries: Vec<Vec<f32>> = (0..64).map(|q| synth(1_000_000 + q)).collect();

    let single = VectorIndexRegistry::new();
    single.register(definition());
    single.bulk_insert("V", "e", population()).expect("load");
    measure("single index", &single, &queries);

    for fan_out in [false, true] {
        let sharded = VectorIndexRegistry::new();
        sharded.register_sharded(definition(), Arc::new(Quadrants { fan_out }));
        sharded.bulk_insert("V", "e", population()).expect("load");
        measure(
            if fan_out {
                "4 partitions, every partition merged"
            } else {
                "4 partitions, one partition"
            },
            &sharded,
            &queries,
        );
    }
}
