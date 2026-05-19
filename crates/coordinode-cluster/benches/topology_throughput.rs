//! Throughput benchmarks for the CE single-node topology hot paths.
//! `placement_candidates` is invoked by Layer 2 on every SST write,
//! so its cost is on the data-write critical path. `shard_for_key`
//! and `shard_leader` are invoked by Layer 5 per query.

#![allow(clippy::unwrap_used)]

use coordinode_cluster::{
    ClusterTopology, CrushRule, FailureDomain, Modality, ShardId, ShardRouting, SingleNodeTopology,
    SingleShardRouting, TopologyTree,
};
use coordinode_storage::engine::config::Tier;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn topology_with_n(n: usize) -> SingleNodeTopology {
    let leaves: Vec<_> = (0..n)
        .map(|i| {
            let tier = match i % 3 {
                0 => Tier::Hot,
                1 => Tier::Warm,
                _ => Tier::Cold,
            };
            FailureDomain::local(format!("ep-{i}"), tier)
        })
        .collect();
    SingleNodeTopology::from_tree(TopologyTree { endpoints: leaves })
}

fn bench_placement_candidates(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_placement_candidates");
    group.sample_size(50);
    for &n in &[1usize, 10, 100, 1000] {
        let topo = topology_with_n(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                topo.placement_candidates(&CrushRule::local_tier(), Modality::Node, Tier::Warm)
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_shard_leader(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_shard_leader");
    group.sample_size(50);
    let topo = topology_with_n(10);
    group.bench_function("zero", |b| {
        b.iter(|| topo.shard_leader(ShardId::ZERO).unwrap())
    });
    group.finish();
}

fn bench_shard_routing(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_shard_routing");
    group.sample_size(50);
    let router = SingleShardRouting::new();
    let key = b"some-routing-key-32-bytes-long-x";
    group.bench_function("shard_for_key", |b| {
        b.iter(|| router.shard_for_key(key));
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_placement_candidates,
    bench_shard_leader,
    bench_shard_routing
);
criterion_main!(benches);
