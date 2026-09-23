//! Benchmark: canonical node-record encoding.
//!
//! Node records are encoded with their property keys ascending so equal
//! records have equal bytes. This measures what the sort costs on the write
//! path, against encoding the same maps in hash-table order.

#![allow(clippy::expect_used)]

use std::collections::HashMap;

use coordinode_core::graph::node::{NodeRecord, PropertyValue};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn record_with(props: u32) -> NodeRecord {
    let mut record = NodeRecord::new("User");
    for id in 0..props {
        record.set(id, PropertyValue::Int(i64::from(id) * 7));
    }
    record
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_record_encode");
    for props in [4u32, 16, 64] {
        let record = record_with(props);
        group.bench_with_input(BenchmarkId::new("canonical", props), &record, |b, r| {
            b.iter(|| r.to_msgpack().expect("encode"));
        });
        // The same fields serialised as they lie in the hash map: the encoding
        // before keys were sorted.
        let unsorted: (&Vec<String>, &HashMap<u32, PropertyValue>) =
            (&record.labels, &record.props);
        group.bench_with_input(BenchmarkId::new("hash_order", props), &unsorted, |b, r| {
            b.iter(|| rmp_serde::to_vec(r).expect("encode"));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode);
criterion_main!(benches);
