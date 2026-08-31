//! Benchmark: transactional read-path overhead on top of the engine read.
//!
//! `Transaction::get` in MVCC mode is what every statement-level and
//! interactive-transaction read goes through: read-your-own-writes probe,
//! then the snapshot read. This measures that wrapper against a hot key,
//! so any bookkeeping added to the read path (conflict tracking, scope
//! maintenance) shows up here as the delta over the raw `engine.get` that
//! `mvcc_point_read` measures.
//!
//! Scenarios:
//!   1. txn_get/hot        — repeated point read of one committed key
//!   2. txn_get/miss       — repeated point read of an absent key
//!   3. txn_scan/paged_1k  — one keyset page over 1k committed rows

#![allow(clippy::expect_used)]

use std::sync::Arc;

use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use criterion::{Criterion, criterion_group, criterion_main};

fn setup() -> (StorageEngine, Arc<TimestampOracle>, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::resume_from(Timestamp::from_raw(1000)));
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Ssd,
        Durability::Durable,
        Tier::Hot,
    )]);
    let engine = StorageEngine::open_with_oracle(&config, oracle.clone()).expect("open");
    engine
        .put(Partition::Node, b"bench:hot", b"value-bytes-of-some-length")
        .expect("seed");
    for i in 0..1000u32 {
        let key = [b"bench:scan:".as_slice(), &i.to_be_bytes()].concat();
        engine.put(Partition::Node, &key, b"row").expect("seed row");
    }
    (engine, oracle, dir)
}

fn mvcc_txn<'a>(engine: &'a StorageEngine, oracle: &'a TimestampOracle) -> Transaction<'a> {
    let snap = engine.snapshot();
    Transaction::new(engine, Some(oracle), Timestamp::from_raw(snap), Some(snap))
}

fn bench_txn_reads(c: &mut Criterion) {
    let (engine, oracle, _dir) = setup();

    // One long-lived transaction per scenario, deliberately: the cost under
    // test is per-READ bookkeeping, and reusing the transaction is also the
    // shape an interactive transaction has (many reads, one handle). A
    // fresh-transaction-per-read variant would drown the read in setup.
    let mut txn = mvcc_txn(&engine, &oracle);
    c.bench_function("txn_get/hot", |b| {
        b.iter(|| {
            let v = txn
                .get(Partition::Node, std::hint::black_box(b"bench:hot"))
                .expect("get");
            std::hint::black_box(v);
        })
    });

    let mut txn = mvcc_txn(&engine, &oracle);
    c.bench_function("txn_get/miss", |b| {
        b.iter(|| {
            let v = txn
                .get(Partition::Node, std::hint::black_box(b"bench:absent"))
                .expect("get");
            std::hint::black_box(v);
        })
    });

    let mut txn = mvcc_txn(&engine, &oracle);
    c.bench_function("txn_scan/paged_1k", |b| {
        b.iter(|| {
            let page = txn
                .prefix_scan_paged(Partition::Node, b"bench:scan:", None, 1000)
                .expect("scan");
            std::hint::black_box(page.rows.len());
        })
    });
}

criterion_group!(benches, bench_txn_reads);
criterion_main!(benches);
