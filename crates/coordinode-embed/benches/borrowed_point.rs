//! Pair the former two-buffer copy path with typed borrowed inspection.
//! Both arms decode and validate the same retained snapshot; this does not
//! claim an end-to-end query speedup. Optional sampling repeats either arm.
#![allow(clippy::expect_used, clippy::print_stdout)]

use coordinode_core::graph::node::{NodeId, NodeRecord, encode_node_key};
use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_storage::engine::{core::StorageEngine, partition::Partition};
use std::time::{Duration, Instant};

#[inline(never)]
fn copied(engine: &StorageEngine, snapshot: u64, id: NodeId) -> NodeRecord {
    let bytes = engine
        .snapshot_get(&snapshot, Partition::Node, &encode_node_key(0, id))
        .expect("read")
        .expect("value")
        .to_vec();
    NodeRecord::from_msgpack(&bytes).expect("decode")
}

#[inline(never)]
fn borrowed(engine: &StorageEngine, snapshot: u64, id: NodeId) -> NodeRecord {
    LocalNodeStore
        .read_record_at_snapshot(engine, Some(snapshot), 0, id)
        .expect("read")
        .1
        .expect("record")
}

fn check(record: &NodeRecord, expected: &str) {
    assert_eq!(record.primary_label(), "PointProbe");
    assert_eq!(
        record.get_extra("payload").and_then(Value::as_str),
        Some(expected)
    );
}

fn report(label: &str, mut times: Vec<Duration>) {
    times.sort_unstable();
    println!(
        "{label}: n={} p50={:?} p99={:?}",
        times.len(),
        times[times.len() / 2],
        times[(times.len() - 1) * 99 / 100]
    );
}

fn main() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    let mut profile_input = None;
    for length in [32, 32768] {
        let mut record = NodeRecord::new("PointProbe");
        let expected = "x".repeat(length);
        record.set_extra("payload", Value::String(expected.clone()));
        // Explicit fixture identities: only used in this isolated benchmark.
        let id = NodeId::from_raw(length as u64);
        let engine = db.engine_shared();
        engine
            .put(
                Partition::Node,
                &encode_node_key(0, id),
                &record.to_msgpack().expect("encode"),
            )
            .expect("put");
        let (snapshot, _pin) = engine.pin_snapshot();
        check(&copied(&engine, snapshot, id), &expected);
        check(&borrowed(&engine, snapshot, id), &expected);
        let mut copy_times = Vec::with_capacity(1000);
        let mut borrow_times = Vec::with_capacity(1000);
        for round in 0..1000 {
            // Alternate ordering to avoid always measuring one arm after the other.
            for is_copy in if round % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let start = Instant::now();
                let got = if is_copy {
                    copied(&engine, snapshot, id)
                } else {
                    borrowed(&engine, snapshot, id)
                };
                let elapsed = start.elapsed();
                check(&got, &expected);
                if is_copy {
                    copy_times.push(elapsed);
                } else {
                    borrow_times.push(elapsed);
                }
            }
        }
        report(&format!("copied {length}B"), copy_times);
        report(&format!("borrowed {length}B"), borrow_times);
        profile_input = Some((id, expected));
    }
    if let Some(mode) = std::env::var_os("BORROWED_POINT_PROFILE") {
        let is_copy = mode == "copied";
        let engine = db.engine_shared();
        let (snapshot, _pin) = engine.pin_snapshot();
        let (id, expected) = profile_input.expect("fixture");
        println!("PROFILE point {mode:?}");
        let until = Instant::now() + Duration::from_secs(60);
        while Instant::now() < until {
            let got = if is_copy {
                copied(&engine, snapshot, id)
            } else {
                borrowed(&engine, snapshot, id)
            };
            check(&got, &expected);
            std::hint::black_box(got);
        }
    }
}
