use super::*;

#[test]
fn counter_merge_no_base_single_delta() {
    let merger = CounterMerge;
    let delta = encode_counter_delta(5);
    let result = merger.merge(b"counter:test", None, &[&delta]).unwrap();
    assert_eq!(decode_counter(&result), 5);
}

#[test]
fn counter_merge_with_base_and_deltas() {
    let merger = CounterMerge;
    let base = 100i64.to_le_bytes();
    let d1 = encode_counter_delta(10);
    let d2 = encode_counter_delta(-3);
    let d3 = encode_counter_delta(7);
    let result = merger
        .merge(b"counter:x", Some(&base), &[&d1, &d2, &d3])
        .unwrap();
    assert_eq!(decode_counter(&result), 114); // 100 + 10 - 3 + 7
}

#[test]
fn counter_merge_no_base_no_operands() {
    let merger = CounterMerge;
    let result = merger.merge(b"counter:empty", None, &[]).unwrap();
    assert_eq!(decode_counter(&result), 0);
}

#[test]
fn counter_merge_negative_result() {
    let merger = CounterMerge;
    let base = 5i64.to_le_bytes();
    let d1 = encode_counter_delta(-10);
    let result = merger.merge(b"counter:neg", Some(&base), &[&d1]).unwrap();
    assert_eq!(decode_counter(&result), -5);
}

#[test]
fn counter_merge_wrapping_overflow() {
    let merger = CounterMerge;
    let base = i64::MAX.to_le_bytes();
    let d1 = encode_counter_delta(1);
    let result = merger.merge(b"counter:wrap", Some(&base), &[&d1]).unwrap();
    // wrapping_add: MAX + 1 = MIN
    assert_eq!(decode_counter(&result), i64::MIN);
}

#[test]
fn counter_merge_multiple_compaction_rounds() {
    // Simulate: first compaction merges base+d1+d2, second merges result+d3.
    let merger = CounterMerge;
    let d1 = encode_counter_delta(10);
    let d2 = encode_counter_delta(20);
    let round1 = merger.merge(b"counter:multi", None, &[&d1, &d2]).unwrap();
    assert_eq!(decode_counter(&round1), 30);

    let d3 = encode_counter_delta(5);
    let round2 = merger
        .merge(b"counter:multi", Some(&round1), &[&d3])
        .unwrap();
    assert_eq!(decode_counter(&round2), 35);
}

#[test]
fn counter_encode_decode_roundtrip() {
    for val in [0i64, 1, -1, 42, -999, i64::MAX, i64::MIN] {
        let encoded = encode_counter_delta(val);
        assert_eq!(decode_counter(&encoded), val);
    }
}

#[test]
fn counter_decode_short_data() {
    // Less than 8 bytes → 0
    assert_eq!(decode_counter(&[]), 0);
    assert_eq!(decode_counter(&[1, 2, 3]), 0);
}

#[test]
fn counter_merge_through_storage_engine() {
    // Integration: merge through real StorageEngine + Counter partition.
    use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
    use crate::engine::core::StorageEngine;
    use crate::engine::partition::Partition;

    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).unwrap();

    let key = b"counter:degree:42";

    // First merge: no base, delta +10
    engine
        .merge(Partition::Counter, key, &encode_counter_delta(10))
        .unwrap();
    // Second merge: delta +5
    engine
        .merge(Partition::Counter, key, &encode_counter_delta(5))
        .unwrap();
    // Third merge: delta -3
    engine
        .merge(Partition::Counter, key, &encode_counter_delta(-3))
        .unwrap();

    // Read back: should be 10 + 5 - 3 = 12
    let value = engine.get(Partition::Counter, key).unwrap().unwrap();
    assert_eq!(
        decode_counter(&value),
        12,
        "counter should be 12 after 3 merges"
    );
}

#[test]
fn counter_merge_concurrent_increments() {
    // Integration: concurrent merges from multiple threads.
    use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
    use crate::engine::core::StorageEngine;
    use crate::engine::partition::Partition;
    use std::sync::Arc;

    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&config).unwrap());

    let key = b"counter:concurrent";
    let n_threads = 4;
    let increments_per_thread = 100;

    let mut handles = Vec::new();
    for _ in 0..n_threads {
        let engine = Arc::clone(&engine);
        handles.push(std::thread::spawn(move || {
            for _ in 0..increments_per_thread {
                engine
                    .merge(Partition::Counter, key, &encode_counter_delta(1))
                    .unwrap();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    let value = engine.get(Partition::Counter, key).unwrap().unwrap();
    let total = decode_counter(&value);
    assert_eq!(
        total,
        (n_threads * increments_per_thread) as i64,
        "concurrent increments should sum correctly"
    );
}
