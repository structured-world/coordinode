use super::*;

#[test]
fn counter_merge_no_base_single_delta() {
    let merger = CounterMerge;
    let delta = encode_counter_delta(5);
    let result = merger.merge(b"counter:test", None, &[&delta]).unwrap();
    assert_eq!(decode_counter(&result).unwrap(), 5);
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
    assert_eq!(decode_counter(&result).unwrap(), 114); // 100 + 10 - 3 + 7
}

#[test]
fn counter_merge_no_base_no_operands() {
    let merger = CounterMerge;
    let result = merger.merge(b"counter:empty", None, &[]).unwrap();
    assert_eq!(decode_counter(&result).unwrap(), 0);
}

#[test]
fn counter_merge_negative_result() {
    let merger = CounterMerge;
    let base = 5i64.to_le_bytes();
    let d1 = encode_counter_delta(-10);
    let result = merger.merge(b"counter:neg", Some(&base), &[&d1]).unwrap();
    assert_eq!(decode_counter(&result).unwrap(), -5);
}

/// A sum that leaves i64 is refused, not folded back into the range.
///
/// Wrapping would answer a counter at its maximum with a large negative
/// number and clamping with a maximum that has stopped counting, and a reader
/// can tell neither from a real value. The refusal it can tell.
#[test]
fn counter_merge_refuses_a_sum_outside_the_range() {
    let merger = CounterMerge;
    for (base, delta) in [(i64::MAX, 1i64), (i64::MIN, -1), (i64::MAX, i64::MAX)] {
        let base_bytes = base.to_le_bytes();
        let operand = encode_counter_delta(delta);
        assert!(
            merger
                .merge(b"counter:edge", Some(&base_bytes), &[&operand])
                .is_err(),
            "{base} + {delta} leaves i64 and must be refused"
        );
    }
    // The boundaries themselves are ordinary values to reach and to hold.
    let base = (i64::MAX - 1).to_le_bytes();
    let one = encode_counter_delta(1);
    let result = merger
        .merge(b"counter:edge", Some(&base), &[&one])
        .expect("reaching the maximum is not leaving the range");
    assert_eq!(decode_counter(&result).unwrap(), i64::MAX);
}

/// The overflow is refused wherever it happens in the fold, including
/// mid-chain, and a chain whose running total passes through the boundary
/// and comes back is still refused: the domain is i64 at every step, not
/// only at the end.
#[test]
fn counter_merge_refuses_an_overflow_inside_the_chain() {
    let merger = CounterMerge;
    let up = encode_counter_delta(i64::MAX);
    let down = encode_counter_delta(-1);
    assert!(
        merger
            .merge(b"counter:mid", None, &[&up, &up, &down])
            .is_err(),
        "a running total that leaves the range mid-chain must be refused"
    );
}

#[test]
fn counter_merge_multiple_compaction_rounds() {
    // Simulate: first compaction merges base+d1+d2, second merges result+d3.
    let merger = CounterMerge;
    let d1 = encode_counter_delta(10);
    let d2 = encode_counter_delta(20);
    let round1 = merger.merge(b"counter:multi", None, &[&d1, &d2]).unwrap();
    assert_eq!(decode_counter(&round1).unwrap(), 30);

    let d3 = encode_counter_delta(5);
    let round2 = merger
        .merge(b"counter:multi", Some(&round1), &[&d3])
        .unwrap();
    assert_eq!(decode_counter(&round2).unwrap(), 35);
}

#[test]
fn counter_encode_decode_roundtrip() {
    for val in [0i64, 1, -1, 42, -999, i64::MAX, i64::MIN] {
        let encoded = encode_counter_delta(val);
        assert_eq!(decode_counter(&encoded).unwrap(), val);
    }
}

/// Corrupt bytes are reported, never read as zero.
///
/// Zero is a value a counter legitimately holds, so decoding damage to it
/// would hide the damage behind a plausible answer: nothing downstream could
/// tell a counter that was reset from a byte range that was destroyed.
#[test]
fn counter_decode_refuses_anything_but_eight_bytes() {
    for bad in [&[][..], &[1, 2, 3][..], &[0; 7][..], &[0; 9][..]] {
        assert!(
            decode_counter(bad).is_err(),
            "{} bytes is not a counter",
            bad.len()
        );
    }
    assert_eq!(decode_counter(&0i64.to_le_bytes()).unwrap(), 0);
}

/// A corrupt operand or base fails the merge instead of being skipped.
#[test]
fn counter_merge_refuses_corrupt_input() {
    let merger = CounterMerge;
    let good = encode_counter_delta(1);

    assert!(
        merger.merge(b"counter:c", None, &[&good, &[9, 9]]).is_err(),
        "a corrupt operand must not be treated as a no-op"
    );
    assert!(
        merger.merge(b"counter:c", Some(&[9, 9]), &[&good]).is_err(),
        "a corrupt base must not be treated as a fresh empty base"
    );
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
        decode_counter(&value).unwrap(),
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
    let total = decode_counter(&value).unwrap();
    assert_eq!(
        total,
        (n_threads * increments_per_thread) as i64,
        "concurrent increments should sum correctly"
    );
}
