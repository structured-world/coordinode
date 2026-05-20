//! Property-based tests for the Layer-3 coordinator
//! ([`coordinode_storage::engine::coordinator::LocalMultiModalCoordinator`]).
//! Pins three contracts that the partition-keyed dispatch must
//! honour regardless of input shape.

#![allow(clippy::unwrap_used)]

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::coordinator::MultiModalCoordinator;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use proptest::prelude::*;
use tempfile::TempDir;

fn open_engine() -> (TempDir, StorageEngine) {
    let dir = TempDir::new().unwrap();
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "ep",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&cfg).unwrap();
    (dir, engine)
}

fn partition_strategy() -> impl Strategy<Value = Partition> {
    prop::sample::select(Partition::all().to_vec())
}

proptest! {
    // 32 cases halves wall time vs 64 with near-identical coverage on the
    // three round-trip invariants pinned here. Override via PROPTEST_CASES
    // env var when running a deep regression check.
    #![proptest_config(ProptestConfig { cases: 32, .. ProptestConfig::default() })]

    /// For any partition + any byte payload, put-then-get round-trips
    /// produce the same value through `engine.get` AND through
    /// `engine.coordinator().get`. Pins that the coordinator's read
    /// path matches the engine's read path byte-for-byte.
    #[test]
    fn put_then_get_round_trip(
        part in partition_strategy(),
        key in prop::collection::vec(any::<u8>(), 1..32),
        value in prop::collection::vec(any::<u8>(), 0..256),
    ) {
        // Skip Raft partition — it's reserved for openraft framework
        // bytes, not arbitrary user payloads.
        prop_assume!(part != Partition::Raft);
        let (_dir, engine) = open_engine();
        engine.put(part, &key, &value).unwrap();
        let via_engine = engine.get(part, &key).unwrap();
        let via_coord = engine.coordinator().get(part, &key).unwrap();
        prop_assert_eq!(
            via_engine.as_deref(),
            Some(value.as_slice()),
        );
        prop_assert_eq!(via_coord, via_engine);
    }

    /// snapshot() is monotonic across an arbitrary number of
    /// no-op observations (no intervening writes) — it returns the
    /// same horizon repeatedly.
    #[test]
    fn snapshot_is_stable_without_writes(probe_count in 1usize..32) {
        let (_dir, engine) = open_engine();
        let first = engine.snapshot();
        for _ in 0..probe_count {
            prop_assert_eq!(engine.snapshot(), first);
        }
    }

    /// has_write_after on a key written N times then probed at a
    /// horizon strictly before the first write always returns true
    /// (the puts are observable across any sequence of operations).
    #[test]
    fn has_write_after_detects_any_recorded_write(
        writes in 1usize..16,
    ) {
        let (_dir, engine) = open_engine();
        let before = engine.snapshot().saturating_sub(1);
        for i in 0..writes {
            engine
                .put(Partition::Node, &[i as u8], b"v")
                .unwrap();
        }
        // Probe any one of the keys we wrote.
        let observed = engine
            .has_write_after(Partition::Node, &[0u8], before)
            .unwrap();
        prop_assert!(observed);
    }
}
