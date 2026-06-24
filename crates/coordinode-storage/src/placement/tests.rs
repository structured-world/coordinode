use super::*;
use crate::internal_test_helpers::memory_engine;

/// An empty partition produces an empty map (no spanning segment).
#[test]
fn empty_partition_yields_empty_map() {
    let (engine, _fs) = memory_engine();
    let map = SegmentMap::build(&engine, Partition::Node, SegmentId(1)).expect("build map");
    assert!(map.is_empty());
    assert_eq!(map.segments().len(), 0);
    assert_eq!(map.total_bytes(), 0);
    assert!(map.lookup(b"node:0:1").is_none());
}

/// A populated partition is covered by one spanning segment whose key range
/// starts at the minimum key and is unbounded above, with aggregated stats.
#[test]
fn populated_partition_one_spanning_segment() {
    let (engine, _fs) = memory_engine();
    for i in 0..50u32 {
        let key = format!("node:0:{i:08}");
        engine
            .put(Partition::Node, key.as_bytes(), b"value")
            .expect("put");
    }
    // Materialise SSTs so per-segment size/item stats are populated.
    engine
        .force_compaction(Partition::Node)
        .expect("force compaction");

    let map = SegmentMap::build(&engine, Partition::Node, SegmentId(7)).expect("build map");
    assert!(!map.is_empty());
    assert_eq!(map.segments().len(), 1);

    let seg = &map.segments()[0];
    assert_eq!(seg.id, SegmentId(7));
    assert_eq!(seg.partition, Partition::Node);
    assert_eq!(seg.data_type, SegmentDataType::PropertyDoc);
    // Whole-partition span: starts at the minimum key, unbounded above.
    assert_eq!(seg.key_range.start, b"node:0:00000000".to_vec());
    assert!(seg.key_range.end.is_empty());
    assert!(seg.item_count >= 50, "item_count was {}", seg.item_count);
    assert!(seg.size_bytes > 0);
    assert_eq!(map.total_bytes(), seg.size_bytes);
}

/// The spanning segment resolves every key at or above the minimum key.
#[test]
fn lookup_resolves_keys_in_span() {
    let (engine, _fs) = memory_engine();
    for i in 0..10u32 {
        let key = format!("node:0:{i:08}");
        engine
            .put(Partition::Node, key.as_bytes(), b"v")
            .expect("put");
    }
    engine
        .force_compaction(Partition::Node)
        .expect("force compaction");

    let map = SegmentMap::build(&engine, Partition::Node, SegmentId(1)).expect("build map");
    // A key inside the span resolves to the spanning segment.
    assert!(map.lookup(b"node:0:00000005").is_some());
    // A key beyond the minimum (unbounded above) still resolves.
    assert!(map.lookup(b"node:0:99999999").is_some());
    // A key below the minimum start does not.
    assert!(map.lookup(b"node:0:0000").is_none());
}

/// Data type is derived from the partition.
#[test]
fn data_type_from_partition() {
    assert_eq!(
        SegmentDataType::from_partition(Partition::Adj),
        SegmentDataType::PostingList
    );
    assert_eq!(
        SegmentDataType::from_partition(Partition::VectorF32),
        SegmentDataType::VectorIndex
    );
    assert_eq!(
        SegmentDataType::from_partition(Partition::Blob),
        SegmentDataType::Blob
    );
    assert_eq!(
        SegmentDataType::from_partition(Partition::Node),
        SegmentDataType::PropertyDoc
    );
    assert_eq!(
        SegmentDataType::from_partition(Partition::Idx),
        SegmentDataType::PropertyDoc
    );
}

/// Every partition's wire tag round-trips; unknown tags decode to `None`.
#[test]
fn partition_wire_tag_round_trips() {
    for &p in Partition::all() {
        let tag = partition_wire_tag(p);
        assert_eq!(partition_from_wire_tag(tag), Some(p));
    }
    assert_eq!(partition_from_wire_tag(250), None);
}

/// Half-open range containment, including the unbounded-above case.
#[test]
fn key_range_containment() {
    let bounded = KeyRange {
        start: b"m".to_vec(),
        end: b"t".to_vec(),
    };
    assert!(!bounded.contains(b"a"));
    assert!(bounded.contains(b"m")); // inclusive start
    assert!(bounded.contains(b"s"));
    assert!(!bounded.contains(b"t")); // exclusive end
    assert!(!bounded.contains(b"z"));

    let unbounded = KeyRange {
        start: b"m".to_vec(),
        end: Vec::new(),
    };
    assert!(!unbounded.contains(b"a"));
    assert!(unbounded.contains(b"m"));
    assert!(unbounded.contains(b"zzzz"));
}
