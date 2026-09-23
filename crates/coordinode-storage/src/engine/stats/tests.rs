use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::error::StorageError;
use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::stats::{NODES_TOTAL_KEY, counter_delta_operand, label_count_key};

fn test_engine(dir: &std::path::Path) -> StorageEngine {
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    StorageEngine::open(&config).expect("open engine")
}

/// Seed the incremental statistics counters the way a committed node write
/// does (total +1, each label +1) — the reader consumes counters, not rows.
fn seed_node_counters(engine: &StorageEngine, labels: &[&str]) {
    engine
        .merge(
            Partition::Counter,
            NODES_TOTAL_KEY,
            &counter_delta_operand(1),
        )
        .unwrap();
    for label in labels {
        engine
            .merge(
                Partition::Counter,
                &label_count_key(label),
                &counter_delta_operand(1),
            )
            .unwrap();
    }
}

#[test]
fn empty_database_stats() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");

    assert_eq!(stats.total_node_count(), 0);
    assert_eq!(stats.label_count(), 0);
    assert_eq!(stats.avg_fan_out(), 0.0);
    assert_eq!(stats.node_count_for_label("User"), None);
    assert_eq!(stats.avg_fan_out_for_type("KNOWS"), None);
}

#[test]
fn node_count_per_label() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());

    // 3 User nodes and 2 Post nodes, as their committed writes would count.
    for _ in 0..3 {
        seed_node_counters(&engine, &["User"]);
    }
    for _ in 0..2 {
        seed_node_counters(&engine, &["Post"]);
    }

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");

    assert_eq!(stats.total_node_count(), 5);
    assert_eq!(stats.label_count(), 2);
    assert_eq!(stats.node_count_for_label("User"), Some(3));
    assert_eq!(stats.node_count_for_label("Post"), Some(2));
    assert_eq!(stats.node_count_for_label("Comment"), None);
}

#[test]
fn fan_out_sampling() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());

    // Create posting lists for KNOWS edges
    // Node 0 knows [1, 2, 3] (fan-out 3)
    // Node 1 knows [2] (fan-out 1)
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};

    let mut pl0 = PostingList::new();
    pl0.insert(1);
    pl0.insert(2);
    pl0.insert(3);
    let key0 = encode_adj_key_forward("KNOWS", NodeId::from_raw(0));
    engine
        .put(Partition::Adj, &key0, &pl0.to_bytes().unwrap())
        .unwrap();

    // Also store reverse keys (these should be skipped in fan-out calc)
    for &tgt in &[1u64, 2, 3] {
        let rev_key = encode_adj_key_reverse("KNOWS", NodeId::from_raw(tgt));
        let mut rev_pl = PostingList::new();
        rev_pl.insert(0);
        engine
            .put(Partition::Adj, &rev_key, &rev_pl.to_bytes().unwrap())
            .unwrap();
    }

    let mut pl1 = PostingList::new();
    pl1.insert(2);
    let key1 = encode_adj_key_forward("KNOWS", NodeId::from_raw(1));
    engine
        .put(Partition::Adj, &key1, &pl1.to_bytes().unwrap())
        .unwrap();

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");

    // avg fan-out for KNOWS: (3 + 1) / 2 = 2.0
    let fan_out = stats.avg_fan_out_for_type("KNOWS").unwrap();
    assert!((fan_out - 2.0).abs() < 0.01, "expected ~2.0, got {fan_out}");

    assert!((stats.avg_fan_out() - 2.0).abs() < 0.01);
}

#[test]
fn multi_label_nodes_counted_per_label() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());

    // One node with labels [User, Admin], one with [User]: a multi-label
    // node contributes one row to the total and one to EACH label count.
    seed_node_counters(&engine, &["User", "Admin"]);
    seed_node_counters(&engine, &["User"]);

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");

    assert_eq!(stats.total_node_count(), 2);
    assert_eq!(stats.node_count_for_label("User"), Some(2));
    assert_eq!(stats.node_count_for_label("Admin"), Some(1));
}

#[test]
fn multiple_edge_types() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());

    use coordinode_core::graph::edge::encode_adj_key_forward;

    // KNOWS: node 0 -> [1,2] (fan-out 2)
    let mut pl = PostingList::new();
    pl.insert(1);
    pl.insert(2);
    engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("KNOWS", NodeId::from_raw(0)),
            &pl.to_bytes().unwrap(),
        )
        .unwrap();

    // LIKES: node 0 -> [1,2,3,4] (fan-out 4)
    let mut pl2 = PostingList::new();
    pl2.insert(1);
    pl2.insert(2);
    pl2.insert(3);
    pl2.insert(4);
    engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("LIKES", NodeId::from_raw(0)),
            &pl2.to_bytes().unwrap(),
        )
        .unwrap();

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");

    assert!((stats.avg_fan_out_for_type("KNOWS").unwrap() - 2.0).abs() < 0.01);
    assert!((stats.avg_fan_out_for_type("LIKES").unwrap() - 4.0).abs() < 0.01);
    // Overall: (2 + 4) / 2 = 3.0
    assert!((stats.avg_fan_out() - 3.0).abs() < 0.01);
}

/// A counter driven below zero (double-decrement drift) is a well-formed value
/// the planner cannot use as a cardinality: the total clamps to zero and the
/// label reports absent.
#[test]
fn reader_clamps_underflowed_counters() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());

    // Drive "Gone" below zero and the total negative.
    engine
        .merge(
            Partition::Counter,
            &label_count_key("Gone"),
            &counter_delta_operand(-2),
        )
        .unwrap();
    engine
        .merge(
            Partition::Counter,
            NODES_TOTAL_KEY,
            &counter_delta_operand(-5),
        )
        .unwrap();

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");
    assert_eq!(stats.total_node_count(), 0, "negative total clamps to zero");
    assert_eq!(
        stats.node_count_for_label("Gone"),
        None,
        "an underflowed label reports absent"
    );
}

/// Assert the reader refuses the engine's state with a serialization error
/// that names `key`, instead of answering with a plausible number.
fn assert_refused(engine: &StorageEngine, key: &str) {
    let err = StorageStatsComputer::compute(engine)
        .err()
        .expect("a corrupt value must not be read as a statistic");
    assert!(
        matches!(&err, StorageError::Serialization(msg) if msg.contains(key)),
        "expected a serialization error naming {key}, got {err}"
    );
}

/// A label counter of the wrong width is corrupt. Reading it as zero would
/// make the planner price every plan over that label as if it were empty,
/// and nothing downstream could tell that from a label that really is.
#[test]
fn corrupt_label_counter_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    seed_node_counters(&engine, &["User"]);
    engine
        .put(Partition::Counter, &label_count_key("Junk"), b"not-an-i64")
        .unwrap();

    assert_refused(&engine, "Junk");
}

/// The total is read by a point get rather than the label walk; a corrupt
/// total is refused on that path too.
#[test]
fn corrupt_total_counter_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    engine
        .put(Partition::Counter, NODES_TOTAL_KEY, b"\x01\x02\x03")
        .unwrap();

    assert_refused(&engine, &String::from_utf8_lossy(NODES_TOTAL_KEY));
}

/// A node id is stored as eight big-endian bytes at the end of the adjacency
/// key, so most ids put a byte of 0x80 or above into it. Such a key is not
/// UTF-8, and a sampler that parses the key as text skips it: here that is
/// every forward list, and the fan-out would read as none at all.
#[test]
fn fan_out_counts_ids_that_are_not_text() {
    use coordinode_core::graph::edge::encode_adj_key_forward;

    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    // 200 = 0xC8 and 0x80 00 00 00 00 00 00 01: neither key is valid UTF-8.
    for (source, targets) in [
        (200u64, &[1u64, 2][..]),
        (0x8000_0000_0000_0001, &[3, 4, 5, 6][..]),
    ] {
        let mut pl = PostingList::new();
        for &t in targets {
            pl.insert(t);
        }
        engine
            .put(
                Partition::Adj,
                &encode_adj_key_forward("KNOWS", NodeId::from_raw(source)),
                &pl.to_bytes().unwrap(),
            )
            .unwrap();
    }

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");
    let knows = stats
        .avg_fan_out_for_type("KNOWS")
        .expect("both lists are sampled");
    assert!((knows - 3.0).abs() < 0.01, "(2 + 4) / 2, got {knows}");
    assert!((stats.avg_fan_out() - 3.0).abs() < 0.01);
}

/// The sample cap is per type. A single cap over the whole walk spends it on
/// the first types in key order and leaves the later ones with no estimate,
/// so the planner prices them with the overall average instead of their own.
#[test]
fn every_edge_type_is_sampled_past_the_cap() {
    use coordinode_core::graph::edge::{encode_adj_key_forward, encode_adj_key_reverse};

    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut one = PostingList::new();
    one.insert(1);
    let one = one.to_bytes().unwrap();
    // "A" sorts first and has more forward lists than the cap, plus reverse
    // lists the walk must skip rather than count.
    for id in 0..FAN_OUT_SAMPLE_PER_TYPE + 5 {
        engine
            .put(
                Partition::Adj,
                &encode_adj_key_forward("A", NodeId::from_raw(id)),
                &one,
            )
            .unwrap();
        engine
            .put(
                Partition::Adj,
                &encode_adj_key_reverse("A", NodeId::from_raw(id)),
                &one,
            )
            .unwrap();
    }
    let mut three = PostingList::new();
    for t in [1, 2, 3] {
        three.insert(t);
    }
    engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("B", NodeId::from_raw(1)),
            &three.to_bytes().unwrap(),
        )
        .unwrap();

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");
    assert!((stats.avg_fan_out_for_type("A").expect("A sampled") - 1.0).abs() < 0.01);
    assert!(
        (stats
            .avg_fan_out_for_type("B")
            .expect("B sampled after A's cap")
            - 3.0)
            .abs()
            < 0.01
    );
}

/// A forward adjacency value that is not a posting list is corrupt; skipping
/// it would silently bias the sample toward the lists that survived.
#[test]
fn corrupt_posting_list_is_refused() {
    use coordinode_core::graph::edge::encode_adj_key_forward;

    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    engine
        .put(
            Partition::Adj,
            &encode_adj_key_forward("KNOWS", NodeId::from_raw(7)),
            b"\xff\xfe not a posting list",
        )
        .unwrap();

    assert_refused(&engine, "KNOWS");
}

/// A key under the adjacency prefix that no adjacency encoder produces is
/// corrupt, and is refused rather than skipped.
#[test]
fn malformed_adjacency_key_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    let mut pl = PostingList::new();
    pl.insert(1);
    engine
        .put(
            Partition::Adj,
            b"adj:KNOWS:sideways:x",
            &pl.to_bytes().unwrap(),
        )
        .unwrap();

    assert_refused(&engine, "sideways");
}

/// Rebuilding the counters from the node rows must not write a total that
/// quietly leaves out a row it could not read: the counters would then
/// claim, durably, a graph smaller than the one on disk.
#[test]
fn rebuild_refuses_an_unreadable_node_row() {
    use coordinode_core::graph::node::encode_node_key;

    let dir = tempfile::tempdir().unwrap();
    let engine = test_engine(dir.path());
    engine
        .put(
            Partition::Node,
            &encode_node_key(1, NodeId::from_raw(9)),
            b"\xc1 not msgpack",
        )
        .unwrap();

    let result = rebuild_node_counters(&engine);
    assert!(
        matches!(result, Err(StorageError::Serialization(_))),
        "expected a serialization error, got {result:?}"
    );
    assert_eq!(
        engine.get(Partition::Counter, NODES_TOTAL_KEY).unwrap(),
        None,
        "a refused rebuild writes no counter"
    );
}
