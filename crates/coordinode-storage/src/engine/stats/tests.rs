use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
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

/// Adversarial counter states must not poison the reader: a counter driven
/// below zero (double-decrement drift) clamps to absent/zero, and a
/// malformed counter value (wrong width) reads as zero instead of erroring.
#[test]
fn reader_clamps_underflow_and_ignores_malformed_counters() {
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
    // A malformed (non-8-byte) value under the label prefix.
    engine
        .put(Partition::Counter, &label_count_key("Junk"), b"not-an-i64")
        .unwrap();

    let stats = StorageStatsComputer::compute(&engine).expect("compute stats");
    assert_eq!(stats.total_node_count(), 0, "negative total clamps to zero");
    assert_eq!(
        stats.node_count_for_label("Gone"),
        None,
        "an underflowed label reports absent"
    );
    assert_eq!(
        stats.node_count_for_label("Junk"),
        None,
        "a malformed counter value reads as zero, not an error"
    );
}
