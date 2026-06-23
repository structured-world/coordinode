use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_core::graph::node::{encode_node_key, NodeId};

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

    // Insert 3 User nodes and 2 Post nodes
    for i in 0..3u64 {
        let key = encode_node_key(0, NodeId::from_raw(i));
        let rec = NodeRecord::new("User");
        engine
            .put(Partition::Node, &key, &rec.to_msgpack().unwrap())
            .unwrap();
    }
    for i in 3..5u64 {
        let key = encode_node_key(0, NodeId::from_raw(i));
        let rec = NodeRecord::new("Post");
        engine
            .put(Partition::Node, &key, &rec.to_msgpack().unwrap())
            .unwrap();
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

    // Node with labels [User, Admin]
    let key = encode_node_key(0, NodeId::from_raw(0));
    let rec = NodeRecord::with_labels(vec!["User".into(), "Admin".into()]);
    engine
        .put(Partition::Node, &key, &rec.to_msgpack().unwrap())
        .unwrap();

    // Node with label [User]
    let key2 = encode_node_key(0, NodeId::from_raw(1));
    let rec2 = NodeRecord::new("User");
    engine
        .put(Partition::Node, &key2, &rec2.to_msgpack().unwrap())
        .unwrap();

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
