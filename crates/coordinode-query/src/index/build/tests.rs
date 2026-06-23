use super::*;
use coordinode_core::graph::node::NodeRecord;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

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

fn insert_node(
    engine: &StorageEngine,
    shard_id: u16,
    node_id: u64,
    label: &str,
    props: &[(&str, Value)],
    interner: &mut FieldInterner,
) {
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_modality::{LocalNodeStore, NodeStore as _};
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};
    let mut record = NodeRecord::new(label);
    for (name, value) in props {
        let field_id = interner.intern(name);
        record.set(field_id, value.clone());
    }
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    LocalNodeStore
        .put(&mut txn, shard_id, NodeId::from_raw(node_id), &record)
        .expect("put");
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit");
}

#[test]
fn build_index_on_existing_nodes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Create some nodes
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("email", Value::String("alice@test.com".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "User",
        &[("email", Value::String("bob@test.com".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        3,
        "Movie",
        &[("title", Value::String("Matrix".into()))],
        &mut interner,
    );

    // Build index on User.email
    let idx = IndexDefinition::btree("user_email", "User", "email");
    let result = build_index(&engine, &idx, &interner, 1);

    assert_eq!(result.state, Some(IndexBuildState::Committed));
    assert_eq!(result.scanned, 3); // All nodes scanned
    assert_eq!(result.indexed, 2); // Only User nodes indexed
    assert_eq!(result.skipped, 1); // Movie skipped
    assert!(result.violations.is_empty());

    // Verify index entries exist
    let alice = super::super::ops::index_scan_exact(
        &engine,
        "user_email",
        &Value::String("alice@test.com".into()),
    )
    .expect("scan");
    assert_eq!(alice, vec![1]);
}

#[test]
fn build_unique_index_aborts_on_duplicates() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    // Two nodes with same email
    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("email", Value::String("same@test.com".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "User",
        &[("email", Value::String("same@test.com".into()))],
        &mut interner,
    );

    let idx = IndexDefinition::btree("user_email", "User", "email").unique();
    let result = build_index(&engine, &idx, &interner, 1);

    assert_eq!(result.state, Some(IndexBuildState::Aborted));
    assert!(!result.violations.is_empty());

    // Index entries should be cleaned up
    let entries = super::super::ops::index_scan(&engine, &idx).expect("scan");
    assert!(entries.is_empty());
}

#[test]
fn build_sparse_index_skips_nulls() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("bio", Value::String("Developer".into()))],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "User",
        &[("name", Value::String("Bob".into()))],
        &mut interner,
    ); // No bio

    let idx = IndexDefinition::btree("user_bio", "User", "bio").sparse();
    let result = build_index(&engine, &idx, &interner, 1);

    assert_eq!(result.state, Some(IndexBuildState::Committed));
    assert_eq!(result.indexed, 1); // Only node 1 has bio
}

#[test]
fn build_index_saves_definition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[("name", Value::String("Alice".into()))],
        &mut interner,
    );

    let idx = IndexDefinition::btree("user_name", "User", "name");
    let result = build_index(&engine, &idx, &interner, 1);
    assert_eq!(result.state, Some(IndexBuildState::Committed));

    // Definition should be loadable
    let loaded = super::super::ops::load_index_definition(&engine, "user_name")
        .expect("load")
        .expect("should exist");
    assert_eq!(loaded.name, "user_name");
}

#[test]
fn build_empty_label_succeeds() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let interner = FieldInterner::new();

    // No nodes exist
    let idx = IndexDefinition::btree("user_email", "User", "email");
    let result = build_index(&engine, &idx, &interner, 1);

    assert_eq!(result.state, Some(IndexBuildState::Committed));
    assert_eq!(result.scanned, 0);
    assert_eq!(result.indexed, 0);
}

#[test]
fn build_index_with_partial_filter() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let mut interner = FieldInterner::new();

    insert_node(
        &engine,
        1,
        1,
        "User",
        &[
            ("email", Value::String("alice@test.com".into())),
            ("status", Value::String("active".into())),
        ],
        &mut interner,
    );
    insert_node(
        &engine,
        1,
        2,
        "User",
        &[
            ("email", Value::String("bob@test.com".into())),
            ("status", Value::String("inactive".into())),
        ],
        &mut interner,
    );

    let idx = IndexDefinition::btree("active_email", "User", "email").with_filter(
        super::super::definition::PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        },
    );

    let result = build_index(&engine, &idx, &interner, 1);
    assert_eq!(result.state, Some(IndexBuildState::Committed));
    assert_eq!(result.indexed, 1); // Only active user
    assert_eq!(result.skipped, 1); // Inactive filtered out
}
