use super::*;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::transaction::Transaction;

use crate::index::IndexDescriptor;

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

/// Publish `descriptor` through the catalog in a direct-mode transaction,
/// whose writes land as they are staged.
fn publish(engine: &StorageEngine, descriptor: IndexDescriptor) -> IndexDefinition {
    LocalIndexStore::new(engine)
        .publish_definition_txn(
            &mut Transaction::new(engine, None, Timestamp::ZERO, None),
            descriptor,
        )
        .expect("publish")
}

#[test]
fn save_and_load_definition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = publish(
        &engine,
        IndexDescriptor::btree("user_email", "User", "email").unique(),
    );
    let loaded = load_index_definition(&engine, idx.id)
        .expect("load")
        .expect("should exist");

    assert_eq!(loaded, idx);
    assert_eq!(loaded.name.as_deref(), Some("user_email"));
    assert_eq!(loaded.label, "User");
    assert_eq!(loaded.property(), "email");
    assert!(loaded.unique);
}

#[test]
fn list_index_definitions_returns_every_persisted_definition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    publish(
        &engine,
        IndexDescriptor::btree("user_email", "User", "email").unique(),
    );
    publish(&engine, IndexDescriptor::btree("user_age", "User", "age"));
    publish(
        &engine,
        IndexDescriptor::compound(
            "order_total_status",
            "Order",
            vec!["total".into(), "status".into()],
        ),
    );

    let mut listed = list_index_definitions(&engine).expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 3);
    assert_eq!(listed[0].name.as_deref(), Some("order_total_status"));
    assert_eq!(listed[1].name.as_deref(), Some("user_age"));
    assert_eq!(listed[2].name.as_deref(), Some("user_email"));
    assert!(listed[2].unique);
}

#[test]
fn list_index_definitions_empty_when_no_definitions_persisted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let listed = list_index_definitions(&engine).expect("list");
    assert!(listed.is_empty());
}

/// A definition record this build cannot read refuses the listing, naming
/// the record. Skipping it served the database without that index, and
/// without the uniqueness it enforced, with only a log line to say so: a
/// directory written by a development build with the earlier name-keyed
/// catalog opened that way.
#[test]
fn list_index_definitions_refuses_an_unreadable_record() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    publish(
        &engine,
        IndexDescriptor::btree("user_email", "User", "email"),
    );
    let key = IndexDefinition::schema_key_of(crate::index::IndexId::from_raw(999));
    engine
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &key,
            b"not-msgpack-bytes",
        )
        .expect("plant garbage");

    match list_index_definitions(&engine) {
        Err(StorageError::UnreadableCatalog {
            kind, key: named, ..
        }) => {
            assert_eq!(kind, "index definition");
            assert_eq!(named, coordinode_storage::error::printable_key(&key));
        }
        other => panic!("expected the unreadable record to refuse the listing, got {other:?}"),
    }
}
