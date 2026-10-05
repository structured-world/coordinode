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

    // A save outside the log rewrites the record under its identity.
    let mut changed = loaded.clone();
    changed.description = Some("emails".into());
    save_index_definition(&engine, &changed).expect("save");
    assert_eq!(
        load_index_definition(&engine, idx.id).expect("load"),
        Some(changed)
    );
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

/// A corrupt definition record must not abort the listing: it is skipped
/// with a warning so one bad definition does not take down the registry on
/// open.
#[test]
fn list_index_definitions_skips_corrupt_bodies() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let real = publish(
        &engine,
        IndexDescriptor::btree("user_email", "User", "email"),
    );
    engine
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            &IndexDefinition::schema_key_of(crate::index::IndexId::from_raw(999)),
            b"not-msgpack-bytes",
        )
        .expect("plant garbage");

    let listed = list_index_definitions(&engine).expect("list");
    assert_eq!(listed, vec![real], "corrupt entry skipped, real one kept");
}

#[test]
fn save_index_state_updates_persisted_state_only() {
    use crate::index::definition::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let def = publish(
        &engine,
        IndexDescriptor::hnsw("v_idx", "Doc", "embed", VectorIndexConfig::default()),
    );

    let updated = save_index_state(
        &engine,
        def.id,
        IndexState::Building {
            written: 100,
            estimated_total: 1000,
        },
    )
    .expect("save state");
    assert!(updated, "save_index_state should report success");

    let reloaded = load_index_definition(&engine, def.id)
        .expect("load")
        .expect("present");
    assert_eq!(
        reloaded.state,
        IndexState::Building {
            written: 100,
            estimated_total: 1000
        }
    );
    assert_eq!(reloaded.name.as_deref(), Some("v_idx"));
    assert_eq!(reloaded.label, "Doc");
    assert_eq!(reloaded.properties, vec!["embed".to_string()]);
    assert_eq!(reloaded.generation, def.generation);

    let updated = save_index_state(&engine, def.id, IndexState::Ready).expect("save ready");
    assert!(updated);
    let reloaded = load_index_definition(&engine, def.id)
        .expect("load")
        .expect("present");
    assert_eq!(reloaded.state, IndexState::Ready);
}

/// A state saved for an identity no record holds changes nothing: a build
/// that outlived its index cannot mark a successor created under the name.
#[test]
fn save_index_state_missing_index_returns_false() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let updated = save_index_state(
        &engine,
        crate::index::IndexId::from_raw(7),
        IndexState::Ready,
    )
    .expect("save state should not error on missing");
    assert!(
        !updated,
        "missing index should report not-found via Ok(false)"
    );
}
