use super::*;
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

#[test]
fn save_and_load_definition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_email", "User", "email").unique();

    save_index_definition(&engine, &idx).expect("save");
    let loaded = load_index_definition(&engine, "user_email")
        .expect("load")
        .expect("should exist");

    assert_eq!(loaded.name, "user_email");
    assert_eq!(loaded.label, "User");
    assert_eq!(loaded.property(), "email");
    assert!(loaded.unique);
}

#[test]
fn list_index_definitions_returns_every_persisted_definition() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    save_index_definition(
        &engine,
        &IndexDefinition::btree("user_email", "User", "email").unique(),
    )
    .expect("save email");
    save_index_definition(&engine, &IndexDefinition::btree("user_age", "User", "age"))
        .expect("save age");
    save_index_definition(
        &engine,
        &IndexDefinition::compound(
            "order_total_status",
            "Order",
            vec!["total".into(), "status".into()],
        ),
    )
    .expect("save order");

    let mut listed = list_index_definitions(&engine).expect("list");
    listed.sort_by(|l, r| l.name.cmp(&r.name));
    assert_eq!(listed.len(), 3);
    assert_eq!(listed[0].name, "order_total_status");
    assert_eq!(listed[1].name, "user_age");
    assert_eq!(listed[2].name, "user_email");
    assert!(listed[2].unique);
}

#[test]
fn list_index_definitions_empty_when_no_definitions_persisted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());
    let listed = list_index_definitions(&engine).expect("list");
    assert!(listed.is_empty());
}

/// A corrupt `schema:idx:` entry must not abort the listing: it is skipped
/// with a warning so one bad definition does not take down the registry on
/// open.
#[test]
fn list_index_definitions_skips_corrupt_bodies() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    save_index_definition(
        &engine,
        &IndexDefinition::btree("user_email", "User", "email"),
    )
    .expect("save real");
    engine
        .put(
            coordinode_storage::engine::partition::Partition::Schema,
            b"schema:idx:garbage",
            b"not-msgpack-bytes",
        )
        .expect("plant garbage");

    let listed = list_index_definitions(&engine).expect("list");
    assert_eq!(listed.len(), 1, "corrupt entry skipped, real one kept");
    assert_eq!(listed[0].name, "user_email");
}

#[test]
fn save_index_state_updates_persisted_state_only() {
    use crate::index::definition::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let def = IndexDefinition::hnsw("v_idx", "Doc", "embed", VectorIndexConfig::default());
    save_index_definition(&engine, &def).expect("save");

    let updated = save_index_state(
        &engine,
        "v_idx",
        IndexState::Building {
            written: 100,
            estimated_total: 1000,
        },
    )
    .expect("save state");
    assert!(updated, "save_index_state should report success");

    let reloaded = load_index_definition(&engine, "v_idx")
        .expect("load")
        .expect("present");
    assert_eq!(
        reloaded.state,
        IndexState::Building {
            written: 100,
            estimated_total: 1000
        }
    );
    assert_eq!(reloaded.name, "v_idx");
    assert_eq!(reloaded.label, "Doc");
    assert_eq!(reloaded.properties, vec!["embed".to_string()]);

    let updated = save_index_state(&engine, "v_idx", IndexState::Ready).expect("save ready");
    assert!(updated);
    let reloaded = load_index_definition(&engine, "v_idx")
        .expect("load")
        .expect("present");
    assert_eq!(reloaded.state, IndexState::Ready);
}

#[test]
fn save_index_state_missing_index_returns_false() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let updated = save_index_state(&engine, "does_not_exist", IndexState::Ready)
        .expect("save state should not error on missing");
    assert!(
        !updated,
        "missing index should report not-found via Ok(false)"
    );
}
