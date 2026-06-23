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
fn create_and_scan_index_entry() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_email", "User", "email");

    // Create entries
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("alice@test.com".into()),
    )
    .expect("create");

    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(2),
        &Value::String("bob@test.com".into()),
    )
    .expect("create");

    // Exact scan
    let results = index_scan_exact(
        &engine,
        "user_email",
        &Value::String("alice@test.com".into()),
    )
    .expect("scan");
    assert_eq!(results, vec![1]);

    let results = index_scan_exact(&engine, "user_email", &Value::String("bob@test.com".into()))
        .expect("scan");
    assert_eq!(results, vec![2]);

    // Non-existent value
    let results = index_scan_exact(
        &engine,
        "user_email",
        &Value::String("nobody@test.com".into()),
    )
    .expect("scan");
    assert!(results.is_empty());
}

#[test]
fn delete_index_entry_removes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_name", "User", "name");

    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("Alice".into()),
    )
    .expect("create");

    // Verify exists
    let results =
        index_scan_exact(&engine, "user_name", &Value::String("Alice".into())).expect("scan");
    assert_eq!(results.len(), 1);

    // Delete
    delete_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("Alice".into()),
    )
    .expect("delete");

    // Verify gone
    let results =
        index_scan_exact(&engine, "user_name", &Value::String("Alice".into())).expect("scan");
    assert!(results.is_empty());
}

#[test]
fn unique_index_rejects_duplicate() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_email", "User", "email").unique();

    // First insert succeeds
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("alice@test.com".into()),
    )
    .expect("first insert");

    // Duplicate value with different node ID should fail
    let result = create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(2),
        &Value::String("alice@test.com".into()),
    );
    assert!(result.is_err());
}

#[test]
fn unique_index_allows_same_node_update() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_email", "User", "email").unique();

    // Insert
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("alice@test.com".into()),
    )
    .expect("insert");

    // Same node, same value should succeed (idempotent)
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("alice@test.com".into()),
    )
    .expect("re-insert same node");
}

#[test]
fn full_index_scan() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_age", "User", "age");

    create_index_entry(&engine, &idx, NodeId::from_raw(1), &Value::Int(30)).expect("create");
    create_index_entry(&engine, &idx, NodeId::from_raw(2), &Value::Int(25)).expect("create");
    create_index_entry(&engine, &idx, NodeId::from_raw(3), &Value::Int(35)).expect("create");

    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 3);
    // Should be sorted by value (25 < 30 < 35)
    assert_eq!(results, vec![2, 1, 3]);
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

#[test]
fn list_index_definitions_skips_corrupt_bodies() {
    // A corrupt `schema:idx:` entry must not abort the listing —
    // it should be skipped with a tracing warning so a single
    // bad definition doesn't take down registry rebuild on
    // engine open.
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    save_index_definition(
        &engine,
        &IndexDefinition::btree("user_email", "User", "email"),
    )
    .expect("save real");
    // Plant garbage under `schema:idx:garbage` so it's in the
    // prefix scan results.
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
fn drop_index_removes_all() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("user_name", "User", "name");

    // Create entries and definition
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &Value::String("Alice".into()),
    )
    .expect("create");
    create_index_entry(
        &engine,
        &idx,
        NodeId::from_raw(2),
        &Value::String("Bob".into()),
    )
    .expect("create");
    save_index_definition(&engine, &idx).expect("save");

    // Drop
    drop_index(&engine, &idx).expect("drop");

    // Definition gone
    let loaded = load_index_definition(&engine, "user_name").expect("load");
    assert!(loaded.is_none());

    // Entries gone
    let results = index_scan(&engine, &idx).expect("scan");
    assert!(results.is_empty());
}

#[test]
fn integer_index_ordering() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let idx = IndexDefinition::btree("score", "Player", "score");

    // Insert in random order
    create_index_entry(&engine, &idx, NodeId::from_raw(3), &Value::Int(100)).expect("create");
    create_index_entry(&engine, &idx, NodeId::from_raw(1), &Value::Int(-50)).expect("create");
    create_index_entry(&engine, &idx, NodeId::from_raw(2), &Value::Int(0)).expect("create");

    // Full scan should return sorted by value
    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results, vec![1, 2, 3]); // -50, 0, 100
}

// ====== Compound index ======

#[test]
fn compound_index_creation() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx = IndexDefinition::compound(
        "user_label_status",
        "User",
        vec!["label".into(), "status".into()],
    );

    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[Value::String("User".into()), Value::String("active".into())],
    )
    .expect("create");

    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(2),
        &[
            Value::String("User".into()),
            Value::String("inactive".into()),
        ],
    )
    .expect("create");

    // Full scan should include both
    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 2);
    // "active" sorts before "inactive"
    assert_eq!(results, vec![1, 2]);
}

#[test]
fn compound_index_ordering() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx =
        IndexDefinition::compound("user_age_name", "User", vec!["age".into(), "name".into()]);

    // Same age (30), different names
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[Value::Int(30), Value::String("Charlie".into())],
    )
    .expect("create");

    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(2),
        &[Value::Int(30), Value::String("Alice".into())],
    )
    .expect("create");

    // Different age (25)
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(3),
        &[Value::Int(25), Value::String("Bob".into())],
    )
    .expect("create");

    // Should sort by age first, then name within same age
    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results, vec![3, 2, 1]); // 25/Bob, 30/Alice, 30/Charlie
}

// ====== Sparse index ======

#[test]
fn sparse_index_skips_nulls() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx = IndexDefinition::btree("user_bio", "User", "bio").sparse();

    // Node 1 has a bio
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[Value::String("Rust developer".into())],
    )
    .expect("create");

    // Node 2 has null bio — should be skipped
    create_index_entries(&engine, &mut idx, NodeId::from_raw(2), &[Value::Null]).expect("create");

    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 1); // Only node 1
    assert_eq!(results[0], 1);
}

#[test]
fn sparse_compound_skips_any_null() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx =
        IndexDefinition::compound("user_name_bio", "User", vec!["name".into(), "bio".into()])
            .sparse();

    // Both non-null
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[
            Value::String("Alice".into()),
            Value::String("Developer".into()),
        ],
    )
    .expect("create");

    // Second value null — skip
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(2),
        &[Value::String("Bob".into()), Value::Null],
    )
    .expect("create");

    let results = index_scan(&engine, &idx).expect("scan");
    assert_eq!(results.len(), 1);
}

// ====== Multikey index ======

#[test]
fn multikey_index_expands_array() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx = IndexDefinition::btree("user_tags", "User", "tags");
    assert!(!idx.multikey);

    // Node with array value
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[Value::Array(vec![
            Value::String("rust".into()),
            Value::String("graph".into()),
            Value::String("database".into()),
        ])],
    )
    .expect("create");

    // Multikey flag should be set
    assert!(idx.multikey);

    // Should be findable by any individual tag
    let rust = index_scan_exact(&engine, "user_tags", &Value::String("rust".into())).expect("scan");
    assert_eq!(rust, vec![1]);

    let graph =
        index_scan_exact(&engine, "user_tags", &Value::String("graph".into())).expect("scan");
    assert_eq!(graph, vec![1]);

    let db =
        index_scan_exact(&engine, "user_tags", &Value::String("database".into())).expect("scan");
    assert_eq!(db, vec![1]);

    // Full scan should have 3 entries
    let all = index_scan(&engine, &idx).expect("scan");
    assert_eq!(all.len(), 3);
}

#[test]
fn multikey_compound_cartesian() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx =
        IndexDefinition::compound("user_name_tags", "User", vec!["name".into(), "tags".into()]);

    // Compound with array in second position
    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[
            Value::String("Alice".into()),
            Value::Array(vec![
                Value::String("rust".into()),
                Value::String("go".into()),
            ]),
        ],
    )
    .expect("create");

    assert!(idx.multikey);

    // Should create 2 entries: (Alice, rust) and (Alice, go)
    let all = index_scan(&engine, &idx).expect("scan");
    assert_eq!(all.len(), 2);
}

#[test]
fn delete_multikey_entries() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut idx = IndexDefinition::btree("user_tags", "User", "tags");

    create_index_entries(
        &engine,
        &mut idx,
        NodeId::from_raw(1),
        &[Value::Array(vec![
            Value::String("a".into()),
            Value::String("b".into()),
        ])],
    )
    .expect("create");

    assert_eq!(index_scan(&engine, &idx).expect("scan").len(), 2);

    // Delete
    delete_index_entries(
        &engine,
        &idx,
        NodeId::from_raw(1),
        &[Value::Array(vec![
            Value::String("a".into()),
            Value::String("b".into()),
        ])],
    )
    .expect("delete");

    assert_eq!(index_scan(&engine, &idx).expect("scan").len(), 0);
}

#[test]
fn save_index_state_updates_persisted_state_only() {
    use crate::index::definition::VectorIndexConfig;

    let dir = tempfile::tempdir().expect("tempdir");
    let engine = test_engine(dir.path());

    let mut def = IndexDefinition::hnsw("v_idx", "Doc", "embed", VectorIndexConfig::default());
    def.unique = false;
    save_index_definition(&engine, &def).expect("save");

    // Update state only.
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
    // Other fields preserved.
    assert_eq!(reloaded.name, "v_idx");
    assert_eq!(reloaded.label, "Doc");
    assert_eq!(reloaded.properties, vec!["embed".to_string()]);

    // Subsequent state update overwrites.
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
