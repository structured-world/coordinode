//! Index CRUD operations: create/delete entries, scan by value.
//!
//! Internally delegates to [`coordinode_modality::LocalIndexStore`]
//! for the typed key encode / put / delete / prefix-scan operations.
//! The module preserves its existing free-function signatures so
//! callers across the executor don't churn — the modality store is
//! the implementation, not the call surface.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_modality::{IndexStore as _, LocalIndexStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::error::StorageError;
use coordinode_storage::Guard;

use super::definition::IndexDefinition;
use coordinode_core::index::encoding::decode_node_id_from_index_key;

/// Convert [`coordinode_modality::StoreError`] back into the
/// existing [`StorageError`] vocabulary used by callers of this
/// module. `StoreError::Storage` unwraps to the underlying engine
/// error; other variants fold into a synthetic `PartitionNotFound`
/// with the message preserved.
fn map_store_err(e: StoreError) -> StorageError {
    match e {
        StoreError::Storage(se) => se,
        other => StorageError::PartitionNotFound {
            name: format!("index store: {other}"),
        },
    }
}

/// Create an index entry for a node's property value.
pub fn create_index_entry(
    engine: &StorageEngine,
    index: &IndexDefinition,
    node_id: NodeId,
    value: &Value,
) -> Result<(), StorageError> {
    let store = LocalIndexStore::new(engine);
    let values_slice = std::slice::from_ref(value);
    // For unique indexes, check for existing entries with the same value.
    if index.unique {
        let existing = store
            .scan_exact(&index.name, values_slice)
            .map_err(map_store_err)?;
        if existing
            .iter()
            .any(|existing_id| existing_id.as_raw() != node_id.as_raw())
        {
            return Err(StorageError::Conflict);
        }
    }
    // Index entries have empty values — the key itself carries the data.
    store
        .put_entry(&index.name, values_slice, node_id)
        .map_err(map_store_err)
}

/// Create index entries for a node, handling compound/sparse/multikey.
///
/// - Compound: encodes multiple property values into a single key.
/// - Sparse: skips the entry if any value is null.
/// - Multikey: for array values, creates one entry per array element.
pub fn create_index_entries(
    engine: &StorageEngine,
    index: &mut IndexDefinition,
    node_id: NodeId,
    values: &[Value],
) -> Result<(), StorageError> {
    // Sparse check: skip if any value is null
    if index.sparse && values.iter().any(|v| v.is_null()) {
        return Ok(());
    }

    // Multikey: check for array values and expand
    let has_array = values.iter().any(|v| matches!(v, Value::Array(_)));

    if has_array {
        // Set multikey flag permanently
        index.multikey = true;

        // Expand: for each array value, create entries for each element
        let expanded = expand_multikey(values);
        for combo in &expanded {
            write_compound_entry(engine, index, node_id, combo)?;
        }
    } else {
        write_compound_entry(engine, index, node_id, values)?;
    }

    Ok(())
}

/// Write a single compound index entry.
fn write_compound_entry(
    engine: &StorageEngine,
    index: &IndexDefinition,
    node_id: NodeId,
    values: &[Value],
) -> Result<(), StorageError> {
    let store = LocalIndexStore::new(engine);
    if index.unique {
        let existing = store
            .scan_exact(&index.name, values)
            .map_err(map_store_err)?;
        if existing
            .iter()
            .any(|existing_id| existing_id.as_raw() != node_id.as_raw())
        {
            return Err(StorageError::Conflict);
        }
    }
    store
        .put_entry(&index.name, values, node_id)
        .map_err(map_store_err)
}

/// Expand multikey values: for each array, produce combinations with elements.
///
/// Example: values = [String("Alice"), Array([Int(1), Int(2)])]
/// → [[String("Alice"), Int(1)], [String("Alice"), Int(2)]]
fn expand_multikey(values: &[Value]) -> Vec<Vec<Value>> {
    let mut results = vec![Vec::new()];

    for value in values {
        if let Value::Array(elements) = value {
            let mut new_results = Vec::new();
            for existing in &results {
                for elem in elements {
                    let mut combo = existing.clone();
                    combo.push(elem.clone());
                    new_results.push(combo);
                }
            }
            results = new_results;
        } else {
            for existing in &mut results {
                existing.push(value.clone());
            }
        }
    }

    results
}

/// Delete an index entry for a node's property value.
pub fn delete_index_entry(
    engine: &StorageEngine,
    index: &IndexDefinition,
    node_id: NodeId,
    value: &Value,
) -> Result<(), StorageError> {
    LocalIndexStore::new(engine)
        .delete_entry(&index.name, std::slice::from_ref(value), node_id)
        .map_err(map_store_err)
}

/// Delete compound index entries for a node.
pub fn delete_index_entries(
    engine: &StorageEngine,
    index: &IndexDefinition,
    node_id: NodeId,
    values: &[Value],
) -> Result<(), StorageError> {
    let store = LocalIndexStore::new(engine);
    let has_array = values.iter().any(|v| matches!(v, Value::Array(_)));
    if has_array {
        for combo in &expand_multikey(values) {
            store
                .delete_entry(&index.name, combo, node_id)
                .map_err(map_store_err)?;
        }
    } else {
        store
            .delete_entry(&index.name, values, node_id)
            .map_err(map_store_err)?;
    }
    Ok(())
}

/// Scan index for exact value match. Returns matching node IDs.
pub fn index_scan_exact(
    engine: &StorageEngine,
    index_name: &str,
    value: &Value,
) -> Result<Vec<u64>, StorageError> {
    let store = LocalIndexStore::new(engine);
    let ids = store
        .scan_exact(index_name, std::slice::from_ref(value))
        .map_err(map_store_err)?;
    Ok(ids.into_iter().map(|id| id.as_raw()).collect())
}

/// Scan index for all entries (full index scan). Returns (value_bytes, node_id) pairs.
/// Used for range queries and SHOW INDEX operations.
pub fn index_scan(
    engine: &StorageEngine,
    index: &IndexDefinition,
) -> Result<Vec<u64>, StorageError> {
    let prefix = index.key_prefix();
    let iter = engine.prefix_scan(Partition::Idx, &prefix)?;
    let mut results = Vec::new();

    for guard in iter {
        let (key, _value) = guard.into_inner().map_err(StorageError::Engine)?;

        if let Some(node_id) = decode_node_id_from_index_key(&key) {
            results.push(node_id);
        }
    }

    Ok(results)
}

/// Save index definition to schema partition.
pub fn save_index_definition(
    engine: &StorageEngine,
    index: &IndexDefinition,
) -> Result<(), StorageError> {
    let key = index.schema_key();
    let value = rmp_serde::to_vec(index).map_err(|e| StorageError::PartitionNotFound {
        name: format!("index serialize: {e}"),
    })?;
    engine.put(Partition::Schema, &key, &value)
}

/// Load index definition from schema partition.
pub fn load_index_definition(
    engine: &StorageEngine,
    name: &str,
) -> Result<Option<IndexDefinition>, StorageError> {
    let mut key = Vec::with_capacity(11 + name.len());
    key.extend_from_slice(b"schema:idx:");
    key.extend_from_slice(name.as_bytes());

    match engine.get(Partition::Schema, &key)? {
        Some(bytes) => {
            let def: IndexDefinition =
                rmp_serde::from_slice(&bytes).map_err(|e| StorageError::PartitionNotFound {
                    name: format!("index deserialize: {e}"),
                })?;
            Ok(Some(def))
        }
        None => Ok(None),
    }
}

/// Delete index definition and all entries.
pub fn drop_index(engine: &StorageEngine, index: &IndexDefinition) -> Result<(), StorageError> {
    // Delete definition
    let schema_key = index.schema_key();
    engine.delete(Partition::Schema, &schema_key)?;

    // Delete all index entries by scanning the prefix
    let prefix = index.key_prefix();
    let iter = engine.prefix_scan(Partition::Idx, &prefix)?;

    for guard in iter {
        let (key, _) = guard.into_inner().map_err(StorageError::Engine)?;
        engine.delete(Partition::Idx, &key)?;
    }

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

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

        let results =
            index_scan_exact(&engine, "user_email", &Value::String("bob@test.com".into()))
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
        create_index_entries(&engine, &mut idx, NodeId::from_raw(2), &[Value::Null])
            .expect("create");

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
        let rust =
            index_scan_exact(&engine, "user_tags", &Value::String("rust".into())).expect("scan");
        assert_eq!(rust, vec![1]);

        let graph =
            index_scan_exact(&engine, "user_tags", &Value::String("graph".into())).expect("scan");
        assert_eq!(graph, vec![1]);

        let db = index_scan_exact(&engine, "user_tags", &Value::String("database".into()))
            .expect("scan");
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
}
