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
use coordinode_storage::error::StorageError;

use super::definition::{IndexDefinition, IndexState};

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
    let ids = LocalIndexStore::new(engine)
        .scan_entry_ids(&index.name)
        .map_err(map_store_err)?;
    Ok(ids.into_iter().map(|id| id.as_raw()).collect())
}

/// Save index definition to the index-store catalog.
pub fn save_index_definition(
    engine: &StorageEngine,
    index: &IndexDefinition,
) -> Result<(), StorageError> {
    LocalIndexStore::new(engine)
        .put_definition(index)
        .map_err(map_store_err)
}

/// Update only the `state` field of a persisted index definition.
///
/// Loads the current definition, swaps `state`, writes the result back. Used
/// by the backfill task to publish progress and terminal states without
/// rewriting any other field. Returns `Ok(false)` if the index has no
/// persisted definition (caller's responsibility to handle the race).
pub fn save_index_state(
    engine: &StorageEngine,
    name: &str,
    state: IndexState,
) -> Result<bool, StorageError> {
    LocalIndexStore::new(engine)
        .set_definition_state(name, state)
        .map_err(map_store_err)
}

/// Load index definition from the index-store catalog.
pub fn load_index_definition(
    engine: &StorageEngine,
    name: &str,
) -> Result<Option<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .load_definition(name)
        .map_err(map_store_err)
}

/// List every persisted index definition in `schema:idx:` order.
///
/// Skips entries whose body fails to decode rather than aborting the
/// whole list (a corrupt index def shouldn't take out the registry).
pub fn list_index_definitions(
    engine: &StorageEngine,
) -> Result<Vec<IndexDefinition>, StorageError> {
    LocalIndexStore::new(engine)
        .list_definitions()
        .map_err(map_store_err)
}

/// Delete index definition and all entries.
pub fn drop_index(engine: &StorageEngine, index: &IndexDefinition) -> Result<(), StorageError> {
    let store = LocalIndexStore::new(engine);
    store
        .delete_definition(&index.name)
        .map_err(map_store_err)?;
    store.clear(&index.name).map_err(map_store_err)?;
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
