//! Index registry: tracks active indexes for automatic maintenance.
//!
//! The executor uses the registry to know which indexes to update
//! when nodes are created, modified, or deleted.

use std::collections::HashMap;
use std::sync::RwLock;

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::error::StorageError;

use super::definition::IndexDefinition;
use super::ops::{create_index_entry, delete_index_entry, save_index_definition};

/// Registry of active indexes.
///
/// Uses interior mutability (`RwLock`) so `register` / `unregister` / `load_all`
/// can be called via `&self`. This allows Cypher DDL (`CREATE INDEX`) executed
/// from within the runner to update the live registry without requiring
/// a `&mut` reference through the `ExecutionContext`.
pub struct IndexRegistry {
    /// Active indexes: name → definition.
    indexes: RwLock<HashMap<String, IndexDefinition>>,
}

impl IndexRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new index. Also persists the definition to storage.
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register(
        &self,
        engine: &StorageEngine,
        index: IndexDefinition,
    ) -> Result<(), StorageError> {
        save_index_definition(engine, &index)?;
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(index.name.clone(), index);
        Ok(())
    }

    /// Register an index in memory only (no storage persistence).
    ///
    /// Used for testing and for pre-populating the registry from cached state.
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register_in_memory(&self, index: IndexDefinition) {
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(index.name.clone(), index);
    }

    /// Unregister an index (does NOT drop entries — use `ops::drop_index` for that).
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn unregister(&self, name: &str) {
        self.indexes
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(name);
    }

    /// Load all index definitions from storage.
    ///
    /// Uses interior mutability — safe to call via `&self`.
    /// Routes through the typed [`super::ops::list_index_definitions`]
    /// helper so this method doesn't reach into raw `Partition::Schema`
    /// prefix scans.
    pub fn load_all(&self, engine: &StorageEngine) -> Result<(), StorageError> {
        let defs = super::ops::list_index_definitions(engine)?;
        let mut map = self.indexes.write().unwrap_or_else(|e| e.into_inner());
        for def in defs {
            map.insert(def.name.clone(), def);
        }
        Ok(())
    }

    /// Get all indexes for a specific label (returns owned clones).
    pub fn indexes_for_label(&self, label: &str) -> Vec<IndexDefinition> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|idx| idx.label == label)
            .cloned()
            .collect()
    }

    /// Get all indexes that cover a specific label + property (returns owned clones).
    pub fn indexes_for_property(&self, label: &str, property: &str) -> Vec<IndexDefinition> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|idx| idx.label == label && idx.properties.contains(&property.to_string()))
            .cloned()
            .collect()
    }

    /// Get an index by name (returns owned clone).
    pub fn get(&self, name: &str) -> Option<IndexDefinition> {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(name)
            .cloned()
    }

    /// Check if any index exists for a label.
    pub fn has_indexes_for(&self, label: &str) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .any(|idx| idx.label == label)
    }

    /// Number of registered indexes.
    pub fn len(&self) -> usize {
        self.indexes.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.indexes
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    /// Maintain indexes when a node is created.
    ///
    /// For each index on the node's label, create an index entry
    /// for the relevant property values. Returns an error if a
    /// unique constraint is violated.
    pub fn on_node_created(
        &self,
        engine: &StorageEngine,
        node_id: NodeId,
        label: &str,
        properties: &[(String, Value)],
    ) -> Result<(), UniqueViolation> {
        for idx in self.indexes_for_label(label) {
            // Check partial filter — skip if node doesn't match
            if !idx.matches_filter(properties) {
                continue;
            }

            // For single-field indexes, find the matching property
            if !idx.is_compound() {
                let prop_name = idx.property().to_string();
                if let Some((_, value)) = properties.iter().find(|(k, _)| k == &prop_name) {
                    if idx.sparse && value.is_null() {
                        continue;
                    }
                    create_index_entry(engine, &idx, node_id, value).map_err(|e| match e {
                        StorageError::Conflict => UniqueViolation {
                            index_name: idx.name.clone(),
                            property: prop_name.clone(),
                            value: value.clone(),
                        },
                        _ => UniqueViolation {
                            index_name: idx.name.clone(),
                            property: prop_name.clone(),
                            value: Value::Null,
                        },
                    })?;
                }
            }
        }
        Ok(())
    }

    /// Maintain indexes when a node property is updated.
    ///
    /// Removes the old index entry and creates a new one.
    pub fn on_property_changed(
        &self,
        engine: &StorageEngine,
        node_id: NodeId,
        label: &str,
        property: &str,
        old_value: Option<&Value>,
        new_value: &Value,
    ) -> Result<(), UniqueViolation> {
        for idx in self.indexes_for_property(label, property) {
            if idx.is_compound() {
                continue; // Compound index maintenance is more complex
            }

            // Delete old entry
            if let Some(old) = old_value {
                let _ = delete_index_entry(engine, &idx, node_id, old);
            }

            // Create new entry
            if !(idx.sparse && new_value.is_null()) {
                create_index_entry(engine, &idx, node_id, new_value).map_err(|e| match e {
                    StorageError::Conflict => UniqueViolation {
                        index_name: idx.name.clone(),
                        property: property.to_string(),
                        value: new_value.clone(),
                    },
                    _ => UniqueViolation {
                        index_name: idx.name.clone(),
                        property: property.to_string(),
                        value: Value::Null,
                    },
                })?;
            }
        }
        Ok(())
    }

    /// Maintain indexes when a node is deleted.
    pub fn on_node_deleted(
        &self,
        engine: &StorageEngine,
        node_id: NodeId,
        label: &str,
        properties: &[(String, Value)],
    ) -> Result<(), StorageError> {
        for idx in self.indexes_for_label(label) {
            if !idx.is_compound() {
                let prop_name = idx.property().to_string();
                if let Some((_, value)) = properties.iter().find(|(k, _)| k == &prop_name) {
                    delete_index_entry(engine, &idx, node_id, value)?;
                }
            }
        }
        Ok(())
    }
}

impl Default for IndexRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Error when a unique constraint is violated.
#[derive(Debug, Clone, thiserror::Error)]
#[error("unique constraint violated on index `{index_name}`: property `{property}` already has value {value:?}")]
pub struct UniqueViolation {
    pub index_name: String,
    pub property: String,
    pub value: Value,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
