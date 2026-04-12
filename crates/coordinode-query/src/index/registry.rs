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
use coordinode_storage::Guard;

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
    pub fn load_all(&self, engine: &StorageEngine) -> Result<(), StorageError> {
        // Scan the schema:idx: prefix for all index definitions
        let iter = engine.prefix_scan(
            coordinode_storage::engine::partition::Partition::Schema,
            b"schema:idx:",
        )?;

        let mut map = self.indexes.write().unwrap_or_else(|e| e.into_inner());
        for guard in iter {
            let (_key, value) = guard.into_inner().map_err(StorageError::Engine)?;
            if let Ok(def) = rmp_serde::from_slice::<IndexDefinition>(&value) {
                map.insert(def.name.clone(), def);
            }
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
mod tests {
    use super::*;
    use coordinode_storage::engine::config::StorageConfig;

    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::new(dir);
        StorageEngine::open(&config).expect("open engine")
    }

    #[test]
    fn register_and_lookup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        let idx = IndexDefinition::btree("user_email", "User", "email").unique();
        reg.register(&engine, idx).expect("register");

        assert_eq!(reg.len(), 1);
        assert!(reg.get("user_email").is_some());
        assert_eq!(reg.indexes_for_label("User").len(), 1);
        assert_eq!(reg.indexes_for_label("Movie").len(), 0);
    }

    #[test]
    fn indexes_for_property() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email"),
        )
        .expect("register");
        reg.register(&engine, IndexDefinition::btree("user_name", "User", "name"))
            .expect("register");

        let email_idxs = reg.indexes_for_property("User", "email");
        assert_eq!(email_idxs.len(), 1);
        assert_eq!(email_idxs[0].name, "user_email");
    }

    #[test]
    fn on_node_created_unique_ok() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email").unique(),
        )
        .expect("register");

        // Create first node — should succeed
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        )
        .expect("first create");

        // Create second node with different email — should succeed
        reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[("email".to_string(), Value::String("bob@test.com".into()))],
        )
        .expect("second create");
    }

    #[test]
    fn on_node_created_unique_violation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email").unique(),
        )
        .expect("register");

        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        )
        .expect("first");

        // Duplicate email — should fail
        let result = reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("unique constraint violated"));
        assert!(err.to_string().contains("user_email"));
    }

    #[test]
    fn on_property_changed_updates_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email").unique(),
        )
        .expect("register");

        // Create initial entry
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("old@test.com".into()))],
        )
        .expect("create");

        // Update property
        reg.on_property_changed(
            &engine,
            NodeId::from_raw(1),
            "User",
            "email",
            Some(&Value::String("old@test.com".into())),
            &Value::String("new@test.com".into()),
        )
        .expect("update");

        // Old value should not be findable
        let old = super::super::ops::index_scan_exact(
            &engine,
            "user_email",
            &Value::String("old@test.com".into()),
        )
        .expect("scan");
        assert!(old.is_empty());

        // New value should be findable
        let new = super::super::ops::index_scan_exact(
            &engine,
            "user_email",
            &Value::String("new@test.com".into()),
        )
        .expect("scan");
        assert_eq!(new, vec![1]);
    }

    #[test]
    fn on_property_changed_unique_violation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email").unique(),
        )
        .expect("register");

        // Two nodes with different emails
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        )
        .expect("create 1");

        reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[("email".to_string(), Value::String("bob@test.com".into()))],
        )
        .expect("create 2");

        // Try to change node 2's email to alice's — should fail
        let result = reg.on_property_changed(
            &engine,
            NodeId::from_raw(2),
            "User",
            "email",
            Some(&Value::String("bob@test.com".into())),
            &Value::String("alice@test.com".into()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn on_node_deleted_removes_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_email", "User", "email"),
        )
        .expect("register");

        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        )
        .expect("create");

        // Delete
        reg.on_node_deleted(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("email".to_string(), Value::String("alice@test.com".into()))],
        )
        .expect("delete");

        // Should not be findable
        let results = super::super::ops::index_scan_exact(
            &engine,
            "user_email",
            &Value::String("alice@test.com".into()),
        )
        .expect("scan");
        assert!(results.is_empty());
    }

    #[test]
    fn load_all_from_storage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());

        // Save definitions
        {
            let reg = IndexRegistry::new();
            reg.register(
                &engine,
                IndexDefinition::btree("idx1", "User", "email").unique(),
            )
            .expect("register");
            reg.register(&engine, IndexDefinition::btree("idx2", "User", "name"))
                .expect("register");
        }

        // Load in new registry
        let reg2 = IndexRegistry::new();
        reg2.load_all(&engine).expect("load");
        assert_eq!(reg2.len(), 2);
        assert!(reg2.get("idx1").is_some());
        assert!(reg2.get("idx2").is_some());
    }

    #[test]
    fn sparse_index_skips_null_on_create() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_bio", "User", "bio").sparse(),
        )
        .expect("register");

        // Create node with null bio — should be skipped
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[("bio".to_string(), Value::Null)],
        )
        .expect("create");

        let idx = reg.get("user_bio").expect("get");
        let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
        assert!(results.is_empty());
    }

    // ====== Partial index ======

    #[test]
    fn partial_index_filters_on_create() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        // Index only active users
        reg.register(
            &engine,
            IndexDefinition::btree("active_email", "User", "email").with_filter(
                super::super::definition::PartialFilter::PropertyEquals {
                    property: "status".into(),
                    value: "active".into(),
                },
            ),
        )
        .expect("register");

        // Active user — should be indexed
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[
                ("email".to_string(), Value::String("alice@test.com".into())),
                ("status".to_string(), Value::String("active".into())),
            ],
        )
        .expect("create active");

        // Inactive user — should NOT be indexed
        reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[
                ("email".to_string(), Value::String("bob@test.com".into())),
                ("status".to_string(), Value::String("inactive".into())),
            ],
        )
        .expect("create inactive");

        // Only active user should be in index
        let idx = reg.get("active_email").expect("get");
        let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1); // Only Alice
    }

    #[test]
    fn partial_index_bool_filter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("verified_email", "User", "email").with_filter(
                super::super::definition::PartialFilter::PropertyEqualsBool {
                    property: "verified".into(),
                    value: true,
                },
            ),
        )
        .expect("register");

        // Verified user
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[
                ("email".to_string(), Value::String("alice@test.com".into())),
                ("verified".to_string(), Value::Bool(true)),
            ],
        )
        .expect("create");

        // Unverified user
        reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[
                ("email".to_string(), Value::String("bob@test.com".into())),
                ("verified".to_string(), Value::Bool(false)),
            ],
        )
        .expect("create");

        let idx = reg.get("verified_email").expect("get");
        let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1);
    }

    #[test]
    fn partial_index_exists_filter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let engine = test_engine(dir.path());
        let reg = IndexRegistry::new();

        reg.register(
            &engine,
            IndexDefinition::btree("user_bio_idx", "User", "bio").with_filter(
                super::super::definition::PartialFilter::PropertyExists {
                    property: "bio".into(),
                },
            ),
        )
        .expect("register");

        // User WITH bio
        reg.on_node_created(
            &engine,
            NodeId::from_raw(1),
            "User",
            &[
                ("name".to_string(), Value::String("Alice".into())),
                ("bio".to_string(), Value::String("Developer".into())),
            ],
        )
        .expect("create");

        // User WITHOUT bio (null)
        reg.on_node_created(
            &engine,
            NodeId::from_raw(2),
            "User",
            &[
                ("name".to_string(), Value::String("Bob".into())),
                ("bio".to_string(), Value::Null),
            ],
        )
        .expect("create");

        let idx = reg.get("user_bio_idx").expect("get");
        let results = super::super::ops::index_scan(&engine, &idx).expect("scan");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], 1);
    }

    #[test]
    fn partial_filter_matches_function() {
        use super::super::definition::PartialFilter;

        let props = vec![
            ("status".to_string(), Value::String("active".into())),
            ("age".to_string(), Value::Int(30)),
        ];

        // String equality
        assert!(PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "active".into(),
        }
        .matches(&props));

        assert!(!PartialFilter::PropertyEquals {
            property: "status".into(),
            value: "inactive".into(),
        }
        .matches(&props));

        // Int equality
        assert!(PartialFilter::PropertyEqualsInt {
            property: "age".into(),
            value: 30,
        }
        .matches(&props));

        // Exists
        assert!(PartialFilter::PropertyExists {
            property: "status".into(),
        }
        .matches(&props));

        // Not exists
        assert!(!PartialFilter::PropertyExists {
            property: "missing".into(),
        }
        .matches(&props));
    }
}
