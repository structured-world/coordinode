//! Text index registry: tracks active tantivy text indexes for full-text search.
//!
//! Holds in-memory tantivy index instances keyed by (label, property).
//! Indexes are populated from stored node text on `Database::open()` and
//! maintained incrementally on node create/update/delete.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use coordinode_core::graph::node::NodeId;
use coordinode_search::tantivy::multi_lang::{MultiLangConfig, MultiLanguageTextIndex};
use coordinode_search::tantivy::TextSearchResult;

use super::definition::{IndexDefinition, TextIndexConfig};

/// Key for text index lookup: (label, property).
type TextIndexKey = (String, String);

/// Thread-safe handle to a multi-language text index.
pub type TextHandle = Arc<RwLock<MultiLanguageTextIndex>>;

/// Registry of active tantivy text indexes.
///
/// Uses interior mutability (`RwLock`) so register/unregister/on_text_written
/// can all be called via `&self`. This allows the executor to manage indexes
/// through an immutable `ExecutionContext` reference.
pub struct TextIndexRegistry {
    /// Active text indexes: (label, property) → live tantivy index.
    indexes: RwLock<HashMap<TextIndexKey, TextHandle>>,
    /// Index definitions keyed by (label, property) for metadata lookup.
    definitions: RwLock<HashMap<TextIndexKey, IndexDefinition>>,
    /// Base directory for tantivy index data. Each index gets a subdirectory.
    base_dir: PathBuf,
}

impl TextIndexRegistry {
    /// Create an empty registry with the given base directory for index storage.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            base_dir: base_dir.into(),
        }
    }

    /// Convert a `TextIndexConfig` to `MultiLangConfig`.
    fn to_multi_lang_config(config: &TextIndexConfig) -> MultiLangConfig {
        let mut field_analyzers = HashMap::new();
        for (field, fc) in &config.fields {
            field_analyzers.insert(field.clone(), fc.analyzer.clone());
        }
        MultiLangConfig {
            default_language: config.default_language.clone(),
            field_analyzers,
            language_override_property: config.language_override_property.clone(),
        }
    }

    /// Register a new text index, creating tantivy directories.
    ///
    /// For multi-field indexes, creates a SEPARATE tantivy index per property
    /// (each with its own per-field analyzer from the config). This is because
    /// tantivy's single-field schema stores one text blob per document —
    /// sharing one index across properties would overwrite on the same node_id.
    ///
    /// Uses interior mutability — safe to call via `&self`.
    pub fn register(&self, def: IndexDefinition) -> Result<(), String> {
        let Some(config) = def.text_config.as_ref() else {
            return Err(format!(
                "register called with non-text IndexDefinition: {}",
                def.name
            ));
        };

        // Create one tantivy index per indexed property.
        for prop in &def.properties {
            // Per-property subdirectory: text_idx_{name}_{prop}
            let idx_dir = self.base_dir.join(format!("text_idx_{}_{prop}", def.name));
            if let Err(e) = std::fs::create_dir_all(&idx_dir) {
                return Err(format!("failed to create text index directory: {e}"));
            }

            // Build per-property MultiLangConfig using the field's specific analyzer.
            let per_field_config = if let Some(fc) = config.fields.get(prop) {
                MultiLangConfig {
                    default_language: fc.analyzer.clone(),
                    field_analyzers: HashMap::new(),
                    language_override_property: config.language_override_property.clone(),
                }
            } else {
                Self::to_multi_lang_config(config)
            };

            let text_index =
                MultiLanguageTextIndex::open_or_create(&idx_dir, 15_000_000, per_field_config)
                    .map_err(|e| format!("failed to create text index for {prop}: {e}"))?;

            let key = (def.label.clone(), prop.clone());
            if let Ok(mut indexes) = self.indexes.write() {
                indexes.insert(key, Arc::new(RwLock::new(text_index)));
            }
        }

        // Store definition under the first property (canonical key for metadata lookup).
        let canonical_key = (def.label.clone(), def.property().to_string());
        if let Ok(mut defs) = self.definitions.write() {
            defs.insert(canonical_key, def);
        }
        Ok(())
    }

    /// Unregister a text index by label and any one of its properties.
    ///
    /// For multi-field indexes, removes all (label, property) entries.
    pub fn unregister(&self, label: &str, property: &str) {
        let canonical_key = (label.to_string(), property.to_string());

        // Find the definition to get all properties.
        let all_properties: Vec<String> = self
            .definitions
            .read()
            .ok()
            .and_then(|defs| defs.get(&canonical_key).map(|d| d.properties.clone()))
            .unwrap_or_else(|| vec![property.to_string()]);

        if let Ok(mut indexes) = self.indexes.write() {
            for prop in &all_properties {
                indexes.remove(&(label.to_string(), prop.clone()));
            }
        }
        if let Ok(mut defs) = self.definitions.write() {
            defs.remove(&canonical_key);
        }
    }

    /// Get a handle to the text index for a (label, property) pair.
    pub fn get(&self, label: &str, property: &str) -> Option<TextHandle> {
        let indexes = self.indexes.read().ok()?;
        indexes
            .get(&(label.to_string(), property.to_string()))
            .cloned()
    }

    /// Check if a text index exists for a (label, property).
    pub fn has_index(&self, label: &str, property: &str) -> bool {
        self.indexes
            .read()
            .map(|m| m.contains_key(&(label.to_string(), property.to_string())))
            .unwrap_or(false)
    }

    /// Number of registered text indexes.
    pub fn len(&self) -> usize {
        self.indexes.read().map(|m| m.len()).unwrap_or(0)
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collect all registered index definitions (snapshot).
    pub fn definitions(&self) -> Vec<IndexDefinition> {
        self.definitions
            .read()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Index or update a text document for applicable text indexes on a label.
    ///
    /// Uses `MultiLanguageTextIndex::add_node()` with a single-property map
    /// so that per-field language resolution is applied (explicit analyzer,
    /// per-node override, auto-detect, or default). This ensures tokens
    /// match the query-time tokenization in `search()`/`search_with_language()`.
    pub fn on_text_written(&self, label: &str, node_id: NodeId, property: &str, text: &str) {
        if let Some(handle) = self.get(label, property) {
            if let Ok(mut idx) = handle.write() {
                let mut props = HashMap::new();
                props.insert(property.to_string(), text.to_string());
                if let Err(e) = idx.add_node(node_id.as_raw(), &props) {
                    tracing::warn!(
                        label,
                        property,
                        node_id = node_id.as_raw(),
                        "failed to index text: {e}"
                    );
                }
            }
        }
    }

    /// Remove a document from applicable text indexes.
    pub fn on_text_deleted(&self, label: &str, node_id: NodeId, property: &str) {
        if let Some(handle) = self.get(label, property) {
            if let Ok(mut idx) = handle.write() {
                if let Err(e) = idx.delete_document(node_id.as_raw()) {
                    tracing::warn!(
                        label,
                        property,
                        node_id = node_id.as_raw(),
                        "failed to delete text from index: {e}"
                    );
                }
            }
        }
    }

    /// Search the text index for a (label, property) pair.
    pub fn search(
        &self,
        label: &str,
        property: &str,
        query: &str,
        limit: usize,
    ) -> Option<Vec<TextSearchResult>> {
        let handle = self.get(label, property)?;
        let idx = handle.read().ok()?;
        idx.search(query, limit).ok()
    }

    /// Get the base directory.
    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }
}

impl Default for TextIndexRegistry {
    fn default() -> Self {
        Self::new(std::env::temp_dir().join("coordinode_text_indexes"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
