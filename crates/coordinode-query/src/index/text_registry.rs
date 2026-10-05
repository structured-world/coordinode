//! Text index registry: tracks active tantivy text indexes for full-text search.
//!
//! Holds in-memory tantivy index instances keyed by (label, property).
//! Indexes are built from stored node text and maintained from the entries
//! applied to the store, never from a statement before it commits; a search
//! reads each index together with the writes it has not folded yet (see
//! [`IndexCoverage`]).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord};
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_search::tantivy::multi_lang::{
    MultiLangConfig, MultiLanguageTextIndex, TextRequest,
};
use coordinode_search::tantivy::pending::{Matches, PendingDocuments};
use coordinode_search::tantivy::{HighlightedResult, TextSearchResult};
use coordinode_storage::engine::transaction::Transaction;

use super::coverage::{IndexCoverage, IndexDelta};
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
    /// The writes the indexes have not folded, when a worker maintains them
    /// from the applied entries; a registry without one serves what it holds.
    coverage: RwLock<Option<Arc<IndexCoverage>>>,
}

/// One search's reading of a text index: the index, and the current
/// documents of the nodes it has not caught up with in place of its own.
pub struct TextView {
    handle: TextHandle,
    pending: PendingDocuments,
}

impl TextView {
    /// Run `request`, best score first, keeping `matches`.
    ///
    /// # Errors
    ///
    /// The query could not be parsed or run.
    pub fn find(
        &self,
        request: TextRequest<'_>,
        matches: Matches,
    ) -> Result<Vec<HighlightedResult>, String> {
        self.handle
            .read()
            .map_err(|_| "text index lock poisoned".to_string())?
            .find(request, matches, &self.pending)
            .map_err(|e| e.to_string())
    }
}

impl TextIndexRegistry {
    /// Create an empty registry with the given base directory for index storage.
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            indexes: RwLock::new(HashMap::new()),
            definitions: RwLock::new(HashMap::new()),
            base_dir: base_dir.into(),
            coverage: RwLock::new(None),
        }
    }

    /// Learn from `coverage` which writes the indexes have not folded, so a
    /// search answers for them itself.
    pub fn set_coverage(&self, coverage: Arc<IndexCoverage>) {
        if let Ok(mut slot) = self.coverage.write() {
            *slot = Some(coverage);
        }
    }

    /// Where searches learn the writes the indexes have not folded, if a
    /// worker maintains them.
    pub fn coverage(&self) -> Option<Arc<IndexCoverage>> {
        self.coverage.read().ok().and_then(|r| r.clone())
    }

    /// What a search of `(label, property)` on `shard_id` reads: the index,
    /// and in place of its documents of the nodes written since its position
    /// and of `also` (the reading transaction's uncommitted writes, or the
    /// nodes written after a named read timestamp), their documents as `read`
    /// sees them. `Ok(None)` when there is no such index.
    ///
    /// # Errors
    ///
    /// The nodes could not be read, or the pending documents not built.
    pub fn view(
        &self,
        label: &str,
        property: &str,
        read: &Transaction<'_>,
        shard_id: u16,
        interner: &FieldInterner,
        also: IndexDelta,
    ) -> Result<Option<TextView>, String> {
        let Some(handle) = self.get(label, property) else {
            return Ok(None);
        };
        let delta = match self.coverage() {
            Some(coverage) => coverage.delta(shard_id),
            None => IndexDelta::Nodes(Default::default()),
        }
        .union(also);
        if delta.is_empty() {
            return Ok(Some(TextView {
                handle,
                pending: PendingDocuments::none(),
            }));
        }
        let field_id = interner.lookup(property);
        // The text a node holds for this index, as the worker reads it.
        let text = |record: &NodeRecord| -> Option<String> {
            (record.primary_label() == label)
                .then(|| record.props.get(&field_id?)?.as_str().map(str::to_string))
                .flatten()
        };
        let (superseded, texts): (Option<Vec<u64>>, Vec<(NodeId, String)>) = match &delta {
            IndexDelta::Nodes(nodes) => {
                let mut ids: Vec<NodeId> = nodes.iter().copied().collect();
                ids.sort_unstable();
                let records = LocalNodeStore
                    .get_many(read, shard_id, &ids)
                    .map_err(|e| format!("read {} written nodes: {e}", ids.len()))?;
                let texts = ids
                    .iter()
                    .zip(&records)
                    .filter_map(|(id, record)| Some((*id, text(record.as_ref()?)?)))
                    .collect();
                (Some(ids.iter().map(|id| id.as_raw()).collect()), texts)
            }
            IndexDelta::Unknown => {
                let mut texts = Vec::new();
                LocalNodeStore
                    .for_each_in_shard(read, shard_id, &mut |id, record| {
                        if let Some(text) = text(&record) {
                            texts.push((id, text));
                        }
                        Ok(())
                    })
                    .map_err(|e| format!("scan the nodes of :{label}: {e}"))?;
                (None, texts)
            }
        };
        let documents = single_property_documents(property, &texts);
        let pending = handle
            .read()
            .map_err(|_| format!("text index :{label}({property}) lock poisoned"))?
            .pending(superseded.as_deref(), &documents)
            .map_err(|e| format!("text index :{label}({property}): {e}"))?;
        Ok(Some(TextView { handle, pending }))
    }

    /// Apply one batch of changes to the index of `(label, property)` in one
    /// commit: `upserts` replace a node's text, `removals` take a node out.
    pub fn apply_changes(
        &self,
        label: &str,
        property: &str,
        upserts: &[(NodeId, String)],
        removals: &[NodeId],
    ) -> Result<(), String> {
        let Some(handle) = self.get(label, property) else {
            return Ok(());
        };
        let upserts = single_property_documents(property, upserts);
        let removals: Vec<u64> = removals.iter().map(|id| id.as_raw()).collect();
        let mut idx = handle
            .write()
            .map_err(|_| format!("text index :{label}({property}) lock poisoned"))?;
        idx.apply_changes(&upserts, &removals)
            .map_err(|e| format!("text index :{label}({property}): {e}"))
    }

    /// Rebuild the index of `(label, property)` from `scan`, the nodes'
    /// texts as the store holds them; how many documents it then holds.
    ///
    /// The index stays locked from before the scan until the rebuilt
    /// contents are committed, so a change folded in meanwhile lands after
    /// the rebuild rather than under it: the scan reads every commit applied
    /// before it started, and the worker folds every one after.
    pub fn rebuild_index(
        &self,
        label: &str,
        property: &str,
        scan: impl FnOnce() -> Result<Vec<(NodeId, String)>, String>,
    ) -> Result<usize, String> {
        let Some(handle) = self.get(label, property) else {
            return Ok(0);
        };
        let mut idx = handle
            .write()
            .map_err(|_| format!("text index :{label}({property}) lock poisoned"))?;
        let documents = single_property_documents(property, &scan()?);
        idx.replace_all(&documents)
            .map_err(|e| format!("text index :{label}({property}): {e}"))?;
        Ok(documents.len())
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
                "register called with non-text IndexDefinition: {def}"
            ));
        };

        // Create one tantivy index per indexed property.
        for prop in &def.properties {
            // Per-property subdirectory of the generation:
            // text_idx_{generation}_{prop}. A rebuild into a new generation
            // gets a directory of its own.
            let idx_dir = self
                .base_dir
                .join(format!("text_idx_{}_{prop}", def.generation.as_raw()));
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

            // Every registration is followed by a rebuild from the store (on
            // open, on CREATE TEXT INDEX, after a restore), so the index starts
            // empty and its files need no durability: a crash loses nothing
            // the next rebuild does not restore.
            let text_index = match MultiLanguageTextIndex::create_scratch(
                &idx_dir,
                15_000_000,
                per_field_config.clone(),
            ) {
                Ok(index) => index,
                // An index replaced in this process may still have its files
                // mapped by a reader finishing a search, and Windows refuses
                // to remove a mapped file: the new one starts in a fresh
                // directory beside it instead.
                Err(first) => {
                    let nanos = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or_default();
                    let fresh = idx_dir.with_extension(nanos.to_string());
                    MultiLanguageTextIndex::create_scratch(&fresh, 15_000_000, per_field_config)
                        .map_err(|e| {
                            format!("failed to create text index for {prop}: {first}; {e}")
                        })?
                }
            };

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

/// The text of `property` on every node of `label` stored on `shard_id`, as
/// the store holds it now: what a text index of `(label, property)` holds
/// when it covers the store.
///
/// # Errors
///
/// The scan of the node partition failed.
pub fn stored_texts(
    engine: &coordinode_storage::engine::core::StorageEngine,
    shard_id: u16,
    interner: &coordinode_core::graph::intern::FieldInterner,
    label: &str,
    property: &str,
) -> Result<Vec<(NodeId, String)>, String> {
    use coordinode_modality::{LocalNodeStore, NodeStore as _};

    // No binding: no stored node carries the property.
    let Some(field_id) = interner.lookup(property) else {
        return Ok(Vec::new());
    };
    let mut texts = Vec::new();
    LocalNodeStore
        .for_each_in_shard_at_snapshot(engine, None, shard_id, &mut |node_id, _key, record| {
            if record.primary_label() == label {
                if let Some(text) = record.props.get(&field_id).and_then(|v| v.as_str()) {
                    texts.push((node_id, text.to_string()));
                }
            }
            Ok(std::ops::ControlFlow::Continue(()))
        })
        .map_err(|e| format!("scan the nodes of :{label} for its text index: {e}"))?;
    Ok(texts)
}

/// `(node, text)` pairs as the per-node property maps the index takes, so
/// each is tokenized by the property's analyzer.
fn single_property_documents(
    property: &str,
    texts: &[(NodeId, String)],
) -> Vec<(u64, HashMap<String, String>)> {
    texts
        .iter()
        .map(|(id, text)| {
            let mut props = HashMap::with_capacity(1);
            props.insert(property.to_string(), text.clone());
            (id.as_raw(), props)
        })
        .collect()
}

impl Default for TextIndexRegistry {
    fn default() -> Self {
        Self::new(std::env::temp_dir().join("coordinode_text_indexes"))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
