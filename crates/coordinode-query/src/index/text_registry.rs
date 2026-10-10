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

use coordinode_core::budget::{BudgetStop, Meter};
use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeId, NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_search::tantivy::multi_lang::{
    MultiLangConfig, MultiLanguageTextIndex, TextRequest,
};
use coordinode_search::tantivy::pending::Matches;
use coordinode_search::tantivy::validity::Validity;
use coordinode_search::tantivy::{HighlightedResult, TextSearchError, TextSearchResult};
use coordinode_storage::engine::transaction::Transaction;

use super::coverage::{IndexCoverage, IndexDelta};
use super::definition::{IndexDefinition, TextIndexConfig};
use crate::executor::temporal_read::{TimelineFields, state_span};

/// Key for text index lookup: (label, property).
type TextIndexKey = (String, String);

/// Why a search of a text index ([`TextIndexRegistry::find`]) did not answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FindError {
    /// The query's budget stopped it.
    Budget(BudgetStop),
    /// The nodes could not be read or the query not run, as described.
    Failed(String),
}

impl core::fmt::Display for FindError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Budget(stop) => stop.fmt(f),
            Self::Failed(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for FindError {}

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

/// The stored rows of one node, as a text index reads them.
#[derive(Debug, Clone)]
pub enum NodeRows {
    /// A node without a timeline: its record, if it exists.
    Plain(Option<NodeRecord>),
    /// A temporal node's versions in ascending `valid_from` order.
    Timeline(Vec<(i64, NodeRecord)>),
}

/// What a text index holds for one node at one valid-time instant.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeText {
    /// The node.
    pub node_id: NodeId,
    /// The indexed text, when the node has it at the instant.
    pub text: Option<String>,
    /// The valid time over which the node keeps that text, or its lack.
    pub validity: Validity,
}

/// How a text index reads one property of one label off a node's rows.
#[derive(Debug, Clone, Copy)]
pub struct TextSource<'a> {
    label: &'a str,
    field_id: Option<u32>,
    timeline: TimelineFields,
}

impl<'a> TextSource<'a> {
    /// The text of `property` on `label`'s nodes, by the field ids of
    /// `interner`.
    pub fn new(interner: &FieldInterner, label: &'a str, property: &str) -> Self {
        Self {
            label,
            field_id: interner.lookup(property),
            timeline: TimelineFields {
                valid_to: interner.lookup("valid_to"),
                deleted: interner.lookup("__deleted__"),
            },
        }
    }

    fn text(&self, record: &NodeRecord) -> Option<String> {
        if record.primary_label() != self.label {
            return None;
        }
        record
            .props
            .get(&self.field_id?)?
            .as_str()
            .map(str::to_string)
    }

    /// What the index holds for `node_id`, stored as `rows`, at valid-time
    /// instant `at`: a temporal node's state valid then, held over the
    /// interval its timeline keeps that state.
    pub fn at(&self, node_id: NodeId, rows: &NodeRows, at: i64) -> NodeText {
        match rows {
            NodeRows::Plain(record) => NodeText {
                node_id,
                text: record.as_ref().and_then(|r| self.text(r)),
                validity: Validity::ALWAYS,
            },
            NodeRows::Timeline(versions) => {
                let span = state_span(
                    versions.iter().map(|(from, record)| (*from, record)),
                    at,
                    self.timeline,
                );
                // A timeline of another label never enters this index.
                let ours = versions
                    .first()
                    .is_some_and(|(_, record)| record.primary_label() == self.label);
                NodeText {
                    node_id,
                    text: span.state.positive().and_then(|(_, r)| self.text(r)),
                    validity: if ours {
                        Validity {
                            from: span.from,
                            until: span.until,
                        }
                    } else {
                        Validity::ALWAYS
                    },
                }
            }
        }
    }
}

/// The rows of `ids` (sorted) as `read` sees them, its own writes included.
///
/// # Errors
///
/// The store could not be read or a row decoded.
pub fn read_nodes(
    read: &Transaction<'_>,
    shard_id: u16,
    ids: &[NodeId],
) -> Result<Vec<(NodeId, NodeRows)>, String> {
    let records = LocalNodeStore
        .get_many(read, shard_id, ids)
        .map_err(|e| format!("read {} written nodes: {e}", ids.len()))?;
    let mut out = Vec::with_capacity(ids.len());
    for (id, record) in ids.iter().copied().zip(records) {
        if record.is_some() {
            out.push((id, NodeRows::Plain(record)));
            continue;
        }
        // No row of its own: a temporal node keeps only its versions.
        let versions = LocalNodeStore
            .versions(read, shard_id, id)
            .map_err(|e| format!("read the versions of node {}: {e}", id.as_raw()))?;
        out.push((
            id,
            if versions.is_empty() {
                NodeRows::Plain(None)
            } else {
                NodeRows::Timeline(versions)
            },
        ));
    }
    Ok(out)
}

/// The visitor of [`NodeGrouper`]: a node and its rows.
type NodeVisit<'v> = dyn FnMut(NodeId, NodeRows) -> Result<(), String> + 'v;

/// Turns the node rows of a shard, in key order, into nodes with their rows:
/// a temporal node's versions sort together under its id, so they are
/// gathered and handed over whole when the next node's row comes.
#[derive(Default)]
struct NodeGrouper {
    timeline: Option<(NodeId, Vec<(i64, NodeRecord)>)>,
}

impl NodeGrouper {
    fn push(&mut self, key: &[u8], value: &[u8], visit: &mut NodeVisit<'_>) -> Result<(), String> {
        let decode = |value: &[u8]| {
            NodeRecord::from_msgpack(value).map_err(|e| format!("decode a node record: {e}"))
        };
        if let Some((_, id, valid_from)) = decode_temporal_node_key(key) {
            let record = decode(value)?;
            match &mut self.timeline {
                Some((of, versions)) if *of == id => versions.push((valid_from, record)),
                _ => {
                    if let Some((of, versions)) =
                        self.timeline.replace((id, vec![(valid_from, record)]))
                    {
                        visit(of, NodeRows::Timeline(versions))?;
                    }
                }
            }
            return Ok(());
        }
        self.flush(visit)?;
        if let Some((_, id)) = decode_node_key(key) {
            visit(id, NodeRows::Plain(Some(decode(value)?)))?;
        }
        Ok(())
    }

    fn flush(&mut self, visit: &mut NodeVisit<'_>) -> Result<(), String> {
        match self.timeline.take() {
            Some((of, versions)) => visit(of, NodeRows::Timeline(versions)),
            None => Ok(()),
        }
    }
}

/// Every node of `shard_id` with its rows, in id order, as `read` sees them.
fn for_each_node_read(
    read: &Transaction<'_>,
    shard_id: u16,
    visit: &mut NodeVisit<'_>,
) -> Result<(), String> {
    let rows = LocalNodeStore
        .shard_rows(read, shard_id)
        .map_err(|e| format!("scan the node rows: {e}"))?;
    let mut grouper = NodeGrouper::default();
    for (key, value) in &rows {
        grouper.push(key, value, visit)?;
    }
    grouper.flush(visit)
}

/// Every node of `shard_id` with its rows, in id order, as `engine` holds
/// them now, streamed.
fn for_each_node_latest(
    engine: &coordinode_storage::engine::core::StorageEngine,
    shard_id: u16,
    visit: &mut NodeVisit<'_>,
) -> Result<(), String> {
    let mut grouper = NodeGrouper::default();
    let mut failure = None;
    LocalNodeStore
        .for_each_row_in_shard(engine, shard_id, &mut |key, value| {
            Ok(match grouper.push(key, value, visit) {
                Ok(()) => std::ops::ControlFlow::Continue(()),
                Err(e) => {
                    failure = Some(e);
                    std::ops::ControlFlow::Break(())
                }
            })
        })
        .map_err(|e| format!("scan the node rows: {e}"))?;
    if let Some(e) = failure {
        return Err(e);
    }
    grouper.flush(visit)
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

    /// Search the index of `(label, property)` on `shard_id` as `read` sees
    /// the store at valid-time instant `at`: the index, and in place of its
    /// documents of the nodes written since its position, of `also` (the
    /// reading transaction's uncommitted writes, or the nodes written after a
    /// named read timestamp) and of the nodes whose held state does not hold
    /// at `at`, their documents as `read` sees them at `at`. `Ok(None)` when
    /// there is no such index.
    ///
    /// The index stays read-locked from choosing those nodes to the end of
    /// the search, so the documents and statistics it answers with are the
    /// ones the choice was made against. The nodes read and the documents
    /// scored are reported to `meter`, which can stop the search.
    ///
    /// # Errors
    ///
    /// [`FindError::Budget`] when `meter` stopped it, [`FindError::Failed`]
    /// when the nodes could not be read or the query not run.
    #[allow(clippy::too_many_arguments)]
    pub fn find<M: Meter>(
        &self,
        label: &str,
        property: &str,
        read: &Transaction<'_>,
        shard_id: u16,
        interner: &FieldInterner,
        also: IndexDelta,
        at: i64,
        request: TextRequest<'_>,
        matches: Matches,
        meter: &mut M,
    ) -> Result<Option<Vec<HighlightedResult>>, FindError>
    where
        TextSearchError: From<M::Stop>,
    {
        let Some(handle) = self.get(label, property) else {
            return Ok(None);
        };
        let failed = |e: &dyn std::fmt::Display| {
            FindError::Failed(format!("text index :{label}({property}): {e}"))
        };
        let searched = |e: TextSearchError| match e {
            TextSearchError::Budget(stop) => FindError::Budget(stop),
            other => failed(&other),
        };
        let index = handle.read().map_err(|_| failed(&"lock poisoned"))?;
        let delta = match self.coverage() {
            Some(coverage) => coverage.delta(shard_id),
            None => IndexDelta::Nodes(Default::default()),
        }
        .union(also)
        .with_nodes(index.outside(at).into_iter().map(NodeId::from_raw));
        let source = TextSource::new(interner, label, property);
        let mut texts: Vec<(NodeId, String)> = Vec::new();
        let superseded = match &delta {
            IndexDelta::Nodes(nodes) if nodes.is_empty() => Some(Vec::new()),
            IndexDelta::Nodes(nodes) => {
                let mut ids: Vec<NodeId> = nodes.iter().copied().collect();
                ids.sort_unstable();
                meter
                    .work(ids.len() as u64)
                    .map_err(|stop| searched(stop.into()))?;
                for (id, rows) in read_nodes(read, shard_id, &ids).map_err(|e| failed(&e))? {
                    if let Some(text) = source.at(id, &rows, at).text {
                        meter
                            .scratch(text.len() as u64)
                            .map_err(|stop| searched(stop.into()))?;
                        texts.push((id, text));
                    }
                }
                Some(ids.iter().map(|id| id.as_raw()).collect::<Vec<u64>>())
            }
            IndexDelta::Unknown => {
                // Every node of the shard is read; a stop ends the walk and
                // is reported below.
                let mut stopped = None;
                for_each_node_read(read, shard_id, &mut |id, rows| {
                    if let Err(stop) = meter.work(1) {
                        stopped = Some(searched(stop.into()));
                        return Err("stopped by the query budget".to_string());
                    }
                    if let Some(text) = source.at(id, &rows, at).text {
                        if let Err(stop) = meter.scratch(text.len() as u64) {
                            stopped = Some(searched(stop.into()));
                            return Err("stopped by the query budget".to_string());
                        }
                        texts.push((id, text));
                    }
                    Ok(())
                })
                .map_err(|e| stopped.take().unwrap_or_else(|| failed(&e)))?;
                None
            }
        };
        let documents = single_property_documents(property, &texts);
        let pending = index
            .pending(superseded.as_deref(), &documents)
            .map_err(searched)?;
        index
            .find(request, matches, &pending, meter)
            .map(Some)
            .map_err(searched)
    }

    /// The earliest instant at which some held state of a temporal node
    /// stops holding, over every index: the next time a fold is due without a
    /// write.
    pub fn first_end(&self) -> Option<i64> {
        let indexes = self.indexes.read().ok()?;
        indexes
            .values()
            .filter_map(|handle| handle.read().ok()?.first_end())
            .min()
    }

    /// The nodes whose held state in the index of `(label, property)` does
    /// not hold at `at`.
    pub fn outside(&self, label: &str, property: &str, at: i64) -> Vec<NodeId> {
        self.get(label, property)
            .and_then(|handle| Some(handle.read().ok()?.outside(at)))
            .unwrap_or_default()
            .into_iter()
            .map(NodeId::from_raw)
            .collect()
    }

    /// Apply one batch of changes to the index of `(label, property)` in one
    /// commit: a node with text has its document replaced, a node without is
    /// taken out, and each node's validity is recorded with its document, so
    /// a reader sees both or neither.
    pub fn apply_changes(
        &self,
        label: &str,
        property: &str,
        changes: &[NodeText],
    ) -> Result<(), String> {
        let Some(handle) = self.get(label, property) else {
            return Ok(());
        };
        let (upserts, removals) = split_changes(property, changes);
        let mut idx = handle
            .write()
            .map_err(|_| format!("text index :{label}({property}) lock poisoned"))?;
        idx.apply_changes(&upserts, &removals)
            .map_err(|e| format!("text index :{label}({property}): {e}"))?;
        for change in changes {
            idx.set_validity(change.node_id.as_raw(), change.validity);
        }
        Ok(())
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
        scan: impl FnOnce() -> Result<Vec<NodeText>, String>,
    ) -> Result<usize, String> {
        let Some(handle) = self.get(label, property) else {
            return Ok(0);
        };
        let mut idx = handle
            .write()
            .map_err(|_| format!("text index :{label}({property}) lock poisoned"))?;
        let scanned = scan()?;
        let (documents, _) = split_changes(property, &scanned);
        idx.replace_all(&documents)
            .map_err(|e| format!("text index :{label}({property}): {e}"))?;
        idx.clear_validities();
        for node in &scanned {
            idx.set_validity(node.node_id.as_raw(), node.validity);
        }
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

/// What a text index of `(label, property)` holds when it covers the store
/// as it stands now, at valid-time instant `at`: every node of `label` on
/// `shard_id` with its text, and every temporal node of `label` with the
/// interval its state at `at` holds over, text or not.
///
/// # Errors
///
/// The scan of the node partition failed or a row did not decode.
pub fn stored_texts(
    engine: &coordinode_storage::engine::core::StorageEngine,
    shard_id: u16,
    interner: &FieldInterner,
    label: &str,
    property: &str,
    at: i64,
) -> Result<Vec<NodeText>, String> {
    let source = TextSource::new(interner, label, property);
    let mut texts = Vec::new();
    for_each_node_latest(engine, shard_id, &mut |id, rows| {
        let held = source.at(id, &rows, at);
        if held.text.is_some() || held.validity != Validity::ALWAYS {
            texts.push(held);
        }
        Ok(())
    })?;
    Ok(texts)
}

/// Documents to hold, by node, and the nodes to take out.
type DocumentChanges = (Vec<(u64, HashMap<String, String>)>, Vec<u64>);

/// `changes` as the documents the index takes for the nodes with text and
/// the nodes to take out.
fn split_changes(property: &str, changes: &[NodeText]) -> DocumentChanges {
    let mut upserts = Vec::new();
    let mut removals = Vec::new();
    for change in changes {
        match &change.text {
            Some(text) => {
                let mut props = HashMap::with_capacity(1);
                props.insert(property.to_string(), text.clone());
                upserts.push((change.node_id.as_raw(), props));
            }
            None => removals.push(change.node_id.as_raw()),
        }
    }
    (upserts, removals)
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
