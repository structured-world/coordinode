//! Multi-language text index with per-document language resolution.
//!
//! `MultiLanguageTextIndex` wraps [`TextIndex`] and provides the 4-level
//! language cascade:
//!
//! 1. **Explicit per-field analyzer** (from index config) — highest priority
//! 2. **Per-node `_language` property** (opt-in override) — mid priority
//! 3. **Auto-detection** (whatlang-rs trigram) — fallback
//! 4. **Index-level default** — lowest priority
//!
//! This enables mixed-language documents in a single tantivy index,
//! where each field value is tokenized with the appropriate language pipeline.

use std::collections::HashMap;
use std::path::Path;

use std::collections::HashSet;

use super::corpus::Corpus;
use super::pending::{Matches, PendingDocuments};
use super::tokenize;
use super::{HighlightedResult, TextIndex, TextSearchError, TextSearchResult};

/// What one write has done so far, so a node it touches twice is counted
/// once and its committed document is read once.
struct Batch {
    /// The committed state the write started from.
    searcher: tantivy::Searcher,
    /// Tokens of the documents this write staged, by node.
    staged: HashMap<u64, Vec<String>>,
    /// Nodes whose committed document this write already removed.
    forgotten: HashSet<u64>,
    /// Every committed document was removed.
    cleared: bool,
}

impl Batch {
    fn new(searcher: tantivy::Searcher) -> Self {
        Self {
            searcher,
            staged: HashMap::new(),
            forgotten: HashSet::new(),
            cleared: false,
        }
    }
}

/// What a search looks for.
#[derive(Debug, Clone, Copy)]
pub enum TextRequest<'a> {
    /// The terms of `query` tokenized by `language` (the index's default
    /// when `None`), any of them matching, `word*` as a prefix; with
    /// highlighted snippets when `snippets` is set.
    Terms {
        query: &'a str,
        language: Option<&'a str>,
        snippets: bool,
    },
    /// `query` in the query syntax with each term widened to its edit-1
    /// neighbourhood, over the index's stored tokens.
    Fuzzy { query: &'a str, snippets: bool },
}

/// Configuration for a multi-language text index.
#[derive(Debug, Clone)]
pub struct MultiLangConfig {
    /// Default language for the index (lowest-priority fallback).
    /// Used when no other resolution level determines the language.
    pub default_language: String,

    /// Per-field explicit analyzer overrides.
    /// Keys are field names, values are language/analyzer names.
    ///
    /// Example: `{"title_en": "english", "title_ru": "russian", "body": "auto_detect"}`
    pub field_analyzers: HashMap<String, String>,

    /// Name of the node property that overrides the index default.
    /// If a node has this property set (e.g., `_language: "russian"`),
    /// all `auto_detect` fields use that language instead of auto-detection.
    ///
    /// Default: `"_language"`
    pub language_override_property: String,
}

impl Default for MultiLangConfig {
    fn default() -> Self {
        Self {
            default_language: "english".to_string(),
            field_analyzers: HashMap::new(),
            language_override_property: "_language".to_string(),
        }
    }
}

impl MultiLangConfig {
    /// Create a config with a specific default language.
    pub fn with_default_language(language: impl Into<String>) -> Self {
        Self {
            default_language: language.into(),
            ..Default::default()
        }
    }

    /// Add a per-field explicit analyzer.
    pub fn with_field_analyzer(
        mut self,
        field_name: impl Into<String>,
        analyzer: impl Into<String>,
    ) -> Self {
        self.field_analyzers
            .insert(field_name.into(), analyzer.into());
        self
    }

    /// Set the language override property name.
    pub fn with_override_property(mut self, property: impl Into<String>) -> Self {
        self.language_override_property = property.into();
        self
    }
}

/// A multi-language text index supporting per-document and per-field
/// language resolution.
///
/// Wraps a single tantivy `TextIndex` where documents may use different
/// tokenizers via `PreTokenizedString`. The language for each field value
/// is resolved through a 4-level cascade.
pub struct MultiLanguageTextIndex {
    inner: TextIndex,
    config: MultiLangConfig,
}

impl MultiLanguageTextIndex {
    /// Open or create a multi-language text index.
    ///
    /// The index uses the config's `default_language` for queries through
    /// the standard `search()` method. Per-document language selection
    /// happens at write time via `add_node()`.
    pub fn open_or_create(
        dir: &Path,
        heap_size_bytes: usize,
        config: MultiLangConfig,
    ) -> Result<Self, TextSearchError> {
        // Use "none" as the schema-level tokenizer — actual tokenization
        // is done per-document via PreTokenizedString. The schema tokenizer
        // is only used as fallback for queries without explicit language.
        let inner = TextIndex::open_or_create(dir, heap_size_bytes, Some("none"))?;
        Self::counted(inner, config)
    }

    /// `inner` with its live documents counted.
    fn counted(mut inner: TextIndex, config: MultiLangConfig) -> Result<Self, TextSearchError> {
        inner.corpus = Some(inner.recount()?);
        Ok(Self { inner, config })
    }

    /// Create an empty index at `dir`, replacing whatever it held, with files
    /// written unsynced (see [`TextIndex::create_scratch`]): for an index the
    /// caller rebuilds from its source every time it creates it.
    pub fn create_scratch(
        dir: &Path,
        heap_size_bytes: usize,
        config: MultiLangConfig,
    ) -> Result<Self, TextSearchError> {
        // Tokenization happens per document via PreTokenizedString, as in
        // `open_or_create`.
        let mut inner = TextIndex::create_scratch(dir, heap_size_bytes, Some("none"))?;
        inner.corpus = Some(Corpus::default());
        Ok(Self { inner, config })
    }

    /// Wrap an existing `TextIndex` with multi-language support.
    ///
    /// Useful for tests and migration from single-language to multi-language.
    /// The wrapped index uses the config's `default_language` for queries.
    /// Its documents were written without their tokens, so it scores with
    /// Tantivy's own statistics.
    pub fn wrap(inner: TextIndex, config: MultiLangConfig) -> Self {
        Self { inner, config }
    }

    /// Add a node's text fields to the index.
    ///
    /// `properties` is a map of field_name → text_value for all text fields
    /// that should be indexed. The language for each field is resolved via
    /// the 4-level cascade.
    ///
    /// Only fields listed in the config's `field_analyzers` (or all fields
    /// if `field_analyzers` is empty) are indexed.
    pub fn add_node(
        &mut self,
        node_id: u64,
        properties: &HashMap<String, String>,
    ) -> Result<(), TextSearchError> {
        self.write(|index, batch| index.stage_node(batch, node_id, properties))
    }

    /// Add multiple nodes in a single batch commit.
    pub fn add_nodes_batch(
        &mut self,
        nodes: &[(u64, HashMap<String, String>)],
    ) -> Result<(), TextSearchError> {
        self.write(|index, batch| {
            let mut changed = false;
            for (node_id, properties) in nodes {
                changed |= index.stage_node(batch, *node_id, properties)?;
            }
            Ok(changed)
        })
    }

    /// Replace the documents of `upserts` and remove those of `removals`, in
    /// one commit: a reader sees all of the changes or none. A node listed in
    /// both ends up as its upsert. A removal of a node the index does not
    /// hold costs no commit, so writes outside the index leave it untouched.
    pub fn apply_changes(
        &mut self,
        upserts: &[(u64, HashMap<String, String>)],
        removals: &[u64],
    ) -> Result<(), TextSearchError> {
        self.write(|index, batch| {
            let mut changed = false;
            for node_id in removals {
                changed |= index.forget(batch, *node_id)?;
            }
            for (node_id, properties) in upserts {
                // A node whose text tokenizes to nothing is left out, as
                // removed.
                changed |= index.stage_node(batch, *node_id, properties)?;
            }
            Ok(changed)
        })
    }

    /// Run `work`, which stages changes and says whether it staged any, and
    /// commit them. A failure discards everything staged and counts the
    /// committed documents again, so the statistics never describe documents
    /// a reader cannot see.
    fn write(
        &mut self,
        work: impl FnOnce(&mut Self, &mut Batch) -> Result<bool, TextSearchError>,
    ) -> Result<(), TextSearchError> {
        let mut batch = Batch::new(self.inner.reader.searcher());
        let outcome = match work(self, &mut batch) {
            Ok(true) => self.publish(),
            Ok(false) => Ok(()),
            Err(e) => Err(e),
        };
        if outcome.is_err() {
            self.inner.writer.rollback()?;
            if self.inner.corpus.is_some() {
                self.inner.corpus = Some(self.inner.recount()?);
            }
        }
        outcome
    }

    /// Stage the removal of `node_id`'s document; whether there was one.
    fn forget(&mut self, batch: &mut Batch, node_id: u64) -> Result<bool, TextSearchError> {
        let tokens = if let Some(staged) = batch.staged.remove(&node_id) {
            Some(Some(staged))
        } else if batch.cleared || !batch.forgotten.insert(node_id) {
            None
        } else {
            self.inner.live_tokens(&batch.searcher, node_id)?
        };
        let Some(tokens) = tokens else {
            return Ok(false);
        };
        if let (Some(corpus), Some(tokens)) = (self.inner.corpus.as_mut(), tokens) {
            corpus.remove(&tokens);
        }
        self.inner.writer.delete_term(tantivy::Term::from_field_u64(
            self.inner.node_id_field,
            node_id,
        ));
        Ok(true)
    }

    /// Whether the index holds a live document of `node_id`.
    pub fn contains(&self, node_id: u64) -> Result<bool, TextSearchError> {
        let term = tantivy::Term::from_field_u64(self.inner.node_id_field, node_id);
        let query = tantivy::query::TermQuery::new(term, tantivy::schema::IndexRecordOption::Basic);
        let live = self
            .inner
            .reader
            .searcher()
            .search(&query, &tantivy::collector::Count)?;
        Ok(live > 0)
    }

    /// Replace the whole index with `documents`, in one commit: a document
    /// of a node not listed is gone, and a reader sees the old set or the new
    /// one, never a mix.
    pub fn replace_all(
        &mut self,
        documents: &[(u64, HashMap<String, String>)],
    ) -> Result<(), TextSearchError> {
        self.write(|index, batch| {
            index.inner.writer.delete_all_documents()?;
            batch.cleared = true;
            batch.staged.clear();
            if index.inner.corpus.is_some() {
                index.inner.corpus = Some(Corpus::default());
            }
            for (node_id, properties) in documents {
                index.stage_node(batch, *node_id, properties)?;
            }
            Ok(true)
        })
    }

    /// Queue `node_id`'s document built from `properties` in place of any
    /// earlier one, or only the removal of that one when there is no text;
    /// whether anything changed. Nothing is visible to readers until
    /// [`Self::publish`].
    fn stage_node(
        &mut self,
        batch: &mut Batch,
        node_id: u64,
        properties: &HashMap<String, String>,
    ) -> Result<bool, TextSearchError> {
        let built = self.document(node_id, properties);
        let removed = self.forget(batch, node_id)?;
        let Some((doc, tokens)) = built else {
            return Ok(removed);
        };
        self.inner.writer.add_document(doc)?;
        if let Some(corpus) = self.inner.corpus.as_mut() {
            corpus.add(&tokens);
        }
        batch.staged.insert(node_id, tokens);
        Ok(true)
    }

    /// `node_id`'s document built from `properties`, each field tokenized by
    /// the language the cascade resolves, with its tokens; `None` when there
    /// is no text.
    fn document(
        &self,
        node_id: u64,
        properties: &HashMap<String, String>,
    ) -> Option<(tantivy::TantivyDocument, Vec<String>)> {
        // Extract the language override from the node properties (level 2)
        let node_language_override = properties
            .get(&self.config.language_override_property)
            .map(|s| s.as_str());

        // Concatenate all indexed fields into a single text,
        // resolving language per-field for tokenization
        let mut all_tokens = Vec::new();
        let mut all_text = String::new();
        let mut position_offset = 0;

        for (field_name, field_value) in properties {
            // Skip the language override property itself
            if field_name == &self.config.language_override_property {
                continue;
            }

            // Skip fields not in the analyzer config (if config is non-empty)
            if !self.config.field_analyzers.is_empty()
                && !self.config.field_analyzers.contains_key(field_name)
            {
                continue;
            }

            let language = self.resolve_language(field_name, field_value, node_language_override);

            let byte_offset = all_text.len();
            if !all_text.is_empty() {
                all_text.push(' ');
            }
            all_text.push_str(field_value);

            let mut tokens = tokenize::tokenize_text(field_value, &language);
            // Adjust offsets and positions for concatenated text
            let text_start = if byte_offset > 0 {
                byte_offset + 1 // account for space separator
            } else {
                byte_offset
            };
            for tok in &mut tokens {
                tok.offset_from += text_start;
                tok.offset_to += text_start;
                tok.position += position_offset;
            }
            position_offset += tokens.len();
            all_tokens.extend(tokens);
        }

        if all_tokens.is_empty() && all_text.is_empty() {
            return None;
        }
        let token_texts: Vec<String> = all_tokens.iter().map(|t| t.text.clone()).collect();

        let pretokenized = tantivy::tokenizer::PreTokenizedString {
            text: all_text,
            tokens: all_tokens,
        };

        let mut doc = tantivy::TantivyDocument::new();
        doc.add_field_value(
            self.inner.node_id_field,
            &tantivy::schema::OwnedValue::U64(node_id),
        );
        doc.add_field_value(
            self.inner.body_field,
            &tantivy::schema::OwnedValue::PreTokStr(pretokenized),
        );
        doc.add_field_value(
            self.inner.commit_ts_field,
            &tantivy::schema::OwnedValue::U64(0),
        );
        doc.add_field_value(
            self.inner.tokens_field,
            &tantivy::schema::OwnedValue::Bytes(super::encode_tokens(&token_texts)),
        );
        Some((doc, token_texts))
    }

    /// What a search reads in place of the index's own documents of the nodes
    /// written since the index's position: `superseded` lists those nodes
    /// (`None`: any node may have changed, so the index answers for none), and
    /// `documents` holds their current text, tokenized as the index would.
    pub fn pending(
        &self,
        superseded: Option<&[u64]>,
        documents: &[(u64, HashMap<String, String>)],
    ) -> Result<PendingDocuments, TextSearchError> {
        let documents = documents
            .iter()
            .filter_map(|(node_id, properties)| self.document(*node_id, properties))
            .collect();
        self.inner.pending(superseded, documents)
    }

    /// Run `request` over the index and `pending`, best score first.
    pub fn find(
        &self,
        request: TextRequest<'_>,
        matches: Matches,
        pending: &PendingDocuments,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        match request {
            TextRequest::Terms {
                query,
                language,
                snippets,
            } => {
                let language = language.unwrap_or(&self.config.default_language);
                match self.inner.language_query(query, language) {
                    Some(query) => self.inner.collect(&query, matches, pending, snippets),
                    None => Ok(Vec::new()),
                }
            }
            TextRequest::Fuzzy { query, snippets } => {
                let query = self.inner.build_query_fuzzy(query, None)?;
                self.inner.collect(&*query, matches, pending, snippets)
            }
        }
    }

    /// Commit the staged changes and make them visible to readers.
    fn publish(&mut self) -> Result<(), TextSearchError> {
        self.inner.writer.commit()?;
        self.inner.reader.reload()?;
        self.inner.reconcile_registry()?;
        Ok(())
    }

    /// Search using the index's default language for query tokenization.
    pub fn search(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        self.inner
            .search_with_language(query_str, limit, &self.config.default_language)
    }

    /// Search with HTML-highlighted snippets using the index's default language.
    ///
    /// Uses the same language-aware tokenization as `search()` so that stemming
    /// and stopword filtering are consistent between indexing and query time.
    /// This avoids the tokenizer mismatch that occurs when using
    /// `inner().search_with_highlights()` directly (which uses the schema-level
    /// `QueryParser` tokenizer, not the per-language pipeline).
    pub fn search_with_highlights(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        self.inner.search_with_highlights_and_language(
            query_str,
            limit,
            &self.config.default_language,
        )
    }

    /// Search using a specific language for query tokenization.
    pub fn search_with_language(
        &self,
        query_str: &str,
        limit: usize,
        language: &str,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        self.inner.search_with_language(query_str, limit, language)
    }

    /// Delete a document by node ID.
    pub fn delete_document(&mut self, node_id: u64) -> Result<(), TextSearchError> {
        self.write(|index, batch| index.forget(batch, node_id))
    }

    /// Number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        self.inner.num_docs()
    }

    /// Access the underlying `TextIndex`.
    pub fn inner(&self) -> &TextIndex {
        &self.inner
    }

    /// Mutable access to the underlying `TextIndex`.
    pub fn inner_mut(&mut self) -> &mut TextIndex {
        &mut self.inner
    }

    /// Access the configuration.
    pub fn config(&self) -> &MultiLangConfig {
        &self.config
    }

    /// Resolve language for a field value using the 4-level cascade.
    ///
    /// Resolution order (highest to lowest priority):
    /// 1. Explicit per-field analyzer from config
    /// 2. Per-node `_language` property override
    /// 3. Auto-detection via whatlang-rs (when field or override is "auto_detect")
    /// 4. Index-level default language
    fn resolve_language(
        &self,
        field_name: &str,
        field_value: &str,
        node_language_override: Option<&str>,
    ) -> String {
        // Level 1: Explicit per-field analyzer
        if let Some(field_lang) = self.config.field_analyzers.get(field_name) {
            if field_lang != "auto_detect" {
                return field_lang.clone();
            }
            // Field is set to "auto_detect" — fall through to level 2
        }

        // Level 2: Per-node language override
        if let Some(override_lang) = node_language_override {
            if override_lang != "auto_detect" {
                return override_lang.to_string();
            }
            // Override is "auto_detect" — fall through to level 3
        }

        // Level 3: Auto-detection (whatlang)
        // Only triggered if field or override explicitly requested "auto_detect",
        // OR if no explicit field analyzer was set
        let should_auto_detect = self
            .config
            .field_analyzers
            .get(field_name)
            .map(|l| l == "auto_detect")
            .unwrap_or(true); // No explicit field → auto-detect by default

        if should_auto_detect {
            if let Some(detected) = crate::lang::detect_language(field_value) {
                return detected.name.to_string();
            }
        }

        // Level 4: Index-level default
        self.config.default_language.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
