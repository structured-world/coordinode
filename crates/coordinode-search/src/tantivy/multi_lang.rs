//! Multi-language text index with per-document language resolution.
//!
//! `MultiLanguageTextIndex` wraps [`TextIndex`] and provides the 4-level
//! language cascade described in `arch/search/fulltext.md`:
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

use super::tokenize;
use super::{HighlightedResult, TextIndex, TextSearchError, TextSearchResult};

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
        Ok(Self { inner, config })
    }

    /// Wrap an existing `TextIndex` with multi-language support.
    ///
    /// Useful for tests and migration from single-language to multi-language.
    /// The wrapped index uses the config's `default_language` for queries.
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
            return Ok(());
        }

        // Use the inner TextIndex's PreTokenizedString path
        let node_id_term = tantivy::Term::from_field_u64(self.inner.node_id_field, node_id);
        self.inner.writer.delete_term(node_id_term);

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
        self.inner.writer.add_document(doc)?;

        self.inner.writer.commit()?;
        self.inner.reader.reload()?;
        self.inner.reconcile_registry()?;
        Ok(())
    }

    /// Add multiple nodes in a single batch commit.
    pub fn add_nodes_batch(
        &mut self,
        nodes: &[(u64, HashMap<String, String>)],
    ) -> Result<(), TextSearchError> {
        for (node_id, properties) in nodes {
            let node_language_override = properties
                .get(&self.config.language_override_property)
                .map(|s| s.as_str());

            let mut all_tokens = Vec::new();
            let mut all_text = String::new();
            let mut position_offset = 0;

            for (field_name, field_value) in properties {
                if field_name == &self.config.language_override_property {
                    continue;
                }
                if !self.config.field_analyzers.is_empty()
                    && !self.config.field_analyzers.contains_key(field_name)
                {
                    continue;
                }

                let language =
                    self.resolve_language(field_name, field_value, node_language_override);

                let byte_offset = all_text.len();
                if !all_text.is_empty() {
                    all_text.push(' ');
                }
                all_text.push_str(field_value);

                let mut tokens = tokenize::tokenize_text(field_value, &language);
                let text_start = if byte_offset > 0 {
                    byte_offset + 1
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
                continue;
            }

            let node_id_term = tantivy::Term::from_field_u64(self.inner.node_id_field, *node_id);
            self.inner.writer.delete_term(node_id_term);

            let pretokenized = tantivy::tokenizer::PreTokenizedString {
                text: all_text,
                tokens: all_tokens,
            };

            let mut doc = tantivy::TantivyDocument::new();
            doc.add_field_value(
                self.inner.node_id_field,
                &tantivy::schema::OwnedValue::U64(*node_id),
            );
            doc.add_field_value(
                self.inner.body_field,
                &tantivy::schema::OwnedValue::PreTokStr(pretokenized),
            );
            doc.add_field_value(
                self.inner.commit_ts_field,
                &tantivy::schema::OwnedValue::U64(0),
            );
            self.inner.writer.add_document(doc)?;
        }

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
        self.inner.delete_document(node_id)
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
