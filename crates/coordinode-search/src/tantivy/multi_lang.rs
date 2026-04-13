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
        self.inner.writer.add_document(doc)?;

        self.inner.writer.commit()?;
        self.inner.reader.reload()?;
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
            self.inner.writer.add_document(doc)?;
        }

        self.inner.writer.commit()?;
        self.inner.reader.reload()?;
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
mod tests {
    use super::*;

    fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn basic_single_language() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(1, &props(&[("body", "the runners are running fast")]))
            .unwrap();

        // Default search uses English stemming
        let results = idx.search("run", 10).unwrap();
        assert!(
            !results.is_empty(),
            "English stemmed search should match 'runners/running'"
        );
    }

    #[test]
    fn explicit_per_field_analyzer() {
        // Level 1: explicit per-field overrides default and auto-detect
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english")
            .with_field_analyzer("title_ru", "russian")
            .with_field_analyzer("title_en", "english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[
                ("title_en", "the runners are running"),
                ("title_ru", "бегущий человек быстро бежал по дороге"),
            ]),
        )
        .unwrap();

        // English field searchable with English stems
        let en = idx.search_with_language("run", 10, "english").unwrap();
        assert!(!en.is_empty(), "English field should be searchable");

        // Russian field searchable with Russian stems
        let ru = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(!ru.is_empty(), "Russian field should be searchable");
    }

    #[test]
    fn per_node_language_override() {
        // Level 2: _language property overrides default
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // Node with Russian override
        idx.add_node(
            1,
            &props(&[
                ("body", "бегущий человек быстро бежал по дороге"),
                ("_language", "russian"),
            ]),
        )
        .unwrap();

        // Should be searchable with Russian stems (despite English default)
        let results = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(
            !results.is_empty(),
            "_language override should apply Russian tokenizer"
        );
    }

    #[test]
    fn auto_detect_fallback() {
        // Level 3: auto-detect when no explicit or override
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // Long enough text for reliable detection
        idx.add_node(
            1,
            &props(&[(
                "body",
                "бегущий человек быстро бежал по дороге к реке через лес",
            )]),
        )
        .unwrap();

        // Auto-detect should identify Russian and use Russian stemming
        let results = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(
            !results.is_empty(),
            "auto-detect should identify Russian for long Russian text"
        );
    }

    #[test]
    fn default_language_fallback() {
        // Level 4: default when detection fails (short text)
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("none");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // Very short text — detection may fail, falls back to "none"
        idx.add_node(1, &props(&[("body", "ok")])).unwrap();

        let results = idx.search_with_language("ok", 10, "none").unwrap();
        assert!(!results.is_empty(), "short text should use default 'none'");
    }

    #[test]
    fn none_language_for_identifiers() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english")
            .with_field_analyzer("key", "none")
            .with_field_analyzer("description", "auto_detect");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[
                ("key", "ERR_CONNECTION_REFUSED"),
                (
                    "description",
                    "Connection to the remote server was refused by the operating system",
                ),
            ]),
        )
        .unwrap();

        // "none" field: exact match (case-insensitive)
        let key_results = idx
            .search_with_language("err_connection_refused", 10, "none")
            .unwrap();
        assert!(
            !key_results.is_empty(),
            "'none' analyzer should match exact identifier"
        );

        // "auto_detect" field: stemmed match
        let desc_results = idx
            .search_with_language("connection", 10, "english")
            .unwrap();
        assert!(
            !desc_results.is_empty(),
            "auto-detected field should be searchable"
        );
    }

    #[test]
    fn mixed_language_documents() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // English doc (auto-detected)
        idx.add_node(
            1,
            &props(&[(
                "body",
                "the runners are running through the beautiful forest in spring",
            )]),
        )
        .unwrap();

        // Russian doc with override
        idx.add_node(
            2,
            &props(&[
                ("body", "бегущий человек быстро бежал по дороге к реке"),
                ("_language", "russian"),
            ]),
        )
        .unwrap();

        assert_eq!(idx.num_docs(), 2);

        // Each doc searchable in its language
        let en = idx.search_with_language("run", 10, "english").unwrap();
        assert!(en.iter().any(|r| r.node_id == 1), "English doc findable");

        let ru = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(ru.iter().any(|r| r.node_id == 2), "Russian doc findable");
    }

    #[test]
    fn batch_add_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_nodes_batch(&[
            (
                1,
                props(&[(
                    "body",
                    "the runners are running through the beautiful forest in spring",
                )]),
            ),
            (
                2,
                props(&[
                    ("body", "бегущий человек быстро бежал по дороге к реке"),
                    ("_language", "russian"),
                ]),
            ),
        ])
        .unwrap();

        assert_eq!(idx.num_docs(), 2);
    }

    #[test]
    fn delete_from_multilang() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[(
                "body",
                "the runners are running through the beautiful forest in spring",
            )]),
        )
        .unwrap();
        assert_eq!(idx.num_docs(), 1);

        idx.delete_document(1).unwrap();
        assert_eq!(idx.num_docs(), 0);
    }

    #[test]
    fn custom_override_property() {
        let dir = tempfile::tempdir().unwrap();
        let config =
            MultiLangConfig::with_default_language("english").with_override_property("lang");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[
                ("body", "бегущий человек быстро бежал по дороге к реке"),
                ("lang", "russian"), // custom property name
            ]),
        )
        .unwrap();

        let results = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(!results.is_empty(), "custom override property should work");
    }

    #[test]
    fn field_filter_respects_config() {
        // When field_analyzers is non-empty, only listed fields are indexed
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english")
            .with_field_analyzer("title", "english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[
                ("title", "graph database technology"),
                ("internal_notes", "secret information should not be indexed"),
            ]),
        )
        .unwrap();

        // Title is indexed
        let title = idx.search("graph", 10).unwrap();
        assert!(!title.is_empty(), "configured field should be indexed");

        // Internal notes should NOT be indexed (not in field_analyzers)
        let notes = idx.search("secret", 10).unwrap();
        assert!(notes.is_empty(), "unconfigured field should not be indexed");
    }

    #[test]
    fn cascade_priority_explicit_over_override() {
        // Level 1 (explicit) takes priority over level 2 (_language override)
        let dir = tempfile::tempdir().unwrap();
        let config =
            MultiLangConfig::with_default_language("english").with_field_analyzer("title", "none"); // explicit: no stemming
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[
                ("title", "runners running"),
                ("_language", "english"), // override says English
            ]),
        )
        .unwrap();

        // Explicit "none" should win — no stemming applied
        // So "run" should NOT match "runners"
        let results = idx.search_with_language("run", 10, "none").unwrap();
        let has_match = results.iter().any(|r| r.node_id == 1);
        // With "none", "runners" is indexed as "runners", not "run"
        // Searching for "run" with "none" tokenizer gives term "run"
        // which doesn't match "runners"
        assert!(
            !has_match,
            "explicit 'none' should override _language 'english'"
        );
    }

    /// Ukrainian default language — indexing + stemmed search via MultiLanguageTextIndex.
    ///
    /// Uses `default_language = "ukrainian"`. Documents are added via `add_node` (the
    /// normal path). Search uses `search_with_language("ukrainian")` which must apply the
    /// same Snowball Ukrainian stemmer at query time, matching the indexed stems.
    ///
    /// "книга" (book) → stem "книг".  Searching for "книга" should find node 1 because
    /// the query is also stemmed to "книг" before lookup.
    #[test]
    fn ukrainian_default_language_stemmed_search() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("ukrainian");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(1, &props(&[("body", "книга про історію міста")]))
            .unwrap();
        idx.add_node(2, &props(&[("body", "кішка сидить на дивані")]))
            .unwrap();

        // Query "книга" → stemmed to "книг" at query time → matches doc 1
        let results = idx.search_with_language("книга", 10, "ukrainian").unwrap();
        assert_eq!(
            results.len(),
            1,
            "Ukrainian stemmer: 'книга' should match indexed 'книга' (both stem to 'книг')"
        );
        assert_eq!(results[0].node_id, 1);

        // Unrelated query should not match
        let no_match = idx
            .search_with_language("автомобіль", 10, "ukrainian")
            .unwrap();
        assert!(no_match.is_empty(), "unrelated term should not match");
    }

    /// Ukrainian auto-detect: document with Ukrainian text, no explicit language set.
    ///
    /// When `default_language = "english"` but the document text is Ukrainian, whatlang
    /// auto-detection should kick in and index with the Ukrainian pipeline. Searching
    /// with explicit `language="ukrainian"` must find it.
    #[test]
    fn ukrainian_auto_detect_indexes_with_ukrainian_stemmer() {
        let dir = tempfile::tempdir().unwrap();
        // Default is english — but Ukrainian text will be auto-detected
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // Distinctly Ukrainian text — whatlang should detect Lang::Ukr
        idx.add_node(
            1,
            &props(&[("body", "Україна розташована у Східній Європі")]),
        )
        .unwrap();

        // Searching with Ukrainian stemmer should find the node
        let results = idx
            .search_with_language("Україна", 10, "ukrainian")
            .unwrap();
        assert!(
            !results.is_empty(),
            "auto-detected Ukrainian text should be findable via ukrainian search"
        );
    }

    /// search_with_highlights on Ukrainian default-language index returns results.
    ///
    /// This exercises Path A of TextService (non-fuzzy, no explicit language):
    /// MultiLanguageTextIndex.search_with_highlights → search_with_highlights_and_language
    /// with `default_language = "ukrainian"`.
    #[test]
    fn ukrainian_search_with_highlights() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("ukrainian");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        idx.add_node(
            1,
            &props(&[("body", "бібліотека програмного забезпечення")]),
        )
        .unwrap();
        idx.add_node(2, &props(&[("body", "рецепти приготування їжі")]))
            .unwrap();

        // "бібліотек" is stem of "бібліотека" — Path A queries via search_with_highlights
        let results = idx.search_with_highlights("бібліотека", 10).unwrap();
        assert!(
            !results.is_empty(),
            "search_with_highlights on Ukrainian index should find 'бібліотека'"
        );
        assert_eq!(results[0].node_id, 1);

        // Unrelated query
        let no = idx.search_with_highlights("автомобіль", 10).unwrap();
        assert!(no.is_empty());
    }

    #[test]
    fn empty_properties_skipped() {
        let dir = tempfile::tempdir().unwrap();
        let config = MultiLangConfig::with_default_language("english");
        let mut idx =
            MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

        // Empty properties — should not crash, should not add doc
        idx.add_node(1, &props(&[])).unwrap();
        assert_eq!(idx.num_docs(), 0, "empty node should not be indexed");
    }
}
