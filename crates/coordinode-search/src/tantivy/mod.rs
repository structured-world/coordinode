//! tantivy full-text search integration.
//!
//! Provides `TextIndex` — an embedded tantivy index with sync write
//! and per-field text analysis. Each `TextIndex` manages a directory
//! on disk containing tantivy segment files.
//!
//! Write path: mutation → add_document() → IndexWriter::commit() (sync)
//! Read path: search() → Searcher::search() → BM25 scored results
//!
//! Segment merging happens asynchronously in tantivy's background thread.

pub mod multi_lang;
pub mod segment_registry;
pub mod tokenize;

use std::path::Path;
use std::sync::RwLock;

use tantivy::collector::{FilterCollector, TopDocs};
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{
    document::Value as TantivyValue, Field, IndexRecordOption, NumericOptions, OwnedValue, Schema,
    TextFieldIndexing, TextOptions,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{
    LowerCaser, PreTokenizedString, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, Tokenizer,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument};

use self::segment_registry::SegmentRegistry;

/// Fast-field name storing each document's originating Raft proposal commit_ts.
/// Used by `search_at` for MVCC snapshot filtering and by `SegmentRegistry`
/// for per-segment min/max reconciliation.
pub const COMMIT_TS_FIELD: &str = "commit_ts";

#[cfg(any(feature = "cjk-zh", feature = "cjk-ja", feature = "cjk-ko"))]
use crate::cjk;
use crate::stem;

/// Custom Snowball stemmer token filter using our rust-stemmers fork.
///
/// Replaces tantivy's built-in `Stemmer` filter. Supports all 20 languages
/// from our fork including Ukrainian — a single code path for all languages.
#[derive(Clone)]
struct SnowballFilter {
    algorithm: rust_stemmers::Algorithm,
}

impl TokenFilter for SnowballFilter {
    type Tokenizer<T: Tokenizer> = SnowballFilterWrapper<T>;

    fn transform<T: Tokenizer>(self, tokenizer: T) -> Self::Tokenizer<T> {
        SnowballFilterWrapper {
            inner: tokenizer,
            algorithm: self.algorithm,
        }
    }
}

#[derive(Clone)]
struct SnowballFilterWrapper<T> {
    inner: T,
    algorithm: rust_stemmers::Algorithm,
}

impl<T: Tokenizer> Tokenizer for SnowballFilterWrapper<T> {
    type TokenStream<'a> = SnowballTokenStream<T::TokenStream<'a>>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        SnowballTokenStream {
            inner: self.inner.token_stream(text),
            stemmer: rust_stemmers::Stemmer::create(self.algorithm),
        }
    }
}

struct SnowballTokenStream<T> {
    inner: T,
    stemmer: rust_stemmers::Stemmer,
}

impl<T: tantivy::tokenizer::TokenStream> tantivy::tokenizer::TokenStream
    for SnowballTokenStream<T>
{
    fn advance(&mut self) -> bool {
        if !self.inner.advance() {
            return false;
        }
        let stemmed = self.stemmer.stem(&self.inner.token().text);
        self.inner.token_mut().text = stemmed.into_owned();
        true
    }

    fn token(&self) -> &Token {
        self.inner.token()
    }

    fn token_mut(&mut self) -> &mut Token {
        self.inner.token_mut()
    }
}

/// Error type for text search operations.
#[derive(Debug, thiserror::Error)]
pub enum TextSearchError {
    #[error("tantivy error: {0}")]
    Tantivy(#[from] tantivy::TantivyError),

    #[error("query parse error: {0}")]
    QueryParse(#[from] tantivy::query::QueryParserError),

    #[error("index not found or corrupted: {0}")]
    IndexCorrupted(String),
}

/// A scored search result from the text index.
#[derive(Debug, Clone, PartialEq)]
pub struct TextSearchResult {
    /// Node ID that owns the indexed text.
    pub node_id: u64,
    /// BM25 relevance score.
    pub score: f32,
}

/// Full-text index backed by tantivy.
///
/// A search result with highlighted text snippet.
#[derive(Debug, Clone, PartialEq)]
pub struct HighlightedResult {
    /// Node ID.
    pub node_id: u64,
    /// BM25 score.
    pub score: f32,
    /// Highlighted snippet with `<b>` tags around matching terms.
    pub snippet_html: String,
}

/// Each `TextIndex` instance manages one tantivy index directory.
/// In production, there's one `TextIndex` per (label, field_set) combination
/// per shard.
///
/// Thread safety: `IndexWriter` requires exclusive access (one writer).
/// `IndexReader` supports concurrent reads.
/// In clustered mode, only the Raft leader writes; followers rebuild
/// from WAL replay.
pub struct TextIndex {
    index: Index,
    writer: IndexWriter,
    reader: IndexReader,
    schema: Schema,
    /// Schema field for the node ID (stored, not indexed).
    node_id_field: Field,
    /// Schema field for the text body (indexed + stored).
    body_field: Field,
    /// Fast field carrying each document's originating Raft proposal commit_ts.
    /// Legacy `add_document` paths write 0, which is ≤ any valid snapshot T
    /// so such documents remain visible to all snapshot readers.
    commit_ts_field: Field,
    /// Per-segment `(min_ts, max_ts)` cache, reconciled after every reader
    /// reload. Used to partition segments on `search_at(T)` into fully-visible,
    /// fully-hidden, and straddle buckets.
    registry: RwLock<SegmentRegistry>,
}

impl TextIndex {
    /// Open or create a text index at the given directory path.
    ///
    /// `heap_size_bytes`: IndexWriter memory budget (default: 50MB).
    /// `language`: Optional language for stemming (e.g. "english", "russian", "ukrainian").
    ///   If None, uses simple tokenizer without stemming.
    pub fn open_or_create(
        dir: &Path,
        heap_size_bytes: usize,
        language: Option<&str>,
    ) -> Result<Self, TextSearchError> {
        let tokenizer_name = language
            .map(|l| format!("coordinode_{l}"))
            .unwrap_or_else(|| "default".to_string());

        let mut schema_builder = Schema::builder();
        let node_id_opts = NumericOptions::default().set_indexed().set_stored();
        let node_id_field = schema_builder.add_u64_field("node_id", node_id_opts);

        // Configure text field with per-language analyzer
        let text_indexing = TextFieldIndexing::default()
            .set_tokenizer(&tokenizer_name)
            .set_index_option(tantivy::schema::IndexRecordOption::WithFreqsAndPositions);
        let text_opts = TextOptions::default()
            .set_indexing_options(text_indexing)
            .set_stored();
        let body_field = schema_builder.add_text_field("body", text_opts);
        // MVCC snapshot support: commit_ts as u64 fast field per document.
        // Default value 0 for legacy writers makes untagged docs visible to
        // every snapshot reader (0 ≤ any T).
        let commit_ts_opts = NumericOptions::default().set_fast().set_stored();
        let commit_ts_field = schema_builder.add_u64_field(COMMIT_TS_FIELD, commit_ts_opts);
        let schema = schema_builder.build();

        let index = if dir.join("meta.json").exists() {
            Index::open_in_dir(dir)?
        } else {
            std::fs::create_dir_all(dir)
                .map_err(|e| TextSearchError::IndexCorrupted(format!("cannot create dir: {e}")))?;
            Index::create_in_dir(dir, schema.clone())?
        };

        // Register language-specific tokenizer.
        // - "none" → whitespace + lowercase, no stemming/stop words
        // - CJK → dictionary-based segmenters (lindera/jieba)
        // - Others → Snowball stemming via rust-stemmers fork
        if let Some(lang) = language {
            Self::register_language_tokenizer(&index, lang, &tokenizer_name);
        }

        let writer = index.writer(heap_size_bytes)?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            writer,
            reader,
            schema,
            node_id_field,
            body_field,
            commit_ts_field,
            registry: RwLock::new(SegmentRegistry::new()),
        })
    }

    /// Reconcile per-segment commit_ts ranges from the current searcher state.
    /// Called after every write path that commits and reloads the reader.
    fn reconcile_registry(&self) -> Result<(), TextSearchError> {
        let searcher = self.reader.searcher();
        let mut reg = self
            .registry
            .write()
            .map_err(|e| TextSearchError::IndexCorrupted(format!("registry poisoned: {e}")))?;
        reg.reconcile(&searcher, COMMIT_TS_FIELD)?;
        Ok(())
    }

    /// Snapshot of the per-segment registry — for tests and observability.
    #[doc(hidden)]
    pub fn registry_snapshot(
        &self,
    ) -> Vec<(tantivy::index::SegmentId, segment_registry::SegmentTsRange)> {
        let searcher = self.reader.searcher();
        let reg = match self.registry.read() {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };
        searcher
            .segment_readers()
            .iter()
            .filter_map(|sr| reg.get(&sr.segment_id()).map(|r| (sr.segment_id(), r)))
            .collect()
    }

    /// Add (or replace) a document in the index.
    ///
    /// If a document with the same `node_id` already exists, it is
    /// first deleted (by term) then re-added. This provides upsert
    /// semantics.
    ///
    /// Commits synchronously — the document is searchable after return.
    ///
    /// Equivalent to `add_document_at(node_id, text, 0)`. Commit_ts 0 means
    /// the document is visible to every snapshot reader (0 ≤ any T).
    pub fn add_document(&mut self, node_id: u64, text: &str) -> Result<(), TextSearchError> {
        self.add_document_at(node_id, text, 0)
    }

    /// MVCC-aware single-document add. `commit_ts` is the originating Raft
    /// proposal's commit timestamp; stored as a fast field so that
    /// `search_at(T)` can filter documents with `commit_ts > T`.
    pub fn add_document_at(
        &mut self,
        node_id: u64,
        text: &str,
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
        self.writer.delete_term(node_id_term);

        self.writer.add_document(doc!(
            self.node_id_field => node_id,
            self.body_field => text,
            self.commit_ts_field => commit_ts,
        ))?;

        self.writer.commit()?;
        self.reader.reload()?;
        self.reconcile_registry()?;
        Ok(())
    }

    /// Add multiple documents in a single batch commit.
    ///
    /// More efficient than individual add_document calls for bulk inserts.
    pub fn add_documents_batch(&mut self, docs: &[(u64, &str)]) -> Result<(), TextSearchError> {
        self.add_documents_batch_at_uniform(docs, 0)
    }

    /// MVCC-aware batch add: all documents share a single `commit_ts`.
    /// Matches the Raft proposal model — one proposal → one batch → one ts.
    pub fn add_documents_batch_at_uniform(
        &mut self,
        docs: &[(u64, &str)],
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        for &(node_id, text) in docs {
            let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
            self.writer.delete_term(node_id_term);

            self.writer.add_document(doc!(
                self.node_id_field => node_id,
                self.body_field => text,
                self.commit_ts_field => commit_ts,
            ))?;
        }

        self.writer.commit()?;
        self.reader.reload()?;
        self.reconcile_registry()?;
        Ok(())
    }

    /// Delete a document by node ID.
    ///
    /// Commits synchronously.
    pub fn delete_document(&mut self, node_id: u64) -> Result<(), TextSearchError> {
        let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
        self.writer.delete_term(node_id_term);
        self.writer.commit()?;
        self.reader.reload()?;
        self.reconcile_registry()?;
        Ok(())
    }

    /// Build a tantivy Query from a query string, handling `word*` prefix syntax.
    ///
    /// tantivy's QueryParser doesn't natively support `word*` wildcard syntax.
    /// This method detects `word*` patterns and rewrites them to PhrasePrefixQuery,
    /// combining with regular terms via BooleanQuery when mixed.
    fn build_query(
        &self,
        query_str: &str,
        boost: Option<f32>,
    ) -> Result<Box<dyn tantivy::query::Query>, TextSearchError> {
        self.build_query_inner(query_str, boost, false)
    }

    fn build_query_fuzzy(
        &self,
        query_str: &str,
        boost: Option<f32>,
    ) -> Result<Box<dyn tantivy::query::Query>, TextSearchError> {
        self.build_query_inner(query_str, boost, true)
    }

    fn build_query_inner(
        &self,
        query_str: &str,
        boost: Option<f32>,
        fuzzy: bool,
    ) -> Result<Box<dyn tantivy::query::Query>, TextSearchError> {
        use tantivy::query::PhrasePrefixQuery;

        let (prefix_terms, remainder) = extract_prefix_terms(query_str);

        // Case 1: no prefix terms — delegate entirely to QueryParser.
        if prefix_terms.is_empty() {
            let mut parser = QueryParser::for_index(&self.index, vec![self.body_field]);
            if let Some(b) = boost {
                parser.set_field_boost(self.body_field, b);
            }
            if fuzzy {
                // Enable Levenshtein-1 fuzzy expansion for all terms.
                // tantivy's QueryParser does NOT support `term~N` syntax — fuzzy
                // must be enabled via set_field_fuzzy before parsing.
                parser.set_field_fuzzy(self.body_field, false, 1, true);
            }
            let q = parser.parse_query(query_str)?;
            return Ok(q);
        }

        // Build prefix sub-queries.
        let mut subqueries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
        for prefix in &prefix_terms {
            let term = tantivy::Term::from_field_text(self.body_field, prefix);
            let pq = PhrasePrefixQuery::new(vec![term]);
            subqueries.push((Occur::Must, Box::new(pq)));
        }

        // Case 2: only prefix terms, no remainder.
        let remainder = remainder.trim();
        if remainder.is_empty() {
            return Ok(Box::new(BooleanQuery::new(subqueries)));
        }

        // Case 3: mixed — parse remainder via QueryParser, combine with prefix queries.
        // Strip leading/trailing boolean operators left over after prefix extraction.
        let clean = strip_leading_operators(remainder);
        if !clean.is_empty() {
            let mut parser = QueryParser::for_index(&self.index, vec![self.body_field]);
            if let Some(b) = boost {
                parser.set_field_boost(self.body_field, b);
            }
            let remainder_query = parser.parse_query(&clean)?;
            subqueries.push((Occur::Must, remainder_query));
        }

        Ok(Box::new(BooleanQuery::new(subqueries)))
    }

    /// Search the index for documents matching a query string.
    ///
    /// Register a custom tokenizer/analyzer by name.
    ///
    /// Overrides any previously registered tokenizer with the same name.
    /// Used for testing with external CJK dictionaries.
    pub fn register_tokenizer(&self, name: &str, analyzer: tantivy::tokenizer::TextAnalyzer) {
        self.index.tokenizers().register(name, analyzer);
    }

    /// Returns up to `limit` results sorted by BM25 score (descending).
    /// Query syntax: boolean (AND/OR/NOT), phrase ("..."), prefix (word*).
    pub fn search(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        let query = self.build_query(query_str, None)?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    results.push(TextSearchResult { node_id, score });
                }
            }
        }

        Ok(results)
    }

    /// Search with per-field boost weights.
    ///
    /// `boost` multiplies the BM25 score for the body field.
    /// Useful when combining multiple TextIndex instances with different weights.
    pub fn search_boosted(
        &self,
        query_str: &str,
        limit: usize,
        boost: f32,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        let query = self.build_query(query_str, Some(boost))?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    results.push(TextSearchResult { node_id, score });
                }
            }
        }

        Ok(results)
    }

    /// Search with highlighted snippets and Levenshtein-1 fuzzy expansion.
    ///
    /// Uses `set_field_fuzzy` on the QueryParser so every bare term in the query
    /// automatically expands to its edit-1 neighbourhood. This is the correct way
    /// to do fuzzy search in tantivy — NOT by appending `~1` to terms (which is
    /// phrase-slop syntax, not fuzzy-term syntax).
    pub fn search_with_highlights_fuzzy(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        let query = self.build_query_fuzzy(query_str, None)?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;
        let snippet_gen = SnippetGenerator::create(&searcher, &*query, self.body_field)?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    let snippet = snippet_gen.snippet_from_doc(&doc);
                    results.push(HighlightedResult {
                        node_id,
                        score,
                        snippet_html: snippet.to_html(),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Search with highlighted snippets of matching text.
    ///
    /// Returns results with HTML-tagged snippets showing where the query matched.
    /// Uses tantivy's SnippetGenerator with BM25 scoring.
    pub fn search_with_highlights(
        &self,
        query_str: &str,
        limit: usize,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        let query = self.build_query(query_str, None)?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;
        let snippet_gen = SnippetGenerator::create(&searcher, &*query, self.body_field)?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    let snippet = snippet_gen.snippet_from_doc(&doc);
                    results.push(HighlightedResult {
                        node_id,
                        score,
                        snippet_html: snippet.to_html(),
                    });
                }
            }
        }

        Ok(results)
    }

    /// Return the number of documents in the index.
    pub fn num_docs(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Access the tantivy schema.
    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    /// Register a tokenizer for the given language in the index.
    ///
    /// Supports: "none" (literal), CJK (dictionary-based), Snowball (stemming).
    /// Unknown languages are silently ignored (will use tantivy's default tokenizer).
    fn register_language_tokenizer(index: &Index, lang: &str, tokenizer_name: &str) {
        match lang {
            // "none" → whitespace split + lowercase, no stemming or stop words.
            // Useful for technical identifiers, codes, transliterated text.
            "none" => {
                let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(LowerCaser)
                    .build();
                index.tokenizers().register(tokenizer_name, analyzer);
            }
            // "auto_detect" → register a simple tokenizer as fallback.
            // Actual per-document auto-detection happens via PreTokenizedString
            // in add_document_with_language(), not at schema level.
            "auto_detect" => {
                let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
                    .filter(LowerCaser)
                    .build();
                index.tokenizers().register(tokenizer_name, analyzer);
            }
            // CJK: dictionary-based segmenters.
            // Uses CjkDictConfig::from_env() for external dictionary paths.
            // When COORDINODE_CJK_ZH/JA/KO_DICT env vars are set, loads from
            // filesystem instead of embedded dictionaries (~70MB savings).
            #[cfg(feature = "cjk-zh")]
            "chinese_jieba" => {
                let cjk_config = cjk::CjkDictConfig::from_env();
                match cjk_config.chinese_tokenizer() {
                    Ok(tokenizer) => {
                        let analyzer = TextAnalyzer::builder(tokenizer).filter(LowerCaser).build();
                        index.tokenizers().register(tokenizer_name, analyzer);
                    }
                    Err(e) => {
                        tracing::error!("failed to create Chinese tokenizer: {e}");
                    }
                }
            }
            #[cfg(feature = "cjk-ja")]
            "japanese_lindera" => {
                let cjk_config = cjk::CjkDictConfig::from_env();
                match cjk_config.japanese_tokenizer() {
                    Ok(tokenizer) => {
                        let analyzer = TextAnalyzer::builder(tokenizer).filter(LowerCaser).build();
                        index.tokenizers().register(tokenizer_name, analyzer);
                    }
                    Err(e) => {
                        tracing::error!("failed to create Japanese tokenizer: {e}");
                    }
                }
            }
            #[cfg(feature = "cjk-ko")]
            "korean_lindera" => {
                let cjk_config = cjk::CjkDictConfig::from_env();
                match cjk_config.korean_tokenizer() {
                    Ok(tokenizer) => {
                        let analyzer = TextAnalyzer::builder(tokenizer).filter(LowerCaser).build();
                        index.tokenizers().register(tokenizer_name, analyzer);
                    }
                    Err(e) => {
                        tracing::error!("failed to create Korean tokenizer: {e}");
                    }
                }
            }
            // Snowball stemmers
            lang => {
                if let Some(algorithm) = stem::algorithm_for_language(lang) {
                    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
                        .filter(LowerCaser)
                        .filter(SnowballFilter { algorithm })
                        .build();
                    index.tokenizers().register(tokenizer_name, analyzer);
                }
            }
        }
    }

    /// Add a document with a specific language override.
    ///
    /// Unlike `add_document()`, this method tokenizes text using the specified
    /// language's pipeline and stores the result as a `PreTokenizedString`.
    /// This allows different documents in the same index to use different
    /// tokenizers — essential for multi-language indexes.
    ///
    /// If `language` is `"auto_detect"`, whatlang-rs detects the language from
    /// the text content. If `"none"`, no stemming is applied.
    pub fn add_document_with_language(
        &mut self,
        node_id: u64,
        text: &str,
        language: &str,
    ) -> Result<(), TextSearchError> {
        self.add_document_with_language_at(node_id, text, language, 0)
    }

    /// MVCC-aware variant of `add_document_with_language`. See
    /// [`Self::add_document_at`] for `commit_ts` semantics.
    pub fn add_document_with_language_at(
        &mut self,
        node_id: u64,
        text: &str,
        language: &str,
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
        self.writer.delete_term(node_id_term);

        let tokens = tokenize::tokenize_text(text, language);
        let pretokenized = PreTokenizedString {
            text: text.to_string(),
            tokens,
        };

        let mut doc = TantivyDocument::new();
        doc.add_field_value(self.node_id_field, &OwnedValue::U64(node_id));
        doc.add_field_value(self.body_field, &OwnedValue::PreTokStr(pretokenized));
        doc.add_field_value(self.commit_ts_field, &OwnedValue::U64(commit_ts));
        self.writer.add_document(doc)?;

        self.writer.commit()?;
        self.reader.reload()?;
        self.reconcile_registry()?;
        Ok(())
    }

    /// Add multiple documents with per-document language overrides.
    ///
    /// Each tuple is (node_id, text, language). More efficient than
    /// individual `add_document_with_language` calls for bulk inserts.
    pub fn add_documents_batch_with_language(
        &mut self,
        docs: &[(u64, &str, &str)],
    ) -> Result<(), TextSearchError> {
        self.add_documents_batch_with_language_at_uniform(docs, 0)
    }

    /// MVCC-aware multi-language batch: all documents share one `commit_ts`.
    pub fn add_documents_batch_with_language_at_uniform(
        &mut self,
        docs: &[(u64, &str, &str)],
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        for &(node_id, text, language) in docs {
            let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
            self.writer.delete_term(node_id_term);

            let tokens = tokenize::tokenize_text(text, language);
            let pretokenized = PreTokenizedString {
                text: text.to_string(),
                tokens,
            };

            let mut doc = TantivyDocument::new();
            doc.add_field_value(self.node_id_field, &OwnedValue::U64(node_id));
            doc.add_field_value(self.body_field, &OwnedValue::PreTokStr(pretokenized));
            doc.add_field_value(self.commit_ts_field, &OwnedValue::U64(commit_ts));
            self.writer.add_document(doc)?;
        }

        self.writer.commit()?;
        self.reader.reload()?;
        self.reconcile_registry()?;
        Ok(())
    }

    /// Search using a specific language for query tokenization.
    ///
    /// Tokenizes the query string using the specified language's pipeline,
    /// then builds a boolean query from the resulting terms. This allows
    /// searching a multi-language index with a language different from
    /// the index default.
    ///
    /// Supports `word*` prefix syntax: prefix terms are extracted before
    /// tokenization and converted to PhrasePrefixQuery.
    ///
    /// Terms are combined with OR (any term match). For AND semantics,
    /// use the standard `search()` method with boolean query syntax.
    pub fn search_with_language(
        &self,
        query_str: &str,
        limit: usize,
        language: &str,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        use tantivy::query::PhrasePrefixQuery;

        let (prefix_terms, remainder) = extract_prefix_terms(query_str);

        // Tokenize the non-prefix remainder with language pipeline.
        let tokens = tokenize::tokenize_text(remainder.trim(), language);

        if tokens.is_empty() && prefix_terms.is_empty() {
            return Ok(Vec::new());
        }

        // Build sub-queries: OR-combined term queries + Must prefix queries.
        let mut subqueries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = tokens
            .into_iter()
            .map(|tok| {
                let term = tantivy::Term::from_field_text(self.body_field, &tok.text);
                let tq = TermQuery::new(term, IndexRecordOption::WithFreqs);
                (
                    Occur::Should,
                    Box::new(tq) as Box<dyn tantivy::query::Query>,
                )
            })
            .collect();

        for prefix in &prefix_terms {
            let term = tantivy::Term::from_field_text(self.body_field, prefix);
            let pq = PhrasePrefixQuery::new(vec![term]);
            subqueries.push((Occur::Should, Box::new(pq)));
        }

        let query = BooleanQuery::new(subqueries);
        let searcher = self.reader.searcher();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    results.push(TextSearchResult { node_id, score });
                }
            }
        }

        Ok(results)
    }

    /// Search with highlighted snippets using language-aware tokenization.
    ///
    /// Unlike `search_with_highlights` (which uses the schema-level `QueryParser`
    /// tokenizer), this method tokenizes the query via the same language pipeline
    /// used at index time, ensuring stemming/stopword consistency.
    ///
    /// Snippets are generated from the same query used for scoring, so highlighted
    /// terms reflect the language-normalized forms.
    pub fn search_with_highlights_and_language(
        &self,
        query_str: &str,
        limit: usize,
        language: &str,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        use tantivy::query::PhrasePrefixQuery;

        let (prefix_terms, remainder) = extract_prefix_terms(query_str);
        let tokens = tokenize::tokenize_text(remainder.trim(), language);

        if tokens.is_empty() && prefix_terms.is_empty() {
            return Ok(Vec::new());
        }

        let mut subqueries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = tokens
            .into_iter()
            .map(|tok| {
                let term = tantivy::Term::from_field_text(self.body_field, &tok.text);
                let tq = TermQuery::new(term, IndexRecordOption::WithFreqs);
                (
                    Occur::Should,
                    Box::new(tq) as Box<dyn tantivy::query::Query>,
                )
            })
            .collect();

        for prefix in &prefix_terms {
            let term = tantivy::Term::from_field_text(self.body_field, prefix);
            let pq = PhrasePrefixQuery::new(vec![term]);
            subqueries.push((Occur::Should, Box::new(pq)));
        }

        let query = BooleanQuery::new(subqueries);
        let searcher = self.reader.searcher();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit).order_by_score())?;
        let snippet_gen = SnippetGenerator::create(&searcher, &query, self.body_field)?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    let snippet = snippet_gen.snippet_from_doc(&doc);
                    results.push(HighlightedResult {
                        node_id,
                        score,
                        snippet_html: snippet.to_html(),
                    });
                }
            }
        }

        Ok(results)
    }

    /// MVCC snapshot search: returns only documents whose originating Raft
    /// proposal `commit_ts ≤ snapshot_ts`.
    ///
    /// Correctness: every document carries its `commit_ts` as a fast field
    /// (set at write time by the MVCC-aware write path; legacy writers use 0
    /// which is ≤ any valid T). A `FilterCollector` wrapping the standard
    /// `TopDocs` collector applies the predicate during BM25 scoring.
    ///
    /// Merge safety: tantivy merges preserve per-doc fast-field values, so
    /// a merged segment's documents retain their original `commit_ts` — no
    /// manual merge-lineage tracking is required for correctness. The
    /// per-segment `SegmentRegistry` tracks `(min_ts, max_ts)` as observability
    /// and as the hook for a future whole-segment fast path (skip segments
    /// with `min_ts > T` without opening their term dictionaries).
    ///
    /// Query syntax matches `search`.
    pub fn search_at(
        &self,
        query_str: &str,
        limit: usize,
        snapshot_ts: u64,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        let query = self.build_query(query_str, None)?;

        let inner = TopDocs::with_limit(limit).order_by_score();
        let filter = FilterCollector::new(
            COMMIT_TS_FIELD.to_string(),
            move |ts: u64| ts <= snapshot_ts,
            inner,
        );

        let top_docs = searcher.search(&*query, &filter)?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field) {
                if let Some(node_id) = node_id_val.as_u64() {
                    results.push(TextSearchResult { node_id, score });
                }
            }
        }

        Ok(results)
    }
}

/// Extract `word*` prefix terms from a query string.
///
/// Returns (prefix_stems, remainder) where prefix_stems are the word parts
/// without the trailing `*`, and remainder is the query string with prefix
/// terms removed (for delegation to QueryParser or tokenization).
///
/// Rules:
/// - `data*` → prefix "data", removed from remainder
/// - `"exact phrase"` → not a prefix, kept in remainder
/// - `data* AND rust` → prefix "data", remainder "AND rust"
/// - `*` alone → ignored (empty prefix not useful)
fn extract_prefix_terms(query_str: &str) -> (Vec<String>, String) {
    let mut prefixes = Vec::new();
    let mut remainder = String::with_capacity(query_str.len());
    let mut in_quotes = false;

    for token in query_str.split_whitespace() {
        // Track quote state: skip prefix detection inside phrases.
        let quote_count = token.chars().filter(|&c| c == '"').count();
        if in_quotes {
            remainder.push(' ');
            remainder.push_str(token);
            if quote_count % 2 == 1 {
                in_quotes = false;
            }
            continue;
        }

        if quote_count % 2 == 1 {
            in_quotes = true;
            if !remainder.is_empty() {
                remainder.push(' ');
            }
            remainder.push_str(token);
            continue;
        }

        // Detect `word*` pattern: alphanumeric stem followed by `*`.
        if token.ends_with('*') && token.len() > 1 {
            let stem = &token[..token.len() - 1];
            // Only treat as prefix if the stem is a plain word (no operators).
            if stem.chars().all(|c| c.is_alphanumeric() || c == '_') {
                prefixes.push(stem.to_lowercase());
                continue;
            }
        }

        if !remainder.is_empty() {
            remainder.push(' ');
        }
        remainder.push_str(token);
    }

    (prefixes, remainder)
}

/// Strip leading/trailing boolean operators (AND, OR, NOT) from a query string.
///
/// After extracting prefix terms, the remainder may start with a dangling
/// boolean operator (e.g., `"AND engine"` from `"data* AND engine"`).
/// tantivy's QueryParser rejects leading boolean operators as syntax errors.
fn strip_leading_operators(query: &str) -> String {
    let trimmed = query.trim();
    let ops = ["AND ", "OR ", "NOT "];

    let mut s = trimmed;
    // Strip leading operators (may be nested: "AND OR engine")
    loop {
        let mut stripped = false;
        for op in &ops {
            if let Some(rest) = s.strip_prefix(op) {
                s = rest.trim_start();
                stripped = true;
            }
        }
        if !stripped {
            break;
        }
    }

    // Strip trailing operators
    let mut result = s.to_string();
    loop {
        let t = result.trim_end();
        let mut stripped = false;
        for op in &[" AND", " OR", " NOT"] {
            if let Some(rest) = t.strip_suffix(op) {
                result = rest.to_string();
                stripped = true;
                break;
            }
        }
        if !stripped {
            break;
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
