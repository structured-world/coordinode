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
pub mod tokenize;

use std::path::Path;

use tantivy::collector::TopDocs;
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
        })
    }

    /// Add (or replace) a document in the index.
    ///
    /// If a document with the same `node_id` already exists, it is
    /// first deleted (by term) then re-added. This provides upsert
    /// semantics.
    ///
    /// Commits synchronously — the document is searchable after return.
    pub fn add_document(&mut self, node_id: u64, text: &str) -> Result<(), TextSearchError> {
        // Delete existing document with this node_id (upsert)
        let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
        self.writer.delete_term(node_id_term);

        self.writer.add_document(doc!(
            self.node_id_field => node_id,
            self.body_field => text,
        ))?;

        self.writer.commit()?;
        self.reader.reload()?;
        Ok(())
    }

    /// Add multiple documents in a single batch commit.
    ///
    /// More efficient than individual add_document calls for bulk inserts.
    pub fn add_documents_batch(&mut self, docs: &[(u64, &str)]) -> Result<(), TextSearchError> {
        for &(node_id, text) in docs {
            let node_id_term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
            self.writer.delete_term(node_id_term);

            self.writer.add_document(doc!(
                self.node_id_field => node_id,
                self.body_field => text,
            ))?;
        }

        self.writer.commit()?;
        self.reader.reload()?;
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
        self.writer.add_document(doc)?;

        self.writer.commit()?;
        self.reader.reload()?;
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
            self.writer.add_document(doc)?;
        }

        self.writer.commit()?;
        self.reader.reload()?;
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
mod tests {
    use super::*;

    fn temp_index(dir: &Path) -> TextIndex {
        TextIndex::open_or_create(dir, 15_000_000, None).unwrap()
    }

    #[test]
    fn create_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "rust is a systems programming language")
            .unwrap();
        idx.add_document(2, "python is great for data science")
            .unwrap();
        idx.add_document(3, "go is fast and concurrent").unwrap();

        let results = idx.search("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn search_multiple_results() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "graph database for analytics").unwrap();
        idx.add_document(2, "vector database for embeddings")
            .unwrap();
        idx.add_document(3, "relational database for transactions")
            .unwrap();

        let results = idx.search("database", 10).unwrap();
        assert_eq!(results.len(), 3, "all docs contain 'database'");
    }

    #[test]
    fn delete_removes_from_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "important document about rust")
            .unwrap();
        assert_eq!(idx.search("rust", 10).unwrap().len(), 1);

        idx.delete_document(1).unwrap();
        let results = idx.search("rust", 10).unwrap();
        assert_eq!(results.len(), 0, "deleted doc should not appear");
    }

    #[test]
    fn upsert_replaces_content() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "original text about cats").unwrap();
        assert_eq!(idx.search("cats", 10).unwrap().len(), 1);

        // Upsert same node_id with different text
        idx.add_document(1, "updated text about dogs").unwrap();
        assert_eq!(idx.search("cats", 10).unwrap().len(), 0, "old text gone");
        assert_eq!(idx.search("dogs", 10).unwrap().len(), 1, "new text found");
    }

    #[test]
    fn batch_add() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_documents_batch(&[
            (1, "first document"),
            (2, "second document"),
            (3, "third document"),
        ])
        .unwrap();

        assert_eq!(idx.num_docs(), 3);
        assert_eq!(idx.search("document", 10).unwrap().len(), 3);
    }

    #[test]
    fn reopen_persisted_index() {
        let dir = tempfile::tempdir().unwrap();

        // Create and populate
        {
            let mut idx = temp_index(dir.path());
            idx.add_document(42, "persistent data survives restart")
                .unwrap();
        }

        // Reopen
        let idx = temp_index(dir.path());
        let results = idx.search("persistent", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 42);
    }

    #[test]
    fn empty_search() {
        let dir = tempfile::tempdir().unwrap();
        let idx = temp_index(dir.path());

        let results = idx.search("nonexistent", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn boolean_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "rust graph database").unwrap();
        idx.add_document(2, "python graph library").unwrap();
        idx.add_document(3, "rust web framework").unwrap();

        // AND: both terms must match
        let results = idx.search("rust AND graph", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn phrase_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "machine learning is powerful").unwrap();
        idx.add_document(2, "learning machine is a toy").unwrap();

        let results = idx.search("\"machine learning\"", 10).unwrap();
        assert_eq!(results.len(), 1, "phrase should match exact order");
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn limit_respected() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        for i in 0..20 {
            idx.add_document(i, &format!("document number {i} about search"))
                .unwrap();
        }

        let results = idx.search("search", 5).unwrap();
        assert_eq!(results.len(), 5, "limit should cap results");
    }

    #[test]
    fn score_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "rust rust rust rust rust").unwrap(); // high tf
        idx.add_document(2, "rust programming").unwrap(); // lower tf

        let results = idx.search("rust", 10).unwrap();
        assert_eq!(results.len(), 2);
        // Higher tf should score higher (BM25 with saturation)
        assert!(
            results[0].score >= results[1].score,
            "higher tf should score higher"
        );
    }

    // -- Per-language stemmed search --

    #[test]
    fn english_stemmed_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();

        idx.add_document(1, "the runners are running quickly")
            .unwrap();
        idx.add_document(2, "the cat sat on the mat").unwrap();

        // "run" should match "runners" and "running" via English stemmer
        let results = idx.search("run", 10).unwrap();
        assert_eq!(
            results.len(),
            1,
            "stemmed 'run' should match doc with 'runners/running'"
        );
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn russian_stemmed_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("russian")).unwrap();

        idx.add_document(1, "бегущий человек быстро бежал по дороге")
            .unwrap();
        idx.add_document(2, "кошка сидела на коврике").unwrap();

        // "бежать" stem should match "бегущий" and "бежал"
        let results = idx.search("бежать", 10).unwrap();
        assert!(
            !results.is_empty(),
            "Russian stemmer should match inflected forms"
        );
    }

    #[test]
    fn no_stemming_without_language() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "the runners are running quickly")
            .unwrap();

        // Without stemming, "run" should NOT match "runners"
        let results = idx.search("run", 10).unwrap();
        assert_eq!(
            results.len(),
            0,
            "without stemmer, 'run' should not match 'runners'"
        );
    }

    #[test]
    fn stemmed_reopen_persists_config() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut idx =
                TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();
            idx.add_document(1, "the dogs are barking loudly").unwrap();
        }

        // Reopen with same language — stemmed search should still work
        let idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();
        let results = idx.search("bark", 10).unwrap();
        assert_eq!(results.len(), 1, "stemmed search should work after reopen");
    }

    #[test]
    fn ukrainian_stemmed_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("ukrainian")).unwrap();

        idx.add_document(1, "книга про історію міста").unwrap();
        idx.add_document(2, "кішка сидить на дивані").unwrap();

        // "книг" is the stem of "книга" — should match via Ukrainian stemmer
        let results = idx.search("книг", 10).unwrap();
        assert_eq!(
            results.len(),
            1,
            "Ukrainian stemmer should match 'книга' from stem 'книг'"
        );
        assert_eq!(results[0].node_id, 1);
    }

    // -- BM25, fuzzy, phrase, wildcard, boost, highlighting --

    #[test]
    fn fuzzy_search() {
        let dir = tempfile::tempdir().unwrap();
        // No stemming — fuzzy works on raw tokens
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "coordinode is a graph database")
            .unwrap();
        idx.add_document(2, "python is a scripting language")
            .unwrap();

        // Use FuzzyTermQuery directly (QueryParser uses set_field_fuzzy, not ~N syntax)
        use tantivy::query::FuzzyTermQuery;
        let term = tantivy::Term::from_field_text(idx.body_field, "coordnode");
        let fuzzy_query = FuzzyTermQuery::new(term, 2, true);
        let searcher = idx.reader.searcher();
        let top_docs = searcher
            .search(&fuzzy_query, &TopDocs::with_limit(10).order_by_score())
            .unwrap();
        assert_eq!(
            top_docs.len(),
            1,
            "fuzzy distance=2 should match 'coordinode'"
        );
    }

    #[test]
    fn prefix_query_via_api() {
        // PhrasePrefixQuery works programmatically — baseline test.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management system").unwrap();
        idx.add_document(2, "dataflow processing engine").unwrap();
        idx.add_document(3, "web framework").unwrap();

        use tantivy::query::PhrasePrefixQuery;
        let prefix_query =
            PhrasePrefixQuery::new(vec![tantivy::Term::from_field_text(idx.body_field, "data")]);
        let searcher = idx.reader.searcher();
        let top_docs = searcher
            .search(&prefix_query, &TopDocs::with_limit(10).order_by_score())
            .unwrap();
        assert_eq!(
            top_docs.len(),
            2,
            "prefix 'data' should match 'database' + 'dataflow'"
        );
    }

    // -- G012: Wildcard prefix `word*` via search() --

    #[test]
    fn prefix_wildcard_via_search() {
        // `data*` in search() should match "database" and "dataflow".
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management system").unwrap();
        idx.add_document(2, "dataflow processing engine").unwrap();
        idx.add_document(3, "web framework").unwrap();

        let results = idx.search("data*", 10).unwrap();
        assert_eq!(results.len(), 2, "data* should match database + dataflow");
    }

    #[test]
    fn prefix_wildcard_mixed_with_terms() {
        // `data* AND engine` → prefix "data" AND regular term "engine".
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management system").unwrap();
        idx.add_document(2, "dataflow processing engine").unwrap();
        idx.add_document(3, "search engine for web").unwrap();

        let results = idx.search("data* AND engine", 10).unwrap();
        assert_eq!(
            results.len(),
            1,
            "data* AND engine should match only 'dataflow processing engine'"
        );
        assert_eq!(results[0].node_id, 2);
    }

    #[test]
    fn prefix_wildcard_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "hello world").unwrap();

        let results = idx.search("xyz*", 10).unwrap();
        assert!(results.is_empty(), "xyz* should match nothing");
    }

    #[test]
    fn prefix_wildcard_single_char() {
        // `w*` should match "web", "world", etc.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "web framework").unwrap();
        idx.add_document(2, "world map").unwrap();
        idx.add_document(3, "rust language").unwrap();

        let results = idx.search("w*", 10).unwrap();
        assert_eq!(results.len(), 2, "w* should match web + world");
    }

    #[test]
    fn prefix_wildcard_star_alone_ignored() {
        // Bare `*` should not cause errors — treated as regular query.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "hello world").unwrap();

        // Bare `*` is not a valid prefix (empty stem) — falls through to QueryParser.
        let results = idx.search("*", 10);
        // Either returns empty or error — both acceptable; must not panic.
        assert!(results.is_ok() || results.is_err());
    }

    #[test]
    fn prefix_inside_phrase_not_treated_as_wildcard() {
        // `"data*"` inside quotes is a phrase, not a prefix.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management").unwrap();

        // Phrase "data*" → literal search, not prefix.
        // tantivy QueryParser will handle it as a phrase containing "data*".
        let results = idx.search("\"data*\"", 10).unwrap();
        // Should NOT match "database" — it's a phrase search for literal "data*".
        assert!(
            results.is_empty(),
            "quoted \"data*\" should not be treated as prefix"
        );
    }

    #[test]
    fn prefix_wildcard_via_search_boosted() {
        // search_boosted() should also handle `word*`.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management").unwrap();
        idx.add_document(2, "web framework").unwrap();

        let results = idx.search_boosted("data*", 10, 2.0).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn prefix_wildcard_via_search_with_language() {
        // search_with_language() should handle `word*` prefix terms.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management system").unwrap();
        idx.add_document(2, "dataflow engine").unwrap();
        idx.add_document(3, "web application").unwrap();

        let results = idx.search_with_language("data*", 10, "english").unwrap();
        assert_eq!(
            results.len(),
            2,
            "data* via search_with_language should match 2"
        );
    }

    #[test]
    fn prefix_wildcard_via_highlights() {
        // search_with_highlights() should handle `word*`.
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "database management system").unwrap();
        idx.add_document(2, "web framework").unwrap();

        let results = idx.search_with_highlights("data*", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
    }

    // -- extract_prefix_terms unit tests --

    #[test]
    fn extract_prefix_simple() {
        let (prefixes, remainder) = super::extract_prefix_terms("data*");
        assert_eq!(prefixes, vec!["data"]);
        assert_eq!(remainder.trim(), "");
    }

    #[test]
    fn extract_prefix_mixed() {
        let (prefixes, remainder) = super::extract_prefix_terms("data* AND engine");
        assert_eq!(prefixes, vec!["data"]);
        assert_eq!(remainder.trim(), "AND engine");
    }

    #[test]
    fn extract_prefix_multiple() {
        let (prefixes, remainder) = super::extract_prefix_terms("data* web*");
        assert_eq!(prefixes, vec!["data", "web"]);
        assert_eq!(remainder.trim(), "");
    }

    #[test]
    fn extract_prefix_inside_quotes_ignored() {
        let (prefixes, remainder) = super::extract_prefix_terms("\"data*\" regular");
        assert!(
            prefixes.is_empty(),
            "prefix inside quotes should be ignored"
        );
        assert!(remainder.contains("\"data*\""));
        assert!(remainder.contains("regular"));
    }

    #[test]
    fn extract_prefix_bare_star_ignored() {
        let (prefixes, remainder) = super::extract_prefix_terms("* hello");
        assert!(prefixes.is_empty(), "bare * is not a valid prefix");
        assert!(remainder.contains("*"));
    }

    #[test]
    fn extract_prefix_no_prefix() {
        let (prefixes, remainder) = super::extract_prefix_terms("hello world");
        assert!(prefixes.is_empty());
        assert_eq!(remainder.trim(), "hello world");
    }

    #[test]
    fn extract_prefix_case_lowered() {
        let (prefixes, _) = super::extract_prefix_terms("Data*");
        assert_eq!(prefixes, vec!["data"], "prefix should be lowercased");
    }

    #[test]
    fn boosted_search_changes_score() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "rust programming language").unwrap();

        let normal = idx.search("rust", 10).unwrap();
        let boosted = idx.search_boosted("rust", 10, 3.0).unwrap();

        assert_eq!(normal.len(), 1);
        assert_eq!(boosted.len(), 1);
        assert!(
            boosted[0].score > normal[0].score,
            "boost=3.0 ({}) should score higher than default ({})",
            boosted[0].score,
            normal[0].score
        );
    }

    #[test]
    fn highlighted_search() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "rust is a systems programming language for safety")
            .unwrap();
        idx.add_document(2, "python is interpreted and dynamically typed")
            .unwrap();

        let results = idx.search_with_highlights("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
        // Snippet should contain <b> tags around "rust"
        assert!(
            results[0].snippet_html.contains("<b>"),
            "highlighted snippet should contain <b> tags: {}",
            results[0].snippet_html
        );
    }

    #[test]
    fn phrase_with_slop() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        idx.add_document(1, "the big bad wolf").unwrap();
        idx.add_document(2, "the big friendly wolf").unwrap();
        idx.add_document(3, "a small cat").unwrap();

        // Exact phrase: only doc 1
        let exact = idx.search("\"big bad wolf\"", 10).unwrap();
        assert_eq!(exact.len(), 1);
        assert_eq!(exact[0].node_id, 1);

        // Phrase with slop ~1: "big wolf" with 1 word between
        let slop = idx.search("\"big wolf\"~1", 10).unwrap();
        assert_eq!(
            slop.len(),
            2,
            "phrase with slop~1 should match both docs with 'big ... wolf'"
        );
    }

    #[test]
    fn bm25_scoring_rare_term_higher() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = temp_index(dir.path());

        // "database" appears in all docs (low IDF), "coordinode" in one (high IDF)
        idx.add_document(1, "coordinode is a database for graphs")
            .unwrap();
        idx.add_document(2, "postgres is a relational database")
            .unwrap();
        idx.add_document(3, "redis is an in-memory database")
            .unwrap();

        // Search for both terms — BM25 should rank "coordinode" higher
        // because it's rarer (IDF component)
        let results = idx.search("coordinode database", 10).unwrap();
        assert!(!results.is_empty());
        // Doc 1 should be first (has the rare term "coordinode")
        assert_eq!(
            results[0].node_id, 1,
            "doc with rare term should rank first"
        );
    }

    // -- CJK integration tests (per-language feature flags) --

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn chinese_jieba_text_index() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();

        idx.add_document(1, "我来到北京清华大学").unwrap();
        idx.add_document(2, "今天天气真好").unwrap();
        idx.add_document(3, "上海是中国最大的城市").unwrap();

        let results = idx.search("北京", 10).unwrap();
        assert_eq!(results.len(), 1, "should find doc with 北京");
        assert_eq!(results[0].node_id, 1);
    }

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn chinese_jieba_multi_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();

        idx.add_document(1, "中国的首都是北京").unwrap();
        idx.add_document(2, "北京大学是中国著名的大学").unwrap();
        idx.add_document(3, "上海比北京更大").unwrap();

        let results = idx.search("北京", 10).unwrap();
        assert_eq!(results.len(), 3, "all docs contain 北京: {results:?}");
    }

    #[cfg(feature = "cjk-ja")]
    #[test]
    fn japanese_lindera_text_index() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(dir.path(), 15_000_000, Some("japanese_lindera")).unwrap();

        idx.add_document(1, "東京都に住んでいます").unwrap();
        idx.add_document(2, "大阪は美しい街です").unwrap();
        idx.add_document(3, "京都の寺院を訪れました").unwrap();

        let results = idx.search("東京", 10).unwrap();
        assert!(!results.is_empty(), "should find doc with 東京");
        assert_eq!(results[0].node_id, 1);
    }

    #[cfg(feature = "cjk-ko")]
    #[test]
    fn korean_lindera_text_index() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(dir.path(), 15_000_000, Some("korean_lindera")).unwrap();

        idx.add_document(1, "대한민국의 수도는 서울입니다").unwrap();
        idx.add_document(2, "부산은 항구 도시입니다").unwrap();

        let results = idx.search("서울", 10).unwrap();
        assert!(!results.is_empty(), "should find doc with 서울");
    }

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn cjk_upsert_and_delete() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();

        idx.add_document(1, "机器学习很有趣").unwrap();
        assert_eq!(idx.search("机器", 10).unwrap().len(), 1);

        idx.add_document(1, "深度学习更有趣").unwrap();
        assert_eq!(idx.search("机器", 10).unwrap().len(), 0, "old content gone");
        assert_eq!(
            idx.search("深度", 10).unwrap().len(),
            1,
            "new content found"
        );

        idx.delete_document(1).unwrap();
        assert_eq!(idx.search("深度", 10).unwrap().len(), 0, "deleted");
    }

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn cjk_reopen_persists() {
        let dir = tempfile::tempdir().unwrap();

        {
            let mut idx =
                TextIndex::open_or_create(dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();
            idx.add_document(42, "数据库系统设计").unwrap();
        }

        let idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();
        let results = idx.search("数据库", 10).unwrap();
        assert_eq!(results.len(), 1, "should find after reopen");
        assert_eq!(results[0].node_id, 42);
    }

    // -- Multi-language text index --

    #[test]
    fn none_analyzer_no_stemming() {
        // "none" analyzer: whitespace + lowercase, no stemming
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document(1, "The Runners are Running quickly")
            .unwrap();

        // Without stemming, "run" should NOT match "Runners"
        let results = idx.search("run", 10).unwrap();
        assert_eq!(
            results.len(),
            0,
            "'none' analyzer should not stem 'run' to match 'runners'"
        );

        // Exact token match should work (case-insensitive)
        let results = idx.search("runners", 10).unwrap();
        assert_eq!(results.len(), 1, "'runners' should match exactly");
    }

    #[test]
    fn none_analyzer_technical_identifiers() {
        // "none" is useful for code, identifiers, API keys
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document(1, "ERR_CONNECTION_REFUSED status_code 503")
            .unwrap();
        idx.add_document(2, "OK_SUCCESS status_code 200").unwrap();

        let results = idx.search("ERR_CONNECTION_REFUSED", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node_id, 1);
    }

    #[test]
    fn add_document_with_language_english() {
        // Per-document language via PreTokenizedString
        let dir = tempfile::tempdir().unwrap();
        // Index created with "none" default — but documents use English stemming
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document_with_language(1, "the runners are running quickly", "english")
            .unwrap();

        // Search with English stemming should find via stem
        let results = idx.search_with_language("run", 10, "english").unwrap();
        assert!(
            !results.is_empty(),
            "stemmed 'run' should match 'runners/running' indexed with English"
        );
    }

    #[test]
    fn add_document_with_language_russian() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document_with_language(1, "бегущий человек быстро бежал по дороге", "russian")
            .unwrap();

        let results = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(
            !results.is_empty(),
            "Russian stemmed search should match inflected forms"
        );
    }

    #[test]
    fn mixed_language_documents_same_index() {
        // Documents in different languages coexist in one index
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document_with_language(1, "the runners are running quickly", "english")
            .unwrap();
        idx.add_document_with_language(2, "бегущий человек быстро бежал по дороге", "russian")
            .unwrap();
        idx.add_document_with_language(3, "ERR_404_NOT_FOUND", "none")
            .unwrap();

        // English search finds English doc
        let en_results = idx.search_with_language("run", 10, "english").unwrap();
        assert!(
            en_results.iter().any(|r| r.node_id == 1),
            "English search should find English doc"
        );

        // Russian search finds Russian doc
        let ru_results = idx.search_with_language("бежать", 10, "russian").unwrap();
        assert!(
            ru_results.iter().any(|r| r.node_id == 2),
            "Russian search should find Russian doc"
        );

        // "none" search finds literal match
        let none_results = idx
            .search_with_language("err_404_not_found", 10, "none")
            .unwrap();
        assert!(
            none_results.iter().any(|r| r.node_id == 3),
            "'none' search should find literal doc"
        );
    }

    #[test]
    fn add_document_with_auto_detect() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        // Auto-detect should identify English
        idx.add_document_with_language(
            1,
            "The runners are running through the beautiful forest in spring",
            "auto_detect",
        )
        .unwrap();

        // Search with English stemming
        let results = idx.search_with_language("run", 10, "english").unwrap();
        assert!(
            !results.is_empty(),
            "auto-detected English doc should be findable with English search"
        );
    }

    #[test]
    fn batch_with_language() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_documents_batch_with_language(&[
            (1, "running in the park", "english"),
            (2, "бегущий по парку", "russian"),
            (3, "API_KEY_12345", "none"),
        ])
        .unwrap();

        assert_eq!(idx.num_docs(), 3);

        let en = idx.search_with_language("run", 10, "english").unwrap();
        assert!(!en.is_empty(), "English batch doc findable");

        let none = idx
            .search_with_language("api_key_12345", 10, "none")
            .unwrap();
        assert!(!none.is_empty(), "None batch doc findable");
    }

    #[test]
    fn search_with_language_empty_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();
        idx.add_document(1, "some text").unwrap();

        let results = idx.search_with_language("", 10, "english").unwrap();
        assert!(
            results.is_empty(),
            "empty query should return empty results"
        );
    }

    #[test]
    fn upsert_with_language() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("none")).unwrap();

        idx.add_document_with_language(1, "running fast", "english")
            .unwrap();

        // Upsert same node with different language
        idx.add_document_with_language(1, "бег быстрый", "russian")
            .unwrap();

        let en = idx.search_with_language("run", 10, "english").unwrap();
        assert!(en.is_empty(), "old English content should be gone");

        let ru = idx.search_with_language("бег", 10, "russian").unwrap();
        assert!(!ru.is_empty(), "new Russian content should be found");
    }

    /// Integration test: TextIndex with external jieba dictionary.
    ///
    /// Tests the full pipeline: custom dict → JiebaTokenizer → TextIndex → search.
    /// Uses explicit CjkDictConfig (not env var) to avoid race conditions
    /// with parallel tests in `cargo test`.
    #[cfg(feature = "cjk-zh")]
    #[test]
    fn chinese_jieba_external_dict_text_index() {
        use std::io::Write;
        use tantivy::tokenizer::{LowerCaser, TextAnalyzer};

        // Create a custom jieba dictionary with a domain-specific word
        let dict_dir = tempfile::tempdir().unwrap();
        let dict_path = dict_dir.path().join("custom_jieba.dict");
        {
            let mut f = std::fs::File::create(&dict_path).unwrap();
            writeln!(f, "coordinode 100 n").unwrap();
            writeln!(f, "图数据库 50 n").unwrap();
        }

        // Create tokenizer from external dict explicitly (no env var)
        let tokenizer = cjk::JiebaTokenizer::with_dict_path(&dict_path).expect("load custom dict");
        let analyzer = TextAnalyzer::builder(tokenizer).filter(LowerCaser).build();

        // Create TextIndex and register the external-dict tokenizer manually
        let idx_dir = tempfile::tempdir().unwrap();
        let mut idx =
            TextIndex::open_or_create(idx_dir.path(), 15_000_000, Some("chinese_jieba")).unwrap();
        // Override the default tokenizer with our custom-dict one
        idx.register_tokenizer("chinese_jieba", analyzer);

        idx.add_document(1, "coordinode是新一代图数据库").unwrap();
        idx.add_document(2, "传统数据库不支持图查询").unwrap();

        // "coordinode" should be a single token (custom dict word)
        let results = idx.search("coordinode", 10).unwrap();
        assert_eq!(
            results.len(),
            1,
            "custom dict word 'coordinode' should match: {results:?}"
        );
        assert_eq!(results[0].node_id, 1);
    }

    // -- search_with_highlights_fuzzy --

    #[test]
    fn search_with_highlights_fuzzy_finds_typo() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "graph database systems").unwrap();
        idx.add_document(2, "cooking recipes collection").unwrap();

        // "groph" is Levenshtein-1 from "graph"
        let results = idx.search_with_highlights_fuzzy("groph", 10).unwrap();
        assert!(
            !results.is_empty(),
            "fuzzy search 'groph' must match 'graph' (edit-1)"
        );
        assert_eq!(results[0].node_id, 1);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn search_with_highlights_fuzzy_no_match_beyond_edit1() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();

        idx.add_document(1, "rust").unwrap();

        // "xxxx" has edit distance > 1 from "rust" — should not match
        let results = idx.search_with_highlights_fuzzy("xxxx", 10).unwrap();
        assert!(
            results.is_empty(),
            "fuzzy edit-distance >1 must not match: {results:?}"
        );
    }

    #[test]
    fn search_with_highlights_fuzzy_empty_query_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, None).unwrap();
        idx.add_document(1, "anything").unwrap();

        let results = idx.search_with_highlights_fuzzy("", 10).unwrap();
        assert!(results.is_empty(), "empty fuzzy query must return empty");
    }

    // -- search_with_highlights_and_language --

    #[test]
    fn search_with_highlights_and_language_english_stemming() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();

        idx.add_document(1, "running databases efficiently")
            .unwrap();
        idx.add_document(2, "baking bread recipes").unwrap();

        // "run" stems to "run"; "database" stems to "databas"
        let results = idx
            .search_with_highlights_and_language("databases", 10, "english")
            .unwrap();
        assert!(
            !results.is_empty(),
            "English language path should find 'running databases'"
        );
        assert_eq!(results[0].node_id, 1);
        assert!(results[0].score > 0.0);
    }

    #[test]
    fn search_with_highlights_and_language_returns_empty_on_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();

        idx.add_document(1, "graph traversal algorithms").unwrap();

        let results = idx
            .search_with_highlights_and_language("бібліотека", 10, "english")
            .unwrap();
        assert!(
            results.is_empty(),
            "Ukrainian query against English index must return empty"
        );
    }

    #[test]
    fn search_with_highlights_and_language_empty_query_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let mut idx = TextIndex::open_or_create(dir.path(), 15_000_000, Some("english")).unwrap();
        idx.add_document(1, "something").unwrap();

        let results = idx
            .search_with_highlights_and_language("", 10, "english")
            .unwrap();
        assert!(results.is_empty(), "empty query must return empty");
    }
}
