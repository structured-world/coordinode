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

mod corpus;
pub mod multi_lang;
pub mod pending;
mod query_syntax;
pub mod scratch_dir;
pub mod segment_registry;
pub mod tokenize;
pub mod validity;

use std::path::Path;
use std::sync::RwLock;

use tantivy::collector::{FilterCollector, TopDocs};
use tantivy::query::{BooleanQuery, Occur, QueryParser, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, NumericOptions, OwnedValue, Schema, TextFieldIndexing, TextOptions,
    document::Value as TantivyValue,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::tokenizer::{
    LowerCaser, PreTokenizedString, SimpleTokenizer, TextAnalyzer, Token, TokenFilter, Tokenizer,
};
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, doc};

use self::pending::{Matches, PendingDocuments};
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

    /// The text search query is malformed.
    #[error(transparent)]
    QuerySyntax(#[from] query_syntax::QuerySyntaxError),

    #[error("index not found or corrupted: {0}")]
    IndexCorrupted(String),

    /// A commit failed and whether it landed cannot be read back: the index
    /// takes no more writes until it is opened again.
    #[error("the outcome of a text index commit is unknown: {0}")]
    CommitUnknown(String),

    /// The index's reader is behind its last commit: the read is refused
    /// rather than answered from an older one, until the index's owner
    /// catches the reader up.
    #[error("the text index reader has not caught up with its last commit")]
    ReaderBehind,

    /// The index holds documents written without a birth time, so a read
    /// filtered by birth time cannot tell which of them a snapshot sees.
    #[error(
        "the index holds documents without a birth time; it cannot answer a read at a timestamp"
    )]
    BirthTimesUnknown,

    /// The query's budget stopped the search: its memory limit, deadline
    /// or cancellation.
    #[error(transparent)]
    Budget(#[from] coordinode_core::budget::BudgetStop),
}

// A search no budget can stop has no stop to carry.
impl From<core::convert::Infallible> for TextSearchError {
    fn from(never: core::convert::Infallible) -> Self {
        match never {}
    }
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
/// Which interpretations of an index's label hold a document while a change
/// of what a type implies keeps the label's carriers apart: the documents of
/// both interpretations share the index, and a search under one leaves the
/// other's own documents out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Scope {
    /// Held under the facts in force and the change alike; every document
    /// outside a change.
    #[default]
    Shared,
    /// Held under the facts in force only: the change takes it out.
    Retired,
    /// Held under the change only: the change brings it in.
    Gained,
}

impl Scope {
    /// The value of the scope field of a document written in this scope
    /// while the index follows change `epoch`: shared documents carry 0, the
    /// others their epoch with their role, so a later change's tags never
    /// name a document an earlier one kept apart.
    const fn tag(self, epoch: u64) -> u64 {
        // An index counts its changes from zero, one per change it follows,
        // far below u64::MAX / 2.
        match self {
            Self::Shared => 0,
            Self::Retired => epoch * 2 + 1,
            Self::Gained => epoch * 2 + 2,
        }
    }

    /// The scope a stored scope field value names under change `epoch`: a
    /// document kept apart under an earlier change, which that change's
    /// settling left in place, is shared now, as is one written without a
    /// value.
    const fn of_tag(tag: u64, epoch: u64) -> Self {
        if tag == 0 || (tag - 1) / 2 != epoch {
            Self::Shared
        } else if tag % 2 == 1 {
            Self::Retired
        } else {
            Self::Gained
        }
    }
}

/// The statistics of the documents of each scope apart from the shared one,
/// so a search that leaves a scope out scores against the rest without
/// counting anything at query time.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Aside {
    pub(crate) retired: corpus::Corpus,
    pub(crate) gained: corpus::Corpus,
}

impl Aside {
    /// The statistics of `scope`'s documents; `None` for the shared one.
    pub(crate) fn of(&self, scope: Scope) -> Option<&corpus::Corpus> {
        match scope {
            Scope::Shared => None,
            Scope::Retired => Some(&self.retired),
            Scope::Gained => Some(&self.gained),
        }
    }

    pub(crate) fn of_mut(&mut self, scope: Scope) -> Option<&mut corpus::Corpus> {
        match scope {
            Scope::Shared => None,
            Scope::Retired => Some(&mut self.retired),
            Scope::Gained => Some(&mut self.gained),
        }
    }
}

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
    /// Fast field carrying each document's birth time: the commit_ts of the
    /// write that produced it. A document written without one carries no
    /// value, and [`TextIndex::search_at`] then refuses to answer.
    commit_ts_field: Field,
    /// Stored-only tokens of a pre-tokenized body (see [`encode_tokens`]).
    tokens_field: Field,
    /// Indexed and stored [`Scope`] of each document.
    scope_field: Field,
    /// Per-segment `(min_ts, max_ts)` cache, reconciled after every reader
    /// reload. Used to partition segments on `search_at(T)` into fully-visible,
    /// fully-hidden, and straddle buckets.
    registry: RwLock<SegmentRegistry>,
    /// Exact statistics of the live documents, for an index whose documents
    /// carry their tokens (a [`multi_lang::MultiLanguageTextIndex`]); `None`
    /// scores with Tantivy's own counts.
    corpus: Option<corpus::Corpus>,
    /// What the documents of the scopes other than the shared one add to
    /// `corpus`, kept in step with it.
    aside: Aside,
    /// The change of what a type implies the index's scoped documents
    /// belong to, counted from zero; a settled change moves it on.
    epoch: u64,
    /// Set while the reader is behind the last commit: a commit stands once
    /// it is durable, and a move of the reader that failed after it is
    /// tried again before the next read, which never reads the commit before.
    reader_behind: core::sync::atomic::AtomicBool,
    /// Why the index takes no more writes: a commit's outcome is unknown,
    /// or the writer could not discard what it staged.
    lost: Option<String>,
    /// Failures a test injects, and the reader moves it counts.
    #[cfg(any(test, feature = "test-hooks"))]
    pub hooks: TestHooks,
}

/// Failure injection a test drives through [`TextIndex::hooks`].
#[cfg(any(test, feature = "test-hooks"))]
#[derive(Debug, Default)]
pub struct TestHooks {
    /// How many of the next moves of the reader fail.
    pub fail_reloads: core::sync::atomic::AtomicU32,
    /// Moves of the reader attempted so far.
    pub reload_attempts: core::sync::atomic::AtomicU64,
    /// The next commit lands, and then reads back as unknown.
    pub lose_commit_readback: core::sync::atomic::AtomicBool,
}

/// The schema of a text index and the fields the index reads by handle.
struct Layout {
    tokenizer_name: String,
    schema: Schema,
    node_id_field: Field,
    body_field: Field,
    commit_ts_field: Field,
    tokens_field: Field,
    scope_field: Field,
}

/// Field holding each document's [`Scope`].
const SCOPE_FIELD: &str = "scope";

/// A live document's tokens (`None` for a body stored as plain text) and
/// scope.
type LiveDocument = (Option<Vec<String>>, Scope);

/// Stored-only field holding the tokens a pre-tokenized body was indexed with.
const TOKENS_FIELD: &str = "tokens";

/// Prefix of the commit payload naming the change the index follows.
const EPOCH_PAYLOAD: &str = "scope-epoch:";

/// The commit payload recording `epoch`.
fn encode_epoch(epoch: u64) -> String {
    format!("{EPOCH_PAYLOAD}{epoch}")
}

/// The epoch a commit payload [`encode_epoch`] wrote records.
fn decode_epoch(payload: &str) -> Result<u64, TextSearchError> {
    payload
        .strip_prefix(EPOCH_PAYLOAD)
        .and_then(|n| n.parse().ok())
        .ok_or_else(|| {
            TextSearchError::IndexCorrupted(format!("unreadable commit payload {payload:?}"))
        })
}

/// `tokens` as the bytes of [`TOKENS_FIELD`]: each token's UTF-8 length as a
/// little-endian u32, then its bytes.
fn encode_tokens(tokens: &[String]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.iter().map(|t| t.len() + 4).sum());
    for token in tokens {
        // A token is a word of a document field, far below 4 GiB.
        out.extend_from_slice(&(token.len() as u32).to_le_bytes());
        out.extend_from_slice(token.as_bytes());
    }
    out
}

/// The tokens [`encode_tokens`] wrote; `None` for bytes it did not write.
fn decode_tokens(mut bytes: &[u8]) -> Option<Vec<String>> {
    let mut tokens = Vec::new();
    while !bytes.is_empty() {
        let (len, rest) = bytes.split_first_chunk::<4>()?;
        let len = usize::try_from(u32::from_le_bytes(*len)).ok()?;
        if rest.len() < len {
            return None;
        }
        let (token, rest) = rest.split_at(len);
        tokens.push(String::from_utf8(token.to_vec()).ok()?);
        bytes = rest;
    }
    Some(tokens)
}

impl Layout {
    fn new(language: Option<&str>) -> Self {
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
        // The doc store keeps a pre-tokenized body as its text only, so the
        // tokens a document was indexed with are stored beside it: replacing
        // the document takes exactly those out of the corpus statistics.
        let tokens_field = schema_builder.add_bytes_field(
            TOKENS_FIELD,
            tantivy::schema::BytesOptions::default().set_stored(),
        );
        // Indexed, so a search leaves a scope out by its term before the
        // best matches are taken; stored, so a replaced document is taken out
        // of its own scope's statistics.
        let scope_field = schema_builder.add_u64_field(
            SCOPE_FIELD,
            NumericOptions::default().set_indexed().set_stored(),
        );
        Self {
            tokenizer_name,
            schema: schema_builder.build(),
            node_id_field,
            body_field,
            commit_ts_field,
            tokens_field,
            scope_field,
        }
    }
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
        let layout = Layout::new(language);
        let index = if dir.join("meta.json").exists() {
            Index::open_in_dir(dir)?
        } else {
            std::fs::create_dir_all(dir)
                .map_err(|e| TextSearchError::IndexCorrupted(format!("cannot create dir: {e}")))?;
            Index::create_in_dir(dir, layout.schema.clone())?
        };
        Self::over(index, layout, heap_size_bytes, language)
    }

    /// Create an empty index at `dir`, replacing whatever the directory
    /// held, whose files are written without syncing them (see
    /// [`scratch_dir::ScratchDirectory`]) and charged to `footprints`. Only
    /// for an index the caller rebuilds from its source every time it
    /// creates it.
    pub fn create_scratch(
        dir: &Path,
        heap_size_bytes: usize,
        language: Option<&str>,
        footprints: coordinode_storage::engine::footprint::Footprints,
    ) -> Result<Self, TextSearchError> {
        let layout = Layout::new(language);
        if dir.exists() {
            let cleared = std::fs::remove_dir_all(dir);
            footprints.removed_tree(dir);
            cleared.map_err(|e| {
                TextSearchError::IndexCorrupted(format!("clear {}: {e}", dir.display()))
            })?;
        }
        std::fs::create_dir_all(dir)
            .map_err(|e| TextSearchError::IndexCorrupted(format!("cannot create dir: {e}")))?;
        let directory = scratch_dir::ScratchDirectory::open(dir, footprints)
            .map_err(|e| TextSearchError::IndexCorrupted(format!("open {}: {e}", dir.display())))?;
        let index = Index::create(
            directory,
            layout.schema.clone(),
            tantivy::IndexSettings::default(),
        )?;
        Self::over(index, layout, heap_size_bytes, language)
    }

    /// The text index over `index`, laid out as `layout`.
    fn over(
        index: Index,
        layout: Layout,
        heap_size_bytes: usize,
        language: Option<&str>,
    ) -> Result<Self, TextSearchError> {
        let Layout {
            tokenizer_name,
            schema,
            node_id_field,
            body_field,
            commit_ts_field,
            tokens_field,
            scope_field,
        } = layout;

        // Register language-specific tokenizer.
        // - "none" → whitespace + lowercase, no stemming/stop words
        // - CJK → dictionary-based segmenters (lindera/jieba)
        // - Others → Snowball stemming via rust-stemmers fork
        if let Some(lang) = language {
            Self::register_language_tokenizer(&index, lang, &tokenizer_name);
        }

        // The change the committed documents' tags name was committed with
        // them; an index without one follows the first.
        let epoch = match index.load_metas()?.payload {
            Some(payload) => decode_epoch(&payload)?,
            None => 0,
        };
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
            tokens_field,
            scope_field,
            registry: RwLock::new(SegmentRegistry::new()),
            corpus: None,
            aside: Aside::default(),
            epoch,
            reader_behind: core::sync::atomic::AtomicBool::new(false),
            lost: None,
            #[cfg(any(test, feature = "test-hooks"))]
            hooks: TestHooks::default(),
        })
    }

    /// Move the reader to the commit just made. The commit stands whatever
    /// happens here: a failed move leaves reads refused until the owner
    /// catches the reader up (see [`Self::catch_up`]).
    fn follow_commit(&self) {
        let moved = self
            .reload_reader()
            .and_then(|()| self.reconcile_registry());
        if let Err(e) = moved {
            self.reader_behind
                .store(true, core::sync::atomic::Ordering::Release);
            tracing::warn!(error = %e, "text index reader left behind its last commit; reads refuse until it catches up");
        }
    }

    /// Whether the reader is behind the last commit, so reads refuse.
    pub fn reader_behind(&self) -> bool {
        self.reader_behind
            .load(core::sync::atomic::Ordering::Acquire)
    }

    /// Move a reader left behind its last commit there: the owner's repair,
    /// run by the index's writer path or its maintainer, never inside a
    /// read. Nothing to do for a reader that follows.
    ///
    /// # Errors
    ///
    /// The reader still cannot be moved; reads keep refusing.
    pub fn catch_up(&self) -> Result<(), TextSearchError> {
        if self.reader_behind() {
            self.reload_reader()?;
            self.reconcile_registry()?;
            self.reader_behind
                .store(false, core::sync::atomic::Ordering::Release);
        }
        Ok(())
    }

    /// A searcher over the last commit.
    ///
    /// # Errors
    ///
    /// [`TextSearchError::ReaderBehind`] while the reader is behind it:
    /// nothing older is read, and the read does not repair it.
    fn searcher(&self) -> Result<tantivy::Searcher, TextSearchError> {
        if self.reader_behind() {
            return Err(TextSearchError::ReaderBehind);
        }
        Ok(self.reader.searcher())
    }

    /// Move the reader to the last commit.
    fn reload_reader(&self) -> Result<(), TextSearchError> {
        #[cfg(any(test, feature = "test-hooks"))]
        {
            use core::sync::atomic::Ordering;
            self.hooks.reload_attempts.fetch_add(1, Ordering::AcqRel);
            if self
                .hooks
                .fail_reloads
                .try_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                .is_ok()
            {
                return Err(TextSearchError::IndexCorrupted(
                    "the reader was not moved, as the test asked".into(),
                ));
            }
        }
        self.reader.reload()?;
        Ok(())
    }

    /// Commit the staged changes with the change the index follows, so the
    /// documents' scope tags and the epoch they are read under are one
    /// commit: a reopen reads both or neither.
    ///
    /// # Errors
    ///
    /// The commit did not land, so nothing staged is visible; or
    /// [`TextSearchError::CommitUnknown`] when whether it landed cannot be
    /// read back.
    fn commit_staged(&mut self) -> Result<(), TextSearchError> {
        let mut prepared = self.writer.prepare_commit()?;
        prepared.set_payload(&encode_epoch(self.epoch));
        let opstamp = prepared.opstamp();
        let committed = prepared.commit();
        #[cfg(any(test, feature = "test-hooks"))]
        if committed.is_ok()
            && self
                .hooks
                .lose_commit_readback
                .swap(false, core::sync::atomic::Ordering::AcqRel)
        {
            return Err(self.unknown(
                "the commit landed and its read-back was lost, as the test asked".into(),
            ));
        }
        let Err(failed) = committed else {
            return Ok(());
        };
        // A commit reporting a failure may still have published its meta:
        // what the directory holds decides.
        match self.index.load_metas() {
            Ok(metas) if metas.opstamp == opstamp => Ok(()),
            Ok(_) => Err(failed.into()),
            Err(read) => {
                Err(self.unknown(format!("{failed}; reading back the last commit: {read}")))
            }
        }
    }

    /// A commit whose outcome is unknown: the index takes no more writes.
    fn unknown(&mut self, reason: String) -> TextSearchError {
        self.lost = Some(format!("a commit's outcome is unknown: {reason}"));
        TextSearchError::CommitUnknown(reason)
    }

    /// The scope field value of a document written in `scope` now.
    pub(crate) fn scope_tag(&self, scope: Scope) -> u64 {
        scope.tag(self.epoch)
    }

    /// The term matching the documents written in `scope` under the change
    /// the index follows now.
    fn scope_term(&self, scope: Scope) -> tantivy::Term {
        tantivy::Term::from_field_u64(self.scope_field, self.scope_tag(scope))
    }

    /// The tokens a stored document's body was indexed with, one per
    /// position; `None` for a document written without them.
    fn stored_tokens(&self, doc: &TantivyDocument) -> Option<Vec<String>> {
        decode_tokens(doc.get_first(self.tokens_field)?.as_bytes()?)
    }

    /// The scope a stored document was written in.
    fn stored_scope(&self, doc: &TantivyDocument) -> Scope {
        Scope::of_tag(
            doc.get_first(self.scope_field)
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            self.epoch,
        )
    }

    /// `node_id`'s live document as `searcher` sees it, by its tokens (`None`
    /// for a body stored as plain text) and its scope; `None` when there is
    /// none.
    fn live_tokens(
        &self,
        searcher: &tantivy::Searcher,
        node_id: u64,
    ) -> Result<Option<LiveDocument>, TextSearchError> {
        let term = tantivy::Term::from_field_u64(self.node_id_field, node_id);
        let query = TermQuery::new(term, IndexRecordOption::Basic);
        let found = searcher.search(&query, &TopDocs::with_limit(1).order_by_score())?;
        let Some((_, address)) = found.first() else {
            return Ok(None);
        };
        let doc: TantivyDocument = searcher.doc(*address)?;
        Ok(Some((self.stored_tokens(&doc), self.stored_scope(&doc))))
    }

    /// Count every live document of the committed index, and those of the
    /// scopes apart from the shared one.
    fn recount(&self) -> Result<(corpus::Corpus, Aside), TextSearchError> {
        let searcher = self.searcher()?;
        let mut counted = corpus::Corpus::default();
        let mut aside = Aside::default();
        for (ordinal, segment) in searcher.segment_readers().iter().enumerate() {
            let ordinal = u32::try_from(ordinal).map_err(|_| {
                TextSearchError::IndexCorrupted("more segments than a u32 holds".into())
            })?;
            for doc_id in segment.doc_ids_alive() {
                let doc: TantivyDocument =
                    searcher.doc(tantivy::DocAddress::new(ordinal, doc_id))?;
                if let Some(tokens) = self.stored_tokens(&doc) {
                    counted.add(&tokens);
                    if let Some(scoped) = aside.of_mut(self.stored_scope(&doc)) {
                        scoped.add(&tokens);
                    }
                }
            }
        }
        Ok((counted, aside))
    }

    /// The node ids of the live documents of `scope`, as the committed
    /// index holds them, ascending.
    fn scope_members(&self, scope: Scope) -> Result<Vec<u64>, TextSearchError> {
        let searcher = self.searcher()?;
        let query = TermQuery::new(self.scope_term(scope), IndexRecordOption::Basic);
        let addresses = searcher.search(&query, &tantivy::collector::DocSetCollector)?;
        let mut members = Vec::with_capacity(addresses.len());
        for address in addresses {
            let doc: TantivyDocument = searcher.doc(address)?;
            if let Some(id) = doc.get_first(self.node_id_field).and_then(|v| v.as_u64()) {
                members.push(id);
            }
        }
        members.sort_unstable();
        Ok(members)
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
    /// The document carries no birth time, so the index no longer answers
    /// [`Self::search_at`]; use [`Self::add_document_at`] for that.
    pub fn add_document(&mut self, node_id: u64, text: &str) -> Result<(), TextSearchError> {
        self.add_documents(&[(node_id, text)], None)
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
        self.add_documents(&[(node_id, text)], Some(commit_ts))
    }

    /// Add multiple documents in a single batch commit, without birth times
    /// (see [`Self::add_document`]).
    ///
    /// More efficient than individual add_document calls for bulk inserts.
    pub fn add_documents_batch(&mut self, docs: &[(u64, &str)]) -> Result<(), TextSearchError> {
        self.add_documents(docs, None)
    }

    /// MVCC-aware batch add: all documents share a single `commit_ts`.
    /// Matches the Raft proposal model — one proposal → one batch → one ts.
    pub fn add_documents_batch_at_uniform(
        &mut self,
        docs: &[(u64, &str)],
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        self.add_documents(docs, Some(commit_ts))
    }

    /// Replace the documents of `docs` in one commit, each born at
    /// `commit_ts` when it is known.
    fn add_documents(
        &mut self,
        docs: &[(u64, &str)],
        commit_ts: Option<u64>,
    ) -> Result<(), TextSearchError> {
        self.commit_writes(|index| {
            for &(node_id, text) in docs {
                let node_id_term = tantivy::Term::from_field_u64(index.node_id_field, node_id);
                index.writer.delete_term(node_id_term);

                let mut doc = doc!(
                    index.node_id_field => node_id,
                    index.body_field => text,
                );
                if let Some(commit_ts) = commit_ts {
                    doc.add_u64(index.commit_ts_field, commit_ts);
                }
                index.writer.add_document(doc)?;
            }
            Ok(())
        })
    }

    /// Delete a document by node ID.
    ///
    /// Commits synchronously.
    pub fn delete_document(&mut self, node_id: u64) -> Result<(), TextSearchError> {
        self.commit_writes(|index| {
            let node_id_term = tantivy::Term::from_field_u64(index.node_id_field, node_id);
            index.writer.delete_term(node_id_term);
            Ok(())
        })
    }

    /// Stage changes with `stage` and commit them: the one path every write
    /// of the index's own takes. Refused before anything is staged once a
    /// commit's outcome is unknown; a failure before the commit landed
    /// discards what was staged.
    fn commit_writes(
        &mut self,
        stage: impl FnOnce(&mut Self) -> Result<(), TextSearchError>,
    ) -> Result<(), TextSearchError> {
        self.writable()?;
        let outcome = stage(self).and_then(|()| self.commit_staged());
        match &outcome {
            Ok(()) => self.follow_commit(),
            // Nothing is known to discard: the index stays refused.
            Err(TextSearchError::CommitUnknown(_)) => {}
            Err(_) => self.discard_staged()?,
        }
        outcome
    }

    /// Refuse a write once a commit's outcome is unknown or the writer is
    /// gone: the documents in memory may then differ from the directory's,
    /// which only reopening the index reads again.
    pub(crate) fn writable(&self) -> Result<(), TextSearchError> {
        match &self.lost {
            Some(reason) => Err(TextSearchError::IndexCorrupted(reason.clone())),
            None => Ok(()),
        }
    }

    /// Discard every staged change; a writer that cannot do so refuses
    /// every later write.
    pub(crate) fn discard_staged(&mut self) -> Result<(), TextSearchError> {
        if let Err(e) = self.writer.rollback() {
            let reason = format!("the index writer is gone after a failed rollback: {e}");
            self.lost = Some(reason.clone());
            return Err(TextSearchError::IndexCorrupted(reason));
        }
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

    /// A `prefix*` query matching every indexed word that starts with
    /// `prefix`: the words in `[prefix, next)`, where `next` is the first
    /// byte string after every word with that prefix.
    ///
    /// A phrase-prefix query would stop after a fixed number of words per
    /// segment, silently dropping matches and letting words written later
    /// decide which earlier ones survive. The range has no such cutoff: it
    /// streams the segment's term dictionary over the range and marks the
    /// matching documents in one bitset of the segment's size, so its memory
    /// does not grow with the number of words and its work is the postings of
    /// the matching words. Every match scores the same, as a single-word
    /// prefix did before.
    fn prefix_query(&self, prefix: &str) -> tantivy::query::RangeQuery {
        use std::ops::Bound;
        let start = tantivy::Term::from_field_text(self.body_field, prefix);
        let end = match prefix_successor(prefix.as_bytes()) {
            Some(next) => {
                let mut end = tantivy::Term::from_field_text(self.body_field, "");
                end.append_bytes(&next);
                Bound::Excluded(end)
            }
            None => Bound::Unbounded,
        };
        tantivy::query::RangeQuery::new(Bound::Included(start), end)
    }

    fn build_query_inner(
        &self,
        query_str: &str,
        boost: Option<f32>,
        fuzzy: bool,
    ) -> Result<Box<dyn tantivy::query::Query>, TextSearchError> {
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
            subqueries.push((Occur::Must, Box::new(self.prefix_query(prefix))));
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
        let searcher = self.searcher()?;
        let query = self.build_query(query_str, None)?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field)
                && let Some(node_id) = node_id_val.as_u64()
            {
                results.push(TextSearchResult { node_id, score });
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
        let searcher = self.searcher()?;
        let query = self.build_query(query_str, Some(boost))?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field)
                && let Some(node_id) = node_id_val.as_u64()
            {
                results.push(TextSearchResult { node_id, score });
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
        let query = self.build_query_fuzzy(query_str, None)?;
        self.collect(
            &*query,
            Matches::Top(limit),
            &PendingDocuments::none(),
            None,
            pending::Highlight::NearWords(query_str),
            &mut coordinode_core::budget::Unmetered,
        )
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
        let searcher = self.searcher()?;
        let query = self.build_query(query_str, None)?;

        let top_docs = searcher.search(&*query, &TopDocs::with_limit(limit).order_by_score())?;
        let snippet_gen = SnippetGenerator::create(&searcher, &*query, self.body_field)?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(node_id_val) = doc.get_first(self.node_id_field)
                && let Some(node_id) = node_id_val.as_u64()
            {
                let snippet = snippet_gen.snippet_from_doc(&doc);
                results.push(HighlightedResult {
                    node_id,
                    score,
                    snippet_html: snippet.to_html(),
                });
            }
        }

        Ok(results)
    }

    /// Return the number of documents in the index.
    ///
    /// # Errors
    ///
    /// [`TextSearchError::ReaderBehind`] while the reader is behind the
    /// last commit.
    pub fn num_docs(&self) -> Result<u64, TextSearchError> {
        Ok(self.searcher()?.num_docs())
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
    /// the text content. If `"none"`, no stemming is applied. The document
    /// carries no birth time (see [`Self::add_document`]).
    pub fn add_document_with_language(
        &mut self,
        node_id: u64,
        text: &str,
        language: &str,
    ) -> Result<(), TextSearchError> {
        self.add_documents_with_language(&[(node_id, text, language)], None)
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
        self.add_documents_with_language(&[(node_id, text, language)], Some(commit_ts))
    }

    /// Add multiple documents with per-document language overrides, without
    /// birth times (see [`Self::add_document`]).
    ///
    /// Each tuple is (node_id, text, language). More efficient than
    /// individual `add_document_with_language` calls for bulk inserts.
    pub fn add_documents_batch_with_language(
        &mut self,
        docs: &[(u64, &str, &str)],
    ) -> Result<(), TextSearchError> {
        self.add_documents_with_language(docs, None)
    }

    /// MVCC-aware multi-language batch: all documents share one `commit_ts`.
    pub fn add_documents_batch_with_language_at_uniform(
        &mut self,
        docs: &[(u64, &str, &str)],
        commit_ts: u64,
    ) -> Result<(), TextSearchError> {
        self.add_documents_with_language(docs, Some(commit_ts))
    }

    /// [`Self::add_documents`] with each text tokenized by its language.
    fn add_documents_with_language(
        &mut self,
        docs: &[(u64, &str, &str)],
        commit_ts: Option<u64>,
    ) -> Result<(), TextSearchError> {
        self.commit_writes(|index| {
            for &(node_id, text, language) in docs {
                let node_id_term = tantivy::Term::from_field_u64(index.node_id_field, node_id);
                index.writer.delete_term(node_id_term);

                let tokens = tokenize::tokenize_text(text, language);
                let pretokenized = PreTokenizedString {
                    text: text.to_string(),
                    tokens,
                };

                let mut doc = TantivyDocument::new();
                doc.add_field_value(index.node_id_field, &OwnedValue::U64(node_id));
                doc.add_field_value(index.body_field, &OwnedValue::PreTokStr(pretokenized));
                if let Some(commit_ts) = commit_ts {
                    doc.add_field_value(index.commit_ts_field, &OwnedValue::U64(commit_ts));
                }
                index.writer.add_document(doc)?;
            }
            Ok(())
        })?;
        Ok(())
    }

    /// Search using a specific language for query tokenization.
    ///
    /// Tokenizes the query string using the specified language's pipeline,
    /// then builds a boolean query from the resulting terms. This allows
    /// searching a multi-language index with a language different from
    /// the index default.
    ///
    /// The query is in the text query language: words (any of them
    /// matching), phrases, `AND` / `OR` / `NOT`, `word*`, `word~N`, `^B`.
    pub fn search_with_language(
        &self,
        query_str: &str,
        limit: usize,
        language: &str,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        let Some(query) = self.language_query(query_str, language)? else {
            return Ok(Vec::new());
        };
        Ok(self
            .collect(
                &*query,
                Matches::Top(limit),
                &PendingDocuments::none(),
                None,
                pending::Highlight::Off,
                &mut coordinode_core::budget::Unmetered,
            )?
            .into_iter()
            .map(|hit| TextSearchResult {
                node_id: hit.node_id,
                score: hit.score,
            })
            .collect())
    }

    /// Every match of `query_str`, analyzed by `language`'s pipeline, with
    /// its score, all from one commit: what a membership predicate needs,
    /// which a top-k cut sized from another read could truncate.
    ///
    /// # Errors
    ///
    /// As [`Self::search_with_language`].
    pub fn search_all_with_language(
        &self,
        query_str: &str,
        language: &str,
    ) -> Result<Vec<TextSearchResult>, TextSearchError> {
        let Some(query) = self.language_query(query_str, language)? else {
            return Ok(Vec::new());
        };
        Ok(self
            .collect(
                &*query,
                Matches::All,
                &PendingDocuments::none(),
                None,
                pending::Highlight::Off,
                &mut coordinode_core::budget::Unmetered,
            )?
            .into_iter()
            .map(|hit| TextSearchResult {
                node_id: hit.node_id,
                score: hit.score,
            })
            .collect())
    }

    /// The query of a language search: `query_str` in the text query
    /// language (see the `query_syntax` module), each word and phrase
    /// analyzed by `language`'s pipeline; `None` when it leaves nothing to
    /// search for.
    ///
    /// # Errors
    ///
    /// [`TextSearchError::QuerySyntax`] for a malformed query.
    fn language_query(
        &self,
        query_str: &str,
        language: &str,
    ) -> Result<Option<Box<dyn tantivy::query::Query>>, TextSearchError> {
        let prefix =
            |stem: &str| -> Box<dyn tantivy::query::Query> { Box::new(self.prefix_query(stem)) };
        Ok(query_syntax::build(
            query_str,
            self.body_field,
            language,
            &prefix,
        )?)
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
        let Some(query) = self.language_query(query_str, language)? else {
            return Ok(Vec::new());
        };
        self.collect(
            &*query,
            Matches::Top(limit),
            &PendingDocuments::none(),
            None,
            pending::Highlight::Query,
            &mut coordinode_core::budget::Unmetered,
        )
    }

    /// Birth-time filtered search: returns only the held documents whose
    /// originating commit has `commit_ts ≤ snapshot_ts`.
    ///
    /// This filters by birth time and nothing else: a document replaced or
    /// deleted after `snapshot_ts` is not brought back, and scores use the
    /// statistics of the documents held now. It is not a read of the index
    /// as it stood at `snapshot_ts`.
    ///
    /// # Errors
    ///
    /// [`TextSearchError::BirthTimesUnknown`] when any document of the index
    /// was written without a birth time: which of them the filter keeps
    /// cannot be told, so the read is refused rather than guessed.
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
        let searcher = self.searcher()?;
        // A column holding a value for every document of its segment is
        // full; anything less means some document has no birth time.
        for segment in searcher.segment_readers() {
            let column = segment.fast_fields().u64(COMMIT_TS_FIELD)?;
            if segment.max_doc() > 0
                && !matches!(
                    column.get_cardinality(),
                    tantivy::columnar::Cardinality::Full
                )
            {
                return Err(TextSearchError::BirthTimesUnknown);
            }
        }
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
            if let Some(node_id_val) = doc.get_first(self.node_id_field)
                && let Some(node_id) = node_id_val.as_u64()
            {
                results.push(TextSearchResult { node_id, score });
            }
        }

        Ok(results)
    }
}

/// The least byte string greater than every string starting with `prefix`:
/// `prefix` with its trailing 0xFF bytes dropped and its last byte raised by
/// one; `None` when every byte is 0xFF (no such string, the range is open).
fn prefix_successor(prefix: &[u8]) -> Option<Vec<u8>> {
    let last = prefix.iter().rposition(|byte| *byte != u8::MAX)?;
    let mut next = prefix[..=last].to_vec();
    // A byte below 0xFF, so raising it cannot overflow.
    next[last] += 1;
    Some(next)
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
