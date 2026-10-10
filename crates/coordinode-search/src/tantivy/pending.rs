//! Searching an index together with the documents it has not caught up with.
//!
//! An index maintained behind its source holds stale or missing documents for
//! the nodes written since its position. A search leaves the index's
//! documents of those nodes out and runs the same query over a transient
//! segment of their current documents, scored with the index's corpus
//! statistics so the two result sets rank on one scale.

use coordinode_core::budget::Meter;
use tantivy::collector::TopDocs;
use tantivy::query::{
    Bm25StatisticsProvider, BooleanQuery, EnableScoring, Occur, Query, TermSetQuery,
};
use tantivy::schema::Field;
use tantivy::schema::document::Value as _;
use tantivy::snippet::SnippetGenerator;
use tantivy::{
    DocAddress, DocId, DocSet as _, Index, ReloadPolicy, Score, Searcher, SegmentOrdinal,
    SingleSegmentIndexWriter, TERMINATED, TantivyDocument, Term,
};

use super::corpus::{Corpus, CorpusChange};
use super::{HighlightedResult, TextIndex, TextSearchError};

/// Memory budget of the writer that builds a pending segment. It sizes the
/// writer's term table only; the arena grows with the documents.
const PENDING_SEGMENT_BUDGET: usize = 1 << 20;

/// Which of the index's documents a search leaves out.
#[derive(Debug, Clone, Default)]
enum Superseded {
    /// None: the index answers for every node.
    #[default]
    None,
    /// The documents of these nodes.
    Nodes(Vec<u64>),
    /// Every document: the index answers for no node.
    All,
}

/// The documents a search reads in place of the index's own for the nodes
/// the index has not caught up with.
#[derive(Clone, Default)]
pub struct PendingDocuments {
    superseded: Superseded,
    /// The current documents of the superseded nodes that have text.
    segment: Option<Searcher>,
    /// What those documents add to the corpus.
    added: Corpus,
}

impl std::fmt::Debug for PendingDocuments {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingDocuments")
            .field("superseded", &self.superseded)
            .field(
                "documents",
                &self.segment.as_ref().map(Searcher::num_docs).unwrap_or(0),
            )
            .finish()
    }
}

impl PendingDocuments {
    /// Nothing pending: the index answers alone.
    pub fn none() -> Self {
        Self::default()
    }
}

/// Which words a search's snippets highlight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Highlight<'a> {
    /// No snippets.
    Off,
    /// The terms of the query.
    Query,
    /// The words of each match within one edit of a word of this query text,
    /// as an edit-1 fuzzy query matches them: its terms are the typed words,
    /// not the ones it found.
    NearWords(&'a str),
}

/// How many matches a search keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Matches {
    /// The best `n` by score.
    Top(usize),
    /// Every match, as a membership predicate needs.
    All,
}

impl TextIndex {
    /// The pending documents of a search: the index's documents of
    /// `superseded` nodes (every node when `None`) are left out, and
    /// `documents` (their current documents) are searched beside it.
    pub(crate) fn pending(
        &self,
        superseded: Option<&[u64]>,
        documents: Vec<(TantivyDocument, Vec<String>)>,
    ) -> Result<PendingDocuments, TextSearchError> {
        let superseded = match superseded {
            None => Superseded::All,
            Some([]) => Superseded::None,
            Some(nodes) => Superseded::Nodes(nodes.to_vec()),
        };
        if documents.is_empty() {
            return Ok(PendingDocuments {
                superseded,
                segment: None,
                added: Corpus::default(),
            });
        }
        let mut index = Index::create_in_ram(self.schema.clone());
        // The same analyzers, for snippets of the pending documents.
        index.set_tokenizers(self.index.tokenizers().clone());
        let mut writer: SingleSegmentIndexWriter =
            SingleSegmentIndexWriter::new(index, PENDING_SEGMENT_BUDGET)?;
        let mut added = Corpus::default();
        for (document, tokens) in documents {
            added.add(&tokens);
            writer.add_document(document)?;
        }
        let index = writer.finalize()?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        Ok(PendingDocuments {
            superseded,
            segment: Some(reader.searcher()),
            added,
        })
    }

    /// Run `query` over the index and `pending`, best score first (ties by
    /// node id), with snippets highlighting what `highlight` says, reporting
    /// the documents it reads and scores to `meter`, which can stop it.
    pub(crate) fn collect<M: Meter>(
        &self,
        query: &dyn Query,
        matches: Matches,
        pending: &PendingDocuments,
        highlight: Highlight<'_>,
        meter: &mut M,
    ) -> Result<Vec<HighlightedResult>, TextSearchError>
    where
        TextSearchError: From<M::Stop>,
    {
        let searcher = self.reader.searcher();
        // One corpus for both sides, so their scores rank on one scale and a
        // term only the pending documents hold still scores there: the
        // index's live documents, less the superseded ones as this same
        // searcher holds them, plus the pending ones.
        let mut removed = Corpus::default();
        if let (Some(_), Superseded::Nodes(nodes)) = (&self.corpus, &pending.superseded) {
            for node_id in nodes {
                meter.work(1)?;
                if let Some(Some(tokens)) = self.live_tokens(&searcher, *node_id)? {
                    removed.add(&tokens);
                }
            }
        }
        let base = match (&pending.superseded, &self.corpus) {
            (Superseded::All, _) => Base::Nothing,
            (_, Some(corpus)) => Base::Exact(corpus),
            (_, None) => Base::Index(&searcher),
        };
        let statistics = CorpusStatistics {
            base,
            searcher: &searcher,
            body: self.body_field,
            added: &pending.added,
            removed: &removed,
        };
        let mut hits = match &pending.superseded {
            Superseded::None => {
                self.hits(&searcher, query, matches, &statistics, highlight, meter)?
            }
            Superseded::Nodes(nodes) => {
                let terms = nodes
                    .iter()
                    .map(|id| Term::from_field_u64(self.node_id_field, *id));
                let current = BooleanQuery::new(vec![
                    (Occur::Must, query.box_clone()),
                    (Occur::MustNot, Box::new(TermSetQuery::new(terms))),
                ]);
                self.hits(&searcher, &current, matches, &statistics, highlight, meter)?
            }
            Superseded::All => Vec::new(),
        };
        if let Some(segment) = &pending.segment {
            hits.extend(self.hits(segment, query, matches, &statistics, highlight, meter)?);
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.node_id.cmp(&b.node_id)));
            if let Matches::Top(n) = matches {
                hits.truncate(n);
            }
        }
        Ok(hits)
    }

    /// The matches of `query` in `searcher`, scored with `statistics`, each
    /// document read and scored reported to `meter`.
    fn hits<M: Meter>(
        &self,
        searcher: &Searcher,
        query: &dyn Query,
        matches: Matches,
        statistics: &CorpusStatistics<'_>,
        highlight: Highlight<'_>,
        meter: &mut M,
    ) -> Result<Vec<HighlightedResult>, TextSearchError>
    where
        TextSearchError: From<M::Stop>,
    {
        let found = match matches {
            Matches::Top(0) => return Ok(Vec::new()),
            // Bounded by `n`, and block-max pruned inside the library.
            Matches::Top(n) => searcher.search_with_statistics_provider(
                query,
                &TopDocs::with_limit(n).order_by_score(),
                statistics,
            )?,
            Matches::All => all_matches(searcher, query, statistics, meter)?,
        };
        // A document is read for its node id; it is kept only for the
        // snippet, whose generation reads it again.
        let keep_documents = highlight != Highlight::Off;
        meter.scratch(
            (found.len() * core::mem::size_of::<(Score, u64, Option<TantivyDocument>)>()) as u64,
        )?;
        let mut docs = Vec::with_capacity(found.len());
        for (score, address) in found {
            meter.work(1)?;
            let doc: TantivyDocument = searcher.doc(address)?;
            if let Some(node_id) = doc.get_first(self.node_id_field).and_then(|v| v.as_u64()) {
                if keep_documents {
                    meter.scratch(document_bytes(&doc))?;
                    docs.push((score, node_id, Some(doc)));
                } else {
                    docs.push((score, node_id, None));
                }
            }
        }
        let snippet_gen = match highlight {
            Highlight::Off => None,
            Highlight::Query => Some(SnippetGenerator::create(searcher, query, self.body_field)?),
            Highlight::NearWords(text) => {
                let near = self.near_words(
                    searcher,
                    text,
                    docs.iter().filter_map(|(_, _, doc)| doc.as_ref()),
                );
                Some(SnippetGenerator::create(searcher, &near, self.body_field)?)
            }
        };
        let mut hits = Vec::with_capacity(docs.len());
        for (score, node_id, doc) in docs {
            let snippet_html = match (&snippet_gen, &doc) {
                (Some(generator), Some(doc)) => {
                    meter.work(1)?;
                    generator.snippet_from_doc(doc).to_html()
                }
                _ => String::new(),
            };
            hits.push(HighlightedResult {
                node_id,
                score,
                snippet_html,
            });
        }
        Ok(hits)
    }

    /// A query of the words in `docs`, as the body's analyzer reads them,
    /// within one edit of a word of `text`: what a snippet highlights for an
    /// edit-1 fuzzy search of `text`. Bounded by the matches being shown.
    fn near_words<'d>(
        &self,
        searcher: &Searcher,
        text: &str,
        docs: impl Iterator<Item = &'d TantivyDocument>,
    ) -> BooleanQuery {
        let typed: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_lowercase)
            .collect();
        let mut near = std::collections::BTreeSet::new();
        if let Ok(mut analyzer) = searcher.index().tokenizer_for_field(self.body_field) {
            for doc in docs {
                for value in doc.get_all(self.body_field) {
                    let Some(body) = value.as_str() else { continue };
                    let mut stream = analyzer.token_stream(body);
                    while let Some(token) = stream.next() {
                        if typed.iter().any(|w| within_one_edit(w, &token.text)) {
                            near.insert(token.text.clone());
                        }
                    }
                }
            }
        }
        BooleanQuery::new(
            near.into_iter()
                .map(|word| {
                    let term = Term::from_field_text(self.body_field, &word);
                    let query: Box<dyn Query> = Box::new(tantivy::query::TermQuery::new(
                        term,
                        tantivy::schema::IndexRecordOption::Basic,
                    ));
                    (Occur::Should, query)
                })
                .collect(),
        )
    }
}

/// Whether `a` becomes `b` by at most one insertion, deletion, substitution
/// or swap of two adjacent characters: the edit-1 distance a fuzzy term
/// query matches with transpositions counted as one edit.
fn within_one_edit(a: &str, b: &str) -> bool {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (short, long) = if a.len() <= b.len() {
        (&a, &b)
    } else {
        (&b, &a)
    };
    if long.len() - short.len() > 1 {
        return false;
    }
    let common = short
        .iter()
        .zip(long.iter())
        .take_while(|(x, y)| x == y)
        .count();
    if common == long.len() {
        return true;
    }
    if short.len() == long.len() {
        // A substitution at `common`, or a swap of `common` and the next.
        let rest_equal = |from: usize| short[from..] == long[from..];
        return rest_equal(common + 1)
            || (common + 1 < short.len()
                && short[common] == long[common + 1]
                && short[common + 1] == long[common]
                && rest_equal(common + 2));
    }
    // One character of the longer word inserted at `common`.
    short[common..] == long[common + 1..]
}

/// What a search's corpus starts from before the pending documents change it.
enum Base<'a> {
    /// The index's exact live-document statistics.
    Exact(&'a Corpus),
    /// Tantivy's own counts, for an index that does not keep exact ones.
    Index(&'a Searcher),
    /// Nothing: the index answers for no node.
    Nothing,
}

/// The corpus a search scores against: the base, less the superseded
/// documents the index holds, plus the pending ones.
struct CorpusStatistics<'a> {
    base: Base<'a>,
    /// The index's searcher, for fields other than the body.
    searcher: &'a Searcher,
    body: Field,
    added: &'a Corpus,
    removed: &'a Corpus,
}

impl CorpusStatistics<'_> {
    fn combine(&self, base: u64, added: u64, removed: u64) -> tantivy::Result<u64> {
        CorpusChange::apply(base, added, removed).ok_or_else(|| {
            tantivy::TantivyError::InternalError(
                "corpus statistics went below zero: the superseded documents are not \
                 in the counted corpus"
                    .to_string(),
            )
        })
    }
}

impl Bm25StatisticsProvider for CorpusStatistics<'_> {
    fn total_num_tokens(&self, field: Field) -> tantivy::Result<u64> {
        if field != self.body {
            return self.searcher.total_num_tokens(field);
        }
        let base = match self.base {
            Base::Exact(corpus) => corpus.tokens,
            Base::Index(searcher) => searcher.total_num_tokens(field)?,
            Base::Nothing => 0,
        };
        self.combine(base, self.added.tokens, self.removed.tokens)
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        let base = match self.base {
            Base::Exact(corpus) => corpus.docs,
            Base::Index(searcher) => Bm25StatisticsProvider::total_num_docs(searcher)?,
            Base::Nothing => 0,
        };
        self.combine(base, self.added.docs, self.removed.docs)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        if term.field() != self.body {
            return self.searcher.doc_freq(term);
        }
        let value = term.value();
        let Some(text) = value.as_str() else {
            return self.searcher.doc_freq(term);
        };
        let count = |corpus: &Corpus| corpus.df.get(text).copied().unwrap_or(0);
        let base = match self.base {
            Base::Exact(corpus) => count(corpus),
            Base::Index(searcher) => searcher.doc_freq(term)?,
            Base::Nothing => 0,
        };
        self.combine(base, count(self.added), count(self.removed))
    }
}

/// Every match of `query` in `searcher` with its score, best first (ties in
/// segment and document order), scored with `statistics`.
///
/// The segments' scorers are walked here rather than through a collector,
/// whose callback cannot stop the walk: every document the scorer visits is
/// reported to `meter`, so a query's deadline or cancellation stops it
/// inside a segment, and the list of matches is charged as it grows.
fn all_matches<M: Meter>(
    searcher: &Searcher,
    query: &dyn Query,
    statistics: &dyn Bm25StatisticsProvider,
    meter: &mut M,
) -> Result<Vec<(Score, DocAddress)>, TextSearchError>
where
    TextSearchError: From<M::Stop>,
{
    let weight = query.weight(EnableScoring::enabled_from_statistics_provider(
        statistics, searcher,
    ))?;
    let mut found: Vec<(Score, DocAddress)> = Vec::new();
    for (ordinal, reader) in searcher.segment_readers().iter().enumerate() {
        // A scorer over a term range or an automaton (prefix, fuzzy) holds a
        // bitset of the whole segment while it runs.
        meter.scratch(u64::from(reader.max_doc()).div_ceil(8))?;
        let segment = SegmentOrdinal::try_from(ordinal).map_err(|_| {
            TextSearchError::IndexCorrupted(format!("segment ordinal {ordinal} out of range"))
        })?;
        let alive = reader.alive_bitset();
        let mut scorer = weight.scorer(reader, 1.0)?;
        let mut doc: DocId = scorer.doc();
        while doc != TERMINATED {
            meter.work(1)?;
            if alive.is_none_or(|alive| alive.is_alive(doc)) {
                if found.len() == found.capacity() {
                    // The list doubles; the growth is charged before it is
                    // made.
                    let more = found.capacity().max(64);
                    meter.scratch((more * core::mem::size_of::<(Score, DocAddress)>()) as u64)?;
                    found.reserve_exact(more);
                }
                found.push((scorer.score(), DocAddress::new(segment, doc)));
            }
            doc = scorer.advance();
        }
    }
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(found)
}

/// What a stored document holds, for the scratch a search keeps it in.
fn document_bytes(doc: &TantivyDocument) -> u64 {
    use tantivy::schema::OwnedValue;
    doc.field_values()
        .map(|(_, value)| {
            let owned: OwnedValue = value.into();
            let payload = match &owned {
                OwnedValue::Str(text) => text.len(),
                OwnedValue::Bytes(bytes) => bytes.len(),
                _ => 0,
            };
            (core::mem::size_of::<OwnedValue>() + payload) as u64
        })
        .sum()
}
