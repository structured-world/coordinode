//! Searching an index together with the documents it has not caught up with.
//!
//! An index maintained behind its source holds stale or missing documents for
//! the nodes written since its position. A search leaves the index's
//! documents of those nodes out and runs the same query over a transient
//! segment of their current documents, scored with the index's corpus
//! statistics so the two result sets rank on one scale.

use tantivy::collector::{Collector, SegmentCollector, TopDocs};
use tantivy::query::{Bm25StatisticsProvider, BooleanQuery, Occur, Query, TermSetQuery};
use tantivy::schema::Field;
use tantivy::schema::document::Value as _;
use tantivy::snippet::SnippetGenerator;
use tantivy::{
    DocAddress, DocId, Index, ReloadPolicy, Score, Searcher, SegmentOrdinal, SegmentReader,
    SingleSegmentIndexWriter, TantivyDocument, Term,
};

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
    Nodes(Vec<Term>),
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
        documents: Vec<TantivyDocument>,
    ) -> Result<PendingDocuments, TextSearchError> {
        let superseded = match superseded {
            None => Superseded::All,
            Some([]) => Superseded::None,
            Some(nodes) => Superseded::Nodes(
                nodes
                    .iter()
                    .map(|id| Term::from_field_u64(self.node_id_field, *id))
                    .collect(),
            ),
        };
        if documents.is_empty() {
            return Ok(PendingDocuments {
                superseded,
                segment: None,
            });
        }
        let mut index = Index::create_in_ram(self.schema.clone());
        // The same analyzers, for snippets of the pending documents.
        index.set_tokenizers(self.index.tokenizers().clone());
        let mut writer: SingleSegmentIndexWriter =
            SingleSegmentIndexWriter::new(index, PENDING_SEGMENT_BUDGET)?;
        for document in documents {
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
        })
    }

    /// Run `query` over the index and `pending`, best score first (ties by
    /// node id), with highlighted snippets when `snippets` is set.
    pub(crate) fn collect(
        &self,
        query: &dyn Query,
        matches: Matches,
        pending: &PendingDocuments,
        snippets: bool,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        let searcher = self.reader.searcher();
        // One corpus for both sides, so their scores rank on one scale and a
        // term only the pending documents hold still scores there.
        let statistics = CorpusStatistics {
            index: (!matches!(pending.superseded, Superseded::All)).then_some(&searcher),
            pending: pending.segment.as_ref(),
        };
        let mut hits = match &pending.superseded {
            Superseded::None => self.hits(&searcher, query, matches, &statistics, snippets)?,
            Superseded::Nodes(terms) => {
                let current = BooleanQuery::new(vec![
                    (Occur::Must, query.box_clone()),
                    (
                        Occur::MustNot,
                        Box::new(TermSetQuery::new(terms.iter().cloned())),
                    ),
                ]);
                self.hits(&searcher, &current, matches, &statistics, snippets)?
            }
            Superseded::All => Vec::new(),
        };
        if let Some(segment) = &pending.segment {
            hits.extend(self.hits(segment, query, matches, &statistics, snippets)?);
            hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.node_id.cmp(&b.node_id)));
            if let Matches::Top(n) = matches {
                hits.truncate(n);
            }
        }
        Ok(hits)
    }

    /// The matches of `query` in `searcher`, scored with `statistics`.
    fn hits(
        &self,
        searcher: &Searcher,
        query: &dyn Query,
        matches: Matches,
        statistics: &CorpusStatistics<'_>,
        snippets: bool,
    ) -> Result<Vec<HighlightedResult>, TextSearchError> {
        let found = match matches {
            Matches::Top(0) => return Ok(Vec::new()),
            Matches::Top(n) => searcher.search_with_statistics_provider(
                query,
                &TopDocs::with_limit(n).order_by_score(),
                statistics,
            )?,
            Matches::All => {
                searcher.search_with_statistics_provider(query, &AllMatches, statistics)?
            }
        };
        let snippet_gen = if snippets {
            Some(SnippetGenerator::create(searcher, query, self.body_field)?)
        } else {
            None
        };
        let mut hits = Vec::with_capacity(found.len());
        for (score, address) in found {
            let doc: TantivyDocument = searcher.doc(address)?;
            let Some(node_id) = doc.get_first(self.node_id_field).and_then(|v| v.as_u64()) else {
                continue;
            };
            let snippet_html = snippet_gen
                .as_ref()
                .map(|g| g.snippet_from_doc(&doc).to_html())
                .unwrap_or_default();
            hits.push(HighlightedResult {
                node_id,
                score,
                snippet_html,
            });
        }
        Ok(hits)
    }
}

/// The corpus a search scores against: the index's documents (unless the
/// index answers for no node) and the pending ones. A superseded document
/// still counts in the index's part until the index replaces it, as a
/// deleted document counts in Tantivy until its segment merges.
struct CorpusStatistics<'a> {
    index: Option<&'a Searcher>,
    pending: Option<&'a Searcher>,
}

impl CorpusStatistics<'_> {
    fn sum(&self, of: impl Fn(&Searcher) -> tantivy::Result<u64>) -> tantivy::Result<u64> {
        let mut total = 0u64;
        for searcher in [self.index, self.pending].into_iter().flatten() {
            // Two document counts of one in-memory index cannot reach 2^64.
            total += of(searcher)?;
        }
        Ok(total)
    }
}

impl Bm25StatisticsProvider for CorpusStatistics<'_> {
    fn total_num_tokens(&self, field: Field) -> tantivy::Result<u64> {
        self.sum(|s| s.total_num_tokens(field))
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        self.sum(Bm25StatisticsProvider::total_num_docs)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        self.sum(|s| s.doc_freq(term))
    }
}

/// Every match with its score, best first.
struct AllMatches;

impl Collector for AllMatches {
    type Fruit = Vec<(Score, DocAddress)>;
    type Child = AllMatchesInSegment;

    fn for_segment(
        &self,
        segment: SegmentOrdinal,
        _reader: &SegmentReader,
    ) -> tantivy::Result<Self::Child> {
        Ok(AllMatchesInSegment {
            segment,
            found: Vec::new(),
        })
    }

    fn requires_scoring(&self) -> bool {
        true
    }

    fn merge_fruits(
        &self,
        segments: Vec<Vec<(Score, DocAddress)>>,
    ) -> tantivy::Result<Self::Fruit> {
        let mut all: Vec<(Score, DocAddress)> = segments.into_iter().flatten().collect();
        all.sort_by(|a, b| b.0.total_cmp(&a.0));
        Ok(all)
    }
}

struct AllMatchesInSegment {
    segment: SegmentOrdinal,
    found: Vec<(Score, DocAddress)>,
}

impl SegmentCollector for AllMatchesInSegment {
    type Fruit = Vec<(Score, DocAddress)>;

    fn collect(&mut self, doc: DocId, score: Score) {
        self.found.push((score, DocAddress::new(self.segment, doc)));
    }

    fn harvest(self) -> Self::Fruit {
        self.found
    }
}
