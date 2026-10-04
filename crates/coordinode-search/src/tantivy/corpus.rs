//! Exact corpus statistics of a text index.
//!
//! BM25 reads the number of documents, the total number of tokens and each
//! term's document frequency. Tantivy's own counts include deleted documents
//! until their segment merges, so an index that replaces documents in place
//! would score against documents it no longer holds. [`Corpus`] counts the
//! live documents only, kept in step with every commit of the index.

use std::collections::HashMap;

/// Live-document statistics of one indexed field.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Corpus {
    pub(crate) docs: u64,
    pub(crate) tokens: u64,
    /// Documents holding each term, by the term's text.
    pub(crate) df: HashMap<String, u64>,
}

impl Corpus {
    /// Count a document whose field tokenized to `tokens` (one entry per
    /// position).
    pub(crate) fn add(&mut self, tokens: &[String]) {
        self.docs += 1;
        self.tokens += tokens.len() as u64;
        for term in distinct(tokens) {
            *self.df.entry(term.to_string()).or_default() += 1;
        }
    }

    /// Stop counting a document [`Self::add`] counted with `tokens`.
    pub(crate) fn remove(&mut self, tokens: &[String]) {
        // A document is only removed after it was added, so no count can go
        // below zero; a violation is a bookkeeping bug.
        debug_assert!(self.docs > 0, "removing a document never counted");
        self.docs -= 1;
        self.tokens -= tokens.len() as u64;
        for term in distinct(tokens) {
            match self.df.get_mut(term) {
                Some(n) if *n > 1 => *n -= 1,
                Some(_) => {
                    self.df.remove(term);
                }
                None => debug_assert!(false, "term {term} never counted"),
            }
        }
    }
}

/// The distinct texts among `tokens`.
fn distinct(tokens: &[String]) -> impl Iterator<Item = &str> {
    let mut seen: Vec<&str> = tokens.iter().map(String::as_str).collect();
    seen.sort_unstable();
    seen.dedup();
    seen.into_iter()
}

/// Statistics added to and taken from a base corpus: the documents a search
/// reads in place of the index's own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct CorpusChange {
    pub(crate) added: Corpus,
    pub(crate) removed: Corpus,
}

impl CorpusChange {
    /// `base` with this change applied, for one statistic.
    pub(crate) fn apply(base: u64, added: u64, removed: u64) -> Option<u64> {
        base.checked_add(added)?.checked_sub(removed)
    }
}

#[cfg(test)]
mod tests;
