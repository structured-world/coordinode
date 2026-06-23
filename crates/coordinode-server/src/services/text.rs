//! TextService gRPC implementation.
//!
//! Provides full-text search over tantivy-indexed text properties.
//! Searches all text-indexed properties for the requested label, merges
//! results by node_id (keeping the highest BM25 score across properties),
//! and returns scored results with optional HTML-highlighted snippets.
//!
//! # Cluster notes
//! TextIndex is per-shard/per-node in both CE and EE. In CE 3-node HA,
//! each node has an independent tantivy index for its local data.
//! In EE, scatter-gather across shards is handled by the EE coordinator.
//! This service operates on the local node's index only — correct for CE.

use std::collections::HashMap;
use std::sync::Arc;

// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use tonic::{Request, Response, Status};

use coordinode_embed::Database;

use crate::proto::query;

/// Default result limit when the request specifies `limit = 0`.
const DEFAULT_TEXT_LIMIT: usize = 10;
/// Hard cap to prevent runaway queries regardless of request limit.
const MAX_TEXT_LIMIT: usize = 1000;

pub struct TextServiceImpl {
    database: Arc<RwLock<Database>>,
}

impl TextServiceImpl {
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self { database }
    }
}

/// Prepare a tantivy query string.
///
/// Currently a pass-through: fuzzy expansion is handled at the QueryParser level
/// via `set_field_fuzzy` in `search_with_highlights_fuzzy`, NOT by appending `~1`
/// to query terms. tantivy's `term~N` syntax is PHRASE SLOP (for `"phrase"~N`),
/// not term fuzzy — appending `~1` to bare terms would be silently misinterpreted.
fn prepare_query(query: &str, _fuzzy: bool) -> String {
    query.to_string()
}

#[tonic::async_trait]
impl query::text_service_server::TextService for TextServiceImpl {
    async fn text_search(
        &self,
        request: Request<query::TextSearchRequest>,
    ) -> Result<Response<query::TextSearchResponse>, Status> {
        let req = request.into_inner();

        if req.label.is_empty() {
            return Err(Status::invalid_argument("label is required"));
        }
        if req.query.is_empty() {
            return Err(Status::invalid_argument("query is required"));
        }

        let limit = match req.limit as usize {
            0 => DEFAULT_TEXT_LIMIT,
            n if n > MAX_TEXT_LIMIT => MAX_TEXT_LIMIT,
            n => n,
        };

        let db = self.database.read();

        let registry = db.text_index_registry();

        // Collect all text-indexed properties for the label.
        // Each IndexDefinition covers one canonical (label, property) entry.
        let indexed_properties: Vec<String> = registry
            .definitions()
            .into_iter()
            .filter(|d| d.label == req.label)
            .flat_map(|d| d.properties)
            .collect();

        if indexed_properties.is_empty() {
            // No text index for this label — return empty results.
            // Mirrors graceful degradation of text_match() in the Cypher executor.
            tracing::debug!(
                label = %req.label,
                "text_search: no text index registered for label"
            );
            return Ok(Response::new(query::TextSearchResponse { results: vec![] }));
        }

        let effective_query = prepare_query(&req.query, req.fuzzy);
        // Over-fetch per property so the final merge can still produce `limit` items
        // even if some properties contain overlapping node IDs.
        let per_property_limit = (limit * 2).max(20);

        // node_id → best BM25 score across all searched properties.
        let mut node_scores: HashMap<u64, f32> = HashMap::new();
        // node_id → HTML snippet from the property with the highest score.
        let mut node_snippets: HashMap<u64, String> = HashMap::new();

        for property in &indexed_properties {
            let handle = match registry.get(&req.label, property) {
                Some(h) => h,
                None => continue,
            };

            let idx_guard = handle
                .read()
                .map_err(|_| Status::internal("text index read lock poisoned"))?;

            // Search path selection:
            //
            // A. Non-fuzzy + default language → language-aware highlights path.
            //    Uses the MultiLanguageTextIndex language pipeline for correct
            //    stemming consistency (index time == query time).
            //
            // B. Fuzzy → QueryParser path (via inner().search_with_highlights).
            //    tantivy's `term~1` syntax is a QueryParser feature; our language-
            //    aware tokenizers do not understand it. The QueryParser uses the
            //    schema-level "none" tokenizer (whitespace + lowercase), so fuzzy
            //    expansion works on the unstemmed indexed forms. Note: for stemmed
            //    languages (e.g. English), the fuzzy typo must be within edit-1 of
            //    the stemmed term, not the original word.
            //
            // C. Explicit language (non-fuzzy) → language-specific search path.
            //    Uses the requested language for correct stemming; no snippets.
            if !req.fuzzy && req.language.is_empty() {
                // Path A: non-fuzzy default language — correct stemming + snippets.
                match idx_guard.search_with_highlights(&effective_query, per_property_limit) {
                    Ok(results) => {
                        for r in results {
                            let current = node_scores.entry(r.node_id).or_insert(0.0_f32);
                            if r.score > *current {
                                *current = r.score;
                                node_snippets.insert(r.node_id, r.snippet_html);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            label = %req.label,
                            property = %property,
                            "text_search highlight error: {e}"
                        );
                    }
                }
            } else if req.fuzzy {
                // Path B: fuzzy via QueryParser set_field_fuzzy.
                // Uses search_with_highlights_fuzzy which enables Levenshtein-1 expansion
                // at the QueryParser level (NOT via `~1` suffix which is phrase slop).
                match idx_guard
                    .inner()
                    .search_with_highlights_fuzzy(&effective_query, per_property_limit)
                {
                    Ok(results) => {
                        for r in results {
                            let current = node_scores.entry(r.node_id).or_insert(0.0_f32);
                            if r.score > *current {
                                *current = r.score;
                                node_snippets.insert(r.node_id, r.snippet_html);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            label = %req.label,
                            property = %property,
                            "text_search fuzzy error: {e}"
                        );
                    }
                }
            } else {
                // Path C: explicit language, non-fuzzy — correct per-language stemming.
                // Snippets not available (language-specific path uses direct TermQuery
                // construction, not tantivy QueryParser, so SnippetGenerator cannot
                // be used with the same query object).
                match idx_guard.search_with_language(
                    &effective_query,
                    per_property_limit,
                    &req.language,
                ) {
                    Ok(results) => {
                        for r in results {
                            let current = node_scores.entry(r.node_id).or_insert(0.0_f32);
                            if r.score > *current {
                                *current = r.score;
                                // Snippet stays empty for explicit-language searches.
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            label = %req.label,
                            property = %property,
                            "text_search language search error: {e}"
                        );
                    }
                }
            }
        }

        // Sort by score descending; take top `limit`.
        let mut results: Vec<(u64, f32)> = node_scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);

        let proto_results = results
            .into_iter()
            .map(|(node_id, score)| query::TextResult {
                node_id,
                score,
                snippet: node_snippets.remove(&node_id).unwrap_or_default(),
            })
            .collect();

        Ok(Response::new(query::TextSearchResponse {
            results: proto_results,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
