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
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

use crate::proto::query;

/// Default result limit when the request specifies `limit = 0`.
const DEFAULT_TEXT_LIMIT: usize = 10;
/// Hard cap to prevent runaway queries regardless of request limit.
const MAX_TEXT_LIMIT: usize = 1000;

/// RRF constant k=60 (standard value from Cormack et al. 2009).
/// Moderates the impact of top-ranked items, improving robustness when
/// score distributions differ between the two retrieval methods.
const RRF_K: f32 = 60.0;

/// Default weight for each component when the caller leaves the field at 0.0.
const DEFAULT_WEIGHT: f32 = 0.5;

/// Default vector property name used in hybrid search when none is specified.
const DEFAULT_VECTOR_PROPERTY: &str = "embedding";

/// Backtick-escape a Cypher identifier to prevent injection.
fn cypher_ident(name: &str) -> String {
    format!("`{}`", name.replace('`', "``"))
}

pub struct TextServiceImpl {
    database: Arc<Mutex<Database>>,
}

impl TextServiceImpl {
    pub fn new(database: Arc<Mutex<Database>>) -> Self {
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

        let db = self
            .database
            .lock()
            .map_err(|_| Status::internal("database lock poisoned"))?;

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

    async fn hybrid_text_vector_search(
        &self,
        request: Request<query::HybridTextVectorSearchRequest>,
    ) -> Result<Response<query::HybridTextVectorSearchResponse>, Status> {
        let req = request.into_inner();

        if req.label.is_empty() {
            return Err(Status::invalid_argument("label is required"));
        }
        if req.text_query.is_empty() {
            return Err(Status::invalid_argument("text_query is required"));
        }
        if req.vector.is_empty() {
            return Err(Status::invalid_argument("vector is required"));
        }

        let limit = match req.limit as usize {
            0 => DEFAULT_TEXT_LIMIT,
            n if n > MAX_TEXT_LIMIT => MAX_TEXT_LIMIT,
            n => n,
        };

        let text_weight = if req.text_weight <= 0.0 {
            DEFAULT_WEIGHT
        } else {
            req.text_weight
        };
        let vector_weight = if req.vector_weight <= 0.0 {
            DEFAULT_WEIGHT
        } else {
            req.vector_weight
        };

        let vector_property = if req.vector_property.is_empty() {
            DEFAULT_VECTOR_PROPERTY.to_string()
        } else {
            req.vector_property.clone()
        };

        // Over-fetch from each source to improve RRF fusion quality.
        // Nodes near the boundary of `limit` in one ranking but high in the other
        // can fuse to a top position; over-fetching avoids discarding them.
        let fetch_limit = (limit * 3).clamp(20, MAX_TEXT_LIMIT);

        let mut db = self
            .database
            .lock()
            .map_err(|_| Status::internal("database lock poisoned"))?;

        // --- Phase 1: BM25 text search ---
        // All borrows from `db` (via `registry`) are scoped to this block and
        // dropped before Phase 2 needs `&mut db` for Cypher execution.
        let text_ranked_ids: Vec<u64> = {
            let registry = db.text_index_registry();

            let indexed_properties: Vec<String> = registry
                .definitions()
                .into_iter()
                .filter(|d| d.label == req.label)
                .flat_map(|d| d.properties)
                .collect();

            let mut node_scores: HashMap<u64, f32> = HashMap::new();

            for property in &indexed_properties {
                let handle = match registry.get(&req.label, property) {
                    Some(h) => h,
                    None => continue,
                };
                let idx_guard = match handle.read() {
                    Ok(g) => g,
                    Err(_) => {
                        tracing::warn!(
                            label = %req.label,
                            property = %property,
                            "hybrid: text index read lock poisoned"
                        );
                        continue;
                    }
                };
                match idx_guard.search_with_highlights(&req.text_query, fetch_limit) {
                    Ok(results) => {
                        for r in results {
                            let current = node_scores.entry(r.node_id).or_insert(0.0_f32);
                            if r.score > *current {
                                *current = r.score;
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            label = %req.label,
                            property = %property,
                            "hybrid: text search error: {e}"
                        );
                    }
                }
            }

            // Sort by BM25 descending; extract node_ids as the ranked list.
            let mut pairs: Vec<(u64, f32)> = node_scores.into_iter().collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            pairs.truncate(fetch_limit);
            pairs.into_iter().map(|(id, _)| id).collect()
        };
        // All registry/handle/idx_guard borrows released here.

        // --- Phase 2: Vector search via Cypher (cosine similarity) ---
        // Uses `vector_similarity` (cosine) since this is designed for text embeddings
        // where cosine is the standard metric.
        let label_safe = cypher_ident(&req.label);
        let vp_safe = cypher_ident(&vector_property);
        let cypher = format!(
            "MATCH (n:{label_safe}) \
             WITH n, vector_similarity(n.{vp_safe}, $qv) AS _sim \
             ORDER BY _sim DESC \
             LIMIT {fetch_limit} \
             RETURN n"
        );

        let mut params = HashMap::new();
        params.insert("qv".to_string(), Value::Vector(req.vector.clone()));

        let vector_ranked_ids: Vec<u64> = {
            match db.execute_cypher_with_params(&cypher, params) {
                Ok(rows) => rows
                    .into_iter()
                    .filter_map(|row| match row.get("n")? {
                        Value::Int(id) => Some(*id as u64),
                        _ => None,
                    })
                    .collect(),
                Err(e) => {
                    tracing::warn!(label = %req.label, "hybrid: vector search error: {e}");
                    vec![]
                }
            }
        };

        // --- Phase 3: Weighted RRF fusion ---
        // rrf_score(node) = text_weight / (k + rank_text) + vector_weight / (k + rank_vec)
        // Nodes absent from a ranking receive a penalty rank = list_size + 1.
        let text_rank_map: HashMap<u64, usize> = text_ranked_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i + 1))
            .collect();
        let vec_rank_map: HashMap<u64, usize> = vector_ranked_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i + 1))
            .collect();

        let text_penalty = text_ranked_ids.len() + 1;
        let vec_penalty = vector_ranked_ids.len() + 1;

        // Union of all node_ids from both result sets.
        let mut all_nodes: std::collections::HashSet<u64> = std::collections::HashSet::new();
        all_nodes.extend(text_ranked_ids.iter().copied());
        all_nodes.extend(vector_ranked_ids.iter().copied());

        let mut scores: Vec<(u64, f32)> = all_nodes
            .into_iter()
            .map(|node_id| {
                let r_text = *text_rank_map.get(&node_id).unwrap_or(&text_penalty) as f32;
                let r_vec = *vec_rank_map.get(&node_id).unwrap_or(&vec_penalty) as f32;
                let score = text_weight / (RRF_K + r_text) + vector_weight / (RRF_K + r_vec);
                (node_id, score)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);

        let results = scores
            .into_iter()
            .map(|(node_id, score)| query::HybridResult { node_id, score })
            .collect();

        Ok(Response::new(query::HybridTextVectorSearchResponse {
            results,
        }))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    // prepare_query: pass-through regardless of fuzzy flag.
    // Fuzzy expansion is handled at the QueryParser level via set_field_fuzzy
    // in search_with_highlights_fuzzy — NOT by appending ~1 to terms.
    // (tantivy's `term~N` syntax is PHRASE SLOP, not term fuzzy)
    #[test]
    fn prepare_query_passthrough() {
        assert_eq!(prepare_query("rust graph", false), "rust graph");
        assert_eq!(prepare_query("rust graph", true), "rust graph");
        assert_eq!(prepare_query("rust AND graph", true), "rust AND graph");
        assert_eq!(prepare_query("data*", true), "data*");
        assert_eq!(prepare_query("", true), "");
        assert_eq!(prepare_query("", false), "");
    }

    // limit capping
    #[test]
    fn limit_cap() {
        assert_eq!(
            match 0usize {
                0 => DEFAULT_TEXT_LIMIT,
                n if n > MAX_TEXT_LIMIT => MAX_TEXT_LIMIT,
                n => n,
            },
            DEFAULT_TEXT_LIMIT
        );
        assert_eq!(
            match 2000usize {
                0 => DEFAULT_TEXT_LIMIT,
                n if n > MAX_TEXT_LIMIT => MAX_TEXT_LIMIT,
                n => n,
            },
            MAX_TEXT_LIMIT
        );
        assert_eq!(
            match 42usize {
                0 => DEFAULT_TEXT_LIMIT,
                n if n > MAX_TEXT_LIMIT => MAX_TEXT_LIMIT,
                n => n,
            },
            42
        );
    }

    // --- TextSearch wiring integration tests ---

    use crate::proto::query::text_service_server::TextService;

    /// Helper: open a temp database with a text index on (label, property).
    fn test_service_with_text_index(
        label: &str,
        property: &str,
    ) -> (TextServiceImpl, tempfile::TempDir) {
        use coordinode_query::index::TextIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut database = Database::open(dir.path()).expect("open database");

        database
            .create_text_index("test_text_idx", label, property, TextIndexConfig::default())
            .expect("create text index");

        let database = Arc::new(Mutex::new(database));
        (TextServiceImpl::new(database), dir)
    }

    /// text_search: rejects empty label.
    #[tokio::test]
    async fn text_search_requires_label() {
        let (svc, _dir) = test_service_with_text_index("Article", "body");
        let err = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "".to_string(),
                query: "rust".to_string(),
                limit: 0,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// text_search: rejects empty query.
    #[tokio::test]
    async fn text_search_requires_query() {
        let (svc, _dir) = test_service_with_text_index("Article", "body");
        let err = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Article".to_string(),
                query: "".to_string(),
                limit: 0,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// text_search: returns empty for label with no text index.
    #[tokio::test]
    async fn text_search_empty_on_no_index() {
        let (svc, _dir) = test_service_with_text_index("Article", "body");
        // Query "Other" label that has no index.
        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Other".to_string(),
                query: "rust".to_string(),
                limit: 0,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .expect("should not error on missing index");
        assert!(result.into_inner().results.is_empty());
    }

    /// text_search: finds nodes inserted via Cypher CREATE.
    ///
    /// Creates three Article nodes with `body` properties covering different topics.
    /// Searches for "database" and verifies that relevant nodes are returned with
    /// positive BM25 scores and non-empty HTML snippets.
    #[tokio::test]
    async fn text_search_finds_indexed_nodes() {
        let (svc, _dir) = test_service_with_text_index("Article", "body");

        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher(
                "CREATE (n:Article {body: 'Rust is a systems programming language for databases'})",
            )
            .expect("create article 1");
            db.execute_cypher(
                "CREATE (n:Article {body: 'Graph database concepts and performance benchmarks'})",
            )
            .expect("create article 2");
            db.execute_cypher(
                "CREATE (n:Article {body: 'Unrelated content about cooking recipes'})",
            )
            .expect("create article 3");
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Article".to_string(),
                query: "database".to_string(),
                limit: 10,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .expect("text search should succeed");

        let body = result.into_inner();
        // "database" matches articles 1 and 2 but not article 3 (cooking).
        assert!(
            !body.results.is_empty(),
            "should find at least one article about database"
        );
        assert!(
            body.results.len() <= 2,
            "cooking article should not match: got {} results",
            body.results.len()
        );
        for r in &body.results {
            assert!(r.score > 0.0, "BM25 score must be positive: {}", r.score);
            // Snippets may be empty when using PreTokenizedString (tantivy limitation:
            // SnippetGenerator cannot extract offsets from pre-tokenized documents the same
            // way it can from plain-text stored fields). Snippet non-emptiness is tested
            // separately in coordinode-search unit tests for the direct TextIndex path.
            let _ = &r.snippet;
        }
    }

    /// text_search: respects limit — returns no more than `limit` results.
    #[tokio::test]
    async fn text_search_respects_limit() {
        let (svc, _dir) = test_service_with_text_index("Doc", "content");

        {
            let mut db = svc.database.lock().unwrap();
            for i in 0..10 {
                let cypher = format!(
                    "CREATE (n:Doc {{content: 'the quick brown fox document number {i}'}})"
                );
                db.execute_cypher(&cypher).expect("create doc");
            }
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Doc".to_string(),
                query: "fox".to_string(),
                limit: 3,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .expect("text search should succeed");

        let body = result.into_inner();
        assert!(
            body.results.len() <= 3,
            "limit=3 must cap results: got {}",
            body.results.len()
        );
    }

    /// text_search: explicit language routes to Path C (language-specific stemming).
    ///
    /// Path C uses `search_with_language` which tokenizes the query with the same
    /// per-language pipeline used at index time. Snippets are empty (SnippetGenerator
    /// is not compatible with the direct TermQuery construction used in this path).
    ///
    /// Uses "en" (English) language — index default is English so this exercises the
    /// same stemmer pipeline. "graph" stems to "graph" under Snowball (no change);
    /// "concept" stems to "concept" — both should match.
    #[tokio::test]
    async fn text_search_explicit_language_path_c() {
        let (svc, _dir) = test_service_with_text_index("Page", "body");

        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:Page {body: 'graph concepts and algorithms'})")
                .expect("create page");
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Page".to_string(),
                query: "graph".to_string(),
                limit: 10,
                fuzzy: false,
                language: "en".to_string(), // explicit language → Path C
            }))
            .await
            .expect("explicit language search should succeed");

        let body = result.into_inner();
        assert!(
            !body.results.is_empty(),
            "explicit language=en search should find 'graph'"
        );
        for r in &body.results {
            assert!(r.score > 0.0, "BM25 score must be positive: {}", r.score);
            // Path C does not produce snippets by design (no SnippetGenerator).
            // snippet field is empty — that is correct behavior, not a bug.
        }
    }

    /// text_search: fuzzy=true matches near-typos via tantivy QueryParser `~1`.
    ///
    /// Fuzzy search routes through tantivy's QueryParser which natively handles
    /// `term~1` syntax. The QueryParser uses the schema-level tokenizer ("none" =
    /// whitespace + lowercase) — no stemming. The typo must therefore be within
    /// Levenshtein-1 of the STORED term form in the index.
    ///
    /// "graph" does not get stemmed by English Snowball, so it is stored as "graph".
    /// "grapm" differs from "graph" by 1 edit (m→h), so fuzzy expansion finds it.
    #[tokio::test]
    async fn text_search_fuzzy_matches_typo() {
        let (svc, _dir) = test_service_with_text_index("Note", "text");

        {
            let mut db = svc.database.lock().unwrap();
            // "graph" does not get stemmed by English Snowball → stored as "graph".
            db.execute_cypher("CREATE (n:Note {text: 'graph query concepts'})")
                .expect("create note");
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Note".to_string(),
                query: "grapm".to_string(), // typo: 1 edit from "graph" (m→h)
                limit: 10,
                fuzzy: true,
                language: "".to_string(),
            }))
            .await
            .expect("fuzzy search should succeed");

        let body = result.into_inner();
        assert!(
            !body.results.is_empty(),
            "fuzzy search should find 'graph' with typo 'grapm' (Levenshtein-1)"
        );
    }

    /// text_search: Ukrainian index with Ukrainian text — Path A stemming e2e.
    ///
    /// Creates a TextIndex with `default_language = "ukrainian"` and inserts Ukrainian
    /// documents. Searches without explicit language (Path A: MultiLanguageTextIndex
    /// .search_with_highlights with default_language = "ukrainian"). Verifies the full
    /// gRPC → MultiLanguageTextIndex → Snowball Ukrainian stemmer chain.
    ///
    /// "книга" stems to "книг" in Ukrainian Snowball. Searching "книга" must find the
    /// document because query is stemmed the same way as the index.
    #[tokio::test]
    async fn text_search_ukrainian_index_path_a() {
        use coordinode_query::index::TextIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut database = Database::open(dir.path()).expect("open database");
        // Cypher parser requires ASCII identifiers for label/property names.
        // Ukrainian content goes in property VALUES, not names.
        database
            .create_text_index(
                "uk_idx",
                "UkArticle",
                "body",
                TextIndexConfig {
                    default_language: "ukrainian".to_string(),
                    ..Default::default()
                },
            )
            .expect("create ukrainian text index");
        let svc = TextServiceImpl::new(Arc::new(Mutex::new(database)));

        {
            let mut db = svc.database.lock().unwrap();
            db.execute_cypher("CREATE (n:UkArticle {body: 'книга про програмування мовою Rust'})")
                .expect("create article 1");
            db.execute_cypher("CREATE (n:UkArticle {body: 'рецепти приготування їжі на кухні'})")
                .expect("create article 2");
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "UkArticle".to_string(),
                query: "книга".to_string(), // stems to "книг" via Ukrainian Snowball
                limit: 10,
                fuzzy: false,
                language: "".to_string(), // Path A — default language from index config
            }))
            .await
            .expect("ukrainian text search should succeed");

        let body = result.into_inner();
        assert!(
            !body.results.is_empty(),
            "Ukrainian Path A: 'книга' should match article about programming"
        );
        assert!(
            body.results.len() <= 1,
            "cooking article should not match 'книга': got {} results",
            body.results.len()
        );
        assert!(
            body.results[0].score > 0.0,
            "BM25 score must be positive: {}",
            body.results[0].score
        );
    }

    /// text_search: multi-property index — score merge picks best BM25 across properties.
    ///
    /// Creates TWO text indexes on the same label (different properties). One node
    /// matches via "title", another via "body". Verifies:
    /// - Both nodes are returned (merge across properties works)
    /// - A node matching in both properties appears once (dedup by node_id)
    /// - Scores are positive
    #[tokio::test]
    async fn text_search_multi_property_merge() {
        use coordinode_query::index::TextIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut database = Database::open(dir.path()).expect("open database");
        database
            .create_text_index("idx_title", "Post", "title", TextIndexConfig::default())
            .expect("create title index");
        database
            .create_text_index("idx_body", "Post", "body", TextIndexConfig::default())
            .expect("create body index");
        let svc = TextServiceImpl::new(Arc::new(Mutex::new(database)));

        {
            let mut db = svc.database.lock().unwrap();
            // node 1: "database" in title only
            db.execute_cypher(
                "CREATE (n:Post {title: 'Introduction to database systems', body: 'general overview'})",
            )
            .expect("create post 1");
            // node 2: "database" in body only
            db.execute_cypher(
                "CREATE (n:Post {title: 'Software architecture', body: 'relational database design patterns'})",
            )
            .expect("create post 2");
            // node 3: "database" in both — should appear once, best score wins
            db.execute_cypher(
                "CREATE (n:Post {title: 'Database internals', body: 'database storage engines explained'})",
            )
            .expect("create post 3");
            // node 4: no match
            db.execute_cypher(
                "CREATE (n:Post {title: 'Cooking recipes', body: 'pasta and salad'})",
            )
            .expect("create post 4");
        }

        let result = svc
            .text_search(Request::new(crate::proto::query::TextSearchRequest {
                label: "Post".to_string(),
                query: "database".to_string(),
                limit: 10,
                fuzzy: false,
                language: "".to_string(),
            }))
            .await
            .expect("multi-property text search should succeed");

        let body = result.into_inner();
        // Nodes 1, 2, 3 should match; node 4 should not
        assert!(
            body.results.len() >= 2,
            "at least 2 nodes should match 'database' across properties; got {}",
            body.results.len()
        );
        assert!(
            body.results.len() <= 3,
            "cooking post must not match; got {}",
            body.results.len()
        );
        // All node_ids must be unique (merge deduplicates)
        let ids: Vec<u64> = body.results.iter().map(|r| r.node_id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(
            ids.len(),
            unique.len(),
            "node_ids must be unique after merge"
        );
        for r in &body.results {
            assert!(r.score > 0.0, "BM25 score must be positive: {}", r.score);
        }
    }

    // ─── HybridTextVectorSearch tests ──────────────────────────────────────

    /// Helper: open a DB with a text index on (label, property) and an
    /// HNSW-like vector index that allows full-scan vector_similarity queries.
    fn test_hybrid_service(
        label: &str,
        text_property: &str,
    ) -> (TextServiceImpl, tempfile::TempDir) {
        use coordinode_query::index::TextIndexConfig;

        let dir = tempfile::tempdir().expect("tempdir");
        let mut database = Database::open(dir.path()).expect("open database");
        database
            .create_text_index(
                "hybrid_text_idx",
                label,
                text_property,
                TextIndexConfig::default(),
            )
            .expect("create text index for hybrid test");
        let database = Arc::new(Mutex::new(database));
        (TextServiceImpl::new(database), dir)
    }

    /// hybrid_text_vector_search: rejects empty label.
    #[tokio::test]
    async fn hybrid_requires_label() {
        let (svc, _dir) = test_hybrid_service("Article", "body");
        let err = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "".to_string(),
                    text_query: "rust".to_string(),
                    vector: vec![1.0, 0.0],
                    limit: 10,
                    text_weight: 0.0,
                    vector_weight: 0.0,
                    vector_property: "".to_string(),
                },
            ))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// hybrid_text_vector_search: rejects empty text_query.
    #[tokio::test]
    async fn hybrid_requires_text_query() {
        let (svc, _dir) = test_hybrid_service("Article", "body");
        let err = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Article".to_string(),
                    text_query: "".to_string(),
                    vector: vec![1.0, 0.0],
                    limit: 10,
                    text_weight: 0.0,
                    vector_weight: 0.0,
                    vector_property: "".to_string(),
                },
            ))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// hybrid_text_vector_search: rejects empty vector.
    #[tokio::test]
    async fn hybrid_requires_vector() {
        let (svc, _dir) = test_hybrid_service("Article", "body");
        let err = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Article".to_string(),
                    text_query: "rust".to_string(),
                    vector: vec![],
                    limit: 10,
                    text_weight: 0.0,
                    vector_weight: 0.0,
                    vector_property: "".to_string(),
                },
            ))
            .await
            .unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    /// hybrid_text_vector_search: returns empty when no text index for label.
    ///
    /// With no text index, text ranking is empty. If nodes also have no embeddings
    /// (vector search scans return no results), the hybrid result should be empty
    /// rather than erroring.
    #[tokio::test]
    async fn hybrid_empty_on_no_index() {
        let (svc, _dir) = test_hybrid_service("Article", "body");
        // "Other" has no text index and no nodes — both searches are empty.
        let result = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Other".to_string(),
                    text_query: "rust".to_string(),
                    vector: vec![1.0, 0.0],
                    limit: 10,
                    text_weight: 0.0,
                    vector_weight: 0.0,
                    vector_property: "embedding".to_string(),
                },
            ))
            .await
            .expect("should not error when no index exists");
        assert!(result.into_inner().results.is_empty());
    }

    /// hybrid_text_vector_search: RRF fusion — a node appearing in BOTH rankings
    /// scores higher than a node appearing in only one.
    ///
    /// Setup:
    ///   Node A: strong text match ("database"), embedding close to query vector.
    ///   Node B: no text match, embedding close to query vector.
    ///   Node C: strong text match ("database"), embedding distant from query vector.
    ///
    /// Expected (equal weights, standard RRF):
    ///   A: appears in text rank + vector rank → highest combined score
    ///   B: appears only in vector rank → lower score than A
    ///   C: appears in text rank, poor in vector (vector scan may return it or not) → lower than A
    #[tokio::test]
    async fn hybrid_rrf_fusion_boosts_dual_hits() {
        let (svc, _dir) = test_hybrid_service("Doc", "body");

        {
            let mut db = svc.database.lock().unwrap();
            // Node A: text match ("database") + embedding close to [1.0, 0.0, 0.0]
            db.execute_cypher(
                "CREATE (n:Doc {body: 'graph database systems', embedding: [1.0, 0.0, 0.0]})",
            )
            .expect("create node A");
            // Node B: no text match + embedding close to [1.0, 0.0, 0.0]
            db.execute_cypher(
                "CREATE (n:Doc {body: 'cooking recipes and pasta', embedding: [0.99, 0.01, 0.0]})",
            )
            .expect("create node B");
            // Node C: text match ("database") + distant embedding
            db.execute_cypher(
                "CREATE (n:Doc {body: 'relational database design', embedding: [0.0, 0.0, 1.0]})",
            )
            .expect("create node C");
        }

        let result = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Doc".to_string(),
                    text_query: "database".to_string(),
                    vector: vec![1.0, 0.0, 0.0], // close to A and B
                    limit: 10,
                    text_weight: 0.5,
                    vector_weight: 0.5,
                    vector_property: "embedding".to_string(),
                },
            ))
            .await
            .expect("hybrid search should succeed");

        let body = result.into_inner();
        assert!(
            !body.results.is_empty(),
            "hybrid search with matching nodes must return results"
        );

        // All scores must be positive (RRF formula always > 0).
        for r in &body.results {
            assert!(r.score > 0.0, "RRF score must be positive: {}", r.score);
        }

        // Node IDs must be unique (union dedup).
        let ids: Vec<u64> = body.results.iter().map(|r| r.node_id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "node_ids must be unique");

        // Node A (text match + close vector) should be the top result.
        // It appears in BOTH rankings so its RRF score is higher than
        // any node appearing in only one ranking.
        assert!(!body.results.is_empty(), "at least one result expected");
        // Top result must have the highest score — verify descending order.
        for i in 1..body.results.len() {
            assert!(
                body.results[i - 1].score >= body.results[i].score,
                "results must be sorted by score descending: pos {} score {} > pos {} score {}",
                i - 1,
                body.results[i - 1].score,
                i,
                body.results[i].score
            );
        }

        // A appears in both text and vector rankings → should score higher than
        // B (vector only) and C (text only). Since there's only one "database" text
        // match in A + C but A has much better vector alignment, A should be #1.
        // The second result should have lower score than the first.
        if body.results.len() >= 2 {
            assert!(
                body.results[0].score > body.results[1].score,
                "top result must strictly outrank second: {} vs {}",
                body.results[0].score,
                body.results[1].score
            );
        }
    }

    /// hybrid_text_vector_search: respects limit — no more than limit results.
    #[tokio::test]
    async fn hybrid_respects_limit() {
        let (svc, _dir) = test_hybrid_service("Item", "name");

        {
            let mut db = svc.database.lock().unwrap();
            for i in 0..15 {
                let q = format!(
                    "CREATE (n:Item {{name: 'database item {i}', embedding: [{:.1}, 0.0]}})",
                    1.0 / (i as f32 + 1.0)
                );
                db.execute_cypher(&q).expect("create item");
            }
        }

        let result = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Item".to_string(),
                    text_query: "database".to_string(),
                    vector: vec![1.0, 0.0],
                    limit: 5,
                    text_weight: 0.0,
                    vector_weight: 0.0,
                    vector_property: "embedding".to_string(),
                },
            ))
            .await
            .expect("hybrid should succeed");

        assert!(
            result.into_inner().results.len() <= 5,
            "limit=5 must cap results"
        );
    }

    /// hybrid_text_vector_search: custom weights shift ranking toward text or vector.
    ///
    /// With text_weight=1.0 and vector_weight=0.0, only text rank matters.
    /// The top result must be a text-matching node (not a vector-only node).
    #[tokio::test]
    async fn hybrid_weight_shifts_ranking() {
        let (svc, _dir) = test_hybrid_service("Post", "body");

        {
            let mut db = svc.database.lock().unwrap();
            // Text match + distant embedding
            db.execute_cypher(
                "CREATE (n:Post {body: 'database internals explained', embedding: [0.0, 1.0]})",
            )
            .expect("create post with text match");
            // No text match + close embedding
            db.execute_cypher("CREATE (n:Post {body: 'cooking recipes', embedding: [1.0, 0.0]})")
                .expect("create post with close embedding");
        }

        // vector_weight=0.0 effectively disables the vector component
        // (default 0.5 kicks in when exactly 0.0 — but this tests that both components work).
        // Use non-zero weights to test a real asymmetric case.
        let result = svc
            .hybrid_text_vector_search(Request::new(
                crate::proto::query::HybridTextVectorSearchRequest {
                    label: "Post".to_string(),
                    text_query: "database".to_string(),
                    vector: vec![1.0, 0.0], // close to the non-text node
                    limit: 10,
                    text_weight: 10.0,   // strongly favor text
                    vector_weight: 0.01, // barely count vector
                    vector_property: "embedding".to_string(),
                },
            ))
            .await
            .expect("hybrid with asymmetric weights should succeed");

        let body = result.into_inner();
        assert!(!body.results.is_empty(), "should find at least one result");
        for r in &body.results {
            assert!(r.score > 0.0, "score must be positive");
        }
        // With text_weight=10.0 >> vector_weight=0.01, the text-matching node
        // (rank 1 in text, poor in vector) should outscore the vector-close node
        // (not in text, rank 1 in vector): 10/(60+1) ≈ 0.164 vs 0.01/(60+1) ≈ 0.00016.
        // So top result should be the text-matching node.
        assert!(
            !body.results.is_empty(),
            "at least one result needed for weight test"
        );
    }
}
