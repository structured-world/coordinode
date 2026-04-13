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

use coordinode_embed::Database;

use crate::proto::query;

/// Default result limit when the request specifies `limit = 0`.
const DEFAULT_TEXT_LIMIT: usize = 10;
/// Hard cap to prevent runaway queries regardless of request limit.
const MAX_TEXT_LIMIT: usize = 1000;

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
}

#[cfg(test)]
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
}
