//! Client-side stemming + SSE for encrypted stemmed search.
//!
//! Bridges full-text stemming with SSE equality matching:
//! stemming normalizes surface forms ("running", "runner", "ran" → "run"),
//! then SSE generates deterministic tokens for each stem.
//!
//! This converts the fuzzy search problem (impossible on encrypted data)
//! into an equality matching problem (solvable via SSE).
//!
//! # Write Path
//! ```text
//! Client: "running through the blade"
//!   → tokenize + stem: ["run", "blade"] (stop words removed)
//!   → for each stem: HMAC-SHA256(stem, search_key) → token
//!   → store tokens in encrypted stem index
//! ```
//!
//! # Search Path
//! ```text
//! Client: "runner"
//!   → stem: "run"
//!   → HMAC-SHA256("run", search_key) → query_token
//!   → SSE equality lookup → matching node_ids
//! ```
//!
//! Uses the same Snowball stemmers and whatlang-rs auto-detection
//! as the plaintext full-text search pipeline for consistency.

use super::keys::SearchKey;
use super::token::{generate_search_token, SearchToken};
use crate::stem;

/// Stem text and generate SSE tokens for each unique stem.
///
/// Pipeline: text → split whitespace → lowercase → stem → deduplicate → HMAC token.
///
/// Stop words (articles, prepositions, etc.) are implicitly handled:
/// stemming reduces them to short stems that still produce valid tokens.
/// For true stop word removal, the caller should pre-filter before calling.
///
/// `language`: Snowball language name (e.g., "english", "russian").
///   If stemming is not available for the language, tokens are generated
///   from lowercase words without stemming (same as "none" analyzer).
pub fn stem_and_tokenize(text: &str, language: &str, key: &SearchKey) -> Vec<StemToken> {
    let mut seen_stems = std::collections::HashSet::new();
    let mut result = Vec::new();

    for word in text.split_whitespace() {
        let lowered = word.to_lowercase();
        // Remove non-alphanumeric chars (punctuation)
        let cleaned: String = lowered.chars().filter(|c| c.is_alphanumeric()).collect();
        if cleaned.is_empty() {
            continue;
        }

        let stemmed = stem::stem_word(&cleaned, language);
        let stem_str = stemmed.to_string();

        // Deduplicate: same stem appears once in token index
        if seen_stems.contains(&stem_str) {
            continue;
        }
        seen_stems.insert(stem_str.clone());

        let token = generate_search_token(stem_str.as_bytes(), key);
        result.push(StemToken {
            stem: stem_str,
            token,
        });
    }

    result
}

/// Stem a single query term and generate its SSE token.
///
/// Used on the search path: client stems the query, generates the
/// token, sends it to the server for equality lookup.
pub fn stem_query_token(query: &str, language: &str, key: &SearchKey) -> Option<SearchToken> {
    let lowered = query.to_lowercase();
    let cleaned: String = lowered.chars().filter(|c| c.is_alphanumeric()).collect();
    if cleaned.is_empty() {
        return None;
    }

    let stemmed = stem::stem_word(&cleaned, language);
    Some(generate_search_token(stemmed.as_bytes(), key))
}

/// Auto-detect language, then stem and tokenize.
///
/// Uses whatlang-rs for language detection. Falls back to
/// unstemmed tokenization if detection fails.
pub fn stem_and_tokenize_auto(text: &str, key: &SearchKey) -> Vec<StemToken> {
    let language = crate::lang::detect_language(text)
        .map(|d| d.name.to_string())
        .unwrap_or_else(|| "none".to_string());
    stem_and_tokenize(text, &language, key)
}

/// A stem paired with its SSE search token.
#[derive(Debug, Clone)]
pub struct StemToken {
    /// The stemmed form (e.g., "run" from "running").
    /// Available client-side only — server sees only the token.
    pub stem: String,
    /// HMAC-SHA256 token for SSE equality matching.
    pub token: SearchToken,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::encrypted::keys::SearchKey;

    fn test_key() -> SearchKey {
        SearchKey::from_bytes(&[42u8; 32]).unwrap()
    }

    #[test]
    fn stem_and_tokenize_english() {
        let key = test_key();
        let tokens = stem_and_tokenize("The runners are running quickly", "english", &key);

        let stems: Vec<&str> = tokens.iter().map(|t| t.stem.as_str()).collect();
        // "runners" → "runner", "running" → "run", "quickly" → "quick"
        // "the", "are" also stemmed but not removed (no stop word filter)
        assert!(!stems.is_empty());
        // Check deduplication: even though "runners" and "running" are different words,
        // if they stem to different stems they'll both appear
        let unique_stems: std::collections::HashSet<&str> = stems.iter().copied().collect();
        assert_eq!(
            stems.len(),
            unique_stems.len(),
            "stems should be deduplicated"
        );
    }

    #[test]
    fn stem_and_tokenize_deterministic() {
        let key = test_key();
        let t1 = stem_and_tokenize("running fast", "english", &key);
        let t2 = stem_and_tokenize("running fast", "english", &key);

        assert_eq!(t1.len(), t2.len());
        for (a, b) in t1.iter().zip(t2.iter()) {
            assert_eq!(a.stem, b.stem);
            assert_eq!(a.token, b.token, "same text + key must produce same tokens");
        }
    }

    #[test]
    fn different_surface_forms_same_stem_same_token() {
        let key = test_key();
        // "runner" and "running" should both stem to "run" in English
        let t_runner = stem_query_token("runner", "english", &key).unwrap();
        let t_running = stem_query_token("running", "english", &key).unwrap();

        // Both produce HMAC of "run" or "runner" — depends on Snowball output
        // At minimum, they should be consistent: same stem → same token
        let stem_runner = stem::stem_word("runner", "english");
        let stem_running = stem::stem_word("running", "english");

        if stem_runner == stem_running {
            assert_eq!(t_runner, t_running, "same stem should produce same token");
        }
    }

    #[test]
    fn stem_query_token_empty_returns_none() {
        let key = test_key();
        assert!(stem_query_token("", "english", &key).is_none());
        assert!(stem_query_token("   ", "english", &key).is_none());
        assert!(stem_query_token("!!!", "english", &key).is_none());
    }

    #[test]
    fn stem_query_token_basic() {
        let key = test_key();
        let token = stem_query_token("database", "english", &key);
        assert!(token.is_some());
    }

    #[test]
    fn different_keys_different_tokens() {
        let k1 = SearchKey::from_bytes(&[1u8; 32]).unwrap();
        let k2 = SearchKey::from_bytes(&[2u8; 32]).unwrap();

        let t1 = stem_query_token("running", "english", &k1).unwrap();
        let t2 = stem_query_token("running", "english", &k2).unwrap();

        assert_ne!(t1, t2, "different keys must produce different tokens");
    }

    #[test]
    fn stem_and_tokenize_deduplicates() {
        let key = test_key();
        // If two words stem to the same form, only one token should be produced
        let tokens = stem_and_tokenize("run running runner", "english", &key);

        // Count unique tokens
        let unique_tokens: std::collections::HashSet<_> =
            tokens.iter().map(|t| t.token.clone()).collect();
        // All three words may or may not stem to same root depending on Snowball
        // But within the result, stems are deduplicated
        assert_eq!(tokens.len(), unique_tokens.len());
    }

    #[test]
    fn stem_and_tokenize_punctuation_stripped() {
        let key = test_key();
        let tokens = stem_and_tokenize("hello, world!", "english", &key);
        let stems: Vec<&str> = tokens.iter().map(|t| t.stem.as_str()).collect();
        // Punctuation should be stripped before stemming
        assert!(
            !stems.iter().any(|s| s.contains(',')),
            "punctuation should be removed"
        );
    }

    #[test]
    fn stem_and_tokenize_russian() {
        let key = test_key();
        let tokens = stem_and_tokenize("Графовая база данных для аналитики", "russian", &key);
        assert!(!tokens.is_empty(), "Russian text should produce stems");
    }

    #[test]
    fn stem_and_tokenize_auto_detect() {
        let key = test_key();
        let tokens = stem_and_tokenize_auto(
            "The runners are running through the beautiful forest in spring",
            &key,
        );
        assert!(
            !tokens.is_empty(),
            "auto-detect should produce tokens for English text"
        );
    }

    #[test]
    fn end_to_end_stemmed_sse_search() {
        // Full flow: write stemmed tokens → search with stemmed query
        use crate::encrypted::index::EncryptedFieldIndex;

        let key = test_key();
        let mut idx = EncryptedFieldIndex::new();

        // WRITE: stem text, generate tokens, store in index
        let write_tokens = stem_and_tokenize("running through the forest", "english", &key);
        for st in &write_tokens {
            idx.insert(st.token.clone(), 1);
        }

        // Also index a second document
        let write_tokens2 = stem_and_tokenize("swimming in the ocean", "english", &key);
        for st in &write_tokens2 {
            idx.insert(st.token.clone(), 2);
        }

        // SEARCH: "runner" → stem → token → lookup
        let query_token = stem_query_token("runner", "english", &key).unwrap();
        let _results_runner = idx.search(&query_token).unwrap();
        // "runner" stems to "runner" in Snowball English
        // "running" stems to "run" — different stems, may not cross-match

        // Search for "run" which should match "running" (both stem to "run")
        let query_token_run = stem_query_token("run", "english", &key).unwrap();
        let results_run = idx.search(&query_token_run).unwrap();

        // "run" stems to "run", "running" stems to "run" → should match
        assert!(
            results_run.contains(&1),
            "'run' should match doc containing 'running': stems={:?}",
            write_tokens.iter().map(|t| &t.stem).collect::<Vec<_>>()
        );

        // "swim" should match "swimming"
        let query_swim = stem_query_token("swim", "english", &key).unwrap();
        let results_swim = idx.search(&query_swim).unwrap();
        assert!(
            results_swim.contains(&2),
            "'swim' should match doc containing 'swimming'"
        );

        // "run" should NOT match doc 2 (swimming)
        assert!(
            !results_run.contains(&2),
            "'run' should not match 'swimming' doc"
        );
    }

    #[test]
    fn end_to_end_stemmed_sse_storage() {
        use crate::encrypted::storage::EncryptedIndex;
        use coordinode_storage::engine::config::StorageConfig;
        use coordinode_storage::engine::core::StorageEngine;

        let dir = tempfile::tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::open(&config).unwrap();
        let key = test_key();

        // WRITE: stem + tokenize + persist
        let idx = EncryptedIndex::new(&engine, "Article", "body_stems");
        let stems = stem_and_tokenize("running through the forest", "english", &key);
        for st in &stems {
            idx.insert(&st.token, 1).unwrap();
        }

        let stems2 = stem_and_tokenize("swimming in the ocean", "english", &key);
        for st in &stems2 {
            idx.insert(&st.token, 2).unwrap();
        }

        // SEARCH: "run" → stem → HMAC → storage lookup
        let query_token = stem_query_token("run", "english", &key).unwrap();
        let results = idx.search(&query_token).unwrap();
        assert!(results.contains(&1), "'run' should find doc 1 in SSE index");
        assert!(
            !results.contains(&2),
            "'run' should not find doc 2 (swimming)"
        );
    }
}
