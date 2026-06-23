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
mod tests;
