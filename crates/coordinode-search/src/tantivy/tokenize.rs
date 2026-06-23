//! Manual tokenization pipeline for per-document language selection.
//!
//! Provides `tokenize_text()` which runs any supported language's tokenizer
//! pipeline (Snowball stemmer, CJK segmenter, "none", or "auto_detect")
//! and returns a `Vec<Token>`. Used by:
//!
//! - **Write path**: build `PreTokenizedString` for per-document language
//! - **Search path**: tokenize query with a specific language

use tantivy::tokenizer::{SimpleTokenizer, Token, TokenStream, Tokenizer};

use crate::stem;

/// Tokenize text using the specified language's full pipeline.
///
/// Returns a list of tokens with correct offsets and positions.
///
/// Language resolution:
/// - `"none"` → whitespace + lowercase, no stemming
/// - `"auto_detect"` → detect via whatlang, then use detected language
/// - CJK names (`"chinese_jieba"`, `"japanese_lindera"`, `"korean_lindera"`) → dictionary segmenter
/// - Snowball names (`"english"`, `"russian"`, etc.) → whitespace + lowercase + stemmer
/// - Unknown → whitespace + lowercase (same as "none")
pub fn tokenize_text(text: &str, language: &str) -> Vec<Token> {
    match language {
        "none" => tokenize_simple(text),
        "auto_detect" => tokenize_auto_detect(text),
        #[cfg(feature = "cjk-zh")]
        "chinese_jieba" => tokenize_cjk_chinese(text),
        #[cfg(feature = "cjk-ja")]
        "japanese_lindera" => tokenize_cjk_japanese(text),
        #[cfg(feature = "cjk-ko")]
        "korean_lindera" => tokenize_cjk_korean(text),
        lang => {
            if let Some(algorithm) = stem::algorithm_for_language(lang) {
                tokenize_stemmed(text, algorithm)
            } else {
                tokenize_simple(text)
            }
        }
    }
}

/// Check if a language name is recognized.
///
/// Returns `true` for "none", "auto_detect", all Snowball languages,
/// and CJK languages (when their features are enabled).
pub fn is_known_language(lang: &str) -> bool {
    match lang {
        "none" | "auto_detect" => true,
        #[cfg(feature = "cjk-zh")]
        "chinese_jieba" => true,
        #[cfg(feature = "cjk-ja")]
        "japanese_lindera" => true,
        #[cfg(feature = "cjk-ko")]
        "korean_lindera" => true,
        other => stem::algorithm_for_language(other).is_some(),
    }
}

/// Simple tokenization using tantivy's SimpleTokenizer + lowercase, no stemming.
///
/// Uses the same tokenizer as tantivy's QueryParser to ensure consistency
/// between index-time and query-time tokenization. SimpleTokenizer splits
/// on non-alphanumeric boundaries (not just whitespace), so "ERR_TIMEOUT"
/// becomes ["err", "timeout"].
fn tokenize_simple(text: &str) -> Vec<Token> {
    run_tantivy_tokenizer(text, None)
}

/// Stemmed tokenization using tantivy's SimpleTokenizer + lowercase + Snowball.
fn tokenize_stemmed(text: &str, algorithm: rust_stemmers::Algorithm) -> Vec<Token> {
    run_tantivy_tokenizer(text, Some(algorithm))
}

/// Run tantivy's SimpleTokenizer, lowercase, and optional stemming.
///
/// This ensures that tokens produced for PreTokenizedString exactly match
/// what tantivy's QueryParser would produce, preventing index/query mismatch.
fn run_tantivy_tokenizer(text: &str, stemmer_algo: Option<rust_stemmers::Algorithm>) -> Vec<Token> {
    let mut tokenizer = SimpleTokenizer::default();
    let mut stream = tokenizer.token_stream(text);
    let stemmer = stemmer_algo.map(rust_stemmers::Stemmer::create);
    let mut tokens = Vec::new();

    while stream.advance() {
        let t = stream.token();
        let lowered = t.text.to_lowercase();
        let final_text = if let Some(ref s) = stemmer {
            s.stem(&lowered).into_owned()
        } else {
            lowered
        };
        tokens.push(Token {
            offset_from: t.offset_from,
            offset_to: t.offset_to,
            position: t.position,
            position_length: t.position_length,
            text: final_text,
        });
    }

    tokens
}

/// Auto-detect language via whatlang, then tokenize with detected language.
/// Falls back to simple tokenization if detection fails.
fn tokenize_auto_detect(text: &str) -> Vec<Token> {
    if let Some(detected) = crate::lang::detect_language(text) {
        tokenize_text(text, detected.name)
    } else {
        tokenize_simple(text)
    }
}

// -- CJK tokenization via dictionary segmenters --

#[cfg(feature = "cjk-zh")]
fn tokenize_cjk_chinese(text: &str) -> Vec<Token> {
    use crate::cjk::CjkDictConfig;
    use tantivy::tokenizer::{TokenStream, Tokenizer};

    let mut tok = match CjkDictConfig::from_env().chinese_tokenizer() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("failed to create Chinese tokenizer: {e}");
            return tokenize_simple(text);
        }
    };
    let mut stream = tok.token_stream(text);
    let mut tokens = Vec::new();

    while stream.advance() {
        let t = stream.token();
        tokens.push(Token {
            offset_from: t.offset_from,
            offset_to: t.offset_to,
            position: t.position,
            position_length: t.position_length,
            // CJK: lowercase for consistency (handles mixed CJK + ASCII)
            text: t.text.to_lowercase(),
        });
    }

    tokens
}

#[cfg(feature = "cjk-ja")]
fn tokenize_cjk_japanese(text: &str) -> Vec<Token> {
    use tantivy::tokenizer::{TokenStream, Tokenizer};

    let cjk_config = crate::cjk::CjkDictConfig::from_env();
    let mut tok = match cjk_config.japanese_tokenizer() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("failed to create Japanese tokenizer: {e}");
            return tokenize_simple(text);
        }
    };
    let mut stream = tok.token_stream(text);
    let mut tokens = Vec::new();

    while stream.advance() {
        let t = stream.token();
        tokens.push(Token {
            offset_from: t.offset_from,
            offset_to: t.offset_to,
            position: t.position,
            position_length: t.position_length,
            text: t.text.to_lowercase(),
        });
    }

    tokens
}

#[cfg(feature = "cjk-ko")]
fn tokenize_cjk_korean(text: &str) -> Vec<Token> {
    use tantivy::tokenizer::{TokenStream, Tokenizer};

    let cjk_config = crate::cjk::CjkDictConfig::from_env();
    let mut tok = match cjk_config.korean_tokenizer() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("failed to create Korean tokenizer: {e}");
            return tokenize_simple(text);
        }
    };
    let mut stream = tok.token_stream(text);
    let mut tokens = Vec::new();

    while stream.advance() {
        let t = stream.token();
        tokens.push(Token {
            offset_from: t.offset_from,
            offset_to: t.offset_to,
            position: t.position,
            position_length: t.position_length,
            text: t.text.to_lowercase(),
        });
    }

    tokens
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
