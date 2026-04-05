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
mod tests {
    use super::*;

    #[test]
    fn tokenize_none_no_stemming() {
        let tokens = tokenize_text("The Runners are Running", "none");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["the", "runners", "are", "running"]);
    }

    #[test]
    fn tokenize_none_splits_on_non_alphanumeric() {
        // SimpleTokenizer splits on non-alphanumeric boundaries
        let tokens = tokenize_text("ERR_TIMEOUT_EXCEEDED", "none");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(texts, vec!["err", "timeout", "exceeded"]);
    }

    #[test]
    fn tokenize_none_preserves_offsets() {
        let text = "hello world";
        let tokens = tokenize_text(text, "none");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].offset_from, 0);
        assert_eq!(tokens[0].offset_to, 5);
        assert_eq!(tokens[1].offset_from, 6);
        assert_eq!(tokens[1].offset_to, 11);
    }

    #[test]
    fn tokenize_none_positions_sequential() {
        let tokens = tokenize_text("a b c d", "none");
        for (i, tok) in tokens.iter().enumerate() {
            assert_eq!(tok.position, i, "position should be sequential");
        }
    }

    #[test]
    fn tokenize_english_stemming() {
        let tokens = tokenize_text("the runners are running quickly", "english");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // "runners" → "runner", "running" → "run", "quickly" → "quick"
        assert!(texts.contains(&"runner") || texts.contains(&"run"));
    }

    #[test]
    fn tokenize_russian_stemming() {
        let tokens = tokenize_text("бегущий человек быстро бежал", "russian");
        // Both "бегущий" and "бежал" should stem to a common root
        assert!(tokens.len() >= 3);
    }

    #[test]
    fn tokenize_unknown_language_falls_back_to_simple() {
        let tokens = tokenize_text("hello world", "klingon");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn tokenize_auto_detect_english() {
        let tokens = tokenize_text(
            "The runners are running through the beautiful forest",
            "auto_detect",
        );
        // Should detect English and apply stemming
        assert!(!tokens.is_empty());
        // At minimum, tokens are lowercased
        assert!(tokens.iter().all(|t| t.text == t.text.to_lowercase()));
    }

    #[test]
    fn tokenize_empty_text() {
        let tokens = tokenize_text("", "english");
        assert!(tokens.is_empty());
    }

    #[test]
    fn is_known_language_basics() {
        assert!(is_known_language("none"));
        assert!(is_known_language("auto_detect"));
        assert!(is_known_language("english"));
        assert!(is_known_language("russian"));
        assert!(is_known_language("en"));
        assert!(!is_known_language("klingon"));
    }

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn tokenize_chinese() {
        let tokens = tokenize_text("我来到北京清华大学", "chinese_jieba");
        assert!(!tokens.is_empty());
        assert!(
            tokens.iter().any(|t| t.text.contains("北京")),
            "should contain 北京: {tokens:?}"
        );
    }

    #[cfg(feature = "cjk-ja")]
    #[test]
    fn tokenize_japanese() {
        let tokens = tokenize_text("東京都に住んでいます", "japanese_lindera");
        assert!(!tokens.is_empty());
        assert!(
            tokens.iter().any(|t| t.text.contains("東京")),
            "should contain 東京: {tokens:?}"
        );
    }

    #[cfg(feature = "cjk-ko")]
    #[test]
    fn tokenize_korean() {
        let tokens = tokenize_text("대한민국의 수도는 서울입니다", "korean_lindera");
        assert!(!tokens.is_empty());
        assert!(
            tokens.iter().any(|t| t.text.contains("서울")),
            "should contain 서울: {tokens:?}"
        );
    }

    #[cfg(feature = "cjk-zh")]
    #[test]
    fn is_known_cjk() {
        assert!(is_known_language("chinese_jieba"));
    }
}
