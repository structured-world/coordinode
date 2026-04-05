//! CJK tokenization: lindera (JP/KR) + jieba-rs (CN).
//!
//! Provides tantivy-compatible tokenizers for Chinese, Japanese, and Korean text.
//! Each tokenizer wraps a dictionary-based segmenter that splits text into
//! meaningful words rather than whitespace-delimited tokens.
//!
//! Feature flags (per-language, include only what you need):
//! - `cjk-zh` — `"chinese_jieba"` analyzer (jieba-rs, +21MB)
//! - `cjk-ja` — `"japanese_lindera"` analyzer (lindera IPAdic, +15MB)
//! - `cjk-ko` — `"korean_lindera"` analyzer (lindera KO-dic, +34MB)
//! - `cjk` — umbrella: all three (~70MB)

#[cfg(any(feature = "cjk-zh", feature = "cjk-ja", feature = "cjk-ko"))]
mod tokenizers;

#[cfg(feature = "cjk-zh")]
pub use tokenizers::JiebaTokenizer;

#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
pub use tokenizers::{CjkLanguage, LinderaTokenizer};

/// Configuration for CJK dictionary loading.
///
/// When paths are `None`, the tokenizer uses the embedded dictionary
/// (requires the corresponding compile-time feature flag: `cjk-zh`, `cjk-ja`, `cjk-ko`).
///
/// When paths are `Some`, the tokenizer loads the dictionary from the filesystem.
/// This enables Docker images without embedded CJK dictionaries (~70MB savings).
///
/// ## Docker usage
/// ```yaml
/// volumes:
///   - ./dictionaries/ipadic:/data/dicts/ipadic:ro
///   - ./dictionaries/jieba:/data/dicts/jieba.txt:ro
/// environment:
///   COORDINODE_CJK_JA_DICT: /data/dicts/ipadic
///   COORDINODE_CJK_ZH_DICT: /data/dicts/jieba.txt
/// ```
#[derive(Debug, Clone, Default)]
pub struct CjkDictConfig {
    /// Path to jieba dictionary file (Chinese). Format: one `word freq [tag]` per line.
    pub chinese_dict_path: Option<std::path::PathBuf>,
    /// Path to lindera IPAdic dictionary directory (Japanese).
    pub japanese_dict_path: Option<std::path::PathBuf>,
    /// Path to lindera KO-dic dictionary directory (Korean).
    pub korean_dict_path: Option<std::path::PathBuf>,
}

impl CjkDictConfig {
    /// Load config from environment variables.
    ///
    /// - `COORDINODE_CJK_ZH_DICT` → Chinese dictionary file path
    /// - `COORDINODE_CJK_JA_DICT` → Japanese dictionary directory path
    /// - `COORDINODE_CJK_KO_DICT` → Korean dictionary directory path
    pub fn from_env() -> Self {
        Self {
            chinese_dict_path: std::env::var("COORDINODE_CJK_ZH_DICT")
                .ok()
                .map(std::path::PathBuf::from),
            japanese_dict_path: std::env::var("COORDINODE_CJK_JA_DICT")
                .ok()
                .map(std::path::PathBuf::from),
            korean_dict_path: std::env::var("COORDINODE_CJK_KO_DICT")
                .ok()
                .map(std::path::PathBuf::from),
        }
    }

    /// Create a Chinese tokenizer using configured path or embedded dictionary.
    #[cfg(feature = "cjk-zh")]
    pub fn chinese_tokenizer(&self) -> Result<JiebaTokenizer, String> {
        if let Some(ref path) = self.chinese_dict_path {
            JiebaTokenizer::with_dict_path(path)
        } else {
            Ok(JiebaTokenizer::new())
        }
    }

    /// Create a Japanese tokenizer using configured path or embedded dictionary.
    #[cfg(feature = "cjk-ja")]
    pub fn japanese_tokenizer(&self) -> Result<LinderaTokenizer, String> {
        if let Some(ref path) = self.japanese_dict_path {
            LinderaTokenizer::with_dict_path(path)
        } else {
            LinderaTokenizer::new(CjkLanguage::Japanese)
        }
    }

    /// Create a Korean tokenizer using configured path or embedded dictionary.
    #[cfg(feature = "cjk-ko")]
    pub fn korean_tokenizer(&self) -> Result<LinderaTokenizer, String> {
        if let Some(ref path) = self.korean_dict_path {
            LinderaTokenizer::with_dict_path(path)
        } else {
            LinderaTokenizer::new(CjkLanguage::Korean)
        }
    }
}
