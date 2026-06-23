//! tantivy Tokenizer implementations for CJK languages.
//!
//! These tokenizers implement tantivy's `Tokenizer` trait, producing
//! a stream of `Token`s from CJK text. Each wraps a dictionary-based
//! segmenter: lindera for Japanese/Korean morphological analysis,
//! jieba-rs for Chinese statistical segmentation.
//!
//! Gated per-language: `cjk-zh`, `cjk-ja`, `cjk-ko`.

use std::sync::Arc;

use tantivy::tokenizer::{Token, TokenStream, Tokenizer};

// ---------------------------------------------------------------------------
// CjkLanguage enum (lindera: JP/KR)
// ---------------------------------------------------------------------------

/// CJK language selector for lindera dictionary loading.
#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CjkLanguage {
    #[cfg(feature = "cjk-ja")]
    Japanese,
    #[cfg(feature = "cjk-ko")]
    Korean,
}

// ---------------------------------------------------------------------------
// Jieba tokenizer (Chinese) — behind cjk-zh
// ---------------------------------------------------------------------------

/// Chinese word segmentation tokenizer using jieba-rs.
///
/// jieba uses a statistical model with a large dictionary for segmentation.
/// The `Jieba` instance is shared (via `Arc`) across clones — it holds
/// ~13MB of dictionary data in memory, so a single instance is reused.
#[cfg(feature = "cjk-zh")]
#[derive(Clone)]
pub struct JiebaTokenizer {
    jieba: Arc<jieba_rs::Jieba>,
}

#[cfg(feature = "cjk-zh")]
impl Default for JiebaTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cjk-zh")]
impl JiebaTokenizer {
    /// Create a new jieba tokenizer with the default embedded dictionary.
    pub fn new() -> Self {
        Self {
            jieba: Arc::new(jieba_rs::Jieba::new()),
        }
    }

    /// Create a jieba tokenizer loading the dictionary from a file path.
    ///
    /// The dictionary file must be in jieba format: one word per line,
    /// each line containing `word freq [tag]` (space-separated).
    ///
    /// This enables Docker images without embedded CJK dictionaries (~21MB savings).
    /// Mount the dictionary file as a volume and pass its path here.
    pub fn with_dict_path(path: &std::path::Path) -> Result<Self, String> {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("failed to open jieba dictionary {}: {e}", path.display()))?;
        let mut reader = std::io::BufReader::new(file);
        let jieba = jieba_rs::Jieba::with_dict(&mut reader)
            .map_err(|e| format!("failed to parse jieba dictionary {}: {e}", path.display()))?;
        Ok(Self {
            jieba: Arc::new(jieba),
        })
    }
}

#[cfg(feature = "cjk-zh")]
impl Tokenizer for JiebaTokenizer {
    type TokenStream<'a> = JiebaTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        JiebaTokenStream {
            jieba: Arc::clone(&self.jieba),
            text,
            tokens: Vec::new(),
            index: 0,
            current_token: Token::default(),
            initialized: false,
        }
    }
}

/// Token stream produced by [`JiebaTokenizer`].
///
/// Lazily segments text on first `advance()` call, then yields tokens
/// one at a time. Uses `cut_for_search` which produces finer-grained
/// segments suitable for search indexing.
#[cfg(feature = "cjk-zh")]
pub struct JiebaTokenStream<'a> {
    jieba: Arc<jieba_rs::Jieba>,
    text: &'a str,
    /// Pre-computed (byte_start, byte_end) pairs from segmentation.
    tokens: Vec<(usize, usize)>,
    index: usize,
    current_token: Token,
    initialized: bool,
}

#[cfg(feature = "cjk-zh")]
impl<'a> TokenStream for JiebaTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if !self.initialized {
            self.initialized = true;
            self.tokens = segment_with_offsets(&self.jieba, self.text);
            self.index = 0;
        }

        if self.index >= self.tokens.len() {
            return false;
        }

        let (start, end) = self.tokens[self.index];
        self.current_token.offset_from = start;
        self.current_token.offset_to = end;
        self.current_token.position = self.index;
        self.current_token.position_length = 1;
        self.current_token.text.clear();
        self.current_token.text.push_str(&self.text[start..end]);

        self.index += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.current_token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current_token
    }
}

/// Segment text using jieba `cut_for_search` and compute byte offsets.
///
/// `cut_for_search` produces finer-grained segments than `cut`, which is
/// better for search indexing (e.g., "中华人民共和国" → "中华", "华人",
/// "人民", "共和", "共和国", "中华人民共和国").
#[cfg(feature = "cjk-zh")]
fn segment_with_offsets(jieba: &jieba_rs::Jieba, text: &str) -> Vec<(usize, usize)> {
    let words = jieba.cut_for_search(text, true);
    let mut offsets = Vec::with_capacity(words.len());
    let mut byte_pos = 0;

    for word in words {
        // Find the word in the remaining text, starting from byte_pos.
        // jieba returns words in order, so we scan forward.
        if let Some(rel_start) = text[byte_pos..].find(word) {
            let abs_start = byte_pos + rel_start;
            let abs_end = abs_start + word.len();
            // Skip empty/whitespace-only tokens
            if !word.trim().is_empty() {
                offsets.push((abs_start, abs_end));
            }
            byte_pos = abs_end;
        }
    }

    offsets
}

// ---------------------------------------------------------------------------
// Lindera tokenizer (Japanese / Korean) — behind cjk-ja / cjk-ko
// ---------------------------------------------------------------------------

/// Japanese/Korean morphological tokenizer using lindera.
///
/// lindera performs dictionary-based morphological analysis:
/// - Japanese: IPAdic dictionary (MeCab-compatible, `cjk-ja` feature)
/// - Korean: KO-dic dictionary (`cjk-ko` feature)
///
/// The lindera `Tokenizer` is wrapped in `Arc` because it holds the
/// entire dictionary in memory (~20MB for IPAdic, ~15MB for KO-dic).
#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
#[derive(Clone)]
pub struct LinderaTokenizer {
    tokenizer: Arc<lindera::tokenizer::Tokenizer>,
}

#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
impl LinderaTokenizer {
    /// Create a new lindera tokenizer for the specified CJK language.
    ///
    /// Loads the embedded dictionary for Japanese (IPAdic) or Korean (KO-dic).
    /// Returns an error if the dictionary cannot be loaded (should not happen
    /// with embedded dictionaries compiled via feature flags).
    /// Create a new lindera tokenizer with the embedded dictionary.
    ///
    /// Requires the corresponding embed feature: `cjk-ja` (embed-ipadic)
    /// or `cjk-ko` (embed-ko-dic).
    pub fn new(language: CjkLanguage) -> Result<Self, String> {
        let dict_uri = match language {
            #[cfg(feature = "cjk-ja")]
            CjkLanguage::Japanese => "embedded://ipadic",
            #[cfg(feature = "cjk-ko")]
            CjkLanguage::Korean => "embedded://ko-dic",
        };
        Self::from_uri(dict_uri)
    }

    /// Create a lindera tokenizer loading the dictionary from a filesystem path.
    ///
    /// The path must point to a compiled lindera dictionary directory
    /// (containing `dict.da`, `char_def.bin`, etc.).
    ///
    /// This enables Docker images without embedded CJK dictionaries
    /// (~15MB savings for Japanese, ~34MB for Korean). Mount the dictionary
    /// directory as a volume and pass its path here.
    pub fn with_dict_path(path: &std::path::Path) -> Result<Self, String> {
        let uri = format!("file://{}", path.display());
        Self::from_uri(&uri)
    }

    fn from_uri(dict_uri: &str) -> Result<Self, String> {
        let dictionary = lindera::dictionary::load_dictionary(dict_uri)
            .map_err(|e| format!("failed to load lindera dictionary {dict_uri}: {e}"))?;

        let segmenter =
            lindera::segmenter::Segmenter::new(lindera::mode::Mode::Normal, dictionary, None);
        let tokenizer = lindera::tokenizer::Tokenizer::new(segmenter);

        Ok(Self {
            tokenizer: Arc::new(tokenizer),
        })
    }
}

#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
impl Tokenizer for LinderaTokenizer {
    type TokenStream<'a> = LinderaTokenStream<'a>;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> Self::TokenStream<'a> {
        LinderaTokenStream {
            tokenizer: Arc::clone(&self.tokenizer),
            text,
            tokens: Vec::new(),
            index: 0,
            current_token: Token::default(),
            initialized: false,
        }
    }
}

/// Token stream produced by [`LinderaTokenizer`].
///
/// Lazily segments text on first `advance()` call via lindera's
/// morphological analyzer, then yields tokens one at a time.
#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
pub struct LinderaTokenStream<'a> {
    tokenizer: Arc<lindera::tokenizer::Tokenizer>,
    text: &'a str,
    /// Pre-computed (byte_start, byte_end, surface) triples.
    tokens: Vec<(usize, usize, String)>,
    index: usize,
    current_token: Token,
    initialized: bool,
}

#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
impl<'a> TokenStream for LinderaTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if !self.initialized {
            self.initialized = true;
            self.tokens = lindera_segment(self.tokenizer.as_ref(), self.text);
            self.index = 0;
        }

        if self.index >= self.tokens.len() {
            return false;
        }

        let (start, end, ref surface) = self.tokens[self.index];
        self.current_token.offset_from = start;
        self.current_token.offset_to = end;
        self.current_token.position = self.index;
        self.current_token.position_length = 1;
        self.current_token.text.clear();
        self.current_token.text.push_str(surface);

        self.index += 1;
        true
    }

    fn token(&self) -> &Token {
        &self.current_token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.current_token
    }
}

/// Run lindera tokenization and extract (byte_start, byte_end, surface) triples.
///
/// Filters out whitespace-only and punctuation-only tokens to produce
/// cleaner search index entries.
#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
fn lindera_segment(
    tokenizer: &lindera::tokenizer::Tokenizer,
    text: &str,
) -> Vec<(usize, usize, String)> {
    let lindera_tokens = match tokenizer.tokenize(text) {
        Ok(tokens) => tokens,
        Err(e) => {
            tracing::warn!("lindera tokenization failed: {e}");
            return Vec::new();
        }
    };

    let mut result = Vec::with_capacity(lindera_tokens.len());
    for tok in &lindera_tokens {
        let surface = tok.surface.as_ref();
        // Skip whitespace-only and punctuation-only tokens
        if surface.trim().is_empty() || is_cjk_punctuation(surface) {
            continue;
        }
        result.push((tok.byte_start, tok.byte_end, surface.to_string()));
    }
    result
}

/// Check if a string consists entirely of CJK punctuation characters.
///
/// Filters out common Japanese/Korean/Chinese punctuation that lindera
/// produces as separate tokens but are not useful for search indexing.
#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
fn is_cjk_punctuation(s: &str) -> bool {
    s.chars().all(|c| {
        matches!(c,
            // ASCII punctuation
            '!' | '"' | '#' | '$' | '%' | '&' | '\'' | '(' | ')' | '*' |
            '+' | ',' | '-' | '.' | '/' | ':' | ';' | '<' | '=' | '>' |
            '?' | '@' | '[' | '\\' | ']' | '^' | '_' | '`' | '{' | '|' |
            '}' | '~' |
            // CJK fullwidth punctuation
            '\u{3000}'..='\u{303F}' |  // CJK symbols and punctuation
            '\u{FF01}'..='\u{FF60}' |  // Fullwidth ASCII variants
            '\u{FE30}'..='\u{FE4F}'    // CJK compatibility forms
        )
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
