//! Language detection via whatlang-rs.
//!
//! Detects the language of a text string using trigram analysis.
//! Returns a language name compatible with `stem::algorithm_for_language()`.

/// Detected language result.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectedLanguage {
    /// Language name (e.g., "english", "russian", "ukrainian").
    pub name: &'static str,
    /// ISO 639-1 code (e.g., "en", "ru", "uk").
    pub code: &'static str,
    /// Detection confidence (0.0..1.0).
    pub confidence: f64,
}

/// Detect the language of a text string.
///
/// Returns `None` if detection fails or confidence is below threshold.
/// Uses whatlang-rs trigram analysis (fast, no dictionary needed).
pub fn detect_language(text: &str) -> Option<DetectedLanguage> {
    let info = whatlang::detect(text)?;

    let (name, code) = map_whatlang_to_stemmer(info.lang())?;

    Some(DetectedLanguage {
        name,
        code,
        confidence: info.confidence(),
    })
}

/// Detect language with minimum confidence threshold.
pub fn detect_language_confident(text: &str, min_confidence: f64) -> Option<DetectedLanguage> {
    let detected = detect_language(text)?;
    if detected.confidence >= min_confidence {
        Some(detected)
    } else {
        None
    }
}

/// Map whatlang::Lang to analyzer language name + ISO code.
///
/// For CJK languages, returns the CJK analyzer name (e.g., "chinese_jieba")
/// which is handled by the `cjk` feature. For others, returns the Snowball
/// stemmer language name. Returns None for unsupported languages.
fn map_whatlang_to_stemmer(lang: whatlang::Lang) -> Option<(&'static str, &'static str)> {
    use whatlang::Lang;
    match lang {
        Lang::Ara => Some(("arabic", "ar")),
        Lang::Hye => Some(("armenian", "hy")),
        Lang::Dan => Some(("danish", "da")),
        Lang::Nld => Some(("dutch", "nl")),
        Lang::Eng => Some(("english", "en")),
        Lang::Fin => Some(("finnish", "fi")),
        Lang::Fra => Some(("french", "fr")),
        Lang::Deu => Some(("german", "de")),
        Lang::Ell => Some(("greek", "el")),
        Lang::Hun => Some(("hungarian", "hu")),
        Lang::Ita => Some(("italian", "it")),
        Lang::Nob => Some(("norwegian", "nb")),
        Lang::Por => Some(("portuguese", "pt")),
        Lang::Ron => Some(("romanian", "ro")),
        Lang::Rus => Some(("russian", "ru")),
        Lang::Spa => Some(("spanish", "es")),
        Lang::Swe => Some(("swedish", "sv")),
        Lang::Tam => Some(("tamil", "ta")),
        Lang::Tur => Some(("turkish", "tr")),
        Lang::Ukr => Some(("ukrainian", "uk")),
        // CJK languages — mapped to CJK analyzer names (per-language feature flags)
        #[cfg(feature = "cjk-zh")]
        Lang::Cmn => Some(("chinese_jieba", "zh")),
        #[cfg(feature = "cjk-ja")]
        Lang::Jpn => Some(("japanese_lindera", "ja")),
        #[cfg(feature = "cjk-ko")]
        Lang::Kor => Some(("korean_lindera", "ko")),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
