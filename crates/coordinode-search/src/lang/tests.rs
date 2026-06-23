use super::*;

#[test]
fn detect_english() {
    let result = detect_language("This is a test of the English language detection system");
    assert!(result.is_some());
    let d = result.unwrap();
    assert_eq!(d.name, "english");
    assert_eq!(d.code, "en");
    assert!(d.confidence > 0.5);
}

#[test]
fn detect_russian() {
    let result = detect_language("Это тест определения русского языка в тексте");
    assert!(result.is_some());
    let d = result.unwrap();
    assert_eq!(d.name, "russian");
    assert_eq!(d.code, "ru");
}

#[test]
fn detect_ukrainian() {
    let result = detect_language(
        "Це тест визначення української мови у тексті який має бути достатньо довгим",
    );
    assert!(result.is_some());
    let d = result.unwrap();
    assert_eq!(d.name, "ukrainian");
    assert_eq!(d.code, "uk");
}

#[test]
fn detect_german() {
    let result = detect_language("Dies ist ein Test der deutschen Spracherkennung im Text");
    assert!(result.is_some());
    let d = result.unwrap();
    assert_eq!(d.name, "german");
    assert_eq!(d.code, "de");
}

#[test]
fn detect_french() {
    let result =
        detect_language("Ceci est un test de détection de la langue française dans le texte");
    assert!(result.is_some());
    let d = result.unwrap();
    assert_eq!(d.name, "french");
    assert_eq!(d.code, "fr");
}

#[test]
fn empty_text_returns_none() {
    assert!(detect_language("").is_none());
}

#[test]
fn short_text_low_confidence() {
    // Very short text may have low confidence
    let result = detect_language_confident("ok", 0.99);
    // Either None or low confidence — both acceptable
    if let Some(d) = result {
        assert!(d.confidence >= 0.99);
    }
}

#[test]
fn confidence_threshold() {
    let high = detect_language_confident(
        "This is definitely English text with enough words for confident detection",
        0.8,
    );
    assert!(high.is_some(), "long English text should be confident");
}

// CJK language detection tests (per-language feature flags)

#[cfg(feature = "cjk-zh")]
#[test]
fn detect_chinese() {
    let result = detect_language("今天天气真好我们去公园散步吧这是一个美丽的日子");
    assert!(result.is_some(), "should detect Chinese");
    let d = result.unwrap();
    assert_eq!(d.name, "chinese_jieba");
    assert_eq!(d.code, "zh");
}

#[cfg(feature = "cjk-ja")]
#[test]
fn detect_japanese() {
    let result = detect_language("東京都に住んでいます。毎日電車で会社に通っています");
    assert!(result.is_some(), "should detect Japanese");
    let d = result.unwrap();
    assert_eq!(d.name, "japanese_lindera");
    assert_eq!(d.code, "ja");
}

#[cfg(feature = "cjk-ko")]
#[test]
fn detect_korean() {
    let result = detect_language("대한민국의 수도는 서울입니다 오늘 날씨가 좋습니다");
    assert!(result.is_some(), "should detect Korean");
    let d = result.unwrap();
    assert_eq!(d.name, "korean_lindera");
    assert_eq!(d.code, "ko");
}
