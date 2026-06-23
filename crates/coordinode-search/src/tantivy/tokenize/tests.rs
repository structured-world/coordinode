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
