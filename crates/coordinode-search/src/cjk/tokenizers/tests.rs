use super::*;

// -- Jieba (Chinese) tests --

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_basic_segmentation() {
    let mut tok = JiebaTokenizer::new();
    let mut stream = tok.token_stream("我来到北京清华大学");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(!tokens.is_empty(), "jieba should produce tokens");
    assert!(
        tokens.iter().any(|t| t == "北京"),
        "expected '北京' in tokens: {tokens:?}"
    );
}

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_search_mode_fine_grained() {
    let mut tok = JiebaTokenizer::new();
    let mut stream = tok.token_stream("中华人民共和国");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(
        tokens.len() > 1,
        "search mode should split compound: {tokens:?}"
    );
}

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_byte_offsets_correct() {
    let mut tok = JiebaTokenizer::new();
    let text = "今天天气真好";
    let mut stream = tok.token_stream(text);
    while stream.advance() {
        let t = stream.token();
        let slice = &text[t.offset_from..t.offset_to];
        assert_eq!(
            slice, t.text,
            "offset [{}, {}) should match token text",
            t.offset_from, t.offset_to
        );
    }
}

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_empty_input() {
    let mut tok = JiebaTokenizer::new();
    let mut stream = tok.token_stream("");
    assert!(!stream.advance(), "empty input should produce no tokens");
}

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_mixed_cjk_and_ascii() {
    let mut tok = JiebaTokenizer::new();
    let mut stream = tok.token_stream("我喜欢Rust编程语言");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(
        tokens.iter().any(|t| t.contains("Rust") || t == "Rust"),
        "should preserve ASCII words: {tokens:?}"
    );
}

// -- Lindera (Japanese) tests --

#[cfg(feature = "cjk-ja")]
#[test]
fn lindera_japanese_segmentation() {
    let mut tok = LinderaTokenizer::new(CjkLanguage::Japanese).unwrap();
    let mut stream = tok.token_stream("東京都に住んでいます");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(!tokens.is_empty(), "lindera should produce tokens for JP");
    assert!(
        tokens.iter().any(|t| t.contains("東京")),
        "expected '東京' in tokens: {tokens:?}"
    );
}

#[cfg(feature = "cjk-ja")]
#[test]
fn lindera_japanese_byte_offsets() {
    let mut tok = LinderaTokenizer::new(CjkLanguage::Japanese).unwrap();
    let text = "日本語のテスト";
    let mut stream = tok.token_stream(text);
    while stream.advance() {
        let t = stream.token();
        let slice = &text[t.offset_from..t.offset_to];
        assert_eq!(
            slice, t.text,
            "offset [{}, {}) should match token text",
            t.offset_from, t.offset_to
        );
    }
}

// -- Lindera (Korean) tests --

#[cfg(feature = "cjk-ko")]
#[test]
fn lindera_korean_segmentation() {
    let mut tok = LinderaTokenizer::new(CjkLanguage::Korean).unwrap();
    let mut stream = tok.token_stream("대한민국의 수도는 서울입니다");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(!tokens.is_empty(), "lindera should produce tokens for KR");
    assert!(
        tokens.iter().any(|t| t.contains("서울")),
        "expected '서울' in tokens: {tokens:?}"
    );
}

#[cfg(feature = "cjk-ja")]
#[test]
fn lindera_empty_input() {
    let mut tok = LinderaTokenizer::new(CjkLanguage::Japanese).unwrap();
    let mut stream = tok.token_stream("");
    assert!(!stream.advance(), "empty input should produce no tokens");
}

// -- Punctuation filter tests --

#[cfg(any(feature = "cjk-ja", feature = "cjk-ko"))]
#[test]
fn cjk_punctuation_detected() {
    assert!(is_cjk_punctuation("。"));
    assert!(is_cjk_punctuation("、"));
    assert!(is_cjk_punctuation("！"));
    assert!(!is_cjk_punctuation("東京"));
    assert!(!is_cjk_punctuation("hello"));
}

#[cfg(feature = "cjk-ja")]
#[test]
fn lindera_filters_punctuation() {
    let mut tok = LinderaTokenizer::new(CjkLanguage::Japanese).unwrap();
    let mut stream = tok.token_stream("東京、大阪。");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    assert!(
        !tokens.iter().any(|t| t == "、" || t == "。"),
        "punctuation should be filtered: {tokens:?}"
    );
}

// -- External dictionary loading tests --

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_with_dict_path_loads_file() {
    use std::io::Write;

    // Create a minimal jieba dictionary file with a custom word
    let dir = tempfile::tempdir().unwrap();
    let dict_path = dir.path().join("custom.dict");
    {
        let mut f = std::fs::File::create(&dict_path).unwrap();
        // jieba dict format: word freq tag (space-separated)
        writeln!(f, "coordinode 5 n").unwrap();
        writeln!(f, "数据库 10 n").unwrap();
    }

    let mut tok = JiebaTokenizer::with_dict_path(&dict_path).expect("load custom dict");
    let mut stream = tok.token_stream("coordinode是一个数据库");
    let mut tokens = Vec::new();
    while stream.advance() {
        tokens.push(stream.token().text.clone());
    }
    // Custom word "coordinode" should be recognized as a single token
    assert!(
        tokens.iter().any(|t| t == "coordinode"),
        "custom dict word should be recognized: {tokens:?}"
    );
}

#[cfg(feature = "cjk-zh")]
#[test]
fn jieba_with_dict_path_nonexistent_errors() {
    let result = JiebaTokenizer::with_dict_path(std::path::Path::new("/nonexistent/dict.txt"));
    assert!(result.is_err(), "nonexistent path should error");
}

#[cfg(feature = "cjk-zh")]
#[test]
fn cjk_dict_config_chinese_embedded() {
    use crate::cjk::CjkDictConfig;
    let config = CjkDictConfig::default();
    let tok = config.chinese_tokenizer().expect("embedded should work");
    let mut tok = tok;
    let mut stream = tok.token_stream("中文测试");
    assert!(stream.advance(), "embedded tokenizer should work");
}

#[cfg(feature = "cjk-ja")]
#[test]
fn lindera_with_dict_path_nonexistent_errors() {
    let result = LinderaTokenizer::with_dict_path(std::path::Path::new("/nonexistent/ipadic/"));
    assert!(result.is_err(), "nonexistent path should error");
}

#[cfg(feature = "cjk-ja")]
#[test]
fn cjk_dict_config_japanese_embedded() {
    use crate::cjk::CjkDictConfig;
    let config = CjkDictConfig::default();
    let tok = config.japanese_tokenizer().expect("embedded should work");
    let mut tok = tok;
    let mut stream = tok.token_stream("東京");
    assert!(stream.advance(), "embedded tokenizer should work");
}
