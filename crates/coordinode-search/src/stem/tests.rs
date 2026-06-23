use super::*;

#[test]
fn english_stemming() {
    assert_eq!(stem_word("running", "english"), "run");
    assert_eq!(stem_word("fruitlessly", "en"), "fruitless");
}

#[test]
fn russian_stemming() {
    assert_eq!(stem_word("бегущий", "russian"), "бегущ");
}

#[test]
fn ukrainian_stemming() {
    assert_eq!(stem_word("книга", "ukrainian"), "книг");
    assert_eq!(stem_word("учитель", "uk"), "учител");
    assert_eq!(stem_word("братові", "ukrainian"), "брат");
}

#[test]
fn german_stemming() {
    assert_eq!(stem_word("aufeinanderschlügen", "de"), "aufeinanderschlug");
}

#[test]
fn unknown_language_returns_original() {
    assert_eq!(stem_word("hello", "klingon"), "hello");
}

#[test]
fn supported_languages_count() {
    assert_eq!(supported_languages().len(), 20);
}

#[test]
fn algorithm_for_all_supported() {
    for lang in supported_languages() {
        assert!(
            algorithm_for_language(lang).is_some(),
            "algorithm_for_language({lang}) should return Some"
        );
    }
}

#[test]
fn iso_code_aliases() {
    assert!(algorithm_for_language("ar").is_some());
    assert!(algorithm_for_language("fr").is_some());
    assert!(algorithm_for_language("uk").is_some());
    assert!(algorithm_for_language("ro").is_some());
}
