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

/// Stored full-text postings and encrypted stem tokens were produced with
/// these stems, so a stemmer release that changes one breaks every index
/// built before it. Each supported language, a few inflected words and the
/// stem the stored data holds for them.
const STABLE_STEMS: &[(&str, &str, &str)] = &[
    ("arabic", "المكتبات", "مكتب"),
    ("arabic", "يكتبون", "يكتب"),
    ("arabic", "والكتاب", "والكتاب"),
    ("armenian", "գրքերը", "գրքերը"),
    ("armenian", "աշխատանքների", "աշխ"),
    ("armenian", "քաղաքում", "քաղա"),
    ("danish", "bøgerne", "bøg"),
    ("danish", "arbejdende", "arbejd"),
    ("danish", "huse", "hus"),
    ("dutch", "boeken", "boek"),
    ("dutch", "werkende", "werk"),
    ("dutch", "huizen", "huis"),
    ("english", "libraries", "librari"),
    ("english", "connections", "connect"),
    ("english", "generously", "generous"),
    ("finnish", "kirjastoissa", "kirjasto"),
    ("finnish", "taloissa", "talo"),
    ("finnish", "juoksemassa", "juoksem"),
    ("french", "bibliothèques", "bibliothequ"),
    ("french", "continuellement", "continuel"),
    ("french", "nationales", "national"),
    ("german", "Bibliotheken", "Bibliothek"),
    ("german", "Verbindungen", "Verbind"),
    ("german", "häuslichen", "hauslich"),
    ("greek", "βιβλιοθήκες", "βιβλιοθηκ"),
    ("greek", "ανθρώπων", "ανθρωπ"),
    ("greek", "γράφοντας", "γραφ"),
    ("hungarian", "könyvtárakban", "könyvtár"),
    ("hungarian", "házakat", "ház"),
    ("hungarian", "emberekkel", "ember"),
    ("italian", "biblioteche", "bibliotec"),
    ("italian", "continuamente", "continu"),
    ("italian", "nazionali", "nazional"),
    ("norwegian", "bøkene", "bøk"),
    ("norwegian", "arbeidende", "arbeid"),
    ("norwegian", "husene", "hus"),
    ("portuguese", "bibliotecas", "bibliotec"),
    ("portuguese", "continuamente", "continu"),
    ("portuguese", "nacionais", "nacion"),
    ("romanian", "bibliotecile", "bibliotec"),
    ("romanian", "continuitate", "continu"),
    ("romanian", "naționale", "național"),
    ("russian", "библиотеками", "библиотек"),
    ("russian", "соединения", "соединен"),
    ("russian", "бегущий", "бегущ"),
    ("spanish", "bibliotecas", "bibliotec"),
    ("spanish", "continuamente", "continu"),
    ("spanish", "nacionales", "nacional"),
    ("swedish", "biblioteken", "bibliotek"),
    ("swedish", "arbetande", "arbet"),
    ("swedish", "husen", "hus"),
    ("tamil", "புத்தகங்கள்", "புத்தகம்"),
    ("tamil", "வீடுகளில்", "வீடு"),
    ("tamil", "மக்களுக்கு", "மக்கள்"),
    ("turkish", "kütüphanelerde", "kütüphane"),
    ("turkish", "evlerimizden", "ev"),
    ("turkish", "kitaplar", "kitap"),
    ("ukrainian", "бібліотеками", "бібліотек"),
    ("ukrainian", "з'єднання", "з'єдн"),
    ("ukrainian", "братові", "брат"),
];

/// A stemmer update must leave every stem stored data relies on as it was;
/// a changed stem means rebuilding full-text indexes and encrypted tokens,
/// which an update must not do silently.
#[test]
fn stems_stay_the_ones_stored_data_was_built_with() {
    for &(lang, word, stem) in STABLE_STEMS {
        assert_eq!(stem_word(word, lang), stem, "{lang}: {word}");
    }
    let covered: std::collections::BTreeSet<&str> =
        STABLE_STEMS.iter().map(|&(lang, _, _)| lang).collect();
    for lang in supported_languages() {
        assert!(covered.contains(lang), "no stem pinned down for {lang}");
    }
}

#[test]
fn iso_code_aliases() {
    assert!(algorithm_for_language("ar").is_some());
    assert!(algorithm_for_language("fr").is_some());
    assert!(algorithm_for_language("uk").is_some());
    assert!(algorithm_for_language("ro").is_some());
}
