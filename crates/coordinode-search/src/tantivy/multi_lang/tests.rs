use super::*;

fn props(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn basic_single_language() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(1, &props(&[("body", "the runners are running fast")]))
        .unwrap();

    // Default search uses English stemming
    let results = idx.search("run", 10).unwrap();
    assert!(
        !results.is_empty(),
        "English stemmed search should match 'runners/running'"
    );
}

#[test]
fn explicit_per_field_analyzer() {
    // Level 1: explicit per-field overrides default and auto-detect
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english")
        .with_field_analyzer("title_ru", "russian")
        .with_field_analyzer("title_en", "english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[
            ("title_en", "the runners are running"),
            ("title_ru", "бегущий человек быстро бежал по дороге"),
        ]),
    )
    .unwrap();

    // English field searchable with English stems
    let en = idx.search_with_language("run", 10, "english").unwrap();
    assert!(!en.is_empty(), "English field should be searchable");

    // Russian field searchable with Russian stems
    let ru = idx.search_with_language("бежать", 10, "russian").unwrap();
    assert!(!ru.is_empty(), "Russian field should be searchable");
}

#[test]
fn per_node_language_override() {
    // Level 2: _language property overrides default
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // Node with Russian override
    idx.add_node(
        1,
        &props(&[
            ("body", "бегущий человек быстро бежал по дороге"),
            ("_language", "russian"),
        ]),
    )
    .unwrap();

    // Should be searchable with Russian stems (despite English default)
    let results = idx.search_with_language("бежать", 10, "russian").unwrap();
    assert!(
        !results.is_empty(),
        "_language override should apply Russian tokenizer"
    );
}

#[test]
fn auto_detect_fallback() {
    // Level 3: auto-detect when no explicit or override
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // Long enough text for reliable detection
    idx.add_node(
        1,
        &props(&[(
            "body",
            "бегущий человек быстро бежал по дороге к реке через лес",
        )]),
    )
    .unwrap();

    // Auto-detect should identify Russian and use Russian stemming
    let results = idx.search_with_language("бежать", 10, "russian").unwrap();
    assert!(
        !results.is_empty(),
        "auto-detect should identify Russian for long Russian text"
    );
}

#[test]
fn default_language_fallback() {
    // Level 4: default when detection fails (short text)
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("none");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // Very short text — detection may fail, falls back to "none"
    idx.add_node(1, &props(&[("body", "ok")])).unwrap();

    let results = idx.search_with_language("ok", 10, "none").unwrap();
    assert!(!results.is_empty(), "short text should use default 'none'");
}

#[test]
fn none_language_for_identifiers() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english")
        .with_field_analyzer("key", "none")
        .with_field_analyzer("description", "auto_detect");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[
            ("key", "ERR_CONNECTION_REFUSED"),
            (
                "description",
                "Connection to the remote server was refused by the operating system",
            ),
        ]),
    )
    .unwrap();

    // "none" field: exact match (case-insensitive)
    let key_results = idx
        .search_with_language("err_connection_refused", 10, "none")
        .unwrap();
    assert!(
        !key_results.is_empty(),
        "'none' analyzer should match exact identifier"
    );

    // "auto_detect" field: stemmed match
    let desc_results = idx
        .search_with_language("connection", 10, "english")
        .unwrap();
    assert!(
        !desc_results.is_empty(),
        "auto-detected field should be searchable"
    );
}

#[test]
fn mixed_language_documents() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // English doc (auto-detected)
    idx.add_node(
        1,
        &props(&[(
            "body",
            "the runners are running through the beautiful forest in spring",
        )]),
    )
    .unwrap();

    // Russian doc with override
    idx.add_node(
        2,
        &props(&[
            ("body", "бегущий человек быстро бежал по дороге к реке"),
            ("_language", "russian"),
        ]),
    )
    .unwrap();

    assert_eq!(idx.num_docs(), 2);

    // Each doc searchable in its language
    let en = idx.search_with_language("run", 10, "english").unwrap();
    assert!(en.iter().any(|r| r.node_id == 1), "English doc findable");

    let ru = idx.search_with_language("бежать", 10, "russian").unwrap();
    assert!(ru.iter().any(|r| r.node_id == 2), "Russian doc findable");
}

#[test]
fn batch_add_nodes() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_nodes_batch(&[
        (
            1,
            props(&[(
                "body",
                "the runners are running through the beautiful forest in spring",
            )]),
        ),
        (
            2,
            props(&[
                ("body", "бегущий человек быстро бежал по дороге к реке"),
                ("_language", "russian"),
            ]),
        ),
    ])
    .unwrap();

    assert_eq!(idx.num_docs(), 2);
}

#[test]
fn delete_from_multilang() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[(
            "body",
            "the runners are running through the beautiful forest in spring",
        )]),
    )
    .unwrap();
    assert_eq!(idx.num_docs(), 1);

    idx.delete_document(1).unwrap();
    assert_eq!(idx.num_docs(), 0);
}

#[test]
fn custom_override_property() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english").with_override_property("lang");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[
            ("body", "бегущий человек быстро бежал по дороге к реке"),
            ("lang", "russian"), // custom property name
        ]),
    )
    .unwrap();

    let results = idx.search_with_language("бежать", 10, "russian").unwrap();
    assert!(!results.is_empty(), "custom override property should work");
}

#[test]
fn field_filter_respects_config() {
    // When field_analyzers is non-empty, only listed fields are indexed
    let dir = tempfile::tempdir().unwrap();
    let config =
        MultiLangConfig::with_default_language("english").with_field_analyzer("title", "english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[
            ("title", "graph database technology"),
            ("internal_notes", "secret information should not be indexed"),
        ]),
    )
    .unwrap();

    // Title is indexed
    let title = idx.search("graph", 10).unwrap();
    assert!(!title.is_empty(), "configured field should be indexed");

    // Internal notes should NOT be indexed (not in field_analyzers)
    let notes = idx.search("secret", 10).unwrap();
    assert!(notes.is_empty(), "unconfigured field should not be indexed");
}

#[test]
fn cascade_priority_explicit_over_override() {
    // Level 1 (explicit) takes priority over level 2 (_language override)
    let dir = tempfile::tempdir().unwrap();
    let config =
        MultiLangConfig::with_default_language("english").with_field_analyzer("title", "none"); // explicit: no stemming
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[
            ("title", "runners running"),
            ("_language", "english"), // override says English
        ]),
    )
    .unwrap();

    // Explicit "none" should win — no stemming applied
    // So "run" should NOT match "runners"
    let results = idx.search_with_language("run", 10, "none").unwrap();
    let has_match = results.iter().any(|r| r.node_id == 1);
    // With "none", "runners" is indexed as "runners", not "run"
    // Searching for "run" with "none" tokenizer gives term "run"
    // which doesn't match "runners"
    assert!(
        !has_match,
        "explicit 'none' should override _language 'english'"
    );
}

/// Ukrainian default language — indexing + stemmed search via MultiLanguageTextIndex.
///
/// Uses `default_language = "ukrainian"`. Documents are added via `add_node` (the
/// normal path). Search uses `search_with_language("ukrainian")` which must apply the
/// same Snowball Ukrainian stemmer at query time, matching the indexed stems.
///
/// "книга" (book) → stem "книг".  Searching for "книга" should find node 1 because
/// the query is also stemmed to "книг" before lookup.
#[test]
fn ukrainian_default_language_stemmed_search() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("ukrainian");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(1, &props(&[("body", "книга про історію міста")]))
        .unwrap();
    idx.add_node(2, &props(&[("body", "кішка сидить на дивані")]))
        .unwrap();

    // Query "книга" → stemmed to "книг" at query time → matches doc 1
    let results = idx.search_with_language("книга", 10, "ukrainian").unwrap();
    assert_eq!(
        results.len(),
        1,
        "Ukrainian stemmer: 'книга' should match indexed 'книга' (both stem to 'книг')"
    );
    assert_eq!(results[0].node_id, 1);

    // Unrelated query should not match
    let no_match = idx
        .search_with_language("автомобіль", 10, "ukrainian")
        .unwrap();
    assert!(no_match.is_empty(), "unrelated term should not match");
}

/// Ukrainian auto-detect: document with Ukrainian text, no explicit language set.
///
/// When `default_language = "english"` but the document text is Ukrainian, whatlang
/// auto-detection should kick in and index with the Ukrainian pipeline. Searching
/// with explicit `language="ukrainian"` must find it.
#[test]
fn ukrainian_auto_detect_indexes_with_ukrainian_stemmer() {
    let dir = tempfile::tempdir().unwrap();
    // Default is english — but Ukrainian text will be auto-detected
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // Distinctly Ukrainian text — whatlang should detect Lang::Ukr
    idx.add_node(
        1,
        &props(&[("body", "Україна розташована у Східній Європі")]),
    )
    .unwrap();

    // Searching with Ukrainian stemmer should find the node
    let results = idx
        .search_with_language("Україна", 10, "ukrainian")
        .unwrap();
    assert!(
        !results.is_empty(),
        "auto-detected Ukrainian text should be findable via ukrainian search"
    );
}

/// search_with_highlights on Ukrainian default-language index returns results.
///
/// This exercises Path A of TextService (non-fuzzy, no explicit language):
/// MultiLanguageTextIndex.search_with_highlights → search_with_highlights_and_language
/// with `default_language = "ukrainian"`.
#[test]
fn ukrainian_search_with_highlights() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("ukrainian");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    idx.add_node(
        1,
        &props(&[("body", "бібліотека програмного забезпечення")]),
    )
    .unwrap();
    idx.add_node(2, &props(&[("body", "рецепти приготування їжі")]))
        .unwrap();

    // "бібліотек" is stem of "бібліотека" — Path A queries via search_with_highlights
    let results = idx.search_with_highlights("бібліотека", 10).unwrap();
    assert!(
        !results.is_empty(),
        "search_with_highlights on Ukrainian index should find 'бібліотека'"
    );
    assert_eq!(results[0].node_id, 1);

    // Unrelated query
    let no = idx.search_with_highlights("автомобіль", 10).unwrap();
    assert!(no.is_empty());
}

#[test]
fn empty_properties_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();

    // Empty properties — should not crash, should not add doc
    idx.add_node(1, &props(&[])).unwrap();
    assert_eq!(idx.num_docs(), 0, "empty node should not be indexed");
}
