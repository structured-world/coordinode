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

/// One batch replaces the upserted documents and takes out the removed ones
/// together; a removal of a node the index does not hold changes nothing.
#[test]
fn a_batch_upserts_and_removes_in_one_commit() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_nodes_batch(&[
        (1, props(&[("body", "old words")])),
        (2, props(&[("body", "doomed text")])),
    ])
    .unwrap();

    idx.apply_changes(&[(1, props(&[("body", "fresh words")]))], &[2, 99])
        .unwrap();

    assert_eq!(idx.num_docs(), 1);
    assert!(idx.contains(1).unwrap());
    assert!(!idx.contains(2).unwrap(), "the removed node is gone");
    assert!(
        idx.search("old", 10).unwrap().is_empty(),
        "the old text is replaced"
    );
    assert_eq!(idx.search("fresh", 10).unwrap().len(), 1);
    assert!(idx.search("doomed", 10).unwrap().is_empty());
}

/// A batch whose removals name only nodes the index does not hold leaves
/// the index exactly as it was.
#[test]
fn removing_nodes_the_index_does_not_hold_changes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_node(1, &props(&[("body", "kept")])).unwrap();

    idx.apply_changes(&[], &[7, 8]).unwrap();

    assert_eq!(idx.num_docs(), 1);
    assert!(idx.contains(1).unwrap());
    assert!(!idx.contains(7).unwrap());
}

/// Replacing the whole index drops every document not listed, so a rebuild
/// from the store leaves nothing of a node deleted meanwhile.
#[test]
fn replace_all_drops_documents_not_listed() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_nodes_batch(&[
        (1, props(&[("body", "stays here")])),
        (2, props(&[("body", "deleted meanwhile")])),
    ])
    .unwrap();

    idx.replace_all(&[(1, props(&[("body", "stays here")]))])
        .unwrap();

    assert_eq!(idx.num_docs(), 1);
    assert!(!idx.contains(2).unwrap());
    assert!(idx.search("deleted", 10).unwrap().is_empty());
    assert_eq!(idx.search("stays", 10).unwrap().len(), 1);
}

/// Scores of `query` by node over the index alone.
fn scores(
    idx: &MultiLanguageTextIndex,
    query: &str,
    pending: &PendingDocuments,
) -> Vec<(u64, f32)> {
    let mut hits: Vec<(u64, f32)> = idx
        .find(terms(query), Matches::All, pending)
        .unwrap()
        .into_iter()
        .map(|hit| (hit.node_id, hit.score))
        .collect();
    hits.sort_by_key(|(id, _)| *id);
    hits
}

fn fresh(dir: &std::path::Path, docs: &[(u64, &str)]) -> MultiLanguageTextIndex {
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::create_scratch(dir, 15_000_000, config).unwrap();
    let docs: Vec<_> = docs
        .iter()
        .map(|(id, body)| (*id, props(&[("body", body)])))
        .collect();
    idx.add_nodes_batch(&docs).unwrap();
    idx
}

fn assert_same_scores(a: &[(u64, f32)], b: &[(u64, f32)]) {
    assert_eq!(a.len(), b.len(), "{a:?} vs {b:?}");
    for ((ia, sa), (ib, sb)) in a.iter().zip(b) {
        assert_eq!(ia, ib, "{a:?} vs {b:?}");
        assert!((sa - sb).abs() < 1e-5, "{a:?} vs {b:?}");
    }
}

/// Scores count the documents the index holds, not the ones it replaced or
/// removed: an index that rewrote its documents ranks exactly as one built
/// from the final documents, although Tantivy still holds the old ones until
/// their segments merge.
#[test]
fn replaced_documents_leave_the_statistics() {
    let dirs = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let mut rewritten = fresh(
        dirs.0.path(),
        &[(1, "rust graph rust"), (2, "rust engine"), (3, "rust")],
    );
    rewritten
        .apply_changes(&[(1, props(&[("body", "python scripts")]))], &[3])
        .unwrap();
    rewritten
        .apply_changes(&[(4, props(&[("body", "graph rust database")]))], &[])
        .unwrap();
    let built = fresh(
        dirs.1.path(),
        &[
            (1, "python scripts"),
            (2, "rust engine"),
            (4, "graph rust database"),
        ],
    );

    for query in ["rust", "graph", "python"] {
        assert_same_scores(
            &scores(&rewritten, query, &PendingDocuments::none()),
            &scores(&built, query, &PendingDocuments::none()),
        );
    }
}

/// A search with pending documents ranks as the index holding the final
/// documents would: the superseded ones leave the statistics and the pending
/// ones join them.
#[test]
fn pending_documents_score_as_the_final_corpus() {
    let dirs = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let behind = fresh(
        dirs.0.path(),
        &[(1, "rust graph"), (2, "rust engine"), (3, "java beans")],
    );
    let pending = behind
        .pending(
            Some(&[1, 3, 5]),
            &[
                (1, props(&[("body", "python scripts")])),
                (5, props(&[("body", "rust rust database")])),
            ],
        )
        .unwrap();
    let built = fresh(
        dirs.1.path(),
        &[
            (1, "python scripts"),
            (2, "rust engine"),
            (5, "rust rust database"),
        ],
    );

    for query in ["rust", "python", "database", "java"] {
        assert_same_scores(
            &scores(&behind, query, &pending),
            &scores(&built, query, &PendingDocuments::none()),
        );
    }
}

fn terms(query: &str) -> TextRequest<'_> {
    TextRequest::Terms {
        query,
        language: None,
        snippets: false,
    }
}

fn found(idx: &MultiLanguageTextIndex, query: &str, pending: &PendingDocuments) -> Vec<u64> {
    let mut ids: Vec<u64> = idx
        .find(terms(query), Matches::All, pending)
        .unwrap()
        .into_iter()
        .map(|hit| hit.node_id)
        .collect();
    ids.sort_unstable();
    ids
}

/// A node written since the index's position is answered from its current
/// document: the stale indexed text no longer matches, the new one does, and
/// a node removed since matches nothing; the index answers for every other
/// node.
#[test]
fn pending_documents_stand_in_for_the_index_own() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_nodes_batch(&[
        (1, props(&[("body", "stale words about graphs")])),
        (2, props(&[("body", "removed words about graphs")])),
        (3, props(&[("body", "untouched words about graphs")])),
    ])
    .unwrap();

    // Node 1 now holds other text, node 2 was deleted, node 4 is new.
    let pending = idx
        .pending(
            Some(&[1, 2, 4]),
            &[
                (1, props(&[("body", "fresh words about vectors")])),
                (4, props(&[("body", "brand new graphs")])),
            ],
        )
        .unwrap();

    assert_eq!(found(&idx, "graphs", &pending), [3, 4]);
    assert_eq!(found(&idx, "vectors", &pending), [1]);
    assert_eq!(found(&idx, "stale", &pending), Vec::<u64>::new());
    assert_eq!(found(&idx, "removed", &pending), Vec::<u64>::new());
    // Nothing pending: the index alone, stale text included.
    assert_eq!(found(&idx, "graphs", &PendingDocuments::none()), [1, 2, 3]);
}

/// When which nodes changed is not known, the index answers for none of them
/// and the pending documents are the whole answer.
#[test]
fn unknown_superseded_nodes_leave_the_index_out() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_node(1, &props(&[("body", "graphs in the index")]))
        .unwrap();

    let pending = idx
        .pending(None, &[(5, props(&[("body", "graphs in the store")]))])
        .unwrap();

    assert_eq!(found(&idx, "graphs", &pending), [5]);
}

/// A pending document is scored with the index's corpus statistics, so the
/// same text ranks the same whether the index or the pending segment holds
/// it, and the merged list is in one score order.
#[test]
fn pending_documents_score_on_the_index_scale() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_nodes_batch(&[
        (1, props(&[("body", "rust graph engine")])),
        (2, props(&[("body", "python scripts")])),
        (3, props(&[("body", "java beans")])),
    ])
    .unwrap();

    let pending = idx
        .pending(Some(&[9]), &[(9, props(&[("body", "rust graph engine")]))])
        .unwrap();
    let hits = idx.find(terms("rust"), Matches::All, &pending).unwrap();

    assert_eq!(hits.len(), 2);
    assert!(
        (hits[0].score - hits[1].score).abs() < 1e-6,
        "identical text scores alike in the index and the pending segment: {hits:?}"
    );
}

/// `Matches::Top` keeps the best `n` of the merged list; `Matches::All`
/// keeps every match, as a membership filter needs.
#[test]
fn matches_bound_the_merged_list() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let mut idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    idx.add_nodes_batch(&[
        (1, props(&[("body", "graph")])),
        (2, props(&[("body", "graph graph graph")])),
    ])
    .unwrap();
    let pending = idx
        .pending(Some(&[3]), &[(3, props(&[("body", "graph graph")]))])
        .unwrap();

    let top = idx.find(terms("graph"), Matches::Top(2), &pending).unwrap();
    assert_eq!(top.len(), 2);
    assert!(top[0].score >= top[1].score, "best first: {top:?}");
    assert_eq!(found(&idx, "graph", &pending), [1, 2, 3]);
}

/// Snippets of a pending document come from its current text.
#[test]
fn pending_documents_carry_snippets() {
    let dir = tempfile::tempdir().unwrap();
    let config = MultiLangConfig::with_default_language("english");
    let idx = MultiLanguageTextIndex::open_or_create(dir.path(), 15_000_000, config).unwrap();
    let pending = idx
        .pending(Some(&[1]), &[(1, props(&[("body", "a fresh graph")]))])
        .unwrap();

    let hits = idx
        .find(
            TextRequest::Terms {
                query: "graph",
                language: None,
                snippets: true,
            },
            Matches::Top(10),
            &pending,
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].snippet_html.contains("<b>graph</b>"),
        "{:?}",
        hits[0].snippet_html
    );
}

/// A write that fails on storage the index can no longer reach returns an
/// error every time it is retried, rather than panicking once the writer
/// was lost to the failed rollback of an earlier attempt.
#[test]
fn writes_to_a_lost_directory_keep_failing_without_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("idx");
    let mut idx =
        MultiLanguageTextIndex::create_scratch(&path, 15_000_000, MultiLangConfig::default())
            .unwrap();
    idx.add_node(1, &props(&[("body", "graph engine")]))
        .unwrap();
    std::fs::remove_dir_all(&path).unwrap();
    for attempt in 0..3 {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            idx.add_node(2, &props(&[("body", "lost write")]))
        }));
        assert!(outcome.is_ok(), "attempt {attempt} panicked");
        assert!(
            outcome.is_ok_and(|result| result.is_err()),
            "attempt {attempt} succeeded"
        );
    }
}
