use super::*;
use crate::encrypted::keys::SearchKey;

fn test_key() -> SearchKey {
    SearchKey::from_bytes(&[42u8; 32]).unwrap()
}

#[test]
fn stem_and_tokenize_english() {
    let key = test_key();
    let tokens = stem_and_tokenize("The runners are running quickly", "english", &key);

    let stems: Vec<&str> = tokens.iter().map(|t| t.stem.as_str()).collect();
    // "runners" → "runner", "running" → "run", "quickly" → "quick"
    // "the", "are" also stemmed but not removed (no stop word filter)
    assert!(!stems.is_empty());
    // Check deduplication: even though "runners" and "running" are different words,
    // if they stem to different stems they'll both appear
    let unique_stems: std::collections::HashSet<&str> = stems.iter().copied().collect();
    assert_eq!(
        stems.len(),
        unique_stems.len(),
        "stems should be deduplicated"
    );
}

#[test]
fn stem_and_tokenize_deterministic() {
    let key = test_key();
    let t1 = stem_and_tokenize("running fast", "english", &key);
    let t2 = stem_and_tokenize("running fast", "english", &key);

    assert_eq!(t1.len(), t2.len());
    for (a, b) in t1.iter().zip(t2.iter()) {
        assert_eq!(a.stem, b.stem);
        assert_eq!(a.token, b.token, "same text + key must produce same tokens");
    }
}

#[test]
fn different_surface_forms_same_stem_same_token() {
    let key = test_key();
    // "runner" and "running" should both stem to "run" in English
    let t_runner = stem_query_token("runner", "english", &key).unwrap();
    let t_running = stem_query_token("running", "english", &key).unwrap();

    // Both produce HMAC of "run" or "runner" — depends on Snowball output
    // At minimum, they should be consistent: same stem → same token
    let stem_runner = stem::stem_word("runner", "english");
    let stem_running = stem::stem_word("running", "english");

    if stem_runner == stem_running {
        assert_eq!(t_runner, t_running, "same stem should produce same token");
    }
}

#[test]
fn stem_query_token_empty_returns_none() {
    let key = test_key();
    assert!(stem_query_token("", "english", &key).is_none());
    assert!(stem_query_token("   ", "english", &key).is_none());
    assert!(stem_query_token("!!!", "english", &key).is_none());
}

#[test]
fn stem_query_token_basic() {
    let key = test_key();
    let token = stem_query_token("database", "english", &key);
    assert!(token.is_some());
}

#[test]
fn different_keys_different_tokens() {
    let k1 = SearchKey::from_bytes(&[1u8; 32]).unwrap();
    let k2 = SearchKey::from_bytes(&[2u8; 32]).unwrap();

    let t1 = stem_query_token("running", "english", &k1).unwrap();
    let t2 = stem_query_token("running", "english", &k2).unwrap();

    assert_ne!(t1, t2, "different keys must produce different tokens");
}

#[test]
fn stem_and_tokenize_deduplicates() {
    let key = test_key();
    // If two words stem to the same form, only one token should be produced
    let tokens = stem_and_tokenize("run running runner", "english", &key);

    // Count unique tokens
    let unique_tokens: std::collections::HashSet<_> =
        tokens.iter().map(|t| t.token.clone()).collect();
    // All three words may or may not stem to same root depending on Snowball
    // But within the result, stems are deduplicated
    assert_eq!(tokens.len(), unique_tokens.len());
}

#[test]
fn stem_and_tokenize_punctuation_stripped() {
    let key = test_key();
    let tokens = stem_and_tokenize("hello, world!", "english", &key);
    let stems: Vec<&str> = tokens.iter().map(|t| t.stem.as_str()).collect();
    // Punctuation should be stripped before stemming
    assert!(
        !stems.iter().any(|s| s.contains(',')),
        "punctuation should be removed"
    );
}

#[test]
fn stem_and_tokenize_russian() {
    let key = test_key();
    let tokens = stem_and_tokenize("Графовая база данных для аналитики", "russian", &key);
    assert!(!tokens.is_empty(), "Russian text should produce stems");
}

#[test]
fn stem_and_tokenize_auto_detect() {
    let key = test_key();
    let tokens = stem_and_tokenize_auto(
        "The runners are running through the beautiful forest in spring",
        &key,
    );
    assert!(
        !tokens.is_empty(),
        "auto-detect should produce tokens for English text"
    );
}

#[test]
fn end_to_end_stemmed_sse_search() {
    // Full flow: write stemmed tokens → search with stemmed query
    use crate::encrypted::index::EncryptedFieldIndex;

    let key = test_key();
    let mut idx = EncryptedFieldIndex::new();

    // WRITE: stem text, generate tokens, store in index
    let write_tokens = stem_and_tokenize("running through the forest", "english", &key);
    for st in &write_tokens {
        idx.insert(st.token.clone(), 1);
    }

    // Also index a second document
    let write_tokens2 = stem_and_tokenize("swimming in the ocean", "english", &key);
    for st in &write_tokens2 {
        idx.insert(st.token.clone(), 2);
    }

    // SEARCH: "runner" → stem → token → lookup
    let query_token = stem_query_token("runner", "english", &key).unwrap();
    let _results_runner = idx.search(&query_token).unwrap();
    // "runner" stems to "runner" in Snowball English
    // "running" stems to "run" — different stems, may not cross-match

    // Search for "run" which should match "running" (both stem to "run")
    let query_token_run = stem_query_token("run", "english", &key).unwrap();
    let results_run = idx.search(&query_token_run).unwrap();

    // "run" stems to "run", "running" stems to "run" → should match
    assert!(
        results_run.contains(&1),
        "'run' should match doc containing 'running': stems={:?}",
        write_tokens.iter().map(|t| &t.stem).collect::<Vec<_>>()
    );

    // "swim" should match "swimming"
    let query_swim = stem_query_token("swim", "english", &key).unwrap();
    let results_swim = idx.search(&query_swim).unwrap();
    assert!(
        results_swim.contains(&2),
        "'swim' should match doc containing 'swimming'"
    );

    // "run" should NOT match doc 2 (swimming)
    assert!(
        !results_run.contains(&2),
        "'run' should not match 'swimming' doc"
    );
}

#[test]
fn end_to_end_stemmed_sse_storage() {
    use crate::encrypted::storage::EncryptedIndex;
    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};

    let dir = tempfile::tempdir().unwrap();
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&config).unwrap();
    let key = test_key();

    // WRITE: stem + tokenize + persist, buffered on one transaction.
    let idx = EncryptedIndex::new("Article", "body_stems");
    let stems = stem_and_tokenize("running through the forest", "english", &key);
    let stems2 = stem_and_tokenize("swimming in the ocean", "english", &key);
    {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(&engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        for st in &stems {
            idx.insert(&mut txn, &st.token, 1).unwrap();
        }
        for st in &stems2 {
            idx.insert(&mut txn, &st.token, 2).unwrap();
        }
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).unwrap();
    }

    // SEARCH: "run" → stem → HMAC → storage lookup
    let query_token = stem_query_token("run", "english", &key).unwrap();
    let rt_oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let rt_read_ts = rt_oracle.next();
    let rtxn = Transaction::new(
        &engine,
        Some(&rt_oracle),
        rt_read_ts,
        Some(engine.snapshot()),
    );
    let results = idx.search(&rtxn, &query_token).unwrap();
    assert!(results.contains(&1), "'run' should find doc 1 in SSE index");
    assert!(
        !results.contains(&2),
        "'run' should not find doc 2 (swimming)"
    );
}
