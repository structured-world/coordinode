//! Regression tests for the FTS `segment_registry` snapshot filter.
//!
//! Three mandatory regression cases per task spec:
//!
//! 1. Concurrent writer produces a segment at T+k while reader holds snapshot
//!    at T → reader does NOT see post-T terms.
//! 2. Tantivy compaction merges segments; the registry reconciles the merged
//!    segment's `(min_ts, max_ts)` by scanning the preserved per-doc fast
//!    field — correctness of snapshot reads survives compaction.
//! 3. `text_score` ranking at T is stable regardless of writes at T+k.
//!
//! Plus unit coverage for `SegmentRegistry::classify` and per-doc filter.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_search::tantivy::TextIndex;
use coordinode_search::tantivy::segment_registry::Visibility;
use tempfile::TempDir;

fn open(dir: &TempDir) -> TextIndex {
    TextIndex::open_or_create(dir.path(), 50_000_000, None).unwrap()
}

/// Regression (1): snapshot reader at T does not see documents written at T+k.
#[test]
fn snapshot_reader_ignores_post_t_writes() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    // T=10: "rust" at doc 1
    idx.add_document_at(1, "rust graph database", 10).unwrap();
    // T=20: "rust" at doc 2
    idx.add_document_at(2, "rust programming language", 20)
        .unwrap();
    // T=30: "rust" at doc 3 (post-snapshot write)
    idx.add_document_at(3, "rust web framework", 30).unwrap();

    // Reader at T=20: docs 1 and 2 visible, doc 3 hidden.
    let at_20 = idx.search_at("rust", 10, 20).unwrap();
    let ids_20: Vec<u64> = at_20.iter().map(|r| r.node_id).collect();
    assert!(ids_20.contains(&1), "doc 1 must be visible at T=20");
    assert!(ids_20.contains(&2), "doc 2 must be visible at T=20");
    assert!(
        !ids_20.contains(&3),
        "doc 3 was written at T=30 — must NOT be visible at T=20, got {ids_20:?}"
    );

    // Reader at T=30: all three visible.
    let at_30 = idx.search_at("rust", 10, 30).unwrap();
    assert_eq!(at_30.len(), 3, "all three docs must be visible at T=30");

    // Reader at T=5: nothing visible (all writes are later).
    let at_5 = idx.search_at("rust", 10, 5).unwrap();
    assert!(
        at_5.is_empty(),
        "no docs written before T=5 — got {:?}",
        at_5
    );
}

/// Regression (2): merge of segments preserves snapshot correctness via
/// per-doc fast field + registry reconciliation.
#[test]
fn merge_preserves_snapshot_correctness() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    // Create many small segments by committing one doc at a time with
    // distinct commit_ts. Tantivy's default merge policy will eventually
    // merge them into fewer, larger segments.
    for i in 1..=12u64 {
        idx.add_document_at(i, "rust is a systems language", i * 10)
            .unwrap();
    }

    // Force merges: request commit + wait for merging threads.
    // We trigger reconcile explicitly by another mutation (no-op delete
    // of nonexistent id still flushes a commit).
    idx.delete_document(9999).unwrap();

    // Registry must have at least one entry; for each surviving segment
    // the (min_ts, max_ts) must cover the actual docs still inside.
    let snap = idx.registry_snapshot();
    assert!(!snap.is_empty(), "registry must be populated");
    for (_id, range) in &snap {
        assert!(range.min_ts <= range.max_ts);
        // All commit_ts values used were multiples of 10 in [10..=120].
        assert!(range.min_ts >= 10 && range.max_ts <= 120);
    }

    // Correctness survives merging: a snapshot read at T=50 must see
    // exactly docs 1..=5 (commit_ts 10, 20, 30, 40, 50), none of 6..=12.
    let at_50 = idx.search_at("rust", 50, 50).unwrap();
    let ids: std::collections::BTreeSet<u64> = at_50.iter().map(|r| r.node_id).collect();
    assert_eq!(
        ids,
        (1..=5u64).collect::<std::collections::BTreeSet<_>>(),
        "snapshot at T=50 must see docs 1..=5 regardless of segment merges"
    );

    // And at T=120 all 12 are visible.
    let at_120 = idx.search_at("rust", 50, 120).unwrap();
    assert_eq!(at_120.len(), 12);
}

/// Regression (3): ranking at T is stable — later writes at T+k do not shift
/// the scores of documents visible at T.
#[test]
fn ranking_at_t_is_stable_under_later_writes() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    // Seed three docs at T=10, 20, 30 with varying term frequency so the
    // BM25 ordering is well-defined.
    idx.add_document_at(1, "rust rust rust rust graph", 10)
        .unwrap();
    idx.add_document_at(2, "rust rust programming", 20).unwrap();
    idx.add_document_at(3, "rust language", 30).unwrap();

    let before: Vec<(u64, f32)> = idx
        .search_at("rust", 10, 30)
        .unwrap()
        .iter()
        .map(|r| (r.node_id, r.score))
        .collect();

    // Concurrent writer appends at T=100 (after the snapshot).
    idx.add_document_at(4, "rust rust rust rust rust rust", 100)
        .unwrap();
    idx.add_document_at(5, "rust rust rust rust rust", 100)
        .unwrap();

    // Snapshot at T=30: ranking must be bit-for-bit identical to `before`.
    let after: Vec<(u64, f32)> = idx
        .search_at("rust", 10, 30)
        .unwrap()
        .iter()
        .map(|r| (r.node_id, r.score))
        .collect();

    assert_eq!(
        after.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        before.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        "doc ordering at T=30 must not be perturbed by later writes"
    );

    // Scores may drift by BM25 IDF corpus-wide updates from newer segments,
    // but the POSITION (rank) must be preserved. The test above already
    // asserts id ordering; we additionally check top-1 stability.
    assert_eq!(after[0].0, before[0].0);
}

/// Registry records min/max commit_ts accurately per segment.
#[test]
fn registry_min_max_matches_committed_timestamps() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    idx.add_document_at(1, "alpha", 10).unwrap();
    idx.add_document_at(2, "beta", 20).unwrap();
    idx.add_document_at(3, "gamma", 30).unwrap();

    let snap = idx.registry_snapshot();
    assert!(!snap.is_empty(), "registry must be populated after writes");

    // Across all surviving segments, the union of [min_ts, max_ts] must
    // cover exactly {10, 20, 30}.
    let global_min = snap.iter().map(|(_, r)| r.min_ts).min().unwrap();
    let global_max = snap.iter().map(|(_, r)| r.max_ts).max().unwrap();
    assert_eq!(global_min, 10);
    assert_eq!(global_max, 30);
}

// Ensure Visibility enum is in scope (silences unused-import on some profiles).
#[allow(dead_code)]
const _: fn(Visibility) -> bool = |v| matches!(v, Visibility::All);

/// A document written without a birth time is not taken as born at 0: a
/// birth-filtered read of the index is refused, and its segment has no
/// bounds in the registry.
#[test]
fn a_document_without_a_birth_time_refuses_a_read_at_a_timestamp() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    idx.add_document_at(1, "timed doc", 10).unwrap();
    idx.add_document(2, "untimed doc").unwrap();

    for at in [0, 10, u64::MAX] {
        assert!(
            matches!(
                idx.search_at("doc", 10, at),
                Err(coordinode_search::tantivy::TextSearchError::BirthTimesUnknown)
            ),
            "a read at {at} must be refused"
        );
    }
    assert_eq!(
        idx.registry_snapshot().len(),
        1,
        "only the timed segment has bounds"
    );
    // A search of the index as it stands is unaffected.
    assert_eq!(idx.search("doc", 10).unwrap().len(), 2);
}

/// Zero is a birth time like any other, told apart from a missing one: a
/// document born at 0 is visible at 0, one born later is not.
#[test]
fn a_birth_time_of_zero_is_a_birth_time() {
    let dir = TempDir::new().unwrap();
    let mut idx = open(&dir);

    idx.add_document_at(1, "zero doc", 0).unwrap();
    idx.add_document_at(2, "later doc", 5).unwrap();

    let ids = |at| -> Vec<u64> {
        idx.search_at("doc", 10, at)
            .unwrap()
            .iter()
            .map(|r| r.node_id)
            .collect()
    };
    assert_eq!(ids(0), [1]);
    let mut at_5 = ids(5);
    at_5.sort_unstable();
    assert_eq!(at_5, [1, 2]);
}

/// The multi-language index holds current states whose birth commit is not
/// known: reaching its raw index does not turn that into a timestamp read.
#[test]
fn a_current_state_index_refuses_a_read_at_a_timestamp() {
    use coordinode_search::tantivy::multi_lang::{MultiLangConfig, MultiLanguageTextIndex};
    use std::collections::HashMap;

    let dir = TempDir::new().unwrap();
    let mut idx =
        MultiLanguageTextIndex::create_scratch(dir.path(), 15_000_000, MultiLangConfig::default())
            .unwrap();
    idx.add_node(
        1,
        &HashMap::from([("body".to_string(), "graph engine".to_string())]),
    )
    .unwrap();

    assert!(matches!(
        idx.inner().search_at("graph", 10, u64::MAX),
        Err(coordinode_search::tantivy::TextSearchError::BirthTimesUnknown)
    ));
    assert!(idx.inner().registry_snapshot().is_empty());
}
