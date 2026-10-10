use coordinode_core::index::encoding::GENERATION_TAGS;
use coordinode_core::index::identity::GenerationId;

use super::super::prefix;
use super::HistoryEntry;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageError;

const GENERATION: GenerationId = GenerationId::from_raw(40);
const TAG_ENTRIES: u8 = GENERATION_TAGS[0];

fn config(path: &std::path::Path) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        path,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn entry(rest: &[u8]) -> Vec<u8> {
    let mut out = prefix(TAG_ENTRIES, GENERATION.as_raw());
    out.extend_from_slice(rest);
    out
}

fn get(engine: &StorageEngine, key: &[u8]) -> Option<Vec<u8>> {
    engine
        .get(Partition::Idx, key)
        .expect("get")
        .map(|v| v.to_vec())
}

fn get_at(engine: &StorageEngine, at: u64, key: &[u8]) -> Option<Vec<u8>> {
    engine
        .snapshot_get(&at, Partition::Idx, key)
        .expect("snapshot get")
        .map(|v| v.to_vec())
}

/// Installation ids holding stored entries of the generation's family, at
/// the latest seqno.
fn stored_installations(engine: &StorageEngine) -> Vec<u64> {
    use lsm_tree::{AbstractTree as _, Guard as _};
    let tree = engine.tree(Partition::Idx).expect("idx");
    let mut out: Vec<u64> = tree
        .range(vec![TAG_ENTRIES]..vec![TAG_ENTRIES + 1], u64::MAX, None)
        .map(|guard| {
            let key = guard.key().expect("key");
            super::super::slot(&key).expect("slot")
        })
        .collect();
    out.dedup();
    out
}

/// A replacement filled from the generation's own history and published
/// takes over every read: the latest values, the values at an earlier
/// snapshot, and a write made while it was prepared. The old copy's entries
/// are gone.
#[test]
fn a_published_replacement_answers_every_read_the_old_copy_did() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    let a = entry(b"a");
    let b = entry(b"b");
    let c = entry(b"c");
    let d = entry(b"d");
    engine.put(Partition::Idx, &a, b"a1").expect("put");
    engine.put(Partition::Idx, &b, b"b1").expect("put");
    let early = engine.snapshot();
    engine.put(Partition::Idx, &a, b"a2").expect("put");
    engine.delete(Partition::Idx, &b).expect("delete");
    let old = stored_installations(&engine);
    assert_eq!(old.len(), 1, "one copy before the replacement");

    engine.stage_generation(GENERATION, 0).expect("stage");
    // Written while the replacement is prepared: reaches both copies.
    engine.put(Partition::Idx, &c, b"c1").expect("put");
    let history = engine
        .export_generation_history(GENERATION)
        .expect("export");
    assert!(
        history
            .entries
            .iter()
            .any(|e| matches!(e, HistoryEntry::Delete { key, .. } if *key == b)),
        "the history carries the delete in logical form: {history:?}"
    );
    // Written after the history was taken: reaches the replacement only by
    // being applied to both copies.
    engine.put(Partition::Idx, &d, b"d1").expect("put");
    engine
        .import_generation_history(GENERATION, &history.entries)
        .expect("import");
    engine
        .finish_generation_import(GENERATION, history.covers_through)
        .expect("finish");
    engine.publish_generation(GENERATION).expect("publish");

    assert_eq!(get(&engine, &a), Some(b"a2".to_vec()));
    assert_eq!(get(&engine, &b), None);
    assert_eq!(get(&engine, &c), Some(b"c1".to_vec()));
    assert_eq!(get(&engine, &d), Some(b"d1".to_vec()));
    assert_eq!(get_at(&engine, early, &a), Some(b"a1".to_vec()));
    assert_eq!(get_at(&engine, early, &b), Some(b"b1".to_vec()));
    let now = stored_installations(&engine);
    assert_eq!(now.len(), 1, "only the replacement holds entries: {now:?}");
    assert_ne!(now, old, "the replacement is a new installation");

    // The swap survives a reopen.
    drop(engine);
    let engine = StorageEngine::open(&config(dir.path())).expect("reopen");
    assert_eq!(get(&engine, &a), Some(b"a2".to_vec()));
    assert_eq!(get(&engine, &c), Some(b"c1".to_vec()));
    assert_eq!(stored_installations(&engine), now);
}

/// A replacement whose contents were never recorded complete cannot be
/// published, and is dropped at the next open: the generation keeps its
/// copy and nothing is left under the replacement.
#[test]
fn an_unfinished_replacement_is_dropped_at_open() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    let a = entry(b"a");
    engine.put(Partition::Idx, &a, b"a1").expect("put");
    let old = stored_installations(&engine);
    engine.stage_generation(GENERATION, 0).expect("stage");
    engine.put(Partition::Idx, &a, b"a2").expect("put");
    assert!(matches!(
        engine.publish_generation(GENERATION),
        Err(StorageError::InstallationCatalog(_))
    ));
    // The replacement's record and entries reach disk.
    {
        use lsm_tree::AbstractTree as _;
        engine
            .tree(Partition::Idx)
            .expect("idx")
            .flush_active_memtable(0)
            .expect("flush");
    }
    assert_eq!(stored_installations(&engine).len(), 2);
    drop(engine);

    let engine = StorageEngine::open(&config(dir.path())).expect("reopen");
    assert_eq!(get(&engine, &a), Some(b"a2".to_vec()));
    assert_eq!(stored_installations(&engine), old);
    // A new replacement can be prepared again.
    engine.stage_generation(GENERATION, 0).expect("stage again");
}

/// After a replacement whose history starts at a seqno is published, a read
/// at an earlier snapshot is refused rather than answered from a history the
/// copy does not hold.
#[test]
fn a_read_below_the_replacement_history_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    let a = entry(b"a");
    engine.put(Partition::Idx, &a, b"a1").expect("put");
    let early = engine.snapshot();
    engine.put(Partition::Idx, &a, b"a2").expect("put");
    let from = engine.snapshot();
    engine.stage_generation(GENERATION, from).expect("stage");
    let history = engine
        .export_generation_history(GENERATION)
        .expect("export");
    let recent: Vec<_> = history
        .entries
        .iter()
        .filter(|e| e.seqno() >= from - 1)
        .cloned()
        .collect();
    engine
        .import_generation_history(GENERATION, &recent)
        .expect("import");
    engine
        .finish_generation_import(GENERATION, history.covers_through)
        .expect("finish");
    engine.publish_generation(GENERATION).expect("publish");

    assert_eq!(get_at(&engine, from, &a), Some(b"a2".to_vec()));
    assert!(matches!(
        engine.snapshot_get(&early, Partition::Idx, &a),
        Err(StorageError::SnapshotOutsideRetention { .. })
    ));
}

/// A history that ends before the point the replacement was registered at
/// lacks effects the generation holds, which reach the replacement neither
/// from the history nor from later writes: its import cannot complete.
#[test]
fn a_history_ending_before_the_registration_is_refused() {
    use coordinode_core::txn::proposal::{Mutation, PartitionId};
    use coordinode_core::txn::timestamp::TimestampOracle;

    let dir = tempfile::tempdir().expect("tempdir");
    let oracle = std::sync::Arc::new(TimestampOracle::new());
    let engine = StorageEngine::open_embedded(&config(dir.path()), oracle.clone()).expect("open");
    let commit = |key: &[u8], value: &[u8]| {
        let mutation = Mutation::Put {
            partition: PartitionId::Idx,
            key: key.to_vec(),
            value: value.to_vec(),
        };
        engine
            .commit_journaled(&[mutation], oracle.next().as_raw())
            .expect("commit");
    };
    let a = entry(b"a");
    let b = entry(b"b");
    commit(&a, b"a1");
    let early = engine
        .export_generation_history(GENERATION)
        .expect("export");
    // Applied after the history was taken and before the registration.
    commit(&b, b"b1");
    engine.stage_generation(GENERATION, 0).expect("stage");
    engine
        .import_generation_history(GENERATION, &early.entries)
        .expect("import");
    assert!(matches!(
        engine.finish_generation_import(GENERATION, early.covers_through),
        Err(StorageError::InstallationCatalog(_))
    ));

    // A history taken after the registration covers it.
    let late = engine
        .export_generation_history(GENERATION)
        .expect("export");
    engine
        .import_generation_history(GENERATION, &late.entries)
        .expect("import");
    engine
        .finish_generation_import(GENERATION, late.covers_through)
        .expect("finish");
    engine.publish_generation(GENERATION).expect("publish");
    assert_eq!(get(&engine, &b), Some(b"b1".to_vec()));
}

/// An imported entry outside the generation is refused, so a history meant
/// for one generation cannot write into another.
#[test]
fn an_entry_of_another_generation_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let engine = StorageEngine::open(&config(dir.path())).expect("open");
    engine
        .put(Partition::Idx, &entry(b"a"), b"a1")
        .expect("put");
    engine.stage_generation(GENERATION, 0).expect("stage");
    let mut other = prefix(TAG_ENTRIES, 41);
    other.extend_from_slice(b"a");
    let entries = [HistoryEntry::Put {
        key: other,
        value: b"x".to_vec(),
        seqno: 1,
    }];
    assert!(matches!(
        engine.import_generation_history(GENERATION, &entries),
        Err(StorageError::InstallationCatalog(_))
    ));
}
