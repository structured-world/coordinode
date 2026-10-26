use std::sync::Arc;

use coordinode_core::graph::blob::{
    CANDIDATE_PREFIX, CatalogEntry, ContentId, DEFAULT_CHUNK_SIZE, ScopedContent, SharingScope,
    catalog_key, direct_owner_key, payload_key, pins_counter_key, refs_counter_key, staging_key,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs};
use tempfile::TempDir;

use super::*;
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

fn plain_config(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

fn crash_config(dir: &TempDir, crash: &Arc<CrashFs>, faults: &Arc<FaultInjector>) -> StorageConfig {
    plain_config(dir).with_fs(Arc::new(FaultFs::with_injector(
        CrashFs::clone(crash),
        Arc::clone(faults),
    )))
}

fn store_over(config: &StorageConfig) -> BlobStore {
    let oracle = Arc::new(TimestampOracle::new());
    let engine = Arc::new(StorageEngine::open_embedded(config, Arc::clone(&oracle)).expect("open"));
    BlobStore::direct(engine, oracle)
}

fn open_store() -> (BlobStore, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    let store = store_over(&plain_config(&dir));
    (store, dir)
}

fn owner(n: u8) -> Vec<u8> {
    direct_owner_key(&[n; 16])
}

/// Put `data` as what owner `n` holds, replacing what it held.
fn put(store: &BlobStore, n: u8, data: &[u8]) -> PutOutcome {
    store
        .put(
            &owner(n),
            SharingScope::STORE,
            data,
            OwnerWrite::Replace,
            None,
        )
        .expect("put")
}

/// Publish `upload` as what owner `n` holds.
fn publish(store: &BlobStore, n: u8, upload: &StagedUpload) -> BlobResult<PutOutcome> {
    store.publish(&owner(n), OwnerWrite::Replace, upload, None)
}

/// Admit and write `data` without publishing it.
fn upload(store: &BlobStore, data: &[u8]) -> StagedUpload {
    store.upload(SharingScope::STORE, data).expect("upload")
}

fn scoped(data: &[u8]) -> ScopedContent {
    ScopedContent::new(SharingScope::STORE, ContentId::of(data))
}

fn count(store: &BlobStore, key: &[u8]) -> i64 {
    store
        .engine()
        .get(Partition::Counter, key)
        .expect("count")
        .map_or(0, |b| {
            crate::engine::merge::decode_counter(&b).expect("decode")
        })
}

fn refs(store: &BlobStore, content: &ScopedContent) -> i64 {
    count(store, &refs_counter_key(content))
}

fn pins(store: &BlobStore, content: &ScopedContent) -> i64 {
    count(store, &pins_counter_key(content))
}

fn entry(store: &BlobStore, content: &ScopedContent) -> CatalogEntry {
    let bytes = store
        .engine()
        .get(Partition::BlobRef, &catalog_key(content))
        .expect("get")
        .expect("catalog entry");
    CatalogEntry::from_msgpack(&bytes).expect("decode")
}

fn eligible_generation(store: &BlobStore, content: &ScopedContent) -> u64 {
    entry(store, content).eligible.expect("eligible").generation
}

fn read_all(store: &BlobStore, owner_key: &[u8]) -> Vec<u8> {
    let reader = store.open(owner_key).expect("open").expect("present");
    (0..reader.record().blob.chunk_count())
        .flat_map(|i| reader.chunk(i).expect("chunk"))
        .collect()
}

fn candidates(store: &BlobStore) -> usize {
    store
        .engine()
        .prefix_scan(Partition::BlobRef, CANDIDATE_PREFIX)
        .expect("scan")
        .count()
}

/// Two owners of one chunk: deleting one leaves the chunk referenced once and
/// readable by the other, and leaves a candidate rather than erasing it.
#[test]
fn a_shared_chunk_survives_one_owner_deletion() {
    let (store, _dir) = open_store();
    let shared = vec![0xAB_u8; DEFAULT_CHUNK_SIZE];
    let mut first = shared.clone();
    first.extend_from_slice(b"first");
    let mut second = shared.clone();
    second.extend_from_slice(b"second");
    put(&store, 1, &first);
    let outcome = put(&store, 2, &second);
    assert_eq!(
        outcome.shared, 1,
        "the second owner shares the common chunk"
    );
    assert_eq!(refs(&store, &scoped(&shared)), 2);

    let released = store.delete(&owner(1)).expect("delete");
    assert!(released.is_some());
    assert_eq!(refs(&store, &scoped(&shared)), 1);
    assert_eq!(refs(&store, &scoped(b"first")), 0);
    assert_eq!(read_all(&store, &owner(2)), second);
    assert!(store.open(&owner(1)).expect("open").is_none());
    // One candidate per content the owner let go.
    assert_eq!(candidates(&store), 2);
}

/// Replacing an owner's object, or putting the same bytes again, moves each
/// count once per distinct content: a retry never counts twice, and content
/// the owner keeps is not released.
#[test]
fn replace_and_repeat_never_double_count() {
    let (store, _dir) = open_store();
    let a = vec![1_u8; DEFAULT_CHUNK_SIZE];
    let b = vec![2_u8; DEFAULT_CHUNK_SIZE];
    let mut ab = a.clone();
    ab.extend_from_slice(&b);
    let mut aa = a.clone();
    aa.extend_from_slice(&a);

    put(&store, 1, &ab);
    put(&store, 1, &ab);
    assert_eq!(refs(&store, &scoped(&a)), 1);
    assert_eq!(refs(&store, &scoped(&b)), 1);

    // The same content twice in one object is one reference.
    put(&store, 1, &aa);
    assert_eq!(refs(&store, &scoped(&a)), 1);
    assert_eq!(refs(&store, &scoped(&b)), 0, "replaced content is released");
    assert_eq!(read_all(&store, &owner(1)), aa);

    store.delete(&owner(1)).expect("delete");
    assert_eq!(
        store.delete(&owner(1)).expect("delete again"),
        None,
        "a repeated delete releases nothing"
    );
    assert_eq!(refs(&store, &scoped(&a)), 0);
    assert_eq!(pins(&store, &scoped(&a)), 0, "no upload holds anything");
}

/// A creation never replaces an object the owner already holds, and leaves
/// nothing held by its own failed upload.
#[test]
fn a_creation_at_a_held_owner_is_refused() {
    let (store, _dir) = open_store();
    put(&store, 1, b"original");
    let refused = store.put(
        &owner(1),
        SharingScope::STORE,
        b"intruder",
        OwnerWrite::Create,
        None,
    );
    assert!(
        matches!(refused, Err(BlobError::OwnerExists)),
        "{refused:?}"
    );
    assert_eq!(read_all(&store, &owner(1)), b"original");
    assert_eq!(refs(&store, &scoped(b"intruder")), 0);
    assert_eq!(pins(&store, &scoped(b"intruder")), 0);
}

/// A representation is retired only once nothing references or holds its
/// content, and the bytes uploaded again afterwards get a new generation, so
/// a cleanup of the old one cannot reach the new copy.
#[test]
fn retirement_waits_for_the_last_owner_and_recreation_gets_a_new_generation() {
    let (store, _dir) = open_store();
    let data = b"retire me".to_vec();
    let content = scoped(&data);
    put(&store, 1, &data);
    let generation = eligible_generation(&store, &content);

    assert_eq!(
        store.retire(&content, generation).expect("retire"),
        Retirement::Held,
        "referenced content is not retired"
    );
    store.delete(&owner(1)).expect("delete");
    assert_eq!(
        store.retire(&content, generation).expect("retire"),
        Retirement::Retired
    );
    assert_eq!(
        store.retire(&content, generation).expect("again"),
        Retirement::NotEligible
    );
    let retired = entry(&store, &content);
    assert_eq!(retired.eligible, None);
    assert_eq!(retired.retired, vec![generation]);

    put(&store, 2, &data);
    let recreated = eligible_generation(&store, &content);
    assert_ne!(
        recreated, generation,
        "a retired generation is never reused"
    );
    // Cleanup of the retired generation, by its exact key, leaves the new
    // copy whole.
    store
        .engine()
        .delete(Partition::Blob, &payload_key(&content, generation))
        .expect("cleanup");
    assert_eq!(read_all(&store, &owner(2)), data);
}

/// An upload that is staged but not yet published holds its content: a
/// retirement between the two is refused, so the shared copy the upload is
/// about to reference is not retired under it.
#[test]
fn a_staged_upload_holds_its_shared_content_against_retirement() {
    let (store, _dir) = open_store();
    let data = b"shared and staged".to_vec();
    let content = scoped(&data);
    put(&store, 1, &data);
    let generation = eligible_generation(&store, &content);

    let staged = upload(&store, &data);
    assert!(
        !staged.staged().contents[0].writes,
        "the eligible copy is shared"
    );
    store.delete(&owner(1)).expect("the only owner goes");
    assert_eq!(
        store.retire(&content, generation).expect("retire"),
        Retirement::Held,
        "the staged upload holds the content"
    );

    publish(&store, 2, &staged).expect("publish");
    assert_eq!(pins(&store, &content), 0);
    assert_eq!(refs(&store, &content), 1);
    assert_eq!(read_all(&store, &owner(2)), data);
}

/// An upload fenced while staged cannot publish afterwards, and its pins and
/// payload are given up: the delayed uploader neither publishes nor leaks.
#[test]
fn a_fenced_upload_cannot_publish() {
    let (store, _dir) = open_store();
    let data = b"fenced upload".to_vec();
    let content = scoped(&data);
    let staged = upload(&store, &data);
    let staging_id = staged.staging_id();
    assert_eq!(pins(&store, &content), 1);

    assert!(store.abandon(staging_id).expect("fence"));
    assert!(
        !store.abandon(staging_id).expect("again"),
        "fencing repeats safely"
    );
    assert_eq!(pins(&store, &content), 0);
    assert_eq!(
        entry(&store, &content).retired,
        vec![staged.staged().contents[0].generation],
        "the payload it wrote waits for cleanup by its exact generation"
    );
    assert!(
        store
            .engine()
            .get(Partition::BlobRef, &staging_key(staging_id))
            .expect("get")
            .is_none()
    );

    let late = publish(&store, 1, &staged);
    assert!(matches!(late, Err(BlobError::Fenced)), "{late:?}");
    assert!(store.open(&owner(1)).expect("open").is_none());
    assert_eq!(refs(&store, &content), 0);
}

/// Two uploads of new content staged together each write their own copy;
/// the first to publish makes its copy eligible, the second references it
/// and leaves its own copy to cleanup by generation.
#[test]
fn concurrent_first_uploads_of_one_content_publish_one_copy() {
    let (store, _dir) = open_store();
    let data = b"raced first upload".to_vec();
    let content = scoped(&data);
    let first = upload(&store, &data);
    let second = upload(&store, &data);
    let (first_copy, second_copy) = (first.staged().contents[0], second.staged().contents[0]);
    assert!(first_copy.writes && second_copy.writes);
    assert_ne!(first_copy.generation, second_copy.generation);

    publish(&store, 1, &first).expect("publish first");
    publish(&store, 2, &second).expect("publish second");
    let published = entry(&store, &content);
    assert_eq!(
        published.eligible.map(|e| e.generation),
        Some(first_copy.generation)
    );
    assert_eq!(published.retired, vec![second_copy.generation]);
    assert_eq!(refs(&store, &content), 2);
    assert_eq!(pins(&store, &content), 0);
    assert_eq!(read_all(&store, &owner(2)), data);
}

/// A reader keeps every chunk it resolved readable, even when the owner is
/// deleted, the representation retired and its payload erased between its
/// opening and its reads.
#[test]
fn a_reader_keeps_its_payload_across_deletion_and_cleanup() {
    let (store, _dir) = open_store();
    let mut data = vec![7_u8; DEFAULT_CHUNK_SIZE];
    data.extend_from_slice(b"tail");
    put(&store, 1, &data);
    let reader = store.open(&owner(1)).expect("open").expect("present");
    let first = reader.chunk(0).expect("first chunk");

    store.delete(&owner(1)).expect("delete");
    for content in reader.record().blob.distinct() {
        let generation = eligible_generation(&store, &content);
        assert_eq!(
            store.retire(&content, generation).expect("retire"),
            Retirement::Retired
        );
        store
            .engine()
            .delete(Partition::Blob, &payload_key(&content, generation))
            .expect("cleanup");
    }

    let mut read = first;
    read.extend(reader.chunk(1).expect("second chunk"));
    assert_eq!(read, data);
    assert!(store.open(&owner(1)).expect("reopen").is_none());
}

/// A payload that is not its content is refused, never served.
#[test]
fn a_damaged_payload_is_refused() {
    let (store, _dir) = open_store();
    let data = b"integrity checked".to_vec();
    let content = scoped(&data);
    put(&store, 1, &data);
    let generation = eligible_generation(&store, &content);
    store
        .engine()
        .put(
            Partition::Blob,
            &payload_key(&content, generation),
            b"integrity checkeD",
        )
        .expect("damage");
    let reader = store.open(&owner(1)).expect("open").expect("present");
    let damaged = reader.chunk(0);
    assert!(
        matches!(damaged, Err(BlobError::IntegrityMismatch { .. })),
        "{damaged:?}"
    );
}

/// An object whose content has no eligible representation is refused at
/// opening, never read as empty or partial.
#[test]
fn an_object_without_an_eligible_representation_is_refused() {
    let (store, _dir) = open_store();
    let data = b"lost catalog".to_vec();
    put(&store, 1, &data);
    store
        .engine()
        .delete(Partition::BlobRef, &catalog_key(&scoped(&data)))
        .expect("drop the catalog entry");
    let refused = store.open(&owner(1));
    assert!(
        matches!(refused, Err(BlobError::Unavailable { .. })),
        "{:?}",
        refused.map(|r| r.map(|r| r.record().clone()))
    );
}

/// Writers putting, replacing and deleting owners of shared content race a
/// retirer that keeps trying to retire every eligible copy, and the retirer
/// erases the payload of every copy it retired, as a collector would. No
/// admitted reference may end up at an erased copy: every surviving owner
/// reads whole, every count equals the owners that reference the content,
/// and no upload is left holding anything.
#[test]
fn admission_and_retirement_race_without_losing_a_referenced_copy() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let (store, _dir) = open_store();
    let store = Arc::new(store);
    // More contents than owners: most of them have no owner at any moment
    // and are retired, then uploaded again, while others are being shared.
    let contents: Vec<Vec<u8>> = (0..8_u8).map(|i| vec![i; 4096]).collect();
    let stop = Arc::new(AtomicBool::new(false));

    let retirer = {
        let store = Arc::clone(&store);
        let contents = contents.clone();
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut retired = 0_u64;
            while !stop.load(Ordering::Relaxed) {
                for data in &contents {
                    let content = scoped(data);
                    let Some(bytes) = store
                        .engine()
                        .get(Partition::BlobRef, &catalog_key(&content))
                        .expect("get")
                    else {
                        continue;
                    };
                    let entry = CatalogEntry::from_msgpack(&bytes).expect("decode");
                    let Some(eligible) = entry.eligible else {
                        continue;
                    };
                    if store.retire(&content, eligible.generation).expect("retire")
                        == Retirement::Retired
                    {
                        store
                            .engine()
                            .delete(Partition::Blob, &payload_key(&content, eligible.generation))
                            .expect("erase");
                        retired += 1;
                    }
                }
            }
            retired
        })
    };

    let writers: Vec<_> = (0..4_u8)
        .map(|w| {
            let store = Arc::clone(&store);
            let contents = contents.clone();
            std::thread::spawn(move || {
                for round in 0..40_usize {
                    let data = &contents[(round * 3 + usize::from(w)) % contents.len()];
                    store
                        .put(
                            &owner(w),
                            SharingScope::STORE,
                            data,
                            OwnerWrite::Replace,
                            None,
                        )
                        .expect("put");
                    if round % 2 == 1 {
                        store.delete(&owner(w)).expect("delete");
                    }
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().expect("writer");
    }
    stop.store(true, Ordering::Relaxed);
    let retired = retirer.join().expect("retirer");

    let mut expected = vec![0_i64; contents.len()];
    for n in 0..4_u8 {
        let Some(reader) = store.open(&owner(n)).expect("open") else {
            continue;
        };
        let read: Vec<u8> = (0..reader.record().blob.chunk_count())
            .flat_map(|i| reader.chunk(i).expect("a referenced copy was erased"))
            .collect();
        let index = contents
            .iter()
            .position(|c| *c == read)
            .expect("an owner reads one of the contents whole");
        expected[index] += 1;
    }
    for (data, owners) in contents.iter().zip(expected) {
        assert_eq!(refs(&store, &scoped(data)), owners);
        assert_eq!(pins(&store, &scoped(data)), 0);
    }
    // The race is only a test if retirement actually happened in it.
    assert!(retired > 0, "the retirer never retired anything");
}

#[test]
fn an_empty_object_is_refused() {
    let (store, _dir) = open_store();
    assert!(matches!(
        store.put(
            &owner(1),
            SharingScope::STORE,
            b"",
            OwnerWrite::Replace,
            None
        ),
        Err(BlobError::Empty)
    ));
}

/// A blob whose put returned survives a power cut right after: an answered
/// upload is a promise the store keeps, and its reference never names
/// missing bytes. A deleted blob stays deleted across the cut.
#[test]
fn a_written_blob_survives_a_crash_right_after() {
    let dir = TempDir::new().expect("tempdir");
    let crash = Arc::new(CrashFs::new(StdFs));
    let faults = Arc::new(FaultInjector::new());
    let config = crash_config(&dir, &crash, &faults);
    let data = vec![0x5a; 300 << 10];
    {
        let store = store_over(&config);
        put(&store, 1, &data);
        put(&store, 2, b"short-lived");
        store.delete(&owner(2)).expect("delete");
        // Killed, not shut down: nothing the drop would flush reaches the
        // disk, while its files and its directory locks are let go as a
        // dead process lets them go.
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for op in [
            FaultOp::Open,
            FaultOp::Write,
            FaultOp::SyncAll,
            FaultOp::SyncData,
            FaultOp::Rename,
        ] {
            faults.arm(FaultRule::new(op, refuse));
        }
        drop(store);
        faults.clear();
    }
    crash.crash();

    let store = store_over(&config);
    // Compared, not printed: the object is hundreds of kilobytes.
    assert!(
        read_all(&store, &owner(1)) == data,
        "the blob was lost or changed across the power cut"
    );
    assert!(
        store.open(&owner(2)).expect("open").is_none(),
        "a deleted blob came back"
    );
    assert_eq!(refs(&store, &scoped(b"short-lived")), 0);
}
