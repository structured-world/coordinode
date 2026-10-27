use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::time::Duration;

use coordinode_core::graph::blob::{
    CANDIDATE_PREFIX, ContentId, DEFAULT_CHUNK_SIZE, OwnerRecord, ScopedContent, SharingScope,
    direct_owner_key, payload_key, pins_counter_key, refs_counter_key, retired_key,
};
use coordinode_core::txn::timestamp::TimestampOracle;
use tempfile::TempDir;

use super::*;
use crate::blob::{BlobStore, OwnerWrite, PutOutcome};
use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;

/// Microseconds since the epoch the simulated clock starts at.
const START: i64 = 1_700_000_000_000_000;

/// The simulated clock release, admission and retirement times are read
/// from. Each test runs in a process of its own, so one clock per test.
static CLOCK: AtomicI64 = AtomicI64::new(START);

fn clock() -> i64 {
    CLOCK.load(Ordering::SeqCst)
}

fn advance(by: Duration) {
    CLOCK.fetch_add(
        i64::try_from(by.as_micros()).expect("fits"),
        Ordering::SeqCst,
    );
}

const HOUR: Duration = Duration::from_secs(60 * 60);

fn policy() -> GcPolicy {
    GcPolicy {
        grace: HOUR,
        upload_expiry: Duration::from_secs(10 * 60),
        batch: 1024,
        reconcile_every: 24 * HOUR,
        reconcile_max_contents: 1_000_000,
        retry_interval: Duration::from_secs(1),
        // Compacted after every cleanup, so a pass shows its whole effect.
        compact_after_bytes: 0,
        compact_after: HOUR,
    }
}

fn config(dir: &TempDir) -> StorageConfig {
    StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )])
}

/// A collector over the store in `dir`, with no retained history, so a
/// compaction may collect what no reader holds.
fn collector_over(dir: &TempDir, policy: GcPolicy) -> Arc<BlobCollector> {
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config(dir), Arc::clone(&oracle)).expect("open"));
    engine.set_retention_window(Duration::ZERO);
    let store = BlobStore::direct(engine, oracle).with_clock(clock);
    BlobCollector::new(store, policy, Arc::new(|| true))
}

fn open() -> (Arc<BlobCollector>, TempDir) {
    let dir = TempDir::new().expect("tempdir");
    (collector_over(&dir, policy()), dir)
}

fn owner(n: u8) -> Vec<u8> {
    direct_owner_key(&[n; 16])
}

fn put(gc: &BlobCollector, n: u8, data: &[u8]) -> PutOutcome {
    gc.store()
        .put(
            &owner(n),
            SharingScope::STORE,
            data,
            OwnerWrite::Replace,
            None,
        )
        .expect("put")
}

fn delete(gc: &BlobCollector, n: u8) {
    gc.store().delete(&owner(n)).expect("delete");
}

/// `chunks` full chunks of bytes no other seed produces, so each chunk is a
/// content of its own, and that do not compress, so the space they take on
/// disk is what they are.
fn data(seed: u8, chunks: usize) -> Vec<u8> {
    // xorshift64: cheap, and as incompressible as a codec can tell.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ u64::from(seed).wrapping_mul(0x2545_F491_4F6C_DD1D);
    let mut out = Vec::with_capacity(chunks * DEFAULT_CHUNK_SIZE);
    while out.len() < chunks * DEFAULT_CHUNK_SIZE {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(chunks * DEFAULT_CHUNK_SIZE);
    out
}

fn read_all(gc: &BlobCollector, n: u8) -> Vec<u8> {
    let reader = gc.store().open(&owner(n)).expect("open").expect("present");
    (0..reader.record().blob.chunk_count())
        .flat_map(|i| reader.chunk(i).expect("chunk"))
        .collect()
}

fn generations(gc: &BlobCollector, record: &OwnerRecord) -> Vec<(ScopedContent, u64)> {
    record
        .blob
        .distinct()
        .into_iter()
        .map(|content| {
            let entry = gc
                .catalog_entry(&content)
                .expect("catalog")
                .expect("eligible");
            (content, entry.eligible.generation)
        })
        .collect()
}

fn has_payload(gc: &BlobCollector, content: &ScopedContent, generation: u64) -> bool {
    gc.store()
        .engine()
        .get(Partition::Blob, &payload_key(content, generation))
        .expect("get")
        .is_some()
}

fn count(gc: &BlobCollector, key: &[u8]) -> i64 {
    gc.count(key).expect("count")
}

fn candidates(gc: &BlobCollector) -> usize {
    gc.store()
        .engine()
        .prefix_scan(Partition::BlobRef, CANDIDATE_PREFIX)
        .expect("scan")
        .count()
}

fn blob_disk_bytes(gc: &BlobCollector) -> u64 {
    use lsm_tree::AbstractTree as _;
    gc.store()
        .engine()
        .tree(Partition::Blob)
        .expect("tree")
        .disk_space()
}

/// The whole path on a real filesystem: a deleted blob's content is kept
/// through its grace, then retired, its payload deleted by exact generation,
/// and the Blob partition compacted until the space is actually back, in
/// that one pass and with nothing written after it, with every stage
/// counted. A pass inside the grace, manual or not, takes nothing.
#[test]
fn a_released_blob_is_kept_through_its_grace_then_its_space_returns() {
    let (gc, _dir) = open();
    let bytes = data(1, 8);
    let record = put(&gc, 1, &bytes).record;
    let copies = generations(&gc, &record);
    gc.run(true).expect("enrolment pass");
    let full = blob_disk_bytes(&gc);
    assert!(full >= bytes.len() as u64, "{full} bytes hold the payload");

    delete(&gc, 1);
    let early = gc.run(true).expect("pass in grace");
    assert_eq!(early.retired, 0);
    assert_eq!(early.cleaned, CleanedUpReport::default());
    let status = gc.status().expect("status");
    assert_eq!(status.candidates, 8);
    assert_eq!(status.candidates_due, 0);
    assert_eq!(status.next_due_micros, Some(START + 3_600_000_000));
    for (content, generation) in &copies {
        assert!(has_payload(&gc, content, *generation), "kept in its grace");
    }

    advance(HOUR + Duration::from_secs(1));
    let pass = gc.run(false).expect("pass after grace");
    assert_eq!(pass.candidates_examined, 8);
    assert_eq!(pass.retired, 8);
    assert_eq!(
        pass.cleaned,
        CleanedUpReport {
            representations: 8,
            chunks: 8,
            bytes: bytes.len() as u64,
        }
    );
    assert!(pass.compaction.is_some(), "the deletion was compacted");
    for (content, generation) in &copies {
        assert!(!has_payload(&gc, content, *generation));
        assert!(gc.catalog_entry(content).expect("catalog").is_none());
    }
    let status = gc.status().expect("status");
    assert_eq!(status.candidates, 0);
    assert_eq!(status.retired_pending, 0);
    assert!(!status.compaction_pending);
    assert_eq!(status.reclaimed_chunks, 8);
    assert_eq!(status.reclaimed_bytes, bytes.len() as u64);
    assert_eq!(status.stale_blob_bytes, 0, "no emptied file is left behind");
    assert!(status.reclaimed_disk_bytes > 0);
    assert!(
        status.blob_disk_bytes < full / 2,
        "{} of {full} bytes still on disk",
        status.blob_disk_bytes
    );
}

/// Compactions are batched: deleted payload short of a value file's worth
/// waits for more, and its space returns once enough is deleted or the
/// oldest deletion has waited out the delay, whichever comes first; the
/// count of what waits goes back to zero with the compaction.
#[test]
fn compactions_wait_for_enough_deleted_payload_or_the_delay() {
    let dir = TempDir::new().expect("tempdir");
    let one = data(11, 2).len() as u64;
    let gc = collector_over(
        &dir,
        GcPolicy {
            compact_after_bytes: 2 * one,
            ..policy()
        },
    );
    gc.run(true).expect("enrolment");
    put(&gc, 1, &data(11, 2));
    put(&gc, 2, &data(12, 2));
    put(&gc, 3, &data(13, 2));

    // One blob's worth deleted: short of the bytes, inside the delay.
    delete(&gc, 1);
    advance(2 * HOUR);
    let pass = gc.run(false).expect("first deletion");
    assert_eq!(pass.cleaned.bytes, one);
    assert_eq!(pass.compaction, None, "one blob's worth waits");
    let status = gc.status().expect("status");
    assert!(status.compaction_pending);
    assert_eq!(status.uncompacted_bytes, one);
    assert_eq!(status.oldest_uncompacted_micros, Some(clock()));

    // A second blob's worth reaches the bytes: compacted at once.
    delete(&gc, 2);
    advance(2 * HOUR);
    let pass = gc.run(false).expect("second deletion");
    assert!(pass.compaction.is_some(), "two blobs' worth is due");
    let status = gc.status().expect("status");
    assert!(!status.compaction_pending);
    assert_eq!(status.uncompacted_bytes, 0);
    assert_eq!(status.oldest_uncompacted_micros, None);

    // A third, alone, waits out the delay instead.
    delete(&gc, 3);
    advance(2 * HOUR);
    assert_eq!(gc.run(false).expect("third deletion").compaction, None);
    advance(HOUR);
    assert!(
        gc.run(false).expect("after the delay").compaction.is_some(),
        "the oldest deletion waited out the delay"
    );
    assert_eq!(gc.status().expect("status").uncompacted_bytes, 0);
}

/// Content another owner still references survives the deletion of one
/// owner: the candidate is dropped, nothing is retired, and the other owner
/// reads whole.
#[test]
fn shared_content_survives_one_owner_deletion() {
    let (gc, _dir) = open();
    let bytes = data(2, 2);
    put(&gc, 1, &bytes);
    put(&gc, 2, &bytes);
    gc.run(true).expect("enrolment");
    delete(&gc, 1);
    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.candidates_examined, 2);
    assert_eq!(pass.retired, 0);
    assert_eq!(
        candidates(&gc),
        0,
        "a candidate an owner outlived is dropped"
    );
    assert_eq!(read_all(&gc, 2), bytes);
}

/// Replacing what an owner holds releases what it no longer references and
/// keeps what it still does: only the dropped chunk is collected.
#[test]
fn a_replacement_collects_only_what_it_let_go() {
    let (gc, _dir) = open();
    let old = data(3, 2);
    let kept = old[..DEFAULT_CHUNK_SIZE].to_vec();
    let record = put(&gc, 1, &old).record;
    let dropped = generations(&gc, &record)
        .into_iter()
        .find(|(content, _)| content.content != ContentId::of(&kept))
        .expect("the second chunk");
    let mut new = kept.clone();
    new.extend_from_slice(b"a new tail");
    assert_eq!(put(&gc, 1, &new).released, 1);
    gc.run(true).expect("enrolment");
    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.retired, 1);
    assert!(!has_payload(&gc, &dropped.0, dropped.1));
    assert_eq!(read_all(&gc, 1), new);
}

/// An upload that never published, as after a crash between its admission
/// and its publication, is fenced once it outlives the upload expiry and its
/// payload is collected; before that it is left alone, and once fenced the
/// late publication is refused.
#[test]
fn an_interrupted_upload_is_fenced_after_its_expiry_and_collected() {
    let (gc, _dir) = open();
    let bytes = data(4, 2);
    let staged = gc
        .store()
        .upload(SharingScope::STORE, &bytes)
        .expect("upload");
    let copies: Vec<_> = staged
        .staged()
        .contents
        .iter()
        .map(|c| (c.content, c.generation))
        .collect();
    gc.run(true).expect("pass before the expiry");
    for (content, generation) in &copies {
        assert!(has_payload(&gc, content, *generation));
        assert_eq!(count(&gc, &pins_counter_key(content)), 1);
    }
    assert_eq!(gc.status().expect("status").staged_uploads, 1);

    advance(Duration::from_secs(11 * 60));
    let pass = gc.run(false).expect("pass after the expiry");
    assert_eq!(pass.expired_uploads, 1);
    assert_eq!(pass.cleaned.chunks, 2);
    for (content, generation) in &copies {
        assert!(!has_payload(&gc, content, *generation));
        assert_eq!(count(&gc, &pins_counter_key(content)), 0);
    }
    let late = gc
        .store()
        .publish(&owner(1), OwnerWrite::Replace, &staged, None);
    assert!(matches!(late, Err(BlobError::Fenced)), "{late:?}");
}

/// An upload holding a released content keeps it from collection: the
/// content is reported as protected and its candidate kept, and it is
/// collected only once the upload lets go.
#[test]
fn an_upload_in_progress_protects_the_content_it_holds() {
    let (gc, _dir) = open();
    let bytes = b"held while staged".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    // The upload stays live through the grace: its expiry is longer.
    gc.set_policy(GcPolicy {
        upload_expiry: 24 * HOUR,
        ..policy()
    });
    let staged = gc
        .store()
        .upload(SharingScope::STORE, &bytes)
        .expect("a second upload shares the copy");
    gc.run(true).expect("enrolment");
    delete(&gc, 1);
    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.protected, 1);
    assert_eq!(pass.retired, 0);
    assert_eq!(candidates(&gc), 1, "the candidate waits for the upload");
    assert!(has_payload(&gc, &content, generation));

    assert!(gc.store().abandon(staged.staging_id()).expect("abandon"));
    let pass = gc.run(false).expect("pass after the upload");
    assert_eq!(pass.retired, 1);
    assert!(!has_payload(&gc, &content, generation));
}

/// A reader opened before collection reads the whole object while the owner
/// is deleted, the copies retired, their payload deleted and the compaction
/// attempted; the compaction waits for the reader and runs once it is gone.
#[test]
fn a_reader_holds_off_the_compaction_and_reads_whole() {
    let (gc, _dir) = open();
    let bytes = data(5, 3);
    put(&gc, 1, &bytes);
    gc.run(true).expect("enrolment");
    let reader = gc.store().open(&owner(1)).expect("open").expect("present");
    let first = reader.chunk(0).expect("first chunk before collection");

    delete(&gc, 1);
    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass under the reader");
    assert_eq!(pass.retired, 3);
    assert_eq!(pass.cleaned.chunks, 3);
    assert_eq!(
        pass.compaction, None,
        "the reader holds the deleted payload"
    );
    assert!(gc.status().expect("status").compaction_pending);

    let mut read = first;
    for i in 1..reader.record().blob.chunk_count() {
        read.extend(reader.chunk(i).expect("a chunk the reader resolved"));
    }
    assert!(read == bytes, "the reader read something else");
    drop(reader);

    let pass = gc.run(false).expect("pass after the reader");
    assert!(
        pass.compaction.is_some(),
        "the compaction runs once nothing holds it"
    );
    assert!(!gc.status().expect("status").compaction_pending);
}

/// A stale cleanup, one of a copy retired before the same bytes were stored
/// again, deletes only that copy: the recreated blob reads whole.
#[test]
fn a_cleanup_after_same_hash_recreation_leaves_the_new_copy() {
    let (gc, _dir) = open();
    gc.run(true).expect("enrolment");
    let bytes = b"recreated after retirement".repeat(200);
    let record = put(&gc, 1, &bytes).record;
    let (content, old) = generations(&gc, &record)[0];
    delete(&gc, 1);
    assert_eq!(
        gc.store().retire(&content, old).expect("retire"),
        Retirement::Retired
    );
    put(&gc, 2, &bytes);
    let new = gc
        .catalog_entry(&content)
        .expect("catalog")
        .expect("eligible")
        .eligible
        .generation;
    assert_ne!(new, old);

    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.cleaned.chunks, 1, "only the retired copy");
    assert_eq!(pass.retired, 0, "the recreated copy is referenced");
    assert!(!has_payload(&gc, &content, old));
    assert!(has_payload(&gc, &content, new));
    assert_eq!(read_all(&gc, 2), bytes);
}

/// Reconciliation corrects counts that drifted either way and gives a lost
/// candidate back: a leaked reference no longer keeps orphaned content, and
/// a lost one no longer exposes a live owner's content to collection.
#[test]
fn reconciliation_corrects_drifted_counts_and_lost_candidates() {
    let (gc, _dir) = open();
    let live = b"live owner".repeat(300);
    let orphan = b"orphaned".repeat(300);
    let live_record = put(&gc, 1, &live).record;
    let orphan_record = put(&gc, 2, &orphan).record;
    let (live_content, _) = generations(&gc, &live_record)[0];
    let (orphan_content, orphan_generation) = generations(&gc, &orphan_record)[0];
    delete(&gc, 2);
    let engine = gc.store().engine();
    // A lost reference for the live owner, a leaked one for the orphan, and
    // the orphan's candidate gone.
    engine
        .merge(
            Partition::Counter,
            &refs_counter_key(&live_content),
            &(-1_i64).to_le_bytes(),
        )
        .expect("lose a reference");
    engine
        .merge(
            Partition::Counter,
            &refs_counter_key(&orphan_content),
            &1_i64.to_le_bytes(),
        )
        .expect("leak a reference");
    for guard in engine
        .prefix_scan(Partition::BlobRef, CANDIDATE_PREFIX)
        .expect("scan")
    {
        let key = guard.key().expect("key");
        engine
            .delete(Partition::BlobRef, &key)
            .expect("lose the candidate");
    }

    let pass = gc.run(true).expect("reconciling pass");
    let found = pass.reconciliation.expect("reconciled");
    assert_eq!(found.corrected_refs, 2);
    assert_eq!(found.recorded_candidates, 1);
    assert!(found.damaged.is_empty());
    assert_eq!(count(&gc, &refs_counter_key(&live_content)), 1);
    assert_eq!(count(&gc, &refs_counter_key(&orphan_content)), 0);

    advance(2 * HOUR);
    let pass = gc.run(false).expect("collecting pass");
    assert_eq!(pass.retired, 1);
    assert!(!has_payload(&gc, &orphan_content, orphan_generation));
    assert_eq!(read_all(&gc, 1), live);
}

/// Payload no catalog entry, upload or retirement names is retired by the
/// reconciliation and collected, while every named copy stays.
#[test]
fn reconciliation_collects_payload_nothing_names() {
    let (gc, _dir) = open();
    let bytes = b"named copy".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    let stray = generation + 1;
    gc.store()
        .engine()
        .put(Partition::Blob, &payload_key(&content, stray), &bytes)
        .expect("stray payload");
    let pass = gc.run(true).expect("pass");
    assert_eq!(pass.reconciliation.expect("reconciled").retired_strays, 1);
    assert_eq!(pass.cleaned.chunks, 1);
    assert!(!has_payload(&gc, &content, stray));
    assert!(has_payload(&gc, &content, generation));
    assert_eq!(read_all(&gc, 1), bytes);
}

/// An owner record that does not decode makes ownership unknown: retirement
/// and cleanup stop, the blocker is reported, nothing is corrected, and a
/// later reconciliation that finds no damage lets collection go on.
#[test]
fn a_damaged_owner_record_stops_collection_until_it_is_gone() {
    let (gc, _dir) = open();
    let bytes = b"waits for verified accounting".repeat(200);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    gc.run(true).expect("enrolment");
    delete(&gc, 1);
    let damaged = direct_owner_key(&[9; 16]);
    gc.store()
        .engine()
        .put(Partition::BlobRef, &damaged, b"not an owner record")
        .expect("damage");

    advance(2 * HOUR);
    let pass = gc.run(true).expect("pass");
    assert_eq!(pass.reconciliation.expect("reconciled").damaged.len(), 1);
    assert_eq!(pass.retired, 0);
    assert!(has_payload(&gc, &content, generation));
    let status = gc.status().expect("status");
    let blocked = status.blocked.expect("blocked");
    assert!(blocked.contains("ownership is unknown"), "{blocked}");
    assert_eq!(status.candidates, 1);

    gc.store()
        .engine()
        .delete(Partition::BlobRef, &damaged)
        .expect("repair");
    let pass = gc.run(true).expect("pass after repair");
    assert_eq!(pass.retired, 1);
    assert!(!has_payload(&gc, &content, generation));
    assert_eq!(gc.status().expect("status").blocked, None);
}

/// A negative count is broken accounting, not an absent owner: the content
/// is reported and kept, not retired.
#[test]
fn a_negative_count_is_reported_and_never_collected() {
    let (gc, _dir) = open();
    let bytes = b"miscounted".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    gc.run(true).expect("enrolment");
    delete(&gc, 1);
    gc.store()
        .engine()
        .merge(
            Partition::Counter,
            &refs_counter_key(&content),
            &(-1_i64).to_le_bytes(),
        )
        .expect("break the count");
    advance(2 * HOUR);
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.retired, 0);
    assert!(
        pass.blockers.iter().any(|b| b.contains("negative count")),
        "{:?}",
        pass.blockers
    );
    assert!(has_payload(&gc, &content, generation));
}

/// A store that never held a blob is left alone: a pass, even one asked to
/// reconcile, writes nothing and reports nothing in the way, so a database
/// that stores no blobs pays no commit for their collection.
#[test]
fn a_store_without_blobs_is_left_untouched() {
    let (gc, _dir) = open();
    let engine = gc.store().engine();
    let before = engine.snapshot();
    let pass = gc.run(true).expect("pass");
    assert_eq!(pass.reconciliation, None);
    assert!(pass.blockers.is_empty(), "{:?}", pass.blockers);
    assert_eq!(engine.snapshot(), before, "the pass committed something");
    assert!(
        engine
            .get(Partition::BlobRef, COLLECTOR_STATE_KEY)
            .expect("get")
            .is_none()
    );
}

/// Nothing is retired or deleted before the first reconciliation: data that
/// was there before the collector ever ran is enrolled first.
#[test]
fn nothing_is_collected_before_the_first_reconciliation() {
    let (gc, _dir) = open();
    let bytes = b"before enrolment".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    delete(&gc, 1);
    advance(2 * HOUR);
    gc.set_policy(GcPolicy {
        reconcile_every: Duration::MAX,
        ..policy()
    });
    // The first pass enrols the store, whatever the schedule says.
    let pass = gc.run(false).expect("first pass");
    assert!(pass.reconciliation.is_some());
    assert_eq!(pass.retired, 1, "collected once enrolled");
    assert!(!has_payload(&gc, &content, generation));
}

/// The grace in force is never shorter than the MVCC retention window: a
/// shorter configured grace still keeps a release for the whole window.
#[test]
fn the_retention_window_raises_the_grace() {
    let (gc, _dir) = open();
    gc.store().engine().set_retention_window(2 * HOUR);
    let bytes = b"history keeps it".repeat(300);
    put(&gc, 1, &bytes);
    gc.run(true).expect("enrolment");
    delete(&gc, 1);
    advance(HOUR + Duration::from_secs(60));
    let pass = gc.run(false).expect("pass");
    assert_eq!(pass.retired, 0, "the window outlasts the configured grace");
    assert_eq!(gc.status().expect("status").grace_secs, 2 * 3600);
}

/// Every stage resumes after a restart: a retirement whose cleanup had not
/// run, and a cleanup whose compaction had not run, are both finished by the
/// first pass of the reopened store, with the counts kept.
#[test]
fn collection_resumes_across_a_restart() {
    let dir = TempDir::new().expect("tempdir");
    let bytes = data(6, 4);
    let copies;
    {
        let gc = collector_over(&dir, policy());
        let record = put(&gc, 1, &bytes).record;
        copies = generations(&gc, &record);
        delete(&gc, 1);
        // Retired, not cleaned up.
        for (content, generation) in &copies[..2] {
            gc.store().retire(content, *generation).expect("retire");
        }
        // Cleaned up, not compacted.
        for (content, generation) in &copies[2..] {
            gc.store().retire(content, *generation).expect("retire");
            gc.store()
                .clean_up(&[(*content, *generation)])
                .expect("clean up");
        }
        gc.store().engine().persist().expect("persist");
    }
    let gc = collector_over(&dir, policy());
    let status = gc.status().expect("status after reopen");
    assert_eq!(status.retired_pending, 2);
    assert!(status.compaction_pending);
    assert_eq!(status.reclaimed_chunks, 2);

    let pass = gc.run(false).expect("first pass after reopen");
    assert_eq!(pass.cleaned.chunks, 2);
    assert!(pass.compaction.is_some());
    for (content, generation) in &copies {
        assert!(!has_payload(&gc, content, *generation));
    }
    let status = gc.status().expect("status");
    assert_eq!(status.reclaimed_chunks, 4);
    assert_eq!(status.reclaimed_bytes, bytes.len() as u64);
    assert!(!status.compaction_pending);
}

/// A power cut in the middle of collection loses nothing and resurrects
/// nothing: a collector killed right after a pass that retired and cleaned
/// up, before anything it did not sync reached the disk, reopens with the
/// deleted blob still gone, the live one whole, and its remaining work done
/// by the next pass without counting anything twice.
#[test]
fn a_crash_after_collection_loses_nothing_and_resurrects_nothing() {
    use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultOp, FaultRule, StdFs};

    let dir = TempDir::new().expect("tempdir");
    let crash = Arc::new(CrashFs::new(StdFs));
    let faults = Arc::new(FaultInjector::new());
    let crash_config = config(&dir).with_fs(Arc::new(FaultFs::with_injector(
        CrashFs::clone(&crash),
        Arc::clone(&faults),
    )));
    let open_over = |config: &StorageConfig| {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            Arc::new(StorageEngine::open_embedded(config, Arc::clone(&oracle)).expect("open"));
        engine.set_retention_window(Duration::ZERO);
        BlobCollector::new(
            BlobStore::direct(engine, oracle).with_clock(clock),
            policy(),
            Arc::new(|| true),
        )
    };
    let live = data(8, 2);
    let gone = data(9, 3);
    let gone_copies;
    let reclaimed_before;
    {
        let gc = open_over(&crash_config);
        put(&gc, 1, &live);
        let record = put(&gc, 2, &gone).record;
        gone_copies = generations(&gc, &record);
        gc.run(true).expect("enrolment");
        delete(&gc, 2);
        advance(2 * HOUR);
        let pass = gc.run(false).expect("collecting pass");
        assert_eq!(pass.retired, 3);
        reclaimed_before = gc.status().expect("status").reclaimed_bytes;
        assert_eq!(reclaimed_before, gone.len() as u64);
        // Killed, not shut down: what the drop would flush never reaches
        // the disk, and its files and locks go as a dead process's go.
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
        drop(gc);
        faults.clear();
    }
    crash.crash();

    let gc = open_over(&crash_config);
    assert!(
        gc.store().open(&owner(2)).expect("open").is_none(),
        "a deleted blob came back"
    );
    assert!(
        read_all(&gc, 1) == live,
        "the live blob was lost or changed"
    );
    gc.run(true).expect("pass after the cut");
    for (content, generation) in &gone_copies {
        assert!(!has_payload(&gc, content, *generation));
        assert!(gc.catalog_entry(content).expect("catalog").is_none());
    }
    let status = gc.status().expect("status");
    assert_eq!(status.reclaimed_bytes, reclaimed_before, "counted once");
    assert_eq!((status.candidates, status.retired_pending), (0, 0));
    assert!(!status.compaction_pending);
}

/// One cut point of the collection sweep: a live blob and a deleted one past
/// its grace, a pass whose `k`-th operation of kind `op` fails, a power cut,
/// and a store that must reopen with the live blob whole, the deleted one
/// gone, none of its payload unaccounted for, and its collection finished by
/// the next pass with every byte counted once. Returns whether the pass
/// reached the cut.
fn collection_cut_at(op: lsm_tree::fs::FaultOp, k: u64) -> bool {
    use lsm_tree::fs::{CrashFs, Fault, FaultFs, FaultInjector, FaultRule, StdFs};

    let dir = TempDir::new().expect("tempdir");
    let crash = Arc::new(CrashFs::new(StdFs));
    let faults = Arc::new(FaultInjector::new());
    let crash_config = config(&dir).with_fs(Arc::new(FaultFs::with_injector(
        CrashFs::clone(&crash),
        Arc::clone(&faults),
    )));
    let open_over = |config: &StorageConfig| {
        let oracle = Arc::new(TimestampOracle::new());
        let engine =
            Arc::new(StorageEngine::open_embedded(config, Arc::clone(&oracle)).expect("open"));
        engine.set_retention_window(Duration::ZERO);
        BlobCollector::new(
            BlobStore::direct(engine, oracle).with_clock(clock),
            policy(),
            Arc::new(|| true),
        )
    };
    let live = data(8, 2);
    let gone = data(9, 3);
    let gone_copies;
    let reached;
    {
        let gc = open_over(&crash_config);
        put(&gc, 1, &live);
        let record = put(&gc, 2, &gone).record;
        gone_copies = generations(&gc, &record);
        gc.run(true).expect("enrolment");
        delete(&gc, 2);
        advance(2 * HOUR);
        faults.arm(FaultRule::new(op, Fault::Error(lsm_tree::io::ErrorKind::Other)).skip(k));
        reached = gc.run(false).is_err();
        faults.clear();
        // Whatever the pass did not sync is lost with the power.
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for kind in [
            lsm_tree::fs::FaultOp::Open,
            lsm_tree::fs::FaultOp::Write,
            lsm_tree::fs::FaultOp::SyncAll,
            lsm_tree::fs::FaultOp::SyncData,
            lsm_tree::fs::FaultOp::Rename,
        ] {
            faults.arm(FaultRule::new(kind, refuse));
        }
        drop(gc);
        faults.clear();
    }
    crash.crash();

    let label = format!("{op:?} cut at {k}");
    let gc = open_over(&crash_config);
    assert!(
        read_all(&gc, 1) == live,
        "{label}: the live blob was lost or changed"
    );
    assert!(
        gc.store().open(&owner(2)).expect("open").is_none(),
        "{label}: the deleted blob came back"
    );
    // Every payload of the deleted blob still there is accounted for: its
    // copy is either still eligible, awaiting retirement, or retired and
    // awaiting cleanup. None is left that nothing would ever delete.
    for (content, generation) in &gone_copies {
        if !has_payload(&gc, content, *generation) {
            continue;
        }
        let eligible = gc
            .catalog_entry(content)
            .expect("catalog")
            .is_some_and(|e| e.eligible.generation == *generation);
        let retired = gc
            .store()
            .engine()
            .get(Partition::BlobRef, &retired_key(content, *generation))
            .expect("get")
            .is_some();
        assert!(eligible || retired, "{label}: payload left unaccounted for");
    }
    gc.run(false).expect("pass after the cut");
    for (content, generation) in &gone_copies {
        assert!(
            !has_payload(&gc, content, *generation),
            "{label}: the next pass left payload behind"
        );
    }
    let status = gc.status().expect("status");
    assert_eq!(
        status.reclaimed_bytes,
        gone.len() as u64,
        "{label}: each deleted byte is counted once"
    );
    assert_eq!(
        (status.candidates, status.retired_pending),
        (0, 0),
        "{label}"
    );
    assert!(!status.compaction_pending, "{label}");
    reached
}

/// A power cut at every operation of `op` a collection pass performs. The
/// cut points are independent stores, run on parallel threads in windows
/// until a whole window completes the pass untouched. Returns how many cut
/// points the pass reached.
fn collection_cut_at_every(op: lsm_tree::fs::FaultOp) -> u64 {
    let window = std::thread::available_parallelism().map_or(4, |n| n.get() as u64);
    let mut reached = 0;
    let mut start = 0;
    loop {
        let hits: Vec<bool> = std::thread::scope(|s| {
            let handles: Vec<_> = (start..start + window)
                .map(|k| s.spawn(move || collection_cut_at(op, k)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("cut point thread"))
                .collect()
        });
        let window_hits = hits.iter().filter(|&&hit| hit).count() as u64;
        reached += window_hits;
        if window_hits < window {
            return reached;
        }
        start += window;
    }
}

/// A collection pass cut by a power loss at any of its syncs or writes, so
/// before or after a retirement, a payload deletion or the compaction that
/// returns the space, leaves a store that loses nothing live, resurrects
/// nothing deleted and finishes the collection in its next pass.
#[test]
fn a_power_cut_at_any_sync_or_write_of_collection_loses_nothing() {
    let syncs = collection_cut_at_every(lsm_tree::fs::FaultOp::SyncAll)
        + collection_cut_at_every(lsm_tree::fs::FaultOp::SyncData);
    let writes = collection_cut_at_every(lsm_tree::fs::FaultOp::Write);
    assert!(syncs > 0, "the pass reached no sync");
    assert!(writes > 0, "the pass reached no write");
}

/// A member of a group that holds other members does not collect: the pass
/// does nothing and says why.
#[test]
fn a_member_of_a_group_does_not_collect() {
    let dir = TempDir::new().expect("tempdir");
    let oracle = Arc::new(TimestampOracle::new());
    let engine =
        Arc::new(StorageEngine::open_embedded(&config(&dir), Arc::clone(&oracle)).expect("open"));
    let store = BlobStore::direct(engine, oracle).with_clock(clock);
    let gc = BlobCollector::new(store, policy(), Arc::new(|| false));
    let bytes = b"replicated elsewhere".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    delete(&gc, 1);
    advance(2 * HOUR);
    let pass = gc.run(true).expect("pass");
    assert!(pass.reconciliation.is_none());
    assert!(
        pass.blockers.iter().any(|b| b.contains("group")),
        "{:?}",
        pass.blockers
    );
    assert!(has_payload(&gc, &content, generation));
    assert!(!gc.status().expect("status").active);
}

/// The background worker collects on its own: woken by the deletion, it
/// finds the candidate in its grace, sleeps until the grace ends, and
/// returns the space without anyone calling it.
#[test]
fn the_worker_collects_without_being_asked() {
    let dir = TempDir::new().expect("tempdir");
    let gc = collector_over(
        &dir,
        GcPolicy {
            grace: Duration::from_millis(200),
            ..policy()
        },
    );
    let bytes = data(7, 2);
    let record = put(&gc, 1, &bytes).record;
    let copies = generations(&gc, &record);
    let _worker = gc.spawn();
    delete(&gc, 1);
    // The simulated clock moves past the grace while the worker sleeps
    // towards it.
    advance(Duration::from_secs(1));
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    while copies.iter().any(|(c, g)| has_payload(&gc, c, *g)) {
        assert!(
            std::time::Instant::now() < deadline,
            "the worker did not collect: {:?}",
            gc.status().expect("status")
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// Repeated cycles of storing and deleting fresh content leave bounded
/// storage behind: every cycle's payload is collected and its space
/// returned, with no candidate, retirement or pending compaction left, and
/// the partition no larger than a few cycles' payload.
#[test]
fn repeated_upload_and_delete_cycles_keep_storage_bounded() {
    let (gc, _dir) = open();
    gc.run(true).expect("enrolment");
    let cycle = data(0, 4).len() as u64;
    let mut peak = 0;
    for round in 0..12_u8 {
        put(&gc, 1, &data(round + 10, 4));
        peak = peak.max(blob_disk_bytes(&gc));
        delete(&gc, 1);
        advance(2 * HOUR);
        gc.run(false).expect("pass");
    }
    let status = gc.status().expect("status");
    assert_eq!(status.candidates, 0);
    assert_eq!(status.retired_pending, 0);
    assert!(!status.compaction_pending);
    assert_eq!(status.reclaimed_bytes, 12 * cycle);
    assert!(
        status.blob_disk_bytes < 2 * cycle,
        "{} bytes left after 12 cycles of {cycle}, peak {peak}",
        status.blob_disk_bytes
    );
}

/// The collector and writers race over shared content: writers keep putting,
/// replacing and deleting owners while passes with no grace collect what is
/// released. No owner standing at the end loses a byte, and every count
/// equals the owners that reference its content.
#[test]
fn collection_racing_writers_never_loses_a_referenced_copy() {
    let dir = TempDir::new().expect("tempdir");
    let gc = collector_over(
        &dir,
        GcPolicy {
            grace: Duration::ZERO,
            ..policy()
        },
    );
    gc.run(true).expect("enrolment");
    let contents: Vec<Vec<u8>> = (0..6_u8).map(|i| vec![i; 4096]).collect();
    let stop = Arc::new(AtomicBool::new(false));
    let collector = {
        let gc = Arc::clone(&gc);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut retired = 0;
            while !stop.load(Ordering::Relaxed) {
                retired += gc.run(false).expect("pass").retired;
            }
            retired
        })
    };
    let writers: Vec<_> = (0..4_u8)
        .map(|w| {
            let gc = Arc::clone(&gc);
            let contents = contents.clone();
            std::thread::spawn(move || {
                for round in 0..30_usize {
                    let data = &contents[(round * 5 + usize::from(w)) % contents.len()];
                    put(&gc, w, data);
                    if round % 2 == 1 {
                        delete(&gc, w);
                    }
                }
            })
        })
        .collect();
    for writer in writers {
        writer.join().expect("writer");
    }
    stop.store(true, Ordering::Relaxed);
    let retired = collector.join().expect("collector");

    let mut owners = vec![0_i64; contents.len()];
    for n in 0..4_u8 {
        if gc.store().open(&owner(n)).expect("open").is_none() {
            continue;
        }
        let read = read_all(&gc, n);
        let index = contents
            .iter()
            .position(|c| *c == read)
            .expect("an owner reads one of the contents whole");
        owners[index] += 1;
    }
    for (bytes, expected) in contents.iter().zip(owners) {
        let content = ScopedContent::new(SharingScope::STORE, ContentId::of(bytes));
        assert_eq!(count(&gc, &refs_counter_key(&content)), expected);
        assert_eq!(count(&gc, &pins_counter_key(&content)), 0);
    }
    assert!(retired > 0, "the collector never collected during the race");
    // A final reconciliation finds nothing to correct.
    let found = gc
        .run(true)
        .expect("final pass")
        .reconciliation
        .expect("reconciled");
    assert_eq!((found.corrected_refs, found.corrected_pins), (0, 0));
    assert_eq!(found.retired_strays, 0);
}

/// The status counts each queue and the oldest work in it.
#[test]
fn the_status_reports_every_queue() {
    let (gc, _dir) = open();
    let bytes = b"status".repeat(300);
    let record = put(&gc, 1, &bytes).record;
    let (content, generation) = generations(&gc, &record)[0];
    gc.store()
        .upload(SharingScope::STORE, b"staged only".repeat(300).as_slice())
        .expect("upload");
    gc.run(true).expect("enrolment");
    advance(Duration::from_secs(5));
    delete(&gc, 1);
    gc.store()
        .engine()
        .put(
            Partition::BlobRef,
            &retired_key(&content, generation + 7),
            &START.to_be_bytes(),
        )
        .expect("a retirement awaiting cleanup");
    let status = gc.status().expect("status");
    assert!(status.active);
    assert_eq!(status.reconciled_at_micros, Some(START));
    assert_eq!(status.grace_secs, 3600);
    assert_eq!(status.candidates, 1);
    assert_eq!(status.oldest_candidate_micros, Some(START + 5_000_000));
    assert_eq!(status.staged_uploads, 1);
    assert_eq!(status.oldest_upload_micros, Some(START));
    assert_eq!(status.retired_pending, 1);
    assert_eq!(status.retired_pending_bytes, bytes.len() as u64);
    assert_eq!(status.oldest_retired_micros, Some(START));
    assert!(status.last_pass.is_some());
    assert_eq!(status.last_error, None);
}
