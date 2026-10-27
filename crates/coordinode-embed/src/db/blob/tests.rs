use std::sync::Arc;

use coordinode_core::graph::blob::DEFAULT_CHUNK_SIZE;

use super::BlobId;
use crate::db::{Database, DatabaseError};
use crate::operation::{IssuerId, OperationOutcome};

fn open() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open");
    (db, dir)
}

fn read(db: &Database, id: &BlobId) -> Vec<u8> {
    let reader = db.open_blob(id).expect("open").expect("present");
    (0..reader.record().blob.chunk_count())
        .flat_map(|i| reader.chunk(i).expect("chunk"))
        .collect()
}

/// Two uploads of the same bytes are two blobs: deleting one leaves the
/// other whole, while the bytes are stored once underneath.
#[test]
fn uploads_of_the_same_bytes_are_separate_blobs() {
    let (db, _dir) = open();
    let data = vec![0x33_u8; DEFAULT_CHUNK_SIZE + 9];
    let first = db.upload_blob(&data, None).expect("first");
    let second = db.upload_blob(&data, None).expect("second");
    assert_ne!(first.id, second.id);
    assert_eq!(first.shared, 0);
    assert_eq!(second.shared, 2, "the second upload shares both chunks");

    assert!(db.delete_blob(&first.id).expect("delete").is_some());
    assert!(db.open_blob(&first.id).expect("open").is_none());
    assert_eq!(read(&db, &second.id), data);
    assert_eq!(db.delete_blob(&first.id).expect("again"), None);
}

/// An upload retried under its operation after a lost reply stores nothing
/// more: it is refused with the operation's committed outcome, and the blob
/// the first attempt stored is the one the operation names.
#[test]
fn an_upload_retried_under_its_operation_stores_once() {
    let (db, _dir) = open();
    let issuer = db.enroll_issuer(IssuerId([0x51; 16])).expect("enroll");
    let descriptor = issuer.next().expect("descriptor");
    let data = b"exactly once".to_vec();

    let stored = db.upload_blob(&data, Some(&descriptor)).expect("upload");
    match db.upload_blob(&data, Some(&descriptor)) {
        Err(DatabaseError::OperationSettled {
            outcome,
            intent_differs,
            ..
        }) => {
            assert!(
                matches!(outcome, OperationOutcome::Committed { .. }),
                "{outcome:?}"
            );
            assert!(!intent_differs);
        }
        other => panic!("a settled retry stores nothing: {other:?}"),
    }
    // The same number reused for other bytes is told apart.
    match db.upload_blob(b"other bytes", Some(&descriptor)) {
        Err(DatabaseError::OperationSettled { intent_differs, .. }) => assert!(intent_differs),
        other => panic!("another upload under a settled number is refused: {other:?}"),
    }
    assert_eq!(BlobId::for_operation(&descriptor), stored.id);
    assert_eq!(read(&db, &stored.id), data);
}

/// Blob writes are refused while the group holds other members, before
/// anything is written; reads stay served.
#[test]
fn blob_writes_are_refused_in_a_group_with_other_members() {
    let (mut db, _dir) = open();
    let kept = db
        .upload_blob(b"before the group grew", None)
        .expect("alone");
    db.set_group_members(Arc::new(|| 3));
    assert!(matches!(
        db.upload_blob(b"refused", None),
        Err(DatabaseError::Unsupported(_))
    ));
    assert!(matches!(
        db.delete_blob(&kept.id),
        Err(DatabaseError::Unsupported(_))
    ));
    assert_eq!(read(&db, &kept.id), b"before the group grew");
}

/// `len` bytes that do not compress, so the space they take is what they are.
fn incompressible(seed: u64, len: usize) -> Vec<u8> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64 ^ seed;
    let mut out = Vec::with_capacity(len + 8);
    while out.len() < len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.extend_from_slice(&state.to_le_bytes());
    }
    out.truncate(len);
    out
}

/// A database whose collector takes released content at once, with no
/// retained history holding it either.
fn collecting_at_once() -> (Database, tempfile::TempDir) {
    let (db, dir) = open();
    db.set_retention_window(std::time::Duration::ZERO);
    db.set_blob_gc_policy(crate::db::GcPolicy {
        grace: std::time::Duration::ZERO,
        compact_after_bytes: 0,
        ..crate::db::GcPolicy::default()
    });
    (db, dir)
}

/// A deleted blob's bytes are collected and their space returned by the
/// database's own collector: the deleted payload is counted, and the Blob
/// partition shrinks back to well under the blob's size.
#[test]
fn a_deleted_blob_is_collected_and_its_space_returns() {
    let (db, _dir) = collecting_at_once();
    let data = incompressible(1, 4 * DEFAULT_CHUNK_SIZE);
    let stored = db.upload_blob(&data, None).expect("upload");
    let full = db.blob_gc_status().expect("status").blob_disk_bytes;
    assert!(full >= data.len() as u64, "{full}");

    db.delete_blob(&stored.id).expect("delete");
    // This pass, or the worker the deletion woke, collects it; either way it
    // is done when the pass returns, passes being run one at a time.
    db.run_blob_gc(true).expect("pass");
    let status = db.blob_gc_status().expect("status");
    assert_eq!(status.reclaimed_bytes, data.len() as u64);
    assert_eq!((status.candidates, status.retired_pending), (0, 0));
    // The emptied value file goes with the next compaction, here the one
    // for another deleted blob.
    let other = db
        .upload_blob(&incompressible(9, DEFAULT_CHUNK_SIZE), None)
        .expect("upload");
    db.delete_blob(&other.id).expect("delete");
    db.run_blob_gc(false).expect("pass");
    let status = db.blob_gc_status().expect("status");
    assert!(
        status.blob_disk_bytes < full / 2,
        "{} of {full} bytes still on disk",
        status.blob_disk_bytes
    );
}

/// Nobody has to ask: the background worker, woken by the deletion,
/// collects the blob on its own.
#[test]
fn the_collector_reclaims_a_deleted_blob_on_its_own() {
    let (db, _dir) = collecting_at_once();
    let data = incompressible(2, 2 * DEFAULT_CHUNK_SIZE);
    let stored = db.upload_blob(&data, None).expect("upload");
    db.delete_blob(&stored.id).expect("delete");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let status = db.blob_gc_status().expect("status");
        if status.reclaimed_bytes == data.len() as u64 && !status.compaction_pending {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the collector did not reclaim the blob: {status:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// A manual run is the automatic one: it does not shorten the grace, so a
/// blob deleted a moment ago keeps its bytes and its candidate waits.
#[test]
fn a_manual_run_keeps_the_grace() {
    let (db, _dir) = open();
    let data = incompressible(3, DEFAULT_CHUNK_SIZE);
    let stored = db.upload_blob(&data, None).expect("upload");
    db.delete_blob(&stored.id).expect("delete");
    let pass = db.run_blob_gc(true).expect("pass");
    assert_eq!(pass.retired, 0);
    let status = db.blob_gc_status().expect("status");
    assert_eq!((status.candidates, status.candidates_due), (1, 0));
    assert_eq!(status.grace_secs, 24 * 60 * 60);
    assert_eq!(status.reclaimed_bytes, 0);
}

/// A member of a group that holds other members collects nothing and says
/// why: blob payload lives only where it was written.
#[test]
fn a_member_of_a_larger_group_collects_nothing() {
    let (mut db, _dir) = open();
    let stored = db
        .upload_blob(&incompressible(4, DEFAULT_CHUNK_SIZE), None)
        .expect("alone");
    db.delete_blob(&stored.id).expect("delete alone");
    db.set_group_members(Arc::new(|| 3));
    // Due now, were the member alone.
    db.set_retention_window(std::time::Duration::ZERO);
    db.set_blob_gc_policy(crate::db::GcPolicy {
        grace: std::time::Duration::ZERO,
        ..crate::db::GcPolicy::default()
    });
    let pass = db.run_blob_gc(true).expect("pass");
    assert_eq!(pass.retired, 0);
    assert_eq!(db.blob_gc_status().expect("status").reclaimed_bytes, 0);
    assert!(
        pass.blockers.iter().any(|b| b.contains("group")),
        "{:?}",
        pass.blockers
    );
    assert!(!db.blob_gc_status().expect("status").active);
}

#[test]
fn blob_ids_round_trip_through_text() {
    let id = BlobId::from_bytes([0xA5; 16]);
    let text = id.to_string();
    assert_eq!(text.len(), 32);
    assert_eq!(text.parse::<BlobId>().expect("parse"), id);
    assert!("short".parse::<BlobId>().is_err());
    assert!("zz".repeat(16).parse::<BlobId>().is_err());
}
