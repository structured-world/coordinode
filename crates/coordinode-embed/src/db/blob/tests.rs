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

#[test]
fn blob_ids_round_trip_through_text() {
    let id = BlobId::from_bytes([0xA5; 16]);
    let text = id.to_string();
    assert_eq!(text.len(), 32);
    assert_eq!(text.parse::<BlobId>().expect("parse"), id);
    assert!("short".parse::<BlobId>().is_err());
    assert!("zz".repeat(16).parse::<BlobId>().is_err());
}
