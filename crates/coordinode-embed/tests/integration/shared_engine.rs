//! Integration tests: graph data and blobs live in one StorageEngine.
//!
//! Verifies that engine_shared() returns an Arc to the SAME storage instance
//! used by Database, so writes through one handle are visible through the
//! other, and that blobs stored through the database live beside graph data
//! in it.
//! No separate blob_store subdirectory; all partitions in one storage DB.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use coordinode_embed::Database;
use coordinode_embed::db::BlobId;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// The bytes of the blob `id` names, read whole.
fn read_blob(db: &Database, id: &BlobId) -> Vec<u8> {
    let reader = db.open_blob(id).expect("open").expect("present");
    (0..reader.record().blob.chunk_count())
        .flat_map(|i| reader.chunk(i).expect("chunk"))
        .collect()
}

// ── engine_shared() identity ────────────────────────────────────────

/// engine_shared() and engine() point to the same storage instance:
/// a write through the Arc is immediately visible through the borrow.
#[test]
fn shared_engine_write_visible_through_database() {
    let (db, _dir) = open_db();
    let shared: Arc<StorageEngine> = db.engine_shared();

    let key = b"blob:shared-identity";
    let value = b"hello coordinode";
    shared
        .put(Partition::Blob, key, value)
        .expect("put via shared");

    // Read through Database.engine() — must see the same data.
    let got = db
        .engine()
        .get(Partition::Blob, key)
        .expect("get via engine()")
        .expect("value must exist");
    assert_eq!(got.as_ref(), value.as_slice());
}

/// Write through Database.engine(), read through engine_shared() — symmetry.
#[test]
fn database_write_visible_through_shared_engine() {
    let (db, _dir) = open_db();
    let shared: Arc<StorageEngine> = db.engine_shared();

    let key = b"blob:deadbeef";
    let value = b"payload from database side";
    db.engine()
        .put(Partition::Blob, key, value)
        .expect("put via engine()");

    let got = shared
        .get(Partition::Blob, key)
        .expect("get via shared")
        .expect("must exist");
    assert_eq!(got.as_ref(), value.as_slice());
}

// ── Cypher + Blob coexistence ───────────────────────────────────────

/// Cypher writes and blob uploads land in the same storage instance without
/// disturbing each other.
#[test]
fn cypher_and_blob_coexist_in_same_engine() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create node");
    let data = b"blob content alongside graph data";
    let uploaded = db.upload_blob(data, None).expect("upload");

    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(read_blob(&db, &uploaded.id), data);
}

// ── Persistence across reopen ───────────────────────────────────────

/// A blob survives Database close + reopen together with the graph.
#[test]
fn blob_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let data = b"persistent blob payload".to_vec();

    // Session 1: write blob and graph data.
    let id = {
        let mut db = Database::open(dir.path()).expect("open");
        db.execute_cypher("CREATE (n:Doc {title: 'Persistence Test'})")
            .expect("create node");
        db.upload_blob(&data, None).expect("upload").id
    };

    // Session 2: reopen and verify both graph and blob data survived.
    let mut db = Database::open(dir.path()).expect("reopen");
    let rows = db
        .execute_cypher("MATCH (n:Doc) RETURN n.title")
        .expect("match");
    assert_eq!(rows.len(), 1);
    assert_eq!(read_blob(&db, &id), data);
}

// ── Concurrent access ───────────────────────────────────────────────

/// Multiple Arc clones can write to different partitions concurrently.
/// Simulates BlobService + CypherService operating in parallel.
#[test]
fn concurrent_blob_and_graph_writes() {
    let (db, _dir) = open_db();
    let shared1 = db.engine_shared();
    let shared2 = db.engine_shared();

    // Spawn threads writing to Blob and Schema partitions concurrently.
    let blob_thread = std::thread::spawn(move || {
        for i in 0..100 {
            let key = format!("blob:concurrent_{i}");
            let val = format!("chunk_data_{i}");
            shared1
                .put(Partition::Blob, key.as_bytes(), val.as_bytes())
                .expect("concurrent blob put");
        }
    });

    let schema_thread = std::thread::spawn(move || {
        for i in 0..100 {
            let key = format!("meta:concurrent_test_{i}");
            let val = format!("schema_val_{i}");
            shared2
                .put(Partition::Schema, key.as_bytes(), val.as_bytes())
                .expect("concurrent schema put");
        }
    });

    blob_thread.join().expect("blob thread");
    schema_thread.join().expect("schema thread");

    // Verify all writes are visible.
    for i in 0..100 {
        let blob_key = format!("blob:concurrent_{i}");
        let got = db
            .engine()
            .get(Partition::Blob, blob_key.as_bytes())
            .expect("get concurrent blob")
            .expect("concurrent blob must exist");
        assert_eq!(
            got.as_ref(),
            format!("chunk_data_{i}").as_bytes(),
            "blob {i} mismatch"
        );
    }
}

// ── Server scenario: Arc<Mutex<Database>> + engine_shared() ─────────

/// The server pattern: one Database behind a lock serves Cypher and blob
/// calls alike, and neither disturbs the other.
#[test]
fn server_pattern_mutex_database_serves_cypher_and_blobs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(Mutex::new(Database::open(dir.path()).expect("open")));

    {
        let mut db = database.lock().expect("lock for cypher");
        db.execute_cypher("CREATE (n:User {name: 'Alice'})")
            .expect("create");
        db.execute_cypher("CREATE (n:Doc {title: 'Report'})")
            .expect("create doc");
    }

    let data = b"binary attachment for the report";
    let id = database
        .lock()
        .expect("lock for upload")
        .upload_blob(data, None)
        .expect("upload")
        .id;

    {
        let mut db = database.lock().expect("lock for query");
        let rows = db
            .execute_cypher("MATCH (n:User) RETURN n.name")
            .expect("match");
        assert_eq!(rows.len(), 1);
    }
    assert_eq!(read_blob(&database.lock().expect("lock"), &id), data);
}

/// Simulates concurrent CypherService and BlobService operations
/// through the server pattern (Arc<Mutex<Database>> + Arc<StorageEngine>).
#[test]
fn server_pattern_concurrent_cypher_and_blob() {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(Mutex::new(Database::open(dir.path()).expect("open")));

    let blob_engine = database.lock().expect("lock").engine_shared();

    // Spawn blob writer thread (simulates BlobService gRPC handler)
    let blob_handle = {
        let engine = Arc::clone(&blob_engine);
        std::thread::spawn(move || {
            for i in 0..50 {
                let data = format!("chunk_payload_{i}");
                let key = format!("blob:server_test_{i}");
                engine
                    .put(Partition::Blob, key.as_bytes(), data.as_bytes())
                    .expect("blob put");
            }
        })
    };

    // Spawn cypher writer thread (simulates CypherService gRPC handler)
    let cypher_handle = {
        let db = Arc::clone(&database);
        std::thread::spawn(move || {
            for i in 0..20 {
                let mut db = db.lock().expect("lock for cypher");
                db.execute_cypher(&format!("CREATE (n:Item {{idx: {i}}})"))
                    .expect("create");
            }
        })
    };

    blob_handle.join().expect("blob thread");
    cypher_handle.join().expect("cypher thread");

    // Verify all blob data written
    for i in 0..50 {
        let key = format!("blob:server_test_{i}");
        let got = blob_engine
            .get(Partition::Blob, key.as_bytes())
            .expect("get blob")
            .expect("blob must exist");
        assert_eq!(got.as_ref(), format!("chunk_payload_{i}").as_bytes(),);
    }

    // Verify all graph data written
    {
        let mut db = database.lock().expect("lock for final verify");
        let rows = db
            .execute_cypher("MATCH (n:Item) RETURN n.idx")
            .expect("match items");
        assert_eq!(rows.len(), 20, "all 20 items must exist");
    }
}

// ── Delete through shared engine ────────────────────────────────────

/// Delete via shared engine, verify absent through engine().
#[test]
fn delete_through_shared_engine() {
    let (db, _dir) = open_db();
    let shared = db.engine_shared();

    let key = b"blob:to_delete";
    shared.put(Partition::Blob, key, b"temp").expect("put");
    assert!(
        db.engine()
            .get(Partition::Blob, key)
            .expect("get")
            .is_some()
    );

    shared.delete(Partition::Blob, key).expect("delete");
    assert!(
        db.engine()
            .get(Partition::Blob, key)
            .expect("get after delete")
            .is_none()
    );
}
