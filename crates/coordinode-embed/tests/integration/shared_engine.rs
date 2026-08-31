//! Integration tests: Database and BlobService share a single StorageEngine.
//!
//! Verifies G023: engine_shared() returns an Arc to the SAME storage instance
//! used by Database — writes through one handle are visible through the other.
//! No separate blob_store subdirectory; all partitions in one storage DB.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::{Arc, Mutex};

use coordinode_core::graph::blob::{self, encode_blob_key};
use coordinode_embed::Database;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

// ── engine_shared() identity ────────────────────────────────────────

/// engine_shared() and engine() point to the same storage instance:
/// a write through the Arc is immediately visible through the borrow.
#[test]
fn shared_engine_write_visible_through_database() {
    let (db, _dir) = open_db();
    let shared: Arc<StorageEngine> = db.engine_shared();

    // Write a blob chunk through the shared Arc handle.
    let data = b"hello coordinode blob";
    let (blob_ref, chunks) = blob::create_blob(data);
    assert_eq!(chunks.len(), 1, "small data = single chunk");

    let (chunk_id, chunk_data) = &chunks[0];
    let key = encode_blob_key(chunk_id);
    shared
        .put(Partition::Blob, &key, chunk_data)
        .expect("put via shared");

    // Read through Database.engine() — must see the same data.
    let got = db
        .engine()
        .get(Partition::Blob, &key)
        .expect("get via engine()")
        .expect("chunk must exist");
    assert_eq!(got.as_ref(), chunk_data.as_slice());

    // Verify blob_ref metadata round-trip through BlobRef partition.
    let meta_key = b"blobmeta:test1";
    let meta_val = blob_ref.to_msgpack().expect("serialize");
    shared
        .put(Partition::BlobRef, meta_key, &meta_val)
        .expect("put blobref via shared");

    let got_meta = db
        .engine()
        .get(Partition::BlobRef, meta_key)
        .expect("get blobref via engine()")
        .expect("blobref must exist");
    assert_eq!(got_meta.as_ref(), meta_val.as_slice());
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

/// Database executes Cypher while BlobService uses the shared engine.
/// Both write to different partitions of the same storage instance.
#[test]
fn cypher_and_blob_coexist_in_same_engine() {
    let (mut db, _dir) = open_db();
    let shared: Arc<StorageEngine> = db.engine_shared();

    // Write graph data via Cypher.
    db.execute_cypher("CREATE (n:User {name: 'Alice'})")
        .expect("create node");

    // Write blob data via shared engine (simulating BlobService).
    let data = b"blob content alongside graph data";
    let (_blob_ref, chunks) = blob::create_blob(data);
    for (chunk_id, chunk_data) in &chunks {
        let key = encode_blob_key(chunk_id);
        shared
            .put(Partition::Blob, &key, chunk_data)
            .expect("put blob chunk");
    }

    // Both are visible: Cypher query returns the node...
    let rows = db
        .execute_cypher("MATCH (n:User) RETURN n.name")
        .expect("match");
    assert_eq!(rows.len(), 1);

    // ...and blob chunk is readable.
    let (chunk_id, chunk_data) = &chunks[0];
    let key = encode_blob_key(chunk_id);
    let got = shared
        .get(Partition::Blob, &key)
        .expect("get blob")
        .expect("blob must exist");
    assert_eq!(got.as_ref(), chunk_data.as_slice());
}

// ── Persistence across reopen ───────────────────────────────────────

/// Blob data written through engine_shared() survives Database close + reopen.
#[test]
fn blob_persists_across_reopen() {
    let dir = tempfile::tempdir().expect("tempdir");
    let blob_key;
    let expected_data;

    // Session 1: write blob and graph data.
    {
        let mut db = Database::open(dir.path()).expect("open");
        let shared = db.engine_shared();

        db.execute_cypher("CREATE (n:Doc {title: 'Persistence Test'})")
            .expect("create node");

        let data = b"persistent blob payload";
        expected_data = data.to_vec();
        let (_blob_ref, chunks) = blob::create_blob(data);
        let (chunk_id, chunk_data) = &chunks[0];
        blob_key = encode_blob_key(chunk_id);
        shared
            .put(Partition::Blob, &blob_key, chunk_data)
            .expect("put blob");
    }

    // Session 2: reopen and verify both graph and blob data survived.
    {
        let mut db = Database::open(dir.path()).expect("reopen");

        // Graph data persisted.
        let rows = db
            .execute_cypher("MATCH (n:Doc) RETURN n.title")
            .expect("match");
        assert_eq!(rows.len(), 1);

        // Blob data persisted — readable through engine() and engine_shared().
        let got = db
            .engine()
            .get(Partition::Blob, &blob_key)
            .expect("get blob after reopen")
            .expect("blob must survive reopen");
        assert_eq!(got.as_ref(), expected_data.as_slice());

        let got_shared = db
            .engine_shared()
            .get(Partition::Blob, &blob_key)
            .expect("get blob via shared after reopen")
            .expect("blob via shared must survive reopen");
        assert_eq!(got_shared.as_ref(), expected_data.as_slice());
    }
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

/// Simulates the exact pattern from coordinode-server main.rs:
/// Database wrapped in Arc<Mutex<>>, BlobService gets engine_shared()
/// through a Mutex lock, then both operate independently.
#[test]
fn server_pattern_mutex_database_with_shared_blob_engine() {
    let dir = tempfile::tempdir().expect("tempdir");
    let database = Arc::new(Mutex::new(Database::open(dir.path()).expect("open")));

    // BlobService gets engine_shared() — same pattern as main.rs line 114-117
    let blob_engine: Arc<StorageEngine> = database
        .lock()
        .expect("lock for engine_shared")
        .engine_shared();

    // CypherService writes graph data through Mutex-locked Database
    {
        let mut db = database.lock().expect("lock for cypher");
        db.execute_cypher("CREATE (n:User {name: 'Alice'})")
            .expect("create");
        db.execute_cypher("CREATE (n:Doc {title: 'Report'})")
            .expect("create doc");
    }

    // BlobService writes blob data through the shared engine (no Mutex needed)
    let data = b"binary attachment for the report";
    let (blob_ref, chunks) = blob::create_blob(data);
    for (chunk_id, chunk_data) in &chunks {
        let key = encode_blob_key(chunk_id);
        blob_engine
            .put(Partition::Blob, &key, chunk_data)
            .expect("blob put");
    }
    let meta_key = b"blobmeta:report_attachment";
    let meta_val = blob_ref.to_msgpack().expect("serialize blobref");
    blob_engine
        .put(Partition::BlobRef, meta_key, &meta_val)
        .expect("blobref put");

    // CypherService queries graph — blob data doesn't interfere
    {
        let mut db = database.lock().expect("lock for query");
        let rows = db
            .execute_cypher("MATCH (n:User) RETURN n.name")
            .expect("match");
        assert_eq!(rows.len(), 1);
    }

    // Blob data readable through Database.engine() (same storage instance)
    {
        let db = database.lock().expect("lock for verify");
        let got = db
            .engine()
            .get(Partition::BlobRef, meta_key)
            .expect("get blobref")
            .expect("blobref must exist");
        assert_eq!(got.as_ref(), meta_val.as_slice());
    }
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
