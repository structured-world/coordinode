use super::*;
use coordinode_core::graph::blob::create_blob;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::CommitContext;

/// Logic-test fixture (memory backing, env-flippable). Blob
/// chunk CRUD verifies store contract, not persistence.
fn open_engine() -> coordinode_test_fixtures::EngineFixture {
    coordinode_test_fixtures::engine_for_logic()
}

/// Apply blob writes in one MVCC transaction and commit, so the
/// buffered chunk / ref mutations land for a subsequent read.
fn write_blob(engine: &StorageEngine, body: impl FnOnce(&LocalBlobStore, &mut Transaction)) {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    body(&LocalBlobStore, &mut txn);
    let wc = WriteConcern::majority();
    let ctx = CommitContext {
        write_concern: &wc,
        pipeline: None,
        id_gen: None,
        drain_buffer: None,
        nvme_write_buffer: None,
    };
    txn.commit(&ctx).expect("commit blob");
}

/// Run a blob read closure against the latest committed snapshot.
fn read_blob<R>(
    engine: &StorageEngine,
    body: impl FnOnce(&LocalBlobStore, &Transaction) -> R,
) -> R {
    let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
    let read_ts = oracle.next();
    let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
    body(&LocalBlobStore, &txn)
}

#[test]
fn chunk_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let id = ChunkId::from_data(b"abc");
    assert!(store.get_chunk(engine, &id).expect("none").is_none());
    store.put_chunk(engine, &id, b"abc").expect("put");
    let got = store.get_chunk(engine, &id).expect("some").expect("Some");
    assert_eq!(got, b"abc");
}

#[test]
fn put_chunks_commits_together() {
    // All chunks land in one engine WriteBatch ⇒ visible together.
    // Smoke check: write three chunks via put_chunks, then verify
    // each individually.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let chunks = vec![
        (ChunkId::from_data(b"a"), b"a".to_vec()),
        (ChunkId::from_data(b"b"), b"b".to_vec()),
        (ChunkId::from_data(b"c"), b"c".to_vec()),
    ];
    store.put_chunks(engine, &chunks).expect("batch");
    for (id, expected) in &chunks {
        let got = store.get_chunk(engine, id).expect("ok");
        assert_eq!(got.as_deref(), Some(expected.as_slice()));
    }
}

#[test]
fn put_chunks_empty_is_noop() {
    let fx = open_engine();
    let engine = &fx.engine;
    LocalBlobStore
        .put_chunks(engine, &[])
        .expect("empty batch ok");
}

#[test]
fn blob_ref_round_trip() {
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let node_id = NodeId::from_raw(42);
    let prop_id = 7;

    // Construct a real BlobRef from a >threshold-byte payload.
    let payload = vec![0xaa_u8; INLINE_THRESHOLD + 64];
    let (blob_ref, chunks) = create_blob(&payload);
    // Chunks land on the data plane (direct), the ref on the
    // metadata plane (transactional).
    store.put_chunks(engine, &chunks).expect("chunks");
    write_blob(engine, |s, txn| {
        s.put_blob_ref(txn, node_id, prop_id, &blob_ref)
            .expect("put ref");
    });

    let loaded = read_blob(engine, |s, txn| {
        s.get_blob_ref(txn, node_id, prop_id).expect("some")
    })
    .expect("Some");
    assert_eq!(loaded, blob_ref);

    // Reassemble payload via chunk gets.
    let mut reassembled = Vec::with_capacity(loaded.total_size as usize);
    for id in &loaded.chunks {
        let chunk = store
            .get_chunk(engine, id)
            .expect("ok")
            .expect("chunk present");
        reassembled.extend_from_slice(&chunk);
    }
    assert_eq!(reassembled, payload);
}

#[test]
fn delete_blob_ref_removes_pointer_keeps_chunks() {
    // Deleting the ref must NOT touch chunks — orphan chunks are
    // GC'd by Layer 3 sweep, not by the store. Verifies the
    // dedup-friendly behaviour.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let node_id = NodeId::from_raw(11);
    let payload = vec![0x33_u8; INLINE_THRESHOLD + 8];
    let (blob_ref, chunks) = create_blob(&payload);
    store.put_chunks(engine, &chunks).expect("chunks");
    write_blob(engine, |s, txn| {
        s.put_blob_ref(txn, node_id, 0, &blob_ref).expect("put ref");
    });

    write_blob(engine, |s, txn| {
        s.delete_blob_ref(txn, node_id, 0).expect("delete");
    });

    assert!(
        read_blob(engine, |s, txn| s
            .get_blob_ref(txn, node_id, 0)
            .expect("ok"))
        .is_none(),
        "ref must be gone",
    );
    for (id, _) in &chunks {
        assert!(
            store.get_chunk(engine, id).expect("ok").is_some(),
            "chunks must remain (Layer 3 GC's job, not store's)",
        );
    }
}

#[test]
fn put_chunk_overwrites_with_identical_bytes() {
    // Content-addressed: same ChunkId means same bytes. Putting
    // twice must be idempotent (the engine accepts the overwrite
    // silently).
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let id = ChunkId::from_data(b"xyz");
    store.put_chunk(engine, &id, b"xyz").expect("first put");
    store.put_chunk(engine, &id, b"xyz").expect("second put");
    let got = store.get_chunk(engine, &id).expect("ok");
    assert_eq!(got.as_deref(), Some(b"xyz".as_slice()));
}

#[test]
fn multiple_blob_refs_per_node_distinct_prop_ids() {
    // (node_id, prop_id) is the key. Same node with two property
    // IDs must keep two distinct refs.
    let fx = open_engine();
    let engine = &fx.engine;
    let store = LocalBlobStore;
    let node = NodeId::from_raw(7);
    let payload_a = vec![0x01_u8; INLINE_THRESHOLD + 4];
    let payload_b = vec![0x02_u8; INLINE_THRESHOLD + 4];
    let (ref_a, chunks_a) = create_blob(&payload_a);
    let (ref_b, chunks_b) = create_blob(&payload_b);
    store.put_chunks(engine, &chunks_a).expect("chunks a");
    store.put_chunks(engine, &chunks_b).expect("chunks b");
    write_blob(engine, |s, txn| {
        s.put_blob_ref(txn, node, 1, &ref_a).expect("ref a");
        s.put_blob_ref(txn, node, 2, &ref_b).expect("ref b");
    });

    let loaded_a = read_blob(engine, |s, txn| s.get_blob_ref(txn, node, 1).expect("ok"))
        .expect("ref a present");
    let loaded_b = read_blob(engine, |s, txn| s.get_blob_ref(txn, node, 2).expect("ok"))
        .expect("ref b present");
    assert_eq!(loaded_a, ref_a);
    assert_eq!(loaded_b, ref_b);
    assert_ne!(loaded_a, loaded_b);
}

#[test]
fn get_chunk_missing_is_none() {
    let fx = open_engine();
    let engine = &fx.engine;
    let id = ChunkId::from_data(b"never");
    assert!(LocalBlobStore.get_chunk(engine, &id).expect("ok").is_none());
}

#[test]
fn corrupt_blob_ref_surfaces_as_decode_error() {
    let fx = open_engine();
    let engine = &fx.engine;
    let node_id = NodeId::from_raw(99);
    // Inject garbage bytes at the BlobRef key through a committed
    // transaction.
    write_blob(engine, |_s, txn| {
        txn.put(
            Partition::BlobRef,
            &encode_blobref_key(node_id, 0),
            &[0xde, 0xad, 0xbe, 0xef],
        )
        .expect("inject");
    });
    let err = read_blob(engine, |s, txn| {
        s.get_blob_ref(txn, node_id, 0).expect_err("must error")
    });
    assert!(matches!(
        err,
        StoreError::Decode {
            kind: "blob ref",
            ..
        }
    ));
}
