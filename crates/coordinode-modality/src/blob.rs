//! Blob store — content-addressed chunks + per-property blob refs.
//!
//! Large property values (≥ [`INLINE_THRESHOLD`]) are split into
//! content-addressed chunks ([`Partition::Blob`]). Each (node, prop)
//! that holds such a value carries a `BlobRef` listing chunk hashes
//! ([`Partition::BlobRef`]). Content-addressing gives automatic
//! deduplication — identical chunks are stored once cluster-wide.
//!
//! The store exposes:
//!
//! - Per-chunk put/get (idempotent — content-addressed, so a re-put
//!   of an existing chunk hash is a no-op semantically).
//! - Per-(node, prop) blob ref put/get/delete.
//!
//! Chunk garbage collection (dropping orphan chunks once no `BlobRef`
//! references them) is a Layer 3 concern and lives outside this
//! trait: it requires a global scan across all blob refs and runs as
//! a background sweep. The store deliberately does not expose
//! `delete_chunk` directly — the only safe way to drop a chunk is
//! through the GC.
//!
//! ## Transaction threading (ADR-041)
//!
//! Every method threads the active [`Transaction`]. Writes
//! (`put_chunk` / `put_chunks` / `put_blob_ref` / `delete_blob_ref`)
//! take `&mut Transaction` and buffer the `Partition::Blob` /
//! `Partition::BlobRef` mutations on it, so the chunks plus the ref
//! that points at them commit atomically with the node write that
//! produced the blob (a multi-chunk `put_chunks` lands as one commit
//! or none). Reads (`get_chunk` / `get_blob_ref`) take `&Transaction`
//! and read through the committed MVCC snapshot via
//! [`Transaction::read_untracked`] — content-addressed chunks are
//! immutable, so they never join the OCC conflict set.
//!
//! [`Transaction`]: coordinode_storage::engine::transaction::Transaction

use coordinode_core::graph::blob::{encode_blob_key, encode_blobref_key, BlobRef, ChunkId};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

pub use coordinode_core::graph::blob::INLINE_THRESHOLD;

/// Layer 4 blob store: chunk-level CAS storage + per-property blob
/// references over a [`Transaction`]. Hides partition layout and key
/// encoders.
pub trait BlobStore {
    /// Persist a chunk by content hash. Idempotent: re-putting an
    /// existing `chunk_id` buffers identical bytes. Buffered on `txn`.
    fn put_chunk(&self, txn: &mut Transaction, chunk_id: &ChunkId, data: &[u8]) -> StoreResult<()>;

    /// Persist many chunks. All buffer on the same `txn`, so they
    /// commit together. Useful when a new blob ref pulls in multiple
    /// new chunks.
    fn put_chunks(&self, txn: &mut Transaction, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()>;

    /// Fetch a chunk by content hash. Returns `None` if not stored
    /// (either never written, or already swept by GC because no ref
    /// pointed at it).
    fn get_chunk(&self, txn: &Transaction, chunk_id: &ChunkId) -> StoreResult<Option<Vec<u8>>>;

    /// Persist the [`BlobRef`] for a (node, prop) pair. Overwrites any
    /// previous ref at the same key. Buffered on `txn`.
    fn put_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
        blob_ref: &BlobRef,
    ) -> StoreResult<()>;

    /// Fetch the [`BlobRef`] for a (node, prop) pair.
    fn get_blob_ref(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobRef>>;

    /// Remove the [`BlobRef`] for a (node, prop) pair. Chunks remain
    /// in [`Partition::Blob`] — orphan chunks are reclaimed by the
    /// Layer 3 GC sweep. Buffered on `txn`.
    fn delete_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`BlobStore`]. Stateless — all
/// storage access flows through the [`Transaction`] passed to each
/// method (ADR-041).
pub struct LocalBlobStore;

impl BlobStore for LocalBlobStore {
    fn put_chunk(&self, txn: &mut Transaction, chunk_id: &ChunkId, data: &[u8]) -> StoreResult<()> {
        txn.put(Partition::Blob, &encode_blob_key(chunk_id), data)?;
        Ok(())
    }

    fn put_chunks(&self, txn: &mut Transaction, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()> {
        for (id, data) in chunks {
            txn.put(Partition::Blob, &encode_blob_key(id), data)?;
        }
        Ok(())
    }

    fn get_chunk(&self, txn: &Transaction, chunk_id: &ChunkId) -> StoreResult<Option<Vec<u8>>> {
        Ok(txn.read_untracked(Partition::Blob, &encode_blob_key(chunk_id))?)
    }

    fn put_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
        blob_ref: &BlobRef,
    ) -> StoreResult<()> {
        let body = blob_ref.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "blob ref",
            message: format!("encode for node {}: {e}", node_id.as_raw()),
        })?;
        txn.put(
            Partition::BlobRef,
            &encode_blobref_key(node_id, prop_id),
            &body,
        )?;
        Ok(())
    }

    fn get_blob_ref(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobRef>> {
        let Some(bytes) =
            txn.read_untracked(Partition::BlobRef, &encode_blobref_key(node_id, prop_id))?
        else {
            return Ok(None);
        };
        BlobRef::from_msgpack(&bytes)
            .map(Some)
            .map_err(|e| StoreError::Decode {
                kind: "blob ref",
                message: format!("decode for node {}: {e}", node_id.as_raw()),
            })
    }

    fn delete_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<()> {
        txn.delete(Partition::BlobRef, &encode_blobref_key(node_id, prop_id))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
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
        let id = ChunkId::from_data(b"abc");
        assert!(read_blob(engine, |s, txn| s.get_chunk(txn, &id).expect("none")).is_none());
        write_blob(engine, |s, txn| {
            s.put_chunk(txn, &id, b"abc").expect("put");
        });
        let got = read_blob(engine, |s, txn| s.get_chunk(txn, &id).expect("some")).expect("Some");
        assert_eq!(got, b"abc");
    }

    #[test]
    fn put_chunks_commits_together() {
        // All chunks buffer on one transaction ⇒ visible together
        // after commit. Smoke check: write three chunks via
        // put_chunks, then verify each individually.
        let fx = open_engine();
        let engine = &fx.engine;
        let chunks = vec![
            (ChunkId::from_data(b"a"), b"a".to_vec()),
            (ChunkId::from_data(b"b"), b"b".to_vec()),
            (ChunkId::from_data(b"c"), b"c".to_vec()),
        ];
        write_blob(engine, |s, txn| {
            s.put_chunks(txn, &chunks).expect("batch");
        });
        for (id, expected) in &chunks {
            let got = read_blob(engine, |s, txn| s.get_chunk(txn, id).expect("ok"));
            assert_eq!(got.as_deref(), Some(expected.as_slice()));
        }
    }

    #[test]
    fn put_chunks_empty_is_noop() {
        let fx = open_engine();
        let engine = &fx.engine;
        write_blob(engine, |s, txn| {
            s.put_chunks(txn, &[]).expect("empty batch ok");
        });
    }

    #[test]
    fn blob_ref_round_trip() {
        let fx = open_engine();
        let engine = &fx.engine;
        let node_id = NodeId::from_raw(42);
        let prop_id = 7;

        // Construct a real BlobRef from a >threshold-byte payload.
        let payload = vec![0xaa_u8; INLINE_THRESHOLD + 64];
        let (blob_ref, chunks) = create_blob(&payload);
        write_blob(engine, |s, txn| {
            s.put_chunks(txn, &chunks).expect("chunks");
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
            let chunk = read_blob(engine, |s, txn| s.get_chunk(txn, id).expect("ok"))
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
        let node_id = NodeId::from_raw(11);
        let payload = vec![0x33_u8; INLINE_THRESHOLD + 8];
        let (blob_ref, chunks) = create_blob(&payload);
        write_blob(engine, |s, txn| {
            s.put_chunks(txn, &chunks).expect("chunks");
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
                read_blob(engine, |s, txn| s.get_chunk(txn, id).expect("ok")).is_some(),
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
        let id = ChunkId::from_data(b"xyz");
        write_blob(engine, |s, txn| {
            s.put_chunk(txn, &id, b"xyz").expect("first put");
            s.put_chunk(txn, &id, b"xyz").expect("second put");
        });
        let got = read_blob(engine, |s, txn| s.get_chunk(txn, &id).expect("ok"));
        assert_eq!(got.as_deref(), Some(b"xyz".as_slice()));
    }

    #[test]
    fn multiple_blob_refs_per_node_distinct_prop_ids() {
        // (node_id, prop_id) is the key. Same node with two property
        // IDs must keep two distinct refs.
        let fx = open_engine();
        let engine = &fx.engine;
        let node = NodeId::from_raw(7);
        let payload_a = vec![0x01_u8; INLINE_THRESHOLD + 4];
        let payload_b = vec![0x02_u8; INLINE_THRESHOLD + 4];
        let (ref_a, chunks_a) = create_blob(&payload_a);
        let (ref_b, chunks_b) = create_blob(&payload_b);
        write_blob(engine, |s, txn| {
            s.put_chunks(txn, &chunks_a).expect("chunks a");
            s.put_chunks(txn, &chunks_b).expect("chunks b");
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
        assert!(read_blob(engine, |s, txn| s.get_chunk(txn, &id).expect("ok")).is_none());
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
}
