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

use coordinode_core::graph::blob::{encode_blob_key, encode_blobref_key, BlobRef, ChunkId};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

use crate::error::{StoreError, StoreResult};

pub use coordinode_core::graph::blob::INLINE_THRESHOLD;

/// Layer 4 blob store: chunk-level CAS storage + per-property blob
/// references. Hides partition layout and key encoders.
pub trait BlobStore {
    /// Persist a chunk by content hash. Idempotent: re-putting an
    /// existing `chunk_id` overwrites identical bytes.
    fn put_chunk(&self, chunk_id: &ChunkId, data: &[u8]) -> StoreResult<()>;

    /// Persist many chunks atomically (single underlying batch).
    /// Useful when a new blob ref pulls in multiple new chunks.
    fn put_chunks(&self, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()>;

    /// Fetch a chunk by content hash. Returns `None` if not stored
    /// (either never written, or already swept by GC because no ref
    /// pointed at it).
    fn get_chunk(&self, chunk_id: &ChunkId) -> StoreResult<Option<Vec<u8>>>;

    /// Persist the [`BlobRef`] for a (node, prop) pair. Overwrites any
    /// previous ref at the same key.
    fn put_blob_ref(&self, node_id: NodeId, prop_id: u32, blob_ref: &BlobRef) -> StoreResult<()>;

    /// Fetch the [`BlobRef`] for a (node, prop) pair.
    fn get_blob_ref(&self, node_id: NodeId, prop_id: u32) -> StoreResult<Option<BlobRef>>;

    /// Remove the [`BlobRef`] for a (node, prop) pair. Chunks remain
    /// in [`Partition::Blob`] — orphan chunks are reclaimed by the
    /// Layer 3 GC sweep.
    fn delete_blob_ref(&self, node_id: NodeId, prop_id: u32) -> StoreResult<()>;
}

/// CE single-shard implementation of [`BlobStore`].
pub struct LocalBlobStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalBlobStore<'a> {
    /// Wrap a storage engine for blob-store operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

impl BlobStore for LocalBlobStore<'_> {
    fn put_chunk(&self, chunk_id: &ChunkId, data: &[u8]) -> StoreResult<()> {
        self.engine
            .put(Partition::Blob, &encode_blob_key(chunk_id), data)?;
        Ok(())
    }

    fn put_chunks(&self, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new(self.engine);
        for (id, data) in chunks {
            batch.put(Partition::Blob, encode_blob_key(id), data.clone());
        }
        batch.commit()?;
        Ok(())
    }

    fn get_chunk(&self, chunk_id: &ChunkId) -> StoreResult<Option<Vec<u8>>> {
        Ok(self
            .engine
            .get(Partition::Blob, &encode_blob_key(chunk_id))?
            .map(|b| b.to_vec()))
    }

    fn put_blob_ref(&self, node_id: NodeId, prop_id: u32, blob_ref: &BlobRef) -> StoreResult<()> {
        let body = blob_ref.to_msgpack().map_err(|e| StoreError::Decode {
            kind: "blob ref",
            message: format!("encode for node {}: {e}", node_id.as_raw()),
        })?;
        self.engine.put(
            Partition::BlobRef,
            &encode_blobref_key(node_id, prop_id),
            &body,
        )?;
        Ok(())
    }

    fn get_blob_ref(&self, node_id: NodeId, prop_id: u32) -> StoreResult<Option<BlobRef>> {
        let Some(bytes) = self
            .engine
            .get(Partition::BlobRef, &encode_blobref_key(node_id, prop_id))?
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

    fn delete_blob_ref(&self, node_id: NodeId, prop_id: u32) -> StoreResult<()> {
        self.engine
            .delete(Partition::BlobRef, &encode_blobref_key(node_id, prop_id))?;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use coordinode_core::graph::blob::create_blob;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use tempfile::TempDir;

    fn open_engine() -> (TempDir, StorageEngine) {
        let dir = TempDir::new().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "ep",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        (dir, engine)
    }

    #[test]
    fn chunk_round_trip() {
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);

        let id = ChunkId::from_data(b"abc");
        assert!(store.get_chunk(&id).expect("none").is_none());
        store.put_chunk(&id, b"abc").expect("put");
        let got = store.get_chunk(&id).expect("some").expect("Some");
        assert_eq!(got, b"abc");
    }

    #[test]
    fn put_chunks_is_atomic_batch() {
        // Single WriteBatch ⇒ all chunks visible together. Smoke
        // check: write three chunks via put_chunks, then verify each
        // individually.
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let chunks = vec![
            (ChunkId::from_data(b"a"), b"a".to_vec()),
            (ChunkId::from_data(b"b"), b"b".to_vec()),
            (ChunkId::from_data(b"c"), b"c".to_vec()),
        ];
        store.put_chunks(&chunks).expect("batch");
        for (id, expected) in &chunks {
            assert_eq!(
                store.get_chunk(id).expect("ok").as_deref(),
                Some(expected.as_slice()),
            );
        }
    }

    #[test]
    fn put_chunks_empty_is_noop() {
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        store.put_chunks(&[]).expect("empty batch ok");
    }

    #[test]
    fn blob_ref_round_trip() {
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let node_id = NodeId::from_raw(42);
        let prop_id = 7;

        // Construct a real BlobRef from a >threshold-byte payload.
        let payload = vec![0xaa_u8; INLINE_THRESHOLD + 64];
        let (blob_ref, chunks) = create_blob(&payload);
        store.put_chunks(&chunks).expect("chunks");
        store
            .put_blob_ref(node_id, prop_id, &blob_ref)
            .expect("put ref");

        let loaded = store
            .get_blob_ref(node_id, prop_id)
            .expect("some")
            .expect("Some");
        assert_eq!(loaded, blob_ref);

        // Reassemble payload via chunk gets.
        let mut reassembled = Vec::with_capacity(loaded.total_size as usize);
        for id in &loaded.chunks {
            let chunk = store.get_chunk(id).expect("ok").expect("chunk present");
            reassembled.extend_from_slice(&chunk);
        }
        assert_eq!(reassembled, payload);
    }

    #[test]
    fn delete_blob_ref_removes_pointer_keeps_chunks() {
        // Deleting the ref must NOT touch chunks — orphan chunks are
        // GC'd by Layer 3 sweep, not by the store. Verifies the
        // dedup-friendly behaviour.
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let node_id = NodeId::from_raw(11);
        let payload = vec![0x33_u8; INLINE_THRESHOLD + 8];
        let (blob_ref, chunks) = create_blob(&payload);
        store.put_chunks(&chunks).expect("chunks");
        store.put_blob_ref(node_id, 0, &blob_ref).expect("put ref");

        store.delete_blob_ref(node_id, 0).expect("delete");

        assert!(
            store.get_blob_ref(node_id, 0).expect("ok").is_none(),
            "ref must be gone",
        );
        for (id, _) in &chunks {
            assert!(
                store.get_chunk(id).expect("ok").is_some(),
                "chunks must remain (Layer 3 GC's job, not store's)",
            );
        }
    }

    #[test]
    fn put_chunk_overwrites_with_identical_bytes() {
        // Content-addressed: same ChunkId means same bytes. Putting
        // twice must be idempotent (the engine accepts the overwrite
        // silently).
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let id = ChunkId::from_data(b"xyz");
        store.put_chunk(&id, b"xyz").expect("first put");
        store.put_chunk(&id, b"xyz").expect("second put");
        assert_eq!(
            store.get_chunk(&id).expect("ok").as_deref(),
            Some(b"xyz".as_slice()),
        );
    }

    #[test]
    fn multiple_blob_refs_per_node_distinct_prop_ids() {
        // (node_id, prop_id) is the key. Same node with two property
        // IDs must keep two distinct refs.
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let node = NodeId::from_raw(7);
        let payload_a = vec![0x01_u8; INLINE_THRESHOLD + 4];
        let payload_b = vec![0x02_u8; INLINE_THRESHOLD + 4];
        let (ref_a, chunks_a) = create_blob(&payload_a);
        let (ref_b, chunks_b) = create_blob(&payload_b);
        store.put_chunks(&chunks_a).expect("chunks a");
        store.put_chunks(&chunks_b).expect("chunks b");
        store.put_blob_ref(node, 1, &ref_a).expect("ref a");
        store.put_blob_ref(node, 2, &ref_b).expect("ref b");

        let loaded_a = store
            .get_blob_ref(node, 1)
            .expect("ok")
            .expect("ref a present");
        let loaded_b = store
            .get_blob_ref(node, 2)
            .expect("ok")
            .expect("ref b present");
        assert_eq!(loaded_a, ref_a);
        assert_eq!(loaded_b, ref_b);
        assert_ne!(loaded_a, loaded_b);
    }

    #[test]
    fn get_chunk_missing_is_none() {
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let id = ChunkId::from_data(b"never");
        assert!(store.get_chunk(&id).expect("ok").is_none());
    }

    #[test]
    fn corrupt_blob_ref_surfaces_as_decode_error() {
        let (_dir, engine) = open_engine();
        let store = LocalBlobStore::new(&engine);
        let node_id = NodeId::from_raw(99);
        // Inject garbage bytes at the BlobRef key.
        engine
            .put(
                Partition::BlobRef,
                &encode_blobref_key(node_id, 0),
                &[0xde, 0xad, 0xbe, 0xef],
            )
            .expect("inject");
        let err = store.get_blob_ref(node_id, 0).expect_err("must error");
        assert!(matches!(
            err,
            StoreError::Decode {
                kind: "blob ref",
                ..
            }
        ));
    }
}
