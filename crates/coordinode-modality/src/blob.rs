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
//! ## Data plane vs metadata plane (ADR-011)
//!
//! Blobs follow the object-store separation that lets Ceph/MinIO scale:
//! bulk object **data** is placed directly on storage, **never** routed
//! through the consensus log, while small **metadata** is transactional.
//!
//! - **Chunks (data plane)** — `put_chunk` / `put_chunks` / `get_chunk`
//!   take an engine handle and read/write `Partition::Blob` directly.
//!   Chunks are content-addressed and immutable: a given [`ChunkId`]
//!   always maps to the same bytes, so writes are idempotent and need
//!   no MVCC version, OCC conflict tracking, or Raft proposal. Payloads
//!   are large (multi-MB) and must never sit in a transaction's
//!   in-memory write buffer or be shipped as a single consensus entry.
//!   Cluster durability for chunks is the placement layer's job —
//!   replication / Reed-Solomon erasure coding under CRUSH failure-
//!   domain rules (see the erasure-coding and segments architecture),
//!   not the transaction commit path.
//! - **Blob refs (metadata plane)** — `put_blob_ref` / `get_blob_ref` /
//!   `delete_blob_ref` thread the active [`Transaction`]: the per-(node,
//!   prop) ref is small and commits atomically with the node write that
//!   produced it, and reads see it through the same MVCC snapshot as the
//!   node (time-travel consistent).
//!
//! Atomicity between a chunk and the ref pointing at it is **eventual**:
//! chunks are placed first, the ref commits second, and orphan chunks
//! (ref never committed, or ref later deleted) are reclaimed by the
//! Layer 3 reference-counting GC sweep — exactly the RADOS / MinIO model.
//!
//! [`Transaction`]: coordinode_storage::engine::transaction::Transaction

use coordinode_core::graph::blob::{encode_blob_key, encode_blobref_key, BlobRef, ChunkId};
use coordinode_core::graph::node::NodeId;
use coordinode_storage::engine::batch::WriteBatch;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};

pub use coordinode_core::graph::blob::INLINE_THRESHOLD;

/// Layer 4 blob store: content-addressed chunk CAS (data plane, direct
/// to the engine) + per-property blob references (metadata plane,
/// transactional). Hides partition layout and key encoders.
pub trait BlobStore {
    /// Persist a chunk by content hash directly to the engine
    /// (data plane). Idempotent: re-putting an existing `chunk_id`
    /// overwrites identical bytes.
    fn put_chunk(&self, engine: &StorageEngine, chunk_id: &ChunkId, data: &[u8])
        -> StoreResult<()>;

    /// Persist many chunks in one engine [`WriteBatch`] (data plane).
    /// Useful when a new blob ref pulls in multiple new chunks.
    fn put_chunks(&self, engine: &StorageEngine, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()>;

    /// Fetch a chunk by content hash (data plane). Returns `None` if not
    /// stored (either never written, or already swept by GC because no
    /// ref pointed at it).
    fn get_chunk(&self, engine: &StorageEngine, chunk_id: &ChunkId)
        -> StoreResult<Option<Vec<u8>>>;

    /// Persist the [`BlobRef`] for a (node, prop) pair (metadata plane).
    /// Overwrites any previous ref at the same key. Buffered on `txn`.
    fn put_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
        blob_ref: &BlobRef,
    ) -> StoreResult<()>;

    /// Fetch the [`BlobRef`] for a (node, prop) pair (metadata plane),
    /// through the transaction's MVCC snapshot.
    fn get_blob_ref(
        &self,
        txn: &Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<Option<BlobRef>>;

    /// Remove the [`BlobRef`] for a (node, prop) pair (metadata plane).
    /// Chunks remain in [`Partition::Blob`] — orphan chunks are reclaimed
    /// by the Layer 3 GC sweep. Buffered on `txn`.
    fn delete_blob_ref(
        &self,
        txn: &mut Transaction,
        node_id: NodeId,
        prop_id: u32,
    ) -> StoreResult<()>;
}

/// CE single-shard implementation of [`BlobStore`]. Stateless: chunk
/// (data-plane) ops take the engine handle directly, blob-ref
/// (metadata-plane) ops thread the transaction.
pub struct LocalBlobStore;

impl BlobStore for LocalBlobStore {
    fn put_chunk(
        &self,
        engine: &StorageEngine,
        chunk_id: &ChunkId,
        data: &[u8],
    ) -> StoreResult<()> {
        engine.put(Partition::Blob, &encode_blob_key(chunk_id), data)?;
        Ok(())
    }

    fn put_chunks(&self, engine: &StorageEngine, chunks: &[(ChunkId, Vec<u8>)]) -> StoreResult<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut batch = WriteBatch::new(engine);
        for (id, data) in chunks {
            batch.put(Partition::Blob, encode_blob_key(id), data.clone());
        }
        batch.commit()?;
        Ok(())
    }

    fn get_chunk(
        &self,
        engine: &StorageEngine,
        chunk_id: &ChunkId,
    ) -> StoreResult<Option<Vec<u8>>> {
        Ok(engine
            .get(Partition::Blob, &encode_blob_key(chunk_id))?
            .map(|b| b.to_vec()))
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
mod tests;
