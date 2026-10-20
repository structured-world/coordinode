//! BlobStore: content-addressed chunks and the metadata naming them.
//!
//! Interim blob payload path: chunks and their metadata are written into the
//! store's own trees and made durable on this member before the call that
//! wrote them returns, which is a single-member (`w:1`) guarantee. Every
//! writer of blob payload goes through here, so the storage substrate's
//! extent/stream write, with payload journaled on the members its write
//! concern names, replaces this module and its callers in one place.

use coordinode_core::graph::blob::{ChunkId, encode_blob_key};

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Write the chunks `chunks` lists that the store does not hold yet, and make
/// every one of them durable before returning: a chunk is never published by
/// a reference before its bytes would survive a power cut. Returns how many
/// were already held.
///
/// # Errors
///
/// A read, write or the flush that makes them durable failed; nothing is
/// promised then.
pub fn put_chunks_durably(
    engine: &StorageEngine,
    chunks: &[(ChunkId, Vec<u8>)],
) -> StorageResult<usize> {
    let mut held = 0;
    for (chunk_id, data) in chunks {
        let key = encode_blob_key(chunk_id);
        if engine.get(Partition::Blob, &key)?.is_some() {
            held += 1;
        } else {
            engine.put(Partition::Blob, &key, data)?;
        }
    }
    // Interim blob payload path: the substrate's durable sealing replaces
    // this flush.
    engine.persist_partition(Partition::Blob)?;
    Ok(held)
}

/// Write blob metadata `value` at `key` and make it durable before
/// returning. Call it only once the chunks it names are durable.
///
/// # Errors
///
/// The write or the flush that makes it durable failed.
pub fn put_meta_durably(engine: &StorageEngine, key: &[u8], value: &[u8]) -> StorageResult<()> {
    engine.put(Partition::BlobRef, key, value)?;
    // Interim blob payload path: catalog publication replaces this flush.
    engine.persist_partition(Partition::BlobRef)
}

/// Remove blob metadata at `key` and make the removal durable before
/// returning, so a deleted blob does not come back after a power cut. The
/// chunks stay: other blobs may share them.
///
/// # Errors
///
/// The delete or the flush that makes it durable failed.
pub fn delete_meta_durably(engine: &StorageEngine, key: &[u8]) -> StorageResult<()> {
    engine.delete(Partition::BlobRef, key)?;
    // Interim blob payload path: catalog retirement replaces this flush.
    engine.persist_partition(Partition::BlobRef)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
