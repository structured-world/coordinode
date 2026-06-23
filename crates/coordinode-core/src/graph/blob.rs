//! BlobStore: content-addressed chunk storage for large values.
//!
//! Large values (embeddings, documents, images) are split into 256KB chunks,
//! each identified by its SHA-256 hash. Content-addressing enables automatic
//! deduplication — identical chunks are stored once regardless of references.
//!
//! ## Storage layout
//!
//! ```text
//! blob:<sha256_hex>           → chunk data (up to 256KB)
//! blobref:<node_id>:<prop_id> → ordered list of chunk hashes
//! ```
//!
//! ## Inline threshold
//!
//! Values smaller than 4KB are stored directly as node properties.
//! Values >= 4KB are chunked and stored via BlobStore.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::node::NodeId;

/// Default chunk size: 256KB.
pub const DEFAULT_CHUNK_SIZE: usize = 256 * 1024;

/// Maximum chunk size: 16MB.
pub const MAX_CHUNK_SIZE: usize = 16 * 1024 * 1024;

/// Inline threshold: values below this are stored as node properties.
pub const INLINE_THRESHOLD: usize = 4 * 1024;

/// A SHA-256 content hash identifying a blob chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChunkId([u8; 32]);

impl ChunkId {
    /// Compute the SHA-256 hash of the given data.
    pub fn from_data(data: &[u8]) -> Self {
        let hash = Sha256::digest(data);
        let mut id = [0u8; 32];
        id.copy_from_slice(&hash);
        Self(id)
    }

    /// Create from raw 32-byte hash.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw 32 bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Format as lowercase hex string (64 chars).
    pub fn to_hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Parse from hex string.
    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut bytes = [0u8; 32];
        for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
            let s = std::str::from_utf8(chunk).ok()?;
            bytes[i] = u8::from_str_radix(s, 16).ok()?;
        }
        Some(Self(bytes))
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for b in &self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

/// A reference from a node property to a sequence of blob chunks.
///
/// The chunks, when concatenated in order, reconstruct the original value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BlobRef {
    /// Ordered list of chunk IDs.
    pub chunks: Vec<ChunkId>,
    /// Total size of the original value in bytes.
    pub total_size: u64,
}

impl BlobRef {
    /// Create a new blob reference.
    pub fn new(chunks: Vec<ChunkId>, total_size: u64) -> Self {
        Self { chunks, total_size }
    }

    /// Number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Serialize to MessagePack.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Deserialize from MessagePack.
    pub fn from_msgpack(data: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(data)
    }
}

// -- Key encoding --

/// Encode a blob chunk key: `blob:<sha256_hex>`.
pub fn encode_blob_key(chunk_id: &ChunkId) -> Vec<u8> {
    let mut key = Vec::with_capacity(5 + 64);
    key.extend_from_slice(b"blob:");
    key.extend_from_slice(chunk_id.to_hex().as_bytes());
    key
}

/// Encode a blob reference key: `blobref:<node_id BE>:<prop_id u32 BE>`.
pub fn encode_blobref_key(node_id: NodeId, prop_id: u32) -> Vec<u8> {
    let mut key = Vec::with_capacity(8 + 8 + 1 + 4);
    key.extend_from_slice(b"blobref:");
    key.extend_from_slice(&node_id.as_raw().to_be_bytes());
    key.push(b':');
    key.extend_from_slice(&prop_id.to_be_bytes());
    key
}

// -- Chunking --

/// Split data into content-addressed chunks.
///
/// Returns a list of `(ChunkId, chunk_data)` pairs. Identical data
/// produces identical ChunkIds, enabling deduplication at the storage layer.
pub fn chunk_data(data: &[u8], chunk_size: usize) -> Vec<(ChunkId, Vec<u8>)> {
    assert!(chunk_size > 0 && chunk_size <= MAX_CHUNK_SIZE);

    if data.is_empty() {
        return Vec::new();
    }

    data.chunks(chunk_size)
        .map(|chunk| {
            let id = ChunkId::from_data(chunk);
            (id, chunk.to_vec())
        })
        .collect()
}

/// Check if a value should be stored inline (< threshold) or via BlobStore.
pub fn should_inline(data: &[u8]) -> bool {
    data.len() < INLINE_THRESHOLD
}

/// Create a `BlobRef` from data by chunking and hashing.
///
/// Returns the `BlobRef` and the individual chunks for storage.
pub fn create_blob(data: &[u8]) -> (BlobRef, Vec<(ChunkId, Vec<u8>)>) {
    create_blob_with_chunk_size(data, DEFAULT_CHUNK_SIZE)
}

/// Create a `BlobRef` with custom chunk size.
pub fn create_blob_with_chunk_size(
    data: &[u8],
    chunk_size: usize,
) -> (BlobRef, Vec<(ChunkId, Vec<u8>)>) {
    let chunks = chunk_data(data, chunk_size);
    let chunk_ids: Vec<ChunkId> = chunks.iter().map(|(id, _)| *id).collect();
    let blob_ref = BlobRef::new(chunk_ids, data.len() as u64);
    (blob_ref, chunks)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
