//! Storage tier trait for the vector indexes (ADR-033 revised).
//!
//! Single persistent tier holding f32 originals:
//!
//! - **Truth tier** (`Partition::VectorF32` on the storage side) — every
//!   inserted vector's f32 bytes. Lets the index regenerate in-RAM
//!   codecs (RaBitQ default, optional SQ8 / PolarQuant / PQ) on
//!   calibration without re-ingest. Phase 1.5 cross-shard rerank
//!   fetches f32 directly here — no intermediate quantized disk tier
//!   (matches Qdrant / Weaviate / ES BBQ pattern).
//!
//! Addressed by `(label_id, property_id, node_id)`. The
//! `coordinode-storage` crate implements this trait against its
//! `StorageEngine`; tests use the in-memory mock below.
//!
//! Why a trait, not a direct `StorageEngine` dependency: keeps
//! `coordinode-vector` from depending on `coordinode-storage`. Per the
//! crate-isolation rule in CLAUDE.md, cross-crate communication goes
//! through traits, never concrete types.
//
// no-std: `Send + Sync` super-bounds use `core` traits; `Arc<dyn _>` is
//         `alloc`-clean. The trait itself does not pull `std` types.

use alloc::sync::Arc;
use alloc::vec::Vec;

extern crate alloc;

pub mod lsm_backed;

/// Errors that the storage backend can report through the vector tier
/// API. Concrete variants are owned by the backend — the trait sees an
/// opaque `Box<dyn Error>` so different backends (LSM, RAM, future
/// remote tier) plug in cleanly.
pub type VectorTierError = alloc::boxed::Box<dyn core::error::Error + Send + Sync>;

/// Persistent backing for the f32 truth tier and the quantized rerank
/// tier of one vector index. Implementations are expected to be
/// thread-safe and idempotent under retries (writes use `put` overwrite
/// semantics, not append).
///
/// Every method takes `(label_id, property_id, node_id)`: the two
/// schema IDs identify the (label, property) the index covers (the
/// HNSW index is per-(label, property)), the node_id identifies the
/// row whose vector is being read or written.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not implement `VectorTierStorage`",
    label = "this type can't back the HNSW truth + rerank tiers",
    note = "the canonical impl is in `coordinode-storage`; tests can use \
            `coordinode_vector::storage::InMemoryVectorTier`."
)]
pub trait VectorTierStorage: Send + Sync {
    /// Write a full-precision f32 vector to the truth tier. Overwrites
    /// any prior value at the same key.
    fn put_f32(
        &self,
        label_id: u32,
        property_id: u32,
        node_id: u64,
        vector: &[f32],
    ) -> Result<(), VectorTierError>;

    /// Batched fetch of f32 vectors. Returns one slot per requested
    /// `node_id`, in the same order; `None` for nodes whose f32 truth
    /// tier doesn't have the key (e.g. between Raft commit and HNSW
    /// worker apply). Used by Phase 1.5 cross-shard rerank,
    /// application-side rerank with custom metrics, and in-RAM codec
    /// (re)calibration.
    fn multi_get_f32(
        &self,
        label_id: u32,
        property_id: u32,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError>;
}

/// Pointer to a vector tier backend an HNSW index is bound to. The
/// optional `Arc` carries the tier itself; the two ids identify which
/// `(label, property)` keys to use on every call.
#[derive(Clone)]
pub struct VectorTierHandle {
    pub backend: Arc<dyn VectorTierStorage>,
    pub label_id: u32,
    pub property_id: u32,
}

impl VectorTierHandle {
    pub fn new(backend: Arc<dyn VectorTierStorage>, label_id: u32, property_id: u32) -> Self {
        Self {
            backend,
            label_id,
            property_id,
        }
    }

    pub fn put_f32(&self, node_id: u64, vector: &[f32]) -> Result<(), VectorTierError> {
        self.backend
            .put_f32(self.label_id, self.property_id, node_id, vector)
    }

    pub fn multi_get_f32(
        &self,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError> {
        self.backend
            .multi_get_f32(self.label_id, self.property_id, node_ids)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
