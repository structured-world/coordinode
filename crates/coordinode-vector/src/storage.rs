//! Storage tier traits for the vector indexes (ADR-033 revised).
//!
//! The HNSW index talks to two persistent tiers via this trait:
//!
//! - **Truth tier** (`Partition::VectorF32` on the storage side) — every
//!   inserted vector's f32 bytes. Lets the index regenerate quantized
//!   layers on calibration / codec migration without re-ingest.
//!
//! - **Rerank quantized tier** (`Partition::VectorRerank` on the storage
//!   side) — codec-specific bytes used for Phase 1.5 cross-shard rerank
//!   (default SQ8, pluggable per the schema-level `disk_rerank_codec`
//!   knob).
//!
//! Both tiers are addressed by `(label_id, property_id, node_id)`. The
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

    /// Write a quantized code to the rerank tier. Codec format is
    /// implicit per the index's active `disk_rerank_codec` schema
    /// setting; the trait passes raw bytes through.
    fn put_quantized(
        &self,
        label_id: u32,
        property_id: u32,
        node_id: u64,
        code: &[u8],
    ) -> Result<(), VectorTierError>;

    /// Batched fetch of f32 vectors. Returns one slot per requested
    /// `node_id`, in the same order; `None` for nodes whose f32 truth
    /// tier was opted out (`f32_storage = "off"`) or whose key isn't in
    /// storage. Used by application-side rerank with custom metrics
    /// and by codec migration.
    fn multi_get_f32(
        &self,
        label_id: u32,
        property_id: u32,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError>;

    /// Batched fetch of quantized codes. Returns one slot per requested
    /// `node_id`, in the same order; `None` for nodes that haven't been
    /// encoded yet (window between insert and calibration). The Phase
    /// 1.5 rerank coordinator drives this method.
    fn multi_get_quantized(
        &self,
        label_id: u32,
        property_id: u32,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<u8>>>, VectorTierError>;
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

    pub fn put_quantized(&self, node_id: u64, code: &[u8]) -> Result<(), VectorTierError> {
        self.backend
            .put_quantized(self.label_id, self.property_id, node_id, code)
    }

    pub fn multi_get_f32(
        &self,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError> {
        self.backend
            .multi_get_f32(self.label_id, self.property_id, node_ids)
    }

    pub fn multi_get_quantized(
        &self,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<u8>>>, VectorTierError> {
        self.backend
            .multi_get_quantized(self.label_id, self.property_id, node_ids)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Test backend: stores everything in RAM under (label, property, node) keys.
    pub struct InMemoryVectorTier {
        f32_store: Mutex<HashMap<(u32, u32, u64), Vec<f32>>>,
        q_store: Mutex<HashMap<(u32, u32, u64), Vec<u8>>>,
    }

    impl InMemoryVectorTier {
        pub fn new() -> Self {
            Self {
                f32_store: Mutex::new(HashMap::new()),
                q_store: Mutex::new(HashMap::new()),
            }
        }
    }

    impl VectorTierStorage for InMemoryVectorTier {
        fn put_f32(&self, l: u32, p: u32, n: u64, v: &[f32]) -> Result<(), VectorTierError> {
            self.f32_store
                .lock()
                .map_err(|e| format!("poisoned: {e}"))?
                .insert((l, p, n), v.to_vec());
            Ok(())
        }

        fn put_quantized(&self, l: u32, p: u32, n: u64, c: &[u8]) -> Result<(), VectorTierError> {
            self.q_store
                .lock()
                .map_err(|e| format!("poisoned: {e}"))?
                .insert((l, p, n), c.to_vec());
            Ok(())
        }

        fn multi_get_f32(
            &self,
            l: u32,
            p: u32,
            ids: &[u64],
        ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError> {
            let g = self
                .f32_store
                .lock()
                .map_err(|e| format!("poisoned: {e}"))?;
            Ok(ids.iter().map(|&n| g.get(&(l, p, n)).cloned()).collect())
        }

        fn multi_get_quantized(
            &self,
            l: u32,
            p: u32,
            ids: &[u64],
        ) -> Result<Vec<Option<Vec<u8>>>, VectorTierError> {
            let g = self.q_store.lock().map_err(|e| format!("poisoned: {e}"))?;
            Ok(ids.iter().map(|&n| g.get(&(l, p, n)).cloned()).collect())
        }
    }

    #[test]
    fn handle_routes_to_backend() {
        let backend = Arc::new(InMemoryVectorTier::new());
        let h = VectorTierHandle::new(backend.clone(), 7, 13);
        h.put_f32(99, &[1.0, 2.0, 3.0]).unwrap();
        h.put_quantized(99, &[1u8, 2, 3, 4]).unwrap();

        let got_f32 = h.multi_get_f32(&[99, 100]).unwrap();
        assert_eq!(got_f32[0].as_deref(), Some(&[1.0, 2.0, 3.0][..]));
        assert!(got_f32[1].is_none());

        let got_q = h.multi_get_quantized(&[99, 100]).unwrap();
        assert_eq!(got_q[0].as_deref(), Some(&[1u8, 2, 3, 4][..]));
        assert!(got_q[1].is_none());
    }

    #[test]
    fn distinct_label_property_pairs_dont_collide() {
        let backend = Arc::new(InMemoryVectorTier::new());
        let h1 = VectorTierHandle::new(backend.clone(), 1, 1);
        let h2 = VectorTierHandle::new(backend.clone(), 1, 2);
        let h3 = VectorTierHandle::new(backend.clone(), 2, 1);

        h1.put_f32(42, &[1.0]).unwrap();
        h2.put_f32(42, &[2.0]).unwrap();
        h3.put_f32(42, &[3.0]).unwrap();

        assert_eq!(
            h1.multi_get_f32(&[42]).unwrap()[0].as_deref(),
            Some(&[1.0][..])
        );
        assert_eq!(
            h2.multi_get_f32(&[42]).unwrap()[0].as_deref(),
            Some(&[2.0][..])
        );
        assert_eq!(
            h3.multi_get_f32(&[42]).unwrap()[0].as_deref(),
            Some(&[3.0][..])
        );
    }
}
