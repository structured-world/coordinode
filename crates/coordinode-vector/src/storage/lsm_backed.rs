//! LSM-backed implementation of [`VectorTierStorage`].
//!
//! Wraps an [`Arc<StorageEngine>`] and routes truth-tier traffic to
//! the dedicated [`Partition::VectorF32`] partition per ADR-033.
//! Keys built by the
//! [`vector_keys`][coordinode_storage::engine::vector_keys] module so
//! the layout stays consistent with what the storage layer's own
//! scrub / migration tooling reads.
//!
//! This is the production binding used by the embed / query layer.
//! Tests in `coordinode-vector` still drive a pure in-RAM mock; this
//! module's own tests open a real tempdir-backed engine and exercise
//! a round-trip.

use std::sync::Arc;

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::vector_keys::{
    decode_f32_value, encode_f32_value, encode_vec_f32_key,
};

use super::{VectorTierError, VectorTierStorage};

/// LSM-backed vector tier. Owns a shared handle to the storage engine
/// so every (label, property)-scoped `VectorTierHandle` can route here.
pub struct LsmVectorTier {
    engine: Arc<StorageEngine>,
}

impl LsmVectorTier {
    pub fn new(engine: Arc<StorageEngine>) -> Self {
        Self { engine }
    }
}

impl VectorTierStorage for LsmVectorTier {
    fn put_f32(
        &self,
        label_id: u32,
        property_id: u32,
        node_id: u64,
        vector: &[f32],
    ) -> Result<(), VectorTierError> {
        let key = encode_vec_f32_key(label_id, property_id, node_id);
        let value = encode_f32_value(vector);
        self.engine
            .put(Partition::VectorF32, &key, &value)
            .map_err(|e| Box::new(e) as VectorTierError)?;
        Ok(())
    }

    fn multi_get_f32(
        &self,
        label_id: u32,
        property_id: u32,
        node_ids: &[u64],
    ) -> Result<Vec<Option<Vec<f32>>>, VectorTierError> {
        // Resolve the whole candidate set through one batched engine
        // `multi_get`: the version snapshot is pinned once and the bloom
        // filter + SST traversal is batched across all keys, instead of
        // re-pinning the snapshot and descending the tree per id. Values
        // come back in input order, one slot per node id.
        let keys: Vec<_> = node_ids
            .iter()
            .map(|&id| encode_vec_f32_key(label_id, property_id, id))
            .collect();
        let key_refs: Vec<&[u8]> = keys.iter().map(|k| k.as_ref()).collect();
        let values = self
            .engine
            .multi_get(Partition::VectorF32, &key_refs)
            .map_err(|e| Box::new(e) as VectorTierError)?;
        Ok(values
            .into_iter()
            .map(|opt| opt.and_then(|bytes| decode_f32_value(bytes.as_ref())))
            .collect())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
