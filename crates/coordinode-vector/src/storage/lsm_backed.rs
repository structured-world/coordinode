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
        // StorageEngine doesn't expose a multi_get yet (lsm-tree has it
        // but it's not surfaced through the engine wrapper). Iterate
        // with per-id `get` for now — acceptable at Phase 1.5 batch
        // sizes (~50 candidates per shard). Replace with a single
        // batched call when the engine exposes one (R-MULTI-GET-EXPOSE).
        let mut out = Vec::with_capacity(node_ids.len());
        for &id in node_ids {
            let key = encode_vec_f32_key(label_id, property_id, id);
            match self
                .engine
                .get(Partition::VectorF32, &key)
                .map_err(|e| Box::new(e) as VectorTierError)?
            {
                Some(bytes) => out.push(decode_f32_value(bytes.as_ref())),
                None => out.push(None),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::storage::VectorTierHandle;

    fn open_engine() -> (Arc<StorageEngine>, tempfile::TempDir) {
        use coordinode_storage::engine::config::{
            Durability, EndpointConfig, Media, StorageConfig, Tier,
        };
        let dir = tempfile::tempdir().expect("tempdir");
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        let engine = StorageEngine::open(&config).expect("open");
        (Arc::new(engine), dir)
    }

    #[test]
    fn f32_round_trip_through_lsm() {
        let (engine, _dir) = open_engine();
        let tier = LsmVectorTier::new(engine);
        let v = vec![1.0_f32, -0.5, 3.25, -42.0, 0.0];
        tier.put_f32(7, 13, 99, &v).expect("put_f32");
        let got = tier.multi_get_f32(7, 13, &[99, 100]).expect("multi_get");
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].as_deref(), Some(v.as_slice()));
        assert!(got[1].is_none(), "missing id must report None, not garbage");
    }

    #[test]
    fn handle_wires_through_to_lsm() {
        let (engine, _dir) = open_engine();
        let tier: Arc<dyn VectorTierStorage> = Arc::new(LsmVectorTier::new(engine));
        let h = VectorTierHandle::new(tier, 5, 9);
        h.put_f32(123, &[7.0, 8.0]).unwrap();
        assert_eq!(
            h.multi_get_f32(&[123]).unwrap()[0].as_deref(),
            Some(&[7.0_f32, 8.0][..])
        );
    }

    #[test]
    fn corrupted_f32_value_decodes_to_none() {
        // Direct put of a non-multiple-of-4 byte string under the
        // f32 key — decoder must report None, not crash.
        let (engine, _dir) = open_engine();
        let key = encode_vec_f32_key(1, 1, 1);
        engine
            .put(Partition::VectorF32, &key, &[0xFFu8, 0xFF, 0xFF])
            .unwrap();
        let tier = LsmVectorTier::new(engine);
        let got = tier.multi_get_f32(1, 1, &[1]).unwrap();
        assert!(got[0].is_none(), "non-mod-4 byte slice must decode to None");
    }
}
