//! Bucket key — the identifier the catalog uses to look up an open
//! bucket. Mirrors `arch/core/timeseries.md` §BucketCatalog:
//!
//! ```text
//! struct BucketKey { label_id: u16, meta_hash: u64 }
//! ```
//!
//! Two distinct meta-field values must hash to distinct
//! `meta_hash`es with overwhelming probability — collisions cost a
//! flush + restart (the bucket boundary is crossed prematurely) but
//! never lose data or misroute reads. The hash is computed once at
//! catalog ingress; the source [`rmpv::Value`] is **not** stored.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Catalog lookup key. `label_id` distinguishes time-series labels
/// (e.g. `SensorReading` vs `EngineMetric`); `meta_hash` is a stable
/// hash of the `metaField` value (e.g. sensor_id or device_id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BucketKey {
    /// Identifier of the time-series label.
    pub label_id: u16,
    /// Stable hash of the measurement's meta-field value.
    pub meta_hash: u64,
}

impl BucketKey {
    /// Build a key from `label_id` and the meta-field value. Uses
    /// [`std::collections::hash_map::DefaultHasher`] (SipHash 1-3),
    /// matching the engine's default keyed-hash family.
    pub fn from_meta(label_id: u16, meta: &rmpv::Value) -> Self {
        let mut hasher = DefaultHasher::new();
        hash_rmpv(meta, &mut hasher);
        Self {
            label_id,
            meta_hash: hasher.finish(),
        }
    }

    /// Stripe index this key routes to. The catalog uses 32 stripes;
    /// a mod-32 of the meta_hash is a single mask.
    pub fn stripe_idx(&self) -> usize {
        (self.meta_hash as usize) & (crate::config::STRIPE_COUNT - 1)
    }
}

/// Deep hash of an `rmpv::Value` tree. `rmpv::Value` does NOT impl
/// `Hash` so we recurse manually. Order-preserving on Maps and
/// Arrays so two equal MessagePack trees produce the same hash.
fn hash_rmpv<H: Hasher>(value: &rmpv::Value, h: &mut H) {
    match value {
        rmpv::Value::Nil => 0u8.hash(h),
        rmpv::Value::Boolean(b) => {
            1u8.hash(h);
            b.hash(h);
        }
        rmpv::Value::Integer(i) => {
            2u8.hash(h);
            if let Some(v) = i.as_i64() {
                v.hash(h);
            } else if let Some(v) = i.as_u64() {
                v.hash(h);
            }
        }
        rmpv::Value::F32(f) => {
            3u8.hash(h);
            f.to_bits().hash(h);
        }
        rmpv::Value::F64(f) => {
            4u8.hash(h);
            f.to_bits().hash(h);
        }
        rmpv::Value::String(s) => {
            5u8.hash(h);
            s.as_str().unwrap_or("").hash(h);
        }
        rmpv::Value::Binary(b) => {
            6u8.hash(h);
            b.hash(h);
        }
        rmpv::Value::Array(a) => {
            7u8.hash(h);
            a.len().hash(h);
            for elem in a {
                hash_rmpv(elem, h);
            }
        }
        rmpv::Value::Map(m) => {
            8u8.hash(h);
            m.len().hash(h);
            for (k, v) in m {
                hash_rmpv(k, h);
                hash_rmpv(v, h);
            }
        }
        rmpv::Value::Ext(ty, payload) => {
            9u8.hash(h);
            ty.hash(h);
            payload.hash(h);
        }
    }
}

#[cfg(test)]
mod tests;
