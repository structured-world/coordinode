//! Key encoding for the vector storage tiers (ADR-033 revised).
//!
//! Two partitions back the per-ADR-033 truth + rerank model:
//!
//! - [`Partition::VectorF32`][crate::engine::partition::Partition::VectorF32]
//!   stores the f32 source-of-truth bytes (`dim × 4` bytes per vector). Every
//!   derived layer regenerates from here on calibration or codec migration.
//!
//! - [`Partition::VectorRerank`][crate::engine::partition::Partition::VectorRerank]
//!   stores the dimension-comparable quantized codec used for Phase 1.5
//!   cross-shard rerank. Default codec is SQ8 (1 byte / dim); pluggable per
//!   `disk_rerank_codec` schema knob.
//!
//! Key format is shared across both partitions:
//!
//! ```text
//! ┌────────────┬───────────┬──────────────┬───────────┐
//! │ prefix     │ label_id  │ property_id  │ node_id   │
//! │ 4 bytes    │ 4 bytes   │ 4 bytes      │ 8 bytes   │
//! │ "vec:" /   │ BE u32    │ BE u32       │ BE u64    │
//! │ "vrn:"     │           │              │           │
//! └────────────┴───────────┴──────────────┴───────────┘
//! ```
//!
//! Fixed-size encoding lets prefix scans target a specific
//! `(label, property)` index without parsing variable-width separators.
//! Big-endian integers preserve lexicographic order = numeric order for
//! `multi_get` batches across contiguous node ids.

const PREFIX_F32: &[u8; 4] = b"vec:";
const PREFIX_RERANK: &[u8; 4] = b"vrn:";

/// Total key length: `prefix(4) + label_id(4) + property_id(4) + node_id(8)`.
pub const VECTOR_KEY_LEN: usize = 4 + 4 + 4 + 8;

/// Encode a key for [`Partition::VectorF32`].
pub fn encode_vec_f32_key(label_id: u32, property_id: u32, node_id: u64) -> [u8; VECTOR_KEY_LEN] {
    encode(PREFIX_F32, label_id, property_id, node_id)
}

/// Encode a key for [`Partition::VectorRerank`].
pub fn encode_vec_rerank_key(
    label_id: u32,
    property_id: u32,
    node_id: u64,
) -> [u8; VECTOR_KEY_LEN] {
    encode(PREFIX_RERANK, label_id, property_id, node_id)
}

/// Decode a key produced by either `encode_vec_f32_key` or
/// `encode_vec_rerank_key`. Returns `(label_id, property_id, node_id)` or
/// `None` if the byte slice has the wrong length or an unknown prefix.
pub fn decode_vector_key(key: &[u8]) -> Option<(u32, u32, u64)> {
    if key.len() != VECTOR_KEY_LEN {
        return None;
    }
    let prefix = &key[0..4];
    if prefix != PREFIX_F32 && prefix != PREFIX_RERANK {
        return None;
    }
    let label_id = u32::from_be_bytes(key[4..8].try_into().ok()?);
    let property_id = u32::from_be_bytes(key[8..12].try_into().ok()?);
    let node_id = u64::from_be_bytes(key[12..20].try_into().ok()?);
    Some((label_id, property_id, node_id))
}

/// Prefix used to range-scan every vector of a given `(label, property)`.
/// Inclusive lower bound; the matching upper bound is the same prefix with
/// the trailing node-id range exhausted.
pub fn vec_f32_index_prefix(label_id: u32, property_id: u32) -> [u8; 12] {
    encode_prefix(PREFIX_F32, label_id, property_id)
}

/// Prefix used to range-scan every quantized code of a given
/// `(label, property)` — symmetric to `vec_f32_index_prefix`.
pub fn vec_rerank_index_prefix(label_id: u32, property_id: u32) -> [u8; 12] {
    encode_prefix(PREFIX_RERANK, label_id, property_id)
}

/// Encode f32 vector to little-endian byte slice for the truth tier value.
/// The on-wire representation is `dim × 4` bytes, identical to what
/// `bytemuck::cast_slice::<f32, u8>` would produce but written explicitly so
/// future SIMD readers can rely on the exact layout.
pub fn encode_f32_value(vector: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vector.len() * 4);
    for v in vector {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode a `dim × 4`-byte truth-tier value back to f32. Returns `None` when
/// the byte length isn't a multiple of 4 (corruption guard).
pub fn decode_f32_value(bytes: &[u8]) -> Option<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(out)
}

// ── Internal helpers ──────────────────────────────────────────────────

fn encode(prefix: &[u8; 4], label_id: u32, property_id: u32, node_id: u64) -> [u8; VECTOR_KEY_LEN] {
    let mut key = [0u8; VECTOR_KEY_LEN];
    key[0..4].copy_from_slice(prefix);
    key[4..8].copy_from_slice(&label_id.to_be_bytes());
    key[8..12].copy_from_slice(&property_id.to_be_bytes());
    key[12..20].copy_from_slice(&node_id.to_be_bytes());
    key
}

fn encode_prefix(prefix: &[u8; 4], label_id: u32, property_id: u32) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(prefix);
    out[4..8].copy_from_slice(&label_id.to_be_bytes());
    out[8..12].copy_from_slice(&property_id.to_be_bytes());
    out
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn f32_key_round_trip() {
        let key = encode_vec_f32_key(7, 13, 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(key.len(), VECTOR_KEY_LEN);
        let (label, prop, node) = decode_vector_key(&key).expect("decode");
        assert_eq!(label, 7);
        assert_eq!(prop, 13);
        assert_eq!(node, 0xDEAD_BEEF_CAFE_F00D);
    }

    #[test]
    fn rerank_key_round_trip() {
        let key = encode_vec_rerank_key(1, 2, 3);
        let (label, prop, node) = decode_vector_key(&key).expect("decode");
        assert_eq!((label, prop, node), (1, 2, 3));
    }

    #[test]
    fn f32_and_rerank_keys_dont_alias() {
        let a = encode_vec_f32_key(1, 2, 3);
        let b = encode_vec_rerank_key(1, 2, 3);
        assert_ne!(a, b);
    }

    #[test]
    fn keys_sort_lexically_by_node_id() {
        // Big-endian node-id encoding must keep prefix scans contiguous AND
        // monotonic in node-id order — important for multi_get batches that
        // expect ranged keys to hit a single SST block.
        let k1 = encode_vec_f32_key(1, 1, 1);
        let k2 = encode_vec_f32_key(1, 1, 1_000_000);
        let k3 = encode_vec_f32_key(1, 1, u64::MAX);
        assert!(k1 < k2);
        assert!(k2 < k3);
    }

    #[test]
    fn prefix_is_proper_prefix_of_full_key() {
        let prefix = vec_f32_index_prefix(42, 17);
        let key = encode_vec_f32_key(42, 17, 99);
        assert_eq!(&key[..prefix.len()], &prefix[..]);
    }

    #[test]
    fn decode_rejects_wrong_length() {
        assert!(decode_vector_key(&[0u8; 0]).is_none());
        assert!(decode_vector_key(&[0u8; VECTOR_KEY_LEN - 1]).is_none());
        assert!(decode_vector_key(&[0u8; VECTOR_KEY_LEN + 1]).is_none());
    }

    #[test]
    fn decode_rejects_unknown_prefix() {
        let mut key = encode_vec_f32_key(1, 2, 3);
        key[0..4].copy_from_slice(b"nope");
        assert!(decode_vector_key(&key).is_none());
    }

    #[test]
    fn f32_value_round_trip() {
        let v = vec![0.0_f32, 1.5, -2.25, f32::INFINITY, f32::NEG_INFINITY];
        let bytes = encode_f32_value(&v);
        assert_eq!(bytes.len(), v.len() * 4);
        let back = decode_f32_value(&bytes).expect("decode");
        assert_eq!(back, v);
    }

    #[test]
    fn f32_value_handles_nan_byte_exact() {
        // NaN comparisons aren't equal, but the byte representation must
        // round-trip exactly. Used by recall regression tests that re-encode
        // truth-tier vectors after a codec migration.
        let v = vec![f32::NAN, 0.0];
        let bytes = encode_f32_value(&v);
        let back = decode_f32_value(&bytes).expect("decode");
        assert!(back[0].is_nan());
        assert_eq!(back[1], 0.0);
        // Byte-exact: re-encode of decoded value should match original.
        assert_eq!(encode_f32_value(&back), bytes);
    }

    #[test]
    fn decode_rejects_non_multiple_of_four() {
        assert!(decode_f32_value(&[0u8; 3]).is_none());
        assert!(decode_f32_value(&[0u8; 5]).is_none());
    }
}
