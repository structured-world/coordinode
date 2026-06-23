//! Key encoding for the vector storage tier (ADR-033 revised).
//!
//! One partition holds the f32 source of truth per ADR-033:
//!
//! - [`Partition::VectorF32`][crate::engine::partition::Partition::VectorF32]
//!   stores the f32 source-of-truth bytes (`dim × 4` bytes per vector).
//!   In-RAM codecs (RaBitQ default, optional SQ8 / PolarQuant / PQ)
//!   regenerate from this on calibration. Phase 1.5 cross-shard rerank
//!   reads f32 directly here — no intermediate quantized disk tier
//!   (matches Qdrant / Weaviate / ES BBQ pattern).
//!
//! Key format:
//!
//! ```text
//! ┌────────────┬───────────┬──────────────┬───────────┐
//! │ prefix     │ label_id  │ property_id  │ node_id   │
//! │ 4 bytes    │ 4 bytes   │ 4 bytes      │ 8 bytes   │
//! │ "vec:"     │ BE u32    │ BE u32       │ BE u64    │
//! └────────────┴───────────┴──────────────┴───────────┘
//! ```
//!
//! Fixed-size encoding lets prefix scans target a specific
//! `(label, property)` index without parsing variable-width separators.
//! Big-endian integers preserve lexicographic order = numeric order for
//! `multi_get` batches across contiguous node ids.

const PREFIX_F32: &[u8; 4] = b"vec:";

/// Total key length: `prefix(4) + label_id(4) + property_id(4) + node_id(8)`.
pub const VECTOR_KEY_LEN: usize = 4 + 4 + 4 + 8;

/// Encode a key for [`Partition::VectorF32`].
pub fn encode_vec_f32_key(label_id: u32, property_id: u32, node_id: u64) -> [u8; VECTOR_KEY_LEN] {
    encode(PREFIX_F32, label_id, property_id, node_id)
}

/// Decode a key produced by `encode_vec_f32_key`. Returns
/// `(label_id, property_id, node_id)` or `None` if the byte slice has
/// the wrong length or an unknown prefix.
pub fn decode_vector_key(key: &[u8]) -> Option<(u32, u32, u64)> {
    if key.len() != VECTOR_KEY_LEN {
        return None;
    }
    let prefix = &key[0..4];
    if prefix != PREFIX_F32 {
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
mod tests;
