//! Vector quantization codecs.
//!
//! - [`Sq8Params`] / [`QuantizedVector`] — scalar quantization (u8 per dim, 4× compression).
//!   Used for the Phase 1.5 disk rerank pool where cross-shard comparability matters.
//! - [`rabitq`] — RaBitQ 1-bit-per-dim with popcount distance kernel. Primary in-RAM
//!   codec per ADR-032 (per-shard rotation, ~30× compression, ~10× kernel speedup vs SQ8).
//! - [`popcount`] — XOR + popcount kernel with runtime SIMD dispatch shared by RaBitQ
//!   and any future binary codec.
//!
//! SQ8 maps each dimension independently using per-dimension min/max:
//!   quantize:   u8 = round((f32 - min) / (max - min) * 255)
//!   dequantize: f32 = u8 / 255 * (max - min) + min

pub mod popcount;
pub mod rabitq;

use serde::{Deserialize, Serialize};

/// Per-dimension calibration parameters for SQ8.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Sq8Params {
    /// Minimum value per dimension.
    pub mins: Vec<f32>,
    /// Maximum value per dimension.
    pub maxs: Vec<f32>,
}

impl Sq8Params {
    /// Calibrate SQ8 parameters from a set of vectors.
    ///
    /// Computes per-dimension min/max from the provided vectors.
    /// All vectors must have the same dimensionality.
    pub fn calibrate(vectors: &[&[f32]]) -> Option<Self> {
        if vectors.is_empty() {
            return None;
        }

        let dims = vectors[0].len();
        if dims == 0 {
            return None;
        }

        let mut mins = vec![f32::INFINITY; dims];
        let mut maxs = vec![f32::NEG_INFINITY; dims];

        for vec in vectors {
            if vec.len() != dims {
                return None; // Dimension mismatch
            }
            for (i, &v) in vec.iter().enumerate() {
                if v < mins[i] {
                    mins[i] = v;
                }
                if v > maxs[i] {
                    maxs[i] = v;
                }
            }
        }

        // Prevent division by zero: if min == max, expand range slightly
        for i in 0..dims {
            if (maxs[i] - mins[i]).abs() < f32::EPSILON {
                mins[i] -= 0.5;
                maxs[i] += 0.5;
            }
        }

        Some(Self { mins, maxs })
    }

    /// Calibrate SQ8 parameters from vectors in an existing HNSW index.
    /// Collects all in-memory f32 vectors and delegates to `calibrate()`.
    pub fn calibrate_from_index(index: &crate::hnsw::HnswIndex) -> Option<Self> {
        let refs: Vec<&[f32]> = (0..index.len())
            .filter_map(|i| index.get_vector(i))
            .collect();
        Self::calibrate(&refs)
    }

    /// Number of dimensions.
    pub fn dims(&self) -> usize {
        self.mins.len()
    }

    /// Quantize a float32 vector to uint8.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dims(), "dimension mismatch");
        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let range = self.maxs[i] - self.mins[i];
                let normalized = (v - self.mins[i]) / range;
                (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect()
    }

    /// Dequantize a uint8 vector back to float32 (approximate).
    pub fn dequantize(&self, quantized: &[u8]) -> Vec<f32> {
        assert_eq!(quantized.len(), self.dims(), "dimension mismatch");
        quantized
            .iter()
            .enumerate()
            .map(|(i, &q)| {
                let range = self.maxs[i] - self.mins[i];
                (q as f32 / 255.0) * range + self.mins[i]
            })
            .collect()
    }
}

/// A quantized vector with its original dimensionality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QuantizedVector {
    /// SQ8-quantized values.
    pub data: Vec<u8>,
}

impl QuantizedVector {
    /// Memory size in bytes (4x smaller than f32).
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }

    /// Original f32 size would have been.
    pub fn original_size_bytes(&self) -> usize {
        self.data.len() * 4
    }
}

/// Maximum supported vector dimensions.
pub const MAX_DIMENSIONS: u32 = 65_536;

/// Validate vector dimensions against schema.
pub fn validate_dimensions(vector: &[f32], expected_dims: u32) -> Result<(), String> {
    if vector.len() != expected_dims as usize {
        return Err(format!(
            "vector has {} dimensions, expected {}",
            vector.len(),
            expected_dims
        ));
    }
    if expected_dims > MAX_DIMENSIONS {
        return Err(format!(
            "dimensions {} exceeds maximum {}",
            expected_dims, MAX_DIMENSIONS
        ));
    }
    Ok(())
}

/// Check if a vector should use KV separation (>256 dims = >1KB at f32).
pub fn should_kv_separate(dims: u32) -> bool {
    dims > 256
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
