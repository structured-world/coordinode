//! Vector quantization codecs.
//!
//! - [`Sq8Params`] / [`QuantizedVector`] — scalar quantization (u8 per dim, 4× compression).
//!   Used for the disk rerank pool where cross-shard comparability matters.
//! - [`rabitq`] — RaBitQ 1-bit-per-dim with popcount distance kernel. Primary in-RAM
//!   codec (per-shard rotation, ~30× compression, ~10× kernel speedup vs SQ8).
//! - [`popcount`] — XOR + popcount kernel with runtime SIMD dispatch shared by RaBitQ
//!   and any future binary codec.
//!
//! SQ8 maps each dimension independently using per-dimension min/max:
//!   quantize:   u8 = round((f32 - min) / (max - min) * 255)
//!   dequantize: f32 = u8 / 255 * (max - min) + min
//!
//! Both run in binary64, so the mapping holds at every scale of finite f32
//! values: a range of `1e-26` or of `6e38` quantizes as a range of `1` does.

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

        // A dimension every vector holds the same value in keeps an empty
        // range, which quantizes to code 0 and dequantizes to that value. A
        // range is never widened by an absolute amount: that drowns a range
        // far below it and vanishes beside one far above it.
        Some(Self { mins, maxs })
    }

    /// Calibrate SQ8 parameters from vectors in an existing HNSW index.
    /// Collects the in-memory f32 vectors of its nodes and delegates to
    /// `calibrate()`.
    pub fn calibrate_from_index(index: &mut crate::hnsw::HnswIndex) -> Option<Self> {
        Self::calibrate(&index.calibration_vectors())
    }

    /// Number of dimensions.
    pub fn dims(&self) -> usize {
        self.mins.len()
    }

    /// The range of dimension `i`, in binary64: the difference of two finite
    /// f32 values, which overflows f32 for opposite values past `f32::MAX / 2`.
    #[inline]
    fn range(&self, i: usize) -> f64 {
        f64::from(self.maxs[i]) - f64::from(self.mins[i])
    }

    /// Quantize a float32 vector to uint8.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        assert_eq!(vector.len(), self.dims(), "dimension mismatch");
        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let range = self.range(i);
                if range == 0.0 {
                    return 0;
                }
                let normalized = (f64::from(v) - f64::from(self.mins[i])) / range;
                // In [0, 255] after the clamp: the cast cannot truncate.
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
                // Between min and max, both finite f32: the narrowing stays
                // finite.
                (f64::from(q) / 255.0 * self.range(i) + f64::from(self.mins[i])) as f32
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
