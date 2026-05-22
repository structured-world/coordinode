//! Scalar quantization (SQ8): compress float32 vectors to uint8.
//!
//! SQ8 maps each dimension independently using per-dimension min/max:
//!   quantize:   u8 = round((f32 - min) / (max - min) * 255)
//!   dequantize: f32 = u8 / 255 * (max - min) + min
//!
//! Provides 4x memory reduction with typically <2% recall loss.
//! Calibration (min/max) must be computed from actual data, not assumed.

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
mod tests {
    use super::*;

    #[test]
    fn calibrate_from_vectors() {
        let v1 = vec![0.0, 1.0, -1.0];
        let v2 = vec![1.0, 0.0, 1.0];
        let v3 = vec![0.5, 0.5, 0.0];

        let params = Sq8Params::calibrate(&[&v1, &v2, &v3]).expect("calibrate");
        assert_eq!(params.dims(), 3);
        assert_eq!(params.mins[0], 0.0);
        assert_eq!(params.maxs[0], 1.0);
        assert_eq!(params.mins[2], -1.0);
        assert_eq!(params.maxs[2], 1.0);
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                vec![
                    (i as f32) / 100.0,
                    ((i * 3) as f32) / 100.0 - 1.0,
                    (i as f32).sin(),
                ]
            })
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = Sq8Params::calibrate(&refs).expect("calibrate");

        for v in &vectors {
            let quantized = params.quantize(v);
            let dequantized = params.dequantize(&quantized);

            for (i, (&orig, &restored)) in v.iter().zip(dequantized.iter()).enumerate() {
                let range = params.maxs[i] - params.mins[i];
                let error = (orig - restored).abs() / range;
                assert!(
                    error < 0.01,
                    "dim {i}: orig={orig}, restored={restored}, error={error}"
                );
            }
        }
    }

    #[test]
    fn quantize_clamps_out_of_range() {
        let params = Sq8Params {
            mins: vec![0.0],
            maxs: vec![1.0],
        };
        assert_eq!(params.quantize(&[-0.5]), vec![0]);
        assert_eq!(params.quantize(&[1.5]), vec![255]);
        assert_eq!(params.quantize(&[0.0]), vec![0]);
        assert_eq!(params.quantize(&[1.0]), vec![255]);
    }

    #[test]
    fn memory_savings_4x() {
        let qv = QuantizedVector {
            data: vec![128u8; 384],
        };
        assert_eq!(qv.size_bytes(), 384);
        assert_eq!(qv.original_size_bytes(), 384 * 4);
        assert_eq!(qv.original_size_bytes() / qv.size_bytes(), 4);
    }

    #[test]
    fn calibrate_empty_returns_none() {
        assert!(Sq8Params::calibrate(&[]).is_none());
    }

    #[test]
    fn calibrate_zero_dims_returns_none() {
        let empty: Vec<f32> = vec![];
        assert!(Sq8Params::calibrate(&[&empty]).is_none());
    }

    #[test]
    fn calibrate_dimension_mismatch_returns_none() {
        let v1 = vec![1.0, 2.0];
        let v2 = vec![1.0, 2.0, 3.0];
        assert!(Sq8Params::calibrate(&[&v1, &v2]).is_none());
    }

    #[test]
    fn calibrate_constant_dimension() {
        let v1 = vec![5.0, 1.0];
        let v2 = vec![5.0, 2.0];
        let params = Sq8Params::calibrate(&[&v1, &v2]).expect("calibrate");
        assert!(params.maxs[0] > params.mins[0]);
        let q = params.quantize(&v1);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn high_dimensional_768() {
        let dims = 768;
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|seed| (0..dims).map(|d| ((seed * d) as f32).sin()).collect())
            .collect();

        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let params = Sq8Params::calibrate(&refs).expect("calibrate");
        assert_eq!(params.dims(), 768);

        let quantized = params.quantize(&vectors[0]);
        assert_eq!(quantized.len(), 768);
    }

    #[test]
    fn serialization_roundtrip() {
        let params = Sq8Params {
            mins: vec![-1.0, 0.0],
            maxs: vec![1.0, 2.0],
        };
        let bytes = rmp_serde::to_vec(&params).expect("serialize");
        let restored: Sq8Params = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(params, restored);
    }

    #[test]
    fn validate_dims_ok() {
        let v = vec![0.0f32; 384];
        assert!(validate_dimensions(&v, 384).is_ok());
    }

    #[test]
    fn validate_dims_mismatch() {
        let v = vec![0.0f32; 384];
        assert!(validate_dimensions(&v, 512).is_err());
    }

    #[test]
    fn validate_dims_exceeds_max() {
        let v = vec![0.0f32; 1];
        assert!(validate_dimensions(&v, MAX_DIMENSIONS + 1).is_err());
    }

    #[test]
    fn kv_separation_threshold() {
        assert!(!should_kv_separate(128));
        assert!(!should_kv_separate(256));
        assert!(should_kv_separate(257));
        assert!(should_kv_separate(768));
    }
}
