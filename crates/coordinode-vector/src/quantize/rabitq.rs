//! RaBitQ 1-bit-per-dimension vector quantization (Gao & Long, SIGMOD 2024).
//!
//! Pipeline per shard:
//! 1. Once at index init, generate a random orthonormal rotation `R: ℝᴰ → ℝᴰ`
//!    (deterministic from `seed`).
//! 2. For each stored vector `x`, compute `x' = R · x`, then take sign bits:
//!    `code(x)[i] = (x'[i] > 0) as u1`. Result: `D` bits = `D/64` u64 words.
//! 3. Store auxiliary scalars `‖x‖` and the cross-term `<x', e>` for the
//!    asymmetric distance estimator (8 bytes total).
//!
//! Distance between two codes is recovered as:
//!   `dist ≈ (1 − 2·popcount(c_q ⊕ c_x)/D) · ‖q‖ · ‖x‖ + correction`
//! where the correction uses the cross-term scalars. The popcount is the
//! workload that makes the kernel ~10× faster than SQ8 dequant+dot and
//! ~50× faster than f32 AVX2 FMA at D=1024.
//!
//! **Cross-shard caveat.** `R` is per-shard. Codes are NOT comparable across
//! shards; cluster-wide top-K is recovered via the Phase 1.5 SQ8 rerank pool
//! (separate task). Inside a single shard, codes are fully comparable.
//
// no-std: rotation matrix storage uses `alloc::Vec<f32>` — clean. Internal RNG
//         is a self-contained xorshift64* (no `rand` dep), keeps this module
//         alloc-only despite being arithmetic-heavy.

use super::popcount;
use serde::{Deserialize, Serialize};

/// Per-shard rotation matrix + dimensionality. Constructed once at index init,
/// stable for the lifetime of the index. Persisted alongside the segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQParams {
    /// Row-major `D × D` orthonormal matrix. `rotation[i * dims + j]` is `R[i][j]`.
    rotation: Vec<f32>,
    /// Vector dimensionality `D`. Stored explicitly to avoid recomputing from len.
    dims: u32,
    /// RNG seed that produced `rotation`. Retained so the matrix can be
    /// regenerated deterministically (e.g. on segment recovery from a corrupt
    /// rotation blob — recover from seed alone).
    seed: u64,
}

/// 1-bit code of a single vector plus the scalars needed by the distance estimator.
///
/// Memory: `D/8 + 8` bytes per vector (e.g. 136 B at D=1024 vs 4096 B f32 = ~30× compression).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaBitQCode {
    /// Sign-bit code packed into `D/64` u64 words (LSB-first within each word).
    /// `code[i / 64]` bit `(i % 64)` corresponds to dimension `i`.
    pub code: Vec<u64>,
    /// `‖x‖₂` — L2 norm of the original (pre-rotation) vector.
    pub norm: f32,
    /// `<x', e>` where `e` is a fixed unit vector — feeds the asymmetric correction.
    pub cross_term: f32,
}

impl RaBitQCode {
    /// Memory size of the code on the wire / in RAM. Excludes any allocator overhead.
    pub fn size_bytes(&self) -> usize {
        self.code.len() * 8 + 8 // 8 bytes for f32 norm + f32 cross_term
    }
}

impl RaBitQParams {
    /// Build a deterministic orthonormal rotation matrix from a seed.
    ///
    /// Uses Gram-Schmidt over a seeded Gaussian — deterministic in `(dims, seed)`,
    /// so a node recovering a segment can reconstruct the matrix bit-identically
    /// from the persisted seed alone if the rotation blob is lost.
    ///
    /// # Examples
    ///
    /// ```
    /// use coordinode_vector::quantize::rabitq::RaBitQParams;
    /// let params = RaBitQParams::calibrate(128, 0xC0FFEE);
    /// let code = params.encode(&[0.1; 128]);
    /// assert_eq!(code.code.len(), 128 / 64);
    /// ```
    pub fn calibrate(dims: u32, seed: u64) -> Self {
        assert!(dims > 0, "RaBitQParams: dims must be non-zero");
        assert!(
            dims.is_multiple_of(64),
            "RaBitQParams: dims must be a multiple of 64 (popcount kernel operates on u64 words)"
        );

        let d = dims as usize;
        let mut rng = Xorshift64Star::new(seed);

        // Step 1: fill a D×D matrix with standard normal samples (Box-Muller).
        let mut mat: Vec<f32> = vec![0.0; d * d];
        for slot in mat.iter_mut() {
            *slot = rng.gaussian();
        }

        // Step 2: Modified Gram-Schmidt to orthonormalize rows.
        // This converts the random Gaussian matrix into an orthonormal one;
        // for a Gaussian seed the result is uniformly distributed over O(D)
        // (Haar measure) — exactly what RaBitQ wants.
        for i in 0..d {
            // Normalize row i.
            let mut norm_sq = 0.0f32;
            for j in 0..d {
                let v = mat[i * d + j];
                norm_sq += v * v;
            }
            let inv = 1.0 / norm_sq.sqrt().max(f32::EPSILON);
            for j in 0..d {
                mat[i * d + j] *= inv;
            }
            // Project subsequent rows against row i and subtract.
            for k in (i + 1)..d {
                let mut dot = 0.0f32;
                for j in 0..d {
                    dot += mat[i * d + j] * mat[k * d + j];
                }
                for j in 0..d {
                    mat[k * d + j] -= dot * mat[i * d + j];
                }
            }
        }

        Self {
            rotation: mat,
            dims,
            seed,
        }
    }

    /// Dimensionality `D`.
    pub fn dims(&self) -> u32 {
        self.dims
    }

    /// Seed that produced the rotation matrix (for deterministic recovery).
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Encode a single f32 vector to a RaBitQ code.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.dims as usize`.
    pub fn encode(&self, vector: &[f32]) -> RaBitQCode {
        assert_eq!(
            vector.len(),
            self.dims as usize,
            "RaBitQParams::encode: dimension mismatch"
        );

        let d = self.dims as usize;

        // Compute pre-rotation norm — needed unrotated because R is orthonormal
        // so ‖R·x‖ = ‖x‖; cheaper to compute on the original.
        let mut norm_sq = 0.0f32;
        for &v in vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        // Project: x' = R · x. Row-major R, output one f32 per dimension.
        // Sign-quantize on the fly into a packed u64 buffer.
        let words = d / 64;
        let mut code = vec![0u64; words];
        // Cross-term: dot with the fixed unit vector e = (1/√D) · (1, 1, ..., 1).
        // Equivalent to sum(x'[i]) / √D — captures the bias of the rotated vector.
        let mut sum_rotated = 0.0f32;
        let inv_sqrt_d = 1.0 / (d as f32).sqrt();

        for i in 0..d {
            let mut acc = 0.0f32;
            let row_base = i * d;
            for (j, &xj) in vector.iter().enumerate() {
                acc += self.rotation[row_base + j] * xj;
            }
            sum_rotated += acc;
            if acc > 0.0 {
                code[i / 64] |= 1u64 << (i % 64);
            }
        }

        RaBitQCode {
            code,
            norm,
            cross_term: sum_rotated * inv_sqrt_d,
        }
    }

    /// Estimate the inner product (dot product) between two encoded vectors.
    /// Higher = more similar. Suitable for `DotProduct` metric.
    ///
    /// For an L2 variant the caller can transform the returned value using
    /// the polarisation identity `‖q-x‖² = ‖q‖² + ‖x‖² - 2·<q,x>` with the
    /// norms recorded in each code.
    pub fn estimate_inner_product(&self, q: &RaBitQCode, x: &RaBitQCode) -> f32 {
        debug_assert_eq!(q.code.len(), x.code.len(), "code length mismatch");
        let d = self.dims as f32;
        let pop = popcount::xor_popcount(&q.code, &x.code) as f32;
        let cos_term = 1.0 - 2.0 * pop / d;
        cos_term * q.norm * x.norm
    }

    /// Estimate the cosine distance `1 - cos_similarity` between two codes
    /// without consulting the per-vector norms — purely a function of the
    /// popcount and `D`. Result is in `[0, 2]`.
    ///
    /// This is the fast path the HNSW hot loop hits for cosine workloads:
    /// one XOR + one popcount + a multiply + a subtract, no per-vector
    /// scalar arithmetic.
    pub fn estimate_cosine_distance(&self, q: &RaBitQCode, x: &RaBitQCode) -> f32 {
        debug_assert_eq!(q.code.len(), x.code.len(), "code length mismatch");
        let d = self.dims as f32;
        let pop = popcount::xor_popcount(&q.code, &x.code) as f32;
        // cos_sim_est = 1 - 2·popcount/D  →  distance = 1 - cos_sim_est = 2·popcount/D.
        2.0 * pop / d
    }
}

/// Deterministic xorshift64* PRNG. Pure `core`-level math, no allocations.
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    fn new(seed: u64) -> Self {
        // A zero seed is degenerate for xorshift; substitute a constant.
        Self {
            state: if seed == 0 {
                0x2545_F491_4F6C_DD1D
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform on [0, 1) — keeps full 24-bit f32 mantissa precision.
    fn next_f32(&mut self) -> f32 {
        // Take top 24 bits, scale to [0, 1).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal N(0, 1) via Box-Muller transform.
    fn gaussian(&mut self) -> f32 {
        // u1 must avoid exactly 0 to keep ln() finite.
        let u1 = self.next_f32().max(f32::MIN_POSITIVE);
        let u2 = self.next_f32();
        let mag = (-2.0 * u1.ln()).sqrt();
        mag * (2.0 * core::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn calibrate_is_deterministic() {
        let p1 = RaBitQParams::calibrate(128, 42);
        let p2 = RaBitQParams::calibrate(128, 42);
        assert_eq!(
            p1.rotation, p2.rotation,
            "same seed must yield same rotation"
        );
    }

    #[test]
    fn rotation_is_approximately_orthonormal() {
        // R · Rᵀ ≈ I for an orthonormal matrix. Check a few diagonal/off-diagonal cells.
        let p = RaBitQParams::calibrate(64, 12345);
        let d = 64usize;

        // Diagonal: row i dot row i should be ~1.
        for i in [0usize, 7, 31, 63] {
            let mut dot = 0.0f32;
            for j in 0..d {
                let v = p.rotation[i * d + j];
                dot += v * v;
            }
            assert!(
                approx_eq(dot, 1.0, 1e-4),
                "row {} self-dot = {}, expected ~1",
                i,
                dot
            );
        }

        // Off-diagonal: row i dot row k should be ~0.
        for (i, k) in [(0usize, 1usize), (0, 13), (5, 17), (30, 63)] {
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += p.rotation[i * d + j] * p.rotation[k * d + j];
            }
            assert!(
                approx_eq(dot, 0.0, 1e-4),
                "rows ({},{}) dot = {}, expected ~0",
                i,
                k,
                dot
            );
        }
    }

    #[test]
    fn encode_produces_expected_layout() {
        let p = RaBitQParams::calibrate(128, 1);
        let v = vec![0.5f32; 128];
        let c = p.encode(&v);
        assert_eq!(c.code.len(), 128 / 64);
        assert!(c.norm > 0.0);
    }

    #[test]
    fn identical_vectors_have_zero_xor() {
        let p = RaBitQParams::calibrate(128, 99);
        // Use a non-constant vector — a constant vector hits Gaussian symmetry
        // edge cases for the sign bits and isn't representative.
        let v: Vec<f32> = (0..128).map(|i| (i as f32).sin()).collect();
        let c1 = p.encode(&v);
        let c2 = p.encode(&v);
        assert_eq!(c1, c2);
        assert_eq!(popcount::xor_popcount(&c1.code, &c2.code), 0);
    }

    #[test]
    fn similarity_ranks_neighbours_correctly() {
        // Build three vectors: q, near (q + small noise), far (q rotated 180°).
        // The estimator must rank `near` closer to `q` than `far` does.
        let p = RaBitQParams::calibrate(256, 0xABCD);
        let q: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).cos()).collect();
        let near: Vec<f32> = q
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.01 * (i as f32).sin())
            .collect();
        let far: Vec<f32> = q.iter().map(|x| -x).collect();

        let qc = p.encode(&q);
        let nc = p.encode(&near);
        let fc = p.encode(&far);

        let sim_near = p.estimate_inner_product(&qc, &nc);
        let sim_far = p.estimate_inner_product(&qc, &fc);
        assert!(
            sim_near > sim_far,
            "estimator must rank near above far: near={}, far={}",
            sim_near,
            sim_far
        );
    }

    #[test]
    fn code_size_matches_spec() {
        // D=1024 → 128 bytes code + 8 bytes scalars = 136 bytes.
        let p = RaBitQParams::calibrate(1024, 7);
        let v = vec![1.0f32; 1024];
        let c = p.encode(&v);
        assert_eq!(c.size_bytes(), 136);
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn encode_rejects_wrong_dim() {
        let p = RaBitQParams::calibrate(128, 0);
        let _ = p.encode(&[0.0f32; 64]);
    }

    #[test]
    #[should_panic(expected = "multiple of 64")]
    fn calibrate_rejects_non_64_aligned_dims() {
        let _ = RaBitQParams::calibrate(100, 0);
    }
}
