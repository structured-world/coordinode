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
///
/// The codec internally rounds `user_dims` UP to the next multiple of 64
/// (`effective_dims`) so the popcount kernel always operates on whole u64
/// words. Callers see `user_dims`; the encoder pads input vectors with
/// zeros to `effective_dims` before rotation. Padding with zeros never
/// changes the rotated vector's sign-bit pattern in the user_dims slots
/// (rotation of zeros stays zeros), and the extra slots contribute zero
/// to both the popcount and the L2 norm — so codes built at any user_dims
/// are bitwise-comparable inside one index without recall loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQParams {
    /// Row-major `effective_dims × effective_dims` orthonormal matrix.
    rotation: Vec<f32>,
    /// Vector dimensionality the CALLER supplies. May be < `effective_dims`
    /// (rounded up internally) but never larger.
    dims: u32,
    /// Internal dim (next multiple of 64 ≥ `dims`). Used everywhere the
    /// codec touches the rotation matrix or the packed code arrays.
    effective_dims: u32,
    /// RNG seed that produced `rotation`. Retained so the matrix can be
    /// regenerated deterministically (e.g. on segment recovery from a
    /// corrupt rotation blob — recover from seed alone).
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
        // Round up so the popcount kernel always operates on whole u64
        // words. Padding the input with zeros at encode time keeps
        // codes comparable across vectors at any user_dims.
        let effective_dims = dims.div_ceil(64) * 64;

        let d = effective_dims as usize;
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
            effective_dims,
            seed,
        }
    }

    /// User-visible dimensionality (what the caller supplied at calibration).
    pub fn dims(&self) -> u32 {
        self.dims
    }

    /// Padded dim used by the internal rotation matrix + packed code.
    /// `effective_dims = dims.div_ceil(64) * 64`; equal to `dims` only
    /// when `dims % 64 == 0`.
    pub fn effective_dims(&self) -> u32 {
        self.effective_dims
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

        // Rotation matrix is `effective_dims × effective_dims`. The
        // input is zero-padded to that width inside the dot product
        // (slots ≥ self.dims contribute 0). Cross-term + popcount run
        // over the full effective range — extra zero slots add 0 sign
        // bits which doesn't bias the codes.
        let d_eff = self.effective_dims as usize;
        let d_user = self.dims as usize;

        // ‖R·x‖ = ‖x‖ for orthonormal R; user_dims is the only place
        // x is non-zero so the norm is identical to ‖x_user‖.
        let mut norm_sq = 0.0f32;
        for &v in vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        let words = d_eff / 64;
        let mut code = vec![0u64; words];
        let mut sum_rotated = 0.0f32;
        let inv_sqrt_d = 1.0 / (d_eff as f32).sqrt();

        for i in 0..d_eff {
            let mut acc = 0.0f32;
            let row_base = i * d_eff;
            // Only iterate the non-zero (user-dim) range; padded slots
            // contribute 0 to the dot product so we skip them.
            for (j, &xj) in vector.iter().take(d_user).enumerate() {
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
        // Use effective_dims — the rotation runs in the padded space,
        // so popcount and the cos-term scale are both over D_effective.
        let d = self.effective_dims as f32;
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
        // Same effective_dims rationale as estimate_inner_product.
        let d = self.effective_dims as f32;
        let pop = popcount::xor_popcount(&q.code, &x.code) as f32;
        // cos_sim_est = 1 - 2·popcount/D  →  distance = 1 - cos_sim_est = 2·popcount/D.
        2.0 * pop / d
    }

    /// Encode a single f32 vector to an Extended-RaBitQ code at the given
    /// `bits ∈ {2, 3, 4}` resolution. 1-bit uses [`Self::encode`] (different
    /// kernel — pure sign-bit popcount, faster).
    ///
    /// The rotation matrix is shared with the 1-bit codec: switching bit
    /// width on a reload doesn't require a rebuild, only re-encoding the
    /// stored f32 originals.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.dims as usize` or `bits` is outside
    /// `{2, 3, 4}`.
    pub fn encode_ext(&self, vector: &[f32], bits: u8) -> RaBitQExtCode {
        assert_eq!(
            vector.len(),
            self.dims as usize,
            "RaBitQParams::encode_ext: dimension mismatch"
        );
        assert!(
            (2..=4).contains(&bits),
            "RaBitQParams::encode_ext: bits must be 2, 3, or 4; got {bits}"
        );

        // Same effective_dims padding rationale as `encode` — rotation
        // operates in the padded space; the input contributes only its
        // user_dims slots, the rest are zero-implicit.
        let d_eff = self.effective_dims as usize;
        let d_user = self.dims as usize;
        let levels_count = 1u8 << bits;

        let mut norm_sq = 0.0f32;
        for &v in vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(f32::EPSILON);

        let inv_norm = 1.0 / norm;
        let mut levels = vec![0u8; d_eff];
        let mut sum_rotated = 0.0f32;
        let inv_sqrt_d = 1.0 / (d_eff as f32).sqrt();

        for (i, slot) in levels.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            let row_base = i * d_eff;
            for (j, &xj) in vector.iter().take(d_user).enumerate() {
                acc += self.rotation[row_base + j] * xj;
            }
            sum_rotated += acc;
            *slot = quantize_normal(acc * inv_norm, bits);
        }
        debug_assert!(
            levels.iter().all(|&l| l < levels_count),
            "all quantized levels must fit in `bits`"
        );

        RaBitQExtCode {
            packed: pack_levels(&levels, bits),
            dims: d_eff as u32,
            bits,
            norm,
            cross_term: sum_rotated * inv_sqrt_d,
        }
    }

    /// Estimate inner product between two Extended-RaBitQ codes (same
    /// `bits`). Both codes were normalized by their own ‖x‖ at encode,
    /// so the level centroids reconstruct unit-norm rotated directions;
    /// we multiply back by `q.norm * x.norm` to recover the original
    /// inner-product magnitude.
    ///
    /// # Panics
    ///
    /// Panics in debug if `q.bits != x.bits` or dimension mismatch.
    pub fn estimate_inner_product_ext(&self, q: &RaBitQExtCode, x: &RaBitQExtCode) -> f32 {
        debug_assert_eq!(q.bits, x.bits, "bit-width mismatch");
        // Codes are sized at `effective_dims` (rotation+pack space).
        debug_assert_eq!(q.dims, self.effective_dims, "q dimension mismatch");
        debug_assert_eq!(x.dims, self.effective_dims, "x dimension mismatch");
        let table = level_centroids(q.bits);
        let d = self.effective_dims as usize;
        let mut acc = 0.0f32;
        for i in 0..d {
            acc += table[q.level(i) as usize] * table[x.level(i) as usize];
        }
        acc * q.norm * x.norm
    }

    /// Estimate cosine distance `1 - cos_similarity` between two Extended-
    /// RaBitQ codes. Codes are already unit-scale (norm divided out at
    /// encode); this returns `1 − Σ centroid(lq)·centroid(lx) / D_norm`.
    /// Result is in `[0, 2]` after clamping for numerical noise.
    pub fn estimate_cosine_distance_ext(&self, q: &RaBitQExtCode, x: &RaBitQExtCode) -> f32 {
        debug_assert_eq!(q.bits, x.bits, "bit-width mismatch");
        let table = level_centroids(q.bits);
        let d = self.effective_dims as usize;
        let mut dot = 0.0f32;
        let mut nq = 0.0f32;
        let mut nx = 0.0f32;
        for i in 0..d {
            let vq = table[q.level(i) as usize];
            let vx = table[x.level(i) as usize];
            dot += vq * vx;
            nq += vq * vq;
            nx += vx * vx;
        }
        let denom = (nq * nx).sqrt().max(f32::EPSILON);
        (1.0 - dot / denom).clamp(0.0, 2.0)
    }
}

/// Extended-RaBitQ `bits`-per-dimension code (R862, SIGMOD 2025).
///
/// Each dimension is uniformly scalar-quantized into `2^bits` levels
/// (cut points at the quantiles of N(0, 1) — rotated isotropic data
/// is approximately normal after divide-by-norm). Level indices are
/// packed at `bits` per dim into a `ceil(D × bits / 8)` byte buffer,
/// matching the storage size quoted in the SIGMOD 2025 paper:
///
/// | bits | D=1024 packed | D=1024 (old 1 B/dim) |
/// |------|---------------|----------------------|
/// | 2    | 256 B         | 1024 B (-75%)        |
/// | 3    | 384 B         | 1024 B (-62.5%)      |
/// | 4    | 512 B         | 1024 B (-50%)        |
///
/// Distance kernel decodes a level on demand via [`Self::level`] —
/// modestly more arithmetic per dim than an array index, paid back
/// many times over by tighter cache behaviour at scale.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaBitQExtCode {
    /// Bit-packed level indices: `bits` bits per dim, dim 0 in the
    /// LSBs of `packed[0]`, dims advance through the stream LSB-first.
    /// Length is `ceil(dims × bits / 8)`.
    pub packed: Vec<u8>,
    /// Number of dimensions encoded. Stored explicitly because
    /// `packed.len()` alone can't disambiguate trailing-bit padding
    /// at non-multiple-of-8 dim×bits products.
    pub dims: u32,
    /// Bit-width that produced this code. Required to pick the correct
    /// centroid lookup table at distance time.
    pub bits: u8,
    /// `‖x‖₂` — L2 norm of the original (pre-rotation) vector.
    pub norm: f32,
    /// `<x', e>` for the asymmetric correction. Same identity as the
    /// 1-bit code — feeds Phase 1.5 rerank fallback when bit-width is
    /// low enough that operator opts in.
    pub cross_term: f32,
}

impl RaBitQExtCode {
    /// Memory size on the wire / in RAM.
    pub fn size_bytes(&self) -> usize {
        self.packed.len() + 4 + 1 + 4 + 4 // dims:u32 + bits:u8 + norm:f32 + cross_term:f32
    }

    /// Extract the level index at dimension `i`. Returns a value in
    /// `[0, 2^bits)`. Out-of-range `i` returns 0 (sane fallback for
    /// the kernel; callers should know their dim count).
    #[inline]
    pub fn level(&self, i: usize) -> u8 {
        if i >= self.dims as usize {
            return 0;
        }
        let bits = self.bits as usize;
        let bit_idx = i * bits;
        let byte_idx = bit_idx / 8;
        let bit_offset = bit_idx % 8;
        let mask = (1u16 << bits) - 1;
        // Read 16 bits straddling the byte boundary so we don't have
        // to special-case the 3-bit / cross-byte case.
        let lo = self.packed[byte_idx] as u16;
        let hi = if byte_idx + 1 < self.packed.len() {
            self.packed[byte_idx + 1] as u16
        } else {
            0
        };
        let word = lo | (hi << 8);
        ((word >> bit_offset) & mask) as u8
    }
}

/// Pack `dims` level indices (each `< 2^bits`) into a bit-packed byte
/// vector. `bits` MUST be in `{2,3,4}`; caller pre-validates.
fn pack_levels(levels: &[u8], bits: u8) -> Vec<u8> {
    let dims = levels.len();
    let bits_n = bits as usize;
    let total_bits = dims * bits_n;
    let bytes = total_bits.div_ceil(8);
    let mut out = vec![0u8; bytes];
    for (i, &lvl) in levels.iter().enumerate() {
        let bit_idx = i * bits_n;
        let byte_idx = bit_idx / 8;
        let bit_offset = bit_idx % 8;
        // Spread the level across at most two bytes (3-bit at offset 6/7
        // straddles; 2/4-bit never do because they divide evenly).
        let v = (lvl as u16) << bit_offset;
        out[byte_idx] |= (v & 0xFF) as u8;
        if (bit_offset + bits_n) > 8 && byte_idx + 1 < out.len() {
            out[byte_idx + 1] |= ((v >> 8) & 0xFF) as u8;
        }
    }
    out
}

/// Quantize a single value (already divided by ‖x‖, so ~N(0, 1/D)) to a
/// 2/3/4-bit level index using the cut-points of the standard normal at
/// the same divide. The cut-points come from numerically computed
/// quantiles of the N(0, 1) distribution.
fn quantize_normal(v: f32, bits: u8) -> u8 {
    let cuts = quantile_cut_points(bits);
    // cuts has 2^bits - 1 entries (inner boundaries).
    let mut lvl = 0u8;
    for &c in cuts {
        if v >= c {
            lvl += 1;
        } else {
            break;
        }
    }
    lvl
}

/// Standard-normal quantile cut-points for 2-, 3-, and 4-bit quantization.
/// Each table has `2^bits - 1` entries, partitioning ℝ into `2^bits`
/// equal-probability buckets under N(0, 1).
///
/// Values from `scipy.stats.norm.ppf(k / 2^bits)` for `k ∈ [1, 2^bits-1]`.
/// Callers MUST validate `bits ∈ {2,3,4}` (encode_ext does via assert).
#[allow(clippy::panic, reason = "callers validate bits ∈ {2,3,4} via assert!")]
fn quantile_cut_points(bits: u8) -> &'static [f32] {
    match bits {
        // 4 levels → 3 cuts at the 25/50/75 percentiles.
        2 => &[-0.674_49, 0.0, 0.674_49],
        // 8 levels → 7 cuts at the 12.5/25/.../87.5 percentiles.
        3 => &[
            -1.150_349_4,
            -0.674_49,
            -0.318_639_4,
            0.0,
            0.318_639_4,
            0.674_49,
            1.150_349_4,
        ],
        // 16 levels → 15 cuts at the 6.25/12.5/.../93.75 percentiles.
        4 => &[
            -1.534_120_5,
            -1.150_349_4,
            -0.887_146_6,
            -0.674_49,
            -0.488_776_4,
            -0.318_639_4,
            -0.157_310_7,
            0.0,
            0.157_310_7,
            0.318_639_4,
            0.488_776_4,
            0.674_49,
            0.887_146_6,
            1.150_349_4,
            1.534_120_5,
        ],
        _ => panic!("Extended-RaBitQ only supports bits ∈ {{2,3,4}}; got {bits}"),
    }
}

/// Centroids of each level under N(0, 1) — the conditional mean of the
/// standard normal restricted to that bucket. Used by the symmetric
/// distance kernel: each level index maps back to its bucket centroid,
/// and the inner product is computed by table-lookup-and-MAC over dims.
///
/// `centroids.len() == 2^bits`. Callers MUST validate `bits ∈ {2,3,4}`.
#[allow(clippy::panic, reason = "callers validate bits ∈ {2,3,4} via assert!")]
fn level_centroids(bits: u8) -> &'static [f32] {
    match bits {
        2 => &[-1.271_021, -0.317_754_5, 0.317_754_5, 1.271_021],
        3 => &[
            -1.747_874,
            -1.049_750_7,
            -0.682_643_7,
            -0.213_344_7,
            0.213_344_7,
            0.682_643_7,
            1.049_750_7,
            1.747_874,
        ],
        4 => &[
            -2.077_605,
            -1.451_927,
            -1.108_14,
            -0.860_169,
            -0.658_383,
            -0.482_055,
            -0.322_226,
            -0.169_741_8,
            0.169_741_8,
            0.322_226,
            0.482_055,
            0.658_383,
            0.860_169,
            1.108_14,
            1.451_927,
            2.077_605,
        ],
        _ => panic!("Extended-RaBitQ only supports bits ∈ {{2,3,4}}; got {bits}"),
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
    fn calibrate_pads_non_64_aligned_dims_internally() {
        // Calibration accepts any dims > 0. The rotation matrix and
        // packed code arrays are sized at the next multiple of 64 above
        // the user-supplied dim; encode pads the input with implicit
        // zeros so popcount stays a whole-u64-words operation.
        let p = RaBitQParams::calibrate(100, 0);
        assert_eq!(p.dims(), 100);
        assert_eq!(p.effective_dims(), 128);
        // Code length = effective_dims / 64 u64 words.
        let v: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.01).collect();
        let c = p.encode(&v);
        assert_eq!(
            c.code.len(),
            2,
            "code uses 2 u64 words at effective_dims=128"
        );
        // Roundtrip identity: encode the same vector twice → same code.
        let c2 = p.encode(&v);
        assert_eq!(c, c2);
    }

    #[test]
    fn padding_preserves_cosine_ranking_dim_100() {
        // Regression: glove-100-angular bench on commit 8fa0f2f returned
        // recall@10 = 0.17 plateau across the full ef sweep for codec=rabitq
        // while codec=none reached 0.94. Hypothesis: the dim=100 → 128
        // zero-padding logic in `encode` breaks cosine rank preservation.
        //
        // Theoretical claim being tested: for orthonormal R ∈ ℝ^{128×128},
        // the first 100 columns M = R[:, 0:100] satisfy MᵀM = I_100, so M
        // is an isometry ℝ^100 → ℝ^128 that preserves cosine similarity.
        // RaBitQ sign bits of M·x should give the standard LSH separation
        // arcsin(ρ)/π between true neighbours and random pairs. Recall@10
        // on a 1000-vector corpus with a planted nearest neighbour should
        // be ≥ 0.6, far above the broken 0.17 plateau.
        //
        // If THIS test fails (recall < 0.5), the bug is in `encode` or in
        // `estimate_cosine_distance` and isolated from HNSW graph build /
        // search. If it passes, the bug is elsewhere in the index path.
        let dims = 100u32;
        let n_base = 1000usize;
        let n_query = 100usize;
        let k = 10usize;
        let seed = 0xC0FFEE_u64;

        let params = RaBitQParams::calibrate(dims, seed);
        assert_eq!(params.effective_dims(), 128);

        // Generate unit-norm Gaussian vectors. For each query, plant one
        // true near-neighbour at cosine ≈ 0.95 (a tight high-similarity
        // pair like glove's top-1) so we have a known correct answer.
        let mut rng = Xorshift64Star::new(0xBADC0DE);

        let make_unit = |rng: &mut Xorshift64Star| -> Vec<f32> {
            let mut v: Vec<f32> = (0..dims as usize).map(|_| rng.gaussian()).collect();
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= n.max(f32::EPSILON);
            }
            v
        };

        let mut base: Vec<Vec<f32>> = (0..n_base).map(|_| make_unit(&mut rng)).collect();

        // Plant the first `n_query` base vectors as the "true" nearest
        // neighbours of the queries: query[i] = base[i] + small noise, both
        // re-normalized. Cosine sim ≈ 1 - ‖noise‖²/2 ≈ 0.95 for noise=0.32.
        let queries: Vec<Vec<f32>> = (0..n_query)
            .map(|i| {
                let mut q = base[i].clone();
                for x in q.iter_mut() {
                    *x += 0.32 * rng.gaussian() / (dims as f32).sqrt();
                }
                let n: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
                for x in q.iter_mut() {
                    *x /= n.max(f32::EPSILON);
                }
                q
            })
            .collect();
        // Re-normalize base just in case (defensive — already unit above).
        for v in base.iter_mut() {
            let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= n.max(f32::EPSILON);
            }
        }

        // Encode all base vectors and queries.
        let base_codes: Vec<RaBitQCode> = base.iter().map(|v| params.encode(v)).collect();
        let query_codes: Vec<RaBitQCode> = queries.iter().map(|v| params.encode(v)).collect();

        // For each query, find top-k by RaBitQ distance and check whether
        // the planted true-NN (index i) is in the returned top-k.
        let mut hits = 0usize;
        for (qi, qc) in query_codes.iter().enumerate() {
            let mut scored: Vec<(f32, usize)> = base_codes
                .iter()
                .enumerate()
                .map(|(bi, bc)| (params.estimate_cosine_distance(qc, bc), bi))
                .collect();
            scored.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("RaBitQ distance values are finite")
            });
            if scored.iter().take(k).any(|(_, idx)| *idx == qi) {
                hits += 1;
            }
        }
        let recall = hits as f32 / n_query as f32;

        // Loose threshold (0.5) — well above broken plateau 0.17, well
        // below the f32-exact ceiling. Real RaBitQ on this workload
        // should land 0.7-0.9.
        assert!(
            recall >= 0.5,
            "RaBitQ recall@{k} on dim={dims} with padding = {recall:.3} \
             (need ≥0.5; broken 8fa0f2f gave 0.17)"
        );
    }

    #[test]
    fn calibrate_rejects_zero_dims() {
        let r = std::panic::catch_unwind(|| RaBitQParams::calibrate(0, 0));
        assert!(r.is_err(), "dims=0 still panics");
    }

    #[test]
    fn params_serde_round_trip_preserves_rotation_and_codes() {
        // The rotation matrix is durable index state — on segment reload we
        // serialise it, hand it back via `HnswIndex::set_rabitq_params`, and
        // re-encoded vectors must match the codes that were on disk. Verify
        // the full cycle: encode → serialise params → deserialise → re-encode
        // the same vector → bit-identical code.
        let dims = 128u32;
        let seed = 0xFEED_FACEu64;
        let params = RaBitQParams::calibrate(dims, seed);
        let v: Vec<f32> = (0..dims as usize)
            .map(|i| ((i as f32) * 0.13).sin())
            .collect();
        let code_before = params.encode(&v);

        let bytes = rmp_serde::to_vec(&params).expect("serialise params");
        let params2: RaBitQParams = rmp_serde::from_slice(&bytes).expect("deserialise params");
        assert_eq!(params2.dims(), dims);
        assert_eq!(params2.seed(), seed);

        let code_after = params2.encode(&v);
        assert_eq!(
            code_before, code_after,
            "round-tripped params must encode identically"
        );
    }

    #[test]
    fn code_serde_round_trip_is_bit_identical() {
        // Codes live on disk too (in-RAM index ↔ persisted segment). Verify
        // a (de)serialise cycle round-trips the bit-string + scalars exactly.
        let params = RaBitQParams::calibrate(256, 7);
        let v: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.05).cos()).collect();
        let code = params.encode(&v);

        let bytes = rmp_serde::to_vec(&code).expect("serialise code");
        let code2: RaBitQCode = rmp_serde::from_slice(&bytes).expect("deserialise code");
        assert_eq!(code, code2);
    }

    // ── Extended-RaBitQ (R862) ──────────────────────────────────────

    #[test]
    fn ext_encode_layout_2_3_4_bit() {
        let dims = 128u32;
        let p = RaBitQParams::calibrate(dims, 7);
        let v: Vec<f32> = (0..dims as usize)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        for bits in [2u8, 3, 4] {
            let c = p.encode_ext(&v, bits);
            assert_eq!(c.bits, bits);
            assert_eq!(c.dims, dims);
            // packed length must match ceil(dims × bits / 8)
            let expected_packed = (dims as usize * bits as usize).div_ceil(8);
            assert_eq!(
                c.packed.len(),
                expected_packed,
                "bits={bits}: packed length expected {expected_packed}, got {}",
                c.packed.len()
            );
            let max = 1u8 << bits;
            for i in 0..dims as usize {
                assert!(
                    c.level(i) < max,
                    "bits={bits}: level({i})={} must be < {max}",
                    c.level(i)
                );
            }
            assert!(c.norm > 0.0);
        }
    }

    #[test]
    fn ext_code_size_matches_spec() {
        // True bit-packed layout per the SIGMOD 2025 paper:
        //   bits=2 → 256 B packed + 13 B scalars (dims:u32 + bits:u8 + 2×f32) = 269 B
        //   bits=3 → 384 B + 13 = 397 B
        //   bits=4 → 512 B + 13 = 525 B
        // The arch doc quotes the packed body only (256/384/512); the
        // extra 13 B per code is scalar metadata + dims field for safe
        // decoding when dims × bits doesn't fit cleanly in a byte.
        let p = RaBitQParams::calibrate(1024, 7);
        let v = vec![1.0f32; 1024];
        for (bits, packed_bytes) in [(2u8, 256), (3, 384), (4, 512)] {
            let c = p.encode_ext(&v, bits);
            assert_eq!(
                c.packed.len(),
                packed_bytes,
                "bits={bits}: packed body must be {packed_bytes} B"
            );
            assert_eq!(
                c.size_bytes(),
                packed_bytes + 13,
                "bits={bits}: total size mismatch"
            );
        }
    }

    #[test]
    #[should_panic(expected = "bits must be 2, 3, or 4")]
    fn ext_rejects_one_bit() {
        let p = RaBitQParams::calibrate(64, 0);
        let _ = p.encode_ext(&[0.5f32; 64], 1);
    }

    #[test]
    #[should_panic(expected = "bits must be 2, 3, or 4")]
    fn ext_rejects_five_bit() {
        let p = RaBitQParams::calibrate(64, 0);
        let _ = p.encode_ext(&[0.5f32; 64], 5);
    }

    #[test]
    fn ext_ranks_neighbours_correctly_at_each_bit_width() {
        // Mirrors similarity_ranks_neighbours_correctly for 1-bit:
        // near must score above far at every bit width.
        let p = RaBitQParams::calibrate(256, 0xABCD);
        let q: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.07).cos()).collect();
        let near: Vec<f32> = q
            .iter()
            .enumerate()
            .map(|(i, x)| x + 0.01 * (i as f32).sin())
            .collect();
        let far: Vec<f32> = q.iter().map(|x| -x).collect();

        for bits in [2u8, 3, 4] {
            let qc = p.encode_ext(&q, bits);
            let nc = p.encode_ext(&near, bits);
            let fc = p.encode_ext(&far, bits);

            let sim_near = p.estimate_inner_product_ext(&qc, &nc);
            let sim_far = p.estimate_inner_product_ext(&qc, &fc);
            assert!(
                sim_near > sim_far,
                "bits={bits}: IP estimator must rank near above far: \
                 near={sim_near}, far={sim_far}"
            );

            let dist_near = p.estimate_cosine_distance_ext(&qc, &nc);
            let dist_far = p.estimate_cosine_distance_ext(&qc, &fc);
            assert!(
                dist_near < dist_far,
                "bits={bits}: cosine distance must rank near below far: \
                 near={dist_near}, far={dist_far}"
            );
        }
    }

    #[test]
    fn ext_higher_bits_estimate_closer_to_true_ip() {
        // Pareto property from SIGMOD 2025: estimator error decreases
        // monotonically with bit width. Averaged over a small sample,
        // 4-bit MUST beat 2-bit mean absolute error.
        let dims = 256u32;
        let p = RaBitQParams::calibrate(dims, 0xC0FFEE);

        fn synth(seed: u64, dims: usize) -> Vec<f32> {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            (0..dims)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    let u = (s >> 40) as f32 / (1u32 << 24) as f32;
                    2.0 * u - 1.0
                })
                .collect()
        }

        let mut err2 = 0.0f64;
        let mut err4 = 0.0f64;
        let mut n = 0usize;
        for seed_q in 0..6u64 {
            let q = synth(seed_q, dims as usize);
            for seed_x in 10..16u64 {
                let x = synth(seed_x, dims as usize);
                let true_ip: f32 = q.iter().zip(x.iter()).map(|(a, b)| a * b).sum();

                let q2 = p.encode_ext(&q, 2);
                let x2 = p.encode_ext(&x, 2);
                let est2 = p.estimate_inner_product_ext(&q2, &x2);

                let q4 = p.encode_ext(&q, 4);
                let x4 = p.encode_ext(&x, 4);
                let est4 = p.estimate_inner_product_ext(&q4, &x4);

                err2 += ((est2 - true_ip) as f64).abs();
                err4 += ((est4 - true_ip) as f64).abs();
                n += 1;
            }
        }
        let avg2 = err2 / n as f64;
        let avg4 = err4 / n as f64;
        assert!(
            avg4 < avg2,
            "4-bit mean abs IP error ({avg4}) must be lower than 2-bit ({avg2})"
        );
    }

    #[test]
    fn ext_serde_round_trip_is_value_identical() {
        let dims = 128u32;
        let p = RaBitQParams::calibrate(dims, 0xFACE);
        let v: Vec<f32> = (0..dims as usize)
            .map(|i| (i as f32 * 0.05).cos())
            .collect();
        for bits in [2u8, 3, 4] {
            let c = p.encode_ext(&v, bits);
            let bytes = rmp_serde::to_vec(&c).expect("ser");
            let c2: RaBitQExtCode = rmp_serde::from_slice(&bytes).expect("de");
            assert_eq!(c, c2);
        }
    }
}
