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
use smallvec::SmallVec;

/// Inline-up-to-4-u64-words storage for the RaBitQ sign-bit code. Covers
/// every `effective_dims ≤ 256` (the most common HNSW workload, including
/// glove-100-angular padded to 128) without a heap allocation, so reading
/// the code on a hot HNSW visit is a struct-field load instead of a
/// pointer-chase through `Vec`'s heap buffer. For larger D the SmallVec
/// spills to heap with the same semantics as `Vec<u64>`.
pub type CodeWords = SmallVec<[u64; 4]>;

/// Squared L2 distance between two equal-length f32 slices. Used by the
/// k-means assignment pass; kept private so the dispatch matches the rest
/// of the codec's vector ops (no SIMD intrinsic here — the assignment
/// loop is `O(N · K · D)` once at calibration, dwarfed by the encode
/// phase that follows).
#[inline]
fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Lloyd's K-means with K-means++ initialisation. Returns the flat
/// `n_clusters * dims` centroid array. Deterministic in `(training,
/// dims, n_clusters, seed)`.
///
/// Caps at 12 iterations — for the data sizes we calibrate on
/// (`calibration_threshold ≤ 100k`) convergence is reached well before
/// that empirically, and the upper bound keeps calibration latency
/// bounded.
fn kmeans_lloyd(training: &[Vec<f32>], dims: usize, k: usize, seed: u64) -> Vec<f32> {
    let n = training.len();
    let mut rng = Xorshift64Star::new(seed);

    // K-means++ init: first centroid uniform-random; each subsequent
    // centroid sampled with probability proportional to `min_dist²` so
    // the initial set spreads across the data.
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dims);
    let first = (rng.next_u64() as usize) % n;
    centroids.extend_from_slice(&training[first]);

    let mut min_dists: Vec<f32> = training
        .iter()
        .map(|v| l2_sq(v, &training[first]))
        .collect();

    for _ in 1..k {
        // Cumulative distribution sampling: pick index where cumulative
        // sum first exceeds u * total_dist.
        let total: f32 = min_dists.iter().sum::<f32>().max(f32::EPSILON);
        let target = (rng.next_u64() as f32 / u64::MAX as f32) * total;
        let mut acc = 0.0f32;
        let mut picked = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            acc += d;
            if acc >= target {
                picked = i;
                break;
            }
        }
        let start = centroids.len();
        centroids.extend_from_slice(&training[picked]);
        let new_centroid = &centroids[start..start + dims];
        for (i, v) in training.iter().enumerate() {
            let d = l2_sq(v, new_centroid);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
    }

    // Lloyd iterations: assign each point to nearest centroid, update
    // centroids to assigned-set mean. 12-iteration ceiling.
    let mut assignments = vec![0u16; n];
    for _iter in 0..12 {
        // Assignment pass.
        let mut changed = false;
        for (i, v) in training.iter().enumerate() {
            let mut best_k = 0u16;
            let mut best_d = f32::INFINITY;
            for ck in 0..k {
                let c = &centroids[ck * dims..(ck + 1) * dims];
                let d = l2_sq(v, c);
                if d < best_d {
                    best_d = d;
                    best_k = ck as u16;
                }
            }
            if assignments[i] != best_k {
                assignments[i] = best_k;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update pass: recompute each centroid as the mean of its
        // assigned vectors. Empty clusters keep their previous centroid
        // (no Forgy-style reset — k-means++ init makes empties rare).
        let mut sums = vec![0.0f32; k * dims];
        let mut counts = vec![0u32; k];
        for (i, v) in training.iter().enumerate() {
            let ck = assignments[i] as usize;
            let base = ck * dims;
            for j in 0..dims {
                sums[base + j] += v[j];
            }
            counts[ck] += 1;
        }
        for (ck, &count) in counts.iter().enumerate().take(k) {
            if count == 0 {
                continue;
            }
            let inv = 1.0 / count as f32;
            let base = ck * dims;
            for j in 0..dims {
                centroids[base + j] = sums[base + j] * inv;
            }
        }
    }

    centroids
}

/// Sign-flip helper: `x[i] *= -1` for every bit set in `signs` (LSB-first
/// within each u64 word). Branchless via the IEEE 754 sign bit XOR.
#[inline]
fn apply_sign_flip(x: &mut [f32], signs: &[u64]) {
    debug_assert_eq!(x.len(), signs.len() * 64);
    for (word_idx, &word) in signs.iter().enumerate() {
        let base = word_idx * 64;
        let mut bits = word;
        while bits != 0 {
            let bit = bits.trailing_zeros() as usize;
            // Flip the IEEE 754 sign bit. Faster than a multiply by -1 on
            // older toolchains and avoids the +/-0.0 quirk for zero inputs.
            let v = &mut x[base + bit];
            *v = f32::from_bits(v.to_bits() ^ 0x8000_0000);
            bits &= bits - 1;
        }
    }
}

/// In-place radix-2 Hadamard butterfly. Output is `√D · H · x` where `H`
/// has entries ±1/√D. The caller scales the result at the end of the Kac
/// composite (`fht_kac_in_place` multiplies by `1/D` once after two passes).
#[inline]
fn fht_in_place_unscaled(x: &mut [f32]) {
    let d = x.len();
    debug_assert!(d.is_power_of_two(), "FHT requires power-of-two length");
    let mut h = 1;
    while h < d {
        let mut i = 0;
        while i < d {
            for j in 0..h {
                let a = x[i + j];
                let b = x[i + j + h];
                x[i + j] = a + b;
                x[i + j + h] = a - b;
            }
            i += h * 2;
        }
        h *= 2;
    }
}

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
    /// First Rademacher sign vector for the FHT-Kac rotation. One ±1 entry
    /// per dimension, packed LSB-first into `effective_dims / 64` u64 words
    /// (bit 0 → sign +1, bit 1 → sign −1).
    ///
    /// Replaces the legacy `Vec<f32>` D×D rotation matrix that was driven by
    /// Gram-Schmidt on a Gaussian RNG: that matrix had Θ(D²) memory and Θ(D²)
    /// encode cost, and in f32 the Gram-Schmidt pass accumulated enough
    /// orthonormality drift on the tail rows that the Eq. 20 estimator
    /// collapsed on real workloads (random-100-angular recall=0.22 at ef=800).
    /// FHT-Kac is exactly orthonormal by construction, runs in Θ(D log D),
    /// uses Θ(D / 8) memory per round, and matches the rotation the RaBitQ
    /// SIGMOD 2024 reference C++ implementation
    /// (<https://github.com/VectorDB-NTU/RaBitQ-Library>) uses to reach
    /// recall=0.87 on the same dataset where our matrix-rotation peaked at
    /// 0.22.
    sign_a: Vec<u64>,
    /// Second Rademacher sign vector (Kac walk needs ≥2 rounds to
    /// approximate uniform Haar measure over O(D)).
    sign_b: Vec<u64>,
    /// Vector dimensionality the CALLER supplies. May be < `effective_dims`
    /// (rounded up internally) but never larger.
    dims: u32,
    /// Internal dim (next power of 2 ≥ max(dims, 64)). Power of 2 because
    /// the FHT butterfly requires it; max with 64 because the popcount
    /// kernel needs whole u64 words.
    effective_dims: u32,
    /// RNG seed that produced `sign_a` / `sign_b`. Retained so a node
    /// recovering a segment can regenerate the sign vectors bit-identically
    /// from the persisted seed.
    seed: u64,
    /// IVF centroids, stored flat K × `dims`. Empty ⇒ no centering
    /// (legacy behaviour). For K clusters, `centroids[k*dims..(k+1)*dims]`
    /// is the k-th centroid. `encode` picks the nearest centroid by L2,
    /// stores `cluster_id` on the code, and encodes the residual
    /// `r = x − centroids[cluster_id]` after rotation. `encode_query`
    /// precomputes `c_dot_q[k] = <centroids[k], q>` for every k so the
    /// asymmetric distance reconstruction is a single array lookup per
    /// neighbour visit.
    ///
    /// K=1 is global-mean centering (cheap, marginal recall lift on data
    /// with non-zero mean direction). K=16 is the size the SIGMOD 2024
    /// reference RaBitQ-Library uses on its public benchmarks — residuals
    /// inside each cluster have ~30% smaller L2 norm than residuals
    /// against a single global mean, so the sign-bit code captures
    /// sharper structure and the cheap distance is closer to truth,
    /// shrinking the gap to the dual-precision rerank path.
    #[serde(default)]
    centroids: Vec<f32>,
    /// `‖centroids[k]‖₂` per cluster (K entries). Empty when un-centered.
    /// Used by the cosine reconstruction
    /// `d_norm² = c_norms[k]² + 2·radial + r_norm²` where k = cluster_id.
    #[serde(default)]
    c_norms: Vec<f32>,
    /// Number of IVF clusters (`centroids.len() / dims`). Carried
    /// separately so the dispatch can read it without dividing on every
    /// call. 0 when un-centered (legacy).
    #[serde(default)]
    n_clusters: u32,
}

/// 1-bit code of a single vector plus the scalars needed by the distance estimator.
///
/// Memory: `D/8 + 12` bytes per vector (e.g. 140 B at D=1024 vs 4096 B f32 = ~29× compression).
///
/// `signed_sum` is precomputed at encode time so the asymmetric paper-Equation-20
/// kernel only needs the bit-plane AND+popcounts plus a multiply-add per code
/// (Chroma's `single_bit.rs` does the same: header carries `signed_sum: i32`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RaBitQCode {
    /// Sign-bit code packed into `D/64` u64 words (LSB-first within each word).
    /// `code[i / 64]` bit `(i % 64)` corresponds to dimension `i`. Backed by
    /// [`CodeWords`] (inline up to 4 u64 = covers `D ≤ 256`) so the hot
    /// HNSW search path reads the bits directly from the struct slot
    /// instead of dereferencing a heap pointer per visit.
    pub code: CodeWords,
    /// `‖x‖₂` — L2 norm of the original (pre-rotation) vector. Doubles as `norm`
    /// in the chroma-style distance reconstruction (we run with cluster
    /// centroid c=0 in pure HNSW, so the residual r equals the data vector d).
    pub norm: f32,
    /// `<x', e>` where `e` is a fixed unit vector — feeds the asymmetric
    /// correction in the symmetric code×code helper.
    pub cross_term: f32,
    /// `Σ sign(R·x)[i]` precomputed = `2·popcount(code) − D_eff`. Used by the
    /// asymmetric distance kernel (paper Eq. 20) to recover `<g, r_q>` without
    /// re-popcounting the stored code per query. Default 0 for codes that were
    /// serialized before this field existed; recompute on first load.
    #[serde(default)]
    pub signed_sum: i32,
    /// `correction = <g, n>` where `g[i] = ±0.5` is the sign-coded rotated
    /// vector and `n = R·x / ‖x‖` is the unit-normalized rotated vector.
    /// Equals `0.5 · ‖R·x‖_1 / ‖x‖_2` after the rotation step in `encode`.
    ///
    /// This is the per-vector scaling factor the paper Eq. 20 estimator needs
    /// to make `<g, r_q>` comparable across DIFFERENT stored vectors. Without
    /// it, `<g_a, r_q>` and `<g_b, r_q>` mix incompatible scales (each vector
    /// has its own `‖R·x‖_1`) and the HNSW heap ranking across nodes is
    /// effectively random — recall plateau at the LSH asymptote no matter
    /// what bit width the query side uses. Chroma stores the same field in
    /// its `CodeHeader1Bit`.
    ///
    /// Default `1.0` for codes that predate this field (treat as unscaled);
    /// the next encode pass overwrites with the true value.
    #[serde(default = "default_correction")]
    pub correction: f32,
    /// `radial = <r, centroids[cluster_id]>` where `r = x − centroids[cluster_id]`
    /// is the residual against the assigned cluster. Default 0.0 means
    /// "no IVF" (legacy codes pre-K=1 — read as `‖d‖ = norm`).
    #[serde(default)]
    pub radial: f32,
    /// Index into the IVF centroid array for the cluster this vector was
    /// assigned to. Default 0 for legacy codes (treat as "always cluster
    /// 0" — matches K=1 behaviour where there's only one cluster). For
    /// K > 1, the cosine reconstruction reads
    /// `c_norms[cluster_id]` and `query.c_dot_q[cluster_id]` to recover
    /// the centroid contribution to `<d, q>` and `‖d‖`.
    #[serde(default)]
    pub cluster_id: u16,
}

fn default_correction() -> f32 {
    1.0
}

impl RaBitQCode {
    /// Memory size of the code on the wire / in RAM. Excludes any allocator overhead.
    pub fn size_bytes(&self) -> usize {
        // 8 bytes for f32 norm + f32 cross_term + 4 bytes for signed_sum i32
        self.code.len() * 8 + 12
    }
}

/// Pre-quantized query in the asymmetric RaBitQ distance kernel (paper §3.3.2,
/// Eq. 20). The query is rotated, residualized into the `[v_l, v_r]` range,
/// quantized to `B_Q = 4` bits per dim, then expanded into 4 bit planes packed
/// into `D_eff / 64` u64 words each.
///
/// Memory: `4 · D_eff/8 + 12` bytes per query. The 4× memory vs the 1-bit code
/// pays for ~4-bit query resolution, which lifts the cosine-distance estimator
/// error bound from `O(1/√(D/B_Q))` (pure XOR popcount) to `O(1/√D)` per paper
/// Theorem 3.2 — the difference between recall=0.17 plateau and 0.85+ on glove.
///
/// Reused across thousands of `compute_distance` calls per search, so the 4×
/// encoding cost is amortized. Encoding itself is `O(D² + B_Q·D)`: the matrix-
/// vector rotation dominates; the bit-plane expansion is a tight scalar loop.
#[derive(Debug, Clone)]
pub struct RaBitQQuery {
    /// `B_Q = 4` bit planes. `planes[j]` holds bit `j` of every quantized
    /// dimension, packed into `D_eff / 64` u64 words. Plane 0 = LSB.
    ///
    /// Inline-up-to-4-u64-words storage per plane — covers every
    /// `effective_dims ≤ 256` (the common HNSW workload, including
    /// glove-100 padded to 128 = 2 words/plane) without a heap
    /// allocation. The 4 heap allocations the previous `[Vec<u64>; 4]`
    /// shape paid per `encode_query` showed up at ~4.5% of total bench
    /// cycles on the 14ec191 profile (10k queries × 4 allocations =
    /// 40k allocator round-trips on the search hot path); SmallVec
    /// inline keeps the planes on the caller's stack frame and lets the
    /// allocator slot stay cold.
    pub planes: [CodeWords; 4],
    /// `v_l = min_i(r_q[i])` — bottom of the per-query quantization range.
    pub v_l: f32,
    /// `delta = (v_r − v_l) / (2^B_Q − 1) = (max − min) / 15`. Step size of
    /// the per-query 4-bit quantizer.
    pub delta: f32,
    /// `Σ_i q_u[i]` — sum of the 4-bit-quantized query coordinates. Folds
    /// into the paper-Eq. 20 `signed_dot_qu = 2·packed_dot_qu − sum_q_u`.
    pub sum_q_u: i32,
    /// `‖q‖₂` — original query L2 norm. Carried for callers that want a
    /// scale-aware inner product or distance (we don't need it for the
    /// monotonic-only HNSW ranking, but keeping it makes the API honest).
    pub norm: f32,
    /// Per-cluster `c_dot_q[k] = <centroids[k], query>` precomputed once
    /// per search. The asymmetric distance reconstruction reads
    /// `c_dot_q[code.cluster_id] + r_dot_r_q` for the inner product term.
    /// Empty when the codec runs without IVF (legacy un-centered).
    pub c_dot_q: Vec<f32>,
}

impl RaBitQQuery {
    /// Memory size of the query encoding. 4 bit planes × D_eff/8 bytes +
    /// 4 f32/i32 scalars.
    pub fn size_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.len() * 8).sum::<usize>() + 12
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
        // Round up to the next power of 2 ≥ 64. Power of 2 is mandatory
        // for the FHT butterfly; ≥64 keeps the popcount kernel on whole
        // u64 words. The encoder pads inputs with zeros up to this width;
        // padding stays consistent across vectors so codes remain
        // bitwise-comparable in one index regardless of caller `dims`.
        let effective_dims = dims.max(64).next_power_of_two();

        let words = effective_dims as usize / 64;
        let mut rng = Xorshift64Star::new(seed);

        // Two independent Rademacher sign vectors. Two rounds of
        // {sign-flip → FHT} approximate uniform Haar measure on O(D)
        // (Kac walk in the Hadamard limit). One bit per dimension, packed
        // LSB-first across u64 words.
        let mut sign_a = vec![0u64; words];
        let mut sign_b = vec![0u64; words];
        for slot in sign_a.iter_mut() {
            *slot = rng.next_u64();
        }
        for slot in sign_b.iter_mut() {
            *slot = rng.next_u64();
        }

        Self {
            sign_a,
            sign_b,
            dims,
            effective_dims,
            seed,
            centroids: Vec::new(),
            c_norms: Vec::new(),
            n_clusters: 0,
        }
    }

    /// Build a calibrated `RaBitQParams` with K IVF centroids supplied
    /// flat (`centroids[k*dims..(k+1)*dims]` = k-th centroid). Encoding
    /// picks the nearest centroid per vector and stores residuals against
    /// it; the asymmetric distance reconstruction reads:
    ///
    /// ```text
    /// <d, q>    = c_dot_q[cluster_id] + ‖r‖ · g_dot_r_q / correction
    /// ‖d‖²      = c_norms[cluster_id]² + 2·radial + ‖r‖²
    /// cos_dist  = 1 − <d, q> / (‖d‖ · ‖q‖)
    /// ```
    ///
    /// vs un-centered, this lifts cosine-search recall on data with
    /// non-trivial cluster structure (glove, sentence-transformers,
    /// OpenAI embeddings). K=16 is the size the SIGMOD 2024 reference
    /// RaBitQ-Library uses on its glove benchmarks — residuals within a
    /// 16-cluster IVF have ~30% smaller L2 norm than residuals against a
    /// single global mean, so the sign-bit code is correspondingly
    /// sharper.
    ///
    /// # Panics
    ///
    /// Panics if `centroids.len() != n_clusters * dims` or
    /// `n_clusters == 0` or `dims == 0`.
    pub fn calibrate_with_centroids(
        dims: u32,
        seed: u64,
        centroids: Vec<f32>,
        n_clusters: u32,
    ) -> Self {
        assert!(n_clusters > 0, "RaBitQParams: n_clusters must be non-zero");
        assert_eq!(
            centroids.len(),
            (n_clusters as usize) * (dims as usize),
            "RaBitQParams::calibrate_with_centroids: centroids must be n_clusters * dims flat",
        );
        let mut base = Self::calibrate(dims, seed);
        let d = dims as usize;
        let mut c_norms = Vec::with_capacity(n_clusters as usize);
        for k in 0..n_clusters as usize {
            let slice = &centroids[k * d..(k + 1) * d];
            let n: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
            c_norms.push(n);
        }
        base.centroids = centroids;
        base.c_norms = c_norms;
        base.n_clusters = n_clusters;
        base
    }

    /// Convenience: run K-means Lloyd on a training sample to derive K
    /// centroids, then build the calibrated params with them. Suitable
    /// for HNSW auto-calibration where the caller has the first
    /// `calibration_threshold` vectors available.
    ///
    /// Uses K-means++ for initialisation (seeded by the same `seed` so
    /// the centroids are deterministic in `(dims, seed, training)`) and
    /// caps iterations at 12 — empirically Lloyd converges in 5-8
    /// iterations on D≤1024 data, and 12 is a safe upper bound that
    /// keeps calibration well under one second even at N=100K.
    ///
    /// # Panics
    ///
    /// Panics if `training.is_empty()`, `n_clusters == 0`, or any
    /// training vector's length differs from `dims as usize`.
    pub fn calibrate_with_kmeans(
        dims: u32,
        seed: u64,
        training: &[Vec<f32>],
        n_clusters: u32,
    ) -> Self {
        assert!(!training.is_empty(), "RaBitQParams: empty training set");
        assert!(n_clusters > 0, "RaBitQParams: n_clusters must be non-zero");
        let d = dims as usize;
        for v in training {
            assert_eq!(
                v.len(),
                d,
                "RaBitQParams::calibrate_with_kmeans: training vector dim mismatch"
            );
        }
        // K must not exceed N; clamp down silently. For tiny indexes we
        // degenerate to "one centroid per training point" which is still
        // a valid (if useless) IVF — the caller chose K, we honour it
        // as a ceiling.
        let k = (n_clusters as usize).min(training.len());

        let centroids_flat = kmeans_lloyd(training, d, k, seed);
        Self::calibrate_with_centroids(dims, seed, centroids_flat, k as u32)
    }

    /// In-place FHT-Kac rotation: `x ← FHT · S_b · FHT · S_a · x`, where
    /// `S_a`, `S_b` are the persisted Rademacher sign vectors and FHT is
    /// the un-normalised Hadamard butterfly. The composite is exactly
    /// orthonormal — `‖R·x‖₂ = ‖x‖₂` to within f32 precision — and runs
    /// in `Θ(D log D)` time, `Θ(1)` extra space.
    ///
    /// `x.len()` MUST equal `self.effective_dims` (caller pads with zeros).
    fn fht_kac_in_place(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.effective_dims as usize);
        apply_sign_flip(x, &self.sign_a);
        fht_in_place_unscaled(x);
        apply_sign_flip(x, &self.sign_b);
        fht_in_place_unscaled(x);
        // Two passes of un-normalised FHT inject a factor of D into ‖x‖².
        // Divide by D to restore orthonormality (‖R·x‖ = ‖x‖).
        let inv_d = 1.0 / self.effective_dims as f32;
        for v in x.iter_mut() {
            *v *= inv_d;
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

    /// `‖centroids[cluster_id]‖₂`. Returns 0.0 when the codec runs
    /// without IVF (`n_clusters == 0`) so the HNSW rerank identity
    /// `‖x‖² = c_norm² + 2·radial + ‖r‖²` collapses to `‖x‖ = ‖r‖` =
    /// `code.norm` for legacy un-centered codes.
    pub fn c_norm(&self, cluster_id: u16) -> f32 {
        if self.n_clusters == 0 {
            0.0
        } else {
            self.c_norms[cluster_id as usize]
        }
    }

    /// Number of IVF clusters. 0 means the codec runs un-centered.
    pub fn n_clusters(&self) -> u32 {
        self.n_clusters
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

        let d_eff = self.effective_dims as usize;
        let d_user = self.dims as usize;
        let words = d_eff / 64;

        // IVF assignment: when `self.centroids` is populated, pick the
        // nearest centroid by L2, encode the residual r = x − c[id], and
        // store `cluster_id + radial`. n_clusters = 0 ⇒ legacy un-centered
        // path (cluster_id = 0, radial = 0, residual = vector).
        let centered = self.n_clusters > 0;
        let cluster_id: u16 = if centered {
            let mut best_k = 0u16;
            let mut best_d = f32::INFINITY;
            for k in 0..self.n_clusters as usize {
                let c = &self.centroids[k * d_user..(k + 1) * d_user];
                let d = l2_sq(vector, c);
                if d < best_d {
                    best_d = d;
                    best_k = k as u16;
                }
            }
            best_k
        } else {
            0
        };

        let mut residual: Vec<f32> = if centered {
            let c =
                &self.centroids[(cluster_id as usize) * d_user..(cluster_id as usize + 1) * d_user];
            vector.iter().zip(c).map(|(v, c)| v - c).collect()
        } else {
            vector.to_vec()
        };

        // ‖r‖ (or ‖x‖ if no centering) and radial = <r, c[cluster_id]>.
        let mut norm_sq = 0.0f32;
        let mut radial = 0.0f32;
        if centered {
            let c =
                &self.centroids[(cluster_id as usize) * d_user..(cluster_id as usize + 1) * d_user];
            for (i, &v) in residual.iter().enumerate() {
                norm_sq += v * v;
                radial += v * c[i];
            }
        } else {
            for &v in &residual {
                norm_sq += v * v;
            }
        }
        let norm = norm_sq.sqrt();

        // Zero-pad to effective_dims, then rotate via FHT-Kac.
        residual.resize(d_eff, 0.0);
        self.fht_kac_in_place(&mut residual);
        let rotated = residual; // rename for clarity below

        // Single pass over the rotated vector: pack sign bits, sum + abs-sum
        // for the asymmetric kernel scalars. SmallVec is sized at 4 u64
        // inline; covers D ≤ 256 with no heap allocation.
        let mut code: CodeWords = SmallVec::from_elem(0u64, words);
        let mut sum_rotated = 0.0f32;
        let mut sum_abs_rotated = 0.0f32;
        for (i, &v) in rotated.iter().enumerate() {
            sum_rotated += v;
            sum_abs_rotated += v.abs();
            if v > 0.0 {
                code[i / 64] |= 1u64 << (i % 64);
            }
        }
        let inv_sqrt_d = 1.0 / (d_eff as f32).sqrt();

        // Σ_i sign(R·r)[i] precomputed once at encode time. Used by the
        // asymmetric kernel (Eq. 20) without re-popcounting the stored
        // code per query: `signed_sum = 2·popcount(code) − D_eff`.
        let popcount: u32 = code.iter().map(|w| w.count_ones()).sum();
        let signed_sum = 2 * popcount as i32 - d_eff as i32;

        // correction = <g, n> = 0.5 · ‖R·r‖_1 / ‖r‖_2 (see RaBitQCode docs).
        let correction = if norm > f32::EPSILON {
            0.5 * sum_abs_rotated / norm
        } else {
            1.0
        };

        RaBitQCode {
            code,
            norm,
            cross_term: sum_rotated * inv_sqrt_d,
            signed_sum,
            correction,
            radial,
            cluster_id,
        }
    }

    /// Encode a query vector for the asymmetric paper-Equation-20 distance
    /// kernel. Quantizes the rotated query to `B_Q = 4` bits per dim, then
    /// transposes into 4 bit planes packed as u64 words.
    ///
    /// This is the "right" 1-bit-data × 4-bit-query path the RaBitQ paper
    /// (§3.3.2) and Chroma's `single_bit.rs` use. Our previous symmetric
    /// 1-bit×1-bit `estimate_cosine_distance(code, code)` short-circuits this
    /// at the cost of an `O(1/√D)` → `O(1/√(D/4))` error blow-up — which is
    /// what made glove-100-angular plateau at recall=0.17.
    ///
    /// # Panics
    ///
    /// Panics if `vector.len() != self.dims as usize`.
    pub fn encode_query(&self, vector: &[f32]) -> RaBitQQuery {
        assert_eq!(
            vector.len(),
            self.dims as usize,
            "RaBitQParams::encode_query: dimension mismatch"
        );

        let d_eff = self.effective_dims as usize;
        let d_user = self.dims as usize;
        let words = d_eff / 64;

        // 1. Rotate the query through the same FHT-Kac composite the data
        // side uses. Zero-pad beyond user_dims so the rotation produces a
        // consistent r_q across all encode paths.
        //
        // IVF (K ≥ 1): we ROTATE the raw query (not any residual) so the
        // popcount kernel approximates <R·r, R·q> = <r, q>. The centroid
        // contribution `<c_k, q>` is per-cluster and precomputed once
        // here as `c_dot_q[k]`; the distance reconstruction picks the
        // right entry by `code.cluster_id`.
        let mut norm_sq = 0.0f32;
        for &v in vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();
        let c_dot_q: Vec<f32> = if self.n_clusters > 0 {
            let mut out = Vec::with_capacity(self.n_clusters as usize);
            for k in 0..self.n_clusters as usize {
                let c = &self.centroids[k * d_user..(k + 1) * d_user];
                let mut s = 0.0f32;
                for (i, &v) in vector.iter().enumerate() {
                    s += c[i] * v;
                }
                out.push(s);
            }
            out
        } else {
            Vec::new()
        };
        let mut r_q = vec![0.0f32; d_eff];
        r_q[..d_user].copy_from_slice(vector);
        self.fht_kac_in_place(&mut r_q);

        // 2. Per-query 4-bit linear quantization with min/max from the
        // rotated query itself. v_l = min, v_r = max, delta = range / 15.
        let mut v_l = r_q[0];
        let mut v_r = r_q[0];
        for &v in &r_q {
            if v < v_l {
                v_l = v;
            }
            if v > v_r {
                v_r = v;
            }
        }
        let range = (v_r - v_l).max(f32::EPSILON);
        let delta = range / 15.0;
        let inv_delta = 15.0 / range;

        // 3. Quantize and transpose to bit-planes simultaneously. Each q_u[i]
        // is a 4-bit integer 0..=15; bit j of q_u[i] lands in planes[j] at
        // position i. sum_q_u accumulates Σ q_u[i] for the Eq. 20 correction.
        // Inline-allocate the four bit-planes via SmallVec — for D_eff ≤ 256
        // (the common case, includes glove-100 padded to 128) each plane
        // fits in the SmallVec's 4-u64 inline buffer, no heap touch on the
        // per-query encode path.
        let mut planes: [CodeWords; 4] = [
            CodeWords::from_elem(0u64, words),
            CodeWords::from_elem(0u64, words),
            CodeWords::from_elem(0u64, words),
            CodeWords::from_elem(0u64, words),
        ];
        let mut sum_q_u: i32 = 0;
        for (i, &v) in r_q.iter().enumerate() {
            let qf = ((v - v_l) * inv_delta).round();
            let q_u = qf.clamp(0.0, 15.0) as u32;
            sum_q_u += q_u as i32;
            let word = i / 64;
            let bit = i % 64;
            for (j, plane) in planes.iter_mut().enumerate() {
                if ((q_u >> j) & 1) != 0 {
                    plane[word] |= 1u64 << bit;
                }
            }
        }

        RaBitQQuery {
            planes,
            v_l,
            delta,
            sum_q_u,
            norm,
            c_dot_q,
        }
    }

    /// Estimate `<g, r_q>` between a stored 1-bit data code and a 4-bit-plane
    /// query using paper Equation 20.
    ///
    /// Returns the inner-product estimate; for cosine distance on unit-norm
    /// vectors callers convert via `dist = -estimate` (monotonic — HNSW
    /// heap-order preserving) or `dist ≈ 1 − estimate / (norm_x · norm_q)`
    /// (scale-aware, slower).
    ///
    /// # Algorithm (paper §3.3.2, our convention `g[i] = ±0.5`)
    ///
    /// ```text
    /// packed_dot_qu = Σ_{j=0..4} 2^j · popcount(code AND planes[j])
    /// signed_dot_qu = 2·packed_dot_qu − sum_q_u
    /// <g, r_q>      = 0.5 · (delta · signed_dot_qu + v_l · signed_sum)
    /// ```
    ///
    /// Four AND+popcount rounds, one per bit plane, summed with `<<j` weights.
    /// On i9-9900K (AVX-512 + VPOPCNTDQ) each plane is one 8-u64 pass; the
    /// four planes share the data code in L1 so the entire kernel is ~4×
    /// cycles vs the legacy XOR popcount but with paper-accurate recall.
    pub fn estimate_inner_product_q(&self, code: &RaBitQCode, query: &RaBitQQuery) -> f32 {
        debug_assert_eq!(
            code.code.len(),
            query.planes[0].len(),
            "code/query plane length mismatch"
        );

        // Fused 4-plane AND+popcount: one pass over the data code, AND
        // against all four query bit planes in tight succession. Replaces
        // four separate `and_popcount` calls — one cache pass instead of
        // four, one dispatch instead of four. Profile on i9-9900K had
        // this kernel at 21% of search cycles via the unfused path; the
        // fused variant pulls the redundant passes out entirely.
        let (pop0, pop1, pop2, pop3) = popcount::and_popcount_4planes(
            &code.code,
            &query.planes[0],
            &query.planes[1],
            &query.planes[2],
            &query.planes[3],
        );
        let packed_dot_qu: i64 =
            pop0 as i64 + ((pop1 as i64) << 1) + ((pop2 as i64) << 2) + ((pop3 as i64) << 3);

        // signed_dot_qu = 2·packed_dot_qu − sum_q_u — turns the 0/1 sign
        // bits into ±1 signs without re-walking the code.
        let signed_dot_qu = 2 * packed_dot_qu - query.sum_q_u as i64;

        // Eq. 20 closed form. signed_sum is precomputed on the code.
        0.5 * (query.delta * signed_dot_qu as f32 + query.v_l * code.signed_sum as f32)
    }

    /// Cosine distance using the asymmetric 4-bit-query kernel.
    ///
    /// Implements the chroma `rabitq_distance_query` reconstruction (paper
    /// §3.2) — `<g, r_q>` alone is NOT a valid cross-vector distance because
    /// each stored code carries its own `correction = <g, n>` scaling factor.
    /// The corrected estimate is:
    ///
    /// ```text
    /// r_dot_r_q ≈ ‖r‖ · <g, r_q> / correction      // per chroma Eq.
    /// <d, q>    = c_dot_q + radial + r_dot_r_q     // pure HNSW: c=0,
    ///                                                 radial=0, c_dot_q=0
    ///                                                 → <d,q> = r_dot_r_q
    /// ‖d‖²      = c_norm² + 2·radial + ‖r‖²        // pure HNSW: = ‖x‖²
    /// cos_dist  = 1 − <d,q> / (‖d‖ · ‖q‖)
    /// ```
    ///
    /// For pure HNSW (no IVF centroid, c=0, r=d) this simplifies to:
    /// `cos_dist = 1 − (norm · g_dot_r_q / correction) / (norm · q.norm)
    ///           = 1 − g_dot_r_q / (correction · q.norm)`
    ///
    /// This is the path the HNSW heap actually needs. Returning naked
    /// `-<g, r_q>` (what an earlier revision did) produces monotonic-only
    /// values WITHIN one query, but mixes per-vector scales ACROSS the
    /// neighbour set — which is what kept glove recall pinned to ~0.23 at
    /// N=50k even after switching from XOR popcount to paper Eq. 20.
    pub fn estimate_cosine_distance_q(&self, code: &RaBitQCode, query: &RaBitQQuery) -> f32 {
        let g_dot_r_q = self.estimate_inner_product_q(code, query);

        // Chroma `rabitq_distance_query` reconstruction:
        //   r_dot_r_q = norm · g_dot_r_q / correction
        //   d_dot_q   = c_dot_q + r_dot_r_q
        //   d_norm²   = c_norm² + 2·radial + norm²
        //   cos_dist  = 1 − d_dot_q / (‖d‖ · ‖q‖)
        //
        // When IVF is OFF (`n_clusters == 0`), `code.radial`,
        // `query.c_dot_q[]` are empty and `self.c_norm()` returns 0 — the
        // formula collapses to the legacy
        // `cos_dist = 1 − g_dot_r_q / (correction · q_norm)`.
        let correction = if code.correction.abs() > f32::EPSILON {
            code.correction
        } else {
            return -g_dot_r_q;
        };
        let r_dot_r_q = code.norm * g_dot_r_q / correction;
        let c_dot_q = if (code.cluster_id as usize) < query.c_dot_q.len() {
            query.c_dot_q[code.cluster_id as usize]
        } else {
            0.0
        };
        let c_norm = self.c_norm(code.cluster_id);
        let d_dot_q = c_dot_q + r_dot_r_q;
        let d_norm_sq = c_norm * c_norm + 2.0 * code.radial + code.norm * code.norm;
        let denom = d_norm_sq.sqrt() * query.norm.max(f32::EPSILON);
        if denom.abs() < f32::EPSILON {
            return -g_dot_r_q;
        }
        1.0 - d_dot_q / denom
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

        // Same effective_dims rationale as `encode`: zero-pad to power-of-2
        // width, then rotate through the shared FHT-Kac composite.
        let d_eff = self.effective_dims as usize;
        let d_user = self.dims as usize;
        let levels_count = 1u8 << bits;

        let mut norm_sq = 0.0f32;
        for &v in vector {
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt().max(f32::EPSILON);

        let inv_norm = 1.0 / norm;
        let mut rotated = vec![0.0f32; d_eff];
        rotated[..d_user].copy_from_slice(vector);
        self.fht_kac_in_place(&mut rotated);
        let mut levels = vec![0u8; d_eff];
        let mut sum_rotated = 0.0f32;
        let inv_sqrt_d = 1.0 / (d_eff as f32).sqrt();

        for (i, &v) in rotated.iter().enumerate() {
            sum_rotated += v;
            levels[i] = quantize_normal(v * inv_norm, bits);
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
    ///
    /// Used by the test harness to drive the Gaussian sampler. Lives behind
    /// `#[cfg(test)]` because production calibration now consumes raw u64
    /// words directly into the Rademacher sign buffers (`sign_a`, `sign_b`)
    /// without going through the f32 → Box-Muller pipeline.
    #[cfg(test)]
    fn next_f32(&mut self) -> f32 {
        // Take top 24 bits, scale to [0, 1).
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// Standard normal N(0, 1) via Box-Muller transform. Test-only —
    /// production uses raw u64 sign masks for FHT-Kac.
    #[cfg(test)]
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
        // FHT-Kac stores the two sign vectors instead of a dense rotation
        // matrix; same seed must reproduce both bit-identical so a recovered
        // segment can re-derive the rotation from the persisted seed alone.
        let v = vec![0.123_f32; 128];
        assert_eq!(p1.encode(&v).code, p2.encode(&v).code);
    }

    #[test]
    fn rotation_is_approximately_orthonormal() {
        // Probe the composite rotation R = FHT · S_b · FHT · S_a indirectly:
        // it preserves L2 norm to within f32 precision (the defining property
        // of an orthonormal transform).
        let p = RaBitQParams::calibrate(64, 12345);
        let d = p.effective_dims() as usize;

        // Norm preservation under R for a handful of test inputs.
        for seed in [1u64, 7, 99, 1_000_000_007] {
            let mut rng = Xorshift64Star::new(seed);
            let mut v: Vec<f32> = (0..d).map(|_| rng.gaussian()).collect();
            let in_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            p.fht_kac_in_place(&mut v);
            let out_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                approx_eq(in_norm, out_norm, 1e-3),
                "norm not preserved: in={in_norm} out={out_norm}",
            );
        }

        // Sanity-shape the function on a one-hot input: rotation should
        // spread mass across all coordinates with comparable magnitudes
        // (the all-coordinates-equal property of Hadamard).
        for (i, k) in [(0usize, 1usize), (0, 13), (5, 17), (30, 63)] {
            let mut e_i = vec![0.0f32; d];
            e_i[i] = 1.0;
            p.fht_kac_in_place(&mut e_i);
            let mut e_k = vec![0.0f32; d];
            e_k[k] = 1.0;
            p.fht_kac_in_place(&mut e_k);
            let dot: f32 = e_i.iter().zip(&e_k).map(|(a, b)| a * b).sum();
            assert!(
                approx_eq(dot, 0.0, 1e-4),
                "columns ({i},{k}) dot = {dot}, expected ~0",
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
        // D=1024 → 128 bytes code + 8 bytes scalars + 4 bytes signed_sum = 140 bytes.
        // signed_sum was added for the asymmetric Eq. 20 kernel (paper §3.3.2).
        let p = RaBitQParams::calibrate(1024, 7);
        let v = vec![1.0f32; 1024];
        let c = p.encode(&v);
        assert_eq!(c.size_bytes(), 140);
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
