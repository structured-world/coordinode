//! XOR + popcount distance kernel for 1-bit codes (RaBitQ, binary embeddings).
//!
//! Runtime dispatch picks the fastest available implementation:
//! - x86_64 with `avx512vpopcntdq` (Ice Lake+, Zen 4+): 8 × 64-bit popcounts per instruction
//! - aarch64 with `neon`: `vcntq_u8` byte-popcount + horizontal sum
//! - scalar fallback: `u64::count_ones()` per word
//!
//! All paths are equivalent in result; SIMD paths are speed-only. The scalar
//! fallback is `core::*` only — no `std` dependency, suitable for no-std targets
//! once the rest of the crate goes alloc-only.
//
// no-std: scalar path already no-std-ready; SIMD dispatch uses `is_x86_feature_detected!`
//         (std-only macro). For pure no-std builds, drop dispatch and call `xor_popcount_scalar`
//         directly via a compile-time feature gate.

/// Compute `popcount(a XOR b)` summed across all words.
///
/// Both slices must be the same length. Returns the total Hamming distance
/// between the two bit-strings represented as `&[u64]`.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn xor_popcount(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "xor_popcount: slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vpopcntdq")
            && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: feature detection above gates the intrinsic; slices are same length.
            return unsafe { xor_popcount_avx512(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: feature detection above; NEON is baseline on aarch64 anyway.
            return unsafe { xor_popcount_neon(a, b) };
        }
    }

    xor_popcount_scalar(a, b)
}

/// Portable scalar implementation — `core::*` only.
#[inline]
pub fn xor_popcount_scalar(a: &[u64], b: &[u64]) -> u32 {
    let mut sum: u32 = 0;
    for i in 0..a.len() {
        sum += (a[i] ^ b[i]).count_ones();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn xor_popcount_avx512(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::{
        _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64, _mm512_xor_si512,
    };

    let len = a.len();
    let mut sum: i64 = 0;
    let mut i = 0;

    // Process 8 u64 = 512 bits per iteration.
    while i + 8 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let vx = _mm512_xor_si512(va, vb);
        let vp = _mm512_popcnt_epi64(vx);
        sum += _mm512_reduce_add_epi64(vp);
        i += 8;
    }

    // Scalar tail.
    while i < len {
        sum += (a[i] ^ b[i]).count_ones() as i64;
        i += 1;
    }

    sum as u32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn xor_popcount_neon(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::aarch64::{vaddvq_u8, vcntq_u8, veorq_u8, vld1q_u8};

    let len = a.len();
    let mut sum: u32 = 0;
    let mut i = 0;

    // Process 2 u64 = 128 bits = 16 bytes per iteration.
    while i + 2 <= len {
        let pa = a.as_ptr().add(i) as *const u8;
        let pb = b.as_ptr().add(i) as *const u8;
        let va = vld1q_u8(pa);
        let vb = vld1q_u8(pb);
        let vx = veorq_u8(va, vb);
        let vp = vcntq_u8(vx);
        sum += vaddvq_u8(vp) as u32;
        i += 2;
    }

    // Scalar tail for trailing odd word.
    while i < len {
        sum += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    sum
}

/// Compute `popcount(a AND b)` summed across all words.
///
/// Same shape as [`xor_popcount`] but with bitwise AND instead of XOR.
/// Used by the asymmetric RaBitQ distance kernel (paper Equation 20):
/// the data side is a 1-bit sign code, the query side is expanded into
/// `B_Q = 4` bit planes, and each plane's contribution is `<<j` weighted
/// AND+popcount against the data code. Four AND+popcount rounds per
/// distance call (one per plane) gives the full paper formula in place
/// of the pure-XOR shortcut, which is what unlocks recall on cosine
/// workloads beyond the LSH-level plateau.
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
#[inline]
pub fn and_popcount(a: &[u64], b: &[u64]) -> u32 {
    assert_eq!(a.len(), b.len(), "and_popcount: slice length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vpopcntdq")
            && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: feature detection above gates the intrinsic; same-length asserted.
            return unsafe { and_popcount_avx512(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON is baseline on aarch64; feature detection gates the call.
            return unsafe { and_popcount_neon(a, b) };
        }
    }

    and_popcount_scalar(a, b)
}

/// Portable scalar AND+popcount — `core::*` only.
#[inline]
pub fn and_popcount_scalar(a: &[u64], b: &[u64]) -> u32 {
    let mut sum: u32 = 0;
    for i in 0..a.len() {
        sum += (a[i] & b[i]).count_ones();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_popcount_avx512(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::x86_64::{
        _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    };

    let len = a.len();
    let mut sum: i64 = 0;
    let mut i = 0;

    while i + 8 <= len {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let vand = _mm512_and_si512(va, vb);
        let vp = _mm512_popcnt_epi64(vand);
        sum += _mm512_reduce_add_epi64(vp);
        i += 8;
    }

    while i < len {
        sum += (a[i] & b[i]).count_ones() as i64;
        i += 1;
    }

    sum as u32
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn and_popcount_neon(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::aarch64::{vaddvq_u8, vandq_u8, vcntq_u8, vld1q_u8};

    let len = a.len();
    let mut sum: u32 = 0;
    let mut i = 0;

    while i + 2 <= len {
        let pa = a.as_ptr().add(i) as *const u8;
        let pb = b.as_ptr().add(i) as *const u8;
        let va = vld1q_u8(pa);
        let vb = vld1q_u8(pb);
        let vand = vandq_u8(va, vb);
        let vp = vcntq_u8(vand);
        sum += vaddvq_u8(vp) as u32;
        i += 2;
    }

    while i < len {
        sum += (a[i] & b[i]).count_ones();
        i += 1;
    }

    sum
}

/// Fused 4-plane AND+popcount for the asymmetric RaBitQ kernel.
///
/// Reads the data code's u64 words **once** and ANDs each word against
/// the matching word from all four query bit planes in tight succession.
/// Returns `(pop0, pop1, pop2, pop3)` — the popcount sums per plane,
/// which the caller weights by `<<j` to form `packed_dot_qu`.
///
/// vs four independent `and_popcount` calls (one per plane), this
/// eliminates three cache-line passes over the code and three rounds
/// of dispatch overhead. On the HNSW search hot path
/// `estimate_inner_product_q` runs once per visited neighbour
/// (typically 2-10k visits per query), so the savings compound. Profile
/// on the i9-9900K (AVX2 + VPOPCNT-free, no AVX-512) showed
/// `estimate_inner_product_q` at 21% of search cycles before this fused
/// kernel landed — the dominant single function in the hot loop.
///
/// All five slices MUST have the same length.
///
/// # Panics
///
/// Panics if any plane length differs from `code.len()`.
#[inline]
pub fn and_popcount_4planes(
    code: &[u64],
    p0: &[u64],
    p1: &[u64],
    p2: &[u64],
    p3: &[u64],
) -> (u32, u32, u32, u32) {
    assert_eq!(
        p0.len(),
        code.len(),
        "and_popcount_4planes: plane0 length mismatch"
    );
    assert_eq!(
        p1.len(),
        code.len(),
        "and_popcount_4planes: plane1 length mismatch"
    );
    assert_eq!(
        p2.len(),
        code.len(),
        "and_popcount_4planes: plane2 length mismatch"
    );
    assert_eq!(
        p3.len(),
        code.len(),
        "and_popcount_4planes: plane3 length mismatch"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512vpopcntdq")
            && std::is_x86_feature_detected!("avx512f")
        {
            // SAFETY: feature detection gates the intrinsic.
            return unsafe { and_popcount_4planes_avx512(code, p0, p1, p2, p3) };
        }
    }

    and_popcount_4planes_scalar(code, p0, p1, p2, p3)
}

/// Portable scalar fused kernel — `core::*` only. The inner loop loads
/// each `code[i]` once and ANDs it against `p0[i]..p3[i]` in succession,
/// keeping the code word in a register the whole time.
#[inline]
pub fn and_popcount_4planes_scalar(
    code: &[u64],
    p0: &[u64],
    p1: &[u64],
    p2: &[u64],
    p3: &[u64],
) -> (u32, u32, u32, u32) {
    let mut s0 = 0u32;
    let mut s1 = 0u32;
    let mut s2 = 0u32;
    let mut s3 = 0u32;
    for i in 0..code.len() {
        let x = code[i];
        s0 += (x & p0[i]).count_ones();
        s1 += (x & p1[i]).count_ones();
        s2 += (x & p2[i]).count_ones();
        s3 += (x & p3[i]).count_ones();
    }
    (s0, s1, s2, s3)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn and_popcount_4planes_avx512(
    code: &[u64],
    p0: &[u64],
    p1: &[u64],
    p2: &[u64],
    p3: &[u64],
) -> (u32, u32, u32, u32) {
    use std::arch::x86_64::{
        _mm512_and_si512, _mm512_loadu_si512, _mm512_popcnt_epi64, _mm512_reduce_add_epi64,
    };

    let len = code.len();
    let mut s0: i64 = 0;
    let mut s1: i64 = 0;
    let mut s2: i64 = 0;
    let mut s3: i64 = 0;
    let mut i = 0;

    while i + 8 <= len {
        let vx = _mm512_loadu_si512(code.as_ptr().add(i) as *const _);
        let v0 = _mm512_loadu_si512(p0.as_ptr().add(i) as *const _);
        let v1 = _mm512_loadu_si512(p1.as_ptr().add(i) as *const _);
        let v2 = _mm512_loadu_si512(p2.as_ptr().add(i) as *const _);
        let v3 = _mm512_loadu_si512(p3.as_ptr().add(i) as *const _);
        s0 += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(vx, v0)));
        s1 += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(vx, v1)));
        s2 += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(vx, v2)));
        s3 += _mm512_reduce_add_epi64(_mm512_popcnt_epi64(_mm512_and_si512(vx, v3)));
        i += 8;
    }

    while i < len {
        let x = code[i];
        s0 += (x & p0[i]).count_ones() as i64;
        s1 += (x & p1[i]).count_ones() as i64;
        s2 += (x & p2[i]).count_ones() as i64;
        s3 += (x & p3[i]).count_ones() as i64;
        i += 1;
    }

    (s0 as u32, s1 as u32, s2 as u32, s3 as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_u64s(seed: u64, n: usize) -> Vec<u64> {
        // Tiny self-contained xorshift64* for reproducible test inputs.
        let mut s = seed.max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s.wrapping_mul(0x2545_F491_4F6C_DD1D)
            })
            .collect()
    }

    #[test]
    fn scalar_matches_naive() {
        let a = vec![0u64, 0xFFFF_FFFF_FFFF_FFFF, 0xAAAA_AAAA_AAAA_AAAA];
        let b = vec![0u64, 0u64, 0x5555_5555_5555_5555];
        // XOR: 0, FFFF.., FFFF.. → popcount 0 + 64 + 64 = 128.
        assert_eq!(xor_popcount_scalar(&a, &b), 128);
    }

    #[test]
    fn dispatch_matches_scalar_random() {
        // 1024-bit strings (16 u64 words) — matches d=1024 RaBitQ code size.
        let a = random_u64s(0xDEAD_BEEF, 16);
        let b = random_u64s(0xCAFE_F00D, 16);
        assert_eq!(xor_popcount(&a, &b), xor_popcount_scalar(&a, &b));
    }

    #[test]
    fn dispatch_matches_scalar_long() {
        // 8192-bit (128 u64) — exercises both the 8-word AVX-512 loop and the tail.
        let a = random_u64s(1, 128);
        let b = random_u64s(2, 128);
        assert_eq!(xor_popcount(&a, &b), xor_popcount_scalar(&a, &b));
    }

    #[test]
    fn dispatch_matches_scalar_odd_tail() {
        // 9 words — triggers tail handling in AVX-512 (8 + 1) and NEON (8 + 1) paths.
        let a = random_u64s(42, 9);
        let b = random_u64s(43, 9);
        assert_eq!(xor_popcount(&a, &b), xor_popcount_scalar(&a, &b));
    }

    #[test]
    fn identical_inputs_give_zero() {
        let a = random_u64s(7, 16);
        assert_eq!(xor_popcount(&a, &a), 0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mismatched_lengths_panic() {
        let a = vec![0u64; 3];
        let b = vec![0u64; 4];
        let _ = xor_popcount(&a, &b);
    }

    #[test]
    fn and_scalar_matches_naive() {
        let a = vec![0u64, 0xFFFF_FFFF_FFFF_FFFF, 0xAAAA_AAAA_AAAA_AAAA];
        let b = vec![0u64, 0xFFFF_FFFF_FFFF_FFFF, 0x5555_5555_5555_5555];
        // AND: 0, FFFF.., 0 → popcount 0 + 64 + 0 = 64.
        assert_eq!(and_popcount_scalar(&a, &b), 64);
    }

    #[test]
    fn and_dispatch_matches_scalar_random() {
        let a = random_u64s(0xDEAD_BEEF, 16);
        let b = random_u64s(0xCAFE_F00D, 16);
        assert_eq!(and_popcount(&a, &b), and_popcount_scalar(&a, &b));
    }

    #[test]
    fn and_dispatch_matches_scalar_long_and_tail() {
        let a = random_u64s(11, 128);
        let b = random_u64s(13, 128);
        assert_eq!(and_popcount(&a, &b), and_popcount_scalar(&a, &b));
        let a9 = random_u64s(101, 9);
        let b9 = random_u64s(103, 9);
        assert_eq!(and_popcount(&a9, &b9), and_popcount_scalar(&a9, &b9));
    }

    #[test]
    fn and_identical_inputs_match_popcount_self() {
        let a = random_u64s(7, 16);
        let expected: u32 = a.iter().map(|w| w.count_ones()).sum();
        assert_eq!(and_popcount(&a, &a), expected);
    }
}
