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
}
