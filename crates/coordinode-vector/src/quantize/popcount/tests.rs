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
