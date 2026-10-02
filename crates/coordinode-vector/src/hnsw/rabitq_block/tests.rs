use super::*;

#[test]
fn align_up_basic() {
    assert_eq!(align_up(0, 8), 0);
    assert_eq!(align_up(1, 8), 8);
    assert_eq!(align_up(7, 8), 8);
    assert_eq!(align_up(8, 8), 8);
    assert_eq!(align_up(9, 8), 16);
    assert_eq!(align_up(3, 4), 4);
    assert_eq!(align_up(4, 4), 4);
    assert_eq!(align_up(5, 4), 8);
}

/// The block carries the code and its scalars only: no neighbour region and
/// no f32 copy, which live in the layer-0 store.
#[test]
fn stride_holds_code_and_scalars_only() {
    // dim 128, 1 bit: 16 code bytes, scalars at 16, 24 bytes, stride 40.
    let l = RabitqBlock::new(4, 128);
    assert_eq!(l.stride_bytes(), 40);
    assert_eq!(l.capacity(), 4);
    assert_eq!(l.rabitq_bits(), 1);
    assert_eq!(l.rabitq_byte_len(), 16);
    // dim 100, 1 bit: 13 code bytes, scalars at 16, stride 40.
    assert_eq!(RabitqBlock::new(1, 100).stride_bytes(), 40);
    // dim 3, 1 bit: 1 code byte, scalars at 4, end 28, stride 32.
    assert_eq!(RabitqBlock::new(1, 3).stride_bytes(), 32);
}

#[test]
fn new_with_bits_changes_rabitq_byte_budget() {
    let l1 = RabitqBlock::new_with_rabitq_bits(2, 128, 1);
    let l2 = RabitqBlock::new_with_rabitq_bits(2, 128, 2);
    let l4 = RabitqBlock::new_with_rabitq_bits(2, 128, 4);
    // 1-bit: 128/8 = 16; 2-bit: 256/8 = 32; 4-bit: 512/8 = 64.
    assert_eq!(l1.rabitq_byte_len(), 16);
    assert_eq!(l2.rabitq_byte_len(), 32);
    assert_eq!(l4.rabitq_byte_len(), 64);
    assert!(l1.stride_bytes() < l2.stride_bytes());
    assert!(l2.stride_bytes() < l4.stride_bytes());
    assert_eq!(l1.rabitq_bits(), 1);
    assert_eq!(l2.rabitq_bits(), 2);
    assert_eq!(l4.rabitq_bits(), 4);
}

/// The 1-bit search reads the code as `&[u64]`, so every node's code must
/// start 8-aligned.
#[test]
fn every_code_slot_is_eight_aligned() {
    let l = RabitqBlock::new(5, 100);
    for idx in 0..5 {
        // SAFETY: idx < 5.
        let p = unsafe { l.rabitq(idx) }.as_ptr();
        assert_eq!(p as usize % 8, 0, "code of node {idx} misaligned");
    }
}

#[test]
fn rabitq_scalars_round_trip() {
    let layer = RabitqBlock::new(4, 128);
    let scalars = RaBitQScalars {
        norm: 1.234_5,
        cross_term: -0.5,
        signed_sum: 17,
        correction: 0.875,
        radial: -2.0,
        cluster_id: 13,
        _pad: 0,
    };
    // SAFETY: idx < 4.
    unsafe {
        layer.set_rabitq_scalars(0, scalars);
        layer.set_rabitq_scalars(3, RaBitQScalars::default());
        assert_eq!(layer.rabitq_scalars(0), scalars);
        assert_eq!(layer.rabitq_scalars(3), RaBitQScalars::default());
        // Untouched idx stays zero.
        assert_eq!(layer.rabitq_scalars(1), RaBitQScalars::default());
    }
}

/// Writing a node's code and scalars leaves both intact and does not spill
/// into the neighbouring nodes' blocks.
#[test]
fn code_and_scalars_do_not_overlap_neighbouring_nodes() {
    let layer = RabitqBlock::new(4, 64);
    let scalars = RaBitQScalars {
        norm: 3.5,
        cross_term: 2.25,
        signed_sum: -42,
        correction: 0.5,
        radial: 1.0,
        cluster_id: 7,
        _pad: 0,
    };
    let code: Vec<u8> = (0..8).map(|i| i ^ 0xA5).collect(); // 1-bit at dim=64 -> 8 bytes
    // SAFETY: idx < 4, code matches rabitq_bytes (8).
    unsafe {
        layer.set_rabitq_scalars(2, scalars);
        layer.set_rabitq(2, &code);

        assert_eq!(layer.rabitq_scalars(2), scalars);
        assert_eq!(layer.rabitq(2), &code[..]);
        for other in [1, 3] {
            assert!(layer.rabitq(other).iter().all(|&b| b == 0));
            assert_eq!(layer.rabitq_scalars(other), RaBitQScalars::default());
        }
    }
}

#[test]
fn rabitq_round_trip() {
    let layer = RabitqBlock::new(4, 128); // rabitq_bytes = 16
    let code_a: Vec<u8> = (0..16).collect();
    let code_b: Vec<u8> = (200..216).collect();
    // SAFETY: idx < 4, code lengths match rabitq_bytes (=16).
    unsafe {
        layer.set_rabitq(0, &code_a);
        layer.set_rabitq(2, &code_b);
        assert_eq!(layer.rabitq(0), &code_a[..]);
        assert_eq!(layer.rabitq(2), &code_b[..]);
        // Untouched node still zero.
        assert!(layer.rabitq(1).iter().all(|&b| b == 0));
    }
}

#[test]
fn large_dim_four_bit_round_trip() {
    let layer = RabitqBlock::new_with_rabitq_bits(2, 1024, 4);
    let code: Vec<u8> = (0..512).map(|i| i as u8).collect(); // 1024 * 4 / 8
    // SAFETY: idx < 2, len matches.
    unsafe {
        layer.set_rabitq(1, &code);
        assert_eq!(layer.rabitq(1), &code[..]);
        assert!(layer.rabitq(0).iter().all(|&b| b == 0));
    }
}

#[test]
#[should_panic(expected = "rabitq_bits must be in 1..=4")]
fn new_with_bits_rejects_zero() {
    let _ = RabitqBlock::new_with_rabitq_bits(2, 64, 0);
}

#[test]
#[should_panic(expected = "rabitq_bits must be in 1..=4")]
fn new_with_bits_rejects_five() {
    let _ = RabitqBlock::new_with_rabitq_bits(2, 64, 5);
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn new_rejects_zero_capacity() {
    let _ = RabitqBlock::new(0, 64);
}

#[test]
#[should_panic(expected = "dim must be > 0")]
fn new_rejects_zero_dim() {
    let _ = RabitqBlock::new(4, 0);
}
