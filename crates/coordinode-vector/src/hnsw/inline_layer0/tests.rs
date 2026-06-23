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

#[test]
fn new_computes_stride_for_small_dim() {
    let l = InlineLayer0::new(4, 16, 128);
    // 16 * 8 = 128 neighbour bytes, +1 len -> aligned to 136 (rabitq_offset).
    // rabitq_bytes = 128/8 = 16 (1-bit), end = 152, aligned to 152
    // (rabitq_scalars_offset). scalars = 24, end = 176, aligned to 176
    // (f32_offset). f32_bytes = 128 * 4 = 512, end = 688, aligned to 688
    // (label_offset). label = 8, end = 696, stride aligned-up = 696.
    assert_eq!(l.stride_bytes(), 696);
    assert_eq!(l.capacity(), 4);
    assert_eq!(l.m_max0(), 16);
    assert_eq!(l.rabitq_bits(), 1);
}

#[test]
fn new_with_bits_changes_rabitq_byte_budget() {
    let l1 = InlineLayer0::new_with_rabitq_bits(2, 16, 128, 1);
    let l2 = InlineLayer0::new_with_rabitq_bits(2, 16, 128, 2);
    let l4 = InlineLayer0::new_with_rabitq_bits(2, 16, 128, 4);
    // 1-bit: 128/8 = 16; 2-bit: 256/8 = 32; 4-bit: 512/8 = 64.
    // Stride grows monotonically with bits (extra bytes between code and
    // the rest of the block).
    assert!(l1.stride_bytes() < l2.stride_bytes());
    assert!(l2.stride_bytes() < l4.stride_bytes());
    assert_eq!(l1.rabitq_bits(), 1);
    assert_eq!(l2.rabitq_bits(), 2);
    assert_eq!(l4.rabitq_bits(), 4);
}

#[test]
fn rabitq_scalars_round_trip() {
    let mut layer = InlineLayer0::new(4, 16, 128);
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

#[test]
fn rabitq_scalars_independent_of_other_payloads() {
    let mut layer = InlineLayer0::new(4, 16, 64);
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
    let vec: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
    // SAFETY: idx < 4, code matches rabitq_bytes (8), vec matches dim (64).
    unsafe {
        layer.set_rabitq_scalars(2, scalars);
        layer.set_rabitq(2, &code);
        layer.set_vector_f32(2, &vec);
        layer.set_label(2, 0xDEAD_BEEF);

        assert_eq!(layer.rabitq_scalars(2), scalars);
        assert_eq!(layer.rabitq(2), &code[..]);
        assert_eq!(layer.vector_f32(2), &vec[..]);
        assert_eq!(layer.label(2).load(Ordering::Relaxed), 0xDEAD_BEEF);
    }
}

#[test]
#[should_panic(expected = "rabitq_bits must be in 1..=4")]
fn new_with_bits_rejects_zero() {
    let _ = InlineLayer0::new_with_rabitq_bits(2, 16, 64, 0);
}

#[test]
#[should_panic(expected = "rabitq_bits must be in 1..=4")]
fn new_with_bits_rejects_five() {
    let _ = InlineLayer0::new_with_rabitq_bits(2, 16, 64, 5);
}

#[test]
fn new_zeros_payload() {
    let layer = InlineLayer0::new(4, 16, 128);
    // SAFETY: idx and slot bounds met explicitly below.
    unsafe {
        for idx in 0..4 {
            assert_eq!(layer.neighbour(idx, 0).load(Ordering::Relaxed), 0);
            assert_eq!(layer.neighbour(idx, 15).load(Ordering::Relaxed), 0);
            assert_eq!(layer.neighbour_len(idx).load(Ordering::Relaxed), 0);
            assert_eq!(layer.label(idx).load(Ordering::Relaxed), 0);
            assert!(layer.rabitq(idx).iter().all(|&b| b == 0));
            assert!(layer.vector_f32(idx).iter().all(|&v| v == 0.0));
        }
    }
}

#[test]
fn neighbours_round_trip_at_multiple_idx() {
    let layer = InlineLayer0::new(8, 16, 64);
    // SAFETY: all idx < 8 and slot < 16.
    unsafe {
        for idx in 0..8 {
            for slot in 0..16 {
                let value = ((idx as u64) << 32) | (slot as u64);
                layer.set_neighbour(idx, slot, value);
            }
        }
        for idx in 0..8 {
            for slot in 0..16 {
                let expected = ((idx as u64) << 32) | (slot as u64);
                let got = layer.neighbour(idx, slot).load(Ordering::Relaxed);
                assert_eq!(got, expected, "mismatch at idx={idx} slot={slot}");
            }
        }
    }
}

#[test]
fn neighbour_len_round_trip() {
    let layer = InlineLayer0::new(4, 32, 64);
    // SAFETY: idx < 4.
    unsafe {
        layer.set_neighbour_len(0, 5);
        layer.set_neighbour_len(1, 17);
        layer.set_neighbour_len(2, 32);
        layer.set_neighbour_len(3, 0);
        assert_eq!(layer.neighbour_len(0).load(Ordering::Relaxed), 5);
        assert_eq!(layer.neighbour_len(1).load(Ordering::Relaxed), 17);
        assert_eq!(layer.neighbour_len(2).load(Ordering::Relaxed), 32);
        assert_eq!(layer.neighbour_len(3).load(Ordering::Relaxed), 0);
    }
}

#[test]
fn drop_f32_preserves_neighbours_and_label() {
    let mut layer = InlineLayer0::new(4, 16, 64);
    // SAFETY: idx < 4, slot < m_max0.
    unsafe {
        layer.set_neighbour(0, 0, 111);
        layer.set_neighbour(0, 1, 222);
        layer.set_label(0, 0xABCD_0000_0000_1234);
        layer.set_neighbour(2, 0, 333);
        layer.set_label(2, 0x9999_8888_7777_6666);
    }
    assert!(layer.has_f32());

    layer.drop_f32();

    assert!(!layer.has_f32(), "f32 marked absent after drop");
    // Neighbours + label survive the re-layout (the label slides up into
    // the space the dropped f32 vector vacated).
    // SAFETY: idx < 4, slot < m_max0.
    unsafe {
        assert_eq!(layer.neighbour(0, 0).load(Ordering::Relaxed), 111);
        assert_eq!(layer.neighbour(0, 1).load(Ordering::Relaxed), 222);
        assert_eq!(
            layer.label(0).load(Ordering::Relaxed),
            0xABCD_0000_0000_1234
        );
        assert_eq!(layer.neighbour(2, 0).load(Ordering::Relaxed), 333);
        assert_eq!(
            layer.label(2).load(Ordering::Relaxed),
            0x9999_8888_7777_6666
        );
    }
    // Idempotent.
    layer.drop_f32();
    assert!(!layer.has_f32());
}

#[test]
fn label_round_trip() {
    let layer = InlineLayer0::new(4, 16, 64);
    // SAFETY: idx < 4.
    unsafe {
        layer.set_label(0, 0x1111_2222_3333_4444);
        layer.set_label(3, 0xFFFF_FFFF_FFFF_FFFE);
        assert_eq!(
            layer.label(0).load(Ordering::Relaxed),
            0x1111_2222_3333_4444
        );
        assert_eq!(
            layer.label(3).load(Ordering::Relaxed),
            0xFFFF_FFFF_FFFF_FFFE
        );
        // Untouched ids stay zero.
        assert_eq!(layer.label(1).load(Ordering::Relaxed), 0);
    }
}

#[test]
fn rabitq_round_trip() {
    let mut layer = InlineLayer0::new(4, 16, 128); // rabitq_bytes = 16
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
fn vector_f32_round_trip() {
    let mut layer = InlineLayer0::new(4, 16, 64);
    let vec_a: Vec<f32> = (0..64).map(|i| i as f32 * 0.125).collect();
    let vec_b: Vec<f32> = (0..64).map(|i| -1.0 - i as f32).collect();
    // SAFETY: idx < 4, vec lengths match dim (=64).
    unsafe {
        layer.set_vector_f32(0, &vec_a);
        layer.set_vector_f32(3, &vec_b);
        assert_eq!(layer.vector_f32(0), &vec_a[..]);
        assert_eq!(layer.vector_f32(3), &vec_b[..]);
        // Untouched node still zero.
        assert!(layer.vector_f32(1).iter().all(|&v| v == 0.0));
    }
}

#[test]
fn payload_writes_do_not_corrupt_neighbours() {
    let mut layer = InlineLayer0::new(4, 16, 128);
    // SAFETY: bounds and lens checked explicitly.
    unsafe {
        for slot in 0..16 {
            layer.set_neighbour(2, slot, 0xDEAD_BEEF_0000_0000 | slot as u64);
        }
        layer.set_neighbour_len(2, 16);
        layer.set_label(2, 0xCAFE_F00D);

        let code: Vec<u8> = (0..16).map(|i| i ^ 0x55).collect();
        let vec: Vec<f32> = (0..128).map(|i| i as f32).collect();
        layer.set_rabitq(2, &code);
        layer.set_vector_f32(2, &vec);

        for slot in 0..16 {
            let got = layer.neighbour(2, slot).load(Ordering::Relaxed);
            assert_eq!(got, 0xDEAD_BEEF_0000_0000 | slot as u64);
        }
        assert_eq!(layer.neighbour_len(2).load(Ordering::Relaxed), 16);
        assert_eq!(layer.label(2).load(Ordering::Relaxed), 0xCAFE_F00D);
    }
}

#[test]
fn large_dim_and_m_max0() {
    let mut layer = InlineLayer0::new(2, 64, 1024);
    let vec: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let code: Vec<u8> = (0..128).map(|i| i as u8).collect(); // 1024/8
                                                             // SAFETY: idx < 2, lens match.
    unsafe {
        layer.set_vector_f32(1, &vec);
        layer.set_rabitq(1, &code);
        assert_eq!(layer.vector_f32(1), &vec[..]);
        assert_eq!(layer.rabitq(1), &code[..]);
    }
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn new_rejects_zero_capacity() {
    let _ = InlineLayer0::new(0, 16, 64);
}

#[test]
#[should_panic(expected = "m_max0 must be > 0")]
fn new_rejects_zero_m_max0() {
    let _ = InlineLayer0::new(4, 0, 64);
}

#[test]
#[should_panic(expected = "dim must be > 0")]
fn new_rejects_zero_dim() {
    let _ = InlineLayer0::new(4, 16, 0);
}
