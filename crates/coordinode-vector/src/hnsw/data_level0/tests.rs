use super::*;

#[test]
fn stride_matches_hnswlib_size_data_per_element_at_sift_128() {
    // hnswlib's `size_data_per_element_` for d=128, M_MAX0=64 is
    // 4 (count) + 64*4 (ids) + 128*4 (vec) = 772 B. Stride rounds
    // up to the next multiple of 8 — same 776 hnswlib uses on
    // alignment-strict configs.
    let block = DataLevel0Block::new(1, 64, 128);
    assert_eq!(block.stride(), 776);
    assert_eq!(block.vector_offset, 4 + 64 * 4);
}

#[test]
fn stride_matches_layout_at_glove_100() {
    // glove d=100 M_MAX0=64: 4 + 256 + 400 = 660 B, stride aligned
    // up to 664.
    let block = DataLevel0Block::new(1, 64, 100);
    assert_eq!(block.stride(), 664);
}

#[test]
fn neighbours_round_trip() {
    let block = DataLevel0Block::new(4, M_MAX0, 32);
    let ids = vec![10u32, 20, 30, 40, 50];

    // SAFETY: idx < capacity, ids.len() <= M_MAX0.
    unsafe {
        block.set_neighbours(2, &ids);
    }

    let mut out = Vec::new();
    // SAFETY: idx < capacity, no concurrent writer.
    unsafe {
        block.read_neighbours_into(2, &mut out);
    }
    assert_eq!(out, ids);

    // Other slots are still zero-count.
    unsafe {
        block.read_neighbours_into(0, &mut out);
    }
    assert!(out.is_empty());
}

#[test]
fn vector_round_trip_and_alignment() {
    let dim = 64;
    let mut block = DataLevel0Block::new(8, M_MAX0, dim);
    let v: Vec<f32> = (0..dim).map(|i| i as f32 * 0.25).collect();

    // SAFETY: idx < capacity, v.len() == dim.
    unsafe {
        block.set_vector(3, &v);
    }

    // SAFETY: idx < capacity.
    let ptr = unsafe { block.vector_ptr(3) };
    assert_eq!(ptr.align_offset(core::mem::align_of::<f32>()), 0);

    // SAFETY: ptr is f32-aligned and points to `dim` valid f32
    // values written by `set_vector`.
    let slice = unsafe { core::slice::from_raw_parts(ptr, dim) };
    assert_eq!(slice, v.as_slice());
}

#[test]
fn ensure_capacity_grows_and_preserves_existing_vectors() {
    let dim = 8;
    let mut block = DataLevel0Block::new(2, M_MAX0, dim);
    let v0: Vec<f32> = (0..dim).map(|i| i as f32).collect();
    let v1: Vec<f32> = (0..dim).map(|i| i as f32 + 100.0).collect();
    // SAFETY: idx < capacity, len == dim.
    unsafe {
        block.set_vector(0, &v0);
        block.set_vector(1, &v1);
    }
    assert_eq!(block.capacity(), 2);

    // Grow to fit idx 5 (beyond the initial capacity).
    block.ensure_capacity(6);
    assert!(block.capacity() >= 6, "capacity must grow to fit");

    // Existing vectors survive the reallocate-and-copy.
    // SAFETY: idx < capacity, ptr is f32-aligned for `dim` values.
    unsafe {
        let s0 = core::slice::from_raw_parts(block.vector_ptr(0), dim);
        let s1 = core::slice::from_raw_parts(block.vector_ptr(1), dim);
        assert_eq!(s0, v0.as_slice());
        assert_eq!(s1, v1.as_slice());
    }

    // The newly available slot is usable.
    let v5: Vec<f32> = (0..dim).map(|i| i as f32 + 200.0).collect();
    // SAFETY: idx 5 < capacity after growth, len == dim.
    unsafe {
        block.set_vector(5, &v5);
        let s5 = core::slice::from_raw_parts(block.vector_ptr(5), dim);
        assert_eq!(s5, v5.as_slice());
    }

    // No-op when already large enough — no shrink, no realloc.
    let cap = block.capacity();
    block.ensure_capacity(3);
    assert_eq!(block.capacity(), cap);
}

#[test]
fn drop_f32_shrinks_stride_and_preserves_neighbours() {
    let dim = 8;
    let mut block = DataLevel0Block::new(4, M_MAX0, dim);
    // SAFETY: idx < capacity, neighbour ids fit M_MAX0, vector len == dim.
    unsafe {
        block.set_neighbours(0, &[10, 20, 30]);
        block.set_neighbours(2, &[40]);
        block.set_vector(0, &[1.0f32; 8]);
    }
    assert!(block.has_f32());
    let stride_before = block.stride();

    block.drop_f32();

    assert!(!block.has_f32(), "f32 marked absent after drop");
    assert!(
        block.stride() < stride_before,
        "stride shrinks once the f32 slot is gone"
    );
    // Neighbours survive the re-layout into the smaller stride.
    // SAFETY: idx < capacity.
    unsafe {
        assert_eq!(block.neighbour_count(0), 3);
        assert_eq!(block.neighbour_count(2), 1);
        assert_eq!(block.neighbour_count(1), 0);
    }
    // Idempotent.
    block.drop_f32();
    assert!(!block.has_f32());
}

#[test]
fn payloads_do_not_alias_across_nodes() {
    let dim = 16;
    let mut block = DataLevel0Block::new(4, M_MAX0, dim);
    let v0: Vec<f32> = (0..dim).map(|i| i as f32).collect();
    let v1: Vec<f32> = (0..dim).map(|i| -(i as f32)).collect();

    // SAFETY: idx < capacity, vector len == dim.
    unsafe {
        block.set_vector(0, &v0);
        block.set_vector(1, &v1);
        block.set_neighbours(0, &[1, 2, 3]);
        block.set_neighbours(1, &[10, 20]);
    }

    let mut ns0 = Vec::new();
    let mut ns1 = Vec::new();
    // SAFETY: idx < capacity.
    unsafe {
        block.read_neighbours_into(0, &mut ns0);
        block.read_neighbours_into(1, &mut ns1);
    }
    assert_eq!(ns0, vec![1, 2, 3]);
    assert_eq!(ns1, vec![10, 20]);

    // SAFETY: vector_ptr borrows are non-overlapping per-node
    // payloads.
    let s0 = unsafe { core::slice::from_raw_parts(block.vector_ptr(0), dim) };
    let s1 = unsafe { core::slice::from_raw_parts(block.vector_ptr(1), dim) };
    assert_eq!(s0, v0.as_slice());
    assert_eq!(s1, v1.as_slice());
}

#[test]
fn neighbour_count_starts_at_zero() {
    let block = DataLevel0Block::new(2, M_MAX0, 8);
    // SAFETY: idx < capacity.
    let c = unsafe { block.neighbour_count(0) };
    assert_eq!(c, 0);
}

#[test]
fn prefetch_out_of_bounds_is_noop() {
    let block = DataLevel0Block::new(2, M_MAX0, 8);
    // Should not panic, just skip the prefetch hint.
    block.prefetch(99);
}

#[test]
#[should_panic(expected = "dim must be > 0")]
fn rejects_zero_dim() {
    let _ = DataLevel0Block::new(1, M_MAX0, 0);
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn rejects_zero_capacity() {
    let _ = DataLevel0Block::new(0, M_MAX0, 8);
}

#[test]
fn cas_append_grows_in_order() {
    let block = DataLevel0Block::new(2, M_MAX0, 8);
    // SAFETY: idx 0 < capacity.
    unsafe {
        assert!(block.cas_append_neighbour(0, 5));
        assert!(block.cas_append_neighbour(0, 7));
        assert!(block.cas_append_neighbour(0, 9));
        assert_eq!(block.neighbour_count(0), 3);
    }
    let mut out = Vec::new();
    // SAFETY: idx 0 < capacity.
    unsafe {
        block.read_neighbours_into(0, &mut out);
    }
    assert_eq!(out, vec![5, 7, 9]);
}

#[test]
fn cas_append_returns_false_when_full() {
    // m_max0 = 4 so the list fills quickly.
    let block = DataLevel0Block::new(1, 4, 8);
    // SAFETY: idx 0 < capacity.
    unsafe {
        for i in 0..4 {
            assert!(block.cas_append_neighbour(0, i), "append {i} should fit");
        }
        assert!(
            !block.cas_append_neighbour(0, 99),
            "append past m_max0 must fail"
        );
    }
    let mut out = Vec::new();
    // SAFETY: idx 0 < capacity.
    unsafe {
        block.read_neighbours_into(0, &mut out);
    }
    assert_eq!(out, vec![0, 1, 2, 3]);
}

#[test]
fn set_neighbours_then_cas_append_extends() {
    let block = DataLevel0Block::new(1, M_MAX0, 8);
    // SAFETY: idx 0 < capacity, ids fit m_max0.
    unsafe {
        block.set_neighbours(0, &[1, 2, 3]);
        assert!(block.cas_append_neighbour(0, 4));
    }
    let mut out = Vec::new();
    // SAFETY: idx 0 < capacity.
    unsafe {
        block.read_neighbours_into(0, &mut out);
    }
    assert_eq!(out, vec![1, 2, 3, 4]);
}

#[test]
fn cas_append_concurrent_writers_keep_count_consistent() {
    let block = DataLevel0Block::new(1, M_MAX0, 8);
    let block_ref = &block;
    let n_writers: u32 = 32;
    std::thread::scope(|s| {
        for w in 0..n_writers {
            s.spawn(move || {
                // SAFETY: idx 0 < capacity; concurrent cas_append is the
                // documented contract.
                unsafe {
                    assert!(block_ref.cas_append_neighbour(0, w));
                }
            });
        }
    });
    let mut out = Vec::new();
    // SAFETY: idx 0 < capacity.
    unsafe {
        assert_eq!(block.neighbour_count(0), n_writers);
        block.read_neighbours_into(0, &mut out);
    }
    out.sort_unstable();
    assert_eq!(out, (0..n_writers).collect::<Vec<_>>());
}

#[test]
fn cas_append_concurrent_append_and_snapshot_no_torn_state() {
    let block = DataLevel0Block::new(1, M_MAX0, 8);
    let block_ref = &block;
    std::thread::scope(|s| {
        // Appender fills the list one id at a time.
        s.spawn(move || {
            for i in 0..M_MAX0 as u32 {
                // SAFETY: idx 0 < capacity.
                unsafe {
                    block_ref.cas_append_neighbour(0, i);
                }
            }
        });
        // Snapshotter races it: every observed id must be a real appended
        // id (filtered EMPTY sentinel, atomic slot => no torn read), and
        // the count never exceeds m_max0.
        s.spawn(move || {
            let mut out = Vec::new();
            for _ in 0..2000 {
                // SAFETY: idx 0 < capacity.
                unsafe {
                    block_ref.read_neighbours_into(0, &mut out);
                }
                assert!(out.len() <= M_MAX0, "count overshoot");
                for &id in &out {
                    assert!((id as usize) < M_MAX0, "snapshot saw garbage id {id}");
                }
            }
        });
    });
    let mut out = Vec::new();
    // SAFETY: idx 0 < capacity.
    unsafe {
        block.read_neighbours_into(0, &mut out);
    }
    assert_eq!(out.len(), M_MAX0, "all appends land");
}
