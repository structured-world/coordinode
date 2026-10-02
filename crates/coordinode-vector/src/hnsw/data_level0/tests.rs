use super::*;

/// Fresh counters for the list writes a test makes.
fn stats() -> Arc<PublicationStats> {
    Arc::new(PublicationStats::new())
}

/// The block holds only the vectors: a node's stride is its f32 vector
/// rounded up to 8 bytes, whatever `m_max0` is, because the neighbours live
/// in their published lists.
#[test]
fn stride_holds_the_vector_only() {
    // sift d=128: 512 B, already a multiple of 8.
    assert_eq!(DataLevel0Block::new(1, 64, 128).stride(), 512);
    assert_eq!(DataLevel0Block::new(1, 16, 128).stride(), 512);
    // glove d=100: 400 B.
    assert_eq!(DataLevel0Block::new(1, 64, 100).stride(), 400);
    // d=3: 12 B rounded up to 16.
    assert_eq!(DataLevel0Block::new(1, 64, 3).stride(), 16);
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
    let block = DataLevel0Block::new(8, M_MAX0, dim);
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

/// Growth adds segments and moves nothing: a vector and a list a reader holds
/// keep their address, and nodes in every segment round-trip.
#[test]
fn growth_keeps_addresses_and_fills_every_segment() {
    let dim = 8;
    let block = DataLevel0Block::new(3, M_MAX0, dim);
    let vector = |i: usize| -> Vec<f32> { (0..dim).map(|d| (i * 100 + d) as f32).collect() };
    // SAFETY: idx < capacity, len == dim, single writer, nothing reachable.
    unsafe {
        block.set_vector(0, &vector(0));
        block.set_vector(2, &vector(2));
        block.set_neighbours(2, &[7, 8]);
    }
    assert_eq!(block.capacity(), 3);
    // SAFETY: idx < capacity.
    let (v0, v2) = unsafe { (block.vector_ptr(0), block.vector_ptr(2)) };

    // Segments of 3, 6, 12: capacity 3 -> 6 -> 12 -> 24.
    block.ensure_capacity(20);
    assert_eq!(block.capacity(), 24);
    // SAFETY: idx < capacity.
    unsafe {
        assert_eq!(block.vector_ptr(0), v0, "node 0 moved on growth");
        assert_eq!(block.vector_ptr(2), v2, "node 2 moved on growth");
    }

    for i in 3..24 {
        // SAFETY: idx < capacity, len == dim, single writer.
        unsafe {
            block.set_vector(i, &vector(i));
            block.set_neighbours(i, &[i as u32]);
        }
    }
    let mut out = Vec::new();
    for i in [0usize, 2, 3, 5, 6, 11, 12, 23] {
        // SAFETY: idx < capacity; the vector is f32-aligned for `dim` values.
        unsafe {
            let got = core::slice::from_raw_parts(block.vector_ptr(i), dim);
            assert_eq!(got, vector(i).as_slice(), "vector of node {i}");
            assert_eq!(block.vector_ptr(i).align_offset(4), 0);
            block.read_neighbours_into(i, &mut out);
        }
        let expected: Vec<u32> = match i {
            0 => vec![],
            2 => vec![7, 8],
            _ => vec![i as u32],
        };
        assert_eq!(out, expected, "list of node {i}");
    }

    // Already large enough: a no-op.
    block.ensure_capacity(5);
    assert_eq!(block.capacity(), 24);
}

/// Concurrent growth installs each segment once and covers the request.
#[test]
fn concurrent_growth_installs_each_segment_once() {
    let block = DataLevel0Block::new(4, M_MAX0, 8);
    std::thread::scope(|s| {
        for t in 0..8 {
            let block = &block;
            s.spawn(move || block.ensure_capacity(40 + t));
        }
    });
    // 4 -> 8 -> 16 -> 32 -> 64.
    assert_eq!(block.capacity(), 64);
    // SAFETY: idx < capacity, single writer, nothing reachable.
    unsafe {
        block.set_neighbours(63, &[1]);
        assert_eq!(block.neighbour_count(63), 1);
    }
}

/// Dropping the vectors after growth frees them in every segment and keeps
/// every list; a segment installed afterwards carries no vectors.
#[test]
fn drop_f32_after_growth_keeps_lists() {
    let mut block = DataLevel0Block::new(2, M_MAX0, 8);
    block.ensure_capacity(6);
    // SAFETY: idx < capacity.
    unsafe {
        block.set_neighbours(1, &[3]);
        block.set_neighbours(5, &[4, 5]);
    }
    block.drop_f32();
    block.ensure_capacity(10);
    assert!(!block.has_f32());
    // SAFETY: idx < capacity.
    unsafe {
        assert_eq!(block.neighbour_count(1), 1);
        assert_eq!(block.neighbour_count(5), 2);
        assert_eq!(block.neighbour_count(9), 0);
    }
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
    // Neighbours survive: they never lived in the vector block.
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
    let block = DataLevel0Block::new(4, M_MAX0, dim);
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
    let stats = stats();
    // SAFETY: idx 0 < capacity.
    unsafe {
        assert!(block.cas_append_neighbour(0, 5, &stats));
        assert!(block.cas_append_neighbour(0, 7, &stats));
        assert!(block.cas_append_neighbour(0, 9, &stats));
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
    let stats = stats();
    // SAFETY: idx 0 < capacity.
    unsafe {
        for i in 0..4 {
            assert!(
                block.cas_append_neighbour(0, i, &stats),
                "append {i} should fit"
            );
        }
        assert!(
            !block.cas_append_neighbour(0, 99, &stats),
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
        assert!(block.cas_append_neighbour(0, 4, &stats()));
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
    let stats = stats();
    let stats_ref = &stats;
    let n_writers: u32 = 32;
    std::thread::scope(|s| {
        for w in 0..n_writers {
            s.spawn(move || {
                // SAFETY: idx 0 < capacity; concurrent cas_append is the
                // documented contract.
                unsafe {
                    assert!(block_ref.cas_append_neighbour(0, w, stats_ref));
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

/// A reader during a whole-list replace of a layer-0 node sees the old list
/// or the new one, never a list assembled from both. The writer alternates
/// two disjoint 16-id lists; any snapshot holding ids of both is a mix.
#[test]
fn replace_is_observed_whole_on_layer0() {
    let block = DataLevel0Block::new(1, M_MAX0, 8);
    let a: Vec<u32> = (1..=16).collect();
    let b: Vec<u32> = (101..=116).collect();
    // SAFETY: idx 0 < capacity, 16 <= m_max0.
    unsafe { block.set_neighbours(0, &a) };
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (block_ref, a_ref, b_ref, stop_ref) = (&block, &a, &b, &stop);
    std::thread::scope(|s| {
        s.spawn(move || {
            for i in 0..200_000 {
                let next = if i % 2 == 0 { b_ref } else { a_ref };
                // SAFETY: idx 0 < capacity, 16 <= m_max0.
                unsafe { block_ref.set_neighbours(0, next) };
            }
            stop_ref.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        for _ in 0..3 {
            s.spawn(move || {
                let mut out = Vec::new();
                while !stop_ref.load(std::sync::atomic::Ordering::Relaxed) {
                    // SAFETY: idx 0 < capacity.
                    unsafe { block_ref.read_neighbours_into(0, &mut out) };
                    assert!(
                        out == *a_ref || out == *b_ref,
                        "reader observed a mixed layer-0 list: {out:?}"
                    );
                }
            });
        }
    });
}

#[test]
fn cas_append_concurrent_append_and_snapshot_no_torn_state() {
    let block = DataLevel0Block::new(1, M_MAX0, 8);
    let block_ref = &block;
    let stats = stats();
    let stats_ref = &stats;
    std::thread::scope(|s| {
        // Appender fills the list one id at a time.
        s.spawn(move || {
            for i in 0..M_MAX0 as u32 {
                // SAFETY: idx 0 < capacity.
                unsafe {
                    block_ref.cas_append_neighbour(0, i, stats_ref);
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
