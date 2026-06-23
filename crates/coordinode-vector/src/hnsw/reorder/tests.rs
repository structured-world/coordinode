use crate::hnsw::{HnswConfig, HnswIndex, VectorMetric};

fn build_index(n: usize, dim: usize) -> HnswIndex {
    let mut index = HnswIndex::new(HnswConfig {
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    });
    for i in 0..n {
        // Deterministic pseudo-random vectors (same generator the recall tests use).
        let v: Vec<f32> = (0..dim)
            .map(|d| {
                let seed = (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
                let bits = (seed ^ (seed >> 13)) & 0x00FF_FFFF;
                (bits as f32 / 16_777_216.0) - 0.5
            })
            .collect();
        index.insert(i as u64, v);
    }
    index
}

/// The permutation must be a bijection of `0..n` — every node gets exactly one
/// new index and no index is reused. A bug here would silently drop or alias
/// nodes when the permutation is applied.
#[test]
fn permutation_is_a_bijection() {
    let index = build_index(500, 16);
    let perm = index.compute_bfs_permutation();
    assert_eq!(perm.len(), 500);

    let mut seen = vec![false; perm.len()];
    for &new in &perm {
        assert!(new < perm.len(), "new index {new} out of range");
        assert!(!seen[new], "new index {new} assigned twice");
        seen[new] = true;
    }
    assert!(seen.iter().all(|&s| s), "every new index must be assigned");
}

/// The entry point is the BFS root, so it must map to new index 0 — the first
/// slot of the reordered arrays.
#[test]
fn entry_point_maps_to_index_zero() {
    let index = build_index(300, 16);
    let entry_old = index
        .entry_point_idx_for_test()
        .expect("non-empty index has an entry point");
    let perm = index.compute_bfs_permutation();
    assert_eq!(perm[entry_old], 0, "entry point must become new index 0");
}

/// BFS order: the entry point's direct layer-0 neighbours must receive the
/// lowest new indices after the root (they are visited in the first BFS level),
/// ahead of nodes only reachable through later levels.
#[test]
fn entry_neighbours_get_early_indices() {
    let index = build_index(400, 16);
    let entry_old = index.entry_point_idx_for_test().expect("entry point");
    let neighbours = index.layer0_neighbours_for_test(entry_old);
    assert!(!neighbours.is_empty(), "entry point should have neighbours");

    let perm = index.compute_bfs_permutation();
    // Root is 0; its `k` neighbours occupy a prefix of [1, 1 + k].
    let k = neighbours.len();
    for &nb in &neighbours {
        assert!(
            perm[nb as usize] <= k,
            "entry neighbour {nb} got new index {} beyond the first BFS level (k={k})",
            perm[nb as usize]
        );
    }
}

/// The core invariant: reorder only renumbers nodes (and remaps stored indices)
/// — the graph and all vectors are unchanged, so every query returns the exact
/// same top-K node ids in the same order, before and after. A remap bug in any
/// store (SoA arrays, layer-0 block, upper lists, entry point, id->idx) would
/// corrupt the graph and change results, failing this test.
#[test]
fn reorder_preserves_search_results() {
    let n = 800;
    let dim = 24;
    let mut index = build_index(n, dim);

    let queries: Vec<Vec<f32>> = (0..40)
        .map(|q| {
            (0..dim)
                .map(|d| {
                    let s = ((q * 97 + d * 31 + 7) as u32).wrapping_mul(2_654_435_761);
                    ((s ^ (s >> 15)) & 0x00FF_FFFF) as f32 / 16_777_216.0 - 0.5
                })
                .collect()
        })
        .collect();

    let k = 10;
    let before: Vec<Vec<u64>> = queries
        .iter()
        .map(|q| index.search(q, k).into_iter().map(|r| r.id).collect())
        .collect();

    index.reorder_for_cache_locality();

    for (q, want) in queries.iter().zip(&before) {
        let got: Vec<u64> = index.search(q, k).into_iter().map(|r| r.id).collect();
        assert_eq!(&got, want, "top-{k} ids must be identical after reorder");
    }
}

/// After reorder, every node id must still resolve (the id->idx map was rebuilt
/// for the new numbering) and the entry point must point at a valid in-range
/// node. Exact entry position is an implementation detail; the binding
/// correctness invariant is covered by `reorder_preserves_search_results`.
#[test]
fn reorder_rebuilds_id_map_and_entry() {
    let n = 300;
    let mut index = build_index(n, 16);
    index.reorder_for_cache_locality();

    for id in 0..n as u64 {
        let idx = index.idx_for_id_for_test(id);
        assert!(
            idx.is_some_and(|i| i < n),
            "id {id} must resolve to an in-range idx after reorder"
        );
    }
    let entry = index
        .entry_point_idx_for_test()
        .expect("non-empty index has an entry point");
    assert!(entry < n, "entry point idx {entry} out of range");
}

#[test]
fn empty_index_yields_empty_permutation() {
    let index = HnswIndex::new(HnswConfig {
        metric: VectorMetric::L2,
        max_dimensions: 64,
        ..Default::default()
    });
    assert!(index.compute_bfs_permutation().is_empty());
}
