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

/// End-to-end through the build path: `bulk_build_cache_optimized` (build then
/// reorder) yields a working graph. Querying a stored point returns that point
/// as its own nearest neighbour — a remap bug in the wired reorder would break
/// this self-recall. (Identity of reorder vs a non-reordered index can't be
/// asserted across two builds: the parallel `insert_batch` build is
/// nondeterministic, so two separate builds differ regardless of reorder; the
/// reorder-preserves-results invariant is covered on a single index above.)
#[test]
fn bulk_build_cache_optimized_yields_working_graph() {
    let n: usize = 600;
    let dim: usize = 20;
    let items: Vec<(u64, Vec<f32>)> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim)
                .map(|d| {
                    let seed = (i.wrapping_mul(2_654_435_761).wrapping_add(d * 6_700_417)) as u32;
                    let bits = (seed ^ (seed >> 13)) & 0x00FF_FFFF;
                    (bits as f32 / 16_777_216.0) - 0.5
                })
                .collect();
            (i as u64, v)
        })
        .collect();

    let mut index = HnswIndex::new(HnswConfig {
        ef_search: 64,
        metric: VectorMetric::L2,
        max_dimensions: 65_536,
        ..Default::default()
    });
    index.bulk_build_cache_optimized(items.clone());

    let mut hits = 0;
    let probes = 50;
    for (id, v) in items.iter().step_by(n / probes).take(probes) {
        let top = index.search(v, 1);
        if top.first().is_some_and(|r| r.id == *id) {
            hits += 1;
        }
    }
    // This only asserts the end-to-end build path yields a functioning graph,
    // not its quality. The parallel follower insert assigns node levels from a
    // shared atomic RNG whose interleaving is nondeterministic, so the seeded
    // approximate bulk-build self-recall varies run to run (observed ~68-88%
    // here). A reorder-corrupted graph instead collapses self-recall toward
    // zero, so the bar sits well below the working-build floor and far above
    // collapse — it separates "working" from "corrupted". The exact-identity
    // correctness of reorder (results unchanged before/after) is covered
    // deterministically by `reorder_preserves_search_results`.
    assert!(
        hits as f64 >= probes as f64 * 0.5,
        "self-recall collapsed after reordered build: {hits}/{probes}"
    );
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
