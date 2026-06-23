//! Cache-locality graph reordering (O6): renumber node indices in BFS
//! visitation order from the entry point so that nodes adjacent in the graph
//! land adjacent in memory. A search walks neighbours of a visited node next, so
//! BFS order maximises the chance the next visit is already in a resident cache
//! line — the SoA arrays + the contiguous `data_level0` block are all indexed by
//! node idx, so permuting the idx space directly improves their locality.
//!
//! Reference: "Graph Reordering for Cache-Efficient Near Neighbor Search"
//! (NeurIPS 2022) — BFS visit-order renumbering as the baseline ordering.
//!
//! This module computes the permutation; applying it to the index storage
//! (per-idx SoA arrays, `data_level0` neighbour ids, upper-layer lists, entry
//! point, id→idx map) lands incrementally on top of this.

use std::collections::VecDeque;

use super::HnswIndex;

impl HnswIndex {
    /// Compute a BFS visit-order permutation of node indices.
    ///
    /// Returns `new_of_old`, where `new_of_old[old_idx]` is the node's position
    /// in a breadth-first traversal of the layer-0 graph starting at the entry
    /// point. Nodes unreachable from the entry point retain their relative order
    /// and are appended after every reachable node. The result is always a
    /// bijection of `0..self.nodes.len()`.
    #[allow(
        dead_code,
        reason = "applied by the reorder-apply increment on the same plan"
    )]
    pub(super) fn compute_bfs_permutation(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut new_of_old = vec![usize::MAX; n];
        if n == 0 {
            return new_of_old;
        }

        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
        let mut next_new = 0usize;
        let mut buf: Vec<u64> = Vec::new();

        // Seed BFS at the entry point; fall back to idx 0 if none is recorded
        // (e.g. a single-node index inserted before the entry point is set).
        let start = self
            .entry_point
            .for_search()
            .map(|(_, idx)| idx)
            .unwrap_or(0);
        if start < n {
            new_of_old[start] = next_new;
            next_new += 1;
            queue.push_back(start);
        }

        while let Some(old) = queue.pop_front() {
            self.read_layer0_neighbours_into(old, &mut buf);
            for &nb in &buf {
                let nb = nb as usize;
                if nb < n && new_of_old[nb] == usize::MAX {
                    new_of_old[nb] = next_new;
                    next_new += 1;
                    queue.push_back(nb);
                }
            }
        }

        // Nodes unreachable from the entry point keep their relative order and
        // follow the reachable set, so the permutation stays a full bijection.
        for slot in new_of_old.iter_mut() {
            if *slot == usize::MAX {
                *slot = next_new;
                next_new += 1;
            }
        }
        debug_assert_eq!(
            next_new, n,
            "permutation must cover every node exactly once"
        );
        new_of_old
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
