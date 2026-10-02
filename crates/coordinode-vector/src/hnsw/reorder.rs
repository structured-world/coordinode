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
    pub(super) fn compute_bfs_permutation(&self) -> Vec<usize> {
        let n = self.node_len();
        let mut new_of_old = vec![usize::MAX; n];
        if n == 0 {
            return new_of_old;
        }

        let mut queue: VecDeque<usize> = VecDeque::with_capacity(n);
        let mut next_new = 0usize;
        let mut buf: Vec<u64> = Vec::new();

        // Seed BFS at the entry point; fall back to idx 0 if none is recorded
        // (e.g. a single-node index inserted before the entry point is set).
        // `for_search` is the accessor the real search uses as its start node and
        // the one apply remaps the entry from, so the entry deterministically
        // becomes new index 0.
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

    /// Reorder the index in place into BFS visit order for cache locality (O6).
    ///
    /// Renumbers every node by [`compute_bfs_permutation`](Self::compute_bfs_permutation)
    /// so graph-adjacent nodes become memory-adjacent across the SoA arrays and
    /// the contiguous layer-0 blocks. The graph is unchanged — only node indices
    /// are permuted and every stored neighbour index is remapped — so search
    /// results are identical before and after. A post-build, single-writer
    /// operation (`&mut self`); not safe to run concurrently with inserts or
    /// searches.
    pub(crate) fn reorder_for_cache_locality(&mut self) {
        let n = self.node_len();
        if n < 2 {
            return;
        }
        let new_of_old = self.compute_bfs_permutation();
        self.apply_permutation(&new_of_old);
    }

    /// Apply a `new_of_old` index permutation to every per-node store.
    ///
    /// Two-phase to avoid read-while-write aliasing in the byte blocks: phase A
    /// snapshots all per-node payload (under `&self`) with neighbour indices
    /// already remapped into the new index space; phase B rebuilds each store in
    /// the new order from those snapshots. `new_of_old` must be a bijection of
    /// `0..self.nodes.len()`.
    fn apply_permutation(&mut self, new_of_old: &[usize]) {
        let n = self.node_len();
        debug_assert_eq!(new_of_old.len(), n);

        // Inverse: old_of_new[new] = old. Drives the rebuild order.
        let mut old_of_new = vec![0usize; n];
        for (old, &new) in new_of_old.iter().enumerate() {
            old_of_new[new] = old;
        }

        // --- Step A: snapshot per-old payload (neighbours remapped) ---
        // Layer-0 f32 + neighbours, indexed by OLD idx.
        let mut l0_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut l0_nbrs: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut nb_buf: Vec<u64> = Vec::new();
        for old in 0..n {
            l0_vecs.push(
                self.read_node_f32(old)
                    .map(<[f32]>::to_vec)
                    .unwrap_or_default(),
            );
            self.read_layer0_neighbours_into(old, &mut nb_buf);
            l0_nbrs.push(
                nb_buf
                    .iter()
                    .map(|&id| new_of_old[id as usize] as u32)
                    .collect(),
            );
        }

        // Upper-layer neighbours per OLD idx: outer = node, mid = layer, inner =
        // remapped neighbour ids. The node's id and norm travel with it.
        let mut upper: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n);
        let mut meta: Vec<(u64, f32)> = Vec::with_capacity(n);
        for old in 0..n {
            let levels = self.node_levels(old);
            let mut per_layer = Vec::with_capacity(levels - 1);
            for level in 1..levels {
                let mut snap = self.neighbours_at(old, level).snapshot();
                for id in snap.iter_mut() {
                    *id = new_of_old[*id as usize] as u64;
                }
                per_layer.push(snap);
            }
            upper.push(per_layer);
            meta.push((
                self.node_id(old),
                self.nodes().norm(old).unwrap_or_default(),
            ));
        }

        // Reconstruct the entry from `load` (the canonical packed representation
        // `try_promote` round-trips), remapping its idx. Using `for_search`'s
        // derived level here would mis-encode the entry and change search starts.
        let entry = self
            .entry_point
            .load()
            .map(|(level, idx)| (level, new_of_old[idx as usize] as u64));

        // --- Step B: rebuild every store in new order ---

        // Node store: every node rebuilt at its new index with its id, norm,
        // f32 vector, codes and remapped lists on every layer.
        if let Some(mut old_block) = self.data_level0.take() {
            let mut codes: Vec<_> = (0..n).map(|old| old_block.take_codes(old)).collect();
            let dim = old_block.dim();
            let m = old_block.m_max0();
            let cap = old_block.capacity().max(n);
            let has_f32 = old_block.has_f32();
            let mut nb = super::data_level0::DataLevel0Block::new(cap, m, dim);
            if !has_f32 {
                nb.drop_f32();
            }
            for (new, &old) in old_of_new.iter().enumerate() {
                let (id, norm) = meta[old];
                let per_layer = std::mem::take(&mut upper[old]);
                let (sq8, rabitq) = std::mem::take(&mut codes[old]);
                // SAFETY: new < n <= cap; `nb` is owned here, so nothing else
                // reads or writes it.
                unsafe {
                    nb.init_node(new, id, norm, per_layer.len());
                    nb.set_codes(new, sq8, rabitq);
                    if has_f32 && !l0_vecs[old].is_empty() {
                        nb.set_vector(new, &l0_vecs[old]);
                    }
                    nb.set_neighbours(new, &l0_nbrs[old]);
                    for (layer, ids) in per_layer.iter().enumerate() {
                        nb.upper(new, layer + 1).set(ids);
                    }
                }
            }
            self.data_level0 = Some(nb);
        }

        // The code block is a copy of the nodes' RaBitQ codes laid out for
        // the search fast path: refill it from the codes moved above.
        self.rabitq_block = None;
        if let (Some(dim), Some(last)) = (
            self.data_level0
                .as_ref()
                .map(super::data_level0::DataLevel0Block::dim),
            n.checked_sub(1),
        ) {
            self.ensure_rabitq_block(last, dim);
        }
        if let (Some(block), Some(store)) = (self.rabitq_block.as_mut(), self.data_level0.as_ref())
        {
            for idx in 0..n.min(block.capacity()) {
                if let Some(enc) = store.rabitq(idx) {
                    Self::install_rabitq(block, idx, enc);
                }
            }
        }

        // id -> idx map and entry point follow the new numbering.
        self.id_to_idx.clear();
        for (new, &old) in old_of_new.iter().enumerate() {
            self.id_to_idx.insert(meta[old].0, new);
        }
        self.entry_point = super::entry_point::EntryPoint::new();
        if let Some((level, idx)) = entry {
            self.entry_point.try_promote(level, idx);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
