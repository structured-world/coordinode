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

/// `new_of_old` entry of a slot the reorder compacts away.
pub(super) const DROPPED: usize = usize::MAX;

impl HnswIndex {
    /// Compute a BFS visit-order numbering of the live nodes.
    ///
    /// Returns `new_of_old`, where `new_of_old[old_idx]` is the node's position
    /// in a breadth-first traversal of the layer-0 graph starting at the entry
    /// point, or [`DROPPED`] for a slot whose node is removed, replaced or
    /// free: those are compacted away. Live nodes unreachable from the entry
    /// point retain their relative order and follow every reachable one. The
    /// live nodes are numbered `0..live` without gaps.
    pub(super) fn compute_bfs_permutation(&self) -> Vec<usize> {
        let n = self.node_len();
        let mut new_of_old = vec![DROPPED; n];
        if n == 0 {
            return new_of_old;
        }
        let store = self.nodes();
        let live = |idx: usize| store.state(idx) == super::data_level0::NodeState::Live;
        // Every slot is visited once; only the live ones are numbered.
        let mut seen = vec![false; n];

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
            .map(|(idx, _top_level)| idx)
            .unwrap_or(0);
        if start < n {
            seen[start] = true;
            if live(start) {
                new_of_old[start] = next_new;
                next_new += 1;
            }
            queue.push_back(start);
        }

        // A retired node still bridges parts of the graph, so the walk goes
        // through it; it only takes no number.
        while let Some(old) = queue.pop_front() {
            self.read_layer0_neighbours_into(old, &mut buf);
            for &nb in &buf {
                let nb = nb as usize;
                if nb < n && !seen[nb] {
                    seen[nb] = true;
                    if live(nb) {
                        new_of_old[nb] = next_new;
                        next_new += 1;
                    }
                    queue.push_back(nb);
                }
            }
        }

        // Live nodes unreachable from the entry point keep their relative
        // order and follow the reachable set.
        for (old, slot) in new_of_old.iter_mut().enumerate() {
            if *slot == DROPPED && live(old) {
                *slot = next_new;
                next_new += 1;
            }
        }
        new_of_old
    }

    /// Reorder the index in place into BFS visit order for cache locality (O6).
    ///
    /// Renumbers every node by [`compute_bfs_permutation`](Self::compute_bfs_permutation)
    /// so graph-adjacent nodes become memory-adjacent across the SoA arrays and
    /// the contiguous layer-0 blocks. The slots of removed and replaced nodes
    /// are compacted away and every edge to them dropped; the live graph is
    /// otherwise unchanged (indices permuted, every stored neighbour index
    /// remapped), so search results are identical before and after. A
    /// post-build, single-writer operation (`&mut self`); not safe to run
    /// concurrently with inserts or searches.
    pub(crate) fn reorder_for_cache_locality(&mut self) {
        let n = self.node_len();
        if n < 2 {
            return;
        }
        let new_of_old = self.compute_bfs_permutation();
        self.apply_permutation(&new_of_old);
    }

    /// Apply a `new_of_old` numbering to every per-node store, dropping the
    /// slots it maps to [`DROPPED`].
    ///
    /// Two-phase to avoid read-while-write aliasing in the byte blocks: phase A
    /// snapshots the payload of every kept node (under `&self`) with neighbour
    /// indices already remapped into the new index space and edges to dropped
    /// slots removed; phase B rebuilds each store in the new order from those
    /// snapshots. The kept nodes must be numbered `0..kept` without gaps.
    fn apply_permutation(&mut self, new_of_old: &[usize]) {
        debug_assert_eq!(new_of_old.len(), self.node_len());
        let remap = |id: u64| {
            let new = new_of_old[id as usize];
            (new != DROPPED).then_some(new)
        };

        // Inverse: old_of_new[new] = old. Drives the rebuild order.
        let kept: Vec<(usize, usize)> = new_of_old
            .iter()
            .enumerate()
            .filter(|&(_, &new)| new != DROPPED)
            .map(|(old, &new)| (old, new))
            .collect();
        let n = kept.len();
        let mut old_of_new = vec![0usize; n];
        for &(old, new) in &kept {
            old_of_new[new] = old;
        }

        // --- Step A: snapshot every kept node's payload, by NEW idx ---
        // Layer-0 f32 + neighbours, upper-layer neighbours (outer = node,
        // mid = layer, inner = remapped ids), and the node's id and norm.
        let mut l0_vecs: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut l0_nbrs: Vec<Vec<u32>> = Vec::with_capacity(n);
        let mut upper: Vec<Vec<Vec<u64>>> = Vec::with_capacity(n);
        let mut meta: Vec<(u64, f32)> = Vec::with_capacity(n);
        let mut nb_buf: Vec<u64> = Vec::new();
        for &old in &old_of_new {
            l0_vecs.push(
                self.read_node_f32(old)
                    .map(<[f32]>::to_vec)
                    .unwrap_or_default(),
            );
            self.read_layer0_neighbours_into(old, &mut nb_buf);
            l0_nbrs.push(
                nb_buf
                    .iter()
                    .filter_map(|&id| remap(id))
                    .map(|new| new as u32)
                    .collect(),
            );
            let levels = self.node_levels(old);
            let per_layer = (1..levels)
                .map(|level| {
                    self.neighbours_at(old, level)
                        .snapshot()
                        .into_iter()
                        .filter_map(remap)
                        .map(|new| new as u64)
                        .collect()
                })
                .collect();
            upper.push(per_layer);
            meta.push((
                self.node_id(old),
                self.nodes().norm(old).unwrap_or_default(),
            ));
        }

        // Reconstruct the entry from `load` (the canonical packed representation
        // `try_promote` round-trips), remapping its idx. Using `for_search`'s
        // derived level here would mis-encode the entry and change search starts.
        // Should the entry name a dropped slot, it moves to the first kept
        // node at that node's own top layer: the entry point never names a
        // slot that no longer exists.
        let entry = match self.entry_point.load() {
            Some((level, idx)) => match remap(idx) {
                Some(new) => Some((level, new as u64)),
                None => (n > 0).then(|| ((upper[0].len()) as u8, 0)),
            },
            None => None,
        };

        // --- Step B: rebuild every store in new order ---

        // Node store: every kept node rebuilt at its new index with its id,
        // norm, f32 vector, codes and remapped lists on every layer.
        if let Some(mut old_block) = self.data_level0.take() {
            let mut codes: Vec<_> = old_of_new
                .iter()
                .map(|&old| old_block.take_codes(old))
                .collect();
            let dim = old_block.dim();
            let m = old_block.m_max0();
            let cap = old_block.capacity().max(n.max(1));
            let has_f32 = old_block.has_f32();
            let mut nb = super::data_level0::DataLevel0Block::new(cap, m, dim);
            if !has_f32 {
                nb.drop_f32();
            }
            for new in 0..n {
                let (id, norm) = meta[new];
                let per_layer = std::mem::take(&mut upper[new]);
                let (sq8, rabitq) = std::mem::take(&mut codes[new]);
                // SAFETY: new < n <= cap; `nb` is owned here, so nothing else
                // reads or writes it.
                unsafe {
                    nb.init_node(new, id, norm, per_layer.len());
                    nb.set_codes(new, sq8, rabitq);
                    if has_f32 && !l0_vecs[new].is_empty() {
                        nb.set_vector(new, &l0_vecs[new]);
                    }
                    nb.set_neighbours(new, &l0_nbrs[new]);
                    for (layer, ids) in per_layer.iter().enumerate() {
                        nb.upper(new, layer + 1).set(ids);
                    }
                }
                nb.set_state(new, super::data_level0::NodeState::Live);
            }
            self.data_level0 = std::sync::OnceLock::from(nb);
        }
        self.node_count = core::sync::atomic::AtomicUsize::new(n);
        self.live_count = core::sync::atomic::AtomicUsize::new(n);
        // Every retired slot is gone and the numbering changed: the slots
        // queued for reuse mean nothing any more.
        self.reclaim.reset();

        // The code block is a copy of the nodes' RaBitQ codes laid out for
        // the search fast path: refill it from the codes moved above.
        self.rabitq_block = std::sync::OnceLock::new();
        if let (Some(dim), Some(last)) = (
            self.data_level0
                .get()
                .map(super::data_level0::DataLevel0Block::dim),
            n.checked_sub(1),
        ) {
            self.ensure_rabitq_block(last, dim);
        }
        if let (Some(block), Some(store)) = (self.rabitq_block.get(), self.data_level0.get()) {
            for idx in 0..n.min(block.capacity()) {
                if let Some(enc) = store.rabitq(idx) {
                    // SAFETY: idx < block capacity; `&mut self` excludes
                    // every other reader and writer.
                    unsafe { Self::install_rabitq(block, idx, enc) };
                }
            }
        }

        // id -> idx map and entry point follow the new numbering.
        self.id_to_idx.clear();
        for (new, &(id, _)) in meta.iter().enumerate() {
            self.id_to_idx.insert(id, new);
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
