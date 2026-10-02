//! External node id to internal index, shared by concurrent inserts.
//!
//! The map is split into shards by id, each behind its own reader-writer
//! lock, so inserts of different ids rarely meet and a lookup never waits for
//! a writer on another shard. A shard lock is held only for one map
//! operation, never across graph work.

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

/// Shards; a power of two so the shard is picked with a mask.
const SHARDS: usize = 64;

/// Sharded `id -> idx` map. See the module doc.
pub(super) struct IdMap {
    shards: Box<[RwLock<FxHashMap<u64, usize>>]>,
}

impl IdMap {
    /// An empty map sized for about `capacity` ids.
    pub(super) fn with_capacity(capacity: usize) -> Self {
        let per_shard = capacity.div_ceil(SHARDS);
        Self {
            shards: (0..SHARDS)
                .map(|_| {
                    RwLock::new(FxHashMap::with_capacity_and_hasher(
                        per_shard,
                        Default::default(),
                    ))
                })
                .collect(),
        }
    }

    /// The shard holding `id`. Ids are spread by a multiplicative hash so
    /// sequential ids land on different shards.
    #[inline]
    fn shard(&self, id: u64) -> &RwLock<FxHashMap<u64, usize>> {
        let spread = id.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 58;
        // SHARDS == 64 == 1 << 6, so `spread` (the top 6 bits) is in range.
        &self.shards[spread as usize]
    }

    /// The index of `id`, if present.
    #[inline]
    pub(super) fn get(&self, id: u64) -> Option<usize> {
        self.shard(id).read().get(&id).copied()
    }

    /// Whether `id` is present.
    #[inline]
    pub(super) fn contains(&self, id: u64) -> bool {
        self.shard(id).read().contains_key(&id)
    }

    /// Point `id` at `idx`, returning the index it pointed at before.
    pub(super) fn insert(&self, id: u64, idx: usize) -> Option<usize> {
        self.shard(id).write().insert(id, idx)
    }

    /// Drop `id`, returning the index it pointed at.
    pub(super) fn remove(&self, id: u64) -> Option<usize> {
        self.shard(id).write().remove(&id)
    }

    /// Number of ids.
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.shards.iter().map(|shard| shard.read().len()).sum()
    }

    /// Remove every id.
    pub(super) fn clear(&mut self) {
        for shard in self.shards.iter_mut() {
            shard.get_mut().clear();
        }
    }

    /// Room for ids without growing, summed over the shards.
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.read().capacity())
            .sum()
    }
}

#[cfg(test)]
mod tests;
