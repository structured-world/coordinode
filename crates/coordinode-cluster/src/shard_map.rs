//! Shard routing map: the per-label chunk-assignment table that resolves a
//! routing key to its shard (R200, the shard-group coordinator's data model).
//!
//! A label's keyspace is partitioned into contiguous half-open chunk ranges over
//! a `u64` routing key (the NodeId, or `hash(prop)` / `range(prop)` per the
//! label's [`PlacementPolicy`]). Each chunk maps to the [`ShardId`] that owns it.
//! The table is stored per label at `schema:chunks:<label>` (MessagePack) in the
//! metadata Raft group, so every node resolves routing identically.
//!
//! CE replica-set / single-shard deployments hold one chunk covering the whole
//! key range; EE (and CE sharded mode) hold per-chunk assignments. The byte
//! layout is identical across tiers — only the number of chunks differs.

use serde::{Deserialize, Serialize};

use crate::types::ShardId;

/// A half-open routing-key range `[start, end)` assigned to one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRange {
    /// Inclusive lower bound of the routing key.
    pub start: u64,
    /// Exclusive upper bound. `u64::MAX` here means "to the end of the keyspace"
    /// (the whole-range CE chunk uses `end = u64::MAX`, treated as inclusive of
    /// `u64::MAX` — see [`ChunkAssignmentTable::shard_for`]).
    pub end: u64,
}

impl ChunkRange {
    /// Whether `key` falls in `[start, end)`, with `end == u64::MAX` treated as
    /// covering `u64::MAX` itself (so a single `[0, u64::MAX]` chunk covers the
    /// entire keyspace including the max key).
    pub fn contains(&self, key: u64) -> bool {
        key >= self.start && (key < self.end || (self.end == u64::MAX && key == u64::MAX))
    }
}

/// One chunk's `(range, owning shard)` assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkAssignment {
    /// Routing-key range this chunk covers.
    pub range: ChunkRange,
    /// Shard that owns the chunk.
    pub shard: ShardId,
}

/// Per-label routing table: ordered, gap-free, non-overlapping chunk
/// assignments covering `[0, u64::MAX]`. Resolves a routing key to its shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkAssignmentTable {
    /// Chunks in ascending `range.start` order; together they tile the whole
    /// `u64` keyspace with no gap or overlap.
    chunks: Vec<ChunkAssignment>,
}

impl ChunkAssignmentTable {
    /// The CE single-shard table: the whole keyspace maps to `shard`. Matches the
    /// `{ranges: [(0, u64::MAX)]}` entry a CE deployment stores at
    /// `schema:chunks:<label>`.
    pub fn single_shard(shard: ShardId) -> Self {
        Self {
            chunks: vec![ChunkAssignment {
                range: ChunkRange {
                    start: 0,
                    end: u64::MAX,
                },
                shard,
            }],
        }
    }

    /// Build from ordered chunk assignments. Returns `None` unless the chunks are
    /// sorted by `start`, non-overlapping, and tile `[0, u64::MAX]` exactly
    /// (first starts at 0, each chunk's `start` equals the previous `end`, last
    /// ends at `u64::MAX`) — the routing-table invariant.
    pub fn from_chunks(chunks: Vec<ChunkAssignment>) -> Option<Self> {
        if chunks.is_empty() {
            return None;
        }
        if chunks.first()?.range.start != 0 {
            return None;
        }
        for pair in chunks.windows(2) {
            if pair[0].range.end != pair[1].range.start || pair[0].range.start >= pair[0].range.end
            {
                return None;
            }
        }
        let last = chunks.last()?;
        if last.range.start >= last.range.end || last.range.end != u64::MAX {
            return None;
        }
        Some(Self { chunks })
    }

    /// Resolve the shard owning `key`. Total over `u64` because the chunks tile
    /// the whole keyspace; the linear scan is fine for the small chunk counts of
    /// CE / early EE (binary search lands when chunk counts grow).
    pub fn shard_for(&self, key: u64) -> ShardId {
        self.chunks
            .iter()
            .find(|c| c.range.contains(key))
            .map_or(ShardId::ZERO, |c| c.shard)
    }

    /// The chunk assignments in key order.
    pub fn chunks(&self) -> &[ChunkAssignment] {
        &self.chunks
    }

    /// Distinct shards referenced by the table (a scatter query's fan-out set).
    pub fn shards(&self) -> Vec<ShardId> {
        let mut out: Vec<ShardId> = self.chunks.iter().map(|c| c.shard).collect();
        out.sort_unstable_by_key(|s| s.raw());
        out.dedup();
        out
    }

    /// Whether this is the degenerate single-shard table (CE replica-set mode).
    pub fn is_single_shard(&self) -> bool {
        self.chunks.len() == 1
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
