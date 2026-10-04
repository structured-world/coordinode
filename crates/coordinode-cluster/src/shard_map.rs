//! Shard routing map: the per-label chunk-assignment table that resolves a
//! routing key to its shard (the shard-group coordinator's data model).
//!
//! A label's keyspace is partitioned into contiguous half-open chunk ranges over
//! a `u64` routing key (the NodeId, or `hash(prop)` / `range(prop)` per the
//! label's `PlacementPolicy`). Each chunk maps to the [`ShardId`] that owns it.
//! The table is stored per label at `schema:chunks:<label>` (MessagePack), so
//! every node resolves routing identically; its revision moves together with
//! the label schema revision that produced it.
//!
//! A single-shard deployment holds one chunk covering the whole key range;
//! a sharded one holds per-chunk assignments. The byte layout is the same,
//! only the number of chunks differs.

use serde::{Deserialize, Serialize};

use crate::types::ShardId;

/// A half-open routing-key range `[start, end)` assigned to one shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkRange {
    /// Inclusive lower bound of the routing key.
    pub start: u64,
    /// Exclusive upper bound. `u64::MAX` here means "to the end of the keyspace"
    /// (the whole-range chunk uses `end = u64::MAX`, treated as inclusive of
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

/// Why a list of chunks is not a routing table.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ChunkTableError {
    /// No chunk at all: some key would have no shard.
    #[error("a chunk table needs at least one chunk")]
    Empty,
    /// The first chunk does not start at key 0.
    #[error("the first chunk starts at {0}, not at 0")]
    FirstNotAtZero(u64),
    /// A chunk does not start where the previous one ended: a gap or an
    /// overlap.
    #[error("the chunk starting at {start} does not follow the previous end {previous_end}")]
    NotContiguous {
        /// End of the previous chunk.
        previous_end: u64,
        /// Start of this chunk.
        start: u64,
    },
    /// A chunk whose end is not past its start covers no key.
    #[error("the chunk [{start}, {end}) is empty")]
    EmptyRange {
        /// Start of the chunk.
        start: u64,
        /// End of the chunk.
        end: u64,
    },
    /// The last chunk does not reach the end of the keyspace.
    #[error("the last chunk ends at {0}, not at the end of the keyspace")]
    LastNotAtMax(u64),
    /// A chunk names shard 0, which is the NodeId hint sentinel and no shard.
    #[error("shard 0 names no shard")]
    SentinelShard,
}

/// Per-label routing table: ordered, gap-free, non-overlapping chunk
/// assignments covering `[0, u64::MAX]`, for one label at one schema
/// revision. Resolves a routing key to its shard.
///
/// Every value of this type tiles the keyspace: the constructors and the
/// decoder check it, so [`Self::shard_for`] is total.
///
/// # Examples
///
/// ```
/// use coordinode_cluster::{ChunkAssignmentTable, ShardId};
/// let table = ChunkAssignmentTable::single_shard("User", 1, ShardId::FIRST);
/// assert_eq!(table.shard_for(42), ShardId::FIRST);
/// let bytes = table.to_msgpack()?;
/// assert_eq!(ChunkAssignmentTable::from_msgpack(&bytes)?, table);
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ChunkTableRecord", into = "ChunkTableRecord")]
pub struct ChunkAssignmentTable {
    label: String,
    revision: u64,
    /// Chunks in ascending `range.start` order; together they tile the whole
    /// `u64` keyspace with no gap or overlap.
    chunks: Vec<ChunkAssignment>,
}

/// The stored form of a [`ChunkAssignmentTable`], checked on decode.
#[derive(Serialize, Deserialize)]
struct ChunkTableRecord {
    label: String,
    revision: u64,
    chunks: Vec<ChunkAssignment>,
}

impl TryFrom<ChunkTableRecord> for ChunkAssignmentTable {
    type Error = ChunkTableError;

    fn try_from(record: ChunkTableRecord) -> Result<Self, Self::Error> {
        Self::from_chunks(record.label, record.revision, record.chunks)
    }
}

impl From<ChunkAssignmentTable> for ChunkTableRecord {
    fn from(table: ChunkAssignmentTable) -> Self {
        Self {
            label: table.label,
            revision: table.revision,
            chunks: table.chunks,
        }
    }
}

impl ChunkAssignmentTable {
    /// The single-shard table of `label` at schema `revision`: the whole
    /// keyspace maps to `shard`.
    pub fn single_shard(label: impl Into<String>, revision: u64, shard: ShardId) -> Self {
        Self {
            label: label.into(),
            revision,
            chunks: vec![ChunkAssignment {
                range: ChunkRange {
                    start: 0,
                    end: u64::MAX,
                },
                shard,
            }],
        }
    }

    /// Build the table of `label` at schema `revision` from chunks sorted by
    /// `start`.
    ///
    /// # Errors
    ///
    /// The chunks do not tile `[0, u64::MAX]` exactly (first starts at 0, each
    /// starts where the previous ended, none is empty, last ends at
    /// `u64::MAX`), or one names shard 0.
    pub fn from_chunks(
        label: impl Into<String>,
        revision: u64,
        chunks: Vec<ChunkAssignment>,
    ) -> Result<Self, ChunkTableError> {
        let (Some(first), Some(last)) = (chunks.first(), chunks.last()) else {
            return Err(ChunkTableError::Empty);
        };
        if first.range.start != 0 {
            return Err(ChunkTableError::FirstNotAtZero(first.range.start));
        }
        if last.range.end != u64::MAX {
            return Err(ChunkTableError::LastNotAtMax(last.range.end));
        }
        for chunk in &chunks {
            if chunk.range.start >= chunk.range.end {
                return Err(ChunkTableError::EmptyRange {
                    start: chunk.range.start,
                    end: chunk.range.end,
                });
            }
            if chunk.shard.raw() == 0 {
                return Err(ChunkTableError::SentinelShard);
            }
        }
        for pair in chunks.windows(2) {
            if pair[0].range.end != pair[1].range.start {
                return Err(ChunkTableError::NotContiguous {
                    previous_end: pair[0].range.end,
                    start: pair[1].range.start,
                });
            }
        }
        Ok(Self {
            label: label.into(),
            revision,
            chunks,
        })
    }

    /// Resolve the shard owning `key`: the chunk with the largest start at or
    /// below it.
    pub fn shard_for(&self, key: u64) -> ShardId {
        // The first chunk starts at 0, so at least one start is <= key.
        let index = self.chunks.partition_point(|c| c.range.start <= key);
        debug_assert!(index > 0, "the chunks tile the keyspace from 0");
        self.chunks[index - 1].shard
    }

    /// The label this table routes.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// The label schema revision this table belongs to.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The chunk assignments in key order.
    pub fn chunks(&self) -> &[ChunkAssignment] {
        &self.chunks
    }

    /// Distinct shards referenced by the table (a scatter query's fan-out set).
    pub fn shards(&self) -> Vec<ShardId> {
        let mut out: Vec<ShardId> = self.chunks.iter().map(|c| c.shard).collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether this is the degenerate single-shard table.
    pub fn is_single_shard(&self) -> bool {
        self.chunks.len() == 1
    }

    /// Encode to MessagePack, the form stored at `schema:chunks:<label>`.
    ///
    /// # Errors
    ///
    /// The encoder fails.
    pub fn to_msgpack(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode from MessagePack, checking that the chunks tile the keyspace.
    ///
    /// # Errors
    ///
    /// The bytes are not a table, or its chunks are not a routing table
    /// ([`ChunkTableError`]).
    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
