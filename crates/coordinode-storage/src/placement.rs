//! Placement-segment storage primitive.
//!
//! A *placement segment* is the atomic unit of placement, migration, split,
//! merge, and recovery: a logical key-range region of one partition, with size
//! and access statistics aggregated from the underlying engine. Its identity is
//! the logical key range, NOT any physical SST file, so a segment keeps a stable
//! [`SegmentId`] and [`KeyRange`] across the continuous compaction churn of a
//! long-lived cluster (binding identity to an SST would invalidate descriptors
//! on every compaction). Per-segment stats are read cheaply from the engine's
//! per-SST metadata, never a data-block scan.
//!
//! This is the storage-side foundation that segment export (read a key range as
//! a portable key-value stream) and segment install (re-ingest that stream under
//! the target's own codec / tier) build on. The model and map are single-node
//! (CE); the swarm transfer and live migration that consume them are EE.
//!
//! See the architecture's segment design for the full model (descriptor fields,
//! auto-split/merge triggers, the optimizer); this module provides the concrete
//! descriptor + per-partition map and derives both from live engine statistics.

use lsm_tree::{AbstractTree, Guard};

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// Logical category of a segment's data, derived from its partition. Drives
/// placement heuristics (tier preference, erasure-coding eligibility, striping).
///
/// Finer classification (distinguishing vector vs full-text vs b-tree index
/// entries within the index partition, or splitting out MVCC history) is a
/// later refinement; this maps each partition to its dominant access character.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentDataType {
    /// Adjacency posting lists (commutative, merge-operator writes).
    PostingList,
    /// Keyed record stores: node / edge properties, schema, indexes, counters,
    /// registry, raft metadata.
    PropertyDoc,
    /// Dense f32 vector truth tier (random-access; never erasure-coded).
    VectorIndex,
    /// Content-addressed blob chunks.
    Blob,
}

impl SegmentDataType {
    /// Map a partition to its dominant data type.
    #[must_use]
    pub fn from_partition(partition: Partition) -> Self {
        match partition {
            Partition::Adj => Self::PostingList,
            Partition::VectorF32 => Self::VectorIndex,
            Partition::Blob => Self::Blob,
            Partition::Node
            | Partition::EdgeProp
            | Partition::BlobRef
            | Partition::Schema
            | Partition::Idx
            | Partition::Raft
            | Partition::Counter
            | Partition::Registry => Self::PropertyDoc,
        }
    }
}

/// Stable wire tag identifying a partition inside a self-describing segment
/// blob, so a single transfer handler installs a received segment into the
/// right partition without out-of-band routing.
///
/// **Stable: never renumber an existing variant** (a tag travels on the wire
/// and must decode the same on every node and across upgrades). Append new
/// partitions with new numbers.
#[must_use]
pub fn partition_wire_tag(partition: Partition) -> u8 {
    match partition {
        Partition::Node => 0,
        Partition::Adj => 1,
        Partition::EdgeProp => 2,
        Partition::Blob => 3,
        Partition::BlobRef => 4,
        Partition::Schema => 5,
        Partition::Idx => 6,
        Partition::Raft => 7,
        Partition::Counter => 8,
        Partition::VectorF32 => 9,
        Partition::Registry => 10,
    }
}

/// Inverse of [`partition_wire_tag`]; `None` for an unknown tag (a newer
/// partition from a future version).
#[must_use]
pub fn partition_from_wire_tag(tag: u8) -> Option<Partition> {
    Some(match tag {
        0 => Partition::Node,
        1 => Partition::Adj,
        2 => Partition::EdgeProp,
        3 => Partition::Blob,
        4 => Partition::BlobRef,
        5 => Partition::Schema,
        6 => Partition::Idx,
        7 => Partition::Raft,
        8 => Partition::Counter,
        9 => Partition::VectorF32,
        10 => Partition::Registry,
        _ => return None,
    })
}

/// Unique identifier of a placement segment within a node. Stable across
/// compaction (it tracks the logical region, not an SST file).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SegmentId(pub u64);

/// Half-open key range `[start, end)` a segment covers. An empty `end` means the
/// range is unbounded above (covers every key `>= start`), which is the
/// whole-partition case before any split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KeyRange {
    /// Inclusive start key.
    pub start: Vec<u8>,
    /// Exclusive end key; empty means unbounded above.
    pub end: Vec<u8>,
}

impl KeyRange {
    /// Whether `key` falls in `[start, end)` (an empty `end` is unbounded above).
    #[must_use]
    pub fn contains(&self, key: &[u8]) -> bool {
        key >= self.start.as_slice() && (self.end.is_empty() || key < self.end.as_slice())
    }
}

/// Descriptor of one placement segment: its identity, the key range it covers,
/// and size + access statistics aggregated from the engine's per-SST metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentDescriptor {
    /// Stable identifier (tracks the logical region across compaction).
    pub id: SegmentId,
    /// Partition this segment belongs to.
    pub partition: Partition,
    /// Key range covered.
    pub key_range: KeyRange,
    /// Dominant data type, for placement heuristics.
    pub data_type: SegmentDataType,
    /// On-disk bytes (summed across overlapping SSTs).
    pub size_bytes: u64,
    /// Entry versions stored (summed across overlapping SSTs).
    pub item_count: u64,
    /// Cumulative data-consulting point reads (hotness), summed across SSTs.
    /// A monotonic counter, not a rate: derive an EMA from successive polls.
    pub reads: u64,
    /// Most recent point read, in unix seconds, across the segment's SSTs
    /// (`0` if never read or on a clockless build).
    pub last_access_secs: u64,
}

/// Ordered, contiguous, non-overlapping set of segments covering one partition's
/// keyspace.
///
/// Built by reading the partition's key bounds and per-SST statistics. In the
/// single-shard (CE) case a partition is one segment spanning its whole key
/// range; finer segmentation (size / hotspot split) is layered on later. The map
/// is the lookup structure that resolves a key to the segment that holds it.
#[derive(Debug, Clone)]
pub struct SegmentMap {
    partition: Partition,
    segments: Vec<SegmentDescriptor>,
}

impl SegmentMap {
    /// Build the segment map for `partition` from live engine state.
    ///
    /// Reads the partition's min/max key (at the current snapshot) and the
    /// aggregate per-SST size + access statistics, producing one logical segment
    /// spanning the partition. An empty partition yields an empty map.
    ///
    /// `segment_id` assigns the stable identifier for the spanning segment
    /// (the caller owns id allocation; a persistent, Raft-replicated map assigns
    /// these durably).
    ///
    /// # Errors
    ///
    /// Returns an error if the partition handle is unavailable or a segment's
    /// file size cannot be stat-ed.
    pub fn build(
        engine: &StorageEngine,
        partition: Partition,
        segment_id: SegmentId,
    ) -> StorageResult<Self> {
        let tree = engine.tree(partition)?;
        let snapshot = engine.snapshot();

        // Key bounds at the current snapshot. No first key => empty partition.
        let Some(first) = tree.first_key_value(snapshot, None) else {
            return Ok(Self {
                partition,
                segments: Vec::new(),
            });
        };
        let start = first.key()?.to_vec();

        // Aggregate size + access stats across every SST of the partition. For
        // the whole-partition segment this is simply the partition total.
        let mut size_bytes = 0u64;
        let mut item_count = 0u64;
        let mut reads = 0u64;
        let mut last_access_secs = 0u64;
        for level in tree.level_segment_stats()? {
            for seg in &level.segments {
                size_bytes = size_bytes.saturating_add(seg.used_bytes);
                item_count = item_count.saturating_add(seg.item_count);
                reads = reads.saturating_add(seg.reads);
                last_access_secs = last_access_secs.max(seg.last_access_secs);
            }
        }

        let descriptor = SegmentDescriptor {
            id: segment_id,
            partition,
            // Whole-partition span: unbounded above (empty end).
            key_range: KeyRange {
                start,
                end: Vec::new(),
            },
            data_type: SegmentDataType::from_partition(partition),
            size_bytes,
            item_count,
            reads,
            last_access_secs,
        };

        Ok(Self {
            partition,
            segments: vec![descriptor],
        })
    }

    /// The partition this map covers.
    #[must_use]
    pub fn partition(&self) -> Partition {
        self.partition
    }

    /// All segments, ordered by key-range start (contiguous, non-overlapping).
    #[must_use]
    pub fn segments(&self) -> &[SegmentDescriptor] {
        &self.segments
    }

    /// The segment whose key range contains `key`, if any.
    #[must_use]
    pub fn lookup(&self, key: &[u8]) -> Option<&SegmentDescriptor> {
        self.segments.iter().find(|s| s.key_range.contains(key))
    }

    /// Total on-disk bytes across all segments.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.segments.iter().map(|s| s.size_bytes).sum()
    }

    /// Whether the map covers no data (empty partition).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
