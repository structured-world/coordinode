//! Oplog entry types: `OplogEntry`, `OplogOp`, `PreImage`, `ShardId`.
//!
//! On-disk encoding: MessagePack via `rmp-serde`. Binary fields use
//! `serde_bytes` so they serialize as msgpack `Bin` (compact byte array)
//! rather than an array of individual u8 values.

use serde::{Deserialize, Serialize};

/// Shard identifier — u32 unique per shard within a cluster.
pub type ShardId = u32;

/// Snapshot of a key's value before a mutation.
///
/// Captured for CDC consumers and PITR rewind. Absent when the `pre_images`
/// feature is disabled in the oplog config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PreImage {
    /// Partition discriminant (matches `Partition::discriminant()`).
    pub partition: u8,
    /// Key bytes.
    #[serde(with = "serde_bytes")]
    pub key: Vec<u8>,
    /// Value bytes as they existed before this mutation.
    #[serde(with = "serde_bytes")]
    pub value: Vec<u8>,
}

/// A single storage mutation captured inside an [`OplogEntry`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OplogOp {
    /// Insert or overwrite a key-value pair.
    Insert {
        /// Partition discriminant.
        partition: u8,
        #[serde(with = "serde_bytes")]
        key: Vec<u8>,
        #[serde(with = "serde_bytes")]
        value: Vec<u8>,
    },
    /// Tombstone a key.
    Delete {
        partition: u8,
        #[serde(with = "serde_bytes")]
        key: Vec<u8>,
    },
    /// Apply a merge operand (posting-list patch, counter delta, etc.).
    Merge {
        partition: u8,
        #[serde(with = "serde_bytes")]
        key: Vec<u8>,
        #[serde(with = "serde_bytes")]
        operand: Vec<u8>,
    },
    /// No-op heartbeat / gap-fill entry.
    Noop,
    /// Serialized openraft log entry (Normal or Membership payload).
    ///
    /// Used by the Raft log storage to persist complete openraft
    /// `Entry<TypeConfig>` objects — including both application data and
    /// membership change payloads — as oplog entries. The `data` field
    /// contains the full msgpack-serialized entry.
    RaftEntry {
        #[serde(with = "serde_bytes")]
        data: Vec<u8>,
    },
    /// Truncation sentinel for the Raft log.
    ///
    /// All oplog entries written before this sentinel with
    /// `index > after_index` are considered invalid — they were written by
    /// a previous leader term and have been superseded by new entries from
    /// the current leader.
    RaftTruncation { after_index: u64 },
}

/// One logical unit written to the oplog: a batch of ops at a Raft index/term.
///
/// Serialized with MessagePack and stored in a segment file wrapped in a
/// varint length prefix and a crc32 suffix per the segment on-disk format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OplogEntry {
    /// Hybrid Logical Clock timestamp (48-bit wall ms | 16-bit logical counter).
    pub ts: u64,
    /// Raft term when this entry was proposed.
    pub term: u64,
    /// Raft log index — monotonically increasing per shard.
    pub index: u64,
    /// Shard that owns this entry.
    pub shard: ShardId,
    /// Storage operations carried by this entry.
    pub ops: Vec<OplogOp>,
    /// True for entries carrying cross-shard migration ops.
    pub is_migration: bool,
    /// Pre-images for CDC/PITR — `None` when the feature is disabled.
    pub pre_images: Option<Vec<PreImage>>,
}

impl OplogEntry {
    /// Encode to MessagePack bytes.
    pub fn encode(&self) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(self)
    }

    /// Decode from MessagePack bytes.
    pub fn decode(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
