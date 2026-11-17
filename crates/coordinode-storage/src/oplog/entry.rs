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
    /// Delete every key in the half-open range `[start, end)` of a partition
    /// with a single range tombstone. Emitted only for a **dense
    /// contiguous run** of deleted keys (run-length coalescing of the sorted
    /// delete set, threshold-gated — see `coordinode_core::txn::coalesce`) or a whole-prefix
    /// DROP (shard / edge-type / index). The range covers only keys that are all
    /// being deleted — never across a gap holding a surviving key — so it is an
    /// exact compact encoding of "delete these keys", applied symmetrically on
    /// leader and follower as one lsm range tombstone.
    RemoveRange {
        partition: u8,
        #[serde(with = "serde_bytes")]
        start: Vec<u8>,
        #[serde(with = "serde_bytes")]
        end: Vec<u8>,
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
    /// Insert a row into a `STORAGE COLUMNAR` table's own tree.
    ///
    /// Columnar tables live outside the `Partition` keyspace (each owns a
    /// separate columnar-mode tree, keyed by `table_id`), so their writes
    /// cannot be tagged with a partition discriminant like the ops above.
    /// Recovery routes this op to the columnar table registry by `table_id`
    /// rather than to a partition tree. Kept before [`Self::Derive`] so the
    /// serde variant indices of the older ops stay stable for on-disk
    /// entries.
    ColumnarInsert {
        /// The columnar table this row belongs to.
        table_id: String,
        #[serde(with = "serde_bytes")]
        key: Vec<u8>,
        #[serde(with = "serde_bytes")]
        value: Vec<u8>,
    },
    /// Sealed entry maintenance of a DERIVED index: the MessagePack encoding
    /// of a [`DerivedIndexWork`](coordinode_core::txn::proposal::DerivedIndexWork).
    /// The journal keeps the work and its inputs, not the entries, so replay
    /// derives them again from the entry itself; a record source names the
    /// position of a node record insert among the entry's ops. Last in the
    /// enum, for the same reason as above.
    Derive {
        #[serde(with = "serde_bytes")]
        work: Vec<u8>,
    },
    /// One application unit in its compact frame
    /// ([`coordinode_core::txn::frame`]): the form a journal records and a
    /// Raft log entry carries each proposal in. Readers expand it into the
    /// operations it encodes. Last in the enum, for the same reason as above.
    Unit {
        #[serde(with = "serde_bytes")]
        frame: Vec<u8>,
    },
    /// The namespace the ops after it, up to the next one, belong to: an op
    /// on a namespace's data follows one, the deployment's own need none.
    /// Last in the enum, for the same reason as above.
    Namespace {
        /// The namespace's id.
        id: u64,
    },
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

/// An [`OplogEntry`] read in place from its encoded bytes: its operations'
/// keys, values and frames borrow the bytes rather than copy them, and its
/// pre-images are passed over. What it holds besides the bytes is its list
/// of operations, one [`OplogOpRef`] each.
#[derive(Debug, Deserialize)]
pub struct OplogEntryRef<'a> {
    /// See [`OplogEntry::ts`].
    pub ts: u64,
    /// See [`OplogEntry::term`].
    pub term: u64,
    /// See [`OplogEntry::index`].
    pub index: u64,
    /// See [`OplogEntry::shard`].
    pub shard: ShardId,
    /// See [`OplogEntry::ops`].
    #[serde(borrow)]
    pub ops: Vec<OplogOpRef<'a>>,
    /// See [`OplogEntry::is_migration`].
    pub is_migration: bool,
    /// Pre-images, passed over.
    #[serde(rename = "pre_images")]
    _pre_images: serde::de::IgnoredAny,
}

/// An [`OplogOp`] read in place: the same variants, in the same order, over
/// borrowed bytes.
#[derive(Debug, Deserialize)]
#[allow(missing_docs)]
pub enum OplogOpRef<'a> {
    Insert {
        partition: u8,
        #[serde(borrow, with = "serde_bytes")]
        key: &'a [u8],
        #[serde(borrow, with = "serde_bytes")]
        value: &'a [u8],
    },
    Delete {
        partition: u8,
        #[serde(borrow, with = "serde_bytes")]
        key: &'a [u8],
    },
    Merge {
        partition: u8,
        #[serde(borrow, with = "serde_bytes")]
        key: &'a [u8],
        #[serde(borrow, with = "serde_bytes")]
        operand: &'a [u8],
    },
    RemoveRange {
        partition: u8,
        #[serde(borrow, with = "serde_bytes")]
        start: &'a [u8],
        #[serde(borrow, with = "serde_bytes")]
        end: &'a [u8],
    },
    Noop,
    RaftEntry {
        #[serde(borrow, with = "serde_bytes")]
        data: &'a [u8],
    },
    RaftTruncation {
        after_index: u64,
    },
    ColumnarInsert {
        #[serde(borrow)]
        table_id: &'a str,
        #[serde(borrow, with = "serde_bytes")]
        key: &'a [u8],
        #[serde(borrow, with = "serde_bytes")]
        value: &'a [u8],
    },
    Derive {
        #[serde(borrow, with = "serde_bytes")]
        work: &'a [u8],
    },
    Unit {
        #[serde(borrow, with = "serde_bytes")]
        frame: &'a [u8],
    },
    Namespace {
        id: u64,
    },
}

impl<'a> OplogEntryRef<'a> {
    /// Read the entry `bytes` encode, in place.
    ///
    /// # Errors
    ///
    /// The bytes are not an encoded entry.
    pub fn decode(bytes: &'a [u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
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

    /// The entry as [`OplogEntryRef`], borrowing its operations' bytes: what
    /// a reader of entries read in place takes, for one already decoded.
    pub fn borrowed(&self) -> OplogEntryRef<'_> {
        OplogEntryRef {
            ts: self.ts,
            term: self.term,
            index: self.index,
            shard: self.shard,
            ops: self.ops.iter().map(OplogOp::borrowed).collect(),
            is_migration: self.is_migration,
            _pre_images: serde::de::IgnoredAny,
        }
    }
}

impl OplogOp {
    /// The operation as [`OplogOpRef`], borrowing its bytes.
    pub fn borrowed(&self) -> OplogOpRef<'_> {
        match self {
            Self::Insert {
                partition,
                key,
                value,
            } => OplogOpRef::Insert {
                partition: *partition,
                key,
                value,
            },
            Self::Delete { partition, key } => OplogOpRef::Delete {
                partition: *partition,
                key,
            },
            Self::Merge {
                partition,
                key,
                operand,
            } => OplogOpRef::Merge {
                partition: *partition,
                key,
                operand,
            },
            Self::RemoveRange {
                partition,
                start,
                end,
            } => OplogOpRef::RemoveRange {
                partition: *partition,
                start,
                end,
            },
            Self::Noop => OplogOpRef::Noop,
            Self::RaftEntry { data } => OplogOpRef::RaftEntry { data },
            Self::RaftTruncation { after_index } => OplogOpRef::RaftTruncation {
                after_index: *after_index,
            },
            Self::ColumnarInsert {
                table_id,
                key,
                value,
            } => OplogOpRef::ColumnarInsert {
                table_id,
                key,
                value,
            },
            Self::Derive { work } => OplogOpRef::Derive { work },
            Self::Unit { frame } => OplogOpRef::Unit { frame },
            Self::Namespace { id } => OplogOpRef::Namespace { id: *id },
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
