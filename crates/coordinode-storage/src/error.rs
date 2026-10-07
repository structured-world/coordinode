//! Storage engine error types.

/// Errors from the storage engine layer.
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// Underlying storage error.
    ///
    /// Converted from [`lsm_tree::Error`] by the hand-written `From` below
    /// rather than a derive, so the engine's own retention refusal is lifted
    /// into [`StorageError::SnapshotOutsideRetention`] instead of arriving as
    /// an opaque engine error.
    #[error("storage engine error: {0}")]
    Engine(lsm_tree::Error),

    /// Partition not found.
    #[error("partition not found: {name}")]
    PartitionNotFound { name: String },

    /// Invalid configuration.
    #[error("invalid storage config: {0}")]
    InvalidConfig(String),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Transaction conflict (OCC retry needed).
    #[error("transaction conflict, retry")]
    Conflict,

    /// I/O error (file read/write, directory operations).
    #[error("I/O error: {0}")]
    Io(String),

    /// CRC32 checksum mismatch — data corruption detected.
    #[error("checksum mismatch in {context}: expected {expected:#010x}, got {actual:#010x}")]
    ChecksumMismatch {
        expected: u32,
        actual: u32,
        context: String,
    },

    /// Endpoint capacity exhausted (hard-limit gate). The named
    /// endpoint's `used_bytes` is at or above its `hard_limit_bytes`
    /// and its `is_writable` flag is currently `false`. Coordinator
    /// may retry on a different endpoint or surface the error to the
    /// client.
    #[error(
        "endpoint {endpoint_id:?} capacity exhausted (used={used_bytes}, \
         hard_limit={hard_limit_bytes}) — writes rejected until cascade \
         eviction or operator cleanup brings usage below the limit"
    )]
    CapacityExhausted {
        endpoint_id: String,
        used_bytes: u64,
        hard_limit_bytes: u64,
    },

    /// A snapshot read below the MVCC retention horizon. History at that seqno
    /// may already be collected, so the read is refused instead of answering
    /// from whatever survived. Raise `retention_window_secs` or read at or
    /// above the horizon.
    ///
    /// Raised by this engine's own policy guard (the configured time-travel
    /// window, see `StorageEngine::set_retention_window`) and, as a backstop,
    /// lifted from the tree's `SnapshotBelowRetention` when the physical
    /// history was pruned past the requested seqno — which a `drop_range`, a
    /// `clear` or a filtering compaction can do independently of the window.
    #[error(
        "snapshot {snapshot} is below the MVCC retention horizon {watermark}: \
         history that old may already be collected"
    )]
    SnapshotOutsideRetention { snapshot: u64, watermark: u64 },

    /// The store has journal entries but no record of which of them its
    /// partition trees physically hold, so replaying them could lose an
    /// acknowledged write or apply a merge twice. Written by a release that
    /// predates apply coverage; the store is left untouched.
    #[error(
        "store at {path} holds {entries} journal entries with no apply-coverage \
         record; refusing to guess which of them are on disk. Dump it with the \
         release that wrote it (`coordinode backup --format raft-snapshot`) and \
         restore the dump with this one (`coordinode restore --format raft-snapshot`)"
    )]
    CoverageUnprovable { path: String, entries: usize },

    /// A copy of a partition stands behind this node's Raft applies:
    /// installing it would lose the entries in between, which this node
    /// has applied already and will not apply again. A peer further along,
    /// or the same one a moment later, serves a usable copy.
    #[error(
        "the copy of {partition} holds the Raft log up to {source_next}, behind this \
         node's {local_next}"
    )]
    PositionBehind {
        partition: String,
        source_next: u64,
        local_next: u64,
    },

    /// The stored field dictionary is not one consistent set of bindings, or
    /// a registration could not be applied: stored data cannot be interpreted
    /// safely, so nothing is served from it.
    #[error("field dictionary: {0}")]
    FieldDictionary(#[from] coordinode_core::graph::intern::DictionaryError),

    /// A unit's DERIVED index work cannot be derived as it was sealed: its
    /// interpretation is unsupported or its inputs are missing or corrupt.
    /// Nothing of the unit is applied; the index is not guessed at.
    #[error("index derivation: {0}")]
    IndexDerivation(#[from] coordinode_core::index::derive::DeriveError),

    /// A unit frame that cannot be written or read: it exceeds a frame
    /// bound, or its bytes are not a frame this build accepts.
    #[error("unit frame: {0}")]
    Frame(#[from] coordinode_core::txn::frame::FrameError),

    /// A log reader's position is below what the oplog still holds: the
    /// entries from `requested` up to `first_retained` were purged, and a
    /// reader resuming there cannot be given them. Never answered by reading
    /// on from `first_retained`, which would hide the gap.
    #[error(
        "retention lost: oplog position {requested} was purged; the log now starts at {first_retained}"
    )]
    RetentionLost {
        /// The next index the reader asked for.
        requested: u64,
        /// The first index the oplog still holds.
        first_retained: u64,
    },

    /// The filesystem under a durable endpoint has less free space than the
    /// reserve, so writes are refused before any reaches the disk; reads go
    /// on. Writes are admitted again once space is freed.
    #[error(
        "no space: {available_bytes} bytes free under {path}, below the {min_free_bytes}-byte \
         reserve; writes are refused until space is freed, reads continue"
    )]
    OutOfSpace {
        /// The endpoint path with the least free space.
        path: String,
        /// Free bytes there at the last reading.
        available_bytes: u64,
        /// The reserve writes need.
        min_free_bytes: u64,
    },

    /// The data directory was written in an engine format this release
    /// cannot open: newer than its own, or older than the one format before
    /// it, which it migrates. Opening it changes nothing.
    #[error(
        "data directory {path} is in engine format {found}; this release runs format {runs} \
         and opens only format {runs} or {previous}{hint}",
        // Format numbering starts at 1, so the one before always exists.
        previous = runs - 1,
        hint = if found < runs { "; open it with the release that runs the next format first" } else { "" }
    )]
    UnsupportedFormat {
        /// The directory refused.
        path: String,
        /// The engine format its marker names.
        found: u32,
        /// The engine format this release runs.
        runs: u32,
    },

    /// A catalog record (an index definition, an index build) does not
    /// decode in this build. Nothing is served without it: skipping it would
    /// serve the database without that index and the uniqueness it enforces.
    /// A directory written by a development build between releases carries
    /// catalog records of an earlier layout and moves through a logical dump.
    #[error(
        "the {kind} record {key} does not decode in this build ({detail}); the data \
         directory was written by another build or is damaged, and is not opened. A \
         directory written by a development build is moved with a logical dump: \
         `coordinode backup --format binary` with the build that wrote it, then \
         `coordinode restore --format binary` with this one"
    )]
    UnreadableCatalog {
        /// What the record is.
        kind: &'static str,
        /// The record's key, printable ([`printable_key`]).
        key: String,
        /// Why it does not decode.
        detail: String,
    },
}

/// `key` as text for a message: printable ASCII as it is, every other byte
/// escaped.
pub fn printable_key(key: &[u8]) -> String {
    key.escape_ascii().to_string()
}

impl From<lsm_tree::Error> for StorageError {
    fn from(err: lsm_tree::Error) -> Self {
        match err {
            // The tree refuses a snapshot whose version history it no longer
            // holds. That is the same condition our policy guard raises, so it
            // carries the same typed error the whole stack already maps to
            // OUT_OF_RANGE / OUTSIDE_RETENTION, not an opaque engine failure.
            // A snapshot is servable iff it is strictly above `oldest_retained`,
            // so the first readable seqno — our `watermark` — is one past it.
            lsm_tree::Error::SnapshotBelowRetention {
                requested,
                oldest_retained,
            } => {
                // `oldest_retained` is a retained version's install seqno, and
                // an install takes its seqno from the generator: HLC
                // microseconds or a counter, both far below the top of the
                // range. `SeqNo::MAX` is the read-latest sentinel a caller
                // passes to a read, never a value a version is installed at,
                // so the increment cannot overflow.
                debug_assert!(
                    oldest_retained < u64::MAX,
                    "retained version installed at the read-latest sentinel"
                );
                Self::SnapshotOutsideRetention {
                    snapshot: requested,
                    // A snapshot is servable strictly above the oldest retained
                    // version, so the first readable seqno is one past it.
                    watermark: oldest_retained + 1,
                }
            }
            other => Self::Engine(other),
        }
    }
}

/// Result type alias for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;
