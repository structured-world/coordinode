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

    /// Endpoint capacity exhausted (INV-D3 hard-limit gate). The named
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
            } => Self::SnapshotOutsideRetention {
                snapshot: requested,
                // Saturating deliberately, and in the safe direction: the
                // clamp is the business rule, not an overflow shrug. A
                // version installed at `SeqNo::MAX` is not a real state (that
                // value is the read-latest sentinel), and if one ever appeared
                // the honest report is "nothing is readable" — which is what
                // saturating to MAX says, where a wrap to 0 would claim the
                // opposite and admit exactly the reads this error exists to
                // refuse.
                watermark: oldest_retained.saturating_add(1),
            },
            other => Self::Engine(other),
        }
    }
}

/// Result type alias for storage operations.
pub type StorageResult<T> = Result<T, StorageError>;
