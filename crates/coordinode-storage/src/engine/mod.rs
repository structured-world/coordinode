//! CoordiNode LSM storage engine — primary KV layer (coordinode-storage).

pub mod batch;
pub mod capacity;
pub(crate) mod compaction;
pub mod config;
pub mod coordinator;
pub mod core;
pub(crate) mod flush;
pub mod merge;
pub(crate) mod mvcc_gc;
pub mod partition;
pub mod routing;
pub mod stats;
pub mod transaction;
pub mod vector_keys;

/// Snapshot type: a sequence number used as a read visibility bound.
///
/// Reads with this seqno see all writes with seqno ≤ this value.
/// Obtain via `StorageEngine::snapshot()` for the current latest,
/// or `StorageEngine::snapshot_at(ts)` for a specific point-in-time.
pub type StorageSnapshot = lsm_tree::SeqNo;

/// Iterator over key-value guards from an lsm-tree prefix or range scan.
pub type StorageIter =
    Box<dyn DoubleEndedIterator<Item = lsm_tree::IterGuardImpl> + Send + 'static>;
