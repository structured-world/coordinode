//! Statement-level transaction: atomic write batching with MVCC timestamps.
//!
//! Each OpenCypher statement executes within a `Transaction` that:
//! - Allocates `start_ts` at begin (for snapshot reads)
//! - Accumulates writes in an in-memory buffer
//! - Assigns `commit_ts` and flushes atomically on commit
//!
//! Multi-document transaction support will extend this to multi-statement
//! transactions with OCC conflict detection and Raft proposal integration.

use std::collections::HashMap;

use crate::txn::timestamp::{Timestamp, TimestampOracle};

/// Write operation buffered in a transaction.
#[derive(Debug, Clone)]
pub enum WriteOp {
    /// Insert or update a key-value pair.
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key (tombstone).
    Delete { key: Vec<u8> },
}

/// Partition identifier for write operations.
///
/// Mirrors `coordinode_storage::engine::partition::Partition` but lives in
/// core to avoid a dependency on the storage crate. The executor maps
/// this to the storage partition on commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TxnPartition {
    Node,
    Adj,
    EdgeProp,
    Blob,
    BlobRef,
    Schema,
    Idx,
}

/// Write statistics tracked during transaction execution.
#[derive(Debug, Clone, Default)]
pub struct WriteStats {
    pub nodes_created: u64,
    pub nodes_deleted: u64,
    pub edges_created: u64,
    pub edges_deleted: u64,
    pub properties_set: u64,
    pub properties_removed: u64,
    pub labels_added: u64,
    pub labels_removed: u64,
}

/// A statement-level transaction that buffers writes for atomic commit.
///
/// All mutations within a single OpenCypher statement are accumulated
/// in the write buffer. On `commit()`, they are flushed as a single
/// atomic batch via the storage engine's `WriteBatch`.
///
/// ## Cluster-readiness
///
/// The write buffer design is compatible with Raft proposal integration
/// the buffered mutations can be serialized into a Raft proposal
/// instead of a local WriteBatch.
///
/// ## Usage
///
/// ```ignore
/// let mut txn = Transaction::begin(&oracle);
/// // ... executor accumulates writes via txn.put() / txn.delete() ...
/// let (ops, stats) = txn.commit(&oracle);
/// // Caller flushes ops via WriteBatch with MVCC-versioned keys
/// ```
pub struct Transaction {
    /// Timestamp at which this transaction reads (snapshot isolation).
    start_ts: Timestamp,

    /// Buffered write operations, grouped by partition.
    write_buffer: HashMap<TxnPartition, Vec<WriteOp>>,

    /// Local read cache: keys read during this transaction.
    /// Used by the write path to read-then-modify without hitting storage twice.
    /// Also serves as the foundation for OCC read-set tracking.
    read_cache: HashMap<(TxnPartition, Vec<u8>), Option<Vec<u8>>>,

    /// Write statistics accumulated during execution.
    stats: WriteStats,
}

impl Transaction {
    /// Begin a new transaction, allocating a `start_ts` from the oracle.
    pub fn begin(oracle: &TimestampOracle) -> Self {
        let start_ts = oracle.next();
        Self {
            start_ts,
            write_buffer: HashMap::new(),
            read_cache: HashMap::new(),
            stats: WriteStats::default(),
        }
    }

    /// Begin a transaction at a specific timestamp (for testing / AS OF TIMESTAMP).
    pub fn begin_at(start_ts: Timestamp) -> Self {
        Self {
            start_ts,
            write_buffer: HashMap::new(),
            read_cache: HashMap::new(),
            stats: WriteStats::default(),
        }
    }

    /// The snapshot timestamp for reads.
    pub fn start_ts(&self) -> Timestamp {
        self.start_ts
    }

    /// Mutable access to write statistics.
    pub fn stats_mut(&mut self) -> &mut WriteStats {
        &mut self.stats
    }

    /// Read-only access to write statistics.
    pub fn stats(&self) -> &WriteStats {
        &self.stats
    }

    /// Buffer a put operation.
    pub fn put(&mut self, partition: TxnPartition, key: Vec<u8>, value: Vec<u8>) {
        self.write_buffer
            .entry(partition)
            .or_default()
            .push(WriteOp::Put { key, value });
    }

    /// Buffer a delete operation.
    pub fn delete(&mut self, partition: TxnPartition, key: Vec<u8>) {
        self.write_buffer
            .entry(partition)
            .or_default()
            .push(WriteOp::Delete { key });
    }

    /// Cache a value read from storage during this transaction.
    ///
    /// Subsequent reads for the same key will return the cached value,
    /// ensuring read-your-own-writes consistency within the transaction.
    pub fn cache_read(&mut self, partition: TxnPartition, key: Vec<u8>, value: Option<Vec<u8>>) {
        self.read_cache.insert((partition, key), value);
    }

    /// Check if a key has a cached read value.
    pub fn get_cached(&self, partition: TxnPartition, key: &[u8]) -> Option<&Option<Vec<u8>>> {
        self.read_cache.get(&(partition, key.to_vec()))
    }

    /// Check if there are pending writes for a specific key.
    ///
    /// Returns the latest write op for the key if any, enabling
    /// read-your-own-writes within the write buffer.
    pub fn get_pending_write(&self, partition: TxnPartition, key: &[u8]) -> Option<&WriteOp> {
        self.write_buffer.get(&partition).and_then(|ops| {
            ops.iter().rev().find(|op| match op {
                WriteOp::Put { key: k, .. } | WriteOp::Delete { key: k } => k == key,
            })
        })
    }

    /// Total number of buffered write operations.
    pub fn write_count(&self) -> usize {
        self.write_buffer.values().map(|ops| ops.len()).sum()
    }

    /// Whether the transaction has any buffered writes.
    pub fn is_read_only(&self) -> bool {
        self.write_buffer.is_empty()
    }

    /// Commit the transaction: assign `commit_ts` and return all buffered writes.
    ///
    /// The caller is responsible for flushing the writes to storage
    /// (via `WriteBatch`) with MVCC-versioned keys using the returned `commit_ts`.
    ///
    /// Returns `(commit_ts, write_buffer, stats)`.
    pub fn commit(
        self,
        oracle: &TimestampOracle,
    ) -> (Timestamp, HashMap<TxnPartition, Vec<WriteOp>>, WriteStats) {
        let commit_ts = if self.is_read_only() {
            // Read-only transactions don't need a commit timestamp.
            self.start_ts
        } else {
            oracle.next()
        };

        (commit_ts, self.write_buffer, self.stats)
    }

    /// Abort the transaction, discarding all buffered writes.
    pub fn abort(self) -> WriteStats {
        // Write buffer is dropped, nothing committed.
        self.stats
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
