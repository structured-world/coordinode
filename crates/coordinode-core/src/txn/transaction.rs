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
mod tests {
    use super::*;

    fn test_oracle() -> TimestampOracle {
        TimestampOracle::resume_from(Timestamp::from_raw(100))
    }

    #[test]
    fn begin_allocates_start_ts() {
        // `begin()` must call `oracle.next()` to allocate start_ts.
        // The exact value is wall-clock-based (HLC), so we verify:
        //   1. start_ts > 0 (non-zero — not ZERO sentinel)
        //   2. oracle has advanced after begin (start_ts was consumed)
        //   3. start_ts equals oracle.current() right after begin
        let oracle = test_oracle();
        let before = oracle.current();
        let txn = Transaction::begin(&oracle);
        let after = oracle.current();

        assert!(!txn.start_ts().is_zero(), "start_ts must be non-zero");
        // Oracle must have advanced by exactly 1 call to next().
        assert!(after >= before, "oracle must not go backward");
        // start_ts was the value returned by next(), so current() == start_ts.
        assert_eq!(
            txn.start_ts(),
            after,
            "start_ts must equal oracle.current() after begin"
        );
    }

    #[test]
    fn begin_at_uses_given_ts() {
        let txn = Transaction::begin_at(Timestamp::from_raw(42));
        assert_eq!(txn.start_ts().as_raw(), 42);
    }

    #[test]
    fn empty_transaction_is_read_only() {
        let oracle = test_oracle();
        let txn = Transaction::begin(&oracle);
        assert!(txn.is_read_only());
        assert_eq!(txn.write_count(), 0);
    }

    #[test]
    fn put_buffers_write() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.put(TxnPartition::Node, b"key1".to_vec(), b"val1".to_vec());
        assert!(!txn.is_read_only());
        assert_eq!(txn.write_count(), 1);
    }

    #[test]
    fn delete_buffers_write() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.delete(TxnPartition::Node, b"key1".to_vec());
        assert_eq!(txn.write_count(), 1);
    }

    #[test]
    fn multiple_partitions_tracked_separately() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.put(TxnPartition::Node, b"n1".to_vec(), b"v1".to_vec());
        txn.put(TxnPartition::Adj, b"a1".to_vec(), b"v2".to_vec());
        txn.delete(TxnPartition::Idx, b"i1".to_vec());

        assert_eq!(txn.write_count(), 3);
    }

    #[test]
    fn commit_assigns_commit_ts() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);
        let start = txn.start_ts();

        txn.put(TxnPartition::Node, b"k".to_vec(), b"v".to_vec());
        let (commit_ts, buffer, _stats) = txn.commit(&oracle);

        assert!(commit_ts > start, "commit_ts must be after start_ts");
        assert!(!buffer.is_empty());
    }

    #[test]
    fn read_only_commit_reuses_start_ts() {
        let oracle = test_oracle();
        let txn = Transaction::begin(&oracle);
        let start = txn.start_ts();

        let (commit_ts, buffer, _stats) = txn.commit(&oracle);
        assert_eq!(commit_ts, start);
        assert!(buffer.is_empty());
    }

    #[test]
    fn stats_tracking() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.stats_mut().nodes_created = 3;
        txn.stats_mut().edges_created = 5;
        txn.stats_mut().properties_set = 10;

        assert_eq!(txn.stats().nodes_created, 3);
        assert_eq!(txn.stats().edges_created, 5);
        assert_eq!(txn.stats().properties_set, 10);
    }

    #[test]
    fn get_pending_write_finds_latest() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.put(TxnPartition::Node, b"k".to_vec(), b"v1".to_vec());
        txn.put(TxnPartition::Node, b"k".to_vec(), b"v2".to_vec());

        let pending = txn.get_pending_write(TxnPartition::Node, b"k");
        assert!(
            matches!(pending, Some(WriteOp::Put { value, .. }) if value == b"v2"),
            "expected Put with v2, got {pending:?}"
        );
    }

    #[test]
    fn get_pending_write_returns_none_for_missing() {
        let oracle = test_oracle();
        let txn = Transaction::begin(&oracle);

        assert!(txn
            .get_pending_write(TxnPartition::Node, b"missing")
            .is_none());
    }

    #[test]
    fn cache_read_and_retrieve() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.cache_read(TxnPartition::Node, b"k".to_vec(), Some(b"cached".to_vec()));

        let cached = txn.get_cached(TxnPartition::Node, b"k");
        assert_eq!(cached, Some(&Some(b"cached".to_vec())));
    }

    #[test]
    fn cache_read_none_for_missing_key() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.cache_read(TxnPartition::Node, b"k".to_vec(), None);

        let cached = txn.get_cached(TxnPartition::Node, b"k");
        assert_eq!(cached, Some(&None));
    }

    #[test]
    fn abort_discards_writes() {
        let oracle = test_oracle();
        let mut txn = Transaction::begin(&oracle);

        txn.put(TxnPartition::Node, b"k".to_vec(), b"v".to_vec());
        txn.stats_mut().nodes_created = 1;

        let stats = txn.abort();
        assert_eq!(stats.nodes_created, 1);
        // write_buffer is dropped — nothing to commit
    }
}
