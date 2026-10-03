//! Backfill: fill a B-tree index with the entries of the nodes already
//! stored.
//!
//! Writers maintain the index from the moment it is registered with them. A
//! transaction that opened before then decided what to maintain without it,
//! so the backfill first waits for every such transaction to end (the wait
//! PostgreSQL's concurrent index build makes for older snapshots): after it,
//! every node write is either visible to the scan or maintains the index
//! itself.
//!
//! The shard's nodes are then read a page at a time, each page in a
//! transaction of its own whose entries commit through the caller's commit
//! path, so they replicate like any other write and no single transaction
//! grows with the data. A page that read a node a writer changed before the
//! page committed conflicts at commit (the page's node keys are in its read
//! set) and is read again: an entry for a value the node no longer holds
//! cannot land.

use coordinode_core::graph::intern::FieldInterner;
use coordinode_core::graph::node::{NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{LocalNodeStore, NodeStore, StoreError};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::{CommitError, Transaction};

use super::definition::IndexDefinition;
use super::registry::{IndexWriteError, UniqueViolation, record_lookup, stage_node_entry};

/// Nodes read per page, and so the most entries one page's transaction
/// commits.
const PAGE: usize = 512;

/// Conflicts one page may meet in a row before the backfill gives up. Each
/// one means a writer changed a node of the page while it was being read.
const MAX_PAGE_CONFLICTS: u32 = 64;

/// How long the backfill waits for the transactions opened before the index
/// was registered, and how often it looks.
const OLDER_TRANSACTIONS_WAIT: std::time::Duration = std::time::Duration::from_secs(60);
const OLDER_TRANSACTIONS_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// Why a backfill stopped.
#[derive(Debug, thiserror::Error)]
pub enum BackfillError {
    /// Two stored nodes hold one value of a unique index: the data breaks
    /// the constraint the index would enforce.
    #[error(transparent)]
    Duplicate(#[from] UniqueViolation),
    /// Reading nodes or staging entries failed.
    #[error(transparent)]
    Store(#[from] StoreError),
    /// A page's commit failed for a reason retrying does not cure.
    #[error("index backfill commit: {0}")]
    Commit(CommitError),
    /// Writers kept changing the nodes of one page.
    #[error("index backfill kept conflicting with concurrent writes")]
    Contended,
    /// Transactions that opened before the index existed did not end in
    /// time; they may still write nodes without its entries.
    #[error(
        "{0} transactions opened before the index was created are still open; \
         retry once they have ended"
    )]
    OlderTransactions(usize),
}

impl From<IndexWriteError> for BackfillError {
    fn from(e: IndexWriteError) -> Self {
        match e {
            IndexWriteError::Unique(v) => Self::Duplicate(v),
            IndexWriteError::Store(s) => Self::Store(s),
        }
    }
}

/// Where a backfill reads and how it opens its transactions.
pub struct Backfill<'a> {
    /// The storage engine.
    pub engine: &'a StorageEngine,
    /// The timestamp oracle; `None` for a direct-mode engine.
    pub oracle: Option<&'a TimestampOracle>,
    /// Resolves property names to the field ids records are keyed by.
    pub interner: &'a FieldInterner,
    /// The shard whose nodes are indexed.
    pub shard_id: u16,
    /// Transactions of the caller itself that are open while it runs the
    /// backfill (the statement creating the index), not waited for.
    pub own_open: usize,
}

impl<'a> Backfill<'a> {
    /// Stage and commit the entries of every stored node of `index`'s label,
    /// committing each page through `commit`. Returns the number of entries
    /// staged: one per node, and one per version of a temporal node.
    /// Call once `index` is registered with the writers.
    ///
    /// # Errors
    ///
    /// See [`BackfillError`]. Pages committed before the error stay; the
    /// caller that abandons the index clears them.
    pub fn run(
        &self,
        index: &IndexDefinition,
        commit: &mut dyn FnMut(&mut Transaction<'a>) -> Result<(), CommitError>,
    ) -> Result<u64, BackfillError> {
        let boundary = self.engine.snapshot_boundary();
        self.engine
            .await_transactions_through(
                boundary,
                self.own_open,
                OLDER_TRANSACTIONS_POLL,
                OLDER_TRANSACTIONS_WAIT,
            )
            .map_err(BackfillError::OlderTransactions)?;
        let nodes = LocalNodeStore;
        let prefix = nodes.shard_scan_prefix(self.shard_id);
        let mut start_after: Option<Vec<u8>> = None;
        let mut indexed = 0u64;
        let mut conflicts = 0u32;
        loop {
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
            };
            let page =
                nodes.prefix_scan_paged_tracked(&mut txn, &prefix, start_after.as_deref(), PAGE)?;
            let mut claims = Vec::new();
            let mut read = Vec::with_capacity(page.rows.len());
            let mut staged = 0u64;
            for (key, bytes) in &page.rows {
                // A temporal node has an entry per version, as its writers
                // stage one for each version they write.
                let owner = match decode_node_key(key) {
                    Some((_, node_id)) => EntryOwner::node(node_id.as_raw()),
                    None => match decode_temporal_node_key(key) {
                        Some((_, node_id, valid_from)) => {
                            EntryOwner::version(node_id.as_raw(), valid_from)
                        }
                        None => continue,
                    },
                };
                let record = NodeRecord::from_msgpack(bytes).map_err(|e| StoreError::Decode {
                    kind: "node record",
                    message: e.to_string(),
                })?;
                if record.primary_label() != index.label {
                    continue;
                }
                let lookup = record_lookup(&record, self.interner);
                let field_of = |name: &str| self.interner.lookup(name);
                if stage_node_entry(
                    self.engine,
                    &mut txn,
                    index,
                    owner,
                    &lookup,
                    &field_of,
                    &mut claims,
                )? {
                    staged += 1;
                }
                read.push(key.clone());
            }
            // The page's entries hold only if the rows they came from are
            // still the rows it read when the page commits.
            let unchanged = nodes.condition_unchanged(&mut txn, &read)?;
            let committed = if unchanged {
                match commit(&mut txn) {
                    Ok(()) => true,
                    Err(CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }) => false,
                    Err(e) => return Err(BackfillError::Commit(e)),
                }
            } else {
                false
            };
            if !committed {
                conflicts += 1;
                if conflicts > MAX_PAGE_CONFLICTS {
                    return Err(BackfillError::Contended);
                }
                continue;
            }
            indexed += staged;
            conflicts = 0;
            if page.exhausted {
                return Ok(indexed);
            }
            start_after = page.last_key;
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
