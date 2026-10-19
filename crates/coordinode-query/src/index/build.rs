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
use coordinode_core::graph::node::{NodeId, NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
use coordinode_modality::{IndexStore as _, LocalNodeStore, NodeStore, StoreError};
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

/// How often the backfill looks whether the transactions opened before the
/// index have ended.
const OLDER_TRANSACTIONS_POLL: std::time::Duration = std::time::Duration::from_millis(5);

/// The longest a backfill waits for older transactions before it looks
/// whether its process is closing.
const STOP_SLICE: std::time::Duration = std::time::Duration::from_millis(50);

/// Wait up to `wait` for every transaction open now to end: after it, a row
/// written by one opened earlier is in the store, and a later one saw what
/// was published before the call. Checked in slices against `stop`, so a
/// closing process is not held for the whole wait.
///
/// # Errors
///
/// [`BackfillError::Stopped`] once `stop` is set;
/// [`BackfillError::OlderTransactions`] when some are still open at the end.
pub(crate) fn await_older_transactions(
    engine: &StorageEngine,
    wait: std::time::Duration,
    stop: Option<&core::sync::atomic::AtomicBool>,
) -> Result<(), BackfillError> {
    let boundary = engine.snapshot_boundary();
    let deadline = std::time::Instant::now() + wait;
    loop {
        if stop.is_some_and(|s| s.load(core::sync::atomic::Ordering::Acquire)) {
            return Err(BackfillError::Stopped);
        }
        // Past the deadline the time left is none, not negative.
        let slice = deadline
            .saturating_duration_since(std::time::Instant::now())
            .min(STOP_SLICE);
        match engine.await_transactions_through(boundary, OLDER_TRANSACTIONS_POLL, slice) {
            Ok(()) => return Ok(()),
            Err(open) if std::time::Instant::now() >= deadline => {
                return Err(BackfillError::OlderTransactions(open));
            }
            Err(_) => {}
        }
    }
}

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
    /// The definition being built was dropped or moved to another
    /// generation meanwhile: this build's generation is no longer the one
    /// to fill, and it writes nothing more into it.
    #[error("the index definition changed while it was being built")]
    Superseded,
    /// A duplicate the build may repair could not be: its value is not a
    /// string, no free value was found, or the repair statement failed.
    #[error("the build could not repair a duplicate: {0}")]
    Repair(String),
    /// The process running the backfill is closing: it stopped between
    /// pages, leaving the build to whoever opens the storage next.
    #[error("the index backfill stopped because its process is closing")]
    Stopped,
    /// The label names of a page could not be resolved to dictionary ids.
    #[error("the label lookup build could not resolve label names: {0}")]
    Names(String),
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
    /// Whether the definition built has a stored record each page is bound
    /// to: a page commits only while that record is still the definition
    /// built, so a page of a build whose index was dropped or recreated
    /// meanwhile writes nothing. `false` for a definition without one.
    pub bind_definition: bool,
    /// How long the backfill waits for the transactions opened before the
    /// index was registered.
    pub older_transactions_wait: std::time::Duration,
    /// Told where the backfill stands: waiting for older transactions,
    /// then the entries committed so far after every page.
    pub progress: Option<&'a dyn Fn(BackfillProgress)>,
    /// Told, after every committed page, the last node key of the shard the
    /// backfill has committed entries through.
    pub covered: Option<CoveredThrough<'a>>,
    /// How a stored duplicate is repaired, when the build may repair one:
    /// the page holding it does not commit, the repair runs, and the page is
    /// read again. `None` fails the backfill on the first duplicate.
    pub repair: Option<DuplicateRepairer<'a>>,
    /// Set when the process is closing: the backfill stops before its next
    /// page or wait slice with [`BackfillError::Stopped`].
    pub stop: Option<&'a core::sync::atomic::AtomicBool>,
    /// Resolves label names to their dictionary ids, registering the ones
    /// without one, for a label lookup build: stored nodes may hold labels
    /// no write has registered yet. `None` for any other build.
    pub labels: Option<ResolveNames<'a>>,
    /// A seqno past which every node row was written by a writer that
    /// maintained the index, known when the index was registered with no
    /// transaction open: the rows past it are passed over. `None`: learned
    /// by waiting for the transactions opened before the index.
    pub maintained_after: Option<u64>,
}

/// What a label lookup build resolves a page's label names through: the ids
/// of the names given, in order, each registered first if it has none.
pub type ResolveNames<'a> = &'a dyn Fn(&[&str]) -> Result<Vec<u32>, String>;

/// What a backfill does about a stored node whose value of a unique index
/// another node holds.
#[derive(Clone, Copy)]
pub struct DuplicateRepairer<'a> {
    /// The property the repair changes.
    pub property: &'a str,
    /// Repair node `node`, whose value of `property` is the one given, or
    /// find that a writer changed it meanwhile; the page is read again
    /// either way.
    pub run: RepairNode<'a>,
}

/// What a backfill calls to repair one node: the node and its value of the
/// repaired property.
pub type RepairNode<'a> = &'a dyn Fn(NodeId, Option<&Value>) -> Result<(), BackfillError>;

/// Repairs one page may need before it commits: each one ends a duplicate
/// the page met, so a page meets at most one per node it holds.
const MAX_PAGE_REPAIRS: usize = PAGE;

/// What a backfill tells the key it has committed entries through.
pub type CoveredThrough<'a> = &'a dyn Fn(&[u8]);

/// Where a running backfill stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackfillProgress {
    /// Waiting for the transactions opened before the index existed to end.
    AwaitingOlderTransactions,
    /// Scanning; the entries committed so far.
    Indexed(u64),
}

/// How long a backfill waits for the transactions opened before its index
/// by default.
pub const DEFAULT_OLDER_TRANSACTIONS_WAIT: std::time::Duration = std::time::Duration::from_secs(60);

impl<'a> Backfill<'a> {
    /// Stage the label postings of the ordinary nodes of `rows`, one page of
    /// the shard, in the label lookup `index`, appending the node keys read
    /// to `read`; returns the postings staged. The page's label names are
    /// resolved in one batch. A row last written after `maintained_after`
    /// was written by a writer that staged its postings itself, and is
    /// passed over rather than written twice.
    fn stage_page_labels(
        &self,
        txn: &mut Transaction<'a>,
        index: &IndexDefinition,
        rows: &[(Vec<u8>, Vec<u8>)],
        maintained_after: u64,
        read: &mut Vec<Vec<u8>>,
    ) -> Result<u64, BackfillError> {
        let resolve = self.labels.ok_or_else(|| {
            BackfillError::Store(StoreError::Invariant(
                "a label lookup build needs the dictionary to resolve label names".into(),
            ))
        })?;
        let mut members: Vec<(NodeId, Vec<String>)> = Vec::with_capacity(rows.len());
        for (key, bytes) in rows {
            // A version of a temporal node gives the node every label it
            // holds: its history keeps them.
            let Some(node) = decode_node_key(key)
                .map(|(_, id)| id)
                .or_else(|| decode_temporal_node_key(key).map(|(_, id, _)| id))
            else {
                continue;
            };
            if txn
                .record_version(coordinode_storage::engine::partition::Partition::Node, key)
                .map_err(StoreError::from)?
                .is_some_and(|version| version > maintained_after)
            {
                continue;
            }
            let labels =
                NodeRecord::labels_from_msgpack(bytes).map_err(|e| StoreError::Decode {
                    kind: "node labels",
                    message: format!("node {}: {e}", node.as_raw()),
                })?;
            members.push((node, labels));
            read.push(key.clone());
        }
        let mut names: Vec<&str> = members
            .iter()
            .flat_map(|(_, labels)| labels.iter().map(String::as_str))
            .collect();
        names.sort_unstable();
        names.dedup();
        if names.is_empty() {
            return Ok(0);
        }
        let ids = resolve(&names).map_err(BackfillError::Names)?;
        if ids.len() != names.len() {
            return Err(BackfillError::Names(format!(
                "{} ids resolved for {} label names",
                ids.len(),
                names.len()
            )));
        }
        let id_of = |name: &str| {
            names
                .binary_search(&name)
                .ok()
                .and_then(|at| ids.get(at).copied())
        };
        let store = coordinode_modality::LocalIndexStore::new(self.engine);
        let mut staged = 0u64;
        for (node, labels) in &members {
            let added: Vec<u32> = labels.iter().filter_map(|l| id_of(l)).collect();
            store.stage_label_postings_txn(txn, index, self.shard_id, *node, &[], &added)?;
            // One posting per label held; a node holds few labels.
            staged += added.len() as u64;
        }
        Ok(staged)
    }

    /// [`BackfillError::Stopped`] once the process is closing.
    fn check_stop(&self) -> Result<(), BackfillError> {
        match self.stop {
            Some(stop) if stop.load(core::sync::atomic::Ordering::Acquire) => {
                Err(BackfillError::Stopped)
            }
            _ => Ok(()),
        }
    }

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
        let report = |p| {
            if let Some(progress) = self.progress {
                progress(p);
            }
        };
        report(BackfillProgress::AwaitingOlderTransactions);
        await_older_transactions(self.engine, self.older_transactions_wait, self.stop)?;
        // Every transaction opened before the index was registered has ended,
        // so a node row written after this point was written by one that
        // maintained the index itself.
        let maintained_after = self
            .maintained_after
            .unwrap_or_else(|| self.engine.snapshot());
        report(BackfillProgress::Indexed(0));
        let nodes = LocalNodeStore;
        let prefix = nodes.shard_scan_prefix(self.shard_id);
        let mut start_after: Option<Vec<u8>> = None;
        let mut indexed = 0u64;
        let mut conflicts = 0u32;
        let mut page_repairs = 0usize;
        loop {
            self.check_stop()?;
            let mut txn = match self.oracle {
                Some(oracle) => Transaction::begin(self.engine, Some(oracle), oracle.next()),
                None => Transaction::new(self.engine, None, Timestamp::ZERO, None),
            };
            if self.bind_definition {
                // Read for each page: a version held across the build could
                // stop matching a record storage renumbered without a write.
                let Some(version) = coordinode_modality::LocalIndexStore::new(self.engine)
                    .definition_version_as(index)?
                else {
                    return Err(BackfillError::Superseded);
                };
                txn.bind_index_definition(&index.schema_key(), Some(version))
                    .map_err(StoreError::from)?;
            }
            let page =
                nodes.prefix_scan_paged_tracked(&mut txn, &prefix, start_after.as_deref(), PAGE)?;
            let mut claims = Vec::new();
            let mut read = Vec::with_capacity(page.rows.len());
            let mut staged = 0u64;
            let mut duplicate = None;
            if index.index_type == super::definition::IndexType::Labels {
                staged = self.stage_page_labels(
                    &mut txn,
                    index,
                    &page.rows,
                    maintained_after,
                    &mut read,
                )?;
            }
            for (key, bytes) in &page.rows {
                if index.index_type == super::definition::IndexType::Labels {
                    break;
                }
                let Some(owner) = stored_owner(key) else {
                    continue;
                };
                let record = decode_record(bytes)?;
                if record.primary_label() != index.label {
                    continue;
                }
                let lookup = record_lookup(&record, self.interner);
                let field_of = |name: &str| self.interner.lookup(name);
                match stage_node_entry(
                    self.engine,
                    &mut txn,
                    self.shard_id,
                    index,
                    owner,
                    &lookup,
                    &field_of,
                    &mut claims,
                ) {
                    Ok(true) => staged += 1,
                    Ok(false) => {}
                    Err(IndexWriteError::Unique(_)) if self.repair.is_some() => {
                        let property = self.repair.map_or("", |r| r.property);
                        duplicate = Some((NodeId::from_raw(owner.node_id), lookup(property)));
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
                read.push(key.clone());
            }
            // A duplicate the build may repair: the page commits nothing, the
            // node is repaired in a transaction of its own, and the page is
            // read again over the repaired data.
            if let (Some((node, value)), Some(repair)) = (duplicate, self.repair) {
                drop(txn);
                page_repairs += 1;
                if page_repairs > MAX_PAGE_REPAIRS {
                    return Err(BackfillError::Contended);
                }
                (repair.run)(node, value.as_ref())?;
                continue;
            }
            page_repairs = 0;
            // The page's entries hold only if the rows they came from are
            // still the rows it read when the page commits.
            let unchanged = nodes.condition_unchanged(&mut txn, &read)?;
            let committed = if unchanged {
                match commit(&mut txn) {
                    Ok(()) => true,
                    Err(CommitError::Conflict(_) | CommitError::RevisionMismatch { .. }) => {
                        // A conflict over the page's nodes is read again; a
                        // definition that moved is not this build's any more.
                        if self.bind_definition
                            && coordinode_modality::LocalIndexStore::new(self.engine)
                                .definition_version_as(index)?
                                .is_none()
                        {
                            return Err(BackfillError::Superseded);
                        }
                        false
                    }
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
            report(BackfillProgress::Indexed(indexed));
            if let (Some(covered), Some(last)) = (self.covered, page.last_key.as_deref()) {
                covered(last);
            }
            conflicts = 0;
            if page.exhausted {
                return Ok(indexed);
            }
            start_after = page.last_key;
        }
    }
}

/// The owner of the index entries of the stored node row under `key`: the
/// node, or for a temporal node the version the key names, as its writers
/// stage an entry for each version they write. `None` for a key that is not
/// a node row.
fn stored_owner(key: &[u8]) -> Option<EntryOwner> {
    match decode_node_key(key) {
        Some((_, node_id)) => Some(EntryOwner::node(node_id.as_raw())),
        None => decode_temporal_node_key(key)
            .map(|(_, node_id, valid_from)| EntryOwner::version(node_id.as_raw(), valid_from)),
    }
}

/// A stored node row's record.
fn decode_record(bytes: &[u8]) -> Result<NodeRecord, StoreError> {
    NodeRecord::from_msgpack(bytes).map_err(|e| StoreError::Decode {
        kind: "node record",
        message: e.to_string(),
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
