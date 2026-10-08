//! Full-text index maintenance from the applied commits.
//!
//! [`TextIndexWorker`] follows the entries this node applies (on an embedded
//! store without Raft, its local commits) and keeps every text index current
//! with the node records they wrote: a document is replaced when its node
//! holds the indexed text and removed when it no longer does (deleted,
//! relabelled, property removed or no longer a string). It works from what
//! has applied, so a statement that rolls back or a commit that is refused
//! never reaches an index, and every member, not only the one that ran the
//! statement, keeps its indexes current.
//!
//! The worker reads each record as it stands rather than the value the entry
//! carried, so a merge, a later write or a delete all reconcile the same way.
//! A partition replaced wholesale or a queue the worker fell behind on is
//! answered with a rebuild of every index from the store. After each fold or
//! rebuild the worker releases the events it covered; a search answers the
//! nodes of the events not released yet from the store (see
//! [`coordinode_query::index::IndexCoverage`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use coordinode_core::graph::intern::FieldRegistrar;
use coordinode_core::graph::node::{NodeId, decode_written_node};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_query::executor::runner::wall_clock_us;
use coordinode_query::index::text_registry::{NodeText, TextSource, read_nodes, stored_texts};
use coordinode_query::index::{IndexCoverage, TextIndexRegistry};
use coordinode_storage::engine::applied::{
    AppliedEvent, AppliedPosition, AppliedStop, AppliedSubscription,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use rustc_hash::FxHashSet;

/// Most applied entries folded together, bounding how long one fold holds
/// the indexes' write locks and how much one Tantivy commit carries.
const BATCH: usize = 256;

/// The least wait before refolding ended temporal states again after an
/// attempt failed.
const REFOLD_RETRY: Duration = Duration::from_secs(1);

/// Background thread that keeps this process's text indexes current with the
/// entries applied on this node.
pub struct TextIndexWorker {
    stop: Arc<AtomicBool>,
    /// Ends the worker's wait for the next applied entry.
    applied_stop: AppliedStop,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TextIndexWorker {
    /// Spawn the worker on `applied`, a retained subscription to the Node
    /// partition opened before the indexes were last built from the store, so
    /// no applied entry falls between the two. It releases the events it has
    /// folded through `coverage`. `shard_id` is the shard whose node rows the
    /// indexes hold.
    pub fn spawn(
        engine: Arc<StorageEngine>,
        applied: AppliedSubscription,
        registry: Arc<TextIndexRegistry>,
        fields: Arc<dyn FieldRegistrar>,
        coverage: Arc<IndexCoverage>,
        shard_id: u16,
    ) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let applied_stop = applied.stopper();
        let worker = Worker {
            engine,
            position: applied.position(),
            applied,
            registry,
            fields,
            coverage,
            shard_id,
            stop: Arc::clone(&stop),
        };
        let handle = std::thread::Builder::new()
            .name("text-applied-worker".to_string())
            .spawn(move || worker.run())
            .map_err(|e| tracing::error!(error = %e, "could not start the text index worker"))
            .ok();
        Self {
            stop,
            applied_stop,
            handle,
        }
    }

    /// Signal the worker to stop and wait for it to exit.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.applied_stop.stop();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::error!("the text index worker panicked");
            }
        }
    }
}

impl Drop for TextIndexWorker {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

struct Worker {
    engine: Arc<StorageEngine>,
    applied: AppliedSubscription,
    /// How far the store's events were offered, read before a rebuild.
    position: AppliedPosition,
    registry: Arc<TextIndexRegistry>,
    /// Read afresh for each fold: an entry can carry a property registered
    /// after the worker started.
    fields: Arc<dyn FieldRegistrar>,
    coverage: Arc<IndexCoverage>,
    shard_id: u16,
    stop: Arc<AtomicBool>,
}

impl Worker {
    fn run(self) {
        tracing::info!("text index worker started");
        // Set while events the indexes do not hold are taken but unreleased:
        // releasing a later batch would release them too, so every round
        // rebuilds until one rebuild covers them.
        let mut behind = false;
        let mut refold_failed = false;
        while !self.stop.load(Ordering::Acquire) {
            // Asleep until an entry applies, a temporal node's indexed state
            // stops holding, or the worker is stopped.
            // After a failed refold the ended states stay due: wait a while
            // before the next attempt instead of spinning on them.
            let wait = match self.until_next_end() {
                Some(wait) if refold_failed => Some(wait.max(REFOLD_RETRY)),
                wait => wait,
            };
            let Some(first) = self.applied.next(wait) else {
                if !self.stop.load(Ordering::Acquire) {
                    refold_failed = self.refold_ended().is_err();
                }
                continue;
            };
            let mut keys: FxHashSet<Vec<u8>> = FxHashSet::default();
            let mut replaced = false;
            let mut last_seq = 0u64;
            let mut taken = 0usize;
            let mut event = Some(first);
            while let Some(current) = event {
                match current {
                    AppliedEvent::Keys {
                        seq, keys: written, ..
                    } => {
                        last_seq = last_seq.max(seq);
                        keys.extend(written.iter().cloned());
                    }
                    AppliedEvent::Replaced => replaced = true,
                }
                taken += 1;
                event = if taken < BATCH {
                    self.applied.try_next()
                } else {
                    None
                };
            }

            let covered = if replaced || behind {
                self.rebuild()
            } else {
                match self.fold(&keys) {
                    Ok(()) => Some(last_seq),
                    Err(e) => {
                        // The entries are applied and stay in the store;
                        // reading them all again is the one way not to drop
                        // these.
                        tracing::warn!(error = %e, "text index worker could not fold applied nodes; rebuilding");
                        self.rebuild()
                    }
                }
            };
            // A failed rebuild releases nothing: searches keep answering those
            // nodes from the store until a later round's rebuild covers them.
            match covered {
                Some(seq) => {
                    behind = false;
                    self.coverage.release(seq);
                }
                None => behind = true,
            }
        }
        tracing::info!("text index worker stopped");
    }

    /// How long until the earliest indexed state of a temporal node stops
    /// holding; `None` (no limit) when none ends.
    fn until_next_end(&self) -> Option<Duration> {
        let end = self.registry.first_end()?;
        let now = wall_clock_us();
        // `now` is never negative, so a later `end` minus it cannot overflow.
        let wait = if end <= now { 0 } else { end - now };
        Some(Duration::from_micros(wait.unsigned_abs()))
    }

    /// Fold again the temporal nodes whose indexed state no longer holds now,
    /// with no write: valid time moved past the end of their state. Searches
    /// read such nodes from their timelines meanwhile, so this only keeps
    /// them from doing so for good.
    fn refold_ended(&self) -> Result<(), String> {
        let now = wall_clock_us();
        let mut ids: Vec<NodeId> = self
            .registry
            .definitions()
            .iter()
            .flat_map(|def| {
                def.properties
                    .iter()
                    .flat_map(|property| self.registry.outside(&def.label, property, now))
                    .collect::<Vec<_>>()
            })
            .collect();
        ids.sort_unstable();
        ids.dedup();
        self.fold_nodes(&ids).inspect_err(|e| {
            // A later wake-up tries again; searches stay exact meanwhile.
            tracing::warn!(error = %e, nodes = ids.len(), "text index worker could not refold ended states");
        })
    }

    /// Bring the documents of the nodes behind `keys` in line with their
    /// records in every text index.
    fn fold(&self, keys: &FxHashSet<Vec<u8>>) -> Result<(), String> {
        let mut ids: Vec<NodeId> = keys
            .iter()
            .filter_map(|key| decode_written_node(key))
            .filter(|(shard, _)| *shard == self.shard_id)
            .map(|(_, id)| id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        self.fold_nodes(&ids)
    }

    /// Bring the documents of `ids` (sorted) in line with the store in every
    /// text index: a temporal node at its state valid now, with the interval
    /// that state holds over.
    fn fold_nodes(&self, ids: &[NodeId]) -> Result<(), String> {
        // A store without text indexes pays no read for its commits.
        let definitions = self.registry.definitions();
        if definitions.is_empty() || ids.is_empty() {
            return Ok(());
        }
        let read = Transaction::new(&self.engine, None, Timestamp::ZERO, None);
        let nodes = read_nodes(&read, self.shard_id, ids)?;
        let interner = self
            .fields
            .view()
            .map_err(|e| format!("read the field dictionary: {e}"))?;
        let now = wall_clock_us();
        for def in &definitions {
            for property in &def.properties {
                let source = TextSource::new(&interner, &def.label, property);
                // A node without text is taken out only if the index holds
                // it, so writes to other labels cost the index nothing.
                let changes: Vec<NodeText> = nodes
                    .iter()
                    .map(|(id, rows)| source.at(*id, rows, now))
                    .collect();
                self.registry
                    .apply_changes(&def.label, property, &changes)?;
            }
        }
        Ok(())
    }

    /// Rebuild every index from the store, since what changed is not known;
    /// the event the rebuilt indexes cover, or `None` when one failed.
    fn rebuild(&self) -> Option<u64> {
        // Every event numbered up to here is in the store the scans read.
        let covered = self.position.delivered();
        let interner = match self.fields.view() {
            Ok(view) => view,
            Err(e) => {
                tracing::error!(error = %e, "text index worker cannot read the field dictionary");
                return None;
            }
        };
        let definitions = self.registry.definitions();
        tracing::info!(
            indexes = definitions.len(),
            "text index worker rebuilding from the store"
        );
        let mut failed = false;
        for def in &definitions {
            for property in &def.properties {
                let rebuilt = self.registry.rebuild_index(&def.label, property, || {
                    stored_texts(
                        &self.engine,
                        self.shard_id,
                        &interner,
                        &def.label,
                        property,
                        wall_clock_us(),
                    )
                });
                if let Err(e) = rebuilt {
                    tracing::warn!(error = %e, label = %def.label, property, "text index rebuild failed");
                    failed = true;
                }
            }
        }
        (!failed).then_some(covered)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
