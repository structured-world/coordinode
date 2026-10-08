//! Incremental HNSW maintenance from the applied commits.
//!
//! [`VectorIndexWorker`] follows the entries this node's Raft state machine
//! applies (on an embedded store without Raft, its local commits) and keeps
//! every registered vector index current with the node records they wrote,
//! removals included. It works from what has applied, never from the log:
//! an entry appended to the log may never commit, and one committed is not in
//! the store until it applies, so reading the log ahead of the applies would
//! insert vectors that never become data and pass over writes that land after
//! an index build has stopped watching. The graph itself is never
//! replicated; each member derives its own.
//!
//! The worker reads each record as it stands rather than the value the entry
//! carried, so a merge, a later write or a delete all reconcile the same way.
//! A partition replaced wholesale (a snapshot installed, a partition installed
//! from a peer) or a queue the worker fell behind on is answered with a
//! rebuild of every index from the store.

use std::sync::Arc;

use coordinode_core::graph::intern::FieldRegistrar;
use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_query::index::{
    BuildTarget, BuildToken, IndexCoverage, VectorBuild, VectorIndexRegistry,
};
use coordinode_storage::engine::applied::{
    AppliedEvent, AppliedPosition, AppliedStop, AppliedSubscription,
};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use rustc_hash::FxHashSet;

/// Most applied entries folded together, bounding how long one fold keeps
/// the indexes' write locks busy.
const BATCH: usize = 256;

/// Background thread that keeps this process's vector indexes current with
/// the Raft entries applied on this node.
pub struct VectorIndexWorker {
    stop: BuildToken,
    /// Ends the worker's wait for the next applied entry.
    applied_stop: AppliedStop,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VectorIndexWorker {
    /// Spawn the worker on `applied`, a retained subscription to the Node
    /// partition opened before the indexes were last built from the store,
    /// so no applied entry falls between the two. It releases the events it
    /// has folded through `coverage`. `shard_id` is the shard whose node rows
    /// the indexes hold.
    pub fn spawn(
        engine: Arc<StorageEngine>,
        applied: AppliedSubscription,
        registry: Arc<VectorIndexRegistry>,
        fields: Arc<dyn FieldRegistrar>,
        coverage: Arc<IndexCoverage>,
        shard_id: u16,
    ) -> Self {
        let stop = registry.new_build_token();
        let applied_stop = applied.stopper();
        let worker = Worker {
            engine,
            position: applied.position(),
            applied,
            registry,
            fields,
            coverage,
            shard_id,
            stop: stop.clone(),
        };
        let handle = std::thread::Builder::new()
            .name("vec-applied-worker".to_string())
            .spawn(move || worker.run())
            .map_err(|e| tracing::error!(error = %e, "could not start the vector index worker"))
            .ok();
        Self {
            stop,
            applied_stop,
            handle,
        }
    }

    /// Signal the worker to stop and wait for it to exit. A rebuild in
    /// progress stops at its next check.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.stop.cancel();
        self.applied_stop.stop();
        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() {
                tracing::error!("the vector index worker panicked");
            }
        }
    }
}

impl Drop for VectorIndexWorker {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

struct Worker {
    engine: Arc<StorageEngine>,
    applied: AppliedSubscription,
    /// How far the store's events were offered, read before a rebuild.
    position: AppliedPosition,
    registry: Arc<VectorIndexRegistry>,
    /// Read afresh for each fold: an entry can carry a property registered
    /// after the worker started.
    fields: Arc<dyn FieldRegistrar>,
    /// Where the events the indexes hold are released, so searches stop
    /// answering their nodes from the store.
    coverage: Arc<IndexCoverage>,
    shard_id: u16,
    /// Cancelled when the worker is asked to stop; also stops its rebuilds.
    stop: BuildToken,
}

impl Worker {
    fn run(self) {
        tracing::info!("vector index worker started");
        // Set while events the indexes do not hold are taken but unreleased:
        // releasing a later batch would release them too, so every round
        // rebuilds until one rebuild covers them.
        let mut behind = false;
        while !self.stop.is_cancelled() {
            // Asleep until an entry applies or the worker is stopped.
            let Some(first) = self.applied.next(None) else {
                continue;
            };
            // A complete snapshot: every commit below it has landed, and a
            // commit offers its event before it becomes visible, so each of
            // theirs is among the events offered by now. Once those are in
            // the indexes, so is every write below the snapshot, whatever
            // order the commits took their timestamps in; the cut is the last
            // timestamp it covers.
            let cut = self.engine.snapshot().checked_sub(1);
            let offered = self.position.delivered();
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
                match self.fold(keys) {
                    Ok(()) => Some(last_seq),
                    Err(e) => {
                        // The entries are applied and stay in the store;
                        // reading them all again is the one way not to drop
                        // these.
                        tracing::warn!(error = %e, "vector index worker could not read applied nodes; rebuilding");
                        self.rebuild()
                    }
                }
            };
            // A failed rebuild releases nothing and claims nothing: searches
            // keep answering those nodes from the store until a later round's
            // rebuild covers them.
            let Some(seq) = covered else {
                behind = true;
                continue;
            };
            behind = false;
            // The watermark reaches the cut only when this round covered
            // every event offered when it was taken; otherwise a later round
            // does.
            if let Some(cut) = cut.filter(|_| seq >= offered) {
                self.registry.advance_indexed_hlc_all(cut);
            }
            self.coverage.release(seq);
        }
        tracing::info!("vector index worker stopped");
    }

    /// Bring the nodes behind `keys` into every index that covers them.
    fn fold(&self, keys: FxHashSet<Vec<u8>>) -> Result<(), String> {
        // A store without vector indexes pays no read for its commits.
        let definitions = self.registry.all_definitions();
        if definitions.is_empty() {
            return Ok(());
        }
        let mut ids: Vec<NodeId> = keys
            .iter()
            .filter_map(|key| coordinode_core::graph::node::decode_node_key(key))
            .filter(|(shard, _)| *shard == self.shard_id)
            .map(|(_, id)| id)
            .collect();
        if ids.is_empty() {
            return Ok(());
        }
        // In id order, which is creation order: the graph an insert order
        // builds (and the quantizer it calibrates) is then the same on every
        // member and every run, not the order a hash set happened to yield.
        ids.sort_unstable();
        let read = Transaction::new(&self.engine, None, Timestamp::ZERO, None);
        let records = LocalNodeStore
            .get_many(&read, self.shard_id, &ids)
            .map_err(|e| format!("read {} applied nodes: {e}", ids.len()))?;
        drop(read);
        let interner = self
            .fields
            .view()
            .map_err(|e| format!("read the field dictionary: {e}"))?;
        // Every index is reconciled against the record as it stands: a member
        // is upserted, and a node that left an index (deleted, relabelled,
        // stripped of its vector) is removed from it. Node ids are never
        // reused, and every later commit that touches the node is folded
        // again, so acting on the committed record never leaves an index
        // behind the data. The upserts of one index go in as one batch, which
        // a bulk load fills with every node of a statement.
        for def in &definitions {
            let property = def.property();
            let mut upserts: Vec<(NodeId, Vec<f32>)> = Vec::new();
            for (node_id, record) in ids.iter().copied().zip(&records) {
                let vector = record
                    .as_ref()
                    .filter(|r| r.primary_label() == def.label)
                    .and_then(|r| r.props.get(&interner.lookup(property)?))
                    .and_then(crate::db::try_extract_vector);
                match vector {
                    Some(vector) => upserts.push((node_id, vector)),
                    None if self.registry.holds(&def.label, property, node_id) => {
                        self.registry
                            .on_vector_removed(&def.label, property, node_id);
                    }
                    None => {}
                }
            }
            self.registry
                .on_vectors_written(&def.label, property, upserts);
        }
        Ok(())
    }

    /// Rebuild every index from the store, since what changed is not known;
    /// the event the rebuilt indexes cover, or `None` when the rebuild
    /// failed.
    fn rebuild(&self) -> Option<u64> {
        // Every event numbered up to here is in the store the build reads.
        let covered = self.position.delivered();
        let definitions = self.registry.all_definitions();
        let interner = match self.fields.view() {
            Ok(view) => view,
            Err(e) => {
                tracing::error!(error = %e, "vector index worker cannot read the field dictionary");
                return None;
            }
        };
        let mut members = Vec::with_capacity(definitions.len());
        for def in &definitions {
            let (Some(hnsw), Some(health), Some(field_id)) = (
                self.registry.get(&def.label, def.property()),
                self.registry.health_handle(&def.label, def.property()),
                interner.lookup(def.property()),
            ) else {
                continue;
            };
            members.push((def, hnsw, health, field_id));
        }
        drop(interner);
        if members.is_empty() {
            return Some(covered);
        }
        for (_, _, health, _) in &members {
            // Incomplete until the build hands it over again.
            health.report_rebuild_progress(0.0, 0);
        }
        let targets: Vec<BuildTarget<'_>> = members
            .iter()
            .map(|(def, hnsw, health, field_id)| BuildTarget {
                hnsw: hnsw.as_ref(),
                health: health.as_ref(),
                label: &def.label,
                field_id: *field_id,
            })
            .collect();
        tracing::info!(
            indexes = targets.len(),
            "vector index worker rebuilding from the store"
        );
        let outcome = VectorBuild {
            engine: &self.engine,
            token: &self.stop,
            shard_id: self.shard_id,
            targets: &targets,
        }
        .run();
        match outcome {
            Ok(outcome) => {
                tracing::info!(?outcome, "vector index worker rebuild done");
                Some(covered)
            }
            Err(reason) => {
                tracing::warn!(%reason, "vector index worker rebuild failed");
                for (_, _, health, _) in &members {
                    health.mark_offline(reason.clone());
                }
                None
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
