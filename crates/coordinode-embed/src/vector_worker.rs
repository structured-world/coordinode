//! Incremental HNSW maintenance from the applied Raft entries.
//!
//! [`VectorIndexWorker`] follows the entries this node's Raft state machine
//! applies and keeps every registered vector index current with the node
//! records they wrote. It works from what has applied, never from the log:
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
use std::time::Duration;

use coordinode_core::graph::intern::FieldRegistrar;
use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_query::index::{BuildTarget, BuildToken, VectorBuild, VectorIndexRegistry};
use coordinode_storage::engine::applied::{AppliedEvent, AppliedSubscription};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::transaction::Transaction;
use rustc_hash::FxHashSet;

/// How long the worker waits for an entry before checking whether it should
/// stop.
const IDLE_POLL: Duration = Duration::from_millis(50);

/// Most applied entries folded together, bounding how long one fold keeps
/// the indexes' write locks busy.
const BATCH: usize = 256;

/// Background thread that keeps this process's vector indexes current with
/// the Raft entries applied on this node.
pub struct VectorIndexWorker {
    stop: BuildToken,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl VectorIndexWorker {
    /// Spawn the worker on `applied`, a subscription to the Node partition
    /// opened before the indexes were last built from the store, so no
    /// applied entry falls between the two. `shard_id` is the shard whose
    /// node rows the indexes hold.
    pub fn spawn(
        engine: Arc<StorageEngine>,
        applied: AppliedSubscription,
        registry: Arc<VectorIndexRegistry>,
        fields: Arc<dyn FieldRegistrar>,
        shard_id: u16,
    ) -> Self {
        let stop = registry.new_build_token();
        let worker = Worker {
            engine,
            applied,
            registry,
            fields,
            shard_id,
            stop: stop.clone(),
        };
        let handle = std::thread::Builder::new()
            .name("vec-applied-worker".to_string())
            .spawn(move || worker.run())
            .map_err(|e| tracing::error!(error = %e, "could not start the vector index worker"))
            .ok();
        Self { stop, handle }
    }

    /// Signal the worker to stop and wait for it to exit. A rebuild in
    /// progress stops at its next check.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.stop.cancel();
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
    registry: Arc<VectorIndexRegistry>,
    /// Read afresh for each fold: an entry can carry a property registered
    /// after the worker started.
    fields: Arc<dyn FieldRegistrar>,
    shard_id: u16,
    /// Cancelled when the worker is asked to stop; also stops its rebuilds.
    stop: BuildToken,
}

impl Worker {
    fn run(self) {
        tracing::info!("vector index worker started");
        while !self.stop.is_cancelled() {
            let Some(first) = self.applied.next(IDLE_POLL) else {
                continue;
            };
            let mut keys: FxHashSet<Vec<u8>> = FxHashSet::default();
            let mut replaced = false;
            let mut max_ts = 0u64;
            let mut taken = 0usize;
            let mut event = Some(first);
            while let Some(current) = event {
                match current {
                    AppliedEvent::Keys {
                        commit_ts,
                        keys: written,
                        ..
                    } => {
                        max_ts = max_ts.max(commit_ts);
                        keys.extend(written);
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

            if replaced {
                self.rebuild();
            } else {
                self.fold(keys);
            }
            // Every entry up to `max_ts` is now in the indexes, and so is
            // every write at or below it that these entries did not touch.
            if max_ts > 0 {
                self.registry.advance_indexed_hlc_all(max_ts);
            }
        }
        tracing::info!("vector index worker stopped");
    }

    /// Bring the nodes behind `keys` into every index that covers them.
    fn fold(&self, keys: FxHashSet<Vec<u8>>) {
        let ids: Vec<NodeId> = keys
            .iter()
            .filter_map(|key| coordinode_core::graph::node::decode_node_key(key))
            .filter(|(shard, _)| *shard == self.shard_id)
            .map(|(_, id)| id)
            .collect();
        if ids.is_empty() {
            return;
        }
        let read = Transaction::new(&self.engine, None, Timestamp::ZERO, None);
        let records = match LocalNodeStore.get_many(&read, self.shard_id, &ids) {
            Ok(records) => records,
            Err(e) => {
                // The entries are applied and stay in the store; reading them
                // all again is the one way not to drop these.
                tracing::warn!(error = %e, nodes = ids.len(), "vector index worker could not read applied nodes; rebuilding");
                drop(read);
                self.rebuild();
                return;
            }
        };
        let interner = match self.fields.view() {
            Ok(view) => view,
            Err(e) => {
                tracing::error!(error = %e, "vector index worker cannot read the field dictionary");
                return;
            }
        };
        for (node_id, record) in ids.into_iter().zip(records) {
            // A deleted node stays in the graph as a stale entry the read path
            // re-validates, the same as a write-path delete leaves it.
            let Some(record) = record else {
                continue;
            };
            let label = record.primary_label();
            for property in self.registry.indexed_properties(label) {
                let Some(field_id) = interner.lookup(&property) else {
                    continue;
                };
                let Some(value) = record.props.get(&field_id) else {
                    continue;
                };
                let Some(vector) = crate::db::try_extract_vector(value) else {
                    continue;
                };
                self.registry
                    .on_vector_written(label, node_id, &property, &vector);
            }
        }
    }

    /// Rebuild every index from the store: what changed is not known.
    fn rebuild(&self) {
        let definitions = self.registry.all_definitions();
        let interner = match self.fields.view() {
            Ok(view) => view,
            Err(e) => {
                tracing::error!(error = %e, "vector index worker cannot read the field dictionary");
                return;
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
            return;
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
            Ok(outcome) => tracing::info!(?outcome, "vector index worker rebuild done"),
            Err(reason) => {
                tracing::warn!(%reason, "vector index worker rebuild failed");
                for (_, _, health, _) in &members {
                    health.mark_offline(reason.clone());
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
