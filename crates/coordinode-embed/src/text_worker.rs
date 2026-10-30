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
use coordinode_query::index::membership::LabelCarriers;
use coordinode_query::index::selection::Held;
use coordinode_query::index::text_registry::{read_nodes, stored_texts};
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
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            tracing::error!("the text index worker panicked");
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
        self.sync_selections_logged();
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
                    // Woken for a change of what a type implies, or for
                    // ended temporal states.
                    self.sync_selections_logged();
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
                // Judged by the interpretations the index holds as it is
                // written, never by a catalog read before: a change the
                // worker has not followed yet leaves the documents in the
                // interpretation readers are answered with. A node without
                // text is taken out only if the index holds it, so writes to
                // other labels cost the index nothing.
                self.registry.apply_rows(
                    (&def.label, property),
                    &interner,
                    (&nodes, now),
                    None,
                    |held: &Held| {
                        // An index not built yet is written by its build.
                        if held.base.is_none() && held.candidate.is_none() {
                            return None;
                        }
                        Some((
                            held.base.clone().flatten(),
                            held.candidate.as_ref().map(|(c, _)| c.clone()),
                        ))
                    },
                )?;
            }
        }
        Ok(())
    }

    /// The carriers of `label` in force and with every published candidate
    /// put in force, both of the one catalog state `snapshot` reads.
    fn interpretations(
        &self,
        label: &str,
        snapshot: coordinode_storage::engine::StorageSnapshot,
    ) -> Result<(Option<LabelCarriers>, Option<LabelCarriers>), String> {
        use coordinode_query::index::membership::{CatalogMembership, label_carriers_in};
        let failed =
            |e: coordinode_modality::StoreError| format!("read the carriers of :{label}: {e}");
        let in_force = label_carriers_in(
            &self.engine,
            label,
            &mut CatalogMembership::at(&self.engine, snapshot),
        )
        .map_err(failed)?;
        let candidate = label_carriers_in(
            &self.engine,
            label,
            &mut CatalogMembership::candidates_at(&self.engine, snapshot),
        )
        .map_err(failed)?;
        Ok((in_force, candidate))
    }

    /// [`Self::sync_selections`], a failure logged: searches stay exact
    /// meanwhile, reading from the store what the indexes do not hold, and
    /// the next wake-up tries again.
    fn sync_selections_logged(&self) {
        if let Err(e) = self.sync_selections() {
            tracing::warn!(error = %e, "text index worker could not follow a change of what a type implies");
        }
    }

    /// Bring every index in line with the interpretations the catalog holds
    /// of its label: while a change of what a type implies is certified,
    /// keep the documents of the candidate beside those in force; once it
    /// ended, bring them back to the one now in force.
    fn sync_selections(&self) -> Result<(), String> {
        let definitions = self.registry.definitions();
        if definitions.is_empty() {
            return Ok(());
        }
        let interner = self
            .fields
            .view()
            .map_err(|e| format!("read the field dictionary: {e}"))?;
        // One catalog state for every index: the facts in force and the
        // candidates are never of two different commits.
        let (snapshot, _pin) = self.engine.pin_latest_snapshot();
        for def in &definitions {
            let (in_force, candidate) = self.interpretations(&def.label, snapshot)?;
            for property in &def.properties {
                let Some((held, revision)) = self.registry.held_at(&def.label, property) else {
                    continue;
                };
                let base_holds = held.base.as_ref() == Some(&in_force);
                if candidate != in_force {
                    // A change is certified: the index keeps both.
                    if !base_holds {
                        self.rebuild_plain(def, property, &interner)?;
                        continue;
                    }
                    if held.candidate != Some((candidate.clone(), true)) {
                        self.prepare(def, property, (&in_force, &candidate), revision, &interner)?;
                    }
                    continue;
                }
                match held.candidate {
                    None if base_holds => {}
                    // The change the index followed was put in force.
                    Some((followed, true)) if followed == in_force => {
                        self.settle(def, property, (true, &in_force), revision)?;
                    }
                    // It was withdrawn: the facts the index's base holds stay.
                    Some(_) if base_holds => {
                        self.settle(def, property, (false, &in_force), revision)?;
                    }
                    // The index holds neither: it is written afresh.
                    _ => self.rebuild_plain(def, property, &interner)?,
                }
            }
        }
        Ok(())
    }

    /// Rebuild `def`'s index of `property` from the store under the facts in
    /// force alone.
    fn rebuild_plain(
        &self,
        def: &coordinode_query::index::IndexDefinition,
        property: &str,
        interner: &coordinode_core::graph::intern::FieldInterner,
    ) -> Result<(), String> {
        self.registry
            .rebuild_index(&def.label, property, || {
                stored_texts(
                    &self.engine,
                    self.shard_id,
                    interner,
                    &def.label,
                    property,
                    wall_clock_us(),
                )
            })
            .map(drop)
    }

    /// Write every stored node of `def`'s label in its scope between the
    /// facts in force and the candidate (`interpretations`), a batch per
    /// commit so searches are never held up, and record the index prepared
    /// for the candidate. Searches under the facts in force leave the gained
    /// documents out meanwhile, so they stay exact while it runs.
    ///
    /// Begun only while the index's interpretations are at `revision`, the
    /// one the decision to prepare was read at, and each later step only
    /// while they are at the revision the beginning set: a rebuild or
    /// another change of them meanwhile ends the pass, however equal the
    /// interpretations it leaves.
    fn prepare(
        &self,
        def: &coordinode_query::index::IndexDefinition,
        property: &str,
        (in_force, candidate): (&Option<LabelCarriers>, &Option<LabelCarriers>),
        revision: u64,
        interner: &coordinode_core::graph::intern::FieldInterner,
    ) -> Result<(), String> {
        use coordinode_query::index::text_registry::for_each_rows_batch;
        // Not prepared until the pass below wrote every node.
        let begun =
            self.registry
                .with_index_at(&def.label, property, Some(revision), |_, held| {
                    *held = Held {
                        base: Some(in_force.clone()),
                        candidate: Some((candidate.clone(), false)),
                    };
                    Ok(())
                })?;
        let Some(((), guard)) = begun else {
            return Ok(());
        };
        let mut interrupted = false;
        let now = wall_clock_us();
        for_each_rows_batch(&self.engine, self.shard_id, BATCH, &mut |rows| {
            let wrote = self.registry.apply_rows(
                (&def.label, property),
                interner,
                (rows, now),
                Some(guard),
                |_: &Held| Some((in_force.clone(), Some(candidate.clone()))),
            )?;
            if wrote {
                Ok(())
            } else {
                interrupted = true;
                Err("the index changed while it was prepared".to_string())
            }
        })
        .or_else(|e| if interrupted { Ok(()) } else { Err(e) })?;
        if interrupted {
            return Ok(());
        }
        // Writes applied while the pass ran are folded under both
        // interpretations by this worker after it, and searches read them
        // from the store until then.
        self.registry
            .with_index_at(&def.label, property, Some(guard), |_, held| {
                held.candidate = Some((candidate.clone(), true));
                Ok(())
            })?;
        Ok(())
    }

    /// Bring the documents of `def`'s label back to one interpretation once
    /// the change the index followed ended, in one step: put in force
    /// (`put_in_force`), the retired documents go and the gained ones count
    /// as shared; withdrawn, the other way round. The index then holds
    /// `in_force`, the facts now in force.
    ///
    /// Only while the index's interpretations are at `revision`, the one the
    /// decision to settle was read at: a rebuild, a re-registration or any
    /// other change of them since refuses it.
    fn settle(
        &self,
        def: &coordinode_query::index::IndexDefinition,
        property: &str,
        (put_in_force, in_force): (bool, &Option<LabelCarriers>),
        revision: u64,
    ) -> Result<(), String> {
        use coordinode_search::tantivy::Scope;
        let dropped = if put_in_force {
            Scope::Retired
        } else {
            Scope::Gained
        };
        self.registry
            .settle(
                (&def.label, property),
                dropped,
                revision,
                Held::built(in_force.clone()),
            )
            .map(drop)
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
