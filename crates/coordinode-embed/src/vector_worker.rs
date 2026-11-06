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
use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_modality::{LocalNodeStore, NodeStore as _};
use coordinode_query::index::membership::LabelCarriers;
use coordinode_query::index::selection::Held;
use coordinode_query::index::{
    BuildTarget, BuildToken, IndexCoverage, VectorBuild, VectorIndexRegistry,
};
use coordinode_search::tantivy::Scope;
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
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            tracing::error!("the vector index worker panicked");
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
        let mut stale = self.sync_selections_logged();
        while !self.stop.is_cancelled() {
            // An index holding neither interpretation of its label is
            // rebuilt under the one in force, then follows a change still
            // certified like any other.
            if stale {
                match self.rebuild() {
                    Some(seq) => {
                        behind = false;
                        self.coverage.release(seq);
                        stale = self.sync_selections_logged();
                    }
                    None => {
                        behind = true;
                        stale = false;
                    }
                }
                continue;
            }
            // Asleep until an entry applies, the catalog changes, or the
            // worker is stopped.
            let Some(first) = self.applied.next(None) else {
                // Woken for a change of what a type implies.
                if !self.stop.is_cancelled() {
                    stale = self.sync_selections_logged();
                }
                continue;
            };
            // A complete cut: every commit at or below it has landed here,
            // and a commit offers its event before it becomes visible, so
            // each of theirs is among the events offered by now. Once those
            // are in the indexes, so is every write at or below the cut,
            // whatever order the commits took their timestamps in or reached
            // this member's log in.
            let cut = self.engine.complete_cut();
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
                        tracing::warn!(error = %e, "vector index worker could not fold applied nodes; rebuilding");
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
            // Judged by the interpretations the index holds as it is written,
            // never by a catalog read before: a change the worker has not
            // followed yet leaves the members in the interpretation readers
            // are answered with. An index without a record covers the
            // carriers in force.
            let (interpretations, revision) = match self.registry.held_at(&def.label, property) {
                Some((held, revision)) => (Interpretations::of(&held), Some(revision)),
                None => (
                    Interpretations::In(
                        coordinode_query::index::membership::label_carriers(
                            &self.engine,
                            &def.label,
                        )
                        .map_err(|e| format!("read the carriers of :{}: {e}", def.label))?,
                    ),
                    None,
                ),
            };
            let mut upserts: Vec<(NodeId, Vec<f32>)> = Vec::new();
            for (node_id, record) in ids.iter().copied().zip(&records) {
                let scope = record
                    .as_ref()
                    .and_then(|r| interpretations.scope(&r.labels));
                let vector = record
                    .as_ref()
                    .filter(|_| scope.is_some())
                    .and_then(|r| r.props.get(&interner.lookup(property)?))
                    .and_then(crate::db::try_extract_vector);
                match (vector, scope, revision) {
                    (Some(vector), Some(scope), revision) => {
                        // Recorded before it is inserted, so no search takes
                        // it for a member of both meanwhile.
                        if let Some(revision) = revision
                            && !self.registry.mark(
                                (&def.label, property),
                                revision,
                                node_id.as_raw(),
                                scope,
                            )
                        {
                            // The record moved on: its own pass writes the node.
                            continue;
                        }
                        upserts.push((node_id, vector));
                    }
                    _ if self.registry.holds(&def.label, property, node_id) => {
                        self.registry
                            .on_vector_removed(&def.label, property, node_id);
                        if let Some(revision) = revision {
                            self.registry.mark(
                                (&def.label, property),
                                revision,
                                node_id.as_raw(),
                                None,
                            );
                        }
                    }
                    _ => {}
                }
            }
            // A write is refused before it commits a vector its index cannot
            // take, so a refusal here is a store the indexes cannot hold: the
            // rebuild that follows reports it as the index's offline reason.
            self.registry
                .on_vectors_written(&def.label, property, upserts)
                .map_err(|e| format!("index :{}({property}): {e}", def.label))?;
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
            let carriers = match coordinode_query::index::membership::label_carriers(
                &self.engine,
                &def.label,
            ) {
                Ok(carriers) => carriers,
                Err(e) => {
                    tracing::error!(index = %def, error = %e, "vector index worker cannot read the carriers of its label");
                    return None;
                }
            };
            members.push((def, hnsw, health, field_id, carriers));
        }
        drop(interner);
        if members.is_empty() {
            return Some(covered);
        }
        for (def, _, health, _, carriers) in &members {
            // A build still running on the index (the one its CREATE started)
            // fills it under the carriers it read when it began: its last fold
            // would take out the members this rebuild inserts under the ones
            // in force now. It stops, and is waited for, before anything here
            // writes the index.
            self.registry.cancel_build(def.generation);
            // Incomplete until the build hands it over again.
            health.report_rebuild_progress(0.0, 0);
            // Filled with these carriers' members alone from here.
            for (id, _) in self.registry.moved(&def.label, def.property()) {
                self.registry
                    .on_vector_removed(&def.label, def.property(), NodeId::from_raw(id));
            }
            self.registry
                .record_built(&def.label, def.property(), carriers.clone());
        }
        let targets: Vec<BuildTarget<'_>> = members
            .iter()
            .map(|(_, hnsw, health, field_id, carriers)| BuildTarget {
                hnsw: hnsw.as_ref(),
                health: health.as_ref(),
                carriers: carriers.as_ref(),
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
                for (_, _, health, _, _) in &members {
                    health.mark_offline(reason.clone());
                }
                None
            }
        }
    }
}

/// The interpretations of its label a fold judges a node's membership in an
/// index by.
enum Interpretations {
    /// The carriers in force alone: an index without a record of its own.
    In(Option<LabelCarriers>),
    /// What the index holds: its base and, while it follows a change of
    /// what a type implies, the candidate.
    Held {
        base: Option<LabelCarriers>,
        candidate: Option<Option<LabelCarriers>>,
    },
}

impl Interpretations {
    fn of(held: &Held) -> Self {
        Self::Held {
            base: held.base.clone().flatten(),
            candidate: held.candidate.as_ref().map(|(c, _)| c.clone()),
        }
    }

    /// Whether a node storing `stored` is a member, and if so, the one
    /// interpretation alone carrying it: `None` for not a member,
    /// `Some(None)` for one of both.
    fn scope(&self, stored: &[LabelId]) -> Option<Option<Scope>> {
        let carried = |c: &Option<LabelCarriers>| c.as_ref().is_some_and(|c| c.carried_by(stored));
        match self {
            Self::In(carriers)
            | Self::Held {
                base: carriers,
                candidate: None,
            } => carried(carriers).then_some(None),
            Self::Held {
                base,
                candidate: Some(candidate),
            } => match (carried(base), carried(candidate)) {
                (true, true) => Some(None),
                (true, false) => Some(Some(Scope::Retired)),
                (false, true) => Some(Some(Scope::Gained)),
                (false, false) => None,
            },
        }
    }
}

impl Worker {
    /// [`Self::sync_selections`], a failure logged: searches stay exact
    /// meanwhile, reading from the store what the indexes do not hold, and
    /// the next wake-up tries again.
    fn sync_selections_logged(&self) -> bool {
        self.sync_selections().unwrap_or_else(|e| {
            tracing::warn!(error = %e, "vector index worker could not follow a change of what a type implies");
            false
        })
    }

    /// Bring every index in line with the interpretations the catalog holds
    /// of its label: while a change of what a type implies is certified,
    /// keep the members of the candidate beside those in force; once it
    /// ended, bring them back to the one now in force. Whether an index
    /// holds neither, which only a rebuild brings back.
    fn sync_selections(&self) -> Result<bool, String> {
        use coordinode_query::index::membership::{CatalogMembership, label_carriers_in};
        let definitions = self.registry.all_definitions();
        if definitions.is_empty() {
            return Ok(false);
        }
        let mut rebuild = false;
        // One catalog state for every index: the facts in force and the
        // candidates are never of two different commits.
        let (snapshot, _pin) = self.engine.pin_latest_snapshot();
        for def in &definitions {
            let property = def.property();
            let failed = |e: coordinode_modality::StoreError| {
                format!("read the carriers of :{}: {e}", def.label)
            };
            let in_force = label_carriers_in(
                &self.engine,
                &def.label,
                &mut CatalogMembership::at(&self.engine, snapshot),
            )
            .map_err(failed)?;
            let candidate = label_carriers_in(
                &self.engine,
                &def.label,
                &mut CatalogMembership::candidates_at(&self.engine, snapshot),
            )
            .map_err(failed)?;
            // An index without a record is filled by its own build.
            let Some((held, revision)) = self.registry.held_at(&def.label, property) else {
                continue;
            };
            let base_holds = held.base.as_ref() == Some(&in_force);
            if candidate != in_force {
                // A change is certified: the index keeps both.
                if !base_holds {
                    // It holds neither: a rebuild writes it afresh.
                    rebuild = true;
                    continue;
                }
                if held.candidate != Some((candidate.clone(), true)) {
                    self.prepare(def, revision, (&in_force, &candidate))?;
                }
                continue;
            }
            match held.candidate {
                None if base_holds => {}
                // The change the index followed was put in force.
                Some((followed, true)) if followed == in_force => {
                    self.registry.settle_change(
                        (&def.label, property),
                        revision,
                        Scope::Retired,
                        in_force,
                    );
                }
                // It was withdrawn: the facts the index's base holds stay.
                Some(_) if base_holds => {
                    self.registry.settle_change(
                        (&def.label, property),
                        revision,
                        Scope::Gained,
                        in_force,
                    );
                }
                // The index holds facts the catalog has left, as on a member
                // that applied the change before following it: it follows
                // the change to the facts now in force from what it holds,
                // a read of those answering to the index's build policy
                // meanwhile, and settles it in the same pass.
                _ => match held.base.clone() {
                    // Built while the label had no identity, the index holds
                    // no member: its label gaining one (its first node) is
                    // no change to follow, and a rebuild writes it afresh.
                    Some(None) => rebuild = true,
                    Some(base) => {
                        self.prepare(def, revision, (&base, &in_force))?;
                        if let Some((followed, revision)) =
                            self.registry.held_at(&def.label, property)
                            && followed.candidate == Some((in_force.clone(), true))
                        {
                            self.registry.settle_change(
                                (&def.label, property),
                                revision,
                                Scope::Retired,
                                in_force,
                            );
                        }
                    }
                    // The index holds nothing whole: a rebuild writes it
                    // afresh.
                    None => rebuild = true,
                },
            }
        }
        Ok(rebuild)
    }

    /// Insert every stored member of `def`'s label only the candidate
    /// carries and record the ones only the facts in force carry, so
    /// searches under either leave the other's out; then record the index
    /// holding the candidate. Begun only while the index's record is at
    /// `revision`, and each later step only while it is at the revision the
    /// beginning set: a rebuild or another change meanwhile ends the pass.
    fn prepare(
        &self,
        def: &coordinode_query::index::IndexDefinition,
        revision: u64,
        (in_force, candidate): (&Option<LabelCarriers>, &Option<LabelCarriers>),
    ) -> Result<(), String> {
        let property = def.property();
        let Some(guard) = self.registry.begin_change(
            (&def.label, property),
            revision,
            in_force.clone(),
            candidate.clone(),
        ) else {
            return Ok(());
        };
        let interpretations = Interpretations::Held {
            base: in_force.clone(),
            candidate: Some(candidate.clone()),
        };
        let interner = self
            .fields
            .view()
            .map_err(|e| format!("read the field dictionary: {e}"))?;
        let Some(field) = interner.lookup(property) else {
            // No node holds the property: nothing to keep apart.
            self.registry.finish_change((&def.label, property), guard);
            return Ok(());
        };
        let read = Transaction::new(&self.engine, None, Timestamp::ZERO, None);
        let mut gained: Vec<(NodeId, Vec<f32>)> = Vec::new();
        let insert = |gained: &mut Vec<(NodeId, Vec<f32>)>| {
            self.registry
                .on_vectors_written(&def.label, property, std::mem::take(gained))
                .map_err(|e| coordinode_modality::StoreError::Invariant(e.to_string()))
        };
        // Set when the pass stops early, the worker stopping or the record
        // moving on: what it marked and inserted stays under a revision no
        // search answers the candidate with until a later pass finishes.
        let mut interrupted = false;
        let scanned =
            LocalNodeStore.for_each_in_shard(&read, self.shard_id, &mut |node_id, record| {
                let Some(Some(scope)) = interpretations.scope(&record.labels) else {
                    return Ok(());
                };
                let Some(vector) = record
                    .props
                    .get(&field)
                    .and_then(crate::db::try_extract_vector)
                else {
                    return Ok(());
                };
                if self.stop.is_cancelled()
                    || !self.registry.mark(
                        (&def.label, property),
                        guard,
                        node_id.as_raw(),
                        Some(scope),
                    )
                {
                    interrupted = true;
                    return Err(coordinode_modality::StoreError::Invariant(
                        "the pass was interrupted".to_string(),
                    ));
                }
                if scope == Scope::Gained {
                    gained.push((node_id, vector));
                    if gained.len() >= BATCH {
                        insert(&mut gained)?;
                    }
                }
                Ok(())
            });
        drop(read);
        if interrupted {
            return Ok(());
        }
        scanned.map_err(|e| format!("scan the nodes of :{}: {e}", def.label))?;
        insert(&mut gained).map_err(|e| format!("index :{}({property}): {e}", def.label))?;
        // Writes applied while the pass ran are folded under both
        // interpretations by this worker after it, and searches read them
        // from the store until then.
        self.registry.finish_change((&def.label, property), guard);
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
