//! Certification of label candidates, on the same executors, seats, records
//! and moves as index builds.
//!
//! The candidate definitions a publication changes are published in their
//! labels' schemas beside the facts in force, with one certification for
//! all of them admitted in the same commit. That commit drains or refuses
//! every writer still on its way under the revision before
//! (`CandidatePublished`), and writers from then on are held to both
//! (`CandidateAdmits`), so the executor checks the stored nodes against the
//! candidates in one pass, with nothing written under the old facts alone
//! still to land. Its last move puts in force every candidate the stored nodes
//! satisfy and withdraws the others, leaving their facts in force
//! untouched, in one commit conditioned on the record still being its own
//! and on each label's schema as it read it.

use coordinode_modality::{
    BuildState, IndexStore as _, LocalIndexStore, LocalSchemaStore, SchemaStore as _, StoreError,
};

use super::{
    BuildError, BuildPhase, Executed, GenerationId, IndexBuildOutcome, IndexBuildService, Shared,
    begin, cannot_move_here, is_contention, move_attempts, next_token, terminal_outcome, text,
};

impl IndexBuildService {
    /// Run the certification `generation` on an executor of this process,
    /// unless one already has it. [`Self::wait`] reports its outcome:
    /// `Published` when every candidate was put in force, `Failed` naming
    /// the labels whose candidate was refused (the others were put in force;
    /// the record tells which).
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn submit_certification(&self, generation: GenerationId) -> Result<(), String> {
        self.spawn(generation, move |shared| shared.certify(generation))
    }

    /// Cancel the certification `generation`: its candidates are withdrawn,
    /// the facts in force stay. `false` when it already had an outcome or no
    /// certification has the number.
    ///
    /// # Errors
    ///
    /// A storage failure, a commit failure other than a conflict, or moves
    /// kept winning for [`super::MOVE_PATIENCE`].
    pub fn cancel_certification(&self, generation: GenerationId) -> Result<bool, String> {
        let env = self.shared.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for () in move_attempts() {
            let Some((mut record, version)) = store.load_certification(generation).map_err(text)?
            else {
                return Ok(false);
            };
            if record.state.is_terminal() {
                return Ok(false);
            }
            record.state = BuildState::Cancelled;
            // Every candidate is withdrawn: none was put in force.
            record.refused = record
                .labels
                .iter()
                .map(|label| {
                    (
                        label.clone(),
                        format!("the certification of :{label} was cancelled"),
                    )
                })
                .collect();
            let committed = {
                let mut txn = begin(env);
                store
                    .put_certification_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                for label in &record.labels {
                    stage_withdrawal(env.engine(), &mut txn, label, generation).map_err(text)?;
                }
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(true),
                Err(e) if is_contention(&e) => continue,
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the certification {} kept moving under the cancellation",
            generation.as_raw()
        ))
    }

    /// The certification records without an outcome, submitted to executors
    /// of this process, as after a restart.
    pub(super) fn resume_certifications(&self) -> Result<Vec<GenerationId>, String> {
        let store = LocalIndexStore::new(self.shared.env.engine());
        let mut resumed = Vec::new();
        for record in store.list_certifications().map_err(text)? {
            if !record.state.is_terminal() {
                self.submit_certification(record.generation)?;
                resumed.push(record.generation);
            }
        }
        Ok(resumed)
    }
}

/// The waiter's view of the certification `generation` as its record has
/// it: `None` while it runs or when no certification has the number.
pub(super) fn certification_outcome(
    store: &LocalIndexStore<'_>,
    generation: GenerationId,
) -> Result<Option<Option<IndexBuildOutcome>>, StoreError> {
    Ok(store
        .load_certification(generation)?
        .map(|(record, _)| terminal_outcome(&record.state)))
}

impl Shared {
    /// Take the certification `generation`, prove the stored nodes and put
    /// each candidate in force or withdraw it.
    fn certify(self: &std::sync::Arc<Self>, generation: GenerationId) -> Executed {
        let _seat = self.seat(generation);
        if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
            return Executed::Lost;
        }
        self.try_certify(generation)
            .unwrap_or_else(|e| Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(e))))
    }

    fn try_certify(&self, generation: GenerationId) -> Result<Executed, String> {
        let (token, labels) = match self.take_certification(generation)? {
            Ok(taken) => taken,
            Err(executed) => return Ok(executed),
        };
        // No wait for older transactions: the publishing commit drained or
        // refused every writer still on its way under the revision before
        // (`CandidatePublished`), and a later one is held to the candidate.
        // Every node the proof must see is in the store already.
        self.set_phase(generation, BuildPhase::Indexing { indexed: None });
        let refused = self.prove(generation, &labels)?;
        self.conclude_certification(generation, token, &labels, refused)
    }

    /// Check the stored nodes of `labels` against their candidates as
    /// published with `generation`, in one pass. Returns the labels refused,
    /// each with why; a candidate no longer there (withdrawn, or the label
    /// dropped) refuses itself.
    fn prove(
        &self,
        generation: GenerationId,
        labels: &[String],
    ) -> Result<Vec<(String, String)>, String> {
        let engine = self.env.engine();
        let store = LocalSchemaStore::new(engine);
        let mut refused = Vec::new();
        let mut candidates = Vec::with_capacity(labels.len());
        for label in labels {
            let candidate = store
                .load_label(label)
                .map_err(text)?
                .filter(|s| s.candidate().is_some_and(|c| c.operation == generation))
                .and_then(|s| s.candidate_schema());
            match candidate {
                Some(candidate) => candidates.push(candidate),
                None => refused.push((
                    label.clone(),
                    format!("the candidate definition of :{label} is no longer published"),
                )),
            }
        }
        match coordinode_storage::engine::claims::evaluate::label_schema_violations(
            engine,
            &candidates,
        ) {
            Ok(violations) => refused.extend(violations.into_iter().map(|(label, v)| {
                let why = format!(
                    "the definition of :{label} is not put in force: node {} breaks it ({})",
                    v.node.to_element_id(),
                    v.reason
                );
                (label, why)
            })),
            Err(e) => refused.extend(
                candidates
                    .iter()
                    .map(|c| (c.name.clone(), format!("certify :{}: {e}", c.name))),
            ),
        }
        Ok(refused)
    }

    /// Take the certification `generation`: its record moves to running
    /// under a fresh token, conditioned on the record as read. `Err` with
    /// how the attempt ended when there is nothing to take.
    fn take_certification(
        &self,
        generation: GenerationId,
    ) -> Result<Result<(u64, Vec<String>), Executed>, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for () in move_attempts() {
            let Some((mut record, version)) = store.load_certification(generation).map_err(text)?
            else {
                return Ok(Err(Executed::Outcome(IndexBuildOutcome::Cancelled)));
            };
            if let Some(outcome) = terminal_outcome(&record.state) {
                return Ok(Err(Executed::Outcome(outcome)));
            }
            let token = next_token();
            record.state = BuildState::Running { executor: token };
            let committed = {
                let mut txn = begin(env);
                store
                    .put_certification_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn)
            };
            match committed {
                Ok(()) => return Ok(Ok((token, record.labels))),
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Ok(Err(Executed::Lost)),
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the certification {} kept moving while it was being taken",
            generation.as_raw()
        ))
    }

    /// The last move of the certification the executor holding `token`
    /// took: every candidate of `labels` not `refused` put in force, the
    /// refused ones withdrawn, and the record ended, in one commit
    /// conditioned on the record still naming the token and on each label's
    /// schema as read; retried while a conflicting move leaves the
    /// certification the executor's.
    fn conclude_certification(
        &self,
        generation: GenerationId,
        token: u64,
        labels: &[String],
        refused: Vec<(String, String)>,
    ) -> Result<Executed, String> {
        let env = self.env.as_ref();
        let store = LocalIndexStore::new(env.engine());
        for () in move_attempts() {
            let Some((mut record, version)) = store.load_certification(generation).map_err(text)?
            else {
                return Ok(Executed::Outcome(IndexBuildOutcome::Cancelled));
            };
            if record.state != (BuildState::Running { executor: token }) {
                return Ok(match terminal_outcome(&record.state) {
                    Some(outcome) => Executed::Outcome(outcome),
                    None => Executed::Lost,
                });
            }
            let committed = {
                let mut txn = begin(env);
                let mut withdrawn = refused.clone();
                for label in labels {
                    if withdrawn.iter().any(|(l, _)| l == label) {
                        stage_withdrawal(env.engine(), &mut txn, label, generation)
                            .map_err(text)?;
                    } else if !stage_activation(env.engine(), &mut txn, label, generation)
                        .map_err(text)?
                    {
                        // Withdrawn or replaced meanwhile: nothing of it is
                        // left to put in force.
                        withdrawn.push((
                            label.clone(),
                            format!("the candidate definition of :{label} is no longer published"),
                        ));
                    }
                }
                let outcome = if withdrawn.is_empty() {
                    record.state = BuildState::Published;
                    IndexBuildOutcome::Published { indexed: None }
                } else {
                    let reason = withdrawn
                        .iter()
                        .map(|(_, why)| why.as_str())
                        .collect::<Vec<_>>()
                        .join("; ");
                    record.state = BuildState::Failed {
                        reason: reason.clone(),
                    };
                    IndexBuildOutcome::Failed(BuildError::Other(reason))
                };
                record.refused = withdrawn;
                store
                    .put_certification_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                env.commit_catalog(&mut txn).map(|()| outcome)
            };
            match committed {
                Ok(outcome) => return Ok(Executed::Outcome(outcome)),
                Err(e) if is_contention(&e) => continue,
                Err(e) if cannot_move_here(&e) => return Ok(Executed::Lost),
                Err(e) => return Err(e.to_string()),
            }
        }
        Err(format!(
            "the certification {} kept moving while it was being concluded",
            generation.as_raw()
        ))
    }
}

/// Stage putting the candidate `generation` of `label` in force, at the
/// next revision, conditioned on the schema as read. `false` when the
/// label's schema carries no such candidate any more.
fn stage_activation(
    engine: &coordinode_storage::engine::core::StorageEngine,
    txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
    label: &str,
    generation: GenerationId,
) -> Result<bool, StoreError> {
    let store = LocalSchemaStore::new(engine);
    let Some(mut schema) = store.load_label_for_update_txn(txn, label)? else {
        return Ok(false);
    };
    if !schema
        .candidate()
        .is_some_and(|c| c.operation == generation)
    {
        return Ok(false);
    }
    schema.activate_candidate();
    // The presence and type constraints published with it were proved by
    // the same pass; a node key's index build activates it, not this.
    for name in validating_checks(&schema) {
        if let Some(constraint) = schema.constraint_mut(&name) {
            constraint.state = coordinode_core::schema::definition::ConstraintState::Active;
        }
    }
    schema.schema_revision = next_revision(&schema)?;
    // Every write since the candidate's publication was held to it, and
    // every stored node was proved against it: no scan at this commit.
    store.save_label_admitting_txn(txn, &schema)?;
    Ok(true)
}

/// Stage the withdrawal of the candidate `generation` of `label`, at the
/// next revision, leaving the facts in force as they are. Nothing when the
/// schema carries no such candidate any more.
fn stage_withdrawal(
    engine: &coordinode_storage::engine::core::StorageEngine,
    txn: &mut coordinode_storage::engine::transaction::Transaction<'_>,
    label: &str,
    generation: GenerationId,
) -> Result<(), StoreError> {
    let store = LocalSchemaStore::new(engine);
    let Some(mut schema) = store.load_label_for_update_txn(txn, label)? else {
        return Ok(());
    };
    if !schema
        .candidate()
        .is_some_and(|c| c.operation == generation)
    {
        return Ok(());
    }
    schema.take_candidate();
    // The presence and type constraints the proof was to put in force go
    // with it, their names released.
    for name in validating_checks(&schema) {
        schema.remove_constraint(&name);
        store.release_constraint_name_txn(txn, &name)?;
    }
    schema.schema_revision = next_revision(&schema)?;
    store.save_label_admitting_txn(txn, &schema)
}

/// The presence and type constraints of `schema` still validating: the
/// ones a certification proves, as opposed to those an index build does.
fn validating_checks(schema: &coordinode_core::schema::definition::LabelSchema) -> Vec<String> {
    schema
        .constraints()
        .iter()
        .filter(|c| {
            c.state == coordinode_core::schema::definition::ConstraintState::Validating
                && c.checks_each_node()
                && !c.owns_index()
        })
        .map(|c| c.name.clone())
        .collect()
}

/// The revision a change of `schema` is published at.
fn next_revision(
    schema: &coordinode_core::schema::definition::LabelSchema,
) -> Result<u64, StoreError> {
    schema.schema_revision.checked_add(1).ok_or_else(|| {
        StoreError::Invariant(format!(
            "label '{}' has no schema revision left",
            schema.name
        ))
    })
}
