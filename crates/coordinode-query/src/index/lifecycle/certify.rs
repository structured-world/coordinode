//! Certification of label candidates, on the same executors, seats, records
//! and moves as index builds.
//!
//! A candidate definition is published in the label's schema beside the
//! facts in force, with its certification admitted in the same commit.
//! Writers from then on are held to both (`CandidateAdmits`). The executor
//! waits for every transaction open before it to end, so each node written
//! under the old facts alone is in the store, then checks the stored nodes
//! against the candidate. Its last move puts the candidate in force, or
//! withdraws it leaving the facts in force untouched, in one commit
//! conditioned on the record still being its own and on the label's schema
//! as it read it.

use coordinode_modality::{
    BuildState, IndexStore as _, LocalIndexStore, LocalSchemaStore, SchemaStore as _, StoreError,
};

use super::{
    BuildError, BuildPhase, Executed, GenerationId, IndexBuildOutcome, IndexBuildService, Shared,
    begin, cannot_move_here, is_contention, move_attempts, next_token, terminal_outcome, text,
};
use crate::index::build::{BackfillError, await_older_transactions};

/// How a certification ended, before its last move lands.
enum Proof {
    /// Every stored node satisfies the candidate; this many were checked.
    Holds(u64),
    /// The candidate is refused, for this reason.
    Refused(String),
}

impl IndexBuildService {
    /// Run the certification `generation` on an executor of this process,
    /// unless one already has it. [`Self::wait`] reports its outcome: the
    /// candidate put in force is `Published`, a refused one `Failed`.
    ///
    /// # Errors
    ///
    /// The executor thread could not be started.
    pub fn submit_certification(&self, generation: GenerationId) -> Result<(), String> {
        self.spawn(generation, move |shared| shared.certify(generation))
    }

    /// Cancel the certification `generation`: its candidate is withdrawn,
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
            let committed = {
                let mut txn = begin(env);
                store
                    .put_certification_txn(&mut txn, &record, Some(version))
                    .map_err(text)?;
                stage_withdrawal(env.engine(), &mut txn, &record.label, generation)
                    .map_err(text)?;
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
    /// the candidate in force or withdraw it.
    fn certify(self: &std::sync::Arc<Self>, generation: GenerationId) -> Executed {
        let _seat = self.seat(generation);
        if self.stopping.load(core::sync::atomic::Ordering::Acquire) {
            return Executed::Lost;
        }
        self.try_certify(generation)
            .unwrap_or_else(|e| Executed::Outcome(IndexBuildOutcome::Failed(BuildError::Other(e))))
    }

    fn try_certify(&self, generation: GenerationId) -> Result<Executed, String> {
        let (token, label) = match self.take_certification(generation)? {
            Ok(taken) => taken,
            Err(executed) => return Ok(executed),
        };
        let env = self.env.as_ref();
        self.set_phase(generation, BuildPhase::AwaitingOlderTransactions);
        let wait = self.config.read().older_transactions_wait;
        match await_older_transactions(env.engine(), wait, Some(&self.stopping)) {
            Ok(()) => {}
            // The record stays running under this token, and the next
            // process to open the storage takes it over.
            Err(BackfillError::Stopped) => return Ok(Executed::Lost),
            Err(e) => {
                return self.conclude_certification(
                    generation,
                    token,
                    &label,
                    Proof::Refused(format!("certify :{label}: {e}")),
                );
            }
        }
        self.set_phase(generation, BuildPhase::Indexing { indexed: None });
        let proof = self.prove(generation, &label)?;
        self.conclude_certification(generation, token, &label, proof)
    }

    /// Check every stored node of `label` against its candidate as published
    /// with `generation`. A candidate no longer there (withdrawn, or the
    /// label dropped) refuses itself.
    fn prove(&self, generation: GenerationId, label: &str) -> Result<Proof, String> {
        let engine = self.env.engine();
        let schema = LocalSchemaStore::new(engine)
            .load_label(label)
            .map_err(text)?;
        let candidate = schema
            .as_ref()
            .filter(|s| s.candidate().is_some_and(|c| c.operation == generation))
            .and_then(|s| s.candidate_schema());
        let Some(candidate) = candidate else {
            return Ok(Proof::Refused(format!(
                "the candidate definition of :{label} is no longer published"
            )));
        };
        let none = std::collections::HashMap::new();
        match coordinode_storage::engine::claims::evaluate::first_label_schema_violation(
            engine, &candidate, &none,
        ) {
            Ok(None) => Ok(Proof::Holds(0)),
            Ok(Some(violation)) => Ok(Proof::Refused(format!(
                "the definition of :{label} is not put in force: node {} breaks it ({})",
                violation.node.to_element_id(),
                violation.reason
            ))),
            Err(e) => Ok(Proof::Refused(format!("certify :{label}: {e}"))),
        }
    }

    /// Take the certification `generation`: its record moves to running
    /// under a fresh token, conditioned on the record as read. `Err` with
    /// how the attempt ended when there is nothing to take.
    fn take_certification(
        &self,
        generation: GenerationId,
    ) -> Result<Result<(u64, String), Executed>, String> {
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
                Ok(()) => return Ok(Ok((token, record.label))),
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
    /// took: the candidate put in force or withdrawn, and the record ended,
    /// in one commit conditioned on the record still naming the token and on
    /// the label's schema as read; retried while a conflicting move leaves
    /// the certification the executor's.
    fn conclude_certification(
        &self,
        generation: GenerationId,
        token: u64,
        label: &str,
        proof: Proof,
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
                let outcome = match &proof {
                    Proof::Holds(checked) => {
                        match stage_activation(env.engine(), &mut txn, label, generation)
                            .map_err(text)?
                        {
                            true => {
                                record.checked = *checked;
                                record.state = BuildState::Published;
                                IndexBuildOutcome::Published {
                                    indexed: Some(*checked),
                                }
                            }
                            // Withdrawn or replaced meanwhile: nothing of it
                            // is left to put in force.
                            false => {
                                record.state = BuildState::Cancelled;
                                IndexBuildOutcome::Cancelled
                            }
                        }
                    }
                    Proof::Refused(reason) => {
                        stage_withdrawal(env.engine(), &mut txn, label, generation)
                            .map_err(text)?;
                        record.state = BuildState::Failed {
                            reason: reason.clone(),
                        };
                        IndexBuildOutcome::Failed(BuildError::Other(reason.clone()))
                    }
                };
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
    schema.schema_revision = next_revision(&schema)?;
    store.save_label_admitting_txn(txn, &schema)
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
