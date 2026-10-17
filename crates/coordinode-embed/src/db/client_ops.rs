//! The host's surface of client operation identity: enrollment, lookup,
//! exclusion and retirement. See [`crate::operation`].

use coordinode_core::group::DeploymentId;
use coordinode_core::txn::operation::{
    DEFAULT_ADMISSION_WINDOW, IntentDigest, IssuerId, OperationCommand, OperationDescriptor,
    OperationOutcome,
};
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, ProposalError, RaftProposal};
use coordinode_core::txn::timestamp::Timestamp;
use coordinode_storage::engine::operations as records;

use super::{Database, DatabaseError};
use crate::operation::{OperationIssuer, Resolve};

/// Enrollments lost to another enrollment of the same issuer before one is
/// given up: each loss means another process enrolled it meanwhile.
const MAX_ENROLL_ATTEMPTS: u32 = 8;

impl Database {
    /// Enroll a new incarnation of `issuer`, closing every earlier one: a
    /// closed incarnation admits no new operation, and what it admitted
    /// stays resolvable. The returned issuer numbers this incarnation's
    /// writes from its first number.
    ///
    /// A host enrolls when it starts a logical client for the first time or
    /// cannot restore its allocation; a host that persisted it resumes with
    /// [`Self::resume_issuer`] instead, so its in-flight operations stay
    /// admissible.
    ///
    /// # Errors
    ///
    /// The enrollment could not be committed (not the leader, no quorum, a
    /// storage failure), or other enrollments of the same issuer won every
    /// attempt.
    pub fn enroll_issuer(&self, issuer: IssuerId) -> Result<OperationIssuer, DatabaseError> {
        let deployment = self.found_deployment()?;
        for _ in 0..MAX_ENROLL_ATTEMPTS {
            let latest = records::incarnations(&self.engine, &issuer)?
                .last()
                .map_or(0, |(number, _)| *number);
            let incarnation = latest.checked_add(1).ok_or_else(|| {
                DatabaseError::Other("the issuer has used every incarnation".into())
            })?;
            self.propose_operation(OperationCommand::Enroll {
                issuer,
                incarnation,
                window: DEFAULT_ADMISSION_WINDOW,
            })?;
            if let Some(record) = records::incarnation_record(&self.engine, &issuer, incarnation)? {
                if record.open {
                    return Ok(OperationIssuer::new(
                        issuer,
                        incarnation,
                        record.retired_below,
                        deployment,
                    ));
                }
            }
        }
        Err(DatabaseError::Other(format!(
            "other enrollments of the issuer won {MAX_ENROLL_ATTEMPTS} attempts"
        )))
    }

    /// Resume `issuer`'s `incarnation` after a restart, numbering from
    /// `next_sequence` (what [`OperationIssuer::next_sequence`] returned
    /// before), or from the incarnation's retired prefix when that is
    /// higher.
    ///
    /// # Errors
    ///
    /// The incarnation was never enrolled here, or a later one closed it:
    /// it admits nothing new, and the host enrolls a new one.
    pub fn resume_issuer(
        &self,
        issuer: IssuerId,
        incarnation: u64,
        next_sequence: u64,
    ) -> Result<OperationIssuer, DatabaseError> {
        // A database that founded no domain issued no incarnation.
        let Some(deployment) = self.deployment()? else {
            return Err(DatabaseError::Other(format!(
                "incarnation {incarnation} of the issuer was never enrolled here"
            )));
        };
        match records::incarnation_record(&self.engine, &issuer, incarnation)? {
            Some(record) if record.open => Ok(OperationIssuer::new(
                issuer,
                incarnation,
                next_sequence.max(record.retired_below),
                deployment,
            )),
            Some(_) => Err(DatabaseError::Other(format!(
                "incarnation {incarnation} of the issuer is closed; enroll a new one"
            ))),
            None => Err(DatabaseError::Other(format!(
                "incarnation {incarnation} of the issuer was never enrolled here"
            ))),
        }
    }

    /// How the operation `descriptor` names stands. With
    /// [`Resolve::Exclude`], an operation not committed (nothing recorded, or
    /// a provisional write admitted and not yet drained) is excluded through
    /// the log first, so the answer is final: committed, or aborted and never
    /// taking effect.
    ///
    /// A lookup reads what this member applied; [`Resolve::Exclude`] is
    /// decided by the group's log, and refused on a member that is not the
    /// leader.
    ///
    /// # Errors
    ///
    /// The descriptor names another deployment, or the exclusion could not
    /// be committed.
    pub fn resolve_operation(
        &self,
        descriptor: &OperationDescriptor,
        resolve: Resolve,
    ) -> Result<OperationOutcome, DatabaseError> {
        self.check_deployment(descriptor)?;
        let op = descriptor.op;
        let outcome = records::operation_outcome(&self.engine, &op)?;
        if resolve == Resolve::Exclude
            && matches!(
                outcome,
                OperationOutcome::Unknown | OperationOutcome::Admitted { .. }
            )
        {
            self.propose_operation(OperationCommand::Exclude(op))?;
            return Ok(records::operation_outcome(&self.engine, &op)?);
        }
        Ok(outcome)
    }

    /// Release the records of the operations of `below`'s incarnation
    /// numbered below it, once the host no longer needs their outcomes: a
    /// later lookup of one reports it retired, and no attempt of one takes
    /// effect, numbers never used included. Retirement stops in front of an
    /// operation admitted and not complete.
    ///
    /// # Errors
    ///
    /// The descriptor names another deployment, or the retirement could not
    /// be committed.
    pub fn retire_operations(&self, below: &OperationDescriptor) -> Result<(), DatabaseError> {
        self.check_deployment(below)?;
        self.propose_operation(OperationCommand::Retire {
            issuer: below.op.issuer,
            incarnation: below.op.incarnation,
            below: below.op.sequence,
        })
    }

    /// Release the outcome of the operation `descriptor` names, once the host
    /// no longer needs it, without waiting for the numbers below it: a later
    /// lookup reports it retired, and no attempt of it takes effect. Only a
    /// settled operation (committed or aborted) is released; one admitted
    /// and not complete, or one nothing is recorded of, stays as it is.
    ///
    /// # Errors
    ///
    /// The descriptor names another deployment, or the release could not be
    /// committed.
    pub fn acknowledge_operation(
        &self,
        descriptor: &OperationDescriptor,
    ) -> Result<(), DatabaseError> {
        self.check_deployment(descriptor)?;
        self.propose_operation(OperationCommand::Release(descriptor.op))
    }

    /// Refuse a write carrying out `descriptor` that should not run: an
    /// attempt of it took effect already, it was excluded or retired, or it
    /// cannot be admitted now. The binding decides again where the write
    /// applies; this answers a retry without running it.
    pub(crate) fn admit_operation(
        &self,
        descriptor: &OperationDescriptor,
        intent: &IntentDigest,
    ) -> Result<(), DatabaseError> {
        self.check_deployment(descriptor)?;
        let op = descriptor.op;
        let incarnation = records::incarnation_record(&self.engine, &op.issuer, op.incarnation)?;
        let record = records::operation_record(&self.engine, &op)?;
        let outcome =
            coordinode_core::txn::operation::outcome_of(&op, incarnation.as_ref(), record.as_ref());
        let admissible = incarnation.is_some_and(|inc| {
            inc.open
                && op.sequence >= inc.retired_below
                && op.sequence - inc.retired_below < inc.window
        });
        if outcome == OperationOutcome::Unknown && admissible {
            return Ok(());
        }
        Err(DatabaseError::OperationSettled {
            op,
            outcome,
            intent_differs: record.is_some_and(|r| r.intent_differs(intent)),
        })
    }

    /// The deployment this database's operation records belong to, when it
    /// has one.
    fn deployment(&self) -> Result<Option<coordinode_core::group::DeploymentId>, DatabaseError> {
        Ok(coordinode_storage::engine::metadata::bound_deployment(
            &self.engine,
        )?)
    }

    /// Give the database its deployment identity before it issues its first
    /// descriptor, unless it has one: descriptors then name the domain they
    /// were issued in, and another database refuses them. A group formed
    /// around this store later keeps the identity, so its descriptors stay
    /// valid.
    fn found_deployment(&self) -> Result<DeploymentId, DatabaseError> {
        if let Some(id) = self.deployment()? {
            return Ok(id);
        }
        // The first recorded identity stands: a concurrent founding loses,
        // and both read back the one recorded.
        self.propose_mutation(Mutation::Command(MetadataCommand::RecordDeployment {
            id: fresh_deployment_id(),
        }))?;
        self.deployment()?
            .ok_or_else(|| DatabaseError::Other("the deployment identity was not recorded".into()))
    }

    /// Refuse a descriptor that does not name this database's domain. A
    /// database that founded none issued no descriptor, so it refuses every
    /// one.
    fn check_deployment(&self, descriptor: &OperationDescriptor) -> Result<(), DatabaseError> {
        if self.deployment()? == Some(descriptor.deployment) {
            return Ok(());
        }
        Err(DatabaseError::ForeignOperation {
            descriptor: descriptor.deployment,
        })
    }

    /// Commit `command` through the write pipeline on its own.
    fn propose_operation(&self, command: OperationCommand) -> Result<(), DatabaseError> {
        self.propose_mutation(Mutation::Operation(command))
    }

    /// Commit the command `mutation` through the write pipeline on its own.
    fn propose_mutation(&self, mutation: Mutation) -> Result<(), DatabaseError> {
        // Held as not yet logged until the entry is handed over, so no
        // closed bound passes over it meanwhile.
        let (commit_ts, _held) = self
            .engine
            .pending_commits()
            .obligate(|| self.oracle.next().as_raw());
        let proposal = RaftProposal {
            id: self.proposal_id_gen.next(),
            mutations: vec![mutation],
            commit_ts: Timestamp::from_raw(commit_ts),
            start_ts: Timestamp::from_raw(0),
            bypass_rate_limiter: true,
        };
        self.pipeline
            .propose_and_wait(&proposal)
            .map(drop)
            .map_err(|e| match e {
                ProposalError::NotLeader { leader_id } => DatabaseError::NotLeader { leader_id },
                ProposalError::Mismatched(m) => DatabaseError::Mismatched(m),
                other => DatabaseError::Other(format!("operation record: {other}")),
            })
    }
}

/// A deployment identity no other database draws: 128 bits from two hashers
/// keyed by the process's random seed, mixed with the clock. It tells domains
/// apart and grants nothing, so it needs uniqueness, not secrecy.
fn fresh_deployment_id() -> DeploymentId {
    use std::hash::{BuildHasher, Hasher};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut id = [0u8; 16];
    for half in id.as_chunks_mut::<8>().0 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u128(now);
        *half = hasher.finish().to_be_bytes();
    }
    DeploymentId(id)
}
