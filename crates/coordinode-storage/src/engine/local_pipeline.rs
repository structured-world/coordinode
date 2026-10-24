//! Proposal pipelines for a store that is not a member of a replicated group.
//!
//! - [`LocalProposalPipeline`] applies a proposal to the engine, the whole
//!   proposal at one seqno equal to its `commit_ts`.
//! - [`OwnedLocalProposalPipeline`] does the same over an `Arc`, journalling
//!   first when the engine keeps a journal.
//!
//! They live beside the engine rather than beside consensus so that a build
//! without replication carries no consensus code at all.

use std::sync::Arc;

use coordinode_core::txn::proposal::{
    ProposalError, ProposalOutcome, ProposalPipeline, RaftProposal,
};
use coordinode_core::txn::write_concern::{Journal, WriteConcern};

use crate::engine::config::FlushPolicy;
use crate::engine::core::StorageEngine;
use crate::error::StorageError;

/// Map a [`StorageError`] from the engine into a [`ProposalError`],
/// preserving the variants a caller reports with structured detail: a full
/// endpoint and a disk below its free-space reserve carry the operator's
/// numbers. Other storage errors become [`ProposalError::Storage`].
pub fn storage_to_proposal_err(e: StorageError) -> ProposalError {
    match e {
        StorageError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        } => ProposalError::CapacityExhausted {
            endpoint_id,
            used_bytes,
            hard_limit_bytes,
        },
        StorageError::OutOfSpace {
            path,
            available_bytes,
            min_free_bytes,
        } => ProposalError::OutOfSpace {
            path,
            available_bytes,
            min_free_bytes,
        },
        other => ProposalError::Storage(other.to_string()),
    }
}

/// Pipeline that applies a proposal straight to the engine it borrows.
///
/// Used where a proposal is applied within the caller's thread with nothing
/// to replicate it to. No journal is written: a caller that needs crash
/// safety uses [`OwnedLocalProposalPipeline`] over a journalled engine.
pub struct LocalProposalPipeline<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalProposalPipeline<'a> {
    /// Create a pipeline that applies directly to the given engine.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

impl ProposalPipeline for LocalProposalPipeline<'_> {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        let _admitted = self
            .engine
            .admit_write(&proposal.mutations)
            .map_err(storage_to_proposal_err)?;
        // The whole proposal lands at one seqno, its commit_ts, so a
        // snapshot at commit_ts sees all of it and one tick earlier none.
        self.engine
            .apply_proposal_at(&proposal.mutations, proposal.commit_ts.as_raw())
            .map_err(storage_to_proposal_err)?;

        tracing::debug!(
            proposal_id = %proposal.id,
            mutations = proposal.mutation_count(),
            commit_ts = proposal.commit_ts.as_raw(),
            "local proposal applied"
        );

        // No log, so no cluster-wide commit index.
        Ok(ProposalOutcome::local())
    }
}

/// Owned form of [`LocalProposalPipeline`] over an `Arc<StorageEngine>`, for
/// the contexts that need a `'static` pipeline (the database handle).
///
/// ## Durability
///
/// An engine opened with a journal (`StorageEngine::open_embedded`) appends
/// each proposal to its retained journal, applies it at `commit_ts` and marks
/// the partitions it touched, in one engine call; recovery replays the
/// journal. The journal entry is synced before the call returns unless the
/// write concern's `j` is volatile. An engine without one applies and then,
/// under [`FlushPolicy::SyncPerBatch`] and `j: journal`, persists the
/// memtables before returning.
pub struct OwnedLocalProposalPipeline {
    engine: Arc<StorageEngine>,
}

impl ProposalPipeline for OwnedLocalProposalPipeline {
    fn propose_and_wait(&self, proposal: &RaftProposal) -> Result<ProposalOutcome, ProposalError> {
        self.apply(proposal, Journal::Journal)
    }

    /// One member, so every `w` is the local apply; `j` decides how far the
    /// journal entry gets before this returns.
    fn propose_with_concern(
        &self,
        proposal: &RaftProposal,
        concern: &WriteConcern,
    ) -> Result<ProposalOutcome, ProposalError> {
        self.apply(proposal, concern.journal)
    }
}

impl OwnedLocalProposalPipeline {
    /// Create a pipeline that applies directly to the given engine.
    pub fn new(engine: &Arc<StorageEngine>) -> Self {
        Self {
            engine: Arc::clone(engine),
        }
    }

    /// Apply `proposal`, its journal entry taken as far as `level`.
    fn apply(
        &self,
        proposal: &RaftProposal,
        level: Journal,
    ) -> Result<ProposalOutcome, ProposalError> {
        if self.engine.has_journal() {
            // `commit_logged` checks free space for itself.
            self.engine
                .commit_logged(&proposal.mutations, proposal.commit_ts.as_raw(), level)
                .map_err(storage_to_proposal_err)?;
        } else {
            // Refused before the apply and the flush that would hit a full
            // disk, or pass a footprint limit.
            let _admitted = self
                .engine
                .admit_write(&proposal.mutations)
                .map_err(storage_to_proposal_err)?;
            // The whole proposal lands at one seqno, its commit_ts.
            self.engine
                .apply_proposal_at(&proposal.mutations, proposal.commit_ts.as_raw())
                .map_err(storage_to_proposal_err)?;
            // Without a journal a flush after every proposal is what makes the
            // write survive a crash: expensive, but correct. A volatile level
            // asked not to wait for it.
            if level == Journal::Journal && self.engine.flush_policy() == FlushPolicy::SyncPerBatch
            {
                self.engine
                    .persist()
                    .map_err(|e| ProposalError::Storage(format!("persist failed: {e}")))?;
            }
        }

        tracing::debug!(
            proposal_id = %proposal.id,
            mutations = proposal.mutation_count(),
            commit_ts = proposal.commit_ts.as_raw(),
            has_journal = self.engine.has_journal(),
            "owned local proposal applied"
        );

        Ok(ProposalOutcome::local())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
