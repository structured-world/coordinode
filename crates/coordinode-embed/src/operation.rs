//! Client operation identity for the host: number the writes, keep the
//! descriptor of each one, and find out after a lost reply, a restart or a
//! leader change whether it took effect, without running it a second time.
//!
//! An [`OperationIssuer`] is enrolled once per incarnation of a logical
//! client ([`Database::enroll_issuer`](crate::Database::enroll_issuer)) and
//! hands out an [`OperationDescriptor`] per write. A host that must survive
//! its own crash persists the descriptor before it sends the write and the
//! issuer's next number with it. A write carrying a descriptor
//! ([`StatementOptions::operation`](crate::db::StatementOptions::operation),
//! [`Database::commit_transaction_as`](crate::Database::commit_transaction_as))
//! takes effect at most once whatever is retried: a retry of a write that
//! already took effect answers with its outcome and runs nothing.
//! [`Database::resolve_operation`](crate::Database::resolve_operation) reports
//! how a write stands, and with [`Resolve::Exclude`] proves that one which
//! has not taken effect never will.

use std::sync::atomic::{AtomicU64, Ordering};

pub use coordinode_core::txn::operation::{
    DEFAULT_ADMISSION_WINDOW, IntentDigest, IssuerId, OperationDescriptor, OperationId,
    OperationOutcome,
};
use sha2::{Digest as _, Sha256};

use coordinode_core::graph::types::Value;
use coordinode_core::group::DeploymentId;

/// An enrolled incarnation of a logical client, numbering its writes.
///
/// Numbers are never handed out twice by one issuer value. Two copies of an
/// issuer (a clone, an allocator restored from an old backup) can hand out a
/// number twice: the database then lets the first write of that number take
/// effect and refuses the other if it means something else.
#[derive(Debug)]
pub struct OperationIssuer {
    issuer: IssuerId,
    incarnation: u64,
    next: AtomicU64,
    deployment: DeploymentId,
}

impl OperationIssuer {
    pub(crate) fn new(
        issuer: IssuerId,
        incarnation: u64,
        next: u64,
        deployment: DeploymentId,
    ) -> Self {
        Self {
            issuer,
            incarnation,
            next: AtomicU64::new(next),
            deployment,
        }
    }

    /// The logical client.
    pub fn issuer(&self) -> IssuerId {
        self.issuer
    }

    /// The incarnation this issuer numbers within.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// The deployment that enrolled the incarnation: the domain every
    /// descriptor it numbers names.
    pub fn deployment(&self) -> DeploymentId {
        self.deployment
    }

    /// The number the next descriptor gets: what a host persists to resume
    /// the issuer after a restart
    /// ([`Database::resume_issuer`](crate::Database::resume_issuer)).
    pub fn next_sequence(&self) -> u64 {
        self.next.load(Ordering::Acquire)
    }

    /// The descriptor of the next write.
    ///
    /// # Errors
    ///
    /// [`IssuerExhausted`] when the incarnation has handed out every number;
    /// a new incarnation is enrolled then.
    pub fn next(&self) -> Result<OperationDescriptor, IssuerExhausted> {
        let mut sequence = self.next.load(Ordering::Acquire);
        loop {
            let following = sequence.checked_add(1).ok_or(IssuerExhausted)?;
            match self.next.compare_exchange_weak(
                sequence,
                following,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(current) => sequence = current,
            }
        }
        Ok(OperationDescriptor {
            op: OperationId {
                issuer: self.issuer,
                incarnation: self.incarnation,
                sequence,
            },
            deployment: self.deployment,
        })
    }
}

/// An incarnation that handed out every operation number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the issuer's incarnation has handed out every operation number; enroll a new one")]
pub struct IssuerExhausted;

/// What [`Database::resolve_operation`](crate::Database::resolve_operation)
/// does when nothing is recorded of the operation yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resolve {
    /// Report what is recorded: an operation with nothing recorded is
    /// [`OperationOutcome::Unknown`], since an attempt may still be on its
    /// way. What is observed, not a durable proof: a commit made under a
    /// volatile write concern is reported before the journal syncs it.
    Lookup,
    /// Exclude it, through the log, so no attempt ever takes effect, and
    /// report [`OperationOutcome::Aborted`]; one that committed first reports
    /// its outcome. Either answer is durable when it is given: the proof a
    /// host needs before it gives the operation up or retries the work as a
    /// new one.
    Exclude,
}

/// Domain separation of a statement's intent digest.
const STATEMENT_INTENT_TAG: &[u8] = b"coordinode/operation/statement/v1";

/// Domain separation of a transaction's statement chain.
const CHAIN_INTENT_TAG: &[u8] = b"coordinode/operation/chain/v1";

/// What a statement means: its dialect, text and parameters. Two attempts
/// of one operation run the same statement with the same parameters, and
/// nothing else enters the digest, so they carry the same intent.
pub(crate) fn statement_intent(
    query: &str,
    params: Option<&std::collections::HashMap<String, Value>>,
) -> IntentDigest {
    let mut hasher = Sha256::new();
    hasher.update(STATEMENT_INTENT_TAG);
    hasher.update(b"cypher");
    hasher.update((query.len() as u64).to_be_bytes());
    hasher.update(query.as_bytes());
    let mut sorted: Vec<(&String, &Value)> = params.into_iter().flatten().collect();
    sorted.sort_by(|a, b| a.0.cmp(b.0));
    hasher.update((sorted.len() as u64).to_be_bytes());
    for (name, value) in sorted {
        hasher.update((name.len() as u64).to_be_bytes());
        hasher.update(name.as_bytes());
        // A value that cannot be encoded is refused before it runs, so the
        // fallback only keeps the digest defined.
        let encoded = rmp_serde::to_vec(value).unwrap_or_default();
        hasher.update((encoded.len() as u64).to_be_bytes());
        hasher.update(&encoded);
    }
    IntentDigest(hasher.finalize().into())
}

/// The intent of a transaction's statements so far, `chain`, extended by
/// one more statement.
pub(crate) fn chain_intent(chain: &IntentDigest, statement: &IntentDigest) -> IntentDigest {
    let mut hasher = Sha256::new();
    hasher.update(CHAIN_INTENT_TAG);
    hasher.update(chain.0);
    hasher.update(statement.0);
    IntentDigest(hasher.finalize().into())
}

/// The intent of a transaction that ran no statement yet.
pub(crate) const EMPTY_CHAIN: IntentDigest =
    IntentDigest([0; coordinode_core::txn::operation::INTENT_DIGEST_LEN]);

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
