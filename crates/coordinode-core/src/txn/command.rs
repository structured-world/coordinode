//! What a command for the database carries besides its text: the ID of the
//! interactive transaction it belongs to, and the caller it acts for.
//!
//! A transaction ID is a random UUID the client draws, as a MongoDB driver
//! draws its session ID: unique across clients, members, restarts and
//! terms, and not guessable from another one. It names no member or term.
//! The leader alone holds open transactions and any member passes their
//! commands on to it; when the leader changes the transaction ends, and the
//! new leader, which never held it, answers that it knows no such
//! transaction.

use alloc::sync::Arc;
use core::fmt;

/// Length of a [`TransactionId`] on the wire: one UUID.
pub const TRANSACTION_ID_LEN: usize = 16;

/// An interactive transaction, as a client names it through any member: the
/// random UUID it drew.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransactionId([u8; TRANSACTION_ID_LEN]);

impl TransactionId {
    /// The transaction whose UUID is `bytes`.
    pub const fn from_raw(bytes: [u8; TRANSACTION_ID_LEN]) -> Self {
        Self(bytes)
    }

    /// The UUID's bytes, as the wire carries them.
    pub const fn to_bytes(&self) -> [u8; TRANSACTION_ID_LEN] {
        self.0
    }

    /// The ID `bytes` carry.
    ///
    /// # Errors
    ///
    /// [`MalformedTransactionId`] when `bytes` is not one UUID long.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MalformedTransactionId> {
        bytes
            .try_into()
            .map(Self)
            .map_err(|_| MalformedTransactionId)
    }
}

/// The UUID in its usual hyphenated form, as logs and clients show it.
impl fmt::Display for TransactionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (at, byte) in self.0.iter().enumerate() {
            if matches!(at, 4 | 6 | 8 | 10) {
                f.write_str("-")?;
            }
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Bytes that are not a [`TransactionId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalformedTransactionId;

impl fmt::Display for MalformedTransactionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a transaction ID is a UUID of 16 bytes")
    }
}

/// What ends an interactive transaction that stays idle past the idle
/// timeout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimeoutOutcome {
    /// It is rolled back: nothing of it is applied.
    #[default]
    Rollback,
    /// It is committed as an explicit commit would commit it, under the
    /// client operation its start named, by which the client learns the
    /// outcome. A transaction one of whose statements failed is rolled back
    /// instead.
    Commit,
}

impl TimeoutOutcome {
    /// Lowercase label for result rows and configuration.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rollback => "rollback",
            Self::Commit => "commit",
        }
    }
}

/// A node and the version a caller believes it is at, which a commit
/// checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpectedVersion {
    /// The node.
    pub node: crate::graph::node::NodeId,
    /// The version read earlier; `None` requires the node not to exist.
    pub version: Option<u64>,
}

/// Told when a command has its place in its transaction's order: it runs
/// now, or waits in the owner's queue behind the commands admitted before it.
/// An ingress that keeps one stream's commands for a transaction in order
/// sends the next one once this is told, rather than once the command has
/// answered. It may be told more than once, and is not told for a command
/// refused before it got a place.
#[derive(Clone)]
pub struct AdmissionSignal(Arc<dyn Fn() + Send + Sync>);

impl AdmissionSignal {
    /// A signal that runs `admitted` each time it is told.
    pub fn new(admitted: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(admitted))
    }

    /// Tell the command's sender it has its place.
    pub fn admitted(&self) {
        (self.0)();
    }
}

impl fmt::Debug for AdmissionSignal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("AdmissionSignal")
    }
}

/// Who a command acts for, as the member that admitted it established.
///
/// Neither a public request field nor a claimed user identifier can produce
/// it: an authentication adapter or a trusted host does. Explicitly trusted
/// access is its own value, never what a missing identity falls back to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ActingContext {
    /// Every caller shares one trust boundary: a single-trust embedding host
    /// or a deployment configured without principals. No caller isolation.
    Trusted,
    /// A verified principal.
    Principal(Principal),
}

/// A verified subject and the scope it acts in.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Principal {
    /// The stable subject, as its issuer names it.
    pub subject: Arc<str>,
    /// The issuer or trust scope the subject belongs to.
    pub trust_scope: Arc<str>,
    /// The namespace the caller is bound to.
    pub namespace: Arc<str>,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
