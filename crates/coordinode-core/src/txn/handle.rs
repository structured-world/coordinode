//! The handle of an open interactive transaction, and the caller a command
//! acts for.
//!
//! An open transaction belongs to the execution leader that began it, not to
//! the connection or member the client reached it through, so its handle
//! names that owner: the database it belongs to, the member that holds it,
//! the leadership tenure it was opened in, and its number there. A handle of
//! another database, an earlier tenure or a restarted process names no live
//! transaction, whatever its number.

use alloc::sync::Arc;
use core::fmt;

use crate::group::DeploymentId;

/// Length of [`TransactionHandle::to_bytes`].
pub const TRANSACTION_HANDLE_LEN: usize = 1 + 16 + 8 + 8 + 8;

/// Version of the handle's byte form.
const HANDLE_FORMAT: u8 = 1;

/// An open interactive transaction, as a client addresses it through any
/// member. Opaque to the client: only its owner interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransactionHandle {
    /// The database's deployment when the transaction began; zero before
    /// the database has recorded one.
    scope: [u8; 16],
    /// The member that holds the transaction; 0 for a database outside a
    /// group.
    owner: u64,
    /// The owner's tenure: its leadership term in a group, a number drawn at
    /// open outside one.
    incarnation: u64,
    /// The transaction's number within the tenure.
    local: u64,
}

impl TransactionHandle {
    /// The handle of transaction `local` of `owner`'s tenure `incarnation`
    /// in the database of deployment `scope`.
    pub fn new(scope: Option<DeploymentId>, owner: u64, incarnation: u64, local: u64) -> Self {
        Self {
            scope: scope.map_or([0; 16], |id| id.0),
            owner,
            incarnation,
            local,
        }
    }

    /// The member that holds the transaction.
    pub fn owner(&self) -> u64 {
        self.owner
    }

    /// The tenure the transaction was opened in.
    pub fn incarnation(&self) -> u64 {
        self.incarnation
    }

    /// The transaction's number within its tenure.
    pub fn local(&self) -> u64 {
        self.local
    }

    /// The byte form a client carries.
    pub fn to_bytes(&self) -> [u8; TRANSACTION_HANDLE_LEN] {
        let mut bytes = [0u8; TRANSACTION_HANDLE_LEN];
        bytes[0] = HANDLE_FORMAT;
        bytes[1..17].copy_from_slice(&self.scope);
        bytes[17..25].copy_from_slice(&self.owner.to_be_bytes());
        bytes[25..33].copy_from_slice(&self.incarnation.to_be_bytes());
        bytes[33..41].copy_from_slice(&self.local.to_be_bytes());
        bytes
    }

    /// The handle `bytes` encode.
    ///
    /// # Errors
    ///
    /// [`MalformedHandle`] when `bytes` is not a handle of this format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MalformedHandle> {
        let bytes: &[u8; TRANSACTION_HANDLE_LEN] = bytes.try_into().map_err(|_| MalformedHandle)?;
        if bytes[0] != HANDLE_FORMAT {
            return Err(MalformedHandle);
        }
        let word = |at: usize| {
            let mut word = [0u8; 8];
            word.copy_from_slice(&bytes[at..at + 8]);
            u64::from_be_bytes(word)
        };
        let mut scope = [0u8; 16];
        scope.copy_from_slice(&bytes[1..17]);
        Ok(Self {
            scope,
            owner: word(17),
            incarnation: word(25),
            local: word(33),
        })
    }
}

impl fmt::Display for TransactionHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.to_bytes() {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Bytes that are not a [`TransactionHandle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MalformedHandle;

impl fmt::Display for MalformedHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("not a transaction handle")
    }
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
