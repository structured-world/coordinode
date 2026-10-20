//! Live registry of client sessions and the transactions they are attached to.
//!
//! Node-local source of truth for the session side of operational
//! introspection (`SHOW SESSIONS` / `SHOW TRANSACTIONS`): every open session,
//! the requests in flight on it, and the interactive transactions it has sent
//! commands for. A transaction is not the session's: it lives on its owner,
//! which reports it, and a session closing only drops its attachments. The
//! session core updates the registry as sessions open/close, requests
//! start/finish and transactions are begun, used and ended through it; the
//! query executor reads a snapshot through [`OperationsView`].

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant; // no-std: caller-provided Clock trait

use coordinode_core::operations::{OperationsView, SessionSnapshot};
use coordinode_core::txn::command::TransactionId;
use parking_lot::RwLock;

/// Node-local registry of live sessions and their transaction attachments.
///
/// Cheap to clone behind an `Arc`: the session binding holds one, and so does
/// the query layer (as a `dyn OperationsView`), both pointing at the same map.
#[derive(Default)]
pub struct SessionRegistry {
    sessions: RwLock<HashMap<u64, SessionEntry>>,
    next_session_id: AtomicU64,
}

struct SessionEntry {
    peer: String,
    opened_at: Instant,
    in_flight: u64,
    attached: HashSet<TransactionId>,
}

impl SessionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new session for `peer` and return its node-local id.
    pub fn register_session(&self, peer: String) -> u64 {
        let id = self.next_session_id.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.sessions.write().insert(
            id,
            SessionEntry {
                peer,
                opened_at: Instant::now(),
                in_flight: 0,
                attached: HashSet::new(),
            },
        );
        id
    }

    /// Remove a session (stream closed). Its transactions stay open on their
    /// owner.
    pub fn close_session(&self, session_id: u64) {
        self.sessions.write().remove(&session_id);
    }

    /// Mark one more request as executing on the session.
    pub fn request_started(&self, session_id: u64) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            e.in_flight += 1;
        }
    }

    /// Mark one request as finished on the session.
    pub fn request_finished(&self, session_id: u64) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            // Balanced with `request_started`; guard the floor defensively
            // rather than wrap, so a stray finish can never underflow.
            if e.in_flight > 0 {
                e.in_flight -= 1;
            }
        }
    }

    /// Record that the session sends commands for `transaction`.
    pub fn attach(&self, session_id: u64, transaction: TransactionId) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            e.attached.insert(transaction);
        }
    }

    /// Record that `transaction` ended through the session.
    pub fn detach(&self, session_id: u64, transaction: &TransactionId) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            e.attached.remove(transaction);
        }
    }
}

impl OperationsView for SessionRegistry {
    fn sessions(&self) -> Vec<SessionSnapshot> {
        let now = Instant::now();
        self.sessions
            .read()
            .iter()
            .map(|(id, e)| SessionSnapshot {
                session_id: format!("s-{id}"),
                peer: e.peer.clone(),
                age_ms: now.duration_since(e.opened_at).as_millis() as u64,
                in_flight: e.in_flight,
                attached: e.attached.iter().map(ToString::to_string).collect(),
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
