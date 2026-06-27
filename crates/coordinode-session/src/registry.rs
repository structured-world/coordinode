//! Live registry of client sessions and their open transactions.
//!
//! Node-local source of truth for operational introspection (`SHOW SESSIONS` /
//! `SHOW TRANSACTIONS`): every open session, the requests in flight on it, and
//! the interactive transactions it holds with the time left before the idle
//! reaper auto-aborts each. The session core updates it as sessions open/close,
//! requests start/finish, and transactions begin/commit/roll back; the query
//! executor reads a snapshot through [`OperationsView`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{Duration, Instant}; // no-std: caller-provided Clock trait

use coordinode_core::operations::{
    OperationsView, SessionSnapshot, TransactionSnapshot, TxnOrdering,
};
use parking_lot::RwLock;

use crate::types::Ordering;

/// Node-local registry of live sessions and their open transactions.
///
/// Cheap to clone behind an `Arc`: the session binding holds one, and so does
/// the query layer (as a `dyn OperationsView`), both pointing at the same map.
pub struct SessionRegistry {
    sessions: RwLock<HashMap<u64, SessionEntry>>,
    next_session_id: AtomicU64,
    /// Idle window after which an untouched interactive transaction is reaped
    /// (auto-aborted). Also drives the `auto_abort_in_ms` countdown.
    idle_timeout: Duration,
}

struct SessionEntry {
    peer: String,
    opened_at: Instant,
    in_flight: u64,
    transactions: HashMap<u64, TxnEntry>,
}

struct TxnEntry {
    ordering: Ordering,
    begun_at: Instant,
    last_activity: Instant,
}

impl SessionRegistry {
    /// Create an empty registry whose transactions auto-abort after
    /// `idle_timeout` of inactivity.
    pub fn new(idle_timeout: Duration) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            next_session_id: AtomicU64::new(0),
            idle_timeout,
        }
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
                transactions: HashMap::new(),
            },
        );
        id
    }

    /// Remove a session and all its open transactions (stream closed).
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

    /// Register an interactive transaction opened by the session.
    pub fn begin_txn(&self, session_id: u64, txid: u64, ordering: Ordering) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            let now = Instant::now();
            e.transactions.insert(
                txid,
                TxnEntry {
                    ordering,
                    begun_at: now,
                    last_activity: now,
                },
            );
        }
    }

    /// Refresh a transaction's activity clock (a statement ran inside it),
    /// resetting its auto-abort countdown.
    pub fn touch_txn(&self, session_id: u64, txid: u64) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            if let Some(t) = e.transactions.get_mut(&txid) {
                t.last_activity = Instant::now();
            }
        }
    }

    /// Remove a transaction (committed or rolled back).
    pub fn end_txn(&self, session_id: u64, txid: u64) {
        if let Some(e) = self.sessions.write().get_mut(&session_id) {
            e.transactions.remove(&txid);
        }
    }

    /// Drop every transaction idle beyond `idle_timeout`, returning the
    /// `(session_id, txid)` pairs reaped so the caller can abort their engine
    /// state. The countdown surfaced by [`OperationsView`] hits zero exactly
    /// when a transaction becomes eligible here.
    pub fn reap_idle(&self) -> Vec<(u64, u64)> {
        let now = Instant::now();
        let mut reaped = Vec::new();
        let mut guard = self.sessions.write();
        for (session_id, entry) in guard.iter_mut() {
            entry.transactions.retain(|txid, t| {
                let expired = now.duration_since(t.last_activity) >= self.idle_timeout;
                if expired {
                    reaped.push((*session_id, *txid));
                }
                !expired
            });
        }
        reaped
    }
}

impl OperationsView for SessionRegistry {
    fn sessions(&self) -> Vec<SessionSnapshot> {
        let now = Instant::now();
        let guard = self.sessions.read();
        guard
            .iter()
            .map(|(id, e)| SessionSnapshot {
                session_id: format!("s-{id}"),
                peer: e.peer.clone(),
                age_ms: now.duration_since(e.opened_at).as_millis() as u64,
                in_flight: e.in_flight,
                transactions: e
                    .transactions
                    .iter()
                    .map(|(txid, t)| {
                        let idle = now.duration_since(t.last_activity);
                        // Remaining = timeout - idle, floored at zero (a txn past
                        // its deadline but not yet reaped reads as 0, not wrapped).
                        let remaining = self.idle_timeout.checked_sub(idle).unwrap_or_default();
                        TransactionSnapshot {
                            txid: *txid,
                            ordering: match t.ordering {
                                Ordering::Ordered => TxnOrdering::Ordered,
                                Ordering::Unordered => TxnOrdering::Unordered,
                            },
                            age_ms: now.duration_since(t.begun_at).as_millis() as u64,
                            auto_abort_in_ms: remaining.as_millis() as u64,
                        }
                    })
                    .collect(),
            })
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
