//! The set of trees a capture takes, held still against what changes it
//! outside a single commit.
//!
//! A checkpoint pins the store's trees in turn. Most writes are one commit,
//! whole or absent at the checkpoint's cut. Some operations change the
//! trees in steps no commit gathers: a statement writing `STORAGE COLUMNAR`
//! rows (each lands as it is made), one creating or dropping such a table
//! (its tree is made or removed before its catalog entry commits), a repair
//! rebuilding a tree (cleared, then filled again). Each such operation holds
//! the set for its whole span ([`TreeSetHold`]); a capture holds it alone
//! until every tree is pinned ([`TreeSetGate::exclusive`]), and so never
//! sees one half done. A repair holds it alone from its first destructive
//! step until its result is installed.
//!
//! A capture or repair waiting for the set keeps new holders out, so a
//! steady stream of operations cannot starve it; it waits a bounded time
//! and is refused, retryably, past it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::engine::core::StorageEngine;
use crate::error::StorageResult;

/// The `STORAGE COLUMNAR` tables a store's catalog declares, read by the
/// layer that owns the catalog: the engine holds the tables' trees, not the
/// catalog's encoding. A capture checks the trees it took against it.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot list the `STORAGE COLUMNAR` tables a catalog declares",
    label = "not a catalog inventory",
    note = "`coordinode_modality::CatalogTables` lists them from the schema store"
)]
pub trait CatalogInventory: Send + Sync {
    /// The tables the catalog of `engine` declares now, by name.
    ///
    /// # Errors
    ///
    /// The catalog cannot be read.
    fn columnar_tables(&self, engine: &StorageEngine) -> StorageResult<Vec<String>>;
}

/// Who has the set.
#[derive(Debug, Default)]
struct Gate {
    /// Operations holding the set shared.
    held: usize,
    /// A capture or repair holds the set alone.
    exclusive: bool,
    /// Captures or repairs waiting for the set alone.
    waiting: usize,
    /// Shared holds ever granted, for tests that count them.
    #[cfg(any(test, feature = "test-faults"))]
    granted: u64,
}

/// The engine's gate over its set of trees; see the [module docs](self).
#[derive(Debug, Default)]
pub struct TreeSetGate {
    state: Mutex<Gate>,
    changed: Condvar,
}

impl TreeSetGate {
    /// Hold the set for an operation that changes the trees in steps: waits
    /// while a capture or repair has it or waits for it.
    pub fn hold(self: &Arc<Self>) -> TreeSetHold {
        let mut gate = self.state.lock();
        while gate.exclusive || gate.waiting > 0 {
            self.changed.wait(&mut gate);
        }
        // Bounded by the operations running at once.
        gate.held += 1;
        #[cfg(any(test, feature = "test-faults"))]
        {
            gate.granted += 1;
        }
        TreeSetHold {
            gate: Arc::clone(self),
        }
    }

    /// Have the set alone, once no operation holds it, or `None` when one
    /// still does after `timeout`.
    #[must_use]
    pub fn exclusive(&self, timeout: Duration) -> Option<TreeSetExclusive<'_>> {
        let deadline = Instant::now() + timeout;
        let mut gate = self.state.lock();
        // Bounded by the captures and repairs waiting at once.
        gate.waiting += 1;
        while gate.held > 0 || gate.exclusive {
            if self.changed.wait_until(&mut gate, deadline).timed_out()
                && (gate.held > 0 || gate.exclusive)
            {
                gate.waiting -= 1;
                // The operations it kept waiting may go on.
                self.changed.notify_all();
                return None;
            }
        }
        gate.waiting -= 1;
        gate.exclusive = true;
        Some(TreeSetExclusive { gate: self })
    }

    /// Shared holds granted so far.
    #[cfg(any(test, feature = "test-faults"))]
    #[must_use]
    pub fn holds_granted(&self) -> u64 {
        self.state.lock().granted
    }
}

/// An operation's shared hold on the set of trees, given back when it
/// drops. Owned, so an interactive transaction parks it between statements.
#[derive(Debug)]
#[must_use = "the set is held until this drops"]
pub struct TreeSetHold {
    gate: Arc<TreeSetGate>,
}

impl Drop for TreeSetHold {
    fn drop(&mut self) {
        let mut gate = self.gate.state.lock();
        gate.held -= 1;
        if gate.held == 0 {
            self.gate.changed.notify_all();
        }
    }
}

/// A capture's or repair's sole hold on the set of trees, given back when it
/// drops.
#[derive(Debug)]
#[must_use = "the set is held alone until this drops"]
pub struct TreeSetExclusive<'g> {
    gate: &'g TreeSetGate,
}

impl Drop for TreeSetExclusive<'_> {
    fn drop(&mut self) {
        self.gate.state.lock().exclusive = false;
        self.gate.changed.notify_all();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
