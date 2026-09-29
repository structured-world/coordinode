//! Idle reaper for interactive transactions.
//!
//! An open interactive transaction pins an MVCC snapshot and buffers its
//! writes in memory until it commits or rolls back. A client that walks away
//! would hold both forever, so this task drops every transaction left
//! untouched past the idle timeout: the transaction itself in the
//! [`Database`] and its entry in the session registry, so the countdown that
//! SHOW TRANSACTIONS surfaces hits zero exactly when the transaction is gone.

use std::sync::Arc;
use std::time::Duration;

use coordinode_embed::Database;
use coordinode_session::SessionRegistry;
use parking_lot::RwLock;

/// Spawn the reaper: every `tick`, drop interactive transactions idle for at
/// least `idle_timeout`. The task runs for the life of the runtime.
pub(crate) fn spawn(
    database: Arc<RwLock<Database>>,
    registry: Arc<SessionRegistry>,
    idle_timeout: Duration,
    tick: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(tick);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            ticker.tick().await;
            crate::services::blocking(|| database.read().reap_idle_transactions(idle_timeout));
            registry.reap_idle();
        }
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
