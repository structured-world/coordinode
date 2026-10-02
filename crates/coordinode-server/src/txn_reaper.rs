//! Idle reaper for interactive transactions.
//!
//! An open interactive transaction pins an MVCC snapshot and buffers its
//! writes in memory until it commits or rolls back. A client that walks away
//! would hold both forever, so this task drops every transaction left
//! untouched past the idle timeout: the transaction itself in the
//! [`Database`] and its entry in the session registry, so the countdown that
//! SHOW TRANSACTIONS surfaces hits zero exactly when the transaction is gone.
//!
//! The task sleeps until the earliest open transaction can expire, and while
//! none is open until one begins: an idle server runs nothing here.

use std::sync::Arc;
use std::time::Duration;

use coordinode_embed::Database;
use coordinode_session::SessionRegistry;
use parking_lot::RwLock;
use tokio::sync::Notify;

/// Spawn the reaper: drop interactive transactions idle for at least
/// `idle_timeout`. `begun` is notified whenever a transaction opens (the
/// database's begin hook). Passes are at least `granularity` apart, so
/// transactions expiring close together are reaped in one pass. The task
/// runs for the life of the runtime.
pub(crate) fn spawn(
    database: Arc<RwLock<Database>>,
    registry: Arc<SessionRegistry>,
    idle_timeout: Duration,
    begun: Arc<Notify>,
    granularity: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let open =
                crate::services::blocking(|| database.read().reap_idle_transactions(idle_timeout));
            registry.reap_idle();
            let next = match (open, registry.next_idle_deadline()) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            match next {
                Some(at) => {
                    let at = at.max(std::time::Instant::now() + granularity);
                    tokio::time::sleep_until(at.into()).await;
                }
                // A begin while the pass ran left its permit, so this returns.
                None => begun.notified().await,
            }
        }
    })
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
