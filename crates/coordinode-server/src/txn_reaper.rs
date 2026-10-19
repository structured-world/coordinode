//! Idle reaper for interactive transactions.
//!
//! An open interactive transaction pins an MVCC snapshot and buffers its
//! writes in memory until it commits or rolls back. A client that walks away
//! would hold both forever, so this task drops every transaction left
//! untouched past the idle timeout. The transactions live in the
//! [`Database`] that owns them, whatever session sent their commands, so
//! the countdown SHOW TRANSACTIONS surfaces hits zero exactly when the
//! transaction is gone.
//!
//! The task sleeps until the earliest open transaction can expire, and while
//! none is open until one begins: an idle server runs nothing here.

use std::sync::Arc;
use std::time::Duration;

use coordinode_embed::Database;
use parking_lot::RwLock;
use tokio::sync::Notify;

/// Spawn the reaper: drop interactive transactions idle for at least
/// `idle_timeout`. `begun` is notified whenever a transaction opens (the
/// database's begin hook). Passes are at least `granularity` apart, so
/// transactions expiring close together are reaped in one pass. The task
/// runs for the life of the runtime.
pub(crate) fn spawn(
    database: Arc<RwLock<Database>>,
    idle_timeout: Duration,
    begun: Arc<Notify>,
    granularity: Duration,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let next =
                crate::services::blocking(|| database.read().reap_idle_transactions(idle_timeout));
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
