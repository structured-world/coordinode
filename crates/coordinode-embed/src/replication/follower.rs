//! Brings a member's B-tree, vector and text indexes in line with the index
//! definitions as their entries apply: a replica's copy of a leader's
//! CREATE / DROP INDEX, and on a single member the definitions the log
//! replays after the database opened.

use std::sync::{Arc, Weak};

use coordinode_storage::engine::applied::{AppliedEvent, AppliedStop};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use parking_lot::RwLock;

use crate::Database;

/// Queue of applied Schema events the follower may fall behind on; a full
/// queue arrives as "replaced" and costs one refresh, never a missed one.
const QUEUE: usize = 1024;

/// The follower thread, stopped and joined when dropped.
pub(super) struct IndexDefinitionFollower {
    stop: AppliedStop,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl IndexDefinitionFollower {
    /// Follow `engine`'s applied Schema entries for `database`. Woken only by
    /// an entry that writes an index definition (or replaces the partition),
    /// so an ordinary write costs nothing here.
    pub(super) fn spawn(engine: &Arc<StorageEngine>, database: Weak<RwLock<Database>>) -> Self {
        let applied = engine.subscribe_applied(Partition::Schema, QUEUE);
        let stop = applied.stopper();
        let thread = std::thread::Builder::new()
            .name("index-def-follower".to_string())
            .spawn(move || {
                // Asleep until a Schema entry applies; `None` once stopped.
                while let Some(first) = applied.next(None) {
                    let mut event = Some(first);
                    let mut definitions_changed = false;
                    while let Some(current) = event {
                        definitions_changed |= match current {
                            AppliedEvent::Keys { keys, .. } => {
                                keys.iter().any(|k| k.starts_with(b"schema:idx:"))
                            }
                            // A command or a replaced partition: which keys
                            // changed is not known.
                            AppliedEvent::Replaced => true,
                        };
                        event = applied.try_next();
                    }
                    if !definitions_changed {
                        continue;
                    }
                    let Some(db) = database.upgrade() else {
                        break;
                    };
                    refresh_index_definitions(&db.read());
                }
            })
            .map_err(|e| tracing::error!(%e, "could not start the index definition follower"))
            .ok();
        Self { stop, thread }
    }
}

impl Drop for IndexDefinitionFollower {
    fn drop(&mut self) {
        self.stop.stop();
        if let Some(thread) = self.thread.take() {
            if thread.join().is_err() {
                tracing::error!("the index definition follower panicked");
            }
        }
    }
}

/// Register, rebuild or drop this member's indexes to match the stored
/// definitions.
fn refresh_index_definitions(db: &Database) {
    // Register + local HNSW rebuild (the graph itself is never replicated).
    match db.refresh_vector_indexes() {
        Ok(0) => {}
        Ok(n) => tracing::info!(n, "vector indexes brought live from apply"),
        Err(e) => tracing::warn!(%e, "vector index refresh failed"),
    }
    // Text index definitions another member created or dropped: registered
    // and rebuilt from the store here; the text worker keeps them current
    // from then on.
    match db.refresh_text_indexes() {
        Ok(0) => {}
        Ok(n) => tracing::info!(n, "text indexes brought in line from apply"),
        Err(e) => tracing::warn!(%e, "text index refresh failed"),
    }
    // B-tree definitions another member created or dropped: their entries
    // arrive in the log, the definitions tell this member to maintain and use
    // them.
    if let Err(e) = db.refresh_btree_indexes() {
        tracing::warn!(%e, "B-tree index refresh failed");
    }
}
