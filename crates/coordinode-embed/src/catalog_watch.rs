//! Wakes the workers of the indexes kept outside the store (full-text,
//! vector) when the catalog changes what a label's carriers are.
//!
//! Those workers follow the node writes applied on this member; a change of
//! what a type implies writes no node, only label schemas and the label
//! lookup's definition. [`CatalogWatch`] follows the applied Schema entries
//! and nudges each worker when one of them writes a label schema or an index
//! definition, so the worker brings its indexes to the new interpretations
//! without waiting for an unrelated write.

use coordinode_storage::engine::applied::{AppliedEvent, AppliedNudge, AppliedStop};
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;

/// Queue of applied Schema events the watch may fall behind on; a full queue
/// arrives as "replaced", which nudges as well.
const QUEUE: usize = 256;

/// The watch thread, stopped and joined when dropped.
pub(crate) struct CatalogWatch {
    stop: AppliedStop,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl CatalogWatch {
    /// Follow `engine`'s applied Schema entries, nudging every one of
    /// `workers` when an entry may change a label's carriers.
    pub(crate) fn spawn(engine: &StorageEngine, workers: Vec<AppliedNudge>) -> Self {
        let applied = engine.subscribe_applied(Partition::Schema, QUEUE);
        let stop = applied.stopper();
        let thread = std::thread::Builder::new()
            .name("catalog-watch".to_string())
            .spawn(move || {
                // Asleep until a Schema entry applies; `None` once stopped.
                while let Some(first) = applied.next(None) {
                    let mut event = Some(first);
                    let mut changed = false;
                    while let Some(current) = event {
                        changed |= match current {
                            AppliedEvent::Keys { keys, .. } => keys.iter().any(|key| {
                                key.starts_with(b"schema:label")
                                    || key.starts_with(
                                        coordinode_modality::IndexDefinition::SCHEMA_PREFIX,
                                    )
                            }),
                            // A command or a replaced partition: which keys
                            // changed is not known.
                            AppliedEvent::Replaced => true,
                        };
                        event = applied.try_next();
                    }
                    if changed {
                        for worker in &workers {
                            worker.nudge();
                        }
                    }
                }
            })
            .map_err(|e| tracing::error!(%e, "could not start the catalog watch"))
            .ok();
        Self { stop, thread }
    }
}

impl Drop for CatalogWatch {
    fn drop(&mut self) {
        self.stop.stop();
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            tracing::error!("the catalog watch panicked");
        }
    }
}
