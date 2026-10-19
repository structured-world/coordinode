//! A named share of the storage engine.
//!
//! Everything in this crate that keeps the engine alive does so through an
//! [`EngineHold`]. After a node's shutdown returns, the caller must be able to
//! reopen the directory, so nothing here may still hold the engine. With the
//! `engine-holders` feature every hold records its name while it lives, and
//! [`engine_holders`] lists the ones still alive, which turns "something still
//! holds the engine" into the name of what does. Without it a hold is the
//! `Arc` alone.

use std::ops::Deref;
use std::sync::Arc;

use coordinode_storage::engine::core::StorageEngine;

/// One named share of the storage engine.
pub struct EngineHold {
    /// Declared first, so the engine is let go before the record of it: the
    /// record never names fewer holders than there are.
    engine: Arc<StorageEngine>,
    #[cfg(feature = "engine-holders")]
    _record: record::Record,
}

impl EngineHold {
    /// Hold `engine` as `name`.
    pub fn new(engine: Arc<StorageEngine>, name: &'static str) -> Self {
        #[cfg(not(feature = "engine-holders"))]
        let _ = name;
        Self {
            #[cfg(feature = "engine-holders")]
            _record: record::Record::new(&engine, name),
            engine,
        }
    }

    /// Another hold of the same engine, as `name`.
    pub fn share(&self, name: &'static str) -> Self {
        Self::new(Arc::clone(&self.engine), name)
    }

    /// The engine's `Arc`, for an API that takes one; a clone taken from it
    /// is not recorded, so a lasting one goes through [`Self::share`].
    pub fn arc(&self) -> &Arc<StorageEngine> {
        &self.engine
    }
}

impl Deref for EngineHold {
    type Target = StorageEngine;

    fn deref(&self) -> &StorageEngine {
        &self.engine
    }
}

impl std::fmt::Debug for EngineHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineHold").finish_non_exhaustive()
    }
}

/// The holds of `engine` this crate still keeps, by name.
#[cfg(feature = "engine-holders")]
pub fn engine_holders(engine: &Arc<StorageEngine>) -> Vec<&'static str> {
    record::of(engine)
}

#[cfg(feature = "engine-holders")]
mod record {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use coordinode_storage::engine::core::StorageEngine;

    /// Every live hold: (engine address, hold id, name). A handful at a time
    /// per engine, so a list is enough.
    static LIVE: parking_lot::Mutex<Vec<(usize, u64, &'static str)>> =
        parking_lot::Mutex::new(Vec::new());
    static NEXT: AtomicU64 = AtomicU64::new(0);

    pub(super) struct Record(u64);

    impl Record {
        pub(super) fn new(engine: &Arc<StorageEngine>, name: &'static str) -> Self {
            let id = NEXT.fetch_add(1, Ordering::Relaxed);
            LIVE.lock().push((address(engine), id, name));
            Self(id)
        }
    }

    impl Drop for Record {
        fn drop(&mut self) {
            LIVE.lock().retain(|(_, id, _)| *id != self.0);
        }
    }

    pub(super) fn of(engine: &Arc<StorageEngine>) -> Vec<&'static str> {
        let at = address(engine);
        LIVE.lock()
            .iter()
            .filter(|(engine, _, _)| *engine == at)
            .map(|(_, _, name)| *name)
            .collect()
    }

    fn address(engine: &Arc<StorageEngine>) -> usize {
        Arc::as_ptr(engine) as usize
    }
}

#[cfg(all(test, feature = "engine-holders"))]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
