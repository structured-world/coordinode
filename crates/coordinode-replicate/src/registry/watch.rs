//! Notices that a registration's record may have changed, so a reader that
//! has to know whether its registration still stands rechecks it when a
//! write to it applies instead of on every read.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

/// The generation of every watched record, bumped as writes to it apply.
#[derive(Debug, Default)]
pub(crate) struct Watches {
    /// Whether applied writes are being relayed. Without the relay nothing
    /// bumps a generation, so every watch reports a change.
    relayed: AtomicBool,
    /// Bumped when an applied write's keys are not known: any record may
    /// have changed.
    all: AtomicU64,
    /// Generations by record key; an entry lives while a watch holds it.
    by_key: Mutex<HashMap<Vec<u8>, Weak<AtomicU64>>>,
}

impl Watches {
    /// Writes to `keys` applied.
    pub(crate) fn applied(&self, keys: &[Vec<u8>]) {
        let by_key = self.by_key.lock();
        for key in keys {
            if let Some(generation) = by_key.get(key).and_then(Weak::upgrade) {
                generation.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// Writes whose keys are not known applied.
    pub(crate) fn applied_unknown(&self) {
        self.all.fetch_add(1, Ordering::Release);
    }

    /// Whether applied writes are relayed from now on.
    pub(crate) fn set_relayed(&self, relayed: bool) {
        self.relayed.store(relayed, Ordering::Release);
    }

    /// A watch over the record stored under `key`.
    pub(crate) fn watch(self: &Arc<Self>, key: Vec<u8>) -> RegistrationWatch {
        let generation = {
            let mut by_key = self.by_key.lock();
            match by_key.get(&key).and_then(Weak::upgrade) {
                Some(generation) => generation,
                None => {
                    let generation = Arc::new(AtomicU64::new(0));
                    by_key.insert(key.clone(), Arc::downgrade(&generation));
                    generation
                }
            }
        };
        RegistrationWatch {
            watches: Arc::clone(self),
            key,
            generation,
            seen: None,
        }
    }
}

/// Reports whether one registration's record may have changed since the
/// last time it was asked. Obtained from
/// [`ShardConsumerRegistry::watch`](super::ShardConsumerRegistry::watch).
#[derive(Debug)]
pub struct RegistrationWatch {
    watches: Arc<Watches>,
    key: Vec<u8>,
    generation: Arc<AtomicU64>,
    /// The generations seen at the last answer; `None` before the first.
    seen: Option<(u64, u64)>,
}

impl RegistrationWatch {
    /// `true` when a write to the record may have applied since the last
    /// call, and on the first call. A write is noticed once it has applied,
    /// so a read of the record after a `true` sees it. Always `true` while
    /// the registry's background service is not relaying applied writes.
    pub fn changed(&mut self) -> bool {
        if !self.watches.relayed.load(Ordering::Acquire) {
            return true;
        }
        let now = (
            self.generation.load(Ordering::Acquire),
            self.watches.all.load(Ordering::Acquire),
        );
        let changed = self.seen != Some(now);
        self.seen = Some(now);
        changed
    }
}

impl Drop for RegistrationWatch {
    fn drop(&mut self) {
        // The map is locked across the count, so no watch of the same key
        // upgrades the entry in between.
        let mut by_key = self.watches.by_key.lock();
        if Arc::strong_count(&self.generation) == 1 {
            by_key.remove(&self.key);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
