//! Notices that a registration's record may have changed, so a reader that
//! has to know whether its registration still stands rechecks it when a
//! write to it applies instead of on every read, and a reader parked until
//! something happens is woken by such a write.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;
use tokio::sync::Notify;

/// A counter bumped as writes apply, and the readers waiting for it to move.
#[derive(Debug, Default)]
struct Generation {
    value: AtomicU64,
    moved: Notify,
}

impl Generation {
    fn bump(&self) {
        self.value.fetch_add(1, Ordering::Release);
        self.moved.notify_waiters();
    }

    fn get(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }
}

/// The generation of every watched record, bumped as writes to it apply.
#[derive(Debug, Default)]
pub(crate) struct Watches {
    /// Whether applied writes are being relayed. Without the relay nothing
    /// bumps a generation, so every watch reports a change.
    relayed: AtomicBool,
    /// Bumped when an applied write's keys are not known: any record may
    /// have changed.
    all: Generation,
    /// Generations by record key; an entry lives while a watch holds it.
    by_key: Mutex<HashMap<Vec<u8>, Weak<Generation>>>,
}

impl Watches {
    /// Writes to `keys` applied.
    pub(crate) fn applied(&self, keys: &[Vec<u8>]) {
        let by_key = self.by_key.lock();
        for key in keys {
            if let Some(generation) = by_key.get(key).and_then(Weak::upgrade) {
                generation.bump();
            }
        }
    }

    /// Writes whose keys are not known applied.
    pub(crate) fn applied_unknown(&self) {
        self.all.bump();
    }

    /// Whether applied writes are relayed from now on.
    pub(crate) fn set_relayed(&self, relayed: bool) {
        self.relayed.store(relayed, Ordering::Release);
        // A reader waiting for the relay to start looks again.
        self.all.moved.notify_waiters();
    }

    /// A watch over the record stored under `key`.
    pub(crate) fn watch(self: &Arc<Self>, key: Vec<u8>) -> RegistrationWatch {
        let generation = {
            let mut by_key = self.by_key.lock();
            match by_key.get(&key).and_then(Weak::upgrade) {
                Some(generation) => generation,
                None => {
                    let generation = Arc::new(Generation::default());
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
    generation: Arc<Generation>,
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
        let now = self.now();
        let changed = self.seen != Some(now);
        self.seen = Some(now);
        changed
    }

    /// Resolves once a write to the record, or one whose keys are not known,
    /// applied since the last [`changed`](Self::changed). A reader parked
    /// until something happens waits on this beside its other wake-ups, so it
    /// does not learn of the end of its registration only at its next one.
    /// Without the relay it waits for the relay to start: such a reader
    /// rechecks at every wake-up anyway.
    pub async fn wait(&self) {
        loop {
            let record = self.generation.moved.notified();
            let any = self.watches.all.moved.notified();
            tokio::pin!(record, any);
            // Registered before the look, so a bump in between still wakes.
            record.as_mut().enable();
            any.as_mut().enable();
            if self.watches.relayed.load(Ordering::Acquire) && self.seen != Some(self.now()) {
                return;
            }
            tokio::select! {
                () = record => {}
                () = any => {}
            }
        }
    }

    fn now(&self) -> (u64, u64) {
        (self.generation.get(), self.watches.all.get())
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
