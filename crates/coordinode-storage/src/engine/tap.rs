//! Taps on the stream of writes a partition receives.
//!
//! A consumer that derives state from a partition (an index built beside
//! live writes) needs two things at once: the partition's content at some
//! snapshot, and every write that snapshot does not contain. No timestamp
//! selects the second set, because a write applies at its own commit
//! timestamp and can land below a snapshot taken before it. A tap selects it
//! by position instead: it sees every write that reaches the partition after
//! the tap is open, whatever its timestamp, and
//! [`StorageEngine::tap_writes`](crate::engine::core::StorageEngine::tap_writes)
//! returns a snapshot holding every write that finished before.
//!
//! A tap records keys, not values: the consumer re-reads what it needs, so a
//! key written many times costs one entry. A write that replaces the
//! partition wholesale (a clear, a range tombstone, a range drop) cannot be
//! listed key by key, and marks the tap [`Tapped::Replaced`] instead.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering, fence};

use parking_lot::{Mutex, RwLock};
use rustc_hash::FxHashSet;

use crate::engine::batch::Mutation;
use crate::engine::partition::Partition;

/// The taps open on one engine.
#[derive(Debug, Default)]
pub(crate) struct WriteTaps {
    /// How many taps are open. Read on every write, so a write with no tap
    /// open pays one load and a fence.
    open: AtomicUsize,
    taps: RwLock<Vec<Arc<TapBuffer>>>,
}

#[derive(Debug)]
struct TapBuffer {
    partition: Partition,
    state: Mutex<TapState>,
}

#[derive(Debug, Default)]
struct TapState {
    keys: FxHashSet<Vec<u8>>,
    replaced: bool,
}

/// What a tap collected since it was last read.
#[derive(Debug, PartialEq, Eq)]
pub enum Tapped {
    /// The keys written, each once, in no particular order.
    Keys(Vec<Vec<u8>>),
    /// The partition was replaced wholesale: the keys it lost are not
    /// known, so the consumer starts over from a fresh snapshot
    /// ([`StorageEngine::rebase_tap`](crate::engine::core::StorageEngine::rebase_tap)).
    Replaced,
}

/// An open tap. Closes when dropped.
#[derive(Debug)]
pub struct WriteTap {
    buffer: Arc<TapBuffer>,
    taps: Arc<WriteTaps>,
}

impl WriteTap {
    /// The partition this tap watches.
    pub fn partition(&self) -> Partition {
        self.buffer.partition
    }

    /// Everything written since the last call, leaving the tap empty.
    pub fn take(&self) -> Tapped {
        let mut state = self.buffer.state.lock();
        if state.replaced {
            state.replaced = false;
            state.keys.clear();
            return Tapped::Replaced;
        }
        Tapped::Keys(state.keys.drain().collect())
    }

    /// Discard what the tap holds: the consumer is about to read the
    /// partition afresh.
    pub(crate) fn reset(&self) {
        let mut state = self.buffer.state.lock();
        state.replaced = false;
        state.keys.clear();
    }
}

impl Drop for WriteTap {
    fn drop(&mut self) {
        let mut taps = self.taps.taps.write();
        taps.retain(|t| !Arc::ptr_eq(t, &self.buffer));
        self.taps.open.store(taps.len(), Ordering::SeqCst);
    }
}

impl WriteTaps {
    /// Open a tap on `partition`. The fence after publishing it pairs with
    /// the one every write takes before reading [`Self::open`]: a write that
    /// reads zero there finished before this fence, so a snapshot taken
    /// after this returns holds it.
    pub(crate) fn open(self: &Arc<Self>, partition: Partition) -> WriteTap {
        let buffer = Arc::new(TapBuffer {
            partition,
            state: Mutex::new(TapState::default()),
        });
        {
            let mut taps = self.taps.write();
            taps.push(Arc::clone(&buffer));
            self.open.store(taps.len(), Ordering::SeqCst);
        }
        fence(Ordering::SeqCst);
        WriteTap {
            buffer,
            taps: Arc::clone(self),
        }
    }

    /// Record that `keys` of `partition` were written. Call after the write
    /// is in the tree.
    #[inline]
    pub(crate) fn wrote<'k>(&self, partition: Partition, keys: impl IntoIterator<Item = &'k [u8]>) {
        // The write is in the tree before this fence, and the tap is
        // published before the fence that opening one takes, so of the two
        // fences whichever comes first decides: either this load sees the tap
        // or the snapshot the opener takes afterwards sees the write.
        fence(Ordering::SeqCst);
        if self.open.load(Ordering::Relaxed) == 0 {
            return;
        }
        self.deliver(partition, keys);
    }

    /// Record one partition's group of an applied batch: its keys, or a
    /// replacement when the group carries a range tombstone.
    #[inline]
    pub(crate) fn applied(&self, partition: Partition, group: &[&Mutation]) {
        fence(Ordering::SeqCst);
        if self.open.load(Ordering::Relaxed) == 0 {
            return;
        }
        if group
            .iter()
            .any(|m| matches!(m, Mutation::RemoveRange { .. }))
        {
            self.mark_replaced(partition);
            return;
        }
        self.deliver(
            partition,
            group.iter().filter_map(|m| match m {
                Mutation::Put { key, .. }
                | Mutation::Delete { key, .. }
                | Mutation::Merge { key, .. } => Some(key.as_slice()),
                Mutation::RemoveRange { .. } => None,
            }),
        );
    }

    /// Record that `partition` was replaced wholesale.
    #[inline]
    pub(crate) fn replaced(&self, partition: Partition) {
        fence(Ordering::SeqCst);
        if self.open.load(Ordering::Relaxed) == 0 {
            return;
        }
        self.mark_replaced(partition);
    }

    #[cold]
    fn mark_replaced(&self, partition: Partition) {
        for tap in self.taps.read().iter() {
            if tap.partition == partition {
                let mut state = tap.state.lock();
                state.replaced = true;
                state.keys.clear();
            }
        }
    }

    #[cold]
    fn deliver<'k>(&self, partition: Partition, keys: impl IntoIterator<Item = &'k [u8]>) {
        let taps = self.taps.read();
        let mut matching = taps.iter().filter(|t| t.partition == partition).peekable();
        if matching.peek().is_none() {
            return;
        }
        let keys: Vec<&[u8]> = keys.into_iter().collect();
        for tap in matching {
            let mut state = tap.state.lock();
            if state.replaced {
                continue;
            }
            for key in &keys {
                if !state.keys.contains(*key) {
                    state.keys.insert(key.to_vec());
                }
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
