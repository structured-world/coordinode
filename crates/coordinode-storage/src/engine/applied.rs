//! A feed of the Raft entries this node has applied.
//!
//! State derived from the replicated data but kept outside it (a vector
//! index's graph) has to follow the entries in the order they apply, and only
//! once they have: an entry appended to the log may never commit, and one
//! committed is not in the store until it applies. The Raft state machine
//! applies through the engine, so the engine is where an entry becomes part
//! of the store; a subscriber receives each applied entry's keys there, with
//! the entry's log index and commit timestamp.
//!
//! The feed never slows the applies down. Each subscriber has a bounded
//! queue; when it is full the event is dropped and the subscription marked
//! lost, and the subscriber rebuilds from the store, which holds every entry
//! the events it missed carried. A store replaced wholesale (a snapshot
//! installed, a partition installed from a peer, a range removed) cannot be
//! listed key by key and arrives as [`AppliedEvent::Replaced`].

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Weak};
use std::time::Duration;

use coordinode_core::txn::proposal::Mutation;
use parking_lot::RwLock;

use crate::engine::partition::Partition;

/// What happened to a subscribed partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppliedEvent {
    /// Raft entry `index` applied at `commit_ts`, writing `keys` (each once
    /// per mutation, in the entry's order) in the subscribed partition.
    Keys {
        /// The entry's position in the Raft log.
        index: u64,
        /// The commit timestamp the entry applied at.
        commit_ts: u64,
        /// The keys the entry wrote in the subscribed partition.
        keys: Vec<Vec<u8>>,
    },
    /// The partition was replaced wholesale, or events were dropped: the
    /// keys that changed are not known, so the subscriber reads the store
    /// afresh.
    Replaced,
}

/// The subscribers of one engine.
#[derive(Debug, Default)]
pub(crate) struct AppliedFeed {
    /// How many subscriptions are open. Read on every apply, so an apply
    /// with none open pays one load.
    open: AtomicUsize,
    subscribers: RwLock<Vec<Arc<Subscriber>>>,
}

#[derive(Debug)]
struct Subscriber {
    partition: Partition,
    tx: SyncSender<AppliedEvent>,
    /// Set when an event could not be queued; cleared by the subscriber.
    lost: AtomicBool,
}

/// An open subscription. Closes when dropped.
#[derive(Debug)]
pub struct AppliedSubscription {
    rx: Receiver<AppliedEvent>,
    subscriber: Arc<Subscriber>,
    feed: Weak<AppliedFeed>,
}

impl AppliedSubscription {
    /// The next event, waiting at most `wait`. `None` when none arrived.
    /// An event the queue could not hold comes back as
    /// [`AppliedEvent::Replaced`], once, before the events queued after it.
    pub fn next(&self, wait: Duration) -> Option<AppliedEvent> {
        if self.subscriber.lost.swap(false, Ordering::AcqRel) {
            return Some(AppliedEvent::Replaced);
        }
        // A timeout or a closed feed both mean nothing arrived.
        self.rx.recv_timeout(wait).ok()
    }

    /// An event already queued, without waiting.
    pub fn try_next(&self) -> Option<AppliedEvent> {
        if self.subscriber.lost.swap(false, Ordering::AcqRel) {
            return Some(AppliedEvent::Replaced);
        }
        self.rx.try_recv().ok()
    }

    /// The partition this subscription follows.
    pub fn partition(&self) -> Partition {
        self.subscriber.partition
    }
}

impl Drop for AppliedSubscription {
    fn drop(&mut self) {
        if let Some(feed) = self.feed.upgrade() {
            let mut subscribers = feed.subscribers.write();
            subscribers.retain(|s| !Arc::ptr_eq(s, &self.subscriber));
            feed.open.store(subscribers.len(), Ordering::Release);
        }
    }
}

impl AppliedFeed {
    /// How many subscriptions are open.
    #[cfg(test)]
    pub(crate) fn open(&self) -> usize {
        self.open.load(Ordering::Acquire)
    }

    /// Subscribe to the entries applied to `partition` from now on, queueing
    /// at most `capacity` of them.
    pub(crate) fn subscribe(
        self: &Arc<Self>,
        partition: Partition,
        capacity: usize,
    ) -> AppliedSubscription {
        let (tx, rx) = sync_channel(capacity.max(1));
        let subscriber = Arc::new(Subscriber {
            partition,
            tx,
            lost: AtomicBool::new(false),
        });
        {
            let mut subscribers = self.subscribers.write();
            subscribers.push(Arc::clone(&subscriber));
            self.open.store(subscribers.len(), Ordering::Release);
        }
        AppliedSubscription {
            rx,
            subscriber,
            feed: Arc::downgrade(self),
        }
    }

    /// Report that Raft entry `index` applied `mutations` at `commit_ts`.
    /// Call after they are in the store.
    #[inline]
    pub(crate) fn applied(&self, index: u64, commit_ts: u64, mutations: &[Mutation]) {
        if self.open.load(Ordering::Acquire) == 0 {
            return;
        }
        self.deliver(index, commit_ts, mutations);
    }

    /// Report that `partition` was replaced wholesale, or every partition
    /// when `None`.
    #[inline]
    pub(crate) fn replaced(&self, partition: Option<Partition>) {
        if self.open.load(Ordering::Acquire) == 0 {
            return;
        }
        for subscriber in self.subscribers.read().iter() {
            if partition.is_none_or(|p| p == subscriber.partition) {
                send(subscriber, AppliedEvent::Replaced);
            }
        }
    }

    #[cold]
    fn deliver(&self, index: u64, commit_ts: u64, mutations: &[Mutation]) {
        for subscriber in self.subscribers.read().iter() {
            let mut keys = Vec::new();
            let mut replaced = false;
            for mutation in mutations {
                match mutation {
                    Mutation::Put { partition, key, .. }
                    | Mutation::Delete { partition, key }
                    | Mutation::Merge { partition, key, .. } => {
                        if Partition::from(*partition) == subscriber.partition {
                            keys.push(key.clone());
                        }
                    }
                    Mutation::RemoveRange { partition, .. } => {
                        if Partition::from(*partition) == subscriber.partition {
                            replaced = true;
                        }
                    }
                    // The keys a command writes are decided as it applies and
                    // are not in the entry, so a Schema follower rereads.
                    Mutation::Command(_) => {
                        if subscriber.partition == Partition::Schema {
                            replaced = true;
                        }
                    }
                    // Derived entry keys are likewise decided as the unit
                    // applies, so an index-partition follower rereads.
                    Mutation::Derive(_) => {
                        if subscriber.partition == Partition::Idx {
                            replaced = true;
                        }
                    }
                }
            }
            if replaced {
                send(subscriber, AppliedEvent::Replaced);
            } else if !keys.is_empty() {
                send(
                    subscriber,
                    AppliedEvent::Keys {
                        index,
                        commit_ts,
                        keys,
                    },
                );
            }
        }
    }
}

/// Queue `event` without waiting; a full queue marks the subscription lost.
fn send(subscriber: &Subscriber, event: AppliedEvent) {
    match subscriber.tx.try_send(event) {
        Ok(()) | Err(TrySendError::Disconnected(_)) => {}
        Err(TrySendError::Full(_)) => subscriber.lost.store(true, Ordering::Release),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
