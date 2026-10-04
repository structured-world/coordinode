//! A feed of the commits this node has applied.
//!
//! State derived from the replicated data but kept outside it (a vector
//! index's graph) has to follow the entries in the order they apply, and only
//! once they have: an entry appended to the log may never commit, and one
//! committed is not in the store until it applies. The Raft state machine
//! applies through the engine, so the engine is where an entry becomes part
//! of the store; a subscriber receives each applied entry's keys there, with
//! the entry's log index and commit timestamp. An engine without Raft reports
//! its local commits the same way, with their journal index (or 0 when it
//! keeps no journal).
//!
//! The feed never slows the applies down. Each subscriber has a bounded
//! queue; when it is full the event is dropped and the subscription marked
//! lost, and the subscriber rebuilds from the store, which holds every entry
//! the events it missed carried. A store replaced wholesale (a snapshot
//! installed, a partition installed from a peer, a range removed) cannot be
//! listed key by key and arrives as [`AppliedEvent::Replaced`].

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError, sync_channel};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use coordinode_core::txn::proposal::Mutation;
use coordinode_core::txn::wake::Wake;
use parking_lot::RwLock;

use crate::engine::partition::Partition;

/// What happened to a subscribed partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppliedEvent {
    /// Entry `index` applied at `commit_ts`, writing `keys` (each once per
    /// mutation, in the entry's order) in the subscribed partition.
    Keys {
        /// This event's number in the subscription, from 1 in the order
        /// events were queued; a dropped event takes a number too. Every
        /// event numbered up to [`AppliedPosition::delivered`] is in the store.
        seq: u64,
        /// The entry's position in the Raft log, or in the journal of an
        /// engine without Raft; 0 for a local commit with no journal.
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
    /// The number of the last event offered (queued or dropped). Written
    /// under `sending`, so numbers enter the queue in order.
    delivered: AtomicU64,
    /// Serializes numbering an event with queueing it: local commits apply
    /// concurrently, and a later number queued ahead of an earlier one would
    /// let the consumer claim the earlier one folded.
    // no-std: spin::Mutex; held for one counter bump and one queue push.
    sending: parking_lot::Mutex<()>,
    tx: SyncSender<AppliedEvent>,
    /// Set when an event could not be queued; cleared by the subscriber.
    lost: AtomicBool,
    /// Notified on every event queued or lost, so a waiting subscriber sleeps
    /// until there is something to take.
    wake: Wake,
    /// Set by [`AppliedStop::stop`]; a waiting subscriber returns `None`.
    stopped: AtomicBool,
}

/// An open subscription. Closes when dropped.
#[derive(Debug)]
pub struct AppliedSubscription {
    rx: Receiver<AppliedEvent>,
    subscriber: Arc<Subscriber>,
    feed: Weak<AppliedFeed>,
}

/// How far the events of one subscription have been offered, readable from
/// any thread.
#[derive(Debug, Clone)]
pub struct AppliedPosition(Arc<Subscriber>);

impl AppliedPosition {
    /// The number of the last event offered to the subscription. Every entry
    /// whose event is numbered at or below it is already in the store, so a
    /// consumer that has folded up to this number covers every entry applied
    /// before this call.
    pub fn delivered(&self) -> u64 {
        self.0.delivered.load(Ordering::Acquire)
    }
}

/// Ends the waits of one subscription from another thread.
#[derive(Debug, Clone)]
pub struct AppliedStop(Arc<Subscriber>);

impl AppliedStop {
    /// Make every current and later [`AppliedSubscription::next`] that would
    /// wait return `None` instead.
    pub fn stop(&self) {
        self.0.stopped.store(true, Ordering::Release);
        self.0.wake.interrupt();
    }
}

impl AppliedSubscription {
    /// The next event, waiting at most `wait`, or until one arrives or the
    /// subscription is stopped when `wait` is `None`. `None` when none
    /// arrived. An event the queue could not hold comes back as
    /// [`AppliedEvent::Replaced`], once, before the events queued after it.
    ///
    /// The thread that waits is the one the events wake, so one thread
    /// consumes a subscription.
    pub fn next(&self, wait: Option<Duration>) -> Option<AppliedEvent> {
        let deadline = wait.map(|wait| Instant::now() + wait);
        loop {
            if let Some(event) = self.try_next() {
                return Some(event);
            }
            if self.subscriber.stopped.load(Ordering::Acquire) {
                return None;
            }
            let timeout = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        return None;
                    }
                    Some(deadline - now)
                }
                None => None,
            };
            // An event queued between the take above and this wait has
            // already notified, so the wait returns at once.
            self.subscriber.wake.bind();
            self.subscriber.wake.wait(timeout);
        }
    }

    /// A handle that stops this subscription's waits.
    pub fn stopper(&self) -> AppliedStop {
        AppliedStop(Arc::clone(&self.subscriber))
    }

    /// A handle reading how far this subscription's events were offered.
    pub fn position(&self) -> AppliedPosition {
        AppliedPosition(Arc::clone(&self.subscriber))
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
            delivered: AtomicU64::new(0),
            sending: parking_lot::Mutex::new(()),
            tx,
            lost: AtomicBool::new(false),
            wake: Wake::default(),
            stopped: AtomicBool::new(false),
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

    /// Report that entry `index` applied `mutations` at `commit_ts`. Call
    /// after they are in the store.
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
                send(subscriber, |_| AppliedEvent::Replaced);
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
                send(subscriber, |_| AppliedEvent::Replaced);
            } else if !keys.is_empty() {
                send(subscriber, |seq| AppliedEvent::Keys {
                    seq,
                    index,
                    commit_ts,
                    keys,
                });
            }
        }
    }
}

/// Number the event `make` builds and queue it without waiting; a full
/// queue marks the subscription lost.
fn send(subscriber: &Subscriber, make: impl FnOnce(u64) -> AppliedEvent) {
    let _order = subscriber.sending.lock();
    // Under `sending`: one writer at a time, so a plain add cannot overflow
    // before 2^64 events.
    let seq = subscriber.delivered.load(Ordering::Relaxed) + 1;
    let result = subscriber.tx.try_send(make(seq));
    // Published after the event is queued: a reader that sees `seq` finds
    // the event queued or the subscription marked lost.
    match result {
        Ok(()) => {
            subscriber.delivered.store(seq, Ordering::Release);
            subscriber.wake.notify();
        }
        Err(TrySendError::Disconnected(_)) => {
            subscriber.delivered.store(seq, Ordering::Release);
        }
        Err(TrySendError::Full(_)) => {
            subscriber.lost.store(true, Ordering::Release);
            subscriber.delivered.store(seq, Ordering::Release);
            subscriber.wake.notify();
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
