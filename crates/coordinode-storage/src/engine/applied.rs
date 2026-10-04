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
//!
//! A retained subscription keeps each event after handing it out, until the
//! consumer releases it: readers then see which keys the consumer has not
//! folded yet ([`AppliedPosition::pending`]) and answer for those themselves
//! instead of waiting for it.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
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
        /// The keys the entry wrote in the subscribed partition, shared with
        /// the copy a retained subscription keeps.
        keys: Arc<[Vec<u8>]>,
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
    /// Whether events stay after they are handed out.
    retained: bool,
    /// Most events the queue holds, handed out and retained ones included.
    capacity: usize,
    /// The number of the last event offered (queued or dropped). Written
    /// under `queue`, so numbers enter the queue in order.
    delivered: AtomicU64,
    /// Numbering an event and queueing it happen under this lock: local
    /// commits apply concurrently, and a later number queued ahead of an
    /// earlier one would let the consumer claim the earlier one folded.
    // no-std: spin::Mutex; held for one counter bump and one queue push.
    queue: parking_lot::Mutex<Queue>,
    /// Notified on every event queued or lost, so a waiting subscriber sleeps
    /// until there is something to take.
    wake: Wake,
    /// Set by [`AppliedStop::stop`]; a waiting subscriber returns `None`.
    stopped: AtomicBool,
}

#[derive(Debug, Default)]
struct Queue {
    /// Numbered events, oldest first: the ones not handed out yet and, on a
    /// retained subscription, the ones handed out and not released.
    events: VecDeque<(u64, AppliedEvent)>,
    /// How many events at the front were handed out.
    taken: usize,
    /// An event could not be queued: handed out as
    /// [`AppliedEvent::Replaced`] before the events queued after it.
    lost: bool,
    /// The last event whose keys are not known: one dropped, or a
    /// replacement.
    unknown_through: u64,
    /// The last event the consumer released.
    released: u64,
}

/// An open subscription. Closes when dropped.
#[derive(Debug)]
pub struct AppliedSubscription {
    subscriber: Arc<Subscriber>,
    feed: Weak<AppliedFeed>,
}

/// How far the events of one subscription have been offered, readable from
/// any thread.
#[derive(Debug, Clone)]
pub struct AppliedPosition(Arc<Subscriber>);

/// The keys a retained subscription's consumer has not released yet.
#[derive(Debug, Clone)]
pub enum PendingKeys {
    /// The keys of every unreleased event, one slice per event; every entry
    /// applied before [`AppliedPosition::pending`] was called is either
    /// released or here.
    Known(Vec<Arc<[Vec<u8>]>>),
    /// An unreleased event's keys are not known (it was dropped, or the
    /// partition was replaced): any key may have changed.
    Unknown,
}

impl AppliedPosition {
    /// The number of the last event offered to the subscription. Every entry
    /// whose event is numbered at or below it is already in the store, so a
    /// consumer that has folded up to this number covers every entry applied
    /// before this call.
    pub fn delivered(&self) -> u64 {
        self.0.delivered.load(Ordering::Acquire)
    }

    /// The keys written by the events the consumer has not released: what a
    /// reader evaluates itself, since the consumer's state may not hold them
    /// yet. Only a retained subscription keeps events after handing them out.
    pub fn pending(&self) -> PendingKeys {
        let queue = self.0.queue.lock();
        if queue.unknown_through > queue.released {
            return PendingKeys::Unknown;
        }
        PendingKeys::Known(
            queue
                .events
                .iter()
                .filter_map(|(_, event)| match event {
                    AppliedEvent::Keys { keys, .. } => Some(Arc::clone(keys)),
                    AppliedEvent::Replaced => None,
                })
                .collect(),
        )
    }

    /// Record that the consumer's state holds every event numbered up to
    /// `seq`: the ones handed out are dropped, and a gap at or below `seq` is
    /// known again.
    pub fn release(&self, seq: u64) {
        let mut queue = self.0.queue.lock();
        queue.released = queue.released.max(seq);
        // Only handed-out events: one still queued is the consumer's to take,
        // and taking it twice costs a repeated fold, never a missed one.
        while queue.taken > 0 && queue.events.front().is_some_and(|(s, _)| *s <= seq) {
            queue.events.pop_front();
            queue.taken -= 1;
        }
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
        let mut queue = self.subscriber.queue.lock();
        if queue.lost {
            queue.lost = false;
            return Some(AppliedEvent::Replaced);
        }
        if !self.subscriber.retained {
            return queue.events.pop_front().map(|(_, event)| event);
        }
        let index = queue.taken;
        let event = queue.events.get(index).map(|(_, event)| event.clone())?;
        queue.taken += 1;
        Some(event)
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
    /// at most `capacity` of them; `retained` keeps each event after it is
    /// handed out until [`AppliedPosition::release`] passes it.
    pub(crate) fn subscribe(
        self: &Arc<Self>,
        partition: Partition,
        capacity: usize,
        retained: bool,
    ) -> AppliedSubscription {
        let subscriber = Arc::new(Subscriber {
            partition,
            retained,
            capacity: capacity.max(1),
            delivered: AtomicU64::new(0),
            queue: parking_lot::Mutex::new(Queue::default()),
            wake: Wake::default(),
            stopped: AtomicBool::new(false),
        });
        {
            let mut subscribers = self.subscribers.write();
            subscribers.push(Arc::clone(&subscriber));
            self.open.store(subscribers.len(), Ordering::Release);
        }
        AppliedSubscription {
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
                let keys: Arc<[Vec<u8>]> = keys.into();
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
    let mut queue = subscriber.queue.lock();
    // Under `queue`: one writer at a time, so a plain add cannot overflow
    // before 2^64 events.
    let seq = subscriber.delivered.load(Ordering::Relaxed) + 1;
    let event = make(seq);
    if matches!(event, AppliedEvent::Replaced) {
        queue.unknown_through = seq;
    }
    if queue.events.len() < subscriber.capacity {
        queue.events.push_back((seq, event));
    } else {
        queue.lost = true;
        queue.unknown_through = seq;
    }
    // Published with the event queued or the subscription marked lost, so a
    // reader that sees `seq` finds one or the other.
    subscriber.delivered.store(seq, Ordering::Release);
    drop(queue);
    subscriber.wake.notify();
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
