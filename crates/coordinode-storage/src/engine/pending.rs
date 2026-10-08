//! Commits that have been admitted but whose writes have not landed yet.
//!
//! Validation reads committed state, and committed state is exactly what a
//! commit in flight is not yet part of. Between the moment one commit decides
//! that nothing has written its keys and the moment its own writes appear,
//! another commit can decide the same thing about the same keys, and both are
//! then applied: the later one silently replaces the earlier, which is the
//! lost update first-committer-wins exists to prevent.
//!
//! So a commit registers what it is about to write before it validates, and
//! validates against the registrations as well as against the state. The
//! registration is what closes the window: it is visible to everyone else from
//! the moment the timestamp is allocated, so there is no interval in which a
//! writer sees neither the effect nor the obligation.
//!
//! An overlap refuses whoever arrives second, in either direction. Letting the
//! earlier timestamp through because it lands first is wrong in the way that
//! is easy to argue oneself into: the commit already in flight read its value
//! at a snapshot that does not contain the arriving one, so when it applies
//! afterwards it overwrites it. Both orders lose an update, so both are
//! refused, and the loser retries against a state that now contains the
//! winner.
//!
//! The table is leader-local and in memory: a registration stands for a commit
//! this process is running, and a commit this process is no longer running
//! needs no registration. Durable protection for work that has reached a
//! promise is the prepared-participant mechanism, with its own lifetime.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::engine::partition::Partition;

/// One admitted commit's write scope, keyed by the timestamp it will land at.
#[derive(Debug)]
struct Admitted {
    commit_ts: u64,
    /// The keys this commit will write, excluding the commutative partitions
    /// whose concurrency story is the merge operator rather than exclusion.
    scope: Vec<(Partition, Vec<u8>)>,
    /// The keys this commit's writes are conditioned on without writing them.
    /// A writer of one of them in flight beside this commit would move it
    /// after the condition was checked; two commits that only condition on
    /// the same key leave it where both found it, so they do not collide.
    guards: Vec<(Partition, Vec<u8>)>,
}

/// The admitted commits and the keys reserved by commits waiting to be.
#[derive(Debug, Default)]
struct Table {
    /// Keyed by an identity of its own rather than by the commit timestamp.
    /// Two commits sharing a timestamp are not something the clock produces,
    /// but a table that assumes it silently drops one of them and releases
    /// the other's registration early, which is a lost update arriving
    /// through the mechanism that exists to prevent it. The assumption is
    /// cheaper to remove than to rely on.
    admitted: HashMap<u64, Admitted>,
    /// Commits waiting for the admitted ones they overlap to land, by the
    /// same identity. Their keys turn away newcomers, so a waiter is not
    /// overtaken forever by writers that keep arriving.
    reserved: HashMap<u64, Admitted>,
    /// Timestamps allocated for log entries that are not in the log yet, by
    /// the same identity: a commit applied locally and drained to the log
    /// later, or metadata proposed outside a commit. They do not hold back
    /// this node's own snapshots, only the closed bound.
    unlogged: HashMap<u64, u64>,
    /// The highest bound handed out for a log entry.
    last_closed: u64,
    /// The highest timestamp withdrawn from the table: a timestamp at or
    /// above `last_closed` is in the log, but no entry says so yet.
    max_released: u64,
    /// How many timestamps were withdrawn, for a waiter on the next one.
    withdrawals: u64,
}

/// The commits this leader has admitted but not yet applied.
#[derive(Debug)]
pub struct PendingCommits {
    inner: Mutex<Table>,
    next_id: std::sync::atomic::AtomicU64,
    /// Woken whenever a commit finishes, so a reader waiting for the view it
    /// asked for does not have to poll for it.
    finished: Condvar,
    max_in_flight: usize,
}

/// Why an admission was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Another commit in flight writes a key this one writes or conditions
    /// on, or conditions on a key this one writes.
    Overlap {
        /// The key both commits touch.
        partition: Partition,
        /// The key itself, for the message the caller has to produce.
        key: Vec<u8>,
        /// The timestamp the commit holding it will land at.
        holder_ts: u64,
    },
    /// A commit waiting for its turn has reserved a key this one writes or
    /// conditions on; retried, this one lands after it.
    Reserved {
        /// The key both commits touch.
        partition: Partition,
        /// The key itself, for the message the caller has to produce.
        key: Vec<u8>,
    },
    /// The table is at its bound. Admitting more would let the memory the
    /// guard holds grow with load, so the commit is refused while it can
    /// still be retried, rather than the bound being discovered later as
    /// exhaustion somewhere else.
    AtCapacity {
        /// The number of commits the table admits at once.
        limit: usize,
    },
}

impl PendingCommits {
    /// An empty table admitting at most `max_in_flight` commits at once.
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            inner: Mutex::new(Table::default()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            finished: Condvar::new(),
            max_in_flight,
        }
    }

    /// Wait until no commit at or below `ts` is still in flight, so a read at
    /// exactly `ts` sees a complete state.
    ///
    /// This is for a caller that named its timestamp: a time-travel read, a
    /// causal read with a lower bound, a cursor resuming at the cut it
    /// started on. Such a timestamp is preserved rather than lowered, so the
    /// only way to make it complete is to let the commits under it land.
    ///
    /// Returns the timestamp of a commit still in flight when the deadline
    /// passed. A caller answers that with an explicit timeout: waiting longer
    /// is its decision to make, and quietly reading an incomplete state or
    /// moving to another timestamp is not.
    pub fn await_complete_at(&self, ts: u64, timeout: Duration) -> Result<(), u64> {
        let deadline = Instant::now() + timeout;
        let mut table = self.inner.lock();
        loop {
            let Some(blocking) = table
                .admitted
                .values()
                .map(|a| a.commit_ts)
                .filter(|c| *c <= ts)
                .min()
            else {
                return Ok(());
            };
            if self.finished.wait_until(&mut table, deadline).timed_out() {
                return Err(blocking);
            }
        }
    }

    /// Admit a commit writing `scope` and conditioned on `guards`, or name
    /// the commit that already holds one of its keys.
    ///
    /// Any overlap of a written key with a key the other commit writes or
    /// conditions on refuses, whichever of the two has the lower timestamp.
    /// The commit already in flight read its inputs at a snapshot that cannot
    /// contain this one, so whether it lands before or after, one of the two
    /// is computed from a state the other replaced.
    ///
    /// The timestamp is allocated and the keys registered without a gap
    /// between the two.
    ///
    /// Two separate steps leave an interval in which the clock has already
    /// passed a number whose commit is not registered anywhere, and a snapshot
    /// taken in that interval covers a commit nobody can be told about. The
    /// allocation happens under the same lock the registration and the
    /// snapshot floor take, so a reader sees either the registration or a
    /// clock that has not reached it.
    pub fn admit_allocated<'p>(
        &'p self,
        allocate: impl FnOnce() -> u64,
        scope: Vec<(Partition, Vec<u8>)>,
        guards: Vec<(Partition, Vec<u8>)>,
    ) -> Result<(u64, Admission<'p>), Refusal> {
        let mut table = self.inner.lock();
        let candidate = Admitted {
            commit_ts: 0,
            scope,
            guards,
        };
        Self::refuse_reserved(&table, &candidate)?;
        if let Some(refusal) = Self::overlap(&table, &candidate) {
            return Err(refusal);
        }
        self.insert_admitted(&mut table, allocate, candidate)
    }

    /// [`Self::admit_allocated`] for a commit that waits its turn instead of
    /// being refused by commits already admitted on its keys: a catalog
    /// change, which every writer of the catalogued object conditions on and
    /// which would otherwise lose to whichever of them came first.
    ///
    /// While it waits its keys are reserved, so a commit arriving later on
    /// them is refused and retried after it rather than overtaking it. The
    /// timestamp is taken only once the overlapping commits have landed, so
    /// this commit is ordered after them. A wait past `wait` gives up with
    /// the overlap still standing; a reservation by another waiter is
    /// refused at once, as two waiters on one key would only queue behind
    /// each other's retries.
    ///
    /// # Errors
    ///
    /// [`Refusal::Reserved`] when another waiter holds one of its keys,
    /// [`Refusal::Overlap`] when an admitted commit still holds one at the
    /// deadline, [`Refusal::AtCapacity`] at the table's bound.
    pub fn admit_allocated_waiting<'p>(
        &'p self,
        allocate: impl FnOnce() -> u64,
        scope: Vec<(Partition, Vec<u8>)>,
        guards: Vec<(Partition, Vec<u8>)>,
        wait: Duration,
    ) -> Result<(u64, Admission<'p>), Refusal> {
        let deadline = Instant::now() + wait;
        let mut table = self.inner.lock();
        let candidate = Admitted {
            commit_ts: 0,
            scope,
            guards,
        };
        Self::refuse_reserved(&table, &candidate)?;
        let mut reservation = None;
        while Self::overlap(&table, &candidate).is_some() {
            if reservation.is_none() {
                let id = self
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                table.reserved.insert(
                    id,
                    Admitted {
                        commit_ts: 0,
                        scope: candidate.scope.clone(),
                        guards: candidate.guards.clone(),
                    },
                );
                reservation = Some(id);
            }
            if self.finished.wait_until(&mut table, deadline).timed_out() {
                if let Some(refusal) = Self::overlap(&table, &candidate) {
                    if let Some(id) = reservation {
                        table.reserved.remove(&id);
                    }
                    return Err(refusal);
                }
            }
        }
        if let Some(id) = reservation {
            table.reserved.remove(&id);
        }
        self.insert_admitted(&mut table, allocate, candidate)
    }

    /// Register `candidate` under a freshly allocated timestamp.
    fn insert_admitted<'p>(
        &'p self,
        table: &mut Table,
        allocate: impl FnOnce() -> u64,
        mut candidate: Admitted,
    ) -> Result<(u64, Admission<'p>), Refusal> {
        // Accounted before the timestamp is taken: a number allocated and then
        // refused is a hole in the clock nobody closes.
        if table.admitted.len() >= self.max_in_flight {
            return Err(Refusal::AtCapacity {
                limit: self.max_in_flight,
            });
        }
        candidate.commit_ts = allocate();
        let commit_ts = candidate.commit_ts;
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.admitted.insert(id, candidate);
        Ok((commit_ts, Admission { pending: self, id }))
    }

    /// The first admitted commit `candidate` collides with: a key it writes
    /// that the other writes or conditions on, or a key it conditions on that
    /// the other writes.
    fn overlap(table: &Table, candidate: &Admitted) -> Option<Refusal> {
        table.admitted.values().find_map(|other| {
            Self::colliding_key(candidate, other).map(|(partition, key)| Refusal::Overlap {
                partition,
                key,
                holder_ts: other.commit_ts,
            })
        })
    }

    /// Refuse `candidate` when a waiter reserved a key it collides with.
    fn refuse_reserved(table: &Table, candidate: &Admitted) -> Result<(), Refusal> {
        match table
            .reserved
            .values()
            .find_map(|waiter| Self::colliding_key(candidate, waiter))
        {
            Some((partition, key)) => Err(Refusal::Reserved { partition, key }),
            None => Ok(()),
        }
    }

    /// A key `candidate` writes that `other` writes or conditions on, or a
    /// key `candidate` conditions on that `other` writes. The relation is
    /// symmetric: a key one side conditions on and the other writes collides
    /// whichever side is asked first.
    fn colliding_key(candidate: &Admitted, other: &Admitted) -> Option<(Partition, Vec<u8>)> {
        let holds = |keys: &[(Partition, Vec<u8>)], partition: &Partition, key: &[u8]| {
            keys.iter().any(|(p, k)| p == partition && k == key)
        };
        candidate
            .scope
            .iter()
            .find(|(p, k)| holds(&other.scope, p, k) || holds(&other.guards, p, k))
            .or_else(|| {
                candidate
                    .guards
                    .iter()
                    .find(|(p, k)| holds(&other.scope, p, k))
            })
            .cloned()
    }

    /// The highest snapshot that covers no commit still in flight, given the
    /// clock's current value.
    ///
    /// The clock is read under the same lock that admits a commit, so the
    /// answer cannot fall between a timestamp being allocated and its scope
    /// being registered.
    pub fn snapshot_floor(&self, read_clock: impl FnOnce() -> u64) -> u64 {
        let table = self.inner.lock();
        let latest = read_clock();
        Self::floor_of(&table, latest)
    }

    /// The freshest complete snapshot, waiting briefly for the commits in
    /// flight to land rather than stepping back behind them.
    ///
    /// Stepping back is correct and was the first answer, but it is paid for
    /// by everyone: the floor is the oldest commit in flight anywhere on the
    /// node, so under any concurrency a transaction is handed a view older
    /// than its own last commit. It then reads its own stale value and
    /// conflicts with itself. Measured with eight writers on keys they did
    /// not share: 59% of attempts refused, none of them for a real overlap.
    ///
    /// A commit holds its registration only from validation to local apply,
    /// which is microseconds, so waiting for that window to pass costs less
    /// than reading behind it. The wait is bounded, and its expiry falls back
    /// to the floor: an older complete view, never an incomplete fresh one.
    pub fn complete_snapshot(&self, read_clock: impl Fn() -> u64, wait: Duration) -> u64 {
        let deadline = Instant::now() + wait;
        let mut table = self.inner.lock();
        loop {
            let latest = read_clock();
            if table.admitted.is_empty() {
                return latest;
            }
            let floor = Self::floor_of(&table, latest);
            if floor == latest {
                return latest;
            }
            if self.finished.wait_until(&mut table, deadline).timed_out() {
                let latest = read_clock();
                return Self::floor_of(&table, latest);
            }
        }
    }

    fn floor_of(table: &Table, latest: u64) -> u64 {
        table
            .admitted
            .values()
            .map(|a| a.commit_ts)
            .filter(|ts| *ts <= latest)
            .min()
            .unwrap_or(latest)
    }

    /// How many commits are admitted and not yet applied.
    pub fn in_flight(&self) -> usize {
        self.inner.lock().admitted.len()
    }

    /// Allocate a timestamp for a log entry and hold it as not yet logged
    /// until the returned [`Unlogged`] is dropped, which the caller does once
    /// the entry is handed to the log (or will never be). Allocation and
    /// registration take one lock, so no bound computed in between can pass
    /// over the timestamp.
    pub fn obligate(self: &Arc<Self>, allocate: impl FnOnce() -> u64) -> (u64, Unlogged) {
        let mut table = self.inner.lock();
        let commit_ts = allocate();
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.unlogged.insert(id, commit_ts);
        (
            commit_ts,
            Unlogged {
                pending: Arc::clone(self),
                id,
            },
        )
    }

    /// Hold `commit_ts` as not yet logged past the hold the caller has on it
    /// now (an [`Admission`] or another [`Unlogged`]): a commit applied
    /// locally whose entry reaches the log later, or an entry handed to the
    /// log from another task. Taking it while that hold stands leaves no
    /// moment in which a bound can pass over it.
    pub fn hold(self: &Arc<Self>, commit_ts: u64) -> Unlogged {
        let mut table = self.inner.lock();
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.unlogged.insert(id, commit_ts);
        Unlogged {
            pending: Arc::clone(self),
            id,
        }
    }

    /// The bound a log entry handed to the log now carries: every timestamp
    /// below it is in the log before that entry, or never will be.
    ///
    /// A timestamp still admitted or not yet logged is at or above it. Any
    /// other allocated timestamp was withdrawn after its entry was handed to
    /// the log, which takes entries in the order they are handed over, or
    /// was refused before it was. Read under the lock that allocates, so the
    /// clock cannot pass a timestamp between its allocation and its
    /// registration.
    pub fn stamp_closed_below(&self, read_clock: impl FnOnce() -> u64) -> u64 {
        let mut table = self.inner.lock();
        let bound = Self::closed_below_of(&table, read_clock());
        table.last_closed = table.last_closed.max(bound);
        bound
    }

    /// The bound a closing entry would carry now, when it covers a
    /// timestamp withdrawn since the last bound was handed out; `None` when
    /// it would cover nothing new. A closing entry carries no timestamp of
    /// its own, so stamping it does not make another one due.
    pub fn closure_due(&self, read_clock: impl FnOnce() -> u64) -> Option<u64> {
        let table = self.inner.lock();
        let bound = Self::closed_below_of(&table, read_clock());
        (table.max_released >= table.last_closed && bound > table.last_closed).then_some(bound)
    }

    /// Wait until more than `seen` withdrawals have happened, or until
    /// `timeout`; the count then. A closing entry can only become due
    /// through a withdrawal.
    pub fn await_withdrawal(&self, seen: u64, timeout: Duration) -> u64 {
        let deadline = Instant::now() + timeout;
        let mut table = self.inner.lock();
        while table.withdrawals <= seen {
            if self.finished.wait_until(&mut table, deadline).timed_out() {
                break;
            }
        }
        table.withdrawals
    }

    fn closed_below_of(table: &Table, latest: u64) -> u64 {
        let held = table
            .admitted
            .values()
            .map(|a| a.commit_ts)
            .chain(table.unlogged.values().copied())
            .min();
        // A clock in microseconds since the epoch is far below u64::MAX, so
        // the bound one past it cannot overflow.
        held.unwrap_or(latest + 1)
    }

    fn withdraw(&self, id: u64) {
        let mut table = self.inner.lock();
        if let Some(admitted) = table.admitted.remove(&id) {
            table.max_released = table.max_released.max(admitted.commit_ts);
            table.withdrawals += 1;
        }
        drop(table);
        // Every waiter is woken rather than one: they are waiting on
        // different timestamps, and the one this commit unblocks is not
        // necessarily the one a single wake would reach.
        self.finished.notify_all();
    }

    fn withdraw_unlogged(&self, id: u64) {
        let mut table = self.inner.lock();
        if let Some(commit_ts) = table.unlogged.remove(&id) {
            table.max_released = table.max_released.max(commit_ts);
            table.withdrawals += 1;
        }
        drop(table);
        self.finished.notify_all();
    }
}

/// A timestamp allocated for a log entry that is not in the log yet.
/// Withdraws it on drop: hand the entry to the log first.
#[derive(Debug)]
pub struct Unlogged {
    pending: Arc<PendingCommits>,
    id: u64,
}

impl Drop for Unlogged {
    fn drop(&mut self) {
        self.pending.withdraw_unlogged(self.id);
    }
}

impl coordinode_core::txn::drain::DrainHold for Unlogged {}

/// An admitted commit's registration. Withdraws it on drop, so a commit that
/// fails after admission holds nothing, and one that succeeds holds nothing
/// once its writes are the state everyone else validates against.
#[derive(Debug)]
pub struct Admission<'p> {
    pending: &'p PendingCommits,
    id: u64,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.pending.withdraw(self.id);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
