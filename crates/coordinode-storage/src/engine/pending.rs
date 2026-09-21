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
}

/// The commits this leader has admitted but not yet applied.
#[derive(Debug)]
pub struct PendingCommits {
    /// Keyed by an identity of its own rather than by the commit timestamp.
    /// Two commits sharing a timestamp are not something the clock produces,
    /// but a table that assumes it silently drops one of them and releases
    /// the other's registration early, which is a lost update arriving
    /// through the mechanism that exists to prevent it. The assumption is
    /// cheaper to remove than to rely on.
    inner: Mutex<HashMap<u64, Admitted>>,
    next_id: std::sync::atomic::AtomicU64,
    /// Woken whenever a commit finishes, so a reader waiting for the view it
    /// asked for does not have to poll for it.
    finished: Condvar,
    max_in_flight: usize,
}

/// Why an admission was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// Another commit in flight writes a key this one writes.
    Overlap {
        /// The key both commits write.
        partition: Partition,
        /// The key itself, for the message the caller has to produce.
        key: Vec<u8>,
        /// The timestamp the commit holding it will land at.
        holder_ts: u64,
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
            inner: Mutex::new(HashMap::new()),
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

    /// Admit `commit_ts` with `scope`, or name the commit that already holds
    /// one of its keys.
    ///
    /// Any overlap refuses, whichever of the two has the lower timestamp. The
    /// commit already in flight read its inputs at a snapshot that cannot
    /// contain this one, so whether it lands before or after, one of the two
    /// writes is computed from a state the other replaced.
    /// Allocate a commit timestamp and register its scope without a gap
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
    ) -> Result<(u64, Admission<'p>), Refusal> {
        let mut table = self.inner.lock();

        // Accounted before the timestamp is taken: a number allocated and then
        // refused is a hole in the clock nobody closes.
        if table.len() >= self.max_in_flight {
            return Err(Refusal::AtCapacity {
                limit: self.max_in_flight,
            });
        }

        let commit_ts = allocate();

        for other in table.values() {
            for (partition, key) in &scope {
                if other.scope.iter().any(|(p, k)| p == partition && k == key) {
                    return Err(Refusal::Overlap {
                        partition: *partition,
                        key: key.clone(),
                        holder_ts: other.commit_ts,
                    });
                }
            }
        }

        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.insert(id, Admitted { commit_ts, scope });
        Ok((commit_ts, Admission { pending: self, id }))
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
            if table.is_empty() {
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

    fn floor_of(table: &HashMap<u64, Admitted>, latest: u64) -> u64 {
        table
            .values()
            .map(|a| a.commit_ts)
            .filter(|ts| *ts <= latest)
            .min()
            .unwrap_or(latest)
    }

    /// How many commits are admitted and not yet applied.
    pub fn in_flight(&self) -> usize {
        self.inner.lock().len()
    }

    fn withdraw(&self, id: u64) {
        self.inner.lock().remove(&id);
        // Every waiter is woken rather than one: they are waiting on
        // different timestamps, and the one this commit unblocks is not
        // necessarily the one a single wake would reach.
        self.finished.notify_all();
    }
}

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
