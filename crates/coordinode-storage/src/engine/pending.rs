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

use parking_lot::Mutex;

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
    inner: Mutex<HashMap<u64, Admitted>>,
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
            max_in_flight,
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

        table.insert(commit_ts, Admitted { commit_ts, scope });
        Ok((
            commit_ts,
            Admission {
                pending: self,
                commit_ts,
            },
        ))
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

    fn withdraw(&self, commit_ts: u64) {
        self.inner.lock().remove(&commit_ts);
    }
}

/// An admitted commit's registration. Withdraws it on drop, so a commit that
/// fails after admission holds nothing, and one that succeeds holds nothing
/// once its writes are the state everyone else validates against.
#[derive(Debug)]
pub struct Admission<'p> {
    pending: &'p PendingCommits,
    commit_ts: u64,
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.pending.withdraw(self.commit_ts);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
