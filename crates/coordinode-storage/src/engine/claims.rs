//! The leader's table of reserved invariant claims.
//!
//! An attempt states what its result depends on; this is where those claims
//! meet the claims of everyone else in flight. Two attempts that hold
//! incompatible claims cannot both be admitted, and the one that arrives
//! second is refused here rather than at the far end of its work.
//!
//! This is the early, advisory half of the two-stage check: it sees every
//! attempt this leader is running and nothing else, so it catches the common
//! case cheaply and proves nothing on its own. What makes a refusal
//! authoritative is the validation against committed state inside the commit
//! guard's protected transition; the table's job is to stop an attempt from
//! doing more work when the answer is already known.
//!
//! The table is leader-local and in memory on purpose. It holds no durable
//! promise: a reservation that disappears with the process is a reservation
//! for an attempt that disappeared with it. Durable protection for work that
//! has reached a promise is a different mechanism with a different lifetime.

pub mod evaluate;

use std::collections::HashMap;

use coordinode_core::txn::invariant::{Claim, ClaimSet};
use parking_lot::Mutex;

/// Why a reservation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClaimRefusal {
    /// Another attempt in flight holds a claim this one cannot coexist with.
    /// Both claims are named so the caller can say which condition refused
    /// the attempt rather than that something did.
    Incompatible {
        /// The claim this attempt wanted.
        wanted: Claim,
        /// The claim already held that excludes it.
        held: Claim,
        /// The attempt holding it.
        holder: u64,
    },
    /// The table is at its bound. Admitting more would let the guard's
    /// memory grow with load, so the attempt is refused while it can still
    /// be retried cheaply.
    AtCapacity {
        /// The number of claims the table admits.
        limit: usize,
    },
}

/// Reserved claims of the attempts this leader is running.
///
/// Bounded by construction: the limit is on claims rather than on attempts,
/// because one attempt with a large scan holds many and one with a single
/// edge holds few, and what has to stay bounded is the memory.
pub struct ClaimRegistry {
    inner: Mutex<Registry>,
    max_claims: usize,
}

#[derive(Default)]
struct Registry {
    by_attempt: HashMap<u64, ClaimSet>,
    total_claims: usize,
}

impl ClaimRegistry {
    /// A registry admitting at most `max_claims` claims across all attempts.
    pub fn new(max_claims: usize) -> Self {
        Self {
            inner: Mutex::new(Registry::default()),
            max_claims,
        }
    }

    /// Reserve `claims` for `attempt`, or say what refused it.
    ///
    /// An attempt that reserves twice replaces its own entry rather than
    /// conflicting with itself: a statement chain states its conditions as it
    /// goes, and the set grows with the attempt.
    pub fn reserve(&self, attempt: u64, claims: &ClaimSet) -> Result<(), Box<ClaimRefusal>> {
        if claims.is_empty() {
            return Ok(());
        }
        let mut reg = self.inner.lock();

        let held_by_this = reg.by_attempt.get(&attempt).map_or(0, ClaimSet::len);
        let would_total = reg.total_claims - held_by_this + claims.len();
        if would_total > self.max_claims {
            return Err(Box::new(ClaimRefusal::AtCapacity {
                limit: self.max_claims,
            }));
        }

        for (holder, other) in reg.by_attempt.iter() {
            if *holder == attempt {
                continue;
            }
            if let Some((wanted, held)) = claims.first_conflict(other) {
                return Err(Box::new(ClaimRefusal::Incompatible {
                    wanted: wanted.clone(),
                    held: held.clone(),
                    holder: *holder,
                }));
            }
        }

        reg.total_claims = would_total;
        reg.by_attempt.insert(attempt, claims.clone());
        Ok(())
    }

    /// Release an attempt's claims. Called when the attempt ends, whichever
    /// way it ended: a refused attempt holds nothing, and a committed one's
    /// durable protection, where it needs any, is not this table's.
    pub fn release(&self, attempt: u64) {
        let mut reg = self.inner.lock();
        if let Some(set) = reg.by_attempt.remove(&attempt) {
            reg.total_claims -= set.len();
        }
    }

    /// Reserve `claims` for `attempt` and hold them until the returned guard
    /// is dropped.
    ///
    /// The reservation protects the window between deciding that a condition
    /// holds and applying the writes built on it, and that window is exactly
    /// the lifetime of the guard. Tying it to a value the compiler drops on
    /// every path, rather than to a call at each exit, is what keeps a
    /// refusal, an error and a success from each needing their own release,
    /// and one forgotten path from filling the table for good.
    pub fn reserve_held<'r>(
        &'r self,
        attempt: u64,
        claims: &ClaimSet,
    ) -> Result<Reservation<'r>, Box<ClaimRefusal>> {
        self.reserve(attempt, claims)?;
        Ok(Reservation {
            registry: self,
            attempt,
        })
    }

    /// Claims currently reserved, across all attempts. The guard budget is
    /// observed through this.
    pub fn reserved_claims(&self) -> usize {
        self.inner.lock().total_claims
    }

    /// Attempts currently holding claims.
    pub fn attempts(&self) -> usize {
        self.inner.lock().by_attempt.len()
    }
}

/// One attempt's live reservation. Releases it on drop.
pub struct Reservation<'r> {
    registry: &'r ClaimRegistry,
    attempt: u64,
}

impl Drop for Reservation<'_> {
    fn drop(&mut self) {
        self.registry.release(self.attempt);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
