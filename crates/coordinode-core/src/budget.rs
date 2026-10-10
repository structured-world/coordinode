//! The execution budget of one query: query-owned memory, examined work,
//! a deadline and cancellation, shared by every operator, access path and
//! alternative plan the query runs.
//!
//! Memory is reserved before the allocation or growth it pays for, and the
//! reservation lives in a [`MemoryCharge`] that returns it when the memory is
//! no longer owned. A reservation that would pass the limit is refused and
//! leaves the budget as it was. Work is counted as it is done, including work
//! on candidates that are then rejected; the deadline and cancellation are
//! checked as work accrues, every [`CHECK_EVERY`] units, so a loop over cheap
//! items does not read the clock per item.
//!
//! The budget is bound when a request is admitted and is never replaced while
//! it runs: an alternative path after damage is found, a retry or a parallel
//! operator spends from the same one.

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Default `query_memory_limit`: 256 MiB.
pub const DEFAULT_QUERY_MEMORY_LIMIT: u64 = 256 << 20;

/// The administrative ceiling a session cannot raise `query_memory_limit`
/// past: 4 GiB.
pub const QUERY_MEMORY_CEILING: u64 = 4 << 30;

/// Units of work between two checks of the deadline and of cancellation.
pub const CHECK_EVERY: u64 = 1024;

/// A monotonic clock in nanoseconds from an arbitrary origin, supplied by the
/// caller (the engine has no clock of its own below `std`).
pub type MonotonicNanos = fn() -> u64;

/// Why a query was stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetStop {
    /// Reserving `requested` bytes on top of `used` would pass `limit`.
    Memory {
        /// The reservation refused.
        requested: u64,
        /// Bytes reserved when it was refused.
        used: u64,
        /// The query's memory limit.
        limit: u64,
    },
    /// The query ran past its deadline.
    Deadline,
    /// The query was cancelled.
    Cancelled,
}

impl core::fmt::Display for BudgetStop {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Memory {
                requested,
                used,
                limit,
            } => write!(
                f,
                "the query needs {requested} more bytes with {used} of its {limit}-byte memory \
                 limit in use"
            ),
            Self::Deadline => f.write_str("the query ran past its deadline"),
            Self::Cancelled => f.write_str("the query was cancelled"),
        }
    }
}

impl core::error::Error for BudgetStop {}

/// The budget of one query. Shared by reference: every method takes `&self`
/// and is safe to call from the operators of one query running in parallel.
#[derive(Debug)]
pub struct QueryBudget {
    limit: u64,
    used: AtomicU64,
    peak: AtomicU64,
    work: AtomicU64,
    /// Work at the last deadline and cancellation check.
    checked_at: AtomicU64,
    cancelled: AtomicBool,
    /// The deadline on `clock`, or `None` without one.
    deadline: Option<(u64, MonotonicNanos)>,
}

impl QueryBudget {
    /// A budget of `memory_limit` bytes with no deadline.
    pub fn new(memory_limit: u64) -> Self {
        Self {
            limit: memory_limit,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            work: AtomicU64::new(0),
            checked_at: AtomicU64::new(0),
            cancelled: AtomicBool::new(false),
            deadline: None,
        }
    }

    /// The same budget ending at `deadline_nanos` on `clock`.
    #[must_use]
    pub fn with_deadline(mut self, deadline_nanos: u64, clock: MonotonicNanos) -> Self {
        self.deadline = Some((deadline_nanos, clock));
        self
    }

    /// The query's memory limit in bytes.
    pub fn memory_limit(&self) -> u64 {
        self.limit
    }

    /// Bytes reserved now.
    pub fn memory_used(&self) -> u64 {
        self.used.load(Ordering::Acquire)
    }

    /// The most bytes reserved at once so far.
    pub fn memory_peak(&self) -> u64 {
        self.peak.load(Ordering::Acquire)
    }

    /// Units of work done so far.
    pub fn work_done(&self) -> u64 {
        self.work.load(Ordering::Acquire)
    }

    /// Stop the query: every later check refuses with
    /// [`BudgetStop::Cancelled`].
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Reserve `bytes` before allocating them. The returned charge returns
    /// them when dropped, so it is kept as long as the memory is owned.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Memory`] when the reservation would pass the limit; the
    /// budget is left as it was.
    pub fn reserve(&self, bytes: u64) -> Result<MemoryCharge<'_>, BudgetStop> {
        self.take(bytes)?;
        Ok(MemoryCharge {
            budget: self,
            bytes,
        })
    }

    /// A charge of no bytes, to grow as memory is kept.
    pub fn empty_charge(&self) -> MemoryCharge<'_> {
        MemoryCharge {
            budget: self,
            bytes: 0,
        }
    }

    /// Count `units` of work, then check the deadline and cancellation when
    /// [`CHECK_EVERY`] units have accrued since the last check.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Deadline`] or [`BudgetStop::Cancelled`].
    pub fn work(&self, units: u64) -> Result<(), BudgetStop> {
        let done = self
            .work
            .fetch_add(units, Ordering::AcqRel)
            .wrapping_add(units);
        let last = self.checked_at.load(Ordering::Acquire);
        if done.wrapping_sub(last) < CHECK_EVERY {
            return Ok(());
        }
        self.checked_at.store(done, Ordering::Release);
        self.check()
    }

    /// Check the deadline and cancellation now.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Deadline`] or [`BudgetStop::Cancelled`].
    pub fn check(&self) -> Result<(), BudgetStop> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(BudgetStop::Cancelled);
        }
        if let Some((deadline, clock)) = self.deadline {
            if clock() >= deadline {
                return Err(BudgetStop::Deadline);
            }
        }
        Ok(())
    }

    fn take(&self, bytes: u64) -> Result<(), BudgetStop> {
        // Compare-and-swap, not add-then-undo: a refused reservation never
        // shows in `used`, so a concurrent one is not refused for it, and a
        // request near u64::MAX is refused rather than wrapping the counter.
        let mut before = self.used.load(Ordering::Acquire);
        loop {
            let after = match before.checked_add(bytes) {
                Some(after) if after <= self.limit => after,
                _ => {
                    return Err(BudgetStop::Memory {
                        requested: bytes,
                        used: before,
                        limit: self.limit,
                    });
                }
            };
            match self.used.compare_exchange_weak(
                before,
                after,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(after, Ordering::AcqRel);
                    return Ok(());
                }
                Err(now) => before = now,
            }
        }
    }

    fn give(&self, bytes: u64) {
        let before = self.used.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(
            before >= bytes,
            "returned {bytes} bytes with {before} reserved"
        );
    }
}

/// Memory reserved from a [`QueryBudget`], returned when dropped.
#[derive(Debug)]
pub struct MemoryCharge<'b> {
    budget: &'b QueryBudget,
    bytes: u64,
}

impl MemoryCharge<'_> {
    /// Bytes this charge holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Reserve `bytes` more under this charge, before the growth they pay
    /// for.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Memory`]; the charge keeps what it held.
    pub fn grow(&mut self, bytes: u64) -> Result<(), BudgetStop> {
        self.budget.take(bytes)?;
        // Within the limit the budget just admitted, so it cannot overflow.
        self.bytes += bytes;
        Ok(())
    }

    /// Return `bytes` of this charge, at most what it holds, once that
    /// memory is no longer owned.
    pub fn shrink(&mut self, bytes: u64) {
        let returned = bytes.min(self.bytes);
        self.budget.give(returned);
        self.bytes -= returned;
    }

    /// Keep this charge's bytes reserved for the rest of the query: for
    /// memory handed to the query's later stages (rows an operator returns),
    /// whose release no single owner sees. A query's budget ends with the
    /// query, so the reservation ends with it.
    pub fn keep_until_query_ends(mut self) {
        self.bytes = 0;
    }
}

impl Drop for MemoryCharge<'_> {
    fn drop(&mut self) {
        if self.bytes > 0 {
            self.budget.give(self.bytes);
        }
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
