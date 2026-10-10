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
//! operator spends from the same one. An evaluation that ends before the
//! query does (a subquery run once per row) spends through a [part] of it,
//! which returns what that evaluation kept when it ends.
//!
//! [part]: QueryBudget::part

use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The switch that cancels a query, held by whoever may cancel it (a
/// session's Cancel, a client that went away) and by the query's budget,
/// which refuses every later check once it is set.
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<CancelState>);

#[derive(Debug, Default)]
struct CancelState {
    thrown: AtomicBool,
    /// The task waiting in [`CancelFlag::thrown`], woken when it is thrown.
    waiter: atomic_waker::AtomicWaker,
}

impl CancelFlag {
    /// A switch not yet thrown.
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancel the query.
    pub fn cancel(&self) {
        self.0.thrown.store(true, Ordering::Release);
        self.0.waiter.wake();
    }

    /// Whether the query was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.0.thrown.load(Ordering::Acquire)
    }

    /// Resolves once the switch is thrown: what a task waiting on work done
    /// elsewhere (a statement passed to another member) races its answer
    /// against. One task waits on a switch at a time; a second one polling
    /// it takes the first one's place.
    pub fn thrown(&self) -> Thrown<'_> {
        Thrown(self)
    }
}

/// The future [`CancelFlag::thrown`] returns.
#[derive(Debug)]
#[must_use = "a future does nothing unless polled"]
pub struct Thrown<'f>(&'f CancelFlag);

impl core::future::Future for Thrown<'_> {
    type Output = ();

    fn poll(
        self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        let flag = self.0;
        if flag.is_cancelled() {
            return core::task::Poll::Ready(());
        }
        flag.0.waiter.register(cx.waker());
        // Read again after registering: a throw between the first read and
        // the registration woke no one.
        if flag.is_cancelled() {
            core::task::Poll::Ready(())
        } else {
            core::task::Poll::Pending
        }
    }
}

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
    cancelled: CancelFlag,
    /// The deadline on `clock`, or `None` without one.
    deadline: Option<(u64, MonotonicNanos)>,
    /// The budget this one is a part of: every reservation is made there
    /// too, work is counted and checked there, and what this part still
    /// holds returns there when it is dropped.
    whole: Option<Arc<QueryBudget>>,
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
            cancelled: CancelFlag::new(),
            deadline: None,
            whole: None,
        }
    }

    /// A part of this budget for an evaluation that ends before the query
    /// does, such as a subquery run once per row. It spends from this
    /// budget's limit, deadline and cancellation; memory kept in it "until
    /// the query ends" is kept until the part is dropped, which is when that
    /// evaluation ends, and returns to this budget then.
    pub fn part(self: &Arc<Self>) -> Self {
        Self {
            limit: self.limit,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
            work: AtomicU64::new(0),
            checked_at: AtomicU64::new(0),
            cancelled: self.cancelled.clone(),
            deadline: self.deadline,
            whole: Some(Arc::clone(self)),
        }
    }

    /// The same budget ending at `deadline_nanos` on `clock`.
    #[must_use]
    pub fn with_deadline(mut self, deadline_nanos: u64, clock: MonotonicNanos) -> Self {
        self.deadline = Some((deadline_nanos, clock));
        self
    }

    /// The same budget cancelled by `flag`, which its holder throws.
    #[must_use]
    pub fn with_cancel(mut self, flag: CancelFlag) -> Self {
        self.cancelled = flag;
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
        self.cancelled.cancel();
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
        if let Some(whole) = &self.whole {
            self.work.fetch_add(units, Ordering::AcqRel);
            return whole.work(units);
        }
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
        if self.cancelled.is_cancelled() {
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
        // The whole decides against the query's limit; the part only counts
        // what it holds of it, which the whole's admission bounds.
        if let Some(whole) = &self.whole {
            whole.take(bytes)?;
            let after = self.used.fetch_add(bytes, Ordering::AcqRel) + bytes;
            self.peak.fetch_max(after, Ordering::AcqRel);
            return Ok(());
        }
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
        if let Some(whole) = &self.whole {
            whole.give(bytes);
        }
    }
}

impl Drop for QueryBudget {
    fn drop(&mut self) {
        // Every charge borrows the budget, so none is left: what is still
        // held was kept for the evaluation this part served, which is over.
        if let Some(whole) = &self.whole {
            let held = *self.used.get_mut();
            if held > 0 {
                whole.give(held);
            }
        }
    }
}

/// Work and scratch memory of a loop too hot for a shared counter update per
/// item, such as a library's search: units are counted locally and handed to
/// the budget [`CHECK_EVERY`] at a time, so the deadline and cancellation are
/// still checked as the work accrues, and scratch memory reserved through it
/// is held until it is dropped.
#[derive(Debug)]
pub struct BatchedWork<'b> {
    budget: &'b QueryBudget,
    pending: u64,
    scratch: MemoryCharge<'b>,
}

impl<'b> BatchedWork<'b> {
    /// A batch over `budget` with nothing counted or held yet.
    pub fn new(budget: &'b QueryBudget) -> Self {
        Self {
            budget,
            pending: 0,
            scratch: budget.empty_charge(),
        }
    }

    /// Count `units` of work.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Deadline`] or [`BudgetStop::Cancelled`], from the check
    /// a full batch makes.
    #[inline]
    pub fn add(&mut self, units: u64) -> Result<(), BudgetStop> {
        // Bounded by CHECK_EVERY plus one call's units before every flush.
        self.pending += units;
        if self.pending < CHECK_EVERY {
            return Ok(());
        }
        self.budget.work(core::mem::take(&mut self.pending))
    }

    /// Reserve `bytes` of scratch memory, held until this batch is dropped.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Memory`].
    pub fn reserve(&mut self, bytes: u64) -> Result<(), BudgetStop> {
        self.scratch.grow(bytes)
    }

    /// Hand the work counted since the last full batch to the budget, and
    /// check the deadline and cancellation once more as the loop ends.
    ///
    /// # Errors
    ///
    /// [`BudgetStop::Deadline`] or [`BudgetStop::Cancelled`].
    pub fn finish(mut self) -> Result<(), BudgetStop> {
        self.budget.work(core::mem::take(&mut self.pending))?;
        self.budget.check()
    }
}

/// The progress a library loop (an index search, a scorer) reports as it
/// runs: the units of work it does and the scratch memory it is about to
/// allocate. A report that returns an error stops the loop there.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot meter a search",
    label = "this type does not implement `Meter`",
    note = "pass `&mut Unmetered` to run without a budget, or a `BatchedWork` to charge one"
)]
pub trait Meter {
    /// Why the loop was stopped.
    type Stop;

    /// `units` more units of work were done.
    ///
    /// # Errors
    ///
    /// The loop must stop.
    fn work(&mut self, units: u64) -> Result<(), Self::Stop>;

    /// The loop is about to allocate `bytes` of scratch it holds until it
    /// returns.
    ///
    /// # Errors
    ///
    /// The loop must not allocate it.
    fn scratch(&mut self, bytes: u64) -> Result<(), Self::Stop>;
}

/// A loop nobody stops: index construction, and callers that run under no
/// query budget.
#[derive(Debug, Default, Clone, Copy)]
pub struct Unmetered;

impl Meter for Unmetered {
    type Stop = core::convert::Infallible;

    #[inline(always)]
    fn work(&mut self, _units: u64) -> Result<(), Self::Stop> {
        Ok(())
    }

    #[inline(always)]
    fn scratch(&mut self, _bytes: u64) -> Result<(), Self::Stop> {
        Ok(())
    }
}

impl Meter for BatchedWork<'_> {
    type Stop = BudgetStop;

    #[inline]
    fn work(&mut self, units: u64) -> Result<(), BudgetStop> {
        self.add(units)
    }

    fn scratch(&mut self, bytes: u64) -> Result<(), BudgetStop> {
        self.reserve(bytes)
    }
}

/// Memory reserved from a [`QueryBudget`], returned when dropped.
#[derive(Debug)]
pub struct MemoryCharge<'b> {
    budget: &'b QueryBudget,
    bytes: u64,
}

impl<'b> MemoryCharge<'b> {
    /// Bytes this charge holds.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// An empty charge on the same budget, for memory owned apart from this.
    pub fn empty_like(&self) -> MemoryCharge<'b> {
        self.budget.empty_charge()
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
