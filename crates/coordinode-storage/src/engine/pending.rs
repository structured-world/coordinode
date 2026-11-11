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
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use coordinode_core::budget::{BudgetStop, QueryBudget};
use coordinode_core::txn::proposal::ProposalId;

use crate::engine::partition::Partition;

/// The longest a command's wait goes without looking at its own deadline and
/// cancellation: the period of the check, not a bound under descheduling.
const STOP_CHECK_SLICE: Duration = Duration::from_millis(50);

/// When a wait bounded by `deadline` on behalf of `stop` wakes next: at the
/// deadline, at the command's own deadline if sooner, and within a slice to
/// look at a cancellation, which throws a flag rather than the condvar.
fn wake_at(deadline: Instant, stop: Option<&QueryBudget>) -> Instant {
    let Some(stop) = stop else {
        return deadline;
    };
    let now = Instant::now();
    let mut wake = deadline.min(now + STOP_CHECK_SLICE);
    if let Some(left) = stop.time_left_nanos() {
        wake = wake.min(now + Duration::from_nanos(left));
    }
    wake
}

/// What [`AFTER_WAKE`] runs.
#[cfg(test)]
type WakeHook = Box<dyn FnOnce(&mut Table)>;

#[cfg(test)]
std::thread_local! {
    /// Runs once on this thread right after a wait for the material scope
    /// wakes, with the table held: what a release and a cancellation landing
    /// together leave.
    pub(super) static AFTER_WAKE: core::cell::RefCell<Option<WakeHook>> =
        const { core::cell::RefCell::new(None) };
}

/// Run this thread's [`AFTER_WAKE`] hook, if one is set.
#[cfg(test)]
fn after_wake(table: &mut Table) {
    AFTER_WAKE.with(|hook| {
        if let Some(hook) = hook.borrow_mut().take() {
            hook(table);
        }
    });
}

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

/// The registered commits holding one key, by identity.
#[derive(Debug, Default)]
struct Holders {
    /// Commits that write the key.
    writers: Vec<u64>,
    /// Commits conditioned on the key without writing it.
    conditions: Vec<u64>,
}

impl Table {
    /// Index `commit`'s keys under `id`.
    fn index(&mut self, id: u64, commit: &Admitted) {
        for key in &commit.scope {
            self.by_key.entry(key.clone()).or_default().writers.push(id);
        }
        for key in &commit.guards {
            self.by_key
                .entry(key.clone())
                .or_default()
                .conditions
                .push(id);
        }
    }

    /// Remove `commit`'s keys indexed under `id`.
    fn unindex(&mut self, id: u64, commit: &Admitted) {
        let keys = commit.scope.iter().chain(commit.guards.iter());
        for key in keys {
            if let Some(holders) = self.by_key.get_mut(key) {
                holders.writers.retain(|h| *h != id);
                holders.conditions.retain(|h| *h != id);
                if holders.writers.is_empty() && holders.conditions.is_empty() {
                    self.by_key.remove(key);
                }
            }
        }
    }

    /// The timestamp registered commit `id` lands at.
    fn holder_ts(&self, id: u64) -> u64 {
        self.admitted.get(&id).map_or(0, |a| a.commit_ts)
    }
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
    /// The admitted commits that write or condition on each key, by
    /// identity: an overlap check looks up its own keys instead of reading
    /// every commit registered, and the check runs on every commit. Keys
    /// hashed with the default, collision-resistant hasher: they carry user
    /// values.
    by_key: HashMap<(Partition, Vec<u8>), Holders>,
    /// Timestamps allocated for log entries that are not in the log yet, by
    /// the same identity: metadata proposed outside a commit. They do not
    /// hold back this node's own snapshots, only the closed bound.
    unlogged: HashMap<u64, u64>,
    /// The proposal each registration's write travels in, by registration
    /// identity, once the caller has sealed it into one: the entry carrying
    /// that proposal applies its write before its own bound, so the bound
    /// that entry carries need not stay below it.
    sealed: HashMap<u64, ProposalId>,
    /// The highest bound handed out for a log entry.
    last_closed: u64,
    /// The highest timestamp withdrawn from the table: a timestamp at or
    /// above `last_closed` is in the log, but no entry says so yet.
    max_released: u64,
    /// How many timestamps were withdrawn, for a waiter on the next one.
    withdrawals: u64,
    /// The tasks waiting without a thread for a commit to finish
    /// ([`CompleteAt`]), each under its wait's own number and only while that
    /// wait lives: as many as the waits in progress. Woken with the threads
    /// that wait for one.
    wakers: HashMap<u64, Waker>,
    /// The registrations sharing the material scope, by identity.
    material: std::collections::HashSet<u64>,
    /// Where a terminal decision on the material stands.
    material_claim: Claim,
    /// The highest timestamp a registration sharing the material scope took:
    /// a decision's protected cut reaches it.
    material_last: u64,
}

/// A terminal decision's hold on the material scope.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Claim {
    /// None: writes that change the material share the scope freely.
    #[default]
    Open,
    /// One waits for the writes sharing the scope to land; writes arriving
    /// meanwhile wait behind it, so it is not overtaken.
    Waiting,
    /// One holds the scope: writes that change the material wait.
    Held,
}

/// Whether a registration's write can change the material a coalesced
/// state-delta consumer is measured on. Writes that can share the material
/// scope with each other; a terminal decision on that measure takes the
/// scope alone, so no such write lands between its measurement and its
/// ordered outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Material {
    /// It writes keys whose closure a consumer is charged for.
    Changes,
    /// It writes only records no consumer is charged for: the registry's,
    /// the log's, a member's own.
    Untouched,
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
    /// The lowest commit timestamp applied here and not yet durable in this
    /// store's own journal; `u64::MAX` when everything applied is durable. A
    /// durable view and the GC watermark stay below it. The journal lowers it
    /// before an entry is applied and raises it after a sync.
    durable_ceiling: Arc<std::sync::atomic::AtomicU64>,
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
    /// A terminal decision on the material held its scope past the wait: a
    /// write that changes the material is retried once it is decided; a
    /// second decision, once the first is.
    MaterialClaimed,
    /// The command waiting for the material scope reached its own deadline
    /// or was cancelled first; nothing is held.
    Stopped(BudgetStop),
}

/// How long a write that changes the material waits for a terminal decision
/// on it to be ordered, when the write names no wait of its own. A decision
/// holds the scope for its final catch-up and its ordered write only.
pub const MATERIAL_WAIT: Duration = Duration::from_secs(5);

impl PendingCommits {
    /// An empty table admitting at most `max_in_flight` commits at once.
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            inner: Mutex::new(Table::default()),
            next_id: std::sync::atomic::AtomicU64::new(0),
            finished: Condvar::new(),
            max_in_flight,
            durable_ceiling: Arc::new(std::sync::atomic::AtomicU64::new(u64::MAX)),
        }
    }

    /// See the field: the first commit timestamp a durable view excludes.
    pub fn durable_ceiling(&self) -> u64 {
        self.durable_ceiling
            .load(std::sync::atomic::Ordering::Acquire)
    }

    /// The cell the journal keeps [`Self::durable_ceiling`] in.
    pub(crate) fn durable_ceiling_cell(&self) -> Arc<std::sync::atomic::AtomicU64> {
        Arc::clone(&self.durable_ceiling)
    }

    /// The highest snapshot at or below `view` holding only durable effects:
    /// below every commit in flight and every journal entry not yet durable.
    /// The in-flight floor is read first: a commit no longer in flight
    /// lowered the ceiling before it released its registration, so read
    /// after that release the ceiling already covers it.
    pub fn durable_floor(&self, view: u64) -> u64 {
        self.snapshot_floor(|| view).min(self.durable_ceiling())
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
            let Some(blocking) = Self::first_at(&table, ts) else {
                return Ok(());
            };
            if self.finished.wait_until(&mut table, deadline).timed_out() {
                return Err(blocking);
            }
        }
    }

    /// [`Self::await_complete_at`] for a task, holding no thread while it
    /// waits: ready once no commit at or below `ts` is in flight. Woken by
    /// the commits that finish, as the threads waiting for one are; its
    /// caller bounds the wait.
    pub fn complete_at(self: &Arc<Self>, ts: u64) -> CompleteAt {
        CompleteAt {
            pending: Arc::clone(self),
            ts,
            registered: None,
        }
    }

    /// The first commit at or below `ts` still in flight, if any.
    pub fn in_flight_at(&self, ts: u64) -> Option<u64> {
        Self::first_at(&self.inner.lock(), ts)
    }

    fn first_at(table: &Table, ts: u64) -> Option<u64> {
        table
            .admitted
            .values()
            .map(|a| a.commit_ts)
            .filter(|c| *c <= ts)
            .min()
    }

    /// Wake every thread and task waiting for a commit to finish.
    fn wake_all(&self, mut table: parking_lot::MutexGuard<'_, Table>) {
        // Taken out, not left: a wait still pending registers again when it
        // is next polled.
        let wakers = std::mem::take(&mut table.wakers);
        drop(table);
        // Every waiter is woken rather than one: they are waiting on
        // different timestamps, and the one this commit unblocks is not
        // necessarily the one a single wake would reach.
        self.finished.notify_all();
        for waker in wakers.into_values() {
            waker.wake();
        }
    }

    /// Wait on `table`'s lock until no terminal decision waits for or holds
    /// the material scope; `false` when `deadline` passed first. A command
    /// waiting on its own behalf passes its budget as `stop`: its deadline
    /// or cancellation ends the wait with that reason, and is checked once
    /// more after every wake, so a command stopped as the scope opened is
    /// not admitted.
    fn await_material_open(
        &self,
        table: &mut parking_lot::MutexGuard<'_, Table>,
        deadline: Instant,
        stop: Option<&QueryBudget>,
    ) -> Result<bool, BudgetStop> {
        loop {
            if let Some(stop) = stop {
                stop.check()?;
            }
            if table.material_claim == Claim::Open {
                return Ok(true);
            }
            let timed_out = self
                .finished
                .wait_until(table, wake_at(deadline, stop))
                .timed_out();
            #[cfg(test)]
            after_wake(table);
            // The command's own stop first: resumed past both its deadline
            // and the wait's, it ends with its own reason.
            if let Some(stop) = stop {
                stop.check()?;
            }
            if timed_out && table.material_claim != Claim::Open && Instant::now() >= deadline {
                return Ok(false);
            }
        }
    }

    /// Record registration `id`, at `ts`, as sharing the material scope when
    /// its write changes the material.
    fn share_material(table: &mut Table, id: u64, ts: u64, material: Material) {
        if material == Material::Changes {
            table.material.insert(id);
            table.material_last = table.material_last.max(ts);
        }
    }

    /// Take the material scope for a terminal decision: wait, up to `wait`,
    /// for every registration sharing it to land, while writes that would
    /// share it wait behind this one. The decision is then measured against
    /// a cut at or past [`MaterialClaim::through`] and ordered with the
    /// claim held; no write that changes the material registers until the
    /// claim is released.
    ///
    /// # Errors
    ///
    /// [`Refusal::MaterialClaimed`] when another decision holds or waits for
    /// the scope, or the registrations sharing it did not all land in
    /// `wait`; nothing is held then.
    pub fn claim_material(self: &Arc<Self>, wait: Duration) -> Result<MaterialClaim, Refusal> {
        let deadline = Instant::now() + wait;
        let mut table = self.inner.lock();
        if table.material_claim != Claim::Open {
            return Err(Refusal::MaterialClaimed);
        }
        table.material_claim = Claim::Waiting;
        while !table.material.is_empty() {
            if self.finished.wait_until(&mut table, deadline).timed_out()
                && !table.material.is_empty()
            {
                table.material_claim = Claim::Open;
                // The writes that waited behind it go on.
                self.wake_all(table);
                return Err(Refusal::MaterialClaimed);
            }
        }
        table.material_claim = Claim::Held;
        Ok(MaterialClaim {
            pending: Arc::clone(self),
            through: table.material_last,
        })
    }

    /// Release the material scope a decision held.
    fn release_material(&self) {
        let mut table = self.inner.lock();
        table.material_claim = Claim::Open;
        self.wake_all(table);
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
    ///
    /// A commit whose writes change the material waits, up to
    /// [`MATERIAL_WAIT`] and no longer than its command's own deadline or
    /// cancellation (`stop`), for a terminal decision on it to be ordered.
    pub fn admit_allocated<'p>(
        &'p self,
        allocate: impl FnOnce() -> u64,
        scope: Vec<(Partition, Vec<u8>)>,
        guards: Vec<(Partition, Vec<u8>)>,
        material: Material,
        stop: Option<&QueryBudget>,
    ) -> Result<(u64, Admission<'p>), Refusal> {
        let mut table = self.inner.lock();
        // A command already stopped is not admitted, whatever it writes.
        if let Some(stop) = stop {
            stop.check().map_err(Refusal::Stopped)?;
        }
        if material == Material::Changes
            && !self
                .await_material_open(&mut table, Instant::now() + MATERIAL_WAIT, stop)
                .map_err(Refusal::Stopped)?
        {
            return Err(Refusal::MaterialClaimed);
        }
        let candidate = Admitted {
            commit_ts: 0,
            scope,
            guards,
        };
        Self::refuse_reserved(&table, &candidate)?;
        if let Some(refusal) = Self::overlap(&table, &candidate) {
            return Err(refusal);
        }
        self.insert_admitted(&mut table, allocate, candidate, material)
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
    /// deadline, [`Refusal::MaterialClaimed`] when a terminal decision on
    /// the material still holds its scope then, [`Refusal::AtCapacity`] at
    /// the table's bound.
    pub fn admit_allocated_waiting<'p>(
        &'p self,
        allocate: impl FnOnce() -> u64,
        scope: Vec<(Partition, Vec<u8>)>,
        guards: Vec<(Partition, Vec<u8>)>,
        wait: Duration,
        material: Material,
        stop: Option<&QueryBudget>,
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
        let unreserve = |table: &mut Table, reservation: Option<u64>| {
            if let Some(id) = reservation {
                table.reserved.remove(&id);
            }
        };
        // Both waits end together: the commits on its keys landed and no
        // terminal decision holds the material, checked again after each
        // wake since either can start while the other is waited for. The
        // command's own deadline and cancellation are checked on every pass,
        // the last one included, under the lock the admission is taken in.
        loop {
            if let Some(stop) = stop
                && let Err(stopped) = stop.check()
            {
                unreserve(&mut table, reservation);
                return Err(Refusal::Stopped(stopped));
            }
            if material == Material::Changes && table.material_claim != Claim::Open {
                match self.await_material_open(&mut table, deadline, stop) {
                    Ok(true) => {}
                    Ok(false) => {
                        unreserve(&mut table, reservation);
                        return Err(Refusal::MaterialClaimed);
                    }
                    Err(stopped) => {
                        unreserve(&mut table, reservation);
                        return Err(Refusal::Stopped(stopped));
                    }
                }
                continue;
            }
            if Self::overlap(&table, &candidate).is_none() {
                break;
            }
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
            let timed_out = self
                .finished
                .wait_until(&mut table, wake_at(deadline, stop))
                .timed_out();
            #[cfg(test)]
            after_wake(&mut table);
            // The command's own stop first, as in the decision's wait.
            if let Some(stop) = stop
                && let Err(stopped) = stop.check()
            {
                unreserve(&mut table, reservation);
                return Err(Refusal::Stopped(stopped));
            }
            if timed_out
                && Instant::now() >= deadline
                && let Some(refusal) = Self::overlap(&table, &candidate)
            {
                unreserve(&mut table, reservation);
                return Err(refusal);
            }
        }
        unreserve(&mut table, reservation);
        self.insert_admitted(&mut table, allocate, candidate, material)
    }

    /// Register `candidate` under a freshly allocated timestamp.
    fn insert_admitted<'p>(
        &'p self,
        table: &mut Table,
        allocate: impl FnOnce() -> u64,
        mut candidate: Admitted,
        material: Material,
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
        table.index(id, &candidate);
        table.admitted.insert(id, candidate);
        Self::share_material(table, id, commit_ts, material);
        Ok((commit_ts, Admission { pending: self, id }))
    }

    /// The first admitted commit `candidate` collides with: a key
    /// it writes that the other writes or conditions on, or a key it
    /// conditions on that the other writes.
    fn overlap(table: &Table, candidate: &Admitted) -> Option<Refusal> {
        let refusal = |key: &(Partition, Vec<u8>), holder: u64| Refusal::Overlap {
            partition: key.0,
            key: key.1.clone(),
            holder_ts: table.holder_ts(holder),
        };
        for key in &candidate.scope {
            if let Some(holders) = table.by_key.get(key)
                && let Some(holder) = holders.writers.first().or(holders.conditions.first())
            {
                return Some(refusal(key, *holder));
            }
        }
        for key in &candidate.guards {
            if let Some(holder) = table.by_key.get(key).and_then(|h| h.writers.first()) {
                return Some(refusal(key, *holder));
            }
        }
        None
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
    /// over the timestamp. For an entry whose writes change no consumer's
    /// material (see [`Self::obligate_changing`] for one that does).
    pub fn obligate(self: &Arc<Self>, allocate: impl FnOnce() -> u64) -> (u64, Unlogged) {
        let mut table = self.inner.lock();
        self.obligate_in(&mut table, allocate, Material::Untouched)
    }

    /// [`Self::obligate`] for an entry whose writes change the material: it
    /// shares the material scope, and waits first, up to [`MATERIAL_WAIT`]
    /// and no longer than its command's own deadline or cancellation
    /// (`stop`), for a terminal decision holding it to be ordered.
    ///
    /// # Errors
    ///
    /// [`Refusal::MaterialClaimed`] when a decision still holds the scope at
    /// the deadline, [`Refusal::Stopped`] when the command's own deadline or
    /// cancellation came first; nothing is held then.
    pub fn obligate_changing(
        self: &Arc<Self>,
        allocate: impl FnOnce() -> u64,
        stop: Option<&QueryBudget>,
    ) -> Result<(u64, Unlogged), Refusal> {
        let mut table = self.inner.lock();
        if !self
            .await_material_open(&mut table, Instant::now() + MATERIAL_WAIT, stop)
            .map_err(Refusal::Stopped)?
        {
            return Err(Refusal::MaterialClaimed);
        }
        Ok(self.obligate_in(&mut table, allocate, Material::Changes))
    }

    fn obligate_in(
        self: &Arc<Self>,
        table: &mut parking_lot::MutexGuard<'_, Table>,
        allocate: impl FnOnce() -> u64,
        material: Material,
    ) -> (u64, Unlogged) {
        let commit_ts = allocate();
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.unlogged.insert(id, commit_ts);
        Self::share_material(table, id, commit_ts, material);
        (
            commit_ts,
            Unlogged {
                pending: Arc::clone(self),
                id,
                local: false,
            },
        )
    }

    /// [`Self::obligate`] for a write that never goes to the log (a member's
    /// own records, which are no consumer's material): its timestamp holds
    /// the bound while it is written, and its release makes no closing entry
    /// due, since no entry reached the log for one to cover.
    pub fn obligate_local(self: &Arc<Self>, allocate: impl FnOnce() -> u64) -> (u64, Unlogged) {
        let (commit_ts, mut held) = self.obligate(allocate);
        held.local = true;
        (commit_ts, held)
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
            local: false,
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
        self.stamp_closed_below_sealing(&mut [], read_clock)
    }

    /// [`Self::stamp_closed_below`] for the entry carrying the proposals
    /// `sealed`: a registration sealed into one of them (see
    /// [`Admission::seal`], [`Unlogged::seal`]) does not hold the bound,
    /// since a member applies every proposal of the entry before its bound.
    /// The registration itself stays: every other entry's bound, and every
    /// overlap check, still sees it until it is withdrawn.
    ///
    /// `sealed` is sorted in place, so each registration is checked against
    /// a batch entry's proposals by search rather than by a scan of them.
    pub fn stamp_closed_below_sealing(
        &self,
        sealed: &mut [ProposalId],
        read_clock: impl FnOnce() -> u64,
    ) -> u64 {
        sealed.sort_unstable_by_key(|p| p.as_raw());
        let mut table = self.inner.lock();
        let latest = read_clock();
        let bound = Self::closed_below_except(&table, latest, sealed);
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
        Self::closed_below_except(table, latest, &[])
    }

    /// The bound below every registration except those sealed into one of
    /// the proposals `sealed`, which is sorted by id.
    fn closed_below_except(table: &Table, latest: u64, sealed: &[ProposalId]) -> u64 {
        let held_elsewhere = |id: &u64| {
            table.sealed.get(id).is_none_or(|proposal| {
                sealed
                    .binary_search_by_key(&proposal.as_raw(), |p| p.as_raw())
                    .is_err()
            })
        };
        let held = table
            .admitted
            .iter()
            .filter(|(id, _)| held_elsewhere(id))
            .map(|(_, a)| a.commit_ts)
            .chain(
                table
                    .unlogged
                    .iter()
                    .filter(|(id, _)| held_elsewhere(id))
                    .map(|(_, ts)| *ts),
            )
            .min();
        // A clock in microseconds since the epoch is far below u64::MAX, so
        // the bound one past it cannot overflow.
        held.unwrap_or(latest + 1)
    }

    fn withdraw(&self, id: u64) {
        let mut table = self.inner.lock();
        table.material.remove(&id);
        table.sealed.remove(&id);
        if let Some(admitted) = table.admitted.remove(&id) {
            table.unindex(id, &admitted);
            table.max_released = table.max_released.max(admitted.commit_ts);
            table.withdrawals += 1;
        }
        self.wake_all(table);
    }

    /// Whether a commit admitted or not yet logged below `ts` still holds the
    /// bound: a timestamp allocated before `ts` that has not reached the log
    /// or been withdrawn. None can appear once there is none, since the
    /// clock hands out only later ones. For tests.
    #[cfg(any(test, feature = "test-faults"))]
    pub fn held_below(&self, ts: u64) -> bool {
        let table = self.inner.lock();
        table.admitted.values().any(|a| a.commit_ts < ts)
            || table.unlogged.values().any(|&held| held < ts)
    }

    /// Whether a registration still standing is sealed into `proposal`. While
    /// one is, a bound past its timestamp was handed out only by an entry
    /// carrying `proposal`: every other entry stays below it.
    pub fn holds_sealed(&self, proposal: ProposalId) -> bool {
        self.inner.lock().sealed.values().any(|p| *p == proposal)
    }

    /// Seal registration `id` into `proposal`, if it is still registered.
    fn seal(&self, id: u64, proposal: ProposalId) {
        let mut table = self.inner.lock();
        if table.admitted.contains_key(&id) || table.unlogged.contains_key(&id) {
            table.sealed.insert(id, proposal);
        }
    }

    /// Register a write applied straight to the trees, outside any log, under
    /// a timestamp `allocate` takes: like a commit, so no complete snapshot
    /// covers it before all of it has landed. It conditions on nothing and
    /// is never refused: the table counts it, but its caller writes one key
    /// at a time and holds it only across that write. One that changes the
    /// material waits first, up to [`MATERIAL_WAIT`], for a terminal decision
    /// on it to be ordered.
    ///
    /// # Errors
    ///
    /// [`Refusal::MaterialClaimed`] when a decision still holds the scope at
    /// the deadline; nothing is held then.
    pub fn admit_direct(
        &self,
        allocate: impl FnOnce() -> u64,
        material: Material,
    ) -> Result<(u64, DirectWrite<'_>), Refusal> {
        let mut table = self.inner.lock();
        if material == Material::Changes
            && !self
                .await_material_open(&mut table, Instant::now() + MATERIAL_WAIT, None)
                .map_err(Refusal::Stopped)?
        {
            return Err(Refusal::MaterialClaimed);
        }
        let candidate = Admitted {
            commit_ts: allocate(),
            scope: Vec::new(),
            guards: Vec::new(),
        };
        let commit_ts = candidate.commit_ts;
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        table.index(id, &candidate);
        table.admitted.insert(id, candidate);
        Self::share_material(&mut table, id, commit_ts, material);
        Ok((commit_ts, DirectWrite { pending: self, id }))
    }

    /// Release a direct write: it stops holding snapshots, and makes no
    /// closing entry due, since no entry reached the log for one to cover.
    fn withdraw_direct(&self, id: u64) {
        let mut table = self.inner.lock();
        table.material.remove(&id);
        if let Some(admitted) = table.admitted.remove(&id) {
            table.unindex(id, &admitted);
        }
        self.wake_all(table);
    }

    /// Release an unlogged timestamp. One whose entry went to the log counts
    /// toward a closing entry; a `local` one never reaches the log, so only
    /// stops holding the bound.
    fn withdraw_unlogged(&self, id: u64, local: bool) {
        let mut table = self.inner.lock();
        table.material.remove(&id);
        table.sealed.remove(&id);
        if let Some(commit_ts) = table.unlogged.remove(&id)
            && !local
        {
            table.max_released = table.max_released.max(commit_ts);
            table.withdrawals += 1;
        }
        self.wake_all(table);
    }
}

/// [`PendingCommits::complete_at`]'s wait. Its waker is registered under its
/// own number while it waits, and withdrawn when it is met or dropped.
#[derive(Debug)]
pub struct CompleteAt {
    pending: Arc<PendingCommits>,
    ts: u64,
    /// The number this wait's waker is registered under, once it was.
    registered: Option<u64>,
}

impl Future for CompleteAt {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        let mut table = this.pending.inner.lock();
        if PendingCommits::first_at(&table, this.ts).is_none() {
            if let Some(id) = this.registered.take() {
                table.wakers.remove(&id);
            }
            return Poll::Ready(());
        }
        // Registered under the lock the finishing commit takes, so its wake
        // cannot fall between this check and the registration. A wake takes
        // every registration out, so one may be gone: it is made again.
        let id = *this.registered.get_or_insert_with(|| {
            this.pending
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        });
        match table.wakers.get_mut(&id) {
            Some(waker) => waker.clone_from(cx.waker()),
            None => {
                table.wakers.insert(id, cx.waker().clone());
            }
        }
        Poll::Pending
    }
}

impl Drop for CompleteAt {
    fn drop(&mut self) {
        if let Some(id) = self.registered {
            self.pending.inner.lock().wakers.remove(&id);
        }
    }
}

/// A timestamp allocated for a log entry that is not in the log yet.
/// Withdraws it on drop: hand the entry to the log first.
#[derive(Debug)]
pub struct Unlogged {
    pending: Arc<PendingCommits>,
    id: u64,
    /// Taken for a write that never goes to the log.
    local: bool,
}

impl Unlogged {
    /// Record that this timestamp's write travels in `proposal`, sealed:
    /// the entry carrying it may close the bound past it (see
    /// [`PendingCommits::stamp_closed_below_sealing`]). Seal only the
    /// proposal that carries the whole write.
    pub fn seal(&self, proposal: ProposalId) {
        self.pending.seal(self.id, proposal);
    }
}

impl Drop for Unlogged {
    fn drop(&mut self) {
        self.pending.withdraw_unlogged(self.id, self.local);
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

impl Admission<'_> {
    /// Record that this commit's writes travel in `proposal`, sealed: the
    /// entry carrying it may close the bound past it (see
    /// [`PendingCommits::stamp_closed_below_sealing`]). Seal only the
    /// proposal that carries every write of the commit.
    pub fn seal(&self, proposal: ProposalId) {
        self.pending.seal(self.id, proposal);
    }
}

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.pending.withdraw(self.id);
    }
}

/// A direct write's registration ([`PendingCommits::admit_direct`]), held
/// until all of it has landed.
#[derive(Debug)]
#[must_use = "the registration holds snapshots below the write until it drops"]
pub struct DirectWrite<'p> {
    pending: &'p PendingCommits,
    id: u64,
}

impl Drop for DirectWrite<'_> {
    fn drop(&mut self) {
        self.pending.withdraw_direct(self.id);
    }
}

/// A terminal decision's hold on the material scope
/// ([`PendingCommits::claim_material`]). Its owner keeps it for as long as
/// the attempt it protects can still take effect, an unknown outcome
/// included, and releases it by dropping it.
#[derive(Debug)]
#[must_use = "the claim holds every write that changes the material until it drops"]
pub struct MaterialClaim {
    pending: Arc<PendingCommits>,
    through: u64,
}

impl MaterialClaim {
    /// The highest timestamp a write sharing the scope took before the
    /// claim: a measurement taken under it is of a complete cut at or past
    /// this one.
    pub fn through(&self) -> u64 {
        self.through
    }
}

impl Drop for MaterialClaim {
    fn drop(&mut self) {
        self.pending.release_material();
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
