//! Interactive transactions on their owner: the registry of open contexts,
//! the leadership tenure they belong to, and the commands that run on them.
//!
//! An open transaction belongs to the member that began it while it led, not
//! to the connection the client used: a command for it carries its
//! [`TransactionHandle`] and the [`ActingContext`] it acts for, whatever path
//! it took, and a connection closing leaves it open. Commands for one
//! transaction run one at a time in the order its owner admits them, or by
//! statement nonce for an ordered one; those waiting for their turn sit in a
//! bounded queue on the owner, so a command arriving while another runs
//! waits instead of being refused. A statement repeated under an admitted
//! nonce is never applied twice. When the tenure that opened a transaction
//! ends, the transaction ends with it: commands for it are told it was
//! aborted, and nothing a command still running does can bring it back, not
//! even a later tenure of the same member.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use coordinode_core::group::DeploymentId;
use coordinode_core::operations::{TransactionSnapshot, TransactionsView, TxnOrdering};
use coordinode_core::txn::handle::{ActingContext, AdmissionSignal, TransactionHandle};
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::transaction::TransactionState;
use parking_lot::{Condvar, Mutex, MutexGuard};

use super::{Database, DatabaseError, QuerySession, Row, StatementOptions, TxnMode};
use crate::operation::IntentDigest;

/// Handles of contexts that ended before their commit, kept so a command for
/// one is told it was aborted rather than that nothing is known of it.
const ENDED_REMEMBERED: usize = 4096;

/// Commands one transaction holds waiting for their turn. A client
/// pipelining past it is refused until earlier commands answer.
const MAILBOX: usize = 64;

/// Admitted statement nonces a transaction remembers with their intent, to
/// tell a retransmission from a reused nonce. An ordered transaction refuses
/// any nonce below its next one even past this window.
const ADMITTED_REMEMBERED: usize = 1024;

/// How long an ordered transaction waits for a missing statement before it
/// is aborted, when its BEGIN named no bound.
pub const DEFAULT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// Why an open transaction ended before it was submitted. Nothing of it was
/// applied in any of these cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionEnd {
    /// The member stopped leading: the tenure that held it is over.
    LeaderChanged,
    /// One of its statements failed, which aborts the transaction.
    StatementFailed,
    /// An ordered transaction waited past its drain timeout for a missing
    /// statement.
    DrainTimedOut,
    /// It stayed idle past the idle timeout.
    IdleExpired,
    /// It was rolled back.
    RolledBack,
}

impl core::fmt::Display for TransactionEnd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::LeaderChanged => "the leader that held it changed before it was submitted",
            Self::StatementFailed => "one of its statements failed",
            Self::DrainTimedOut => "a statement it waited for did not arrive in time",
            Self::IdleExpired => "it stayed idle past the idle timeout",
            Self::RolledBack => "it was rolled back",
        })
    }
}

/// What a transaction is opened with and keeps until it ends.
#[derive(Debug, Clone)]
pub struct TransactionOptions {
    /// Who the transaction acts for; every later command must act for the
    /// same. `None` is the embedding host's own trusted access.
    pub actor: Option<ActingContext>,
    /// The read concern of every statement; the database's when `None`.
    pub read_concern: Option<ReadConcernLevel>,
    /// The write concern of the commit; the database's when `None`.
    pub write_concern: Option<WriteConcern>,
    /// How its statements are ordered.
    pub ordering: TxnOrdering,
    /// How long an ordered transaction waits for a missing statement before
    /// it is aborted; the database's setting
    /// ([`Database::set_interactive_drain_timeout`]) when `None`.
    pub drain_timeout: Option<Duration>,
}

impl TransactionOptions {
    /// The host's trusted access with the database's concerns, statements
    /// applied as they come.
    pub fn trusted() -> Self {
        Self {
            actor: None,
            read_concern: None,
            write_concern: None,
            ordering: TxnOrdering::Unordered,
            drain_timeout: None,
        }
    }
}

/// What a commit carries besides the handle.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommitOptions<'a> {
    /// The client operation the commit carries out.
    pub operation: Option<&'a crate::operation::OperationDescriptor>,
    /// Who the commit acts for; the host's trusted access when `None`.
    pub actor: Option<&'a ActingContext>,
    /// The last nonce of an ordered transaction's chain: the commit waits,
    /// within the drain timeout, until every statement up to it has run.
    pub last_nonce: Option<u64>,
    /// Told when the commit has its place behind the commands admitted
    /// before it.
    pub admission: Option<&'a AdmissionSignal>,
}

/// What a rollback carries besides the handle.
#[derive(Debug, Clone, Copy, Default)]
pub struct RollbackOptions<'a> {
    /// Who the rollback acts for; the host's trusted access when `None`.
    pub actor: Option<&'a ActingContext>,
    /// Told when the rollback has its place behind the command running on
    /// the transaction.
    pub admission: Option<&'a AdmissionSignal>,
}

/// One open transaction.
struct Context {
    handle: TransactionHandle,
    /// `None` while a command runs on it.
    state: Option<TransactionState>,
    begun: Instant,
    touched: Instant,
    /// The statements run so far, the intent a commit binds.
    chain: IntentDigest,
    actor: ActingContext,
    read_concern: ReadConcernLevel,
    write_concern: WriteConcern,
    ordering: TxnOrdering,
    drain: Duration,
    /// The nonce an ordered transaction runs next, from 1.
    next_nonce: u64,
    /// Nonces admitted so far with their statements' intent, newest last.
    admitted: VecDeque<(u64, IntentDigest)>,
    /// Commands waiting for their turn.
    waiting: usize,
    /// A commit waits for its chain to complete; the idle sweep leaves the
    /// transaction to the commit's own drain timeout.
    draining: bool,
}

/// The member's leadership tenure: who holds new transactions, and which
/// tenure they belong to. Incarnation 0 means no tenure: the member does not
/// lead, and `leader` is the member that does, when one is known.
#[derive(Debug, Clone, Copy)]
struct Tenure {
    owner: u64,
    incarnation: u64,
    leader: Option<u64>,
}

/// The open transactions of a database, on the member that holds them.
pub(crate) struct Registry {
    contexts: Mutex<HashMap<u64, Context>>,
    /// Woken whenever a context is returned, advanced or ended: the commands
    /// waiting for their turn look again.
    turn: Condvar,
    next: AtomicU64,
    /// The database's deployment, once recorded: it never changes after.
    scope: std::sync::OnceLock<DeploymentId>,
    tenure: Mutex<Tenure>,
    ended: Mutex<VecDeque<(TransactionHandle, TransactionEnd)>>,
    /// The idle timeout, for the countdown a snapshot reports.
    idle_timeout: Mutex<Duration>,
    /// The drain timeout of a transaction whose BEGIN names none.
    drain_timeout: Mutex<Duration>,
}

/// What a transaction's commit or rollback waits for before it takes the
/// context.
#[derive(Debug, Clone, Copy)]
enum Settle {
    /// Every statement admitted so far, and an ordered chain up to its last
    /// nonce.
    Commit { last_nonce: Option<u64> },
    /// Only the command running on it.
    Rollback,
}

impl Registry {
    /// A registry for a database outside a group: it holds its transactions
    /// from the start, in a tenure drawn now, so a handle of an earlier open
    /// of the same store names nothing here.
    pub(crate) fn new(idle_timeout: Duration) -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
            turn: Condvar::new(),
            next: AtomicU64::new(0),
            scope: std::sync::OnceLock::new(),
            tenure: Mutex::new(Tenure {
                owner: 0,
                incarnation: fresh_incarnation(),
                leader: Some(0),
            }),
            ended: Mutex::new(VecDeque::new()),
            idle_timeout: Mutex::new(idle_timeout),
            drain_timeout: Mutex::new(DEFAULT_DRAIN_TIMEOUT),
        }
    }

    pub(crate) fn set_idle_timeout(&self, timeout: Duration) {
        *self.idle_timeout.lock() = timeout;
    }

    /// The tenure that opens transactions now, or the refusal naming the
    /// leader when this member holds none.
    fn current(&self) -> Result<Tenure, DatabaseError> {
        let tenure = *self.tenure.lock();
        if tenure.incarnation == 0 {
            return Err(DatabaseError::NotLeader {
                leader_id: tenure.leader,
            });
        }
        Ok(tenure)
    }

    /// Hand the tenure over: member `owner` sees `leader` leading in `term`.
    /// It holds new transactions in that term when it is the leader, none
    /// otherwise. Contexts of any other tenure end, and commands for them are
    /// told so.
    fn set_tenure(&self, owner: u64, leader: Option<u64>, term: u64) {
        let incarnation = if leader == Some(owner) { term } else { 0 };
        let mut tenure = self.tenure.lock();
        tenure.leader = leader;
        if tenure.owner == owner && tenure.incarnation == incarnation {
            return;
        }
        tenure.owner = owner;
        tenure.incarnation = incarnation;
        // The tenure lock is held across the sweep, so no command admitted
        // under the old tenure parks its context after it.
        let mut contexts = self.contexts.lock();
        let mut ended = self.ended.lock();
        contexts.retain(|_, context| {
            let keep =
                context.handle.owner() == owner && context.handle.incarnation() == incarnation;
            if !keep {
                remember_end(&mut ended, context.handle, TransactionEnd::LeaderChanged);
            }
            keep
        });
        drop(ended);
        drop(contexts);
        self.turn.notify_all();
    }

    /// Why `handle`'s transaction ended, when it ended before its commit.
    fn ended_with(&self, handle: &TransactionHandle) -> Option<TransactionEnd> {
        self.ended
            .lock()
            .iter()
            .find(|(ended, _)| ended == handle)
            .map(|(_, why)| *why)
    }

    /// The refusal for `handle`, which names no context a command may use.
    fn missing(&self, handle: &TransactionHandle) -> DatabaseError {
        match self.ended_with(handle) {
            Some(end) => DatabaseError::TransactionAborted {
                handle: *handle,
                end,
            },
            None => DatabaseError::UnknownTransaction(*handle),
        }
    }

    /// End `handle`'s context for `end`, under the context lock, and wake
    /// the commands waiting on it.
    fn end_locked(
        &self,
        contexts: &mut MutexGuard<'_, HashMap<u64, Context>>,
        handle: &TransactionHandle,
        end: TransactionEnd,
    ) {
        if contexts
            .get(&handle.local())
            .is_some_and(|context| context.handle == *handle)
        {
            contexts.remove(&handle.local());
            remember_end(&mut self.ended.lock(), *handle, end);
        }
        self.turn.notify_all();
    }

    /// End `handle`'s transaction for `end`: a command found it cannot
    /// continue.
    fn end(&self, handle: &TransactionHandle, end: TransactionEnd) {
        let mut contexts = self.contexts.lock();
        self.end_locked(&mut contexts, handle, end);
    }

    /// Take `handle`'s context for one statement acting for `actor`, once it
    /// is the statement's turn: the running command has answered and, in an
    /// ordered transaction, every earlier nonce has run. A context of another
    /// actor is refused as if it did not exist: a handle grants nothing by
    /// itself. A statement repeating an admitted nonce is refused rather than
    /// applied again.
    fn check_out(
        &self,
        handle: &TransactionHandle,
        actor: &ActingContext,
        nonce: Option<u64>,
        intent: &IntentDigest,
        admission: Option<&AdmissionSignal>,
    ) -> Result<CheckedOut, DatabaseError> {
        let nonce = nonce.filter(|n| *n != 0);
        let local = handle.local();
        let mut contexts = self.contexts.lock();
        let mut gap_deadline: Option<Instant> = None;
        loop {
            let Some(context) = contexts
                .get_mut(&local)
                .filter(|context| context.handle == *handle && context.actor == *actor)
            else {
                drop(contexts);
                return Err(self.missing(handle));
            };
            let ordered = context.ordering == TxnOrdering::Ordered;
            if let Some(n) = nonce {
                if let Some((_, admitted)) = context.admitted.iter().find(|(m, _)| *m == n) {
                    return Err(if admitted == intent {
                        DatabaseError::StatementAlreadyAdmitted {
                            handle: *handle,
                            nonce: n,
                        }
                    } else {
                        DatabaseError::StatementNonceReused {
                            handle: *handle,
                            nonce: n,
                        }
                    });
                }
                if ordered && n < context.next_nonce {
                    // Admitted before the remembered window.
                    return Err(DatabaseError::StatementAlreadyAdmitted {
                        handle: *handle,
                        nonce: n,
                    });
                }
            } else if ordered {
                return Err(DatabaseError::StatementNonceRequired(*handle));
            }
            let turn = !ordered || nonce == Some(context.next_nonce);
            if let Some(state) = context.state.take_if(|_| turn) {
                if ordered {
                    context.next_nonce += 1;
                }
                if let Some(n) = nonce {
                    if context.admitted.len() == ADMITTED_REMEMBERED {
                        context.admitted.pop_front();
                    }
                    context.admitted.push_back((n, *intent));
                }
                context.touched = Instant::now();
                if let Some(admission) = admission {
                    admission.admitted();
                }
                return Ok(CheckedOut {
                    state,
                    chain: context.chain,
                    read_concern: context.read_concern,
                    write_concern: context.write_concern,
                });
            }
            if context.waiting >= MAILBOX {
                return Err(DatabaseError::TransactionBusy(*handle));
            }
            // Idle but not this statement's turn: an earlier nonce has not
            // arrived. That wait is bounded by the drain timeout; a wait for
            // a running command is bounded by that command's own budget.
            let gap = context.state.is_some();
            let drain = context.drain;
            context.waiting += 1;
            // Queued: the commands after it may follow.
            if let Some(admission) = admission {
                admission.admitted();
            }
            let timed_out = if gap {
                let deadline = *gap_deadline.get_or_insert_with(|| Instant::now() + drain);
                self.turn.wait_until(&mut contexts, deadline).timed_out()
            } else {
                self.turn.wait(&mut contexts);
                false
            };
            let still_gap = match contexts
                .get_mut(&local)
                .filter(|context| context.handle == *handle)
            {
                Some(context) => {
                    context.waiting -= 1;
                    context.state.is_some() && nonce != Some(context.next_nonce)
                }
                None => false,
            };
            if timed_out && still_gap {
                self.end_locked(&mut contexts, handle, TransactionEnd::DrainTimedOut);
                return Err(DatabaseError::TransactionAborted {
                    handle: *handle,
                    end: TransactionEnd::DrainTimedOut,
                });
            }
        }
    }

    /// Put a context back after its statement, with the statements it has
    /// run now. Refused once it has ended, by its tenure or otherwise: the
    /// statement finished, but the transaction cannot continue.
    fn check_in(
        &self,
        handle: &TransactionHandle,
        state: TransactionState,
        chain: IntentDigest,
    ) -> Result<(), DatabaseError> {
        let mut contexts = self.contexts.lock();
        let Some(context) = contexts
            .get_mut(&handle.local())
            .filter(|context| context.handle == *handle)
        else {
            drop(contexts);
            return Err(self.missing(handle));
        };
        context.state = Some(state);
        context.chain = chain;
        context.touched = Instant::now();
        drop(contexts);
        self.turn.notify_all();
        Ok(())
    }

    /// Remove `handle`'s context for `actor` to finish it, once what `settle`
    /// waits for has run. `None` when no context of `actor` is held under it.
    fn take(
        &self,
        handle: &TransactionHandle,
        actor: &ActingContext,
        settle: Settle,
        admission: Option<&AdmissionSignal>,
    ) -> Result<Option<Finished>, DatabaseError> {
        let local = handle.local();
        let mut contexts = self.contexts.lock();
        let mut deadline: Option<Instant> = None;
        loop {
            let Some(context) = contexts
                .get(&local)
                .filter(|context| context.handle == *handle && context.actor == *actor)
            else {
                return Ok(None);
            };
            // Whether it takes the context now or waits for it, nothing
            // sent after it can get ahead.
            if let Some(admission) = admission {
                admission.admitted();
            }
            let settled = context.state.is_some()
                && match settle {
                    Settle::Rollback => true,
                    Settle::Commit { last_nonce } => {
                        context.waiting == 0
                            && match (context.ordering, last_nonce) {
                                (TxnOrdering::Ordered, Some(last)) => context.next_nonce > last,
                                _ => true,
                            }
                    }
                };
            if settled {
                let context = contexts
                    .remove(&local)
                    .ok_or_else(|| DatabaseError::Other("transaction context vanished".into()))?;
                if matches!(settle, Settle::Rollback) {
                    remember_end(&mut self.ended.lock(), *handle, TransactionEnd::RolledBack);
                }
                drop(contexts);
                self.turn.notify_all();
                return Ok(Some(Finished {
                    state: context.state.ok_or_else(|| {
                        DatabaseError::Other("transaction context vanished".into())
                    })?,
                    chain: context.chain,
                    write_concern: context.write_concern,
                }));
            }
            let at = *deadline.get_or_insert_with(|| Instant::now() + context.drain);
            if let Some(context) = contexts.get_mut(&local) {
                context.draining = true;
            }
            let timed_out = self.turn.wait_until(&mut contexts, at).timed_out();
            if let Some(context) = contexts.get_mut(&local).filter(|c| c.handle == *handle) {
                context.draining = false;
            }
            if timed_out {
                let Some(context) = contexts.get(&local).filter(|c| c.handle == *handle) else {
                    return Ok(None);
                };
                if context.state.is_none() {
                    // A statement still runs past the drain timeout: the
                    // transaction stays as it is, and the caller tries again.
                    return Err(DatabaseError::TransactionBusy(*handle));
                }
                // Idle, yet the chain it must drain is incomplete.
                self.end_locked(&mut contexts, handle, TransactionEnd::DrainTimedOut);
                return Err(DatabaseError::TransactionAborted {
                    handle: *handle,
                    end: TransactionEnd::DrainTimedOut,
                });
            }
        }
    }
}

impl TransactionsView for Registry {
    fn transactions(&self) -> Vec<TransactionSnapshot> {
        let now = Instant::now();
        let timeout = *self.idle_timeout.lock();
        self.contexts
            .lock()
            .values()
            .map(|context| TransactionSnapshot {
                transaction: context.handle.to_string(),
                ordering: context.ordering,
                age_ms: now.duration_since(context.begun).as_millis() as u64,
                // A context a command runs on is never reaped.
                auto_abort_in_ms: context.state.as_ref().map(|_| {
                    let idle = now.duration_since(context.touched);
                    timeout.checked_sub(idle).unwrap_or_default().as_millis() as u64
                }),
            })
            .collect()
    }
}

/// Remember that `handle`'s transaction ended for `end`, forgetting the
/// oldest when the record is full.
fn remember_end(
    ended: &mut VecDeque<(TransactionHandle, TransactionEnd)>,
    handle: TransactionHandle,
    end: TransactionEnd,
) {
    if ended.len() == ENDED_REMEMBERED {
        ended.pop_front();
    }
    ended.push_back((handle, end));
}

/// A context taken for one statement.
struct CheckedOut {
    state: TransactionState,
    chain: IntentDigest,
    read_concern: ReadConcernLevel,
    write_concern: WriteConcern,
}

/// A context removed to commit it.
struct Finished {
    state: TransactionState,
    chain: IntentDigest,
    write_concern: WriteConcern,
}

/// A tenure number for a database outside a group, never 0 and different
/// at every open.
fn fresh_incarnation() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u128(now);
    hasher.finish().max(1)
}

/// The host's own access: an embedding host is its own trust boundary.
fn host_actor(actor: Option<&ActingContext>) -> ActingContext {
    actor.cloned().unwrap_or(ActingContext::Trusted)
}

impl Database {
    /// Tell the database who leads its group: this member `owner` sees
    /// `leader` (`None` during an election) leading in `term`. While `owner`
    /// leads, transactions are opened in that term; otherwise BEGIN is
    /// refused naming `leader`. Transactions of an earlier tenure end: a
    /// command for one is refused as aborted, since nothing of it was
    /// submitted. Called by the group member as leadership moves.
    pub fn set_execution_tenure(&self, owner: u64, leader: Option<u64>, term: u64) {
        self.interactive.set_tenure(owner, leader, term);
    }

    /// How long an ordered transaction waits for a missing statement before
    /// it is aborted, when its BEGIN names no bound ([`DEFAULT_DRAIN_TIMEOUT`]
    /// until set). Applies to transactions begun from now on.
    pub fn set_interactive_drain_timeout(&self, timeout: Duration) {
        *self.interactive.drain_timeout.lock() = timeout;
    }

    /// The open transactions this database holds, for introspection.
    pub fn transactions_view(&self) -> &dyn TransactionsView {
        &self.interactive
    }

    /// Begin an interactive multi-statement transaction with the database's
    /// concerns, acting for the host.
    ///
    /// Pass the returned handle to [`Self::execute_in_transaction`] for each
    /// statement, then [`Self::commit_transaction`] or
    /// [`Self::rollback_transaction`]. Every statement reads the snapshot
    /// pinned here (repeatable read). The transaction is ephemeral: nothing of
    /// it is durable before the commit, and one left idle longer than the
    /// configured timeout ([`Self::set_interactive_idle_timeout`]) is
    /// discarded.
    ///
    /// # Errors
    ///
    /// As [`Self::begin_transaction_with`].
    pub fn begin_transaction(&self) -> Result<TransactionHandle, DatabaseError> {
        self.begin_transaction_with(&TransactionOptions::trusted())
    }

    /// Begin an interactive transaction with `options`, which it keeps until
    /// it ends: who it acts for, its concerns and how its statements are
    /// ordered. Later commands must act for the same caller.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::NotLeader`] when this member holds no transactions:
    /// it does not lead its group.
    pub fn begin_transaction_with(
        &self,
        options: &TransactionOptions,
    ) -> Result<TransactionHandle, DatabaseError> {
        self.reap_idle_transactions(self.interactive_idle_timeout);
        let tenure = self.interactive.current()?;
        let local = self.interactive.next.fetch_add(1, Ordering::Relaxed) + 1;
        let handle = TransactionHandle::new(
            self.handle_scope()?,
            tenure.owner,
            tenure.incarnation,
            local,
        );
        let read_concern = options.read_concern.unwrap_or(self.read_concern);
        let read_ts = self.oracle.next();
        // The view is the engine's complete snapshot, not the timestamp just
        // allocated: a fresh number from the clock can already cover a commit
        // that has not applied, and a transaction starting there would see
        // neither that write nor any sign of it. `read_ts` stays the
        // allocated value because it identifies this attempt. A read concern
        // that must not see a write a crash can still take back reads the
        // durable view.
        let mut txn = if read_concern.is_durable() {
            coordinode_storage::engine::transaction::Transaction::begin_durable(
                &self.engine,
                Some(&self.oracle),
                read_ts,
            )
        } else {
            coordinode_storage::engine::transaction::Transaction::begin(
                &self.engine,
                Some(&self.oracle),
                read_ts,
            )
        };
        let now = Instant::now();
        let context = Context {
            handle,
            state: Some(txn.take_state()),
            begun: now,
            touched: now,
            chain: crate::operation::EMPTY_CHAIN,
            actor: host_actor(options.actor.as_ref()),
            read_concern,
            write_concern: options.write_concern.unwrap_or(self.write_concern),
            ordering: options.ordering,
            drain: options
                .drain_timeout
                .unwrap_or_else(|| *self.interactive.drain_timeout.lock()),
            next_nonce: 1,
            admitted: VecDeque::new(),
            waiting: 0,
            draining: false,
        };
        {
            // Under the tenure lock: a context of a tenure that just ended is
            // not parked after its sweep.
            let tenure_now = self.interactive.tenure.lock();
            if tenure_now.owner != tenure.owner || tenure_now.incarnation != tenure.incarnation {
                return Err(DatabaseError::NotLeader {
                    leader_id: tenure_now.leader,
                });
            }
            self.interactive.contexts.lock().insert(local, context);
        }
        if let Some(begun) = &self.interactive_begun {
            begun();
        }
        Ok(handle)
    }

    /// The deployment new handles name, read once it is recorded.
    fn handle_scope(&self) -> Result<Option<DeploymentId>, DatabaseError> {
        if let Some(scope) = self.interactive.scope.get() {
            return Ok(Some(*scope));
        }
        let recorded = self.deployment()?;
        if let Some(scope) = recorded {
            // A concurrent BEGIN read the same recorded identity.
            let _ = self.interactive.scope.set(scope);
        }
        Ok(recorded)
    }

    /// Call `begun` after every interactive transaction opens. The server's
    /// idle reaper uses it to sleep while no transaction is open.
    pub fn set_interactive_begun_hook(&mut self, begun: std::sync::Arc<dyn Fn() + Send + Sync>) {
        self.interactive_begun = Some(begun);
    }

    /// Run one statement of an interactive transaction.
    ///
    /// The statement reads at the transaction's pinned snapshot and its writes
    /// buffer on the transaction without committing. A statement error aborts
    /// the transaction (SQL "current transaction is aborted"): later commands
    /// for it are refused as aborted. A statement arriving while another runs
    /// waits for its turn.
    pub fn execute_in_transaction(
        &self,
        handle: TransactionHandle,
        query: &str,
        params: Option<HashMap<String, coordinode_core::graph::types::Value>>,
    ) -> Result<Vec<Row>, DatabaseError> {
        self.execute_in_transaction_with(handle, query, params, &StatementOptions::default())
    }

    /// [`Self::execute_in_transaction`] with the vector settings, budget,
    /// caller and nonce of `options`. Its read and write concerns are the
    /// transaction's, fixed when it began, so those fields of `options` are
    /// not read here.
    ///
    /// # Errors
    ///
    /// The handle names no open transaction of this caller, or one that
    /// ended; the queue of waiting commands is full; the nonce was admitted
    /// already, or is missing in an ordered transaction; or the statement
    /// failed, which ends the transaction.
    pub fn execute_in_transaction_with(
        &self,
        handle: TransactionHandle,
        query: &str,
        params: Option<HashMap<String, coordinode_core::graph::types::Value>>,
        options: &StatementOptions,
    ) -> Result<Vec<Row>, DatabaseError> {
        let actor = host_actor(options.actor.as_ref());
        let params = params.filter(|p| !p.is_empty());
        let intent = crate::operation::statement_intent(query, params.as_ref());
        let checked = self.interactive.check_out(
            &handle,
            &actor,
            options.nonce,
            &intent,
            options.admission.as_ref(),
        )?;
        let failed = |e: DatabaseError| {
            self.interactive
                .end(&handle, TransactionEnd::StatementFailed);
            e
        };
        let session = QuerySession {
            read_concern: checked.read_concern,
            snapshot_read_ts: None,
            write_concern: checked.write_concern,
            vector_consistency: options.vector_consistency.or(self.vector_consistency),
            vector_build_wait: options.vector_build_wait.unwrap_or(self.vector_build_wait),
            after_commit_generation: 0,
            budget: self.options_budget(options).map_err(failed)?,
            // The transaction's commit carries its operation.
            operation: None,
        };
        // The chain a commit of the transaction binds as its intent: every
        // statement it ran, in order.
        let chain = crate::operation::chain_intent(&checked.chain, &intent);
        let (rows, _stats, out_state) = self
            .execute_cypher_impl(
                query,
                None,
                params,
                &session,
                TxnMode::Interactive(Box::new(checked.state)),
                &mut None,
            )
            .map_err(failed)?;
        let Some(state) = out_state else {
            self.interactive
                .end(&handle, TransactionEnd::StatementFailed);
            return Ok(rows);
        };
        // Cap buffered (uncommitted) memory: a client that keeps writing
        // without committing must not grow leader memory unbounded.
        let buffered = state.buffered_bytes();
        if buffered > self.max_interactive_txn_bytes {
            return Err(failed(DatabaseError::TransactionTooLarge {
                id: handle,
                buffered,
                limit: self.max_interactive_txn_bytes,
            }));
        }
        self.interactive.check_in(&handle, state, chain)?;
        Ok(rows)
    }

    /// Commit an interactive transaction: validate the accumulated write
    /// set, assign `commit_ts`, and persist every buffered mutation in a
    /// single proposal under the write concern fixed when it began. The
    /// transaction ends whether the commit succeeds or fails; on a write
    /// conflict the client retries the whole transaction from `begin`.
    /// Returns the [`CommitReceipt`](coordinode_core::txn::transaction::CommitReceipt):
    /// `commit_ts` (the `AS OF TIMESTAMP` anchor for this write) plus the
    /// committed Raft index in a group (`None` outside one).
    pub fn commit_transaction(
        &self,
        handle: TransactionHandle,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        self.commit_transaction_with(handle, &CommitOptions::default())
    }

    /// [`Self::commit_transaction`] carrying out client operation
    /// `operation`, whose intent is the transaction's statements in the
    /// order they ran: the writes take effect at most once however often
    /// the operation is committed. When an attempt of it took effect already
    /// or it cannot be admitted, this one applies nothing and the error is
    /// [`DatabaseError::OperationSettled`] with how it stands; a lost reply
    /// is resolved with [`Self::resolve_operation`], never by running the
    /// statements again under a new operation.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_transaction`], and the above.
    pub fn commit_transaction_as(
        &self,
        handle: TransactionHandle,
        operation: &crate::operation::OperationDescriptor,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        self.commit_transaction_with(
            handle,
            &CommitOptions {
                operation: Some(operation),
                ..CommitOptions::default()
            },
        )
    }

    /// Commit `handle`'s transaction as `options` say: once every statement
    /// admitted to it has run (an ordered chain up to its last nonce, within
    /// the drain timeout), acting for its caller and carrying out its
    /// operation when given.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_transaction_as`]; the chain did not complete within
    /// the drain timeout, which aborts the transaction; or a statement still
    /// runs past it, which leaves it open.
    pub fn commit_transaction_with(
        &self,
        handle: TransactionHandle,
        options: &CommitOptions<'_>,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        let actor = host_actor(options.actor);
        let settle = Settle::Commit {
            last_nonce: options.last_nonce,
        };
        let Some(finished) = self
            .interactive
            .take(&handle, &actor, settle, options.admission)?
        else {
            // A commit repeated after its reply was lost: the context is gone
            // with the attempt, and the operation's outcome is the answer.
            // The attempt's intent went with its buffer, so it is not
            // compared.
            if let Some(descriptor) = options.operation {
                return Err(DatabaseError::OperationSettled {
                    op: descriptor.op,
                    outcome: self
                        .resolve_operation(descriptor, crate::operation::Resolve::Lookup)?,
                    intent_differs: false,
                });
            }
            return Err(self.interactive.missing(&handle));
        };
        self.commit_finished(handle, finished, options.operation)
    }

    fn commit_finished(
        &self,
        handle: TransactionHandle,
        finished: Finished,
        operation: Option<&crate::operation::OperationDescriptor>,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        let Finished {
            state,
            chain,
            write_concern,
        } = finished;
        let mut txn = coordinode_storage::engine::transaction::Transaction::resume(
            &self.engine,
            Some(&self.oracle),
            state,
        );
        if let Some(descriptor) = operation {
            self.admit_operation(descriptor, &chain)?;
            txn.bind_operation(descriptor.op, chain);
        }
        // Constraints judge each node as the whole transaction leaves it,
        // not as one of its statements did.
        let fields = self.fields.current()?;
        coordinode_query::executor::runner::check_post_state(&mut txn, &self.engine, &fields)?;
        let commit_ctx = coordinode_storage::engine::transaction::CommitContext {
            write_concern: &write_concern,
            pipeline: Some(self.pipeline.as_ref()),
            id_gen: Some(&self.proposal_id_gen),
        };
        let outcome = txn
            .commit(&commit_ctx)
            .map_err(|e| self.commit_error(handle, e))?;
        // An interactive transaction is always opened against the oracle, so
        // the storage commit always carries a commit timestamp. A missing one
        // is an engine invariant violation, surfaced as an error rather than
        // a made-up timestamp the host would then trust as a cursor.
        let commit_ts = outcome.commit_ts.ok_or_else(|| {
            DatabaseError::Other(format!(
                "transaction {handle} committed without a commit timestamp"
            ))
        })?;
        Ok(coordinode_core::txn::transaction::CommitReceipt {
            commit_ts,
            applied_index: outcome.applied_index,
        })
    }

    /// The error a refused commit of `handle`'s transaction reports. Split by
    /// variant, because the failures call for opposite reactions: only an OCC
    /// conflict is worth retrying from begin; a storage or pipeline failure
    /// retried immediately just fails again.
    fn commit_error(
        &self,
        handle: TransactionHandle,
        e: coordinode_storage::engine::transaction::CommitError,
    ) -> DatabaseError {
        use coordinode_query::executor::runner::ExecutionError;
        use coordinode_storage::engine::transaction::CommitError;
        match e {
            CommitError::Conflict(detail) => DatabaseError::TransactionConflict {
                id: handle,
                source_message: detail,
            },
            CommitError::Storage(s) => DatabaseError::Storage(s),
            CommitError::Serialization(detail) => {
                DatabaseError::Other(format!("commit failed: {detail}"))
            }
            // Retryable like a conflict, but with a delay: the engine sheds
            // writes until compaction drains its debt.
            CommitError::Backpressure => DatabaseError::WriteBackpressure,
            // Retryable at a different address: the leader.
            CommitError::NotLeader { leader_id } => DatabaseError::NotLeader { leader_id },
            // Retryable at the leader, or here once this member is updated.
            CommitError::Mismatched(m) => DatabaseError::Mismatched(m),
            // Not retryable: the same statement stages the same deltas.
            CommitError::CounterOverflow { key } => DatabaseError::Other(format!(
                "counter '{key}' would leave the i64 range; nothing was written"
            )),
            // Not retryable either: the transaction has to change fewer entries.
            e @ CommitError::IndexFanOut { .. } => DatabaseError::Other(e.to_string()),
            // A statement of the transaction outgrew its budget: its writes
            // were cut short, so the transaction cannot commit as staged.
            CommitError::Budget(stop) => DatabaseError::Execution(ExecutionError::Budget(stop)),
            // Retryable from begin, like a conflict, but for a different
            // reason: the condition is re-evaluated against the state the
            // retry reads.
            CommitError::InvariantRefused { reason } => {
                DatabaseError::InvariantRefused { id: handle, reason }
            }
            // Not retryable: the value is held, by the node named. The
            // statements that claimed it are gone, so the index is found by
            // the generation the refusal names.
            CommitError::UniqueValueHeld {
                generation,
                values,
                holder,
            } => match self
                .index_registry
                .all()
                .into_iter()
                .find(|index| index.generation == generation)
            {
                Some(index) => DatabaseError::Execution(
                    coordinode_query::index::UniqueViolation::new(&index, &values, holder).into(),
                ),
                None => DatabaseError::InvariantRefused {
                    id: handle,
                    reason: format!(
                        "a unique value of {generation} is held by node {}",
                        holder.to_element_id()
                    ),
                },
            },
            // Not retryable: the node written breaks a rule being certified.
            e @ CommitError::CandidateViolation { .. } => {
                DatabaseError::Execution(ExecutionError::SchemaViolation(e.to_string()))
            }
            // Retryable once the build is done.
            CommitError::UniquenessUnresolved { generation, limit } => {
                let index = self
                    .index_registry
                    .all()
                    .into_iter()
                    .find(|index| index.generation == generation)
                    .map_or_else(|| generation.to_string(), |index| index.to_string());
                DatabaseError::Execution(ExecutionError::UniquenessUnresolved { index, limit })
            }
            // Retryable, but whether retrying is the right answer is the
            // caller's to decide: the record moved, and what that means
            // depends on what it was writing. The new version travels with
            // the refusal so that decision needs no second read.
            CommitError::RevisionMismatch { expected, current } => {
                DatabaseError::RevisionMismatch {
                    id: handle,
                    expected,
                    current,
                }
            }
            // The operation the commit carries out stands already: the
            // caller resolves it rather than running it again.
            CommitError::OperationSettled {
                op,
                outcome,
                intent_differs,
            } => DatabaseError::OperationSettled {
                op,
                outcome,
                intent_differs,
            },
        }
    }

    /// Roll back an interactive transaction acting for the host: discard all
    /// buffered writes and the OCC read-set. No proposal is emitted (nothing
    /// was durable). A statement running on it finishes first; statements
    /// still waiting for their turn are refused as aborted.
    ///
    /// # Errors
    ///
    /// The handle names no open transaction, or a statement still runs on it
    /// past the drain timeout.
    pub fn rollback_transaction(&self, handle: TransactionHandle) -> Result<(), DatabaseError> {
        self.rollback_transaction_with(handle, &RollbackOptions::default())
    }

    /// [`Self::rollback_transaction`] as `options` say: acting for its caller
    /// (the host's trusted access when `None`).
    ///
    /// # Errors
    ///
    /// As [`Self::rollback_transaction`].
    pub fn rollback_transaction_with(
        &self,
        handle: TransactionHandle,
        options: &RollbackOptions<'_>,
    ) -> Result<(), DatabaseError> {
        match self.interactive.take(
            &handle,
            &host_actor(options.actor),
            Settle::Rollback,
            options.admission,
        )? {
            Some(_) => Ok(()),
            None => Err(self.interactive.missing(&handle)),
        }
    }

    /// Commit this transaction only while the node is still at `expected`.
    ///
    /// `None` requires the node not to exist, which is the create-if-absent
    /// form of the same condition. The condition is checked when the
    /// transaction commits, against the state at that moment, and a mismatch
    /// refuses the whole transaction with the version that is there instead.
    ///
    /// Stating it does not read the node or write anything by itself: it is
    /// the caller saying what its statements were built on. A caller that
    /// wants the ordinary read-modify-write pairs this with
    /// [`Self::node_version`] before the statements that change the node.
    ///
    /// # Errors
    ///
    /// The handle names no open transaction for the host.
    pub fn expect_node_version(
        &self,
        handle: TransactionHandle,
        node_id: coordinode_core::graph::node::NodeId,
        expected: Option<u64>,
    ) -> Result<(), DatabaseError> {
        let key = coordinode_core::graph::node::encode_node_key(self.shard_id, node_id);
        // Not a statement: it takes no nonce and adds nothing to the chain.
        let mut checked = self.interactive.check_out(
            &handle,
            &ActingContext::Trusted,
            None,
            &crate::operation::EMPTY_CHAIN,
            None,
        )?;
        let stated = checked.state.expect_version(
            coordinode_storage::engine::partition::Partition::Node,
            &key,
            expected,
        );
        self.interactive
            .check_in(&handle, checked.state, checked.chain)?;
        Ok(stated?)
    }

    /// Drop interactive transactions idle longer than `timeout`. An open
    /// transaction pins an MVCC snapshot and buffers writes in memory, so an
    /// abandoned one would leak retention and leader memory. A transaction a
    /// command is running on is never idle. Called opportunistically on
    /// `begin`; the server's reaper also runs it at the returned instant, so
    /// reaping does not wait for the next `begin`.
    ///
    /// Returns when the earliest transaction still open becomes idle for
    /// `timeout`, or `None` when none is open.
    pub fn reap_idle_transactions(&self, timeout: Duration) -> Option<Instant> {
        let now = Instant::now();
        let registry = &self.interactive;
        let mut contexts = registry.contexts.lock();
        let mut reaped = false;
        contexts.retain(|_, context| {
            let keep = context.state.is_none()
                || context.draining
                || now.duration_since(context.touched) < timeout;
            if !keep {
                remember_end(
                    &mut registry.ended.lock(),
                    context.handle,
                    TransactionEnd::IdleExpired,
                );
                reaped = true;
            }
            keep
        });
        let next = contexts
            .values()
            .map(|context| context.touched + timeout)
            .min();
        drop(contexts);
        if reaped {
            registry.turn.notify_all();
        }
        next
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
