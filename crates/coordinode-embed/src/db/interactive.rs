//! Interactive transactions on the leader: the registry of open contexts,
//! the leadership tenure they belong to, and the commands that run on them.
//!
//! An open transaction is held by the leader, not by the connection the
//! client used: a command for it carries the [`TransactionId`] the client
//! drew and the [`ActingContext`] it acts for, whatever path it took, and a
//! connection closing leaves it open. Commands for one transaction run one
//! at a time, in the order they reach the leader; those waiting for their
//! turn sit in a bounded queue, so a command arriving while another runs
//! waits instead of being refused. There is no per-statement number: a
//! stream keeps a transaction's commands in the order its client sent them,
//! and a statement is never run twice, so one whose stream was lost while it
//! ran aborts its transaction. When the member stops leading, its open
//! transactions are dropped: nothing of them was submitted, nothing a
//! command still running does can bring one back, and a command for one is
//! told no such transaction is known, as the new leader tells it.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use coordinode_core::operations::{TransactionSnapshot, TransactionsView};
use coordinode_core::txn::command::{
    ActingContext, AdmissionSignal, ExpectedVersion, TimeoutOutcome, TransactionId,
};
use coordinode_core::txn::read_concern::ReadConcernLevel;
use coordinode_core::txn::write_concern::WriteConcern;
use coordinode_storage::engine::transaction::TransactionState;
use parking_lot::{Condvar, Mutex, MutexGuard};

use super::{Database, DatabaseError, QuerySession, Row, StatementOptions, TxnMode};
use crate::operation::{IntentDigest, OperationDescriptor};

/// IDs of transactions that ended on this leader before their commit, kept
/// so a command for one is told it was aborted rather than that nothing is
/// known of it.
const ENDED_REMEMBERED: usize = 4096;

/// Commands one transaction holds waiting for their turn. A client
/// pipelining past it is refused until earlier commands answer.
const MAILBOX: u64 = 64;

/// Why an open transaction ended on its leader other than by a commit the
/// client asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransactionEnd {
    /// One of its statements failed, which aborts the transaction.
    StatementFailed,
    /// It stayed idle past the idle timeout and was rolled back.
    IdleExpired,
    /// It stayed idle past the idle timeout and its timeout outcome
    /// committed it.
    IdleCommitted,
    /// It stayed idle past the idle timeout and the commit its timeout
    /// outcome made was refused; nothing of it was applied.
    IdleCommitRefused,
    /// It stayed idle past the idle timeout and the commit its timeout
    /// outcome made failed without proving whether it applied; its
    /// operation resolves how it stands.
    IdleCommitUnknown,
    /// It was rolled back.
    #[default]
    RolledBack,
    /// The stream one of its statements ran on was lost before the answer
    /// reached the client, which cannot know what the statement did.
    StreamLost,
}

impl core::fmt::Display for TransactionEnd {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::StatementFailed => "one of its statements failed",
            Self::IdleExpired => "it stayed idle past the idle timeout",
            Self::IdleCommitted => "it was committed when it stayed idle past the idle timeout",
            Self::IdleCommitRefused => {
                "its commit when it stayed idle past the idle timeout was refused"
            }
            Self::IdleCommitUnknown => {
                "its commit when it stayed idle past the idle timeout has an unknown outcome"
            }
            Self::RolledBack => "it was rolled back",
            Self::StreamLost => "the stream one of its statements ran on was lost",
        })
    }
}

/// What a transaction is opened with and keeps until it ends.
#[derive(Debug, Clone, Default)]
pub struct TransactionOptions {
    /// Who the transaction acts for; every later command must act for the
    /// same. `None` is the embedding host's own trusted access.
    pub actor: Option<ActingContext>,
    /// The read concern of every statement; the database's when `None`.
    pub read_concern: Option<ReadConcernLevel>,
    /// The write concern of the commit; the database's when `None`.
    pub write_concern: Option<WriteConcern>,
    /// What ends it when it stays idle past the idle timeout; the
    /// database's setting ([`Database::set_interactive_timeout_outcome`])
    /// when `None`.
    pub timeout_outcome: Option<TimeoutOutcome>,
    /// The client operation its commit carries out. Required when it
    /// commits at idle expiry, so that its client can resolve that commit.
    pub operation: Option<OperationDescriptor>,
}

impl TransactionOptions {
    /// The host's trusted access with the database's settings.
    pub fn trusted() -> Self {
        Self::default()
    }
}

/// What a commit carries besides the transaction ID.
#[derive(Debug, Clone, Copy, Default)]
pub struct CommitOptions<'a> {
    /// The client operation the commit carries out; the one the transaction
    /// was started under when `None`.
    pub operation: Option<&'a OperationDescriptor>,
    /// Who the commit acts for; the host's trusted access when `None`.
    pub actor: Option<&'a ActingContext>,
    /// Told when the commit has its place behind the commands admitted
    /// before it.
    pub admission: Option<&'a AdmissionSignal>,
    /// The versions the caller built its statements on: a node that is at
    /// another refuses the whole commit, as
    /// [`Database::expect_node_version`] states for the host.
    pub expect: &'a [ExpectedVersion],
}

/// What a rollback carries besides the transaction ID.
#[derive(Debug, Clone, Copy, Default)]
pub struct RollbackOptions<'a> {
    /// Who the rollback acts for; the host's trusted access when `None`.
    pub actor: Option<&'a ActingContext>,
    /// Told when the rollback has its place behind the commands admitted
    /// before it.
    pub admission: Option<&'a AdmissionSignal>,
    /// Why the transaction ends: [`TransactionEnd::RolledBack`] for a
    /// rollback the client asked for, [`TransactionEnd::StreamLost`] for one
    /// its ingress makes after losing the stream a statement ran on.
    pub end: TransactionEnd,
}

/// One open transaction.
struct Context {
    id: TransactionId,
    /// `None` while a command runs on it.
    state: Option<TransactionState>,
    begun: Instant,
    touched: Instant,
    /// The statements run so far, the intent a commit binds.
    chain: IntentDigest,
    actor: ActingContext,
    read_concern: ReadConcernLevel,
    write_concern: WriteConcern,
    timeout_outcome: TimeoutOutcome,
    operation: Option<OperationDescriptor>,
    /// The turn handed to the next command that arrives.
    next_turn: u64,
    /// The turn that runs now, or runs next once the context is back.
    serving: u64,
    /// Its timeout outcome is committing it; it ends once the commit's
    /// result is known.
    committing: bool,
}

impl Context {
    /// Commands that arrived and have not finished, the running one included.
    fn pending(&self) -> u64 {
        self.next_turn - self.serving
    }
}

/// The member's leadership tenure. `term` 0 means none: the member does not
/// lead, and `leader` is the member that does, when one is known.
#[derive(Debug, Clone, Copy)]
struct Tenure {
    owner: u64,
    term: u64,
    leader: Option<u64>,
}

/// The open transactions of a database, on the leader that holds them.
pub(crate) struct Registry {
    contexts: Mutex<HashMap<TransactionId, Context>>,
    /// Woken whenever a context is returned or ended: the commands waiting
    /// for their turn look again.
    turn: Condvar,
    tenure: Mutex<Tenure>,
    ended: Mutex<VecDeque<(TransactionId, TransactionEnd)>>,
    /// The idle timeout, for the countdown a snapshot reports.
    idle_timeout: Mutex<Duration>,
    /// The timeout outcome of a transaction whose start names none.
    timeout_outcome: Mutex<TimeoutOutcome>,
}

impl Registry {
    /// A registry for a database outside a group: it holds its transactions
    /// from the start.
    pub(crate) fn new(idle_timeout: Duration) -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
            turn: Condvar::new(),
            tenure: Mutex::new(Tenure {
                owner: 0,
                term: 1,
                leader: Some(0),
            }),
            ended: Mutex::new(VecDeque::new()),
            idle_timeout: Mutex::new(idle_timeout),
            timeout_outcome: Mutex::new(TimeoutOutcome::default()),
        }
    }

    pub(crate) fn set_idle_timeout(&self, timeout: Duration) {
        *self.idle_timeout.lock() = timeout;
    }

    /// The tenure that opens transactions now, or the refusal naming the
    /// leader when this member holds none.
    fn current(&self) -> Result<Tenure, DatabaseError> {
        let tenure = *self.tenure.lock();
        if tenure.term == 0 {
            return Err(DatabaseError::NotLeader {
                leader_id: tenure.leader,
            });
        }
        Ok(tenure)
    }

    /// Hand the tenure over: member `owner` sees `leader` leading in `term`.
    /// It holds transactions in that term when it is the leader, none
    /// otherwise. Any other tenure drops every open transaction: all of them
    /// were opened in the tenure that just ended, nothing of them was
    /// submitted, and a command for one is told no such transaction is
    /// known, as the new leader tells it.
    fn set_tenure(&self, owner: u64, leader: Option<u64>, term: u64) {
        let term = if leader == Some(owner) { term } else { 0 };
        let mut tenure = self.tenure.lock();
        tenure.leader = leader;
        if tenure.owner == owner && tenure.term == term {
            return;
        }
        tenure.owner = owner;
        tenure.term = term;
        // The tenure lock is held across the sweep, and a BEGIN parks its
        // context under it, so no transaction of the old tenure is left.
        self.contexts.lock().clear();
        self.ended.lock().clear();
        drop(tenure);
        self.turn.notify_all();
    }

    /// Why `id`'s transaction ended, when it ended here before its commit.
    fn ended_with(&self, id: &TransactionId) -> Option<TransactionEnd> {
        self.ended
            .lock()
            .iter()
            .find(|(ended, _)| ended == id)
            .map(|(_, why)| *why)
    }

    /// The refusal for `id`, which names no context a command may use.
    fn missing(&self, id: &TransactionId) -> DatabaseError {
        match self.ended_with(id) {
            Some(TransactionEnd::IdleCommitted) => {
                DatabaseError::TransactionCommittedOnTimeout(*id)
            }
            Some(TransactionEnd::IdleCommitUnknown) => {
                DatabaseError::TransactionOutcomeUnknown(*id)
            }
            Some(end) => DatabaseError::TransactionAborted { id: *id, end },
            None => DatabaseError::UnknownTransaction(*id),
        }
    }

    /// End `id`'s context for `end`, under the context lock, and wake the
    /// commands waiting on it.
    fn end_locked(
        &self,
        contexts: &mut MutexGuard<'_, HashMap<TransactionId, Context>>,
        id: &TransactionId,
        end: TransactionEnd,
    ) {
        if contexts.remove(id).is_some() {
            remember_end(&mut self.ended.lock(), *id, end);
        }
        self.turn.notify_all();
    }

    /// End `id`'s transaction for `end`: a command found it cannot continue.
    fn end(&self, id: &TransactionId, end: TransactionEnd) {
        let mut contexts = self.contexts.lock();
        self.end_locked(&mut contexts, id, end);
    }

    /// Wait for the turn of a command for `id` acting for `actor`, in the
    /// order commands reach the context: the command is handed the next
    /// turn and runs once every command before it has finished. A context of
    /// another actor is refused as if it did not exist: an ID grants nothing
    /// by itself. Returns with the context lock held and the command's turn
    /// due.
    fn await_turn<'a>(
        &'a self,
        id: &TransactionId,
        actor: &ActingContext,
        admission: Option<&AdmissionSignal>,
    ) -> Result<MutexGuard<'a, HashMap<TransactionId, Context>>, DatabaseError> {
        let mut contexts = self.contexts.lock();
        let turn = {
            let Some(context) = contexts
                .get_mut(id)
                .filter(|context| context.actor == *actor)
            else {
                drop(contexts);
                return Err(self.missing(id));
            };
            if context.pending() > MAILBOX {
                return Err(DatabaseError::TransactionBusy(*id));
            }
            let turn = context.next_turn;
            context.next_turn += 1;
            turn
        };
        // It has its place: the commands after it may follow.
        if let Some(admission) = admission {
            admission.admitted();
        }
        loop {
            match contexts.get(id) {
                Some(context) if context.serving == turn && context.state.is_some() => {
                    return Ok(contexts);
                }
                Some(_) => self.turn.wait(&mut contexts),
                None => {
                    drop(contexts);
                    return Err(self.missing(id));
                }
            }
        }
    }

    /// Take `id`'s context for one command acting for `actor`, once it is
    /// the command's turn.
    fn check_out(
        &self,
        id: &TransactionId,
        actor: &ActingContext,
        admission: Option<&AdmissionSignal>,
    ) -> Result<CheckedOut, DatabaseError> {
        let mut contexts = self.await_turn(id, actor, admission)?;
        let context = contexts
            .get_mut(id)
            .ok_or_else(|| DatabaseError::Other("transaction context vanished".into()))?;
        let state = context
            .state
            .take()
            .ok_or_else(|| DatabaseError::Other("transaction context is in use".into()))?;
        context.touched = Instant::now();
        Ok(CheckedOut {
            state,
            chain: context.chain,
            read_concern: context.read_concern,
            write_concern: context.write_concern,
        })
    }

    /// Put a context back after its command, with the statements it has
    /// run now, and pass the turn on. Refused once it has ended, with its
    /// tenure or otherwise: the command finished, but the transaction cannot
    /// continue.
    fn check_in(
        &self,
        id: &TransactionId,
        state: TransactionState,
        chain: IntentDigest,
    ) -> Result<(), DatabaseError> {
        let mut contexts = self.contexts.lock();
        let Some(context) = contexts.get_mut(id) else {
            drop(contexts);
            return Err(self.missing(id));
        };
        context.state = Some(state);
        context.chain = chain;
        context.serving += 1;
        context.touched = Instant::now();
        drop(contexts);
        self.turn.notify_all();
        Ok(())
    }

    /// Remove `id`'s context for `actor` to finish it, once every command
    /// that reached it before has finished. A rollback records why the
    /// transaction ended. `None` when no context of `actor` is held under
    /// it.
    fn take(
        &self,
        id: &TransactionId,
        actor: &ActingContext,
        rollback: Option<TransactionEnd>,
        admission: Option<&AdmissionSignal>,
    ) -> Result<Option<Finished>, DatabaseError> {
        let mut contexts = match self.await_turn(id, actor, admission) {
            Ok(contexts) => contexts,
            Err(
                DatabaseError::UnknownTransaction(_) | DatabaseError::TransactionAborted { .. },
            ) => {
                return Ok(None);
            }
            Err(
                DatabaseError::TransactionCommittedOnTimeout(_)
                | DatabaseError::TransactionOutcomeUnknown(_),
            ) => return Ok(None),
            Err(other) => return Err(other),
        };
        let context = contexts
            .remove(id)
            .ok_or_else(|| DatabaseError::Other("transaction context vanished".into()))?;
        if let Some(end) = rollback {
            remember_end(&mut self.ended.lock(), *id, end);
        }
        drop(contexts);
        self.turn.notify_all();
        Ok(Some(Finished {
            state: context
                .state
                .ok_or_else(|| DatabaseError::Other("transaction context vanished".into()))?,
            chain: context.chain,
            write_concern: context.write_concern,
            operation: context.operation,
        }))
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
                transaction: context.id.to_string(),
                timeout_outcome: context.timeout_outcome,
                age_ms: now.duration_since(context.begun).as_millis() as u64,
                // A context a command runs on is never reaped.
                expires_in_ms: context.state.as_ref().map(|_| {
                    let idle = now.duration_since(context.touched);
                    timeout.checked_sub(idle).unwrap_or_default().as_millis() as u64
                }),
            })
            .collect()
    }
}

/// Remember that `id`'s transaction ended for `end`, forgetting the oldest
/// when the record is full.
fn remember_end(
    ended: &mut VecDeque<(TransactionId, TransactionEnd)>,
    id: TransactionId,
    end: TransactionEnd,
) {
    if ended.len() == ENDED_REMEMBERED {
        ended.pop_front();
    }
    ended.push_back((id, end));
}

/// How a commit made at idle expiry ended, read from its result. Success is
/// only a candidate for committed: the caller confirms it against the
/// operation's record, since an early acknowledgement precedes apply. A refusal
/// counts as applying nothing only when its source proves it: validation,
/// constraint and admission refusals are decided before the proposal is
/// handed over, and the pipeline reports a member that does not lead only
/// when no attempt reached the log (a refused retry after an attempt that
/// did is reported as a timeout). Any other failure, a lost leadership or a
/// storage error after the hand-over included, proves nothing; nor does a
/// settled operation, whose record rather than this attempt says whether it
/// took effect. Those are unknown, for the operation to resolve.
fn idle_commit_end(
    result: &Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError>,
) -> TransactionEnd {
    match result {
        Ok(_) => TransactionEnd::IdleCommitted,
        Err(
            DatabaseError::TransactionConflict { .. }
            | DatabaseError::InvariantRefused { .. }
            | DatabaseError::RevisionMismatch { .. }
            | DatabaseError::WriteBackpressure
            | DatabaseError::NotLeader { .. }
            | DatabaseError::Mismatched(_)
            | DatabaseError::Execution(_),
        ) => TransactionEnd::IdleCommitRefused,
        Err(_) => TransactionEnd::IdleCommitUnknown,
    }
}

/// A new transaction ID: a random UUID (version 4), as any client draws one.
pub fn draw_transaction_id() -> TransactionId {
    TransactionId::from_raw(*uuid::Uuid::new_v4().as_bytes())
}

/// A context taken for one command.
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
    operation: Option<OperationDescriptor>,
}

/// The host's own access: an embedding host is its own trust boundary.
fn host_actor(actor: Option<&ActingContext>) -> ActingContext {
    actor.cloned().unwrap_or(ActingContext::Trusted)
}

impl Database {
    /// Tell the database who leads its group: this member `owner` sees
    /// `leader` (`None` during an election) leading in `term`. While `owner`
    /// leads, it holds transactions; otherwise BEGIN is refused naming
    /// `leader`. Transactions of an earlier tenure are dropped: nothing of
    /// them was submitted, and a command for one is told no such transaction
    /// is known. Called by the group member as leadership moves.
    pub fn set_execution_tenure(&self, owner: u64, leader: Option<u64>, term: u64) {
        self.interactive.set_tenure(owner, leader, term);
    }

    /// What ends a transaction that stays idle past the idle timeout, when
    /// its start names nothing ([`TimeoutOutcome::Rollback`] until set).
    /// Applies to transactions begun from now on.
    pub fn set_interactive_timeout_outcome(&self, outcome: TimeoutOutcome) {
        *self.interactive.timeout_outcome.lock() = outcome;
    }

    /// The open transactions this database holds, for introspection.
    pub fn transactions_view(&self) -> &dyn TransactionsView {
        &self.interactive
    }

    /// Begin an interactive multi-statement transaction with the database's
    /// settings, acting for the host, under a newly drawn ID.
    ///
    /// Pass the returned ID to [`Self::execute_in_transaction`] for each
    /// statement, then [`Self::commit_transaction`] or
    /// [`Self::rollback_transaction`]. Every statement reads the snapshot
    /// pinned here (repeatable read). The transaction is ephemeral: nothing of
    /// it is durable before the commit, and one left idle longer than the
    /// configured timeout ([`Self::set_interactive_idle_timeout`]) ends with
    /// its timeout outcome.
    ///
    /// # Errors
    ///
    /// As [`Self::begin_transaction_with`].
    pub fn begin_transaction(&self) -> Result<TransactionId, DatabaseError> {
        self.begin_transaction_with(&TransactionOptions::trusted())
    }

    /// Begin an interactive transaction with `options` under a newly drawn
    /// ID, as [`Self::begin_transaction_as`] does for an ID a client drew.
    ///
    /// # Errors
    ///
    /// As [`Self::begin_transaction_as`].
    pub fn begin_transaction_with(
        &self,
        options: &TransactionOptions,
    ) -> Result<TransactionId, DatabaseError> {
        let id = draw_transaction_id();
        self.begin_transaction_as(id, options)?;
        Ok(id)
    }

    /// Begin an interactive transaction under `id`, the random UUID a client
    /// drew for it, with `options`, which it keeps until it ends: who it acts
    /// for, its concerns, its timeout outcome and its operation. Later
    /// commands must act for the same caller.
    ///
    /// # Errors
    ///
    /// [`DatabaseError::NotLeader`] when this member holds no transactions:
    /// it does not lead its group. [`DatabaseError::TransactionIdInUse`]
    /// when a transaction is open under `id`, or ended under it recently: an
    /// ID names one transaction, and a start is never joined to another.
    /// [`DatabaseError::OperationRequired`] when the transaction commits at
    /// idle expiry and names no operation.
    pub fn begin_transaction_as(
        &self,
        id: TransactionId,
        options: &TransactionOptions,
    ) -> Result<(), DatabaseError> {
        let timeout_outcome = options
            .timeout_outcome
            .unwrap_or_else(|| *self.interactive.timeout_outcome.lock());
        if timeout_outcome == TimeoutOutcome::Commit && options.operation.is_none() {
            return Err(DatabaseError::OperationRequired(id));
        }
        self.reap_idle_transactions(self.interactive_idle_timeout);
        let tenure = self.interactive.current()?;
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
            id,
            state: Some(txn.take_state()),
            begun: now,
            touched: now,
            chain: crate::operation::EMPTY_CHAIN,
            actor: host_actor(options.actor.as_ref()),
            read_concern,
            write_concern: options.write_concern.unwrap_or(self.write_concern),
            timeout_outcome,
            operation: options.operation,
            next_turn: 0,
            serving: 0,
            committing: false,
        };
        {
            // Under the tenure lock: a context of a tenure that just ended is
            // not parked after its sweep.
            let tenure_now = self.interactive.tenure.lock();
            if tenure_now.owner != tenure.owner || tenure_now.term != tenure.term {
                return Err(DatabaseError::NotLeader {
                    leader_id: tenure_now.leader,
                });
            }
            let mut contexts = self.interactive.contexts.lock();
            if contexts.contains_key(&id) || self.interactive.ended_with(&id).is_some() {
                return Err(DatabaseError::TransactionIdInUse(id));
            }
            contexts.insert(id, context);
        }
        if let Some(begun) = &self.interactive_begun {
            begun();
        }
        Ok(())
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
        id: TransactionId,
        query: &str,
        params: Option<HashMap<String, coordinode_core::graph::types::Value>>,
    ) -> Result<Vec<Row>, DatabaseError> {
        self.execute_in_transaction_with(id, query, params, &StatementOptions::default())
    }

    /// [`Self::execute_in_transaction`] with the vector settings, budget and
    /// caller of `options`. Its read and write concerns are the
    /// transaction's, fixed when it began, so those fields of `options` are
    /// not read here.
    ///
    /// # Errors
    ///
    /// The ID names no open transaction of this caller, or one that ended;
    /// the queue of waiting commands is full; or the statement failed, which
    /// ends the transaction.
    pub fn execute_in_transaction_with(
        &self,
        id: TransactionId,
        query: &str,
        params: Option<HashMap<String, coordinode_core::graph::types::Value>>,
        options: &StatementOptions,
    ) -> Result<Vec<Row>, DatabaseError> {
        let actor = host_actor(options.actor.as_ref());
        let params = params.filter(|p| !p.is_empty());
        let intent = crate::operation::statement_intent(query, params.as_ref());
        let checked = self
            .interactive
            .check_out(&id, &actor, options.admission.as_ref())?;
        let failed = |e: DatabaseError| {
            self.interactive.end(&id, TransactionEnd::StatementFailed);
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
            self.interactive.end(&id, TransactionEnd::StatementFailed);
            return Ok(rows);
        };
        // Cap buffered (uncommitted) memory: a client that keeps writing
        // without committing must not grow leader memory unbounded.
        let buffered = state.buffered_bytes();
        if buffered > self.max_interactive_txn_bytes {
            return Err(failed(DatabaseError::TransactionTooLarge {
                id,
                buffered,
                limit: self.max_interactive_txn_bytes,
            }));
        }
        self.interactive.check_in(&id, state, chain)?;
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
        id: TransactionId,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        self.commit_transaction_with(id, &CommitOptions::default())
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
        id: TransactionId,
        operation: &OperationDescriptor,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        self.commit_transaction_with(
            id,
            &CommitOptions {
                operation: Some(operation),
                ..CommitOptions::default()
            },
        )
    }

    /// Commit `id`'s transaction as `options` say: once every command that
    /// reached it before has run, acting for its caller and carrying out its
    /// operation, or the one it was started under.
    ///
    /// # Errors
    ///
    /// As [`Self::commit_transaction_as`];
    /// [`DatabaseError::OperationDiffers`] when `options` names another
    /// operation than the start did, which leaves the transaction open.
    pub fn commit_transaction_with(
        &self,
        id: TransactionId,
        options: &CommitOptions<'_>,
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        let actor = host_actor(options.actor);
        if let Some(named) = options.operation {
            let started = self
                .interactive
                .contexts
                .lock()
                .get(&id)
                .filter(|context| context.actor == actor)
                .and_then(|context| context.operation);
            if started.is_some_and(|started| started != *named) {
                return Err(DatabaseError::OperationDiffers(id));
            }
        }
        let Some(finished) = self
            .interactive
            .take(&id, &actor, None, options.admission)?
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
            return Err(self.interactive.missing(&id));
        };
        let operation = options.operation.copied().or(finished.operation);
        self.commit_finished(id, finished, operation.as_ref(), options.expect)
    }

    fn commit_finished(
        &self,
        id: TransactionId,
        finished: Finished,
        operation: Option<&OperationDescriptor>,
        expect: &[ExpectedVersion],
    ) -> Result<coordinode_core::txn::transaction::CommitReceipt, DatabaseError> {
        let Finished {
            mut state,
            chain,
            write_concern,
            operation: _,
        } = finished;
        for expected in expect {
            let key = coordinode_core::graph::node::encode_node_key(self.shard_id, expected.node);
            state.expect_version(
                coordinode_storage::engine::partition::Partition::Node,
                &key,
                expected.version,
            )?;
        }
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
            .map_err(|e| self.commit_error(id, e))?;
        // An interactive transaction is always opened against the oracle, so
        // the storage commit always carries a commit timestamp. A missing one
        // is an engine invariant violation, surfaced as an error rather than
        // a made-up timestamp the host would then trust as a cursor.
        let commit_ts = outcome.commit_ts.ok_or_else(|| {
            DatabaseError::Other(format!(
                "transaction {id} committed without a commit timestamp"
            ))
        })?;
        Ok(coordinode_core::txn::transaction::CommitReceipt {
            commit_ts,
            applied_index: outcome.applied_index,
        })
    }

    /// The error a refused commit of `id`'s transaction reports. Split by
    /// variant, because the failures call for opposite reactions: only an OCC
    /// conflict is worth retrying from begin; a storage or pipeline failure
    /// retried immediately just fails again.
    fn commit_error(
        &self,
        id: TransactionId,
        e: coordinode_storage::engine::transaction::CommitError,
    ) -> DatabaseError {
        use coordinode_query::executor::runner::ExecutionError;
        use coordinode_storage::engine::transaction::CommitError;
        match e {
            CommitError::Conflict(detail) => DatabaseError::TransactionConflict {
                id,
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
                DatabaseError::InvariantRefused { id, reason }
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
                    id,
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
                    id,
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
    /// was durable). It takes its turn behind the commands that reached the
    /// transaction before it.
    ///
    /// # Errors
    ///
    /// The ID names no open transaction.
    pub fn rollback_transaction(&self, id: TransactionId) -> Result<(), DatabaseError> {
        self.rollback_transaction_with(id, &RollbackOptions::default())
    }

    /// [`Self::rollback_transaction`] as `options` say: acting for its caller
    /// (the host's trusted access when `None`), and recording why the
    /// transaction ended.
    ///
    /// # Errors
    ///
    /// As [`Self::rollback_transaction`].
    pub fn rollback_transaction_with(
        &self,
        id: TransactionId,
        options: &RollbackOptions<'_>,
    ) -> Result<(), DatabaseError> {
        match self.interactive.take(
            &id,
            &host_actor(options.actor),
            Some(options.end),
            options.admission,
        )? {
            Some(_) => Ok(()),
            None => Err(self.interactive.missing(&id)),
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
    /// The ID names no open transaction for the host.
    pub fn expect_node_version(
        &self,
        id: TransactionId,
        node_id: coordinode_core::graph::node::NodeId,
        expected: Option<u64>,
    ) -> Result<(), DatabaseError> {
        let key = coordinode_core::graph::node::encode_node_key(self.shard_id, node_id);
        // Not a statement: it adds nothing to the chain.
        let mut checked = self
            .interactive
            .check_out(&id, &ActingContext::Trusted, None)?;
        let stated = checked.state.expect_version(
            coordinode_storage::engine::partition::Partition::Node,
            &key,
            expected,
        );
        self.interactive
            .check_in(&id, checked.state, checked.chain)?;
        Ok(stated?)
    }

    /// End interactive transactions idle longer than `timeout` with their
    /// timeout outcome: roll them back, or commit those that commit at
    /// expiry, under the operation their start named. An open transaction
    /// pins an MVCC snapshot and buffers writes in memory, so an abandoned
    /// one would leak retention and leader memory. A transaction a command
    /// runs on, or has commands waiting, is never idle. Called
    /// opportunistically on `begin`; the server's reaper also runs it at the
    /// returned instant, so expiry does not wait for the next `begin`.
    ///
    /// Returns when the earliest transaction still open becomes idle for
    /// `timeout`, or `None` when none is open.
    pub fn reap_idle_transactions(&self, timeout: Duration) -> Option<Instant> {
        let now = Instant::now();
        let registry = &self.interactive;
        let mut contexts = registry.contexts.lock();
        let expired: Vec<TransactionId> = contexts
            .values()
            .filter(|context| {
                context.state.is_some()
                    && context.pending() == 0
                    && now.duration_since(context.touched) >= timeout
            })
            .map(|context| context.id)
            .collect();
        let mut commits = Vec::new();
        for id in &expired {
            let Some(context) = contexts.get_mut(id) else {
                continue;
            };
            if context.timeout_outcome == TimeoutOutcome::Rollback {
                registry.end_locked(&mut contexts, id, TransactionEnd::IdleExpired);
                continue;
            }
            // The commit takes the context's turn and keeps it in the
            // registry until its result is known: a command arriving
            // meanwhile waits behind it and is then told how it ended, never
            // that it committed before it did.
            let Some(state) = context.state.take() else {
                continue;
            };
            context.next_turn += 1;
            context.committing = true;
            commits.push((
                *id,
                Finished {
                    state,
                    chain: context.chain,
                    write_concern: context.write_concern,
                    operation: context.operation,
                },
            ));
        }
        let next = contexts
            .values()
            .filter(|context| !context.committing)
            .map(|context| context.touched + timeout)
            .min();
        drop(contexts);
        // Outside the registry's lock: a commit waits on consensus.
        for (id, finished) in commits {
            let operation = finished.operation;
            let result = self.commit_finished(id, finished, operation.as_ref(), &[]);
            if let Err(e) = &result {
                tracing::warn!(transaction = %id, error = %e, "commit at idle expiry failed");
            }
            let mut end = idle_commit_end(&result);
            // A successful answer can come before the commit is applied (a
            // write concern acknowledged at submission or at append). The
            // operation's record is applied in the commit's own log entry, so
            // only a committed record here proves the commit took effect.
            if end == TransactionEnd::IdleCommitted
                && !operation.as_ref().is_some_and(|op| {
                    matches!(
                        self.resolve_operation(op, crate::operation::Resolve::Lookup),
                        Ok(crate::operation::OperationOutcome::Committed { .. })
                    )
                })
            {
                end = TransactionEnd::IdleCommitUnknown;
            }
            // A context dropped with its tenure meanwhile is not recorded:
            // a command for it is told it is unknown, as a new leader tells
            // it, and its operation still resolves the commit.
            let mut contexts = registry.contexts.lock();
            if contexts.get(&id).is_some_and(|context| context.committing) {
                registry.end_locked(&mut contexts, &id, end);
            }
        }
        next
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
