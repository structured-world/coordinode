//! A host's calls on an embedded member: each command runs where it must,
//! here when this member may run it and at the leader otherwise, and the
//! host gets the same answer either way.
//!
//! An autonomous statement is checked and fenced here; one this member
//! cannot serve goes to the leader whole. Every command of a transaction,
//! and every call on operation records, runs at the leader. A command marked
//! leader-local is never routed: on a member that does not lead it fails with
//! [`DatabaseError::NotLeader`] naming the leader.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use coordinode_core::budget::{BudgetStop, CancelFlag};
use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_query::advisor::source::SourceContext;
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use super::admission::{Admission, Admitter, Requested, leader_hint};
use super::executor;
use super::routing::{
    CommandRouter, RoutedAnswer, RoutedBegin, RoutedCommit, RoutedOperationAnswer,
    RoutedOperationCall, RoutedRollback, RoutedStatement,
};
use crate::{Database, DatabaseError};

/// Where a command may run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Placement {
    /// Here when this member may run it, at the leader otherwise.
    #[default]
    Anywhere,
    /// Only here, and only while this member leads; never routed. A host
    /// whose writes must originate at the leader uses it, so a deposed
    /// writer's work cannot reach the new leader by another road.
    LeaderLocal,
}

/// An autonomous statement a host runs on a member.
#[derive(Debug, Clone)]
pub struct Statement {
    /// The statement's text.
    pub query: String,
    /// Its parameters.
    pub params: HashMap<String, Value>,
    /// The settings it names; the rest are the executing member's.
    pub requested: Requested,
    /// Who it acts for; a host acts as the trusted host by default.
    pub actor: ActingContext,
    /// Where in the host's code it was issued.
    pub source: Option<SourceContext>,
    /// Where it may run.
    pub placement: Placement,
}

impl Statement {
    /// `query` with no parameters, under the member's settings, acting as
    /// the trusted host and free to run wherever it must.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            params: HashMap::new(),
            requested: Requested::default(),
            actor: ActingContext::Trusted,
            source: None,
            placement: Placement::Anywhere,
        }
    }
}

/// What a command answered, and where it went to be answered.
#[derive(Debug, Clone, PartialEq)]
pub struct Routed<T> {
    /// The answer.
    pub value: T,
    /// The member that ran the command when it was not this one.
    pub forwarded_to: Option<u64>,
    /// The member this one knew to lead once the answer came back; `None`
    /// when it knew of none. Advisory: it can change at once, and it is not
    /// evidence of a commit.
    pub leader: Option<u64>,
}

/// Runs a host's commands on this member or routes them to the leader.
///
/// Cheap to clone: the admitter is shared.
#[derive(Clone)]
pub struct Ingress {
    admitter: Admitter,
}

impl Ingress {
    /// An ingress over `admitter`, the checks and fence of this member.
    pub fn new(admitter: Admitter) -> Self {
        Self { admitter }
    }

    /// The checks and fence this ingress admits statements with.
    pub fn admitter(&self) -> &Admitter {
        &self.admitter
    }

    fn database(&self) -> Arc<RwLock<Database>> {
        Arc::clone(self.admitter.database())
    }

    /// `value` as answered by `ran_at`, `None` for this member, with the
    /// leader this member knows of now: no round trip is spent on it.
    fn answered<T>(&self, value: T, ran_at: Option<u64>) -> Routed<T> {
        Routed {
            value,
            forwarded_to: ran_at,
            leader: self.admitter.node().and_then(|n| n.current_leader()),
        }
    }

    /// Run `statement` and return its answer.
    ///
    /// # Errors
    ///
    /// What the statement failed with wherever it ran; how the answer was
    /// lost when it ran at another member; [`DatabaseError::NotLeader`] when
    /// no member leads, or when a leader-local statement reached a member
    /// that does not.
    pub async fn execute(
        &self,
        statement: Statement,
    ) -> Result<Routed<RoutedAnswer>, DatabaseError> {
        if statement
            .requested
            .deadline
            .is_some_and(|d| d <= Instant::now())
        {
            return Err(DatabaseError::Budget(BudgetStop::Deadline));
        }
        let checked = self
            .admitter
            .check(&statement.query, &statement.requested)?;
        if statement.placement == Placement::LeaderLocal {
            self.require_leading()?;
        }
        let admitted = match self.admitter.fence(checked, false).await? {
            Admission::Run(admitted) => admitted,
            Admission::Forward(leader) => return self.route(leader, statement).await,
        };
        // A write that reaches a follower is refused by consensus before
        // anything is applied, and goes to the leader as if it had been
        // routed in the first place. Only then is the statement needed again,
        // so only a follower keeps a copy.
        let retry = (!admitted.served_by_leader && statement.placement == Placement::Anywhere)
            .then(|| statement.clone());
        let db = self.database();
        let Statement {
            query,
            params,
            source,
            ..
        } = statement;
        let ran = blocking(move || {
            executor::run_admitted(&db.read(), &query, params, source.as_ref(), &admitted)
        })
        .await;
        match (ran, retry) {
            (Err(e), Some(statement)) => match leader_hint(&e) {
                Some(leader) => self.route(leader, statement).await,
                None => Err(e),
            },
            (ran, _) => ran.map(|answer| self.answered(answer, None)),
        }
    }

    /// Route an autonomous statement to `leader`, unless it may not leave
    /// this member.
    async fn route(
        &self,
        leader: u64,
        statement: Statement,
    ) -> Result<Routed<RoutedAnswer>, DatabaseError> {
        let Statement {
            query,
            params,
            requested,
            actor,
            source,
            placement,
        } = statement;
        if placement == Placement::LeaderLocal {
            return Err(DatabaseError::NotLeader {
                leader_id: Some(leader),
            });
        }
        let routed = RoutedStatement {
            actor,
            query,
            params,
            settings: executor::routed_settings(&requested),
            transaction: None,
            operation: requested.operation,
            deadline: None,
            source,
            start: None,
        };
        let admitted = AdmissionSignal::new(|| {});
        let (answer, ran_at) = self
            .at_leader(leader, requested.deadline, |router| {
                let mut routed = routed.clone();
                let (cancel, admitted) = (requested.cancel.as_ref(), &admitted);
                async move {
                    // What is left of the caller's time when this attempt goes.
                    routed.deadline = time_left(requested.deadline)?;
                    cancellable(cancel, router.execute(routed, admitted)).await
                }
            })
            .await?;
        Ok(self.answered(answer, Some(ran_at)))
    }

    /// Run `call` at `leader`. A call proven not sent there
    /// ([`DatabaseError::PeerUnreachable`], such as a leader whose address
    /// this member has not learned yet) goes again to the leader known then,
    /// a bounded number of times within `deadline`; a call that may have run
    /// is never sent again. Returns the answer and the member that gave it.
    async fn at_leader<T, Fut>(
        &self,
        mut leader: u64,
        deadline: Option<Instant>,
        mut call: impl FnMut(CommandRouter) -> Fut,
    ) -> Result<(T, u64), DatabaseError>
    where
        Fut: std::future::Future<Output = Result<T, DatabaseError>>,
    {
        let mut pauses = REROUTE_PAUSES.iter();
        loop {
            let sent = match self.admitter.router(leader) {
                Ok(router) => call(router).await,
                Err(e) => Err(e),
            };
            let unsent = match sent {
                Err(e @ DatabaseError::PeerUnreachable { .. }) => e,
                other => return other.map(|answer| (answer, leader)),
            };
            let Some(pause) = pauses.next() else {
                return Err(unsent);
            };
            if deadline.is_some_and(|d| Instant::now() + *pause >= d) {
                return Err(unsent);
            }
            tokio::time::sleep(*pause).await;
            // This member leading now, or none leading, ends the retries:
            // the caller hears why the command was not sent.
            match self.admitter.transaction_leader() {
                Ok(Some(next)) => leader = next,
                Ok(None) | Err(_) => return Err(unsent),
            }
        }
    }

    /// Open a transaction under the ID the host drew, at the leader.
    ///
    /// # Errors
    ///
    /// Why it was not opened, or how the leader's answer was lost.
    pub async fn begin(
        &self,
        begin: RoutedBegin,
        placement: Placement,
    ) -> Result<Routed<()>, DatabaseError> {
        let ran_at = match self.leader_for(placement)? {
            None => {
                let db = self.database();
                blocking(move || executor::begin(&db.read(), begin)).await?;
                None
            }
            Some(leader) => {
                let ((), ran_at) = self
                    .at_leader(leader, None, |router| {
                        let begin = begin.clone();
                        async move { router.begin(begin).await }
                    })
                    .await?;
                Some(ran_at)
            }
        };
        Ok(self.answered((), ran_at))
    }

    /// Run a statement of a transaction, at the leader that holds it.
    /// `admitted` is told when the statement has its place in the
    /// transaction's order; `cancel` stops it.
    ///
    /// # Errors
    ///
    /// What the statement failed with, or how its answer was lost.
    pub async fn execute_in_transaction(
        &self,
        statement: RoutedStatement,
        admitted: &AdmissionSignal,
        cancel: &CancelFlag,
        placement: Placement,
    ) -> Result<Routed<RoutedAnswer>, DatabaseError> {
        let (answer, ran_at) = match self.leader_for(placement)? {
            None => {
                let db = self.database();
                let (admitted, cancel) = (admitted.clone(), cancel.clone());
                let answer = blocking(move || {
                    executor::run_in_transaction(&db.read(), statement, &cancel, &admitted)
                })
                .await?;
                (answer, None)
            }
            Some(leader) => {
                let (answer, ran_at) = self
                    .at_leader(leader, None, |router| {
                        let statement = statement.clone();
                        async move {
                            cancellable(Some(cancel), router.execute(statement, admitted)).await
                        }
                    })
                    .await?;
                (answer, Some(ran_at))
            }
        };
        Ok(self.answered(answer, ran_at))
    }

    /// Commit a transaction at the leader that holds it.
    ///
    /// # Errors
    ///
    /// What the commit failed with, or how its answer was lost: a lost
    /// answer is [`DatabaseError::PeerOutcomeUnknown`], resolved by the
    /// commit's operation.
    pub async fn commit(
        &self,
        commit: RoutedCommit,
        placement: Placement,
    ) -> Result<Routed<CommitReceipt>, DatabaseError> {
        let (receipt, ran_at) = match self.leader_for(placement)? {
            None => {
                let db = self.database();
                (
                    blocking(move || executor::commit(&db.read(), commit)).await?,
                    None,
                )
            }
            Some(leader) => {
                let (receipt, ran_at) = self
                    .at_leader(leader, None, |router| {
                        let commit = commit.clone();
                        async move { router.commit(commit).await }
                    })
                    .await?;
                (receipt, Some(ran_at))
            }
        };
        Ok(self.answered(receipt, ran_at))
    }

    /// Roll back a transaction at the leader that holds it.
    ///
    /// # Errors
    ///
    /// Why the rollback was refused, or how its answer was lost.
    pub async fn rollback(
        &self,
        rollback: RoutedRollback,
        placement: Placement,
    ) -> Result<Routed<()>, DatabaseError> {
        let ran_at = match self.leader_for(placement)? {
            None => {
                let db = self.database();
                blocking(move || executor::rollback(&db.read(), rollback)).await?;
                None
            }
            Some(leader) => {
                let ((), ran_at) = self
                    .at_leader(leader, None, |router| {
                        let rollback = rollback.clone();
                        async move { router.rollback(rollback).await }
                    })
                    .await?;
                Some(ran_at)
            }
        };
        Ok(self.answered((), ran_at))
    }

    /// Carry out a call on operation records at the leader, whose log
    /// decides them.
    ///
    /// # Errors
    ///
    /// Why the leader refused it, or how its answer was lost.
    pub async fn operation(
        &self,
        actor: ActingContext,
        call: RoutedOperationCall,
        placement: Placement,
    ) -> Result<Routed<RoutedOperationAnswer>, DatabaseError> {
        let (answer, ran_at) = match self.leader_for(placement)? {
            None => {
                let db = self.database();
                (
                    blocking(move || executor::run_operation(&db.read(), call)).await?,
                    None,
                )
            }
            Some(leader) => {
                let (answer, ran_at) = self
                    .at_leader(leader, None, |router| {
                        let (call, actor) = (call.clone(), &actor);
                        async move { router.operation(actor, call).await }
                    })
                    .await?;
                (answer, Some(ran_at))
            }
        };
        Ok(self.answered(answer, ran_at))
    }

    /// The member a command that needs the leader runs at: `None` for this
    /// one. A leader-local command on a member that does not lead is refused.
    fn leader_for(&self, placement: Placement) -> Result<Option<u64>, DatabaseError> {
        match self.admitter.transaction_leader()? {
            Some(leader) if placement == Placement::LeaderLocal => Err(DatabaseError::NotLeader {
                leader_id: Some(leader),
            }),
            leader => Ok(leader),
        }
    }

    /// Refuse unless this member leads.
    fn require_leading(&self) -> Result<(), DatabaseError> {
        self.leader_for(Placement::LeaderLocal).map(|_| ())
    }
}

/// Pauses before sending a command again that was proven not sent: a
/// member that just joined learns its leader's address within the first of
/// them, and the whole series stays well inside a statement's time.
const REROUTE_PAUSES: [std::time::Duration; 3] = [
    std::time::Duration::from_millis(25),
    std::time::Duration::from_millis(50),
    std::time::Duration::from_millis(100),
];

/// What is left until `deadline`; `None` without one.
///
/// # Errors
///
/// [`BudgetStop::Deadline`] once it has passed: nothing is routed that its
/// caller no longer waits for.
fn time_left(deadline: Option<Instant>) -> Result<Option<std::time::Duration>, DatabaseError> {
    match deadline {
        None => Ok(None),
        Some(deadline) => deadline
            .checked_duration_since(Instant::now())
            .filter(|left| !left.is_zero())
            .map(Some)
            .ok_or(DatabaseError::Budget(BudgetStop::Deadline)),
    }
}

/// Wait for `call` unless `cancel` is thrown first; a cancelled call drops
/// the request, and with it the statement at the other member.
async fn cancellable<T>(
    cancel: Option<&CancelFlag>,
    call: impl std::future::Future<Output = Result<T, DatabaseError>>,
) -> Result<T, DatabaseError> {
    let Some(cancel) = cancel else {
        return call.await;
    };
    tokio::select! {
        biased;
        () = cancel.thrown() => Err(DatabaseError::Budget(BudgetStop::Cancelled)),
        answered = call => answered,
    }
}

/// Run `call` on the blocking pool: a write commits through consensus inside
/// it.
async fn blocking<T: Send + 'static>(
    call: impl FnOnce() -> Result<T, DatabaseError> + Send + 'static,
) -> Result<T, DatabaseError> {
    tokio::task::spawn_blocking(call)
        .await
        .map_err(|e| DatabaseError::Other(format!("the command did not finish: {e}")))?
}
