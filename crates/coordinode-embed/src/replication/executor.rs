//! What a member does with a command once it is the one to run it: the
//! library's [`CommandExecutor`] for an embedded member, and the steps a
//! server's executor shares with it.
//!
//! A command of a transaction, a transaction's begin, commit and rollback,
//! and a call on operation records run the same way on every member. Only
//! the autonomous statement differs between consumers: a server records it
//! for its advisor and metrics, an embedded member runs it as it is.

use std::collections::HashMap;

use coordinode_core::budget::CancelFlag;
use coordinode_core::graph::types::Value;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_query::executor::row::Row;
use coordinode_raft::read_fence::ReadPreference;

use super::admission::{Admission, Admitted, Admitter, Requested};
use super::routing::{
    CommandExecutor, RoutedAnswer, RoutedBegin, RoutedCommit, RoutedOperationAnswer,
    RoutedOperationCall, RoutedRollback, RoutedSettings, RoutedStatement, RoutedStats,
};
use crate::db::{
    CommitOptions, CypherResult, RollbackOptions, StatementOptions, TransactionOptions,
};
use crate::{Database, DatabaseError};

/// The column header and the rows in column order: the keys of the first row
/// (every row has the same), and a missing value as `Null` so each row is as
/// wide as the header. No rows have no columns.
pub fn columns_and_values(rows: &[Row]) -> (Vec<String>, Vec<Vec<Value>>) {
    let columns: Vec<String> = rows
        .first()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    let values = rows
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| row.get(col).cloned().unwrap_or(Value::Null))
                .collect()
        })
        .collect();
    (columns, values)
}

/// The answer of an autonomous statement `admitted` ran with `result`.
pub fn answer(result: &CypherResult, admitted: &Admitted) -> RoutedAnswer {
    let (columns, rows) = columns_and_values(&result.rows);
    let ws = &result.write_stats;
    RoutedAnswer {
        columns,
        rows,
        stats: RoutedStats {
            nodes_created: ws.nodes_created as i64,
            nodes_deleted: ws.nodes_deleted as i64,
            edges_created: ws.edges_created as i64,
            edges_deleted: ws.edges_deleted as i64,
            properties_set: ws.properties_set as i64,
            applied_index: ws.applied_index.unwrap_or(0),
            commit_ts: ws.commit_ts.unwrap_or(0),
            read_index: admitted.applied_index,
            read_as_of_ts: admitted.read_as_of_ts,
        },
    }
}

/// The settings a statement routed from here runs under: what the statement
/// and its session named. A setting neither named is the executing member's
/// to choose, as it would be for a client connected there.
pub fn routed_settings(requested: &Requested) -> RoutedSettings {
    RoutedSettings {
        read_concern: requested.read_concern,
        after_index: (requested.after_index > 0).then_some(requested.after_index),
        at_timestamp: (requested.at_timestamp > 0).then_some(requested.at_timestamp),
        write_concern: requested.write_concern,
        vector_consistency: requested.vector_consistency,
        vector_build_wait: requested.vector_build_wait,
        query_memory_limit: requested.query_memory_limit,
    }
}

/// The settings an autonomous statement routed here is admitted with: the
/// ones its ingress resolved, read from the leader, which this member must
/// be. Never routed on.
pub fn routed_requested(statement: &RoutedStatement, cancel: &CancelFlag) -> Requested {
    let settings = &statement.settings;
    Requested {
        read_concern: settings.read_concern,
        after_index: settings.after_index.unwrap_or(0),
        at_timestamp: settings.at_timestamp.unwrap_or(0),
        read_preference: Some(ReadPreference::Primary),
        write_concern: settings.write_concern,
        vector_consistency: settings.vector_consistency,
        vector_build_wait: settings.vector_build_wait,
        query_memory_limit: settings.query_memory_limit,
        deadline: deadline_after(statement.deadline),
        cancel: Some(cancel.clone()),
        operation: statement.operation,
    }
}

/// The instant `left` from now; a span past what the clock can name bounds
/// nothing.
fn deadline_after(left: Option<std::time::Duration>) -> Option<std::time::Instant> {
    left.and_then(|left| std::time::Instant::now().checked_add(left))
}

/// Run a statement of a transaction this member holds: it reads at the
/// transaction's pinned snapshot and buffers its writes, with no fence and no
/// commit of its own. `admitted` is told when it has its place in the
/// transaction's order.
///
/// # Errors
///
/// What the statement failed with; a transaction this member does not hold
/// is not found.
pub fn run_in_transaction(
    db: &Database,
    statement: RoutedStatement,
    cancel: &CancelFlag,
    admitted: &AdmissionSignal,
) -> Result<RoutedAnswer, DatabaseError> {
    let RoutedStatement {
        actor,
        query,
        params,
        settings,
        transaction,
        deadline,
        ..
    } = statement;
    let Some(id) = transaction else {
        return Err(DatabaseError::Other(
            "a statement outside a transaction is not run as one".to_owned(),
        ));
    };
    let options = StatementOptions {
        vector_consistency: settings.vector_consistency,
        vector_build_wait: settings.vector_build_wait,
        query_memory_limit: settings.query_memory_limit,
        deadline: deadline_after(deadline),
        cancel: Some(cancel.clone()),
        actor: Some(actor),
        admission: Some(admitted.clone()),
        ..StatementOptions::default()
    };
    let params = (!params.is_empty()).then_some(params);
    let rows = db.execute_in_transaction_with(id, &query, params, &options)?;
    let (columns, rows) = columns_and_values(&rows);
    Ok(RoutedAnswer {
        columns,
        rows,
        stats: RoutedStats::default(),
    })
}

/// Open a transaction under the ID the client drew.
///
/// # Errors
///
/// Why the transaction was not opened: this member does not lead, the ID is
/// taken, the settings are refused.
pub fn begin(db: &Database, begin: RoutedBegin) -> Result<(), DatabaseError> {
    db.begin_transaction_as(
        begin.transaction,
        &TransactionOptions {
            actor: Some(begin.actor),
            read_concern: begin.read_concern,
            write_concern: begin.write_concern,
            timeout_outcome: begin.timeout_outcome,
            operation: begin.operation,
        },
    )
}

/// Commit a transaction this member holds.
///
/// # Errors
///
/// What the commit failed with.
pub fn commit(db: &Database, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
    db.commit_transaction_with(
        commit.transaction,
        &CommitOptions {
            operation: commit.operation.as_ref(),
            actor: Some(&commit.actor),
            admission: None,
            expect: &commit.expect,
        },
    )
}

/// Roll back a transaction this member holds.
///
/// # Errors
///
/// Why the rollback was refused.
pub fn rollback(db: &Database, rollback: RoutedRollback) -> Result<(), DatabaseError> {
    db.rollback_transaction_with(
        rollback.transaction,
        &RollbackOptions {
            actor: Some(&rollback.actor),
            admission: None,
            end: rollback.end(),
        },
    )
}

/// Carry out `call` on `db`'s operation records: what a member does with the
/// call whether its client reached it or another member passed it here.
///
/// # Errors
///
/// What the database refused the call with; a member that does not lead
/// refuses every call that changes the records.
pub fn run_operation(
    db: &Database,
    call: RoutedOperationCall,
) -> Result<RoutedOperationAnswer, DatabaseError> {
    Ok(match call {
        RoutedOperationCall::Enroll(issuer) => {
            let enrolled = db.enroll_issuer(issuer)?;
            RoutedOperationAnswer::Enrolled {
                incarnation: enrolled.incarnation(),
                next_sequence: enrolled.next_sequence(),
                deployment: enrolled.deployment(),
            }
        }
        RoutedOperationCall::Resolve(descriptor, resolve) => {
            RoutedOperationAnswer::Outcome(db.resolve_operation(&descriptor, resolve)?)
        }
        RoutedOperationCall::Acknowledge(descriptors) => {
            for descriptor in &descriptors {
                db.acknowledge_operation(descriptor)?;
            }
            RoutedOperationAnswer::Done
        }
        RoutedOperationCall::Retire(below) => {
            db.retire_operations(&below)?;
            RoutedOperationAnswer::Done
        }
    })
}

/// Run an admitted autonomous statement on `db` and answer with its result.
///
/// # Errors
///
/// What the statement failed with.
pub fn run_admitted(
    db: &Database,
    query: &str,
    params: HashMap<String, Value>,
    source: Option<&coordinode_query::advisor::source::SourceContext>,
    admitted: &Admitted,
) -> Result<RoutedAnswer, DatabaseError> {
    let params = (!params.is_empty()).then_some(params);
    let result = db.execute_cypher_shared_with(query, params, source, &admitted.options())?;
    Ok(answer(&result, admitted))
}

/// Runs the commands other members route to an embedded member, on the path
/// the member's own host calls take: the same checks, fence and refusals.
pub struct MemberCommandExecutor {
    admitter: Admitter,
    runtime: tokio::runtime::Handle,
}

impl MemberCommandExecutor {
    /// An executor over `admitter`, fencing reads on `runtime`.
    pub fn new(admitter: Admitter, runtime: tokio::runtime::Handle) -> Self {
        Self { admitter, runtime }
    }
}

impl CommandExecutor for MemberCommandExecutor {
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, DatabaseError> {
        if statement.transaction.is_some() {
            return run_in_transaction(
                &self.admitter.database().read(),
                statement,
                cancel,
                admitted,
            );
        }
        // A statement routed here needs this member to lead: it is fenced as
        // such and never routed on.
        let requested = routed_requested(&statement, cancel);
        let fenced =
            match self
                .runtime
                .block_on(self.admitter.admit(&statement.query, &requested, true))?
            {
                Admission::Run(fenced) => fenced,
                // An admission that may not route on never asks to.
                Admission::Forward(leader) => {
                    return Err(DatabaseError::NotLeader {
                        leader_id: Some(leader),
                    });
                }
            };
        run_admitted(
            &self.admitter.database().read(),
            &statement.query,
            statement.params,
            statement.source.as_ref(),
            &fenced,
        )
    }

    fn begin(&self, call: RoutedBegin) -> Result<(), DatabaseError> {
        begin(&self.admitter.database().read(), call)
    }

    fn commit(&self, call: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
        commit(&self.admitter.database().read(), call)
    }

    fn rollback(&self, call: RoutedRollback) -> Result<(), DatabaseError> {
        rollback(&self.admitter.database().read(), call)
    }

    fn operation(
        &self,
        _actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, DatabaseError> {
        run_operation(&self.admitter.database().read(), call)
    }
}
