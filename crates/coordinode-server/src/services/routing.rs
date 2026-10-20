//! The commands other members route to this one, run on the path this
//! member's own clients take: the same fence, execution, advisor and
//! refusals, so a client gets the same answer whichever member it reached.

use std::time::Instant;

use coordinode_core::budget::CancelFlag;
use coordinode_core::operations::TxnOrdering;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_embed::db::{CommitOptions, RollbackOptions, StatementOptions, TransactionOptions};
use coordinode_embed::replication::routing::{
    CommandExecutor, RoutedAnswer, RoutedBegin, RoutedCommit, RoutedOperationAnswer,
    RoutedOperationCall, RoutedRollback, RoutedSettings, RoutedStatement, RoutedStats,
};
use coordinode_embed::{Database, DatabaseError};
use coordinode_raft::read_fence::ReadPreference;
use tonic::Status;

use crate::services::cypher::db_error_to_status;
use crate::services::statement::{Admission, Requested, StatementExecutor, columns_and_values};

/// Runs routed commands through this member's [`StatementExecutor`]. A
/// member serves it to its peers with
/// [`coordinode_embed::replication::Member::add_command_routing`].
pub struct ServerCommandExecutor {
    executor: StatementExecutor,
}

impl ServerCommandExecutor {
    /// An executor running routed commands through `executor`, the one this
    /// member's own clients are served by.
    pub fn new(executor: StatementExecutor) -> Self {
        Self { executor }
    }
}

impl CommandExecutor for ServerCommandExecutor {
    fn execute(
        &self,
        statement: RoutedStatement,
        cancel: &CancelFlag,
        admitted: &AdmissionSignal,
    ) -> Result<RoutedAnswer, Status> {
        let RoutedStatement {
            actor,
            query,
            params,
            settings,
            transaction,
            start_transaction,
            nonce,
            operation,
            deadline,
            source,
        } = statement;
        let params = (!params.is_empty()).then_some(params);
        let deadline = deadline.map(|left| Instant::now() + left);

        // A transaction's statement reads at its pinned snapshot and buffers
        // its writes: no fence and no commit of its own. The start opens the
        // transaction under the client's ID first.
        if let Some(id) = transaction {
            let db = self.executor.database().read();
            if start_transaction {
                db.begin_transaction_as(
                    id,
                    &TransactionOptions {
                        actor: Some(actor.clone()),
                        read_concern: settings.read_concern,
                        write_concern: settings.write_concern,
                        ordering: TxnOrdering::Unordered,
                        drain_timeout: None,
                    },
                )
                .map_err(db_error_to_status)?;
            }
            let options = StatementOptions {
                vector_consistency: settings.vector_consistency,
                vector_build_wait: settings.vector_build_wait,
                query_memory_limit: settings.query_memory_limit,
                deadline,
                cancel: Some(cancel.clone()),
                actor: Some(actor),
                nonce,
                admission: Some(admitted.clone()),
                ..StatementOptions::default()
            };
            let rows = db
                .execute_in_transaction_with(id, &query, params, &options)
                .map_err(db_error_to_status)?;
            let (columns, rows) = columns_and_values(&rows);
            return Ok(RoutedAnswer {
                columns,
                rows,
                stats: RoutedStats::default(),
            });
        }

        // A statement routed here needs this member to lead: it is fenced as
        // such and never routed on.
        let requested = Requested {
            read_concern: settings.read_concern,
            after_index: settings.after_index.unwrap_or(0),
            at_timestamp: settings.at_timestamp.unwrap_or(0),
            read_preference: Some(ReadPreference::Primary),
            write_concern: settings.write_concern,
            vector_consistency: settings.vector_consistency,
            vector_build_wait: settings.vector_build_wait,
            query_memory_limit: settings.query_memory_limit,
            deadline,
            cancel: Some(cancel.clone()),
            operation,
        };
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|e| Status::internal(format!("no runtime to fence the read on: {e}")))?;
        let fenced = match runtime.block_on(self.executor.admit(&query, &requested, true))? {
            Admission::Run(fenced) => fenced,
            // An admission that may not route on never asks to.
            Admission::Forward(leader) => {
                return Err(Status::failed_precondition(format!(
                    "not the leader; leader is node {leader}"
                )));
            }
        };
        let start = Instant::now();
        let result = self
            .executor
            .execute(&query, params, source.as_ref(), &fenced)
            .map_err(db_error_to_status)?;
        self.executor.record(&query, source.as_ref(), start);
        let (columns, rows) = columns_and_values(&result.rows);
        let ws = &result.write_stats;
        Ok(RoutedAnswer {
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
                read_index: fenced.applied_index,
                read_as_of_ts: fenced.read_as_of_ts,
            },
        })
    }

    fn begin(&self, begin: RoutedBegin) -> Result<(), Status> {
        self.executor
            .database()
            .read()
            .begin_transaction_as(
                begin.transaction,
                &TransactionOptions {
                    actor: Some(begin.actor),
                    read_concern: begin.read_concern,
                    write_concern: begin.write_concern,
                    ordering: begin.ordering,
                    drain_timeout: begin.drain_timeout,
                },
            )
            .map_err(db_error_to_status)
    }

    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, Status> {
        self.executor
            .database()
            .read()
            .commit_transaction_with(
                commit.transaction,
                &CommitOptions {
                    operation: commit.operation.as_ref(),
                    actor: Some(&commit.actor),
                    last_nonce: commit.last_nonce,
                    admission: None,
                    expect: &commit.expect,
                },
            )
            .map_err(db_error_to_status)
    }

    fn operation(
        &self,
        _actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, Status> {
        run_operation(&self.executor.database().read(), call).map_err(db_error_to_status)
    }

    fn rollback(&self, rollback: RoutedRollback) -> Result<(), Status> {
        self.executor
            .database()
            .read()
            .rollback_transaction_with(
                rollback.transaction,
                &RollbackOptions {
                    actor: Some(&rollback.actor),
                    admission: None,
                },
            )
            .map_err(db_error_to_status)
    }
}

/// Carry out `call` on `db`'s operation records: what a member does with the
/// call whether its client reached it or another member passed it here.
///
/// # Errors
///
/// What the database refused the call with; a member that does not lead
/// refuses every call that changes the records.
pub(crate) fn run_operation(
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

/// The settings a statement routed from here runs under: what the statement
/// and its session named. A setting neither named is the executing member's
/// to choose, as it would be for a client connected there.
pub(crate) fn routed_settings(requested: &Requested) -> RoutedSettings {
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
