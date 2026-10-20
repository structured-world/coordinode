//! The commands other members route to this one, run on the path this
//! member's own clients take: the same fence, execution, advisor and
//! refusals, so a client gets the same answer whichever member it reached.

use std::time::Instant;

use coordinode_core::budget::CancelFlag;
use coordinode_core::txn::command::{ActingContext, AdmissionSignal};
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_embed::DatabaseError;
use coordinode_embed::replication::executor;
use coordinode_embed::replication::routing::{
    AnswerWriter, CommandExecutor, RoutedBegin, RoutedCommit, RoutedOperationAnswer,
    RoutedOperationCall, RoutedRollback, RoutedStatement, RoutedStats,
};

use crate::services::statement::{Admission, StatementExecutor};

/// Runs routed commands through this member's [`StatementExecutor`]. A
/// member serves it to its peers with
/// [`coordinode_embed::replication::Member::add_command_routing`].
///
/// Only an autonomous statement takes the server's own path, which records
/// it for the advisor and the query metrics; every other command runs as an
/// embedded member runs it.
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
        answer: &mut AnswerWriter,
    ) -> Result<RoutedStats, DatabaseError> {
        if statement.transaction.is_some() {
            return executor::write_in_transaction(
                self.executor.database(),
                statement,
                cancel,
                admitted,
                answer,
            );
        }

        // A statement routed here needs this member to lead: it is fenced as
        // such and never routed on.
        let requested = executor::routed_requested(&statement, cancel);
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|e| DatabaseError::Other(format!("no runtime to fence the read on: {e}")))?;
        let fenced =
            match runtime.block_on(self.executor.admit(&statement.query, &requested, true))? {
                Admission::Run(fenced) => fenced,
                // An admission that may not route on never asks to.
                Admission::Forward(leader) => {
                    return Err(DatabaseError::NotLeader {
                        leader_id: Some(leader),
                    });
                }
            };
        let RoutedStatement {
            query,
            params,
            source,
            ..
        } = statement;
        let params = (!params.is_empty()).then_some(params);
        let start = Instant::now();
        // A read that pages is answered page by page, as a session here
        // pages it: one page of memory whatever the result size.
        if self.executor.database().read().keyset_pageable(&query) {
            let stats =
                executor::write_paged(self.executor.database(), &query, params, &fenced, answer);
            crate::services::statement::observe_query(
                if stats.is_ok() {
                    crate::services::statement::QueryKind::Read
                } else {
                    crate::services::statement::QueryKind::Failed
                },
                start,
            );
            self.executor.record(&query, source.as_ref(), start);
            return stats;
        }
        let result = self
            .executor
            .execute(&query, params, source.as_ref(), &fenced)?;
        self.executor.record(&query, source.as_ref(), start);
        executor::write_rows(&result.rows, answer);
        Ok(executor::stats(&result.write_stats, &fenced))
    }

    fn begin(&self, begin: RoutedBegin) -> Result<(), DatabaseError> {
        executor::begin(&self.executor.database().read(), begin)
    }

    fn commit(&self, commit: RoutedCommit) -> Result<CommitReceipt, DatabaseError> {
        executor::commit(&self.executor.database().read(), commit)
    }

    fn operation(
        &self,
        _actor: ActingContext,
        call: RoutedOperationCall,
    ) -> Result<RoutedOperationAnswer, DatabaseError> {
        executor::run_operation(&self.executor.database().read(), call)
    }

    fn rollback(&self, rollback: RoutedRollback) -> Result<(), DatabaseError> {
        executor::rollback(&self.executor.database().read(), rollback)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
