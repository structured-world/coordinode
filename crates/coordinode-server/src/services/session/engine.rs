//! Database-backed [`CursorEngine`] for the gRPC session binding.
//!
//! Opens a server-side cursor over the embedded [`Database`]. Two cursor shapes
//! back the same [`QueryCursor`] protocol:
//!
//! - **Keyset** ([`KeysetCursor`]): for a non-blocking single-`NodeScan` plan
//!   (read, auto-commit). The cursor pins one MVCC snapshot and pages by storage
//!   key, so memory stays `O(batch)` no matter the result size and the snapshot
//!   is stable for the cursor's life.
//! - **Materialize-once** ([`MaterializedCursor`]): for everything else, a
//!   blocking operator (sort, aggregate, `DISTINCT`), a multi-source plan
//!   (traverse, join, union), a bounded `LIMIT`/`SKIP`, or an interactive
//!   transaction. The plan runs to completion and the rows page out of memory.
//!
//! [`Database::keyset_pageable`] is the classifier that routes between them.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use coordinode_core::graph::types::Value;
use coordinode_core::txn::transaction::CommitReceipt;
use coordinode_embed::db::{SessionSetting, StatementOptions};
use coordinode_embed::{Database, DatabaseError};
use coordinode_query::advisor::source::{self, SourceContext, grpc_keys};
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::WriteStats;
use coordinode_session::{
    ConnectionSettings, CursorEngine, EngineError, QueryCursor, SessionStats, StatementSource,
};
// no-std: spin::RwLock (drop-in).
use parking_lot::RwLock;

use super::failure;
use crate::proto::{query, replication};
use crate::services::cypher::{
    build_wait_ms, db_error_to_status, proto_to_value_pub, read_concern_level, read_preference,
    value_to_proto_pub, vector_consistency_to_proto, write_concern_to_proto,
};
use crate::services::statement::{Admission, Requested, StatementExecutor, leader_hint};

/// Storage keys scanned per keyset page. Independent of the client's batch size:
/// a page may yield fewer output rows than this when a `Filter` rejects some, so
/// the cursor refills across pages until the client's batch is full or the scan
/// is exhausted.
const KEYSET_PAGE: usize = 1024;

/// A [`CursorEngine`] that runs statements through the embedded [`Database`],
/// on the same path as the unary RPC: settings, fence, forwarding, execution
/// and the advisor.
pub struct DatabaseCursorEngine {
    executor: StatementExecutor,
}

impl DatabaseCursorEngine {
    /// An engine that runs statements through `executor`.
    pub fn from_executor(executor: StatementExecutor) -> Self {
        Self { executor }
    }

    fn database(&self) -> &Arc<RwLock<Database>> {
        self.executor.database()
    }

    /// Check and fence an autonomous statement. The fence waits on the
    /// consensus node, which only a cluster has: the session calls in from
    /// the blocking pool, where waiting on the runtime is allowed.
    fn admit(&self, query: &str, requested: &Requested) -> Result<Admission, EngineError> {
        let checked = self
            .executor
            .check(query, requested)
            .map_err(|s| EngineError(failure(&s)))?;
        if !self.executor.needs_fence() {
            return Ok(Admission::Run(self.executor.admit_standalone(checked)));
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|e| EngineError::internal(format!("no runtime to fence the read on: {e}")))?;
        runtime
            .block_on(self.executor.fence(checked, false))
            .map_err(|s| EngineError(failure(&s)))
    }

    /// Run `query` at the leader and page its answer out of memory.
    fn forwarded(
        &self,
        leader_id: u64,
        query: &str,
        params: Option<HashMap<String, Value>>,
        settings: &ConnectionSettings,
        source: Option<&SourceContext>,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        let request = forwarded_request(query, params, settings);
        let runtime = tokio::runtime::Handle::try_current().map_err(|e| {
            EngineError::internal(format!("no runtime to reach the leader on: {e}"))
        })?;
        let response = runtime
            // A session statement carries no call deadline; its stream lives on.
            .block_on(self.executor.forward(leader_id, request, source, None))
            .map_err(|s| EngineError(failure(&s)))?
            .into_inner();
        Ok(Box::new(MaterializedCursor {
            rows: response
                .rows
                .iter()
                .map(|row| row.values.iter().map(proto_to_value_pub).collect())
                .collect(),
            columns: response.columns,
            pos: 0,
            stats: response.stats.map(stats_from_proto).unwrap_or_default(),
        }))
    }
}

impl CursorEngine for DatabaseCursorEngine {
    fn session_setting(&self, query: &str) -> Option<Result<ConnectionSettings, EngineError>> {
        Database::parse_session_set(query).map(|setting| match setting {
            Ok(SessionSetting::VectorConsistency(mode)) => Ok(ConnectionSettings {
                vector_consistency: Some(mode),
                ..ConnectionSettings::default()
            }),
            Ok(SessionSetting::VectorBuildWait(wait)) => Ok(ConnectionSettings {
                vector_build_wait: Some(wait),
                ..ConnectionSettings::default()
            }),
            Ok(SessionSetting::QueryMemoryLimit(bytes)) => Ok(ConnectionSettings {
                query_memory_limit: Some(bytes),
                ..ConnectionSettings::default()
            }),
            Err(refused) => Err(EngineError(failure(
                &crate::services::cypher::db_error_to_status(refused.into()),
            ))),
        })
    }

    fn open_cursor(
        &self,
        query: &str,
        params: HashMap<String, Value>,
        txid: u64,
        settings: &ConnectionSettings,
        source: Option<&StatementSource>,
        cancel: &coordinode_core::budget::CancelFlag,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };
        let source = source.and_then(source_context);
        // A statement of an interactive transaction reads at the snapshot
        // pinned when it began and buffers its writes until the commit: no
        // fence and no commit of its own here. Its session's vector settings
        // still apply.
        if txid != 0 {
            let options = StatementOptions {
                vector_consistency: settings.vector_consistency,
                vector_build_wait: settings.vector_build_wait,
                query_memory_limit: settings.query_memory_limit,
                cancel: Some(cancel.clone()),
                ..StatementOptions::default()
            };
            let rows = self
                .database()
                .read()
                .execute_in_transaction_with(txid, query, params, &options)
                .map_err(engine_error)?;
            let (columns, rows) = rows_to_values(&rows);
            return Ok(Box::new(MaterializedCursor {
                columns,
                rows,
                pos: 0,
                stats: SessionStats::default(),
            }));
        }

        let start = Instant::now();
        let admitted = match self.admit(query, &requested(settings, cancel))? {
            Admission::Run(admitted) => admitted,
            Admission::Forward(leader_id) => {
                return self.forwarded(leader_id, query, params, settings, source.as_ref());
            }
        };
        let fenced = SessionStats {
            applied_index: admitted.applied_index,
            served_by_leader: admitted.served_by_leader,
            read_as_of_ts: admitted.read_as_of_ts,
            ..SessionStats::default()
        };

        // Keyset path: an auto-commit read whose plan pages by a single
        // NodeScan pins one snapshot and pages by key, so memory stays one
        // page whatever the result size.
        if self.database().read().keyset_pageable(query) {
            let cursor = KeysetCursor::open(
                Arc::clone(self.database()),
                query.to_string(),
                params,
                admitted.read_concern.at_timestamp,
                fenced,
            )?;
            // Counted at its first page; later pages are served as the
            // client asks for them.
            crate::services::statement::observe_query(
                crate::services::statement::QueryKind::Read,
                start,
            );
            self.executor.record(query, source.as_ref(), start);
            return Ok(Box::new(cursor));
        }

        // A write that reaches a follower is refused by consensus and goes to
        // the leader, as on the unary path: nothing was applied here. Only a
        // follower needs the parameters again for that, so only a follower
        // keeps a copy; on the leader, a leadership lost mid-statement is
        // answered with NOT_LEADER and the leader to retry at.
        let retry =
            (self.executor.needs_fence() && !admitted.served_by_leader).then(|| params.clone());
        let result = match self
            .executor
            .execute(query, params, source.as_ref(), &admitted)
        {
            Ok(result) => result,
            Err(e) => match (leader_hint(&e), retry) {
                (Some(leader_id), Some(params)) => {
                    return self.forwarded(leader_id, query, params, settings, source.as_ref());
                }
                _ => return Err(engine_error(e)),
            },
        };
        self.executor.record(query, source.as_ref(), start);
        let (columns, rows) = rows_to_values(&result.rows);
        let written = write_stats(&result.write_stats);
        Ok(Box::new(MaterializedCursor {
            columns,
            rows,
            pos: 0,
            stats: SessionStats {
                // A replicated write reports its own committed index, a read
                // the index the fence saw.
                applied_index: result
                    .write_stats
                    .applied_index
                    .unwrap_or(fenced.applied_index),
                execution_time_ms: start.elapsed().as_millis() as i64,
                served_by_leader: fenced.served_by_leader,
                read_as_of_ts: fenced.read_as_of_ts,
                ..written
            },
        }))
    }

    fn begin_transaction(&self) -> Result<u64, EngineError> {
        Ok(self.database().read().begin_transaction())
    }

    fn commit_transaction(&self, txid: u64) -> Result<CommitReceipt, EngineError> {
        self.database()
            .read()
            .commit_transaction(txid)
            .map_err(engine_error)
    }

    fn rollback_transaction(&self, txid: u64) -> Result<(), EngineError> {
        self.database()
            .read()
            .rollback_transaction(txid)
            .map_err(engine_error)
    }
}

/// A database failure as the session reports it: the same status, reason and
/// details the unary path answers with, so a client of either path retries,
/// redirects or gives up on the same signal.
fn engine_error(err: DatabaseError) -> EngineError {
    EngineError(failure(&db_error_to_status(err)))
}

/// The advisor's view of where a session statement came from, read the way
/// the unary path reads its metadata: a statement with no file has no
/// location, and a Windows path counts as the same one written with `/`.
fn source_context(source: &StatementSource) -> Option<SourceContext> {
    let get = |key: &str| -> Option<String> {
        let value = match key {
            grpc_keys::FILE => source.file.clone(),
            grpc_keys::LINE => source.line.to_string(),
            grpc_keys::FUNCTION => source.function.clone(),
            grpc_keys::APP => source.app.clone(),
            grpc_keys::VERSION => source.version.clone(),
            _ => return None,
        };
        (!value.is_empty()).then_some(value)
    };
    source::extract_from_map(
        &get,
        grpc_keys::FILE,
        grpc_keys::LINE,
        grpc_keys::FUNCTION,
        grpc_keys::APP,
        grpc_keys::VERSION,
    )
}

/// The settings a session statement runs under, as the executor takes them.
/// A level or preference of zero is the wire's "unspecified": the server's
/// default, like a setting left out. The binding admits only values the
/// server knows into a session's settings, so none is refused here. `cancel`
/// is the switch the session's Cancel throws.
fn requested(
    settings: &ConnectionSettings,
    cancel: &coordinode_core::budget::CancelFlag,
) -> Requested {
    Requested {
        read_concern: settings
            .read_concern
            .and_then(|level| read_concern_level("", i32::from(level)).ok().flatten()),
        after_index: settings.after_index.unwrap_or(0),
        at_timestamp: settings.at_timestamp.unwrap_or(0),
        read_preference: settings
            .read_preference
            .and_then(|preference| read_preference("", i32::from(preference)).ok().flatten()),
        write_concern: settings.write_concern,
        vector_consistency: settings.vector_consistency,
        vector_build_wait: settings.vector_build_wait,
        query_memory_limit: settings.query_memory_limit,
        // A session statement has no call deadline of its own; the session's
        // stream lives on, and Cancel stops a statement.
        deadline: None,
        cancel: Some(cancel.clone()),
    }
}

/// The unary request that carries a session statement to the leader, with the
/// settings it runs under spelled out: the leader does not know the session.
fn forwarded_request(
    query: &str,
    params: Option<HashMap<String, Value>>,
    settings: &ConnectionSettings,
) -> query::ExecuteCypherRequest {
    let read_concern = (settings.read_concern.is_some()
        || settings.after_index.is_some()
        || settings.at_timestamp.is_some())
    .then(|| replication::ReadConcern {
        level: settings.read_concern.map_or(0, i32::from),
        after_index: settings.after_index.unwrap_or(0),
        at_timestamp: settings.at_timestamp.unwrap_or(0),
    });
    query::ExecuteCypherRequest {
        query: query.to_string(),
        parameters: params
            .unwrap_or_default()
            .iter()
            .map(|(k, v)| (k.clone(), value_to_proto_pub(v)))
            .collect(),
        read_preference: settings.read_preference.map_or(0, i32::from),
        read_concern,
        write_concern: settings.write_concern.as_ref().map(write_concern_to_proto),
        transaction_id: 0,
        vector_consistency: settings
            .vector_consistency
            .map_or(0, vector_consistency_to_proto),
        vector_build_wait_ms: settings.vector_build_wait.map(build_wait_ms),
        query_memory_limit_mb: settings
            .query_memory_limit
            .and_then(|bytes| u32::try_from(bytes >> 20).ok()),
    }
}

fn stats_from_proto(stats: query::QueryStats) -> SessionStats {
    SessionStats {
        nodes_created: stats.nodes_created,
        nodes_deleted: stats.nodes_deleted,
        edges_created: stats.edges_created,
        edges_deleted: stats.edges_deleted,
        properties_set: stats.properties_set,
        execution_time_ms: stats.execution_time_ms,
        applied_index: stats.applied_index,
        served_by_leader: stats.served_by_leader,
        commit_ts: stats.commit_ts,
        read_as_of_ts: stats.read_as_of_ts,
    }
}

/// A keyset-resumable cursor: pins one MVCC snapshot and pages the result by
/// storage key through [`Database::execute_cypher_paged`].
///
/// The first non-empty page is prefetched at open so [`columns`](QueryCursor::columns)
/// is known before the first batch; `read_ts` is then echoed into every later
/// page so the whole scan reads against the same snapshot.
struct KeysetCursor {
    database: Arc<RwLock<Database>>,
    query: String,
    params: Option<HashMap<String, Value>>,
    columns: Vec<String>,
    /// Pinned snapshot timestamp, set from the first page.
    read_ts: Option<u64>,
    /// Keyset resume token for the next page (`None` once exhausted).
    resume: Option<Vec<u8>>,
    exhausted: bool,
    pending: VecDeque<Vec<Value>>,
    stats: SessionStats,
    /// What the fence learned: the applied index the read was served at,
    /// whether the leader served it, and what a read-only member's read is as
    /// of. Every page reads the snapshot pinned when the cursor opened, so it
    /// holds for the whole scan.
    fenced: SessionStats,
    /// Time spent reading pages so far.
    busy: std::time::Duration,
}

impl KeysetCursor {
    /// Open the cursor, prefetching pages until the first row is found (so the
    /// column header is known) or the scan is exhausted. `at` pins the scan to
    /// that snapshot; `None` pins a fresh one.
    fn open(
        database: Arc<RwLock<Database>>,
        query: String,
        params: Option<HashMap<String, Value>>,
        at: Option<u64>,
        fenced: SessionStats,
    ) -> Result<Self, EngineError> {
        let mut cursor = Self {
            database,
            query,
            params,
            columns: Vec::new(),
            read_ts: at,
            resume: None,
            exhausted: false,
            pending: VecDeque::new(),
            stats: fenced.clone(),
            fenced,
            busy: std::time::Duration::ZERO,
        };
        // Drive the scan until columns are established. A heavy Filter can empty
        // a leading page while later pages still produce rows, so loop rather
        // than trust the first page alone.
        while cursor.columns.is_empty() && !cursor.exhausted {
            cursor.fetch_page()?;
        }
        Ok(cursor)
    }

    /// Fetch the next keyset page, extend `pending`, and advance the resume
    /// token + exhaustion flag. Establishes `columns` from the first row seen.
    fn fetch_page(&mut self) -> Result<(), EngineError> {
        let started = Instant::now();
        let page = self
            .database
            .read()
            .execute_cypher_paged(
                &self.query,
                self.params.clone(),
                self.read_ts,
                self.resume.clone(),
                KEYSET_PAGE,
            )
            .map_err(engine_error)?;
        self.busy += started.elapsed();
        self.read_ts = Some(page.read_ts);
        self.resume = page.last_key;
        self.exhausted = page.exhausted;
        self.stats = SessionStats {
            applied_index: self.fenced.applied_index,
            served_by_leader: self.fenced.served_by_leader,
            read_as_of_ts: self.fenced.read_as_of_ts,
            execution_time_ms: self.busy.as_millis() as i64,
            ..write_stats(&page.write_stats)
        };
        if self.columns.is_empty() {
            if let Some(first) = page.rows.first() {
                self.columns = first.keys().cloned().collect();
            }
        }
        for row in &page.rows {
            self.pending.push_back(project_row(row, &self.columns));
        }
        Ok(())
    }
}

impl QueryCursor for KeysetCursor {
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    fn next_batch(&mut self, max: usize) -> Result<Vec<Vec<Value>>, EngineError> {
        // Refill until the batch is full or the scan is drained. Filters may
        // make a page yield fewer rows than it scanned, so several pages can be
        // needed to satisfy one batch.
        while self.pending.len() < max && !self.exhausted {
            self.fetch_page()?;
        }
        let n = max.min(self.pending.len());
        Ok(self.pending.drain(..n).collect())
    }

    fn stats(&self) -> SessionStats {
        self.stats.clone()
    }
}

/// Derive the column header and per-row value vectors from executor rows.
///
/// Columns are the keys of the first row (all rows share keys); the column list
/// is empty for an empty result, matching the unary query path.
fn rows_to_values(rows: &[Row]) -> (Vec<String>, Vec<Vec<Value>>) {
    let columns: Vec<String> = rows
        .first()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    let values: Vec<Vec<Value>> = rows.iter().map(|row| project_row(row, &columns)).collect();
    (columns, values)
}

/// Project a single executor row onto the established column order, filling
/// missing columns with `Null` so every page's row width matches the header.
fn project_row(row: &Row, columns: &[String]) -> Vec<Value> {
    columns
        .iter()
        .map(|col| row.get(col).cloned().unwrap_or(Value::Null))
        .collect()
}

fn write_stats(ws: &WriteStats) -> SessionStats {
    SessionStats {
        nodes_created: ws.nodes_created as i64,
        nodes_deleted: ws.nodes_deleted as i64,
        edges_created: ws.edges_created as i64,
        edges_deleted: ws.edges_deleted as i64,
        properties_set: ws.properties_set as i64,
        // The version of what this statement wrote, when it committed on its
        // own; zero otherwise, which is what the session protocol documents.
        commit_ts: ws.commit_ts.unwrap_or(0),
        ..SessionStats::default()
    }
}

/// A cursor over a fully-materialized result, paged out by `next_batch`.
struct MaterializedCursor {
    columns: Vec<String>,
    rows: Vec<Vec<Value>>,
    pos: usize,
    stats: SessionStats,
}

impl QueryCursor for MaterializedCursor {
    fn columns(&self) -> Vec<String> {
        self.columns.clone()
    }

    fn next_batch(&mut self, max: usize) -> Result<Vec<Vec<Value>>, EngineError> {
        let end = (self.pos + max).min(self.rows.len());
        let batch = self.rows[self.pos..end].to_vec();
        self.pos = end;
        Ok(batch)
    }

    fn stats(&self) -> SessionStats {
        self.stats.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
