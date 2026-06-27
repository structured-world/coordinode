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

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::WriteStats;
use coordinode_session::{CursorEngine, EngineError, QueryCursor, SessionStats};
use parking_lot::RwLock;

/// Storage keys scanned per keyset page. Independent of the client's batch size:
/// a page may yield fewer output rows than this when a `Filter` rejects some, so
/// the cursor refills across pages until the client's batch is full or the scan
/// is exhausted.
const KEYSET_PAGE: usize = 1024;

/// A [`CursorEngine`] that runs statements through the embedded [`Database`].
pub struct DatabaseCursorEngine {
    database: Arc<RwLock<Database>>,
}

impl DatabaseCursorEngine {
    pub fn new(database: Arc<RwLock<Database>>) -> Self {
        Self { database }
    }
}

impl CursorEngine for DatabaseCursorEngine {
    fn open_cursor(
        &self,
        query: &str,
        params: HashMap<String, Value>,
        txid: u64,
    ) -> Result<Box<dyn QueryCursor>, EngineError> {
        let params = if params.is_empty() {
            None
        } else {
            Some(params)
        };
        // Keyset path: auto-commit read whose plan pages by a single NodeScan.
        // An interactive statement (txid != 0) reuses the parked transaction
        // and cannot re-pin a snapshot per page, so it always materializes.
        if txid == 0 && self.database.read().keyset_pageable(query) {
            let cursor = KeysetCursor::open(Arc::clone(&self.database), query.to_string(), params)?;
            return Ok(Box::new(cursor));
        }

        let db = self.database.read();
        let (rows, stats) = if txid == 0 {
            let result = db
                .execute_cypher_shared(query, params, None, None, None)
                .map_err(|e| EngineError(e.to_string()))?;
            (
                rows_to_values(&result.rows),
                write_stats(&result.write_stats),
            )
        } else {
            let rows = db
                .execute_in_transaction(txid, query, params)
                .map_err(|e| EngineError(e.to_string()))?;
            (rows_to_values(&rows), SessionStats::default())
        };
        Ok(Box::new(MaterializedCursor {
            columns: rows.0,
            rows: rows.1,
            pos: 0,
            stats,
        }))
    }

    fn begin_transaction(&self) -> Result<u64, EngineError> {
        Ok(self.database.read().begin_transaction())
    }

    fn commit_transaction(&self, txid: u64) -> Result<u64, EngineError> {
        self.database
            .read()
            .commit_transaction(txid)
            .map_err(|e| EngineError(e.to_string()))
    }

    fn rollback_transaction(&self, txid: u64) -> Result<(), EngineError> {
        self.database
            .read()
            .rollback_transaction(txid)
            .map_err(|e| EngineError(e.to_string()))
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
}

impl KeysetCursor {
    /// Open the cursor, prefetching pages until the first row is found (so the
    /// column header is known) or the scan is exhausted.
    fn open(
        database: Arc<RwLock<Database>>,
        query: String,
        params: Option<HashMap<String, Value>>,
    ) -> Result<Self, EngineError> {
        let mut cursor = Self {
            database,
            query,
            params,
            columns: Vec::new(),
            read_ts: None,
            resume: None,
            exhausted: false,
            pending: VecDeque::new(),
            stats: SessionStats::default(),
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
            .map_err(|e| EngineError(e.to_string()))?;
        self.read_ts = Some(page.read_ts);
        self.resume = page.last_key;
        self.exhausted = page.exhausted;
        self.stats = write_stats(&page.write_stats);
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
