//! Database-backed [`CursorEngine`] for the gRPC session binding.
//!
//! Opens a server-side cursor by running the statement through the embedded
//! [`Database`] and paging the result. This is the **materialize-once** path: it
//! runs the plan to completion and pages the rows out. A non-blocking plan
//! should later resume by keyset (`seek_to`) so the cursor stays O(batch); that
//! is a separate increment. Blocking operators (sort, aggregation) materialise
//! by construction regardless.

use std::collections::HashMap;
use std::sync::Arc;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;
use coordinode_query::executor::row::Row;
use coordinode_query::executor::runner::WriteStats;
use coordinode_session::{CursorEngine, EngineError, QueryCursor, SessionStats};
use parking_lot::RwLock;

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
    let values: Vec<Vec<Value>> = rows
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
