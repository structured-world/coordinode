//! PostgreSQL wire-protocol frontend (:7085).
//!
//! Exposes the database over the Postgres wire protocol so any Postgres client
//! (psql, JDBC, BI tools, language drivers) can run SQL against CoordiNode
//! relational tables. This is the network binding for the SQL frontend: the
//! [`pgwire`] crate handles framing, SSL negotiation, and the startup handshake;
//! a query arrives here as text, runs through [`Database::execute_sql`] (the same
//! dialect-agnostic execution path the embedded API uses), and the result set is
//! encoded back as Postgres rows.
//!
//! Scope: the Simple Query sub-protocol with trust authentication (no password).
//! The extended (parameterized / prepared-statement) protocol and authentication
//! are not wired yet; until then this binding is meant for trusted local / inter-
//! service access, gated behind an explicitly-configured listen address.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream;
use parking_lot::RwLock;
use tokio::net::TcpListener;
use tracing::{error, info};

use pgwire::api::query::SimpleQueryHandler;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo, QueryResponse, Response, Tag};
use pgwire::api::{ClientInfo, PgWireServerHandlers, Type};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::tokio::process_socket;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

/// The database-backed Simple Query handler.
///
/// Holds the shared database handle the gRPC/REST services also use, so SQL over
/// the wire sees the same state. SQL execution is synchronous and serialized
/// through the database write lock (it seeds the plan cache), which is adequate
/// for the trusted, low-concurrency access this binding currently targets.
struct PgBackend {
    database: Arc<RwLock<Database>>,
}

/// Map a CoordiNode [`Value`] to the Postgres type advertised in `RowDescription`.
///
/// Simple Query always returns values in text format, so the client reads them
/// as text regardless; the type is metadata. Scalar table columns map to their
/// natural Postgres type; everything richer is surfaced as `TEXT`.
fn pg_type(value: &Value) -> Type {
    match value {
        Value::Null | Value::String(_) => Type::TEXT,
        Value::Bool(_) => Type::BOOL,
        Value::Int(_) | Value::Timestamp(_) => Type::INT8,
        Value::Float(_) => Type::FLOAT8,
        _ => Type::TEXT,
    }
}

/// A readable text rendering for a non-scalar value (vectors, maps, blobs, …).
/// SQL table columns are scalar, so this is only reached for graph-shaped data
/// read back through a SQL query; a debug rendering is enough to be lossless to
/// the eye without inventing a wire encoding for each modality.
fn value_text(value: &Value) -> String {
    format!("{value:?}")
}

/// Encode one cell into the row encoder, matching the Rust type to the value so
/// the text encoding is correct. `Null` is encoded as a SQL NULL.
fn encode_cell(encoder: &mut DataRowEncoder, value: &Value) -> PgWireResult<()> {
    match value {
        Value::Null => encoder.encode_field(&None::<&str>),
        Value::Bool(b) => encoder.encode_field(b),
        Value::Int(i) => encoder.encode_field(i),
        Value::Timestamp(t) => encoder.encode_field(t),
        Value::Float(f) => encoder.encode_field(f),
        Value::String(s) => encoder.encode_field(s),
        other => encoder.encode_field(&value_text(other)),
    }
}

/// Does this statement return a row set (vs. an affected-row count)? Decided from
/// the leading keyword, because the execution path returns the same row vector
/// for every statement (empty for writes).
fn returns_rows(query: &str) -> bool {
    let verb = query
        .trim_start()
        .split(|c: char| c.is_whitespace() || c == '(')
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    matches!(
        verb.as_str(),
        "SELECT" | "WITH" | "VALUES" | "TABLE" | "SHOW"
    )
}

/// The Postgres command tag for a non-row statement, derived from its verb. The
/// affected-row count is not surfaced by the execution path yet, so it is
/// reported as 0.
fn command_tag(query: &str) -> Tag {
    let verb = query
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    match verb.as_str() {
        "INSERT" => Tag::new("INSERT").with_oid(0).with_rows(0),
        "UPDATE" => Tag::new("UPDATE").with_rows(0),
        "DELETE" => Tag::new("DELETE").with_rows(0),
        "CREATE" => Tag::new("CREATE TABLE"),
        "DROP" => Tag::new("DROP TABLE"),
        _ => Tag::new("OK"),
    }
}

#[async_trait]
impl SimpleQueryHandler for PgBackend {
    async fn do_query<C>(&self, _client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        let rows = self.database.write().execute_sql(query).map_err(|e| {
            PgWireError::UserError(Box::new(ErrorInfo::new(
                "ERROR".to_owned(),
                "XX000".to_owned(),
                e.to_string(),
            )))
        })?;

        if !returns_rows(query) {
            return Ok(vec![Response::Execution(command_tag(query))]);
        }

        // Columns come from the first row's keys (a row is a sorted key->value
        // map, so header order and per-row encode order both follow that key
        // order and stay consistent). A zero-row result carries no column info,
        // so it is described with an empty schema.
        let fields: Vec<FieldInfo> = rows
            .first()
            .map(|row| {
                row.iter()
                    .map(|(name, v)| {
                        FieldInfo::new(name.clone(), None, None, pg_type(v), FieldFormat::Text)
                    })
                    .collect()
            })
            .unwrap_or_default();
        let schema = Arc::new(fields);

        let mut encoded = Vec::with_capacity(rows.len());
        for row in &rows {
            let mut encoder = DataRowEncoder::new(Arc::clone(&schema));
            for value in row.values() {
                encode_cell(&mut encoder, value)?;
            }
            encoded.push(Ok(encoder.take_row()));
        }

        Ok(vec![Response::Query(QueryResponse::new(
            schema,
            stream::iter(encoded),
        ))])
    }
}

/// The handler factory `process_socket` needs. Only the Simple Query handler is
/// overridden; startup falls back to the trust (no-auth) `NoopHandler`, and the
/// extended-query / copy / cancel handlers to their no-op defaults.
struct PgHandlers {
    backend: Arc<PgBackend>,
}

impl PgWireServerHandlers for PgHandlers {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        Arc::clone(&self.backend)
    }
}

/// Bind the Postgres wire listener on `addr` and serve connections until the
/// task is dropped. Each accepted socket is handled on its own task.
///
/// # Errors
///
/// Returns an error if the listen address cannot be bound.
pub async fn serve(addr: SocketAddr, database: Arc<RwLock<Database>>) -> std::io::Result<()> {
    let handlers = Arc::new(PgHandlers {
        backend: Arc::new(PgBackend { database }),
    });
    let listener = TcpListener::bind(addr).await?;
    info!(port = addr.port(), "PostgreSQL wire server listening");
    loop {
        let (socket, peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(e) => {
                error!(error = %e, "pg: accept failed");
                continue;
            }
        };
        let handlers = Arc::clone(&handlers);
        tokio::spawn(async move {
            if let Err(e) = process_socket(socket, None, handlers).await {
                error!(%peer, error = %e, "pg: connection error");
            }
        });
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
