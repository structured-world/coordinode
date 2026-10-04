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
//! A small [`catalog`] shim answers the introspection probes drivers send on
//! connect (`version()`, `SHOW ...`). The extended (parameterized /
//! prepared-statement) protocol, authentication, and tabular catalog
//! introspection (`information_schema` / `pg_catalog` relations) are not wired
//! yet; until then this binding is meant for trusted local / inter-service
//! access, gated behind an explicitly-configured listen address.

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

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;
use coordinode_embed::Database;

mod catalog;

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

/// Build a `Query` response from a result set of column-keyed rows.
///
/// Columns come from the first row's keys (a row is a sorted key->value map, so
/// header order and per-row encode order both follow that key order and stay
/// consistent). A zero-row result carries no column info, so it is described
/// with an empty schema.
fn query_response(rows: &[BTreeMap<String, Value>]) -> PgWireResult<Response> {
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
    for row in rows {
        let mut encoder = DataRowEncoder::new(Arc::clone(&schema));
        for value in row.values() {
            encode_cell(&mut encoder, value)?;
        }
        encoded.push(Ok(encoder.take_row()));
    }
    Ok(Response::Query(QueryResponse::new(
        schema,
        stream::iter(encoded),
    )))
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

/// The SQLSTATE a PostgreSQL driver branches on for `error` (PostgreSQL
/// documentation, Appendix A "PostgreSQL Error Codes"). A duplicate key is
/// `23505 unique_violation`, a missing required value `23502
/// not_null_violation`, a value of another type than a constraint requires
/// `23514 check_violation`; changing a key column, which PostgreSQL permits
/// and this server does not, is `42P10 invalid_column_reference`. A write
/// conflict is `40001 serialization_failure`, the class drivers retry the
/// whole transaction on; write pressure is `53000 insufficient_resources`;
/// a write refused because the disk is below its reserve is `53100
/// disk_full`;
/// a write that reached a follower is `25006 read_only_sql_transaction`, what
/// a PostgreSQL standby answers. A name a table already holds is `42P07
/// duplicate_table`, one another catalog object holds `42710
/// duplicate_object`; a missing table is `42P01 undefined_table`, another
/// missing object `42704 undefined_object`; a catalog change the catalog's
/// state refuses is `55000 object_not_in_prerequisite_state`. Everything else
/// stays `XX000 internal_error`.
fn sqlstate(error: &coordinode_embed::db::DatabaseError) -> &'static str {
    use coordinode_core::schema::definition::ConstraintKind;
    use coordinode_embed::db::DatabaseError;
    use coordinode_query::executor::runner::{CatalogObject, ExecutionError};
    match error {
        DatabaseError::Execution(ExecutionError::CatalogObjectExists {
            object: CatalogObject::Label,
            ..
        }) => "42P07",
        DatabaseError::Execution(ExecutionError::CatalogObjectExists { .. }) => "42710",
        DatabaseError::Execution(ExecutionError::CatalogObjectMissing {
            object: CatalogObject::Label,
            ..
        }) => "42P01",
        DatabaseError::Execution(ExecutionError::CatalogObjectMissing { .. }) => "42704",
        DatabaseError::Execution(ExecutionError::CatalogRefused(_)) => "55000",
        DatabaseError::Execution(
            ExecutionError::DuplicateKey { .. } | ExecutionError::UniqueViolation { .. },
        ) => "23505",
        DatabaseError::Execution(ExecutionError::ConstraintViolation { kind, .. })
            if matches!(**kind, ConstraintKind::Type(_)) =>
        {
            "23514"
        }
        DatabaseError::Execution(ExecutionError::ConstraintViolation { .. }) => "23502",
        DatabaseError::Execution(ExecutionError::KeyImmutable { .. }) => "42P10",
        DatabaseError::TransactionConflict { .. }
        | DatabaseError::Execution(ExecutionError::Conflict(_)) => "40001",
        // insufficient_resources: both are the server catching up, and the
        // same statement succeeds once it has.
        DatabaseError::WriteBackpressure
        | DatabaseError::Execution(ExecutionError::Backpressure)
        | DatabaseError::Execution(ExecutionError::IndexBehind(_)) => "53000",
        // disk_full: the disk is below its reserve, writes wait for space.
        DatabaseError::Storage(coordinode_storage::error::StorageError::OutOfSpace { .. })
        | DatabaseError::Execution(ExecutionError::Storage(
            coordinode_storage::error::StorageError::OutOfSpace { .. },
        )) => "53100",
        // read_only_sql_transaction: this member takes no writes, the leader
        // does, and a read-only member is read-only for the same reason.
        DatabaseError::NotLeader { .. }
        | DatabaseError::Execution(ExecutionError::NotLeader { .. })
        | DatabaseError::Mismatched(_)
        | DatabaseError::Execution(ExecutionError::Mismatched(_)) => "25006",
        _ => "XX000",
    }
}

#[async_trait]
impl SimpleQueryHandler for PgBackend {
    async fn do_query<C>(&self, _client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + Unpin + Send + Sync,
    {
        // Answer driver / client introspection probes (version(), SHOW, ...)
        // without touching the execution path — they reference server built-ins
        // the SQL frontend does not implement.
        if let Some(rows) = catalog::intercept(query) {
            return Ok(vec![query_response(&rows)?]);
        }

        let rows = crate::services::blocking(|| self.database.write().execute_sql(query)).map_err(
            |e| {
                PgWireError::UserError(Box::new(ErrorInfo::new(
                    "ERROR".to_owned(),
                    sqlstate(&e).to_owned(),
                    e.to_string(),
                )))
            },
        )?;

        if !returns_rows(query) {
            return Ok(vec![Response::Execution(command_tag(query))]);
        }
        Ok(vec![query_response(&rows)?])
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
