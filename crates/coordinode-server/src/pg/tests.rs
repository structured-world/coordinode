//! End-to-end tests for the Postgres wire frontend: a real `tokio-postgres`
//! client connects over TCP, runs SQL through the Simple Query protocol, and
//! verifies the rows the server encodes back.

use std::sync::Arc;

use parking_lot::RwLock;
use tempfile::TempDir;
use tokio_postgres::{NoTls, SimpleQueryMessage};

use coordinode_embed::Database;

/// Open a database, bind the pg listener on an ephemeral port, and return a
/// connected client plus the temp dir (kept alive for the test's duration).
async fn connect() -> (tokio_postgres::Client, TempDir) {
    let dir = TempDir::new().expect("temp dir");
    let db = Database::open(dir.path()).expect("open db");
    let database = Arc::new(RwLock::new(db));

    // Bind to port 0 so the OS picks a free port, then read it back.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let handlers = Arc::new(super::PgHandlers {
        backend: Arc::new(super::PgBackend { database }),
    });
    tokio::spawn(async move {
        loop {
            let (socket, _) = listener.accept().await.expect("accept");
            let handlers = Arc::clone(&handlers);
            tokio::spawn(async move {
                let _ = pgwire::tokio::process_socket(socket, None, handlers).await;
            });
        }
    });

    let conn_str = format!("host=127.0.0.1 port={} user=postgres", addr.port());
    let (client, connection) = tokio_postgres::connect(&conn_str, NoTls)
        .await
        .expect("connect");
    tokio::spawn(async move {
        let _ = connection.await;
    });
    (client, dir)
}

/// Collect the data rows of a simple query into a vector of column-keyed maps.
fn data_rows(messages: &[SimpleQueryMessage]) -> Vec<std::collections::BTreeMap<String, String>> {
    let mut out = Vec::new();
    for msg in messages {
        if let SimpleQueryMessage::Row(row) = msg {
            let mut map = std::collections::BTreeMap::new();
            for i in 0..row.len() {
                let name = row.columns()[i].name().to_string();
                let val = row.get(i).unwrap_or("").to_string();
                map.insert(name, val);
            }
            out.push(map);
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn sql_crud_over_the_wire() {
    let (client, _dir) = connect().await;

    // DDL + INSERT through the wire.
    client
        .simple_query("CREATE TABLE Account (id BIGINT PRIMARY KEY, name STRING)")
        .await
        .expect("create table");
    client
        .simple_query("INSERT INTO Account (id, name) VALUES (1, 'Alice')")
        .await
        .expect("insert");

    // SELECT round-trips the row.
    let rows = data_rows(
        &client
            .simple_query("SELECT id, name FROM Account WHERE id = 1")
            .await
            .expect("select"),
    );
    assert_eq!(rows.len(), 1, "expected one row");
    assert_eq!(rows[0].get("name").map(String::as_str), Some("Alice"));
    assert_eq!(rows[0].get("id").map(String::as_str), Some("1"));

    // UPDATE then read back the new value.
    client
        .simple_query("UPDATE Account SET name = 'Alicia' WHERE id = 1")
        .await
        .expect("update");
    let rows = data_rows(
        &client
            .simple_query("SELECT name FROM Account WHERE id = 1")
            .await
            .expect("select after update"),
    );
    assert_eq!(rows[0].get("name").map(String::as_str), Some("Alicia"));

    // DELETE removes the row.
    client
        .simple_query("DELETE FROM Account WHERE id = 1")
        .await
        .expect("delete");
    let rows = data_rows(
        &client
            .simple_query("SELECT name FROM Account WHERE id = 1")
            .await
            .expect("select after delete"),
    );
    assert_eq!(rows.len(), 0, "deleted row must be gone");
}

#[tokio::test(flavor = "multi_thread")]
async fn invalid_sql_returns_error_not_disconnect() {
    let (client, _dir) = connect().await;
    // A parse error must come back as a Postgres ErrorResponse, and the
    // connection must stay usable for the next query.
    let err = client.simple_query("SELECT FROM WHERE bogus").await;
    assert!(err.is_err(), "invalid SQL must surface as an error");
    // Connection still alive: a valid statement succeeds afterwards.
    client
        .simple_query("CREATE TABLE T (id BIGINT PRIMARY KEY)")
        .await
        .expect("connection still usable after error");
}
