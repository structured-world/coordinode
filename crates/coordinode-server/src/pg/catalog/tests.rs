use super::*;

/// The namespace the probing connection named as its database.
const DATABASE: &str = "tenant";

/// The single string value of a one-row, one-column synthesized result.
fn scalar(rows: &[Row]) -> String {
    assert_eq!(rows.len(), 1, "expected exactly one row");
    let value = rows[0].values().next().expect("one column");
    match value {
        Value::String(s) => s.clone(),
        other => panic!("expected a string value, got {other:?}"),
    }
}

#[test]
fn version_is_intercepted() {
    let rows = intercept("SELECT version()", DATABASE).expect("version is a catalog probe");
    assert!(rows[0].contains_key("version"));
    assert!(scalar(&rows).starts_with("PostgreSQL 15.0"));
}

/// `current_database()` is the namespace the connection named: a namespace
/// is what a database is inside a PostgreSQL server.
#[test]
fn current_schema_and_database_and_user() {
    assert_eq!(
        scalar(&intercept("SELECT current_schema()", DATABASE).unwrap()),
        "public"
    );
    assert_eq!(
        scalar(&intercept("select current_database()", DATABASE).unwrap()),
        DATABASE
    );
    assert_eq!(
        scalar(&intercept("SELECT current_user", DATABASE).unwrap()),
        "postgres"
    );
}

#[test]
fn show_returns_param_named_column() {
    let rows = intercept("SHOW transaction_isolation", DATABASE).expect("SHOW is intercepted");
    assert!(rows[0].contains_key("transaction_isolation"));
    assert_eq!(scalar(&rows), "read committed");

    let enc = intercept("SHOW server_encoding", DATABASE).unwrap();
    assert_eq!(scalar(&enc), "UTF8");

    // An unknown parameter is reported as empty, not an error.
    let unknown = intercept("SHOW some_unknown_gucparam", DATABASE).unwrap();
    assert_eq!(scalar(&unknown), "");
}

#[test]
fn trailing_semicolon_and_case_are_tolerated() {
    assert!(intercept("SELECT VERSION();", DATABASE).is_some());
    assert!(intercept("show  Server_Version ;", DATABASE).is_some());
}

/// Statements, `SHOW NAMESPACES` among them, run on the real path.
#[test]
fn ordinary_sql_is_not_intercepted() {
    for sql in [
        "SELECT id, name FROM Account WHERE id = 1",
        "INSERT INTO Account (id) VALUES (1)",
        "CREATE TABLE T (id BIGINT PRIMARY KEY)",
        "SHOW NAMESPACES",
        "show namespaces;",
    ] {
        assert!(intercept(sql, DATABASE).is_none(), "{sql}");
    }
}
