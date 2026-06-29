use super::*;

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
    let rows = intercept("SELECT version()").expect("version is a catalog probe");
    assert!(rows[0].contains_key("version"));
    assert!(scalar(&rows).starts_with("PostgreSQL 15.0"));
}

#[test]
fn current_schema_and_database_and_user() {
    assert_eq!(
        scalar(&intercept("SELECT current_schema()").unwrap()),
        "public"
    );
    assert_eq!(
        scalar(&intercept("select current_database()").unwrap()),
        "coordinode"
    );
    assert_eq!(
        scalar(&intercept("SELECT current_user").unwrap()),
        "postgres"
    );
}

#[test]
fn show_returns_param_named_column() {
    let rows = intercept("SHOW transaction_isolation").expect("SHOW is intercepted");
    assert!(rows[0].contains_key("transaction_isolation"));
    assert_eq!(scalar(&rows), "read committed");

    let enc = intercept("SHOW server_encoding").unwrap();
    assert_eq!(scalar(&enc), "UTF8");

    // An unknown parameter is reported as empty, not an error.
    let unknown = intercept("SHOW some_unknown_gucparam").unwrap();
    assert_eq!(scalar(&unknown), "");
}

#[test]
fn trailing_semicolon_and_case_are_tolerated() {
    assert!(intercept("SELECT VERSION();").is_some());
    assert!(intercept("show  Server_Version ;").is_some());
}

#[test]
fn ordinary_sql_is_not_intercepted() {
    assert!(intercept("SELECT id, name FROM Account WHERE id = 1").is_none());
    assert!(intercept("INSERT INTO Account (id) VALUES (1)").is_none());
    assert!(intercept("CREATE TABLE T (id BIGINT PRIMARY KEY)").is_none());
}
