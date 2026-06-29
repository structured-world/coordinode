//! Minimal Postgres catalog / introspection shim.
//!
//! Postgres clients and drivers send a handful of probe queries right after
//! connecting (`SELECT version()`, `current_schema()`, `SHOW
//! transaction_isolation`, …) before they run any user SQL. Those reference
//! server built-ins that the SQL frontend does not implement, so without an
//! answer a driver's connection handshake fails. This module recognizes those
//! probes and synthesizes a correct single-column reply, leaving every other
//! query to the real execution path.
//!
//! Scope: scalar session functions and `SHOW`. Tabular catalog introspection
//! (`information_schema.*`, `pg_catalog.pg_class`) needs real virtual relations
//! that honour the client's exact projection — a fixed column set would break
//! index-based driver reads — and is a separate feature, not faked here.

use std::collections::BTreeMap;

use coordinode_core::graph::types::Value;

/// A synthesized catalog row: column name -> value, matching the [`Row`] shape
/// the execution path produces so the wire encoder treats both identically.
type Row = BTreeMap<String, Value>;

/// The `server_version` CoordiNode reports. The major version gates feature
/// detection in many drivers, so it must look like a real Postgres release.
const SERVER_VERSION: &str = "15.0";

/// Full `version()` banner.
fn version_banner() -> String {
    format!(
        "PostgreSQL {SERVER_VERSION} (CoordiNode {}) on {}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::ARCH,
    )
}

/// One single-column row: `column -> string value`.
fn one(column: &str, value: &str) -> Vec<Row> {
    let mut row = Row::new();
    row.insert(column.to_owned(), Value::String(value.to_owned()));
    vec![row]
}

/// If `query` is a recognized catalog / introspection probe, return its
/// synthesized result; otherwise `None` so the caller runs the real SQL path.
pub fn intercept(query: &str) -> Option<Vec<Row>> {
    let normalized = query.trim().trim_end_matches(';').to_ascii_lowercase();
    let collapsed: String = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

    // `SHOW <param>` -> one row, one column named after the parameter.
    if let Some(rest) = collapsed.strip_prefix("show ") {
        let param = rest.trim();
        return Some(one(param, show_value(param)));
    }

    // Scalar session functions. Matched on substring so the surrounding
    // `SELECT ... ` wrapper (with or without parens / aliases) still resolves.
    if collapsed.contains("version()") {
        return Some(one("version", &version_banner()));
    }
    if collapsed.contains("current_schema") {
        return Some(one("current_schema", "public"));
    }
    if collapsed.contains("current_database") || collapsed.contains("current_catalog") {
        return Some(one("current_database", "coordinode"));
    }
    if collapsed.contains("session_user") {
        return Some(one("session_user", "postgres"));
    }
    if collapsed.contains("current_user") {
        return Some(one("current_user", "postgres"));
    }

    None
}

/// The value reported for a `SHOW <param>`. Unknown parameters report an empty
/// string (the parameter exists but is unset) rather than erroring.
fn show_value(param: &str) -> &'static str {
    match param {
        "server_version" => SERVER_VERSION,
        "server_encoding" | "client_encoding" => "UTF8",
        "transaction_isolation" | "default_transaction_isolation" => "read committed",
        "standard_conforming_strings" | "is_superuser" => "on",
        "datestyle" => "ISO, MDY",
        "timezone" => "UTC",
        "application_name" => "",
        _ => "",
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests;
