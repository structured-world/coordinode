//! Client-side source location tracking.
//!
//! Injects `x-source-file`, `x-source-line`, `x-source-app`, and
//! `x-source-version` gRPC metadata keys into outgoing requests when the
//! client is configured with `debug_source_tracking(true)`.
//!
//! The server reads these keys via `coordinode_query::advisor::source`
//! and attributes query fingerprint statistics to the originating call site.
//!
//! # Protocol keys
//!
//! | gRPC metadata key     | Source                                |
//! |-----------------------|---------------------------------------|
//! | `x-source-file`       | `std::panic::Location::file()`        |
//! | `x-source-line`       | `std::panic::Location::line()` (str)  |
//! | `x-source-app`        | `ClientConfig::app_name`              |
//! | `x-source-version`    | `ClientConfig::app_version`           |
//!
//! `x-source-function` is intentionally omitted: Rust's `Location` does not
//! expose the enclosing function name. TypeScript and Python drivers can send
//! it via `Error.captureStackTrace` / `inspect.stack()` respectively.

use std::panic::Location;

use tonic::metadata::MetadataMap;

use crate::config::ClientConfig;

/// Inject caller source location into gRPC request metadata.
///
/// Silently skips any metadata key whose value fails to parse (e.g. exotic
/// characters in a path). In practice file paths and integers are always
/// valid ASCII.
pub(crate) fn inject_grpc_metadata(
    meta: &mut MetadataMap,
    location: &'static Location<'static>,
    config: &ClientConfig,
) {
    if let Ok(v) = location.file().parse() {
        meta.insert("x-source-file", v);
    }
    if let Ok(v) = location.line().to_string().parse() {
        meta.insert("x-source-line", v);
    }
    if !config.app_name.is_empty() {
        if let Ok(v) = config.app_name.parse() {
            meta.insert("x-source-app", v);
        }
    }
    if !config.app_version.is_empty() {
        if let Ok(v) = config.app_version.parse() {
            meta.insert("x-source-version", v);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
