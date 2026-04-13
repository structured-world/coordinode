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
mod tests {
    use super::*;
    use std::panic::Location;

    fn make_config(tracking: bool) -> ClientConfig {
        ClientConfig {
            endpoint: "http://localhost:7080".into(),
            debug_source_tracking: tracking,
            app_name: "test-app".into(),
            app_version: "v1.0.0".into(),
        }
    }

    /// inject_grpc_metadata writes file and line to the metadata map.
    #[test]
    fn injects_file_and_line() {
        let loc: &'static Location<'static> = Location::caller();
        let mut meta = MetadataMap::new();
        let config = make_config(true);

        inject_grpc_metadata(&mut meta, loc, &config);

        let file = meta.get("x-source-file").unwrap().to_str().unwrap();
        let line = meta.get("x-source-line").unwrap().to_str().unwrap();
        let app = meta.get("x-source-app").unwrap().to_str().unwrap();
        let ver = meta.get("x-source-version").unwrap().to_str().unwrap();

        assert!(!file.is_empty(), "x-source-file must not be empty");
        assert!(line.parse::<u32>().is_ok(), "x-source-line must be numeric");
        assert_eq!(app, "test-app");
        assert_eq!(ver, "v1.0.0");
    }

    /// Empty app_name / app_version are not injected.
    #[test]
    fn skips_empty_app_fields() {
        let loc: &'static Location<'static> = Location::caller();
        let mut meta = MetadataMap::new();
        let config = ClientConfig {
            endpoint: "http://localhost:7080".into(),
            debug_source_tracking: true,
            app_name: String::new(),
            app_version: String::new(),
        };

        inject_grpc_metadata(&mut meta, loc, &config);

        assert!(meta.get("x-source-app").is_none());
        assert!(meta.get("x-source-version").is_none());
        // file and line still present
        assert!(meta.get("x-source-file").is_some());
        assert!(meta.get("x-source-line").is_some());
    }

    /// Function name (x-source-function) is intentionally absent.
    #[test]
    fn no_function_key() {
        let loc: &'static Location<'static> = Location::caller();
        let mut meta = MetadataMap::new();
        inject_grpc_metadata(&mut meta, loc, &make_config(true));
        assert!(meta.get("x-source-function").is_none());
    }
}
