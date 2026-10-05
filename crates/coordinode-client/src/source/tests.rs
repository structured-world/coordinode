use super::*;
use std::panic::Location;

fn make_config(tracking: bool) -> ClientConfig {
    ClientConfig {
        endpoint: "http://localhost:7080".into(),
        debug_source_tracking: tracking,
        app_name: "test-app".into(),
        app_version: "v1.0.0".into(),
        transport: crate::Transport::default(),
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
        transport: crate::Transport::default(),
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

/// A session stream names only the application; its statements carry their
/// own location, so the stream's metadata holds no file or line.
#[test]
fn app_metadata_names_the_application_only() {
    let mut meta = MetadataMap::new();
    inject_app_metadata(&mut meta, &make_config(true));
    assert_eq!(
        meta.get("x-source-app").unwrap().to_str().unwrap(),
        "test-app"
    );
    assert_eq!(
        meta.get("x-source-version").unwrap().to_str().unwrap(),
        "v1.0.0"
    );
    assert!(meta.get("x-source-file").is_none());
    assert!(meta.get("x-source-line").is_none());
}

/// A statement's source is the caller's file and line.
#[test]
fn a_statement_source_is_the_call_site() {
    let loc: &'static Location<'static> = Location::caller();
    let source = statement_source(loc);
    assert_eq!(source.file, loc.file());
    assert_eq!(source.line, loc.line());
    assert!(source.function.is_empty());
}
