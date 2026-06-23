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
