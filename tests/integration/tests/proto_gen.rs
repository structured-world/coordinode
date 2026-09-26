//! The committed `proto_gen/` fallback must match what the proto submodule
//! generates, so a build without the submodule compiles the same API.
//!
//! Refresh it with:
//!
//! ```bash
//! COORDINODE_UPDATE_PROTO_GEN=1 cargo build -p coordinode-integration
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::path::Path;

fn rust_files(dir: &Path) -> BTreeMap<String, String> {
    std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|e| e == "rs"))
        .map(|path| {
            let name = path
                .file_name()
                .expect("file name")
                .to_string_lossy()
                .into_owned();
            let body = std::fs::read_to_string(&path).expect("read generated file");
            (name, body)
        })
        .collect()
}

/// Every generated binding is committed, with the same content: a missing or
/// stale file breaks every build that falls back to `proto_gen/`.
#[test]
fn proto_gen_matches_the_proto_submodule() {
    let generated = rust_files(Path::new(env!("OUT_DIR")));
    let committed = rust_files(&Path::new(env!("CARGO_MANIFEST_DIR")).join("proto_gen"));

    let generated_names: Vec<&String> = generated.keys().collect();
    let committed_names: Vec<&String> = committed.keys().collect();
    assert_eq!(
        committed_names, generated_names,
        "proto_gen/ holds a different file set than the build generates; \
         refresh it with COORDINODE_UPDATE_PROTO_GEN=1"
    );
    let stale: Vec<&String> = generated
        .iter()
        .filter(|(name, body)| committed.get(*name) != Some(*body))
        .map(|(name, _)| name)
        .collect();
    assert!(
        stale.is_empty(),
        "proto_gen/ is stale for {stale:?}; refresh it with COORDINODE_UPDATE_PROTO_GEN=1"
    );
}
