//! A store written by a released build (v0.6.0) predates apply coverage.
//! This release refuses to open it, leaving it byte for byte as it was.
//!
//! `tests/fixtures/v0.6.0/` holds what v0.6.0 itself wrote: `embedded/` (a
//! journalled store: Alice-[:KNOWS]->Bob and Carol) and `raft/` (a
//! single-node Raft store with three applied proposals).
#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use coordinode_embed::Database;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v0.6.0")
        .join(name)
}

/// Copy a fixture store into a scratch directory, so a test that opens it
/// can never alter the committed bytes.
fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), &target).unwrap();
        }
    }
}

/// Every file under `dir`, relative path and bytes, sorted.
fn snapshot_tree(dir: &Path) -> Vec<(PathBuf, Vec<u8>)> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, Vec<u8>)>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &entry.path(), out);
            } else {
                let rel = entry.path().strip_prefix(root).unwrap().to_path_buf();
                out.push((rel, std::fs::read(entry.path()).unwrap()));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

#[test]
fn a_released_embedded_store_is_refused_untouched() {
    let scratch = tempfile::tempdir().unwrap();
    let store = scratch.path().join("embedded");
    copy_tree(&fixture("embedded"), &store);
    let before = snapshot_tree(&store);

    let err = Database::open(&store)
        .err()
        .expect("a store without a coverage record must be refused");
    assert!(
        err.to_string().contains("no apply-coverage record"),
        "the refusal names the reason, got: {err}"
    );
    assert_eq!(
        snapshot_tree(&store),
        before,
        "the refused store is left exactly as the released build wrote it"
    );
}

#[test]
fn a_released_raft_store_is_refused_untouched() {
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };
    use coordinode_storage::engine::core::StorageEngine;

    let scratch = tempfile::tempdir().unwrap();
    let store = scratch.path().join("raft");
    copy_tree(&fixture("raft"), &store);
    let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        &store,
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);

    let err = {
        let engine = Arc::new(StorageEngine::open(&config).expect("the trees themselves open"));
        coordinode_raft::storage::CoordinodeStateMachine::new(engine)
            .err()
            .expect("a Raft store without a coverage record must be refused")
    };
    assert!(
        err.to_string().contains("without an apply-coverage record"),
        "the refusal names the reason, got: {err}"
    );
}
