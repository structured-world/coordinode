//! Encoder lockdown — regression gate for the `coordinode-embed` integration
//! suite.
//!
//! Fixtures seed and assert graph state through the Layer-4 stores
//! (`LocalNodeStore` / `LocalEdgeStore` / `LocalIndexStore`), never by
//! hand-building the data-plane partition keys. This test fails if any
//! integration test file grows a raw data-plane key-encoder call.
//!
//! Scope note: only the data-plane encoders (node / edge-property / adjacency /
//! secondary-index) are forbidden. Schema and trigger DDL encoders are not in
//! this list — a white-box test legitimately verifies the on-disk
//! current-revision pointer scheme by reading those raw keys directly, which
//! the store API hides by design.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs;
use std::path::Path;

/// Raw data-plane key builders forbidden anywhere under `tests/`.
const DATA_PLANE_ENCODER_NEEDLES: &[&str] = &[
    "encode_node_key(",
    "encode_temporal_node_key(",
    "encode_edgeprop_key(",
    "encode_temporal_edgeprop_key(",
    "encode_adj_key_forward(",
    "encode_adj_key_reverse(",
    "write_adj_key_forward(",
    "write_adj_key_reverse(",
    "encode_index_key(",
    "encode_compound_index_key(",
];

#[test]
fn tests_dir_has_no_raw_data_plane_encoders() {
    let tests_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut bad: Vec<String> = Vec::new();

    fn walk(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    let mut files: Vec<std::path::PathBuf> = Vec::new();
    walk(&tests_root, &mut files);

    for path in &files {
        // Skip this file — it holds the needle strings as search literals.
        if path.file_name().is_some_and(|n| n == "encoder_lockdown.rs") {
            continue;
        }
        let rel = path
            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let content = fs::read_to_string(path).unwrap_or_default();
        for needle in DATA_PLANE_ENCODER_NEEDLES {
            if content.contains(needle) {
                bad.push(format!(
                    "{rel} uses `{needle}` — seed fixtures and assert graph state \
                     through the Layer-4 stores (LocalNodeStore / LocalEdgeStore / \
                     LocalIndexStore), not raw data-plane key encoders.",
                ));
            }
        }
    }

    assert!(
        bad.is_empty(),
        "tests/ raw data-plane key-encoder regressions:\n{}",
        bad.join("\n"),
    );
}
