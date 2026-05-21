//! R165 encoder lockdown — regression gate.
//!
//! Asserts that production source files in `coordinode-query` do not
//! grow new raw `encode_node_key` / `encode_temporal_node_key` /
//! `encode_edgeprop_key` / `encode_temporal_edgeprop_key` usages
//! outside the sanctioned Layer-5 helper module. New executor code
//! must go through the typed helpers on `ExecutionContext`
//! (`mvcc_get_node`, `mvcc_put_edge_props_either`, etc.) instead of
//! the raw encoders.
//!
//! The lockdown is a structural test, not a clippy lint, because
//! clippy's `disallowed-methods` cannot allow specific call sites
//! while forbidding others in the same crate.
//!
//! Allowed-list rationale per site:
//! - `executor/runner.rs` helper-module region — the typed helpers
//!   themselves call the encoders. Bounded by a sentinel-comment
//!   range so the gate stays mechanical.
//! - `#[cfg(test)]` modules — test fixtures use raw encoders to
//!   construct probe keys. Test-only code migrates in R166.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::fs;
use std::path::Path;

const RAW_ENCODER_NEEDLES: &[&str] = &[
    "encode_node_key(",
    "encode_temporal_node_key(",
    "encode_edgeprop_key(",
    "encode_temporal_edgeprop_key(",
];

/// Files where the test runs. The scan is whitelist-based so the
/// gate cannot accidentally start covering crates outside our scope.
const SCAN_FILES: &[&str] = &[
    "src/executor/runner.rs",
    "src/index/ops.rs",
    "src/index/build.rs",
    "src/index/ttl.rs",
    "src/index/ttl_reaper.rs",
    "src/index/registry.rs",
];

/// Per-file allowance: `(file, allowed_max_hits)`. The number is the
/// CURRENT post-migration count; the gate fails if a future PR
/// raises it. Update with the migration audit comment when lowering.
const ALLOWED: &[(&str, usize)] = &[
    // runner.rs baseline (R166 audit-suite slice): the OCC-scope
    // audit suite now uses typed `OccScope::contains_node[_temporal]`
    // / `contains_edge_props[_temporal]` overloads (added to
    // coordinode-storage in the same slice) so the assertions are
    // raw-encoder-free. Remaining ~26 in cfg(test) seed-fixture
    // setups (need raw keys to construct probe payloads with the
    // EdgeProp `Vec<(field_id, Value)>` shape that LocalEdgeStore
    // doesn't accept yet) + ~15 typed-helper internals. Total = 41.
    ("src/executor/runner.rs", 41),
    // ops.rs — fully routed through LocalIndexStore after slice 12.
    ("src/index/ops.rs", 0),
    // build.rs (R166): cfg(test) `insert_node` helper now routes
    // through LocalNodeStore. 0 raw encoder usages.
    ("src/index/build.rs", 0),
    // ttl.rs (R166): cfg(test) `insert_node_with_timestamp` helper
    // + verify-deleted assertions now route through LocalNodeStore.
    // 0 raw encoder usages.
    ("src/index/ttl.rs", 0),
    // ttl_reaper.rs: fully migrated. cfg(test) fixtures route through
    // LocalNodeStore; the production `prepare_subtree_mutations` builds
    // its EdgeProp delete via the typed `Mutation::delete_edge_props`
    // constructor in `coordinode-core` (which holds the encoder call
    // at the right layer, co-located with the encoder it wraps).
    // 0 raw encoder usages.
    ("src/index/ttl_reaper.rs", 0),
    ("src/index/registry.rs", 0),
];

#[test]
fn encoder_lockdown_no_raw_encoder_growth() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations: Vec<String> = Vec::new();

    for file_rel in SCAN_FILES {
        let path = crate_root.join(file_rel);
        let content = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {file_rel}: {e}"));

        let mut total = 0_usize;
        for needle in RAW_ENCODER_NEEDLES {
            total += content.matches(needle).count();
        }

        let allowed = ALLOWED
            .iter()
            .find_map(|(p, n)| if p == file_rel { Some(*n) } else { None })
            .unwrap_or(0);

        if total > allowed {
            violations.push(format!(
                "{file_rel}: {total} raw encoder usages (allowed: {allowed}). \
                 New raw `encode_node_key` / `encode_temporal_*_key` / \
                 `encode_edgeprop_key` calls must route through the \
                 ExecutionContext typed helpers (mvcc_*_node / \
                 mvcc_*_edge_props) instead.",
            ));
        }
    }

    assert!(
        violations.is_empty(),
        "Encoder lockdown violations:\n{}",
        violations.join("\n"),
    );
}

/// Catches the second failure mode: a new `coordinode-query/src/*.rs`
/// file ships with raw encoder calls but isn't listed in `SCAN_FILES`
/// — the gate would silently pass. This test walks every `.rs` under
/// `src/` and asserts that any file using a raw encoder appears in
/// the whitelist. New files without raw encoders are fine.
#[test]
fn encoder_lockdown_no_new_files_with_raw_encoders() {
    let src_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
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

    let mut all_src_files: Vec<std::path::PathBuf> = Vec::new();
    walk(&src_root, &mut all_src_files);

    for path in &all_src_files {
        let rel = path
            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let content = fs::read_to_string(path).unwrap_or_default();
        let uses_raw = RAW_ENCODER_NEEDLES.iter().any(|n| content.contains(n));
        if uses_raw && !SCAN_FILES.iter().any(|s| rel.ends_with(s)) {
            bad.push(format!(
                "{rel} uses raw encoders but is not in SCAN_FILES — add it (with a baseline allowance) to the lockdown table.",
            ));
        }
    }

    assert!(
        bad.is_empty(),
        "Encoder lockdown coverage gaps:\n{}",
        bad.join("\n"),
    );
}
