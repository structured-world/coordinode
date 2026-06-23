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

/// Full raw key-encoder surface forbidden in the integration test
/// suites (`tests/*.rs`). Broader than `RAW_ENCODER_NEEDLES` (which gates
/// the `src/` baseline for node / edgeprop only): test fixtures must seed
/// and assert through the Layer-4 stores, so NO raw key builder of any
/// partition may appear in `tests/`.
const TEST_ENCODER_NEEDLES: &[&str] = &[
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

/// Files where the test runs. The scan is whitelist-based so the
/// gate cannot accidentally start covering crates outside our scope.
const SCAN_FILES: &[&str] = &[
    "src/executor/runner.rs",
    "src/executor/vector_predicate.rs",
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
    // runner.rs baseline. After R166 the cfg(test) seed fixtures route
    // through the real Layer-4 stores (`LocalNodeStore::put[_temporal]`,
    // `LocalEdgeStore::put_edge[_temporal]`) — unblocked by ADR-040, which
    // made `EdgeProperties` and the executor's `Vec<(field_id, Value)>`
    // shape wire-identical — and the OCC-scope asserts use typed
    // `OccScope::contains_node` / `contains_edge_props`. Remaining 34 =
    // ~30 production typed-helper internals + R165 intentional production
    // residuals (raw-byte edgeprop transfer, prefix-scan harvest) + 4
    // legit-raw cfg(test) sites that MUST build raw keys: three
    // decode-error tests planting non-MessagePack garbage at node /
    // temporal-node / edgeprop keys, and one white-box flush test poking
    // a dummy key into `mvcc_write_buffer`.
    ("src/executor/runner.rs", 34),
    // vector_predicate.rs: one raw encode_node_key on the ACORN-filtered
    // hot path (predicate evaluator point-get, production) + one
    // cfg(test) corrupt-record test that plants invalid bytes at a raw
    // node key. The valid-fixture seed migrated to LocalNodeStore (R166).
    // Total = 2.
    ("src/executor/vector_predicate.rs", 2),
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
        // Extracted unit-test modules live in sibling files whose names end in
        // `tests.rs` (`<module>/tests.rs`, plus multi-module cases like
        // `runner/fusion_kernel_tests.rs`, `builder/numeric_predicate_tests.rs`),
        // moved out of the inline `#[cfg(test)] mod ...`. They are test code,
        // exempt from the production encoder lockdown exactly as the inline
        // `cfg(test)` modules they replaced were — fixtures may build raw probe
        // keys (migrating in R166). No production source file is named `*tests.rs`,
        // so this gates production only.
        if path
            .file_name()
            .is_some_and(|n| n.to_string_lossy().ends_with("tests.rs"))
        {
            continue;
        }
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

/// Enforce that the integration test suites contain ZERO raw key-encoder
/// calls: fixtures are seeded and storage state asserted through the
/// Layer-4 stores (`LocalNodeStore` / `LocalEdgeStore` / `LocalIndexStore`)
/// and typed `OccScope` probes, never by reaching into the partition key
/// layout. This file itself is skipped — it holds the needle strings as
/// search literals, not as encoder calls.
#[test]
fn encoder_lockdown_tests_dir_is_clean() {
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
        if path.file_name().is_some_and(|n| n == "encoder_lockdown.rs") {
            continue;
        }
        let rel = path
            .strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let content = fs::read_to_string(path).unwrap_or_default();
        for needle in TEST_ENCODER_NEEDLES {
            if content.contains(needle) {
                bad.push(format!(
                    "{rel} uses `{needle}` — seed fixtures and assert storage \
                     state through the Layer-4 stores (LocalNodeStore / \
                     LocalEdgeStore / LocalIndexStore) and typed OccScope \
                     probes, not raw key encoders.",
                ));
            }
        }
    }

    assert!(
        bad.is_empty(),
        "tests/ raw key-encoder regressions:\n{}",
        bad.join("\n"),
    );
}
