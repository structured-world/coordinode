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
    // runner.rs current baseline: typed-helper internals call the raw
    // encoders by construction (~15 calls across mvcc_*_node /
    // mvcc_*_edge_props helpers), plus test fixtures in the cfg(test)
    // module use them to build probe keys (~37 calls). Total = 52.
    // Lowering this means a test-fixture migration (R166); raising
    // it means new raw-encoder usage in production — reject.
    ("src/executor/runner.rs", 52),
    // ops.rs — fully routed through LocalIndexStore after slice 12.
    ("src/index/ops.rs", 0),
    // build.rs (R165 slice 2): 1 residual usage in the cfg(test)
    // fixture for seeding probe nodes; production path migrated.
    ("src/index/build.rs", 1),
    // ttl.rs (R165 slice 1): 3 residual usages in cfg(test) fixtures.
    ("src/index/ttl.rs", 3),
    // ttl_reaper.rs: 9 residual usages — production discover_ttl_targets
    // uses schema:label: prefix encoder (legitimate non-node encoder),
    // and cfg(test) fixtures use raw node-key encoders for probe data.
    // Pending R166 + a SchemaStore::list_labels API.
    ("src/index/ttl_reaper.rs", 9),
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
