//! cfg(test)-only fixture helpers for tests INSIDE `coordinode-storage`.
//!
//! ## Why not use `coordinode-test-fixtures`?
//!
//! `coordinode-test-fixtures` depends on `coordinode-storage` for the
//! `StorageEngine` type. A storage-internal test that imported the
//! fixture crate would create a cyclic dependency. So this small
//! shim lives inline: same dual-FS pattern, but built directly on
//! `StorageEngine` + `lsm_tree::fs::MemFs` without the dependency
//! hop.
//!
//! Downstream crates (`coordinode-query`, `coordinode-modality`,
//! `coordinode-embed`, etc.) still use `coordinode-test-fixtures` —
//! this module is for in-crate tests only.
//!
//! ## API
//!
//! - [`memory_engine`] — `lsm_tree::fs::MemFs`-backed `StorageEngine`
//!   for logic tests. Returns the engine plus the `MemFs` Arc so
//!   the FS lives as long as the engine.
//! - [`disk_engine`] — tempdir + `StdFs`-backed engine for tests
//!   that exercise actual persistence semantics (WAL recovery,
//!   crash safety, SST flush + reopen). Returns the engine plus
//!   the `TempDir` so the directory survives until test end.
//!
//! ## Policy
//!
//! Migrate test functions to `memory_engine()` when the test
//! verifies behaviour that doesn't depend on disk semantics
//! (OCC scope, batch composition, partition routing, snapshot
//! dispatch). Keep `disk_engine()` (or equivalent inline
//! tempdir) for tests that verify durability: WAL replay,
//! checkpoint clearing, reopen round-trips, crash recovery.

#![allow(clippy::expect_used, dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use crate::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use crate::engine::core::StorageEngine;

/// Memory-backed engine for logic tests. Returns the engine plus
/// the `MemFs` Arc to keep the FS alive (the engine has its own
/// reference; this Arc is for the test's lifetime guard).
pub(crate) fn memory_engine() -> (StorageEngine, Arc<lsm_tree::fs::MemFs>) {
    let virtual_path = PathBuf::from("/coordinode-test-memory");
    let fs = Arc::new(lsm_tree::fs::MemFs::new());
    let cfg = StorageConfig::with_endpoints_no_persistence(vec![EndpointConfig::new(
        "default-memfs",
        &virtual_path,
        Media::Ram,
        Durability::Volatile,
        Tier::Memory,
    )])
    .with_fs(Arc::clone(&fs) as Arc<dyn lsm_tree::fs::Fs>);
    let engine = StorageEngine::open(&cfg).expect("open memory engine");
    (engine, fs)
}

/// Tempdir-backed engine for persistence tests. Returns the engine
/// plus the `TempDir` whose lifetime must outlast every access
/// through the engine.
pub(crate) fn disk_engine() -> (StorageEngine, tempfile::TempDir) {
    let dir = tempfile::TempDir::new().expect("create tempdir");
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = StorageEngine::open(&cfg).expect("open disk engine");
    (engine, dir)
}
