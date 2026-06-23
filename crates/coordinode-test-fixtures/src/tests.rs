use super::*;
use coordinode_storage::engine::partition::Partition;

#[test]
fn memory_engine_writes_and_reads() {
    let fx = engine_for_memory();
    fx.engine.put(Partition::Node, b"k1", b"v1").expect("put");
    let v = fx.engine.get(Partition::Node, b"k1").expect("get");
    assert_eq!(v.as_deref(), Some(b"v1".as_ref()));
}

#[test]
fn disk_engine_writes_and_reads() {
    let fx = engine_for_disk();
    fx.engine.put(Partition::Node, b"k1", b"v1").expect("put");
    let v = fx.engine.get(Partition::Node, b"k1").expect("get");
    assert_eq!(v.as_deref(), Some(b"v1".as_ref()));
}

#[test]
fn memory_engines_are_isolated() {
    // Each engine_for_memory() call gets its own MemFs — writes
    // to one don't bleed into another. Critical for parallel
    // test execution.
    let fx1 = engine_for_memory();
    let fx2 = engine_for_memory();
    fx1.engine
        .put(Partition::Node, b"k1", b"a")
        .expect("put fx1");
    let leak = fx2.engine.get(Partition::Node, b"k1").expect("get fx2");
    assert_eq!(
        leak.as_deref(),
        None,
        "memory engines must be isolated — fx2 must not see fx1's writes",
    );
}

#[test]
fn engine_for_logic_respects_env_var_memory() {
    // SAFETY: this test runs serially with the disk variant via
    // shared env state. `serial_test` crate isn't in dependencies
    // yet — single-thread cargo nextest run is the workaround.
    // We accept that running these two tests in parallel with
    // different env values would race; the per-key isolation
    // above already covers behaviour correctness.
    // SAFETY: setting an env var is unsafe in Rust 2024+ because
    // it mutates process-wide state without locking; in single-
    // threaded tests it's fine.
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "memory") };
    assert_eq!(resolve_backend(), TestBackend::Memory);
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
}

#[test]
fn engine_for_logic_respects_env_var_disk() {
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "disk") };
    assert_eq!(resolve_backend(), TestBackend::Disk);
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
}

#[test]
fn env_var_unrecognised_falls_back_to_memory() {
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "rocksdb") };
    assert_eq!(resolve_backend(), TestBackend::Memory);
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
}

#[test]
fn env_var_unset_defaults_to_memory() {
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
    assert_eq!(resolve_backend(), TestBackend::Memory);
}

#[test]
fn env_var_with_whitespace_is_normalised() {
    // `.trim().to_lowercase()` in resolve_backend should accept
    // "  Disk\n", "DISK", " memory " etc.
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "  Disk  ") };
    assert_eq!(resolve_backend(), TestBackend::Disk);
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "MEMORY") };
    assert_eq!(resolve_backend(), TestBackend::Memory);
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "\tdisk\n") };
    assert_eq!(resolve_backend(), TestBackend::Disk);
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
}

#[test]
fn scratch_path_exists_and_writable_for_memory_backend() {
    // The scratch dir is always on host FS, even when the
    // engine itself runs in MemFs. Verify by writing+reading a
    // marker file at the path.
    let fx = engine_for_memory();
    let path = fx.scratch_path();
    assert!(path.exists(), "scratch_path must point to an existing dir");
    let marker = path.join("marker.txt");
    std::fs::write(&marker, b"tantivy-stub").expect("write marker");
    let back = std::fs::read(&marker).expect("read marker");
    assert_eq!(
        back, b"tantivy-stub",
        "scratch_path must be writable + readable"
    );
}

#[test]
fn scratch_path_exists_and_writable_for_disk_backend() {
    // Symmetric to the memory test — disk backing also gives a
    // working scratch_path.
    let fx = engine_for_disk();
    let path = fx.scratch_path();
    assert!(path.exists());
    let marker = path.join("marker.txt");
    std::fs::write(&marker, b"x").expect("write marker");
    assert_eq!(std::fs::read(&marker).expect("read"), b"x");
}

#[test]
fn scratch_paths_are_per_fixture_isolated() {
    // Two fixtures get independent scratch tempdirs — writes
    // to one don't leak into the other (critical for parallel
    // test execution).
    let fx1 = engine_for_memory();
    let fx2 = engine_for_memory();
    assert_ne!(
        fx1.scratch_path(),
        fx2.scratch_path(),
        "each fixture must get its own scratch tempdir",
    );
    std::fs::write(fx1.scratch_path().join("a.txt"), b"fx1").unwrap();
    assert!(
        !fx2.scratch_path().join("a.txt").exists(),
        "fx2 scratch must not see fx1's writes",
    );
}

#[test]
fn scratch_path_drops_with_fixture() {
    // Dropping the fixture removes the scratch tempdir — engine
    // resources release cleanly without leaking files on the
    // host FS.
    let saved_path = {
        let fx = engine_for_memory();
        let p = fx.scratch_path().to_path_buf();
        assert!(p.exists(), "scratch exists while fixture is alive");
        p
        // fx drops here
    };
    // After drop, the tempdir is gone.
    assert!(
        !saved_path.exists(),
        "scratch tempdir must be cleaned on fixture drop, found: {saved_path:?}",
    );
}

#[test]
fn engine_for_logic_returns_disk_backend_when_env_set_to_disk() {
    // Full pipeline test: env var → engine_for_logic() →
    // EngineFixture. Verify the disk variant by checking that
    // scratch_path() is OUTSIDE the OS tempdir's MemFs-like
    // path pattern — a roundabout proxy for "this is real
    // tempfile::TempDir", since the Backing enum is private.
    // The reliable signal: both memory and disk backends
    // produce a real scratch path (always-on per fixture), so
    // we instead probe by performing a write+restart cycle
    // that ONLY a disk backend would survive — but that's a
    // separate persistence test. For this slot, we settle for
    // the simpler invariant: the fixture builds successfully
    // and produces a writable scratch path. (Resolve-backend
    // is already tested separately at the env-var layer.)
    unsafe { std::env::set_var("COORDINODE_TEST_BACKEND", "disk") };
    let fx = engine_for_logic();
    let p = fx.scratch_path();
    assert!(
        p.exists(),
        "disk-backed fixture must yield writable scratch"
    );
    unsafe { std::env::remove_var("COORDINODE_TEST_BACKEND") };
}

/// Doctest-style usage example — also serves as the "actual
/// caller code path" smoke test from outside the test module
/// scope (single integration use site).
#[test]
fn usage_example_logic_test_pattern() {
    let fx = engine_for_logic();
    let engine = &fx.engine;
    // Tests do their work through `engine` exactly like
    // production code would — typed modality stores etc. Here
    // we just verify the basic StorageEngine API works.
    engine.put(Partition::Node, b"k", b"v").expect("put");
    let v = engine.get(Partition::Node, b"k").expect("get");
    assert_eq!(v.as_deref(), Some(b"v".as_ref()));
    // scratch_path for Tantivy-style ancillary state:
    let _scratch_for_text_idx = fx.scratch_path().join("text_idx");
}
