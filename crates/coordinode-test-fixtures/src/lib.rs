//! Test fixture helpers — `engine_for_logic()`, `engine_for_disk()`,
//! `engine_for_memory()`.
//!
//! **no-std tier:** `std-only`. Test fixtures inherently need
//! `std::env` (env var lookup), `std::path::PathBuf`, `tempfile`
//! crate, and full `lsm_tree::fs::StdFs` / `MemFs` plumbing. Out
//! of scope for any no-std readiness — this crate exists only to
//! be a dev-dependency of other crates' test suites.
//!
//! **Purpose**: avoid splitting tempfile across `[dependencies]` +
//! `[dev-dependencies]` in `coordinode-storage`. By extracting the
//! fixture helpers into a standalone crate, downstream consumers
//! depend on `coordinode-test-fixtures` only in their own
//! `[dev-dependencies]` — no feature gates, no optional deps, no
//! Cargo duplicate-key conflicts.
//!
//! ## Purpose
//!
//! CoordiNode tests have two distinct shapes:
//!
//! 1. **Logic tests** — verify behaviour that doesn't depend on disk
//!    semantics: schema CRUD, OCC scope tracking, query plan
//!    correctness, modality store contracts. These should run on
//!    [`lsm_tree::fs::MemFs`] (in-memory FS) for 2–5× speed-up and
//!    cleaner isolation.
//! 2. **Persistence tests** — verify behaviour that *requires* a
//!    real disk: WAL recovery (R076a / R091a), Tier-2 bucket reopen,
//!    SST flush + reopen round-trips, crash safety. These must run
//!    on a tempdir-backed [`lsm_tree::fs::StdFs`] because they
//!    exercise the actual durability path.
//!
//! Picking the wrong fixture either burns time (logic test on disk)
//! or silently skips a real bug (persistence test on MemFs — a
//! crash-recovery test on in-memory FS proves nothing).
//!
//! ## Selection
//!
//! Most tests should use [`engine_for_logic`] (which honours the
//! `COORDINODE_TEST_BACKEND` env var) and let the default fall to
//! memory. Tests that genuinely exercise disk semantics MUST call
//! [`engine_for_disk`] explicitly — the env var is ignored for those
//! so CI can run the matrix without breaking persistence tests.
//!
//! ## Env var: `COORDINODE_TEST_BACKEND`
//!
//! - **`memory`** (default) — use `MemFs` for [`engine_for_logic`].
//!   Fast, isolated, no FS I/O.
//! - **`disk`** — use tempdir + `StdFs` for [`engine_for_logic`].
//!   Slower, real I/O. CI matrix runs this leg to catch
//!   FS-specific bugs that MemFs would miss (path encoding,
//!   fsync ordering, file descriptor exhaustion under load).
//!
//! Unrecognised values fall back to `memory` with a warning.
//!
//! ## Returned types
//!
//! Both fixture functions return [`EngineFixture`] — a struct that
//! owns the engine PLUS the lifetime-binding state (`TempDir` for
//! disk, `Arc<MemFs>` reference for memory). Drop order matters:
//! engine must drop before the fixture state.

use std::path::PathBuf;
use std::sync::Arc;

use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};
use coordinode_storage::engine::core::StorageEngine;

/// Owned engine fixture — drops in the right order on test exit.
/// The lifetime-binding state (`TempDir` for disk, `MemFs` Arc for
/// memory) is held alongside the engine so the engine has somewhere
/// to write/read for the duration of the test.
///
/// Every fixture also carries an ancillary on-disk scratch tempdir
/// (`scratch`) usable for state that lives OUTSIDE the engine —
/// Tantivy text indexes, model files, JSON dumps. The scratch dir
/// is on the host FS regardless of engine backend (memory engines
/// keep their data in `MemFs` but the scratch tempdir is real),
/// because Tantivy and similar consumers are disk-only and benefit
/// from the same lifetime guard as the engine.
pub struct EngineFixture {
    /// The configured engine, wrapped in `Arc` so tests that spawn
    /// threads can `Arc::clone(&fx.engine)` and `move` the clone
    /// into a `'static` closure. The vast majority of callers use
    /// `&fx.engine` and rely on `Arc<T>` deref to `&T` — that path
    /// is unchanged. Only thread-spawning tests need the explicit
    /// clone.
    pub engine: Arc<StorageEngine>,
    /// Backend-specific state. For disk: a `TempDir` whose lifetime
    /// must outlast every access through `engine`. For memory: an
    /// `Arc<MemFs>` that keeps the in-memory FS alive. Field is not
    /// read directly — its lifetime IS the value.
    #[allow(dead_code)]
    backing: Backing,
    /// Ancillary on-disk scratch directory — same lifetime as the
    /// engine. Tests use this for Tantivy index roots, etc. Always
    /// present (memory engines get a real tempdir even though
    /// their engine bytes live in MemFs).
    scratch: tempfile::TempDir,
}

impl EngineFixture {
    /// On-disk scratch path for ancillary state — Tantivy index
    /// root, ML model files, dump fixtures. Same lifetime as the
    /// engine. Available regardless of engine backend.
    pub fn scratch_path(&self) -> &std::path::Path {
        self.scratch.path()
    }
}

/// Backend-specific state. Internal — tests interact only with
/// `EngineFixture::engine`. The enum variants hold lifetime-binding
/// state which is never read directly; the `#[allow(dead_code)]`
/// silences the warning because the data is meaningful (drop order
/// matters) even though no field is ever accessed.
#[allow(dead_code)]
enum Backing {
    /// Tempdir holding real disk files. Cleaned on drop.
    Disk(tempfile::TempDir),
    /// In-memory FS instance shared with the engine.
    Memory(Arc<lsm_tree::fs::MemFs>),
}

/// Build an engine for **logic** tests. Honours
/// `COORDINODE_TEST_BACKEND` (defaults to `memory`).
///
/// Use this for the 95% of tests that verify behaviour without
/// depending on disk semantics. For tests that need real disk (WAL
/// recovery, crash safety, Tier-2 reopen) call [`engine_for_disk`]
/// directly instead.
///
/// # Examples
///
/// ```
/// use coordinode_test_fixtures::engine_for_logic;
/// use coordinode_storage::engine::partition::Partition;
///
/// let fx = engine_for_logic();
/// let engine = &fx.engine;
/// engine.put(Partition::Node, b"hello", b"world").unwrap();
/// assert_eq!(
///     engine.get(Partition::Node, b"hello").unwrap().as_deref(),
///     Some(b"world".as_ref()),
/// );
/// // Tantivy text indexes etc. use the always-on scratch_path:
/// let _text_idx_root = fx.scratch_path().join("text_idx");
/// ```
///
/// # Panics
///
/// Panics if the storage engine fails to open. Test-only — production
/// code MUST NOT call this function (it's not in the `pub` API at
/// crate root precisely because of these panics).
pub fn engine_for_logic() -> EngineFixture {
    match resolve_backend() {
        TestBackend::Memory => engine_for_memory(),
        TestBackend::Disk => engine_for_disk(),
    }
}

/// Build an engine for **persistence** tests. Always returns a
/// tempdir-backed engine on `StdFs`, regardless of env var. Use this
/// for tests that exercise WAL recovery, crash safety, SST flush +
/// reopen round-trips, or any other behaviour that MemFs would
/// silently mock out.
///
/// # Panics
///
/// Same as [`engine_for_logic`] — test-only.
pub fn engine_for_disk() -> EngineFixture {
    let dir = tempfile::TempDir::new().expect("create tempdir");
    let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
        "default",
        dir.path(),
        Media::Hdd,
        Durability::Durable,
        Tier::Warm,
    )]);
    let engine = Arc::new(StorageEngine::open(&cfg).expect("open disk engine"));
    let scratch = tempfile::TempDir::new().expect("create scratch tempdir");
    EngineFixture {
        engine,
        backing: Backing::Disk(dir),
        scratch,
    }
}

/// Build an engine on `MemFs`. Internal — most tests should call
/// [`engine_for_logic`] which respects the env var. Exposed for tests
/// that want to force memory regardless of env var (e.g. tests that
/// would be unbearably slow on disk for irrelevant reasons).
pub fn engine_for_memory() -> EngineFixture {
    // Virtual path under MemFs root. The path doesn't have to exist
    // on the host FS — MemFs maintains its own tree under this root.
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
    let engine = Arc::new(StorageEngine::open(&cfg).expect("open memory engine"));
    let scratch = tempfile::TempDir::new().expect("create scratch tempdir");
    EngineFixture {
        engine,
        backing: Backing::Memory(fs),
        scratch,
    }
}

/// Ports this process has already handed out.
///
/// Deliberately global: the property being enforced is that no port is
/// returned twice for the lifetime of the process, and a caller that had to
/// thread state through to get that could simply forget to. Contention is
/// nil (a handful of allocations per test), so a plain std mutex is enough
/// and keeps this crate free of another dependency.
static HANDED_OUT: std::sync::Mutex<Vec<u16>> = std::sync::Mutex::new(Vec::new());

/// Reserve a loopback port for a test that needs to know an address before
/// the code under test binds it.
///
/// Binding to `:0` asks the kernel for a free port, but the probe has to be
/// closed before the port can be handed over, and a closed probe leaves the
/// port free for the next one to be given the same answer. A test that
/// allocates several addresses up front could therefore hand two nodes the
/// same port, and the second to bind it dies with `EADDRINUSE`. Remembering
/// what has been handed out and asking again on a repeat removes that.
///
/// What remains is a genuine race with the rest of the machine: the port is
/// free between this call and the bind, so bind as early as the code under
/// test allows.
pub fn alloc_port() -> u16 {
    // Generous relative to the number of ports a test asks for, and small
    // enough that a kernel with nothing left to give fails the test rather
    // than hanging it.
    const ATTEMPTS: usize = 64;

    for _ in 0..ATTEMPTS {
        let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind :0 for port alloc");
        let port = probe.local_addr().expect("local_addr").port();
        drop(probe);

        let mut handed_out = HANDED_OUT.lock().expect("port registry poisoned");
        if !handed_out.contains(&port) {
            handed_out.push(port);
            return port;
        }
    }
    panic!("no unused loopback port after {ATTEMPTS} attempts");
}

/// Reserve `N` distinct loopback ports at once, for tests that read better
/// naming them together. Each comes from [`alloc_port`], so they are
/// distinct for the same reason.
///
/// ```
/// let [a, b] = coordinode_test_fixtures::alloc_ports();
/// assert_ne!(a, b);
/// ```
pub fn alloc_ports<const N: usize>() -> [u16; N] {
    std::array::from_fn(|_| alloc_port())
}

/// Which backend the env var selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestBackend {
    Memory,
    Disk,
}

/// Parse `COORDINODE_TEST_BACKEND`. Defaults to `Memory`. Unrecognised
/// values warn (to stderr — tracing isn't configured in most tests)
/// and fall back to `Memory`.
fn resolve_backend() -> TestBackend {
    match std::env::var("COORDINODE_TEST_BACKEND")
        .as_deref()
        .map(str::trim)
        .map(str::to_lowercase)
    {
        Ok(s) if s == "memory" => TestBackend::Memory,
        Ok(s) if s == "disk" => TestBackend::Disk,
        Ok(other) => {
            eprintln!(
                "COORDINODE_TEST_BACKEND={other:?} unrecognised — \
                 expected 'memory' or 'disk'; falling back to memory"
            );
            TestBackend::Memory
        }
        Err(_) => TestBackend::Memory, // unset → default to memory
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
