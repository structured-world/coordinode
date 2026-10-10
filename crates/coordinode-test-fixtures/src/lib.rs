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
//!    real disk: WAL recovery, Tier-2 bucket reopen,
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

/// The filesystem operations [`PowerRig`] can fail, for crates that do not
/// depend on the storage engine's tree crate.
pub use lsm_tree::fs::FaultOp;

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

/// A durable store whose power can be cut, for crash tests.
///
/// Writes go through a fault injector over a crash simulator. Dropping an
/// engine flushes its memtables, so a plain drop is a clean shutdown;
/// [`PowerRig::cut`] first makes every write and sync fail, so that last
/// flush never reaches the disk, then rolls every file back to its last
/// fsync. Reopen with [`PowerRig::config`] to see what survived.
pub struct PowerRig {
    dir: tempfile::TempDir,
    crash: Arc<lsm_tree::fs::CrashFs>,
    faults: Arc<lsm_tree::fs::FaultInjector>,
    fs: Arc<dyn lsm_tree::fs::Fs>,
}

impl PowerRig {
    /// A fresh store in a new temporary directory.
    ///
    /// # Panics
    ///
    /// If the temporary directory cannot be created. Test-only.
    pub fn new() -> Self {
        let dir = tempfile::TempDir::new().expect("create tempdir");
        let crash = Arc::new(lsm_tree::fs::CrashFs::new(lsm_tree::fs::StdFs));
        let faults = Arc::new(lsm_tree::fs::FaultInjector::new());
        let fs: Arc<dyn lsm_tree::fs::Fs> = Arc::new(lsm_tree::fs::FaultFs::with_injector(
            lsm_tree::fs::CrashFs::clone(&crash),
            Arc::clone(&faults),
        ));
        Self {
            dir,
            crash,
            faults,
            fs,
        }
    }

    /// The store's configuration: one durable endpoint on the rig's
    /// filesystem.
    pub fn config(&self) -> StorageConfig {
        StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            self.dir.path(),
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )])
        .with_fs(Arc::clone(&self.fs))
    }

    /// Make every operation of `op` after the first `skip` fail, as a disk
    /// that stops accepting them would; pair it with [`Self::cut`] to model a
    /// power cut at that point.
    pub fn fail_from(&self, op: lsm_tree::fs::FaultOp, skip: u64) {
        use lsm_tree::fs::{Fault, FaultRule};
        self.faults
            .arm(FaultRule::new(op, Fault::Error(lsm_tree::io::ErrorKind::Other)).skip(skip));
    }

    /// Make the next `times` operations of `op` fail, then let them through
    /// again: a transient disk error.
    pub fn fail_next(&self, op: lsm_tree::fs::FaultOp, times: u64) {
        use lsm_tree::fs::{Fault, FaultRule};
        self.faults
            .arm(FaultRule::new(op, Fault::Error(lsm_tree::io::ErrorKind::Other)).times(times));
    }

    /// Lose power under `engine`, which must be its last owner: nothing
    /// written from now on is durable, the engine goes away, and every file
    /// falls back to what was fsynced before the cut.
    ///
    /// # Panics
    ///
    /// If another handle to the engine is still alive: its drop would run
    /// after power came back and write what the cut was meant to lose.
    pub fn cut(&self, engine: Arc<StorageEngine>) {
        use lsm_tree::fs::{Fault, FaultOp, FaultRule};
        let engine = Arc::into_inner(engine).expect("the cut engine must have no other owner");
        let refuse = Fault::Error(lsm_tree::io::ErrorKind::Other);
        for op in [
            FaultOp::Open,
            FaultOp::Write,
            FaultOp::SyncAll,
            FaultOp::SyncData,
            FaultOp::Rename,
        ] {
            self.faults.arm(FaultRule::new(op, refuse));
        }
        drop(engine);
        self.faults.clear();
        self.crash.crash();
    }
}

impl Default for PowerRig {
    fn default() -> Self {
        Self::new()
    }
}

/// Ports handed out are drawn from here: below the Linux ephemeral range
/// (32768-60999) and the IANA dynamic range macOS and Windows use
/// (49152-65535), so the kernel never gives one of them to an outgoing
/// connection.
const PORT_BASE: u16 = 20_000;
const PORT_SPAN: u16 = 12_000;

/// How far into [`PORT_SPAN`] this process has walked.
///
/// Deliberately global: the property being enforced is that no port is
/// returned twice for the lifetime of the process, and a caller that had to
/// thread state through to get that could simply forget to.
static NEXT_OFFSET: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);

/// Reserve a loopback port for a test that needs to know an address before
/// the code under test binds it.
///
/// A port from `:0` is an ephemeral one, and the kernel hands the same range
/// to every outgoing connection on the machine: under a parallel test run a
/// neighbouring process's client socket can take the port between this call
/// and the bind, which then fails with `EADDRINUSE`. The ports here come from
/// a range the kernel never assigns on its own. Each process starts at an
/// offset derived from its id and walks forward, so a process never repeats
/// itself and concurrent processes start far apart; a probe bind skips any
/// port something already holds.
///
/// Only test processes draw from this range, so a port is also reserved
/// against them for the rest of this process: a lock file per port, which
/// the OS releases when the process exits. A neighbour can therefore not be
/// handed a port this process allocated but has not bound yet, such as the
/// ports of a cluster whose members start one after another.
pub fn alloc_port() -> u16 {
    alloc_port_on(std::net::Ipv4Addr::LOCALHOST.into())
}

/// [`alloc_port`] for a server that binds `ip` rather than `127.0.0.1`, such
/// as `[::1]`: the probe has to bind the address the server will.
pub fn alloc_port_on(ip: std::net::IpAddr) -> u16 {
    // Generous relative to the number of ports a test asks for, and small
    // enough that a machine with nothing left to give fails the test rather
    // than hanging it.
    const ATTEMPTS: u32 = 256;
    // Coprime with the span, so consecutive process ids land far apart.
    const PID_STRIDE: u32 = 7_919;

    let start = std::process::id().wrapping_mul(PID_STRIDE);
    for _ in 0..ATTEMPTS {
        let step = NEXT_OFFSET.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
        // The remainder is below PORT_SPAN, so the sum stays below 32000.
        let offset = start.wrapping_add(step) % u32::from(PORT_SPAN);
        let port = PORT_BASE + u16::try_from(offset).expect("offset below the span");
        if reserve_port(port) && std::net::TcpListener::bind((ip, port)).is_ok() {
            return port;
        }
    }
    panic!(
        "no free port on {ip} in {PORT_BASE}..{} after {ATTEMPTS} attempts",
        PORT_BASE + PORT_SPAN
    );
}

/// Take the cross-process reservation of `port`, or report another test
/// process holds it. The lock lives until this process exits: its file is
/// never closed, so a port stays reserved exactly as long as any server this
/// process may still start on it.
fn reserve_port(port: u16) -> bool {
    let path = reservation_dir().join(port.to_string());
    let file = open_reservation(&path);
    match file.try_lock() {
        Ok(()) => {
            // Held for the life of the process; the OS drops the lock at exit.
            std::mem::forget(file);
            true
        }
        Err(std::fs::TryLockError::WouldBlock) => false,
        Err(std::fs::TryLockError::Error(e)) => {
            panic!("lock the port reservation {}: {e}", path.display())
        }
    }
}

/// The directory holding one lock file per reserved port.
fn reservation_dir() -> &'static std::path::Path {
    static DIR: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let dir = std::env::temp_dir().join("coordinode-test-ports");
        if let Err(e) = std::fs::create_dir_all(&dir) {
            panic!("create the port reservation dir {}: {e}", dir.display());
        }
        #[cfg(unix)]
        share_with_every_user(&dir);
        dir
    })
}

/// Ports are one resource per machine whoever runs the tests (a CI account
/// and a person on the same host), so the directory is shared the way `/tmp`
/// is: every user may add a file, only its owner may remove one.
#[cfg(unix)]
fn share_with_every_user(dir: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    match std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o1777)) {
        Ok(()) => {}
        // Another user created it and shared it then; only they may change it.
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {}
        Err(e) => panic!("share the port reservation dir {}: {e}", dir.display()),
    }
}

/// Open the lock file of one port, creating it when absent. A file another
/// user created cannot be opened for writing; the lock needs no write access,
/// so it is opened for reading instead.
fn open_reservation(path: &std::path::Path) -> std::fs::File {
    let writable = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path);
    match writable {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("open the port reservation {}: {e}", path.display())),
        Err(e) => panic!("open the port reservation {}: {e}", path.display()),
    }
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
