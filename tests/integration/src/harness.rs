//! Process harness: spawn `coordinode` binary, wait for gRPC readiness,
//! provide typed clients, and kill on drop.
//!
//! Panics and expect() are intentional here — test infrastructure failures
//! should abort with a clear message rather than being silently swallowed.

// Test harness: panic!/expect!/unwrap! are appropriate for infrastructure failures.
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::process::CommandExt as _;

use crate::proto::{
    admin::cluster_service_client::ClusterServiceClient,
    graph::schema_service_client::SchemaServiceClient,
    query::cypher_service_client::CypherServiceClient,
};

/// A running coordinode process bound to an ephemeral port + temp data dir.
///
/// Killed (and data dir removed) when dropped. Use [`CoordinodeProcess::restart`]
/// to kill → respawn while keeping the same data directory (restart regression tests).
pub struct CoordinodeProcess {
    child: Child,
    pub port: u16,
    // Wrapped in Option so `restart()` can take it without needing unsafe.
    // Always `Some` except briefly during `restart()`.
    data_dir: Option<tempfile::TempDir>,
}

impl CoordinodeProcess {
    /// Spawn `coordinode serve` against `data_dir` on a free ephemeral port.
    ///
    /// Waits up to 15 seconds for the gRPC port to become available.
    pub async fn start() -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        let port = free_port();
        let child = spawn_binary(port, data_dir.path().to_path_buf());
        let proc = Self {
            child,
            port,
            data_dir: Some(data_dir),
        };
        proc.wait_for_grpc(Duration::from_secs(15)).await;
        proc
    }

    /// Spawn a cluster member: `serve --node-id N --addr [::1]:port
    /// --advertise-addr http://[::1]:port --peers <peer advertise addrs> ...`.
    ///
    /// `node_id == 1` bootstraps as the single-voter leader (the cluster grows
    /// via `JoinNode`); `node_id > 1` starts in joining-wait state until the
    /// leader adds it. `peer_ports` are the gRPC ports of the OTHER members —
    /// a non-empty list is what puts the binary in cluster mode (and `node_id
    /// > 1` requires it).
    ///
    /// Waits only for the gRPC port to accept TCP: a joining member is not a
    /// leader, so [`wait_for_leader`](Self::wait_for_leader) would never return
    /// for it.
    pub async fn start_cluster_member(node_id: u64, port: u16, peer_ports: &[u16]) -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        let peers: Vec<String> = peer_ports
            .iter()
            .map(|p| format!("http://[::1]:{p}"))
            .collect();
        let child = spawn_cluster_binary(node_id, port, &peers, data_dir.path().to_path_buf());
        let proc = Self {
            child,
            port,
            data_dir: Some(data_dir),
        };
        proc.wait_for_grpc(Duration::from_secs(15)).await;
        proc
    }

    /// Crash the running process with SIGKILL then re-spawn against the same
    /// data directory.
    ///
    /// Unlike [`restart`](Self::restart) (which uses SIGTERM for graceful
    /// shutdown), this method simulates an unclean shutdown: no memtable flush,
    /// no WAL seal, no LSM key writes that happen after fsync.
    ///
    /// The server must be able to restart cleanly from the on-disk state even
    /// after SIGKILL — this is what crash-recovery tests verify.
    pub async fn restart_unclean(mut self) -> Self {
        // SIGKILL: immediate termination, no cleanup, no Drop.
        let _ = self.child.kill();
        let _ = self.child.wait();

        // Take the TempDir out before self is dropped.
        let data_dir = self
            .data_dir
            .take()
            .expect("data_dir missing — restart called twice?");

        let port = free_port();
        let data_path = data_dir.path().to_path_buf();
        // `self` drops here: child already waited, data_dir is None.

        let child = spawn_binary(port, data_path);
        let proc = Self {
            child,
            port,
            data_dir: Some(data_dir),
        };
        proc.wait_for_grpc(Duration::from_secs(15)).await;
        // After SIGKILL the Raft node must re-elect itself as leader.
        // wait_for_leader retries PRIMARY reads until election completes.
        proc.wait_for_leader(Duration::from_secs(10)).await;
        proc
    }

    /// Kill the running process then re-spawn against the same data directory.
    ///
    /// This simulates a server restart while keeping persisted data intact.
    /// A fresh ephemeral port is chosen to avoid "address already in use" races.
    ///
    /// Sends SIGTERM so the server can flush its LSM memtable before exiting
    /// (StorageEngine::Drop is called during graceful shutdown). Falls back to
    /// SIGKILL after 10 s if the process does not exit on its own.
    pub async fn restart(mut self) -> Self {
        // Graceful shutdown (SIGTERM lets StorageEngine::Drop flush memtables to
        // SST files), 10 s grace for the flush + file sync, then SIGKILL. The
        // wait MUST be async: the server drains open client connections before
        // it exits, and the tonic channels this test dropped only close when
        // the runtime gets to poll them. A blocking sleep here starves the
        // current-thread runtime, the connections never close, the grace
        // period expires and SIGKILL loses the unflushed memtable.
        send_sigterm(&self.child);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline && !has_exited(&mut self.child) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        force_reap(&mut self.child);

        // Take the TempDir out of self before self is dropped.
        // This means the Drop impl for this struct won't delete the directory.
        let data_dir = self
            .data_dir
            .take()
            .expect("data_dir missing — restart called twice?");

        // Pick a NEW port — the old one may still be in TIME_WAIT.
        let port = free_port();
        let data_path = data_dir.path().to_path_buf();
        // `self` drops here: child is already killed, data_dir is None → no cleanup.

        let child = spawn_binary(port, data_path);
        let proc = Self {
            child,
            port,
            data_dir: Some(data_dir),
        };
        proc.wait_for_grpc(Duration::from_secs(15)).await;
        proc
    }

    /// gRPC endpoint URL for use with tonic.
    pub fn endpoint(&self) -> String {
        format!("http://[::1]:{}", self.port)
    }

    /// Build a `SchemaServiceClient` connected to this process.
    pub async fn schema_client(&self) -> SchemaServiceClient<tonic::transport::Channel> {
        let channel = tonic::transport::Endpoint::from_shared(self.endpoint())
            .expect("valid endpoint")
            .connect()
            .await
            .expect("connect to schema service");
        SchemaServiceClient::new(channel)
    }

    /// Build a `CypherServiceClient` connected to this process.
    pub async fn cypher_client(&self) -> CypherServiceClient<tonic::transport::Channel> {
        let channel = tonic::transport::Endpoint::from_shared(self.endpoint())
            .expect("valid endpoint")
            .connect()
            .await
            .expect("connect to cypher service");
        CypherServiceClient::new(channel)
    }

    /// Build a `ClusterServiceClient` connected to this process.
    pub async fn cluster_client(&self) -> ClusterServiceClient<tonic::transport::Channel> {
        let channel = tonic::transport::Endpoint::from_shared(self.endpoint())
            .expect("valid endpoint")
            .connect()
            .await
            .expect("connect to cluster service");
        ClusterServiceClient::new(channel)
    }

    /// Block (async) until the gRPC port accepts TCP connections or `timeout` elapses.
    async fn wait_for_grpc(&self, timeout: Duration) {
        let deadline = Instant::now() + timeout;
        loop {
            if std::net::TcpStream::connect(format!("[::1]:{}", self.port)).is_ok() {
                // Small extra sleep to let gRPC handshake initialise.
                tokio::time::sleep(Duration::from_millis(100)).await;
                return;
            }
            if Instant::now() >= deadline {
                panic!(
                    "coordinode did not start on port {} within {:?}",
                    self.port, timeout
                );
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// Block (async) until this node is the Raft leader and accepts PRIMARY reads.
    ///
    /// After an unclean shutdown (SIGKILL) the single-node Raft instance must
    /// re-elect itself as leader.  TCP connectivity is available before leader
    /// election completes, so this method retries a trivial PRIMARY read until
    /// it succeeds or `timeout` elapses.
    pub async fn wait_for_leader(&self, timeout: Duration) {
        use crate::proto::query::{
            ExecuteCypherRequest, cypher_service_client::CypherServiceClient,
        };

        let deadline = Instant::now() + timeout;
        loop {
            let channel = tonic::transport::Endpoint::from_shared(self.endpoint())
                .expect("valid endpoint")
                .connect_lazy();
            let mut client = CypherServiceClient::new(channel);
            let result = client
                .execute_cypher(ExecuteCypherRequest {
                    query: "RETURN 1".to_string(),
                    parameters: Default::default(),
                    read_preference: 0, // PRIMARY
                    read_concern: None,
                    write_concern: None,
                    transaction_id: 0, // auto-commit (no interactive transaction)
                })
                .await;
            if result.is_ok() {
                return;
            }
            if Instant::now() >= deadline {
                panic!(
                    "coordinode on port {} did not become Raft leader within {:?}",
                    self.port, timeout
                );
            }
            tokio::time::sleep(Duration::from_millis(150)).await;
        }
    }
}

impl Drop for CoordinodeProcess {
    fn drop(&mut self) {
        // Graceful shutdown first (SIGTERM so StorageEngine::Drop can flush
        // memtables), SIGKILL after a 5 s grace. This is a synchronous
        // destructor, so the poll blocks; the child is reaped on every path
        // (see `force_reap`), so no coordinode process can outlive the test
        // that spawned it.
        if has_exited(&mut self.child) {
            return; // restart() / restart_unclean() already reaped it
        }
        send_sigterm(&self.child);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !has_exited(&mut self.child) {
            std::thread::sleep(Duration::from_millis(50));
        }
        force_reap(&mut self.child);
        // data_dir: Option<TempDir> is dropped here.
        // In the restart() path it's already None — no cleanup happens.
        // In the normal path it's Some — directory is cleaned up.
    }
}

/// SIGTERM via the system `kill` utility: no `nix` crate, no unsafe.
fn send_sigterm(child: &Child) {
    let _ = Command::new("kill")
        .args(["-s", "TERM", &child.id().to_string()])
        .status();
}

/// Non-blocking "has the child exited?" probe. A `try_wait` error (a transient
/// `waitpid` failure) reads as "still running": the caller keeps polling and
/// ends in [`force_reap`], so an error can never abandon a live process.
fn has_exited(child: &mut Child) -> bool {
    matches!(child.try_wait(), Ok(Some(_)))
}

/// Make sure `child` is dead and reaped. A no-op for an already-exited child
/// (`kill` on a reaped child is an error we ignore, `wait` returns the cached
/// status); for a live one it is SIGKILL + `wait()`.
///
/// The child inherits the test's stdout/stderr, and nextest marks a test LEAKY
/// when those pipes are still held open after the test exits, which is exactly
/// what a coordinode process left alive would do. Every shutdown path in this
/// harness therefore ends here, never in an early return with the child alive.
fn force_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the path to the `coordinode` binary.
///
/// Uses `COORDINODE_BIN` env var first (CI / explicit override), then falls
/// back to the Cargo debug build in the workspace target directory.
///
/// Exposed `pub` so integration tests can directly `Command::new(binary_path())`
/// for non-standard startup scenarios (e.g. checking `--mode=compute` is rejected).
pub fn binary_path() -> PathBuf {
    if let Ok(path) = std::env::var("COORDINODE_BIN") {
        return PathBuf::from(path);
    }

    // Walk up from this crate's manifest dir to the workspace root.
    // tests/integration/ → tests/ → workspace root (2 levels up).
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    let bin = workspace.join("target/debug/coordinode");
    if bin.exists() {
        return bin;
    }
    let release = workspace.join("target/release/coordinode");
    if release.exists() {
        return release;
    }

    panic!(
        "coordinode binary not found.\n\
         Build it first: cargo build -p coordinode-server\n\
         Or set COORDINODE_BIN=/path/to/coordinode"
    );
}

/// Spawn `coordinode serve --addr [::1]:PORT --ops-addr [::1]:0 --data DATA_DIR`.
///
/// `--ops-addr [::1]:0` lets the OS assign an ephemeral port for the ops HTTP
/// server, avoiding "Address already in use" conflicts when multiple test
/// processes run concurrently (each would otherwise fight over the default :7084).
fn spawn_binary(port: u16, data_dir: PathBuf) -> Child {
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--ops-addr")
        .arg("[::1]:0")
        .arg("--data")
        .arg(&data_dir)
        // Suppress server logs from test output; set RUST_LOG=debug for debugging.
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
        );

    // Place the child in its own process group (Unix only), so a Ctrl-C /
    // SIGINT delivered to the test runner's group is not fanned out to the
    // server mid-flush; the harness owns the child's lifetime through
    // `terminate_and_reap` (kill by PID, group-independent).
    //
    // Note this does NOT hide the child from nextest's leak detection, which
    // is based on the inherited stdout/stderr pipes, not on process groups:
    // the only thing that prevents a LEAKY verdict is actually reaping the
    // child before the test exits, which the Drop impl guarantees.
    #[cfg(unix)]
    cmd.process_group(0);

    cmd.spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e))
}

/// Spawn a cluster member with explicit `--node-id`, `--advertise-addr`, and
/// `--peers`. See [`CoordinodeProcess::start_cluster_member`].
fn spawn_cluster_binary(node_id: u64, port: u16, peers: &[String], data_dir: PathBuf) -> Child {
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg("--node-id")
        .arg(node_id.to_string())
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--advertise-addr")
        .arg(format!("http://[::1]:{port}"))
        .arg("--peers")
        .arg(peers.join(","))
        .arg("--ops-addr")
        .arg("[::1]:0")
        .arg("--data")
        .arg(&data_dir)
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
        );

    #[cfg(unix)]
    cmd.process_group(0);

    cmd.spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e))
}

/// Bind port 0 to get a free ephemeral port from the OS.
///
/// Exposed `pub` so multi-node tests can pre-allocate every member's port
/// before spawning (each member's `--peers` needs the others' ports up front).
pub fn free_port() -> u16 {
    // Bind to [::1] with port 0 — the OS assigns a free port.
    // We immediately close the listener so coordinode can bind the same port.
    // Tiny race window, but acceptable for local integration tests.
    let listener = TcpListener::bind("[::1]:0").expect("bind [::1]:0");
    listener.local_addr().expect("local addr").port()
}
