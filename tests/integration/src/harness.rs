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

use crate::proto::{
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

    /// Kill the running process then re-spawn against the same data directory.
    ///
    /// This simulates a server restart while keeping persisted data intact.
    /// A fresh ephemeral port is chosen to avoid "address already in use" races.
    ///
    /// Sends SIGTERM so the server can flush its LSM memtable before exiting
    /// (StorageEngine::Drop is called during graceful shutdown). Falls back to
    /// SIGKILL after 10 s if the process does not exit on its own.
    pub async fn restart(mut self) -> Self {
        // Send SIGTERM for graceful shutdown (allows StorageEngine::Drop to flush
        // memtables to SST files).  We shell out to the system `kill` utility so
        // we don't need the `nix` crate or any unsafe code.
        let pid = self.child.id();
        let _ = std::process::Command::new("kill")
            .args(["-s", "TERM", &pid.to_string()])
            .status();

        // Wait up to 10 s for graceful shutdown (memtable flush + file sync).
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break, // exited cleanly — memtables flushed
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        // Graceful shutdown timed out — force kill.
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                Err(_) => break,
            }
        }

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
}

impl Drop for CoordinodeProcess {
    fn drop(&mut self) {
        // If the process already exited (e.g. after restart()), nothing to do.
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }

        // Best-effort graceful shutdown: send SIGTERM so StorageEngine::Drop
        // can flush memtables.  We are in a synchronous destructor so we
        // poll try_wait() instead of doing an async wait.
        let pid = self.child.id();
        let _ = std::process::Command::new("kill")
            .args(["-s", "TERM", &pid.to_string()])
            .status();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => break,
                Ok(None) => {
                    if std::time::Instant::now() >= deadline {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                Err(_) => break,
            }
        }
        // data_dir: Option<TempDir> is dropped here.
        // In the restart() path it's already None — no cleanup happens.
        // In the normal path it's Some — directory is cleaned up.
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the path to the `coordinode` binary.
///
/// Uses `COORDINODE_BIN` env var first (CI / explicit override), then falls
/// back to the Cargo debug build in the workspace target directory.
fn binary_path() -> PathBuf {
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
    Command::new(&bin)
        .arg("serve")
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
        )
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e))
}

/// Bind port 0 to get a free ephemeral port from the OS.
fn free_port() -> u16 {
    // Bind to [::1] with port 0 — the OS assigns a free port.
    // We immediately close the listener so coordinode can bind the same port.
    // Tiny race window, but acceptable for local integration tests.
    let listener = TcpListener::bind("[::1]:0").expect("bind [::1]:0");
    listener.local_addr().expect("local addr").port()
}
