//! Process harness: spawn `coordinode` binary, wait for gRPC readiness,
//! provide typed clients, and kill on drop.
//!
//! Panics and expect() are intentional here — test infrastructure failures
//! should abort with a clear message rather than being silently swallowed.

// Test harness: panic!/expect!/unwrap! are appropriate for infrastructure failures.
#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

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
        Self::spawn_on_free_port(data_dir, spawn_binary).await
    }

    /// Spawn on a port the harness picks, retrying on a fresh port when the
    /// process exits during startup.
    ///
    /// Other test processes cannot take a port [`free_port`] handed out, but
    /// anything outside the test allocator still can, and the server then
    /// exits at its bind. Retrying on another port turns that into a restart
    /// instead of a timeout or a test that talks to someone else's server.
    async fn spawn_on_free_port(
        data_dir: tempfile::TempDir,
        spawn: impl Fn(u16, PathBuf) -> Child,
    ) -> Self {
        const ATTEMPTS: u32 = 5;
        let mut last_exit = None;
        for _ in 0..ATTEMPTS {
            let port = free_port();
            let mut proc = Self {
                child: spawn(port, data_dir.path().to_path_buf()),
                port,
                data_dir: None,
            };
            match proc.wait_for_grpc(Duration::from_secs(15)).await {
                Ok(()) => {
                    proc.data_dir = Some(data_dir);
                    return proc;
                }
                Err(status) => last_exit = Some((port, status)),
            }
        }
        panic!("coordinode exited during startup on {ATTEMPTS} ports; last: {last_exit:?}");
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
        let mut proc = Self {
            child,
            port,
            data_dir: Some(data_dir),
        };
        // The port is fixed by the caller (peers already name it), so a
        // process that lost it cannot move to another one.
        if let Err(status) = proc.wait_for_grpc(Duration::from_secs(15)).await {
            panic!("cluster member {node_id} exited during startup on port {port}: {status}");
        }
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

        // `self` drops at the end of this call: child already waited, data_dir is None.
        let proc = Self::spawn_on_free_port(data_dir, spawn_binary).await;
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

        // A NEW port: the old one may still be in TIME_WAIT. `self` drops at
        // the end of this call: child is already killed, data_dir is None.
        Self::spawn_on_free_port(data_dir, spawn_binary).await
    }

    /// Stop the process and bring the SAME data directory back up as a cluster
    /// member, the way an operator turns a single machine into a replicated
    /// one: same data, `--node-id` and `--peers` added.
    ///
    /// The port changes because the old one may still be in `TIME_WAIT`, which
    /// is what a member's advertised address is for: peers dial what the
    /// membership records, not what it recorded yesterday.
    pub async fn restart_as_cluster_member(mut self, node_id: u64, peer_ports: &[u16]) -> Self {
        send_sigterm(&self.child);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline && !has_exited(&mut self.child) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        force_reap(&mut self.child);

        let data_dir = self
            .data_dir
            .take()
            .expect("data_dir missing: restart called twice?");
        let peers: Vec<String> = peer_ports
            .iter()
            .map(|p| format!("http://[::1]:{p}"))
            .collect();
        Self::spawn_on_free_port(data_dir, |port, data| {
            spawn_cluster_binary(node_id, port, &peers, data)
        })
        .await
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

    /// Block (async) until this process's gRPC port accepts TCP connections,
    /// or return the exit status if the process ended first.
    ///
    /// A connection alone does not prove the port is ours: another holder may
    /// answer while this process fails its bind. The server binds before
    /// opening storage and exits at once when the port is taken, so the
    /// process still running a moment after the port answers is the check.
    async fn wait_for_grpc(&mut self, timeout: Duration) -> Result<(), std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(status);
            }
            if std::net::TcpStream::connect(format!("[::1]:{}", self.port)).is_ok() {
                // Small extra sleep to let gRPC handshake initialise.
                tokio::time::sleep(Duration::from_millis(100)).await;
                return match self.child.try_wait() {
                    Ok(Some(status)) => Err(status),
                    _ => Ok(()),
                };
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

    /// Stop the process and hand the data directory over, so a test can start
    /// something else against the same bytes. The directory lives as long as
    /// the returned handle.
    pub async fn stop_keeping_data(mut self) -> tempfile::TempDir {
        send_sigterm(&self.child);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline && !has_exited(&mut self.child) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        force_reap(&mut self.child);
        self.data_dir
            .take()
            .expect("data_dir missing: the process was already restarted")
    }
}

/// Start a cluster member that is expected to refuse to start, and return what
/// the operator sees: the exit status and everything the process printed.
///
/// Waits for the process to exit rather than for a port, because the point of
/// the call is that no port is ever served.
pub async fn start_cluster_member_expecting_refusal(
    node_id: u64,
    port: u16,
    peer_ports: &[u16],
    data_dir: &std::path::Path,
) -> (std::process::ExitStatus, String) {
    let peers: Vec<String> = peer_ports
        .iter()
        .map(|p| format!("http://[::1]:{p}"))
        .collect();
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
        .arg("--rest-addr")
        .arg("[::1]:0")
        .arg("--data")
        .arg(data_dir)
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    #[cfg(unix)]
    cmd.process_group(0);

    let child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e));
    let out = tokio::task::spawn_blocking(move || {
        child
            .wait_with_output()
            .expect("wait for the refusing process")
    })
    .await
    .expect("join the wait task");

    let mut printed = String::from_utf8_lossy(&out.stderr).into_owned();
    printed.push_str(&String::from_utf8_lossy(&out.stdout));
    (out.status, printed)
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
/// Uses `COORDINODE_BIN` env var first (CI / explicit override), then the
/// build next to this test binary, then the release build of the same target
/// directory.
///
/// The test binary lives at `<target>/<profile>/deps/<test>`, so its profile
/// directory holds the server build whatever `CARGO_TARGET_DIR` points at.
///
/// Exposed `pub` so integration tests can directly `Command::new(binary_path())`
/// for non-standard startup scenarios (e.g. checking `--mode=compute` is rejected).
pub fn binary_path() -> PathBuf {
    if let Ok(path) = std::env::var("COORDINODE_BIN") {
        return PathBuf::from(path);
    }

    let name = format!("coordinode{}", std::env::consts::EXE_SUFFIX);
    let exe = std::env::current_exe().expect("test binary path");
    let profile = exe
        .parent()
        .and_then(|deps| deps.parent())
        .expect("test binary sits in <target>/<profile>/deps");

    let bin = profile.join(&name);
    if bin.exists() {
        return bin;
    }
    if let Some(target) = profile.parent() {
        let release = target.join("release").join(&name);
        if release.exists() {
            return release;
        }
    }

    panic!(
        "coordinode binary not found.\n\
         Build it first: cargo build -p coordinode-server\n\
         Or set COORDINODE_BIN=/path/to/coordinode"
    );
}

/// Spawn `coordinode serve --addr [::1]:PORT --ops-addr [::1]:0
/// --rest-addr [::1]:0 --data DATA_DIR`.
///
/// Port 0 lets the OS assign an ephemeral port for the ops and REST HTTP
/// servers: concurrent test servers would otherwise fight over the defaults
/// (:7084, :7081), and a taken port fails the start.
fn spawn_binary(port: u16, data_dir: PathBuf) -> Child {
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.arg("serve")
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--ops-addr")
        .arg("[::1]:0")
        .arg("--rest-addr")
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
        .arg("--rest-addr")
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

/// A free port on `[::1]`, where the spawned server listens.
///
/// Exposed `pub` so multi-node tests can pre-allocate every member's port
/// before spawning (each member's `--peers` needs the others' ports up front).
pub fn free_port() -> u16 {
    coordinode_test_fixtures::alloc_port_on(std::net::Ipv6Addr::LOCALHOST.into())
}

#[cfg(test)]
mod tests;
