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

use crate::proto::{
    admin::cluster_service_client::ClusterServiceClient,
    query::cypher_service_client::CypherServiceClient,
    v2::graph::schema_service_client::SchemaServiceClient,
};

/// A running coordinode process bound to an ephemeral port + temp data dir.
///
/// Killed (and data dir removed) when dropped. Use [`CoordinodeProcess::restart`]
/// to kill → respawn while keeping the same data directory (restart regression tests).
pub struct CoordinodeProcess {
    child: Child,
    pub port: u16,
    /// Ops port, whose `/ready` says the server is serving: the listeners
    /// are bound before storage opens, so an open port proves nothing yet.
    ops_port: u16,
    /// REST/JSON port, which transcodes to the gRPC services.
    pub rest_port: u16,
    // Wrapped in Option so `restart()` can take it without needing unsafe.
    // Always `Some` except briefly during `restart()`.
    data_dir: Option<tempfile::TempDir>,
    /// The `--config` file the process runs with, carried over every restart.
    config: Option<tempfile::NamedTempFile>,
    /// Environment variables set on the process.
    env: Vec<(String, String)>,
}

impl CoordinodeProcess {
    /// Spawn `coordinode serve` against `data_dir` on a free ephemeral port.
    ///
    /// Waits up to 15 seconds for the gRPC port to become available.
    pub async fn start() -> Self {
        Self::start_configured(None).await
    }

    /// [`start`](Self::start) with `yaml` as the `--config` file; every
    /// restart keeps it.
    pub async fn start_with_config(yaml: &str) -> Self {
        use std::io::Write as _;
        let mut file = tempfile::Builder::new()
            .suffix(".conf")
            .tempfile()
            .expect("config tempfile");
        file.write_all(yaml.as_bytes()).expect("write the config");
        Self::start_configured(Some(file)).await
    }

    async fn start_configured(config: Option<tempfile::NamedTempFile>) -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        let proc = Self::spawn_standalone(data_dir, config).await;
        // A standalone server serves before it has elected itself; every
        // caller of `start` expects a node that takes writes.
        proc.wait_for_leader(Duration::from_secs(15)).await;
        proc
    }

    /// Spawn a standalone server over `data_dir`, with `config` if given.
    async fn spawn_standalone(
        data_dir: tempfile::TempDir,
        config: Option<tempfile::NamedTempFile>,
    ) -> Self {
        let path = config.as_ref().map(|f| f.path().to_path_buf());
        let mut proc = Self::spawn_on_free_port(data_dir, |port, ops_port, rest_port, data| {
            spawn_binary(port, ops_port, rest_port, data, path.as_deref())
        })
        .await;
        proc.config = config;
        proc
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
        spawn: impl Fn(u16, u16, u16, PathBuf) -> Child,
    ) -> Self {
        const ATTEMPTS: u32 = 5;
        let mut last_exit = None;
        for _ in 0..ATTEMPTS {
            let (port, ops_port, rest_port) = (free_port(), free_port(), free_port());
            let mut proc = Self {
                child: spawn(port, ops_port, rest_port, data_dir.path().to_path_buf()),
                port,
                ops_port,
                rest_port,
                data_dir: None,
                config: None,
                env: Vec::new(),
            };
            match proc.wait_until_ready(Duration::from_secs(15)).await {
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
    /// --advertise-addr [::1]:port --peers <peer advertise addrs> ...`.
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
        Self::start_cluster_member_with_env(node_id, port, peer_ports, &[]).await
    }

    /// [`start_cluster_member`](Self::start_cluster_member) with `env` set on
    /// the process and on every restart of it.
    pub async fn start_cluster_member_with_env(
        node_id: u64,
        port: u16,
        peer_ports: &[u16],
        env: &[(&str, &str)],
    ) -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        Self::start_cluster_member_over(node_id, port, peer_ports, data_dir, env).await
    }

    /// Start cluster member `node_id` over `data_dir`, which may already hold
    /// a directory, with `env` set.
    pub async fn start_cluster_member_over(
        node_id: u64,
        port: u16,
        peer_ports: &[u16],
        data_dir: tempfile::TempDir,
        env: &[(&str, &str)],
    ) -> Self {
        let peers: Vec<String> = peer_ports.iter().map(|&p| member_addr(p)).collect();
        let (ops_port, rest_port) = (free_port(), free_port());
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let child = spawn_cluster_binary(
            node_id,
            port,
            ops_port,
            rest_port,
            &peers,
            data_dir.path().to_path_buf(),
            &env,
        );
        let mut proc = Self {
            child,
            port,
            ops_port,
            rest_port,
            data_dir: Some(data_dir),
            config: None,
            env,
        };
        // The port is fixed by the caller (peers already name it), so a
        // process that lost it cannot move to another one.
        if let Err(status) = proc.wait_until_ready(Duration::from_secs(15)).await {
            panic!("cluster member {node_id} exited during startup on port {port}: {status}");
        }
        proc
    }

    /// Stop cluster member `node_id` (gracefully, or with SIGKILL when
    /// `kill`) and start it again over the same directory, on the same port
    /// its group records, with `env` replacing the variables it ran with.
    /// Retries while the port is still held by the stopped process.
    pub async fn restart_member(
        mut self,
        node_id: u64,
        peer_ports: &[u16],
        env: &[(&str, &str)],
        kill: bool,
    ) -> Self {
        if kill {
            force_reap(&mut self.child);
        } else {
            send_sigterm(&self.child);
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline && !has_exited(&mut self.child) {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
            force_reap(&mut self.child);
        }
        let data_dir = self
            .data_dir
            .take()
            .expect("data_dir missing: restart called twice?");
        let peers: Vec<String> = peer_ports.iter().map(|&p| member_addr(p)).collect();
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let port = self.port;
        let mut last = None;
        for _ in 0..20 {
            let (ops_port, rest_port) = (free_port(), free_port());
            let mut proc = Self {
                child: spawn_cluster_binary(
                    node_id,
                    port,
                    ops_port,
                    rest_port,
                    &peers,
                    data_dir.path().to_path_buf(),
                    &env,
                ),
                port,
                ops_port,
                rest_port,
                data_dir: None,
                config: None,
                env: env.clone(),
            };
            match proc.wait_until_ready(Duration::from_secs(30)).await {
                Ok(()) => {
                    proc.data_dir = Some(data_dir);
                    return proc;
                }
                Err(status) => {
                    last = Some(status);
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
            }
        }
        panic!("member {node_id} did not start again on port {port}: {last:?}");
    }

    /// The member's version report, from `GET /version` on its ops port.
    pub fn version_report(&self) -> serde_json::Value {
        use std::io::{Read, Write};

        let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, self.ops_port));
        let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .expect("connect to the ops port");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        stream
            .write_all(b"GET /version HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .expect("send the request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read the response");
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body)
            .unwrap_or_else(|| panic!("no body in {response:?}"));
        serde_json::from_str(body).unwrap_or_else(|e| panic!("version report {body:?}: {e}"))
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
        let proc = Self::spawn_standalone(data_dir, self.config.take()).await;
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
        Self::spawn_standalone(data_dir, self.config.take()).await
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
        let peers: Vec<String> = peer_ports.iter().map(|&p| member_addr(p)).collect();
        let env = std::mem::take(&mut self.env);
        let mut proc = Self::spawn_on_free_port(data_dir, |port, ops_port, rest_port, data| {
            spawn_cluster_binary(node_id, port, ops_port, rest_port, &peers, data, &env)
        })
        .await;
        proc.env = env;
        proc
    }

    /// gRPC endpoint URL for use with tonic.
    pub fn endpoint(&self) -> String {
        format!("http://[::1]:{}", self.port)
    }

    /// This process's address as a cluster member (see [`member_addr`]).
    pub fn member_addr(&self) -> String {
        member_addr(self.port)
    }

    /// Send one HTTP/1.1 request with an optional JSON body to this process's
    /// REST port; returns the status code and the raw response body.
    pub fn rest_request(&self, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
        use std::io::{Read, Write};

        let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, self.rest_port));
        let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .expect("connect to the REST port");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        let body = body.unwrap_or("");
        let request = format!(
            "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream
            .write_all(request.as_bytes())
            .expect("send the request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read the response");
        let status = response
            .split(' ')
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("no status line in {response:?}"));
        let body = response
            .split_once("\r\n\r\n")
            .map(|(_, body)| body.to_string())
            .unwrap_or_default();
        (status, body)
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

    /// Block (async) until the server answers `/ready` with 200 on its ops
    /// port, or return the exit status if the process ended first.
    ///
    /// The listeners are bound before storage opens (a taken port fails the
    /// start right away), so an open gRPC port only says the process is
    /// starting. `/ready` turns 200 once it serves: storage, consensus and
    /// the query engine are open. A server that lost a port to another holder
    /// exits at its bind, which the exit check catches first.
    async fn wait_until_ready(
        &mut self,
        timeout: Duration,
    ) -> Result<(), std::process::ExitStatus> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(status);
            }
            if answers_ready(self.ops_port) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                panic!(
                    "coordinode on port {} was not ready within {:?}",
                    self.port, timeout
                );
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
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
                    read_preference: 0, // the server default
                    read_concern: None,
                    write_concern: None,
                    transaction_id: 0, // auto-commit (no interactive transaction)
                    ..Default::default()
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

    /// Kill the process outright (no flush, no shutdown path) and hand the
    /// data directory over, as a crash would leave it.
    pub fn kill_keeping_data(mut self) -> tempfile::TempDir {
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
    start_cluster_member_expecting_refusal_with_env(node_id, port, peer_ports, data_dir, &[]).await
}

/// [`start_cluster_member_expecting_refusal`] with `env` set on the process.
pub async fn start_cluster_member_expecting_refusal_with_env(
    node_id: u64,
    port: u16,
    peer_ports: &[u16],
    data_dir: &std::path::Path,
    env: &[(&str, &str)],
) -> (std::process::ExitStatus, String) {
    let peers: Vec<String> = peer_ports.iter().map(|&p| member_addr(p)).collect();
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.envs(env.iter().copied());
    cmd.arg("serve")
        .arg("--node-id")
        .arg(node_id.to_string())
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--advertise-addr")
        .arg(member_addr(port))
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
    own_process_group(&mut cmd);

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e));
    let stderr = drain(child.stderr.take());
    let stdout = drain(child.stdout.take());

    // A process that does not refuse keeps serving; bound the wait so that
    // failure reports what the process printed instead of hanging the run.
    let deadline = Instant::now() + Duration::from_secs(60);
    let status = loop {
        if let Ok(Some(status)) = child.try_wait() {
            break Some(status);
        }
        if Instant::now() >= deadline {
            force_reap(&mut child);
            break None;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };

    let mut printed = stderr.join().expect("join the stderr reader");
    printed.push_str(&stdout.join().expect("join the stdout reader"));
    match status {
        Some(status) => (status, printed),
        None => panic!("the process never exited, so it did not refuse; it printed: {printed}"),
    }
}

/// Read a child's pipe to the end on its own thread, so a full pipe never
/// blocks the child while the caller waits for it to exit.
fn drain(pipe: Option<impl std::io::Read + Send + 'static>) -> std::thread::JoinHandle<String> {
    std::thread::spawn(move || {
        let mut bytes = Vec::new();
        if let Some(mut pipe) = pipe {
            // A read error ends the capture early; what was read still counts,
            // and the error is part of what the test reports.
            if let Err(e) = pipe.read_to_end(&mut bytes) {
                bytes.extend_from_slice(format!("\n[pipe read failed: {e}]").as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    })
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

/// Ask `child` to stop, the way a service manager does. A request that could
/// not be sent is reported; the callers go on to wait and then force-reap.
fn send_sigterm(child: &Child) {
    if let Err(e) = request_shutdown(child) {
        eprintln!("could not ask the server to stop: {e}");
    }
}

/// Ask `child` to shut down gracefully: SIGTERM on Unix (what `docker stop`
/// and systemd send), CTRL_BREAK on Windows (what reaches a console process
/// in its own process group; Ctrl+C is disabled for such a group). The child
/// must have been started through [`own_process_group`].
///
/// # Errors
///
/// The request could not be delivered.
pub fn request_shutdown(child: &Child) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        // The system `kill` utility: no `nix` crate, no unsafe.
        let status = Command::new("kill")
            .args(["-s", "TERM", &child.id().to_string()])
            .status()?;
        if status.success() {
            Ok(())
        } else {
            Err(std::io::Error::other(format!("kill exited with {status}")))
        }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, GenerateConsoleCtrlEvent};
        // SAFETY: an FFI call taking two integers and no pointers. The group
        // id is the child's process id, which names the process group
        // `own_process_group` started it in; the event reaches that group
        // only, never this process.
        #[allow(unsafe_code)]
        let delivered = unsafe { GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, child.id()) };
        if delivered != 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

/// Start the child in a process group of its own, so a shutdown request (and
/// a signal the test runner receives) reaches the child alone.
pub fn own_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt as _;
        cmd.creation_flags(windows_sys::Win32::System::Threading::CREATE_NEW_PROCESS_GROUP);
    }
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

/// Whether the ops endpoint on `[::1]:port` answers `GET /ready` with 200.
fn answers_ready(port: u16) -> bool {
    use std::io::{Read, Write};

    let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port));
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(500))
    else {
        return false;
    };
    if stream
        .set_read_timeout(Some(Duration::from_secs(1)))
        .is_err()
        || stream
            .write_all(b"GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .is_err()
    {
        return false;
    }
    let mut status_line = [0u8; 12];
    stream.read_exact(&mut status_line).is_ok() && status_line.ends_with(b" 200")
}

/// Spawn `coordinode serve [--config FILE] --addr [::1]:PORT --ops-addr
/// [::1]:OPS_PORT --rest-addr [::1]:REST_PORT --data DATA_DIR`.
///
/// Every port is chosen by the harness, so a test can ask `/ready` and reach
/// the REST surface; concurrent test servers would otherwise fight over the
/// defaults (:7084, :7081), and a taken port fails the start.
fn spawn_binary(
    port: u16,
    ops_port: u16,
    rest_port: u16,
    data_dir: PathBuf,
    config: Option<&std::path::Path>,
) -> Child {
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.arg("serve");
    if let Some(config) = config {
        cmd.arg("--config").arg(config);
    }
    cmd.arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--ops-addr")
        .arg(format!("[::1]:{ops_port}"))
        .arg("--rest-addr")
        .arg(format!("[::1]:{rest_port}"))
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
    own_process_group(&mut cmd);

    cmd.spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {}: {}", bin.display(), e))
}

/// A member's address in the form the operator guide gives for
/// `--advertise-addr`, `--peers` and `admin node join --addr`: `host:port`,
/// no scheme. Every cluster test goes through this form.
pub fn member_addr(port: u16) -> String {
    format!("[::1]:{port}")
}

/// Spawn a cluster member with explicit `--node-id`, `--advertise-addr`, and
/// `--peers`. See [`CoordinodeProcess::start_cluster_member`].
fn spawn_cluster_binary(
    node_id: u64,
    port: u16,
    ops_port: u16,
    rest_port: u16,
    peers: &[String],
    data_dir: PathBuf,
    env: &[(String, String)],
) -> Child {
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    cmd.arg("serve")
        .arg("--node-id")
        .arg(node_id.to_string())
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--advertise-addr")
        .arg(member_addr(port))
        .arg("--peers")
        .arg(peers.join(","))
        .arg("--ops-addr")
        .arg(format!("[::1]:{ops_port}"))
        .arg("--rest-addr")
        .arg(format!("[::1]:{rest_port}"))
        .arg("--data")
        .arg(&data_dir)
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
        );
    own_process_group(&mut cmd);

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
