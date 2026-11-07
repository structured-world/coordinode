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
    /// A cluster member's credentials, trust bundle and the config naming
    /// them; issued afresh at every start, as a renewal would. Held so the
    /// files outlive the process that reads them.
    identity: Option<tempfile::TempDir>,
    /// The deployment a joining member was configured with, kept for its
    /// restarts.
    deployment: Option<String>,
    /// Settings a cluster member's config carries beside its identity, kept
    /// for its restarts.
    member_settings: String,
    /// Environment variables set on the process.
    env: Vec<(String, String)>,
    /// The command line this process was started with, so a start that is not
    /// ready in time can be run again the same way to diagnose it.
    launch: Launch,
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
            (
                standalone_launch(port, ops_port, rest_port, data, path.as_deref()),
                None,
            )
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
        launch: impl Fn(u16, u16, u16, PathBuf) -> (Launch, Option<tempfile::TempDir>),
    ) -> Self {
        const ATTEMPTS: u32 = 5;
        let mut last_exit = None;
        for _ in 0..ATTEMPTS {
            let (port, ops_port, rest_port) = (free_port(), free_port(), free_port());
            let (launch, identity) =
                launch(port, ops_port, rest_port, data_dir.path().to_path_buf());
            let mut proc = Self {
                child: launch.spawn(),
                port,
                ops_port,
                rest_port,
                data_dir: None,
                config: None,
                identity,
                deployment: None,
                member_settings: String::new(),
                env: Vec::new(),
                launch,
            };
            match proc.ready_or_diagnose(Duration::from_secs(15)).await {
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
        Self::start_member(node_id, port, peer_ports, data_dir, env, "", None).await
    }

    /// [`start_cluster_member`](Self::start_cluster_member) with `settings`,
    /// top-level YAML keys, added to its config file for this start and
    /// every restart.
    pub async fn start_cluster_member_with_settings(
        node_id: u64,
        port: u16,
        peer_ports: &[u16],
        settings: &str,
    ) -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        Self::start_member(node_id, port, peer_ports, data_dir, &[], settings, None).await
    }

    /// [`start_cluster_member`](Self::start_cluster_member) with a
    /// certificate valid until `not_after`; a restart issues a lasting one.
    pub async fn start_cluster_member_until(
        node_id: u64,
        port: u16,
        peer_ports: &[u16],
        not_after: std::time::SystemTime,
    ) -> Self {
        let data_dir = tempfile::TempDir::new().expect("tempdir");
        Self::start_member(
            node_id,
            port,
            peer_ports,
            data_dir,
            &[],
            "",
            Some(not_after),
        )
        .await
    }

    async fn start_member(
        node_id: u64,
        port: u16,
        peer_ports: &[u16],
        data_dir: tempfile::TempDir,
        env: &[(&str, &str)],
        settings: &str,
        valid_until: Option<std::time::SystemTime>,
    ) -> Self {
        let (ops_port, rest_port) = (free_port(), free_port());
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let deployment = if node_id > 1 {
            Some(
                deployment_of(peer_ports, Duration::from_secs(15))
                    .await
                    .unwrap_or_else(|| {
                        panic!("no member among {peer_ports:?} reports its deployment")
                    }),
            )
        } else {
            None
        };
        let (launch, identity) = member_launch(
            Member {
                node_id,
                port,
                peer_ports,
                deployment: deployment.as_deref(),
                settings,
                valid_until,
            },
            ops_port,
            rest_port,
            data_dir.path().to_path_buf(),
            &env,
        );
        let mut proc = Self {
            child: launch.spawn(),
            port,
            ops_port,
            rest_port,
            data_dir: Some(data_dir),
            config: None,
            identity: Some(identity),
            deployment,
            member_settings: settings.to_owned(),
            env,
            launch,
        };
        // The port is fixed by the caller (peers already name it), so a
        // process that lost it cannot move to another one.
        if let Err(status) = proc.ready_or_diagnose(Duration::from_secs(15)).await {
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
        let env: Vec<(String, String)> = env
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect();
        let port = self.port;
        let deployment = self.deployment.take();
        let settings = std::mem::take(&mut self.member_settings);
        let mut last = None;
        for _ in 0..20 {
            let (ops_port, rest_port) = (free_port(), free_port());
            let (launch, identity) = member_launch(
                Member {
                    node_id,
                    port,
                    peer_ports,
                    deployment: deployment.as_deref(),
                    settings: &settings,
                    valid_until: None,
                },
                ops_port,
                rest_port,
                data_dir.path().to_path_buf(),
                &env,
            );
            let mut proc = Self {
                child: launch.spawn(),
                port,
                ops_port,
                rest_port,
                data_dir: None,
                config: None,
                identity: Some(identity),
                deployment: deployment.clone(),
                member_settings: settings.clone(),
                env: env.clone(),
                launch,
            };
            match proc.ready_or_diagnose(Duration::from_secs(30)).await {
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

    /// The value of the unlabelled metric `name` in the member's
    /// `GET /metrics` scrape, `None` while nothing has recorded it.
    pub fn metric(&self, name: &str) -> Option<f64> {
        use std::io::{Read, Write};

        let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, self.ops_port));
        let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .expect("connect to the ops port");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        stream
            .write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .expect("send the request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read the response");
        response.lines().find_map(|line| {
            let (metric, value) = line.split_once(' ')?;
            (metric == name)
                .then(|| value.trim().parse().ok())
                .flatten()
        })
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
        let env = std::mem::take(&mut self.env);
        // The first member keeps the deployment its store recorded when it
        // ran alone; any other joins the one its peers report.
        let deployment = if node_id > 1 {
            deployment_of(peer_ports, Duration::from_secs(15)).await
        } else {
            None
        };
        let settings = std::mem::take(&mut self.member_settings);
        let mut proc = Self::spawn_on_free_port(data_dir, |port, ops_port, rest_port, data| {
            let (launch, identity) = member_launch(
                Member {
                    node_id,
                    port,
                    peer_ports,
                    deployment: deployment.as_deref(),
                    settings: &settings,
                    valid_until: None,
                },
                ops_port,
                rest_port,
                data,
                &env,
            );
            (launch, Some(identity))
        })
        .await;
        proc.env = env;
        proc.deployment = deployment;
        proc.member_settings = settings;
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

    /// The identity this process proves as a cluster member (see
    /// [`member_identity`]).
    pub fn member_identity(&self) -> String {
        member_identity(self.port)
    }

    /// The server's process id.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Stall the disk under the server's Raft log for `pause`: each of its
    /// syncs meanwhile first writes and syncs 64 MiB, so the process waits
    /// on real block I/O; or, for a process started with
    /// `COORDINODE_TEST_DISK_STALL_OWN_MS`, holds each of its own syncs that
    /// long. Needs a server built with `test-faults`.
    #[cfg(unix)]
    pub async fn stall_disk_for(&self, pause: Duration) {
        signal(&self.child, "USR2");
        tokio::time::sleep(pause).await;
        signal(&self.child, "USR2");
    }

    /// Freeze the process for `pause` and let it go on, as a process the
    /// operating system does not schedule: it holds its sockets and files
    /// and runs nothing meanwhile.
    #[cfg(unix)]
    pub async fn freeze_for(&self, pause: Duration) {
        signal(&self.child, "STOP");
        tokio::time::sleep(pause).await;
        signal(&self.child, "CONT");
    }

    /// Replace this member's certificate and key in place with fresh ones for
    /// its own identity, valid until `not_after` when given, as a renewal
    /// does. The running process reads them on its next reload.
    pub fn renew_identity(&self, not_after: Option<std::time::SystemTime>) {
        let _ = PKI.write_credentials_until(
            self.identity_dir(),
            "node",
            &member_identity(self.port),
            not_after,
        );
    }

    /// Replace this member's certificate and key in place with ones proving
    /// the identity of the member on `port`: a set the member must refuse.
    pub fn install_identity_of(&self, port: u16) {
        let _ = PKI.write_credentials(self.identity_dir(), "node", &member_identity(port));
    }

    fn identity_dir(&self) -> &std::path::Path {
        self.identity
            .as_ref()
            .expect("a cluster member has an identity")
            .path()
    }

    /// `POST /reload` on the ops port: the member reads its credentials and
    /// trust again. Returns the status code.
    pub fn reload(&self) -> u16 {
        use std::io::{Read, Write};

        let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, self.ops_port));
        let mut stream = std::net::TcpStream::connect_timeout(&addr, Duration::from_secs(5))
            .expect("connect to the ops port");
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .expect("read timeout");
        stream
            .write_all(
                b"POST /reload HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\
                  Connection: close\r\n\r\n",
            )
            .expect("send the request");
        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .expect("read the response");
        response
            .split(' ')
            .nth(1)
            .and_then(|code| code.parse().ok())
            .unwrap_or_else(|| panic!("no status line in {response:?}"))
    }

    /// Ask the process to stop, as a service manager does, without waiting.
    pub fn request_stop(&self) {
        send_sigterm(&self.child);
    }

    /// Wait up to `within` for the process to exit on its own; whether it did.
    pub async fn exits_within(&mut self, within: Duration) -> bool {
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            if has_exited(&mut self.child) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        has_exited(&mut self.child)
    }

    /// Send the process `SIGHUP`, which makes it reload as `POST /reload`
    /// does; the outcome is only logged.
    #[cfg(unix)]
    pub fn send_sighup(&self) {
        let status = Command::new("kill")
            .args(["-s", "HUP", &self.child.id().to_string()])
            .status()
            .expect("run kill");
        assert!(status.success(), "kill -s HUP exited with {status}");
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
    async fn wait_until_ready(&mut self, timeout: Duration) -> Result<(), StartFailure> {
        let started = Instant::now();
        let deadline = started + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                return Err(StartFailure::Exited(status));
            }
            let probe = probe_ready(self.ops_port);
            if probe == ReadyProbe::Status(200) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(StartFailure::NotReady(NotReady {
                    pid: self.child.id(),
                    waited: started.elapsed(),
                    last: probe,
                }));
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }

    /// [`wait_until_ready`](Self::wait_until_ready), with a start that is not
    /// ready in time diagnosed (see [`diagnose`](Self::diagnose)) and the test
    /// failed with both attempts reported; the exit status of a process that
    /// ended during startup.
    async fn ready_or_diagnose(
        &mut self,
        timeout: Duration,
    ) -> Result<(), std::process::ExitStatus> {
        match self.wait_until_ready(timeout).await {
            Ok(()) => Ok(()),
            Err(StartFailure::Exited(status)) => Err(status),
            Err(StartFailure::NotReady(first)) => {
                let report = self.diagnose(first, DIAGNOSTIC_START_TIMEOUT).await;
                panic!("{report}");
            }
        }
    }

    /// After `first`, a start of this process that was not ready in time:
    /// stop it, run the same command line once more on the same ports and
    /// data directory with the start's phases traced, wait up to `within`,
    /// and stop that one too. The first attempt stands whatever the second
    /// does: the report keeps both.
    async fn diagnose(&mut self, first: NotReady, within: Duration) -> StartReport {
        // Only this process: its ports and files are what the second start
        // needs, and nothing else is touched.
        force_reap(&mut self.child);
        eprintln!(
            "--- coordinode on port {} (pid {}) was not ready within {:?}; its output is above. \
             A diagnostic start of the same command follows. ---",
            self.port, first.pid, first.waited
        );
        let trace = tempfile::NamedTempFile::new().expect("diagnostic trace file");
        let sink = || {
            std::fs::OpenOptions::new()
                .append(true)
                .open(trace.path())
                .expect("open the diagnostic trace file")
        };
        let filter = format!(
            "{},{STARTUP_TRACE}",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into())
        );
        let mut launch = self.launch.clone();
        launch.set_env("RUST_LOG", &filter);
        self.child = launch
            .command()
            .stdout(sink())
            .stderr(sink())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn the diagnostic start: {e}"));
        let started = Instant::now();
        let outcome = match self.wait_until_ready(within).await {
            Ok(()) => DiagnosticOutcome::Ready(started.elapsed()),
            Err(StartFailure::Exited(status)) => {
                DiagnosticOutcome::Exited(status, started.elapsed())
            }
            Err(StartFailure::NotReady(again)) => DiagnosticOutcome::NotReady(again),
        };
        // Stopped gracefully when it can be: the server's log is written
        // through a queue that a kill would cut short.
        send_sigterm(&self.child);
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && !has_exited(&mut self.child) {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        force_reap(&mut self.child);
        let output = std::fs::read_to_string(trace.path())
            .unwrap_or_else(|e| format!("[the diagnostic trace could not be read: {e}]"));
        StartReport {
            port: self.port,
            first,
            filter,
            within,
            outcome,
            output,
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
                    // auto-commit: no interactive transaction
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
    // A joiner is configured with the deployment its peers report, or, with
    // none of them up, with one of its own: the refusal under test comes
    // before any peer is asked.
    let deployment = if node_id > 1 {
        Some(
            deployment_of(peer_ports, Duration::from_secs(2))
                .await
                .unwrap_or_else(|| "00000000000000000000000000000001".to_owned()),
        )
    } else {
        None
    };
    let identity = write_member_identity(&Member {
        node_id,
        port,
        peer_ports,
        deployment: deployment.as_deref(),
        settings: "",
        valid_until: None,
    });
    let bin = binary_path();
    let mut cmd = Command::new(&bin);
    cmd.envs(env.iter().copied());
    cmd.arg("serve")
        .arg("--config")
        .arg(identity.path().join(MEMBER_CONFIG))
        .arg("--node-id")
        .arg(node_id.to_string())
        .arg("--addr")
        .arg(format!("[::1]:{port}"))
        .arg("--advertise-addr")
        .arg(member_addr(port))
        .arg("--peers")
        .arg(peers.join(","))
        .arg("--data")
        .arg(data_dir);
    serve_expecting_refusal(cmd).await
}

/// Run `serve` with `args` (the ops and REST listeners and a quiet log are
/// added) expecting it to refuse to start; returns the exit status and
/// everything it printed.
pub async fn serve_args_expecting_refusal(args: &[&str]) -> (std::process::ExitStatus, String) {
    let mut cmd = Command::new(binary_path());
    cmd.arg("serve").args(args);
    serve_expecting_refusal(cmd).await
}

/// Run `cmd`, a `serve` command line, expecting it to refuse to start.
async fn serve_expecting_refusal(mut cmd: Command) -> (std::process::ExitStatus, String) {
    cmd.arg("--ops-addr")
        .arg("[::1]:0")
        .arg("--rest-addr")
        .arg("[::1]:0")
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
        )
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    own_process_group(&mut cmd);
    let bin = cmd.get_program().to_owned();

    let mut child = cmd
        .spawn()
        .unwrap_or_else(|e| panic!("failed to spawn {bin:?}: {e}"));
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

/// Send `child` the signal `name` through the system `kill` utility.
#[cfg(unix)]
fn signal(child: &Child, name: &str) {
    let sent = Command::new("kill")
        .args(["-s", name, &child.id().to_string()])
        .status()
        .is_ok_and(|status| status.success());
    assert!(sent, "could not send {name} to the server");
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

/// How long the one diagnostic start of a process that was not ready in time
/// may take: long enough to see where a slow start spends its time.
const DIAGNOSTIC_START_TIMEOUT: Duration = Duration::from_secs(60);

/// The log filter a diagnostic start adds: the server's start phases.
const STARTUP_TRACE: &str = "coordinode::startup=debug";

/// Why a start did not become ready.
enum StartFailure {
    /// The process ended during startup.
    Exited(std::process::ExitStatus),
    /// The process kept running and was not ready in time.
    NotReady(NotReady),
}

/// A start that was not ready in time.
#[derive(Debug, Clone, Copy)]
struct NotReady {
    pid: u32,
    waited: Duration,
    /// The last answer `/ready` gave before the wait ended.
    last: ReadyProbe,
}

/// What `GET /ready` on the ops port got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadyProbe {
    /// No connection: the ports were not bound yet.
    Refused,
    /// Connected, no answer: the ports are bound (they are bound before the
    /// store opens) and the ops server is not serving yet, so the start is
    /// still opening the store, consensus or the database.
    NoAnswer,
    /// The ops server answered with this status: 503 while the services are
    /// still being set up, 200 once the node serves.
    Status(u16),
}

impl std::fmt::Display for ReadyProbe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused => f.write_str("connection refused: the ports were not bound yet"),
            Self::NoAnswer => f.write_str(
                "no answer: the ports were bound and the ops server was not serving yet \
                 (opening the store, consensus or the database)",
            ),
            Self::Status(503) => {
                f.write_str("503: store and consensus open, services not serving yet")
            }
            Self::Status(code) => write!(f, "status {code}"),
        }
    }
}

/// How the diagnostic start ended.
enum DiagnosticOutcome {
    Ready(Duration),
    Exited(std::process::ExitStatus, Duration),
    NotReady(NotReady),
}

/// A start that was not ready in time and the diagnostic start that followed.
struct StartReport {
    port: u16,
    first: NotReady,
    /// The log filter the diagnostic start ran with.
    filter: String,
    within: Duration,
    outcome: DiagnosticOutcome,
    /// What the diagnostic start printed.
    output: String,
}

impl std::fmt::Display for StartReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            port,
            first,
            filter,
            within,
            outcome,
            output,
        } = self;
        writeln!(
            f,
            "coordinode on port {port} was not ready within {:?} (pid {}); last /ready: {}.",
            first.waited, first.pid, first.last
        )?;
        write!(
            f,
            "Diagnostic start (the same command, ports and data directory, RUST_LOG={filter}, \
             limit {within:?}): "
        )?;
        match outcome {
            DiagnosticOutcome::Ready(after) => writeln!(f, "ready after {after:?}.")?,
            DiagnosticOutcome::Exited(status, after) => {
                writeln!(f, "exited with {status} after {after:?}.")?;
            }
            DiagnosticOutcome::NotReady(again) => writeln!(
                f,
                "not ready within {:?} either (pid {}); last /ready: {}.",
                again.waited, again.pid, again.last
            )?,
        }
        writeln!(
            f,
            "The first attempt's output is in this test's output above; \
             the diagnostic start printed:"
        )?;
        f.write_str(output)
    }
}

/// What the ops endpoint on `[::1]:port` answers `GET /ready`.
fn probe_ready(port: u16) -> ReadyProbe {
    use std::io::{Read, Write};

    let addr = std::net::SocketAddr::from((std::net::Ipv6Addr::LOCALHOST, port));
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, Duration::from_millis(500))
    else {
        return ReadyProbe::Refused;
    };
    if stream
        .set_read_timeout(Some(Duration::from_secs(1)))
        .is_err()
        || stream
            .write_all(b"GET /ready HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
            .is_err()
    {
        return ReadyProbe::NoAnswer;
    }
    // "HTTP/1.1 200": the code is the status line's last three bytes.
    let mut status_line = [0u8; 12];
    if stream.read_exact(&mut status_line).is_err() {
        return ReadyProbe::NoAnswer;
    }
    std::str::from_utf8(&status_line[9..])
        .ok()
        .and_then(|code| code.parse().ok())
        .map_or(ReadyProbe::NoAnswer, ReadyProbe::Status)
}

/// A `coordinode` command line: program, arguments and the variables set on
/// it, kept so the same start can be run again.
#[derive(Clone)]
struct Launch {
    program: PathBuf,
    args: Vec<std::ffi::OsString>,
    env: Vec<(String, String)>,
}

impl Launch {
    /// `coordinode` with `args`, logging at `RUST_LOG` (errors only unless
    /// the test run sets it).
    fn new(args: Vec<std::ffi::OsString>) -> Self {
        Self {
            program: binary_path(),
            args,
            env: vec![(
                "RUST_LOG".to_string(),
                std::env::var("RUST_LOG").unwrap_or_else(|_| "error".into()),
            )],
        }
    }

    /// Set `key` to `value`, replacing an earlier value.
    fn set_env(&mut self, key: &str, value: &str) {
        self.env.retain(|(k, _)| k != key);
        self.env.push((key.to_string(), value.to_string()));
    }

    /// The command, in a process group of its own.
    ///
    /// The group keeps a Ctrl-C / SIGINT delivered to the test runner's group
    /// from reaching the server mid-flush; the harness owns the child's
    /// lifetime by PID. It does NOT hide the child from nextest's leak
    /// detection, which follows the inherited stdout/stderr pipes: only
    /// reaping the child before the test exits prevents a LEAKY verdict,
    /// which the Drop impl guarantees.
    fn command(&self) -> Command {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .envs(self.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        own_process_group(&mut cmd);
        cmd
    }

    /// Start it, its output going to the test's.
    fn spawn(&self) -> Child {
        self.command()
            .spawn()
            .unwrap_or_else(|e| panic!("failed to spawn {}: {e}", self.program.display()))
    }
}

/// `coordinode serve [--config FILE] --addr [::1]:PORT --ops-addr
/// [::1]:OPS_PORT --rest-addr [::1]:REST_PORT --data DATA_DIR`.
///
/// Every port is chosen by the harness, so a test can ask `/ready` and reach
/// the REST surface; concurrent test servers would otherwise fight over the
/// defaults (:7084, :7081), and a taken port fails the start.
fn standalone_launch(
    port: u16,
    ops_port: u16,
    rest_port: u16,
    data_dir: PathBuf,
    config: Option<&std::path::Path>,
) -> Launch {
    let mut args: Vec<std::ffi::OsString> = vec!["serve".into()];
    if let Some(config) = config {
        args.extend(["--config".into(), config.into()]);
    }
    args.extend([
        "--addr".into(),
        format!("[::1]:{port}").into(),
        "--ops-addr".into(),
        format!("[::1]:{ops_port}").into(),
        "--rest-addr".into(),
        format!("[::1]:{rest_port}").into(),
        "--data".into(),
        data_dir.into(),
    ]);
    Launch::new(args)
}

/// A member's address in the form the operator guide gives for
/// `--advertise-addr`, `--peers` and `admin node join --addr`: `host:port`,
/// no scheme. Every cluster test goes through this form.
pub fn member_addr(port: u16) -> String {
    format!("[::1]:{port}")
}

/// The CA that issues every cluster member of this test process its identity.
static PKI: std::sync::LazyLock<coordinode_test_fixtures::TestPki> =
    std::sync::LazyLock::new(coordinode_test_fixtures::TestPki::new);

/// The config file in a member's identity directory.
const MEMBER_CONFIG: &str = "member.conf";

/// The deployment a member among `peer_ports` reports in its cluster
/// status, as an operator reads it to configure a joining node; `None` when
/// none reports one within `within`.
async fn deployment_of(peer_ports: &[u16], within: Duration) -> Option<String> {
    use crate::proto::admin::GetClusterStatusRequest;

    let deadline = Instant::now() + within;
    loop {
        for &port in peer_ports {
            let Ok(endpoint) =
                tonic::transport::Endpoint::from_shared(format!("http://[::1]:{port}"))
            else {
                continue;
            };
            let channel = endpoint
                .connect_timeout(Duration::from_millis(500))
                .timeout(Duration::from_secs(2))
                .connect_lazy();
            let status = ClusterServiceClient::new(channel)
                .get_cluster_status(GetClusterStatusRequest {})
                .await;
            if let Ok(status) = status {
                let deployment = status.into_inner().deployment;
                if !deployment.is_empty() {
                    return Some(deployment);
                }
            }
        }
        if Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// The identity the member listening on `port` proves: its SPIFFE ID. Tied
/// to the port, so a member's peers know who must answer at each address.
pub fn member_identity(port: u16) -> String {
    format!("spiffe://{}/port/{port}", PKI.trust_domain())
}

/// A cluster member as its command line and config file name it.
struct Member<'a> {
    node_id: u64,
    port: u16,
    /// The gRPC ports of the other members.
    peer_ports: &'a [u16],
    /// The deployment a joining member joins.
    deployment: Option<&'a str>,
    /// Top-level YAML keys added to the config file.
    settings: &'a str,
    /// When its certificate expires, if not the CA's default.
    valid_until: Option<std::time::SystemTime>,
}

/// Issue `member` fresh credentials and write them, the trust bundle and a
/// config naming them into a new directory; a joining member joins its
/// deployment and learns its group's membership from its peers.
fn write_member_identity(member: &Member<'_>) -> tempfile::TempDir {
    let Member {
        node_id,
        port,
        peer_ports,
        deployment,
        settings,
        valid_until,
    } = *member;
    let dir = tempfile::TempDir::new().expect("identity tempdir");
    let (cert, key) =
        PKI.write_credentials_until(dir.path(), "node", &member_identity(port), valid_until);
    let bundle = PKI.write_bundle(dir.path());
    let quote = |path: &std::path::Path| {
        serde_json::to_string(&path.display().to_string()).expect("a string")
    };
    let mut yaml = format!(
        "node_identity:\n  cert: {}\n  key: {}\n  trust_bundles:\n    {}: {}\n",
        quote(&cert),
        quote(&key),
        PKI.trust_domain(),
        quote(&bundle),
    );
    if let Some(deployment) = deployment {
        yaml.push_str(&format!("  deployment: \"{deployment}\"\n"));
    }
    if node_id > 1 {
        yaml.push_str("  enroll_from:\n");
        for &peer in peer_ports {
            yaml.push_str(&format!(
                "    - addr: \"{}\"\n      identity: \"{}\"\n",
                member_addr(peer),
                member_identity(peer)
            ));
        }
    }
    yaml.push_str(settings);
    std::fs::write(dir.path().join(MEMBER_CONFIG), yaml).expect("write the member config");
    dir
}

/// `member` with explicit `--node-id`, `--advertise-addr`, and `--peers`,
/// `env` set, and its identity written for it. See
/// [`CoordinodeProcess::start_cluster_member`].
fn member_launch(
    member: Member<'_>,
    ops_port: u16,
    rest_port: u16,
    data_dir: PathBuf,
    env: &[(String, String)],
) -> (Launch, tempfile::TempDir) {
    let (node_id, port) = (member.node_id, member.port);
    let peers: Vec<String> = member.peer_ports.iter().map(|&p| member_addr(p)).collect();
    let identity = write_member_identity(&member);
    let mut launch = Launch::new(vec![
        "serve".into(),
        "--config".into(),
        identity.path().join(MEMBER_CONFIG).into(),
        "--node-id".into(),
        node_id.to_string().into(),
        "--addr".into(),
        format!("[::1]:{port}").into(),
        "--advertise-addr".into(),
        member_addr(port).into(),
        "--peers".into(),
        peers.join(",").into(),
        "--ops-addr".into(),
        format!("[::1]:{ops_port}").into(),
        "--rest-addr".into(),
        format!("[::1]:{rest_port}").into(),
        "--data".into(),
        data_dir.into(),
    ]);
    for (key, value) in env {
        launch.set_env(key, value);
    }
    (launch, identity)
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
