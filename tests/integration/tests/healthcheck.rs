//! `coordinode healthcheck` against a real server: the exit code a container
//! runtime reads while the server starts, serves, drains and stops, and the
//! address it probes when it is given a config file.
//!
//! ## Running
//!
//! ```bash
//! cargo build -p coordinode-server
//! cargo nextest run -p coordinode-integration --test healthcheck
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use coordinode_integration::harness::{
    binary_path, free_port, own_process_group, request_shutdown,
};
use coordinode_integration::proto::session::session_service_client::SessionServiceClient;
use coordinode_integration::proto::session::{ClientFrame, Configure, client_frame};

/// A server on explicit loopback ports, killed if the test leaves it running.
struct Server {
    child: Child,
    grpc_port: u16,
    ops_port: u16,
    _data: tempfile::TempDir,
}

impl Server {
    /// Start `serve`, with `extra` arguments ahead of the listener flags.
    fn start(extra: &[&str]) -> Self {
        let data = tempfile::tempdir().expect("tempdir");
        let (grpc_port, ops_port) = (free_port(), free_port());
        let mut cmd = Command::new(binary_path());
        cmd.arg("serve")
            .args(extra)
            .arg("--addr")
            .arg(format!("[::1]:{grpc_port}"))
            .arg("--ops-addr")
            .arg(format!("[::1]:{ops_port}"))
            .arg("--rest-addr")
            .arg("[::1]:0")
            .arg("--data")
            .arg(data.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        own_process_group(&mut cmd);
        let child = cmd.spawn().expect("spawn coordinode serve");
        Self {
            child,
            grpc_port,
            ops_port,
            _data: data,
        }
    }

    fn ops_addr(&self) -> String {
        format!("[::1]:{}", self.ops_port)
    }

    fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Ask the server to stop the way a service manager does.
    fn sigterm(&self) {
        request_shutdown(&self.child).expect("ask the server to stop");
    }

    /// Wait up to `limit` for the process to exit.
    fn exited_within(&mut self, limit: Duration) -> bool {
        let deadline = Instant::now() + limit;
        while Instant::now() < deadline {
            if !self.running() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        false
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Run `coordinode healthcheck <args>`: whether it succeeded, and its stderr.
fn healthcheck(args: &[&str]) -> (bool, String) {
    let output = Command::new(binary_path())
        .arg("healthcheck")
        .args(args)
        .args(["--timeout-ms", "1000"])
        .output()
        .expect("run coordinode healthcheck");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

/// Poll `healthcheck` until it passes, for up to 30 s.
fn wait_healthy(args: &[&str]) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let (ok, stderr) = healthcheck(args);
        if ok {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "the server never became healthy: {stderr}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// A running server passes; once it has stopped, the check fails because
/// nothing answers, so a runtime restarts or replaces the container.
#[test]
fn a_running_server_passes_and_a_stopped_one_fails() {
    let mut server = Server::start(&[]);
    let ops = server.ops_addr();
    wait_healthy(&["--ops-addr", &ops]);

    server.sigterm();
    assert!(
        server.exited_within(Duration::from_secs(30)),
        "the server did not stop on SIGTERM"
    );
    let (ok, stderr) = healthcheck(&["--ops-addr", &ops]);
    assert!(!ok, "a stopped server must fail the check");
    assert!(stderr.contains("connect"), "got: {stderr}");
}

/// Given the config file the server runs with, the check probes that file's
/// `ops_addr`, and `--ops-addr` on the command line wins over it.
#[test]
fn the_check_reads_the_servers_config_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = dir.path().join("coordinode.conf");
    let ops_port = free_port();
    std::fs::write(&config, format!("ops_addr: \"[::1]:{ops_port}\"\n")).expect("write config");
    let config = config.to_str().expect("utf-8 path").to_owned();

    // The server takes its ops address from the same file: no --ops-addr.
    let data = tempfile::tempdir().expect("tempdir");
    let grpc_port = free_port();
    let child = Command::new(binary_path())
        .args(["serve", "--config", &config])
        .arg("--addr")
        .arg(format!("[::1]:{grpc_port}"))
        .args(["--rest-addr", "[::1]:0", "--data"])
        .arg(data.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn coordinode serve");
    let mut server = Server {
        child,
        grpc_port,
        ops_port,
        _data: data,
    };
    assert!(server.running(), "the server started");

    wait_healthy(&["--config", &config]);

    // The command line overrides the file: a free port nothing listens on.
    let elsewhere = format!("[::1]:{}", free_port());
    let (ok, stderr) = healthcheck(&["--config", &config, "--ops-addr", &elsewhere]);
    assert!(!ok, "--ops-addr must override the config file");
    assert!(stderr.contains(&elsewhere), "got: {stderr}");
}

/// Between the shutdown signal and the end of the drain the server is alive
/// but no longer ready: the check fails with the server's own 503, so a
/// balancer stops sending it work while its open streams finish.
#[tokio::test(flavor = "multi_thread")]
async fn a_draining_server_is_alive_but_not_ready() {
    let mut server = Server::start(&[]);
    let ops = server.ops_addr();
    let wait = ops.clone();
    tokio::task::spawn_blocking(move || wait_healthy(&["--ops-addr", &wait]))
        .await
        .expect("wait for readiness");

    // An open session stream holds the drain open until the client ends it.
    let channel =
        tonic::transport::Endpoint::from_shared(format!("http://[::1]:{}", server.grpc_port))
            .expect("endpoint")
            .connect()
            .await
            .expect("connect");
    let (tx, rx) = tokio::sync::mpsc::channel::<ClientFrame>(4);
    tx.send(ClientFrame {
        request_id: 1,
        op: Some(client_frame::Op::Configure(Configure::default())),
    })
    .await
    .expect("send the first frame");
    let mut inbound = SessionServiceClient::new(channel)
        .session(tokio_stream::wrappers::ReceiverStream::new(rx))
        .await
        .expect("open session")
        .into_inner();
    inbound
        .message()
        .await
        .expect("a frame")
        .expect("the stream is open");

    server.sigterm();
    let probe = ops.clone();
    let (ok, stderr) = tokio::task::spawn_blocking(move || {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let (ok, stderr) = healthcheck(&["--ops-addr", &probe]);
            if !ok || Instant::now() >= deadline {
                return (ok, stderr);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })
    .await
    .expect("probe while draining");
    assert!(!ok, "a draining server must fail the check");
    assert!(
        stderr.contains("503"),
        "the draining server itself must answer 503, got: {stderr}"
    );
    assert!(server.running(), "the server is still draining");

    // Ending the stream lets the drain finish.
    drop(tx);
    drop(inbound);
    let exited = tokio::task::spawn_blocking(move || server.exited_within(Duration::from_secs(30)))
        .await
        .expect("wait for exit");
    assert!(exited, "the server did not finish draining");
}
