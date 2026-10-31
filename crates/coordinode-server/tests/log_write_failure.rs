//! A server whose Raft log write fails, as on a full disk, goes out of
//! service instead of serving on: the write is not acknowledged, `/ready`
//! stops answering 200, and the process exits with an error. Started again
//! on the same store, it serves every write acknowledged before the failure.
//!
//! Drives the real binary; its test fault hook (SIGUSR1) makes the log
//! refuse writes.

#![cfg(unix)]
#![allow(clippy::expect_used, clippy::panic)]

use std::io::{BufRead as _, BufReader, Read as _, Write as _};
use std::net::TcpStream;
use std::path::Path;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use coordinode_server::proto::query::{
    ExecuteCypherRequest, cypher_service_client::CypherServiceClient,
};

/// A running server and the lines it logs, killed when dropped.
struct Server {
    child: Child,
    lines: mpsc::Receiver<String>,
    grpc: u16,
    ops: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(data: &Path) -> Self {
        let [grpc, rest, ops, pg] = coordinode_test_fixtures::alloc_ports::<4>();
        let mut child = Command::new(env!("CARGO_BIN_EXE_coordinode"))
            .args(["serve", "--data"])
            .arg(data)
            .args(["--addr", &format!("127.0.0.1:{grpc}")])
            .args(["--rest-addr", &format!("127.0.0.1:{rest}")])
            .args(["--ops-addr", &format!("127.0.0.1:{ops}")])
            .args(["--pg-addr", &format!("127.0.0.1:{pg}")])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start coordinode serve");
        let stdout = child.stdout.take().expect("piped stdout");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                eprintln!("{line}");
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let server = Server {
            child,
            lines,
            grpc,
            ops,
        };
        let deadline = Instant::now() + Duration::from_secs(60);
        while ready(ops) != Some(true) {
            assert!(Instant::now() < deadline, "the server did not become ready");
            std::thread::sleep(Duration::from_millis(100));
        }
        server
    }

    /// Wait until the server logs a line containing `text`.
    fn logged(&self, text: &str) {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            match self.lines.recv_timeout(left) {
                Ok(line) if line.contains(text) => return,
                Ok(_) => {}
                Err(e) => panic!("the server did not log {text:?}: {e}"),
            }
        }
    }

    /// Wait for the process to end: its output closes when it exits.
    fn exited(mut self) -> ExitStatus {
        while self.lines.recv().is_ok() {}
        self.child.wait().expect("reap the server")
    }

    fn cypher(&self, query: &str) -> Result<usize, String> {
        let grpc = self.grpc;
        let query = query.to_string();
        tokio::runtime::Runtime::new()
            .expect("runtime")
            .block_on(async move {
                let mut client = CypherServiceClient::connect(format!("http://127.0.0.1:{grpc}"))
                    .await
                    .map_err(|e| e.to_string())?;
                let call = client.execute_cypher(ExecuteCypherRequest {
                    query,
                    ..Default::default()
                });
                match tokio::time::timeout(Duration::from_secs(30), call).await {
                    Ok(Ok(response)) => Ok(response.into_inner().rows.len()),
                    Ok(Err(status)) => Err(status.to_string()),
                    Err(_) => Err("no answer within 30s".to_string()),
                }
            })
    }
}

/// `Some(true)` when `/ready` answers 200, `Some(false)` for any other
/// answer, `None` when nothing answers.
fn ready(port: u16) -> Option<bool> {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).ok()?;
    stream.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    stream
        .write_all(b"GET /ready HTTP/1.0\r\nHost: localhost\r\n\r\n")
        .ok()?;
    let mut reply = String::new();
    let _ = stream.read_to_string(&mut reply);
    if reply.is_empty() {
        return None;
    }
    Some(reply.starts_with("HTTP/1.0 200") || reply.starts_with("HTTP/1.1 200"))
}

#[test]
fn a_failed_log_write_takes_the_server_out_of_service() {
    let data = tempfile::tempdir().expect("data dir");
    let server = Server::start(data.path());
    server
        .cypher("CREATE (:Kept {v: 1})")
        .expect("a write before the failure commits");

    let pid = server.child.id().to_string();
    let signalled = Command::new("kill")
        .args(["-USR1", &pid])
        .status()
        .expect("signal the server");
    assert!(signalled.success(), "the fault signal is delivered");
    server.logged("test fault: Raft log writes now fail");

    let refused = server.cypher("CREATE (:Lost {v: 2})");
    assert!(
        refused.is_err(),
        "a write the log could not hold is not acknowledged: {refused:?}"
    );
    server.logged("consensus stopped on a fatal error");
    assert_ne!(
        ready(server.ops),
        Some(true),
        "a node whose consensus stopped is not reported ready"
    );
    let status = server.exited();
    assert!(
        !status.success(),
        "the process exits with an error after a fatal consensus stop: {status}"
    );

    let server = Server::start(data.path());
    assert_eq!(
        server.cypher("MATCH (n:Kept) RETURN n.v"),
        Ok(1),
        "the write acknowledged before the failure is served after the restart"
    );
    server
        .cypher("CREATE (:Again {v: 3})")
        .expect("the restarted server commits");
}
