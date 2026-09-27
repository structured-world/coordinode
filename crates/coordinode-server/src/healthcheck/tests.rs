use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Duration;

use super::{ops_addr, probe_ready, probe_targets};

const TIMEOUT: Duration = Duration::from_millis(500);

/// A one-shot server on a free port that reads the request and answers with
/// `response`; returns its address and the request it received.
fn answer_once(response: &'static str) -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = [0u8; 512];
        let n = stream.read(&mut buf).expect("read request");
        stream
            .write_all(response.as_bytes())
            .expect("write response");
        String::from_utf8_lossy(&buf[..n]).into_owned()
    });
    (addr, handle)
}

#[test]
fn a_ready_server_passes_and_is_asked_for_ready() {
    let (addr, server) =
        answer_once("HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}");
    assert_eq!(probe_ready(&addr, TIMEOUT), Ok(()));
    let request = server.join().expect("server thread");
    assert!(
        request.starts_with("GET /ready HTTP/1.1\r\n"),
        "got: {request:?}"
    );
}

#[test]
fn a_server_that_is_not_ready_fails_with_its_status() {
    let (addr, server) = answer_once(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 15\r\n\r\n{\"ready\":false}",
    );
    let err = probe_ready(&addr, TIMEOUT).expect_err("503 is not ready");
    assert!(err.contains("503"), "got: {err}");
    server.join().expect("server thread");
}

#[test]
fn an_answer_without_an_http_status_fails() {
    let (addr, server) = answer_once("hello\r\n");
    let err = probe_ready(&addr, TIMEOUT).expect_err("no status line");
    assert!(err.contains("no HTTP status"), "got: {err}");
    server.join().expect("server thread");
}

#[test]
fn nothing_listening_fails() {
    // Bound and dropped: the port is free and refuses connections.
    let addr = TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("addr")
        .to_string();
    let err = probe_ready(&addr, TIMEOUT).expect_err("nothing listens");
    assert!(err.starts_with("connect"), "got: {err}");
}

#[test]
fn a_server_that_never_answers_fails_within_the_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    // Accepted by the kernel backlog, never read or answered.
    let started = std::time::Instant::now();
    let err = probe_ready(&addr, Duration::from_millis(200)).expect_err("no answer");
    assert!(err.starts_with("read"), "got: {err}");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the probe waited {:?}",
        started.elapsed()
    );
    drop(listener);
}

/// A config file holding only `ops_addr`, kept alive by the returned guard.
fn config_with_ops_addr(addr: &str) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("coordinode.conf");
    std::fs::write(&path, format!("ops_addr: \"{addr}\"\n")).expect("write config");
    let path = path.to_str().expect("utf-8 path").to_owned();
    (dir, path)
}

/// With nothing given, the probe targets the server's own default ops
/// address, which listens on every interface.
#[test]
fn the_address_defaults_to_the_servers_default() {
    assert_eq!(ops_addr(None, None).as_deref(), Ok("[::]:7084"));
}

#[test]
fn the_config_files_ops_addr_is_used() {
    let (_dir, path) = config_with_ops_addr("127.0.0.1:9184");
    assert_eq!(ops_addr(None, Some(&path)).as_deref(), Ok("127.0.0.1:9184"));
}

/// The command line overrides the config file, as it does for `serve`.
#[test]
fn the_command_line_overrides_the_config_file() {
    let (_dir, path) = config_with_ops_addr("127.0.0.1:9184");
    assert_eq!(
        ops_addr(Some("127.0.0.1:9284".into()), Some(&path)).as_deref(),
        Ok("127.0.0.1:9284")
    );
}

/// A config file that cannot be read is an error, not a silent fall back to
/// the default address, which could belong to another server.
#[test]
fn an_unreadable_config_file_fails() {
    let err = ops_addr(None, Some("/nonexistent/coordinode.conf")).expect_err("no such file");
    assert!(err.starts_with("config error"), "got: {err}");
}

/// A listen address on every interface is reached on this host's loopback,
/// its own family first; a concrete address is used as given.
#[test]
fn a_wildcard_address_is_probed_on_loopback() {
    assert_eq!(
        probe_targets("[::]:7084"),
        Ok(vec![
            "[::1]:7084".parse().expect("addr"),
            "127.0.0.1:7084".parse().expect("addr"),
        ])
    );
    assert_eq!(
        probe_targets("0.0.0.0:7084"),
        Ok(vec![
            "127.0.0.1:7084".parse().expect("addr"),
            "[::1]:7084".parse().expect("addr"),
        ])
    );
    assert_eq!(
        probe_targets("10.1.2.3:7084"),
        Ok(vec!["10.1.2.3:7084".parse().expect("addr")])
    );
    assert!(probe_targets("not an address").is_err());
}

/// A server on IPv4 loopback only, probed through the wildcard `[::]`: the
/// IPv6 loopback refuses, and the probe goes on to the IPv4 one.
#[test]
fn the_probe_moves_on_when_a_loopback_refuses() {
    let (addr, server) =
        answer_once("HTTP/1.1 200 OK\r\nContent-Length: 14\r\n\r\n{\"ready\":true}");
    let port = addr.rsplit(':').next().expect("port").to_owned();
    assert_eq!(probe_ready(&format!("[::]:{port}"), TIMEOUT), Ok(()));
    server.join().expect("server thread");
}
