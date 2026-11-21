//! The shipped REST listener transcodes to the database without proxy JWT auth.

#![cfg(feature = "rest-proxy")]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        if let Err(error) = self.0.kill() {
            eprintln!("stop test server: {error}");
        }
        if let Err(error) = self.0.wait() {
            eprintln!("reap test server: {error}");
        }
    }
}

/// The namespace a fresh server opens with, which a client names.
const NAMESPACE: &str = "default";

fn request(port: u16, path: &str, body: Option<&str>) -> std::io::Result<String> {
    request_in(port, path, body, None)
}

/// [`request`] naming `namespace` in the namespace header.
fn request_in(
    port: u16,
    path: &str,
    body: Option<&str>,
    namespace: Option<&str>,
) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let method = if body.is_some() { "POST" } else { "GET" };
    let body = body.unwrap_or_default();
    let namespace = namespace
        .map(|name| format!("X-CoordiNode-Namespace: {name}\r\n"))
        .unwrap_or_default();
    // HTTP/1.0 closes after the response, avoiding a second HTTP implementation
    // in this test and allowing the response body to be read to EOF.
    write!(
        stream,
        "{method} {path} HTTP/1.0\r\nHost: localhost\r\n{namespace}Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )?;
    let mut response = String::new();
    stream.take(64 * 1024).read_to_string(&mut response)?;
    Ok(response)
}

/// Exercise the actual binary, its provider installation, REST router and
/// upstream gRPC service. Disabling proxy defaults must preserve both writes
/// and reads without enabling a second authentication authority.
#[test]
fn rest_write_is_visible_over_grpc() -> Result<(), Box<dyn std::error::Error>> {
    let data = tempfile::tempdir()?;
    let [grpc, rest, ops, pg] = coordinode_test_fixtures::alloc_ports::<4>();
    let mut server = Server(
        Command::new(env!("CARGO_BIN_EXE_coordinode"))
            .args(["serve", "--data"])
            .arg(data.path())
            .args(["--addr", &format!("127.0.0.1:{grpc}")])
            .args(["--rest-addr", &format!("127.0.0.1:{rest}")])
            .args(["--ops-addr", &format!("127.0.0.1:{ops}")])
            .args(["--pg-addr", &format!("127.0.0.1:{pg}")])
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()?,
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        assert!(
            server.0.try_wait()?.is_none(),
            "server exited before readiness"
        );
        if let Ok(response) = request(ops, "/ready", None)
            && (response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200"))
        {
            break;
        }
        assert!(Instant::now() < deadline, "server did not become ready");
        std::thread::sleep(Duration::from_millis(100));
    }
    // A REST call that names no namespace is refused before it runs.
    let unnamed = request(
        rest,
        "/v1/query/cypher",
        Some(r#"{"query":"CREATE (:RestProbe {value: 1})"}"#),
    )?;
    assert!(
        unnamed.starts_with("HTTP/1.0 400") || unnamed.starts_with("HTTP/1.1 400"),
        "a REST call naming no namespace was served: {unnamed}"
    );
    let response = request_in(
        rest,
        "/v1/query/cypher",
        Some(r#"{"query":"CREATE (:RestProbe {value: 37})"}"#),
        Some(NAMESPACE),
    )?;
    assert!(
        response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200"),
        "REST write failed: {response}"
    );
    let (_, body) = response.split_once("\r\n\r\n").ok_or("missing HTTP body")?;
    let body: serde_json::Value = serde_json::from_str(body)?;
    assert_eq!(
        body["stats"]["nodesCreated"], "1",
        "REST must commit one node"
    );

    tokio::runtime::Runtime::new()?.block_on(async {
        use coordinode_server::proto::query::{
            ExecuteCypherRequest, cypher_service_client::CypherServiceClient,
        };
        let mut client = CypherServiceClient::connect(format!("http://127.0.0.1:{grpc}")).await?;
        let mut call = tonic::Request::new(ExecuteCypherRequest {
            query: "MATCH (n:RestProbe) RETURN n.value".into(),
            ..Default::default()
        });
        call.metadata_mut()
            .insert("x-coordinode-namespace", NAMESPACE.parse()?);
        let response = client.execute_cypher(call).await?.into_inner();
        assert_eq!(
            response.rows.len(),
            1,
            "gRPC sees the REST commit, and nothing of the refused call"
        );
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    Ok(())
}
