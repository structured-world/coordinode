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

fn request(port: u16, path: &str, body: Option<&str>) -> std::io::Result<String> {
    let mut stream = TcpStream::connect(("127.0.0.1", port))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;
    let method = if body.is_some() { "POST" } else { "GET" };
    let body = body.unwrap_or_default();
    // HTTP/1.0 closes after the response, avoiding a second HTTP implementation
    // in this test and allowing the response body to be read to EOF.
    write!(
        stream,
        "{method} {path} HTTP/1.0\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
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
        if let Ok(response) = request(ops, "/ready", None) {
            if response.starts_with("HTTP/1.0 200") || response.starts_with("HTTP/1.1 200") {
                break;
            }
        }
        assert!(Instant::now() < deadline, "server did not become ready");
        std::thread::sleep(Duration::from_millis(100));
    }
    let response = request(
        rest,
        "/v1/query/cypher",
        Some(r#"{"query":"CREATE (:RestProbe {value: 37})"}"#),
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
        let response = client
            .execute_cypher(ExecuteCypherRequest {
                query: "MATCH (n:RestProbe {value: 37}) RETURN n.value".into(),
                ..Default::default()
            })
            .await?
            .into_inner();
        assert_eq!(response.rows.len(), 1, "gRPC sees the REST commit");
        Ok::<(), Box<dyn std::error::Error>>(())
    })?;
    Ok(())
}
