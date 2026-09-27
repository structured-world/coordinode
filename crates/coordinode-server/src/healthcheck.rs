//! `coordinode healthcheck`: ask a running server's ops port whether it is
//! ready, from the same binary, so an image without a shell or an HTTP client
//! (scratch, distroless) can still declare a container health check.

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

/// How long the probe waits to connect, and then for the answer, by default.
pub(crate) const DEFAULT_TIMEOUT_MS: u64 = 2_000;

/// Largest response head read; `/ready` answers with a few dozen bytes.
const MAX_RESPONSE: u64 = 4 * 1024;

/// The ops address to probe: `--ops-addr` when given, else `ops_addr` from
/// `--config` when given, else the server's built-in default.
///
/// # Errors
///
/// The config file cannot be read or parsed.
pub(crate) fn ops_addr(
    cli_ops_addr: Option<String>,
    config_path: Option<&str>,
) -> Result<String, String> {
    match cli_ops_addr {
        Some(addr) => Ok(addr),
        None => crate::config::ServerConfig::load(config_path)
            .map(|cfg| cfg.ops_addr)
            .map_err(|e| format!("config error: {e}")),
    }
}

/// The addresses a probe of `ops_addr` connects to, in order. A listen
/// address on every interface (`[::]`, `0.0.0.0`) is reached on this host's
/// loopback, its own family first; a host name gives every address it
/// resolves to.
///
/// # Errors
///
/// `ops_addr` is neither `ip:port` nor a resolvable `host:port`.
pub(crate) fn probe_targets(ops_addr: &str) -> Result<Vec<SocketAddr>, String> {
    if let Ok(addr) = ops_addr.parse::<SocketAddr>() {
        if !addr.ip().is_unspecified() {
            return Ok(vec![addr]);
        }
        let v4 = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), addr.port());
        let v6 = SocketAddr::new(IpAddr::V6(Ipv6Addr::LOCALHOST), addr.port());
        return Ok(if addr.is_ipv6() {
            vec![v6, v4]
        } else {
            vec![v4, v6]
        });
    }
    let targets: Vec<SocketAddr> = ops_addr
        .to_socket_addrs()
        .map_err(|e| format!("resolve {ops_addr}: {e}"))?
        .collect();
    if targets.is_empty() {
        return Err(format!("{ops_addr} resolves to no address"));
    }
    Ok(targets)
}

/// Ask the server at `ops_addr` for `GET /ready` and succeed only on `200`.
/// Each of [`probe_targets`] is tried until one accepts the connection; the
/// first that answers decides.
///
/// # Errors
///
/// No target accepts the connection, the exchange fails or times out, or the
/// server answers anything but `200`: a server that is still starting or
/// already draining answers `503`.
pub(crate) fn probe_ready(ops_addr: &str, timeout: Duration) -> Result<(), String> {
    let mut refused = Vec::new();
    for addr in probe_targets(ops_addr)? {
        match TcpStream::connect_timeout(&addr, timeout) {
            Ok(stream) => return ask_ready(stream, addr, timeout),
            Err(e) => refused.push(format!("connect {addr}: {e}")),
        }
    }
    Err(refused.join("; "))
}

/// One `/ready` exchange on a connected `stream`.
fn ask_ready(mut stream: TcpStream, addr: SocketAddr, timeout: Duration) -> Result<(), String> {
    stream
        .set_read_timeout(Some(timeout))
        .and_then(|()| stream.set_write_timeout(Some(timeout)))
        .map_err(|e| format!("set timeouts: {e}"))?;
    stream
        .write_all(
            format!("GET /ready HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n").as_bytes(),
        )
        .map_err(|e| format!("send to {addr}: {e}"))?;

    let mut response = Vec::new();
    stream
        .take(MAX_RESPONSE)
        .read_to_end(&mut response)
        .map_err(|e| format!("read from {addr}: {e}"))?;
    let status_line = response
        .split(|&b| b == b'\n')
        .next()
        .map(|line| String::from_utf8_lossy(line).trim_end().to_owned())
        .unwrap_or_default();
    // RFC 9112 4: status-line = HTTP-version SP status-code SP [ reason-phrase ].
    match status_line.split(' ').nth(1) {
        Some("200") => Ok(()),
        Some(code) => Err(format!("{addr} is not ready: /ready answered {code}")),
        None => Err(format!("{addr} answered no HTTP status: {status_line:?}")),
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
