#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::time::SystemTime;

use futures_util::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::peer::test_pki::{Ca, Credentials, ca, leaf, leaf_until, node, provider, sid, spiffe};
use crate::peer::{PeerProfile, TrustAnchors};

/// A listener on a free loopback port and its address.
async fn bind() -> (TcpListener, String) {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr").to_string();
    (listener, addr)
}

/// Dial `addr` as `from`, expecting `expected`, and exchange one byte with
/// whatever `server` accepted. Returns the dialer's error, if any.
async fn exchange(
    from: &Arc<SecurityContext>,
    expected: PeerIdentity,
    addr: &str,
    server: &mut Incoming,
) -> Result<InboundStream, String> {
    let config = Arc::new(from.client_config(expected).expect("client"));
    // The listener's side of the handshake runs in its own task; the
    // stream hands over only what completed it.
    let mut stream = connect_tls(Arc::clone(from), config, addr)
        .await
        .map_err(|e| e.to_string())?;
    stream.write_all(&[7]).await.map_err(|e| e.to_string())?;
    stream.flush().await.map_err(|e| e.to_string())?;
    let mut accepted = tokio::time::timeout(Duration::from_secs(5), server.next())
        .await
        .ok()
        .flatten()
        .ok_or("nothing accepted")?
        .expect("accepted");
    let mut byte = [0u8; 1];
    accepted
        .read_exact(&mut byte)
        .await
        .map_err(|e| e.to_string())?;
    Ok(accepted)
}

fn node_only() -> Accept {
    Accept {
        inbound: InboundAuth::NodeOnly,
        plaintext_clients: false,
        handshake_timeout: PeerTimeouts::default().handshake,
    }
}

/// A node dialing another expecting its identity is accepted as that node,
/// and the server learns the dialer's identity from the connection, not
/// from anything it says.
#[tokio::test]
async fn a_dialed_node_learns_who_called() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let b = node(&ca, "spiffe://td.test/b");
    let (listener, addr) = bind().await;
    let mut server = incoming(listener, &a, node_only()).expect("incoming");
    let accepted = exchange(&b, sid("spiffe://td.test/a"), &addr, &mut server)
        .await
        .expect("b reaches a");
    let info = tonic::transport::server::Connected::connect_info(&accepted);
    let caller = info.node.expect("a node connection");
    assert_eq!(
        caller.identity().expect("current"),
        sid("spiffe://td.test/b")
    );
}

/// The dialer refuses a listener that proves another identity than the
/// one expected, though both are trusted.
#[tokio::test]
async fn the_dialer_refuses_an_unexpected_listener() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let b = node(&ca, "spiffe://td.test/b");
    let (listener, addr) = bind().await;
    let mut server = incoming(listener, &a, node_only()).expect("incoming");
    let error = exchange(&b, sid("spiffe://td.test/c"), &addr, &mut server)
        .await
        .err()
        .expect("the listener is not c");
    assert!(error.contains("expected spiffe://td.test/c"), "{error}");
}

/// A plain-text client gets a connection only on a port that takes them,
/// and never as a node; a node-only port refuses it.
#[tokio::test]
async fn plain_text_is_a_client_where_allowed_and_refused_elsewhere() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");

    let (listener, addr) = bind().await;
    let shared = Accept {
        inbound: InboundAuth::Shared { clients: None },
        plaintext_clients: true,
        handshake_timeout: PeerTimeouts::default().handshake,
    };
    let mut server = incoming(listener, &a, shared).expect("incoming");
    let mut client = TcpStream::connect(&addr).await.expect("connect");
    client.write_all(b"PRI * HTTP/2.0").await.expect("write");
    let accepted = tokio::time::timeout(Duration::from_secs(5), server.next())
        .await
        .expect("accepted in time")
        .expect("a connection")
        .expect("accepted");
    assert!(matches!(accepted, InboundStream::ClientPlain(..)));
    let info = tonic::transport::server::Connected::connect_info(&accepted);
    assert!(info.node.is_none(), "plain text is never a node");

    let (nodes_only, addr) = bind().await;
    let mut server = incoming(nodes_only, &a, node_only()).expect("incoming");
    let mut client = TcpStream::connect(&addr).await.expect("connect");
    client
        .write_all(b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
        .await
        .expect("write");
    let accepted = tokio::time::timeout(Duration::from_millis(500), server.next()).await;
    assert!(accepted.is_err(), "a node-only port accepts no plain text");
}

/// On a port shared with clients, a client presenting an ordinary
/// certificate the client verifier accepts, issued by the very CA that
/// issues node identities, is a client: its connection carries no node
/// identity, so no inter-node service admits it.
#[tokio::test]
async fn an_ordinary_client_certificate_is_never_a_node() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let clients = crate::tls::client_verifier(vec![ca.der.clone()], true).expect("verifier");
    let (listener, addr) = bind().await;
    let shared = Accept {
        inbound: InboundAuth::Shared {
            clients: Some(clients),
        },
        plaintext_clients: false,
        handshake_timeout: PeerTimeouts::default().handshake,
    };
    let mut server = incoming(listener, &a, shared).expect("incoming");

    let config = crate::peer::test_pki::client_with_server_name(
        rustls::RootCertStore::empty(),
        Some(leaf(&ca, &["client.example"])),
    );
    let tcp = TcpStream::connect(&addr).await.expect("connect");
    let mut tls = tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(ServerName::try_from("client.example").expect("name"), tcp)
        .await
        .expect("the client's certificate is accepted");
    tls.write_all(&[7]).await.expect("write");
    tls.flush().await.expect("flush");

    let accepted = tokio::time::timeout(Duration::from_secs(5), server.next())
        .await
        .expect("accepted in time")
        .expect("a connection")
        .expect("accepted");
    assert!(matches!(accepted, InboundStream::ClientTls(..)));
    let info = tonic::transport::server::Connected::connect_info(&accepted);
    assert!(
        info.node.is_none(),
        "a client certificate is no node identity"
    );
}

/// Dial `addr` presenting `identity` and return what the shared port made
/// of the connection, or `None` when it took none.
async fn shared_port_takes(
    addr: &str,
    server: &mut Incoming,
    identity: Credentials,
) -> Option<InboundStream> {
    let config = crate::peer::test_pki::client_with_server_name(
        rustls::RootCertStore::empty(),
        Some(identity),
    );
    let tcp = TcpStream::connect(addr).await.expect("connect");
    let mut tls = tokio_rustls::TlsConnector::from(Arc::new(config))
        .connect(ServerName::try_from("client.example").expect("name"), tcp)
        .await
        .ok()?;
    // A TLS 1.3 server judges the client's certificate after the client
    // has finished; its verdict arrives with the first exchange.
    tls.write_all(&[7]).await.ok()?;
    tls.flush().await.ok()?;
    tokio::time::timeout(Duration::from_secs(2), server.next())
        .await
        .ok()
        .flatten()
        .map(|accepted| accepted.expect("accepted"))
}

/// A port shared with clients that trusts `client_ca` for client
/// certificates, held by node `spiffe://td.test/a`.
async fn shared_port_trusting(node_ca: &Ca, client_ca: &Ca) -> (Incoming, String) {
    let a = node(node_ca, "spiffe://td.test/a");
    let clients = crate::tls::client_verifier(vec![client_ca.der.clone()], true).expect("verifier");
    let (listener, addr) = bind().await;
    let shared = Accept {
        inbound: InboundAuth::Shared {
            clients: Some(clients),
        },
        plaintext_clients: false,
        handshake_timeout: PeerTimeouts::default().handshake,
    };
    (incoming(listener, &a, shared).expect("incoming"), addr)
}

/// An application whose client certificate is an X.509-SVID of a trust
/// domain the node holds no bundle for is a client like any other: the
/// client verifier decides, and its connection carries no node identity.
#[tokio::test]
async fn a_workload_identity_of_another_trust_domain_is_a_client() {
    let node_ca = ca();
    let apps_ca = ca();
    let (mut server, addr) = shared_port_trusting(&node_ca, &apps_ca).await;
    let accepted = shared_port_takes(
        &addr,
        &mut server,
        leaf(&apps_ca, &["spiffe://apps.test/web"]),
    )
    .await
    .expect("the client verifier accepts the workload's certificate");
    let info = tonic::transport::server::Connected::connect_info(&accepted);
    assert!(info.node.is_none(), "a client, not a node");
}

/// A certificate naming an identity in the node's own trust domain is
/// judged as a node identity only, under that domain's bundle: one the
/// bundle does not verify is refused, though the client verifier would
/// accept its issuer.
#[tokio::test]
async fn an_identity_in_a_node_trust_domain_is_never_left_to_the_clients() {
    let node_ca = ca();
    let apps_ca = ca();
    let (mut server, addr) = shared_port_trusting(&node_ca, &apps_ca).await;
    let taken =
        shared_port_takes(&addr, &mut server, leaf(&apps_ca, &["spiffe://td.test/b"])).await;
    assert!(taken.is_none(), "a node identity the bundle refuses");
    let malformed =
        shared_port_takes(&addr, &mut server, leaf(&apps_ca, &["spiffe://TD.test/b"])).await;
    assert!(
        malformed.is_none(),
        "a SPIFFE ID that does not parse claims a node"
    );
}

/// A connection authenticated under trust that is then withdrawn stops
/// carrying traffic: its next read fails, and its identity no longer
/// resolves. A trust update the peer still verifies under leaves it be.
#[tokio::test]
async fn withdrawn_trust_ends_an_established_connection() {
    let a_ca = ca();
    let b_ca = ca();
    let both = || bundles(&[("a.test", &a_ca), ("b.test", &b_ca)]);
    let (a_chain, a_key) = leaf(&a_ca, &["spiffe://a.test/a"]);
    let (b_chain, b_key) = leaf(&b_ca, &["spiffe://b.test/b"]);
    let a = Arc::new(SecurityContext::new(provider(), a_chain, a_key, both()).expect("a"));
    let b = Arc::new(SecurityContext::new(provider(), b_chain, b_key, both()).expect("b"));
    let (listener, addr) = bind().await;
    let mut server = incoming(listener, &a, node_only()).expect("incoming");
    let mut accepted = exchange(&b, sid("spiffe://a.test/a"), &addr, &mut server)
        .await
        .expect("connected");
    let caller = tonic::transport::server::Connected::connect_info(&accepted)
        .node
        .expect("node");

    // Renewing trust b still verifies under changes nothing.
    a.update_trust(both()).expect("same trust");
    assert_eq!(
        caller.identity().expect("still b"),
        sid("spiffe://b.test/b")
    );

    a.update_trust(bundles(&[("a.test", &a_ca)]))
        .expect("a drops b.test");
    assert!(caller.identity().is_err(), "b.test is no longer trusted");
    let mut byte = [0u8; 1];
    let read = accepted.read(&mut byte).await;
    assert_eq!(
        read.expect_err("the connection is cut").kind(),
        io::ErrorKind::PermissionDenied
    );
}

/// A connection's peer stops being admitted the moment its certificate
/// expires, without any trust update.
#[tokio::test]
async fn an_expired_certificate_ends_an_established_connection() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let soon = SystemTime::now() + Duration::from_secs(2);
    let (chain, key) = leaf_until(&ca, &["spiffe://td.test/b"], Some(soon));
    let b =
        Arc::new(SecurityContext::new(provider(), chain, key, spiffe("td.test", &ca)).expect("b"));
    let (listener, addr) = bind().await;
    let mut server = incoming(listener, &a, node_only()).expect("incoming");
    let accepted = exchange(&b, sid("spiffe://td.test/a"), &addr, &mut server)
        .await
        .expect("connected");
    let caller = tonic::transport::server::Connected::connect_info(&accepted)
        .node
        .expect("node");
    assert!(caller.identity().is_ok());
    tokio::time::sleep(Duration::from_secs(3)).await;
    let error = caller.identity().expect_err("expired");
    assert!(error.to_string().contains("xpired"), "{error}");
}

/// A peer's standing lapses when its certificate does: right after the last
/// valid second of notAfter, not up to a second later because the moment it
/// was verified fell late in a second.
#[test]
fn a_standing_lapses_with_the_certificate_not_a_second_later() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let now = SystemTime::now();
    let whole = now.duration_since(UNIX_EPOCH).expect("clock").as_secs() + 3;
    let not_after = UNIX_EPOCH + Duration::from_secs(whole);
    let (chain, _) = leaf_until(&ca, &["spiffe://td.test/b"], Some(not_after));
    // Verify late in a second, where rounding to whole seconds errs most.
    let into_second = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .subsec_millis();
    if into_second < 900 {
        std::thread::sleep(Duration::from_millis(u64::from(900 - into_second)));
    }
    let peer = AuthenticatedPeer::establish(a, chain, PeerRole::Client).expect("valid");
    let until = peer.standing.load().until.expect("a deadline");
    // The certificate is invalid from the second after notAfter on
    // (RFC 5280 4.1.2.5).
    let invalid_from = UNIX_EPOCH + Duration::from_secs(whole + 1);
    let left = invalid_from
        .duration_since(SystemTime::now())
        .expect("still valid");
    let boundary = Instant::now() + left;
    assert!(
        until <= boundary + Duration::from_millis(50),
        "the standing outlives the certificate by {:?}",
        until - boundary
    );
}

/// One connection per peer identity: asking again reuses it, a peer found
/// at a new address gets a new one, and an address without a port is
/// refused before anything is dialed.
#[tokio::test]
async fn connections_are_kept_per_peer_identity() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let connections = PeerConnections::new(a);
    let b = PeerTarget {
        addr: "http://127.0.0.1:1".to_owned(),
        identity: sid("spiffe://td.test/b"),
    };
    connections.channel(&b).expect("b");
    connections.channel(&b).expect("b again");
    assert_eq!(connections.len(), 1);
    let moved = PeerTarget {
        addr: "127.0.0.1:2".to_owned(),
        ..b.clone()
    };
    connections.channel(&moved).expect("b moved");
    assert_eq!(connections.len(), 1, "still one peer");
    let c = PeerTarget {
        addr: "127.0.0.1:3".to_owned(),
        identity: sid("spiffe://td.test/c"),
    };
    connections.channel(&c).expect("c");
    assert_eq!(connections.len(), 2);
    let bare = PeerTarget {
        addr: "no-port".to_owned(),
        identity: sid("spiffe://td.test/d"),
    };
    assert!(matches!(
        connections.channel(&bare),
        Err(DialError::Address { .. })
    ));
}

/// A dial to a peer that takes the TCP connection and never answers the TLS
/// handshake fails at the configured connect bound, not the default one.
#[tokio::test]
async fn a_silent_peer_fails_the_dial_at_its_bound() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    // Bound but never accepted: the kernel completes TCP, nobody speaks TLS.
    let (_silent, addr) = bind().await;
    let connect = Duration::from_millis(300);
    let connections = PeerConnections::new(a).with_timeouts(PeerTimeouts {
        connect,
        ..PeerTimeouts::default()
    });
    let mut channel = connections
        .channel(&PeerTarget {
            addr: addr.clone(),
            identity: sid("spiffe://td.test/b"),
        })
        .expect("a channel");
    let started = Instant::now();
    // The channel is lazy: only a call makes it dial.
    let call = async {
        std::future::poll_fn(|cx| tonic::codegen::Service::poll_ready(&mut channel, cx)).await?;
        let request = http::Request::builder()
            .uri(format!("http://{addr}/test.Peer/Call"))
            .body(tonic::body::Body::empty())
            .expect("request");
        tonic::codegen::Service::call(&mut channel, request).await
    };
    let answered = tokio::time::timeout(Duration::from_secs(4), call)
        .await
        .expect("the dial ends before the default bound");
    assert!(answered.is_err(), "nobody answered the handshake");
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "took {:?}",
        started.elapsed()
    );
}

/// A connection that opens a TLS handshake and stalls is dropped at the
/// listener's handshake bound, freeing what it held.
#[tokio::test]
async fn a_stalled_handshake_is_dropped_at_its_bound() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let (listener, addr) = bind().await;
    let accept = Accept {
        handshake_timeout: Duration::from_millis(300),
        ..node_only()
    };
    let _server = incoming(listener, &a, accept).expect("incoming");
    let mut stalled = TcpStream::connect(&addr).await.expect("connect");
    // The first byte of a TLS handshake record, then nothing.
    stalled.write_all(&[22]).await.expect("write");
    let mut byte = [0u8; 1];
    let closed = tokio::time::timeout(Duration::from_secs(4), stalled.read(&mut byte))
        .await
        .expect("the listener gives up before the default bound");
    assert!(
        matches!(closed, Ok(0) | Err(_)),
        "the stalled connection is closed: {closed:?}"
    );
}

fn bundles(domains: &[(&str, &Ca)]) -> PeerProfile {
    PeerProfile::Spiffe {
        bundles: domains
            .iter()
            .map(|(d, ca)| {
                let anchors = TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("a");
                ((*d).to_owned(), anchors)
            })
            .collect(),
    }
}
