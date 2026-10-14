#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType,
};
use rustls::pki_types::PrivatePkcs8KeyDer;
use rustls::server::WebPkiClientVerifier;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::peer::{HostIdentity, HostIdentityAdapter, SpiffeId, TrustAnchors};

struct Ca {
    der: CertificateDer<'static>,
    issuer: Issuer<'static, KeyPair>,
}

fn ca() -> Ca {
    let key = KeyPair::generate().expect("key");
    let mut params = CertificateParams::new(Vec::<String>::new()).expect("params");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![KeyUsagePurpose::KeyCertSign, KeyUsagePurpose::CrlSign];
    let der = params.self_signed(&key).expect("ca").der().clone();
    Ca {
        der,
        issuer: Issuer::new(params, key),
    }
}

/// A leaf for `sans` (URIs if they start with `spiffe:`, DNS names
/// otherwise) and its key.
fn leaf(ca: &Ca, sans: &[&str]) -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
    let key = KeyPair::generate().expect("key");
    let mut params = CertificateParams::new(Vec::<String>::new()).expect("params");
    params.subject_alt_names = sans
        .iter()
        .map(|s| {
            if s.starts_with("spiffe:") {
                SanType::URI((*s).try_into().expect("uri"))
            } else {
                SanType::DnsName((*s).try_into().expect("dns"))
            }
        })
        .collect();
    params.key_usages = vec![KeyUsagePurpose::DigitalSignature];
    params.extended_key_usages = vec![
        ExtendedKeyUsagePurpose::ClientAuth,
        ExtendedKeyUsagePurpose::ServerAuth,
    ];
    let cert = params.signed_by(&key, &ca.issuer).expect("leaf");
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der()));
    (vec![cert.der().clone()], key)
}

fn provider() -> Arc<CryptoProvider> {
    crate::tls::rustcrypto_provider()
}

fn spiffe(domain: &str, ca: &Ca) -> PeerProfile {
    PeerProfile::Spiffe {
        bundles: HashMap::from([(
            domain.to_owned(),
            TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("anchor"),
        )]),
    }
}

fn node(ca: &Ca, id: &str) -> Arc<SecurityContext> {
    let (chain, key) = leaf(ca, &[id]);
    Arc::new(SecurityContext::new(provider(), chain, key, spiffe("td.test", ca)).expect("context"))
}

fn sid(s: &str) -> PeerIdentity {
    PeerIdentity::Spiffe(SpiffeId::parse(s).expect("id"))
}

/// One TLS handshake over an in-memory pipe, then one byte each way.
/// Returns the client-side error, if any, and the leaf the client saw.
async fn handshake(
    server: ServerConfig,
    client: ClientConfig,
) -> Result<CertificateDer<'static>, String> {
    let (a, b) = tokio::io::duplex(64 * 1024);
    let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server));
    let connector = tokio_rustls::TlsConnector::from(Arc::new(client));
    let name = ServerName::try_from("ignored.invalid").expect("name");
    let serve = async move {
        let mut stream = acceptor.accept(a).await.map_err(|e| e.to_string())?;
        let mut byte = [0u8; 1];
        stream
            .read_exact(&mut byte)
            .await
            .map_err(|e| e.to_string())?;
        stream.write_all(&byte).await.map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())?;
        Ok::<_, String>(())
    };
    let dial = async move {
        let mut stream = connector
            .connect(name, b)
            .await
            .map_err(|e| e.to_string())?;
        stream.write_all(&[7]).await.map_err(|e| e.to_string())?;
        stream.flush().await.map_err(|e| e.to_string())?;
        let mut byte = [0u8; 1];
        stream
            .read_exact(&mut byte)
            .await
            .map_err(|e| e.to_string())?;
        let leaf = stream
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|c| c.first().cloned())
            .ok_or("no peer certificate")?;
        Ok::<_, String>(leaf.into_owned())
    };
    let (served, dialed) = tokio::join!(serve, dial);
    // The dialing side's own reason first: the listener only sees the
    // alert it was sent.
    let leaf = dialed?;
    served?;
    Ok(leaf)
}

/// Two nodes of one trust domain connect, each proving the identity the
/// other expects; the address dialed plays no part.
#[tokio::test]
async fn nodes_connect_when_each_is_who_the_other_expects() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let b = node(&ca, "spiffe://td.test/b");
    handshake(
        a.server_config(InboundAuth::NodeOnly).expect("server"),
        b.client_config(sid("spiffe://td.test/a")).expect("client"),
    )
    .await
    .expect("a node meets the peer it expects");
}

/// A dialing node refuses a peer that proves another identity than the
/// one it expected, even one it would trust otherwise.
#[tokio::test]
async fn an_unexpected_peer_is_refused_outbound() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let b = node(&ca, "spiffe://td.test/b");
    let error = handshake(
        a.server_config(InboundAuth::NodeOnly).expect("server"),
        b.client_config(sid("spiffe://td.test/c")).expect("client"),
    )
    .await
    .expect_err("the peer is not the one expected");
    assert!(error.contains("expected spiffe://td.test/c"), "{error}");
}

/// A node-only listener refuses an ordinary client certificate from the
/// same CA and a client with no certificate.
#[tokio::test]
async fn a_node_only_listener_admits_nodes_only() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let (client_chain, client_key) = leaf(&ca, &["client.example"]);
    let mut roots = rustls::RootCertStore::empty();
    roots.add(ca.der.clone()).expect("root");
    let client = ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .expect("versions")
        .with_root_certificates(roots.clone())
        .with_client_auth_cert(client_chain, client_key)
        .expect("client cert");
    assert!(
        handshake(a.server_config(InboundAuth::NodeOnly).expect("s"), client)
            .await
            .is_err(),
        "a client certificate is not a node identity"
    );
    let anonymous = ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .expect("versions")
        .with_root_certificates(roots)
        .with_no_client_auth();
    assert!(
        handshake(
            a.server_config(InboundAuth::NodeOnly).expect("s"),
            anonymous
        )
        .await
        .is_err(),
        "a node-only listener requires a certificate"
    );
}

fn client_with_server_name(
    roots: rustls::RootCertStore,
    identity: Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>,
) -> ClientConfig {
    let builder = ClientConfig::builder_with_provider(provider())
        .with_safe_default_protocol_versions()
        .expect("versions")
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AcceptAnyServer));
    let _ = roots;
    match identity {
        Some((chain, key)) => builder.with_client_auth_cert(chain, key).expect("cert"),
        None => builder.with_no_client_auth(),
    }
}

/// A test client that does not check the server: the shared-port cases
/// test the server's decision about the client only.
#[derive(Debug)]
struct AcceptAnyServer;

impl ServerCertVerifier for AcceptAnyServer {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &ServerName<'_>,
        _: &[u8],
        _: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        Ok(ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &provider().signature_verification_algorithms,
        )
    }
    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &provider().signature_verification_algorithms,
        )
    }
    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// On a port shared with clients, a client without a certificate and one
/// with an ordinary certificate the client verifier accepts get through,
/// while a certificate claiming a node identity it cannot prove does not
/// fall through to client authentication.
#[tokio::test]
async fn a_shared_port_keeps_client_and_node_authentication_apart() {
    let other = ca();
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let mut client_roots = rustls::RootCertStore::empty();
    client_roots.add(ca.der.clone()).expect("root");
    let client_verifier =
        WebPkiClientVerifier::builder_with_provider(Arc::new(client_roots.clone()), provider())
            .allow_unauthenticated()
            .build()
            .expect("client verifier");
    let shared = || {
        a.server_config(InboundAuth::Shared {
            clients: Some(Arc::clone(&client_verifier)),
        })
        .expect("server")
    };

    handshake(
        shared(),
        client_with_server_name(client_roots.clone(), None),
    )
    .await
    .expect("a client without a certificate");
    handshake(
        shared(),
        client_with_server_name(client_roots.clone(), Some(leaf(&ca, &["client.example"]))),
    )
    .await
    .expect("an ordinary client certificate goes to client authentication");

    // A node-looking certificate from an untrusted CA: the client
    // verifier would refuse it too, but the point is that it is judged as
    // a node and refused there.
    let forged = leaf(&other, &["spiffe://td.test/a"]);
    assert!(
        handshake(
            shared(),
            client_with_server_name(client_roots.clone(), Some(forged))
        )
        .await
        .is_err()
    );
    // With no client verifier, a shared port refuses ordinary client
    // certificates rather than treating them as anonymous.
    let no_client_certs = a
        .server_config(InboundAuth::Shared { clients: None })
        .expect("server");
    assert!(
        handshake(
            no_client_certs,
            client_with_server_name(client_roots, Some(leaf(&ca, &["client.example"])))
        )
        .await
        .is_err()
    );
}

/// A renewal keeping the identity takes effect for the next handshake;
/// a replacement whose key does not match, or that names another
/// identity, is refused and the node keeps presenting what it had.
#[tokio::test]
async fn credential_updates_are_validated_whole_before_they_count() {
    let ca = ca();
    let a = node(&ca, "spiffe://td.test/a");
    let b = node(&ca, "spiffe://td.test/b");
    let first = handshake(
        a.server_config(InboundAuth::NodeOnly).expect("s"),
        b.client_config(sid("spiffe://td.test/a")).expect("c"),
    )
    .await
    .expect("before renewal");

    let (renewed_chain, renewed_key) = leaf(&ca, &["spiffe://td.test/a"]);
    a.update_credentials(renewed_chain.clone(), renewed_key)
        .expect("a renewal keeping the identity");
    let second = handshake(
        a.server_config(InboundAuth::NodeOnly).expect("s"),
        b.client_config(sid("spiffe://td.test/a")).expect("c"),
    )
    .await
    .expect("after renewal");
    assert_ne!(first, second);
    assert_eq!(second, renewed_chain[0]);

    let (other_chain, _) = leaf(&ca, &["spiffe://td.test/a"]);
    let (_, unrelated_key) = leaf(&ca, &["spiffe://td.test/a"]);
    assert!(matches!(
        a.update_credentials(other_chain, unrelated_key),
        Err(ContextError::KeyMismatch(_))
    ));
    let (moved_chain, moved_key) = leaf(&ca, &["spiffe://td.test/z"]);
    assert!(matches!(
        a.update_credentials(moved_chain, moved_key),
        Err(ContextError::IdentityChanged { .. })
    ));
    let third = handshake(
        a.server_config(InboundAuth::NodeOnly).expect("s"),
        b.client_config(sid("spiffe://td.test/a")).expect("c"),
    )
    .await
    .expect("refused updates left the renewal in force");
    assert_eq!(third, renewed_chain[0]);
}

/// Withdrawn trust takes effect for the next handshake; a trust update
/// under which the node itself would not be valid is refused.
#[tokio::test]
async fn trust_updates_take_effect_and_cannot_orphan_the_node() {
    let a_ca = ca();
    let b_ca = ca();
    let bundles = |domains: &[(&str, &Ca)]| PeerProfile::Spiffe {
        bundles: domains
            .iter()
            .map(|(d, ca)| {
                let anchors = TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("a");
                ((*d).to_owned(), anchors)
            })
            .collect(),
    };
    let both = || bundles(&[("a.test", &a_ca), ("b.test", &b_ca)]);
    let (a_chain, a_key) = leaf(&a_ca, &["spiffe://a.test/a"]);
    let (b_chain, b_key) = leaf(&b_ca, &["spiffe://b.test/b"]);
    let a = Arc::new(SecurityContext::new(provider(), a_chain, a_key, both()).expect("a"));
    let b = Arc::new(SecurityContext::new(provider(), b_chain, b_key, both()).expect("b"));
    let dial = || async {
        handshake(
            a.server_config(InboundAuth::NodeOnly).expect("s"),
            b.client_config(sid("spiffe://a.test/a")).expect("c"),
        )
        .await
    };
    dial().await.expect("both trust domains trusted");

    // B withdraws a.test: the next connection to A is refused.
    b.update_trust(bundles(&[("b.test", &b_ca)]))
        .expect("b stays valid under its own domain");
    let error = dial().await.expect_err("a.test is no longer trusted");
    assert!(error.contains("a.test"), "{error}");

    // A trust update under which the node itself would not verify is
    // refused, and the node keeps its trust.
    assert!(matches!(
        a.update_trust(bundles(&[("b.test", &b_ca)])),
        Err(ContextError::OwnIdentity(_))
    ));
    b.update_trust(both()).expect("b trusts a.test again");
    dial().await.expect("a's trust was left in place");
}

/// A host adapter counting the chains it is asked about.
#[derive(Debug, Default)]
struct Counting(AtomicUsize);

impl HostIdentityAdapter for Counting {
    fn identify(
        &self,
        chain: &[CertificateDer<'_>],
        _now: UnixTime,
    ) -> Result<HostIdentity, String> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let fields = crate::peer::der::leaf_fields(&chain[0]).map_err(|e| e.to_string())?;
        let subject = fields
            .uris
            .first()
            .map(|u| String::from_utf8_lossy(u).into_owned())
            .ok_or("no subject")?;
        Ok(HostIdentity {
            scope: "instance-1".to_owned(),
            subject,
        })
    }
}

/// Every connection proves its certificate again: nothing resumes a
/// session past verification, so evidence that expired or was withdrawn
/// cannot be revived by resumption.
#[tokio::test]
async fn every_connection_is_verified_afresh() {
    let ca = ca();
    let counter = Arc::new(Counting::default());
    let profile = || PeerProfile::Host {
        anchors: TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("anchor"),
        adapter: Arc::clone(&counter) as Arc<dyn HostIdentityAdapter>,
    };
    let (a_chain, a_key) = leaf(&ca, &["spiffe://h/a"]);
    let (b_chain, b_key) = leaf(&ca, &["spiffe://h/b"]);
    let a = Arc::new(SecurityContext::new(provider(), a_chain, a_key, profile()).expect("a"));
    let b = Arc::new(SecurityContext::new(provider(), b_chain, b_key, profile()).expect("b"));
    let expected_a = PeerIdentity::Host(HostIdentity {
        scope: "instance-1".to_owned(),
        subject: "spiffe://h/a".to_owned(),
    });
    let server = Arc::new(a.server_config(InboundAuth::NodeOnly).expect("s"));
    let client = Arc::new(b.client_config(expected_a).expect("c"));
    let before = counter.0.load(Ordering::SeqCst);
    for _ in 0..3 {
        handshake((*server).clone(), (*client).clone())
            .await
            .expect("handshake");
    }
    // Each handshake verifies both certificates once.
    assert_eq!(counter.0.load(Ordering::SeqCst) - before, 6);
}
