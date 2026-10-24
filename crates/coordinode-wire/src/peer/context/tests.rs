#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};

use rustls::server::WebPkiClientVerifier;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use super::*;
use crate::peer::test_pki::{Ca, ca, client_with_server_name, leaf, node, provider, sid};
use crate::peer::{HostIdentity, HostIdentityAdapter, TrustAnchors};

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

/// Explicit backends cannot bypass identity admission on construction or rotation.
#[tokio::test]
async fn external_key_backend_obeys_identity_policy() {
    use rustls::crypto::KeyProvider;
    use rustls::pki_types::{PrivatePkcs1KeyDer, PrivatePkcs8KeyDer};

    #[derive(Debug)]
    struct Resolver {
        signing: Arc<dyn rustls::sign::SigningKey>,
        calls: AtomicUsize,
    }
    impl KeyProvider for Resolver {
        fn load_private_key(
            &self,
            _: PrivateKeyDer<'static>,
        ) -> Result<Arc<dyn rustls::sign::SigningKey>, rustls::Error> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::clone(&self.signing))
        }
    }
    fn refused_keys() -> [PrivateKeyDer<'static>; 2] {
        // The PKCS#8 header identifies RSA; admission must refuse it before any backend sees its body.
        let rsa_pkcs8 = vec![
            0x30, 0x14, 0x02, 0x01, 0x00, 0x30, 0x0d, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7,
            0x0d, 0x01, 0x01, 0x01, 0x05, 0x00, 0x04, 0x00,
        ];
        [
            PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(Vec::new())),
            PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(rsa_pkcs8)),
        ]
    }
    let ca = ca();
    let (chain, key) = leaf(&ca, &["spiffe://td.test/a"]);
    let base = provider();
    let resolver = Box::leak(Box::new(Resolver {
        signing: base.key_provider.load_private_key(key.clone_key()).unwrap(),
        calls: AtomicUsize::new(0),
    }));
    let external = Arc::new(CryptoProvider {
        key_provider: resolver,
        ..base.as_ref().clone()
    });
    for refused in refused_keys() {
        assert!(
            SecurityContext::new(
                Arc::clone(&external),
                chain.clone(),
                refused,
                crate::peer::test_pki::spiffe("td.test", &ca)
            )
            .is_err()
        );
    }
    assert_eq!(resolver.calls.load(Ordering::Relaxed), 0);
    let context = Arc::new(
        SecurityContext::new(
            external,
            chain.clone(),
            key,
            crate::peer::test_pki::spiffe("td.test", &ca),
        )
        .unwrap(),
    );
    let before = context.state.load_full();
    for refused in refused_keys() {
        assert!(context.update_credentials(chain.clone(), refused).is_err());
    }
    assert_eq!(resolver.calls.load(Ordering::Relaxed), 1);
    assert!(Arc::ptr_eq(&before, &context.state.load_full()));
    let peer = node(&ca, "spiffe://td.test/b");
    let presented = handshake(
        context.server_config(InboundAuth::NodeOnly).unwrap(),
        peer.client_config(sid("spiffe://td.test/a")).unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(presented, chain[0]);
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

/// A CA replaced outright changes the certificate and the trust at once:
/// neither half validates on its own, the replacement of both does, and
/// connections made under the old trust are verified again. A replacement
/// naming another identity leaves everything in place.
#[tokio::test]
async fn credentials_and_trust_are_replaced_together() {
    let old = ca();
    let new = ca();
    let only = |ca: &Ca| crate::peer::test_pki::spiffe("td.test", ca);
    let (chain, key) = leaf(&old, &["spiffe://td.test/a"]);
    let a = Arc::new(SecurityContext::new(provider(), chain, key, only(&old)).expect("a"));
    let generation = a.generation();

    let (renewed, renewed_key) = leaf(&new, &["spiffe://td.test/a"]);
    assert!(
        a.update_credentials(renewed.clone(), renewed_key).is_err(),
        "the new certificate is not trusted under the old bundle"
    );
    assert!(
        a.update_trust(only(&new)).is_err(),
        "the old certificate is not trusted under the new bundle"
    );

    let (moved, moved_key) = leaf(&new, &["spiffe://td.test/z"]);
    assert!(matches!(
        a.replace(moved, moved_key, only(&new)),
        Err(ContextError::IdentityChanged { .. })
    ));
    assert_eq!(
        a.generation(),
        generation,
        "a refused replacement changes nothing"
    );

    let (renewed, renewed_key) = leaf(&new, &["spiffe://td.test/a"]);
    a.replace(renewed.clone(), renewed_key, only(&new))
        .expect("both halves together");
    assert!(
        a.generation() > generation,
        "connections are verified again"
    );
    let b = node(&new, "spiffe://td.test/b");
    let presented = handshake(
        a.server_config(InboundAuth::NodeOnly).expect("s"),
        b.client_config(sid("spiffe://td.test/a")).expect("c"),
    )
    .await
    .expect("a peer of the new CA reaches a");
    assert_eq!(presented, renewed[0]);
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
