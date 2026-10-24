use alloc::{format, sync::Arc, vec, vec::Vec};
use std::io::{Read, Write};

use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{
    ClientConfig, ClientConnection, Error, RootCertStore, ServerConfig, ServerConnection,
};

struct Authority {
    issuer: rcgen::Issuer<'static, KeyPair>,
    root: CertificateDer<'static>,
}

impl Authority {
    fn new() -> Self {
        let key = KeyPair::generate().unwrap();
        let mut params = CertificateParams::new(Vec::new()).unwrap();
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let root = params.self_signed(&key).unwrap().der().clone();
        Self {
            issuer: rcgen::Issuer::new(params, key),
            root,
        }
    }

    fn leaf(
        &self,
        algorithm: &'static rcgen::SignatureAlgorithm,
        expired: bool,
    ) -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
        let key = KeyPair::generate_for(algorithm).unwrap();
        let mut params = CertificateParams::new(vec!["localhost".into()]).unwrap();
        params.extended_key_usages = vec![
            ExtendedKeyUsagePurpose::ServerAuth,
            ExtendedKeyUsagePurpose::ClientAuth,
        ];
        if expired {
            params.not_before = rcgen::date_time_ymd(1999, 1, 1);
            params.not_after = rcgen::date_time_ymd(2000, 1, 1);
        }
        let certificate = params.signed_by(&key, &self.issuer).unwrap();
        (vec![certificate.der().clone()], key.into())
    }

    fn roots(&self) -> RootCertStore {
        let mut roots = RootCertStore::empty();
        roots.add(self.root.clone()).unwrap();
        roots
    }
}

fn configs(
    server_provider: CryptoProvider,
    client_provider: CryptoProvider,
    authority: &Authority,
    leaf: &(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>),
    mutual: bool,
) -> (ServerConfig, ClientConfig) {
    let server_provider = Arc::new(server_provider);
    let server = ServerConfig::builder_with_provider(Arc::clone(&server_provider))
        .with_safe_default_protocol_versions()
        .unwrap();
    let server = if mutual {
        let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
            Arc::new(authority.roots()),
            server_provider,
        )
        .build()
        .unwrap();
        server.with_client_cert_verifier(verifier)
    } else {
        server.with_no_client_auth()
    };
    let server = server
        .with_single_cert(leaf.0.clone(), leaf.1.clone_key())
        .unwrap();
    let client = ClientConfig::builder_with_provider(Arc::new(client_provider))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_root_certificates(authority.roots());
    let client = if mutual {
        client
            .with_client_auth_cert(leaf.0.clone(), leaf.1.clone_key())
            .unwrap()
    } else {
        client.with_no_client_auth()
    };
    (server, client)
}

fn handshake(server: ServerConfig, client: ClientConfig, name: &'static str) -> Result<(), Error> {
    let mut server = ServerConnection::new(Arc::new(server))?;
    let mut client = ClientConnection::new(Arc::new(client), ServerName::try_from(name).unwrap())?;
    for _ in 0..32 {
        let mut flight = Vec::new();
        client.write_tls(&mut flight).unwrap();
        server.read_tls(&mut flight.as_slice()).unwrap();
        server.process_new_packets()?;
        flight.clear();
        server.write_tls(&mut flight).unwrap();
        client.read_tls(&mut flight.as_slice()).unwrap();
        client.process_new_packets()?;
        if !client.is_handshaking() && !server.is_handshaking() {
            client.writer().write_all(b"client data").unwrap();
            flight.clear();
            client.write_tls(&mut flight).unwrap();
            server.read_tls(&mut flight.as_slice()).unwrap();
            server.process_new_packets()?;
            let mut payload = [0; 11];
            server.reader().read_exact(&mut payload).unwrap();
            if &payload != b"client data" {
                return Err(Error::General("decrypted client data changed".into()));
            }
            server.writer().write_all(b"server data").unwrap();
            flight.clear();
            server.write_tls(&mut flight).unwrap();
            client.read_tls(&mut flight.as_slice()).unwrap();
            client.process_new_packets()?;
            client.reader().read_exact(&mut payload).unwrap();
            if &payload != b"server data" {
                return Err(Error::General("decrypted server data changed".into()));
            }
            return Ok(());
        }
    }
    Err(Error::General("TLS handshake did not complete".into()))
}

fn profile(
    mut provider: CryptoProvider,
    suite: rustls::CipherSuite,
    group: rustls::NamedGroup,
) -> CryptoProvider {
    provider
        .cipher_suites
        .retain(|candidate| candidate.suite() == suite);
    provider
        .kx_groups
        .retain(|candidate| candidate.name() == group);
    assert_eq!(provider.cipher_suites.len(), 1);
    assert_eq!(provider.kx_groups.len(), 1);
    provider
}

fn interoperability(tls13: bool) {
    let ca = Authority::new();
    for algorithm in [
        &rcgen::PKCS_ECDSA_P256_SHA256,
        &rcgen::PKCS_ECDSA_P384_SHA384,
        &rcgen::PKCS_ED25519,
    ] {
        let leaf = ca.leaf(algorithm, false);
        for suite in super::super::ALL_CIPHER_SUITES {
            if suite.tls13().is_some() != tls13 {
                continue;
            }
            if !tls13 && !format!("{:?}", suite.suite()).contains("ECDSA") {
                continue;
            }
            for group in super::super::kx::ALL_KX_GROUPS {
                let our = || profile(super::super::provider(), suite.suite(), group.name());
                let other = || {
                    profile(
                        rustls::crypto::ring::default_provider(),
                        suite.suite(),
                        group.name(),
                    )
                };
                let (server, client) = configs(our(), other(), &ca, &leaf, true);
                handshake(server, client, "localhost").unwrap();
                let (server, client) = configs(other(), our(), &ca, &leaf, true);
                handshake(server, client, "localhost").unwrap();
            }
        }
    }
}

// Each TLS 1.3 AEAD, exchange group and allowed local key works against an independent backend.
#[test]
fn tls13_interoperability_and_mutual_authentication() {
    interoperability(true);
}

// TLS 1.2 record and signature adapters preserve the same bidirectional contract.
#[cfg(feature = "tls12")]
#[test]
fn tls12_interoperability_and_mutual_authentication() {
    interoperability(false);
}

// Updating the primitive backend must preserve rustls name, trust and time checks.
#[test]
fn certificate_validation_is_not_bypassed() {
    let ca = Authority::new();
    let leaf = ca.leaf(&rcgen::PKCS_ECDSA_P256_SHA256, false);
    let (server, client) = configs(
        super::super::provider(),
        super::super::provider(),
        &ca,
        &leaf,
        false,
    );
    assert!(matches!(
        handshake(server, client, "wrong.invalid"),
        Err(Error::InvalidCertificate(_))
    ));
    let expired = ca.leaf(&rcgen::PKCS_ECDSA_P256_SHA256, true);
    let (server, client) = configs(
        super::super::provider(),
        super::super::provider(),
        &ca,
        &expired,
        false,
    );
    assert!(matches!(
        handshake(server, client, "localhost"),
        Err(Error::InvalidCertificate(_))
    ));
    let other = Authority::new();
    let (server, _) = configs(
        super::super::provider(),
        super::super::provider(),
        &ca,
        &leaf,
        false,
    );
    let (_, client) = configs(
        super::super::provider(),
        super::super::provider(),
        &other,
        &leaf,
        false,
    );
    assert!(matches!(
        handshake(server, client, "localhost"),
        Err(Error::InvalidCertificate(_))
    ));
}

fn rsa_oracle_key() -> KeyPair {
    use rsa::pkcs8::EncodePrivateKey;
    let key = rsa::RsaPrivateKey::new(
        &mut getrandom::rand_core::UnwrapErr(getrandom::SysRng),
        2048,
    )
    .unwrap();
    let der = rustls::pki_types::PrivatePkcs8KeyDer::from(
        key.to_pkcs8_der().unwrap().as_bytes().to_vec(),
    );
    KeyPair::from_pkcs8_der_and_sign_algo(&der, &rcgen::PKCS_RSA_SHA256).unwrap()
}

// RSA issuer verification remains supported even though every local identity uses EC or Ed25519.
#[test]
fn rsa_issuer_with_ecdsa_identities_works() {
    let key = rsa_oracle_key();
    let mut params = CertificateParams::new(Vec::new()).unwrap();
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let root = params.self_signed(&key).unwrap().der().clone();
    let ca = Authority {
        issuer: rcgen::Issuer::new(params, key),
        root,
    };
    let leaf = ca.leaf(&rcgen::PKCS_ECDSA_P256_SHA256, false);
    let (server, client) = configs(
        super::super::provider(),
        super::super::provider(),
        &ca,
        &leaf,
        true,
    );
    handshake(server, client, "localhost").unwrap();
}

// A client can verify an external TLS 1.2 RSA server without acquiring a local RSA signing path.
#[cfg(feature = "tls12")]
#[test]
fn rsa_tls12_servers_remain_verifiable() {
    let ca = Authority::new();
    let key = rsa_oracle_key();
    let mut params = CertificateParams::new(vec!["localhost".into()]).unwrap();
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let certificate = params.signed_by(&key, &ca.issuer).unwrap();
    let leaf = (vec![certificate.der().clone()], key.into());
    for suite in [
        rustls::CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        rustls::CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
        rustls::CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
    ] {
        let server_provider = profile(
            rustls::crypto::ring::default_provider(),
            suite,
            rustls::NamedGroup::X25519,
        );
        let client_provider = profile(super::super::provider(), suite, rustls::NamedGroup::X25519);
        let (server, client) = configs(server_provider, client_provider, &ca, &leaf, false);
        handshake(server, client, "localhost").unwrap();
    }
}
