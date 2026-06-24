#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair};
use std::sync::Arc;

/// Generate a self-signed CA (cert PEM, the cert, its key).
fn gen_ca() -> (String, rcgen::Certificate, KeyPair) {
    let key = KeyPair::generate().expect("ca key");
    let mut params = CertificateParams::new(Vec::new()).expect("ca params");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let cert = params.self_signed(&key).expect("ca self-sign");
    (cert.pem(), cert, key)
}

/// Issue a leaf cert signed by `ca` for `san`, with the given EKU. Returns
/// (cert PEM, key PEM).
fn issue(
    ca_cert: &rcgen::Certificate,
    ca_key: &KeyPair,
    san: &str,
    eku: ExtendedKeyUsagePurpose,
) -> (String, String) {
    let key = KeyPair::generate().expect("leaf key");
    let mut params = CertificateParams::new(vec![san.to_string()]).expect("leaf params");
    params.extended_key_usages = vec![eku];
    let cert = params.signed_by(&key, ca_cert, ca_key).expect("leaf sign");
    (cert.pem(), key.serialize_pem())
}

async fn handshake(server: ServerConfig, client: ClientConfig) -> Result<(), String> {
    use rustls::pki_types::ServerName;
    use tokio_rustls::{TlsAcceptor, TlsConnector};

    let acceptor = TlsAcceptor::from(Arc::new(server));
    let connector = TlsConnector::from(Arc::new(client));
    let (client_io, server_io) = tokio::io::duplex(16 * 1024);
    let domain = ServerName::try_from("localhost").map_err(|e| e.to_string())?;

    // Drive both halves concurrently on this task so the handshake round-trips
    // (a spawned server on the current-thread test runtime can starve).
    let (client_res, server_res) = tokio::join!(
        connector.connect(domain, client_io),
        acceptor.accept(server_io),
    );
    // Surface the server-side error first — a handshake failure on the server
    // shows up as a generic "broken pipe" on the client once the server drops.
    server_res.map_err(|e| e.to_string())?;
    client_res.map_err(|e| e.to_string())?;
    Ok(())
}

#[tokio::test]
async fn server_cert_tls_handshake_completes() {
    // CA-signed server cert; client trusts the CA. Validates the pure-Rust
    // (rustls-rustcrypto) provider performs a real TLS handshake.
    let (ca_pem, ca_cert, ca_key) = gen_ca();
    let (srv_cert, srv_key) = issue(
        &ca_cert,
        &ca_key,
        "localhost",
        ExtendedKeyUsagePurpose::ServerAuth,
    );

    let server = server_config(srv_cert.as_bytes(), srv_key.as_bytes(), None).expect("server cfg");
    let client = client_config(ca_pem.as_bytes(), None).expect("client cfg");

    handshake(server, client).await.expect("handshake");
}

#[tokio::test]
async fn mtls_handshake_completes_with_client_cert() {
    let (ca_pem, ca_cert, ca_key) = gen_ca();
    let (srv_cert, srv_key) = issue(
        &ca_cert,
        &ca_key,
        "localhost",
        ExtendedKeyUsagePurpose::ServerAuth,
    );
    let (cli_cert, cli_key) = issue(
        &ca_cert,
        &ca_key,
        "node-2",
        ExtendedKeyUsagePurpose::ClientAuth,
    );

    // Server requires a client cert chaining to the CA; client presents one.
    let server = server_config(
        srv_cert.as_bytes(),
        srv_key.as_bytes(),
        Some(ca_pem.as_bytes()),
    )
    .expect("server cfg");
    let client = client_config(
        ca_pem.as_bytes(),
        Some((cli_cert.as_bytes(), cli_key.as_bytes())),
    )
    .expect("client cfg");

    handshake(server, client).await.expect("mTLS handshake");
}

#[tokio::test]
async fn mtls_rejects_client_without_cert() {
    let (ca_pem, ca_cert, ca_key) = gen_ca();
    let (srv_cert, srv_key) = issue(
        &ca_cert,
        &ca_key,
        "localhost",
        ExtendedKeyUsagePurpose::ServerAuth,
    );

    // Server requires a client cert; client presents none → handshake must fail.
    let server = server_config(
        srv_cert.as_bytes(),
        srv_key.as_bytes(),
        Some(ca_pem.as_bytes()),
    )
    .expect("server cfg");
    let client = client_config(ca_pem.as_bytes(), None).expect("client cfg");

    let result = handshake(server, client).await;
    assert!(
        result.is_err(),
        "mTLS must reject a client with no certificate"
    );
}

#[test]
fn load_certs_and_key_round_trip() {
    let (_ca_pem, ca_cert, ca_key) = gen_ca();
    let (cert_pem, key_pem) = issue(
        &ca_cert,
        &ca_key,
        "localhost",
        ExtendedKeyUsagePurpose::ServerAuth,
    );
    assert_eq!(load_certs(cert_pem.as_bytes()).expect("certs").len(), 1);
    load_private_key(key_pem.as_bytes()).expect("key");
}

#[test]
fn load_private_key_missing_is_error() {
    // A PEM with only a certificate, no key.
    let (_ca_pem, ca_cert, ca_key) = gen_ca();
    let (cert_pem, _key) = issue(
        &ca_cert,
        &ca_key,
        "localhost",
        ExtendedKeyUsagePurpose::ServerAuth,
    );
    assert!(matches!(
        load_private_key(cert_pem.as_bytes()),
        Err(TlsError::NoKey)
    ));
}
