#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use rcgen::{BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, KeyPair};

/// Generate a self-signed CA: an [`rcgen::Issuer`] that owns the CA params
/// + key for signing leaf certs.
fn gen_ca() -> rcgen::Issuer<'static, KeyPair> {
    let key = KeyPair::generate().expect("ca key");
    let mut params = CertificateParams::new(Vec::new()).expect("ca params");
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    rcgen::Issuer::new(params, key)
}

/// Issue a leaf cert signed by `issuer` for `san`. Returns (cert PEM, key PEM).
fn issue(issuer: &rcgen::Issuer<'_, KeyPair>, san: &str) -> (String, String) {
    let key = KeyPair::generate().expect("leaf key");
    let mut params = CertificateParams::new(vec![san.to_string()]).expect("leaf params");
    params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ServerAuth];
    let cert = params.signed_by(&key, issuer).expect("leaf sign");
    (cert.pem(), key.serialize_pem())
}

#[test]
fn load_certs_and_key_round_trip() {
    let ca = gen_ca();
    let (cert_pem, key_pem) = issue(&ca, "localhost");
    assert_eq!(load_certs(cert_pem.as_bytes()).expect("certs").len(), 1);
    load_private_key(key_pem.as_bytes()).expect("key");
}

#[test]
fn load_private_key_missing_is_error() {
    // A PEM with only a certificate, no key.
    let ca = gen_ca();
    let (cert_pem, _key) = issue(&ca, "localhost");
    assert!(matches!(
        load_private_key(cert_pem.as_bytes()),
        Err(TlsError::NoKey)
    ));
}
