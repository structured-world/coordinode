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

// Admission must preserve every supported identity algorithm before backend key loading.
#[test]
fn local_identity_admission_accepts_supported_algorithms() {
    let provider = rustcrypto_provider();
    let policy = IdentityKeyProvider::new(provider.key_provider);
    for algorithm in [
        &rcgen::PKCS_ECDSA_P256_SHA256,
        &rcgen::PKCS_ECDSA_P384_SHA384,
        &rcgen::PKCS_ED25519,
    ] {
        let key = KeyPair::generate_for(algorithm).expect("identity key");
        let key = load_private_key(key.serialize_pem().as_bytes()).expect("admitted identity");
        rustls::crypto::KeyProvider::load_private_key(&policy, key).expect("backend signing key");
    }
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

// An already installed process provider must not be hidden by a separate wire selection.
#[test]
fn conflicting_process_provider_is_an_error() {
    rustcrypto_provider()
        .as_ref()
        .clone()
        .install_default()
        .unwrap();
    assert!(matches!(
        install_crypto_provider(rustcrypto_provider()),
        Err(TlsError::ProviderConflict)
    ));
    assert!(matches!(
        selected_provider(),
        Err(TlsError::ProviderConflict)
    ));
}

// Reusing one explicit selection is idempotent and every transport sees rustls' actual default.
#[test]
fn selected_and_process_providers_are_identical() {
    let provider = rustcrypto_provider();
    install_crypto_provider(Arc::clone(&provider)).unwrap();
    install_crypto_provider(provider).unwrap();
    let selected = selected_provider().unwrap();
    assert!(Arc::ptr_eq(
        &selected,
        CryptoProvider::get_default().unwrap()
    ));
    assert!(matches!(
        install_crypto_provider(rustcrypto_provider()),
        Err(TlsError::ProviderConflict)
    ));
}
