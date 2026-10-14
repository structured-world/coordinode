//! Certificates for the peer tests: a CA and leaves signed by it.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::sync::Arc;

use rcgen::{
    BasicConstraints, CertificateParams, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyPair,
    KeyUsagePurpose, SanType,
};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

use super::{PeerIdentity, PeerProfile, SecurityContext, SpiffeId, TrustAnchors};

pub(crate) struct Ca {
    pub(crate) der: CertificateDer<'static>,
    issuer: Issuer<'static, KeyPair>,
}

pub(crate) fn ca() -> Ca {
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

/// A chain and key.
pub(crate) type Credentials = (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>);

/// A leaf for `sans` (URIs if they start with `spiffe:`, DNS names
/// otherwise) and its key.
pub(crate) fn leaf(ca: &Ca, sans: &[&str]) -> Credentials {
    leaf_until(ca, sans, None)
}

/// [`leaf`], valid until `not_after` when given.
pub(crate) fn leaf_until(
    ca: &Ca,
    sans: &[&str],
    not_after: Option<std::time::SystemTime>,
) -> Credentials {
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
    if let Some(not_after) = not_after {
        params.not_after = not_after.into();
    }
    let cert = params.signed_by(&key, &ca.issuer).expect("leaf");
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key.serialize_der()));
    (vec![cert.der().clone()], key)
}

pub(crate) fn provider() -> Arc<CryptoProvider> {
    crate::tls::rustcrypto_provider()
}

/// The SPIFFE profile trusting `ca` for `domain`.
pub(crate) fn spiffe(domain: &str, ca: &Ca) -> PeerProfile {
    PeerProfile::Spiffe {
        bundles: HashMap::from([(
            domain.to_owned(),
            TrustAnchors::from_der(std::slice::from_ref(&ca.der)).expect("anchor"),
        )]),
    }
}

/// A node of trust domain `td.test` named `id`.
pub(crate) fn node(ca: &Ca, id: &str) -> Arc<SecurityContext> {
    let (chain, key) = leaf(ca, &[id]);
    Arc::new(SecurityContext::new(provider(), chain, key, spiffe("td.test", ca)).expect("context"))
}

pub(crate) fn sid(s: &str) -> PeerIdentity {
    PeerIdentity::Spiffe(SpiffeId::parse(s).expect("id"))
}

/// A client configuration presenting `identity` if given and accepting any
/// server: the shared-port cases test the server's decision about the
/// client only.
pub(crate) fn client_with_server_name(
    roots: rustls::RootCertStore,
    identity: Option<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)>,
) -> rustls::ClientConfig {
    let builder = rustls::ClientConfig::builder_with_provider(provider())
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

/// A test client's verifier that does not check the server.
#[derive(Debug)]
struct AcceptAnyServer;

impl rustls::client::danger::ServerCertVerifier for AcceptAnyServer {
    fn verify_server_cert(
        &self,
        _: &CertificateDer<'_>,
        _: &[CertificateDer<'_>],
        _: &rustls::pki_types::ServerName<'_>,
        _: &[u8],
        _: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }
    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
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
        dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &provider().signature_verification_algorithms,
        )
    }
    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}
