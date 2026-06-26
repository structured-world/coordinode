//! Inter-node TLS / mTLS configuration (rustls).
//!
//! Builds rustls server and client configs for the inter-node wire from PEM
//! certificate material, with optional mutual TLS. The crypto provider is
//! pluggable per tier: CE uses the pure-Rust [`rustls_rustcrypto`] provider
//! (no C FFI, ADR-013); EE swaps the process default to `aws-lc-rs`. All builders
//! here pin the provider explicitly via `*_with_provider`, so they are correct
//! regardless of which (if any) process default is installed.
//!
//! Encryption is server-cert TLS on the shared `:7080` listener (covers
//! client-to-server and intra-cluster TLS); mutual TLS is opt-in — pass a client
//! CA to [`server_config`] (server then requires + verifies a peer cert) and a
//! client identity to [`client_config`].

use std::sync::Arc;

use rustls::pki_types::pem::{Error as PemError, PemObject};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use rustls::{ClientConfig, RootCertStore, ServerConfig};

/// Errors building a TLS config from PEM material.
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    /// A PEM certificate or key block failed to parse.
    #[error("parse {what}: {source}")]
    Pem {
        /// What was being parsed (e.g. "certificate", "private key").
        what: &'static str,
        /// Underlying PEM decode error.
        source: PemError,
    },
    /// No private key found in the supplied PEM.
    #[error("no private key in PEM")]
    NoKey,
    /// rustls rejected the config (bad cert/key pairing, etc.).
    #[error("rustls: {0}")]
    Rustls(#[from] rustls::Error),
    /// The mTLS client-certificate verifier could not be built.
    #[error("client verifier: {0}")]
    Verifier(String),
}

/// The CE crypto provider: pure-Rust RustCrypto algorithms, zero C FFI.
fn ce_provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls_rustcrypto::provider())
}

/// Install the CE (pure-Rust) crypto provider as the process default. Idempotent:
/// a no-op if a default is already installed (e.g. EE installed aws-lc-rs first).
/// Call once at startup before building any TLS config that relies on the default.
pub fn install_ce_crypto_provider() {
    let _ = rustls_rustcrypto::provider().install_default();
}

/// Parse one or more PEM certificates into DER.
///
/// # Errors
/// [`TlsError::Pem`] if a block cannot be decoded.
pub fn load_certs(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>, TlsError> {
    CertificateDer::pem_slice_iter(pem)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|source| TlsError::Pem {
            what: "certificate",
            source,
        })
}

/// Parse a single PEM private key (PKCS#8 / PKCS#1 / SEC1) into DER.
///
/// # Errors
/// [`TlsError::Pem`] on a decode error, [`TlsError::NoKey`] if none is present.
pub fn load_private_key(pem: &[u8]) -> Result<PrivateKeyDer<'static>, TlsError> {
    PrivateKeyDer::from_pem_slice(pem).map_err(|source| match source {
        PemError::NoItemsFound => TlsError::NoKey,
        source => TlsError::Pem {
            what: "private key",
            source,
        },
    })
}

/// Build a [`RootCertStore`] from PEM CA certificates.
///
/// # Errors
/// [`TlsError::Pem`] if a CA cert cannot be decoded, [`TlsError::Rustls`] if
/// rustls rejects one.
pub fn root_store(ca_pem: &[u8]) -> Result<RootCertStore, TlsError> {
    let mut roots = RootCertStore::empty();
    for cert in load_certs(ca_pem)? {
        roots.add(cert)?;
    }
    Ok(roots)
}

/// Build a server TLS config: present `cert_pem`/`key_pem`. When `client_ca_pem`
/// is `Some`, require and verify a client certificate against that CA (mTLS);
/// when `None`, accept any client (server-cert TLS only).
///
/// # Errors
/// [`TlsError`] on a PEM parse failure, a cert/key mismatch, or a verifier-build
/// failure.
pub fn server_config(
    cert_pem: &[u8],
    key_pem: &[u8],
    client_ca_pem: Option<&[u8]>,
) -> Result<ServerConfig, TlsError> {
    let certs = load_certs(cert_pem)?;
    let key = load_private_key(key_pem)?;
    let provider = ce_provider();

    let builder = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .map_err(TlsError::Rustls)?;

    let config = match client_ca_pem {
        Some(ca) => {
            let roots = Arc::new(root_store(ca)?);
            let verifier =
                rustls::server::WebPkiClientVerifier::builder_with_provider(roots, provider)
                    .build()
                    .map_err(|e| TlsError::Verifier(e.to_string()))?;
            builder
                .with_client_cert_verifier(verifier)
                .with_single_cert(certs, key)?
        }
        None => builder.with_no_client_auth().with_single_cert(certs, key)?,
    };
    Ok(config)
}

/// Build a client TLS config trusting `root_ca_pem`. When `client_identity` is
/// `Some((cert_pem, key_pem))`, present that certificate for mTLS; when `None`,
/// connect without a client certificate.
///
/// # Errors
/// [`TlsError`] on a PEM parse failure or a cert/key mismatch.
pub fn client_config(
    root_ca_pem: &[u8],
    client_identity: Option<(&[u8], &[u8])>,
) -> Result<ClientConfig, TlsError> {
    let roots = root_store(root_ca_pem)?;
    let provider = ce_provider();

    let builder = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(TlsError::Rustls)?
        .with_root_certificates(roots);

    let config = match client_identity {
        Some((cert_pem, key_pem)) => {
            let certs = load_certs(cert_pem)?;
            let key = load_private_key(key_pem)?;
            builder.with_client_auth_cert(certs, key)?
        }
        None => builder.with_no_client_auth(),
    };
    Ok(config)
}

#[cfg(test)]
mod tls_tests;
