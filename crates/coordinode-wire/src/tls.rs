//! TLS crypto provider selection and PEM loading.
//!
//! The crypto provider is chosen once at startup: the stock server picks
//! the pure-Rust [`rustls_rustcrypto`] one (no C FFI), and a downstream
//! distribution may select another, `aws-lc-rs` being the reason the seam
//! exists. [`install_crypto_provider`] also makes the selection rustls'
//! process default, so a dependency that builds its own TLS configuration
//! (tonic's client-facing TLS) agrees with the inter-node wire.

use std::sync::{Arc, OnceLock};

use rustls::pki_types::pem::{Error as PemError, PemObject};

/// Re-exported so a caller naming a provider is guaranteed to be naming this
/// crate's rustls, not a second copy resolved elsewhere in the tree.
pub use rustls::crypto::CryptoProvider;
/// Re-exported for the same reason: what [`load_certs`] and
/// [`load_private_key`] return.
pub use rustls::pki_types::{CertificateDer, PrivateKeyDer};

/// Errors loading PEM certificate material.
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
}

/// The pure-Rust crypto provider: RustCrypto algorithms, zero C FFI.
///
/// What the stock server selects, and the fallback whenever nothing was
/// selected.
#[must_use]
pub fn rustcrypto_provider() -> Arc<CryptoProvider> {
    Arc::new(rustls_rustcrypto::provider())
}

/// The provider selected for this process.
static SELECTED: OnceLock<Arc<CryptoProvider>> = OnceLock::new();

/// Select the crypto provider for this process and install it as rustls'
/// default, so dependencies that build their own configs agree with the wire.
///
/// Call once at startup, before any TLS config is built. First call wins: a
/// later one is ignored and returns `false`, because changing the provider once
/// connections exist would leave two halves of the process disagreeing about
/// what was negotiated.
pub fn install_crypto_provider(provider: Arc<CryptoProvider>) -> bool {
    if SELECTED.set(Arc::clone(&provider)).is_err() {
        return false;
    }
    let _ = provider.as_ref().clone().install_default();
    true
}

/// The selected provider, falling back to the pure-Rust one when startup
/// selected nothing (an embedding host or a test building a context
/// directly, say).
#[must_use]
pub fn selected_provider() -> Arc<CryptoProvider> {
    SELECTED.get_or_init(rustcrypto_provider).clone()
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

/// A verifier of ordinary client certificates issued by `roots`, on the
/// selected provider, for a port shared with clients. With `mandatory`
/// false a client may also connect without one.
///
/// # Errors
///
/// A root is not a usable trust anchor, or none is given.
pub fn client_verifier(
    roots: Vec<CertificateDer<'static>>,
    mandatory: bool,
) -> Result<Arc<dyn rustls::server::danger::ClientCertVerifier>, String> {
    let mut store = rustls::RootCertStore::empty();
    for root in roots {
        store.add(root).map_err(|e| e.to_string())?;
    }
    let builder = rustls::server::WebPkiClientVerifier::builder_with_provider(
        Arc::new(store),
        selected_provider(),
    );
    let builder = if mandatory {
        builder
    } else {
        builder.allow_unauthenticated()
    };
    builder.build().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tls_tests;
