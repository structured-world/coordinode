//! TLS crypto provider selection and PEM loading.
//!
//! The crypto provider is chosen once at startup: the stock server picks
//! the pure-Rust [`coordinode_crypto`] one (no C FFI), and a downstream
//! distribution may select another, `aws-lc-rs` being the reason the seam
//! exists. [`install_crypto_provider`] also makes the selection rustls'
//! process default, so a dependency that builds its own TLS configuration
//! (tonic's client-facing TLS) agrees with the inter-node wire.

use std::sync::Arc;

use once_cell::sync::OnceCell;

use rustls::pki_types::pem::{Error as PemError, PemObject};

pub(crate) mod identity;

use identity::IdentityKeyProvider;

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
    /// A different provider was already installed or selected for this process.
    #[error("a different TLS crypto provider is already selected for this process")]
    ProviderConflict,
    /// The private key does not satisfy the local identity policy.
    #[error("unsupported local TLS identity: {0}")]
    Identity(rustls::Error),
}

/// The pure-Rust crypto provider: RustCrypto algorithms, zero C FFI.
///
/// What the stock server selects, and the fallback whenever nothing was
/// selected.
#[must_use]
pub fn rustcrypto_provider() -> Arc<CryptoProvider> {
    Arc::new(coordinode_crypto::provider())
}

/// The provider selected for this process.
static SELECTED: OnceCell<InstalledProvider> = OnceCell::new();
static IDENTITY_POLICY: OnceCell<IdentityKeyProvider> = OnceCell::new();

struct InstalledProvider {
    requested: Arc<CryptoProvider>,
    effective: Arc<CryptoProvider>,
}

fn install(provider: Arc<CryptoProvider>) -> Result<InstalledProvider, TlsError> {
    if CryptoProvider::get_default().is_some() {
        return Err(TlsError::ProviderConflict);
    }
    let mut effective = provider.as_ref().clone();
    effective.key_provider =
        IDENTITY_POLICY.get_or_init(|| IdentityKeyProvider::new(provider.key_provider));
    effective
        .install_default()
        .map_err(|_| TlsError::ProviderConflict)?;
    let effective = Arc::clone(CryptoProvider::get_default().ok_or(TlsError::ProviderConflict)?);
    Ok(InstalledProvider {
        requested: provider,
        effective,
    })
}

/// Select the crypto provider for this process and install it as rustls'
/// default, so dependencies that build their own configs agree with the wire.
///
/// Call before any TLS config is built. Reusing the same selection is idempotent;
/// selecting a different provider or finding an independently installed default
/// is an error. The effective provider enforces the local identity key policy.
///
/// # Errors
/// Returns [`TlsError::ProviderConflict`] for incompatible process initialization.
pub fn install_crypto_provider(provider: Arc<CryptoProvider>) -> Result<(), TlsError> {
    let selected = SELECTED.get_or_try_init(|| install(Arc::clone(&provider)))?;
    if !Arc::ptr_eq(&selected.requested, &provider) {
        return Err(TlsError::ProviderConflict);
    }
    Ok(())
}

/// The selected provider, falling back to the pure-Rust one when startup
/// selected nothing (an embedding host or a test building a context
/// directly, say).
/// # Errors
/// Returns [`TlsError::ProviderConflict`] if another component installed a default first.
pub fn selected_provider() -> Result<Arc<CryptoProvider>, TlsError> {
    let selected = SELECTED.get_or_try_init(|| install(rustcrypto_provider()))?;
    Ok(Arc::clone(&selected.effective))
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

/// Parse an allowed ECDSA or Ed25519 PEM private key into DER.
///
/// # Errors
/// [`TlsError::Pem`] on a decode error, [`TlsError::NoKey`] if none is present.
pub fn load_private_key(pem: &[u8]) -> Result<PrivateKeyDer<'static>, TlsError> {
    let key = PrivateKeyDer::from_pem_slice(pem).map_err(|source| match source {
        PemError::NoItemsFound => TlsError::NoKey,
        source => TlsError::Pem {
            what: "private key",
            source,
        },
    })?;
    identity::check_identity_key(&key).map_err(TlsError::Identity)?;
    Ok(key)
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
        selected_provider().map_err(|e| e.to_string())?,
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
