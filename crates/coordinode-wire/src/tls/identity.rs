//! Local TLS identity admission, independent of the selected crypto backend.

use std::sync::Arc;

use der::Decode;
use pkcs8::{ObjectIdentifier, PrivateKeyInfoRef};
use rustls::crypto::KeyProvider;
use rustls::pki_types::PrivateKeyDer;
use rustls::sign::SigningKey;
use rustls::{Error, SignatureAlgorithm};

const EC_PUBLIC_KEY: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.2.1");
const ED25519: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.101.112");
const P256: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.2.840.10045.3.1.7");
const P384: ObjectIdentifier = ObjectIdentifier::new_unwrap("1.3.132.0.34");

fn unsupported() -> Error {
    Error::General("local TLS identities require P-256, P-384 or Ed25519 private keys".into())
}

/// Admit the algorithm before the backend can execute a private-key operation.
pub(crate) fn check_identity_key(key: &PrivateKeyDer<'_>) -> Result<(), Error> {
    match key {
        PrivateKeyDer::Pkcs8(key) => {
            let info =
                PrivateKeyInfoRef::from_der(key.secret_pkcs8_der()).map_err(|_| unsupported())?;
            if info.algorithm.oid == ED25519 {
                if info.algorithm.parameters.is_some() {
                    return Err(unsupported());
                }
                return Ok(());
            }
            if info.algorithm.oid != EC_PUBLIC_KEY {
                return Err(unsupported());
            }
            let curve: ObjectIdentifier = info
                .algorithm
                .parameters
                .ok_or_else(unsupported)?
                .decode_as()
                .map_err(|_| unsupported())?;
            let length = curve_width(curve)?;
            let inner = sec1::EcPrivateKey::from_der(info.private_key.as_bytes())
                .map_err(|_| unsupported())?;
            if inner.private_key.len() != length {
                return Err(unsupported());
            }
            Ok(())
        }
        PrivateKeyDer::Sec1(key) => {
            let inner =
                sec1::EcPrivateKey::from_der(key.secret_sec1_der()).map_err(|_| unsupported())?;
            let Some(sec1::EcParameters::NamedCurve(curve)) = inner.parameters else {
                return Err(unsupported());
            };
            if inner.private_key.len() != curve_width(curve)? {
                return Err(unsupported());
            }
            Ok(())
        }
        _ => Err(unsupported()),
    }
}

fn curve_width(curve: ObjectIdentifier) -> Result<usize, Error> {
    // RFC 5480 section 2.1.1.1 names the curves; RFC 5915 section 3 fixes scalar width.
    if curve == P256 {
        return Ok(32);
    }
    if curve == P384 {
        return Ok(48);
    }
    Err(unsupported())
}

/// The host-owned policy wrapper around its separately qualified key backend.
#[derive(Debug)]
pub(crate) struct IdentityKeyProvider {
    inner: &'static dyn KeyProvider,
}

impl IdentityKeyProvider {
    pub(crate) const fn new(inner: &'static dyn KeyProvider) -> Self {
        Self { inner }
    }
}

impl KeyProvider for IdentityKeyProvider {
    fn load_private_key(&self, key: PrivateKeyDer<'static>) -> Result<Arc<dyn SigningKey>, Error> {
        check_identity_key(&key)?;
        let key = self.inner.load_private_key(key)?;
        match key.algorithm() {
            SignatureAlgorithm::ECDSA | SignatureAlgorithm::ED25519 => Ok(key),
            _ => Err(unsupported()),
        }
    }
}
