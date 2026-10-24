//! Bounded RSA public-key decoding shared by TLS and token verification.

use crypto_bigint::BoxedUint;
use der::Decode;
use pki_types::InvalidSignature;
use rsa::RsaPublicKey;

/// Decode a PKCS#1 RSA public key without allocating unbounded integers.
///
/// # Errors
/// Rejects malformed DER and keys outside the supported 2048..=8192-bit range.
pub fn rsa_from_pkcs1(bytes: &[u8]) -> Result<RsaPublicKey, InvalidSignature> {
    let key = rsa::pkcs1::RsaPublicKeyRef::from_der(bytes).map_err(|_| InvalidSignature)?;
    rsa_from_components(key.modulus.as_bytes(), key.public_exponent.as_bytes())
}

/// Decode unsigned, minimal big-endian RSA public components.
///
/// RFC 7518 sections 3.3 and 6.3.1 require a modulus of at least 2048 bits
/// and minimal unsigned encodings. The 8192-bit upper bound matches the
/// RustCrypto public-key implementation; exponent bounds are checked before
/// allocating or constructing modular arithmetic state.
///
/// # Errors
/// Rejects noncanonical, out-of-range or otherwise invalid public keys.
pub fn rsa_from_components(n: &[u8], e: &[u8]) -> Result<RsaPublicKey, InvalidSignature> {
    if n.len() < 256 || n.len() > 1024 || e.is_empty() || e.len() > 8 {
        return Err(InvalidSignature);
    }
    if n[0] == 0 || e[0] == 0 {
        return Err(InvalidSignature);
    }
    let n = BoxedUint::from_be_slice_vartime(n);
    if n.bits() < 2048 {
        return Err(InvalidSignature);
    }
    let e = BoxedUint::from_be_slice_vartime(e);
    RsaPublicKey::new(n, e).map_err(|_| InvalidSignature)
}
