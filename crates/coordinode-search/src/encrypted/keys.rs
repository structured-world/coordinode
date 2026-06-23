//! Cryptographic key types for SSE.
//!
//! Each encrypted field uses a key pair:
//! - `FieldKey` (32 bytes) — AES-256-GCM encryption/decryption of values
//! - `SearchKey` (32 bytes) — HMAC-SHA256 search token generation
//!
//! Keys are generated randomly and must be stored securely (KMS, env config).
//! In CE, key management is the caller's responsibility.
//! In EE, KMS integration (Vault, AWS KMS, etc.) wraps these keys.

use rand::Rng;

/// AES-256-GCM key for field value encryption/decryption.
///
/// 32 bytes (256 bits). Used client-side only — the server never holds this key.
#[derive(Clone)]
pub struct FieldKey {
    bytes: [u8; 32],
}

impl FieldKey {
    /// Generate a random field key using the system CSPRNG.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Create a field key from raw bytes.
    ///
    /// Returns `None` if the slice is not exactly 32 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 32 {
            return None;
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Some(Self { bytes: key })
    }

    /// Raw key bytes. Handle with care — do not log or expose.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

impl Drop for FieldKey {
    fn drop(&mut self) {
        // Zeroize key material on drop to prevent memory leaks
        self.bytes.fill(0);
    }
}

// Intentionally no Debug/Display to prevent accidental key logging
impl std::fmt::Debug for FieldKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FieldKey([REDACTED])")
    }
}

/// HMAC-SHA256 key for search token generation.
///
/// 32 bytes (256 bits). Used client-side to generate deterministic tokens.
/// The server holds tokens (not the key) and compares them.
#[derive(Clone)]
pub struct SearchKey {
    bytes: [u8; 32],
}

impl SearchKey {
    /// Generate a random search key using the system CSPRNG.
    pub fn generate() -> Self {
        let mut bytes = [0u8; 32];
        rand::rng().fill_bytes(&mut bytes);
        Self { bytes }
    }

    /// Create a search key from raw bytes.
    ///
    /// Returns `None` if the slice is not exactly 32 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != 32 {
            return None;
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Some(Self { bytes: key })
    }

    /// Raw key bytes. Handle with care — do not log or expose.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.bytes
    }
}

impl Drop for SearchKey {
    fn drop(&mut self) {
        self.bytes.fill(0);
    }
}

impl std::fmt::Debug for SearchKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SearchKey([REDACTED])")
    }
}

/// A paired field_key + search_key for a single encrypted field.
///
/// Each encrypted property on a node label uses its own key pair.
/// Compromising one pair does not expose other fields.
#[derive(Debug, Clone)]
pub struct KeyPair {
    /// AES-256-GCM encryption key.
    pub field_key: FieldKey,
    /// HMAC-SHA256 search token key.
    pub search_key: SearchKey,
}

impl KeyPair {
    /// Generate a new random key pair.
    pub fn generate() -> Self {
        Self {
            field_key: FieldKey::generate(),
            search_key: SearchKey::generate(),
        }
    }

    /// Create from raw bytes (32 + 32 = 64 bytes).
    pub fn from_bytes(field_key_bytes: &[u8], search_key_bytes: &[u8]) -> Option<Self> {
        Some(Self {
            field_key: FieldKey::from_bytes(field_key_bytes)?,
            search_key: SearchKey::from_bytes(search_key_bytes)?,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
