//! AES-256-GCM field encryption and decryption.
//!
//! Client-side encryption: the server stores ciphertext and never sees plaintext.
//! Each encryption uses a random 96-bit nonce (prepended to ciphertext).
//!
//! Wire format: `[12-byte nonce][ciphertext + 16-byte GCM tag]`

use aes_gcm::aead::{Aead, OsRng};
use aes_gcm::{AeadCore, Aes256Gcm, KeyInit, Nonce};

use super::keys::FieldKey;

/// An encrypted field value (nonce + ciphertext + GCM tag).
///
/// The raw bytes can be stored as a BLOB property on a node.
/// Only the client with the corresponding `FieldKey` can decrypt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedField {
    /// Raw encrypted bytes: `[12-byte nonce][ciphertext][16-byte tag]`
    bytes: Vec<u8>,
}

impl EncryptedField {
    /// Create from raw bytes (as stored in the database).
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Raw bytes for storage.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume and return the raw bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

/// Encrypt a plaintext value using AES-256-GCM.
///
/// Returns an `EncryptedField` containing `[nonce || ciphertext || tag]`.
/// Each call produces different ciphertext (random nonce), which is
/// essential for IND-CPA security.
///
/// # Errors
/// Returns error if encryption fails (should not happen with valid key).
pub fn encrypt_field(plaintext: &[u8], key: &FieldKey) -> Result<EncryptedField, SseError> {
    let cipher = Aes256Gcm::new(key.as_bytes().into());
    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext)
        .map_err(|_| SseError::EncryptionFailed)?;

    // Wire format: [12-byte nonce][ciphertext + 16-byte tag]
    let mut result = Vec::with_capacity(12 + ciphertext.len());
    result.extend_from_slice(&nonce);
    result.extend_from_slice(&ciphertext);

    Ok(EncryptedField { bytes: result })
}

/// Decrypt an `EncryptedField` back to plaintext.
///
/// # Errors
/// Returns `SseError::DecryptionFailed` if the key is wrong, the data
/// is corrupted, or the GCM tag verification fails. This is intentionally
/// vague to prevent oracle attacks.
pub fn decrypt_field(encrypted: &EncryptedField, key: &FieldKey) -> Result<Vec<u8>, SseError> {
    let data = encrypted.as_bytes();
    if data.len() < 12 + 16 {
        // Minimum: 12-byte nonce + 16-byte tag (empty plaintext)
        return Err(SseError::DecryptionFailed);
    }

    let nonce = Nonce::from_slice(&data[..12]);
    let ciphertext = &data[12..];

    let cipher = Aes256Gcm::new(key.as_bytes().into());
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| SseError::DecryptionFailed)
}

/// Errors from SSE operations.
#[derive(Debug, thiserror::Error)]
pub enum SseError {
    #[error("encryption failed")]
    EncryptionFailed,

    #[error("decryption failed (wrong key or corrupted data)")]
    DecryptionFailed,

    #[error("token not found in index")]
    TokenNotFound,

    #[error("storage error: {0}")]
    Storage(String),
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
