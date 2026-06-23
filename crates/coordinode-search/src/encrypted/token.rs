//! HMAC-SHA256 search token generation for SSE equality matching.
//!
//! Tokens are deterministic: the same (value, search_key) pair always
//! produces the same token. This enables equality comparison on the
//! server without revealing plaintext values.
//!
//! Song-Wagner-Perrig 2000 base scheme: T_w = HMAC(K, w)

use hmac::digest::KeyInit;
use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::keys::SearchKey;

type HmacSha256 = Hmac<Sha256>;

/// Length of a search token in bytes (SHA-256 output = 32 bytes).
pub const SEARCH_TOKEN_LEN: usize = 32;

/// A search token — the HMAC-SHA256 of a plaintext value under a search key.
///
/// Tokens are stored server-side alongside encrypted values.
/// The server compares tokens without knowing the underlying values.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct SearchToken {
    bytes: [u8; SEARCH_TOKEN_LEN],
}

impl SearchToken {
    /// Create a token from raw bytes.
    ///
    /// Returns `None` if the slice is not exactly 32 bytes.
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != SEARCH_TOKEN_LEN {
            return None;
        }
        let mut token = [0u8; SEARCH_TOKEN_LEN];
        token.copy_from_slice(bytes);
        Some(Self { bytes: token })
    }

    /// Raw token bytes for storage and comparison.
    pub fn as_bytes(&self) -> &[u8; SEARCH_TOKEN_LEN] {
        &self.bytes
    }
}

// Debug shows hex prefix only — tokens are not secret but should not
// clutter logs with full 64-char hex strings.
impl std::fmt::Debug for SearchToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchToken({:02x}{:02x}{:02x}{:02x}...)",
            self.bytes[0], self.bytes[1], self.bytes[2], self.bytes[3]
        )
    }
}

/// Generate a search token for a plaintext value.
///
/// `T_w = HMAC-SHA256(search_key, value)`
///
/// The same (value, key) pair always produces the same token,
/// enabling deterministic equality matching.
pub fn generate_search_token(value: &[u8], key: &SearchKey) -> SearchToken {
    // HMAC-SHA256 accepts any key length via new_from_slice.
    // A 32-byte key is always valid — the Ok branch is the only reachable path.
    let Ok(mut mac) = HmacSha256::new_from_slice(key.as_bytes()) else {
        // Unreachable: HMAC accepts any key length, 32 bytes is always valid.
        // Return a zeroed token as a safe fallback (never happens in practice).
        return SearchToken {
            bytes: [0u8; SEARCH_TOKEN_LEN],
        };
    };
    mac.update(value);
    let result = mac.finalize();
    let bytes: [u8; SEARCH_TOKEN_LEN] = result.into_bytes().into();
    SearchToken { bytes }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
