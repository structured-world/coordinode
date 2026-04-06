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
mod tests {
    use super::*;

    #[test]
    fn token_deterministic() {
        let key = SearchKey::generate();
        let value = b"alice@example.com";

        let t1 = generate_search_token(value, &key);
        let t2 = generate_search_token(value, &key);

        assert_eq!(t1, t2, "same value + key must produce same token");
    }

    #[test]
    fn different_values_different_tokens() {
        let key = SearchKey::generate();

        let t1 = generate_search_token(b"alice@example.com", &key);
        let t2 = generate_search_token(b"bob@example.com", &key);

        assert_ne!(t1, t2, "different values must produce different tokens");
    }

    #[test]
    fn different_keys_different_tokens() {
        let k1 = SearchKey::generate();
        let k2 = SearchKey::generate();
        let value = b"same_value";

        let t1 = generate_search_token(value, &k1);
        let t2 = generate_search_token(value, &k2);

        assert_ne!(t1, t2, "different keys must produce different tokens");
    }

    #[test]
    fn empty_value_produces_valid_token() {
        let key = SearchKey::generate();
        let token = generate_search_token(b"", &key);
        assert_eq!(token.as_bytes().len(), SEARCH_TOKEN_LEN);
    }

    #[test]
    fn token_from_bytes_roundtrip() {
        let key = SearchKey::generate();
        let token = generate_search_token(b"test", &key);
        let raw = token.as_bytes().to_vec();

        let restored = SearchToken::from_bytes(&raw).unwrap();
        assert_eq!(restored, token);
    }

    #[test]
    fn token_from_bytes_wrong_len() {
        assert!(SearchToken::from_bytes(&[0u8; 16]).is_none());
        assert!(SearchToken::from_bytes(&[0u8; 64]).is_none());
    }

    #[test]
    fn token_debug_does_not_show_full_bytes() {
        let key = SearchKey::generate();
        let token = generate_search_token(b"secret", &key);
        let debug = format!("{token:?}");
        assert!(debug.contains("SearchToken("));
        assert!(debug.contains("...)"));
        // Full hex would be 64 chars, debug shows only 8
        assert!(debug.len() < 40);
    }

    #[test]
    fn token_hash_and_eq() {
        use std::collections::HashSet;
        let key = SearchKey::generate();
        let t1 = generate_search_token(b"a", &key);
        let t2 = generate_search_token(b"a", &key);
        let t3 = generate_search_token(b"b", &key);

        let mut set = HashSet::new();
        set.insert(t1.clone());
        assert!(set.contains(&t2), "equal tokens should be in set");
        assert!(!set.contains(&t3), "different token should not be in set");
    }
}
