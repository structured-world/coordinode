//! Storage-backed persistent SSE token index.
//!
//! Stores search tokens in the `Idx` partition using the same key pattern
//! as B-tree indexes: `idx:sse:<label>:<field>:<token_hex>:<node_id>` → `[]`.
//!
//! This enables prefix scanning to find all nodes matching a token,
//! and per-field/per-label index isolation.

use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::Guard;

use super::field::SseError;
use super::token::{SearchToken, SEARCH_TOKEN_LEN};

/// Persistent SSE token index backed by CoordiNode storage.
///
/// Each (label, field) pair has its own namespace in the `Idx` partition.
/// Tokens are stored as hex-encoded keys for safe binary-to-key conversion.
///
/// In clustered mode, writes go through the Raft proposal pipeline
/// and the Idx partition is replicated across nodes.
pub struct EncryptedIndex<'a> {
    engine: &'a StorageEngine,
    /// Label name (e.g., "User") for key prefix scoping.
    label: String,
    /// Field name (e.g., "email") for key prefix scoping.
    field: String,
}

impl<'a> EncryptedIndex<'a> {
    /// Create a persistent SSE index for a specific label + field.
    pub fn new(engine: &'a StorageEngine, label: &str, field: &str) -> Self {
        Self {
            engine,
            label: label.to_string(),
            field: field.to_string(),
        }
    }

    /// Insert a token → node_id mapping.
    ///
    /// Idempotent: re-inserting the same (token, node_id) is a no-op
    /// (overwrites with same empty value).
    pub fn insert(&self, token: &SearchToken, node_id: u64) -> Result<(), SseError> {
        let key = self.entry_key(token, node_id);
        self.engine
            .put(Partition::Idx, &key, &[])
            .map_err(|e| SseError::Storage(e.to_string()))
    }

    /// Remove a specific (token, node_id) mapping.
    pub fn remove(&self, token: &SearchToken, node_id: u64) -> Result<(), SseError> {
        let key = self.entry_key(token, node_id);
        self.engine
            .delete(Partition::Idx, &key)
            .map_err(|e| SseError::Storage(e.to_string()))
    }

    /// Remove all tokens for a given node_id in this (label, field).
    ///
    /// Scans the entire (label, field) prefix and removes entries matching node_id.
    /// Used when deleting or updating a node's encrypted field.
    pub fn remove_node(&self, node_id: u64) -> Result<(), SseError> {
        let prefix = self.field_prefix();
        let node_suffix = format!(":{node_id:016x}");

        let iter = self
            .engine
            .prefix_scan(Partition::Idx, &prefix)
            .map_err(|e| SseError::Storage(e.to_string()))?;

        let mut keys_to_delete = Vec::new();
        for item in iter {
            let (key, _value) = item
                .into_inner()
                .map_err(|e| SseError::Storage(e.to_string()))?;
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if key_str.ends_with(&node_suffix) {
                    keys_to_delete.push(key.to_vec());
                }
            }
        }

        for key in keys_to_delete {
            self.engine
                .delete(Partition::Idx, &key)
                .map_err(|e| SseError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Search: find all node IDs matching a search token.
    ///
    /// Scans entries with prefix `idx:sse:<label>:<field>:<token_hex>:`
    /// and extracts node IDs from the key suffix.
    pub fn search(&self, query_token: &SearchToken) -> Result<Vec<u64>, SseError> {
        let prefix = self.token_prefix(query_token);

        let iter = self
            .engine
            .prefix_scan(Partition::Idx, &prefix)
            .map_err(|e| SseError::Storage(e.to_string()))?;

        let mut node_ids = Vec::new();
        for item in iter {
            let (key, _value) = item
                .into_inner()
                .map_err(|e| SseError::Storage(e.to_string()))?;
            if let Some(node_id) = self.extract_node_id(&key) {
                node_ids.push(node_id);
            }
        }

        Ok(node_ids)
    }

    /// Count of entries for a specific token.
    pub fn count(&self, token: &SearchToken) -> Result<usize, SseError> {
        Ok(self.search(token)?.len())
    }

    /// Build the full entry key: `idx:sse:<label>:<field>:<token_hex>:<node_id_hex>`
    fn entry_key(&self, token: &SearchToken, node_id: u64) -> Vec<u8> {
        let token_hex = hex_encode(token.as_bytes());
        let key = format!(
            "idx:sse:{}:{}:{}:{:016x}",
            self.label, self.field, token_hex, node_id
        );
        key.into_bytes()
    }

    /// Prefix for all entries of this (label, field): `idx:sse:<label>:<field>:`
    fn field_prefix(&self) -> Vec<u8> {
        format!("idx:sse:{}:{}:", self.label, self.field).into_bytes()
    }

    /// Prefix for a specific token: `idx:sse:<label>:<field>:<token_hex>:`
    fn token_prefix(&self, token: &SearchToken) -> Vec<u8> {
        let token_hex = hex_encode(token.as_bytes());
        format!("idx:sse:{}:{}:{}:", self.label, self.field, token_hex).into_bytes()
    }

    /// Extract node_id from the last segment of a key.
    fn extract_node_id(&self, key: &[u8]) -> Option<u64> {
        let key_str = std::str::from_utf8(key).ok()?;
        let last_colon = key_str.rfind(':')?;
        let hex_str = &key_str[last_colon + 1..];
        u64::from_str_radix(hex_str, 16).ok()
    }
}

/// Hex-encode bytes to a lowercase hex string.
fn hex_encode(bytes: &[u8; SEARCH_TOKEN_LEN]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::encrypted::keys::{KeyPair, SearchKey};
    use crate::encrypted::token::generate_search_token;
    use coordinode_storage::engine::config::{
        Durability, EndpointConfig, Media, StorageConfig, Tier,
    };

    fn test_engine(dir: &std::path::Path) -> StorageEngine {
        let config = StorageConfig::with_endpoints(vec![EndpointConfig::new(
            "default",
            dir,
            Media::Hdd,
            Durability::Durable,
            Tier::Warm,
        )]);
        StorageEngine::open(&config).unwrap()
    }

    fn make_token(value: &[u8]) -> SearchToken {
        let key = SearchKey::from_bytes(&[1u8; 32]).unwrap();
        generate_search_token(value, &key)
    }

    #[test]
    fn insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let token = make_token(b"alice@example.com");
        idx.insert(&token, 1).unwrap();

        let results = idx.search(&token).unwrap();
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn search_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let token = make_token(b"nonexistent");
        let results = idx.search(&token).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn multiple_nodes_same_token() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "role");

        let token = make_token(b"admin");
        idx.insert(&token, 1).unwrap();
        idx.insert(&token, 2).unwrap();
        idx.insert(&token, 3).unwrap();

        let results = idx.search(&token).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn remove_specific_entry() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let token = make_token(b"alice@example.com");
        idx.insert(&token, 1).unwrap();
        idx.insert(&token, 2).unwrap();

        idx.remove(&token, 1).unwrap();

        let results = idx.search(&token).unwrap();
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn remove_node_across_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let t1 = make_token(b"alice@example.com");
        let t2 = make_token(b"alice_alt@example.com");

        idx.insert(&t1, 1).unwrap();
        idx.insert(&t2, 1).unwrap();
        idx.insert(&t1, 2).unwrap();

        idx.remove_node(1).unwrap();

        assert!(idx.search(&t1).unwrap().contains(&2));
        assert!(!idx.search(&t1).unwrap().contains(&1));
        assert!(idx.search(&t2).unwrap().is_empty());
    }

    #[test]
    fn different_labels_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());

        let idx_user = EncryptedIndex::new(&engine, "User", "email");
        let idx_admin = EncryptedIndex::new(&engine, "Admin", "email");

        let token = make_token(b"shared@example.com");
        idx_user.insert(&token, 1).unwrap();
        idx_admin.insert(&token, 2).unwrap();

        assert_eq!(idx_user.search(&token).unwrap(), vec![1]);
        assert_eq!(idx_admin.search(&token).unwrap(), vec![2]);
    }

    #[test]
    fn different_fields_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());

        let idx_email = EncryptedIndex::new(&engine, "User", "email");
        let idx_phone = EncryptedIndex::new(&engine, "User", "phone");

        let t_email = make_token(b"alice@example.com");
        let t_phone = make_token(b"+1234567890");

        idx_email.insert(&t_email, 1).unwrap();
        idx_phone.insert(&t_phone, 1).unwrap();

        assert_eq!(idx_email.search(&t_email).unwrap(), vec![1]);
        assert!(idx_email.search(&t_phone).unwrap().is_empty());
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let token = make_token(b"persistent@example.com");

        // Write and close
        {
            let engine = test_engine(dir.path());
            let idx = EncryptedIndex::new(&engine, "User", "email");
            idx.insert(&token, 42).unwrap();
        }

        // Reopen and verify
        {
            let engine = test_engine(dir.path());
            let idx = EncryptedIndex::new(&engine, "User", "email");
            let results = idx.search(&token).unwrap();
            assert_eq!(results, vec![42], "token should survive reopen");
        }
    }

    #[test]
    fn end_to_end_encrypted_field_with_storage() {
        use crate::encrypted::field::{decrypt_field, encrypt_field, EncryptedField};

        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let pair = KeyPair::generate();

        // WRITE: encrypt + tokenize + store
        let plaintext = b"alice@example.com";
        let encrypted = encrypt_field(plaintext, &pair.field_key).unwrap();
        let token = generate_search_token(plaintext, &pair.search_key);

        // Store encrypted value in Node partition
        engine
            .put(Partition::Node, b"node:1:1:email", encrypted.as_bytes())
            .unwrap();

        // Store token in SSE index
        let idx = EncryptedIndex::new(&engine, "User", "email");
        idx.insert(&token, 1).unwrap();

        // SEARCH: generate query token + lookup + decrypt
        let query_token = generate_search_token(b"alice@example.com", &pair.search_key);
        let matching_ids = idx.search(&query_token).unwrap();
        assert_eq!(matching_ids, vec![1]);

        // Retrieve and decrypt
        let stored = engine
            .get(Partition::Node, b"node:1:1:email")
            .unwrap()
            .unwrap();
        let restored = EncryptedField::from_bytes(stored.to_vec());
        let decrypted = decrypt_field(&restored, &pair.field_key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn count_entries() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let token = make_token(b"test");
        idx.insert(&token, 1).unwrap();
        idx.insert(&token, 2).unwrap();

        assert_eq!(idx.count(&token).unwrap(), 2);
    }

    #[test]
    fn insert_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new(&engine, "User", "email");

        let token = make_token(b"test");
        idx.insert(&token, 1).unwrap();
        idx.insert(&token, 1).unwrap(); // duplicate

        assert_eq!(
            idx.count(&token).unwrap(),
            1,
            "duplicate should be idempotent"
        );
    }
}
