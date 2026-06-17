//! Storage-backed persistent SSE token index.
//!
//! Stores search tokens in the `Idx` partition using the same key pattern
//! as B-tree indexes: `idx:sse:<label>:<field>:<token_hex>:<node_id>` → `[]`.
//!
//! This enables prefix scanning to find all nodes matching a token,
//! and per-field/per-label index isolation.

use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use super::field::SseError;
use super::token::{SearchToken, SEARCH_TOKEN_LEN};

/// Persistent SSE token index backed by CoordiNode storage.
///
/// Each (label, field) pair has its own namespace in the `Idx` partition.
/// Tokens are stored as hex-encoded keys for safe binary-to-key conversion.
///
/// ## Transaction threading (ADR-041)
///
/// The index is a typed handle scoped to one `(label, field)` pair; it
/// holds no engine reference. Every method threads the active
/// [`Transaction`]: writes ([`Self::insert`] / [`Self::remove`] /
/// [`Self::remove_node`]) take `&mut Transaction` and buffer their
/// `Partition::Idx` mutations on it, so token postings commit
/// atomically with the node write that produced them (and, in
/// clustered mode, replicate through the Raft proposal pipeline that
/// the commit drives). Reads ([`Self::search`] / [`Self::count`]) take
/// `&Transaction` and walk the committed MVCC snapshot via
/// [`Transaction::base_prefix_scan`].
pub struct EncryptedIndex {
    /// Label name (e.g., "User") for key prefix scoping.
    label: String,
    /// Field name (e.g., "email") for key prefix scoping.
    field: String,
}

impl EncryptedIndex {
    /// Create a persistent SSE index handle for a specific label + field.
    pub fn new(label: &str, field: &str) -> Self {
        Self {
            label: label.to_string(),
            field: field.to_string(),
        }
    }

    /// Insert a token → node_id mapping. Buffered on `txn`.
    ///
    /// Idempotent: re-inserting the same (token, node_id) is a no-op
    /// (overwrites with same empty value).
    pub fn insert(
        &self,
        txn: &mut Transaction,
        token: &SearchToken,
        node_id: u64,
    ) -> Result<(), SseError> {
        let key = self.entry_key(token, node_id);
        txn.put(Partition::Idx, &key, &[])
            .map_err(|e| SseError::Storage(e.to_string()))
    }

    /// Remove a specific (token, node_id) mapping. Buffered on `txn`.
    pub fn remove(
        &self,
        txn: &mut Transaction,
        token: &SearchToken,
        node_id: u64,
    ) -> Result<(), SseError> {
        let key = self.entry_key(token, node_id);
        txn.delete(Partition::Idx, &key)
            .map_err(|e| SseError::Storage(e.to_string()))
    }

    /// Remove all tokens for a given node_id in this (label, field).
    ///
    /// Scans the entire (label, field) prefix and buffers a tombstone
    /// for every entry matching node_id on `txn`. Used when deleting or
    /// updating a node's encrypted field.
    pub fn remove_node(&self, txn: &mut Transaction, node_id: u64) -> Result<(), SseError> {
        let prefix = self.field_prefix();
        let node_suffix = format!(":{node_id:016x}");

        let pairs = txn
            .base_prefix_scan(Partition::Idx, &prefix)
            .map_err(|e| SseError::Storage(e.to_string()))?;

        let mut keys_to_delete = Vec::new();
        for (key, _value) in pairs {
            if let Ok(key_str) = std::str::from_utf8(&key) {
                if key_str.ends_with(&node_suffix) {
                    keys_to_delete.push(key);
                }
            }
        }

        for key in keys_to_delete {
            txn.delete(Partition::Idx, &key)
                .map_err(|e| SseError::Storage(e.to_string()))?;
        }

        Ok(())
    }

    /// Search: find all node IDs matching a search token.
    ///
    /// Scans entries with prefix `idx:sse:<label>:<field>:<token_hex>:`
    /// and extracts node IDs from the key suffix.
    pub fn search(
        &self,
        txn: &Transaction,
        query_token: &SearchToken,
    ) -> Result<Vec<u64>, SseError> {
        let prefix = self.token_prefix(query_token);

        let pairs = txn
            .base_prefix_scan(Partition::Idx, &prefix)
            .map_err(|e| SseError::Storage(e.to_string()))?;

        let mut node_ids = Vec::new();
        for (key, _value) in pairs {
            if let Some(node_id) = self.extract_node_id(&key) {
                node_ids.push(node_id);
            }
        }

        Ok(node_ids)
    }

    /// Count of entries for a specific token.
    pub fn count(&self, txn: &Transaction, token: &SearchToken) -> Result<usize, SseError> {
        Ok(self.search(txn, token)?.len())
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

    use coordinode_core::txn::timestamp::{Timestamp, TimestampOracle};
    use coordinode_core::txn::write_concern::WriteConcern;
    use coordinode_storage::engine::core::StorageEngine;
    use coordinode_storage::engine::transaction::{CommitContext, Transaction};

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

    /// Run SSE index writes in one MVCC transaction and commit.
    fn commit_txn(engine: &StorageEngine, body: impl FnOnce(&mut Transaction)) {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let mut txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        body(&mut txn);
        let wc = WriteConcern::majority();
        let ctx = CommitContext {
            write_concern: &wc,
            pipeline: None,
            id_gen: None,
            drain_buffer: None,
            nvme_write_buffer: None,
        };
        txn.commit(&ctx).unwrap();
    }

    /// Run an SSE index read against the latest committed snapshot.
    fn read_txn<R>(engine: &StorageEngine, body: impl FnOnce(&Transaction) -> R) -> R {
        let oracle = TimestampOracle::resume_from(Timestamp::from_raw(1));
        let read_ts = oracle.next();
        let txn = Transaction::new(engine, Some(&oracle), read_ts, Some(engine.snapshot()));
        body(&txn)
    }

    fn make_token(value: &[u8]) -> SearchToken {
        let key = SearchKey::from_bytes(&[1u8; 32]).unwrap();
        generate_search_token(value, &key)
    }

    #[test]
    fn insert_and_search() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "email");

        let token = make_token(b"alice@example.com");
        commit_txn(&engine, |txn| idx.insert(txn, &token, 1).unwrap());

        let results = read_txn(&engine, |txn| idx.search(txn, &token).unwrap());
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn search_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "email");

        let token = make_token(b"nonexistent");
        let results = read_txn(&engine, |txn| idx.search(txn, &token).unwrap());
        assert!(results.is_empty());
    }

    #[test]
    fn multiple_nodes_same_token() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "role");

        let token = make_token(b"admin");
        commit_txn(&engine, |txn| {
            idx.insert(txn, &token, 1).unwrap();
            idx.insert(txn, &token, 2).unwrap();
            idx.insert(txn, &token, 3).unwrap();
        });

        let results = read_txn(&engine, |txn| idx.search(txn, &token).unwrap());
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn remove_specific_entry() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "email");

        let token = make_token(b"alice@example.com");
        commit_txn(&engine, |txn| {
            idx.insert(txn, &token, 1).unwrap();
            idx.insert(txn, &token, 2).unwrap();
        });

        commit_txn(&engine, |txn| idx.remove(txn, &token, 1).unwrap());

        let results = read_txn(&engine, |txn| idx.search(txn, &token).unwrap());
        assert_eq!(results, vec![2]);
    }

    #[test]
    fn remove_node_across_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "email");

        let t1 = make_token(b"alice@example.com");
        let t2 = make_token(b"alice_alt@example.com");

        commit_txn(&engine, |txn| {
            idx.insert(txn, &t1, 1).unwrap();
            idx.insert(txn, &t2, 1).unwrap();
            idx.insert(txn, &t1, 2).unwrap();
        });

        commit_txn(&engine, |txn| idx.remove_node(txn, 1).unwrap());

        assert!(read_txn(&engine, |txn| idx.search(txn, &t1).unwrap()).contains(&2));
        assert!(!read_txn(&engine, |txn| idx.search(txn, &t1).unwrap()).contains(&1));
        assert!(read_txn(&engine, |txn| idx.search(txn, &t2).unwrap()).is_empty());
    }

    #[test]
    fn different_labels_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());

        let idx_user = EncryptedIndex::new("User", "email");
        let idx_admin = EncryptedIndex::new("Admin", "email");

        let token = make_token(b"shared@example.com");
        commit_txn(&engine, |txn| {
            idx_user.insert(txn, &token, 1).unwrap();
            idx_admin.insert(txn, &token, 2).unwrap();
        });

        assert_eq!(
            read_txn(&engine, |txn| idx_user.search(txn, &token).unwrap()),
            vec![1]
        );
        assert_eq!(
            read_txn(&engine, |txn| idx_admin.search(txn, &token).unwrap()),
            vec![2]
        );
    }

    #[test]
    fn different_fields_isolated() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());

        let idx_email = EncryptedIndex::new("User", "email");
        let idx_phone = EncryptedIndex::new("User", "phone");

        let t_email = make_token(b"alice@example.com");
        let t_phone = make_token(b"+1234567890");

        commit_txn(&engine, |txn| {
            idx_email.insert(txn, &t_email, 1).unwrap();
            idx_phone.insert(txn, &t_phone, 1).unwrap();
        });

        assert_eq!(
            read_txn(&engine, |txn| idx_email.search(txn, &t_email).unwrap()),
            vec![1]
        );
        assert!(read_txn(&engine, |txn| idx_email.search(txn, &t_phone).unwrap()).is_empty());
    }

    #[test]
    fn persistence_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let token = make_token(b"persistent@example.com");

        // Write and close
        {
            let engine = test_engine(dir.path());
            let idx = EncryptedIndex::new("User", "email");
            commit_txn(&engine, |txn| idx.insert(txn, &token, 42).unwrap());
        }

        // Reopen and verify
        {
            let engine = test_engine(dir.path());
            let idx = EncryptedIndex::new("User", "email");
            let results = read_txn(&engine, |txn| idx.search(txn, &token).unwrap());
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
        let idx = EncryptedIndex::new("User", "email");
        commit_txn(&engine, |txn| idx.insert(txn, &token, 1).unwrap());

        // SEARCH: generate query token + lookup + decrypt
        let query_token = generate_search_token(b"alice@example.com", &pair.search_key);
        let matching_ids = read_txn(&engine, |txn| idx.search(txn, &query_token).unwrap());
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
        let idx = EncryptedIndex::new("User", "email");

        let token = make_token(b"test");
        commit_txn(&engine, |txn| {
            idx.insert(txn, &token, 1).unwrap();
            idx.insert(txn, &token, 2).unwrap();
        });

        assert_eq!(read_txn(&engine, |txn| idx.count(txn, &token).unwrap()), 2);
    }

    #[test]
    fn insert_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let engine = test_engine(dir.path());
        let idx = EncryptedIndex::new("User", "email");

        let token = make_token(b"test");
        commit_txn(&engine, |txn| {
            idx.insert(txn, &token, 1).unwrap();
            idx.insert(txn, &token, 1).unwrap(); // duplicate
        });

        assert_eq!(
            read_txn(&engine, |txn| idx.count(txn, &token).unwrap()),
            1,
            "duplicate should be idempotent"
        );
    }
}
