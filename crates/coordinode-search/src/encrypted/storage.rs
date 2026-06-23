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
mod tests;
