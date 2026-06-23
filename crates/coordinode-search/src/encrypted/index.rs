//! In-memory SSE token index for equality search.
//!
//! Stores `SearchToken → Vec<node_id>` mappings. On the server side,
//! this index is populated during writes and queried during encrypted
//! equality searches.
//!
//! In production, tokens are persisted in CoordiNode storage (sse: column family).
//! This module provides the in-memory index logic; storage integration
//! happens at the storage layer.
//!
//! Song-Wagner-Perrig 2000: simple lookup table, no complex data structures.

use std::collections::HashMap;

use super::field::SseError;
use super::token::SearchToken;

/// In-memory SSE token index.
///
/// Maps search tokens to sets of node IDs. Thread-safe access is the
/// caller's responsibility (use behind `RwLock` if shared).
///
/// In clustered mode, this index is replicated via Raft — all writes
/// go through the Raft proposal pipeline, and the index is rebuilt
/// from the Raft log on follower nodes.
pub struct EncryptedFieldIndex {
    /// Token → node IDs mapping.
    /// Multiple nodes can have the same encrypted value (same token).
    entries: HashMap<SearchToken, Vec<u64>>,
}

impl EncryptedFieldIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Insert a token → node_id mapping.
    ///
    /// Idempotent: if the (token, node_id) pair already exists, this is a no-op.
    pub fn insert(&mut self, token: SearchToken, node_id: u64) {
        let ids = self.entries.entry(token).or_default();
        if !ids.contains(&node_id) {
            ids.push(node_id);
        }
    }

    /// Remove a node_id from a token's mapping.
    ///
    /// Used when updating or deleting an encrypted field.
    /// If the token has no more node_ids, the entry is removed.
    pub fn remove(&mut self, token: &SearchToken, node_id: u64) {
        if let Some(ids) = self.entries.get_mut(token) {
            ids.retain(|&id| id != node_id);
            if ids.is_empty() {
                self.entries.remove(token);
            }
        }
    }

    /// Remove all entries for a node_id across all tokens.
    ///
    /// Used when deleting a node entirely.
    pub fn remove_node(&mut self, node_id: u64) {
        self.entries.retain(|_, ids| {
            ids.retain(|&id| id != node_id);
            !ids.is_empty()
        });
    }

    /// Search: find all node IDs that have a matching token.
    ///
    /// This is the server-side equality comparison:
    /// client sends `query_token = HMAC(search_key, query_value)`,
    /// server looks up `query_token` in the index.
    pub fn search(&self, query_token: &SearchToken) -> Result<Vec<u64>, SseError> {
        match self.entries.get(query_token) {
            Some(ids) => Ok(ids.clone()),
            None => Ok(Vec::new()),
        }
    }

    /// Number of distinct tokens in the index.
    pub fn num_tokens(&self) -> usize {
        self.entries.len()
    }

    /// Total number of (token, node_id) entries.
    pub fn num_entries(&self) -> usize {
        self.entries.values().map(|ids| ids.len()).sum()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all (token, node_ids) entries.
    ///
    /// Useful for persistence: serialize the index to sse: keyspace.
    pub fn iter(&self) -> impl Iterator<Item = (&SearchToken, &[u64])> {
        self.entries.iter().map(|(k, v)| (k, v.as_slice()))
    }
}

impl Default for EncryptedFieldIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
