//! Searchable Symmetric Encryption (SSE) for equality queries on encrypted fields.
//!
//! Implements the Song-Wagner-Perrig 2000 base scheme (ADR-012 patent constraint):
//! - **AES-256-GCM** for per-field value encryption (client-side)
//! - **HMAC-SHA256** for search token generation (deterministic equality matching)
//! - Server compares tokens without seeing plaintext
//!
//! # Write Path
//! ```text
//! Client: value = "alice@example.com"
//!   → ciphertext = AES-256-GCM(value, field_key, random_nonce)
//!   → token = HMAC-SHA256(value, search_key)
//!   → store: { encrypted_value: ciphertext, search_token: token }
//! ```
//!
//! # Search Path
//! ```text
//! Client: query = "alice@example.com"
//!   → query_token = HMAC-SHA256(query, search_key)
//!   → Server: compare query_token against stored tokens
//!   → Return matching ciphertext
//!   → Client: decrypt with field_key
//! ```
//!
//! # Security Properties
//! - Server never sees plaintext values
//! - Deterministic tokens enable equality matching
//! - Per-field key pairs (field_key for encryption, search_key for tokens)
//! - SSE leaks access patterns (which documents match a query)
//!
//! # Patent Safety (ADR-012)
//! - Song-Wagner-Perrig 2000 base scheme only
//! - No dynamic-update techniques from US 8,533,489
//! - Simple HMAC token store + lookup, no fancy index structures

mod field;
mod index;
mod keys;
mod stemmed;
mod storage;
mod token;

pub use field::{EncryptedField, SseError, decrypt_field, encrypt_field};
pub use index::EncryptedFieldIndex;
pub use keys::{FieldKey, KeyPair, SearchKey};
pub use stemmed::{StemToken, stem_and_tokenize, stem_and_tokenize_auto, stem_query_token};
pub use storage::EncryptedIndex;
pub use token::{SEARCH_TOKEN_LEN, SearchToken, generate_search_token};
