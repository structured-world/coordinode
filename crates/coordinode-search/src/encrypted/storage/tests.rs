use super::*;
use crate::encrypted::keys::{KeyPair, SearchKey};
use crate::encrypted::token::generate_search_token;
use coordinode_storage::engine::config::{Durability, EndpointConfig, Media, StorageConfig, Tier};

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
