//! Integration tests: SSE encrypted search via Cypher (G017).
//!
//! Tests CREATE/DROP ENCRYPTED INDEX DDL and encrypted_match() function
//! through the full Cypher pipeline: parse → plan → execute.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use coordinode_embed::Database;
use coordinode_search::encrypted::{generate_search_token, KeyPair};
use coordinode_storage::engine::partition::Partition;

fn open_db() -> (Database, tempfile::TempDir) {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = Database::open(dir.path()).expect("open db");
    (db, dir)
}

/// CREATE ENCRYPTED INDEX parses and executes without error.
#[test]
fn create_encrypted_index_ddl() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE ENCRYPTED INDEX idx_ssn ON :Patient(ssn)")
        .expect("CREATE ENCRYPTED INDEX should succeed");

    // Verify index metadata persisted in schema partition.
    // Key format: encrypted_index:{name}
    let key = b"encrypted_index:idx_ssn";
    let val = db.engine().get(Partition::Schema, key).expect("get");
    assert!(
        val.is_some(),
        "encrypted index metadata should be persisted"
    );
}

/// DROP ENCRYPTED INDEX removes the metadata.
#[test]
fn drop_encrypted_index_ddl() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE ENCRYPTED INDEX idx_email ON :User(email)")
        .expect("create");
    db.execute_cypher("DROP ENCRYPTED INDEX idx_email")
        .expect("drop");

    let key = b"encrypted_index:idx_email";
    let val = db.engine().get(Partition::Schema, key).expect("get");
    assert!(val.is_none(), "dropped index metadata should be gone");
}

/// E2E: insert encrypted data with token, then search via encrypted_match().
#[test]
fn encrypted_match_e2e_search() {
    let (mut db, _dir) = open_db();

    // Create encrypted index
    db.execute_cypher("CREATE ENCRYPTED INDEX idx_ssn ON :Patient(ssn)")
        .expect("create index");

    // Generate keys and tokens (client-side)
    let keys = KeyPair::generate();
    let token_alice = generate_search_token(b"123-45-6789", &keys.search_key);
    let token_bob = generate_search_token(b"987-65-4321", &keys.search_key);

    // Create patients
    db.execute_cypher("CREATE (p:Patient {name: 'Alice', ssn: 'encrypted_alice'})")
        .expect("create alice");
    db.execute_cypher("CREATE (p:Patient {name: 'Bob', ssn: 'encrypted_bob'})")
        .expect("create bob");

    // Get node IDs first (needs &mut db for execute_cypher)
    let alice_rows = db
        .execute_cypher("MATCH (p:Patient {name: 'Alice'}) RETURN p")
        .expect("find alice");
    let bob_rows = db
        .execute_cypher("MATCH (p:Patient {name: 'Bob'}) RETURN p")
        .expect("find bob");

    let alice_id = match alice_rows[0].get("p") {
        Some(coordinode_core::graph::types::Value::Int(id)) => *id as u64,
        other => panic!("expected Int for alice id, got {other:?}"),
    };
    let bob_id = match bob_rows[0].get("p") {
        Some(coordinode_core::graph::types::Value::Int(id)) => *id as u64,
        other => panic!("expected Int for bob id, got {other:?}"),
    };

    // Insert tokens into SSE index (server-side, via programmatic API)
    {
        let index =
            coordinode_search::encrypted::EncryptedIndex::new(db.engine(), "Patient", "ssn");
        index
            .insert(&token_alice, alice_id)
            .expect("insert alice token");
        index.insert(&token_bob, bob_id).expect("insert bob token");
    }

    // Search via encrypted_match() — pass token as hex-encoded string parameter
    let token_hex = hex_encode(token_alice.as_bytes());
    let results = db
        .execute_cypher(&format!(
            "MATCH (p:Patient) WHERE encrypted_match(p.ssn, '{token_hex}') RETURN p.name AS name"
        ))
        .expect("encrypted_match query");

    assert_eq!(results.len(), 1, "should find exactly Alice");
    assert_eq!(
        results[0].get("name"),
        Some(&coordinode_core::graph::types::Value::String(
            "Alice".into()
        ))
    );
}

/// encrypted_match with non-matching token returns empty.
#[test]
fn encrypted_match_no_results() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE ENCRYPTED INDEX idx_email ON :User(email)")
        .expect("create index");
    db.execute_cypher("CREATE (u:User {name: 'Test', email: 'encrypted_data'})")
        .expect("create user");

    // Search with a random token that was never inserted
    let keys = KeyPair::generate();
    let random_token = generate_search_token(b"nonexistent@email.com", &keys.search_key);
    let token_hex = hex_encode(random_token.as_bytes());

    let results = db
        .execute_cypher(&format!(
            "MATCH (u:User) WHERE encrypted_match(u.email, '{token_hex}') RETURN u.name AS name"
        ))
        .expect("encrypted_match with no results");

    assert!(results.is_empty(), "no matching token → empty results");
}

/// encrypted_match combined with property filter in compound WHERE.
#[test]
fn encrypted_match_compound_where() {
    let (mut db, _dir) = open_db();

    db.execute_cypher("CREATE ENCRYPTED INDEX idx_ssn ON :Patient(ssn)")
        .expect("create index");

    let keys = KeyPair::generate();
    let token = generate_search_token(b"111-22-3333", &keys.search_key);

    // Create two patients with the same SSN token but different active status
    db.execute_cypher("CREATE (p:Patient {name: 'Active', ssn: 'enc1', active: true})")
        .expect("create active");
    db.execute_cypher("CREATE (p:Patient {name: 'Inactive', ssn: 'enc2', active: false})")
        .expect("create inactive");

    // Get IDs and insert same token for both
    let active_rows = db
        .execute_cypher("MATCH (p:Patient {name: 'Active'}) RETURN p")
        .expect("find active");
    let inactive_rows = db
        .execute_cypher("MATCH (p:Patient {name: 'Inactive'}) RETURN p")
        .expect("find inactive");

    let active_id = match active_rows[0].get("p") {
        Some(coordinode_core::graph::types::Value::Int(id)) => *id as u64,
        other => panic!("expected Int, got {other:?}"),
    };
    let inactive_id = match inactive_rows[0].get("p") {
        Some(coordinode_core::graph::types::Value::Int(id)) => *id as u64,
        other => panic!("expected Int, got {other:?}"),
    };

    {
        let index =
            coordinode_search::encrypted::EncryptedIndex::new(db.engine(), "Patient", "ssn");
        index.insert(&token, active_id).expect("insert active");
        index.insert(&token, inactive_id).expect("insert inactive");
    }

    // Compound WHERE: encrypted_match AND property filter
    let token_hex = hex_encode(token.as_bytes());
    let results = db
        .execute_cypher(&format!(
            "MATCH (p:Patient) \
             WHERE encrypted_match(p.ssn, '{token_hex}') AND p.active = true \
             RETURN p.name AS name"
        ))
        .expect("compound where");

    assert_eq!(results.len(), 1, "should find only the active patient");
    assert_eq!(
        results[0].get("name"),
        Some(&coordinode_core::graph::types::Value::String(
            "Active".into()
        ))
    );
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
