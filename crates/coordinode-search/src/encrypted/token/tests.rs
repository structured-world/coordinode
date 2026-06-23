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
