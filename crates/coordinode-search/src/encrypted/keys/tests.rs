use super::*;

#[test]
fn field_key_generation_unique() {
    let k1 = FieldKey::generate();
    let k2 = FieldKey::generate();
    assert_ne!(
        k1.as_bytes(),
        k2.as_bytes(),
        "two random keys should differ"
    );
}

#[test]
fn search_key_generation_unique() {
    let k1 = SearchKey::generate();
    let k2 = SearchKey::generate();
    assert_ne!(k1.as_bytes(), k2.as_bytes());
}

#[test]
fn field_key_from_bytes_valid() {
    let bytes = [42u8; 32];
    let key = FieldKey::from_bytes(&bytes).unwrap();
    assert_eq!(key.as_bytes(), &bytes);
}

#[test]
fn field_key_from_bytes_wrong_len() {
    assert!(FieldKey::from_bytes(&[0u8; 16]).is_none());
    assert!(FieldKey::from_bytes(&[0u8; 64]).is_none());
    assert!(FieldKey::from_bytes(&[]).is_none());
}

#[test]
fn search_key_from_bytes_valid() {
    let bytes = [7u8; 32];
    let key = SearchKey::from_bytes(&bytes).unwrap();
    assert_eq!(key.as_bytes(), &bytes);
}

#[test]
fn search_key_from_bytes_wrong_len() {
    assert!(SearchKey::from_bytes(&[0u8; 31]).is_none());
}

#[test]
fn key_pair_generate() {
    let pair = KeyPair::generate();
    assert_ne!(pair.field_key.as_bytes(), pair.search_key.as_bytes());
}

#[test]
fn key_pair_from_bytes() {
    let fk = [1u8; 32];
    let sk = [2u8; 32];
    let pair = KeyPair::from_bytes(&fk, &sk).unwrap();
    assert_eq!(pair.field_key.as_bytes(), &fk);
    assert_eq!(pair.search_key.as_bytes(), &sk);
}

#[test]
fn debug_does_not_leak_key_material() {
    let key = FieldKey::generate();
    let debug = format!("{key:?}");
    assert!(debug.contains("REDACTED"));
    assert!(!debug.contains(&format!("{:?}", key.as_bytes())));
}
