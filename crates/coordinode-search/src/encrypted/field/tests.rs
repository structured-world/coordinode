use super::*;

#[test]
fn encrypt_decrypt_roundtrip() {
    let key = FieldKey::generate();
    let plaintext = b"alice@example.com";

    let encrypted = encrypt_field(plaintext, &key).unwrap();
    let decrypted = decrypt_field(&encrypted, &key).unwrap();

    assert_eq!(decrypted, plaintext);
}

#[test]
fn encrypt_produces_different_ciphertext_each_time() {
    let key = FieldKey::generate();
    let plaintext = b"same value";

    let enc1 = encrypt_field(plaintext, &key).unwrap();
    let enc2 = encrypt_field(plaintext, &key).unwrap();

    // Different nonces → different ciphertext (IND-CPA)
    assert_ne!(enc1.as_bytes(), enc2.as_bytes());
}

#[test]
fn wrong_key_fails_decryption() {
    let key1 = FieldKey::generate();
    let key2 = FieldKey::generate();
    let plaintext = b"secret data";

    let encrypted = encrypt_field(plaintext, &key1).unwrap();
    let result = decrypt_field(&encrypted, &key2);

    assert!(result.is_err(), "wrong key should fail decryption");
}

#[test]
fn corrupted_data_fails_decryption() {
    let key = FieldKey::generate();
    let plaintext = b"test";

    let mut encrypted = encrypt_field(plaintext, &key).unwrap();
    // Corrupt a byte in the ciphertext
    let bytes = &mut encrypted.bytes;
    if bytes.len() > 20 {
        bytes[20] ^= 0xFF;
    }

    let result = decrypt_field(&encrypted, &key);
    assert!(
        result.is_err(),
        "corrupted data should fail GCM verification"
    );
}

#[test]
fn too_short_data_fails() {
    let key = FieldKey::generate();
    let short = EncryptedField::from_bytes(vec![0u8; 10]);
    assert!(decrypt_field(&short, &key).is_err());
}

#[test]
fn empty_plaintext_roundtrip() {
    let key = FieldKey::generate();
    let encrypted = encrypt_field(b"", &key).unwrap();
    let decrypted = decrypt_field(&encrypted, &key).unwrap();
    assert!(decrypted.is_empty());
}

#[test]
fn large_plaintext_roundtrip() {
    let key = FieldKey::generate();
    let plaintext = vec![0xABu8; 1_000_000]; // 1MB

    let encrypted = encrypt_field(&plaintext, &key).unwrap();
    let decrypted = decrypt_field(&encrypted, &key).unwrap();

    assert_eq!(decrypted, plaintext);
}

#[test]
fn encrypted_field_from_bytes_roundtrip() {
    let key = FieldKey::generate();
    let encrypted = encrypt_field(b"test", &key).unwrap();
    let raw = encrypted.as_bytes().to_vec();

    let restored = EncryptedField::from_bytes(raw);
    let decrypted = decrypt_field(&restored, &key).unwrap();
    assert_eq!(decrypted, b"test");
}
