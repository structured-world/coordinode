#![cfg(feature = "tls12")]
#![allow(clippy::unwrap_used, clippy::panic)]

use rustls::crypto::CryptoProvider;
use rustls::crypto::cipher::{
    AeadKey, InboundOpaqueMessage, OutboundPlainMessage, Tls12AeadAlgorithm,
};
use rustls::{CipherSuite, ContentType, Error, ProtocolVersion, SupportedCipherSuite};

fn algorithm(provider: &CryptoProvider, name: CipherSuite) -> &'static dyn Tls12AeadAlgorithm {
    for suite in &provider.cipher_suites {
        if suite.suite() == name {
            if let SupportedCipherSuite::Tls12(suite) = suite {
                return suite.aead_alg;
            }
        }
    }
    panic!("required TLS 1.2 suite is absent: {name:?}");
}

fn check_record_boundary(suite: CipherSuite) {
    let oracle = algorithm(&rustls::crypto::ring::default_provider(), suite);
    let candidate = algorithm(&coordinode_crypto::provider(), suite);
    let shape = oracle.key_block_shape();
    assert_eq!(shape.enc_key_len, 32);
    let fixed_iv = vec![0; shape.fixed_iv_len];
    let explicit = vec![0; shape.explicit_nonce_len];

    for length in [16_384, 16_385] {
        let plaintext = vec![7; length];
        let encrypted = oracle
            .encrypter(AeadKey::from([1; 32]), &fixed_iv, &explicit)
            .encrypt(
                OutboundPlainMessage {
                    typ: ContentType::ApplicationData,
                    version: ProtocolVersion::TLSv1_2,
                    payload: plaintext.as_slice().into(),
                },
                0,
            )
            .unwrap();
        // Both implementations need independent mutable buffers for decryption.
        let mut oracle_bytes = encrypted.payload.as_ref().to_vec();
        let mut candidate_bytes = encrypted.payload.as_ref().to_vec();
        let oracle_result = oracle
            .decrypter(AeadKey::from([1; 32]), &fixed_iv)
            .decrypt(
                InboundOpaqueMessage::new(encrypted.typ, encrypted.version, &mut oracle_bytes),
                0,
            )
            .map(|message| message.payload.to_vec());
        let result = candidate
            .decrypter(AeadKey::from([1; 32]), &fixed_iv)
            .decrypt(
                InboundOpaqueMessage::new(encrypted.typ, encrypted.version, &mut candidate_bytes),
                0,
            )
            .map(|message| message.payload.to_vec());
        if length == 16_384 {
            assert_eq!(oracle_result, Ok(plaintext.clone()));
            assert_eq!(result, Ok(plaintext));
        } else {
            assert_eq!(oracle_result, Err(Error::PeerSentOversizedRecord));
            assert!(matches!(result, Err(Error::PeerSentOversizedRecord)));
        }
    }
}

// RFC 5246 section 6.2.1 rejects even authenticated plaintext above 2^14 bytes.
#[test]
fn aes_gcm_enforces_the_plaintext_record_boundary() {
    check_record_boundary(CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384);
}

// The same boundary applies to ChaCha20's different nonce and record layout.
#[test]
fn chacha20_enforces_the_plaintext_record_boundary() {
    check_record_boundary(CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256);
}
