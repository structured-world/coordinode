#![allow(clippy::unwrap_used)]

use getrandom::rand_core::{TryCryptoRng, TryRng};
use rustls::crypto::SupportedKxGroup;

struct FailedEntropy;

impl TryRng for FailedEntropy {
    type Error = getrandom::Error;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Err(getrandom::Error::UNSUPPORTED)
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Err(getrandom::Error::UNSUPPORTED)
    }

    fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Self::Error> {
        Err(getrandom::Error::UNSUPPORTED)
    }
}

impl TryCryptoRng for FailedEntropy {}

// Entropy failure must return a TLS error before an exchange is constructed.
#[test]
fn entropy_failure_does_not_panic_or_create_an_exchange() {
    assert!(matches!(
        super::X25519::start_with_rng(&mut FailedEntropy),
        Err(rustls::Error::General(_))
    ));
}

fn rejects_compressed_share(group: &dyn SupportedKxGroup, wire_len: usize) {
    let peer = group.start().unwrap();
    let encoded = peer.pub_key();
    assert_eq!(encoded.len(), wire_len);
    assert_eq!(encoded[0], 4);
    assert!(group.start().unwrap().complete(encoded).is_ok());

    let coordinate_len = (wire_len - 1) / 2;
    let mut compressed = alloc::vec::Vec::with_capacity(coordinate_len + 1);
    compressed.push(2 | (encoded[wire_len - 1] & 1));
    compressed.extend_from_slice(&encoded[1..coordinate_len + 1]);
    assert!(matches!(
        group.start().unwrap().complete(&compressed),
        Err(rustls::Error::PeerMisbehaved(
            rustls::PeerMisbehaved::InvalidKeyShare
        ))
    ));
}

// A valid SEC1 compressed point is still forbidden on the TLS key-share wire.
#[test]
fn p256_rejects_compressed_wire_share() {
    rejects_compressed_share(&super::SecP256R1, 65);
}

// P-384 follows the same TLS format rule with 48-byte coordinates.
#[test]
fn p384_rejects_compressed_wire_share() {
    rejects_compressed_share(&super::SecP384R1, 97);
}
