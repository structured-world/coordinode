use getrandom::rand_core::{TryCryptoRng, TryRng};

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
