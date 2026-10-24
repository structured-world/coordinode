#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use crypto::{SharedSecret, SupportedKxGroup};
use crypto_common::Generate;
use getrandom::rand_core::TryCryptoRng;
use rustls::crypto;
use zeroize::Zeroizing;

#[derive(Debug)]
pub struct X25519;

impl X25519 {
    fn start_with_rng(
        rng: &mut impl TryCryptoRng,
    ) -> Result<Box<X25519KeyExchange>, rustls::Error> {
        let mut secret = Zeroizing::new([0u8; 32]);
        rng.try_fill_bytes(secret.as_mut())
            .map_err(|_| rustls::Error::General("entropy source failed".into()))?;
        let priv_key = x25519_dalek::StaticSecret::from(*secret);
        let pub_key = (&priv_key).into();
        Ok(Box::new(X25519KeyExchange { priv_key, pub_key }))
    }
}

impl crypto::SupportedKxGroup for X25519 {
    fn name(&self) -> rustls::NamedGroup {
        rustls::NamedGroup::X25519
    }

    fn start(&self) -> Result<Box<dyn crypto::ActiveKeyExchange>, rustls::Error> {
        Ok(Self::start_with_rng(&mut getrandom::SysRng)?)
    }
}

pub struct X25519KeyExchange {
    // The enclosing exchange is consumed on completion and is never cloned.
    priv_key: x25519_dalek::StaticSecret,
    pub_key: x25519_dalek::PublicKey,
}

impl crypto::ActiveKeyExchange for X25519KeyExchange {
    fn complete(self: Box<X25519KeyExchange>, peer: &[u8]) -> Result<SharedSecret, rustls::Error> {
        let peer_array: [u8; 32] = peer
            .try_into()
            .map_err(|_| rustls::Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
        let shared = self.priv_key.diffie_hellman(&peer_array.into());
        // RFC 8446 section 7.4.2 requires rejecting every all-zero shared secret.
        if !shared.was_contributory() {
            return Err(rustls::PeerMisbehaved::InvalidKeyShare.into());
        }
        Ok(shared.as_ref().into())
    }

    fn pub_key(&self) -> &[u8] {
        self.pub_key.as_bytes()
    }

    fn group(&self) -> rustls::NamedGroup {
        X25519.name()
    }
}

macro_rules! impl_kx {
    ($name:ident, $exchange:ident, $kx_name:expr, $secret:ty, $public_key:ty) => {
        #[derive(Debug)]
        pub struct $name;

        impl crypto::SupportedKxGroup for $name {
            fn name(&self) -> rustls::NamedGroup {
                $kx_name
            }

            fn start(&self) -> Result<Box<dyn crypto::ActiveKeyExchange>, rustls::Error> {
                let priv_key = <$secret>::try_generate_from_rng(&mut getrandom::SysRng)
                    .map_err(|_| rustls::Error::General("entropy source failed".into()))?;
                let pub_key: $public_key = (&priv_key).into();
                Ok(Box::new($exchange {
                    priv_key,
                    pub_key: pub_key.to_sec1_bytes(),
                }))
            }
        }

        pub struct $exchange {
            priv_key: $secret,
            pub_key: Box<[u8]>,
        }

        impl crypto::ActiveKeyExchange for $exchange {
            fn complete(self: Box<$exchange>, peer: &[u8]) -> Result<SharedSecret, rustls::Error> {
                let their_pub = <$public_key>::from_sec1_bytes(peer)
                    .map_err(|_| rustls::Error::from(rustls::PeerMisbehaved::InvalidKeyShare))?;
                Ok(self
                    .priv_key
                    .diffie_hellman(&their_pub)
                    .raw_secret_bytes()
                    .as_slice()
                    .into())
            }

            fn pub_key(&self) -> &[u8] {
                &self.pub_key
            }

            fn group(&self) -> rustls::NamedGroup {
                $name.name()
            }
        }
    };
}

impl_kx! {SecP256R1, SecP256R1KeyExchange, rustls::NamedGroup::secp256r1, p256::ecdh::EphemeralSecret, p256::PublicKey}
impl_kx! {SecP384R1, SecP384R1KeyExchange, rustls::NamedGroup::secp384r1, p384::ecdh::EphemeralSecret, p384::PublicKey}

pub const ALL_KX_GROUPS: &[&dyn SupportedKxGroup] = &[&X25519, &SecP256R1, &SecP384R1];

#[cfg(test)]
mod tests;
