#[cfg(feature = "alloc")]
use alloc::{boxed::Box, format, sync::Arc};
use core::marker::PhantomData;

use der::Decode;
use pkcs8::DecodePrivateKey;
use pki_types::PrivateKeyDer;
use rustls::sign::SigningKey;
use rustls::{SignatureAlgorithm, SignatureScheme};

macro_rules! impl_ecdsa {
    ($name: ident, $scheme: expr, $scalar_len: expr, $secret_key: ty, $signing_key: ty, $signature: ty) => {
        #[derive(Debug)]
        pub struct $name {
            key: Arc<$signing_key>,
            scheme: SignatureScheme,
        }

        impl TryFrom<&PrivateKeyDer<'_>> for $name {
            type Error = rustls::Error;

            fn try_from(value: &PrivateKeyDer<'_>) -> Result<Self, Self::Error> {
                let pkey = match value {
                    PrivateKeyDer::Pkcs8(der) => {
                        <$signing_key>::from_pkcs8_der(der.secret_pkcs8_der())
                            .map_err(|e| format!("invalid PKCS#8 EC private key: {e}"))
                    }
                    PrivateKeyDer::Pkcs1(_) => Err("RSA private keys are not supported".into()),
                    PrivateKeyDer::Sec1(der) => sec1::EcPrivateKey::from_der(der.secret_sec1_der())
                        .map_err(|e| format!("failed to parse SEC1 EC private key: {e}"))
                        .and_then(|key| {
                            // RFC 5915 section 3 fixes the scalar width and requires named parameters.
                            if key.private_key.len() != $scalar_len || key.parameters.is_none() {
                                return Err("invalid SEC1 scalar width or missing curve".into());
                            }
                            <$secret_key>::try_from(key)
                                .map(<$signing_key>::from)
                                .map_err(|e| format!("invalid SEC1 EC private key: {e}"))
                        }),
                    _ => Err("not supported".into()),
                };
                pkey.map(|kp| Self {
                    key: Arc::new(kp),
                    scheme: $scheme,
                })
                .map_err(rustls::Error::General)
            }
        }

        impl SigningKey for $name {
            fn choose_scheme(
                &self,
                offered: &[SignatureScheme],
            ) -> Option<Box<dyn rustls::sign::Signer>> {
                if offered.contains(&self.scheme) {
                    Some(Box::new(super::GenericRandomizedSigner::<$signature, _> {
                        _marker: PhantomData,
                        key: self.key.clone(),
                        scheme: self.scheme,
                    }))
                } else {
                    None
                }
            }

            fn algorithm(&self) -> SignatureAlgorithm {
                SignatureAlgorithm::ECDSA
            }
        }
    };
}

impl_ecdsa! {EcdsaSigningKeyP256, SignatureScheme::ECDSA_NISTP256_SHA256, 32, p256::SecretKey, p256::ecdsa::SigningKey, p256::ecdsa::DerSignature}
impl_ecdsa! {EcdsaSigningKeyP384, SignatureScheme::ECDSA_NISTP384_SHA384, 48, p384::SecretKey, p384::ecdsa::SigningKey, p384::ecdsa::DerSignature}
