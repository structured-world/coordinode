#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use super::{DecryptBufferAdapter, EncryptBufferAdapter};

use aead::AeadInOut;
use crypto_common::{KeyInit, KeySizeUser};
use rustls::crypto::cipher::{
    self, AeadKey, InboundOpaqueMessage, InboundPlainMessage, MessageDecrypter, MessageEncrypter,
    OutboundOpaqueMessage, OutboundPlainMessage, PrefixedPayload, Tls13AeadAlgorithm,
};
use rustls::{ConnectionTrafficSecrets, ContentType, ProtocolVersion};

#[cfg(feature = "tls12")]
use {
    aead::AeadCore,
    crypto_common::typenum::Unsigned,
    rustls::crypto::cipher::{Iv, KeyBlockShape, Tls12AeadAlgorithm},
};

#[cfg(feature = "tls12")]
const TLS12_GCM_EXPLICIT_NONCE_LEN: usize = 8;

#[cfg(feature = "tls12")]
const TLS12_GCM_OVERHEAD: usize = TLS12_GCM_EXPLICIT_NONCE_LEN + 16;

macro_rules! impl_gcm_tls13 {
    ($name: ident, $cipher: ident, $variant: ident, $aead: ty, $overhead: expr) => {
        pub struct $name;

        impl Tls13AeadAlgorithm for $name {
            fn encrypter(&self, key: AeadKey, iv: cipher::Iv) -> Box<dyn MessageEncrypter> {
                Box::new($cipher(
                    <$aead>::new_from_slice(key.as_ref())
                        .expect("rustls supplies the negotiated AES key length"),
                    iv,
                ))
            }

            fn decrypter(&self, key: AeadKey, iv: cipher::Iv) -> Box<dyn MessageDecrypter> {
                Box::new($cipher(
                    <$aead>::new_from_slice(key.as_ref())
                        .expect("rustls supplies the negotiated AES key length"),
                    iv,
                ))
            }

            fn key_len(&self) -> usize {
                <$aead>::key_size()
            }
            fn extract_keys(
                &self,
                key: AeadKey,
                iv: cipher::Iv,
            ) -> Result<ConnectionTrafficSecrets, cipher::UnsupportedOperationError> {
                Ok(ConnectionTrafficSecrets::$variant { key, iv })
            }
        }

        struct $cipher($aead, cipher::Iv);

        impl MessageEncrypter for $cipher {
            fn encrypt(
                &mut self,
                m: OutboundPlainMessage<'_>,
                seq: u64,
            ) -> Result<OutboundOpaqueMessage, rustls::Error> {
                let total_len = self.encrypted_payload_len(m.payload.len());
                let mut payload = PrefixedPayload::with_capacity(total_len);

                let nonce = cipher::Nonce::new(&self.1, seq).0;
                let aad = cipher::make_tls13_aad(total_len);
                payload.extend_from_chunks(&m.payload);
                payload.extend_from_slice(&m.typ.to_array());

                self.0
                    .encrypt_in_place(&nonce.into(), &aad, &mut EncryptBufferAdapter(&mut payload))
                    .map_err(|_| rustls::Error::EncryptError)
                    .map(|_| {
                        OutboundOpaqueMessage::new(
                            ContentType::ApplicationData,
                            ProtocolVersion::TLSv1_2,
                            payload,
                        )
                    })
            }

            fn encrypted_payload_len(&self, payload_len: usize) -> usize {
                payload_len + 1 + $overhead
            }
        }

        impl MessageDecrypter for $cipher {
            fn decrypt<'a>(
                &mut self,
                mut m: InboundOpaqueMessage<'a>,
                seq: u64,
            ) -> Result<InboundPlainMessage<'a>, rustls::Error> {
                let payload = &mut m.payload;
                let nonce = cipher::Nonce::new(&self.1, seq).0;
                let aad = cipher::make_tls13_aad(payload.len());

                self.0
                    .decrypt_in_place(&nonce.into(), &aad, &mut DecryptBufferAdapter(payload))
                    .map_err(|_| rustls::Error::DecryptError)?;

                m.into_tls13_unpadded_message()
            }
        }
    };
}

#[cfg(feature = "tls12")]
macro_rules! impl_gcm_tls12 {
    ($name: ident, $encrypter: ident, $decrypter: ident, $variant: ident, $aead: ty, $nonce: expr, $overhead: expr) => {
        #[cfg(feature = "tls12")]
        pub struct $name;

        #[cfg(feature = "tls12")]
        impl Tls12AeadAlgorithm for $name {
            fn encrypter(
                &self,
                key: AeadKey,
                write_iv: &[u8],
                explicit: &[u8],
            ) -> Box<dyn MessageEncrypter> {
                Box::new($encrypter(
                    <$aead>::new_from_slice(key.as_ref())
                        .expect("rustls supplies the negotiated AES key length"),
                    {
                        let mut iv: [u8; 12] = [0; 12];
                        iv[..4].copy_from_slice(write_iv);
                        iv[4..].copy_from_slice(explicit);
                        iv
                    },
                ))
            }

            fn decrypter(&self, dec_key: AeadKey, dec_iv: &[u8]) -> Box<dyn MessageDecrypter> {
                Box::new($decrypter(
                    <$aead>::new_from_slice(dec_key.as_ref())
                        .expect("rustls supplies the negotiated AES key length"),
                    dec_iv
                        .try_into()
                        .expect("rustls supplies a 4-byte TLS 1.2 AES fixed IV"),
                ))
            }

            fn key_block_shape(&self) -> KeyBlockShape {
                KeyBlockShape {
                    enc_key_len: <$aead>::key_size(),
                    fixed_iv_len: 4,
                    explicit_nonce_len: 8,
                }
            }

            fn extract_keys(
                &self,
                key: AeadKey,
                iv: &[u8],
                explicit: &[u8],
            ) -> Result<ConnectionTrafficSecrets, cipher::UnsupportedOperationError> {
                let mut nonce = [0; 12];
                nonce[..4].copy_from_slice(iv);
                nonce[4..].copy_from_slice(explicit);
                Ok(ConnectionTrafficSecrets::$variant {
                    key,
                    iv: Iv::new(nonce),
                })
            }
        }

        #[cfg(feature = "tls12")]
        struct $encrypter($aead, [u8; 12]);

        #[cfg(feature = "tls12")]
        impl MessageEncrypter for $encrypter {
            fn encrypt(
                &mut self,
                m: OutboundPlainMessage<'_>,
                seq: u64,
            ) -> Result<OutboundOpaqueMessage, rustls::Error> {
                let total_len = self.encrypted_payload_len(m.payload.len());
                let mut payload = PrefixedPayload::with_capacity(total_len);

                let nonce = cipher::Nonce::new(&self.1.into(), seq).0;
                let aad = cipher::make_tls12_aad(seq, m.typ, m.version, m.payload.len());
                payload.extend_from_slice(&nonce.as_ref()[4..]); // explicit
                payload.extend_from_chunks(&m.payload);

                self.0
                    .encrypt_inout_detached(
                        &nonce.into(),
                        &aad,
                        (&mut payload.as_mut()[$nonce..]).into(),
                    )
                    .map(|tag| payload.extend(tag.as_ref() as &[u8]))
                    .map_err(|_| rustls::Error::EncryptError)
                    .map(|_| OutboundOpaqueMessage::new(m.typ, m.version, payload))
            }
            fn encrypted_payload_len(&self, payload_len: usize) -> usize {
                payload_len + $nonce + <$aead as AeadCore>::TagSize::USIZE
            }
        }

        #[cfg(feature = "tls12")]
        struct $decrypter($aead, [u8; 4]);

        #[cfg(feature = "tls12")]
        impl MessageDecrypter for $decrypter {
            fn decrypt<'a>(
                &mut self,
                mut m: InboundOpaqueMessage<'a>,
                seq: u64,
            ) -> Result<InboundPlainMessage<'a>, rustls::Error> {
                type TagSize = <$aead as AeadCore>::TagSize;

                let payload = &m.payload;

                if payload.len() < $overhead {
                    return Err(rustls::Error::DecryptError);
                }

                let nonce: aead::Nonce<$aead> = {
                    let mut nonce = [0u8; 12];
                    nonce[..4].copy_from_slice(&self.1); // dec_iv
                    nonce[4..].copy_from_slice(&payload[..$nonce]);
                    nonce.into()
                };

                let aad = cipher::make_tls12_aad(seq, m.typ, m.version, payload.len() - $overhead);

                let payload = &mut m.payload;
                let tag_pos = {
                    let payload = &mut payload[$nonce..];
                    let tag_pos = payload.len() - TagSize::to_usize();
                    let (msg, tag) = payload.split_at_mut(tag_pos);

                    let tag = aes_gcm::Tag::<TagSize>::try_from(&*tag)
                        .map_err(|_| rustls::Error::DecryptError)?;
                    self.0
                        .decrypt_inout_detached(&nonce, &aad, msg.into(), &tag)
                        .map_err(|_| rustls::Error::DecryptError)?;
                    tag_pos
                };

                // We defer the truncation to here, because we may inadvertently shifted the
                // original data if the decryption failed. Another way to avoid this is
                // to clone the payload slice starting after the explicit nonce,
                // but this will cause an additional cloning and copying
                payload.rotate_left($nonce);
                payload.truncate(tag_pos);
                Ok(m.into_plain_message())
            }
        }
    };
}

impl_gcm_tls13! {Tls13Aes128Gcm, Tls13CipherAes128Gcm, Aes128Gcm, aes_gcm::Aes128Gcm, 16}
impl_gcm_tls13! {Tls13Aes256Gcm, Tls13CipherAes256Gcm, Aes256Gcm, aes_gcm::Aes256Gcm, 16}

#[cfg(feature = "tls12")]
impl_gcm_tls12! {Tls12Aes128Gcm, Tls12CipherAes128GcmEncrypter, Tls12CipherAes128GcmDecrypter, Aes128Gcm, aes_gcm::Aes128Gcm, TLS12_GCM_EXPLICIT_NONCE_LEN, TLS12_GCM_OVERHEAD}

#[cfg(feature = "tls12")]
impl_gcm_tls12! {Tls12Aes256Gcm, Tls12CipherAes256GcmEncrypter, Tls12CipherAes256GcmDecrypter, Aes256Gcm, aes_gcm::Aes256Gcm, TLS12_GCM_EXPLICIT_NONCE_LEN, TLS12_GCM_OVERHEAD}
