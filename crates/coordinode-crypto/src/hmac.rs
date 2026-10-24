#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use crypto_common::OutputSizeUser;
use hmac::{KeyInit, Mac};
use rustls::crypto;
use sha2::{Sha256, Sha384};

macro_rules! impl_hmac {
    (
        $name: ident,
        $hmac: ident,
        $key: ident,
        $ty: ty
    ) => {
        struct $hmac;

        impl crypto::hmac::Hmac for $hmac {
            fn with_key(&self, key: &[u8]) -> Box<dyn crypto::hmac::Key> {
                Box::new($key(
                    hmac::Hmac::<$ty>::new_from_slice(key).expect("HMAC accepts any key length"),
                ))
            }

            fn hash_output_len(&self) -> usize {
                <$ty>::output_size()
            }
        }

        struct $key(hmac::Hmac<$ty>);

        impl crypto::hmac::Key for $key {
            fn sign_concat(
                &self,
                first: &[u8],
                middle: &[&[u8]],
                last: &[u8],
            ) -> crypto::hmac::Tag {
                let mut ctx = self.0.clone();
                ctx.update(first);
                for m in middle {
                    ctx.update(m);
                }
                ctx.update(last);
                crypto::hmac::Tag::new(&ctx.finalize().into_bytes()[..])
            }

            fn tag_len(&self) -> usize {
                <$ty>::output_size()
            }
        }
        pub const $name: &dyn crypto::hmac::Hmac = &$hmac;
    };
}

impl_hmac! {SHA256, Sha256Hmac, Sha256HmacKey, Sha256}
impl_hmac! {SHA384, Sha384Hmac, Sha384HmacKey, Sha384}
