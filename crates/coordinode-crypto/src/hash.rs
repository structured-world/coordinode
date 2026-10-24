#[cfg(feature = "alloc")]
use alloc::boxed::Box;

use digest::{Digest, OutputSizeUser};
use rustls::crypto::{self, hash};
use sha2::{Sha256, Sha384};

macro_rules! impl_hash {
    ($name:ident, $hash:ident, $context:ident, $ty:ty, $algo:expr) => {
        struct $hash;

        impl hash::Hash for $hash {
            fn start(&self) -> Box<dyn hash::Context> {
                Box::new($context(<$ty>::new()))
            }

            fn hash(&self, data: &[u8]) -> hash::Output {
                hash::Output::new(&<$ty>::digest(data)[..])
            }

            fn output_len(&self) -> usize {
                <$ty as OutputSizeUser>::output_size()
            }

            fn algorithm(&self) -> hash::HashAlgorithm {
                $algo
            }
        }

        struct $context($ty);

        impl hash::Context for $context {
            fn fork_finish(&self) -> hash::Output {
                hash::Output::new(&self.0.clone().finalize()[..])
            }

            fn fork(&self) -> Box<dyn hash::Context> {
                Box::new($context(self.0.clone()))
            }

            fn finish(self: Box<Self>) -> hash::Output {
                hash::Output::new(&self.0.finalize()[..])
            }

            fn update(&mut self, data: &[u8]) {
                self.0.update(data);
            }
        }

        pub const $name: &dyn crypto::hash::Hash = &$hash;
    };
}

impl_hash! {SHA256, Sha256Hash, Sha256Context, Sha256, hash::HashAlgorithm::SHA256}
impl_hash! {SHA384, Sha384Hash, Sha384Context, Sha384, hash::HashAlgorithm::SHA384}
