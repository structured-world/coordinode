use pki_types::{AlgorithmIdentifier, InvalidSignature, SignatureVerificationAlgorithm, alg_id};
use rsa::traits::PublicKeyParts;
use rsa::{pkcs1v15, pss};
use sha2::{Sha256, Sha384, Sha512};
use signature::Verifier;

macro_rules! impl_generic_rsa_verifer {
    (
        $name:ident,
        $verifier:ident,
        $public_key_algo:expr,
        $signature_alg_id:expr,
        $verifying_key:ty,
        $signature:ty
    ) => {
        #[derive(Debug)]
        struct $verifier;

        impl SignatureVerificationAlgorithm for $verifier {
            fn public_key_alg_id(&self) -> AlgorithmIdentifier {
                $public_key_algo
            }

            fn signature_alg_id(&self) -> AlgorithmIdentifier {
                $signature_alg_id
            }

            fn verify_signature(
                &self,
                public_key: &[u8],
                message: &[u8],
                signature: &[u8],
            ) -> Result<(), InvalidSignature> {
                let public_key = crate::public_key::rsa_from_pkcs1(public_key)?;
                if signature.len() != public_key.size() {
                    return Err(InvalidSignature);
                }
                let signature = <$signature>::try_from(signature).map_err(|_| InvalidSignature)?;
                <$verifying_key>::new(public_key)
                    .verify(message, &signature)
                    .map_err(|_| InvalidSignature)
            }
        }

        pub const $name: &dyn SignatureVerificationAlgorithm = &$verifier;
    };
}

impl_generic_rsa_verifer!(
    RSA_PKCS1_SHA256,
    RsaPkcs1Sha256Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PKCS1_SHA256,
    pkcs1v15::VerifyingKey<Sha256>,
    pkcs1v15::Signature
);
impl_generic_rsa_verifer!(
    RSA_PKCS1_SHA384,
    RsaPkcs1Sha384Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PKCS1_SHA384,
    pkcs1v15::VerifyingKey<Sha384>,
    pkcs1v15::Signature
);
impl_generic_rsa_verifer!(
    RSA_PKCS1_SHA512,
    RsaPkcs1Sha512Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PKCS1_SHA512,
    pkcs1v15::VerifyingKey<Sha512>,
    pkcs1v15::Signature
);

impl_generic_rsa_verifer!(
    RSA_PSS_SHA256,
    RsaPssSha256Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PSS_SHA256,
    pss::VerifyingKey<Sha256>,
    pss::Signature
);
impl_generic_rsa_verifer!(
    RSA_PSS_SHA384,
    RsaPssSha384Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PSS_SHA384,
    pss::VerifyingKey<Sha384>,
    pss::Signature
);
impl_generic_rsa_verifer!(
    RSA_PSS_SHA512,
    RsaPssSha512Verifier,
    alg_id::RSA_ENCRYPTION,
    alg_id::RSA_PSS_SHA512,
    pss::VerifyingKey<Sha512>,
    pss::Signature
);
