# CoordiNode RustCrypto provider

A pure-Rust cryptography provider for rustls, based on
[RustCrypto/rustls-rustcrypto](https://github.com/RustCrypto/rustls-rustcrypto)
revision `70f76c039e587192688af18a80d5d6435dedaf22`.

Rustls owns TLS and certificate-path validation. This package adapts RustCrypto
primitives to rustls and retains the upstream MIT OR Apache-2.0 licensing.

## Use

```rust
let provider = std::sync::Arc::new(coordinode_crypto::provider());
let builder = rustls::ClientConfig::builder_with_provider(provider)
    .with_safe_default_protocol_versions()?;
# Ok::<(), rustls::Error>(())
```

The provider supports TLS 1.3 and, with `tls12`, TLS 1.2. It supplies AES-GCM,
ChaCha20-Poly1305, X25519, P-256 and P-384 key exchange, and ECDSA/Ed25519
local identities. X25519 rejects every all-zero shared secret. RSA certificate
and peer signatures remain verifiable with 2048..=8192-bit public keys, while
RSA private signing and decryption are unavailable through the provider.
`identity::IdentityKeyProvider` applies the same local identity admission to
an explicitly selected, separately qualified backend.

Standalone SEC1 identities require named P-256/P-384 parameters and canonical
scalar widths. PKCS#8 identities retain algorithm checks; a PEM container name
does not identify its private-key algorithm. Rustls retains certificate-chain,
hostname, purpose and validity checks.

## Allocation-only builds

The library is `no_std` with an explicit `alloc` feature. The default `std`
feature enables rustls' standard-library integration. Secret key storage is
zeroized independently of that choice.

Targets without an operating-system entropy source must supply a secure custom
backend using the [getrandom backend contract](https://docs.rs/getrandom/0.4.3/getrandom/#custom-backend).
An allocation-only compile check is:

```sh
RUSTFLAGS='--cfg getrandom_backend="custom"' cargo check -p coordinode-crypto \
  --no-default-features --features alloc --target thumbv7em-none-eabihf
```

This checks compilation, not the availability or security of a target's entropy
implementation. Production entropy errors propagate as failures.

The package has an independent version and is shared by CoordiNode and
structured-proxy. See `LICENSE-MIT`, `LICENSE-APACHE` and `NOTICES` for licensing.
