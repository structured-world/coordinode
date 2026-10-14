//! One database's inter-node security context: its node credentials, the
//! profile its peers are verified under, and the rustls configurations
//! built from them.
//!
//! Each independent database owns its own context, so two databases in one
//! process never share credentials or trust. A handshake reads the current
//! credentials and profile from memory; an update validates a complete
//! replacement and publishes it atomically, or leaves the last valid one in
//! place and says why.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arc_swap::ArcSwap;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::{ClientConfig, ResolvesClientCert, Resumption};
use rustls::crypto::{CryptoProvider, WebPkiSupportedAlgorithms};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls::server::{ClientHello, NoServerSessionStorage, ResolvesServerCert, ServerConfig};
use rustls::sign::CertifiedKey;
use rustls::{CertificateError, DigitallySignedStruct, DistinguishedName, Error, SignatureScheme};

use super::verify::{PeerError, PeerIdentity, PeerProfile, PeerRole, VerifiedPeer};

/// Why a security context or an update to it was refused. The context
/// keeps what it had.
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    /// The certificate or key does not parse.
    #[error("node credentials: {0}")]
    Credentials(String),
    /// The private key does not belong to the certificate, or the
    /// provider cannot confirm that it does.
    #[error("the node key does not match its certificate: {0}")]
    KeyMismatch(String),
    /// The node's own certificate establishes no identity under its
    /// profile, so no peer could accept it.
    #[error("the node certificate is not a valid node identity: {0}")]
    OwnIdentity(PeerError),
    /// A credential update names another identity than the node has.
    /// Renewal keeps the identity; a new identity is a new enrollment.
    #[error("renewed credentials name {new}, the node is {current}")]
    IdentityChanged {
        /// The identity the node has.
        current: PeerIdentity,
        /// The identity the update names.
        new: PeerIdentity,
    },
}

/// How an inbound listener treats a connecting peer's certificate.
#[derive(Clone)]
pub enum InboundAuth {
    /// Every connection is a node and must present a node identity
    /// (an embedded database's only listener).
    NodeOnly,
    /// A port shared with clients: a certificate presenting a node
    /// identity is verified as one, any other is left to `clients`, and
    /// a connection without a certificate is a client without one.
    Shared {
        /// Ordinary client certificates, when the port accepts them.
        clients: Option<Arc<dyn ClientCertVerifier>>,
    },
}

impl fmt::Debug for InboundAuth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeOnly => f.write_str("NodeOnly"),
            Self::Shared { clients } => f
                .debug_struct("Shared")
                .field("client_certificates", &clients.is_some())
                .finish(),
        }
    }
}

/// What a context holds, replaced as a whole.
struct State {
    credentials: Arc<CertifiedKey>,
    profile: Arc<PeerProfile>,
    identity: PeerIdentity,
}

/// One database's inter-node security context.
pub struct SecurityContext {
    provider: Arc<CryptoProvider>,
    state: ArcSwap<State>,
    /// Raised by every trust update, so a connection verified under older
    /// trust is verified again before it carries more traffic.
    generation: AtomicU64,
}

impl fmt::Debug for SecurityContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state.load();
        f.debug_struct("SecurityContext")
            .field("identity", &state.identity)
            .field("profile", &state.profile)
            .finish_non_exhaustive()
    }
}

impl SecurityContext {
    /// A context presenting `chain` (leaf first) with `key`, verifying
    /// peers under `profile`, on `provider`.
    ///
    /// # Errors
    ///
    /// The credentials do not parse or match, or the node's own
    /// certificate is not a valid identity under `profile`.
    pub fn new(
        provider: Arc<CryptoProvider>,
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
        profile: PeerProfile,
    ) -> Result<Self, ContextError> {
        let profile = Arc::new(profile);
        let (credentials, own) = certify(&provider, chain, key, &profile)?;
        Ok(Self {
            provider,
            state: ArcSwap::from_pointee(State {
                credentials,
                profile,
                identity: own.identity,
            }),
            generation: AtomicU64::new(0),
        })
    }

    /// This node's identity.
    #[must_use]
    pub fn identity(&self) -> PeerIdentity {
        self.state.load().identity.clone()
    }

    /// The trust generation: raised by every trust update.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Whether `end_entity` presents itself as a node identity under the
    /// current profile.
    #[must_use]
    pub fn claims_node_identity(&self, end_entity: &CertificateDer<'_>) -> bool {
        self.state.load().profile.claims_node_identity(end_entity)
    }

    /// Replace the node's certificate and key: a renewal keeping the same
    /// identity. New handshakes use them; established connections keep
    /// the certificate they were authenticated with until they end.
    ///
    /// # Errors
    ///
    /// The replacement is incomplete or names another identity; the
    /// current credentials stay in force and their validity is unchanged.
    pub fn update_credentials(
        &self,
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
    ) -> Result<(), ContextError> {
        let current = self.state.load_full();
        let (credentials, own) = certify(&self.provider, chain, key, &current.profile)?;
        if own.identity != current.identity {
            return Err(ContextError::IdentityChanged {
                current: current.identity.clone(),
                new: own.identity,
            });
        }
        self.state.store(Arc::new(State {
            credentials,
            profile: Arc::clone(&current.profile),
            identity: own.identity,
        }));
        Ok(())
    }

    /// Replace the profile peers are verified under (new trust bundles, a
    /// withdrawn trust domain). The node's own certificate must remain a
    /// valid identity under the new profile.
    ///
    /// # Errors
    ///
    /// The node's own certificate is not a valid identity under
    /// `profile`; the current profile stays in force.
    pub fn update_trust(&self, profile: PeerProfile) -> Result<(), ContextError> {
        let current = self.state.load_full();
        let profile = Arc::new(profile);
        let own = verify_own(&self.provider, &current.credentials, &profile)?;
        if own.identity != current.identity {
            return Err(ContextError::IdentityChanged {
                current: current.identity.clone(),
                new: own.identity,
            });
        }
        self.state.store(Arc::new(State {
            credentials: Arc::clone(&current.credentials),
            profile,
            identity: own.identity,
        }));
        // After the store: a connection that sees the new generation
        // verifies against the new profile.
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Replace the credentials and the profile together, as one validated
    /// replacement: what a reload of a node's certificate, key and trust
    /// bundles publishes when a CA rotation changes both at once. Every
    /// established connection is verified again under the new profile.
    ///
    /// # Errors
    ///
    /// The credentials do not parse or match, are no valid identity under
    /// `profile`, or name another identity than the node has; the current
    /// credentials and profile stay in force.
    pub fn replace(
        &self,
        chain: Vec<CertificateDer<'static>>,
        key: PrivateKeyDer<'static>,
        profile: PeerProfile,
    ) -> Result<(), ContextError> {
        let current = self.state.load_full();
        let profile = Arc::new(profile);
        let (credentials, own) = certify(&self.provider, chain, key, &profile)?;
        if own.identity != current.identity {
            return Err(ContextError::IdentityChanged {
                current: current.identity.clone(),
                new: own.identity,
            });
        }
        self.state.store(Arc::new(State {
            credentials,
            profile,
            identity: own.identity,
        }));
        self.generation.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    /// Verify every established connection's peer again before it carries
    /// more traffic, under the trust in force: what a host calls once its
    /// adapter stops admitting an identity (its issuer revoked it), so the
    /// revocation reaches connections made before it.
    pub fn reverify_peers(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Verify a peer's chain under the current profile.
    ///
    /// # Errors
    ///
    /// The chain establishes no node identity.
    pub fn verify_peer(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
        role: PeerRole,
    ) -> Result<VerifiedPeer, PeerError> {
        self.state.load().profile.verify(
            end_entity,
            intermediates,
            now,
            role,
            self.provider.signature_verification_algorithms.all,
        )
    }

    /// The server configuration of an inbound listener. Session
    /// resumption is off: a resumed session skips certificate
    /// verification, which would revive evidence that expired or was
    /// withdrawn.
    ///
    /// # Errors
    ///
    /// The provider offers no TLS protocol version.
    pub fn server_config(self: &Arc<Self>, inbound: InboundAuth) -> Result<ServerConfig, Error> {
        let verifier = Arc::new(InboundVerifier {
            context: Arc::clone(self),
            inbound,
        });
        let mut config = ServerConfig::builder_with_provider(Arc::clone(&self.provider))
            .with_safe_default_protocol_versions()?
            .with_client_cert_verifier(verifier)
            .with_cert_resolver(Arc::new(Credentials(Arc::clone(self))));
        config.session_storage = Arc::new(NoServerSessionStorage {});
        config.send_tls13_tickets = 0;
        config.alpn_protocols = vec![b"h2".to_vec()];
        Ok(config)
    }

    /// The client configuration for dialing the peer `expected`: the
    /// connection succeeds only if the peer proves that identity. The
    /// address dialed is a hint and is not checked. Resumption is off, as
    /// for [`Self::server_config`].
    ///
    /// # Errors
    ///
    /// The provider offers no TLS protocol version.
    pub fn client_config(self: &Arc<Self>, expected: PeerIdentity) -> Result<ClientConfig, Error> {
        let verifier = Arc::new(OutboundVerifier {
            context: Arc::clone(self),
            expected,
        });
        let mut config = ClientConfig::builder_with_provider(Arc::clone(&self.provider))
            .with_safe_default_protocol_versions()?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_client_cert_resolver(Arc::new(Credentials(Arc::clone(self))));
        config.resumption = Resumption::disabled();
        config.alpn_protocols = vec![b"h2".to_vec()];
        Ok(config)
    }

    fn algorithms(&self) -> &WebPkiSupportedAlgorithms {
        &self.provider.signature_verification_algorithms
    }
}

/// Parse and check `chain` + `key`, and the identity they establish.
fn certify(
    provider: &CryptoProvider,
    chain: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
    profile: &PeerProfile,
) -> Result<(Arc<CertifiedKey>, VerifiedPeer), ContextError> {
    if chain.is_empty() {
        return Err(ContextError::Credentials("no certificate".to_owned()));
    }
    let signing = provider
        .key_provider
        .load_private_key(key)
        .map_err(|e| ContextError::Credentials(e.to_string()))?;
    let credentials = Arc::new(CertifiedKey::new(chain, signing));
    prove_possession(provider, &credentials)?;
    let own = verify_own(provider, &credentials, profile)?;
    Ok((credentials, own))
}

/// Sign a fixed message with the key and verify it with the certificate's
/// public key, on the same provider: proof that the key belongs to the
/// certificate whether or not the key can report its public half.
fn prove_possession(
    provider: &CryptoProvider,
    credentials: &CertifiedKey,
) -> Result<(), ContextError> {
    const MESSAGE: &[u8] = b"coordinode node credential possession check";
    let mismatch = |reason: String| ContextError::KeyMismatch(reason);
    let algorithms = &provider.signature_verification_algorithms;
    let signer = credentials
        .key
        .choose_scheme(&algorithms.supported_schemes())
        .ok_or_else(|| mismatch("the key signs with no scheme the provider verifies".to_owned()))?;
    let scheme = signer.scheme();
    let signature = signer
        .sign(MESSAGE)
        .map_err(|e| mismatch(format!("signing: {e}")))?;
    let leaf = credentials
        .end_entity_cert()
        .map_err(|e| mismatch(e.to_string()))?;
    let cert = webpki::EndEntityCert::try_from(leaf).map_err(|e| mismatch(e.to_string()))?;
    let verified = algorithms
        .mapping
        .iter()
        .filter(|(s, _)| *s == scheme)
        .flat_map(|(_, algs)| algs.iter())
        .any(|alg| cert.verify_signature(*alg, MESSAGE, &signature).is_ok());
    if verified {
        Ok(())
    } else {
        Err(mismatch(
            "the certificate does not verify the key's signature".to_owned(),
        ))
    }
}

/// The node's own certificate verified as its peers will verify it, in
/// both roles.
fn verify_own(
    provider: &CryptoProvider,
    credentials: &CertifiedKey,
    profile: &PeerProfile,
) -> Result<VerifiedPeer, ContextError> {
    let (leaf, intermediates) = credentials
        .cert
        .split_first()
        .ok_or_else(|| ContextError::Credentials("no certificate".to_owned()))?;
    let now = UnixTime::now();
    let algorithms = provider.signature_verification_algorithms.all;
    profile
        .verify(leaf, intermediates, now, PeerRole::Server, algorithms)
        .and_then(|_| profile.verify(leaf, intermediates, now, PeerRole::Client, algorithms))
        .map_err(ContextError::OwnIdentity)
}

/// The current node credentials, read at every handshake.
struct Credentials(Arc<SecurityContext>);

impl fmt::Debug for Credentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Credentials")
    }
}

impl ResolvesServerCert for Credentials {
    fn resolve(&self, _hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(Arc::clone(&self.0.state.load().credentials))
    }
}

impl ResolvesClientCert for Credentials {
    fn resolve(&self, _hints: &[&[u8]], _schemes: &[SignatureScheme]) -> Option<Arc<CertifiedKey>> {
        Some(Arc::clone(&self.0.state.load().credentials))
    }

    fn has_certs(&self) -> bool {
        true
    }
}

fn refusal(error: PeerError) -> Error {
    Error::InvalidCertificate(CertificateError::Other(rustls::OtherError(Arc::new(error))))
}

/// Verifies a connecting peer.
struct InboundVerifier {
    context: Arc<SecurityContext>,
    inbound: InboundAuth,
}

impl fmt::Debug for InboundVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InboundVerifier")
            .field("inbound", &self.inbound)
            .finish_non_exhaustive()
    }
}

impl InboundVerifier {
    /// The verifier of ordinary clients, for a certificate that presents
    /// no node identity on a shared port.
    fn clients(&self, end_entity: &CertificateDer<'_>) -> Option<&Arc<dyn ClientCertVerifier>> {
        match &self.inbound {
            InboundAuth::Shared { clients } => {
                let profile = Arc::clone(&self.context.state.load().profile);
                if profile.claims_node_identity(end_entity) {
                    None
                } else {
                    clients.as_ref()
                }
            }
            InboundAuth::NodeOnly => None,
        }
    }
}

impl ClientCertVerifier for InboundVerifier {
    fn client_auth_mandatory(&self) -> bool {
        match &self.inbound {
            InboundAuth::NodeOnly => true,
            // A port whose clients must present certificates leaves no one
            // without one.
            InboundAuth::Shared { clients } => clients
                .as_ref()
                .is_some_and(|clients| clients.client_auth_mandatory()),
        }
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        match &self.inbound {
            InboundAuth::Shared {
                clients: Some(clients),
            } => clients.root_hint_subjects(),
            _ => &[],
        }
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> Result<ClientCertVerified, Error> {
        if let InboundAuth::Shared { clients } = &self.inbound {
            if !self
                .context
                .state
                .load()
                .profile
                .claims_node_identity(end_entity)
            {
                return match clients {
                    Some(clients) => clients.verify_client_cert(end_entity, intermediates, now),
                    None => Err(Error::InvalidCertificate(
                        CertificateError::ApplicationVerificationFailure,
                    )),
                };
            }
        }
        self.context
            .verify_peer(end_entity, intermediates, now, PeerRole::Client)
            .map(|_| ClientCertVerified::assertion())
            .map_err(refusal)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        if let Some(clients) = self.clients(cert) {
            return clients.verify_tls12_signature(message, cert, dss);
        }
        rustls::crypto::verify_tls12_signature(message, cert, dss, self.context.algorithms())
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        if let Some(clients) = self.clients(cert) {
            return clients.verify_tls13_signature(message, cert, dss);
        }
        rustls::crypto::verify_tls13_signature(message, cert, dss, self.context.algorithms())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.context.algorithms().supported_schemes()
    }
}

/// Verifies the peer this node dialed against the identity it expected.
struct OutboundVerifier {
    context: Arc<SecurityContext>,
    expected: PeerIdentity,
}

impl fmt::Debug for OutboundVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OutboundVerifier")
            .field("expected", &self.expected)
            .finish_non_exhaustive()
    }
}

/// The peer this node dialed presented another identity than expected.
#[derive(thiserror::Error)]
#[error("the peer is {found}, expected {expected}")]
struct UnexpectedPeer {
    found: PeerIdentity,
    expected: PeerIdentity,
}

/// rustls reports a verifier's error through `Debug`; keep it the
/// sentence `Display` gives.
impl fmt::Debug for UnexpectedPeer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl ServerCertVerifier for OutboundVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        let peer = self
            .context
            .verify_peer(end_entity, intermediates, now, PeerRole::Server)
            .map_err(refusal)?;
        if peer.identity != self.expected {
            return Err(Error::InvalidCertificate(CertificateError::Other(
                rustls::OtherError(Arc::new(UnexpectedPeer {
                    found: peer.identity,
                    expected: self.expected.clone(),
                })),
            )));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        rustls::crypto::verify_tls12_signature(message, cert, dss, self.context.algorithms())
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        rustls::crypto::verify_tls13_signature(message, cert, dss, self.context.algorithms())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.context.algorithms().supported_schemes()
    }
}

#[cfg(test)]
mod tests;
