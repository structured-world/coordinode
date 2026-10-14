//! Establishing a peer's node identity from the certificate chain it
//! presented, under one of two explicitly selected profiles.
//!
//! Both profiles validate the chain on the common provider path (the
//! signature algorithms of the selected rustls crypto provider) and only
//! then interpret the leaf. Neither falls back to the other, and neither
//! accepts an identity that is not bound to the presented certificate.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, SignatureVerificationAlgorithm, TrustAnchor, UnixTime};

use super::der::{self, LeafFields};
use super::spiffe::{SpiffeId, SpiffeIdError};

/// Who a peer is, as its verified certificate establishes it. Compared
/// whole: two identities are the same only when every part is equal.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PeerIdentity {
    /// An X.509-SVID's SPIFFE ID.
    Spiffe(SpiffeId),
    /// An identity an embedding host's adapter established.
    Host(HostIdentity),
}

impl fmt::Display for PeerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spiffe(id) => write!(f, "{id}"),
            Self::Host(id) => write!(f, "{}#{}", id.scope, id.subject),
        }
    }
}

/// A node identity an embedding host's adapter established: the issuer
/// scope it is valid in and the subject inside that scope.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct HostIdentity {
    /// The issuer scope the identity is valid in, so the same subject
    /// under two issuers is two identities.
    pub scope: String,
    /// The node inside that scope.
    pub subject: String,
}

/// A peer whose certificate chain verified, with the identity it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedPeer {
    /// The identity the certificate establishes.
    pub identity: PeerIdentity,
    /// When the leaf certificate stops being valid, in seconds since the
    /// Unix epoch: authorization founded on it ends there.
    pub not_after: u64,
}

/// Why a presented certificate establishes no node identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PeerError {
    /// The leaf certificate is not encoded as X.509 requires.
    #[error(transparent)]
    Encoding(#[from] der::DerError),
    /// An X.509-SVID carries exactly one URI SAN (X509-SVID §2).
    #[error("an X.509-SVID carries exactly one URI SAN, this one carries {0}")]
    UriCount(usize),
    /// The URI SAN is not text.
    #[error("the URI SAN is not text")]
    UriNotText,
    /// The URI SAN is not a SPIFFE ID.
    #[error("the URI SAN is not a SPIFFE ID: {0}")]
    NotSpiffe(#[from] SpiffeIdError),
    /// An SVID leaf must allow digitalSignature and must not allow
    /// keyCertSign or cRLSign (X509-SVID §4.3).
    #[error("the leaf's key usage does not fit an SVID: {0}")]
    KeyUsage(&'static str),
    /// No trust bundle is configured for the identity's trust domain.
    #[error("no trust bundle for trust domain {0}")]
    UnknownTrustDomain(String),
    /// The chain does not verify against the trusted anchors.
    #[error("the certificate chain does not verify: {0}")]
    Chain(String),
    /// The embedding host's adapter refused the chain.
    #[error("the host identity adapter refused the chain: {0}")]
    Host(String),
}

/// Which end of a TLS connection the peer is: the extended key usage its
/// leaf must allow (RFC 5280 §4.2.1.12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerRole {
    /// The peer connected to this node (id-kp-clientAuth).
    Client,
    /// This node connected to the peer (id-kp-serverAuth).
    Server,
}

impl PeerRole {
    fn usage(self) -> webpki::KeyUsage {
        match self {
            Self::Client => webpki::KeyUsage::client_auth(),
            Self::Server => webpki::KeyUsage::server_auth(),
        }
    }
}

/// Trust anchors a chain must end at.
#[derive(Clone, Default)]
pub struct TrustAnchors(Vec<TrustAnchor<'static>>);

impl fmt::Debug for TrustAnchors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TrustAnchors({} anchors)", self.0.len())
    }
}

impl TrustAnchors {
    /// Anchors from DER certificates.
    ///
    /// # Errors
    ///
    /// A certificate is not usable as a trust anchor.
    pub fn from_der(certs: &[CertificateDer<'_>]) -> Result<Self, PeerError> {
        certs
            .iter()
            .map(|cert| {
                webpki::anchor_from_trusted_cert(cert)
                    .map(|anchor| anchor.to_owned())
                    .map_err(|e| PeerError::Chain(format!("trust anchor: {e}")))
            })
            .collect::<Result<_, _>>()
            .map(Self)
    }

    /// Whether no anchor is configured.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// An embedding host's interpretation of a chain that already verified
/// against the host's anchors on the common provider path.
///
/// The adapter decides who the chain's holder is, in which issuer scope,
/// and whether the host's revocation and validity rules still admit it.
/// It never sees an unverified chain and never replaces verification.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot establish node identities for an embedding host",
    label = "not a host identity adapter",
    note = "implement `identify` to map a verified chain to the host's issuer scope and subject"
)]
pub trait HostIdentityAdapter: Send + Sync + fmt::Debug {
    /// The node identity of the verified `chain` (leaf first) at `now`.
    ///
    /// # Errors
    ///
    /// A reason the host does not admit the chain as a node identity.
    fn identify(&self, chain: &[CertificateDer<'_>], now: UnixTime)
    -> Result<HostIdentity, String>;
}

/// How a node's peers are identified: explicitly one of the two profiles.
#[derive(Clone)]
pub enum PeerProfile {
    /// X.509-SVIDs: each trust domain checked against its own bundle.
    Spiffe {
        /// The trust bundle of every trusted trust domain.
        bundles: HashMap<String, TrustAnchors>,
    },
    /// An embedding host's identities, validated against its anchors.
    Host {
        /// The anchors the host's chains end at.
        anchors: TrustAnchors,
        /// The host's interpretation of a verified chain.
        adapter: Arc<dyn HostIdentityAdapter>,
    },
}

impl fmt::Debug for PeerProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spiffe { bundles } => {
                let mut domains: Vec<&String> = bundles.keys().collect();
                domains.sort();
                f.debug_struct("Spiffe").field("domains", &domains).finish()
            }
            Self::Host { anchors, adapter } => f
                .debug_struct("Host")
                .field("anchors", anchors)
                .field("adapter", adapter)
                .finish(),
        }
    }
}

impl PeerProfile {
    /// Whether `end_entity` presents itself as a node identity of this
    /// profile. On a port shared with clients, a certificate that does
    /// not is left to client authentication; one that does is verified
    /// here and never handed to it.
    #[must_use]
    pub fn claims_node_identity(&self, end_entity: &CertificateDer<'_>) -> bool {
        match self {
            Self::Spiffe { .. } => der::leaf_fields(end_entity).map_or(true, |fields| {
                fields
                    .uris
                    .iter()
                    .any(|uri| uri.starts_with(b"spiffe:") || uri.starts_with(b"SPIFFE:"))
            }),
            // A host's anchors are its own: anything ending there is a node.
            Self::Host { .. } => true,
        }
    }

    /// Verify the chain `end_entity` + `intermediates` presented by a
    /// peer acting as `role`, and establish its node identity.
    ///
    /// # Errors
    ///
    /// The reason the chain establishes no node identity.
    pub fn verify(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
        role: PeerRole,
        algorithms: &[&'static dyn SignatureVerificationAlgorithm],
    ) -> Result<VerifiedPeer, PeerError> {
        let fields = der::leaf_fields(end_entity)?;
        match self {
            Self::Spiffe { bundles } => {
                let id = svid_identity(&fields)?;
                let bundle = bundles
                    .get(id.trust_domain())
                    .ok_or_else(|| PeerError::UnknownTrustDomain(id.trust_domain().to_owned()))?;
                verify_chain(end_entity, intermediates, now, role, bundle, algorithms)?;
                Ok(VerifiedPeer {
                    identity: PeerIdentity::Spiffe(id),
                    not_after: fields.not_after,
                })
            }
            Self::Host { anchors, adapter } => {
                verify_chain(end_entity, intermediates, now, role, anchors, algorithms)?;
                let chain: Vec<CertificateDer<'_>> = core::iter::once(end_entity.clone())
                    .chain(intermediates.iter().cloned())
                    .collect();
                let identity = adapter.identify(&chain, now).map_err(PeerError::Host)?;
                Ok(VerifiedPeer {
                    identity: PeerIdentity::Host(identity),
                    not_after: fields.not_after,
                })
            }
        }
    }
}

/// The SPIFFE ID of an X.509-SVID leaf (X509-SVID §2, §4.3).
fn svid_identity(fields: &LeafFields) -> Result<SpiffeId, PeerError> {
    let [uri] = fields.uris.as_slice() else {
        return Err(PeerError::UriCount(fields.uris.len()));
    };
    let text = core::str::from_utf8(uri).map_err(|_| PeerError::UriNotText)?;
    let id = SpiffeId::parse(text)?;
    let usage = fields
        .key_usage
        .ok_or(PeerError::KeyUsage("keyUsage is missing"))?;
    if !usage.digital_signature {
        return Err(PeerError::KeyUsage("digitalSignature is not set"));
    }
    if usage.key_cert_sign || usage.crl_sign {
        return Err(PeerError::KeyUsage("keyCertSign or cRLSign is set"));
    }
    Ok(id)
}

/// Verify the chain against `anchors` for `role`'s key usage. Webpki
/// refuses a leaf that is a CA (RFC 5280 §4.2.1.9).
fn verify_chain(
    end_entity: &CertificateDer<'_>,
    intermediates: &[CertificateDer<'_>],
    now: UnixTime,
    role: PeerRole,
    anchors: &TrustAnchors,
    algorithms: &[&'static dyn SignatureVerificationAlgorithm],
) -> Result<(), PeerError> {
    let cert =
        webpki::EndEntityCert::try_from(end_entity).map_err(|e| PeerError::Chain(e.to_string()))?;
    cert.verify_for_usage(
        algorithms,
        &anchors.0,
        intermediates,
        now,
        role.usage(),
        None,
        None,
    )
    .map(|_| ())
    .map_err(|e| PeerError::Chain(e.to_string()))
}

#[cfg(test)]
mod tests;
