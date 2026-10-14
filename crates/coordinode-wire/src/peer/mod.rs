//! Inter-node peer identity: who is at the other end of an authenticated
//! connection, established from the certificate it proved possession of.
//!
//! A verified identity says only who the peer is. Whether it may take part
//! in a group, and in which role, is decided afterwards against the
//! deployment's membership and admission state.

mod context;
mod der;
mod spiffe;
#[cfg(test)]
mod test_pki;
mod transport;
mod verify;

pub use context::{ContextError, InboundAuth, SecurityContext};
pub use der::DerError;
pub use spiffe::{SpiffeId, SpiffeIdError};
pub use transport::{
    Accept, AuthenticatedPeer, DialError, Guarded, InboundStream, Incoming, PeerAdmission,
    PeerConnectInfo, PeerConnections, PeerTarget, incoming,
};
pub use verify::{
    HostIdentity, HostIdentityAdapter, PeerError, PeerIdentity, PeerProfile, PeerRole,
    TrustAnchors, VerifiedPeer,
};
