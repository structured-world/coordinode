//! A group member as the membership records it: where the others reach it
//! and the node identity it proves on the inter-node channel.

use core::fmt;

use coordinode_wire::peer::{HostIdentity, PeerIdentity, SpiffeId, SpiffeIdError};
use serde::{Deserialize, Serialize};

/// The node identity recorded for a member, in the form the membership
/// stores and replicates. The same identity as the verified
/// [`PeerIdentity`] a connection establishes, compared whole.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemberIdentity {
    /// A SPIFFE ID, kept as the whole ID.
    Spiffe(String),
    /// An identity an embedding host's adapter establishes.
    Host {
        /// The issuer scope the identity is valid in.
        scope: String,
        /// The node inside that scope.
        subject: String,
    },
}

impl MemberIdentity {
    /// The identity a connection must prove to be this member.
    ///
    /// # Errors
    ///
    /// A recorded SPIFFE ID that does not parse; nothing can prove it.
    pub fn to_peer(&self) -> Result<PeerIdentity, SpiffeIdError> {
        Ok(match self {
            Self::Spiffe(id) => PeerIdentity::Spiffe(SpiffeId::parse(id)?),
            Self::Host { scope, subject } => PeerIdentity::Host(HostIdentity {
                scope: scope.clone(),
                subject: subject.clone(),
            }),
        })
    }
}

impl From<&PeerIdentity> for MemberIdentity {
    fn from(identity: &PeerIdentity) -> Self {
        match identity {
            PeerIdentity::Spiffe(id) => Self::Spiffe(id.as_str().to_owned()),
            PeerIdentity::Host(HostIdentity { scope, subject }) => Self::Host {
                scope: scope.clone(),
                subject: subject.clone(),
            },
        }
    }
}

impl fmt::Display for MemberIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spiffe(id) => f.write_str(id),
            Self::Host { scope, subject } => write!(f, "{scope}#{subject}"),
        }
    }
}

/// A member of a group: the address the others dial, which is only a hint,
/// and the identity it must prove there.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemberNode {
    /// Where the other members dial this one. Never identity: a member
    /// that moves is reached at its new address with no membership change.
    pub addr: String,
    /// The node identity the member proves on the inter-node channel.
    /// `None` for a member that has never opened the channel (the only
    /// member of a group that has not grown); such a member is admitted to
    /// no peer service.
    pub identity: Option<MemberIdentity>,
}

impl MemberNode {
    /// A member at `addr` proving `identity`.
    #[must_use]
    pub fn new(addr: String, identity: Option<MemberIdentity>) -> Self {
        Self { addr, identity }
    }
}

impl fmt::Display for MemberNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.identity {
            Some(identity) => write!(f, "{identity} at {}", self.addr),
            None => write!(f, "unidentified at {}", self.addr),
        }
    }
}

#[cfg(test)]
mod tests;
