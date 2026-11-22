//! Who a write acts for, as the log carries it: the principal a session was
//! opened as, which every member checks the write's grants against when it
//! applies the entry; and the records of the deployment catalog those grants
//! are, each with the commit timestamp of the change that produced it.
//!
//! The catalog lives among the deployment's records. A grant names who holds
//! it: a principal, or a database role whose holders carry every right the
//! role has. A role has a home namespace, may be exclusive among a named set
//! of eligible principals, and is held through membership records.

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::namespace::NamespaceId;

/// Most bytes a principal's subject or trust scope may hold: far above any
/// issuer's identifier, and small enough that a log entry naming one stays
/// small.
pub const MAX_PRINCIPAL_PART_BYTES: usize = 1024;

/// A verified subject and the issuer or trust scope that names it: the
/// identity a grant is held by. Two principals are the same exactly when
/// both parts are.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PrincipalId {
    /// The stable subject, as its issuer names it.
    pub subject: Arc<str>,
    /// The issuer or trust scope the subject belongs to.
    pub trust_scope: Arc<str>,
}

/// A principal part that is empty or longer than
/// [`MAX_PRINCIPAL_PART_BYTES`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("a principal's {part} holds {len} bytes; it must hold 1 to {MAX_PRINCIPAL_PART_BYTES}")]
pub struct BadPrincipal {
    /// Which part: `"subject"` or `"trust scope"`.
    pub part: &'static str,
    /// Its length in bytes.
    pub len: usize,
}

impl PrincipalId {
    /// The principal `subject` of `trust_scope`.
    ///
    /// # Errors
    ///
    /// [`BadPrincipal`] for an empty part or one past
    /// [`MAX_PRINCIPAL_PART_BYTES`].
    pub fn new(subject: &str, trust_scope: &str) -> Result<Self, BadPrincipal> {
        check_part("subject", subject)?;
        check_part("trust scope", trust_scope)?;
        Ok(Self {
            subject: Arc::from(subject),
            trust_scope: Arc::from(trust_scope),
        })
    }

    /// Bytes both parts hold.
    #[must_use]
    pub fn len(&self) -> usize {
        self.subject.len() + self.trust_scope.len()
    }

    /// Never true: both parts hold at least one byte.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }
}

/// Serialized as `(subject, trust_scope)`.
impl serde::Serialize for PrincipalId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&(&*self.subject, &*self.trust_scope), serializer)
    }
}

/// Read back from `(subject, trust_scope)`, refusing what [`PrincipalId::new`]
/// refuses.
impl<'de> serde::Deserialize<'de> for PrincipalId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (subject, trust_scope): (alloc::string::String, alloc::string::String) =
            serde::Deserialize::deserialize(deserializer)?;
        Self::new(&subject, &trust_scope).map_err(serde::de::Error::custom)
    }
}

fn check_part(part: &'static str, value: &str) -> Result<(), BadPrincipal> {
    if value.is_empty() || value.len() > MAX_PRINCIPAL_PART_BYTES {
        return Err(BadPrincipal {
            part,
            len: value.len(),
        });
    }
    Ok(())
}

impl PrincipalId {
    /// The principal's bytes inside a catalog key: each part as a two-byte
    /// big-endian length and its bytes, so one principal's keys never
    /// prefix another's.
    fn encode_into(&self, out: &mut Vec<u8>) {
        for part in [&*self.subject, &*self.trust_scope] {
            // At most MAX_PRINCIPAL_PART_BYTES, far below u16::MAX.
            out.extend_from_slice(&(part.len() as u16).to_be_bytes());
            out.extend_from_slice(part.as_bytes());
        }
    }

    /// The principal [`Self::encode_into`] wrote at the start of `bytes`,
    /// and what follows it.
    fn decode_prefix(bytes: &[u8]) -> Option<(Self, &[u8])> {
        let part = |bytes: &[u8]| -> Option<(String, usize)> {
            let (len, rest) = bytes.split_first_chunk::<2>()?;
            let len = usize::from(u16::from_be_bytes(*len));
            let text = core::str::from_utf8(rest.get(..len)?).ok()?;
            Some((text.into(), 2 + len))
        };
        let (subject, used) = part(bytes)?;
        let (trust_scope, more) = part(&bytes[used..])?;
        let principal = Self::new(&subject, &trust_scope).ok()?;
        Some((principal, &bytes[used + more..]))
    }
}

/// A right on one namespace. Each includes the ones below it.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum NamespaceRole {
    /// Reads its data.
    Reader,
    /// Reads and writes its data.
    Writer,
    /// Writes its data, changes its schema and indexes, grants and revokes
    /// rights on it, and drops it.
    Admin,
}

impl NamespaceRole {
    /// The role's name as a statement spells it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Reader => "reader",
            Self::Writer => "writer",
            Self::Admin => "admin",
        }
    }

    /// The role a statement names: `reader`, `writer` or `admin`,
    /// case-insensitive.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "reader" => Some(Self::Reader),
            "writer" => Some(Self::Writer),
            "admin" => Some(Self::Admin),
            _ => None,
        }
    }

    const fn tag(self) -> u8 {
        match self {
            Self::Reader => 1,
            Self::Writer => 2,
            Self::Admin => 3,
        }
    }

    const fn from_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Reader),
            2 => Some(Self::Writer),
            3 => Some(Self::Admin),
            _ => None,
        }
    }
}

impl core::fmt::Display for NamespaceRole {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// A database role, by the identity the catalog handed it. Never handed out
/// twice, so a role dropped and created again under its name is another
/// role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RoleId(u64);

impl RoleId {
    /// The id from its number.
    #[must_use]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// The id's number.
    #[must_use]
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

/// Who holds a grant on a namespace.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Grantee {
    /// A principal, in its own name.
    Principal(PrincipalId),
    /// A database role: every holder of the role has the right.
    Role(RoleId),
}

/// A right on one namespace and the commit timestamp of the change that
/// produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantRecord {
    /// The right.
    pub role: NamespaceRole,
    /// The commit timestamp of the change that made the grant what it is.
    pub changed_ts: u64,
}

impl GrantRecord {
    /// The record's bytes: the role's tag, then the timestamp big-endian.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.push(self.role.tag());
        bytes.extend_from_slice(&self.changed_ts.to_be_bytes());
        bytes
    }

    /// The record `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&tag, rest) = bytes.split_first()?;
        Some(Self {
            role: NamespaceRole::from_tag(tag)?,
            changed_ts: u64::from_be_bytes(rest.try_into().ok()?),
        })
    }
}

/// The rights a principal holds over the whole deployment rather than one
/// namespace, and the commit timestamp of the change that produced them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeploymentRights {
    /// Creates namespaces and lists the deployment's namespaces, becoming
    /// admin of each it creates. No right on any existing namespace.
    pub create_namespace: bool,
    /// Every right on every namespace of the deployment.
    pub admin: bool,
    /// The commit timestamp of the change that made them what they are.
    pub changed_ts: u64,
}

impl DeploymentRights {
    /// The record's bytes: a flags byte, then the timestamp big-endian.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.push(u8::from(self.create_namespace) | (u8::from(self.admin) << 1));
        bytes.extend_from_slice(&self.changed_ts.to_be_bytes());
        bytes
    }

    /// The record `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&flags, rest) = bytes.split_first()?;
        if flags & !0b11 != 0 {
            return None;
        }
        Some(Self {
            create_namespace: flags & 1 != 0,
            admin: flags & 2 != 0,
            changed_ts: u64::from_be_bytes(rest.try_into().ok()?),
        })
    }
}

/// The longest name a database role may have, in bytes.
pub const MAX_ROLE_NAME_LEN: usize = 63;

/// The catalog's record of one database role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleRecord {
    /// Its name, unique within its home namespace.
    pub name: String,
    /// The namespace it was created in, whose admin alone changes who may
    /// hold it.
    pub home: NamespaceId,
    /// For an exclusive role, the principals that may hold it, in order:
    /// exactly one of them holds it, and any of them may take it. `None`
    /// for a role any number of principals may hold.
    pub eligible: Option<Vec<PrincipalId>>,
    /// The commit timestamp of the change that last changed the role or who
    /// holds it.
    pub changed_ts: u64,
}

impl RoleRecord {
    /// The record's bytes: the home namespace, the timestamp, an
    /// exclusivity byte, the eligible principals with their count, then the
    /// name.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32 + self.name.len());
        bytes.extend_from_slice(&self.home.as_raw().to_be_bytes());
        bytes.extend_from_slice(&self.changed_ts.to_be_bytes());
        match &self.eligible {
            None => bytes.push(0),
            Some(eligible) => {
                bytes.push(1);
                bytes.extend_from_slice(&(eligible.len() as u32).to_be_bytes());
                for principal in eligible {
                    principal.encode_into(&mut bytes);
                }
            }
        }
        bytes.extend_from_slice(self.name.as_bytes());
        bytes
    }

    /// The record `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (home, rest) = bytes.split_first_chunk::<8>()?;
        let (changed_ts, rest) = rest.split_first_chunk::<8>()?;
        let (&exclusive, mut rest) = rest.split_first()?;
        let eligible = match exclusive {
            0 => None,
            1 => {
                let (count, more) = rest.split_first_chunk::<4>()?;
                let count = u32::from_be_bytes(*count) as usize;
                // Every principal takes at least six bytes.
                if count > more.len() / 6 {
                    return None;
                }
                rest = more;
                let mut eligible = Vec::with_capacity(count);
                for _ in 0..count {
                    let (principal, more) = PrincipalId::decode_prefix(rest)?;
                    eligible.push(principal);
                    rest = more;
                }
                Some(eligible)
            }
            _ => return None,
        };
        let name = core::str::from_utf8(rest).ok()?;
        if name.is_empty() || name.len() > MAX_ROLE_NAME_LEN {
            return None;
        }
        Some(Self {
            name: name.into(),
            home: NamespaceId::from_raw(u64::from_be_bytes(*home)),
            eligible,
            changed_ts: u64::from_be_bytes(*changed_ts),
        })
    }
}

/// Prefix of a principal's grant on a namespace:
/// `acl:np:<namespace><principal>`.
pub const PRINCIPAL_GRANT_PREFIX: &[u8] = b"acl:np:";
/// Prefix of a role's grant on a namespace: `acl:nr:<namespace><role>`.
pub const ROLE_GRANT_PREFIX: &[u8] = b"acl:nr:";
/// Prefix of a principal's deployment rights: `acl:dp:<principal>`.
pub const DEPLOYMENT_RIGHTS_PREFIX: &[u8] = b"acl:dp:";
/// Prefix of a role's record: `acl:role:<role>`.
pub const ROLE_RECORD_PREFIX: &[u8] = b"acl:role:";
/// Prefix of the index of roles by name: `acl:rname:<home><name>`.
pub const ROLE_NAME_PREFIX: &[u8] = b"acl:rname:";
/// Prefix of a role's holders: `acl:rh:<role><principal>`.
pub const ROLE_HOLDER_PREFIX: &[u8] = b"acl:rh:";
/// Prefix of the roles a principal holds: `acl:ph:<principal><role>`.
pub const HELD_ROLE_PREFIX: &[u8] = b"acl:ph:";
/// The catalog's record of the next role id it hands out.
pub const ROLE_ALLOCATOR_KEY: &[u8] = b"acl:role-next";

/// Whether `key` of the deployment's records belongs to the grant catalog.
#[must_use]
pub fn is_access_catalog_key(key: &[u8]) -> bool {
    key.starts_with(b"acl:")
}

/// The key of `grantee`'s grant on `namespace`.
#[must_use]
pub fn grant_key(namespace: NamespaceId, grantee: &Grantee) -> Vec<u8> {
    let mut key = Vec::with_capacity(16 + 8 + 2 * MAX_PRINCIPAL_PART_BYTES.min(64));
    match grantee {
        Grantee::Principal(principal) => {
            key.extend_from_slice(PRINCIPAL_GRANT_PREFIX);
            key.extend_from_slice(&namespace.as_raw().to_be_bytes());
            principal.encode_into(&mut key);
        }
        Grantee::Role(role) => {
            key.extend_from_slice(ROLE_GRANT_PREFIX);
            key.extend_from_slice(&namespace.as_raw().to_be_bytes());
            key.extend_from_slice(&role.as_raw().to_be_bytes());
        }
    }
    key
}

/// The prefix of every grant on `namespace` of one kind: principals' with
/// [`PRINCIPAL_GRANT_PREFIX`], roles' with [`ROLE_GRANT_PREFIX`].
#[must_use]
pub fn namespace_grants_prefix(kind: &[u8], namespace: NamespaceId) -> Vec<u8> {
    let mut key = Vec::with_capacity(kind.len() + 8);
    key.extend_from_slice(kind);
    key.extend_from_slice(&namespace.as_raw().to_be_bytes());
    key
}

/// The grantee a [`grant_key`] names, with its namespace; `None` for any
/// other key.
#[must_use]
pub fn decode_grant_key(key: &[u8]) -> Option<(NamespaceId, Grantee)> {
    if let Some(rest) = key.strip_prefix(PRINCIPAL_GRANT_PREFIX) {
        let (namespace, rest) = rest.split_first_chunk::<8>()?;
        let (principal, rest) = PrincipalId::decode_prefix(rest)?;
        return rest.is_empty().then(|| {
            (
                NamespaceId::from_raw(u64::from_be_bytes(*namespace)),
                Grantee::Principal(principal),
            )
        });
    }
    let rest = key.strip_prefix(ROLE_GRANT_PREFIX)?;
    let (namespace, rest) = rest.split_first_chunk::<8>()?;
    let role: [u8; 8] = rest.try_into().ok()?;
    Some((
        NamespaceId::from_raw(u64::from_be_bytes(*namespace)),
        Grantee::Role(RoleId::from_raw(u64::from_be_bytes(role))),
    ))
}

/// The key of `principal`'s deployment rights.
#[must_use]
pub fn deployment_rights_key(principal: &PrincipalId) -> Vec<u8> {
    let mut key = Vec::with_capacity(DEPLOYMENT_RIGHTS_PREFIX.len() + 4 + principal.len());
    key.extend_from_slice(DEPLOYMENT_RIGHTS_PREFIX);
    principal.encode_into(&mut key);
    key
}

/// The principal a [`deployment_rights_key`] names; `None` for any other
/// key.
#[must_use]
pub fn decode_deployment_rights_key(key: &[u8]) -> Option<PrincipalId> {
    let (principal, rest) =
        PrincipalId::decode_prefix(key.strip_prefix(DEPLOYMENT_RIGHTS_PREFIX)?)?;
    rest.is_empty().then_some(principal)
}

/// The key of `role`'s record.
#[must_use]
pub fn role_record_key(role: RoleId) -> Vec<u8> {
    let mut key = Vec::with_capacity(ROLE_RECORD_PREFIX.len() + 8);
    key.extend_from_slice(ROLE_RECORD_PREFIX);
    key.extend_from_slice(&role.as_raw().to_be_bytes());
    key
}

/// The key naming the role called `name` in `home`.
#[must_use]
pub fn role_name_key(home: NamespaceId, name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(ROLE_NAME_PREFIX.len() + 8 + name.len());
    key.extend_from_slice(ROLE_NAME_PREFIX);
    key.extend_from_slice(&home.as_raw().to_be_bytes());
    key.extend_from_slice(name.as_bytes());
    key
}

/// The key recording that `principal` holds `role`, by role.
#[must_use]
pub fn role_holder_key(role: RoleId, principal: &PrincipalId) -> Vec<u8> {
    let mut key = role_holders_prefix(role);
    principal.encode_into(&mut key);
    key
}

/// The prefix of every holder of `role`.
#[must_use]
pub fn role_holders_prefix(role: RoleId) -> Vec<u8> {
    let mut key = Vec::with_capacity(ROLE_HOLDER_PREFIX.len() + 8 + 64);
    key.extend_from_slice(ROLE_HOLDER_PREFIX);
    key.extend_from_slice(&role.as_raw().to_be_bytes());
    key
}

/// The holder a [`role_holder_key`] names; `None` for any other key.
#[must_use]
pub fn decode_role_holder_key(key: &[u8]) -> Option<(RoleId, PrincipalId)> {
    let rest = key.strip_prefix(ROLE_HOLDER_PREFIX)?;
    let (role, rest) = rest.split_first_chunk::<8>()?;
    let (principal, rest) = PrincipalId::decode_prefix(rest)?;
    rest.is_empty()
        .then(|| (RoleId::from_raw(u64::from_be_bytes(*role)), principal))
}

/// The key recording that `principal` holds `role`, by principal.
#[must_use]
pub fn held_role_key(principal: &PrincipalId, role: RoleId) -> Vec<u8> {
    let mut key = held_roles_prefix(principal);
    key.extend_from_slice(&role.as_raw().to_be_bytes());
    key
}

/// The prefix of every role `principal` holds.
#[must_use]
pub fn held_roles_prefix(principal: &PrincipalId) -> Vec<u8> {
    let mut key = Vec::with_capacity(HELD_ROLE_PREFIX.len() + 4 + principal.len() + 8);
    key.extend_from_slice(HELD_ROLE_PREFIX);
    principal.encode_into(&mut key);
    key
}

/// The role a [`held_role_key`] of `principal` names; `None` for any other
/// key.
#[must_use]
pub fn decode_held_role_key(principal: &PrincipalId, key: &[u8]) -> Option<RoleId> {
    let role: [u8; 8] = key
        .strip_prefix(held_roles_prefix(principal).as_slice())?
        .try_into()
        .ok()?;
    Some(RoleId::from_raw(u64::from_be_bytes(role)))
}

/// The right a refused unit lacked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MissingRight {
    /// A right on one namespace.
    Namespace {
        /// The namespace.
        namespace: NamespaceId,
        /// The least role that would have sufficed.
        role: NamespaceRole,
    },
    /// The right to create namespaces.
    CreateNamespace,
    /// Every right on the whole deployment.
    DeploymentAdmin,
}

/// Why a unit was applied as a refusal: its principal lacked a right at the
/// unit's place in the log.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{principal:?} lacks {missing:?}")]
pub struct AccessRefusal {
    /// The principal the unit acted for.
    pub principal: PrincipalId,
    /// The right it lacked.
    pub missing: MissingRight,
}

impl AccessRefusal {
    /// The refusal's bytes: what lacked, then the principal.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + self.principal.len());
        match self.missing {
            MissingRight::Namespace { namespace, role } => {
                bytes.push(role.tag());
                bytes.extend_from_slice(&namespace.as_raw().to_be_bytes());
            }
            MissingRight::CreateNamespace => bytes.push(0x10),
            MissingRight::DeploymentAdmin => bytes.push(0x11),
        }
        self.principal.encode_into(&mut bytes);
        bytes
    }

    /// The refusal `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&tag, rest) = bytes.split_first()?;
        let (missing, rest) = match tag {
            0x10 => (MissingRight::CreateNamespace, rest),
            0x11 => (MissingRight::DeploymentAdmin, rest),
            tag => {
                let role = NamespaceRole::from_tag(tag)?;
                let (namespace, rest) = rest.split_first_chunk::<8>()?;
                (
                    MissingRight::Namespace {
                        namespace: NamespaceId::from_raw(u64::from_be_bytes(*namespace)),
                        role,
                    },
                    rest,
                )
            }
        };
        let (principal, rest) = PrincipalId::decode_prefix(rest)?;
        rest.is_empty().then_some(Self { principal, missing })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
