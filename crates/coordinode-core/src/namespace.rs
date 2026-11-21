//! The namespace a piece of data belongs to.
//!
//! A namespace is the tenancy unit of a deployment: its data lives in a set
//! of trees of its own, and keys carry no namespace. What tells a write's
//! namespace is the unit it travels in, which names it before the writes it
//! holds of that namespace.

use alloc::string::String;
use alloc::vec::Vec;

/// Bytes of the token a namespace creation carries.
pub const NAMESPACE_TOKEN_LEN: usize = 16;

/// The name of the one namespace a store of one namespace creates for
/// itself when it is initialized as its own store.
pub const DEFAULT_NAMESPACE: &str = "default";

/// The longest name a namespace may have, in bytes.
pub const MAX_NAMESPACE_NAME_LEN: usize = 63;

/// The deployment catalog's record of the next namespace id it hands out.
/// Ids start at 1 and are never handed out twice.
pub const NAMESPACE_ALLOCATOR_KEY: &[u8] = b"ns:next";

/// Prefix of the catalog's record of each namespace, by id.
pub const NAMESPACE_RECORD_PREFIX: &[u8] = b"ns:id:";

/// Prefix of the catalog's index of live namespaces by name.
pub const NAMESPACE_NAME_PREFIX: &[u8] = b"ns:name:";

/// The catalog key of namespace `id`'s record: ids in big-endian order, so
/// the records list by id.
#[must_use]
pub fn namespace_record_key(id: NamespaceId) -> Vec<u8> {
    let mut key = Vec::with_capacity(NAMESPACE_RECORD_PREFIX.len() + 8);
    key.extend_from_slice(NAMESPACE_RECORD_PREFIX);
    key.extend_from_slice(&id.as_raw().to_be_bytes());
    key
}

/// The id a record key of [`namespace_record_key`] names; `None` for any
/// other key.
#[must_use]
pub fn decode_namespace_record_key(key: &[u8]) -> Option<NamespaceId> {
    let id = key.strip_prefix(NAMESPACE_RECORD_PREFIX)?;
    Some(NamespaceId(u64::from_be_bytes(id.try_into().ok()?)))
}

/// Whether `key` of the deployment's records is a record of the namespace
/// catalog: its id allocator, a namespace's record or its name index.
#[must_use]
pub fn is_namespace_catalog_key(key: &[u8]) -> bool {
    key == NAMESPACE_ALLOCATOR_KEY
        || key.starts_with(NAMESPACE_RECORD_PREFIX)
        || key.starts_with(NAMESPACE_NAME_PREFIX)
}

/// The catalog key naming the live namespace called `name`.
#[must_use]
pub fn namespace_name_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(NAMESPACE_NAME_PREFIX.len() + name.len());
    key.extend_from_slice(NAMESPACE_NAME_PREFIX);
    key.extend_from_slice(name.as_bytes());
    key
}

/// Why a name is not one a namespace may have.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NamespaceNameError {
    /// The name is empty.
    #[error("a namespace name cannot be empty")]
    Empty,
    /// The name is longer than [`MAX_NAMESPACE_NAME_LEN`] bytes.
    #[error("a namespace name holds at most {MAX_NAMESPACE_NAME_LEN} bytes, not {len}")]
    TooLong {
        /// Its length in bytes.
        len: usize,
    },
    /// The name does not start with a lowercase ASCII letter.
    #[error("a namespace name starts with a lowercase letter, not {first:?}")]
    Start {
        /// Its first character.
        first: char,
    },
    /// The name holds a character other than a lowercase ASCII letter, a
    /// digit, `_` or `-`.
    #[error("a namespace name holds lowercase letters, digits, '_' and '-', not {found:?}")]
    Character {
        /// The first character not admitted.
        found: char,
    },
}

/// Admit `name` as a namespace's name: a lowercase ASCII letter, then
/// lowercase letters, digits, `_` and `-`, at most [`MAX_NAMESPACE_NAME_LEN`]
/// bytes. One spelling per name, so two names never differ only in case.
///
/// # Errors
///
/// The [`NamespaceNameError`] the name breaks.
pub fn validate_namespace_name(name: &str) -> Result<(), NamespaceNameError> {
    let Some(first) = name.chars().next() else {
        return Err(NamespaceNameError::Empty);
    };
    if name.len() > MAX_NAMESPACE_NAME_LEN {
        return Err(NamespaceNameError::TooLong { len: name.len() });
    }
    if !first.is_ascii_lowercase() {
        return Err(NamespaceNameError::Start { first });
    }
    match name
        .chars()
        .find(|c| !(c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_' || *c == '-'))
    {
        Some(found) => Err(NamespaceNameError::Character { found }),
        None => Ok(()),
    }
}

/// Where a namespace stands in the deployment's catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespaceState {
    /// It serves its data.
    Live,
    /// It was dropped: nothing reads or writes it, and its trees are removed;
    /// the record stays until they are gone, so its id is never handed out
    /// again and its leftovers are known for what they are.
    Dropping,
}

/// The deployment catalog's record of one namespace incarnation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceRecord {
    /// Its name while it was live.
    pub name: String,
    /// Where it stands.
    pub state: NamespaceState,
    /// The token of the creation that made it.
    pub token: [u8; NAMESPACE_TOKEN_LEN],
}

impl NamespaceRecord {
    /// The record's bytes: its state, its token, then its name.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(1 + NAMESPACE_TOKEN_LEN + self.name.len());
        bytes.push(match self.state {
            NamespaceState::Live => 0,
            NamespaceState::Dropping => 1,
        });
        bytes.extend_from_slice(&self.token);
        bytes.extend_from_slice(self.name.as_bytes());
        bytes
    }

    /// The record `bytes` hold; `None` for bytes no [`Self::encode`] wrote.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let (&state, rest) = bytes.split_first()?;
        let state = match state {
            0 => NamespaceState::Live,
            1 => NamespaceState::Dropping,
            _ => return None,
        };
        let (token, name) = rest.split_first_chunk::<NAMESPACE_TOKEN_LEN>()?;
        let name = core::str::from_utf8(name).ok()?;
        validate_namespace_name(name).ok()?;
        Some(Self {
            name: name.into(),
            state,
            token: *token,
        })
    }
}

/// A namespace incarnation of one deployment. Allocated by the deployment's
/// catalog and never reused within it: a namespace dropped and created again
/// under the same name is a new incarnation with a new id, so nothing bound
/// to the old one (a dictionary handle, a tree, a delayed worker) can reach
/// the new one.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct NamespaceId(u64);

impl NamespaceId {
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

    /// The name of the namespace's directory in a store: fixed width, so a
    /// listing sorts by id.
    #[must_use]
    pub fn dir_name(self) -> String {
        alloc::format!("{:016x}", self.0)
    }

    /// The id a namespace directory is named for, `None` for a name no
    /// namespace directory has.
    #[must_use]
    pub fn from_dir_name(name: &str) -> Option<Self> {
        (name.len() == 16 && name.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| u64::from_str_radix(name, 16).ok().map(Self))
            .flatten()
    }
}

impl core::fmt::Display for NamespaceId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "namespace {}", self.0)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
