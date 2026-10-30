//! Label identity: the catalog-owned id a node record stores for each of its
//! labels, and the catalog records that bind names to ids.
//!
//! A label's name is bound to one identity at a time. Declaring a label used
//! so far without a definition, or changing its definition, keeps the
//! identity; dropping the label retires it, and a label created later under
//! the same name gets a new one. A record stores ids, so a record of the
//! dropped label is never read as one of its successor.
//!
//! Three records live in the Schema partition:
//!
//! - the name binding at `schema:current_revision:label:<name>`, a
//!   [`LabelBinding`]: the identity, whether it is declared, implicit or
//!   retired, and the revision of its definition, stored under the identity
//!   ([`crate::schema::definition::encode_label_schema_key`]);
//! - the identity record at `schema:label_id:<id>`, the name the id was
//!   allocated for, written once and never removed: it is what keeps an id
//!   from being handed out twice and resolves the id of any stored record;
//! - the allocation frontier at `schema:label_id_frontier`, the largest id
//!   handed out.

use core::num::NonZeroU32;

/// The identity of one incarnation of a label, as node records store it.
/// Id 0 is never allocated.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct LabelId(NonZeroU32);

impl LabelId {
    /// The id `raw`, or `None` for 0.
    pub const fn new(raw: u32) -> Option<Self> {
        match NonZeroU32::new(raw) {
            Some(raw) => Some(Self(raw)),
            None => None,
        }
    }

    /// The raw id.
    pub const fn get(self) -> u32 {
        self.0.get()
    }

    /// The id after this one, or `None` once the 32-bit space is used up.
    pub fn next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

impl core::fmt::Display for LabelId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// What a label name is bound to now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingState {
    /// Used by nodes without a definition: a FLEXIBLE type label.
    Implicit,
    /// Declared: the definition at the binding's revision is current.
    Declared,
    /// Dropped. The identity is never handed out again; a label created
    /// under the name later gets a new one.
    Retired,
}

/// The value of a name binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LabelBinding {
    /// The identity the name is bound to.
    pub id: LabelId,
    /// The identity's last definition revision; 0 when it never had one.
    pub revision: u64,
    /// Declared, implicit or retired.
    pub state: BindingState,
}

/// Length of an encoded [`LabelBinding`]: id, revision, state.
const BINDING_LEN: usize = 4 + 8 + 1;

impl LabelBinding {
    /// The revision of the current definition; 0 when the label has none.
    pub fn declared_revision(&self) -> u64 {
        match self.state {
            BindingState::Declared => self.revision,
            BindingState::Implicit | BindingState::Retired => 0,
        }
    }

    /// The identity nodes use the name for now; `None` once retired.
    pub fn live_id(&self) -> Option<LabelId> {
        match self.state {
            BindingState::Implicit | BindingState::Declared => Some(self.id),
            BindingState::Retired => None,
        }
    }

    /// The stored bytes.
    pub fn encode(&self) -> [u8; BINDING_LEN] {
        let mut out = [0u8; BINDING_LEN];
        out[..4].copy_from_slice(&self.id.get().to_be_bytes());
        out[4..12].copy_from_slice(&self.revision.to_be_bytes());
        out[12] = match self.state {
            BindingState::Implicit => 0,
            BindingState::Declared => 1,
            BindingState::Retired => 2,
        };
        out
    }

    /// Decode stored bytes; `None` when they are not a binding.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let bytes: &[u8; BINDING_LEN] = bytes.try_into().ok()?;
        let id = LabelId::new(u32::from_be_bytes(bytes[..4].try_into().ok()?))?;
        let revision = u64::from_be_bytes(bytes[4..12].try_into().ok()?);
        let state = match bytes[12] {
            0 => BindingState::Implicit,
            1 => BindingState::Declared,
            2 => BindingState::Retired,
            _ => return None,
        };
        Some(Self {
            id,
            revision,
            state,
        })
    }
}

/// Key prefix of the identity records. Each is written once, and the newest
/// identity issued is the last of them: allocation reads that, never a
/// separately kept counter a commit could move back.
pub const LABEL_ID_KEY_PREFIX: &[u8] = b"schema:label_id:";

/// Key of the identity record of `id`. Big-endian, so the records sort by id.
pub fn label_id_key(id: LabelId) -> [u8; LABEL_ID_KEY_PREFIX.len() + 4] {
    let mut key = [0u8; LABEL_ID_KEY_PREFIX.len() + 4];
    key[..LABEL_ID_KEY_PREFIX.len()].copy_from_slice(LABEL_ID_KEY_PREFIX);
    key[LABEL_ID_KEY_PREFIX.len()..].copy_from_slice(&id.get().to_be_bytes());
    key
}

/// The id an identity record key names, or `None` for any other key.
pub fn decode_label_id_key(key: &[u8]) -> Option<LabelId> {
    let raw: [u8; 4] = key.strip_prefix(LABEL_ID_KEY_PREFIX)?.try_into().ok()?;
    LabelId::new(u32::from_be_bytes(raw))
}

/// The authority that publishes label identities a write uses for the first
/// time.
///
/// A database implements this over its ordered write path: a binding is a
/// proposal whose application assigns the identity, outside the transaction
/// that needs it, so two transactions first using one label converge on one
/// identity instead of one of them aborting. An identity bound for a write
/// that later rolls back stays bound and empty, as a field binding does.
#[diagnostic::on_unimplemented(
    message = "`{Self}` cannot publish label identities",
    label = "not a label registrar",
    note = "an embedded `Database` provides one beside its field registrar"
)]
pub trait LabelRegistrar: Send + Sync {
    /// The live identities of `names`, in order, binding the ones that have
    /// none. Every returned binding is applied when this returns.
    ///
    /// # Errors
    ///
    /// The batch is invalid, the identity space is exhausted, or the binding
    /// could not be published.
    fn bind_labels(
        &self,
        names: &[&str],
    ) -> Result<alloc::vec::Vec<LabelId>, crate::graph::intern::DictionaryError>;
}

/// The first id a database hands out.
pub const FIRST_LABEL_ID: LabelId = LabelId(NonZeroU32::MIN);

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
