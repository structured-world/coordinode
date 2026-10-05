//! Identities of an index: the stable logical object and its physical
//! representations.
//!
//! An [`IndexId`] names the logical index for its whole life: rename,
//! description edits, rebuilds and a change of maintenance profile keep it,
//! and only DROP followed by CREATE makes a new one. A [`GenerationId`] names
//! one physical representation of it, the entries one build wrote under one
//! key layout and interpretation; a rebuild writes a new generation. Entry
//! keys carry the generation alone, which identifies its index.
//!
//! Both are allocated from one catalog counter each, in the transaction that
//! publishes the definition, and never reused within the catalog: a dropped
//! index's numbers stay taken, so a delayed writer or cleanup bound to them
//! cannot reach an object created later.

use serde::{Deserialize, Serialize};

/// The stable identity of a logical index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct IndexId(u64);

impl IndexId {
    /// The identity numbered `raw`.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Its number.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for IndexId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "index#{}", self.0)
    }
}

/// The identity of one physical representation of an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct GenerationId(u64);

impl GenerationId {
    /// The generation numbered `raw`.
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Its number.
    pub const fn as_raw(self) -> u64 {
        self.0
    }
}

impl core::fmt::Display for GenerationId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "generation#{}", self.0)
    }
}

/// The catalog's next unallocated numbers. Both only move forward.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct IdentityAllocator {
    /// The next [`IndexId`] to allocate.
    pub next_index: u64,
    /// The next [`GenerationId`] to allocate.
    pub next_generation: u64,
}

/// The catalog has no number left to allocate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the index catalog has allocated every {0} identity")]
pub struct IdentityExhausted(pub &'static str);

impl IdentityAllocator {
    /// Take the next index identity.
    ///
    /// # Errors
    ///
    /// [`IdentityExhausted`] once every number has been taken: a number is
    /// never handed out twice, so there is no wrapping around.
    pub fn allocate_index(&mut self) -> Result<IndexId, IdentityExhausted> {
        let id = self.next_index;
        self.next_index = id.checked_add(1).ok_or(IdentityExhausted("index"))?;
        Ok(IndexId(id))
    }

    /// Take the next generation identity.
    ///
    /// # Errors
    ///
    /// As [`Self::allocate_index`].
    pub fn allocate_generation(&mut self) -> Result<GenerationId, IdentityExhausted> {
        let id = self.next_generation;
        self.next_generation = id.checked_add(1).ok_or(IdentityExhausted("generation"))?;
        Ok(GenerationId(id))
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
