//! Deterministic maintenance of key-shaped index entries: which values a node
//! holds in an index, and the entry effects a change of those values makes.
//!
//! Both maintenance profiles run these functions. A RESOLVED index runs them
//! on the member executing the write and logs the resulting entries; a
//! DERIVED index logs the sealed [`IndexInterpretation`] and exact inputs, and
//! every member runs them again at application. They read nothing but their
//! arguments: no clock, no catalog, no current row.

use serde::{Deserialize, Serialize};

use super::encoding::{
    encode_index_key, encode_tuple, encode_unique_index_key, encode_version_index_key,
};
use crate::graph::node::NodeRecord;
use crate::graph::types::Value;

/// The entry key layout these functions produce.
pub const KEY_CODEC: u32 = 1;

/// A property as a record stores it: under its interned field id when the
/// name was bound at sealing, otherwise, or in addition, by name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PropertyRef {
    /// The field id bound to the name when the effect was sealed.
    pub field: Option<u32>,
    /// The property name.
    pub name: String,
}

/// A partial index's membership test on one property, with the typed
/// equality of the filter it was declared with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MembershipFilter {
    /// The property is this string.
    EqualsString(PropertyRef, String),
    /// The property is this integer.
    EqualsInt(PropertyRef, i64),
    /// The property is this boolean.
    EqualsBool(PropertyRef, bool),
    /// The property is present and not null.
    Exists(PropertyRef),
}

impl MembershipFilter {
    /// The property the filter tests.
    pub fn property(&self) -> &PropertyRef {
        match self {
            Self::EqualsString(p, _)
            | Self::EqualsInt(p, _)
            | Self::EqualsBool(p, _)
            | Self::Exists(p) => p,
        }
    }

    fn admits(&self, value: &Value) -> bool {
        match self {
            Self::EqualsString(_, want) => value.as_str() == Some(want.as_str()),
            Self::EqualsInt(_, want) => value.as_int() == Some(*want),
            Self::EqualsBool(_, want) => value.as_bool() == Some(*want),
            Self::Exists(_) => !value.is_null(),
        }
    }
}

/// Everything that decides a key-shaped B-tree index's entries: the sealed
/// interpretation an effect carries, so that deriving it never consults the
/// current catalog.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexInterpretation {
    /// Entry key layout, [`KEY_CODEC`] for every interpretation this build
    /// writes.
    pub codec: u32,
    /// The index name its entry keys carry.
    pub name: String,
    /// One entry per value, keyed by the value alone.
    pub unique: bool,
    /// A node missing any indexed property has no entry.
    pub sparse: bool,
    /// The indexed properties, in key order.
    pub properties: Vec<PropertyRef>,
    /// A partial index's membership test.
    pub filter: Option<MembershipFilter>,
}

/// An interpretation this build cannot derive entries for.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("index `{name}` uses key codec {codec}, which this build does not derive")]
pub struct UnsupportedInterpretation {
    /// The index.
    pub name: String,
    /// Its codec.
    pub codec: u32,
}

/// One entry change: a put of `value`, or a delete when it is `None`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryEffect {
    /// The entry key.
    pub key: Vec<u8>,
    /// The entry value to put, or `None` to delete the entry.
    pub value: Option<Vec<u8>>,
}

impl IndexInterpretation {
    /// Refuse an interpretation this build does not derive.
    ///
    /// # Errors
    ///
    /// [`UnsupportedInterpretation`] for another codec.
    pub fn check_supported(&self) -> Result<(), UnsupportedInterpretation> {
        if self.codec == KEY_CODEC {
            Ok(())
        } else {
            Err(UnsupportedInterpretation {
                name: self.name.clone(),
                codec: self.codec,
            })
        }
    }

    /// The values the index holds for a node whose properties `value_of`
    /// answers, or `None` when the node has no entry: a sparse index skips a
    /// node missing an indexed property, a partial index one its filter
    /// rejects. A missing property is null.
    pub fn membership(
        &self,
        value_of: &dyn Fn(&PropertyRef) -> Option<Value>,
    ) -> Option<Vec<Value>> {
        let values: Vec<Value> = self
            .properties
            .iter()
            .map(|p| value_of(p).unwrap_or(Value::Null))
            .collect();
        if self.sparse && values.iter().any(Value::is_null) {
            return None;
        }
        if let Some(filter) = &self.filter {
            let tested = value_of(filter.property()).unwrap_or(Value::Null);
            if !filter.admits(&tested) {
                return None;
            }
        }
        Some(values)
    }

    /// [`Self::membership`] of a stored node record.
    pub fn record_membership(&self, record: &NodeRecord) -> Option<Vec<Value>> {
        self.membership(&|p: &PropertyRef| {
            p.field
                .and_then(|field| record.get(field))
                .or_else(|| record.get_extra(&p.name))
                .cloned()
        })
    }

    /// The entry effects of a membership moving from `old` to `new`; see
    /// [`membership_effects`].
    pub fn membership_effects(
        &self,
        owner: EntryOwner,
        old: Option<&[Value]>,
        new: Option<&[Value]>,
    ) -> Vec<EntryEffect> {
        membership_effects(&self.name, self.unique, owner, old, new)
    }
}

/// Whose membership an entry records: a node, or one version of a temporal
/// node, named by the `valid_from` it starts at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntryOwner {
    /// The node.
    pub node_id: u64,
    /// The version's `valid_from`; `None` for a node that is not temporal.
    pub valid_from: Option<i64>,
}

impl EntryOwner {
    /// A node that is not temporal: one membership per node.
    pub const fn node(node_id: u64) -> Self {
        Self {
            node_id,
            valid_from: None,
        }
    }

    /// The version of temporal node `node_id` that starts at `valid_from`.
    pub const fn version(node_id: u64, valid_from: i64) -> Self {
        Self {
            node_id,
            valid_from: Some(valid_from),
        }
    }
}

/// The entry of `owner` under `tuple` in the index `name`: keyed by the
/// value alone and holding the node when `unique`, keyed by value and owner
/// with an empty value otherwise.
pub fn entry(name: &str, unique: bool, tuple: &[u8], owner: EntryOwner) -> (Vec<u8>, Vec<u8>) {
    if unique {
        (
            encode_unique_index_key(name, tuple),
            owner.node_id.to_be_bytes().to_vec(),
        )
    } else {
        let key = match owner.valid_from {
            Some(valid_from) => encode_version_index_key(name, tuple, owner.node_id, valid_from),
            None => encode_index_key(name, tuple, owner.node_id),
        };
        (key, Vec::new())
    }
}

/// The entry effects in the index `name` of `owner`'s membership moving from
/// `old` to `new`: a delete for each tuple it leaves, a put for each it
/// enters. A tuple in both is untouched.
///
/// A unique entry is keyed by the value alone and claims it for the node,
/// whichever of its versions holds it. A version of a temporal node leaving a
/// value therefore keeps the claim: another version of the node may hold the
/// value too, and the node holds it in its history either way.
pub fn membership_effects(
    name: &str,
    unique: bool,
    owner: EntryOwner,
    old: Option<&[Value]>,
    new: Option<&[Value]>,
) -> Vec<EntryEffect> {
    let before = old.map(tuples).unwrap_or_default();
    let after = new.map(tuples).unwrap_or_default();
    let mut effects = Vec::with_capacity(before.len() + after.len());
    let releases = !(unique && owner.valid_from.is_some());
    for tuple in &before {
        if releases && after.binary_search(tuple).is_err() {
            effects.push(EntryEffect {
                key: entry(name, unique, tuple, owner).0,
                value: None,
            });
        }
    }
    for tuple in &after {
        if before.binary_search(tuple).is_err() {
            let (key, value) = entry(name, unique, tuple, owner);
            effects.push(EntryEffect {
                key,
                value: Some(value),
            });
        }
    }
    effects
}

/// The key tuples `values` index under, sorted and distinct: one per
/// combination of list elements (multikey), none for a combination that has
/// no key.
pub fn tuples(values: &[Value]) -> Vec<Vec<u8>> {
    let mut combos: Vec<Vec<Value>> = vec![Vec::with_capacity(values.len())];
    for value in values {
        match value {
            Value::Array(elements) => {
                let mut next = Vec::with_capacity(combos.len() * elements.len());
                for combo in &combos {
                    for element in elements {
                        let mut extended = combo.clone();
                        extended.push(element.clone());
                        next.push(extended);
                    }
                }
                combos = next;
            }
            other => {
                for combo in &mut combos {
                    combo.push(other.clone());
                }
            }
        }
    }
    let mut out: Vec<Vec<u8>> = combos
        .iter()
        .filter_map(|combo| encode_tuple(combo).ok())
        .collect();
    // One list may repeat an element; its entry is one entry.
    out.sort_unstable();
    out.dedup();
    out
}

/// One entry effect of a unit, with what [`combine_unit_effects`] needs to
/// order it against the unit's other effects on the same key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitEffect {
    /// Whether the entry belongs to a unique index.
    pub unique: bool,
    /// The node whose membership produced the effect.
    pub node_id: u64,
    /// The effect.
    pub effect: EntryEffect,
}

/// Combine the entry effects of one application unit into one effect per
/// key, in first-touched order.
///
/// Within a unit a later effect on a key replaces an earlier one, with one
/// exception that keeps a unique claim: a unique entry put for one node is
/// not removed by another node leaving the same value in that unit (a value
/// handed from one node to another), whatever order the two changes were
/// staged in. A node releasing its own claim does remove it.
pub fn combine_unit_effects(effects: impl IntoIterator<Item = UnitEffect>) -> Vec<EntryEffect> {
    let mut order: Vec<Vec<u8>> = Vec::new();
    let mut last: rustc_hash::FxHashMap<Vec<u8>, UnitEffect> = rustc_hash::FxHashMap::default();
    for item in effects {
        match last.get(&item.effect.key) {
            None => {
                order.push(item.effect.key.clone());
                last.insert(item.effect.key.clone(), item);
            }
            Some(held) => {
                let keeps_other_claim = item.unique
                    && item.effect.value.is_none()
                    && held.effect.value.is_some()
                    && held.node_id != item.node_id;
                if !keeps_other_claim {
                    last.insert(item.effect.key.clone(), item);
                }
            }
        }
    }
    order
        .into_iter()
        .filter_map(|key| last.remove(&key).map(|item| item.effect))
        .collect()
}

/// Why a unit's DERIVED work could not be resolved into entry effects.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DeriveError {
    /// The work names an interpretation this build cannot derive.
    #[error(transparent)]
    Unsupported(#[from] UnsupportedInterpretation),
    /// The work names a position of the unit that holds no node record put.
    #[error("derivation source {0} is not a node record put of the unit")]
    BadSource(u32),
    /// The node record the work names does not decode.
    #[error("the node record at position {ordinal} does not decode: {message}")]
    Record {
        /// The position.
        ordinal: u32,
        /// The decode failure.
        message: String,
    },
    /// The unit derives more entry effects than it is allowed.
    #[error("the unit derives more than {0} index entry effects")]
    FanOut(usize),
}

/// Resolve a unit into the mutations it applies: every mutation that is not
/// DERIVED work, in order, followed by the entry effects the DERIVED work
/// derives, combined with [`combine_unit_effects`]. A unit without DERIVED
/// work is returned as it is.
///
/// # Errors
///
/// [`DeriveError`] when the work cannot be derived as sealed, or derives more
/// than `max_effects` entry effects.
pub fn resolve_unit(
    mutations: &[crate::txn::proposal::Mutation],
    max_effects: usize,
) -> Result<std::borrow::Cow<'_, [crate::txn::proposal::Mutation]>, DeriveError> {
    use crate::txn::proposal::{DerivedSource, Mutation, PartitionId};

    if !mutations.iter().any(|m| matches!(m, Mutation::Derive(_))) {
        return Ok(std::borrow::Cow::Borrowed(mutations));
    }
    let mut resolved = Vec::with_capacity(mutations.len());
    let mut effects = Vec::new();
    for mutation in mutations {
        let Mutation::Derive(work) = mutation else {
            resolved.push(mutation.clone());
            continue;
        };
        let interpretation = &work.binding.interpretation;
        interpretation.check_supported()?;
        let new = match &work.new {
            DerivedSource::Values(values) => values.clone(),
            DerivedSource::UnitRecord(ordinal) => {
                let Some(Mutation::Put {
                    partition: PartitionId::Node,
                    value,
                    ..
                }) = mutations.get(*ordinal as usize)
                else {
                    return Err(DeriveError::BadSource(*ordinal));
                };
                let record = NodeRecord::from_msgpack(value).map_err(|e| DeriveError::Record {
                    ordinal: *ordinal,
                    message: e.to_string(),
                })?;
                interpretation.record_membership(&record)
            }
        };
        let owner = EntryOwner {
            node_id: work.node_id,
            valid_from: work.valid_from,
        };
        let derived = interpretation.membership_effects(owner, work.old.as_deref(), new.as_deref());
        if effects.len() + derived.len() > max_effects {
            return Err(DeriveError::FanOut(max_effects));
        }
        effects.extend(derived.into_iter().map(|effect| UnitEffect {
            unique: interpretation.unique,
            node_id: work.node_id,
            effect,
        }));
    }
    resolved.extend(
        combine_unit_effects(effects)
            .into_iter()
            .map(|effect| match effect.value {
                Some(value) => Mutation::Put {
                    partition: PartitionId::Idx,
                    key: effect.key,
                    value,
                },
                None => Mutation::Delete {
                    partition: PartitionId::Idx,
                    key: effect.key,
                },
            }),
    );
    Ok(std::borrow::Cow::Owned(resolved))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
