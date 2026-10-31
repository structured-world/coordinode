//! Deterministic maintenance of key-shaped index entries: which values a node
//! holds in an index, and the entry effects a change of those values makes.
//!
//! Both maintenance profiles run these functions. A RESOLVED index runs them
//! on the member executing the write and logs the resulting entries; a
//! DERIVED index logs the sealed [`IndexInterpretation`] and exact inputs, and
//! every member runs them again at application. They read nothing but their
//! arguments: no clock, no catalog, no current row.

use serde::{Deserialize, Serialize};

use super::encoding::append_entry_key;
use super::identity::GenerationId;
use crate::graph::node::NodeRecord;
use crate::graph::types::Value;

/// The entry key layout these functions produce: entries keyed by their
/// index generation.
pub const KEY_CODEC: u32 = 2;

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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexInterpretation {
    /// Entry key layout, [`KEY_CODEC`] for every interpretation this build
    /// writes.
    pub codec: u32,
    /// The index generation its entry keys carry.
    pub generation: GenerationId,
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
#[error("{generation} uses key codec {codec}, which this build does not derive")]
pub struct UnsupportedInterpretation {
    /// The index generation.
    pub generation: GenerationId,
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
                generation: self.generation,
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
    ///
    /// # Errors
    ///
    /// As [`membership_effects`].
    pub fn membership_effects(
        &self,
        owner: EntryOwner,
        old: Option<&[Value]>,
        new: Option<&[Value]>,
    ) -> Result<Vec<EntryEffect>, KeyTooLong> {
        membership_effects(self.generation, owner, old, new)
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

/// The entry of `owner` under `tuple` in the index generation `generation`:
/// keyed by value and owner, with an empty value. A unique index keys its
/// entries the same way: who may hold a value is decided at admission, so
/// several owners of one value (two versions of a node, or the holders two
/// interpretations of a label keep while one replaces the other) each keep
/// their own entry.
pub fn entry(generation: GenerationId, tuple: &[u8], owner: EntryOwner) -> (Vec<u8>, Vec<u8>) {
    let mut key = Vec::new();
    append_entry_key_of(&mut key, generation, tuple, owner);
    (key, Vec::new())
}

/// The key of [`entry`] appended to `out`: what a caller building many entry
/// keys in one buffer writes, with no allocation per key.
pub fn append_entry_key_of(
    out: &mut Vec<u8>,
    generation: GenerationId,
    tuple: &[u8],
    owner: EntryOwner,
) {
    append_entry_key(out, generation, tuple, owner.node_id, owner.valid_from);
}

/// The longest key the index stores: the store keeps a key's length in two
/// bytes.
pub const MAX_INDEX_KEY_LEN: usize = u16::MAX as usize;

/// An index entry whose key is longer than [`MAX_INDEX_KEY_LEN`]: the value
/// it would hold cannot be indexed, and nothing of the write holding it is
/// staged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "an index entry of {len} bytes is longer than the {MAX_INDEX_KEY_LEN} bytes a key may take"
)]
pub struct KeyTooLong {
    /// The length of the key, in bytes.
    pub len: usize,
}

/// The entry effects in the index generation `generation` of `owner`'s
/// membership moving from `old` to `new`: a delete for each tuple it leaves,
/// a put for each it enters. A tuple in both is untouched.
///
/// # Errors
///
/// [`KeyTooLong`] for an entry it enters whose key the index cannot hold.
pub fn membership_effects(
    generation: GenerationId,
    owner: EntryOwner,
    old: Option<&[Value]>,
    new: Option<&[Value]>,
) -> Result<Vec<EntryEffect>, KeyTooLong> {
    let unbounded = |_| Ok::<(), KeyTooLong>(());
    let before = old.map(|v| tuples_within(v, unbounded)).transpose()?;
    let after = new.map(|v| tuples_within(v, unbounded)).transpose()?;
    effects_of(
        generation,
        owner,
        before.as_deref().unwrap_or_default(),
        after.as_deref().unwrap_or_default(),
        unbounded,
    )
}

/// [`membership_effects`] of the tuples `before` and `after` (each sorted
/// and distinct, as [`tuples_within`] gives them), passing `charge` the bytes
/// of each effect before it is built: the list, then each key.
///
/// A key the index cannot hold is never built. Entering it is refused;
/// leaving it removes nothing, since no such entry can be stored.
///
/// # Errors
///
/// [`KeyTooLong`] for an entry it enters whose key the index cannot hold,
/// or what `charge` returns.
pub fn effects_of<E: From<KeyTooLong>>(
    generation: GenerationId,
    owner: EntryOwner,
    before: &[Vec<u8>],
    after: &[Vec<u8>],
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<Vec<EntryEffect>, E> {
    let leaving = before.iter().filter(|t| after.binary_search(t).is_err());
    let entering = after.iter().filter(|t| before.binary_search(t).is_err());
    let count = leaving.clone().count() + entering.clone().count();
    charge(count * core::mem::size_of::<EntryEffect>())?;
    let mut effects = Vec::with_capacity(count);
    let mut push = |tuple: &Vec<u8>, value: Option<Vec<u8>>| -> Result<(), E> {
        let len = super::encoding::entry_key_len(tuple.len(), owner.valid_from.is_some());
        if len > MAX_INDEX_KEY_LEN {
            return match value {
                Some(_) => Err(KeyTooLong { len }.into()),
                None => Ok(()),
            };
        }
        charge(len)?;
        let mut key = Vec::with_capacity(len);
        append_entry_key_of(&mut key, generation, tuple, owner);
        effects.push(EntryEffect { key, value });
        Ok(())
    };
    for tuple in leaving {
        push(tuple, None)?;
    }
    for tuple in entering {
        push(tuple, Some(Vec::new()))?;
    }
    Ok(effects)
}

/// The key tuples `values` index under, sorted and distinct: one per
/// combination of list elements (multikey), none for a combination that has
/// no key.
pub fn tuples(values: &[Value]) -> Vec<Vec<u8>> {
    match tuples_within(values, |_| Ok::<(), core::convert::Infallible>(())) {
        Ok(tuples) => tuples,
        Err(never) => match never {},
    }
}

/// [`tuples`], passing `charge` the bytes it is about to allocate before it
/// allocates them: the encoding of each value and list element (linear in
/// the values), the list of combinations, and each tuple at its exact
/// length. The combinations of several lists multiply, so a refusal comes
/// before any of them is built; nothing is cloned.
///
/// # Errors
///
/// What `charge` returns.
pub fn tuples_within<E>(
    values: &[Value],
    mut charge: impl FnMut(usize) -> Result<(), E>,
) -> Result<Vec<Vec<u8>>, E> {
    // Each value, or each element of a list, encoded once: a segment, or
    // `None` for one that has no key. A tuple is one segment per value.
    let elements: usize = values
        .iter()
        .map(|v| match v {
            Value::Array(items) => items.len(),
            _ => 1,
        })
        .sum();
    charge(
        elements * core::mem::size_of::<Option<Vec<u8>>>()
            + values.len() * core::mem::size_of::<Vec<Option<Vec<u8>>>>(),
    )?;
    let mut segments: Vec<Vec<Option<Vec<u8>>>> = Vec::with_capacity(values.len());
    let mut scratch = Vec::new();
    for value in values {
        let items: &[Value] = match value {
            Value::Array(items) => items,
            other => core::slice::from_ref(other),
        };
        let mut choices = Vec::with_capacity(items.len());
        for item in items {
            // Room for the element's encoding, charged before the scratch
            // grows to it, so the encoding never grows it.
            scratch.clear();
            let room = super::encoding::element_len_bound(item);
            if let Some(more) = room.checked_sub(scratch.capacity()).filter(|m| *m > 0) {
                charge(more)?;
                scratch.reserve_exact(room);
            }
            choices.push(match super::encoding::encode_element(item, &mut scratch) {
                Ok(()) => {
                    // Its kept copy, at its exact length.
                    charge(scratch.len())?;
                    Some(scratch.clone())
                }
                Err(_) => None,
            });
        }
        segments.push(choices);
    }
    // How many combinations the lists make, each made below.
    let Some(count) = segments
        .iter()
        .try_fold(1usize, |n, choices| n.checked_mul(choices.len()))
    else {
        charge(usize::MAX)?;
        return Ok(Vec::new());
    };
    charge(
        count * core::mem::size_of::<Vec<u8>>() + segments.len() * core::mem::size_of::<usize>(),
    )?;
    let mut out: Vec<Vec<u8>> = Vec::with_capacity(count);
    // An odometer over the choices of each value.
    let mut at = vec![0usize; segments.len()];
    for _ in 0..count {
        let chosen = segments.iter().zip(&at).map(|(choices, &i)| &choices[i]);
        if chosen.clone().all(Option::is_some) {
            let len: usize = chosen.clone().flatten().map(Vec::len).sum();
            charge(len)?;
            let mut tuple = Vec::with_capacity(len);
            for segment in chosen.flatten() {
                tuple.extend_from_slice(segment);
            }
            out.push(tuple);
        }
        for (digit, choices) in at.iter_mut().zip(&segments).rev() {
            *digit += 1;
            if *digit < choices.len() {
                break;
            }
            *digit = 0;
        }
    }
    // One list may repeat an element; its entry is one entry.
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Combine the entry effects of one application unit into one effect per
/// key, in first-touched order: a later effect on a key replaces an earlier
/// one. An entry is keyed by its owner, so two owners' effects never meet on
/// one key, a value handed from one node to another included.
pub fn combine_unit_effects(effects: impl IntoIterator<Item = EntryEffect>) -> Vec<EntryEffect> {
    let mut order: Vec<Vec<u8>> = Vec::new();
    let mut last: rustc_hash::FxHashMap<Vec<u8>, EntryEffect> = rustc_hash::FxHashMap::default();
    for effect in effects {
        if last.insert(effect.key.clone(), effect.clone()).is_none() {
            order.push(effect.key);
        }
    }
    order
        .into_iter()
        .filter_map(|key| last.remove(&key))
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
    /// The work enters an entry whose key the index cannot hold. The member
    /// that sealed it refused such a write before logging it, so a logged
    /// unit meets this only if it was sealed elsewhere without that check.
    #[error(transparent)]
    KeyTooLong(#[from] KeyTooLong),
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
        let derived =
            interpretation.membership_effects(owner, work.old.as_deref(), new.as_deref())?;
        if effects.len() + derived.len() > max_effects {
            return Err(DeriveError::FanOut(max_effects));
        }
        effects.extend(derived);
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
