//! The DERIVED index work a transaction stages, and how it is sealed into
//! the unit the transaction commits.
//!
//! A DERIVED index's entries are staged in the write buffer like any other
//! write, so the transaction reads its own index changes and its unique
//! values are claimed by write-write conflict. They are left out of the
//! logged mutations; the unit carries, per index and node, the membership the
//! node had when the transaction began and the one it ends with, and every
//! member derives the entries from those.

use coordinode_core::graph::node::{NodeRecord, decode_node_key, decode_temporal_node_key};
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::EntryOwner;
use coordinode_core::index::identity::GenerationId;
use coordinode_core::txn::proposal::{
    DerivedIndexWork, DerivedSource, IndexBinding, Mutation, PartitionId,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// One node's (or temporal node version's) membership change in one DERIVED
/// index.
#[derive(Debug, Clone)]
struct Change {
    binding: IndexBinding,
    owner: EntryOwner,
    /// The membership before the transaction.
    old: Option<Vec<Value>>,
    /// The membership after the latest statement.
    new: Option<Vec<Value>>,
}

/// DERIVED index work staged by one transaction.
#[derive(Debug, Clone, Default)]
pub(crate) struct DerivedLedger {
    /// In the order each (index, owner) was first changed.
    changes: Vec<Change>,
    by_target: FxHashMap<(GenerationId, EntryOwner), usize>,
    /// The index-partition keys staged for DERIVED indexes.
    keys: FxHashSet<Vec<u8>>,
    /// The index definitions this transaction's effects are bound to, in
    /// either profile: each is conditioned once on its version.
    bound: FxHashSet<Vec<u8>>,
    /// Entry effects staged over all statements: an upper bound of those the
    /// sealed changes derive, since a change's effects are the difference of
    /// its first and last memberships and each statement staged the
    /// difference of its own step. Each addend counts effects held in memory.
    staged_effects: usize,
}

impl DerivedLedger {
    /// Record that `owner`'s membership in the index `binding` names moves
    /// from `old` to `new`, staging `keys` in the write buffer. A later change
    /// of the same owner in the same index keeps the first `old`.
    pub(crate) fn stage(
        &mut self,
        binding: &IndexBinding,
        owner: EntryOwner,
        old: Option<Vec<Value>>,
        new: Option<Vec<Value>>,
        keys: impl IntoIterator<Item = Vec<u8>>,
    ) {
        for key in keys {
            self.staged_effects += 1;
            self.keys.insert(key);
        }
        let target = (binding.interpretation.generation, owner);
        match self.by_target.get(&target) {
            Some(&at) => self.changes[at].new = new,
            None => {
                self.by_target.insert(target, self.changes.len());
                self.changes.push(Change {
                    binding: binding.clone(),
                    owner,
                    old,
                    new,
                });
            }
        }
    }

    /// Record that this transaction's effects are bound to the definition
    /// stored at `key`; `true` the first time.
    pub(crate) fn bind(&mut self, key: &[u8]) -> bool {
        self.bound.insert(key.to_vec())
    }

    /// Whether `key` is an index entry of a DERIVED index, logged as work
    /// rather than as a mutation.
    pub(crate) fn owns(&self, key: &[u8]) -> bool {
        self.keys.contains(key)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }

    /// Refuse, with the bound, work whose sealed changes derive more than
    /// `max` entry effects: every member would refuse the unit when it
    /// derives it. The staged count settles the common case; only past the
    /// bound are the sealed changes derived to count them exactly.
    pub(crate) fn check_fan_out(&self, max: usize) -> Result<(), usize> {
        if self.staged_effects <= max {
            return Ok(());
        }
        let mut sealed = 0usize;
        for change in self.changes.iter().filter(|c| c.old != c.new) {
            sealed += change
                .binding
                .interpretation
                .membership_effects(change.owner, change.old.as_deref(), change.new.as_deref())
                .len();
            if sealed > max {
                return Err(max);
            }
        }
        Ok(())
    }

    /// Append the staged work to `mutations`, the unit's final data
    /// mutations. A node whose final record is put whole in the unit, with no
    /// merge operand after it, is derived from that record; otherwise the new
    /// membership travels as values. A change that ends where it began
    /// derives nothing and is left out.
    pub(crate) fn seal(self, mutations: &mut Vec<Mutation>) {
        // The last whole-record put of each node or temporal node version,
        // unless an operand follows it.
        let owner_of = |key: &[u8]| match decode_node_key(key) {
            Some((_, node)) => Some(EntryOwner::node(node.as_raw())),
            None => decode_temporal_node_key(key)
                .map(|(_, node, valid_from)| EntryOwner::version(node.as_raw(), valid_from)),
        };
        let mut records: FxHashMap<EntryOwner, Option<u32>> = FxHashMap::default();
        for (at, mutation) in mutations.iter().enumerate() {
            match mutation {
                Mutation::Put {
                    partition: PartitionId::Node,
                    key,
                    ..
                } => {
                    if let (Some(owner), Ok(at)) = (owner_of(key), u32::try_from(at)) {
                        records.insert(owner, Some(at));
                    }
                }
                Mutation::Merge {
                    partition: PartitionId::Node,
                    key,
                    ..
                } => {
                    if let Some(owner) = owner_of(key) {
                        records.insert(owner, None);
                    }
                }
                _ => {}
            }
        }
        for change in self.changes {
            if change.old == change.new {
                continue;
            }
            let from_record = records.get(&change.owner).copied().flatten().filter(|&at| {
                // The record must yield exactly the membership the
                // transaction saw; anything else travels as values.
                record_membership(mutations, at, &change.binding) == Some(change.new.clone())
            });
            let new = match from_record {
                Some(at) => DerivedSource::UnitRecord(at),
                None => DerivedSource::Values(change.new),
            };
            mutations.push(Mutation::Derive(DerivedIndexWork {
                binding: change.binding,
                node_id: change.owner.node_id,
                valid_from: change.owner.valid_from,
                old: change.old,
                new,
            }));
        }
    }
}

/// The membership the record put at `at` gives in `binding`'s index, or
/// `None` when that position holds no decodable record.
fn record_membership(
    mutations: &[Mutation],
    at: u32,
    binding: &IndexBinding,
) -> Option<Option<Vec<Value>>> {
    let Some(Mutation::Put { value, .. }) = mutations.get(at as usize) else {
        return None;
    };
    let record = NodeRecord::from_msgpack(value).ok()?;
    Some(binding.interpretation.record_membership(&record))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
