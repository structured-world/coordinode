//! Index store: secondary B-tree entries in [`Partition::Idx`] and the
//! index-definition catalog in [`Partition::Schema`].
//!
//! Entries are staged in the writing statement's [`Transaction`], so they
//! commit in the same log entry and at the same version as the data they
//! index: a crash keeps both or neither, a replica applies them like any other
//! key, a rolled-back statement leaves none behind, and a scan inside the
//! transaction sees its own writes.
//!
//! Two entry shapes, keyed by the values' tuple encoding
//! ([`coordinode_core::index::encoding`]):
//!
//! - a non-unique index writes `idx:<name>:<tuple>:<node_id>` with an empty
//!   value, one entry per node, found by a prefix scan;
//! - a unique index writes `uidx:<name>:<tuple>` whose value is the holder's
//!   node id. Keyed by the value alone, the entry is its own uniqueness claim:
//!   two transactions inserting one value write one key, and write-write
//!   conflict detection lets only one of them commit. Checking the value is a
//!   point read the partition's bloom filters answer without touching tables
//!   that cannot hold it.
//!
//! A list value indexes each of its elements (multikey). A value with no key
//! (NaN, a map, a vector) is not indexed; a lookup of it reports so, and the
//! caller answers with a scan.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::{membership_effects, tuples};
use coordinode_core::index::encoding::{
    decode_node_id, encode_tuple, encode_unique_index_key, index_prefix, index_value_prefix,
    legacy_index_prefix, unique_index_prefix,
};
use coordinode_core::txn::proposal::{DerivedIndexWork, DerivedSource, Mutation, PartitionId};
use coordinode_storage::Guard;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;

use crate::error::{StoreError, StoreResult};
use crate::index_def::{IndexDefinition, IndexProfile, IndexState, NamespaceIndexPolicy};

/// Layer 4 store for secondary B-tree entries and the index catalog.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not store index entries",
    label = "an index store is required here",
    note = "use `LocalIndexStore`, the CE implementation over a statement transaction"
)]
pub trait IndexStore {
    /// Stage the entry changes of `node_id`'s membership in `index` moving
    /// from `old` to `new` (`None`: no entry), in the index's profile. A
    /// RESOLVED index stages the entries as writes the unit logs; a DERIVED
    /// one stages them for this transaction's reads and conflicts, and the
    /// unit logs the change sealed under `index`'s binding, with property
    /// field ids from `field_of`. A unique entry is removed only while
    /// `node_id` holds it; for a unique value the caller has checked
    /// [`Self::unique_conflict`] first. Returns how many entries were put.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable unique entry.
    fn stage_membership(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        field_of: &dyn Fn(&str) -> Option<u32>,
        node_id: NodeId,
        old: Option<&[Value]>,
        new: Option<&[Value]>,
    ) -> StoreResult<usize>;

    /// The node other than `node_id` that holds one of the entries `values`
    /// would take in the unique `index`, as the transaction sees it.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable entry.
    fn unique_conflict(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        values: &[Value],
        node_id: NodeId,
    ) -> StoreResult<Option<NodeId>>;

    /// [`Self::unique_conflict`] against the latest committed state, outside
    /// any transaction. Tells a transaction that lost a race for a value
    /// which node won it.
    ///
    /// # Errors
    ///
    /// As [`Self::unique_conflict`].
    fn committed_conflict(
        &self,
        index: &IndexDefinition,
        values: &[Value],
        node_id: NodeId,
    ) -> StoreResult<Option<NodeId>>;

    /// The nodes whose entry holds exactly `values`, as the transaction sees
    /// it. `None` when the values have no key, so the index cannot answer.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable entry.
    fn scan_exact(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        values: &[Value],
    ) -> StoreResult<Option<Vec<NodeId>>>;

    /// Every node with an entry in `index`, in key order.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable entry.
    fn scan_entry_ids(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
    ) -> StoreResult<Vec<NodeId>>;

    /// The mutations removing the entries of `node_id` holding `values`, for
    /// a writer that deletes nodes by submitting mutations directly (the TTL
    /// reaper). The node is being deleted, so its entries go unconditionally:
    /// as deletes for a RESOLVED index, as sealed work for a DERIVED one.
    fn entry_delete_mutations(
        &self,
        index: &IndexDefinition,
        field_of: &dyn Fn(&str) -> Option<u32>,
        values: &[Value],
        node_id: NodeId,
    ) -> Vec<Mutation>;

    /// The mutations removing every entry of the index `name`, of both
    /// shapes: one range tombstone each.
    fn clear_mutations(&self, name: &str) -> Vec<Mutation>;

    /// The mutation storing `def` in the catalog, for DDL that commits it in
    /// one log entry with other effects.
    ///
    /// # Errors
    ///
    /// An encoding failure.
    fn definition_put_mutation(&self, def: &IndexDefinition) -> StoreResult<Mutation>;

    /// The mutation removing the definition `name` from the catalog.
    fn definition_delete_mutation(&self, name: &str) -> Mutation;

    /// Apply one unit of mutations straight to the engine as one batch, for a
    /// context that has no log to replicate them through.
    ///
    /// # Errors
    ///
    /// A storage failure, or DERIVED work that cannot be derived.
    fn apply_unreplicated(&self, mutations: &[Mutation]) -> StoreResult<()>;

    /// Remove every entry the index `name` wrote in the layout that preceded
    /// transactional entries. Local: those entries were written outside the
    /// log on the member that ran the statement, so each member removes its
    /// own before rebuilding the index.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn clear_legacy(&self, name: &str) -> StoreResult<()>;

    /// Persist an index definition into the schema catalog (keyed by
    /// `schema:idx:<name>`) directly, outside the log. For state this member
    /// keeps about itself (a vector index's build state).
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn put_definition(&self, def: &IndexDefinition) -> StoreResult<()>;

    /// Load a persisted index definition by name.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable definition.
    fn load_definition(&self, name: &str) -> StoreResult<Option<IndexDefinition>>;

    /// Every persisted index definition in `schema:idx:` key order. A
    /// definition whose bytes do not decode is skipped with a warning, so one
    /// corrupt record does not take down the registry on open.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>>;

    /// Update only the build `state` of a persisted definition, directly.
    /// `Ok(false)` when no definition is stored under `name`.
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn set_definition_state(&self, name: &str, state: IndexState) -> StoreResult<bool>;

    /// Persist an index definition through a statement [`Transaction`]: it
    /// commits, and replicates, with the statement.
    ///
    /// # Errors
    ///
    /// An encoding failure.
    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()>;

    /// Delete a persisted index definition through a statement
    /// [`Transaction`]. Tombstone semantics: no error when absent.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()>;

    /// The namespace index policy and the version of its record (`None`
    /// before the first change: the RESOLVED default at revision 0).
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn index_policy(&self) -> StoreResult<(NamespaceIndexPolicy, Option<u64>)>;

    /// Stage `policy` through a statement [`Transaction`], only while its
    /// record is still at `version`: two concurrent changes cannot both
    /// build on one revision.
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn put_index_policy_txn(
        &self,
        txn: &mut Transaction,
        policy: &NamespaceIndexPolicy,
        version: Option<u64>,
    ) -> StoreResult<()>;

    /// The version of the stored definition `name`, the value a writer binds
    /// its effects to so a transition between staging and commit refuses it.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn definition_version(&self, name: &str) -> StoreResult<Option<u64>>;
}

/// CE implementation of [`IndexStore`].
///
/// # Examples
///
/// ```no_run
/// use coordinode_modality::{IndexDefinition, IndexStore, LocalIndexStore};
/// use coordinode_core::graph::{node::NodeId, types::Value};
/// # use coordinode_storage::engine::{config::*, core::StorageEngine, transaction::Transaction};
/// # use coordinode_core::txn::timestamp::Timestamp;
/// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
/// #     "ep", std::path::Path::new("/tmp/x"),
/// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
/// # let engine = StorageEngine::open(&cfg)?;
/// # let mut txn = Transaction::begin(&engine, None, Timestamp::from_raw(1));
/// let index = IndexDefinition::btree("user_email", "User", "email").unique();
/// let email = [Value::String("a@x".into())];
/// let store = LocalIndexStore::new(&engine);
/// assert_eq!(store.unique_conflict(&mut txn, &index, &email, NodeId::from_raw(1))?, None);
/// let no_fields = |_: &str| None;
/// store.stage_membership(&mut txn, &index, &no_fields, NodeId::from_raw(1), None, Some(&email))?;
/// assert_eq!(
///     store.unique_conflict(&mut txn, &index, &email, NodeId::from_raw(2))?,
///     Some(NodeId::from_raw(1))
/// );
/// # Ok::<_, Box<dyn std::error::Error>>(())
/// ```
pub struct LocalIndexStore<'a> {
    engine: &'a StorageEngine,
}

impl<'a> LocalIndexStore<'a> {
    /// Wrap a storage engine for index-store operations.
    pub fn new(engine: &'a StorageEngine) -> Self {
        Self { engine }
    }
}

fn decode_holder(bytes: &[u8]) -> StoreResult<NodeId> {
    match <[u8; 8]>::try_from(bytes) {
        Ok(raw) => Ok(NodeId::from_raw(u64::from_be_bytes(raw))),
        Err(_) => Err(StoreError::Decode {
            kind: "unique index entry",
            message: format!("{} bytes, expected 8", bytes.len()),
        }),
    }
}

/// Smallest key strictly greater than every key starting with `prefix`, the
/// exclusive end of the prefix's range. Every prefix built here ends in `:`,
/// so an end always exists.
fn prefix_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last < 0xFF {
            *last += 1;
            return end;
        }
        end.pop();
    }
    end
}

fn definition_key(name: &str) -> Vec<u8> {
    let mut key = Vec::with_capacity(11 + name.len());
    key.extend_from_slice(b"schema:idx:");
    key.extend_from_slice(name.as_bytes());
    key
}

impl IndexStore for LocalIndexStore<'_> {
    fn stage_membership(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        field_of: &dyn Fn(&str) -> Option<u32>,
        node_id: NodeId,
        old: Option<&[Value]>,
        new: Option<&[Value]>,
    ) -> StoreResult<usize> {
        let mut effects = membership_effects(&index.name, index.unique, node_id.as_raw(), old, new);
        if index.unique {
            // A value another node holds now (taken in this transaction) is
            // not this node's to release.
            let mut kept = Vec::with_capacity(effects.len());
            for effect in effects {
                let keep = effect.value.is_some()
                    || match txn.get(Partition::Idx, &effect.key)? {
                        Some(bytes) => decode_holder(&bytes)? == node_id,
                        None => false,
                    };
                if keep {
                    kept.push(effect);
                }
            }
            effects = kept;
        }
        let puts = effects.iter().filter(|e| e.value.is_some()).count();
        match index.maintenance.profile {
            IndexProfile::Resolved => {
                for effect in &effects {
                    match &effect.value {
                        Some(value) => txn.put(Partition::Idx, &effect.key, value)?,
                        None => txn.delete(Partition::Idx, &effect.key)?,
                    }
                }
            }
            IndexProfile::Derived => txn.stage_derived(
                &index.binding(field_of),
                node_id.as_raw(),
                old.map(<[Value]>::to_vec),
                new.map(<[Value]>::to_vec),
                &effects,
            )?,
        }
        Ok(puts)
    }

    fn unique_conflict(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        values: &[Value],
        node_id: NodeId,
    ) -> StoreResult<Option<NodeId>> {
        for tuple in tuples(values) {
            let key = encode_unique_index_key(&index.name, &tuple);
            if let Some(bytes) = txn.get(Partition::Idx, &key)? {
                let holder = decode_holder(&bytes)?;
                if holder != node_id {
                    return Ok(Some(holder));
                }
            }
        }
        Ok(None)
    }

    fn committed_conflict(
        &self,
        index: &IndexDefinition,
        values: &[Value],
        node_id: NodeId,
    ) -> StoreResult<Option<NodeId>> {
        for tuple in tuples(values) {
            let key = encode_unique_index_key(&index.name, &tuple);
            if let Some(bytes) = self.engine.get(Partition::Idx, &key)? {
                let holder = decode_holder(&bytes)?;
                if holder != node_id {
                    return Ok(Some(holder));
                }
            }
        }
        Ok(None)
    }

    fn scan_exact(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        values: &[Value],
    ) -> StoreResult<Option<Vec<NodeId>>> {
        let Ok(tuple) = encode_tuple(values) else {
            return Ok(None);
        };
        if index.unique {
            let key = encode_unique_index_key(&index.name, &tuple);
            return match txn.get(Partition::Idx, &key)? {
                Some(bytes) => Ok(Some(vec![decode_holder(&bytes)?])),
                None => Ok(Some(Vec::new())),
            };
        }
        let prefix = index_value_prefix(&index.name, &tuple);
        let mut out = Vec::new();
        for (key, _) in txn.prefix_scan(Partition::Idx, &prefix)? {
            // The scan overlays buffered values but not buffered tombstones:
            // an entry this transaction removed is gone for it.
            if matches!(txn.buffered(Partition::Idx, &key), Some(None)) {
                continue;
            }
            if let Some(id) = decode_node_id(&key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(Some(out))
    }

    fn scan_entry_ids(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
    ) -> StoreResult<Vec<NodeId>> {
        let prefix = if index.unique {
            unique_index_prefix(&index.name)
        } else {
            index_prefix(&index.name)
        };
        // The scan appends this transaction's own entries after the stored
        // ones; key order is restored here.
        let mut entries = txn.prefix_scan(Partition::Idx, &prefix)?;
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut out = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if matches!(txn.buffered(Partition::Idx, &key), Some(None)) {
                continue;
            }
            if index.unique {
                out.push(decode_holder(&value)?);
            } else if let Some(id) = decode_node_id(&key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(out)
    }

    fn entry_delete_mutations(
        &self,
        index: &IndexDefinition,
        field_of: &dyn Fn(&str) -> Option<u32>,
        values: &[Value],
        node_id: NodeId,
    ) -> Vec<Mutation> {
        match index.maintenance.profile {
            IndexProfile::Resolved => membership_effects(
                &index.name,
                index.unique,
                node_id.as_raw(),
                Some(values),
                None,
            )
            .into_iter()
            .map(|effect| Mutation::Delete {
                partition: PartitionId::Idx,
                key: effect.key,
            })
            .collect(),
            IndexProfile::Derived => vec![Mutation::Derive(DerivedIndexWork {
                binding: index.binding(field_of),
                node_id: node_id.as_raw(),
                old: Some(values.to_vec()),
                new: DerivedSource::Values(None),
            })],
        }
    }

    fn clear_mutations(&self, name: &str) -> Vec<Mutation> {
        [index_prefix(name), unique_index_prefix(name)]
            .into_iter()
            .map(|start| Mutation::RemoveRange {
                partition: PartitionId::Idx,
                end: prefix_end(&start),
                start,
            })
            .collect()
    }

    fn definition_put_mutation(&self, def: &IndexDefinition) -> StoreResult<Mutation> {
        Ok(Mutation::Put {
            partition: PartitionId::Schema,
            key: def.schema_key(),
            value: rmp_serde::to_vec(def)
                .map_err(|e| StoreError::Invariant(format!("index definition serialize: {e}")))?,
        })
    }

    fn definition_delete_mutation(&self, name: &str) -> Mutation {
        Mutation::Delete {
            partition: PartitionId::Schema,
            key: definition_key(name),
        }
    }

    fn apply_unreplicated(&self, mutations: &[Mutation]) -> StoreResult<()> {
        // One batch, as a proposal applies: commands decided, DERIVED work
        // derived, all effects visible together.
        Ok(self.engine.apply_proposal_at(mutations, 0)?)
    }

    fn clear_legacy(&self, name: &str) -> StoreResult<()> {
        let start = legacy_index_prefix(name);
        self.engine
            .remove_range(Partition::Idx, &start, &prefix_end(&start))?;
        Ok(())
    }

    fn put_definition(&self, def: &IndexDefinition) -> StoreResult<()> {
        let value = rmp_serde::to_vec(def)
            .map_err(|e| StoreError::Invariant(format!("index definition serialize: {e}")))?;
        self.engine
            .put(Partition::Schema, &def.schema_key(), &value)?;
        Ok(())
    }

    fn load_definition(&self, name: &str) -> StoreResult<Option<IndexDefinition>> {
        match self.engine.get(Partition::Schema, &definition_key(name))? {
            Some(bytes) => Ok(Some(rmp_serde::from_slice(&bytes).map_err(|e| {
                StoreError::Decode {
                    kind: "index definition",
                    message: e.to_string(),
                }
            })?)),
            None => Ok(None),
        }
    }

    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>> {
        let mut out = Vec::new();
        for guard in self.engine.prefix_scan(Partition::Schema, b"schema:idx:")? {
            let (_key, value) = guard.into_inner()?;
            match rmp_serde::from_slice::<IndexDefinition>(&value) {
                Ok(def) => out.push(def),
                Err(e) => tracing::warn!("list_definitions: skipping corrupt index def: {e}"),
            }
        }
        Ok(out)
    }

    fn set_definition_state(&self, name: &str, state: IndexState) -> StoreResult<bool> {
        let Some(mut def) = self.load_definition(name)? else {
            return Ok(false);
        };
        def.state = state;
        self.put_definition(&def)?;
        Ok(true)
    }

    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()> {
        let value = rmp_serde::to_vec(def)
            .map_err(|e| StoreError::Invariant(format!("index definition serialize: {e}")))?;
        txn.put(Partition::Schema, &def.schema_key(), &value)?;
        Ok(())
    }

    fn delete_definition_txn(&self, txn: &mut Transaction, name: &str) -> StoreResult<()> {
        txn.delete(Partition::Schema, &definition_key(name))?;
        Ok(())
    }

    fn index_policy(&self) -> StoreResult<(NamespaceIndexPolicy, Option<u64>)> {
        let key = NamespaceIndexPolicy::KEY;
        let version = self.engine.record_version(Partition::Schema, key)?;
        let policy = match self.engine.get(Partition::Schema, key)? {
            Some(bytes) => rmp_serde::from_slice(&bytes).map_err(|e| StoreError::Decode {
                kind: "index policy",
                message: e.to_string(),
            })?,
            None => NamespaceIndexPolicy::default(),
        };
        Ok((policy, version))
    }

    fn put_index_policy_txn(
        &self,
        txn: &mut Transaction,
        policy: &NamespaceIndexPolicy,
        version: Option<u64>,
    ) -> StoreResult<()> {
        let value = rmp_serde::to_vec(policy)
            .map_err(|e| StoreError::Invariant(format!("index policy serialize: {e}")))?;
        txn.expect_version(Partition::Schema, NamespaceIndexPolicy::KEY, version)?;
        txn.put(Partition::Schema, NamespaceIndexPolicy::KEY, &value)?;
        Ok(())
    }

    fn definition_version(&self, name: &str) -> StoreResult<Option<u64>> {
        Ok(self
            .engine
            .record_version(Partition::Schema, &definition_key(name))?)
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
