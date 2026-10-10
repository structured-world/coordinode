//! Index store: secondary B-tree entries in [`Partition::Idx`] and the
//! index-definition catalog in [`Partition::Schema`].
//!
//! Entries are staged in the writing statement's [`Transaction`], so they
//! commit in the same log entry and at the same version as the data they
//! index: a crash keeps both or neither, a replica applies them like any other
//! key, a rolled-back statement leaves none behind, and a scan inside the
//! transaction sees its own writes.
//!
//! Two entry shapes, keyed by the index generation and the values' tuple
//! encoding ([`coordinode_core::index::encoding`]):
//!
//! - a non-unique index writes `<generation><tuple>:<node_id>` with an
//!   empty value, one entry per node, found by a prefix scan; on a temporal
//!   label the key also carries the version's `valid_from`, one entry per
//!   version;
//! - a unique index writes `<generation><tuple>` whose value is the holder's
//!   node id. Keyed by the value alone, the entry is its own uniqueness claim:
//!   two transactions inserting one value write one key, and write-write
//!   conflict detection lets only one of them commit. Checking the value is a
//!   point read the partition's bloom filters answer without touching tables
//!   that cannot hold it.
//!
//! A list value indexes each of its elements (multikey). A value with no key
//! (NaN, a map, a vector) is not indexed; a lookup of it reports so, and the
//! caller answers with a scan.
//!
//! The catalog keeps a definition under its [`IndexId`], a name binding per
//! named index, and the identity allocator. Publishing a definition allocates
//! its identities and binds its name in the publishing transaction, so a
//! failed or refused publication takes nothing, and no number is reused.

use coordinode_core::graph::node::NodeId;
use coordinode_core::graph::types::Value;
use coordinode_core::index::derive::{EntryOwner, entry, membership_effects, tuples};
use coordinode_core::index::encoding::{
    decode_entry, decode_entry_parts, decode_unique_entry_tuple, encode_element, encode_tuple,
    encode_unique_entry_key, entries_prefix, entry_value_prefix, generation_ranges,
    unique_entries_prefix,
};
use coordinode_core::index::identity::IdentityAllocator;
use coordinode_core::txn::proposal::Mutation;
use coordinode_storage::Guard;
use coordinode_storage::engine::core::StorageEngine;
use coordinode_storage::engine::partition::Partition;
use coordinode_storage::engine::transaction::Transaction;
use coordinode_storage::error::StorageError;

use crate::error::{StoreError, StoreResult};
use crate::index_def::{
    DuplicateRepairRecord, GenerationId, IndexBuildRecord, IndexDefinition, IndexDescriptor,
    IndexId, IndexIntegrityRecord, IndexProfile, NamespaceIndexPolicy,
};

/// One page of an ordered read of index entries
/// ([`IndexStore::scan_entries_in`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntryPage {
    /// The entries' owners in value order: the node, and the version's
    /// `valid_from` for an entry of one version of a temporal node.
    pub entries: Vec<(NodeId, Option<i64>)>,
    /// The key of the last entry read, where the next page resumes; `None`
    /// for an empty page.
    pub resume: Option<Vec<u8>>,
    /// No entry in the range follows this page.
    pub exhausted: bool,
}

/// One stored entry of a generation, as a check reads it: the value tuple
/// and the owner it names. A unique entry whose holder does not decode
/// names no owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEntry {
    /// The encoded value tuple.
    pub tuple: Vec<u8>,
    /// The node, or temporal version, the entry names; `None` for a unique
    /// entry whose holder does not decode.
    pub owner: Option<EntryOwner>,
}

/// One page of a generation's stored entries in key order
/// ([`IndexStore::entries_page`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoredEntryPage {
    /// The entries.
    pub entries: Vec<StoredEntry>,
    /// The key of the last entry read, where the next page resumes; `None`
    /// for an empty page.
    pub resume: Option<Vec<u8>>,
    /// No entry of the generation follows this page.
    pub exhausted: bool,
}

/// The latest committed state of one entry key, with the version of that
/// write: what a repair conditions its commit on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatestEntry {
    /// The holder a unique entry names (`None` when it does not decode);
    /// always `None` for a non-unique entry.
    pub holder: Option<NodeId>,
    /// The version of the write that left it.
    pub version: u64,
}

/// Layer 4 store for secondary B-tree entries and the index catalog.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not store index entries",
    label = "an index store is required here",
    note = "use `LocalIndexStore`, the CE implementation over a statement transaction"
)]
pub trait IndexStore {
    /// Stage the entry changes of `owner`'s membership in `index` moving
    /// from `old` to `new` (`None`: no entry), in the index's profile. The
    /// owner is a node, or one version of a temporal node. A RESOLVED index
    /// stages the entries as writes the unit logs; a DERIVED one stages them
    /// for this transaction's reads and conflicts, and the unit logs the
    /// change sealed under `index`'s binding, with property field ids from
    /// `field_of`. A unique entry is removed only while the owner's node
    /// holds it; for a unique value the caller has checked
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
        owner: EntryOwner,
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

    /// The nodes with an entry holding exactly `values`, as the transaction
    /// sees it, each once (a temporal node has an entry per version that
    /// holds them). `None` when the values have no key, so the index cannot
    /// answer.
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

    /// One page of the entries of the single-property `index` whose value is
    /// in `[from, to)`, in value order, after the entry key `after` when
    /// given: the ordered read that finds the values below a bound without
    /// reading the rest. `None` when `from` or `to` has no key.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable entry.
    fn scan_entries_in(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        from: &Value,
        to: &Value,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Option<EntryPage>>;

    /// The node of every entry in `index`, in key order: a temporal node once
    /// per version with an entry.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable entry.
    fn scan_entry_ids(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
    ) -> StoreResult<Vec<NodeId>>;

    /// Stage the removal of every entry of the index generation `generation`,
    /// of both shapes, with the transaction's commit: for DDL that drops or
    /// replaces the definition in the same commit, which fences every writer
    /// bound to it.
    ///
    /// # Errors
    ///
    /// A storage failure in legacy (no-MVCC) mode, which removes at once.
    fn clear_txn(&self, txn: &mut Transaction, generation: GenerationId) -> StoreResult<()>;

    /// Apply one unit of mutations straight to the engine as one batch, for a
    /// context that has no log to replicate them through.
    ///
    /// # Errors
    ///
    /// A storage failure, or DERIVED work that cannot be derived.
    fn apply_unreplicated(&self, mutations: &[Mutation]) -> StoreResult<()>;

    /// Load a persisted index definition.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable definition.
    fn load_definition(&self, id: IndexId) -> StoreResult<Option<IndexDefinition>>;

    /// The live index the name `name` binds, if any.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable binding.
    fn resolve_name(&self, name: &str) -> StoreResult<Option<IndexId>>;

    /// Every persisted index definition in identity order. A definition
    /// whose bytes do not decode is skipped with a warning, so one corrupt
    /// record does not take down the registry on open.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>>;

    /// Publish a new index through a statement [`Transaction`]: allocate its
    /// identity and first generation, bind its name if it has one, and stage
    /// its definition, all with the statement's commit. The allocation is
    /// conditioned on the allocator record, so two concurrent publications
    /// cannot take one number, and a name another live index holds refuses
    /// the publication without staging anything.
    ///
    /// # Errors
    ///
    /// [`StoreError::IndexNameTaken`], [`StoreError::IdentityExhausted`], or
    /// a storage or encoding failure.
    fn publish_definition_txn(
        &self,
        txn: &mut Transaction,
        descriptor: IndexDescriptor,
    ) -> StoreResult<IndexDefinition>;

    /// Allocate a new generation for `def` through a statement
    /// [`Transaction`], for a rebuild that writes a new representation. The
    /// caller stages the definition that serves from it.
    ///
    /// # Errors
    ///
    /// As [`Self::publish_definition_txn`].
    fn allocate_generation_txn(&self, txn: &mut Transaction) -> StoreResult<GenerationId>;

    /// Persist an existing index definition through a statement
    /// [`Transaction`]: it commits, and replicates, with the statement. The
    /// name binding is not touched.
    ///
    /// # Errors
    ///
    /// An encoding failure.
    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()>;

    /// Delete a persisted index definition and its name binding through a
    /// statement [`Transaction`]. Tombstone semantics: no error when absent.
    /// The identities stay taken.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn delete_definition_txn(
        &self,
        txn: &mut Transaction,
        def: &IndexDefinition,
    ) -> StoreResult<()>;

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

    /// The version of the stored definition `id`, the value a writer binds
    /// its effects to so a transition between staging and commit refuses it.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn definition_version(&self, id: IndexId) -> StoreResult<Option<u64>>;

    /// Write the transaction only while the definition record of `id` is at
    /// `version` when it commits, so DDL and builds act on the definition
    /// they inspected.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn expect_definition_txn(
        &self,
        txn: &mut Transaction,
        id: IndexId,
        version: Option<u64>,
    ) -> StoreResult<()>;

    /// The version of the binding of the name `name`.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn name_version(&self, name: &str) -> StoreResult<Option<u64>>;

    /// Write the transaction only while the binding of `name` is at
    /// `version` when it commits (`None`: while no index holds the name), so
    /// a statement that resolved a name acts on the index it resolved.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn expect_name_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
        version: Option<u64>,
    ) -> StoreResult<()>;

    /// The record of the build of `generation` with the version of the same
    /// write, read together: an executor that moves the build conditions its
    /// commit on exactly the record it read.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn load_build(&self, generation: GenerationId) -> StoreResult<Option<(IndexBuildRecord, u64)>>;

    /// Every build record in generation order. A record whose bytes do not
    /// decode is skipped with a warning.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn list_builds(&self) -> StoreResult<Vec<IndexBuildRecord>>;

    /// Stage `record` through a statement [`Transaction`], only while the
    /// stored record of its generation is at `version` when it commits
    /// (`None`: while there is none), so two movers of one build cannot
    /// both commit.
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn put_build_txn(
        &self,
        txn: &mut Transaction,
        record: &IndexBuildRecord,
        version: Option<u64>,
    ) -> StoreResult<()>;

    /// Stage `repair` through the [`Transaction`] that makes the data change
    /// it records, so the record exists exactly when the change does.
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn put_repair_txn(
        &self,
        txn: &mut Transaction,
        repair: &DuplicateRepairRecord,
    ) -> StoreResult<()>;

    /// The repairs the build of `generation` committed, in node order.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn list_repairs(&self, generation: GenerationId) -> StoreResult<Vec<DuplicateRepairRecord>>;

    /// Stage the removal of the finished build records of the index `index`,
    /// with the repairs they record, in the commit that drops it. A build
    /// still without an outcome is its executor's to end: it finds the index
    /// gone and removes the record itself, so the drop and the executor never
    /// write one record at once.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn delete_finished_builds_txn(&self, txn: &mut Transaction, index: IndexId) -> StoreResult<()>;

    /// Stage the removal of the build record of `generation`, only while it
    /// is at `version` when the transaction commits: an executor removing
    /// the record of a build whose index is gone.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn delete_build_txn(
        &self,
        txn: &mut Transaction,
        generation: GenerationId,
        version: u64,
    ) -> StoreResult<()>;

    /// The integrity record of `generation` with the version of the same
    /// write, read together, so a move conditions its commit on exactly the
    /// record it read.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn load_integrity(
        &self,
        generation: GenerationId,
    ) -> StoreResult<Option<(IndexIntegrityRecord, u64)>>;

    /// Every integrity record in generation order.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn list_integrity(&self) -> StoreResult<Vec<IndexIntegrityRecord>>;

    /// Stage `record` through a statement [`Transaction`], only while the
    /// stored record of its generation is at `version` when it commits
    /// (`None`: while there is none).
    ///
    /// # Errors
    ///
    /// A storage or encoding failure.
    fn put_integrity_txn(
        &self,
        txn: &mut Transaction,
        record: &IndexIntegrityRecord,
        version: Option<u64>,
    ) -> StoreResult<()>;

    /// Stage the removal of the integrity records of the index `index`, in
    /// the commit that drops it.
    ///
    /// # Errors
    ///
    /// A storage failure or an undecodable record.
    fn delete_integrity_txn(&self, txn: &mut Transaction, index: IndexId) -> StoreResult<()>;

    /// One page of the stored entries of `index`'s generation after the
    /// entry key `after`, at most `limit`, in key order, as `txn` sees them.
    /// Untracked: a check reads, it does not claim.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn entries_page(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<StoredEntryPage>;

    /// The holder the unique entry of `tuple` in `index` names, as `txn`
    /// sees it: `None` when there is no entry, `Some(None)` when its holder
    /// does not decode. Untracked.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn unique_holder(
        &self,
        txn: &Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
    ) -> StoreResult<Option<Option<NodeId>>>;

    /// Whether the non-unique `index` has the entry of `owner` under
    /// `tuple`, as `txn` sees it. Untracked.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn has_entry(
        &self,
        txn: &Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
    ) -> StoreResult<bool>;

    /// The latest committed state of the entry `owner` holds under `tuple`
    /// in `index` (for a unique index, the entry of `tuple` whoever it
    /// names), with the version of its write; `None` when there is none.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn latest_entry(
        &self,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
    ) -> StoreResult<Option<LatestEntry>>;

    /// Stage a repair of one entry of `index` through a statement
    /// [`Transaction`]: `owner`'s entry under `tuple` put (`present`) or
    /// removed, only while the entry is still at `version` when the
    /// transaction commits (`None`: while there is none). The repair is an
    /// exact entry effect in either profile: every member applies the same
    /// key.
    ///
    /// # Errors
    ///
    /// A storage failure.
    fn repair_entry_txn(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
        present: bool,
        version: Option<u64>,
    ) -> StoreResult<()>;
}

/// CE implementation of [`IndexStore`].
///
/// # Examples
///
/// ```no_run
/// use coordinode_modality::{IndexDescriptor, IndexStore, LocalIndexStore};
/// use coordinode_core::graph::{node::NodeId, types::Value};
/// # use coordinode_storage::engine::{config::*, core::StorageEngine, transaction::Transaction};
/// # use coordinode_core::txn::timestamp::Timestamp;
/// # let cfg = StorageConfig::with_endpoints(vec![EndpointConfig::new(
/// #     "ep", std::path::Path::new("/tmp/x"),
/// #     Media::Hdd, Durability::Durable, Tier::Warm)]);
/// # let engine = StorageEngine::open(&cfg)?;
/// # let mut txn = Transaction::begin(&engine, None, Timestamp::from_raw(1));
/// let store = LocalIndexStore::new(&engine);
/// let index = store.publish_definition_txn(
///     &mut txn,
///     IndexDescriptor::btree("user_email", "User", "email").unique(),
/// )?;
/// let email = [Value::String("a@x".into())];
/// assert_eq!(store.unique_conflict(&mut txn, &index, &email, NodeId::from_raw(1))?, None);
/// let no_fields = |_: &str| None;
/// let owner = coordinode_core::index::derive::EntryOwner::node(1);
/// store.stage_membership(&mut txn, &index, &no_fields, owner, None, Some(&email))?;
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

fn encode<T: serde::Serialize>(what: &'static str, value: &T) -> StoreResult<Vec<u8>> {
    rmp_serde::to_vec(value).map_err(|e| StoreError::Invariant(format!("{what} serialize: {e}")))
}

fn decode<T: serde::de::DeserializeOwned>(kind: &'static str, bytes: &[u8]) -> StoreResult<T> {
    rmp_serde::from_slice(bytes).map_err(|e| StoreError::Decode {
        kind,
        message: e.to_string(),
    })
}

/// Decode a catalog record listed at `key`. One that does not decode refuses
/// the listing: a catalog read without it would serve the database without
/// that index, or that build, as if it had never been created.
fn decode_catalog<T: serde::de::DeserializeOwned>(
    kind: &'static str,
    key: &[u8],
    bytes: &[u8],
) -> StoreResult<T> {
    rmp_serde::from_slice(bytes).map_err(|e| {
        StoreError::Storage(StorageError::UnreadableCatalog {
            kind,
            key: coordinode_storage::error::printable_key(key),
            detail: e.to_string(),
        })
    })
}

fn decode_index_id(bytes: &[u8]) -> StoreResult<IndexId> {
    match <[u8; 8]>::try_from(bytes) {
        Ok(raw) => Ok(IndexId::from_raw(u64::from_be_bytes(raw))),
        Err(_) => Err(StoreError::Decode {
            kind: "index name binding",
            message: format!("{} bytes, expected 8", bytes.len()),
        }),
    }
}

impl LocalIndexStore<'_> {
    /// The allocator as this transaction sees it, conditioning the
    /// transaction on the record it read. A transaction that already
    /// allocated reads its own staged record, which is conditioned already.
    fn allocator_txn(&self, txn: &mut Transaction) -> StoreResult<IdentityAllocator> {
        let key = IndexDefinition::ALLOCATOR_KEY;
        if let Some(Some(bytes)) = txn.buffered(Partition::Schema, key) {
            return decode("index identity allocator", bytes);
        }
        // The value and the version of one write, read together: a version
        // newer than the value would let the condition pass on a stale
        // allocator and hand out a number twice.
        let (allocator, version) = match self.engine.get_versioned(Partition::Schema, key)? {
            Some((bytes, version)) => (decode("index identity allocator", &bytes)?, Some(version)),
            None => (IdentityAllocator::default(), None),
        };
        txn.expect_version(Partition::Schema, key, version)?;
        Ok(allocator)
    }

    fn put_allocator_txn(
        &self,
        txn: &mut Transaction,
        allocator: &IdentityAllocator,
    ) -> StoreResult<()> {
        let value = encode("index identity allocator", allocator)?;
        txn.put(Partition::Schema, IndexDefinition::ALLOCATOR_KEY, &value)?;
        Ok(())
    }
}

impl IndexStore for LocalIndexStore<'_> {
    fn stage_membership(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        field_of: &dyn Fn(&str) -> Option<u32>,
        owner: EntryOwner,
        old: Option<&[Value]>,
        new: Option<&[Value]>,
    ) -> StoreResult<usize> {
        let node_id = NodeId::from_raw(owner.node_id);
        let mut effects = membership_effects(index.generation, index.unique, owner, old, new);
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
                owner,
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
            let key = encode_unique_entry_key(index.generation, &tuple);
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
            let key = encode_unique_entry_key(index.generation, &tuple);
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
            let key = encode_unique_entry_key(index.generation, &tuple);
            return match txn.get(Partition::Idx, &key)? {
                Some(bytes) => Ok(Some(vec![decode_holder(&bytes)?])),
                None => Ok(Some(Vec::new())),
            };
        }
        let prefix = entry_value_prefix(index.generation, &tuple);
        let mut out = Vec::new();
        for (key, _) in txn.prefix_scan(Partition::Idx, &prefix)? {
            if let Some((id, _)) = decode_entry(index.generation, &key) {
                out.push(NodeId::from_raw(id));
            }
        }
        // A temporal node's versions holding the value are one node. Entries
        // of one value come in key order, which orders them by node, so its
        // versions are adjacent.
        out.dedup();
        Ok(Some(out))
    }

    fn scan_entries_in(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        from: &Value,
        to: &Value,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<Option<EntryPage>> {
        let prefix = if index.unique {
            unique_entries_prefix(index.generation)
        } else {
            entries_prefix(index.generation)
        };
        // An element is self-delimiting and sorts in value order, so every
        // entry holding a value in [from, to) lies between the generation
        // prefix followed by each bound.
        let mut start = prefix.to_vec();
        let mut end = prefix.to_vec();
        if encode_element(from, &mut start).is_err() || encode_element(to, &mut end).is_err() {
            return Ok(None);
        }
        let page = txn.range_scan_paged(Partition::Idx, &start, &end, after, limit)?;
        let mut entries = Vec::with_capacity(page.rows.len());
        for (key, value) in &page.rows {
            if index.unique {
                entries.push((decode_holder(value)?, None));
            } else if let Some((id, valid_from)) = decode_entry(index.generation, key) {
                entries.push((NodeId::from_raw(id), valid_from));
            }
        }
        Ok(Some(EntryPage {
            entries,
            resume: page.last_key,
            exhausted: page.exhausted,
        }))
    }

    fn scan_entry_ids(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
    ) -> StoreResult<Vec<NodeId>> {
        let prefix = if index.unique {
            unique_entries_prefix(index.generation)
        } else {
            entries_prefix(index.generation)
        };
        let entries = txn.prefix_scan(Partition::Idx, &prefix)?;
        let mut out = Vec::with_capacity(entries.len());
        for (key, value) in entries {
            if index.unique {
                out.push(decode_holder(&value)?);
            } else if let Some((id, _)) = decode_entry(index.generation, &key) {
                out.push(NodeId::from_raw(id));
            }
        }
        Ok(out)
    }

    fn clear_txn(&self, txn: &mut Transaction, generation: GenerationId) -> StoreResult<()> {
        for (start, end) in generation_ranges(generation) {
            txn.remove_range(Partition::Idx, &start, &end)?;
        }
        Ok(())
    }

    fn apply_unreplicated(&self, mutations: &[Mutation]) -> StoreResult<()> {
        // One batch, as a proposal applies: commands decided, DERIVED work
        // derived, all effects visible together.
        Ok(self.engine.apply_proposal_at(mutations, 0)?)
    }

    fn load_definition(&self, id: IndexId) -> StoreResult<Option<IndexDefinition>> {
        self.engine
            .get(Partition::Schema, &IndexDefinition::schema_key_of(id))?
            .map(|bytes| decode("index definition", &bytes))
            .transpose()
    }

    fn resolve_name(&self, name: &str) -> StoreResult<Option<IndexId>> {
        self.engine
            .get(Partition::Schema, &IndexDefinition::name_key_of(name))?
            .map(|bytes| decode_index_id(&bytes))
            .transpose()
    }

    fn list_definitions(&self) -> StoreResult<Vec<IndexDefinition>> {
        let mut out = Vec::new();
        for guard in self
            .engine
            .prefix_scan(Partition::Schema, IndexDefinition::SCHEMA_PREFIX)?
        {
            let (key, value) = guard.into_inner()?;
            out.push(decode_catalog("index definition", &key, &value)?);
        }
        Ok(out)
    }

    fn publish_definition_txn(
        &self,
        txn: &mut Transaction,
        descriptor: IndexDescriptor,
    ) -> StoreResult<IndexDefinition> {
        if let Some(name) = &descriptor.name {
            let key = IndexDefinition::name_key_of(name);
            if txn.get(Partition::Schema, &key)?.is_some() {
                return Err(StoreError::IndexNameTaken(name.clone()));
            }
            // Bound only while it is free when the statement commits: two
            // publications of one name cannot both commit.
            txn.expect_version(Partition::Schema, &key, None)?;
        }
        let mut allocator = self.allocator_txn(txn)?;
        let id = allocator.allocate_index()?;
        let generation = allocator.allocate_generation()?;
        self.put_allocator_txn(txn, &allocator)?;
        let def = descriptor.bind(id, generation);
        if let Some(name) = &def.name {
            txn.put(
                Partition::Schema,
                &IndexDefinition::name_key_of(name),
                &id.as_raw().to_be_bytes(),
            )?;
        }
        self.put_definition_txn(txn, &def)?;
        Ok(def)
    }

    fn allocate_generation_txn(&self, txn: &mut Transaction) -> StoreResult<GenerationId> {
        let mut allocator = self.allocator_txn(txn)?;
        let generation = allocator.allocate_generation()?;
        self.put_allocator_txn(txn, &allocator)?;
        Ok(generation)
    }

    fn put_definition_txn(&self, txn: &mut Transaction, def: &IndexDefinition) -> StoreResult<()> {
        let value = encode("index definition", def)?;
        txn.put(Partition::Schema, &def.schema_key(), &value)?;
        Ok(())
    }

    fn delete_definition_txn(
        &self,
        txn: &mut Transaction,
        def: &IndexDefinition,
    ) -> StoreResult<()> {
        txn.delete(Partition::Schema, &def.schema_key())?;
        if let Some(name) = &def.name {
            let key = IndexDefinition::name_key_of(name);
            // The binding goes only while it is this index's: a name it no
            // longer holds belongs to another index.
            match txn.get(Partition::Schema, &key)? {
                Some(bytes) if decode_index_id(&bytes)? == def.id => {
                    txn.delete(Partition::Schema, &key)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn index_policy(&self) -> StoreResult<(NamespaceIndexPolicy, Option<u64>)> {
        // The value and the version of one write, read together, so a
        // change conditioned on the version builds on this value.
        match self
            .engine
            .get_versioned(Partition::Schema, NamespaceIndexPolicy::KEY)?
        {
            Some((bytes, version)) => Ok((decode("index policy", &bytes)?, Some(version))),
            None => Ok((NamespaceIndexPolicy::default(), None)),
        }
    }

    fn put_index_policy_txn(
        &self,
        txn: &mut Transaction,
        policy: &NamespaceIndexPolicy,
        version: Option<u64>,
    ) -> StoreResult<()> {
        let value = encode("index policy", policy)?;
        txn.expect_version(Partition::Schema, NamespaceIndexPolicy::KEY, version)?;
        txn.put(Partition::Schema, NamespaceIndexPolicy::KEY, &value)?;
        Ok(())
    }

    fn definition_version(&self, id: IndexId) -> StoreResult<Option<u64>> {
        Ok(self
            .engine
            .record_version(Partition::Schema, &IndexDefinition::schema_key_of(id))?)
    }

    fn expect_definition_txn(
        &self,
        txn: &mut Transaction,
        id: IndexId,
        version: Option<u64>,
    ) -> StoreResult<()> {
        Ok(txn.expect_version(
            Partition::Schema,
            &IndexDefinition::schema_key_of(id),
            version,
        )?)
    }

    fn name_version(&self, name: &str) -> StoreResult<Option<u64>> {
        Ok(self
            .engine
            .record_version(Partition::Schema, &IndexDefinition::name_key_of(name))?)
    }

    fn expect_name_txn(
        &self,
        txn: &mut Transaction,
        name: &str,
        version: Option<u64>,
    ) -> StoreResult<()> {
        Ok(txn.expect_version(
            Partition::Schema,
            &IndexDefinition::name_key_of(name),
            version,
        )?)
    }

    fn load_build(&self, generation: GenerationId) -> StoreResult<Option<(IndexBuildRecord, u64)>> {
        self.engine
            .get_versioned(Partition::Schema, &IndexBuildRecord::key_of(generation))?
            .map(|(bytes, version)| Ok((decode("index build record", &bytes)?, version)))
            .transpose()
    }

    fn list_builds(&self) -> StoreResult<Vec<IndexBuildRecord>> {
        let mut out = Vec::new();
        for guard in self
            .engine
            .prefix_scan(Partition::Schema, IndexBuildRecord::PREFIX)?
        {
            let (key, value) = guard.into_inner()?;
            out.push(decode_catalog("index build record", &key, &value)?);
        }
        Ok(out)
    }

    fn put_build_txn(
        &self,
        txn: &mut Transaction,
        record: &IndexBuildRecord,
        version: Option<u64>,
    ) -> StoreResult<()> {
        let key = IndexBuildRecord::key_of(record.generation);
        txn.expect_version(Partition::Schema, &key, version)?;
        txn.put(
            Partition::Schema,
            &key,
            &encode("index build record", record)?,
        )?;
        Ok(())
    }

    fn put_repair_txn(
        &self,
        txn: &mut Transaction,
        repair: &DuplicateRepairRecord,
    ) -> StoreResult<()> {
        txn.put(
            Partition::Schema,
            &repair.key(),
            &encode("index build repair", repair)?,
        )?;
        Ok(())
    }

    fn list_repairs(&self, generation: GenerationId) -> StoreResult<Vec<DuplicateRepairRecord>> {
        let mut out = Vec::new();
        for guard in self.engine.prefix_scan(
            Partition::Schema,
            &DuplicateRepairRecord::prefix_of(generation),
        )? {
            let (key, value) = guard.into_inner()?;
            out.push(decode_catalog("index build repair", &key, &value)?);
        }
        Ok(out)
    }

    fn delete_finished_builds_txn(&self, txn: &mut Transaction, index: IndexId) -> StoreResult<()> {
        for record in self.list_builds()? {
            if record.index == index && record.state.is_terminal() {
                txn.delete(
                    Partition::Schema,
                    &IndexBuildRecord::key_of(record.generation),
                )?;
                self.delete_repairs_txn(txn, record.generation)?;
            }
        }
        Ok(())
    }

    fn delete_build_txn(
        &self,
        txn: &mut Transaction,
        generation: GenerationId,
        version: u64,
    ) -> StoreResult<()> {
        let key = IndexBuildRecord::key_of(generation);
        txn.expect_version(Partition::Schema, &key, Some(version))?;
        txn.delete(Partition::Schema, &key)?;
        self.delete_repairs_txn(txn, generation)
    }

    fn load_integrity(
        &self,
        generation: GenerationId,
    ) -> StoreResult<Option<(IndexIntegrityRecord, u64)>> {
        self.engine
            .get_versioned(Partition::Schema, &IndexIntegrityRecord::key_of(generation))?
            .map(|(bytes, version)| Ok((decode("index integrity record", &bytes)?, version)))
            .transpose()
    }

    fn list_integrity(&self) -> StoreResult<Vec<IndexIntegrityRecord>> {
        let mut out = Vec::new();
        for guard in self
            .engine
            .prefix_scan(Partition::Schema, IndexIntegrityRecord::PREFIX)?
        {
            let (key, value) = guard.into_inner()?;
            out.push(decode_catalog("index integrity record", &key, &value)?);
        }
        Ok(out)
    }

    fn put_integrity_txn(
        &self,
        txn: &mut Transaction,
        record: &IndexIntegrityRecord,
        version: Option<u64>,
    ) -> StoreResult<()> {
        let key = IndexIntegrityRecord::key_of(record.generation);
        txn.expect_version(Partition::Schema, &key, version)?;
        txn.put(
            Partition::Schema,
            &key,
            &encode("index integrity record", record)?,
        )?;
        Ok(())
    }

    fn delete_integrity_txn(&self, txn: &mut Transaction, index: IndexId) -> StoreResult<()> {
        for record in self.list_integrity()? {
            if record.index == index {
                txn.delete(
                    Partition::Schema,
                    &IndexIntegrityRecord::key_of(record.generation),
                )?;
            }
        }
        Ok(())
    }

    fn entries_page(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        after: Option<&[u8]>,
        limit: usize,
    ) -> StoreResult<StoredEntryPage> {
        let [plain, unique] = generation_ranges(index.generation);
        let (start, end) = if index.unique { unique } else { plain };
        let page = txn.range_scan_paged(Partition::Idx, &start, &end, after, limit)?;
        let mut entries = Vec::with_capacity(page.rows.len());
        for (key, value) in &page.rows {
            if index.unique {
                if let Some(tuple) = decode_unique_entry_tuple(index.generation, key) {
                    entries.push(StoredEntry {
                        tuple: tuple.to_vec(),
                        owner: decode_holder(value)
                            .ok()
                            .map(|holder| EntryOwner::node(holder.as_raw())),
                    });
                }
            } else if let Some((tuple, node_id, valid_from)) =
                decode_entry_parts(index.generation, key)
            {
                entries.push(StoredEntry {
                    tuple: tuple.to_vec(),
                    owner: Some(EntryOwner {
                        node_id,
                        valid_from,
                    }),
                });
            }
        }
        Ok(StoredEntryPage {
            entries,
            resume: page.last_key,
            exhausted: page.exhausted,
        })
    }

    fn unique_holder(
        &self,
        txn: &Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
    ) -> StoreResult<Option<Option<NodeId>>> {
        let key = encode_unique_entry_key(index.generation, tuple);
        Ok(txn
            .read_untracked(Partition::Idx, &key)?
            .map(|bytes| decode_holder(&bytes).ok()))
    }

    fn has_entry(
        &self,
        txn: &Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
    ) -> StoreResult<bool> {
        let key = entry(index.generation, false, tuple, owner).0;
        Ok(txn.read_untracked(Partition::Idx, &key)?.is_some())
    }

    fn latest_entry(
        &self,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
    ) -> StoreResult<Option<LatestEntry>> {
        let key = entry(index.generation, index.unique, tuple, owner).0;
        Ok(self
            .engine
            .get_versioned(Partition::Idx, &key)?
            .map(|(bytes, version)| LatestEntry {
                holder: if index.unique {
                    decode_holder(&bytes).ok()
                } else {
                    None
                },
                version,
            }))
    }

    fn repair_entry_txn(
        &self,
        txn: &mut Transaction,
        index: &IndexDefinition,
        tuple: &[u8],
        owner: EntryOwner,
        present: bool,
        version: Option<u64>,
    ) -> StoreResult<()> {
        let (key, value) = entry(index.generation, index.unique, tuple, owner);
        // The condition before the write it guards: a direct-mode engine
        // decides a condition when it is stated.
        txn.expect_version(Partition::Idx, &key, version)?;
        if present {
            txn.put(Partition::Idx, &key, &value)?;
        } else {
            txn.delete(Partition::Idx, &key)?;
        }
        Ok(())
    }
}

impl LocalIndexStore<'_> {
    /// Stage the removal of the repair records of the build of `generation`,
    /// which go with its build record. The data changes they record stay.
    fn delete_repairs_txn(
        &self,
        txn: &mut Transaction,
        generation: GenerationId,
    ) -> StoreResult<()> {
        for guard in self.engine.prefix_scan(
            Partition::Schema,
            &DuplicateRepairRecord::prefix_of(generation),
        )? {
            let (key, _) = guard.into_inner()?;
            txn.delete(Partition::Schema, &key)?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;
