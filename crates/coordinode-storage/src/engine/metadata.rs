//! Metadata the ordered application decides: field bindings, implicit label
//! identities and NodeId leases.
//!
//! A [`MetadataCommand`] is decided against the state the engine holds when it
//! applies, under the engine's decision lock, and turned into plain puts
//! before anything else sees the proposal: field bindings and label
//! identities in a namespace's Schema tree, NodeId leases and the group's
//! version and identity among the deployment's records. Every record it writes is
//! write-once (one key per binding direction, one per lease), so which commit
//! timestamp a record carries never changes what a reader finds; only the
//! order of application does, and that order is the log's.
//!
//! The decision is the same on every member and on every replay: it reads
//! nothing but the records earlier commands wrote, and a record it would
//! write again already holds the same value.

use std::collections::HashSet;

use coordinode_core::graph::intern::{
    DictionaryError, FIELD_ID_KEY_PREFIX, FIELD_NAME_KEY_PREFIX, FieldInterner, decode_field_id,
    decode_field_id_key, encode_field_id, field_id_key, field_name_key, validate_registration,
};
use coordinode_core::graph::node::{
    NODE_ID_MAX_SEQUENCE, NODE_LEASE_KEY_PREFIX, NODE_LEASE_TOKEN_LEN, decode_node_lease_key,
    node_lease_key,
};
use coordinode_core::group::{
    DEPLOYMENT_KEY, DeploymentId, ENROLLED_DEPLOYMENT_KEY, retired_identity_key,
};
use coordinode_core::namespace::{
    NAMESPACE_ALLOCATOR_KEY, NAMESPACE_RECORD_PREFIX, NAMESPACE_TOKEN_LEN, NamespaceId,
    NamespaceRecord, NamespaceState, decode_namespace_record_key, namespace_name_key,
    namespace_record_key, validate_namespace_name,
};
use coordinode_core::schema::definition::{
    encode_label_binding_history_prefix, encode_label_current_revision_key,
};
use coordinode_core::schema::label_id::{
    BindingState, FIRST_LABEL_ID, LABEL_ID_KEY_PREFIX, LabelBinding, LabelId, decode_label_id_key,
    label_id_key,
};
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, PartitionId};
use coordinode_core::version::{
    GROUP_PAIR_KEY_PREFIX, RecordedPair, VersionPair, decode_group_pair_key, decode_pair,
    encode_pair, group_pair_key,
};

use crate::engine::core::{Deployment, StorageEngine};
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// The effects `command`, committing at `commit_ts`, has against the state
/// `deployment` holds now: the records of the namespace `scope` is bound to
/// for a command of a namespace, the deployment's own otherwise. An empty
/// result is a refusal: the command's whole batch publishes nothing.
///
/// # Errors
///
/// A read failure; [`StorageError::UnknownNamespace`] for a namespace's
/// command without `scope`.
pub(crate) fn decide(
    deployment: &Deployment,
    scope: Option<&StorageEngine>,
    command: &MetadataCommand,
    commit_ts: u64,
    actor: Option<&coordinode_core::access::PrincipalId>,
) -> StorageResult<Vec<Mutation>> {
    let namespace = || {
        scope.ok_or_else(|| {
            StorageError::InvalidConfig(format!(
                "the metadata command {command:?} names no namespace"
            ))
        })
    };
    if let Some(effects) = crate::engine::access::decide(deployment, command, commit_ts)? {
        return Ok(effects);
    }
    match command {
        MetadataCommand::RegisterFields { names } => decide_registration(namespace()?, names),
        MetadataCommand::AdoptFields { bindings } => decide_adoption(namespace()?, bindings),
        MetadataCommand::BindLabels { names } => {
            label_binding_effects(namespace()?, names, commit_ts)
        }
        MetadataCommand::GrantNodeLease {
            base,
            ceiling,
            token,
        } => decide_lease(deployment, *base, *ceiling, token),
        MetadataCommand::RecordGroupPair { pair } => decide_group_pair(deployment, *pair),
        MetadataCommand::RecordDeployment { id } => {
            Ok(if recorded_deployment(deployment)?.is_some() {
                Vec::new()
            } else {
                vec![deployment_put(
                    DEPLOYMENT_KEY.to_vec(),
                    id.as_bytes().to_vec(),
                )]
            })
        }
        MetadataCommand::ConditionalRegistryWrite { expect, put } => {
            decide_registry_write(deployment, expect, put)
        }
        MetadataCommand::CreateNamespace {
            name,
            token,
            storage,
        } => {
            #[cfg_attr(not(feature = "tenancy"), allow(unused_mut))]
            let mut effects = decide_namespace_creation(deployment, name, token, *storage)?;
            // The creator becomes admin of what it created, in the same
            // change: no state has the namespace without its admin.
            #[cfg(feature = "tenancy")]
            if let (Some(creator), Some(id)) = (actor, created_id(&effects)) {
                effects.push(crate::engine::access::creator_grant(id, creator, commit_ts));
            }
            #[cfg(not(feature = "tenancy"))]
            let _ = actor;
            Ok(effects)
        }
        MetadataCommand::DropNamespace { name } => decide_namespace_drop(deployment, name),
        MetadataCommand::SetNamespaceStorage { name, change } => {
            decide_namespace_storage(deployment, name, *change)
        }
        // Decided above.
        MetadataCommand::GrantAccess { .. }
        | MetadataCommand::RevokeAccess { .. }
        | MetadataCommand::RevokeGrant { .. }
        | MetadataCommand::TransferAccess { .. }
        | MetadataCommand::SetDeploymentRights { .. }
        | MetadataCommand::CreateRole { .. }
        | MetadataCommand::DropRole { .. }
        | MetadataCommand::SetRoleEligible { .. }
        | MetadataCommand::GrantRole { .. }
        | MetadataCommand::RevokeRole { .. }
        | MetadataCommand::TransferRole { .. } => Err(StorageError::InvalidConfig(format!(
            "the grant-catalog command {command:?} was not decided"
        ))),
    }
}

/// The namespace a creation's effects create, from its record's key.
#[cfg(feature = "tenancy")]
fn created_id(effects: &[Mutation]) -> Option<NamespaceId> {
    effects.iter().find_map(|m| match m {
        Mutation::Put { key, .. } => decode_namespace_record_key(key),
        _ => None,
    })
}

/// The record of the live namespace `name` with the settings `change`
/// names replaced. None when no live namespace has the name.
fn decide_namespace_storage(
    engine: &Deployment,
    name: &str,
    change: coordinode_core::namespace::NamespaceStorage,
) -> StorageResult<Vec<Mutation>> {
    let Some(id) = namespace_by_name(engine, name)? else {
        return Ok(Vec::new());
    };
    let Some(record) = namespace_record(engine, id)? else {
        return Err(StorageError::Serialization(format!(
            "namespace {name:?} is named in the catalog without a record of {id}"
        )));
    };
    let changed = NamespaceRecord {
        storage: record.storage.with(change),
        ..record
    };
    Ok(vec![deployment_put(
        namespace_record_key(id),
        changed.encode(),
    )])
}

/// The records creating namespace `name` under the next id: its record,
/// its name's entry and the allocator moved past the id. None when the
/// name is not one a namespace may have, a live namespace holds it, or the
/// ids are exhausted.
fn decide_namespace_creation(
    engine: &Deployment,
    name: &str,
    token: &[u8; NAMESPACE_TOKEN_LEN],
    storage: coordinode_core::namespace::NamespaceStorage,
) -> StorageResult<Vec<Mutation>> {
    if let Err(e) = validate_namespace_name(name) {
        tracing::warn!(name, error = %e, "namespace creation refused");
        return Ok(Vec::new());
    }
    if namespace_by_name(engine, name)?.is_some() {
        return Ok(Vec::new());
    }
    let next = namespace_allocator(engine)?;
    let Some(after) = next.checked_add(1) else {
        tracing::warn!(name, "namespace creation refused: the ids are exhausted");
        return Ok(Vec::new());
    };
    let id = NamespaceId::from_raw(next);
    let record = NamespaceRecord {
        name: name.to_owned(),
        state: NamespaceState::Live,
        token: *token,
        storage,
    };
    Ok(vec![
        deployment_put(namespace_record_key(id), record.encode()),
        deployment_put(namespace_name_key(name), id.as_raw().to_be_bytes().to_vec()),
        deployment_put(
            NAMESPACE_ALLOCATOR_KEY.to_vec(),
            after.to_be_bytes().to_vec(),
        ),
    ])
}

/// The records beginning to drop the live namespace `name`: its record
/// marked dropping and its name freed. None when no live namespace has the
/// name.
fn decide_namespace_drop(engine: &Deployment, name: &str) -> StorageResult<Vec<Mutation>> {
    let Some(id) = namespace_by_name(engine, name)? else {
        return Ok(Vec::new());
    };
    let Some(record) = namespace_record(engine, id)? else {
        return Err(StorageError::Serialization(format!(
            "namespace {name:?} is named in the catalog without a record of {id}"
        )));
    };
    let dropping = NamespaceRecord {
        state: NamespaceState::Dropping,
        ..record
    };
    Ok(vec![
        deployment_put(namespace_record_key(id), dropping.encode()),
        Mutation::Delete {
            partition: PartitionId::Deployment,
            key: namespace_name_key(name),
        },
    ])
}

/// A token no other namespace creation draws, to tell this creation's
/// record from another one of the same name: as unique as a
/// [`deployment_candidate`], and as little secret.
#[must_use]
pub fn namespace_token() -> [u8; NAMESPACE_TOKEN_LEN] {
    deployment_candidate().0
}

/// The id the deployment's catalog hands the next namespace: 1 before the
/// first.
///
/// # Errors
///
/// The allocator record does not decode.
pub fn namespace_allocator(engine: &Deployment) -> StorageResult<u64> {
    match engine.system_get(Partition::Deployment, NAMESPACE_ALLOCATOR_KEY)? {
        None => Ok(1),
        Some(bytes) => <[u8; 8]>::try_from(bytes.as_ref())
            .map(u64::from_be_bytes)
            .map_err(|_| {
                StorageError::Serialization("the namespace allocator does not decode".into())
            }),
    }
}

/// The live namespace called `name`, if one is.
///
/// # Errors
///
/// The name's entry does not decode.
pub fn namespace_by_name(engine: &Deployment, name: &str) -> StorageResult<Option<NamespaceId>> {
    match engine.system_get(Partition::Deployment, &namespace_name_key(name))? {
        None => Ok(None),
        Some(bytes) => <[u8; 8]>::try_from(bytes.as_ref())
            .map(|raw| Some(NamespaceId::from_raw(u64::from_be_bytes(raw))))
            .map_err(|_| {
                StorageError::Serialization(format!(
                    "the catalog entry of namespace {name:?} does not decode"
                ))
            }),
    }
}

/// The catalog's record of namespace `id`, if it has one.
///
/// # Errors
///
/// The record does not decode.
pub fn namespace_record(
    engine: &Deployment,
    id: NamespaceId,
) -> StorageResult<Option<NamespaceRecord>> {
    engine
        .system_get(Partition::Deployment, &namespace_record_key(id))?
        .map(|bytes| {
            NamespaceRecord::decode(&bytes).ok_or_else(|| {
                StorageError::Serialization(format!("the catalog record of {id} does not decode"))
            })
        })
        .transpose()
}

/// Every namespace the catalog records, live or dropping, in id order.
///
/// # Errors
///
/// A record or its key does not decode.
pub fn namespaces(engine: &Deployment) -> StorageResult<Vec<(NamespaceId, NamespaceRecord)>> {
    let mut out = Vec::new();
    for guard in engine.system_prefix_scan(Partition::Deployment, NAMESPACE_RECORD_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        let malformed = || {
            StorageError::Serialization("a namespace record of the catalog does not decode".into())
        };
        let id = decode_namespace_record_key(&key).ok_or_else(malformed)?;
        out.push((id, NamespaceRecord::decode(&value).ok_or_else(malformed)?));
    }
    Ok(out)
}

/// The writes of a conditional registry write: `put`, when every record in
/// `expect` holds exactly the bytes given against the registry as it is now,
/// else none. Read at the store's latest applied state, which during an
/// ordered apply is every earlier entry's registry writes and none of a
/// later one's, on every member and on a replay alike.
fn decide_registry_write(
    engine: &Deployment,
    expect: &[(Vec<u8>, Option<Vec<u8>>)],
    put: &[(Vec<u8>, Vec<u8>)],
) -> StorageResult<Vec<Mutation>> {
    for (key, expected) in expect {
        let held = engine.system_get(Partition::Registry, key)?;
        if held.as_deref() != expected.as_deref() {
            return Ok(Vec::new());
        }
    }
    Ok(put
        .iter()
        .map(|(key, value)| Mutation::Put {
            partition: PartitionId::Registry,
            key: key.clone(),
            value: value.clone(),
        })
        .collect())
}

/// The deployment's identity, `None` before the founding leader recorded it.
///
/// # Errors
///
/// The record does not decode.
pub fn recorded_deployment(engine: &Deployment) -> StorageResult<Option<DeploymentId>> {
    match engine.system_get(Partition::Deployment, DEPLOYMENT_KEY)? {
        None => Ok(None),
        Some(bytes) => DeploymentId::from_slice(&bytes).map(Some).ok_or_else(|| {
            StorageError::Serialization("the deployment record does not decode".into())
        }),
    }
}

/// The deployment this store belongs to: its record, or, on a member that
/// joined a group and has not received the record (a joiner installs the
/// group's state from a snapshot, which leaves each member's own records
/// out), the deployment it was enrolled in. `None` before either exists.
/// It is the domain of the store's operations and of its history alike.
///
/// # Errors
///
/// A record does not decode; [`StorageError::DeploymentConflict`] when the
/// record and the enrollment both exist and disagree.
pub fn bound_deployment(engine: &Deployment) -> StorageResult<Option<DeploymentId>> {
    binding_of(
        recorded_deployment(engine)?,
        engine.system_get(Partition::Deployment, ENROLLED_DEPLOYMENT_KEY)?,
    )
}

/// [`bound_deployment`] as it stood at `snapshot`, a seqno the caller pinned.
///
/// # Errors
///
/// As [`bound_deployment`]; the snapshot is no longer retained.
pub fn bound_deployment_at(
    engine: &Deployment,
    snapshot: lsm_tree::SeqNo,
) -> StorageResult<Option<DeploymentId>> {
    let recorded =
        match engine.system_snapshot_get(&snapshot, Partition::Deployment, DEPLOYMENT_KEY)? {
            None => None,
            Some(bytes) => Some(DeploymentId::from_slice(&bytes).ok_or_else(|| {
                StorageError::Serialization("the deployment record does not decode".into())
            })?),
        };
    binding_of(
        recorded,
        engine.system_snapshot_get(&snapshot, Partition::Deployment, ENROLLED_DEPLOYMENT_KEY)?,
    )
}

/// The binding a deployment record and an enrollment record give together.
fn binding_of(
    recorded: Option<DeploymentId>,
    enrolled: Option<impl AsRef<[u8]>>,
) -> StorageResult<Option<DeploymentId>> {
    let enrolled = match enrolled {
        None => None,
        Some(bytes) => Some(DeploymentId::from_slice(bytes.as_ref()).ok_or_else(|| {
            StorageError::Serialization("the enrollment record does not decode".into())
        })?),
    };
    match (recorded, enrolled) {
        (Some(recorded), Some(enrolled)) if recorded != enrolled => {
            Err(StorageError::DeploymentConflict { recorded, enrolled })
        }
        (recorded, enrolled) => Ok(recorded.or(enrolled)),
    }
}

/// A deployment identity no other database draws, for a founder to propose:
/// 128 bits from two hashers keyed by the process's random seed, mixed with
/// the clock. It tells domains apart and grants nothing, so it needs
/// uniqueness, not secrecy. Only the candidate the ordered decision records
/// first is ever the domain; every founder reads that one back.
pub fn deployment_candidate() -> DeploymentId {
    use std::hash::{BuildHasher, Hasher};
    // no-std: a caller-provided entropy source.
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let mut id = [0u8; 16];
    for half in id.as_chunks_mut::<8>().0 {
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u128(now);
        *half = hasher.finish().to_be_bytes();
    }
    DeploymentId(id)
}

/// The deployment to found this store's domain with: the one it was
/// enrolled in when it joined a group whose record has not reached it yet,
/// else a fresh candidate. An enrolled member never mints a replacement, and
/// neither does a store that already holds history without a binding: that
/// history belongs to a domain whose binding was lost, not to a new one.
///
/// # Errors
///
/// As [`bound_deployment`]; [`StorageError::HistoryDomainMissing`] for an
/// unbound store holding history.
pub fn founding_candidate(engine: &Deployment) -> StorageResult<DeploymentId> {
    founding_candidate_from(|| holds_history(engine), || bound_deployment(engine))
}

/// [`founding_candidate`] over the store's two reads, made in the order this
/// function chooses: history first. A binding lands before the history it
/// binds, so history seen by the first read has its binding visible to the
/// second; read the other way round, another founder landing both between
/// the reads would look like history whose binding was lost.
fn founding_candidate_from(
    holds_history: impl FnOnce() -> StorageResult<bool>,
    bound: impl FnOnce() -> StorageResult<Option<DeploymentId>>,
) -> StorageResult<DeploymentId> {
    let holds = holds_history()?;
    if let Some(bound) = bound()? {
        return Ok(bound);
    }
    if holds {
        return Err(StorageError::HistoryDomainMissing);
    }
    Ok(deployment_candidate())
}

/// Whether the store holds any retained assertion of a node, in any of its
/// namespaces. The kept label bindings do not count: the ordered first-use
/// binding keeps them before a store has founded anything, and they answer
/// no read without node history.
///
/// # Errors
///
/// A read failure.
pub fn holds_history(engine: &Deployment) -> StorageResult<bool> {
    for namespace in engine.coordinator().sets().namespace_ids() {
        if engine
            .engine(namespace)?
            .prefix_scan(Partition::NodeHistory, &[])?
            .next()
            .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the node identity `identity` (canonically encoded) was retired.
///
/// # Errors
///
/// The record cannot be read.
pub fn identity_retired(engine: &Deployment, identity: &[u8]) -> StorageResult<bool> {
    Ok(engine
        .system_get(Partition::Deployment, &retired_identity_key(identity))?
        .is_some())
}

/// A new record only when the pair changes: a leader re-elected at the pair
/// the group already runs records nothing.
fn decide_group_pair(engine: &Deployment, pair: VersionPair) -> StorageResult<Vec<Mutation>> {
    let last = recorded_group_pair(engine)?;
    if last.is_some_and(|r| r.pair == pair) {
        return Ok(Vec::new());
    }
    let seq = last.map_or(1, |r| r.seq + 1);
    Ok(vec![deployment_put(group_pair_key(seq), encode_pair(pair))])
}

/// Every pair the group recorded, oldest first.
///
/// # Errors
///
/// A record's key or value does not decode.
pub fn group_pair_records(engine: &Deployment) -> StorageResult<Vec<RecordedPair>> {
    let malformed = || StorageError::Serialization("a group pair record does not decode".into());
    let mut records = Vec::new();
    for guard in engine.system_prefix_scan(Partition::Deployment, GROUP_PAIR_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        records.push(RecordedPair {
            seq: decode_group_pair_key(&key).ok_or_else(malformed)?,
            pair: decode_pair(&value).ok_or_else(malformed)?,
        });
    }
    Ok(records)
}

/// The last pair the group recorded, `None` before its first record.
///
/// # Errors
///
/// The last record's key or value does not decode.
pub fn recorded_group_pair(engine: &Deployment) -> StorageResult<Option<RecordedPair>> {
    let Some(guard) = engine
        .system_prefix_scan_rev(Partition::Deployment, GROUP_PAIR_KEY_PREFIX)?
        .next()
    else {
        return Ok(None);
    };
    let (key, value) = guard.into_inner()?;
    let malformed = || StorageError::Serialization("a group pair record does not decode".into());
    let seq = decode_group_pair_key(&key).ok_or_else(malformed)?;
    let pair = decode_pair(&value).ok_or_else(malformed)?;
    Ok(Some(RecordedPair { pair, seq }))
}

fn schema_put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Schema,
        key,
        value,
    }
}

fn deployment_put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Deployment,
        key,
        value,
    }
}

/// The two records of one binding.
fn binding_puts(name: &str, id: u32, out: &mut Vec<Mutation>) {
    out.push(schema_put(
        field_name_key(name),
        encode_field_id(id).to_vec(),
    ));
    out.push(schema_put(field_id_key(id), name.as_bytes().to_vec()));
}

fn decide_registration(engine: &StorageEngine, names: &[String]) -> StorageResult<Vec<Mutation>> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    if let Err(e) = validate_registration(&refs) {
        tracing::warn!(error = %e, "field registration refused");
        return Ok(Vec::new());
    }
    let mut seen = HashSet::with_capacity(refs.len());
    let mut new_names = Vec::new();
    for name in refs {
        if seen.insert(name) && bound_id(engine, name)?.is_none() {
            new_names.push(name);
        }
    }
    if new_names.is_empty() {
        return Ok(Vec::new());
    }
    let frontier = field_frontier(engine)?;
    // The whole batch is checked against the id space before any binding of
    // it is published: a partial batch would leave its proposer without ids
    // for names it was told were registered.
    let room = u32::MAX - frontier;
    if new_names.len() > room as usize {
        tracing::warn!(
            asked = new_names.len(),
            room,
            "field registration refused: the id space is exhausted"
        );
        return Ok(Vec::new());
    }
    let mut effects = Vec::with_capacity(new_names.len() * 2);
    for (offset, name) in new_names.into_iter().enumerate() {
        // `offset < room`, so the id stays within u32.
        binding_puts(name, frontier + 1 + offset as u32, &mut effects);
    }
    Ok(effects)
}

fn decide_adoption(
    engine: &StorageEngine,
    bindings: &[(String, u32)],
) -> StorageResult<Vec<Mutation>> {
    // The batch itself must be one consistent set of bindings.
    if let Err(e) = FieldInterner::from_bindings(bindings.iter().cloned()) {
        tracing::warn!(error = %e, "field adoption refused");
        return Ok(Vec::new());
    }
    let mut effects = Vec::new();
    for (name, id) in bindings {
        let by_name = bound_id(engine, name)?;
        let by_id = bound_name(engine, *id)?;
        match (by_name, by_id.as_deref()) {
            (Some(existing), Some(existing_name)) if existing == *id && existing_name == name => {}
            (None, None) => binding_puts(name, *id, &mut effects),
            _ => {
                tracing::warn!(
                    name = %name,
                    id,
                    ?by_name,
                    ?by_id,
                    "field adoption refused: it contradicts a published binding"
                );
                return Ok(Vec::new());
            }
        }
    }
    Ok(effects)
}

fn decide_lease(
    engine: &Deployment,
    base: u64,
    ceiling: u64,
    token: &[u8; NODE_LEASE_TOKEN_LEN],
) -> StorageResult<Vec<Mutation>> {
    if ceiling <= base || ceiling > NODE_ID_MAX_SEQUENCE {
        tracing::warn!(base, ceiling, "NodeId lease refused: not a range");
        return Ok(Vec::new());
    }
    if node_lease_ceiling(engine)? != base {
        // Another grant moved the ceiling since this one read it.
        return Ok(Vec::new());
    }
    Ok(vec![deployment_put(
        node_lease_key(ceiling),
        token.to_vec(),
    )])
}

/// The id `name` is bound to, if any.
///
/// # Errors
///
/// The record exists but does not hold an id.
pub fn bound_id(engine: &StorageEngine, name: &str) -> StorageResult<Option<u32>> {
    match engine.get(Partition::Schema, &field_name_key(name))? {
        None => Ok(None),
        Some(value) => decode_field_id(&value).map(Some).ok_or_else(|| {
            DictionaryError::Malformed(format!("the binding record of {name:?} holds no id")).into()
        }),
    }
}

/// The name `id` is bound to, if any.
///
/// # Errors
///
/// The record exists but its name is not UTF-8.
pub fn bound_name(engine: &StorageEngine, id: u32) -> StorageResult<Option<String>> {
    match engine.get(Partition::Schema, &field_id_key(id))? {
        None => Ok(None),
        Some(value) => String::from_utf8(value.to_vec()).map(Some).map_err(|_| {
            DictionaryError::Malformed(format!("the binding record of id {id} is not UTF-8")).into()
        }),
    }
}

/// The largest bound id; 0 when nothing is bound.
///
/// # Errors
///
/// The last id record's key does not name an id.
pub fn field_frontier(engine: &StorageEngine) -> StorageResult<u32> {
    let Some(guard) = engine
        .prefix_scan_rev(Partition::Schema, FIELD_ID_KEY_PREFIX)?
        .next()
    else {
        return Ok(0);
    };
    let key = guard.key()?;
    decode_field_id_key(&key).ok_or_else(|| {
        StorageError::from(DictionaryError::Malformed(
            "an id record key does not name an id".into(),
        ))
    })
}

/// Every binding the engine holds, checked record against record.
///
/// # Errors
///
/// A record that does not decode, an id record without its name record (or
/// the reverse), or two records that disagree.
pub fn load_field_dictionary(engine: &StorageEngine) -> StorageResult<FieldInterner> {
    // No binding is written while this reads: a binding applied between the
    // two scans would show its name record without its id record.
    let _exclusive = engine.metadata_exclusive();
    dictionary_from(|prefix| engine.prefix_scan(Partition::Schema, prefix))
}

/// [`load_field_dictionary`] as it stood at `snapshot`, a seqno the caller
/// pinned: both scans read the same cut, so no lock is taken.
///
/// # Errors
///
/// As [`load_field_dictionary`]; the snapshot is no longer retained.
pub fn load_field_dictionary_at(
    engine: &StorageEngine,
    snapshot: lsm_tree::SeqNo,
) -> StorageResult<FieldInterner> {
    dictionary_from(|prefix| engine.snapshot_prefix_iter(&snapshot, Partition::Schema, prefix))
}

/// The dictionary the Schema records `scan` yields hold, checked record
/// against record.
fn dictionary_from(
    scan: impl Fn(&[u8]) -> StorageResult<crate::engine::StorageIter>,
) -> StorageResult<FieldInterner> {
    let by_id = read_id_records(scan(FIELD_ID_KEY_PREFIX)?)?;
    let mut by_name = 0usize;
    for guard in scan(FIELD_NAME_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        let name = std::str::from_utf8(&key[FIELD_NAME_KEY_PREFIX.len()..])
            .map_err(|_| DictionaryError::Malformed("a name record key is not UTF-8".into()))?;
        let id = decode_field_id(&value).ok_or_else(|| {
            DictionaryError::Malformed(format!("the binding record of {name:?} holds no id"))
        })?;
        match by_id.get(&id) {
            Some(recorded) if recorded == name => by_name += 1,
            Some(recorded) => {
                return Err(DictionaryError::IdConflict {
                    id,
                    first: recorded.clone(),
                    second: name.to_owned(),
                }
                .into());
            }
            None => {
                return Err(DictionaryError::Malformed(format!(
                    "{name:?} is bound to {id}, which has no id record"
                ))
                .into());
            }
        }
    }
    if by_name != by_id.len() {
        return Err(DictionaryError::Malformed(format!(
            "{} id records but {by_name} name records",
            by_id.len()
        ))
        .into());
    }
    Ok(FieldInterner::from_bindings(
        by_id.into_iter().map(|(id, name)| (name, id)),
    )?)
}

/// The bindings with ids above `frontier`, each checked against its name
/// record.
///
/// # Errors
///
/// As [`load_field_dictionary`], for the records read.
pub fn field_bindings_above(
    engine: &StorageEngine,
    frontier: u32,
) -> StorageResult<Vec<(String, u32)>> {
    let Some(from) = frontier.checked_add(1) else {
        return Ok(Vec::new());
    };
    // As in `load_field_dictionary`: no binding lands half read, whatever
    // order the tree takes a batch's two records in.
    let _exclusive = engine.metadata_exclusive();
    let mut bindings = Vec::new();
    for guard in engine.range_scan(
        Partition::Schema,
        &field_id_key(from),
        &field_id_key(u32::MAX),
    )? {
        let (key, value) = guard.into_inner()?;
        let id = decode_field_id_key(&key).ok_or_else(|| {
            DictionaryError::Malformed("an id record key does not name an id".into())
        })?;
        let name = String::from_utf8(value.to_vec()).map_err(|_| {
            DictionaryError::Malformed(format!("the binding record of id {id} is not UTF-8"))
        })?;
        bindings.push((name, id));
    }
    let keys: Vec<Vec<u8>> = bindings.iter().map(|(n, _)| field_name_key(n)).collect();
    let refs: Vec<&[u8]> = keys.iter().map(Vec::as_slice).collect();
    let recorded = engine.multi_get(Partition::Schema, &refs)?;
    for ((name, id), value) in bindings.iter().zip(recorded) {
        if value.as_deref().and_then(decode_field_id) != Some(*id) {
            return Err(DictionaryError::Malformed(format!(
                "id {id} names {name:?}, but {name:?} does not name {id}"
            ))
            .into());
        }
    }
    Ok(bindings)
}

fn read_id_records(
    records: crate::engine::StorageIter,
) -> StorageResult<std::collections::BTreeMap<u32, String>> {
    let mut by_id = std::collections::BTreeMap::new();
    for guard in records {
        let (key, value) = guard.into_inner()?;
        let id = decode_field_id_key(&key).ok_or_else(|| {
            DictionaryError::Malformed("an id record key does not name an id".into())
        })?;
        let name = String::from_utf8(value.to_vec()).map_err(|_| {
            DictionaryError::Malformed(format!("the binding record of id {id} is not UTF-8"))
        })?;
        by_id.insert(id, name);
    }
    Ok(by_id)
}

/// The newest label identity issued, `None` before the first. Read from the
/// identity records, which are written once, so it never moves back whatever
/// order their commit timestamps land in.
///
/// # Errors
///
/// The newest identity record's key does not name an identity.
pub fn label_frontier(engine: &StorageEngine) -> StorageResult<Option<LabelId>> {
    let Some(guard) = engine
        .prefix_scan_rev(Partition::Schema, LABEL_ID_KEY_PREFIX)?
        .next()
    else {
        return Ok(None);
    };
    let key = guard.key()?;
    decode_label_id_key(&key).map(Some).ok_or_else(|| {
        StorageError::Serialization("a label identity record key names no identity".into())
    })
}

/// The identity after `frontier`, the newest issued (`None` when none was).
///
/// # Errors
///
/// The identity space is exhausted.
pub fn next_label_id(frontier: Option<LabelId>) -> StorageResult<LabelId> {
    match frontier {
        None => Ok(FIRST_LABEL_ID),
        Some(id) => id.next().ok_or_else(|| {
            StorageError::Serialization("the label identity space is exhausted".into())
        }),
    }
}

/// The records binding each of `names` that has no live binding to a new
/// identity, in the order given, committing at `commit_ts`: its write-once
/// identity record and an implicit name binding. A name already bound to a
/// live identity is left as it is, so concurrent first uses converge on the
/// identity applied first. An invalid batch publishes nothing, and so does a
/// batch rebinding a retired name whose retirement carries a later version
/// than `commit_ts`: written below it, the binding would read as older than
/// the retirement it replaces. Its proposer retries at a fresh timestamp.
///
/// # Errors
///
/// A binding or identity record does not decode, or the identity space is
/// exhausted.
pub fn label_binding_effects(
    engine: &StorageEngine,
    names: &[String],
    commit_ts: u64,
) -> StorageResult<Vec<Mutation>> {
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    if let Err(e) = validate_registration(&refs) {
        tracing::warn!(error = %e, "label binding refused");
        return Ok(Vec::new());
    }
    let mut seen = HashSet::with_capacity(refs.len());
    let mut frontier = label_frontier(engine)?;
    let mut effects = Vec::new();
    for name in refs {
        if !seen.insert(name) {
            continue;
        }
        let key = encode_label_current_revision_key(name);
        if let Some((bytes, version)) = engine.get_versioned(Partition::Schema, &key)? {
            let binding = LabelBinding::decode(&bytes).ok_or_else(|| {
                StorageError::Serialization(format!(
                    "the binding of label {name:?} does not decode"
                ))
            })?;
            if binding.live_id().is_some() {
                continue;
            }
            if version > commit_ts {
                tracing::debug!(name, version, commit_ts, "label binding deferred");
                return Ok(Vec::new());
            }
        }
        let id = next_label_id(frontier)?;
        frontier = Some(id);
        effects.push(schema_put(
            label_id_key(id).to_vec(),
            name.as_bytes().to_vec(),
        ));
        let binding = LabelBinding {
            id,
            revision: 0,
            state: BindingState::Implicit,
        };
        // The name's binding history keeps the binding under this commit,
        // which every member applies at the same timestamp.
        let mut history = encode_label_binding_history_prefix(name);
        history.extend_from_slice(&commit_ts.to_be_bytes());
        effects.push(schema_put(history, binding.encode().to_vec()));
        effects.push(schema_put(key, binding.encode().to_vec()));
    }
    Ok(effects)
}

/// The ceiling of the last granted NodeId lease; 0 before the first grant.
///
/// # Errors
///
/// The last lease record's key does not name a ceiling.
pub fn node_lease_ceiling(engine: &Deployment) -> StorageResult<u64> {
    let Some(guard) = engine
        .system_prefix_scan_rev(Partition::Deployment, NODE_LEASE_KEY_PREFIX)?
        .next()
    else {
        return Ok(0);
    };
    let key = guard.key()?;
    decode_node_lease_key(&key).ok_or_else(|| {
        StorageError::Serialization("a NodeId lease record key names no ceiling".into())
    })
}

/// The token of the grant whose ceiling is `ceiling`, if one was granted.
///
/// # Errors
///
/// A storage read fails.
pub fn node_lease_holder(
    engine: &Deployment,
    ceiling: u64,
) -> StorageResult<Option<[u8; NODE_LEASE_TOKEN_LEN]>> {
    Ok(engine
        .system_get(Partition::Deployment, &node_lease_key(ceiling))?
        .and_then(|v| <[u8; NODE_LEASE_TOKEN_LEN]>::try_from(v.as_ref()).ok()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
