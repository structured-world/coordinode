//! Metadata the ordered application decides: field bindings and NodeId leases.
//!
//! A [`MetadataCommand`] is decided against the state the engine holds when it
//! applies, under the engine's decision lock, and turned into plain Schema
//! puts before anything else sees the proposal. Every record it writes is
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
use coordinode_core::txn::proposal::{MetadataCommand, Mutation, PartitionId};
use lsm_tree::Guard as _;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};

/// The effects `command` has against the state `engine` holds now. An empty
/// result is a refusal: the command's whole batch publishes nothing.
pub(crate) fn decide(
    engine: &StorageEngine,
    command: &MetadataCommand,
) -> StorageResult<Vec<Mutation>> {
    match command {
        MetadataCommand::RegisterFields { names } => decide_registration(engine, names),
        MetadataCommand::AdoptFields { bindings } => decide_adoption(engine, bindings),
        MetadataCommand::GrantNodeLease {
            base,
            ceiling,
            token,
        } => decide_lease(engine, *base, *ceiling, token),
    }
}

fn schema_put(key: Vec<u8>, value: Vec<u8>) -> Mutation {
    Mutation::Put {
        partition: PartitionId::Schema,
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
    engine: &StorageEngine,
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
    Ok(vec![schema_put(node_lease_key(ceiling), token.to_vec())])
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
    let by_id = read_id_records(engine, FIELD_ID_KEY_PREFIX)?;
    let mut by_name = 0usize;
    for guard in engine.prefix_scan(Partition::Schema, FIELD_NAME_KEY_PREFIX)? {
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
    engine: &StorageEngine,
    prefix: &[u8],
) -> StorageResult<std::collections::BTreeMap<u32, String>> {
    let mut by_id = std::collections::BTreeMap::new();
    for guard in engine.prefix_scan(Partition::Schema, prefix)? {
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

/// The ceiling of the last granted NodeId lease; 0 before the first grant.
///
/// # Errors
///
/// The last lease record's key does not name a ceiling.
pub fn node_lease_ceiling(engine: &StorageEngine) -> StorageResult<u64> {
    let Some(guard) = engine
        .prefix_scan_rev(Partition::Schema, NODE_LEASE_KEY_PREFIX)?
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
    engine: &StorageEngine,
    ceiling: u64,
) -> StorageResult<Option<[u8; NODE_LEASE_TOKEN_LEN]>> {
    Ok(engine
        .get(Partition::Schema, &node_lease_key(ceiling))?
        .and_then(|v| <[u8; NODE_LEASE_TOKEN_LEN]>::try_from(v.as_ref()).ok()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests;
