//! Conversion from logical [`Mutation`]s into [`OplogOp`]s.
//!
//! The embedded (oracle-backed, no-Raft) write path journals each proposal as
//! an [`OplogEntry`](crate::oplog::entry::OplogEntry) carrying granular ops —
//! the same `Insert`/`Delete`/`Merge`/`RemoveRange` shapes the cluster Raft log
//! stores alongside its `RaftEntry` wrapper. Routing back to a partition on the
//! replay / repair side uses [`partition_wire_tag`]'s inverse, so the tag
//! written here must match that mapping exactly.

use std::borrow::Cow;

use coordinode_core::index::derive::resolve_unit;
use coordinode_core::txn::frame::decode_unit;
use coordinode_core::txn::operation::LogPosition;
use coordinode_core::txn::proposal::{Mutation, PartitionId};

use crate::engine::MAX_DERIVED_EFFECTS;
use crate::engine::partition::Partition;
use crate::error::{StorageError, StorageResult};
use crate::oplog::entry::OplogOp;
use crate::placement::{partition_from_wire_tag, partition_wire_tag};

/// Convert one logical mutation into its oplog op, tagging it with the
/// partition's wire discriminant.
///
/// # Errors
///
/// A metadata command: the journal records the effects a command was
/// decided to have, never the command, so replay does not decide it again.
pub fn mutation_to_op(m: &Mutation) -> StorageResult<OplogOp> {
    Ok(match m {
        Mutation::Put {
            partition,
            key,
            value,
        } => OplogOp::Insert {
            partition: partition_wire_tag(Partition::from(*partition)),
            key: key.clone(),
            value: value.clone(),
        },
        Mutation::Delete { partition, key } => OplogOp::Delete {
            partition: partition_wire_tag(Partition::from(*partition)),
            key: key.clone(),
        },
        Mutation::Merge {
            partition,
            key,
            operand,
        } => OplogOp::Merge {
            partition: partition_wire_tag(Partition::from(*partition)),
            key: key.clone(),
            operand: operand.clone(),
        },
        Mutation::RemoveRange {
            partition,
            start,
            end,
        } => OplogOp::RemoveRange {
            partition: partition_wire_tag(Partition::from(*partition)),
            start: start.clone(),
            end: end.clone(),
        },
        Mutation::Command(_) | Mutation::Operation(_) => {
            return Err(StorageError::InvalidConfig(
                "a metadata or operation command reached the journal undecided".into(),
            ));
        }
        Mutation::Derive(work) => OplogOp::Derive {
            work: rmp_serde::to_vec(work)
                .map_err(|e| StorageError::Serialization(format!("derived index work: {e}")))?,
        },
    })
}

/// Convert a batch of mutations into oplog ops, preserving order.
///
/// # Errors
///
/// As [`mutation_to_op`].
pub fn mutations_to_ops(mutations: &[Mutation]) -> StorageResult<Vec<OplogOp>> {
    mutations.iter().map(mutation_to_op).collect()
}

/// The mutation a journalled op records, for the ops a unit's DERIVED work
/// can name: data ops and the work itself.
fn op_to_mutation(op: &OplogOp) -> StorageResult<Mutation> {
    let partition = |tag: u8| -> StorageResult<PartitionId> {
        partition_from_wire_tag(tag)
            .and_then(Partition::proposal_id)
            .ok_or_else(|| {
                StorageError::Serialization(format!("partition tag {tag} is not a data partition"))
            })
    };
    Ok(match op {
        OplogOp::Insert {
            partition: tag,
            key,
            value,
        } => Mutation::Put {
            partition: partition(*tag)?,
            key: key.clone(),
            value: value.clone(),
        },
        OplogOp::Delete {
            partition: tag,
            key,
        } => Mutation::Delete {
            partition: partition(*tag)?,
            key: key.clone(),
        },
        OplogOp::Merge {
            partition: tag,
            key,
            operand,
        } => Mutation::Merge {
            partition: partition(*tag)?,
            key: key.clone(),
            operand: operand.clone(),
        },
        OplogOp::RemoveRange {
            partition: tag,
            start,
            end,
        } => Mutation::RemoveRange {
            partition: partition(*tag)?,
            start: start.clone(),
            end: end.clone(),
        },
        OplogOp::Derive { work } => Mutation::Derive(
            rmp_serde::from_slice(work)
                .map_err(|e| StorageError::Serialization(format!("derived index work: {e}")))?,
        ),
        OplogOp::Noop
        | OplogOp::RaftEntry { .. }
        | OplogOp::RaftTruncation { .. }
        | OplogOp::ColumnarInsert { .. }
        | OplogOp::Unit { .. } => {
            return Err(StorageError::Serialization(
                "derived index work shares its entry with a non-data op".into(),
            ));
        }
    })
}

/// Whether the unit at a log position was refused by its operation binding,
/// as the member that applied it recorded.
pub type UnitDecisions<'a> = &'a (dyn Fn(LogPosition) -> StorageResult<bool> + Send + Sync);

/// The ops a journalled entry records, each unit frame expanded into the
/// operations it encodes, in order. DERIVED work stays work; a metadata or
/// operation command a Raft log entry carries is left out, its effects being
/// decided at application and never a data change. An entry without a frame
/// is returned as it is.
///
/// # Errors
///
/// A frame that does not decode, or a unit bound to a client operation: only
/// the member's decision says whether it took effect, see
/// [`expand_units_decided`].
pub fn expand_units(ops: &[OplogOp]) -> StorageResult<Cow<'_, [OplogOp]>> {
    expand_units_decided(ops, 0, None)
}

/// [`expand_units`] for the entry at log `index`, leaving out every unit
/// `decisions` says its binding refused. Without `decisions`, a bound unit
/// is an error rather than a guess.
///
/// # Errors
///
/// A frame that does not decode, a decision that cannot be read, or a bound
/// unit without `decisions`.
pub fn expand_units_decided<'o>(
    ops: &'o [OplogOp],
    index: u64,
    decisions: Option<UnitDecisions<'_>>,
) -> StorageResult<Cow<'o, [OplogOp]>> {
    if !ops.iter().any(|op| matches!(op, OplogOp::Unit { .. })) {
        return Ok(Cow::Borrowed(ops));
    }
    let mut out = Vec::with_capacity(ops.len());
    let mut sub = 0u32;
    for op in ops {
        let OplogOp::Unit { frame } = op else {
            out.push(op.clone());
            continue;
        };
        let at = LogPosition { index, sub };
        sub = sub.checked_add(1).ok_or_else(|| {
            StorageError::Serialization(format!("entry {index} carries over 2^32 units"))
        })?;
        let mutations = decode_unit(frame)?;
        if mutations
            .iter()
            .any(|m| matches!(m, Mutation::Operation(_)))
        {
            let Some(decisions) = decisions else {
                return Err(StorageError::InvalidConfig(format!(
                    "unit {index}/{} is bound to a client operation, and no decision source \
                     says whether it took effect",
                    at.sub
                )));
            };
            if decisions(at)? {
                continue;
            }
        }
        for mutation in mutations {
            if !matches!(mutation, Mutation::Command(_) | Mutation::Operation(_)) {
                out.push(mutation_to_op(&mutation)?);
            }
        }
    }
    Ok(Cow::Owned(out))
}

/// The ops a journalled entry applies: its unit frames expanded and its
/// DERIVED work replaced by the index entry ops it derives from the entry
/// itself. An entry without either is returned as it is.
///
/// # Errors
///
/// An op or frame that does not decode, or work that cannot be derived as
/// sealed.
pub fn resolve_entry_ops(ops: &[OplogOp]) -> StorageResult<Cow<'_, [OplogOp]>> {
    let expanded = expand_units(ops)?;
    if !expanded
        .iter()
        .any(|op| matches!(op, OplogOp::Derive { .. }))
    {
        return Ok(expanded);
    }
    let unit = expanded
        .iter()
        .map(op_to_mutation)
        .collect::<StorageResult<Vec<_>>>()?;
    let resolved = resolve_unit(&unit, MAX_DERIVED_EFFECTS)?;
    Ok(Cow::Owned(mutations_to_ops(&resolved)?))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
