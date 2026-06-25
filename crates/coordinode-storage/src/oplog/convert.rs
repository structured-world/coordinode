//! Conversion from logical [`Mutation`]s into [`OplogOp`]s.
//!
//! The embedded (oracle-backed, no-Raft) write path journals each proposal as
//! an [`OplogEntry`](crate::oplog::entry::OplogEntry) carrying granular ops —
//! the same `Insert`/`Delete`/`Merge`/`RemoveRange` shapes the cluster Raft log
//! stores alongside its `RaftEntry` wrapper. Routing back to a partition on the
//! replay / repair side uses [`partition_wire_tag`]'s inverse, so the tag
//! written here must match that mapping exactly.

use coordinode_core::txn::proposal::Mutation;

use crate::engine::partition::Partition;
use crate::oplog::entry::OplogOp;
use crate::placement::partition_wire_tag;

/// Convert one logical mutation into its oplog op, tagging it with the
/// partition's wire discriminant.
#[must_use]
pub fn mutation_to_op(m: &Mutation) -> OplogOp {
    match m {
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
    }
}

/// Convert a batch of mutations into oplog ops, preserving order.
#[must_use]
pub fn mutations_to_ops(mutations: &[Mutation]) -> Vec<OplogOp> {
    mutations.iter().map(mutation_to_op).collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests;
