//! Deciding a claim against authoritative state and the attempt's own writes.
//!
//! The registry answers whether two attempts can coexist. This answers the
//! other half: whether the condition an attempt validated against is still
//! true where it counts. A coherent earlier snapshot is not that proof, so
//! the evaluation reads the authoritative state at the moment it is asked and
//! applies the attempt's staged mutations on top, because a node and its
//! mandatory edge created together are one valid post-state and the
//! intermediate absence of the edge is not a violation.
//!
//! The two cardinality measures are counted from different places and neither
//! is inferred from the other. Distinct neighbours come from the adjacency
//! posting, which is a set of neighbours by construction. Edge instances come
//! from the edge-property entries, one per discriminator value, because two
//! instances to one neighbour are two identities and one neighbour. Asking
//! the posting for an instance count would answer the wrong question with a
//! plausible number.

use coordinode_core::graph::edge::{PostingList, encode_adj_key_forward, encode_adj_key_reverse};
use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::invariant::{
    Adjacency, CardinalityMeasure, Claim, ClaimPredicate, ClaimScope, Direction,
};

use lsm_tree::Guard;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::engine::transaction::AdjOp;
use crate::error::StorageResult;

/// The attempt's own staged adjacency mutations, in the order they were
/// staged, so the evaluation sees the post-state rather than the state before
/// the attempt ran.
pub type StagedAdj<'a> = &'a [(Vec<u8>, AdjOp)];

/// The attempt's staged point writes, as the commit will apply them: `Some`
/// for a value written, `None` for a row deleted.
///
/// The evaluation needs them for the same reason it needs the adjacency
/// operands. A node and the edge attached to it, created by one statement,
/// are one valid post-state; reading only committed state would find the node
/// absent and call the edge dangling, refusing the most ordinary write there
/// is.
pub type StagedPoints<'a> = &'a std::collections::HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>;

/// Whether a claim still holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The condition holds over the whole post-state.
    Holds,
    /// The condition does not hold; the attempt must not be admitted.
    Broken,
    /// This evaluator cannot decide the claim, so nothing may be concluded
    /// from it. A caller treats it as a refusal rather than as a pass: an
    /// undecidable condition is not a satisfied one.
    Undecidable,
}

/// Decide `claim` against the engine's authoritative state plus the attempt's
/// staged adjacency writes.
pub fn evaluate(
    engine: &StorageEngine,
    claim: &Claim,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
    read_ts: u64,
) -> StorageResult<Verdict> {
    match (&claim.scope, &claim.predicate) {
        (
            ClaimScope::Incident {
                node,
                edge_type,
                direction,
            },
            ClaimPredicate::CardinalityBound {
                measure,
                at_most,
                at_least,
            },
        ) => {
            let count = match measure {
                CardinalityMeasure::DistinctNeighbours => {
                    distinct_neighbours(engine, *node, edge_type, *direction, staged)?
                }
                CardinalityMeasure::EdgeInstances => {
                    match edge_instances(engine, *node, edge_type, *direction, staged_points)? {
                        Some(n) => n,
                        // Instances are counted from the edge-property
                        // entries of each pair, which an incoming scope
                        // cannot enumerate without the neighbour set it is
                        // being asked about.
                        None => return Ok(Verdict::Undecidable),
                    }
                }
            };
            let within_upper = at_most.is_none_or(|limit| count <= limit as usize);
            let within_lower = at_least.is_none_or(|limit| count >= limit as usize);
            Ok(if within_upper && within_lower {
                Verdict::Holds
            } else {
                Verdict::Broken
            })
        }

        (
            ClaimScope::Pair {
                source,
                target,
                edge_type,
            },
            ClaimPredicate::PairAdjacency { observed },
        ) => pair_observation_survived(engine, *source, *target, edge_type, *observed, read_ts),

        (ClaimScope::Node(node), ClaimPredicate::EndpointAlive) => {
            endpoint_alive(engine, *node, staged_points, read_ts)
        }

        // The destruction is the attempt's own intent, not a condition on the
        // state it reads: there is nothing to re-check, and what the claim
        // buys is stated entirely in the registry, where it excludes every
        // reference right held on the same node. Answering `Undecidable` here
        // would refuse every deletion.
        (ClaimScope::Node(_), ClaimPredicate::EndpointDestroyed) => Ok(Verdict::Holds),

        // An enumeration is proved by what it enumerated over: the attempt
        // read the whole incident set at its view, and the claim is that
        // nothing joined or left it since. A member added after the scan is
        // exactly the one the scan could not have found.
        (
            ClaimScope::Incident {
                node,
                edge_type,
                direction,
            },
            ClaimPredicate::IncidentSetComplete,
        ) => incident_set_unchanged(engine, *node, edge_type, *direction, read_ts),

        // The condition was evaluated against one version of the record, so
        // any later write to it is a renewal that invalidates the condition,
        // whatever it wrote.
        (ClaimScope::Record(key), ClaimPredicate::CleanupCondition { observed_version }) => Ok(
            if engine.written_since_snapshot(Partition::Node, key, *observed_version)? {
                Verdict::Broken
            } else {
                Verdict::Holds
            },
        ),

        // A predicate evaluated under one schema generation says nothing
        // about the graph under another, and the generation the attempt
        // stamped on the claim is what it read when it evaluated.
        (ClaimScope::SchemaElement(_), ClaimPredicate::SchemaApplicability) => {
            Ok(if claim.schema_revision == engine.schema_generation() {
                Verdict::Holds
            } else {
                Verdict::Broken
            })
        }

        // An overlap this evaluator has no evidence for stays undecided, and
        // a caller treats that as a refusal rather than a pass.
        _ => Ok(Verdict::Undecidable),
    }
}

/// Whether the incident set is still the one the attempt enumerated.
///
/// A scan that decides what to delete, redirect or move is only as good as
/// its completeness, and completeness is not preserved by reading rows: a
/// member that arrives after the scan passed is invisible to it and to
/// first-committer-wins alike, because the two write different keys.
///
/// Any change to the set breaks it, in either direction. A member that left
/// matters as much as one that joined: an enumeration that carries a row
/// which no longer exists is describing a different operation than the one
/// being committed.
fn incident_set_unchanged(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    read_ts: u64,
) -> StorageResult<Verdict> {
    let key = match direction {
        Direction::Outgoing => encode_adj_key_forward(edge_type, node),
        Direction::Incoming => encode_adj_key_reverse(edge_type, node),
    };
    let members = |bytes: Option<Vec<u8>>| -> Vec<u64> {
        bytes
            .and_then(|b| PostingList::from_bytes(&b).ok())
            .map(|p| p.as_slice().to_vec())
            .unwrap_or_default()
    };

    let at_view = members(
        engine
            .snapshot_get(&read_ts, Partition::Adj, &key)?
            .map(|b| b.to_vec()),
    );
    let now = members(engine.get(Partition::Adj, &key)?.map(|b| b.to_vec()));

    Ok(if at_view == now {
        Verdict::Holds
    } else {
        Verdict::Broken
    })
}

/// Whether the observation an attempt built its result on still stands.
///
/// The question is not what the pair looks like now: the attempt is usually
/// the reason it looks different, and asking that of a MERGE would refuse
/// every MERGE for having done what the observation told it to do. The
/// question is whether somebody else moved the pair under it, which is the
/// comparison between the state its view showed and the state now.
///
/// Two attempts that observed the same thing and act the same way are left
/// alone here, as they are in the registry: both add the same member to a set
/// and the result is one edge either way. What this catches is the erase of
/// the last qualifying row racing the insertion that was built on its absence,
/// or the reverse, where the two write different keys and first-committer-wins
/// sees nothing.
fn pair_observation_survived(
    engine: &StorageEngine,
    source: NodeId,
    target: NodeId,
    edge_type: &str,
    observed: Adjacency,
    read_ts: u64,
) -> StorageResult<Verdict> {
    let key = encode_adj_key_forward(edge_type, source);

    let at_view = match engine.snapshot_get(&read_ts, Partition::Adj, &key)? {
        Some(bytes) => PostingList::from_bytes(&bytes)
            .map(|p| p.as_slice().contains(&target.as_raw()))
            .unwrap_or(false),
        None => false,
    };
    let now = match engine.get(Partition::Adj, &key)? {
        Some(bytes) => PostingList::from_bytes(&bytes)
            .map(|p| p.as_slice().contains(&target.as_raw()))
            .unwrap_or(false),
        None => false,
    };

    // The view has to agree with what the attempt says it saw; a claim whose
    // own observation was already stale when it was made is no evidence.
    let seen = match observed {
        Adjacency::Present => true,
        Adjacency::Absent => false,
    };
    Ok(if at_view == seen && now == seen {
        Verdict::Holds
    } else {
        Verdict::Broken
    })
}

/// Whether the identity an attempt is attaching to survived until its commit.
///
/// The claim is about destruction, not about existence. An attempt that
/// references a node needs that node not to be reclaimed under it; whether the
/// node was ever materialised is a different question, answered where the
/// statement resolves its endpoints, and answering it here would turn every
/// writer of an edge into an enforcer of referential integrity and refuse the
/// bulk paths (restore, import) that write adjacency before node rows.
///
/// So the row's absence alone decides nothing. What decides is absence
/// together with a write to that row after the attempt's snapshot: the node
/// was there when the attempt built its result and is gone now, which is
/// exactly the destruction the claim guards against.
fn endpoint_alive(
    engine: &StorageEngine,
    node: NodeId,
    staged_points: StagedPoints<'_>,
    read_ts: u64,
) -> StorageResult<Verdict> {
    let shard = engine.node_shard();
    let key = coordinode_core::graph::node::encode_node_key(shard, node);

    // The attempt's own writes decide first. A node it is creating is alive
    // for its own edge, and one it is deleting in the same breath it attaches
    // to is a result it cannot have both halves of.
    //
    // The emptiness check is not a micro-optimisation dressed as a guard: the
    // lookup key has to be built by cloning, and an attempt that staged no
    // point writes at all (every plain edge attachment) would pay that
    // allocation to look inside an empty map.
    if !staged_points.is_empty() {
        if let Some(staged) = staged_points.get(&(Partition::Node, key.clone())) {
            return Ok(if staged.is_some() {
                Verdict::Holds
            } else {
                Verdict::Broken
            });
        }
    }

    // The ordinary answer is decided without reading the node at all. Nobody
    // has touched the row since this attempt's view, so whatever it was then
    // it still is, and nothing was taken away. This is the case nearly every
    // edge write is in, and it costs one key-only probe: reading the record
    // to prove a node was not deleted copies a value the answer never uses.
    if !engine.written_since_snapshot(Partition::Node, &key, read_ts)? {
        return Ok(Verdict::Holds);
    }

    // The row moved under the attempt, so what it is now has to be looked at.
    // An update is not a destruction; only an absence is.
    if engine.get(Partition::Node, &key)?.is_some() {
        return Ok(Verdict::Holds);
    }

    // A label in temporal mode stores its nodes only as versions under the
    // per-version key, so the plain row is absent for a node that very much
    // exists.
    let prefix = coordinode_core::graph::node::temporal_node_id_prefix(shard, node);
    if staged_points.iter().any(|((part, k), value)| {
        *part == Partition::Node && k.starts_with(&prefix) && value.is_some()
    }) {
        return Ok(Verdict::Holds);
    }
    for guard in engine.prefix_scan(Partition::Node, &prefix)? {
        if guard.into_inner().is_ok() {
            return Ok(Verdict::Holds);
        }
    }

    Ok(Verdict::Broken)
}

/// The neighbour set of one incident scope, with the attempt's staged writes
/// applied in the order they were staged.
fn adjacency(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    staged: StagedAdj<'_>,
) -> StorageResult<Vec<u64>> {
    let key = match direction {
        Direction::Outgoing => encode_adj_key_forward(edge_type, node),
        Direction::Incoming => encode_adj_key_reverse(edge_type, node),
    };

    let mut plist = match engine.get(Partition::Adj, &key)? {
        Some(bytes) => PostingList::from_bytes(&bytes).unwrap_or_else(|_| PostingList::new()),
        None => PostingList::new(),
    };

    for (staged_key, op) in staged {
        if staged_key.as_slice() != key.as_slice() {
            continue;
        }
        match op {
            AdjOp::Add(uid) => {
                plist.insert(*uid);
            }
            AdjOp::Remove(uid) => {
                plist.remove(*uid);
            }
        }
    }

    Ok(plist.as_slice().to_vec())
}

/// Distinct neighbours of the scope: the adjacency posting is a set of
/// neighbours, so its size is the measure directly.
fn distinct_neighbours(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    staged: StagedAdj<'_>,
) -> StorageResult<usize> {
    Ok(adjacency(engine, node, edge_type, direction, staged)?.len())
}

/// Logical edge identities of the scope, counted from the edge-property
/// entries of each neighbouring pair: a discriminated edge type stores one
/// entry per discriminator value, so the entries are the identities.
///
/// Returns `None` for an incoming scope, whose pairs are keyed by their
/// source and so are not reachable from the target's side by prefix.
///
/// The attempt's own entries count, and its own deletions do not: a bound is
/// decided against the post-state the attempt would leave, and counting only
/// what is committed would admit the instance that breaks it and refuse the
/// deletion that repairs it.
fn edge_instances(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Option<usize>> {
    if direction == Direction::Incoming {
        return Ok(None);
    }

    let mut prefix = Vec::new();
    prefix.extend_from_slice(b"edgeprop:");
    prefix.extend_from_slice(edge_type.as_bytes());
    prefix.push(b':');
    prefix.extend_from_slice(&node.as_raw().to_be_bytes());

    let mut count: usize = 0;
    for guard in engine.prefix_scan(Partition::EdgeProp, &prefix)? {
        let Ok(key) = guard.key() else {
            continue;
        };
        // A row this attempt tombstones is already gone as far as the bound
        // is concerned; one it rewrites is still the same identity.
        match staged_points.get(&(Partition::EdgeProp, key.to_vec())) {
            Some(None) => {}
            _ => count += 1,
        }
    }

    // Entries the attempt adds that the committed scan could not see.
    for ((part, key), value) in staged_points {
        if *part != Partition::EdgeProp || value.is_none() || !key.starts_with(&prefix) {
            continue;
        }
        if engine.get(Partition::EdgeProp, key)?.is_none() {
            count += 1;
        }
    }

    Ok(Some(count))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
