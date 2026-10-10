//! Maintained exact counts of declared relationship cardinality constraints.
//!
//! A constraint's count changes only with the logical edge effects a commit
//! admits, so the commit is where it is kept: whatever writer staged the
//! effect, the commit sees the pair before and after it and stages the
//! difference with the writes, in the same unit. A writer cannot forget to
//! count, and the count cannot land without the effect or the other way
//! round.
//!
//! The difference is taken against the pair's state at the moment the commit
//! is admitted, not at the attempt's snapshot, and two attempts on one pair
//! are kept apart by a [`ClaimPredicate::PairCounted`] claim on it: both would
//! otherwise see the pair empty, both would count it joining, and one
//! neighbour would be counted twice.
//!
//! A count is evidence only once its descriptor is covered: the
//! [`rebuild`] that reconciles it with the stored edges writes the coverage
//! marker in the same unit as its corrections. Without the marker the count
//! is not read and the bound is decided by enumeration, so a missing count
//! never stands in for zero.

use coordinode_core::graph::cardinality::{
    CardinalityDescriptor, CardinalityMeasure, CardinalityProfile, IncidentCount, IncidentInstance,
    InstanceKey, PairInstances, encode_cardinality_profile_key,
};
use coordinode_core::graph::edge::{
    PostingList, encode_adj_key_forward, encode_adj_key_reverse, split_discriminated_edgeprop_key,
    temporal_edgeprop_pair_prefix,
};
use coordinode_core::graph::node::NodeId;
use coordinode_core::txn::invariant::{Claim, ClaimPredicate, ClaimScope, CountTrend, Direction};
use rustc_hash::FxHashMap;

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::engine::transaction::{AdjOp, Transaction};
use crate::error::{StorageError, StorageResult};

/// The attempt's staged adjacency operands, in the order they were staged.
pub type StagedAdj<'a> = &'a [(Vec<u8>, AdjOp)];

/// The attempt's staged point writes: `Some` written, `None` deleted.
pub type StagedPoints<'a> = &'a std::collections::HashMap<(Partition, Vec<u8>), Option<Vec<u8>>>;

/// One ordered pair of a counted type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CountedPair {
    edge_type: String,
    source: NodeId,
    target: NodeId,
}

/// What one commit counts: the profiles of the counted types it writes, the
/// pairs of those types it changes, and the postings it drops whole with
/// the members they held when it planned.
#[derive(Debug, Default)]
pub(crate) struct CommitCounts {
    profiles: FxHashMap<String, CardinalityProfile>,
    /// Each pair with whether the staged writes may add to it and may take
    /// from it.
    pairs: std::collections::BTreeMap<CountedPair, Movement>,
    dropped: Vec<(Vec<u8>, Vec<u64>)>,
}

/// Whether staged writes may add to something counted and may take from it.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct Movement {
    grows: bool,
    shrinks: bool,
}

impl Movement {
    fn trend(self) -> CountTrend {
        CountTrend::of(self.grows, self.shrinks)
    }
}

impl CommitCounts {
    fn note(
        &mut self,
        edge_type: &str,
        source: NodeId,
        target: NodeId,
        grows: bool,
        shrinks: bool,
    ) {
        let movement = self
            .pairs
            .entry(CountedPair {
                edge_type: edge_type.to_string(),
                source,
                target,
            })
            .or_default();
        movement.grows |= grows;
        movement.shrinks |= shrinks;
    }

    /// Whether the commit changes nothing that is counted.
    pub(crate) fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// The claims that keep another count of the same pairs out while this
    /// one is decided, and keep the postings it drops from changing under it.
    pub(crate) fn claims(&self, schema_generation: u64) -> Vec<Claim> {
        let mut claims: Vec<Claim> = self
            .pairs
            .iter()
            .map(|(pair, movement)| {
                Claim::new(
                    ClaimScope::Pair {
                        source: pair.source,
                        target: pair.target,
                        edge_type: pair.edge_type.clone(),
                    },
                    ClaimPredicate::PairCounted {
                        trend: movement.trend(),
                    },
                    schema_generation,
                )
            })
            .collect();
        for (key, _) in &self.dropped {
            if let Some((edge_type, direction, node)) = adj_parts(key) {
                claims.push(Claim::new(
                    ClaimScope::Incident {
                        node,
                        edge_type: edge_type.to_string(),
                        direction,
                    },
                    ClaimPredicate::IncidentSetComplete,
                    schema_generation,
                ));
            }
        }
        claims
    }
}

/// The type, side and node of an adjacency key `adj:<type>:out:<node>` or
/// `adj:<type>:in:<node>`. Edge type names carry no ':', so the type ends at
/// the first one; nothing is allocated, since every staged operand of every
/// commit is read through here.
fn adj_parts(key: &[u8]) -> Option<(&str, Direction, NodeId)> {
    let rest = key.strip_prefix(b"adj:")?;
    let colon = rest.iter().position(|b| *b == b':')?;
    let edge_type = core::str::from_utf8(&rest[..colon]).ok()?;
    let tail = &rest[colon..];
    let (direction, id) = match tail.strip_prefix(b":out:") {
        Some(id) => (Direction::Outgoing, id),
        None => (Direction::Incoming, tail.strip_prefix(b":in:")?),
    };
    let raw: [u8; 8] = id.try_into().ok()?;
    Some((
        edge_type,
        direction,
        NodeId::from_raw(u64::from_be_bytes(raw)),
    ))
}

/// The ordered pair an adjacency member of `key` stands for.
fn pair_of(direction: Direction, node: NodeId, member: u64) -> (NodeId, NodeId) {
    match direction {
        Direction::Outgoing => (node, NodeId::from_raw(member)),
        Direction::Incoming => (NodeId::from_raw(member), node),
    }
}

/// Edge type `edge_type`'s profile as committed now; `None` when it
/// declares no constraint.
pub(crate) fn load_profile(
    engine: &StorageEngine,
    edge_type: &str,
) -> StorageResult<Option<CardinalityProfile>> {
    let key = encode_cardinality_profile_key(edge_type);
    let Some(bytes) = engine.get(Partition::Schema, &key)? else {
        return Ok(None);
    };
    // A profile that does not decode would leave its counts unkept while
    // the constraint still stands; it is reported, never skipped.
    CardinalityProfile::from_msgpack(&bytes)
        .map(Some)
        .map_err(|e| StorageError::UnreadableCatalog {
            kind: "edge type cardinality profile",
            key: crate::error::printable_key(&key),
            detail: e.to_string(),
        })
}

fn decode_posting(key: &[u8], bytes: &[u8]) -> StorageResult<PostingList> {
    PostingList::from_bytes(bytes).map_err(|e| {
        StorageError::Serialization(format!(
            "adjacency posting {}: {e}",
            crate::error::printable_key(key)
        ))
    })
}

/// The adjacency posting at `key` as committed now.
fn committed_posting(engine: &StorageEngine, key: &[u8]) -> StorageResult<PostingList> {
    match engine.get(Partition::Adj, key)? {
        Some(bytes) => decode_posting(key, &bytes),
        None => Ok(PostingList::new()),
    }
}

/// The adjacency posting at `key` as the attempt leaves it: the committed
/// posting, replaced or dropped by a staged point write, then the staged
/// operands in order. The commit applies them in that order, point writes
/// before operands, so this is the state it produces.
pub(crate) fn post_posting(
    engine: &StorageEngine,
    key: &[u8],
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> StorageResult<PostingList> {
    let mut plist = match staged_points.get(&(Partition::Adj, key.to_vec())) {
        Some(Some(bytes)) => decode_posting(key, bytes)?,
        Some(None) => PostingList::new(),
        None => committed_posting(engine, key)?,
    };
    for (staged_key, op) in staged {
        if staged_key.as_slice() != key {
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
    Ok(plist)
}

/// Plan what the commit counts: the pairs of counted types its staged writes
/// touch. One schema read per distinct edge type touched; a commit writing
/// only unconstrained types finds no profile and plans nothing.
pub(crate) fn plan(
    engine: &StorageEngine,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> StorageResult<CommitCounts> {
    // A commit usually writes one or two types, so a short list beats a set.
    let mut types: Vec<&str> = Vec::new();
    for (key, _) in staged {
        if let Some((edge_type, _, _)) = adj_parts(key) {
            if !types.contains(&edge_type) {
                types.push(edge_type);
            }
        }
    }
    for (part, key) in staged_points.keys() {
        let edge_type = match part {
            Partition::Adj => adj_parts(key).map(|(t, _, _)| t),
            Partition::EdgeProp => split_discriminated_edgeprop_key(key).map(|(t, _, _, _)| t),
            _ => None,
        };
        if let Some(edge_type) = edge_type {
            if !types.contains(&edge_type) {
                types.push(edge_type);
            }
        }
    }
    let mut counts = CommitCounts::default();
    for edge_type in types {
        if let Some(profile) = load_profile(engine, edge_type)? {
            counts.profiles.insert(edge_type.to_string(), profile);
        }
    }
    if counts.profiles.is_empty() {
        return Ok(counts);
    }

    for (key, op) in staged {
        let Some((edge_type, direction, node)) = adj_parts(key) else {
            continue;
        };
        if !counts.profiles.contains_key(edge_type) {
            continue;
        }
        let (member, grows) = match op {
            AdjOp::Add(uid) => (*uid, true),
            AdjOp::Remove(uid) => (*uid, false),
        };
        let (source, target) = pair_of(direction, node, member);
        counts.note(edge_type, source, target, grows, !grows);
    }
    for ((part, key), value) in staged_points {
        match part {
            Partition::Adj => {
                let Some((edge_type, direction, node)) = adj_parts(key) else {
                    continue;
                };
                if !counts.profiles.contains_key(edge_type) {
                    continue;
                }
                // A posting written whole changes every pair it held and
                // every pair it now holds.
                let held = committed_posting(engine, key)?.as_slice().to_vec();
                let written = match value {
                    Some(bytes) => decode_posting(key, bytes)?.as_slice().to_vec(),
                    None => Vec::new(),
                };
                // A drop only takes; a posting written whole may do either.
                let grows = value.is_some();
                for member in held.iter().chain(written.iter()) {
                    let (source, target) = pair_of(direction, node, *member);
                    counts.note(edge_type, source, target, grows, true);
                }
                counts.dropped.push((key.clone(), held));
            }
            Partition::EdgeProp => {
                let Some((edge_type, source, target, _)) = split_discriminated_edgeprop_key(key)
                else {
                    continue;
                };
                if counts
                    .profiles
                    .get(edge_type)
                    .is_some_and(|profile| profile.discriminated)
                {
                    // An instance written may be new or an upsert; one
                    // deleted may be the pair's last.
                    counts.note(edge_type, source, target, value.is_some(), value.is_none());
                }
            }
            _ => {}
        }
    }
    Ok(counts)
}

/// Which way the staged writes can move the counts of the scope of `node`'s
/// `edge_type` edges in `direction`. Read from the kind of each write that
/// can reach the scope: an adjacency addition or a written instance can only
/// add, a removal or a deleted instance can only take, a posting written
/// whole may do either. Whether an instance write is new or an upsert does
/// not matter: neither takes anything away.
pub(crate) fn scope_trend(
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> CountTrend {
    let mut movement = Movement::default();
    for (key, op) in staged {
        let Some((t, side, owner)) = adj_parts(key) else {
            continue;
        };
        let (member, grows) = match op {
            AdjOp::Add(uid) => (*uid, true),
            AdjOp::Remove(uid) => (*uid, false),
        };
        let reaches = t == edge_type
            && if side == direction {
                owner == node
            } else {
                member == node.as_raw()
            };
        if reaches {
            movement.grows |= grows;
            movement.shrinks |= !grows;
        }
    }
    for ((part, key), value) in staged_points {
        match part {
            Partition::Adj => {
                let Some((t, side, owner)) = adj_parts(key) else {
                    continue;
                };
                // The opposite side's posting may hold the node anywhere in
                // it, so a whole write there is taken as reaching the scope.
                if t == edge_type && (side != direction || owner == node) {
                    movement.grows |= value.is_some();
                    movement.shrinks = true;
                }
            }
            Partition::EdgeProp => {
                let Some((t, source, target, _)) = split_discriminated_edgeprop_key(key) else {
                    continue;
                };
                let ours = match direction {
                    Direction::Outgoing => source == node,
                    Direction::Incoming => target == node,
                };
                if t == edge_type && ours {
                    movement.grows |= value.is_some();
                    movement.shrinks |= value.is_none();
                }
            }
            _ => {}
        }
    }
    movement.trend()
}

/// Whether `pair` is adjacent as committed now and as the attempt leaves it.
///
/// The two adjacency postings hold the same membership, written in one unit
/// by every edge writer, so either answers. The one read is the side away
/// from the node the constraint counts: a constrained node is the one that
/// gathers edges, and its posting is the long one, a merge chain a read
/// would have to fold on every commit that touches it. With both directions
/// declared the target's reverse posting is read. The committed posting is
/// read once and the staged writes applied to it, as the commit applies
/// them.
fn pair_membership(
    engine: &StorageEngine,
    profile: &CardinalityProfile,
    pair: &CountedPair,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> StorageResult<(bool, bool)> {
    let counts_target_only = profile
        .descriptors
        .iter()
        .all(|d| d.direction == Direction::Incoming);
    let (key, member) = if counts_target_only {
        (
            encode_adj_key_forward(&pair.edge_type, pair.source),
            pair.target,
        )
    } else {
        (
            encode_adj_key_reverse(&pair.edge_type, pair.target),
            pair.source,
        )
    };
    let committed = committed_posting(engine, &key)?;
    let before = committed.contains(member.as_raw());
    let mut after = match staged_points.get(&(Partition::Adj, key.clone())) {
        Some(Some(bytes)) => decode_posting(&key, bytes)?.contains(member.as_raw()),
        Some(None) => false,
        None => before,
    };
    for (staged_key, op) in staged {
        if staged_key.as_slice() != key.as_slice() {
            continue;
        }
        match op {
            AdjOp::Add(uid) if *uid == member.as_raw() => after = true,
            AdjOp::Remove(uid) if *uid == member.as_raw() => after = false,
            _ => {}
        }
    }
    Ok((before, after))
}

/// The instance identities of a discriminated pair under `prefix`, as
/// committed now and as the attempt leaves them.
fn pair_entries(
    engine: &StorageEngine,
    prefix: &[u8],
    staged_points: StagedPoints<'_>,
) -> StorageResult<(Vec<InstanceKey>, Vec<InstanceKey>)> {
    let mut committed = Vec::new();
    for guard in engine.prefix_scan(Partition::EdgeProp, prefix)? {
        // A row that cannot be read is not a row that is absent.
        let key = guard.key()?;
        committed.push(key[prefix.len()..].to_vec());
    }
    let mut after: Vec<InstanceKey> = committed
        .iter()
        .filter(|suffix| {
            let mut full = prefix.to_vec();
            full.extend_from_slice(suffix);
            !matches!(staged_points.get(&(Partition::EdgeProp, full)), Some(None))
        })
        .cloned()
        .collect();
    for ((part, key), value) in staged_points {
        if *part == Partition::EdgeProp && value.is_some() && key.starts_with(prefix) {
            after.push(key[prefix.len()..].to_vec());
        }
    }
    Ok((committed, after))
}

/// The pair's instances on one side, from its adjacency on that side and,
/// for a discriminated type, its entries. An adjacent discriminated pair
/// with no entry, or entries with no adjacency, is evidence missing: no
/// count is derived from it.
fn instances(
    adjacent: bool,
    discriminated: bool,
    entries: &[InstanceKey],
) -> Result<PairInstances, ()> {
    if !discriminated {
        return Ok(if adjacent {
            PairInstances::from_keys([InstanceKey::new()])
        } else {
            PairInstances::default()
        });
    }
    if adjacent == entries.is_empty() {
        return Err(());
    }
    Ok(PairInstances::from_keys(entries.iter().cloned()))
}

/// The count changes the commit stages, keyed by counter, against the pairs'
/// state now. Called once the commit's claims are reserved, so no other
/// count of these pairs can land in between; `Err` carries why a pair cannot
/// be counted, and the commit is refused for it.
pub(crate) fn deltas(
    engine: &StorageEngine,
    counts: &CommitCounts,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Result<FxHashMap<Vec<u8>, i64>, String>> {
    for (key, held) in &counts.dropped {
        if committed_posting(engine, key)?.as_slice() != held.as_slice() {
            return Ok(Err(format!(
                "the adjacency {} changed while this commit was dropping it",
                crate::error::printable_key(key)
            )));
        }
    }

    let mut out: FxHashMap<Vec<u8>, i64> = FxHashMap::default();
    for pair in counts.pairs.keys() {
        let Some(profile) = counts.profiles.get(&pair.edge_type) else {
            continue;
        };
        let entries = if profile.discriminated {
            let prefix = temporal_edgeprop_pair_prefix(&pair.edge_type, pair.source, pair.target);
            Some(pair_entries(engine, &prefix, staged_points)?)
        } else {
            None
        };
        let (before, after) = pair_membership(engine, profile, pair, staged, staged_points)?;
        let (committed, staged_after) = match &entries {
            Some((c, a)) => (c.as_slice(), a.as_slice()),
            None => (&[][..], &[][..]),
        };
        let (Ok(was), Ok(will)) = (
            instances(before, profile.discriminated, committed),
            instances(after, profile.discriminated, staged_after),
        ) else {
            return Ok(Err(format!(
                "the {} pair {} -> {} has adjacency and instances that disagree, so \
                 its count cannot be derived",
                pair.edge_type,
                pair.source.as_raw(),
                pair.target.as_raw()
            )));
        };
        let transition = was.transition_to(&will);
        for descriptor in &profile.descriptors {
            let delta = transition.delta(descriptor.measure);
            if delta == 0 {
                continue;
            }
            let node = match descriptor.direction {
                Direction::Outgoing => pair.source,
                Direction::Incoming => pair.target,
            };
            let entry = out.entry(descriptor.counter_key(node)).or_insert(0);
            let Some(sum) = entry.checked_add(delta) else {
                return Ok(Err(format!(
                    "the change to the {} count of node {} leaves the counter's range",
                    pair.edge_type,
                    node.as_raw()
                )));
            };
            *entry = sum;
        }
    }
    Ok(Ok(out))
}

/// The descriptor kept for `edge_type` over `direction` and `measure`, and
/// whether its counts are covered.
fn covered_descriptor(
    engine: &StorageEngine,
    edge_type: &str,
    direction: Direction,
    measure: CardinalityMeasure,
) -> StorageResult<Option<CardinalityDescriptor>> {
    let Some(profile) = load_profile(engine, edge_type)? else {
        return Ok(None);
    };
    let Some(descriptor) = profile
        .descriptors
        .into_iter()
        .find(|d| d.direction == direction && d.measure == measure)
    else {
        return Ok(None);
    };
    if engine
        .get(Partition::Schema, &descriptor.coverage_key())?
        .is_none()
    {
        return Ok(None);
    }
    Ok(Some(descriptor))
}

/// The change the attempt stages to the covered count of `measure` over the
/// scope of `node`; `None` when no covered count is kept for the scope.
pub(crate) fn kept_change(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    measure: CardinalityMeasure,
    staged_counts: &FxHashMap<Vec<u8>, i64>,
) -> StorageResult<Option<i64>> {
    let Some(descriptor) = covered_descriptor(engine, edge_type, direction, measure)? else {
        return Ok(None);
    };
    Ok(Some(
        staged_counts
            .get(&descriptor.counter_key(node))
            .copied()
            .unwrap_or(0),
    ))
}

/// The kept count of `measure` over the scope of `node`, in the post-state
/// the attempt leaves: the committed count plus the change it stages.
/// `Ok(None)` when no covered count is kept for the scope, and the caller
/// enumerates instead. A count below zero, or one the staged change carries
/// out of range, is `Err`: it is corruption, never a count.
pub(crate) fn kept_count(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    measure: CardinalityMeasure,
    staged_counts: &FxHashMap<Vec<u8>, i64>,
) -> StorageResult<Option<Result<u64, String>>> {
    let Some(descriptor) = covered_descriptor(engine, edge_type, direction, measure)? else {
        return Ok(None);
    };
    let key = descriptor.counter_key(node);
    let committed = match engine.get(Partition::Counter, &key)? {
        Some(bytes) => crate::engine::merge::decode_counter(&bytes).map_err(|_| {
            StorageError::Serialization(format!(
                "cardinality count {}: holds {} bytes, expected 8",
                crate::error::printable_key(&key),
                bytes.len()
            ))
        })?,
        // Covered, so an absent count is a proved zero: the rebuild wrote
        // every non-zero one and every change since went through here.
        None => 0,
    };
    let staged = staged_counts.get(&key).copied().unwrap_or(0);
    Ok(Some(match committed.checked_add(staged) {
        Some(value) => u64::try_from(value).map_err(|_| {
            format!(
                "the {edge_type} count of node {} is {value}, below zero",
                node.as_raw()
            )
        }),
        None => Err(format!(
            "the {edge_type} count of node {} leaves the counter's range",
            node.as_raw()
        )),
    }))
}

/// Both measures of one incident scope over the post-state the attempt
/// leaves, by enumeration: neighbours from the adjacency, instances from it
/// for a single-edge type and from the pair entries for a discriminated one.
/// `None` when the scope cannot be counted exactly here: a temporal type,
/// whose bound holds over valid time, or a discriminated pair that is
/// adjacent with no identity behind it, which is evidence missing.
pub(crate) fn enumerated_count(
    engine: &StorageEngine,
    node: NodeId,
    edge_type: &str,
    direction: Direction,
    staged: StagedAdj<'_>,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Option<IncidentCount>> {
    use coordinode_core::graph::cardinality::IdentityShape;

    let key = match direction {
        Direction::Outgoing => encode_adj_key_forward(edge_type, node),
        Direction::Incoming => encode_adj_key_reverse(edge_type, node),
    };
    let neighbours = post_posting(engine, &key, staged, staged_points)?;
    let schema = load_edge_type(engine, edge_type)?;
    let shape = IdentityShape::of(schema.as_ref());
    let mut found = Vec::with_capacity(neighbours.len());
    for raw in neighbours.as_slice() {
        let neighbour = NodeId::from_raw(*raw);
        match shape {
            IdentityShape::Temporal => return Ok(None),
            IdentityShape::Single => found.push(IncidentInstance {
                neighbour,
                key: InstanceKey::new(),
            }),
            IdentityShape::Discriminated => {
                let (source, target) = pair_of(direction, node, *raw);
                let prefix = temporal_edgeprop_pair_prefix(edge_type, source, target);
                let (_, keys) = pair_entries(engine, &prefix, staged_points)?;
                if keys.is_empty() {
                    return Ok(None);
                }
                found.extend(
                    keys.into_iter()
                        .map(|key| IncidentInstance { neighbour, key }),
                );
            }
        }
    }
    Ok(Some(IncidentCount::of_set(found)))
}

/// Edge type `edge_type`'s committed definition. A type with no definition,
/// or only the marker an edge write leaves, has none. A definition that does
/// not decode is an error: guessing its shape would count by the wrong rule.
pub(crate) fn load_edge_type(
    engine: &StorageEngine,
    edge_type: &str,
) -> StorageResult<Option<coordinode_core::schema::definition::EdgeTypeSchema>> {
    use coordinode_core::schema::definition::{
        EdgeTypeSchema, encode_edge_type_current_revision_key, encode_edge_type_schema_key,
    };

    let unreadable = |key: &[u8], detail: String| StorageError::UnreadableCatalog {
        kind: "edge type",
        key: crate::error::printable_key(key),
        detail,
    };
    let pointer_key = encode_edge_type_current_revision_key(edge_type);
    let Some(pointer) = engine.get(Partition::Schema, &pointer_key)? else {
        return Ok(None);
    };
    let revision = u64::from_be_bytes((&pointer[..]).try_into().map_err(|_| {
        unreadable(
            &pointer_key,
            format!("the pointer is {} bytes, not 8", pointer.len()),
        )
    })?);
    let body_key = encode_edge_type_schema_key(edge_type, revision);
    let Some(body) = engine.get(Partition::Schema, &body_key)? else {
        return Err(unreadable(
            &pointer_key,
            format!("it points at revision {revision}, which is not stored"),
        ));
    };
    if body.is_empty() {
        return Ok(None);
    }
    EdgeTypeSchema::from_msgpack(&body)
        .map(Some)
        .map_err(|e| unreadable(&body_key, e.to_string()))
}

/// What a [`rebuild`] reconciled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rebuilt {
    /// Scopes whose count is not zero.
    pub scopes: u64,
    /// Scopes whose kept count differed from the stored edges and was
    /// corrected.
    pub corrected: u64,
}

/// Reconcile `descriptor`'s counts with the stored edges and cover them, in
/// `txn`'s commit.
///
/// Every scope's exact count and its kept count are read at one complete
/// snapshot, and the difference is staged as a change to the count. Commits
/// landing after the snapshot add their own changes, which the counts are
/// merged with, so the result is exact whatever order the two land in. The
/// coverage marker is written in the same unit, conditioned on its absence,
/// so two rebuilds of one descriptor cannot both correct it.
///
/// Precondition: the descriptor is committed, and every transaction open
/// when it landed has finished
/// ([`StorageEngine::await_transactions_through`]). A commit that read the
/// profile before the descriptor existed stages no change; one that lands
/// after the snapshot without its change would be missed by the
/// correction.
///
/// # Errors
///
/// A stored posting, entry or count does not decode; a discriminated pair is
/// adjacent with no instance, whose count cannot be built; a correction
/// leaves the counter's range; or the reads fail.
pub fn rebuild(
    txn: &mut Transaction<'_>,
    descriptor: &CardinalityDescriptor,
) -> StorageResult<Rebuilt> {
    let engine = txn.engine();
    let profile = load_profile(engine, &descriptor.edge_type)?;
    let discriminated = match &profile {
        Some(profile) if profile.descriptors.contains(descriptor) => profile.discriminated,
        _ => {
            return Err(StorageError::InvalidConfig(format!(
                "edge type '{}' does not declare this cardinality constraint",
                descriptor.edge_type
            )));
        }
    };
    let (snapshot, _pin) = engine.pin_latest_snapshot();

    let side = match descriptor.direction {
        Direction::Outgoing => ":out:",
        Direction::Incoming => ":in:",
    };
    let mut prefix = b"adj:".to_vec();
    prefix.extend_from_slice(descriptor.edge_type.as_bytes());
    prefix.extend_from_slice(side.as_bytes());

    let mut exact: FxHashMap<NodeId, u64> = FxHashMap::default();
    for guard in engine.snapshot_prefix_iter(&snapshot, Partition::Adj, &prefix)? {
        let (key, value) = guard.into_inner()?;
        let Some((_, direction, node)) = adj_parts(&key) else {
            return Err(StorageError::Serialization(format!(
                "{} is not an adjacency key",
                crate::error::printable_key(&key)
            )));
        };
        let members = decode_posting(&key, &value)?;
        let mut found = Vec::with_capacity(members.len());
        for raw in members.as_slice() {
            let neighbour = NodeId::from_raw(*raw);
            if !discriminated {
                found.push(IncidentInstance {
                    neighbour,
                    key: InstanceKey::new(),
                });
                continue;
            }
            let (source, target) = pair_of(direction, node, *raw);
            let entry_prefix = temporal_edgeprop_pair_prefix(&descriptor.edge_type, source, target);
            let before = found.len();
            for entry in
                engine.snapshot_prefix_iter(&snapshot, Partition::EdgeProp, &entry_prefix)?
            {
                let entry_key = entry.key()?;
                found.push(IncidentInstance {
                    neighbour,
                    key: entry_key[entry_prefix.len()..].to_vec(),
                });
            }
            if found.len() == before {
                return Err(StorageError::Serialization(format!(
                    "the {} pair {} -> {} is adjacent with no instance, so its count \
                     cannot be built",
                    descriptor.edge_type,
                    source.as_raw(),
                    target.as_raw()
                )));
            }
        }
        let count = IncidentCount::of_set(found).of(descriptor.measure);
        if count > 0 {
            exact.insert(node, count);
        }
    }

    let mut kept: FxHashMap<NodeId, i64> = FxHashMap::default();
    let counter_prefix = descriptor.counter_prefix();
    for guard in engine.snapshot_prefix_iter(&snapshot, Partition::Counter, &counter_prefix)? {
        let (key, value) = guard.into_inner()?;
        let node = descriptor.counter_node(&key).ok_or_else(|| {
            StorageError::Serialization(format!(
                "{} is not a cardinality count key",
                crate::error::printable_key(&key)
            ))
        })?;
        let value = crate::engine::merge::decode_counter(&value).map_err(|_| {
            StorageError::Serialization(format!(
                "cardinality count {}: holds {} bytes, expected 8",
                crate::error::printable_key(&key),
                value.len()
            ))
        })?;
        kept.insert(node, value);
    }

    let mut rebuilt = Rebuilt {
        scopes: exact.len() as u64,
        corrected: 0,
    };
    let mut nodes: Vec<NodeId> = exact.keys().chain(kept.keys()).copied().collect();
    nodes.sort_unstable();
    nodes.dedup();
    for node in nodes {
        let want = i64::try_from(exact.get(&node).copied().unwrap_or(0)).map_err(|_| {
            StorageError::Serialization(format!(
                "the {} count of node {} does not fit a counter",
                descriptor.edge_type,
                node.as_raw()
            ))
        })?;
        let have = kept.get(&node).copied().unwrap_or(0);
        let correction = want.checked_sub(have).ok_or_else(|| {
            StorageError::Serialization(format!(
                "the {} count of node {} cannot be corrected within the counter's range",
                descriptor.edge_type,
                node.as_raw()
            ))
        })?;
        if correction != 0 {
            txn.push_counter_delta(&descriptor.counter_key(node), correction);
            rebuilt.corrected += 1;
        }
    }
    let marker = descriptor.coverage_key();
    txn.expect_version(Partition::Schema, &marker, None)?;
    txn.put(Partition::Schema, &marker, b"")?;
    Ok(rebuilt)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
