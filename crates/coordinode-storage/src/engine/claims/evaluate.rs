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
    Evaluation::new(engine, staged, staged_points, read_ts).decide(claim)
}

/// One attempt's claims decided together, against one authoritative state.
///
/// What every claim of the attempt reads the same way is read once: the edge
/// types a destroyed node is probed under do not change between two deletions
/// of one commit, and a bulk delete would otherwise list them per node.
pub struct Evaluation<'a> {
    engine: &'a StorageEngine,
    staged: StagedAdj<'a>,
    staged_points: StagedPoints<'a>,
    read_ts: u64,
    edge_types: core::cell::OnceCell<Vec<String>>,
}

impl<'a> Evaluation<'a> {
    /// The context of one attempt viewing the store at `read_ts`.
    pub fn new(
        engine: &'a StorageEngine,
        staged: StagedAdj<'a>,
        staged_points: StagedPoints<'a>,
        read_ts: u64,
    ) -> Self {
        Self {
            engine,
            staged,
            staged_points,
            read_ts,
            edge_types: core::cell::OnceCell::new(),
        }
    }

    /// Every edge type the store has recorded, listed on first use.
    fn edge_types(&self) -> StorageResult<&[String]> {
        use coordinode_core::schema::definition::{
            EDGE_TYPE_SCHEMA_KEY_PREFIX, decode_edge_type_schema_key_name,
        };

        if let Some(types) = self.edge_types.get() {
            return Ok(types);
        }
        let mut types: Vec<String> = Vec::new();
        for guard in self
            .engine
            .prefix_scan(Partition::Schema, EDGE_TYPE_SCHEMA_KEY_PREFIX)?
        {
            let (key, _) = guard.into_inner()?;
            if let Some(name) = decode_edge_type_schema_key_name(&key) {
                if !types.iter().any(|t| t == name) {
                    types.push(name.to_string());
                }
            }
        }
        Ok(self.edge_types.get_or_init(|| types))
    }

    /// Decide `claim` in this context.
    pub fn decide(&self, claim: &Claim) -> StorageResult<Verdict> {
        let Self {
            engine,
            staged,
            staged_points,
            read_ts,
            ..
        } = *self;
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

            // A footprint, not a condition: it exists so the registry keeps an
            // erase of the pair out while this write is in flight, and there is
            // nothing about the pair this write's result depends on.
            (ClaimScope::Pair { .. }, ClaimPredicate::PairInstanceWritten) => Ok(Verdict::Holds),

            // The erase enumerated the pair's instances at its view; an
            // instance committed since is one it never removed.
            (
                ClaimScope::Pair {
                    source,
                    target,
                    edge_type,
                },
                ClaimPredicate::PairInstancesComplete,
            ) => pair_instances_unchanged(engine, *source, *target, edge_type, read_ts),

            (ClaimScope::Node(node), ClaimPredicate::EndpointAlive) => {
                endpoint_alive(engine, *node, staged_points, read_ts)
            }

            // The registry excludes the reference rights still in flight; this
            // answers for the ones that already committed. A reference committed
            // after the attempt's view is one its decision to delete never saw.
            (ClaimScope::Node(node), ClaimPredicate::EndpointDestroyed) => {
                endpoint_unreferenced(engine, self.edge_types()?, *node, staged, read_ts)
            }

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

            // The whole node: every edge type the store knows at the commit,
            // which includes one created after the attempt read, in both
            // directions.
            (ClaimScope::Node(node), ClaimPredicate::IncidentSetComplete) => {
                for edge_type in self.edge_types()? {
                    for direction in [Direction::Outgoing, Direction::Incoming] {
                        if incident_set_unchanged(engine, *node, edge_type, direction, read_ts)?
                            == Verdict::Broken
                        {
                            return Ok(Verdict::Broken);
                        }
                    }
                }
                Ok(Verdict::Holds)
            }

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

            // The writes were validated under the schema the attempt read. Any
            // write to its pointer or to the revision it named since then is a
            // different schema, including a property edited in place at the
            // same revision.
            (ClaimScope::LabelSchema(label), ClaimPredicate::SchemaRead { revision }) => {
                label_schema_unchanged(engine, label, *revision, read_ts)
            }

            // The new schema governs the nodes already stored, so every one of
            // them must satisfy it. The registry keeps writers validated under
            // the old schema out while this runs, and those that committed
            // before are in the state read here.
            (ClaimScope::LabelSchema(label), ClaimPredicate::SchemaActivated { revision }) => {
                label_schema_admits_stored_nodes(engine, label, *revision, staged_points)
            }

            // An overlap this evaluator has no evidence for stays undecided, and
            // a caller treats that as a refusal rather than a pass.
            _ => Ok(Verdict::Undecidable),
        }
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
    let at_view = posting(
        engine
            .snapshot_get(&read_ts, Partition::Adj, &key)?
            .as_deref(),
    )?;
    let now = posting(engine.get(Partition::Adj, &key)?.as_deref())?;

    Ok(if at_view == now {
        Verdict::Holds
    } else {
        Verdict::Broken
    })
}

/// Whether `label`'s schema is still the one an attempt viewing the store at
/// `read_ts` read at `revision`.
fn label_schema_unchanged(
    engine: &StorageEngine,
    label: &str,
    revision: u64,
    read_ts: u64,
) -> StorageResult<Verdict> {
    use coordinode_core::schema::definition::{
        encode_label_current_revision_key, encode_label_schema_key,
    };

    let pointer = encode_label_current_revision_key(label);
    if engine.written_since_snapshot(Partition::Schema, &pointer, read_ts)? {
        return Ok(Verdict::Broken);
    }
    // A label with no schema has no body to have changed.
    if revision != 0 {
        let body = encode_label_schema_key(label, revision);
        if engine.written_since_snapshot(Partition::Schema, &body, read_ts)? {
            return Ok(Verdict::Broken);
        }
    }
    Ok(Verdict::Holds)
}

/// Whether every stored node whose primary label is `label` satisfies the
/// schema the attempt activates at `revision`.
///
/// The primary label is the one a write validates against, so it is the one
/// the activation answers for. The attempt's own staged node writes are part
/// of the post-state and replace what is stored under the same key.
fn label_schema_admits_stored_nodes(
    engine: &StorageEngine,
    label: &str,
    revision: u64,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Verdict> {
    use coordinode_core::schema::definition::{LabelSchema, encode_label_schema_key};

    // Removing the schema admits everything.
    if revision == 0 {
        return Ok(Verdict::Holds);
    }
    let body_key = encode_label_schema_key(label, revision);
    let body = match staged_points.get(&(Partition::Schema, body_key.clone())) {
        Some(Some(bytes)) => Some(bytes.clone()),
        Some(None) => None,
        None => engine
            .get(Partition::Schema, &body_key)?
            .map(|b| b.to_vec()),
    };
    let Some(body) = body else {
        // An activation of a revision that is nowhere describes nothing the
        // nodes could be checked against.
        return Ok(Verdict::Undecidable);
    };
    let schema = LabelSchema::from_msgpack(&body).map_err(|e| {
        crate::error::StorageError::Serialization(format!("label schema '{label}': {e}"))
    })?;
    Ok(
        match first_label_schema_violation(engine, &schema, staged_points)? {
            None => Verdict::Holds,
            Some(_) => Verdict::Broken,
        },
    )
}

/// A stored node that breaks a label schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSchemaViolation {
    /// The node.
    pub node: NodeId,
    /// What it breaks, as the write path would report it.
    pub reason: String,
}

/// The first stored node whose primary label is `schema`'s label and that
/// breaks `schema`, or `None` when every one satisfies it.
///
/// The primary label is the one a write validates against, so it is the one
/// a schema answers for. `staged_points` are writes of the attempt asking:
/// they are part of the state checked and replace what is stored under the
/// same key. Overflow properties count under the names they were stored by.
///
/// # Errors
///
/// A stored record or the field dictionary does not decode.
pub fn first_label_schema_violation(
    engine: &StorageEngine,
    schema: &coordinode_core::schema::definition::LabelSchema,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Option<LabelSchemaViolation>> {
    use coordinode_core::graph::node::NodeRecord;

    // A flexible schema checks only NOT NULL and the per-node constraints;
    // without either there is nothing a stored node can break, and the scan
    // is skipped. Uniqueness is decided by the constraint's index build.
    if !schema.mode.validates_declared()
        && !schema.mode.rejects_unknown()
        && !schema
            .properties
            .values()
            .any(|p| p.not_null && p.default.is_none())
        && !schema
            .constraints()
            .iter()
            .any(coordinode_core::schema::definition::NodeConstraint::checks_each_node)
    {
        return Ok(None);
    }

    let dictionary = crate::engine::metadata::load_field_dictionary(engine)?;
    let names: std::collections::HashMap<u32, String> = dictionary
        .iter()
        .map(|(name, id)| (id, name.to_string()))
        .collect();
    let check = |key: &[u8], bytes: &[u8]| -> StorageResult<Option<LabelSchemaViolation>> {
        let record = NodeRecord::from_msgpack(bytes)
            .map_err(|e| crate::error::StorageError::Serialization(format!("node record: {e}")))?;
        if record.labels.first() != Some(&schema.name) {
            return Ok(None);
        }
        Ok(
            record_violation(schema, record, &names, dictionary.frontier()).map(|reason| {
                LabelSchemaViolation {
                    node: node_of_record_key(key),
                    reason,
                }
            }),
        )
    };

    for guard in engine.prefix_scan(Partition::Node, NODE_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        if !is_node_record_key(&key) || staged_points.contains_key(&(Partition::Node, key.to_vec()))
        {
            continue;
        }
        if let Some(violation) = check(&key, &value)? {
            return Ok(Some(violation));
        }
    }
    for ((part, key), value) in staged_points.iter() {
        if *part != Partition::Node || !is_node_record_key(key) {
            continue;
        }
        if let Some(bytes) = value {
            if let Some(violation) = check(key, bytes)? {
                return Ok(Some(violation));
            }
        }
    }
    Ok(None)
}

/// The node a record key addresses: the id follows `node:<shard:2>:`.
fn node_of_record_key(key: &[u8]) -> NodeId {
    let mut id = [0u8; 8];
    // `is_node_record_key` admitted only keys at least 16 bytes long.
    id.copy_from_slice(&key[8..16]);
    NodeId::from_raw(u64::from_be_bytes(id))
}

/// The prefix every node record key starts with.
const NODE_KEY_PREFIX: &[u8] = b"node:";

/// Whether `key` addresses a node record: the current record of a node, or
/// one valid-time version of a temporal one.
fn is_node_record_key(key: &[u8]) -> bool {
    key.starts_with(NODE_KEY_PREFIX) && (key.len() == 16 || key.len() == 25)
}

/// What `record` breaks in `schema`, its overflow properties included (they
/// are properties of the node under the names they were stored by), or
/// `None` when it satisfies it.
fn record_violation(
    schema: &coordinode_core::schema::definition::LabelSchema,
    record: coordinode_core::graph::node::NodeRecord,
    names: &std::collections::HashMap<u32, String>,
    frontier: u32,
) -> Option<String> {
    use coordinode_core::schema::validation::validate_properties;

    let first = |errors: Vec<coordinode_core::schema::validation::ValidationError>| {
        errors
            .into_iter()
            .next()
            .map_or_else(|| "invalid".to_string(), |e| e.to_string())
    };
    let mut props = record.props;
    // The engine stamps every version of a temporal node with its own
    // metadata after the write was validated; the definition answers only
    // for what users wrote.
    if schema.temporal {
        props.retain(|id, _| {
            !names.get(id).is_some_and(|name| {
                coordinode_core::schema::definition::TEMPORAL_ENGINE_FIELDS.contains(&name.as_str())
            })
        });
    }
    let Some(extra) = record.extra.filter(|e| !e.is_empty()) else {
        return validate_properties(schema, &props, names).err().map(first);
    };
    // The overflow names take ids past every bound one, so they can neither
    // collide with a stored property nor resolve to another name.
    let mut names = names.clone();
    for (offset, (name, value)) in extra.into_iter().enumerate() {
        let Some(id) = u32::try_from(offset)
            .ok()
            .and_then(|o| frontier.checked_add(o))
        else {
            return Some(format!(
                "more overflow properties than field ids, at '{name}'"
            ));
        };
        names.insert(id, name);
        props.insert(id, value);
    }
    validate_properties(schema, &props, &names).err().map(first)
}

/// Decode an adjacency posting, refusing bytes that do not decode.
///
/// A claim decided on a posting it could not read would be decided on an
/// empty set, which can pass an absence or an unchanged-set check that the
/// real data fails. Corruption is an error here, never an answer.
fn decode_posting(bytes: &[u8]) -> StorageResult<PostingList> {
    PostingList::from_bytes(bytes)
        .map_err(|e| crate::error::StorageError::Serialization(format!("adjacency posting: {e}")))
}

/// The members of an adjacency posting, or none when the key is absent.
fn posting(bytes: Option<&[u8]>) -> StorageResult<Vec<u64>> {
    Ok(match bytes {
        Some(b) => decode_posting(b)?.as_slice().to_vec(),
        None => Vec::new(),
    })
}

/// Whether the pair's instances are the ones the erase enumerated.
///
/// Every instance of a pair lives under the pair's own edge-property key or
/// directly beneath it (one key per version or discriminator), so one prefix
/// covers them all. Only the keys are compared: an instance rewritten in place
/// is the same key, and the erase deletes that key, which first-committer-wins
/// already decides.
fn pair_instances_unchanged(
    engine: &StorageEngine,
    source: NodeId,
    target: NodeId,
    edge_type: &str,
    read_ts: u64,
) -> StorageResult<Verdict> {
    let prefix = coordinode_core::graph::edge::encode_edgeprop_key(edge_type, source, target);
    let mut at_view = engine.snapshot_prefix_iter(&read_ts, Partition::EdgeProp, &prefix)?;
    let mut now = engine.prefix_scan(Partition::EdgeProp, &prefix)?;
    loop {
        match (at_view.next(), now.next()) {
            (None, None) => return Ok(Verdict::Holds),
            (Some(seen), Some(current)) => {
                if seen.key()? != current.key()? {
                    return Ok(Verdict::Broken);
                }
            }
            _ => return Ok(Verdict::Broken),
        }
    }
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

    let at_view = posting(
        engine
            .snapshot_get(&read_ts, Partition::Adj, &key)?
            .as_deref(),
    )?
    .contains(&target.as_raw());
    let now = posting(engine.get(Partition::Adj, &key)?.as_deref())?.contains(&target.as_raw());

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

/// Whether a node being destroyed gained no edge the attempt did not see.
///
/// A deletion decides on the incident set at its view: a plain DELETE that
/// found none, a DETACH that removes the ones it found. An edge committed to
/// the node afterwards, of any type and in either direction, is a member that
/// decision never accounted for, and admitting the deletion would leave it
/// pointing at a node that is gone. The probe is key-only per edge type; the
/// adjacency is read only for a key that moved, and then with the attempt's
/// own removals applied, so a DETACH that removed what it saw is not refused
/// for its own work.
fn endpoint_unreferenced(
    engine: &StorageEngine,
    edge_types: &[String],
    node: NodeId,
    staged: StagedAdj<'_>,
    read_ts: u64,
) -> StorageResult<Verdict> {
    for edge_type in edge_types {
        for direction in [Direction::Outgoing, Direction::Incoming] {
            let adj_key = match direction {
                Direction::Outgoing => encode_adj_key_forward(edge_type, node),
                Direction::Incoming => encode_adj_key_reverse(edge_type, node),
            };
            if engine.written_since_snapshot(Partition::Adj, &adj_key, read_ts)?
                && !adjacency(engine, node, edge_type, direction, staged)?.is_empty()
            {
                return Ok(Verdict::Broken);
            }
        }
    }
    Ok(Verdict::Holds)
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
        Some(bytes) => decode_posting(&bytes)?,
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
        // A row that cannot be read is not a row that is absent: skipping it
        // would undercount and admit the instance that breaks the bound.
        let key = guard.key()?;
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
