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
    Adjacency, Claim, ClaimPredicate, ClaimScope, Direction, UncoveredSource,
};

use crate::engine::core::StorageEngine;
use crate::engine::partition::Partition;
use crate::error::StorageResult;

/// The attempt's own staged adjacency mutations, in the order they were
/// staged, so the evaluation sees the post-state rather than the state before
/// the attempt ran.
pub type StagedAdj<'a> = crate::engine::cardinality::StagedAdj<'a>;

/// The attempt's staged point writes, as the commit will apply them: `Some`
/// for a value written, `None` for a row deleted.
///
/// The evaluation needs them for the same reason it needs the adjacency
/// operands. A node and the edge attached to it, created by one statement,
/// are one valid post-state; reading only committed state would find the node
/// absent and call the edge dangling, refusing the most ordinary write there
/// is.
pub type StagedPoints<'a> = crate::engine::cardinality::StagedPoints<'a>;

/// Whether a claim still holds.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// The condition holds over the whole post-state.
    Holds,
    /// The condition does not hold; the attempt must not be admitted.
    Broken,
    /// This evaluator cannot decide the claim, so nothing may be concluded
    /// from it. A caller treats it as a refusal rather than as a pass: an
    /// undecidable condition is not a satisfied one.
    Undecidable,
    /// A unique value the attempt takes is held by another node in the
    /// post-state.
    HeldBy {
        /// The node holding it.
        holder: NodeId,
        /// The values the holder has in the index.
        values: Vec<coordinode_core::graph::types::Value>,
    },
    /// Deciding the claim would read more than `limit` stored rows. Nothing
    /// is concluded either way; the same attempt may be admitted once the
    /// work needed shrinks.
    OverLimit {
        /// The most rows the decision reads.
        limit: u64,
    },
    /// A node the attempt writes breaks a rule the claim holds it to.
    Violates(LabelSchemaViolation),
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
    /// Node rows the attempt changes through document deltas, keyed first.
    node_deltas: &'a [(Vec<u8>, Vec<u8>)],
    /// The changes to kept cardinality counts the attempt derived.
    kept_counts: Option<&'a rustc_hash::FxHashMap<Vec<u8>, i64>>,
    /// The view the attempt's state reads are compared at.
    read_ts: u64,
    /// The first sequence number the attempt could not have seen: a write at
    /// or after it is one its reads missed. Below `read_ts` when commits were
    /// in flight under the view as it was taken.
    unseen_from: u64,
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
            node_deltas: &[],
            kept_counts: None,
            read_ts,
            unseen_from: read_ts,
            edge_types: core::cell::OnceCell::new(),
        }
    }

    /// The first sequence number the attempt could not have seen, when
    /// commits in flight under its view put it below the view.
    pub fn with_unseen_from(mut self, unseen_from: u64) -> Self {
        self.unseen_from = unseen_from.min(self.read_ts);
        self
    }

    /// The changes to kept cardinality counts the attempt derived from its
    /// edge effects. A bound over a covered count is decided from the count
    /// these leave; without them it is decided by enumeration, which reads
    /// the same post-state.
    pub fn with_kept_counts(mut self, kept: &'a rustc_hash::FxHashMap<Vec<u8>, i64>) -> Self {
        self.kept_counts = Some(kept);
        self
    }

    /// The node rows the attempt changes through document deltas, whose
    /// committed bytes are not their post-state.
    pub fn with_node_deltas(mut self, node_deltas: &'a [(Vec<u8>, Vec<u8>)]) -> Self {
        self.node_deltas = node_deltas;
        self
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
            node_deltas,
            read_ts,
            unseen_from,
            ..
        } = *self;
        match (&claim.scope, &claim.predicate) {
            (
                ClaimScope::Incident {
                    node,
                    edge_type,
                    direction,
                },
                ClaimPredicate::CardinalityBound { measure, bound, .. },
            ) => {
                // A kept count is only as good as the changes derived for
                // this attempt: an evaluation that was not handed them reads
                // the scope instead.
                // A kept count is never below zero, so an attempt that adds to
                // the scope's count proves a lower bound by its own change,
                // without folding the scope's count, which on a node many
                // writers add to is a long chain of changes.
                if *bound == coordinode_core::txn::invariant::CardinalityBound::AtLeastOne {
                    if let Some(kept) = self.kept_counts {
                        if crate::engine::cardinality::kept_change(
                            engine, *node, edge_type, *direction, *measure, kept,
                        )?
                        .is_some_and(|change| change >= 1)
                        {
                            return Ok(Verdict::Holds);
                        }
                    }
                }
                let kept = match self.kept_counts {
                    Some(kept) => crate::engine::cardinality::kept_count(
                        engine, *node, edge_type, *direction, *measure, kept,
                    )?,
                    None => None,
                };
                let count = match kept {
                    Some(Ok(count)) => count,
                    // A count below zero or out of range is corruption: no
                    // bound is decided from it.
                    Some(Err(_)) => return Ok(Verdict::Undecidable),
                    None => {
                        let Some(count) = crate::engine::cardinality::enumerated_count(
                            engine,
                            *node,
                            edge_type,
                            *direction,
                            staged,
                            staged_points,
                        )?
                        else {
                            return Ok(Verdict::Undecidable);
                        };
                        count.of(*measure)
                    }
                };
                Ok(if bound.admits(count) {
                    Verdict::Holds
                } else {
                    Verdict::Broken
                })
            }

            // A footprint: it keeps a second count of the pair out while this
            // one is derived, and asserts nothing about the pair itself.
            (ClaimScope::Pair { .. }, ClaimPredicate::PairCounted { .. }) => Ok(Verdict::Holds),

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
            (ClaimScope::Node(node), ClaimPredicate::EndpointDestroyed) => endpoint_unreferenced(
                engine,
                self.edge_types()?,
                *node,
                staged,
                staged_points,
                read_ts,
            ),

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

            // The registry and the write scope keep admissions in flight out;
            // this answers for those already committed, whose references and
            // upload pins are in the counts as they stand now.
            (ClaimScope::Record(key), ClaimPredicate::BlobRetired { .. }) => {
                blob_content_unheld(engine, key)
            }

            // The writes were validated under the schema the attempt read. Any
            // write to its pointer or to the revision it named since then is a
            // different schema, including a property edited in place at the
            // same revision.
            (ClaimScope::LabelSchema(label), ClaimPredicate::SchemaRead { revision }) => {
                label_schema_unchanged(engine, label, *revision, unseen_from)
            }

            // The new schema governs the nodes already stored, so every one of
            // them must satisfy it. The registry keeps writers validated under
            // the old schema out while this runs, and those that committed
            // before are in the state read here.
            (ClaimScope::LabelSchema(label), ClaimPredicate::SchemaActivated { revision }) => {
                label_schema_admits_stored_nodes(engine, label, *revision, staged_points)
            }

            // Publishing a candidate decides nothing over the data: its proof
            // does, after the commit. The registry is what keeps writers
            // under the previous revision out while it lands.
            (ClaimScope::LabelSchema(_), ClaimPredicate::CandidatePublished { .. }) => {
                Ok(Verdict::Holds)
            }

            // The proof of a candidate covers what was stored when it began;
            // a node written after it is held to the candidate here, so the
            // records read are only the attempt's own.
            (ClaimScope::LabelSchema(label), ClaimPredicate::CandidateAdmits { revision }) => {
                candidate_admits_written_nodes(
                    engine,
                    label,
                    *revision,
                    staged_points,
                    self.node_deltas,
                )
            }

            // A published index answers for the value through its entry key:
            // every holder writes it, and the attempt writes it too, so a
            // holder committed since the attempt's view meets it there. While
            // the index is being built, a node the build has not reached holds
            // its value without an entry, and only reading those nodes tells.
            (
                ClaimScope::UniqueValue { tuple, .. },
                ClaimPredicate::UniqueHolder { node, uncovered },
            ) => match uncovered {
                None => Ok(Verdict::Holds),
                Some(source) => uncovered_holders(
                    engine,
                    &core::iter::once((tuple.as_slice(), *node)).collect(),
                    source,
                    staged_points,
                    node_deltas,
                ),
            },

            // An overlap this evaluator has no evidence for stays undecided, and
            // a caller treats that as a refusal rather than a pass.
            _ => Ok(Verdict::Undecidable),
        }
    }
}

/// The unique values an attempt takes in one index generation while it is
/// being built, each keyed by its encoded tuple and naming the node taking it.
pub type WantedValues<'a> = rustc_hash::FxHashMap<&'a [u8], NodeId>;

/// The verdict on the unique values `wanted` while the index is being built:
/// `HeldBy` a node of `source`'s label, other than the one taking the value,
/// whose post-state record holds one of them, among the stored nodes after
/// the key the build covered through and the attempt's own node writes;
/// `Holds` when none does. One read decides every value of the generation,
/// so a statement taking many values reads the uncovered nodes once.
///
/// The nodes up to that key need no reading: the build committed their
/// entries, which the attempt's entry key meets as any holder's does. A node
/// the attempt changes through a document delta is not judged by its
/// committed bytes, which are not its post-state: the attempt maintained that
/// node's entries as it changed it, and its own entries are checked against
/// each other as they are staged. The read is bounded by `source.read_limit`
/// stored rows; past it the claim is `OverLimit`, which is neither a pass nor
/// a duplicate.
pub fn uncovered_holders(
    engine: &StorageEngine,
    wanted: &WantedValues<'_>,
    source: &UncoveredSource,
    staged_points: StagedPoints<'_>,
    node_deltas: &[(Vec<u8>, Vec<u8>)],
) -> StorageResult<Verdict> {
    use coordinode_core::graph::node::{
        NodeRecord, decode_node_key, decode_temporal_node_key, encode_node_key,
    };
    use coordinode_core::index::derive::tuples;

    let mut closures = Closures::new(engine, staged_points);
    let mut holder = |key: &[u8], bytes: &[u8]| -> StorageResult<Option<Verdict>> {
        // A temporal node holds the values of each of its versions.
        let Some((shard, holder)) = decode_node_key(key)
            .or_else(|| decode_temporal_node_key(key).map(|(shard, id, _)| (shard, id)))
        else {
            return Ok(None);
        };
        if shard != source.shard_id {
            return Ok(None);
        }
        let record = NodeRecord::from_msgpack(bytes)
            .map_err(|e| crate::error::StorageError::Serialization(format!("node record: {e}")))?;
        // The index covers every node carrying its label, in any role: its
        // type, implied by its type, or auxiliary.
        if !closures.carries(&record.labels, &source.label)? {
            return Ok(None);
        }
        let Some(values) = source.interpretation.record_membership(&record) else {
            return Ok(None);
        };
        // A node taking a value it already holds holds it still.
        if !tuples(&values).iter().any(|t| {
            wanted
                .get(t.as_slice())
                .is_some_and(|taker| *taker != holder)
        }) {
            return Ok(None);
        }
        Ok(Some(Verdict::HeldBy { holder, values }))
    };

    // The rows the attempt writes are judged by their post-state, below, or
    // not at all; looked up per stored row, so collected once.
    let touched: rustc_hash::FxHashSet<&[u8]> = staged_points
        .keys()
        .filter(|(part, _)| *part == Partition::Node)
        .map(|(_, key)| key.as_slice())
        .chain(node_deltas.iter().map(|(key, _)| key.as_slice()))
        .collect();

    let prefix = encode_node_key(source.shard_id, NodeId::from_raw(0))[..8].to_vec();
    let start = match &source.covered_through {
        // The smallest key after the covered one.
        Some(covered) => {
            let mut start = covered.clone();
            start.push(0);
            start
        }
        None => prefix.clone(),
    };
    // Inclusive end past every node key of the shard: the keys are at most
    // 25 bytes, the prefix 8.
    let mut end = prefix.clone();
    end.extend_from_slice(&[0xFF; 32]);
    let mut read = 0u64;
    for guard in engine.range_scan(Partition::Node, &start, &end)? {
        let (key, value) = guard.into_inner()?;
        if !key.starts_with(&prefix) || touched.contains(&*key) {
            continue;
        }
        read += 1;
        if read > source.read_limit {
            return Ok(Verdict::OverLimit {
                limit: source.read_limit,
            });
        }
        if let Some(held) = holder(&key, &value)? {
            return Ok(held);
        }
    }
    for ((part, key), value) in staged_points.iter() {
        if *part != Partition::Node || !key.starts_with(&prefix) {
            continue;
        }
        if let Some(bytes) = value {
            if let Some(held) = holder(key, bytes)? {
                return Ok(held);
            }
        }
    }
    Ok(Verdict::Holds)
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

/// Whether `label`'s schema is still the one an attempt read at `revision`,
/// `unseen_from` being the first sequence number it could not have seen.
///
/// Decided from the current state: the pointer must name `revision` now
/// (be absent for `0`), and neither it nor that revision's body may have been
/// written since. Only keys present now are asked "written since", which
/// reads no history, so a view below the retention horizon decides as well.
fn label_schema_unchanged(
    engine: &StorageEngine,
    label: &str,
    revision: u64,
    unseen_from: u64,
) -> StorageResult<Verdict> {
    use coordinode_core::schema::definition::{
        encode_label_current_revision_key, encode_label_schema_key,
    };

    let pointer = encode_label_current_revision_key(label);
    let now = engine.get(Partition::Schema, &pointer)?;
    let Some(now) = now else {
        // No schema now: the read stands only if it saw none either.
        return Ok(if revision == 0 {
            Verdict::Holds
        } else {
            Verdict::Broken
        });
    };
    let named = <[u8; 8]>::try_from(now.as_ref())
        .ok()
        .map(u64::from_be_bytes);
    if revision == 0 || named != Some(revision) {
        return Ok(Verdict::Broken);
    }
    // The pointer names the same revision; a rewrite of it or of the body
    // in place since the view is still a different schema. The body of the
    // revision the pointer names is present, so neither probe reads history.
    let body = encode_label_schema_key(label, revision);
    if engine.written_since_snapshot(Partition::Schema, &pointer, unseen_from)?
        || engine.written_since_snapshot(Partition::Schema, &body, unseen_from)?
    {
        return Ok(Verdict::Broken);
    }
    Ok(Verdict::Holds)
}

/// Whether every stored node carrying `label` satisfies the schema the
/// attempt activates at `revision`: its kind and implied labels, the whole
/// definition for the nodes it types and the explicit constraints for the
/// others.
///
/// The attempt's own staged node writes are part of the post-state and
/// replace what is stored under the same key.
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
    // The kind and the implied labels bind the stored nodes as the
    // properties do: a node of the type written beside the declaration must
    // carry every implied label, and none may have an auxiliary label as its
    // type.
    if schema.is_auxiliary() || !schema.implies().is_empty() {
        let implied = implied_closure(engine, &schema, staged_points)?;
        if first_node_breaking_label_kind(engine, label, schema.kind(), &implied, staged_points)?
            .is_some()
        {
            return Ok(Verdict::Broken);
        }
    }
    Ok(
        match first_label_schema_violation(engine, &schema, staged_points)? {
            None => Verdict::Holds,
            Some(_) => Verdict::Broken,
        },
    )
}

/// Every label `schema` implies, transitively, without itself, under the
/// definitions the attempt leaves.
fn implied_closure(
    engine: &StorageEngine,
    schema: &coordinode_core::schema::definition::LabelSchema,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Vec<String>> {
    let mut closure: Vec<String> = Vec::new();
    let mut pending: Vec<String> = schema.implies().to_vec();
    while let Some(label) = pending.pop() {
        if label == schema.name || closure.contains(&label) {
            continue;
        }
        if let Some(implied) = current_label_schema(engine, &label, staged_points)? {
            pending.extend(implied.implies().iter().cloned());
        }
        closure.push(label);
    }
    Ok(closure)
}

/// What each type met implies, under the definitions an attempt leaves (its
/// own staged schema writes over the stored ones), compiled once per type:
/// a judgement over many records reads no definition per record.
pub(crate) struct Closures<'e> {
    engine: &'e StorageEngine,
    staged_points: StagedPoints<'e>,
    implied: coordinode_core::schema::labels::ImpliedCache,
}

impl<'e> Closures<'e> {
    pub(crate) fn new(engine: &'e StorageEngine, staged_points: StagedPoints<'e>) -> Self {
        Self {
            engine,
            staged_points,
            implied: coordinode_core::schema::labels::ImpliedCache::default(),
        }
    }

    /// What type `primary` implies, compiled on first use.
    pub(crate) fn implied(
        &mut self,
        primary: &str,
    ) -> StorageResult<&coordinode_core::schema::labels::ImpliedLabels> {
        let (engine, staged_points) = (self.engine, self.staged_points);
        self.implied
            .compile(primary, |l| {
                Ok::<_, crate::error::StorageError>(
                    current_label_schema(engine, l, staged_points)?
                        .as_ref()
                        .map(coordinode_core::schema::labels::LabelFacts::of)
                        .unwrap_or_default(),
                )
            })?
            .map_err(|e| crate::error::StorageError::Serialization(e.to_string()))?;
        self.implied.implied(primary).ok_or_else(|| {
            crate::error::StorageError::Serialization(format!(
                "what :{primary} implies was not compiled"
            ))
        })
    }

    /// Whether a node storing `stored` carries `label`: stored, or implied
    /// by its type.
    fn carries(&mut self, stored: &[String], label: &str) -> StorageResult<bool> {
        if stored.iter().any(|l| l == label) {
            return Ok(true);
        }
        match stored.first() {
            Some(primary) => Ok(self.implied(primary)?.contains(label)),
            None => Ok(false),
        }
    }
}

/// The current definition of `label` as the attempt leaves it: its own
/// staged pointer and body over the stored ones. `None` for a label without
/// a definition.
fn current_label_schema(
    engine: &StorageEngine,
    label: &str,
    staged_points: StagedPoints<'_>,
) -> StorageResult<Option<coordinode_core::schema::definition::LabelSchema>> {
    use coordinode_core::schema::definition::{
        LabelSchema, encode_label_current_revision_key, encode_label_schema_key,
    };

    let read = |key: Vec<u8>| -> StorageResult<Option<Vec<u8>>> {
        match staged_points.get(&(Partition::Schema, key.clone())) {
            Some(value) => Ok(value.clone()),
            None => Ok(engine.get(Partition::Schema, &key)?.map(|b| b.to_vec())),
        }
    };
    let serialization = |what: String| crate::error::StorageError::Serialization(what);
    let Some(pointer) = read(encode_label_current_revision_key(label))? else {
        return Ok(None);
    };
    let revision: [u8; 8] = pointer.as_slice().try_into().map_err(|_| {
        serialization(format!(
            "label schema pointer '{label}': {} bytes",
            pointer.len()
        ))
    })?;
    let Some(body) = read(encode_label_schema_key(label, u64::from_be_bytes(revision)))? else {
        return Ok(None);
    };
    LabelSchema::from_msgpack(&body)
        .map(Some)
        .map_err(|e| serialization(format!("label schema '{label}': {e}")))
}

/// Whether every node the attempt writes carrying `label` satisfies the
/// candidate the label's schema carried at `revision`.
///
/// Only the attempt's own node writes are read: a whole record staged, or a
/// stored one under the document deltas the attempt adds to it. A revision
/// without a candidate (activated or withdrawn since) holds nothing more;
/// the attempt's own read of the schema is decided by its `SchemaRead`.
fn candidate_admits_written_nodes(
    engine: &StorageEngine,
    label: &str,
    revision: u64,
    staged_points: StagedPoints<'_>,
    node_deltas: &[(Vec<u8>, Vec<u8>)],
) -> StorageResult<Verdict> {
    use coordinode_core::graph::doc_delta::{DocDelta, PREFIX_DOC_DELTA};
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::schema::definition::{LabelSchema, encode_label_schema_key};

    let serialization = |what: &str, e: &dyn core::fmt::Display| {
        crate::error::StorageError::Serialization(format!("{what}: {e}"))
    };
    let body_key = encode_label_schema_key(label, revision);
    let body = match staged_points.get(&(Partition::Schema, body_key.clone())) {
        Some(Some(bytes)) => Some(bytes.clone()),
        Some(None) => None,
        None => engine
            .get(Partition::Schema, &body_key)?
            .map(|b| b.to_vec()),
    };
    let Some(body) = body else {
        return Ok(Verdict::Undecidable);
    };
    let schema = LabelSchema::from_msgpack(&body)
        .map_err(|e| serialization(&format!("label schema '{label}'"), &e))?;
    let Some(candidate) = schema.candidate_schema() else {
        return Ok(Verdict::Holds);
    };

    let mut names: Option<(std::collections::HashMap<u32, String>, u32)> = None;
    let mut closures = Closures::new(engine, staged_points);
    let mut check = |key: &[u8], record: NodeRecord| -> StorageResult<Option<Verdict>> {
        if !closures.carries(&record.labels, label)? {
            return Ok(None);
        }
        if names.is_none() {
            let dictionary = crate::engine::metadata::load_field_dictionary(engine)?;
            names = Some((
                dictionary
                    .iter()
                    .map(|(name, id)| (id, name.to_string()))
                    .collect(),
                dictionary.frontier(),
            ));
        }
        let Some((names, frontier)) = names.as_ref() else {
            return Ok(None);
        };
        // A carrier answers to the candidate as the proof holds it: whole
        // as its type, by its explicit constraints otherwise.
        Ok(
            carrier_violation(&candidate, record, true, names, *frontier).map(|reason| {
                Verdict::Violates(LabelSchemaViolation {
                    node: node_of_record_key(key),
                    reason,
                })
            }),
        )
    };

    for ((part, key), value) in staged_points.iter() {
        let (Partition::Node, Some(bytes)) = (*part, value) else {
            continue;
        };
        if !is_node_record_key(key) {
            continue;
        }
        let record =
            NodeRecord::from_msgpack(bytes).map_err(|e| serialization("node record", &e))?;
        if let Some(verdict) = check(key, record)? {
            return Ok(verdict);
        }
    }
    // A document delta on a stored node: the record it leaves is the stored
    // one with every delta of the attempt applied, in order.
    let mut deltas: rustc_hash::FxHashMap<&[u8], Vec<DocDelta>> = rustc_hash::FxHashMap::default();
    for (key, operand) in node_deltas {
        let Some((&PREFIX_DOC_DELTA, body)) = operand.split_first() else {
            return Ok(Verdict::Undecidable);
        };
        let delta = DocDelta::decode(body).map_err(|e| serialization("node document delta", &e))?;
        deltas.entry(key.as_slice()).or_default().push(delta);
    }
    for (key, deltas) in deltas {
        if !is_node_record_key(key) {
            continue;
        }
        let Some(stored) = engine.get(Partition::Node, key)? else {
            continue;
        };
        let mut record = crate::engine::merge::decode_node_record(&stored)
            .map_err(|e| serialization("node record under a delta", &e))?;
        crate::engine::merge::apply_doc_deltas_to_record(&mut record, &deltas);
        if let Some(verdict) = check(key, record)? {
            return Ok(verdict);
        }
    }
    Ok(Verdict::Holds)
}

/// A stored node that breaks a label schema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSchemaViolation {
    /// The node.
    pub node: NodeId,
    /// What it breaks, as the write path would report it.
    pub reason: String,
}

/// The first stored node carrying `schema`'s label that breaks it, or `None`
/// when every one satisfies it.
///
/// A node whose type the label is answers for the whole definition; one
/// carrying it implied or auxiliary answers for its explicit constraints
/// (`carrier_violation`). `staged_points` are writes of the attempt asking:
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

    // Nothing a stored node could break: the scan is skipped.
    if !schema.can_refuse_nodes() {
        return Ok(None);
    }

    let dictionary = crate::engine::metadata::load_field_dictionary(engine)?;
    let names: std::collections::HashMap<u32, String> = dictionary
        .iter()
        .map(|(name, id)| (id, name.to_string()))
        .collect();
    let mut closures = Closures::new(engine, staged_points);
    let mut check = |key: &[u8], bytes: &[u8]| -> StorageResult<Option<LabelSchemaViolation>> {
        let record = NodeRecord::from_msgpack(bytes)
            .map_err(|e| crate::error::StorageError::Serialization(format!("node record: {e}")))?;
        let carried = closures.carries(&record.labels, &schema.name)?;
        Ok(
            carrier_violation(schema, record, carried, &names, dictionary.frontier()).map(
                |reason| LabelSchemaViolation {
                    node: node_of_record_key(key),
                    reason,
                },
            ),
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

/// The first stored node that declaring `label` as `kind`, implying the
/// labels `implied` (the whole closure), would leave invalid: for an
/// auxiliary label a node it types, for a type label a node it types that
/// lacks one of the implied labels. `None` when every node stays valid.
///
/// Only labels are decoded. `staged_points` are the asking attempt's own
/// writes and replace what is stored under their keys.
///
/// # Errors
///
/// A stored record's labels do not decode.
pub fn first_node_breaking_label_kind(
    engine: &StorageEngine,
    label: &str,
    kind: coordinode_core::schema::definition::LabelKind,
    implied: &[String],
    staged_points: StagedPoints<'_>,
) -> StorageResult<Option<NodeId>> {
    use coordinode_core::graph::node::NodeRecord;
    use coordinode_core::schema::definition::LabelKind;

    let breaks = |bytes: &[u8]| -> StorageResult<bool> {
        let labels = NodeRecord::labels_from_msgpack(bytes)
            .map_err(|e| crate::error::StorageError::Serialization(format!("node labels: {e}")))?;
        if labels.first().map(String::as_str) != Some(label) {
            return Ok(false);
        }
        Ok(match kind {
            LabelKind::Auxiliary => true,
            LabelKind::Type => implied.iter().any(|l| !labels.contains(l)),
        })
    };

    for guard in engine.prefix_scan(Partition::Node, NODE_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        if !is_node_record_key(&key) || staged_points.contains_key(&(Partition::Node, key.to_vec()))
        {
            continue;
        }
        if breaks(&value)? {
            return Ok(Some(node_of_record_key(&key)));
        }
    }
    for ((part, key), value) in staged_points.iter() {
        if *part != Partition::Node || !is_node_record_key(key) {
            continue;
        }
        if let Some(bytes) = value {
            if breaks(bytes)? {
                return Ok(Some(node_of_record_key(key)));
            }
        }
    }
    Ok(None)
}

/// For each of `schemas` (one per label), the first stored node whose
/// primary label is that label and that breaks it, in one pass over the
/// stored nodes; labels every node satisfies are absent from the answer.
/// The pass stops early once every label has a violation.
///
/// # Errors
///
/// A stored record or the field dictionary does not decode.
pub fn label_schema_violations(
    engine: &StorageEngine,
    schemas: &[coordinode_core::schema::definition::LabelSchema],
) -> StorageResult<Vec<(String, LabelSchemaViolation)>> {
    use coordinode_core::graph::node::NodeRecord;

    // Only definitions that can refuse a node are read for.
    let mut pending: rustc_hash::FxHashMap<
        &str,
        &coordinode_core::schema::definition::LabelSchema,
    > = schemas
        .iter()
        .filter(|s| s.can_refuse_nodes())
        .map(|s| (s.name.as_str(), s))
        .collect();
    let mut found = Vec::new();
    if pending.is_empty() {
        return Ok(found);
    }
    let dictionary = crate::engine::metadata::load_field_dictionary(engine)?;
    let names: std::collections::HashMap<u32, String> = dictionary
        .iter()
        .map(|(name, id)| (id, name.to_string()))
        .collect();
    let nothing_staged = std::collections::HashMap::new();
    let mut closures = Closures::new(engine, &nothing_staged);
    for guard in engine.prefix_scan(Partition::Node, NODE_KEY_PREFIX)? {
        let (key, value) = guard.into_inner()?;
        if !is_node_record_key(&key) {
            continue;
        }
        let record = NodeRecord::from_msgpack(&value)
            .map_err(|e| crate::error::StorageError::Serialization(format!("node record: {e}")))?;
        // Each pending label the node carries, its type's implied ones
        // included, judges it.
        let mut carried: Vec<&coordinode_core::schema::definition::LabelSchema> = Vec::new();
        for (name, schema) in &pending {
            if closures.carries(&record.labels, name)? {
                carried.push(schema);
            }
        }
        for schema in carried {
            let Some(reason) =
                carrier_violation(schema, record.clone(), true, &names, dictionary.frontier())
            else {
                continue;
            };
            found.push((
                schema.name.clone(),
                LabelSchemaViolation {
                    node: node_of_record_key(&key),
                    reason,
                },
            ));
            pending.remove(schema.name.as_str());
        }
        if pending.is_empty() {
            break;
        }
    }
    Ok(found)
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

/// What `record` breaks in `schema`: the whole definition when the label is
/// the node's type, only the explicit constraints declared on the label when
/// the node carries it implied or auxiliary (`carried`, which the caller
/// judges with what the type implies). `None` for a node that does not carry
/// the label or breaks nothing.
fn carrier_violation(
    schema: &coordinode_core::schema::definition::LabelSchema,
    record: coordinode_core::graph::node::NodeRecord,
    carried: bool,
    names: &std::collections::HashMap<u32, String>,
    frontier: u32,
) -> Option<String> {
    if record.labels.first() == Some(&schema.name) {
        return record_violation(schema, record, names, frontier);
    }
    if !carried {
        return None;
    }
    let value_of = |name: &str| {
        names
            .iter()
            .find(|(_, n)| n.as_str() == name)
            .and_then(|(id, _)| record.props.get(id))
            .or_else(|| record.get_extra(name))
            .cloned()
    };
    coordinode_core::schema::validation::check_node_constraints(schema, &value_of)
        .err()
        .map(|e| e.to_string())
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
    staged_points: StagedPoints<'_>,
    read_ts: u64,
) -> StorageResult<Verdict> {
    for edge_type in edge_types {
        for direction in [Direction::Outgoing, Direction::Incoming] {
            let adj_key = match direction {
                Direction::Outgoing => encode_adj_key_forward(edge_type, node),
                Direction::Incoming => encode_adj_key_reverse(edge_type, node),
            };
            if engine.written_since_snapshot(Partition::Adj, &adj_key, read_ts)?
                && !crate::engine::cardinality::post_posting(
                    engine,
                    &adj_key,
                    staged,
                    staged_points,
                )?
                .is_empty()
            {
                return Ok(Verdict::Broken);
            }
        }
    }
    Ok(Verdict::Holds)
}

/// Whether the blob content whose catalog entry is `catalog_key` has no
/// owner referencing it and no upload holding it, as committed now.
///
/// A count below zero is a broken invariant, and an entry key that names no
/// content is not a blob claim: neither decides the retirement either way.
fn blob_content_unheld(engine: &StorageEngine, catalog_key: &[u8]) -> StorageResult<Verdict> {
    use coordinode_core::graph::blob::{
        CATALOG_PREFIX, ScopedContent, pins_counter_key, refs_counter_key,
    };

    let Some(content) = catalog_key
        .strip_prefix(CATALOG_PREFIX)
        .and_then(ScopedContent::decode)
    else {
        return Ok(Verdict::Undecidable);
    };
    for key in [refs_counter_key(&content), pins_counter_key(&content)] {
        let count = match engine.get(Partition::Counter, &key)? {
            Some(bytes) => crate::engine::merge::decode_counter(&bytes).map_err(|_| {
                crate::error::StorageError::Serialization(format!(
                    "blob count {}: holds {} bytes, expected 8",
                    crate::error::printable_key(&key),
                    bytes.len()
                ))
            })?,
            None => 0,
        };
        if count < 0 {
            return Ok(Verdict::Undecidable);
        }
        if count > 0 {
            return Ok(Verdict::Broken);
        }
    }
    Ok(Verdict::Holds)
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests;
