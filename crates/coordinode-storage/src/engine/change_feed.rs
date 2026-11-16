//! The change feed's logical projection of one committed unit: the
//! application effects its row changes and effect context state, numbered in
//! one canonical order before any reader filters them.
//!
//! A unit is projected from what it carries and nothing else: its rows and
//! the context its writers sealed with them ([`UnitEffects`]). No row is read
//! from storage, so an event reads the same after the rows it describes were
//! collected, and every reader of one unit sees the same events under the
//! same ordinals.
//!
//! The vocabulary is put, patch and remove over typed targets. A put states
//! a replacement of its target, not that the target was created; a remove
//! states its absence, not that it was seen. Relationship presence follows
//! the forward adjacency changes of the unit and the removals its context
//! names; reverse adjacency, node history, index entries, counters and the
//! derived vector and blob copies are physical forms of those effects and
//! project to nothing.

use std::borrow::Cow;

use coordinode_core::schema::label_id::LabelId;
use coordinode_core::txn::proposal::{
    EFFECTS_INTERPRETATION, Mutation, NodeBasis, PartitionId, UnitEffects,
};
use rustc_hash::FxHashSet;

/// What an event does to its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// The target is replaced by the payload, or comes to exist.
    Put,
    /// The target changes by the payload's delta.
    Patch,
    /// The target ceases to exist.
    Remove,
}

/// What an event changes. A key is the unit's own bytes, or owned when the
/// unit names it only as part of a run of point deletes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target<'u> {
    /// A node's current record, or one version of a temporal node: its key in
    /// the node partition.
    Node(Cow<'u, [u8]>),
    /// The presence of a relationship: its type's name and its endpoints.
    Relationship {
        /// The relationship type's name.
        edge_type: &'u [u8],
        /// The source node's id.
        source: u64,
        /// The target node's id.
        target: u64,
    },
    /// The properties of a relationship that is neither versioned nor
    /// discriminated: its edge property key.
    RelationshipProperties(Cow<'u, [u8]>),
    /// One version or one discriminated instance of a relationship, its
    /// properties included: its edge property key.
    RelationshipInstance(Cow<'u, [u8]>),
}

/// The labels a node stored before and after its event: what routes the
/// event to subscribers of a label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routing<'u> {
    /// The labels it stored before the unit; none when it did not exist.
    pub before: Cow<'u, [LabelId]>,
    /// The labels it stores after the event; none once it is removed.
    pub after: Cow<'u, [LabelId]>,
}

/// How an event reaches a reader interested in some labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delivery {
    /// Neither before nor after: the reader gets no data event.
    None,
    /// The node joins the reader's scope with the record the event writes.
    Enter,
    /// The node stays in scope and the event applies.
    Apply,
    /// The node leaves the reader's scope: a removal there.
    Leave,
}

impl Routing<'_> {
    /// How the event reaches a reader whose scope holds the labels for which
    /// `in_scope` answers yes. One answer per event, whatever number of the
    /// reader's labels match.
    pub fn delivery(&self, mut in_scope: impl FnMut(LabelId) -> bool) -> Delivery {
        let before = self.before.iter().any(|&l| in_scope(l));
        let after = self.after.iter().any(|&l| in_scope(l));
        match (before, after) {
            (false, false) => Delivery::None,
            (false, true) => Delivery::Enter,
            (true, true) => Delivery::Apply,
            (true, false) => Delivery::Leave,
        }
    }
}

/// One logical effect of a committed unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalEvent<'u> {
    /// Its position in the unit's projection, counted before any reader
    /// filters it out.
    pub ordinal: u32,
    /// What it does.
    pub action: Action,
    /// What it does it to.
    pub target: Target<'u>,
    /// The replacement for a put of a node or of relationship properties, the
    /// delta for a patch: the unit's own bytes. None for a removal or a
    /// relationship's presence.
    pub payload: Option<&'u [u8]>,
    /// For a node, the labels that route it.
    pub routing: Option<Routing<'u>>,
}

/// Why a unit has no projection under this build.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProjectionError {
    /// The unit carries no interpretation this build projects: it was sealed
    /// without an effect context, or under another interpretation, or it
    /// removes a key range that names no logical targets.
    #[error("the unit has no projection under this build: {0}")]
    Unsupported(String),
    /// The unit contradicts its own context: a node removal without the
    /// labels that route it, or a key or operand of no known shape.
    #[error("the unit's changes do not project: {0}")]
    Malformed(String),
}

/// Most keys one run of point deletes may stand for: no unit holds more
/// operations than a frame does.
const MAX_RUN_KEYS: u64 = coordinode_core::txn::frame::DecodeLimits::DEFAULT.max_ops as u64;

/// Project the operations of one committed unit, as a reader of rows
/// decodes them, into its logical events in canonical order.
///
/// # Errors
///
/// [`ProjectionError::Unsupported`] for a unit changing application data
/// without an effect context this build reads, or removing a range that is
/// not a run of point deletes; [`ProjectionError::Malformed`] for one that
/// contradicts its context.
pub fn project_unit(mutations: &[Mutation]) -> Result<Vec<LogicalEvent<'_>>, ProjectionError> {
    let effects = mutations.iter().find_map(|m| match m {
        Mutation::Effects(effects) => Some(effects),
        _ => None,
    });
    let Some(effects) = effects else {
        if mutations.iter().any(changes_application_data) {
            return Err(ProjectionError::Unsupported(
                "it changes application data and carries no effect context".into(),
            ));
        }
        return Ok(Vec::new());
    };
    if effects.interpretation != EFFECTS_INTERPRETATION {
        return Err(ProjectionError::Unsupported(format!(
            "its effect context is of interpretation {}, this build reads {}",
            effects.interpretation, EFFECTS_INTERPRETATION
        )));
    }
    Projector::new(mutations, effects).run()
}

/// Whether `m` changes data the feed projects or must account for.
fn changes_application_data(m: &Mutation) -> bool {
    match m {
        Mutation::Put { partition, .. }
        | Mutation::Delete { partition, .. }
        | Mutation::Merge { partition, .. }
        | Mutation::RemoveRange { partition, .. } => matches!(
            partition,
            PartitionId::Node
                | PartitionId::Adj
                | PartitionId::EdgeProp
                | PartitionId::Blob
                | PartitionId::BlobRef
                | PartitionId::VectorF32
                | PartitionId::NodeHistory
        ),
        Mutation::Command(_)
        | Mutation::Derive(_)
        | Mutation::Operation(_)
        | Mutation::Effects(_) => false,
    }
}

/// A relationship's identity: type name, source, target.
type Pair<'u> = (&'u [u8], u64, u64);

struct Projector<'u> {
    mutations: &'u [Mutation],
    effects: &'u UnitEffects,
    /// Pairs whose last forward adjacency change in the unit removes them,
    /// with the pairs the context names removed: their plain edge property
    /// tombstone is part of the removal, not a clearing of properties.
    removed: FxHashSet<Pair<'u>>,
    /// Pairs some forward adjacency change of the unit names: a removal the
    /// context also names is already an event.
    named: FxHashSet<Pair<'u>>,
    events: Vec<LogicalEvent<'u>>,
}

impl<'u> Projector<'u> {
    fn new(mutations: &'u [Mutation], effects: &'u UnitEffects) -> Self {
        Self {
            mutations,
            effects,
            removed: FxHashSet::default(),
            named: FxHashSet::default(),
            events: Vec::new(),
        }
    }

    fn run(mut self) -> Result<Vec<LogicalEvent<'u>>, ProjectionError> {
        self.collect_presence()?;
        let mut runs = self.effects.runs.iter().copied().peekable();
        let mut row = 0u32;
        for mutation in self.mutations {
            let is_row = !matches!(
                mutation,
                Mutation::Command(_)
                    | Mutation::Derive(_)
                    | Mutation::Operation(_)
                    | Mutation::Effects(_)
            );
            let is_run = is_row && runs.next_if_eq(&row).is_some();
            match mutation {
                Mutation::Put {
                    partition,
                    key,
                    value,
                } => self.put(*partition, key, value)?,
                Mutation::Delete { partition, key } => {
                    self.delete(*partition, Cow::Borrowed(key))?;
                }
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => self.merge(*partition, key, operand)?,
                Mutation::RemoveRange {
                    partition,
                    start,
                    end,
                } => self.remove_range(*partition, start, end, is_run)?,
                Mutation::Effects(effects) => self.removed_with_lists(effects)?,
                Mutation::Command(_) | Mutation::Derive(_) | Mutation::Operation(_) => {}
            }
            if is_row {
                row = row.checked_add(1).ok_or_else(|| {
                    ProjectionError::Malformed("the unit holds over 2^32 row changes".into())
                })?;
            }
        }
        Ok(self.events)
    }

    /// The pairs the unit's forward adjacency changes name, and those whose
    /// last change removes them, read before any event so a property
    /// tombstone that comes before its pair's removal is classified right.
    fn collect_presence(&mut self) -> Result<(), ProjectionError> {
        for mutation in self.mutations {
            let Mutation::Merge {
                partition: PartitionId::Adj,
                key,
                operand,
            } = mutation
            else {
                continue;
            };
            let Some((edge_type, source)) = forward_adj_key(key) else {
                continue;
            };
            for (present, target) in adj_operand(operand)? {
                let pair = (edge_type, source, target);
                self.named.insert(pair);
                if present {
                    self.removed.remove(&pair);
                } else {
                    self.removed.insert(pair);
                }
            }
        }
        for key in &self.effects.removed_edges {
            let (pair, _) = edgeprop_key(key).ok_or_else(|| {
                ProjectionError::Malformed("a removed relationship has no edge property key".into())
            })?;
            self.removed.insert(pair);
        }
        Ok(())
    }

    fn emit(
        &mut self,
        action: Action,
        target: Target<'u>,
        payload: Option<&'u [u8]>,
        routing: Option<Routing<'u>>,
    ) -> Result<(), ProjectionError> {
        let ordinal = u32::try_from(self.events.len())
            .map_err(|_| ProjectionError::Malformed("the unit projects over 2^32 events".into()))?;
        self.events.push(LogicalEvent {
            ordinal,
            action,
            target,
            payload,
            routing,
        });
        Ok(())
    }

    /// The labels node `key` stored before the unit, as its context records.
    fn basis(&self, key: &[u8]) -> Option<&'u NodeBasis> {
        let nodes = &self.effects.nodes;
        nodes
            .binary_search_by(|basis| basis.key.as_slice().cmp(key))
            .ok()
            .map(|at| &nodes[at])
    }

    /// The basis of a node whose change cannot state its own labels.
    fn required_basis(&self, key: &[u8], what: &str) -> Result<&'u NodeBasis, ProjectionError> {
        self.basis(key).ok_or_else(|| {
            ProjectionError::Malformed(format!(
                "a node {what} carries no labels it stored before the unit"
            ))
        })
    }

    fn put(
        &mut self,
        partition: PartitionId,
        key: &'u [u8],
        value: &'u [u8],
    ) -> Result<(), ProjectionError> {
        match partition {
            PartitionId::Node => {
                let after = coordinode_core::graph::node::NodeRecord::labels_from_msgpack(value)
                    .map_err(|e| ProjectionError::Malformed(format!("a node record: {e}")))?;
                let before = match self.basis(key) {
                    Some(basis) => Cow::Borrowed(basis.labels.as_slice()),
                    // Its labels are unchanged: the record states both.
                    None => Cow::Owned(after.clone()),
                };
                let routing = Routing {
                    before,
                    after: Cow::Owned(after),
                };
                let target = Target::Node(Cow::Borrowed(key));
                self.emit(Action::Put, target, Some(value), Some(routing))
            }
            PartitionId::Adj => {
                // A whole list written at once, as a restore writes it.
                let Some((edge_type, source)) = forward_adj_key(key) else {
                    return Ok(());
                };
                let list = coordinode_core::graph::edge::PostingList::from_bytes(value)
                    .map_err(|e| ProjectionError::Malformed(format!("a posting list: {e}")))?;
                for target in list.iter() {
                    self.emit(
                        Action::Put,
                        Target::Relationship {
                            edge_type,
                            source,
                            target,
                        },
                        None,
                        None,
                    )?;
                }
                Ok(())
            }
            PartitionId::EdgeProp => {
                let target = edge_target(Cow::Borrowed(key))?;
                self.emit(Action::Put, target, Some(value), None)
            }
            _ => Ok(()),
        }
    }

    fn delete(
        &mut self,
        partition: PartitionId,
        key: Cow<'u, [u8]>,
    ) -> Result<(), ProjectionError> {
        match partition {
            PartitionId::Node => {
                let basis = self.required_basis(&key, "removal")?;
                let routing = Routing {
                    before: Cow::Borrowed(basis.labels.as_slice()),
                    after: Cow::Borrowed(&[]),
                };
                self.emit(Action::Remove, Target::Node(key), None, Some(routing))
            }
            PartitionId::EdgeProp => {
                let part_of_removal = edgeprop_key(&key)
                    .is_some_and(|(pair, suffix)| suffix.is_none() && self.removed.contains(&pair));
                if part_of_removal {
                    // The pair's property record goes with the relationship.
                    return Ok(());
                }
                let target = edge_target(key)?;
                self.emit(Action::Remove, target, None, None)
            }
            // A whole adjacency list goes with its node: the relationships it
            // held are named by forward removals or the context.
            _ => Ok(()),
        }
    }

    fn merge(
        &mut self,
        partition: PartitionId,
        key: &'u [u8],
        operand: &'u [u8],
    ) -> Result<(), ProjectionError> {
        match partition {
            PartitionId::Node => {
                let basis = self.required_basis(key, "document delta")?;
                let labels = basis.labels.as_slice();
                let routing = Routing {
                    before: Cow::Borrowed(labels),
                    after: Cow::Borrowed(labels),
                };
                self.emit(
                    Action::Patch,
                    Target::Node(Cow::Borrowed(key)),
                    Some(operand),
                    Some(routing),
                )
            }
            PartitionId::Adj => {
                let Some((edge_type, source)) = forward_adj_key(key) else {
                    return Ok(());
                };
                for (present, target) in adj_operand(operand)? {
                    let action = if present { Action::Put } else { Action::Remove };
                    self.emit(
                        action,
                        Target::Relationship {
                            edge_type,
                            source,
                            target,
                        },
                        None,
                        None,
                    )?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn remove_range(
        &mut self,
        partition: PartitionId,
        start: &'u [u8],
        end: &'u [u8],
        is_run: bool,
    ) -> Result<(), ProjectionError> {
        let projected = matches!(
            partition,
            PartitionId::Node | PartitionId::EdgeProp | PartitionId::Adj
        );
        if !projected {
            return Ok(());
        }
        if !is_run {
            return Err(ProjectionError::Unsupported(
                "it removes a key range that names no logical targets".into(),
            ));
        }
        if partition == PartitionId::Adj {
            // Adjacency lists, as their nodes go: named elsewhere.
            return Ok(());
        }
        // The run's keys are every key of the start's length up to the end,
        // each projected as its own delete: the unit holds them only as the
        // run, so each event owns its key.
        for key in run_keys(start, end)? {
            self.delete(partition, Cow::Owned(key))?;
        }
        Ok(())
    }

    /// The relationships removed with whole forward lists, which no forward
    /// change of the unit names.
    fn removed_with_lists(&mut self, effects: &'u UnitEffects) -> Result<(), ProjectionError> {
        for key in &effects.removed_edges {
            let Some(((edge_type, source, target), _)) = edgeprop_key(key) else {
                continue;
            };
            if self.named.contains(&(edge_type, source, target)) {
                continue;
            }
            self.emit(
                Action::Remove,
                Target::Relationship {
                    edge_type,
                    source,
                    target,
                },
                None,
                None,
            )?;
        }
        Ok(())
    }
}

/// The type name and source of a forward adjacency key
/// `adj:<type>:out:<source BE>`; `None` for any other key.
fn forward_adj_key(key: &[u8]) -> Option<(&[u8], u64)> {
    let rest = key.strip_prefix(b"adj:")?;
    let (head, source) = rest.split_at_checked(rest.len().checked_sub(8)?)?;
    let edge_type = head.strip_suffix(b":out:")?;
    Some((edge_type, u64::from_be_bytes(source.try_into().ok()?)))
}

/// An edge property key `edgeprop:<type>:<source BE>:<target BE>` with its
/// optional `:<suffix>`: the pair, and the suffix when there is one. Type
/// names carry no ':', so the type ends at the first one.
fn edgeprop_key(key: &[u8]) -> Option<(Pair<'_>, Option<&[u8]>)> {
    let rest = key.strip_prefix(b"edgeprop:")?;
    let colon = rest.iter().position(|&b| b == b':')?;
    let (edge_type, rest) = (&rest[..colon], &rest[colon + 1..]);
    if rest.len() < 17 || rest[8] != b':' {
        return None;
    }
    let source = u64::from_be_bytes(rest[..8].try_into().ok()?);
    let target = u64::from_be_bytes(rest[9..17].try_into().ok()?);
    let suffix = match &rest[17..] {
        [] => None,
        [b':', suffix @ ..] if !suffix.is_empty() => Some(suffix),
        _ => return None,
    };
    Some(((edge_type, source, target), suffix))
}

/// The relationship target an edge property key names.
fn edge_target(key: Cow<'_, [u8]>) -> Result<Target<'_>, ProjectionError> {
    match edgeprop_key(&key) {
        Some((_, None)) => Ok(Target::RelationshipProperties(key)),
        Some((_, Some(_))) => Ok(Target::RelationshipInstance(key)),
        None => Err(ProjectionError::Malformed(
            "an edge property key of no known shape".into(),
        )),
    }
}

/// The members an adjacency operand adds (`true`) or removes (`false`), in
/// order.
fn adj_operand(operand: &[u8]) -> Result<Vec<(bool, u64)>, ProjectionError> {
    let bad = || ProjectionError::Malformed("an adjacency operand of no known shape".into());
    let uid = |bytes: &[u8]| -> Result<u64, ProjectionError> {
        Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| bad())?))
    };
    match operand.split_first() {
        Some((&crate::engine::merge::TAG_ADD, rest)) => Ok(vec![(true, uid(rest)?)]),
        Some((&crate::engine::merge::TAG_REMOVE, rest)) => Ok(vec![(false, uid(rest)?)]),
        Some((&crate::engine::merge::TAG_ADD_BATCH, rest)) => {
            let (count, uids) = rest.split_at_checked(4).ok_or_else(bad)?;
            let count = u32::from_be_bytes(count.try_into().map_err(|_| bad())?) as usize;
            if uids.len() != count.checked_mul(8).ok_or_else(bad)? {
                return Err(bad());
            }
            let (chunks, _) = uids.as_chunks::<8>();
            Ok(chunks
                .iter()
                .map(|chunk| (true, u64::from_be_bytes(*chunk)))
                .collect())
        }
        _ => Err(bad()),
    }
}

/// Every key of `start`'s length from `start` up to `end`, both of one
/// length: the point deletes a run stands for.
fn run_keys(start: &[u8], end: &[u8]) -> Result<Vec<Vec<u8>>, ProjectionError> {
    let bad = |what: &str| ProjectionError::Malformed(format!("a run of deletes {what}"));
    if start.len() != end.len() || start >= end {
        return Err(bad("with bounds of different lengths or out of order"));
    }
    let mut keys = Vec::new();
    let mut key = start.to_vec();
    while key.as_slice() < end {
        if keys.len() as u64 >= MAX_RUN_KEYS {
            return Err(bad("longer than any unit"));
        }
        keys.push(key.clone());
        // Big-endian increment within the length; a key of all 0xFF bytes
        // has no successor of its length, and nothing of it is below `end`.
        let Some(at) = key.iter().rposition(|&b| b != 0xFF) else {
            break;
        };
        key[at] += 1;
        for byte in &mut key[at + 1..] {
            *byte = 0;
        }
    }
    Ok(keys)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
