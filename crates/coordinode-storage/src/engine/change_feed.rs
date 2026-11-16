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
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Action {
    /// The target is replaced by the payload, or comes to exist.
    Put,
    /// The target changes by the payload's delta.
    Patch,
    /// The target ceases to exist.
    Remove,
    /// What a type implies changed: every node of the type gains or loses
    /// labels without a row of its own changing. A reader keeping state by
    /// label reconciles the scope, or rebuilds it.
    ScopeChange,
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
    /// The labels a type implies, as one activation changed them.
    Scope {
        /// The type label.
        label: LabelId,
        /// The activation, distinct for every change.
        activation: u64,
    },
}

/// The labels a node carried before and after its event, its type's
/// implied ones included: what routes the event to subscribers of a label.
/// For a scope change, the type and what it implied before and after.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Routing<'u> {
    /// The labels it stored before the unit; none when it did not exist.
    pub before: Cow<'u, [LabelId]>,
    /// The labels it stores after the event; none once it is removed.
    pub after: Cow<'u, [LabelId]>,
}

/// How an event reaches a reader interested in some labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// One committed unit of the journal as a change feed reads it: where it is,
/// its own commit timestamp and its operations as a reader of rows decodes
/// them, effect context included.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeUnit {
    /// The journal entry and the unit's place in it.
    pub position: coordinode_core::txn::operation::LogPosition,
    /// The unit's own commit timestamp, not its entry's.
    pub commit_ts: u64,
    /// Whether its operation binding refused it: it took no effect and has
    /// no events.
    pub refused: bool,
    /// Its operations; empty when it was refused.
    pub mutations: Vec<Mutation>,
}

impl ChangeUnit {
    /// The unit's logical events, in canonical order; none for a refused
    /// unit.
    ///
    /// # Errors
    ///
    /// As [`project_unit`].
    pub fn events(&self) -> Result<Vec<LogicalEvent<'_>>, ProjectionError> {
        if self.refused {
            return Ok(Vec::new());
        }
        project_unit(&self.mutations)
    }
}

/// A read of the journal for a change feed: its units in log order, and the
/// position the next read starts at.
#[derive(Debug, Clone, PartialEq)]
pub struct ChangeRead {
    /// The units of the entries read, in log order.
    pub units: Vec<ChangeUnit>,
    /// The position after the last entry read; where the read started when
    /// the journal held nothing more yet.
    pub next: u64,
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

    /// Every label a node storing `stored` carries under the unit's
    /// interpretation: its stored labels and what its type implied when the
    /// unit committed.
    fn carried(&self, stored: Cow<'u, [LabelId]>) -> Cow<'u, [LabelId]> {
        let implications = &self.effects.implications;
        let implied = stored.first().and_then(|primary| {
            implications
                .binary_search_by(|i| i.label.cmp(primary))
                .ok()
                .map(|at| &implications[at].implies)
        });
        match implied {
            None => stored,
            Some(implied) => {
                let mut all = stored.into_owned();
                for &label in implied {
                    if !all.contains(&label) {
                        all.push(label);
                    }
                }
                Cow::Owned(all)
            }
        }
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
                let after = self.carried(Cow::Owned(after));
                let before = match self.basis(key) {
                    Some(basis) => self.carried(Cow::Borrowed(basis.labels.as_slice())),
                    // Its labels are unchanged: the record states both.
                    None => after.clone(),
                };
                let routing = Routing { before, after };
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
                    before: self.carried(Cow::Borrowed(basis.labels.as_slice())),
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
                let labels = self.carried(Cow::Borrowed(basis.labels.as_slice()));
                let routing = Routing {
                    before: labels.clone(),
                    after: labels,
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
        // A change of what a type implies: one control for the whole type,
        // routed to every label it reaches before or after, the type's own
        // included, without a node of it enumerated.
        for change in &effects.scope_changes {
            let with_type = |implied: &[LabelId]| {
                let mut all = Vec::with_capacity(implied.len() + 1);
                all.push(change.label);
                all.extend(implied.iter().copied().filter(|&l| l != change.label));
                Cow::Owned(all)
            };
            self.emit(
                Action::ScopeChange,
                Target::Scope {
                    label: change.label,
                    activation: change.activation,
                },
                None,
                Some(Routing {
                    before: with_type(&change.before),
                    after: with_type(&change.after),
                }),
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

/// The parts of an edge property key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgePropParts<'k> {
    /// The relationship type's name.
    pub edge_type: &'k [u8],
    /// The source node's id.
    pub source: u64,
    /// The target node's id.
    pub target: u64,
    /// The instance's version or discriminator, when the key names one.
    pub instance: Option<&'k [u8]>,
}

/// The parts of an edge property key; `None` for a key of no known shape.
#[must_use]
pub fn edgeprop_parts(key: &[u8]) -> Option<EdgePropParts<'_>> {
    edgeprop_key(key).map(|((edge_type, source, target), instance)| EdgePropParts {
        edge_type,
        source,
        target,
        instance,
    })
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

/// What a reader of the feed is interested in: nodes routed by some labels,
/// relationships of some types. `None` takes every one of that kind; an
/// empty list takes none.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Interest {
    labels: Option<Vec<LabelId>>,
    edge_types: Option<Vec<Vec<u8>>>,
}

impl Interest {
    /// Every event.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Node events routed by any of `labels` before or after them, and
    /// relationship events of any of `edge_types`; `None` for either takes
    /// every event of that kind.
    #[must_use]
    pub fn new(labels: Option<Vec<LabelId>>, edge_types: Option<Vec<Vec<u8>>>) -> Self {
        let labels = labels.map(|mut l| {
            l.sort_unstable();
            l.dedup();
            l
        });
        let edge_types = edge_types.map(|mut t| {
            t.sort_unstable();
            t.dedup();
            t
        });
        Self { labels, edge_types }
    }

    /// How `event` reaches this reader: `None` when it does not.
    #[must_use]
    pub fn delivery(&self, event: &LogicalEvent<'_>) -> Option<Delivery> {
        if let Some(routing) = &event.routing {
            let delivery = match &self.labels {
                None => Delivery::Apply,
                Some(labels) => routing.delivery(|l| labels.binary_search(&l).is_ok()),
            };
            return (delivery != Delivery::None).then_some(delivery);
        }
        let edge_type = match &event.target {
            Target::Relationship { edge_type, .. } => *edge_type,
            Target::RelationshipProperties(key) | Target::RelationshipInstance(key) => {
                edgeprop_key(key).map_or(&[][..], |((edge_type, _, _), _)| edge_type)
            }
            // Routed above: every node and scope event carries its labels.
            Target::Node(_) | Target::Scope { .. } => return Some(Delivery::Apply),
        };
        let wanted = match &self.edge_types {
            None => true,
            Some(types) => types.iter().any(|t| t.as_slice() == edge_type),
        };
        wanted.then_some(Delivery::Apply)
    }

    /// The bytes a cursor binds it by: equal interests encode equally.
    fn encode(&self, out: &mut Vec<u8>) {
        match &self.labels {
            None => out.push(0),
            Some(labels) => {
                out.push(1);
                out.extend_from_slice(&(labels.len() as u32).to_be_bytes());
                for label in labels {
                    out.extend_from_slice(&label.get().to_be_bytes());
                }
            }
        }
        match &self.edge_types {
            None => out.push(0),
            Some(types) => {
                out.push(1);
                out.extend_from_slice(&(types.len() as u32).to_be_bytes());
                for name in types {
                    out.extend_from_slice(&(name.len() as u32).to_be_bytes());
                    out.extend_from_slice(name);
                }
            }
        }
    }
}

/// Why a cursor is not one this reader can continue from.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CursorError {
    /// Bytes no build wrote as a cursor, or of a layout this build does not
    /// read.
    #[error("not a change cursor this build reads: {0}")]
    Malformed(&'static str),
    /// A cursor of another history: its positions count another log.
    #[error("the cursor belongs to another history")]
    OtherHistory,
    /// A cursor made under another projection interpretation.
    #[error(
        "the cursor was made under interpretation {0}, this build reads {EFFECTS_INTERPRETATION}"
    )]
    OtherInterpretation(u32),
}

/// Layout of the cursors this build writes.
const CURSOR_VERSION: u8 = 1;
const HISTORY_JOURNAL: u8 = 0;
const HISTORY_GROUP: u8 = 1;

/// Where a reader of the feed continues: the history it reads, the
/// interpretation and interest it read under, the next journal entry and,
/// inside it, the next unit and event. Opaque to the reader; saved and
/// handed back as bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeCursor {
    history: crate::engine::fixed_cut::CutHistory,
    interest: Interest,
    next: u64,
    /// The first unit of entry `next` not wholly read, and the ordinal of its
    /// first event not read: a read stopped inside a large unit continues
    /// there. Both zero at the start of an entry.
    within: Within,
}

/// Where a read starts or stopped: a journal entry and a place inside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadPlace {
    pub(crate) entry: u64,
    pub(crate) within: Within,
}

/// A place inside one journal entry: a unit and an event ordinal of it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Within {
    pub(crate) sub: u32,
    pub(crate) ordinal: u32,
}

impl ChangeCursor {
    /// A cursor at the start of entry `next` in `history`, for `interest`.
    pub(crate) fn new(
        history: crate::engine::fixed_cut::CutHistory,
        interest: Interest,
        next: u64,
    ) -> Self {
        Self::within(history, interest, next, Within::default())
    }

    /// A cursor inside entry `next`, at `within`.
    pub(crate) fn within(
        history: crate::engine::fixed_cut::CutHistory,
        interest: Interest,
        next: u64,
        within: Within,
    ) -> Self {
        Self {
            history,
            interest,
            next,
            within,
        }
    }

    /// The journal entry the next read starts at: the position a flat
    /// consumer acknowledges. Every entry before it is read whole; a read
    /// stopped inside it has read part of it.
    #[must_use]
    pub fn position(&self) -> u64 {
        self.next
    }

    /// The unit and event ordinal inside [`Self::position`]'s entry the next
    /// read starts at.
    pub(crate) fn within_entry(&self) -> Within {
        self.within
    }

    /// Whether this cursor is at the start of its entry: no part of it read.
    #[must_use]
    pub fn at_entry_start(&self) -> bool {
        self.within == Within::default()
    }

    /// Whether this cursor is at the start of a unit: no part of the unit it
    /// stands at read.
    #[must_use]
    pub fn at_unit_start(&self) -> bool {
        self.within.ordinal == 0
    }

    /// This cursor's history and interest at the start of the unit at
    /// `position`: where a reader that holds everything before that unit
    /// continues.
    #[must_use]
    pub fn before_unit(&self, position: coordinode_core::txn::operation::LogPosition) -> Self {
        Self::within(
            self.history,
            self.interest.clone(),
            position.index,
            Within {
                sub: position.sub,
                ordinal: 0,
            },
        )
    }

    /// The history the cursor reads.
    #[must_use]
    pub fn history(&self) -> crate::engine::fixed_cut::CutHistory {
        self.history
    }

    /// The interest the cursor reads for.
    #[must_use]
    pub fn interest(&self) -> &Interest {
        &self.interest
    }

    /// The cursor's bytes, to save and hand back.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        use crate::engine::fixed_cut::CutHistory;
        let mut out = vec![CURSOR_VERSION];
        match self.history {
            CutHistory::Journal(deployment) => {
                out.push(HISTORY_JOURNAL);
                out.extend_from_slice(&deployment.0);
            }
            CutHistory::Group(history) => {
                out.push(HISTORY_GROUP);
                out.extend_from_slice(&history.deployment.0);
                out.extend_from_slice(&history.group.0.to_be_bytes());
            }
        }
        out.extend_from_slice(&EFFECTS_INTERPRETATION.to_be_bytes());
        out.extend_from_slice(&self.next.to_be_bytes());
        out.extend_from_slice(&self.within.sub.to_be_bytes());
        out.extend_from_slice(&self.within.ordinal.to_be_bytes());
        self.interest.encode(&mut out);
        out
    }

    /// Read back a cursor's bytes.
    ///
    /// # Errors
    ///
    /// [`CursorError::Malformed`] for bytes no cursor of this layout has;
    /// [`CursorError::OtherInterpretation`] for one made under another
    /// interpretation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CursorError> {
        use crate::engine::fixed_cut::CutHistory;
        use coordinode_core::group::{DeploymentId, GroupId};
        let mut r = CursorReader(bytes);
        if r.u8()? != CURSOR_VERSION {
            return Err(CursorError::Malformed("an unknown layout version"));
        }
        let history = match r.u8()? {
            HISTORY_JOURNAL => CutHistory::Journal(DeploymentId(r.array()?)),
            HISTORY_GROUP => {
                let deployment = DeploymentId(r.array()?);
                let group = GroupId(u64::from_be_bytes(r.array()?));
                CutHistory::Group(coordinode_core::txn::read_concern::HistoryId {
                    deployment,
                    group,
                })
            }
            _ => return Err(CursorError::Malformed("an unknown history kind")),
        };
        let interpretation = u32::from_be_bytes(r.array()?);
        if interpretation != EFFECTS_INTERPRETATION {
            return Err(CursorError::OtherInterpretation(interpretation));
        }
        let next = u64::from_be_bytes(r.array()?);
        let within = Within {
            sub: u32::from_be_bytes(r.array()?),
            ordinal: u32::from_be_bytes(r.array()?),
        };
        let labels = match r.u8()? {
            0 => None,
            1 => {
                let count = r.count(4)?;
                let mut labels = Vec::with_capacity(count);
                for _ in 0..count {
                    let raw = u32::from_be_bytes(r.array()?);
                    labels.push(LabelId::new(raw).ok_or(CursorError::Malformed("label id 0"))?);
                }
                Some(labels)
            }
            _ => return Err(CursorError::Malformed("an unknown label interest")),
        };
        let edge_types = match r.u8()? {
            0 => None,
            1 => {
                let count = r.count(4)?;
                let mut types = Vec::with_capacity(count);
                for _ in 0..count {
                    let len = r.count(1)?;
                    types.push(r.take(len)?.to_vec());
                }
                Some(types)
            }
            _ => return Err(CursorError::Malformed("an unknown type interest")),
        };
        if !r.0.is_empty() {
            return Err(CursorError::Malformed("bytes after the cursor"));
        }
        let interest = Interest::new(labels.clone(), edge_types.clone());
        if interest.labels != labels || interest.edge_types != edge_types {
            return Err(CursorError::Malformed("an interest not in canonical order"));
        }
        Ok(Self {
            history,
            interest,
            next,
            within,
        })
    }
}

/// A cursor's bytes, read from the front.
struct CursorReader<'a>(&'a [u8]);

impl<'a> CursorReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], CursorError> {
        let (head, rest) = self
            .0
            .split_at_checked(n)
            .ok_or(CursorError::Malformed("a truncated cursor"))?;
        self.0 = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, CursorError> {
        Ok(self.take(1)?[0])
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CursorError> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N)?);
        Ok(out)
    }

    /// A count of items at least `min` bytes each, refused when the cursor
    /// cannot hold that many.
    fn count(&mut self, min: usize) -> Result<usize, CursorError> {
        let count = u32::from_be_bytes(self.array()?) as usize;
        if count
            .checked_mul(min)
            .is_none_or(|bytes| bytes > self.0.len())
        {
            return Err(CursorError::Malformed("a count the cursor cannot hold"));
        }
        Ok(count)
    }
}

/// An event's identity: the unit it belongs to and its ordinal in the unit's
/// projection. The same in every reader's delivery of it, whatever each
/// filtered out, and in every read of it again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct EventId {
    /// The unit's journal entry and place in it.
    pub position: coordinode_core::txn::operation::LogPosition,
    /// The event's place in the unit's projection.
    pub ordinal: u32,
}

/// One event as a reader receives it: owned, with its identity and how it
/// reaches the reader.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChangeEvent {
    /// Its identity in its history.
    pub id: EventId,
    /// What it does.
    pub action: Action,
    /// What it does it to.
    pub target: OwnedTarget,
    /// The replacement for a put, the delta for a patch.
    #[serde(with = "serde_bytes")]
    pub payload: Option<Vec<u8>>,
    /// How it reaches this reader: enter, apply or leave its scope.
    pub delivery: Delivery,
    /// For a node, the labels it stored before and after.
    pub labels: Option<(Vec<LabelId>, Vec<LabelId>)>,
}

/// [`Target`] with its bytes owned.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OwnedTarget {
    /// A node's record or a temporal node's version: its node key.
    Node(#[serde(with = "serde_bytes")] Vec<u8>),
    /// A relationship's presence.
    Relationship {
        /// The relationship type's name.
        #[serde(with = "serde_bytes")]
        edge_type: Vec<u8>,
        /// The source node's id.
        source: u64,
        /// The target node's id.
        target: u64,
    },
    /// The properties of a relationship: its edge property key.
    RelationshipProperties(#[serde(with = "serde_bytes")] Vec<u8>),
    /// A relationship's version or discriminated instance: its edge property
    /// key.
    RelationshipInstance(#[serde(with = "serde_bytes")] Vec<u8>),
    /// What a type implies, as one activation changed it.
    Scope {
        /// The type label.
        label: LabelId,
        /// The activation, distinct for every change.
        activation: u64,
    },
}

impl From<&Target<'_>> for OwnedTarget {
    fn from(target: &Target<'_>) -> Self {
        match target {
            Target::Node(key) => Self::Node(key.to_vec()),
            Target::Relationship {
                edge_type,
                source,
                target,
            } => Self::Relationship {
                edge_type: edge_type.to_vec(),
                source: *source,
                target: *target,
            },
            Target::RelationshipProperties(key) => Self::RelationshipProperties(key.to_vec()),
            Target::RelationshipInstance(key) => Self::RelationshipInstance(key.to_vec()),
            Target::Scope { label, activation } => Self::Scope {
                label: *label,
                activation: *activation,
            },
        }
    }
}

/// One committed unit, or a fragment of one, as a reader receives it: every
/// event of it in the ordinal range `[first, through)` the reader's interest
/// takes, and the count of events its projection holds, so the reader can
/// tell when it has all of the unit it asked for.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct UnitChanges {
    /// The unit's journal entry and place in it.
    pub position: coordinode_core::txn::operation::LogPosition,
    /// The unit's commit timestamp.
    pub commit_ts: u64,
    /// The events the reader's interest takes, in ordinal order.
    pub events: Vec<ChangeEvent>,
    /// How many events the unit's projection holds, taken or not.
    pub projected: u32,
    /// The first ordinal this delivery covers: zero unless an earlier read
    /// delivered the unit's start.
    pub first: u32,
    /// The ordinal after the last one this delivery covers, taken or not:
    /// `projected` when the delivery reaches the unit's end.
    pub through: u32,
}

impl UnitChanges {
    /// Whether this delivery reaches the end of the unit.
    #[must_use]
    pub fn completes_unit(&self) -> bool {
        self.through == self.projected
    }

    /// Whether this delivery is the whole unit.
    #[must_use]
    pub fn is_whole_unit(&self) -> bool {
        self.first == 0 && self.completes_unit()
    }
}

/// A read of the feed: units in log order, the last one possibly a fragment
/// when the read's memory ran out inside it, and the cursor the next read
/// continues from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBatch {
    /// The units read; a unit none of whose events the reader takes is here
    /// too, with no events. Every unit but the last is whole from where the
    /// read started.
    pub units: Vec<UnitChanges>,
    /// Where the next read continues.
    pub cursor: ChangeCursor,
}

/// Bytes one delivered event holds.
fn event_bytes(event: &LogicalEvent<'_>) -> u64 {
    let target = match &event.target {
        Target::Node(key)
        | Target::RelationshipProperties(key)
        | Target::RelationshipInstance(key) => key.len(),
        Target::Relationship { edge_type, .. } => edge_type.len(),
        Target::Scope { .. } => 0,
    };
    let labels = event.routing.as_ref().map_or(0, |r| {
        (r.before.len() + r.after.len()) * core::mem::size_of::<LabelId>()
    });
    (core::mem::size_of::<ChangeEvent>() + target + event.payload.map_or(0, <[u8]>::len) + labels)
        as u64
}

/// The changes a reader at `from` receives of `units`, the units of the
/// entries read from its entry on: units before `from` inside its entry are
/// passed over, and the unit it stopped in continues at its ordinal. Every
/// event and unit kept is charged to `charge` first; when the charge refuses
/// one, the read stops there, and the place it stopped is returned. A read
/// that cannot keep its first event refuses with the budget's stop.
///
/// # Errors
///
/// [`StorageError::ChangeFeedUnsupported`] for a unit the feed cannot
/// project, [`StorageError::Serialization`] for a malformed one, and
/// [`StorageError::Budget`] as above.
pub(crate) fn unit_changes(
    units: &[ChangeUnit],
    interest: &Interest,
    from: ReadPlace,
    charge: &mut coordinode_core::budget::MemoryCharge<'_>,
) -> Result<(Vec<UnitChanges>, Option<ReadPlace>), crate::error::StorageError> {
    use crate::error::StorageError;
    let projection = |e: ProjectionError| match e {
        ProjectionError::Unsupported(reason) => StorageError::ChangeFeedUnsupported { reason },
        ProjectionError::Malformed(reason) => StorageError::Serialization(reason),
    };
    let ReadPlace {
        entry: from_entry,
        within: from_within,
    } = from;
    let mut out: Vec<UnitChanges> = Vec::new();
    let mut kept_any = false;
    for unit in units {
        let first = if unit.position.index == from_entry {
            if unit.position.sub < from_within.sub {
                continue;
            }
            if unit.position.sub == from_within.sub {
                from_within.ordinal
            } else {
                0
            }
        } else {
            0
        };
        let events = unit.events().map_err(projection)?;
        let projected = u32::try_from(events.len()).map_err(|_| {
            StorageError::Serialization("the unit projects over 2^32 events".into())
        })?;
        let stopped = |ordinal: u32| {
            Some(ReadPlace {
                entry: unit.position.index,
                within: Within {
                    sub: unit.position.sub,
                    ordinal,
                },
            })
        };
        if let Err(stop) = charge.grow(core::mem::size_of::<UnitChanges>() as u64) {
            if !kept_any {
                return Err(StorageError::Budget(stop));
            }
            return Ok((out, stopped(first)));
        }
        let mut taken = Vec::new();
        let mut through = first;
        for event in events.iter().skip(first as usize) {
            if let Some(delivery) = interest.delivery(event) {
                if let Err(stop) = charge.grow(event_bytes(event)) {
                    if !kept_any {
                        return Err(StorageError::Budget(stop));
                    }
                    out.push(UnitChanges {
                        position: unit.position,
                        commit_ts: unit.commit_ts,
                        events: taken,
                        projected,
                        first,
                        through,
                    });
                    return Ok((out, stopped(through)));
                }
                kept_any = true;
                taken.push(ChangeEvent {
                    id: EventId {
                        position: unit.position,
                        ordinal: event.ordinal,
                    },
                    action: event.action,
                    target: OwnedTarget::from(&event.target),
                    payload: event.payload.map(<[u8]>::to_vec),
                    delivery,
                    labels: event
                        .routing
                        .as_ref()
                        .map(|r| (r.before.to_vec(), r.after.to_vec())),
                });
            }
            through = event.ordinal + 1;
        }
        kept_any = true;
        out.push(UnitChanges {
            position: unit.position,
            commit_ts: unit.commit_ts,
            events: taken,
            projected,
            first,
            through: projected,
        });
    }
    Ok((out, None))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
