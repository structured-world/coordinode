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
    /// A label is dropped: its identity ends. Nodes of it removed in the
    /// same transaction are removals of their own; a reader keeping state by
    /// the label drops what is left of the scope.
    Drop,
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
    /// What a blob owner holds: a node's large property, a stored object or
    /// an S3 object, by its owner key. A put carries the owner record (the
    /// blob it now holds), not the blob's bytes.
    BlobOwner(Cow<'u, [u8]>),
    /// A label, by the identity it had.
    Label(LabelId),
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

    /// Hand the unit's events from ordinal `first` on to `take` one at a
    /// time, as [`project_within`] does; none for a refused unit. Returns
    /// how many events the unit's projection holds.
    pub(crate) fn project_with<'u, R: Room>(
        &'u self,
        first: u32,
        room: &mut R,
        take: impl FnMut(LogicalEvent<'u>, &mut R) -> Take,
    ) -> Result<u32, ProjectStop> {
        if self.refused {
            return Ok(0);
        }
        project_within(&self.mutations, first, room, take)
    }
}

/// Whether a consumer of a projection takes the events after the one it
/// was handed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Take {
    /// Hand over the next event too.
    More,
    /// Hand over no more: the projection still counts the rest, so the
    /// unit's whole count is known, without building them for anyone.
    Enough,
}

/// Memory a projection's working state, and the events its consumer keeps,
/// are taken from before they are allocated: a read's budget, a cut's
/// pool, or none.
pub(crate) trait Room {
    /// Take `bytes` before they are allocated; `false` when they do not
    /// fit, nothing taken.
    fn take(&mut self, bytes: u64) -> bool;
    /// Return `bytes` taken earlier, once their memory is freed.
    fn give(&mut self, bytes: u64);
}

/// No bound: a projection whose caller holds what it builds as its own.
pub(crate) struct Unbounded;

impl Room for Unbounded {
    fn take(&mut self, _: u64) -> bool {
        true
    }

    fn give(&mut self, _: u64) {}
}

/// Why a bounded projection stopped.
#[derive(Debug)]
pub(crate) enum ProjectStop {
    /// The unit does not project.
    Projection(ProjectionError),
    /// Its room refused the working state the unit's projection needs
    /// before any event of it could be handed over.
    NoRoom,
}

impl From<ProjectionError> for ProjectStop {
    fn from(e: ProjectionError) -> Self {
        Self::Projection(e)
    }
}

/// Grow `v`'s capacity to hold one more element, its room taken first: the
/// old and the new buffer while the elements move, then the new one only.
/// `false`, nothing grown, when the room refuses it.
pub(crate) fn grow_within<T>(v: &mut Vec<T>, room: &mut impl Room) -> bool {
    if v.len() < v.capacity() {
        return true;
    }
    let slot = core::mem::size_of::<T>() as u64;
    let old = v.capacity();
    let new = old.max(2) * 2;
    if !room.take(new as u64 * slot) {
        return false;
    }
    v.reserve_exact(new - v.len());
    room.give(old as u64 * slot);
    true
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
    let mut events = Vec::new();
    project_unit_with(mutations, |event| {
        events.push(event);
        Take::More
    })?;
    Ok(events)
}

/// [`project_unit`] handing each event to `take` as it is projected, rather
/// than collecting the unit's events: a reader keeps only the events it
/// takes, and a run of deletes is walked without listing its keys. Once
/// `take` answers [`Take::Enough`] no event is built for it; the rest are
/// counted. Returns how many events the unit's projection holds.
///
/// # Errors
///
/// As [`project_unit`].
pub fn project_unit_with<'u>(
    mutations: &'u [Mutation],
    mut take: impl FnMut(LogicalEvent<'u>) -> Take,
) -> Result<u32, ProjectionError> {
    project_within(mutations, 0, &mut Unbounded, |event, _| take(event)).map_err(
        |stop| match stop {
            ProjectStop::Projection(e) => e,
            // An unbounded room refuses nothing.
            ProjectStop::NoRoom => ProjectionError::Unsupported("no room to project it".into()),
        },
    )
}

/// [`project_unit_with`] within `room`: the events before ordinal `first`,
/// handed over by an earlier read, are neither built nor handed again, and
/// once `take` answers [`Take::Enough`] the rest are only counted, neither
/// built nor checked; only what the events handed over need of the unit
/// is kept. The working state the unit's projection needs is taken from
/// `room` before it is allocated and given back once freed; `take` charges
/// what it keeps of each event to the same room.
///
/// # Errors
///
/// [`ProjectStop::Projection`] as [`project_unit`], for the part of the
/// unit read before the consumer had enough; [`ProjectStop::NoRoom`] when
/// `room` refuses the working state.
pub(crate) fn project_within<'u, R: Room>(
    mutations: &'u [Mutation],
    first: u32,
    room: &mut R,
    take: impl FnMut(LogicalEvent<'u>, &mut R) -> Take,
) -> Result<u32, ProjectStop> {
    let effects = mutations.iter().find_map(|m| match m {
        Mutation::Effects(effects) => Some(effects),
        _ => None,
    });
    let Some(effects) = effects else {
        if mutations.iter().any(changes_application_data) {
            return Err(ProjectionError::Unsupported(
                "it changes application data and carries no effect context".into(),
            )
            .into());
        }
        return Ok(0);
    };
    if effects.interpretation != EFFECTS_INTERPRETATION {
        return Err(ProjectionError::Unsupported(format!(
            "its effect context is of interpretation {}, this build reads {}",
            effects.interpretation, EFFECTS_INTERPRETATION
        ))
        .into());
    }
    Projector::new(mutations, effects, first, room, take).run()
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

/// What a change outside nodes and relationships is.
enum Component {
    /// A blob owner's record: readers receive it.
    BlobOwner,
    /// A record kept for the engine's own work.
    Internal,
}

/// What the next event's ordinal asks of the projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    /// Handed over by an earlier read: not built; only what a later change
    /// of the same node needs of it is kept.
    Delivered,
    /// Built and handed to the consumer.
    Build,
    /// Past what the consumer takes: only counted.
    Counted,
}

/// One change of a relationship's presence the unit makes, or a removal
/// its context names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct PairChange<'u> {
    pair: Pair<'u>,
    /// Its place among the unit's forward adjacency changes; [`u32::MAX`]
    /// for a removal the context names, which no change overrides.
    seq: u32,
    present: bool,
}

/// A run of node deletes of the unit.
#[derive(Debug, Clone, Copy)]
struct NodeRun<'u> {
    start: &'u [u8],
    end: &'u [u8],
    /// The run's place among the unit's operations.
    at: u32,
    /// The greatest end of this run and every run ordered before it: a
    /// search for the runs holding a key stops where it falls below it.
    reach: &'u [u8],
}

/// Bytes a hash table of `capacity` entries of `T` allocates at most.
fn table_bytes<T>(capacity: usize) -> u64 {
    if capacity == 0 {
        return 0;
    }
    let buckets = match capacity {
        0..4 => 4,
        4..8 => 8,
        _ => (capacity * 8 / 7).next_power_of_two(),
    };
    // Its slots, its control bytes and one group of them past the end.
    ((buckets * core::mem::size_of::<T>()).next_multiple_of(16) + buckets + 16) as u64
}

/// The labels a node changed again later in the unit carries after its
/// last change so far.
type Left<'u> = rustc_hash::FxHashMap<Cow<'u, [u8]>, Vec<LabelId>>;

struct Projector<'u, 'r, R, F> {
    mutations: &'u [Mutation],
    effects: &'u UnitEffects,
    /// Every change of a relationship's presence the unit makes and every
    /// removal its context names, by pair and then in unit order: whether
    /// a pair is named, and whether its last change removes it, so its
    /// plain edge property tombstone is part of the removal.
    pairs: Vec<PairChange<'u>>,
    /// The key of every node a put, delete or merge of the unit changes,
    /// with that operation's place, by key and then place: whether a node
    /// changes again later, and which versions the unit writes.
    nodes: Vec<(&'u [u8], u32)>,
    /// The unit's runs of node deletes, by start.
    runs: Vec<NodeRun<'u>>,
    /// What each node changed again later carries after its change so far:
    /// the later change starts from it, not from what the node had before
    /// the unit. Only such nodes are kept.
    left: Left<'u>,
    /// The room `left`'s table was given.
    table: u64,
    /// The most the label lists of one node event take while it is built:
    /// what its record decodes to and the lists that route it.
    routing: u64,
    /// Bytes of working state taken from the room.
    held: u64,
    /// The operation being projected.
    at: u32,
    /// The first ordinal to build: those before it were handed over.
    first: u32,
    room: &'r mut R,
    /// The consumer each event is handed to as it is built.
    take: F,
    /// Whether the consumer takes more events.
    taking: bool,
    /// Events projected so far: the next one's ordinal.
    count: u32,
}

/// Hand `each` every operation of `mutations` with its place and whether
/// it is one of the `runs` of point deletes.
fn for_each_operation<'u>(
    mutations: &'u [Mutation],
    runs: &[u32],
    mut each: impl FnMut(u32, &'u Mutation, bool) -> Result<(), ProjectStop>,
) -> Result<(), ProjectStop> {
    let mut runs = runs.iter().copied().peekable();
    let mut row = 0u32;
    for (at, mutation) in mutations.iter().enumerate() {
        let at = u32::try_from(at).map_err(|_| {
            ProjectionError::Malformed("the unit holds over 2^32 operations".into())
        })?;
        let is_row = !matches!(
            mutation,
            Mutation::Command(_)
                | Mutation::Derive(_)
                | Mutation::Operation(_)
                | Mutation::Effects(_)
        );
        let is_run = is_row && runs.next_if_eq(&row).is_some();
        each(at, mutation, is_run)?;
        if is_row {
            row = row.checked_add(1).ok_or_else(|| {
                ProjectionError::Malformed("the unit holds over 2^32 row changes".into())
            })?;
        }
    }
    Ok(())
}

impl<'u, 'r, R: Room, F: FnMut(LogicalEvent<'u>, &mut R) -> Take> Projector<'u, 'r, R, F> {
    fn new(
        mutations: &'u [Mutation],
        effects: &'u UnitEffects,
        first: u32,
        room: &'r mut R,
        take: F,
    ) -> Self {
        Self {
            mutations,
            effects,
            pairs: Vec::new(),
            nodes: Vec::new(),
            runs: Vec::new(),
            left: Left::default(),
            table: 0,
            routing: 0,
            held: 0,
            at: 0,
            first,
            room,
            take,
            taking: true,
            count: 0,
        }
    }

    fn run(mut self) -> Result<u32, ProjectStop> {
        let walked = self.prepare().and_then(|()| self.walk());
        let count = self.count;
        // The working state is freed before its room is given back.
        let Self {
            pairs,
            nodes,
            runs,
            left,
            held,
            room,
            ..
        } = self;
        drop((pairs, nodes, runs, left));
        room.give(held);
        walked.map(|()| count)
    }

    /// Take `bytes` of working state from the room.
    fn lend(&mut self, bytes: u64) -> Result<(), ProjectStop> {
        if !self.room.take(bytes) {
            return Err(ProjectStop::NoRoom);
        }
        self.held += bytes;
        Ok(())
    }

    /// Give back `bytes` of working state, freed.
    fn repay(&mut self, bytes: u64) {
        self.held -= bytes;
        self.room.give(bytes);
    }

    /// The unit's node changes, runs and relationship presence changes,
    /// sorted to be searched, read before any event so a property tombstone
    /// before its pair's removal, and a node's change before a later one of
    /// it, are read right. Counted first, their room taken, then built at
    /// that size.
    fn prepare(&mut self) -> Result<(), ProjectStop> {
        let (mutations, run_rows) = (self.mutations, self.effects.runs.as_slice());
        let (mut nodes, mut runs) = (0usize, 0usize);
        let mut pairs = self.effects.removed_edges.len();
        for_each_operation(mutations, run_rows, |_, mutation, is_run| {
            match mutation {
                Mutation::Put {
                    partition: PartitionId::Node,
                    ..
                }
                | Mutation::Delete {
                    partition: PartitionId::Node,
                    ..
                }
                | Mutation::Merge {
                    partition: PartitionId::Node,
                    ..
                } => nodes += 1,
                Mutation::RemoveRange {
                    partition: PartitionId::Node,
                    ..
                } if is_run => runs += 1,
                Mutation::Merge {
                    partition: PartitionId::Adj,
                    key,
                    operand,
                } if forward_adj_key(key).is_some() => pairs += adj_operand(operand)?.len(),
                _ => {}
            }
            Ok(())
        })?;
        let bytes = nodes * core::mem::size_of::<(&[u8], u32)>()
            + runs * core::mem::size_of::<NodeRun<'_>>()
            + pairs * core::mem::size_of::<PairChange<'_>>();
        self.lend(bytes as u64)?;
        self.nodes = Vec::with_capacity(nodes);
        self.runs = Vec::with_capacity(runs);
        self.pairs = Vec::with_capacity(pairs);
        let mut seq = 0u32;
        for_each_operation(mutations, run_rows, |at, mutation, is_run| {
            match mutation {
                Mutation::Put {
                    partition: PartitionId::Node,
                    key,
                    ..
                }
                | Mutation::Delete {
                    partition: PartitionId::Node,
                    key,
                }
                | Mutation::Merge {
                    partition: PartitionId::Node,
                    key,
                    ..
                } => self.nodes.push((key, at)),
                Mutation::RemoveRange {
                    partition: PartitionId::Node,
                    start,
                    end,
                } if is_run => self.runs.push(NodeRun {
                    start,
                    end,
                    at,
                    reach: end,
                }),
                Mutation::Merge {
                    partition: PartitionId::Adj,
                    key,
                    operand,
                } => {
                    if let Some((edge_type, source)) = forward_adj_key(key) {
                        for (present, target) in adj_operand(operand)? {
                            self.pairs.push(PairChange {
                                pair: (edge_type, source, target),
                                seq,
                                present,
                            });
                            seq = seq.checked_add(1).ok_or_else(|| {
                                ProjectionError::Malformed(
                                    "the unit changes over 2^32 relationships".into(),
                                )
                            })?;
                        }
                    }
                }
                _ => {}
            }
            Ok(())
        })?;
        for key in &self.effects.removed_edges {
            let (pair, _) = edgeprop_key(key).ok_or_else(|| {
                ProjectionError::Malformed("a removed relationship has no edge property key".into())
            })?;
            self.pairs.push(PairChange {
                pair,
                seq: u32::MAX,
                present: false,
            });
        }
        self.pairs.sort_unstable();
        self.nodes.sort_unstable();
        self.runs.sort_unstable_by(|a, b| a.start.cmp(b.start));
        let mut reach: &[u8] = &[];
        for run in &mut self.runs {
            reach = reach.max(run.end);
            run.reach = reach;
        }
        // A node changed again later is one named twice, or named and in a
        // run: at most this many are kept at once.
        let again = (0..self.nodes.len())
            .filter(|&i| {
                let key = self.nodes[i].0;
                (i > 0 && self.nodes[i - 1].0 == key)
                    || self.nodes.get(i + 1).is_some_and(|n| n.0 == key)
                    || self.in_run_after(key, None)
            })
            .count();
        self.table = table_bytes::<(Cow<'u, [u8]>, Vec<LabelId>)>(again);
        self.lend(self.table)?;
        self.left = Left::with_capacity_and_hasher(again, Default::default());
        // A record stores at most a label per byte; a node carries at most
        // the most labels a record or a basis of the unit stores, and what
        // its type implies. Its event decodes one record and builds two
        // lists, a delta's two copies of one.
        let stored = mutations
            .iter()
            .filter_map(|m| match m {
                Mutation::Put {
                    partition: PartitionId::Node,
                    value,
                    ..
                } => Some(value.len()),
                _ => None,
            })
            .chain(self.effects.nodes.iter().map(|basis| basis.labels.len()))
            .max()
            .unwrap_or(0);
        let implied = self
            .effects
            .implications
            .iter()
            .map(|i| i.implies.len())
            .max()
            .unwrap_or(0);
        self.routing = (3 * stored + 2 * implied) as u64 * core::mem::size_of::<LabelId>() as u64;
        Ok(())
    }

    /// Run `build`, the label lists one node event decodes and builds while
    /// it is built, and `extra` bytes more it copies, given room first.
    /// While building, a room that refuses them ends what the consumer is
    /// handed, this event included.
    fn with_routing(
        &mut self,
        slot: Slot,
        extra: u64,
        build: impl FnOnce(&mut Self) -> Result<(), ProjectStop>,
    ) -> Result<(), ProjectStop> {
        let bytes = self.routing + extra;
        match self.lend(bytes) {
            Ok(()) => {}
            Err(ProjectStop::NoRoom) if slot == Slot::Build => {
                self.taking = false;
                return Ok(());
            }
            Err(stop) => return Err(stop),
        }
        let built = build(self);
        self.repay(bytes);
        built
    }

    /// Run `build`, an event's own lists of `bytes` given room first; a
    /// room that refuses them ends what the consumer is handed.
    fn with_lists(&mut self, bytes: u64, build: impl FnOnce(&mut Self)) {
        if !self.room.take(bytes) {
            self.taking = false;
            return;
        }
        build(self);
        self.room.give(bytes);
    }

    /// Whether a run of node deletes placed after operation `after` (any,
    /// for `None`) holds `key`.
    fn in_run_after(&self, key: &[u8], after: Option<u32>) -> bool {
        let to = self.runs.partition_point(|run| run.start <= key);
        for run in self.runs[..to].iter().rev() {
            if run.reach <= key {
                break;
            }
            if run.end > key && run.start.len() == key.len() && after.is_none_or(|at| run.at > at) {
                return true;
            }
        }
        false
    }

    /// Whether an operation after the current one changes node `key`.
    fn changes_later(&self, key: &[u8]) -> bool {
        let at = self.at;
        let next = self.nodes.partition_point(|&(k, i)| (k, i) <= (key, at));
        self.nodes.get(next).is_some_and(|&(k, _)| k == key) || self.in_run_after(key, Some(at))
    }

    /// Whether a put or delete of the unit writes node version `key`.
    fn writes_version(&self, key: &[u8]) -> bool {
        let from = self.nodes.partition_point(|&(k, _)| k < key);
        self.nodes[from..]
            .iter()
            .take_while(|&&(k, _)| k == key)
            .any(|&(_, at)| {
                matches!(
                    self.mutations.get(at as usize),
                    Some(Mutation::Put { .. } | Mutation::Delete { .. })
                )
            })
    }

    /// Whether a forward adjacency change of the unit names `pair`: a
    /// removal the context also names is already an event.
    fn named(&self, pair: Pair<'_>) -> bool {
        let from = self.pairs.partition_point(|c| c.pair < pair);
        self.pairs
            .get(from)
            .is_some_and(|c| c.pair == pair && c.seq != u32::MAX)
    }

    /// Whether the unit removes `pair`: its context names the removal, or
    /// its last forward adjacency change is one.
    fn removed(&self, pair: Pair<'_>) -> bool {
        let to = self.pairs.partition_point(|c| c.pair <= pair);
        to > 0 && {
            let last = self.pairs[to - 1];
            last.pair == pair && (last.seq == u32::MAX || !last.present)
        }
    }

    /// Keep what node `key` carries after its change, for its later change
    /// in the unit: the unit's own key as `unit_key`, or `None` for a key of
    /// a run, copied when it is kept; `labels` borrowed are copied, owned
    /// are moved in. Every copy and the table's growth have their room taken
    /// before they are allocated; a list moved in is charged as it changes
    /// owner.
    fn remember(
        &mut self,
        key: &[u8],
        unit_key: Option<&'u [u8]>,
        labels: Cow<'_, [LabelId]>,
    ) -> Result<(), ProjectStop> {
        let slot = core::mem::size_of::<LabelId>() as u64;
        let value = match &labels {
            Cow::Borrowed(labels) => labels.len() as u64 * slot,
            Cow::Owned(labels) => labels.capacity() as u64 * slot,
        };
        if let Some(old) = self.left.get(key) {
            let freed = old.capacity() as u64 * slot;
            self.lend(value)?;
            let labels = labels.into_owned();
            if let Some(old) = self.left.get_mut(key) {
                *old = labels;
            }
            self.repay(freed);
            return Ok(());
        }
        if self.left.len() == self.left.capacity() {
            // Past what the unit was counted to keep: the table doubles.
            let grown = (self.left.capacity().max(2)) * 2;
            let bytes = table_bytes::<(Cow<'u, [u8]>, Vec<LabelId>)>(grown);
            self.lend(bytes)?;
            self.left.reserve(grown - self.left.len());
            let old = core::mem::replace(&mut self.table, bytes);
            self.repay(old);
        }
        let copied = if unit_key.is_some() {
            0
        } else {
            key.len() as u64
        };
        self.lend(value + copied)?;
        let key = match unit_key {
            Some(unit_key) => Cow::Borrowed(unit_key),
            None => Cow::Owned(key.to_vec()),
        };
        self.left.insert(key, labels.into_owned());
        Ok(())
    }

    /// The labels node `key` carries before its next change in the unit:
    /// what its last change in the unit left, else what its context says it
    /// had before the unit; `None` when neither is known.
    fn labels_before(&self, key: &[u8]) -> Result<Option<Cow<'u, [LabelId]>>, ProjectionError> {
        if let Some(left) = self.left.get(key) {
            return Ok(Some(Cow::Owned(left.clone())));
        }
        self.basis(key)
            .map(|basis| self.carried(Cow::Borrowed(basis.labels.as_slice())))
            .transpose()
    }

    /// The labels a node record stores, its type's implied ones included.
    fn after_labels(&self, record: &[u8]) -> Result<Vec<LabelId>, ProjectionError> {
        let stored = coordinode_core::graph::node::NodeRecord::labels_from_msgpack(record)
            .map_err(|e| ProjectionError::Malformed(format!("a node record: {e}")))?;
        Ok(self.carried(Cow::Owned(stored))?.into_owned())
    }

    fn walk(&mut self) -> Result<(), ProjectStop> {
        let (mutations, run_rows) = (self.mutations, self.effects.runs.as_slice());
        for_each_operation(mutations, run_rows, |at, mutation, is_run| {
            self.at = at;
            match mutation {
                Mutation::Put {
                    partition,
                    key,
                    value,
                } => self.put(*partition, key, value),
                Mutation::Delete { partition, key } => self.delete(*partition, key, Some(key)),
                Mutation::Merge {
                    partition,
                    key,
                    operand,
                } => self.merge(*partition, key, operand),
                Mutation::RemoveRange {
                    partition,
                    start,
                    end,
                } => self.remove_range(*partition, start, end, is_run),
                Mutation::Effects(effects) => self.removed_with_lists(effects),
                Mutation::Command(_) | Mutation::Derive(_) | Mutation::Operation(_) => Ok(()),
            }
        })
    }

    /// Count the next event, and say what its ordinal asks.
    fn slot(&mut self) -> Result<Slot, ProjectionError> {
        let ordinal = self.count;
        self.count = ordinal.checked_add(1).ok_or_else(|| {
            ProjectionError::Malformed("the unit projects over 2^32 events".into())
        })?;
        Ok(if !self.taking {
            Slot::Counted
        } else if ordinal < self.first {
            Slot::Delivered
        } else {
            Slot::Build
        })
    }

    /// Whether the next event would be built: what the operations before it
    /// are checked for, those of a part the consumer reads.
    fn building(&self) -> bool {
        self.taking && self.count >= self.first
    }

    /// Hand the event of the last slot to the consumer.
    fn hand(
        &mut self,
        action: Action,
        target: Target<'u>,
        payload: Option<&'u [u8]>,
        routing: Option<Routing<'u>>,
    ) {
        let event = LogicalEvent {
            ordinal: self.count - 1,
            action,
            target,
            payload,
            routing,
        };
        self.taking = (self.take)(event, &mut *self.room) == Take::More;
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
    /// unit committed. A type the unit records no implication for is refused:
    /// its absence does not say the type implies nothing.
    fn carried(&self, stored: Cow<'u, [LabelId]>) -> Result<Cow<'u, [LabelId]>, ProjectionError> {
        let Some(&primary) = stored.first() else {
            return Ok(stored);
        };
        let implications = &self.effects.implications;
        let implied = implications
            .binary_search_by(|i| i.label.cmp(&primary))
            .map(|at| &implications[at].implies)
            .map_err(|_| {
                ProjectionError::Malformed(format!(
                    "a node of type {primary} changes, and the unit records no implication for it"
                ))
            })?;
        if implied.is_empty() {
            return Ok(stored);
        }
        // One list at the size it can reach, not grown push by push.
        let mut all = Vec::with_capacity(stored.len() + implied.len());
        all.extend_from_slice(&stored);
        for &label in implied {
            if !stored.contains(&label) {
                all.push(label);
            }
        }
        Ok(Cow::Owned(all))
    }

    /// The basis of a node whose change cannot state its own labels.
    fn required_labels(
        &self,
        key: &[u8],
        what: &str,
    ) -> Result<Cow<'u, [LabelId]>, ProjectionError> {
        self.labels_before(key)?.ok_or_else(|| {
            ProjectionError::Malformed(format!(
                "a node {what} carries no labels it stored before the unit"
            ))
        })
    }

    /// Keep what node `key` carries after its change, when a later
    /// operation changes it again. While building, a room that refuses it
    /// ends what the consumer is handed, this event included.
    /// As [`Self::remember`] keeps them, nothing copied when it is not kept.
    fn keep_for_later(
        &mut self,
        slot: Slot,
        key: &[u8],
        unit_key: Option<&'u [u8]>,
        labels: Cow<'_, [LabelId]>,
    ) -> Result<bool, ProjectStop> {
        if !self.changes_later(key) {
            return Ok(true);
        }
        match self.remember(key, unit_key, labels) {
            Ok(()) => Ok(true),
            Err(ProjectStop::NoRoom) if slot == Slot::Build => {
                self.taking = false;
                Ok(false)
            }
            Err(stop) => Err(stop),
        }
    }

    fn put(
        &mut self,
        partition: PartitionId,
        key: &'u [u8],
        value: &'u [u8],
    ) -> Result<(), ProjectStop> {
        match partition {
            PartitionId::Node => {
                let slot = self.slot()?;
                if slot == Slot::Counted || (slot == Slot::Delivered && !self.changes_later(key)) {
                    return Ok(());
                }
                self.with_routing(slot, 0, |this| {
                    let after = this.after_labels(value)?;
                    if slot == Slot::Delivered {
                        // Its labels move in: nothing else holds them.
                        this.keep_for_later(slot, key, Some(key), Cow::Owned(after))?;
                        return Ok(());
                    }
                    // Its writer bound the labels the record replaces.
                    let before = this.required_labels(key, "write")?;
                    // The event routes by them too: the kept copy is made
                    // within the room taken for it.
                    if this.keep_for_later(slot, key, Some(key), Cow::Borrowed(&after))? {
                        let routing = Routing {
                            before,
                            after: Cow::Owned(after),
                        };
                        this.hand(
                            Action::Put,
                            Target::Node(Cow::Borrowed(key)),
                            Some(value),
                            Some(routing),
                        );
                    }
                    Ok(())
                })
            }
            PartitionId::Adj => {
                // A whole list written at once, as a restore writes it.
                let Some((edge_type, source)) = forward_adj_key(key) else {
                    return Ok(());
                };
                self.posting_list(edge_type, source, value)
            }
            PartitionId::EdgeProp => {
                if self.slot()? == Slot::Build {
                    let target = edge_target(Cow::Borrowed(key))?;
                    self.hand(Action::Put, target, Some(value), None);
                }
                Ok(())
            }
            _ if is_blob_owner(partition, key) => {
                if self.slot()? == Slot::Build {
                    self.hand(
                        Action::Put,
                        Target::BlobOwner(Cow::Borrowed(key)),
                        Some(value),
                        None,
                    );
                }
                Ok(())
            }
            _ => {
                if self.building() {
                    // Refuses a record of no known kind.
                    self.component(partition, key)?;
                }
                Ok(())
            }
        }
    }

    /// The relationships a whole posting list puts. Its blocks, and its
    /// members when one of them is built, are decoded within room taken
    /// first and given back once they are dropped.
    fn posting_list(
        &mut self,
        edge_type: &'u [u8],
        source: u64,
        value: &'u [u8],
    ) -> Result<(), ProjectStop> {
        use coordinode_core::graph::codec::{UidBlock, UidPack, decode_uids};
        let bad = |e: rmp_serde::decode::Error| {
            ProjectStop::Projection(ProjectionError::Malformed(format!("a posting list: {e}")))
        };
        // Every block takes more than a byte of its encoding.
        let pack_bytes = value.len() as u64 * (1 + core::mem::size_of::<UidBlock>() as u64);
        self.lend(pack_bytes)?;
        let pack: UidPack = rmp_serde::from_slice(value).map_err(bad)?;
        let total = pack.total_uids();
        // A sum of block counts each under 2^32, over fewer blocks than the
        // frame has bytes: far below 2^64, past an event count of 2^32.
        let builds = self.taking && u64::from(self.count) + total > u64::from(self.first);
        if builds {
            let uid_bytes = total * core::mem::size_of::<u64>() as u64;
            self.lend(uid_bytes)?;
            let uids = decode_uids(&pack);
            for target in uids.iter().copied() {
                if self.slot()? == Slot::Build {
                    self.hand(
                        Action::Put,
                        Target::Relationship {
                            edge_type,
                            source,
                            target,
                        },
                        None,
                        None,
                    );
                }
            }
            drop(uids);
            self.repay(uid_bytes);
        } else {
            for _ in 0..total {
                self.slot()?;
            }
        }
        drop(pack);
        self.repay(pack_bytes);
        Ok(())
    }

    /// What a change of `key` in `partition`, outside nodes and
    /// relationships, is: a blob owner's record, which readers receive, or
    /// a record kept for the engine's own work, which they do not. A record
    /// of neither kind is refused, never passed over.
    fn component(&self, partition: PartitionId, key: &[u8]) -> Result<Component, ProjectionError> {
        use coordinode_core::graph::blob;
        match partition {
            PartitionId::BlobRef => {
                // The catalog of copies, staged uploads, release candidates,
                // retired copies and the collector's own records: how the
                // blobs owners hold are kept, not what they hold.
                let internal = [
                    blob::CATALOG_PREFIX,
                    blob::STAGING_PREFIX,
                    blob::CANDIDATE_PREFIX,
                    blob::RETIRED_PREFIX,
                    blob::COMPACTION_PENDING_KEY,
                    blob::COLLECTOR_STATE_KEY,
                ];
                if is_blob_owner(partition, key) {
                    Ok(Component::BlobOwner)
                } else if internal.iter().any(|p| key.starts_with(p)) {
                    Ok(Component::Internal)
                } else {
                    Err(ProjectionError::Unsupported(
                        "it changes a blob record of no known kind".into(),
                    ))
                }
            }
            // Bytes of blob contents, shared by every owner of the content
            // and resolved through the catalog: their owners' records are the
            // changes.
            PartitionId::Blob => Ok(Component::Internal),
            // A temporal version's history assertion is written with the
            // version itself, whose change is the event.
            PartitionId::NodeHistory => {
                let id = coordinode_core::graph::node::decode_node_history_key(key)
                    .ok_or_else(|| ProjectionError::Malformed("a history assertion key".into()))?;
                let version = coordinode_core::graph::node::encode_temporal_node_key(
                    id.shard_id,
                    id.node_id,
                    id.valid_from,
                );
                if self.writes_version(&version) {
                    Ok(Component::Internal)
                } else {
                    Err(ProjectionError::Unsupported(
                        "it asserts history of a version it does not change".into(),
                    ))
                }
            }
            // The f32 copy a vector index keeps beside the node's record is
            // written outside transactions; one written in a unit is not a
            // change the feed knows how to read.
            PartitionId::VectorF32 => Err(ProjectionError::Unsupported(
                "it writes a vector index's copy in a transaction".into(),
            )),
            // Index entries, counters, the catalog and consensus records.
            _ => Ok(Component::Internal),
        }
    }

    /// A delete of `key`: the unit's own bytes as `unit_key`, or `None` for
    /// a key of a run, which is copied only when an event or a later change
    /// of it keeps it.
    fn delete(
        &mut self,
        partition: PartitionId,
        key: &[u8],
        unit_key: Option<&'u [u8]>,
    ) -> Result<(), ProjectStop> {
        let owned = |key: &[u8]| -> Cow<'u, [u8]> {
            match unit_key {
                Some(unit_key) => Cow::Borrowed(unit_key),
                None => Cow::Owned(key.to_vec()),
            }
        };
        match partition {
            PartitionId::Node => {
                let slot = self.slot()?;
                if slot == Slot::Counted {
                    return Ok(());
                }
                if slot == Slot::Delivered {
                    self.keep_for_later(slot, key, unit_key, Cow::Borrowed(&[]))?;
                    return Ok(());
                }
                // A run's key is copied into the event it is handed in.
                let copied = if unit_key.is_some() {
                    0
                } else {
                    key.len() as u64
                };
                self.with_routing(slot, copied, |this| {
                    let before = this.required_labels(key, "removal")?;
                    if this.keep_for_later(slot, key, unit_key, Cow::Borrowed(&[]))? {
                        let routing = Routing {
                            before,
                            after: Cow::Borrowed(&[]),
                        };
                        this.hand(
                            Action::Remove,
                            Target::Node(owned(key)),
                            None,
                            Some(routing),
                        );
                    }
                    Ok(())
                })
            }
            PartitionId::EdgeProp => {
                let part_of_removal = edgeprop_key(key)
                    .is_some_and(|(pair, suffix)| suffix.is_none() && self.removed(pair));
                // The pair's property record goes with the relationship.
                if !part_of_removal && self.slot()? == Slot::Build {
                    if edgeprop_key(key).is_none() {
                        return Err(ProjectionError::Malformed(
                            "an edge property key of no known shape".into(),
                        )
                        .into());
                    }
                    // A run's key is copied into the event it is handed in.
                    let copied = if unit_key.is_some() {
                        0
                    } else {
                        key.len() as u64
                    };
                    self.with_lists(copied, |this| {
                        if let Ok(target) = edge_target(owned(key)) {
                            this.hand(Action::Remove, target, None, None);
                        }
                    });
                }
                Ok(())
            }
            // A whole adjacency list goes with its node: the relationships it
            // held are named by forward removals or the context.
            PartitionId::Adj => Ok(()),
            _ if is_blob_owner(partition, key) => {
                if self.slot()? == Slot::Build {
                    self.hand(Action::Remove, Target::BlobOwner(owned(key)), None, None);
                }
                Ok(())
            }
            _ => {
                if self.building() {
                    self.component(partition, key)?;
                }
                Ok(())
            }
        }
    }

    fn merge(
        &mut self,
        partition: PartitionId,
        key: &'u [u8],
        operand: &'u [u8],
    ) -> Result<(), ProjectStop> {
        match partition {
            PartitionId::Node => {
                if self.slot()? != Slot::Build {
                    return Ok(());
                }
                self.with_routing(Slot::Build, 0, |this| {
                    let labels = this.required_labels(key, "document delta")?;
                    let routing = Routing {
                        before: labels.clone(),
                        after: labels,
                    };
                    this.hand(
                        Action::Patch,
                        Target::Node(Cow::Borrowed(key)),
                        Some(operand),
                        Some(routing),
                    );
                    Ok(())
                })
            }
            PartitionId::Adj => {
                let Some((edge_type, source)) = forward_adj_key(key) else {
                    return Ok(());
                };
                for (present, target) in adj_operand(operand)? {
                    if self.slot()? == Slot::Build {
                        let action = if present { Action::Put } else { Action::Remove };
                        self.hand(
                            action,
                            Target::Relationship {
                                edge_type,
                                source,
                                target,
                            },
                            None,
                            None,
                        );
                    }
                }
                Ok(())
            }
            _ => {
                if self.building() {
                    match self.component(partition, key)? {
                        // A blob owner's record is written whole, never merged.
                        Component::BlobOwner => {
                            return Err(ProjectionError::Unsupported(
                                "it merges into a blob owner's record".into(),
                            )
                            .into());
                        }
                        Component::Internal => {}
                    }
                }
                Ok(())
            }
        }
    }

    fn remove_range(
        &mut self,
        partition: PartitionId,
        start: &'u [u8],
        end: &'u [u8],
        is_run: bool,
    ) -> Result<(), ProjectStop> {
        let projected = matches!(
            partition,
            PartitionId::Node | PartitionId::EdgeProp | PartitionId::Adj
        );
        if !projected {
            // A range names no blob owner or history assertion it removes,
            // and no writer removes those by range.
            return match partition {
                PartitionId::BlobRef | PartitionId::NodeHistory | PartitionId::VectorF32 => {
                    Err(ProjectionError::Unsupported(format!(
                        "it removes a {partition:?} key range that names no logical targets"
                    ))
                    .into())
                }
                _ => Ok(()),
            };
        }
        if !is_run {
            return Err(ProjectionError::Unsupported(
                "it removes a key range that names no logical targets".into(),
            )
            .into());
        }
        if partition == PartitionId::Adj {
            // Adjacency lists, as their nodes go: named elsewhere.
            return Ok(());
        }
        // The run's keys are every key of the start's length up to the end,
        // each projected as its own delete, walked in one buffer whose room
        // is taken first: a key is copied only when it is kept.
        let buffer = start.len() as u64;
        self.lend(buffer)?;
        let walked = for_each_run_key(start, end, |key| self.delete(partition, key, None));
        self.repay(buffer);
        walked
    }

    /// The relationships removed with whole forward lists, which no forward
    /// change of the unit names.
    fn removed_with_lists(&mut self, effects: &'u UnitEffects) -> Result<(), ProjectStop> {
        for key in &effects.removed_edges {
            let Some(((edge_type, source, target), _)) = edgeprop_key(key) else {
                continue;
            };
            if self.named((edge_type, source, target)) {
                continue;
            }
            if self.slot()? == Slot::Build {
                self.hand(
                    Action::Remove,
                    Target::Relationship {
                        edge_type,
                        source,
                        target,
                    },
                    None,
                    None,
                );
            }
        }
        // A change of what a type implies: one control for the whole type,
        // routed to every label it reaches before or after, the type's own
        // included, without a node of it enumerated.
        for change in &effects.scope_changes {
            if self.slot()? != Slot::Build {
                continue;
            }
            let with_type = |implied: &[LabelId]| {
                let mut all = Vec::with_capacity(implied.len() + 1);
                all.push(change.label);
                all.extend(implied.iter().copied().filter(|&l| l != change.label));
                Cow::Owned(all)
            };
            let lists = (2 + change.before.len() + change.after.len()) as u64
                * core::mem::size_of::<LabelId>() as u64;
            self.with_lists(lists, |this| {
                this.hand(
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
                );
            });
        }
        // A dropped label: its scope ends, routed by the identity it had.
        for &label in &effects.dropped {
            if self.slot()? == Slot::Build {
                self.with_lists(core::mem::size_of::<LabelId>() as u64, |this| {
                    this.hand(
                        Action::Drop,
                        Target::Label(label),
                        None,
                        Some(Routing {
                            before: Cow::Owned(vec![label]),
                            after: Cow::Borrowed(&[]),
                        }),
                    );
                });
            }
        }
        Ok(())
    }
}

/// Whether `key` in `partition` is a blob owner's record: a node's large
/// property, a stored object or an S3 object.
fn is_blob_owner(partition: PartitionId, key: &[u8]) -> bool {
    use coordinode_core::graph::blob;
    partition == PartitionId::BlobRef
        && [
            blob::PROPERTY_OWNER_PREFIX,
            blob::DIRECT_OWNER_PREFIX,
            blob::S3_OWNER_PREFIX,
        ]
        .iter()
        .any(|p| key.starts_with(p))
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
/// order, read in place.
enum AdjMembers<'o> {
    One(Option<(bool, u64)>),
    Added(core::slice::Iter<'o, [u8; 8]>),
}

impl Iterator for AdjMembers<'_> {
    type Item = (bool, u64);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(member) => member.take(),
            Self::Added(uids) => uids.next().map(|uid| (true, u64::from_be_bytes(*uid))),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = match self {
            Self::One(member) => usize::from(member.is_some()),
            Self::Added(uids) => uids.len(),
        };
        (left, Some(left))
    }
}

impl ExactSizeIterator for AdjMembers<'_> {}

fn adj_operand(operand: &[u8]) -> Result<AdjMembers<'_>, ProjectionError> {
    let bad = || ProjectionError::Malformed("an adjacency operand of no known shape".into());
    let uid = |bytes: &[u8]| -> Result<u64, ProjectionError> {
        Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| bad())?))
    };
    match operand.split_first() {
        Some((&crate::engine::merge::TAG_ADD, rest)) => {
            Ok(AdjMembers::One(Some((true, uid(rest)?))))
        }
        Some((&crate::engine::merge::TAG_REMOVE, rest)) => {
            Ok(AdjMembers::One(Some((false, uid(rest)?))))
        }
        Some((&crate::engine::merge::TAG_ADD_BATCH, rest)) => {
            let (count, uids) = rest.split_at_checked(4).ok_or_else(bad)?;
            let count = u32::from_be_bytes(count.try_into().map_err(|_| bad())?) as usize;
            if uids.len() != count.checked_mul(8).ok_or_else(bad)? {
                return Err(bad());
            }
            let (chunks, _) = uids.as_chunks::<8>();
            Ok(AdjMembers::Added(chunks.iter()))
        }
        _ => Err(bad()),
    }
}

/// Hand `each` every key of `start`'s length from `start` up to `end`, both
/// of one length, in order: the point deletes a run stands for, one at a
/// time in one buffer, never listed.
fn for_each_run_key(
    start: &[u8],
    end: &[u8],
    mut each: impl FnMut(&[u8]) -> Result<(), ProjectStop>,
) -> Result<(), ProjectStop> {
    let bad = |what: &str| {
        ProjectStop::Projection(ProjectionError::Malformed(format!(
            "a run of deletes {what}"
        )))
    };
    if start.len() != end.len() || start >= end {
        return Err(bad("with bounds of different lengths or out of order"));
    }
    let mut key = start.to_vec();
    let mut walked = 0u64;
    while key.as_slice() < end {
        if walked >= MAX_RUN_KEYS {
            return Err(bad("longer than any unit"));
        }
        walked += 1;
        each(&key)?;
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
    Ok(())
}

/// What a reader of the feed is interested in: nodes routed by some labels,
/// relationships of some types, and whether blob objects. `None` takes every
/// one of that kind; an empty list takes none.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interest {
    labels: Option<Vec<LabelId>>,
    edge_types: Option<Vec<Vec<u8>>>,
    /// Whether the reader follows the blob objects owners hold: a node's
    /// large property, a stored object, an S3 object. They belong to no
    /// label, so a label interest does not choose them.
    blobs: bool,
}

impl Interest {
    /// Every event.
    #[must_use]
    pub fn all() -> Self {
        Self {
            labels: None,
            edge_types: None,
            blobs: true,
        }
    }

    /// The same interest, following blob objects or not.
    #[must_use]
    pub fn with_blobs(mut self, blobs: bool) -> Self {
        self.blobs = blobs;
        self
    }

    /// Whether it follows blob objects.
    #[must_use]
    pub fn blobs(&self) -> bool {
        self.blobs
    }

    /// The label identities whose node events it takes, in identity order;
    /// `None` for every label.
    #[must_use]
    pub fn labels(&self) -> Option<&[LabelId]> {
        self.labels.as_deref()
    }

    /// The relationship type names whose events it takes, in name order;
    /// `None` for every type.
    #[must_use]
    pub fn edge_types(&self) -> Option<&[Vec<u8>]> {
        self.edge_types.as_deref()
    }

    /// Node events routed by any of `labels` before or after them, and
    /// relationship events of any of `edge_types`; `None` for either takes
    /// every event of that kind. No blob object: [`Self::with_blobs`] adds
    /// them.
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
        Self {
            labels,
            edge_types,
            blobs: false,
        }
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
            // Routed above: every node, scope and label event carries its
            // labels.
            Target::Node(_) | Target::Scope { .. } | Target::Label(_) => {
                return Some(Delivery::Apply);
            }
            Target::BlobOwner(_) => return self.blobs.then_some(Delivery::Apply),
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
        out.push(u8::from(self.blobs));
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

    /// Whether this cursor reads the same stream as `earlier` (the same
    /// history and interest) at or past its place.
    #[must_use]
    pub fn continues(&self, earlier: &Self) -> bool {
        self.history == earlier.history
            && self.interest == earlier.interest
            && (self.next, self.within) >= (earlier.next, earlier.within)
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
        let blobs = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err(CursorError::Malformed("an unknown blob interest")),
        };
        if !r.0.is_empty() {
            return Err(CursorError::Malformed("bytes after the cursor"));
        }
        let interest = Interest::new(labels.clone(), edge_types.clone()).with_blobs(blobs);
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
    /// What a blob owner holds, by its owner key.
    BlobOwner(#[serde(with = "serde_bytes")] Vec<u8>),
    /// A label, by the identity it had.
    Label(LabelId),
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
            Target::BlobOwner(key) => Self::BlobOwner(key.to_vec()),
            Target::Label(label) => Self::Label(*label),
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
    /// Where this read started: the units continue a reader that stood
    /// there, and no other.
    start: ChangeCursor,
}

impl ChangeBatch {
    /// The batch a read from `start` handed over.
    pub(crate) fn read_from(
        start: ChangeCursor,
        units: Vec<UnitChanges>,
        cursor: ChangeCursor,
    ) -> Self {
        Self {
            units,
            cursor,
            start,
        }
    }

    /// The cursor this batch was read from: it continues a reader that
    /// stood there.
    #[must_use]
    pub fn start(&self) -> &ChangeCursor {
        &self.start
    }
}

/// `event` of the unit at `position` as a reader receives it, reaching it
/// as `delivery`.
pub(crate) fn owned_event(
    position: coordinode_core::txn::operation::LogPosition,
    event: &LogicalEvent<'_>,
    delivery: Delivery,
) -> ChangeEvent {
    ChangeEvent {
        id: EventId {
            position,
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
    }
}

/// Bytes the heap holds for one delivered event beyond its slot in its
/// unit's list of events: its key, payload and labels, each allocated at
/// its length.
pub(crate) fn event_heap_bytes(event: &LogicalEvent<'_>) -> u64 {
    let target = match &event.target {
        Target::Node(key)
        | Target::RelationshipProperties(key)
        | Target::RelationshipInstance(key)
        | Target::BlobOwner(key) => key.len(),
        Target::Relationship { edge_type, .. } => edge_type.len(),
        Target::Scope { .. } | Target::Label(_) => 0,
    };
    let labels = event.routing.as_ref().map_or(0, |r| {
        (r.before.len() + r.after.len()) * core::mem::size_of::<LabelId>()
    });
    (target + event.payload.map_or(0, <[u8]>::len) + labels) as u64
}

/// What a reader's own form of what it reads takes, charged to the read's
/// budget with the events, before they are kept: a server encoding each
/// batch for its wire holds the batch and its encoding at once, within one
/// limit.
#[diagnostic::on_unimplemented(
    message = "`{Self}` does not say what a reader's form of a change read takes",
    label = "not a change encoding",
    note = "pass `Unencoded` for a reader that keeps the batch as it is read"
)]
pub trait ChangeEncoding {
    /// Bytes the reader's form of `event` takes. What the reader keeps to
    /// build that form (names it looks up, say) is taken from `take` before
    /// it is allocated, and stays charged for the read; `None` when `take`
    /// refuses it, which ends the read there.
    fn event(&mut self, event: &LogicalEvent<'_>, take: &mut dyn FnMut(u64) -> bool)
    -> Option<u64>;

    /// Bytes the reader's form of a unit takes besides its events.
    fn unit(&mut self) -> u64;
}

/// A reader that keeps the batch as it is read: no form of its own.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unencoded;

impl ChangeEncoding for Unencoded {
    fn event(&mut self, _: &LogicalEvent<'_>, _: &mut dyn FnMut(u64) -> bool) -> Option<u64> {
        Some(0)
    }

    fn unit(&mut self) -> u64 {
        0
    }
}

/// Bytes a read takes from its budget at a time for the events it keeps:
/// the budget is asked once per block, not once per event.
const CHARGE_BLOCK: u64 = 64 * 1024;

/// A charge taken from its budget a block at a time: the events kept, the
/// lists holding them and the projection's working state spend the block
/// and give back to it without asking the budget again.
struct BlockCharge<'c, 'b> {
    charge: &'c mut coordinode_core::budget::MemoryCharge<'b>,
    /// Taken from the budget and not yet spent.
    left: u64,
    /// Why the budget last refused a block.
    refused: Option<coordinode_core::budget::BudgetStop>,
}

impl<'c, 'b> BlockCharge<'c, 'b> {
    fn new(charge: &'c mut coordinode_core::budget::MemoryCharge<'b>) -> Self {
        Self {
            charge,
            left: 0,
            refused: None,
        }
    }

    /// Return what was taken and not spent.
    fn release(&mut self) {
        self.charge.shrink(self.left);
        self.left = 0;
    }
}

impl Room for BlockCharge<'_, '_> {
    /// Spend `bytes`, taking another block when the one held is spent; a
    /// block the budget refuses is asked for again at exactly what is
    /// missing, so the budget's last bytes are not lost to the block size.
    fn take(&mut self, bytes: u64) -> bool {
        if let Some(left) = self.left.checked_sub(bytes) {
            self.left = left;
            return true;
        }
        let missing = bytes - self.left;
        let block = missing.max(CHARGE_BLOCK);
        if self.charge.grow(block).is_ok() {
            // `block` covers what was missing: the rest stays held.
            self.left = block - missing;
            return true;
        }
        match self.charge.grow(missing) {
            Ok(()) => {
                self.left = 0;
                true
            }
            Err(stop) => {
                self.refused = Some(stop);
                false
            }
        }
    }

    fn give(&mut self, bytes: u64) {
        // At most what the charge holds: it was taken from it.
        self.left += bytes;
    }
}

/// The changes a reader at `from` receives of `units`, the units of the
/// entries read from its entry on: units before `from` inside its entry are
/// passed over, and the unit it stopped in continues at its ordinal. The
/// room of every list grown, every event kept and the working state of each
/// unit's projection is taken from `charge` before it is allocated; when
/// the charge refuses it, the read stops there, and the place it stopped is
/// returned. A read that can keep nothing refuses with the budget's stop.
/// What `encoding` says the reader's form of each unit and event takes is
/// charged with it.
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
    encoding: &mut impl ChangeEncoding,
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
    let mut room = BlockCharge::new(charge);
    let mut out: Vec<UnitChanges> = Vec::new();
    // Where the read stops: nothing kept at all is the budget's refusal.
    let stop = |room: &mut BlockCharge<'_, '_>,
                out: Vec<UnitChanges>,
                place: ReadPlace|
     -> Result<(Vec<UnitChanges>, Option<ReadPlace>), StorageError> {
        room.release();
        if out.is_empty() {
            return Err(StorageError::Budget(room.refused.take().unwrap_or(
                coordinode_core::budget::BudgetStop::Memory {
                    requested: 0,
                    used: 0,
                    limit: 0,
                },
            )));
        }
        Ok((out, Some(place)))
    };
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
        let stopped = |ordinal: u32| ReadPlace {
            entry: unit.position.index,
            within: Within {
                sub: unit.position.sub,
                ordinal,
            },
        };
        // The unit's place in the batch and in the reader's form, taken
        // before it is read.
        let unit_form = encoding.unit();
        if !grow_within(&mut out, &mut room) || !room.take(unit_form) {
            return stop(&mut room, out, stopped(first));
        }
        // Events are handed over one at a time from `first` on; those after
        // the one the budget refuses are only counted.
        let mut taken: Vec<ChangeEvent> = Vec::new();
        let mut through = first;
        let mut refused = false;
        let projected = unit.project_with(first, &mut room, |event, room| {
            if let Some(delivery) = interest.delivery(&event) {
                let Some(form) = encoding.event(&event, &mut |bytes| room.take(bytes)) else {
                    refused = true;
                    return Take::Enough;
                };
                let kept = event_heap_bytes(&event) + form;
                if !grow_within(&mut taken, room) || !room.take(kept) {
                    refused = true;
                    return Take::Enough;
                }
                taken.push(owned_event(unit.position, &event, delivery));
            }
            through = event.ordinal + 1;
            Take::More
        });
        let projected = match projected {
            Ok(projected) => projected,
            // The unit's working state does not fit beside what the read
            // keeps: the read stops before the unit.
            Err(ProjectStop::NoRoom) => {
                drop(taken);
                return stop(&mut room, out, stopped(first));
            }
            Err(ProjectStop::Projection(e)) => return Err(projection(e)),
        };
        // A projection that stopped handing events over before the unit's
        // end: the read stops inside it.
        if refused || through < projected {
            if through > first {
                out.push(UnitChanges {
                    position: unit.position,
                    commit_ts: unit.commit_ts,
                    events: taken,
                    projected,
                    first,
                    through,
                });
            }
            return stop(&mut room, out, stopped(through));
        }
        out.push(UnitChanges {
            position: unit.position,
            commit_ts: unit.commit_ts,
            events: taken,
            projected,
            first,
            through: projected,
        });
    }
    room.release();
    Ok((out, None))
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
